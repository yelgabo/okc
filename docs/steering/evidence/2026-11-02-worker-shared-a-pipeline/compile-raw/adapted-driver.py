"""WSL CPU-only NVRTC12.8/PTXAS sm90a gate. No CUDA driver or GPU execution.

The exact device prefix before torch/extension.h is kept, with virtual standard
headers documented below. Host torch extension/launch wrappers are not compiled.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess


ADAPTERS = {
    'type_traits': '''#pragma once
namespace std { template<class T,T v> struct integral_constant {
static constexpr T value=v; constexpr operator T() const noexcept {return value;}
}; using true_type=integral_constant<bool,true>; using false_type=integral_constant<bool,false>; }
''',
    'stdint.h': '''#pragma once
typedef signed char int8_t; typedef unsigned char uint8_t;
typedef signed short int16_t; typedef unsigned short uint16_t;
typedef signed int int32_t; typedef unsigned int uint32_t;
typedef signed long int64_t; typedef unsigned long uint64_t;
typedef signed long intptr_t; typedef unsigned long uintptr_t;
''',
}


def compile_source(args):
    cache = args.cache
    lib = C.CDLL(str(cache / 'nvidia/cu13/lib/libnvrtc.so.13'))
    lib.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                     C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)]
    lib.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    lib.nvrtcAddNameExpression.argtypes = [C.c_void_p, C.c_char_p]
    lib.nvrtcGetLoweredName.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_char_p)]
    for name in ('nvrtcGetProgramLogSize', 'nvrtcGetPTXSize'):
        getattr(lib, name).argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    for name in ('nvrtcGetProgramLog', 'nvrtcGetPTX'):
        getattr(lib, name).argtypes = [C.c_void_p, C.c_void_p]
    def check(rc):
        if rc: raise RuntimeError('NVRTC status ' + str(rc))
    raw = args.source.read_bytes()
    source = raw.decode().replace('\r\n', '\n')
    marker = '#include <torch/extension.h>'
    assert source.count(marker) == 1, 'Expected one explicit host-extension boundary'
    device = source[:source.index(marker)]
    original_prefix_hash = hashlib.sha256(device.encode()).hexdigest()
    assert 'std::' not in device.replace('std::true_type', '').replace('std::false_type', '')
    device=device.replace('}  // namespace okc','\n__device__ __constant__ unsigned ssa_layout[]={sizeof(Smem),((sizeof(Smem)+127)/128)*128,SSA_XOFF+WGS_NX*WGS_XS,JMAX,WGS_FIXF,NTL,SSA_AOFF,SSA_XOFF};\n'+'}  // namespace okc')
    declaration_adapter = None
    if args.forward_declare_wgs:
        anchor = 'template <bool PB>\n__device__ __forceinline__ void wgs_phase('
        assert device.count(anchor) == 1
        declaration_adapter = '__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'
        assert device.count(declaration_adapter[:-2] + ' {') == 1
        device = device.replace(anchor, declaration_adapter + anchor)
    args.out.mkdir(parents=True, exist_ok=True)
    program = C.c_void_p()
    headers = (C.c_char_p * len(ADAPTERS))(*(s.encode() for s in ADAPTERS.values()))
    names = (C.c_char_p * len(ADAPTERS))(*(s.encode() for s in ADAPTERS))
    check(lib.nvrtcCreateProgram(C.byref(program), device.encode(), b'exact_device.cu', len(ADAPTERS), headers, names))
    expressions = [b'&okc::moe_fused<false>', b'&okc::moe_fused<true>']
    for name in expressions+[b"&okc::ssa_layout"]: check(lib.nvrtcAddNameExpression(program, name))
    options = ['--gpu-architecture=compute_90a', '--std=c++20', '--generate-line-info',
               '--device-as-default-execution-space',
               '--include-path=' + str(cache / 'nvidia/cu13/include'),
               '--include-path=' + str(cache / 'nvidia/cu13/include'),
               '--include-path=' + str(cache / 'cuda_cccl-linux-x86_64-13.0.85-archive/include'),
               *args.define]
    argv = (C.c_char_p * len(options))(*(o.encode() for o in options))
    status = lib.nvrtcCompileProgram(program, len(options), argv)
    size = C.c_size_t(); check(lib.nvrtcGetProgramLogSize(program, C.byref(size)))
    log = C.create_string_buffer(size.value); check(lib.nvrtcGetProgramLog(program, log))
    (args.out / 'nvrtc.log').write_bytes(log.value)
    print(log.value.decode(), flush=True)
    manifest = dict(source=str(args.source), source_sha256=hashlib.sha256(raw).hexdigest(),
                    original_device_prefix_sha256=original_prefix_hash,
                    compiled_device_prefix_sha256=hashlib.sha256(device.encode()).hexdigest(),
                    host_boundary=marker, virtual_headers=ADAPTERS, options=options,
                    declaration_adapter=declaration_adapter, nvrtc_status=status, gpu_execution=False)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    check(status)
    lowered = []
    for expression in expressions:
        name = C.c_char_p(); check(lib.nvrtcGetLoweredName(program, expression, C.byref(name)))
        lowered.append(name.value.decode())
    check(lib.nvrtcGetPTXSize(program, C.byref(size)))
    ptx = C.create_string_buffer(size.value); check(lib.nvrtcGetPTX(program, ptx))
    path = args.out / 'device.ptx'; path.write_bytes(ptx.value)
    ptxas = cache / 'nvidia/cu13/bin/ptxas'
    command = [str(ptxas), '-arch=sm_90a', '-O3', '-v', '--warn-on-spills', str(path), '-o', str(args.out / 'device.cubin')]
    result = subprocess.run(command, text=True, capture_output=True)
    (args.out / 'ptxas.log').write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr, flush=True)
    entries = {}
    for match in re.finditer(r"Compiling entry function '([^']+)'.*?Function properties.*?\n"
                             r'\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\n'
                             r'ptxas info\s+: Used (\d+) registers, used (\d+) barriers', result.stderr, re.S):
        entries[match[1]] = dict(zip(('stack_bytes', 'spill_store_bytes', 'spill_load_bytes', 'registers', 'barriers'),
                                    map(int, match.groups()[1:])))
    manifest.update(lowered=lowered, ptx_sha256=hashlib.sha256(ptx.value).hexdigest(),
                    cubin_sha256=hashlib.sha256((args.out / 'device.cubin').read_bytes()).hexdigest() if result.returncode == 0 else None,
                    ptxas=command, exitcode=result.returncode, resources=entries,
                    ptx_launch_bounds=re.findall(r'\.(?:maxntid|minnctapersm|maxnreg)[^\n]*', ptx.value.decode()),
                    ptxas_wgmma_serialization_warnings=result.stderr.count('(C7513)'))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result.check_returncode()
    assert set(entries) == set(lowered), 'Missing actual lowered entry resources'
    print('LOCAL_COMPILE_RESOURCES', json.dumps(entries), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--cache', type=Path, default=Path('/root/.cache/okc-cuda128'))
    p.add_argument('--sc6', type=int, choices=(0, 1), default=1)
    p.add_argument('--define', action='append', default=[])
    p.add_argument('--forward-declare-wgs', action='store_true', help='Disclosed declaration-only NVRTC two-phase lookup adapter')
    compile_source(p.parse_args())
