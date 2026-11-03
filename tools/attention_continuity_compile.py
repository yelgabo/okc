"""Fixed secondary CUDA13 selected-instance compile; never uses a CUDA driver."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
from local_cuda128_compile import ADAPTERS
ADAPTERS = dict(ADAPTERS, **{'stdlib.h': '#pragma once\ntypedef unsigned long size_t; typedef long ptrdiff_t;\n'})


def run(source, out, length):
    raw = source.read_text()
    device = raw[:raw.index('template <int L>\nvoid launch(')]
    device = device.replace('#include <torch/extension.h>', '#include <cuda_runtime.h>')
    device = re.sub(r'#include <(?:ATen/[^>]+|algorithm|vector)>\n', '', device)
    device = device.replace('#include <cstdint>', '#include <stdint.h>')
    device = '#define INFINITY (__int_as_float((int)0x7f800000u))\n' + device
    # Constant-only layout witness; no change inside either selected kernel.
    device += '\nextern "C" { __device__ __constant__ unsigned continuity_layout[2] = {sizeof(Smem<1,true>), sizeof(Smem<8,true>)}; }\n'
    cache = Path('/root/.cache/okc-cuda130-88')
    toolkit = cache/'nvidia/cu13'
    lib = C.CDLL(str(toolkit/'lib/libnvrtc.so.13'))
    lib.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)]
    lib.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    lib.nvrtcAddNameExpression.argtypes = [C.c_void_p, C.c_char_p]
    lib.nvrtcGetLoweredName.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_char_p)]
    for n in ('nvrtcGetProgramLogSize', 'nvrtcGetPTXSize'):
        getattr(lib, n).argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    for n in ('nvrtcGetProgramLog', 'nvrtcGetPTX'):
        getattr(lib, n).argtypes = [C.c_void_p, C.c_void_p]
    def check(x):
        if x: raise RuntimeError('NVRTC status '+str(x))
    out.mkdir(parents=True, exist_ok=True)
    (out/'device.cu').write_text(device)
    headers = (C.c_char_p*len(ADAPTERS))(*(s.encode() for s in ADAPTERS.values()))
    names = (C.c_char_p*len(ADAPTERS))(*(s.encode() for s in ADAPTERS))
    p = C.c_void_p()
    check(lib.nvrtcCreateProgram(C.byref(p), device.encode(), b'attention.cu', len(ADAPTERS), headers, names))
    expr = f'&attn_kernel<{length},true,0>'.encode()
    check(lib.nvrtcAddNameExpression(p, expr))
    check(lib.nvrtcAddNameExpression(p, b'&continuity_layout'))
    flags = ['-DOKC_QDIRECT=0', '-DOKC_PK2=0', '-DOKC_W0ENABLE=1', '-O3', '-DOKC_SM90=1', '--expt-relaxed-constexpr']
    # NVRTC optimizes at its default level; retain original ordered target flags
    # separately and omit only driver-only -O3 from NVRTC, as in prior screens.
    options = ['--gpu-architecture=compute_90a', '--std=c++17', '--device-as-default-execution-space',
        '--include-path='+str(toolkit/'include'),
        '--include-path='+str(cache/'cuda_cccl-linux-x86_64-13.0.85-archive/include'), *[x for x in flags if x not in ('-O3', '--expt-relaxed-constexpr')]]
    rc = lib.nvrtcCompileProgram(p, len(options), (C.c_char_p*len(options))(*(x.encode() for x in options)))
    size = C.c_size_t(); check(lib.nvrtcGetProgramLogSize(p, C.byref(size)))
    log = C.create_string_buffer(size.value); check(lib.nvrtcGetProgramLog(p, log))
    (out/'nvrtc.log').write_bytes(log.value)
    manifest = dict(source_sha256=hashlib.sha256(raw.encode()).hexdigest(), device_sha256=hashlib.sha256(device.encode()).hexdigest(),
        target_flags=flags, options=options, omitted_nvcc_only_flags=['-O3', '--expt-relaxed-constexpr'], virtual_headers=ADAPTERS, nvrtc_status=rc, expression=expr.decode(),
        declaration_adapters=['Torch/ATen host includes removed; cuda_runtime added', 'INFINITY exact positive-FP32-infinity declaration', 'constant-only continuity_layout sizeof witness outside selected kernels'],
        scope='secondary NVRTC13/PTXAS13 device-only; host NVCC, admission, numerical accuracy NOT_RUN')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if rc: print(log.value.decode()); check(rc)
    check(lib.nvrtcGetPTXSize(p, C.byref(size)))
    ptx = C.create_string_buffer(size.value); check(lib.nvrtcGetPTX(p, ptx)); (out/'device.ptx').write_bytes(ptx.value)
    cmd = [str(toolkit/'bin/ptxas'), '-arch=sm_90a', '-O3', '-v', '--warn-on-spills', str(out/'device.ptx'), '-o', str(out/'device.cubin')]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out/'ptxas.log').write_text(result.stdout+result.stderr)
    manifest.update(ptxas_command=cmd, ptxas_status=result.returncode,
        ptx_sha256=hashlib.sha256(ptx.value).hexdigest(),
        ptxas_output=result.stdout+result.stderr)
    if result.returncode == 0:
        manifest['cubin_sha256'] = hashlib.sha256((out/'device.cubin').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(result.stderr); result.check_returncode()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source', type=Path); p.add_argument('out', type=Path); p.add_argument('--length', type=int, choices=(1,8), required=True)
    a = p.parse_args(); run(a.source, a.out, a.length)
