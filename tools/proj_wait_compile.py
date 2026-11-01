"""Compile readable device-only CU to sm90a CUBIN, without GPU execution.

The pinned exporter uses the source's existing SK_DEVICE_ONLY host guards.
No device algorithm is removed here. Virtual declaration headers are disclosed.
"""
import argparse
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess


def extract(source):
    # Input is a readable exact device-only generated CU; source exporter records
    # original Python pin and selected existing instantiations separately.
    return source


def run(args):
    source=args.source.read_text(encoding='utf-8');device=extract(source)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'device.cu').write_text(device)
    lib=C.CDLL(str(args.cache/'packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'))
    lib.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)]
    lib.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
    lib.nvrtcAddNameExpression.argtypes=[C.c_void_p,C.c_char_p]
    lib.nvrtcGetLoweredName.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_char_p)]
    for name in ('nvrtcGetProgramLogSize','nvrtcGetPTXSize'):getattr(lib,name).argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
    for name in ('nvrtcGetProgramLog','nvrtcGetPTX'):getattr(lib,name).argtypes=[C.c_void_p,C.c_void_p]
    def check(rc):
        if rc:raise RuntimeError('NVRTC '+str(rc))
    hdr='''#pragma once
typedef unsigned char uint8_t; typedef signed char int8_t;
typedef unsigned short uint16_t; typedef short int16_t;
typedef unsigned int uint32_t; typedef int int32_t;
typedef unsigned long uint64_t; typedef long int64_t;
typedef unsigned long uintptr_t; typedef long intptr_t;
'''
    stdlib='#pragma once\ntypedef unsigned long size_t; typedef long ptrdiff_t;\n'
    program=C.c_void_p();names=(C.c_char_p*2)(b'stdint.h',b'stdlib.h');headers=(C.c_char_p*2)(hdr.encode(),stdlib.encode())
    check(lib.nvrtcCreateProgram(C.byref(program),device.encode(),b'projection_control.cu',2,headers,names))
    expressions=[]
    for x in expressions:check(lib.nvrtcAddNameExpression(program,x.encode()))
    options=['--gpu-architecture=compute_90a','--std=c++17','--generate-line-info','--device-as-default-execution-space',
             '--include-path='+str(args.cache/'packages/nvidia/cuda_runtime/include'),
             '--include-path='+str(args.cache/'packages/nvidia/cuda_nvcc/include'),
             '--include-path='+str(args.cache/'cccl/nvidia/cuda_cccl/include')]
    argv=(C.c_char_p*len(options))(*(x.encode() for x in options))
    rc=lib.nvrtcCompileProgram(program,len(options),argv)
    size=C.c_size_t();check(lib.nvrtcGetProgramLogSize(program,C.byref(size)))
    log=C.create_string_buffer(size.value);check(lib.nvrtcGetProgramLog(program,log))
    (args.out/'nvrtc.log').write_bytes(log.value);print(log.value.decode(),flush=True)
    manifest=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),device_sha256=hashlib.sha256(device.encode()).hexdigest(),
                  adapters=['Generated device-only input, no device body removed','stdint.h virtual fixed-width declarations'], virtual_stdint=hdr,virtual_stdlib=stdlib,
                  options=options,expressions=expressions,nvrtc_status=rc,gpu_execution=False)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    check(rc)
    lowered=[]
    for x in expressions:
        value=C.c_char_p();check(lib.nvrtcGetLoweredName(program,x.encode(),C.byref(value)));lowered.append(value.value.decode())
    check(lib.nvrtcGetPTXSize(program,C.byref(size)));ptx=C.create_string_buffer(size.value);check(lib.nvrtcGetPTX(program,ptx))
    (args.out/'device.ptx').write_bytes(ptx.value)
    lowered=re.findall(r'\.visible \.entry\s+([^\s(]+)',ptx.value.decode())
    cmd=[str(args.cache/'packages/nvidia/cuda_nvcc/bin/ptxas'),'-arch=sm_90a','-O3','-v','--warn-on-spills',str(args.out/'device.ptx'),'-o',str(args.out/'device.cubin')]
    result=subprocess.run(cmd,capture_output=True,text=True)
    (args.out/'ptxas.log').write_text(result.stdout+result.stderr);print(result.stdout+result.stderr,flush=True)
    result.check_returncode()
    disasm=subprocess.run([str(args.cache/'cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin/nvdisasm'),str(args.out/'device.cubin')],capture_output=True,text=True,check=True)
    (args.out/'device.sass').write_text(disasm.stdout)
    manifest.update(lowered=lowered,cubin_sha256=hashlib.sha256((args.out/'device.cubin').read_bytes()).hexdigest(),
                    ptx_sha256=hashlib.sha256(ptx.value).hexdigest(),sass_sha256=hashlib.sha256(disasm.stdout.encode()).hexdigest(),
                    ptxas_command=cmd,ptxas_status=result.returncode)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--cache',type=Path,default=Path('/root/.cache/okc-cuda128'))
    run(p.parse_args())
