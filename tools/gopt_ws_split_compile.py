"""Inspect selected local device builds and compile full host translation units."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess

from gopt_ws_split import build, ROOT, sha


def inspect(folder):
    old = ROOT / 'docs/steering/evidence/2026-11-02-gopt-skinny-depth'
    for key in ('tiny', 'tiny2', 'mid'):
        current = (folder / (key+'-control.sass')).read_bytes()
        archived = gzip.decompress((old / (key+'-control.sass.gz')).read_bytes())
        assert current == archived, key+' disabled SASS drift'
    print('PASS all disabled emitted SASS matches archived NS3', flush=True)


def full_host(out):
    out.mkdir(parents=True, exist_ok=True)
    source, manifest, controls, candidates = build()
    rows = []
    for key, call in zip(('tiny', 'tiny2', 'mid'), candidates):
        native = call['cuda_sources']
        anchor = 'template <bool PB>\n__device__ __forceinline__ void wgs_phase('
        native = native.replace(anchor, '__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'+anchor)
        (out/(key+'.cu')).write_text('#include <torch/types.h>\n'+native)
        (out/(key+'.cpp')).write_text('#include <torch/extension.h>\n'+call['cpp_sources']+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'+ '\n'.join('m.def("'+name+'", &'+name+');' for name in call['functions'])+'\n}\n')
        flags = call['extra_cuda_cflags'] + ['-std=c++20','-Xptxas=-v,--warn-on-spills',
                 '-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__',
                 '-D__CUDA_NO_BFLOAT16_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__','--expt-relaxed-constexpr',
                 '-DTORCH_EXTENSION_NAME=gopt_ws_split_'+key,'-DTORCH_API_INCLUDE_EXTENSION_H','-D_GLIBCXX_USE_CXX11_ABI=1','-Xcompiler=-fPIC']
        (out/(key+'.flags.json')).write_text(json.dumps(flags))
        script='''set -eu
mkdir -p /w/stub/c10/cuda/impl
printf '#pragma once\n' > /w/stub/c10/cuda/impl/cuda_cmake_macros.h
python3 - "$1" <<'INNER'
import json,subprocess,sys,sysconfig
import torch.utils.cpp_extension as c
key=sys.argv[1]
inc=['-I'+p for p in c.include_paths()]+['-I/w/stub','-I'+sysconfig.get_paths()['include']]
flags=json.load(open('/w/'+key+'.flags.json'))
subprocess.run(['nvcc',*flags,*inc,'-c','/w/'+key+'.cu','-o','/w/'+key+'.o'],check=True)
subprocess.run(['g++','-std=c++20','-fPIC','-DTORCH_EXTENSION_NAME=gopt_ws_split_'+key,'-D_GLIBCXX_USE_CXX11_ABI=1',*inc,'-c','/w/'+key+'.cpp','-o','/w/'+key+'-binding.o'],check=True)
INNER
'''
        command = ['docker','run','--rm','--platform','linux/arm64','--entrypoint','',
                   '-v',str(out.resolve())+':/w','okc-torch-headers','sh','-c',script,'sh',key]
        process=subprocess.run(command,capture_output=True,text=True)
        (out/(key+'.log')).write_text(process.stdout+process.stderr)
        rows.append(dict(key=key,exit_code=process.returncode,native_sha256=sha(call['cuda_sources']),flags=flags))
        (out/'compile.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(key,'full translation compiler exit',process.returncode,flush=True)
        process.check_returncode()
    print('PASS full native host/device translation and binding objects; target shared linking/admission still pending',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',type=Path,required=True)
    p.add_argument('--host',type=Path,required=True)
    a=p.parse_args()
    inspect(a.device)
    full_host(a.host)
