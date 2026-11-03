"""Compile the fixed row-owned entry locally without claiming target parity."""
import argparse
import inspect
import json
import subprocess
from pathlib import Path

from gopt_ws_owned import build, sha


def compile_device(out):
    out.mkdir(parents=True, exist_ok=True)
    _, _, call = build()
    source = call['cuda_sources'].split('#include <torch/extension.h>')[0]
    source += '''
void compile_entries(okc::Params p) {
  okc::moe_fused<false><<<1,160>>>(p);
  okc::moe_fused<true><<<1,160>>>(p);
}
#include <cstdio>
int main() {
  const size_t base = ((sizeof(okc::Smem)+127)/128)*128;
  std::printf("{\\"smem_base\\":%zu,\\"dynamic_smem\\":%zu,\\"threads\\":160}\\n",base,base+3*36864+8*1032*2);
}
'''
    (out/'device.cu').write_text(source)
    flags = call['extra_cuda_cflags'] + ['-std=c++17','-Xptxas=-v,--warn-on-spills']
    docker = ['docker','run','--rm','--platform','linux/arm64','--entrypoint','',
              '-v',str(out.resolve())+':/w','nvidia/cuda:12.8.1-devel-ubuntu24.04']
    process = subprocess.run(docker+['nvcc',*flags,'/w/device.cu','-o','/w/device'],capture_output=True,text=True)
    (out/'device.log').write_text(process.stdout+process.stderr)
    (out/'compile.json').write_text(json.dumps(dict(exit_code=process.returncode,flags=flags,native_sha256=sha(call['cuda_sources'])),indent=2)+'\n')
    print('local device compiler exit',process.returncode,flush=True)
    process.check_returncode()
    result = subprocess.run(docker+['/w/device'],capture_output=True,text=True,check=True)
    resources = json.loads(result.stdout)
    assert resources['dynamic_smem'] <= 160*1024
    (out/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    print(resources,flush=True)
    result = subprocess.run(docker+['cuobjdump','-res-usage','-sass','/w/device'],capture_output=True,text=True,check=True)
    (out/'device.sass').write_text(result.stdout)


def compile_host(out):
    import gopt_ws_split_compile as original
    source = inspect.getsource(original.full_host)
    source = source.replace('source, manifest, controls, candidates = build()',
                            'source, manifest, call = build(); candidates = [call]')
    source = source.replace("(('tiny', 'tiny2', 'mid'), candidates)", "(('owned',), candidates)")
    namespace = dict(original.__dict__)
    namespace['build'] = build
    exec(source, namespace)
    namespace['full_host'](out)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--host',action='store_true')
    a=p.parse_args()
    (compile_host if a.host else compile_device)(a.out)
