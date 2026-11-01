"""CPU-only NVRTC gate for the G1b4 boundary device body, no host/API proof."""
import argparse,ast,json
from pathlib import Path
from types import SimpleNamespace
from attn_local_cuda128_compile import compile_source

def run(source,out,cache):
    s=source.read_text(encoding='utf8');marker='template<typename F> void glue_launch'
    assert s.count(marker)==1
    device=s[:s.index(marker)]
    for h in ('#include <tvm/ffi/container/tensor.h>\n','#include <tvm/ffi/extra/c_env_api.h>\n','#include <stdexcept>\n','using TV = tvm::ffi::TensorView;\n'):
        assert device.count(h)==1;device=device.replace(h,'')
    # Reuse established attention NVRTC driver with explicit declaration/extract adapter.
    headers='\n'.join(line for line in device.splitlines() if line.startswith('#include'))+'\n'
    device='\n'.join(line for line in device.splitlines() if not line.startswith('#include'))+'\n'
    adapted=headers+'namespace {\nconstexpr float NEG_INF = -INFINITY;\n'+device+'\ntemplate <int L, int MODE>\nstruct Instance {\n'
    out.mkdir(parents=True,exist_ok=True);p=out/'device-adapter.cu';p.write_bytes(adapted.encode())
    compile_source(SimpleNamespace(source=p,out=out,cache=cache,expression='&glue_cluster_boundary<4,32,false,false,true,false>',define=['--use_fast_math']))
    (out/'adapter.json').write_text(json.dumps({'original_source':str(source),'removed':'TVM/host-only includes and TensorView alias; source truncated at host glue_launch','added':'anonymous namespace and unused infinity declaration for shared driver interface','device_math_changed':False,'host_api_validated':False,'gpu_execution':False},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('--cache',type=Path,default=Path('/root/.cache/okc-cuda128'));a=p.parse_args();run(a.source,a.out,a.cache)
