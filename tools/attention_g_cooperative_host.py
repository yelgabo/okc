"""OFF offline host-only repair for selected607 ATT7 software-grid joins.

No GPU dispatch. Disabled returns exact607. Preserve all native math/fences/ABI.
"""
import argparse,ast,hashlib,json
from pathlib import Path
PIN='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
ANCHOR='  if (cl || pdl) {\n    // cluster dims and / or programmatic stream serialization'
REPAIR=r'''  // OFF-prototype common progress repair: capacity alone is not cooperative admission.
  const bool waiting_finalize = q.coop && ((!p.flat && !cl) ||
      (p.flat && (L == 8 || OKC_FLATCOOP1C)));
  if (waiting_finalize) {
    TORCH_CHECK(!pdl && !cl, "ATT7_PROGRESS requires sequential non-cluster launch");
    int device=0, resident=0; cudaDeviceProp prop{}; cudaFuncAttributes attrs{};
    TORCH_CHECK(cudaGetDevice(&device)==cudaSuccess &&
                cudaGetDeviceProperties(&prop,device)==cudaSuccess,
                "ATT7_PROGRESS device properties");
    TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch &&
                prop.multiProcessorCount>0, "ATT7_PROGRESS SM90 cooperative support");
    const int threads=p.flat ? Cfg<L,true>::NTX : Cfg<L,false>::NTX;
    TORCH_CHECK(cudaFuncSetAttribute((const void*)kern,cudaFuncAttributeMaxDynamicSharedMemorySize,sm)==cudaSuccess &&
                cudaFuncGetAttributes(&attrs,(const void*)kern)==cudaSuccess,
                "ATT7_PROGRESS selected function attributes");
    const unsigned long long blocks=(unsigned long long)grid.x*grid.y*grid.z;
    TORCH_CHECK(grid.x>0 && grid.y>0 && grid.z>0 && threads>0 &&
                threads<=attrs.maxThreadsPerBlock && sm>=0 &&
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kern,threads,sm)==cudaSuccess &&
                resident>0 && blocks<=(unsigned long long)resident*prop.multiProcessorCount,
                "ATT7_PROGRESS selected function admission");
    void* args[]={&q};
    const cudaError_t launch=cudaLaunchCooperativeKernel((const void*)kern,grid,
        dim3((unsigned)threads,1,1),args,sm,stream);
    TORCH_CHECK(launch==cudaSuccess,"ATT7_PROGRESS launch ",cudaGetErrorString(launch));
    return;
  }
'''

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def build(source,enabled=False):
    assert sha(source)==PIN,'exact607 LF source required'
    if not enabled:return source
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='ATT7_CUDA_SRC')
    old=ast.literal_eval(node.value)
    assert old.count(ANCHOR)==1 and 'cudaLaunchCooperativeKernel' not in old
    assert old.count('programmaticStreamSerializationAllowed = 0;')==1
    new=old.replace(ANCHOR,REPAIR+ANCHOR)
    assert new.replace(REPAIR,'')==old
    lines=source.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['ATT7_CUDA_SRC = r"""'+new+'"""\n']
    output=''.join(lines);ast.parse(output)
    a=ast.dump(ast.parse(source),include_attributes=False);b=ast.dump(ast.parse(output.replace(REPAIR,'')),include_attributes=False)
    assert a==b,'non-host source changed'
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--enable',action='store_true');a=p.parse_args()
    raw=a.source.read_bytes();assert hashlib.sha256(raw).hexdigest()==PIN
    out=build(raw.decode(),a.enable);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(out.encode())
    manifest=dict(base_sha256=PIN,enabled=a.enable,source_sha256=sha(out),bytes=len(out.encode()),repair_sha256=sha(REPAIR),scope='OFF source-only; no runtime qualification or speedup',gpu_execution=False)
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');print(json.dumps(manifest))
