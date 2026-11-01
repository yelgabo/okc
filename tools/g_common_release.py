"""Single-arm, prepare-only full-model qualifier for two strictly pinned G arms.

No dispatch or comparative timing. Normal kernel_fn and timed GPU work unchanged.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from g_common_release_attn import patch_cuda
from g_common_release_body import BODY

BASE='ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
PINS={'common':'4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280',
      'short-rmw':'2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd'}
MOE_RESOURCE=r'''
std::vector<int64_t> qual_native_resources() {
  std::vector<int64_t> out;const size_t shared=okc_smem_bytes();
  int dev=0;cudaDeviceProp prop{};
  TORCH_CHECK(cudaGetDevice(&dev)==cudaSuccess && cudaGetDeviceProperties(&prop,dev)==cudaSuccess,
      "GQUAL expert properties");
  for(int big=0;big<2;++big) {
    auto kernel=big?okc::moe_fused<true>:okc::moe_fused<false>;
    cudaFuncAttributes a{};int resident=0;
    TORCH_CHECK(cudaFuncSetAttribute((const void*)kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared)==cudaSuccess &&
        cudaFuncGetAttributes(&a,(const void*)kernel)==cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,okc::NTL,shared)==cudaSuccess,
        "GQUAL expert selected resources");
    const int64_t row[]={a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,
        (int64_t)shared,okc::NTL,a.maxThreadsPerBlock,resident,prop.multiProcessorCount,
        prop.cooperativeLaunch,(int64_t)resident*prop.multiProcessorCount};
    out.insert(out.end(),row,row+10);
  }
  return out;
}
'''


def sha(s):return hashlib.sha256(s.encode()).hexdigest()


def load(path):
    spec=importlib.util.spec_from_file_location('_gq_'+path.stem,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def assignment(source,key):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==key)


def replace_nodes(source,changes):
    lines=source.splitlines(keepends=True)
    for n,text in sorted(changes,key=lambda p:p[0].lineno,reverse=True):lines[n.lineno-1:n.end_lineno]=[text+'\n']
    return ''.join(lines)


def literal(value):
    quote=next(q for q in ('"""',"'''") if q not in value)
    result='r'+quote+value+quote;assert ast.literal_eval(result)==value;return result


def native_exports(source):
    edits=[]
    for key in ('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"):
        node=assignment(source,key);value=ast.literal_eval(node.value)
        edits.append((node,key+' = '+literal(value+MOE_RESOURCE)))
    node=assignment(source,'MOE_CPP_SRC');value=ast.literal_eval(node.value)
    edits.append((node,'MOE_CPP_SRC = '+literal(value+'\nstd::vector<int64_t> qual_native_resources();\n')))
    source=replace_nodes(source,edits)
    old='functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),'
    assert source.count(old)==1
    source=source.replace(old,old[:-1]+' + ["qual_native_resources"],',1)
    return source


def attention_exports(source):
    node=assignment(source,'ATT7_CUDA_SRC');before=ast.literal_eval(node.value);after=patch_cuda(before)
    assert '#define OKC_FLATCOOP1C 0' in before,'waiting predicate scope changed'
    edits=[(node,'ATT7_CUDA_SRC = '+literal(after))]
    node=assignment(source,'ATT7_CPP_SRC');value=ast.literal_eval(node.value)
    edits.append((node,'ATT7_CPP_SRC = '+literal(value+'\n#include <vector>\nvoid qual_att7_begin();\nstd::vector<int64_t> qual_att7_end();\nstd::vector<int64_t> qual_att7_versions();\n')))
    result=replace_nodes(source,edits)
    old='functions=["run"], extra_cuda_cflags='
    assert result.count(old)==1
    return result.replace(old,'functions=["run","qual_att7_begin","qual_att7_end","qual_att7_versions"], extra_cuda_cflags=',1)


def build(root,arm='common'):
    assert arm in PINS
    root=Path(root);base=(root/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf-8')
    assert sha(base)==BASE
    sys.path.insert(0,str(root/'tools'))
    try:
        source,_=load(root/'tools/g_integration_rmw_release.py').build(base,arm=='short-rmw')
        assert sha(source)==PINS[arm],'known fixed arm drift'
        qual=load(root/'tools/prod_size_qualify.py')
        oracle,ref_manifest=qual.helpers(root)
        functions={n.name:n for n in ast.parse(qual.BODY).body if isinstance(n,ast.FunctionDef)}
        support='\n\n'.join(ast.get_source_segment(qual.BODY,functions[name])
            for name in ('_qual_finite','_qual_restore','_qual_poison','_qual_validate'))
    finally:sys.path.pop(0)
    original=source
    source=native_exports(attention_exports(source))
    owners=(root/'tools/dl_g1_layer_owners.py').read_text(encoding='utf-8')
    owners=owners.replace(";arm['ma_ext'].pop('main',None);arm['ma_ext'].pop('b4',None)",'')
    owners=owners.replace("arm['ma_ext']=dict(state['ma_ext']);arm['ma_ext'].pop('main',None);arm['ma_ext'].pop('b4',None)","arm['ma_ext']=dict(state['ma_ext'])")
    owners=owners.replace("    arm['_wl_stage_on']=bool(on)\n",'')
    assert "pop('main'" not in owners and "pop('b4'" not in owners
    # Shared reference constants must not alter an existing product global.
    def values(text):
        return {t.id:ast.dump(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign)
                for t in n.targets if isinstance(t,ast.Name)}
    old_values,new_values=values(original),values(oracle)
    collisions=old_values.keys()&new_values.keys()
    assert all(old_values[k]==new_values[k] for k in collisions),'reference changes product constants'
    suffix='\n# Single-arm all12 startup qualification; no comparative timings.\nimport torch,time,math\n'+oracle+'\n'+support+'\n'+owners
    suffix+='\n_GQ_ARM='+repr(arm)+'\n_GQ_BASE_SHA='+repr(PINS[arm])+'\n'+BODY
    for n in ast.walk(ast.parse(suffix)):
        if isinstance(n,ast.Attribute):assert n.attr not in ('Event','elapsed_time','profiler'),'new timing/profiling code'
    output=source+suffix
    before=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn']
    after=[n for n in ast.parse(output).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn']
    assert len(before)==len(after) and all(ast.dump(a)==ast.dump(b) for a,b in zip(before,after)),'timed entry changed'
    assert len(output.encode())<1_000_000
    return output,dict(arm=arm,base607_sha256=BASE,common_input_sha256=PINS[arm],source_sha256=sha(output),
        bytes=len(output.encode()),reference=ref_manifest,owners_sha256=sha(owners),body_sha256=sha(BODY),
        source_gpu_work_unchanged=True,timed_kernel_fn_AST_unchanged=True,
        prepare_only_host_audit=True,normal_audit_disabled=True,
        H200_qualified=False,dispatch_enabled=False,comparative_timings=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--arm',choices=tuple(PINS),default='common');args=parser.parse_args()
    source,manifest=build(args.root,args.arm);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(source.encode());args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('arm','source_sha256','bytes')}))
