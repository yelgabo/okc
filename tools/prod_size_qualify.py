"""Prepare-only single-arm qualifier for the repaired full-four-cell hybrid."""
import argparse,ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from prod_size_qualify_body import BODY

BASE='81f056f5e495f5cab0a66183a5b56cab783e7fa47732ed58e0ffa4feed5127a1'
OWNED='2b4e0740de26a92fa9ef0f260c28e55f48f9fb2ae8fa20de863e4d8e8718635d'

MOE_GETTER=r'''
std::vector<int64_t> qual_moe_resources() {
    auto kernel=moe1::moe_main_kernel<51>;
    const int bytes=sizeof(moe1::SmemMain),threads=moe1::kNTM;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    int blocks=0,device=0,sms=0,coop=0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,bytes));
    return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,threads,
            a.maxThreadsPerBlock,blocks,sms,coop,blocks*sms};
}
'''
ONE_GETTER=r'''
template<int MODE> void qual_one_append(std::vector<int64_t>& out) {
    using I=one::Instance<8,MODE>;I::prepare();auto kernel=one::attn_one_kernel<8,MODE>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,I::BYTES));
    cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    int blocks=0,device=0,sms=0,coop=0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,I::NT,I::BYTES));
    const int64_t row[]={a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,I::BYTES,
                         I::NT,a.maxThreadsPerBlock,blocks,sms,coop,blocks*sms};
    out.insert(out.end(),row,row+10);
}
std::vector<int64_t> qual_one_resources() {
    std::vector<int64_t> out;qual_one_append<3>(out);qual_one_append<51>(out);return out;
}
'''

def replace(source,node,text):
 lines=source.splitlines(keepends=True);starts=[0]
 for line in lines:starts.append(starts[-1]+len(line))
 a=starts[node.lineno-1]+len(lines[node.lineno-1].encode()[:node.col_offset].decode())
 z=starts[node.end_lineno-1]+len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
 return source[:a]+text+source[z:]

def visible(value):
 q=next(q for q in ("'''",'"""') if q not in value);text='r'+q+value+q;assert ast.literal_eval(text)==value;return text

def assignment(source,name):
 return next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)

def resource_api(source):
 original=source
 for name,tail in [('_MOE1_HOST_SRC',MOE_GETTER),('_ONE_HOST_SRC',ONE_GETTER),('_MOE1_CPP_SRC','\nstd::vector<int64_t> qual_moe_resources();\n'),('_ONE_CPP_SRC','\nstd::vector<int64_t> qual_one_resources();\n')]:
  node=assignment(source,name);value=ast.literal_eval(node.value);source=replace(source,node.value,visible(value+tail))
 for fn_name,export in [('_moe1_load','qual_moe_resources'),('_one_load','qual_one_resources')]:
  tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn_name)
  call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and getattr(n.func,'id','')=='load_inline')
  kw=next(k for k in call.keywords if k.arg=='functions');exports=ast.literal_eval(kw.value);assert export not in exports
  source=replace(source,kw.value,repr(exports+[export]))
  tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn_name)
  call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and getattr(n.func,'id','')=='load_inline')
  kw=next(k for k in call.keywords if k.arg=='name');old=ast.literal_eval(kw.value)
  source=replace(source,kw.value,repr(old+'_startup_attrs_v1'))
 for name in ('_MOE1_SRC','_ONE_SRC'):
  assert ast.literal_eval(assignment(original,name).value)==ast.literal_eval(assignment(source,name).value),'device literal changed'
 return source

def helpers(root):
 original=(root/'tools/dl_make_bench_pub.py').read_text(encoding='utf8');node=assignment(original,'BODY');text=ast.literal_eval(node.value)
 gate=ast.parse((root/'tools/dl_g1_layer_gate.py').read_text(encoding='utf8'))
 names=next(ast.literal_eval(n.value) for n in gate.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NAMES')
 prelude=next(ast.literal_eval(n.value) for n in gate.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='PRELUDE')
 functions={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 copied='\n\n'.join(ast.get_source_segment(text,functions[n]) for n in names)
 # Identical adaptive arithmetic and thresholds, plus explicit finite guards.
 oracle=copy.deepcopy(functions['_ab_validate']);oracle.name='_qual_oracle'
 bound=next(n for n in ast.walk(oracle) if isinstance(n,ast.FunctionDef) and n.name=='bound')
 bound.body.insert(0,ast.parse("assert math.isfinite(e),'QUAL_NONFINITE_CALIBRATION'").body[0])
 rel=next(n for n in ast.walk(oracle) if isinstance(n,ast.FunctionDef) and n.name=='rel')
 assert len(rel.body)==1 and isinstance(rel.body[0],ast.Return)
 rel.body=[ast.Assign(targets=[ast.Name(id='value',ctx=ast.Store())],value=rel.body[0].value),ast.parse("assert math.isfinite(value),'QUAL_NONFINITE_RELATIVE_METRIC'").body[0],ast.Return(value=ast.Name(id='value',ctx=ast.Load()))]
 oracle=ast.fix_missing_locations(oracle)
 return prelude+'\nHIDDEN=8192\nD_EXPERT=4096\nLOCAL_EXPERTS=32\nN_EXPERTS=256\nTOP_K=8\nROPE_THETA=500000.\n'+copied+'\n\n'+ast.unparse(oracle)+'\n',dict(helper_source_sha256=hashlib.sha256(text.encode()).hexdigest(),prelude_sha256=hashlib.sha256(prelude.encode()).hexdigest(),copied_helpers_sha256=hashlib.sha256(copied.encode()).hexdigest(),helper_names=names,helper_bytes=len(copied.encode()))

def build(root,source):
 assert hashlib.sha256(source.encode()).hexdigest()==BASE,'unreviewed progress production source'
 tree=ast.parse(source);node=assignment(source,'_VERIFY_HYBRID_SOURCE_OWNED');owned=ast.literal_eval(node.value)
 assert hashlib.sha256(owned.encode()).hexdigest()==OWNED,'owned progress epoch drift'
 changed=resource_api(owned)
 # Reuse the existing readable outer literal, preserving every backend byte.
 sys.path.insert(0,str(root/'tools'))
 try:
  from dl_selected_pair import literal
  revised=replace(source,node.value,literal(changed))
 finally:sys.path.pop(0)
 oracle,manifest=helpers(root)
 suffix='\n# Single-arm startup checks; timed kernel_fn unchanged.\nimport torch,time,math\n'+oracle+'\n'+BODY
 for n in ast.walk(ast.parse(suffix)):
  if isinstance(n,ast.Attribute):assert n.attr not in ('Event','elapsed_time','profiler')
 result=revised+suffix
 before=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
 after=[n for n in ast.parse(result).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn']
 assert len(after)==1 and ast.dump(after[0])==ast.dump(before),'timed path changed'
 assert len(result.encode())<1000000,('QUALIFIER_NOFIT',len(result.encode()))
 manifest.update(bytes=len(result.encode()),sha256=hashlib.sha256(result.encode()).hexdigest(),base_sha256=BASE,owned_base=OWNED,owned_with_resource_api=hashlib.sha256(changed.encode()).hexdigest(),body_sha256=hashlib.sha256(BODY.encode()).hexdigest(),timed_kernel_unchanged=True,new_comparative_timings=False,H200_qualified=False)
 return result,manifest

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 text,m=build(a.root,a.source.read_text(encoding='utf8'));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(text.encode());a.output.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(m,indent=2))
