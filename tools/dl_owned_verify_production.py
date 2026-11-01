"""Source-transparent owned verify backend pruning. No GPU/public actions.

Input is the frozen four-cell diagnostic. Only unreachable registrations, Python
closures and entire unreferenced C++ definitions are removed. Active kernels kept.
"""
import argparse,ast,hashlib,json,re,importlib.util,sys
from functools import lru_cache
from pathlib import Path

PIN='384933697aadb2ebfc2abe672b216751e55fd5949e01c0c57d03caa586fe6fe3'

def once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)

def function(s,anchor):
    a=s.index(anchor);z=s.index('{',a)+1;depth=1
    while depth:
        depth+=(s[z]=='{')-(s[z]=='}');z+=1
    return s[a:z]

def literal(s,name):
    return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)

def visible_literal(s):
    for q in ("'''",'"""'):
        if q in s:continue
        value='r'+q+s+q
        try:
            if ast.literal_eval(value)==s:return value
        except SyntaxError:pass
    raise ValueError('native source must remain a visible raw multiline literal')

def replace_node(s,name,value,func=False):
    node=next(n for n in ast.parse(s).body if getattr(n,'name',None)==name if func) if func else next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)
    text=value if func else name+' = '+visible_literal(value)
    return once(s,ast.get_source_segment(s,node),text)

def route_only(s):
    prefix=s[:s.index('// Marlin\'s moe_align')]
    rounding=function(s,'__device__ __forceinline__ float round_bf16(')
    constants=s[s.index('constexpr int NORM_N'):s.index('template <typename InT>\n__global__')]
    a=s.index('constexpr int ROUTE_EXPERTS');z=s.index('// Warms L2 with weights',a)
    route=s[a:z]
    host=function(s,'std::vector<torch::Tensor> route_fwd(')
    return prefix+rounding+'\n'+constants+route+'\n} // namespace\n'+host+'\n'

def wd_helpers(s):
    # All dense-WD dependencies are in this prefix plus the tensor-map encoder.
    a=s.index('template <int MT,')
    return s[:a]+'\n#ifndef SK_DEVICE_ONLY\n'+function(s,'PFN_cuTensorMapEncodeTiled_v12000 encode_fn(')+'\n#endif\n} // namespace\n'

def boundary_only(s):
    prefix=s[:s.index('__device__ __forceinline__ float block_sum(')]
    helpers=function(s,'__device__ __forceinline__ float4 bf16x4(')+'\n'+function(s,'__device__ __forceinline__ float bf16r(')
    lo=s.index('template <int NT>\n__device__ __forceinline__ float block_sum_n(')
    hi=s.index('// launch_ex with a runtime cluster size',lo)
    body=s[lo:hi]
    launch=s[s.index('template <typename Kern, typename... Args>\nvoid launch_cl('):s.index('template <typename F>\nvoid with_geo(')]
    dispatch='template <typename F>\nvoid with_geo(int geo, F fn) {\n    TORCH_CHECK(geo==4,"owned verify boundary geometry");\n    fn(attn_boundary_geo_kernel<1,512>);\n}\n'
    host=function(s,'std::vector<torch::Tensor> attn_boundary(')
    start=host.index('    if (geo == 0) {');mid=host.index('    } else {',start);end=host.index('    C10_CUDA_KERNEL_LAUNCH_CHECK();',mid)
    selected=host[mid+len('    } else {'):end];assert selected.rstrip().endswith('}')
    selected=selected.rstrip()[:-1]
    host=host[:start]+selected+'\n'+host[end:]
    host=host.replace('geo >= 0 && geo < kGeos','geo == 4')
    return prefix+helpers+'\n'+body+launch+dispatch+'\n} // bnd\n} // namespace\n'+host+'\n'

def closure(source,roots):
    t=ast.parse(source);nodes={}
    for n in t.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):keys=[n.name]
        elif isinstance(n,ast.Assign):keys=[v.id for v in n.targets if isinstance(v,ast.Name)]
        else:continue
        for k in keys:nodes[k]=n
    todo=set(roots);seen=set();keep=set()
    while todo-seen:
        key=(todo-seen).pop();seen.add(key)
        if key not in nodes:continue
        n=nodes[key];keep.add(id(n));todo.update(x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load))
    return '\n\n'.join(ast.get_source_segment(source,n) for n in t.body if id(n) in keep or isinstance(n,(ast.Import,ast.ImportFrom)))+'\n'

def _build(source,tokens):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN,'unreviewed owned verify diagnostic'
    assert tokens in ((128,),(256,),(128,256))
    out=source;initial={name:literal(source,name) for name in ('_CUDA_SRC','_ONE_SRC','_ONE_HOST_SRC','_BND_SRC','_MOE1_SRC','_SKP_SRC','_SKP_WD_SRC')}
    out=replace_node(out,'_CUDA_SRC',route_only(initial['_CUDA_SRC']))
    cpp=literal(out,'_CPP_SRC');keep=['std::vector<torch::Tensor> route_fwd(']
    if 256 in tokens:keep.append('std::vector<torch::Tensor> attn_boundary(')
    cpp_new='#include <torch/extension.h>\n'+'\n'.join(cpp[cpp.index(a):cpp.index(';',cpp.index(a))+1] for a in keep)
    out=replace_node(out,'_CPP_SRC',cpp_new)
    if 256 in tokens:out=replace_node(out,'_BND_SRC',boundary_only(initial['_BND_SRC']))
    fn=next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name=='_ext')
    text=ast.get_source_segment(out,fn)
    if 256 not in tokens:text=text.replace('cuda_sources=[_CUDA_SRC, _BND_SRC]','cuda_sources=_CUDA_SRC').replace('functions=["route_fwd", "attn_boundary"]','functions=["route_fwd"]')
    out=replace_node(out,'_ext',text,True)
    if 128 in tokens:
        out=replace_node(out,'_SKP_SRC',wd_helpers(initial['_SKP_SRC']))
        fn=next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name=='_skp_load');text=ast.get_source_segment(out,fn)
        text=text.replace('_SKP_SRC + _SKP_GEMV_SRC + _SKP_WD_SRC','_SKP_SRC + _SKP_WD_SRC')
        text=once(text,'functions=["sk_fwd", "skf_fwd", "cgemv_fwd", "rgemv_fwd", "sgemv_fwd",\n                             "wd_fwd"]','functions=["wd_fwd"]')
        out=replace_node(out,'_skp_load',text,True)
    fn=next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name=='prepare');text=ast.get_source_segment(out,fn)
    if 128 not in tokens:
        a=text.index('    state["verify_diag"]');z=text.index('    for batch in (16,32):',a);text=text[:a]+text[z:]
    text=text.replace('for batch in (16,32):','for batch in '+repr(tuple(t//8 for t in tokens))+':')
    out=replace_node(out,'prepare',text,True)
    fn=next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name=='_slice_proj');text=ast.get_source_segment(out,fn)
    if 128 not in tokens:
        text=text[:text.index('\n')]+'\n'+text[text.index('    entry ='):]
    out=replace_node(out,'_slice_proj',text,True)
    fn=next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name=='_slice_lt')
    text=ast.get_source_segment(out,fn);tree=ast.parse(text)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For))
    assert isinstance(loop.iter,ast.Tuple)
    rows=[ast.literal_eval(v) for v in loop.iter.elts];rows=[v for v in rows if v[1] in tokens]
    text=once(text,ast.get_source_segment(text,loop.iter),repr(tuple(rows)))
    out=replace_node(out,'_slice_lt',text,True)
    fn=next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn');text=ast.get_source_segment(out,fn)
    text=text.replace('batch in (16,32)','batch in '+repr(tuple(t//8 for t in tokens)))
    out=replace_node(out,'kernel_fn',text,True)
    # Existing setup queries only the selected L8 builds; unsupported resource/plan rejects.
    host=literal(out,'_ONE_HOST_SRC');a=host.index('#define DL_ONE_INSTANCES(X)');z=host.index('\n\ntemplate',a)
    host=host[:a]+'#define DL_ONE_INSTANCES(X) X(8,3) X(8,51)'+host[z:]
    out=replace_node(out,'_ONE_HOST_SRC',host)
    out=closure(out,('prepare','kernel_fn','_slice_main','_slice_moe','_slice_one'))
    out='# okc: problem=muse_decode_layer\n# Owned verify backend closure; source/CPU only, no GPU qualification.\n'+out
    ast.parse(out)
    revised={name:literal(out,name) for name in ('_ONE_SRC','_MOE1_SRC')}
    assert all(initial[k]==v for k,v in revised.items())
    return out,{'input_pin':PIN,'tokens':tokens,'bytes':len(out.encode()),'sha256':hashlib.sha256(out.encode()).hexdigest(),
        'unchanged_large_active_literals':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in revised.items()},'GPU_qualified':False}

@lru_cache(maxsize=2)
def frozen_source(root):
    root=Path(root);path=root/'tools/dl_verify_pair_slices.py'
    spec=importlib.util.spec_from_file_location('_owned_production_frozen_slice',path)
    module=importlib.util.module_from_spec(spec);saved=list(sys.path)
    try:
        sys.path.insert(0,str(root/'tools'));spec.loader.exec_module(module)
        source=module.build(root)[0]['owned']
    finally:sys.path[:]=saved
    assert hashlib.sha256(source.encode()).hexdigest()==PIN
    return source

def build(root,batches=(16,32)):
    assert tuple(batches) in ((16,),(32,),(16,32))
    source=frozen_source(Path(root).resolve())
    return _build(source,tuple(b*8 for b in batches))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);src=a.input.read_text(encoding='utf8');reports=[]
    for ts in ((128,),(256,),(128,256)):
        s,m=_build(src,ts);reports.append(m);(a.output/('owned-'+str(ts).replace(' ','')+'.py')).write_text(s,encoding='utf8',newline='\n')
    (a.output/'manifest.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf8');print(json.dumps(reports,indent=2))
