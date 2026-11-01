"""Wrap a pure prefill candidate with two validated large-M BF16 Lt plans.

Usage: python tools/prefill_proj_lt_integrate.py INPUT.py OUTPUT.py
Source/output are explicit: this tool never overwrites a worker's candidate.
"""
from pathlib import Path
import ast
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = r'''

# Large-M BF16 projection plans; e15f6a9a9ff047ef paired diagnostic.
from torch.utils.cpp_extension import load_inline as _pplt_load_inline
_PPLT_CPP=@@CPP@@
_PPLT_FUNCS=@@FUNCS@@
_PPLT_CFG={'pooler':[66,29,1,0,0,1,35,0,3],
           'unpooler':[66,197,1,0,0,2,35,0,5]}

def _pplt_build():
    errors=[]
    for i,flag in enumerate(('-l:libcublasLt.so.13','-lcublasLt','-l:libcublasLt.so.12')):
        try:
            return _pplt_load_inline(name='okc_prefill_dense_lt_v1_'+str(i),cpp_sources=_PPLT_CPP,
                functions=_PPLT_FUNCS,with_cuda=True,extra_cflags=['-O2','-std=c++20'],
                extra_ldflags=[flag],verbose=False)
        except Exception as exc:errors.append(str(exc)[-300:])
    raise RuntimeError('prefill dense Lt build failed: '+' | '.join(errors))

def _pplt_forward(x,w,state,name):
    # Shape/stride/alignment dispatch only; no input-value branching or caching.
    if x.shape[0]!=16384 or not x.is_contiguous() or x.data_ptr()%256:
        return _gemm(x,w)
    lt,p,ws,n,k=state['_pplt'][name]
    if x.shape[1]!=k or tuple(w.shape)!=(n,k):return _gemm(x,w)
    out=torch.empty((16384,n),dtype=torch.bfloat16,device=x.device)
    rc=lt.lt_run(p,0,w,x,out,ws)
    if rc:raise RuntimeError('prefill dense Lt status '+str(rc))
    return out

_pplt_original_prepare=prepare
def prepare(weights):
    state=_pplt_original_prepare(weights)
    lt=_pplt_build();plans={}
    for name,cfg in _PPLT_CFG.items():
        w=state['dense'][name];n,k=w.shape
        assert w.dtype==torch.bfloat16 and w.is_contiguous() and w.data_ptr()%256==0
        ws=torch.empty(32<<20,dtype=torch.uint8,device=w.device)
        p=lt.lt_plan_new(16384,n,k,0,ws.numel())
        if p<0:raise RuntimeError('prefill dense Lt plan rejected '+name)
        lt.lt_add_config(p,cfg)
        if lt.lt_count(p)!=1:raise RuntimeError('prefill dense Lt config rejected '+name)
        plans[name]=(lt,p,ws,n,k)
    state['_pplt']=plans
    print('prefill dense Lt enabled M16384 pooler/unpooler layout0 FP32compute BF16io',flush=True)
    return state
'''

def generate(source,projection_source):
    if not source.startswith('# okc: problem=muse_prefill_layer\n'):
        raise ValueError('expected explicit prefill problem header')
    tree=ast.parse(projection_source)
    values={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('LT_CPP','LT_FUNCS'):
                    values[target.id]=ast.literal_eval(node.value)
    replacements={
        '_gemm(ffn_input, dense["pooler"])':'_pplt_forward(ffn_input, dense["pooler"], state, "pooler")',
        '_gemm(expert_output, dense["unpooler"])':'_pplt_forward(expert_output, dense["unpooler"], state, "unpooler")',
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('expected one exact pure-candidate call: '+old)
        source=source.replace(old,new)
    result=source+CODE.replace('@@CPP@@',repr(values['LT_CPP'])).replace('@@FUNCS@@',repr(values['LT_FUNCS']))
    ast.parse(result)
    return result

def cpu_host_gate():
    """Execute actual injected host control flow with fake metadata/callables."""
    from types import SimpleNamespace
    class Tensor:
        def __init__(self,shape):self.shape=shape;self.dtype='bf16';self.device='fake'
        def is_contiguous(self):return True
        def data_ptr(self):return 512
        def numel(self):return 32<<20
    state={'dense':{'pooler':Tensor((4096,8192)),'unpooler':Tensor((8192,4096))},'sentinel':object()}
    old_sentinel=state['sentinel']
    class Lt:
        def __init__(self):self.plans=[]
        def lt_plan_new(self,*args):self.plans.append(args);return len(self.plans)-1
        def lt_add_config(self,p,c):return 0
        def lt_count(self,p):return 1
        def lt_run(self,*args):return 0
    lt=Lt();fallback=object()
    env={'prepare':lambda weights:state,
         '_gemm':lambda x,w:fallback,
         'torch':SimpleNamespace(bfloat16='bf16',uint8='u8',empty=lambda shape,**kw:Tensor(shape))}
    tree=ast.parse(CODE.replace('@@CPP@@',repr('cpu')).replace('@@FUNCS@@',repr([])))
    tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(tree,'<actual injected host scaffold>','exec'),env)
    env['_pplt_build']=lambda:lt
    assert env['prepare']({}) is state and state['sentinel'] is old_sentinel
    a,b=state['_pplt'].values()
    assert a[1]!=b[1] and a[2] is not b[2] and len(lt.plans)==2
    forward=env['_pplt_forward'];w=state['dense']['pooler']
    assert forward(Tensor((753,8192)),w,state,'pooler') is fallback
    assert forward(Tensor((16384,8192)),w,state,'pooler').shape==(16384,4096)
    assert forward(Tensor((16384,4096)),w,state,'pooler') is fallback
    print('CPU actual host scaffold: preserved state, independent plans/workspaces, shape fallback PASS')

if __name__=='__main__':
    cpu_host_gate()
    inp,out=map(Path,sys.argv[1:3])
    if inp.resolve()==out.resolve():raise ValueError('output must be a separate review artifact')
    result=generate(inp.read_text(),(ROOT/'muse_bf16_proj/submission.py').read_text())
    out.write_text(result)
    print('Wrote pure review artifact',out,'sha256',hashlib.sha256(result.encode()).hexdigest())
