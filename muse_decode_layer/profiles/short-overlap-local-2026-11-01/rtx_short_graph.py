"""Run the proposed stream helper using its native FP32 router device body and a PyTorch BF16 pooler."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
import torch
from rtx_short_router import Router

parser = argparse.ArgumentParser()
parser.add_argument("source",type=Path)
args = parser.parse_args()
def extract(root, name, env):
    n=next(n for n in reversed(root.body) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),name,"exec"),env)
    return env[name]
text=args.source.read_text(encoding="utf8")
submitted=ast.parse(text)
print('SOURCE',hashlib.sha256(text.encode()).hexdigest(),'CUDA',torch.version.cuda,torch.__version__,torch.cuda.get_device_name(),flush=True)
torch.manual_seed(314159)
torch.backends.cuda.matmul.allow_tf32=False
dev=torch.device('cuda:0')
wpool=(torch.randn(4096,8192,device=dev)/8192**.5).bfloat16()
wrouter=torch.randn(256,8192,device=dev)/8192**.5
pool=lambda h:torch.nn.functional.linear(h,wpool)
env={'torch':torch,'D_EXPERT':4096,'HIDDEN':8192,'_pdl':lambda t:False,'GLE_T':{},'bfp_SK2_PDL':0}
env['_bfp_gemm']=lambda st,kind,h,t,n,k:pool(h)
def unexpected(*args):raise AssertionError("unexpected fallback")
env['_g_overlap_parent_front']=unexpected
env['_bucket']=lambda t:t
original=next(n for n in submitted.body if isinstance(n,ast.FunctionDef) and n.name=='_router_fp32_gemv')
exec(compile(ast.Module(body=[original],type_ignores=[]),'original-router','exec'),env)
env['_kfix_router_fp32_gemv']=env['_router_fp32_gemv']
for name in ('_GOverlapStream','_g_overlap_join','_g_overlap_choice','_g_overlap_eligible','_router_fp32_gemv','_pool_router_fp32'):
    extract(submitted,name,env)
base={'_g_overlap_active':True, '_g_overlap_stream':env['_GOverlapStream'](dev),
      'router_fp32':wrouter,'router_fp32_ext':Router(),
      'bfp':{'dummy':torch.empty(1,device=dev),'choice':{('pooler',1):('sk2',),('pooler',8):('sk2',)}},'mm_pick':{(1,'pr'):'bfpf',(8,'pr'):'bfpf'}}

class Bank:
    def __init__(self,m):
        self.m=m
        self.parent=torch.cuda.Stream(device=dev)
        self.state=dict(base)
        self.state['_g_overlap_stream']=copy.deepcopy(base['_g_overlap_stream'])
        self.side=self.state['_g_overlap_stream'].stream
        self.h=torch.randn(m,8192,device=dev).bfloat16()
        self.hr=torch.randn(m,8192,device=dev)
        self.graph=torch.cuda.CUDAGraph()
    def run(self):
        (p,q),marker=env['_pool_router_fp32'](self.h,self.hr,self.m,self.state,'bfpf')
        assert marker==-1
        # A real parent-stream consumer must see both branches.
        return p[:,:256].float()+q,q

banks=[Bank(1),Bank(8)]
assert len({x.cuda_stream for b in banks for x in (b.parent,b.side)})==4
torch.cuda.synchronize()
for b in banks:
    with torch.cuda.stream(b.parent):
        for _ in range(3): b.run()
torch.cuda.synchronize()
sync=Barrier(2)
def capture(b):
    torch.cuda.set_device(dev)
    with torch.cuda.stream(b.parent):
        # Warm thread-local cuBLAS handle before either capture starts.
        b.run()
        b.parent.synchronize()
        sync.wait(timeout=25)
        b.graph.capture_begin(capture_error_mode="thread_local")
        b.output=b.run()
        b.graph.capture_end()
with ThreadPoolExecutor(max_workers=2) as workers:
    list(workers.map(capture,banks))
print("CONCURRENT_CAPTURE_PASS",flush=True)
max_error=0.
for step in range(10):
    expected=[]
    for b in banks:
        hr=torch.randn_like(b.hr)
        h=torch.randn(b.h.size(),device=dev).bfloat16()
        q=(hr.double()@wrouter.double().T).float()
        expected.append((pool(h)[:,:256].float()+q,q))
        current=torch.cuda.current_stream()
        b.parent.wait_stream(current)
        with torch.cuda.stream(b.parent):
            b.hr.copy_(hr);b.h.copy_(h)
            b.graph.replay()
        h.record_stream(b.parent);hr.record_stream(b.parent)
    for b in banks:torch.cuda.current_stream().wait_stream(b.parent)
    for b,ref in zip(banks,expected):
        for out,want in zip(b.output,ref):
            max_error=max(max_error,float((out-want).abs().max()))
            torch.testing.assert_close(out,want,atol=1e-5,rtol=1e-5)
print("FRESH_REPLAY_PASS",20,"max_abs",max_error,flush=True)

# Run the candidate's complete extra startup validator. The only numerical
# substitute is the BF16 pooler. Route/normalization use the real Triton body.
route_node=next(n for n in submitted.body if isinstance(n,ast.FunctionDef) and n.name=='_sl_tile_route')
with tempfile.TemporaryDirectory() as directory:
    routefile=Path(directory)/'okc_route_fixture.py'
    routefile.write_text('import triton\nimport triton.language as tl\n'
        'from triton.language.extra.cuda import gdc_wait as _gdc_wait, gdc_launch_dependents as _gdc_launch\n'
        '@triton.jit\n'+ast.get_source_segment(text,route_node)+'\n',encoding='utf8')
    spec=importlib.util.spec_from_file_location('okc_route_fixture',routefile)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    env['_sl_tile_route']=mod._sl_tile_route
    for name in ('SL_TILE_CONFIG','SL_ROUTE_CONFIG'):
        node=next(n for n in submitted.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)
        env[name]=ast.literal_eval(node.value)
    weights={'router':wrouter,'pooler':wpool,'pre_ffn_norm':.1*torch.randn(8192,device=dev)}
    base['bias']=.05*torch.randn(256,device=dev)
    base['g_preexp']=(.1*torch.randn(4096,device=dev)).bfloat16()
    extract(submitted,'_g_overlap_selftest',env)(torch,weights,base,time.monotonic())
