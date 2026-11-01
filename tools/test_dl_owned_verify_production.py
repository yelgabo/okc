import ast,builtins,hashlib,symtable,unittest,os
from pathlib import Path
from types import SimpleNamespace
from dl_owned_verify_production import build,literal,function,PIN,frozen_source

ROOT=Path(os.environ.get('OKC_HYBRID_ROOT',Path(__file__).resolve().parents[1]))

class Production(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.outputs={bs:build(ROOT,bs) for bs in ((16,),(32,),(16,32))}
  cls.old=frozen_source(ROOT.resolve())
 def test_sources_globals_and_visible_literals(self):
  for bs,(s,m) in self.outputs.items():
   self.assertEqual(m['input_pin'],PIN)
   t=symtable.symtable(s,'production','exec');available=set(dir(builtins))|{v.get_name() for v in t.get_symbols() if v.is_imported() or v.is_assigned()}
   def used(scope):return {v.get_name() for v in scope.get_symbols() if v.is_global()}.union(*(used(c) for c in scope.get_children()))
   self.assertEqual(set().union(*(used(c) for c in t.get_children()))-available,set(),bs)
   for n in ast.parse(s).body:
    if isinstance(n,ast.FunctionDef):self.assertFalse(n.decorator_list,'closure must not silently drop decorators')
   for k in ('_ONE_SRC','_MOE1_SRC'):self.assertEqual(literal(s,k),literal(self.old,k))
   self.assertNotIn('_SKP_GEMV_SRC',s)
   if bs==(32,):self.assertNotIn('_SKP_WD_SRC',s)
   self.assertIn('r\'\'\'\n#include',s)

 def test_selected_device_bodies_exact(self):
  s=self.outputs[(16,32)][0]
  for name,anchors in (
   ('_CUDA_SRC',['__device__ __forceinline__ float round_bf16(','__global__ void __launch_bounds__(NORM_THREADS) route_kernel(','std::vector<torch::Tensor> route_fwd(']),
   ('_BND_SRC',['__device__ __forceinline__ float4 bf16x4(','__device__ __forceinline__ float bf16r(','__device__ __forceinline__ float block_sum_n(','__global__ void __launch_bounds__(NT) attn_boundary_geo_kernel(']),
   ('_SKP_SRC',['__device__ __forceinline__ bool mbar_wait(','__device__ __forceinline__ void fence_proxy_async(','PFN_cuTensorMapEncodeTiled_v12000 encode_fn('])):
   for a in anchors:self.assertEqual(function(literal(s,name),a),function(literal(self.old,name),a),(name,a))
  self.assertEqual(literal(s,'_SKP_WD_SRC'),literal(self.old,'_SKP_WD_SRC'))

 def test_projection_pins_and_selected_policies(self):
  class X:
   dtype='bf16'
   def __init__(self,m):self.shape=(m,8192)
   def contiguous(self):return self
   def is_contiguous(self):return True
   def data_ptr(self):return 256
  for bs,(s,m) in self.outputs.items():
   events=[]
   def log(v):return lambda *args:events.append(v)
   n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_slice_proj')
   ns={'torch':SimpleNamespace(bfloat16='bf16'),'_gemm':log('torch')}
   exec(compile(ast.Module([n],[]),'<actual-proj>','exec'),ns)
   st={'dense':{'qkv_gate':None,'o':None},'verify_qkv128':log('WD128'),'dlplt':{('qkv_gate',256):log('Lt256')}}
   for b in bs:
    ns['_slice_proj'](X(b*8),'qkv_gate',st);ns['_slice_proj'](X(b*8),'o',st)
   self.assertEqual(events,sum(([('WD128' if b==16 else 'Lt256'),'torch'] for b in bs),[]))
   lt=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_slice_lt')
   loop=next(n for n in ast.walk(lt) if isinstance(n,ast.For));pins=ast.literal_eval(loop.iter)
   self.assertEqual({v[1] for v in pins},{b*8 for b in bs})
   self.assertIn('case 10: launch_main<51>',s)

if __name__=='__main__':unittest.main()
