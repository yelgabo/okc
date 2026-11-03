"""Actual one-arm qualification orchestration with explicit CPU native/CUDA stand-ins."""
import ast,contextlib,importlib.util,io,os,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from hybrid_onearm_cpu_standins import Cuda,inputs,front,back,kernel
from hybrid_packed_source_calls import missing_imports,Ext
import torch.utils.cpp_extension as cpp
ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))
OWNER=Path(os.environ['OKC_QUALIFY_OWNER'])
BODY=ast.literal_eval(next(n.value for n in ast.parse(OWNER.read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='BODY'))
BASE=(ROOT/'tools/dl_verify_hybrid_body.py').read_text()

def weights():return {'w13_sf':torch.full((32,1),0x38,dtype=torch.uint8),'w2_sf':torch.full((32,1),0x38,dtype=torch.uint8),'attn_residual_gate':torch.ones(8192),'ffn_residual_gate':torch.ones(8192)}

def fixture(flags=None):
 flags={} if flags is None else flags;cuda=Cuda();cuda.stale=flags.get('stale',False)
 calls={'G':0,'owned':0,'G_kernel':0,'owned_kernel':0,'empty':0};productions=[]
 def owned_prepare(w):
  calls['owned']+=1
  if flags.get('state_alias') and productions:return productions[0]
  plans={}
  for layer,m in [('unpooler',128),('unpooler',256),('qkv_gate',256)]:
   ws=torch.zeros(16)
   if flags.get('lt_alias') and productions:
    fn=productions[0]['dlplt'][layer,m];ws=dict(zip(fn.__code__.co_varnames[1:fn.__code__.co_argcount],fn.__defaults__))['ws']
   def run(x,plan=1,idx=0,wa=w['w13_sf'],ws=ws,n=8192,m=m):return x
   plans[layer,m]=run
  state={'dlplt':plans,'verify_diag':torch.zeros(4,dtype=torch.int32),'moe1':{'scratch':{'ctr':torch.zeros(64,dtype=torch.int32),'ring':torch.zeros(16)},'w':torch.ones(16)},'one':{'coop_launch':True,'resident':{(8,3):264,(8,51):264}},'nn':torch.nn.Linear(2,2)}
  if flags.get('wd_bad'):state['verify_diag'][0]=1
  if flags.get('guard_bad'):state['moe1']['scratch']['ctr'][8]=1
  productions.append(state);return state
 def run_owned(d,s):
  calls['owned_kernel']+=1
  if flags.get('occupied_after_empty_bad') and s.get('_after_empty',False):flags['output_nan']=True
  s['_after_empty']=False
  return kernel(d,s,cuda,flags)
 def empty(x,w,ids,local,state,variant,pdl):
  calls['empty']+=1
  assert variant==10 and pdl is False and int(local)==0 and torch.equal(ids,torch.arange(32,40,dtype=torch.int32).expand(x.shape[0],8))
  out=torch.empty_like(x)
  def op():
   out.fill_(1 if flags.get('empty_bad') else 0);state['_after_empty']=True
  op()
  if cuda.active is not None:
   def replay():
    if not flags.get('empty_stale'):op()
   cuda.active.ops.append(replay)
  return out
 def run_g(d,s):calls['G_kernel']+=1;return kernel(d,s,cuda,flags)
 moe=[96,0,0,228736,576,1024,1,132,1,132];one=[82,0,0,90704,256,1024,2,132,1,264]*2
 if flags.get('resource_bad'):moe[6]=0
 if flags.get('resource_resident'):one[9]=263
 if flags.get('resource_nan'):moe[0]=float('nan')
 if flags.get('resource_local'):moe[1]=8
 ext=types.SimpleNamespace(qual_moe_resources=lambda:moe,qual_one_resources=lambda:one)
 owned=types.SimpleNamespace(prepare=owned_prepare,kernel_fn=run_owned,torch=torch,_w4_scales_ok=lambda b:bool((b>0).all()),_moe1_ext=lambda:ext,_one_ext=lambda:ext,_moe1_run=empty)
 def gprepare(w):calls['G']+=1;return {'preserved_startup':object(),'scratch':torch.zeros(8)}
 artifact=os.environ.get('OKC_QUALIFY_ARTIFACT')
 if artifact:
  source=Path(artifact).read_text();embedded=[ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').startswith('_VERIFY_HYBRID_SOURCE_')];imports=missing_imports([source,*embedded]);ns={'__name__':'_qual_actual_fixture','__file__':'<actual-qualifier>'}
  # Real module imports/loader/qualifier/oracle; native libraries/builds are stand-ins.
  if 'tvm_ffi.cpp' in imports:imports['tvm_ffi.cpp'].load_inline=lambda *a,**kw:Ext('tvm')
  with patch.dict(sys.modules,imports),patch.object(cpp,'load_inline',lambda *a,**kw:Ext('native')):
   exec(compile(source,'<actual-qualifier>','exec'),ns)
  real_owned=ns['_VERIFY_HYBRID_OWNED'];real_g=ns['_VERIFY_HYBRID_G']
  for key,value in vars(owned).items():
   if key not in ('torch','_w4_scales_ok'):setattr(real_owned,key,value)
  real_g.prepare=gprepare;real_g.kernel_fn=run_g
 else:
  ns=dict(torch=torch,_VERIFY_OVERRIDE_MASK=15,_VERIFY_HYBRID_G=types.SimpleNamespace(prepare=gprepare,kernel_fn=run_g),_VERIFY_HYBRID_OWNED=owned)
  exec(BASE,ns)
  # Real checker/routing helper arithmetic is preserved; mock only model arithmetic.
  bench=(ROOT/'tools/dl_make_bench_pub.py').read_text();raw=next(ast.literal_eval(n.value) for n in ast.parse(bench).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='BODY')
  helpers=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('_ab_local','_ab_check_routing','_ab_compare','_ab_validate')]
  exec(compile(ast.Module(helpers,[]),'<real-oracle-checkers>','exec'),ns)
 import time,math
 ns.update(time=time,math=math,_AB_DEQ={'test':object()},TOP_K=8,N_EXPERTS=256,LOCAL_EXPERTS=32,FP8=torch.float8_e4m3fn,_ab_inputs=inputs,_ab_ref_front=front,_ab_coeffs=lambda w:(0.,1.))
 def ref_back(f,w,ids):
  h,s=back(f,w,ids)
  if flags.get('ref_nan'):h=h.clone();h[0,0]=float('nan')
  return h,s
 if artifact:ns['_QUAL_BACK_ORIGINAL']=ref_back
 else:
  ns['_ab_ref_back']=ref_back;ns['_qual_oracle']=ns['_ab_validate'];exec(BODY,ns)
 return ns,cuda,calls,productions

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def execute(self,flags=None,bank=None):
  ns,cuda,calls,states=fixture(flags);log=io.StringIO();previous=torch.backends.cuda.matmul.allow_tf32
  try:
   with patch.object(torch,'cuda',cuda),contextlib.redirect_stdout(log):result=ns['prepare'](weights() if bank is None else bank)
  finally:
   self.assertEqual(torch.backends.cuda.matmul.allow_tf32,previous);self.assertFalse(ns['_AB_DEQ'])
  return ns,calls,states,result,log.getvalue()
 def test_whole_qualifier(self):
  ns,calls,states,result,log=self.execute();self.assertEqual((calls['G'],calls['owned'],calls['G_kernel'],calls['owned_kernel']),(1,2,0,24));self.assertEqual(len(states),2);self.assertIs(result['owned'],states[0]);self.assertIn('checks=48 captures=8 fresh=24',log);self.assertEqual(log.count('HYBRID_QUAL_RESOURCE'),3);self.assertIn('empty_checks=40 empty_captures=8 restored=32',log);self.assertEqual(calls['empty'],16)
 def test_stale_output_cache_route_reference_fail_closed(self):
  for key,reason in [('stale','QUAL_'),('output_nan','QUAL_NONFINITE'),('cache_bad','QUAL_MODEL_ROUTER_CACHE'),('route_bad','QUAL_MODEL_ROUTER_CACHE'),('ref_nan','QUAL_NONFINITE')]:
   with self.subTest(key=key),self.assertRaisesRegex(AssertionError,reason):self.execute({key:True})
 def test_owners_guards_and_resources(self):
  for key,reason in [('state_alias','QUAL_OWNED_STATE_ALIAS'),('lt_alias','QUAL_LT_ALIAS'),('wd_bad','QUAL_WD_DIAGNOSTIC'),('guard_bad','QUAL_EXPERT_DIAGNOSTIC'),('resource_bad','QUAL_RESOURCE_ADMISSION'),('resource_resident','QUAL_RESOURCE_RESIDENT'),('resource_nan','QUAL_RESOURCE_VALUES'),('resource_local','QUAL_RESOURCE_LOCAL')]:
   with self.subTest(key=key),self.assertRaisesRegex(AssertionError,reason):self.execute({key:True})
 def test_empty_and_restore_fail_closed(self):
  for key,reason in [('empty_bad','QUAL_EMPTY_OUTPUT'),('empty_stale','QUAL_EMPTY_OUTPUT'),('occupied_after_empty_bad','QUAL_NONFINITE')]:
   with self.subTest(key=key),self.assertRaisesRegex(AssertionError,reason):self.execute({key:True})
 def test_budget_and_sync_errors(self):
  for expire in (2,12):
   ns,cuda,_,_=fixture();ticks=[0]
   def monotonic():ticks[0]+=1;return 296. if ticks[0]>=expire else 0.
   ns['time']=types.SimpleNamespace(monotonic=monotonic)
   with patch.object(torch,'cuda',cuda),self.assertRaisesRegex(AssertionError,'QUAL_PREPARE_BUDGET'):ns['prepare'](weights())
  ns,cuda,_,_=fixture();cuda.sync_fail=True
  with patch.object(torch,'cuda',cuda),self.assertRaisesRegex(RuntimeError,'CUDA sync failure'):ns['prepare'](weights())
 def test_unsupported_scales_preserve_only_g(self):
  bank=weights();bank['w13_sf'].zero_();ns,calls,states,result,log=self.execute(bank=bank);self.assertEqual(set(result),{'G'});self.assertEqual((calls['G'],calls['owned'],len(states)),(1,0,0));self.assertIn('fallback=unsupported_weights checks=0',log)
if __name__=='__main__':unittest.main()
