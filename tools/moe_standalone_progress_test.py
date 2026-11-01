"""CPU/source tests only. This OFF prototype fails its local no-new-spills gate."""
import ast,os,subprocess,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from moe_standalone_progress import build,core,PIN,sha,function
ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))
RAW=subprocess.check_output(['git','-C',str(ROOT),'show','ec565a7:muse_moe_w4a16/submission.py']).decode()
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.source,cls.manifest=build(ROOT,RAW)
 def env(self):
  t=ast.parse(self.source);nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_spg_') or isinstance(n,ast.Assign) and any(getattr(x,'id','').startswith('_SPG_') for x in n.targets)]
  import time
  ns=dict(torch=torch,time=time,_REF_CACHE={},SKINNY_VARIANTS=((8,4,0),(16,4,0),(8,4,1),(8,3,0)),MAIN_MODES=(0,16,26,28,37,47),VARIANT_MAIN0=4,MAX_LOCAL=32)
  exec(compile(ast.Module(nodes,[]),'<actual-prospective-host>','exec'),ns);return ns
 def test_source_scope_and_public_off(self):
  self.assertEqual(sha(RAW),PIN);c,m=core(RAW,ROOT);self.assertTrue(m['native_delta_only_two_launchers_and_watchdog'])
  ns={};nodes=[n for n in ast.parse(self.source).body if isinstance(n,ast.Assign) and any(getattr(x,'id','').startswith('_SPG_') for x in n.targets)]
  nodes+=[n for n in ast.parse(self.source).body if isinstance(n,ast.FunctionDef) and n.name=='prepare'][-1:];exec(compile(ast.Module(nodes,[]),'<off>','exec'),ns)
  for flag,msg in [(None,'SPG_OFF'),('_SPG_ENABLED','SPG_STATIC'),('_SPG_STATIC_GATE_PASSED','SPG_FRESH')]:
   if flag:ns[flag]=True
   with self.assertRaisesRegex(AssertionError,msg):ns['prepare']({})
  with self.assertRaises(AssertionError):build(ROOT,RAW+'\n')
 def test_strict_math_finite_rows_and_empty(self):
  ns=self.env();r=torch.ones(256,16);o=r.bfloat16();self.assertEqual(ns['_spg_check'](o,r),(0.,0.));bad=o.clone();bad[0]=1.078125
  with self.assertRaisesRegex(AssertionError,'SPG_FP32'):ns['_spg_check'](bad,r)
  for target in (0,1):
   a,b=o.clone(),r.clone();(a if target==0 else b)[0,0]=float('nan')
   with self.assertRaisesRegex(AssertionError,'SPG_NONFINITE'):ns['_spg_check'](a,b)
  self.assertEqual(ns['_spg_check'](torch.zeros_like(o),torch.zeros_like(r)),(0.,0.))
 def test_fixed_prepare_private_scratch_and_boolean(self):
  ns=self.env();calls=[];scratches=[]
  def scratch(dev):
   s={'ctr':torch.zeros(64,dtype=torch.int32)};scratches.append(s);return s
  ns.update(_repack=lambda *a:{'w13':torch.zeros(1)},_scratch=scratch,_ext=lambda:calls.append('build'),_bench=lambda *a:(_ for _ in ()).throw(AssertionError('tuning forbidden')))
  def selftest(state,weights,sizes,label):
   self.assertEqual(state['policy'],{1:3,8:3,16:3,32:3,128:9,256:9});self.assertEqual(sizes[-2:],(512,513));calls.append('strict_selected');return True
  ns['_self_test']=selftest;w={'w13':torch.zeros(1),'post_norm_w':torch.zeros(16,dtype=torch.bfloat16)};old=torch.backends.cuda.matmul.allow_tf32
  with patch.object(torch.cuda,'synchronize',lambda:None):state=ns['_spg_prepare_candidate'](w)
  self.assertEqual(calls,['build','strict_selected']);self.assertIs(state['scratch'],scratches[0]);self.assertIsNot(scratches[0],scratches[1]);self.assertEqual(torch.backends.cuda.matmul.allow_tf32,old);self.assertFalse(ns['_REF_CACHE'])
  ns['_self_test']=lambda *a:False
  with patch.object(torch.cuda,'synchronize',lambda:None),self.assertRaisesRegex(AssertionError,'SPG_SELECTED_SELFTEST_FALSE'):ns['_spg_prepare_candidate'](w)
 def test_actual_selected_selftest_diagnostics(self):
  ns=self.env();ns.update(_make_inputs=lambda bs,*a:{'x':torch.ones(bs,16)},_ref_forward=lambda d,w:d['x'],_run=lambda d,s,v:d['x'].bfloat16())
  state={'scratch':{'ctr':torch.zeros(64,dtype=torch.int32)}}
  with patch.object(torch.cuda,'synchronize',lambda:None):self.assertTrue(ns['_spg_selected_selftest'](state,{},(1,4),'cpu'))
  state['scratch']['ctr'][8]=4
  with patch.object(torch.cuda,'synchronize',lambda:None),self.assertRaisesRegex(AssertionError,'SPG_POLL'):ns['_spg_selected_selftest'](state,{},(1,),'cpu')
  self.assertEqual(int(state['scratch']['ctr'][8]),4,'diagnostic may not silently reset')
 def test_per_chunk_scope_and_budget(self):
  ns=self.env();dev=types.SimpleNamespace(type='cuda');tensor=lambda shape,dtype:types.SimpleNamespace(shape=shape,ndim=len(shape),dtype=dtype,device=dev,is_floating_point=lambda:True)
  d={'config':dict(bs=513,d_hidden=4096,d_expert=4096,n_experts=256,local_start=0,local_count=32,top_k=8,sharding='ep8'),'hidden_states':tensor((513,4096),torch.bfloat16),'topk_ids':tensor((513,8),torch.int64),'topk_weights':tensor((513,8),torch.float32)}
  state={'w':{'w13':tensor((1,),torch.uint8)}};ns['_spg_input_scope'](d,state);d['config']['local_start']=32
  with self.assertRaisesRegex(AssertionError,'SPG_UNSUPPORTED'):ns['_spg_input_scope'](d,state)
  ns['time']=types.SimpleNamespace(monotonic=lambda:296.)
  with self.assertRaisesRegex(AssertionError,'SPG_PREPARE_BUDGET'):ns['_spg_budget'](0.)
if __name__=='__main__':unittest.main()
