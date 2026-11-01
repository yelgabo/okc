"""Real host harness execution with CPU tensors and explicit graph/clock stubs."""
import ast
from contextlib import contextmanager,redirect_stdout
import io
from pathlib import Path
import unittest
import sys
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]


class Api:
    active=None
    stale=False
    dispatch=[]
    class Event:
        def __init__(self,**kw):pass
        def record(self):pass
        def synchronize(self):pass
        def elapsed_time(self,end):return .1
    class Graph:
        def __init__(self):self.calls=[]
        def replay(self):
            for f in self.calls:f()
    @contextmanager
    def graph(self,g):
        self.active=g
        try:yield
        finally:self.active=None


API=Api()
sys.modules.setdefault('test_moe_cached_band_host_cpu',sys.modules[__name__])
TOY='''import torch
from test_moe_cached_band_host_cpu import API
NEW=False
FLAVOR=0
def _ext():
 class Ext:
  def multiply_resources(self):return [0,128 if FLAVOR==1 else 130,0,0,150000+(1408 if FLAVOR==1 else 0),1,288,1024,1,96,16,0,228736,1,576,1024]
 return Ext()
def _repack(w,lo,n):return {k:torch.zeros((1,36864),dtype=torch.uint8) for k in ('w13','w2','w13f','w2f')}
def _scratch(dev):return {'ctr':torch.zeros(64,dtype=torch.int32)}
def _make_inputs(m,seed,dist):
 g=torch.Generator().manual_seed(seed)
 return dict(hidden_states=torch.randn((m,4096),generator=g).bfloat16(),topk_ids=torch.arange(8,dtype=torch.int32)[None,:].expand(m,8).contiguous(),topk_weights=torch.full((m,8),.125),config={'local_start':0,'local_count':32})
def _ref_forward(d,w):
 scale=(((d['topk_ids']>=0)&(d['topk_ids']<32)).float()*d['topk_weights']).sum(-1,keepdim=True)
 return d['hidden_states'].float()*scale
def _run(d,s,v):
 API.dispatch.append((d['hidden_states'].shape[0],FLAVOR))
 assert v==(0 if d['hidden_states'].shape[0]<=32 else 1)
 out=torch.empty_like(d['hidden_states'])
 frozen=d['hidden_states'].clone() if API.stale and API.active is not None else None
 def run():
  dd=dict(d,hidden_states=frozen) if frozen is not None else d
  out.copy_(_ref_forward(dd,None))
 if API.active is not None:API.active.calls.append(run)
 run()
 return out
'''


class HostTests(unittest.TestCase):
    def namespace(self):
        tree=ast.parse((ROOT/'tools/moe_cached_band_check_body.py').read_text(encoding='utf-8'))
        ns={'torch':torch,'time':__import__('time'),'_BAND_SOURCE_MANIFEST':{'test':True},'_BC_CONTROL':TOY,'_BC_CANDIDATE':TOY.replace('FLAVOR=0','FLAVOR=1'),'_BC_DIRECT':TOY.replace('FLAVOR=0','FLAVOR=2')}
        exec(compile(tree,'<actual-owned-multiply-host-harness>','exec'),ns)
        return ns

    @contextmanager
    def facade(self):
        original_empty=torch.empty
        def empty(*args,**kwargs):
            if args==(64*1024*1024,):args=(1,)
            return original_empty(*args,**kwargs)
        with patch.object(torch.cuda,'synchronize',lambda:None),patch.object(torch.cuda,'CUDAGraph',API.Graph),patch.object(torch.cuda,'graph',API.graph),patch.object(torch.cuda,'Event',API.Event),patch.object(torch,'empty',empty):
            yield

    def test_actual_all18cases_and_summary(self):
        API.stale=False
        API.dispatch=[]
        weights=dict(w13=torch.empty((256,1)),w2=torch.empty((256,1)),post_norm_w=(.1*torch.randn(4096,generator=torch.Generator().manual_seed(718))).bfloat16())
        with self.facade(),redirect_stdout(io.StringIO()) as out:
            result=self.namespace()['prepare'](weights)
        self.assertEqual(len(result['states']),6)
        expected=(weights['post_norm_w'].float()+1).bfloat16().float()
        self.assertGreater(torch.count_nonzero(expected!=(weights['post_norm_w'].float()+1)).item(),0)
        for state in result['states']:
            self.assertEqual(state['nw'].dtype,torch.float32)
            self.assertTrue(torch.equal(state['nw'],expected))
        import json
        report=json.loads(next(l.split(' ',1)[1] for l in out.getvalue().splitlines() if l.startswith('CACHED_BAND32_REPORT ')))
        self.assertEqual(report['checks'],dict(normal=108,fresh=108,skew=36,empty=36,post=108))
        self.assertEqual(len(report['records']),18)
        self.assertLess(len(out.getvalue().encode()),11000)
        self.assertIn('CACHED_BAND32_DONE',out.getvalue())
        self.assertFalse(report['gate']['keep'])
        self.assertTrue(all(m in (16,32) or not new for m,new in API.dispatch))
        self.assertTrue(any(m==16 and new for m,new in API.dispatch))
        self.assertTrue(any(m==32 and new for m,new in API.dispatch))

    def test_stale_captured_inputs_fail_fresh_gate(self):
        API.stale=True
        weights=dict(w13=torch.empty((256,1)),w2=torch.empty((256,1)),post_norm_w=torch.zeros(4096,dtype=torch.bfloat16))
        prior=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=True
        try:
            with self.facade(),redirect_stdout(io.StringIO()),self.assertRaisesRegex(AssertionError,'CACHED_BAND32_MATH'):
                self.namespace()['prepare'](weights)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:torch.backends.cuda.matmul.allow_tf32=prior
        API.stale=False

    def test_nan_reference_is_fatal(self):
        ns=self.namespace()
        with self.assertRaisesRegex(AssertionError,'CACHED_BAND32_NONFINITE'):
            ns['_bc_check'](torch.zeros(2,4),torch.full((2,4),float('nan')))

    def test_inner_budget_restores_tf32(self):
        ns=self.namespace();weights=dict(w13=torch.empty((256,1)),w2=torch.empty((256,1)),post_norm_w=torch.zeros(4096,dtype=torch.bfloat16))
        calls=[0]
        def clock():
            calls[0]+=1
            return 280. if calls[0]>12 else 0.
        prior=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=True
        try:
            with self.facade(),patch('time.monotonic',side_effect=clock),redirect_stdout(io.StringIO()) as out:
                with self.assertRaisesRegex(AssertionError,'CACHED_BAND32_PREPARE_BUDGET'):ns['prepare'](weights)
            self.assertNotIn('CACHED_BAND32_DONE',out.getvalue())
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:torch.backends.cuda.matmul.allow_tf32=prior

    def test_actual_summary_win_loss_null_and_regress(self):
        fn=self.namespace()['_bc_summary']
        def rows(t):return [(m,seed,t) for m in (1,8,16,32,128,256) for seed in (993727,993733,993739)]
        with redirect_stdout(io.StringIO()):
            self.assertTrue(fn(rows([100,100,98,98,120,120]))['keep'])
            self.assertFalse(fn(rows([100,100,100,100,120,120]))['keep'])
            self.assertFalse(fn(rows([100,100,98,96,120,120]))['keep'])
            self.assertFalse(fn(rows([100,100,102,102,120,120]))['keep'])
            partly=rows([100,100,96,96,120,120])
            partly=[(m,s,[100,100,100,100,120,120] if s==993739 else t) for m,s,t in partly]
            self.assertFalse(fn(partly)['keep']) # pooled win cannot conceal a flat heldout seed

    def test_bad_timings_fail_named_before_log(self):
        fn=self.namespace()['_bc_summary']
        for bad in (0.,-1.,float('nan'),float('inf')):
            rows=[(m,s,[100.,100.,98.,98.,120.,120.]) for m in (1,8,16,32,128,256) for s in (993727,993733,993739)]
            rows[0][2][0]=bad
            with self.assertRaisesRegex(AssertionError,'CACHED_BAND32_BAD_TIMING'):fn(rows)

    def test_realistic_transport_with_full_manifest(self):
        import json,random
        artifact=ROOT/'.build/today-proj/owned_cached_band32_check.py'
        tree=ast.parse(artifact.read_text(encoding='utf8'))
        manifest=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_BAND_SOURCE_MANIFEST')
        rng=random.Random(993727)
        rows=[(m,s,[rng.uniform(25,300) for _ in range(6)]) for m in (1,8,16,32,128,256) for s in (993727,993733,993739)]
        report=dict(schema='OWNED_CACHED_BAND32_SIXARM_V1',source=manifest,
                    checks=dict(normal=108,fresh=108,skew=36,empty=36,post=108),worst=[.004123456789,.005123456789],
                    resources=[[0,130,0,0,155520,1,288,288,1,96,8,0,228736,1,576,576]]*6,
                    records=rows,gate=self.namespace()['_bc_summary'](rows),treatment_m=[16,32],seconds=274.123456789)
        encoded='CACHED_BAND32_REPORT '+json.dumps(report,separators=(',',':'),allow_nan=False)+'\nCACHED_BAND32_DONE 18cases_complete CHECK_ONLY no_submit\n'
        self.assertLess(len(encoded.encode()),11000)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
