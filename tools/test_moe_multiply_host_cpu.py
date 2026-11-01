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
sys.modules.setdefault('test_moe_multiply_host_cpu',sys.modules[__name__])
TOY='''import torch
from test_moe_multiply_host_cpu import API
NEW=False
def _ext():
 class Ext:
  def multiply_resources(self):return [0,128 if NEW else 140,0,0,150000,1,288,1024,1,96,16,0,228736,1,576,1024]
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
        tree=ast.parse((ROOT/'tools/moe_multiply_check_body.py').read_text(encoding='utf-8'))
        ns={'torch':torch,'time':__import__('time'),'_MD_CONTROL':TOY,'_MD_CANDIDATE':TOY.replace('NEW=False','NEW=True')}
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
        weights=dict(w13=torch.empty((256,1)),w2=torch.empty((256,1)),post_norm_w=(.1*torch.randn(4096,generator=torch.Generator().manual_seed(718))).bfloat16())
        with self.facade(),redirect_stdout(io.StringIO()) as out:
            result=self.namespace()['prepare'](weights)
        self.assertEqual(len(result['states']),4)
        expected=(weights['post_norm_w'].float()+1).bfloat16().float()
        self.assertGreater(torch.count_nonzero(expected!=(weights['post_norm_w'].float()+1)).item(),0)
        for state in result['states']:
            self.assertEqual(state['nw'].dtype,torch.float32)
            self.assertTrue(torch.equal(state['nw'],expected))
        self.assertIn("'normal': 72, 'fresh': 72, 'skew': 24, 'empty': 24, 'post': 72",out.getvalue())
        self.assertIn('MDDONE',out.getvalue())
        self.assertIn('KEEP False',out.getvalue())

    def test_stale_captured_inputs_fail_fresh_gate(self):
        API.stale=True
        weights=dict(w13=torch.empty((256,1)),w2=torch.empty((256,1)),post_norm_w=torch.zeros(4096,dtype=torch.bfloat16))
        prior=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=True
        try:
            with self.facade(),redirect_stdout(io.StringIO()),self.assertRaisesRegex(AssertionError,'MULTIPLY_MATH'):
                self.namespace()['prepare'](weights)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:torch.backends.cuda.matmul.allow_tf32=prior
        API.stale=False

    def test_actual_summary_win_loss_null_and_regress(self):
        fn=self.namespace()['_md_summary']
        def rows(t):return [(m,seed,t) for m in (1,8,16,32,128,256) for seed in (991,997,1009)]
        with redirect_stdout(io.StringIO()):
            self.assertTrue(fn(rows([100,100,98,98]))[2])
            self.assertFalse(fn(rows([100,100,100,100]))[2])
            self.assertFalse(fn(rows([100,100,98,96]))[2])
            self.assertFalse(fn(rows([100,100,102,102]))[2])
            partly=rows([100,100,96,96])
            partly=[(m,s,[100,100,100,100] if s==1009 else t) for m,s,t in partly]
            self.assertFalse(fn(partly)[2]) # pooled win cannot conceal a flat heldout seed


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
