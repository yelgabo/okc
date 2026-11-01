"""Independent real-TorchCPU execution of the actual VP2 host scaffold.

CUDA modules/graphs and full attention math are explicit stand-ins. Actual
checker/score/prepare code is executed; this is not a GPU correctness result.
"""
import ast
from contextlib import contextmanager,redirect_stdout
import io
import json
import math
from pathlib import Path
import statistics
import time
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
MOE=ROOT.parent/'okc-today-moe'
ARTIFACT=MOE/'.build/today-moe/vp2_bf16_private.py'


def literals():
    t=ast.parse(ARTIFACT.read_text(encoding='utf8'))
    names={'_VP_OWNED','_VP_MANIFEST','_VP_OFFSETS'}
    return {n.targets[0].id:ast.literal_eval(n.value) for n in t.body
            if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in names}


class CUDA:
    active=None
    stale=False
    calls=[]
    registry={}
    class Graph:
        def __init__(self):self.calls=[]
        def replay(self):
            for f in self.calls:f()
    CUDAGraph=Graph
    class Event:
        def __init__(self,**kwargs):pass
        def record(self):pass
        def synchronize(self):pass
        def elapsed_time(self,end):return .1
    @contextmanager
    def graph(self,g):
        self.active=g
        try:yield
        finally:self.active=None
    def synchronize(self):pass
    def current_device(self):return 0
    def current_stream(self):return type('Stream',(),{'cuda_stream':0})()


API=CUDA()


def values(q,b,l):
    q=q.reshape(b,l,-1)
    out=torch.sin(q[...,:1024].float()).bfloat16()
    kn=q[...,1024:1152].float().tanh().reshape(b,l,2,64)
    vn=q[...,1152:1280].float().tanh().reshape(b,l,2,64)
    return out,kn,vn


def execute(q,p,seq,kc,vc,out,b,l,kind):
    frozen=q.clone() if API.stale and API.active is not None else None
    def call():
        oq,kn,vn=values(frozen if frozen is not None else q,b,l)
        out.copy_(oq)
        pos=seq.long()[:,None]+torch.arange(l)
        slots=torch.gather(p.long(),1,pos)
        kc.view(-1,2,64)[slots]=kn;vc.view(-1,2,64)[slots]=vn
        API.calls.append((b,l,kind,q.untyped_storage().data_ptr()))
    if API.active is not None:API.active.calls.append(call)
    call()


class Module:
    def __init__(self,kind):self.kind=kind
    def vp_owned_resources(self):
        return [v for l in (1,8) for m in (1,3,7,9,11,13,15,17,19,49,51)
                for v in (l,m,80,0,150000,256,2)]
    def vp2_resources(self):return [128,0,0,170000,256,1,2]
    def attn_one(self,*args):
        q,p,seq=args[:3];kc,vc=args[7:9];b,l=args[12:14]
        out=torch.empty((b,l,1024),dtype=torch.bfloat16)
        execute(q,p,seq,kc,vc,out,b,l,'owned')
        return out
    def muse_vp2_launch(self,*args):
        q,p,seq,kc,vc,out=[API.registry[args[i]] for i in (0,1,3,5,6,8)]
        execute(q,p,seq,kc,vc,out,1,8,'vp2')


class TorchFacade:
    cuda=API
    def __getattr__(self,name):return getattr(torch,name)
    def device(self,*args):return torch.device('cpu')
    def empty(self,*args,**kwargs):
        if args==(256<<20,):args=(1,)
        kwargs['device']='cpu';return torch.empty(*args,**kwargs)
    def randn(self,*args,**kwargs):
        kwargs['device']='cpu';return torch.randn(*args,**kwargs)
    def Generator(self,**kwargs):return torch.Generator(device='cpu')


def inputs(batch,ctx,head,layer,length,spread,dist,page,scale,seed):
    # Small cache allocation; actual input RNG/native attention is not simulated.
    g=torch.Generator().manual_seed(seed)
    c=dict(batch=batch,q_len=length,page_size=page,window=2048 if layer=='sliding' else 0,
           num_q_heads=16,num_kv_heads=2,head_dim=64,kv_scale=scale,
           rope_theta=0.,sm_scale=.125,qk_scale_factor=1.)
    width=max(2060,ctx+length+1) if ctx<3000 else 2060
    q=torch.randn((batch,length,2304),generator=g).bfloat16()
    seq=torch.full((batch,),min(ctx,8),dtype=torch.int32)
    table=torch.arange(batch*width,dtype=torch.int32).reshape(batch,width)
    return dict(config=c,qkv=q,page_table=table,seq_lens=seq,
                cache_loc=torch.gather(table,1,(seq[:,None]+torch.arange(length,dtype=torch.int32)).long()),
                k_cache=torch.zeros(batch*width,2,64),v_cache=torch.zeros(batch*width,2,64),
                k_scale=torch.ones(1),v_scale=torch.ones(1))


def refs(data):
    c=data['config'];out,kn,vn=values(data['qkv'],c['batch'],c['q_len'])
    return out,out.clone(),kn,vn,.03


def owned_toy(source):
    tree=ast.parse(source)
    funcs={'_one_ok','_one_plan','_one','kernel_fn'}
    consts={'_ONE_ROUTE','_ONE_FLAT','_ONE_Q8_PROD','_ONE_PROD','_ONE_TARGET','_ONE_WS_ROWS','_ONE_BUILT'}
    pieces=[ast.get_source_segment(source,n) for n in tree.body if
            isinstance(n,ast.FunctionDef) and n.name in funcs or
            isinstance(n,ast.Assign) and any(getattr(t,'id','') in consts for t in n.targets)]
    return '''from test_vp2_private_host_cpu import torch,Module
_ONE=None
_ONE_CPP=''
_ONE_SRC=''
_ONE_WS_PICK=0
_ONE_STATE={}
_ONE_DBG=None
LOG2E=1.4426950408889634
def _one_state(device):
 if device not in _ONE_STATE:_ONE_STATE[device]={'resident':{(l,m):264 for l in (1,8) for m in (1,3,7,9,11,13,15,17,19,49,51)},'acc':torch.zeros(8),'l':torch.zeros(8),'cnt':torch.zeros(8,dtype=torch.int32),'coop_launch':True}
 return _ONE_STATE[device]
def _inv_freq(*args):return torch.ones(32)
def _no_dbg(*args):return torch.empty(0,dtype=torch.int32)
def _fused_ok(c):return False
def _ws_ok(c):return False
_PREP=None
def _sglang(data):return 'sglang-fallback'
'''+'\n'.join(pieces)


def namespace():
    pins=literals();body=(MOE/'tools/moe_attn_vp2_body.py').read_text(encoding='utf8')
    tree=ast.parse(body)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Tuple) for t in n.targets))]
    ns=dict(torch=TorchFacade(),time=time,math=math,statistics=statistics,
            concurrent=__import__('concurrent.futures'),_VP_OWNED=owned_toy(pins['_VP_OWNED']),
            _VP_CPP='',_VP_CUDA='',_VP_MANIFEST=pins['_VP_MANIFEST'],_VP_OFFSETS=pins['_VP_OFFSETS'])
    exec(compile(tree,'<actual-VP2-host>','exec'),ns)
    ns['_ob_inputs']=inputs;ns['_vp_refs']=refs
    ns['_ob_rel']=lambda a,b:float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
    return ns


class HostTests(unittest.TestCase):
    @contextmanager
    def facade(self):
        original=torch.Tensor.data_ptr
        def pointer(t):
            p=original(t);API.registry[p]=t;return p
        def build(name,**kwargs):return Module('vp2' if 'foreign' in name else 'owned')
        with patch('torch.utils.cpp_extension.load_inline',build),patch.object(torch.Tensor,'data_ptr',pointer):yield
    def setUp(self):
        torch.set_num_threads(1);API.registry={};API.calls=[];API.stale=False;API.active=None
    def test_whole_prepare_36rows_owners_modes_and_transport(self):
        ns=namespace();prior=torch.backends.cuda.matmul.allow_tf32
        with self.facade(),redirect_stdout(io.StringIO()) as log:owners,modules=ns['_vp_prepare']()
        report=json.loads(next(l.split(' ',1)[1] for l in log.getvalue().splitlines() if l.startswith('VP2_PRIVATE_REPORT ')))
        self.assertEqual(report['counts'],dict(eager_extra=16,normal=144,fresh=336,post=144))
        self.assertEqual(len(report['rows']),36);self.assertEqual(len(report['plans']),48)
        self.assertLess(len(log.getvalue().encode()),11000)
        self.assertTrue(all(len(n['_ONE_STATE'])==1 for n in owners))
        self.assertTrue(all(b==1 and l==8 for b,l,k,_ in API.calls if k=='vp2'))
        self.assertTrue(any(k=='vp2' for b,l,k,_ in API.calls))
        d=inputs(1,8,'tp8','full',1,'fixed','soft',1,1,1);d['config']['num_q_heads']=24
        self.assertEqual(owners[0]['kernel_fn'](d),'sglang-fallback')
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,prior)
    def test_stale_capture_is_fatal_and_restores_tf32(self):
        ns=namespace();API.stale=True;prior=torch.backends.cuda.matmul.allow_tf32
        with self.facade(),redirect_stdout(io.StringIO()),self.assertRaisesRegex(AssertionError,'VP2_MATH'):ns['_vp_prepare']()
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,prior)
    def test_nonunit_scale_delegates_to_actual_owned_path(self):
        ns=namespace()
        def scaled(*args,**kwargs):
            data=inputs(*args,**kwargs);data['config']['kv_scale']=.5;return data
        ns['_ob_inputs']=scaled
        with self.facade(),redirect_stdout(io.StringIO()):ns['_vp_prepare']()
        self.assertTrue(API.calls)
        self.assertFalse(any(k=='vp2' for b,l,k,_ in API.calls))
    def test_real_small_cpu_oracles_and_actual_checker(self):
        source=ARTIFACT.read_text(encoding='utf8');tree=ast.parse(source)
        names={'_ob_rms','_ob_rope','_ob_reference','_ob_rel','_vp_bf_reference','_vp_refs','_vp_check'}
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        ns=dict(torch=torch,math=math,FP8=torch.float8_e4m3fn,RMS_EPS=1e-5,_OB_THETA=500000.)
        exec(compile(tree,'<actual-real-CPU-oracles>','exec'),ns)
        data=inputs(1,8,'tp8','sliding',8,'fixed','soft',1,1.,37)
        rr=ns['_vp_refs'](data)
        slots=data['cache_loc'].long()
        data['k_cache'][slots]=rr[2].float();data['v_cache'][slots]=rr[3].float()
        errors=ns['_vp_check'](data,rr[1],rr)
        self.assertTrue(all(math.isfinite(x) for x in errors))
        self.assertEqual(errors[2:],[0.,0.,0.,0.])
        self.assertGreater(float((rr[0].float()-rr[1].float()).abs().max()),0.)
    def test_actual_checker_finite_output_reference_cache(self):
        ns=namespace();d=inputs(1,0,'tp8','sliding',8,'fixed','soft',1,1,4);rr=refs(d)
        execute(d['qkv'],d['page_table'],d['seq_lens'],d['k_cache'],d['v_cache'],rr[0].clone(),1,8,'test')
        self.assertEqual(ns['_vp_check'](d,rr[0],rr),[0.]*6)
        with self.assertRaisesRegex(AssertionError,'VP2_OUTPUT_NONFINITE'):ns['_vp_check'](d,torch.full_like(rr[0],float('nan')),rr)
        with self.assertRaisesRegex(AssertionError,'VP2_REFERENCE_NONFINITE'):ns['_vp_check'](d,rr[0],(torch.full_like(rr[0],float('nan')),*rr[1:]))
        d['k_cache'].fill_(17)
        with self.assertRaisesRegex(AssertionError,'VP2_CACHE'):ns['_vp_check'](d,rr[0],rr)
    def test_score_winner_loser_null_regression_and_bad_timing(self):
        ns=namespace();fn=ns['_vp_score'];seeds=ns['_VP_MANIFEST']['seeds']
        def rows(t):return [(b,l,w,s,t) for b in (1,16,32) for l in (1,8) for w in (0,1) for s in seeds]
        self.assertTrue(fn(rows([100,100,98,98]))['keep'])
        for t in ([100,100,100,100],[100,100,98,96],[100,100,101,101]):self.assertFalse(fn(rows(t))['keep'])
        for bad in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaisesRegex(AssertionError,'VP2_BAD_TIMING'):fn(rows([bad,100,98,98]))
    def test_budget_and_calibration_fail_closed(self):
        ns=namespace();ns['_VP_BEGIN']=-1000
        with self.facade(),self.assertRaisesRegex(AssertionError,'VP2_PRIVATE_BUDGET'):ns['_vp_prepare']()
        ns=namespace();ns['_VP_OFFSETS']={k:101. for k in ns['_VP_OFFSETS']}
        with self.facade(),redirect_stdout(io.StringIO()),self.assertRaisesRegex(AssertionError,'VP2_BAD_CALIBRATION'):ns['_vp_prepare']()


if __name__=='__main__':unittest.main()
