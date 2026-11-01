"""Actual startup orchestration on Torch CPU; CUDA/native arithmetic stand-ins.

Reference routing/cache/adaptive/finite validators execute unchanged. This does
not execute Hopper kernels, validate GPU capture, or forecast setup/latency.
"""
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

from g_common_release import build
from g_native_patch_ladder import flags

ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'tools'))
import reset_host_facade_test as old


class Graph:
    active=None
    def __init__(self):self.ops=[];self.kind=None
    def replay(self):
        for op in self.ops:op()


class Stream:
    def wait_stream(self,other):pass


class Cuda:
    CUDAGraph=Graph;Stream=Stream
    def __init__(self,stats):self.stats=stats
    def synchronize(self):pass
    def mem_get_info(self):return 100<<30,140<<30
    def current_stream(self):return Stream()
    def current_device(self):return 0
    def get_device_properties(self,device):
        return types.SimpleNamespace(name='CPU_MOCK_NOT_GPU',major=9,minor=0,
            multi_processor_count=132,total_memory=140<<30,uuid='MOCK')
    @contextlib.contextmanager
    def stream(self,stream):yield
    @contextlib.contextmanager
    def graph(self,g):
        assert Graph.active is None
        Graph.active=g
        try:yield
        finally:
            Graph.active=None
            if g.kind is not None:self.stats[g.kind+'_captures']+=1


class Torch:
    Tensor=torch.Tensor
    def __init__(self,stats):self.cuda=Cuda(stats)
    def __getattr__(self,name):return getattr(torch,name)


class Native(types.ModuleType):
    def __init__(self,key,flags):super().__init__(key);self.key=key;self.flags=flags
    def qual_native_resources(self):
        local=8 if self.key=='main' else 16 if self.key=='b4' else 0
        return [96,local,0,231168,576 if self.key in ('main','b4') else 288,1024,1,132,1,132]*2
    def okc_run(self,w,s,x,ids,tw,out,*args):
        op=lambda:out.zero_()
        if Graph.active is not None:
            Graph.active.kind='empty';Graph.active.ops.append(op)
        op()


class Attention:
    def __init__(self):self.active=False;self.rows=[];self.failure=False;self.corrupt=None;self.disabled_calls=0
    def qual_att7_begin(self):
        assert not self.active
        self.rows=[];self.active=True
    def qual_att7_end(self):
        self.active=False;rows=self.rows;self.rows=[];return rows
    def qual_att7_versions(self):return [12080,12080,12,8,93] # mocked identities
    def run(self,data):
        if self.failure:raise RuntimeError('mock native failure caught by model fallback')
        if not self.active:self.disabled_calls+=1;return
        cfg=data['config'];b,l=cfg['batch'],cfg['q_len'];flat=int(l==8 and b>=16)
        splits=2 if b==1 else 1
        row=[1,l,b,flat,1,0,0,splits,1,1,cfg['window'],splits,1 if flat else b,1,
             128,1000,96,0,0,1024,232448,2,132,1,1,100 if flat else 3,0]
        if self.corrupt:self.corrupt(row)
        self.rows+=row


def fixture(source,arm):
    stats=dict(model_captures=0,empty_captures=0,validators=0,prepares=0)
    ns=dict(torch=Torch(stats),time=__import__('time'),math=math,PDL=False,SCRATCH_SENTINEL=0x5eadbeef,
        _GUARDS=[],_GUARDED_IDS=set(),_AB_DEQ={},_check_guards=lambda:None,
        FP8=torch.float32,HIDDEN=4,D_EXPERT=4096,LOCAL_EXPERTS=32,N_EXPERTS=256,TOP_K=8,QKV_ROWS=8)
    ns.update(flags(source));modules={};production=[];audit=Attention()
    def raw_build(defines=None):
        defs=tuple(defines)
        key='mid' if '@moe_helper0df' in defs else 'b4' if '-DOKC_B4FIN=1' in defs else 'main' if '-DOKC_WGS=1' in defs else 'tiny2' if '-DOKC_WSBLK=1' in defs else 'tiny'
        if (key,defs) not in modules:modules[key,defs]=Native(key,defs)
        return modules[key,defs]
    ns.update(_ma_ext=raw_build,_RouterExt=old._RouterExt,
        _main_defines=lambda sfx:ns['MOE_DEFINES'],_att7=lambda:audit)
    tree=ast.parse(source)
    native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_moe')
    exec(ast.get_source_segment(source,native),ns)
    def moe(n=None,s=None):return ns['_fork_native_moe'](n,s)
    def prepare(weights):
        stats['prepares']+=1;s=old.state();s['qkv']=torch.ones(8,4)
        s['_g_preexp_effective']=torch.ones(4,dtype=torch.bfloat16)
        s['att_ok']=True;s['ma_ext']={};s['ma_gamma_effective']=False
        backing=torch.full((64,),ns['SCRATCH_SENTINEL'],dtype=torch.int32)
        view=backing[:16];view.zero_();s['ma_scratch'][0]=view
        ns['_GUARDS'].append((backing,16));ns['_GUARDED_IDS'].add(id(view))
        for n in (1,8,16,32,128):ns['_fork_native_moe'](n,s)
        production.append(s);return s
    def inputs(b,l,context,seed):
        d=old.inputs(b,l,context,seed);n=b*l
        d['config'].update(page_size=1,num_kv_heads=2,window=2048 if context=='sliding' else 0)
        d.update(seq_lens=torch.zeros(b,dtype=torch.int32),page_table=torch.arange(n,dtype=torch.int32).reshape(b,l),
            cache_loc=torch.arange(n,dtype=torch.int64),k_scale=torch.ones(1),v_scale=torch.ones(1),
            k_cache=torch.zeros(n,2,64),v_cache=torch.zeros(n,2,64))
        return d
    def front(d,w,*unused):
        f=old.front(d,w);n=d['config']['tokens'];b,l=d['config']['batch'],d['config']['q_len']
        f.update(ref_h=d['stream']+d['attn_input'],ref_s=d['attn_input']*2,
            k_new=d['attn_input'][:,0,None,None].expand(n,2,64).reshape(b,l,2,64).clone(),
            v_new=d['stream'][:,0,None,None].expand(n,2,64).reshape(b,l,2,64).clone())
        return f
    def attention(d,s,qkv):audit.run(d);return None
    def model(d,s):
        try:ns['_decode_attention7'](d,s,None)
        except RuntimeError:pass # mirrors source's FA3 fallback, deliberately same toy math
        ns['_moe'](d['config']['tokens'],s)
        f=ns['_ab_ref_front'](d,None)
        output=(f['ref_h'].clone(),f['ref_s'].clone(),f['biased'].topk(8,dim=-1).indices.to(torch.int32))
        def update():
            current=ns['_ab_ref_front'](d,None)
            output[0].copy_(current['ref_h']);output[1].copy_(current['ref_s'])
            output[2].copy_(current['biased'].topk(8,dim=-1).indices)
            d['k_cache'].copy_(current['k_new'].reshape_as(d['k_cache']))
            d['v_cache'].copy_(current['v_new'].reshape_as(d['v_cache']))
        update()
        if Graph.active is not None:Graph.active.kind='model';Graph.active.ops.append(update)
        return output
    ns.update(prepare=prepare,kernel_fn=model,_moe=moe,_decode_attention7=attention,
        _ab_inputs=inputs,_ab_ref_front=front,_ab_ref_back=lambda f,w,ids:(f['ref_h'],f['ref_s']),_ab_coeffs=lambda w:(0.,1.))
    # Execute real oracle/cache/routing/poison/restore/finite code; only reference
    # attention/expert arithmetic and CUDA are explicit small CPU stand-ins.
    for name in ('_ab_local','_ab_check_routing','_ab_compare','_qual_oracle','_qual_finite','_qual_restore','_qual_poison','_qual_validate'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(ast.get_source_segment(source,node),ns)
    validator=ns['_qual_oracle']
    def counted(*args):stats['validators']+=1;return validator(*args)
    ns['_qual_oracle']=counted
    exec(source[source.index('def _wl_storage('):],ns)
    return ns,stats,production,audit


class Startup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources={arm:build(ROOT,arm)[0] for arm in ('common','short-rmw')}
    def call(self,ns):
        log=io.StringIO();old_tf32=torch.backends.cuda.matmul.allow_tf32
        old_hooks=tuple(ns[n] for n in ('_ma_ext','_decode_attention7','_moe'))
        fill=torch.Tensor.fill_
        def recorded_fill(t,value):
            if Graph.active is not None:Graph.active.ops.append(lambda:fill(t,value))
            return fill(t,value)
        try:
            with patch.object(torch.Tensor,'fill_',recorded_fill),contextlib.redirect_stdout(log):
                state=ns['prepare']({'attn_residual_gate':torch.ones(1),'ffn_residual_gate':torch.ones(1)})
        finally:
            self.assertEqual(old_tf32,torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(old_hooks,tuple(ns[n] for n in ('_ma_ext','_decode_attention7','_moe')))
            self.assertFalse(ns['_AB_DEQ']);self.assertIsNone(Graph.active)
            self.assertFalse(ns['_att7']().active)
        report=json.loads(next(l.split(' ',1)[1] for l in log.getvalue().splitlines() if l.startswith('G_COMMON_QUAL_DONE ')))
        return state,report
    def test_complete_actual_orchestration_both_fixed_arms(self):
        for arm,source in self.sources.items():
            ns,stats,prod,audit=fixture(source,arm);state,report=self.call(ns)
            self.assertIs(state,prod[0]);self.assertEqual(stats['prepares'],1)
            self.assertEqual((report['cells'],report['checks'],report['captures'],report['fresh_checked']),(12,72,12,36))
            self.assertEqual((report['empty_checks'],report['empty_captures'],report['restored']),(60,12,48))
            self.assertEqual(stats['model_captures'],12);self.assertEqual(stats['empty_captures'],12)
            self.assertEqual(stats['validators'],72)
            self.assertEqual(len(report['actual_attention']),12)
            self.assertEqual(report['rmw_exercised'],arm=='short-rmw')
            self.assertLess(len(json.dumps(report,separators=(',',':'))),11000)
            self.assertEqual(len(ns['_GUARDS']),1)
            d=ns['_ab_inputs'](1,1,'sliding',2);ns['kernel_fn'](d,state)
            self.assertFalse(audit.active);self.assertEqual(audit.rows,[])
    def test_caught_native_failure_and_actual_metadata(self):
        ns,_,_,audit=fixture(self.sources['common'],'common');audit.failure=True
        with self.assertRaisesRegex(AssertionError,'CAUGHT_ATTENTION_FALLBACK'):self.call(ns)
        for change,reason in ((lambda r:r.__setitem__(24,0),'WAITING_LAUNCH'),(lambda r:r.__setitem__(10,44),'FLAGS_WINDOW')):
            ns,_,_,audit=fixture(self.sources['common'],'common');audit.corrupt=change
            with self.assertRaisesRegex(AssertionError,reason):self.call(ns)
    def test_stale_empty_capture_and_model_capture(self):
        ns,_,_,_=fixture(self.sources['common'],'common')
        def no_capture(self,*args):args[5].zero_()
        with patch.object(Native,'okc_run',no_capture):
            with self.assertRaisesRegex(AssertionError,'EMPTY_REPLAY|EMPTY_CAPTURE'):self.call(ns)
        ns,_,_,_=fixture(self.sources['common'],'common');original=Graph.replay
        def stale(graph):
            if graph.kind!='model':original(graph)
        with patch.object(Graph,'replay',stale):
            with self.assertRaisesRegex(AssertionError,'NONFINITE|MODEL_ROUTER_CACHE'):self.call(ns)
    def test_resource_owner_and_budget_failures(self):
        ns,_,_,_=fixture(self.sources['common'],'common')
        with patch.object(Native,'qual_native_resources',return_value=[96,8,0,231168,576,1024,0,132,1,0]*2):
            with self.assertRaisesRegex(AssertionError,'EXPERT_RESOURCE'):self.call(ns)
        ns,_,_,_=fixture(self.sources['common'],'common');clone=ns['_wl_clone']
        def alias(s,ro,on):a=clone(s,ro,on);a['ma_scratch']=s['ma_scratch'];return a
        ns['_wl_clone']=alias
        with self.assertRaisesRegex(AssertionError,'overlap'):self.call(ns)
        ns,_,_,_=fixture(self.sources['common'],'common');ns['time']=types.SimpleNamespace(monotonic=lambda:300 if ns['_GQ_CURRENT'] else 0)
        # Actual inner guard: force deadline after native-capture activity.
        calls=[0]
        def clock():calls[0]+=1;return 300 if calls[0]>18 else 0
        ns['time']=types.SimpleNamespace(monotonic=clock)
        with self.assertRaisesRegex(AssertionError,'PREPARE_BUDGET'):self.call(ns)

    def test_reference_output_cache_and_guard_negatives(self):
        ns,_,_,_=fixture(self.sources['common'],'common');back=ns['_GQ_REF_BACK']
        ns['_GQ_REF_BACK']=lambda *a:(back(*a)[0]*float('nan'),back(*a)[1])
        with self.assertRaisesRegex(AssertionError,'NONFINITE'):self.call(ns)
        for target in ('output','cache'):
            ns,_,_,_=fixture(self.sources['common'],'common');model=ns['_GQ_KERNEL']
            def bad(d,s):
                out=model(d,s)
                (out[0] if target=='output' else d['k_cache']).fill_(float('nan') if target=='output' else 99)
                return out
            ns['_GQ_KERNEL']=bad
            with self.assertRaisesRegex(AssertionError,'NONFINITE|MODEL_ROUTER_CACHE'):self.call(ns)
        ns,_,_,_=fixture(self.sources['common'],'common');clone=ns['_wl_clone']
        def guard(s,ro,on):
            private=clone(s,ro,on)
            raw,words=private['_wl_guard_storage'][0];raw.view(torch.int32)[words]=0
            return private
        ns['_wl_clone']=guard
        with self.assertRaisesRegex(AssertionError,'PRIVATE_GUARD'):self.call(ns)

    def test_fixed_arm_flag_and_fatal_cuda(self):
        ns,_,_,_=fixture(self.sources['short-rmw'],'short-rmw');ns['_GQ_ARM']='common'
        with self.assertRaisesRegex(AssertionError,'RESOURCE_POLICY|EXPERT_POLICY'):self.call(ns)
        ns,_,_,_=fixture(self.sources['common'],'common')
        ns['torch'].cuda.synchronize=lambda:(_ for _ in ()).throw(RuntimeError('fatal CUDA'))
        with self.assertRaisesRegex(RuntimeError,'fatal CUDA'):self.call(ns)


if __name__=='__main__':unittest.main()
