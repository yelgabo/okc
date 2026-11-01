"""Actual prepare orchestration with real TorchCPU and explicit native stand-ins."""
import ast,contextlib,copy,importlib.util,io,time,types,unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch
from proj_stage_xor_release import build,audited_helpers
from proj_stage_xor_release_body import BODY

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT.parent/'okc'
HELPER=(ROOT.parent/'okc-today-attn/tools/proj_make_bench_sk_xor_body.py').read_text(encoding='utf8')

@lru_cache(maxsize=1)
def source():return build(MAIN,HELPER)


class Fixture:
    def __init__(self):
        torch.set_num_threads(1);torch.manual_seed(18)
        self.builds=0;self.prepares=0;self.stale=False;self.resource_bad=False;self.diag_bad=False;self.too_large=False;self.runner_calls=0
        self.reject=set();self.graphs=[];self.last_state=None
        self.weights={k:torch.randn(4,65).bfloat16() for k in ('qkv_gate','o','pooler','unpooler')}
        self.weights['router']=torch.randn(4,65)
        def normal(weights):
            self.prepares+=1
            self.last_state={**weights,'pick':{(layer,m):(lambda x,key=(layer,m):key)
                for layer in self.weights for m in range(1,257)}}
            self.originals=dict(self.last_state['pick']);return self.last_state
        self.ns=dict(torch=torch,time=time,prepare=normal,_SXSR_PINS={'test':True},SELFTEST_TOL={'bf16':.005},PROBE_TOL=.0005)
        exec(audited_helpers(HELPER)+'\n'+BODY,self.ns)
        original_kernel=next(n for n in ast.parse(source()[0]).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
        self.ns.update(_cublas=lambda x,w:'cublas_fallback',_router_baseline=lambda x,s:'router_fallback')
        exec(compile(ast.Module([original_kernel],[]),'actual-C2-timed-entry','exec'),self.ns)
        configs=[];indices={};metadata={}
        for (layer,m),name in self.ns['_SX_TARGET'].items():
            upw=4 if name=='qkv8F' else 1 if name=='slice8F' else 2
            indices[name]=len(configs);configs.append((name,1,m,1,1,1,upw,4,True,False,0,1,'bf16',False))
            metadata[name]={'original_cfg':configs[-1],'geom':dict(NW=1,R=8,UPW=upw,NT=1,WSK=1,UPS=upw,comp=True,hl=False,cl=1,fuse=False)}
        self.ns.update(SK_CONFIGS=configs,SK_INDEX=indices,_XOR_CONFIG=metadata)
        tree=ast.parse(source()[0]);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('xor_expected_layout','sxs_expected_layout')]
        exec(compile(ast.Module(nodes,[]),'actual-stage-layout','exec'),self.ns)
        actual_layout=self.ns['sxs_expected_layout']
        def expected(name,v,pmax):
            result=actual_layout(name,v,pmax)
            if self.too_large and v==2:result=dict(result,dynamic=227*1024+1)
            return result
        self.ns['sxs_expected_layout']=expected
        def build_module(tag):self.builds+=1;return types.SimpleNamespace(__name__=tag)
        self.ns['sxs_build']=build_module
        def pack(name,w,v):
            if v and id(w) in self.reject:return None
            return dict(name=name,variant=v,immutable=[torch.ones(2),torch.ones(3)],n=4,k=65,pmax=1,table=torch.arange(4,dtype=torch.int32),stage_record_bytes=16)
        self.ns['sxs_pack']=pack
        def runner(module,name,w,v,nearby=False,packed=None):
            self.runner_calls+=1
            def fn(x):return (x.double()@w.double().T).bfloat16()
            layout=self.ns['sxs_expected_layout'](name,v,1)
            diag=torch.zeros(4,dtype=torch.int32)
            if self.diag_bad and v:diag[0]=1
            resource=[32,0,0,layout['dynamic'],layout['threads'],layout['threads'],1 if self.resource_bad and v else 2]
            return fn,dict(resources=resource,variant=v,pack=packed,private_diag=diag)
        self.ns['sxs_runner']=runner
        fixture=self
        class Graph:
            def __init__(self,fn,x):
                self.fn=fn;self.x=x;self.old=x.clone();self.out=fn(x);self.replays=0
            def replay(self):
                self.replays+=1;self.out.copy_(self.fn(self.old if fixture.stale else self.x))
        def capture(fn,x):
            graph=Graph(fn,x);self.graphs.append(graph);return graph,graph.out,object()
        self.ns['_sx_capture']=capture

    def run(self):
        with patch.object(torch,'cuda',types.SimpleNamespace(synchronize=lambda:None)),contextlib.redirect_stdout(io.StringIO()) as output:
            state=self.ns['prepare'](self.weights)
        return state,output.getvalue()


class Release(unittest.TestCase):
    def test_whole_prepare_and_all_M1_33_policy(self):
        f=Fixture();state,log=f.run();report=state['stage_xor_release']
        self.assertEqual((f.prepares,f.builds),(1,1))
        self.assertEqual((report['overrides'],report['installed_pick_keys'],report['nearby_pick_keys']),(10,110,100))
        self.assertEqual((report['math_checks'],report['captures'],report['fresh_checks']),(160,10,30))
        self.assertTrue(all(g.replays==3 for g in f.graphs))
        self.assertIn('SXSRELEASE_DONE overrides=10 fallbacks=0 pick_keys=110',log)
        self.assertNotIn('samples_us',log);self.assertNotIn('median',log)
        for layer in f.weights:
            for m in range(1,257):
                expected=(layer in ('qkv_gate','unpooler') and 2<=m<=32 or layer in ('o','pooler') and 9<=m<=32)
                self.assertEqual(state['pick'][layer,m] is not f.originals[layer,m],expected,(layer,m))
        for row in report['routes']:self.assertEqual(len(row['resources']),4)
        for layer in f.weights:
            for m in set(range(1,34))|{128,256,257}:
                x=torch.ones(m,65,dtype=torch.bfloat16)
                out=f.ns['kernel_fn']({'x':x,'config':dict(layer=layer,m=m,n=4,k=65)},state)
                if m==257:self.assertEqual(out,'router_fallback' if layer=='router' else 'cublas_fallback')
                elif state['pick'][layer,m] is f.originals[layer,m]:self.assertEqual(out,(layer,m))
                else:self.assertTrue(torch.equal(out,(x.double()@f.weights[layer].double().T).bfloat16()))
        # Every official cell outside the fixed10 retains the exact callable.
        for layer in f.weights:
            for m in (1,8,16,32,128,256):
                if (layer,m) not in f.ns['_SX_TARGET']:self.assertIs(state['pick'][layer,m],f.originals[layer,m])

    def test_unsupported_pack_explicit_incumbent_fallback(self):
        f=Fixture();f.reject={id(w) for w in f.weights.values()};state,log=f.run()
        self.assertEqual(state['pick'],f.originals)
        self.assertEqual(state['stage_xor_release']['overrides'],0)
        self.assertEqual(state['stage_xor_release']['fallbacks'],10)
        self.assertIn('overrides=0 fallbacks=10 pick_keys=0',log)
        f=Fixture();f.reject={id(f.weights['pooler'])};state,log=f.run()
        self.assertEqual(state['stage_xor_release']['overrides'],8)
        self.assertTrue(all(state['pick']['pooler',m] is f.originals['pooler',m] for m in range(1,257)))

    def test_math_stale_resource_diag_fail_before_publication(self):
        for field,label in (('stale','SX official math'),('resource_bad','SXSRELEASE_ADMISSION'),('diag_bad','SXSRELEASE_DEVICE_DIAGNOSTIC')):
            f=Fixture();setattr(f,field,True)
            with self.assertRaisesRegex(AssertionError,label):f.run()
            self.assertEqual(f.last_state['pick'],f.originals)

    def test_finite_and_deadline_negative(self):
        f=Fixture();x=torch.ones(2,65,dtype=torch.bfloat16);ref=torch.ones(2,4,dtype=torch.bfloat16)
        with self.assertRaisesRegex(AssertionError,'REFERENCE_NONFINITE'):
            f.ns['_sxsr_math'](ref,torch.full_like(ref,float('nan')),x)
        f=Fixture();f.ns['_sx_guard']=lambda start:(_ for _ in ()).throw(AssertionError('expired'))
        with self.assertRaisesRegex(AssertionError,'expired'):f.run()
        self.assertEqual(f.prepares,1);self.assertEqual(f.builds,0)
        f=Fixture();original=f.ns['_sx_guard'];calls=[0]
        def guard(start):
            calls[0]+=1
            if calls[0]>6:raise AssertionError('inner expiry')
            original(start)
        f.ns['_sx_guard']=guard
        with self.assertRaisesRegex(AssertionError,'inner expiry'):f.run()
        self.assertEqual(f.last_state['pick'],f.originals)
        f=Fixture();f.too_large=True
        with self.assertRaisesRegex(AssertionError,'SXSRELEASE_DYNAMIC_EXTENT'):f.run()
        self.assertEqual(f.runner_calls,0)
        self.assertEqual(f.last_state['pick'],f.originals)

    def test_builder_preserves_timed_entry_and_no_new_benchmark(self):
        emitted,manifest=source()
        self.assertLess(manifest['bytes'],1000000)
        self.assertFalse(manifest['new_comparative_timings']);self.assertFalse(manifest['H200_qualified'])
        tree=ast.parse(emitted)
        kernel=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn']
        self.assertEqual(len(kernel),1)
        self.assertEqual(ast.unparse(kernel[0]).count("state['pick']"),1)
        self.assertNotIn('_sx_cold(',emitted);self.assertNotIn('SXREPORT',emitted)


if __name__=='__main__':unittest.main()
