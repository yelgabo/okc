"""Actual pinned encoder and instruction-equivalent SC6 decoder on Torch CPU."""
import ast
import os
from pathlib import Path
import random
import contextlib
import io
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
DONOR=Path(os.environ.get('OKC_SC6_SOURCE',ROOT.parent/'okc/.build/today/kel/kernels/muse_moe_w4a16.py'))


def functions():
    t=ast.parse(DONOR.read_text(encoding='utf-8'))
    names={'_sc6_ok','_sc6_wgs','_stored_pos','_split_nibbles','_scatter_code_bits','_pack_codes_nat','_kperm_src','_kperm_codes','_kperm_rows','_wgs_copy'}
    nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'torch':torch,'DQ7':True}
    for n in t.body:
        if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='_REG_HJ' for a in n.targets):
            exec(compile(ast.Module(body=[n],type_ignores=[]),'<pinned-reg>','exec'),ns)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-functions>','exec'),ns)
    ns['_POS']=ns['_stored_pos']()
    return ns


def perm(a,b,p):
    result=0
    for i in range(4):
        n=(p>>(4*i))&15
        assert n<8
        x=(a if n<4 else b)>>(8*(n&3))&255
        result|=x<<(8*i)
    return result


def gpu_decode(w,h):
    x=w&0xfcfcfcfc;hm=h&0xfcfc
    p6=((w&0x00030303)*0x04104000)&0xffffffff
    p7=(((h&0x0303)|(w&0x03000000))*0x10400004)&0xffffffff
    return [((a*4)+0x60006000)&0xffffffff for a in
            (perm(x,0,0x4140),perm(x,0,0x4342),perm(hm,0,0x4140),perm(p6,p7,0x0703))]


class Sc6CpuTests(unittest.TestCase):
    def test_actual_encoder_instruction_decoder(self):
        ns=functions();rng=random.Random(5181)
        cases=[[rng.randrange(64,128) for _ in range(8)] for _ in range(4096)]
        cases+=[[64+v if j==i else 127-j for j in range(8)] for i in range(8) for v in range(64)]
        # WGS pieces contain128 threads; pad fixture to whole pieces.
        cases+=[[64]*8]*((-len(cases))%128)
        raw=torch.tensor(cases,dtype=torch.uint8).reshape(-1,1024)
        enc=ns['_sc6_wgs'](raw)
        self.assertTrue(ns['_sc6_ok'](raw))
        for piece in range(raw.shape[0]):
            packed=enc[piece].tolist()
            for lane in range(128):
                w=sum(packed[lane*4+i]<<(8*i) for i in range(4))
                h=sum(packed[512+lane*2+i]<<(8*i) for i in range(2))
                decoded=gpu_decode(w,h)
                expected=[((cases[piece*128+lane][j*2]|(cases[piece*128+lane][j*2+1]<<16))*16+0x5c005c00)&0xffffffff for j in range(4)]
                self.assertEqual(decoded,expected)
        print('SC6CPU instruction-equivalent decode exact',len(cases),'thread-tuples')

    def test_guard_rejects_outside_lossless_domain(self):
        ns=functions()
        for bad in (0,63,128,255):
            raw=torch.full((1,1024),64,dtype=torch.uint8);raw[0,0]=bad
            self.assertFalse(ns['_sc6_ok'](raw))
            with self.assertRaises(AssertionError):ns['_sc6_wgs'](raw)

    def test_actual_wgs_copy_keeps_codes(self):
        ns=functions();g=torch.Generator().manual_seed(811)
        codes=torch.randint(0,256,(256,2048),dtype=torch.uint8,generator=g)
        sf=torch.randint(64,128,(256,256),dtype=torch.uint8,generator=g)
        rows=torch.arange(256)
        off=ns['_wgs_copy'](codes,sf,rows,False).reshape(1,16,4,9216)
        on=ns['_wgs_copy'](codes,sf,rows,True).reshape(1,16,4,8960)
        self.assertTrue(torch.equal(off[...,:8192],on[...,:8192]))
        self.assertTrue(torch.equal(ns['_sc6_wgs'](off[...,8192:]),on[...,8192:]))

    def test_actual_host_harness_all_cases_and_fresh_negative(self):
        body=ast.parse((ROOT/'tools/moe_sc6_compare_body.py').read_text(encoding='utf-8'))
        defs=[n for n in body.body if isinstance(n,ast.FunctionDef)]
        source_tree=ast.parse(DONOR.read_text(encoding='utf-8'))
        gen=next(n for n in source_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gen_input')
        class Event:
            def __init__(self,**kw):pass
            def record(self):pass
            def synchronize(self):pass
            def elapsed_time(self,other):return .1
        class Cuda:
            active=None
            def synchronize(self):pass
            class CUDAGraph:
                def __init__(self):self.calls=[]
                def replay(self):
                    for f in self.calls:f()
            @contextlib.contextmanager
            def graph(self,g):
                self.active=g
                try:yield
                finally:self.active=None
        Cuda.Event=Event
        class Api:
            cuda=Cuda()
            def __getattr__(self,k):return getattr(torch,k)
            def empty(self,*args,**kw):
                if args==(64*1024*1024,):args=(1,)
                return torch.empty(*args,**kw)
        api=Api()
        def toy(data,x=None):
            active=((data['topk_ids']>=0)&(data['topk_ids']<32)).float()
            return (data['hidden_states'] if x is None else x)*((active*data['topk_weights']).sum(-1,keepdim=True)).to(torch.bfloat16)
        class Ext:
            def sc6_resources(self):return [96,0,0,208896,1,576]*2
        baseline='''import torch
def prepare(w):return None
def kernel_fn(d,s):
 a=((d['topk_ids']>=0)&(d['topk_ids']<32)).float()
 return d['hidden_states']*((a*d['topk_weights']).sum(-1,keepdim=True)).to(torch.bfloat16)
'''
        def fixture(stale):
            def kernel(data,state):
                out=torch.empty_like(data['hidden_states'])
                frozen=data['hidden_states'].clone() if stale and api.cuda.active else None
                def run():out.copy_(toy(data,frozen))
                if api.cuda.active is not None:api.cuda.active.calls.append(run)
                run();return out
            st=dict(lo=0,cnt=32,w=[],w_ns=[],sfx=('-DOKC_PRMTSC=1',),sfxm=('-DOKC_PRMTSC=1','-DOKC_SC6W=1'),sfxs=('-DOKC_PRMTSC=1',))
            ns=dict(torch=api,_SC_PREPARE=lambda w:dict(st),_SC_KERNEL=kernel,
                    _main_defines=lambda s:('-DOKC_WGS=1',),_ext=lambda d:Ext(),
                    _make_scratch=lambda dev:[torch.zeros(1)],_SC_BASELINE=baseline,
                    TINY_DEFINES=(),TINY2_DEFINES=(),MID_DEFINES=(),
                    HIDDEN=4096,N_EXPERTS_TOTAL=256,LOCAL_EXPERTS=32,LOCAL_START=0,TOP_K=8,SHARDING='ep8')
            exec(compile(ast.Module(body=defs+[gen],type_ignores=[]),'<actual-full-host-harness>','exec'),ns)
            return ns
        weights={k:torch.zeros(256) for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs')}
        weights['post_norm_w']=torch.zeros(4096,dtype=torch.bfloat16)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            result=fixture(False)['prepare'](weights)
        self.assertIn('SC6CORE 48 48 48',captured.getvalue())
        self.assertIn('SC6DONE',captured.getvalue())
        self.assertEqual(len(result['scarms']),4)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(AssertionError,'SC6_MATH'):
            fixture(True)['prepare'](weights)
        print('SC6CPU actual whole host harness12cases/4arms and fresh-stale negative PASS')


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
