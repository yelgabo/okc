import ast
import os
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('OKC_OWNED_LAYER',ROOT.parent/'okc/muse_decode_layer/cand/layer_lt_boundary.py'))


def finish(y,gate,ss_input=None):
    stats=y if ss_input is None else ss_input
    return (y*torch.rsqrt(stats.square().mean(-1,keepdim=True)+1.e-5)*gate.sigmoid()).bfloat16()


class NumericTests(unittest.TestCase):
    def test_rne_and_pre_rms_placement_observable(self):
        g=torch.Generator().manual_seed(610091)
        raw=torch.randn((17,16,64),generator=g)
        raw[0]=0
        gate=torch.randn(raw.shape,generator=g).bfloat16().float()
        rounded=raw.bfloat16().float()
        bits=raw.view(torch.int32)
        manual=(((bits+0x7fff+((bits>>16)&1))>>16)<<16).view(torch.float32)
        self.assertTrue(torch.equal(rounded,manual))
        target=finish(rounded,gate)
        self.assertGreater(torch.count_nonzero(target!=finish(raw,gate)).item(),0)
        self.assertGreater(torch.count_nonzero(target!=finish(rounded,gate,ss_input=raw)).item(),0)
        self.assertEqual(target.dtype,torch.bfloat16)
        self.assertEqual(torch.count_nonzero(target[0]).item(),0)

    def test_actual_python_fallbacks_round_before_gate(self):
        tree=ast.parse(BASE.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_attn_sglang','_attn_prep')]
        gen=torch.Generator().manual_seed(729)
        values=torch.randn((2,16,64),generator=gen)
        qkv=torch.randn((2,2304),generator=gen).bfloat16()
        observations=[]
        def gate(out,rawgate):
            observations.append(out.dtype)
            self.assertEqual(out.dtype,torch.bfloat16)
            return finish(out.float().reshape(2,16,64),rawgate.float().reshape(2,16,64)).reshape(2,1024)
        class Ext:
            def attn_prep(self,*args):return torch.empty((2,16,64),dtype=torch.float8_e4m3fn)
            def attn_gate(self,out,x):return gate(out,x[:,1280:])
        ns={'torch':torch,'FP8':torch.float8_e4m3fn,'store_cache':lambda *a,**kw:None,
            '_fa3':lambda *args:values,'_ext':lambda:Ext()}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-fallback-functions>','exec'),ns)
        data=dict(config=dict(tokens=2,num_q_heads=16,num_kv_heads=2,head_dim=64,rope_theta=0.,kv_scale=1.),
                  k_cache=torch.empty((2,2,64),dtype=torch.float8_e4m3fn),v_cache=torch.empty((2,2,64),dtype=torch.float8_e4m3fn),
                  cache_loc=torch.arange(2),positions=torch.arange(2),k_scale=torch.ones(1),v_scale=torch.ones(1))
        state=dict(qk_norm=lambda x:(x[:,:1024],x[:,1024:1152]),o_norm_gate=gate,inv_freq=torch.empty(32))
        a=ns['_attn_sglang'](qkv,data,state)
        b=ns['_attn_prep'](qkv,data,state)
        expected=gate(values.bfloat16().reshape(2,1024),qkv[:,1280:])
        self.assertTrue(torch.equal(a,expected))
        self.assertTrue(torch.equal(b,expected))
        self.assertEqual(observations,[torch.bfloat16]*3)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
