import ast
import os
from pathlib import Path
import unittest

from dl_owned_attention_boundary import BASE_SHA,PINS,OLD_LOOP,NEW_LOOP,OLD_PAIR,NEW_PAIR,patch_owned_attention

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('OKC_OWNED_LAYER',ROOT.parent/'okc/muse_decode_layer/cand/layer_lt_boundary.py'))


def literals(source):
    return {getattr(n.targets[0],'id',''):ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}


def routes(source):
    tree=ast.parse(source)
    names={'_attn_path','_one_wins','_fused_wins','_ws_wins'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'_attn_'+name:name for name in ('one','fused','ws','prep','sglang')}
    ns['_ONE_ROUTE']={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-owned-routing>','exec'),ns)
    return ns


class SourceTests(unittest.TestCase):
    def test_exact_source_delta(self):
        source=BASE.read_text(encoding='utf-8')
        revised=patch_owned_attention(source)
        a,b=literals(source),literals(revised)
        for name in a:
            if name=='_ONE_SRC':
                self.assertEqual(b[name].count(NEW_LOOP),2)
                self.assertEqual(b[name].replace(NEW_LOOP,OLD_LOOP),a[name])
            elif name in PINS:
                self.assertEqual(b[name].count(NEW_PAIR),1)
                self.assertEqual(b[name].replace(NEW_PAIR,OLD_PAIR),a[name])
            else:self.assertEqual(a[name],b[name],name)
        with self.assertRaises(AssertionError):patch_owned_attention(source+'\n')
        with self.assertRaises(AssertionError):patch_owned_attention(revised)

    def test_actual_dispatch_and_all_fallbacks(self):
        ns=routes(BASE.read_text(encoding='utf-8'))
        active={f'attn_{n}_ours':True for n in ('one','fused','ws','prep')}
        for b in (1,16,32):
            for length in (1,8):
                for window in (0,2048):
                    cfg=dict(batch=b,q_len=length,window=window,page_size=1,num_q_heads=16,num_kv_heads=2,head_dim=64)
                    self.assertEqual(ns['_attn_path'](cfg,active),'one')
                    fallback=dict(active,attn_one_ours=False)
                    expected='fused' if length==1 and (window or b>2) else 'ws' if length==8 and (window or b>2) else 'prep'
                    self.assertEqual(ns['_attn_path'](cfg,fallback),expected)
                    self.assertEqual(ns['_attn_path'](cfg,dict(fallback,attn_fused_ours=False,attn_ws_ours=False)),'prep')
                    self.assertEqual(ns['_attn_path'](cfg,dict.fromkeys(active,False)),'sglang')
        cfg.update(num_q_heads=8)
        self.assertEqual(ns['_attn_path'](cfg,active),'sglang')

    def test_existing_bf16_fallbacks_and_diagnostic_unchanged(self):
        source=BASE.read_text(encoding='utf-8')
        revised=patch_owned_attention(source)
        t=ast.parse(revised)
        defs={n.name:ast.get_source_segment(revised,n) for n in t.body if isinstance(n,ast.FunctionDef)}
        self.assertIn('.to(torch.bfloat16), gate)',defs['_attn_sglang'])
        self.assertIn('out = out.to(torch.bfloat16)',defs['_attn_prep'])
        self.assertIn('return rms(out)',defs['_attn_ref32']) # mathematical diagnostic intentionally stays FP32
        prep=literals(source)['_PREP_SRC']
        self.assertEqual(prep.count('__global__ void gate_kernel(const __nv_bfloat16*'),1)
        gate=prep.split('__global__ void gate_kernel(',1)[1]
        self.assertIn('__bfloat1622float2',gate)
        self.assertLess(gate.index('__bfloat1622float2'),gate.index('rsqrtf('))


if __name__=='__main__':unittest.main()
