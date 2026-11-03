import ast
import contextlib
import io
from pathlib import Path
import unittest

from attn_pr_qk2 import FLAG, NEW, OLD, ROUND_NEW, ROUND_OLD, _assignment, patch_source, resource_source

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT.parent/'okc/.build/today/attention_repeat2.py'


class SourceTests(unittest.TestCase):
    def test_exact_scope_and_abi(self):
        source=CONTROL.read_text(encoding='utf-8')
        patched=patch_source(source)
        original=ast.literal_eval(_assignment(source,'_ONE_SRC').value)
        device=ast.literal_eval(_assignment(patched,'_ONE_SRC').value)
        self.assertEqual(device.replace(FLAG,'').replace(NEW,OLD),original)
        for length in (1,8):
            for mode in range(128):
                scope=bool(length==1 and mode in (11,13))
                self.assertEqual(scope, (length,mode) in ((1,11),(1,13)))
        with self.assertRaises(AssertionError): patch_source(source+'\n')
        # Existing branch uses even0/2 and odd1/3 chains without changing any
        # fragment load/conversion, key mask, cache writer or synchronization.
        self.assertIn('for (int j2 = 0; j2 < 4; j2 += 2)',device)
        self.assertEqual(device.count('if constexpr (QK2)'),1)
        resource=resource_source(patched)
        ast.parse(resource)
        common_device=ast.literal_eval(_assignment(resource,'_ONE_SRC').value)
        self.assertEqual(common_device.count(ROUND_NEW),2)
        self.assertNotIn(ROUND_OLD,common_device)
        cpp=ast.literal_eval(_assignment(resource,'_ONE_CPP').value)
        self.assertIn('\nstd::vector<int64_t> attn_q2_resources();',cpp)
        self.assertNotIn('\\nstd::vector<int64_t> attn_q2_resources();',cpp)


class HostTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        body=(ROOT/'tools/attn_pr_qk2_body.py').read_text(encoding='utf-8')
        definitions=[n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef)]
        self.ns={'torch':torch}
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'<actual-host-functions>','exec'),self.ns)

    def test_private_inputs_and_complete_copy(self):
        t=self.torch
        original={'qkv':t.ones(2,3),'k_cache':t.zeros(3,4),'v_cache':t.ones(3,4),
                  'seq_lens':t.tensor([2]),'page_table':t.tensor([[0,1,2]]),'config':{'length':1}}
        arms=[self.ns['_q2_clone'](original) for _ in range(4)]
        for key in original:
            if isinstance(original[key],t.Tensor):
                self.assertEqual(len({a[key].data_ptr() for a in arms}),4)
        fresh=self.ns['_q2_clone'](original)
        for value in fresh.values():
            if isinstance(value,t.Tensor): value.add_(2)
        for arm in arms:
            self.ns['_q2_copy'](arm,fresh)
            for key in original:
                if isinstance(original[key],t.Tensor): self.assertTrue(t.equal(arm[key],fresh[key]))
        fresh['config']={'length':8}
        with self.assertRaises(AssertionError): self.ns['_q2_copy'](arms[0],fresh)

    def test_actual_score_and_noise_gates(self):
        def rows(v):return [(seed,kind,list(v)) for seed in (713301,713302,713303) for kind in ('sliding','full')]
        gate=self.ns['_q2_gate']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(gate(rows([10,10,9.8,9.8])))
            self.assertFalse(gate(rows([10,10,10,10])))
            self.assertFalse(gate(rows([10,10,9.8,9.4])))
            partial=rows([10,10,9.8,9.8]);partial[-1]=(713303,'full',[10,10,10.2,10.2])
            self.assertFalse(gate(partial))
            with self.assertRaises(AssertionError): gate(partial[:-1])

    def test_math_fail_closed(self):
        t=self.torch
        self.ns['_ob_check']=lambda data,out:(False,'deliberate negative')
        with self.assertRaises(AssertionError):self.ns['_q2_math']({},t.ones(1))
        self.ns['_ob_check']=lambda data,out:(True,'')
        self.ns['_q2_baseline_check']=lambda data,out:(False,'baseline negative')
        with self.assertRaises(AssertionError):self.ns['_q2_math']({},t.ones(1))
        self.ns['_q2_baseline_check']=lambda data,out:(True,'')
        with self.assertRaises(AssertionError):self.ns['_q2_math']({},t.tensor([float('nan')]))

    def test_actual_oracle_bf16_boundary_and_negative(self):
        t=self.torch
        generated=(ROOT/'.build/attn_pr_qk2.py').read_text(encoding='utf-8')
        tree=ast.parse(generated)
        self.assertTrue(generated.startswith('# okc: problem=muse_attention_decode\n'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_ob_rms','_ob_reference','_q2_baseline_reference')]
        ns={'torch':t,'FP8':t.float8_e4m3fn,'RMS_EPS':1e-5}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-oracle>','exec'),ns)
        src=ast.unparse(next(n for n in nodes if n.name=='_q2_baseline_reference'))
        self.assertIn('_ob_rms(out.to(torch.bfloat16))',src)
        judge=ast.unparse(next(n for n in nodes if n.name=='_ob_reference'))
        self.assertIn('_ob_rms(out)',judge)
        self.assertNotIn('_ob_rms(out.to(torch.bfloat16))',judge)
        gen=t.Generator().manual_seed(711)
        data={'qkv':t.randn((1,1,2304),generator=gen).to(t.bfloat16),
              'seq_lens':t.tensor([3]),'page_table':t.arange(4)[None,:],
              'k_cache':t.randn((4,2,64),generator=gen).to(t.float8_e4m3fn).float(),
              'v_cache':t.randn((4,2,64),generator=gen).to(t.float8_e4m3fn).float(),
              'k_scale':t.tensor(1.),'v_scale':t.tensor(1.),
              'config':dict(batch=1,q_len=1,num_q_heads=16,num_kv_heads=2,head_dim=64,
                            qk_scale_factor=5.4730064863875,sm_scale=.125,page_size=1,window=0,rope_theta=0.)}
        newkv=tuple(t.randn((1,1,2,64),generator=gen).to(t.float8_e4m3fn).float() for _ in range(2))
        for query_fp8 in (False,True):
            actual=ns['_q2_baseline_reference'](data,query_fp8,newkv)
            old=ns['_ob_reference'](data,query_fp8,newkv)
            self.assertTrue(t.isfinite(actual).all())
            self.assertGreater(float((actual-old).abs().max()),1e-4)


if __name__=='__main__':unittest.main()
