import ast
import contextlib
import io
from pathlib import Path
import unittest

from attn_pr_qk2 import FLAG, NEW, OLD, _assignment, patch_source, resource_source

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
        with self.assertRaises(AssertionError):self.ns['_q2_math']({},t.tensor([float('nan')]))


if __name__=='__main__':unittest.main()
