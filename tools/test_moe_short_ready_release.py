"""Bind the release draft to the running helper without promoting it."""
import ast
import unittest

from moe_short_ready_release import build
from moe_control_release import build as control_release
from expert_top100_moe_test import environment,prepare_small,inputs


def definition(source,name):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node)


class Draft(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.source,cls.manifest=build()

    def test_preserved_operator_contract_and_exact_private_native_requests(self):
        original,_=control_release()
        for name in ('_expert_moe_weights','_expert_moe_run','_ma_make_scratch','_fork_native_ma_prepare','_moe_bind_control'):
            self.assertEqual(definition(original,name),definition(self.source,name))
        self.assertEqual(len(self.manifest['compiler_calls']),5)
        self.assertFalse(self.manifest['public_submission_approved'])
        self.assertIn('DRAFT',self.manifest['qualification'])

    def test_actual_prepare_and_forwarding_keep_original_three_buffers(self):
        ns,calls,launches=environment(self.source)
        state,_=prepare_small(ns)
        self.assertEqual(len(state['ma_w']),14);self.assertEqual(len(state['ma_scratch']),3)
        for tokens in (1,4,8,16,32,128,256):
            data=inputs(tokens);out=ns['kernel_fn'](data,state)
            self.assertEqual(out.shape,data['hidden_states'].shape)
            self.assertIs(launches[-1][0],state['ma_ext'][next(r['key'] for r in self.manifest['routes'] if r['tokens']==(1 if tokens==4 else tokens))]._module)


if __name__=='__main__':unittest.main()
