"""Exercise generated state cloning with real CPU Torch storage."""
import ast
import types
import unittest

import torch
import d32_attn_worker_harness as builder


class Owners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text, _ = builder.build()
        tree = ast.parse(text)
        wanted = {'_wl_storage', '_wl_tensors', '_wl_clone', '_wl_intervals',
                  '_wl_overlap', '_wl_check_owners', '_d32_pair_bind'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        cls.env = {'torch': torch, '_GUARDS': []}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'generated-owners', 'exec'), cls.env)

    def test_four_private_states_and_shared_immutable_module(self):
        env = self.env
        weight = torch.ones(8)
        module = types.ModuleType('treatment')
        source = {'attention_scratch': {'rows': {}, 'cluster': {}, 'retained': []},
                  'ma_ext': {}, 'weight': weight, 'scratch': torch.arange(16.),
                  'd32_attention_module': module}
        readonly = {env['_wl_storage'](weight)}
        arms = [env['_wl_clone'](source, readonly, i >= 2) for i in range(4)]
        for i, arm in enumerate(arms):
            env['_d32_pair_bind'](arm, i >= 2)
        env['_wl_check_owners']([source, *arms], readonly)
        self.assertEqual(len({arm['scratch'].untyped_storage().data_ptr() for arm in arms}), 4)
        self.assertTrue(all(arm['weight'] is weight and arm['d32_attention_module'] is module for arm in arms))
        self.assertEqual([arm['_d32_pair_arm'] for arm in arms], [False, False, True, True])
        with self.assertRaises(AssertionError):
            env['_wl_check_owners']([source, arms[0], arms[0]], readonly)
        with self.assertRaises(RuntimeError):
            env['_wl_clone'](dict(source, opaque=object()), readonly, False)


if __name__ == '__main__':
    unittest.main()
