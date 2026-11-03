"""Exercise generated entry selection and fail-closed preparation without CUDA."""
import ast
from types import SimpleNamespace as NS
import unittest

import d32_attn_worker_harness as builder
from attention_top100_screen import function
from test_attention_layer_continuity import config, environment, invoke


class MatchedEntries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.manifest = builder.build()
        cls.tree = ast.parse(cls.source)

    def env(self):
        captures, owners = [], []
        env, _, _ = environment(self.source, captures, owners)
        wanted = {'_d32_attention_scope', '_d32_attention_module', '_d32_pair_bind'}
        aliases = {'_d32_pair_parent_module', '_d32_pair_selected_module'}
        nodes = [node for node in self.tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted
                 or isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in aliases for t in node.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-pair-entry', 'exec'), env)
        env['_WL_KERNEL'] = env['kernel_fn']
        function(self.tree, '_wl_run', env)
        treatment = NS(run=lambda *args: captures.append(('treatment', args)))
        env['_prepare_attention_modules7'] = lambda state: state.update(
            d32_attention_module=treatment, d32_attention_eligible=True)
        return env, captures

    def test_actual_selection_across_standard_cells(self):
        env, captures = self.env()
        for length in (1, 8):
            for batch in (1, 16, 32):
                for window in (0, 2048):
                    for arm in (False, True):
                        with self.subTest(batch=batch, length=length, window=window, arm=arm):
                            cfg = config(batch, length, window)
                            state, data, _ = invoke(env, cfg)
                            env['_d32_pair_bind'](state, arm)
                            env['_wl_run'](data, state)
                            expected = arm and batch == 32 and length == 1 and window == 0
                            self.assertEqual(captures[-1][0], 'treatment' if expected else 'control')
                            self.assertEqual(state['_d32_probe_seen'], [expected])

    def test_missing_or_wrong_selection_and_offline_hold_fail(self):
        env, _ = self.env()
        state, data, _ = invoke(env, config(32, 1, 0))
        env['_d32_pair_bind'](state, True)
        state['d32_attention_module'] = None
        with self.assertRaises(ValueError):
            env['_wl_run'](data, state)
        state['d32_attention_module'] = object()
        state['_d32_pair_arm'] = False
        with self.assertRaises(RuntimeError):
            env['_wl_run'](data, state)
        prepares = [node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare']
        hold_env = {'_EH_OFFLINE_HOLD': True}
        exec(compile(ast.Module(body=[prepares[-1]], type_ignores=[]), 'held-prepare', 'exec'), hold_env)
        with self.assertRaisesRegex(RuntimeError, 'STOP_CPU_HOST_GATE'):
            hold_env['prepare']({})

    def test_private_payload_changes_only_hold(self):
        enabled, manifest = builder.build(private_check=True)
        self.assertEqual(enabled, self.source.replace('_EH_OFFLINE_HOLD=True', '_EH_OFFLINE_HOLD=False', 1))
        self.assertTrue(manifest['dispatch_ready'])


if __name__ == '__main__':
    unittest.main()
