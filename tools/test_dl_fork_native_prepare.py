"""Exercise the generated G1b2 host dispatch with actual CPU tensors and mocked CUDA services."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

from dl_fork_native_prepare import patch
import test_dl_fork_moe_cpu as fixtures

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / 'muse_decode_layer/cand/g1b_native_experts.py').read_text('utf8')
NODES = ast.parse(patch(BASE)).body


class Preparation(unittest.TestCase):
    def exercise(self, kind='success', suffix=True, owned=True):
        env, state, weights = fixtures.FallbackHost().scope(suffix=suffix, kind=kind, installed=owned)
        env['torch'].cuda.device = lambda dev: nullcontext()
        mock_native = env['_fork_native_moe']
        env['_ma_ext'] = lambda flags: mock_native(128, state)
        env['_RouterExt'] = lambda module: module
        state.update(ma_ext={}, ma_sc6=suffix)
        names = {'_main_defines','_fork_native_moe','_fork_build_tasks','_fork_install_fallback','_ma_prepare'}
        constants = {'MOE_DEFINES','MOE_MID_DEFINES','MOE_TINY_DEFINES','MOE_TINY2_DEFINES',
                     'MOE_MID_MAX_M','MOE_TINY_MAX_M','MOE_TINY2_MAX_M','MAIN_FALLBACK_DEFINES','MOE_HAS_PDL'}
        selected = []
        for node in NODES:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                selected.append(node)
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(target, ast.Name) and target.id in constants:
                    selected.append(node)
        exec(compile(ast.Module(selected, []), '<final host>', 'exec'), env)
        tasks = env['_fork_build_tasks'](weights, state)
        self.assertEqual(len(tasks), 5 if suffix else 0)
        with ThreadPoolExecutor(max_workers=5) as pool:
            for result in [pool.submit(task) for task in tasks]:
                result.result()
        env['_ma_prepare'](weights, state)
        return env, state

    def test_every_size_uses_a_built_module_without_allocating_fallback(self):
        env, state = self.exercise()
        self.assertTrue(state['_fork_native_eligible'])
        self.assertEqual(set(state['ma_ext']), {'tiny','tiny2','mid','b4','main'})
        count = len(env['calls'])
        for m in range(1, 385):
            env['_fork_native_moe'](m, state)
        self.assertEqual(len(env['calls']), count)
        self.assertNotIn('install', env['calls'])
        self.assertEqual(env['calls'].count('run'), 7)

    def test_real_fallback_and_fatal_error_paths(self):
        for kind in ('compile','resource'):
            env, state = self.exercise(kind)
            self.assertFalse(state['_fork_native_eligible'])
            self.assertEqual(env['calls'].count('install'), 1)
        env, state = self.exercise(suffix=False, owned=False)
        self.assertEqual(state['_fork_fallback_kind'], 'Marlin')
        self.assertEqual(env['calls'].count('run'), 7)
        for kind in ('sync','launch'):
            with self.assertRaises(RuntimeError):
                self.exercise(kind)


if __name__ == '__main__':
    unittest.main()
