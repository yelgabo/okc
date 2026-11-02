"""Exercise real extracted dual-build dispatch, without importing CUDA/Torch."""
import ast
import types
import unittest

import expert_top100_pair_package as package
from expert_top100_pregamma import capture, transform
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings


def environment(source):
    tree = ast.parse(source)
    env = bindings(source)
    env.update(flags(source))
    calls = []

    class Extension:
        def okc_set_pdl(self, value):
            assert value == 0

    def loader(**kwargs):
        calls.append(kwargs)
        return Extension()

    env.update(load_inline=loader, _MA_EXTS={}, _MA_MAIN_NS=3, _MA_PDL=False,
               torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
               _RouterExt=lambda module: module)
    for name in ('_main_defines', '_ma_ext', '_fork_native_moe', '_expert_pair_source', '_expert_pair_bind'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(ast.get_source_segment(source, node), env)
    env['_MA_ALT_SRCS']['expert_top100_pregamma'] = env['_expert_pair_source']()
    return env, calls


def state():
    return dict(ma_sfx=('-DOKC_PRMTSC=1',), ma_sc6=True, ma_ext={})


class Packaging(unittest.TestCase):
    def test_real_compiler_inputs_and_source_identity(self):
        source, manifest = package.build()
        env, calls = environment(source)
        base = package.BASE.read_text(encoding='utf8')
        original, _ = capture(base)
        control = state()
        for tokens in (1, 8, 16, 32, 128, 256):
            env['_fork_native_moe'](tokens, control)
        self.assertEqual(calls, original)  # Exact host/CUDA/flags/names/exports.
        treatment = state()
        env['_expert_pair_bind'](treatment, True)
        native = bindings(base)['MOE_CUDA_SRC']
        new_native, _ = transform(native)
        self.assertEqual(len(calls), 7)
        for old, new in zip(original[-2:], calls[-2:]):
            self.assertEqual(new['cuda_sources'], new_native)
            for key in ('cpp_sources', 'functions', 'extra_cuda_cflags', 'verbose'):
                self.assertEqual(old[key], new[key])
            self.assertNotEqual(old['name'], new['name'])
        for tokens in (1, 8, 16, 32, 128, 256):
            env['_fork_native_moe'](tokens, treatment)
        self.assertEqual(len(calls), 7)
        self.assertIsNot(control['ma_ext']['b4'], treatment['ma_ext']['b4'])
        for key in ('tiny', 'tiny2', 'mid'):
            self.assertIs(control['ma_ext'][key], treatment['ma_ext'][key])
        self.assertFalse(manifest['dispatch_ready'])

    def test_state_binding_guards_and_duplicate_build_reuse(self):
        env, calls = environment(package.build()[0])
        first, second = state(), state()
        env['_expert_pair_bind'](first, True)
        env['_expert_pair_bind'](second, True)
        self.assertEqual(len(calls), 2)
        self.assertIsNot(first['ma_ext'], second['ma_ext'])
        with self.assertRaisesRegex(RuntimeError, 'INVALID_REBIND'):
            env['_expert_pair_bind'](first, False)
        first['_expert_pair_arm'] = False
        with self.assertRaisesRegex(RuntimeError, 'ARM_MUTATED'):
            env['_fork_native_moe'](32, first)
        with self.assertRaisesRegex(RuntimeError, 'INVALID_REBIND'):
            env['_expert_pair_bind'](state(), 1)

    def test_source_drift_rejects_before_build(self):
        env, calls = environment(package.build()[0])
        env['MOE_CUDA_SRC'] = env['MOE_CUDA_SRC'].replace('fork_bf16(d4[h] * gsc) * w0', 'BROKEN')
        with self.assertRaisesRegex(RuntimeError, 'SOURCE_DRIFT'):
            env['_expert_pair_source']()
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
