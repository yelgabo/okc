import ast
from pathlib import Path
import types
import unittest

from g_integration_common import build as common
from g_short_outgoing_seam import build, MARKER, PIN, sha

ROOT = Path(__file__).resolve().parents[1]
BASE, _ = common((ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8'), True)


def function(source, name, last=True):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    return ast.get_source_segment(source, nodes[-1 if last else 0])


def literals():
    namespace = {}
    for node in ast.parse(BASE).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return namespace


class Tensor:
    device = 'cpu'
    def __init__(self, shape):
        self.shape = shape
    def view(self, *shape):
        return Tensor(shape)


class Extension:
    def __init__(self):
        self.calls = []
    def seam_sk2(self, *args):
        self.calls.append(args)


def namespace(enabled):
    source, manifest = build(BASE, enabled)
    scope = literals()
    extension = Extension()
    captures = []
    fallback = []
    scope.update(torch=types.SimpleNamespace(bfloat16='bf16', empty=lambda shape, **kwargs: Tensor(shape), cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
        load_inline=lambda **kwargs: captures.append(kwargs) or extension,
        prepare=lambda weights: weights,
        _bfp_gemm=lambda *args: fallback.append(args) or 'ordinary',
        bfp__run_sk2=lambda *args: 'ordinary-run', bfp__SK2_DELAY=0)
    for name in ('bfp__sk2_geom', 'bfp__sk2_tokr', 'bfp__sk2_name', 'bfp_kernel_fn', 'bfp__run'):
        exec(function(BASE, name), scope)
    exec(function(BASE, 'dense', False), scope)
    scope['_vj_dense'] = scope['dense']
    exec(function(BASE, 'dense'), scope)
    scope.update(_BFP_LAYER={'un': 'unpooler'}, BFP_MAX_T=32, bfp_GEMM_DIAG=False)
    exec(source.split(MARKER)[1], scope)
    return scope, extension, captures, fallback


def state():
    return {'bfp': {'choice': {('unpooler', 1): ('sk2',), ('unpooler', 8): ('sk2',)},
                    'sk2': {'slice8_r192': (17, object(), object(), 64, 0, 1024)}, 'dummy': object()},
            'mm_pick': {(1, 'un'): 'bfp', (8, 'un'): 'bfp'}}


def original_factory_flags(delay):
    scope = literals()
    captures = []
    scope.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
        bfp__ext=None, bfp__SK2_DELAY=delay, load_inline=lambda **kwargs: captures.append(kwargs) or object(),
        bfp_wgmma_defs=lambda *args: '')
    for name in ('bfp__tma_srcs', 'bfp__gemm_src', 'bfp__wide_src', 'bfp__sk2_src', 'bfp__gq_src'):
        scope[name] = lambda *args: ''
    exec(function(BASE, 'bfp__get_ext'), scope)
    scope['bfp__get_ext']()
    return captures[0]['extra_cuda_cflags']


class OutgoingSeam(unittest.TestCase):
    def test_reverse_pins_and_exact_single_policy_delta(self):
        off, off_manifest = build(BASE)
        on, on_manifest = build(BASE, True)
        self.assertEqual(sha(BASE), PIN)
        self.assertEqual(off.split(MARKER)[0], BASE)
        self.assertEqual(on.split(MARKER)[0], BASE)
        self.assertEqual(off.replace('_OS_ENABLED = False', '_OS_ENABLED = True'), on)
        self.assertFalse(off_manifest['GPU_qualified'])
        self.assertFalse(on_manifest['dispatch_enabled'])
        with self.assertRaisesRegex(AssertionError, 'exact raw common OFF'):
            build(on, True)

    def test_real_dense_dispatch_both_arms_tokens_and_private_states(self):
        for enabled in (False, True):
            scope, extension, captures, fallback = namespace(enabled)
            prepared = scope['prepare'](state())
            self.assertNotIn('_os_launch', prepared['bfp'])
            self.assertIsNot(prepared['_os_unpool_bfp'], prepared['bfp'])
            self.assertIs(prepared['_os_unpool_bfp']['sk2'], prepared['bfp']['sk2'])
            for tokens in (1, 8):
                out, splits = scope['dense'](Tensor((tokens, 4096)), Tensor((1024, 4096)), tokens, prepared, 'un')
                self.assertEqual(out.shape, (1, tokens, 1024))
                self.assertEqual(splits, 1)
                self.assertEqual(extension.calls[-1][-1], 1073741826 if enabled else 0)
            self.assertEqual(len(captures), 1)
            self.assertEqual(captures[0]['functions'], ['seam_sk2'])
            self.assertFalse(fallback)

    def test_interleaved_controls_have_no_setter_or_policy_leak(self):
        off, off_ext, _, _ = namespace(False)
        on, on_ext, _, _ = namespace(True)
        off_state = off['prepare'](state())
        on_state = on['prepare'](state())
        for scope, prepared in ((on, on_state), (off, off_state), (on, on_state), (off, off_state)):
            scope['_bfp_gemm'](prepared, 'un', Tensor((1, 4096)), 1, 1024, 4096)
        self.assertEqual([args[-1] for args in off_ext.calls], [0, 0])
        self.assertEqual([args[-1] for args in on_ext.calls], [1073741826, 1073741826])
        self.assertIsNot(off_state['_os_unpool_bfp']['sk2'], on_state['_os_unpool_bfp']['sk2'])

    def test_only_role_un_and_tokens_one_eight(self):
        scope, extension, _, fallback = namespace(True)
        prepared = scope['prepare'](state())
        for role, tokens in (('pool', 1), ('qkv', 8), ('router', 1), ('un', 16), ('un', 32)):
            self.assertEqual(scope['_bfp_gemm'](prepared, role, Tensor((tokens, 4096)), tokens, 1024, 4096), 'ordinary')
        self.assertEqual(len(fallback), 5)
        self.assertFalse(extension.calls)
        self.assertEqual(scope['bfp__run_sk2'](None, 'unpooler', 1, 1024, 4096, prepared['bfp']), 'ordinary-run')

    def test_shared_extension_and_later_python_flag_changes_cannot_leak(self):
        shared = Extension()
        off, _, _, _ = namespace(False)
        on, _, _, _ = namespace(True)
        for scope in (off, on):
            scope['load_inline'] = lambda **kwargs: shared
        control = off['prepare'](state())
        treatment = on['prepare'](state())
        off['_OS_ENABLED'] = True
        on['_OS_ENABLED'] = False
        for scope, prepared in ((on, treatment), (off, control), (on, treatment)):
            scope['_bfp_gemm'](prepared, 'un', Tensor((1, 4096)), 1, 1024, 4096)
        self.assertEqual([args[-1] for args in shared.calls], [1073741826, 0, 1073741826])

    def test_bad_shape_route_selection_and_missing_state_fail_closed(self):
        scope, extension, _, _ = namespace(True)
        prepared = scope['prepare'](state())
        with self.assertRaisesRegex(AssertionError, 'shape'):
            scope['_bfp_gemm'](prepared, 'un', Tensor((1, 4096)), 1, 8192, 4096)
        prepared['_os_unpool_bfp']['choice'][('unpooler', 1)] = ('torch',)
        with self.assertRaisesRegex(AssertionError, 'route'):
            scope['_bfp_gemm'](prepared, 'un', Tensor((1, 4096)), 1, 1024, 4096)
        with self.assertRaises(KeyError):
            scope['_bfp_gemm'](state(), 'un', Tensor((1, 4096)), 1, 1024, 4096)
        prepared = scope['prepare'](state())
        scope['bfp_SK2_PICK'][('unpooler', 8)] = 'wrong'
        with self.assertRaisesRegex(AssertionError, 'selection'):
            scope['_bfp_gemm'](prepared, 'un', Tensor((1, 4096)), 1, 1024, 4096)
        self.assertFalse(extension.calls)

    def test_native_device_prefix_exact_and_scoped_host_argument(self):
        scope, _, captures, _ = namespace(True)
        scope['prepare'](state())
        source = captures[0]['cuda_sources']
        old = scope['bfp_SK2_SRC']
        boundary = old.index('static void launch_sk2(')
        self.assertIn(old[:boundary], source)
        host = source[source.index('static void launch_sk2('):]
        self.assertNotIn('const int pdl = g_sk2_pdl;', host)
        self.assertIn('const int pdl = seam_pdl;', host)
        self.assertIn('programmaticStreamSerializationAllowed = 1;', host)
        self.assertIn('dbg, (int)seam_pdl); return;', host)
        self.assertIn('torch::Tensor dbgt, int64_t seam_pdl);', captures[0]['cpp_sources'])
        self.assertNotIn('sk2_set_pdl', captures[0]['cpp_sources'])
        self.assertEqual(captures[0]['extra_cuda_cflags'], original_factory_flags(0))

    def test_actual_unpool_factory_flags_including_nonzero_delay(self):
        scope, _, captures, _ = namespace(True)
        for delay in (0, 7):
            scope['bfp__SK2_DELAY'] = delay
            scope['prepare'](state())
            self.assertEqual(captures[-1]['extra_cuda_cflags'], original_factory_flags(delay))
        self.assertNotEqual(captures[0]['name'], captures[1]['name'])

    def test_selected_cl_one_x_data_is_after_wait_proxy(self):
        source = literals()['bfp_SK2_SRC']
        start = source.index('if (pdl) asm volatile("griddepcontrol.wait;"', source.index('__syncthreads();'))
        proxy = source.index('if (pdl) asm volatile("fence.proxy.async.global;"', start)
        first_x = source.index('tma_load_3d', proxy)
        self.assertLess(start, proxy)
        self.assertLess(proxy, first_x)
        preamble = source[source.index('if (tid == 0) {'):start]
        self.assertIn('if constexpr (CL > 1)', preamble)
        self.assertIn('bulk_g2s', preamble)
        self.assertNotIn('x[', preamble)
        self.assertEqual(preamble.count('tma_load_3d'), 1)
        cluster_else = preamble[preamble.index('    } else {\n      if (pdl) asm volatile("griddepcontrol.wait;"'):]
        self.assertIn('fence.proxy.async.global;', cluster_else)
        self.assertIn('tma_load_3d', cluster_else)


if __name__ == '__main__':
    unittest.main()
