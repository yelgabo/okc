import ast
import io
import tokenize
import types
import unittest

import g_short_serial_producer as treatment
from test_g_timed_silu_table import capture

ORIGINAL = treatment.BASE.read_bytes().decode()
SOURCE = treatment.build(ORIGINAL)


def definitions(source, name):
    return [ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == name]


class Tensor:
    def __init__(self, dtype, tokens=1, contiguous=True):
        self.dtype, self.shape = dtype, (tokens, 8192)
        self.device, self.contiguous = 'stub-device', contiguous

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return id(self)


def isolated(source):
    events, states = [], []
    def empty(shape, dtype, device):
        assert dtype == 'f32'
        output = Tensor(dtype, shape[0])
        events.append(('router_output', output))
        return output
    def gemm(state, name, h, tokens, width, hidden):
        assert h.dtype == 'bf16'
        events.append((name, state, h, tokens))
        return Tensor('bf16', tokens)
    class Router:
        def router_fp32_launch(self, xptr, wptr, optr, tokens, stream):
            events.append(('router_launch', xptr, wptr, optr, tokens, stream))
            return 0
    def fresh_state():
        state = dict(bfp=object(), router_fp32=Tensor('f32'), router_fp32_ext=Router(), pr=list(range(4096)))
        states.append(state)
        return state
    def fused(h, hr, state):
        events.append(('fused', state))
        return Tensor('bf16', h.shape[0]), Tensor('f32', h.shape[0])
    env = dict(torch=types.SimpleNamespace(empty=empty, float32='f32', bfloat16='bf16',
                   cuda=types.SimpleNamespace(current_stream=lambda: types.SimpleNamespace(cuda_stream=7))),
               GLE_FRT=True, _bfp_gemm=gemm, D_EXPERT=4096, HIDDEN=8192, N_EXPERTS=256,
               _bfp_pdl=lambda tokens, name: events.append(('existing_pool_setup', tokens, name)),
               F=types.SimpleNamespace(linear=lambda h, weight: Tensor('bf16', h.shape[0])),
               bfp_fused_pooler_router=lambda *args: ('existing_fused_long', args),
               dense=lambda *args: ('existing_dense', 0), _gle_frt=fused)
    router_defs = definitions(ORIGINAL, '_router_fp32_gemv')
    exec(router_defs[0], env)
    env['_kfix_router_fp32_gemv'] = env['_router_fp32_gemv']
    exec(router_defs[-1], env)
    selector_defs = definitions(ORIGINAL, '_pool_router_fp32')
    exec(selector_defs[0], env)
    serial = env['_pool_router_fp32']
    env['_gle_prior_pool_router_fp32'] = serial
    exec(selector_defs[-1], env)
    final_selector = env['_pool_router_fp32']
    def original_normal_prepare(weights):
        state = fresh_state()
        events.append(('normal_begin', state))
        assert '_short_serial_producer' not in state
        env['_pool_router_fp32'](Tensor('bf16'), Tensor('f32'), 1, state, 'bfp')
        events.append(('normal_done', state))
        if weights == 'fail':
            raise RuntimeError('original preparation failure')
        return state
    def startup(torch, weights, state, *args, **kwargs):
        assert '_short_serial_producer' not in state
        events.append(('startup_begin', state))
        env['_pool_router_fp32'](Tensor('bf16'), Tensor('f32'), 1, state, 'bfp')
        events.append(('startup_done', state))
    env.update(_release_normal_prepare=original_normal_prepare, _release_expert_self_test=startup,
               _moe=object(), _fork_native_moe=object(), _ma_make_scratch=object(),
               _fork_prepare_marlin=object(), SCRATCH_SENTINEL=123)
    exec(definitions(ORIGINAL, 'prepare')[-1], env)
    original_prepare = env['prepare']
    exec(source[len(ORIGINAL):], env)
    assert env['_short_serial_parent_selector'] is final_selector
    assert env['_short_serial_parent_prepare'] is original_prepare
    assert env['_gle_prior_pool_router_fp32'] is serial
    return env, events, states


def contract_evidence(source):
    treatment.validate(source, ORIGINAL)
    env, events, states = isolated(source)
    first = env['prepare']('first')
    assert first['_short_serial_producer'] is True
    expected = ['normal_begin', 'fused', 'normal_done', 'startup_begin', 'fused', 'startup_done']
    assert [event[0] for event in events] == expected
    events.clear()
    second = env['prepare']('second')
    assert first is not second and first['_short_serial_producer'] is True
    assert [event[0] for event in events] == expected
    second['router_fp32_ext'] = first['router_fp32_ext']
    parent = env['_short_serial_parent_selector']
    def tracked_parent(*args):
        events.append(('parent_selector',))
        return parent(*args)
    env['_short_serial_parent_selector'] = tracked_parent
    selected = []
    for state in (first, second, first):
        for tokens in (1, 8):
            for pick in ('bfp', 'bfpf'):
                events.clear()
                h, hr = Tensor('bf16', tokens), Tensor('f32', tokens)
                result, split = env['_pool_router_fp32'](h, hr, tokens, state, pick)
                assert split == -1 and result[0].dtype == 'bf16' and result[1].dtype == 'f32'
                names = [event[0] for event in events]
                assert 'fused' not in names and 'parent_selector' not in names
                assert [name for name in names if name in ('pool', 'router_launch')] == ['pool', 'router_launch']
                router = next(event for event in events if event[0] == 'router_launch')
                assert router[1] == hr.data_ptr() and router[2] == state['router_fp32'].data_ptr()
                selected.append((tokens, pick))
    fallbacks = []
    for tokens in (2, 16, 32, 33, 128, 256, 384):
        events.clear()
        env['_pool_router_fp32'](Tensor('bf16', tokens), Tensor('f32', tokens), tokens, first, 'bfp')
        assert events[0][0] == 'parent_selector'
        fallbacks.append(('tokens', tokens))
    variants = [('inactive', dict(first, _short_serial_producer=False), Tensor('bf16'), Tensor('f32'), 'bfp'),
                ('missing_flag', {key:value for key,value in first.items() if key != '_short_serial_producer'}, Tensor('bf16'), Tensor('f32'), 'bfp'),
                ('missing_bfp', dict(first, bfp=None), Tensor('bf16'), Tensor('f32'), 'bfp'),
                ('missing_router', dict(first, router_fp32=None), Tensor('bf16'), Tensor('f32'), 'bfp'),
                ('missing_hr', first, Tensor('bf16'), None, 'bfp'),
                ('h_precision', first, Tensor('f32'), Tensor('f32'), 'bfp'),
                ('hr_precision', first, Tensor('bf16'), Tensor('bf16'), 'bfp'),
                ('weight_precision', dict(first, router_fp32=Tensor('bf16')), Tensor('bf16'), Tensor('f32'), 'bfp'),
                ('h_layout', first, Tensor('bf16', contiguous=False), Tensor('f32'), 'bfp'),
                ('hr_layout', first, Tensor('bf16'), Tensor('f32', contiguous=False), 'bfp'),
                ('pick', first, Tensor('bf16'), Tensor('f32'), False),
                ('absent_state', None, Tensor('bf16'), Tensor('f32'), 'bfp')]
    for label, state, h, hr, pick in variants:
        events.clear()
        try:
            env['_pool_router_fp32'](h, hr, 1, state, pick)
        except (AttributeError, TypeError):
            pass
        assert events[0][0] == 'parent_selector', label
        fallbacks.append((label, 1))
    env['GLE_FRT'] = False
    events.clear()
    env['_pool_router_fp32'](Tensor('bf16'), Tensor('f32'), 1, first, 'bfp')
    assert events[0][0] == 'parent_selector'
    env['GLE_FRT'] = True
    try:
        env['prepare']('fail')
    except RuntimeError as error:
        assert str(error) == 'original preparation failure'
    else:
        raise AssertionError('original prepare failure swallowed')
    assert '_short_serial_producer' not in states[-1]
    old_calls, old_routes = capture(ORIGINAL)
    new_calls, new_routes = capture(source)
    assert len(old_calls) == len(new_calls) == 7
    assert old_calls == new_calls and old_routes == new_routes
    factories = [dict(index=index, cuda_sha256=treatment.sha(call['cuda_sources']),
                      cpp_sha256=treatment.sha(call['cpp_sources']),
                      ordered_flags=call['extra_cuda_cflags']) for index, call in enumerate(new_calls)]
    return dict(independent_prepare_states=2, startup_before_activation=True,
                selected_short_calls=len(selected), fallback_cases=len(fallbacks)+1,
                all_seven_factories_identical=True, factories=factories,
                route_requests=new_routes, CUDA_numeric_proof=False)


class Contracts(unittest.TestCase):
    def test_actual_prepare_selector_router_and_all_factories(self):
        self.assertEqual(contract_evidence(SOURCE)['selected_short_calls'], 12)

    def test_exact_prefix_notices_and_determinism(self):
        self.assertEqual(SOURCE, treatment.build(ORIGINAL))
        self.assertTrue(SOURCE.startswith(ORIGINAL))
        comments = lambda source: [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT]
        self.assertEqual(comments(ORIGINAL), comments(SOURCE))
        self.assertNotIn('\r', SOURCE)
        self.assertLess(len(SOURCE.encode()), 1000000)

    def test_interleaved_prepare_and_state_local_activation(self):
        env, events, states = isolated(SOURCE)
        first = env['prepare']('first')
        normal = env['_release_normal_prepare']
        def interleaved(weights):
            events.clear()
            env['_pool_router_fp32'](Tensor('bf16'), Tensor('f32'), 1, first, 'bfp')
            self.assertEqual([event[0] for event in events], ['pool', 'router_output', 'router_launch'])
            events.clear()
            return normal(weights)
        env['_release_normal_prepare'] = interleaved
        second = env['prepare']('second')
        self.assertEqual([event[0] for event in events],
                         ['normal_begin', 'fused', 'normal_done', 'startup_begin', 'fused', 'startup_done'])
        first['_short_serial_producer'] = False
        second['router_fp32_ext'] = first['router_fp32_ext']
        for state, expected in ((first, 'fused'), (second, 'pool'), (first, 'fused')):
            events.clear()
            env['_pool_router_fp32'](Tensor('bf16'), Tensor('f32'), 1, state, 'bfp')
            self.assertEqual(events[0][0], expected)
        self.assertIs(env['GLE_FRT'], True)

    def test_source_negative_contracts(self):
        mutations = [('early_enable', 'state = _short_serial_parent_prepare(weights)', 'weights["_short_serial_producer"] = True\n    state = _short_serial_parent_prepare(weights)'),
                     ('global_setter', '_short_serial_parent_prepare = prepare', '_short_serial_parent_prepare = prepare\nGLE_FRT = False'),
                     ('long_switch', 'T in (1, 8)', 'T <= 384'),
                     ('wrong_alias', '_short_serial_parent_selector = _pool_router_fp32', '_short_serial_parent_selector = _gle_prior_pool_router_fp32'),
                     ('wrong_prepare', '_short_serial_parent_prepare = prepare', '_short_serial_parent_prepare = _release_normal_prepare'),
                     ('router_precision', 'hr.dtype == torch.float32', 'hr.dtype == torch.bfloat16'),
                     ('h_precision', 'h.dtype == torch.bfloat16', 'h.dtype == torch.float32'),
                     ('weight_precision', 'state["router_fp32"].dtype == torch.float32', 'state["router_fp32"].dtype == torch.bfloat16'),
                     ('wrong_call', 'return _gle_prior_pool_router_fp32(h, hr, T, state, ppick)', 'return _short_serial_parent_selector(h, hr, T, state, ppick)'),
                     ('missing_activation', 'state["_short_serial_producer"] = True', 'state["_short_serial_producer"] = False'),
                     ('native_source', '#include <cuda.h>', '#include <cuda.h>\n#define SHORT_SERIAL_MUTATION 1'),
                     ('flags', '-DOKC_PRMTSC=1', '-DOKC_PRMTSC=0')]
        for label, before, after in mutations:
            with self.subTest(label=label):
                self.assertIn(before, SOURCE)
                changed = SOURCE.replace(before, after, 1)
                with self.assertRaises(AssertionError):
                    treatment.validate(changed, ORIGINAL)

    def test_runtime_negative_alias_or_startup_bypass(self):
        for before, after in [('_short_serial_parent_selector = _pool_router_fp32', '_short_serial_parent_selector = _gle_prior_pool_router_fp32'),
                              ('_short_serial_parent_prepare = prepare', '_short_serial_parent_prepare = _release_normal_prepare')]:
            with self.subTest(before=before), self.assertRaises(AssertionError):
                isolated(SOURCE.replace(before, after))


if __name__ == '__main__':
    unittest.main()
