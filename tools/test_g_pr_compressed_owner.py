import ast
import ctypes
import gc
import types
import unittest
import weakref

import g_pr_compressed_owner as treatment


class Tensor:
    next_pointer = 4096

    def __init__(self, shape, dtype='bf16'):
        self.shape, self.dtype, self.device = shape, dtype, 'cpu-mock-cuda-device'
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += 256

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return 64 << 20

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def t(self):
        return Tensor(self.shape[::-1], self.dtype)


def actual_entries():
    source = treatment.pinned_source()
    events, captured = [], []

    class Lt:
        def __init__(self, transpose):
            self.handle, self.desc = ctypes.c_void_p(1), ctypes.c_void_p(2)
            self.workspace = Tensor((64 << 20,))
            self.alpha, self.beta = ctypes.c_float(1), ctypes.c_float(0)
            self.lib = types.SimpleNamespace(cublasLtMatmulAlgoCheck=lambda *args: 0)

        def call(self, name, *args):
            if name == 'cublasLtMatmul':
                captured.append(args[3].value)

    def ordinary_prepare(weights):
        events.append('normal_prepare')
        return dict(pr=Tensor((4608, 8192)), o=Tensor((8192, 1024)),
                    unpool=Tensor((8192, 4096)), qkv=Tensor((1024, 8192)))

    def startup(torch, weights, state, *args, **kwargs):
        assert all(state['graft_lt'][('pr', tokens)]['weight'] is state['pr']
                   for tokens in (128, 256))
        events.append('original_startup')

    torch = types.SimpleNamespace(float32='f32', bfloat16='bf16',
                empty=lambda shape, **kwargs: Tensor(shape, kwargs['dtype']),
                cuda=types.SimpleNamespace(synchronize=lambda: events.append('complete_prior_work'),
                    current_stream=lambda: types.SimpleNamespace(cuda_stream=7)))
    env = dict(_C=ctypes, _GP=ctypes.c_void_p, _GI=ctypes.c_int, _GSZ=ctypes.c_size_t,
               torch=torch, _GraftLt=Lt, _GRAFT_LT_T={128, 256}, _GRAFT_GLUE_T=(),
               _vj_prepare=ordinary_prepare, _fork_lt_no_split=lambda *args: True,
               _graft_layouts=lambda *args: [ctypes.c_void_p(3)] * 3)
    for name in ('_GraftAlgo', '_GraftHeur', '_GraftPlan'):
        exec(treatment.segment(source, name), env)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id in ('_GRAFT_LT_PICKS', '_GRAFT_QKV256', '_GRAFT_QKV256_ATTRS')
                for target in node.targets):
            exec(ast.get_source_segment(source, node), env)
    for name in ('_graft_lt_plan', '_graft_lt_run'):
        exec(treatment.segment(source, name), env)
    exec(treatment.graft_prepare(source)[0], env)
    env['_release_normal_prepare'] = env['prepare']
    env.update(_release_expert_self_test=startup, _moe=object(), _fork_native_moe=object(),
               _ma_make_scratch=object(), _fork_prepare_marlin=object(), SCRATCH_SENTINEL=123)
    exec(treatment.segment(source, 'prepare', -1), env)
    return env, events, captured


def contract_evidence():
    env, events, captured = actual_entries()
    first, second = env['prepare']({}), env['prepare']({})
    assert events == ['normal_prepare', 'complete_prior_work', 'original_startup'] * 2
    assert first['pr'] is not second['pr']
    first_pr, second_pr = weakref.ref(first['pr']), weakref.ref(second['pr'])
    for state, tokens in ((first, 128), (second, 256), (first, 256), (second, 128)):
        env['_graft_lt_run'](Tensor((tokens, 8192)), state['graft_lt'][('pr', tokens)])
    del state
    assert captured == [first['pr'].data_ptr(), second['pr'].data_ptr()] * 2
    graph_commands = tuple(captured)
    env['torch'].cuda.synchronize()
    del first
    gc.collect()
    assert first_pr() is None and second_pr() is not None
    assert graph_commands and all(isinstance(pointer, int) for pointer in graph_commands)
    del second
    gc.collect()
    assert second_pr() is None
    return dict(actual_entries=['final prepare', 'pre-plan prepare', '_graft_lt_plan', '_graft_lt_run'],
                prepared_states=2, bound_pr_plans_per_state=2, interleaved_calls=4,
                raw_pointer_commands_retained=list(graph_commands),
                first_pr_owner_gone_after_prior_work_completion=True,
                second_state_independent=True, future_commands_can_remain_after_owner_drop=True,
                mock_only=True, observed_judge_failure=False,
                allocated_vmm_memory=False, candidate_generated=False)


class RetirementGateTests(unittest.TestCase):
    def test_exact_source_plan_scope(self):
        result = treatment.source_evidence()
        self.assertEqual(result['pr_bytes'], 75497472)
        self.assertEqual(set(result['pr_picks']), {'128', '256'})

    def test_actual_entries_and_lifetime_counterexample(self):
        result = contract_evidence()
        self.assertTrue(result['future_commands_can_remain_after_owner_drop'])

    def test_fail_closed_no_candidate(self):
        with self.assertRaisesRegex(treatment.MissingCallerRetirement, treatment.STOP):
            treatment.build()

    def test_pointer_commands_are_not_tensor_owners(self):
        env, _, captured = actual_entries()
        state = env['prepare']({})
        owner = weakref.ref(state['pr'])
        env['_graft_lt_run'](Tensor((128, 8192)), state['graft_lt'][('pr', 128)])
        self.assertIsNotNone(owner())
        del state
        gc.collect()
        self.assertIsNone(owner())
        self.assertTrue(captured)


if __name__ == '__main__':
    unittest.main()
