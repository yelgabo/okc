"""Real Torch CPU storage; explicitly facaded CUDA graphs, events, native math and packing leaves."""
import ast
import contextlib
import copy
import io
import json
import unittest
import argparse
from pathlib import Path
from unittest.mock import patch

import torch

import expert_top100_moe_harness as harness
import expert_top100_moe_harness_read as reader
from expert_top100_moe_harness_contract import BANKS, DISTRIBUTIONS, LOADS, decision, paired_samples
from expert_top100_moe_test import HostTensor, TorchFacade, prepare_small


class HostRuntime(TorchFacade):
    def __init__(self, namespace, fault=None):
        super().__init__()
        self.namespace, self.fault = namespace, fault
        self.active = None
        self.graphs, self.states, self.builds, self.orders = [], [], [], []
        runtime = self

        class Graph:
            def __init__(self):
                self.calls = []
                self.reset_done = False
                runtime.graphs.append(self)

            def replay(self):
                runtime.orders.append(runtime.graphs.index(self) % 4)
                for candidate, arguments in self.calls:
                    runtime.launch(candidate, arguments, replay=True)

            def reset(self):
                self.reset_done = True

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 1.0

        @contextlib.contextmanager
        def graph_context(graph):
            self.active = graph
            try:
                yield
            finally:
                self.active = None

        self.cuda.CUDAGraph = Graph
        self.cuda.graph = graph_context
        self.cuda.Event = Event

    def randn(self, *args, **kwargs):
        return torch.randn(*args, **kwargs).as_subclass(HostTensor)

    def rand(self, *args, **kwargs):
        return torch.rand(*args, **kwargs).as_subclass(HostTensor)

    def zeros(self, *args, **kwargs):
        if args == (64 * 1024 * 1024,):
            args = (64,)
        return torch.zeros(*args, **kwargs)

    def load(self, **kwargs):
        self.builds.append(kwargs)
        runtime = self
        candidate = harness.sha(kwargs['cuda_sources']) == harness.adapter.CANDIDATE_NATIVE

        class Extension:
            def okc_set_pdl(self, value):
                if value != 0:
                    raise AssertionError('PDL')

            def okc_run(self, *arguments):
                if runtime.active is not None:
                    runtime.active.calls.append((candidate, arguments))
                runtime.launch(candidate, arguments)
        return Extension()

    def launch(self, candidate, arguments, replay=False):
        weights, scratch, hidden, ids, routing, output, lo, count, nsm = arguments
        scratch[0].add_(1)
        factor = (routing * ((ids + 1) * (ids < 32)).float()).sum(-1, keepdim=True)
        if not (self.fault == 'stale' and replay):
            output.copy_((hidden.float() * factor).to(torch.bfloat16))
        if replay and candidate and self.fault == 'corrupt':
            output.fill_(float('nan'))
        if replay and candidate and self.fault == 'wrong':
            output.add_(100)
        if replay and self.fault == 'guard':
            self.states[0]['guards'][0][0][-1] = 0


def environment(source, fault=None):
    namespace = {}
    exec(compile(source, '<complete-generated-moe-harness>', 'exec'), namespace)
    runtime = HostRuntime(namespace, fault)
    namespace.update(torch=runtime, load_inline=runtime.load, _H_NATIVE_LOADER=runtime.load, H_HOLD=False)
    ownership = namespace['_h_ownership']

    def inventory(states):
        runtime.states = states
        return ownership(states)
    namespace['_h_ownership'] = inventory
    return namespace, runtime


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source, cls.manifest = harness.build()

    def run_full(self, fault=None, mutate=None):
        namespace, runtime = environment(self.source, fault)
        if mutate:
            mutate(namespace, runtime)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            state, rows = prepare_small(namespace)
        return namespace, runtime, state, output.getvalue()

    def test_complete_generated_orchestration_storage_replay_and_official_candidate(self):
        namespace, runtime, state, log = self.run_full()
        report, verdict = reader.capsule(log, self.manifest)
        self.assertFalse(verdict['component_pass'])
        self.assertIs(state, runtime.states[2])
        self.assertTrue(state['_expert_pair_arm'])
        self.assertEqual(len(runtime.builds), 7)
        self.assertEqual(len(runtime.graphs), 216)
        self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
        self.assertEqual(report['counters'], dict(observations=378, restorations=54, captures=216,
                                                 timed_replays=1152, worst=[0., 0.]))
        for arm in runtime.states:
            self.assertTrue(all(torch.count_nonzero(tensor) == 0 for tensor in arm['ma_scratch']))
        self.assertEqual(len({tensor.data_ptr() for graph in runtime.graphs[:4]
                              for tensor in graph.calls[0][1][2:5]}), 12)
        self.assertIs(namespace['load_inline'].__self__, runtime)
        first = runtime.graphs[0].calls[0][1]
        original = namespace['generate_input'](1, 'ep8', BANKS[0], device='cpu')
        for tensor, key in zip(first[2:5], namespace['_H_KEYS']):
            self.assertTrue(torch.equal(tensor, original[key]))
        data = namespace['generate_input'](32, 'ep8', 234, device='cpu')
        output = namespace['kernel_fn'](data, state)
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaisesRegex(RuntimeError, 'IDENTITY_OR_ARM'):
            namespace['kernel_fn'](data, runtime.states[0])
        self.__class__.complete_log = log

    def test_full_failures_restore_inputs_graphs_loader_and_scratch(self):
        for fault in ('stale', 'corrupt', 'wrong'):
            with self.subTest(fault=fault):
                namespace, runtime = environment(self.source, fault)
                with self.assertRaisesRegex(RuntimeError, 'paired output error|nonfinite output'):
                    prepare_small(namespace)
                self.assertTrue(runtime.states)
                self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
                self.assertTrue(all(torch.count_nonzero(tensor) == 0 for state in runtime.states for tensor in state['ma_scratch']))
                self.assertIs(namespace['load_inline'].__self__, runtime)

    def test_guard_corruption_fails_closed(self):
        namespace, runtime = environment(self.source, 'guard')
        with self.assertRaisesRegex(RuntimeError, 'SCRATCH_OVERRUN'):
            prepare_small(namespace)
        self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
        self.assertIs(namespace['load_inline'].__self__, runtime)
        self.assertTrue(all(torch.count_nonzero(tensor) == 0 for state in runtime.states for tensor in state['ma_scratch']))

    def test_full_inference_context(self):
        with torch.inference_mode():
            namespace, runtime, state, log = self.run_full()
            self.assertTrue(torch.is_inference(state['ma_w'][0]))
            data = namespace['generate_input'](33, 'ep8', 876, device='cpu')
            output = namespace['kernel_fn'](data, state)
            self.assertTrue(torch.isfinite(output).all())
            reader.capsule(log, self.manifest)

    def test_complete_entry_budget_and_ownership_fail_closed(self):
        for fault in ('budget', 'alias'):
            namespace, runtime = environment(self.source)
            ownership = namespace['_h_ownership']

            def inject(states):
                if fault == 'alias':
                    states[1]['ma_scratch'] = states[0]['ma_scratch']
                    states[1]['guards'] = states[0]['guards']
                result = ownership(states)
                if fault == 'budget':
                    namespace['time'] = type('Clock', (), {'monotonic': staticmethod(lambda: float('inf'))})
                return result
            namespace['_h_ownership'] = inject
            with self.assertRaisesRegex(RuntimeError, 'storage|PREPARE_295S'):
                prepare_small(namespace)
            self.assertIs(namespace['load_inline'].__self__, runtime)
            self.assertTrue(all(torch.count_nonzero(tensor) == 0 for state in runtime.states for tensor in state['ma_scratch']))

    def test_bad_route_and_weight_normalization_through_prepare(self):
        for fault in ('duplicate', 'negative', 'weights'):
            namespace, runtime = environment(self.source)
            generate = namespace['generate_input']

            def bad(*args, **kwargs):
                data = generate(*args, **kwargs)
                if fault == 'duplicate':
                    data['topk_ids'][:, 1] = data['topk_ids'][:, 0]
                elif fault == 'negative':
                    data['topk_ids'][0, 0] = -1
                else:
                    data['topk_weights'].zero_()
                return data
            namespace['generate_input'] = bad
            with self.assertRaisesRegex(RuntimeError, 'route|distinct'):
                prepare_small(namespace)
            self.assertTrue(all(torch.count_nonzero(tensor) == 0 for state in runtime.states for tensor in state['ma_scratch']))

    def test_target_proposal_changes_only_explicit_hold(self):
        source, manifest = harness.build(target_proposal=True)
        self.assertEqual(source, self.source.replace('H_HOLD = True\n', 'H_HOLD = False\n'))
        self.assertFalse(manifest['hold'])
        self.assertFalse(manifest['dispatch_ready'])
        self.assertFalse(manifest['gpu_qualified'])
        self.assertEqual(manifest['runtime_id'], self.manifest['runtime_id'])
        self.assertNotEqual(manifest['source_sha256'], self.manifest['source_sha256'])
        with self.assertRaises(RuntimeError):
            harness.build(target_proposal=1)

    def test_deterministic_pins_hold_and_compiler_rejection(self):
        self.assertEqual((self.source, self.manifest), harness.build())
        self.assertLess(len(self.source.encode()), 1_000_000)
        self.assertTrue(self.source.startswith(harness.adapter.build()[0]))
        delivered = harness.ROOT / 'docs/steering/evidence/2026-11-02-top100-moe-harness/payload.py'
        self.assertEqual(delivered.read_bytes(), self.source.encode())
        namespace, runtime = environment(self.source)
        namespace['H_HOLD'] = True
        with self.assertRaisesRegex(RuntimeError, 'HOLD'):
            namespace['prepare']({})
        self.assertFalse(runtime.builds)
        namespace['H_HOLD'] = False
        namespace['H_BUILD_CALLS'] = []
        with self.assertRaisesRegex(RuntimeError, 'compiler'):
            prepare_small(namespace)
        self.assertFalse(runtime.builds)

    def test_exact_extracted_routing_and_capacity(self):
        namespace, runtime = environment(self.source)
        reference = (harness.ROOT / 'muse_moe_w4a16/reference.py').read_text()
        nvfp4 = (harness.ROOT / 'muse_moe_w4a16/nvfp4.py').read_text()
        original = dict(torch=torch, D_HIDDEN=4096, D_EXPERT=4096, N_EXPERTS=256,
                        ROUTED_PER_TOKEN=8, POPULARITY=.03, LOCAL={'ep8': (0, 32)})
        exec(harness.extract(nvfp4, 'outlier_channels'), original)
        exec(harness.extract(reference, 'generate_input'), original)
        for seed in BANKS:
            for dist in DISTRIBUTIONS:
                for tokens in namespace['TOKENS']:
                    actual = namespace['generate_input'](tokens, 'ep8', seed, dist=dist, device='cpu')
                    expected = original['generate_input'](tokens, 'ep8', seed, dist=dist, device='cpu')
                    for key in namespace['_H_KEYS']:
                        self.assertTrue(torch.equal(actual[key], expected[key]))
                    namespace['_h_routes'](actual)
                    groups = torch.bincount(actual['topk_ids'].flatten().long() // 32, minlength=8)
                    self.assertEqual(groups[0], groups.max())

    def test_near_zero_and_row_errors(self):
        namespace, runtime = environment(self.source)
        reference = torch.ones((4, 4096), dtype=torch.bfloat16)
        reference[0].zero_()
        actual = reference.clone()
        actual[0, 0] = 1e-8
        with self.assertRaisesRegex(RuntimeError, 'zero-reference'):
            namespace['_h_errors'](actual, reference)
        reference.fill_(1e-9)
        self.assertEqual(namespace['_h_errors'](reference.clone(), reference), [0., 0.])
        with self.assertRaisesRegex(RuntimeError, 'paired'):
            namespace['_h_errors'](reference * 2, reference)

    def test_frozen_equal_six_cross_null_regression_gates(self):
        rows = [[seed, tokens, dist, [100., 100., 96., 96.], [tokens, 1, tokens], [100 / 96] * 4, [0., 0.]]
                for seed in BANKS for tokens in LOADS for dist in DISTRIBUTIONS]
        self.assertTrue(decision(rows)['component_pass'])
        for mutate in ('cross', 'null', 'regress', 'nan', 'missing'):
            bad = copy.deepcopy(rows)
            if mutate == 'cross':
                for row in bad:
                    row[5][3] = 1.
            elif mutate == 'null':
                for row in bad:
                    row[6][0] = .03
            elif mutate == 'regress':
                next(row for row in bad if row[1] == 32)[5][3] = 1 / 1.01
            elif mutate == 'nan':
                bad[0][3][0] = float('nan')
            else:
                bad.pop()
            if mutate in ('nan', 'missing'):
                with self.assertRaises(RuntimeError):
                    decision(bad)
            else:
                self.assertFalse(decision(bad)['component_pass'])
        times, ratios, nulls = paired_samples([[100., 200.] * 4, [100., 200.] * 4,
                                               [90., 180.] * 4, [90., 180.] * 4])
        self.assertEqual(times, [150., 150., 135., 135.])
        self.assertTrue(all(abs(value - 100 / 90) < 1e-12 for value in ratios))
        self.assertEqual(nulls, [0., 0.])

    def test_receipts_actual_schema_and_negative_associations(self):
        raw = json.loads((harness.ROOT / 'muse_moe_w4a16/profiles/sc6_compare.05e51ddeb7014dd9.receipt.json').read_text())
        log = getattr(self.__class__, 'complete_log', None)
        if log is None:
            log = self.run_full()[3]
        source = self.source.encode()
        job = raw['id']
        upload = dict(job=job, team='yelgabo', mode='check', board=raw['board'], content_hash=raw['content_hash'],
                      payload_ref=raw['payload_ref'], source_sha256=harness.sha(source), source_bytes=len(source))
        attest = dict(job=job, owning_log=raw['output']['log'], sha256=harness.sha(log), source_sha256=harness.sha(source))
        result = reader.read(raw, job, log, self.manifest, source, upload, attest)
        self.assertFalse(result['target_qualified'])
        target, target_manifest = harness.build(target_proposal=True)
        target_upload = dict(upload, source_sha256=harness.sha(target), source_bytes=len(target.encode()))
        target_attest = dict(attest, source_sha256=harness.sha(target))
        target_result = reader.read(raw, job, log, target_manifest, target.encode(), target_upload, target_attest)
        self.assertFalse(target_result['target_qualified'])  # synthetic null still fails
        with self.assertRaises(RuntimeError):
            reader.read(raw, job, log, target_manifest, source, target_upload, target_attest)
        with self.assertRaisesRegex(RuntimeError, 'boolean hold'):
            reader.read(raw, job, log, dict(target_manifest, hold=0), target.encode(), target_upload, target_attest)
        for field, value in (('team', 'other'), ('id', '0' * 16), ('mode', 'submit'), ('board', 'muse_decode_layer'),
                             ('payload_ref', 'wrong'), ('status', 'running')):
            wrong = copy.deepcopy(raw)
            wrong[field] = value
            with self.assertRaises(RuntimeError):
                reader.read(wrong, job, log, self.manifest, source, upload, attest)
        for event in ('test', 'benchmark', 'profile'):
            wrong = copy.deepcopy(raw)
            wrong['result']['records'].remove(next(row for row in wrong['result']['records'] if row['event'] == event))
            with self.assertRaises(RuntimeError):
                reader.read(wrong, job, log, self.manifest, source, upload, attest)
        for bad in (log + log, log.replace('candidate2', 'control'), log.replace('378', '377'), 'EXPERT_MOE_CAPSULE ' + 'x' * 14000):
            with self.assertRaises((RuntimeError, ValueError)):
                reader.capsule(bad, self.manifest)
        for bad in ('{"a":1,"a":2}', '{"x":NaN}'):
            with self.assertRaises(RuntimeError):
                reader.strict_json(bad)
        wrong = dict(upload, source_sha256='0' * 64)
        with self.assertRaises(RuntimeError):
            reader.read(raw, job, log, self.manifest, source, wrong, attest)
        for mutate in ('profile', 'duplicate-index', 'time', 'official-fail'):
            wrong = copy.deepcopy(raw)
            if mutate == 'profile':
                next(row for row in wrong['result']['records'] if row['event'] == 'profile')['case']['seed'] += 1
            elif mutate == 'duplicate-index':
                next(row for row in wrong['result']['records'] if row['event'] == 'test')['index'] = 1
            elif mutate == 'time':
                next(row for row in wrong['result']['records'] if row['event'] == 'benchmark')['median_ns'] = 0
            else:
                wrong['result']['passed'] = False
            with self.assertRaises(RuntimeError):
                reader.read(wrong, job, log, self.manifest, source, upload, attest)
        with self.assertRaises(RuntimeError):
            reader.read(raw, job, log, self.manifest, source + b'\n', upload, attest)
        with self.assertRaises(RuntimeError):
            reader.read(raw, job, log, self.manifest, source, upload, dict(attest, sha256='0' * 64))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path)
    args, remaining = parser.parse_known_args()
    program = unittest.main(argv=['harness-tests', *remaining], verbosity=2, exit=False)
    if args.evidence:
        args.evidence.mkdir(parents=True, exist_ok=True)
        log = getattr(HarnessTests, 'complete_log', '')
        args.evidence.joinpath('host-facade-capsule.log').write_bytes(log.encode())
        receipt_path = harness.ROOT / 'muse_moe_w4a16/profiles/sc6_compare.05e51ddeb7014dd9.receipt.json'
        raw = json.loads(receipt_path.read_text())
        receipt = dict(tests=program.result.testsRun, passed=program.result.wasSuccessful(), torch=torch.__version__,
                       payload_sha256=HarnessTests.manifest['source_sha256'], runtime_id=HarnessTests.manifest['runtime_id'],
                       capsule_sha256=harness.sha(log), dispatch_ready=False, gpu_qualified=False,
                       real='CPU Tensor storage, aliases, clones, input-generation math, host dispatch, guard/reset and checks',
                       facaded='Native MoE math/builds, CUDA device/graph/event operations, cache flush size, bank/packing/scratch leaves',
                       receipt_fixture='Historical receipt schema plus fabricated upload/log association ONLY; not candidate evidence',
                       historical_schema=dict(job=raw['id'], sha256=harness.sha(receipt_path.read_bytes()),
                            events={event: sum(row['event'] == event for row in raw['result']['records'])
                                    for event in ('setup', 'test', 'benchmark', 'profile')}))
        args.evidence.joinpath('host-receipt.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
