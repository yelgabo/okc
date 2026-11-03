"""Check publication partitions, FTZ behavior and the private harness entry."""
import contextlib
import io
import json
import math
import random
import struct
import unittest
from collections import Counter

import torch

from expert_top100_moe_harness_test import HostRuntime
from expert_top100_moe_test import prepare_small
from gopt_ws_norm import build, contract_source


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def ftz(value):
    return math.copysign(0., value) if abs(value) < 2. ** -126 else value


def add_ftz(left, right):
    return ftz(f32(ftz(left) + ftz(right)))


class NormRuntime(HostRuntime):
    def load(self, **kwargs):
        self.builds.append(kwargs)
        candidate = '-DOKC_WS_NORM=1' in kwargs['extra_cuda_cflags']
        runtime = self

        class Extension:
            def okc_set_pdl(self, value):
                assert value == 0

            def okc_run(self, *arguments):
                if runtime.active is not None:
                    runtime.active.calls.append((candidate, arguments))
                runtime.launch(candidate, arguments)
        return Extension()


class NormAggregation(unittest.TestCase):
    def test_ftz_counterexample_and_nonfinite(self):
        tiny = struct.unpack('<f', struct.pack('<I', 0x00400000))[0]
        self.assertEqual(bits(add_ftz(add_ftz(0., tiny), tiny)), 0)
        self.assertEqual(bits(f32(tiny + tiny)), 0x00800000)
        self.assertEqual(bits(add_ftz(-tiny, -tiny)), 0x80000000)
        self.assertEqual(add_ftz(0., float('inf')), float('inf'))
        self.assertTrue(math.isnan(add_ftz(1., float('nan'))))
        self.assertTrue(math.isnan(add_ftz(float('inf'), -float('inf'))))

    def test_nonnegative_regrouping_bound(self):
        rng = random.Random(314159)
        worst = 0.
        for case in range(2048):
            exponent = rng.randrange(-145, 100)
            values = [f32(math.ldexp(rng.random(), exponent + rng.randrange(-20, 2))) for _ in range(256)]
            exact = math.fsum(ftz(v) for v in values)
            ordered = 0.
            for v in values:
                ordered = add_ftz(ordered, v)
            grouped = 0.
            for begin in range(0, 256, 17):
                acc = 0.
                for v in values[begin:begin+17]:
                    acc = add_ftz(acc, v)
                grouped = add_ftz(grouped, acc)
            error = abs(grouped - ordered) / max(exact, 2. ** -126)
            worst = max(worst, error)
            self.assertLessEqual(error, 2 * 256 * 2. ** -24)
        print('FTZ_MODEL', json.dumps(dict(cases=2048, contributions=524288, worst_relative_difference=worst)), flush=True)

    def test_partition_flushes_no_lost_or_cross_job_contributions(self):
        cases = 0
        for jobs in range(1, 45):
            for tail in (1, 2):
                coverage = Counter()
                for cta in range(132):
                    begin, end = jobs*256*cta//132, jobs*256*(cta+1)//132
                    helper_end = end - min(tail, end-begin)
                    pending = []
                    for v in range(begin, helper_end):
                        pending.append(v)
                        if v+1 == helper_end or (v+1) % 256 == 0:
                            self.assertEqual(len({u//256 for u in pending}), 1)
                            coverage.update(pending)
                            pending = []
                    self.assertFalse(pending)
                    coverage.update(range(helper_end, end))
                self.assertEqual(coverage, Counter(range(jobs*256)))
                cases += 1
        print('PARTITION_MODEL', cases, 'passed', flush=True)

    def test_generated_entry_and_negative_replay(self):
        torch.set_num_threads(2)
        source, manifest, _, _ = build()
        for fault in (None, 'corrupt'):
            ns = {}
            exec(compile(source, '<ws-norm-pair>', 'exec'), ns)
            runtime = NormRuntime(ns, fault)
            ns.update(torch=runtime, load_inline=runtime.load, _H_NATIVE_LOADER=runtime.load)
            inventory = ns['_h_ownership']

            def ownership(states):
                runtime.states = states
                return inventory(states)
            ns['_h_ownership'] = ownership
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                if fault:
                    with self.assertRaisesRegex(RuntimeError, 'nonfinite output'):
                        prepare_small(ns)
                else:
                    state, _ = prepare_small(ns)
                    self.assertIs(state, runtime.states[2])
            if fault is not None:
                self.assertIn('"stage":"failed"', log.getvalue())
            if fault is None:
                report = json.loads(log.getvalue().split('EXPERT_MOE_CAPSULE ')[1])
                self.assertEqual(report['runtime_id'], manifest['runtime_id'])
                self.assertEqual(len(runtime.builds), 7)
                self.assertEqual(sum('-DOKC_WS_NORM=1' in call['extra_cuda_cflags'] for call in runtime.builds), 2)
                self.assertEqual(report['counters']['timed_replays'], 1152)
            self.assertTrue(all(graph.reset_done for graph in runtime.graphs))

    def test_affected_gates_and_unchanged_drift(self):
        ns = {}
        exec(contract_source(), ns)
        rows = [[seed, tokens, dist, [100., 100., 98., 98.], [tokens, 1, tokens],
                 [1.02 if tokens in (8, 16) else 1.] * 4, [.001, .001]]
                for seed in ns['BANKS'] for tokens in ns['LOADS'] for dist in ns['DISTRIBUTIONS']]
        self.assertTrue(ns['decision'](rows)['component_pass'])
        rows[0][5] = [1.02] * 4
        self.assertFalse(ns['decision'](rows)['component_pass'])
        rows[0][5] = [1.] * 4
        rows[2][5] = [.98] * 4
        self.assertFalse(ns['decision'](rows)['component_pass'])


if __name__ == '__main__':
    unittest.main()
