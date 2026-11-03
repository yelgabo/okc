"""Check affected-load gates and the generated four-state entry point."""
import contextlib
import io
import json
import unittest

import torch

from expert_top100_moe_harness_test import HostRuntime
from expert_top100_moe_test import prepare_small
from gopt_skinny_depth import build, contract_source


class DepthRuntime(HostRuntime):
    def load(self, **kwargs):
        self.builds.append(kwargs)
        candidate = '-DOKC_NS=4' in kwargs['extra_cuda_cflags']
        runtime = self

        class Extension:
            def okc_set_pdl(self, value):
                assert value == 0

            def okc_run(self, *arguments):
                if runtime.active is not None:
                    runtime.active.calls.append((candidate, arguments))
                runtime.launch(candidate, arguments)
        return Extension()


class SkinnyDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source, cls.manifest, _, _ = build()

    def test_complete_generated_entry_and_corrupt_candidate(self):
        for fault in (None, 'corrupt'):
            with self.subTest(fault=fault):
                ns = {}
                exec(compile(self.source, '<generated-depth-pair>', 'exec'), ns)
                runtime = DepthRuntime(ns, fault)
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
                self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
                self.assertTrue(all(torch.count_nonzero(tensor) == 0
                                    for state in runtime.states for tensor in state['ma_scratch']))
                if fault is None:
                    self.assertIs(state, runtime.states[2])
                    self.assertEqual(len(runtime.builds), 8)
                    self.assertEqual(sum('-DOKC_NS=4' in call['extra_cuda_cflags']
                                         for call in runtime.builds), 3)
                    report = json.loads(log.getvalue().split('EXPERT_MOE_CAPSULE ')[1])
                    self.assertEqual(report['runtime_id'], self.manifest['runtime_id'])
                    self.assertEqual(report['counters'], dict(observations=378, restorations=54,
                                                              captures=216, timed_replays=1152, worst=[0., 0.]))
                    self.assertFalse(report['component_pass'])

    def test_gate_uses_small_loads_and_rejects_unchanged_drift(self):
        ns = {}
        exec(contract_source(), ns)
        rows = [[seed, tokens, dist, [100., 100., 98., 98.], [tokens, 1, tokens],
                 [1.02 if tokens < 32 else 1.] * 4, [.001, .001]]
                for seed in ns['BANKS'] for tokens in ns['LOADS'] for dist in ns['DISTRIBUTIONS']]
        self.assertTrue(ns['decision'](rows)['component_pass'])
        rows[0][5] = [.98] * 4
        self.assertFalse(ns['decision'](rows)['component_pass'])
        rows[0][5] = [1.02] * 4
        next(row for row in rows if row[1] == 128)[5] = [1.03] * 4
        verdict = ns['decision'](rows)
        self.assertFalse(verdict['component_pass'])
        self.assertTrue(verdict['unresolved_drift'])


if __name__ == '__main__':
    unittest.main()
