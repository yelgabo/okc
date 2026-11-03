"""Check independent Torch storage and replay orchestration under a CPU CUDA facade."""
import ast
import base64
import contextlib
import io
import json
import unittest
import zlib

import torch

import expert_top100_moe_adapter as adapter
import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small
from moe_factor_y_harness import build


class Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source,cls.manifest=build()

    def environment(self,fault=None):
        ns,runtime=old.environment(self.source,fault);native=runtime.launch
        def launch(candidate,args,replay=False):
            candidate=any(args[0] is state['ma_w'] and state['_expert_pair_arm'] for state in runtime.states)
            native(candidate,args,replay)
        runtime.launch=launch
        return ns,runtime

    def test_packing_and_preparation_functions_stay_original(self):
        prefix,_=adapter.build()
        def functions(source):return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        control=functions(prefix);candidate=functions(self.source)
        for name in ('_pack_codes_nat','_wgs_copy','_fork_native_ma_prepare'):
            self.assertEqual(candidate[name],control[name])

    def test_complete_ownership_replays_and_original_weights(self):
        ns,runtime=self.environment();log=io.StringIO()
        with contextlib.redirect_stdout(log):state,_=prepare_small(ns)
        report=json.loads(next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('EXPERT_MOE_CAPSULE ')))
        self.assertEqual(report['table_words'],[256]*4)
        self.assertEqual(report['loaded_builds'],list(range(7)))
        self.assertEqual(report['counters'],dict(observations=378,restorations=54,captures=216,timed_replays=1152,worst=[0.,0.]))
        self.assertEqual(len(report['rows']),36)
        self.assertIs(state,runtime.states[2])
        self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
        for owner in runtime.states:
            for actual,control in zip(owner['ma_w'],runtime.states[0]['ma_w']):self.assertTrue(torch.equal(actual,control))
        self.assertFalse(report['factor_y_decision']['passed'])
        self.assertLess(len(log.getvalue()),16000)

    def test_failed_math_and_guard_preserve_failure(self):
        for fault in ('stale','corrupt','guard'):
            ns,runtime=self.environment(fault);log=io.StringIO()
            with self.subTest(fault=fault),contextlib.redirect_stdout(log):
                with self.assertRaisesRegex(RuntimeError,'paired output error|nonfinite output|SCRATCH_OVERRUN'):prepare_small(ns)
            partial=json.loads(zlib.decompress(base64.b64decode(next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_FACTOR_Y_PARTIAL ')))))
            self.assertEqual(partial['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
            self.assertTrue(all(graph.reset_done for graph in runtime.graphs))

    def test_final_serialization_failure_retains_complete_rows(self):
        ns,runtime=self.environment();log=io.StringIO()
        def fail(report):raise RuntimeError('injected serialization')
        ns['_fy_encode']=fail
        with contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(RuntimeError,'injected serialization'):prepare_small(ns)
        partial=json.loads(zlib.decompress(base64.b64decode(next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_FACTOR_Y_PARTIAL ')))))
        self.assertEqual(len(partial['rows']),36)
        self.assertEqual(partial['counters']['timed_replays'],1152)
        self.assertTrue(all(graph.reset_done for graph in runtime.graphs))


if __name__=='__main__':unittest.main()
