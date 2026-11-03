"""Exercise the generated observer host flow with explicitly synthetic native timestamps."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
import zlib

import torch

import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small
from moe_phase_harness import build
from moe_phase_read import capsule
from test_moe_phase import synthetic_trace


class Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source,cls.manifest=build()

    def environment(self,fault=None):
        ns,runtime=old.environment(self.source,fault)
        original=runtime.launch
        def launch(candidate,args,replay=False):
            original(candidate,args,replay)
            scratch,ids,grid=args[1],args[3],args[-1]
            if len(scratch)==4:
                counts=torch.bincount(ids.flatten().long(),minlength=256)[:32]
                jobs=int(((counts+63)//64).sum())
                raw=synthetic_trace(grid,jobs)
                scratch[3][:len(raw)].copy_(torch.tensor(raw,dtype=torch.int64))
        runtime.launch=launch
        entry=ns['prepare']
        def prepared(weights):
            scratch=ns['_ma_make_scratch']
            ns['_ma_make_scratch']=lambda dev:scratch(dev)+[torch.zeros(136*24,dtype=torch.int64)]
            try:return entry(weights)
            finally:ns['_ma_make_scratch']=scratch
        ns['_phase_host_entry']=prepared
        return ns,runtime

    def test_complete_control_entry_coverage_and_bounded_report(self):
        ns,runtime=self.environment();log=io.StringIO()
        with contextlib.redirect_stdout(log):state,_=prepare_small(ns,'_phase_host_entry')
        encoded=next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PHASE_REPORT '))
        report,_=capsule(log.getvalue(),self.manifest)
        self.assertLess(len(encoded),12000)
        if os.environ.get('PHASE_HOST_LOG'):Path(os.environ['PHASE_HOST_LOG']).write_text(log.getvalue())
        self.assertEqual(report['official_arm'],'control0')
        self.assertIs(state,runtime.states[0])
        self.assertFalse(state['_expert_pair_arm'])
        self.assertEqual(len(report['rows']),36)
        self.assertEqual(len(report['phases']),18)
        self.assertEqual(report['counters']['timed_replays'],4608)
        self.assertEqual(report['counters']['captures'],216)
        self.assertEqual(report['counters']['trace_replays'],180)
        self.assertEqual(len(report['sampling']),18)
        self.assertEqual(len(report['repeated']),18)
        self.assertTrue(report['observer_valid'])
        self.assertEqual(len(runtime.builds),7)
        self.assertTrue(all(g.reset_done for g in runtime.graphs))

    def test_guard_failure_preserves_partial_and_failure(self):
        ns,runtime=self.environment('guard');log=io.StringIO()
        with contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(RuntimeError,'SCRATCH_OVERRUN'):
                prepare_small(ns,'_phase_host_entry')
        encoded=next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PHASE_PARTIAL '))
        report=json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertEqual(report['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
        self.assertEqual(report['runtime_id'],self.manifest['runtime_id'])
        self.assertIn('SCRATCH_OVERRUN',report['error'])

    def test_completed_rows_survive_every_final_failure(self):
        for fault in ('cleanup','post-cleanup-budget','serialization','final-budget'):
            with self.subTest(fault=fault):
                ns,runtime=self.environment();log=io.StringIO()
                if fault=='cleanup':
                    reset=ns['_h_reset']
                    def failed_reset(states):
                        reset(states)
                        raise RuntimeError('injected cleanup')
                    ns['_h_reset']=failed_reset
                elif fault=='serialization':
                    def failed_encode(report):raise RuntimeError('injected serialization')
                    ns['_ph_encode']=failed_encode
                else:
                    budget=ns['_expert_moe_budget']
                    def failed_budget(deadline):
                        if ns['_PH_LIVE']['stage']==fault:raise RuntimeError('injected '+fault)
                        budget(deadline)
                    ns['_expert_moe_budget']=failed_budget
                with contextlib.redirect_stdout(log):
                    with self.assertRaisesRegex(RuntimeError,'injected '+fault):
                        prepare_small(ns,'_phase_host_entry')
                self.assertNotIn('MOE_PHASE_REPORT ',log.getvalue())
                encoded=next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PHASE_PARTIAL '))
                report=json.loads(zlib.decompress(base64.b64decode(encoded)))
                self.assertLess(len(encoded),12000)
                self.assertEqual(len(report['rows']),36)
                self.assertEqual(len(report['phases']),18)
                self.assertEqual(report['counters']['timed_replays'],4608)
                self.assertEqual(report['counters']['trace_replays'],180)

    def test_cleanup_failure_preserves_primary_failure(self):
        ns,runtime=self.environment('guard');log=io.StringIO()
        def failed_reset(states):raise RuntimeError('injected cleanup')
        ns['_h_reset']=failed_reset
        with contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(RuntimeError,'SCRATCH_OVERRUN'):
                prepare_small(ns,'_phase_host_entry')
        encoded=next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PHASE_PARTIAL '))
        report=json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertIn('SCRATCH_OVERRUN',report['error'])
        self.assertIn('injected cleanup',report['cleanup_error'])


if __name__=='__main__':unittest.main()
