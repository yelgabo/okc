import base64
import contextlib
import io
import json
import unittest

import torch
torch.set_num_threads(2)
import zlib

import expert_top100_moe_harness_test as old
import moe_bqueue_harness as harness
from expert_top100_moe_test import prepare_small
from expert_top100_moe_harness_read import capsule


class Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.manifest = harness.build()

    def test_full_flow(self):
        ns, runtime = old.environment(self.source)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            state = prepare_small(ns)
        report, decision = capsule(log.getvalue(), self.manifest)
        self.assertEqual(report['counters']['observations'],378)
        self.assertEqual(report['counters']['timed_replays'],1152)
        self.assertEqual(len(report['rows']),36)
        self.assertTrue(all(g.reset_done for g in runtime.graphs))

    def test_partial_failure_preserved(self):
        ns, runtime = old.environment(self.source,'guard')
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(RuntimeError,'SCRATCH_OVERRUN'):
                prepare_small(ns)
        text = log.getvalue()
        encoded = next(x.split(' ',1)[1] for x in text.splitlines() if x.startswith('MOE_BQUEUE_PARTIAL '))
        data = json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertEqual(data['runtime_id'],self.manifest['runtime_id'])
        self.assertEqual(data['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
        self.assertIn('SCRATCH_OVERRUN',data['error'])
        self.assertTrue(all(g.reset_done for g in runtime.graphs))


if __name__=='__main__':unittest.main()
