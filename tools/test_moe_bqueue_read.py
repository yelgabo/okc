"""Check strict receipt binding with synthetic capsules on an archived schema."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import expert_top100_moe_harness_read as reader
from moe_bqueue_harness import ROOT, build
from expert_top100_moe_harness_contract import BANKS, LOADS, DISTRIBUTIONS


class Receipt(unittest.TestCase):
    def test_complete_binding_and_incomplete_rejection(self):
        source, manifest = build()
        rows = [[seed, tokens, dist, [100., 100., 95., 95.], [tokens, 1, tokens],
                 [100 / 95] * 4, [0., 0.]]
                for seed in BANKS for tokens in LOADS for dist in DISTRIBUTIONS]
        report = dict(runtime_id=manifest['runtime_id'], official_arm='candidate2',
                      placement_arms=[False, False, True, True], cleanup_restored=True,
                      shared_tensor_storage=[], owners_bytes=[1024]*4, prepare_s=100.,
                      counters=dict(observations=378,restorations=54,captures=216,
                                    timed_replays=1152,worst=[0.,0.]),rows=rows,component_pass=True)
        receipt = json.loads((ROOT/'docs/steering/evidence/2026-11-02-cuda-group-scale/status-wait.raw.json').read_text())
        job = receipt['id']
        def run(data, job_override=None):
            log = 'EXPERT_MOE_CAPSULE '+json.dumps(data)+'\n'
            owning = receipt['output']['log']
            upload = dict(job=job,team='yelgabo',mode='check',board='muse_moe_w4a16',
                          content_hash=receipt['content_hash'],payload_ref=receipt['payload_ref'],
                          source_sha256=manifest['source_sha256'],source_bytes=len(source.encode()))
            attestation = dict(job=job,owning_log=owning,sha256=reader.sha(log),source_sha256=reader.sha(source))
            with patch.object(reader,'build',build):
                return reader.read(receipt,job_override or job,log,manifest,source.encode(),upload,attestation)
        self.assertTrue(run(report)['target_qualified'])
        for mutation in ('coverage','runtime','timing','job'):
            bad=copy.deepcopy(report)
            if mutation=='coverage':bad['counters']['captures']=215
            elif mutation=='runtime':bad['runtime_id']='0'*64
            elif mutation=='timing':bad['rows'].pop()
            with self.assertRaises(RuntimeError):run(bad,'0'*16 if mutation=='job' else None)


if __name__=='__main__':unittest.main()
