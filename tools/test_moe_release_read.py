"""Exercise release gates with a synthetic association over an archived result schema."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from moe_release_read import validate


class ReleaseRead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.receipt=json.loads((root/'docs/steering/evidence/2026-11-02-moe-bqueue/status-wait.raw.json').read_text())
        cls.payload=b'# synthetic source association for reader tests\n'
        sha=hashlib.sha256(cls.payload).hexdigest()
        cls.accepted=dict(job_id=cls.receipt['id'],source_sha256=sha)
        cls.manifest=dict(source_sha256=sha)

    def test_complete_official_point_is_a_release_lead_not_submission_authority(self):
        data=validate(self.receipt,self.accepted,self.manifest,self.payload)
        self.assertTrue(data['material_release_lead'])
        self.assertFalse(data['public_submission_approved'])
        self.assertEqual(len(data['loads']),6)
        self.assertFalse(data['target_met'])

    def test_failed_result_stays_failed(self):
        receipt=copy.deepcopy(self.receipt)
        receipt['result'].update(passed=False,returncode=112,score_us=None,speedup=None,speedup_ci=None)
        data=validate(receipt,self.accepted,self.manifest,self.payload)
        self.assertFalse(data['passed'])
        self.assertFalse(data['material_release_lead'])

    def test_incomplete_wrong_source_or_owner_rejected(self):
        for fault in ('owner','mode','test','benchmark','profile','payload'):
            receipt=copy.deepcopy(self.receipt);payload=self.payload
            if fault=='owner':receipt['id']='another-job'
            elif fault=='mode':receipt['mode']='submit'
            elif fault=='payload':payload=b'changed source'
            else:
                event={'test':'test','benchmark':'benchmark','profile':'profile'}[fault]
                index=next(i for i,x in enumerate(receipt['result']['records']) if x.get('event')==event)
                receipt['result']['records'].pop(index)
            with self.subTest(fault=fault):
                with self.assertRaises(ValueError):validate(receipt,self.accepted,self.manifest,payload)


if __name__=='__main__':unittest.main()
