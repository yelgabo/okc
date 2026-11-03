"""Exercise failed-receipt identity and partial-row retention with synthetic receipts."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from gopt_ws_owned import build
from gopt_ws_owned_read import partial, read


class PartialReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.manifest, _ = build()

    def test_failure_stays_failed_and_wrong_identity_is_rejected(self):
        job = '0123456789abcdef'
        report = dict(runtime_id=self.manifest['runtime_id'],status='FAILED_PARTIAL_NOT_QUALIFIED',
                      rows=[[1102031,1,'normal',[30.,30.,31.,31.],[2,2,1],[.97,.97,.97,.97],[.001,.001]]],error='injected',counters={})
        blob = base64.b64encode(zlib.compress(json.dumps(report).encode())).decode()
        receipt = dict(id=job,team='yelgabo',mode='check',board='muse_moe_w4a16',status='done',
                       content_hash='a'*64,payload_ref='muse_moe_w4a16/'+'a'*64+'.py',
                       output=dict(log='profiles/paired.'+job+'.log'),
                       result=dict(passed=False,returncode=112,score_us=None,error='injected',flags=[],
                                   log_tail='GOPT_WS_OWN_PARTIAL '+blob))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'paired.py').write_text(self.source)
            (folder/'paired.manifest.json').write_text(json.dumps(self.manifest))
            (folder/'accepted.json').write_text(json.dumps(dict(job_id=job,source_sha256=self.manifest['source_sha256'])))
            def save(): (folder/'status-wait.raw.json').write_text(json.dumps(receipt))
            save()
            with patch('gopt_ws_owned_read.build',return_value=(self.source,self.manifest,{})):
                result=partial(folder,job)
                self.assertFalse(result['qualified'])
                self.assertEqual(result['report']['rows'],report['rows'])
                self.assertEqual(result['receipt_returncode'],112)
                receipt['id']='fedcba9876543210';save()
                with self.assertRaisesRegex(RuntimeError,'own terminal'): partial(folder,job)
                receipt['id']=job;receipt['payload_ref']='muse_moe_w4a16/'+'b'*64+'.py';save()
                with self.assertRaisesRegex(RuntimeError,'payload association'): partial(folder,job)
                receipt['payload_ref']='muse_moe_w4a16/'+'a'*64+'.py';receipt['result']['passed']=True;save()
                with self.assertRaisesRegex(RuntimeError,'retain failed'): partial(folder,job)

    def test_full_reader_binds_synthetic_receipt_to_payload_and_arm(self):
        from gopt_ws_owned import ROOT, contract_source
        archived=ROOT/'docs/steering/evidence/2026-11-02-gopt-ws-split'
        receipt=json.loads((archived/'status-wait.raw.json').read_text())
        report=json.loads((archived/'analysis.json').read_text())['report']
        report['runtime_id']=self.manifest['runtime_id']
        report['counters'].pop('finalizer_reference_cases')
        report['counters']['fp32_reference_cases']=8
        ns={};exec(contract_source(),ns)
        report['component_pass']=ns['decision'](report['rows'])['component_pass']
        job='0123456789abcdef'
        receipt['id']=job
        receipt['output']['log']='profiles/paired.'+job+'.log'
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'profiles').mkdir()
            (folder/'paired.py').write_text(self.source)
            (folder/'paired.manifest.json').write_text(json.dumps(self.manifest))
            (folder/'accepted.json').write_text(json.dumps(dict(job_id=job,source_sha256=self.manifest['source_sha256'])))
            (folder/'status-wait.raw.json').write_text(json.dumps(receipt))
            def save_report():
                blob=base64.b64encode(zlib.compress(json.dumps(report).encode())).decode()
                (folder/receipt['output']['log']).write_text('GOPT_WS_OWN_REPORT '+blob+'\n')
            save_report()
            result=read(folder,job)
            self.assertFalse(result['target_qualified'])
            self.assertFalse(result['decision']['component_pass'])
            report['official_arm']='control';save_report()
            with self.assertRaisesRegex(RuntimeError,'identity/arm'):read(folder,job)
            report['official_arm']='candidate2';save_report()
            (folder/'paired.py').write_text(self.source+'\n')
            with self.assertRaisesRegex(RuntimeError,'accepted job/source'):read(folder,job)


if __name__ == '__main__': unittest.main()
