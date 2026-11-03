"""Protect against losing private numerical leads during experiment decisions."""
import copy
import unittest

from release_audit import receipt_row, ledger_gaps


class PromotionAudit(unittest.TestCase):
    def receipt(self):
        return dict(team='yelgabo',board='muse_moe_w4a16',status='done',id='1234567890abcdef',
                    mode='check',finished_at=1793670000,
                    result=dict(passed=True,status='ok',returncode=0,score_us=5725.7,
                                flags=[],error=None,checks=dict(ok=True),records=[],
                                log_tail='component_pass=false official control'))

    def test_failed_treatment_does_not_hide_better_official_arm(self):
        row=receipt_row(self.receipt())
        self.assertEqual(row['score_us'],5725.7)
        self.assertIn('component_pass',row['markers'])
        gaps=ledger_gaps(dict(leads=[row]),dict(runs=[]))
        self.assertEqual(gaps,[dict(job=row['job'],reason='missing promotion disposition')])

    def test_failed_or_flagged_official_run_is_excluded(self):
        for field,value in (('passed',False),('returncode',112),('flags',['flag']),('error','error')):
            data=self.receipt();data['result'][field]=value
            self.assertIsNone(receipt_row(data))

    def test_disposition_requires_owner_evidence_and_user_facing_report(self):
        row=receipt_row(self.receipt())
        entry=dict(job=row['job'],problem=row['problem'],score_us=row['score_us'],
                   owner='MoE Worker',judged_arm='official control',source_identity='own payload',release_state='qualification',reason='Exact source review',
                   evidence='receipt.json',disclosure_report='report.md')
        audit=dict(leads=[row]);self.assertEqual(ledger_gaps(audit,dict(runs=[entry])),[])
        for field in ('owner','judged_arm','source_identity','release_state','reason','evidence','disclosure_report'):
            bad=copy.deepcopy(entry);bad[field]=''
            self.assertTrue(ledger_gaps(audit,dict(runs=[bad])))
        bad=copy.deepcopy(entry);bad['score_us']=6000
        self.assertTrue(ledger_gaps(audit,dict(runs=[bad])))


if __name__=='__main__':unittest.main()
