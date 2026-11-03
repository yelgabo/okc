"""Protect against losing private numerical leads during experiment decisions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release_audit import PROBLEMS, inventory, receipt_row, ledger_gaps, merge_receipt, own_jobs_index


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


class CLIExportAudit(unittest.TestCase):
    def own_job(self):
        return dict(job_id='f8a55f1043804158', problem='muse_decode_layer', mode='check',
                    flags=[], status='done', result_status='ok', passed=True,
                    score_us=10297.394859538828, submitted_at=1793744332.1344743)

    def export(self):
        case = dict(load='decode 1', layer='sliding', weight=46)
        return dict(job_id=self.own_job()['job_id'], problem='muse_decode_layer',
                    submission='/someone/else/untrusted-path.py',
                    summary=dict(passed=True, status='ok', error=None, tests=13, tests_passed=13,
                                 score_us=self.own_job()['score_us'], speedup=2.1967,
                                 speedup_ci=[2.1637, 2.2297]),
                    benchmarks=[dict(event='benchmark', index=0, case=case, status='pass')],
                    profiles=[dict(event='profile', index=0, case=case, report='kernel time')])

    def index(self, row=None):
        return own_jobs_index([self.own_job() if row is None else row])

    def test_explicit_own_index_binds_export_and_exposes_ledger_gap(self):
        row = receipt_row(self.export(), self.index())
        self.assertEqual((row['job'], row['problem'], row['mode']),
                         ('f8a55f1043804158', 'muse_decode_layer', 'check'))
        self.assertEqual(row['record_counts'], dict(test=13, benchmark=1, profile=1))
        self.assertIn('individual test records unavailable', row['record_count_basis']['test'])
        self.assertIsNone(row['finished_at'])
        self.assertIsNone(row['content_hash'])
        self.assertNotIn('returncode', row)
        self.assertEqual(ledger_gaps(dict(leads=[row]), dict(runs=[])),
                         [dict(job=row['job'], reason='missing promotion disposition')])

    def test_path_or_job_name_alone_never_establishes_ownership(self):
        data = self.export()
        data['submission'] = '/yelgabo/own/profiles/f8a55f1043804158.py'
        self.assertIsNone(receipt_row(data))
        self.assertIsNone(receipt_row(data, {}))
        other = self.own_job(); other['job_id'] = 'another-job'
        self.assertIsNone(receipt_row(data, self.index(other)))

    def test_own_listing_accepts_only_explicit_supported_envelopes(self):
        rows = [self.own_job()]
        for envelope in (rows, dict(jobs=rows), dict(data=dict(jobs=rows))):
            self.assertIn(rows[0]['job_id'], own_jobs_index(envelope))
        for envelope in ({'rows': rows}, {}, None, 'jobs'):
            with self.assertRaises(ValueError):
                own_jobs_index(envelope)
        foreign = self.own_job(); foreign['team'] = 'other-team'
        with self.assertRaisesRegex(ValueError, 'Foreign team'):
            self.index(foreign)
        conflict = self.own_job(); conflict['problem'] = 'muse_bf16_proj'
        with self.assertRaisesRegex(ValueError, 'Conflicting own-jobs'):
            own_jobs_index([self.own_job(), conflict])

    def test_unclean_or_incomplete_listing_is_not_a_clean_binding(self):
        for field, value in (('flags', ['flag']), ('flags', None), ('status', 'running'),
                             ('result_status', 'failed'), ('passed', False), ('error', 'error')):
            with self.subTest(field=field, value=value):
                job = self.own_job(); job[field] = value
                self.assertIsNone(receipt_row(self.export(), self.index(job)))

    def test_bound_export_identity_or_summary_conflicts_fail_closed(self):
        variants = []
        data = self.export(); data['problem'] = 'muse_moe_w4a16'; variants.append(data)
        for field, value in (('score_us', 1), ('score_us', float('nan')), ('score_us', True),
                             ('passed', False), ('passed', 1), ('status', 'failed'), ('error', 'error'),
                             ('tests', 0), ('tests', True), ('tests_passed', 12), ('flags', ['flag'])):
            data = self.export(); data['summary'][field] = value; variants.append(data)
        data = self.export(); del data['summary']['error']; variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, 'Conflicting own CLI export'):
                    receipt_row(data, self.index())

    def test_record_counts_indices_events_and_coverage_are_checked(self):
        variants = []
        data = self.export(); data['profiles'] = []; variants.append(data)
        data = self.export(); data['profiles'][0]['index'] = 1; variants.append(data)
        data = self.export(); data['profiles'][0]['event'] = 'test'; variants.append(data)
        data = self.export(); data['profiles'][0]['report'] = ''; variants.append(data)
        data = self.export(); data['profiles'][0]['case'] = dict(load='verify 1'); variants.append(data)
        data = self.export(); data['benchmarks'][0]['status'] = 'fail'; variants.append(data)
        data = self.export(); data['summary']['benchmarks'] = 12; variants.append(data)
        for data in variants:
            with self.assertRaisesRegex(ValueError, 'Conflicting own CLI export'):
                receipt_row(data, self.index())

    def test_raw_and_cli_merge_without_inventing_raw_metadata(self):
        cli = receipt_row(self.export(), self.index())
        raw = PromotionAudit().receipt()
        raw.update(id=cli['job'], board=cli['problem'], content_hash='actual-server-hash')
        raw['result']['score_us'] = cli['score_us']
        for first, second in ((cli, receipt_row(raw)), (receipt_row(raw), cli)):
            receipts = {}
            merge_receipt(receipts, copy.deepcopy(first), 'first.json')
            merge_receipt(receipts, copy.deepcopy(second), 'second.json')
            row = receipts[cli['job']]
            self.assertEqual(row['receipt_formats'], ['cli-export', 'raw'])
            self.assertEqual(row['content_hash'], 'actual-server-hash')
            self.assertEqual(row['receipt_format'], 'raw')
            self.assertIsNotNone(row['finished_at'])
            conflicting = copy.deepcopy(row); conflicting['content_hash'] = 'different'
            with self.assertRaisesRegex(ValueError, 'content_hash'):
                merge_receipt(receipts, conflicting, 'conflict.json')
            conflicting = copy.deepcopy(cli); conflicting['score_us'] += 1
            with self.assertRaisesRegex(ValueError, 'score_us'):
                merge_receipt(receipts, conflicting, 'conflict.json')
            conflicting = copy.deepcopy(cli); conflicting['problem'] = 'muse_bf16_proj'
            with self.assertRaisesRegex(ValueError, 'problem'):
                merge_receipt(receipts, conflicting, 'conflict.json')

    def test_inventory_discloses_exclusion_and_limited_index_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshots = root / 'boards'; snapshots.mkdir()
            for problem in PROBLEMS:
                (snapshots / (problem+'.leaderboard.json')).write_text(json.dumps(dict(rows=[
                    dict(team='yelgabo', job='public-job', value=10343.105996394634,
                         rank=1, speedup=2.18)])))
            (root / 'own-export.json').write_text(json.dumps(self.export()))
            other = self.export(); other['job_id'] = 'unlisted'
            (root / 'unlisted-export.json').write_text(json.dumps(other))
            index = root / 'own-jobs.json'
            index.write_text(json.dumps(dict(data=dict(jobs=[self.own_job()]))))
            files = b'own-export.json\nunlisted-export.json\n'
            with patch('release_audit.subprocess.check_output', return_value=files):
                excluded = inventory(root, snapshots, tracked_only=True)
                self.assertEqual(excluded['leads'], [])
                self.assertEqual(excluded['cli_export_coverage']['excluded_no_index'], 2)
                self.assertIn('CLI exports excluded', excluded['scope'])
                included = inventory(root, snapshots, tracked_only=True, own_jobs_path=index)
            self.assertEqual(len(included['leads']), 1)
            self.assertEqual(included['leads'][0]['job'], self.own_job()['job_id'])
            coverage = included['cli_export_coverage']
            self.assertEqual((coverage['indexed_jobs'], coverage['bound_clean_exports'],
                              coverage['distinct_bound_jobs'], coverage['excluded_unlisted']), (1, 1, 1, 1))
            self.assertIn('unlisted exports excluded', included['scope'])


if __name__=='__main__':unittest.main()
