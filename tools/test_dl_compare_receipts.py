import copy
import json
import math
from pathlib import Path
import unittest

import dl_compare_receipts as comparison

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = json.loads((ROOT / 'muse_decode_layer/profiles/scored.607eb52bdaf24160.json').read_text())
JOB = '607eb52bdaf24160'


class ReceiptComparisonTests(unittest.TestCase):
    def test_identity_control_has_no_gain(self):
        control = comparison.validate(RECEIPT, JOB, 'run')
        result = comparison.compare(control, control)
        self.assertAlmostEqual(result['adjusted_latency_factor'], 1)
        self.assertFalse(result['descriptive_screen'])
        self.assertFalse(result['performance_qualified'])
        self.assertTrue(all(row['same_seed'] for row in result['cells']))
        self.assertEqual(result['cells'][0]['control_seed'], 2594376621983479753)

    def test_latency_and_baseline_decomposition(self):
        candidate = copy.deepcopy(RECEIPT)
        for record in candidate['result']['records']:
            if record.get('event') == 'benchmark':
                record['median_ns'] /= 1.02
                record['baseline_ns'] *= 0.98
        candidate['result']['speedup'] *= 1.02 * 0.98
        result = comparison.compare(comparison.validate(RECEIPT, JOB, 'run'),
                                    comparison.validate(candidate, JOB, 'run'))
        self.assertAlmostEqual(result['adjusted_latency_factor'], 1.02)
        self.assertAlmostEqual(result['baseline_factor'], 0.98)
        self.assertAlmostEqual(result['official_score_factor'], 0.9996)
        self.assertTrue(result['descriptive_screen'])
        self.assertFalse(result['promotion_authorized'])
        self.assertAlmostEqual(result['short_relative_to_long_full_score_factor'], 1)

    def test_short_effect_is_not_long_load_drift(self):
        candidate = copy.deepcopy(RECEIPT)
        for record in candidate['result']['records']:
            if record.get('event') == 'benchmark' and record['case']['batch'] == 1:
                record['median_ns'] /= 1.06
        candidate['result']['speedup'] *= 1.06 ** (2 / 6)
        result = comparison.compare(comparison.validate(RECEIPT, JOB, 'run'),
                                    comparison.validate(candidate, JOB, 'run'))
        self.assertAlmostEqual(result['short_load_factor'], 1.06)
        self.assertAlmostEqual(result['unchanged_long_load_factor'], 1)
        self.assertAlmostEqual(result['short_only_counterfactual_factor'], 1.06 ** (2 / 6))
        self.assertAlmostEqual(result['short_relative_to_long_full_score_factor'], 1.06 ** (2 / 6))

    def test_invalid_receipts_rejected(self):
        mutations = [('wrapper_ok_but_failed', lambda receipt: receipt['result'].update(passed=False, status='ok')),
                     ('wrong_id', lambda receipt: receipt.update(id='foreign')),
                     ('wrong_mode', lambda receipt: receipt.update(mode='check')),
                     ('wrong_board', lambda receipt: receipt.update(board='muse_moe')),
                     ('wrong_team', lambda receipt: receipt.update(team='foreign')),
                     ('queued', lambda receipt: receipt.update(status='queued')),
                     ('source_association', lambda receipt: receipt.update(payload_ref='foreign.py')),
                     ('profile_seed', lambda receipt: next(record for record in receipt['result']['records']
                         if record.get('event') == 'profile')['case'].update(seed=1)),
                     ('returncode', lambda receipt: receipt['result'].update(returncode=112)),
                     ('score', lambda receipt: receipt['result'].update(speedup=10)),
                     ('missing_test', lambda receipt: receipt['result']['records'].remove(next(
                         record for record in receipt['result']['records'] if record.get('event') == 'test'))),
                     ('duplicate', lambda receipt: receipt['result']['records'].append(next(
                         record for record in receipt['result']['records'] if record.get('event') == 'benchmark'))),
                     ('negative_time', lambda receipt: next(record for record in receipt['result']['records']
                         if record.get('event') == 'benchmark').update(median_ns=-1)),
                     ('nan_time', lambda receipt: next(record for record in receipt['result']['records']
                         if record.get('event') == 'benchmark').update(median_ns=math.nan)),
                     ('weight', lambda receipt: next(record for record in receipt['result']['records']
                         if record.get('event') == 'benchmark')['case'].update(weight=1))]
        for name, mutate in mutations:
            with self.subTest(name=name):
                receipt = copy.deepcopy(RECEIPT)
                mutate(receipt)
                with self.assertRaises(ValueError):
                    comparison.validate(receipt, JOB, 'run')


if __name__ == '__main__':
    unittest.main()
