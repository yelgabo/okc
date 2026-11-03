import copy
import math
import unittest

from score_oracle import ARMS, SEEDS, TOKENS, evaluate


def fixture():
    return [dict(seed=seed, tokens=tokens, context=context, times_us=[100.0] * 4)
            for seed in SEEDS for tokens in TOKENS for context in ('sliding', 'full')]


class OracleTests(unittest.TestCase):
    def winning(self):
        rows = fixture()
        for row in rows:
            if row['tokens'] in (1, 8):
                row['times_us'][3] = 95.0
        return rows

    def test_short_win_and_six_load_dilution(self):
        result = evaluate(self.winning())
        self.assertTrue(result['arithmetic_gate'])
        for bank in result['banks']:
            self.assertAlmostEqual(bank['net_log_gains'][0], math.log(100 / 95) / 3)
            self.assertEqual(bank['adapter_off_log_gains'], [0.0, 0.0])

    def test_isolated_adapter_win_is_not_net_win(self):
        rows = fixture()
        for row in rows:
            if row['tokens'] in (1, 8):
                row['times_us'][2:] = [120.0, 110.0]
        result = evaluate(rows)
        self.assertFalse(result['arithmetic_gate'])
        self.assertGreater(result['banks'][0]['isolated_pdl_log_gain'], math.log(1.01))

    def test_duplicate_noise_raises_threshold(self):
        rows = self.winning()
        for row in rows:
            row['times_us'][1] = 99.0
        result = evaluate(rows)
        self.assertFalse(result['arithmetic_gate'])
        self.assertAlmostEqual(result['banks'][0]['duplicate_log_null'], math.log(100 / 99))

    def test_context_weights_are_46_to_16_before_log(self):
        rows = fixture()
        for row in rows:
            if row['tokens'] == 1:
                row['times_us'][3] = 90.0 if row['context'] == 'sliding' else 80.0
        expected = math.log((62 * 100) / (46 * 90 + 16 * 80)) / 6
        self.assertAlmostEqual(evaluate(rows)['banks'][0]['net_log_gains'][0], expected)

    def test_null_cannot_cancel_across_loads(self):
        rows = self.winning()
        for row in rows:
            if row['tokens'] == 16:
                row['times_us'][1] = 110.0
            if row['tokens'] == 32:
                row['times_us'][1] = 100 / 1.1
        expected = 2 * math.log(1.1) / 6
        self.assertAlmostEqual(evaluate(rows)['banks'][0]['duplicate_log_null'], expected)

    def test_all_banks_required(self):
        rows = self.winning()
        for row in rows:
            if row['seed'] == SEEDS[-1]:
                row['times_us'][3] = 100.0
        self.assertFalse(evaluate(rows)['arithmetic_gate'])

    def test_context_regression_not_hidden_by_weighted_win(self):
        rows = self.winning()
        for row in rows:
            if row['tokens'] == 1:
                row['times_us'][3] = 80.0 if row['context'] == 'sliding' else 102.0
        result = evaluate(rows)
        self.assertGreater(result['banks'][0]['net_log_gains'][0], math.log(1.01))
        self.assertFalse(result['arithmetic_gate'])

    def test_unchanged_adapter_not_required_to_win(self):
        self.assertTrue(evaluate(self.winning())['arithmetic_gate'])
        self.assertFalse(evaluate(fixture())['arithmetic_gate'])

    def test_subthreshold_short_improvement(self):
        rows = self.winning()
        for row in rows:
            if row['tokens'] in (1, 8):
                row['times_us'][3] = 99.0
        self.assertFalse(evaluate(rows)['arithmetic_gate'])

    def test_mapping_and_coverage_rejected(self):
        rows = fixture()
        with self.assertRaisesRegex(AssertionError, 'mapping'):
            evaluate(rows, (ARMS[0], ARMS[1], ARMS[3], ARMS[2]))
        for bad in (rows[:-1], rows + [copy.deepcopy(rows[0])], [copy.deepcopy(rows[0])] * 36):
            with self.assertRaises(AssertionError):
                evaluate(bad)

    def test_invalid_times_rejected(self):
        for value in (True, 0, -1, float('nan'), float('inf')):
            rows = fixture()
            rows[0]['times_us'][3] = value
            with self.assertRaisesRegex(AssertionError, 'invalid time'):
                evaluate(rows)


if __name__ == '__main__':
    unittest.main(verbosity=2)
