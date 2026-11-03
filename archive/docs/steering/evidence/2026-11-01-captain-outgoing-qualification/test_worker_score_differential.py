"""Differential arithmetic tests, not runtime or candidate qualification."""
import copy
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from g_outgoing_pair_read import decision
from score_oracle import SEEDS, TOKENS, evaluate


def fixture():
    return [dict(seed=seed, tokens=tokens, context=context, times_us=[100.0] * 4)
            for seed in SEEDS for tokens in TOKENS for context in ('sliding', 'full')]


class WorkerDifferentialTests(unittest.TestCase):
    def compare(self, rows):
        expected = evaluate(rows)
        actual = decision(rows)
        self.assertEqual(actual['banks'], expected['banks'])
        self.assertEqual(actual['gate'], expected['arithmetic_gate'])

    def test_baseline_and_adapter_only_false_win(self):
        rows = fixture()
        self.compare(rows)
        for row in rows:
            if row['tokens'] in (1, 8):
                row['times_us'][2:] = [120.0, 110.0]
        self.compare(rows)
        self.assertFalse(decision(rows)['gate'])

    def test_deterministic_varied_inputs(self):
        generator = random.Random(20261102)
        for sample in range(400):
            rows = fixture()
            for row in rows:
                baseline = generator.uniform(1.0, 250.0)
                treatment = generator.uniform(0.80, 1.04) if row['tokens'] in (1, 8) else generator.uniform(0.999, 1.001)
                row['times_us'] = [baseline, baseline * generator.uniform(0.997, 1.003), baseline * generator.uniform(0.95, 1.20), baseline * treatment]
            with self.subTest(sample=sample):
                self.compare(rows)

    def test_winner_bank_failure_and_context_regression(self):
        rows = fixture()
        for row in rows:
            if row['tokens'] in (1, 8):
                row['times_us'][3] = 95.0
        self.compare(rows)
        self.assertTrue(decision(rows)['gate'])
        altered = copy.deepcopy(rows)
        for row in altered:
            if row['seed'] == SEEDS[-1]:
                row['times_us'][3] = 100.0
        self.compare(altered)
        self.assertFalse(decision(altered)['gate'])
        for row in rows:
            if row['tokens'] == 1:
                row['times_us'][3] = 80.0 if row['context'] == 'sliding' else 102.0
        self.compare(rows)
        self.assertFalse(decision(rows)['gate'])

    def test_invalid_times_and_coverage_fail_closed(self):
        for value in (True, 0, -1, float('nan'), float('inf')):
            rows = fixture()
            rows[0]['times_us'][3] = value
            for implementation in (evaluate, decision):
                with self.assertRaises(AssertionError):
                    implementation(rows)
        for rows in (fixture()[:-1], fixture() + fixture()[:1]):
            for implementation in (evaluate, decision):
                with self.assertRaises(AssertionError):
                    implementation(rows)


if __name__ == '__main__':
    unittest.main(verbosity=2)
