"""Decision tests use complete paired context/load records."""
import unittest
from d32_attn_worker_contract import decision


def rows():
    return [dict(tokens=t, context=c, seed=s,
                 times_us=([100., 100.1, 97., 97.1] if t == 32 and c == 'full'
                           else [100., 100.1, 100., 100.1]))
            for t in (1, 8, 16, 32, 128, 256)
            for c in ('sliding', 'full') for s in (1102031, 1102037, 1102049)]


class DecisionTests(unittest.TestCase):
    def test_full_only_improvement_can_pass(self):
        self.assertTrue(decision(rows())['passed'])

    def test_noise_regression_and_wrong_coverage_fail(self):
        noisy = rows()
        noisy[0]['times_us'] = [100., 102., 100., 100.1]
        self.assertFalse(decision(noisy)['passed'])
        regression = rows()
        regression[0]['times_us'] = [100., 100.1, 104., 104.1]
        self.assertFalse(decision(regression)['passed'])
        with self.assertRaises(ValueError):
            decision(rows()[:-1])


if __name__ == '__main__':
    unittest.main()
