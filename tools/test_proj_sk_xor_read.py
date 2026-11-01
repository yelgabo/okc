import copy
import math
import unittest

from proj_sk_xor_read import LAYERS, M_VALUES, ROUNDS, SEEDS, TARGETS, score_model


def rows(multiplier=1.):
    result = []
    for seed in SEEDS:
        for layer in LAYERS:
            for m in M_VALUES:
                target = (layer, m) in TARGETS
                arms = ("c0", "c1", "t0", "t1", "anchor") if target else ("common",)
                samples = {arm: [100. * (multiplier if arm.startswith("t") else 1.)] * ROUNDS
                           for arm in arms}
                result.append(dict(layer=layer, m=m, seed=seed, target=target, samples_us=samples))
    return result


class ScoreGate(unittest.TestCase):
    def test_full_thirty_formula_and_no_win_on_identical_arms(self):
        equal = score_model(rows())
        self.assertEqual(equal["minimum_factor"], 1.)
        self.assertFalse(equal["keep"])
        result = score_model(rows(.8))
        # M8 has two changed operators; M16/M32 each have four. Three other
        # loads remain unchanged, including every 128/256 projection.
        expected = ((500/460) * (500/420)**2)**(1/6)
        self.assertAlmostEqual(result["minimum_factor"], expected)
        self.assertTrue(result["keep"])

    def test_one_losing_seed_cannot_be_hidden_by_pooled_gain(self):
        data = rows(.8)
        for row in data:
            if row["seed"] == SEEDS[-1] and row["target"]:
                for arm in ("t0", "t1"): row["samples_us"][arm] = [100.] * ROUNDS
        self.assertFalse(score_model(data)["keep"])

    def test_real_anchor_prevents_credit_for_slower_rebuilt_controls(self):
        data = rows(.8)
        for row in data:
            if row["target"]:
                row["samples_us"]["anchor"] = [79.] * ROUNDS
        result = score_model(data)
        self.assertFalse(result["keep"])
        self.assertTrue(result["anchor_drift"])

    def test_regression_and_duplicate_noise_reject_apparent_gain(self):
        data = rows(.8)
        target = next(row for row in data if row["target"])
        target["samples_us"]["t0"] = [101.1] * ROUNDS
        result = score_model(data)
        self.assertTrue(result["regressions"])
        self.assertFalse(result["keep"])
        noisy = rows(.8)
        for row in noisy:
            if row["target"]: row["samples_us"]["c1"] = [80.1] * ROUNDS
        self.assertFalse(score_model(noisy)["keep"])

    def test_current_offsets_apply_to_all_arms_without_changing_raw_samples(self):
        data = rows(.8)
        original = copy.deepcopy(data)
        offsets = {(layer,m): 20. for layer in LAYERS for m in M_VALUES}
        result = score_model(data, offsets)
        expected = ((400/360) * (400/320)**2)**(1/6)
        self.assertAlmostEqual(result["minimum_factor"], expected)
        self.assertEqual(data, original)
        with self.assertRaises(ValueError): score_model(data, {key:100. for key in offsets})

    def test_partial_or_malformed_evidence_is_rejected(self):
        data = rows(.8)
        bads = [data[:-1], data[:-1] + [copy.deepcopy(data[0])]]
        for change in ("target", "samples", "nan", "arm"):
            bad = copy.deepcopy(data)
            target = next(row for row in bad if row["target"])
            if change=="target": target["target"] = False
            elif change=="samples": target["samples_us"]["c0"].pop()
            elif change=="nan": target["samples_us"]["c0"][0] = math.nan
            else: target["samples_us"].pop("anchor")
            bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError): score_model(bad)


if __name__ == "__main__": unittest.main()
