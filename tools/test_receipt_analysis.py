"""Regression checks for current status receipts and the older profile format."""

import json
from pathlib import Path
import tempfile
import unittest

import dl_score_model as model
import hotspots

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "muse_decode_layer/profiles/scored.7b84ed4346d5497b.json"
BASELINE = ROOT / "muse_decode_layer/baseline_profile/profiles/baseline.7d42771c3384449d.json"


class ReceiptAnalysis(unittest.TestCase):
    def test_current_score_reconstructs_judge(self):
        job, judged, shapes = model.load_run(RECEIPT)
        self.assertEqual(job, "7b84ed4346d5497b")
        self.assertEqual(len(shapes), 12)
        self.assertAlmostEqual(model.score(shapes), judged, places=10)
        scaled = {n: s["ours"] * 0.8 for n, s in shapes.items()}
        self.assertAlmostEqual(model.score(shapes, scaled), judged / 0.8, places=10)

    def test_normalized_format_matches_status(self):
        normalized = hotspots.load(RECEIPT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalized.json"
            path.write_text(json.dumps(normalized), encoding="utf-8")
            self.assertEqual(model.load_run(path), model.load_run(RECEIPT))

    def test_profiles_join_despite_seed_and_repeats(self):
        run = hotspots.load(RECEIPT)
        baseline = hotspots.profiles_by_case(hotspots.load(BASELINE))
        own = hotspots.profiles_by_case(run)
        for row in run["benchmarks"]:
            key = hotspots.case_key(row["case"])
            self.assertIn(key, baseline)
            self.assertIn(key, own)
            self.assertTrue(hotspots.kernels(baseline[key]))

    def test_incomplete_receipt_is_rejected(self):
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        records = receipt["result"]["records"]
        records.remove(next(r for r in records if r.get("event") == "benchmark"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "twelve scored shapes"):
                model.load_run(path)


if __name__ == "__main__":
    unittest.main()
