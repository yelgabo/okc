import unittest
from pathlib import Path
import tempfile

from g_integration_source_ledger import atoms, compare, inventory, FILES


class SourceLedgerTest(unittest.TestCase):
    def test_wrappers_and_registration_are_not_lost(self):
        source = "def prepare(w): return w\nfirst=prepare\ndef prepare(w): return first(w)\nregister(prepare)\n"
        rows = atoms(source)
        defs = [r for r in rows if r["label"] == "FunctionDef:prepare"]
        self.assertEqual([r["occurrence"] for r in defs], [1, 2])
        new = atoms(source.replace("register(prepare)", "assert callable(prepare)\nregister(prepare)"))
        changes = compare(rows, new)
        self.assertEqual(sum("before" in r for r in changes), 4)
        self.assertEqual(sum("after" in r for r in changes), 5)

    def test_whole_module_inventory_is_pinned_and_complete(self):
        root = Path(__file__).resolve().parents[1]
        result = inventory(root)
        for key, side in (("G0", "before"), ("G607", "after")):
            self.assertEqual(sum(side in r for r in result["changes"]),
                             result["sources"][key]["top_level_statements"])
        att = [r for r in result["changes"] if r.get("before", {}).get("label") == "Assign:ATT7_CUDA_SRC"]
        self.assertEqual(len(att), 1)
        self.assertEqual(att[0]["status"], "changed")
        self.assertTrue(att[0]["after"]["string_value"]["native_source"])

    def test_wrong_source_fails_before_any_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / FILES["G0"]
            path.parent.mkdir(parents=True)
            path.write_text("raise RuntimeError('must never execute')\n")
            with self.assertRaisesRegex(ValueError, "unrecognized G0"):
                inventory(root)


if __name__ == "__main__":
    unittest.main()
