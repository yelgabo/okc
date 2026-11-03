import copy
from pathlib import Path
import unittest

import d32_b4_prefix_audit as audit


class SelectedPrefix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (Path(__file__).resolve().parents[1] / audit.BASE).read_text()
        cls.result = audit.audit(cls.source)

    def test_actual_factory_selection(self):
        audit.verify(self.result["builds"])

    def test_helper_substitution_is_detected(self):
        old = 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)'
        prefix, found, suffix = self.source.rpartition(old)
        self.assertEqual(found, old)
        changed = prefix + 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "@moe_helper0")' + suffix
        with self.assertRaisesRegex(AssertionError, "selected native literal changed"):
            audit.verify(audit.capture(changed))

    def test_serial_call_is_detected(self):
        records = copy.deepcopy(self.result["builds"])
        b4 = next(r for r in records if "-DOKC_B4FIN=1" in r["flags"])
        b4["producer_body"] = b4["producer_body"].replace(
            "wgs_ranges(p, S);  // all 32 lanes", "if ((TIDX & 31) == 0) wgs_ranges(p, S);")
        with self.assertRaises(AssertionError):
            audit.verify(records)


if __name__ == "__main__":
    unittest.main()
