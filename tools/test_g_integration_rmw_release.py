import ast
from pathlib import Path
import unittest

from g_integration_common import build as common
from g_integration_rmw_release import build, common_delta, COMMON
from g_short_final_rmw import build as short_rmw


SOURCE = (Path(__file__).resolve().parents[1] / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf-8")


class FixedRMWSource(unittest.TestCase):
    def test_complete_disjoint_AST_composition(self):
        control, _ = common(SOURCE, True)
        self.assertEqual(build(SOURCE)[0], control)
        rmw, _ = short_rmw(SOURCE, True)
        base, expected, rtree = ast.parse(SOURCE), ast.parse(control), ast.parse(rmw)
        self.assertEqual(len(rtree.body), len(base.body) + 2)
        changed = []
        for index, (old, new) in enumerate(zip(base.body, rtree.body)):
            if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
                self.assertIsInstance(old, ast.Assign)
                self.assertEqual(ast.unparse(old.targets[0]), "_MA_ALT_SRCS['moe_helper0']")
                expected.body[index] = new
                changed.append(index)
        self.assertEqual(len(changed), 1)
        expected.body.extend(rtree.body[len(base.body):])
        result, manifest = build(SOURCE, True)
        self.assertEqual(ast.dump(ast.parse(result), include_attributes=False),
                         ast.dump(expected, include_attributes=False))
        self.assertEqual(manifest["common_control_sha256"], COMMON)
        self.assertFalse(manifest["startup_qualified"])
        self.assertFalse(manifest["dispatch_enabled"])

    def test_rejects_overlapping_or_other_source_epochs(self):
        control, _ = common(SOURCE, True)
        with self.assertRaisesRegex(AssertionError, "exact607"):
            build(control, True)
        with self.assertRaisesRegex(AssertionError, "overlapping|ambiguous"):
            common_delta(SOURCE, control, control)


if __name__ == "__main__":
    unittest.main()
