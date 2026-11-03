import ast
from pathlib import Path
import unittest

from g_integration_common import build, attention_literal
from attention_g_cooperative_host import build as attention, REPAIR
from proj_g_preexpert_gamma import build as preexpert


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf-8")


class CommonSource(unittest.TestCase):
    def test_off_identity_and_rejects_other_epochs(self):
        result, manifest = build(SOURCE)
        self.assertEqual(result, SOURCE)
        self.assertFalse(manifest["enabled"])
        for treatment in (attention(SOURCE, True), preexpert(SOURCE, True)[0]):
            with self.assertRaisesRegex(AssertionError, "exact607"):
                build(treatment, True)

    def test_disjoint_complete_composition(self):
        result, manifest = build(SOURCE, True)
        coeff = preexpert(SOURCE, True)[0]
        expected = ast.parse(coeff)
        host_node = attention_literal(attention(SOURCE, True))
        for index, node in enumerate(expected.body):
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "ATT7_CUDA_SRC":
                expected.body[index] = host_node
        self.assertEqual(ast.dump(ast.parse(result), include_attributes=False),
                         ast.dump(expected, include_attributes=False))
        self.assertEqual(result.replace(REPAIR, ""), coeff)
        self.assertEqual(manifest["bytes"], len(result.encode()))
        self.assertFalse(manifest["H200_qualified"])
        self.assertFalse(manifest["dispatch_enabled"])


if __name__ == "__main__":
    unittest.main()
