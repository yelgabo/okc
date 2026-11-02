"""CPU rejection tests for the actual offline layout and admission gates."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import unittest

import g_expert_g224_prefix_nvrtc_layout as layout

ROOT = Path(__file__).resolve().parents[1]


class OfflineGates(unittest.TestCase):
    def test_fixed_four_entry_receipts_and_layout(self):
        evidence = ROOT / "docs/steering/evidence/2026-11-01-worker-g224-prefix-resource-stop"
        manifests = {}
        for family, stack, stores, loads in (("B4", 208, 384, 280), ("MAIN", 224, 364, 260)):
            manifest = json.loads((evidence / (family + "-manifest.json")).read_text())
            manifests[family] = manifest
            self.assertEqual(manifest["source_sha256"], "ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f")
            self.assertEqual(len(manifest["resources"]), 2)
            self.assertEqual(set(manifest["resources"]), set(manifest["lowered"]))
            for row in manifest["resources"].values():
                self.assertEqual(row, dict(stack_bytes=stack, spill_store_bytes=stores, spill_load_bytes=loads, registers=96, barriers=16))
            for artifact, key in (("device.ptx", "ptx_sha256"), ("device.cubin", "cubin_sha256")):
                data = gzip.decompress((evidence / (family + "-" + artifact + ".gz")).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), manifest[key])
                if artifact == "device.ptx":
                    self.assertEqual(layout.inspect_ptx(data.decode()), manifest["layout_probe_results"])
            self.assertEqual(manifest["ptx_launch_bounds"].count(".maxntid 320, 1, 1"), 2)
            self.assertEqual(manifest["ptx_launch_bounds"].count(".minnctapersm 2"), 2)
        self.assertEqual(set(manifests["B4"]["options"]) - set(manifests["MAIN"]["options"]), {"-DOKC_B4FIN=1"})
        self.assertFalse(set(manifests["MAIN"]["options"]) - set(manifests["B4"]["options"]))

    def test_actual_emitted_layout_and_mutations(self):
        ptx = gzip.decompress((ROOT / "docs/steering/evidence/2026-11-01-worker-g224-prefix-resource-stop/B4-device.ptx.gz").read_bytes()).decode()
        result = layout.inspect_ptx(ptx)
        self.assertEqual({name: row["value"] for name, row in result.items()},
                         {name: expected for name, (_, expected) in layout.PROBES.items()})
        for original, replacement in (("mov.u64 %rd1, 1248;", "mov.u64 %rd1, 1256;"),
                                      ("g224_probe_wcpB()", "not_a_probe()"),
                                      ("mov.u64 %rd1, 2272;", "mov.u64 %rd1, 2048;"),
                                      ("mov.u64 %rd1, 23096;", "mov.u64 %rd1, 23776;")):
            self.assertIn(original, ptx)
            with self.assertRaises(AssertionError):
                layout.inspect_ptx(ptx.replace(original, replacement, 1))

    def test_adapter_preserves_actual_source_and_rejects_missing_guard(self):
        native = (ROOT / "docs/steering/evidence/2026-11-01-worker-g2-24-route-prefix/selected.cu").read_text()
        prefix = native.split("#include <torch/extension.h>", 1)[0]
        adapted = layout.adapt_device(prefix)
        for assertion in layout.OFFSET_ASSERTIONS:
            self.assertIn(assertion, prefix)
            self.assertNotIn(assertion, adapted)
        self.assertIn("sizeof(Smem) == 23096", adapted)
        self.assertIn("== 115328", adapted)
        self.assertIn("g224_layout_anchor", adapted)
        with self.assertRaises(AssertionError):
            layout.adapt_device(prefix.replace(layout.OFFSET_ASSERTIONS[0], "", 1))

    def test_actual_rounded_header_equation_and_old_ceiling_negative(self):
        source = (ROOT / "tools/g_expert_g224_prefix_compile.py").read_text()
        tree = ast.parse(source)
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Subscript)
                               and isinstance(target.slice, ast.Constant)
                               and target.slice.value == "rounded_register_blocks" for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        expression = compile(ast.Expression(assignments[0].value), "actual_rounded_gate", "eval")
        expected = {96: 2, 97: 1, 98: 1, 99: 1, 100: 1, 101: 1, 102: 1}
        for registers, blocks in expected.items():
            manifest = {"resources": {"entry": {"registers": registers}}}
            actual = eval(expression, {"manifest": manifest})
            self.assertEqual(actual, {"entry": blocks})
        self.assertEqual(65536 // (2 * 320), 102)
        self.assertLess(expected[102], 2)


if __name__ == "__main__":
    unittest.main()
