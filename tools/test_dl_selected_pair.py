"""Packaging regressions for the private cross-family comparison."""
import ast
import hashlib
import inspect
import copy
from pathlib import Path
import unittest

from dl_selected_pair import build, literal, LOADER

ROOT = Path(__file__).resolve().parents[1]


class SelectedPair(unittest.TestCase):
    def test_plain_source_round_trip(self):
        source = '"""module doc"""\nx = r"""CUDA\\nline"""\n'
        self.assertEqual(ast.literal_eval(literal(source)), source)
        env = {}
        exec(LOADER, env)
        first = env["_pair_module"]("_pair_test_first", "VALUE = 7\ndef read():\n    return VALUE\n")
        second = env["_pair_module"]("_pair_test_second", "VALUE = 11\ndef read():\n    return VALUE\n")
        self.assertEqual((first.read(), second.read()), (7, 11))
        self.assertIn("return VALUE", inspect.getsource(first.read))

    def test_complete_package_and_explicit_scope(self):
        source, manifest = build(ROOT)
        self.assertLess(manifest["bytes"], 900000)
        self.assertFalse(manifest["H200_qualified"])
        tree = ast.parse(source)
        modules = {n.targets[0].id.removeprefix("_PAIR_SOURCE_"): ast.literal_eval(n.value)
                   for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id.startswith("_PAIR_SOURCE_")}
        self.assertEqual(set(modules), {"G", "owned", "baseline"})
        for key, content in modules.items():
            self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), manifest["arm_pins"][key])
            ast.parse(content)
        self.assertEqual(modules["baseline"], (ROOT / "muse_decode_layer/baseline.py").read_text(encoding="utf8"))
        self.assertIn("linecache.cache[filename]", source)
        self.assertIn("PAIR_NATURAL_ROUTING_DRIFT", source)
        self.assertIn("PAIR_FRESH_ROUTING_DRIFT", source)
        self.assertNotIn("copy.deepcopy(owned)", source)
        self.assertIn('owned2 = _PAIR_MODULES["owned"].prepare(weights)', source)
        self.assertIn("_wl_check_owners([graft, graft2], readonly)", source)

    def test_owned_lt_workspace_alias_is_rejected(self):
        import torch
        env = {"torch": torch}
        exec((ROOT / "tools/dl_g1_layer_owners.py").read_text(encoding="utf8"), env)
        source = (ROOT / "tools/dl_selected_pair_body.py").read_text(encoding="utf8")
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                  and n.name == "_pair_owned_owners")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "owned-owners", "exec"), env)
        def owner():
            def runner():
                workspace = torch.empty(32, dtype=torch.uint8)
                def run(x, ws=workspace):
                    return x
                return run
            return {"dlplt": {(key, 256): runner() for key in ("qkv_gate", "unpooler")},
                    "scratch": torch.empty(64)}
        first = owner()
        env["_pair_owned_owners"](first, owner(), {})
        with self.assertRaisesRegex(AssertionError, "PAIR_OWNED_LT_ALIAS"):
            env["_pair_owned_owners"](first, copy.deepcopy(first), {})


if __name__ == "__main__":
    unittest.main()
