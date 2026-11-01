"""Source and host ABI regressions for the private selected-load G arm."""
import ast
import builtins
from pathlib import Path
import symtable
from types import SimpleNamespace
import unittest

from dl_g_verify_slice import build
from dl_owned_main_slice import specialize
from dl_owned_slice import build as build_owned

ROOT = Path(__file__).resolve().parents[1]


class GSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8")
        cls.output, cls.manifest = build(cls.source)

    def test_literals_and_combined_budget(self):
        def literals(source):
            return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
        original, selected = literals(self.source), literals(self.output)
        for name in ("ATT7_CUDA_SRC", "MOE_CUDA_SRC"):
            self.assertEqual(original[name], selected[name])
        owned, _ = build_owned((ROOT / "muse_decode_layer/cand/layer_lt_boundary.py").read_text(encoding="utf8"))
        small, _, _ = specialize(owned)
        self.assertLess(len(small.encode()) + len(self.output.encode()) + 50000, 1_000_000)
        self.assertNotIn("_MA_ALT_SRCS['moe_helper0']", self.output)
        with self.assertRaises(AssertionError):
            build(self.source + "# unpinned")

    def test_all_global_dependencies_resolve(self):
        table = symtable.symtable(self.output, "G-slice", "exec")
        available = set(dir(builtins)) | {s.get_name() for s in table.get_symbols()
                                         if s.is_assigned() or s.is_imported()}
        def globals_used(scope):
            names = {s.get_name() for s in scope.get_symbols() if s.is_global()}
            return names.union(*(globals_used(child) for child in scope.get_children()))
        self.assertEqual(set().union(*(globals_used(child) for child in table.get_children())) - available, set())
        tree = ast.parse(self.output)
        row = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_graft_row_boundary")
        self.assertTrue(row.decorator_list)

    def test_entry_scope_and_actual_stage_arguments(self):
        import torch
        node = next(n for n in ast.parse(self.output).body if isinstance(n, ast.FunctionDef)
                    and n.name == "kernel_fn")
        events = []
        class Launch:
            def __init__(self, name):
                self.name = name
            def __getitem__(self, grid):
                def invoke(*args, **kwargs):
                    events.append((self.name, grid, args, kwargs))
                return invoke
        cfg = dict(batch=32, q_len=8, tokens=256, window=2048, page_size=1, hidden=8192,
                   num_q_heads=16, num_kv_heads=2, head_dim=64, top_k=8,
                   n_experts=256, local_experts=32, kv_scale=1.)
        data = dict(config=cfg, attn_input=torch.empty(256, 8192, dtype=torch.bfloat16),
                    stream=torch.empty(256, 8192), page_table=SimpleNamespace(shape=(32, 86008)),
                    local_start=torch.tensor([0], dtype=torch.int32))
        def projection(x, plan):
            events.append((plan,))
            return torch.empty(256, {"qkv": 2304, "o": 8192, "pr": 4608, "un": 8192}[plan],
                               dtype=torch.float32 if plan == "pr" else torch.bfloat16)
        def attention(*args):
            events.append(("attention",))
            return torch.empty(256, 1024, dtype=torch.bfloat16)
        def native(*args):
            events.append(("native", args))
        graft = {key: torch.empty(8192) for key in ("pre1", "post2", "pre2", "alpha1", "alpha2", "beta1", "beta2")}
        state = {"graft_lt": {(k, 256): k for k in ("qkv", "o", "pr", "un")}, "graft": graft,
                 "bias": torch.empty(256), "g_preexp": torch.empty(4096, dtype=torch.bfloat16),
                 "native": SimpleNamespace(okc_run=native), "ma_w": [], "ma_scratch": [], "nsm": 132}
        env = {"torch": torch, "_decode_attention7": attention, "_graft_lt_run": projection,
               "_graft_row_boundary": Launch("boundary"), "_graft_route": Launch("route"),
               "_GRAFT_ROUTE_EARLY": False}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "private-G-entry", "exec"), env)
        result = env["kernel_fn"](data, state)
        self.assertEqual([e[0] for e in events],
                         ["qkv", "attention", "o", "boundary", "pr", "route", "native", "un", "boundary"])
        boundaries = [e for e in events if e[0] == "boundary"]
        self.assertEqual([e[2][8:] for e in boundaries],
                         [(8192, 8192, False, 4, 16), (8192, 8192, True, 4, 16)])
        self.assertIs(boundaries[0][2][3], graft["pre1"])
        self.assertIs(boundaries[1][2][3], graft["pre2"])
        route = next(e for e in events if e[0] == "route")
        self.assertEqual(route[3], dict(SPLIT=False, EARLY=False, PACKED=1, num_warps=4, launch_pdl=False))
        self.assertEqual([x.dtype for x in result], [torch.bfloat16, torch.float32, torch.int32])
        for key, wrong in (("kv_scale", .5), ("head_dim", 128), ("tokens", 128)):
            saved = cfg[key]
            cfg[key] = wrong
            with self.assertRaises(RuntimeError):
                env["kernel_fn"](data, state)
            cfg[key] = saved
        data["page_table"].shape = (32, 80000)
        with self.assertRaises(RuntimeError):
            env["kernel_fn"](data, state)


if __name__ == "__main__":
    unittest.main()
