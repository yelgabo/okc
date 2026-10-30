"""CPU dry run of muse_bf16_proj/submission.py's prepare() and kernel_fn with a fake extension.

Packs the real shapes, wires every candidate, runs the self-test and autotune loops and the
dispatch for the judge's shapes, with the GPU kernels replaced by argument checks that enforce
the host interface (tensor dtypes, layouts, sizes, pmax parity, out shapes). Numerics are not
checked here (tools/emulate_proj.py and tools/emulate_gemv.py do that); the self-test's
comparison is bypassed. Catches the Python mistakes that would otherwise cost a judge check.

    uv run --with torch python tools/proj_dryrun.py [muse_bf16_proj/submission.py]
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "router": (256, 8192)}


class FakeExt:
    """Checks every argument the real kernels would reject, then fills the outputs with zeros."""

    def __init__(self, mod):
        self.mod = mod
        self.calls = []

    def _cfg(self, cid):
        return self.mod.SK_CONFIGS[cid]

    def sk_fwd(self, cid, ws, x, out, ptab, pmax, base4, n, k, diag, fused=False):
        c = self._cfg(cid)
        name, mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, nacc, cl, outt, fuse = c
        assert fuse == fused, name
        g = self.mod._sk_geom(name)
        m = x.shape[0]
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape == (m, k), name
        assert 1 <= m <= tokr, (name, m, tokr)
        assert ws.dtype == torch.uint8 and ws.is_contiguous(), name
        assert n % g["R"] == 0 and k % (g["UPS"] * 64) == 0, name
        tu = 768 if comp else (2048 if hl else 1024)
        wby = g["NW"] * g["UPW"] * g["NT"] * tu
        assert ws.numel() == (k // (g["UPS"] * 64)) * (n // g["R"]) * wby, (name, ws.numel())
        assert out.shape == (m, n) and out.dtype == (torch.float32 if outt == "f32" else torch.bfloat16), name
        if comp:
            assert ptab.dtype == torch.int32 and ptab.is_contiguous() and pmax % 2 == 0 and pmax >= 2, name
            assert ptab.shape == (n // g["R"], 2 * pmax + g["R"]), (name, tuple(ptab.shape), pmax)
            assert int(ptab[:, 0].max()) < pmax, name
            assert 0 <= (base4 & 0xFF) <= 120 and base4 == (base4 & 0xFF) * 0x01010101, name
            rowmap = ptab[:, 2 * pmax:].flatten()
            assert sorted(rowmap.tolist()) == list(range(n)), name
        assert diag.dtype == torch.int32
        self.calls.append((name, m))
        out.zero_()

    def skf_fwd(self, cid, ws, rws, x, out, rout, ptab, pmax, base4, n, k, rpart, rcount, diag):
        self.sk_fwd(cid, ws, x, out, ptab, pmax, base4, n, k, diag, fused=True)
        g = self.mod._sk_geom(self._cfg(cid)[0])
        ncta = n // g["R"]
        assert ncta % 32 == 0 and (k // (g["UPS"] * 64)) % (ncta // 32) == 0
        assert rws.dtype == torch.uint8 and rws.numel() == 256 * 8192 * 4, rws.numel()
        assert rout.shape == (x.shape[0], 256) and rout.dtype == torch.float32
        assert rpart.shape[0] >= ncta // 32 and rpart.shape[1] >= x.shape[0] and rpart.shape[2] == 256
        assert rcount.dtype == torch.int32 and rcount.numel() == 32
        rout.zero_()

    def cgemv_fwd(self, stream, table, pmax, base4, x, out, nthreads):
        assert stream.dtype == torch.uint8 and stream.dim() == 3 and stream.shape[2] == 48 and stream.is_contiguous()
        half, chunks = stream.shape[0], stream.shape[1]
        assert table.shape == (half, pmax, 2) and table.dtype == torch.int32 and pmax % 2 == 0
        assert x.shape == (1, chunks * 16) and x.dtype == torch.bfloat16
        assert out.shape == (1, 2 * half) and out.dtype == torch.bfloat16
        assert nthreads in (64, 128, 256, 512) and chunks % nthreads == 0 and chunks // nthreads in (1, 2, 4)
        self.calls.append(("cgemv", 1))
        out.zero_()

    def rgemv_fwd(self, w, x, out):
        assert w.dtype == torch.float32 and w.shape[1] == 8192 and w.is_contiguous()
        assert x.shape == (1, 8192) and out.shape == (1, w.shape[0]) and out.dtype == torch.float32
        self.calls.append(("rgemv", 1))
        out.zero_()

    def sgemv_fwd(self, w, x, out):
        assert w.dtype == torch.bfloat16 and w.shape[1] == 4096 and w.shape[0] % 4 == 0 and w.is_contiguous()
        assert x.shape == (1, 4096) and out.shape == (1, w.shape[0]) and out.dtype == torch.bfloat16
        self.calls.append(("sgemv", 1))
        out.zero_()

    def wd_fwd(self, cid, w, x, out, n, k, diag):
        name = self.mod.WD_NAMES[cid]
        wg, bn, clk, clr, hl, nst = self.mod.wd_geom(name)
        m = x.shape[0]
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape == (m, k) and 1 <= m <= 256, name
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.shape == ((2 if hl else 1) * n, k), name
        assert n % (64 * wg * clr) == 0 and k // 64 >= clk and k % 64 == 0, name
        assert clk * clr <= 16 and bn % clr == 0 and (bn // clr) % 8 == 0, name
        assert out.shape == (m, n) and out.dtype == (torch.float32 if hl else torch.bfloat16), name
        assert self.mod.wd_smem(wg, bn, clk, clr, hl, nst) <= 227 * 1024, name
        self.calls.append((name, m))
        out.zero_()


class FakeLt:
    """cuBLASLt plans: a few fake candidates per plan; lt_run checks shapes and fills zeros."""

    def __init__(self):
        self.plans = []
        self.runs = 0

    def lt_version(self):
        return 0

    def lt_plan_new(self, m, n, k, layout, ws_cap):
        assert layout in (0, 1, 2, 3) and ws_cap > 0
        self.plans.append({"mnk": (m, n, k), "layout": layout, "algos": []})
        return len(self.plans) - 1

    def lt_heuristic(self, plan, ws_bytes, req):
        a = self.plans[plan]["algos"]
        a.append([66, len(a), 1, 0, 0, 0, 35, 0, 2])
        return 1

    def lt_expand(self, plan, idx, max_add):
        a = self.plans[plan]["algos"]
        a.append(list(a[idx][:2]) + [2, 2, 0, 0, 35, 0, 2])
        return 1

    def lt_sweep(self, plan, idx, clusters, max_add):
        a = self.plans[plan]["algos"]
        assert 0 <= idx < len(a) and clusters
        a.append([a[idx][0], 100 + len(a), 1, 0, 0, 0, 35, 0, clusters[-1]])
        return 1

    def lt_add_config(self, plan, c):
        self.plans[plan]["algos"].append(list(c))
        return len(self.plans[plan]["algos"]) - 1

    def lt_count(self, plan):
        return len(self.plans[plan]["algos"])

    def lt_rejected(self, plan):
        return 0

    def lt_info(self, plan, idx):
        return self.plans[plan]["algos"][idx] + [0, 1000]

    def lt_run(self, plan, idx, w, x, out, ws):
        m, n, k = self.plans[plan]["mnk"]
        layout = self.plans[plan]["layout"]
        assert 0 <= idx < len(self.plans[plan]["algos"])
        assert tuple(x.shape) == (m, k) and x.dtype == torch.bfloat16 and x.is_contiguous()
        assert tuple(w.shape) == ((k, n) if layout % 2 else (n, k)) and w.is_contiguous()
        assert tuple(out.shape) == (m, n) and out.dtype == torch.bfloat16
        out.zero_()
        self.runs += 1
        return 0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "muse_bf16_proj/submission.py"
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ext = FakeExt(mod)
    mod._build_ext = lambda: ext
    mod._check = lambda out, ref, probes, tol: (True, 0.0, 0.0)
    lt = FakeLt()
    mod._build_lt = lambda: lt
    mod.LT_ALIGN = 1

    def fake_screen(fns, x, flush, reps):
        for fn in fns:
            fn(x)
        return [float(i) for i in range(len(fns))]
    mod._lt_screen = fake_screen
    timed = []

    def fake_time(fns, x, flush):
        for fn in fns:
            fn(x)
        timed.append(len(fns))
        # Last candidate wins, so the wide buckets dispatch to a wd kernel.
        return [float(len(fns) - i) for i in range(len(fns))]
    mod._time_candidates = fake_time
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
        # CPU torch has no mm with out_dtype; the split router baseline is SGLang's, not ours.
        mod._router_baseline = lambda x, state: (x.float() @ state["router"].T)
    gen = torch.Generator().manual_seed(5151)
    weights = {name: (torch.randn((n, k), generator=gen) / math.sqrt(k)).to(
        torch.float32 if name == "router" else torch.bfloat16) for name, (n, k) in LAYERS.items()}
    t0 = time.time()
    state = mod.prepare(weights)
    print(f"prepare dry run {time.time() - t0:.1f} s, {len(ext.calls)} kernel calls, {len(timed)} tuned shapes")
    assert state["fused"].keys() == {8, 16}, state["fused"].keys()
    picks = state["pick"]
    print(f"lt plans {len(lt.plans)}, lt runs {lt.runs}, lt picks {sorted(state['lt_pick'])}")
    assert len(state["lt_pick"]) == len(mod.LT_TARGETS), state["lt_pick"]
    assert all((layer, m) in picks for layer in LAYERS for m in range(1, 257)), "every m <= 256 has a pick"
    wd_calls = sorted({c for c in ext.calls if c[0] in mod.WD_GEOM})
    print(f"wd self-test/tune calls: {len(wd_calls)} distinct (config, m)")
    ext.calls.clear()
    shapes = [(m, layer) for layer in LAYERS for m in (1, 2, 5, 7, 8, 9, 16, 31, 32, 33, 128, 130, 256, 512, 1000)]
    for m, layer in shapes:
        n = 1024 if layer == "unpooler" and m <= 8 else LAYERS[layer][0]
        x = torch.randn((m, LAYERS[layer][1])).to(torch.bfloat16)
        out = mod.kernel_fn({"x": x, "config": {"m": m, "n": n, "k": LAYERS[layer][1], "layer": layer}}, state)
        assert out.shape == (m, n), (layer, m, tuple(out.shape))
        assert out.dtype == (torch.float32 if layer == "router" else torch.bfloat16), (layer, m)
    print("dispatched wd:", sorted({c for c in ext.calls if c[0] in mod.WD_GEOM}))
    for b in (8, 16):
        for m in ((1, 8) if b == 8 else (9, 16)):
            out, rout = state["fused"][b](torch.randn((m, 8192)).to(torch.bfloat16))
            assert out.shape == (m, 4096) and rout.shape == (m, 256)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
