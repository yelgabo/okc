"""CPU dry run of muse_moe_w4a16/submission.py's prepare() and kernel_fn.

The GPU pieces (the extensions, the repacks, the scratch, graph timing, the FP32 reference)
are replaced by stand-ins: a fake moe_run that checks the variant id against the dispatch table
and writes the reference into `out` for every variant except those named in FAIL, so the
self-test, the per-mode exclusion, the bench and the policy run as on the judge. Numerics are
not checked (tools/moe_emulate_fused.py does the weight path).

    uv run --with torch python tools/moe_dryrun.py [muse_moe_w4a16/submission.py] [FAIL mode ...]
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "muse_moe_w4a16/submission.py")
    fail_modes = {int(a) for a in sys.argv[2:]}
    mod = load(path)
    n_variants = mod.VARIANT_MAIN0 + len(mod.MAIN_MODES)
    current = {}

    class Ctr(torch.Tensor):
        pass

    class FakeExt:
        def moe_run(self, x, ids, tw, *rest):
            out = rest[10]
            variant, cost = rest[-2], rest[-1]
            assert len(cost) in (0, 18) and all(c > 0 for c in cost[1:9] + cost[10:18]), cost
            assert 0 <= variant < n_variants, variant
            mode = None
            if variant >= mod.VARIANT_MAIN0:
                mode = mod.MAIN_MODES[variant - mod.VARIANT_MAIN0]
            if mode == 11:
                stamps(rest[5], ids.shape[0], cost)
            ref = current.get("ref")
            if ref is not None and ref.shape == out.shape and mode not in fail_modes:
                out.copy_(ref.to(out.dtype))
            else:
                out.normal_()
            # A stand-in time per variant so the bench has an order to rank.
            current["t"] = 100.0 + variant + (0 if mode is None else -mode)

        def moe_smem_bytes(self):
            return 0

    # Mode 11's stamps from a known work model (us per COST_TERMS unit, the last the CTA term),
    # so the cost-model fit can be checked against it: random class mixes per CTA, scaled by
    # the launch's batch size and shifted by the cost table so the tables differ.
    true = torch.tensor([0.30, 0.05, 2.0, 3.0, 0.25, 0.04, 1.5, 2.5, 20.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(1)

    def stamps(inter, bs, cost):
        g = 132
        shift = 0 if not cost else (1 if cost[1] == 100 else 2)
        dg = torch.zeros(g, 18, 8, dtype=torch.int64)
        w = torch.rand(g, 16, generator=gen) * (bs / 32) * (1 + shift * torch.linspace(0, 1, 16))
        st = (w * 20).round().long()
        dg[:, :16, 6] = st
        dg[:, :16, 7] = (st + 15) // 16
        dg[:, 16, 1:3] = torch.randint(0, 3, (g, 2), generator=gen)
        x = mod._cost_features(dg.double())
        work = x @ true + torch.randn(g, generator=gen, dtype=torch.float64)
        cyc = 1800.0
        dg[:, :16, 2] = (0.8 * work * cyc).long().unsqueeze(1)
        dg[:, :16, 3] = (0.2 * work * cyc).long().unsqueeze(1)
        dg[:, 17, 5] = (0.6 * work * cyc).long()
        dg[:, 17, 6] = (0.4 * work * cyc).long()
        start = 10_000
        done = start + ((work + 5 * torch.rand(g, generator=gen, dtype=torch.float64)) * 1e3).long()
        dg[:, 16, 3] = 0
        dg[:, 16, 4] = start
        dg[:, 16, 5] = done
        dg[:, 16, 6] = done.max() + 1000
        dg[:, 16, 7] = done.max() + 2000
        dg[:, 17, 3] = 0
        dg[:, 17, 4] = ((done.max() + 2000).double() * cyc / 1e3).long()
        inter.view(-1).view(torch.int64)[: dg.numel()] = dg.flatten()

    mod._ext = lambda: FakeExt()
    mod._legacy_route_ext = lambda: None
    mod._legacy_w4_ext = lambda: None
    mod._legacy_prepare = lambda *a: None
    mod._repack = lambda *a: {k: torch.zeros(1) for k in ("w13", "w2", "w13_gs", "w2_gs", "w13f", "w2f")}
    def scratch(dev):
        sc = {k: torch.zeros(16, dtype=torch.int32) for k in (
            "ybuf", "ss", "ctr", "done_a", "xa", "xb", "fix", "fixcnt", "xready", "done_as")}
        # Mode 11's stamps as the kernel would leave them: 1.8 GHz, 200 us per CTA.
        dg = torch.randint(0, 100000, (132, 18, 8), dtype=torch.int64)
        dg[:, 16, 3:8] = torch.tensor([0, 10_000, 180_000, 190_000, 200_000])
        dg[:, 17, 3] = 0
        dg[:, 17, 4] = 360_000
        sc["inter"] = torch.zeros(4096, 4096, dtype=torch.bfloat16)
        sc["inter"].view(-1).view(torch.int64)[: dg.numel()] = dg.flatten()
        return sc

    mod._scratch = scratch

    class Props:
        multi_processor_count = 132

    torch.cuda.get_device_properties = lambda i: Props()
    legacy_calls = []

    def run_legacy(data, state, pol):
        legacy_calls.append(pol)
        return current["ref"].to(torch.bfloat16)

    mod._run_legacy = run_legacy
    real_make = mod._make_inputs
    mod._make_inputs = lambda bs, seed, dist="normal", device="cpu": real_make(bs, seed, dist, "cpu")
    nref = [0]

    def ref_forward(data, weights):
        nref[0] += 1
        return torch.randn(data["hidden_states"].shape[0], mod.HIDDEN,
                           generator=torch.Generator().manual_seed(data["hidden_states"].shape[0]))

    mod._ref_forward = ref_forward
    real_run = mod._run

    def run(data, state, variant=None, cost=None):
        current["ref"] = ref_forward(data, None)
        return real_run(data, state, variant, cost)

    mod._run = run

    def graph_time(fn, flush, iters=20):
        fn()
        return current["t"]

    mod._graph_time = graph_time
    torch.cuda.synchronize = lambda: None
    real_empty = torch.empty
    torch.empty = lambda *a, device=None, **k: real_empty(*a, **{kk: v for kk, v in k.items()})
    weights = {"w13": torch.zeros(1), "post_norm_w": torch.zeros(mod.HIDDEN, dtype=torch.bfloat16)}
    state = mod.prepare(weights)
    print("true work model", " ".join(f"{n}:{float(c):.3f}" for n, c in zip(mod.COST_TERMS, true)))
    print("main_bad", sorted(mod._vname(v) for v in state["main_bad"]), "main_ok", state["main_ok"])
    print("references computed", nref[0], "cache left", len(mod._REF_CACHE))
    for bs in (1, 8, 16, 32, 128, 256):
        data = mod._make_inputs(bs, 5)
        out = mod.kernel_fn(data, state)
        assert out.shape == (bs, mod.HIDDEN) and out.dtype == torch.bfloat16
        print(f"kernel_fn bs={bs}: {mod._vname(mod._policy_variant(state, bs))}")


if __name__ == "__main__":
    main()
