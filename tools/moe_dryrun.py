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
            variant = rest[-1]
            assert 0 <= variant < n_variants, variant
            mode = None
            if variant >= mod.VARIANT_MAIN0:
                mode = mod.MAIN_MODES[variant - mod.VARIANT_MAIN0]
            ref = current.get("ref")
            if ref is not None and ref.shape == out.shape and mode not in fail_modes:
                out.copy_(ref.to(out.dtype))
            else:
                out.normal_()
            # A stand-in time per variant so the bench has an order to rank.
            current["t"] = 100.0 + variant + (0 if mode is None else -mode)

        def moe_smem_bytes(self):
            return 0

    mod._ext = lambda: FakeExt()
    mod._legacy_route_ext = lambda: None
    mod._legacy_w4_ext = lambda: None
    mod._legacy_prepare = lambda *a: None
    mod._repack = lambda *a: {k: torch.zeros(1) for k in ("w13", "w2", "w13_gs", "w2_gs", "w13f", "w2f")}
    mod._scratch = lambda dev: {k: torch.zeros(16, dtype=torch.int32) for k in (
        "inter", "ybuf", "ss", "ctr", "done_a", "xa", "xb", "fix", "fixcnt", "xready", "done_as")}
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

    def run(data, state, variant=None):
        current["ref"] = ref_forward(data, None)
        return real_run(data, state, variant)

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
    print("main_bad", sorted(mod._vname(v) for v in state["main_bad"]), "main_ok", state["main_ok"])
    print("references computed", nref[0], "cache left", len(mod._REF_CACHE))
    for bs in (1, 8, 16, 32, 128, 256):
        data = mod._make_inputs(bs, 5)
        out = mod.kernel_fn(data, state)
        assert out.shape == (bs, mod.HIDDEN) and out.dtype == torch.bfloat16
        print(f"kernel_fn bs={bs}: {mod._vname(mod._policy_variant(state, bs))}")


if __name__ == "__main__":
    main()
