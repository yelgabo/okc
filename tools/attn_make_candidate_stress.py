"""Generate candidate-only official check plus fresh queued graph replay stress.

The generated file is not a scored submission: it runs extra checks at import.
Usage: python3 tools/attn_make_candidate_stress.py OUTPUT.py
"""
from pathlib import Path
import sys
import attn_make_bench_one as base

ROOT = Path(__file__).resolve().parent.parent

FRESH = r'''

def _late_fresh_stress(replays=2000):
    # Each replay gets distinct Q/new-K/new-V/gate values at unchanged addresses.
    # All copies, replays and result copies queue on one stream, without host
    # synchronization between replays. Sampled outputs are checked afterwards.
    for kind in ("sliding", "full"):
        data = _ob_inputs(1, 80000, "tp8", kind, 1, "fixed",
                          "soft" if kind == "sliding" else "coherent", 1, 1., seed=311104)
        for _ in range(3):
            kernel_fn(data)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = kernel_fn(data)
        gen = torch.Generator(device="cuda").manual_seed(311105)
        bank = torch.randn((replays, *data["qkv"].shape), generator=gen,
                           device="cuda", dtype=torch.bfloat16)
        saved = torch.empty((replays, *out.shape), device="cuda", dtype=out.dtype)
        for i in range(replays):
            data["qkv"].copy_(bank[i])
            graph.replay()
            saved[i].copy_(out)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(saved.float()).all()), "nonfinite queued outputs"
        clean, ws = _ob_ws_check(data["qkv"].device)
        assert clean, ws
        for i in (0, replays // 2, replays - 1):
            data["qkv"].copy_(bank[i])
            graph.replay()
            torch.cuda.synchronize()
            ok, msg = _ob_check(data, saved[i])
            rel = _ob_rel(saved[i], out)
            assert ok and rel < 1e-4, f"fresh {kind} input{i}: {msg}, replay miss {rel}"
        # Reuse the captured graph after making the shape-based speculation wrong.
        data["seq_lens"].sub_(64)
        pos = data["seq_lens"].long()[:, None]
        data["positions"].copy_(pos.reshape(-1))
        data["cache_loc"].copy_(torch.gather(data["page_table"].long(), 1, pos).reshape(-1))
        for _ in range(50):
            graph.replay()
        torch.cuda.synchronize()
        ok, msg = _ob_check(data, out)
        clean, ws = _ob_ws_check(data["qkv"].device)
        assert ok and clean, f"SPEC miss {kind}: {msg} {ws}"
        print(f"LATEFRESH PASS {kind} unique_inputs={replays} queued={replays} "
              f"reference_samples=3 spec_miss_replays=50 {ws}", flush=True)
        del graph, bank, saved, data, out
        torch.cuda.empty_cache()

_ob_selftest()
_late_fresh_stress()
'''


def main():
    src = (ROOT / "muse_attention_decode/submission.py").read_text()
    assert src.count("#define ONE_LATE_TID 0") == 1
    src = src.replace("#define ONE_LATE_TID 0", "#define ONE_LATE_TID 1")
    runner = base.RUNNER.format(variants=["_one"], coop_probe=True)
    head, sep, _ = runner.rpartition("\nimport os as _os\n")
    assert sep
    Path(sys.argv[1]).write_text(src + head + FRESH)


if __name__ == "__main__":
    main()
