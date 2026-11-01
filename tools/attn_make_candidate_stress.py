"""Generate candidate-only official check plus fresh queued graph replay stress.

The generated file is not a scored submission: it runs extra checks at import.
Usage: python3 tools/attn_make_candidate_stress.py OUTPUT.py
"""
from pathlib import Path
import sys
import attn_make_bench_one as base

ROOT = Path(__file__).resolve().parent.parent

FRESH = r'''

import types as _late_types
import triton as _late_triton
import triton.language as _lt
_late_control = _late_types.ModuleType("owned_late_stress_control")
exec(compile(_LATE_CONTROL_SRC, "owned_late_stress_control.py", "exec"), _late_control.__dict__)

@_late_triton.jit
def _late_compare(A, B, Verdict, N: _lt.constexpr, BLOCK: _lt.constexpr):
    i = _lt.arange(0, BLOCK)
    a = _lt.load(A+i, i<N, other=0).to(_lt.float32)
    b = _lt.load(B+i, i<N, other=0).to(_lt.float32)
    error = _lt.sum((a-b)*(a-b), 0)
    denom = _lt.maximum(_lt.sum(b*b, 0), 1e-20)
    finite = _lt.sum(((i<N) & ((_lt.abs(a) == float("inf")) |
                      (_lt.abs(b) == float("inf")) | (a != a) | (b != b))).to(_lt.int32), 0) == 0
    bad = (~finite) | (error > 1e-8*denom)
    _lt.atomic_add(Verdict, bad.to(_lt.float32))
    _lt.atomic_max(Verdict+1, error/denom)

def _late_fresh_stress(replays=2000):
    # Each replay gets distinct Q/new-K/new-V/gate values at unchanged addresses.
    # All copies, replays and result copies queue on one stream, without host
    # synchronization between replays. Sampled outputs are checked afterwards.
    for kind in ("sliding", "full"):
        data = _ob_inputs(1, 80000, "tp8", kind, 1, "fixed",
                          "soft" if kind == "sliding" else "coherent", 1, 1., seed=311104)
        for _ in range(3):
            kernel_fn(data)
            _late_control.kernel_fn(data)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = kernel_fn(data)
        control_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(control_graph):
            control_out = _late_control.kernel_fn(data)
        graph.replay()
        control_graph.replay()
        verdict = torch.zeros(2, device="cuda", dtype=torch.float32)
        _late_compare[(1,)](out, control_out, verdict, out.numel(),
                            _late_triton.next_power_of_2(out.numel()))
        verdict.zero_()
        gen = torch.Generator(device="cuda").manual_seed(311105)
        bank = torch.randn((replays, *data["qkv"].shape), generator=gen,
                           device="cuda", dtype=torch.bfloat16)
        saved = torch.empty((replays, *out.shape), device="cuda", dtype=out.dtype)
        for i in range(replays):
            data["qkv"].copy_(bank[i])
            graph.replay()
            control_graph.replay()
            _late_compare[(1,)](out, control_out, verdict, out.numel(),
                                _late_triton.next_power_of_2(out.numel()))
            saved[i].copy_(out)
        torch.cuda.synchronize()
        bad, max_relative_sq = verdict.tolist()
        assert bad == 0, f"{kind}: {bad} fresh outputs differ; max relative squared {max_relative_sq}"
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
              f"owned_control_checks={replays} max_relative_sq={max_relative_sq:.2e} "
              f"FP32_reference_samples=3 spec_miss_replays=50 {ws}", flush=True)
        del graph, control_graph, control_out, verdict, bank, saved, data, out
        torch.cuda.empty_cache()

_ob_selftest()
_late_fresh_stress()
'''


def main():
    src = (ROOT / "muse_attention_decode/submission.py").read_text()
    assert src.count("#define ONE_LATE_TID 0") == 1
    # Host-only prepare diagnostics; kernel code and timed launches are unchanged.
    marker = "            resident = blocks * sms;"
    assert src.count(marker) == 1
    src = src.replace(marker, marker + '''
            cudaFuncAttributes attr = {};
            cudaFuncGetAttributes(&attr, kern);
            printf("LATERES L=%d mode=%d regs=%d local_bytes=%zu static_smem=%zu dynamic_smem=%d threads=%d blocks_per_sm=%d\\n",
                   L, MODE, attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, BYTES, NT, blocks);
''')
    control = src.replace('name="okc_attn_one_v1"', 'name="okc_attn_late_stress_control"')
    assert control != src
    src = src.replace("#define ONE_LATE_TID 0", "#define ONE_LATE_TID 1")
    runner = base.RUNNER.format(variants=["_one"], coop_probe=True)
    head, sep, _ = runner.rpartition("\nimport os as _os\n")
    assert sep
    Path(sys.argv[1]).write_text(src + head + "\n_LATE_CONTROL_SRC = " + repr(control) + "\n" + FRESH)


if __name__ == "__main__":
    main()
