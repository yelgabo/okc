"""Candidate-only check of the cluster merge (ONE_CL=1 as the judged path) with the fresh-input
gates at import: the registered official-math FP32 gate of attn_make_fp32_fresh_gate.py
(2000 unique inputs on both batch-1 q_len 1 shapes, every candidate and control output
against its own FP32 reference under the official bounds, negative controls), plus for each
batch-1 sliding q_len (1 and 8) 2000 fresh queued replays compared with an independent
ONE_CL=0 control at 5e-3 relative L2 (the strict 1e-4 gate fails null against null,
docs/ruled-out.md), three FP32 reference samples, 50 replays after a 64-token speculation
miss and a workspace scan. The judge's official tests and timing follow.

Usage: python3 tools/attn_make_cl_gate.py OUTPUT.py
"""
from pathlib import Path
import sys
import attn_make_bench_one as base
import attn_make_fp32_fresh_gate as fp32

ROOT = Path(__file__).resolve().parent.parent

STRESS = r'''
@_fg_triton.jit
def _cl_compare(A, B, Verdict, N: _ft.constexpr, BLOCK: _ft.constexpr):
    i = _ft.arange(0, BLOCK)
    a = _ft.load(A + i, i < N, other=0).to(_ft.float32)
    b = _ft.load(B + i, i < N, other=0).to(_ft.float32)
    error = _ft.sum((a - b) * (a - b), 0)
    denom = _ft.maximum(_ft.sum(b * b, 0), 1e-20)
    finite = _ft.sum(((i < N) & ((_ft.abs(a) == float("inf")) | (_ft.abs(b) == float("inf"))
                                 | (a != a) | (b != b))).to(_ft.int32), 0) == 0
    bad = (~finite) | (error > 2.5e-5 * denom)
    _ft.atomic_add(Verdict, bad.to(_ft.float32))
    _ft.atomic_max(Verdict + 1, error / denom)


def _cl_stress(q_len, replays=2000):
    _fg_budget()
    data = _ob_inputs(1, 80000, "tp8", "sliding", q_len, "fixed", "soft", 1, 1., seed=311106 + q_len)
    fns = {"control": _fg_control.kernel_fn, "candidate": kernel_fn}
    for fn in fns.values():
        for _ in range(3):
            fn(data)
    torch.cuda.synchronize()
    graphs = {}
    outs = {}
    for tag, fn in fns.items():
        graphs[tag] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[tag]):
            outs[tag] = fn(data)
        graphs[tag].replay()
    torch.cuda.synchronize()
    cand = outs["candidate"]
    gen = torch.Generator(device="cuda").manual_seed(311107 + q_len)
    bank = torch.randn((replays, *data["qkv"].shape), device="cuda", dtype=torch.bfloat16,
                       generator=gen)
    verdict = torch.zeros(2, device="cuda")
    saved = torch.empty((replays, *cand.shape), device="cuda", dtype=cand.dtype)
    for idx in range(replays):
        data["qkv"].copy_(bank[idx])
        for tag in fns:
            graphs[tag].replay()
        _cl_compare[(1,)](cand, outs["control"], verdict, cand.numel(),
                          _fg_triton.next_power_of_2(cand.numel()))
        saved[idx].copy_(cand)
    torch.cuda.synchronize()
    bad, max_sq = verdict.tolist()
    print(f"CLFRESH sliding q_len={q_len} candidate_vs_control inputs={replays} bad={bad:g} "
          f"max_rel={max_sq ** .5:.3e} threshold=5e-3", flush=True)
    assert bool(torch.isfinite(saved.float()).all()), "nonfinite queued outputs"
    assert bad == 0, f"q_len {q_len}: {bad:g} fresh outputs differ from the control"
    for idx in (0, replays // 2, replays - 1):
        data["qkv"].copy_(bank[idx])
        graphs["candidate"].replay()
        torch.cuda.synchronize()
        ok, msg = _ob_check(data, cand)
        rel = _ob_rel(saved[idx], cand)
        print(f"CLFRESH reference q_len={q_len} input={idx} ok={ok} replay_miss={rel:.2e} {msg}",
              flush=True)
        assert ok and rel < 1e-4, f"fresh q_len {q_len} input {idx}: {msg}, replay miss {rel}"
    # The captured graph after the shape-based speculation is wrong by 64 tokens.
    data["seq_lens"].sub_(64)
    pos = data["seq_lens"].long()[:, None] + torch.arange(q_len, device="cuda")
    data["positions"].copy_(pos.reshape(-1))
    data["cache_loc"].copy_(torch.gather(data["page_table"].long(), 1, pos).reshape(-1))
    for _ in range(50):
        graphs["candidate"].replay()
    torch.cuda.synchronize()
    ok, msg = _ob_check(data, cand)
    clean, ws = _ob_ws_check(data["qkv"].device)
    print(f"CLFRESH SPEC miss q_len={q_len} ok={ok} clean={clean} {msg} {ws}", flush=True)
    assert ok and clean, f"SPEC miss q_len {q_len}: {msg} {ws}"
    del data, bank, graphs, outs, saved, verdict
    torch.cuda.empty_cache()


_fg_saved_tf32 = torch.backends.cuda.matmul.allow_tf32
try:
    _fg_run()
    _cl_stress(1)
    _cl_stress(8)
finally:
    torch.backends.cuda.matmul.allow_tf32 = _fg_saved_tf32
print(f"CLGATE PASS prepare_elapsed={_fg_clock.monotonic() - _FG_STARTED:.2f}", flush=True)
'''


def main():
    source = (ROOT / "muse_attention_decode/submission.py").read_text()
    assert source.count("#define ONE_CL 0") == 1
    decl = "std::vector<int64_t> attn_one_prepare();"
    assert source.count(decl) == 1
    source = source.replace(decl, decl + "\nstd::vector<int64_t> attn_one_math_resources();")
    marker = "std::vector<int64_t> attn_one_prepare() {"
    assert source.count(marker) == 1
    diag = '''std::vector<int64_t> attn_one_math_resources() {
    std::vector<int64_t> out;
#define MRES(L,M) { Instance<L,M>::prepare(); cudaFuncAttributes a = {}; \\
    cudaFuncGetAttributes(&a, Instance<L,M>::kernel()); \\
    out.insert(out.end(), {L,M,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,Instance<L,M>::BYTES,Instance<L,M>::NT,Instance<L,M>::per_sm}); }
    ONE_INSTANCES(MRES)
#undef MRES
    return out;
}
'''
    source = source.replace(marker, diag + "\n" + marker)
    exports = 'functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],'
    assert source.count(exports) == 1
    source = source.replace(
        exports,
        'functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop", "attn_one_math_resources"],')
    candidate = source.replace("#define ONE_CL 0", "#define ONE_CL 1")
    header, rest = candidate.split("\n", 1)
    candidate = header + "\nimport time as _fg_clock\n_FG_STARTED = _fg_clock.monotonic()\n" + rest
    control = source.replace('name="okc_attn_one_v1"', 'name="okc_attn_cl_gate_control"')
    assert control != source
    runner = base.RUNNER.format(variants=["_one"], coop_probe=True)
    head, sep, _ = runner.rpartition("\nimport os as _os\n")
    assert sep
    # The registered FP32 gate body up to its run call; this file runs the gate and the
    # cluster stress under one try/finally.
    body = fp32.BODY
    tail = "_fg_saved_tf32=torch.backends.cuda.matmul.allow_tf32\ntry:\n    _fg_run()\n"
    assert body.count(tail) == 1
    body = body[:body.index(tail)]
    Path(sys.argv[1]).write_text(candidate + head + "\n_FG_CONTROL_SRC = " + repr(control)
                                 + "\n" + body + STRESS)


if __name__ == "__main__":
    main()
