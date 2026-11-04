"""CL-1.1: correctness of the phase A cluster kernel (cl_phase_a.cuh).

For each T in --tokens, --inputs inputs from reference.generate_input(bs=T, sharding="ep8", seed) (the judge's
routing) with the problem's weights (reference.generate_weights(WEIGHTS_SEED)). The schedule is the cl_schedule.py
host mirror with the fill policy at target 15 (gate G0). The kernel writes each slice's BF16 intermediate to the
global copy; the test assembles every routed local expert's [ntok, 4096] intermediate from its slices and compares:

1. against a local FP32 PyTorch reference, h = x @ dequantize(w13)^T, inter = silu(gate) * up, using nvfp4.dequantize
   (the function reference.py imports; reference.py is not edited). Reported: the strict element-wise relative
   error |k - r| / |r| (r != 0) against 2^-7, the same count for helper0's rows, the relative L2 overall and per row.
   Also reported: max |k - r| / max(|r|, rms of r's row). Gated: the per-row relative L2 at most 2^-7, and the
   kernel's over-bound count equal to helper0's;
2. bitwise against helper0's inter rows from the CL-0.5 debug build (helper0_debug/build_debug.py) of the public
   file, TINY at T = 1, TINY2 at T = 8, MID at T = 32, as tests/run_unit_math.py runs them.

Rows past ntok, and every slice column, are also checked to be written exactly once: the global copy is filled with
0xFFFF before each launch, and every element the assembly reads must have been overwritten.

Usage (box): python muse_moe_w4a16/cl/tests/test_phase_a.py --lock /root/okc/.build/box/gpu.lock
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.dirname(HERE)
MOE = os.path.dirname(CL)
sys.path.insert(0, MOE)
sys.path.insert(0, CL)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "helper0_debug"))

import cl_phase_a_bench as pab  # noqa: E402

REL_BOUND = 2.0 ** -7


def bits(t):
    return t.contiguous().view(torch.int16)


def kernel_inter(runner, data):
    """Per routed local expert: (token list, [ntok, 4096] bf16) assembled from the kernel's global copy."""
    sched_t, d = pab.schedule(data)
    n = d["n_slices"]
    assert n > 0, d["n_slices"]
    runner.inter.fill_(float("nan"))
    runner.inter.view(torch.int16).fill_(-1)
    runner.launch(data["hidden_states"].contiguous(), sched_t, n)
    torch.cuda.synchronize()
    g = runner.inter_view(n)
    experts = {}
    for i in range(n):
        e, c0, nc = d["slice_expert"][i], d["slice_col0"][i], d["slice_ncol"][i]
        t0, nt = d["slice_tok0"][i], d["slice_ntok"][i]
        if e not in experts:
            experts[e] = (d["tok_list"][t0:t0 + nt], torch.full((nt, 4096), -1, dtype=torch.int16, device="cuda"),
                          torch.zeros(4096, dtype=torch.int32, device="cuda"))
        toks, buf, cover = experts[e]
        buf[:, c0:c0 + nc] = bits(g[i, :nt, :nc])
        cover[c0:c0 + nc] += 1
    unwritten = 0
    out = {}
    for e, (toks, buf, cover) in experts.items():
        assert bool((cover == 1).all()), (e, "slice columns do not cover the expert exactly once")
        unwritten += int((buf == -1).sum())
        out[e] = (toks, buf.view(torch.bfloat16))
    return out, d, unwritten


def fp32_reference(weights, data, e, toks):
    from nvfp4 import dequantize
    x = data["hidden_states"][toks].float()
    w = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e])
    h = x @ w.T
    return torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]


def helper0_inter(pub, rum, ext, st, data):
    """helper0 debug build's inter rows per local expert: {e: (sorted token list, [n, 4096] bf16)}."""
    import build_debug
    out_d, scr = rum.run_helper0(pub, ext, st, data, True)
    inter = scr[2][: rum.EMAX * 4096].view(rum.EMAX, 4096)
    dbgw = scr[3]
    rt = dbgw[build_debug.DBG_ROUTE_OFF:].cpu()
    njF, njS = int(rt[0]), int(rt[1])
    assert njF == 0, njF
    jobs = [(int(rt[2 + 3 * j]), int(rt[3 + 3 * j]), int(rt[4 + 3 * j])) for j in range(njS)]
    T = data["hidden_states"].shape[0]
    pos = dbgw[build_debug.DBG_POS_OFF: build_debug.DBG_POS_OFF + T * 8].cpu().tolist()
    nE = sum(c for _, _, c in jobs)
    ent_tok = [None] * nE
    for i, q in enumerate(pos):
        if q >= 0:
            ent_tok[q] = i >> 3
    experts = {}
    for e, s0, c in jobs:
        if e in experts:
            p0, n = experts[e]
            assert p0 + n == s0
            experts[e] = (p0, n + c)
        else:
            experts[e] = (s0, c)
    return {e: (ent_tok[p0:p0 + n], inter[p0:p0 + n]) for e, (p0, n) in experts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,8,32")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--lock", default="")
    ap.add_argument("--no-helper0", action="store_true", help="skip the bitwise comparison with helper0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    Ts = [int(t) for t in a.tokens.split(",")]
    t0 = time.time()
    ext = pab.build()
    pub = pab.load_public()
    import run_unit_math as rum
    exts = {}
    if not a.no_helper0:
        import build_debug
        defs = rum.build_defines(pub)
        exts = {T: build_debug.build(pub, defs[T], pub._MA_SFX_IF_PRMT) for T in Ts}
    print(f"builds {time.time() - t0:.1f} s; cl_phase_a.cuh {pab.sha(os.path.join(CL, 'cl_phase_a.cuh'))[:16]} "
          f"cl_math.cuh {pab.sha(os.path.join(CL, 'cl_math.cuh'))[:16]}", flush=True)
    import reference
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    with pab.GpuLock(a.lock):
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        assert pub._ma_scales_prmt_exact(weights["w13_sf"][:32], weights["w2_sf"][:32]), "PRMT scale path not exact"
        w13u, gs = pab.pack_w13(pub, weights)
        runner = pab.Runner(ext, w13u, gs)
        st = None
        if not a.no_helper0:
            local = {k: weights[k][:32].contiguous() for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
            local["post_expert_norm"] = weights["post_norm_w"]
            st = {"ma_sc6": False, "guards": [], "dev": torch.device("cuda"),
                  "nsm": torch.cuda.get_device_properties(0).multi_processor_count}
            pub._fork_native_ma_prepare(local, st)
            del local
            torch.cuda.synchronize()
        for T in Ts:
            for k in range(a.inputs):
                seed = a.seed + 100 * T + k
                data = reference.generate_input(bs=T, sharding="ep8", seed=seed)
                kin, d, unwritten = kernel_inter(runner, data)
                r = {"T": T, "seed": seed, "distinct": len(kin), "n_slices": d["n_slices"], "unwritten": unwritten,
                     "max_ntok": max(v[1].shape[0] for v in kin.values())}
                worst, over, nel, rowrel, se, sr, rs_worst = 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0
                refs = {}
                for e, (toks, kv) in kin.items():
                    ref = fp32_reference(weights, data, e, toks)
                    refs[e] = ref
                    kf = kv.float()
                    nz = ref != 0
                    rel = ((kf - ref).abs() / ref.abs().clamp(min=1e-30))[nz]
                    worst = max(worst, float(rel.max()))
                    over += int((rel > REL_BOUND).sum())
                    nel += int(nz.sum())
                    rms = ref.pow(2).mean(-1, keepdim=True).sqrt()
                    rs_worst = max(rs_worst, float(((kf - ref).abs() / torch.maximum(ref.abs(), rms)).max()))
                    rowrel = max(rowrel, float(((kf - ref).norm(dim=-1) / ref.norm(dim=-1)).max()))
                    se += float((kf - ref).pow(2).sum())
                    sr += float(ref.pow(2).sum())
                r.update(rel_max=worst, rel_over=over, elems=nel, rowscaled_max=rs_worst, row_relL2_max=rowrel,
                         relL2=(se / sr) ** 0.5)
                if st is not None:
                    hin = helper0_inter(pub, rum, exts[T], st, data)
                    assert sorted(hin) == sorted(kin), (sorted(hin), sorted(kin))
                    mism = 0
                    for e, (toks, kv) in kin.items():
                        htoks, hv = hin[e]
                        assert list(htoks) == list(toks), (e, htoks, toks)
                        mism += int((bits(kv) != bits(hv)).sum())
                    r["helper0_mismatch"] = mism
                    hover = 0
                    for e, (toks, kv) in kin.items():
                        ref = refs[e]
                        nz = ref != 0
                        hrel = ((hin[e][1].float() - ref).abs() / ref.abs().clamp(min=1e-30))[nz]
                        hover += int((hrel > REL_BOUND).sum())
                    r["helper0_rel_over"] = hover
                results.append(r)
                print("CASE " + json.dumps(r), flush=True)
    ok = True
    for T in Ts:
        rs = [r for r in results if r["T"] == T]
        s = {"inputs": len(rs), "elems": sum(r["elems"] for r in rs), "rel_max": max(r["rel_max"] for r in rs),
             "rel_over": sum(r["rel_over"] for r in rs), "unwritten": sum(r["unwritten"] for r in rs),
             "rowscaled_max": max(r["rowscaled_max"] for r in rs), "row_relL2_max": max(r["row_relL2_max"] for r in rs),
             "relL2_max": max(r["relL2"] for r in rs),
             "distinct": sorted(set(r["distinct"] for r in rs)), "max_ntok": max(r["max_ntok"] for r in rs)}
        if st is not None:
            s["helper0_mismatch"] = sum(r["helper0_mismatch"] for r in rs)
            s["helper0_rel_over"] = sum(r["helper0_rel_over"] for r in rs)
            ok &= s["helper0_mismatch"] == 0
        # The strict element-wise bound is reported, not gated: BF16 rounding of gate, up and inter (CL-0.5 items 9
        # and 11) alone exceeds 2^-7 where |gate| is large and negative or h cancels, and helper0's rows exceed it
        # at the same elements. The gate is bitwise equality with helper0, every element written, and the per-row
        # relative L2 within 2^-7.
        ok &= s["unwritten"] == 0 and s["row_relL2_max"] <= REL_BOUND
        if st is not None:
            ok &= s["rel_over"] == s["helper0_rel_over"]
        print(f"SUMMARY T={T} " + json.dumps(s), flush=True)
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
