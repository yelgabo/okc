"""CL-1.2 phase B correctness (docs/plans/2026-11-04-moe-cluster-rewrite.md task 1.2).

Part A, reference: 32 local experts (run_unit_math.make_weights, per-expert global scales), inputs from
reference.generate_input through cl_bench.pick_inputs (judge-like routing), 20 per T at T = 1, 8, 32. The host
computes the intermediate with a copy of reference.py's math (x @ W13^T, silu(gate) * up, FP32, then BF16: the
kernel's operand) and y_ref = inter_bf16 @ W2^T in FP32. The kernel runs from that intermediate; the host sums
each expert's S slice partials in slice order (the phase 2 combine) and applies gs, BF16, (1 + w). Checks:
v within 2^-7 relative per element (with an absolute floor of 2^-7 x 2^-10 x the row's RMS for elements near 0;
the strict count is printed too); the kernel's per-token sum of squares (S = 1 experts) within 1e-5 relative of the
FP64 sum of its own v^2; and the whole MoE output (host route scale and weighted add) through
reference.check_implementation.

Part B, bitwise against helper0: run_unit_math's inputs and the helper0 debug build (tests/helper0_debug); the
kernel runs from helper0's own intermediate rows. Where the slice K ranges match helper0's per-warp ranges (S = 1,
and S = 8 summed in slice order) the raw FP32 sums, ybuf values and per-unit sums of squares must be bitwise
equal; other policies report their mismatch counts.

Usage (box): python muse_moe_w4a16/cl/tests/test_phase_b.py --lock /root/okc/.build/box/gpu.lock \
    --json muse_moe_w4a16/cl/profiles/1.2/test.json
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

import cl_phase_b_bench as pb  # noqa: E402
import cl_schedule  # noqa: E402

TWO62 = 4.611686018427388e18
REL = 2.0 ** -7
EMAX = 384 * 8


def nwf(normw):
    return (1.0 + normw.float()).to(torch.bfloat16).float()


def expert_slices(d):
    out = {}
    for i in range(d["n_slices"]):
        out.setdefault(d["slice_expert"][i], []).append(i)
    return out


def phase2(runner, d, w2gs, normw):
    """Host phase 2 per expert: (v [ntok, 4096], raw total or None, kernel sumsq or None)."""
    nw = nwf(normw)
    res = {}
    for e, sl in expert_slices(d).items():
        n = d["slice_ntok"][sl[0]]
        if len(sl) == 1:
            res[e] = (runner.y[sl[0], :n].clone(), None, runner.ss[sl[0], :n].clone())
        else:
            acc = torch.zeros((n, 4096), dtype=torch.float32, device="cuda")
            for i in sl:
                acc = acc + runner.y[i, :n]
            gsc = w2gs[e] * torch.tensor(TWO62, dtype=torch.float32, device="cuda")
            v = (acc * gsc).to(torch.bfloat16).float() * nw
            res[e] = (v, acc, None)
    return res


_DQ = {}


def _dq(wref, name, e):
    from nvfp4 import dequantize
    if (name, e) not in _DQ:
        _DQ[(name, e)] = dequantize(wref[name][e], wref[name + "_sf"][e], wref[name + "_gs"][e])
    return _DQ[(name, e)]


def host_reference(wref, data, d):
    """Intermediate rows in tok_list order [pairs, 4096] BF16, and per expert y_ref FP32 from the BF16 operand.
    The math is reference.ref_kernel's (dequantize, x @ W13^T, silu(gate) * up, @ W2^T)."""
    x = data["hidden_states"].float()
    inter = torch.zeros((256, 4096), dtype=torch.bfloat16, device="cuda")
    yref = {}
    for e, sl in expert_slices(d).items():
        t0, n = d["slice_tok0"][sl[0]], d["slice_ntok"][sl[0]]
        tok = torch.tensor(d["tok_list"][t0:t0 + n], dtype=torch.long, device="cuda")
        h = x[tok] @ _dq(wref, "w13", e).T
        it = (torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]).to(torch.bfloat16)
        inter[t0:t0 + n] = it
        yref[e] = it.float() @ _dq(wref, "w2", e).T
    return inter, yref


def combine_out(data, d, p2):
    """out[t] = bf16(sum over local slots of g * v * rsqrt(mean(v^2) + eps)) in FP32, as the reference forms it."""
    T = data["hidden_states"].shape[0]
    out = torch.zeros((T, 4096), dtype=torch.float32, device="cuda")
    for e, sl in expert_slices(d).items():
        t0, n = d["slice_tok0"][sl[0]], d["slice_ntok"][sl[0]]
        tok = torch.tensor(d["tok_list"][t0:t0 + n], dtype=torch.long, device="cuda")
        g = torch.tensor(d["tok_gate"][t0:t0 + n], dtype=torch.float32, device="cuda")
        v = p2[e][0]
        r = g * torch.rsqrt((v.double() ** 2).sum(-1) / 4096 + 1e-8).float()
        out.index_add_(0, tok, v * r[:, None])
    return out.to(torch.bfloat16)


def part_a(ext, wref, w2u, w2gs, normw, ref, Ts, n_inputs, policies, modes):
    import cl_bench
    results = []
    for T in Ts:
        picked, seen, _ = cl_bench.pick_inputs(ref, T, "normal", 1000, n_inputs, None, n_inputs)
        tmax = 8 if T <= 8 else 32
        agg = {}
        for seed, dist in picked:
            data = ref.generate_input(bs=T, sharding="ep8", seed=seed, dist="normal")
            for pol in policies:
                b = pb.make_sched(data["topk_ids"], data["topk_weights"], pol)
                d, n_sl, sc = pb.sched_dims(b)
                inter, yref = host_reference(wref, data, d)
                for mode in modes:
                    xmode, order, rot = pb.MODES[mode][:3]
                    runner = pb.Runner(ext, w2u, w2gs, normw, tmax, n_sl, sc)
                    runner.sched.copy_(pb.sched_tensor(b))
                    runner.inter.copy_(inter)
                    runner.y.fill_(float("nan"))
                    runner.launch(xmode, order, rot)
                    torch.cuda.synchronize()
                    p2 = phase2(runner, d, w2gs, normw)
                    nw = nwf(normw)
                    worst, worst_floor, strict_bad, floor_bad, nel, ss_worst, ss_n = 0.0, 0.0, 0, 0, 0, 0.0, 0
                    for e, (v, _, kss) in p2.items():
                        vr = yref[e] * nw
                        err = (v - vr).abs()
                        rel = err / vr.abs().clamp_min(1e-30)
                        rms = vr.pow(2).mean(-1, keepdim=True).sqrt()
                        relf = err / torch.maximum(vr.abs(), rms * 2.0 ** -10)
                        worst = max(worst, float(rel.max()))
                        worst_floor = max(worst_floor, float(relf.max()))
                        strict_bad += int((rel > REL).sum())
                        floor_bad += int((relf > REL).sum())
                        nel += v.numel()
                        if not torch.isfinite(v).all():
                            floor_bad += v.numel()
                        if kss is not None:
                            s64 = (v.double() ** 2).sum(-1)
                            ss_worst = max(ss_worst, float(((kss.double() - s64).abs() / s64).max()))
                            ss_n += kss.numel()
                    out = combine_out(data, d, p2)
                    ok, msg = ref.check_implementation(data, out, wref)
                    key = (pol, mode)
                    a = agg.setdefault(key, {"inputs": 0, "elems": 0, "floor_bad": 0, "strict_bad": 0,
                                             "worst_rel_floor": 0.0, "worst_rel_strict": 0.0, "ss_tokens": 0,
                                             "ss_worst_rel": 0.0, "ref_pass": 0, "ref_msgs": []})
                    a["inputs"] += 1
                    a["elems"] += nel
                    a["floor_bad"] += floor_bad
                    a["strict_bad"] += strict_bad
                    a["worst_rel_floor"] = max(a["worst_rel_floor"], worst_floor)
                    a["worst_rel_strict"] = max(a["worst_rel_strict"], worst)
                    a["ss_tokens"] += ss_n
                    a["ss_worst_rel"] = max(a["ss_worst_rel"], ss_worst)
                    a["ref_pass"] += int(bool(ok))
                    if len(a["ref_msgs"]) < 3 or not ok:
                        a["ref_msgs"].append(f"seed {seed} distinct {dist} slices {n_sl}: {str(msg)[:90]}")
        for (pol, mode), a in agg.items():
            ok = a["floor_bad"] == 0 and a["ss_worst_rel"] <= 1e-5 and a["ref_pass"] == a["inputs"]
            print(f"PARTA T={T:2d} {pol:6s} {mode:6s} inputs {a['inputs']} elems {a['elems']} bad(2^-7, floor) "
                  f"{a['floor_bad']} worst {a['worst_rel_floor']:.2e} | strict bad {a['strict_bad']} worst "
                  f"{a['worst_rel_strict']:.2e} | sumsq tokens {a['ss_tokens']} worst rel {a['ss_worst_rel']:.2e} | "
                  f"reference {a['ref_pass']}/{a['inputs']} | {'PASS' if ok else 'FAIL'} | {a['ref_msgs'][0]}",
                  flush=True)
            results.append({"T": T, "policy": pol, "mode": mode, "pass": ok, "distinct_scanned": seen, **a})
    return results


def helper0_dump(rum, build_debug, pub, ext, st, data):
    T = data["hidden_states"].shape[0]
    _, scr = rum.run_helper0(pub, ext, st, data, True)
    inter_d = scr[2][: EMAX * 4096].view(EMAX, 4096)
    ybuf_d = scr[1][EMAX: EMAX + EMAX * 4096].view(EMAX, 4096)
    ss_d = scr[1][:EMAX]
    dbgw = scr[3]
    sq_d = dbgw[: build_debug.DBG_SQ_FLOATS].view(torch.float32).view(EMAX, 256)
    accB_d = dbgw[build_debug.DBG_ACC_B_OFF: build_debug.DBG_WORDS].view(torch.float32).view(EMAX, 4096)
    rt = dbgw[build_debug.DBG_ROUTE_OFF:].cpu()
    njF, njS = int(rt[0]), int(rt[1])
    assert njF == 0, njF
    experts = {}
    for j in range(njS):
        e, s0, c = int(rt[2 + 3 * j]), int(rt[3 + 3 * j]), int(rt[4 + 3 * j])
        if e in experts:
            p0, n = experts[e]
            assert p0 + n == s0
            experts[e] = (p0, n + c)
        else:
            experts[e] = (s0, c)
    return {"inter": inter_d, "ybuf": ybuf_d, "ss": ss_d, "sq": sq_d, "acc": accB_d, "experts": experts, "T": T}


def bits32(t):
    return t.contiguous().view(torch.int32)


def part_b(ext, pub, st, w2u, w2gs, normw, Ts, n_inputs, seed, policies, modes):
    import build_debug
    import run_unit_math as rum
    defs = rum.build_defines(pub)
    sfx = pub._MA_SFX_IF_PRMT
    results = []
    for T in Ts:
        dext = build_debug.build(pub, defs[T], sfx)
        tmax = 8 if T <= 8 else 32
        for k in range(n_inputs):
            data = rum.make_input(T, seed * 1000 + T * 10 + k, st["dev"], outliers=(k == n_inputs - 1))
            h0 = helper0_dump(rum, build_debug, pub, dext, st, data)
            for pol in policies:
                b = pb.make_sched(data["topk_ids"], data["topk_weights"], pol)
                d, n_sl, sc = pb.sched_dims(b)
                es = expert_slices(d)
                inter = torch.zeros((256, 4096), dtype=torch.bfloat16, device="cuda")
                for e, sl in es.items():
                    t0, n = d["slice_tok0"][sl[0]], d["slice_ntok"][sl[0]]
                    p0, n0 = h0["experts"][e]
                    assert n0 == n, (e, n0, n)
                    toks = sorted(d["tok_list"][t0:t0 + n])
                    assert toks == d["tok_list"][t0:t0 + n]
                    inter[t0:t0 + n] = h0["inter"][p0:p0 + n]
                for mode in modes:
                    xmode, order, rot = pb.MODES[mode][:3]
                    runner = pb.Runner(ext, w2u, w2gs, normw, tmax, n_sl, sc, dbg=True)
                    runner.sched.copy_(pb.sched_tensor(b))
                    runner.inter.copy_(inter)
                    runner.launch(xmode, order, rot)
                    torch.cuda.synchronize()
                    p2 = phase2(runner, d, w2gs, normw)
                    c = {"T": T, "input": k, "policy": pol, "mode": mode, "S": sorted({len(s) for s in es.values()}),
                         "acc_elems": 0, "acc_bad": 0, "y_elems": 0, "y_bad": 0, "sq_units": 0, "sq_bad": 0,
                         "ss_tokens": 0, "ss_vs_h0_max_rel": 0.0, "ss_vs_units_max_rel": 0.0}
                    for e, sl in es.items():
                        p0, n = h0["experts"][e]
                        ref_acc, ref_y = h0["acc"][p0:p0 + n], h0["ybuf"][p0:p0 + n]
                        v, total, kss = p2[e]
                        if len(sl) == 1:
                            total = runner.dacc[sl[0], :n]
                            ku = runner.dsq[sl[0], :n]
                            c["sq_units"] += ku.numel()
                            c["sq_bad"] += int((bits32(ku) != bits32(h0["sq"][p0:p0 + n])).sum())
                            c["ss_tokens"] += n
                            c["ss_vs_h0_max_rel"] = max(c["ss_vs_h0_max_rel"], float(
                                ((kss - h0["ss"][p0:p0 + n]).abs() / h0["ss"][p0:p0 + n].abs()).max()))
                            us = h0["sq"][p0:p0 + n].double().sum(-1)
                            c["ss_vs_units_max_rel"] = max(c["ss_vs_units_max_rel"], float(
                                ((kss.double() - us).abs() / us).max()))
                        c["acc_elems"] += total.numel()
                        c["acc_bad"] += int((bits32(total) != bits32(ref_acc)).sum())
                        c["y_elems"] += v.numel()
                        c["y_bad"] += int((bits32(v) != bits32(ref_y)).sum())
                    results.append(c)
                    print(f"PARTB T={T:2d} in{k} {pol:6s} {mode:6s} S {c['S']} raw-sum mismatches {c['acc_bad']}/"
                          f"{c['acc_elems']} | ybuf mismatches {c['y_bad']}/{c['y_elems']} | unit sumsq mismatches "
                          f"{c['sq_bad']}/{c['sq_units']} | row sumsq vs helper0 max rel {c['ss_vs_h0_max_rel']:.2e}, "
                          f"vs unit partials {c['ss_vs_units_max_rel']:.2e}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--tokens", default="1,8,32")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--policies", default="fill15,s1")
    ap.add_argument("--modes", default="b_rot1,b_nat,c_rot1")
    ap.add_argument("--bitwise-inputs", type=int, default=3)
    ap.add_argument("--bitwise-policies", default="s1,s8,fill15,s2")
    ap.add_argument("--bitwise-modes", default="b_nat,b_rot1")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    ext = pb.build()
    Ts = [int(t) for t in a.tokens.split(",")]
    pub_pre = None
    if not a.skip_b:
        import build_debug
        import run_unit_math as rum
        pub_pre = rum.load_public()
        build_debug.write_source(pub_pre)
        defs = rum.build_defines(pub_pre)
        for T in Ts:
            build_debug.build(pub_pre, defs[T], pub_pre._MA_SFX_IF_PRMT)
    print(f"builds {time.time() - t0:.1f} s; sha cl_phase_b.cuh {pb.sha(os.path.join(CL, 'cl_phase_b.cuh'))[:16]} "
          f"test {pb.sha(os.path.abspath(__file__))[:16]}", flush=True)
    import reference as ref
    out = {"argv": sys.argv, "sha": {"cl_phase_b.cuh": pb.sha(os.path.join(CL, "cl_phase_b.cuh")),
                                     "test_phase_b.py": pb.sha(os.path.abspath(__file__)),
                                     "cl_phase_b_bench.py": pb.sha(os.path.join(CL, "cl_phase_b_bench.py"))}}
    torch.backends.cuda.matmul.allow_tf32 = False
    with pb.GpuLock(a.lock):
        pub, wref, w2u, w2gs, normw, st = pb.prep_weights(keep_state=True)
        if not a.skip_a:
            out["part_a"] = part_a(ext, wref, w2u, w2gs, normw, ref, Ts, a.inputs, a.policies.split(","),
                                   a.modes.split(","))
        if not a.skip_b:
            out["part_b"] = part_b(ext, pub, st, w2u, w2gs, normw, Ts, a.bitwise_inputs, 505,
                                   a.bitwise_policies.split(","), a.bitwise_modes.split(","))
    fails = [r for r in out.get("part_a", []) if not r["pass"]]
    print(f"SUMMARY part A {len(out.get('part_a', [])) - len(fails)}/{len(out.get('part_a', []))} PASS", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
