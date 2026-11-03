"""Correctness of a routing-slack layer file on many fresh inputs, with the margin to each slack bound.

python tools/box_rs_check.py <layer.py> --router-seed S --judge J.json [--loads ...] [--inputs N]
    [--big-inputs M] [--exact] [--out f.json]

Per load and layer: N fresh inputs (seeds case seed + j, local_start from case_fields per input, as
tools/box_run.py times them), each run eagerly through kernel_fn and judged by reference.check_implementation.
Per input it records, under the reference's FP64 biased scores: the distinct local experts and local routes of
the reference top 8 and of the file's picks, the largest 9th-best gap among dropped local reference picks
(bound KEEP_TOL), the largest distance of a pick below the 8th best (bound ROUTE_TOL), the routes and distinct
counts above their bounds, and the largest |file FP32 score - reference FP64 score|. --exact sets the file's
score, route and distinct margins to 0, so its picks sit on the bounds as far as its own scores can tell.
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_run  # noqa: E402


TESTS = [  # TASK.md "Before timing, a run must pass these 13 tests"
    dict(batch=1, ctx=10, layer="sliding"),
    dict(batch=1, ctx=3000, layer="full"),
    dict(batch=8, ctx=2047, layer="sliding"),
    dict(batch=9, ctx=5000, layer="sliding", spread="mixed", page=1),
    dict(batch=3, ctx=6000, layer="full", q_len=8, dist="outliers"),
    dict(batch=5, ctx=5000, layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=33, ctx=20000, layer="full", spread="mixed", page=1),
    dict(batch=2, ctx=300000, layer="sliding", q_len=8),
    dict(batch=1, ctx=1048576, layer="full", q_len=8),
    dict(batch=4, ctx=7000, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=1, ctx=80000, layer="full", dist="soft", page=1),
    dict(batch=4, ctx=6000, layer="sliding", q_len=8, dist="soft"),
    dict(batch=16, ctx=20000, layer="full", spread="mixed", dist="soft", page=1),
]


def margins(ref, ids, biased, ls):
    T = ids.shape[0]
    ids = ids.long()
    best = biased.topk(ref.TOP_K + 1, dim=-1)
    top, v8, v9 = best.indices[:, :ref.TOP_K], best.values[:, ref.TOP_K - 1], best.values[:, ref.TOP_K]
    keep_tol = ref.KEEP_TOL[0] if T <= ref.KEEP_TOKENS else ref.KEEP_TOL[1]
    is_local = lambda x: (x - ls) % ref.N_EXPERTS < ref.LOCAL_EXPERTS  # noqa: E731
    kept = (top[:, :, None] == ids[:, None, :]).any(-1)
    dropped = is_local(top) & ~kept
    gaps = (best.values[:, :ref.TOP_K] - v9[:, None])[dropped]
    below = (v8[:, None] - biased.gather(1, ids)).max()
    r, d = ref._local_load(ids, ls)
    rr, rd = ref._local_load(top, ls)
    return dict(ref_routes=rr, ref_distinct=rd, routes=r, distinct=d, dropped=int(dropped.sum()),
                keep_margin=float(keep_tol - gaps.max()) if gaps.numel() else None,
                win_margin=float(ref.ROUTE_TOL - below),
                route_margin=r - (rr * (1 - ref.LOCAL_SLACK[0]) - ref.LOCAL_SLACK[1]),
                distinct_margin=d - (rd * (1 - ref.DISTINCT_SLACK[0]) - ref.DISTINCT_SLACK[1]))


def gap_errors(ref, ours, b):
    """Error of the two quantities the slack thresholds: a top-8 pick's lead over the 9th best, and a candidate's
    distance below the 8th best, each with the file's scores against the reference's (reference's own ranking)."""
    k = ref.TOP_K
    rb, ro = b.topk(k + 1, dim=-1), ours.topk(k + 1, dim=-1)
    lead_ref = rb.values[:, :k] - rb.values[:, k:]
    lead_ours = ours.gather(1, rb.indices[:, :k]) - ro.values[:, k:]
    near = b >= rb.values[:, k - 1:k] - 0.06
    below_ref = rb.values[:, k - 1:k] - b
    below_ours = ro.values[:, k - 1:k] - ours
    return dict(lead_err=float((lead_ours - lead_ref).abs().max()),
                below_err=float((below_ours - below_ref).abs()[near].max()),
                tok_err=[float(x) for x in (ours - b).abs().masked_fill(~near, 0).max(-1).values])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--loads", default="")
    ap.add_argument("--inputs", type=int, default=100, help="fresh inputs per cell at T <= 8")
    ap.add_argument("--big-inputs", type=int, default=10, help="fresh inputs per cell at T > 8")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--tests", action="store_true", help="also TASK.md's 13 test shapes")
    ap.add_argument("--test-inputs", type=int, default=20)
    ap.add_argument("--e4m3", action="store_true", help="also compare the scores with the E4M3-query reference")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = box_run.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    _, cases = box_run.read_judge(a.judge)
    cases = cases or [dict(c, seed=1000 + i) for i, c in enumerate(box_run.CASES["muse_decode_layer"])]
    if a.loads:
        keep = {s.strip() for s in a.loads.split(",")}
        cases = [c for c in cases if c.get("load") in keep]
    if a.tests:
        cases += [dict(c, load="test", seed=77000 + 100 * i) for i, c in enumerate(TESTS)]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    mod = box_run.load_module("okc_submission", a.submission)
    if a.exact:
        mod.RS_KEEP_MARGIN, mod.RS_WIN_MARGIN, mod.RS_RANK_EPS, mod.RS_COUNT_MARGIN = 0.0, 0.0, 0.0, 0.0
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    bias = weights["router_bias"].float()
    rows = []
    for case in cases:
        T = case.get("batch", 1) * case.get("q_len", 1)
        n = a.test_inputs if case.get("load") == "test" else a.inputs if T <= 8 else a.big_inputs
        res = []
        for jdx in range(n):
            c = {**case, "seed": case["seed"] + jdx}
            ls = ref.case_fields(c, weights)["local_start"]
            data = ref.generate_input(**{**c, "local_start": ls})
            mod._RS_DEBUG = []
            with torch.no_grad():
                out = mod.kernel_fn(data, state)
                torch.cuda.synchronize()
                logits = mod._RS_DEBUG[-1].clone() if mod._RS_DEBUG else None
                mod._RS_DEBUG = None
                raw = ref.collect(data, out)
                ok, msg = ref.check_implementation(data, raw, weights)
                front = ref.ref_front(data, weights)
                m = margins(ref, out[2], front["biased"], ls)
                if logits is not None:
                    ours = torch.sigmoid(logits.float() * ref.ROUTER_MULTIPLIER) + bias
                    b = front["biased"]
                    near = b >= b.topk(ref.TOP_K, dim=-1).values[:, -1:] - 0.06
                    m["score_err"] = float((ours - b).abs()[near].max())
                    m["score_err_all"] = float((ours - b).abs().max())
                    m.update(gap_errors(ref, ours, b))
                    if a.e4m3:
                        b8 = ref.ref_front(data, weights, torch.float8_e4m3fn)["biased"]
                        m["e4m3_err"] = float((ours - b8).abs()[near].max())
                        m["e4m3_ref_err"] = float((b8 - b).abs()[near].max())
            m.update(ok=bool(ok), msg=msg if not ok else "", seed=c["seed"], local_start=ls)
            res.append(m)
            if not ok:
                print(f"FAIL {case['load']} {case['layer']} seed {c['seed']}: {msg}", flush=True)
        km = [r["keep_margin"] for r in res if r["keep_margin"] is not None]
        se = [r["score_err"] for r in res if "score_err" in r]
        bd = [r["ref_distinct"] for r in res]
        ad = [r["distinct"] for r in res]
        print(f"RSCHECK {case['load']} {case['layer']}: {sum(r['ok'] for r in res)}/{n} pass; distinct ref "
              f"{statistics.mean(bd):.2f} -> picked {statistics.mean(ad):.2f}; routes ref "
              f"{statistics.mean(r['ref_routes'] for r in res):.2f} -> {statistics.mean(r['routes'] for r in res):.2f};"
              f" inputs with a drop {sum(r['dropped'] > 0 for r in res)}; min margins: keep "
              f"{min(km) if km else float('nan'):.4f} win {min(r['win_margin'] for r in res):.4f} routes "
              f"{min(r['route_margin'] for r in res):.2f} distinct {min(r['distinct_margin'] for r in res):.2f};"
              f" max score err near the top {max(se) if se else float('nan'):.2e} (p99 "
              f"{sorted(se)[int(0.99 * (len(se) - 1))] if se else float('nan'):.2e}, all experts "
              f"{max(r.get('score_err_all', 0) for r in res):.2e})", flush=True)
        if se:
            le = sorted(r["lead_err"] for r in res)
            be = sorted(r["below_err"] for r in res)
            pos = [round(max(r["tok_err"][i] for r in res), 4) for i in range(len(res[0]["tok_err"]))]
            print(f"RSERR {case['load']} {case['layer']}: lead-over-9th error max {le[-1]:.2e} median "
                  f"{statistics.median(le):.2e}; below-8th error max {be[-1]:.2e} median {statistics.median(be):.2e};"
                  f" near-top score error by token {pos}" + (
                      f"; against the E4M3-query reference max {max(r['e4m3_err'] for r in res):.2e}, E4M3 reference"
                      f" against the reference max {max(r['e4m3_ref_err'] for r in res):.2e}" if a.e4m3 else ""),
                  flush=True)
        rows.append(dict(case=case, inputs=res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(submission=a.submission, exact=a.exact, rows=rows), f)


if __name__ == "__main__":
    main()
