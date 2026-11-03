"""How many local experts per input the layer's routing tolerances would let a kernel skip.

python tools/box_route_slack.py --router-seed S --judge J.json [--loads "decode 1,verify 1"] [--runs 60]

Measurement only, no kernel. For each fresh input (local_start = its busiest range, as on the
judge) a local expert in the reference's top 8 is droppable for a token when check_routing would
still pass with it replaced by that token's best non-local expert outside its top 8: the local
expert beats the 9th best by at most KEEP_TOL, the replacement is within ROUTE_TOL of the 8th
best, and the routes / distinct totals stay inside LOCAL_SLACK / DISTINCT_SLACK. An expert is
skipped only when every token routed to it can drop it. Greedy over experts by fewest tokens.
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_run  # noqa: E402


def skippable(ref, biased, ls, km=0.0, wm=0.0, cm=0.0):
    best = biased.topk(ref.TOP_K + 16, dim=-1)
    top, vals = best.indices[:, :ref.TOP_K], best.values
    ninth = vals[:, ref.TOP_K:ref.TOP_K + 1]
    eighth = vals[:, ref.TOP_K - 1:ref.TOP_K]
    keep_tol = (ref.KEEP_TOL[0] if len(top) <= ref.KEEP_TOKENS else ref.KEEP_TOL[1]) - km
    is_local = lambda e: (e - ls) % ref.N_EXPERTS < ref.LOCAL_EXPERTS  # noqa: E731
    routes = int(is_local(top).sum())
    experts = sorted(set(int(e) for e in top[is_local(top)].tolist()))
    distinct = len(experts)
    # a replacement for token t: a non-local expert ranked 9th or later within ROUTE_TOL of the 8th
    has_repl = []
    for t in range(len(top)):
        ok = False
        for j in range(ref.TOP_K, ref.TOP_K + 16):
            e = int(best.indices[t, j])
            if float(eighth[t] - vals[t, j]) > ref.ROUTE_TOL - wm:
                break
            if not is_local(torch.tensor(e)):
                ok = True
                break
        has_repl.append(ok)
    min_routes = routes * (1 - ref.LOCAL_SLACK[0]) - ref.LOCAL_SLACK[1] + cm
    min_distinct = distinct * (1 - ref.DISTINCT_SLACK[0]) - ref.DISTINCT_SLACK[1] + cm
    toks = {e: [(t, s) for t in range(len(top)) for s in range(ref.TOP_K) if int(top[t, s]) == e] for e in experts}
    skipped, cur_routes, cur_distinct, used = [], routes, distinct, set()
    for e in sorted(experts, key=lambda e: len(toks[e])):
        ts = toks[e]
        if any(float(vals[t, s] - ninth[t]) > keep_tol or not has_repl[t] or t in used for t, s in ts):
            continue
        if cur_routes - len(ts) < min_routes or cur_distinct - 1 < min_distinct:
            continue
        skipped.append(e)
        cur_routes -= len(ts)
        cur_distinct -= 1
        used.update(t for t, _ in ts)  # one replacement per token keeps the greedy check simple
    return distinct, len(skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--loads", default="decode 1,verify 1")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--margins", default="0,0,0", help="keep, window and count margins, as a file would use")
    a = ap.parse_args()
    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = box_run.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    _, cases = box_run.read_judge(a.judge)
    keep = {s.strip() for s in a.loads.split(",")}
    cases = [c for c in cases if c.get("load") in keep]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    with torch.no_grad():
        for case in cases:
            rows = []
            for j in range(a.runs):
                c = {**case, "seed": case["seed"] + j}
                ls = ref.case_fields(c, weights)["local_start"]
                front = ref.ref_front(ref.generate_input(**{**c, "local_start": ls}), weights)
                rows.append(skippable(ref, front["biased"], ls, *(float(x) for x in a.margins.split(","))))
            d = [r[0] for r in rows]
            k = [r[1] for r in rows]
            print(f"SLACK {case['load']} {case['layer']} margins {a.margins}: mean distinct {statistics.mean(d):.2f}, mean skippable "
                  f"{statistics.mean(k):.2f}, inputs with >= 1 skippable {sum(x > 0 for x in k)} / {len(k)}, "
                  f"histogram {dict(sorted({x: k.count(x) for x in set(k)}.items()))}", flush=True)


if __name__ == "__main__":
    main()
