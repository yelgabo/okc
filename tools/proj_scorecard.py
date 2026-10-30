"""Standalone muse_bf16_proj scorecard from a judged result JSON (`okc status <job> --json`).

    python3 tools/proj_scorecard.py RESULT.json [--ref CSV] [--vs RESULT2.json]

Prints per case the judge-derived time (SGLang median over the paired ratio, replay overhead
removed), per load C_T = sum of the five operator times, the load ratio, and the score
sensitivity 1 / (6 C_T) in percent per microsecond. --ref adds a reference CSV
(load,m,layer,weight,baseline_us,solution_us,...) such as the independent reference's
results/v18_shapes.csv; --vs adds a second judged run. Columns marked "-" are absent.
"""

import csv
import json
import math
import sys


def cases_of(path):
    d = json.load(open(path))
    r = d["result"]
    out = {}
    for x in r["records"]:
        if x.get("event") != "benchmark":
            continue
        c = x["case"]
        out[(c["load"], c["layer"])] = (x["baseline_ns"] / 1e3, x["median_ns"] / 1e3)
    return d.get("id"), r.get("speedup"), out


def ref_of(path):
    out = {}
    for row in csv.DictReader(open(path)):
        out[(row["load"], row["layer"])] = (float(row["baseline_us"]), float(row["solution_us"]))
    return out


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    job, score, ours = cases_of(args.pop(0))
    ref = vs = None
    while args:
        a = args.pop(0)
        if a == "--ref":
            ref = ref_of(args.pop(0))
        elif a == "--vs":
            vs = cases_of(args.pop(0))
    print(f"job {job}: {score:.4f}x")
    loads = []
    for load, _ in ours:
        if load not in loads:
            loads.append(load)
    hdr = f"{'load':10s} {'layer':9s} {'sglang':>7s} {'ours':>7s}"
    if vs:
        hdr += f" {vs[0][:8]:>8s} {'d':>6s}"
    if ref:
        hdr += f" {'ref':>7s} {'ours-ref':>8s}"
    print(hdr)
    geo = 0.0
    for load in loads:
        cb = co = cv = cr = 0.0
        for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
            b, o = ours[(load, layer)]
            cb += b
            co += o
            line = f"{load:10s} {layer:9s} {b:7.2f} {o:7.2f}"
            if vs:
                v = vs[2].get((load, layer), (0, float("nan")))[1]
                cv += v
                line += f" {v:8.2f} {v - o:+6.2f}"
            if ref:
                rv = ref.get((load, layer), (0, float("nan")))[1]
                cr += rv
                line += f" {rv:7.2f} {o - rv:+8.2f}"
            print(line)
        geo += math.log(cb / co)
        line = f"{load:10s} {'C_T':9s} {cb:7.2f} {co:7.2f}"
        if vs:
            line += f" {cv:8.2f} {cv - co:+6.2f}"
        if ref:
            line += f" {cr:7.2f} {co - cr:+8.2f}"
        print(line + f"   ratio {cb / co:.4f}, {100 / (6 * co):.3f} %/us")
    print(f"geomean {math.exp(geo / len(loads)):.4f}")


if __name__ == "__main__":
    main()
