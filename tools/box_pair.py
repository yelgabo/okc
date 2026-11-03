"""Paired box A/B from tools/box_run.py JSONs of a candidate and its control (same seeds, same router seed).

python tools/box_pair.py --control c1.json [c2.json ...] --candidate k1.json [k2.json ...] [--check rs_check.json]

Per cell: net medians (us) of each side pooled over its JSONs, the median per-input ratio candidate / control
(replay i of both runs times the same fresh input), and each side's SGLang-paired "shown" time. With --check
(tools/box_rs_check.py on the candidate with the same seeds), the mean distinct local experts of the timed inputs
under the reference top 8 (= the control's picks up to near ties) and under the candidate's picks.
"""
import argparse
import json
import statistics


def cells(paths):
    out = {}
    for p in paths:
        for r in json.load(open(p))["rows"]:
            k = (r["case"]["load"], r["case"]["layer"])
            ov = r["replay_overhead"]["overhead_ns"]
            out.setdefault(k, []).append(dict(net=[(x - ov) / 1e3 for x in r["runs_ns"]],
                                              shown=r.get("median_ns", 0) / 1e3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", nargs="+", required=True)
    ap.add_argument("--candidate", nargs="+", required=True)
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    ctl, cand = cells(a.control), cells(a.candidate)
    dist = {}
    if a.check:
        for row in json.load(open(a.check))["rows"]:
            dist[(row["case"]["load"], row["case"]["layer"])] = row["inputs"]
    print("| cell | control net | candidate net | delta | per-input ratio | control shown | candidate shown |"
          " distinct ref -> picked |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in ctl:
        if k not in cand:
            continue
        cn = [x for r in ctl[k] for x in r["net"]]
        kn = [x for r in cand[k] for x in r["net"]]
        ratios = [b / c for rc, rk in zip(ctl[k], cand[k]) for c, b in zip(rc["net"], rk["net"])]
        n = min(len(r["net"]) for r in ctl[k] + cand[k])
        d = dist.get(k, [])[:n]
        ds = (f"{statistics.mean(x['ref_distinct'] for x in d):.2f} -> {statistics.mean(x['distinct'] for x in d):.2f}"
              if d else "-")
        print(f"| {k[0]} {k[1]} | {statistics.median(cn):.2f} | {statistics.median(kn):.2f} | "
              f"{statistics.median(kn) - statistics.median(cn):+.2f} | {statistics.median(ratios):.4f} | "
              f"{statistics.mean(r['shown'] for r in ctl[k]):.2f} | {statistics.mean(r['shown'] for r in cand[k]):.2f} |"
              f" {ds} |")


if __name__ == "__main__":
    main()
