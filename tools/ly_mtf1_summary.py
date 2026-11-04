#!/usr/bin/env python3
"""Medians over inputs of the ly_mtf1 decode 1 FAST1 path stamps (tools/dl_ly_mtf1_patch.py, run by
tools/ly_box_pmt.py --raw N), us from the first MoE CTA's entry. "mean" is the mean over CTAs, "max" the latest CTA;
the d_ rows are per-CTA differences averaged over CTAs.

    python3 tools/ly_mtf1_summary.py A.json [B.json ...]
"""
import json
import statistics as st
import sys

NAMES = {0: "entry", 4: "f1top", 5: "f1init", 7: "f1mask", 3: "f1issued", 1: "route", 6: "slot", 2: "aend"}
DIFFS = (("f1top", "entry"), ("f1init", "f1top"), ("f1mask", "f1top"), ("f1mask", "f1init"), ("f1issued", "f1mask"), ("slot", "f1issued"), ("aend", "slot"))


def main():
    for path in sys.argv[1:]:
        print(path)
        for row in json.load(open(path))["rows"]:
            agg = {}
            for x in row["inputs"]:
                moe = [m for m in x["raw"]["moe"] if m[0]]
                t0 = min(m[0] for m in moe)
                per = {n: [(m[k] - t0) / 1e3 if m[k] else None for m in moe] for k, n in NAMES.items()}
                for n, v in per.items():
                    v = [a for a in v if a is not None]
                    agg.setdefault(n, []).append(st.mean(v))
                    agg.setdefault(n + "_max", []).append(max(v))
                for a, b in DIFFS:
                    d = [p - q for p, q in zip(per[a], per[b]) if p is not None and q is not None]
                    agg.setdefault(f"d_{a}-{b}", []).append(st.mean(d))
            print(" ", row["case"]["layer"], len(row["inputs"]), "inputs")
            for k, v in agg.items():
                print(f"    {k:24s} {st.median(v):7.2f}")


if __name__ == "__main__":
    main()
