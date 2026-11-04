#!/usr/bin/env python3
"""Per-cell table of tools/ly_micro_ab.sh runs: control runs c_* and, per variant, its runs <var>_*.

    python3 tools/ly_micro_table.py DIR VAR [VAR ...]

Net us per run (median over the run's fresh inputs). Control spread = max - min over the control runs; threshold =
max(2 x spread, 0.5 us at decode 1 / verify 1, 1% of the control mean at the other loads). Delta = variant mean minus
control mean; also every variant run minus the mean control. Every run's reference check result is checked.
"""
import glob
import json
import os
import sys


def cells(path):
    d = json.load(open(path))
    return {(r["case"]["load"], r["case"]["layer"]): (r["median_net_ns"] / 1e3, r["check"][0]) for r in d["rows"]}


def main():
    root, variants = sys.argv[1], sys.argv[2:]
    ctl = {os.path.basename(p)[:-5]: cells(p) for p in sorted(glob.glob(os.path.join(root, "c_*.json")))}
    keys = list(next(iter(ctl.values())))
    bad = [(n, k) for n, c in ctl.items() for k in keys if not str(c[k][1]).upper().startswith(("PASS", "TRUE"))]
    print("controls:", " ".join(ctl))
    for v in variants:
        runs = {os.path.basename(p)[:-5]: cells(p) for p in sorted(glob.glob(os.path.join(root, f"{v}_*.json")))}
        bad += [(n, k) for n, c in runs.items() for k in keys if not str(c[k][1]).upper().startswith(("PASS", "TRUE"))]
        print(f"\n{v}: runs {' '.join(runs)}")
        print("| cell | " + " | ".join(ctl) + " | " + " | ".join(runs)
              + " | spread | delta (means) | per run | threshold | verdict |")
        print("|---" * (len(ctl) + len(runs) + 6) + "|")
        for k in keys:
            c = [ctl[n][k][0] for n in ctl]
            w = [runs[n][k][0] for n in runs]
            cm = sum(c) / len(c)
            sp = max(c) - min(c)
            short = k[0] in ("decode 1", "verify 1")
            thr = max(2 * sp, 0.5 if short else 0.01 * cm)
            d = sum(w) / len(w) - cm
            verdict = "win" if d < -thr else "loss" if d > thr else "inside"
            print(f"| {k[0]} {k[1]} | " + " | ".join(f"{x:.2f}" for x in c) + " | "
                  + " | ".join(f"{x:.2f}" for x in w) + f" | {sp:.2f} | {d:+.2f} | "
                  + " ".join(f"{x - cm:+.2f}" for x in w) + f" | {thr:.2f} | {verdict} |")
    print("\nchecks not passing:", bad or "none")


if __name__ == "__main__":
    main()
