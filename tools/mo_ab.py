#!/usr/bin/env python3
"""Paired per-cell deltas between box runs of the same seeds (tools/bx_box_run.py JSONs).

python tools/mo_ab.py --control c.json --candidate k.json [k2.json ...] [--controls c1.json c2.json ...]

Per cell: each side's net median (us) and the paired delta, the median over inputs of (candidate - control) for
input i of both runs (same seed). --controls lists every control-vs-control run of the same bytes; the per-cell
spread is the largest |paired delta| over all pairs of them. A knob wins or loses a cell when its delta exceeds the
threshold max(2 x spread, floor): the floor is 0.5 us at decode 1 and verify 1 and 1% of the control median at the
long loads, the standard error of the paired median at 20 runs (docs/box-audit.md section 5).
"""
import argparse
import itertools
import json
import statistics


def cells(path):
    out = {}
    for r in json.load(open(path))["rows"]:
        ov = r["replay_overhead"]["overhead_ns"]
        out[f"{r['case']['load']} {r['case']['layer']}"] = [(x - ov) / 1e3 for x in r["runs_ns"]]
    return out


def paired(a, b):
    n = min(len(a), len(b))
    return statistics.median(b[i] - a[i] for i in range(n))


def floor(cell, med):
    return 0.5 if cell.startswith(("decode 1 ", "verify 1 ")) else 0.01 * med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--candidate", nargs="+", required=True)
    ap.add_argument("--controls", nargs="*", default=[])
    a = ap.parse_args()
    ctl = cells(a.control)
    runs = [cells(p) for p in a.controls]
    spread = {}
    for x, y in itertools.combinations(runs, 2):
        for cell in x:
            if cell in y:
                spread[cell] = max(spread.get(cell, 0.0), abs(paired(x[cell], y[cell])))
    for path in a.candidate:
        k = cells(path)
        print(f"## {path} against {a.control}")
        print("| cell | control net | candidate net | paired delta | threshold | verdict |")
        print("|---|---:|---:|---:|---:|---|")
        for cell, kv in k.items():
            if cell not in ctl:
                continue
            med = statistics.median(ctl[cell])
            d = paired(ctl[cell], kv)
            t = max(2 * spread.get(cell, 0.0), floor(cell, med))
            verdict = "win" if d < -t else "loss" if d > t else "noise"
            print(f"| {cell} | {med:.2f} | {statistics.median(kv):.2f} | {d:+.2f} | {t:.2f} | {verdict} |")


if __name__ == "__main__":
    main()
