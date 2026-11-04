"""Decode 16 / 32 timeline tables (critical path, floor, excess) from a tools/ly_tl_run.py trace directory.

python3 tools/ly_d16_tltab.py DIR   (cells 0..3 = decode 16 sliding, decode 16 full, decode 32 sliding, decode 32 full)

Critical path = this kernel's median end minus the previous kernel's end (per replay, then the median); gap = start
minus the previous end (negative: PDL overlap). Floors at 4.44 TB/s from "Decode 16 / 32 sliding: floor and gap"; the
MoE floor is taken at the judge's distinct-expert target (7.5 / 9.0), not the profiled inputs' own count.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ly_tl_parse import replays  # noqa: E402

NAMES = ["qkv sk2", "attention", "o sk2", "b1", "pooler + router sk2", "route", "moe_fused", "unpooler sk2",
         "b2 row boundary"]
FLOOR = {16: [6.39, 1.89, 2.84, 0.35, 13.26, 0, 47.9, 11.36, 0.35],
         32: [6.39, 3.78, 2.84, 0.71, 13.26, 0, 55.9, 11.36, 0.71]}
ATT_FULL = {16: 73.9, 32: 147.7}


def table(path, T, full):
    reps = replays(path)
    n = min(len(r) for r in reps)
    reps = [r for r in reps if len(r) == n]
    med = statistics.median
    t0 = [r[0]["ts"] for r in reps]
    print(f"\n{path} ({len(reps)} replays of {n} kernels)")
    print("| # | kernel | start | dur | end | gap | critical path | floor | over floor |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i in range(n):
        st = med(r[i]["ts"] - a for r, a in zip(reps, t0))
        du = med(r[i]["dur"] for r in reps)
        en = med(r[i]["ts"] + r[i]["dur"] - a for r, a in zip(reps, t0))
        gp = med(r[i]["ts"] - (r[i - 1]["ts"] + r[i - 1]["dur"]) for r in reps) if i else None
        cp = med(r[i]["ts"] + r[i]["dur"] - ((r[i - 1]["ts"] + r[i - 1]["dur"]) if i else r[0]["ts"]) for r in reps)
        fl = ATT_FULL[T] if (full and i == 1) else FLOOR[T][i]
        name = NAMES[i] if n == len(NAMES) else reps[0][i]["name"][:40]
        gps = "-" if gp is None else f"{gp:+.2f}"
        print(f"| {i} | {name} | {st:.2f} | {du:.2f} | {en:.2f} | {gps} | {cp:.2f} | {fl:g} | {cp - fl:.2f} |")
    span = med(r[-1]["ts"] + r[-1]["dur"] - r[0]["ts"] for r in reps)
    busy = med(sum(e["dur"] for e in r) for r in reps)
    print(f"span {span:.2f} us, sum of durations {busy:.2f} us")


if __name__ == "__main__":
    d = sys.argv[1]
    for c, T, full in ((0, 16, False), (1, 16, True), (2, 32, False), (3, 32, True)):
        table(os.path.join(d, f"cell{c}.json"), T, full)
