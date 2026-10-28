"""Which shapes and kernels can move the score, per run.

The score is the weight-averaged geometric mean of per-shape speedups, so a kernel's leverage is
its weight share times the log of how much faster its shape gets. For each kernel this prints the
score multiplier if that kernel's time were halved on every shape where it runs, and per shape
the multiplier if the shape reached its floor.

    python3 tools/leverage.py <problem>/profiles/submission.<job>.json [...]
"""

import json
import math
import re
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    return d.get("result", d) if "benchmarks" not in d else d


def case_key(case):
    return tuple(sorted((k, v) for k, v in case.items() if k != "seed"))


def label(case):
    skip = ("load", "seed", "weight", "sharding", "ctx", "heads", "page", "spread")
    return " ".join([case.get("load", "")] + [f"{k}={v}" for k, v in sorted(case.items())
                                               if k not in skip])


def kernels(report):
    rows = []
    for line in report.splitlines()[1:]:
        m = re.match(r"(.+?)\s+([\d.]+)\s*µs\s+([\d.]+)%\s+(\d+)\s*$", line)
        if m and not m.group(1).startswith("total"):
            name = re.sub(r"^void\s+|\(anonymous namespace\)::|sglang::|device::", "",
                          m.group(1))
            rows.append((name.split("(")[0].strip()[:44], float(m.group(2))))
    return rows


def report(path):
    run = load(path)
    benchmarks = run["benchmarks"]
    total_weight = sum(b["case"]["weight"] for b in benchmarks)
    profiles = {case_key(p["case"]): p["report"] for p in run.get("profiles", [])}
    print(f"\n## {run['problem']} {run['job_id']} {run['summary']['speedup']:.3f}x")
    print(f"{'shape':40} {'w%':>5} {'base':>7} {'ours':>7} {'floor':>7} {'spd':>6} "
          f"{'o/fl':>5} {'ceil':>6} {'gpu':>6} {'gap':>5}")
    leverage = defaultdict(float)
    ceiling = 0.0
    for b in benchmarks:
        case = b["case"]
        share = case["weight"] / total_weight
        base, ours, floor = (b[k] / 1e3 for k in ("baseline_ns", "median_ns",
                                                   "speed_of_light_ns"))
        shape_ceiling = share * math.log(ours / floor)
        ceiling += shape_ceiling
        rows = kernels(profiles.get(case_key(case), ""))
        gpu = sum(t for _, t in rows)
        print(f"{label(case)[:40]:40} {share * 100:5.1f} {base:7.1f} {ours:7.1f} {floor:7.1f} "
              f"{base / ours:6.3f} {ours / floor:5.1f} {math.exp(shape_ceiling):6.3f} "
              f"{gpu:6.1f} {ours - gpu:5.1f}")
        for name, t in rows:
            leverage[name] += share * -math.log(1 - 0.5 * t / ours)
        # Negative gaps come from overlapping kernels (PDL), whose rows double-count waits.
        leverage["(time outside kernels)"] += share * -math.log(1 - 0.5 * max(ours - gpu, 0)
                                                                 / ours)
    print(f"every shape at its floor: x{math.exp(ceiling):.2f}")
    print("score multiplier if the kernel's time were halved on every shape:")
    for name, value in sorted(leverage.items(), key=lambda x: -x[1])[:8]:
        print(f"  {math.exp(value):.4f}  {name}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        report(path)
