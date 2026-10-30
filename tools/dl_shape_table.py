"""Per-shape judge readings (our median, baseline median, us) from layer result JSONs, and the
per-shape mean of a reference group against one or more candidate readings.

    python3 tools/dl_shape_table.py --ref=A.json,B.json,... CAND.json [CAND2.json ...]

Each JSON is a committed profile (muse_decode_layer/profiles/*.json) or an `okc status --json`
result. Shapes are labelled by load and layer (d1s ... v32f); loads use 46 sliding + 16 full.
"""

import json
import math
import statistics
import sys

LOADS = ("decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32")
SHORT = {"decode 1": "d1", "decode 16": "d16", "decode 32": "d32", "verify 1": "v1",
         "verify 16": "v16", "verify 32": "v32"}


def readings(path):
    d = json.load(open(path))
    if "benchmarks" not in d:
        d = d.get("result", d)
    out = {}
    for b in d["benchmarks"]:
        c = b["case"]
        out[(c["load"], c["layer"])] = (b["median_ns"] / 1000, b["baseline_ns"] / 1000)
    return out, d.get("summary", {}).get("speedup", d.get("speedup"))


def main():
    refs, cands = [], []
    for a in sys.argv[1:]:
        if a.startswith("--ref="):
            refs = a.split("=", 1)[1].split(",")
        else:
            cands.append(a)
    rr = [readings(p) for p in refs]
    cc = [(p, readings(p)) for p in cands]
    print("shape  ref mean (min-max, n)  " + "  ".join(p.split("/")[-1][-21:-5] for p, _ in cc))
    for load in LOADS:
        for layer in ("sliding", "full"):
            k = (load, layer)
            vals = [r[0][k][0] for r in rr if k in r[0]]
            m = statistics.mean(vals) if vals else float("nan")
            line = f"{SHORT[load]}{layer[0]:<2} {m:7.1f} ({min(vals):.1f}-{max(vals):.1f}, {len(vals)})"
            for _, (cr, _) in cc:
                if k in cr:
                    line += f"  {cr[k][0]:7.1f} ({cr[k][0] - m:+.1f})"
            print(line)
    print("load (46 s + 16 f) / 62 and ratio to the ref mean load:")
    logs = {p: [] for p, _ in cc}
    for load in LOADS:
        def ld(r):
            return (46 * r[(load, "sliding")][0] + 16 * r[(load, "full")][0]) / 62
        refl = statistics.mean(ld(r[0]) for r in rr)
        line = f"{SHORT[load]:<4} ref {refl:6.1f}"
        for p, (cr, _) in cc:
            c = ld(cr)
            logs[p].append(math.log(refl / c))
            line += f"  {c:6.1f} ({refl / c:.4f})"
        print(line)
    print("ref speedups: " + " ".join(f"{s:.3f}" for _, s in rr if s))
    for p, (_, s) in cc:
        g = math.exp(sum(logs[p]) / len(logs[p]))
        print(f"{p.split('/')[-1]}: speedup {s}; geometric mean of ref load / cand load {g:.4f}")


if __name__ == "__main__":
    main()
