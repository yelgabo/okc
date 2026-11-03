#!/usr/bin/env python3
"""Per-load comparison of layer receipts (okc status --json files): medians per case in us, the load totals
(46 x sliding + 16 x full) and the T <= 8 profile rows.

    python3 tools/dl_unp_compare.py NEW.json REF.json [REF2.json ...]
"""
import json
import sys


def cases(path):
    d = json.load(open(path))
    out = {}
    for b in d["benchmarks"]:
        c = b["case"]
        out[(c["load"], c["layer"])] = b["median_ns"] / 1000.0
    return d, out


def main():
    paths = sys.argv[1:]
    data = [cases(p) for p in paths]
    names = [p.rsplit("/", 1)[-1][:40] for p in paths]
    loads = ["decode 1", "verify 1", "decode 16", "decode 32", "verify 16", "verify 32"]
    print("case".ljust(22) + "".join(n.rjust(42) for n in names))
    for load in loads:
        for layer in ("sliding", "full"):
            k = (load, layer)
            print(f"{load} {layer}".ljust(22) + "".join(f"{c[k]:42.2f}" for _, c in data))
    print("load totals (46 s + 16 f), us, and delta of the first against each other")
    for load in loads:
        tot = [46 * c[(load, "sliding")] + 16 * c[(load, "full")] for _, c in data]
        print(f"{load}".ljust(22) + "".join(f"{t:42.1f}" for t in tot) + "   delta vs others: "
              + ", ".join(f"{tot[0] - t:+.1f} ({(tot[0] - t) / t * 100:+.2f}%)" for t in tot[1:]))
    for i, (d, _) in enumerate(data):
        s = d.get("summary", {})
        print(names[i], "speedup", s.get("speedup"), "score_us", s.get("score_us"), "tests", s.get("tests_passed"), "/", s.get("tests"))
    d0 = data[0][0]
    for p in d0.get("profiles", []):
        c = p["case"]
        if c["load"] in ("decode 1", "verify 1"):
            print(f"--- profile {p['index']}: {c['load']} {c['layer']}")
            print(p["report"])


if __name__ == "__main__":
    main()
