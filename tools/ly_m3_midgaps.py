#!/usr/bin/env python3
"""MID unit retire gaps by kind: first unit of a phase, first unit after a job switch, steady.

    python3 tools/ly_m3_midgaps.py muse_decode_layer/profiles/box/moe3/tl16_stk1.json.gz
"""
import json, statistics as st, sys
import gzip


def load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))

d = load(sys.argv[1])
for row in d["rows"]:
    lab = f"{row['case']['load']} {row['case']['layer']}"
    agg = {}
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        lids = inp["lids"]
        flat = [y for x in lids for y in (x if isinstance(x, list) else [x])]
        cnt = [0] * 32
        for e in flat:
            if e >= 0: cnt[e] += 1
        nj = sum((c + 7) // 8 for c in cnt if c > 0)
        v = inp["raw"]; G = len(v)
        nA, nB = nj * 512, nj * 256
        for c, r in enumerate(v):
            a0, a1 = nA * c // G, nA * (c + 1) // G
            b0, b1 = nB * c // G, nB * (c + 1) // G
            units = [("A", u, u >> 9) for u in range(a0, a1)] + [("B", u, u >> 8) for u in range(b0, b1)]
            ret = [r[18 + q] for q in range(46) if r[18 + q]]
            prev = r[2] if r[2] else None   # first slot full
            for q, t in enumerate(ret):
                if q >= len(units): break
                ph, u, j = units[q]
                if q == 0: kind = "A first (from first slot)"
                elif q == a1 - a0: kind = "B first (from A end)"; prev = r[3]
                elif units[q - 1][2] != j: kind = f"{ph} job switch"
                else: kind = f"{ph} steady"
                if prev: agg.setdefault(kind, []).append((t - prev) / 1e3)
                prev = t
    print(lab)
    for k, x in sorted(agg.items()):
        print(f"   {k:28s} n {len(x):5d} median {st.median(x):.2f} mean {st.mean(x):.2f} p90 {sorted(x)[9 * len(x) // 10]:.2f}")
