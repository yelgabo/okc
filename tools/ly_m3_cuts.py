#!/usr/bin/env python3
"""B4 raw stamps: phase A end and phase B end per CTA grouped by the number of cut (stream-K) shares in its ranges.

    python3 tools/ly_m3_cuts.py muse_decode_layer/profiles/box/moe3/tlb4_stk1.json.gz
"""
import json, statistics as st, sys
import gzip


def load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))

d = load(sys.argv[1])
for row in d["rows"]:
    lab = f"{row['case']['load']} {row['case']['layer']}"
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        v = inp["raw"]; t0 = min(r[0] for r in v); us = lambda t: (t - t0) / 1e3
        by = {}
        byb = {}
        for r in v:
            g = [r[21] >> (16 * k) & 0xFFFF for k in range(4)]
            ca = int((g[0] & 15) != 0) + int((g[1] & 15) != 0)
            cb = int((g[2] & 15) != 0) + int((g[3] & 15) != 0)
            by.setdefault(ca, []).append(us(r[2]))
            byb.setdefault((ca, cb), []).append(us(r[3]))
        print(lab, inp["seed"], "A end by phase A cut shares:", {k: (len(x), round(st.mean(x), 2), round(max(x), 2)) for k, x in sorted(by.items())},
              " B end by (A, B) cuts:", {k: (len(x), round(st.mean(x), 2), round(max(x), 2)) for k, x in sorted(byb.items())})
