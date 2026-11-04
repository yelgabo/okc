#!/usr/bin/env python3
"""B4 phase A per-CTA duration against the split's cost model: stages by n-tile class, segments, cut shares (numpy).

    python tools/ly_m3_cost.py muse_decode_layer/profiles/box/moe3/tlb4_stk1.json.gz
"""
import json, statistics as st, sys
import numpy as np
import gzip


def load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))

d = load(sys.argv[1])
CA = [112, 135, 140, 158, 173, 185, 200, 207]; UA = 160
CB = [100, 104, 115, 123, 132, 144, 156, 174]; UB = 140
X, Y, P, XB, YB, PB = [], [], [], [], [], []
for row in d["rows"]:
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        lids = [x for x in inp["lids"]] if not isinstance(inp["lids"][0], list) else [y for x in inp["lids"] for y in x]
        cnt = [0] * 32
        for e in lids:
            if e >= 0: cnt[e] += 1
        jobs = [c for c in cnt if c > 8] + [c for c in cnt if 0 < c <= 8]
        v = inp["raw"]; t0 = min(r[0] for r in v); us = lambda t: (t - t0) / 1e3
        for r in v:
            g = [r[21] >> (16 * k) & 0xFFFF for k in range(4)]
            for ph, (s0, s1, tstart, tend) in enumerate([(g[0], g[1], us(r[8]), us(r[2])), (g[2], g[3], us(r[2]), us(r[3]))]):
                if s1 <= s0: continue
                sh = 9 - ph
                byn = [0] * 4
                segs = 0; cuts = 0; prevu = None
                cost = 0
                for s in range(s0, s1):
                    j = s >> sh; ntt = (jobs[j] + 7) >> 3
                    byn[min(ntt, 4) - 1] += 1
                    cost += (CB if ph else CA)[ntt - 1]
                    u = s >> 4
                    if u != prevu:
                        segs += 1; prevu = u
                        cost += (UB if ph else UA) * 0  # the split charges the unit cost at stage 0 only
                        if (s & 15) == 0: cost += (UB if ph else UA)
                ufirst = (s0 & 15) != 0; ulast = (s1 & 15) != 0
                cuts = int(ufirst) + int(ulast)
                feat = byn + [segs, cuts]
                if ph == 0: X.append(feat); Y.append(tend - tstart); P.append(cost)
                else: XB.append(feat); YB.append(tend - tstart); PB.append(cost)
for name, x, y, p in (("A", X, Y, P), ("B", XB, YB, PB)):
    x = np.array(x, float); y = np.array(y); p = np.array(p, float)
    A = np.c_[x, np.ones(len(y))]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    k = np.polyfit(p, y, 1)
    r2 = y - np.polyval(k, p)
    print(f"phase {name}: n {len(y)}  us per stage NTT1..4 {np.round(coef[:4], 3)}  per segment {coef[4]:.3f}  per cut share {coef[5]:.3f}"
          f"  const {coef[6]:.2f}  resid sd {res.std():.2f}  | split-cost fit resid sd {r2.std():.2f}  dur sd {y.std():.2f}"
          f"  stage counts by NTT (sum) {x[:, :4].sum(0)}")
