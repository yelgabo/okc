#!/usr/bin/env python3
"""B4 raw stamps (tools/ly_box_moe_tlb4g.py --raw): each CTA's first phase B doneAs wait against the completion proxy
of the two phase A units its first stage reads, and the stall if the unit's stages ran in readiness order.

    python3 tools/ly_m3_bdep.py muse_decode_layer/profiles/box/moe3/tlb4_stk1.json.gz [--noreord]
"""
import json, statistics as st, sys
import gzip


def load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))

d = load(sys.argv[1])
REORD = "--noreord" not in sys.argv
for row in d["rows"]:
    lab = f"{row['case']['load']} {row['case']['layer']}"
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        v = inp["raw"]; t0 = min(r[0] for r in v); us = lambda t: (t - t0) / 1e3
        C = []
        for c, r in enumerate(v):
            g = [r[21] >> (16 * k) & 0xFFFF for k in range(4)]
            g0, g1 = g[0], g[1]
            ue = ((g1 - 1) >> 4) << 4
            sp = ue if (REORD and (g1 & 15) != 0 and ue > g0) else g0
            seq = [sp + i if sp + i < g1 else g0 + (sp + i - g1) for i in range(g1 - g0)]
            iss = [r[24 + k] for k in range(40)]
            C.append(dict(g=g, seq=seq, iss=iss, aend=us(r[2]), bend=us(r[3]), w0=us(r[11]) if r[11] else None, w1=us(r[12]) if r[12] else None, i0=us(r[8])))
        def done_time(stage):
            for c in C:
                if c["g"][0] <= stage < c["g"][1]:
                    p = c["seq"].index(stage)
                    k = p + 3
                    if k < 40 and c["iss"][k]: return us(c["iss"][k])
                    return c["aend"]
            return None
        def unit_done(ua):
            return max(done_time(ua * 16 + s) for s in range(16))
        rows = []
        for c in C:
            h0, h1 = c["g"][2], c["g"][3]
            if h1 <= h0 or c["w1"] is None: continue
            uB = h0 >> 4; j = uB >> 4; s = h0 & 15
            dep = max(unit_done(j * 32 + 2 * s), unit_done(j * 32 + 2 * s + 1))
            hs = range(h0, min(h1, (uB + 1) * 16))
            alt = min(max(unit_done(j * 32 + 2 * (x & 15)), unit_done(j * 32 + 2 * (x & 15) + 1)) for x in hs)
            rows.append((c["aend"], c["w0"], c["w1"], dep, alt, c["bend"]))
        wait = [r[2] - r[1] for r in rows]
        lag = [r[2] - r[3] for r in rows]
        ml = st.median(lag)
        gain = [max(0, r[2] - max(r[1], r[4] + ml)) for r in rows]
        print(f"{lab} seed {inp['seed']} n {len(rows)} first wait mean {st.mean(wait):.2f}  wait end - dep done median {ml:.2f}"
              f" (p10 {sorted(lag)[len(lag)//10]:.2f} p90 {sorted(lag)[9*len(lag)//10]:.2f})"
              f"  dep done - A end mean {st.mean(r[3]-r[0] for r in rows):.2f}  best dep - A end mean {st.mean(r[4]-r[0] for r in rows):.2f}"
              f"  modelled wait cut mean {st.mean(gain):.2f}  latest B end {max(r[5] for r in rows):.2f}")
        st_now = [max(0.0, r[2] - r[0]) for r in rows]
        st_alt = [max(0.0, max(r[1], r[4] + ml) - r[0]) for r in rows]
        late = sorted(rows, key=lambda r: -r[5])[:10]
        print(f"   stall at B start (wait end - A end, >= 0): mean {st.mean(st_now):.2f} median {st.median(st_now):.2f} max {max(st_now):.2f};"
              f" best-stage model mean {st.mean(st_alt):.2f};  10 latest-B-end CTAs: now {st.mean(max(0, r[2]-r[0]) for r in late):.2f}"
              f" model {st.mean(max(0.0, max(r[1], r[4] + ml) - r[0]) for r in late):.2f}")
