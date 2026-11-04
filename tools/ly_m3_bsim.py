#!/usr/bin/env python3
"""Phase B pipeline model on B4 raw stamps: forward K-stage order vs per-unit order sorted by the phase A dependency's
completion proxy. stage i of a CTA's B walk starts at max(previous end, ready(i)) and takes TS us; ready(i) = latest of
the two A units' completion proxies + LAG (the deferred flush, release and the x copy).

    python3 tools/ly_m3_bsim.py muse_decode_layer/profiles/box/moe3/tlb4_stk1.json.gz [TS LAG] [--noreord]   (--noreord: RFIN's in-order walk)
"""
import json, statistics as st, sys
import gzip


def load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))

d = load(sys.argv[1])
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REORD = "--noreord" not in sys.argv
TS, LAG = float(args[1]) if len(args) > 1 else 1.0, float(args[2]) if len(args) > 2 else 1.5
for row in d["rows"]:
    lab = f"{row['case']['load']} {row['case']['layer']}"
    out = []
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        v = inp["raw"]; t0 = min(r[0] for r in v); us = lambda t: (t - t0) / 1e3
        C = []
        for r in v:
            g = [r[21] >> (16 * k) & 0xFFFF for k in range(4)]
            g0, g1 = g[0], g[1]
            ue = ((g1 - 1) >> 4) << 4
            sp = ue if (REORD and (g1 & 15) != 0 and ue > g0) else g0
            seq = [sp + i if sp + i < g1 else g0 + (sp + i - g1) for i in range(g1 - g0)]
            C.append(dict(g=g, seq=seq, iss=[r[24 + k] for k in range(40)], aend=us(r[2]), bend=us(r[3]), w1=us(r[12]) if r[12] else None))
        owner = {}
        for ci, c in enumerate(C):
            for s in range(c["g"][0], c["g"][1]): owner[s] = ci
        def done(stage):
            c = C[owner[stage]]; p = c["seq"].index(stage); k = p + 3
            return us(c["iss"][k]) if k < 40 and c["iss"][k] else c["aend"]
        ucache = {}
        def udone(ua):
            if ua not in ucache: ucache[ua] = max(done(ua * 16 + s) for s in range(16))
            return ucache[ua]
        def ready(j, k): return max(udone(j * 32 + 2 * k), udone(j * 32 + 2 * k + 1)) + LAG
        fw, so, act = [], [], []
        for c in C:
            h0, h1 = c["g"][2], c["g"][3]
            if h1 <= h0: continue
            segs = []
            gs = h0
            while gs < h1:
                u = gs >> 4; s0 = gs & 15; s1 = min(16, s0 + h1 - gs)
                segs.append((u, s0, s1)); gs += s1 - s0
            def run(order):
                t = c["aend"]
                for u, ks in order:
                    for k in ks:
                        t = max(t, ready(u >> 4, k)) + TS
                return t
            fwd = [(u, list(range(s0, s1))) for u, s0, s1 in segs]
            # sorted: the unit's 16 K stages by readiness; the share at positions s0..s1 takes those positions
            srt = []
            for u, s0, s1 in segs:
                perm = sorted(range(16), key=lambda k: ready(u >> 4, k))
                srt.append((u, perm[s0:s1]))
            fw.append(run(fwd)); so.append(run(srt)); act.append(c["bend"])
        out.append((max(act), max(fw), max(so), st.mean(act), st.mean(fw), st.mean(so)))
        print(f"{lab} seed {inp['seed']}: latest B end actual {max(act):.2f} model fwd {max(fw):.2f} sorted {max(so):.2f}"
              f" | mean actual {st.mean(act):.2f} fwd {st.mean(fw):.2f} sorted {st.mean(so):.2f}")
    print(f"{lab}: latest B end model fwd - sorted mean {st.mean(o[1] - o[2] for o in out):.2f}")
