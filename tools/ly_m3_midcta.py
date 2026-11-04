#!/usr/bin/env python3
"""MID raw stamps (tools/ly_box_moe_tl16g.py --raw): per CTA A end, first phase B unit past doneA, B end, unit counts,
the latest CTAs' phase B retire gaps, and correlations of B end with counts and phase ends.

    python3 tools/ly_m3_midcta.py muse_decode_layer/profiles/box/moe3/tl16_stk1.json.gz
"""
import json, statistics as st, sys
import gzip


def load(path):
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))

d = load(sys.argv[1])
def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5; sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")
for row in d["rows"]:
    lab = f"{row['case']['load']} {row['case']['layer']}"
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        v = inp["raw"]; t0 = min(r[0] for r in v if r[0]); us = lambda t: (t - t0) / 1e3 if t else float("nan")
        R = []
        for c, r in enumerate(v):
            nA, nB = r[17] & 0xFFFF, (r[17] >> 16) & 0xFFFF
            ret = [us(r[18 + q]) for q in range(46) if r[18 + q]]
            R.append(dict(c=c, nA=nA, nB=nB, first=us(r[2]), aend=us(r[3]), dep1=us(r[5]), bend=us(r[6]), barr=us(r[9]),
                          brel=us(r[10]), fin=us(r[11]), ret=ret, aw=r[12] / 1e3, bw=r[13] / 1e3))
        n = len(R)
        aend = [x["aend"] for x in R]; bend = [x["bend"] for x in R]; dep1 = [x["dep1"] for x in R]
        tot = [x["nA"] + x["nB"] for x in R]
        hand = [x["dep1"] - x["aend"] for x in R]
        bdur = [x["bend"] - x["dep1"] for x in R]
        print(f"{lab} seed {inp['seed']} distinct {inp.get('distinct')} units/CTA {min(tot)}-{max(tot)} nA {min(x['nA'] for x in R)}-{max(x['nA'] for x in R)}"
              f" nB {min(x['nB'] for x in R)}-{max(x['nB'] for x in R)}")
        print(f"  first slot {st.mean(x['first'] for x in R):.2f} (max {max(x['first'] for x in R):.2f})  A end {min(aend):.2f}-{max(aend):.2f}"
              f" (mean {st.mean(aend):.2f})  dep1 {min(dep1):.2f}-{max(dep1):.2f}  B end {min(bend):.2f}-{max(bend):.2f} (mean {st.mean(bend):.2f})"
              f"  barrier release {st.mean(x['brel'] for x in R):.2f}  exit max {max(x['fin'] for x in R):.2f}")
        print(f"  handoff mean {st.mean(hand):.2f} (min {min(hand):.2f} max {max(hand):.2f})  B dur mean {st.mean(bdur):.2f}"
              f"  corr(B end, nA+nB) {corr(bend, tot):.2f}  corr(B end, nB) {corr(bend, [x['nB'] for x in R]):.2f}"
              f"  corr(B end, A end) {corr(bend, aend):.2f}  corr(B end, dep1) {corr(bend, dep1):.2f}  corr(A end, nA) {corr(aend, [x['nA'] for x in R]):.2f}")
        late = sorted(R, key=lambda x: -x["bend"])[:4]
        for x in late:
            rb = x["ret"][x["nA"]:]
            gaps = [round(b - a, 2) for a, b in zip([x["dep1"]] + rb[:-1], rb)]
            print(f"   late c {x['c']} nA {x['nA']} nB {x['nB']} A end {x['aend']:.2f} dep1 {x['dep1']:.2f} B end {x['bend']:.2f} wwait A {x['aw']:.2f} B {x['bw']:.2f} B gaps {gaps}")
        early = sorted(R, key=lambda x: x["bend"])[:2]
        for x in early:
            print(f"   early c {x['c']} nA {x['nA']} nB {x['nB']} A end {x['aend']:.2f} dep1 {x['dep1']:.2f} B end {x['bend']:.2f}")
