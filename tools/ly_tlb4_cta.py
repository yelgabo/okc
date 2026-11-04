#!/usr/bin/env python3
"""Per-CTA reading of tools/ly_box_moe_tlb4g.py raw stamps (--raw N inputs per cell).

    python3 tools/ly_tlb4_cta.py muse_decode_layer/profiles/box/lyb4/tlb4g.json

Per CTA (times from the first CTA's entry, us): stage counts A / B, first weight issue, A end, B end, MFIN wait end,
exit; the weights producer's issue gaps over the first 40 stages split into the ring fill (stages 0-2), steady gaps
and long gaps (> 1.6 x the median), and the gap at the A-to-B switch. Then the correlation of A end and B end with the
CTA's stage counts, which separates static imbalance (more stages, later end) from per-CTA delays.
"""
import json
import statistics
import sys


def corr(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def main():
    d = json.load(open(sys.argv[1]))
    for row in d["rows"]:
        lab = f"{row['case']['load']} {row['case']['layer']}"
        for inp in row["inputs"]:
            if "raw" not in inp:
                continue
            v = inp["raw"]
            t0 = min(r[0] for r in v)
            us = lambda t: (t - t0) / 1e3  # noqa: E731
            rec = []
            for r in v:
                g = [r[21] >> (16 * k) & 0xFFFF for k in range(4)]
                nA, nB = g[1] - g[0], g[3] - g[2]
                iss = [r[24 + k] for k in range(40) if r[24 + k]]
                gaps = [(iss[k] - iss[k - 1]) / 1e3 for k in range(1, len(iss))]
                rec.append(dict(nA=nA, nB=nB, i0=us(r[8]), aend=us(r[2]), bend=us(r[3]), fin=us(r[4]) if r[4] else None,
                                ex=us(r[5]), gaps=gaps, g=g, dep=r[13] / 1e3, depf=(r[12] - r[11]) / 1e3))
            steady = [x for c in rec for k, x in enumerate(c["gaps"], 1) if k >= 3 and k not in (c["nA"], c["nA"] + 1, c["nA"] + 2)]
            med = statistics.median(steady)
            longs = [x for x in steady if x > 1.6 * med]
            print(f"{lab} seed {inp['seed']} distinct {inp['distinct']} span {inp['span']:.2f} median gap {med:.3f} "
                  f"mean gap {statistics.mean(steady):.3f} long gaps {len(longs)} of {len(steady)} sum {sum(longs):.2f} "
                  f"(per CTA {sum(longs) / len(rec):.2f}; excess over median {sum(x - med for x in longs) / len(rec):.2f})")
            nA = [c["nA"] for c in rec]
            nB = [c["nB"] for c in rec]
            ae = [c["aend"] for c in rec]
            be = [c["bend"] for c in rec]
            print(f"  A stages {min(nA)}-{max(nA)}  B stages {min(nB)}-{max(nB)}  A end {min(ae):.2f}-{max(ae):.2f}"
                  f"  B end {min(be):.2f}-{max(be):.2f}  corr(A end, nA) {corr(ae, nA):.2f}  corr(B end, nA + nB)"
                  f" {corr(be, [a + b for a, b in zip(nA, nB)]):.2f}  corr(B end, A end) {corr(be, ae):.2f}")
            per = [(c["aend"] - c["i0"]) / c["nA"] for c in rec if c["nA"]]
            perb = [(c["bend"] - c["aend"]) / c["nB"] for c in rec if c["nB"]]
            print(f"  per-stage time over the phase (end - start) / stages: A median {statistics.median(per):.3f}"
                  f" (min {min(per):.3f} max {max(per):.3f}), B median {statistics.median(perb):.3f}"
                  f" (min {min(perb):.3f} max {max(perb):.3f}); x producer phase B doneAs wait mean"
                  f" {statistics.mean(c['dep'] for c in rec):.2f} (first wait {statistics.mean(c['depf'] for c in rec):.2f})")
            late = sorted(rec, key=lambda c: -c["bend"])[:5]
            for c in late:
                print(f"  late CTA: nA {c['nA']} nB {c['nB']} A {c['g'][0]}-{c['g'][1]} B {c['g'][2]}-{c['g'][3]} i0 {c['i0']:.2f}"
                      f" A end {c['aend']:.2f} B end {c['bend']:.2f} dep {c['dep']:.2f} max gap {max(c['gaps']):.2f}")
            # issue k waits for the consumers' release of stage k - 3 and the landing of stage k - 2; classify each gap
            # by the position of stage k - 3 inside its segment (a segment: one CTA's contiguous stages of one unit;
            # position 0 is the stage in which the consumers run the previous segment's deferred wgs_flush)
            for ph in ("A", "B"):
                by = {}
                for c in rec:
                    g0, g1, h0, h1 = c["g"]
                    ue = ((g1 - 1) >> 4) << 4
                    sp = ue if ((g1 & 15) != 0 and ue > g0) else g0
                    seq = [sp + i if sp + i < g1 else g0 + (sp + i - g1) for i in range(g1 - g0)] + list(range(h0, h1))
                    pos, cur = [], 0
                    for i, s in enumerate(seq):
                        cur = 0 if i == 0 or i == g1 - g0 or (s >> 4) != (seq[i - 1] >> 4) or s != seq[i - 1] + 1 else cur + 1
                        pos.append(cur)
                    for k, x in enumerate(c["gaps"], 1):
                        if k < 3 or k - 3 >= len(pos) or (ph == "A") != (k - 3 < c["nA"]) or k - 3 in (c["nA"], c["nA"] - 1, c["nA"] - 2):
                            continue
                        by.setdefault(min(pos[k - 3], 3), []).append(x)
                print(f"  phase {ph} issue gaps by the released stage's position in its segment (0, 1, 2, 3+): " + "; ".join(
                    f"{q}: n {len(x)} median {statistics.median(x):.2f} mean {statistics.mean(x):.2f} excess over"
                    f" {med:.2f} per CTA {sum(y - med for y in x) / len(rec):.2f}" for q, x in sorted(by.items())))


if __name__ == "__main__":
    main()
