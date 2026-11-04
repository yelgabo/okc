#!/usr/bin/env python3
"""Segments per CTA by kind from a tools/ly_box_moe_tlv16g.py JSON with raw stamps (--raw N), and the switch cost
after each kind: the first and second issue-gap excess over the steady gap of the stage's n-tile class, attributed to
the previous segment's kind. Kinds: whole unit; k2 = cut share that does not hold the designated finisher's stage
(phase A: stage 15, phase B: stage 0, the base's OKC_WGSDFIN labels); k3 = the share that does. With --no-reorder
(ly_rfin bases) phase A runs in order and the finisher is the share holding stage 0, so the phase A labels read
k2 = finisher (the CTA's last segment, its cost at the phase end) and k3 = depositor (its first).

    python3 tools/ly_seg_kind.py tlv16g.json [--no-reorder]
"""
import json
import os
import statistics
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ly_box_moe_tlv16g as T
if "--no-reorder" in sys.argv:
    T.REORDER = False
issue_order = T.issue_order
d = json.load(open(sys.argv[1]))
for row in d["rows"]:
    lab = f"{row['case']['load']} {row['case']['layer']}"
    cnt = {}      # (ph, kind) -> count over CTAs and inputs
    ex = {}       # (ph, prevkind, prevntt) -> list of excess
    exk = {}      # (ph, prevkind) -> list
    ncta = 0
    segs_cta = {0: [], 1: []}
    cuts_cta = {0: [], 1: []}
    for inp in row["inputs"]:
        if "raw" not in inp:
            continue
        v = inp["raw"]; jobs = inp["jobs"]; ntt_of = [(c + 7) >> 3 for c in jobs["jC"]]
        steady = {}
        for ph, key in ((0, "gapsA"), (1, "gapsB")):
            for k, g in inp[key].items():
                n, p = k.split(":")
                if p == "2":
                    steady[(ph, int(n))] = statistics.median(g)
        for r in v:
            ncta += 1
            g = [r[21] >> (16 * k) & 0xFFFF for k in range(4)]
            seqA, seqB = issue_order(0, g[0], g[1]), issue_order(1, g[2], g[3])
            seq = seqA + seqB
            stamps = [r[24 + k] for k in range(min(296, len(seq)))]
            segl = {0: [], 1: []}
            for ph, sq, off in ((0, seqA, 0), (1, seqB, len(seqA))):
                i = 0
                while i < len(sq):
                    u = sq[i] >> 4; j = i
                    while j + 1 < len(sq) and (sq[j + 1] >> 4) == u and sq[j + 1] == sq[j] + 1:
                        j += 1
                    s0, s1 = sq[i] & 15, (sq[j] & 15) + 1
                    whole = (s1 - s0 == 16)
                    kind = "whole" if whole else ("k3" if ((ph == 1 and s0 == 0) or (ph == 0 and s1 == 16)) else "k2")
                    ntt = ntt_of[u >> (5 - ph)]
                    segl[ph].append((off + i, off + j, kind, ntt))
                    i = j + 1
                segs_cta[ph].append(len(segl[ph]))
                cuts_cta[ph].append(sum(1 for s in segl[ph] if s[2] != "whole"))
                for s in segl[ph]:
                    cnt[(ph, s[2])] = cnt.get((ph, s[2]), 0) + 1
                for si in range(1, len(segl[ph])):
                    first = segl[ph][si][0]
                    prev = segl[ph][si - 1]
                    e = 0.0; ok = True
                    for kk in (first + 3, first + 4):
                        if kk >= len(stamps) or stamps[kk] == 0 or stamps[kk - 1] == 0:
                            ok = False; break
                        if kk - 3 > segl[ph][si][1]:
                            continue
                        gap = (stamps[kk] - stamps[kk - 1]) / 1e3
                        n = segl[ph][si][3]
                        e += gap - steady.get((ph, n), gap)
                    if not ok:
                        continue
                    ex.setdefault((ph, prev[2], prev[3]), []).append(e)
                    exk.setdefault((ph, prev[2]), []).append(e)
    print(f"== {lab}: CTAs x inputs {ncta}")
    for ph in (0, 1):
        print(f"  phase {'AB'[ph]}: segments per CTA median {statistics.median(segs_cta[ph])} mean {statistics.mean(segs_cta[ph]):.2f} "
              f"(whole {cnt.get((ph,'whole'),0)/ncta:.2f} k2 {cnt.get((ph,'k2'),0)/ncta:.2f} k3 {cnt.get((ph,'k3'),0)/ncta:.2f} per CTA); cuts per CTA mean {statistics.mean(cuts_cta[ph]):.2f}")
        for k in ("whole", "k2", "k3"):
            xs = exk.get((ph, k))
            if xs:
                print(f"    after {k:5s}: n {len(xs):4d} excess mean {statistics.mean(xs):.2f} med {statistics.median(xs):.2f} us")
        for n in range(1, 9):
            parts = []
            for k in ("whole", "k2", "k3"):
                xs = ex.get((ph, k, n))
                if xs:
                    parts.append(f"{k} {statistics.mean(xs):.2f} (n {len(xs)})")
            if parts:
                print(f"    prev NTT {n}: " + "; ".join(parts))
