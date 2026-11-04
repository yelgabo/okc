#!/usr/bin/env python3
"""Ordering check of the ly_rfin2 walk (tools/dl_ly_rfin2_patch.py) on a Python replica of the MAIN build's stream-K
range functions (wgs_prefix / wgs_stage_at / wgs_range / wgs_cta_of, 32-bit as in the kernel, cepi unit costs).

For random job lists (jobs of 1 to 64 tokens, mostly 8) and G = 132 CTAs it checks, per phase: every CTA walks exactly
the stages of its range, once; a range holds at most one depositing share (a cut share not holding stage 0) and it is
the first segment; and for every cut unit the slack, in stages, between the moment the finisher (the share holding
stage 0) reaches its cut share and the moment the last depositor's count release is issued (its share plus one stage
for the deferred flush at the next segment's first stage, or its range end when the share is its last segment),
assuming every CTA starts the phase at the same time and runs its stages at the same rate. A negative slack is a wait
the finisher would take even at equal phase starts; a slack below the phase start spread (phase B: the phase A end
spread, 10 to 16 us) is a wait it may take.

    python3 tools/ly_rfin2_order_check.py [--lists 800] [--mask 1|3] [--seed 1]
"""
import random
import sys

CA = [112, 135, 140, 158, 173, 185, 200, 207]
CB = [100, 104, 115, 123, 132, 144, 156, 174]
U = (160, 140)
M32 = 0xFFFFFFFF


def cost(pb, cnt):
    return (CB if pb else CA)[((cnt + 7) >> 3) - 1]


class Ranges:
    def __init__(self, jC, G):
        self.jC, self.G, self.nj = jC, G, len(jC)
        self.pre = [[0] * (self.nj + 1), [0] * (self.nj + 1)]
        acc = [0, 0]
        for j, c in enumerate(jC):
            for pb in (0, 1):
                self.pre[pb][j] = acc[pb]
                acc[pb] = (acc[pb] + ((cost(pb, c) * 16 + U[pb]) << (5 - pb))) & M32
        for pb in (0, 1):
            self.pre[pb][self.nj] = acc[pb]
        self.tot = acc
        assert self.tot[0] * (G - 1) <= M32, "tot * c overflows 32 bits"

    def stage_at(self, pb, x):
        pre = self.pre[pb]
        lo, hi = 0, self.nj
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if pre[mid] <= x:
                lo = mid
            else:
                hi = mid
        cs, u = cost(pb, self.jC[lo]), U[pb]
        cu = 16 * cs + u
        r = (x - pre[lo]) & M32
        un = r // cu
        rem = r - un * cu
        st = 0 if rem < u else min(15, (rem - u) // cs)
        return (lo << (9 - pb)) + un * 16 + st

    def range(self, pb, c):
        tot, G = self.tot[pb], self.G
        g0 = self.stage_at(pb, ((tot * c) & M32) // G)
        g1 = (self.nj << (9 - pb)) if c == G - 1 else self.stage_at(pb, ((tot * (c + 1)) & M32) // G)
        return g0, g1

    def cta_of(self, pb, s):
        tot, G = self.tot[pb], self.G
        j = s >> (9 - pb)
        sj = s & ((1 << (9 - pb)) - 1)
        cs, u = cost(pb, self.jC[j]), U[pb]
        xe = (self.pre[pb][j] + (sj >> 4) * (16 * cs + u) + u + ((sj & 15) + 1) * cs) & M32
        c = ((xe - 1) * G) // tot
        c = min(c, G - 1)
        while c + 1 < G and ((tot * (c + 1)) & M32) // G < xe:
            c += 1
        while c > 0 and ((tot * c) & M32) // G >= xe:
            c -= 1
        return c


def walk(pb, g0, g1, mask):
    """sub-ranges in processing order"""
    ue = ((g1 - 1) >> 4) << 4
    uw = ue - 16
    if ((mask >> pb) & 1) and (g1 & 15) != 0 and uw >= g0:
        return [(g0, uw), (ue, g1), (uw, ue)]
    return [(g0, g1)]


def segments(subs):
    out = []
    for a, b in subs:
        gs = a
        while gs < b:
            u, s0 = gs >> 4, gs & 15
            s1 = min(16, s0 + (b - gs))
            out.append((u, s0, s1))
            gs += s1 - s0
    return out


def main():
    lists = int(sys.argv[sys.argv.index("--lists") + 1]) if "--lists" in sys.argv else 800
    mask = int(sys.argv[sys.argv.index("--mask") + 1]) if "--mask" in sys.argv else 1
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    rng = random.Random(seed)
    G = 132
    stats = {pb: {"cut": 0, "reordered_ranges": 0, "cut_ranges": 0, "slack": [], "slack_base": [], "neg": 0,
                  "three_share": 0, "finisher_wait_ranges": 0} for pb in (0, 1)}
    for _ in range(lists):
        nj = rng.randint(14, 26)
        jC = []
        for _ in range(nj):
            r = rng.random()
            jC.append(8 if r < 0.55 else rng.randint(1, 7) if r < 0.7 else rng.choice([16, 24, 32, 40, 48, 56, 64]))
        R = Ranges(jC, G)
        for pb in (0, 1):
            st = stats[pb]
            rngs = [R.range(pb, c) for c in range(G)]
            assert rngs[0][0] == 0 and rngs[-1][1] == nj << (9 - pb)
            for c in range(G - 1):
                assert rngs[c][1] == rngs[c + 1][0], "ranges do not tile"
            segs, pos = {}, {}
            for c, (g0, g1) in enumerate(rngs):
                subs = walk(pb, g0, g1, mask)
                if len(subs) == 3:
                    st["reordered_ranges"] += 1
                if (g1 & 15) != 0:
                    st["cut_ranges"] += 1
                seq = [s for a, b in subs for s in range(a, b)]
                assert sorted(seq) == list(range(g0, g1)), f"walk of CTA {c} is not its range"
                sg = segments(subs)
                dep = [s for s in sg if s[1] != 0]
                assert len(dep) <= 1, "two depositing shares in one range"
                if dep:
                    assert sg[0] == dep[0], "the depositing share is not the first segment"
                p = 0
                for i, (u, s0, s1) in enumerate(sg):
                    segs.setdefault(u, []).append((c, s0, s1, p, i == len(sg) - 1, g1 - g0))
                    p += s1 - s0
                pos[c] = p
            for u, shares in segs.items():
                if len(shares) == 1:
                    continue
                st["cut"] += 1
                if len(shares) > 2:
                    st["three_share"] += 1
                fin = [s for s in shares if s[1] == 0]
                assert len(fin) == 1, "no single finisher"
                fc, _, fs1, fp, _, _ = fin[0]
                assert R.cta_of(pb, u << 4) == fc, "wgs_cta_of(stage 0) is not the finisher"
                assert R.cta_of(pb, (u << 4) | 15) == max(s[0] for s in shares), "wgs_cta_of(stage 15) is not the last depositor"
                for s in shares:
                    if s[1] != 0:
                        assert s[0] > fc, "a depositor before the finisher's CTA"
                last_dep = max(s[5] if s[4] else s[3] + (s[2] - s[1]) + 1 for s in shares if s[1] != 0)
                slack = fp - last_dep
                st["slack"].append(slack)
                st["slack_base"].append((pos[fc] - fs1) - last_dep)
                if slack < 0:
                    st["neg"] += 1
    for pb in (0, 1):
        st = stats[pb]
        sl = sorted(st["slack"])
        sb = sorted(st["slack_base"])
        q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
        print(f"phase {'B' if pb else 'A'} (mask bit {'on' if (mask >> pb) & 1 else 'off'}): {lists} lists, {st['cut']} cut units "
              f"({st['three_share']} with 3+ shares), {st['cut_ranges']} ranges end inside a unit, {st['reordered_ranges']} reordered; "
              f"finisher slack in stages: min {sl[0]}, p1 {q(sl, 0.01)}, p10 {q(sl, 0.1)}, median {q(sl, 0.5)}, max {sl[-1]}; "
              f"negative {st['neg']}; in-order (ly_rfin) slack min {sb[0]}, median {q(sb, 0.5)}")


if __name__ == "__main__":
    main()
