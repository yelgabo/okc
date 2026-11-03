"""Work decomposition of the W4 GEMMs under the tuner's synthetic routing (_make_inputs in
muse_moe_w4a16/submission.py, ported to pure Python): local routes, unique experts, hottest
expert, items per GEMM at 8/16 rows per item, L2->SM bytes, waves and tail."""
import math, random, statistics, sys

N_EXP, TOPK, LOCAL = 256, 8, 32
W13_MB, W2_MB = 18.9, 9.4

def routing(bs, seed, dist="normal"):
    rng = random.Random(seed)
    if dist == "normal":
        bias = []
        for _ in range(N_EXP):
            z = rng.gauss(0, 1)
            u = max(rng.random(), 1e-12)
            bias.append(0.03 * z / math.sqrt(-math.log(u)))
    else:
        bias = [rng.random() ** 6 * 4 for _ in range(N_EXP)]
    ids = []
    for _ in range(bs):
        sc = [rng.random() + bias[e] for e in range(N_EXP)]
        top = sorted(range(N_EXP), key=lambda e: -sc[e])[:TOPK]
        ids.append(top)
    groups = [0] * (N_EXP // LOCAL)
    for row in ids:
        for e in row:
            groups[e // LOCAL] += 1
    rot = LOCAL * max(range(len(groups)), key=lambda g: groups[g])
    counts = [0] * LOCAL
    for row in ids:
        for e in row:
            le = (e - rot) % N_EXP
            if le < LOCAL:
                counts[le] += 1
    return counts

def summarize(bs, cfg1, cfg2, seeds, dist="normal"):
    # cfg = (nt, rows, S, W): rows per item = 8*nt; blocks per item per GEMM = (U/W)*S
    nt1, rows1, s1, w1 = cfg1
    nt2, rows2, s2, w2 = cfg2
    out = []
    for seed in seeds:
        c = routing(bs, seed, dist)
        local = sum(c)
        uniq = sum(1 for v in c if v)
        hot = max(c)
        it1 = sum(math.ceil(v / (8 * nt1)) for v in c)
        it2 = sum(math.ceil(v / (8 * nt2)) for v in c)
        # GEMM1: U=256 units, blocks per item = 256/w1 * s1, each stages nrows*(4096/s1)*2 B
        b1 = 256 // w1 * s1
        act1 = sum(b1 * min(v - j * 8 * nt1, 8 * nt1) * (4096 // s1) * 2
                   for v in c for j in range(math.ceil(v / (8 * nt1)))) / 1e6
        b2 = 128 // w2 * s2
        act2 = sum(b2 * min(v - j * 8 * nt2, 8 * nt2) * (4096 // s2) * 2
                   for v in c for j in range(math.ceil(v / (8 * nt2)))) / 1e6
        hot_items1 = math.ceil(hot / (8 * nt1))
        out.append(dict(local=local, uniq=uniq, hot=hot, it1=it1, it2=it2,
                        w1=it1 * W13_MB, u1=uniq * W13_MB, act1=act1,
                        w2=it2 * W2_MB, u2=uniq * W2_MB, act2=act2,
                        blocks1=it1 * b1, hotblocks1=hot_items1 * b1,
                        part1=local * s1 * 8192 * 4 / 1e6, part2=local * s2 * 4096 * 4 / 1e6))
    keys = list(out[0].keys())
    med = {k: statistics.median(o[k] for o in out) for k in keys}
    lo = {k: min(o[k] for o in out) for k in keys}
    hi = {k: max(o[k] for o in out) for k in keys}
    print(f"bs={bs} {dist} cfg1(nt,rows,S,W)={cfg1} cfg2={cfg2}  [median (min-max) over {len(seeds)} seeds]")
    for k in keys:
        f = (lambda v: f"{v:.0f}") if med[k] >= 20 else (lambda v: f"{v:.1f}")
        print(f"  {k:10s} {f(med[k]):>8s}  ({f(lo[k])}-{f(hi[k])})")
    return med

seeds = list(range(11, 41))
# Layer policy: T=128 g1 cfg 4 (1,1,8,2,8,4), g2 cfg 29 (2,2,16,4,4,4); T=256 g1 cfg 30 (1,2,16,4,4,4), g2 cfg 13 (2,2,16,4,4,4)
summarize(256, (2, 16, 4, 4), (2, 16, 4, 4), seeds)
summarize(256, (1, 8, 4, 4), (2, 16, 4, 4), seeds)
summarize(128, (1, 8, 2, 8), (2, 16, 4, 4), seeds)
summarize(128, (2, 16, 4, 4), (2, 16, 4, 4), seeds)
summarize(32, (1, 8, 4, 4), (1, 8, 4, 4), seeds)
summarize(16, (1, 8, 4, 4), (1, 8, 4, 4), seeds)
summarize(8, (1, 8, 4, 4), (1, 8, 2, 8), seeds)
summarize(1, (1, 8, 2, 8), (1, 1, 4, 4), seeds)
summarize(256, (2, 16, 4, 4), (2, 16, 4, 4), seeds, "skewed")
summarize(128, (1, 8, 2, 8), (2, 16, 4, 4), seeds, "skewed")
