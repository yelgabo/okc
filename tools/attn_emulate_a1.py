"""Numerics of backlog A1 (bounded-shift online softmax) in attn_ws.cu, emulated in numpy.

Models one KV head's 64 query rows (row 8j + g: block token j, query head g) the way var 16
and its A1 variants (var 144: tau 4, var 272: tau 8) run them: the context split as _ws_splits
splits it, 64-key tiles with the window and causal masks, the block's own 8 keys as an extra
tile in the last split, then the merge kernel. Per tile, in FP32 unless noted:

    S = Q.K (E4M3 operands, one FP32 rounding), tile max over unmasked scores, times sl2
    tau 0 (var 16): c = max(c, tile_max, -1e30)
    tau > 0 (A1):   c = max(tile_max > c + tau ? tile_max : c, -1e30)
    alpha = 2^(c_old - c);  x = fma(S, sl2, -c);  e = 2^x
    l (per thread, 16 columns each) = l * alpha + sum(e);  O = O * alpha, then += f16(e) . V
    in four k16 steps, each rounded to FP32

The merge weights each split by 2^(c_s - max c) and divides. Error is relative L2 against
FP64 attention with unrounded queries over the same FP8 keys and values; "qfp8" is FP64 with
E4M3 queries, from which the judge's bound is 2.5 x its error + 1.5%, clipped to 3-15%. The
"own q" column is the error against FP64 with the same E4M3 queries, which leaves only the
accumulators, the softmax and the P rounding: the part a shift change can move.

Cases: sliding soft (window edges inside the first tile), sliding edges (a key planted along
token 0's head-0 query at the oldest window position, masked for tokens 1-7), full coherent
over 80K keys, full edges (key at position 0) over 6K keys, and two adversarial full cases:
a ramp (row 0's scores rise across the whole context, so the max grows almost every tile) and
an outlier key late in the context. Also reports the largest P seen, the largest per-thread l
and |O| before the merge, and the share of (warp, tile) steps that take the O rescale branch.

    python3 tools/attn_emulate_a1.py [--seeds S] [--full-keys N]

Not modelled: the order of FP32 adds inside one wgmma, and MUFU ex2's own error (2^-22).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_emulate_fp32 import (  # noqa: E402
    BN, D, LOG2E, QK_SCALE_FACTOR, SM_SCALE, SOFT_C2, e4m3, f16, f32, fma32)

SL2 = QK_SCALE_FACTOR * SM_SCALE * LOG2E
WINDOW = 2048
L = 8
TAUS = (0, 4, 8)


def ws_splits(layer, batch):
    if layer == "sliding":
        return 16 if batch <= 2 else 8 if batch <= 16 else 4
    return 128 if batch <= 2 else max(4, 128 // batch)


def make_case(rng, layer, dist, seq):
    """Queries [64, D] (unit RMS, unrounded), keys and values [seq + 8, D] in E4M3, by
    position (the last 8 are the block's own)."""
    n = seq + L
    q = rng.standard_normal((64, D))
    q /= np.sqrt(np.mean(q ** 2, 1, keepdims=True) + 1e-5)
    if dist in ("soft", "coherent"):
        lo, hi = SOFT_C2["sliding" if dist == "soft" else "full"]
        u = rng.standard_normal(D)
        u /= np.sqrt(np.mean(u ** 2))
        c2 = rng.uniform(lo, hi)
        k = np.sqrt(c2) * u + np.sqrt(1 - c2) * rng.standard_normal((n, D))
    else:
        k = rng.standard_normal((n, D))
    v = rng.standard_normal((n, D))
    if dist == "coherent":
        best = (q * QK_SCALE_FACTOR) @ k.T
        top = np.argsort(-best.max(0))[: min(2048, n // 4)]
        tail = np.ones(n, bool)
        tail[top] = False
        w = rng.standard_normal(D)
        w /= np.sqrt(np.mean(w ** 2))
        v = v + tail[:, None] * w
    if dist == "edges":
        spots = [seq - WINDOW + 1, seq - WINDOW] if layer == "sliding" else [0]
        for t in spots:
            if 0 <= t < seq:
                k[t] = q[0]
    if dist == "ramp":
        # Row 0's score climbs by about 40 log2 units over the context, a little per tile.
        r = np.linspace(0, 1, n)[:, None]
        k = 0.5 * k + r * q[0]
    if dist == "outlier":
        k[int(0.9 * seq)] = 1.2 * q[0]
    return q, e4m3(k), e4m3(v)


def masks(layer, seq):
    """valid [64, seq + 8]: row 8j + g sees positions up to seq + j, and on sliding layers
    only the last 2,048 of them."""
    pos = np.arange(seq + L)
    qpos = seq + np.repeat(np.arange(L), 8)
    ok = pos[None, :] <= qpos[:, None]
    if layer == "sliding":
        ok &= pos[None, :] >= (qpos - WINDOW + 1)[:, None]
    return ok


def reference(q, k, v, ok):
    s = (q * QK_SCALE_FACTOR * SM_SCALE) @ k.T
    s = np.where(ok, s, -np.inf)
    p = np.exp(s - s.max(1, keepdims=True))
    return (p @ v) / p.sum(1, keepdims=True)


def run_tile(state, s_raw, vt, valid, tau, stats):
    m, lth, o = state
    s = np.where(valid, s_raw, -np.inf)
    mx = s.max(1)
    tmax = f32(mx * SL2)
    if tau == 0:
        mu = np.maximum(np.maximum(m, tmax), -1e30)
    else:
        with np.errstate(invalid="ignore"):
            grow = tmax > f32(m + tau)
        mu = np.maximum(np.where(grow, tmax, m), -1e30)
    with np.errstate(invalid="ignore"):
        alpha = f32(2.0 ** f32(m - mu))
    alpha = np.where(np.isnan(alpha), 0.0, alpha)
    x = fma32(s, SL2, -mu[:, None])
    e = f32(2.0 ** x)
    # Thread c of a row's quad holds columns 8i + 2c and 8i + 2c + 1, summed pair by pair.
    part = np.zeros((64, 4))
    for i in range(8):
        for c in range(4):
            part[:, c] = f32(part[:, c] + f32(e[:, 8 * i + 2 * c] + e[:, 8 * i + 2 * c + 1]))
    lth = fma32(lth, alpha[:, None], part)
    # One warp holds rows 16w .. 16w + 15 (tokens 2w and 2w + 1); it skips the O rescale
    # only if all of them keep alpha = 1.
    stats["steps"] += 4
    stats["rescales"] += int(np.sum((alpha != 1).reshape(4, 16).any(1)))
    o = f32(o * alpha[:, None])
    p = f16(e)
    stats["pmax"] = max(stats["pmax"], float(p.max()))
    for kk in range(4):
        o = f32(o + p[:, 16 * kk:16 * kk + 16] @ vt[16 * kk:16 * kk + 16])
    stats["lmax"] = max(stats["lmax"], float(lth.max()))
    stats["omax"] = max(stats["omax"], float(np.abs(o).max()))
    return mu, lth, o


def run_kernel(q, k, v, ok, layer, seq, splits, tau, stats):
    qq = e4m3(q)
    start = max(0, seq - WINDOW + 1) if layer == "sliding" else 0
    per = (seq - start + splits - 1) // splits
    per = (per + BN - 1) // BN * BN
    parts = []
    for split in range(splits):
        lo = min(seq, start + split * per)
        hi = min(seq, lo + per)
        state = (np.full(64, -np.inf), np.zeros((64, 4)), np.zeros((64, D)))
        tiles = [(t0, hi) for t0 in range(lo, hi, BN)]
        if split == splits - 1:
            tiles.append((seq, seq + L))
        for t0, lim in tiles:
            pos = np.arange(t0, t0 + BN)
            inside = pos < lim
            kt = np.zeros((BN, D))
            vt = np.zeros((BN, D))
            kt[inside] = k[pos[inside]]
            vt[inside] = v[pos[inside]]
            valid = np.zeros((64, BN), bool)
            valid[:, inside] = ok[:, pos[inside]]
            state = run_tile(state, f32(qq @ kt.T), vt, valid, tau, stats)
        m, lth, o = state
        lsum = f32(f32(lth[:, 0] + lth[:, 1]) + f32(lth[:, 2] + lth[:, 3]))
        parts.append((m, lsum, o))
    mmax = np.max([p[0] for p in parts], 0)
    lsum = np.zeros(64)
    acc = np.zeros((64, D))
    for m, ls, o in parts:
        with np.errstate(invalid="ignore"):
            w = np.where(m == -np.inf, 0.0, f32(2.0 ** f32(m - mmax)))
        lsum = f32(lsum + f32(w * ls))
        acc = f32(acc + f32(w[:, None] * o))
    return f32(acc * f32(1 / lsum)[:, None])


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


CASES = [
    ("sliding soft", "sliding", "soft", 80000, 32),
    ("sliding edges", "sliding", "edges", 6000, 4),
    ("full coherent", "full", "coherent", None, 32),
    ("full edges", "full", "edges", 6000, 4),
    ("full ramp", "full", "ramp", None, 32),
    ("full outlier", "full", "outlier", None, 32),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--full-keys", type=int, default=80000)
    args = ap.parse_args()
    worst = {tau: 0.0 for tau in TAUS}
    worst_own = {tau: 0.0 for tau in TAUS}
    for name, layer, dist, seq, batch in CASES:
        seq = seq or args.full_keys
        splits = ws_splits(layer, batch)
        errs = {tau: [] for tau in TAUS}
        own = {tau: [] for tau in TAUS}
        qerr = []
        stats = {tau: {"steps": 0, "rescales": 0, "pmax": 0.0, "lmax": 0.0, "omax": 0.0}
                 for tau in TAUS}
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            q, k, v = make_case(rng, layer, dist, seq)
            ok = masks(layer, seq)
            ref = reference(q, k, v, ok)
            ref_q = reference(e4m3(q), k, v, ok)
            qerr.append(rel(ref_q, ref))
            for tau in TAUS:
                out = run_kernel(q, k, v, ok, layer, seq, splits, tau, stats[tau])
                if not np.isfinite(out).all():
                    raise SystemExit(f"{name} tau {tau}: non-finite output")
                errs[tau].append(rel(out, ref))
                own[tau].append(rel(out, ref_q))
        bound = float(np.clip(2.5 * np.mean(qerr) + 0.015, 0.03, 0.15))
        print(f"{name:14s} keys={seq} splits={splits} seeds={args.seeds} qfp8 "
              f"{np.mean(qerr):.2e} judge bound ~{bound:.3f}")
        for tau in TAUS:
            st = stats[tau]
            label = "var 16 " if tau == 0 else f"tau {tau}  "
            worst[tau] = max(worst[tau], max(errs[tau]))
            worst_own[tau] = max(worst_own[tau], max(own[tau]))
            print(f"  {label} rel L2 mean {np.mean(errs[tau]):.3e} max {np.max(errs[tau]):.3e}"
                  f"  own q {np.mean(own[tau]):.2e}/{np.max(own[tau]):.2e}"
                  f"  rescale steps {st['rescales'] / st['steps']:.3f}  max P {st['pmax']:.4g}"
                  f"  max thread l {st['lmax']:.4g}  max |O| {st['omax']:.4g}")
    print("worst rel L2 over cases: "
          + "  ".join(f"tau {t}: {e:.3e}" for t, e in worst.items()))
    print("worst own-q rel L2 over cases: "
          + "  ".join(f"tau {t}: {e:.3e}" for t, e in worst_own.items()))


if __name__ == "__main__":
    main()
