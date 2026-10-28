"""Numerics of the q_len 8 attention paths, emulated in numpy against an FP64 reference.

  POLY   fits the degree-5 polynomial for 2^f, f in [-0.5, 0.5], used by ex2_poly in
         attn_ws.cu (constant term fixed at 1, minimax on relative error by Lawson iterations),
         rounds its coefficients to FP32 and measures the whole FP32 routine (round-to-nearest
         split, Horner by FMA, exponent add) against 2^x over [-126, 0].
  ATTN   one KV head's 64 query rows (8 tokens x 8 heads) over a soft-key cache, as the judge's
         generate_input builds it (SOFT_C2 key mixing; on full layers the coherent value tail),
         in 64-key tiles with an online softmax, for four paths:
           f16    attn_ws var 0/2: scale folded into Q before E4M3, Q.K accumulated in f16
                  (rounded after each k32 wgmma), half2 max/exp/sum, f16 P, FP32 P.V
           fp32   attn_ws var 16+: unscaled Q in E4M3, FP32 S, FP32 max/sum/exp, f16 P, FP32 P.V
           poly   fp32 with a quarter of the exponentials from ex2_poly (var 64+)
           fa3    FA3's FP8 path: FP32 S and softmax, P as E4M3 of exp2(x + 8), FP32 P.V
         The error is relative L2 of the attention output against FP64 with unrounded queries
         over the same FP8 cache; "qfp8" is the reference with E4M3 queries, from which the
         judge's bound is 2.5 x its error + 1.5%, clipped to 3-15%. A second line compares each
         path with FP64 attention over its own rounded queries, which leaves only the error of
         the accumulators, the softmax and the P rounding.

    python3 tools/attn_emulate_fp32.py [--keys N] [--seeds S]

What it cannot see: the hardware's internal accumulation order inside one wgmma (the f16 path
rounds once per k32 instruction here) and MUFU's own ex2 error (2^-22, below everything else).
"""

import argparse

import numpy as np

LOG2E = 1.4426950408889634
QK_SCALE_FACTOR = 5.473
SM_SCALE = 1 / 8
SOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}
D = 64
BN = 64


def e4m3(x):
    """Round to FP8 E4M3 (round to nearest even, saturating at 448), returned as float64."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -20)))
    e = np.maximum(e, -6)
    step = 2.0 ** (e - 3)
    r = np.round(a / step) * step
    return np.sign(x) * np.minimum(r, 448.0)


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def fma32(a, b, c):
    return f32(np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64))


def fit_poly(deg=5, iters=60):
    f = np.cos(np.linspace(0, np.pi, 4001)) * 0.5
    target = 2.0 ** f
    basis = np.stack([f ** k for k in range(1, deg + 1)], 1)
    w = np.ones_like(f)
    for _ in range(iters):
        sw = np.sqrt(w) / target
        c, *_ = np.linalg.lstsq(basis * sw[:, None], (target - 1) * sw, rcond=None)
        err = np.abs((1 + basis @ c - target) / target)
        w = w * err
        w /= w.sum()
    return np.concatenate([[1.0], c])


def ex2_poly(x, coef):
    """attn_ws.cu's ex2_poly, step by step in FP32."""
    x = np.maximum(f32(x), -126.0)
    magic = 12582912.0
    t = f32(x + magic)
    f = f32(x - f32(t - magic))
    c = f32(coef)
    y = fma32(c[5], f, c[4])
    for k in (3, 2, 1, 0):
        y = fma32(y, f, c[k])
    n = (t - magic).astype(np.int64)
    bits = y.astype(np.float32).view(np.int32).astype(np.int64) + (n << 23)
    return bits.astype(np.int32).view(np.float32).astype(np.float64)


def poly_report():
    coef = fit_poly()
    x = f32(np.concatenate([np.linspace(-126, 0, 2_000_001),
                            -np.random.default_rng(0).random(10 ** 6)]))
    got = ex2_poly(x, coef)
    rel = np.abs(got / 2.0 ** x - 1)
    fit = np.linspace(-0.5, 0.5, 100_001)
    fit_err = np.abs(np.polyval(coef[::-1], fit) / 2.0 ** fit - 1).max()
    print(f"POLY degree-5 fit in FP64 over [-0.5, 0.5]: max rel error {fit_err:.1e}")
    print("POLY coefficients (c0..c5, FP32):",
          ", ".join(f"{v:.8e}" for v in f32(coef)))
    print(f"POLY max rel error over [-126, 0]: {rel.max():.2e} (median {np.median(rel):.1e}); "
          f"ex2.approx.f32 is documented at 2^-22 = {2.0 ** -22:.1e}")
    return f32(coef)


def make_case(rng, layer, nkeys):
    lo, hi = SOFT_C2[layer]
    u = rng.standard_normal(D)
    u /= np.sqrt(np.mean(u ** 2))
    c2 = rng.uniform(lo, hi)
    k = e4m3(np.sqrt(c2) * u + np.sqrt(1 - c2) * rng.standard_normal((nkeys, D)))
    v = rng.standard_normal((nkeys, D))
    q = 2 * rng.standard_normal((64, D))
    q = q / np.sqrt(np.mean(q ** 2, 1, keepdims=True) + 1e-5)
    if layer == "full":
        best = (q * QK_SCALE_FACTOR) @ k.T
        top = np.argsort(-best.max(0))[: min(2048, nkeys // 4)]
        tail = np.ones(nkeys, bool)
        tail[top] = False
        w = rng.standard_normal(D)
        w /= np.sqrt(np.mean(w ** 2))
        v = v + tail[:, None] * w
    return q, k, e4m3(v)


def reference(q, k, v):
    s = (q * QK_SCALE_FACTOR * SM_SCALE) @ k.T
    p = np.exp(s - s.max(1, keepdims=True))
    return (p @ v) / p.sum(1, keepdims=True)


def run_path(path, q, k, v, coef):
    sl2 = QK_SCALE_FACTOR * SM_SCALE * LOG2E
    rows = q.shape[0]
    if path == "f16":
        qq = e4m3(q * sl2)
    else:
        qq = e4m3(q)
    o = np.zeros((rows, D))
    m = np.full(rows, -np.inf)
    lsum = np.zeros(rows)
    for t0 in range(0, k.shape[0], BN):
        kt = k[t0:t0 + BN]
        vt = v[t0:t0 + BN]
        if path == "f16":
            s = f16(qq[:, :32] @ kt[:, :32].T)
            s = f16(s + qq[:, 32:] @ kt[:, 32:].T)
            mx = s.max(1)
            mnew = np.maximum(np.maximum(m, mx), -65504.0)
            alpha = f16(2.0 ** f16(m - mnew))
            p = f16(2.0 ** f16(s - mnew[:, None]))
            # Eight half2 adds per thread per row (two columns each), then the two halves and
            # the quad in FP32.
            ps = p.reshape(rows, 8, 4, 2)
            acc = np.zeros((rows, 4, 2))
            for i in range(8):
                acc = f16(acc + ps[:, i])
            tile_sum = acc.sum((1, 2))
            pv = p
        else:
            s = f32(qq @ kt.T)
            mx = s.max(1)
            mnew = np.maximum(np.maximum(m, f32(mx * sl2)), -1e30)
            alpha = f32(2.0 ** f32(m - mnew))
            x = fma32(s, sl2, -mnew[:, None])
            if path == "fa3":
                e = f32(2.0 ** (x + 8))
                tile_sum = e.sum(1)
                pv = e4m3(e)
            else:
                e = f32(2.0 ** x)
                if path == "poly":
                    cols = (np.arange(BN) // 8) % 4 == 3
                    e[:, cols] = ex2_poly(x[:, cols], coef)
                tile_sum = e.sum(1)
                pv = f16(e)
        m = mnew
        lsum = lsum * alpha + tile_sum
        o = f32(o * alpha[:, None] + pv @ vt)
    return o / lsum[:, None]


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=int, default=16384)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    coef = poly_report()
    for layer, nkeys in (("sliding", 2048), ("full", args.keys)):
        errs = {name: [] for name in ("qfp8", "f16", "fp32", "poly", "fa3")}
        own = {name: [] for name in ("f16", "fp32", "poly", "fa3")}
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            q, k, v = make_case(rng, layer, nkeys)
            ref = reference(q, k, v)
            errs["qfp8"].append(rel(reference(e4m3(q), k, v), ref))
            sl2 = QK_SCALE_FACTOR * SM_SCALE * LOG2E
            own_q = {"f16": e4m3(q * sl2) / sl2}
            for path in ("f16", "fp32", "poly", "fa3"):
                out = run_path(path, q, k, v, coef)
                errs[path].append(rel(out, ref))
                own[path].append(rel(out, reference(own_q.get(path, e4m3(q)), k, v)))
        bound = float(np.clip(2.5 * np.mean(errs["qfp8"]) + 0.015, 0.03, 0.15))
        print(f"ATTN {layer:7s} keys={nkeys} seeds={args.seeds} rel L2 vs FP64 (mean, max): "
              + "  ".join(f"{n} {np.mean(e):.2e}/{np.max(e):.2e}" for n, e in errs.items())
              + f"  judge bound ~{bound:.3f}")
        print(f"ATTN {layer:7s} same paths vs FP64 with each path's own E4M3 queries (mean, max): "
              + "  ".join(f"{n} {np.mean(e):.2e}/{np.max(e):.2e}" for n, e in own.items()))


if __name__ == "__main__":
    main()
