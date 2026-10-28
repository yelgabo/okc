"""CPU check of the m = 1 compressed GEMV in muse_bf16_proj/submission.py.

Packs each layer's weights with pack_cgemv, decodes the stream in numpy the way cgemv_kernel
does (48-byte chunks, the two high-byte formulas on uint32 words, byte_perm with the kernel's
selectors, shift and mask to fp32, the escape table's corrections) and requires every weight
back exactly. Then it runs the kernel's summation order in float32 (per-thread chunk sums,
warp butterflies, warp order, the escape warp's terms, one bf16 rounding) for a Gaussian x, an
outlier x and a probe x, against the float64 product and the judge's bf16-rounded reference.
Two matrices with 1% of entries scaled by 2^20 and 2^-20 must be refused under the
N*K/MAX_ESCAPE_DENOM rule; one with 0.4% scaled by 2^20 packs and exercises escape counts
past one per lane.

    uv run --with torch --with numpy python tools/emulate_gemv.py
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (1024, 4096)}
ADVERSARIAL = (("1% x 2^20", 0.01, 2.0 ** 20), ("1% x 2^-20", 0.01, 2.0 ** -20),
               ("0.4% x 2^20", 0.004, 2.0 ** 20))
M7, M8 = np.uint32(0x07070707), np.uint32(0x80808080)
U4, U8, U16, HI = np.uint32(4), np.uint32(8), np.uint32(16), np.uint32(0xFFFF0000)


def load_module():
    spec = importlib.util.spec_from_file_location("gemv_m1", ROOT / "muse_bf16_proj/submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def byte_perm(a, b, sel):
    pool = [(a >> np.uint32(8 * i)) & np.uint32(0xFF) for i in range(4)]
    pool += [(b >> np.uint32(8 * i)) & np.uint32(0xFF) for i in range(4)]
    out = np.zeros_like(a)
    for i in range(4):
        out |= pool[(sel >> (4 * i)) & 0xF] << np.uint32(8 * i)
    return out


def decode_stream(stream, base4, n, k):
    """The kernel's decode of every weight, before corrections: float32 [n, k]."""
    half, chunks = n // 2, k // 16
    words = np.ascontiguousarray(stream).reshape(half * chunks, 48).view(np.uint32)
    words = words.reshape(half, chunks, 12)
    b4 = np.uint32(base4)

    def hi_lo(c):
        return ((c & M7) + b4) | ((c << U4) & M8)

    def hi_hi(c):
        return (((c >> U4) & M7) + b4) | (c & M8)

    out = np.empty((half, 2, chunks, 16), np.uint32)
    for r in range(2):
        c0, c1 = words[..., 8 + 2 * r], words[..., 9 + 2 * r]
        highs = (hi_lo(c0), hi_lo(c1), hi_hi(c0), hi_hi(c1))
        for q in range(4):
            lo = words[..., 4 * r + q]
            p01, p23 = byte_perm(lo, highs[q], 0x5140), byte_perm(lo, highs[q], 0x7362)
            out[:, r, :, 4 * q] = p01 << U16
            out[:, r, :, 4 * q + 1] = p01 & HI
            out[:, r, :, 4 * q + 2] = p23 << U16
            out[:, r, :, 4 * q + 3] = p23 & HI
    return out.reshape(n, k).view(np.float32)


def table_entries(table, base, escapes):
    """(cta, entry, row_local, k, delta) of every live table entry, after checking its layout."""
    half, pmax, _ = table.shape
    count = table[:, 0, 0]
    e = np.arange(pmax)[None, :]
    live = (e >= 1) & (e <= count[:, None])
    assert pmax % 2 == 0 and count.max() < pmax and count.sum() == escapes, "table size"
    assert not table[:, 0, 1].any() and not table[~live & (e >= 1)].any(), "table padding"
    cs, es = np.nonzero(live)
    key = table[cs, es, 0]
    rows, kks = key >> 16, key & 0xFFFF
    assert np.all((np.diff(kks) >= 0) | (np.diff(cs) != 0)), "table order"
    bits = table[cs, es, 1].astype(np.uint32)
    off = np.clip(((bits >> U8) & np.uint32(0x7F)).astype(np.int64) - base, 0, 7)
    dec = (bits & np.uint32(0x80FF)) | ((base + off) << 8).astype(np.uint32)
    delta = (bits << U16).view(np.float32) - (dec << U16).view(np.float32)
    return cs, es, rows, kks, delta


def butterfly(v):
    lane = np.arange(32)
    for off in (16, 8, 4, 2, 1):
        v = v + v[..., lane ^ off]
    return v[..., 0]


def product(dec, ents, x, nth, cpt):
    """cgemv_kernel's float32 result per output, in its summation order."""
    n, k = dec.shape
    half, nw = n // 2, nth // 32
    w = dec.reshape(half, 2, cpt, nth, 16)
    xs = x.reshape(cpt, nth, 16)
    acc = np.zeros((half, 2, nth), np.float32)
    for i in range(cpt):
        for kk in range(16):
            acc = acc + w[:, :, i, :, kk] * xs[i, :, kk]
    red = butterfly(acc.reshape(half, 2, nw, 32))
    cs, es, rows, kks, delta = ents
    corr = np.zeros((half, 2, 32), np.float32)
    prod = delta * x[kks]
    lane, it = (es - 1) % 32, (es - 1) // 32
    for i in range(int(it.max()) + 1 if len(it) else 0):
        sel = it == i
        corr[cs[sel], rows[sel], lane[sel]] = corr[cs[sel], rows[sel], lane[sel]] + prod[sel]
    corr = butterfly(corr)
    r = np.zeros((half, 2), np.float32)
    for i in range(nw):
        r = r + red[:, :, i]
    return (r + corr).reshape(n)


def bf16_bits(f):
    u = f.astype(np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def bf16_key(bits):
    """Monotone integer per bf16 pattern, so ulp distance is a difference."""
    b = bits.astype(np.int64)
    return np.where(b & 0x8000, -(b & 0x7FFF), b)


def make_xs(k, gen):
    x = torch.randn(k, generator=gen)
    xo = x.clone()
    xo[torch.randperm(k, generator=gen)[: k // 100]] *= 30
    probe = torch.zeros(k)
    probe[int(torch.randint(k, (1,), generator=gen))] = 8.0
    return (("gaussian", x), ("outliers", xo), ("probe", probe))


def check_products(mod, w, dec, ents, gen):
    n, k = w.shape
    nth = mod.cgemv_threads(n, k)
    cpt = k // 16 // nth
    w64 = w.float().numpy().astype(np.float64)
    ok = True
    for xname, x in make_xs(k, gen):
        x32 = x.to(torch.bfloat16).float().numpy()
        r = product(dec, ents, x32, nth, cpt)
        ref64 = w64 @ x32.astype(np.float64)
        ref_bf16 = torch.from_numpy(ref64).to(torch.bfloat16).float().numpy().astype(np.float64)
        out_bits = bf16_bits(r)
        out = (out_bits.astype(np.uint32) << U16).view(np.float32).astype(np.float64)
        err = np.linalg.norm(out - ref_bf16) / np.linalg.norm(ref_bf16)
        err_acc = np.linalg.norm(r.astype(np.float64) - ref64) / np.linalg.norm(ref64)
        ulps = np.abs(bf16_key(out_bits) - bf16_key(bf16_bits(ref_bf16))).max()
        good = np.isfinite(r).all() and err < 1e-3 and (xname != "probe" or ulps <= 1)
        ok &= bool(good)
        print(f"    x {xname:9s} threads {nth:3d} x {cpt} chunks  rel L2 vs bf16 ref {err:.2e}  "
              f"fp32 acc vs fp64 {err_acc:.2e}  max ulp {ulps}  {'ok' if good else 'FAIL'}")
    return ok


def check(mod, name, w, gen):
    n, k = w.shape
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    base = mod.window_base(bits)
    e71 = (bits >> 8) & 0x7F
    escapes = int(((e71 < base) | (e71 >= base + mod.EXP_WINDOW)).sum())
    reject = escapes > n * k // mod.MAX_ESCAPE_DENOM
    packed = mod.pack_cgemv(w)
    print(f"{name:12s} {n:5d}x{k:5d}  base {base:3d}  escapes {escapes:7d} "
          f"({escapes / (n * k):.2e} of weights)  {'rejected' if packed is None else 'packed'}")
    if (packed is None) != reject:
        print("    FAIL: pack_cgemv disagrees with the escape rule")
        return False
    if packed is None:
        return True
    ok = packed["base"] == base and packed["escapes"] == escapes
    ok &= packed["base4"] == base * 0x01010101
    stream, table, pmax = packed["stream"].numpy(), packed["table"].numpy(), packed["pmax"]
    ok &= stream.shape == (n // 2, k // 16, 48) and stream.dtype == np.uint8
    ok &= table.shape == (n // 2, pmax, 2) and table.dtype == np.int32
    dec = decode_stream(stream, packed["base4"], n, k)
    ents = table_entries(table, base, escapes)
    cs, es, rows, kks, delta = ents
    rec = dec.copy()
    rec[2 * cs + rows, kks] += delta
    exact = np.array_equal(rec, w.float().numpy())
    ratio = stream.nbytes / (2 * n * k)
    print(f"    pmax {pmax:3d}  max per CTA {table[:, 0, 0].max():3d}  stream/bf16 {ratio:.4f}  "
          f"with table {(stream.nbytes + table.nbytes) / (2 * n * k):.4f}  "
          f"{'bit-exact' if exact else 'MISMATCH'}")
    ok &= exact and abs(ratio - 0.75) < 1e-9
    return ok and check_products(mod, w, dec, ents, gen)


def main():
    mod = load_module()
    gen = torch.Generator().manual_seed(5151)
    cases = [(name, (torch.randn((n, k), generator=gen) / math.sqrt(k)).to(torch.bfloat16))
             for name, (n, k) in LAYERS.items()]
    n, k = LAYERS["unpooler"]
    for name, frac, scale in ADVERSARIAL:
        w = torch.randn((n, k), generator=gen) / math.sqrt(k)
        w[torch.rand((n, k), generator=gen) < frac] *= scale
        cases.append((name, w.to(torch.bfloat16)))
    ok = True
    for name, w in cases:
        ok &= check(mod, name, w, gen)
    print("ALL CHECKS PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
