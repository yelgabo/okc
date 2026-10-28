"""Bit-exact CPU check of the 12-bit weight code in muse_bf16_proj/submission.py.

window_base, code_bits and comp_units pack Gaussian matrices of the judge's shapes and two
adversarial ones; the packed bytes are decoded with decode8's integer arithmetic (prmt
included) and mapped back to (row, k) through the tile order. Every non-escape must come back
bit-exact, every escape must be restored by its table delta, and the escape table must have
the layout the kernel walks, with and without the balancing row permutation.

    uv run --with torch --with numpy python tools/compress_emulate.py
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
RS = (24, 32)
SEED = 5151
LANE = np.arange(32)


def load_submission():
    spec = importlib.util.spec_from_file_location("submission", ROOT / "muse_bf16_proj" / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prmt(a, b, sel):
    pool = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    d = np.zeros_like(a)
    for i in range(4):
        d |= pool[(sel >> (4 * i)) & 7] << (8 * i)
    return d


def decode8(lw0, lw1, codes, base4):
    ha = ((codes & 0x07070707) + base4) | ((codes << 4) & 0x80808080)
    hb = (((codes >> 4) & 0x07070707) + base4) | (codes & 0x80808080)
    return [prmt(lw0, ha, 0x5140), prmt(lw0, ha, 0x7362), prmt(lw1, hb, 0x5140), prmt(lw1, hb, 0x7362)]


def decode_units(units, base4):
    """uint8 [tiles, units, 768] -> the 16 bf16 patterns per lane the kernel's registers hold,
    uint32 [tiles, units, 32, 16], value 2i + h being half h of register i."""
    nt, nu = units.shape[:2]
    lw = np.ascontiguousarray(units[:, :, :512]).reshape(nt, nu, 32, 16).view(np.uint32)
    cw = np.ascontiguousarray(units[:, :, 512:]).reshape(nt, nu, 32, 8).view(np.uint32)
    regs = np.stack(decode8(lw[..., 0], lw[..., 1], cw[..., 0], base4)
                    + decode8(lw[..., 2], lw[..., 3], cw[..., 1], base4), axis=-1)
    return np.stack([regs & 0xFFFF, regs >> 16], axis=-1).reshape(nt, nu, 32, 16)


def untile(vals, n, k):
    """[n/8, k/64, 32, 16] -> [n, k]: lane 4g+t holds row g, value 4j+2h+p is column 16j+8h+2t+p."""
    v = np.arange(16)
    row = np.arange(n // 8)[:, None, None, None] * 8 + (LANE >> 2)[None, None, :, None]
    col = (np.arange(k // 64)[None, :, None, None] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1)
           + 2 * (LANE & 3)[None, None, :, None] + (v & 1))
    out = np.full((n, k), -1, np.int64)
    out[row, col] = vals
    assert (out >= 0).all(), "the lane/value map does not cover the matrix"
    return out


def f32_of(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def check_table(tab, pmax, R, n, k, escape, rowmap, w32, wd32):
    """The kernel's view of one escape table: counts, k-sorted (row_local<<16 | k, delta)
    entries covering exactly the escapes of each CTA, then the row map."""
    C = n // R
    assert tab.shape == (C, 2 * pmax + R) and tab.dtype == np.int32
    assert pmax % 2 == 0 and ((2 * pmax + R) * 4) % 16 == 0
    counts = escape.reshape(C, R * k).sum(1)
    assert np.array_equal(tab[:, 0], counts) and pmax >= counts.max() + 1
    assert (tab[:, 1] == 0).all()
    delta = w32 - wd32
    for c in range(C):
        cnt = int(tab[c, 0])
        e0 = tab[c, 2:2 * cnt + 1:2].astype(np.int64)
        e1 = tab[c, 3:2 * cnt + 2:2]
        kk, rl = e0 & 0xFFFF, e0 >> 16
        assert (np.diff(kk) >= 0).all(), "entries not sorted by k"
        assert (rl >= 0).all() and (rl < R).all()
        rows, cols = np.nonzero(escape[c * R:(c + 1) * R])
        assert set(zip(rl.tolist(), kk.tolist())) == set(zip(rows.tolist(), cols.tolist())), "entry set"
        assert np.array_equal(e1.view(np.float32), delta[c * R + rl, kk]), "delta bits"
        assert not tab[c, 2 * cnt + 2:2 * pmax].any(), "unused entries must be zero"
    assert np.array_equal(tab[:, 2 * pmax:].reshape(-1), rowmap), "row map"
    return counts


def check_matrix(mod, name, w):
    n, k = w.shape
    bits_t = mod.bf16_bits(w)
    base = mod.window_base(bits_t)
    _, escape_t, decoded_t = mod.code_bits(bits_t, base)
    units = mod.comp_units(bits_t, base).numpy()
    assert units.shape == (n // 8, k // 64, 768) and units.dtype == np.uint8
    bits, dec, escape = bits_t.numpy().astype(np.int64), decoded_t.numpy().astype(np.int64), escape_t.numpy()
    got = untile(decode_units(units, base * 0x01010101), n, k)
    exact_plain = np.array_equal(got[~escape], bits[~escape])
    exact_dec = np.array_equal(got, dec)
    w32, wd32 = f32_of(bits), f32_of(got)
    delta = w32 - wd32
    inexact = int(((w32.astype(np.float64) - wd32.astype(np.float64)) != delta.astype(np.float64))[escape].sum())
    restored = (wd32 + delta) == w32
    not_restored = int((~restored & escape).sum())
    bad_exact = int((~restored & escape & ((w32.astype(np.float64) - wd32.astype(np.float64)) == delta)).sum())
    n_esc = int(escape.sum())
    ok = exact_plain and exact_dec and bad_exact == 0
    line = (f"{name:9s} {n:5d}x{k:<5d} base {base:3d} escapes {n_esc:7d} ({n_esc / (n * k):.2e})"
            f" inexact {inexact} not restored {not_restored}")
    for R in [r for r in RS if n % r == 0]:
        tab, pmax, inexact_t = mod.escape_table(bits_t, decoded_t, escape_t, R, n)
        assert inexact_t == inexact
        counts = check_table(tab.numpy(), pmax, R, n, k, escape, np.arange(n), w32, wd32)
        perm_t = mod.balance_perm(escape_t, R)
        perm = perm_t.numpy()
        assert np.array_equal(np.sort(perm), np.arange(n)), "balance_perm is not a permutation"
        assert np.array_equal(perm.reshape(-1, 4) - perm.reshape(-1, 4)[:, :1], np.tile(np.arange(4), (n // 4, 1)))
        bits_p = bits_t[perm_t]
        _, escape_p, decoded_p = mod.code_bits(bits_p, base)
        assert np.array_equal(escape_p.numpy(), escape[perm])
        tab_p, pmax_p, _ = mod.escape_table(bits_p, decoded_p, escape_p, R, n, perm_t)
        counts_p = check_table(tab_p.numpy(), pmax_p, R, n, k, escape[perm], perm, w32[perm], wd32[perm])
        ratio = (768 * (n // 8) * (k // 64) + (n // R) * (2 * pmax + R) * 4) / (2 * n * k)
        line += (f" | R={R}: pmax {pmax} ratio {ratio:.4f} per-cta {counts.min()}..{counts.max()}"
                 f" balanced {counts_p.min()}..{counts_p.max()} pmax {pmax_p}")
    print(line + ("  ok" if ok else "  FAIL"), flush=True)
    if not ok:
        print(f"  non-escapes exact: {exact_plain}, decoded == code_bits: {exact_dec}, "
              f"escapes with an exact delta not restored: {bad_exact}")
    return ok


def main():
    mod = load_submission()
    gen = torch.Generator().manual_seed(SEED)
    first = next(iter(SHAPES.values()))
    probe = torch.randint(0, 1 << 16, (16, 128), generator=gen, dtype=torch.int32)
    assert np.array_equal(untile(mod.tile_values(probe).numpy(), 16, 128), probe.numpy()), "untile"
    ok = True
    for name, (n, k) in SHAPES.items():
        w = (torch.randn((n, k), generator=gen) / math.sqrt(k)).to(torch.bfloat16)
        ok &= check_matrix(mod, name, w)
    n, k = first
    for name, scale in (("huge1%", 2.0 ** 20), ("tiny1%", 2.0 ** -20)):
        w = (torch.randn((n, k), generator=gen) / math.sqrt(k)).to(torch.bfloat16)
        mask = torch.rand((n, k), generator=gen) < 0.01
        w = torch.where(mask, w * scale, w)
        ok &= check_matrix(mod, name, w)
    print("ALL CHECKS PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
