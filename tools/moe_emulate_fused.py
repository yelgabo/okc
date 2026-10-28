"""CPU emulation of the fused MoE kernel's weight path in muse_moe_w4a16/submission.py.

Runs the submission's own repack (repack_expert) on CPU torch tensors, then decodes every
word of every unit exactly as the kernel does (the 36,864-byte unit layout, the per-lane
fragment addressing, the bit permutation, the scale byte shifted into a bf16 exponent, a bf16
multiply) and scatters the values through the mma.m16n8k16 A-fragment layout and the
epilogue's row mapping. A mistake in any of them shows up as a mismatch against the plain
NVFP4 dequantization.
    uv run --with torch --with numpy python tools/moe_emulate_fused.py
"""

import ast
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NAMES = ("W4_BITPOS", "unit_rows", "repack_expert", "repack_expert_fat", "_perm", "_stack_last", "_to_i64",
         "_to_u8_words", "_cat_last", "scales_ok")


def load_repack():
    tree = ast.parse((ROOT / "muse_moe_w4a16/submission.py").read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in NAMES for t in n.targets))]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "submission", "exec"), ns)
    return ns


def bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)


def round_bf16(x):
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return bf16(b.astype(np.uint32))


def scale_word(byte):
    return ((int(byte) * 16) + 0x7780) & 0xFFFF


def dequant(w, sg, sg8):
    """The kernel's dequant(): four bf16x2 registers from one word; returns [(lo, hi)] * 4."""
    w = np.uint32(w)
    m = np.uint32(0x81C081C0)
    r = [w & m, (w << np.uint32(6)) & m,
         ((w >> np.uint32(6)) | (w << np.uint32(26))) & m,
         ((w >> np.uint32(4)) & np.uint32(0x00400040)) | ((w << np.uint32(4)) & np.uint32(0x81808180))]
    out = []
    for j, rj in enumerate(r):
        s = bf16(sg8 if j & 1 else sg)
        lo = bf16(int(rj) & 0xFFFF) * s
        hi = bf16(int(rj) >> 16) * s
        assert round_bf16(lo) == lo and round_bf16(hi) == hi, "product not exact in bf16"
        out.append((lo, hi))
    return out


def check(kind, n, seed):
    k = 4096
    ns = load_repack()
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    exp = rng.integers(1, 16, size=(n, k // 16))
    man = rng.integers(0, 8, size=(n, k // 16))
    sf = (exp << 3 | man).astype(np.uint8)
    sf[sf == 0x7F] = 0x7E
    assert ns["scales_ok"](torch.from_numpy(sf))
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(n, k).astype(np.int64)
    val = GRID[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    scale = (2.0 ** ((sf >> 3).astype(np.int64) - 7)) * (1 + (sf & 7) / 8.0)
    ref = val * np.repeat(scale, 16, axis=1)

    units = ns["repack_expert"](torch.from_numpy(packed), torch.from_numpy(sf), kind).numpy()
    assert units.shape == (n // 16, 36864), units.shape
    got = np.full((n, k), np.nan)
    for rb in range(n // 16):
        unit = units[rb]
        words = unit[:32768].view(np.uint32)
        for w in range(8):
            for c in range(8):
                for lane in range(32):
                    g, t4 = lane // 4, lane % 4
                    sb = unit[32768 + w * 512 + lane * 16 + 2 * c: 32768 + w * 512 + lane * 16 + 2 * c + 2]
                    sg, sg8 = scale_word(sb[0]), scale_word(sb[1])
                    for s in range(4):
                        regs = dequant(words[w * 1024 + c * 128 + lane * 4 + s], sg, sg8)
                        for j, (lo, hi) in enumerate(regs):
                            h, jhi = j & 1, j >> 1
                            row = (h * (n // 2) + rb * 8 + g) if kind == 1 else (rb * 16 + h * 8 + g)
                            col = 512 * w + 64 * c + 16 * t4 + 4 * s + 2 * jhi
                            got[row, col] = lo * 128
                            got[row, col + 1] = hi * 128
    assert not np.isnan(got).any(), "some weights were never produced"
    bad = np.argwhere(got != ref)
    print(f"kind {kind} n={n}: {len(bad)} mismatches of {n * k}")
    if len(bad):
        r, c = bad[0]
        print("  first at", r, c, got[r, c], ref[r, c])
    return not len(bad)


if __name__ == "__main__":
    ok = check(1, 64, 1) and check(2, 48, 2)
    print("OK" if ok else "FAILED")


def check_fat(kind, n, seed):
    """repack_expert_fat against plain dequantization, decoded as the fat path's consumer
    warps read it: warp (wg, w) of unit u holds 16-row block 16 u + 4 wg + w; word (lane, step)
    of stage st is K = 256 st + 16 step + 8 khi + 2 t4 + jlo; scales [wg][w][g][step][h]."""
    k = 4096
    ns = load_repack()
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    exp = rng.integers(1, 16, size=(n, k // 16))
    man = rng.integers(0, 8, size=(n, k // 16))
    sf = (exp << 3 | man).astype(np.uint8)
    sf[sf == 0x7F] = 0x7E
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(n, k).astype(np.int64)
    val = GRID[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    scale = (2.0 ** ((sf >> 3).astype(np.int64) - 7)) * (1 + (sf & 7) / 8.0)
    ref = val * np.repeat(scale, 16, axis=1)
    units = ns["repack_expert_fat"](torch.from_numpy(packed), torch.from_numpy(sf), kind).numpy()
    assert units.shape == (n // 256, 16, 36864), units.shape
    got = np.full((n, k), np.nan)
    for u in range(n // 256):
        for st in range(16):
            stage = units[u, st]
            words = stage[:32768].view(np.uint32)
            for wg in range(4):
                for w in range(4):
                    rb = 16 * u + 4 * wg + w
                    for lane in range(32):
                        g, t4 = lane // 4, lane % 4
                        sb = stage[32768 + (wg * 4 + w) * 256 + g * 32: 32768 + (wg * 4 + w) * 256 + g * 32 + 32]
                        for step in range(16):
                            sg, sg8 = scale_word(sb[2 * step]), scale_word(sb[2 * step + 1])
                            regs = dequant(words[(wg * 4 + w) * 512 + lane * 16 + step], sg, sg8)
                            for j, (lo, hi) in enumerate(regs):
                                h, khi = j & 1, j >> 1
                                row = (h * (n // 2) + rb * 8 + g) if kind == 1 else (rb * 16 + h * 8 + g)
                                col = 256 * st + 16 * step + 8 * khi + 2 * t4
                                got[row, col] = lo * 128
                                got[row, col + 1] = hi * 128
    assert not np.isnan(got).any(), "some weights were never produced"
    bad = np.argwhere(got != ref)
    print(f"fat kind {kind} n={n}: {len(bad)} mismatches of {n * k}")
    if len(bad):
        r, c = bad[0]
        print("  first at", r, c, got[r, c], ref[r, c])
    return not len(bad)


if __name__ == "__main__":
    ok = check_fat(1, 512, 3) and check_fat(2, 256, 4)
    print("FAT OK" if ok else "FAT FAILED")
