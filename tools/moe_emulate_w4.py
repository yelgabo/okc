"""CPU emulation of the W4A16 GEMM's weight path in muse_moe_w4a16/submission.py.

Runs the submission's own repack functions (w4_repack_values, w4_repack_scales) on numpy
arrays, then decodes every lane's words exactly as dequant() does (mask, shift, rotate, the
scale byte shifted into a bf16 exponent, a bf16 multiply) and scatters them through the
mma.m16n8k16 A-fragment layout and the kernel's epilogue row mapping. A mistake in the bit
permutation, the fragment order, the scale order or the unit-to-row mapping shows up as a
mismatch against the plain NVFP4 dequantization.
    uv run --with numpy python tools/moe_emulate_w4.py
"""

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NAMES = ("W4_BITPOS", "w4_unit_rows", "w4_repack_values", "w4_repack_scales", "w4_group_units")


def load_repack():
    tree = ast.parse((ROOT / "muse_moe_w4a16/submission.py").read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in NAMES for t in n.targets))]
    ns = {
        "_perm": lambda a, dims: np.transpose(a, dims),
        "_stack_last": lambda a, b: np.stack((a, b), axis=-1),
        "_to_i64": lambda a: a.astype(np.int64),
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), "submission", "exec"), ns)
    return ns


def bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)


def round_bf16(x):
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return bf16(b.astype(np.uint32))


def dequant(w, sw, tile):
    w = np.uint32(w)
    m = np.uint32(0x81C081C0)
    r = [w & m, (w << np.uint32(6)) & m,
         ((w >> np.uint32(6)) | (w << np.uint32(26))) & m,
         ((w >> np.uint32(4)) & np.uint32(0x00400040)) | ((w << np.uint32(4)) & np.uint32(0x81808180))]
    b = [(int(sw) >> (8 * i)) & 0xFF for i in range(4)]
    sel = (b[2 * tile], b[2 * tile + 1])
    sc = [(x * 16 + 0x7780) & 0xFFFF for x in sel]
    out = []
    for j, rj in enumerate(r):
        s = bf16(sc[j & 1])
        lo = bf16(int(rj) & 0xFFFF) * s
        hi = bf16(int(rj) >> 16) * s
        # bf16 multiply: the product must already be a bf16 value (exact).
        assert round_bf16(lo) == lo and round_bf16(hi) == hi
        out.append((lo, hi))
    return out


def check(kind, n, k, seed):
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

    words = ns["w4_repack_values"](packed, kind).astype(np.uint64).astype(np.uint32)
    sbytes = ns["w4_repack_scales"](sf, kind).astype(np.uint8)
    nst = k // 64
    units = n // 32
    wv = words.reshape(units, nst, 2, 32, 4)
    sv = sbytes.reshape(units, nst, 8, 16).view(np.uint32).reshape(units, nst, 8, 4)
    got = np.full((n, k), np.nan)
    for u in range(units):
        for st in range(nst):
            for h in range(2):
                for lane in range(32):
                    g, t = lane // 4, lane % 4
                    for i in range(4):
                        j1, tile = i // 2, i % 2
                        regs = dequant(wv[u, st, h, lane, i], sv[u, st, g, h * 2 + j1], tile)
                        kb = st * 4 + h * 2 + j1
                        for j, (lo, hi) in enumerate(regs):
                            hh, khi = j & 1, j >> 1
                            if kind == 1:
                                row = tile * (n // 2) + u * 16 + hh * 8 + g
                            else:
                                row = u * 32 + tile * 16 + hh * 8 + g
                            col = kb * 16 + khi * 8 + 2 * t
                            got[row, col] = lo * 128
                            got[row, col + 1] = hi * 128
    assert not np.isnan(got).any(), "some weights were never produced"
    bad = np.argwhere(got != ref)
    print(f"kind {kind} n={n} k={k}: {len(bad)} mismatches of {n * k}")
    if len(bad):
        r, c = bad[0]
        print("  first at", r, c, got[r, c], ref[r, c])
    return not len(bad)


def epilogue_rows(kind):
    """The epilogue's output feature for accumulator (tile, hh) at lane g must be the weight
    row the fragments came from (unit rows u * 16 + ... for gate/up, u * 32 + ... for W2)."""
    ok = True
    for u in range(3):
        for tile in range(2):
            for hh in range(2):
                for g in range(8):
                    if kind == 1:
                        kernel = (u * 16 + hh * 8 + g) + tile * 4096
                        frag = tile * 4096 + u * 16 + hh * 8 + g
                    else:
                        kernel = u * 32 + tile * 16 + hh * 8 + g
                        frag = u * 32 + tile * 16 + hh * 8 + g
                    ok &= kernel == frag
    return ok


def layout1(units, experts=3, seed=4):
    """w4_group_units against the kernel's LAYOUT 1 addressing: the uint4 a warp loads for
    (expert, unit, K split z of S, stage, half, lane), and its scale uint4, must be the ones
    layout 0 puts at that unit's stage."""
    ns = load_repack()
    vals = np.arange(experts * units * 64 * 64)
    scl = np.arange(experts * units * 64 * 8)
    gv = ns["w4_group_units"](vals, 64 * 64)
    gs = ns["w4_group_units"](scl, 64 * 8)
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        e, u = int(rng.integers(experts)), int(rng.integers(units))
        splits = int(rng.choice([1, 2, 4, 8]))
        nst = 64 // splits
        z, s = int(rng.integers(splits)), int(rng.integers(nst))
        h, lane = int(rng.integers(2)), int(rng.integers(32))
        st0 = ((e * (units // 8) + u // 8) * 64 + z * nst) * 8 + (u & 7)
        w1 = gv[st0 * 64 + lane + s * 512 + h * 32]
        s1 = gs[st0 * 8 + lane // 4 + s * 64]
        st = (e * units + u) * 64 + z * nst + s
        if w1 != vals[st * 64 + h * 32 + lane] or s1 != scl[st * 8 + lane // 4]:
            print("layout 1 mismatch", e, u, splits, z, s, h, lane)
            return False
    print(f"layout 1 addressing ok ({units} units)")
    return True


if __name__ == "__main__":
    ok = check(1, 64, 128, 1) and check(2, 64, 128, 2) and check(1, 128, 192, 3)
    ok = ok and epilogue_rows(1) and epilogue_rows(2) and layout1(256) and layout1(128)
    print("OK" if ok else "FAILED")
