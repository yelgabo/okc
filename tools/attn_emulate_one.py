"""Layouts and numerics of muse_attention_decode/attn_one.cu, emulated in numpy.

  LAYOUT  replays the kernel's data movement on random tiles with the PTX fragment layouts of
          mma.sync.m16n8k16 (f16), wgmma.m64nNk32 (e4m3, accumulator and register-A fragments),
          ldmatrix.trans and stmatrix, and the shared-memory swizzles (128B for the q_len 8
          stages, 64B for V^T, the q_len 1 key/value patterns), and checks that
            q_len 1: S = Q K^T over all 64 dims, O = P V with the thread's dims 16c+e, 16c+8+e;
            q_len 8: the K stage read by the SW128 descriptor is K; the V^T buffer read by the
                     SW64 descriptor is V[key perm(k)][dim sigma(n)] plus the rows of ones; the
                     P bytes the thread packs multiply the right keys; the merge's 4-float
                     groups are consecutive dims.
  NUMERIC the fixed-frame merge over splits against FP64: q_len 8 (E4M3 queries, FP32 scores,
          lazy offset moved by more than 8 log2 units, E4M3 P with row sums from the ones
          rows), q_len 1 (f16 pre-scaled queries, FP32 scores, f16 P), each split adding
          o 2^m and l 2^m in FP32; relative L2 against FP64 with unrounded queries, next to
          the judge's bound (2.5 x the E4M3-query reference error + 1.5%, within 3-15%).

    uv run --with numpy python tools/attn_emulate_one.py [--seeds N] [--keys N]
"""

import argparse

import numpy as np

LOG2E = 1.4426950408889634
QK_SCALE_FACTOR = 5.4730064863875
SM_SCALE = 0.125
SOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}
D = 64
BN = 64


# ---------------------------------------------------------------- number formats


def e4m3_round(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -20)))
    e = np.maximum(e, -6)
    step = 2.0 ** (e - 3)
    r = np.round(a / step) * step
    return np.sign(x) * np.minimum(r, 448.0)


def e4m3_encode(v):
    """Byte encoding of an E4M3 value (already representable)."""
    v = float(v)
    sign = 0x80 if v < 0 else 0
    a = abs(v)
    if a == 0:
        return sign
    e = int(np.floor(np.log2(a)))
    if e < -6:
        mant = int(round(a / 2.0 ** -9))
        return sign | mant
    mant = int(round((a / 2.0 ** e - 1) * 8))
    return sign | ((e + 7) << 3) | mant


def e4m3_decode(byte):
    byte = int(byte) & 0xFF
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0xF
    mant = byte & 7
    if exp == 0:
        return sign * mant * 2.0 ** -9
    return sign * (1 + mant / 8) * 2.0 ** (exp - 7)


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


# ---------------------------------------------------------------- layouts


def k_swz1(chunk, row):
    return chunk ^ ((row & 1) << 2)


def v_swz1(chunk, row):
    return chunk ^ (((row >> 1) & 3) << 1)


def swz128(chunk, row):
    return chunk ^ (row & 7)


def stage_from_tile(tile_bytes, swz):
    """tile_bytes [64 keys][128 bytes] -> the stage as the loader writes it."""
    st = np.zeros((64, 128), np.uint8)
    for r in range(64):
        for ch in range(8):
            st[r, swz(ch, r) * 16:swz(ch, r) * 16 + 16] = tile_bytes[r, ch * 16:ch * 16 + 16]
    return st


def check_q1(rng):
    """One warp (hk, wsub) of the q_len 1 path on one tile."""
    hk, wsub = 1, 1
    keys = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    vals = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    # Avoid NaN encodings (0x7f, 0xff).
    keys[(keys & 0x7F) == 0x7F] = 0
    vals[(vals & 0x7F) == 0x7F] = 0
    kst = stage_from_tile(keys, k_swz1)
    vst = stage_from_tile(vals, v_swz1)
    kval = np.vectorize(e4m3_decode)(keys[:, hk * 64:(hk + 1) * 64])  # [key][dim]
    vval = np.vectorize(e4m3_decode)(vals[:, hk * 64:(hk + 1) * 64])
    q = f16(rng.standard_normal((8, 64)))  # heads of KV head hk, f16
    # Thread (g, c) holds qf[w] = dims 16c + 2w, 16c + 2w + 1 of head g.
    s = np.zeros((8, 32))
    for kb in range(4):
        for g in range(8):
            for c in range(4):
                key = wsub * 32 + kb * 8 + g
                chunk = kst[key, k_swz1(hk * 4 + c, key) * 16:k_swz1(hk * 4 + c, key) * 16 + 16]
                kf = [e4m3_decode(bb) for bb in chunk]  # dims 16c .. 16c + 15 of key
                assert np.allclose(kf, kval[key, 16 * c:16 * c + 16])
        # MMA j2: thread c supplies b0 = dims 16c+4j2+{0,1}, b1 = dims 16c+4j2+{2,3} of key g;
        # a0 = q[g][16c+4j2+{0,1}], a2 = q[g][16c+4j2+{2,3}]; C[g][n] = sum_k A[g][k] B[k][n].
        for j2 in range(4):
            dims = []
            for c in range(4):
                dims += [16 * c + 4 * j2, 16 * c + 4 * j2 + 1]
            for c in range(4):
                dims += [16 * c + 4 * j2 + 2, 16 * c + 4 * j2 + 3]
            # A[g][k] = q[g][dims[k]]; B[k][n] = K[key wsub*32+kb*8+n][dims[k]]
            a = q[:, dims]
            bmat = kval[wsub * 32 + kb * 8:wsub * 32 + kb * 8 + 8][:, dims].T
            s[:, kb * 8:kb * 8 + 8] += a @ bmat
    ref = q @ kval[wsub * 32:wsub * 32 + 32].T
    assert np.allclose(s, ref, atol=1e-9), "q1 QK fragment mapping"
    # Thread (g, c) holds s[kb][e] = S[g][kb*8 + 2c + e].
    p = f16(np.exp(s - s.max(1, keepdims=True)))
    o = np.zeros((8, 64))
    for kk in range(2):
        for e in range(8):
            # n-tile e: B[k][n] = V[key(k)][8n + e]; k index 2c,2c+1 -> keys 2c, 2c+1; 2c+8,.. -> +8.
            keys_k = []
            for c in range(4):
                keys_k += [wsub * 32 + kk * 16 + 2 * c, wsub * 32 + kk * 16 + 2 * c + 1]
            for c in range(4):
                keys_k += [wsub * 32 + kk * 16 + 2 * c + 8, wsub * 32 + kk * 16 + 2 * c + 9]
            # a0 = P[g][keys 2c,2c+1 of the step] (thread c), a2 = keys 2c+8, 2c+9
            a = p[:, [k - wsub * 32 for k in keys_k]]
            bmat = np.zeros((16, 8))
            for k in range(16):
                for n in range(8):
                    key = keys_k[k]
                    # the thread (g = n, c) loads 8 bytes at chunk hk*4 + (g>>1), half g&1
                    cidx = v_swz1(hk * 4 + (n >> 1), key)
                    raw = vst[key, cidx * 16 + (n & 1) * 8:cidx * 16 + (n & 1) * 8 + 8]
                    bmat[k, n] = e4m3_decode(raw[e])
                    assert bmat[k, n] == vval[key, 8 * n + e]
            cmat = a @ bmat  # C[g][n] = O[g][dim 8n + e]
            for n in range(8):
                o[:, 8 * n + e] += cmat[:, n]
    ref_o = p @ vval[wsub * 32:wsub * 32 + 32]
    assert np.allclose(o, ref_o, atol=1e-9), "q1 PV fragment mapping"
    # Thread (g, c): o[e][0] = C[g][n = 2c] = dim 16c + e, o[e][1] = dim 16c + 8 + e.
    for c in range(4):
        dims0 = [8 * (2 * c) + e for e in range(8)]
        dims1 = [8 * (2 * c + 1) + e for e in range(8)]
        assert dims0 == list(range(16 * c, 16 * c + 8)) and dims1 == list(range(16 * c + 8, 16 * c + 16))
    print("LAYOUT q_len 1: QK and PV fragment maps and the merge dims agree with Q K^T, P V")


def ldmatrix_trans_x4(addr_rows):
    """addr_rows: 4 matrices x 8 rows of 16 bytes (as uint8 arrays). Returns per lane (32) the
    four 32-bit registers as byte quadruples, with .trans semantics: lane (g, c) of matrix m
    holds the b16 elements (rows 2c, 2c+1; column g)."""
    out = np.zeros((32, 4, 4), np.uint8)
    for m in range(4):
        rows = addr_rows[m]  # [8][16]
        for lane in range(32):
            g, c = lane >> 2, lane & 3
            e0 = rows[2 * c, 2 * g:2 * g + 2]
            e1 = rows[2 * c + 1, 2 * g:2 * g + 2]
            out[lane, m] = np.concatenate([e0, e1])
    return out


def stmatrix_x4(regs, row_addrs, dst):
    """regs: [32 lanes][4 matrices][4 bytes]; row_addrs[m][r] -> (row index, byte offset) into
    dst; lane (g, c) of matrix m supplies (row g, bytes 4c .. 4c + 3)."""
    for m in range(4):
        for lane in range(32):
            g, c = lane >> 2, lane & 3
            n, off = row_addrs[m][g]
            dst[n, off + 4 * c:off + 4 * c + 4] = regs[lane, m]


def prmt(a, b, sel):
    src = np.concatenate([a, b])
    return np.array([src[(sel >> (4 * i)) & 7] for i in range(4)], np.uint8)


def key_perm(k):
    """V^T K index k (0..31 inside a 32-key block) -> key offset."""
    hb, rest = k >> 4, k & 15
    q, hl, b = rest >> 2, (rest >> 1) & 1, rest & 1
    return 8 * (2 * hb + hl) + 2 * q + b


def sigma(n):
    return 16 * (n >> 4) + 2 * (n & 7) + ((n >> 3) & 1)


def check_q8(rng):
    hk = 1
    vals = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    vals[(vals & 0x7F) == 0x7F] = 0
    vst = stage_from_tile(vals, swz128)
    vval = np.vectorize(e4m3_decode)(vals[:, hk * 64:(hk + 1) * 64])  # [key][dim]
    # The K stage read through the SW128 descriptor: logical byte x of row n sits at chunk
    # (x >> 4) ^ (n & 7); the loader wrote chunk ch of row r at (ch ^ (r & 7)): identical.
    keys = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    kst = stage_from_tile(keys, swz128)
    for n in range(64):
        for x in range(128):
            assert kst[n, ((x >> 4) ^ (n & 7)) * 16 + (x & 15)] == keys[n, x]
    # V^T: every warp wq transposes dims 16wq .. 16wq + 15 of head hk.
    vt = np.zeros((72, 64), np.uint8)  # logical [n][k] bytes; the kernel stores swizzled
    vt_phys = np.zeros((72, 64), np.uint8)
    for wq in range(4):
        for half in range(2):
            mats = []
            for h4 in range(4):
                rows = []
                for r8 in range(8):
                    key = half * 32 + h4 * 8 + r8
                    cidx = (hk * 4 + wq) ^ r8
                    rows.append(vst[key, cidx * 16:cidx * 16 + 16])
                mats.append(np.stack(rows))
            regs = ldmatrix_trans_x4(mats)
            out = np.zeros((32, 4, 4), np.uint8)
            for lane in range(32):
                t0, t1, t2, t3 = regs[lane]
                out[lane, 0] = prmt(t0, t1, 0x6420)
                out[lane, 1] = prmt(t0, t1, 0x7531)
                out[lane, 2] = prmt(t2, t3, 0x6420)
                out[lane, 3] = prmt(t2, t3, 0x7531)
            row_addrs = []
            for m in range(4):
                rows = []
                for r8 in range(8):
                    lane = m * 8 + r8
                    h4 = lane >> 3
                    n = 8 * (2 * wq + (h4 & 1)) + r8
                    chunk = 2 * half + (h4 >> 1)
                    rows.append((n, ((chunk ^ ((n >> 1) & 3)) << 4)))
                row_addrs.append(rows)
            stmatrix_x4(out, row_addrs, vt_phys)
    vt_phys[64:72, :] = 0x38
    # Read back as the SW64 K-major descriptor does: byte k of row n at chunk (k>>4) ^ ((n>>1)&3).
    for n in range(72):
        for k in range(64):
            vt[n, k] = vt_phys[n, ((k >> 4) ^ ((n >> 1) & 3)) * 16 + (k & 15)]
    for n in range(72):
        for k in range(64):
            kk, kin = k >> 5, k & 31
            key = kk * 32 + key_perm(kin)
            want = 1.0 if n >= 64 else vval[key, sigma(n)]
            got = e4m3_decode(vt[n, k])
            assert got == want, (n, k, got, want)
    # P fragments: thread (wq, g, c) holds s[4i + 2hh + e] = S[16wq + 8hh + g][8i + 2c + e].
    # Register-A layout of the e4m3 wgmma: reg 0 = row g, k 4c..4c+3; reg 1 = row g+8; reg 2, 3
    # at k + 16. The kernel packs reg 0 = (s[0], s[1], s[4], s[5]) of the k32 step etc.
    S = rng.standard_normal((64, 64))
    P = e4m3_round(np.exp(S - S.max(1, keepdims=True)))
    for wq in range(4):
        for lane in range(32):
            g, c = lane >> 2, lane & 3
            s = np.zeros(32)
            for i in range(8):
                for hh in range(2):
                    for e in range(2):
                        s[4 * i + 2 * hh + e] = P[16 * wq + 8 * hh + g, 8 * i + 2 * c + e]
            for kk in range(2):
                q0 = s[16 * kk:16 * kk + 16]
                regs = [[q0[0], q0[1], q0[4], q0[5]], [q0[2], q0[3], q0[6], q0[7]],
                        [q0[8], q0[9], q0[12], q0[13]], [q0[10], q0[11], q0[14], q0[15]]]
                for reg, (row, kbase) in enumerate([(g, 4 * c), (g + 8, 4 * c), (g, 16 + 4 * c),
                                                    (g + 8, 16 + 4 * c)]):
                    for bidx in range(4):
                        k = kbase + bidx
                        key = kk * 32 + key_perm(k)
                        assert regs[reg][bidx] == P[16 * wq + row, key], (wq, lane, kk, reg, bidx)
    # The merge: o[4i + 2hh + e] = O[row][col 8i + 2c + e], col n = dim sigma(n).
    for c in range(4):
        for pp in range(4):
            cols = [8 * (2 * pp) + 2 * c + 0, 8 * (2 * pp + 1) + 2 * c + 0,
                    8 * (2 * pp) + 2 * c + 1, 8 * (2 * pp + 1) + 2 * c + 1]
            dims = [sigma(n) for n in cols]
            assert dims == list(range(16 * pp + 4 * c, 16 * pp + 4 * c + 4)), (c, pp, dims)
    # Row sums: columns 64 + 2c, 65 + 2c -> registers 32 + 2hh + e.
    print("LAYOUT q_len 8: SW128 K stage, V^T transpose (perm + sigma + ones rows), P register "
          "order and the merge dims agree")


# ---------------------------------------------------------------- numerics


def make_case(rng, layer, nkeys):
    lo, hi = SOFT_C2[layer]
    u = rng.standard_normal(D)
    u /= np.sqrt(np.mean(u ** 2))
    c2 = rng.uniform(lo, hi)
    k = e4m3_round(np.sqrt(c2) * u + np.sqrt(1 - c2) * rng.standard_normal((nkeys, D)))
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
    return q, k, e4m3_round(v)


def reference(q, k, v):
    s = (q * QK_SCALE_FACTOR * SM_SCALE) @ k.T
    p = np.exp(s - s.max(1, keepdims=True))
    return (p @ v) / p.sum(1, keepdims=True)


def run_one(path, q, k, v, splits):
    """The kernel's arithmetic over `splits` key ranges, merged in the fixed frame."""
    sl2 = QK_SCALE_FACTOR * SM_SCALE * LOG2E
    rows = q.shape[0]
    nkeys = k.shape[0]
    ntl = (nkeys + BN - 1) // BN
    chunk = (ntl + splits - 1) // splits
    acc = np.zeros((rows, D))
    lacc = np.zeros(rows)
    moves = 0
    if path == "q8":
        qq = e4m3_round(q)
    else:
        qq = f16(q * sl2)
    for s0 in range(0, ntl, chunk):
        k0, k1 = s0 * BN, min(nkeys, (s0 + chunk) * BN)
        o = np.zeros((rows, D))
        l = np.zeros(rows)
        off = np.full(rows, -1e30)
        m = np.full(rows, -np.inf)
        for t0 in range(k0, k1, BN):
            kt = k[t0:t0 + BN]
            vt = v[t0:t0 + BN]
            s = f32(qq @ kt.T)
            if path == "q8":
                tmax = f32(s.max(1) * sl2)
                move = tmax > off + 8
                moves += int(move.sum())
                nxt = np.where(move, tmax, off)
                scale = f32(2.0 ** f32(off - nxt))
                o = f32(o * scale[:, None])
                l = f32(l * scale)
                off = nxt
                p = e4m3_round(f32(2.0 ** f32(s * sl2 - off[:, None])))
                l = f32(l + p.sum(1))
                o = f32(o + p @ vt)
            else:
                tmax = s.max(1)
                mu = np.maximum(np.maximum(m, tmax), -1e30)
                alpha = f32(2.0 ** f32(m - mu))
                m = mu
                p32 = f32(2.0 ** f32(s - mu[:, None]))
                l = f32(l * alpha + p32.sum(1))
                o = f32(o * alpha[:, None] + f16(p32) @ vt)
        fr = f32(2.0 ** np.minimum(off if path == "q8" else m, 126))
        acc = f32(acc + o * fr[:, None])
        lacc = f32(lacc + l * fr)
    return acc / lacc[:, None], moves


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def numeric_report(seeds, nkeys_full):
    for layer, nkeys, splits in (("sliding", 2048, 33), ("full", nkeys_full, 16)):
        errs = {n: [] for n in ("qfp8", "q8", "q1")}
        own = {n: [] for n in ("q8", "q1")}
        moves = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            q, k, v = make_case(rng, layer, nkeys)
            ref = reference(q, k, v)
            errs["qfp8"].append(rel(reference(e4m3_round(q), k, v), ref))
            out8, mv = run_one("q8", q, k, v, splits)
            moves.append(mv)
            errs["q8"].append(rel(out8, ref))
            own["q8"].append(rel(out8, reference(e4m3_round(q), k, v)))
            out1, _ = run_one("q1", q, k, v, splits)
            errs["q1"].append(rel(out1, ref))
            sl2 = QK_SCALE_FACTOR * SM_SCALE * LOG2E
            own["q1"].append(rel(out1, reference(f16(q * sl2) / sl2, k, v)))
        bound = float(np.clip(2.5 * np.mean(errs["qfp8"]) + 0.015, 0.03, 0.15))
        print(f"NUMERIC {layer:7s} keys={nkeys} splits={splits} rel L2 vs FP64 (mean/max): "
              + "  ".join(f"{n} {np.mean(e):.2e}/{np.max(e):.2e}" for n, e in errs.items())
              + f"  judge bound ~{bound:.3f}")
        print(f"NUMERIC {layer:7s} vs FP64 over each path's own rounded queries: "
              + "  ".join(f"{n} {np.mean(e):.2e}/{np.max(e):.2e}" for n, e in own.items())
              + f"  lazy-offset moves per 64 rows x {splits} splits: {np.mean(moves):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--keys", type=int, default=16384)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    check_q1(rng)
    check_q8(rng)
    numeric_report(args.seeds, args.keys)


if __name__ == "__main__":
    main()
