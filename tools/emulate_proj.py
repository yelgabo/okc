"""CPU emulation of proj_kernel in muse_bf16_proj/submission.py.

The fragment layouts below are mma.m16n8k16's, as the PTX ISA documents them for .bf16. The
emulator rebuilds each A and B tile from what the kernel would hold per lane, multiplies, and
scatters C through the kernel's own partial-sum and reduction indexing. A mistake in the weight
repacking, the token staging or the output indexing shows up as a mismatch against x @ W.T.
"""

import numpy as np

# (k, n, warps, s) for every proj_fwd branch; n is the output width.
CONFIGS = {
    "o": (1024, 8192, 8, 2),
    "unpooler sliced": (4096, 1024, 4, 8),
    "unpooler": (4096, 8192, 8, 4),
    "qkv_gate": (8192, 2304, 8, 8),
    "pooler": (8192, 4096, 8, 8),
}

LANE = np.arange(32)
G, C = LANE // 4, LANE % 4


def pack(w):
    """The same view and permute as _pack_mma_a in the submission."""
    n, k = w.shape
    t = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2)
    return np.ascontiguousarray(t.transpose(0, 3, 2, 5, 4, 1, 6)).reshape(n // 16, k // 16, 32, 8)


def a_tiles(packed_rt):
    """[kt, 32 lanes, 8 elements] -> [kt, 16, 16] using the documented A layout."""
    kt = packed_rt.shape[0]
    a = np.zeros((kt, 16, 16), dtype=packed_rt.dtype)
    for reg, (dr, dc) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
        for e in range(2):
            a[:, G + dr, C * 2 + e + dc] = packed_rt[:, :, reg * 2 + e]
    return a


def staged_b(x, m, k, kt0, t, mt):
    """Shared-memory staging as the kernel writes it, then ldmatrix.x2 as PTX documents it.

    Returns what each lane holds: [t, mt, 32 lanes, 4 elements (b.x pair, b.y pair)].
    """
    ks, stride = t * 16, t * 16 + 8
    xs = np.zeros(mt * 8 * stride, dtype=x.dtype)
    vecs_per_row = ks // 8
    for i in range(mt * 8 * vecs_per_row):
        tok, c = divmod(i, vecs_per_row)
        if tok < m:
            xs[tok * stride + c * 8:tok * stride + c * 8 + 8] = x[tok, kt0 * 16 + c * 8:][:8]
    # Lane l supplies row l % 8 of matrix (l // 8) % 2; lane l receives, from matrix j, row l // 4,
    # elements (l % 4) * 2 and +1.
    row_addr = (LANE & 7) * stride + ((LANE >> 3) & 1) * 8
    frags = np.zeros((t, mt, 32, 4), dtype=x.dtype)
    for kt in range(t):
        for nt in range(mt):
            addr = row_addr + nt * 8 * stride + kt * 16
            for j in range(2):
                rows = addr[j * 8 + G]
                for e in range(2):
                    frags[kt, nt, :, j * 2 + e] = xs[rows + C * 2 + e]
    return frags


def b_tiles(xs):
    """[t, mt, 32, 4] -> [t, mt, 16 k, 8 n] using the documented B layout."""
    t, mt = xs.shape[:2]
    b = np.zeros((t, mt, 16, 8), dtype=xs.dtype)
    for reg in range(2):
        for e in range(2):
            b[:, :, C * 2 + e + reg * 8, G] = xs[:, :, :, reg * 2 + e]
    return b


def emulate(w, x, n_out, warps, s):
    m, k = x.shape
    mt = 1 if m <= 8 else 2 if m <= 16 else 4
    rows, tokens = warps * 16, mt * 8
    kt_total = k // 16
    t = kt_total // s
    packed = pack(w[:n_out])
    out = np.zeros((m, n_out))
    for bx in range(n_out // rows):
        partials = []
        for by in range(s):
            kt0 = by * t
            b = b_tiles(staged_b(x, m, k, kt0, t, mt))
            partial = np.zeros(tokens * rows)
            for warp in range(warps):
                rt = bx * warps + warp
                a = a_tiles(packed[rt, kt0:kt0 + t])
                for nt in range(mt):
                    c_tile = np.einsum("tij,tjn->in", a, b[:, nt])
                    # Documented C layout, then the kernel's partial[t * ROWS + r] writes.
                    r = warp * 16 + G
                    tk = nt * 8 + C * 2
                    partial[tk * rows + r] = c_tile[G, C * 2]
                    partial[(tk + 1) * rows + r] = c_tile[G, C * 2 + 1]
                    partial[tk * rows + r + 8] = c_tile[G + 8, C * 2]
                    partial[(tk + 1) * rows + r + 8] = c_tile[G + 8, C * 2 + 1]
            partials.append(partial)
        for rank in range(s):
            for i in range(tokens * rows // s):
                j = i * s + rank
                tk = j // rows
                if tk < m:
                    out[tk, bx * rows + j % rows] = sum(p[j] for p in partials)
    return out


def main():
    rng = np.random.default_rng(0)
    for name, (k, n_out, warps, s) in CONFIGS.items():
        n_full = max(n_out, 8192) if name.startswith("unpooler") else n_out
        w = rng.standard_normal((n_full, k)) / np.sqrt(k)
        assert np.array_equal(
            np.concatenate([a_tiles(p) for p in pack(w)[:1]], axis=0)[1],
            w[:16, 16:32]), "A layout"
        for m in (1, 7, 16, 32):
            if name == "unpooler sliced" and m > 8 or name == "unpooler" and m <= 8:
                continue
            x = rng.standard_normal((m, k))
            out = emulate(w, x, n_out, warps, s)
            ref = x @ w[:n_out].T
            err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
            print(f"{name:16s} m={m:2d}  rel err {err:.1e}")
            assert err < 1e-12, name


if __name__ == "__main__":
    main()
