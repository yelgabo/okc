"""CPU emulation of router_kernel's work split in muse_bf16_proj/submission.py.

Walks the same grid, warp, lane and step indices as the CUDA kernel, so an indexing mistake
shows up here without a GPU: every weight must be read exactly once, and the result must match
an FP64 product.
"""

import numpy as np

N, K = 256, 8192
ROWS_PER_BLOCK, CLUSTER = 16, 8
K_SLICE = K // CLUSTER
STEPS = K_SLICE // 128
GRID = (N // ROWS_PER_BLOCK, CLUSTER)


def emulate(w, x):
    m = x.shape[0]
    partial = np.zeros((CLUSTER, m, N), dtype=np.float32)
    reads = np.zeros((N, K), dtype=np.int32)
    for bx in range(GRID[0]):
        for by in range(GRID[1]):
            for warp in range(ROWS_PER_BLOCK):
                n = bx * ROWS_PER_BLOCK + warp
                lane_sums = np.zeros((32, m), dtype=np.float32)
                for lane in range(32):
                    k0 = by * K_SLICE + lane * 4
                    for s in range(STEPS):
                        k = k0 + s * 128
                        reads[n, k:k + 4] += 1
                        for j in range(4):
                            lane_sums[lane] += w[n, k + j] * x[:, k + j]
                partial[by, :, n] = lane_sums.sum(axis=0, dtype=np.float32)
    out = np.zeros((m, N), dtype=np.float32)
    for c in range(CLUSTER):
        out += partial[c]
    return out, reads


def main():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.02
    for m in (1, 9, 32):
        x = rng.standard_normal((m, K)).astype(np.float32)
        x[0] = 0.0
        x[0, 4097] = 2.0
        out, reads = emulate(w, x)
        ref = x.astype(np.float64) @ w.astype(np.float64).T
        err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        probe_exact = np.array_equal(out[0], (2.0 * w[:, 4097]).astype(np.float32))
        print(f"m={m:2d}  reads min/max {reads.min()}/{reads.max()}  "
              f"rel L2 err {err:.2e}  probe exact {probe_exact}")
        assert reads.min() == 1 and reads.max() == 1
        assert err < 1e-5 and probe_exact


if __name__ == "__main__":
    main()
