"""CPU check of the compressed BF16 path in muse_bf16_proj/submission.py.

Runs the submission's own packer (_compress_bf16, pulled out of the file with ast, since the
file imports sglang) on CPU torch, then decodes the result in numpy the way cproj_kernel does:
the same flat cp.async indexing per block, descriptor lookup by lane, nibble arithmetic and
byte_perm selectors. Every decoded A fragment must equal _pack_mma_a's bytes exactly.

    uv run --with torch --with numpy python tools/compress_emulate.py
"""

import ast
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("_pack_mma_a", "_compress_bf16", "_compressed_bytes", "_adversarial_weights",
         "_escaped_tiles")
CCH, CVECS, CWARPS, EXC_VECS = 8, 6, 4, 16
# The judge's shapes on the compressed path (unpooler: the 1024-row slice used up to 8 tokens).
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (1024, 4096)}
POPC8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def load_submission():
    tree = ast.parse((ROOT / "muse_bf16_proj/submission.py").read_text())
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "submission", "exec"), ns)
    return ns


def byte_perm(a, b, sel):
    pool = np.stack([(a >> (8 * i)) & 0xFF for i in range(4)]
                    + [(b >> (8 * i)) & 0xFF for i in range(4)])
    out = np.zeros_like(a)
    for i in range(4):
        out |= pool[(sel >> (4 * i)) & 0x7] << (8 * i)
    return out


def splits(n, k):
    """cproj_dispatch's choice of S."""
    return 8 if (k == 4096 and n == 1024) else 4 if k == 8192 else 2


def decode(main, desc, exc, base, n, k):
    """Every A fragment cproj_kernel builds, as uint32 [n/16, k/16, 32 lanes, 4 regs]."""
    s = splits(n, k)
    kt_total, t_per_block = k // 16, k // 16 // s
    ch_total, chunks = kt_total // CCH, t_per_block // CCH
    assert t_per_block % CCH == 0 and chunks <= 32 and n % (CWARPS * 16) == 0
    wc = np.frombuffer(main.numpy().tobytes(), dtype=np.uint32).astype(np.int64).reshape(-1, 4)
    dflat = desc.numpy().view(np.uint32).astype(np.int64).reshape(-1)
    ex = np.frombuffer(exc.numpy().tobytes(), dtype=np.uint32).astype(np.int64).reshape(-1, 2)
    base4 = base * 0x01010101
    out = np.zeros((n // 16, kt_total, 32, 4), dtype=np.int64)
    tile = np.arange(n // 16)[:, None]
    lane = np.arange(32)[None, :]
    for by in range(s):
        kt0 = by * t_per_block
        wsrc = (tile * ch_total + kt0 // CCH) * CVECS * 32 + lane
        # Lane c's descriptor, read by every lane through __shfl_sync.
        my_desc = dflat[(tile * ch_total + kt0 // CCH + np.arange(chunks)[None, :])]
        for c in range(chunks):
            d = my_desc[:, c][:, None]
            mask = d & 0xFF
            first = d >> 8
            vecs = [wc[wsrc + (c * CVECS + v) * 32] for v in range(CVECS)]
            for i in range(CCH):
                lo = vecs[i // 2]
                l0, l1 = (lo[..., 0], lo[..., 1]) if i % 2 == 0 else (lo[..., 2], lo[..., 3])
                word = vecs[4 + i // 4][..., i % 4]
                h0 = ((word << 4) & 0x80808080) | ((word & 0x07070707) + base4)
                h1 = (word & 0x80808080) | (((word >> 4) & 0x07070707) + base4)
                flagged = ((mask >> i) & 1).astype(bool)
                slot = POPC8[mask & ((1 << i) - 1)]
                eidx = np.where(flagged, (first + slot) * 32 + lane, 0)
                e = ex.reshape(-1, 2)[eidx]
                h0 = np.where(flagged, e[..., 0], h0)
                h1 = np.where(flagged, e[..., 1], h1)
                kt = kt0 + c * CCH + i
                out[:, kt, :, 0] = byte_perm(l0, h0, 0x5140)
                out[:, kt, :, 1] = byte_perm(l0, h0, 0x7362)
                out[:, kt, :, 2] = byte_perm(l1, h1, 0x5140)
                out[:, kt, :, 3] = byte_perm(l1, h1, 0x7362)
    return out


def check(ns, name, w):
    n, k = w.shape
    comp = ns["_compress_bf16"](w)
    got = decode(*comp, n, k).astype(np.uint32)
    want = np.frombuffer(ns["_pack_mma_a"](w).view(torch.int16).numpy().tobytes(),
                         dtype=np.uint32).reshape(got.shape)
    ok = np.array_equal(got, want)
    escaped = ns["_escaped_tiles"](comp)
    ratio = ns["_compressed_bytes"](comp) / (w.numel() * 2)
    print(f"{name:10s} {str(tuple(w.shape)):14s} base {comp[3]:3d}  escaped tiles "
          f"{escaped:6d} of {w.numel() // 256:6d}  ratio {ratio:.4f}  "
          f"{'bit-exact' if ok else 'MISMATCH'}")
    return ok


def main():
    ns = load_submission()
    ok = True
    # CPU generator: the judge's distribution, not its exact values (it seeds a CUDA generator).
    gen = torch.Generator().manual_seed(5151)
    for name, (n, k) in LAYERS.items():
        w = (torch.randn((n, k), generator=gen) / math.sqrt(k)).to(torch.bfloat16)
        ok &= check(ns, name, w)
    for name, w in ns["_adversarial_weights"]("cpu").items():
        ok &= check(ns, name, w)
    print("ALL BIT-EXACT" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
