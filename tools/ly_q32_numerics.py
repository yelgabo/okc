#!/usr/bin/env python3
"""qkv_gate at M = 16 / 32: a candidate's path against the base wide sk2 kernel (`qkv16_r176`, `qkv32_r208`) and FP64,
CG_GATE's metric.

python tools/ly_q32_numerics.py muse_decode_layer/cand/ly_q32n.py [--inputs 8] [--lock /root/okc/.build/box/gpu.lock]

Both outputs come from the candidate file's own prepare: the base kernel is launched directly from the wide
extension with its packed stream, the candidate through the file's `bfp__run_sk2` (the path its graph takes). x is seeded
BF16 randn at four scales (1, 0.1, 3, 0.5).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--inputs", type=int, default=8)
    ap.add_argument("--layer", default="qkv_gate", help="qkv_gate (N 2304, K 8192) or unpooler (N 8192, K 4096)")
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    a = ap.parse_args()
    if a.lock:
        br.release_lock_for_triton()
        br.LOCK = br.GpuLock(a.lock)
        br.LOCK.acquire("start")
    br.content_addressed_extensions(a.ext_cache)
    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = br.load_module("reference", os.path.join(pdir, "reference.py"))
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    bst = state["bfp"]
    ext = mod.bfp__get_ext()
    L = a.layer
    N, K = {"qkv_gate": (2304, 8192), "unpooler": (8192, 4096)}[L]
    w = weights[L].double()
    g = torch.Generator(device="cuda")
    g.manual_seed(4242)
    worst, eq_tot, n_tot = 0.0, 0, 0
    for m in (16, 32):
        ci, stream, tab, pmax, b4, _ = bst["sk2"][mod.bfp_SK2_PICK[(L, m)]]
        for i in range(a.inputs):
            scale = [1.0, 0.1, 3.0, 0.5][i % 4]
            x = (torch.randn(m, K, device="cuda", generator=g) * scale).to(torch.bfloat16)
            base = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
            mod.bfp_SK2_PDL = 0
            ext.sk2_set_pdl(0)
            ext.sk2(stream, x, base, tab, pmax, b4, m, N, K, ci, m, 0, bst["dummy"])
            cand = mod.bfp__run_sk2(x, L, m, N, K, bst)
            torch.cuda.synchronize()
            r = (x.double() @ w.T).to(torch.bfloat16).double()
            b, c = base.double(), cand.double()
            rel_cb = ((c - b).norm() / b.norm()).item()
            rel_c, rel_b = ((c - r).norm() / r.norm()).item(), ((b - r).norm() / r.norm()).item()
            eq = int((c == b).sum())
            worst = max(worst, rel_cb)
            eq_tot += eq
            n_tot += b.numel()
            print(f"Q32NUM {L} m {m} input {i} scale {scale} cand_vs_base {rel_cb:.2e} cand_vs_fp64 {rel_c:.2e} "
                  f"base_vs_fp64 {rel_b:.2e} equal {eq}/{b.numel()} maxabs {float((c - b).abs().max()):.3e}", flush=True)
    print(f"Q32NUM summary worst cand_vs_base {worst:.2e} (CG_GATE_REL {mod.CG_GATE_REL}) equal {eq_tot}/{n_tot}", flush=True)


if __name__ == "__main__":
    main()
