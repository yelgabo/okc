#!/usr/bin/env python3
"""Bitwise check of the layer-b2 candidates' boundary kernels against ly_stk3's, standalone (no prepare).

python tools/ly_b2_bitwise.py   (on the box, from /root/okc, under gpu.lock)

g32: b1 at T = 32, ly_stk3's Gluon `_graft_row_boundary_pre_cg` (POST False) against ly_b2g32's glue b1.
sp: b2 at T = 16 and 32, ly_stk3's Gluon POST kernel against ly_b2sp's. Seeded inputs at four scales; NewStream
(FP32) and Out (BF16) compared bit for bit.
"""
import importlib.util
import os
import sys

import torch

C = "muse_decode_layer/cand"
sys.path.insert(0, os.path.abspath("muse_decode_layer"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{C}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base, g32, sp = load("ly_stk3"), load("ly_b2g32"), load("ly_b2sp")
H = 8192
bad = 0
for seed in range(8):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = [1e-3, 1.0, 30.0, 300.0][seed % 4]
    for T in (16, 32):
        br = (torch.randn(T, H, device="cuda", generator=g) * scale).bfloat16()
        st = torch.randn(T, H, device="cuda", generator=g) * 3.0
        pre = (1 + 0.1 * torch.randn(H, device="cuda", generator=g)).bfloat16()
        post = (1 + 0.1 * torch.randn(H, device="cuda", generator=g)).bfloat16()
        al = 1 + 0.05 * torch.randn(H, device="cuda", generator=g)
        be = 0.5 + 0.05 * torch.randn(H, device="cuda", generator=g)
        outs = {}
        for tag, mod, post_ in (("base_b1", base, False), ("base_b2", base, True), ("sp_b2", sp, True)):
            h = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
            s = torch.empty(T, H, device="cuda")
            mod._graft_row_boundary_pre_cg[(T,)](br, st, post, pre, al, be, h, s, H, H, post_, 4, 32, num_warps=32)
            outs[tag] = (h, s)
        if T == 32:
            h = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
            s = torch.empty(T, H, device="cuda")
            g32._pdl8_glue_call('b1', br, st, pre, al, be, h, s, s, False)
            outs["g32_b1"] = (h, s)
        torch.cuda.synchronize()
        for a_, b_ in (("base_b2", "sp_b2"), ("base_b1", "g32_b1")):
            if b_ not in outs:
                continue
            (h0, s0), (h1, s1) = outs[a_], outs[b_]
            eq_s = bool(torch.equal(s0.view(torch.int32), s1.view(torch.int32)))
            eq_h = bool(torch.equal(h0.view(torch.int16), h1.view(torch.int16)))
            nh = int((h0.view(torch.int16) != h1.view(torch.int16)).sum())
            ns = int((s0.view(torch.int32) != s1.view(torch.int32)).sum())
            bad += (not eq_s) + (not eq_h)
            print(f"B2BIT seed {seed} scale {scale} T {T} {a_} vs {b_}: NewStream equal {eq_s} ({ns} differ), "
                  f"Out equal {eq_h} ({nh} differ)", flush=True)
print(f"B2BIT done, mismatching tensors {bad}")
