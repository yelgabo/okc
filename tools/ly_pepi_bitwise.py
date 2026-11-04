#!/usr/bin/env python3
"""Bitwise comparison of the fused pooler + FP32 router kernel (sk2_kernel_gle, T <= 8) between two layer files.

    flock /root/okc/.build/box/gpu.lock python tools/ly_pepi_bitwise.py dump muse_decode_layer/cand/ly_fabq.py A.pt
    flock /root/okc/.build/box/gpu.lock python tools/ly_pepi_bitwise.py dump muse_decode_layer/cand/ly_pepi.py B.pt
    python tools/ly_pepi_bitwise.py cmp A.pt B.pt

dump: for T in 1, 2, 3, 5, 8, seeded random BF16 rows h and FP32 rows x32, the file's own gle_pr launch with an empty
outr (the route-side quarter-sum path the layer runs) and once more with a real outr (the counter path, compiled but
not launched by the layer): pooled rows, the four quarter partials and the logits are saved. cmp: every tensor must be
bitwise equal between the two dumps.
"""
import os
import sys

import torch


def dump(path, out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bx_box_run as br
    br.content_addressed_extensions("/root/.cache/bx_ext")
    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = br.load_module("reference", os.path.join(pdir, "reference.py"))
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=106218864389449120)
    m = br.load_module("okc_submission", path)
    if hasattr(m, "_pdl8_setup"):
        m._pdl8_setup = lambda st: None
    state = m.prepare(weights)
    bst = state["bfp"]
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(11)
    m.bfp_SK2_PDL = 0
    ext = m._gle_ext()
    ext.sk2_set_pdl(0)
    stream, tab, pmax, b4 = bst["sk2"]["pooler"][1], bst["sk2"]["pooler"][2], bst["sk2"]["pooler"][3], bst["sk2"]["pooler"][4]
    rstream = bst["sk2"]["router"][1]
    res = {}
    for T in (1, 2, 3, 5, 8):
        h = torch.randn((T, 8192), generator=g, device=dev).to(torch.bfloat16)
        hr = torch.randn((T, 8192), generator=g, device=dev)
        if "fused_part" not in bst:
            m._gle_frt(h, hr, state)
        part = bst["fused_part"]
        for mode in ("empty", "counter"):
            pooled = torch.empty((T, 4096), device=dev, dtype=torch.bfloat16)
            L = torch.empty((T, 256) if mode == "counter" else (0,), device=dev, dtype=torch.float32)
            part.fill_(float("nan"))
            bst["fused_cnt"].zero_()
            ext.gle_pr(stream, rstream, h, pooled, L, tab, pmax, b4, part, bst["fused_cnt"], T, hr, state["router_fp32"], int(m.GLE_ILV))
            torch.cuda.synchronize()
            res[(T, mode)] = dict(pooled=pooled.cpu(), part=part.view(4, 32, 256)[:, :T].clone().cpu(), logits=L.cpu())
            print(f"DUMP T {T} {mode}: pooled sum {pooled.float().sum().item():.6e} part nan {int(torch.isnan(part.view(4, 32, 256)[:, :T]).sum())}", flush=True)
    torch.save(res, out)
    print("DUMP saved", out)


def cmp(a, b):
    A, B = torch.load(a), torch.load(b)
    bad = 0
    for k in A:
        for name in ("pooled", "part", "logits"):
            x, y = A[k][name], B[k][name]
            n = int((x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32) != y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)).sum())
            bad += n
            print(f"PEPIBIT T {k[0]} {k[1]} {name}: {n} differing / {x.numel()}")
    print("PEPIBIT", "ALL BITWISE EQUAL" if bad == 0 else f"{bad} differing values")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3])
    else:
        cmp(sys.argv[2], sys.argv[3])
