#!/usr/bin/env python3
"""Bitwise check of ly_prq's router quarter sum, at T = 1, 8, 16, 32 on random rows.

    flock /root/okc/.build/box/gpu.lock python tools/ly_prq_bitwise.py muse_decode_layer/cand/ly_prq.py

1. The fused pooler + router kernel run with a real outr (the counter path, as in ly_micro2) gives logits L and
   leaves the four quarter partials in fused_part; ((p0 + p1) + p2) + p3 in torch FP32 must equal L bitwise.
2. The same launch with an empty outr (ly_prq's path) must leave the same partials and the same pooled rows.
3. _sl_tile_route_cg on (pooled, L) and _prq_sl_tile_route_cg on (pooled, partials) must give bitwise-equal ids,
   local ids, weights and expert inputs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402


def main():
    path = sys.argv[1]
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
    g = torch.Generator(device=dev).manual_seed(7)
    bad = 0
    for T in (1, 8, 16, 32):
        h = torch.randn((T, 8192), generator=g, device=dev).to(torch.bfloat16)
        hr = torch.randn((T, 8192), generator=g, device=dev)
        ci, stream, tab, pmax, b4, _ = bst["sk2"]["pooler"]
        rstream = bst["sk2"]["router"][1]
        L = torch.empty((T, 256), device=dev, dtype=torch.float32)
        pooled = torch.empty((T, 4096), device=dev, dtype=torch.bfloat16)
        pooled2 = torch.empty_like(pooled)
        nolog = torch.empty((0,), device=dev, dtype=torch.float32)
        m.bfp_SK2_PDL = 0
        if T <= 8:
            ext = m._gle_ext()
            ext.sk2_set_pdl(0)
            run = lambda out, lg: ext.gle_pr(stream, rstream, h, out, lg, tab, pmax, b4, bst["fused_part"],  # noqa: E731
                                             bst["fused_cnt"], T, hr, state["router_fp32"], int(m.GLE_ILV))
            if "fused_part" not in bst:
                m._gle_frt(h, hr, state)
            part = bst["fused_part"]
        else:
            ext = m.bfp__get_ext()
            ext.sk2_set_pdl(0)
            if "fused_part" not in bst:
                m.bfp_fused_pooler_router(h, bst)
            run = lambda out, lg: ext.sk2f(stream, rstream, h, out, lg, tab, pmax, b4, bst["fused_part"],  # noqa: E731
                                           bst["fused_cnt"], T, m.bfp__sk2_tokr(T))
            part = bst["fused_part"]
        part.fill_(float("nan"))
        run(pooled, L)
        torch.cuda.synchronize()
        p1 = part.clone()
        qt = getattr(m, "_PRQ_LAYOUT", "tq") == "qt"   # ly_prq2: [quarter][token][256]; ly_prq: [tile][quarter][token][8]
        if qt:
            q = p1.view(4, 32, 256)
            qs = (((q[0] + q[1]) + q[2]) + q[3])[:T]
        else:
            q = p1.view(32, 4, 32, 8)
            qs = (((q[:, 0] + q[:, 1]) + q[:, 2]) + q[:, 3])[:, :T, :]            # [tile, t, row]
            qs = qs.permute(1, 0, 2).reshape(T, 256)
        e1 = int((qs.view(torch.int32) != L.view(torch.int32)).sum())
        part.fill_(float("nan"))
        run(pooled2, nolog)
        torch.cuda.synchronize()
        neq = part.view(torch.int32) != p1.view(torch.int32)
        e2 = int((neq.view(4, 32, 256)[:, :T] if qt else neq.view(32, 4, 32, 8)[:, :, :T]).sum())
        e3 = int((pooled2.view(torch.int16) != pooled.view(torch.int16)).sum())
        outs = []
        for kern, lg in ((m._sl_tile_route_cg.base, L), (m._prq_sl_tile_route_cg, part)):
            ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
            lids = torch.empty_like(ids)
            wts = torch.empty((T, 8), device=dev, dtype=torch.float32)
            xm = torch.empty((T, 4096), device=dev, dtype=torch.bfloat16)
            ls = torch.zeros((1,), device=dev, dtype=torch.int32)
            cfg = m.SL_TILE_CONFIG.get(T, (8, 1, True))
            rc = m.SL_ROUTE_CONFIG.get(T, (0, 0, False))
            kern[(T, cfg[1] + 1)](pooled, lg, lg, state["bias"], state["g_preexp"], ids, wts, xm, ls, lids, SPLIT=True,
                                  EARLY=rc[2], MODE=rc[1], PARTS=cfg[1], PRELOAD=cfg[2], num_warps=cfg[0], launch_pdl=False)
            torch.cuda.synchronize()
            outs.append((ids, lids, wts.view(torch.int32), xm.view(torch.int16)))
        e4 = [int((a != b).sum()) for a, b in zip(*outs)]
        ok = e1 == 0 and e2 == 0 and e3 == 0 and not any(e4)
        bad += not ok
        print(f"PRQBIT T {T}: logits vs quarter sum {e1} differing / {T * 256}; partials empty-outr vs counter {e2};"
              f" pooled {e3}; route ids / lids / wts / xm {e4} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    print("PRQBIT", "ALL OK" if not bad else f"{bad} T values mismatch")


if __name__ == "__main__":
    main()
