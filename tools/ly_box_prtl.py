#!/usr/bin/env python3
"""Pooler + router timeline inside the layer's CUDA graph (stamps of tools/dl_ly_prtl_patch.py).

python tools/ly_box_prtl.py muse_decode_layer/cand/ly_prtl.py --judge J.json --router-seed S [--inputs 5]
    [--loads "decode 1,verify 1,decode 16,decode 32"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F]
    [--out f.json] [--raw N]

Per cell two graphs of the whole layer are captured as bx_box_run.py captures them (PDL on at T = 1, 8, 16, 32 with
all six T = 8 edges, the file's gate skipped): plain (sk2_set_tlb(0)) and instrumented (the fused pooler + router
launch gets the stamp buffer as dbg). Each fresh input (local_start per input) is replayed through both with a cold
L2; graph times and the largest output difference between the two are printed with the per-CTA stamps' summary.

Times are us from the first CTA's entry. Floor: the bytes the kernel reads from HBM (pooler 12-bit stream, router
weights, escape tables, activations) at --bw TB/s (4.443, the B4 section's measured read rate).
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

SLOTS = 64
POOL_STAGE = 24576   # 8 warps x 4 n8 tiles x 768 B (12-bit codes of 32 rows x 512 k)
ROUTER_STAGE = 16384  # 8 router rows x 512 k x 4 B (FP32 at T <= 8, BF16 hi + lo at T >= 16)


def nb(t):
    return t.numel() * t.element_size()


def stats(s, T, P, bw):
    v = s.tolist()
    n = len(v)
    t0 = min(r[0] for r in v)
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    col = lambda k: [us(r[k]) for r in v if r[k] > 0]  # noqa: E731
    mean = statistics.mean
    entry, wait = col(0), col(3)
    loop_end = [max(us(r[56 + w]) for w in range(8)) for r in v]
    loop_first = [min(us(r[56 + w]) for w in range(8)) for r in v]
    land0 = col(36)
    rpart, ratom, rout = col(5), col(6), col(7)
    epi0, epi1, fin = col(8), col(9), col(10)
    # producer issue interval of stage s (s >= NSTAGE waits for the release of stage s - NSTAGE)
    iv_pool, iv_rt, land_iv = [], [], []
    ilv = T <= 8
    for r in v:
        for s_ in range(P + 1, 20):
            a, b = r[16 + s_ - 1], r[16 + s_]
            if a and b:
                rt = (s_ % 5 == 4) if ilv else s_ >= 16
                (iv_rt if rt else iv_pool).append((b - a) / 1e3)
        for s_ in range(1, 20):
            a, b = r[36 + s_ - 1], r[36 + s_]
            if a and b:
                land_iv.append((b - a) / 1e3)
    late = sorted(entry)[-8:]
    out = dict(
        grid=n, entry_spread=max(entry) - min(entry), entry_late8=mean(late), n_late1us=sum(e > 1.0 for e in entry),
        first_issue=mean(col(1)), wait_min=min(wait), wait_mean=mean(wait), wait_max=max(wait),
        land0=mean(land0), land0_after_wait=mean([us(r[36]) - us(r[3]) for r in v]),
        loop_end=mean(loop_end), loop_end_min=min(loop_end), loop_end_max=max(loop_end),
        warp_spread=mean([b - a for a, b in zip(loop_first, loop_end)]),
        router_part=mean([a - b for a, b in zip(rpart, loop_end)]), router_atom=mean([a - b for a, b in zip(ratom, rpart)]),
        router_out=mean([r_ - a for r_, a in zip(rout, ratom[: len(rout)])]) if rout else math.nan, n_rlast=len(rout),
        epi_start=mean([a - b for a, b in zip(epi0, ratom)]), epi_red=mean([a - b for a, b in zip(epi1, epi0)]),
        epi_store=mean([a - b for a, b in zip(fin, epi1)]), fin=mean(fin), fin_min=min(fin), fin_max=max(fin),
        span=max(fin), post_wait=max(fin) - min(wait), loop_from_wait=mean([le - w for le, w in zip(loop_end, wait)]),
        iv_pool=statistics.median(iv_pool) if iv_pool else math.nan,
        iv_rt=statistics.median(iv_rt) if iv_rt else math.nan,
        land_iv=statistics.median(land_iv) if land_iv else math.nan,
        stage_floor=n * POOL_STAGE / bw / 1e6, rstage_floor=n * ROUTER_STAGE / bw / 1e6,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 1,verify 1,decode 16,decode 32")
    ap.add_argument("--layers", default="sliding,full")
    ap.add_argument("--inputs", type=int, default=5)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--bw", type=float, default=4.443)
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", type=int, default=0)
    a = ap.parse_args()
    if a.lock:
        br.release_lock_for_triton()
        br.BudgetClock.install()
        br.LOCK = br.GpuLock(a.lock)
        br.LOCK.acquire("start")
    br.content_addressed_extensions(a.ext_cache)
    pdir = os.path.abspath("muse_decode_layer")
    sys.path.insert(0, pdir)
    ref = br.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    _, cases = br.read_judge(a.judge)
    keep = {x.strip() for x in a.loads.split(",")}
    lay = {x.strip() for x in a.layers.split(",")}
    cases = [c for c in cases if c.get("load") in keep and c.get("layer") in lay]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    state["pdl8_on"] = {T: True for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
    state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)}
    exts = [mod._gle_ext(), mod.bfp__get_ext()]
    bst = state["bfp"]
    ci, pstream, ptab, pmax, b4, nrows = bst["sk2"]["pooler"]
    rstream = bst["sk2"]["router"][1]
    w_bytes = dict(pool=nb(pstream), ptab=nb(ptab), router_hl=nb(rstream), router_fp32=nb(state["router_fp32"]))
    print("PRTL bytes", w_bytes, "pmax", pmax, flush=True)
    buf = torch.zeros(160 * SLOTS, dtype=torch.int64, device="cuda")

    def tlb(on):
        for e in exts:
            e.sk2_set_tlb(buf.data_ptr() if on else 0)

    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    timer = br.Timer()
    rows = []
    for case in cases:
        T = case["batch"] * case.get("q_len", 1)
        seeds = [case["seed"] + j for j in range(a.inputs)]
        fresh = []
        for sd in seeds:
            extra = lsc.get(ref, case, sd, weights) if lsc else ref.case_fields({**case, "seed": sd}, weights)
            fresh.append(ref.generate_input(**{**case, "seed": sd, **extra}))
        if lsc:
            lsc.save()
        static = ref.generate_input(**{**case, "seed": seeds[0]})
        tlb(False)
        gp, outp = br.capture(mod.kernel_fn, static, state)
        tlb(True)
        gi, outi = br.capture(mod.kernel_fn, static, state)
        tlb(False)
        act = T * 8192 * 2 + (T * 8192 * 4 if T <= 8 else 0)
        rbytes = w_bytes["router_fp32"] if T <= 8 else w_bytes["router_hl"]
        tot = w_bytes["pool"] + w_bytes["ptab"] + rbytes + act
        floor = tot / a.bw / 1e6
        P = 1 if T == 32 else 2
        res = []
        for j, f in enumerate(fresh):
            br.copy_into(static, f)
            torch.cuda.synchronize()
            tp = timer.cold(gp) / 1e3
            op = outp.clone() if torch.is_tensor(outp) else None
            buf.zero_()
            torch.cuda.synchronize()
            ti = timer.cold(gi) / 1e3
            diff = float((outi.float() - op.float()).abs().max()) if op is not None else math.nan
            nct = int((buf.view(-1, SLOTS)[:, 0] != 0).sum())
            s = buf[: nct * SLOTS].view(nct, SLOTS).cpu()
            st = stats(s, T, P, a.bw)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j], floor=floor, bytes=tot)
            if j < a.raw:
                st["raw"] = s.tolist()
            res.append(st)
        label = f"{case['load']} {case['layer']}"
        keys = [k for k in res[0] if k not in ("raw", "seed")]
        med = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"PRTL {label} T {T} n {len(res)} medians " + " ".join(f"{k} {v:.3f}" for k, v in med.items()), flush=True)
        rows.append(dict(case=case, T=T, inputs=res))
    if a.out:
        json.dump(dict(bw=a.bw, bytes=w_bytes, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
