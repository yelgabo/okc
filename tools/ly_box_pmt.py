#!/usr/bin/env python3
"""Decode 1 post-MoE tail timeline inside the layer's CUDA graph (stamps of tools/dl_ly_pmt_patch.py).

python tools/ly_box_pmt.py muse_decode_layer/cand/ly_pmt.py --judge J.json --router-seed S [--inputs 20]
    [--loads "decode 1"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json] [--raw N]

Per cell two graphs of the whole layer are captured as bx_box_run.py captures them (PDL pinned on at every T, the
file's gate skipped): plain (every stamp switch off: the base's kernels) and instrumented (cgemv TL instantiation,
the MoE's fourth scratch tensor, the finish kernel's TLON). Each fresh input (local_start per input) is replayed
through both with a cold L2; graph times, the largest output difference and the per-CTA stamp summary are printed.

Times are us. Reference points: moe_trig = the last MoE CTA's launch_dependents (the slice may launch from here),
moe_exit = the last MoE CTA's exit (the slice's griddepcontrol.wait can return from here), slice_end = the last
slice CTA's store, fin_end = the last finish program's stores.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

CG_SLOTS, MOE_SLOTS, FIN_SLOTS = 16, 8, 8


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def summarize(cg, moe, fin):
    """cg [512, 16], moe [grid, 8], fin [programs, 8] int64 stamp rows (ns). Returns a dict of us values."""
    moe = [r for r in moe if r[0] > 0]
    cg = [r for r in cg if r[0] > 0]
    fin = [r for r in fin if r[0] > 0]
    t0 = min(r[0] for r in moe)
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    mean = statistics.mean
    med = statistics.median
    m3 = [us(r[3]) for r in moe]
    m4 = [us(r[4]) for r in moe]
    m5 = [us(r[5]) for r in moe]
    moe_trig, moe_bar, moe_exit = max(m3), max(m4), max(m5)
    fin_ctas = sorted(range(len(moe)), key=lambda i: -moe[i][5])[:8]   # the final-pass CTAs are the last to exit
    out = dict(
        moe_n=len(moe), moe_trig_min=min(m3), moe_trig=moe_trig, moe_bar_min=min(m4), moe_bar=moe_bar,
        moe_exit_min=min(m5), moe_exit_p50=med(m5), moe_exit_last8=mean(us(moe[i][5]) for i in fin_ctas), moe_exit=moe_exit,
        moe_bar_to_exit_last8=mean(us(moe[i][5]) - us(moe[i][4]) for i in fin_ctas),
    )
    # slice
    e = [us(r[0]) for r in cg]
    w1 = [us(r[1]) for r in cg]
    w2 = [us(r[2]) for r in cg]
    x3 = [us(r[3]) for r in cg]
    f4 = [us(r[4]) for r in cg]
    e5 = [us(r[5]) for r in cg]
    e6 = [us(r[6]) for r in cg]
    e7 = [us(r[7]) for r in cg]
    s8 = [us(r[8]) for r in cg]
    s9 = [us(r[9]) for r in cg]
    cnt = [int(r[11]) for r in cg]
    late = [i for i in range(len(cg)) if e[i] > moe_exit]
    crit = max(range(len(cg)), key=lambda i: s9[i])
    r = cg[crit]
    out.update(
        sl_n=len(cg), sl_entry_min=min(e), sl_entry_p50=med(e), sl_entry_max=max(e),
        sl_entry_after_trig=min(e) - moe_trig, sl_n_late=len(late),
        sl_wready_p50=med(w1), sl_wready_max=max(w1), sl_wready_before_exit=sum(1 for v in w1 if v < moe_exit),
        sl_wait_min=min(w2), sl_wait_p50=med(w2), sl_wait_p90=q(w2, 0.9), sl_wait_max=max(w2),
        sl_wait_lat_min=min(w2) - moe_exit, sl_wait_lat_p50=med(w2) - moe_exit, sl_wait_lat_max=max(w2) - moe_exit,
        sl_x_p50=med(b - a for a, b in zip(w2, x3)), sl_x_p90=q([b - a for a, b in zip(w2, x3)], 0.9), sl_x_max=max(b - a for a, b in zip(w2, x3)),
        sl_fma_p50=med(b - a for a, b in zip(x3, f4)), sl_fma_max=max(b - a for a, b in zip(x3, f4)),
        sl_esc_cnt_p50=med(cnt), sl_esc_cnt_max=max(cnt),
        sl_esc_ready_p50=med(e5), sl_esc_wait_lat_p50=med(e6) - moe_exit,
        sl_esc_walk_p50=med(b - a for a, b in zip(e6, e7)), sl_esc_walk_p90=q([b - a for a, b in zip(e6, e7)], 0.9),
        sl_esc_walk_max=max(b - a for a, b in zip(e6, e7)),
        sl_esc_after_fma_p50=med(b - a for a, b in zip(f4, e7)),   # positive: the escape warp is the CTA's laggard
        sl_sync_p50=med(c - max(a, b) for a, b, c in zip(f4, e7, s8)),
        sl_store_p50=med(b - a for a, b in zip(s8, s9)),
        sl_cta_wait_to_store_p50=med(b - a for a, b in zip(w2, s9)), sl_cta_wait_to_store_max=max(b - a for a, b in zip(w2, s9)),
        sl_end_p50=med(s9), sl_end_p90=q(s9, 0.9), sl_end=max(s9), sl_end_after_exit=max(s9) - moe_exit,
        crit_cta=crit, crit_entry=us(r[0]), crit_wready=us(r[1]), crit_wait=us(r[2]), crit_wait_lat=us(r[2]) - moe_exit,
        crit_x=us(r[3]) - us(r[2]), crit_fma=us(r[4]) - us(r[3]), crit_esc_walk=us(r[7]) - us(r[6]),
        crit_esc_after_fma=us(r[7]) - us(r[4]), crit_sync=us(r[8]) - max(us(r[4]), us(r[7])), crit_store=us(r[9]) - us(r[8]),
        crit_cnt=int(r[11]), crit_sm=int(r[10]),
    )
    # finish
    if fin:
        f0 = [us(r[0]) for r in fin]
        f1 = [us(r[1]) for r in fin]
        f2 = [us(r[2]) for r in fin]
        f3 = [us(r[3]) for r in fin]
        f4_ = [us(r[4]) for r in fin]
        sl_end = max(s9)
        out.update(
            fin_n=len(fin), fin_entry_min=min(f0), fin_entry_max=max(f0), fin_entry_after_sl_end=min(f0) - sl_end,
            fin_wait_min=min(f1) - sl_end, fin_wait_p50=med(f1) - sl_end, fin_wait_max=max(f1) - sl_end,
            fin_p_p50=med(b - a for a, b in zip(f1, f2)), fin_p_max=max(b - a for a, b in zip(f1, f2)),
            fin_carry_p50=med(b - a for a, b in zip(f2, f3)), fin_store_p50=med(b - a for a, b in zip(f3, f4_)),
            fin_end=max(f4_), fin_end_after_sl_end=max(f4_) - sl_end, fin_end_after_exit=max(f4_) - moe_exit,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 1")
    ap.add_argument("--layers", default="sliding,full")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
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
    grid = int(state["nsm"])
    e = (state.get("bfp", {}).get("cg") or state.get("cg") or {}).get("unpooler") or {}
    print(f"PMT slice cgemv n {e.get('n')} nth {e.get('nth')} pmax {e.get('pmax')} escapes {e.get('escapes')}", flush=True)
    cg_buf = torch.zeros(4096 * CG_SLOTS, dtype=torch.int64, device="cuda")
    moe_buf = torch.zeros(grid * MOE_SLOTS, dtype=torch.int64, device="cuda")
    fin_buf = torch.zeros(64 * 8 * FIN_SLOTS, dtype=torch.int64, device="cuda")

    def inst(on):
        mod._CG_EXT.cg_set_tlb(cg_buf.data_ptr() if on else 0)
        mod._PMT_MOE_DBG = moe_buf if on else None
        mod._PMT_FIN_TLB = fin_buf if on else None

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
        inst(False)
        gp, outp = br.capture(mod.kernel_fn, static, state)
        inst(True)
        gi, outi = br.capture(mod.kernel_fn, static, state)
        inst(False)
        res = []
        first = lambda o: o if torch.is_tensor(o) else next((x for x in o if torch.is_tensor(x)), None)  # noqa: E731
        for j, f in enumerate(fresh):
            br.copy_into(static, f)
            torch.cuda.synchronize()
            tp = timer.cold(gp) / 1e3
            op = first(outp).clone()
            cg_buf.zero_()
            moe_buf.zero_()
            fin_buf.zero_()
            torch.cuda.synchronize()
            ti = timer.cold(gi) / 1e3
            diff = float((first(outi).float() - op.float()).abs().max())
            cg = cg_buf.view(-1, CG_SLOTS).cpu().tolist()
            moe = moe_buf.view(-1, MOE_SLOTS).cpu().tolist()
            fin = fin_buf.view(-1, FIN_SLOTS).cpu().tolist()
            st = summarize(cg, moe, fin)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j])
            if j < a.raw:
                st["raw"] = dict(cg=[r for r in cg if r[0]], moe=[r for r in moe if r[0]], fin=[r for r in fin if r[0]])
            res.append(st)
            print(f"PMTI {case['load']} {case['layer']} in {j} plain {tp:.2f} inst {ti:.2f} diff {diff:.2e} "
                  f"exit {st['moe_exit']:.2f} wait_lat p50 {st['sl_wait_lat_p50']:.2f} max {st['sl_wait_lat_max']:.2f} "
                  f"x {st['sl_x_p50']:.2f} esc_walk {st['sl_esc_walk_p50']:.2f} sl_end+ {st['sl_end_after_exit']:.2f} "
                  f"fin_end+ {st.get('fin_end_after_sl_end', float('nan')):.2f} late {st['sl_n_late']}", flush=True)
        label = f"{case['load']} {case['layer']}"
        keys = [k for k in res[0] if k not in ("raw", "seed")]
        med = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"PMT {label} T {T} n {len(res)} medians", flush=True)
        for k in keys:
            print(f"  {k:28s} {med[k]:9.3f}", flush=True)
        rows.append(dict(case=case, T=T, inputs=res))
    if a.out:
        json.dump(dict(rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
