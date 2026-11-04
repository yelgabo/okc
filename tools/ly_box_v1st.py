#!/usr/bin/env python3
"""Verify 1 (T = 8) slice and finish timeline inside the layer's CUDA graph (stamps of tools/dl_ly_v1st_patch.py).

python tools/ly_box_v1st.py muse_decode_layer/cand/ly_v1st.py --judge J.json --router-seed S [--inputs 20]
    [--loads "verify 1"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json] [--raw N]

Per cell two graphs of the whole layer, captured as bx_box_run.py captures them (PDL pinned on at every T, the
file's gate skipped): plain (every stamp switch off) and instrumented (sk2_set_tlu, the MoE's fourth scratch tensor,
the finish kernel's TLON). Each fresh input (local_start per input) is replayed through both with a cold L2.

Times are us from the first MoE CTA's entry. moe_exit = the last MoE CTA's exit (the slice's griddepcontrol.wait
returns only after it); slice_end = the last slice CTA's output store; fin_end = the last finish program's stores.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

SL_SLOTS, MOE_SLOTS, FIN_SLOTS = 32, 8, 8


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def summarize(sl, moe, fin):
    moe = [r for r in moe if r[0] > 0]
    sl = [r for r in sl if r[0] > 0]
    fin = [r for r in fin if r[0] > 0]
    t0 = min(r[0] for r in moe)
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    med = statistics.median
    m3 = [us(r[3]) for r in moe]
    m5 = [us(r[5]) for r in moe]
    moe_trig, moe_exit = max(m3), max(m5)
    out = dict(moe_n=len(moe), moe_trig_min=min(m3), moe_trig=moe_trig, moe_exit_min=min(m5), moe_exit_p50=med(m5),
               moe_exit=moe_exit)
    col = lambda k: [us(r[k]) for r in sl]  # noqa: E731
    ent, pro, wt, xi, prod = col(0), col(1), col(2), col(3), col(4)
    tab = col(5)
    w0 = [col(6 + s) for s in range(4)]
    w7 = [col(10 + s) for s in range(4)]
    e0 = [col(26 + s) for s in range(4)]
    c0, c7 = col(14), col(15)
    esc = [col(16 + e) for e in range(4)]
    ep20, ep21, ep22, st23 = col(20), col(21), col(22), col(23)
    n = len(sl)
    esc_done = [max(esc[e][i] for e in range(4)) for i in range(n)]
    cons_done = [max(c0[i], c7[i]) for i in range(n)]
    zero = us(0)
    rel = lambda a, b: [y - x for x, y in zip(a, b) if x != zero and y != zero] or [float("nan")]  # noqa: E731
    # CL > 1 builds: ranks > 0 return before slots 21 to 23 and skip slots 1 and 3; the slice ends at rank 0's store
    st23 = [v if v != zero else ep20[i] for i, v in enumerate(st23)]
    crit = max(range(n), key=lambda i: st23[i])
    r = sl[crit]
    out.update(
        sl_n=n, sl_entry_min=min(ent), sl_entry_p50=med(ent), sl_entry_max=max(ent),
        sl_entry_before_exit=sum(1 for v in ent if v < moe_exit), sl_entry_after_trig=min(ent) - moe_trig,
        sl_prologue_p50=med(rel(ent, pro)), sl_prologue_max=max(rel(ent, pro)),
        sl_wait_lat_min=min(wt) - moe_exit, sl_wait_lat_p50=med(wt) - moe_exit, sl_wait_lat_max=max(wt) - moe_exit,
        sl_xissue_p50=med(rel(wt, xi)), sl_prod_issue_p50=med(rel(wt, prod)),
        sl_tab_before_wait=sum(1 for a, b in zip(tab, wt) if a < b),
        sl_esc_cnt_p50=med(int(r_[25]) for r_ in sl), sl_esc_cnt_max=max(int(r_[25]) for r_ in sl),
    )
    for s in range(4):
        out[f"sl_w0_st{s}_after_wait_p50"] = med(rel(wt, w0[s]))
        out[f"sl_w0_st{s}_after_wait_max"] = max(rel(wt, w0[s]))
        out[f"sl_w7_st{s}_after_wait_p50"] = med(rel(wt, w7[s]))
        out[f"sl_e0_st{s}_after_wait_p50"] = med(rel(wt, e0[s]))
        out[f"sl_st{s}_landed_before_wait"] = sum(1 for a, b in zip(w0[s], wt) if a < b)
    out.update(
        sl_cons_after_last_stage_p50=med(rel([max(w0[3][i], w7[3][i]) for i in range(n)], cons_done)),
        sl_esc_after_cons_p50=med(rel(cons_done, esc_done)), sl_esc_after_cons_max=max(rel(cons_done, esc_done)),
        sl_cbar_after_cons_p50=med(rel(cons_done, ep20)),
        sl_red_p50=med(rel(ep20, ep22)), sl_store_p50=med(rel(ep22, st23)),
        sl_wait_to_store_p50=med(rel(wt, st23)), sl_wait_to_store_max=max(rel(wt, st23)),
        sl_end_p50=med(st23), sl_end=max(st23), sl_end_after_exit=max(st23) - moe_exit,
        sl_end_spread=max(st23) - med(st23),
        crit_cta=crit, crit_sm=int(r[24]), crit_entry=us(r[0]), crit_wait_lat=us(r[2]) - moe_exit,
        crit_st0=us(r[6]) - us(r[2]), crit_st1=us(r[7]) - us(r[2]), crit_st2=us(r[8]) - us(r[2]), crit_st3=us(r[9]) - us(r[2]),
        crit_cons_done=max(us(r[14]), us(r[15])) - us(r[2]), crit_esc_done=max(us(r[16 + e]) for e in range(4)) - us(r[2]),
        crit_cbar=us(r[20]) - us(r[2]), crit_store=us(r[23]) - us(r[2]),
    )
    if fin:
        f0, f1, f2, f3, f4 = ([us(r_[k]) for r_ in fin] for k in range(5))
        sl_end = max(st23)
        out.update(
            fin_n=len(fin), fin_entry_min=min(f0) - sl_end, fin_entry_max=max(f0) - sl_end,
            fin_wait_min=min(f1) - sl_end, fin_wait_p50=med(f1) - sl_end, fin_wait_max=max(f1) - sl_end,
            fin_p_p50=med(rel(f1, f2)), fin_p_max=max(rel(f1, f2)),
            fin_carry_p50=med(rel(f2, f3)), fin_carry_max=max(rel(f2, f3)),
            fin_store_p50=med(rel(f3, f4)), fin_store_max=max(rel(f3, f4)),
            fin_end=max(f4), fin_end_after_sl_end=max(f4) - sl_end, fin_end_after_exit=max(f4) - moe_exit,
            fin_end_spread=max(f4) - med(f4),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="verify 1")
    ap.add_argument("--layers", default="sliding,full")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", type=int, default=0)
    ap.add_argument("--dg", type=int, default=0, help="sk2 diagnostic bits for the stamped slice (a --diag instrument)")
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
    sl_buf = torch.zeros(1024 * SL_SLOTS, dtype=torch.int64, device="cuda")
    moe_buf = torch.zeros(4 * grid * MOE_SLOTS, dtype=torch.int64, device="cuda")
    fin_buf = torch.zeros(64 * 8 * FIN_SLOTS, dtype=torch.int64, device="cuda")
    ext = mod.bfp__get_ext()

    if a.dg:
        ext.sk2_set_tdg(a.dg)

    def inst(on):
        ext.sk2_set_tlu(sl_buf.data_ptr() if on else 0)
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
            sl_buf.zero_()
            moe_buf.zero_()
            fin_buf.zero_()
            torch.cuda.synchronize()
            ti = timer.cold(gi) / 1e3
            diff = float((first(outi).float() - op.float()).abs().max())
            sl = sl_buf.view(-1, SL_SLOTS).cpu().tolist()
            moe = moe_buf.view(-1, MOE_SLOTS).cpu().tolist()
            fin = fin_buf.view(-1, FIN_SLOTS).cpu().tolist()
            st = summarize(sl, moe, fin)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j])
            if j < a.raw:
                st["raw"] = dict(sl=[r for r in sl if r[0]], moe=[r for r in moe if r[0]], fin=[r for r in fin if r[0]])
            res.append(st)
            print(f"V1STI {case['load']} {case['layer']} in {j} plain {tp:.2f} inst {ti:.2f} diff {diff:.2e} "
                  f"n {st['sl_n']} exit {st['moe_exit']:.2f} wait_lat {st['sl_wait_lat_p50']:.2f}/{st['sl_wait_lat_max']:.2f} "
                  f"st0..3 {st['sl_w0_st0_after_wait_p50']:.2f} {st['sl_w0_st1_after_wait_p50']:.2f} "
                  f"{st['sl_w0_st2_after_wait_p50']:.2f} {st['sl_w0_st3_after_wait_p50']:.2f} "
                  f"w2s {st['sl_wait_to_store_p50']:.2f}/{st['sl_wait_to_store_max']:.2f} sl_end+ {st['sl_end_after_exit']:.2f} "
                  f"fin_end+ {st.get('fin_end_after_sl_end', float('nan')):.2f}", flush=True)
        label = f"{case['load']} {case['layer']}"
        keys = [k for k in res[0] if k not in ("raw", "seed")]
        md = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"V1ST {label} T {T} n {len(res)} medians", flush=True)
        for k in keys:
            print(f"  {k:30s} {md[k]:9.3f}", flush=True)
        rows.append(dict(case=case, T=T, inputs=res))
    if a.out:
        json.dump(dict(rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
