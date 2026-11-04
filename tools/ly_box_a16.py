#!/usr/bin/env python3
"""Decode 16 / 32 sliding attention timeline inside the layer's CUDA graph (stamps of tools/dl_ly_a16_patch.py).

python tools/ly_box_a16.py muse_decode_layer/cand/ly_a16.py --judge J.json --router-seed S [--inputs 20]
    [--loads "decode 16,decode 32"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json]

Same capture and replay as tools/ly_box_ast.py (plain and instrumented graph per cell, each fresh input through
both with a cold L2), but the attention grid is (n_splits, B): each request has its own workspace region, ticket
and last CTA. Times are us after qkv_end, the last qkv CTA's end stamp.

Per input: per-slot p50 / max over all CTAs; for each request the latest CTA per slot; the critical request (the
one whose finalize stores issue last) and its parts.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

AS, KS = 32, 32
NAMES = {0: "entry", 1: "idx", 15: "wait", 8: "kv_iss", 16: "q_vis", 17: "kv_own", 9: "rope", 10: "qpass",
         11: "newkv", 2: "patch", 12: "st0", 13: "m0", 22: "st1", 25: "m1", 23: "st2", 26: "m2", 24: "st3", 27: "m3",
         3: "loop", 19: "xchg", 20: "reds", 21: "presync", 4: "ticket", 14: "fin0", 5: "fin1"}


def summarize(att_all, qkv, o, B):
    qkv = [r for r in qkv if r[0] > 0]
    o = [r for r in o if r[0] > 0]
    qend = max(r[23] for r in qkv)
    us = lambda x: (x - qend) / 1e3  # noqa: E731
    n_all = sum(1 for r in att_all if r[0] > 0)
    ns = n_all // B
    out = dict(qkv_entry=us(min(r[0] for r in qkv)), qkv_end_min=us(min(r[23] for r in qkv)), att_n=n_all, ns=ns)
    att = [r for r in att_all[:n_all] if r[0] > 0]
    for k, nm in NAMES.items():
        v = [us(r[k]) for r in att if r[k] > 0]
        if v:
            out[f"a_{nm}_min"], out[f"a_{nm}_p50"], out[f"a_{nm}_max"] = min(v), statistics.median(v), max(v)
    reqs = []
    for b in range(B):
        rows = att_all[b * ns:(b + 1) * ns]
        last = [r for r in rows if r[6] == 1] or sorted((r for r in rows if r[5] > 0), key=lambda r: -r[5])[:1]   # cluster mode: every CTA finalizes a slice
        if not last:
            continue
        d = {nm: max(us(r[k]) for r in rows if r[k] > 0) for k, nm in NAMES.items() if any(r[k] > 0 for r in rows)}
        d["fin0"], d["fin1"] = us(last[0][14]), us(last[0][5])
        d["last_loop"] = us(last[0][3])
        d["sms"] = sorted({int(r[18]) for r in rows})
        reqs.append(d)
    if reqs:
        crit = max(reqs, key=lambda d: d["fin1"])
        for nm, v in crit.items():
            if nm != "sms":
                out[f"C_{nm}"] = v
        out["att_end"] = crit["fin1"]
        for nm in ("loop", "ticket", "fin1"):
            out[f"R_{nm}_p50"] = statistics.median(d[nm] for d in reqs)
            out[f"R_{nm}_min"] = min(d[nm] for d in reqs)
        out["R_n"] = len(reqs)
        # SM sharing: CTAs per SM over the grid
        from collections import Counter
        c = Counter(int(r[18]) for r in att)
        out["sm_used"] = len(c)
        out["sm_max_ctas"] = max(c.values())
    if o:
        out["o_entry_min"] = us(min(r[0] for r in o))
        w = [us(r[2]) for r in o if r[2] > 0]
        if w:
            out["o_wait_min"], out["o_wait_p50"], out["o_wait_max"] = min(w), statistics.median(w), max(w)
        out["o_end"] = us(max(r[23] for r in o))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 16,decode 32")
    ap.add_argument("--layers", default="sliding")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", type=int, default=2)
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
    att_buf = torch.zeros(4096 * AS, dtype=torch.int64, device="cuda")
    qkv_buf = torch.zeros(1024 * KS, dtype=torch.int64, device="cuda")
    o_buf = torch.zeros(1024 * KS, dtype=torch.int64, device="cuda")
    ext = mod.bfp__get_ext()
    nq = int(state["qkv"].shape[0])

    def inst(on):
        ext.sk2_set_ast(0, nq, mod.HIDDEN, qkv_buf.data_ptr() if on else 0)
        ext.sk2_set_ast(1, mod.HIDDEN, mod.Q_DIM, o_buf.data_ptr() if on else 0)
        mod._AST_DBG = att_buf if on else None

    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    timer = br.Timer()
    rows = []
    for case in cases:
        B = case["batch"]
        T = B * case.get("q_len", 1)
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
            for bf in (att_buf, qkv_buf, o_buf):
                bf.zero_()
            br.copy_into(static, f)
            torch.cuda.synchronize()
            ti = timer.cold(gi) / 1e3
            diff = float((first(outi).float() - op.float()).abs().max())
            att = att_buf.view(-1, AS).cpu().tolist()
            qk = qkv_buf.view(-1, KS).cpu().tolist()
            oo = o_buf.view(-1, KS).cpu().tolist()
            st = summarize(att, qk, oo, B)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j])
            if j < a.raw:
                st["raw"] = dict(att=att[:st["att_n"]], qkv=[r for r in qk if r[0]], o=[r for r in oo if r[0]])
            res.append(st)
            g = lambda k: st.get(k, float("nan"))  # noqa: E731
            print(f"A16I {case['load']} {case['layer']} in {j} plain {tp:.2f} inst {ti:.2f} diff {diff:.2e} n {st['att_n']} "
                  f"ns {st['ns']} wait {g('a_wait_min'):.2f}/{g('a_wait_max'):.2f} q {g('a_q_vis_p50'):.2f} "
                  f"st0 {g('a_st0_p50'):.2f} loop {g('a_loop_p50'):.2f}/{g('a_loop_max'):.2f} "
                  f"ticket {g('a_ticket_p50'):.2f}/{g('a_ticket_max'):.2f} end {g('att_end'):.2f} "
                  f"o_wait {g('o_wait_min'):.2f} o_end {g('o_end'):.2f}", flush=True)
        label = f"{case['load']} {case['layer']}"
        keys = sorted({k for r in res for k in r if k not in ("raw", "seed")})
        md = {k: statistics.median([r[k] for r in res if k in r]) for k in keys}
        print(f"A16 {label} T {T} n {len(res)} medians", flush=True)
        for k in keys:
            print(f"  {k:24s} {md[k]:9.3f}", flush=True)
        rows.append(dict(case=case, T=T, inputs=res))
    if a.out:
        json.dump(dict(rows=rows), open(a.out, "w"))
    if a.lock:
        br.LOCK.release("done")


if __name__ == "__main__":
    main()
