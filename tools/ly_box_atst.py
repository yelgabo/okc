#!/usr/bin/env python3
"""Verify 1 (T = 8) attention timeline inside the layer's CUDA graph (stamps of tools/dl_ly_atst_patch.py).

python tools/ly_box_atst.py muse_decode_layer/cand/ly_atst.py --judge J.json --router-seed S [--inputs 20]
    [--loads "verify 1"] [--layers sliding,full] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json]

Per cell two graphs of the whole layer, captured as bx_box_run.py captures them (PDL pinned on at every T, the
file's gate skipped): plain (stamp tensors unset) and instrumented. Each fresh input (local_start per input) is
replayed through both with a cold L2. Times are us from qkv_end, the last qkv CTA's output store.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

AS = 32          # attention slots per CTA (DBGSTRIDE)
KS = 32          # sk2 slots per CTA
NAMES = {0: "entry", 20: "prewait_done", 16: "wait_ret", 1: "idx_landed", 8: "kv_issued", 17: "q_landed",
         9: "rope", 10: "qpass", 11: "newkv", 2: "patched", 12: "tile_landed", 13: "tile_done", 3: "merge_start",
         7: "reds_issued", 6: "reds_bar", 18: "ticket_ret", 4: "gbar_passed", 5: "fin_end",
         21: "qfence_pre", 22: "pfence_pre"}


def summarize(att, qkv, o):
    att = [r for r in att if r[0] > 0]
    qkv = [r for r in qkv if r[0] > 0]
    o = [r for r in o if r[0] > 0]
    med = statistics.median
    t0 = max(r[23] for r in qkv)
    us = lambda x: (x - t0) / 1e3 if x > 0 else float("nan")  # noqa: E731
    out = dict(n_att=len(att), n_qkv=len(qkv), n_o=len(o),
               qkv_end_p50=med(us(r[23]) for r in qkv), qkv_entry_min=min(us(r[0]) for r in qkv))
    for k, nm in NAMES.items():
        v = [us(r[k]) for r in att if r[k] > 0]
        if not v:
            continue
        out[f"a_{nm}_min"] = min(v)
        out[f"a_{nm}_p50"] = med(v)
        out[f"a_{nm}_max"] = max(v)
    att_end = max(us(r[5]) for r in att if r[5] > 0)
    out["att_end"] = att_end
    crit = max(att, key=lambda r: r[18] if r[18] > 0 else r[4])
    for k, nm in NAMES.items():
        out[f"c_{nm}"] = us(crit[k])
    out["crit_sm"] = int(crit[19])
    out["n_kv_issued_before_wait"] = sum(1 for r in att if 0 < r[8] < r[16])
    ow = [us(r[2]) for r in o if r[2] > 0]
    if ow:
        out.update(o_entry_min=min(us(r[0]) for r in o), o_wait_min=min(ow), o_wait_p50=med(ow), o_wait_max=max(ow),
                   o_wait_after_att_end=min(ow) - att_end, o_end=max(us(r[23]) for r in o))
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
    att_buf = torch.zeros(1024 * AS, dtype=torch.int64, device="cuda")
    sk_buf = torch.zeros(2048 * KS, dtype=torch.int64, device="cuda")
    exts = [mod.bfp__get_ext(), mod._de_ext()]

    def inst(on):
        for e in exts:
            e.sk2_set_tlu(sk_buf.data_ptr() if on else 0)
        mod._ATST_DBG = att_buf if on else None

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
            att_buf.zero_()
            sk_buf.zero_()
            torch.cuda.synchronize()
            ti = timer.cold(gi) / 1e3
            diff = float((first(outi).float() - op.float()).abs().max())
            att = att_buf.view(-1, AS).cpu().tolist()
            sk = sk_buf.view(-1, KS).cpu().tolist()
            st = summarize(att, sk[:1024], sk[1024:])
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j])
            if j < a.raw:
                st["raw"] = dict(att=[r for r in att if r[0]], qkv=[r for r in sk[:1024] if r[0]],
                                 o=[r for r in sk[1024:] if r[0]])
            res.append(st)
            print(f"ATSTI {case['load']} {case['layer']} in {j} plain {tp:.2f} inst {ti:.2f} diff {diff:.2e} "
                  f"n {st['n_att']} wait {st.get('a_wait_ret_min', float('nan')):.2f}/{st.get('a_wait_ret_max', float('nan')):.2f} "
                  f"kv_iss {st.get('a_kv_issued_p50', float('nan')):.2f} tile {st.get('a_tile_landed_p50', float('nan')):.2f} "
                  f"merge {st.get('a_merge_start_p50', float('nan')):.2f} ticket {st.get('a_ticket_ret_max', float('nan')):.2f} "
                  f"gbar {st.get('a_gbar_passed_max', float('nan')):.2f} end {st['att_end']:.2f} "
                  f"o_wait {st.get('o_wait_min', float('nan')):.2f}", flush=True)
        label = f"{case['load']} {case['layer']}"
        keys = [k for k in res[0] if k not in ("raw", "seed")]
        md = {k: statistics.median(r[k] for r in res if k in r) for k in keys}
        print(f"ATST {label} T {T} n {len(res)} medians", flush=True)
        for k in keys:
            print(f"  {k:30s} {md[k]:9.3f}", flush=True)
        rows.append(dict(case=case, T=T, inputs=res))
    if a.out:
        json.dump(dict(rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
