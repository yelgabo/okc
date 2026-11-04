#!/usr/bin/env python3
"""Decode 1 attention timeline inside the layer's CUDA graph (stamps of tools/dl_ly_ast_patch.py).

python tools/ly_box_ast.py muse_decode_layer/cand/ly_ast.py --judge J.json --router-seed S [--inputs 20]
    [--loads "decode 1"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json]

Per cell two graphs of the whole layer, captured as bx_box_run.py captures them (PDL pinned on at every T, the
file's gate skipped): plain (dbg null everywhere) and instrumented (attention dbg tensor, sk2 stamps for the T = 1
qkv and o launches). Each fresh input is replayed through both with a cold L2.

All times are us after qkv_end = the last qkv CTA's end stamp (its output stores issued).
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
ATT_NAMES = {0: "entry", 1: "idx", 8: "kv_iss", 15: "wait", 16: "q_vis", 17: "kv_own", 9: "rope", 10: "qpass",
             11: "newkv", 2: "patch", 12: "stage", 13: "math0", 3: "loop", 19: "xchg", 20: "reds", 21: "presync",
             4: "ticket", 14: "fin0", 5: "fin1"}


def summarize(att, qkv, o):
    qkv = [r for r in qkv if r[0] > 0]
    o = [r for r in o if r[0] > 0]
    att = [r for r in att if r[0] > 0]
    qend = max(r[23] for r in qkv)
    us = lambda x: (x - qend) / 1e3 if x > 0 else float("nan")  # noqa: E731
    out = dict(qkv_n=len(qkv), qkv_entry=us(min(r[0] for r in qkv)), qkv_end_min=us(min(r[23] for r in qkv)),
               att_n=len(att))
    for k, nm in ATT_NAMES.items():
        v = [us(r[k]) for r in att if r[k] > 0]
        if v:
            out[f"a_{nm}_min"] = min(v)
            out[f"a_{nm}_p50"] = statistics.median(v)
            out[f"a_{nm}_max"] = max(v)
    last = [r for r in att if r[6] == 1]
    if last:
        r = last[0]
        out["last_n"] = len(last)
        for k, nm in ATT_NAMES.items():
            out[f"L_{nm}"] = us(r[k])
        out["last_sm"] = int(r[18])
        out["att_end"] = us(r[5])
    if o:
        out["o_n"] = len(o)
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
    ap.add_argument("--loads", default="decode 1")
    ap.add_argument("--layers", default="sliding,full")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", type=int, default=0)
    ap.add_argument("--warm", action="store_true", help="stamp a second back-to-back replay (no L2 flush before it)")
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
            for b in (att_buf, qkv_buf, o_buf):
                b.zero_()
            br.copy_into(static, f)
            torch.cuda.synchronize()
            if a.warm:
                gi.replay()
                torch.cuda.synchronize()
                for b in (att_buf, qkv_buf, o_buf):
                    b.zero_()
                br.copy_into(static, f)
                torch.cuda.synchronize()
                timer.a.record()
                gi.replay()
                timer.b.record()
                timer.b.synchronize()
                ti = timer.a.elapsed_time(timer.b) * 1e3
            else:
                ti = timer.cold(gi) / 1e3
            diff = float((first(outi).float() - op.float()).abs().max())
            att = att_buf.view(-1, AS).cpu().tolist()
            qk = qkv_buf.view(-1, KS).cpu().tolist()
            oo = o_buf.view(-1, KS).cpu().tolist()
            st = summarize(att, qk, oo)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j])
            if j < a.raw:
                st["raw"] = dict(att=[r for r in att if r[0]], qkv=[r for r in qk if r[0]], o=[r for r in oo if r[0]])
            res.append(st)
            g = lambda k: st.get(k, float("nan"))  # noqa: E731
            print(f"ASTI {case['load']} {case['layer']} in {j} plain {tp:.2f} inst {ti:.2f} diff {diff:.2e} "
                  f"n {st['att_n']} entry {g('a_entry_min'):.2f}/{g('a_entry_max'):.2f} wait {g('a_wait_min'):.2f}/{g('a_wait_max'):.2f} "
                  f"stage {g('a_stage_p50'):.2f}/{g('a_stage_max'):.2f} loop {g('a_loop_max'):.2f} ticket {g('a_ticket_max'):.2f} "
                  f"fin {g('L_fin0'):.2f}..{g('L_fin1'):.2f} o_wait {g('o_wait_min'):.2f}/{g('o_wait_max'):.2f}", flush=True)
        label = f"{case['load']} {case['layer']}"
        keys = sorted({k for r in res for k in r if k not in ("raw", "seed")})
        md = {k: statistics.median([r[k] for r in res if k in r]) for k in keys}
        print(f"AST {label} T {T} n {len(res)} medians", flush=True)
        for k in keys:
            print(f"  {k:24s} {md[k]:9.3f}", flush=True)
        rows.append(dict(case=case, T=T, inputs=res))
    if a.out:
        json.dump(dict(rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
