#!/usr/bin/env python3
"""Decode 1 qkv_gate timeline inside the layer's CUDA graph (stamps of tools/dl_ly_qtl_patch.py).

python tools/ly_box_qtl.py muse_decode_layer/cand/ly_qtl.py --judge J.json --router-seed S [--inputs 10]
    [--loads "decode 1"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json] [--raw N]

Per cell two graphs of the whole layer are captured as bx_box_run.py captures them (PDL on at T = 1, 8, 16, 32 with
all T = 8 edges, the file's gate skipped): plain (sk2_set_qtl(0)) and instrumented (the qkv sk2 launch at M = 1 gets
the stamp buffer as dbg). Each fresh input is replayed through both with a cold L2. Times are us from the first
CTA's entry. Floor: the bytes the kernel reads (12-bit qkv stream, escape table, x) at --bw TB/s.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

SLOTS = 128
NSTG = 16
STAGE_W = 18432   # 4 warps x 2 units x 3 n8 tiles x 768 B (12-bit codes of 24 rows x 512 k)


def nb(t):
    return t.numel() * t.element_size()


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


def stats(s, nring, bw):
    v = s.tolist()
    n = len(v)
    t0 = min(r[0] for r in v)
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    mean, med = statistics.mean, statistics.median
    entry = [us(r[0]) for r in v]
    issue0 = [us(r[16]) for r in v]
    pre_done = [us(r[1]) for r in v]
    wait_x = [us(r[3]) for r in v]
    land = [[us(r[32 + s]) for s in range(NSTG)] for r in v]
    issue = [[us(r[16 + s]) if r[16 + s] else float("nan") for s in range(NSTG)] for r in v]
    loop_end = [max(us(r[48 + w]) for w in range(4)) for r in v]
    epi = [us(r[6]) for r in v]
    fin = [us(r[7]) for r in v]
    sms = [r[4] for r in v]
    lat = [land[c][s] - issue[c][s] for c in range(n) for s in range(nring)]       # first-ring issue -> land
    lat_late = [land[c][s] - issue[c][s] for c in range(n) for s in range(nring, NSTG)]
    rel = [issue[c][s + nring] - land[c][s] for c in range(n) for s in range(NSTG - nring)]   # land(s) -> refill of its slot
    ivl = [land[c][s] - land[c][s - 1] for c in range(n) for s in range(nring + 1, NSTG)]
    alll = sorted(x for row in land for x in row)
    tot = n * NSTG * STAGE_W
    t10, t90 = q(alll, 0.1), q(alll, 0.9)
    rate = 0.8 * tot / (t90 - t10) / 1e6 if t90 > t10 else float("nan")   # TB/s between 10% and 90% of the stages landed
    per_cta_land = [statistics.mean([land[c][s] for c in range(n)]) for s in range(NSTG)]
    cret = [[us(r[64 + s]) for s in range(NSTG)] for r in v]
    cdone = [[us(r[80 + s]) for s in range(NSTG)] for r in v]
    cwait = [[cret[c][s] - cdone[c][s - 1] for s in range(1, NSTG)] for c in range(n)]
    cproc = [cdone[c][s] - cret[c][s] for c in range(n) for s in range(NSTG)]
    gate = [issue[c][s + nring] - cdone[c][s] for c in range(n) for s in range(NSTG - nring)]
    esc_end = [max(us(r[96 + e]) for e in range(4)) for r in v]
    past_cbar = [us(r[100]) for r in v]
    return dict(
        grid=n, sms=len(set(sms)), entry_last=max(entry), entry_spread=max(entry) - min(entry),
        issue0=mean(issue0), pre_done=mean(pre_done), wait_x=mean(wait_x),
        land0_first=min(r[0] for r in land), land0_mean=mean(r[0] for r in land), land0_last=max(r[0] for r in land),
        lat_ring=med(lat), lat_late=med(lat_late), slot_refill=med(rel), land_ivl=med(ivl),
        land15_mean=mean(r[NSTG - 1] for r in land), land15_last=max(r[NSTG - 1] for r in land),
        loop_end=mean(loop_end), loop_end_max=max(loop_end), last_land_to_loop=mean(le - r[NSTG - 1] for le, r in zip(loop_end, land)),
        epi=mean(e - le for e, le in zip(epi, loop_end)), store=mean(f - e for f, e in zip(fin, epi)),
        fin_mean=mean(fin), fin_max=max(fin), fin_spread=max(fin) - min(fin),
        c_first=mean(r[0] for r in cret), c_wait_mid=med([w for r in cwait for w in r[:9]]),
        c_wait_sum=mean(sum(r) for r in cwait), c_wait_late=med([w for r in cwait for w in r[9:]]),
        c_proc=med(cproc), c_done15=mean(r[NSTG - 1] for r in cdone),
        refill_after_c0=med(gate), esc_end_after_loop=mean(e - le for e, le in zip(esc_end, loop_end)),
        cbar_after_loop=mean(p - le for p, le in zip(past_cbar, loop_end)),
        t10=t10, t90=t90, rate_tbs=rate, stage_land_mean=per_cta_land,
        floor_w=tot / bw / 1e6,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 1")
    ap.add_argument("--layers", default="sliding,full")
    ap.add_argument("--inputs", type=int, default=10)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--bw", type=float, default=4.443)
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", type=int, default=0)
    ap.add_argument("--tokr", type=int, default=8, help="qkv config of the stamped load: 8 (T <= 8), 16, 32 "
                    "(instrument built with dl_ly_qtl_patch.py --m-any)")
    ap.add_argument("--narrow", action="store_true", help="T = 1 qkv runs the narrow build (ly_q1n): stamp it; its "
                    "escape warps leave the ring, so the escape-landing columns are empty")
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
    ext = mod._de_ext() if a.narrow else mod.bfp__get_ext()
    bst = state["bfp"]
    qname = mod.bfp_SK2_PICK[("qkv_gate", a.tokr)]
    ci, qstream, qtab, pmax, b4, nrows = bst["sk2"][qname]
    nring = mod.bfp__sk2_geom(next(c for c in mod.bfp_SK2_CONFIGS if c[0] == qname), a.tokr)[1]
    if a.narrow:
        ncap = {8: "qkv8n_r144", 16: "qkv16n_r176", 32: "qkv32n_r208"}[a.tokr]
        nring = mod.bfp__sk2_geom((ncap, 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1), a.tokr)[1]
    w_bytes = dict(qkv=nb(qstream), qtab=nb(qtab), x=8192 * 2 * a.tokr, out=2304 * 2 * a.tokr)
    print("QTL bytes", w_bytes, "pmax", pmax, "nstage", nring, "choice", bst["choice"].get(("qkv_gate", 1)), flush=True)
    buf = torch.zeros(96 * SLOTS, dtype=torch.int64, device="cuda")
    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    timer = br.Timer()
    rows = []
    for case in cases:
        seeds = [case["seed"] + j for j in range(a.inputs)]
        fresh = []
        for sd in seeds:
            extra = lsc.get(ref, case, sd, weights) if lsc else ref.case_fields({**case, "seed": sd}, weights)
            fresh.append(ref.generate_input(**{**case, "seed": sd, **extra}))
        if lsc:
            lsc.save()
        static = ref.generate_input(**{**case, "seed": seeds[0]})
        ext.sk2_set_qtl(0)
        gp, outp = br.capture(mod.kernel_fn, static, state)
        ext.sk2_set_qtl(buf.data_ptr())
        gi, outi = br.capture(mod.kernel_fn, static, state)
        ext.sk2_set_qtl(0)
        tot = sum(w_bytes.values())
        res = []
        for j, f in enumerate(fresh):
            br.copy_into(static, f)
            torch.cuda.synchronize()
            tp = timer.cold(gp) / 1e3
            first = lambda o: o if torch.is_tensor(o) else next((x for x in o if torch.is_tensor(x)), None)  # noqa: E731
            op = first(outp).clone()
            buf.zero_()
            torch.cuda.synchronize()
            ti = timer.cold(gi) / 1e3
            diff = float((first(outi).float() - op.float()).abs().max())
            nct = int((buf.view(-1, SLOTS)[:, 0] != 0).sum())
            s = buf[: nct * SLOTS].view(nct, SLOTS).cpu()
            st = stats(s, nring, a.bw)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j], floor_all=tot / a.bw / 1e6)
            if j < a.raw:
                st["raw"] = s.tolist()
            res.append(st)
        label = f"{case['load']} {case['layer']}"
        keys = [k for k in res[0] if k not in ("raw", "seed", "stage_land_mean")]
        md = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"QTL {label} n {len(res)} medians " + " ".join(f"{k} {v:.3f}" for k, v in md.items()), flush=True)
        slm = [statistics.median(r["stage_land_mean"][s] for r in res) for s in range(NSTG)]
        print(f"QTL {label} stage land mean " + " ".join(f"{x:.2f}" for x in slm), flush=True)
        rows.append(dict(case=case, inputs=res))
    if a.out:
        json.dump(dict(bw=a.bw, bytes=w_bytes, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
