#!/usr/bin/env python3
"""Decode 16 / 32 row boundaries inside the layer's CUDA graph (stamps of tools/dl_ly_rbst_patch.py).

python tools/ly_box_rbst.py muse_decode_layer/cand/ly_rbst.py --judge J.json --router-seed S [--inputs 20]
    [--loads "decode 16,decode 32"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json]

Per cell two graphs of the whole layer, captured as bx_box_run.py captures them (PDL pinned on at every T, the
file's gate skipped): plain (every stamp switch off) and instrumented. Each fresh input (local_start per input) is
replayed through both with a cold L2.

Times are us after the producer's end (the last producer CTA's tid-0 stamp after its output stores): o for b1,
the unpooler for b2. Per boundary: entry (first / last program), wait return (p50 / last), then per-program steps
from its own wait return (p50 over programs), and the last program's end (after its final store issue).
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

SK_SLOTS, SK_CTAS, GL_SLOTS, GL_ROWS, GE_SLOTS, GE_CTAS = 32, 1024, 16, 64, 16, 512
med = statistics.median


def producer(rows):
    rows = [r for r in rows if r[0] > 0]
    if not rows:
        return None
    return dict(n=len(rows), entry0=min(r[0] for r in rows), wait_last=max(r[2] for r in rows),
                end_p50=med(r[23] for r in rows), end=max(r[23] for r in rows))


def gluon(rows, t0):
    rows = [r for r in rows if r[0] > 0]
    if not rows:
        return None
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    step = lambda a, b: med((r[b] - r[a]) / 1e3 for r in rows)  # noqa: E731
    last = max(rows, key=lambda r: r[8])
    return dict(
        n=len(rows), entry_first=us(min(r[0] for r in rows)), entry_last=us(max(r[0] for r in rows)),
        pre_landed_last=us(max(r[1] for r in rows)),
        wait_p50=us(med(r[2] for r in rows)), wait_last=us(max(r[2] for r in rows)),
        wait_first=us(min(r[2] for r in rows)),
        branch=step(2, 3), red1=step(3, 4), stream=step(4, 5), ns_store=step(5, 6), red2=step(6, 7),
        out_store=step(7, 8), wait_to_end=step(2, 8), wait_to_end_max=max((r[8] - r[2]) / 1e3 for r in rows),
        end_p50=us(med(r[8] for r in rows)), end=us(last[8]),
        last_row_wait=us(last[2]), last_row_entry=us(last[0]),
        sms=len({int(r[9]) for r in rows}),
    )


def glue(rows, t0):
    rows = [r for r in rows if r[0] > 0]
    if not rows:
        return None
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    step = lambda a, b: med((r[b] - r[a]) / 1e3 for r in rows)  # noqa: E731
    last = max(rows, key=lambda r: r[6])
    return dict(
        n=len(rows), entry_first=us(min(r[0] for r in rows)), entry_last=us(max(r[0] for r in rows)),
        sync_last=us(max(r[1] for r in rows)),
        wait_p50=us(med(r[2] for r in rows)), wait_last=us(max(r[2] for r in rows)),
        wait_first=us(min(r[2] for r in rows)),
        branch=step(2, 3), red1=step(3, 4), red2=step(4, 5), store=step(5, 6),
        wait_to_end=step(2, 6), wait_to_end_max=max((r[6] - r[2]) / 1e3 for r in rows),
        end_p50=us(med(r[6] for r in rows)), end=us(last[6]), last_cta_wait=us(last[2]),
        sms=len({int(r[7]) for r in rows}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 16,decode 32")
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
    sk_buf = torch.zeros(2 * SK_CTAS * SK_SLOTS, dtype=torch.int64, device="cuda")
    gl_buf = torch.zeros(2 * GL_ROWS * GL_SLOTS, dtype=torch.int64, device="cuda")
    ge_buf = torch.zeros(GE_CTAS * GE_SLOTS, dtype=torch.int64, device="cuda")
    ext = mod.bfp__get_ext()
    gmod = mod._pdl8_graft_glue_module()

    def inst(on):
        ext.sk2_set_rbt(sk_buf.data_ptr() if on else 0)
        gmod.glue_set_ts(ge_buf.data_ptr() if on else 0)
        mod._RBST_BUF = gl_buf if on else None

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
            sk_buf.zero_()
            gl_buf.zero_()
            ge_buf.zero_()
            torch.cuda.synchronize()
            ti = timer.cold(gi) / 1e3
            diff = float((first(outi).float() - op.float()).abs().max())
            sk = sk_buf.view(2, SK_CTAS, SK_SLOTS).cpu().tolist()
            gl = gl_buf.view(2, GL_ROWS, GL_SLOTS).cpu().tolist()
            ge = ge_buf.view(GE_CTAS, GE_SLOTS).cpu().tolist()
            po, pu = producer(sk[0]), producer(sk[1])
            st = dict(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j], T=T)
            if po:
                st["o"] = dict(po, end_spread=(po["end"] - po["end_p50"]) / 1e3, wait_to_end=(po["end"] - po["wait_last"]) / 1e3)
                b1 = gluon(gl[0], po["end"]) or glue(ge, po["end"])
                st["b1_kind"] = "gluon" if gluon(gl[0], po["end"]) else "glue"
                st["b1"] = b1
            if pu:
                st["un"] = dict(pu, end_spread=(pu["end"] - pu["end_p50"]) / 1e3, wait_to_end=(pu["end"] - pu["wait_last"]) / 1e3)
                st["b2"] = gluon(gl[1], pu["end"])
            if j < a.raw:
                st["raw"] = dict(sk_o=[r for r in sk[0] if r[0]], sk_un=[r for r in sk[1] if r[0]],
                                 gl_b1=[r for r in gl[0] if r[0]], gl_b2=[r for r in gl[1] if r[0]], ge=[r for r in ge if r[0]])
            res.append(st)
            b1, b2 = st.get("b1") or {}, st.get("b2") or {}
            print(f"RBSTI {case['load']} {case['layer']} in {j} plain {tp:.2f} inst {ti:.2f} diff {diff:.2e} "
                  f"b1[{st.get('b1_kind')}] n {b1.get('n')} wait {b1.get('wait_last', float('nan')):.2f} end {b1.get('end', float('nan')):.2f} "
                  f"b2 n {b2.get('n')} wait {b2.get('wait_last', float('nan')):.2f} end {b2.get('end', float('nan')):.2f}", flush=True)
        label = f"{case['load']} {case['layer']}"
        print(f"RBST {label} T {T} n {len(res)} medians", flush=True)
        print(f"  graph_plain {med(r['graph_plain'] for r in res):.3f} graph_inst {med(r['graph_inst'] for r in res):.3f} "
              f"maxdiff {max(r['maxdiff'] for r in res):.2e}", flush=True)
        for part in ("o", "un", "b1", "b2"):
            ok = [r[part] for r in res if r.get(part)]
            if not ok:
                continue
            for k in ok[0]:
                vals = [x[k] for x in ok if isinstance(x.get(k), (int, float))]
                if vals and part in ("b1", "b2") or k in ("end_spread", "wait_to_end", "n"):
                    print(f"  {part:3s} {k:18s} {med(vals):9.3f}", flush=True)
        rows.append(dict(case=case, T=T, inputs=res))
    if a.out:
        json.dump(dict(rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
