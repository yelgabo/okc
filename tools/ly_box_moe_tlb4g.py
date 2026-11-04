#!/usr/bin/env python3
"""The decode 32 B4 MoE timeline inside the layer's CUDA graph (stamps of tools/ly_moe_tlb4.py, level 1).

python tools/ly_box_moe_tlb4g.py muse_decode_layer/cand/ly_tlb4.py --judge J.json --router-seed S [--inputs 20]
    [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json] [--raw N]

Two graphs of the whole layer are captured per cell as bx_box_run.py captures it (PDL on at T = 1, 8, 16, 32 with all
six T = 8 edges, the file's gate skipped): one plain, one whose 32-token B4 launch passes a fourth scratch tensor (the
stamp buffer). Each fresh input (local_start per input) is replayed through both, each with a cold L2; the plain and
instrumented graph times are printed side by side, with the largest output difference between them.

Per input: span (first CTA entry to last exit), the mean CTA's route / gather / producer release / first weight issue,
phase A (release to A end), phase B (A end to B end), the MFIN wait (B end to first final item ready), the final pass,
the B-end and exit spreads, stage counts against 768 per distinct local expert, units split between CTAs, and the
weights producer's per-stage issue interval in A and B (median over stages 3..39 of every CTA), against the byte floor
of one 35,840-byte stage per CTA.
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402
import ly_box_moe_tl16 as tl16  # noqa: E402

import torch  # noqa: E402

SLOTS = 64
STAGE_BYTES = 35840


class Stamping:
    def __init__(self, ext, buf):
        self.ext, self.buf, self.lids, self.on = ext, buf, None, False

    def okc_run(self, *args):
        args = list(args)
        if self.on and len(args[1]) == 3 and args[2].shape[0] == 32:
            self.lids = args[3]
            args[1] = list(args[1]) + [self.buf]
        return self.ext.okc_run(*args)

    def __getattr__(self, name):
        return getattr(self.ext, name)


def segs(g0, g1):
    """units touched by stage range [g0, g1), and how many of its two ends cut a unit (each cut is counted by both
    CTAs that share the unit, hence the halving in stats)"""
    if g1 <= g0:
        return 0, 0
    u0, u1 = g0 >> 4, (g1 - 1) >> 4
    split = int((g0 & 15) != 0) + int((g1 & 15) != 0)
    return u1 - u0 + 1, split


def stats(s, lids, bw):
    n = s.shape[0]
    v = s.tolist()
    t0 = min(r[0] for r in v)
    rel = lambda r, k: (r[k] - t0) / 1e3 if r[k] > 0 else math.nan  # noqa: E731
    mean = lambda xs: statistics.mean([x for x in xs if not math.isnan(x)])  # noqa: E731
    col = lambda k: [rel(r, k) for r in v]  # noqa: E731
    rng = [(r[21] & 0xFFFF, (r[21] >> 16) & 0xFFFF, (r[21] >> 32) & 0xFFFF, (r[21] >> 48) & 0xFFFF) for r in v]
    nA = [b - a for a, b, _, _ in rng]
    nB = [d - c for _, _, c, d in rng]
    sa = [segs(a, b) for a, b, _, _ in rng]
    sb = [segs(c, d) for _, _, c, d in rng]
    ia, ib = [], []
    for r, a in zip(v, nA):
        prev = None
        for k in range(40):
            t = r[24 + k]
            if t == 0:
                break
            if prev is not None and k >= 3 and k != a and k != a + 1 and k != a + 2:
                (ia if k < a else ib).append((t - prev) / 1e3)
            prev = t
    bend, fin, ex = col(3), col(4), col(5)
    loc = lids[lids >= 0]
    distinct = int(torch.unique(loc).numel())
    hw = [(r[12] - r[11]) / 1e3 if r[11] > 0 else math.nan for r in v]
    return dict(
        span=max(x for x in ex if not math.isnan(x)), route=mean(col(1)), gather=mean(col(7)), join=mean(col(6)),
        issue0=mean(col(8)), A=mean(col(2)) - mean(col(6)), aend_spread=max(col(2)) - min(col(2)),
        B=mean(bend) - mean(col(2)), bend_spread=max(bend) - min(bend), mfin_wait=mean([f - b for b, f in zip(bend, fin)]),
        final=mean([e - f for f, e in zip(fin, ex)]), exit_spread=max(ex) - min(ex),
        x_rdy_wait=mean([r[14] / 1e3 for r in v]), b_dep_wait=mean([r[13] / 1e3 for r in v]), b_dep_first=mean(hw),
        b_issue0=mean(col(9)), stages_a=sum(nA), stages_b=sum(nB), stages=sum(nA) + sum(nB), stages_min=768 * distinct,
        stg_cta_min=min(a + b for a, b in zip(nA, nB)), stg_cta_max=max(a + b for a, b in zip(nA, nB)),
        segs_a=sum(x for x, _ in sa), segs_b=sum(x for x, _ in sb), split_a=sum(y for _, y in sa) / 2,
        split_b=sum(y for _, y in sb) / 2,
        int_a=statistics.median(ia) if ia else math.nan, int_b=statistics.median(ib) if ib else math.nan,
        int_floor=n * STAGE_BYTES / bw / 1e6, distinct=distinct, routes=int(loc.numel()), grid=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 32")
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
    cases = [c for c in cases if c.get("load") in keep]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    bw = tl16.bandwidth()["sum (read)"]
    print(f"TLB4 bandwidth read {bw:.3f} TB/s", flush=True)
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    state["pdl8_on"] = {T: True for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
    state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)}
    mod._moe(32, state)
    ext = state["ma_ext"]["b4"]
    print("TLB4 b4 attrs (regs, local, smem) x2", list(ext.okc_attrs()), flush=True)
    buf = torch.zeros(136 * SLOTS, dtype=torch.int64, device="cuda")
    wrap = Stamping(ext, buf)
    state["ma_ext"]["b4"] = wrap
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
        wrap.on = False
        gp, outp = br.capture(mod.kernel_fn, static, state)
        wrap.on = True
        gi, outi = br.capture(mod.kernel_fn, static, state)
        wrap.on = False
        if wrap.lids is None:
            sys.exit("no 32-token B4 launch was captured")
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
            nsm = int((buf.view(-1, SLOTS)[:, 0] != 0).sum())
            s = buf[: nsm * SLOTS].view(nsm, SLOTS).cpu()
            st = stats(s, wrap.lids, bw)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j])
            if j < a.raw:
                st["raw"] = s.tolist()
                st["lids"] = wrap.lids.cpu().tolist()
            res.append(st)
        label = f"{case['load']} {case['layer']}"
        keys = ["graph_plain", "graph_inst", "span", "route", "gather", "join", "issue0", "A", "aend_spread", "B",
                "bend_spread", "mfin_wait", "final", "exit_spread", "x_rdy_wait", "b_dep_first", "b_dep_wait", "b_issue0",
                "stages", "stages_min", "stg_cta_min", "stg_cta_max", "segs_a", "segs_b", "split_a", "split_b", "int_a",
                "int_b", "int_floor", "distinct", "routes", "maxdiff"]
        med = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"TLB4G {label} n {len(res)} medians " + " ".join(f"{k} {v:.3f}" for k, v in med.items()), flush=True)
        for d in sorted({r["distinct"] for r in res}):
            b = [r for r in res if r["distinct"] == d]
            print(f"TLB4G {label} distinct {d:2d} n {len(b):2d} " + " ".join(
                f"{k} {statistics.median(r[k] for r in b):.2f}" for k in
                ("graph_plain", "graph_inst", "span", "issue0", "A", "B", "bend_spread", "mfin_wait", "final", "int_a",
                 "int_b")), flush=True)
        rows.append(dict(case=case, inputs=res))
    if a.out:
        json.dump(dict(bandwidth=bw, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
