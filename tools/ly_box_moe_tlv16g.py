#!/usr/bin/env python3
"""The verify 16 / 32 MAIN-build MoE timeline inside the layer's CUDA graph (stamps of tools/ly_moe_tlv16.py).

python tools/ly_box_moe_tlv16g.py muse_decode_layer/cand/ly_fabq_tlv16.py --judge J.json --router-seed S [--inputs 20]
    [--loads "verify 16,verify 32"] [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json] [--raw N]

As tools/ly_box_moe_tlb4g.py (two graphs per cell, plain and stamped, one cold replay of each per fresh input, local_start
per input), on the MAIN build's launch (M = 128 / 256). Per input it adds: the SM clock in each phase (clock64 against
globaltimer), the job table (fat jobs of up to 64 tokens, skinny jobs, re-streamed experts, stages per n-tile count),
the weights producer's issue gaps bucketed by the n-tile count of the stage whose release freed the slot (stage n - 3,
ring depth 3) and by that stage's place in its segment, and a least-squares fit of each CTA's phase time on its stage
counts per n-tile class (the achieved per-stage time by N, phase A and B separately), pooled over the cell's inputs.
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

SLOTS = 336
NSTAMP = 296
JOBS_AT = 136 * SLOTS
STAGE_BYTES = 35840
UNIT_STAGES = 16


class Stamping:
    def __init__(self, ext, buf, ms):
        self.ext, self.buf, self.ms, self.lids, self.on = ext, buf, ms, None, False

    def okc_run(self, *args):
        args = list(args)
        if self.on and len(args[1]) == 3 and args[2].shape[0] in self.ms:
            self.lids = args[3]   # the capture's routing; per-replay routing comes from the stamped job table
            args[1] = list(args[1]) + [self.buf]
        return self.ext.okc_run(*args)

    def __getattr__(self, name):
        return getattr(self.ext, name)


REORDER = True  # the base's OKC_WGSREORD; --no-reorder for in-order bases (ly_rfin)


def wgs_split(pb, g0, g1):
    ue = ((g1 - 1) >> 4) << 4
    return ue if (REORDER and pb == 0 and (g1 & 15) != 0 and ue > g0) else g0


def issue_order(pb, g0, g1):
    """stages in the order the producers issue them (phase A: the trailing partial unit first)"""
    sp = wgs_split(pb, g0, g1)
    return list(range(sp, g1)) + list(range(g0, sp))


def stats(s, lids, jobs, bw):
    n = s.shape[0]
    v = s.tolist()
    t0 = min(r[0] for r in v)
    rel = lambda r, k: (r[k] - t0) / 1e3 if r[k] > 0 else math.nan  # noqa: E731
    mean = lambda xs: statistics.mean([x for x in xs if not math.isnan(x)])  # noqa: E731
    col = lambda k: [rel(r, k) for r in v]  # noqa: E731
    nj, njf = jobs["nj"], jobs["njf"]
    jC, jE = jobs["jC"], jobs["jE"]
    ntt_of = [(c + 7) >> 3 for c in jC]
    rng = [(r[21] & 0xFFFF, (r[21] >> 16) & 0xFFFF, (r[21] >> 32) & 0xFFFF, (r[21] >> 48) & 0xFFFF) for r in v]
    nA = [b - a for a, b, _, _ in rng]
    nB = [d - c for _, _, c, d in rng]
    # per CTA: stage counts per n-tile class in each phase, phase times, clocks
    rowsA, rowsB, yA, yB = [], [], [], []
    gapsA, gapsB = {}, {}   # (ntt, pos) -> list of gaps
    switch = []
    mhz = []
    for r, (a0, a1, b0, b1) in zip(v, rng):
        seqA = issue_order(0, a0, a1)
        seqB = issue_order(1, b0, b1)
        cntA = [0] * 9
        cntB = [0] * 9
        for gs in seqA:
            cntA[ntt_of[gs >> 9]] += 1
        for gs in seqB:
            cntB[ntt_of[gs >> 8]] += 1
        rowsA.append(cntA[1:])
        rowsB.append(cntB[1:])
        yA.append((r[2] - r[8]) / 1e3 if r[8] > 0 and r[2] > 0 else math.nan)     # first weight issue to A end
        yB.append((r[3] - r[2]) / 1e3 if r[3] > 0 and r[2] > 0 else math.nan)     # A end to B end
        if r[320] and r[321] and r[2] > r[1]:
            mhz.append((r[321] - r[320]) / (r[2] - r[1]) * 1e3)
        # issue gaps: stamp n is the issue of the n-th stage over both phases; gap n is attributed to stage n - 3
        seq = seqA + seqB
        stamps = [r[24 + k] for k in range(min(NSTAMP, len(seq)))]
        pos = {}
        for ph, sq in ((0, seqA), (1, seqB)):
            prev_u = None
            run = 0
            for gs in sq:
                u = gs >> 4
                run = run + 1 if u == prev_u else 0
                pos[(ph, gs)] = min(run, 2)
                prev_u = u
        for k in range(1, len(stamps)):
            if stamps[k] == 0 or stamps[k - 1] == 0:
                break
            g = (stamps[k] - stamps[k - 1]) / 1e3
            rel_k = k - 3
            if rel_k < 0:
                continue
            if rel_k < len(seqA):
                if k >= len(seqA) and k < len(seqA) + 3:
                    switch.append(g)
                    continue
                gs = seqA[rel_k]
                key = (ntt_of[gs >> 9], pos[(0, gs)])
                gapsA.setdefault(key, []).append(g)
            else:
                gs = seqB[rel_k - len(seqA)]
                key = (ntt_of[gs >> 8], pos[(1, gs)])
                gapsB.setdefault(key, []).append(g)
    loc = lids[lids >= 0]
    distinct = int(torch.unique(loc).numel())
    tok = {}
    for c, e in zip(jC, jE):
        tok[e] = tok.get(e, 0) + c
    bend, fin, ex = col(3), col(4), col(5)
    stA, stB = sum(nA), sum(nB)
    out = dict(
        span=max(x for x in ex if not math.isnan(x)), route=mean(col(1)), gather=mean(col(7)), join=mean(col(6)),
        issue0=mean(col(8)), A=mean(col(2)) - mean(col(8)), aend_spread=max(col(2)) - min(col(2)),
        B=mean(bend) - mean(col(2)), bend_spread=max(bend) - min(bend),
        mfin_wait=mean([f - b for b, f in zip(bend, fin)]) if any(not math.isnan(f) for f in fin) else math.nan,
        final=mean([e - f for f, e in zip(fin, ex)]) if any(not math.isnan(f) for f in fin) else mean([e - b for b, e in zip(bend, ex)]),
        exit_spread=max(ex) - min(ex), x_rdy_wait=mean([r[14] / 1e3 for r in v]), b_dep_wait=mean([r[13] / 1e3 for r in v]),
        stages_a=stA, stages_b=stB, stages=stA + stB, stages_min=768 * distinct,
        stg_cta_min=min(a + b for a, b in zip(nA, nB)), stg_cta_max=max(a + b for a, b in zip(nA, nB)),
        tbs_a=statistics.median([a * STAGE_BYTES / y / 1e6 for a, y in zip(nA, yA) if y > 0]),
        tbs_b=statistics.median([b * STAGE_BYTES / y / 1e6 for b, y in zip(nB, yB) if y > 0]),
        tbs_a_agg=stA * STAGE_BYTES / (mean(col(2)) - mean(col(8))) / 1e6,
        int_floor=n * STAGE_BYTES / bw / 1e6, distinct=distinct, routes=int(loc.numel()), grid=n,
        nj=nj, njf=njf, restream=nj - distinct, hot=sum(1 for t in tok.values() if t > 64),
        max_tokens=max(tok.values()) if tok else 0,
        stg_ntt_a=[sum(r[i] for r in rowsA) for i in range(8)], stg_ntt_b=[sum(r[i] for r in rowsB) for i in range(8)],
        smclk=statistics.median(mhz) if mhz else math.nan, switch_gap=statistics.median(switch) if switch else math.nan,
        rowsA=rowsA, rowsB=rowsB, yA=yA, yB=yB,
        gapsA={f"{k[0]}:{k[1]}": g for k, g in gapsA.items()}, gapsB={f"{k[0]}:{k[1]}": g for k, g in gapsB.items()},
    )
    return out


def fit(rows, y):
    """least squares y = sum_i c_i rows_i + c0 over CTAs and inputs; returns c (8) and c0 and the residual sd"""
    import numpy as np
    X = np.array([r + [1.0] for r in rows], dtype=float)
    Y = np.array(y, dtype=float)
    ok = ~np.isnan(Y)
    X, Y = X[ok], Y[ok]
    used = [i for i in range(8) if X[:, i].sum() > 0] + [8]
    c, *_ = np.linalg.lstsq(X[:, used], Y, rcond=None)
    full = [math.nan] * 9
    for i, u in enumerate(used):
        full[u] = float(c[i])
    res = Y - X[:, used] @ c
    return full, float(res.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="verify 16,verify 32")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", type=int, default=0)
    ap.add_argument("--no-reorder", action="store_true", help="the base runs its phase A range in order (OKC_WGSREORD 0)")
    a = ap.parse_args()
    global REORDER
    REORDER = not a.no_reorder
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
    print(f"TLV16 bandwidth read {bw:.3f} TB/s", flush=True)
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    state["pdl8_on"] = {T: True for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
    state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)}
    mod._moe(128, state)
    ext = state["ma_ext"]["main"]
    print("TLV16 main attrs (regs, local, smem) x2", list(ext.okc_attrs()), flush=True)
    buf = torch.zeros(JOBS_AT + 512, dtype=torch.int64, device="cuda")
    wrap = Stamping(ext, buf, (128, 256))
    state["ma_ext"]["main"] = wrap
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
            sys.exit("no 128 / 256-token MAIN launch was captured")
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
            nsm = int((buf.view(-1)[: 136 * SLOTS].view(-1, SLOTS)[:, 0] != 0).sum())
            s = buf[: nsm * SLOTS].view(nsm, SLOTS).cpu()
            jb = buf[JOBS_AT: JOBS_AT + 512].cpu().tolist()
            nj, njf = int(jb[511]), int(jb[510])
            jobs = dict(nj=nj, njf=njf, jC=[int(x & 0xFF) for x in jb[:nj]], jE=[int((x >> 8) & 0xFF) for x in jb[:nj]],
                        jS=[int((x >> 16) & 0xFFFF) for x in jb[:nj]], jrow=[int((x >> 32) & 0xFFFF) for x in jb[:nj]])
            # the routing of this replay is the one the graph just ran; lids was captured on the static input, so
            # recompute distinct / routes from the job table (every local route is in some job)
            e = torch.tensor(jobs["jE"], dtype=torch.int64)
            c = torch.tensor(jobs["jC"], dtype=torch.int64)
            lids = torch.repeat_interleave(e, c) if nj else torch.full((0,), -1, dtype=torch.int64)
            st = stats(s, lids, jobs, bw)
            st.update(graph_plain=tp, graph_inst=ti, maxdiff=diff, seed=seeds[j], jobs=jobs)
            if j < a.raw:
                st["raw"] = s.tolist()
            res.append(st)
        label = f"{case['load']} {case['layer']}"
        keys = ["graph_plain", "graph_inst", "span", "route", "gather", "join", "issue0", "A", "aend_spread", "B",
                "bend_spread", "mfin_wait", "final", "exit_spread", "x_rdy_wait", "b_dep_wait", "stages", "stages_min",
                "stg_cta_min", "stg_cta_max", "tbs_a", "tbs_b", "tbs_a_agg", "int_floor", "distinct", "routes", "nj", "njf",
                "restream", "hot", "max_tokens", "smclk", "switch_gap", "maxdiff"]
        med = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"TLV16G {label} n {len(res)} medians " + " ".join(f"{k} {v:.3f}" for k, v in med.items()), flush=True)
        sa = [sum(r["stg_ntt_a"][i] for r in res) for i in range(8)]
        sb = [sum(r["stg_ntt_b"][i] for r in res) for i in range(8)]
        print(f"TLV16G {label} stages by n-tiles (1..8) A {sa} B {sb}", flush=True)
        cA, rA = fit([row for r in res for row in r["rowsA"]], [y for r in res for y in r["yA"]])
        cB, rB = fit([row for r in res for row in r["rowsB"]], [y for r in res for y in r["yB"]])
        print(f"TLV16G {label} fit A us per stage by n-tiles " + " ".join(f"{x:.3f}" for x in cA[:8])
              + f" + {cA[8]:.2f} (resid sd {rA:.2f})", flush=True)
        print(f"TLV16G {label} fit B us per stage by n-tiles " + " ".join(f"{x:.3f}" for x in cB[:8])
              + f" + {cB[8]:.2f} (resid sd {rB:.2f})", flush=True)
        for ph in ("gapsA", "gapsB"):
            pool = {}
            for r in res:
                for k, g in r[ph].items():
                    pool.setdefault(k, []).extend(g)
            line = " ".join(f"{k}:{statistics.median(g):.2f}({len(g)})" for k, g in sorted(pool.items(), key=lambda kv: (int(kv[0].split(':')[0]), kv[0])))
            print(f"TLV16G {label} issue gaps us by released stage n-tiles:pos(0 first,1 second,2 later) {ph} {line}", flush=True)
        for d in sorted({r["distinct"] for r in res}):
            b = [r for r in res if r["distinct"] == d]
            print(f"TLV16G {label} distinct {d:2d} n {len(b):2d} " + " ".join(
                f"{k} {statistics.median(r[k] for r in b):.2f}" for k in
                ("graph_plain", "graph_inst", "span", "issue0", "A", "B", "bend_spread", "final", "nj", "restream", "routes")), flush=True)
        rows.append(dict(case=case, inputs=res))
    if a.out:
        json.dump(dict(bandwidth=bw, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
