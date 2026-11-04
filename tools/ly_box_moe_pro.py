#!/usr/bin/env python3
"""MoE prologue stamps inside the layer's CUDA graph: MID (decode 16, tools/ly_moe_pro16.py) or B4 (decode 32,
tools/ly_moe_prob4.py), 128 stamps per CTA.

python tools/ly_box_moe_pro.py FILE --build mid|b4 --judge J.json --router-seed S [--inputs 20] [--lock L]
    [--ls-cache F] [--out f.json] [--raw N] [--touch]

Captured as tools/ly_box_moe_tl16g.py captures the layer (PDL on at T = 1, 8, 16, 32 with all six T = 8 edges, the
file's gate skipped); each fresh input (local_start per input) gets a cold L2 and a zeroed stamp buffer. Per input,
every named slot is reduced to its mean over CTAs, as us after each CTA's own pre-wait entry (slot 64); the table
prints the median over inputs of the per-input CTA means, plus the first CTA's entry to the mean CTA's entry. Probe slots
(100..102, ns) are printed as durations.
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
NAMES = {
    "mid": {0: "entry after wait", 65: "x prefetch issued", 66: "lut done", 67: "gen load back", 68: "route start",
            69: "topk load back", 70: "bar 1", 71: "count loop", 72: "bar 2", 73: "warp0 scan", 74: "bar 3",
            75: "early_issue", 76: "position loop", 77: "bar 4 (route end)", 78: "ss zeroed", 79: "before WS_JOIN",
            1: "after WS_JOIN", 80: "w0 x load issue", 81: "w0 x load back", 2: "first slot full",
            84: "F1 entry", 85: "F1 cnt zero", 86: "F1 topk 0 back", 87: "F1 topk 1 back", 88: "F1 topk 2 back",
            89: "F1 topk 3 back", 90: "F1 scan done", 91: "F1 mbar init", 92: "F1 TMA issued", 93: "F1 x prefetch",
            94: "F1 lut done", 96: "helper entry", 97: "helper after WS_JOIN", 98: "helper ring issued",
            103: "r2: second route start", 99: "r2: second route end"},
    "b4": {0: "entry after wait", 65: "x prefetch issued", 66: "lut done", 67: "mbar init", 68: "gen atomic back",
           69: "route start", 70: "topk loads back", 71: "bar 1", 72: "count loop", 73: "bar 2", 74: "warp0 scan",
           75: "bar 3", 76: "early_issue", 77: "position loop", 78: "bar 4 (route end)", 1: "route done (gt)",
           79: "parity zeroed", 80: "jrow scan", 81: "jrow bar", 82: "gather batch 0 back", 83: "gather stored",
           84: "proxy fence", 85: "gather bar", 86: "xready released", 7: "gather done (gt)", 6: "WS_JOIN (gt)",
           90: "wprod entry", 91: "wprod after WS_JOIN", 92: "wprod range read", 8: "first weight issue (gt)",
           94: "xprod entry", 95: "xprod bar 7", 96: "xprod ranges", 97: "xprod after WS_JOIN",
           10: "xprod first xready (gt)", 103: "r2: second route start", 99: "r2: second route end",
           104: "r2: second wgs_ranges end"},
}
PROBE = {100: "probe first touch (ns)", 101: "probe same 2 MiB, new line (ns)", 102: "probe L2 hit (ns)"}


class Stamping:
    def __init__(self, ext, buf, m):
        self.ext, self.buf, self.m, self.lids = ext, buf, m, None

    def okc_run(self, *args):
        args = list(args)
        if len(args[1]) == 3 and args[2].shape[0] == self.m:
            self.lids = args[3]
            args[1] = list(args[1]) + [self.buf]
        return self.ext.okc_run(*args)

    def __getattr__(self, name):
        return getattr(self.ext, name)


def per_input(s, names, exit_slot):
    """slots < 64 are globaltimer (ns), measured from slot 122 (globaltimer beside the clock64 entry stamp 64);
    slots >= 64 are SM clock64 cycles, converted with each CTA's own rate (clock64 / globaltimer from entry to exit)"""
    s = s[(s[:, 64] > 0) & (s[:, 122] > 0) & (s[:, 120] > 0)].double()
    gt0 = s[:, 122]
    ghz = (s[:, 120] - s[:, 64]) / (s[:, exit_slot] - gt0)
    first = float(gt0.min())
    row = {"cta_entry_spread_mean": (float(gt0.mean()) - first) / 1e3,
           "cta_entry_spread_max": (float(gt0.max()) - first) / 1e3, "ghz": float(ghz.median())}
    for k in names:
        v = s[:, k]
        good = v > 0
        if good.sum() == 0:
            continue
        own = (v - gt0) if k < 64 else (v - s[:, 64]) / ghz
        row[k] = float(own[good].mean()) / 1e3
        row[f"{k}_abs"] = float((own + gt0 - first)[good].mean()) / 1e3
        row[f"{k}_max"] = float((own + gt0 - first)[good].max()) / 1e3
    for k in PROBE:
        v = s[:, k]
        if (v > 0).sum():
            row[k] = float((v / ghz)[v > 0].median())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--build", choices=["mid", "b4"], required=True)
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--raw", type=int, default=0)
    ap.add_argument("--warm", action="store_true", help="also stamp a second replay right after, without the L2 flush")
    a = ap.parse_args()
    m, load = (16, "decode 16") if a.build == "mid" else (32, "decode 32")
    names = NAMES[a.build]
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
    cases = [c for c in cases if c.get("load") == load]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    state["pdl8_on"] = {T: True for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
    state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)}
    mod._moe(m, state)
    buf = torch.zeros(136 * SLOTS, dtype=torch.int64, device="cuda")
    wrap = Stamping(state["ma_ext"][a.build], buf, m)
    state["ma_ext"][a.build] = wrap
    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    timer = br.Timer()
    out = []
    for case in cases:
        seeds = [case["seed"] + j for j in range(a.inputs)]
        fresh = []
        for s in seeds:
            extra = lsc.get(ref, case, s, weights) if lsc else ref.case_fields({**case, "seed": s}, weights)
            fresh.append(ref.generate_input(**{**case, "seed": s, **extra}))
        if lsc:
            lsc.save()
        static = ref.generate_input(**{**case, "seed": seeds[0]})
        g, _ = br.capture(mod.kernel_fn, static, state)
        rows, wrows = [], []
        for j, f in enumerate(fresh):
            br.copy_into(static, f)
            for warm in ((False, True) if a.warm else (False,)):
                buf.zero_()
                torch.cuda.synchronize()
                if warm:
                    timer.a.record()
                    g.replay()
                    timer.b.record()
                    timer.b.synchronize()
                    t = timer.a.elapsed_time(timer.b) * 1e3
                else:
                    t = timer.cold(g) / 1e3
                s = buf.view(-1, SLOTS)[:136].cpu()
                nsm = int((s[:, 64] != 0).sum())
                r = per_input(s, names, 11 if a.build == "mid" else 5)
                r.update(graph_us=t, seed=seeds[j], grid=nsm)
                if j < a.raw:
                    r["raw"] = s[s[:, 64] > 0].tolist()
                (wrows if warm else rows).append(r)
        for tag, rows in (("cold", rows), ("warm", wrows)):
            if rows:
                report(f"{case['load']} {case['layer']} {tag}", rows, names)
        out.append(dict(case=case, inputs=rows, warm=wrows))
    if a.out:
        json.dump(dict(rows=out), open(a.out, "w"))


def report(label, rows, names):
    if rows:
        med = lambda k: statistics.median(r[k] for r in rows if k in r)  # noqa: E731
        print(f"PRO {label} n {len(rows)} graph_us {med('graph_us'):.2f} grid {rows[0]['grid']} sm GHz {med('ghz'):.3f} "
              f"cta entry spread mean {med('cta_entry_spread_mean'):.2f} max {med('cta_entry_spread_max'):.2f}",
              flush=True)
        order = sorted((k for k in names if any(k in r for r in rows)), key=lambda k: med(k))
        for k in order:
            print(f"PRO {label} slot {k:3d} {names[k]:<24s} own {med(k):6.2f}  first-cta mean {med(f'{k}_abs'):6.2f} "
                  f"latest {med(f'{k}_max'):6.2f}", flush=True)
        for k, nm in PROBE.items():
            if any(k in r for r in rows):
                print(f"PRO {label} slot {k:3d} {nm:<32s} {med(k):8.0f}", flush=True)


if __name__ == "__main__":
    main()
