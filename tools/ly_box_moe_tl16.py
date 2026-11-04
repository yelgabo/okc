"""tools/box_moe_tl8.py for the layer's T = 16 MID build (decode 16), on muse_decode_layer/cand/ly_tl16.py.

Original notes (T = 8 / TINY2 there, T = 16 / MID here):

python tools/box_moe_tl8.py muse_decode_layer/cand/mo_tl8.py --judge J.json --router-seed S [--inputs 40]
    [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json]

For each fresh verify 1 input (local_start per input, as the judge) the layer runs eagerly once; the TINY2 launch's
arguments are recorded, then replayed twice after a 1 GiB L2 flush: a plain launch timed with CUDA events, and an
instrumented launch (fourth scratch tensor, tools/dl_moe_tl.py) whose 64 stamps per CTA give ramp, phase A, A-to-B
handoff, phase B, finish spread, barrier and final pass. The floor is the unit bytes the CTAs stream (each unit
32,768 code bytes + 4,096 scale bytes) over the box's measured read bandwidth.
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402

import torch  # noqa: E402

UNIT_BYTES = 32768 + 4096
SLOTS = 64


def bandwidth():
    n = 4 << 30
    a = torch.empty(n // 4, dtype=torch.float32, device="cuda").normal_()
    b = torch.empty_like(a)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    out = {}
    for name, fn, nbytes in (("copy (read + write)", lambda: b.copy_(a), 2 * n), ("sum (read)", lambda: a.sum(), n)):
        fn()
        ts = []
        for _ in range(5):
            torch.cuda.synchronize()
            ev[0].record()
            fn()
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]) / 1e3)
        out[name] = nbytes / min(ts) / 1e12
    del a, b
    torch.cuda.empty_cache()
    return out


class Recorder:
    def __init__(self, ext):
        self.ext, self.calls = ext, []

    def okc_run(self, *args):
        if len(args[1]) == 3 and args[2].shape[0] == 16:
            self.calls.append([x.clone() if torch.is_tensor(x) else x for x in args])
        return self.ext.okc_run(*args)

    def __getattr__(self, name):
        return getattr(self.ext, name)


def stats(s, lids):
    n = s.shape[0]
    t0 = int(s[:, 0].min())
    col = lambda k: [(int(v) - t0) / 1e3 if int(v) > 0 else math.nan for v in s[:, k]]  # noqa: E731
    dur = lambda k: [int(v) / 1e3 for v in s[:, k]]  # noqa: E731
    m = lambda v: statistics.mean(x for x in v if not math.isnan(x))  # noqa: E731
    cnt = s[:, 17]
    nA = (cnt & 0xFFFF).tolist()
    nB = ((cnt >> 16) & 0xFFFF).tolist()
    entry, route, first, aend = col(0), col(1), col(2), col(3)
    dep1, bend, barr, brel, fin = col(5), col(6), col(9), col(10), col(11)
    dts, dta, dtb = [], [], []
    for c in range(n):
        prev = None
        for q in range(46):
            v = int(s[c, 18 + q])
            if v == 0:
                break
            if prev is not None and q != nA[c]:
                dts.append((v - prev) / 1e3)
                (dta if q < nA[c] else dtb).append((v - prev) / 1e3)
            prev = v
    units = sum(nA) + sum(nB)
    loc = lids[lids >= 0]
    return dict(
        span=max(x for x in fin if not math.isnan(x)), ramp=m(first), route=m(route), A=m(aend) - m(first),
        handoff=m(dep1) - m(aend), B=m(bend) - m(dep1), bend_spread=max(bend) - min(bend),
        bend_to_bar=m(barr) - m(bend), bar_idle=m([b - a for a, b in zip(barr, brel)]), final=m(fin) - m(brel),
        exit_spread=max(fin) - min(fin), unit_us=statistics.median(dts) if dts else math.nan,
        unit_a=statistics.median(dta) if dta else math.nan, unit_b=statistics.median(dtb) if dtb else math.nan,
        a_wwait=m(dur(12)), b_wwait=m(dur(13)), units=units, units_per_cta=(min(a + b for a, b in zip(nA, nB)),
                                                                            max(a + b for a, b in zip(nA, nB))),
        distinct=int(torch.unique(loc).numel()), routes=int(loc.numel()), grid=n)


def l2_probe(mod, state, rec, flush, n):
    """Issue-rate probe for the unit loop: tokens 0 and 1 routed to local expert 0 only (one expert, 768 units,
    28.3 MB, fits the 50 MB L2), a TINY2 build with normal-priority weight loads (OKC_L2H=0) so the weights stay in
    L2, then the same instrumented launch cold (after the flush) and warm (right after). The warm per-unit interval
    is the loop's rate with the weights near; the cold one is the HBM-fed rate on the same schedule."""
    ext = mod._ma_ext(tuple(mod.MOE_TINY2_DEFINES) + ("-DOKC_L2H=0",) + tuple(state["ma_sfx"]))
    out = []
    w_, scr, x, lids, wts, o, l0, l1, nsm = rec.calls[-1]
    pl = torch.full_like(lids, -1)
    pl[0, 0] = 0
    pl[1, 0] = 0
    for i in range(n):
        res = {}
        for tag in ("cold", "warm", "warm2"):
            tl = torch.zeros(136 * SLOTS, dtype=torch.int64, device="cuda")
            if tag == "cold":
                flush.fill_(1.0)
            torch.cuda.synchronize()
            ext.okc_run(w_, list(scr) + [tl], x, pl, wts, torch.empty_like(o), l0, l1, nsm)
            torch.cuda.synchronize()
            res[tag] = stats(tl[: nsm * SLOTS].view(nsm, SLOTS).cpu(), pl)
        print("TL16 l2probe " + " | ".join(
            f"{t}: span {r['span']:.2f} A {r['A']:.2f} B {r['B']:.2f} unit A {r['unit_a']:.3f} B {r['unit_b']:.3f} awwait {r['a_wwait']:.2f}"
            f" units/cta {r['units_per_cta']}" for t, r in res.items()), flush=True)
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 16")
    ap.add_argument("--inputs", type=int, default=40)
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext")
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--l2test", type=int, default=0,
                    help="also N issue-rate probes: one local expert, weights L2-resident (TINY2 + OKC_L2H=0, second launch)")
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
    bw = bandwidth()
    print("TL16 bandwidth TB/s", {k: round(v, 3) for k, v in bw.items()}, flush=True)
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    state["pdl8_on"] = {T: True for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
    state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)}
    torch.cuda.synchronize()
    mw = state["ma_w"]
    tens = [x for x in (mw if isinstance(mw, (list, tuple)) else [mw]) if torch.is_tensor(x)]
    wbytes = sum(x.numel() * x.element_size() for x in tens)
    print("TL16 ma_w tensors", [(tuple(x.shape), str(x.dtype)) for x in tens], "bytes", wbytes,
          "per local expert", wbytes / 32, flush=True)
    mod._moe(16, state)
    rec = Recorder(state["ma_ext"]["mid"])
    state["ma_ext"]["mid"] = rec
    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    flush = torch.empty(1 << 28, dtype=torch.float32, device="cuda")
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    rows = []
    probes = []
    for case in cases:
        res = []
        for j in range(a.inputs):
            seed = case["seed"] + j
            ls = (lsc.get(ref, case, seed, weights) if lsc else ref.case_fields({**case, "seed": seed}, weights))
            data = ref.generate_input(**{**case, "seed": seed, "local_start": ls["local_start"]})
            rec.calls.clear()
            with torch.no_grad():
                mod.kernel_fn(data, state)
            torch.cuda.synchronize()
            if not rec.calls:
                print("TL16 no MID launch recorded", flush=True)
                continue
            w_, scr, x, lids, wts, out, l0, l1, nsm = rec.calls[-1]
            plain = torch.empty_like(out)
            flush.fill_(1.0)
            torch.cuda.synchronize()
            ev[0].record()
            rec.ext.okc_run(w_, scr, x, lids, wts, plain, l0, l1, nsm)
            ev[1].record()
            torch.cuda.synchronize()
            t_plain = ev[0].elapsed_time(ev[1]) * 1e3
            tl = torch.zeros(136 * SLOTS, dtype=torch.int64, device="cuda")
            inst = torch.empty_like(out)
            flush.fill_(1.0)
            torch.cuda.synchronize()
            rec.ext.okc_run(w_, list(scr) + [tl], x, lids, wts, inst, l0, l1, nsm)
            torch.cuda.synchronize()
            st = stats(tl[: nsm * SLOTS].view(nsm, SLOTS).cpu(), lids)
            st.update(plain_us=t_plain, maxdiff=float((inst.float() - plain.float()).abs().max()), seed=seed)
            res.append(st)
        if lsc:
            lsc.save()
        label = f"{case['load']} {case['layer']}"
        keys = ["plain_us", "span", "ramp", "route", "A", "handoff", "B", "bend_spread", "bend_to_bar", "bar_idle",
                "final", "unit_us", "unit_a", "unit_b", "a_wwait", "b_wwait", "units", "distinct", "routes"]
        med = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"TL16 {label} n {len(res)} medians " + " ".join(f"{k} {v:.2f}" for k, v in med.items()), flush=True)
        for d in sorted({r["distinct"] for r in res}):
            b = [r for r in res if r["distinct"] == d]
            print(f"TL16 {label} distinct {d:2d} n {len(b):2d} plain {statistics.median(r['plain_us'] for r in b):6.2f}"
                  f" span {statistics.median(r['span'] for r in b):6.2f} units {statistics.median(r['units'] for r in b):6.0f}"
                  f" ramp {statistics.median(r['ramp'] for r in b):4.2f} A {statistics.median(r['A'] for r in b):5.2f}"
                  f" handoff {statistics.median(r['handoff'] for r in b):4.2f} B {statistics.median(r['B'] for r in b):5.2f}"
                  f" spread {statistics.median(r['bend_spread'] for r in b):4.2f}"
                  f" tail {statistics.median(r['bend_to_bar'] + r['bar_idle'] + r['final'] for r in b):4.2f}"
                  f" maxdiff {max(r['maxdiff'] for r in b):.2e}", flush=True)
        rows.append(dict(case=case, inputs=res))
        if a.l2test and not probes:
            probes.extend(l2_probe(mod, state, rec, flush, a.l2test))
    if a.out:
        json.dump(dict(bandwidth=bw, ma_w_bytes=wbytes, rows=rows, l2_probes=probes), open(a.out, "w"))


if __name__ == "__main__":
    main()
