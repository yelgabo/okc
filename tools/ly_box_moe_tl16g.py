#!/usr/bin/env python3
"""The decode 16 MID timeline inside the layer's CUDA graph (tools/ly_box_moe_tl16.py replays the MID launch eagerly).

python tools/ly_box_moe_tl16g.py muse_decode_layer/cand/<tl file>.py --judge J.json --router-seed S [--inputs 20]
    [--lock /root/okc/.build/box/gpu.lock] [--ls-cache F] [--out f.json] [--raw N]

The MID extension is wrapped so that every 16-token launch passes a fourth scratch tensor (the stamp buffer), then the
whole layer is captured as bx_box_run.py captures it (PDL on at T = 1, 8, 16, 32 with all six T = 8 edges, the file's
gate skipped), and each replay gets its own fresh input (local_start per input), a cold L2 and a zeroed stamp buffer.
Stamps are those of tools/ly_moe_tl16.py; the summary is ly_box_moe_tl16.stats(). The graph replay's CUDA-event time
is printed beside the span.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as br  # noqa: E402
import ly_box_moe_tl16 as tl16  # noqa: E402

import torch  # noqa: E402

SLOTS = tl16.SLOTS


class Stamping:
    def __init__(self, ext, buf):
        self.ext, self.buf, self.lids = ext, buf, None

    def okc_run(self, *args):
        args = list(args)
        if len(args[1]) == 3 and args[2].shape[0] == 16:
            self.lids = args[3]
            args[1] = list(args[1]) + [self.buf]
        return self.ext.okc_run(*args)

    def __getattr__(self, name):
        return getattr(self.ext, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--loads", default="decode 16")
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
    mod = br.load_module("okc_submission", a.submission)
    if hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: None
    state = mod.prepare(weights)
    state["pdl8_on"] = {T: True for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
    state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)}
    mod._moe(16, state)
    buf = torch.zeros(136 * SLOTS, dtype=torch.int64, device="cuda")
    wrap = Stamping(state["ma_ext"]["mid"], buf)
    state["ma_ext"]["mid"] = wrap
    lsc = br.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED) \
        if a.ls_cache else None
    timer = br.Timer()
    rows = []
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
        res = []
        for j, f in enumerate(fresh):
            br.copy_into(static, f)
            buf.zero_()
            torch.cuda.synchronize()
            t = timer.cold(g) / 1e3
            nsm = int((buf.view(-1, SLOTS)[:, 0] != 0).sum())
            s = buf[: nsm * SLOTS].view(nsm, SLOTS).cpu()
            st = tl16.stats(s, wrap.lids)
            st.update(graph_us=t, seed=seeds[j])
            if j < a.raw:
                st["raw"] = s.tolist()
                st["lids"] = wrap.lids.cpu().tolist()
            res.append(st)
        label = f"{case['load']} {case['layer']}"
        keys = ["graph_us", "span", "ramp", "route", "A", "handoff", "B", "bend_spread", "bend_to_bar", "bar_idle",
                "final", "unit_us", "unit_a", "unit_b", "a_wwait", "b_wwait", "units", "distinct", "routes"]
        med = {k: statistics.median(r[k] for r in res) for k in keys}
        print(f"TL16G {label} n {len(res)} medians " + " ".join(f"{k} {v:.2f}" for k, v in med.items()), flush=True)
        rows.append(dict(case=case, inputs=res))
    if a.out:
        json.dump(dict(rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
