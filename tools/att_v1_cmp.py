"""Attention kernel alone at the verify 1 shapes: the decode layer's ATT7 launcher (_launch7 from a layer file)
against the standalone attention board's file (kernel_fn), same fresh inputs, same process, cold L2, CUDA graphs.

python tools/att_v1_cmp.py LAYER.py STANDALONE.py [--runs 60] [--cells sliding:soft,full:soft,full:coherent]
    [--lock /root/okc/.build/box/gpu.lock] [--ext-cache /root/.cache/bx_ext] [--plans] [--out F.json]

Arms: std (standalone kernel_fn), lay (layer _launch7, pdl off, no pooler prefetch), lay_pf (as lay plus the 4 MB
L2 prefetch the layer issues at T = 8, from a dummy buffer). Each run builds one fresh input (seed base + run), copies
it into every arm's graph buffers, flushes L2 with a 1 GiB read and replays each arm once, arm order rotated per run.
Net = raw - fixed replay cost (2 * one - two, as the judge).
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bx_box_run as bx  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer")
    ap.add_argument("standalone")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--cells", default="sliding:soft,full:soft,full:coherent")
    ap.add_argument("--arms", default="std,lay,lay_pf")
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="")
    ap.add_argument("--seed", type=int, default=880001)
    ap.add_argument("--plans", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.ext_cache:
        bx.content_addressed_extensions(a.ext_cache)
    if a.lock:
        bx.LOCK = bx.GpuLock(a.lock)
        bx.LOCK.acquire("start")
    sys.path.insert(0, os.path.join(ROOT, "muse_attention_decode"))
    ref = load("att_reference", os.path.join(ROOT, "muse_attention_decode", "reference.py"))
    lay = load("lay_mod", a.layer)
    std = load("std_mod", a.standalone)
    dev = "cuda"
    st = {"attention_scratch": {"rows": {}, "cluster": {}, "retained": []}}
    pf_buf = torch.empty(4 << 20, dtype=torch.uint8, device=dev).random_(0, 255)

    def f_std(d):
        return std.kernel_fn(d)

    def f_lay(d):
        return lay._launch7(d, attention_state=st, pdl=False, prefetch=(0, 0))

    def f_lay_pf(d):
        return lay._launch7(d, attention_state=st, pdl=False, prefetch=(pf_buf.data_ptr(), 4 << 20))

    every = {"std": f_std, "lay": f_lay, "lay_pf": f_lay_pf}
    arms = {n: every[n] for n in a.arms.split(",")}
    timer = bx.Timer()
    ovh = timer.overhead()
    print(f"OVH {ovh}", flush=True)
    results = {"overhead": ovh, "cells": {}, "argv": sys.argv}
    for cell in a.cells.split(","):
        layer, dist = cell.split(":")
        case = dict(batch=1, ctx=80000, heads="tp8", layer=layer, q_len=8, page=1, dist=dist)
        static = ref.generate_input(**case, seed=a.seed)
        if a.plans:
            cfg, pt_w = static["config"], static["page_table"].shape[1]
            s1 = std._one_state(static["qkv"].device)
            print(f"PLAN {cell} std (n_splits, nc, mode) = {std._one_plan(cfg, pt_w, s1['resident'])}", flush=True)
            print(f"PLAN {cell} lay _plan7 = {lay._plan7(1, 1, 8, cfg['window'], pt_w, 1, static['qkv'].device)} "
                  f"tuned {lay._TUNED7.get((1, 8, bool(cfg['window']), 1))}", flush=True)
        graphs, outs, statics = {}, {}, {}
        for name, fn in arms.items():
            d = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in static.items()}
            chk = ref.generate_input(**case, seed=a.seed + 7919)
            o = fn(chk)
            torch.cuda.synchronize()
            ok, msg = ref.check_implementation(chk, ref.collect(chk, o))
            print(f"CHECK {cell} {name} {'PASS' if ok else 'FAIL'} {msg}", flush=True)
            g, out = bx.capture(fn, d, None)
            graphs[name], outs[name], statics[name] = g, out, d
        names = list(arms)
        times = {n: [] for n in names}
        for r in range(a.runs):
            fresh = ref.generate_input(**case, seed=a.seed + 1 + r)
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                bx.copy_into(statics[n], fresh)
                torch.cuda.synchronize()
                times[n].append(timer.cold(graphs[n]) - ovh["overhead_ns"])
                if r == 0:
                    torch.cuda.synchronize()
                    ok, msg = ref.check_implementation(statics[n], ref.collect(statics[n], outs[n]))
                    print(f"RCHECK {cell} {n} {'PASS' if ok else 'FAIL'} {msg}", flush=True)
        row = {n: dict(median_us=statistics.median(times[n]) / 1e3, ns=times[n]) for n in names}
        line = " ".join(f"{n} {row[n]['median_us']:.3f}" for n in names)
        if "std" in names:
            for n in names:
                if n != "std":
                    row[n]["paired_minus_std_us"] = statistics.median(
                        [x - s for x, s in zip(times[n], times["std"])]) / 1e3
                    line += f" | {n}-std {row[n]['paired_minus_std_us']:+.3f}"
        print(f"CELL verify1 {cell} {line}", flush=True)
        results["cells"][cell] = row
        del graphs, outs, statics
        torch.cuda.synchronize()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f)
    if bx.LOCK:
        bx.LOCK.release("end")


if __name__ == "__main__":
    main()
