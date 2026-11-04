"""Paired in-process arms of one layer file on the box: prepare once, then per cell capture one CUDA graph per arm
and replay the arms in rotating order on the same fresh input (per-input local_start, cold L2, judge-style
overhead subtracted), as tools/bx_box_run.py does for one side. An arm is a state override and / or module
globals set while its graph is captured (PCX_CFG entries are read at launch, so they are baked into the graph).

    python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/F.py --loads "verify 16,verify 32" --runs 40 \
        --arm 'off={"state": {"v16pdl_on": {}}}' --arm 'all={}' --arm 'attn={"state": {"v16pdl_edges": {128: {"attn"}, 256: {"attn"}}}}' \
        --arm 'un3={"pcx": {("un", 16): (3, 0, 0, 0)}}' --lock /root/okc/.build/box/gpu.lock ...

The first arm is the reference of the paired differences. Every arm gets the reference check on one input.
"""
import argparse
import ast
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bx_box_run as bx  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("submission")
    ap.add_argument("--loads", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--ls-cache", default="")
    ap.add_argument("--lock", default="")
    ap.add_argument("--ext-cache", default="")
    ap.add_argument("--pin-pdl8", default="1,8,16,32")
    ap.add_argument("--skip-pdl8-gate", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        name, _, expr = spec.partition("=")
        arms.append((name, ast.literal_eval(expr) if expr.strip() else {}))
    if a.lock:
        a.ext_cache = a.ext_cache or "/root/.cache/bx_ext"
        bx.release_lock_for_triton()
        bx.BudgetClock.install()
        bx.LOCK = bx.GpuLock(a.lock)
        bx.LOCK.acquire("start")
    if a.ext_cache:
        bx.content_addressed_extensions(a.ext_cache)
    pdir = os.path.abspath(a.problem)
    sys.path.insert(0, pdir)
    ref = bx.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    judge, judge_cases = bx.read_judge(a.judge)
    cases = judge_cases or [dict(c, seed=1000 + i) for i, c in enumerate(bx.CASES[os.path.basename(pdir.rstrip("/"))])]
    if a.loads:
        keep = {s.strip() for s in a.loads.split(",")}
        cases = [c for c in cases if c.get("load") in keep]
    if a.layers:
        keep = {s.strip() for s in a.layers.split(",")}
        cases = [c for c in cases if c.get("layer") in keep]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    mod = bx.load_module("okc_submission", a.submission)
    if a.skip_pdl8_gate and hasattr(mod, "_pdl8_setup"):
        mod._pdl8_setup = lambda st: print("BX in-prepare PDL8 gate skipped (--skip-pdl8-gate)", flush=True)
    state = mod.prepare(weights)
    if a.pin_pdl8 is not None and isinstance(state, dict) and "pdl8_on" in state:
        pins = {int(t) for t in a.pin_pdl8.split(",") if t.strip()}
        state["pdl8_on"] = {T: T in pins for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
        state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)} if 8 in pins else {}
    print(f"ARMS state v16pdl_on {state.get('v16pdl_on')} pdl8_on {state.get('pdl8_on')}", flush=True)
    torch.cuda.synchronize()
    lsc = (bx.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED)
           if a.ls_cache else None)
    timer = bx.Timer()
    out = []

    def arm_state(spec):
        return dict(state, **spec.get("state", {}))

    def with_globals(spec, fn):
        saved = {}
        pcx = spec.get("pcx")
        if pcx:
            saved["PCX_CFG"] = mod.PCX_CFG
            merged = dict(mod.PCX_CFG)
            merged.update({tuple(k): tuple(v) for k, v in pcx.items()})
            mod.PCX_CFG = merged
        for k, v in spec.get("glob", {}).items():
            saved[k] = getattr(mod, k)
            setattr(mod, k, v)
        try:
            return fn()
        finally:
            for k, v in saved.items():
                setattr(mod, k, v)

    for case in cases:
        case = dict(case)
        case.update(ref.case_fields(case, weights))
        seeds = [case["seed"] + j for j in range(a.runs)]
        fresh = []
        for s in seeds:
            extra = lsc.get(ref, case, s, weights) if lsc is not None else ref.case_fields({**case, "seed": s}, weights)
            fresh.append(ref.generate_input(**{**case, "seed": s, **extra}))
        if lsc is not None:
            lsc.save()
        checks = {}
        if not a.no_check:
            data = ref.generate_input(**{**case, "seed": case["seed"] + 7919})
            for name, spec in arms:
                st = arm_state(spec)
                ok, msg = bx.check(ref, data, with_globals(spec, lambda: bx.call(mod.kernel_fn, data, st)), weights)
                checks[name] = bool(ok)
            torch.cuda.synchronize()
            del data
        graphs = {}
        for name, spec in arms:
            static = ref.generate_input(**{**case, "seed": seeds[0]})
            st = arm_state(spec)
            graphs[name] = (with_globals(spec, lambda: bx.capture(mod.kernel_fn, static, st))[0], static)
        ov = timer.overhead()["overhead_ns"]
        runs = {name: [] for name, _ in arms}
        names = [n for n, _ in arms]
        for i in range(a.runs):
            k = i % len(names)
            order = names[k:] + names[:k]
            if (i // len(names)) % 2:
                order = order[::-1]
            for name in order:
                g, static = graphs[name]
                bx.copy_into(static, fresh[i])
                torch.cuda.synchronize()
                runs[name].append((timer.cold(g) - ov) / 1e3)
        label = f"{case['load']} {case['layer']}"
        base = runs[names[0]]
        row = dict(cell=label, checks=checks, runs=runs)
        for name in names:
            d = [x - y for x, y in zip(runs[name], base)]
            med = statistics.median(runs[name])
            print(f"ARMS {label:18s} {name:12s} net {med:8.2f} us  d {statistics.median(d):+6.2f}  "
                  f"neg {sum(1 for v in d if v < 0)}/{len(d)}  check {'PASS' if checks.get(name, True) else 'FAIL'}",
                  flush=True)
        out.append(row)
        del graphs, fresh
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f)
    if bx.LOCK is not None:
        bx.LOCK.release("end")


if __name__ == "__main__":
    main()
