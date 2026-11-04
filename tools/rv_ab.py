"""In-process paired A/B of layer files on the box: every arm prepared in one process, the arms'
replays interleaved per repetition (rotating order) on the same fresh inputs, cold L2 each replay.

python tools/rv_ab.py muse_decode_layer ctl=PATH ctl2=PATH cand=PATH [box_run options]

The first arm is the control; the report gives each arm's per-cell net median, the per-load time
(sum over shapes of weight x net), and each arm's paired median difference against the control.
Listing the control twice gives an in-run null for every comparison. Reuses tools/box_run.py's
loader, timer, router and case seeds unchanged.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_run as br  # noqa: E402
import bx_box_run as bx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("arms", nargs="+", help="name=path; the first is the control")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--loads", default="")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--router-seed", type=int, required=True)
    ap.add_argument("--pdl8-tries", type=int, default=5, help="prepares per arm to reach the control's PDL8 enable set")
    ap.add_argument("--ls-cache", default="", help="bx_box_run.py's per-input local_start cache (docs/box-audit.md)")
    ap.add_argument("--ext-cache", default="", help="content-addressed extension builds, as bx_box_run.py")
    ap.add_argument("--pin-pdl8", default=None, help="T values with PDL on in every arm, e.g. 1,16,32 (replaces the re-prepares)")
    ap.add_argument("--skip-pdl8-gate", action="store_true", help="with --pin-pdl8: skip the in-prepare gate")
    ap.add_argument("--pdl8-want", default="", help="the control's PDL8 enable set as printed, e.g. "
                    "'{1: True, 8: False, 16: True, 32: True}'")
    a = ap.parse_args()
    if a.skip_pdl8_gate and a.pin_pdl8 is None:
        ap.error("--skip-pdl8-gate needs --pin-pdl8")
    if a.ext_cache:
        bx.content_addressed_extensions(a.ext_cache)

    arms = [s.split("=", 1) for s in a.arms]
    pdir = os.path.abspath(a.problem)
    name = os.path.basename(pdir.rstrip("/"))
    sys.path.insert(0, pdir)
    ref = br.load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    _, judge_cases = br.read_judge(a.judge)
    cases = judge_cases or [dict(c, seed=1000 + i) for i, c in enumerate(br.CASES[name])]
    if a.loads:
        keep = {s.strip() for s in a.loads.split(",")}
        cases = [c for c in cases if c.get("load") in keep]
    weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    torch.cuda.synchronize()

    impl, pdl = {}, {}
    dropped = []
    for idx, (arm, path) in enumerate(arms):
        mod = br.load_module(f"okc_rv_{arm}", path)
        if a.skip_pdl8_gate and hasattr(mod, "_pdl8_setup"):
            mod._pdl8_setup = lambda st: print("RV in-prepare PDL8 gate skipped (--skip-pdl8-gate)", flush=True)
        for attempt in range(1 if a.pin_pdl8 is not None else a.pdl8_tries):
            t0 = time.monotonic()
            impl.pop(arm, None)
            state = None
            torch.cuda.empty_cache()
            try:
                state = mod.prepare(weights) if hasattr(mod, "prepare") else None
            except Exception as e:  # a variant that fails to build drops out; the control must not
                if idx == 0:
                    raise
                print(f"RV {arm} {path} prepare FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                dropped.append(arm)
                break
            torch.cuda.synchronize()
            if a.pin_pdl8 is not None and isinstance(state, dict) and "pdl8_on" in state:
                pins = {int(t) for t in a.pin_pdl8.split(",") if t.strip()}
                print(f"RV {arm} gate chose {state['pdl8_on']}; pinned to {sorted(pins)}", flush=True)
                state["pdl8_on"] = {T: T in pins for T in getattr(mod, "PDL8_T", (1, 8, 16, 32))}
                state["pdl8_edges"] = {8: frozenset(mod.PDL8D_EDGES)} if 8 in pins else {}
            impl[arm] = (mod.kernel_fn, state)
            # the PDL8 gate picks its enable set per prepare, so two preparations of the same bytes can time
            # different paths (verify 1 and decode 16 / 32 move by 6 to 8 us); arms are compared on one set
            pdl[arm] = state.get("pdl8_on") if isinstance(state, dict) else None
            print(f"RV {arm} {path} import+prepare {time.monotonic() - t0:.1f} s; pdl8_on {pdl[arm]}"
                  f" (attempt {attempt + 1})", flush=True)
            on = pdl[arm]
            if not isinstance(on, dict):
                break
            if idx == 0:
                if a.pdl8_want and str(on) != a.pdl8_want:
                    continue
                if any(on.get(t) for t in (8, 16, 32)):
                    break
            elif on == pdl[arms[0][0]]:
                break
        else:
            print(f"RV {arm}: pdl8_on {pdl[arm]} after {a.pdl8_tries} prepares does not match", flush=True)

    lsc = (bx.LocalStartCache(a.ls_cache, os.path.join(pdir, "reference.py"), a.router_seed, ref.WEIGHTS_SEED)
           if a.ls_cache else None)
    timer = br.Timer()
    rows, loads = [], {}
    names = [n for n, _ in arms if n not in dropped]
    for case in cases:
        case = dict(case)
        case.update(ref.case_fields(case, weights))
        seeds = [case["seed"] + j for j in range(a.runs)]
        fresh = [ref.generate_input(**{**case, "seed": s, **(lsc.get(ref, case, s, weights) if lsc else
                                                              ref.case_fields({**case, "seed": s}, weights))})
                 for s in seeds]
        row = dict(case=case, check={})
        if not a.no_check:
            data = ref.generate_input(**{**case, "seed": case["seed"] + 7919})
            for arm in names:
                fn, state = impl[arm]
                ok, msg = br.check(ref, data, br.call(fn, data, state), weights)
                torch.cuda.synchronize()
                row["check"][arm] = [bool(ok), msg]
            del data
        graphs = {}
        for arm in names:
            fn, state = impl[arm]
            static = ref.generate_input(**{**case, "seed": seeds[0]})
            graphs[arm] = (br.capture(fn, static, state), static)
        ov = timer.overhead()["overhead_ns"]
        runs = {arm: [] for arm in names}
        k = len(names)
        for i in range(a.runs):
            for arm in names[i % k:] + names[:i % k]:
                (g, _), static = graphs[arm]
                br.copy_into(static, fresh[i])
                torch.cuda.synchronize()
                runs[arm].append(timer.cold(g) - ov)
        row["net_ns"] = runs
        row["median_net_us"] = {arm: statistics.median(v) / 1e3 for arm, v in runs.items()}
        row["paired_diff_us"] = {arm: statistics.median(x - c for x, c in zip(runs[arm], runs[names[0]])) / 1e3
                                 for arm in names[1:]}
        rows.append(row)
        load = case.get("load")
        w = float(case.get("weight", 1))
        for arm in names:
            loads.setdefault(load, {}).setdefault(arm, 0.0)
            loads[load][arm] += w * row["median_net_us"][arm]
        chk = " ".join(f"{arm}:{'PASS' if v[0] else 'FAIL'}" for arm, v in row["check"].items())
        cells = " ".join(f"{arm} {row['median_net_us'][arm]:.2f}" for arm in names)
        diffs = " ".join(f"d_{arm} {row['paired_diff_us'][arm]:+.2f}" for arm in names[1:])
        print(f"RV {load} {case.get('layer', '')}: {cells} | paired {diffs} | {chk}", flush=True)
        del graphs, fresh
        torch.cuda.empty_cache()
    for load, v in loads.items():
        print(f"RVLOAD {load}: " + " ".join(f"{arm} {v[arm]:.1f}" for arm in names)
              + " | " + " ".join(f"{arm}-{names[0]} {v[arm] - v[names[0]]:+.1f}" for arm in names[1:]), flush=True)
    if lsc:
        lsc.save()
    for arm in names[1:]:
        tot = sum(v[arm] - v[names[0]] for v in loads.values())
        gm = math.exp(statistics.fmean(math.log(v[names[0]] / v[arm]) for v in loads.values()))
        print(f"RVSUM {arm}-{names[0]}: six-load sum {tot:+.1f} us (weighted), geomean speed ratio "
              f"{names[0]}/{arm} {gm:.4f}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(problem=name, arms=arms, pdl8_on={k: str(v) for k, v in pdl.items()}, rows=rows, loads=loads), f)


if __name__ == "__main__":
    main()
