"""Run an okc submission file on a local H200 the way the judge times it.

python tools/box_run.py <problem dir> <submission.py> [--baseline] [--judge J.json ...]
    [--loads "decode 1,verify 1"] [--runs N] [--no-check] [--out result.json]

Per shape: prepare(weights) once, a correctness check against the problem's reference.py on one
fresh input, then one CUDA graph captured on a static input set and replayed `runs` times. Each
replay gets a fresh input set copied into the graph's buffers, a cold L2 (a 1 GiB read), and is
timed with CUDA events. With --baseline, SGLang's baseline.py is captured the same way and the two
replays alternate order per repetition, so `paired` is the median of baseline/submission ratios.
The judge's fixed replay cost is 2 * t(1-kernel graph) - t(2-kernel graph), read from its
replay_overhead records (one_ns, two_ns, overhead_ns); it is measured and subtracted here the same
way. --judge takes judge JSONs (okc profile JSON with "benchmarks", or a job JSON with
result.records), reuses their case seeds and prints judge vs box per cell.
"""
import argparse
import importlib.util
import inspect
import json
import os
import statistics
import sys
import time

import torch

CASES = {
    "muse_decode_layer": [
        dict(batch=b, ctx=80000, dist="soft", layer=layer, load=load, page=1, weight=w,
             **({"q_len": 8} if q else {}), **({"spread": "agentic"} if b > 1 else {}))
        for load, b, q in [("decode 1", 1, 0), ("decode 16", 16, 0), ("decode 32", 32, 0),
                           ("verify 1", 1, 1), ("verify 16", 16, 1), ("verify 32", 32, 1)]
        for layer, w in (("sliding", 46), ("full", 16))
    ],
    "muse_attention_decode": [
        dict(batch=b, ctx=80000, heads="tp8", dist="soft" if layer == "sliding" else "coherent", layer=layer,
             load=load, page=1, weight=w, **({"q_len": 8} if q else {}), **({"spread": "agentic"} if b > 1 else {}))
        for load, b, q in [("decode 1", 1, 0), ("decode 16", 16, 0), ("decode 32", 32, 0),
                           ("verify 1", 1, 1), ("verify 16", 16, 1), ("verify 32", 32, 1)]
        for layer, w in (("sliding", 46), ("full", 16))
    ],
    "muse_moe_w4a16": [dict(bs=bs, sharding="ep8", load=load, weight=60) for load, bs in
                       [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8),
                        ("verify 16", 128), ("verify 32", 256)]],
    "muse_bf16_proj": [dict(m=m, layer=layer, load=load, weight=62) for load, m in
                       [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8),
                        ("verify 16", 128), ("verify 32", 256)]
                       for layer in ("qkv_gate", "o", "pooler", "unpooler", "router")],
}
IGNORED = {"seed", "repeats", "weight", "load"}
FLUSH_BYTES = 1 << 30


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cell_key(case):
    return tuple(sorted((k, str(v)) for k, v in case.items() if k not in IGNORED))


def read_judge(paths):
    """{cell key: [record, ...]} over every judge JSON given, plus the case dicts in order."""
    cells, order = {}, []
    for p in paths:
        d = json.load(open(p))
        recs = d.get("benchmarks") or d.get("result", {}).get("records", [])
        for r in recs:
            if r.get("event") != "benchmark":
                continue
            k = cell_key(r["case"])
            if k not in cells:
                order.append(r["case"])
            cells.setdefault(k, []).append(r)
    return cells, order


HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}


def implied_distinct(ref, record):
    """The router_distinct target a judge cell's speed_of_light_ns implies (bisection on the
    reference's own bound); None when the bound does not depend on it."""
    case, sol = record["case"], record.get("speed_of_light_ns")
    if sol is None or not hasattr(ref, "speed_of_light_ns"):
        return None
    f = lambda d: ref.speed_of_light_ns(**{**case, "router_distinct": d}, hw=HW)  # noqa: E731
    lo, hi = 0.0, 64.0
    if f(hi) < sol or f(lo) > sol * (1 + 1e-9) or abs(f(hi) - f(lo)) < 1e-6:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < sol else (lo, mid)
    return round((lo + hi) / 2, 3)


def distinct_router_seed(ref, weights, cases, targets, seed, attempts):
    """weights_fields restricted to the distinct-experts column: the judge also matches mean
    routes (router_load), which no public record exposes."""
    loads = {}
    for i, c in enumerate(cases):
        if c.get("load") in targets:
            loads.setdefault(c["load"], []).append(i)
    hs = []
    with torch.no_grad():
        for c in cases:
            n = ref.ROUTE_SAMPLES if c.get("batch", 1) * c.get("q_len", 1) <= ref.SAMPLE_TOKENS else 1
            seeds = [c["seed"]] + [(c["seed"] + j * 0x2545F4914F6CDD1D) % (2**63 - 1) for j in range(1, n)]
            hs.append((torch.cat([ref.ref_front(ref.generate_input(**{**c, "seed": s, "local_start": 0}), weights)["h"]
                                  for s in seeds]), n))
        best = (float("inf"), None)
        for i in range(1, attempts + 1):
            rs = (seed + i * 0x9E3779B97F4A7C15) % (2**63 - 1)
            router = ref._router(torch.Generator(device="cuda").manual_seed(rs), "cuda", weights["pre_ffn_norm"])
            got = {}
            for load, members in loads.items():
                w = [float(cases[m].get("weight", 1)) for m in members]
                vals = [float(ref._mean_busiest_load(hs[m][0], router, hs[m][1])[1]) for m in members]
                got[load] = sum(a * b for a, b in zip(w, vals)) / sum(w)
            err = max(abs(got[k] / targets[k] - 1) for k in got)
            if err < best[0]:
                best = (err, rs, got)
            if err <= ref.LOAD_TOL:
                break
    return best


def call(fn, data, state):
    return fn(data, state) if state is not None else fn(data)


def tensors(data):
    return {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}


def copy_into(static, fresh):
    for k, v in tensors(static).items():
        f = fresh.get(k)
        if isinstance(f, torch.Tensor) and f.shape == v.shape and f.dtype == v.dtype:
            v.copy_(f, non_blocking=True)


def capture(fn, data, state):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            call(fn, data, state)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = call(fn, data, state)
    torch.cuda.synchronize()
    return g, out


class Timer:
    def __init__(self):
        self.flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda").uniform_()
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")
        self.a = torch.cuda.Event(enable_timing=True)
        self.b = torch.cuda.Event(enable_timing=True)

    def cold(self, g):
        torch.sum(self.flush, dim=0, out=self.sink)
        self.a.record()
        g.replay()
        self.b.record()
        self.b.synchronize()
        return self.a.elapsed_time(self.b) * 1e6

    def overhead(self, n=200):
        x = torch.zeros(1, device="cuda")
        graphs = []
        for kernels in (1, 2):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(kernels):
                    x.add_(1)
            graphs.append(g)
        one = statistics.median(self.cold(graphs[0]) for _ in range(n))
        two = statistics.median(self.cold(graphs[1]) for _ in range(n))
        return dict(one_ns=one, two_ns=two, overhead_ns=2 * one - two)


def check(ref, data, out, weights):
    fn = ref.check_implementation
    nparams = len(inspect.signature(fn).parameters)
    if hasattr(ref, "collect"):
        raw = ref.collect(data, out)
        return fn(data, raw, weights) if nparams == 3 else fn(data, raw)
    return fn(data, out, weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("submission")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--loads", default="")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--inputs", type=int, default=0, help="distinct fresh input sets cycled per shape (0: one per run)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--router-seed", type=int, default=None, help="skip the search and use this router seed")
    ap.add_argument("--router-attempts", type=int, default=2000)
    ap.add_argument("--case-local-start", action="store_true",
                    help="reuse the case seed's local_start for every fresh input (the runner before 2026-11-03)")
    ap.add_argument("--kernels", action="store_true", help="also print a per-kernel GPU time table (torch.profiler, 10 cold replays)")
    ap.add_argument("--profile", action="store_true",
                    help="no timing: one eager call per case inside cudaProfilerStart/Stop (ncu --profile-from-start off)")
    a = ap.parse_args()

    pdir = os.path.abspath(a.problem)
    name = os.path.basename(pdir.rstrip("/"))
    sys.path.insert(0, pdir)
    ref = load_module("reference", os.path.join(pdir, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False

    judge, judge_cases = read_judge(a.judge)
    cases = judge_cases or [dict(c, seed=1000 + i) for i, c in enumerate(CASES[name])]
    if a.loads:
        keep = {s.strip() for s in a.loads.split(",")}
        cases = [c for c in cases if c.get("load") in keep]

    weights = ref.generate_weights(ref.WEIGHTS_SEED) if hasattr(ref, "generate_weights") else None
    targets = {}
    for recs in judge.values():
        d = implied_distinct(ref, recs[0])
        if d is not None:
            targets.setdefault(recs[0]["case"]["load"], []).append(d)
    targets = {k: sum(v) / len(v) for k, v in targets.items()}
    if a.router_seed is not None:
        weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=a.router_seed)
    elif targets and hasattr(ref, "_mean_busiest_load"):
        t0 = time.monotonic()
        err, rs, got = distinct_router_seed(ref, weights, cases, targets, ref.WEIGHTS_SEED, a.router_attempts)
        print(f"BOX router_distinct targets {targets}; router_seed {rs} gives {got} (max err {err:.3f},"
              f" {time.monotonic() - t0:.0f} s)", flush=True)
        weights = ref.generate_weights(ref.WEIGHTS_SEED, router_seed=rs)
    torch.cuda.synchronize()

    sides = [("submission", a.submission)]
    if a.baseline:
        sides.append(("baseline", os.path.join(pdir, "baseline.py")))
    impl = {}
    for side, path in sides:
        t0 = time.monotonic()
        mod = load_module(f"okc_{side}", path)
        state = mod.prepare(weights) if hasattr(mod, "prepare") else None
        torch.cuda.synchronize()
        impl[side] = (mod.kernel_fn, state)
        print(f"BOX {side} import+prepare {time.monotonic() - t0:.1f} s", flush=True)

    timer = Timer()
    rows = []
    for case in cases:
        case = dict(case)
        if hasattr(ref, "case_fields") and weights is not None:
            case.update(ref.case_fields(case, weights))
        seeds = [case["seed"] + j for j in range(a.inputs or a.runs)]
        fresh = []
        for s in seeds:
            # case_fields runs per input on the judge: each fresh input's local_start is its own
            # busiest range, the range the router_load / router_distinct targets are measured on.
            # One case-seed local_start for every input left about a third of decode 1 inputs with
            # no local expert (docs/box.md, "Routing").
            extra = (ref.case_fields({**case, "seed": s}, weights)
                     if hasattr(ref, "case_fields") and weights is not None and not a.case_local_start else {})
            fresh.append(ref.generate_input(**{**case, "seed": s, **extra}))
        row = dict(case=case)
        if a.profile:
            fn, state = impl["submission"]
            data = fresh[0]
            call(fn, data, state)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            call(fn, data, state)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print(f"BOX profiled {case.get('load')} {case.get('layer', '')}", flush=True)
            continue
        if not a.no_check:
            data = ref.generate_input(**{**case, "seed": case["seed"] + 7919})
            fn, state = impl["submission"]
            ok, msg = check(ref, data, call(fn, data, state), weights)
            torch.cuda.synchronize()
            row["check"] = [bool(ok), msg]
            del data
        graphs = {}
        for side, (fn, state) in impl.items():
            static = ref.generate_input(**{**case, "seed": seeds[0]})
            graphs[side] = (capture(fn, static, state), static)
        ov = timer.overhead()
        runs = {side: [] for side in graphs}
        for i in range(a.runs):
            order = list(graphs) if i % 2 == 0 else list(graphs)[::-1]
            for side in order:
                (g, _), static = graphs[side]
                copy_into(static, fresh[i % len(fresh)])
                torch.cuda.synchronize()
                runs[side].append(timer.cold(g))
        sub = runs["submission"]
        row.update(replay_overhead=ov, runs_ns=sub, median_raw_ns=statistics.median(sub),
                   median_net_ns=statistics.median(sub) - ov["overhead_ns"])
        if "baseline" in runs:
            base = runs["baseline"]
            net = lambda t: max(t - ov["overhead_ns"], 1.0)  # noqa: E731
            ratios = [net(b) / net(s) for b, s in zip(base, sub)]
            row.update(baseline_runs_ns=base, baseline_raw_ns=statistics.median(base),
                       paired_ratio=statistics.median(ratios),
                       median_ns=(statistics.median(base) - ov["overhead_ns"]) / statistics.median(ratios))
        jr = judge.get(cell_key({k: v for k, v in case.items() if k != "local_start"}), [])
        if jr:
            row["judge_net_us"] = [(r["median_raw_ns"] - r["replay_overhead"]["overhead_ns"]) / 1e3 for r in jr]
            row["judge_shown_us"] = [r["median_ns"] / 1e3 for r in jr]
            row["judge_baseline_net_us"] = [r["baseline_ns"] / 1e3 for r in jr]
        if a.kernels:
            (g, _), static = graphs["submission"]
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(10):
                    copy_into(static, fresh[i % len(fresh)])
                    torch.cuda.synchronize()
                    timer.cold(g)
            agg = {}
            for ev in prof.events():
                if ev.device_type == torch.autograd.DeviceType.CUDA and "reduce_kernel" not in ev.name and not ev.name.startswith(("Memcpy", "Memset")):
                    agg[ev.name] = agg.get(ev.name, 0.0) + ev.device_time / 10
            row["kernels_us"] = agg
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
                print(f"  KERNEL {v:8.2f} us  {k[:90]}", flush=True)
            print(f"  KERNEL {sum(agg.values()):8.2f} us  total", flush=True)
        rows.append(row)
        label = f"{case.get('load')} {case.get('layer', '')}".strip()
        line = (f"BOX {label}: raw {row['median_raw_ns'] / 1e3:.2f} net {row['median_net_ns'] / 1e3:.2f} us"
                f" (overhead {ov['overhead_ns'] / 1e3:.2f})")
        if "paired_ratio" in row:
            line += (f" | sglang raw {row['baseline_raw_ns'] / 1e3:.2f} paired {row['paired_ratio']:.3f}"
                     f" shown {row['median_ns'] / 1e3:.2f}")
        if jr:
            jn = statistics.median(row["judge_net_us"])
            line += (f" | judge net {jn:.2f} (n={len(jr)}, {min(row['judge_net_us']):.2f}-"
                     f"{max(row['judge_net_us']):.2f}) box/judge {row['median_net_ns'] / 1e3 / jn:.3f}")
        if jr and "baseline_raw_ns" in row:
            jb = statistics.median(row["judge_baseline_net_us"])
            bb = (row["baseline_raw_ns"] - ov["overhead_ns"]) / 1e3
            line += f" | sglang box/judge {bb:.2f}/{jb:.2f} = {bb / jb:.3f}"
        if "check" in row:
            line += f" | check {'PASS' if row['check'][0] else 'FAIL'}"
        print(line, flush=True)
        del graphs, fresh
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(problem=name, submission=a.submission, gpu=torch.cuda.get_device_name(),
                           torch=torch.__version__, rows=rows), f)


if __name__ == "__main__":
    main()
