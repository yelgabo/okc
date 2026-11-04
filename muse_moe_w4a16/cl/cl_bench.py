"""Paired box bench for muse_moe_w4a16 candidates: one shape, one routing shape, candidate against control.

python muse_moe_w4a16/cl/cl_bench.py --kernel <file.py> --control <file.py> --T 8 --distinct 7 \
    --inputs 20 --runs 30 [--lock /root/okc/.build/box/gpu.lock] [--out result.json]

Both files are submission-style modules (prepare(weights), kernel_fn(data, state)). Weights come from
reference.generate_weights(WEIGHTS_SEED) once and both prepares get the same dict. Inputs are
reference.generate_input(bs=T, sharding="ep8", seed=s) for s = --seed, --seed + 1, ...; with
--distinct D only inputs whose distinct local expert count is D are kept. Per input: both graphs get
the input copied into their static buffers, then `runs` repetitions alternate the order
(candidate-control, control-candidate, ...), each replay after a 1 GiB read and timed with CUDA events.
Every input's graph output is checked with reference.check_implementation for both sides.
See README.md beside this file.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.dirname(HERE)
ROOT = os.path.dirname(PDIR)
sys.path.insert(0, PDIR)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bx_box_run as bx  # noqa: E402  (imports torch; its GpuLock, compile unlock, capture and Timer are reused)
import torch  # noqa: E402

# 3 x 4096 x 4096 weights per expert at 4 bits plus one e4m3 scale per 16 values: 28.31 MB.
EXPERT_BYTES = 3 * 4096 * 4096 * (0.5 + 1 / 16)
# Box read bandwidth (`sum` over 4 GiB), docs/runs-moe.md "T = 8 MoE floor and gap"; TASK.md uses 4.8 TB/s.
BOX_BYTES_PER_S = 4.45e12
FLOOR_US_PER_EXPERT = EXPERT_BYTES / BOX_BYTES_PER_S * 1e6
FIRST_LINE = "# okc: problem=muse_moe_w4a16"


def load_submission(name, path):
    with open(path) as f:
        first = f.readline().rstrip("\n")
    if first != FIRST_LINE:
        sys.exit(f"{path}: first line is {first!r}, expected {FIRST_LINE!r}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def distinct_local(data):
    ids = data["topk_ids"]
    cfg = data["config"]
    lo, hi = cfg["local_start"], cfg["local_start"] + cfg["local_count"]
    return int(torch.unique(ids[(ids >= lo) & (ids < hi)]).numel())


def pick_inputs(ref, T, dist, seed, n, target, max_scan):
    """The first n seeds from `seed` on whose input has `target` distinct local experts (any when target is None)."""
    picked, seen = [], {}
    s = seed
    while len(picked) < n and s < seed + max_scan:
        d = distinct_local(ref.generate_input(bs=T, sharding="ep8", seed=s, dist=dist))
        seen[d] = seen.get(d, 0) + 1
        if target is None or d == target:
            picked.append((s, d))
        s += 1
    if len(picked) < n:
        sys.exit(f"only {len(picked)} of {n} inputs with {target} distinct experts in seeds {seed}..{s - 1}; "
                 f"scanned distribution {dict(sorted(seen.items()))}")
    return picked, dict(sorted(seen.items())), s - seed


def iqr(v):
    q = statistics.quantiles(v, n=4, method="inclusive")
    return q[2] - q[0]


def bootstrap_ci(v, reps=2000, seed=0):
    if len(v) < 2:
        return v[0], v[0]
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def kernel_time_us(timer, g, n=3):
    """Mean summed GPU kernel time of one cold replay (torch.profiler; the read or write flush, memcpy and memset
    left out)."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            timer.cold(g)
    total = 0.0
    for ev in prof.events():
        if (ev.device_type == torch.autograd.DeviceType.CUDA and "reduce_kernel" not in ev.name
                and "FillFunctor" not in ev.name and not ev.name.startswith(("Memcpy", "Memset"))):
            total += ev.device_time
    return total / n


class WriteFlushTimer(bx.Timer):
    """The flush tools/box_moe_tl8.py uses (a 1 GiB fill_): it leaves L2 full of dirty lines, which the timed
    kernel then writes back. The default read flush is the judge's and bx_box_run's."""

    def cold(self, g):
        self.flush.fill_(1.0)
        self.a.record()
        g.replay()
        self.b.record()
        self.b.synchronize()
        return self.a.elapsed_time(self.b) * 1e6


def summarize(rows, ov_us, failed, a):
    cm = statistics.median(r["cand_med"] for r in rows)
    km = statistics.median(r["ctl_med"] for r in rows)
    dm = [r["delta_med"] for r in rows]
    lo, hi = bootstrap_ci(dm)
    counts = {}
    for r in rows:
        counts[r["distinct"]] = counts.get(r["distinct"], 0) + 1
    mean_d = statistics.mean(r["distinct"] for r in rows)
    s = dict(T=a.T, flush=a.flush, distinct_target=a.distinct, distinct_counts=dict(sorted(counts.items())), distinct_mean=mean_d,
             floor_us_at_mean=mean_d * FLOOR_US_PER_EXPERT, overhead_us=ov_us,
             cand_med_raw=cm, cand_med_net=cm - ov_us, ctl_med_raw=km, ctl_med_net=km - ov_us,
             cand_iqr_med=statistics.median(r["cand_iqr"] for r in rows),
             ctl_iqr_med=statistics.median(r["ctl_iqr"] for r in rows),
             delta_med=statistics.median(dm), delta_ci95=[lo, hi],
             delta_iqr_med=statistics.median(r["delta_iqr"] for r in rows),
             checks_failed=failed, inputs=len(rows), runs=a.runs)
    if a.kernels:
        s["cand_kernel_med"] = statistics.median(r["kernels_us"]["candidate"] for r in rows)
        s["ctl_kernel_med"] = statistics.median(r["kernels_us"]["control"] for r in rows)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, help="candidate submission file")
    ap.add_argument("--control", required=True, help="control submission file (the public file's bytes)")
    ap.add_argument("--T", type=int, required=True, help="tokens (bs): 1, 16, 32 decode; 8, 128, 256 verify")
    ap.add_argument("--distinct", type=int, default=None,
                    help="keep only inputs with this many distinct local experts; omitted: consecutive seeds")
    ap.add_argument("--inputs", type=int, default=20)
    ap.add_argument("--runs", type=int, default=30, help="paired repetitions per input")
    ap.add_argument("--seed", type=int, default=1000, help="first input seed of the series")
    ap.add_argument("--dist", default="normal", help="generate_input dist: normal (the scored shapes) or skewed")
    ap.add_argument("--max-scan", type=int, default=20000)
    ap.add_argument("--flush", choices=("read", "write"), default="read",
                    help="read: 1 GiB sum before each replay (judge); write: 1 GiB fill_ (tools/box_moe_tl8.py)")
    ap.add_argument("--kernels", action="store_true",
                    help="also record each side's summed kernel time per input (torch.profiler, 3 cold replays)")
    ap.add_argument("--lock", default="",
                    help="flock file (e.g. /root/okc/.build/box/gpu.lock), held in-process and let go while an "
                         "extension or Triton kernel compiles; implies --ext-cache /root/.cache/bx_ext")
    ap.add_argument("--ext-cache", default="", help="content-addressed load_inline build root")
    ap.add_argument("--out", default="", help="write every replay and the summary as JSON here")
    a = ap.parse_args()

    if a.lock:
        if bx.outer_flock_holds(a.lock):
            ap.error(f"--lock {a.lock}: a parent flock(1) already holds it; drop the outer flock or --lock")
        a.ext_cache = a.ext_cache or "/root/.cache/bx_ext"
        bx.release_lock_for_triton()
        bx.BudgetClock.install()
        bx.LOCK = bx.GpuLock(a.lock)
        bx.LOCK.acquire("start")
    if a.ext_cache:
        bx.content_addressed_extensions(a.ext_cache)

    ref = bx.load_module("reference", os.path.join(PDIR, "reference.py"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    torch.cuda.synchronize()

    sides = {}
    for side, path in (("candidate", a.kernel), ("control", a.control)):
        t0 = bx.REAL_MONO()
        mod = load_submission(f"cl_{side}", path)
        state = mod.prepare(weights)
        torch.cuda.synchronize()
        sides[side] = (mod.kernel_fn, state)
        print(f"CL {side} {path} sha256 {sha(path)[:16]} import+prepare {bx.REAL_MONO() - t0:.1f} s", flush=True)

    picked, scanned, nscan = pick_inputs(ref, a.T, a.dist, a.seed, a.inputs, a.distinct, a.max_scan)
    print(f"CL T={a.T} dist={a.dist} distinct target {a.distinct}: {len(picked)} inputs from {nscan} seeds "
          f"(scanned distinct distribution {scanned})", flush=True)

    timer = WriteFlushTimer() if a.flush == "write" else bx.Timer()
    graphs = {}
    for side, (fn, state) in sides.items():
        static = ref.generate_input(bs=a.T, sharding="ep8", seed=picked[0][0], dist=a.dist)
        graphs[side] = (bx.capture(fn, static, state), static)
    ov = timer.overhead()
    ov_us = ov["overhead_ns"] / 1e3
    print(f"CL replay overhead {ov_us:.2f} us (1-kernel graph {ov['one_ns'] / 1e3:.2f}, "
          f"2-kernel {ov['two_ns'] / 1e3:.2f})", flush=True)

    rows, failed = [], []
    print("CL us; raw = CUDA events around one cold-L2 graph replay; net = raw - replay overhead; "
          "delta = median of paired (candidate - control)", flush=True)
    print(f"CL {'seed':>6} {'dist':>4} {'floor':>6} | {'cand':>7} {'iqr':>5} | {'ctl':>7} {'iqr':>5} | "
          f"{'delta':>6} {'iqr':>5} | net/floor | checks", flush=True)
    for seed, d in picked:
        fresh = ref.generate_input(bs=a.T, sharding="ep8", seed=seed, dist=a.dist)
        for _, static in graphs.values():
            bx.copy_into(static, fresh)
        torch.cuda.synchronize()
        times = {side: [] for side in graphs}
        order = list(graphs)
        for j in range(a.runs):
            for side in (order if j % 2 == 0 else order[::-1]):
                times[side].append(timer.cold(graphs[side][0][0]) / 1e3)
        checks = {}
        for side, ((_, out), _) in graphs.items():
            ok, msg = ref.check_implementation(fresh, out.clone(), weights)
            checks[side] = [bool(ok), msg]
        if not checks["candidate"][0]:
            failed.append(seed)
        ktime = {side: kernel_time_us(timer, graphs[side][0][0]) for side in graphs} if a.kernels else None
        c, k = times["candidate"], times["control"]
        deltas = [x - y for x, y in zip(c, k)]
        floor = d * FLOOR_US_PER_EXPERT
        row = dict(seed=seed, distinct=d, floor_us=floor, cand_us=c, ctl_us=k,
                   cand_med=statistics.median(c), cand_iqr=iqr(c), ctl_med=statistics.median(k), ctl_iqr=iqr(k),
                   delta_med=statistics.median(deltas), delta_iqr=iqr(deltas), checks=checks, kernels_us=ktime)
        rows.append(row)
        kt = f" | kernel {ktime['candidate']:.2f} / {ktime['control']:.2f}" if ktime else ""
        print(f"CL {seed:>6} {d:>4} {floor:6.2f} | {row['cand_med']:7.2f} {row['cand_iqr']:5.2f} | "
              f"{row['ctl_med']:7.2f} {row['ctl_iqr']:5.2f} | {row['delta_med']:+6.2f} {row['delta_iqr']:5.2f} | "
              f"{(row['cand_med'] - ov_us) / floor:9.3f} | cand {'PASS' if checks['candidate'][0] else 'FAIL'} "
              f"ctl {'PASS' if checks['control'][0] else 'FAIL'}{kt}", flush=True)
        if not checks["candidate"][0]:
            print(f"CL   candidate check: {checks['candidate'][1]}", flush=True)

    s = summarize(rows, ov_us, failed, a)
    lo, hi = s["delta_ci95"]
    print(f"CLSUM T={a.T} flush={a.flush} inputs={len(rows)} runs={a.runs} distinct {s['distinct_counts']} mean "
          f"{s['distinct_mean']:.2f} floor {s['floor_us_at_mean']:.2f} us | candidate raw {s['cand_med_raw']:.2f} "
          f"net {s['cand_med_net']:.2f} (iqr {s['cand_iqr_med']:.2f}) | control raw {s['ctl_med_raw']:.2f} net "
          f"{s['ctl_med_net']:.2f} (iqr {s['ctl_iqr_med']:.2f}) | paired delta {s['delta_med']:+.2f} us, 95% CI over "
          f"inputs [{lo:+.2f}, {hi:+.2f}], per-input delta iqr {s['delta_iqr_med']:.2f} | candidate checks "
          f"{len(rows) - len(failed)}/{len(rows)} PASS"
          + (f" | kernel time cand {s['cand_kernel_med']:.2f} ctl {s['ctl_kernel_med']:.2f}" if a.kernels else ""),
          flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(argv=sys.argv, kernel=a.kernel, kernel_sha256=sha(a.kernel), control=a.control,
                           control_sha256=sha(a.control), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                           replay_overhead=ov, floor_us_per_expert=FLOOR_US_PER_EXPERT, summary=s, rows=rows,
                           lock_events=bx.LOCK.events if bx.LOCK else None), f)
    if bx.LOCK is not None:
        bx.LOCK.release("done")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
