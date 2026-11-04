"""CL-0.1 stream skeleton bench (docs/plans/2026-11-04-moe-cluster-rewrite.md, task 0.1).

On the box:
  . tools/box_env.sh
  python muse_moe_w4a16/cl/cl_stream_bench.py --lock /root/okc/.build/box/gpu.lock \
      --out muse_moe_w4a16/cl/profiles/0.1/run.json

Builds cl_stream.cu (outside the lock), then under the lock: generates the problem's weights
with reference.generate_weights(WEIGHTS_SEED) at the real shapes, packs the 32 local experts into
36,864-byte units with submission.repack_expert (the helper0 layout), and for every (C, depth, k)
captures one launch in a CUDA graph and replays it after a 1 GiB L2 read. Each replay gets its
own random set of k distinct local experts. Span is the in-kernel globaltimer span (first CTA entry
to last CTA's consumers done); the CUDA-event time of the replay is printed beside it.
"""
import argparse
import fcntl
import json
import os
import random
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEM = os.path.dirname(HERE)
sys.path.insert(0, PROBLEM)

UNIT = 36864
UNITS_PER_EXPERT = 768
EXPERT_BYTES = UNIT * UNITS_PER_EXPERT
FLOOR_RATE = 4.45e12  # box 4 GiB sum rate, docs/runs-moe.md "T = 8 MoE floor and gap"
N_SM = 132


class GpuLock:
    def __init__(self, path):
        self.path, self.fd = path, None

    def __enter__(self):
        if self.path:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
            t = time.monotonic()
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            print(f"LOCK acquired after {time.monotonic() - t:.1f} s", flush=True)
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            print("LOCK released", flush=True)


def build():
    bdir = os.path.expanduser("~/.cache/cl_stream_ext")
    os.makedirs(bdir, exist_ok=True)
    return load(name="cl_stream_ext", sources=[os.path.join(HERE, "cl_stream.cu")], build_directory=bdir,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                verbose=False)


def packed_local_experts(n_local=32):
    import reference
    import submission
    w = reference.generate_weights(reference.WEIGHTS_SEED)
    assert tuple(w["w13"].shape) == (256, 8192, 2048) and tuple(w["w2"].shape) == (256, 4096, 2048)
    assert tuple(w["w13_sf"].shape) == (256, 8192, 256) and tuple(w["w2_sf"].shape) == (256, 4096, 256)
    out = {}
    for name, kind in (("w13", 1), ("w2", 2)):
        packed = w[name][:n_local]
        sf = w[name + "_sf"][:n_local].view(torch.uint8)
        out[name] = torch.stack([submission.repack_expert(packed[e], sf[e], kind) for e in range(n_local)])
    del w
    torch.cuda.empty_cache()
    assert tuple(out["w13"].shape) == (n_local, 512, UNIT) and tuple(out["w2"].shape) == (n_local, 256, UNIT)
    return out


def read_rate(nbytes=4 << 30, reps=10):
    buf = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
    buf.fill_(1)
    ts = []
    for _ in range(reps + 2):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        buf.sum()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e-3)
    del buf
    return nbytes / statistics.median(ts[2:])


def quantiles(xs):
    xs = sorted(xs)
    return {"median": statistics.median(xs), "min": xs[0], "max": xs[-1],
            "p10": xs[len(xs) // 10], "p90": xs[(9 * len(xs)) // 10]}


def run_config(ext, W, flush, C, depth, k, S, cons_warps, mode, evict_first, pad, replays, rng, persist=0, consume=0):
    n_slices = k * S
    grid = (persist or n_slices) * C
    threads = 32 * (1 + cons_warps)
    bounds = [round(i * UNITS_PER_EXPERT / S) for i in range(S + 1)]
    u0 = torch.tensor([bounds[i % S] for i in range(n_slices)], dtype=torch.int32, device="cuda")
    u1 = torch.tensor([bounds[i % S + 1] for i in range(n_slices)], dtype=torch.int32, device="cuda")
    sexp = torch.zeros(n_slices, dtype=torch.int32, device="cuda")
    stamps = torch.zeros(grid, 4, dtype=torch.int64, device="cuda")
    sink = torch.zeros(grid * threads, dtype=torch.int32, device="cuda")
    slice_ctr = torch.zeros(1, dtype=torch.int32, device="cuda")

    def launch():
        slice_ctr.zero_()
        ext.launch(W["w13"], W["w2"], sexp, u0, u1, stamps, sink, slice_ctr, persist, C, depth, cons_warps, mode,
                   evict_first, pad, consume)

    launch()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            launch()
    torch.cuda.synchronize()
    rows = []
    for r in range(replays + 2):
        experts = rng.sample(range(32), k)
        sexp.copy_(torch.tensor([experts[i // S] for i in range(n_slices)], dtype=torch.int32))
        flush.sum()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        st = stamps.cpu().tolist()
        if r < 2:
            continue
        entry = [x[0] for x in st]
        ready = [x[1] for x in st]
        done = [x[2] for x in st]
        taken = [x[3] & 0xFFFFFFFF for x in st]
        smid = [x[3] >> 32 for x in st]
        assert sum(taken) == k * UNITS_PER_EXPERT, (sum(taken), k)
        t0, t1 = min(entry), max(done)
        first_done = min(done)
        late = sum(1 for e in entry if e > first_done)
        busy = [(d - e) * 1e-9 for e, d in zip(entry, done)]
        per_cta_bw = [n * UNIT / ((d - q) * 1e-9) for n, q, d in zip(taken, ready, done) if d > q and n]
        rows.append({
            "event_us": a.elapsed_time(b) * 1e3,
            "span_us": (t1 - t0) * 1e-3,
            "entry_spread_us": (max(entry) - t0) * 1e-3,
            "first_ready_us": statistics.median([(q - e) * 1e-3 for e, q in zip(entry, ready)]),
            "end_spread_us": (t1 - first_done) * 1e-3,
            "late_ctas": late,
            "distinct_sms": len(set(smid)),
            "units_min": min(taken), "units_max": max(taken),
            "cta_bw_median_gbs": statistics.median(per_cta_bw) * 1e-9,
            "cta_busy_median_us": statistics.median(busy) * 1e6,
        })
    span = quantiles([x["span_us"] for x in rows])
    floor_us = k * EXPERT_BYTES / FLOOR_RATE * 1e6
    res = {
        "C": C, "depth": depth, "k": k, "S": S, "persist": persist, "consume": consume, "grid": grid, "threads": threads, "mode": mode,
        "evict_first": evict_first, "smem": depth * UNIT + pad, "replays": len(rows),
        "span": span, "event": quantiles([x["event_us"] for x in rows]),
        "floor_us": floor_us, "span_over_floor": span["median"] / floor_us,
        "achieved_tbs": k * EXPERT_BYTES / (span["median"] * 1e-6) * 1e-12,
        "per_sm_gbs": k * EXPERT_BYTES / (span["median"] * 1e-6) / grid * 1e-9,
        "cta_bw_median_gbs": statistics.median([x["cta_bw_median_gbs"] for x in rows]),
        "first_ready_us": statistics.median([x["first_ready_us"] for x in rows]),
        "end_spread_us": statistics.median([x["end_spread_us"] for x in rows]),
        "entry_spread_us": statistics.median([x["entry_spread_us"] for x in rows]),
        "late_ctas_max": max(x["late_ctas"] for x in rows),
        "distinct_sms_min": min(x["distinct_sms"] for x in rows),
        "units_min": min(x["units_min"] for x in rows), "units_max": max(x["units_max"] for x in rows),
        "rows": rows,
    }
    print(f"RUN C={C:2d} D={depth} k={k:2d} S={S:3d} P={persist:2d} R={consume} mode={mode} ef={evict_first} grid={grid:4d} "
          f"span {span['median']:7.2f} [{span['min']:.2f}-{span['max']:.2f}] us  event {res['event']['median']:7.2f}  "
          f"floor {floor_us:6.2f}  ratio {res['span_over_floor']:.3f}  {res['achieved_tbs']:.2f} TB/s  "
          f"per-SM {res['per_sm_gbs']:.1f} GB/s (CTA {res['cta_bw_median_gbs']:.1f})  ready {res['first_ready_us']:.2f}  "
          f"end-spread {res['end_spread_us']:.2f}  late {res['late_ctas_max']}  sms {res['distinct_sms_min']}  "
          f"units {res['units_min']}-{res['units_max']}", flush=True)
    return res


def occupancy_table(ext, cons_warps):
    threads = 32 * (1 + cons_warps)
    optin = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    out = []
    for C in (1, 8, 16):
        for smem in (3 * UNIT, 4 * UNIT, 5 * UNIT, 160 * 1024, optin - 1024):
            n = ext.occupancy(C, threads, smem, 32)
            out.append({"C": C, "smem": smem, "threads": threads, "max_active_clusters": n, "ctas": n * C})
            print(f"OCC C={C:2d} smem={smem:6d} threads={threads} -> max active clusters {n:3d} ({n * C} CTAs)",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--C", default="8,16")
    ap.add_argument("--k", default="3,7,9,12")
    ap.add_argument("--depth", default="3,4")
    ap.add_argument("--S", default="1", help="comma list of slices per expert; 'fill' picks S so k*S*C is about 128")
    ap.add_argument("--cons-warps", type=int, default=4)
    ap.add_argument("--mode", type=int, default=0, help="0 claim from the leader's counter, 1 static round-robin")
    ap.add_argument("--evict-first", type=int, default=1)
    ap.add_argument("--pad", type=int, default=0, help="extra dynamic shared memory per CTA in bytes")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--persist", default="0",
                    help="clusters launched that claim slices from a global counter; 0 launches one cluster per "
                         "slice, 'max' uses cudaOccupancyMaxActiveClusters at this footprint")
    ap.add_argument("--consume", default="0",
                    help="comma list; 0: consumers XOR one word per unit (the task's discard), 1: every word")
    ap.add_argument("--flush", default="float32", help="dtype of the 1 GiB L2 flush read (int32 sums slowly)")
    ap.add_argument("--occupancy-only", action="store_true")
    args = ap.parse_args()

    t = time.monotonic()
    ext = build()
    print(f"BUILD {time.monotonic() - t:.1f} s", flush=True)
    with GpuLock(args.lock):
        print("APPS", os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip(),
              flush=True)
        occ = occupancy_table(ext, args.cons_warps)
        result = {"args": vars(args), "occupancy": occ, "runs": []}
        if not args.occupancy_only:
            W = packed_local_experts()
            rate = read_rate()
            print(f"READ_RATE 4 GiB sum: {rate * 1e-12:.3f} TB/s", flush=True)
            result["session_read_tbs"] = rate * 1e-12
            flush = torch.ones((1 << 30) // 4, dtype=getattr(torch, args.flush), device="cuda")
            rng = random.Random(args.seed)
            for C in [int(x) for x in args.C.split(",")]:
                for depth in [int(x) for x in args.depth.split(",")]:
                    threads = 32 * (1 + args.cons_warps)
                    pmax = ext.occupancy(C, threads, depth * UNIT + args.pad, 32)
                    for k in [int(x) for x in args.k.split(",")]:
                        for S_arg in args.S.split(","):
                            for consume in [int(x) for x in args.consume.split(",")]:
                                S = max(1, 128 // (k * C)) if S_arg == "fill" else int(S_arg)
                                persist = pmax if args.persist == "max" else int(args.persist)
                                result["runs"].append(run_config(ext, W, flush, C, depth, k, S, args.cons_warps,
                                                                 args.mode, args.evict_first, args.pad,
                                                                 args.replays, rng, persist, consume))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
