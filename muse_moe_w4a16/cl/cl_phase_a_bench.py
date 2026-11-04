"""CL-1.1 phase A: build, inputs, schedule and timing (docs/plans/2026-11-04-moe-cluster-rewrite.md, task 1.1).

On the box:
  . tools/box_env.sh
  python muse_moe_w4a16/cl/cl_phase_a_bench.py --lock /root/okc/.build/box/gpu.lock \
      --out muse_moe_w4a16/cl/profiles/1.1/bench.json

The extension (cl_phase_a.cuh plus a launch wrapper) builds outside the lock. Under the lock: the problem's weights
(reference.generate_weights(WEIGHTS_SEED)), the 32 local experts' w13 packed into helper0 unit order with the public
file's own packers, then per (T, distinct) a set of inputs, each one launch captured in a CUDA graph and replayed
after a 1 GiB L2 read. Span = first CTA entry to last epilogue warp done (globaltimer stamps). The floor is the
w13 bytes of the input's distinct local experts at 4.45 TB/s (4.24 us per expert).

Inputs: reference.generate_input(bs=T, sharding="ep8", seed) (the judge's routing). "distinct = k" keeps the seeds
whose input has k distinct local experts; when --restrict is given (T = 32 has 12 to 21 distinct experts, never 7,
in seeds 1000-1399) the input keeps only its k most-routed local experts and every other local slot is moved to an
unused non-local expert id, so the k experts keep the judge's token counts.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MOE = os.path.dirname(HERE)
sys.path.insert(0, MOE)
sys.path.insert(0, HERE)

import cl_schedule  # noqa: E402

C = 8
FILL_TARGET = 15  # gate G0, D2: 15 co-resident clusters of 8 (cl_schedule.fill_target gives 128 // C = 16)
UNIT = 36864
W13_EXPERT_BYTES = 8192 * 4096 * (0.5 + 1 / 16)
FLOOR_RATE = 4.45e12
FLOOR_US = W13_EXPERT_BYTES / FLOOR_RATE * 1e6
PUBLIC = os.path.join(MOE, "cand", "current_control_release.py")
NSTAMP = 12
CLOCK_GHZ = 1.98  # SM clock during box runs (CL-0.2, CL-0.3: 1.97 to 2.07 GHz); converts the CLA_PROF sums

WRAP_CPP = """
void launch(torch::Tensor x, torch::Tensor w13u, torch::Tensor gs, torch::Tensor sched, torch::Tensor inter_g,
            torch::Tensor stamps, int64_t n_slices, int64_t t_max, int64_t pitch, int64_t slab_cols, bool use_stamps,
            int64_t slice0);
int64_t occupancy(int64_t t_max, int64_t slab_cols);
int64_t smem_bytes(int64_t t_max, int64_t slab_cols);
"""

WRAP_CU = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "cl_phase_a.cuh"

void launch(torch::Tensor x, torch::Tensor w13u, torch::Tensor gs, torch::Tensor sched, torch::Tensor inter_g,
            torch::Tensor stamps, int64_t n_slices, int64_t t_max, int64_t pitch, int64_t slab_cols, bool use_stamps,
            int64_t slice0) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && inter_g.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(sched.numel() * sched.element_size() == sizeof(ClSchedule));
  cla::PhaseAParams p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.w13u = w13u.data_ptr<uint8_t>();
  p.w13_gs = gs.data_ptr<float>();
  p.sched = reinterpret_cast<const ClSchedule*>(sched.data_ptr());
  p.inter_g = reinterpret_cast<__nv_bfloat16*>(inter_g.data_ptr());
  p.stamps = use_stamps ? reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>()) : nullptr;
  p.t_max = static_cast<int>(t_max);
  p.pitch = static_cast<int>(pitch);
  p.slice0 = static_cast<int>(slice0);
  p.zero = 0;
  const int smem = cla::smem_bytes(static_cast<int>(t_max), static_cast<int>(slab_cols));
  auto fn = cla::cl_phase_a_kernel<8>;
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
  fn<<<static_cast<unsigned>(n_slices * 8), cla::THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(p);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "cl_phase_a launch: ", cudaGetErrorString(e));
}

int64_t smem_bytes(int64_t t_max, int64_t slab_cols) {
  return cla::smem_bytes(static_cast<int>(t_max), static_cast<int>(slab_cols));
}

int64_t occupancy(int64_t t_max, int64_t slab_cols) {
  const int smem = cla::smem_bytes(static_cast<int>(t_max), static_cast<int>(slab_cols));
  auto fn = cla::cl_phase_a_kernel<8>;
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(32 * 8);
  cfg.blockDim = dim3(cla::THREADS);
  cfg.dynamicSmemBytes = smem;
  int n = -1;
  TORCH_CHECK(cudaOccupancyMaxActiveClusters(&n, fn, &cfg) == cudaSuccess);
  return n;
}
"""


class GpuLock:
    def __init__(self, path):
        self.path, self.fd = path, None

    def __enter__(self):
        if self.path:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
            t = time.monotonic()
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            print(f"LOCK acquired after {time.monotonic() - t:.1f} s", flush=True)
            print("APPS", os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
                  .read().strip(), flush=True)
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            print("LOCK released", flush=True)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(defines=()):
    """The extension. Flags are the public _ma_ext flags (-O3, sm_90a, -lineinfo, no fast-math), so cl_math's
    expressions compile as helper0's do."""
    from torch.utils.cpp_extension import load_inline
    hdrs = [os.path.join(HERE, f) for f in ("cl_phase_a.cuh", "cl_math.cuh", "cl_schedule.h")]
    tag = hashlib.md5((WRAP_CU + repr(defines) + "".join(open(h).read() for h in hdrs)).encode()).hexdigest()[:12]
    return load_inline(name=f"cl11_phase_a_{tag}", cpp_sources=WRAP_CPP, cuda_sources=WRAP_CU,
                       functions=["launch", "occupancy", "smem_bytes"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", *defines],
                       extra_include_paths=[HERE], verbose=False)


def load_public():
    spec = importlib.util.spec_from_file_location("cl11_public", PUBLIC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pack_w13(pub, weights, n_local=32):
    """[32, 512, 36864] uint8 in helper0 unit order (the public prepare's ma_w[10]) and the [32] global scales."""
    dev = weights["w13"].device
    I = 4096
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    out = []
    for e in range(n_local):
        wa = pub._pack_codes(weights["w13"][e].view(torch.uint8), rowsA)
        sa = pub._pack_scales(weights["w13_sf"][e], rowsA)
        out.append(pub._unit_order(wa, sa))
    w13u = torch.cat(out).view(n_local, 512, UNIT)
    return w13u, weights["w13_gs"][:n_local].float().contiguous()


def distinct_local(data):
    ids = data["topk_ids"]
    return int(torch.unique(ids[(ids >= 0) & (ids < 32)]).numel())


def restrict(data, k):
    """Keep the k local experts with the most pairs (ties to the lower id); move every other local slot to a
    non-local id that its row does not use."""
    ids = data["topk_ids"].clone().cpu()
    cnt = torch.bincount(ids[(ids >= 0) & (ids < 32)].long(), minlength=32)
    order = sorted(range(32), key=lambda e: (-int(cnt[e]), e))
    keep = set(e for e in order[:k] if cnt[e] > 0)
    for t in range(ids.shape[0]):
        used = set(ids[t].tolist())
        for s in range(ids.shape[1]):
            e = int(ids[t, s])
            if 0 <= e < 32 and e not in keep:
                for cand in range(32, 256):
                    if cand not in used:
                        ids[t, s] = cand
                        used.add(cand)
                        break
    out = dict(data)
    out["topk_ids"] = ids.to(data["topk_ids"].device).contiguous()
    return out


def pick_inputs(ref, T, distinct, seed, n, restrict_k=None, max_scan=4000):
    picked = []
    s = seed
    while len(picked) < n and s < seed + max_scan:
        d = ref.generate_input(bs=T, sharding="ep8", seed=s)
        if restrict_k:
            d = restrict(d, restrict_k)
        if distinct is None or distinct_local(d) == distinct:
            picked.append((s, d))
        s += 1
    if len(picked) < n:
        raise SystemExit(f"only {len(picked)} inputs with {distinct} distinct at T={T} in {max_scan} seeds")
    return picked


POLICY = "fill"
SCHED_HOT = True
NTP2_COST = 1.43  # per-unit time of a two-tile pass over a one-tile pass (CL-1.1, solo clusters: 1.16 / 0.81 us)


def _cost_slices(cnt, policy, target, _orig=cl_schedule._slices_per_expert):
    """Policy "cost" (local experiment, not in cl_schedule.py): an expert's work is the sum over its passes of 1
    (one n-tile) or NTP2_COST (two); slices go one at a time to the expert with the most work per slice."""
    if policy != "cost":
        return _orig(cnt, policy, target)
    work = []
    for c in cnt:
        ntt = (c + 7) // 8
        work.append(sum(NTP2_COST if ntt - 2 * p >= 2 else 1.0 for p in range((ntt + 1) // 2)) if c else 0.0)
    S = [1 if c else 0 for c in cnt]
    while sum(S) < target:
        cand = [e for e in range(len(cnt)) if cnt[e] and S[e] < cl_schedule.N_CHUNKS]
        if not cand:
            break
        e = max(cand, key=lambda e: (work[e] / S[e], -e))
        S[e] += 1
    return S


cl_schedule._slices_per_expert = _cost_slices


def schedule(data, policy=None):
    """ClSchedule bytes from the host mirror with the G0 fill target, as an int32 device tensor, plus its fields."""
    policy = policy or POLICY
    cl_schedule.fill_target = lambda c: FILL_TARGET
    b = cl_schedule.build(data["topk_ids"].cpu(), data["topk_weights"].cpu(), 0, 32, C, policy, 0)
    d = cl_schedule.decode(b)
    n = d["n_slices"]
    d["next_slice"] = n
    b = b[:-4] + int(n).to_bytes(4, "little", signed=True)
    t = torch.frombuffer(bytearray(b), dtype=torch.int32).to("cuda")
    return t, d


class Runner:
    def __init__(self, ext, w13u, gs, t_max=32, pitch=4096, slab_cols=512):
        self.ext, self.w13u, self.gs = ext, w13u, gs
        self.t_max, self.pitch, self.slab_cols = t_max, pitch, slab_cols
        self.inter = torch.empty(cl_schedule.MAX_SLICES * t_max * pitch, dtype=torch.bfloat16, device="cuda")
        self.stamps = torch.zeros(cl_schedule.MAX_SLICES * C * NSTAMP, dtype=torch.int64, device="cuda")

    def launch(self, x, sched_t, n_slices, stamps=True, slice0=0):
        self.ext.launch(x, self.w13u, self.gs, sched_t, self.inter, self.stamps, n_slices, self.t_max, self.pitch,
                        self.slab_cols, stamps, slice0)

    def inter_view(self, n_slices):
        return self.inter[: n_slices * self.t_max * self.pitch].view(n_slices, self.t_max, self.pitch)


def quant(xs):
    xs = sorted(xs)
    return {"median": statistics.median(xs), "min": xs[0], "max": xs[-1]}


def time_input(runner, data, flush, replays, solo=None):
    """solo: None (every slice), or 'first' / 'hot' (only slice 0 / the slice with the most tokens, alone)."""
    sched_t, d = schedule(data)
    n = 1 if solo else d["n_slices"]
    s0 = 0
    if solo == "hot":
        s0 = max(range(d["n_slices"]), key=lambda i: (d["slice_ntok"][i], -i))
    x = data["hidden_states"].contiguous()
    sched_src = sched_t.clone()
    runner.launch(x, sched_t, n, slice0=s0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            if SCHED_HOT:
                # the schedule as a prologue kernel leaves it: written just before the launch, so L2-resident
                sched_t.copy_(sched_src)
            runner.launch(x, sched_t, n, slice0=s0)
    torch.cuda.synchronize()
    rows = []
    for r in range(replays + 2):
        flush.sum()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        if r < 2:
            continue
        st = runner.stamps[: n * C * NSTAMP].view(n * C, NSTAMP).cpu().tolist()
        entry = [v[0] for v in st]
        ready = [v[1] for v in st]
        cdone = [v[2] for v in st]
        edone = [v[3] for v in st]
        t0 = min(entry)
        rows.append({
            "event_us": a.elapsed_time(b) * 1e3,
            "span_us": (max(edone) - t0) * 1e-3,
            "entry_spread_us": (max(entry) - t0) * 1e-3,
            "ramp_us": statistics.median([(ready[i] - entry[i]) * 1e-3 for i in range(len(st))]),
            "loop_us": statistics.median([(cdone[i] - ready[i]) * 1e-3 for i in range(len(st))]),
            "tail_us": statistics.median([(edone[i] - cdone[i]) * 1e-3 for i in range(len(st))]),
            "end_spread_us": (max(edone) - min(edone)) * 1e-3,
            "first_done_us": (min(edone) - t0) * 1e-3,
            "distinct_sms": len(set(v[4] for v in st)),
            **{k: statistics.median([v[j] / CLOCK_GHZ * 1e-3 for v in st]) for j, k in
               ((5, "c_full_wait_us"), (6, "c_red_wait_us"), (7, "c_loop_us"), (8, "e_busy_us"), (9, "e_wait_us"))},
            "sm_ghz": statistics.median([(v[11] - v[10]) / max(1, v[3] - v[0]) for v in st]),
            "per_cluster_span_us": [(max(edone[c * C:(c + 1) * C]) - min(entry[c * C:(c + 1) * C])) * 1e-3
                                    for c in range(n)],
        })
    del g
    dist = sum(1 for v in d["expert_nslices"] if v > 0)
    floor = dist * FLOOR_US
    if solo:
        floor = d["slice_ncol"][s0] * 2 * 4096 * (0.5 + 1 / 16) / FLOOR_RATE * 1e6
    span = quant([r["span_us"] for r in rows])
    out = {"distinct": dist, "n_slices": n, "floor_us": floor, "span": span, "ratio": span["median"] / floor,
           "event": quant([r["event_us"] for r in rows]),
           "slice_ntok": d["slice_ntok"][:n], "slice_ncol": d["slice_ncol"][:n],
           **{k: statistics.median([r[k] for r in rows]) for k in
              ("ramp_us", "loop_us", "tail_us", "end_spread_us", "entry_spread_us", "first_done_us",
               "c_full_wait_us", "c_red_wait_us", "c_loop_us", "e_busy_us", "e_wait_us", "sm_ghz")},
           "slice0": s0,
           "distinct_sms_min": min(r["distinct_sms"] for r in rows),
           "per_cluster_span_med": [statistics.median([r["per_cluster_span_us"][c] for r in rows]) for c in range(n)],
           "rows": rows}
    return out


def main():
    global POLICY, SCHED_HOT, FILL_TARGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--configs", default="8:7,32:7r,32:n,1:n",
                    help="T:distinct list; 'Nr' = restrict to the N most-routed experts, 'n' = any distinct count")
    ap.add_argument("--inputs", type=int, default=5)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--define", action="append", default=[])
    ap.add_argument("--target", type=int, default=FILL_TARGET, help="fill / cost policy target slice count")
    ap.add_argument("--sched-cold", action="store_true",
                    help="no in-graph schedule copy before the launch: the schedule is read from HBM after the flush")
    ap.add_argument("--solo", default=None, choices=("first", "hot"),
                    help="launch one cluster alone: the first slice, or the slice with the most tokens")
    ap.add_argument("--policy", default="fill", help="cl_schedule policy: fill (target 15), s1, s2, or the local "
                                                         "experiment cost (slices by work per expert)")
    a = ap.parse_args()
    POLICY = a.policy
    FILL_TARGET = a.target
    SCHED_HOT = not a.sched_cold
    t = time.monotonic()
    ext = build(tuple(a.define))
    pub = load_public()
    import reference
    print(f"BUILD {time.monotonic() - t:.1f} s  cl_phase_a.cuh {sha(os.path.join(HERE, 'cl_phase_a.cuh'))[:16]} "
          f"defines {a.define}", flush=True)
    res = {"args": vars(a), "sha_cl_phase_a": sha(os.path.join(HERE, "cl_phase_a.cuh")), "runs": []}
    with GpuLock(a.lock):
        res["smem_bytes"] = ext.smem_bytes(32, 512)
        res["occupancy_clusters"] = ext.occupancy(32, 512)
        print(f"SMEM {res['smem_bytes']} B  max active clusters of 8: {res['occupancy_clusters']}", flush=True)
        weights = reference.generate_weights(reference.WEIGHTS_SEED)
        w13u, gs = pack_w13(pub, weights)
        del weights
        torch.cuda.empty_cache()
        runner = Runner(ext, w13u, gs)
        flush = torch.ones((1 << 30) // 4, dtype=torch.float32, device="cuda")
        for cfg in a.configs.split(","):
            T, dk = cfg.split(":")
            T = int(T)
            rk = None
            if dk == "n":
                dist = None
            elif dk.endswith("r"):
                dist = rk = int(dk[:-1])
            else:
                dist = int(dk)
            for seed, data in pick_inputs(reference, T, dist, a.seed, a.inputs, rk):
                r = time_input(runner, data, flush, a.replays, a.solo)
                r.update(T=T, seed=seed, cfg=cfg)
                res["runs"].append(r)
                print(f"RUN T={T:2d} cfg={cfg:4s} seed={seed} distinct={r['distinct']:2d} slices={r['n_slices']:2d} "
                      f"span {r['span']['median']:6.2f} [{r['span']['min']:.2f}-{r['span']['max']:.2f}] us "
                      f"floor {r['floor_us']:6.2f} ratio {r['ratio']:.3f} | event {r['event']['median']:6.2f} | "
                      f"ramp {r['ramp_us']:.2f} loop {r['loop_us']:.2f} tail {r['tail_us']:.2f} "
                      f"end-spread {r['end_spread_us']:.2f} entry-spread {r['entry_spread_us']:.2f} "
                      f"sms {r['distinct_sms_min']} ntok {sorted(set(r['slice_ntok']))}"
                      + (f" | prof c0 full-wait {r['c_full_wait_us']:.2f} red-wait {r['c_red_wait_us']:.2f} loop "
                         f"{r['c_loop_us']:.2f}, epi busy {r['e_busy_us']:.2f} wait {r['e_wait_us']:.2f} "
                         f"clock {r['sm_ghz']:.2f} GHz slice {r['slice0']} ntok {r['slice_ntok'][r['slice0']]}"
                         if r['c_loop_us'] > 0 else ""), flush=True)
            sel = [r for r in res["runs"] if r["cfg"] == cfg]
            print(f"SUM T={T} cfg={cfg} inputs={len(sel)} median ratio "
                  f"{statistics.median(r['ratio'] for r in sel):.3f} (min {min(r['ratio'] for r in sel):.3f}, "
                  f"max {max(r['ratio'] for r in sel):.3f}) median span "
                  f"{statistics.median(r['span']['median'] for r in sel):.2f} us", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
