"""CL-1.2 phase B bench (docs/plans/2026-11-04-moe-cluster-rewrite.md task 1.2; docs/runs-moe.md "CL-1.2 phase B").

On the box:
  . tools/box_env.sh
  python muse_moe_w4a16/cl/cl_phase_b_bench.py --lock /root/okc/.build/box/gpu.lock \
      --out muse_moe_w4a16/cl/profiles/1.2/bench.json

Builds cl_phase_b.cuh (outside the lock), then under the lock: 32 local experts of NVFP4 weights packed by the
public file's prepare (unit order, ma_w[11]), inputs from reference.generate_input with cl_bench.pick_inputs
(judge-like routing) at the requested distinct count (when fewer than --inputs seeds in --max-scan have it, the
busiest `distinct` local experts of judge-like inputs are kept and the rest moved off this GPU), the ClSchedule
with the fill rule at the decided target of 15 clusters (policy fill15; cl_schedule.build's own fill uses
128 // C = 16), a random BF16 intermediate. Each mode is one kernel launch captured in a CUDA graph; each replay
follows a 1 GiB L2 read and gets its own input's schedule and intermediate copied into the graph's buffers. Span is
the in-kernel globaltimer span (first CTA entry to last epilogue store issued); from-slab starts at the first CTA's
slab loaded instead. --sass-check prints the fence census of the consumer path.
"""
import argparse
import fcntl
import hashlib
import json
import os
import statistics
import struct
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEM = os.path.dirname(HERE)
sys.path.insert(0, PROBLEM)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import cl_schedule  # noqa: E402

C = 8
W2_EXPERT_BYTES = 4096 * 4096 * (0.5 + 1 / 16)   # 9,437,184 B
FLOOR_RATE = 4.45e12                              # box 4 GiB sum rate (CL-0.1)
FLOOR_US_PER_EXPERT = W2_EXPERT_BYTES / FLOOR_RATE * 1e6
NSTAMP = 10
MODES = {
    # name: (xmode, order, rot)
    "b_rot1": (0, 1, 1),     # D1 (b), rotated, CTA c starts at peer c + 1 (the task's statement)
    "b_rot0": (0, 1, 0),     # D1 (b), rotated, own block first
    "b_nat": (0, 0, 1),      # D1 (b), blocks in K order (bitwise order), pieces rotated
    "c_rot1": (1, 1, 1),     # exchange (c): global write, cluster barrier, L2 read-back
    "none": (2, 1, 1),       # no exchange: stream and compute only (control; numerics garbage)
    "stream": (2, 1, 1, 1),  # no exchange, consumers skip the math: the ring's delivery rate alone
    "compute": (2, 1, 1, 2), # no exchange, producer skips the loads: the consumers' math rate alone
}


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
                  .read().strip() or "none", flush=True)
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            print("LOCK released", flush=True)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(verbose=False):
    from torch.utils.cpp_extension import load
    deps = [os.path.join(HERE, f) for f in ("cl_phase_b.cuh", "cl_math.cuh", "cl_schedule.h")]
    tag = hashlib.md5(b"".join(open(f, "rb").read() for f in deps)).hexdigest()[:12]
    bdir = os.path.expanduser(f"~/.cache/clb_ext/{tag}")
    os.makedirs(bdir, exist_ok=True)
    src = os.path.join(bdir, "clb.cu")
    with open(src, "w") as f:
        f.write('#define CLB_WITH_TORCH\n#include "cl_phase_b.cuh"\n')
    # The helper0 flags (-O3, sm_90a, no fast-math, no ftz): cl_math.cuh's bitwise claim depends on them.
    return load(name=f"clb_{tag}", sources=[src], build_directory=bdir, extra_include_paths=[HERE],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   "-Xptxas", "-v"], verbose=verbose)


def sass_check():
    """Builds cl_phase_b.cuh with the consumer body out of line (-DCLB_CONS_NOINLINE) and prints, per function, the
    count of fence-like SASS instructions: the consumer functions must show none."""
    import re
    import subprocess
    from collections import Counter
    from torch.utils.cpp_extension import load
    deps = [os.path.join(HERE, f) for f in ("cl_phase_b.cuh", "cl_math.cuh", "cl_schedule.h")]
    tag = hashlib.md5(b"noinline" + b"".join(open(f, "rb").read() for f in deps)).hexdigest()[:12]
    bdir = os.path.expanduser(f"~/.cache/clb_ext/chk_{tag}")
    os.makedirs(bdir, exist_ok=True)
    src = os.path.join(bdir, "clbchk.cu")
    with open(src, "w") as f:
        f.write('#define CLB_WITH_TORCH\n#define CLB_CONS_NOINLINE\n#include "cl_phase_b.cuh"\n')
    load(name=f"clbchk_{tag}", sources=[src], build_directory=bdir, extra_include_paths=[HERE],
         extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    so = [f for f in os.listdir(bdir) if f.endswith(".so")][0]
    sass = subprocess.run(["cuobjdump", "-sass", os.path.join(bdir, so)], capture_output=True, text=True).stdout
    out = {}
    for blk in sass.split("Function : ")[1:]:
        name = blk.split("\n")[0].strip()
        ops = Counter(m.group(1) for m in re.finditer(
            r"\b(MEMBAR\.\S+|CCTL\.\S+|UCGABAR_\w+|ERRBAR|BAR\.SYNC\S*|SYNCS\.\S+|HMMA\.\S+|REDG\S*|STG\S*|ATOMG?\S*)", blk))
        kind = "consumer" if "consumer" in name else ("kernel" if "phase_b_kernel" in name else "other")
        out[name] = {"kind": kind, "ops": dict(ops)}
        print(f"SASS {kind:8s} {name[:70]}: " + ", ".join(f"{k} {v}" for k, v in sorted(ops.items())), flush=True)
    return out


def prep_weights(seed=505, keep_state=False):
    """32 local experts with per-expert global scales (run_unit_math.make_weights), packed by the public file's
    prepare. Returns (pub, wref in reference format, w2u [32*256, 36864], w2_gs [32], normw bf16 [4096]) and,
    with keep_state, the prepare state (helper0 runs need it)."""
    import run_unit_math as rum
    pub = rum.load_public()
    dev = torch.device("cuda")
    wref = rum.make_weights(seed, dev)
    local = {k: wref[k] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    local["post_expert_norm"] = wref["post_norm_w"]
    assert pub._ma_scales_prmt_exact(local["w13_sf"], local["w2_sf"]), "PRMT scale path not exact"
    st = {"ma_sc6": False, "guards": [], "dev": dev, "nsm": torch.cuda.get_device_properties(dev).multi_processor_count}
    pub._fork_native_ma_prepare(local, st)
    w = st["ma_w"]
    w2u, w2gs, normw = w[11].contiguous(), w[5].contiguous(), w[7].contiguous()
    assert w2u.numel() == 32 * 256 * 36864, w2u.shape
    if keep_state:
        return pub, wref, w2u, w2gs, normw, st
    del st, w
    torch.cuda.empty_cache()
    return pub, wref, w2u, w2gs, normw


def encode_sched(d):
    """Inverse of cl_schedule.decode."""
    out = bytearray(cl_schedule.SCHED_BYTES)
    for name, off in cl_schedule.OFF.items():
        v = d[name]
        vals = v if isinstance(v, list) else [v]
        fmt = "<%df" % len(vals) if name == "tok_gate" else "<%di" % len(vals)
        struct.pack_into(fmt, out, 4 * off, *vals)
    return bytes(out)


def sched_with_S(topk_ids, topk_w, S_of):
    """ClSchedule with a given slice count per routed expert (S_of(e, cnt) -> S), the fill policy's column split."""
    d = cl_schedule.decode(cl_schedule.build(topk_ids, topk_w, 0, 32, C, "s1", 0))
    cnt = [0] * 32
    for i in range(d["n_slices"]):
        cnt[d["slice_expert"][i]] = d["slice_ntok"][i]
    tok0 = {d["slice_expert"][i]: d["slice_tok0"][i] for i in range(d["n_slices"])}
    for k in ("slice_expert", "slice_col0", "slice_ncol", "slice_tok0", "slice_ntok"):
        d[k] = [0] * 64
    d["expert_nslices"] = [0] * 32
    if sum(S_of(e, cnt[e]) for e in range(32) if cnt[e]) > cl_schedule.MAX_SLICES:
        return None
    i = 0
    for e in range(32):
        if cnt[e] == 0:
            continue
        S = S_of(e, cnt[e])
        d["expert_nslices"][e] = S
        q, rem = divmod(cl_schedule.N_CHUNKS, S)
        for j in range(S):
            d["slice_expert"][i] = e
            d["slice_col0"][i] = cl_schedule.COL_CHUNK * (j * q + min(j, rem))
            d["slice_ncol"][i] = cl_schedule.COL_CHUNK * (q + (1 if j < rem else 0))
            d["slice_tok0"][i] = tok0[e]
            d["slice_ntok"][i] = cnt[e]
            i += 1
    d["n_slices"] = i
    d["next_slice"] = i
    return encode_sched(d)


def make_sched(topk_ids, topk_w, policy):
    """policy: s1 / s2 / fill (cl_schedule.build, fill target 128 // C = 16), fill15 (the fill rule with the
    decided target of 15 co-resident clusters of 8, through cl_schedule._slices_per_expert), sN (S = N for all)."""
    if policy == "fill15":
        d = cl_schedule.decode(cl_schedule.build(topk_ids, topk_w, 0, 32, C, "s1", 0))
        cnt = [0] * 32
        for i in range(d["n_slices"]):
            cnt[d["slice_expert"][i]] = d["slice_ntok"][i]
        S = cl_schedule._slices_per_expert(cnt, "fill", 15)
        return sched_with_S(topk_ids, topk_w, lambda e, c: S[e])
    if policy in ("s1", "s2", "fill"):
        d = cl_schedule.decode(cl_schedule.build(topk_ids, topk_w, 0, 32, C, policy, 0))
        n = d["n_slices"]
        return cl_schedule.build(topk_ids, topk_w, 0, 32, C, policy, n)
    S = int(policy[1:])           # "s4", "s8": fixed S per expert
    return sched_with_S(topk_ids, topk_w, lambda e, c: S)


def sched_tensor(b):
    return torch.frombuffer(bytearray(b), dtype=torch.int32).cuda()


class Runner:
    """Buffers and one launch of the phase B kernel for a schedule."""

    def __init__(self, ext, w2u, w2gs, normw, tmax, n_slices, slab_cols, depth=4, dbg=False, ncw=None):
        dev = torch.device("cuda")
        self.ext, self.w2u, self.w2gs, self.normw = ext, w2u, w2gs, normw
        self.tmax, self.n_slices, self.slab_cols = tmax, n_slices, slab_cols
        self.ncw = ncw if ncw is not None else (16 if tmax == 8 else 8)
        optin = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        static = 8 * 13 + 4 * C * tmax + 64
        while depth > 1 and ext.smem(tmax, depth, slab_cols) + static > optin:
            depth -= 1
        self.depth = depth
        self.smem = ext.smem(tmax, depth, slab_cols)
        self.sched = torch.zeros(cl_schedule.SCHED_WORDS, dtype=torch.int32, device=dev)
        self.inter = torch.zeros((256, 4096), dtype=torch.bfloat16, device=dev)
        self.y = torch.zeros((n_slices, tmax, 4096), dtype=torch.float32, device=dev)
        self.ss = torch.zeros((n_slices, tmax), dtype=torch.float32, device=dev)
        self.xg = torch.zeros((n_slices, tmax, 4096), dtype=torch.bfloat16, device=dev)
        self.stamps = torch.zeros((n_slices * C, NSTAMP), dtype=torch.int64, device=dev)
        if dbg:
            self.dacc = torch.zeros((n_slices, tmax, 4096), dtype=torch.float32, device=dev)
            self.dsq = torch.zeros((n_slices, tmax, 256), dtype=torch.float32, device=dev)
        else:
            self.dacc = torch.zeros(0, dtype=torch.float32, device=dev)
            self.dsq = self.dacc

    def launch(self, xmode, order, rot, diag=0):
        self.ext.launch(self.w2u, self.w2gs, self.normw, self.inter, self.sched, self.y, self.ss, self.xg,
                        self.stamps, self.dacc, self.dsq, self.n_slices, self.tmax, self.depth, order, rot, xmode,
                        self.slab_cols, self.ncw, diag)


def sched_dims(b):
    d = cl_schedule.decode(b)
    n = d["n_slices"]
    sc = max(d["slice_ncol"][i] // C for i in range(n))
    return d, n, sc


def quant(xs):
    xs = sorted(xs)
    return {"median": statistics.median(xs), "min": xs[0], "max": xs[-1]}


def stamps_summary(st):
    """st: [grid, 8] int64 list rows. Times in us relative to the first entry."""
    t0 = min(r[0] for r in st)
    end = max(r[4] for r in st)
    cons = max(r[3] for r in st)
    t1 = min(r[1] for r in st)
    return {
        "span_us": (end - t0) * 1e-3,
        "span_slab_us": (end - t1) * 1e-3,
        "slab_load_us": statistics.median((r[1] - r[0]) * 1e-3 for r in st),
        "cons_span_us": (cons - t0) * 1e-3,
        "slab_pub_us": statistics.median((r[8] - r[0]) * 1e-3 for r in st),
        "first_stage_us": statistics.median((r[2] - r[0]) * 1e-3 for r in st),
        "gather_end_us": statistics.median((r[5] - t0) * 1e-3 for r in st),
        "wait_b_med_us": statistics.median(r[6] * 1e-3 for r in st),
        "wait_b_max_us": max(r[6] * 1e-3 for r in st),
        "wait_f_med_us": statistics.median(r[7] * 1e-3 for r in st),
        "end_spread_us": (end - min(r[4] for r in st)) * 1e-3,
    }


def pick(ref, T, n, distinct, seed, max_scan):
    import cl_bench
    try:
        picked, seen, nscan = cl_bench.pick_inputs(ref, T, "normal", seed, n, distinct, max_scan)
        return [ref.generate_input(bs=T, sharding="ep8", seed=s, dist="normal") for s, _ in picked], \
            {"seeds": [s for s, _ in picked], "scanned": seen, "nscan": nscan, "synthetic": False}
    except SystemExit as e:
        print(f"PICK {e}; synthesizing", flush=True)
    # Synthetic: judge-like inputs with only their `distinct` busiest local experts kept; the other local ids are
    # moved off this GPU (+32), so the kept experts keep their judge-like token counts.
    out, seeds = [], []
    s = seed
    while len(out) < n:
        data = ref.generate_input(bs=T, sharding="ep8", seed=s, dist="normal")
        ids = data["topk_ids"]
        loc = ids[ids < 32]
        cnt = torch.bincount(loc.long(), minlength=32)
        if int((cnt > 0).sum()) >= distinct:
            keep = torch.argsort(-cnt.float(), stable=True)[:distinct]
            mask = torch.zeros(32, dtype=torch.bool, device=ids.device)
            mask[keep] = True
            local = ids < 32
            drop = local & ~mask[ids.clamp(max=31).long()]
            data["topk_ids"] = torch.where(drop, ids + 32, ids).to(torch.int32)
            out.append(data)
            seeds.append(s)
        s += 1
    return out, {"seeds": seeds, "synthetic": True}


def bench(ext, w2u, w2gs, normw, ref, T, distinct, n_inputs, replays, modes, seed, max_scan, policy="fill15",
          ncw=None, depth=4):
    inputs, meta = pick(ref, T, n_inputs, distinct, seed, max_scan)
    tmax = 8 if T <= 8 else 32
    scheds = [make_sched(d["topk_ids"], d["topk_weights"], policy) for d in inputs]
    dims = [sched_dims(b) for b in scheds]
    n_slices = {n for _, n, _ in dims}
    assert len(n_slices) == 1, n_slices
    n_slices = n_slices.pop()
    sc = max(s for _, _, s in dims)
    runner = Runner(ext, w2u, w2gs, normw, tmax, n_slices, sc, ncw=ncw, depth=depth)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    inters = [(0.05 * torch.randn((256, 4096), generator=gen, device="cuda")).to(torch.bfloat16) for _ in inputs]
    flush = torch.ones((1 << 30) // 4, dtype=torch.float32, device="cuda")
    floor = distinct * FLOOR_US_PER_EXPERT
    res = {"T": T, "distinct": distinct, "policy": policy, "ncw": runner.ncw, "n_slices": n_slices, "slab_cols": sc, "depth": runner.depth,
           "smem": runner.smem, "floor_us": floor, "inputs": meta,
           "ntok": [[d["slice_ntok"][i] for i in range(n)] for d, n, _ in dims],
           "S": [[d["expert_nslices"][e] for e in range(32) if d["expert_nslices"][e]] for d, _, _ in dims],
           "modes": {}}
    print(f"BENCH T={T} distinct={distinct} policy={policy} ncw={runner.ncw} slices={n_slices} sc_max={sc} depth={runner.depth} smem={runner.smem} "
          f"floor {floor:.2f} us inputs {meta.get('seeds')} synthetic={meta['synthetic']}", flush=True)
    for name in modes:
        xmode, order, rot, diag = (MODES[name] + (0,))[:4]
        runner.sched.copy_(sched_tensor(scheds[0]))
        runner.launch(xmode, order, rot, diag)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                runner.launch(xmode, order, rot, diag)
        torch.cuda.synchronize()
        rows = []
        for k, (b, it) in enumerate(zip(scheds, inters)):
            runner.sched.copy_(sched_tensor(b))
            runner.inter.copy_(it)
            for r in range(replays + 2):
                flush.sum()
                a, e = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                g.replay()
                e.record()
                torch.cuda.synchronize()
                if r < 2:
                    continue
                row = stamps_summary(runner.stamps.cpu().tolist())
                row["event_us"] = a.elapsed_time(e) * 1e3
                row["input"] = k
                rows.append(row)
        span = quant([x["span_us"] for x in rows])
        summ = {"span": span, "ratio": span["median"] / floor,
                "event": quant([x["event_us"] for x in rows]),
                **{k: statistics.median(x[k] for x in rows)
                   for k in ("span_slab_us", "slab_load_us", "cons_span_us", "slab_pub_us", "first_stage_us", "gather_end_us", "wait_b_med_us",
                             "wait_b_max_us", "wait_f_med_us", "end_spread_us")},
                "per_input_span_med": [statistics.median(x["span_us"] for x in rows if x["input"] == k)
                                       for k in range(len(inputs))],
                "rows": rows}
        res["modes"][name] = summ
        print(f"RUN T={T:2d} {name:7s} span {span['median']:6.2f} [{span['min']:.2f}-{span['max']:.2f}] us "
              f"ratio {summ['ratio']:.3f} from-slab {summ['span_slab_us']:.2f} ({summ['span_slab_us'] / floor:.3f}) event "
              f"{summ['event']['median']:6.2f} | slab-load {summ['slab_load_us']:.2f} cons {summ['cons_span_us']:.2f} "
              f"slab-pub {summ['slab_pub_us']:.2f} first-stage {summ['first_stage_us']:.2f} gather-end "
              f"{summ['gather_end_us']:.2f} | cons wait on gather med {summ['wait_b_med_us']:.2f} max "
              f"{summ['wait_b_max_us']:.2f}, on weights {summ['wait_f_med_us']:.2f} | end spread "
              f"{summ['end_spread_us']:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--T", default="8,32")
    ap.add_argument("--distinct", type=int, default=7)
    ap.add_argument("--inputs", type=int, default=5)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--modes", default="b_rot1,b_rot0,b_nat,c_rot1,none")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--policy", default="fill15")
    ap.add_argument("--depth", default="4", help="comma list of ring depths (reduced to fit shared memory)")
    ap.add_argument("--ncw", default="", help="comma list of consumer warp counts (8, 16 at T <= 8); default per T")
    ap.add_argument("--max-scan", type=int, default=3000)
    ap.add_argument("--sass-check", action="store_true", help="only the consumer-path fence census")
    a = ap.parse_args()
    if a.sass_check:
        res = sass_check()
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
        return
    t = time.monotonic()
    ext = build()
    print(f"BUILD {time.monotonic() - t:.1f} s; sha cl_phase_b.cuh {sha(os.path.join(HERE, 'cl_phase_b.cuh'))[:16]} "
          f"bench {sha(os.path.abspath(__file__))[:16]}", flush=True)
    import reference as ref
    out = {"argv": sys.argv, "sha": {f: sha(os.path.join(HERE, f)) for f in ("cl_phase_b.cuh", "cl_phase_b_bench.py")},
           "runs": []}
    with GpuLock(a.lock):
        _, _, w2u, w2gs, normw = prep_weights()
        for T in [int(x) for x in a.T.split(",")]:
            for ncw in ([int(x) for x in a.ncw.split(",")] if a.ncw else [None]):
                for depth in [int(x) for x in a.depth.split(",")]:
                    if T > 8 and ncw == 16:
                        continue
                    out["runs"].append(bench(ext, w2u, w2gs, normw, ref, T, a.distinct, a.inputs, a.replays,
                                             a.modes.split(","), a.seed, a.max_scan, a.policy, ncw, depth))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
