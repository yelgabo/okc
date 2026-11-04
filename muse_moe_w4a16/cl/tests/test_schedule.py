"""Device ClSchedule (cl_prologue.cu) against the host mirror (cl_schedule.py), byte for byte, and the prologue timing.

  python muse_moe_w4a16/cl/tests/test_schedule.py            # 200 routings x policies, exits 1 on any mismatch
  python muse_moe_w4a16/cl/tests/test_schedule.py --bench    # prologue timing at T = 1, 8, 32 (plain graph and PDL)
"""

import argparse
import hashlib
import os
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

CL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CL)
sys.path.insert(0, os.path.dirname(CL))
import cl_schedule  # noqa: E402
import reference  # noqa: E402

CPP = """
void cl_prologue(torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor out, int64_t first, int64_t count,
                 int64_t policy, int64_t target, int64_t n_clusters, bool pdl, torch::Tensor stamps);
void cl_producer(torch::Tensor src_ids, torch::Tensor src_w, torch::Tensor dst_ids, torch::Tensor dst_w, int64_t ns,
                 torch::Tensor stamps);
void cl_empty();
"""


def build_ext():
    src = open(os.path.join(CL, "cl_prologue.cu")).read()
    hdr = open(os.path.join(CL, "cl_schedule.h")).read()
    tag = hashlib.sha256((src + hdr).encode()).hexdigest()[:12]
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    return load_inline(name=f"cl_prologue_{tag}", cpp_sources=CPP, cuda_sources=src,
                       functions=["cl_prologue", "cl_producer", "cl_empty"], extra_include_paths=[CL],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)


def run_device(ext, ids, w, first, count, C, policy, n_clusters):
    out = torch.full((cl_schedule.SCHED_WORDS,), 0x7EADBEEF, dtype=torch.int32, device="cuda")
    ext.cl_prologue(ids, w, out, first, count, cl_schedule.POLICIES[policy], cl_schedule.fill_target(C), n_clusters,
                    False, torch.empty(0, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()
    return out.cpu().numpy().tobytes()


def random_routing(gen, T, n_experts=256, dup=False):
    ids = torch.rand((T, n_experts), generator=gen, device="cuda").topk(8, dim=-1).indices.to(torch.int32)
    if dup and T > 0:
        ids[:, 7] = ids[:, 0]
    w = torch.rand((T, 8), generator=gen, device="cuda")
    return ids.contiguous(), (w / w.sum(-1, keepdim=True)).float().contiguous()


def routings():
    """200 (name, topk_ids, topk_weights, first, count)."""
    out = []
    harness = [(4, "normal"), (33, "normal"), (257, "normal"), (128, "skewed"), (1, "normal")]
    for seed in (11, 12, 13, 14):
        for bs, dist in harness:
            d = reference.generate_input(bs, "ep8", seed, dist=dist)
            out.append((f"harness bs{bs} {dist} s{seed}", d["topk_ids"], d["topk_weights"], 0, 32))
    for bs in (1, 8, 16, 32):
        for seed in range(100, 120):
            d = reference.generate_input(bs, "ep8", seed)
            out.append((f"judge bs{bs} s{seed}", d["topk_ids"], d["topk_weights"], 0, 32))
    for bs in (8, 16, 32):
        for seed in range(200, 205):
            d = reference.generate_input(bs, "ep8", seed, dist="skewed")
            out.append((f"skewed bs{bs} s{seed}", d["topk_ids"], d["topk_weights"], 0, 32))
    gen = torch.Generator(device="cuda").manual_seed(7)
    ranges = [(0, 32), (32, 32), (0, 16), (224, 32), (100, 7)]
    for i in range(50):
        T = (1, 2, 3, 5, 8, 13, 16, 24, 31, 32)[i % 10]
        first, count = ranges[i % len(ranges)]
        ids, w = random_routing(gen, T, dup=(i % 17 == 3))
        out.append((f"uniform T{T} [{first},{count}) {i}", ids, w, first, count))
    # Concentrated: every slot local, so pair totals reach and pass CL_MAX_PAIRS.
    for T in (1, 8, 31, 32, 33, 40):
        ids, w = random_routing(gen, T, n_experts=32)
        out.append((f"all-local T{T}", ids, w, 0, 32))
    for T in (8, 32):
        ids = torch.arange(8, dtype=torch.int32, device="cuda").repeat(T, 1).contiguous()
        _, w = random_routing(gen, T)
        out.append((f"same-8 T{T}", ids, w, 0, 32))
    ids, w = random_routing(gen, 16)
    out.append(("no-local T16", (ids % 200 + 56).contiguous(), w, 0, 32))
    ids, w = random_routing(gen, 32)
    out.append(("one-expert-hot T32", torch.where(ids < 128, 5, ids).to(torch.int32).contiguous(), w, 0, 32))
    gen2 = torch.Generator(device="cuda").manual_seed(8)
    while len(out) < 200:
        T = int(torch.randint(1, 289, (1,), generator=gen2, device="cuda").item())
        ids, w = random_routing(gen2, T, n_experts=int(torch.randint(32, 257, (1,), generator=gen2,
                                                                       device="cuda").item()))
        out.append((f"wide T{T} {len(out)}", ids, w, 0, 32))
    assert len(out) == 200, len(out)
    return out


CONFIGS = [(C, p) for C in (4, 8, 16) for p in ("s1", "s2", "fill")]


def test(ext):
    bad = 0
    n = 0
    stats = {"overflow": 0, "max_pairs": 0, "max_slices": 0}
    for name, ids, w, first, count in routings():
        ids = ids.to(torch.int32).contiguous()
        w = w.float().contiguous()
        for C, policy in CONFIGS:
            n_clusters = 128 // C
            want = cl_schedule.build(ids.cpu(), w.cpu(), first, count, C, policy, n_clusters)
            got = run_device(ext, ids, w, first, count, C, policy, n_clusters)
            n += 1
            dw = cl_schedule.decode(want)
            if dw["n_slices"] < 0:
                stats["overflow"] += 1
            else:
                stats["max_slices"] = max(stats["max_slices"], dw["n_slices"])
                last = dw["n_slices"] - 1
                pairs = dw["slice_tok0"][last] + dw["slice_ntok"][last] if last >= 0 else 0
                stats["max_pairs"] = max(stats["max_pairs"], pairs)
            if got != want:
                bad += 1
                dg = cl_schedule.decode(got)
                diff = [k for k in dw if dw[k] != dg[k]]
                print(f"MISMATCH {name} C={C} {policy}: fields {diff}")
                for k in diff[:3]:
                    print(f"  want {k} = {dw[k]}\n  got  {k} = {dg[k]}")
    print(f"TEST cl_schedule: {n - bad}/{n} byte-identical over 200 routings x {len(CONFIGS)} (C, policy); "
          f"overflow cases {stats['overflow']}, max pairs {stats['max_pairs']}, max slices {stats['max_slices']}")
    return bad == 0


def time_graph(g, reps=50, warm=10):
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def bench(ext, rounds):
    C, policy = 8, "fill"
    target, n_clusters = cl_schedule.fill_target(C), 128 // C
    pol = cl_schedule.POLICIES[policy]
    ns = 10_000
    print(f"BENCH prologue, C={C} policy={policy}, median of 50 graph replays (CUDA events), {rounds} rounds; "
          f"producer spins {ns} ns")
    for T in (1, 8, 32):
        d = reference.generate_input(T, "ep8", 4242)
        src_ids, src_w = d["topk_ids"].contiguous(), d["topk_weights"].float().contiguous()
        ids, w = torch.zeros_like(src_ids), torch.zeros_like(src_w)
        out = torch.zeros(cl_schedule.SCHED_WORDS, dtype=torch.int32, device="cuda")
        stamps = torch.zeros(4, dtype=torch.int64, device="cuda")
        none = torch.empty(0, dtype=torch.int64, device="cuda")
        want = cl_schedule.build(src_ids.cpu(), src_w.cpu(), 0, 32, C, policy, n_clusters)

        def pro(pdl, st=none):
            ext.cl_prologue(ids, w, out, 0, 32, pol, target, n_clusters, pdl, st)

        def prod(st=none):
            ext.cl_producer(src_ids, src_w, ids, w, ns, st)

        graphs = {
            "empty": capture(ext.cl_empty),
            "pro": capture(lambda: pro(False)),
            "pro x20": capture(lambda: [pro(False) for _ in range(20)]),
            "prod": capture(prod),
            "prod+pro serial": capture(lambda: (prod(), pro(False))),
            "prod+pro pdl": capture(lambda: (prod(), pro(True))),
        }
        res = {k: [] for k in graphs}
        for _ in range(rounds):
            for k, g in graphs.items():
                res[k].append(time_graph(g))
        # Device-side stamps under PDL: entry, after wait, stored; and the producer's publish time.
        st_g = {"serial": capture(lambda: (prod(stamps), pro(False, stamps))),
                "pdl": capture(lambda: (prod(stamps), pro(True, stamps)))}
        st_res = {}
        for k, g in st_g.items():
            spans = []
            for _ in range(50):
                ids.zero_()
                out.zero_()
                g.replay()
                torch.cuda.synchronize()
                s0, s1, s2, s3 = stamps.tolist()
                spans.append((s2 - s3, s2 - s1, s1 - s0, s0 - s3))
            assert out.cpu().numpy().tobytes() == want, f"T={T} {k}: device bytes differ under the producer"
            st_res[k] = [statistics.median(x[i] for x in spans) for i in range(4)]
        med = {k: statistics.median(v) for k, v in res.items()}
        spread = {k: max(v) - min(v) for k, v in res.items()}
        print(f"BENCH T={T}: " + ", ".join(f"{k} {med[k]:.2f} (spread {spread[k]:.2f})" for k in graphs) + " us")
        print(f"BENCH T={T}: prologue alone {med['pro'] - med['empty']:.2f} us over the empty-kernel graph; "
              f"per launch in a 20-launch graph {(med['pro x20'] - med['pro']) / 19:.2f} us; "
              f"exposed behind the producer: serial {med['prod+pro serial'] - med['prod']:.2f} us, "
              f"pdl {med['prod+pro pdl'] - med['prod']:.2f} us")
        for k, (pub_to_end, wait_to_end, entry_to_wait, entry_minus_pub) in st_res.items():
            print(f"BENCH T={T} stamps {k}: producer publish -> struct stored {pub_to_end / 1000:.2f} us, "
                  f"after wait -> stored {wait_to_end / 1000:.2f} us, entry -> after wait {entry_to_wait / 1000:.2f} "
                  f"us, entry minus publish {entry_minus_pub / 1000:.2f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    ext = build_ext()
    if a.bench:
        bench(ext, a.rounds)
        return 0
    return 0 if test(ext) else 1


if __name__ == "__main__":
    sys.exit(main())
