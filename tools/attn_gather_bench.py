#!/usr/bin/env python3
"""Achievable DRAM rate for the full-context attention gather: the K and V rows of the layer's decode inputs
(muse_decode_layer/reference.generate_input, page 1: one 128-byte row per token in each of k_cache and v_cache,
pages a random permutation) read with a kernel that does nothing but keep many rows in flight.

Each CTA takes a contiguous slice of all requests' rows (the flat partition); a thread reads 16-byte chunk
(tid & 7) of rows (tid >> 3) + (threads / 8) * k, DEPTH rows of K and V per batch issued before any of them is
used, and folds the words into one register. Cold L2 (1 GiB flush) before every run, CUDA events around the
kernel. Orders: asis (the page table's rows), sorted (each CTA's rows ascending: DRAM locality, diagnostic),
seq (rows 0..N-1 regardless of the table: the streaming rate for 128-byte rows).

    python tools/attn_gather_bench.py --batch 32 --ctas 264,528,1056 --threads 256,512 --depth 4,8,16 --runs 15
"""
import argparse
import os
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = r"""
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
template <int DEPTH, int THREADS, bool KV>
__global__ void __launch_bounds__(THREADS) gather_kernel(const int* __restrict__ rows, long long n_rows,
                                                         const uint8_t* __restrict__ k, const uint8_t* __restrict__ v,
                                                         unsigned* __restrict__ sink) {
  constexpr int RPP = THREADS / 8;   // rows per pass
  const long long r0 = (long long)blockIdx.x * n_rows / gridDim.x;
  const long long r1 = (long long)(blockIdx.x + 1) * n_rows / gridDim.x;
  const int ch = threadIdx.x & 7, rr = threadIdx.x >> 3;
  unsigned acc = 0;
  for (long long base = r0 + rr; base < r1; base += (long long)RPP * DEPTH) {
    int idx[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
      const long long r = base + (long long)d * RPP;
      idx[d] = r < r1 ? __ldg(rows + r) : -1;
    }
    uint4 kk[DEPTH], vv[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
      if (idx[d] >= 0) {
        const uint8_t* pk = k + ((size_t)idx[d] << 7) + ch * 16;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(kk[d].x), "=r"(kk[d].y), "=r"(kk[d].z), "=r"(kk[d].w) : "l"(pk));
        if (KV) {
          const uint8_t* pv = v + ((size_t)idx[d] << 7) + ch * 16;
          asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(vv[d].x), "=r"(vv[d].y), "=r"(vv[d].z), "=r"(vv[d].w) : "l"(pv));
        } else {
          vv[d] = make_uint4(0, 0, 0, 0);
        }
      } else {
        kk[d] = make_uint4(0, 0, 0, 0);
        vv[d] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) acc ^= kk[d].x ^ kk[d].y ^ kk[d].z ^ kk[d].w ^ vv[d].x ^ vv[d].y ^ vv[d].z ^ vv[d].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;   // keeps the loads alive without a store on the hot path
}

template <int DEPTH, int THREADS, bool KV>
void run_one(torch::Tensor rows, torch::Tensor k, torch::Tensor v, torch::Tensor sink, int64_t ctas) {
  gather_kernel<DEPTH, THREADS, KV><<<(unsigned)ctas, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      rows.data_ptr<int>(), (long long)rows.numel(), k.data_ptr<uint8_t>(), v.data_ptr<uint8_t>(), (unsigned*)sink.data_ptr<int>());
}

void run(torch::Tensor rows, torch::Tensor k, torch::Tensor v, torch::Tensor sink, int64_t ctas, int64_t threads,
         int64_t depth, bool kv) {
#define CASE(D, T) \
  if (depth == D && threads == T) { if (kv) run_one<D, T, true>(rows, k, v, sink, ctas); else run_one<D, T, false>(rows, k, v, sink, ctas); return; }
  CASE(2, 256) CASE(4, 256) CASE(8, 256) CASE(16, 256) CASE(32, 256)
  CASE(2, 512) CASE(4, 512) CASE(8, 512) CASE(16, 512)
  CASE(4, 128) CASE(8, 128) CASE(16, 128) CASE(32, 128)
  CASE(4, 1024) CASE(8, 1024)
#undef CASE
  TORCH_CHECK(false, "no instance for depth / threads");
}
"""

CPP = "void run(torch::Tensor rows, torch::Tensor k, torch::Tensor v, torch::Tensor sink, int64_t ctas, int64_t threads, int64_t depth, bool kv);"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--ctas", default="264,528,1056")
    ap.add_argument("--threads", default="256")
    ap.add_argument("--depth", default="4,8,16")
    ap.add_argument("--orders", default="asis,sorted,seq")
    ap.add_argument("--what", default="kv,k")
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--ext-cache", default="/root/.cache/bx_ext/attn_gather_bench")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "muse_decode_layer"))
    import reference as ref  # noqa: E402

    os.makedirs(a.ext_cache, exist_ok=True)
    ext = load_inline(name="attn_gather_bench", cpp_sources=CPP, cuda_sources=SRC, functions=["run"],
                      extra_cuda_cflags=["-O3"], build_directory=a.ext_cache, verbose=False)

    data = ref.generate_input(a.batch, a.ctx, "full", a.seed, q_len=1, spread="agentic" if a.batch > 1 else "fixed",
                              dist="soft", page=1)
    k, v, pt, seq = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"], data["seq_lens"]
    assert k.stride(0) == 128, k.stride()
    rows_asis = torch.cat([pt[b, : int(seq[b])] for b in range(a.batch)]).contiguous().int()
    n = rows_asis.numel()
    print(f"batch {a.batch} ctx {a.ctx} seed {a.seed}: {n} rows, K+V {2 * n * 128 / 1e6:.1f} MB, K {n * 128 / 1e6:.1f} MB, "
          f"cache {k.numel() / 1e6:.1f} MB each")
    flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")

    def order_rows(order, ctas):
        if order == "asis":
            return rows_asis
        if order == "seq":
            return torch.arange(n, dtype=torch.int32, device="cuda")
        out = rows_asis.clone()
        for c in range(ctas):
            lo, hi = c * n // ctas, (c + 1) * n // ctas
            out[lo:hi] = torch.sort(out[lo:hi]).values
        return out

    print("order what ctas threads depth : median us  min us  TB/s(median)  bytes_in_flight/SM(KB, if all resident)")
    for what in a.what.split(","):
        kv = what == "kv"
        nbytes = n * 128 * (2 if kv else 1)
        for order in a.orders.split(","):
            for ctas in map(int, a.ctas.split(",")):
                rows = order_rows(order, ctas)
                for threads in map(int, a.threads.split(",")):
                    for depth in map(int, a.depth.split(",")):
                        ts = []
                        for _ in range(a.runs):
                            flush.zero_()
                            e0 = torch.cuda.Event(enable_timing=True)
                            e1 = torch.cuda.Event(enable_timing=True)
                            e0.record()
                            ext.run(rows, k, v, sink, ctas, threads, depth, kv)
                            e1.record()
                            torch.cuda.synchronize()
                            ts.append(e0.elapsed_time(e1) * 1e3)
                        med, mn = statistics.median(ts), min(ts)
                        per_sm = ctas / 132 * threads * 16 * depth * (2 if kv else 1) / 1024
                        print(f"{order:6s} {what:2s} {ctas:5d} {threads:4d} {depth:3d} : {med:8.2f} {mn:8.2f} "
                              f"{nbytes / med / 1e6:6.3f}  {per_sm:7.0f}", flush=True)


if __name__ == "__main__":
    main()
