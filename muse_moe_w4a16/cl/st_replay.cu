// Weight-stream replay (docs/runs-decode-layer.md "2026-11-04 - Phase A stream efficiency (box)").
// Each CTA streams its own list of 36,864-byte units through a cp.async.bulk ring of depth D, the way helper0's
// skinny_ring_ws does (one issuing thread, a slot refilled as soon as the consumer releases it), with no math. The
// unit lists are built on the host, so any assignment of units to CTAs and any order can be replayed against the
// same bytes. The consumer optionally spins `delay_ns` per unit to stand in for helper0's unit loop.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

constexpr int kUnitBytes = 36864;
constexpr int kMaxDepth = 8;

struct ReplayParams {
  const uint8_t* base;          // units at base + idx * kUnitBytes
  const int* units;             // concatenated per-CTA unit lists
  const int* starts;            // [grid + 1] offsets into units
  unsigned long long* stamps;   // [grid, 4]: entry, first unit full, last unit released, smid
  unsigned long long* ready;    // [n_units] globaltimer when the consumer saw the unit full (or null)
  unsigned long long* issued;   // [n_units] globaltimer when the unit's copy was issued (or null)
  unsigned* sink;
  int depth;
  int delay_ns;
  int evict_first;
  int burn;                     // extra warps that run an integer + mma.sync loop until the stream is done (power load)
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
               :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}

__device__ __forceinline__ void burn_loop(volatile int* stop, unsigned* sink) {
  // roughly helper0's mix per weight word (several integer ops, one HMMA per 4 words) on 4 independent chains
  uint32_t r[8] = {threadIdx.x, 0x9e3779b9u, 0x7f4a7c15u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu, 0x165667b1u, 0xd3a2646cu};
  float acc[4][4] = {};
  while (!*stop) {
#pragma unroll
    for (int it = 0; it < 16; ++it) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint32_t a0 = (r[c] & 0x81C081C0u) ^ (r[c + 4] >> 3), a1 = (r[c] << 2) & 0x81C081C0u;
        uint32_t a2 = __funnelshift_l(r[c], r[c + 4], 5) & 0x81C081C0u, a3 = (r[c] * 3u) ^ a0;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(r[4]), "r"(r[5]));
        r[c] = a3 + a1;
      }
    }
  }
  unsigned o = r[0] ^ r[1] ^ r[2] ^ r[3];
#pragma unroll
  for (int c = 0; c < 4; ++c) o ^= __float_as_uint(acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3]);
  if (o == 0x12345678u) *sink = o;
}

__global__ void __launch_bounds__(512, 1) st_replay_kernel(ReplayParams p) {
  extern __shared__ __align__(1024) uint8_t ring[];
  __shared__ __align__(8) uint64_t full[kMaxDepth];
  __shared__ __align__(8) uint64_t empty[kMaxDepth];
  __shared__ int stop;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int D = p.depth;
  const int u0 = p.starts[blockIdx.x], n = p.starts[blockIdx.x + 1] - u0;
  unsigned long long* st = p.stamps + 4ull * blockIdx.x;
  if (threadIdx.x == 0) {
    st[0] = gtimer();
    for (int s = 0; s < D; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); }
    stop = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) {
      uint64_t pol = 0;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      for (int i = 0; i < n; ++i) {
        const int s = i % D;
        if (i >= D) mbar_wait(&empty[s], ((i / D) - 1) & 1);
        const uint8_t* src = p.base + static_cast<size_t>(p.units[u0 + i]) * kUnitBytes;
        if (p.issued) p.issued[u0 + i] = gtimer();
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(&full[s])), "r"(kUnitBytes) : "memory");
        if (p.evict_first)
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                       :: "r"(smem_u32(ring + s * kUnitBytes)), "l"(src), "r"(kUnitBytes), "r"(smem_u32(&full[s])), "l"(pol) : "memory");
        else
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(smem_u32(ring + s * kUnitBytes)), "l"(src), "r"(kUnitBytes), "r"(smem_u32(&full[s])) : "memory");
      }
    }
  } else if (warp >= 2) {
    burn_loop(&stop, p.sink + gridDim.x + blockIdx.x);
  } else if (lane == 0) {
    unsigned acc = 0;
    for (int i = 0; i < n; ++i) {
      const int s = i % D;
      mbar_wait(&full[s], (i / D) & 1);
      const unsigned long long t = gtimer();
      if (i == 0) st[1] = t;
      if (p.ready) p.ready[u0 + i] = t;
      acc ^= reinterpret_cast<const unsigned*>(ring + s * kUnitBytes)[(i * 977) % (kUnitBytes / 4)];
      if (p.delay_ns > 0)
        while (gtimer() - t < static_cast<unsigned long long>(p.delay_ns)) {}
      // generic reads of a TMA-filled slot are ordered before the next bulk copy into it (CL-1.1's refill race)
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(&empty[s])) : "memory");
    }
    st[2] = gtimer();
    *reinterpret_cast<volatile int*>(&stop) = 1;
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    st[3] = smid;
    p.sink[blockIdx.x] = acc;
  }
}

void launch(torch::Tensor base, torch::Tensor units, torch::Tensor starts, torch::Tensor stamps, torch::Tensor ready,
            torch::Tensor issued, torch::Tensor sink, int64_t depth, int64_t delay_ns, int64_t evict_first,
            int64_t smem, int64_t burn) {
  TORCH_CHECK(depth >= 1 && depth <= kMaxDepth && smem >= depth * kUnitBytes);
  ReplayParams p;
  p.base = base.data_ptr<uint8_t>();
  p.units = units.data_ptr<int>();
  p.starts = starts.data_ptr<int>();
  p.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>());
  p.ready = ready.numel() ? reinterpret_cast<unsigned long long*>(ready.data_ptr<int64_t>()) : nullptr;
  p.issued = issued.numel() ? reinterpret_cast<unsigned long long*>(issued.data_ptr<int64_t>()) : nullptr;
  p.sink = reinterpret_cast<unsigned*>(sink.data_ptr<int>());
  p.depth = static_cast<int>(depth);
  p.delay_ns = static_cast<int>(delay_ns);
  p.evict_first = static_cast<int>(evict_first);
  p.burn = static_cast<int>(burn);
  TORCH_CHECK(sink.numel() >= 2 * (starts.numel() - 1) && burn >= 0 && burn <= 14);
  const int grid = static_cast<int>(starts.numel()) - 1;
  TORCH_CHECK(cudaFuncSetAttribute(st_replay_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
  st_replay_kernel<<<grid, 64 + 32 * static_cast<int>(burn), smem, c10::cuda::getCurrentCUDAStream()>>>(p);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "st_replay launch: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch);
  m.attr("UNIT_BYTES") = kUnitBytes;
}
