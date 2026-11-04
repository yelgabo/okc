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
  unsigned long long* stamps;   // [grid, 8]: entry, first unit full, last unit released, smid, clock64 at first full / at done
  unsigned long long* ready;    // [n_units] globaltimer when the consumer saw the unit full (or null)
  unsigned long long* issued;   // [n_units] globaltimer when the unit's copy was issued (or null)
  unsigned* sink;
  int depth;
  int delay_ns;
  int evict_first;
  int inl;                      // 1: helper0's refill: consumer warp 1 lane 0 issues unit i + D after a named barrier
  int slot_off;                 // byte offset of the ring inside dynamic shared memory (helper0: 128-aligned)
  int cons;                     // consumer warps that each wait on the full barrier and arrive on empty (helper0: 8)
  int polc;                     // 1: the L2 hint is helper0's constant policy 0x12F0000000000000 instead of createpolicy
  int stores;                   // 1: the consumer writes 64 bf16 per unit with an evict_last hint, as helper0's epilogue
  unsigned short* scratch;
  int burn;                     // extra warps that run an integer + mma.sync loop until the stream is done (power load)
  const int* pro_src;           // prologue stand-in: pro_ld cold ints read per CTA, 6 CTA barriers over shared memory,
  int pro_ld;                   // then a spin until pro_ns + (CTA % 8) * pro_stag ns after entry (helper0's routing
  int pro_ns;                   // prologue is 3 to 8 us and CTAs enter phase A spread over a few us)
  int pro_stag;
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

__device__ __forceinline__ void issue(const ReplayParams& p, uint8_t* ring, uint64_t* full, int u0, int i, int D, uint64_t pol) {
  const int s = i % D;
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

__global__ void __launch_bounds__(512, 1) st_replay_kernel(ReplayParams p) {
  extern __shared__ __align__(1024) uint8_t ring_raw[];
  uint8_t* ring = ring_raw + p.slot_off;
  __shared__ __align__(8) uint64_t full[kMaxDepth];
  __shared__ __align__(8) uint64_t empty[kMaxDepth];
  __shared__ int stop;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int D = p.depth;
  const int u0 = p.starts[blockIdx.x], n = p.starts[blockIdx.x + 1] - u0;
  unsigned long long* st = p.stamps + 8ull * blockIdx.x;
  __shared__ int pro_tmp[512];
  if (threadIdx.x == 0) {
    st[0] = gtimer();
    for (int s = 0; s < D; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], p.cons); }
    stop = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (p.pro_ns > 0 || p.pro_ld > 0) {
    int v = 0;
    for (int i = threadIdx.x; i < p.pro_ld; i += blockDim.x) v += __ldg(p.pro_src + i);
    pro_tmp[threadIdx.x] = v;
    for (int k = 0; k < 6; ++k) { __syncthreads(); v += pro_tmp[(threadIdx.x + 37 * k) % blockDim.x]; }
    if (v == 0x7fffffff) p.sink[0] = v;
    const unsigned long long until = st[0] + (unsigned long long)p.pro_ns + (unsigned long long)(blockIdx.x % 8) * p.pro_stag;
    while (gtimer() < until) {}
    __syncthreads();
  }
  if (warp == 0) {
    if (lane == 0) {
      uint64_t pol = 0;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      if (p.polc) pol = 0x12F0000000000000ull;
      for (int i = 0; i < (p.inl ? min(n, D) : n); ++i) {
        if (i >= D) mbar_wait(&empty[i % D], ((i / D) - 1) & 1);
        issue(p, ring, full, u0, i, D, pol);
      }
    }
  } else if (warp > p.cons) {
    burn_loop(&stop, p.sink + gridDim.x + blockIdx.x);
  } else {
    unsigned acc = 0;
    for (int i = 0; i < n; ++i) {
      const int s = i % D;
      mbar_wait(&full[s], (i / D) & 1);
      const unsigned long long t = gtimer();
      if (warp == 1 && lane == 0 && i == 0) { st[1] = t; st[4] = clock64(); }
      if (warp == 1 && lane == 0 && p.ready) p.ready[u0 + i] = t;
      acc ^= reinterpret_cast<const unsigned*>(ring + s * kUnitBytes)[(i * 977 + lane) % (kUnitBytes / 4)];
      if (p.stores) {
        // helper0's phase A epilogue shape: token (lane & 3) * 2 + h, row (lane >> 2), 8 rows per unit, 4096-wide rows
        const size_t row = static_cast<size_t>((p.units[u0 + i] & 511) * 8 + (lane >> 2));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const size_t tk = (lane & 3) * 2 + h;
          asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;" :: "l"(p.scratch + tk * 4096 + row),
                       "h"((unsigned short)acc), "l"(0x14F0000000000000ull) : "memory");
        }
      }
      __syncwarp();
      if (p.delay_ns > 0)
        while (gtimer() - t < static_cast<unsigned long long>(p.delay_ns)) {}
      // generic reads of a TMA-filled slot are ordered before the next bulk copy into it (CL-1.1's refill race)
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (p.inl) {
        asm volatile("bar.sync 1, %0;" :: "r"(32 * p.cons) : "memory");
        if (warp == 1 && lane == 0 && i + D < n)
          issue(p, ring, full, u0, i + D, D, p.polc ? 0x12F0000000000000ull : 0x12F0000000000000ull);
      } else if (lane == 0) {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(&empty[s])) : "memory");
      }
    }
    if (lane != 0 || warp != 1) return;
    st[2] = gtimer();
    st[5] = clock64();
    *reinterpret_cast<volatile int*>(&stop) = 1;
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    st[3] = smid;
    p.sink[blockIdx.x] = acc;
  }
}

void launch(torch::Tensor base, torch::Tensor units, torch::Tensor starts, torch::Tensor stamps, torch::Tensor ready,
            torch::Tensor issued, torch::Tensor sink, int64_t depth, int64_t delay_ns, int64_t evict_first,
            int64_t smem, int64_t burn, int64_t polc, torch::Tensor scratch, int64_t cons, int64_t slot_off, int64_t coop, int64_t inl,
            torch::Tensor pro_src, int64_t pro_ld, int64_t pro_ns, int64_t pro_stag) {
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
  p.cons = static_cast<int>(cons);
  p.slot_off = static_cast<int>(slot_off);
  p.inl = static_cast<int>(inl);
  p.pro_src = pro_src.numel() ? pro_src.data_ptr<int>() : nullptr;
  p.pro_ld = p.pro_src ? static_cast<int>(std::min<int64_t>(pro_ld, pro_src.numel())) : 0;
  p.pro_ns = static_cast<int>(pro_ns);
  p.pro_stag = static_cast<int>(pro_stag);
  TORCH_CHECK(stamps.numel() >= 8 * (starts.numel() - 1));
  TORCH_CHECK(slot_off % 16 == 0 && smem >= depth * kUnitBytes + slot_off);
  p.polc = static_cast<int>(polc);
  p.stores = scratch.numel() >= 8 * 4096 ? 1 : 0;
  p.scratch = p.stores ? reinterpret_cast<unsigned short*>(scratch.data_ptr()) : nullptr;
  TORCH_CHECK(sink.numel() >= 2 * (starts.numel() - 1) && burn >= 0 && cons >= 1 && cons + burn <= 15);
  const int grid = static_cast<int>(starts.numel()) - 1;
  TORCH_CHECK(cudaFuncSetAttribute(st_replay_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
  cudaError_t e;
  if (coop) {
    // helper0 is launched this way (cudaLaunchCooperativeKernel)
    void* args[] = {&p};
    e = cudaLaunchCooperativeKernel((const void*)st_replay_kernel, dim3(grid), dim3(32 * static_cast<int>(1 + cons + burn)),
                                    args, (size_t)smem, c10::cuda::getCurrentCUDAStream());
  } else {
    st_replay_kernel<<<grid, 32 * static_cast<int>(1 + cons + burn), smem, c10::cuda::getCurrentCUDAStream()>>>(p);
    e = cudaGetLastError();
  }
  TORCH_CHECK(e == cudaSuccess, "st_replay launch: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch);
  m.attr("UNIT_BYTES") = kUnitBytes;
}
