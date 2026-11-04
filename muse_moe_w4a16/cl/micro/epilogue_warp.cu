// CL-0.6: who executes the per-tile store, fence and release (docs/plans/2026-11-04-moe-cluster-rewrite.md, 2.3 rule 1).
//
// Warps 0-7 are consumers: per step each produces one 16 x T BF16 tile with a chained mma.sync loop.
//   FORM 0 (A): tile -> double-buffered shared slot, one mbarrier arrive per warp, continue. Warp 8 (epilogue) waits,
//               stores the slot to global, __syncwarp, red.release.gpu on the CTA counter, frees the slot.
//   FORM 1 (B): each consumer stores its tile to global, bar.sync per warpgroup, thread 0 of the warpgroup issues
//               red.release.gpu, as wgs_flush kind 1 does in the current kernel.
//   FORM 2 (C): compute only, no per-tile store or release; the stall baseline.
// With nepi = 2, warps 8 and 9 are epilogue warps, each serving one slot. Warp 10 (LOAD only) streams an unrelated 1 GiB buffer through a cp.async.bulk ring so the stores and the release
// drain under HBM load, as they do in the real kernel where TMA streams weights during the epilogue.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define NCONS 8
#define NTHREADS 352
#define NT_MAX 64
#define RING 4
#define CHUNK 32768

struct Params {
  const uint32_t* seed;          // [64]
  uint32_t* out;                 // [grid][ntiles][8][16*T/2] u32
  int* cnt;                      // [grid*2]
  const uint8_t* wbuf;           // LOAD source
  unsigned long long region;     // bytes per CTA in wbuf, multiple of CHUNK
  uint32_t* stamps;              // [grid][8][NT_MAX][2]: (t_after_compute, t_after_handoff) - t_loop_start
  unsigned long long* spans;     // [grid][10][2]: per warp loop start / end, globaltimer ns
  unsigned long long* chunks;    // [grid]: LOAD chunks completed
  int ntiles;
  int kiters;
  int nepi;                      // FORM 0: epilogue warps (1 or 2; with 2, warp 8 + e serves slot e)
  int inflight;                  // LOAD: bulk chunks in flight per CTA (1..RING)
};

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.shared::cta.b64 st, [%0]; }" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* b, uint32_t tx) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1; }" ::"r"(smem_u32(b)), "r"(tx)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par) {
  asm volatile(
      "{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" ::"r"(smem_u32(b)),
      "r"(par)
      : "memory");
}
__device__ __forceinline__ void release_add(int* c) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(c), "r"(1) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

template <int T>
__device__ __forceinline__ void produce_tile(const uint32_t (&a)[4], const uint32_t (&b)[2], int kiters, int step,
                                             uint32_t (&pk)[T / 4]) {
  constexpr int NN = T / 8;
  float c[NN][4];
#pragma unroll
  for (int n = 0; n < NN; ++n) c[n][0] = c[n][1] = c[n][2] = c[n][3] = (float)(step + n);
  for (int k = 0; k < kiters; ++k) {
#pragma unroll
    for (int n = 0; n < NN; ++n)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
          "{%0,%1,%2,%3};"
          : "+f"(c[n][0]), "+f"(c[n][1]), "+f"(c[n][2]), "+f"(c[n][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0] ^ (uint32_t)n), "r"(b[1]));
  }
#pragma unroll
  for (int n = 0; n < NN; ++n) {
    pk[2 * n] = pack_bf16(c[n][0], c[n][1]);
    pk[2 * n + 1] = pack_bf16(c[n][2], c[n][3]);
  }
}

template <int FORM, int T, bool LOAD, bool STAMPS>
__global__ void __launch_bounds__(NTHREADS, 1) epi_kernel(Params p) {
  constexpr int TW = 16 * T / 2;       // u32 per warp tile
  constexpr int SLOT = NCONS * TW;     // u32 per slot
  extern __shared__ __align__(1024) uint8_t smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + 2;
  uint64_t* ringb = full + 4;
  volatile int* done = reinterpret_cast<volatile int*>(smem + 96);
  uint32_t* slots = reinterpret_cast<uint32_t*>(smem + 128);
  uint32_t* st_s = slots + 2 * SLOT;   // [8][NT_MAX][2]
  uint8_t* ring = smem + ((128 + 2 * SLOT * 4 + NCONS * NT_MAX * 8 + 1023) & ~1023);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  if (tid == 0) {
    mbar_init(&full[0], NCONS);
    mbar_init(&full[1], NCONS);
    mbar_init(&empty[0], 1);
    mbar_init(&empty[1], 1);
    for (int r = 0; r < RING; ++r) mbar_init(&ringb[r], 1);
    *done = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nt = p.ntiles;
  unsigned long long* sp = p.spans + (size_t)cta * 22;

  if (warp < NCONS) {
    uint32_t a[4], b[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = p.seed[(lane + i * 7) & 63];
    b[0] = p.seed[(lane + 3) & 63] ^ warp;
    b[1] = p.seed[(lane + 11) & 63];
    uint32_t sink = 0;
    uint32_t* gout = p.out + ((size_t)cta * nt * NCONS + warp) * TW;
    uint32_t* my_st = st_s + warp * NT_MAX * 2;
    const int wg = warp >> 2;
    const unsigned long long t0 = gtime();
    for (int i = 0; i < nt; ++i) {
      uint32_t pk[T / 4];
      produce_tile<T>(a, b, p.kiters, i, pk);
      unsigned long long tb = 0;
      if constexpr (STAMPS) tb = gtime();
      if constexpr (FORM == 0) {
        const int s = i & 1;
        if (i >= 2) mbar_wait(&empty[s], ((i >> 1) - 1) & 1);
        uint32_t* dst = slots + s * SLOT + warp * TW;
#pragma unroll
        for (int n = 0; n < T / 8; ++n)
          *reinterpret_cast<uint2*>(dst + n * 64 + lane * 2) = make_uint2(pk[2 * n], pk[2 * n + 1]);
        __syncwarp();
        if (lane == 0) mbar_arrive(&full[s]);
      } else if constexpr (FORM == 1) {
        uint32_t* dst = gout + (size_t)i * NCONS * TW;
#pragma unroll
        for (int n = 0; n < T / 8; ++n)
          *reinterpret_cast<uint2*>(dst + n * 64 + lane * 2) = make_uint2(pk[2 * n], pk[2 * n + 1]);
        asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");
        if ((tid & 127) == 0) release_add(p.cnt + cta * 2 + wg);
      } else {
#pragma unroll
        for (int n = 0; n < T / 4; ++n) sink ^= pk[n];
      }
      if constexpr (STAMPS) {
        const unsigned long long tc = gtime();
        if (lane == 0 && i < NT_MAX) {
          my_st[i * 2] = (uint32_t)(tb - t0);
          my_st[i * 2 + 1] = (uint32_t)(tc - t0);
        }
      }
    }
    const unsigned long long t1 = gtime();
    if (FORM == 2 && sink == 0x9e3779b9u) gout[lane] = sink;  // keeps the compute live
    if (lane == 0) {
      sp[warp * 2] = t0;
      sp[warp * 2 + 1] = t1;
      atomicAdd((int*)done, 1);
    }
    if constexpr (STAMPS) {
      __syncwarp();
      uint32_t* g = p.stamps + ((size_t)cta * NCONS + warp) * NT_MAX * 2;
      for (int j = lane; j < NT_MAX * 2; j += 32) g[j] = my_st[j];
    }
  } else if (warp < NCONS + 2) {
    const int e = warp - NCONS;
    if (FORM == 0 && e < p.nepi) {
      const unsigned long long t0 = gtime();
      for (int i = e; i < nt; i += p.nepi) {
        const int s = i & 1;
        mbar_wait(&full[s], (i >> 1) & 1);
        const uint4* src = reinterpret_cast<const uint4*>(slots + s * SLOT);
        uint4* dst = reinterpret_cast<uint4*>(p.out + ((size_t)cta * nt + i) * SLOT);
#pragma unroll
        for (int j = lane; j < SLOT / 4; j += 32) dst[j] = src[j];
        __syncwarp();
        if (lane == 0) {
          release_add(p.cnt + cta * 2);
          mbar_arrive(&empty[s]);
        }
      }
      if (lane == 0) {
        sp[warp * 2] = t0;
        sp[warp * 2 + 1] = gtime();
      }
    }
  } else if (LOAD && lane == 0) {
    const unsigned long long t0 = gtime();
    const uint8_t* src = p.wbuf + (size_t)cta * p.region;
    const unsigned long long nreg = p.region / CHUNK;
    const unsigned long long D = (unsigned long long)p.inflight;
    unsigned long long c = 0;
    for (;; ++c) {
      const int r = (int)(c % D);
      if (c >= D) mbar_wait(&ringb[r], (uint32_t)((c / D - 1) & 1));
      if (*done >= NCONS) break;
      mbar_arrive_tx(&ringb[r], CHUNK);
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
              smem_u32(ring + r * CHUNK)),
          "l"(src + (c % nreg) * CHUNK), "r"(CHUNK), "r"(smem_u32(&ringb[r]))
          : "memory");
    }
    // drain: chunks c-RING+1 .. c-1 are still in flight (chunk c was not issued)
    for (unsigned long long d = (c >= D ? c - D + 1 : 0); d < c; ++d)
      mbar_wait(&ringb[d % D], (uint32_t)((d / D) & 1));
    sp[20] = t0;
    sp[21] = gtime();
    p.chunks[cta] = c;
  }
}

template <int FORM, int T, bool LOAD, bool STAMPS>
static int launch_t(const Params& p, int grid, cudaStream_t s) {
  constexpr int SLOT = NCONS * 16 * T / 2;
  int sm = ((128 + 2 * SLOT * 4 + NCONS * NT_MAX * 8 + 1023) & ~1023) + (LOAD ? RING * CHUNK : 0);
  auto k = epi_kernel<FORM, T, LOAD, STAMPS>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  k<<<grid, NTHREADS, sm, s>>>(p);
  return (int)cudaGetLastError();
}

#define DISPATCH_L(F, T, L)                                                     \
  if (stamps) return launch_t<F, T, L, true>(p, grid, (cudaStream_t)stream); \
  return launch_t<F, T, L, false>(p, grid, (cudaStream_t)stream);
#define DISPATCH_T(F, T) \
  if (load) { DISPATCH_L(F, T, true) } else { DISPATCH_L(F, T, false) }
#define DISPATCH_F(F)                \
  if (T == 8) { DISPATCH_T(F, 8) } \
  if (T == 32) { DISPATCH_T(F, 32) }

extern "C" int epi_launch(int form, int T, int load, int stamps, int grid, int ntiles, int kiters, int nepi,
                          int inflight, const void* seed,
                          void* out, void* cnt, const void* wbuf, unsigned long long region, void* st, void* spans,
                          void* chunks, void* stream) {
  if ((ntiles > NT_MAX && stamps) || nepi < 1 || nepi > 2 || inflight < 1 || inflight > RING) return -1;
  Params p{(const uint32_t*)seed, (uint32_t*)out, (int*)cnt, (const uint8_t*)wbuf, region, (uint32_t*)st,
           (unsigned long long*)spans, (unsigned long long*)chunks, ntiles, kiters, nepi, inflight};
  if (form == 0) { DISPATCH_F(0) }
  if (form == 1) { DISPATCH_F(1) }
  if (form == 2) { DISPATCH_F(2) }
  return -2;
}
