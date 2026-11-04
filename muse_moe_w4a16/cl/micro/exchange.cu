// CL-0.2 exchange micro-benchmarks for decision D1 of docs/plans/2026-11-04-moe-cluster-rewrite.md.
// Build and run through micro/run.py; the binary prints one "XCH" or "LAT" line per configuration.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));     \
      exit(1);                                                                               \
    }                                                                                        \
  } while (0)

constexpr int H = 4096;
constexpr int NT = 256;
// Forces one CTA per SM, as the real kernel's weight ring does (2 x 116 KB exceeds 228 KB).
constexpr int MIN_SMEM = 116 * 1024;
constexpr int GCHUNK = 16384;
constexpr int NSLOT = 4;
constexpr int LAT_ITERS = 1000;

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
__device__ __forceinline__ uint64_t clk() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory");
  return t;
}
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_arrive_relaxed_na() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cl_wait_na() { asm volatile("barrier.cluster.wait;" ::: "memory"); }
__device__ __forceinline__ void bar_exch() { asm volatile("bar.sync 1, %0;" ::"n"(256) : "memory"); }
__device__ __forceinline__ void cl_sync() {
  cl_arrive();
  cl_wait();
}
__device__ __forceinline__ uint32_t cl_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ uint4 ld_dsm_v4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a)
               : "memory");
  return v;
}
__device__ __forceinline__ float4 ld_dsm_f4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(a)
               : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_sm_v4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_cg_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void mb_init(uint32_t a, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint32_t a) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.shared::cta.b64 s, [%0]; }" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mb_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("{ .reg .b64 s; mbarrier.arrive.expect_tx.shared::cta.b64 s, [%0], %1; }" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mb_try_wait(uint32_t a, uint32_t parity) {
  uint32_t ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok)
               : "r"(a), "r"(parity)
               : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mb_wait(uint32_t a, uint32_t parity) {
  while (!mb_try_wait(a, parity)) {
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(mbar)
               : "memory");
}
__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
  asm volatile("red.global.v4.f32.add [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void bulk_s2dsm(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "r"(src), "r"(bytes), "r"(mbar)
               : "memory");
}
__device__ __forceinline__ void xor4(uint4& a, const uint4& v) {
  a.x ^= v.x;
  a.y ^= v.y;
  a.z ^= v.z;
  a.w ^= v.w;
}

// The BF16 intermediate's content: a hash of (t, col) so every CTA's XOR checksum is predictable on the host.
__host__ __device__ __forceinline__ uint32_t elem16(int t, int col) {
  return ((uint32_t)(t * H + col) * 2654435761u) >> 16;
}
__host__ __device__ __forceinline__ uint32_t word2(int t, int col) { return elem16(t, col) | (elem16(t, col + 1) << 16); }

enum Mode { A_DIST = 0, A_ONE = 1, B_DSM = 2, B_LOCAL = 3, C_GLB = 4, C_BULK = 5, EMPTY = 6, A_L2 = 7, A_RED = 8, B_ROT = 9, B_PUSH = 10 };
static const char* mode_name[] = {"a_dist", "a_one", "b_dsmem", "b_local", "c_global", "c_bulk", "empty", "a_l2", "a_red", "b_dsmem_rot", "b_push_bw"};
static constexpr bool is_a(int m) { return m == A_DIST || m == A_ONE || m == A_L2 || m == A_RED; }

struct Args {
  float* y;                       // [ncl, T, H] reduced result of mode A
  float* part;                    // [ncl, C, T, H] FP32 partials of A_L2
  uint16_t* gint;                 // [ncl, T, H] BF16 intermediate of mode C, chunk-major
  uint4* sink;                    // [grid, NT] XOR checksums
  unsigned long long* stamps;     // [grid, 3] globaltimer t0 / after the publication barrier / t1
  int chunk_bytes;                // mode A partial chunk per CTA
  const char* lbuf;               // 1 GiB background stream source (LOAD)
  unsigned long long* lstat;      // [grid, 3] bytes streamed, globaltimer start, end (LOAD)
  int ring_off;                   // dynamic smem offset of the background stream ring
};

template <int LOAD>
__device__ __forceinline__ void xsync(int tid, volatile int& bar_seen) {
  if (LOAD && tid == 0) bar_seen = bar_seen + 1;
  cl_sync();
}

constexpr int LSLOT = 4;
constexpr size_t LREGION = size_t(8) << 20;  // per-CTA slice of lbuf; 128 slices cover 1 GiB, no L2 reuse

// LOAD = 1 adds warp 8, which streams lbuf through a 4 x 16 KB cp.async.bulk ring (discarding the data) for the
// whole exchange, the way the real kernel's weight ring keeps HBM busy while a slice exchanges. barrier.cluster
// counts every thread, so the warp joins each barrier with a split arrive.relaxed / wait and keeps the ring fed
// between them; thread 0 bumps `bar_seen` just before it arrives at each counted barrier.
__device__ __forceinline__ void lissue(uint32_t ring, uint32_t mbar, int s, const char* src, uint64_t& off) {
  mb_expect_tx(mbar + 8 * s, GCHUNK);
  bulk_g2s(ring + s * GCHUNK, src + off, GCHUNK, mbar + 8 * s);
  off = (off + GCHUNK) % LREGION;
}

template <int C, int T, int MODE, int LOAD>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(NT + 32 * LOAD, 1) xk(Args a) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ volatile int bar_seen, done;
  const uint32_t rank = cl_rank();
  const int cid = blockIdx.x / C;
  const int tid = threadIdx.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  constexpr int WOWN = H / C;              // intermediate columns owned per CTA (B, C)
  constexpr int KC = GCHUNK / (T * 2);     // columns per 16 KB BF16 chunk
  constexpr int NCH = H / KC;
  constexpr int VPC = GCHUNK / 16;         // 16-byte vectors per chunk
  static_assert(VPC % NT == 0, "vectors per chunk");
  uint64_t t0 = 0, tm = 0;

  if (MODE == B_DSM || MODE == B_LOCAL || MODE == B_ROT || MODE == B_PUSH) {
    for (int v = tid; v < T * WOWN / 8; v += NT) {
      const int t = v / (WOWN / 8), c = rank * WOWN + (v % (WOWN / 8)) * 8;
      uint4 w = make_uint4(word2(t, c), word2(t, c + 2), word2(t, c + 4), word2(t, c + 6));
      reinterpret_cast<uint4*>(sm)[v] = w;
    }
  }
  uint64_t* mbar = reinterpret_cast<uint64_t*>(sm + NSLOT * GCHUNK);
  if (MODE == C_BULK && tid == 0) {
    for (int s = 0; s < NSLOT; s++) mb_init(smem_u32(&mbar[s]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // B_PUSH is a bandwidth probe only: receive slots alias between senders, so its data is not checked.
  constexpr int SLAB = T * WOWN * 2, PIECE = SLAB < GCHUNK ? SLAB : GCHUNK, NPIECE = SLAB / PIECE, NRSLOT = 8;
  const uint32_t rslots = smem_u32(sm + 32768), rmbar = smem_u32(sm + 32768 + NRSLOT * GCHUNK);
  if (MODE == B_PUSH) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (tid == 0) {
      mb_init(rmbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      mb_expect_tx(rmbar, (C - 1) * SLAB);
    }
  }
  const uint32_t lring = smem_u32(sm + a.ring_off), lmbar = lring + LSLOT * GCHUNK;
  if (LOAD && tid == 0) {
    bar_seen = 0;
    done = 0;
    for (int s = 0; s < LSLOT; s++) mb_init(lmbar + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (LOAD && tid >= NT) {
    const int nb = 1 + (MODE == A_DIST || MODE == A_ONE ? H / (a.chunk_bytes / (T * 4))
                                                         : (MODE == B_PUSH || MODE == EMPTY ? 0 : 1));
    const int lane = tid - NT;
    const char* src = a.lbuf + (size_t)blockIdx.x * LREGION;
    uint64_t off = 0, bytes = 0, ts = gtime();
    uint32_t ph = 0;
    if (lane == 0)
      for (int s = 0; s < LSLOT; s++) lissue(lring, lmbar, s, src, off);
    int k = 0;
    bool arrived = false;
    while (true) {
      if (!arrived && k < nb) {
        cl_arrive_relaxed_na();
        arrived = true;
      }
      if (arrived && bar_seen > k) {
        cl_wait_na();
        arrived = false;
        k++;
        continue;
      }
      if (k >= nb && done) break;
      if (lane == 0)
        for (int s = 0; s < LSLOT; s++)
          if (mb_try_wait(lmbar + 8 * s, (ph >> s) & 1)) {
            ph ^= 1u << s;
            bytes += GCHUNK;
            lissue(lring, lmbar, s, src, off);
          }
    }
    if (lane == 0) {
      a.lstat[blockIdx.x * 3 + 0] = bytes;
      a.lstat[blockIdx.x * 3 + 1] = ts;
      a.lstat[blockIdx.x * 3 + 2] = gtime();
      for (int s = 0; s < LSLOT; s++) mb_wait(lmbar + 8 * s, (ph >> s) & 1);
    }
    cl_arrive_relaxed_na();
    cl_wait_na();
    return;
  }
  if (LOAD) {
    const uint64_t w = gtime();
    while (gtime() - w < 3000) {
    }
  }
  xsync<LOAD>(tid, bar_seen);
  if (tid == 0) t0 = gtime();
  if (MODE == B_PUSH) {
    const int n = (C - 1) * NPIECE;
    if (tid < n) {
      const int j = 1 + tid / NPIECE, q = tid % NPIECE;
      const uint32_t dst = (rank + j) % C;
      const uint32_t slot = rslots + (uint32_t)((rank * NPIECE + q) % NRSLOT) * GCHUNK;
      bulk_s2dsm(mapa(slot, dst), smem_u32(sm) + q * PIECE, PIECE, mapa(rmbar, dst));
    }
    mb_wait(rmbar, 0);
  }

  if (MODE == A_DIST || MODE == A_ONE) {
    const int wc = a.chunk_bytes / (T * 4);
    const int nch = H / wc;
    const int wo = wc / C;
    float* bufs[2] = {reinterpret_cast<float*>(sm), reinterpret_cast<float*>(sm) + T * wc};
    float* yc = a.y + (size_t)cid * T * H;
    for (int ch = 0; ch < nch; ch++) {
      float* buf = bufs[ch & 1];
      for (int i = tid; i < T * wc / 4; i += NT) {
        const int t = i / (wc / 4), col = ch * wc + (i % (wc / 4)) * 4;
        const float b = (float)(rank + 1 + t);
        reinterpret_cast<float4*>(buf)[i] =
            make_float4(b + (col & 63), b + ((col + 1) & 63), b + ((col + 2) & 63), b + ((col + 3) & 63));
      }
      // One barrier per chunk suffices with two buffers: a CTA arrives here only after reading chunk ch-1's peers.
      xsync<LOAD>(tid, bar_seen);
      const uint32_t base = smem_u32(buf);
      if (MODE == A_DIST) {
        const int n = T * wo / 4;
        for (int i = tid; i < n; i += NT) {
          const int t = i / (wo / 4), lc = rank * wo + (i % (wo / 4)) * 4;
          const uint32_t off = base + (uint32_t)(t * wc + lc) * 4;
          float4 v[C];
#pragma unroll
          for (int p = 0; p < C; p++) v[p] = ld_dsm_f4(mapa(off, p));
          float4 s = v[0];
#pragma unroll
          for (int p = 1; p < C; p++) s = make_float4(s.x + v[p].x, s.y + v[p].y, s.z + v[p].z, s.w + v[p].w);
          *reinterpret_cast<float4*>(yc + (size_t)t * H + ch * wc + lc) = s;
        }
      } else if (rank == 0) {
        const int n = T * wc / 4;
        for (int i = tid; i < n; i += NT) {
          const int t = i / (wc / 4), lc = (i % (wc / 4)) * 4;
          const uint32_t off = base + (uint32_t)(t * wc + lc) * 4;
          float4 v[C];
#pragma unroll
          for (int p = 0; p < C; p++) v[p] = ld_dsm_f4(mapa(off, p));
          float4 s = v[0];
#pragma unroll
          for (int p = 1; p < C; p++) s = make_float4(s.x + v[p].x, s.y + v[p].y, s.z + v[p].z, s.w + v[p].w);
          *reinterpret_cast<float4*>(yc + (size_t)t * H + ch * wc + lc) = s;
        }
      }
    }
  } else if (MODE == A_L2 || MODE == A_RED) {
    float* yc = a.y + (size_t)cid * T * H;
    float* mine = a.part + ((size_t)cid * C + rank) * T * H;
    for (int i = tid; i < T * H / 4; i += NT) {
      const int t = i / (H / 4), col = (i % (H / 4)) * 4;
      const float b = (float)(rank + 1 + t);
      const float4 v = make_float4(b + (col & 63), b + ((col + 1) & 63), b + ((col + 2) & 63), b + ((col + 3) & 63));
      if (MODE == A_RED) red_add_v4(yc + (size_t)t * H + col, v);
      else *reinterpret_cast<float4*>(mine + (size_t)t * H + col) = v;
    }
    xsync<LOAD>(tid, bar_seen);
    if (tid == 0) tm = gtime();
    if (MODE == A_L2) {
      constexpr int WO = H / C;
      const float* pc = a.part + (size_t)cid * C * T * H;
      for (int i = tid; i < T * WO / 4; i += NT) {
        const int t = i / (WO / 4), col = rank * WO + (i % (WO / 4)) * 4;
        uint4 v[C];
#pragma unroll
        for (int p = 0; p < C; p++) v[p] = ld_cg_v4(pc + ((size_t)p * T + t) * H + col);
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int p = 0; p < C; p++)
          s = make_float4(s.x + __uint_as_float(v[p].x), s.y + __uint_as_float(v[p].y), s.z + __uint_as_float(v[p].z),
                          s.w + __uint_as_float(v[p].w));
        *reinterpret_cast<float4*>(yc + (size_t)t * H + col) = s;
      }
    }
  } else if (MODE == B_DSM || MODE == B_LOCAL || MODE == B_ROT) {
    xsync<LOAD>(tid, bar_seen);  // the publication barrier at the end of phase A
    if (tid == 0) tm = gtime();
    const uint32_t base = smem_u32(sm);
    for (int ch = 0; ch < NCH; ch++) {
#pragma unroll
      for (int j = 0; j < VPC / NT; j++) {
        const int v = tid + j * NT;
        const int t = v / (KC / 8);
        int col = ch * KC + (v % (KC / 8)) * 8;
        // B_ROT starts each CTA at its own slab so the C readers do not all hit one owner at a time.
        if (MODE == B_ROT) col = (col + rank * WOWN) % H;
        const int owner = col / WOWN, lc = col % WOWN;
        const uint32_t off = base + (uint32_t)(t * WOWN + lc) * 2;
        xor4(acc, MODE == B_LOCAL ? ld_sm_v4(off) : ld_dsm_v4(mapa(off, owner)));
      }
    }
  } else if (MODE == C_GLB || MODE == C_BULK) {
    uint16_t* g = a.gint + (size_t)cid * T * H;
    for (int v = tid; v < T * WOWN / 8; v += NT) {
      const int t = v / (WOWN / 8), col = rank * WOWN + (v % (WOWN / 8)) * 8;
      const int ch = col / KC, kc = col % KC;
      uint4 w = make_uint4(word2(t, col), word2(t, col + 2), word2(t, col + 4), word2(t, col + 6));
      *reinterpret_cast<uint4*>(g + (size_t)ch * T * KC + t * KC + kc) = w;
    }
    // Generic stores feed an async-proxy bulk read in C_BULK; the writer-side proxy fence precedes the release.
    if (MODE == C_BULK) asm volatile("fence.proxy.async.global;" ::: "memory");
    xsync<LOAD>(tid, bar_seen);
    if (tid == 0) tm = gtime();
    if (MODE == C_GLB) {
      for (int ch = 0; ch < NCH; ch++) {
        const uint4* src = reinterpret_cast<const uint4*>(g + (size_t)ch * T * KC);
#pragma unroll
        for (int j = 0; j < VPC / NT; j++) xor4(acc, ld_cg_v4(src + tid + j * NT));
      }
    } else {
      if (tid == 0) {
        for (int s = 0; s < NSLOT && s < NCH; s++) {
          mb_expect_tx(smem_u32(&mbar[s]), GCHUNK);
          bulk_g2s(smem_u32(sm + s * GCHUNK), g + (size_t)s * T * KC, GCHUNK, smem_u32(&mbar[s]));
        }
      }
      for (int ch = 0; ch < NCH; ch++) {
        const int s = ch % NSLOT;
        mb_wait(smem_u32(&mbar[s]), (ch / NSLOT) & 1);
        const uint32_t sb = smem_u32(sm + s * GCHUNK);
#pragma unroll
        for (int j = 0; j < VPC / NT; j++) xor4(acc, ld_sm_v4(sb + (uint32_t)(tid + j * NT) * 16));
        bar_exch();
        if (tid == 0 && ch + NSLOT < NCH) {
          mb_expect_tx(smem_u32(&mbar[s]), GCHUNK);
          bulk_g2s(smem_u32(sm + s * GCHUNK), g + (size_t)(ch + NSLOT) * T * KC, GCHUNK, smem_u32(&mbar[s]));
        }
      }
    }
  }
  bar_exch();
  if (tid == 0) {
    const uint64_t t1 = gtime();
    if (LOAD) done = 1;
    a.stamps[blockIdx.x * 3 + 0] = t0;
    a.stamps[blockIdx.x * 3 + 1] = tm ? tm : t1;
    a.stamps[blockIdx.x * 3 + 2] = t1;
  }
  a.sink[blockIdx.x * NT + tid] = acc;
  cl_sync();  // no CTA may exit while a peer still reads its shared memory
}

// Cluster barrier round trip: every thread of every CTA runs arrive.release + wait.acquire LAT_ITERS times;
// thread 0 of each cluster's rank 0 records the clock64 delta between consecutive completions.
// SEM 0: every warp arrives with .release (SASS: MEMBAR.ALL.GPU before UCGABAR_ARV); 1: every warp .relaxed;
// 2: warp 0 .release and the other warps .relaxed, the pattern rule 1 needs (only the epilogue warp publishes).
template <int C, int SEM>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(NT, 1) bark(unsigned long long* cyc, unsigned long long* gt) {
  __shared__ unsigned int d[LAT_ITERS];
  const int cid = blockIdx.x / C;
  cl_sync();
  uint64_t g0 = gtime(), c0 = clk(), prev = c0;
  for (int i = 0; i < LAT_ITERS; i++) {
    if (SEM == 0 || (SEM == 2 && threadIdx.x < 32)) cl_arrive();
    else cl_arrive_relaxed();
    cl_wait();
    if (threadIdx.x == 0) {
      uint64_t c = clk();
      d[i] = (unsigned)(c - prev);
      prev = c;
    }
  }
  if (threadIdx.x == 0 && cl_rank() == 0) {
    uint64_t g1 = gtime();
    for (int i = 0; i < LAT_ITERS; i++) cyc[(size_t)cid * LAT_ITERS + i] = d[i];
    gt[cid * 2 + 0] = g1 - g0;
    gt[cid * 2 + 1] = prev - c0;
  }
}

// mbarrier handoff: 8 producer warps arrive on mbarrier A (ARR = 8: lane 0 of each warp; ARR = 256: every
// thread), warp 8 waits on A and records clock64 at wake minus the latest producer clock64 taken just before
// its arrive; it then arrives on B so the producers cannot run more than one phase ahead.
template <int ARR>
__global__ void __launch_bounds__(288, 1) mbk(unsigned long long* lat, unsigned long long* rt, unsigned long long* gt) {
  __shared__ __align__(8) uint64_t mA, mB;
  __shared__ long long arr[8];
  __shared__ unsigned int dl[LAT_ITERS], dr[LAT_ITERS];
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t a = smem_u32(&mA), b = smem_u32(&mB);
  if (threadIdx.x == 0) {
    mb_init(a, ARR);
    mb_init(b, 1);
  }
  __syncthreads();
  uint64_t g0 = gtime(), prevc = clk(), c0 = prevc;
  for (int i = 0; i < LAT_ITERS; i++) {
    const uint32_t par = i & 1;
    if (w < 8) {
      __syncwarp();
      if (ARR == 8) {
        if (lane == 0) {
          arr[w] = (long long)clk();
          mb_arrive(a);
        }
      } else {
        if (lane == 0) arr[w] = (long long)clk();
        __syncwarp();
        mb_arrive(a);
      }
      mb_wait(b, par);
    } else {
      mb_wait(a, par);
      const uint64_t c = clk();
      if (lane == 0) {
        long long m = arr[0];
        for (int k = 1; k < 8; k++) m = arr[k] > m ? arr[k] : m;
        dl[i] = (unsigned)((long long)c - m);
        dr[i] = (unsigned)(c - prevc);
        prevc = c;
        mb_arrive(b);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 256) {
    uint64_t g1 = gtime();
    for (int i = 0; i < LAT_ITERS; i++) {
      lat[i] = dl[i];
      rt[i] = dr[i];
    }
    gt[0] = g1 - g0;
    gt[1] = prevc - c0;
  }
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double vmin(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
static double vmax(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

struct Ctx {
  cudaStream_t s;
  uint4* flush = nullptr;
  uint4* fsink = nullptr;
  size_t fn = (size_t(1) << 30) / 16;
  int replays = 20;
};

__global__ void l2_flush(const uint4* __restrict__ p, size_t n, uint4* sink) {
  uint4 a = make_uint4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    xor4(a, ld_cg_v4(p + i));
  if ((a.x & a.y & a.z & a.w) == 0x9e3779b9u) sink[0] = a;
}

// Captures `launch` into a graph and replays it ctx.replays times, each after a 1 GiB L2 flush; returns the
// CUDA-event time of each replay in us and calls `after` after each replay (outside the timed region).
template <class L, class F>
static std::vector<double> graph_time(Ctx& ctx, L launch, F after) {
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(ctx.s, cudaStreamCaptureModeGlobal));
  launch();
  CK(cudaStreamEndCapture(ctx.s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<double> ev;
  for (int r = -2; r < ctx.replays; r++) {
    l2_flush<<<132 * 4, 512, 0, ctx.s>>>(ctx.flush, ctx.fn, ctx.fsink);
    CK(cudaEventRecord(e0, ctx.s));
    CK(cudaGraphLaunch(ge, ctx.s));
    CK(cudaEventRecord(e1, ctx.s));
    CK(cudaStreamSynchronize(ctx.s));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (r >= 0) {
      ev.push_back(ms * 1e3);
      after();
    }
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  return ev;
}

template <int C, int T, int MODE, int LOAD>
static void run_xch(Ctx& ctx, bool full, int chunk_bytes) {
  auto k = xk<C, T, MODE, LOAD>;
  int need = 32768;
  if (MODE == A_DIST || MODE == A_ONE) need = std::max(need, 2 * chunk_bytes);
  if (MODE == C_BULK) need = std::max(need, NSLOT * GCHUNK + 64);
  if (MODE == B_PUSH) need = std::max(need, 32768 + 8 * GCHUNK + 64);
  const int ring_off = (need + 1023) / 1024 * 1024;
  const int smem = std::max(MIN_SMEM, ring_off + LOAD * (LSLOT * GCHUNK + 64));
  const int nthr = NT + 32 * LOAD;
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if (C > 8) CK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  int ncl = 1;
  if (full) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(C);
    cfg.blockDim = dim3(nthr);
    cfg.dynamicSmemBytes = smem;
    int mx = 0;
    CK(cudaOccupancyMaxActiveClusters(&mx, k, &cfg));
    ncl = std::min(mx, 128 / C);
  }
  const int grid = ncl * C;
  Args a;
  a.chunk_bytes = chunk_bytes;
  a.ring_off = ring_off;
  a.lbuf = reinterpret_cast<const char*>(ctx.flush);
  CK(cudaMalloc(&a.lstat, (size_t)grid * 24));
  std::vector<unsigned long long> ls(grid * 3);
  std::vector<double> lrate;
  CK(cudaMalloc(&a.y, (size_t)ncl * T * H * 4));
  a.part = nullptr;
  if (MODE == A_L2) CK(cudaMalloc(&a.part, (size_t)ncl * C * T * H * 4));
  CK(cudaMalloc(&a.gint, (size_t)ncl * T * H * 2));
  CK(cudaMalloc(&a.sink, (size_t)grid * NT * 16));
  CK(cudaMalloc(&a.stamps, (size_t)grid * 24));
  CK(cudaMemset(a.y, 0, (size_t)ncl * T * H * 4));
  std::vector<unsigned long long> st(grid * 3);
  std::vector<double> span_med, span_max, pub_med;
  auto launch = [&]() {
    if (MODE == A_RED) CK(cudaMemsetAsync(a.y, 0, (size_t)ncl * T * H * 4, ctx.s));
    k<<<grid, nthr, smem, ctx.s>>>(a);
  };
  auto after = [&]() {
    CK(cudaMemcpy(st.data(), a.stamps, grid * 24, cudaMemcpyDeviceToHost));
    std::vector<double> sp, pb;
    for (int c = 0; c < ncl; c++) {
      unsigned long long lo = ~0ull, mid = 0, hi = 0;
      for (int r = 0; r < C; r++) {
        lo = std::min(lo, st[(c * C + r) * 3]);
        mid = std::max(mid, st[(c * C + r) * 3 + 1]);
        hi = std::max(hi, st[(c * C + r) * 3 + 2]);
      }
      sp.push_back((hi - lo) * 1e-3);
      pb.push_back((mid - lo) * 1e-3);
    }
    span_med.push_back(median(sp));
    pub_med.push_back(median(pb));
    if (LOAD) {
      CK(cudaMemcpy(ls.data(), a.lstat, grid * 24, cudaMemcpyDeviceToHost));
      double b = 0;
      unsigned long long lo = ~0ull, hi = 0;
      for (int i = 0; i < grid; i++) {
        b += ls[i * 3];
        lo = std::min(lo, ls[i * 3 + 1]);
        hi = std::max(hi, ls[i * 3 + 2]);
      }
      lrate.push_back(b / (double)(hi - lo));
    }
    span_max.push_back(vmax(sp));
  };
  std::vector<double> ev = graph_time(ctx, launch, after);
  CK(cudaGetLastError());

  int ok = 1;
  if (is_a(MODE)) {
    std::vector<float> y((size_t)ncl * T * H);
    CK(cudaMemcpy(y.data(), a.y, y.size() * 4, cudaMemcpyDeviceToHost));
    for (int c = 0; c < ncl && ok; c++)
      for (int t = 0; t < T && ok; t++)
        for (int col = 0; col < H; col++) {
          float want = (float)(C * (C + 1) / 2 + C * (t + (col & 63)));
          if (y[((size_t)c * T + t) * H + col] != want) {
            ok = 0;
            break;
          }
        }
  } else if (MODE == B_DSM || MODE == B_ROT || MODE == C_GLB || MODE == C_BULK) {
    uint32_t want[4] = {0, 0, 0, 0};
    for (int t = 0; t < T; t++)
      for (int col = 0; col < H; col += 8)
        for (int q = 0; q < 4; q++) want[q] ^= word2(t, col + 2 * q);
    std::vector<uint4> sk((size_t)grid * NT);
    CK(cudaMemcpy(sk.data(), a.sink, sk.size() * 16, cudaMemcpyDeviceToHost));
    for (int b = 0; b < grid; b++) {
      uint32_t got[4] = {0, 0, 0, 0};
      for (int i = 0; i < NT; i++) {
        uint4 v = sk[(size_t)b * NT + i];
        got[0] ^= v.x;
        got[1] ^= v.y;
        got[2] ^= v.z;
        got[3] ^= v.w;
      }
      for (int q = 0; q < 4; q++) ok &= got[q] == want[q];
    }
  } else {
    ok = -1;
  }
  printf("XCH mode=%s load=%d T=%d C=%d ncl=%d grid=%d chunk=%d ev_med=%.3f ev_min=%.3f ev_max=%.3f "
         "span_med=%.3f span_min=%.3f span_max=%.3f clmax_med=%.3f pub_med=%.3f load_gbps=%.0f ok=%d\n",
         mode_name[MODE], LOAD, T, C, ncl, grid, MODE <= A_ONE ? chunk_bytes : (is_a(MODE) || MODE == EMPTY ? 0 : GCHUNK),
         median(ev), vmin(ev), vmax(ev), median(span_med), vmin(span_med), vmax(span_med), median(span_max), median(pub_med),
         LOAD ? median(lrate) : 0.0, ok);
  fflush(stdout);
  CK(cudaFree(a.y));
  if (a.part) CK(cudaFree(a.part));
  CK(cudaFree(a.gint));
  CK(cudaFree(a.sink));
  CK(cudaFree(a.stamps));
  CK(cudaFree(a.lstat));
}

template <int C, int SEM>
static void run_bar(Ctx& ctx, bool full) {
  auto k = bark<C, SEM>;
  if (C > 8) CK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, MIN_SMEM));
  int ncl = 1;
  if (full) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(C);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = MIN_SMEM;
    int mx = 0;
    CK(cudaOccupancyMaxActiveClusters(&mx, k, &cfg));
    ncl = std::min(mx, 128 / C);
  }
  unsigned long long *cyc, *gt;
  CK(cudaMalloc(&cyc, (size_t)ncl * LAT_ITERS * 8));
  CK(cudaMalloc(&gt, (size_t)ncl * 16));
  std::vector<unsigned long long> hc((size_t)ncl * LAT_ITERS), hg(ncl * 2);
  std::vector<double> med_ns, p99_ns, ghz;
  auto launch = [&]() { k<<<ncl * C, NT, MIN_SMEM, ctx.s>>>(cyc, gt); };
  auto after = [&]() {
    CK(cudaMemcpy(hc.data(), cyc, hc.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hg.data(), gt, hg.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> all;
    for (int c = 0; c < ncl; c++) {
      double f = (double)hg[c * 2] / (double)hg[c * 2 + 1];
      ghz.push_back(1.0 / f);
      for (int i = 0; i < LAT_ITERS; i++) all.push_back(hc[(size_t)c * LAT_ITERS + i] * f);
    }
    med_ns.push_back(median(all));
    std::sort(all.begin(), all.end());
    p99_ns.push_back(all[(size_t)(all.size() * 0.99)]);
  };
  std::vector<double> ev = graph_time(ctx, launch, after);
  printf("LAT kind=cluster_barrier sem=%s C=%d ncl=%d iters=%d med_ns=%.1f med_min=%.1f med_max=%.1f p99_ns=%.1f "
         "ghz=%.3f ev_med=%.3f\n",
         SEM == 0 ? "release" : SEM == 1 ? "relaxed" : "warp0_release", C, ncl, LAT_ITERS, median(med_ns), vmin(med_ns), vmax(med_ns), median(p99_ns), median(ghz), median(ev));
  fflush(stdout);
  CK(cudaFree(cyc));
  CK(cudaFree(gt));
}

template <int ARR>
static void run_mb(Ctx& ctx) {
  unsigned long long *lat, *rt, *gt;
  CK(cudaMalloc(&lat, LAT_ITERS * 8));
  CK(cudaMalloc(&rt, LAT_ITERS * 8));
  CK(cudaMalloc(&gt, 16));
  std::vector<unsigned long long> hl(LAT_ITERS), hr(LAT_ITERS), hg(2);
  std::vector<double> lmed, rmed, lp99, ghz;
  auto launch = [&]() { mbk<ARR><<<1, 288, 0, ctx.s>>>(lat, rt, gt); };
  auto after = [&]() {
    CK(cudaMemcpy(hl.data(), lat, LAT_ITERS * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(), rt, LAT_ITERS * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hg.data(), gt, 16, cudaMemcpyDeviceToHost));
    double f = (double)hg[0] / (double)hg[1];
    ghz.push_back(1.0 / f);
    std::vector<double> l, r;
    for (int i = 1; i < LAT_ITERS; i++) {
      l.push_back(hl[i] * f);
      r.push_back(hr[i] * f);
    }
    lmed.push_back(median(l));
    rmed.push_back(median(r));
    std::sort(l.begin(), l.end());
    lp99.push_back(l[(size_t)(l.size() * 0.99)]);
  };
  graph_time(ctx, launch, after);
  printf("LAT kind=mbarrier_handoff arrivals=%d iters=%d med_ns=%.1f med_min=%.1f med_max=%.1f p99_ns=%.1f "
         "roundtrip_ns=%.1f ghz=%.3f\n",
         ARR, LAT_ITERS, median(lmed), vmin(lmed), vmax(lmed), median(lp99), median(rmed), median(ghz));
  fflush(stdout);
  CK(cudaFree(lat));
  CK(cudaFree(rt));
  CK(cudaFree(gt));
}

template <int C, int T, int LOAD>
static void run_all_modes(Ctx& ctx, bool full) {
  run_xch<C, T, EMPTY, LOAD>(ctx, full, 0);
  run_xch<C, T, A_DIST, LOAD>(ctx, full, 32768);
  run_xch<C, T, A_DIST, LOAD>(ctx, full, 65536);
  if (!LOAD) run_xch<C, T, A_ONE, LOAD>(ctx, full, 32768);
  run_xch<C, T, A_L2, LOAD>(ctx, full, 0);
  run_xch<C, T, A_RED, LOAD>(ctx, full, 0);
  if (!LOAD) run_xch<C, T, B_DSM, LOAD>(ctx, full, 0);
  run_xch<C, T, B_ROT, LOAD>(ctx, full, 0);
  run_xch<C, T, B_PUSH, LOAD>(ctx, full, 0);
  run_xch<C, T, B_LOCAL, LOAD>(ctx, full, 0);
  run_xch<C, T, C_GLB, LOAD>(ctx, full, 0);
  run_xch<C, T, C_BULK, LOAD>(ctx, full, 0);
}

int main(int argc, char** argv) {
  Ctx ctx;
  if (argc > 1) ctx.replays = atoi(argv[1]);
  CK(cudaStreamCreateWithFlags(&ctx.s, cudaStreamNonBlocking));
  CK(cudaMalloc(&ctx.flush, ctx.fn * 16));
  CK(cudaMemset(ctx.flush, 1, ctx.fn * 16));
  CK(cudaMalloc(&ctx.fsink, 16));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("DEV %s sms=%d replays=%d\n", p.name, p.multiProcessorCount, ctx.replays);
  for (int full = 0; full < 2; full++) {
    run_all_modes<8, 8, 0>(ctx, full);
    run_all_modes<8, 32, 0>(ctx, full);
    run_all_modes<16, 8, 0>(ctx, full);
    run_all_modes<16, 32, 0>(ctx, full);
    run_bar<8, 0>(ctx, full);
    run_bar<16, 0>(ctx, full);
    run_bar<8, 1>(ctx, full);
    run_bar<16, 1>(ctx, full);
    run_bar<8, 2>(ctx, full);
    run_bar<16, 2>(ctx, full);
  }
  run_all_modes<8, 8, 1>(ctx, true);
  run_all_modes<8, 32, 1>(ctx, true);
  run_all_modes<16, 8, 1>(ctx, true);
  run_all_modes<16, 32, 1>(ctx, true);
  run_mb<8>(ctx);
  run_mb<256>(ctx);
  return 0;
}
