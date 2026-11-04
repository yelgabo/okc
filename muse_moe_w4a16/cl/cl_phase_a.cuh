// CL-1.1 phase A of the MoE cluster kernel (docs/plans/2026-11-04-moe-cluster-rewrite.md, task 1.1).
//
// One cluster of C CTAs runs one slice of ClSchedule (cluster id = slice index, one launch per input). CTA c owns
// the slice's intermediate columns [col0 + c cpc, col0 + (c + 1) cpc), cpc = slice_ncol / C, which are w13 units
// rb = (col0 + c cpc) / 8 .. + cpc / 8 of the expert (unit rb = gate and up rows of columns 8 rb .. 8 rb + 7). No
// reduction across CTAs and no DSMEM traffic in phase A.
//
// Warps: 0 = TMA producer (lane 0 streams the CTA's units through a ring of CLA_DEPTH 36,864-byte slots, the
// cl_stream.cu ring), 1-2 = epilogue, 4..11 = consumers (the cl_math K split: consumer w = warp - 4 owns
// K [512 w, +512) of every unit). Work moves in unit pairs: the ring holds CLA_DEPTH / 2 pair slots with one full
// and one empty mbarrier each; a consumer computes its warp partials of both units of a pair (unit_warp_mma2: the
// cl_math chains and c0 + ((c1 + c2) + c3) combine per unit), frees the pair slot, writes the partials to one of
// two pair buffers and arrives on that buffer's mbarrier. Epilogue warp 1 + k serves the pairs j = k mod 2: it
// waits on the buffer, sums the 8 partials in warp order from 0.f (cl_math::sum_warps), applies
// cl_math::swiglu_bf16 and stores the BF16 result to the slab; at the end one bulk copy per slab row writes the
// global copy. The consumer loop has no GPU-scope fence, no bar.sync and no global store (task 0.6 rule 1): it
// synchronises by mbarrier try_wait / arrive, plus one CTA-local fence.proxy.async per pair before it frees the
// slot (see consume_pass).
// The numerics are bitwise those of helper0 (CL-0.5).
//
// x operand: the B fragments of up to two 8-token n-tiles live in consumer registers (64 registers per tile), loaded
// once per pass from global. An expert with more than 16 tokens runs ceil(ntt / 2) passes over the CTA's units
// (tokens 16 p .. 16 p + 15 in pass p); every pass but the last loads its units with an L2 evict_last hint so the
// next pass re-reads them from L2.
//
// Intermediate slab (phase B, task 1.2, reads it over DSMEM): BF16 at dynamic shared offset CLA_SLAB_OFF of every
// CTA, row-major [ntok][cpc] with row pitch cpc = slice_ncol / C (so the CTA's block is contiguous, ntok x cpc x 2
// bytes), row i = the expert's i-th routed (token, slot) pair (tok_list[slice_tok0 + i]), column j = slice column
// c cpc + j (intermediate column slice_col0 + c cpc + j). Rows ntok .. t_max - 1 are not written. Capacity
// t_max x slab_cols x 2 bytes, slab_cols >= max cpc (512 when a slice is a whole expert). The slab is complete once
// both epilogue warps have left their loops; a joined kernel orders it before phase B with a cluster barrier (every
// slab store is an epilogue warp's).
//
// Global copy: inter_g[(slice * t_max + i) * pitch + j_slice], j_slice = c cpc + j in [0, slice_ncol), pitch >= the
// largest slice_ncol, written by cp.async.bulk from the slab after the last unit (16-byte aligned rows: pitch and
// cpc are multiples of 8). No release follows (the test reads it after the kernel; exchange (c) would add one).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#include "cl_math.cuh"
#include "cl_schedule.h"

#ifndef CLA_DEPTH
#define CLA_DEPTH 4
#endif
// CLA_PROF=1: clock64 accumulators for the timeline (stamps 5..11); off in the measured build
#ifndef CLA_PROF
#define CLA_PROF 0
#endif
// CLA_EXP, timing experiments with wrong numerics: 1 drops the scale multiply, 2 drops the MMAs, 3 replaces the
// code unpack by one mask
#ifndef CLA_EXP
#define CLA_EXP 0
#endif

namespace cla {

constexpr int DEPTH = CLA_DEPTH;
constexpr int NPS = DEPTH / 2;                  // pair slots in the ring
constexpr int NPB = 2;                          // pair buffers of partials, one per epilogue warp
constexpr int NCONS = clm::NWU;                 // consumer warps = the cl_math K split
constexpr int NTP_MAX = 2;                      // n-tiles held in registers per pass
static_assert(DEPTH % 2 == 0, "the ring holds unit pairs");
// Three warpgroups. Warpgroup 0: warp 0 producer, warps 1-2 epilogue, warp 3 leaves at once; warpgroups 1-2: the
// eight consumers. The kernel launches at 168 registers per thread (384 x 168 = 64,512); warpgroup 0 gives back
// down to 56 and the consumers take 224 (4 x 56 + 8 x 224 = 2,016 per lane, the same total; 40 / 232 spills the
// epilogue). A consumer of a
// two-tile pass holds 128 registers of x, which does not fit in 168 or in the 200 a 320-thread launch would allow
// (ptxas budgets registers in whole warpgroups).
constexpr int THREADS = 384;
constexpr int CONS_WARP0 = 4;
constexpr int REG_LOW = 56, REG_HIGH = 224;
constexpr int UB = clm::UNIT_BYTES;
constexpr int RING_BYTES = DEPTH * UB;
constexpr int RED_OFF = RING_BYTES;
constexpr int RED_PAIR_F4 = 2 * NTP_MAX * NCONS * 32;   // float4 per pair buffer: [unit 2][n-tile][warp][lane]
constexpr int RED_BYTES = NPB * RED_PAIR_F4 * 16;
constexpr int CLA_SLAB_OFF = RED_OFF + RED_BYTES;
// per CTA: 0 entry, 1 first slot ready (consumer 0), 2 last consumer warp done, 3 last epilogue warp done, 4 smid;
// with CLA_PROF, clock64 sums: 5 consumer 0 waiting on full, 6 consumer 0 waiting on a pair buffer, 7 consumer 0
// loop, 8 epilogue warp 1 busy (after its wait to its arrive), 9 epilogue warp 1 waiting, 10 clock64 at entry,
// 11 clock64 at epilogue warp 1's end
constexpr int NSTAMP = 12;

struct Prof {
  long long full_wait = 0, red_wait = 0, busy = 0, wait = 0;
};
__device__ __forceinline__ long long pclock() {
#if CLA_PROF
  return clock64();
#else
  return 0;
#endif
}

__host__ __device__ constexpr int smem_bytes(int t_max, int slab_cols) { return CLA_SLAB_OFF + t_max * slab_cols * 2; }

struct PhaseAParams {
  const __nv_bfloat16* x;      // [T, 4096] hidden states
  const uint8_t* w13u;         // [32][512][36,864] helper0 unit order (public _unit_order(_pack_codes, _pack_scales))
  const float* w13_gs;         // [32]
  const ClSchedule* sched;
  __nv_bfloat16* inter_g;      // [n_slices][t_max][pitch]
  unsigned long long* stamps;  // [grid][NSTAMP] or null
  int t_max;
  int pitch;
  int slice0;                  // cluster i runs slice slice0 + i (0 except when timing one slice alone)
  uint32_t zero;               // 0 at run time; keeps the scale decode's 16 opaque to ptxas (cl_math::scd)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  unsigned ok;
  do {
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, unsigned bytes, uint64_t* bar, uint64_t policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1], %2, [%3], %4;"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}

// The CTA's share of one slice. units is a multiple of 4: slice_ncol is a multiple of 256 (cl_schedule.h), so
// cpc = slice_ncol / 8 is a multiple of 32 columns.
struct Slice {
  int expert, ntok, tok0, cpc, rb0, units, passes, ntt;
};

__device__ __forceinline__ Slice read_slice(const ClSchedule* s, int slice, int rank, int C) {
  Slice r;
  r.expert = s->slice_expert[slice];
  r.ntok = s->slice_ntok[slice];
  r.tok0 = s->slice_tok0[slice];
  const int col0 = s->slice_col0[slice], ncol = s->slice_ncol[slice];
  r.cpc = ncol / C;
  r.rb0 = (col0 + rank * r.cpc) >> 3;
  r.units = r.cpc >> 3;
  r.ntt = (r.ntok + 7) >> 3;
  r.passes = (r.ntt + NTP_MAX - 1) / NTP_MAX;
  return r;
}

// Two units at once with the per-unit arithmetic of clm::unit_warp_mma (same chains, same combine expression), so
// each unit's partial is bitwise the single-unit one.
template <int NTP, class XF>
__device__ __forceinline__ void unit_warp_mma2(const unsigned char* ua, const unsigned char* ub, int warp, int lane,
                                               uint32_t k16, XF&& xf, float (&acc)[2][NTP][4]) {
  float c1[2][NTP][4], c2[2][NTP][4], c3[2][NTP][4];
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int n = 0; n < NTP; ++n)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc[q][n][k] = 0.f; c1[q][n][k] = 0.f; c2[q][n][k] = 0.f; c3[q][n][k] = 0.f; }
  const unsigned char* un[2] = {ua, ub};
  uint4 sc[2];
#pragma unroll
  for (int q = 0; q < 2; ++q) sc[q] = clm::unit_scales(un[q], warp, lane);
#pragma unroll
  for (int c = 0; c < clm::NCH; ++c) {
    uint32_t ww[2][4], sg[2], sg8[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const uint4 w = clm::unit_codes(un[q], warp, c, lane);
      ww[q][0] = w.x; ww[q][1] = w.y; ww[q][2] = w.z; ww[q][3] = w.w;
      clm::scale_pair(sc[q], c, k16, sg[q], sg8[q]);
    }
    uint32_t b[NTP][8];
#pragma unroll
    for (int n = 0; n < NTP; ++n) {
      uint4 lo, hi;
      xf(n, c, lo, hi);
      b[n][0] = lo.x; b[n][1] = lo.y; b[n][2] = lo.z; b[n][3] = lo.w;
      b[n][4] = hi.x; b[n][5] = hi.y; b[n][6] = hi.z; b[n][7] = hi.w;
    }
#pragma unroll
    for (int st = 0; st < 4; ++st) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        uint32_t a[4];
        if (CLA_EXP == 1) {
          clm::dq4(ww[q][st], a[0], a[1], a[2], a[3]);
          a[0] ^= sg[q]; a[3] ^= sg8[q];
        } else if (CLA_EXP == 3) {
          a[0] = clm::bmul2(ww[q][st] & 0x81C081C0u, sg[q]); a[1] = clm::bmul2(ww[q][st], sg8[q]);
          a[2] = clm::bmul2(ww[q][st], sg[q]); a[3] = clm::bmul2(ww[q][st], sg8[q]);
        } else {
          clm::decode_step(ww[q][st], sg[q], sg8[q], a);
        }
#pragma unroll
        for (int n = 0; n < NTP; ++n)
          if (CLA_EXP == 2) acc[q][n][0] += __uint_as_float(a[0] ^ a[1] ^ a[2] ^ a[3] ^ b[n][2 * st] ^ b[n][2 * st + 1]);
          else clm::mma_bf16(st == 0 ? acc[q][n] : st == 1 ? c1[q][n] : st == 2 ? c2[q][n] : c3[q][n], a[0], a[1], a[2],
                             a[3], b[n][2 * st], b[n][2 * st + 1]);
      }
    }
  }
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int n = 0; n < NTP; ++n)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[q][n][k] += c1[q][n][k] + c2[q][n][k] + c3[q][n][k];
}

struct Bars {
  uint64_t* full;    // [NPS] pair slot loaded (1 arrival + 2 units of tx)
  uint64_t* empty;   // [NPS] pair slot free (8 consumer warps)
  uint64_t* rfull;   // [NPB] pair buffer written (8 consumer warps)
  uint64_t* rempty;  // [NPB] pair buffer read (its epilogue warp)
  uint64_t* first0;  // the kernel's first unit alone (1 arrival + 1 unit of tx); full[0] then covers the second
};

// One consumer pass over the CTA's units with NTP n-tiles of x in registers; j = the CTA's running pair index
// (slot and buffer parities run on across passes).
template <int NTP>
__device__ __forceinline__ void consume_pass(const PhaseAParams& p, const Slice& sl, int pass, uint8_t* ring,
                                             float4* red, const Bars& bb, int w, int lane, int& j,
                                             unsigned long long* st, Prof& pf) {
  const int g = lane >> 2;
  uint4 xr[NTP][2 * clm::NCH];
#pragma unroll
  for (int n = 0; n < NTP; ++n) {
    const int li = pass * 16 + n * 8 + g;
    if (li < sl.ntok) {
      const int tok = p.sched->tok_list[sl.tok0 + li];
      const uint4* b = clm::x_frag_base(p.x + static_cast<size_t>(tok) * clm::HD, w, lane);
#pragma unroll
      for (int c = 0; c < clm::NCH; ++c) {
        xr[n][2 * c] = __ldg(b + 8 * c);
        xr[n][2 * c + 1] = __ldg(b + 8 * c + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * clm::NCH; ++c) xr[n][c] = make_uint4(0, 0, 0, 0);
    }
  }
  const uint32_t k16 = 16u + p.zero;
  auto xf = [&](int n, int c, uint4& lo, uint4& hi) {
    lo = xr[n][2 * c];
    hi = xr[n][2 * c + 1];
  };
  for (int u = 0; u < sl.units; u += 2, ++j) {
    const int ps = j % NPS, pb = j % NPB;
    const uint8_t* slot = ring + static_cast<size_t>(ps) * 2 * UB;
    float acc[2][NTP][4];
    const long long c0 = pclock();
    if (j == 0 || NTP == 2) {
      // Single units: the kernel's first pair, so that the first wait is for one unit, not two; and every two-tile
      // pair, which keeps 32 accumulator registers live instead of 64 next to the 128 of x (solo cluster, 15-token
      // slice: 41.3 against 47.9 us paired)
      if (j == 0) {
        mbar_wait(bb.first0, 0);
        if (st && w == 0 && lane == 0) st[1] = gtimer();
      } else {
        mbar_wait(&bb.full[ps], (j / NPS) & 1);
      }
      if (CLA_PROF) pf.full_wait += pclock() - c0;
      clm::unit_warp_mma<NTP>(slot, w, lane, k16, xf, acc[0]);
      const long long c2 = pclock();
      if (j == 0) mbar_wait(&bb.full[0], 0);
      if (CLA_PROF) pf.full_wait += pclock() - c2;
      clm::unit_warp_mma<NTP>(slot + UB, w, lane, k16, xf, acc[1]);
    } else {
      mbar_wait(&bb.full[ps], (j / NPS) & 1);
      if (CLA_PROF) pf.full_wait += pclock() - c0;
      unit_warp_mma2<NTP>(slot, slot + UB, w, lane, k16, xf, acc);
    }
    // The slot's next contents come from the async proxy (TMA). Without this fence the refill landed on reads not
    // yet done: 9 of 12 runs of a two-pass T = 32 input (seed 4209) had unit 31 of pass 0 wrong in lanes 0-15, with
    // lane-0 or all-lane arrives alike; with it 0 of 12, at about 0.3 us of T = 8 span.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if (lane == 0) mbar_arrive(&bb.empty[ps]);
    const long long c1 = pclock();
    if (j >= NPB) mbar_wait(&bb.rempty[pb], ((j / NPB) - 1) & 1);
    if (CLA_PROF) pf.red_wait += pclock() - c1;
    float4* rp = red + pb * RED_PAIR_F4;
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
      for (int n = 0; n < NTP; ++n)
        rp[((q * NTP_MAX + n) * NCONS + w) * 32 + lane] =
            make_float4(acc[q][n][0], acc[q][n][1], acc[q][n][2], acc[q][n][3]);
    __syncwarp();
    if (lane == 0) mbar_arrive(&bb.rfull[pb]);
  }
}

// Phase A body for one CTA of the slice's cluster. Every thread of the CTA calls it.
template <int C>
__device__ __forceinline__ void phase_a(const PhaseAParams& p, int slice, int rank, uint8_t* smem, const Bars& bb) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned long long* st = p.stamps ? p.stamps + static_cast<unsigned long long>(NSTAMP) * blockIdx.x : nullptr;
  if (st && threadIdx.x == 0) {
    st[0] = gtimer();
    if (CLA_PROF) st[10] = clock64();
  }
  const Slice sl = read_slice(p.sched, slice, rank, C);
  uint8_t* ring = smem;
  float4* red = reinterpret_cast<float4*>(smem + RED_OFF);
  __nv_bfloat16* slab = reinterpret_cast<__nv_bfloat16*>(smem + CLA_SLAB_OFF);
  const int pairs = sl.passes * (sl.units >> 1);

  if (threadIdx.x == 0) {
    for (int s = 0; s < NPS; ++s) {
      mbar_init(&bb.full[s], 1);
      mbar_init(&bb.empty[s], NCONS);
    }
    for (int r = 0; r < NPB; ++r) {
      mbar_init(&bb.rfull[r], NCONS);
      mbar_init(&bb.rempty[r], 1);
    }
    mbar_init(bb.first0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  // warpgroup index through a shuffle so ptxas sees it as warp-uniform and sizes each branch by its setmaxnreg
  const int wg = __shfl_sync(0xffffffffu, static_cast<int>(threadIdx.x) >> 7, 0);
  if (wg > 0) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(REG_HIGH));
    const int w = warp - CONS_WARP0;
    int j = 0;
    Prof pf;
    const long long cl0 = pclock();
    for (int pass = 0; pass < sl.passes; ++pass) {
      if (sl.ntt - pass * NTP_MAX >= 2)
        consume_pass<2>(p, sl, pass, ring, red, bb, w, lane, j, st, pf);
      else
        consume_pass<1>(p, sl, pass, ring, red, bb, w, lane, j, st, pf);
    }
    if (st && lane == 0) {
      atomicMax(st + 2, gtimer());
      if (CLA_PROF && w == 0) {
        st[5] = pf.full_wait;
        st[6] = pf.red_wait;
        st[7] = pclock() - cl0;
      }
    }
    return;
  }
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(REG_LOW));
  if (warp == 0) {
    if (lane == 0) {
      uint64_t pol_first, pol_last;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol_first));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol_last));
      const uint8_t* base = p.w13u + (static_cast<size_t>(sl.expert) * clm::RB_A + sl.rb0) * UB;
      int j = 0;
      for (int pass = 0; pass < sl.passes; ++pass) {
        const uint64_t pol = pass + 1 < sl.passes ? pol_last : pol_first;
        for (int u = 0; u < sl.units; u += 2, ++j) {
          const int ps = j % NPS;
          if (j >= NPS) mbar_wait(&bb.empty[ps], ((j / NPS) - 1) & 1);
          uint8_t* slot = ring + static_cast<size_t>(ps) * 2 * UB;
          uint64_t* bar0 = j == 0 ? bb.first0 : &bb.full[ps];
          if (j == 0) mbar_expect_tx(bar0, UB);
          mbar_expect_tx(&bb.full[ps], j == 0 ? UB : 2 * UB);
          bulk_load(slot, base + static_cast<size_t>(u) * UB, UB, bar0, pol);
          bulk_load(slot + UB, base + static_cast<size_t>(u + 1) * UB, UB, &bb.full[ps], pol);
        }
      }
    }
  } else if (warp <= NPB) {
    const int k = warp - 1;
    const int g = lane >> 2, t = lane & 3;
    const float gsc = clm::gscale(p.w13_gs[sl.expert]);
    const int upp = sl.units >> 1;
    Prof pf;
    for (int j = k; j < pairs; j += NPB) {
      const int pass = j / upp, u0 = 2 * (j - pass * upp);
      const int ntp = min(NTP_MAX, sl.ntt - pass * NTP_MAX);
      const long long c0 = pclock();
      mbar_wait(&bb.rfull[k], (j / NPB) & 1);
      const long long c1 = pclock();
      if (CLA_PROF) pf.wait += c1 - c0;
      const float4* rp = red + k * RED_PAIR_F4;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int col = 8 * (u0 + q) + g;
#pragma unroll
        for (int n = 0; n < NTP_MAX; ++n) {
          if (n < ntp) {
            const float4 sm = clm::sum_warps(rp + ((q * NTP_MAX + n) * NCONS) * 32 + lane, 32);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int li = pass * 16 + n * 8 + 2 * t + h;
              if (li < sl.ntok) {
                const __nv_bfloat16 v = clm::swiglu_bf16(h ? sm.y : sm.x, h ? sm.w : sm.z, gsc);
                slab[li * sl.cpc + col] = v;
              }
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&bb.rempty[k]);
      if (CLA_PROF) pf.busy += pclock() - c1;
    }
    // global copy: one bulk copy per slab row once both epilogue warps are done (named barrier 1, 64 threads); each
    // warp fences its generic-proxy slab stores before the barrier, ahead of the async-proxy reads
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NPB) : "memory");
    if (k == 0) {
      __nv_bfloat16* gout = p.inter_g + static_cast<size_t>(slice) * p.t_max * p.pitch + rank * sl.cpc;
      for (int i = lane; i < sl.ntok; i += 32)
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                     ::"l"(gout + static_cast<size_t>(i) * p.pitch), "r"(smem_u32(slab + i * sl.cpc)), "r"(sl.cpc * 2)
                     : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    if (st && lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      atomicMax(st + 3, gtimer());
      st[4] = smid;
      if (CLA_PROF && k == 0) {
        st[11] = clock64();
        st[8] = pf.busy;
        st[9] = pf.wait;
      }
    }
  }
}

__device__ __forceinline__ int cluster_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return static_cast<int>(r);
}
__device__ __forceinline__ int cluster_id() {
  unsigned r;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r));
  return static_cast<int>(r);
}

// Standalone phase A kernel: grid = n_slices x C, cluster i runs slice slice0 + i.
template <int C>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(THREADS, 1) cl_phase_a_kernel(PhaseAParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t bars[2 * NPS + 2 * NPB + 1];
  const int slice = cluster_id() + p.slice0;
  if (slice >= p.sched->n_slices) return;
  const Bars bb{bars, bars + NPS, bars + 2 * NPS, bars + 2 * NPS + NPB, bars + 2 * NPS + 2 * NPB};
  phase_a<C>(p, slice, cluster_rank(), smem, bb);
}

}  // namespace cla
