// CL-1.2 phase B of the MoE cluster kernel (docs/plans/2026-11-04-moe-cluster-rewrite.md task 1.2,
// docs/runs-moe.md "CL-1.2 phase B"). One cluster of C = 8 CTAs runs one expert slice from the ClSchedule.
//
// Slice i owns intermediate (K) columns [col0, col0 + ncol) of expert e. CTA c owns w2 output rows
// [512 c, 512 c + 512) (32 units of 16 rows) and produces the slice's FP32 partial y for those rows: phase B is
// split by output columns inside the cluster, with no intra-cluster reduction (D1 = b).
//
// Intermediate slab (the phase A / phase B interface, assumed here because task 1.1's header had not landed):
//   each CTA c holds slab[T_MAX][sc] BF16, row-major, sc = ncol / C columns, row stride sc elements;
//   slab row i = the expert's i-th (token, slot) pair in ClSchedule tok_list order (tok_list[tok0 + i]),
//   slab column j = slice column c * sc + j = intermediate column col0 + c * sc + j.
//   Rows >= ntok are never read. In this test the kernel loads its slab from a global intermediate
//   inter[pair][4096] (row = tok_list position); in the real kernel phase A writes it.
//
// K blocks: the slice's K range is cut at multiples of 512 (one helper0 warp range). A block is
// [max(col0, 512 kb), min(col0 + ncol, 512 kb + 512)) for kb = col0 / 512 .. (col0 + ncol - 1) / 512; it holds
// 64-K chunks [clo, chi) of helper0 warp kb, with clo in {0, 4} and chi in {4, 8} (slices are multiples of 256).
// K-outer loop: for each block (natural or rotated order), the 32 units of the CTA stream through the ring as
// 4 stages of 8 unit-blocks; consumer warp w computes unit 8 j + w of stage j with the helper0 per-warp
// arithmetic (4 chains, c0 + ((c1 + c2) + c3)) over the block's chunks, and adds it to its running sum.
// The running sum over blocks in natural K order is helper0's warp-order cross-warp sum, so S = 1 in natural
// order is bitwise helper0 (docs/runs-moe.md CL-0.5 items 4-7).
//
// Ring stage (36,864 B, the helper0 unit layout): unit-block u8 (0..7) codes at u8 * 4096 + c * 512 for chunk c,
// its warp-kb scale block (512 B) at 32768 + u8 * 512, so clm::unit_codes / unit_scales index it as "warp u8".
//
// Gather (D1 = b): two staging warps copy each block's columns from the C slabs over DSMEM
// (ld.shared::cluster.v4) into a local double-buffered block buffer [T_MAX][520] BF16, pieces in rotated
// owner order. xmode 1 is exchange (c): slabs written to global, cluster barrier, read back through L2.
// xmode 2 copies nothing (the stream-and-compute-only control; numerics are garbage).
//
// Outputs:
//   y_partial[slice][T_MAX][4096] FP32, rows i < ntok, columns of the CTA's 512 output rows:
//     expert with S = 1: v = bf16(acc * gs * 2^62) * bf16(1 + w), helper0's ybuf value (CL-0.5 items 12-15);
//     expert with S > 1: the raw FP32 partial acc over the slice's K range (no scale, no rounding). BF16
//       rounding and the sum of squares need the full-K y, so phase 2 sums the S partials in slice order from
//       0.f, then applies gs, BF16, (1 + w) and the sum of squares.
//   sumsq_partial[slice][T_MAX] FP32: S = 1: the token's full-row sum of v^2 (per unit with clm::ss_unit, then
//     units 0..31 in order inside a CTA, then CTAs 0..7 in rank order through the leader's shared memory); S > 1: 0.
// Stores are done by an epilogue warp after an mbarrier handoff through shared memory; consumer warps execute
// no GPU-scope fence (rule 1, CL-0.6).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#include "cl_math.cuh"
#include "cl_schedule.h"
#include "cl_unit_loop.cuh"

namespace clb {

constexpr int C = 8;
// Warp roles for NCW consumer warps (8 or 16): consumers 0 .. NCW - 1 in NCW / 8 groups of 8; group g takes the
// stages j = g (mod NCW / 8) of every K block. Then the producer, two staging warps, the epilogue warp.
constexpr int UPS = 8;                     // unit-blocks per stage = consumer warps per group
// NCW = 16 runs 18 warps (one staging warp, the epilogue folded into the producer warp after its issue loop) so the
// launch bound gives the consumer 112 registers without setmaxnreg (which ptxas 13.0 mishandles here, CL-1.3a).
__host__ __device__ constexpr int n_staging(int ncw) { return ncw == 16 ? 1 : 2; }
__host__ __device__ constexpr int n_threads(int ncw) { return 32 * (ncw + 1 + n_staging(ncw) + (ncw == 16 ? 0 : 1)); }
constexpr int SLOT = clm::UNIT_BYTES;      // 36,864
constexpr int MAX_DEPTH = 6;
constexpr int ROWS = clm::HD / C;          // 512 output rows per CTA
constexpr int UPC = ROWS / 16;             // 32 units per CTA
constexpr int SPB = UPC / UPS;             // 4 stages per K block
constexpr int BROW = 512 + 8;              // block-buffer row in bf16; the 16 B pad keeps rows g and g + 1 off the same banks
constexpr int YROW = ROWS + 4;             // ystage row in floats; the pad spreads the 4 t-lanes over banks
constexpr int NSTAMP = 10;   // 0 entry, 1 slab loaded, 2 first stage, 3 consumers done, 4 epilogue stored,
                              // 5 gather done, 6 / 7 consumer wait on gather / weights (ns), 8 slabs published

struct Params {
  const uint8_t* w2u;              // [32][256][36864] w2 in unit order (public prepare ma_w[11])
  const float* w2_gs;              // [32]
  const unsigned short* normw;     // [4096] post_norm_w bf16 bits
  const __nv_bfloat16* inter;      // [pairs][4096] test-provided intermediate, row = tok_list position
  const ClSchedule* sched;
  float* y_partial;                // [n_slices][T_MAX][4096]
  float* sumsq_partial;            // [n_slices][T_MAX]
  __nv_bfloat16* xg;               // xmode 1: [n_slices][T_MAX][4096] slice-relative columns
  unsigned long long* stamps;      // [grid][NSTAMP]
  float* dbg_acc;                  // DBG: [n_slices][T_MAX][4096] raw FP32 sums (consumer stores, test only)
  float* dbg_sq;                   // DBG: [n_slices][T_MAX][256] per-unit sums of squares (S = 1)
  int depth;
  int order;                       // 0: blocks in K order; 1: rotated, CTA c starts at the block of peer c + rot
  int rot;                         // rotation offset (1: peer c + 1 first, as task 1.2 states; 0: own block first)
  int xmode;                       // 0: DSMEM gather (b); 1: global + L2 (c); 2: no exchange (control)
  int slab_cols;                   // max sc over the launch (shared-memory layout)
  int diag;                        // diagnostics: bit 0 consumers skip the math, bit 1 producer skips the loads
  uint32_t zero;                   // runtime 0 (k16 opaque to ptxas, as in helper0)
};

// Loop knobs (CL-1.3a). CLB_LOOP_BREG: -1 auto (B fragments of the block's first 4 chunks in registers at
// NTT = 1, none otherwise), 0 never, N > 0 the first N chunks of every block (as far as the block has them). CLB_LOOP_DQ: 1 r3 merge on the FMA pipe, 0 the LOP3 form. CLB_LOOP_DIAG: bench
// diagnostics, see cl_unit_loop.cuh. CLB_NREG16: consumer registers at NCW = 16 through setmaxnreg (0 keeps the
// launch split of 96 for every warp); the four other warps get what remains of the 64K file, rounded down to 8.
#ifndef CLB_LOOP_BREG
#define CLB_LOOP_BREG -1
#endif
#ifndef CLB_LOOP_DQ
#define CLB_LOOP_DQ 0
#endif
#ifndef CLB_LOOP_DIAG
#define CLB_LOOP_DIAG 0
#endif
#ifndef CLB_NREG16
#define CLB_NREG16 0
#endif
#ifndef CLB_ACC_SMEM
#define CLB_ACC_SMEM 0
#endif
#ifndef CLB_UPW
#define CLB_UPW 1          // unit-blocks per warp per stage at T_MAX = 8 (1 or 2)
#endif
#ifndef CLB_UPW32
#define CLB_UPW32 1        // the same at T_MAX = 32
#endif
#ifndef CLB_LOOP_FENCE
#define CLB_LOOP_FENCE 1   // 0 (diagnosis only) drops the proxy fence at the slot release
#endif
__host__ __device__ constexpr int upw(int tmax) { return tmax == 8 ? CLB_UPW : CLB_UPW32; }
__host__ __device__ constexpr int wpg(int tmax) { return UPS / upw(tmax); }   // warps per stage group
#ifndef CLB_NREG8
#define CLB_NREG8 208      // NCW = 8 consumer registers (0: no setmaxnreg)
#endif
#ifndef CLB_DEC_SITE
#define CLB_DEC_SITE 0     // 1: one setmaxnreg.dec for the non-consumer warps before the role dispatch
#endif
// setmaxnreg.inc can only take registers that setmaxnreg.dec released inside the CTA, so the split stays within
// the launch allocation 640 x 96 (an inc beyond it blocks forever: the first 1.3a build hung 2.5 h on that).
#ifndef CLB_NREG16_OTHER
#define CLB_NREG16_OTHER (((640 * 96 - 16 * 32 * CLB_NREG16) / (4 * 32)) / 8 * 8)
#endif
constexpr int NREG16_OTHER = CLB_NREG16_OTHER;
static_assert(CLB_NREG16 == 0 || (NREG16_OTHER >= 24 && NREG16_OTHER <= 96 && CLB_NREG16 % 8 == 0 &&
                                  16 * 32 * CLB_NREG16 + 4 * 32 * NREG16_OTHER <= 640 * 96), "CLB_NREG16");
// Running per-unit sums in shared memory at T_MAX = 8 (16 KB: 32 units x 32 lanes x float4) instead of 16 registers
// per lane; with the 64-register B cache the 16-warp consumer spills otherwise.
__host__ __device__ constexpr bool acc_smem(int tmax) { return CLB_ACC_SMEM != 0 && tmax == 8; }
__host__ __device__ constexpr int acc_bytes(int tmax) { return acc_smem(tmax) ? 32 * 32 * 16 : 0; }

__host__ __device__ constexpr int buf_bytes(int tmax) { return tmax * BROW * 2; }
__host__ __device__ constexpr int smem_bytes(int tmax, int depth, int slab_cols) {
  return depth * SLOT + 2 * buf_bytes(tmax) + tmax * slab_cols * 2 + acc_bytes(tmax);
}

// ---- ring and barrier primitives, copied from cl_stream.cu (task 0.1) ----
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint64_t* bar, unsigned parity) {
  unsigned ok;
  asm volatile("{\n.reg .pred p;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  while (!mbar_try(bar, parity)) {
  }
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, unsigned bytes, uint64_t* bar, uint64_t policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}
// ---- cluster primitives ----
__device__ __forceinline__ unsigned cta_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned rank) {
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ uint4 ld_dsmem(unsigned addr) {
  uint4 v;
  asm volatile("ld.shared::cluster.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void st_dsmem_f32(unsigned addr, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(addr), "f"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_cg(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
// Non-aligned forms: warps arrive and wait at different points of their code.
__device__ __forceinline__ void cl_arrive_release() { asm volatile("barrier.cluster.arrive.release;" ::: "memory"); }
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire;" ::: "memory"); }

template <int NTT, int NCHB, int UPW>
__host__ __device__ constexpr int n_cache() {
  return CLB_LOOP_BREG == -1 ? (NTT == 1 && UPW == 1 ? (NCHB < 4 ? NCHB : 4) : 0)
                             : (CLB_LOOP_BREG < NCHB ? CLB_LOOP_BREG : NCHB);
}

// Per-slice geometry, computed identically by every thread of the cluster.
struct Slice {
  int slice, e, k0, k1, ncol, sc, tok0, ntok, full, wlo, nb, start;
  __device__ __forceinline__ int kb(int bi) const { return wlo + (bi + start) % nb; }
  __device__ __forceinline__ int clo(int kb_) const { return (max(k0, 512 * kb_) - 512 * kb_) >> 6; }
  __device__ __forceinline__ int chi(int kb_) const { return (min(k1, 512 * kb_ + 512) - 512 * kb_) >> 6; }
};

__device__ __forceinline__ Slice slice_info(const Params& p, int slice, int rank) {
  const ClSchedule* s = p.sched;
  Slice q;
  q.slice = slice;
  q.e = s->slice_expert[slice];
  q.k0 = s->slice_col0[slice];
  q.ncol = s->slice_ncol[slice];
  q.k1 = q.k0 + q.ncol;
  q.sc = q.ncol / C;
  q.tok0 = s->slice_tok0[slice];
  q.ntok = s->slice_ntok[slice];
  q.full = s->expert_nslices[q.e] == 1;
  q.wlo = q.k0 >> 9;
  q.nb = ((q.k1 - 1) >> 9) - q.wlo + 1;
  q.start = p.order ? (((q.k0 + ((rank + p.rot) % C) * q.sc) >> 9) - q.wlo) : 0;
  return q;
}

struct Smem {
  uint8_t* ring;
  __nv_bfloat16* buf;    // [2][T_MAX][BROW]
  __nv_bfloat16* slab;   // [T_MAX][sc]
  float* accs;           // [32 units][32 lanes][4] running sums when acc_smem(T_MAX)
  uint64_t* full;        // [MAX_DEPTH]
  uint64_t* empty;
  uint64_t* bfull;       // [2]
  uint64_t* bempty;      // [2]
  uint64_t* yready;
  uint64_t* slabbar;
  float* sumsq_cl;       // [C][T_MAX], used in the leader
};

// CLB_CONS_NOINLINE (SASS check build only): the consumer body becomes its own function, so cuobjdump can list
// every fence the consumer path executes.
#ifdef CLB_CONS_NOINLINE
#define CLB_CONS_ATTR __noinline__
#else
#define CLB_CONS_ATTR __forceinline__
#endif

// One K block of the CTA's units for this warp: the B fragments once (BREG) and then the R stages the warp's
// group owns, rolled. Unrolled, the 4 stages x 3 chunk-range variants made a ~120 KB hot loop (CL-1.2); the R
// accumulator sets rotate through registers instead (accr[0] is always the current unit; after R stages the order
// is back to unit 8 j + warp in accr[j]).
template <int NTT, int TM, int NCW, int CLO, int CHI>
__device__ __forceinline__ void consume_block(const Params& p, const Smem& sm, const Slice& q, int bi, int grp, int u8,
                                              int lane, clu::Opaque op, const __nv_bfloat16* buf,
                                              float (&accr)[SPB / (NCW / wpg(TM))][upw(TM)][NTT][4], bool stamp,
                                              unsigned long long* st, unsigned long long& wait_f) {
  constexpr int UPW = upw(TM), WPG = wpg(TM), NG = NCW / WPG, R = SPB / NG;
  constexpr int NCACHE = n_cache<NTT, CHI - CLO, UPW>();
  const int D = p.depth;
  clu::BFrag<NTT, NCACHE> bf;
  if (NCACHE > 0 && CLB_LOOP_DIAG != 1) clu::load_bfrag<NTT, CLO, NCACHE>(buf, BROW, lane, bf);
#pragma unroll 1
  for (int jj = 0; jj < R; ++jj) {
    const int it = bi * SPB + jj * NG + grp, s = it % D;
    if (CLB_LOOP_DIAG == 6) {
      if (stamp && it == 0) st[2] = gtimer();
    } else if (stamp) {
      const unsigned long long t0 = gtimer();
      mbar_wait(&sm.full[s], (it / D) & 1);
      const unsigned long long t1 = gtimer();
      wait_f += t1 - t0;
      if (it == 0) st[2] = t1;
    } else {
      mbar_wait(&sm.full[s], (it / D) & 1);
    }
    const unsigned char* slot = sm.ring + s * SLOT;
    float a[UPW][NTT][4];
    if (p.diag & 1) {
#pragma unroll
      for (int k = 0; k < UPW; ++k)
#pragma unroll
        for (int n = 0; n < NTT; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) a[k][n][i] = 0.f;
    } else {
      clu::unit_block<NTT, CLO, CHI, NCACHE, UPW, CLB_LOOP_DQ, CLB_LOOP_DIAG>(slot, u8, WPG, lane, op, buf, BROW, &bf, a);
    }
    __syncwarp();
    // The warp's generic-proxy reads of the slot must be complete before the TMA refill lands in it (CL-1.1 saw the
    // refill overtake outstanding LDS on 9 of 12 two-pass runs): CTA-scope proxy fence before the release.
    if (CLB_LOOP_FENCE) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (lane == 0) mbar_arrive(&sm.empty[s]);
    if (acc_smem(TM)) {
#pragma unroll
      for (int k = 0; k < UPW; ++k) {
        float4* ac = reinterpret_cast<float4*>(sm.accs) + (UPS * (jj * NG + grp) + u8 + WPG * k) * 32 + lane;
        float4 v = *ac;
        v.x += a[k][0][0]; v.y += a[k][0][1]; v.z += a[k][0][2]; v.w += a[k][0][3];
        *ac = v;
      }
    } else {
#pragma unroll
      for (int k = 0; k < UPW; ++k)
#pragma unroll
        for (int n = 0; n < NTT; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float t0 = accr[0][k][n][i] + a[k][n][i];
#pragma unroll
            for (int r = 0; r + 1 < R; ++r) accr[r][k][n][i] = accr[r + 1][k][n][i];
            accr[R - 1][k][n][i] = t0;
          }
    }
  }
}

template <int NTT, int TM, int NCW, bool DBG>
__device__ CLB_CONS_ATTR void consumer(const Params& p, const Smem& sm, const Slice& q, int rank, int warp, int lane,
                                         unsigned long long* st) {
  constexpr int UPW = upw(TM), WPG = wpg(TM), NG = NCW / WPG, R = SPB / NG;
  static_assert(NCW % WPG == 0 && SPB % NG == 0, "consumer warps must form whole stage groups");
  const int grp = warp / WPG, u8 = warp % WPG;
  float accr[R][UPW][NTT][4];
#pragma unroll
  for (int j = 0; j < R; ++j)
#pragma unroll
    for (int k = 0; k < UPW; ++k)
#pragma unroll
      for (int n = 0; n < NTT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) accr[j][k][n][i] = 0.f;
  if (acc_smem(TM)) {
#pragma unroll
    for (int j = 0; j < R; ++j)
#pragma unroll
      for (int k = 0; k < UPW; ++k)
        reinterpret_cast<float4*>(sm.accs)[(UPS * (j * NG + grp) + u8 + WPG * k) * 32 + lane] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  const bool stamp = warp == 0 && lane == 0;
  unsigned long long wait_b = 0, wait_f = 0;
  const clu::Opaque op{16u + p.zero, 1u + p.zero};
  for (int bi = 0; bi < q.nb; ++bi) {
    const int kb = q.kb(bi), clo = q.clo(kb), chi = q.chi(kb);
    const int bsel = bi & 1;
    if (stamp) {
      const unsigned long long t0 = gtimer();
      mbar_wait(&sm.bfull[bsel], (bi >> 1) & 1);
      wait_b += gtimer() - t0;
    } else {
      mbar_wait(&sm.bfull[bsel], (bi >> 1) & 1);
    }
    const __nv_bfloat16* buf = sm.buf + bsel * TM * BROW;
    if (clo == 0 && chi == 8) consume_block<NTT, TM, NCW, 0, 8>(p, sm, q, bi, grp, u8, lane, op, buf, accr, stamp, st, wait_f);
    else if (clo == 0) consume_block<NTT, TM, NCW, 0, 4>(p, sm, q, bi, grp, u8, lane, op, buf, accr, stamp, st, wait_f);
    else consume_block<NTT, TM, NCW, 4, 8>(p, sm, q, bi, grp, u8, lane, op, buf, accr, stamp, st, wait_f);
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.bempty[bsel]);
  }
  if (stamp) {
    st[3] = gtimer();
    st[6] = wait_b;
    st[7] = wait_f;
  }
  // Every consumer has finished reading the ring (and every bulk copy has landed) before it becomes ystage.
  asm volatile("bar.sync 1, %0;" :: "n"(32 * NCW) : "memory");
  float* ystage = reinterpret_cast<float*>(sm.ring);          // [TM][YROW]
  float* sqstage = ystage + TM * YROW;                         // [UPC][TM]
  const int g = lane >> 2, t = lane & 3;
  const float gsc = clm::gscale(p.w2_gs[q.e]);
#pragma unroll
  for (int jj = 0; jj < R; ++jj)
#pragma unroll
  for (int k = 0; k < UPW; ++k) {
    const int j = jj * NG + grp;
    const int ul = UPS * j + u8 + WPG * k;       // unit inside the CTA
    const int rl = 16 * ul + g;                  // local output row
    const int n0 = ROWS * rank + rl;             // global output row
    const float nw0 = clm::norm_mul(p.normw[n0]), nw8 = clm::norm_mul(p.normw[n0 + 8]);
#pragma unroll
    for (int n = 0; n < NTT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 8 * n + 2 * t + h;
        float a0 = accr[jj][k][n][h], a8 = accr[jj][k][n][2 + h];
        if (acc_smem(TM)) {
          const float4 v = reinterpret_cast<const float4*>(sm.accs)[ul * 32 + lane];
          a0 = h ? v.y : v.x;
          a8 = h ? v.w : v.z;
        }
        if (DBG && tk < q.ntok) {
          float* da = p.dbg_acc + ((size_t)q.slice * TM + tk) * clm::HD;
          da[n0] = a0;
          da[n0 + 8] = a8;
        }
        if (q.full) {
          float v0, v8;
          const float sq = clm::ss_unit(clm::ynorm_pair(a0, a8, gsc, nw0, nw8, v0, v8));
          ystage[tk * YROW + rl] = v0;
          ystage[tk * YROW + rl + 8] = v8;
          if (g == 0) sqstage[ul * TM + tk] = sq;
        } else {
          ystage[tk * YROW + rl] = a0;
          ystage[tk * YROW + rl + 8] = a8;
        }
      }
    }
  }
  __syncwarp();
  if (lane == 0) mbar_arrive(sm.yready);
}

// Epilogue (rule 1): after the consumers' mbarrier handoff, store y_partial and the per-token sums of squares,
// then the end cluster barriers. Run by its own warp at NCW = 8 and by the producer warp at NCW = 16.
template <int TM, bool DBG>
__device__ __forceinline__ void epilogue(const Params& p, const Smem& sm, const Slice& q, int rank, int slice, int lane,
                                         unsigned long long* st) {
  mbar_wait(sm.yready, 0);
  const float* ystage = reinterpret_cast<const float*>(sm.ring);
  const float* sqstage = ystage + TM * YROW;
  float* yp = p.y_partial + (size_t)slice * TM * clm::HD + ROWS * rank;
  for (int i = 0; i < q.ntok; ++i)
    for (int c4 = lane; c4 < ROWS / 4; c4 += 32)
      *reinterpret_cast<float4*>(yp + (size_t)i * clm::HD + 4 * c4) =
          *reinterpret_cast<const float4*>(ystage + i * YROW + 4 * c4);
  if (q.full) {
    const unsigned dst = mapa(smem_u32(sm.sumsq_cl), 0);
    for (int i = lane; i < q.ntok; i += 32) {
      float s = 0.f;
      for (int u = 0; u < UPC; ++u) {
        const float v = sqstage[u * TM + i];
        s += v;
        if (DBG) p.dbg_sq[((size_t)slice * TM + i) * clm::RB_B + UPC * rank + u] = v;
      }
      st_dsmem_f32(dst + (rank * TM + i) * 4, s);
    }
  }
  if (lane == 0) st[4] = gtimer();
  cl_wait();
  cl_arrive_release();   // the leader's sumsq_cl entries
  cl_wait();
  if (rank == 0) {
    for (int i = lane; i < q.ntok; i += 32) {
      float s = 0.f;
      if (q.full)
        for (int r = 0; r < C; ++r) s += sm.sumsq_cl[r * TM + i];
      p.sumsq_partial[(size_t)slice * TM + i] = s;
    }
  }
}

template <int TM, int NCW, bool DBG>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(n_threads(NCW), 1) phase_b_kernel(Params p) {
  constexpr int WP = NCW, WS0 = NCW + 1, NSW = n_staging(NCW);
  constexpr bool EPI_IN_PROD = NCW == 16;
  extern __shared__ __align__(1024) uint8_t dsm[];
  __shared__ __align__(8) uint64_t bars[2 * MAX_DEPTH + 6];
  __shared__ float sumsq_cl[C * TM];

  const int rank = static_cast<int>(cta_rank());
  const int slice = blockIdx.x / C;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned long long* st = p.stamps + (size_t)NSTAMP * blockIdx.x;
  const int D = p.depth;

  Smem sm;
  sm.ring = dsm;
  sm.buf = reinterpret_cast<__nv_bfloat16*>(dsm + D * SLOT);
  sm.slab = reinterpret_cast<__nv_bfloat16*>(dsm + D * SLOT + 2 * buf_bytes(TM));
  sm.accs = reinterpret_cast<float*>(dsm + D * SLOT + 2 * buf_bytes(TM) + TM * p.slab_cols * 2);
  sm.full = bars;
  sm.empty = bars + MAX_DEPTH;
  sm.bfull = bars + 2 * MAX_DEPTH;
  sm.bempty = bars + 2 * MAX_DEPTH + 2;
  sm.yready = bars + 2 * MAX_DEPTH + 4;
  sm.slabbar = bars + 2 * MAX_DEPTH + 5;
  sm.sumsq_cl = sumsq_cl;

  if (threadIdx.x == 0) {
    st[0] = gtimer();
    for (int s = 0; s < D; ++s) {
      mbar_init(&sm.full[s], 1);
      mbar_init(&sm.empty[s], wpg(TM));
    }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&sm.bfull[b], NSW);
      mbar_init(&sm.bempty[b], NCW);
    }
    mbar_init(sm.yready, NCW);
    mbar_init(sm.slabbar, NSW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const Slice q = slice_info(p, slice, rank);

  // Register split. NCW = 8: 384 threads x 168 at launch; warpgroups 0-1 (consumers) take 208, warpgroup 2 keeps
  // 88 (256 x 208 + 128 x 88 = 64,512, the launch total). NCW = 16: 640 x 96 at launch; warpgroups 0-3 take
  // CLB_NREG16, warpgroup 4 keeps NREG16_OTHER. The setmaxnreg sits at the top of each role branch: ptxas raises
  // the register budget only for code it can see is dominated by the inc (a shared branch before the role
  // dispatch left the 1.3a consumer at 96 registers with 40 to 104 bytes of spills).
  if (CLB_DEC_SITE == 1 && warp >= NCW) {
    if (NCW == 8 && CLB_NREG8 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    if (NCW == 16 && CLB_NREG16 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NREG16_OTHER));
  }
  if (warp < NCW) {
    if (NCW == 8 && CLB_NREG8 > 0) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(CLB_NREG8));
    if (NCW == 16 && CLB_NREG16 > 0) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(CLB_NREG16));
    cl_arrive_relaxed();   // phase 0 (slabs published): consumers publish nothing
    const int ntt = (q.ntok + 7) >> 3;
    if (TM <= 8 || ntt == 1) consumer<1, TM, NCW, DBG>(p, sm, q, rank, warp, lane, st);
    else if (ntt == 2) consumer<(TM > 8 ? 2 : 1), TM, NCW, DBG>(p, sm, q, rank, warp, lane, st);
    else if (ntt == 3) consumer<(TM > 8 ? 3 : 1), TM, NCW, DBG>(p, sm, q, rank, warp, lane, st);
    else consumer<(TM > 8 ? 4 : 1), TM, NCW, DBG>(p, sm, q, rank, warp, lane, st);
    // Consumers leave here: barrier.cluster.wait waits only for non-exited threads, so the end-of-kernel
    // barrier does not need them, and they never execute its acquire (UCGABAR_WAIT + CCTL.IVALL).
  } else if (warp == WP) {
    if (CLB_DEC_SITE == 0 && NCW == 8 && CLB_NREG8 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    if (CLB_DEC_SITE == 0 && NCW == 16 && CLB_NREG16 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NREG16_OTHER));
    cl_arrive_relaxed();
    if (lane == 0) {
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      // The test loads the slab from global; without this wait those loads queue behind the ring's first
      // 4 x 36 KB x 120 CTAs of weight requests (5.4 us to the slab instead of ~1). In the real kernel the slab
      // is phase A's output and is already in shared memory.
      mbar_wait(sm.slabbar, 0);
      const uint8_t* wexp = p.w2u + (size_t)q.e * clm::RB_B * SLOT;
      const int n_it = q.nb * SPB;
      for (int it = 0; it < n_it; ++it) {
        const int bi = it / SPB, j = it % SPB, s = it % D;
        const int kb = q.kb(bi), clo = q.clo(kb), chi = q.chi(kb);
        if (it >= D) mbar_wait(&sm.empty[s], ((it / D) - 1) & 1);
        uint8_t* slot = sm.ring + s * SLOT;
        const unsigned cbytes = (chi - clo) * 512;
        if (p.diag & 2) {
          mbar_arrive(&sm.full[s]);
          continue;
        }
        mbar_expect_tx(&sm.full[s], UPS * (cbytes + 512));
#pragma unroll 1
        for (int u8 = 0; u8 < UPS; ++u8) {
          const uint8_t* unit = wexp + (size_t)(UPC * rank + UPS * j + u8) * SLOT;
          bulk_load(slot + u8 * 4096 + clo * 512, unit + (kb * 8 + clo) * 512, cbytes, &sm.full[s], pol);
          bulk_load(slot + clm::UNIT_CODES + u8 * 512, unit + clm::UNIT_CODES + kb * 512, 512, &sm.full[s], pol);
        }
      }
    }
    __syncwarp();
    if (EPI_IN_PROD) {
      epilogue<TM, DBG>(p, sm, q, rank, slice, lane, st);
    } else {
      cl_wait();
      cl_arrive_relaxed();
      cl_wait();
    }
  } else if (warp < WS0 + NSW) {
    if (CLB_DEC_SITE == 0 && NCW == 8 && CLB_NREG8 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    if (CLB_DEC_SITE == 0 && NCW == 16 && CLB_NREG16 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NREG16_OTHER));
    // staging warps: load the own slab, publish it, gather each K block into the block buffers
    const int tid = threadIdx.x - 32 * WS0;
    const int nst = 32 * NSW;
    const int sc = q.sc, n16 = sc / 8;
    {
      const __nv_bfloat16* src = p.inter + (size_t)q.tok0 * clm::HD + q.k0 + rank * sc;
      __nv_bfloat16* xg = p.xg + (size_t)slice * TM * clm::HD + rank * sc;
      for (int idx = tid; idx < q.ntok * n16; idx += nst) {
        const int r = idx / n16, cu = idx - r * n16;
        const uint4 v = *reinterpret_cast<const uint4*>(src + (size_t)r * clm::HD + cu * 8);
        *reinterpret_cast<uint4*>(sm.slab + r * sc + cu * 8) = v;
        if (p.xmode == 1) *reinterpret_cast<uint4*>(xg + (size_t)r * clm::HD + cu * 8) = v;
      }
      // rows [ntok, 8 ntt) of both block buffers are read by the mma (their outputs are discarded): zero them once
      const int r1 = ((q.ntok + 7) >> 3) * 8;
      const int zr = (r1 - q.ntok) * (BROW / 8);
      for (int b = 0; b < 2; ++b)
        for (int idx = tid; idx < zr; idx += nst)
          reinterpret_cast<uint4*>(sm.buf + b * TM * BROW + q.ntok * BROW)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(sm.slabbar);
    if (tid == 0) st[1] = gtimer();
    cl_arrive_release();
    cl_wait();
    if (tid == 0) st[8] = gtimer();
    const unsigned slab0 = smem_u32(sm.slab);
    const __nv_bfloat16* xg_s = p.xg + (size_t)slice * TM * clm::HD;
    for (int bi = 0; bi < q.nb; ++bi) {
      const int bsel = bi & 1;
      if (bi >= 2) mbar_wait(&sm.bempty[bsel], ((bi >> 1) - 1) & 1);
      if (p.xmode != 2) {
        const int kb = q.kb(bi);
        const int lo = max(q.k0, 512 * kb) - q.k0, hi = min(q.k1, 512 * kb + 512) - q.k0;
        const int plo = lo / sc, phi = (hi - 1) / sc, np = phi - plo + 1;
        const int off = (rank + p.rot) % np;
        __nv_bfloat16* buf = sm.buf + bsel * TM * BROW;
        for (int i = 0; i < np; ++i) {
          const int pr = plo + (i + off) % np;
          const int a = max(lo, pr * sc), b = min(hi, pr * sc + sc);
          const int m16 = (b - a) / 8;
          const int scol = a - pr * sc, dcol = a + q.k0 - 512 * kb;
          const int total = q.ntok * m16;
          const unsigned rbase = mapa(slab0, pr) + scol * 2;
          constexpr int U = (NCW == 16 && NREG16_OTHER < 64) ? 4 : 8;   // in-flight pieces within the warp's registers
          for (int i0 = tid; i0 < total; i0 += nst * U) {
            uint4 v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
              const int idx = i0 + u * nst;
              if (idx < total) {
                const int r = idx / m16, cu = idx - r * m16;
                if (p.xmode == 0) v[u] = ld_dsmem(rbase + (r * sc + cu * 8) * 2);
                else v[u] = ld_cg(xg_s + (size_t)r * clm::HD + a + cu * 8);
              }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
              const int idx = i0 + u * nst;
              if (idx < total) {
                const int r = idx / m16, cu = idx - r * m16;
                *reinterpret_cast<uint4*>(buf + r * BROW + dcol + cu * 8) = v[u];
              }
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&sm.bfull[bsel]);
    }
    if (tid == 0) st[5] = gtimer();
    cl_arrive_relaxed();   // done reading the peers' slabs
    cl_wait();
  } else {
    // epilogue warp
    if (CLB_DEC_SITE == 0 && NCW == 8 && CLB_NREG8 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    if (CLB_DEC_SITE == 0 && NCW == 16 && CLB_NREG16 > 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NREG16_OTHER));
    cl_arrive_relaxed();
    epilogue<TM, DBG>(p, sm, q, rank, slice, lane, st);
  }
}

}  // namespace clb

#ifdef CLB_WITH_TORCH
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <int TM, int NCW, bool DBG>
static void clb_launch_t(const clb::Params& p, int n_slices, int smem, cudaStream_t s) {
  auto fn = clb::phase_b_kernel<TM, NCW, DBG>;
  TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
              "smem attribute ", smem);
  fn<<<n_slices * clb::C, clb::n_threads(NCW), smem, s>>>(p);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "phase_b launch: ", cudaGetErrorString(e));
}

int64_t clb_smem(int64_t tmax, int64_t depth, int64_t slab_cols) {
  return clb::smem_bytes((int)tmax, (int)depth, (int)slab_cols);
}

void clb_launch(torch::Tensor w2u, torch::Tensor w2gs, torch::Tensor normw, torch::Tensor inter, torch::Tensor sched,
                torch::Tensor y_partial, torch::Tensor sumsq, torch::Tensor xg, torch::Tensor stamps,
                torch::Tensor dbg_acc, torch::Tensor dbg_sq, int64_t n_slices, int64_t tmax, int64_t depth,
                int64_t order, int64_t rot, int64_t xmode, int64_t slab_cols, int64_t ncw, int64_t diag) {
  TORCH_CHECK(depth >= 1 && depth <= clb::MAX_DEPTH && (tmax == 8 || tmax == 32));
  TORCH_CHECK(y_partial.numel() >= n_slices * tmax * 4096 && sumsq.numel() >= n_slices * tmax);
  TORCH_CHECK(stamps.numel() >= n_slices * clb::C * clb::NSTAMP);
  clb::Params p{};
  p.w2u = w2u.data_ptr<uint8_t>();
  p.w2_gs = w2gs.data_ptr<float>();
  p.normw = reinterpret_cast<const unsigned short*>(normw.data_ptr());
  p.inter = reinterpret_cast<const __nv_bfloat16*>(inter.data_ptr());
  p.sched = reinterpret_cast<const ClSchedule*>(sched.data_ptr());
  p.y_partial = y_partial.data_ptr<float>();
  p.sumsq_partial = sumsq.data_ptr<float>();
  p.xg = reinterpret_cast<__nv_bfloat16*>(xg.data_ptr());
  p.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>());
  const bool dbg = dbg_acc.numel() > 0;
  p.dbg_acc = dbg ? dbg_acc.data_ptr<float>() : nullptr;
  p.dbg_sq = dbg ? dbg_sq.data_ptr<float>() : nullptr;
  p.depth = (int)depth;
  p.order = (int)order;
  p.rot = (int)rot;
  p.xmode = (int)xmode;
  p.slab_cols = (int)slab_cols;
  p.diag = (int)diag;
  p.zero = 0;
  const int smem = clb::smem_bytes((int)tmax, (int)depth, (int)slab_cols);
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  const int n = (int)n_slices;
  if (tmax == 8 && ncw == 16) dbg ? clb_launch_t<8, 16, true>(p, n, smem, s) : clb_launch_t<8, 16, false>(p, n, smem, s);
  else if (tmax == 8) dbg ? clb_launch_t<8, 8, true>(p, n, smem, s) : clb_launch_t<8, 8, false>(p, n, smem, s);
  else dbg ? clb_launch_t<32, 8, true>(p, n, smem, s) : clb_launch_t<32, 8, false>(p, n, smem, s);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &clb_launch);
  m.def("smem", &clb_smem);
}
#endif
