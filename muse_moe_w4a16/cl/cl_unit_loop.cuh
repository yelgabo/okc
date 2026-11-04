// CL-1.3a unit loop (docs/plans/2026-11-04-moe-cluster-rewrite.md task 1.3a, docs/runs-moe.md "CL-1.3a unit loop").
// One warp's partial of one 16-row unit-block over the 64-K chunks [CLO, CHI) of a ring stage, for NTT n-tiles of
// 8 tokens. The arithmetic is clm::unit_warp_mma's (CL-0.5): E2M1 x 2^-126 times bf16(scale x 2^64) in one BF16
// multiply, mma.m16n8k16 with four accumulator chains (one per k16 step index), combined as c0 + ((c1 + c2) + c3).
// Both phases include it: phase B feeds the BF16 intermediate block buffer, phase A feeds x rows (same fragment
// layout, see clm::x_frag_base).
//
// What differs from clm::unit_warp_mma / clb::block_mma:
//   BREG: the token operand (B fragments) is loaded once per K block into registers (BFrag) and reused by every
//     unit-block of the block instead of two LDS.128 per chunk per unit-block. For NTT = 1 and 8 chunks that is 64
//     registers; with 16 consumer warps the consumer warpgroups take them from setmaxnreg.
//   DQ = 1: the r3 merge of the DQ7 unpack is an IMAD (x * one + y, `one` a runtime 1), which moves one LOP3 per
//     word from the ALU pipe (16 lanes) to the FMA pipe. Bitwise identical: the two masked operands are disjoint.
//   DIAG (bench only, numerics garbage): 1 B fragments are lane constants (no B loads), 2 no E2M1 unpack
//     (a = word, scale multiply kept), 3 no unpack and no scale multiply, 4 no mma (one FADD per step per tile).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#include "cl_math.cuh"

namespace clu {

// Runtime constants that ptxas cannot fold: 16 (scale decode multiply) and 1 (r3 merge multiply).
struct Opaque {
  uint32_t k16;
  uint32_t one;
};

// B fragments of NTT n-tiles over NCHB chunks: w[n][c][2 st], w[n][c][2 st + 1] are k16 step st of chunk c.
template <int NTT, int NCHB>
struct BFrag {
  uint32_t w[NTT][NCHB][8];
};

// buf: token rows (8 n + g for n-tile n, g = lane >> 2) with a row stride of `brow` bf16; the lane's 16 B pieces
// at bf16 column 64 c + 16 t (t = lane & 3), as clb's block buffer and clm::x_frag_base lay them out.
template <int NTT, int CLO, int CHI>
__device__ __forceinline__ void load_bfrag(const __nv_bfloat16* buf, int brow, int lane, BFrag<NTT, CHI - CLO>& bf) {
#pragma unroll
  for (int n = 0; n < NTT; ++n) {
    const uint4* base = reinterpret_cast<const uint4*>(buf + (8 * n + (lane >> 2)) * brow + 16 * (lane & 3));
#pragma unroll
    for (int c = CLO; c < CHI; ++c) {
      const uint4 lo = base[8 * c], hi = base[8 * c + 1];
      uint32_t* w = bf.w[n][c - CLO];
      w[0] = lo.x; w[1] = lo.y; w[2] = lo.z; w[3] = lo.w;
      w[4] = hi.x; w[5] = hi.y; w[6] = hi.z; w[7] = hi.w;
    }
  }
}

template <int DQ>
__device__ __forceinline__ void dq4(uint32_t q, uint32_t one, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  if (DQ == 1) r3 = ((q * 64u) & 0x80C080C0u) * one + (__funnelshift_l(q, q, 14) & 0x01000100u);
  else r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
}

template <int DQ, int DIAG>
__device__ __forceinline__ void decode_step(uint32_t word, uint32_t sg, uint32_t sg8, uint32_t one, uint32_t (&a)[4]) {
  if (DIAG == 2) {
    a[0] = clm::bmul2(word, sg); a[1] = clm::bmul2(word, sg8); a[2] = clm::bmul2(word ^ 1u, sg); a[3] = clm::bmul2(word ^ 2u, sg8);
  } else if (DIAG == 3) {
    a[0] = word; a[1] = word ^ 1u; a[2] = word ^ 2u; a[3] = word ^ 3u;
  } else {
    uint32_t r0, r1, r2, r3;
    dq4<DQ>(word, one, r0, r1, r2, r3);
    a[0] = clm::bmul2(r0, sg); a[1] = clm::bmul2(r1, sg8); a[2] = clm::bmul2(r2, sg); a[3] = clm::bmul2(r3, sg8);
  }
}

// slot: the ring stage; u8: the unit-block inside it (clm::unit_codes / unit_scales index it as warp u8).
// BREG = 1 reads the B fragments from `bf`; BREG = 0 reads them per chunk from `buf` (stride `brow`).
template <int NTT, int CLO, int CHI, int BREG, int DQ, int DIAG>
__device__ __forceinline__ void unit_block(const unsigned char* slot, int u8, int lane, Opaque op,
                                           const __nv_bfloat16* buf, int brow, const BFrag<NTT, CHI - CLO>* bf,
                                           float (&acc)[NTT][4]) {
  float c1[NTT][4], c2[NTT][4], c3[NTT][4];
#pragma unroll
  for (int n = 0; n < NTT; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) { acc[n][k] = 0.f; c1[n][k] = 0.f; c2[n][k] = 0.f; c3[n][k] = 0.f; }
  const uint4 sc = clm::unit_scales(slot, u8, lane);
  const uint4* base[NTT];
  if (!BREG) {
#pragma unroll
    for (int n = 0; n < NTT; ++n)
      base[n] = reinterpret_cast<const uint4*>(buf + (8 * n + (lane >> 2)) * brow + 16 * (lane & 3));
  }
#pragma unroll
  for (int c = CLO; c < CHI; ++c) {
    const uint4 w = clm::unit_codes(slot, u8, c, lane);
    uint32_t sg, sg8;
    clm::scale_pair(sc, c, op.k16, sg, sg8);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    uint32_t b[NTT][8];
#pragma unroll
    for (int n = 0; n < NTT; ++n) {
      if (DIAG == 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) b[n][i] = 0x3F803F80u + (uint32_t)lane + (uint32_t)(8 * c + i);
      } else if (BREG) {
#pragma unroll
        for (int i = 0; i < 8; ++i) b[n][i] = bf->w[n][c - CLO][i];
      } else {
        const uint4 lo = base[n][8 * c], hi = base[n][8 * c + 1];
        b[n][0] = lo.x; b[n][1] = lo.y; b[n][2] = lo.z; b[n][3] = lo.w;
        b[n][4] = hi.x; b[n][5] = hi.y; b[n][6] = hi.z; b[n][7] = hi.w;
      }
    }
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      uint32_t a[4];
      decode_step<DQ, DIAG>(ww[st], sg, sg8, op.one, a);
#pragma unroll
      for (int n = 0; n < NTT; ++n) {
        float* ch = st == 0 ? acc[n] : st == 1 ? c1[n] : st == 2 ? c2[n] : c3[n];
        if (DIAG == 4) ch[0] += __uint_as_float(a[0] ^ b[n][2 * st]) + __uint_as_float(a[1] ^ b[n][2 * st + 1]);
        else clm::mma_bf16(ch, a[0], a[1], a[2], a[3], b[n][2 * st], b[n][2 * st + 1]);
      }
    }
  }
#pragma unroll
  for (int n = 0; n < NTT; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[n][k] += c1[n][k] + c2[n][k] + c3[n][k];
}

}  // namespace clu
