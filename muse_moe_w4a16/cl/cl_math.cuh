// Device math of one fused-MoE unit, lifted from the helper0 kernel source in
// muse_moe_w4a16/cand/current_control_release.py (SHA256 5df257af..., _MA_ALT_SRCS['moe_helper0']).
// No ring, mbarrier, Params or Smem dependence. Every function keeps the helper0 expression form, so with the same
// nvcc flags (-O3, no --use_fast_math, no -ftz, default -fmad=true) the results are bitwise those of the helper0
// builds (TINY / TINY2 / MID with -DOKC_DQ7=1 -DOKC_PRMTSC=1). docs/runs-moe.md "CL-0.5 math" lists the rounding
// points and the accumulation orders.
//
// Unit: 16 weight rows x K = 4096, 36,864 bytes in the packer's unit order (_unit_order of _pack_codes /
// _pack_scales): 32,768 B codes [warp 8][chunk 8][lane 32][16 B], then 4,096 B scales [warp 8][lane 32][16 B].
// Phase A unit rb of an expert: rows g = gate column 8 rb + g, rows g + 8 = up column 8 rb + g (g = 0..7).
// Phase B unit rb: rows g and g + 8 = output columns 16 rb + g and 16 rb + g + 8.
//
// The numerics depend on the K split: 8 warps each own K [512 w, 512 w + 512); a warp runs 8 chunks of 64 K, each
// 4 mma k16 steps into 4 separate accumulator chains (one per step index); the chains combine as
// c0 + ((c1 + c2) + c3); the 8 warp partials sum in warp order from 0.f. A kernel that splits K differently is
// not bitwise equal to helper0.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace clm {

constexpr int HD = 4096, ID = 4096;
constexpr int NWU = 8;                 // warps per unit (the K split)
constexpr int NCH = 64 / NWU;          // 64-K chunks per warp
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // units per expert: 512 (w13), 256 (w2)
// dq4 yields e2m1 * 2^-126 and the scale decode yields scale * 2^64, so the global scale carries the 2^62 back
constexpr float TWO62 = 4.611686018427388e18f;
constexpr float POST_NORM_EPS = 1e-8f;

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// 8 E2M1 codes of one prepacked word -> the 4 bf16x2 A-fragment registers, values e2m1 * 2^-126 (exact).
// DQ7 bit layout (_stored_pos with DQ7 = True): r0 as stored, r1 = word << 4, r2 = rotl 10,
// r3 = (word << 6 on 6 bits) | (rotl 14 on the two e1 bits). The shifts are multiplies so ptxas puts them on
// the FMA pipe (IMAD.SHL) instead of the ALU pipe the LOP3 / SHF of the unpack saturate.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
}

// e4m3 scale byte b (0..3) of `word` -> bf16x2(scale * 2^64) in both halves: one PRMT copies the byte into both
// halves, (x 16) moves exponent and mantissa into the bf16 fields, + 0x5C00 per half rebiases the exponent by 184.
// Exact only for positive normal bytes 0x08..0x7E; prepare() checks every scale byte (_ma_scales_prmt_exact) and
// helper0 uses scale_lut otherwise. k16 is 16 made opaque to ptxas (16 + a runtime zero) so the multiply is an
// IMAD on the FMA pipe, not a LEA on the ALU pipe.
__device__ __forceinline__ uint32_t scd(uint32_t word, int b, uint32_t k16) {
  return __byte_perm(word, 0u, 0x4040u | ((uint32_t)b * 0x0101u)) * k16 + 0x5C005C00u;
}

// The non-PRMT build's table: entry i = bf16x2(e4m3(i) * 2^64), 0xFFFF for NaN (helper0's S.lut fill).
__device__ __forceinline__ uint32_t scale_lut(int i) {
  const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
  uint32_t b;
  if (ex == 15 && mt == 7) {
    b = 0xFFFFu;
  } else {
    float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
    v *= 0x1p64f;
    b = __float_as_uint(sgn ? -v : v) >> 16;
  }
  return b | (b << 16);
}

__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Lane (g = lane >> 2, t = lane & 3) of warp w, chunk c: 16 B of codes = 4 words, word st = k16 step st.
__device__ __forceinline__ uint4 unit_codes(const unsigned char* unit, int warp, int c, int lane) {
  return *reinterpret_cast<const uint4*>(unit + (warp * NCH + c) * 512 + lane * 16);
}
// Lane's 16 scale bytes: byte 2c + h = scale of row g (h = 0) / g + 8 (h = 1) for K [512 w + 64 c + 16 t, +16).
__device__ __forceinline__ uint4 unit_scales(const unsigned char* unit, int warp, int lane) {
  return *reinterpret_cast<const uint4*>(unit + UNIT_CODES + warp * 512 + lane * 16);
}
__device__ __forceinline__ void scale_pair(const uint4& sc, int c, uint32_t k16, uint32_t& sg, uint32_t& sg8) {
  const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
  sg = scd(swd, (c & 1) * 2, k16);
  sg8 = scd(swd, (c & 1) * 2 + 1, k16);
}
// One k16 step's A fragment: E2M1 * 2^-126 times bf16(scale * 2^64), one BF16 multiply (exact for e4m3 normals).
__device__ __forceinline__ void decode_step(uint32_t word, uint32_t sg, uint32_t sg8, uint32_t (&a)[4]) {
  uint32_t r0, r1, r2, r3;
  dq4(word, r0, r1, r2, r3);
  a[0] = bmul2(r0, sg); a[1] = bmul2(r1, sg8); a[2] = bmul2(r2, sg); a[3] = bmul2(r3, sg8);
}

// B fragment: lane (g, t) supplies token g of an 8-token n-tile, K [512 w + 64 c + 16 t, +16) as two uint4
// (lo = first 8 bf16, hi = next 8); k16 step st takes words 2 st and 2 st + 1 of those 8 words. With
// base = x_frag_base(row, w, lane), chunk c is base[8 c] and base[8 c + 1]. Rows are token rows of x (phase A) or
// of the BF16 intermediate (phase B); absent tokens are zeros.
__device__ __forceinline__ const uint4* x_frag_base(const __nv_bfloat16* row, int warp, int lane) {
  return reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * (lane & 3));
}

// One warp's partial of one unit over NTT n-tiles of 8 tokens. xf(tile, c, lo, hi) delivers the lane's B fragment
// words for chunk c (see x_frag_base). acc[tile][0..3] = (row g, token 2t), (row g, token 2t + 1),
// (row g + 8, token 2t), (row g + 8, token 2t + 1). Each tile's numbers equal a single-tile helper0 unit; the A
// fragment is decoded once per step and reused across tiles.
template <int NTT, class XF>
__device__ __forceinline__ void unit_warp_mma(const unsigned char* unit, int warp, int lane, uint32_t k16, XF&& xf,
                                              float (&acc)[NTT][4]) {
  float c1[NTT][4], c2[NTT][4], c3[NTT][4];
#pragma unroll
  for (int n = 0; n < NTT; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) { acc[n][k] = 0.f; c1[n][k] = 0.f; c2[n][k] = 0.f; c3[n][k] = 0.f; }
  const uint4 sc = unit_scales(unit, warp, lane);
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    const uint4 w = unit_codes(unit, warp, c, lane);
    uint32_t sg, sg8;
    scale_pair(sc, c, k16, sg, sg8);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    uint32_t b[NTT][8];
#pragma unroll
    for (int n = 0; n < NTT; ++n) {
      uint4 lo, hi;
      xf(n, c, lo, hi);
      b[n][0] = lo.x; b[n][1] = lo.y; b[n][2] = lo.z; b[n][3] = lo.w;
      b[n][4] = hi.x; b[n][5] = hi.y; b[n][6] = hi.z; b[n][7] = hi.w;
    }
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      uint32_t a[4];
      decode_step(ww[st], sg, sg8, a);
#pragma unroll
      for (int n = 0; n < NTT; ++n)
        mma_bf16(st == 0 ? acc[n] : st == 1 ? c1[n] : st == 2 ? c2[n] : c3[n], a[0], a[1], a[2], a[3],
                 b[n][2 * st], b[n][2 * st + 1]);
    }
  }
#pragma unroll
  for (int n = 0; n < NTT; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[n][k] += c1[n][k] + c2[n][k] + c3[n][k];
}

// Cross-warp sum of one tile for this lane: parts[w * stride] is warp w's float4, summed in warp order from 0.f.
__device__ __forceinline__ float4 sum_warps(const float4* parts, int stride_f4) {
  float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int w = 0; w < NWU; ++w) {
    const float4 t = parts[w * stride_f4];
    sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
  }
  return sm;
}

__device__ __forceinline__ float gscale(float gs) { return gs * TWO62; }

// Phase A epilogue for one (inter column 8 rb + g, token): gate / up are sm.x / sm.z (token 2t) or sm.y / sm.w
// (token 2t + 1); gsc = gscale(w13_gs[e]). h is rounded to BF16, SiLU runs in FP32 with __expf (MUFU.EX2 after the
// non-ftz range fix) and __fdividef (MUFU.RCP), the product with up is rounded once to BF16.
__device__ __forceinline__ __nv_bfloat16 swiglu_bf16(float gate_acc, float up_acc, float gsc) {
  const float gt = bf16r(gate_acc * gsc), up = bf16r(up_acc * gsc);
  return __float2bfloat16(__fdividef(gt, 1.f + __expf(-gt)) * up);
}

// (1 + post_norm_w[n]) as helper0 forms it: FP32 add, rounded to BF16. raw = the bf16 bits of post_norm_w[n].
__device__ __forceinline__ float norm_mul(unsigned short raw) { return bf16r(1.f + __uint_as_float((uint32_t)raw << 16)); }

// Phase B epilogue for one token: rows n0 = 16 rb + g and n0 + 8; gsc = gscale(w2_gs[e]). y is rounded to BF16, then
// times bf16(1 + w) in FP32 (exact: both factors are BF16); v0 / v8 are what helper0 stores to ybuf (FP32).
// Returns the lane's v0^2 + v8^2.
__device__ __forceinline__ float ynorm_pair(float acc0, float acc8, float gsc, float nw0, float nw8, float& v0, float& v8) {
  v0 = bf16r(acc0 * gsc) * nw0;
  v8 = bf16r(acc8 * gsc) * nw8;
  return v0 * v0 + v8 * v8;
}
// Sum over the 8 g-lanes of one t (16 rows): the unit's partial sum of squares for that token, in every lane.
// helper0 adds it to the token's row total with one atomicAdd per unit, so the row total's order is the arrival order.
__device__ __forceinline__ float ss_unit(float sq) {
  sq += __shfl_xor_sync(0xffffffffu, sq, 4);
  sq += __shfl_xor_sync(0xffffffffu, sq, 8);
  sq += __shfl_xor_sync(0xffffffffu, sq, 16);
  return sq;
}

// Route scale of one (token, slot): topk_w * rsqrt(ss / 4096 + eps); ss is the full-row sum of squares.
__device__ __forceinline__ float route_scale(float topk_w, float ss) {
  return topk_w * rsqrtf(ss * (1.f / HD) + POST_NORM_EPS);
}
// out[t, col .. col + 3] = bf16(sum over slots s = 0..7, in slot order, of r_s * y_s); y_s are the FP32 ybuf rows.
// Slots with q[s] < 0 (not local) are skipped.
__device__ __forceinline__ uint2 combine4(const int (&q)[8], const float (&r)[8], const float4 (&y)[8]) {
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (q[s] >= 0) {
      acc.x += r[s] * y[s].x; acc.y += r[s] * y[s].y; acc.z += r[s] * y[s].z; acc.w += r[s] * y[s].w;
    }
  }
  __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
  uint2 o;
  o.x = *reinterpret_cast<uint32_t*>(&o0);
  o.y = *reinterpret_cast<uint32_t*>(&o1);
  return o;
}

}  // namespace clm
