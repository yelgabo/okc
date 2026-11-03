#!/usr/bin/env python3
"""Build the UNP layer candidate from the public release file: at T in SL_TILE_CONFIG (1 and 8) the unpooler slice GEMV
and _sl_cluster_finish run in the tail of the fused MoE launch (tiny / tiny2 builds, okc_run_unp), gated by a
prepare-time self-test against the separate launches; the separate path stays the fallback (UNP_FUSED, the gate).

    python3 tools/dl_unp_patch.py muse_decode_layer/cand/g1b4_stage_release.py muse_decode_layer/cand/g1b4_unp_release.py
"""
import re
import sys


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one occurrence, found {n}")
    return text.replace(old, new)


CUDA_EDITS = [
    ("params", """  __nv_bfloat16* out;
  unsigned long long* dbg;
};
""", """  __nv_bfloat16* out;
  unsigned long long* dbg;
  // UNP (tiny builds, T <= 8): the unpooler slice and the post-FFN boundary fused into this launch; nU = 0 keeps the
  // plain launch
  const __nv_bfloat16* unp;      // [1024][HD] bf16 unpooler slice rows
  __nv_bfloat16* P;              // [M][1024] slice output
  const float* carry; const float* tail; const float* beta;   // residual carry [M][2 HD], its tail sum [M], beta [>= 1024]
  const __nv_bfloat16* post; const __nv_bfloat16* nextw;      // 1 + w of the post-FFN norm [1024] and the next pre-attention norm [2 HD]
  float* sout; __nv_bfloat16* hout;                           // [M][2 HD] outputs
  int* ucnt;    // [4]: final items done, slice units done, each by launch generation like the barrier counters
  int nU;       // slice units in this launch (UNP_UNITS) or 0
};
"""),
    ("wsr", """  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
""", """  int wsr[6];                          // WS: this CTA's phase A, phase B and (UNP) slice unit ranges
"""),
    ("define", """#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#define OKC_NOLOAD 0
""", """#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#define OKC_NOLOAD 0
// fused unpooler slice + post-FFN boundary tail (okc_run_unp): the warp-specialized tiny ring with the column final pass
#define OKC_UNP (OKC_TINY && OKC_WS && !OKC_WGS && !OKC_SPLIT && OKC_FINCOL)
"""),
    ("constants", """constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
""", """constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
// UNP: a slice unit is 4 unpooler rows x HD bf16 = 32 KB in a ring slot; a boundary item is one token x NT * 4 columns
constexpr int UNP_ROWS = 1024, UNP_UR = 4, UNP_UNITS = UNP_ROWS / UNP_UR, UNP_UB = UNP_UR * HD * 2;
constexpr int UNP_FSEG = HD / (NT * 4), HD2 = 2 * HD, UNP_FB = NT * 4, UNP_NSEG = HD2 / UNP_FB;
static_assert(UNP_UB <= UNIT_BYTES && UNP_ROWS == NT * 4, "UNP: a slice unit fits a ring slot; one thread per 4 head columns");
"""),
    ("issue_u", """template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);
}
""", """template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);
}
#if OKC_UNP
// one slice unit (read-only weights, written by prepare) into a ring slot: the same bulk copy and mbarrier as ring_issue
__device__ __forceinline__ void ring_issue_u(const Params& p, int u, unsigned char* slot, uint64_t* bar) {
  const unsigned char* src = reinterpret_cast<const unsigned char*>(p.unp) + (size_t)u * UNP_UB;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNP_UB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNP_UB), "r"(smem_u32(bar)) L2H_ARG : "memory");
}
#endif
"""),
    ("refill", """    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
""", """    if (tid == 0) {
      // UNP: the slice units queue behind phase B, so the last phase-B refills already fetch them
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]) + (OKC_UNP ? S.wsr[5] - S.wsr[4] : 0);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
"""),
    ("ws_issue", """__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}
""", """__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
#if OKC_UNP
  else if (q >= nA + (S.wsr[3] - S.wsr[2])) ring_issue_u(p, S.wsr[4] + (q - nA - (S.wsr[3] - S.wsr[2])), ring + sl * UNIT_BYTES, S.bars + sl);
#endif
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}
"""),
    ("preissue", """  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
""", """  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N + (OKC_UNP ? S.wsr[5] - S.wsr[4] : 0); ++q) ws_issue(p, S, q, nA, a0, b0, ring);
"""),
    ("ranges", """    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
""", """    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
#if OKC_UNP
    S.wsr[4] = muldiv_g(p.nU, c, G, rG); S.wsr[5] = muldiv_g(p.nU, c + 1, G, rG);
#endif
  }
  WS_JOIN();
"""),
    ("zero", """    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
""", """    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
#if OKC_UNP
    if (blockIdx.x == 0 && tid == 0) { p.ucnt[(S.gen + 1) & 1] = 0; p.ucnt[2 + ((S.gen + 1) & 1)] = 0; }
#endif
  }
"""),
    ("nfin0", """  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
""", """  int it0 = blockIdx.x;
  int nfin = 0;  // UNP: final items this CTA wrote
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
"""),
    ("nfin1", """    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
""", """    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
    ++nfin;
  }
  it0 += gridDim.x;
#endif
"""),
    ("nfin2", """    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
""", """    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
    ++nfin;
  }
#if OKC_UNP
  if (p.nU > 0) unp_tail(p, S, reinterpret_cast<unsigned char*>(xbuf), nfin);
#endif
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
"""),
    ("tail_fn", """template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
""", r'''#if OKC_UNP
// Fused tail (tiny builds, T <= 8): the unpooler slice GEMV (UNP_ROWS x HD bf16 rows as 4-row units through the same
// ring, queued behind the last phase-B unit) and the post-FFN boundary of _sl_cluster_finish, in place of two launches.
// Both waits are counters, not grid barriers: a slice unit waits for every final-pass item (its x rows are this
// launch's out), the boundary waits for every slice unit. Each wait repeats the grid barrier's pattern (PTX ISA 8.9
// memory consistency model: the writer's fence.sc.gpu then its atomic, the reader's ld.acquire.gpu, bar.sync carrying
// the order to the CTA's other threads). Everything read after an acquire is a generic store read by a generic load
// (__ldcg: no L1 copy), so that edge needs no proxy fence; a ring refill behind the generic reads of a slot keeps the
// phase loops' fence.proxy.async.shared::cta (PTX ISA 9.7.12.4, membar/fence: generic-proxy before async-proxy).
__device__ void unp_tail(const Params& p, Smem& S, unsigned char* ring, int nfin) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  int* fin = p.ucnt + (S.gen & 1);
  int* ucnt = p.ucnt + 2 + (S.gen & 1);
  const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
  const int u0 = S.wsr[4], nU = S.wsr[5] - u0;
  const int M = p.M;
  if (nfin > 0) {  // this CTA's final items are published
    CTA_SYNC();
    if (tid == 0) { __threadfence(); atomicAdd(fin, nfin); }
  }
  if (nU > 0) {
    if (tid == 0) { while (ld_acquire(fin) < M * UNP_FSEG) __nanosleep(64); }
    CTA_SYNC();
    uint4 xr[16];  // out rows: this warp's 512-K slice of every token, 16 bf16 per lane (rows >= M are zero)
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      if (t < M) {
        const uint4* src = reinterpret_cast<const uint4*>(p.out + (size_t)t * HD + warp * 512 + lane * 16);
        xr[2 * t] = __ldcg(src); xr[2 * t + 1] = __ldcg(src + 1);
      } else {
        xr[2 * t] = make_uint4(0u, 0u, 0u, 0u); xr[2 * t + 1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    for (int i = 0; i < nU; ++i) {
      const int q = N + i, sl = q % OKC_NS;
      mbar_wait(S.bars + sl, (uint32_t)((q / OKC_NS) & 1));
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      float acc[UNP_UR][8];
#pragma unroll
      for (int r = 0; r < UNP_UR; ++r)
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[r][t] = 0.f;
#pragma unroll
      for (int r = 0; r < UNP_UR; ++r) {
        const uint4 w0 = *reinterpret_cast<const uint4*>(slot + r * (HD * 2) + warp * 1024 + lane * 32);
        const uint4 w1 = *reinterpret_cast<const uint4*>(slot + r * (HD * 2) + warp * 1024 + lane * 32 + 16);
        const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const uint32_t xx[8] = {xr[2 * t].x, xr[2 * t].y, xr[2 * t].z, xr[2 * t].w,
                                  xr[2 * t + 1].x, xr[2 * t + 1].y, xr[2 * t + 1].z, xr[2 * t + 1].w};
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            acc[r][t] = fmaf(__uint_as_float(ww[k] << 16), __uint_as_float(xx[k] << 16), acc[r][t]);
            acc[r][t] = fmaf(__uint_as_float(ww[k] & 0xffff0000u), __uint_as_float(xx[k] & 0xffff0000u), acc[r][t]);
          }
        }
      }
      float mine = 0.f;  // lane (r, t) = (lane >> 3, lane & 7) keeps that pair's sum over the warp's K range
#pragma unroll
      for (int r = 0; r < UNP_UR; ++r)
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          float v = acc[r][t];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
          mine = (lane == r * 8 + t) ? v : mine;
        }
      const int bb = i & 1;
      S.red[bb][warp][lane][0] = mine;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // this warp's generic reads of the slot precede its refill
      CTA_SYNC();
      if (tid == 0 && q + OKC_NS < N + nU) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (warp == 0) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += S.red[bb][w][lane][0];
        const int r = lane >> 3, t = lane & 7;
        if (t < M) p.P[(size_t)t * UNP_ROWS + (u0 + i) * UNP_UR + r] = __float2bfloat16_rn(s);
      }
    }
    CTA_SYNC();
    if (tid == 0) { __threadfence(); atomicAdd(ucnt, nU); }
  }
  // the boundary: item = (token, UNP_FB columns), thread = 4 columns; both row reductions run per item
  const int nitems = M * UNP_NSEG;
  if ((int)blockIdx.x >= nitems) return;
  if (tid == 0) { while (ld_acquire(ucnt) < p.nU) __nanosleep(64); }
  CTA_SYNC();
  float* fr = &S.red[2][0][0][0];
  const int c4 = tid * 4;
  for (int it = blockIdx.x; it < nitems; it += gridDim.x) {
    const int tk = it / UNP_NSEG, seg = it - tk * UNP_NSEG, o = seg * UNP_FB + c4;
    const float4 oc = __ldg(reinterpret_cast<const float4*>(p.carry + (size_t)tk * HD2 + o));
    const uint2 gnb = __ldg(reinterpret_cast<const uint2*>(p.nextw + o));
    const float tlv = __ldg(p.tail + tk);
    const uint2 pb = __ldcg(reinterpret_cast<const uint2*>(p.P + (size_t)tk * UNP_ROWS + c4));
    const uint2 gpb = __ldg(reinterpret_cast<const uint2*>(p.post + c4));
    const float4 hc = __ldg(reinterpret_cast<const float4*>(p.carry + (size_t)tk * HD2 + c4));
    const float4 bt = __ldg(reinterpret_cast<const float4*>(p.beta + c4));
    const uint32_t pw[2] = {pb.x, pb.y}, gw[2] = {gpb.x, gpb.y}, gn[2] = {gnb.x, gnb.y};
    float f[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float pv = __uint_as_float((k & 1) ? (pw[k >> 1] & 0xffff0000u) : (pw[k >> 1] << 16));
      const float gv = __uint_as_float((k & 1) ? (gw[k >> 1] & 0xffff0000u) : (gw[k >> 1] << 16));
      f[k] = fork_bf16(pv * gv);
    }
    float q = f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3];
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o2);
    if (lane == 0) fr[warp] = q;
    CTA_SYNC();
    q = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) q += fr[w];
    const float inv = rsqrtf(q * (1.f / HD2) + 1e-8f);
    const float cv[4] = {hc.x, hc.y, hc.z, hc.w}, bv[4] = {bt.x, bt.y, bt.z, bt.w};
    float hd[4], q2 = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) { hd[k] = fmaf(bv[k], fork_bf16(f[k] * inv), cv[k]); q2 += hd[k] * hd[k]; }
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) q2 += __shfl_xor_sync(0xffffffffu, q2, o2);
    if (lane == 0) fr[NW + warp] = q2;
    CTA_SYNC();
    q2 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) q2 += fr[NW + w];
    const float invn = rsqrtf((tlv + q2) * (1.f / HD2) + 1e-5f);
    const float res[4] = {seg == 0 ? hd[0] : oc.x, seg == 0 ? hd[1] : oc.y, seg == 0 ? hd[2] : oc.z, seg == 0 ? hd[3] : oc.w};
    *reinterpret_cast<float4*>(p.sout + (size_t)tk * HD2 + o) = make_float4(res[0], res[1], res[2], res[3]);
    uint32_t hw[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const float g0 = __uint_as_float(gn[k] << 16), g1 = __uint_as_float(gn[k] & 0xffff0000u);
      const __nv_bfloat162 v = __floats2bfloat162_rn((res[2 * k] * invn) * g0, (res[2 * k + 1] * invn) * g1);
      hw[k] = *reinterpret_cast<const uint32_t*>(&v);
    }
    *reinterpret_cast<uint2*>(p.hout + (size_t)tk * HD2 + o) = make_uint2(hw[0], hw[1]);
    CTA_SYNC();  // fr is reused by the next item
  }
}
#endif

template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
'''),
    ("host_head", """void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr();""", """static void okc_launch(std::vector<torch::Tensor>& w, std::vector<torch::Tensor>& scratch, torch::Tensor& x, torch::Tensor& ids,
                       torch::Tensor& tw, torch::Tensor& out, int64_t lo, int64_t nloc, int64_t grid, okc::Params& p) {
  p.wA = (const uint4*)w[0].data_ptr();"""),
    ("host_cnt", """  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
""", """  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
#if OKC_UNP
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX + 4, "UNP counters follow the second ss copy");
  p.ucnt = ib + 8 + 1024 + okc::EMAX;
#endif
"""),
    ("host_tail", """  TORCH_CHECK(launch==cudaSuccess,"FORK_MOE_LAUNCH ",cudaGetErrorString(launch));

}
""", """  TORCH_CHECK(launch==cudaSuccess,"FORK_MOE_LAUNCH ",cudaGetErrorString(launch));

}

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p{};
  okc_launch(w, scratch, x, ids, tw, out, lo, nloc, grid, p);
}

// the MoE launch with the unpooler slice GEMV and _sl_cluster_finish's boundary fused into its tail (T <= 8)
void okc_run_unp(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
                 torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid, torch::Tensor unp,
                 torch::Tensor P, torch::Tensor carry, torch::Tensor tail, torch::Tensor beta, torch::Tensor post,
                 torch::Tensor nextw, torch::Tensor sout, torch::Tensor hout) {
#if OKC_UNP
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "okc_run_unp: M <= 8");
  auto ok16 = [](const torch::Tensor& t) { return t.is_contiguous() && (((uintptr_t)t.data_ptr()) & 15) == 0; };
  TORCH_CHECK(unp.scalar_type() == torch::kBFloat16 && unp.dim() == 2 && unp.size(0) == okc::UNP_ROWS && unp.size(1) == okc::HD && ok16(unp),
              "okc_run_unp: unpooler slice [1024, 4096] bf16");
  TORCH_CHECK(P.scalar_type() == torch::kBFloat16 && P.numel() == M * okc::UNP_ROWS && ok16(P), "okc_run_unp: P [M, 1024] bf16");
  TORCH_CHECK(carry.scalar_type() == torch::kFloat32 && carry.numel() == M * okc::HD2 && ok16(carry), "okc_run_unp: carry [M, 8192] f32");
  TORCH_CHECK(tail.scalar_type() == torch::kFloat32 && tail.numel() >= M && tail.is_contiguous(), "okc_run_unp: tail [M] f32");
  TORCH_CHECK(beta.scalar_type() == torch::kFloat32 && beta.numel() >= okc::UNP_ROWS && ok16(beta), "okc_run_unp: beta f32");
  TORCH_CHECK(post.scalar_type() == torch::kBFloat16 && post.numel() >= okc::UNP_ROWS && ok16(post), "okc_run_unp: post [1024] bf16");
  TORCH_CHECK(nextw.scalar_type() == torch::kBFloat16 && nextw.numel() >= okc::HD2 && ok16(nextw), "okc_run_unp: next norm [8192] bf16");
  TORCH_CHECK(sout.scalar_type() == torch::kFloat32 && sout.numel() == M * okc::HD2 && ok16(sout), "okc_run_unp: sout [M, 8192] f32");
  TORCH_CHECK(hout.scalar_type() == torch::kBFloat16 && hout.numel() == M * okc::HD2 && ok16(hout), "okc_run_unp: hout [M, 8192] bf16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.numel() == M * okc::HD && ok16(out), "okc_run_unp: out [M, 4096] bf16");
  okc::Params p{};
  p.unp = (const __nv_bfloat16*)unp.data_ptr(); p.P = (__nv_bfloat16*)P.data_ptr();
  p.carry = (const float*)carry.data_ptr(); p.tail = (const float*)tail.data_ptr(); p.beta = (const float*)beta.data_ptr();
  p.post = (const __nv_bfloat16*)post.data_ptr(); p.nextw = (const __nv_bfloat16*)nextw.data_ptr();
  p.sout = (float*)sout.data_ptr(); p.hout = (__nv_bfloat16*)hout.data_ptr();
  p.nU = okc::UNP_UNITS;
  okc_launch(w, scratch, x, ids, tw, out, lo, nloc, grid, p);
#else
  TORCH_CHECK(false, "okc_run_unp: this build has no fused tail");
#endif
}
"""),
]

PY_EDITS = [
    ("cpp_decl", """void okc_set_pdl(int64_t on);
std::vector<int64_t> okc_attrs();
\"\"\"
""", """void okc_set_pdl(int64_t on);
std::vector<int64_t> okc_attrs();
void okc_run_unp(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
                 torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid, torch::Tensor unp,
                 torch::Tensor P, torch::Tensor carry, torch::Tensor tail, torch::Tensor beta, torch::Tensor post,
                 torch::Tensor nextw, torch::Tensor sout, torch::Tensor hout);
\"\"\"
"""),
    ("functions", """            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),
""", """            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else [])
            + (["okc_run_unp"] if "okc_run_unp" in src else []),
"""),
]

TILE_EDIT = ("""        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
""", """        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        if _unp_fused(state, T, ext):
            out, s2 = _unp_tail_run(ext, state, xm, lids, wts, m, s1, tail, g)
            return out, s2, ids
        for m0 in range(0, T, _MA_MMAX):
""")

PY_TAIL = r'''

UNP_FUSED = True   # T in SL_TILE_CONFIG: the unpooler slice GEMV and _sl_cluster_finish in the MoE launch's tail (gated below)
UNP_SIZES = (1, 8)
UNP_BOUNDS = {"m": 1e-3, "P": 2e-3, "s2": 2e-3, "out": 2e-3, "elem": 1.0 / 64}
UNP_LOSS_US = 0.5   # the fused tail stays off at a T where the prepare-time tails time it slower by more than this


def _unp_fused(state, T, ext):
    return UNP_FUSED and T in SL_CLUSTER_CONFIG and state.get("unp_fused", {}).get(T, False) and hasattr(ext, "okc_run_unp")


def _unp_tail_run(ext, state, xm, lids, wts, m, s1, tail, g, P=None):
    dev = xm.device
    T = xm.shape[0]
    if P is None:
        P = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
    s2 = torch.empty((T, HIDDEN), device=dev, dtype=torch.float32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    ext.okc_run_unp(state["ma_w"], state["ma_scratch"], xm, lids, wts, m, 0, 32, state["nsm"],
                    state["unpool_slice"], P, s1, tail, g["beta2"], g["post2"], g["pre2"], s2, out)
    return out, s2


def _unp_err(a, b):
    a, b = a.float(), b.float()
    if not bool(torch.isfinite(a).all()):
        return float("inf"), float("inf")
    rel = float((a - b).norm() / b.norm().clamp_min(1e-8))
    rms = b.pow(2).mean(-1, keepdim=True).sqrt()
    elem = float(((a - b).abs() / (b.abs() + rms + 1e-30)).max())
    return rel, elem


def _unp_self_test(state, deadline=None):
    """Fused tail against the separate launches (MoE, sk2 slice, _sl_cluster_finish) on fresh inputs, eagerly and as
    graph replays with fresh inputs at the captured pointers; then both tails timed as graphs. A T is enabled only
    when every comparison is within UNP_BOUNDS and the fused tail is not slower by more than UNP_LOSS_US."""
    import time
    picks = state.setdefault("unp_fused", {})
    if not (UNP_FUSED and state.get("_fork_native_eligible") and "graft" in state and state.get("bfp") is not None):
        print("UNP_GATE off: native MoE or graft path unavailable", flush=True)
        return
    dev = state["unpool_slice"].device
    g = state["graft"]
    gen = torch.Generator(device=dev).manual_seed(20261102)
    flush = torch.ones(16 * 1024 * 1024, dtype=torch.float32, device=dev)   # 64 MB read sweep between timed replays
    global _FLUSH_SINK
    if _FLUSH_SINK is None:
        _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    for T in UNP_SIZES:
        if deadline is not None and time.monotonic() >= deadline:
            print(f"UNP_GATE T={T} skipped: prepare budget", flush=True)
            continue
        if T not in SL_CLUSTER_CONFIG or T not in SL_TILE_CONFIG:
            continue
        ext = _fork_native_moe(T, state)
        if not hasattr(ext, "okc_run_unp"):
            print(f"UNP_GATE T={T} off: build has no okc_run_unp", flush=True)
            continue
        routes = _release_routes(torch, T, dev)
        banks = torch.randn((4, T, D_EXPERT), device=dev, dtype=torch.bfloat16, generator=gen)
        scores = torch.randn((4, T, 8), device=dev, dtype=torch.float32, generator=gen).softmax(-1)
        carries = torch.randn((4, T, HIDDEN), device=dev, dtype=torch.float32, generator=gen) * 3.0
        tails = torch.rand((4, T), device=dev, dtype=torch.float32, generator=gen) * 4096.0 + 1.0
        xm, lids, wts = banks[0].clone(), routes[0].clone(), scores[0].clone()
        s1, tail = carries[0].clone(), tails[0].clone()
        nw, bc = SL_CLUSTER_CONFIG[T]

        def sep():
            m = torch.empty_like(xm)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm, lids, wts, m, 0, 32, state["nsm"])
            U, S = dense(m, state["unpool_slice"], T, state, "un")
            if S != 1 or U.dtype != torch.bfloat16:
                raise RuntimeError("UNP gate: the separate path is not the BF16 sliced unpooler")
            s2 = torch.empty((T, HIDDEN), device=dev, dtype=torch.float32)
            out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            _sl_cluster_finish[(T, 8192 // bc)](U, T, s1, tail, g["beta2"], g["post2"], g["pre2"], s2, out, BC=bc,
                                                num_warps=nw, launch_pdl=False)
            return m, U.view(T, 1024), s2, out

        def fused():
            m = torch.empty_like(xm)
            P = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
            out, s2 = _unp_tail_run(ext, state, xm, lids, wts, m, s1, tail, g, P=P)
            return m, P, s2, out

        def load(i):
            xm.copy_(banks[i]); lids.copy_(routes[i]); wts.copy_(scores[i]); s1.copy_(carries[i]); tail.copy_(tails[i])

        try:
            refs = []
            for i in range(4):
                load(i)
                refs.append(tuple(t.clone() for t in sep()))
                torch.cuda.synchronize()
            worst = {k: 0.0 for k in ("m", "P", "s2", "out")}
            worst_elem = 0.0
            ctrl = {k: 0.0 for k in ("m", "P", "s2", "out")}

            def judge(got, ref, into, into_elem):
                for k, a, b in zip(("m", "P", "s2", "out"), got, ref):
                    rel, elem = _unp_err(a, b)
                    into[k] = max(into[k], rel)
                    into_elem[0] = max(into_elem[0], elem)

            we = [0.0]
            load(0)
            judge(fused(), refs[0], worst, we)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = fused()
            ce = [0.0]
            cgraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cgraph):
                cgot = sep()
            for i in (1, 2, 3, 0, 2, 1, 3, 0):
                load(i)
                graph.replay()
                torch.cuda.synchronize()
                judge(got, refs[i], worst, we)
                cgraph.replay()
                torch.cuda.synchronize()
                judge(cgot, refs[i], ctrl, ce)
            worst_elem = we[0]
            ok = all(worst[k] <= UNP_BOUNDS[k] for k in worst) and worst_elem <= UNP_BOUNDS["elem"]
            load(0)
            t_sep = min(_graph_time(lambda: sep()[3], flush, reps=15), _graph_time(lambda: sep()[3], flush, reps=15))
            t_fused = min(_graph_time(lambda: fused()[3], flush, reps=15), _graph_time(lambda: fused()[3], flush, reps=15))
            del graph, cgraph
            win = t_fused <= t_sep + UNP_LOSS_US
            picks[T] = bool(ok and win)
            print(f"UNP_GATE T={T} {'ok' if ok else 'FAIL'} fused rel m {worst['m']:.1e} P {worst['P']:.1e} s2 {worst['s2']:.1e} "
                  f"out {worst['out']:.1e} elem {worst_elem:.1e} | control replay rel m {ctrl['m']:.1e} P {ctrl['P']:.1e} "
                  f"s2 {ctrl['s2']:.1e} out {ctrl['out']:.1e} elem {ce[0]:.1e} | tails us sep {t_sep:.1f} fused {t_fused:.1f} "
                  f"-> {'fused' if picks[T] else 'separate'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            picks[T] = False
            print(f"UNP_GATE T={T} FAIL {type(exc).__name__}: {str(exc)[:200]} -> separate", flush=True)
            torch.cuda.synchronize()
    del flush
    _check_guards()


_unp_parent_prepare = prepare
def prepare(weights):
    import time
    t0 = time.monotonic()
    state = _unp_parent_prepare(weights)
    _unp_self_test(state, deadline=t0 + 285)
    return state
'''


def build(src_path, dst_path):
    text = open(src_path).read()
    key = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\""
    a = text.index(key) + len(key)
    b = text.index('"""', a)
    helper = text[a:b]
    for label, old, new in CUDA_EDITS:
        helper = once(helper, old, new, "cuda " + label)
    text = text[:a] + helper + text[b:]
    for label, old, new in PY_EDITS:
        text = once(text, old, new, "py " + label)
    fa = text.index("def _sl_tile_after_attention(")
    fb = text.index("_sl_tile_prior_after=_after_attention", fa)
    func = once(text[fa:fb], TILE_EDIT[0], TILE_EDIT[1], "py tile")
    text = text[:fa] + func + text[fb:]
    if not text.endswith("\n"):
        text += "\n"
    text += PY_TAIL
    open(dst_path, "w").write(text)
    print(dst_path, len(text))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
