# okc: problem=muse_moe_w4a16

"""PRIVATE SOURCE-ONLY ADAPTER. dispatch_ready=False; never submit publicly."""

# stack_all1: stack_mf1 (MFIN, no MoE grid barrier) + SC6 (6-bit block scales, WGS MoE builds) + PRF32 + PDL at T=128 (moe-fin bundle; L2 hints at T=1 left out: judge d1 25.7-26.6 us with them vs 24.5-25.8 without).
# st2l: st1l with OKC_WGSFCI 4 (E374).
# st1l: ybfl.py + OKC_WGSFCI 5 + weight-ring pacing at M <= 64 (E372).
# ybfl: cepil_u160.py + WGS ybuf in bf16 (epilogue store + final-pass reads; harness/make_ybf.py).
# cepil_u160: pcx_h1 + in-layer MoE per-unit stream-K cost U = 160 (A) / 140 (B), zhebrak SK_CEPI (harness/make_cepi.py --layer).
# pcx_h1: pre-wait control of the sk2 kernels: hoist (harness/make_pcx.py; team keell0renz; base stack_all1_snap1)
# + alygg77/okc-muse-spark-kernels kernels/muse_decode_layer.py (Apache-2.0, commit 6cae49a); header below is theirs. Patched by team keell0renz: trig, rq, atrig0, rtrig8, pf4.
# bt_sc6: lossless 6-bit block scales in the layer's WGS MoE builds (T >= 17), moe-scale SC6 via harness/make_bt_sc6.py.
# stack_df1: data-flag A->B hand-off in the MoE MID build (T = 9..16), harness/make_dl_df.py (moe-small E173).
# + chenwainuo/okc muse_bf16_proj rs128 GEMVs (Apache-2.0) at T = 128 / 256 for qkv and T = 128 for the unpooler (team keell0renz).
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
# Modified by yelgabo: G1b4 stage-retire experiment, BF16 attention value, empty routing and Marlin fallback, parallel native preparation, native MoE, repaired precision and ordered layer dispatch.

import time
import torch
from torch.utils.cpp_extension import load_inline

EXPERT_MOE_DISPATCH_READY = False

EXPERT_MOE_IDENTITIES = ('moe_pairlut', 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab', 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71', '8c2cfc7cd87dd863bf32cd6389b9f056e954329201fee1d7011a39efe5ce31c8')

MOE_CPP_SRC = '\n#include <torch/extension.h>\nvoid okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);\nvoid okc_set_pdl(int64_t on);\nstd::vector<int64_t> okc_attrs();\n'

MOE_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1')

MOE_TINY_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_L2H=0', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1')

MOE_TINY_MAX_M = 4

MOE_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0')

MOE_TINY2_MAX_M = 8

MOE_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '@moe_helper0df')

MOE_MID_MAX_M = 16

MOE_HAS_PDL = False

DQ7 = True

_REG_HJ = ((0, 0), (1, 0), (0, 2), (1, 2))

SC6_WGS = True

_MA_PRMT_SCALES = True

_MA_SFX_IF_PRMT = ('-DOKC_PRMTSC=1',)

MAIN_FALLBACK_DEFINES = ('-DOKC_CPF=0', '-DOKC_CPFF=0', '-DOKC_DYNK=7', '-DOKC_EARLYN=2', '-DOKC_EPI4=0', '-DOKC_FCOSTB=35', '-DOKC_FCOSTBB=36', '-DOKC_FINCOL=0', '-DOKC_FPF=5')

_MA_MAIN_NS = 3

_MA_PDL = False

SCRATCH_SENTINEL = 1588444911

_EXPERT_PAIR_EDITS = [("          uint32_t r0, r1, r2, r3;\n          dq4(q, r0, r1, r2, r3);\n          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in\n          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer\n#if OKC_SC6W\n          const uint32_t sgp = sgs[js];\n#else\n          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;\n          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n#endif\n          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);", '#if OKC_SC6W\n          const uint32_t sgp = sgs[js];\n          const uint32_t lo = (sgp & 0x03f0u) << 4;\n          const uint32_t hi = ((sgp >> 16) & 0x03f0u) << 4;\n          af[ci & 1][jj][0] = pair_lookup(p.lut,lo | (q & 255u));\n          af[ci & 1][jj][1] = pair_lookup(p.lut,hi | ((q >> 8) & 255u));\n          af[ci & 1][jj][2] = pair_lookup(p.lut,lo | ((q >> 16) & 255u));\n          af[ci & 1][jj][3] = pair_lookup(p.lut,hi | (q >> 24));\n#else\n          uint32_t r0, r1, r2, r3;\n          dq4(q, r0, r1, r2, r3);\n          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;\n          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);\n#endif', 1), ('template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(', '__device__ __forceinline__ uint32_t pair_lookup(const uint32_t* table,uint32_t index) {\n  uint32_t value;\n  asm volatile("{ .reg .u64 a; mad.wide.u32 a, %1, 4, %2; ld.global.nc.b32 %0, [a+1024]; }"\n               :"=r"(value):"r"(index),"l"(table));\n  return value;\n}\ntemplate <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(', 1)]

_MA_MMAX = 384

_MA_EMAX = _MA_MMAX * 8

WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16

WGS_XROWS = _MA_EMAX + 8 * (32 + _MA_EMAX // 32 + 8)

SCRATCH_GUARD = 1 << 16

MOE_CUDA_SRC = r'''
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_WGSFINSS 1
#ifndef OKC_MFIN
#define OKC_MFIN 1   // WGS: per-expert completion counters instead of the grid barrier before the final pass
#endif
#ifndef OKC_B4L
#define OKC_B4L 0    // WGS: sum-of-squares partials from the phase-B epilogue, single-read final pass
#endif
#ifndef OKC_MFP
#define OKC_MFP 15
#endif
#ifndef OKC_MFS
#define OKC_MFS 0    // nanosleep ns in the final-pass readiness poll (0: none)
#endif
#ifndef OKC_MFQ
#define OKC_MFQ 47   // 1 doneAs parity, 2 xready parity, 4 no grid barrier, 8 launch parity / zeroing, 32 volatile S.gen loads
#endif
#if !OKC_WGS
#undef OKC_MFIN
#define OKC_MFIN 0
#undef OKC_B4L
#define OKC_B4L 0
#endif
#ifndef OKC_B4FIN
#define OKC_B4FIN 0  // bet4_claude: column-parallel final pass, fixed-order ss partials (own build at 16 < M <= 32)
#endif
#ifndef B4_EARLY_Y
#define B4_EARLY_Y 1
#endif  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
#endif
#define OKC_WGMIN 9
#define OKC_XPRE 1
#define FAST1_ARCH 1
#ifndef OKC_FAST1
#define OKC_FAST1 3
#endif
#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)
#ifndef OKC_CPFF
#define OKC_CPFF 1
#endif
#ifndef OKC_FCOSTBB
#define OKC_FCOSTBB OKC_FCOSTB
#endif
#define OKC_FCOSTSB OKC_FCOST
#define OKC_DYN 1
#ifndef OKC_DYNK
#define OKC_DYNK 4
#endif
#define OKC_TDYN 0
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#define OKC_FT_T 64
#ifndef OKC_SPLIT
#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)
#endif
#ifndef OKC_TINY
#define OKC_TINY 0
#endif
#ifndef OKC_NW
#define OKC_NW 8
#endif
#ifndef OKC_EARLYN
#define OKC_EARLYN 1
#endif
#define OKC_BIGM 64
#define OKC_XPF 0
#ifndef OKC_CPF
#define OKC_CPF 1
#endif
#define OKC_FATSTATIC 1
#define OKC_FCOST 40
#ifndef OKC_FCOSTB
#define OKC_FCOSTB 40
#endif
#ifndef OKC_WS
#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue
#endif
#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped
#ifndef OKC_WGS
#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp
#endif
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef OKC_UWARP
#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers
#endif
#if OKC_UWARP
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#else
#define WARP_IDX(t) ((t) >> 5)
#endif
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#ifndef OKC_FINPF
#define OKC_FINPF 0   // skinny builds: final item's topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)
#endif
#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#define OKC_NOLOAD 0

namespace okc {

#if !OKC_WGS && !OKC_WS
#error FORK_MOE_UNPROVEN_FALLBACK_select_owned_or_Marlin_adapter
#endif
#if OKC_WGS
#undef OKC_MFIN
#define OKC_MFIN 0
#endif


#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
#else
#define L2H_SUFFIX ""
#define L2H_OPERAND ""
#define L2H_ARG
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
#endif

constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (OKC_DYNK per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr int NST = 3;          // fat x stages
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)
  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready;                           // WGS: per job, rows of xA written
  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
  int* mfl;                              // MFIN flag region (cE / exit counter / xready / doneAs, two parities)
  uint32_t zero;  // always 0 (opaque to the compiler)
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA;      // [JMAX]
  __nv_bfloat16* inter;  // [EMAX, ID]
  float* ybuf;     // [EMAX, HD]
  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1;      // [EMAX]
  __nv_bfloat16* out;
  unsigned long long* dbg;
  float* sspart;  // bet4_claude B4FIN: per-(route, 16-column tile) sum-of-squares partials (WGS only)
};

struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums
#else
  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free
  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
#if OKC_WGS
  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty
  int wglast[4];      // WGS: this warpgroup completed a split unit
  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job
  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job
  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)
  uint32_t wtot;      // WGS: phase-A total cost
  uint32_t wtotB;     // WGS: phase-B total cost
#else
  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)
#endif
#if OKC_WGS
  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue's job -> expert map)
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
};

// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).
// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,
// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe
  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}


// ---------------------------------------------------------------- routing prologue (every CTA, identical result)
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)
  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
#else
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = __ldg(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
#endif
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if OKC_SPLIT
    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }
#else
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
#endif
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
#if !OKC_WGS  // (the WGS build has no fat queue)
    // fat queue order: largest n-tile count first (LPT), stable by job index
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
  early_issue(p, S);
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}



// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float fork_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float probe_q(float v) {
  return v;
}
__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    if (tk < ntok) {
      float gt = fork_bf16(r[h] * gsc), up = fork_bf16(r[2 + h] * gsc);
      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));
    }
  }
}
// returns per-token partial sum of squares (2 values) for this lane's two tokens
__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,
                                      float* sq) {
  const float w0 = fork_bf16(1.f + __bfloat162float(p.normw[n0])), w1 = fork_bf16(1.f + __bfloat162float(p.normw[n0 + 8]));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    float v0 = fork_bf16(r[h] * gsc) * w0, v1 = fork_bf16(r[2 + h] * gsc) * w1;
    sq[h] = v0 * v0 + v1 * v1;
    if (tk < ntok) {
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);
    }
  }
}
__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);
  }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}

// ---------------------------------------------------------------- the persistent kernel


// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
#ifndef OKC_SC6S
#define OKC_SC6S 0   // skinny units carry 6-bit scales (3072 B instead of 4096 B)
#endif
#ifndef OKC_SC6W
#define OKC_SC6W 0   // WGS stages carry 6-bit scales (768 B instead of 1024 B per warpgroup piece)
#endif
constexpr int UNIT_CODES = 32768, UNIT_SC = OKC_SC6S ? 3072 : 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the
// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes
// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the
// ALU pipe, which the FP4 decode saturates
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#if OKC_SC6S
// SC6: the scale bytes come as 4 v (v = byte - 0x40, 6 bits): * 4 + 0x6000 per half == (0x40 + v) * 16 + 0x5C00
#undef SCDK
#define SCDK(word, b, k4) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k4) + 0x60006000u)
#endif
#endif
#define SCK ((OKC_SC6S ? 4u : 16u))
// a lane's 16 scale bytes of a skinny unit as 4 words (the layout of the old uint4): SC6 rebuilds them from 3 words
__device__ __forceinline__ uint4 sc_load(const unsigned char* slot, int warp, int lane) {
#if OKC_SC6S
  const unsigned char* b = slot + UNIT_CODES + warp * 128 + lane * 4;
  const uint32_t w0 = *reinterpret_cast<const uint32_t*>(b), w1 = *reinterpret_cast<const uint32_t*>(b + 1024),
                 w2 = *reinterpret_cast<const uint32_t*>(b + 2048);
  uint4 r;
  r.x = w0 & 0xFCFCFCFCu; r.y = w1 & 0xFCFCFCFCu; r.z = w2 & 0xFCFCFCFCu;
  r.w = ((w0 << 2) & 0x0C0C0C0Cu) | ((w1 << 4) & 0x30303030u) | ((w2 << 6) & 0xC0C0C0C0u);
  return r;
#else
  return *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);
}

template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB) row = p.inter + (size_t)(st + r) * ID;
      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

constexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4
  const int cnt = S.jC[j], st = S.jS[j];
  for (int i = lane; i < 8 * 64; i += 32) {
    const int r = i >> 6, c = i & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (8 + r < cnt) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;
      v = PB ? __ldcg(src) : __ldg(src);
    }
    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;
  }
  __syncwarp();
}

#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;  // opaque 16 (4 with SC6S): the scale decode IMAD
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill
    if (!PB) {
      const int jj = njF + (v >> rsh);
      S.dync[jj] += 1;
      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);
    }
    return v;
  };
  auto refill = [&](int sl) {
    const int v = take();
    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }
    else if (cpf && preB < nB) {
      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);
      S.uslot[sl] = BFLAG | (v0B + preB);
      ++preB;
    } else {
      S.uslot[sl] = -1;
    }
  };
  if (tid == 0)
    for (int i = pre; i < OKC_NS; ++i) {
      const int sl = (seq0 + i) % OKC_NS;
      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) {
    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);
    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);
    nwc0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwc1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)));
  }
  int nproc = 0;
  for (int i = 0;; ++i) {
    int v;
    if (i < nU) {
      v = v0 + i;  // static units: no slot lookup
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B's prefetched units follow)
      v &= ~BFLAG;
    }
    const int j = njF + (v >> rsh), rb = v & (rbs - 1);
    float nwn0 = 0.f, nwn1 = 0.f;
    int vn = -1;
    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];
    if (PB && tid < 128 && vn >= 0) {
      const int rbq = (vn & ~BFLAG) & (rbs - 1);
      nwn0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwn1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = sc_load(slot, warp, lane);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = fork_bf16(s0 * gsc), up = fork_bf16(s1 * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = fork_bf16(s0 * gsc) * nwc0, v1f = fork_bf16(s1 * gsc) * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;
  }
  seq = seq0 + nproc;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      if (nU > 0) {
        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;  // opaque 16 (4 with SC6S): the scale decode IMAD
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  for (int i = pre; i < OKC_NS; ++i) {
    const int sl = (seq0 + i) % OKC_NS;
    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }
  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwc1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8))); }
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    float nwn0 = 0.f, nwn1 = 0.f;
    if (PB && tid < 128 && i + 1 < nU) {
      const int rbq = (v + 1) % rbs;
      nwn0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwn1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = sc_load(slot, warp, lane);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i + OKC_NS - nU < nB) {
      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);
      ++preB;
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = fork_bf16(s0 * gsc), up = fork_bf16(s1 * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = fork_bf16(s0 * gsc) * nwc0, v1f = fork_bf16(s1 * gsc) * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
  }
  seq = seq0 + nU;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#endif

// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens
// ---------------------------------------------------------------- fat unit through the smem ring
// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x
// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile
// for the stage is staged by cp.async into two buffers that live in slots 3..4.
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
}

// ---------------------------------------------------------------- fat units: one continuous stream per CTA
// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps
// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit's weights).
// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};

// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from
// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads
  // k block cc at byte offset 32 st
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired
  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact
      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2
      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)
      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);
      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;
      dq4((&w0.x)[st], r0, r1, r2, r3);
      dq4((&w1.x)[st], q0, q1, q2, q3);
      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);
      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);
      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);
      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);
#undef FMG0
#undef FMG80
#undef FMG1
#undef FMG81
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);
      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);
      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);
    }
    wg_commit();
    if (cc >= 1) {
      wg_wait<1>();  // the previous chunk's group retired: its register set may be rewritten
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f
}

// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)
template <int NTT>
__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);
#if OKC_NW == 8
  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);
#endif
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
#if OKC_NW == 8
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
#endif
#if OKC_PRMTSC
    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);
#if OKC_NW == 8
    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);
#endif
#else
    const int sh = (cc & 1) * 16;
    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif
    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;
#pragma unroll
    for (int sp = 0; sp < 2; ++sp) {
      uint32_t a0[2][4], a1[2][4];
      (void)a1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s2 = 2 * sp + h;
        uint32_t r0, r1, r2, r3;
        dq4((&w0.x)[s2], r0, r1, r2, r3);
        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);
#if OKC_NW == 8
        dq4((&w1.x)[s2], r0, r1, r2, r3);
        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);
        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);
#endif
        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);
#endif
      }
    }
  }
}

template <bool PB, bool BIG>
__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;  // 16
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete
    if (PB) {
      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }
      CTA_SYNC();
    }
  };
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
  auto stage_x = [&](int j, int ks, int xsl) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int ntl = (cnt + 7) >> 3;
    const int nrow = ntl * 8;
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
    const int per = nrow * (KS / 8);
    if (is_nat(j)) {
      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)
      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);
      for (int i = tid; i < per; i += NT) {
        const int r = i >> 5, ch = i & 31;
        const bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;
        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);
      }
    } else {
      for (int i = tid; i < per; i += NT) {
        int r = i / (KS / 8), c = i - r * (KS / 8);
        bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;
        cp_async16(xs + r * XP_FT + c * 8, src, v);
      }
    }
    cp_commit();
  };
  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)
  const __nv_bfloat16* xsrc[8];
  uint32_t xdst[8];
  uint32_t xval = 0;
  int xn = 0;
  auto prep_x = [&](int j) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);
    xval = 0;
    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = tid + k * NT;
      const int r = i >> 5, c = i & 31;
      const bool v = i < per && r < cnt;
      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
      xdst[k] = r * XP_FT + c * 8;
      if (v) xval |= 1u << k;
    }
  };
  auto stage_xp = [&](int ks, int xsl) {
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);
    cp_commit();
  };
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int f = S.fseq;  // ring position of the current unit's k-stage 0
  int xq = 0;      // x stage parity counter
  int j, ub;
  job_of(u, j, ub);
#pragma unroll
  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));
  wait_A(j);
  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);
  while (true) {
    const int un = u + gridDim.x;
    int jn = 0, ubn = 0;
    if (un < nF) job_of(un, jn, ubn);
    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];
    const int ntile = (cnt + 7) >> 3;
    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)
    const bool nat = is_nat(j);
    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;
    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];
#pragma unroll
    for (int i = 0; i < FT_T / 8; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }
    for (int ks = 0; ks < NKS; ++ks) {
      if (OKC_XPRE) {
        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }
        else cp_commit();
      } else {
        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }
        else cp_commit();
      }
      cp_wait<1>();
      const int q = f + ks, sl = q % NSF;
      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // cp.async x writes -> wgmma reads
      CTA_SYNC();  // x stage visible to every warp
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);
      if (nat) {
        switch (ntc) {
          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
        }
      } else {
        switch (ntc) {
          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;
          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;
        }
      }
      CTA_SYNC();  // slot sl and the x stage consumed
      ++xq;
      // refill: the stage NSF ahead, in this unit or the next one
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
    // epilogue of unit u (registers -> global)
    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;
    if (!PB) {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);
#if OKC_NW == 8
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);
#endif
        }
      __threadfence();
      CTA_SYNC();
      if (tid == 0) atomicAdd(p.doneA + j, 1);
    } else {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          float sq0[2], sq1[2];
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);
#if OKC_NW == 8
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);
          sq0[0] += sq1[0]; sq0[1] += sq1[1];
#else
          (void)sq1;
#endif
          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);
        }
    }
    f += NKS;
    if (un >= nF) break;
    u = un; j = jn; ub = ubn;
  }
  cp_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}

template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}



// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring
#endif
  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region
    if (threadIdx.x == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in
  // skinny_ring issues the rest after routing, so every CTA's first unit is served ahead of the second ones
  const int n = min(OKC_EARLYN, v1 - v0);
  if (threadIdx.x == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}

// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost
#if OKC_WS
// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)
// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper
// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit's
// epilogue, so no CTA-wide barrier sits in the per-unit loop.
// the last WSBLK units keep their partial buffers until the compute warps' own epilogues: WSBLK > WSNR deadlocks the
// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;  // opaque 16 (4 with SC6S): the scale decode IMAD
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = sc_load(slot, warp, lane);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
  }
  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA's last units: epilogues here, not behind the helper's queue
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = fork_bf16(1.f + __uint_as_float(lnw0 << 16)), nw1 = fork_bf16(1.f + __uint_as_float(lnw1 << 16));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = fork_bf16((h ? sm.y : sm.x) * gsc) * nw0, v1f = fork_bf16((h ? sm.w : sm.z) * gsc) * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = fork_bf16(1.f + __bfloat162float(p.normw[n0])), nw1 = fork_bf16(1.f + __bfloat162float(p.normw[n0 + 8]));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = fork_bf16((h ? sm.y : sm.x) * gsc) * nw0, v1f = fork_bf16((h ? sm.w : sm.z) * gsc) * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B's skinny units are slower, so its ratio is lower)
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c); v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);
  }
}

#if OKC_WGS
// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)
// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,
// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales
// [grp][warp][g][h][j]; the stage's x slice (the job's token rows, rounded up to 8, x 256 K as wgmma B core matrices
// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16
// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).
// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a
// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.
#define OKC_WGSNX 3                             // x ring slots
constexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;  // one warpgroup's weights in a stage (SC6: 8 KB codes + 768 B scales)
constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;  // bytes of a stage in global memory (the smem slot keeps UNIT_BYTES)
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;                  // fixup slots per phase (>= grid + 1)
constexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
constexpr int MF_CE = 0, MF_EXIT = 64, MF_XR = 128, MF_XRS = 96, MF_DA = 384, MF_INTS = 4096;
static_assert(JMAX <= MF_XRS && MF_DA + 2 * JMAX * 16 <= MF_INTS, "MFIN flag region");
// phase-A hand-off flags: one set per launch parity under MFIN (the old code zeroed them after the grid barrier)
// (volatile shared load: a plain S.gen load is loop-invariant and would stay live in a register across the stage loops)
__device__ __forceinline__ int mf_gen(const Smem& S) {
#if OKC_MFQ & 32
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(&S.gen)));
  return v;
#else
  return S.gen;
#endif
}
__device__ __forceinline__ int* mf_xr(const Params& p, const Smem& S) { return p.xready + ((OKC_MFIN && (OKC_MFQ & 2)) ? mf_gen(S) * MF_XRS : 0); }
__device__ __forceinline__ int* mf_da(const Params& p, const Smem& S) { return p.doneAs + ((OKC_MFIN && (OKC_MFQ & 1)) ? mf_gen(S) * (JMAX * 16) : 0); }
// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at
// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {OKC_WGSA};
constexpr int WGS_CB[8] = {OKC_WGSB};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
}
#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)
#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time
#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather
#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B
#define OKC_WGSFCI 4  // st2 (E374): the deferred flush at commit group 4 (FCI 3..7 sweep best)
#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits
// Phase A runs a CTA's trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit's
// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase
// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the
// order (its first stages wait for phase A's units in order).
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      // st1 (wp2s): at M <= 64 stage n is issued only once stage n - 2 has landed (no 3-stage burst; E371 / E372)
      if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
// x producer: one bulk copy per stage (lane 0): the job's x slice for the stage is contiguous in xA (phase A, written
// by every CTA's gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B
constexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the
  // stage ranges while the consumers gather phase A's x
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  wgs_ranges(p, S);  // all 32 lanes (warp-parallel)
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x, nj = S.njF + S.njS;
  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
  // all of a batch's row loads first, then the stores: one x row at a time was one dependent HBM round trip per row
  // (the next row's load could not move above the previous row's store: they may alias), 1.3-7 us at bs256
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
    uint4* d[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j], cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
          {  // KPERM3: k8 group G = tid & 31 of stage tid >> 5 holds the bf16 pairs at stage offsets 2 B(G) + 64 t,
             // B(G) = 8 ((G >> 1) & 3) + 2 (G >> 3) + (G & 1)
            const int G = tid & 31;
            const unsigned* xs32 = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (tid >> 5) * 256)
                                   + (8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1));
            v[k] = make_uint4(__ldg(xs32), __ldg(xs32 + 32), __ldg(xs32 + 64), __ldg(xs32 + 96));
          }
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  asm volatile("fence.proxy.async.global;\n" ::: "memory");
  CTA_SYNC();
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
#if !OKC_B4FIN
#define OKC_YBF 0   // fork: preserve FP32 gamma product after BF16 GEMM2 result
#else
#define OKC_YBF 0
#endif
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = fork_bf16(d4[h] * gsc), up = fork_bf16(d4[2 + h] * gsc);
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
  } else {
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0))), w1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0 + 8)));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
#if OKC_YBF
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0), __float2bfloat16(fork_bf16(d4[h] * gsc) * w0));
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0 + 8), __float2bfloat16(fork_bf16(d4[2 + h] * gsc) * w1));
#else
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), fork_bf16(d4[h] * gsc) * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), fork_bf16(d4[2 + h] * gsc) * w1);
#endif
      }
    }
#if OKC_B4FIN
    {  // bet4_claude: this tile's 16-column sum of squares per token, reduced over the 8 column lanes (fixed tree)
      float q0 = 0.f, q1 = 0.f;
      if (tok0 < cnt) { const float a = fork_bf16(d4[0] * gsc) * w0, b = fork_bf16(d4[2] * gsc) * w1; q0 = a * a + b * b; }
      if (tok0 + 1 < cnt) { const float a = fork_bf16(d4[1] * gsc) * w0, b = fork_bf16(d4[3] * gsc) * w1; q1 = a * a + b * b; }
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        q0 += __shfl_xor_sync(0xffffffffu, q0, o);
        q1 += __shfl_xor_sync(0xffffffffu, q1, o);
      }
      if (g == 0) {
        if (tok0 < cnt) p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0;
        if (tok0 + 1 < cnt) p.sspart[(size_t)(jst + tok0 + 1) * 256 + r16] = q1;
      }
    }
#endif
  }
}
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  if constexpr (!PB) { asm volatile("fence.proxy.async.global;\n" ::: "memory"); }
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
#if OKC_MFIN && (OKC_MFP & 1)
  if (pd.kind == 4) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + ((OKC_MFP & 8) ? pd.slot : (int)(mf_gen(S) * 32 + S.jE[j]))), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
#endif
  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  // two n-tiles' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load
  // waits behind the previous tile's stores (they may alias), ntt serialized L2 round trips under full HBM load
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
#if OKC_MFIN && (OKC_MFP & 2)
  if (PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1) : "memory");
  }
#endif
  if (!PB) {
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = (OKC_SC6W ? 4u : 16u) + p.zero;  // opaque 16 (4 with SC6W): the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)
  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas
  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot's offset (the address field cannot
  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    // codes and scales one 4-step group at a time (fewer live registers)
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    // KPERM3: K is permuted inside the stage so that step j of every load group is scale block 4 t + j for this lane:
    // the 4 scale pairs ([j][h] bytes) come from one 8-byte load per stage and are built once per stage
#if OKC_SC6W
    // SC6: 768 B of scales per piece: plane W (uint32 per thread) | plane H (uint16 per thread); the 4 scale pairs of the stage
    // (bytes of 4 v, see harness/make_sc6.py) are built here once and stay in registers for the 4 load groups
    uint32_t sgs[4];
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2));
      const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;
      sgs[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      sgs[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;
    }
#else
    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
#endif
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
#if !OKC_SC6W
      const uint2 scg = scs;
#endif
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
      } else {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;  // commit group index in the stage
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;      // k16 step within the 4-step load group
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
#if OKC_SC6W
          const uint32_t sgp = sgs[js];
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }
        wg_commit();
        wg_wait<1>();
        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    // All reads of this activation slot are complete before the ring can reuse it.
    wg_wait<0>();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    mbar_arrive1(S.bars + BX_E + xs);
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
#if OKC_MFIN && (OKC_MFP & 1)
  else pd = WgsPend{4, u, (OKC_MFP & 8) ? (int)(mf_gen(S) * 32 + S.jE[j]) : -1, 0};  // phase B: the expert counter, same deferral
#endif
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    // a unit split between CTAs: its fixup slot is the CTA holding its first stage
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
    switch (ntt) {
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
    }
    gs += s1 - s0;
  }
  wgs_flush<PB>(p, S, pd);
}
// cost-weighted stage ranges: per-stage cost by the job's n-tiles, a prefix per job and phase (S.wcp, S.wcpB)
// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run
// from a cold instruction cache at the start of every launch
#define OKC_WGSU 160   // cepi: per-unit cost, phase A (zhebrak SK_CEPI)
#define OKC_WGSUB 140  // cepi: per-unit cost, phase B
__device__ __forceinline__ int wgs_unitc(int pb) { return pb ? OKC_WGSUB : OKC_WGSU; }
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;  // cepi: 32 units x (16 stages + unit cost)
    accB += ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
// stage at cost position x (phase pb), the stage whose cost interval holds x
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  if (nj == 0) return 0;  // Empty routing: no initialized job metadata exists.
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = (uint32_t)wgs_unitc(pb), cu = 16u * cs + U;
  const uint32_t r = x - (uint32_t)pre[lo], un = r / cu, rem = r - un * cu;
  const uint32_t st = rem < U ? 0u : min(15u, (rem - U) / cs);  // cepi: the unit cost sits before its stage 0
  return (lo << (9 - pb)) + (int)(un * 16u + st);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
// this CTA's stage ranges, warp-parallel (all 32 lanes of the x-producer warp; bs256-hot E457): the job-cost prefix as a
// shuffle scan over 32-job chunks, then the four stage_at searches on lanes 0..3 (lane = 2 pb + k). The serial one-lane version
// was the prologue's critical path (~1500 instructions before the weight stream can start)
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int lane = threadIdx.x & 31;
  const int nj = S.njF + S.njS;
  const int G = gridDim.x, c = blockIdx.x;
  uint32_t carryA = 0, carryB = 0;
  for (int j0 = 0; j0 < nj; j0 += 32) {
    const int j = j0 + lane;
    uint32_t ia = 0, ib = 0;
    if (j < nj) {
      ia = ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;
      ib = ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;
    }
    uint32_t sa = ia, sb = ib;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t va = __shfl_up_sync(0xffffffffu, sa, d), vb = __shfl_up_sync(0xffffffffu, sb, d);
      if (lane >= d) { sa += va; sb += vb; }
    }
    if (j < nj) { S.wcp[j] = (int)(carryA + sa - ia); S.wcpB[j] = (int)(carryB + sb - ib); }
    carryA += __shfl_sync(0xffffffffu, sa, 31);
    carryB += __shfl_sync(0xffffffffu, sb, 31);
  }
  if (lane == 0) { S.wcp[nj] = (int)carryA; S.wcpB[nj] = (int)carryB; S.wtot = carryA; S.wtotB = carryB; }
  __syncwarp();
  const int l4 = lane & 3, pbl = l4 >> 1, kk = l4 & 1;
  const uint32_t tot = pbl ? carryB : carryA;
  int res;
  if (kk == 1 && c == G - 1) res = nj << (9 - pbl);
  else res = wgs_stage_at(S, nj, pbl, tot * (uint32_t)(c + kk) / (uint32_t)G);
  if (lane < 4) S.wgr[lane] = res;
  __syncwarp();
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t sj = (uint32_t)(s & ((1 << (9 - pb)) - 1)), cs = (uint32_t)wgs_cost(pb, S.jC[j]), U = (uint32_t)wgs_unitc(pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (sj >> 4) * (16u * cs + U) + U + ((sj & 15u) + 1u) * cs;  // cepi
  // largest c with floor(tot * c / G) < xe
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);
  if (c > G - 1) c = G - 1;
  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;
  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;
  return c;
}
#endif

template <bool PB, bool BIG>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF, njS = S.njS;
  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs, nF = njF * ubs;
  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost
#if OKC_TINY
  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A's last ring
  // refills already fetch those units (same split function as phase B uses)
  int cB0 = 0, cBn = 0;
  bool cpf = false;
  if (!PB) {
    const int nFB = njF * UB_B;
    if (c >= nFB) {
      int b1;
      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);
      cBn = b1 - cB0;
      cpf = OKC_CPFF || njF == 0;
    }
  }
#if OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
#else
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);
#endif
}

// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = threadIdx.x;
  DBG(0);
  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes
  // scratch, so wait for it here; without the launch attribute this returns at once
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif
#if FAST1_ON
  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA's first units follow
  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
    const int n1 = min(OKC_FAST1, v11 - v01);
    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];
#pragma unroll
    for (int i = 0; i < OKC_FAST1; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the
    // barriers while they are in flight (the inits used to wait behind the cold loads)
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#endif
    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if OKC_SPLIT
#pragma unroll
      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),
    // partial buffer free (the epilogue warpgroup)
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
#if OKC_MFIN && (OKC_MFQ & 8)
    // launch parity: the exit counter wraps at 2 G and every CTA bumps it once at its start, so it reads G * n (mod 2 G) plus
    // the CTAs of this launch that got here first (< G): >= G exactly in odd launches
    const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
#endif
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
  route(p, S);
#if OKC_WGS && OKC_WGSPR
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges
#endif
  DBG(1);
#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)
  {  // the other parity's counters and hand-off flags belong to the next launch (the previous launch is over): zero them
    const int o = 1 - S.gen;
    for (int i = blockIdx.x * NT + tid; i < 32 + MF_XRS + JMAX * 16; i += gridDim.x * NT) {
      if (i < 32) p.mfl[MF_CE + o * 32 + i] = 0;
      else if (i < 32 + MF_XRS) p.mfl[MF_XR + o * MF_XRS + (i - 32)] = 0;
      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
  }
#else
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#endif
#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {  // padded-row base of every job in xA / xB (warp scan)
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages
  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0;
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
  }
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
#if OKC_WS
  WS_JOIN();  // the helper's last epilogue stores are visible
#endif
  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still
  // waits for this whole grid to finish and its writes to be visible)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // grid barrier on this launch's counter (two, by generation: the other one was zeroed by this launch for the next);
  // the last arriver bumps the generation, which no CTA of this launch reads again
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the barrier
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
#if !(OKC_MFIN && OKC_WGS && (OKC_MFQ & 4))
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_B4FIN
  // bet4_claude: item = (token, 1024-column segment), thread = 2 columns: 4 x M items over all CTAs. Each route's sum
  // of squares = fixed-order sum of its 256 epilogue partials (warp w: route w / 2, half w & 1; xor tree; 2 halves)
  {
    static_assert(NT == 512 && NW == 16, "B4FIN: 16 warps = 8 routes x 2 halves, 512 threads x 2 columns");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    constexpr int SEG = 1024, NSEG = HD / SEG;
    for (int it = blockIdx.x; it < p.M * NSEG; it += gridDim.x) {
      const int tk = it / NSEG, col = (it % NSEG) * SEG + tid * 2;
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {  // the token's local experts are complete when all their jobs' units were counted by all 4 warpgroups
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
#endif
#if B4_EARLY_Y
      float2 yv[8];  // A2: the 8 routes' y loads go out before the ss partial loads and the reduction
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        yv[sl] = q >= 0 ? __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col)) : make_float2(0.f, 0.f);
      }
#endif
      {
        const int sl = warp >> 1, hh = warp & 1;
        const int q = S.pos[tk * 8 + sl];
        float v = 0.f;
        if (q >= 0) {
          const float4 a = __ldcg(reinterpret_cast<const float4*>(p.sspart + (size_t)q * 256 + hh * 128 + lane * 4));
          v = (a.x + a.y) + (a.z + a.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp] = v;
      }
      CTA_SYNC();
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        if (q >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
#if B4_EARLY_Y
          const float2 v = yv[sl];
#else
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col));
#endif
          acc.x += r * v.x; acc.y += r * v.y;
        }
      }
      *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);
      CTA_SYNC();  // red is reused by the next item
    }
  }
#elif OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
  // shuffles, then the 16 warps' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {  // the token's local experts are complete when all their jobs' units were counted by all 4 warpgroups
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
      if (tk == blockIdx.x) DBG(4);
#endif
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#else
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#endif
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {  // uniform over the CTA (one token)
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#else
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#endif
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();  // red is reused by the next token
    }
  }
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float4 v = __ldcg(src + i);
          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;
        }
      }
    }
    uint4 o[2];
    uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      ow[i] = *reinterpret_cast<uint32_t*>(&v);
    }
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
    dst[0] = o[0]; dst[1] = o[1];
  }
#endif
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}

}  // namespace okc

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if OKC_WGS
  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;
#endif
  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  return base + (xa > xb ? xa : xb);
}

static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }

std::vector<int64_t> okc_attrs() {
  cudaFuncAttributes a0, a1;
  cudaFuncGetAttributes(&a0, okc::moe_fused<false>);
  cudaFuncGetAttributes(&a1, okc::moe_fused<true>);
  return {a0.numRegs, (int64_t)a0.localSizeBytes, (int64_t)a0.sharedSizeBytes, a1.numRegs, (int64_t)a1.localSizeBytes, (int64_t)a1.sharedSizeBytes};
}

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)
#if OKC_WGS
  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");
  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,
              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");
  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
#if OKC_WGS && OKC_RLOAD
  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");
#endif
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)scratch[1].data_ptr();
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
#if OKC_WGS
  p.mfl = ib + 8 + 1024 + okc::EMAX;
#else
  p.mfl = nullptr;
#endif
  p.ybuf = p.ss + okc::EMAX;
#if OKC_WGS && OKC_B4FIN
  // B4FIN runs only at M <= 32: ybuf rows (sorted routes) stay below M * 8 <= 256, so its last 64 rows (1024 routes x 256 partials) are free
  TORCH_CHECK(x.size(0) <= 32 && x.size(0) * 8 <= okc::EMAX - 64, "B4FIN build is for M <= 32");
  p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD;
#else
  p.sspart = nullptr;
#endif
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
#if OKC_MFIN
  p.xready = p.mfl + okc::MF_XR;
#endif
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#if OKC_MFIN
  p.doneAs = p.mfl + okc::MF_DA;
#endif
  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS), "MFIN needs the flag region in scratch[0]");
#else
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  static bool attr = false;
  size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;
    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();

  // Ordinary same-stream cooperative launch: prior PDL producers finish first.
  // No native launch may depend on an oversubscribed software grid barrier.
  int device=0; cudaDeviceProp prop{}; int resident=0;
  TORCH_CHECK(cudaGetDevice(&device)==cudaSuccess && cudaGetDeviceProperties(&prop,device)==cudaSuccess,
              "FORK_MOE_RESOURCE device properties");
  TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch,
              "FORK_MOE_RESOURCE requires SM90 cooperative launch; use owned/Marlin adapter");
  TORCH_CHECK(grid>0 && grid<=prop.multiProcessorCount && grid<=136 && grid<160,
              "FORK_MOE_RESOURCE grid/fixup bound; use owned/Marlin adapter");
  TORCH_CHECK(p.M>0 && p.M<=okc::MMAX && p.nloc==32 && p.lo==0,
              "FORK_MOE_RESOURCE shape/local-ID ABI");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && out.scalar_type()==torch::kBFloat16
              && ids.scalar_type()==torch::kInt32 && tw.scalar_type()==torch::kFloat32,
              "FORK_MOE_RESOURCE activation/routing dtype ABI");
  auto kernel = p.M>=OKC_BIGM ? okc::moe_fused<true> : okc::moe_fused<false>;
  TORCH_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)==cudaSuccess,
              "FORK_MOE_RESOURCE dynamic shared-memory attribute");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,okc::NTL,sm)==cudaSuccess
              && resident>=1 && grid<=resident*prop.multiProcessorCount,
              "FORK_MOE_RESOURCE cooperative residency");
  void* args[]={&p};
  const cudaError_t launch=cudaLaunchCooperativeKernel((const void*)kernel,dim3((unsigned)grid),
      dim3(okc::NTL),args,sm,st);
  TORCH_CHECK(launch==cudaSuccess,"FORK_MOE_LAUNCH ",cudaGetErrorString(launch));

}


'''

_MA_EXTS = {}
_MA_ALT_SRCS = {'moe_helper0': r'''
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
#endif
#define OKC_WGMIN 9
#define OKC_XPRE 1
#define FAST1_ARCH 1
#ifndef OKC_FAST1
#define OKC_FAST1 3
#endif
#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)
#ifndef OKC_CPFF
#define OKC_CPFF 1
#endif
#ifndef OKC_FCOSTBB
#define OKC_FCOSTBB OKC_FCOSTB
#endif
#define OKC_FCOSTSB OKC_FCOST
#define OKC_DYN 1
#ifndef OKC_DYNK
#define OKC_DYNK 4
#endif
#define OKC_TDYN 0
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#define OKC_FT_T 64
#ifndef OKC_SPLIT
#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)
#endif
#ifndef OKC_TINY
#define OKC_TINY 0
#endif
#ifndef OKC_NW
#define OKC_NW 8
#endif
#ifndef OKC_EARLYN
#define OKC_EARLYN 1
#endif
#define OKC_BIGM 64
#define OKC_XPF 0
#ifndef OKC_CPF
#define OKC_CPF 1
#endif
#define OKC_FATSTATIC 1
#define OKC_FCOST 40
#ifndef OKC_FCOSTB
#define OKC_FCOSTB 40
#endif
#ifndef OKC_WS
#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue
#endif
#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped
#ifndef OKC_WGS
#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp
#endif
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef OKC_UWARP
#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers
#endif
#if OKC_UWARP
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#else
#define WARP_IDX(t) ((t) >> 5)
#endif
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#ifndef OKC_FINPF
#define OKC_FINPF 0   // skinny builds: final item's topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)
#endif
#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#define OKC_NOLOAD 0

namespace okc {

#if !OKC_WGS && !OKC_WS
#error FORK_MOE_UNPROVEN_FALLBACK_select_owned_or_Marlin_adapter
#endif
#if OKC_WGS
#undef OKC_MFIN
#define OKC_MFIN 0
#endif


#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
#else
#define L2H_SUFFIX ""
#define L2H_OPERAND ""
#define L2H_ARG
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
#endif

constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
#define TIDX ((int)((threadIdx.x + NT) % NTL))
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (OKC_DYNK per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr int NST = 3;          // fat x stages
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)
  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready;                           // WGS: per job, rows of xA written
  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
  uint32_t zero;  // always 0 (opaque to the compiler)
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA;      // [JMAX]
  __nv_bfloat16* inter;  // [EMAX, ID]
  float* ybuf;     // [EMAX, HD]
  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1;      // [EMAX]
  __nv_bfloat16* out;
  unsigned long long* dbg;
};

struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums
#else
  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free
  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
#if OKC_WGS
  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty
  int wglast[4];      // WGS: this warpgroup completed a split unit
  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job
  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job
  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)
  uint32_t wtot;      // WGS: phase-A total cost
  uint32_t wtotB;     // WGS: phase-B total cost
#else
  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)
#endif
#if OKC_WGS
  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue's job -> expert map)
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
};

// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).
// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,
// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe
  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}


// ---------------------------------------------------------------- routing prologue (every CTA, identical result)
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)
  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
#else
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = __ldg(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
#endif
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if OKC_SPLIT
    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }
#else
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
#endif
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
#if !OKC_WGS  // (the WGS build has no fat queue)
    // fat queue order: largest n-tile count first (LPT), stable by job index
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
  early_issue(p, S);
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}



// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float fork_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float probe_q(float v) {
  return v;
}
__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    if (tk < ntok) {
      float gt = fork_bf16(r[h] * gsc), up = fork_bf16(r[2 + h] * gsc);
      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));
    }
  }
}
// returns per-token partial sum of squares (2 values) for this lane's two tokens
__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,
                                      float* sq) {
  const float w0 = fork_bf16(1.f + __bfloat162float(p.normw[n0])), w1 = fork_bf16(1.f + __bfloat162float(p.normw[n0 + 8]));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    float v0 = fork_bf16(r[h] * gsc) * w0, v1 = fork_bf16(r[2 + h] * gsc) * w1;
    sq[h] = v0 * v0 + v1 * v1;
    if (tk < ntok) {
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);
    }
  }
}
__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);
  }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}

// ---------------------------------------------------------------- the persistent kernel


// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the
// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes
// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the
// ALU pipe, which the FP4 decode saturates
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);
}

template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB) row = p.inter + (size_t)(st + r) * ID;
      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

constexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4
  const int cnt = S.jC[j], st = S.jS[j];
  for (int i = lane; i < 8 * 64; i += 32) {
    const int r = i >> 6, c = i & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (8 + r < cnt) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;
      v = PB ? __ldcg(src) : __ldg(src);
    }
    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;
  }
  __syncwarp();
}

#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill
    if (!PB) {
      const int jj = njF + (v >> rsh);
      S.dync[jj] += 1;
      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);
    }
    return v;
  };
  auto refill = [&](int sl) {
    const int v = take();
    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }
    else if (cpf && preB < nB) {
      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);
      S.uslot[sl] = BFLAG | (v0B + preB);
      ++preB;
    } else {
      S.uslot[sl] = -1;
    }
  };
  if (tid == 0)
    for (int i = pre; i < OKC_NS; ++i) {
      const int sl = (seq0 + i) % OKC_NS;
      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) {
    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);
    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);
    nwc0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwc1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)));
  }
  int nproc = 0;
  for (int i = 0;; ++i) {
    int v;
    if (i < nU) {
      v = v0 + i;  // static units: no slot lookup
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B's prefetched units follow)
      v &= ~BFLAG;
    }
    const int j = njF + (v >> rsh), rb = v & (rbs - 1);
    float nwn0 = 0.f, nwn1 = 0.f;
    int vn = -1;
    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];
    if (PB && tid < 128 && vn >= 0) {
      const int rbq = (vn & ~BFLAG) & (rbs - 1);
      nwn0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwn1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = fork_bf16(s0 * gsc), up = fork_bf16(s1 * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = fork_bf16(s0 * gsc) * nwc0, v1f = fork_bf16(s1 * gsc) * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;
  }
  seq = seq0 + nproc;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      if (nU > 0) {
        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  for (int i = pre; i < OKC_NS; ++i) {
    const int sl = (seq0 + i) % OKC_NS;
    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }
  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwc1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8))); }
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    float nwn0 = 0.f, nwn1 = 0.f;
    if (PB && tid < 128 && i + 1 < nU) {
      const int rbq = (v + 1) % rbs;
      nwn0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))); nwn1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i + OKC_NS - nU < nB) {
      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);
      ++preB;
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = fork_bf16(s0 * gsc), up = fork_bf16(s1 * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = fork_bf16(s0 * gsc) * nwc0, v1f = fork_bf16(s1 * gsc) * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
  }
  seq = seq0 + nU;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#endif

// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens
// ---------------------------------------------------------------- fat unit through the smem ring
// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x
// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile
// for the stage is staged by cp.async into two buffers that live in slots 3..4.
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
  if (TIDX == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
}

// ---------------------------------------------------------------- fat units: one continuous stream per CTA
// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps
// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit's weights).
// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};

// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from
// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads
  // k block cc at byte offset 32 st
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired
  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact
      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2
      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)
      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);
      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;
      dq4((&w0.x)[st], r0, r1, r2, r3);
      dq4((&w1.x)[st], q0, q1, q2, q3);
      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);
      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);
      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);
      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);
#undef FMG0
#undef FMG80
#undef FMG1
#undef FMG81
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);
      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);
      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);
    }
    wg_commit();
    if (cc >= 1) {
      wg_wait<1>();  // the previous chunk's group retired: its register set may be rewritten
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f
}

// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)
template <int NTT>
__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);
#if OKC_NW == 8
  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);
#endif
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
#if OKC_NW == 8
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
#endif
#if OKC_PRMTSC
    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);
#if OKC_NW == 8
    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);
#endif
#else
    const int sh = (cc & 1) * 16;
    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif
    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;
#pragma unroll
    for (int sp = 0; sp < 2; ++sp) {
      uint32_t a0[2][4], a1[2][4];
      (void)a1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s2 = 2 * sp + h;
        uint32_t r0, r1, r2, r3;
        dq4((&w0.x)[s2], r0, r1, r2, r3);
        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);
#if OKC_NW == 8
        dq4((&w1.x)[s2], r0, r1, r2, r3);
        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);
        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);
#endif
        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);
#endif
      }
    }
  }
}

template <bool PB, bool BIG>
__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;  // 16
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete
    if (PB) {
      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }
      CTA_SYNC();
    }
  };
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
  auto stage_x = [&](int j, int ks, int xsl) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int ntl = (cnt + 7) >> 3;
    const int nrow = ntl * 8;
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
    const int per = nrow * (KS / 8);
    if (is_nat(j)) {
      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)
      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);
      for (int i = tid; i < per; i += NT) {
        const int r = i >> 5, ch = i & 31;
        const bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;
        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);
      }
    } else {
      for (int i = tid; i < per; i += NT) {
        int r = i / (KS / 8), c = i - r * (KS / 8);
        bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;
        cp_async16(xs + r * XP_FT + c * 8, src, v);
      }
    }
    cp_commit();
  };
  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)
  const __nv_bfloat16* xsrc[8];
  uint32_t xdst[8];
  uint32_t xval = 0;
  int xn = 0;
  auto prep_x = [&](int j) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);
    xval = 0;
    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = tid + k * NT;
      const int r = i >> 5, c = i & 31;
      const bool v = i < per && r < cnt;
      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
      xdst[k] = r * XP_FT + c * 8;
      if (v) xval |= 1u << k;
    }
  };
  auto stage_xp = [&](int ks, int xsl) {
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);
    cp_commit();
  };
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int f = S.fseq;  // ring position of the current unit's k-stage 0
  int xq = 0;      // x stage parity counter
  int j, ub;
  job_of(u, j, ub);
#pragma unroll
  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));
  wait_A(j);
  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);
  while (true) {
    const int un = u + gridDim.x;
    int jn = 0, ubn = 0;
    if (un < nF) job_of(un, jn, ubn);
    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];
    const int ntile = (cnt + 7) >> 3;
    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)
    const bool nat = is_nat(j);
    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;
    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];
#pragma unroll
    for (int i = 0; i < FT_T / 8; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }
    for (int ks = 0; ks < NKS; ++ks) {
      if (OKC_XPRE) {
        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }
        else cp_commit();
      } else {
        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }
        else cp_commit();
      }
      cp_wait<1>();
      const int q = f + ks, sl = q % NSF;
      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // cp.async x writes -> wgmma reads
      CTA_SYNC();  // x stage visible to every warp
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);
      if (nat) {
        switch (ntc) {
          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
        }
      } else {
        switch (ntc) {
          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;
          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;
        }
      }
      CTA_SYNC();  // slot sl and the x stage consumed
      ++xq;
      // refill: the stage NSF ahead, in this unit or the next one
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
    // epilogue of unit u (registers -> global)
    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;
    if (!PB) {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);
#if OKC_NW == 8
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);
#endif
        }
      __threadfence();
      CTA_SYNC();
      if (tid == 0) atomicAdd(p.doneA + j, 1);
    } else {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          float sq0[2], sq1[2];
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);
#if OKC_NW == 8
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);
          sq0[0] += sq1[0]; sq0[1] += sq1[1];
#else
          (void)sq1;
#endif
          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);
        }
    }
    f += NKS;
    if (un >= nF) break;
    u = un; j = jn; ub = ubn;
  }
  cp_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}

template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}



// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring
#endif
  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region
    if (TIDX == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in
  // skinny_ring issues the rest after routing, so every CTA's first unit is served ahead of the second ones
  const int n = min(OKC_EARLYN, v1 - v0);
  if (TIDX == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}

// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost
#if OKC_WS
// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)
// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper
// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit's
// epilogue, so no CTA-wide barrier sits in the per-unit loop.
// the last WSBLK units keep their partial buffers until the compute warps' own epilogues: WSBLK > WSNR deadlocks the
// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
  }
  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA's last units: epilogues here, not behind the helper's queue
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = fork_bf16(1.f + __uint_as_float(lnw0 << 16)), nw1 = fork_bf16(1.f + __uint_as_float(lnw1 << 16));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = fork_bf16((h ? sm.y : sm.x) * gsc) * nw0, v1f = fork_bf16((h ? sm.w : sm.z) * gsc) * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = fork_bf16(1.f + __bfloat162float(p.normw[n0])), nw1 = fork_bf16(1.f + __bfloat162float(p.normw[n0 + 8]));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = fork_bf16((h ? sm.y : sm.x) * gsc) * nw0, v1f = fork_bf16((h ? sm.w : sm.z) * gsc) * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B's skinny units are slower, so its ratio is lower)
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c); v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);
  }
}

#if OKC_WGS
// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)
// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,
// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales
// [grp][warp][g][h][j]; the stage's x slice (the job's token rows, rounded up to 8, x 256 K as wgmma B core matrices
// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16
// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).
// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a
// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.
#define OKC_WGSNX 3                             // x ring slots
constexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = 9216;                 // one warpgroup's weights in a stage
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;                  // fixup slots per phase (>= grid + 1)
constexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at
// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {OKC_WGSA};
constexpr int WGS_CB[8] = {OKC_WGSB};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
}
#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)
#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time
#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather
#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B
#define OKC_WGSFCI 3
#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits
// Phase A runs a CTA's trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit's
// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase
// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the
// order (its first stages wait for phase A's units in order).
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((TIDX & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
// x producer: one bulk copy per stage (lane 0): the job's x slice for the stage is contiguous in xA (phase A, written
// by every CTA's gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B
constexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the
  // stage ranges while the consumers gather phase A's x
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  if ((TIDX & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((TIDX & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = TIDX, nj = S.njF + S.njS;
  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
  // all of a batch's row loads first, then the stores: one x row at a time was one dependent HBM round trip per row
  // (the next row's load could not move above the previous row's store: they may alias), 1.3-7 us at bs256
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
    uint4* d[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j], cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = fork_bf16(d4[h] * gsc), up = fork_bf16(d4[2 + h] * gsc);
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
  } else {
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0))), w1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0 + 8)));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), fork_bf16(d4[h] * gsc) * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), fork_bf16(d4[2 + h] * gsc) * w1);
      }
    }
  }
}
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = TIDX, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  // two n-tiles' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load
  // waits behind the previous tile's stores (they may alias), ntt serialized L2 round trips under full HBM load
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = TIDX, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)
  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas
  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  int prev = -1;  // x slot the last committed group may still read
  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot's offset (the address field cannot
  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    // codes and scales one 4-step group at a time (fewer live registers)
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      const uint2 scg = scn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;  // commit group index in the stage
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;      // k16 step within the 4-step load group
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }
        wg_commit();
        wg_wait<1>();
        if (ci == 0) {
          if (prev >= 0) {  // the previous stage's last group just retired: its x slot is free
            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    prev = xs;
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    // a unit split between CTAs: its fixup slot is the CTA holding its first stage
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
    switch (ntt) {
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
    }
    gs += s1 - s0;
  }
  wgs_flush<PB>(p, S, pd);
}
// cost-weighted stage ranges: per-stage cost by the job's n-tiles, a prefix per job and phase (S.wcp, S.wcpB)
// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run
// from a cold instruction cache at the start of every launch
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
// stage at cost position x (phase pb), the stage whose cost interval holds x
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
  // largest c with floor(tot * c / G) < xe
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);
  if (c > G - 1) c = G - 1;
  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;
  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;
  return c;
}
#endif

template <bool PB, bool BIG>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF, njS = S.njS;
  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs, nF = njF * ubs;
  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost
#if OKC_TINY
  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A's last ring
  // refills already fetch those units (same split function as phase B uses)
  int cB0 = 0, cBn = 0;
  bool cpf = false;
  if (!PB) {
    const int nFB = njF * UB_B;
    if (c >= nFB) {
      int b1;
      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);
      cBn = b1 - cB0;
      cpf = OKC_CPFF || njF == 0;
    }
  }
#if OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
#else
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);
#endif
}

// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = TIDX;
  DBG(0);
  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes
  // scratch, so wait for it here; without the launch attribute this returns at once
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif
#if FAST1_ON
  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA's first units follow
  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
    const int n1 = min(OKC_FAST1, v11 - v01);
    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];
#pragma unroll
    for (int i = 0; i < OKC_FAST1; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the
    // barriers while they are in flight (the inits used to wait behind the cold loads)
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#endif
    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if OKC_SPLIT
#pragma unroll
      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),
    // partial buffer free (the epilogue warpgroup)
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
  route(p, S);
#if OKC_WGS && OKC_WGSPR
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges
#endif
  DBG(1);
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {  // padded-row base of every job in xA / xB (warp scan)
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages
  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0;
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
  }
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
#if OKC_WS
  WS_JOIN();  // the helper's last epilogue stores are visible
#endif
  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still
  // waits for this whole grid to finish and its writes to be visible)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // grid barrier on this launch's counter (two, by generation: the other one was zeroed by this launch for the next);
  // the last arriver bumps the generation, which no CTA of this launch reads again
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the barrier
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
  // shuffles, then the 16 warps' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {  // uniform over the CTA (one token)
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();  // red is reused by the next token
    }
  }
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float4 v = __ldcg(src + i);
          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;
        }
      }
    }
    uint4 o[2];
    uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      ow[i] = *reinterpret_cast<uint32_t*>(&v);
    }
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
    dst[0] = o[0]; dst[1] = o[1];
  }
#endif
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}

}  // namespace okc

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if OKC_WGS
  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;
#endif
  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  return base + (xa > xb ? xa : xb);
}

static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)
#if OKC_WGS
  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");
  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,
              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");
  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
#if OKC_WGS && OKC_RLOAD
  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");
#endif
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)scratch[1].data_ptr();
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#else
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  static bool attr = false;
  size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;
    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();

  // Ordinary same-stream cooperative launch: prior PDL producers finish first.
  // No native launch may depend on an oversubscribed software grid barrier.
  int device=0; cudaDeviceProp prop{}; int resident=0;
  TORCH_CHECK(cudaGetDevice(&device)==cudaSuccess && cudaGetDeviceProperties(&prop,device)==cudaSuccess,
              "FORK_MOE_RESOURCE device properties");
  TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch,
              "FORK_MOE_RESOURCE requires SM90 cooperative launch; use owned/Marlin adapter");
  TORCH_CHECK(grid>0 && grid<=prop.multiProcessorCount && grid<=136 && grid<160,
              "FORK_MOE_RESOURCE grid/fixup bound; use owned/Marlin adapter");
  TORCH_CHECK(p.M>0 && p.M<=okc::MMAX && p.nloc==32 && p.lo==0,
              "FORK_MOE_RESOURCE shape/local-ID ABI");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && out.scalar_type()==torch::kBFloat16
              && ids.scalar_type()==torch::kInt32 && tw.scalar_type()==torch::kFloat32,
              "FORK_MOE_RESOURCE activation/routing dtype ABI");
  auto kernel = p.M>=OKC_BIGM ? okc::moe_fused<true> : okc::moe_fused<false>;
  TORCH_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)==cudaSuccess,
              "FORK_MOE_RESOURCE dynamic shared-memory attribute");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,okc::NTL,sm)==cudaSuccess
              && resident>=1 && grid<=resident*prop.multiProcessorCount,
              "FORK_MOE_RESOURCE cooperative residency");
  void* args[]={&p};
  const cudaError_t launch=cudaLaunchCooperativeKernel((const void*)kernel,dim3((unsigned)grid),
      dim3(okc::NTL),args,sm,st);
  TORCH_CHECK(launch==cudaSuccess,"FORK_MOE_LAUNCH ",cudaGetErrorString(launch));

}


'''}

_MA_ALT_SRCS['moe_helper0df'] = _MA_ALT_SRCS['moe_helper0']

def _stored_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if DQ7:
                    d = (0, 28, 22, 18 if b == 2 else 26)[R]
                elif R == 0:
                    d = 0
                elif R == 1:
                    d = 6
                elif R == 2:
                    d = 10
                else:
                    d = 4 if b == 0 else 28   # m via rotr 4, (e0, e1, s) via rotl 4 == rotr 28
                pos[(R, up, b)] = (o + d) % 32
    assert sorted(pos.values()) == list(range(32))
    return pos

def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)

def _pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]."""
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  # rb, h, g, w, c, t
    s = s.permute(0, 3, 2, 5, 4, 1)  # rb, w, g, t, c, h
    return s.contiguous().view(-1)

def _sc6_ok(*sfs):
    """SC6: every scale byte is in 0x40..0x7F (bits 7:6 == 01), so byte - 0x40 is a lossless 6-bit value."""
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x40) | (b > 0x7F)).any()):
            return False
    return True

def _sc6_wgs(s):
    """SC6 WGS piece scales: [..., 1024] bytes (thread t = 8 bytes s0..s7 = [j][h]) -> [..., 768]: plane W (t * 4 B: bytes 4 v_n | piece_n,
    n = 0..3) then plane H (t * 2 B: bytes 4 v_(4+n) | piece_(4+n)); v6 = pieces 0..2, v7 = pieces 3..5 (see harness/make_sc6.py)."""
    lead = s.shape[:-1]
    v = s.view(*lead, 128, 8).to(torch.int32) - 0x40
    assert int(v.min()) >= 0 and int(v.max()) <= 63
    v6, v7 = v[..., 6], v[..., 7]
    piece = [(v6 >> 0) & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
    wb = torch.stack([(v[..., n] << 2) | piece[n] for n in range(4)], -1).to(torch.uint8)
    hb = torch.stack([(v[..., 4 + n] << 2) | piece[4 + n] for n in range(2)], -1).to(torch.uint8)
    return torch.cat([wb.reshape(*lead, 512), hb.reshape(*lead, 256)], -1).contiguous()

def _fat_order(codes, scales):
    """Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]]."""
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)

def _unit_order(codes, scales):
    """Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit)."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)

def _pack_codes_nat(codes, rows):
    """Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)

def _kperm_src():
    """KPERM3: stage position p -> stage column k. p = 16 s + c (k16 step s = 4 lg + j, fragment column c of lane
    t = (c >> 1) & 3, i = (c & 1) | ((c >> 3) << 1)); k = 16 (4 t + j) + 4 lg + i, so lane t's columns of step j in all
    four load groups are exactly scale block 4 t + j."""
    p = torch.arange(256)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    k = 16 * (4 * t + j) + 4 * lg + i
    assert sorted(k.tolist()) == list(range(256))
    return k

def _kperm_codes(codes):
    """codes uint8 [R, K/2] (low nibble first) -> the same codes with K permuted inside every 256-K stage."""
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
    nib = nib.index_select(2, _kperm_src().to(codes.device)).reshape(R, K // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()

def _kperm_rows(rowsA):
    """Gate/up row order whose 16-row block r16, lane g holds inter column colmap(P), P = 8 r16 + g, where
    colmap(P) = 256 (P >> 8) + _kperm_src()[P & 255] is the column phase B reads at K position P (the phase-A
    epilogue then stores at its usual position)."""
    r = rowsA.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rowsA.device)
    cm = (P >> 8) * 256 + _kperm_src().to(rowsA.device)[P & 255]
    return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)

def _wgs_copy(codes, sf, rows, sc6=False):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, t, j (block 16 st + 4 t + j)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, warp, g, t, j, h (KPERM3)
    if sc6:
        s = _sc6_wgs(s)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)

def _wgs_on():
    return any("-DOKC_WGS=1" in tuple(d) for d in (MOE_DEFINES, MOE_TINY_DEFINES, MOE_TINY2_DEFINES, MOE_MID_DEFINES))

def _ma_scales_prmt_exact(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True

def _main_defines(sfx):
    if not sfx and "-DOKC_WGS=1" in tuple(MOE_DEFINES):
        return MAIN_FALLBACK_DEFINES
    return MOE_DEFINES

def _ma_ext(defines=None):
    defines = tuple(MOE_DEFINES if defines is None else defines)
    if DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in _MA_EXTS:
        src = MOE_CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        for d in defines:
            if d.startswith("@"):
                src = _MA_ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        import hashlib as _native_build_hash
        source_tag = _native_build_hash.md5((MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS))).encode()).hexdigest()[:16]
        _MA_EXTS[defines] = load_inline(
            name=f"glue_moe_{source_tag}_sm{arch}",
            cpp_sources=MOE_CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={_MA_MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        _MA_EXTS[defines].okc_set_pdl(1 if _MA_PDL else 0)
    return _MA_EXTS[defines]

def _df_inter(inter):
    """DF: inter rows [128, 256) start (and, after every MID launch, end) as bf16 0xFFFF = not yet written."""
    inter[128 * 4096:256 * 4096].view(torch.int16).fill_(-1)
    return inter

def _ma_make_scratch(dev):
    return [torch.zeros(8 + 1024 + _MA_EMAX + 4096, dtype=torch.int32, device=dev),  # counters + doneA + ss copy 1 + MFIN flags
            torch.zeros(_MA_EMAX + _MA_EMAX * 4096 + WGS_FIXF, dtype=torch.float32, device=dev),
            _df_inter(torch.zeros((_MA_EMAX + 2 * WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev))]

def _fork_native_ma_prepare(weights, st):
    dev = weights["w13"].device
    I = H = 4096
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    wA, sA, wB, sB = [], [], [], []
    for e in range(32):
        wA.append(_pack_codes(weights["w13"][e].view(torch.uint8), rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e].view(torch.uint8), rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    st["ma_w"] = [torch.cat(wA), torch.cat(sA), weights["w13_gs"].float().contiguous(),
                  torch.cat(wB), torch.cat(sB), weights["w2_gs"].float().contiguous(),
                  lut.contiguous(), weights["post_expert_norm"].contiguous(),  # bf16; kernel forms 1 + w
                  torch.cat([_fat_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_fat_order(x, y) for x, y in zip(wB, sB)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wB, sB)])]
    if "_wgs_on" in globals() and _wgs_on():   # their prepare(): w[12], w[13] = the WGS copies (per 256-row unit)
        st["ma_w"].append(torch.cat([(_wgs_copy_pairlut if st.get('_expert_pair_arm',False) and st['ma_sc6'] else _wgs_copy)(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], _kperm_rows(rowsA), st["ma_sc6"])
                                     for e in range(32)]))
        st["ma_w"].append(torch.cat([(_wgs_copy_pairlut if st.get('_expert_pair_arm',False) and st['ma_sc6'] else _wgs_copy)(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, st["ma_sc6"])
                                     for e in range(32)]))
    if st.get('_expert_pair_arm',False) and st['ma_sc6']:
        st['ma_w'][6]=torch.cat((st['ma_w'][6],torch.tensor(_PAIR_WORDS,dtype=torch.int32,device=dev)))
    st["ma_scratch"] = [_expert_moe_guarded(t, st) for t in _ma_make_scratch(dev)]

def _fork_native_moe(M=None, state=None):
    if state is None:
        return _moe_global_cache(M)
    sfx = tuple(state["ma_sfx"])
    if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        key, defines = "tiny", MOE_TINY_DEFINES
    elif M is not None and MOE_TINY2_DEFINES and M <= MOE_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES)):
        key, defines = "tiny2", MOE_TINY2_DEFINES
    elif M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
        key, defines = "mid", MOE_MID_DEFINES
    elif M is not None and 16 < M <= 32 and tuple(_main_defines(sfx)) == tuple(MOE_DEFINES):
        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)
    else:
        key, defines = "main", _main_defines(sfx)
    if state["ma_sc6"] and "-DOKC_WGS=1" in defines:
        defines = tuple(defines) + ("-DOKC_SC6W=1",)
    arm = state.get('_expert_pair_arm', False)
    if type(arm) is not bool or state.get('_expert_pair_arm_bound', arm) is not arm:
        raise RuntimeError('EXPERT_PAIR_ARM_MUTATED')
    if arm and key in ('b4', 'main'):
        defines = tuple(defines) + ('@moe_pairlut',)
    ext = state["ma_ext"].get(key)
    if ext is None:
        ext = _RouterExt(_ma_ext(tuple(defines) + sfx))
        if MOE_HAS_PDL:
            ext.okc_set_pdl(0)
        state["ma_ext"][key] = ext
    return ext

def _expert_pair_source():
    source = MOE_CUDA_SRC
    for old, new, count in _EXPERT_PAIR_EDITS:
        if source.count(old) != count:
            raise RuntimeError('EXPERT_PAIR_SOURCE_DRIFT')
        source = source.replace(old, new)
    return source

def _expert_pair_bind(state, arm):
    # Call only on a separately owned prepared state, before any captures.
    if type(arm) is not bool or '_expert_pair_arm_bound' in state:
        raise RuntimeError('EXPERT_PAIR_INVALID_REBIND')
    state['_expert_pair_arm'] = arm
    state['_expert_pair_arm_bound'] = arm
    for key in ('b4', 'main'):
        state['ma_ext'].pop(key, None)
    for tokens in (32, 128):
        _fork_native_moe(tokens, state)
    return state

class _RouterExt:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __deepcopy__(self, memo):
        return self

_POS = _stored_pos()

_MA_ALT_SRCS['moe_pairlut'] = _expert_pair_source()

"""Standalone adapter body; embedded after the exact pinned native dependency closure."""


def _moe_global_cache(tokens):
    raise RuntimeError('EXPERT_MOE_STATE_REQUIRED')


def _expert_moe_budget(deadline):
    if time.monotonic() >= deadline:
        raise RuntimeError('EXPERT_MOE_PREPARE_295S')


def _expert_moe_tensor(tensor, shape, dtype, device, name):
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype or tensor.device != device or not tensor.is_cuda
            or not tensor.is_contiguous() or tensor.data_ptr() % 16):
        raise RuntimeError('EXPERT_MOE_TENSOR_CONTRACT: ' + name)


def _expert_moe_weights(weights):
    specs = {
        'w13': ((256, 8192, 2048), torch.uint8),
        'w13_sf': ((256, 8192, 256), torch.float8_e4m3fn),
        'w13_gs': ((256,), torch.float32),
        'w2': ((256, 4096, 2048), torch.uint8),
        'w2_sf': ((256, 4096, 256), torch.float8_e4m3fn),
        'w2_gs': ((256,), torch.float32),
        'post_norm_w': ((4096,), torch.bfloat16),
    }
    if not isinstance(weights, dict) or not specs.keys() <= weights.keys():
        raise RuntimeError('EXPERT_MOE_WEIGHT_KEYS')
    if not isinstance(weights['w13'], torch.Tensor):
        raise RuntimeError('EXPERT_MOE_TENSOR_CONTRACT: w13')
    device = weights['w13'].device
    for name, (shape, dtype) in specs.items():
        _expert_moe_tensor(weights[name], shape, dtype, device, name)
    local = {name: weights[name][:32].clone() for name in specs if name != 'post_norm_w'}
    local['post_expert_norm'] = weights['post_norm_w'].clone()
    return local


def _expert_moe_guarded(tensor, state):
    nbytes = tensor.numel() * tensor.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL,
                      dtype=torch.int32, device=tensor.device)
    full.view(torch.uint8)[:nbytes].copy_(tensor.reshape(-1).view(torch.uint8))
    view = full.view(torch.uint8)[:nbytes].view(tensor.dtype).view(tensor.shape)
    state['guards'].append((full, words))
    return view


def _expert_moe_check_guards(state):
    torch.cuda.synchronize(state['device'])
    for full, words in state['guards']:
        if bool((full[words:] != SCRATCH_SENTINEL).any()):
            raise RuntimeError('EXPERT_MOE_SCRATCH_OVERRUN')


def _expert_moe_prepare(weights, arm, deadline=None):
    if type(arm) is not bool:
        raise RuntimeError('EXPERT_MOE_ARM_TYPE')
    if deadline is None:
        deadline = time.monotonic() + 295
    _expert_moe_budget(deadline)
    local = _expert_moe_weights(weights)
    device = local['w13'].device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('EXPERT_MOE_PREPARE_DURING_CAPTURE')
        properties = torch.cuda.get_device_properties(device)
        if ((properties.major, properties.minor) != (9, 0)
                or not 0 < properties.multi_processor_count <= 136):
            raise RuntimeError('EXPERT_MOE_NATIVE_DEVICE_CONTRACT')
        state = dict(device=device, nsm=properties.multi_processor_count,
                     ma_ext={}, guards=[], identities=EXPERT_MOE_IDENTITIES)
        state['ma_sfx'] = _MA_SFX_IF_PRMT if (
            _MA_PRMT_SCALES and _ma_scales_prmt_exact(local['w13_sf'], local['w2_sf'])) else ()
        if not state['ma_sfx']:
            raise RuntimeError('EXPERT_MOE_NATIVE_SCALE_CONTRACT_NO_FALLBACK')
        state['ma_sc6'] = bool(SC6_WGS and state['ma_sfx'] and _wgs_on()
                               and _sc6_ok(local['w13_sf'], local['w2_sf']))
        _expert_moe_budget(deadline)
        _expert_pair_bind(state, arm)
        _expert_moe_budget(deadline)
        for tokens in (1, 4, 8, 16, 32, 33, 128, 256, 257):
            _fork_native_moe(tokens, state)
            _expert_moe_budget(deadline)
        _fork_native_ma_prepare(local, state)
        torch.cuda.synchronize(device)
        _expert_moe_budget(deadline)
        if set(state['ma_ext']) != {'tiny', 'tiny2', 'mid', 'b4', 'main'}:
            raise RuntimeError('EXPERT_MOE_INCOMPLETE_BUILD')
        _expert_moe_check_guards(state)
        _expert_moe_budget(deadline)
        return state


def prepare(weights):
    return _expert_moe_prepare(weights, False)


def expert_moe_prepare_candidate(weights):
    return _expert_moe_prepare(weights, True)


def expert_moe_prepare_four(weights):
    deadline = time.monotonic() + 295
    return tuple(_expert_moe_prepare(weights, arm, deadline)
                 for arm in (False, False, True, True))


def _expert_moe_run(data, state, arm):
    if (state.get('identities') != EXPERT_MOE_IDENTITIES
            or state.get('_expert_pair_arm') is not arm
            or state.get('_expert_pair_arm_bound') is not arm):
        raise RuntimeError('EXPERT_MOE_STATE_IDENTITY_OR_ARM')
    if not isinstance(data, dict) or not {'config', 'hidden_states', 'topk_ids', 'topk_weights'} <= data.keys():
        raise RuntimeError('EXPERT_MOE_DATA_KEYS')
    config = data['config']
    expected = dict(d_hidden=4096, d_expert=4096, n_experts=256,
                    local_start=0, local_count=32, top_k=8)
    if not isinstance(config, dict) or any(type(config.get(key)) is not int or config[key] != value
                                         for key, value in expected.items()):
        raise RuntimeError('EXPERT_MOE_CONFIG_UNSUPPORTED_LOCAL_OR_DIMENSION')
    tokens = config.get('bs')
    if type(tokens) is not int or tokens not in (1, 4, 8, 16, 32, 33, 128, 256, 257):
        raise RuntimeError('EXPERT_MOE_TOKEN_CONTRACT')
    device = state['device']
    inputs = (('hidden_states', (tokens, 4096), torch.bfloat16),
              ('topk_ids', (tokens, 8), torch.int32),
              ('topk_weights', (tokens, 8), torch.float32))
    for name, shape, dtype in inputs:
        _expert_moe_tensor(data[name], shape, dtype, device, name)
    if torch.cuda.current_device() != device.index:
        raise RuntimeError('EXPERT_MOE_CURRENT_DEVICE')
    output = torch.empty_like(data['hidden_states'])
    _fork_native_moe(tokens, state).okc_run(
        state['ma_w'], state['ma_scratch'], data['hidden_states'], data['topk_ids'],
        data['topk_weights'], output, 0, 32, state['nsm'])
    return output


def kernel_fn(data, state):
    return _expert_moe_run(data, state, False)


def expert_moe_candidate_fn(data, state):
    return _expert_moe_run(data, state, True)

def _pack_codes_nat_pairlut(codes, rows):
    """Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            word |= code << (Rg*8+up*4)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)
def _wgs_copy_pairlut(codes, sf, rows, sc6=False):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = _pack_codes_nat_pairlut(_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, t, j (block 16 st + 4 t + j)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, warp, g, t, j, h (KPERM3)
    if sc6:
        s = _sc6_wgs(s)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)
_PAIR_WORDS = (0, 8320, 8448, 8512, 8576, 8640, 8704, 8768, 32768, 41088, 41216, 41280, 41344, 41408, 41472, 41536, 545259520, 545267840, 545267968, 545268032, 545268096, 545268160, 545268224, 545268288, 545292288, 545300608, 545300736, 545300800, 545300864, 545300928, 545300992, 545301056, 553648128, 553656448, 553656576, 553656640, 553656704, 553656768, 553656832, 553656896, 553680896, 553689216, 553689344, 553689408, 553689472, 553689536, 553689600, 553689664, 557842432, 557850752, 557850880, 557850944, 557851008, 557851072, 557851136, 557851200, 557875200, 557883520, 557883648, 557883712, 557883776, 557883840, 557883904, 557883968, 562036736, 562045056, 562045184, 562045248, 562045312, 562045376, 562045440, 562045504, 562069504, 562077824, 562077952, 562078016, 562078080, 562078144, 562078208, 562078272, 566231040, 566239360, 566239488, 566239552, 566239616, 566239680, 566239744, 566239808, 566263808, 566272128, 566272256, 566272320, 566272384, 566272448, 566272512, 566272576, 570425344, 570433664, 570433792, 570433856, 570433920, 570433984, 570434048, 570434112, 570458112, 570466432, 570466560, 570466624, 570466688, 570466752, 570466816, 570466880, 574619648, 574627968, 574628096, 574628160, 574628224, 574628288, 574628352, 574628416, 574652416, 574660736, 574660864, 574660928, 574660992, 574661056, 574661120, 574661184, -2147483648, -2147475328, -2147475200, -2147475136, -2147475072, -2147475008, -2147474944, -2147474880, -2147450880, -2147442560, -2147442432, -2147442368, -2147442304, -2147442240, -2147442176, -2147442112, -1602224128, -1602215808, -1602215680, -1602215616, -1602215552, -1602215488, -1602215424, -1602215360, -1602191360, -1602183040, -1602182912, -1602182848, -1602182784, -1602182720, -1602182656, -1602182592, -1593835520, -1593827200, -1593827072, -1593827008, -1593826944, -1593826880, -1593826816, -1593826752, -1593802752, -1593794432, -1593794304, -1593794240, -1593794176, -1593794112, -1593794048, -1593793984, -1589641216, -1589632896, -1589632768, -1589632704, -1589632640, -1589632576, -1589632512, -1589632448, -1589608448, -1589600128, -1589600000, -1589599936, -1589599872, -1589599808, -1589599744, -1589599680, -1585446912, -1585438592, -1585438464, -1585438400, -1585438336, -1585438272, -1585438208, -1585438144, -1585414144, -1585405824, -1585405696, -1585405632, -1585405568, -1585405504, -1585405440, -1585405376, -1581252608, -1581244288, -1581244160, -1581244096, -1581244032, -1581243968, -1581243904, -1581243840, -1581219840, -1581211520, -1581211392, -1581211328, -1581211264, -1581211200, -1581211136, -1581211072, -1577058304, -1577049984, -1577049856, -1577049792, -1577049728, -1577049664, -1577049600, -1577049536, -1577025536, -1577017216, -1577017088, -1577017024, -1577016960, -1577016896, -1577016832, -1577016768, -1572864000, -1572855680, -1572855552, -1572855488, -1572855424, -1572855360, -1572855296, -1572855232, -1572831232, -1572822912, -1572822784, -1572822720, -1572822656, -1572822592, -1572822528, -1572822464, 0, 8336, 8464, 8536, 8592, 8664, 8720, 8792, 32768, 41104, 41232, 41304, 41360, 41432, 41488, 41560, 546308096, 546316432, 546316560, 546316632, 546316688, 546316760, 546316816, 546316888, 546340864, 546349200, 546349328, 546349400, 546349456, 546349528, 546349584, 546349656, 554696704, 554705040, 554705168, 554705240, 554705296, 554705368, 554705424, 554705496, 554729472, 554737808, 554737936, 554738008, 554738064, 554738136, 554738192, 554738264, 559415296, 559423632, 559423760, 559423832, 559423888, 559423960, 559424016, 559424088, 559448064, 559456400, 559456528, 559456600, 559456656, 559456728, 559456784, 559456856, 563085312, 563093648, 563093776, 563093848, 563093904, 563093976, 563094032, 563094104, 563118080, 563126416, 563126544, 563126616, 563126672, 563126744, 563126800, 563126872, 567803904, 567812240, 567812368, 567812440, 567812496, 567812568, 567812624, 567812696, 567836672, 567845008, 567845136, 567845208, 567845264, 567845336, 567845392, 567845464, 571473920, 571482256, 571482384, 571482456, 571482512, 571482584, 571482640, 571482712, 571506688, 571515024, 571515152, 571515224, 571515280, 571515352, 571515408, 571515480, 576192512, 576200848, 576200976, 576201048, 576201104, 576201176, 576201232, 576201304, 576225280, 576233616, 576233744, 576233816, 576233872, 576233944, 576234000, 576234072, -2147483648, -2147475312, -2147475184, -2147475112, -2147475056, -2147474984, -2147474928, -2147474856, -2147450880, -2147442544, -2147442416, -2147442344, -2147442288, -2147442216, -2147442160, -2147442088, -1601175552, -1601167216, -1601167088, -1601167016, -1601166960, -1601166888, -1601166832, -1601166760, -1601142784, -1601134448, -1601134320, -1601134248, -1601134192, -1601134120, -1601134064, -1601133992, -1592786944, -1592778608, -1592778480, -1592778408, -1592778352, -1592778280, -1592778224, -1592778152, -1592754176, -1592745840, -1592745712, -1592745640, -1592745584, -1592745512, -1592745456, -1592745384, -1588068352, -1588060016, -1588059888, -1588059816, -1588059760, -1588059688, -1588059632, -1588059560, -1588035584, -1588027248, -1588027120, -1588027048, -1588026992, -1588026920, -1588026864, -1588026792, -1584398336, -1584390000, -1584389872, -1584389800, -1584389744, -1584389672, -1584389616, -1584389544, -1584365568, -1584357232, -1584357104, -1584357032, -1584356976, -1584356904, -1584356848, -1584356776, -1579679744, -1579671408, -1579671280, -1579671208, -1579671152, -1579671080, -1579671024, -1579670952, -1579646976, -1579638640, -1579638512, -1579638440, -1579638384, -1579638312, -1579638256, -1579638184, -1576009728, -1576001392, -1576001264, -1576001192, -1576001136, -1576001064, -1576001008, -1576000936, -1575976960, -1575968624, -1575968496, -1575968424, -1575968368, -1575968296, -1575968240, -1575968168, -1571291136, -1571282800, -1571282672, -1571282600, -1571282544, -1571282472, -1571282416, -1571282344, -1571258368, -1571250032, -1571249904, -1571249832, -1571249776, -1571249704, -1571249648, -1571249576, 0, 8352, 8480, 8560, 8608, 8688, 8736, 8816, 32768, 41120, 41248, 41328, 41376, 41456, 41504, 41584, 547356672, 547365024, 547365152, 547365232, 547365280, 547365360, 547365408, 547365488, 547389440, 547397792, 547397920, 547398000, 547398048, 547398128, 547398176, 547398256, 555745280, 555753632, 555753760, 555753840, 555753888, 555753968, 555754016, 555754096, 555778048, 555786400, 555786528, 555786608, 555786656, 555786736, 555786784, 555786864, 560988160, 560996512, 560996640, 560996720, 560996768, 560996848, 560996896, 560996976, 561020928, 561029280, 561029408, 561029488, 561029536, 561029616, 561029664, 561029744, 564133888, 564142240, 564142368, 564142448, 564142496, 564142576, 564142624, 564142704, 564166656, 564175008, 564175136, 564175216, 564175264, 564175344, 564175392, 564175472, 569376768, 569385120, 569385248, 569385328, 569385376, 569385456, 569385504, 569385584, 569409536, 569417888, 569418016, 569418096, 569418144, 569418224, 569418272, 569418352, 572522496, 572530848, 572530976, 572531056, 572531104, 572531184, 572531232, 572531312, 572555264, 572563616, 572563744, 572563824, 572563872, 572563952, 572564000, 572564080, 577765376, 577773728, 577773856, 577773936, 577773984, 577774064, 577774112, 577774192, 577798144, 577806496, 577806624, 577806704, 577806752, 577806832, 577806880, 577806960, -2147483648, -2147475296, -2147475168, -2147475088, -2147475040, -2147474960, -2147474912, -2147474832, -2147450880, -2147442528, -2147442400, -2147442320, -2147442272, -2147442192, -2147442144, -2147442064, -1600126976, -1600118624, -1600118496, -1600118416, -1600118368, -1600118288, -1600118240, -1600118160, -1600094208, -1600085856, -1600085728, -1600085648, -1600085600, -1600085520, -1600085472, -1600085392, -1591738368, -1591730016, -1591729888, -1591729808, -1591729760, -1591729680, -1591729632, -1591729552, -1591705600, -1591697248, -1591697120, -1591697040, -1591696992, -1591696912, -1591696864, -1591696784, -1586495488, -1586487136, -1586487008, -1586486928, -1586486880, -1586486800, -1586486752, -1586486672, -1586462720, -1586454368, -1586454240, -1586454160, -1586454112, -1586454032, -1586453984, -1586453904, -1583349760, -1583341408, -1583341280, -1583341200, -1583341152, -1583341072, -1583341024, -1583340944, -1583316992, -1583308640, -1583308512, -1583308432, -1583308384, -1583308304, -1583308256, -1583308176, -1578106880, -1578098528, -1578098400, -1578098320, -1578098272, -1578098192, -1578098144, -1578098064, -1578074112, -1578065760, -1578065632, -1578065552, -1578065504, -1578065424, -1578065376, -1578065296, -1574961152, -1574952800, -1574952672, -1574952592, -1574952544, -1574952464, -1574952416, -1574952336, -1574928384, -1574920032, -1574919904, -1574919824, -1574919776, -1574919696, -1574919648, -1574919568, -1569718272, -1569709920, -1569709792, -1569709712, -1569709664, -1569709584, -1569709536, -1569709456, -1569685504, -1569677152, -1569677024, -1569676944, -1569676896, -1569676816, -1569676768, -1569676688, 0, 8368, 8496, 8580, 8624, 8708, 8752, 8836, 32768, 41136, 41264, 41348, 41392, 41476, 41520, 41604, 548405248, 548413616, 548413744, 548413828, 548413872, 548413956, 548414000, 548414084, 548438016, 548446384, 548446512, 548446596, 548446640, 548446724, 548446768, 548446852, 556793856, 556802224, 556802352, 556802436, 556802480, 556802564, 556802608, 556802692, 556826624, 556834992, 556835120, 556835204, 556835248, 556835332, 556835376, 556835460, 562298880, 562307248, 562307376, 562307460, 562307504, 562307588, 562307632, 562307716, 562331648, 562340016, 562340144, 562340228, 562340272, 562340356, 562340400, 562340484, 565182464, 565190832, 565190960, 565191044, 565191088, 565191172, 565191216, 565191300, 565215232, 565223600, 565223728, 565223812, 565223856, 565223940, 565223984, 565224068, 570687488, 570695856, 570695984, 570696068, 570696112, 570696196, 570696240, 570696324, 570720256, 570728624, 570728752, 570728836, 570728880, 570728964, 570729008, 570729092, 573571072, 573579440, 573579568, 573579652, 573579696, 573579780, 573579824, 573579908, 573603840, 573612208, 573612336, 573612420, 573612464, 573612548, 573612592, 573612676, 579076096, 579084464, 579084592, 579084676, 579084720, 579084804, 579084848, 579084932, 579108864, 579117232, 579117360, 579117444, 579117488, 579117572, 579117616, 579117700, -2147483648, -2147475280, -2147475152, -2147475068, -2147475024, -2147474940, -2147474896, -2147474812, -2147450880, -2147442512, -2147442384, -2147442300, -2147442256, -2147442172, -2147442128, -2147442044, -1599078400, -1599070032, -1599069904, -1599069820, -1599069776, -1599069692, -1599069648, -1599069564, -1599045632, -1599037264, -1599037136, -1599037052, -1599037008, -1599036924, -1599036880, -1599036796, -1590689792, -1590681424, -1590681296, -1590681212, -1590681168, -1590681084, -1590681040, -1590680956, -1590657024, -1590648656, -1590648528, -1590648444, -1590648400, -1590648316, -1590648272, -1590648188, -1585184768, -1585176400, -1585176272, -1585176188, -1585176144, -1585176060, -1585176016, -1585175932, -1585152000, -1585143632, -1585143504, -1585143420, -1585143376, -1585143292, -1585143248, -1585143164, -1582301184, -1582292816, -1582292688, -1582292604, -1582292560, -1582292476, -1582292432, -1582292348, -1582268416, -1582260048, -1582259920, -1582259836, -1582259792, -1582259708, -1582259664, -1582259580, -1576796160, -1576787792, -1576787664, -1576787580, -1576787536, -1576787452, -1576787408, -1576787324, -1576763392, -1576755024, -1576754896, -1576754812, -1576754768, -1576754684, -1576754640, -1576754556, -1573912576, -1573904208, -1573904080, -1573903996, -1573903952, -1573903868, -1573903824, -1573903740, -1573879808, -1573871440, -1573871312, -1573871228, -1573871184, -1573871100, -1573871056, -1573870972, -1568407552, -1568399184, -1568399056, -1568398972, -1568398928, -1568398844, -1568398800, -1568398716, -1568374784, -1568366416, -1568366288, -1568366204, -1568366160, -1568366076, -1568366032, -1568365948, 0, 8384, 8512, 8592, 8640, 8720, 8768, 8848, 32768, 41152, 41280, 41360, 41408, 41488, 41536, 41616, 549453824, 549462208, 549462336, 549462416, 549462464, 549462544, 549462592, 549462672, 549486592, 549494976, 549495104, 549495184, 549495232, 549495312, 549495360, 549495440, 557842432, 557850816, 557850944, 557851024, 557851072, 557851152, 557851200, 557851280, 557875200, 557883584, 557883712, 557883792, 557883840, 557883920, 557883968, 557884048, 563085312, 563093696, 563093824, 563093904, 563093952, 563094032, 563094080, 563094160, 563118080, 563126464, 563126592, 563126672, 563126720, 563126800, 563126848, 563126928, 566231040, 566239424, 566239552, 566239632, 566239680, 566239760, 566239808, 566239888, 566263808, 566272192, 566272320, 566272400, 566272448, 566272528, 566272576, 566272656, 571473920, 571482304, 571482432, 571482512, 571482560, 571482640, 571482688, 571482768, 571506688, 571515072, 571515200, 571515280, 571515328, 571515408, 571515456, 571515536, 574619648, 574628032, 574628160, 574628240, 574628288, 574628368, 574628416, 574628496, 574652416, 574660800, 574660928, 574661008, 574661056, 574661136, 574661184, 574661264, 579862528, 579870912, 579871040, 579871120, 579871168, 579871248, 579871296, 579871376, 579895296, 579903680, 579903808, 579903888, 579903936, 579904016, 579904064, 579904144, -2147483648, -2147475264, -2147475136, -2147475056, -2147475008, -2147474928, -2147474880, -2147474800, -2147450880, -2147442496, -2147442368, -2147442288, -2147442240, -2147442160, -2147442112, -2147442032, -1598029824, -1598021440, -1598021312, -1598021232, -1598021184, -1598021104, -1598021056, -1598020976, -1597997056, -1597988672, -1597988544, -1597988464, -1597988416, -1597988336, -1597988288, -1597988208, -1589641216, -1589632832, -1589632704, -1589632624, -1589632576, -1589632496, -1589632448, -1589632368, -1589608448, -1589600064, -1589599936, -1589599856, -1589599808, -1589599728, -1589599680, -1589599600, -1584398336, -1584389952, -1584389824, -1584389744, -1584389696, -1584389616, -1584389568, -1584389488, -1584365568, -1584357184, -1584357056, -1584356976, -1584356928, -1584356848, -1584356800, -1584356720, -1581252608, -1581244224, -1581244096, -1581244016, -1581243968, -1581243888, -1581243840, -1581243760, -1581219840, -1581211456, -1581211328, -1581211248, -1581211200, -1581211120, -1581211072, -1581210992, -1576009728, -1576001344, -1576001216, -1576001136, -1576001088, -1576001008, -1576000960, -1576000880, -1575976960, -1575968576, -1575968448, -1575968368, -1575968320, -1575968240, -1575968192, -1575968112, -1572864000, -1572855616, -1572855488, -1572855408, -1572855360, -1572855280, -1572855232, -1572855152, -1572831232, -1572822848, -1572822720, -1572822640, -1572822592, -1572822512, -1572822464, -1572822384, -1567621120, -1567612736, -1567612608, -1567612528, -1567612480, -1567612400, -1567612352, -1567612272, -1567588352, -1567579968, -1567579840, -1567579760, -1567579712, -1567579632, -1567579584, -1567579504, 0, 8400, 8528, 8604, 8656, 8732, 8784, 8860, 32768, 41168, 41296, 41372, 41424, 41500, 41552, 41628, 550502400, 550510800, 550510928, 550511004, 550511056, 550511132, 550511184, 550511260, 550535168, 550543568, 550543696, 550543772, 550543824, 550543900, 550543952, 550544028, 558891008, 558899408, 558899536, 558899612, 558899664, 558899740, 558899792, 558899868, 558923776, 558932176, 558932304, 558932380, 558932432, 558932508, 558932560, 558932636, 563871744, 563880144, 563880272, 563880348, 563880400, 563880476, 563880528, 563880604, 563904512, 563912912, 563913040, 563913116, 563913168, 563913244, 563913296, 563913372, 567279616, 567288016, 567288144, 567288220, 567288272, 567288348, 567288400, 567288476, 567312384, 567320784, 567320912, 567320988, 567321040, 567321116, 567321168, 567321244, 572260352, 572268752, 572268880, 572268956, 572269008, 572269084, 572269136, 572269212, 572293120, 572301520, 572301648, 572301724, 572301776, 572301852, 572301904, 572301980, 575668224, 575676624, 575676752, 575676828, 575676880, 575676956, 575677008, 575677084, 575700992, 575709392, 575709520, 575709596, 575709648, 575709724, 575709776, 575709852, 580648960, 580657360, 580657488, 580657564, 580657616, 580657692, 580657744, 580657820, 580681728, 580690128, 580690256, 580690332, 580690384, 580690460, 580690512, 580690588, -2147483648, -2147475248, -2147475120, -2147475044, -2147474992, -2147474916, -2147474864, -2147474788, -2147450880, -2147442480, -2147442352, -2147442276, -2147442224, -2147442148, -2147442096, -2147442020, -1596981248, -1596972848, -1596972720, -1596972644, -1596972592, -1596972516, -1596972464, -1596972388, -1596948480, -1596940080, -1596939952, -1596939876, -1596939824, -1596939748, -1596939696, -1596939620, -1588592640, -1588584240, -1588584112, -1588584036, -1588583984, -1588583908, -1588583856, -1588583780, -1588559872, -1588551472, -1588551344, -1588551268, -1588551216, -1588551140, -1588551088, -1588551012, -1583611904, -1583603504, -1583603376, -1583603300, -1583603248, -1583603172, -1583603120, -1583603044, -1583579136, -1583570736, -1583570608, -1583570532, -1583570480, -1583570404, -1583570352, -1583570276, -1580204032, -1580195632, -1580195504, -1580195428, -1580195376, -1580195300, -1580195248, -1580195172, -1580171264, -1580162864, -1580162736, -1580162660, -1580162608, -1580162532, -1580162480, -1580162404, -1575223296, -1575214896, -1575214768, -1575214692, -1575214640, -1575214564, -1575214512, -1575214436, -1575190528, -1575182128, -1575182000, -1575181924, -1575181872, -1575181796, -1575181744, -1575181668, -1571815424, -1571807024, -1571806896, -1571806820, -1571806768, -1571806692, -1571806640, -1571806564, -1571782656, -1571774256, -1571774128, -1571774052, -1571774000, -1571773924, -1571773872, -1571773796, -1566834688, -1566826288, -1566826160, -1566826084, -1566826032, -1566825956, -1566825904, -1566825828, -1566801920, -1566793520, -1566793392, -1566793316, -1566793264, -1566793188, -1566793136, -1566793060, 0, 8416, 8544, 8616, 8672, 8744, 8800, 8872, 32768, 41184, 41312, 41384, 41440, 41512, 41568, 41640, 551550976, 551559392, 551559520, 551559592, 551559648, 551559720, 551559776, 551559848, 551583744, 551592160, 551592288, 551592360, 551592416, 551592488, 551592544, 551592616, 559939584, 559948000, 559948128, 559948200, 559948256, 559948328, 559948384, 559948456, 559972352, 559980768, 559980896, 559980968, 559981024, 559981096, 559981152, 559981224, 564658176, 564666592, 564666720, 564666792, 564666848, 564666920, 564666976, 564667048, 564690944, 564699360, 564699488, 564699560, 564699616, 564699688, 564699744, 564699816, 568328192, 568336608, 568336736, 568336808, 568336864, 568336936, 568336992, 568337064, 568360960, 568369376, 568369504, 568369576, 568369632, 568369704, 568369760, 568369832, 573046784, 573055200, 573055328, 573055400, 573055456, 573055528, 573055584, 573055656, 573079552, 573087968, 573088096, 573088168, 573088224, 573088296, 573088352, 573088424, 576716800, 576725216, 576725344, 576725416, 576725472, 576725544, 576725600, 576725672, 576749568, 576757984, 576758112, 576758184, 576758240, 576758312, 576758368, 576758440, 581435392, 581443808, 581443936, 581444008, 581444064, 581444136, 581444192, 581444264, 581468160, 581476576, 581476704, 581476776, 581476832, 581476904, 581476960, 581477032, -2147483648, -2147475232, -2147475104, -2147475032, -2147474976, -2147474904, -2147474848, -2147474776, -2147450880, -2147442464, -2147442336, -2147442264, -2147442208, -2147442136, -2147442080, -2147442008, -1595932672, -1595924256, -1595924128, -1595924056, -1595924000, -1595923928, -1595923872, -1595923800, -1595899904, -1595891488, -1595891360, -1595891288, -1595891232, -1595891160, -1595891104, -1595891032, -1587544064, -1587535648, -1587535520, -1587535448, -1587535392, -1587535320, -1587535264, -1587535192, -1587511296, -1587502880, -1587502752, -1587502680, -1587502624, -1587502552, -1587502496, -1587502424, -1582825472, -1582817056, -1582816928, -1582816856, -1582816800, -1582816728, -1582816672, -1582816600, -1582792704, -1582784288, -1582784160, -1582784088, -1582784032, -1582783960, -1582783904, -1582783832, -1579155456, -1579147040, -1579146912, -1579146840, -1579146784, -1579146712, -1579146656, -1579146584, -1579122688, -1579114272, -1579114144, -1579114072, -1579114016, -1579113944, -1579113888, -1579113816, -1574436864, -1574428448, -1574428320, -1574428248, -1574428192, -1574428120, -1574428064, -1574427992, -1574404096, -1574395680, -1574395552, -1574395480, -1574395424, -1574395352, -1574395296, -1574395224, -1570766848, -1570758432, -1570758304, -1570758232, -1570758176, -1570758104, -1570758048, -1570757976, -1570734080, -1570725664, -1570725536, -1570725464, -1570725408, -1570725336, -1570725280, -1570725208, -1566048256, -1566039840, -1566039712, -1566039640, -1566039584, -1566039512, -1566039456, -1566039384, -1566015488, -1566007072, -1566006944, -1566006872, -1566006816, -1566006744, -1566006688, -1566006616, 0, 8432, 8560, 8628, 8688, 8756, 8816, 8884, 32768, 41200, 41328, 41396, 41456, 41524, 41584, 41652, 552599552, 552607984, 552608112, 552608180, 552608240, 552608308, 552608368, 552608436, 552632320, 552640752, 552640880, 552640948, 552641008, 552641076, 552641136, 552641204, 560988160, 560996592, 560996720, 560996788, 560996848, 560996916, 560996976, 560997044, 561020928, 561029360, 561029488, 561029556, 561029616, 561029684, 561029744, 561029812, 565444608, 565453040, 565453168, 565453236, 565453296, 565453364, 565453424, 565453492, 565477376, 565485808, 565485936, 565486004, 565486064, 565486132, 565486192, 565486260, 569376768, 569385200, 569385328, 569385396, 569385456, 569385524, 569385584, 569385652, 569409536, 569417968, 569418096, 569418164, 569418224, 569418292, 569418352, 569418420, 573833216, 573841648, 573841776, 573841844, 573841904, 573841972, 573842032, 573842100, 573865984, 573874416, 573874544, 573874612, 573874672, 573874740, 573874800, 573874868, 577765376, 577773808, 577773936, 577774004, 577774064, 577774132, 577774192, 577774260, 577798144, 577806576, 577806704, 577806772, 577806832, 577806900, 577806960, 577807028, 582221824, 582230256, 582230384, 582230452, 582230512, 582230580, 582230640, 582230708, 582254592, 582263024, 582263152, 582263220, 582263280, 582263348, 582263408, 582263476, -2147483648, -2147475216, -2147475088, -2147475020, -2147474960, -2147474892, -2147474832, -2147474764, -2147450880, -2147442448, -2147442320, -2147442252, -2147442192, -2147442124, -2147442064, -2147441996, -1594884096, -1594875664, -1594875536, -1594875468, -1594875408, -1594875340, -1594875280, -1594875212, -1594851328, -1594842896, -1594842768, -1594842700, -1594842640, -1594842572, -1594842512, -1594842444, -1586495488, -1586487056, -1586486928, -1586486860, -1586486800, -1586486732, -1586486672, -1586486604, -1586462720, -1586454288, -1586454160, -1586454092, -1586454032, -1586453964, -1586453904, -1586453836, -1582039040, -1582030608, -1582030480, -1582030412, -1582030352, -1582030284, -1582030224, -1582030156, -1582006272, -1581997840, -1581997712, -1581997644, -1581997584, -1581997516, -1581997456, -1581997388, -1578106880, -1578098448, -1578098320, -1578098252, -1578098192, -1578098124, -1578098064, -1578097996, -1578074112, -1578065680, -1578065552, -1578065484, -1578065424, -1578065356, -1578065296, -1578065228, -1573650432, -1573642000, -1573641872, -1573641804, -1573641744, -1573641676, -1573641616, -1573641548, -1573617664, -1573609232, -1573609104, -1573609036, -1573608976, -1573608908, -1573608848, -1573608780, -1569718272, -1569709840, -1569709712, -1569709644, -1569709584, -1569709516, -1569709456, -1569709388, -1569685504, -1569677072, -1569676944, -1569676876, -1569676816, -1569676748, -1569676688, -1569676620, -1565261824, -1565253392, -1565253264, -1565253196, -1565253136, -1565253068, -1565253008, -1565252940, -1565229056, -1565220624, -1565220496, -1565220428, -1565220368, -1565220300, -1565220240, -1565220172, 0, 8448, 8576, 8640, 8704, 8768, 8832, 8896, 32768, 41216, 41344, 41408, 41472, 41536, 41600, 41664, 553648128, 553656576, 553656704, 553656768, 553656832, 553656896, 553656960, 553657024, 553680896, 553689344, 553689472, 553689536, 553689600, 553689664, 553689728, 553689792, 562036736, 562045184, 562045312, 562045376, 562045440, 562045504, 562045568, 562045632, 562069504, 562077952, 562078080, 562078144, 562078208, 562078272, 562078336, 562078400, 566231040, 566239488, 566239616, 566239680, 566239744, 566239808, 566239872, 566239936, 566263808, 566272256, 566272384, 566272448, 566272512, 566272576, 566272640, 566272704, 570425344, 570433792, 570433920, 570433984, 570434048, 570434112, 570434176, 570434240, 570458112, 570466560, 570466688, 570466752, 570466816, 570466880, 570466944, 570467008, 574619648, 574628096, 574628224, 574628288, 574628352, 574628416, 574628480, 574628544, 574652416, 574660864, 574660992, 574661056, 574661120, 574661184, 574661248, 574661312, 578813952, 578822400, 578822528, 578822592, 578822656, 578822720, 578822784, 578822848, 578846720, 578855168, 578855296, 578855360, 578855424, 578855488, 578855552, 578855616, 583008256, 583016704, 583016832, 583016896, 583016960, 583017024, 583017088, 583017152, 583041024, 583049472, 583049600, 583049664, 583049728, 583049792, 583049856, 583049920, -2147483648, -2147475200, -2147475072, -2147475008, -2147474944, -2147474880, -2147474816, -2147474752, -2147450880, -2147442432, -2147442304, -2147442240, -2147442176, -2147442112, -2147442048, -2147441984, -1593835520, -1593827072, -1593826944, -1593826880, -1593826816, -1593826752, -1593826688, -1593826624, -1593802752, -1593794304, -1593794176, -1593794112, -1593794048, -1593793984, -1593793920, -1593793856, -1585446912, -1585438464, -1585438336, -1585438272, -1585438208, -1585438144, -1585438080, -1585438016, -1585414144, -1585405696, -1585405568, -1585405504, -1585405440, -1585405376, -1585405312, -1585405248, -1581252608, -1581244160, -1581244032, -1581243968, -1581243904, -1581243840, -1581243776, -1581243712, -1581219840, -1581211392, -1581211264, -1581211200, -1581211136, -1581211072, -1581211008, -1581210944, -1577058304, -1577049856, -1577049728, -1577049664, -1577049600, -1577049536, -1577049472, -1577049408, -1577025536, -1577017088, -1577016960, -1577016896, -1577016832, -1577016768, -1577016704, -1577016640, -1572864000, -1572855552, -1572855424, -1572855360, -1572855296, -1572855232, -1572855168, -1572855104, -1572831232, -1572822784, -1572822656, -1572822592, -1572822528, -1572822464, -1572822400, -1572822336, -1568669696, -1568661248, -1568661120, -1568661056, -1568660992, -1568660928, -1568660864, -1568660800, -1568636928, -1568628480, -1568628352, -1568628288, -1568628224, -1568628160, -1568628096, -1568628032, -1564475392, -1564466944, -1564466816, -1564466752, -1564466688, -1564466624, -1564466560, -1564466496, -1564442624, -1564434176, -1564434048, -1564433984, -1564433920, -1564433856, -1564433792, -1564433728, 0, 8464, 8592, 8664, 8720, 8792, 8848, 8920, 32768, 41232, 41360, 41432, 41488, 41560, 41616, 41688, 554696704, 554705168, 554705296, 554705368, 554705424, 554705496, 554705552, 554705624, 554729472, 554737936, 554738064, 554738136, 554738192, 554738264, 554738320, 554738392, 563085312, 563093776, 563093904, 563093976, 563094032, 563094104, 563094160, 563094232, 563118080, 563126544, 563126672, 563126744, 563126800, 563126872, 563126928, 563127000, 567803904, 567812368, 567812496, 567812568, 567812624, 567812696, 567812752, 567812824, 567836672, 567845136, 567845264, 567845336, 567845392, 567845464, 567845520, 567845592, 571473920, 571482384, 571482512, 571482584, 571482640, 571482712, 571482768, 571482840, 571506688, 571515152, 571515280, 571515352, 571515408, 571515480, 571515536, 571515608, 576192512, 576200976, 576201104, 576201176, 576201232, 576201304, 576201360, 576201432, 576225280, 576233744, 576233872, 576233944, 576234000, 576234072, 576234128, 576234200, 579862528, 579870992, 579871120, 579871192, 579871248, 579871320, 579871376, 579871448, 579895296, 579903760, 579903888, 579903960, 579904016, 579904088, 579904144, 579904216, 584581120, 584589584, 584589712, 584589784, 584589840, 584589912, 584589968, 584590040, 584613888, 584622352, 584622480, 584622552, 584622608, 584622680, 584622736, 584622808, -2147483648, -2147475184, -2147475056, -2147474984, -2147474928, -2147474856, -2147474800, -2147474728, -2147450880, -2147442416, -2147442288, -2147442216, -2147442160, -2147442088, -2147442032, -2147441960, -1592786944, -1592778480, -1592778352, -1592778280, -1592778224, -1592778152, -1592778096, -1592778024, -1592754176, -1592745712, -1592745584, -1592745512, -1592745456, -1592745384, -1592745328, -1592745256, -1584398336, -1584389872, -1584389744, -1584389672, -1584389616, -1584389544, -1584389488, -1584389416, -1584365568, -1584357104, -1584356976, -1584356904, -1584356848, -1584356776, -1584356720, -1584356648, -1579679744, -1579671280, -1579671152, -1579671080, -1579671024, -1579670952, -1579670896, -1579670824, -1579646976, -1579638512, -1579638384, -1579638312, -1579638256, -1579638184, -1579638128, -1579638056, -1576009728, -1576001264, -1576001136, -1576001064, -1576001008, -1576000936, -1576000880, -1576000808, -1575976960, -1575968496, -1575968368, -1575968296, -1575968240, -1575968168, -1575968112, -1575968040, -1571291136, -1571282672, -1571282544, -1571282472, -1571282416, -1571282344, -1571282288, -1571282216, -1571258368, -1571249904, -1571249776, -1571249704, -1571249648, -1571249576, -1571249520, -1571249448, -1567621120, -1567612656, -1567612528, -1567612456, -1567612400, -1567612328, -1567612272, -1567612200, -1567588352, -1567579888, -1567579760, -1567579688, -1567579632, -1567579560, -1567579504, -1567579432, -1562902528, -1562894064, -1562893936, -1562893864, -1562893808, -1562893736, -1562893680, -1562893608, -1562869760, -1562861296, -1562861168, -1562861096, -1562861040, -1562860968, -1562860912, -1562860840, 0, 8480, 8608, 8688, 8736, 8816, 8864, 8944, 32768, 41248, 41376, 41456, 41504, 41584, 41632, 41712, 555745280, 555753760, 555753888, 555753968, 555754016, 555754096, 555754144, 555754224, 555778048, 555786528, 555786656, 555786736, 555786784, 555786864, 555786912, 555786992, 564133888, 564142368, 564142496, 564142576, 564142624, 564142704, 564142752, 564142832, 564166656, 564175136, 564175264, 564175344, 564175392, 564175472, 564175520, 564175600, 569376768, 569385248, 569385376, 569385456, 569385504, 569385584, 569385632, 569385712, 569409536, 569418016, 569418144, 569418224, 569418272, 569418352, 569418400, 569418480, 572522496, 572530976, 572531104, 572531184, 572531232, 572531312, 572531360, 572531440, 572555264, 572563744, 572563872, 572563952, 572564000, 572564080, 572564128, 572564208, 577765376, 577773856, 577773984, 577774064, 577774112, 577774192, 577774240, 577774320, 577798144, 577806624, 577806752, 577806832, 577806880, 577806960, 577807008, 577807088, 580911104, 580919584, 580919712, 580919792, 580919840, 580919920, 580919968, 580920048, 580943872, 580952352, 580952480, 580952560, 580952608, 580952688, 580952736, 580952816, 586153984, 586162464, 586162592, 586162672, 586162720, 586162800, 586162848, 586162928, 586186752, 586195232, 586195360, 586195440, 586195488, 586195568, 586195616, 586195696, -2147483648, -2147475168, -2147475040, -2147474960, -2147474912, -2147474832, -2147474784, -2147474704, -2147450880, -2147442400, -2147442272, -2147442192, -2147442144, -2147442064, -2147442016, -2147441936, -1591738368, -1591729888, -1591729760, -1591729680, -1591729632, -1591729552, -1591729504, -1591729424, -1591705600, -1591697120, -1591696992, -1591696912, -1591696864, -1591696784, -1591696736, -1591696656, -1583349760, -1583341280, -1583341152, -1583341072, -1583341024, -1583340944, -1583340896, -1583340816, -1583316992, -1583308512, -1583308384, -1583308304, -1583308256, -1583308176, -1583308128, -1583308048, -1578106880, -1578098400, -1578098272, -1578098192, -1578098144, -1578098064, -1578098016, -1578097936, -1578074112, -1578065632, -1578065504, -1578065424, -1578065376, -1578065296, -1578065248, -1578065168, -1574961152, -1574952672, -1574952544, -1574952464, -1574952416, -1574952336, -1574952288, -1574952208, -1574928384, -1574919904, -1574919776, -1574919696, -1574919648, -1574919568, -1574919520, -1574919440, -1569718272, -1569709792, -1569709664, -1569709584, -1569709536, -1569709456, -1569709408, -1569709328, -1569685504, -1569677024, -1569676896, -1569676816, -1569676768, -1569676688, -1569676640, -1569676560, -1566572544, -1566564064, -1566563936, -1566563856, -1566563808, -1566563728, -1566563680, -1566563600, -1566539776, -1566531296, -1566531168, -1566531088, -1566531040, -1566530960, -1566530912, -1566530832, -1561329664, -1561321184, -1561321056, -1561320976, -1561320928, -1561320848, -1561320800, -1561320720, -1561296896, -1561288416, -1561288288, -1561288208, -1561288160, -1561288080, -1561288032, -1561287952, 0, 8496, 8624, 8708, 8752, 8836, 8880, 8964, 32768, 41264, 41392, 41476, 41520, 41604, 41648, 41732, 556793856, 556802352, 556802480, 556802564, 556802608, 556802692, 556802736, 556802820, 556826624, 556835120, 556835248, 556835332, 556835376, 556835460, 556835504, 556835588, 565182464, 565190960, 565191088, 565191172, 565191216, 565191300, 565191344, 565191428, 565215232, 565223728, 565223856, 565223940, 565223984, 565224068, 565224112, 565224196, 570687488, 570695984, 570696112, 570696196, 570696240, 570696324, 570696368, 570696452, 570720256, 570728752, 570728880, 570728964, 570729008, 570729092, 570729136, 570729220, 573571072, 573579568, 573579696, 573579780, 573579824, 573579908, 573579952, 573580036, 573603840, 573612336, 573612464, 573612548, 573612592, 573612676, 573612720, 573612804, 579076096, 579084592, 579084720, 579084804, 579084848, 579084932, 579084976, 579085060, 579108864, 579117360, 579117488, 579117572, 579117616, 579117700, 579117744, 579117828, 581959680, 581968176, 581968304, 581968388, 581968432, 581968516, 581968560, 581968644, 581992448, 582000944, 582001072, 582001156, 582001200, 582001284, 582001328, 582001412, 587464704, 587473200, 587473328, 587473412, 587473456, 587473540, 587473584, 587473668, 587497472, 587505968, 587506096, 587506180, 587506224, 587506308, 587506352, 587506436, -2147483648, -2147475152, -2147475024, -2147474940, -2147474896, -2147474812, -2147474768, -2147474684, -2147450880, -2147442384, -2147442256, -2147442172, -2147442128, -2147442044, -2147442000, -2147441916, -1590689792, -1590681296, -1590681168, -1590681084, -1590681040, -1590680956, -1590680912, -1590680828, -1590657024, -1590648528, -1590648400, -1590648316, -1590648272, -1590648188, -1590648144, -1590648060, -1582301184, -1582292688, -1582292560, -1582292476, -1582292432, -1582292348, -1582292304, -1582292220, -1582268416, -1582259920, -1582259792, -1582259708, -1582259664, -1582259580, -1582259536, -1582259452, -1576796160, -1576787664, -1576787536, -1576787452, -1576787408, -1576787324, -1576787280, -1576787196, -1576763392, -1576754896, -1576754768, -1576754684, -1576754640, -1576754556, -1576754512, -1576754428, -1573912576, -1573904080, -1573903952, -1573903868, -1573903824, -1573903740, -1573903696, -1573903612, -1573879808, -1573871312, -1573871184, -1573871100, -1573871056, -1573870972, -1573870928, -1573870844, -1568407552, -1568399056, -1568398928, -1568398844, -1568398800, -1568398716, -1568398672, -1568398588, -1568374784, -1568366288, -1568366160, -1568366076, -1568366032, -1568365948, -1568365904, -1568365820, -1565523968, -1565515472, -1565515344, -1565515260, -1565515216, -1565515132, -1565515088, -1565515004, -1565491200, -1565482704, -1565482576, -1565482492, -1565482448, -1565482364, -1565482320, -1565482236, -1560018944, -1560010448, -1560010320, -1560010236, -1560010192, -1560010108, -1560010064, -1560009980, -1559986176, -1559977680, -1559977552, -1559977468, -1559977424, -1559977340, -1559977296, -1559977212, 0, 8512, 8640, 8720, 8768, 8848, 8896, 8976, 32768, 41280, 41408, 41488, 41536, 41616, 41664, 41744, 557842432, 557850944, 557851072, 557851152, 557851200, 557851280, 557851328, 557851408, 557875200, 557883712, 557883840, 557883920, 557883968, 557884048, 557884096, 557884176, 566231040, 566239552, 566239680, 566239760, 566239808, 566239888, 566239936, 566240016, 566263808, 566272320, 566272448, 566272528, 566272576, 566272656, 566272704, 566272784, 571473920, 571482432, 571482560, 571482640, 571482688, 571482768, 571482816, 571482896, 571506688, 571515200, 571515328, 571515408, 571515456, 571515536, 571515584, 571515664, 574619648, 574628160, 574628288, 574628368, 574628416, 574628496, 574628544, 574628624, 574652416, 574660928, 574661056, 574661136, 574661184, 574661264, 574661312, 574661392, 579862528, 579871040, 579871168, 579871248, 579871296, 579871376, 579871424, 579871504, 579895296, 579903808, 579903936, 579904016, 579904064, 579904144, 579904192, 579904272, 583008256, 583016768, 583016896, 583016976, 583017024, 583017104, 583017152, 583017232, 583041024, 583049536, 583049664, 583049744, 583049792, 583049872, 583049920, 583050000, 588251136, 588259648, 588259776, 588259856, 588259904, 588259984, 588260032, 588260112, 588283904, 588292416, 588292544, 588292624, 588292672, 588292752, 588292800, 588292880, -2147483648, -2147475136, -2147475008, -2147474928, -2147474880, -2147474800, -2147474752, -2147474672, -2147450880, -2147442368, -2147442240, -2147442160, -2147442112, -2147442032, -2147441984, -2147441904, -1589641216, -1589632704, -1589632576, -1589632496, -1589632448, -1589632368, -1589632320, -1589632240, -1589608448, -1589599936, -1589599808, -1589599728, -1589599680, -1589599600, -1589599552, -1589599472, -1581252608, -1581244096, -1581243968, -1581243888, -1581243840, -1581243760, -1581243712, -1581243632, -1581219840, -1581211328, -1581211200, -1581211120, -1581211072, -1581210992, -1581210944, -1581210864, -1576009728, -1576001216, -1576001088, -1576001008, -1576000960, -1576000880, -1576000832, -1576000752, -1575976960, -1575968448, -1575968320, -1575968240, -1575968192, -1575968112, -1575968064, -1575967984, -1572864000, -1572855488, -1572855360, -1572855280, -1572855232, -1572855152, -1572855104, -1572855024, -1572831232, -1572822720, -1572822592, -1572822512, -1572822464, -1572822384, -1572822336, -1572822256, -1567621120, -1567612608, -1567612480, -1567612400, -1567612352, -1567612272, -1567612224, -1567612144, -1567588352, -1567579840, -1567579712, -1567579632, -1567579584, -1567579504, -1567579456, -1567579376, -1564475392, -1564466880, -1564466752, -1564466672, -1564466624, -1564466544, -1564466496, -1564466416, -1564442624, -1564434112, -1564433984, -1564433904, -1564433856, -1564433776, -1564433728, -1564433648, -1559232512, -1559224000, -1559223872, -1559223792, -1559223744, -1559223664, -1559223616, -1559223536, -1559199744, -1559191232, -1559191104, -1559191024, -1559190976, -1559190896, -1559190848, -1559190768, 0, 8528, 8656, 8732, 8784, 8860, 8912, 8988, 32768, 41296, 41424, 41500, 41552, 41628, 41680, 41756, 558891008, 558899536, 558899664, 558899740, 558899792, 558899868, 558899920, 558899996, 558923776, 558932304, 558932432, 558932508, 558932560, 558932636, 558932688, 558932764, 567279616, 567288144, 567288272, 567288348, 567288400, 567288476, 567288528, 567288604, 567312384, 567320912, 567321040, 567321116, 567321168, 567321244, 567321296, 567321372, 572260352, 572268880, 572269008, 572269084, 572269136, 572269212, 572269264, 572269340, 572293120, 572301648, 572301776, 572301852, 572301904, 572301980, 572302032, 572302108, 575668224, 575676752, 575676880, 575676956, 575677008, 575677084, 575677136, 575677212, 575700992, 575709520, 575709648, 575709724, 575709776, 575709852, 575709904, 575709980, 580648960, 580657488, 580657616, 580657692, 580657744, 580657820, 580657872, 580657948, 580681728, 580690256, 580690384, 580690460, 580690512, 580690588, 580690640, 580690716, 584056832, 584065360, 584065488, 584065564, 584065616, 584065692, 584065744, 584065820, 584089600, 584098128, 584098256, 584098332, 584098384, 584098460, 584098512, 584098588, 589037568, 589046096, 589046224, 589046300, 589046352, 589046428, 589046480, 589046556, 589070336, 589078864, 589078992, 589079068, 589079120, 589079196, 589079248, 589079324, -2147483648, -2147475120, -2147474992, -2147474916, -2147474864, -2147474788, -2147474736, -2147474660, -2147450880, -2147442352, -2147442224, -2147442148, -2147442096, -2147442020, -2147441968, -2147441892, -1588592640, -1588584112, -1588583984, -1588583908, -1588583856, -1588583780, -1588583728, -1588583652, -1588559872, -1588551344, -1588551216, -1588551140, -1588551088, -1588551012, -1588550960, -1588550884, -1580204032, -1580195504, -1580195376, -1580195300, -1580195248, -1580195172, -1580195120, -1580195044, -1580171264, -1580162736, -1580162608, -1580162532, -1580162480, -1580162404, -1580162352, -1580162276, -1575223296, -1575214768, -1575214640, -1575214564, -1575214512, -1575214436, -1575214384, -1575214308, -1575190528, -1575182000, -1575181872, -1575181796, -1575181744, -1575181668, -1575181616, -1575181540, -1571815424, -1571806896, -1571806768, -1571806692, -1571806640, -1571806564, -1571806512, -1571806436, -1571782656, -1571774128, -1571774000, -1571773924, -1571773872, -1571773796, -1571773744, -1571773668, -1566834688, -1566826160, -1566826032, -1566825956, -1566825904, -1566825828, -1566825776, -1566825700, -1566801920, -1566793392, -1566793264, -1566793188, -1566793136, -1566793060, -1566793008, -1566792932, -1563426816, -1563418288, -1563418160, -1563418084, -1563418032, -1563417956, -1563417904, -1563417828, -1563394048, -1563385520, -1563385392, -1563385316, -1563385264, -1563385188, -1563385136, -1563385060, -1558446080, -1558437552, -1558437424, -1558437348, -1558437296, -1558437220, -1558437168, -1558437092, -1558413312, -1558404784, -1558404656, -1558404580, -1558404528, -1558404452, -1558404400, -1558404324, 0, 8544, 8672, 8744, 8800, 8872, 8928, 9000, 32768, 41312, 41440, 41512, 41568, 41640, 41696, 41768, 559939584, 559948128, 559948256, 559948328, 559948384, 559948456, 559948512, 559948584, 559972352, 559980896, 559981024, 559981096, 559981152, 559981224, 559981280, 559981352, 568328192, 568336736, 568336864, 568336936, 568336992, 568337064, 568337120, 568337192, 568360960, 568369504, 568369632, 568369704, 568369760, 568369832, 568369888, 568369960, 573046784, 573055328, 573055456, 573055528, 573055584, 573055656, 573055712, 573055784, 573079552, 573088096, 573088224, 573088296, 573088352, 573088424, 573088480, 573088552, 576716800, 576725344, 576725472, 576725544, 576725600, 576725672, 576725728, 576725800, 576749568, 576758112, 576758240, 576758312, 576758368, 576758440, 576758496, 576758568, 581435392, 581443936, 581444064, 581444136, 581444192, 581444264, 581444320, 581444392, 581468160, 581476704, 581476832, 581476904, 581476960, 581477032, 581477088, 581477160, 585105408, 585113952, 585114080, 585114152, 585114208, 585114280, 585114336, 585114408, 585138176, 585146720, 585146848, 585146920, 585146976, 585147048, 585147104, 585147176, 589824000, 589832544, 589832672, 589832744, 589832800, 589832872, 589832928, 589833000, 589856768, 589865312, 589865440, 589865512, 589865568, 589865640, 589865696, 589865768, -2147483648, -2147475104, -2147474976, -2147474904, -2147474848, -2147474776, -2147474720, -2147474648, -2147450880, -2147442336, -2147442208, -2147442136, -2147442080, -2147442008, -2147441952, -2147441880, -1587544064, -1587535520, -1587535392, -1587535320, -1587535264, -1587535192, -1587535136, -1587535064, -1587511296, -1587502752, -1587502624, -1587502552, -1587502496, -1587502424, -1587502368, -1587502296, -1579155456, -1579146912, -1579146784, -1579146712, -1579146656, -1579146584, -1579146528, -1579146456, -1579122688, -1579114144, -1579114016, -1579113944, -1579113888, -1579113816, -1579113760, -1579113688, -1574436864, -1574428320, -1574428192, -1574428120, -1574428064, -1574427992, -1574427936, -1574427864, -1574404096, -1574395552, -1574395424, -1574395352, -1574395296, -1574395224, -1574395168, -1574395096, -1570766848, -1570758304, -1570758176, -1570758104, -1570758048, -1570757976, -1570757920, -1570757848, -1570734080, -1570725536, -1570725408, -1570725336, -1570725280, -1570725208, -1570725152, -1570725080, -1566048256, -1566039712, -1566039584, -1566039512, -1566039456, -1566039384, -1566039328, -1566039256, -1566015488, -1566006944, -1566006816, -1566006744, -1566006688, -1566006616, -1566006560, -1566006488, -1562378240, -1562369696, -1562369568, -1562369496, -1562369440, -1562369368, -1562369312, -1562369240, -1562345472, -1562336928, -1562336800, -1562336728, -1562336672, -1562336600, -1562336544, -1562336472, -1557659648, -1557651104, -1557650976, -1557650904, -1557650848, -1557650776, -1557650720, -1557650648, -1557626880, -1557618336, -1557618208, -1557618136, -1557618080, -1557618008, -1557617952, -1557617880, 0, 8560, 8688, 8756, 8816, 8884, 8944, 9012, 32768, 41328, 41456, 41524, 41584, 41652, 41712, 41780, 560988160, 560996720, 560996848, 560996916, 560996976, 560997044, 560997104, 560997172, 561020928, 561029488, 561029616, 561029684, 561029744, 561029812, 561029872, 561029940, 569376768, 569385328, 569385456, 569385524, 569385584, 569385652, 569385712, 569385780, 569409536, 569418096, 569418224, 569418292, 569418352, 569418420, 569418480, 569418548, 573833216, 573841776, 573841904, 573841972, 573842032, 573842100, 573842160, 573842228, 573865984, 573874544, 573874672, 573874740, 573874800, 573874868, 573874928, 573874996, 577765376, 577773936, 577774064, 577774132, 577774192, 577774260, 577774320, 577774388, 577798144, 577806704, 577806832, 577806900, 577806960, 577807028, 577807088, 577807156, 582221824, 582230384, 582230512, 582230580, 582230640, 582230708, 582230768, 582230836, 582254592, 582263152, 582263280, 582263348, 582263408, 582263476, 582263536, 582263604, 586153984, 586162544, 586162672, 586162740, 586162800, 586162868, 586162928, 586162996, 586186752, 586195312, 586195440, 586195508, 586195568, 586195636, 586195696, 586195764, 590610432, 590618992, 590619120, 590619188, 590619248, 590619316, 590619376, 590619444, 590643200, 590651760, 590651888, 590651956, 590652016, 590652084, 590652144, 590652212, -2147483648, -2147475088, -2147474960, -2147474892, -2147474832, -2147474764, -2147474704, -2147474636, -2147450880, -2147442320, -2147442192, -2147442124, -2147442064, -2147441996, -2147441936, -2147441868, -1586495488, -1586486928, -1586486800, -1586486732, -1586486672, -1586486604, -1586486544, -1586486476, -1586462720, -1586454160, -1586454032, -1586453964, -1586453904, -1586453836, -1586453776, -1586453708, -1578106880, -1578098320, -1578098192, -1578098124, -1578098064, -1578097996, -1578097936, -1578097868, -1578074112, -1578065552, -1578065424, -1578065356, -1578065296, -1578065228, -1578065168, -1578065100, -1573650432, -1573641872, -1573641744, -1573641676, -1573641616, -1573641548, -1573641488, -1573641420, -1573617664, -1573609104, -1573608976, -1573608908, -1573608848, -1573608780, -1573608720, -1573608652, -1569718272, -1569709712, -1569709584, -1569709516, -1569709456, -1569709388, -1569709328, -1569709260, -1569685504, -1569676944, -1569676816, -1569676748, -1569676688, -1569676620, -1569676560, -1569676492, -1565261824, -1565253264, -1565253136, -1565253068, -1565253008, -1565252940, -1565252880, -1565252812, -1565229056, -1565220496, -1565220368, -1565220300, -1565220240, -1565220172, -1565220112, -1565220044, -1561329664, -1561321104, -1561320976, -1561320908, -1561320848, -1561320780, -1561320720, -1561320652, -1561296896, -1561288336, -1561288208, -1561288140, -1561288080, -1561288012, -1561287952, -1561287884, -1556873216, -1556864656, -1556864528, -1556864460, -1556864400, -1556864332, -1556864272, -1556864204, -1556840448, -1556831888, -1556831760, -1556831692, -1556831632, -1556831564, -1556831504, -1556831436, 0, 8576, 8704, 8768, 8832, 8896, 8960, 9024, 32768, 41344, 41472, 41536, 41600, 41664, 41728, 41792, 562036736, 562045312, 562045440, 562045504, 562045568, 562045632, 562045696, 562045760, 562069504, 562078080, 562078208, 562078272, 562078336, 562078400, 562078464, 562078528, 570425344, 570433920, 570434048, 570434112, 570434176, 570434240, 570434304, 570434368, 570458112, 570466688, 570466816, 570466880, 570466944, 570467008, 570467072, 570467136, 574619648, 574628224, 574628352, 574628416, 574628480, 574628544, 574628608, 574628672, 574652416, 574660992, 574661120, 574661184, 574661248, 574661312, 574661376, 574661440, 578813952, 578822528, 578822656, 578822720, 578822784, 578822848, 578822912, 578822976, 578846720, 578855296, 578855424, 578855488, 578855552, 578855616, 578855680, 578855744, 583008256, 583016832, 583016960, 583017024, 583017088, 583017152, 583017216, 583017280, 583041024, 583049600, 583049728, 583049792, 583049856, 583049920, 583049984, 583050048, 587202560, 587211136, 587211264, 587211328, 587211392, 587211456, 587211520, 587211584, 587235328, 587243904, 587244032, 587244096, 587244160, 587244224, 587244288, 587244352, 591396864, 591405440, 591405568, 591405632, 591405696, 591405760, 591405824, 591405888, 591429632, 591438208, 591438336, 591438400, 591438464, 591438528, 591438592, 591438656, -2147483648, -2147475072, -2147474944, -2147474880, -2147474816, -2147474752, -2147474688, -2147474624, -2147450880, -2147442304, -2147442176, -2147442112, -2147442048, -2147441984, -2147441920, -2147441856, -1585446912, -1585438336, -1585438208, -1585438144, -1585438080, -1585438016, -1585437952, -1585437888, -1585414144, -1585405568, -1585405440, -1585405376, -1585405312, -1585405248, -1585405184, -1585405120, -1577058304, -1577049728, -1577049600, -1577049536, -1577049472, -1577049408, -1577049344, -1577049280, -1577025536, -1577016960, -1577016832, -1577016768, -1577016704, -1577016640, -1577016576, -1577016512, -1572864000, -1572855424, -1572855296, -1572855232, -1572855168, -1572855104, -1572855040, -1572854976, -1572831232, -1572822656, -1572822528, -1572822464, -1572822400, -1572822336, -1572822272, -1572822208, -1568669696, -1568661120, -1568660992, -1568660928, -1568660864, -1568660800, -1568660736, -1568660672, -1568636928, -1568628352, -1568628224, -1568628160, -1568628096, -1568628032, -1568627968, -1568627904, -1564475392, -1564466816, -1564466688, -1564466624, -1564466560, -1564466496, -1564466432, -1564466368, -1564442624, -1564434048, -1564433920, -1564433856, -1564433792, -1564433728, -1564433664, -1564433600, -1560281088, -1560272512, -1560272384, -1560272320, -1560272256, -1560272192, -1560272128, -1560272064, -1560248320, -1560239744, -1560239616, -1560239552, -1560239488, -1560239424, -1560239360, -1560239296, -1556086784, -1556078208, -1556078080, -1556078016, -1556077952, -1556077888, -1556077824, -1556077760, -1556054016, -1556045440, -1556045312, -1556045248, -1556045184, -1556045120, -1556045056, -1556044992, 0, 8592, 8720, 8792, 8848, 8920, 8976, 9048, 32768, 41360, 41488, 41560, 41616, 41688, 41744, 41816, 563085312, 563093904, 563094032, 563094104, 563094160, 563094232, 563094288, 563094360, 563118080, 563126672, 563126800, 563126872, 563126928, 563127000, 563127056, 563127128, 571473920, 571482512, 571482640, 571482712, 571482768, 571482840, 571482896, 571482968, 571506688, 571515280, 571515408, 571515480, 571515536, 571515608, 571515664, 571515736, 576192512, 576201104, 576201232, 576201304, 576201360, 576201432, 576201488, 576201560, 576225280, 576233872, 576234000, 576234072, 576234128, 576234200, 576234256, 576234328, 579862528, 579871120, 579871248, 579871320, 579871376, 579871448, 579871504, 579871576, 579895296, 579903888, 579904016, 579904088, 579904144, 579904216, 579904272, 579904344, 584581120, 584589712, 584589840, 584589912, 584589968, 584590040, 584590096, 584590168, 584613888, 584622480, 584622608, 584622680, 584622736, 584622808, 584622864, 584622936, 588251136, 588259728, 588259856, 588259928, 588259984, 588260056, 588260112, 588260184, 588283904, 588292496, 588292624, 588292696, 588292752, 588292824, 588292880, 588292952, 592969728, 592978320, 592978448, 592978520, 592978576, 592978648, 592978704, 592978776, 593002496, 593011088, 593011216, 593011288, 593011344, 593011416, 593011472, 593011544, -2147483648, -2147475056, -2147474928, -2147474856, -2147474800, -2147474728, -2147474672, -2147474600, -2147450880, -2147442288, -2147442160, -2147442088, -2147442032, -2147441960, -2147441904, -2147441832, -1584398336, -1584389744, -1584389616, -1584389544, -1584389488, -1584389416, -1584389360, -1584389288, -1584365568, -1584356976, -1584356848, -1584356776, -1584356720, -1584356648, -1584356592, -1584356520, -1576009728, -1576001136, -1576001008, -1576000936, -1576000880, -1576000808, -1576000752, -1576000680, -1575976960, -1575968368, -1575968240, -1575968168, -1575968112, -1575968040, -1575967984, -1575967912, -1571291136, -1571282544, -1571282416, -1571282344, -1571282288, -1571282216, -1571282160, -1571282088, -1571258368, -1571249776, -1571249648, -1571249576, -1571249520, -1571249448, -1571249392, -1571249320, -1567621120, -1567612528, -1567612400, -1567612328, -1567612272, -1567612200, -1567612144, -1567612072, -1567588352, -1567579760, -1567579632, -1567579560, -1567579504, -1567579432, -1567579376, -1567579304, -1562902528, -1562893936, -1562893808, -1562893736, -1562893680, -1562893608, -1562893552, -1562893480, -1562869760, -1562861168, -1562861040, -1562860968, -1562860912, -1562860840, -1562860784, -1562860712, -1559232512, -1559223920, -1559223792, -1559223720, -1559223664, -1559223592, -1559223536, -1559223464, -1559199744, -1559191152, -1559191024, -1559190952, -1559190896, -1559190824, -1559190768, -1559190696, -1554513920, -1554505328, -1554505200, -1554505128, -1554505072, -1554505000, -1554504944, -1554504872, -1554481152, -1554472560, -1554472432, -1554472360, -1554472304, -1554472232, -1554472176, -1554472104, 0, 8608, 8736, 8816, 8864, 8944, 8992, 9072, 32768, 41376, 41504, 41584, 41632, 41712, 41760, 41840, 564133888, 564142496, 564142624, 564142704, 564142752, 564142832, 564142880, 564142960, 564166656, 564175264, 564175392, 564175472, 564175520, 564175600, 564175648, 564175728, 572522496, 572531104, 572531232, 572531312, 572531360, 572531440, 572531488, 572531568, 572555264, 572563872, 572564000, 572564080, 572564128, 572564208, 572564256, 572564336, 577765376, 577773984, 577774112, 577774192, 577774240, 577774320, 577774368, 577774448, 577798144, 577806752, 577806880, 577806960, 577807008, 577807088, 577807136, 577807216, 580911104, 580919712, 580919840, 580919920, 580919968, 580920048, 580920096, 580920176, 580943872, 580952480, 580952608, 580952688, 580952736, 580952816, 580952864, 580952944, 586153984, 586162592, 586162720, 586162800, 586162848, 586162928, 586162976, 586163056, 586186752, 586195360, 586195488, 586195568, 586195616, 586195696, 586195744, 586195824, 589299712, 589308320, 589308448, 589308528, 589308576, 589308656, 589308704, 589308784, 589332480, 589341088, 589341216, 589341296, 589341344, 589341424, 589341472, 589341552, 594542592, 594551200, 594551328, 594551408, 594551456, 594551536, 594551584, 594551664, 594575360, 594583968, 594584096, 594584176, 594584224, 594584304, 594584352, 594584432, -2147483648, -2147475040, -2147474912, -2147474832, -2147474784, -2147474704, -2147474656, -2147474576, -2147450880, -2147442272, -2147442144, -2147442064, -2147442016, -2147441936, -2147441888, -2147441808, -1583349760, -1583341152, -1583341024, -1583340944, -1583340896, -1583340816, -1583340768, -1583340688, -1583316992, -1583308384, -1583308256, -1583308176, -1583308128, -1583308048, -1583308000, -1583307920, -1574961152, -1574952544, -1574952416, -1574952336, -1574952288, -1574952208, -1574952160, -1574952080, -1574928384, -1574919776, -1574919648, -1574919568, -1574919520, -1574919440, -1574919392, -1574919312, -1569718272, -1569709664, -1569709536, -1569709456, -1569709408, -1569709328, -1569709280, -1569709200, -1569685504, -1569676896, -1569676768, -1569676688, -1569676640, -1569676560, -1569676512, -1569676432, -1566572544, -1566563936, -1566563808, -1566563728, -1566563680, -1566563600, -1566563552, -1566563472, -1566539776, -1566531168, -1566531040, -1566530960, -1566530912, -1566530832, -1566530784, -1566530704, -1561329664, -1561321056, -1561320928, -1561320848, -1561320800, -1561320720, -1561320672, -1561320592, -1561296896, -1561288288, -1561288160, -1561288080, -1561288032, -1561287952, -1561287904, -1561287824, -1558183936, -1558175328, -1558175200, -1558175120, -1558175072, -1558174992, -1558174944, -1558174864, -1558151168, -1558142560, -1558142432, -1558142352, -1558142304, -1558142224, -1558142176, -1558142096, -1552941056, -1552932448, -1552932320, -1552932240, -1552932192, -1552932112, -1552932064, -1552931984, -1552908288, -1552899680, -1552899552, -1552899472, -1552899424, -1552899344, -1552899296, -1552899216, 0, 8624, 8752, 8836, 8880, 8964, 9008, 9092, 32768, 41392, 41520, 41604, 41648, 41732, 41776, 41860, 565182464, 565191088, 565191216, 565191300, 565191344, 565191428, 565191472, 565191556, 565215232, 565223856, 565223984, 565224068, 565224112, 565224196, 565224240, 565224324, 573571072, 573579696, 573579824, 573579908, 573579952, 573580036, 573580080, 573580164, 573603840, 573612464, 573612592, 573612676, 573612720, 573612804, 573612848, 573612932, 579076096, 579084720, 579084848, 579084932, 579084976, 579085060, 579085104, 579085188, 579108864, 579117488, 579117616, 579117700, 579117744, 579117828, 579117872, 579117956, 581959680, 581968304, 581968432, 581968516, 581968560, 581968644, 581968688, 581968772, 581992448, 582001072, 582001200, 582001284, 582001328, 582001412, 582001456, 582001540, 587464704, 587473328, 587473456, 587473540, 587473584, 587473668, 587473712, 587473796, 587497472, 587506096, 587506224, 587506308, 587506352, 587506436, 587506480, 587506564, 590348288, 590356912, 590357040, 590357124, 590357168, 590357252, 590357296, 590357380, 590381056, 590389680, 590389808, 590389892, 590389936, 590390020, 590390064, 590390148, 595853312, 595861936, 595862064, 595862148, 595862192, 595862276, 595862320, 595862404, 595886080, 595894704, 595894832, 595894916, 595894960, 595895044, 595895088, 595895172, -2147483648, -2147475024, -2147474896, -2147474812, -2147474768, -2147474684, -2147474640, -2147474556, -2147450880, -2147442256, -2147442128, -2147442044, -2147442000, -2147441916, -2147441872, -2147441788, -1582301184, -1582292560, -1582292432, -1582292348, -1582292304, -1582292220, -1582292176, -1582292092, -1582268416, -1582259792, -1582259664, -1582259580, -1582259536, -1582259452, -1582259408, -1582259324, -1573912576, -1573903952, -1573903824, -1573903740, -1573903696, -1573903612, -1573903568, -1573903484, -1573879808, -1573871184, -1573871056, -1573870972, -1573870928, -1573870844, -1573870800, -1573870716, -1568407552, -1568398928, -1568398800, -1568398716, -1568398672, -1568398588, -1568398544, -1568398460, -1568374784, -1568366160, -1568366032, -1568365948, -1568365904, -1568365820, -1568365776, -1568365692, -1565523968, -1565515344, -1565515216, -1565515132, -1565515088, -1565515004, -1565514960, -1565514876, -1565491200, -1565482576, -1565482448, -1565482364, -1565482320, -1565482236, -1565482192, -1565482108, -1560018944, -1560010320, -1560010192, -1560010108, -1560010064, -1560009980, -1560009936, -1560009852, -1559986176, -1559977552, -1559977424, -1559977340, -1559977296, -1559977212, -1559977168, -1559977084, -1557135360, -1557126736, -1557126608, -1557126524, -1557126480, -1557126396, -1557126352, -1557126268, -1557102592, -1557093968, -1557093840, -1557093756, -1557093712, -1557093628, -1557093584, -1557093500, -1551630336, -1551621712, -1551621584, -1551621500, -1551621456, -1551621372, -1551621328, -1551621244, -1551597568, -1551588944, -1551588816, -1551588732, -1551588688, -1551588604, -1551588560, -1551588476, 0, 8640, 8768, 8848, 8896, 8976, 9024, 9104, 32768, 41408, 41536, 41616, 41664, 41744, 41792, 41872, 566231040, 566239680, 566239808, 566239888, 566239936, 566240016, 566240064, 566240144, 566263808, 566272448, 566272576, 566272656, 566272704, 566272784, 566272832, 566272912, 574619648, 574628288, 574628416, 574628496, 574628544, 574628624, 574628672, 574628752, 574652416, 574661056, 574661184, 574661264, 574661312, 574661392, 574661440, 574661520, 579862528, 579871168, 579871296, 579871376, 579871424, 579871504, 579871552, 579871632, 579895296, 579903936, 579904064, 579904144, 579904192, 579904272, 579904320, 579904400, 583008256, 583016896, 583017024, 583017104, 583017152, 583017232, 583017280, 583017360, 583041024, 583049664, 583049792, 583049872, 583049920, 583050000, 583050048, 583050128, 588251136, 588259776, 588259904, 588259984, 588260032, 588260112, 588260160, 588260240, 588283904, 588292544, 588292672, 588292752, 588292800, 588292880, 588292928, 588293008, 591396864, 591405504, 591405632, 591405712, 591405760, 591405840, 591405888, 591405968, 591429632, 591438272, 591438400, 591438480, 591438528, 591438608, 591438656, 591438736, 596639744, 596648384, 596648512, 596648592, 596648640, 596648720, 596648768, 596648848, 596672512, 596681152, 596681280, 596681360, 596681408, 596681488, 596681536, 596681616, -2147483648, -2147475008, -2147474880, -2147474800, -2147474752, -2147474672, -2147474624, -2147474544, -2147450880, -2147442240, -2147442112, -2147442032, -2147441984, -2147441904, -2147441856, -2147441776, -1581252608, -1581243968, -1581243840, -1581243760, -1581243712, -1581243632, -1581243584, -1581243504, -1581219840, -1581211200, -1581211072, -1581210992, -1581210944, -1581210864, -1581210816, -1581210736, -1572864000, -1572855360, -1572855232, -1572855152, -1572855104, -1572855024, -1572854976, -1572854896, -1572831232, -1572822592, -1572822464, -1572822384, -1572822336, -1572822256, -1572822208, -1572822128, -1567621120, -1567612480, -1567612352, -1567612272, -1567612224, -1567612144, -1567612096, -1567612016, -1567588352, -1567579712, -1567579584, -1567579504, -1567579456, -1567579376, -1567579328, -1567579248, -1564475392, -1564466752, -1564466624, -1564466544, -1564466496, -1564466416, -1564466368, -1564466288, -1564442624, -1564433984, -1564433856, -1564433776, -1564433728, -1564433648, -1564433600, -1564433520, -1559232512, -1559223872, -1559223744, -1559223664, -1559223616, -1559223536, -1559223488, -1559223408, -1559199744, -1559191104, -1559190976, -1559190896, -1559190848, -1559190768, -1559190720, -1559190640, -1556086784, -1556078144, -1556078016, -1556077936, -1556077888, -1556077808, -1556077760, -1556077680, -1556054016, -1556045376, -1556045248, -1556045168, -1556045120, -1556045040, -1556044992, -1556044912, -1550843904, -1550835264, -1550835136, -1550835056, -1550835008, -1550834928, -1550834880, -1550834800, -1550811136, -1550802496, -1550802368, -1550802288, -1550802240, -1550802160, -1550802112, -1550802032, 0, 8656, 8784, 8860, 8912, 8988, 9040, 9116, 32768, 41424, 41552, 41628, 41680, 41756, 41808, 41884, 567279616, 567288272, 567288400, 567288476, 567288528, 567288604, 567288656, 567288732, 567312384, 567321040, 567321168, 567321244, 567321296, 567321372, 567321424, 567321500, 575668224, 575676880, 575677008, 575677084, 575677136, 575677212, 575677264, 575677340, 575700992, 575709648, 575709776, 575709852, 575709904, 575709980, 575710032, 575710108, 580648960, 580657616, 580657744, 580657820, 580657872, 580657948, 580658000, 580658076, 580681728, 580690384, 580690512, 580690588, 580690640, 580690716, 580690768, 580690844, 584056832, 584065488, 584065616, 584065692, 584065744, 584065820, 584065872, 584065948, 584089600, 584098256, 584098384, 584098460, 584098512, 584098588, 584098640, 584098716, 589037568, 589046224, 589046352, 589046428, 589046480, 589046556, 589046608, 589046684, 589070336, 589078992, 589079120, 589079196, 589079248, 589079324, 589079376, 589079452, 592445440, 592454096, 592454224, 592454300, 592454352, 592454428, 592454480, 592454556, 592478208, 592486864, 592486992, 592487068, 592487120, 592487196, 592487248, 592487324, 597426176, 597434832, 597434960, 597435036, 597435088, 597435164, 597435216, 597435292, 597458944, 597467600, 597467728, 597467804, 597467856, 597467932, 597467984, 597468060, -2147483648, -2147474992, -2147474864, -2147474788, -2147474736, -2147474660, -2147474608, -2147474532, -2147450880, -2147442224, -2147442096, -2147442020, -2147441968, -2147441892, -2147441840, -2147441764, -1580204032, -1580195376, -1580195248, -1580195172, -1580195120, -1580195044, -1580194992, -1580194916, -1580171264, -1580162608, -1580162480, -1580162404, -1580162352, -1580162276, -1580162224, -1580162148, -1571815424, -1571806768, -1571806640, -1571806564, -1571806512, -1571806436, -1571806384, -1571806308, -1571782656, -1571774000, -1571773872, -1571773796, -1571773744, -1571773668, -1571773616, -1571773540, -1566834688, -1566826032, -1566825904, -1566825828, -1566825776, -1566825700, -1566825648, -1566825572, -1566801920, -1566793264, -1566793136, -1566793060, -1566793008, -1566792932, -1566792880, -1566792804, -1563426816, -1563418160, -1563418032, -1563417956, -1563417904, -1563417828, -1563417776, -1563417700, -1563394048, -1563385392, -1563385264, -1563385188, -1563385136, -1563385060, -1563385008, -1563384932, -1558446080, -1558437424, -1558437296, -1558437220, -1558437168, -1558437092, -1558437040, -1558436964, -1558413312, -1558404656, -1558404528, -1558404452, -1558404400, -1558404324, -1558404272, -1558404196, -1555038208, -1555029552, -1555029424, -1555029348, -1555029296, -1555029220, -1555029168, -1555029092, -1555005440, -1554996784, -1554996656, -1554996580, -1554996528, -1554996452, -1554996400, -1554996324, -1550057472, -1550048816, -1550048688, -1550048612, -1550048560, -1550048484, -1550048432, -1550048356, -1550024704, -1550016048, -1550015920, -1550015844, -1550015792, -1550015716, -1550015664, -1550015588, 0, 8672, 8800, 8872, 8928, 9000, 9056, 9128, 32768, 41440, 41568, 41640, 41696, 41768, 41824, 41896, 568328192, 568336864, 568336992, 568337064, 568337120, 568337192, 568337248, 568337320, 568360960, 568369632, 568369760, 568369832, 568369888, 568369960, 568370016, 568370088, 576716800, 576725472, 576725600, 576725672, 576725728, 576725800, 576725856, 576725928, 576749568, 576758240, 576758368, 576758440, 576758496, 576758568, 576758624, 576758696, 581435392, 581444064, 581444192, 581444264, 581444320, 581444392, 581444448, 581444520, 581468160, 581476832, 581476960, 581477032, 581477088, 581477160, 581477216, 581477288, 585105408, 585114080, 585114208, 585114280, 585114336, 585114408, 585114464, 585114536, 585138176, 585146848, 585146976, 585147048, 585147104, 585147176, 585147232, 585147304, 589824000, 589832672, 589832800, 589832872, 589832928, 589833000, 589833056, 589833128, 589856768, 589865440, 589865568, 589865640, 589865696, 589865768, 589865824, 589865896, 593494016, 593502688, 593502816, 593502888, 593502944, 593503016, 593503072, 593503144, 593526784, 593535456, 593535584, 593535656, 593535712, 593535784, 593535840, 593535912, 598212608, 598221280, 598221408, 598221480, 598221536, 598221608, 598221664, 598221736, 598245376, 598254048, 598254176, 598254248, 598254304, 598254376, 598254432, 598254504, -2147483648, -2147474976, -2147474848, -2147474776, -2147474720, -2147474648, -2147474592, -2147474520, -2147450880, -2147442208, -2147442080, -2147442008, -2147441952, -2147441880, -2147441824, -2147441752, -1579155456, -1579146784, -1579146656, -1579146584, -1579146528, -1579146456, -1579146400, -1579146328, -1579122688, -1579114016, -1579113888, -1579113816, -1579113760, -1579113688, -1579113632, -1579113560, -1570766848, -1570758176, -1570758048, -1570757976, -1570757920, -1570757848, -1570757792, -1570757720, -1570734080, -1570725408, -1570725280, -1570725208, -1570725152, -1570725080, -1570725024, -1570724952, -1566048256, -1566039584, -1566039456, -1566039384, -1566039328, -1566039256, -1566039200, -1566039128, -1566015488, -1566006816, -1566006688, -1566006616, -1566006560, -1566006488, -1566006432, -1566006360, -1562378240, -1562369568, -1562369440, -1562369368, -1562369312, -1562369240, -1562369184, -1562369112, -1562345472, -1562336800, -1562336672, -1562336600, -1562336544, -1562336472, -1562336416, -1562336344, -1557659648, -1557650976, -1557650848, -1557650776, -1557650720, -1557650648, -1557650592, -1557650520, -1557626880, -1557618208, -1557618080, -1557618008, -1557617952, -1557617880, -1557617824, -1557617752, -1553989632, -1553980960, -1553980832, -1553980760, -1553980704, -1553980632, -1553980576, -1553980504, -1553956864, -1553948192, -1553948064, -1553947992, -1553947936, -1553947864, -1553947808, -1553947736, -1549271040, -1549262368, -1549262240, -1549262168, -1549262112, -1549262040, -1549261984, -1549261912, -1549238272, -1549229600, -1549229472, -1549229400, -1549229344, -1549229272, -1549229216, -1549229144, 0, 8688, 8816, 8884, 8944, 9012, 9072, 9140, 32768, 41456, 41584, 41652, 41712, 41780, 41840, 41908, 569376768, 569385456, 569385584, 569385652, 569385712, 569385780, 569385840, 569385908, 569409536, 569418224, 569418352, 569418420, 569418480, 569418548, 569418608, 569418676, 577765376, 577774064, 577774192, 577774260, 577774320, 577774388, 577774448, 577774516, 577798144, 577806832, 577806960, 577807028, 577807088, 577807156, 577807216, 577807284, 582221824, 582230512, 582230640, 582230708, 582230768, 582230836, 582230896, 582230964, 582254592, 582263280, 582263408, 582263476, 582263536, 582263604, 582263664, 582263732, 586153984, 586162672, 586162800, 586162868, 586162928, 586162996, 586163056, 586163124, 586186752, 586195440, 586195568, 586195636, 586195696, 586195764, 586195824, 586195892, 590610432, 590619120, 590619248, 590619316, 590619376, 590619444, 590619504, 590619572, 590643200, 590651888, 590652016, 590652084, 590652144, 590652212, 590652272, 590652340, 594542592, 594551280, 594551408, 594551476, 594551536, 594551604, 594551664, 594551732, 594575360, 594584048, 594584176, 594584244, 594584304, 594584372, 594584432, 594584500, 598999040, 599007728, 599007856, 599007924, 599007984, 599008052, 599008112, 599008180, 599031808, 599040496, 599040624, 599040692, 599040752, 599040820, 599040880, 599040948, -2147483648, -2147474960, -2147474832, -2147474764, -2147474704, -2147474636, -2147474576, -2147474508, -2147450880, -2147442192, -2147442064, -2147441996, -2147441936, -2147441868, -2147441808, -2147441740, -1578106880, -1578098192, -1578098064, -1578097996, -1578097936, -1578097868, -1578097808, -1578097740, -1578074112, -1578065424, -1578065296, -1578065228, -1578065168, -1578065100, -1578065040, -1578064972, -1569718272, -1569709584, -1569709456, -1569709388, -1569709328, -1569709260, -1569709200, -1569709132, -1569685504, -1569676816, -1569676688, -1569676620, -1569676560, -1569676492, -1569676432, -1569676364, -1565261824, -1565253136, -1565253008, -1565252940, -1565252880, -1565252812, -1565252752, -1565252684, -1565229056, -1565220368, -1565220240, -1565220172, -1565220112, -1565220044, -1565219984, -1565219916, -1561329664, -1561320976, -1561320848, -1561320780, -1561320720, -1561320652, -1561320592, -1561320524, -1561296896, -1561288208, -1561288080, -1561288012, -1561287952, -1561287884, -1561287824, -1561287756, -1556873216, -1556864528, -1556864400, -1556864332, -1556864272, -1556864204, -1556864144, -1556864076, -1556840448, -1556831760, -1556831632, -1556831564, -1556831504, -1556831436, -1556831376, -1556831308, -1552941056, -1552932368, -1552932240, -1552932172, -1552932112, -1552932044, -1552931984, -1552931916, -1552908288, -1552899600, -1552899472, -1552899404, -1552899344, -1552899276, -1552899216, -1552899148, -1548484608, -1548475920, -1548475792, -1548475724, -1548475664, -1548475596, -1548475536, -1548475468, -1548451840, -1548443152, -1548443024, -1548442956, -1548442896, -1548442828, -1548442768, -1548442700, 0, 8704, 8832, 8896, 8960, 9024, 9088, 9152, 32768, 41472, 41600, 41664, 41728, 41792, 41856, 41920, 570425344, 570434048, 570434176, 570434240, 570434304, 570434368, 570434432, 570434496, 570458112, 570466816, 570466944, 570467008, 570467072, 570467136, 570467200, 570467264, 578813952, 578822656, 578822784, 578822848, 578822912, 578822976, 578823040, 578823104, 578846720, 578855424, 578855552, 578855616, 578855680, 578855744, 578855808, 578855872, 583008256, 583016960, 583017088, 583017152, 583017216, 583017280, 583017344, 583017408, 583041024, 583049728, 583049856, 583049920, 583049984, 583050048, 583050112, 583050176, 587202560, 587211264, 587211392, 587211456, 587211520, 587211584, 587211648, 587211712, 587235328, 587244032, 587244160, 587244224, 587244288, 587244352, 587244416, 587244480, 591396864, 591405568, 591405696, 591405760, 591405824, 591405888, 591405952, 591406016, 591429632, 591438336, 591438464, 591438528, 591438592, 591438656, 591438720, 591438784, 595591168, 595599872, 595600000, 595600064, 595600128, 595600192, 595600256, 595600320, 595623936, 595632640, 595632768, 595632832, 595632896, 595632960, 595633024, 595633088, 599785472, 599794176, 599794304, 599794368, 599794432, 599794496, 599794560, 599794624, 599818240, 599826944, 599827072, 599827136, 599827200, 599827264, 599827328, 599827392, -2147483648, -2147474944, -2147474816, -2147474752, -2147474688, -2147474624, -2147474560, -2147474496, -2147450880, -2147442176, -2147442048, -2147441984, -2147441920, -2147441856, -2147441792, -2147441728, -1577058304, -1577049600, -1577049472, -1577049408, -1577049344, -1577049280, -1577049216, -1577049152, -1577025536, -1577016832, -1577016704, -1577016640, -1577016576, -1577016512, -1577016448, -1577016384, -1568669696, -1568660992, -1568660864, -1568660800, -1568660736, -1568660672, -1568660608, -1568660544, -1568636928, -1568628224, -1568628096, -1568628032, -1568627968, -1568627904, -1568627840, -1568627776, -1564475392, -1564466688, -1564466560, -1564466496, -1564466432, -1564466368, -1564466304, -1564466240, -1564442624, -1564433920, -1564433792, -1564433728, -1564433664, -1564433600, -1564433536, -1564433472, -1560281088, -1560272384, -1560272256, -1560272192, -1560272128, -1560272064, -1560272000, -1560271936, -1560248320, -1560239616, -1560239488, -1560239424, -1560239360, -1560239296, -1560239232, -1560239168, -1556086784, -1556078080, -1556077952, -1556077888, -1556077824, -1556077760, -1556077696, -1556077632, -1556054016, -1556045312, -1556045184, -1556045120, -1556045056, -1556044992, -1556044928, -1556044864, -1551892480, -1551883776, -1551883648, -1551883584, -1551883520, -1551883456, -1551883392, -1551883328, -1551859712, -1551851008, -1551850880, -1551850816, -1551850752, -1551850688, -1551850624, -1551850560, -1547698176, -1547689472, -1547689344, -1547689280, -1547689216, -1547689152, -1547689088, -1547689024, -1547665408, -1547656704, -1547656576, -1547656512, -1547656448, -1547656384, -1547656320, -1547656256, 0, 8720, 8848, 8920, 8976, 9048, 9104, 9176, 32768, 41488, 41616, 41688, 41744, 41816, 41872, 41944, 571473920, 571482640, 571482768, 571482840, 571482896, 571482968, 571483024, 571483096, 571506688, 571515408, 571515536, 571515608, 571515664, 571515736, 571515792, 571515864, 579862528, 579871248, 579871376, 579871448, 579871504, 579871576, 579871632, 579871704, 579895296, 579904016, 579904144, 579904216, 579904272, 579904344, 579904400, 579904472, 584581120, 584589840, 584589968, 584590040, 584590096, 584590168, 584590224, 584590296, 584613888, 584622608, 584622736, 584622808, 584622864, 584622936, 584622992, 584623064, 588251136, 588259856, 588259984, 588260056, 588260112, 588260184, 588260240, 588260312, 588283904, 588292624, 588292752, 588292824, 588292880, 588292952, 588293008, 588293080, 592969728, 592978448, 592978576, 592978648, 592978704, 592978776, 592978832, 592978904, 593002496, 593011216, 593011344, 593011416, 593011472, 593011544, 593011600, 593011672, 596639744, 596648464, 596648592, 596648664, 596648720, 596648792, 596648848, 596648920, 596672512, 596681232, 596681360, 596681432, 596681488, 596681560, 596681616, 596681688, 601358336, 601367056, 601367184, 601367256, 601367312, 601367384, 601367440, 601367512, 601391104, 601399824, 601399952, 601400024, 601400080, 601400152, 601400208, 601400280, -2147483648, -2147474928, -2147474800, -2147474728, -2147474672, -2147474600, -2147474544, -2147474472, -2147450880, -2147442160, -2147442032, -2147441960, -2147441904, -2147441832, -2147441776, -2147441704, -1576009728, -1576001008, -1576000880, -1576000808, -1576000752, -1576000680, -1576000624, -1576000552, -1575976960, -1575968240, -1575968112, -1575968040, -1575967984, -1575967912, -1575967856, -1575967784, -1567621120, -1567612400, -1567612272, -1567612200, -1567612144, -1567612072, -1567612016, -1567611944, -1567588352, -1567579632, -1567579504, -1567579432, -1567579376, -1567579304, -1567579248, -1567579176, -1562902528, -1562893808, -1562893680, -1562893608, -1562893552, -1562893480, -1562893424, -1562893352, -1562869760, -1562861040, -1562860912, -1562860840, -1562860784, -1562860712, -1562860656, -1562860584, -1559232512, -1559223792, -1559223664, -1559223592, -1559223536, -1559223464, -1559223408, -1559223336, -1559199744, -1559191024, -1559190896, -1559190824, -1559190768, -1559190696, -1559190640, -1559190568, -1554513920, -1554505200, -1554505072, -1554505000, -1554504944, -1554504872, -1554504816, -1554504744, -1554481152, -1554472432, -1554472304, -1554472232, -1554472176, -1554472104, -1554472048, -1554471976, -1550843904, -1550835184, -1550835056, -1550834984, -1550834928, -1550834856, -1550834800, -1550834728, -1550811136, -1550802416, -1550802288, -1550802216, -1550802160, -1550802088, -1550802032, -1550801960, -1546125312, -1546116592, -1546116464, -1546116392, -1546116336, -1546116264, -1546116208, -1546116136, -1546092544, -1546083824, -1546083696, -1546083624, -1546083568, -1546083496, -1546083440, -1546083368, 0, 8736, 8864, 8944, 8992, 9072, 9120, 9200, 32768, 41504, 41632, 41712, 41760, 41840, 41888, 41968, 572522496, 572531232, 572531360, 572531440, 572531488, 572531568, 572531616, 572531696, 572555264, 572564000, 572564128, 572564208, 572564256, 572564336, 572564384, 572564464, 580911104, 580919840, 580919968, 580920048, 580920096, 580920176, 580920224, 580920304, 580943872, 580952608, 580952736, 580952816, 580952864, 580952944, 580952992, 580953072, 586153984, 586162720, 586162848, 586162928, 586162976, 586163056, 586163104, 586163184, 586186752, 586195488, 586195616, 586195696, 586195744, 586195824, 586195872, 586195952, 589299712, 589308448, 589308576, 589308656, 589308704, 589308784, 589308832, 589308912, 589332480, 589341216, 589341344, 589341424, 589341472, 589341552, 589341600, 589341680, 594542592, 594551328, 594551456, 594551536, 594551584, 594551664, 594551712, 594551792, 594575360, 594584096, 594584224, 594584304, 594584352, 594584432, 594584480, 594584560, 597688320, 597697056, 597697184, 597697264, 597697312, 597697392, 597697440, 597697520, 597721088, 597729824, 597729952, 597730032, 597730080, 597730160, 597730208, 597730288, 602931200, 602939936, 602940064, 602940144, 602940192, 602940272, 602940320, 602940400, 602963968, 602972704, 602972832, 602972912, 602972960, 602973040, 602973088, 602973168, -2147483648, -2147474912, -2147474784, -2147474704, -2147474656, -2147474576, -2147474528, -2147474448, -2147450880, -2147442144, -2147442016, -2147441936, -2147441888, -2147441808, -2147441760, -2147441680, -1574961152, -1574952416, -1574952288, -1574952208, -1574952160, -1574952080, -1574952032, -1574951952, -1574928384, -1574919648, -1574919520, -1574919440, -1574919392, -1574919312, -1574919264, -1574919184, -1566572544, -1566563808, -1566563680, -1566563600, -1566563552, -1566563472, -1566563424, -1566563344, -1566539776, -1566531040, -1566530912, -1566530832, -1566530784, -1566530704, -1566530656, -1566530576, -1561329664, -1561320928, -1561320800, -1561320720, -1561320672, -1561320592, -1561320544, -1561320464, -1561296896, -1561288160, -1561288032, -1561287952, -1561287904, -1561287824, -1561287776, -1561287696, -1558183936, -1558175200, -1558175072, -1558174992, -1558174944, -1558174864, -1558174816, -1558174736, -1558151168, -1558142432, -1558142304, -1558142224, -1558142176, -1558142096, -1558142048, -1558141968, -1552941056, -1552932320, -1552932192, -1552932112, -1552932064, -1552931984, -1552931936, -1552931856, -1552908288, -1552899552, -1552899424, -1552899344, -1552899296, -1552899216, -1552899168, -1552899088, -1549795328, -1549786592, -1549786464, -1549786384, -1549786336, -1549786256, -1549786208, -1549786128, -1549762560, -1549753824, -1549753696, -1549753616, -1549753568, -1549753488, -1549753440, -1549753360, -1544552448, -1544543712, -1544543584, -1544543504, -1544543456, -1544543376, -1544543328, -1544543248, -1544519680, -1544510944, -1544510816, -1544510736, -1544510688, -1544510608, -1544510560, -1544510480, 0, 8752, 8880, 8964, 9008, 9092, 9136, 9220, 32768, 41520, 41648, 41732, 41776, 41860, 41904, 41988, 573571072, 573579824, 573579952, 573580036, 573580080, 573580164, 573580208, 573580292, 573603840, 573612592, 573612720, 573612804, 573612848, 573612932, 573612976, 573613060, 581959680, 581968432, 581968560, 581968644, 581968688, 581968772, 581968816, 581968900, 581992448, 582001200, 582001328, 582001412, 582001456, 582001540, 582001584, 582001668, 587464704, 587473456, 587473584, 587473668, 587473712, 587473796, 587473840, 587473924, 587497472, 587506224, 587506352, 587506436, 587506480, 587506564, 587506608, 587506692, 590348288, 590357040, 590357168, 590357252, 590357296, 590357380, 590357424, 590357508, 590381056, 590389808, 590389936, 590390020, 590390064, 590390148, 590390192, 590390276, 595853312, 595862064, 595862192, 595862276, 595862320, 595862404, 595862448, 595862532, 595886080, 595894832, 595894960, 595895044, 595895088, 595895172, 595895216, 595895300, 598736896, 598745648, 598745776, 598745860, 598745904, 598745988, 598746032, 598746116, 598769664, 598778416, 598778544, 598778628, 598778672, 598778756, 598778800, 598778884, 604241920, 604250672, 604250800, 604250884, 604250928, 604251012, 604251056, 604251140, 604274688, 604283440, 604283568, 604283652, 604283696, 604283780, 604283824, 604283908, -2147483648, -2147474896, -2147474768, -2147474684, -2147474640, -2147474556, -2147474512, -2147474428, -2147450880, -2147442128, -2147442000, -2147441916, -2147441872, -2147441788, -2147441744, -2147441660, -1573912576, -1573903824, -1573903696, -1573903612, -1573903568, -1573903484, -1573903440, -1573903356, -1573879808, -1573871056, -1573870928, -1573870844, -1573870800, -1573870716, -1573870672, -1573870588, -1565523968, -1565515216, -1565515088, -1565515004, -1565514960, -1565514876, -1565514832, -1565514748, -1565491200, -1565482448, -1565482320, -1565482236, -1565482192, -1565482108, -1565482064, -1565481980, -1560018944, -1560010192, -1560010064, -1560009980, -1560009936, -1560009852, -1560009808, -1560009724, -1559986176, -1559977424, -1559977296, -1559977212, -1559977168, -1559977084, -1559977040, -1559976956, -1557135360, -1557126608, -1557126480, -1557126396, -1557126352, -1557126268, -1557126224, -1557126140, -1557102592, -1557093840, -1557093712, -1557093628, -1557093584, -1557093500, -1557093456, -1557093372, -1551630336, -1551621584, -1551621456, -1551621372, -1551621328, -1551621244, -1551621200, -1551621116, -1551597568, -1551588816, -1551588688, -1551588604, -1551588560, -1551588476, -1551588432, -1551588348, -1548746752, -1548738000, -1548737872, -1548737788, -1548737744, -1548737660, -1548737616, -1548737532, -1548713984, -1548705232, -1548705104, -1548705020, -1548704976, -1548704892, -1548704848, -1548704764, -1543241728, -1543232976, -1543232848, -1543232764, -1543232720, -1543232636, -1543232592, -1543232508, -1543208960, -1543200208, -1543200080, -1543199996, -1543199952, -1543199868, -1543199824, -1543199740, 0, 8768, 8896, 8976, 9024, 9104, 9152, 9232, 32768, 41536, 41664, 41744, 41792, 41872, 41920, 42000, 574619648, 574628416, 574628544, 574628624, 574628672, 574628752, 574628800, 574628880, 574652416, 574661184, 574661312, 574661392, 574661440, 574661520, 574661568, 574661648, 583008256, 583017024, 583017152, 583017232, 583017280, 583017360, 583017408, 583017488, 583041024, 583049792, 583049920, 583050000, 583050048, 583050128, 583050176, 583050256, 588251136, 588259904, 588260032, 588260112, 588260160, 588260240, 588260288, 588260368, 588283904, 588292672, 588292800, 588292880, 588292928, 588293008, 588293056, 588293136, 591396864, 591405632, 591405760, 591405840, 591405888, 591405968, 591406016, 591406096, 591429632, 591438400, 591438528, 591438608, 591438656, 591438736, 591438784, 591438864, 596639744, 596648512, 596648640, 596648720, 596648768, 596648848, 596648896, 596648976, 596672512, 596681280, 596681408, 596681488, 596681536, 596681616, 596681664, 596681744, 599785472, 599794240, 599794368, 599794448, 599794496, 599794576, 599794624, 599794704, 599818240, 599827008, 599827136, 599827216, 599827264, 599827344, 599827392, 599827472, 605028352, 605037120, 605037248, 605037328, 605037376, 605037456, 605037504, 605037584, 605061120, 605069888, 605070016, 605070096, 605070144, 605070224, 605070272, 605070352, -2147483648, -2147474880, -2147474752, -2147474672, -2147474624, -2147474544, -2147474496, -2147474416, -2147450880, -2147442112, -2147441984, -2147441904, -2147441856, -2147441776, -2147441728, -2147441648, -1572864000, -1572855232, -1572855104, -1572855024, -1572854976, -1572854896, -1572854848, -1572854768, -1572831232, -1572822464, -1572822336, -1572822256, -1572822208, -1572822128, -1572822080, -1572822000, -1564475392, -1564466624, -1564466496, -1564466416, -1564466368, -1564466288, -1564466240, -1564466160, -1564442624, -1564433856, -1564433728, -1564433648, -1564433600, -1564433520, -1564433472, -1564433392, -1559232512, -1559223744, -1559223616, -1559223536, -1559223488, -1559223408, -1559223360, -1559223280, -1559199744, -1559190976, -1559190848, -1559190768, -1559190720, -1559190640, -1559190592, -1559190512, -1556086784, -1556078016, -1556077888, -1556077808, -1556077760, -1556077680, -1556077632, -1556077552, -1556054016, -1556045248, -1556045120, -1556045040, -1556044992, -1556044912, -1556044864, -1556044784, -1550843904, -1550835136, -1550835008, -1550834928, -1550834880, -1550834800, -1550834752, -1550834672, -1550811136, -1550802368, -1550802240, -1550802160, -1550802112, -1550802032, -1550801984, -1550801904, -1547698176, -1547689408, -1547689280, -1547689200, -1547689152, -1547689072, -1547689024, -1547688944, -1547665408, -1547656640, -1547656512, -1547656432, -1547656384, -1547656304, -1547656256, -1547656176, -1542455296, -1542446528, -1542446400, -1542446320, -1542446272, -1542446192, -1542446144, -1542446064, -1542422528, -1542413760, -1542413632, -1542413552, -1542413504, -1542413424, -1542413376, -1542413296, 0, 8784, 8912, 8988, 9040, 9116, 9168, 9244, 32768, 41552, 41680, 41756, 41808, 41884, 41936, 42012, 575668224, 575677008, 575677136, 575677212, 575677264, 575677340, 575677392, 575677468, 575700992, 575709776, 575709904, 575709980, 575710032, 575710108, 575710160, 575710236, 584056832, 584065616, 584065744, 584065820, 584065872, 584065948, 584066000, 584066076, 584089600, 584098384, 584098512, 584098588, 584098640, 584098716, 584098768, 584098844, 589037568, 589046352, 589046480, 589046556, 589046608, 589046684, 589046736, 589046812, 589070336, 589079120, 589079248, 589079324, 589079376, 589079452, 589079504, 589079580, 592445440, 592454224, 592454352, 592454428, 592454480, 592454556, 592454608, 592454684, 592478208, 592486992, 592487120, 592487196, 592487248, 592487324, 592487376, 592487452, 597426176, 597434960, 597435088, 597435164, 597435216, 597435292, 597435344, 597435420, 597458944, 597467728, 597467856, 597467932, 597467984, 597468060, 597468112, 597468188, 600834048, 600842832, 600842960, 600843036, 600843088, 600843164, 600843216, 600843292, 600866816, 600875600, 600875728, 600875804, 600875856, 600875932, 600875984, 600876060, 605814784, 605823568, 605823696, 605823772, 605823824, 605823900, 605823952, 605824028, 605847552, 605856336, 605856464, 605856540, 605856592, 605856668, 605856720, 605856796, -2147483648, -2147474864, -2147474736, -2147474660, -2147474608, -2147474532, -2147474480, -2147474404, -2147450880, -2147442096, -2147441968, -2147441892, -2147441840, -2147441764, -2147441712, -2147441636, -1571815424, -1571806640, -1571806512, -1571806436, -1571806384, -1571806308, -1571806256, -1571806180, -1571782656, -1571773872, -1571773744, -1571773668, -1571773616, -1571773540, -1571773488, -1571773412, -1563426816, -1563418032, -1563417904, -1563417828, -1563417776, -1563417700, -1563417648, -1563417572, -1563394048, -1563385264, -1563385136, -1563385060, -1563385008, -1563384932, -1563384880, -1563384804, -1558446080, -1558437296, -1558437168, -1558437092, -1558437040, -1558436964, -1558436912, -1558436836, -1558413312, -1558404528, -1558404400, -1558404324, -1558404272, -1558404196, -1558404144, -1558404068, -1555038208, -1555029424, -1555029296, -1555029220, -1555029168, -1555029092, -1555029040, -1555028964, -1555005440, -1554996656, -1554996528, -1554996452, -1554996400, -1554996324, -1554996272, -1554996196, -1550057472, -1550048688, -1550048560, -1550048484, -1550048432, -1550048356, -1550048304, -1550048228, -1550024704, -1550015920, -1550015792, -1550015716, -1550015664, -1550015588, -1550015536, -1550015460, -1546649600, -1546640816, -1546640688, -1546640612, -1546640560, -1546640484, -1546640432, -1546640356, -1546616832, -1546608048, -1546607920, -1546607844, -1546607792, -1546607716, -1546607664, -1546607588, -1541668864, -1541660080, -1541659952, -1541659876, -1541659824, -1541659748, -1541659696, -1541659620, -1541636096, -1541627312, -1541627184, -1541627108, -1541627056, -1541626980, -1541626928, -1541626852, 0, 8800, 8928, 9000, 9056, 9128, 9184, 9256, 32768, 41568, 41696, 41768, 41824, 41896, 41952, 42024, 576716800, 576725600, 576725728, 576725800, 576725856, 576725928, 576725984, 576726056, 576749568, 576758368, 576758496, 576758568, 576758624, 576758696, 576758752, 576758824, 585105408, 585114208, 585114336, 585114408, 585114464, 585114536, 585114592, 585114664, 585138176, 585146976, 585147104, 585147176, 585147232, 585147304, 585147360, 585147432, 589824000, 589832800, 589832928, 589833000, 589833056, 589833128, 589833184, 589833256, 589856768, 589865568, 589865696, 589865768, 589865824, 589865896, 589865952, 589866024, 593494016, 593502816, 593502944, 593503016, 593503072, 593503144, 593503200, 593503272, 593526784, 593535584, 593535712, 593535784, 593535840, 593535912, 593535968, 593536040, 598212608, 598221408, 598221536, 598221608, 598221664, 598221736, 598221792, 598221864, 598245376, 598254176, 598254304, 598254376, 598254432, 598254504, 598254560, 598254632, 601882624, 601891424, 601891552, 601891624, 601891680, 601891752, 601891808, 601891880, 601915392, 601924192, 601924320, 601924392, 601924448, 601924520, 601924576, 601924648, 606601216, 606610016, 606610144, 606610216, 606610272, 606610344, 606610400, 606610472, 606633984, 606642784, 606642912, 606642984, 606643040, 606643112, 606643168, 606643240, -2147483648, -2147474848, -2147474720, -2147474648, -2147474592, -2147474520, -2147474464, -2147474392, -2147450880, -2147442080, -2147441952, -2147441880, -2147441824, -2147441752, -2147441696, -2147441624, -1570766848, -1570758048, -1570757920, -1570757848, -1570757792, -1570757720, -1570757664, -1570757592, -1570734080, -1570725280, -1570725152, -1570725080, -1570725024, -1570724952, -1570724896, -1570724824, -1562378240, -1562369440, -1562369312, -1562369240, -1562369184, -1562369112, -1562369056, -1562368984, -1562345472, -1562336672, -1562336544, -1562336472, -1562336416, -1562336344, -1562336288, -1562336216, -1557659648, -1557650848, -1557650720, -1557650648, -1557650592, -1557650520, -1557650464, -1557650392, -1557626880, -1557618080, -1557617952, -1557617880, -1557617824, -1557617752, -1557617696, -1557617624, -1553989632, -1553980832, -1553980704, -1553980632, -1553980576, -1553980504, -1553980448, -1553980376, -1553956864, -1553948064, -1553947936, -1553947864, -1553947808, -1553947736, -1553947680, -1553947608, -1549271040, -1549262240, -1549262112, -1549262040, -1549261984, -1549261912, -1549261856, -1549261784, -1549238272, -1549229472, -1549229344, -1549229272, -1549229216, -1549229144, -1549229088, -1549229016, -1545601024, -1545592224, -1545592096, -1545592024, -1545591968, -1545591896, -1545591840, -1545591768, -1545568256, -1545559456, -1545559328, -1545559256, -1545559200, -1545559128, -1545559072, -1545559000, -1540882432, -1540873632, -1540873504, -1540873432, -1540873376, -1540873304, -1540873248, -1540873176, -1540849664, -1540840864, -1540840736, -1540840664, -1540840608, -1540840536, -1540840480, -1540840408, 0, 8816, 8944, 9012, 9072, 9140, 9200, 9268, 32768, 41584, 41712, 41780, 41840, 41908, 41968, 42036, 577765376, 577774192, 577774320, 577774388, 577774448, 577774516, 577774576, 577774644, 577798144, 577806960, 577807088, 577807156, 577807216, 577807284, 577807344, 577807412, 586153984, 586162800, 586162928, 586162996, 586163056, 586163124, 586163184, 586163252, 586186752, 586195568, 586195696, 586195764, 586195824, 586195892, 586195952, 586196020, 590610432, 590619248, 590619376, 590619444, 590619504, 590619572, 590619632, 590619700, 590643200, 590652016, 590652144, 590652212, 590652272, 590652340, 590652400, 590652468, 594542592, 594551408, 594551536, 594551604, 594551664, 594551732, 594551792, 594551860, 594575360, 594584176, 594584304, 594584372, 594584432, 594584500, 594584560, 594584628, 598999040, 599007856, 599007984, 599008052, 599008112, 599008180, 599008240, 599008308, 599031808, 599040624, 599040752, 599040820, 599040880, 599040948, 599041008, 599041076, 602931200, 602940016, 602940144, 602940212, 602940272, 602940340, 602940400, 602940468, 602963968, 602972784, 602972912, 602972980, 602973040, 602973108, 602973168, 602973236, 607387648, 607396464, 607396592, 607396660, 607396720, 607396788, 607396848, 607396916, 607420416, 607429232, 607429360, 607429428, 607429488, 607429556, 607429616, 607429684, -2147483648, -2147474832, -2147474704, -2147474636, -2147474576, -2147474508, -2147474448, -2147474380, -2147450880, -2147442064, -2147441936, -2147441868, -2147441808, -2147441740, -2147441680, -2147441612, -1569718272, -1569709456, -1569709328, -1569709260, -1569709200, -1569709132, -1569709072, -1569709004, -1569685504, -1569676688, -1569676560, -1569676492, -1569676432, -1569676364, -1569676304, -1569676236, -1561329664, -1561320848, -1561320720, -1561320652, -1561320592, -1561320524, -1561320464, -1561320396, -1561296896, -1561288080, -1561287952, -1561287884, -1561287824, -1561287756, -1561287696, -1561287628, -1556873216, -1556864400, -1556864272, -1556864204, -1556864144, -1556864076, -1556864016, -1556863948, -1556840448, -1556831632, -1556831504, -1556831436, -1556831376, -1556831308, -1556831248, -1556831180, -1552941056, -1552932240, -1552932112, -1552932044, -1552931984, -1552931916, -1552931856, -1552931788, -1552908288, -1552899472, -1552899344, -1552899276, -1552899216, -1552899148, -1552899088, -1552899020, -1548484608, -1548475792, -1548475664, -1548475596, -1548475536, -1548475468, -1548475408, -1548475340, -1548451840, -1548443024, -1548442896, -1548442828, -1548442768, -1548442700, -1548442640, -1548442572, -1544552448, -1544543632, -1544543504, -1544543436, -1544543376, -1544543308, -1544543248, -1544543180, -1544519680, -1544510864, -1544510736, -1544510668, -1544510608, -1544510540, -1544510480, -1544510412, -1540096000, -1540087184, -1540087056, -1540086988, -1540086928, -1540086860, -1540086800, -1540086732, -1540063232, -1540054416, -1540054288, -1540054220, -1540054160, -1540054092, -1540054032, -1540053964, 0, 8832, 8960, 9024, 9088, 9152, 9216, 9280, 32768, 41600, 41728, 41792, 41856, 41920, 41984, 42048, 578813952, 578822784, 578822912, 578822976, 578823040, 578823104, 578823168, 578823232, 578846720, 578855552, 578855680, 578855744, 578855808, 578855872, 578855936, 578856000, 587202560, 587211392, 587211520, 587211584, 587211648, 587211712, 587211776, 587211840, 587235328, 587244160, 587244288, 587244352, 587244416, 587244480, 587244544, 587244608, 591396864, 591405696, 591405824, 591405888, 591405952, 591406016, 591406080, 591406144, 591429632, 591438464, 591438592, 591438656, 591438720, 591438784, 591438848, 591438912, 595591168, 595600000, 595600128, 595600192, 595600256, 595600320, 595600384, 595600448, 595623936, 595632768, 595632896, 595632960, 595633024, 595633088, 595633152, 595633216, 599785472, 599794304, 599794432, 599794496, 599794560, 599794624, 599794688, 599794752, 599818240, 599827072, 599827200, 599827264, 599827328, 599827392, 599827456, 599827520, 603979776, 603988608, 603988736, 603988800, 603988864, 603988928, 603988992, 603989056, 604012544, 604021376, 604021504, 604021568, 604021632, 604021696, 604021760, 604021824, 608174080, 608182912, 608183040, 608183104, 608183168, 608183232, 608183296, 608183360, 608206848, 608215680, 608215808, 608215872, 608215936, 608216000, 608216064, 608216128, -2147483648, -2147474816, -2147474688, -2147474624, -2147474560, -2147474496, -2147474432, -2147474368, -2147450880, -2147442048, -2147441920, -2147441856, -2147441792, -2147441728, -2147441664, -2147441600, -1568669696, -1568660864, -1568660736, -1568660672, -1568660608, -1568660544, -1568660480, -1568660416, -1568636928, -1568628096, -1568627968, -1568627904, -1568627840, -1568627776, -1568627712, -1568627648, -1560281088, -1560272256, -1560272128, -1560272064, -1560272000, -1560271936, -1560271872, -1560271808, -1560248320, -1560239488, -1560239360, -1560239296, -1560239232, -1560239168, -1560239104, -1560239040, -1556086784, -1556077952, -1556077824, -1556077760, -1556077696, -1556077632, -1556077568, -1556077504, -1556054016, -1556045184, -1556045056, -1556044992, -1556044928, -1556044864, -1556044800, -1556044736, -1551892480, -1551883648, -1551883520, -1551883456, -1551883392, -1551883328, -1551883264, -1551883200, -1551859712, -1551850880, -1551850752, -1551850688, -1551850624, -1551850560, -1551850496, -1551850432, -1547698176, -1547689344, -1547689216, -1547689152, -1547689088, -1547689024, -1547688960, -1547688896, -1547665408, -1547656576, -1547656448, -1547656384, -1547656320, -1547656256, -1547656192, -1547656128, -1543503872, -1543495040, -1543494912, -1543494848, -1543494784, -1543494720, -1543494656, -1543494592, -1543471104, -1543462272, -1543462144, -1543462080, -1543462016, -1543461952, -1543461888, -1543461824, -1539309568, -1539300736, -1539300608, -1539300544, -1539300480, -1539300416, -1539300352, -1539300288, -1539276800, -1539267968, -1539267840, -1539267776, -1539267712, -1539267648, -1539267584, -1539267520, 0, 8848, 8976, 9048, 9104, 9176, 9232, 9304, 32768, 41616, 41744, 41816, 41872, 41944, 42000, 42072, 579862528, 579871376, 579871504, 579871576, 579871632, 579871704, 579871760, 579871832, 579895296, 579904144, 579904272, 579904344, 579904400, 579904472, 579904528, 579904600, 588251136, 588259984, 588260112, 588260184, 588260240, 588260312, 588260368, 588260440, 588283904, 588292752, 588292880, 588292952, 588293008, 588293080, 588293136, 588293208, 592969728, 592978576, 592978704, 592978776, 592978832, 592978904, 592978960, 592979032, 593002496, 593011344, 593011472, 593011544, 593011600, 593011672, 593011728, 593011800, 596639744, 596648592, 596648720, 596648792, 596648848, 596648920, 596648976, 596649048, 596672512, 596681360, 596681488, 596681560, 596681616, 596681688, 596681744, 596681816, 601358336, 601367184, 601367312, 601367384, 601367440, 601367512, 601367568, 601367640, 601391104, 601399952, 601400080, 601400152, 601400208, 601400280, 601400336, 601400408, 605028352, 605037200, 605037328, 605037400, 605037456, 605037528, 605037584, 605037656, 605061120, 605069968, 605070096, 605070168, 605070224, 605070296, 605070352, 605070424, 609746944, 609755792, 609755920, 609755992, 609756048, 609756120, 609756176, 609756248, 609779712, 609788560, 609788688, 609788760, 609788816, 609788888, 609788944, 609789016, -2147483648, -2147474800, -2147474672, -2147474600, -2147474544, -2147474472, -2147474416, -2147474344, -2147450880, -2147442032, -2147441904, -2147441832, -2147441776, -2147441704, -2147441648, -2147441576, -1567621120, -1567612272, -1567612144, -1567612072, -1567612016, -1567611944, -1567611888, -1567611816, -1567588352, -1567579504, -1567579376, -1567579304, -1567579248, -1567579176, -1567579120, -1567579048, -1559232512, -1559223664, -1559223536, -1559223464, -1559223408, -1559223336, -1559223280, -1559223208, -1559199744, -1559190896, -1559190768, -1559190696, -1559190640, -1559190568, -1559190512, -1559190440, -1554513920, -1554505072, -1554504944, -1554504872, -1554504816, -1554504744, -1554504688, -1554504616, -1554481152, -1554472304, -1554472176, -1554472104, -1554472048, -1554471976, -1554471920, -1554471848, -1550843904, -1550835056, -1550834928, -1550834856, -1550834800, -1550834728, -1550834672, -1550834600, -1550811136, -1550802288, -1550802160, -1550802088, -1550802032, -1550801960, -1550801904, -1550801832, -1546125312, -1546116464, -1546116336, -1546116264, -1546116208, -1546116136, -1546116080, -1546116008, -1546092544, -1546083696, -1546083568, -1546083496, -1546083440, -1546083368, -1546083312, -1546083240, -1542455296, -1542446448, -1542446320, -1542446248, -1542446192, -1542446120, -1542446064, -1542445992, -1542422528, -1542413680, -1542413552, -1542413480, -1542413424, -1542413352, -1542413296, -1542413224, -1537736704, -1537727856, -1537727728, -1537727656, -1537727600, -1537727528, -1537727472, -1537727400, -1537703936, -1537695088, -1537694960, -1537694888, -1537694832, -1537694760, -1537694704, -1537694632, 0, 8864, 8992, 9072, 9120, 9200, 9248, 9328, 32768, 41632, 41760, 41840, 41888, 41968, 42016, 42096, 580911104, 580919968, 580920096, 580920176, 580920224, 580920304, 580920352, 580920432, 580943872, 580952736, 580952864, 580952944, 580952992, 580953072, 580953120, 580953200, 589299712, 589308576, 589308704, 589308784, 589308832, 589308912, 589308960, 589309040, 589332480, 589341344, 589341472, 589341552, 589341600, 589341680, 589341728, 589341808, 594542592, 594551456, 594551584, 594551664, 594551712, 594551792, 594551840, 594551920, 594575360, 594584224, 594584352, 594584432, 594584480, 594584560, 594584608, 594584688, 597688320, 597697184, 597697312, 597697392, 597697440, 597697520, 597697568, 597697648, 597721088, 597729952, 597730080, 597730160, 597730208, 597730288, 597730336, 597730416, 602931200, 602940064, 602940192, 602940272, 602940320, 602940400, 602940448, 602940528, 602963968, 602972832, 602972960, 602973040, 602973088, 602973168, 602973216, 602973296, 606076928, 606085792, 606085920, 606086000, 606086048, 606086128, 606086176, 606086256, 606109696, 606118560, 606118688, 606118768, 606118816, 606118896, 606118944, 606119024, 611319808, 611328672, 611328800, 611328880, 611328928, 611329008, 611329056, 611329136, 611352576, 611361440, 611361568, 611361648, 611361696, 611361776, 611361824, 611361904, -2147483648, -2147474784, -2147474656, -2147474576, -2147474528, -2147474448, -2147474400, -2147474320, -2147450880, -2147442016, -2147441888, -2147441808, -2147441760, -2147441680, -2147441632, -2147441552, -1566572544, -1566563680, -1566563552, -1566563472, -1566563424, -1566563344, -1566563296, -1566563216, -1566539776, -1566530912, -1566530784, -1566530704, -1566530656, -1566530576, -1566530528, -1566530448, -1558183936, -1558175072, -1558174944, -1558174864, -1558174816, -1558174736, -1558174688, -1558174608, -1558151168, -1558142304, -1558142176, -1558142096, -1558142048, -1558141968, -1558141920, -1558141840, -1552941056, -1552932192, -1552932064, -1552931984, -1552931936, -1552931856, -1552931808, -1552931728, -1552908288, -1552899424, -1552899296, -1552899216, -1552899168, -1552899088, -1552899040, -1552898960, -1549795328, -1549786464, -1549786336, -1549786256, -1549786208, -1549786128, -1549786080, -1549786000, -1549762560, -1549753696, -1549753568, -1549753488, -1549753440, -1549753360, -1549753312, -1549753232, -1544552448, -1544543584, -1544543456, -1544543376, -1544543328, -1544543248, -1544543200, -1544543120, -1544519680, -1544510816, -1544510688, -1544510608, -1544510560, -1544510480, -1544510432, -1544510352, -1541406720, -1541397856, -1541397728, -1541397648, -1541397600, -1541397520, -1541397472, -1541397392, -1541373952, -1541365088, -1541364960, -1541364880, -1541364832, -1541364752, -1541364704, -1541364624, -1536163840, -1536154976, -1536154848, -1536154768, -1536154720, -1536154640, -1536154592, -1536154512, -1536131072, -1536122208, -1536122080, -1536122000, -1536121952, -1536121872, -1536121824, -1536121744, 0, 8880, 9008, 9092, 9136, 9220, 9264, 9348, 32768, 41648, 41776, 41860, 41904, 41988, 42032, 42116, 581959680, 581968560, 581968688, 581968772, 581968816, 581968900, 581968944, 581969028, 581992448, 582001328, 582001456, 582001540, 582001584, 582001668, 582001712, 582001796, 590348288, 590357168, 590357296, 590357380, 590357424, 590357508, 590357552, 590357636, 590381056, 590389936, 590390064, 590390148, 590390192, 590390276, 590390320, 590390404, 595853312, 595862192, 595862320, 595862404, 595862448, 595862532, 595862576, 595862660, 595886080, 595894960, 595895088, 595895172, 595895216, 595895300, 595895344, 595895428, 598736896, 598745776, 598745904, 598745988, 598746032, 598746116, 598746160, 598746244, 598769664, 598778544, 598778672, 598778756, 598778800, 598778884, 598778928, 598779012, 604241920, 604250800, 604250928, 604251012, 604251056, 604251140, 604251184, 604251268, 604274688, 604283568, 604283696, 604283780, 604283824, 604283908, 604283952, 604284036, 607125504, 607134384, 607134512, 607134596, 607134640, 607134724, 607134768, 607134852, 607158272, 607167152, 607167280, 607167364, 607167408, 607167492, 607167536, 607167620, 612630528, 612639408, 612639536, 612639620, 612639664, 612639748, 612639792, 612639876, 612663296, 612672176, 612672304, 612672388, 612672432, 612672516, 612672560, 612672644, -2147483648, -2147474768, -2147474640, -2147474556, -2147474512, -2147474428, -2147474384, -2147474300, -2147450880, -2147442000, -2147441872, -2147441788, -2147441744, -2147441660, -2147441616, -2147441532, -1565523968, -1565515088, -1565514960, -1565514876, -1565514832, -1565514748, -1565514704, -1565514620, -1565491200, -1565482320, -1565482192, -1565482108, -1565482064, -1565481980, -1565481936, -1565481852, -1557135360, -1557126480, -1557126352, -1557126268, -1557126224, -1557126140, -1557126096, -1557126012, -1557102592, -1557093712, -1557093584, -1557093500, -1557093456, -1557093372, -1557093328, -1557093244, -1551630336, -1551621456, -1551621328, -1551621244, -1551621200, -1551621116, -1551621072, -1551620988, -1551597568, -1551588688, -1551588560, -1551588476, -1551588432, -1551588348, -1551588304, -1551588220, -1548746752, -1548737872, -1548737744, -1548737660, -1548737616, -1548737532, -1548737488, -1548737404, -1548713984, -1548705104, -1548704976, -1548704892, -1548704848, -1548704764, -1548704720, -1548704636, -1543241728, -1543232848, -1543232720, -1543232636, -1543232592, -1543232508, -1543232464, -1543232380, -1543208960, -1543200080, -1543199952, -1543199868, -1543199824, -1543199740, -1543199696, -1543199612, -1540358144, -1540349264, -1540349136, -1540349052, -1540349008, -1540348924, -1540348880, -1540348796, -1540325376, -1540316496, -1540316368, -1540316284, -1540316240, -1540316156, -1540316112, -1540316028, -1534853120, -1534844240, -1534844112, -1534844028, -1534843984, -1534843900, -1534843856, -1534843772, -1534820352, -1534811472, -1534811344, -1534811260, -1534811216, -1534811132, -1534811088, -1534811004, 0, 8896, 9024, 9104, 9152, 9232, 9280, 9360, 32768, 41664, 41792, 41872, 41920, 42000, 42048, 42128, 583008256, 583017152, 583017280, 583017360, 583017408, 583017488, 583017536, 583017616, 583041024, 583049920, 583050048, 583050128, 583050176, 583050256, 583050304, 583050384, 591396864, 591405760, 591405888, 591405968, 591406016, 591406096, 591406144, 591406224, 591429632, 591438528, 591438656, 591438736, 591438784, 591438864, 591438912, 591438992, 596639744, 596648640, 596648768, 596648848, 596648896, 596648976, 596649024, 596649104, 596672512, 596681408, 596681536, 596681616, 596681664, 596681744, 596681792, 596681872, 599785472, 599794368, 599794496, 599794576, 599794624, 599794704, 599794752, 599794832, 599818240, 599827136, 599827264, 599827344, 599827392, 599827472, 599827520, 599827600, 605028352, 605037248, 605037376, 605037456, 605037504, 605037584, 605037632, 605037712, 605061120, 605070016, 605070144, 605070224, 605070272, 605070352, 605070400, 605070480, 608174080, 608182976, 608183104, 608183184, 608183232, 608183312, 608183360, 608183440, 608206848, 608215744, 608215872, 608215952, 608216000, 608216080, 608216128, 608216208, 613416960, 613425856, 613425984, 613426064, 613426112, 613426192, 613426240, 613426320, 613449728, 613458624, 613458752, 613458832, 613458880, 613458960, 613459008, 613459088, -2147483648, -2147474752, -2147474624, -2147474544, -2147474496, -2147474416, -2147474368, -2147474288, -2147450880, -2147441984, -2147441856, -2147441776, -2147441728, -2147441648, -2147441600, -2147441520, -1564475392, -1564466496, -1564466368, -1564466288, -1564466240, -1564466160, -1564466112, -1564466032, -1564442624, -1564433728, -1564433600, -1564433520, -1564433472, -1564433392, -1564433344, -1564433264, -1556086784, -1556077888, -1556077760, -1556077680, -1556077632, -1556077552, -1556077504, -1556077424, -1556054016, -1556045120, -1556044992, -1556044912, -1556044864, -1556044784, -1556044736, -1556044656, -1550843904, -1550835008, -1550834880, -1550834800, -1550834752, -1550834672, -1550834624, -1550834544, -1550811136, -1550802240, -1550802112, -1550802032, -1550801984, -1550801904, -1550801856, -1550801776, -1547698176, -1547689280, -1547689152, -1547689072, -1547689024, -1547688944, -1547688896, -1547688816, -1547665408, -1547656512, -1547656384, -1547656304, -1547656256, -1547656176, -1547656128, -1547656048, -1542455296, -1542446400, -1542446272, -1542446192, -1542446144, -1542446064, -1542446016, -1542445936, -1542422528, -1542413632, -1542413504, -1542413424, -1542413376, -1542413296, -1542413248, -1542413168, -1539309568, -1539300672, -1539300544, -1539300464, -1539300416, -1539300336, -1539300288, -1539300208, -1539276800, -1539267904, -1539267776, -1539267696, -1539267648, -1539267568, -1539267520, -1539267440, -1534066688, -1534057792, -1534057664, -1534057584, -1534057536, -1534057456, -1534057408, -1534057328, -1534033920, -1534025024, -1534024896, -1534024816, -1534024768, -1534024688, -1534024640, -1534024560, 0, 8912, 9040, 9116, 9168, 9244, 9296, 9372, 32768, 41680, 41808, 41884, 41936, 42012, 42064, 42140, 584056832, 584065744, 584065872, 584065948, 584066000, 584066076, 584066128, 584066204, 584089600, 584098512, 584098640, 584098716, 584098768, 584098844, 584098896, 584098972, 592445440, 592454352, 592454480, 592454556, 592454608, 592454684, 592454736, 592454812, 592478208, 592487120, 592487248, 592487324, 592487376, 592487452, 592487504, 592487580, 597426176, 597435088, 597435216, 597435292, 597435344, 597435420, 597435472, 597435548, 597458944, 597467856, 597467984, 597468060, 597468112, 597468188, 597468240, 597468316, 600834048, 600842960, 600843088, 600843164, 600843216, 600843292, 600843344, 600843420, 600866816, 600875728, 600875856, 600875932, 600875984, 600876060, 600876112, 600876188, 605814784, 605823696, 605823824, 605823900, 605823952, 605824028, 605824080, 605824156, 605847552, 605856464, 605856592, 605856668, 605856720, 605856796, 605856848, 605856924, 609222656, 609231568, 609231696, 609231772, 609231824, 609231900, 609231952, 609232028, 609255424, 609264336, 609264464, 609264540, 609264592, 609264668, 609264720, 609264796, 614203392, 614212304, 614212432, 614212508, 614212560, 614212636, 614212688, 614212764, 614236160, 614245072, 614245200, 614245276, 614245328, 614245404, 614245456, 614245532, -2147483648, -2147474736, -2147474608, -2147474532, -2147474480, -2147474404, -2147474352, -2147474276, -2147450880, -2147441968, -2147441840, -2147441764, -2147441712, -2147441636, -2147441584, -2147441508, -1563426816, -1563417904, -1563417776, -1563417700, -1563417648, -1563417572, -1563417520, -1563417444, -1563394048, -1563385136, -1563385008, -1563384932, -1563384880, -1563384804, -1563384752, -1563384676, -1555038208, -1555029296, -1555029168, -1555029092, -1555029040, -1555028964, -1555028912, -1555028836, -1555005440, -1554996528, -1554996400, -1554996324, -1554996272, -1554996196, -1554996144, -1554996068, -1550057472, -1550048560, -1550048432, -1550048356, -1550048304, -1550048228, -1550048176, -1550048100, -1550024704, -1550015792, -1550015664, -1550015588, -1550015536, -1550015460, -1550015408, -1550015332, -1546649600, -1546640688, -1546640560, -1546640484, -1546640432, -1546640356, -1546640304, -1546640228, -1546616832, -1546607920, -1546607792, -1546607716, -1546607664, -1546607588, -1546607536, -1546607460, -1541668864, -1541659952, -1541659824, -1541659748, -1541659696, -1541659620, -1541659568, -1541659492, -1541636096, -1541627184, -1541627056, -1541626980, -1541626928, -1541626852, -1541626800, -1541626724, -1538260992, -1538252080, -1538251952, -1538251876, -1538251824, -1538251748, -1538251696, -1538251620, -1538228224, -1538219312, -1538219184, -1538219108, -1538219056, -1538218980, -1538218928, -1538218852, -1533280256, -1533271344, -1533271216, -1533271140, -1533271088, -1533271012, -1533270960, -1533270884, -1533247488, -1533238576, -1533238448, -1533238372, -1533238320, -1533238244, -1533238192, -1533238116, 0, 8928, 9056, 9128, 9184, 9256, 9312, 9384, 32768, 41696, 41824, 41896, 41952, 42024, 42080, 42152, 585105408, 585114336, 585114464, 585114536, 585114592, 585114664, 585114720, 585114792, 585138176, 585147104, 585147232, 585147304, 585147360, 585147432, 585147488, 585147560, 593494016, 593502944, 593503072, 593503144, 593503200, 593503272, 593503328, 593503400, 593526784, 593535712, 593535840, 593535912, 593535968, 593536040, 593536096, 593536168, 598212608, 598221536, 598221664, 598221736, 598221792, 598221864, 598221920, 598221992, 598245376, 598254304, 598254432, 598254504, 598254560, 598254632, 598254688, 598254760, 601882624, 601891552, 601891680, 601891752, 601891808, 601891880, 601891936, 601892008, 601915392, 601924320, 601924448, 601924520, 601924576, 601924648, 601924704, 601924776, 606601216, 606610144, 606610272, 606610344, 606610400, 606610472, 606610528, 606610600, 606633984, 606642912, 606643040, 606643112, 606643168, 606643240, 606643296, 606643368, 610271232, 610280160, 610280288, 610280360, 610280416, 610280488, 610280544, 610280616, 610304000, 610312928, 610313056, 610313128, 610313184, 610313256, 610313312, 610313384, 614989824, 614998752, 614998880, 614998952, 614999008, 614999080, 614999136, 614999208, 615022592, 615031520, 615031648, 615031720, 615031776, 615031848, 615031904, 615031976, -2147483648, -2147474720, -2147474592, -2147474520, -2147474464, -2147474392, -2147474336, -2147474264, -2147450880, -2147441952, -2147441824, -2147441752, -2147441696, -2147441624, -2147441568, -2147441496, -1562378240, -1562369312, -1562369184, -1562369112, -1562369056, -1562368984, -1562368928, -1562368856, -1562345472, -1562336544, -1562336416, -1562336344, -1562336288, -1562336216, -1562336160, -1562336088, -1553989632, -1553980704, -1553980576, -1553980504, -1553980448, -1553980376, -1553980320, -1553980248, -1553956864, -1553947936, -1553947808, -1553947736, -1553947680, -1553947608, -1553947552, -1553947480, -1549271040, -1549262112, -1549261984, -1549261912, -1549261856, -1549261784, -1549261728, -1549261656, -1549238272, -1549229344, -1549229216, -1549229144, -1549229088, -1549229016, -1549228960, -1549228888, -1545601024, -1545592096, -1545591968, -1545591896, -1545591840, -1545591768, -1545591712, -1545591640, -1545568256, -1545559328, -1545559200, -1545559128, -1545559072, -1545559000, -1545558944, -1545558872, -1540882432, -1540873504, -1540873376, -1540873304, -1540873248, -1540873176, -1540873120, -1540873048, -1540849664, -1540840736, -1540840608, -1540840536, -1540840480, -1540840408, -1540840352, -1540840280, -1537212416, -1537203488, -1537203360, -1537203288, -1537203232, -1537203160, -1537203104, -1537203032, -1537179648, -1537170720, -1537170592, -1537170520, -1537170464, -1537170392, -1537170336, -1537170264, -1532493824, -1532484896, -1532484768, -1532484696, -1532484640, -1532484568, -1532484512, -1532484440, -1532461056, -1532452128, -1532452000, -1532451928, -1532451872, -1532451800, -1532451744, -1532451672, 0, 8944, 9072, 9140, 9200, 9268, 9328, 9396, 32768, 41712, 41840, 41908, 41968, 42036, 42096, 42164, 586153984, 586162928, 586163056, 586163124, 586163184, 586163252, 586163312, 586163380, 586186752, 586195696, 586195824, 586195892, 586195952, 586196020, 586196080, 586196148, 594542592, 594551536, 594551664, 594551732, 594551792, 594551860, 594551920, 594551988, 594575360, 594584304, 594584432, 594584500, 594584560, 594584628, 594584688, 594584756, 598999040, 599007984, 599008112, 599008180, 599008240, 599008308, 599008368, 599008436, 599031808, 599040752, 599040880, 599040948, 599041008, 599041076, 599041136, 599041204, 602931200, 602940144, 602940272, 602940340, 602940400, 602940468, 602940528, 602940596, 602963968, 602972912, 602973040, 602973108, 602973168, 602973236, 602973296, 602973364, 607387648, 607396592, 607396720, 607396788, 607396848, 607396916, 607396976, 607397044, 607420416, 607429360, 607429488, 607429556, 607429616, 607429684, 607429744, 607429812, 611319808, 611328752, 611328880, 611328948, 611329008, 611329076, 611329136, 611329204, 611352576, 611361520, 611361648, 611361716, 611361776, 611361844, 611361904, 611361972, 615776256, 615785200, 615785328, 615785396, 615785456, 615785524, 615785584, 615785652, 615809024, 615817968, 615818096, 615818164, 615818224, 615818292, 615818352, 615818420, -2147483648, -2147474704, -2147474576, -2147474508, -2147474448, -2147474380, -2147474320, -2147474252, -2147450880, -2147441936, -2147441808, -2147441740, -2147441680, -2147441612, -2147441552, -2147441484, -1561329664, -1561320720, -1561320592, -1561320524, -1561320464, -1561320396, -1561320336, -1561320268, -1561296896, -1561287952, -1561287824, -1561287756, -1561287696, -1561287628, -1561287568, -1561287500, -1552941056, -1552932112, -1552931984, -1552931916, -1552931856, -1552931788, -1552931728, -1552931660, -1552908288, -1552899344, -1552899216, -1552899148, -1552899088, -1552899020, -1552898960, -1552898892, -1548484608, -1548475664, -1548475536, -1548475468, -1548475408, -1548475340, -1548475280, -1548475212, -1548451840, -1548442896, -1548442768, -1548442700, -1548442640, -1548442572, -1548442512, -1548442444, -1544552448, -1544543504, -1544543376, -1544543308, -1544543248, -1544543180, -1544543120, -1544543052, -1544519680, -1544510736, -1544510608, -1544510540, -1544510480, -1544510412, -1544510352, -1544510284, -1540096000, -1540087056, -1540086928, -1540086860, -1540086800, -1540086732, -1540086672, -1540086604, -1540063232, -1540054288, -1540054160, -1540054092, -1540054032, -1540053964, -1540053904, -1540053836, -1536163840, -1536154896, -1536154768, -1536154700, -1536154640, -1536154572, -1536154512, -1536154444, -1536131072, -1536122128, -1536122000, -1536121932, -1536121872, -1536121804, -1536121744, -1536121676, -1531707392, -1531698448, -1531698320, -1531698252, -1531698192, -1531698124, -1531698064, -1531697996, -1531674624, -1531665680, -1531665552, -1531665484, -1531665424, -1531665356, -1531665296, -1531665228, 0, 8960, 9088, 9152, 9216, 9280, 9344, 9408, 32768, 41728, 41856, 41920, 41984, 42048, 42112, 42176, 587202560, 587211520, 587211648, 587211712, 587211776, 587211840, 587211904, 587211968, 587235328, 587244288, 587244416, 587244480, 587244544, 587244608, 587244672, 587244736, 595591168, 595600128, 595600256, 595600320, 595600384, 595600448, 595600512, 595600576, 595623936, 595632896, 595633024, 595633088, 595633152, 595633216, 595633280, 595633344, 599785472, 599794432, 599794560, 599794624, 599794688, 599794752, 599794816, 599794880, 599818240, 599827200, 599827328, 599827392, 599827456, 599827520, 599827584, 599827648, 603979776, 603988736, 603988864, 603988928, 603988992, 603989056, 603989120, 603989184, 604012544, 604021504, 604021632, 604021696, 604021760, 604021824, 604021888, 604021952, 608174080, 608183040, 608183168, 608183232, 608183296, 608183360, 608183424, 608183488, 608206848, 608215808, 608215936, 608216000, 608216064, 608216128, 608216192, 608216256, 612368384, 612377344, 612377472, 612377536, 612377600, 612377664, 612377728, 612377792, 612401152, 612410112, 612410240, 612410304, 612410368, 612410432, 612410496, 612410560, 616562688, 616571648, 616571776, 616571840, 616571904, 616571968, 616572032, 616572096, 616595456, 616604416, 616604544, 616604608, 616604672, 616604736, 616604800, 616604864, -2147483648, -2147474688, -2147474560, -2147474496, -2147474432, -2147474368, -2147474304, -2147474240, -2147450880, -2147441920, -2147441792, -2147441728, -2147441664, -2147441600, -2147441536, -2147441472, -1560281088, -1560272128, -1560272000, -1560271936, -1560271872, -1560271808, -1560271744, -1560271680, -1560248320, -1560239360, -1560239232, -1560239168, -1560239104, -1560239040, -1560238976, -1560238912, -1551892480, -1551883520, -1551883392, -1551883328, -1551883264, -1551883200, -1551883136, -1551883072, -1551859712, -1551850752, -1551850624, -1551850560, -1551850496, -1551850432, -1551850368, -1551850304, -1547698176, -1547689216, -1547689088, -1547689024, -1547688960, -1547688896, -1547688832, -1547688768, -1547665408, -1547656448, -1547656320, -1547656256, -1547656192, -1547656128, -1547656064, -1547656000, -1543503872, -1543494912, -1543494784, -1543494720, -1543494656, -1543494592, -1543494528, -1543494464, -1543471104, -1543462144, -1543462016, -1543461952, -1543461888, -1543461824, -1543461760, -1543461696, -1539309568, -1539300608, -1539300480, -1539300416, -1539300352, -1539300288, -1539300224, -1539300160, -1539276800, -1539267840, -1539267712, -1539267648, -1539267584, -1539267520, -1539267456, -1539267392, -1535115264, -1535106304, -1535106176, -1535106112, -1535106048, -1535105984, -1535105920, -1535105856, -1535082496, -1535073536, -1535073408, -1535073344, -1535073280, -1535073216, -1535073152, -1535073088, -1530920960, -1530912000, -1530911872, -1530911808, -1530911744, -1530911680, -1530911616, -1530911552, -1530888192, -1530879232, -1530879104, -1530879040, -1530878976, -1530878912, -1530878848, -1530878784, 0, 8976, 9104, 9176, 9232, 9304, 9360, 9432, 32768, 41744, 41872, 41944, 42000, 42072, 42128, 42200, 588251136, 588260112, 588260240, 588260312, 588260368, 588260440, 588260496, 588260568, 588283904, 588292880, 588293008, 588293080, 588293136, 588293208, 588293264, 588293336, 596639744, 596648720, 596648848, 596648920, 596648976, 596649048, 596649104, 596649176, 596672512, 596681488, 596681616, 596681688, 596681744, 596681816, 596681872, 596681944, 601358336, 601367312, 601367440, 601367512, 601367568, 601367640, 601367696, 601367768, 601391104, 601400080, 601400208, 601400280, 601400336, 601400408, 601400464, 601400536, 605028352, 605037328, 605037456, 605037528, 605037584, 605037656, 605037712, 605037784, 605061120, 605070096, 605070224, 605070296, 605070352, 605070424, 605070480, 605070552, 609746944, 609755920, 609756048, 609756120, 609756176, 609756248, 609756304, 609756376, 609779712, 609788688, 609788816, 609788888, 609788944, 609789016, 609789072, 609789144, 613416960, 613425936, 613426064, 613426136, 613426192, 613426264, 613426320, 613426392, 613449728, 613458704, 613458832, 613458904, 613458960, 613459032, 613459088, 613459160, 618135552, 618144528, 618144656, 618144728, 618144784, 618144856, 618144912, 618144984, 618168320, 618177296, 618177424, 618177496, 618177552, 618177624, 618177680, 618177752, -2147483648, -2147474672, -2147474544, -2147474472, -2147474416, -2147474344, -2147474288, -2147474216, -2147450880, -2147441904, -2147441776, -2147441704, -2147441648, -2147441576, -2147441520, -2147441448, -1559232512, -1559223536, -1559223408, -1559223336, -1559223280, -1559223208, -1559223152, -1559223080, -1559199744, -1559190768, -1559190640, -1559190568, -1559190512, -1559190440, -1559190384, -1559190312, -1550843904, -1550834928, -1550834800, -1550834728, -1550834672, -1550834600, -1550834544, -1550834472, -1550811136, -1550802160, -1550802032, -1550801960, -1550801904, -1550801832, -1550801776, -1550801704, -1546125312, -1546116336, -1546116208, -1546116136, -1546116080, -1546116008, -1546115952, -1546115880, -1546092544, -1546083568, -1546083440, -1546083368, -1546083312, -1546083240, -1546083184, -1546083112, -1542455296, -1542446320, -1542446192, -1542446120, -1542446064, -1542445992, -1542445936, -1542445864, -1542422528, -1542413552, -1542413424, -1542413352, -1542413296, -1542413224, -1542413168, -1542413096, -1537736704, -1537727728, -1537727600, -1537727528, -1537727472, -1537727400, -1537727344, -1537727272, -1537703936, -1537694960, -1537694832, -1537694760, -1537694704, -1537694632, -1537694576, -1537694504, -1534066688, -1534057712, -1534057584, -1534057512, -1534057456, -1534057384, -1534057328, -1534057256, -1534033920, -1534024944, -1534024816, -1534024744, -1534024688, -1534024616, -1534024560, -1534024488, -1529348096, -1529339120, -1529338992, -1529338920, -1529338864, -1529338792, -1529338736, -1529338664, -1529315328, -1529306352, -1529306224, -1529306152, -1529306096, -1529306024, -1529305968, -1529305896, 0, 8992, 9120, 9200, 9248, 9328, 9376, 9456, 32768, 41760, 41888, 41968, 42016, 42096, 42144, 42224, 589299712, 589308704, 589308832, 589308912, 589308960, 589309040, 589309088, 589309168, 589332480, 589341472, 589341600, 589341680, 589341728, 589341808, 589341856, 589341936, 597688320, 597697312, 597697440, 597697520, 597697568, 597697648, 597697696, 597697776, 597721088, 597730080, 597730208, 597730288, 597730336, 597730416, 597730464, 597730544, 602931200, 602940192, 602940320, 602940400, 602940448, 602940528, 602940576, 602940656, 602963968, 602972960, 602973088, 602973168, 602973216, 602973296, 602973344, 602973424, 606076928, 606085920, 606086048, 606086128, 606086176, 606086256, 606086304, 606086384, 606109696, 606118688, 606118816, 606118896, 606118944, 606119024, 606119072, 606119152, 611319808, 611328800, 611328928, 611329008, 611329056, 611329136, 611329184, 611329264, 611352576, 611361568, 611361696, 611361776, 611361824, 611361904, 611361952, 611362032, 614465536, 614474528, 614474656, 614474736, 614474784, 614474864, 614474912, 614474992, 614498304, 614507296, 614507424, 614507504, 614507552, 614507632, 614507680, 614507760, 619708416, 619717408, 619717536, 619717616, 619717664, 619717744, 619717792, 619717872, 619741184, 619750176, 619750304, 619750384, 619750432, 619750512, 619750560, 619750640, -2147483648, -2147474656, -2147474528, -2147474448, -2147474400, -2147474320, -2147474272, -2147474192, -2147450880, -2147441888, -2147441760, -2147441680, -2147441632, -2147441552, -2147441504, -2147441424, -1558183936, -1558174944, -1558174816, -1558174736, -1558174688, -1558174608, -1558174560, -1558174480, -1558151168, -1558142176, -1558142048, -1558141968, -1558141920, -1558141840, -1558141792, -1558141712, -1549795328, -1549786336, -1549786208, -1549786128, -1549786080, -1549786000, -1549785952, -1549785872, -1549762560, -1549753568, -1549753440, -1549753360, -1549753312, -1549753232, -1549753184, -1549753104, -1544552448, -1544543456, -1544543328, -1544543248, -1544543200, -1544543120, -1544543072, -1544542992, -1544519680, -1544510688, -1544510560, -1544510480, -1544510432, -1544510352, -1544510304, -1544510224, -1541406720, -1541397728, -1541397600, -1541397520, -1541397472, -1541397392, -1541397344, -1541397264, -1541373952, -1541364960, -1541364832, -1541364752, -1541364704, -1541364624, -1541364576, -1541364496, -1536163840, -1536154848, -1536154720, -1536154640, -1536154592, -1536154512, -1536154464, -1536154384, -1536131072, -1536122080, -1536121952, -1536121872, -1536121824, -1536121744, -1536121696, -1536121616, -1533018112, -1533009120, -1533008992, -1533008912, -1533008864, -1533008784, -1533008736, -1533008656, -1532985344, -1532976352, -1532976224, -1532976144, -1532976096, -1532976016, -1532975968, -1532975888, -1527775232, -1527766240, -1527766112, -1527766032, -1527765984, -1527765904, -1527765856, -1527765776, -1527742464, -1527733472, -1527733344, -1527733264, -1527733216, -1527733136, -1527733088, -1527733008, 0, 9008, 9136, 9220, 9264, 9348, 9392, 9476, 32768, 41776, 41904, 41988, 42032, 42116, 42160, 42244, 590348288, 590357296, 590357424, 590357508, 590357552, 590357636, 590357680, 590357764, 590381056, 590390064, 590390192, 590390276, 590390320, 590390404, 590390448, 590390532, 598736896, 598745904, 598746032, 598746116, 598746160, 598746244, 598746288, 598746372, 598769664, 598778672, 598778800, 598778884, 598778928, 598779012, 598779056, 598779140, 604241920, 604250928, 604251056, 604251140, 604251184, 604251268, 604251312, 604251396, 604274688, 604283696, 604283824, 604283908, 604283952, 604284036, 604284080, 604284164, 607125504, 607134512, 607134640, 607134724, 607134768, 607134852, 607134896, 607134980, 607158272, 607167280, 607167408, 607167492, 607167536, 607167620, 607167664, 607167748, 612630528, 612639536, 612639664, 612639748, 612639792, 612639876, 612639920, 612640004, 612663296, 612672304, 612672432, 612672516, 612672560, 612672644, 612672688, 612672772, 615514112, 615523120, 615523248, 615523332, 615523376, 615523460, 615523504, 615523588, 615546880, 615555888, 615556016, 615556100, 615556144, 615556228, 615556272, 615556356, 621019136, 621028144, 621028272, 621028356, 621028400, 621028484, 621028528, 621028612, 621051904, 621060912, 621061040, 621061124, 621061168, 621061252, 621061296, 621061380, -2147483648, -2147474640, -2147474512, -2147474428, -2147474384, -2147474300, -2147474256, -2147474172, -2147450880, -2147441872, -2147441744, -2147441660, -2147441616, -2147441532, -2147441488, -2147441404, -1557135360, -1557126352, -1557126224, -1557126140, -1557126096, -1557126012, -1557125968, -1557125884, -1557102592, -1557093584, -1557093456, -1557093372, -1557093328, -1557093244, -1557093200, -1557093116, -1548746752, -1548737744, -1548737616, -1548737532, -1548737488, -1548737404, -1548737360, -1548737276, -1548713984, -1548704976, -1548704848, -1548704764, -1548704720, -1548704636, -1548704592, -1548704508, -1543241728, -1543232720, -1543232592, -1543232508, -1543232464, -1543232380, -1543232336, -1543232252, -1543208960, -1543199952, -1543199824, -1543199740, -1543199696, -1543199612, -1543199568, -1543199484, -1540358144, -1540349136, -1540349008, -1540348924, -1540348880, -1540348796, -1540348752, -1540348668, -1540325376, -1540316368, -1540316240, -1540316156, -1540316112, -1540316028, -1540315984, -1540315900, -1534853120, -1534844112, -1534843984, -1534843900, -1534843856, -1534843772, -1534843728, -1534843644, -1534820352, -1534811344, -1534811216, -1534811132, -1534811088, -1534811004, -1534810960, -1534810876, -1531969536, -1531960528, -1531960400, -1531960316, -1531960272, -1531960188, -1531960144, -1531960060, -1531936768, -1531927760, -1531927632, -1531927548, -1531927504, -1531927420, -1531927376, -1531927292, -1526464512, -1526455504, -1526455376, -1526455292, -1526455248, -1526455164, -1526455120, -1526455036, -1526431744, -1526422736, -1526422608, -1526422524, -1526422480, -1526422396, -1526422352, -1526422268, 0, 9024, 9152, 9232, 9280, 9360, 9408, 9488, 32768, 41792, 41920, 42000, 42048, 42128, 42176, 42256, 591396864, 591405888, 591406016, 591406096, 591406144, 591406224, 591406272, 591406352, 591429632, 591438656, 591438784, 591438864, 591438912, 591438992, 591439040, 591439120, 599785472, 599794496, 599794624, 599794704, 599794752, 599794832, 599794880, 599794960, 599818240, 599827264, 599827392, 599827472, 599827520, 599827600, 599827648, 599827728, 605028352, 605037376, 605037504, 605037584, 605037632, 605037712, 605037760, 605037840, 605061120, 605070144, 605070272, 605070352, 605070400, 605070480, 605070528, 605070608, 608174080, 608183104, 608183232, 608183312, 608183360, 608183440, 608183488, 608183568, 608206848, 608215872, 608216000, 608216080, 608216128, 608216208, 608216256, 608216336, 613416960, 613425984, 613426112, 613426192, 613426240, 613426320, 613426368, 613426448, 613449728, 613458752, 613458880, 613458960, 613459008, 613459088, 613459136, 613459216, 616562688, 616571712, 616571840, 616571920, 616571968, 616572048, 616572096, 616572176, 616595456, 616604480, 616604608, 616604688, 616604736, 616604816, 616604864, 616604944, 621805568, 621814592, 621814720, 621814800, 621814848, 621814928, 621814976, 621815056, 621838336, 621847360, 621847488, 621847568, 621847616, 621847696, 621847744, 621847824, -2147483648, -2147474624, -2147474496, -2147474416, -2147474368, -2147474288, -2147474240, -2147474160, -2147450880, -2147441856, -2147441728, -2147441648, -2147441600, -2147441520, -2147441472, -2147441392, -1556086784, -1556077760, -1556077632, -1556077552, -1556077504, -1556077424, -1556077376, -1556077296, -1556054016, -1556044992, -1556044864, -1556044784, -1556044736, -1556044656, -1556044608, -1556044528, -1547698176, -1547689152, -1547689024, -1547688944, -1547688896, -1547688816, -1547688768, -1547688688, -1547665408, -1547656384, -1547656256, -1547656176, -1547656128, -1547656048, -1547656000, -1547655920, -1542455296, -1542446272, -1542446144, -1542446064, -1542446016, -1542445936, -1542445888, -1542445808, -1542422528, -1542413504, -1542413376, -1542413296, -1542413248, -1542413168, -1542413120, -1542413040, -1539309568, -1539300544, -1539300416, -1539300336, -1539300288, -1539300208, -1539300160, -1539300080, -1539276800, -1539267776, -1539267648, -1539267568, -1539267520, -1539267440, -1539267392, -1539267312, -1534066688, -1534057664, -1534057536, -1534057456, -1534057408, -1534057328, -1534057280, -1534057200, -1534033920, -1534024896, -1534024768, -1534024688, -1534024640, -1534024560, -1534024512, -1534024432, -1530920960, -1530911936, -1530911808, -1530911728, -1530911680, -1530911600, -1530911552, -1530911472, -1530888192, -1530879168, -1530879040, -1530878960, -1530878912, -1530878832, -1530878784, -1530878704, -1525678080, -1525669056, -1525668928, -1525668848, -1525668800, -1525668720, -1525668672, -1525668592, -1525645312, -1525636288, -1525636160, -1525636080, -1525636032, -1525635952, -1525635904, -1525635824, 0, 9040, 9168, 9244, 9296, 9372, 9424, 9500, 32768, 41808, 41936, 42012, 42064, 42140, 42192, 42268, 592445440, 592454480, 592454608, 592454684, 592454736, 592454812, 592454864, 592454940, 592478208, 592487248, 592487376, 592487452, 592487504, 592487580, 592487632, 592487708, 600834048, 600843088, 600843216, 600843292, 600843344, 600843420, 600843472, 600843548, 600866816, 600875856, 600875984, 600876060, 600876112, 600876188, 600876240, 600876316, 605814784, 605823824, 605823952, 605824028, 605824080, 605824156, 605824208, 605824284, 605847552, 605856592, 605856720, 605856796, 605856848, 605856924, 605856976, 605857052, 609222656, 609231696, 609231824, 609231900, 609231952, 609232028, 609232080, 609232156, 609255424, 609264464, 609264592, 609264668, 609264720, 609264796, 609264848, 609264924, 614203392, 614212432, 614212560, 614212636, 614212688, 614212764, 614212816, 614212892, 614236160, 614245200, 614245328, 614245404, 614245456, 614245532, 614245584, 614245660, 617611264, 617620304, 617620432, 617620508, 617620560, 617620636, 617620688, 617620764, 617644032, 617653072, 617653200, 617653276, 617653328, 617653404, 617653456, 617653532, 622592000, 622601040, 622601168, 622601244, 622601296, 622601372, 622601424, 622601500, 622624768, 622633808, 622633936, 622634012, 622634064, 622634140, 622634192, 622634268, -2147483648, -2147474608, -2147474480, -2147474404, -2147474352, -2147474276, -2147474224, -2147474148, -2147450880, -2147441840, -2147441712, -2147441636, -2147441584, -2147441508, -2147441456, -2147441380, -1555038208, -1555029168, -1555029040, -1555028964, -1555028912, -1555028836, -1555028784, -1555028708, -1555005440, -1554996400, -1554996272, -1554996196, -1554996144, -1554996068, -1554996016, -1554995940, -1546649600, -1546640560, -1546640432, -1546640356, -1546640304, -1546640228, -1546640176, -1546640100, -1546616832, -1546607792, -1546607664, -1546607588, -1546607536, -1546607460, -1546607408, -1546607332, -1541668864, -1541659824, -1541659696, -1541659620, -1541659568, -1541659492, -1541659440, -1541659364, -1541636096, -1541627056, -1541626928, -1541626852, -1541626800, -1541626724, -1541626672, -1541626596, -1538260992, -1538251952, -1538251824, -1538251748, -1538251696, -1538251620, -1538251568, -1538251492, -1538228224, -1538219184, -1538219056, -1538218980, -1538218928, -1538218852, -1538218800, -1538218724, -1533280256, -1533271216, -1533271088, -1533271012, -1533270960, -1533270884, -1533270832, -1533270756, -1533247488, -1533238448, -1533238320, -1533238244, -1533238192, -1533238116, -1533238064, -1533237988, -1529872384, -1529863344, -1529863216, -1529863140, -1529863088, -1529863012, -1529862960, -1529862884, -1529839616, -1529830576, -1529830448, -1529830372, -1529830320, -1529830244, -1529830192, -1529830116, -1524891648, -1524882608, -1524882480, -1524882404, -1524882352, -1524882276, -1524882224, -1524882148, -1524858880, -1524849840, -1524849712, -1524849636, -1524849584, -1524849508, -1524849456, -1524849380, 0, 9056, 9184, 9256, 9312, 9384, 9440, 9512, 32768, 41824, 41952, 42024, 42080, 42152, 42208, 42280, 593494016, 593503072, 593503200, 593503272, 593503328, 593503400, 593503456, 593503528, 593526784, 593535840, 593535968, 593536040, 593536096, 593536168, 593536224, 593536296, 601882624, 601891680, 601891808, 601891880, 601891936, 601892008, 601892064, 601892136, 601915392, 601924448, 601924576, 601924648, 601924704, 601924776, 601924832, 601924904, 606601216, 606610272, 606610400, 606610472, 606610528, 606610600, 606610656, 606610728, 606633984, 606643040, 606643168, 606643240, 606643296, 606643368, 606643424, 606643496, 610271232, 610280288, 610280416, 610280488, 610280544, 610280616, 610280672, 610280744, 610304000, 610313056, 610313184, 610313256, 610313312, 610313384, 610313440, 610313512, 614989824, 614998880, 614999008, 614999080, 614999136, 614999208, 614999264, 614999336, 615022592, 615031648, 615031776, 615031848, 615031904, 615031976, 615032032, 615032104, 618659840, 618668896, 618669024, 618669096, 618669152, 618669224, 618669280, 618669352, 618692608, 618701664, 618701792, 618701864, 618701920, 618701992, 618702048, 618702120, 623378432, 623387488, 623387616, 623387688, 623387744, 623387816, 623387872, 623387944, 623411200, 623420256, 623420384, 623420456, 623420512, 623420584, 623420640, 623420712, -2147483648, -2147474592, -2147474464, -2147474392, -2147474336, -2147474264, -2147474208, -2147474136, -2147450880, -2147441824, -2147441696, -2147441624, -2147441568, -2147441496, -2147441440, -2147441368, -1553989632, -1553980576, -1553980448, -1553980376, -1553980320, -1553980248, -1553980192, -1553980120, -1553956864, -1553947808, -1553947680, -1553947608, -1553947552, -1553947480, -1553947424, -1553947352, -1545601024, -1545591968, -1545591840, -1545591768, -1545591712, -1545591640, -1545591584, -1545591512, -1545568256, -1545559200, -1545559072, -1545559000, -1545558944, -1545558872, -1545558816, -1545558744, -1540882432, -1540873376, -1540873248, -1540873176, -1540873120, -1540873048, -1540872992, -1540872920, -1540849664, -1540840608, -1540840480, -1540840408, -1540840352, -1540840280, -1540840224, -1540840152, -1537212416, -1537203360, -1537203232, -1537203160, -1537203104, -1537203032, -1537202976, -1537202904, -1537179648, -1537170592, -1537170464, -1537170392, -1537170336, -1537170264, -1537170208, -1537170136, -1532493824, -1532484768, -1532484640, -1532484568, -1532484512, -1532484440, -1532484384, -1532484312, -1532461056, -1532452000, -1532451872, -1532451800, -1532451744, -1532451672, -1532451616, -1532451544, -1528823808, -1528814752, -1528814624, -1528814552, -1528814496, -1528814424, -1528814368, -1528814296, -1528791040, -1528781984, -1528781856, -1528781784, -1528781728, -1528781656, -1528781600, -1528781528, -1524105216, -1524096160, -1524096032, -1524095960, -1524095904, -1524095832, -1524095776, -1524095704, -1524072448, -1524063392, -1524063264, -1524063192, -1524063136, -1524063064, -1524063008, -1524062936, 0, 9072, 9200, 9268, 9328, 9396, 9456, 9524, 32768, 41840, 41968, 42036, 42096, 42164, 42224, 42292, 594542592, 594551664, 594551792, 594551860, 594551920, 594551988, 594552048, 594552116, 594575360, 594584432, 594584560, 594584628, 594584688, 594584756, 594584816, 594584884, 602931200, 602940272, 602940400, 602940468, 602940528, 602940596, 602940656, 602940724, 602963968, 602973040, 602973168, 602973236, 602973296, 602973364, 602973424, 602973492, 607387648, 607396720, 607396848, 607396916, 607396976, 607397044, 607397104, 607397172, 607420416, 607429488, 607429616, 607429684, 607429744, 607429812, 607429872, 607429940, 611319808, 611328880, 611329008, 611329076, 611329136, 611329204, 611329264, 611329332, 611352576, 611361648, 611361776, 611361844, 611361904, 611361972, 611362032, 611362100, 615776256, 615785328, 615785456, 615785524, 615785584, 615785652, 615785712, 615785780, 615809024, 615818096, 615818224, 615818292, 615818352, 615818420, 615818480, 615818548, 619708416, 619717488, 619717616, 619717684, 619717744, 619717812, 619717872, 619717940, 619741184, 619750256, 619750384, 619750452, 619750512, 619750580, 619750640, 619750708, 624164864, 624173936, 624174064, 624174132, 624174192, 624174260, 624174320, 624174388, 624197632, 624206704, 624206832, 624206900, 624206960, 624207028, 624207088, 624207156, -2147483648, -2147474576, -2147474448, -2147474380, -2147474320, -2147474252, -2147474192, -2147474124, -2147450880, -2147441808, -2147441680, -2147441612, -2147441552, -2147441484, -2147441424, -2147441356, -1552941056, -1552931984, -1552931856, -1552931788, -1552931728, -1552931660, -1552931600, -1552931532, -1552908288, -1552899216, -1552899088, -1552899020, -1552898960, -1552898892, -1552898832, -1552898764, -1544552448, -1544543376, -1544543248, -1544543180, -1544543120, -1544543052, -1544542992, -1544542924, -1544519680, -1544510608, -1544510480, -1544510412, -1544510352, -1544510284, -1544510224, -1544510156, -1540096000, -1540086928, -1540086800, -1540086732, -1540086672, -1540086604, -1540086544, -1540086476, -1540063232, -1540054160, -1540054032, -1540053964, -1540053904, -1540053836, -1540053776, -1540053708, -1536163840, -1536154768, -1536154640, -1536154572, -1536154512, -1536154444, -1536154384, -1536154316, -1536131072, -1536122000, -1536121872, -1536121804, -1536121744, -1536121676, -1536121616, -1536121548, -1531707392, -1531698320, -1531698192, -1531698124, -1531698064, -1531697996, -1531697936, -1531697868, -1531674624, -1531665552, -1531665424, -1531665356, -1531665296, -1531665228, -1531665168, -1531665100, -1527775232, -1527766160, -1527766032, -1527765964, -1527765904, -1527765836, -1527765776, -1527765708, -1527742464, -1527733392, -1527733264, -1527733196, -1527733136, -1527733068, -1527733008, -1527732940, -1523318784, -1523309712, -1523309584, -1523309516, -1523309456, -1523309388, -1523309328, -1523309260, -1523286016, -1523276944, -1523276816, -1523276748, -1523276688, -1523276620, -1523276560, -1523276492, 0, 9088, 9216, 9280, 9344, 9408, 9472, 9536, 32768, 41856, 41984, 42048, 42112, 42176, 42240, 42304, 595591168, 595600256, 595600384, 595600448, 595600512, 595600576, 595600640, 595600704, 595623936, 595633024, 595633152, 595633216, 595633280, 595633344, 595633408, 595633472, 603979776, 603988864, 603988992, 603989056, 603989120, 603989184, 603989248, 603989312, 604012544, 604021632, 604021760, 604021824, 604021888, 604021952, 604022016, 604022080, 608174080, 608183168, 608183296, 608183360, 608183424, 608183488, 608183552, 608183616, 608206848, 608215936, 608216064, 608216128, 608216192, 608216256, 608216320, 608216384, 612368384, 612377472, 612377600, 612377664, 612377728, 612377792, 612377856, 612377920, 612401152, 612410240, 612410368, 612410432, 612410496, 612410560, 612410624, 612410688, 616562688, 616571776, 616571904, 616571968, 616572032, 616572096, 616572160, 616572224, 616595456, 616604544, 616604672, 616604736, 616604800, 616604864, 616604928, 616604992, 620756992, 620766080, 620766208, 620766272, 620766336, 620766400, 620766464, 620766528, 620789760, 620798848, 620798976, 620799040, 620799104, 620799168, 620799232, 620799296, 624951296, 624960384, 624960512, 624960576, 624960640, 624960704, 624960768, 624960832, 624984064, 624993152, 624993280, 624993344, 624993408, 624993472, 624993536, 624993600, -2147483648, -2147474560, -2147474432, -2147474368, -2147474304, -2147474240, -2147474176, -2147474112, -2147450880, -2147441792, -2147441664, -2147441600, -2147441536, -2147441472, -2147441408, -2147441344, -1551892480, -1551883392, -1551883264, -1551883200, -1551883136, -1551883072, -1551883008, -1551882944, -1551859712, -1551850624, -1551850496, -1551850432, -1551850368, -1551850304, -1551850240, -1551850176, -1543503872, -1543494784, -1543494656, -1543494592, -1543494528, -1543494464, -1543494400, -1543494336, -1543471104, -1543462016, -1543461888, -1543461824, -1543461760, -1543461696, -1543461632, -1543461568, -1539309568, -1539300480, -1539300352, -1539300288, -1539300224, -1539300160, -1539300096, -1539300032, -1539276800, -1539267712, -1539267584, -1539267520, -1539267456, -1539267392, -1539267328, -1539267264, -1535115264, -1535106176, -1535106048, -1535105984, -1535105920, -1535105856, -1535105792, -1535105728, -1535082496, -1535073408, -1535073280, -1535073216, -1535073152, -1535073088, -1535073024, -1535072960, -1530920960, -1530911872, -1530911744, -1530911680, -1530911616, -1530911552, -1530911488, -1530911424, -1530888192, -1530879104, -1530878976, -1530878912, -1530878848, -1530878784, -1530878720, -1530878656, -1526726656, -1526717568, -1526717440, -1526717376, -1526717312, -1526717248, -1526717184, -1526717120, -1526693888, -1526684800, -1526684672, -1526684608, -1526684544, -1526684480, -1526684416, -1526684352, -1522532352, -1522523264, -1522523136, -1522523072, -1522523008, -1522522944, -1522522880, -1522522816, -1522499584, -1522490496, -1522490368, -1522490304, -1522490240, -1522490176, -1522490112, -1522490048, 0, 9104, 9232, 9304, 9360, 9432, 9488, 9560, 32768, 41872, 42000, 42072, 42128, 42200, 42256, 42328, 596639744, 596648848, 596648976, 596649048, 596649104, 596649176, 596649232, 596649304, 596672512, 596681616, 596681744, 596681816, 596681872, 596681944, 596682000, 596682072, 605028352, 605037456, 605037584, 605037656, 605037712, 605037784, 605037840, 605037912, 605061120, 605070224, 605070352, 605070424, 605070480, 605070552, 605070608, 605070680, 609746944, 609756048, 609756176, 609756248, 609756304, 609756376, 609756432, 609756504, 609779712, 609788816, 609788944, 609789016, 609789072, 609789144, 609789200, 609789272, 613416960, 613426064, 613426192, 613426264, 613426320, 613426392, 613426448, 613426520, 613449728, 613458832, 613458960, 613459032, 613459088, 613459160, 613459216, 613459288, 618135552, 618144656, 618144784, 618144856, 618144912, 618144984, 618145040, 618145112, 618168320, 618177424, 618177552, 618177624, 618177680, 618177752, 618177808, 618177880, 621805568, 621814672, 621814800, 621814872, 621814928, 621815000, 621815056, 621815128, 621838336, 621847440, 621847568, 621847640, 621847696, 621847768, 621847824, 621847896, 626524160, 626533264, 626533392, 626533464, 626533520, 626533592, 626533648, 626533720, 626556928, 626566032, 626566160, 626566232, 626566288, 626566360, 626566416, 626566488, -2147483648, -2147474544, -2147474416, -2147474344, -2147474288, -2147474216, -2147474160, -2147474088, -2147450880, -2147441776, -2147441648, -2147441576, -2147441520, -2147441448, -2147441392, -2147441320, -1550843904, -1550834800, -1550834672, -1550834600, -1550834544, -1550834472, -1550834416, -1550834344, -1550811136, -1550802032, -1550801904, -1550801832, -1550801776, -1550801704, -1550801648, -1550801576, -1542455296, -1542446192, -1542446064, -1542445992, -1542445936, -1542445864, -1542445808, -1542445736, -1542422528, -1542413424, -1542413296, -1542413224, -1542413168, -1542413096, -1542413040, -1542412968, -1537736704, -1537727600, -1537727472, -1537727400, -1537727344, -1537727272, -1537727216, -1537727144, -1537703936, -1537694832, -1537694704, -1537694632, -1537694576, -1537694504, -1537694448, -1537694376, -1534066688, -1534057584, -1534057456, -1534057384, -1534057328, -1534057256, -1534057200, -1534057128, -1534033920, -1534024816, -1534024688, -1534024616, -1534024560, -1534024488, -1534024432, -1534024360, -1529348096, -1529338992, -1529338864, -1529338792, -1529338736, -1529338664, -1529338608, -1529338536, -1529315328, -1529306224, -1529306096, -1529306024, -1529305968, -1529305896, -1529305840, -1529305768, -1525678080, -1525668976, -1525668848, -1525668776, -1525668720, -1525668648, -1525668592, -1525668520, -1525645312, -1525636208, -1525636080, -1525636008, -1525635952, -1525635880, -1525635824, -1525635752, -1520959488, -1520950384, -1520950256, -1520950184, -1520950128, -1520950056, -1520950000, -1520949928, -1520926720, -1520917616, -1520917488, -1520917416, -1520917360, -1520917288, -1520917232, -1520917160, 0, 9120, 9248, 9328, 9376, 9456, 9504, 9584, 32768, 41888, 42016, 42096, 42144, 42224, 42272, 42352, 597688320, 597697440, 597697568, 597697648, 597697696, 597697776, 597697824, 597697904, 597721088, 597730208, 597730336, 597730416, 597730464, 597730544, 597730592, 597730672, 606076928, 606086048, 606086176, 606086256, 606086304, 606086384, 606086432, 606086512, 606109696, 606118816, 606118944, 606119024, 606119072, 606119152, 606119200, 606119280, 611319808, 611328928, 611329056, 611329136, 611329184, 611329264, 611329312, 611329392, 611352576, 611361696, 611361824, 611361904, 611361952, 611362032, 611362080, 611362160, 614465536, 614474656, 614474784, 614474864, 614474912, 614474992, 614475040, 614475120, 614498304, 614507424, 614507552, 614507632, 614507680, 614507760, 614507808, 614507888, 619708416, 619717536, 619717664, 619717744, 619717792, 619717872, 619717920, 619718000, 619741184, 619750304, 619750432, 619750512, 619750560, 619750640, 619750688, 619750768, 622854144, 622863264, 622863392, 622863472, 622863520, 622863600, 622863648, 622863728, 622886912, 622896032, 622896160, 622896240, 622896288, 622896368, 622896416, 622896496, 628097024, 628106144, 628106272, 628106352, 628106400, 628106480, 628106528, 628106608, 628129792, 628138912, 628139040, 628139120, 628139168, 628139248, 628139296, 628139376, -2147483648, -2147474528, -2147474400, -2147474320, -2147474272, -2147474192, -2147474144, -2147474064, -2147450880, -2147441760, -2147441632, -2147441552, -2147441504, -2147441424, -2147441376, -2147441296, -1549795328, -1549786208, -1549786080, -1549786000, -1549785952, -1549785872, -1549785824, -1549785744, -1549762560, -1549753440, -1549753312, -1549753232, -1549753184, -1549753104, -1549753056, -1549752976, -1541406720, -1541397600, -1541397472, -1541397392, -1541397344, -1541397264, -1541397216, -1541397136, -1541373952, -1541364832, -1541364704, -1541364624, -1541364576, -1541364496, -1541364448, -1541364368, -1536163840, -1536154720, -1536154592, -1536154512, -1536154464, -1536154384, -1536154336, -1536154256, -1536131072, -1536121952, -1536121824, -1536121744, -1536121696, -1536121616, -1536121568, -1536121488, -1533018112, -1533008992, -1533008864, -1533008784, -1533008736, -1533008656, -1533008608, -1533008528, -1532985344, -1532976224, -1532976096, -1532976016, -1532975968, -1532975888, -1532975840, -1532975760, -1527775232, -1527766112, -1527765984, -1527765904, -1527765856, -1527765776, -1527765728, -1527765648, -1527742464, -1527733344, -1527733216, -1527733136, -1527733088, -1527733008, -1527732960, -1527732880, -1524629504, -1524620384, -1524620256, -1524620176, -1524620128, -1524620048, -1524620000, -1524619920, -1524596736, -1524587616, -1524587488, -1524587408, -1524587360, -1524587280, -1524587232, -1524587152, -1519386624, -1519377504, -1519377376, -1519377296, -1519377248, -1519377168, -1519377120, -1519377040, -1519353856, -1519344736, -1519344608, -1519344528, -1519344480, -1519344400, -1519344352, -1519344272, 0, 9136, 9264, 9348, 9392, 9476, 9520, 9604, 32768, 41904, 42032, 42116, 42160, 42244, 42288, 42372, 598736896, 598746032, 598746160, 598746244, 598746288, 598746372, 598746416, 598746500, 598769664, 598778800, 598778928, 598779012, 598779056, 598779140, 598779184, 598779268, 607125504, 607134640, 607134768, 607134852, 607134896, 607134980, 607135024, 607135108, 607158272, 607167408, 607167536, 607167620, 607167664, 607167748, 607167792, 607167876, 612630528, 612639664, 612639792, 612639876, 612639920, 612640004, 612640048, 612640132, 612663296, 612672432, 612672560, 612672644, 612672688, 612672772, 612672816, 612672900, 615514112, 615523248, 615523376, 615523460, 615523504, 615523588, 615523632, 615523716, 615546880, 615556016, 615556144, 615556228, 615556272, 615556356, 615556400, 615556484, 621019136, 621028272, 621028400, 621028484, 621028528, 621028612, 621028656, 621028740, 621051904, 621061040, 621061168, 621061252, 621061296, 621061380, 621061424, 621061508, 623902720, 623911856, 623911984, 623912068, 623912112, 623912196, 623912240, 623912324, 623935488, 623944624, 623944752, 623944836, 623944880, 623944964, 623945008, 623945092, 629407744, 629416880, 629417008, 629417092, 629417136, 629417220, 629417264, 629417348, 629440512, 629449648, 629449776, 629449860, 629449904, 629449988, 629450032, 629450116, -2147483648, -2147474512, -2147474384, -2147474300, -2147474256, -2147474172, -2147474128, -2147474044, -2147450880, -2147441744, -2147441616, -2147441532, -2147441488, -2147441404, -2147441360, -2147441276, -1548746752, -1548737616, -1548737488, -1548737404, -1548737360, -1548737276, -1548737232, -1548737148, -1548713984, -1548704848, -1548704720, -1548704636, -1548704592, -1548704508, -1548704464, -1548704380, -1540358144, -1540349008, -1540348880, -1540348796, -1540348752, -1540348668, -1540348624, -1540348540, -1540325376, -1540316240, -1540316112, -1540316028, -1540315984, -1540315900, -1540315856, -1540315772, -1534853120, -1534843984, -1534843856, -1534843772, -1534843728, -1534843644, -1534843600, -1534843516, -1534820352, -1534811216, -1534811088, -1534811004, -1534810960, -1534810876, -1534810832, -1534810748, -1531969536, -1531960400, -1531960272, -1531960188, -1531960144, -1531960060, -1531960016, -1531959932, -1531936768, -1531927632, -1531927504, -1531927420, -1531927376, -1531927292, -1531927248, -1531927164, -1526464512, -1526455376, -1526455248, -1526455164, -1526455120, -1526455036, -1526454992, -1526454908, -1526431744, -1526422608, -1526422480, -1526422396, -1526422352, -1526422268, -1526422224, -1526422140, -1523580928, -1523571792, -1523571664, -1523571580, -1523571536, -1523571452, -1523571408, -1523571324, -1523548160, -1523539024, -1523538896, -1523538812, -1523538768, -1523538684, -1523538640, -1523538556, -1518075904, -1518066768, -1518066640, -1518066556, -1518066512, -1518066428, -1518066384, -1518066300, -1518043136, -1518034000, -1518033872, -1518033788, -1518033744, -1518033660, -1518033616, -1518033532, 0, 9152, 9280, 9360, 9408, 9488, 9536, 9616, 32768, 41920, 42048, 42128, 42176, 42256, 42304, 42384, 599785472, 599794624, 599794752, 599794832, 599794880, 599794960, 599795008, 599795088, 599818240, 599827392, 599827520, 599827600, 599827648, 599827728, 599827776, 599827856, 608174080, 608183232, 608183360, 608183440, 608183488, 608183568, 608183616, 608183696, 608206848, 608216000, 608216128, 608216208, 608216256, 608216336, 608216384, 608216464, 613416960, 613426112, 613426240, 613426320, 613426368, 613426448, 613426496, 613426576, 613449728, 613458880, 613459008, 613459088, 613459136, 613459216, 613459264, 613459344, 616562688, 616571840, 616571968, 616572048, 616572096, 616572176, 616572224, 616572304, 616595456, 616604608, 616604736, 616604816, 616604864, 616604944, 616604992, 616605072, 621805568, 621814720, 621814848, 621814928, 621814976, 621815056, 621815104, 621815184, 621838336, 621847488, 621847616, 621847696, 621847744, 621847824, 621847872, 621847952, 624951296, 624960448, 624960576, 624960656, 624960704, 624960784, 624960832, 624960912, 624984064, 624993216, 624993344, 624993424, 624993472, 624993552, 624993600, 624993680, 630194176, 630203328, 630203456, 630203536, 630203584, 630203664, 630203712, 630203792, 630226944, 630236096, 630236224, 630236304, 630236352, 630236432, 630236480, 630236560, -2147483648, -2147474496, -2147474368, -2147474288, -2147474240, -2147474160, -2147474112, -2147474032, -2147450880, -2147441728, -2147441600, -2147441520, -2147441472, -2147441392, -2147441344, -2147441264, -1547698176, -1547689024, -1547688896, -1547688816, -1547688768, -1547688688, -1547688640, -1547688560, -1547665408, -1547656256, -1547656128, -1547656048, -1547656000, -1547655920, -1547655872, -1547655792, -1539309568, -1539300416, -1539300288, -1539300208, -1539300160, -1539300080, -1539300032, -1539299952, -1539276800, -1539267648, -1539267520, -1539267440, -1539267392, -1539267312, -1539267264, -1539267184, -1534066688, -1534057536, -1534057408, -1534057328, -1534057280, -1534057200, -1534057152, -1534057072, -1534033920, -1534024768, -1534024640, -1534024560, -1534024512, -1534024432, -1534024384, -1534024304, -1530920960, -1530911808, -1530911680, -1530911600, -1530911552, -1530911472, -1530911424, -1530911344, -1530888192, -1530879040, -1530878912, -1530878832, -1530878784, -1530878704, -1530878656, -1530878576, -1525678080, -1525668928, -1525668800, -1525668720, -1525668672, -1525668592, -1525668544, -1525668464, -1525645312, -1525636160, -1525636032, -1525635952, -1525635904, -1525635824, -1525635776, -1525635696, -1522532352, -1522523200, -1522523072, -1522522992, -1522522944, -1522522864, -1522522816, -1522522736, -1522499584, -1522490432, -1522490304, -1522490224, -1522490176, -1522490096, -1522490048, -1522489968, -1517289472, -1517280320, -1517280192, -1517280112, -1517280064, -1517279984, -1517279936, -1517279856, -1517256704, -1517247552, -1517247424, -1517247344, -1517247296, -1517247216, -1517247168, -1517247088, 0, 9168, 9296, 9372, 9424, 9500, 9552, 9628, 32768, 41936, 42064, 42140, 42192, 42268, 42320, 42396, 600834048, 600843216, 600843344, 600843420, 600843472, 600843548, 600843600, 600843676, 600866816, 600875984, 600876112, 600876188, 600876240, 600876316, 600876368, 600876444, 609222656, 609231824, 609231952, 609232028, 609232080, 609232156, 609232208, 609232284, 609255424, 609264592, 609264720, 609264796, 609264848, 609264924, 609264976, 609265052, 614203392, 614212560, 614212688, 614212764, 614212816, 614212892, 614212944, 614213020, 614236160, 614245328, 614245456, 614245532, 614245584, 614245660, 614245712, 614245788, 617611264, 617620432, 617620560, 617620636, 617620688, 617620764, 617620816, 617620892, 617644032, 617653200, 617653328, 617653404, 617653456, 617653532, 617653584, 617653660, 622592000, 622601168, 622601296, 622601372, 622601424, 622601500, 622601552, 622601628, 622624768, 622633936, 622634064, 622634140, 622634192, 622634268, 622634320, 622634396, 625999872, 626009040, 626009168, 626009244, 626009296, 626009372, 626009424, 626009500, 626032640, 626041808, 626041936, 626042012, 626042064, 626042140, 626042192, 626042268, 630980608, 630989776, 630989904, 630989980, 630990032, 630990108, 630990160, 630990236, 631013376, 631022544, 631022672, 631022748, 631022800, 631022876, 631022928, 631023004, -2147483648, -2147474480, -2147474352, -2147474276, -2147474224, -2147474148, -2147474096, -2147474020, -2147450880, -2147441712, -2147441584, -2147441508, -2147441456, -2147441380, -2147441328, -2147441252, -1546649600, -1546640432, -1546640304, -1546640228, -1546640176, -1546640100, -1546640048, -1546639972, -1546616832, -1546607664, -1546607536, -1546607460, -1546607408, -1546607332, -1546607280, -1546607204, -1538260992, -1538251824, -1538251696, -1538251620, -1538251568, -1538251492, -1538251440, -1538251364, -1538228224, -1538219056, -1538218928, -1538218852, -1538218800, -1538218724, -1538218672, -1538218596, -1533280256, -1533271088, -1533270960, -1533270884, -1533270832, -1533270756, -1533270704, -1533270628, -1533247488, -1533238320, -1533238192, -1533238116, -1533238064, -1533237988, -1533237936, -1533237860, -1529872384, -1529863216, -1529863088, -1529863012, -1529862960, -1529862884, -1529862832, -1529862756, -1529839616, -1529830448, -1529830320, -1529830244, -1529830192, -1529830116, -1529830064, -1529829988, -1524891648, -1524882480, -1524882352, -1524882276, -1524882224, -1524882148, -1524882096, -1524882020, -1524858880, -1524849712, -1524849584, -1524849508, -1524849456, -1524849380, -1524849328, -1524849252, -1521483776, -1521474608, -1521474480, -1521474404, -1521474352, -1521474276, -1521474224, -1521474148, -1521451008, -1521441840, -1521441712, -1521441636, -1521441584, -1521441508, -1521441456, -1521441380, -1516503040, -1516493872, -1516493744, -1516493668, -1516493616, -1516493540, -1516493488, -1516493412, -1516470272, -1516461104, -1516460976, -1516460900, -1516460848, -1516460772, -1516460720, -1516460644, 0, 9184, 9312, 9384, 9440, 9512, 9568, 9640, 32768, 41952, 42080, 42152, 42208, 42280, 42336, 42408, 601882624, 601891808, 601891936, 601892008, 601892064, 601892136, 601892192, 601892264, 601915392, 601924576, 601924704, 601924776, 601924832, 601924904, 601924960, 601925032, 610271232, 610280416, 610280544, 610280616, 610280672, 610280744, 610280800, 610280872, 610304000, 610313184, 610313312, 610313384, 610313440, 610313512, 610313568, 610313640, 614989824, 614999008, 614999136, 614999208, 614999264, 614999336, 614999392, 614999464, 615022592, 615031776, 615031904, 615031976, 615032032, 615032104, 615032160, 615032232, 618659840, 618669024, 618669152, 618669224, 618669280, 618669352, 618669408, 618669480, 618692608, 618701792, 618701920, 618701992, 618702048, 618702120, 618702176, 618702248, 623378432, 623387616, 623387744, 623387816, 623387872, 623387944, 623388000, 623388072, 623411200, 623420384, 623420512, 623420584, 623420640, 623420712, 623420768, 623420840, 627048448, 627057632, 627057760, 627057832, 627057888, 627057960, 627058016, 627058088, 627081216, 627090400, 627090528, 627090600, 627090656, 627090728, 627090784, 627090856, 631767040, 631776224, 631776352, 631776424, 631776480, 631776552, 631776608, 631776680, 631799808, 631808992, 631809120, 631809192, 631809248, 631809320, 631809376, 631809448, -2147483648, -2147474464, -2147474336, -2147474264, -2147474208, -2147474136, -2147474080, -2147474008, -2147450880, -2147441696, -2147441568, -2147441496, -2147441440, -2147441368, -2147441312, -2147441240, -1545601024, -1545591840, -1545591712, -1545591640, -1545591584, -1545591512, -1545591456, -1545591384, -1545568256, -1545559072, -1545558944, -1545558872, -1545558816, -1545558744, -1545558688, -1545558616, -1537212416, -1537203232, -1537203104, -1537203032, -1537202976, -1537202904, -1537202848, -1537202776, -1537179648, -1537170464, -1537170336, -1537170264, -1537170208, -1537170136, -1537170080, -1537170008, -1532493824, -1532484640, -1532484512, -1532484440, -1532484384, -1532484312, -1532484256, -1532484184, -1532461056, -1532451872, -1532451744, -1532451672, -1532451616, -1532451544, -1532451488, -1532451416, -1528823808, -1528814624, -1528814496, -1528814424, -1528814368, -1528814296, -1528814240, -1528814168, -1528791040, -1528781856, -1528781728, -1528781656, -1528781600, -1528781528, -1528781472, -1528781400, -1524105216, -1524096032, -1524095904, -1524095832, -1524095776, -1524095704, -1524095648, -1524095576, -1524072448, -1524063264, -1524063136, -1524063064, -1524063008, -1524062936, -1524062880, -1524062808, -1520435200, -1520426016, -1520425888, -1520425816, -1520425760, -1520425688, -1520425632, -1520425560, -1520402432, -1520393248, -1520393120, -1520393048, -1520392992, -1520392920, -1520392864, -1520392792, -1515716608, -1515707424, -1515707296, -1515707224, -1515707168, -1515707096, -1515707040, -1515706968, -1515683840, -1515674656, -1515674528, -1515674456, -1515674400, -1515674328, -1515674272, -1515674200, 0, 9200, 9328, 9396, 9456, 9524, 9584, 9652, 32768, 41968, 42096, 42164, 42224, 42292, 42352, 42420, 602931200, 602940400, 602940528, 602940596, 602940656, 602940724, 602940784, 602940852, 602963968, 602973168, 602973296, 602973364, 602973424, 602973492, 602973552, 602973620, 611319808, 611329008, 611329136, 611329204, 611329264, 611329332, 611329392, 611329460, 611352576, 611361776, 611361904, 611361972, 611362032, 611362100, 611362160, 611362228, 615776256, 615785456, 615785584, 615785652, 615785712, 615785780, 615785840, 615785908, 615809024, 615818224, 615818352, 615818420, 615818480, 615818548, 615818608, 615818676, 619708416, 619717616, 619717744, 619717812, 619717872, 619717940, 619718000, 619718068, 619741184, 619750384, 619750512, 619750580, 619750640, 619750708, 619750768, 619750836, 624164864, 624174064, 624174192, 624174260, 624174320, 624174388, 624174448, 624174516, 624197632, 624206832, 624206960, 624207028, 624207088, 624207156, 624207216, 624207284, 628097024, 628106224, 628106352, 628106420, 628106480, 628106548, 628106608, 628106676, 628129792, 628138992, 628139120, 628139188, 628139248, 628139316, 628139376, 628139444, 632553472, 632562672, 632562800, 632562868, 632562928, 632562996, 632563056, 632563124, 632586240, 632595440, 632595568, 632595636, 632595696, 632595764, 632595824, 632595892, -2147483648, -2147474448, -2147474320, -2147474252, -2147474192, -2147474124, -2147474064, -2147473996, -2147450880, -2147441680, -2147441552, -2147441484, -2147441424, -2147441356, -2147441296, -2147441228, -1544552448, -1544543248, -1544543120, -1544543052, -1544542992, -1544542924, -1544542864, -1544542796, -1544519680, -1544510480, -1544510352, -1544510284, -1544510224, -1544510156, -1544510096, -1544510028, -1536163840, -1536154640, -1536154512, -1536154444, -1536154384, -1536154316, -1536154256, -1536154188, -1536131072, -1536121872, -1536121744, -1536121676, -1536121616, -1536121548, -1536121488, -1536121420, -1531707392, -1531698192, -1531698064, -1531697996, -1531697936, -1531697868, -1531697808, -1531697740, -1531674624, -1531665424, -1531665296, -1531665228, -1531665168, -1531665100, -1531665040, -1531664972, -1527775232, -1527766032, -1527765904, -1527765836, -1527765776, -1527765708, -1527765648, -1527765580, -1527742464, -1527733264, -1527733136, -1527733068, -1527733008, -1527732940, -1527732880, -1527732812, -1523318784, -1523309584, -1523309456, -1523309388, -1523309328, -1523309260, -1523309200, -1523309132, -1523286016, -1523276816, -1523276688, -1523276620, -1523276560, -1523276492, -1523276432, -1523276364, -1519386624, -1519377424, -1519377296, -1519377228, -1519377168, -1519377100, -1519377040, -1519376972, -1519353856, -1519344656, -1519344528, -1519344460, -1519344400, -1519344332, -1519344272, -1519344204, -1514930176, -1514920976, -1514920848, -1514920780, -1514920720, -1514920652, -1514920592, -1514920524, -1514897408, -1514888208, -1514888080, -1514888012, -1514887952, -1514887884, -1514887824, -1514887756, 0, 9216, 9344, 9408, 9472, 9536, 9600, 9664, 32768, 41984, 42112, 42176, 42240, 42304, 42368, 42432, 603979776, 603988992, 603989120, 603989184, 603989248, 603989312, 603989376, 603989440, 604012544, 604021760, 604021888, 604021952, 604022016, 604022080, 604022144, 604022208, 612368384, 612377600, 612377728, 612377792, 612377856, 612377920, 612377984, 612378048, 612401152, 612410368, 612410496, 612410560, 612410624, 612410688, 612410752, 612410816, 616562688, 616571904, 616572032, 616572096, 616572160, 616572224, 616572288, 616572352, 616595456, 616604672, 616604800, 616604864, 616604928, 616604992, 616605056, 616605120, 620756992, 620766208, 620766336, 620766400, 620766464, 620766528, 620766592, 620766656, 620789760, 620798976, 620799104, 620799168, 620799232, 620799296, 620799360, 620799424, 624951296, 624960512, 624960640, 624960704, 624960768, 624960832, 624960896, 624960960, 624984064, 624993280, 624993408, 624993472, 624993536, 624993600, 624993664, 624993728, 629145600, 629154816, 629154944, 629155008, 629155072, 629155136, 629155200, 629155264, 629178368, 629187584, 629187712, 629187776, 629187840, 629187904, 629187968, 629188032, 633339904, 633349120, 633349248, 633349312, 633349376, 633349440, 633349504, 633349568, 633372672, 633381888, 633382016, 633382080, 633382144, 633382208, 633382272, 633382336, -2147483648, -2147474432, -2147474304, -2147474240, -2147474176, -2147474112, -2147474048, -2147473984, -2147450880, -2147441664, -2147441536, -2147441472, -2147441408, -2147441344, -2147441280, -2147441216, -1543503872, -1543494656, -1543494528, -1543494464, -1543494400, -1543494336, -1543494272, -1543494208, -1543471104, -1543461888, -1543461760, -1543461696, -1543461632, -1543461568, -1543461504, -1543461440, -1535115264, -1535106048, -1535105920, -1535105856, -1535105792, -1535105728, -1535105664, -1535105600, -1535082496, -1535073280, -1535073152, -1535073088, -1535073024, -1535072960, -1535072896, -1535072832, -1530920960, -1530911744, -1530911616, -1530911552, -1530911488, -1530911424, -1530911360, -1530911296, -1530888192, -1530878976, -1530878848, -1530878784, -1530878720, -1530878656, -1530878592, -1530878528, -1526726656, -1526717440, -1526717312, -1526717248, -1526717184, -1526717120, -1526717056, -1526716992, -1526693888, -1526684672, -1526684544, -1526684480, -1526684416, -1526684352, -1526684288, -1526684224, -1522532352, -1522523136, -1522523008, -1522522944, -1522522880, -1522522816, -1522522752, -1522522688, -1522499584, -1522490368, -1522490240, -1522490176, -1522490112, -1522490048, -1522489984, -1522489920, -1518338048, -1518328832, -1518328704, -1518328640, -1518328576, -1518328512, -1518328448, -1518328384, -1518305280, -1518296064, -1518295936, -1518295872, -1518295808, -1518295744, -1518295680, -1518295616, -1514143744, -1514134528, -1514134400, -1514134336, -1514134272, -1514134208, -1514134144, -1514134080, -1514110976, -1514101760, -1514101632, -1514101568, -1514101504, -1514101440, -1514101376, -1514101312, 0, 9232, 9360, 9432, 9488, 9560, 9616, 9688, 32768, 42000, 42128, 42200, 42256, 42328, 42384, 42456, 605028352, 605037584, 605037712, 605037784, 605037840, 605037912, 605037968, 605038040, 605061120, 605070352, 605070480, 605070552, 605070608, 605070680, 605070736, 605070808, 613416960, 613426192, 613426320, 613426392, 613426448, 613426520, 613426576, 613426648, 613449728, 613458960, 613459088, 613459160, 613459216, 613459288, 613459344, 613459416, 618135552, 618144784, 618144912, 618144984, 618145040, 618145112, 618145168, 618145240, 618168320, 618177552, 618177680, 618177752, 618177808, 618177880, 618177936, 618178008, 621805568, 621814800, 621814928, 621815000, 621815056, 621815128, 621815184, 621815256, 621838336, 621847568, 621847696, 621847768, 621847824, 621847896, 621847952, 621848024, 626524160, 626533392, 626533520, 626533592, 626533648, 626533720, 626533776, 626533848, 626556928, 626566160, 626566288, 626566360, 626566416, 626566488, 626566544, 626566616, 630194176, 630203408, 630203536, 630203608, 630203664, 630203736, 630203792, 630203864, 630226944, 630236176, 630236304, 630236376, 630236432, 630236504, 630236560, 630236632, 634912768, 634922000, 634922128, 634922200, 634922256, 634922328, 634922384, 634922456, 634945536, 634954768, 634954896, 634954968, 634955024, 634955096, 634955152, 634955224, -2147483648, -2147474416, -2147474288, -2147474216, -2147474160, -2147474088, -2147474032, -2147473960, -2147450880, -2147441648, -2147441520, -2147441448, -2147441392, -2147441320, -2147441264, -2147441192, -1542455296, -1542446064, -1542445936, -1542445864, -1542445808, -1542445736, -1542445680, -1542445608, -1542422528, -1542413296, -1542413168, -1542413096, -1542413040, -1542412968, -1542412912, -1542412840, -1534066688, -1534057456, -1534057328, -1534057256, -1534057200, -1534057128, -1534057072, -1534057000, -1534033920, -1534024688, -1534024560, -1534024488, -1534024432, -1534024360, -1534024304, -1534024232, -1529348096, -1529338864, -1529338736, -1529338664, -1529338608, -1529338536, -1529338480, -1529338408, -1529315328, -1529306096, -1529305968, -1529305896, -1529305840, -1529305768, -1529305712, -1529305640, -1525678080, -1525668848, -1525668720, -1525668648, -1525668592, -1525668520, -1525668464, -1525668392, -1525645312, -1525636080, -1525635952, -1525635880, -1525635824, -1525635752, -1525635696, -1525635624, -1520959488, -1520950256, -1520950128, -1520950056, -1520950000, -1520949928, -1520949872, -1520949800, -1520926720, -1520917488, -1520917360, -1520917288, -1520917232, -1520917160, -1520917104, -1520917032, -1517289472, -1517280240, -1517280112, -1517280040, -1517279984, -1517279912, -1517279856, -1517279784, -1517256704, -1517247472, -1517247344, -1517247272, -1517247216, -1517247144, -1517247088, -1517247016, -1512570880, -1512561648, -1512561520, -1512561448, -1512561392, -1512561320, -1512561264, -1512561192, -1512538112, -1512528880, -1512528752, -1512528680, -1512528624, -1512528552, -1512528496, -1512528424, 0, 9248, 9376, 9456, 9504, 9584, 9632, 9712, 32768, 42016, 42144, 42224, 42272, 42352, 42400, 42480, 606076928, 606086176, 606086304, 606086384, 606086432, 606086512, 606086560, 606086640, 606109696, 606118944, 606119072, 606119152, 606119200, 606119280, 606119328, 606119408, 614465536, 614474784, 614474912, 614474992, 614475040, 614475120, 614475168, 614475248, 614498304, 614507552, 614507680, 614507760, 614507808, 614507888, 614507936, 614508016, 619708416, 619717664, 619717792, 619717872, 619717920, 619718000, 619718048, 619718128, 619741184, 619750432, 619750560, 619750640, 619750688, 619750768, 619750816, 619750896, 622854144, 622863392, 622863520, 622863600, 622863648, 622863728, 622863776, 622863856, 622886912, 622896160, 622896288, 622896368, 622896416, 622896496, 622896544, 622896624, 628097024, 628106272, 628106400, 628106480, 628106528, 628106608, 628106656, 628106736, 628129792, 628139040, 628139168, 628139248, 628139296, 628139376, 628139424, 628139504, 631242752, 631252000, 631252128, 631252208, 631252256, 631252336, 631252384, 631252464, 631275520, 631284768, 631284896, 631284976, 631285024, 631285104, 631285152, 631285232, 636485632, 636494880, 636495008, 636495088, 636495136, 636495216, 636495264, 636495344, 636518400, 636527648, 636527776, 636527856, 636527904, 636527984, 636528032, 636528112, -2147483648, -2147474400, -2147474272, -2147474192, -2147474144, -2147474064, -2147474016, -2147473936, -2147450880, -2147441632, -2147441504, -2147441424, -2147441376, -2147441296, -2147441248, -2147441168, -1541406720, -1541397472, -1541397344, -1541397264, -1541397216, -1541397136, -1541397088, -1541397008, -1541373952, -1541364704, -1541364576, -1541364496, -1541364448, -1541364368, -1541364320, -1541364240, -1533018112, -1533008864, -1533008736, -1533008656, -1533008608, -1533008528, -1533008480, -1533008400, -1532985344, -1532976096, -1532975968, -1532975888, -1532975840, -1532975760, -1532975712, -1532975632, -1527775232, -1527765984, -1527765856, -1527765776, -1527765728, -1527765648, -1527765600, -1527765520, -1527742464, -1527733216, -1527733088, -1527733008, -1527732960, -1527732880, -1527732832, -1527732752, -1524629504, -1524620256, -1524620128, -1524620048, -1524620000, -1524619920, -1524619872, -1524619792, -1524596736, -1524587488, -1524587360, -1524587280, -1524587232, -1524587152, -1524587104, -1524587024, -1519386624, -1519377376, -1519377248, -1519377168, -1519377120, -1519377040, -1519376992, -1519376912, -1519353856, -1519344608, -1519344480, -1519344400, -1519344352, -1519344272, -1519344224, -1519344144, -1516240896, -1516231648, -1516231520, -1516231440, -1516231392, -1516231312, -1516231264, -1516231184, -1516208128, -1516198880, -1516198752, -1516198672, -1516198624, -1516198544, -1516198496, -1516198416, -1510998016, -1510988768, -1510988640, -1510988560, -1510988512, -1510988432, -1510988384, -1510988304, -1510965248, -1510956000, -1510955872, -1510955792, -1510955744, -1510955664, -1510955616, -1510955536, 0, 9264, 9392, 9476, 9520, 9604, 9648, 9732, 32768, 42032, 42160, 42244, 42288, 42372, 42416, 42500, 607125504, 607134768, 607134896, 607134980, 607135024, 607135108, 607135152, 607135236, 607158272, 607167536, 607167664, 607167748, 607167792, 607167876, 607167920, 607168004, 615514112, 615523376, 615523504, 615523588, 615523632, 615523716, 615523760, 615523844, 615546880, 615556144, 615556272, 615556356, 615556400, 615556484, 615556528, 615556612, 621019136, 621028400, 621028528, 621028612, 621028656, 621028740, 621028784, 621028868, 621051904, 621061168, 621061296, 621061380, 621061424, 621061508, 621061552, 621061636, 623902720, 623911984, 623912112, 623912196, 623912240, 623912324, 623912368, 623912452, 623935488, 623944752, 623944880, 623944964, 623945008, 623945092, 623945136, 623945220, 629407744, 629417008, 629417136, 629417220, 629417264, 629417348, 629417392, 629417476, 629440512, 629449776, 629449904, 629449988, 629450032, 629450116, 629450160, 629450244, 632291328, 632300592, 632300720, 632300804, 632300848, 632300932, 632300976, 632301060, 632324096, 632333360, 632333488, 632333572, 632333616, 632333700, 632333744, 632333828, 637796352, 637805616, 637805744, 637805828, 637805872, 637805956, 637806000, 637806084, 637829120, 637838384, 637838512, 637838596, 637838640, 637838724, 637838768, 637838852, -2147483648, -2147474384, -2147474256, -2147474172, -2147474128, -2147474044, -2147474000, -2147473916, -2147450880, -2147441616, -2147441488, -2147441404, -2147441360, -2147441276, -2147441232, -2147441148, -1540358144, -1540348880, -1540348752, -1540348668, -1540348624, -1540348540, -1540348496, -1540348412, -1540325376, -1540316112, -1540315984, -1540315900, -1540315856, -1540315772, -1540315728, -1540315644, -1531969536, -1531960272, -1531960144, -1531960060, -1531960016, -1531959932, -1531959888, -1531959804, -1531936768, -1531927504, -1531927376, -1531927292, -1531927248, -1531927164, -1531927120, -1531927036, -1526464512, -1526455248, -1526455120, -1526455036, -1526454992, -1526454908, -1526454864, -1526454780, -1526431744, -1526422480, -1526422352, -1526422268, -1526422224, -1526422140, -1526422096, -1526422012, -1523580928, -1523571664, -1523571536, -1523571452, -1523571408, -1523571324, -1523571280, -1523571196, -1523548160, -1523538896, -1523538768, -1523538684, -1523538640, -1523538556, -1523538512, -1523538428, -1518075904, -1518066640, -1518066512, -1518066428, -1518066384, -1518066300, -1518066256, -1518066172, -1518043136, -1518033872, -1518033744, -1518033660, -1518033616, -1518033532, -1518033488, -1518033404, -1515192320, -1515183056, -1515182928, -1515182844, -1515182800, -1515182716, -1515182672, -1515182588, -1515159552, -1515150288, -1515150160, -1515150076, -1515150032, -1515149948, -1515149904, -1515149820, -1509687296, -1509678032, -1509677904, -1509677820, -1509677776, -1509677692, -1509677648, -1509677564, -1509654528, -1509645264, -1509645136, -1509645052, -1509645008, -1509644924, -1509644880, -1509644796, 0, 9280, 9408, 9488, 9536, 9616, 9664, 9744, 32768, 42048, 42176, 42256, 42304, 42384, 42432, 42512, 608174080, 608183360, 608183488, 608183568, 608183616, 608183696, 608183744, 608183824, 608206848, 608216128, 608216256, 608216336, 608216384, 608216464, 608216512, 608216592, 616562688, 616571968, 616572096, 616572176, 616572224, 616572304, 616572352, 616572432, 616595456, 616604736, 616604864, 616604944, 616604992, 616605072, 616605120, 616605200, 621805568, 621814848, 621814976, 621815056, 621815104, 621815184, 621815232, 621815312, 621838336, 621847616, 621847744, 621847824, 621847872, 621847952, 621848000, 621848080, 624951296, 624960576, 624960704, 624960784, 624960832, 624960912, 624960960, 624961040, 624984064, 624993344, 624993472, 624993552, 624993600, 624993680, 624993728, 624993808, 630194176, 630203456, 630203584, 630203664, 630203712, 630203792, 630203840, 630203920, 630226944, 630236224, 630236352, 630236432, 630236480, 630236560, 630236608, 630236688, 633339904, 633349184, 633349312, 633349392, 633349440, 633349520, 633349568, 633349648, 633372672, 633381952, 633382080, 633382160, 633382208, 633382288, 633382336, 633382416, 638582784, 638592064, 638592192, 638592272, 638592320, 638592400, 638592448, 638592528, 638615552, 638624832, 638624960, 638625040, 638625088, 638625168, 638625216, 638625296, -2147483648, -2147474368, -2147474240, -2147474160, -2147474112, -2147474032, -2147473984, -2147473904, -2147450880, -2147441600, -2147441472, -2147441392, -2147441344, -2147441264, -2147441216, -2147441136, -1539309568, -1539300288, -1539300160, -1539300080, -1539300032, -1539299952, -1539299904, -1539299824, -1539276800, -1539267520, -1539267392, -1539267312, -1539267264, -1539267184, -1539267136, -1539267056, -1530920960, -1530911680, -1530911552, -1530911472, -1530911424, -1530911344, -1530911296, -1530911216, -1530888192, -1530878912, -1530878784, -1530878704, -1530878656, -1530878576, -1530878528, -1530878448, -1525678080, -1525668800, -1525668672, -1525668592, -1525668544, -1525668464, -1525668416, -1525668336, -1525645312, -1525636032, -1525635904, -1525635824, -1525635776, -1525635696, -1525635648, -1525635568, -1522532352, -1522523072, -1522522944, -1522522864, -1522522816, -1522522736, -1522522688, -1522522608, -1522499584, -1522490304, -1522490176, -1522490096, -1522490048, -1522489968, -1522489920, -1522489840, -1517289472, -1517280192, -1517280064, -1517279984, -1517279936, -1517279856, -1517279808, -1517279728, -1517256704, -1517247424, -1517247296, -1517247216, -1517247168, -1517247088, -1517247040, -1517246960, -1514143744, -1514134464, -1514134336, -1514134256, -1514134208, -1514134128, -1514134080, -1514134000, -1514110976, -1514101696, -1514101568, -1514101488, -1514101440, -1514101360, -1514101312, -1514101232, -1508900864, -1508891584, -1508891456, -1508891376, -1508891328, -1508891248, -1508891200, -1508891120, -1508868096, -1508858816, -1508858688, -1508858608, -1508858560, -1508858480, -1508858432, -1508858352, 0, 9296, 9424, 9500, 9552, 9628, 9680, 9756, 32768, 42064, 42192, 42268, 42320, 42396, 42448, 42524, 609222656, 609231952, 609232080, 609232156, 609232208, 609232284, 609232336, 609232412, 609255424, 609264720, 609264848, 609264924, 609264976, 609265052, 609265104, 609265180, 617611264, 617620560, 617620688, 617620764, 617620816, 617620892, 617620944, 617621020, 617644032, 617653328, 617653456, 617653532, 617653584, 617653660, 617653712, 617653788, 622592000, 622601296, 622601424, 622601500, 622601552, 622601628, 622601680, 622601756, 622624768, 622634064, 622634192, 622634268, 622634320, 622634396, 622634448, 622634524, 625999872, 626009168, 626009296, 626009372, 626009424, 626009500, 626009552, 626009628, 626032640, 626041936, 626042064, 626042140, 626042192, 626042268, 626042320, 626042396, 630980608, 630989904, 630990032, 630990108, 630990160, 630990236, 630990288, 630990364, 631013376, 631022672, 631022800, 631022876, 631022928, 631023004, 631023056, 631023132, 634388480, 634397776, 634397904, 634397980, 634398032, 634398108, 634398160, 634398236, 634421248, 634430544, 634430672, 634430748, 634430800, 634430876, 634430928, 634431004, 639369216, 639378512, 639378640, 639378716, 639378768, 639378844, 639378896, 639378972, 639401984, 639411280, 639411408, 639411484, 639411536, 639411612, 639411664, 639411740, -2147483648, -2147474352, -2147474224, -2147474148, -2147474096, -2147474020, -2147473968, -2147473892, -2147450880, -2147441584, -2147441456, -2147441380, -2147441328, -2147441252, -2147441200, -2147441124, -1538260992, -1538251696, -1538251568, -1538251492, -1538251440, -1538251364, -1538251312, -1538251236, -1538228224, -1538218928, -1538218800, -1538218724, -1538218672, -1538218596, -1538218544, -1538218468, -1529872384, -1529863088, -1529862960, -1529862884, -1529862832, -1529862756, -1529862704, -1529862628, -1529839616, -1529830320, -1529830192, -1529830116, -1529830064, -1529829988, -1529829936, -1529829860, -1524891648, -1524882352, -1524882224, -1524882148, -1524882096, -1524882020, -1524881968, -1524881892, -1524858880, -1524849584, -1524849456, -1524849380, -1524849328, -1524849252, -1524849200, -1524849124, -1521483776, -1521474480, -1521474352, -1521474276, -1521474224, -1521474148, -1521474096, -1521474020, -1521451008, -1521441712, -1521441584, -1521441508, -1521441456, -1521441380, -1521441328, -1521441252, -1516503040, -1516493744, -1516493616, -1516493540, -1516493488, -1516493412, -1516493360, -1516493284, -1516470272, -1516460976, -1516460848, -1516460772, -1516460720, -1516460644, -1516460592, -1516460516, -1513095168, -1513085872, -1513085744, -1513085668, -1513085616, -1513085540, -1513085488, -1513085412, -1513062400, -1513053104, -1513052976, -1513052900, -1513052848, -1513052772, -1513052720, -1513052644, -1508114432, -1508105136, -1508105008, -1508104932, -1508104880, -1508104804, -1508104752, -1508104676, -1508081664, -1508072368, -1508072240, -1508072164, -1508072112, -1508072036, -1508071984, -1508071908, 0, 9312, 9440, 9512, 9568, 9640, 9696, 9768, 32768, 42080, 42208, 42280, 42336, 42408, 42464, 42536, 610271232, 610280544, 610280672, 610280744, 610280800, 610280872, 610280928, 610281000, 610304000, 610313312, 610313440, 610313512, 610313568, 610313640, 610313696, 610313768, 618659840, 618669152, 618669280, 618669352, 618669408, 618669480, 618669536, 618669608, 618692608, 618701920, 618702048, 618702120, 618702176, 618702248, 618702304, 618702376, 623378432, 623387744, 623387872, 623387944, 623388000, 623388072, 623388128, 623388200, 623411200, 623420512, 623420640, 623420712, 623420768, 623420840, 623420896, 623420968, 627048448, 627057760, 627057888, 627057960, 627058016, 627058088, 627058144, 627058216, 627081216, 627090528, 627090656, 627090728, 627090784, 627090856, 627090912, 627090984, 631767040, 631776352, 631776480, 631776552, 631776608, 631776680, 631776736, 631776808, 631799808, 631809120, 631809248, 631809320, 631809376, 631809448, 631809504, 631809576, 635437056, 635446368, 635446496, 635446568, 635446624, 635446696, 635446752, 635446824, 635469824, 635479136, 635479264, 635479336, 635479392, 635479464, 635479520, 635479592, 640155648, 640164960, 640165088, 640165160, 640165216, 640165288, 640165344, 640165416, 640188416, 640197728, 640197856, 640197928, 640197984, 640198056, 640198112, 640198184, -2147483648, -2147474336, -2147474208, -2147474136, -2147474080, -2147474008, -2147473952, -2147473880, -2147450880, -2147441568, -2147441440, -2147441368, -2147441312, -2147441240, -2147441184, -2147441112, -1537212416, -1537203104, -1537202976, -1537202904, -1537202848, -1537202776, -1537202720, -1537202648, -1537179648, -1537170336, -1537170208, -1537170136, -1537170080, -1537170008, -1537169952, -1537169880, -1528823808, -1528814496, -1528814368, -1528814296, -1528814240, -1528814168, -1528814112, -1528814040, -1528791040, -1528781728, -1528781600, -1528781528, -1528781472, -1528781400, -1528781344, -1528781272, -1524105216, -1524095904, -1524095776, -1524095704, -1524095648, -1524095576, -1524095520, -1524095448, -1524072448, -1524063136, -1524063008, -1524062936, -1524062880, -1524062808, -1524062752, -1524062680, -1520435200, -1520425888, -1520425760, -1520425688, -1520425632, -1520425560, -1520425504, -1520425432, -1520402432, -1520393120, -1520392992, -1520392920, -1520392864, -1520392792, -1520392736, -1520392664, -1515716608, -1515707296, -1515707168, -1515707096, -1515707040, -1515706968, -1515706912, -1515706840, -1515683840, -1515674528, -1515674400, -1515674328, -1515674272, -1515674200, -1515674144, -1515674072, -1512046592, -1512037280, -1512037152, -1512037080, -1512037024, -1512036952, -1512036896, -1512036824, -1512013824, -1512004512, -1512004384, -1512004312, -1512004256, -1512004184, -1512004128, -1512004056, -1507328000, -1507318688, -1507318560, -1507318488, -1507318432, -1507318360, -1507318304, -1507318232, -1507295232, -1507285920, -1507285792, -1507285720, -1507285664, -1507285592, -1507285536, -1507285464, 0, 9328, 9456, 9524, 9584, 9652, 9712, 9780, 32768, 42096, 42224, 42292, 42352, 42420, 42480, 42548, 611319808, 611329136, 611329264, 611329332, 611329392, 611329460, 611329520, 611329588, 611352576, 611361904, 611362032, 611362100, 611362160, 611362228, 611362288, 611362356, 619708416, 619717744, 619717872, 619717940, 619718000, 619718068, 619718128, 619718196, 619741184, 619750512, 619750640, 619750708, 619750768, 619750836, 619750896, 619750964, 624164864, 624174192, 624174320, 624174388, 624174448, 624174516, 624174576, 624174644, 624197632, 624206960, 624207088, 624207156, 624207216, 624207284, 624207344, 624207412, 628097024, 628106352, 628106480, 628106548, 628106608, 628106676, 628106736, 628106804, 628129792, 628139120, 628139248, 628139316, 628139376, 628139444, 628139504, 628139572, 632553472, 632562800, 632562928, 632562996, 632563056, 632563124, 632563184, 632563252, 632586240, 632595568, 632595696, 632595764, 632595824, 632595892, 632595952, 632596020, 636485632, 636494960, 636495088, 636495156, 636495216, 636495284, 636495344, 636495412, 636518400, 636527728, 636527856, 636527924, 636527984, 636528052, 636528112, 636528180, 640942080, 640951408, 640951536, 640951604, 640951664, 640951732, 640951792, 640951860, 640974848, 640984176, 640984304, 640984372, 640984432, 640984500, 640984560, 640984628, -2147483648, -2147474320, -2147474192, -2147474124, -2147474064, -2147473996, -2147473936, -2147473868, -2147450880, -2147441552, -2147441424, -2147441356, -2147441296, -2147441228, -2147441168, -2147441100, -1536163840, -1536154512, -1536154384, -1536154316, -1536154256, -1536154188, -1536154128, -1536154060, -1536131072, -1536121744, -1536121616, -1536121548, -1536121488, -1536121420, -1536121360, -1536121292, -1527775232, -1527765904, -1527765776, -1527765708, -1527765648, -1527765580, -1527765520, -1527765452, -1527742464, -1527733136, -1527733008, -1527732940, -1527732880, -1527732812, -1527732752, -1527732684, -1523318784, -1523309456, -1523309328, -1523309260, -1523309200, -1523309132, -1523309072, -1523309004, -1523286016, -1523276688, -1523276560, -1523276492, -1523276432, -1523276364, -1523276304, -1523276236, -1519386624, -1519377296, -1519377168, -1519377100, -1519377040, -1519376972, -1519376912, -1519376844, -1519353856, -1519344528, -1519344400, -1519344332, -1519344272, -1519344204, -1519344144, -1519344076, -1514930176, -1514920848, -1514920720, -1514920652, -1514920592, -1514920524, -1514920464, -1514920396, -1514897408, -1514888080, -1514887952, -1514887884, -1514887824, -1514887756, -1514887696, -1514887628, -1510998016, -1510988688, -1510988560, -1510988492, -1510988432, -1510988364, -1510988304, -1510988236, -1510965248, -1510955920, -1510955792, -1510955724, -1510955664, -1510955596, -1510955536, -1510955468, -1506541568, -1506532240, -1506532112, -1506532044, -1506531984, -1506531916, -1506531856, -1506531788, -1506508800, -1506499472, -1506499344, -1506499276, -1506499216, -1506499148, -1506499088, -1506499020)

D_HIDDEN = D_EXPERT = 4096
N_EXPERTS = 256
ROUTED_PER_TOKEN = 8
POPULARITY = .03
LOCAL = {"ep8": (0, 32), "full": (0, 256)}

def outlier_channels(x: torch.Tensor, *, generator: torch.Generator, frac: float = 0.01,
                     scale: float = 30.0) -> torch.Tensor:
    """x with a few channels (last dim) scaled up, like the outlier features of real LLM
    activations."""
    k = x.shape[-1]
    n = max(1, int(k * frac))
    idx = torch.randperm(k, generator=generator, device=x.device)[:n]
    out = x.clone()
    out[..., idx] = (out[..., idx].float() * scale).to(x.dtype)
    return out

def generate_input(bs: int, sharding: str, seed: int, dist: str = "normal", rank: str = "busiest",
                   device: str = "cuda", **_):
    """hidden_states [bs, D_HIDDEN] bf16; topk_ids [bs, 8] int32 (global expert ids);
    topk_weights [bs, 8] fp32, summing to 1 per token.

    dist="normal": top-8 of uniform scores plus a heavy-tailed per-expert popularity.
    dist="skewed": a few hot experts take most tokens, plus outlier activation channels.
    rank="busiest" (ep8): expert ids are rotated by multiples of 32 so experts 0-31 have the
    most pairs of any GPU."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    if dist == "normal":  # per-expert popularity, heavy-tailed
        z = torch.randn((N_EXPERTS,), generator=gen, device=device)
        u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + POPULARITY * z / (-u.log()).sqrt()  # Student-t, 2 degrees of freedom
    if dist == "skewed":
        heat = torch.rand((N_EXPERTS,), generator=gen, device=device).pow(6) * 4
        scores = scores + heat
        x = outlier_channels(x, generator=gen)
    routed = scores.topk(ROUTED_PER_TOKEN, dim=-1)
    start, count = LOCAL[sharding]
    ids = routed.indices
    if rank == "busiest" and count < N_EXPERTS:
        pairs = torch.bincount((ids // count).flatten(), minlength=N_EXPERTS // count)
        ids = (ids - count * pairs.argmax()) % N_EXPERTS  # busiest GPU's experts become 0-31
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": D_HIDDEN, "d_expert": D_EXPERT, "n_experts": N_EXPERTS,
                   "local_start": start, "local_count": count, "top_k": ROUTED_PER_TOKEN, "sharding": sharding},
    }
"""Frozen standalone gates, embedded verbatim in the payload."""
import math
import statistics

BANKS = (1102031, 1102037, 1102049)
LOADS = (1, 8, 16, 32, 128, 256)
TOKENS = (1, 4, 8, 16, 32, 33, 128, 256, 257)
DISTRIBUTIONS = ('normal', 'skewed')
CROSSES = ((0, 2), (0, 3), (1, 2), (1, 3))
GLOBAL_TOL = .015
ROW_TOL = .075
REPEATS = 8
CAPSULE_LIMIT = 14000


def require(condition, message):
    if not condition:
        raise RuntimeError('EXPERT_MOE_HARNESS: ' + message)


def positive(value):
    require(type(value) in (float, int) and math.isfinite(value) and value > 0, 'positive finite timing')
    return value


def paired_samples(samples):
    require(len(samples) == 4 and all(len(sample) == REPEATS for sample in samples), 'paired sample count')
    for sample in samples:
        for value in sample:
            positive(value)

    def paired_logs(left, right):
        return [(math.log(samples[left][index] / samples[right][index])
                 + math.log(samples[left][index + 1] / samples[right][index + 1])) / 2
                for index in range(0, REPEATS, 2)]
    ratios = [math.exp(statistics.median(paired_logs(left, right))) for left, right in CROSSES]
    nulls = [statistics.mean(abs(value) for value in paired_logs(left, right)) for left, right in ((0, 1), (2, 3))]
    return [statistics.median(sample) for sample in samples], ratios, nulls


def decision(rows):
    expected = {(seed, tokens, dist) for seed in BANKS for tokens in LOADS for dist in DISTRIBUTIONS}
    require(isinstance(rows, list) and len(rows) == len(expected), '36 timed rows')
    require({tuple(row[:3]) for row in rows} == expected, 'bank/load/distribution coverage')
    summaries, drift, regressions = [], [], []
    for row in rows:
        require(len(row) == 7 and len(row[3]) == 4 and len(row[4]) == 3
                and len(row[5]) == 4 and len(row[6]) == 2, 'timing row layout')
        for value in row[3]:
            positive(value)
        for value in row[5]:
            positive(value)
        require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in row[6]), 'duplicate log null')
        total, occupied, maximum = row[4]
        require(all(type(value) is int for value in row[4]) and 1 <= occupied <= 32
                and 1 <= maximum <= row[1] and maximum <= total <= row[1] * 8, 'route census capacity')
        ratios = row[5]
        if row[1] >= 32 and any(ratio <= 1 / 1.01 for ratio in ratios):
            regressions.append(row[:3])
        if row[1] < 32:
            drift.append([*row[:3], ratios])
    for seed in BANKS:
        for dist in DISTRIBUTIONS:
            selected = [row for row in rows if row[0] == seed and row[2] == dist]
            nulls = [sum(row[6][index] for row in selected) / 6 for index in range(2)]
            logs = [sum(math.log(row[5][index]) for row in selected) / 6 for index in range(4)]
            passed = all(gain >= math.log(1.01) and gain > 2 * max(nulls) for gain in logs)
            summaries.append(dict(bank=seed, dist=dist, gains=[math.exp(gain) for gain in logs],
                                  duplicate_log_nulls=nulls, passed=passed))
    return dict(component_pass=not regressions and all(row['passed'] for row in summaries),
                groups=summaries, affected_regressions=regressions, unchanged_drift=drift)

H_RUNTIME_ID = '4fd6296182b841b9a6f9adea41c4d8c08c044369320f0fcc478bc22f5118ec25'
H_HOLD = False
H_BUILD_CALLS = [{'name': 'glue_moe_415a2bdc6eb73960_sm90a', 'cpp_sources': 'bf9238761276c88ea043a8ebf161b30d912de2b4418b1b379273fb757d9b230d', 'cuda_sources': 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716', 'functions': ['okc_run', 'okc_set_pdl'], 'extra_cuda_cflags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_L2H=0', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1'], 'verbose': False}, {'name': 'glue_moe_b2495a009a910301_sm90a', 'cpp_sources': 'bf9238761276c88ea043a8ebf161b30d912de2b4418b1b379273fb757d9b230d', 'cuda_sources': 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716', 'functions': ['okc_run', 'okc_set_pdl'], 'extra_cuda_cflags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1'], 'verbose': False}, {'name': 'glue_moe_1b555edc8284ed23_sm90a', 'cpp_sources': 'bf9238761276c88ea043a8ebf161b30d912de2b4418b1b379273fb757d9b230d', 'cuda_sources': 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716', 'functions': ['okc_run', 'okc_set_pdl'], 'extra_cuda_cflags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1'], 'verbose': False}, {'name': 'glue_moe_7e37cd900cb88979_sm90a', 'cpp_sources': 'bf9238761276c88ea043a8ebf161b30d912de2b4418b1b379273fb757d9b230d', 'cuda_sources': 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71', 'functions': ['okc_run', 'okc_set_pdl', 'okc_attrs'], 'extra_cuda_cflags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_B4FIN=1', '-DOKC_SC6W=1', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1'], 'verbose': False}, {'name': 'glue_moe_45a005177ff52c30_sm90a', 'cpp_sources': 'bf9238761276c88ea043a8ebf161b30d912de2b4418b1b379273fb757d9b230d', 'cuda_sources': 'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71', 'functions': ['okc_run', 'okc_set_pdl', 'okc_attrs'], 'extra_cuda_cflags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_SC6W=1', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1'], 'verbose': False}, {'name': 'glue_moe_c6156fd1b1e68a60_sm90a', 'cpp_sources': 'bf9238761276c88ea043a8ebf161b30d912de2b4418b1b379273fb757d9b230d', 'cuda_sources': '8c2cfc7cd87dd863bf32cd6389b9f056e954329201fee1d7011a39efe5ce31c8', 'functions': ['okc_run', 'okc_set_pdl', 'okc_attrs'], 'extra_cuda_cflags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_B4FIN=1', '-DOKC_SC6W=1', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1'], 'verbose': False}, {'name': 'glue_moe_8d08d58831ac01c5_sm90a', 'cpp_sources': 'bf9238761276c88ea043a8ebf161b30d912de2b4418b1b379273fb757d9b230d', 'cuda_sources': '8c2cfc7cd87dd863bf32cd6389b9f056e954329201fee1d7011a39efe5ce31c8', 'functions': ['okc_run', 'okc_set_pdl', 'okc_attrs'], 'extra_cuda_cflags': ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_SC6W=1', '-DOKC_PRMTSC=1', '-DOKC_DQ7=1'], 'verbose': False}]
"""Complete endpoint gate including unchanged-path drift."""
import math


def pairlut_decision(rows,result):
    drift=[]
    for row in rows:
        if row[1]<32:
            limit=max(math.log(1.01),2*max(row[6]))
            if any(abs(math.log(r))>limit for r in row[5]):drift.append(row[:3])
    return dict(passed=result['component_pass'] and not drift,component_pass=result['component_pass'],unchanged_drift=drift)


_PL_LIVE={}
_PL_BUILDS=[]
def _pl_encode(report):
    return json.dumps(report,separators=(',',':'),allow_nan=False)

def _pl_partial(error):
    import base64,zlib
    report={k:v for k,v in _PL_LIVE.items() if k!='primary_error'}
    report.update(runtime_id=H_RUNTIME_ID,status='FAILED_PARTIAL_NOT_QUALIFIED',
                  elapsed_s=time.monotonic()-report['began'],error=type(error).__name__+':'+str(error)[:300])
    blob=base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    require(len(blob)<12000,'bounded partial evidence')
    print('MOE_PAIRLUT_PARTIAL '+blob,flush=True)

"""Appended unchanged to the pinned adapter and offline-extracted input generator."""
import hashlib
import json
import random
import statistics

_H_NATIVE_LOADER = load_inline
_H_KEYS = ('hidden_states', 'topk_ids', 'topk_weights')


def _h_loader(**kwargs):
    descriptor = dict(kwargs)
    for key in ('cpp_sources', 'cuda_sources'):
        descriptor[key] = hashlib.sha256(descriptor[key].encode()).hexdigest()
    require(descriptor in H_BUILD_CALLS, 'exact compiler source/flags/name/exports')
    _PL_BUILDS.append(H_BUILD_CALLS.index(descriptor))
    return _H_NATIVE_LOADER(**kwargs)


def _h_routes(data):
    ids, weights = data['topk_ids'], data['topk_weights']
    require(bool(((ids >= 0) & (ids < 256)).all()), 'global route ID range')
    ordered = ids.sort(dim=-1).values
    require(bool((ordered[:, 1:] != ordered[:, :-1]).all()), 'distinct top8 experts')
    require(bool(torch.isfinite(weights).all()) and bool((weights >= 0).all())
            and bool(((weights.sum(-1) - 1).abs() <= 2e-6).all()), 'route normalization')
    require(bool(torch.isfinite(data['hidden_states']).all()), 'finite inputs')
    counts = torch.bincount(ids.flatten().long(), minlength=256)[:32]
    require(int(counts.max()) <= data['config']['bs'], 'per-expert capacity')
    return [int(counts.sum()), int((counts > 0).sum()), int(counts.max())]


def _h_errors(output, reference):
    require(output.shape == reference.shape and output.dtype == reference.dtype == torch.bfloat16,
            'output metadata')
    actual, expected = output.float(), reference.float()
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), 'nonfinite output')
    delta = actual - expected
    global_error = float(delta.norm() / expected.norm().clamp_min(1e-12))
    row_norm = expected.norm(dim=-1)
    floor = (.1 * row_norm.mean()).clamp_min(1e-12)
    row_error = float((delta.norm(dim=-1) / torch.maximum(row_norm, floor)).max())
    require(global_error <= GLOBAL_TOL and row_error <= ROW_TOL, 'paired output error')
    require(not bool((actual[row_norm == 0] != 0).any()), 'zero-reference rows must be exact zero')
    return [global_error, row_error]


def _h_ownership(states):
    seen, result = set(), []
    require(len({id(state['ma_ext']) for state in states}) == 4, 'private extension maps')
    for index, state in enumerate(states):
        require(state['_expert_pair_arm'] is (index >= 2), 'placement binding')
        owned, byte_count = set(), 0
        for tensor in [*state['ma_w'], *(full for full, words in state['guards'])]:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            require(key not in seen and key not in owned, 'private weight/scratch storage')
            owned.add(key)
            byte_count += storage.nbytes()
        require(len(state['ma_scratch']) == len(state['guards']), 'scratch guard inventory')
        for tensor, (full, words) in zip(state['ma_scratch'], state['guards']):
            require(tensor.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
                    and tensor.data_ptr() == full.data_ptr() and tensor.numel() * tensor.element_size() <= words * 4,
                    'scratch guard owner')
        seen.update(owned)
        result.append(byte_count)
    return result


def _h_reset(states):
    failure = None
    for state in states:
        try:
            initial = _ma_make_scratch(state['device'])
            require(len(initial) == len(state['ma_scratch']), 'scratch reset length')
            for target, source in zip(state['ma_scratch'], initial):
                require(target.shape == source.shape and target.dtype == source.dtype, 'scratch reset metadata')
                target.copy_(source)
            del initial
            torch.cuda.synchronize(state['device'])
            _expert_moe_check_guards(state)
        except Exception as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def _h_case(states, seed, tokens, dist, flush, deadline, counters):
    device = states[0]['device']
    data = generate_input(tokens, 'ep8', seed, dist=dist, device=device)
    census = _h_routes(data)
    saved = {key: data[key].clone() for key in _H_KEYS}
    pointers = tuple(data[key].data_ptr() for key in _H_KEYS)
    captured = [dict(config=dict(data['config']), **{key: data[key].clone() for key in _H_KEYS}) for state in states]
    captured_pointers = [tuple(item[key].data_ptr() for key in _H_KEYS) for item in captured]
    require(len({pointer for group in captured_pointers for pointer in group}) == 12, 'private graph inputs')
    graphs, outputs = [], []
    try:
        for index, state in enumerate(states):
            _expert_moe_run(captured[index], state, index >= 2)
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            with torch.cuda.graph(graph):
                outputs.append(_expert_moe_run(captured[index], state, index >= 2))
            counters['captures'] += 1
        require(len({output.untyped_storage().data_ptr() for output in outputs}) == 4, 'independent outputs')
        require(not {output.data_ptr() for output in outputs}.intersection(
                pointer for group in captured_pointers for pointer in group), 'output/input alias')

        def observe(empty=False):
            _expert_moe_budget(deadline)
            _h_routes(data)
            require(tuple(data[key].data_ptr() for key in _H_KEYS) == pointers, 'captured input pointers')
            for item, expected_pointers in zip(captured, captured_pointers):
                require(tuple(item[key].data_ptr() for key in _H_KEYS) == expected_pointers, 'private captured pointers')
                for key in _H_KEYS:
                    item[key].copy_(data[key])
            references = [_expert_moe_run(data, states[index], False).clone() for index in (0, 1)]
            _h_errors(references[1], references[0])
            for output, graph in zip(outputs, graphs):
                output.fill_(17)
                graph.replay()
            torch.cuda.synchronize(device)
            for output in outputs:
                for reference in references:
                    errors = _h_errors(output, reference)
                    counters['worst'] = [max(old, new) for old, new in zip(counters['worst'], errors)]
                if empty:
                    require(int(torch.count_nonzero(output)) == 0, 'empty routes output')
            for state in states:
                _expert_moe_check_guards(state)
            counters['observations'] += 1

        observe()
        fresh = generate_input(tokens, 'ep8', seed + 97, dist=dist, device=device)
        for key in _H_KEYS:
            data[key].copy_(fresh[key])
        observe()
        for transition in ('empty', 'occupied', 'empty', 'occupied'):
            first = 32 if transition == 'empty' else 0
            ids = torch.arange(first, first + 8, device=device, dtype=torch.int32)
            data['topk_ids'].copy_(ids[None, :].expand(tokens, 8))
            observe(transition == 'empty')
        for key in _H_KEYS:
            data[key].copy_(saved[key])
        observe()
        timings = None
        if tokens in LOADS:
            samples = [[] for state in states]
            rng = random.Random(seed + tokens + (1000 if dist == 'skewed' else 0))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for repeat in range(REPEATS // 2):
                order = list(range(4))
                rng.shuffle(order)
                for traversal in (order, list(reversed(order))):
                    for index in traversal:
                        _expert_moe_budget(deadline)
                        flush.add_(1)
                        start.record()
                        graphs[index].replay()
                        end.record()
                        end.synchronize()
                        samples[index].append(positive(start.elapsed_time(end) * 1000))
                        counters['timed_replays'] += 1
            timings, ratios, nulls = paired_samples(samples)
            for output in outputs:
                for index in (0, 1):
                    _h_errors(output, _expert_moe_run(data, states[index], False))
        return [seed, tokens, dist, timings, census, ratios, nulls] if timings is not None else None
    finally:
        for key in _H_KEYS:
            data[key].copy_(saved[key])
        for item, expected_pointers in zip(captured, captured_pointers):
            for key in _H_KEYS:
                item[key].copy_(saved[key])
            require(tuple(item[key].data_ptr() for key in _H_KEYS) == expected_pointers
                    and all(torch.equal(item[key], saved[key]) for key in _H_KEYS), 'private input restoration')
        torch.cuda.synchronize(device)
        for graph in graphs:
            graph.reset()
        require(tuple(data[key].data_ptr() for key in _H_KEYS) == pointers
                and all(torch.equal(data[key], saved[key]) for key in _H_KEYS), 'input restoration')
        counters['restorations'] += 1


def _pl_prepare_inner(weights):
    global load_inline
    require(not H_HOLD, 'HOLD: parent review and explicit captain allocation required')
    began = time.monotonic()
    deadline = began + 295
    states = []
    original_loader = load_inline
    counters = dict(observations=0, restorations=0, captures=0, timed_replays=0, worst=[0., 0.])
    rows = []
    report = None
    _PL_LIVE.update(rows=rows,counters=counters,began=began,stage="preparation")
    try:
        load_inline = _h_loader
        for arm in (False, False, True, True):
            states.append(_expert_moe_prepare(weights, arm, deadline))
        owners = _h_ownership(states)
        require(sorted(set(_PL_BUILDS))==list(range(7)),"all registered builds observed")
        require([s["ma_w"][6].numel() for s in states]==[256,256,16640,16640],"bound generic LUT owners")
        require(all(torch.equal(s["ma_w"][6][:256],states[0]["ma_w"][6]) for s in states),"original LUT prefix")
        require(all(state['ma_sc6'] and tuple(state['ma_sfx']) == ('-DOKC_PRMTSC=1',) for state in states),
                'pinned SC6/PRMT route only; no fallback')
        flush = torch.zeros(64 * 1024 * 1024, dtype=torch.int32, device=states[0]['device'])
        for seed in BANKS:
            for dist in DISTRIBUTIONS:
                for tokens in TOKENS:
                    _PL_LIVE["stage"]=[seed,tokens,dist]
                    row = _h_case(states, seed, tokens, dist, flush, deadline, counters)
                    if row is not None:
                        rows.append(row)
        report = dict(runtime_id=H_RUNTIME_ID, official_arm='candidate2', placement_arms=[False, False, True, True],
                      owners_bytes=owners, shared_tensor_storage=[], rows=rows, counters=counters,
                      loaded_builds=sorted(set(_PL_BUILDS)),table_words=[s["ma_w"][6].numel() for s in states])
    except BaseException as error:
        _PL_LIVE["primary_error"]=error
        raise
    finally:
        load_inline = original_loader
        _PL_LIVE["stage"]="cleanup"
        try:_h_reset(states)
        except BaseException as cleanup_error:
            _PL_LIVE["cleanup_error"]=str(cleanup_error)[:300]
            if "primary_error" in _PL_LIVE:raise _PL_LIVE["primary_error"] from cleanup_error
            raise
    _PL_LIVE["stage"]="post-cleanup-budget"
    _expert_moe_budget(deadline)
    report.update(prepare_s=time.monotonic() - began, cleanup_restored=True)
    report['component_pass'] = decision(rows)['component_pass']
    report['pairlut_decision']=pairlut_decision(rows,decision(rows))
    _PL_LIVE["stage"]="serialization"
    encoded = _pl_encode(report)
    require(len(encoded.encode()) < CAPSULE_LIMIT, 'bounded capsule')
    _PL_LIVE["stage"]="final-budget"
    _expert_moe_budget(deadline)
    print('EXPERT_MOE_CAPSULE ' + encoded, flush=True)
    return states[2]


def kernel_fn(data, state):
    return _expert_moe_run(data, state, True)


def prepare(weights):
    _PL_LIVE.clear()
    _PL_LIVE.update(rows=[],counters={},began=time.monotonic(),stage='entry',loaded_builds=_PL_BUILDS)
    try:return _pl_prepare_inner(weights)
    except BaseException as error:
        try:_pl_partial(error)
        except BaseException as reporting_error:
            print('MOE_PAIRLUT_REPORTING_FAILURE '+type(reporting_error).__name__+':'+str(reporting_error)[:300],flush=True)
        raise
