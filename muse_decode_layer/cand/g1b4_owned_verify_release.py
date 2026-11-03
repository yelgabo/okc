# okc: problem=muse_decode_layer
# Fixed-mask production dispatcher; H200 qualification still required.

import linecache
import types

def _verify_hybrid_module(name, source):
    module = types.ModuleType(name)
    filename = "<" + name + ">"
    module.__file__ = filename
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)
    return module

_VERIFY_OVERRIDE_MASK = 15

_VERIFY_HYBRID_SOURCE_G = """\
# okc: problem=muse_decode_layer
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
\"\"\"Muse Spark decode layer: one CUDA-graph-capturable kernel_fn.

Pipeline (one decode step, T = batch * q_len tokens)
-----------------------------------------------------
1.  qkv projection (cuBLAS, the bf16_proj kernel, or a Triton split-K GEMM --
    prepare() times the three and pins the winner per (T, matrix)).
2.  Attention: the muse_attention_decode entry's fused kernel (QK-norm,
    RoPE, FP8 cache write, split-KV FP8 attention, merge, per-head
    RMSNorm * sigmoid(gate)) for the shapes it supports; otherwise the glue's
    own QK-prep Triton kernel plus SGLang's FA3 (or a Triton split-KV
    attention fallback when SGLang is unavailable).
3.  o projection, then the first residual boundary (residual add, residual
    gate, pre-FFN norm) -- one-program-per-token kernels or parallel
    atomic-sum kernels, per T, timed in prepare().
4.  Pooler + router (bf16_proj's fused launch, its two separate launches, or
    the Triton split-K GEMM), routing (top-8 of biased sigmoid scores), and
    the pre-expert norm.
5.  Experts: the muse_moe_w4a16 entry's NVFP4 kernel (three compile-time
    builds by token count), or SGLang's Marlin NVFP4 path, whichever
    prepare() timed faster per T.
6.  Unpooler, then the second residual boundary (post-FFN norm, next layer's
    pre-attention norm).

Provenance: the attention, MoE and dense kernels are our own team's entries
on this challenge's component boards, copied (not imported). The CUDA sources
(ATT7_CUDA_SRC, MOE_CUDA_SRC, the bfp_* template strings) and the bfp_*
generators are kept verbatim -- they are the tuned, proven kernels. The glue
(boundaries, routing, tuning, launchers) is this file's own code.

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding
with exact escape correction); NVFP4 experts stay NVFP4; norms apply
(1 + w) formed in FP32 at use; FP8 queries / FP8 KV cache as SGLang;
softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too;
FP32 accumulation throughout; activations BF16 wherever SGLang keeps BF16.

Selection: prepare() times each backend per (token count, matrix) and the
expert kernels per token count, on the judge's own GPU with synthetic
activations (it reads only weights, never data), then fixes the choices;
untuned token counts take the next tuned bucket's.

OKC rules: one Python file; native CUDA only via
torch.utils.cpp_extension.load_inline; CUDA-graph capturable (no host syncs
in timed paths -- prepare() and the tuners sync, kernel_fn does not); fresh
inputs each call, no caching across calls (module-level caches hold only
compiled extensions and per-device scratch, never activations).
Safety: every scratch/workspace buffer is followed by sentinel words;
prepare() checks them after the tuner and the attention self-test.
\"\"\"

import os
import threading
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
QKV_ROWS = 2304
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
Q_DIM = NUM_Q_HEADS * HEAD_DIM  
ROPE_THETA = 500000.0
ROUTER_MULT = 0.1767766952966369  
LOG2E = 1.4426950408889634

_TUNE_LOG = False
FORCE_PBMAX = None  
PDL = False
PDL_MAX_T = 128  


def _pdl(T):
    return PDL and T <= PDL_MAX_T


DEFAULT_MOE = "agent"
TUNE_MARGIN = 0.95
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp", (32, "pr"): "bfp",
            (128, "qkv"): True, (256, "qkv"): True}
FORCE_GLUEV2 = False  


ATT7_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdint.h>

namespace {

#define OKC_NP 128       
#define OKC_WS 1         
#define OKC_WS8 1        
#define OKC_NP8 128      
#define OKC_WSMINB 2
#define OKC_SETMAXNREG 1
#define OKC_PREGS 0
#define OKC_TMAPG 1      
#define OKC_XHALFPV 0    
#define OKC_XNOWAIT 0    
#define OKC_MRGDBG 0     
#define OKC_MERGE2 1     
#define OKC_MERGE3 1     
#define OKC_IDXFIRST 1   
#define OKC_MAGICDIV 1   
#define OKC_SC1 1        
#define OKC_QP16 1       
#define OKC_ONENEW 1     
#define OKC_SC8 1        
#define OKC_PINDIV 1     
#define OKC_ONEPRED 1    
#define OKC_LATETID 1    
#define OKC_PGCONT 0     
#define OKC_TMAPG8 0     
#define OKC_ST8F 4       
#define OKC_COOP 1       
#define OKC_PVW 0        
#define OKC_WGDEC 1      
#define OKC_SEGDBG 0
#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)
#define OKC_PV8 1       // q_len 8 on sm_90: PV as an FP8 wgmma (P in e4m3, as SGLang's FA3 FP8 path does: pod/p16probe*.py)
#define OKC_PROF 0       
#define OKC_PROFPV8 0    
#define PV8ON (OKC_PV8 && (!OKC_PROF || OKC_PROFPV8) && !OKC_PVW)
#define OKC_PV8FLAT 1    
#define OKC_QSS 1        
#define OKC_PV8SYNC 0    
#define OKC_Q8RAW 1      
#define OKC_FLATCOOP1C 0   
#define OKC_RAWTHR 0     
#define OKC_PCHK 1       
#define OKC_TURN 0       
                         
#define OKC_CLSM 0       
#define OKC_PDLKV 0      
                         
#define OKC_RARE 1       
#define OKC_PCHKTEST 0   
#define OKC_UNEV 0       
                         
#define OKC_W0L1 0       
                         
#define OKC_SEGK8 600    
#define OKC_SEGK1 400    
#define OKC_W0SPREAD 0   
#define OKC_W0EARLY 1    
#define OKC_W0FORCEBAD 0   
#define OKC_W0BID 1      
                         
#define OKC_DBGON 0      
#define OKC_BALPART 0    
#define OKC_CONT 0       
#define OKC_DYN 0        
#define OKC_FINWARM 0    
#define OKC_PVW1 1       
#define OKC_NOXCH1 0     
#define OKC_PVSPLIT 1    
                         
#define OKC_QKZ 0        
#define OKC_INV2 1       
#define OKC_OZF 1        
#define OKC_XP_ARRALL 1  
                         
#define OKC_XP_NOMASK 0  
#define OKC_IRANGE 0     
#define OKC_PATCHNI 1    
#define OKC_PARITY 1     
#define OKC_SENSE2 0     
#define OKC_UNRN 1       
#define OKC_ADDRN 1      
#define OKC_OFF32 1      
#define OKC_UNR8 2       
#define OKC_RED4 1       
#define OKC_SWAPHALF 0
#define OKC_LATESYNC 1   
                         
#define OKC_LATESYNCN 1  
                         
#define OKC_QKFIRST 1    
                         
                         
#define OKC_FPART 1      
                         
                         
#define OKC_EARLYBAR 1   
                         
#define OKC_PTPF 1       
                         
#define OKC_FPRO 0       
#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); if (*_dp == 0) *_dp = gtime(); } } while (0)
#define OKC_COOPGUARD 1  
                         
                         
__device__ __noinline__ void spin_fail() { __trap(); }    
#define SPIN_T0 unsigned _spin_n = 0    
#define SPIN_CHECK do { if (OKC_COOPGUARD && ++_spin_n > (1u << 24)) spin_fail(); } while (0)    
#define OKC_ST1F 3       
#define OKC_STN 3        
#define OKC_UWARP 1
#define OKC_INEW 0       
#define OKC_ADDR 1       
                         
#define OKC_STSM 0       
#define OKC_STSMPR 1     
#define OKC_TCHUNK 1     
#define OKC_EXPPOLY 0    
#define OKC_EXH8 0       
#define OKC_NOSYNC0 1    
#define OKC_LAZYMAX 1    
#define OKC_SENSE 0      
#define OKC_WSPLIT 500   
#define OKC_F2 0         
#define OKC_S2 0         
#define OKC_PTR 0        
#define OKC_PTRD 2       
#define OKC_XP_NOTR 0       
#define OKC_XP_NOFENCE 0    
#define OKC_XP_NOEXP 0      
#define OKC_CLDBG 0
#define OKC_SPINDBG 0
template <int L, bool F>
struct Cfg {
   
   
  static constexpr bool S2 = (L == 8) && !F && OKC_S2 && OKC_SM90 && PV8ON;
   
   
  static constexpr bool F2 = (L == 8) && F && OKC_F2 && OKC_SM90 && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT && !OKC_PTR;
  static constexpr int NT = (L == 1) ? 128 : ((S2 || F2) ? 512 : 256);    
   
   
  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);
  static constexpr int NP = WS ? ((L == 1) ? OKC_NP : OKC_NP8) : 0;    
  static constexpr int NTX = NT + NP;                      
  static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);    
   
  static constexpr int BASEREGS = (65536 / (MINB * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (MINB * NTX)) / 8 * 8;
  static constexpr int PREGS = OKC_PREGS > 0 ? OKC_PREGS : ((L == 1 && NP <= 128) ? 56 : 32);
  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8 > 248
                                   ? 248 : ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8;
   
   
  static constexpr int STAGES = F ? ((L == 1) ? OKC_ST1F : (F2 ? 8 : OKC_ST8F)) : (S2 ? 6 : OKC_STN);
   
   
  static_assert(!WS || (NP % 8 == 0 && NP >= 8 && 64 % (NP >= 8 ? NP / 8 : 1) == 0),
                "producer threads must cover a tile in whole passes");
};
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int ROWB = 128;                     
constexpr int STAGE_BYTES = TILE * ROWB;      
constexpr int IDX_SLOTS = 2 * STAGES;
constexpr int MAXB = 512;                
constexpr int MAXSEG = 64;               

struct Params {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws_acc;
  float* ws_l;
  int* ws_cnt;
  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;
  long long* dbg;       
  int cs, ncl;           
  float* scratch;        
  int* bar;              
  int ld_qkv, row_bytes;
  float scale_log2;    
  int tma;               
  int coop;              
  int lp;                
  int lrb;               
  int off32;             
  int w0;                
  int resident;          
  unsigned ns_magic;     
  int unev;              
  int pdlf;              
  CUtensorMap tk, tv;    
  CUtensorMap tpt;       
  const uint8_t* pf;     
  unsigned pf_chunk;     
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {
   
  asm volatile("{\\n .reg .pred q;\\n setp.ne.b32 q, %3, 0;\\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\\n}\\n"
               ::"r"(dst), "l"(src), "r"(src_bytes), "r"((int)pred));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }

#define OKC_TMA 0
 
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");
}
 
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  asm volatile("{\\n.reg .pred q;\\nWAIT_%=:\\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\\n@!q bra WAIT_%=;\\n}\\n"
               ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
 
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\\n" ::"r"(smem_u32(bar)) : "memory");
}
 
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
 
__device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\\n" ::"r"(dst), "l"(any_src));
}
 
 
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgsync(int wg) {
  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
}
template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory");
  else __syncthreads();
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
 
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\\n\\t.reg .b16 t;\\n\\tmov.b32 {_, t}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, t;\\n\\t}" : "=r"(d) : "r"(x));
  return d;
}
 
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
  uint32_t d;
  asm("{\\n\\t.reg .b16 t;\\n\\tmov.b32 {t, _}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, t;\\n\\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
 
__device__ __forceinline__ uint32_t ex2h2(float lo, float hi) {
  return pack_h2(ex2(lo), ex2(hi));
}
__device__ __forceinline__ float pr8(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }

 
__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,
                                             uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * tig;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);
}

__device__ __forceinline__ void gpu_fence() {
  __threadfence();
}
__device__ __forceinline__ void red_f32(float* a, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");
}
__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}

__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k)                                                                                      \\
  if (OKC_DBGON && p.dbg && threadIdx.x == 0)                                                       \\
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k)] = gtime();

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
 
 
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}

 
#define PROF_INIT
#define PROF_MARK(k)
#define PROF_TILE
#define PROF_DUMP

 
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
 
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
 
__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc, int scale_d) {
  asm volatile(
      "{\\n.reg .pred p;\\nsetp.ne.b32 p, %37, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1;\\n}\\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));
}
 
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\\n.reg .pred p;\\nsetp.ne.b32 p, %34, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\\n}\\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
 
 
 
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss_z(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\\n.reg .pred p;\\nsetp.ne.b32 p, %34, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\\n}\\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
        "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
        "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "l"(da), "l"(db), "r"(0));
}
 
__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
 
__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "{%36, %37, %38, %39}, %40, 1, 1, 1;\\n}\\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
 
 
__device__ __forceinline__ int fdiv_i(int x, int d) {
   
  const float rd = __fdividef(1.f, __int2float_rn(d));
  int q = __float2int_rz(__int2float_rn(x) * rd);
  const int r = x - q * d;
  q += (r >= d) - (r < 0);
  return q;
}
 
 
__device__ __forceinline__ float sigm(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));
}
 
 
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() {
   
   
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
 
 
typedef int ix_t;
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);
  return base + row * p.row_bytes + col_bytes;
}
 
 
template <int N, int NTH, class F>
__device__ __forceinline__ void tfor(int tid, F&& f) {
#pragma unroll
  for (int e0 = 0; e0 < N; e0 += NTH) {
    const int e = e0 + tid;
    if ((N % NTH) == 0 || e < N) f(e);
  }
}
 
__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
 
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {
  __half2_raw a, b;
  a.x = (unsigned short)(h01 & 0xffffu); a.y = (unsigned short)(h01 >> 16);
  b.x = (unsigned short)(h23 & 0xffffu); b.y = (unsigned short)(h23 >> 16);
  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);
  return (lo & 0xffffu) | (hi << 16);
}

__device__ __forceinline__ void red_v2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(a), "f"(x), "f"(y) : "memory");
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float oct_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}

 
__device__ __forceinline__ int cl_rank(const Params& p) {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return (int)r;
}
__device__ __forceinline__ void cl_sync(const Params& p, int phase) {
  asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;\\n" ::: "memory");
}
__device__ __forceinline__ void cl_done(const Params& p) {
}
 
__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(r));
  return v;
}
__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));
  return v;
}

template <int L, bool F>
struct Smem {
  uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint64_t mbar[8];    
  uint64_t pbar[4];    
  int pgc;             
  int dbgn;            
  int vidx;            
  alignas(128) int pidx[4][80];    
  uint64_t full[8];    
  uint64_t empty[8];   
  uint64_t vtf[2];     
  uint64_t vte[2];     
  uint64_t qdone;      
  int claim[2];        
  int nxt[4][2];       
  int idx[2 * Cfg<L, F>::STAGES * TILE];
   
   
   
   
  alignas(1024) uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : (OKC_PVW ? 32768 : (PV8ON ? ((Cfg<L, F>::S2 || Cfg<L, F>::F2) ? 36864 : 18432) : 16384))];
   
  alignas(1024) uint8_t q8s[(L == 8 && PV8ON && OKC_QSS && (!F || OKC_PV8FLAT)) ? 8192 : 16];
  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }
  __nv_bfloat16 kvst[L * 4 * 64];    
  __nv_bfloat16 gst[L * 4 * 64];     
  uint8_t kn[L * ROWB];         
  uint8_t vn[L * ROWB];
  float cs[L * 32];
  float sn[L * 32];
  float ifreq[32];           
  float scl[4];              
  long long cloc[L];         
  int flag;
  int sense;           
  int cum[F ? MAXB + 1 : 1];
  int seg[F ? MAXSEG : 1][3];
  int nseg;
  int fcs, fce;              
   
   
  int pseg[F ? MAXSEG : 1][3];
  int pnseg, pchunk;
  int dq[F ? 16 : 1][4];
  uint64_t dqf[F ? 16 : 1];
  uint64_t dqe[F ? 16 : 1];
};

 
 
 
template <int L, typename SM>
__device__ __noinline__ bool cluster_multi(const Params& p, SM& S, const int CS, const int rank, const int RPC,
                                           const int row0, const int bp, float* RB, float* RL) {
  constexpr int NT = Cfg<L, false>::NT;
  const int tid = threadIdx.x;
  bool fin = true;
    const int clu = fdiv_i(blockIdx.x, CS);
    const int copy = clu - fdiv_i(clu, p.nc) * p.nc;
    for (int e = tid; e < RPC * 16; e += NT) {
      const int row = row0 + (e >> 4);
      float* acc = p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * 64 + (e & 15) * 4;
      const float4 a = *reinterpret_cast<const float4*>(&RB[e * 4]);
      red_v4(acc, a.x, a.y, a.z, a.w);
    }
    for (int e = tid; e < RPC; e += NT) red_f32(p.ws_l + (bp * p.nc + copy) * L * 16 + row0 + e, RL[e]);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
       
      const int old = atomicAdd(p.ws_cnt + bp * CS + rank, 1);
      S.flag = (old == p.ncl - 1);
    }
    __syncthreads();
    fin = S.flag;
    if (fin) {
      __threadfence();
      for (int e = tid; e < RPC * 16; e += NT) {
        const int row = row0 + (e >> 4);
        constexpr int NCM = 4;
        float4 v[NCM];
#pragma unroll
        for (int c = 0; c < NCM; ++c)
          if (c < p.nc) v[c] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4));
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int c = 0; c < NCM; ++c) {
          if (c < p.nc) {
            a.x += v[c].x; a.y += v[c].y; a.z += v[c].z; a.w += v[c].w;
            *reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        *reinterpret_cast<float4*>(&RB[e * 4]) = a;
      }
      for (int e = tid; e < RPC; e += NT) {
        float a = 0.f;
        for (int c = 0; c < p.nc; ++c) {
          float* lw = p.ws_l + (bp * p.nc + c) * L * 16 + row0 + e;
          a += __ldcg(lw);
          *lw = 0.f;
        }
        RL[e] = a;
      }
      if (tid == 0) p.ws_cnt[bp * CS + rank] = 0;
      __syncthreads();
    }
  return fin;
}

 
 
 
 
template <int L, typename SM>
__device__ __forceinline__ void cluster_finish(const Params& p, SM& S, const int b, const int pair,
                                               const int n_keys, const int chunk) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int tid = threadIdx.x;
  const int CS = p.cs, rank = cl_rank(p);
  const int RPC = fdiv_i(L * 16, CS);
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
   
  float* PB = reinterpret_cast<float*>(S.k);
  float* PL = PB + L * 16 * 64;
  float* RB = PL + 1024;
  float* RL = RB + RPC * 64;
  constexpr int REGION = L * 16 * 64 + 1024;    
  if (OKC_CLDBG) DBG(7);
  cl_sync(p, 1);
  DBG(4);
  long long dbgc0 = 0;
  if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
  const int row0 = rank * RPC;
  if constexpr (L == 1) {    
       
       
    const int TOT0 = RPC * 17 * CS;
    const int lcs = 31 - __clz(CS);
#pragma unroll 1
    for (int rep2 = (OKC_CLDBG == 2 ? 0 : 1); rep2 < 2; ++rep2) {
    int TOT = TOT0;
    if (rep2 == 0) asm volatile("mov.u32 %0, 0;" : "=r"(TOT));    
    if (OKC_CLDBG == 2 && rep2 == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
    for (int k0 = 0; k0 < (rep2 == 0 ? 2 * NT : TOT); k0 += 2 * NT) {    
      float4 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < TOT) {
          if (slot < 16) v[u] = cl_ld4(p, PB, q, (row0 + rl) * 64 + slot * 4, REGION);
          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);
        }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        for (int off = 1; off < CS; off <<= 1) {
          v[u].x += __shfl_xor_sync(0xffffffffu, v[u].x, off);
          v[u].y += __shfl_xor_sync(0xffffffffu, v[u].y, off);
          v[u].z += __shfl_xor_sync(0xffffffffu, v[u].z, off);
          v[u].w += __shfl_xor_sync(0xffffffffu, v[u].w, off);
        }
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        if (k < TOT && q == 0) {
          if (slot < 16) *reinterpret_cast<float4*>(&RB[(rl * 16 + slot) * 4]) = v[u];
          else RL[rl] = v[u].x;
        }
      }
    }
    }
  } else
  {
  constexpr int CSMAX = 16;
   
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll 1
    for (int q0 = 0; q0 < CS; q0 += 4) {
      float4 v[4];
      float lv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        v[u] = cl_ld4(p, PB, q0 + u, off, REGION);
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; la += lv[u]; }
    }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  if (false)
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;    
    float4 v[CSMAX];
    float lv[CSMAX];
#pragma unroll
    for (int q = 0; q < CSMAX; ++q) {    
      if (q < CS) {
        v[q] = cl_ld4(p, PB, q, off, REGION);
        lv[q] = lrow ? cl_ld1(p, PL, q, row0 + (e >> 4), REGION) : 0.f;
      }
    }
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll
    for (int q = 0; q < CSMAX; ++q)
      if (q < CS) { a.x += v[q].x; a.y += v[q].y; a.z += v[q].z; a.w += v[q].w; la += lv[q]; }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  }
  if (OKC_CLDBG) {
    long long c1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));
    if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 11] = c1 - dbgc0;
  }
  if (OKC_CLDBG) DBG(8);
  DBG(14);
  cl_sync(p, 2);    
  cl_done(p);
  DBG(6);

  bool fin = true;
  if (OKC_CLDBG) DBG(9);
  if (p.ncl > 1) {
    fin = cluster_multi<L>(p, S, CS, rank, RPC, row0, bp, RB, RL);
  }
  if (fin) {
     
    if (OKC_CLDBG) DBG(10);
    long long dbgc2 = 0;
    if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc2));
    const float vscale = S.scl[1];
    for (int eb = tid - (tid & 31); eb < RPC * 16; eb += NT) {    
      const int e = eb + (tid & 31);
      const bool act = e < RPC * 16;
      const int rl = act ? (e >> 4) : 0, c4 = e & 15;
      const int row = row0 + rl;
      const int j = row >> 4, hh = row & 15;
      const float lsum = act ? RL[rl] : 0.f;
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      const float4 a = act ? *reinterpret_cast<const float4*>(&RB[e * 4]) : make_float4(0.f, 0.f, 0.f, 0.f);
      float y[4] = {pr8(a.x * inv), pr8(a.y * inv), pr8(a.z * inv), pr8(a.w * inv)};
      float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 8);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint2 graw = *reinterpret_cast<const uint2*>(&S.gst[rl * 64 + c4 * 4]);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint2 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float2 gt = __bfloat1622float2(g2[k]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * k] * rn * s0, y[2 * k + 1] * rn * s1);
        ow[k] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint2*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + c4 * 4) = ov;
    }
    if (OKC_CLDBG) {
      long long c3;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(c3));
      if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 15] = c3 - dbgc2;
    }
    DBG(5);
  }
}

 
template <int L, typename SM>
__device__ __forceinline__ void cluster_stage_gate(const Params& p, SM& S, const int b, const int pair) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;
  for (int e = threadIdx.x; e < RPC * 8; e += NT) {
    const int rl = e >> 3, ch = e & 7;
    const int row = row0 + rl, j = row >> 4, hh = row & 15;
    cp_async16(smem_u32(&S.gst[rl * 64 + ch * 8]),
               p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8, 16);
  }
}

 
 
 
 
template <int L>
__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,
                                           const bool act, float (&y)[8], float& lsum) {
  if (OKC_SC1 && p.nc == 1) {    
    if (act) {
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
    }
    return;
  }
  if (OKC_SC8 && p.nc == 8) {    
    if (act) {                    
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * 8 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < p.nc; c0 += 4) {
    float4 xa[4], xb[4];
    float lc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = 0.f;
      if (act && c < p.nc) {
        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[u] = __ldcg(p.ws_l + ((bp * p.nc + c) * L + j) * 16 + hh);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      if (act && c < p.nc) {
        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
        lsum += lc[u];
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * p.nc + c) * L + j) * 16 + hh] = 0.f;
      }
    }
  }
}

 
 
 
template <int L, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
   
   
  constexpr int PASSES = L * 16 / (NT / 8);
  constexpr int GP = PASSES >= 2 ? 2 : 1;
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; p0 += GP) {
    float4 xa[GP], xb[GP];
    float lc[GP];
    uint4 graw[GP];
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
    }
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
      }
      const float inv = lc[u] > 0.f ? __fdividef(vscale, lc[u]) : 0.f;
      float y[8] = {pr8(xa[u].x * inv), pr8(xa[u].y * inv), pr8(xa[u].z * inv), pr8(xa[u].w * inv),
                    pr8(xb[u].x * inv), pr8(xb[u].y * inv), pr8(xb[u].z * inv), pr8(xb[u].w * inv)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (!dry) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
}

 
 
template <int L, int NT, bool PTR, class SM>
__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {
  SM& S = *Sp;
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq), hi = min(t0 + TILE, end);
  if (lo < hi) {
    for (int e = tid; e < (hi - lo) * 8; e += NT) {
      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
      const int sw = ((c ^ (r & 7)) << 4);
      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
      if constexpr (!PTR)
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    }
    if (L == 8) fence_proxy_async();
  }
  return lo < hi;
}

template <int L, bool FLAT, int ROLE, int MODE = 0>
__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_, const bool first,
                                        const int start_, const int end_, const int seq_, const int n_keys_,
                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,
                                        const bool skip_init_ = false) {
  // (returns false only in the OKC_SPEC builds, when the speculated length missed: nothing global was written yet and
  // the caller re-runs the segment with the real length)
  // per-segment values as lane-0 broadcasts (OKC_UARGS): provably warp-uniform, so ptxas keeps the address math
  // derived from them in uniform registers (they come from shared memory or per-CTA math, uniform but unproven)
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
  const int b = UNI(b_), pair = UNI(pair_), start = UNI(start_), end = UNI(end_), seq = UNI(seq_);
  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);
#undef UNI
   
   
   
   
   
   
   
  constexpr bool ONE = !FLAT && (MODE & 4) != 0;
  const bool M_CL = !FLAT && ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2);
  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && OKC_CONT;
   
  constexpr bool PTR = (L == 8) && FLAT && Cfg<L, FLAT>::WS && OKC_PTR && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT;
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool WS = Cfg<L, FLAT>::WS;
  // (OKC_UWARP: the warp index as a lane-0 broadcast, so ptxas can prove it warp-uniform and keep the addresses
  // derived from it (KV head, descriptors, buffers) in uniform registers; CUTLASS's canonical_warp_idx_sync idiom)
  const int tid = threadIdx.x, lane = tid & 31, warp = OKC_UWARP ? __shfl_sync(0xffffffffu, tid >> 5, 0) : tid >> 5;
  const int g = lane >> 2, tig = lane & 3;
  const int split = first ? 0 : 1;
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;    

  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;
    constexpr int PR = NP / 8;             
    constexpr int PK = TILE / PR;          
    if (!CONT && !skip_init_) {    
      if (ROLE == 2 && tid == 0) {    
        for (int st = 0; st < STAGES; ++st) {
          mbar_init_n(&S.full[st], NP);
          mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? NT : NT / 32));    
        }
        if constexpr (PTR) {
          for (int k = 0; k < 2; ++k) {
            mbar_init_n(&S.vtf[k], NP);
            mbar_init_n(&S.vte[k], NT / 32);
          }
          mbar_init_n(&S.qdone, NT / 32);
        }
        mbar_fence_init();
      }
      __syncthreads();
    }
    if constexpr (ROLE == 1) {
       
       
      const int q = tid - NT;
      if (q == 0) FPRO_STAMP(42);
      const int ch = q & 7, r0 = q >> 3;
      const int* ptr = p.page_table + (ix_t)b * p.max_pages;
      const int cache_end = min(end, seq);
      int pg[PK];
      static_assert(PR % 8 == 0, "PFAST: the swizzle chunk is the same for all of a thread's rows");
      const bool pfast = OKC_OFF32 && p.off32 && p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
       
      constexpr bool TMAPG_ON = OKC_TMAPG && (L == 1 || OKC_TMAPG8);
       
      constexpr int PS = 4, PD = 3;
      int pgb = 0;    
      if constexpr (TMAPG_ON) {
        if constexpr (OKC_PGCONT) {
          pgb = S.pgc;
        } else {
          if (q == 0) {
            for (int st = 0; st < PS; ++st) mbar_init_n(&S.pbar[st], 1);
            mbar_fence_init();
          }
          asm volatile("bar.sync 2, %0;\\n" ::"n"(NP) : "memory");
        }
      }
       
       
      const long long pt_total = (long long)p.B * p.max_pages;
      const long long pt_lim = (pt_total + 3) & ~3LL;
      auto fetch_pages = [&](int x) {
        if (q == 0 && x < ntiles) {
          const int slot = (pgb + x) % PS;
          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);
          const long long w0 = e0 & ~3LL;
          const long long w1 = min(pt_lim, (e0 + 64 + 3) & ~3LL);
          const int bytes = (int)(w1 - w0) * 4;
          mbar_expect_tx(&S.pbar[slot], bytes);
          bulk_load(&S.pidx[slot][0], p.page_table + w0, bytes, &S.pbar[slot]);
        }
      };
      if constexpr (TMAPG_ON) {
        for (int x = 0; x < PD; ++x) fetch_pages(x);
      }
      auto load_pages_t = [&](int x, int* dst) {
        const int slot = (pgb + x) % PS;
        mbar_wait_full(&S.pbar[slot], ((pgb + x) / PS) & 1);
        const int fp = (start + x * TILE) >> p.lp;
        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);    
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;
        }
      };
      auto load_pages = [&](int x, int* dst) {
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
      };
      if constexpr (!TMAPG_ON) load_pages(0, pg);
      if (OKC_FPRO >= 2 && q == 0) {    
        asm volatile("" :: "r"(pg[0]), "r"(pg[PK - 1]));
        volatile int sink = pg[0] + pg[PK - 1];
        (void)sink;
        FPRO_STAMP(45);
      }
       
       
      auto ptr_transpose = [&](int y) {
        if constexpr (PTR) {
          const int sty = (tb + y) % STAGES;
          if (y == 0) {    
            mbar_wait_full(&S.qdone, 0);
            for (int e = q; e < 4 * 32; e += NP)
              *reinterpret_cast<uint4*>(S.kvq + (e >> 5) * 4608 + 4096 + (e & 31) * 16) =
                  make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
          }
          mbar_wait_full(&S.full[sty], ((tb + y) / STAGES) & 1);    
          const int t0 = start + y * TILE;
          const int nlo = max(t0, seq), nhi = min(t0 + TILE, end);
          if (nlo < nhi) {    
            for (int e = q; e < (nhi - nlo) * 8; e += NP) {
              const int pos = nlo + (e >> 3), c = e & 7, r = pos - t0;
              *reinterpret_cast<uint4*>(S.v + sty * STAGE_BYTES + r * ROWB + ((c ^ (r & 7)) << 4)) =
                  *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
            }
            asm volatile("bar.sync 2, %0;\\n" ::"n"(NP) : "memory");
          }
          if (y >= 2) mbar_wait_full(&S.vte[y & 1], ((y >> 1) - 1) & 1);    
          const int pl = q & 31, pw = (q >> 5) & 3, pvq = pl & 3, pvw = pl >> 2;
          const uint32_t s0 = (pvw >= 4) ? 0x7531u : 0x6420u, s1 = (pvw >= 4) ? 0x6420u : 0x7531u;
          const int pr0 = (pvw >= 4) ? 1 : 0;
          const uint32_t o0 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * pr0;
          const uint32_t o1 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * (1 - pr0);
          const int pkey = 16 * pw + 8 * ((pl >> 3) & 1) + (pl & 7);
          const uint32_t vsrc = smem_u32(S.v + sty * STAGE_BYTES) + pkey * ROWB;
          uint8_t* vt = S.kvq + (y & 1) * 2 * 4608;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              uint32_t r0, r1, r2, r3;
              ldsm_x4_t(vsrc + (((4 * h + 2 * u + ((pl >> 3) >> 1)) ^ (pl & 7)) << 4), r0, r1, r2, r3);
#pragma unroll
              for (int g2 = 0; g2 < 2; ++g2) {
                const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o0 + 1024 * (2 * u + g2)) = prmt(R0, R1, s0);
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o1 + 1024 * (2 * u + g2)) = prmt(R0, R1, s1);
              }
            }
          }
          fence_proxy_async();    
          mbar_arrive(&S.vtf[y & 1]);
        }
      };
      PROF_INIT
      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];
        if constexpr (TMAPG_ON) {
          fetch_pages(x + PD);
          load_pages_t(x, pg);
        } else {
          load_pages(x + 1, pgn);    
        }
        PROF_MARK(2);
        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
        PROF_MARK(0);
        if (x == 0 && q == 0) FPRO_STAMP(46);
        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
         
         
        if (pfast && start + x * TILE + TILE <= cache_end) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[k] << p.lrb) + pcol;
            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
          }
        } else
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int r = r0 + PR * k;
          const int pos = start + x * TILE + r;
          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
          if (pos < cache_end) {
            if (p.off32) {    
              const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            } else
            {
              const int64_t off = (((int64_t)pg[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }
          } else if (pos >= end) {
            cp_async16_zero(kb + so, p.k_cache);
            cp_async16_zero(vb + so, p.v_cache);
          }
        }
        cp_async_mbar_arrive(&S.full[st]);
        if (x == 0 && q == 0) FPRO_STAMP(43);
        if constexpr (PTR) {
          if (x >= OKC_PTRD) ptr_transpose(x - OKC_PTRD);
        }
        PROF_MARK(1);
        PROF_TILE;
        if constexpr (TMAPG_ON) {
          asm volatile("bar.sync 2, %0;\\n" ::"n"(NP) : "memory");    
        } else {
#pragma unroll
          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];
        }
      }
      if constexpr (PTR) {
        for (int y = max(0, ntiles - OKC_PTRD); y < ntiles; ++y) ptr_transpose(y);
      }
      if constexpr (TMAPG_ON && OKC_PGCONT) {    
        if (q == 0) S.pgc = pgb + ntiles;
      }
      PROF_DUMP;
      if constexpr (!CONT) {
        cp_wait<0>();
        __syncthreads();    
      }
      return true;
    }
  }
  if constexpr (ROLE != 1) {

  const int prow = tid >> 1, phalf = tid & 1;
  const int* pt_row = p.page_table + (ix_t)b * p.max_pages;
  const int part = tid & 7;

   
   
   
  const int cchunk = tid & 7, crow0 = tid >> 3;
  constexpr int RPP = NT / 8;    
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % STAGES;
    const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];
    bool generic = false;    
     
     
    if (with_new && p.off32 && p.lp == 0 && start + x * TILE + TILE <= min(end, seq)) {    
       
      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        const uint32_t off = ((uint32_t)idxs[r] << p.lrb) + col;
        cp_async16(kst + so, p.k_cache + off, 16);
        cp_async16(vst + so, p.v_cache + off, 16);
      }
      return;
    }
#pragma unroll
    for (int k = 0; k < TILE / RPP; ++k) {
      const int r = crow0 + RPP * k;
      const int pos = start + x * TILE + r;
      const int sw = ((cchunk ^ (r & 7)) << 4);
      uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;
      uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;
      if (pos < end && pos < seq) {
        if (p.off32) {    
          const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        } else
        {
          const int64_t row = ((int64_t)idxs[r] << p.lp) + (pos & (p.P - 1));
          const int64_t off = row * p.row_bytes + pair * 128 + cchunk * 16;
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        }
      } else if (pos < end) {
        if (with_new) {
          *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
          generic = true;
        }
      } else {    
        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);
      }
    }
    if (L == 8 && generic) fence_proxy_async();
  };
  auto issue_idx = [&](int y) {
    if (phalf == 0 && prow < TILE && y < ntiles) {
      const int pos = start + y * TILE + prow;
      if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));
    }
  };
   
  auto ws_patch = [&](int i, int st) {    
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {    
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };
  auto ws_acquire = [&](int i, int st) {
    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {    
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)    
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };

   
  int tidL = tid, partL = part;    
  constexpr int QPASS = L * 16 / RPP;             
  const bool has_new = end > seq;
  constexpr int KPASS = (L * 4 + RPP - 1) / RPP;  
  constexpr bool QDIR = OKC_QDIRECT && L == 8 && !WS && !Cfg<L, FLAT>::S2 && PV8ON && OKC_QSS && OKC_SM90 && OKC_Q8RAW &&
                        (!FLAT || OKC_PV8FLAT);
  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;
  {
  if constexpr (!WS) {
#pragma unroll
    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
  }
  cp_commit();
  if (OKC_LATETID && !WS && (MODE & 8)) {    
     
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));
    partL = tidL & 7;
  }

  pdl_wait();    
  pdl_launch_dependents();
  if (tid == 0) FPRO_STAMP(44);
   
  tfor<L * 16 * 8, NT>(tidL, [&](int e) {    
    const int rr = e >> 3, ch = e & 7;
    const int j = rr >> 4, hh = rr & 15;
    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);
  });
  if (has_new) {
    tfor<L * 4 * 8, NT>(tidL, [&](int e) {    
      const int rr = e >> 3, ch = e & 7;
      const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;
      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);
    });
  }
  if (M_CL) cluster_stage_gate<L>(p, S, b, pair);
   
  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
  else if ((OKC_SENSE2 || OKC_PARITY) && !FLAT && M_COOP && tidL == 10)    
    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
  else if (tidL >= 16 && tidL < 16 + L) {
    const int jj = tidL - 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
  }
  cp_commit();
  if (split == 0 && (FLAT || !M_CL)) {   
    tfor<L * 16, NT>(tidL, [&](int e) {    
      const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
    });
  }

   
   
   
  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
  if constexpr (!FLAT && (MODE & 8) != 0) {
     
     
     
    if (seq_real_ != seq) {
      cp_wait<0>();
      __syncthreads();
      return false;
    }
  }
  DBG(1);
  if constexpr (!WS) {
    constexpr int PRE = Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1;    
    bool one = false;
     
     
     
    if (ONE && !Cfg<L, FLAT>::S2 && p.off32 && p.lp == 0 && (OKC_ONENEW || start + TILE <= min(end, seq))) {
      one = true;
      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
      const int cend = min(end, seq);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        if constexpr (OKC_ONEPRED) {
           
           
           
          const int pr = start + r;
          const bool cached = !OKC_ONENEW || pr < cend;
          const bool doit = cached || pr >= end;
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;    
          cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
          cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
        } else if (!OKC_ONENEW || start + r < cend) {
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;
          cp_async16(kst + so, p.k_cache + off, 16);
          cp_async16(vst + so, p.v_cache + off, 16);
        } else if (start + r >= end) {    
          cp_async16_zero(kst + so, p.k_cache);
          cp_async16_zero(vst + so, p.v_cache);
        }    
      }
#pragma unroll
      for (int k = 0; k < STAGES - 1; ++k) cp_commit();
    }
    if (!one) {
#pragma unroll
    for (int k = 0; k < STAGES - 1; ++k) {    
      if (k < PRE && k < ntiles) issue_kv(k, false);
      issue_idx(k + STAGES - 1);
      if (k < PRE - 1 || k == STAGES - 2) cp_commit();
    }
    }
  }
  DBG(8);
  }

  if constexpr (IDXF) {    
    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
   
  if (p.rope) {
    tfor<L * 32, NT>(tidL, [&](int e) {    
      const int j = e >> 5, i = e & 31;
      const float ang = (float)(seq + j) * S.ifreq[i];    
      float sv, cv;
       
       
       
      const float kq = rintf(ang * 0.15915493667125702f);
      float rr = fmaf(-kq, 6.25f, ang);
      rr = fmaf(-kq, 0.03318530693650246f, rr);
      rr = fmaf(-kq, 2.430837753308879e-10f, rr);
      __sincosf(rr, &sv, &cv);
      S.cs[e] = cv;
      S.sn[e] = sv;
    });
  }
  csync<WS, NT>();
  DBG(9);

  const float kscale = S.scl[0], vscale = S.scl[1];
  const float sl2 = p.scale_log2 * kscale;

  if constexpr (QDIR && OKC_QP16) {
     
     
    constexpr int RP4 = NT / 4;                     
    constexpr int QP4 = L * 16 / RP4;
    const int q4 = tidL & 3;
#pragma unroll
    for (int ps = 0; ps < QP4; ++ps) {
      const int rr = ps * RP4 + (tidL >> 2);
      const int j = rr >> 4, hh = rr & 15;
      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);
      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);
      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
      float x[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fa = __bfloat1622float2(ha[e]);
        const float2 fb = __bfloat1622float2(hb[e]);
        x[2 * e] = fa.x; x[2 * e + 1] = fa.y;
        x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 16; ++e) x[e] *= r;
      if (p.rope) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = q4 * 8 + e;
          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * sn;
          x[2 * e + 1] = a0 * sn + a1 * c;
        }
      }
      const int hk4 = hh >> 3, gg4 = hh & 7;
      const int r4 = j * 8 + gg4, kb = q4 * 16;
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
    }
  } else
#pragma unroll
  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    float x[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(h2[e]);
      x[2 * e] = f.x;
      x[2 * e + 1] = f.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    ss = oct_sum(ss);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] *= r;
    if (p.rope) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = partL * 4 + e;
        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * sn;
        x[2 * e + 1] = a0 * sn + a1 * c;
      }
    }
     
    uint4 qw;
    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
    // (q_len 8 on sm_90 rounds the query to e4m3: keep it unit-RMS like SGLang does -- pre-scaling it by the
    // kv scale would push it into e4m3's subnormals -- and scale the scores instead)
    const float qsc = (L == 8 && OKC_SM90 && OKC_Q8RAW) ? 1.f : sl2;
    const int hk = hh >> 3, gg = hh & 7;
    if constexpr (QDIR) {
       
       
      const int r = j * 8 + gg, kb = partL * 8;
      *reinterpret_cast<uint2*>(S.q8s + hk * 4096 + r * 64 + (((kb >> 4) ^ ((r >> 1) & 3)) << 4) + (kb & 15)) =
          make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
      continue;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }    
    if constexpr (WS) {    
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }
  if constexpr (QDIR) fence_proxy_async();    
  DBG(10);

  if (has_new) {
#pragma unroll
    for (int ps = 0; ps < KPASS; ++ps) {
      const int rr = ps * RPP + (tidL >> 3);
      const bool live = rr < L * 4;
      const int kind = rr / (2 * L);
      const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      float x[8];
      const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[2 * e] = f.x;
        x[2 * e + 1] = f.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
      ss = oct_sum(ss);    
      float sc;
      if (kind == 0) {
        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] *= r;
        if (p.rope && live) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = partL * 4 + e;
            const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];
            const float a0 = x[2 * e], a1 = x[2 * e + 1];
            x[2 * e] = a0 * c - a1 * sn;
            x[2 * e + 1] = a0 * sn + a1 * c;
          }
        }
        sc = __fdividef(1.f, kscale);    
      } else {                           
        sc = __fdividef(1.f, vscale);
      }
      if (live) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
        }
        uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;
        *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
        const int pos = seq + jj;
        if (pos >= start && pos < end) {
          const int64_t row = S.cloc[jj];
          uint8_t* g8 = const_cast<uint8_t*>(cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));
          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
        }
      }
    }
  }
  csync<WS, NT>();
  DBG(11);
  if (!WS && has_new) {
     
    for (int x = 0; x < (Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1) && x < ntiles; ++x) {
      const int st = x % STAGES;
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const int pos = start + x * TILE + r;
        if (pos >= seq && pos < end) {
          const int sw = ((cchunk ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
        }
      }
    }
    if (L == 8) fence_proxy_async();    
  }

  if constexpr (!FLAT && OKC_FINWARM) {    
    if (p.cs <= 1 && !p.coop && p.nc == 1) fin_nc1<L, NT>(p, b, pair, bp, 0.f, 1);
  }
  DBG(2);
   
  constexpr int MT = 1;                     
  constexpr int NROW = (L == 1) ? 1 : 2;    
  constexpr bool S2 = Cfg<L, FLAT>::S2;
  constexpr bool F2 = Cfg<L, FLAT>::F2;
  const int sid = (S2 || F2) ? (warp >> 3) : 0;             
   
  const int hk = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0)
                          : ((L == 1) ? (warp & 1) : ((warp & 7) >> 2));       
  const int wsub = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp >> 1) : (warp & 3), 0)
                            : ((L == 1) ? (warp >> 1) : (warp & 3));     
  const int wgi = (S2 || F2) ? 2 * sid + hk : hk;           
  float o[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;
  float m_r[NROW], l_r[NROW];
#pragma unroll
  for (int r = 0; r < NROW; ++r) { m_r[r] = -INFINITY; l_r[r] = 0.f; }
   
  const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);
  const int hi_all = OKC_INV2 ? __shfl_sync(0xffffffffu, min(end - 1, seq), 0) : min(end - 1, seq);

  if constexpr (L == 1) {
   
   
  uint32_t qa[4][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
    const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);
    const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);
     
    qa[0][0] = l0.x; qa[0][1] = l0.y; qa[1][0] = l0.z; qa[1][1] = l0.w;
    qa[2][0] = l1.x; qa[2][1] = l1.y; qa[3][0] = l1.z; qa[3][1] = l1.w;
  }

  PROF_INIT
  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
      PROF_MARK(1);
    }

    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;
    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
     
    float s[4][4];
    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {    
      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));
        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));
      }
      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    PROF_MARK(2);
    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      if constexpr (ONE) {    
        l_r[0] = ls;
      } else {
        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : 2); ++ks) {    
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
        const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
        const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    if constexpr (WS) {    
      __syncwarp();
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[(tb + i) % STAGES]);
    }
    PROF_MARK(3);
    PROF_TILE;
    if (i == 0) DBG(13);
  }
  PROF_DUMP;
  } else {
   
   
   
  uint8_t* V16 = S.kvq;
  const uint32_t v16_base = smem_u32(V16) + hk * 8192;
  uint32_t qa8[2][4];    
  if constexpr (!QDIR) {
    const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;    
    const int r0 = 16 * wsub + g, r1 = r0 + 8;
     
    const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;
    const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {         
        const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;
        const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);
        const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);
        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
      }
    }
    if constexpr (PV8ON && OKC_QSS && (!FLAT || OKC_PV8FLAT)) {    
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int kb = 32 * ks + 16 * hf + 4 * tig;
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf];
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf + 1];
        }
      fence_proxy_async();    
    }
    if constexpr (PTR) {    
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.qdone);
    }
  }
  float cr[2] = {0.f, 0.f};    
  const float ssc = OKC_Q8RAW ? sl2 : 1.f;    
   
  const float rssc = OKC_RAWTHR ? __fdividef(1.f, ssc) : 1.f;
  float thr[2] = {0.f, 0.f};
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
  if constexpr (OKC_OZF) {    
    float* oz = &o[0][0][0];    
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(oz[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  const uint32_t ONES = 0x3C003C00u;    
  const int ckey = tid >> 2, cq = tid & 3;    
   
  constexpr bool PV8 = PV8ON && (!FLAT || OKC_PV8FLAT);
   
  const int vwq = warp & 3, vq = lane & 3, vw = lane >> 2;    
  const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;
  const int vr0 = (vw >= 4) ? 1 : 0;    
   
   
   
  const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
  const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
  const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  const uint32_t vld_off1 = vkey_t * ROWB + (((4 * hk + 2 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
   
   
   
  const int stj = lane & 7, stm = lane >> 3;
  const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);
   
  constexpr bool STSM_ON = OKC_STSM || (OKC_STSMPR && !FLAT);
   
   
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  // OKC_ADDR: per-tile addressing as loop-carried values instead of re-derived from (tb + i) every tile: stage and
  // full-barrier phase counters, wgmma descriptors as base + stage offset (the 14-bit address field takes it without
  // a carry: shared memory < 256 KB), the V^T buffer and its descriptor toggled, new-row tiles found once
  constexpr bool ADDR = OKC_ADDR && !S2 && !F2 && (WS || OKC_ADDRN);    
  constexpr bool PVSPLIT = OKC_PVSPLIT && !S2 && !F2 && PV8 && !PTR && !OKC_PROF;
   
  constexpr bool LSYNCN = OKC_LATESYNCN && !WS && !FLAT && PV8 && !PTR && PVSPLIT && !OKC_XP_NOFENCE && !S2 && !F2;
  constexpr bool LSYNC = OKC_LATESYNC && OKC_PK2 && (OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF &&
                         !PTR && WS) && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                         !(OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF) && !OKC_XP_NOFENCE;
  int st_c = tb % STAGES;
  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;
   
   
  const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);
  const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);
  const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;
  const uint64_t kdesc0 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | klo0;
  const uint32_t vsrc0 = smem_u32(S.v);
  uint8_t* vtb_c = S.kvq + hk * 4608;
  const uintptr_t vtb_x = (uintptr_t)(S.kvq + hk * 4608) ^ (uintptr_t)(S.kvq + (2 + hk) * 4608);
  uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));
  const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));
  const int i_new = seq > start ? (seq - start) / TILE : 0;    
   
  uint32_t i_lo = 0, i_span = 0;
  if constexpr (OKC_IRANGE) {
    const int a = (lo_all - start + TILE - 1) / TILE;               
    const int z = hi_all - (TILE - 1) - start;                       
    const int bz = z >= 0 ? z / TILE : -1;
    i_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    i_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }

   
   
   
  uint32_t r_lo = 0x7fffffffu, r_span = 0u;
  if constexpr (OKC_RARE && WS && ADDR) {
    const int hi2 = min(end - 1, seq - 1);
    const int a = (lo_all - start + TILE - 1) / TILE;
    const int z = hi2 - (TILE - 1) - start;
    const int bz = z >= 0 ? z / TILE : -1;
    r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }
  PROF_INIT
   
   
   
  constexpr bool TURN = OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF;
  if constexpr (TURN && OKC_TURN == 1) {
    if (hk == 1 && ntiles > 0) bar_arrive_n(5, 256);    
  }
  int i_cur = 0, cnt2 = 0;    
  if constexpr (F2) {
    if (tid == 0) { S.claim[0] = 0; S.claim[1] = 0; }
    csync<WS, NT>();    
    if ((tid & 127) == 0) S.nxt[wgi][0] = atomicAdd(&S.claim[hk], 1);
    wgsync(wgi);
    i_cur = S.nxt[wgi][0];
  }
  const int nit = S2 ? ((ntiles + 1) >> 1) : (F2 ? 0x7fffffff : ntiles);
  constexpr int UNR8 = WS ? OKC_UNR8 : ((MODE & 3) == 1 ? OKC_UNRN : 1);    
   
#pragma unroll (UNR8)
  for (int jj = 0; jj < nit; ++jj) {
    const int i = S2 ? 2 * jj + sid : (F2 ? i_cur : jj);    
    if (F2 && i >= ntiles) break;
    if constexpr (F2) {
       
      if ((tid & 127) == 0) S.nxt[wgi][(cnt2 + 1) & 1] = atomicAdd(&S.claim[hk], 1);
      const int st0 = (tb + i) % STAGES;
      mbar_wait_full(&S.full[st0], ((tb + i) / STAGES) & 1);    
      const int t0n = start + i * TILE;
      const int nlo = max(t0n, seq), nhi = min(t0n + TILE, end);
      if (nlo < nhi) {    
        for (int e = (tid & 127); e < (nhi - nlo) * 4; e += 128) {
          const int pos = nlo + (e >> 2), c = 4 * hk + (e & 3), r = pos - t0n;
          const int sw = ((c ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
          *reinterpret_cast<uint4*>(S.v + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
        }
        fence_proxy_async();
        wgsync(wgi);
      }
      PROF_MARK(0);
    } else if constexpr (WS) {
       
       
      bool patched;
      if constexpr (ADDR) {
        mbar_wait_full(&S.full[st_c], ph_c);
        if constexpr (OKC_RARE) patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else if constexpr (OKC_PATCHNI && WS) patched = (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else patched = (i >= i_new) && ws_patch(i, st_c);
      } else if constexpr (OKC_INEW) {    
        mbar_wait_full(&S.full[(tb + i) % STAGES], ((tb + i) / STAGES) & 1);
        patched = (i >= i_new) && ws_patch(i, (tb + i) % STAGES);
      } else {
        patched = ws_acquire(i, (tb + i) % STAGES);
      }
      if (patched || i == 0 || !OKC_WGDEC) csync<WS, NT>();
      else if (!(PV8ON && OKC_PV8FLAT && OKC_NOSYNC0)) wgsync(hk);
       
       
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else if constexpr (S2) {
      cp_wait<2>();           
      __syncthreads();        
      PROF_MARK(0);
      if (jj == 0) DBG(12);
      if (2 * jj + 4 < ntiles) issue_kv(2 * jj + 4, true);
      issue_idx(2 * jj + 2 * STAGES - 2);
      cp_commit();
      if (2 * jj + 5 < ntiles) issue_kv(2 * jj + 5, true);
      issue_idx(2 * jj + 2 * STAGES - 1);
      cp_commit();
      PROF_MARK(1);
      if (i >= ntiles) continue;    
    } else {
      if constexpr (!OKC_XNOWAIT) cp_wait<STAGES - 2>();    
      __syncthreads();        
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if constexpr (!OKC_QKFIRST) {
        if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
        issue_idx(i + 2 * STAGES - 2);
        cp_commit();
      }
      PROF_MARK(1);
    }

    const int st = ADDR ? st_c : (tb + i) % STAGES;
    const uint32_t kaddr = smem_u32(S.k + st * STAGE_BYTES) + hk * 64;
    if constexpr (PVSPLIT && OKC_PVSPLIT == 2) {    
      wgmma_wait<0>();
      float* od = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    float s[32];
    wgmma_fence();
    if constexpr (PV8 && OKC_QSS && ADDR) {
      const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, qdesc0, kd);
      else wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
    } else if constexpr (PV8 && OKC_QSS) {
      const uint32_t qaddr = smem_u32(S.q8s) + hk * 4096;
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, desc_sw64(qaddr), desc_sw128(kaddr));
      else wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr), desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr + 32), desc_sw128(kaddr + 32), 1);
    } else {
      wgmma_m64n64k32_e4m3(s, qa8[0][0], qa8[0][1], qa8[0][2], qa8[0][3], desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3(s, qa8[1][0], qa8[1][1], qa8[1][2], qa8[1][3], desc_sw128(kaddr + 32), 1);
    }
    wgmma_commit();
    if constexpr (OKC_QKFIRST && !WS && !Cfg<L, FLAT>::S2) {    
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }
    if constexpr (PV8 && FLAT && OKC_PVW1) {
       
       
      wgmma_wait<1>();
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    }
    uint8_t* V16t = V16 + (OKC_PVW ? (i & 1) * 16384 : 0);    
    const uint32_t v16b = smem_u32(V16t) + hk * 8192;
     
    uint8_t* vtb = ADDR ? vtb_c : S.kvq + ((S2 ? 2 * sid + (jj & 1) : (F2 ? 2 * sid + (cnt2 & 1) : (i & 1))) * 2 + hk) * 4608;
    if constexpr (PV8 && !PTR) {
      if (F2 ? cnt2 == 0 : jj == 0) {    
                                      
        const int wt = tid & 127;
        *reinterpret_cast<uint2*>(S.kvq + ((((S2 || F2) ? 2 * sid : 0) + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
            make_uint2(0x38383838u, 0x38383838u);
      }
    }
    if constexpr (PTR) {
       
    } else if constexpr (PV8) {    
         
      const uint32_t vsrc = ADDR ? vsrc0 + st * STAGE_BYTES : smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
      for (int u = 0; u < (OKC_XP_NOTR ? 0 : 2); ++u) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vsrc + (u ? (OKC_INV2 ? (vld_off0 ^ 32u) : vld_off1) : vld_off0), r0, r1, r2, r3);    
         
        if constexpr (STSM_ON) {
          stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                  prmt(r2, r3, vsel1));
        } else {
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2) {
          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);
          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;    
          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;    
        }
        }
      }
      if (!OKC_XP_NOFENCE && !LSYNC && !LSYNCN) fence_proxy_async();    
    } else {    
      const uint8_t* V = S.v + st * STAGE_BYTES;
      const bool dec = WS && OKC_WGDEC;
      const int wt = tid & 127;
      const int vkey = dec ? (wt >> 1) : ckey;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = dec ? (hk * 4 + (wt & 1) * 2 + u) : (2 * cq + u);
        const int hd = c >> 2, fc = (c & 3) * 2;
        const int ckey = vkey;
        const uint4 vb = lds128(V + ckey * ROWB + ((c ^ (ckey & 7)) << 4));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + ((fc ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.x), cvt2(vb.x >> 16), cvt2(vb.y), cvt2(vb.y >> 16));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + (((fc + 1) ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.z), cvt2(vb.z >> 16), cvt2(vb.w), cvt2(vb.w >> 16));
      }
      if (OKC_PVW) fence_proxy_async();    
    }
    if constexpr (PTR) {
    } else if (LSYNC || LSYNCN) {    
    } else if ((WS && OKC_WGDEC) || PV8) wgsync(wgi);    
    else csync<WS, NT>();                           
    PROF_MARK(2);
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
    if constexpr (PV8) {
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    if constexpr (WS) {    
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[st]);
    }
    if constexpr (PTR) {    
      if (i >= 1 && lane == 0) mbar_arrive(&S.vte[(i - 1) & 1]);
    }
    PROF_MARK(3);

    const bool common_t = OKC_RARE && WS && ADDR && ((uint32_t)i - r_lo <= r_span);
    const int t0 = start + i * TILE;
    const bool interior = OKC_XP_NOMASK || common_t ||
                          (OKC_IRANGE ? ((uint32_t)i - i_lo <= i_span) : ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all)));
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 8 * nt + 2 * tig + e;
            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
          }
      }
    }
     
     
     
    constexpr bool PCHK = OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF && !PTR && WS;
    const bool chk = PCHK && !first;
    float mlo = -INFINITY, mhi = -INFINITY;
    if (!chk) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
      mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    }
     
     
    if (!OKC_RAWTHR && !chk) {
      mlo *= ssc;    
      mhi *= ssc;
    }
    float dlo, dhi;
    if (first) {
      if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }
      if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
      dlo = (mlo == -INFINITY) ? 0.f : mlo;
      dhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = dlo; cr[1] = dhi;
      if (OKC_RAWTHR) { thr[0] = (dlo + 8.f) * rssc; thr[1] = (dhi + 8.f) * rssc; }
      first = false;
    } else {
      dlo = cr[0]; dhi = cr[1];
      if (!PCHK && __any_sync(0xffffffffu, (OKC_LAZYMAX && OKC_RAWTHR) ? (mlo > thr[0] || mhi > thr[1])
                                                                       : (mlo > cr[0] + 8.f || mhi > cr[1] + 8.f))) {
         
        if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }    
        if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
        const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
         
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;    
        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;
        if (OKC_RAWTHR) { thr[0] = (nlo + 8.f) * rssc; thr[1] = (nhi + 8.f) * rssc; }
      }
    }
    PROF_MARK(4);
    if constexpr (TURN && OKC_TURN == 1) bar_sync_n(5 + hk, 256);    
    if constexpr (TURN && OKC_TURN == 2) {    
      if (hk == 1 && jj == 0) bar_sync_n(6, 256);
    }
    if constexpr (PV8) {    
         
      constexpr bool PK2 = OKC_PK2 && PCHK && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_EXH8 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                           !TURN && !PTR;
      if constexpr (PK2) {
        auto pack2 = [&]() {    
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)), ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));
            pa[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)), ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));
            pa[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)), ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));
            pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)), ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));
          }
        };
        pack2();
        if (chk) {
           
          constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
          const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                               (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
          if (__any_sync(0xffffffffu, ov != 0u)) {
             
             
            float hlo = -INFINITY, hhi = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
              hlo = fmaxf(hlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
              hhi = fmaxf(hhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
            }
            hlo = quad_max(hlo) * ssc;
            hhi = quad_max(hhi) * ssc;
            const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
            cr[0] = nlo; cr[1] = nhi;
            dlo = nlo; dhi = nhi;
            pack2();
          }
        }
        float* odk = &o[0][0][0];
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(odk[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        if constexpr (LSYNC) {    
          fence_proxy_async();
          wgsync(wgi);
        }
        wgmma_fence();
        if constexpr (ADDR) {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
        } else {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], desc_sw64(smem_u32(vtb)));
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], desc_sw64(smem_u32(vtb) + 32));
        }
        wgmma_commit();
      }
#pragma unroll
      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {
        const float* sv = s + 16 * kk;
#define EXQ(x, q) ex2(x)
        bool rchk = chk;    
                            
#pragma unroll 1
        for (;;) {    
        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));
        if constexpr (!PCHK) break;
        else {
          if (!rchk) break;
          {
             
            constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
            const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
            if (!__any_sync(0xffffffffu, ov != 0u)) break;
            {
               
               
              float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
              float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
              hlo = quad_max(hlo) * ssc;
              hhi = quad_max(hhi) * ssc;
              const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
              const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
              if (PVSPLIT && (kk == 1 || OKC_PVSPLIT == 2)) {    
                wgmma_wait<0>();
                float* odw = &o[0][0][0];
#pragma unroll
                for (int e = 0; e < 32; ++e) reg_fence(odw[e]);
#pragma unroll
                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
              }
#pragma unroll
              for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
              lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
              cr[0] = nlo; cr[1] = nhi;
              dlo = nlo; dhi = nhi;
              rchk = false;
            }
          }
        }
        }    
#undef EXQ
        if constexpr (PVSPLIT) {    
          float* od = &o[0][0][0];
          if constexpr (LSYNCN) {    
            if (kk == 0) {
              fence_proxy_async();
              wgsync(wgi);
            }
          }
          if (kk == 0) {
#pragma unroll
            for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));
          wgmma_fence();
          if constexpr (ADDR) wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);
          else wgmma_m64n72k32_e4m3(od, lacc, pa[kk], desc_sw64(smem_u32(vtb) + 32 * kk));
          wgmma_commit();
        }
      }
      if constexpr (TURN && OKC_TURN == 1) {    
        if (!(hk == 1 && i == ntiles - 1)) bar_arrive_n(6 - hk, 256);
      }
      if constexpr (TURN && OKC_TURN == 2) {
        if (hk == 0 && jj == 0) bar_arrive_n(6, 256);
      }
      float* od = &o[0][0][0];
      if constexpr (PVSPLIT) {
        if constexpr (OKC_PVSPLIT == 1) {    
          wgmma_wait<0>();
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        }
      } else {
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      if constexpr (PTR) mbar_wait_full(&S.vtf[i & 1], (i >> 1) & 1);    
      wgmma_fence();
      if constexpr (ADDR) {
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], vdesc_c);
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], vdesc_c + 2);
      } else {
        const uint32_t vtaddr = smem_u32(vtb);
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], desc_sw64(vtaddr));
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], desc_sw64(vtaddr + 32));
      }
      wgmma_commit();
      if (OKC_PV8SYNC) {
        wgmma_wait<0>();
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
      }
      }    
    } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t a0 = ex2h2(fmaf(s[8 * ks], ssc, -dlo), fmaf(s[8 * ks + 1], ssc, -dlo));
      const uint32_t a1 = ex2h2(fmaf(s[8 * ks + 2], ssc, -dhi), fmaf(s[8 * ks + 3], ssc, -dhi));
      const uint32_t a2 = ex2h2(fmaf(s[8 * ks + 4], ssc, -dlo), fmaf(s[8 * ks + 5], ssc, -dlo));
      const uint32_t a3 = ex2h2(fmaf(s[8 * ks + 6], ssc, -dhi), fmaf(s[8 * ks + 7], ssc, -dhi));
      const int mi = lane >> 3;
      const int key = 16 * ks + (mi & 1) * 8 + (lane & 7);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4_t(v16b + key * 128 + (((2 * dp + (mi >> 1)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(o[0][2 * dp], a0, a1, a2, a3, b0a, b1a);
        mma_f16(o[0][2 * dp + 1], a0, a1, a2, a3, b0b, b1b);
      }
      mma_f16(lacc, a0, a1, a2, a3, ONES, ONES);
    }
    }
    PROF_MARK(5);
    PROF_TILE;
    if (i == 0) DBG(13);
    if constexpr (F2) {
      i_cur = S.nxt[wgi][(cnt2 + 1) & 1];    
      ++cnt2;
    }
    if constexpr (ADDR) {
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    }
  }
  PROF_DUMP;
  if constexpr ((OKC_PVW && !OKC_PROF) || PV8) {
  wgmma_wait<0>();
  if constexpr (PV8) {
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
  }
  {
    float* od = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  }
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];    
  l_r[1] = 0.25f * lacc[2];
  if constexpr (F2) {    
    csync<WS, NT>();     
    float* xs = reinterpret_cast<float*>(S.k) + (hk * 128 + (tid & 127)) * 37;
    if (sid == 1) {
      xs[0] = m_r[0]; xs[1] = m_r[1]; xs[2] = l_r[0]; xs[3] = l_r[1];
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) xs[4 + 4 * t + e] = o[0][t][e];
    }
    csync<WS, NT>();
    if (sid == 0) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float m1 = xs[r], mm = fmaxf(m_r[r], m1);
        const float f0 = (m_r[r] == -INFINITY) ? 0.f : ex2(m_r[r] - mm);
        const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
        m_r[r] = mm;
        l_r[r] = l_r[r] * f0 + xs[2 + r] * f1;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          o[0][t][2 * r] = o[0][t][2 * r] * f0 + xs[4 + 4 * t + 2 * r] * f1;
          o[0][t][2 * r + 1] = o[0][t][2 * r + 1] * f0 + xs[4 + 4 * t + 2 * r + 1] * f1;
        }
      }
    }
  }
  }

  DBG(3);
   
  bool do_red = true;
  if constexpr (Cfg<L, FLAT>::F2) do_red = (warp >> 3) == 0;    
  if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL)) {
     
     
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;
    if (wsub == 1) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      const float m1 = xs[0], l1 = xs[1];
      const float mm = fmaxf(m_r[0], m1);
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[0] = mm;
      l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
      }
    } else {
      do_red = false;
    }
  }
  if (M_CL) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB = reinterpret_cast<float*>(S.k);
    float* PL = PB + L * 16 * 64;
     
#pragma unroll 1
    for (int ph = 0; ph < (Cfg<L, FLAT>::S2 ? 2 : 1); ++ph) {
    if (ph == 1) csync<WS, NT>();
    const bool mine = !Cfg<L, FLAT>::S2 || (warp >> 3) == ph;
#pragma unroll
    for (int r = 0; r < NROW; ++r) {
      const int mt = r >> 1, hf = r & 1;
      const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
      const int row = j * 16 + hk * 8 + g;
      const float lsum = quad_sum(l_r[r]);
      if (do_red && mine) {
        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
        if constexpr (L == 1) {    
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(&PB[row * 64 + 16 * tig + 8 * c + t4]) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
        } else {                   
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            float2* d2 = reinterpret_cast<float2*>(&PB[row * 64 + 8 * t + 2 * tig]);
            const float2 pv = ph ? *d2 : make_float2(0.f, 0.f);
            *d2 = make_float2(pv.x + o[mt][t][2 * hf] * f, pv.y + o[mt][t][2 * hf + 1] * f);
          }
        }
        if (tig == 0) PL[row] = (ph ? PL[row] : 0.f) + lsum * f;
      }
    }
    }
    csync<WS, NT>();
    cluster_finish<L>(p, S, b, pair, n_keys, chunk);
    return true;
  }
  if (do_red) {    
#pragma unroll
  for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else if constexpr (OKC_RED4) {
       
       
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
         
        const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
        const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
         
        const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
        const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
         
         
        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 8; ++t) red_v2(acc + 8 * t + 2 * tig, o[mt][t][2 * hf] * f, o[mt][t][2 * hf + 1] * f);
    }
    if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);
  }
  }
  if (OKC_MRGDBG) DBG(7);    
   
   
   
  constexpr bool MRG2 = OKC_MERGE2;    
  const bool mode1 = !((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) && !(!FLAT && M_COOP);
   
  if (!((MRG2 && mode1) || (OKC_MERGE3 && !mode1))) gpu_fence();
  csync<WS, NT>();
  if (OKC_MRGDBG) DBG(6);    
  if ((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) {
     
    DBG(4);
  } else if (!FLAT && M_COOP) {
     
     
     
    const int split_idx = (int)blockIdx.x;
    int nact = (OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && ((n_keys + 63) >> 6) >= 2 * p.n_splits)
                   ? p.n_splits : fdiv_i(n_keys + chunk - 1, chunk);     
    int R = fdiv_i(L * 16 + nact - 1, nact);           
     
     
    if (OKC_PINDIV) asm volatile("" : "+r"(nact), "+r"(R));
    if (tid == 0 && (OKC_SENSE || OKC_SENSE2)) {
       
       
       
      int* flag = p.ws_cnt + 24576 + bp;
      int sense;
      if (OKC_SENSE2) sense = S.sense;
      else asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(sense) : "l"(flag) : "memory");
      const int mine = end - start;
      int old;    
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
      if (old + mine == n_keys) {
        p.ws_cnt[bp] = 0;
        asm volatile("st.release.gpu.global.s32 [%0], %1;\\n" ::"l"(flag), "r"(sense ^ 1) : "memory");
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(flag) : "memory");
          if (v != sense) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0 && OKC_PARITY) {
       
       
       
      const int ep = S.sense;
      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start;
      int old;
      if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      else old = atomicAdd(c, mine);
      if (old + mine == n_keys) {
        p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
        p.ws_cnt[24576 + bp] = ep + 1;
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(c) : "memory");
          if (v >= n_keys) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0) {
      atomicAdd(p.ws_cnt + bp, end - start);
      int v;
      SPIN_T0;
      for (int spin = 0;; ++spin) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p.ws_cnt + bp) : "memory");
        if (v >= n_keys) break;
        SPIN_CHECK;
        if (OKC_SPINDBG && spin > (1 << 20)) {    
          printf("coop wait stuck: block (%d,%d,%d) bp %d counter %d n_keys %d start %d end %d\\n", blockIdx.x,
                 blockIdx.y, blockIdx.z, bp, v, n_keys, start, end);
          break;
        }
        __nanosleep(64);
      }
    }
    csync<WS, NT>();
    DBG(4);
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);
    const int part = tid & 7;
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {    
      const int rr = rbase + (tid >> 3);
      const bool act = rr < r_hi;
      const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                             : make_uint4(0, 0, 0, 0);
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
    if (!OKC_SENSE && !OKC_SENSE2 && !OKC_PARITY) {
      csync<WS, NT>();
      if (tid == 0) {    
        const int old = atomicAdd(p.ws_cnt + 16384 + bp, 1);
        if (old + 1 == nact) {
          p.ws_cnt[bp] = 0;
          p.ws_cnt[16384 + bp] = 0;
        }
      }
    }
    DBG(5);
  } else {
  if (tid == 0) {
    const int mine = end - start;
    int old;
    if constexpr (MRG2) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    else old = atomicAdd(p.ws_cnt + bp, mine);
    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
  DBG(4);
  if (S.flag) {
  if constexpr (!MRG2) gpu_fence();
  DBG(14);

   
  if (p.nc == 1) {
    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);
  } else
  {
    const int part = tid & 7;
    for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {
      const int j = rr >> 4, hh = rr & 15;
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
       
      const uint4 graw = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
      sum_copies<L>(p, bp, j, hh, part, true, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
  if (tid == 0) p.ws_cnt[bp] = 0;
  DBG(5);
  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 6] = 1;
  }
  }
  if constexpr (CONT) csync<WS, NT>();    
  else __syncthreads();                    
  }
  return true;
}

 
 
 
template <int L, bool WS, int NT>
__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {
  const int tid = threadIdx.x;
  pdl_wait();    
  const int G = gridDim.x * gridDim.y * gridDim.z;
  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int* cnt = p.ws_cnt + 30000;
  if constexpr (!OKC_MERGE3) gpu_fence();    
  csync<WS, NT>();
  constexpr bool PAR = OKC_PARITY;
  if (tid == 0 && PAR) {
     
     
    cp_wait<0>();
    const int ep = *reinterpret_cast<const volatile int*>(sense_s);
    int* ac = p.ws_cnt + 30020 + (ep & 1);
    int old;
    if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\\n" : "=r"(old) : "l"(ac) : "memory");
    else old = atomicAdd(ac, 1);
    if (old + 1 == G) {
      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;
      p.ws_cnt[30022] = ep + 1;
      if (OKC_W0ENABLE) {    
        cnt[2] = 0;
        cnt[3] = 0;
        for (int k = 0; k < 8; ++k) p.ws_cnt[30030 + k] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(ac) : "memory");
        if (v >= G) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0 && OKC_SENSE) {    
    int sense;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(sense) : "l"(cnt + 8) : "memory");
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\\n" : "=r"(old) : "l"(cnt) : "memory");
    if (old + 1 == G) {
      cnt[0] = 0;
      asm volatile("st.release.gpu.global.s32 [%0], %1;\\n" ::"l"(cnt + 8), "r"(sense ^ 1) : "memory");
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(cnt + 8) : "memory");
        if (v != sense) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0) {
    atomicAdd(cnt, 1);
    int v;
    SPIN_T0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(cnt) : "memory");
      if (v >= G) break;
      SPIN_CHECK;
      __nanosleep(64);
    }
  }
  csync<WS, NT>();
  const int pairs = p.Hkv >> 1;
  const int rows = p.B * pairs * L * 16;
  const int R = fdiv_i(rows + G - 1, G);
  const int r_lo = c * R, r_hi = min(rows, r_lo + R);
  const float vscale = __ldg(p.v_scale);
  const int part = tid & 7;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {    
    const int r = rbase + (tid >> 3);
    const bool act = r < r_hi;
    const int bp = act ? r / (L * 16) : 0, rr = act ? r % (L * 16) : 0;
    const int j = rr >> 4, hh = rr & 15;
    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs), pair = bp - b * pairs;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
  if (!OKC_SENSE && !PAR) {
    csync<WS, NT>();
    if (tid == 0) {
      const int old = atomicAdd(cnt + 1, 1);
      if (old + 1 == G) {
        cnt[0] = 0;
        cnt[1] = 0;
        cnt[2] = 0;    
        cnt[3] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    }
  }
}

template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
   
  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int pair = blockIdx.z;
  DBG(0);
  if (p.pf_chunk && threadIdx.x == 0) {
    const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;
    for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {
      const unsigned n = min(32768u, p.pf_chunk - o);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + o), "r"(n) : "memory");
    }
  }
  if (!FLAT) {
     
    const int split = blockIdx.x, b = blockIdx.y;
    // OKC_SPEC (MODE | 8 builds, chosen by the host for one request with page size 1): SGLang's FA3 backend builds
    // page_table = req_to_token[req_pool_indices, :max_seq_len_k] (sglang/srt/layers/attention/
    // flashattention_backend.py, init_forward_metadata: decode / target-verify paths), so with one request its row
    // is exactly seq_len + q_len entries wide. The page table's width is a shape: speculate seq = width - q_len and
    // issue the window's page indices in the same trip as cache_seqlens; the real length is checked (segment) before
    // the indices are used for any K/V address or any softmax work, and a miss re-runs everything with the real one
    const int seq_ld = __ldg(p.seq_lens + b);
    constexpr bool SPEC = (MODE & 8) != 0;
    int seq = SPEC ? p.max_pages - L : seq_ld;
    for (;;) {
    const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
    const int n_keys = seq + L - lo;
    int chunk = fdiv_i(n_keys + p.n_splits - 1, p.n_splits);
    chunk = (chunk + 15) & ~15;
     
     
     
    if ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2) {
      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
       
       
      if constexpr (OKC_MAGICDIV) chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
      else chunk = fdiv_i(ntl, p.n_splits) << 6;
    }
    int start = lo + split * chunk;
    int end = min(seq + L, start + chunk);
     
     
     
    const int unev_tiles = (n_keys + 63) >> 6;
    const bool unev = OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && unev_tiles >= 2 * p.n_splits;
    if (unev) {
      const int H = p.unev;
      const long long wn = (long long)p.w0 * min(p.n_splits, H) + 1024LL * max(0, p.n_splits - H);
      const long long w_s = (long long)p.w0 * min(split, H) + 1024LL * max(0, split - H);
      const long long w_e = (long long)p.w0 * min(split + 1, H) + 1024LL * max(0, split + 1 - H);
      start = lo + (int)(unev_tiles * w_s / wn) * 64;
      end = min(seq + L, lo + (int)(unev_tiles * w_e / wn) * 64);
    }
    if (start >= end) {
      if (SPEC && seq != seq_ld) { seq = seq_ld; continue; }    
      if ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2) {    
        pdl_wait();      
        cluster_stage_gate<L>(p, S, b, pair);
        if (threadIdx.x == 0) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
        cp_commit();
        float* PB = reinterpret_cast<float*>(S.k);
        float* PL = PB + L * 16 * 64;
        for (int e = threadIdx.x; e < L * 16 * 64; e += NT) PB[e] = 0.f;
        for (int e = threadIdx.x; e < L * 16; e += NT) PL[e] = 0.f;
        cp_wait<0>();
        __syncthreads();
        cluster_finish<L>(p, S, b, pair, n_keys, chunk);
      }
      return;
    }
    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys, split - fdiv_i(split, p.nc) * p.nc,
                                  chunk, 0, SPEC ? seq_ld : seq))
      return;
    seq = seq_ld;    
    }
  }
  if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C) && OKC_PARITY) {
    if (p.coop && threadIdx.x == 0) {    
      cp_async4(smem_u32(&S.sense), p.ws_cnt + 30022);
      cp_commit();
    }
  }
   
  const bool w0on = OKC_W0ENABLE && p.coop && p.resident && p.w0 != 1024 && gridDim.y * gridDim.z == 1 && (gridDim.x & 1) == 0;
   
   
   
  const bool w0g = OKC_W0L1 && OKC_W0ENABLE && !p.coop && p.resident && p.w0 != 1024;
  int w0g_rank = 2, w0g_par = 0, w0g_gen = 0;
  unsigned w0_smid = 0;
  int w0_rank = 0;
  if (OKC_W0EARLY && w0on && threadIdx.x == 32) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(w0_smid));
    const int Gx = gridDim.x;
    w0_rank = ((int)w0_smid < Gx / 2) ? atomicAdd(p.ws_cnt + 31000 + w0_smid, 1) : 2;
    if (w0_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 30003, 1);
    if (OKC_W0BID && w0_rank < 2) p.ws_cnt[31200 + 2 * w0_smid + w0_rank] = (int)blockIdx.x + 1;
    gpu_fence();
    atomicAdd(p.ws_cnt + (OKC_W0SPREAD ? 30030 + (blockIdx.x & 7) : 30002), 1);    
  }
  if (w0g && threadIdx.x == 32) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const int Gx = gridDim.x;
    const unsigned og = (unsigned)atomicAdd(p.ws_cnt + 32000, 1);    
    w0g_gen = (int)(og / (unsigned)Gx);
    w0g_par = w0g_gen & 1;
    if (og % (unsigned)Gx == (unsigned)Gx - 1) {    
      for (int i = 0; i < Gx / 2; ++i) p.ws_cnt[32010 + (w0g_par ^ 1) * 136 + i] = 0;
      p.ws_cnt[32002 + (w0g_par ^ 1)] = 0;
    }
    w0g_rank = ((int)smid < Gx / 2) ? atomicAdd(p.ws_cnt + 32010 + w0g_par * 136 + smid, 1) : 2;
    if (w0g_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 32002 + w0g_par, 1);
    if (w0g_rank < 2) p.ws_cnt[32300 + 2 * smid + w0g_rank] = (int)blockIdx.x + 1;
    __threadfence();
    atomicAdd(p.ws_cnt + 32001, 1);    
    w0g_rank = (w0g_rank & 3) | ((int)smid << 2);
  }
   
   
  constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 && OKC_WSPLIT == 500 && !OKC_BALPART &&
                         !OKC_SWAPHALF && !OKC_DYN;    
  if constexpr (OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR) {
    if (threadIdx.x == 32) {    
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32));
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x < 32) {
    int run = 0;
    int incl_l = 0, n_l = 0, seq_l = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int b = b0 + (int)threadIdx.x;
      int n = 0;
      if (b < p.B) {
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        n = seq + L - lo;
        seq_l = seq;
      }
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)threadIdx.x >= o) incl += t;
      }
      if (b < p.B) S.cum[b + 1] = run + incl;
      run += __shfl_sync(0xffffffffu, incl, 31);
      incl_l = incl;
      n_l = n;
    }
    if (threadIdx.x == 0) S.cum[0] = 0;
    if (fpart && p.B > 32) {    
      __syncwarp();
      if (threadIdx.x == 0) {
        constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
        const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);
        if (SEGKF > 0) {
          auto vmapg = [&](int v) -> int {
            int lo_b = 0, hi_b = p.B - 1;
            while (lo_b < hi_b) {
              const int mid = (lo_b + hi_b + 1) >> 1;
              if (S.cum[mid] + mid * SEGKF <= v) lo_b = mid; else hi_b = mid - 1;
            }
            return min(v - lo_b * SEGKF, S.cum[lo_b + 1]);
          };
          csg = vmapg(csg);
          ceg = vmapg(ceg);
        }
        int b = 0, n = 0;
        while (S.cum[b + 1] <= csg) ++b;
        for (; b < p.B && S.cum[b] < ceg && n < MAXSEG; ++b) {
          S.seg[n][0] = b;
          S.seg[n][1] = max(csg, S.cum[b]) - S.cum[b];
          S.seg[n][2] = min(ceg, S.cum[b + 1]) - S.cum[b];
          ++n;
        }
        S.nseg = n;
        S.fcs = csg;
        S.fce = ceg;
      }
    } else if (fpart) {    
      const unsigned FULL = 0xffffffffu;
      const int lane_f = (int)threadIdx.x;
      constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
      const int excl = incl_l - n_l;
      const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
      const int cf = blockIdx.x, Gf = gridDim.x;
      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);
      if (SEGKF > 0) {    
        const int vstart = excl + lane_f * SEGKF;
        const unsigned ms = __ballot_sync(FULL, lane_f < p.B && vstart <= csf);
        const unsigned me = __ballot_sync(FULL, lane_f < p.B && vstart <= cef);
        const int bs = 31 - __clz(ms), be = 31 - __clz(me);    
        const int is = __shfl_sync(FULL, incl_l, bs), ie = __shfl_sync(FULL, incl_l, be);
        csf = min(csf - bs * SEGKF, is);
        cef = min(cef - be * SEGKF, ie);
      }
       
      const bool in = lane_f < p.B && incl_l > csf && excl < cef;
      const unsigned mi = __ballot_sync(FULL, in);
      const int rank = __popc(mi & ((1u << lane_f) - 1u));
      if (in && rank < MAXSEG) {
        S.seg[rank][0] = lane_f;
        S.seg[rank][1] = max(csf, excl) - excl;
        S.seg[rank][2] = min(cef, incl_l) - excl;
      }
      if (lane_f == 0) {
        S.nseg = min(__popc(mi), MAXSEG);
        S.fcs = csf;
        S.fce = cef;
      }
      if (OKC_PTPF && mi != 0u && csf < cef) {    
        const int b0 = __ffs(mi) - 1;
        const int seqb = __shfl_sync(FULL, seq_l, b0), exb = __shfl_sync(FULL, excl, b0), inb = __shfl_sync(FULL, incl_l, b0);
        const int lob = p.window ? max(0, seqb + 1 - p.window) : 0;
        const int s0 = max(csf, exb) - exb + lob, s1 = min(cef, inb) - exb + lob;
        const int cend = min(min(s1, seqb), s0 + Cfg<L, FLAT>::STAGES * TILE);
        if (cend > s0) {
          const long long e0 = (long long)b0 * p.max_pages + (s0 >> p.lp);
          const long long e1 = (long long)b0 * p.max_pages + ((cend - 1) >> p.lp);
          const long long l0 = e0 >> 5, l1 = e1 >> 5;    
          if (l0 + lane_f <= l1) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((l0 + lane_f) << 5)));
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) FPRO_STAMP(40);
  const int64_t T0 = S.cum[p.B];
   
   
   
  constexpr int SEGK = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
  const int64_t T = T0 + (SEGK > 0 ? (int64_t)(p.B - 1) * SEGK : 0);
  const int c = blockIdx.x, G = gridDim.x;
   
  const int cpart = (OKC_SWAPHALF && (G & 1) == 0) ? (c + (G >> 1)) % G : c;
  int cs, ce;
  if constexpr (fpart) {    
    cs = S.fcs;
    ce = S.fce;
  } else {
    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);
  }    
   
  if (OKC_WSPLIT != 500 && (G & 1) == 0 && gridDim.y * gridDim.z == 1) {
     
     
    const int64_t Th = T * OKC_WSPLIT / 1000, H = G >> 1;
    if (c < H) { cs = (int)((int64_t)c * Th / H); ce = (int)((int64_t)(c + 1) * Th / H); }
    else { cs = (int)(Th + (int64_t)(c - H) * (T - Th) / H); ce = (int)(Th + (int64_t)(c + 1 - H) * (T - Th) / H); }
  }
  int vc = c;
  if (w0on) {
     
     
     
    if (threadIdx.x == (OKC_W0EARLY ? 32 : 0)) {
      unsigned smid = w0_smid;
      int* reg = p.ws_cnt + 31000;
      int* arrive = p.ws_cnt + 30002;
      int* bad = p.ws_cnt + 30003;
      int rank = w0_rank;
      if (!OKC_W0EARLY) {
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        rank = ((int)smid < G / 2) ? atomicAdd(reg + smid, 1) : 2;
        if (rank >= 2 || OKC_W0FORCEBAD) atomicExch(bad, 1);    
        if (OKC_W0BID && rank < 2) p.ws_cnt[31200 + 2 * smid + rank] = c + 1;    
        gpu_fence();
        atomicAdd(OKC_W0SPREAD ? p.ws_cnt + 30030 + (c & 7) : arrive, 1);
      }
      int v;
      SPIN_T0;
      if (OKC_W0SPREAD) {    
        for (int k = 0; k < 8; ++k) {
          const int need = (G - k + 7) >> 3;
          for (;;) {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p.ws_cnt + 30030 + k) : "memory");
            if (v >= need) break;
            SPIN_CHECK;
          }
        }
      } else {
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(arrive) : "memory");
          if (v >= G) break;
          SPIN_CHECK;
        }
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(bad) : "memory");
      if (OKC_W0BID && !v) {    
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(other) : "l"(p.ws_cnt + 31200 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (w0g) {
    if (threadIdx.x == 32) {
      const unsigned smid = (unsigned)w0g_rank >> 2;
      int rank = w0g_rank & 3;
      int v;
      const unsigned need = ((unsigned)w0g_gen + 1u) * (unsigned)G;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p.ws_cnt + 32001) : "memory");
        if ((int)((unsigned)v - need) >= 0) break;
        SPIN_CHECK;
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p.ws_cnt + 32002 + w0g_par) : "memory");
      if (!v) {
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(other) : "l"(p.ws_cnt + 32300 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (SEGK > 0 && !fpart) {    
    auto vmap = [&](int v) -> int {
      int lo_b = 0, hi_b = p.B - 1;    
      while (lo_b < hi_b) {
        const int mid = (lo_b + hi_b + 1) >> 1;
        if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
      }
      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);
    };
    cs = vmap(cs);
    ce = vmap(ce);
  }
  if (!(OKC_DYN && Cfg<L, FLAT>::WS) && cs >= ce) {
    if constexpr (L == 8 || OKC_FLATCOOP1C) {
      if (p.coop) {    
        if constexpr (Cfg<L, FLAT>::WS) {
          if (threadIdx.x < Cfg<L, FLAT>::NT) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
        } else {
          flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
        }
      }
    }
    return;
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_TMAPG && OKC_PGCONT && (L == 1 || OKC_TMAPG8)) {
     
     
    if (threadIdx.x == Cfg<L, FLAT>::NT) {
      for (int st = 0; st < 4; ++st) mbar_init_n(&S.pbar[st], 1);
      S.pgc = 0;
      mbar_fence_init();
    }
    __syncthreads();
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) {    
    if (threadIdx.x == 0) {
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32);
      }
      mbar_fence_init();
    }
  }
   
  if (threadIdx.x == 0 && !fpart) {
    int b = 0, n = 0;
    while (S.cum[b + 1] <= cs) ++b;
    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
      S.seg[n][0] = b;
      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];
      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
      ++n;
    }
    S.nseg = n;
  }
  if (!fpart) __syncthreads();    
  if (threadIdx.x == 0) FPRO_STAMP(41);
  const int nseg = S.nseg;
  auto run_segments = [&](auto role) {
    constexpr int R = decltype(role)::value;
    int tb = 0;
#pragma unroll 1
    for (int k = 0; k < nseg; ++k) {
      const int b = S.seg[k][0];
      const int seq = __ldg(p.seq_lens + b);
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;
      const bool lead = OKC_SEGDBG && p.dbg && threadIdx.x == 0 && k < 2;
      const int dbase = (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 16 + 4 * k;
      if (lead) { p.dbg[dbase] = gtime(); p.dbg[dbase + 2] = s1 - s0; p.dbg[dbase + 3] = b; }
      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            OKC_CONT ? tb : 0, 0,
                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
      }
      if (lead) p.dbg[dbase + 1] = gtime();
    }
    if constexpr (R == 1 && OKC_CONT) cp_wait<0>();    
  };
  if constexpr (Cfg<L, FLAT>::WS) {
     
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::PREGS));
      run_segments(std::integral_constant<int, 1>());
    } else {
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::CREGS));
      run_segments(std::integral_constant<int, 2>());
      if constexpr (L == 8 || OKC_FLATCOOP1C) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }
    }
  } else {
    run_segments(std::integral_constant<int, 0>());
    if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C)) {
      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
    }
  }
}

}   

template <int L>
void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool clusters, bool pdl) {
  Params p = p0;
  p.pdlf = pdl ? 1 : 0;
  static bool init = false;
  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;
  const int sm = p.flat ? sm_f : sm_n;
  static int occ_c = 0, occ_f = 0, nsm = 0;
  static int occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;    
  typedef void (*KernFn)(const Params);
   
   
  static const KernFn kn[4] = {attn_kernel<L, false, OKC_SM90 ? 2 : 0>, attn_kernel<L, false, 1>,
                               attn_kernel<L, false, 2>, attn_kernel<L, false, 3>};
  static const KernFn kn1[4] = {nullptr, attn_kernel<L, false, 5>, nullptr, attn_kernel<L, false, 7>};
  static const KernFn ksp[4] = {nullptr, attn_kernel<L, false, 9>, nullptr, attn_kernel<L, false, 11>};
  static const KernFn ksp1[4] = {nullptr, attn_kernel<L, false, 13>, nullptr, attn_kernel<L, false, 15>};
  if (!init) {
    for (int m = 1; m < 4; ++m)
      cudaFuncSetAttribute(kn[m], cudaFuncAttributeMaxDynamicSharedMemorySize, (m == 2 && OKC_CLSM > sm_n) ? OKC_CLSM : sm_n);
    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    for (int m = 1; m < 4; m += 2) {
      cudaFuncSetAttribute(ksp[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }
    cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
    if (clusters) cudaFuncSetAttribute(kn[2], cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    init = true;
  }
   
  Params q = p;
  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;
  q.resident = (p.flat && grid.y * grid.z == 1 && (int)grid.x == 2 * nsm && occ_f >= 2) ? 1 : 0;
  q.unev = (OKC_UNEV && !p.flat && q.coop && p.w0 != 1024 && grid.y * grid.z == 1 && (int)grid.x > nsm &&
            (int)grid.x <= 2 * nsm) ? nsm : 0;
  const bool cl = cs > 1 && clusters;
  KernFn kern = p.flat ? (KernFn)attn_kernel<L, true> : kn[cl ? 2 : (q.coop ? 3 : 1)];
  if (!p.flat && !cl && !(OKC_UNEV && q.unev)) {
     
     
    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)
                                    : (long long)p.max_pages * p.P;
    if (OKC_TCHUNK && (long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }
   
  if (!p.flat && !cl && !(OKC_UNEV && q.unev) && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {
    const bool one = kern != kn[q.coop ? 3 : 1];
    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];
  }
  if (OKC_COOPGUARD && q.coop && !p.flat && !cl) {
     
    int occ_k = occ_c;
    KernFn plain = kn[1];
    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }
    if (kern == ksp[3]) { occ_k = occ_sp; plain = ksp[1]; }
    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }
    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {
      q.coop = 0;
      kern = plain;
    }
  }
  if (!p.flat && cs > 1 && !clusters) kern = kn[0];    
  if (cl || pdl) {
     
     
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, 1, 1);
    cfg.dynamicSmemBytes = (cl && OKC_CLSM > sm) ? OKC_CLSM : sm;    
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (cl) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = cs;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 0;
      ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
 
static CUtensorMap row_map(void* base, long long rows, long long row_bytes) {
  static EncodeTiledFn encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled unavailable");
    encode = reinterpret_cast<EncodeTiledFn>(fn);
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {128, 1};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.ws_cnt = ws_cnt.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;
  TORCH_CHECK(P > 0 && (P & (P - 1)) == 0, "page size must be a power of two");
  TORCH_CHECK(k_cache.numel() / (Hkv * 64) < (1LL << 31), "cache rows must fit in 31 bits");
  p.lp = 0;
  while ((1 << p.lp) < P) ++p.lp;
  p.max_pages = (int)page_table.size(1);
  p.window = (int)window; p.rope = (int)rope; p.n_splits = (int)n_splits; p.flat = (int)flat; p.nc = (int)nc;
  p.ns_magic = (n_splits >= 2 && n_splits <= 1024) ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits) : 0u;
  TORCH_CHECK(n_splits <= 1024 || !OKC_MAGICDIV, "OKC_MAGICDIV: n_splits <= 1024");
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.cs = (int)cs; p.ncl = (int)ncl;
  p.scratch = scratch.data_ptr<float>();
  p.bar = bar.data_ptr<int>();
  p.ld_qkv = (int)qkv.size(-1);
  p.row_bytes = (int)(Hkv * 64);
  TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");
  p.lrb = 0;
  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;
  p.off32 = (k_cache.numel() < (1LL << 32) && v_cache.numel() < (1LL << 32)) ? 1 : 0;    
  TORCH_CHECK(!OKC_OFF32 || ws_acc.numel() < (1LL << 31), "OKC_OFF32: workspace too large");
  TORCH_CHECK(B * (Hkv / 2) <= 4096, "at most 4096 (request, kv pair) slots in the counter workspace");
  TORCH_CHECK(!OKC_OFF32 || qkv.numel() < (1LL << 31), "OKC_OFF32: qkv too large");
  TORCH_CHECK(!OKC_OFF32 || (out.numel() < (1LL << 31) && page_table.numel() < (1LL << 31)), "OKC_OFF32: out / page table too large");
  p.scale_log2 = (float)scale_log2;
  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));
  p.pf = reinterpret_cast<const uint8_t*>(pf_ptr);
  p.pf_chunk = pf_ptr ? (unsigned)((pf_bytes / ((int64_t)grid.x * grid.y * grid.z)) & ~15ll) : 0u;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static int major = -1;
  if (major < 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  }
  p.tma = (OKC_TMA && (tma & 1) && major >= 9) ? 1 : 0;
  p.coop = OKC_COOP ? (int)((tma >> 1) & 1) : 0;
  p.w0 = (int)((tma >> 2) & 4095);
  if (p.w0 == 0) p.w0 = 1024;
  if (p.tma) {
    const long long rows = k_cache.numel() / (Hkv * 64);
    p.tk = row_map(k_cache.data_ptr(), rows, Hkv * 64);
    p.tv = row_map(v_cache.data_ptr(), rows, Hkv * 64);
  } else {
    memset(&p.tk, 0, sizeof(p.tk));
    memset(&p.tv, 0, sizeof(p.tv));
  }
  memset(&p.tpt, 0, sizeof(p.tpt));
  const bool pdl = ((tma >> 16) & 1) && major >= 9;    
  if (L == 1) launch<1>(p, grid, stream, (int)cs, major >= 9, pdl);
  else launch<8>(p, grid, stream, (int)cs, major >= 9, pdl);
}
'''
ATT7_CPP_SRC = r\"\"\"
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes);
\"\"\"
_ATT7 = None


def _att7():
    global _ATT7
    if _ATT7 is None:
        import os
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major >= 9 else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
        _ATT7 = load_inline(name=f"muse_att_v298w1pf_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,
                            functions=["run"], extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],
                            verbose=False)
    return _ATT7


_NUM_SMS7 = {}


def _workspace7(n_rows, device, owner):
    \"\"\"Persistent FP32 accumulators (zero between calls: the last CTA resets
    them), each followed by sentinel words -- see _guarded / _check_guards.\"\"\"
    key = str(device)
    ws = owner["rows"].get(key)
    if ws is None or ws[0].numel() < n_rows * 64:
        assert not torch.cuda.is_current_stream_capturing(), "attention rows must be warmed before capture"
        n = max(n_rows, 1 << 15)
        ws = tuple(_guarded(t) for t in (
            torch.zeros(n * 64, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device)))
        owner["rows"][key] = ws
        owner["retained"].append(ws)  
    return ws


def _num_sms7(device):
    key = str(device)
    if key not in _NUM_SMS7:
        _NUM_SMS7[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS7[key]


_TUNED7 = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),   
}


OKC_FLATCOOP = 1
OKC_PDL = 0      
OKC_FLAT1B = 0   
OKC_PLAIN8 = 1   
OKC_F2 = 0   
OKC_S2 = 0   
OKC_W0_L1 = 1024   
OKC_W0_PR = 1024   
OKC_W0_L8 = 1135   


def _plan7(B, pairs, L, window, max_pages, P, device):
    \"\"\"Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all
    requests; clusters of cs CTAs reduce through DSMEM (ncl clusters per request); nc workspace copies.\"\"\"
    sms = _num_sms7(device)
    target = 2 * sms
    if not window and 1 < B <= 512:
        ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target
        return dict(n_splits=max(1, ft // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    if L == 8 and OKC_S2 and torch.cuda.get_device_capability(device)[0] >= 9:
        target = sms   
    if not window:
        if L == 1 and B == 1 and OKC_FLAT1B:
            return dict(n_splits=max(1, ((target * 21) // 22) // pairs), flat=1, cs=0, ncl=1, nc=4)
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
    if L == 1 and reqs < 24:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16 and OKC_PLAIN8:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)   
    if L == 8 and reqs >= 16 and OKC_PLAIN8:
        n = max(1, min(tiles, (target if reqs >= 24 else sms) // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16   
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and ncl > 1:
        ncl -= 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))




def _cluster_bufs7(n_ctas, L, device, owner):
    \"\"\"sm_80 emulation buffers for the cluster path (unused on sm_90 beyond a
    pointer), each followed by sentinel words -- see _guarded.\"\"\"
    key = str(device)
    need = n_ctas * (L * 16 * 64 + 1024)
    cur = owner["cluster"].get(key)
    if cur is None or cur[0].numel() < need or cur[1].numel() < n_ctas:
        assert not torch.cuda.is_current_stream_capturing(), "attention cluster buffers must be warmed before capture"
        cur = tuple(_guarded(t) for t in (
            torch.zeros(max(need, 1 << 20), dtype=torch.float32, device=device),
            torch.zeros(max(n_ctas, 1 << 12), dtype=torch.int32, device=device)))
        owner["cluster"][key] = cur
        owner["retained"].append(cur)
    return cur


_OKC_VISIBLE_MOE_0 = r'''#ifndef OKC_FINCOL
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
#define OKC_SPLIT 0   
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
#define OKC_WS 0   
#endif
#define OKC_WGSX 0   
#ifndef OKC_WGS
#define OKC_WGS 0   
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
#define OKC_FINPF 0    
#endif
#define OKC_WSNR 4   
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
 
 
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
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
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);   
'''

_OKC_VISIBLE_MOE_1 = r'''constexpr int NRB = OKC_WS ? OKC_WSNR : 2;    
#if OKC_WS || OKC_WGS
 
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;      
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;    
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;    
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;    
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;   
 
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");    
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;    
constexpr int KS = 256;          
constexpr int XP_FT = KS + 8;    
constexpr int NST = 3;           
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;   
  const unsigned char* wAf; const unsigned char* wBf;   
  const unsigned char* wAu; const unsigned char* wBu;   
  const unsigned char* wAn; const unsigned char* wBn;   
  const unsigned char* wAs; const unsigned char* wBs;   
  __nv_bfloat16* xA; __nv_bfloat16* xB;   
  int* xready;                            
  int* doneAs;                            
'''

_OKC_VISIBLE_MOE_2 = r'''  uint32_t zero;   
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;         
  int* doneA;       
  __nv_bfloat16* inter;   
  float* ybuf;      
  float* ss;        
  float* ss1;       
  __nv_bfloat16* out;
  unsigned long long* dbg;
'''

_OKC_VISIBLE_MOE_3 = r'''};

struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];   
#else
  float red[1][1][1][RED_W];       
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];   
  int wsr[4];                           
#endif
#if OKC_WGS
  uint64_t bars[24];   
  int wglast[4];       
  int wcp[JMAX + 1];   
  int wcpB[JMAX + 1];  
  int jrow[JMAX + 1];  
  uint32_t wtot;       
  uint32_t wtotB;      
#else
  uint64_t bars[12];   
#endif
#if OKC_WGS
  int wgr[4];          
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];      
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;           
  float* ssp;        
  int uslot[8];     
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];    
  short pos[EMAX];    
};

 
 
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
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
'''

_OKC_VISIBLE_MOE_4 = r'''extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();   
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));
  return v;
}

 
 
 
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
   
   
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
   
   
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}


 
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
'''

_OKC_VISIBLE_MOE_5 = r'''  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS   
  {   
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
     
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;    
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
#if !OKC_WGS   
     
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

 


 
 
 
 
'''

_OKC_VISIBLE_MOE_6 = r'''

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
 
 
 
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
 
 
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
'''

_OKC_VISIBLE_MOE_7 = r'''__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\\n .reg .pred p;\\n WAIT_%=:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n @!p bra WAIT_%=;\\n}\\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;   
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
'''

_OKC_VISIBLE_MOE_8 = r'''}

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

constexpr int X1P = 512 + 8;   
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
   
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
'''

_OKC_VISIBLE_MOE_9 = r'''  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;   
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
   
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
   
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
   
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);   
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
  if (nU < OKC_NS) CTA_SYNC();   
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
   
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
      v = v0 + i;   
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;   
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
     
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
'''

_OKC_VISIBLE_MOE_10 = r'''#endif
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
    CTA_SYNC();   
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
      for (int jj = jdmin; jj <= jdmax; ++jj) {   
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
'''

_OKC_VISIBLE_MOE_11 = r'''  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
   
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
   
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
'''

_OKC_VISIBLE_MOE_12 = r'''    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
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
    CTA_SYNC();   
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

 
 
 
 
 
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
'''

_OKC_VISIBLE_MOE_13 = r'''    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
}

 
 
 
 
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {   
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\\n}\\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};

 
 
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
   
   
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];   
  uint32_t keep = 0;     
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
       
       
      const uint32_t K92 = 0x6D806D80u;   
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
      wg_wait<1>();   
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);   
}

 
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
'''

_OKC_VISIBLE_MOE_14 = r'''  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;   
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;   
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {   
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
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
  int f = S.fseq;   
  int xq = 0;       
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
    const int ntc = ntile;   
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
      if (nat) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");   
      CTA_SYNC();   
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
      CTA_SYNC();   
      ++xq;
       
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
     
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
  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}

template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}



 
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;   
#endif
  if (OKC_FATSTATIC && S.njF > 0) {   
'''

_OKC_VISIBLE_MOE_15 = r'''    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);   
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
   
   
  const int n = min(OKC_EARLYN, v1 - v0);
'''

_OKC_VISIBLE_MOE_16 = r'''    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}

 
#if OKC_WS
 
 
 
 
 
 
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
'''

_OKC_VISIBLE_MOE_17 = r'''  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                              
  uint32_t lnw0 = 0u, lnw1 = 0u;                 
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
'''

_OKC_VISIBLE_MOE_18 = r'''    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
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
     
     
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));   
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);    
    }
  }
  if (PB && warp < OKC_WSBLK && warp < nU) {   
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
'''

_OKC_VISIBLE_MOE_19 = r'''  WS_JOIN();   
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {   
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
    if (lane == 0) mbar_arrive1(S.wsre + bb);   
     
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
  WS_JOIN();   
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
     
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);   
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
     
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
 
 
 
 
 
 
 
 
#define OKC_WGSNX 3                              
constexpr int WGS_NS = 3;                        
constexpr int WGS_NX = OKC_WGSNX;                
constexpr int WGS_NT = FT_T / 8;                 
'''

_OKC_VISIBLE_MOE_20 = r'''constexpr int WGS_XS = WGS_NT * 4096;            
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;   
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;                   
constexpr int WGS_FIXW = WGS_NT * 512;           
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;   
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
'''

_OKC_VISIBLE_MOE_21 = r''' 
 
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207   
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
#define OKC_WGSREORD 1   
#define OKC_GATHB 8   
#define OKC_WGSPR 1   
#define OKC_WGSDFIN 2   
'''

_OKC_VISIBLE_MOE_22 = r'''#define OKC_WGSGS 2   
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
 
 
 
 
 
 
 
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;   
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);   
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();   
'''

_OKC_VISIBLE_MOE_23 = r'''  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;   
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;   
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
'''

_OKC_VISIBLE_MOE_24 = r'''      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
 
 
constexpr int WGS_XROWS = EMAX + 8 * JMAX;   
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
   
   
  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");
'''

_OKC_VISIBLE_MOE_25 = r'''  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
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
      if (gs == g1) gs = g0;   
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
'''

_OKC_VISIBLE_MOE_26 = r'''      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
 
 
__device__ void wgs_gather(const Params& p, Smem& S) {
'''

_OKC_VISIBLE_MOE_27 = r'''  const int R = S.jrow[nj];   
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
   
   
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
'''

_OKC_VISIBLE_MOE_28 = r'''  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
     
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
     
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0))), w1 = fork_bf16(1.f + __bfloat162float(__ldg(p.normw + n0 + 8)));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
'''

_OKC_VISIBLE_MOE_29 = r'''    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;   
  if constexpr (DF) {
   
   
   
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
   
   
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
'''

_OKC_VISIBLE_MOE_30 = r'''   
   
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
     
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
'''

_OKC_VISIBLE_MOE_31 = r'''        mbar_arrive1(S.bars + BW_E + ws);   
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;   
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;       
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
           
           
'''

_OKC_VISIBLE_MOE_32 = r'''          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
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
'''

_OKC_VISIBLE_MOE_33 = r'''  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {   
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
     
     
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};   
'''

_OKC_VISIBLE_MOE_34 = r'''}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
   
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {   
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
     
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
 
 
 
'''

_OKC_VISIBLE_MOE_35 = r'''}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
 
'''

_OKC_VISIBLE_MOE_36 = r'''   
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
   
#if OKC_TINY
  if (nF > 0) __trap();   
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);   
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
   
   
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

 
 
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;   
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
'''

_OKC_VISIBLE_MOE_37 = r'''  DBG(0);
   
   
  asm volatile("griddepcontrol.wait;\\n" ::: "memory");
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
   
   
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
     
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
      const int jj = (v01 + i) >> RB_A_LOG;    
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
     
     
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
    }
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {   
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
      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);   
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
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
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
   
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {   
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
     
     
'''

_OKC_VISIBLE_MOE_38 = r'''    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
  route(p, S);
#if OKC_WGS && OKC_WGSPR
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");   
#endif
  DBG(1);
'''

_OKC_VISIBLE_MOE_39 = r'''#if OKC_WS
  if (tid == 0) {   
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {   
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
  WS_JOIN();   
#endif
   
   
  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
   
   
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
   
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
'''

_OKC_VISIBLE_MOE_40 = r'''  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
   
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;   
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
'''

_OKC_VISIBLE_MOE_41 = r'''          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {   
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
'''

_OKC_VISIBLE_MOE_42 = r'''          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
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
      CTA_SYNC();   
    }
  }
#elif OKC_FINCOL
   
  constexpr int FSEG = HD / (NT * 4);   
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {    
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
  DBG(5);   
}

}   

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

static int g_okc_pdl = 0;   

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }
'''

_OKC_VISIBLE_MOE_43 = r'''
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }   
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
'''

_OKC_VISIBLE_MOE_44 = r'''  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
'''

_OKC_VISIBLE_MOE_45 = r'''#else
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

MOE_CUDA_SRC = (
r'''
 
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1   
#define OKC_F1EARLY 1   
#define OKC_WGSFINSS 1
#ifndef OKC_MFIN
#define OKC_MFIN 1    
#endif
#ifndef OKC_B4L
#define OKC_B4L 0     
#endif
#ifndef OKC_MFP
#define OKC_MFP 15
#endif
#ifndef OKC_MFS
#define OKC_MFS 0     
#endif
#ifndef OKC_MFQ
#define OKC_MFQ 47    
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
#endif   
'''
 + _OKC_VISIBLE_MOE_0
 + _OKC_VISIBLE_MOE_1
 + r'''  int* mfl;                               
'''
 + _OKC_VISIBLE_MOE_2
 + r'''  float* sspart;  // bet4_claude B4FIN: per-(route, 16-column tile) sum-of-squares partials (WGS only)
'''
 + _OKC_VISIBLE_MOE_3
 + r'''#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
'''
 + _OKC_VISIBLE_MOE_4
 + r'''  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
'''
 + _OKC_VISIBLE_MOE_5
 + r'''#ifndef OKC_SC6S
#define OKC_SC6S 0    
#endif
#ifndef OKC_SC6W
#define OKC_SC6W 0    
#endif
constexpr int UNIT_CODES = 32768, UNIT_SC = OKC_SC6S ? 3072 : 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
'''
 + _OKC_VISIBLE_MOE_6
 + r'''#if OKC_SC6S
 
#undef SCDK
#define SCDK(word, b, k4) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k4) + 0x60006000u)
#endif
#endif
#define SCK ((OKC_SC6S ? 4u : 16u))
 
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
'''
 + _OKC_VISIBLE_MOE_7
 + r'''  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);
'''
 + _OKC_VISIBLE_MOE_8
 + r'''  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;   
'''
 + _OKC_VISIBLE_MOE_9
 + r'''    const uint4 sc = sc_load(slot, warp, lane);
'''
 + _OKC_VISIBLE_MOE_10
 + r'''  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;   
'''
 + _OKC_VISIBLE_MOE_11
 + r'''    const uint4 sc = sc_load(slot, warp, lane);
'''
 + _OKC_VISIBLE_MOE_12
 + r'''  if (threadIdx.x == 0) {
'''
 + _OKC_VISIBLE_MOE_13
 + r'''  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
'''
 + _OKC_VISIBLE_MOE_14
 + r'''    if (threadIdx.x == 0) S.preissued = 0;
'''
 + _OKC_VISIBLE_MOE_15
 + r'''  if (threadIdx.x == NT - 32) {
'''
 + _OKC_VISIBLE_MOE_16
 + r'''  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;   
'''
 + _OKC_VISIBLE_MOE_17
 + r'''    const uint4 sc = sc_load(slot, warp, lane);
'''
 + _OKC_VISIBLE_MOE_18
 + r'''  const int lane = threadIdx.x & 31;
'''
 + _OKC_VISIBLE_MOE_19
 + r'''constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;   
constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;   
'''
 + _OKC_VISIBLE_MOE_20
 + r'''constexpr int MF_CE = 0, MF_EXIT = 64, MF_XR = 128, MF_XRS = 96, MF_DA = 384, MF_INTS = 4096;
static_assert(JMAX <= MF_XRS && MF_DA + 2 * JMAX * 16 <= MF_INTS, "MFIN flag region");
 
 
__device__ __forceinline__ int mf_gen(const Smem& S) {
#if OKC_MFQ & 32
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];\\n" : "=r"(v) : "r"(smem_u32(&S.gen)));
  return v;
#else
  return S.gen;
#endif
}
__device__ __forceinline__ int* mf_xr(const Params& p, const Smem& S) { return p.xready + ((OKC_MFIN && (OKC_MFQ & 2)) ? mf_gen(S) * MF_XRS : 0); }
__device__ __forceinline__ int* mf_da(const Params& p, const Smem& S) { return p.doneAs + ((OKC_MFIN && (OKC_MFQ & 1)) ? mf_gen(S) * (JMAX * 16) : 0); }
'''
 + _OKC_VISIBLE_MOE_21
 + r'''#define OKC_WGSFCI 4   
'''
 + _OKC_VISIBLE_MOE_22
 + r'''  if ((threadIdx.x & 31) != 0) return;
'''
 + _OKC_VISIBLE_MOE_23
 + r'''       
      if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
'''
 + _OKC_VISIBLE_MOE_24
 + r'''  wgs_ranges(p, S);   
  __syncwarp();
  WS_JOIN();   
  if ((threadIdx.x & 31) != 0) return;
'''
 + _OKC_VISIBLE_MOE_25
 + r'''        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }   
      }
       
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }
'''
 + _OKC_VISIBLE_MOE_26
 + r'''  const int tid = threadIdx.x, nj = S.njF + S.njS;
'''
 + _OKC_VISIBLE_MOE_27
 + r'''          {   
              
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
  asm volatile("fence.proxy.async.global;\\n" ::: "memory");
  CTA_SYNC();
  if (tid == 0) {   
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
 
 
 
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
#if !OKC_B4FIN
#define OKC_YBF 0    
#else
#define OKC_YBF 0
#endif
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
'''
 + _OKC_VISIBLE_MOE_28
 + r'''#if OKC_YBF
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
  if constexpr (!PB) { asm volatile("fence.proxy.async.global;\\n" ::: "memory"); }
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
#if OKC_MFIN && (OKC_MFP & 1)
  if (pd.kind == 4) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.mfl + MF_CE + ((OKC_MFP & 8) ? pd.slot : (int)(mf_gen(S) * 32 + S.jE[j]))), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
#endif
  if (pd.kind == 1) {
     
     
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
'''
 + _OKC_VISIBLE_MOE_29
 + r'''#if OKC_MFIN && (OKC_MFP & 2)
  if (PB) {
    asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1) : "memory");
  }
#endif
  if (!PB) {
    asm volatile("fence.proxy.async.global;\\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
 
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = (OKC_SC6W ? 4u : 16u) + p.zero;   
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];   
   
   
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
'''
 + _OKC_VISIBLE_MOE_30
 + r'''     
     
#if OKC_SC6W
     
     
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
        asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
'''
 + _OKC_VISIBLE_MOE_31
 + r'''#if OKC_SC6W
          const uint32_t sgp = sgs[js];
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
'''
 + _OKC_VISIBLE_MOE_32
 + r'''         
         
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
     
    wg_wait<0>();
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    mbar_arrive1(S.bars + BX_E + xs);
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
'''
 + _OKC_VISIBLE_MOE_33
 + r'''#if OKC_MFIN && (OKC_MFP & 1)
  else pd = WgsPend{4, u, (OKC_MFP & 8) ? (int)(mf_gen(S) * 32 + S.jE[j]) : -1, 0};   
#endif
'''
 + _OKC_VISIBLE_MOE_34
 + r'''#define OKC_WGSU 160    
#define OKC_WGSUB 140   
__device__ __forceinline__ int wgs_unitc(int pb) { return pb ? OKC_WGSUB : OKC_WGSU; }
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;   
    accB += ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
 
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  if (nj == 0) return 0;   
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;   
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = (uint32_t)wgs_unitc(pb), cu = 16u * cs + U;
  const uint32_t r = x - (uint32_t)pre[lo], un = r / cu, rem = r - un * cu;
  const uint32_t st = rem < U ? 0u : min(15u, (rem - U) / cs);   
  return (lo << (9 - pb)) + (int)(un * 16u + st);
'''
 + _OKC_VISIBLE_MOE_35
 + r''' 
 
 
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
 
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t sj = (uint32_t)(s & ((1 << (9 - pb)) - 1)), cs = (uint32_t)wgs_cost(pb, S.jC[j]), U = (uint32_t)wgs_unitc(pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (sj >> 4) * (16u * cs + U) + U + ((sj & 15u) + 1u) * cs;   
'''
 + _OKC_VISIBLE_MOE_36
 + r'''  const int tid = threadIdx.x;
'''
 + _OKC_VISIBLE_MOE_37
 + r'''    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
     
#if OKC_MFIN && (OKC_MFQ & 8)
     
     
    const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);   
#endif
'''
 + _OKC_VISIBLE_MOE_38
 + r'''#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)
  {   
    const int o = 1 - S.gen;
    for (int i = blockIdx.x * NT + tid; i < 32 + MF_XRS + JMAX * 16; i += gridDim.x * NT) {
      if (i < 32) p.mfl[MF_CE + o * 32 + i] = 0;
      else if (i < 32 + MF_XRS) p.mfl[MF_XR + o * MF_XRS + (i - 32)] = 0;
      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
  }
#else
  {   
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#endif
'''
 + _OKC_VISIBLE_MOE_39
 + r'''#if !(OKC_MFIN && OKC_WGS && (OKC_MFQ & 4))
'''
 + _OKC_VISIBLE_MOE_40
 + r'''#endif
   
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
      if (tid < 8) {   
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
      float2 yv[8];   
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
      CTA_SYNC();   
    }
  }
#elif OKC_WGS && OKC_WGSFINSS
   
   
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {   
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
'''
 + _OKC_VISIBLE_MOE_41
 + r'''#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#else
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#endif
'''
 + _OKC_VISIBLE_MOE_42
 + r'''
std::vector<int64_t> okc_attrs() {
  cudaFuncAttributes a0, a1;
  cudaFuncGetAttributes(&a0, okc::moe_fused<false>);
  cudaFuncGetAttributes(&a1, okc::moe_fused<true>);
  return {a0.numRegs, (int64_t)a0.localSizeBytes, (int64_t)a0.sharedSizeBytes, a1.numRegs, (int64_t)a1.localSizeBytes, (int64_t)a1.sharedSizeBytes};
}
'''
 + _OKC_VISIBLE_MOE_43
 + r'''#if OKC_WGS
  p.mfl = ib + 8 + 1024 + okc::EMAX;
#else
  p.mfl = nullptr;
#endif
  p.ybuf = p.ss + okc::EMAX;
#if OKC_WGS && OKC_B4FIN
   
  TORCH_CHECK(x.size(0) <= 32 && x.size(0) * 8 <= okc::EMAX - 64, "B4FIN build is for M <= 32");
  p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD;
#else
  p.sspart = nullptr;
#endif
'''
 + _OKC_VISIBLE_MOE_44
 + r'''#if OKC_MFIN
  p.xready = p.mfl + okc::MF_XR;
#endif
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#if OKC_MFIN
  p.doneAs = p.mfl + okc::MF_DA;
#endif
  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS), "MFIN needs the flag region in scratch[0]");
'''
 + _OKC_VISIBLE_MOE_45
)
MOE_CPP_SRC = r\"\"\"
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
std::vector<int64_t> okc_attrs();
\"\"\"
_MOE = None
_MOE_B4 = None


_MOE_TINY = None
_MOE_TINY2 = None
_MOE_MID = None
MOE_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")              
MOE_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2")    
MOE_TINY_MAX_M = 4
MOE_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')  
MOE_TINY2_MAX_M = 8
MOE_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')      
MOE_MID_MAX_M = 16


_MA_PRMT_SCALES = True
_MA_SFX_IF_PRMT = ("-DOKC_PRMTSC=1",)


def _ma_scales_prmt_exact(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True


MOE_HAS_PDL = False   


SC6_WGS = True      
_MA_SFX = ()   


_MA_EXTS = {}
_MA_ALT_SRCS = {}
_MA_MAIN_NS = 3
_MA_PDL = False   


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


def _moe_build(defines):
    \"\"\"Their own extension builder (lifted verbatim as _ma_ext: flags, DQ7 define, module name), called with the
    build's defines + the weights' suffix, exactly as their kernel_fn calls _ext.\"\"\"
    ext = _ma_ext(tuple(defines) + tuple(_MA_SFX))
    if MOE_HAS_PDL:
        ext.okc_set_pdl(0)
    return ext


def _moe(M=None):
    \"\"\"Their kernel, in the build their kernel_fn would pick for this token count.\"\"\"
    global _MOE, _MOE_TINY, _MOE_TINY2, _MOE_MID, _MOE_B4
    if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        if _MOE_TINY is None:
            _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
        return _MOE_TINY
    if M is not None and _moe_tiny2_ok() and M <= MOE_TINY2_MAX_M:
        if _MOE_TINY2 is None:
            _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
        return _MOE_TINY2
    if M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
        if _MOE_MID is None:
            _MOE_MID = _moe_build(MOE_MID_DEFINES)
        return _MOE_MID
    if (M is not None and 16 < M <= 32 and "_main_defines" in globals()
            and tuple(_main_defines(_MA_SFX)) == tuple(MOE_DEFINES)):  # bet4_claude: decode 32 gets the B4FIN WGS build
        if _MOE_B4 is None:
            _MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",))
        return _MOE_B4
    if _MOE is None:   
        _MOE = _moe_build(_main_defines(_MA_SFX) if "_main_defines" in globals() else MOE_DEFINES)
    return _MOE




def _moe_tiny2_ok():
    \"\"\"Their kernel_fn's rule: the TINY2 build only with the PRMT scale decode when it is a WGS build.\"\"\"
    return bool(MOE_TINY2_DEFINES) and (bool(_MA_SFX) or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES))






_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


DQ7 = True


def _stored_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  
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
                    d = 4 if b == 0 else 28   
                pos[(R, up, b)] = (o + d) % 32
    assert sorted(pos.values()) == list(range(32))
    return pos


_POS = _stored_pos()


def _pack_codes(codes, rows):
    \"\"\"codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk).\"\"\"
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _pack_scales(sf, rows):
    \"\"\"sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h].\"\"\"
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  
    s = s.permute(0, 3, 2, 5, 4, 1)  
    return s.contiguous().view(-1)


def _sc6_ok(*sfs):
    \"\"\"SC6: every scale byte is in 0x40..0x7F (bits 7:6 == 01), so byte - 0x40 is a lossless 6-bit value.\"\"\"
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x40) | (b > 0x7F)).any()):
            return False
    return True




def _sc6_wgs(s):
    \"\"\"SC6 WGS piece scales: [..., 1024] bytes (thread t = 8 bytes s0..s7 = [j][h]) -> [..., 768]: plane W (t * 4 B: bytes 4 v_n | piece_n,
    n = 0..3) then plane H (t * 2 B: bytes 4 v_(4+n) | piece_(4+n)); v6 = pieces 0..2, v7 = pieces 3..5 (see harness/make_sc6.py).\"\"\"
    lead = s.shape[:-1]
    v = s.view(*lead, 128, 8).to(torch.int32) - 0x40
    assert int(v.min()) >= 0 and int(v.max()) <= 63
    v6, v7 = v[..., 6], v[..., 7]
    piece = [(v6 >> 0) & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
    wb = torch.stack([(v[..., n] << 2) | piece[n] for n in range(4)], -1).to(torch.uint8)
    hb = torch.stack([(v[..., 4 + n] << 2) | piece[4 + n] for n in range(2)], -1).to(torch.uint8)
    return torch.cat([wb.reshape(*lead, 512), hb.reshape(*lead, 256)], -1).contiguous()



def _fat_order(codes, scales):
    \"\"\"Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]].\"\"\"
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)


def _unit_order(codes, scales):
    \"\"\"Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit).\"\"\"
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)


def _pack_codes_nat(codes, rows):
    \"\"\"Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1).\"\"\"
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _kperm_src():
    \"\"\"KPERM3: stage position p -> stage column k. p = 16 s + c (k16 step s = 4 lg + j, fragment column c of lane
    t = (c >> 1) & 3, i = (c & 1) | ((c >> 3) << 1)); k = 16 (4 t + j) + 4 lg + i, so lane t's columns of step j in all
    four load groups are exactly scale block 4 t + j.\"\"\"
    p = torch.arange(256)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    k = 16 * (4 * t + j) + 4 * lg + i
    assert sorted(k.tolist()) == list(range(256))
    return k


def _kperm_codes(codes):
    \"\"\"codes uint8 [R, K/2] (low nibble first) -> the same codes with K permuted inside every 256-K stage.\"\"\"
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
    nib = nib.index_select(2, _kperm_src().to(codes.device)).reshape(R, K // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def _kperm_rows(rowsA):
    \"\"\"Gate/up row order whose 16-row block r16, lane g holds inter column colmap(P), P = 8 r16 + g, where
    colmap(P) = 256 (P >> 8) + _kperm_src()[P & 255] is the column phase B reads at K position P (the phase-A
    epilogue then stores at its usual position).\"\"\"
    r = rowsA.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rowsA.device)
    cm = (P >> 8) * 256 + _kperm_src().to(rowsA.device)[P & 255]
    return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)


def _wgs_copy(codes, sf, rows, sc6=False):
    \"\"\"WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout).\"\"\"
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)  
    if sc6:
        s = _sc6_wgs(s)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7", "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0", "-DOKC_FPF=5")


def _main_defines(sfx):
    if not sfx and "-DOKC_WGS=1" in tuple(MOE_DEFINES):
        return MAIN_FALLBACK_DEFINES
    return MOE_DEFINES


def _wgs_on():
    return any("-DOKC_WGS=1" in tuple(d) for d in (MOE_DEFINES, MOE_TINY_DEFINES, MOE_TINY2_DEFINES, MOE_MID_DEFINES))


bfp__SK2_DELAY = int(os.environ.get("SK2_DELAY_ESC", "0"))  


bfp_CUDA_SRC = r'''
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>


__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
   
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

 
 
 
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

 
 
 
 
 
 
 
@@MBAR@@
 
 
__device__ __noinline__ void okc_spin_trap() { asm volatile("trap;"); }
__device__ __forceinline__ void okc_spin_ge(const int* p, int target, int ns) {
  for (unsigned it = 0;; ++it) {
    if (*((volatile const int*)p) - target >= 0) return;
    if (ns > 0) __nanosleep(ns);
    if (it >= (1u << 24)) okc_spin_trap();
  }
}
@@BODY@@
'''

bfp_MBAR_HELPERS = r'''
 
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
 
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\\n selp.u32 %0, 1, 0, p;\\n}\\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\\n .reg .b64 st;\\n mbarrier.arrive.shared.b64 st, [%0];\\n}\\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\\n .reg .b64 st;\\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\\n}\\n" :: "r"(bar), "r"(bytes) : "memory");
}
 
 
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(l2_pol_ef()) : "memory");
}
 
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\\n"
                 " selp.u32 %0, 1, 0, p;\\n}\\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}

'''

bfp_WG_HELPERS = r\"\"\"
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


\"\"\"


def bfp_wgmma_defs(bns):
    \"\"\"Generate wgmma_call overloads for m64nBNk16 bf16 with A from registers.\"\"\"
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f\"\"\"
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\\\n"
      ".reg .pred p;\\\\n"
      "setp.ne.b32 p, %{pi}, 0;\\\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\\\n"
      "}}\\\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}\"\"\")
    return "\\n".join(out)


bfp_TMA_SRC = r'''
 
 
 
 
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
   
   
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                        
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;   
   
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
   
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
   
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;    
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);        
  const int nst = NS / KS;               
   
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
     
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;   
  __syncthreads();

  if (warp == NW) {
     
    const int st0 = NSTAGE < nst ? NSTAGE : nst;    
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
    }
    return;
  }

   
  const int rg = rb * RG + warp % RG;    
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
     
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
   
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;   
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);    
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;   

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;    
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;   
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
        if constexpr (WGM) fence_async_smem();    
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
     
    constexpr int NL = CHUNK / 512;   
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (dg == 3) {
      if (wr[0].x == 0x9e3779b9u && wr[NL - 1].w == 0x7f4a7c15u) acc[0][0] += 1.f;    
      continue;
    }
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;    
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();    
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  if (dg >= 1) {
    if (acc[0][0] == 1.2345e-30f) out[threadIdx.x] = cvt_out<OutT>(acc[0][1]);
    return;    
  }
   
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
     
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;    
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {    
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
   
   
   
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + rb, target, 20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }   
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
'''

bfp_MAP_SRC = r'''
#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(l2_pol_el()) : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

 
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

'''

bfp_GEMM_SRC = r'''

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;        
  constexpr int B_BYTES = BN * 128;        
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;            
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;           
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
     
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if constexpr (AS)    
          bulk_g2s(sbase + slot * ST_BYTES, Wst + ((size_t)rb * nkb_all + kb0 + i) * A_BYTES, A_BYTES, full0 + 8 * slot);
        else
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

   
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
     
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    okc_spin_ge(counters + rb, target, 32);
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
'''


bfp_SK2_SRC = r'''

#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
 
 
static int g_sk2_pdl = 0;
void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }
#define SK2_DIAG 0
 
 
 
 
 
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK, bool FUSE>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {
  const int dgv = SK2_DIAG ? dg : 0;    
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
   
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);    
  constexpr int CP = R + 1;                          
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
   
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                        
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));    
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);    
   
   
   
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);    

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
   
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                       
  const int gs0 = (crank * tst) / CL;                   
  const int nst = ((crank + 1) * tst) / CL - gs0;       
   
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
   
   
   
  constexpr int NSTR = FUSE ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;
  const int nstt = nst + NSTR;
   
  const long long pcx_d0 = (long long)((pdl >> 4) & 31) * 448, pcx_g = (long long)((pdl >> 9) & 31) * 448;
  const int pcx_e = (pdl >> 14) & 7;
   
  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;

  if (tid == 0) {
     
     
     
     
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL == 1)    
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
     
     
    const long long pcx_t0 = clock64();
    for (int st = 0; st < P; ++st) {
      const long long pcx_tg = st < pcx_e ? 0 : pcx_d0 + (long long)(st - pcx_e) * pcx_g;    
      if (pcx_tg > 0) { while (clock64() - pcx_t0 < pcx_tg) { } }
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
    if constexpr (CL == 1) {
       
       
      if (COMP) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
      }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      if (pdl) asm volatile("fence.proxy.async.global;" ::: "memory");   
      if (!(dgv & 4))
        for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
      if (COMP) {
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
   
   
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if (pdl) asm volatile("fence.proxy.async.global;" ::: "memory");   
  if constexpr (CL == 1) {    
    if (tid == 0 && !(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
  }
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");    

  if (warp == NW) {
     
    if (lane == 0) {
      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
        } else {   // router piece: global router stage rq * 4 + j of tile rt
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
     
     
     
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      int pe = 1;
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);    
          continue;
        }
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
         
         
         
         
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {    
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

   
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
       
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
       
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
       
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
       
      if (dgv & 8) {    
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");    
  if constexpr (FUSE) {
    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
    float accr[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
        }
    }
     
    if constexpr (COMP) mbar_wait(cbar, 0);    
    named_bar(1, NC);
    float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4, tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
     
     
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);    
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();    

   
  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);    
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");    
     
     
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {    
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[2];
      int na = 0;
      if (CL > 1) {
        at_[na].id = cudaLaunchAttributeClusterDimension;
        at_[na].val.clusterDim.x = CL;
        at_[na].val.clusterDim.y = 1;
        at_[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl > 0) {    
        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at_[na].val.programmaticStreamSerializationAllowed = 0;
        ++na;
      }
      lc.attrs = at_;
      lc.numAttrs = na;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,
                                        wsr, outr, part, rcnt, pdl));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,
                                                                               wsr, outr, part, rcnt, 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {
  @@SK2F_DISPATCH@@
  TORCH_CHECK(false, "sk2f: no config");
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
'''


bfp_WIDE_SRC = r'''
 
 
 
 
 
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
   
   
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
   
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                     
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

   
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
  };
  const bool producer = tid == 0;
  if (producer) {    
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
   
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

   
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {    
          uint32_t* af = afrag[h];
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                       : "r"(wa + h * 2048 + j * 512));
#pragma unroll
          for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
       
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
    }
    }   
     
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

   
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    const int old = atomicAdd(counters + zh * nrb + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + zh * nrb + rb, target, 32);    
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  __threadfence();
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;    
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
'''


def bfp_wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f\"\"\"
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\\\n"
      ".reg .pred p;\\\\n"
      "setp.ne.b32 p, %{p}, 0;\\\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\\\n"
      "}}\\\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}\"\"\")
    return "\\n".join(out)


bfp_CPP_SRC = \"\"\"
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,
          int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr);
void sk2_set_pdl(int64_t n);
\"\"\"

bfp_w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
bfp_COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}

bfp__ext = None


bfp_TMA_CONFIGS = {  
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
bfp_USE_WGM = True   
bfp_WGM_SM80_TEST = False  


def bfp__tma_nstage(nw, comp, sm90, ks, hl=False):
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = (112 if sm90 else 40) * 1024
    return max(2, min(8, budget // (nw * chunk)))


bfp_TMA_TU = {}  


bfp_TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def bfp__tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(bfp_TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(bfp_TMA_ALT.items())]
    return items


def bfp__tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in bfp__tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], bfp__mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = bfp__tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  
        n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else bfp_w_rows[layer]
        wgm = bool((sm90 or bfp_WGM_SM80_TEST) and comp and bfp_USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    keys = list(lines)
    for ln in keys:
        for cfg in lines[ln]:
            bfp_TMA_TU[cfg] = 0
    return bfp_TMA_SRC.replace("@@TMA_DISPATCH@@", "\\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


bfp_GEMM_CONFIGS = {  
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


bfp_DIAG_LAYERS = ()
bfp_GEMM_DIAG = False   


def bfp_swizzle_tiles(w, bm):
    \"\"\"W [N, K] -> [N/bm][K/64][bm rows x 128 B] with the 128B swizzle (16-byte chunk c of row r at c ^ (r % 8)).\"\"\"
    n, k = w.shape
    t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  
    r = torch.arange(bm, device=w.device).view(bm, 1)
    p = torch.arange(8, device=w.device).view(1, 8)
    src_c = (p ^ (r % 8)).view(1, 1, bm, 8, 1).expand(t.shape[0], t.shape[1], bm, 8, 8)
    return torch.gather(t, 3, src_c).contiguous()


def bfp__gemm_src(sm90):
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(bfp_GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and bfp_GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in bfp_GEMM_CONFIGS})
    return (bfp_GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", bfp_wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines))))


bfp_GQ_SRC = r'''
 
 
 
 
 
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);    
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);    

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);    
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
     
     
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
   
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();    
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
     
     
     
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;           
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;    
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");    
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
     
     
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;    
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;                
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
  if (CL > 1 && crank > 0) {
     
    float* mine = reinterpret_cast<float*>(gbase);    
    if constexpr (NWG > 1) named_bar(1, NWG * 128);    
#pragma unroll
    for (int q = 0; q < NACC; ++q) mine[(wg * NACC + q) * 128 + t] = acc[q];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar(1, NWG * 128);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");    
    if (tid == 0) {
      const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * NWG * NACC * 128);
      uint32_t peer_r, qbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(peer_r), "r"((uint32_t)__cvta_generic_to_shared(mine)), "r"(NWG * NACC * 128 * 4), "r"(qbar_r)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\\n\\tcp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    return;
  }
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
#pragma unroll
    for (int r = 0; r < CL - 1; ++r)
#pragma unroll
      for (int q = 0; q < NACC; ++q) acc[q] += peer[((r * NWG + wg) * NACC + q) * 128 + t];
  }
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) {
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(acc[4 * j]);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(acc[4 * j + 1]);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
    kern<<<N / BM, NWG * 128 + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mW, mX, op, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
'''

bfp_GQ_CONFIGS = {  
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 7),
}
bfp_USE_GQ = True


def bfp__gq_src(sm90):
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(bfp_GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return bfp_GQ_SRC.replace("@@GQ_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))


def bfp__gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def bfp__run_gq(x, layer, m, n, k, state):
    nwg, cl, ns = bfp_GQ_CONFIGS[(layer, bfp__gq_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    bfp__get_ext().gq(state["raw"][layer], x, out, m, n, k, bfp__gq_bn(m), nwg, cl, ns)
    return out


bfp_WIDE_CONFIGS = {  
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def bfp__wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def bfp__wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(bfp_WIDE_CONFIGS.items()):
        hl = layer == "router"
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return bfp_WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))


def bfp__get_ext():
    \"\"\"Compile everything as one CUDA translation unit with load_inline.  no_implicit_headers: the CUDA file includes
    only the ATen headers it needs (torch/types.h alone costs ~45 s of nvcc time).\"\"\"
    global bfp__ext
    if bfp__ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", bfp_WG_HELPERS if sm90 else "")
        defs = bfp_wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + bfp__tma_srcs(sm90) + bfp_MAP_SRC + bfp__gemm_src(sm90) + bfp__wide_src(sm90) + bfp__sk2_src(sm90)
                    + bfp__gq_src(sm90))
        bfp__ext = load_inline(
            name=f"dl_okc_bf16proj_v86y4_{arch}" + (f"_dly{bfp__SK2_DELAY}" if bfp__SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\\n" + bfp_CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={bfp__SK2_DELAY}"] if bfp__SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return bfp__ext


def bfp_pack_fragments(w: torch.Tensor) -> torch.Tensor:
    \"\"\"W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order.\"\"\"
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def bfp_pack_compressed(w: torch.Tensor):
    \"\"\"Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it.\"\"\"
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  
    e71 = hi & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = (e71 < base) | (e71 > base + 7)
    n_esc = int(esc.sum())
    if n_esc > n * k // 200:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lo_f = bfp_pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = bfp_pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * 0x01010101,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def bfp_build_patch_table(c, n_cfg, k, splits, rg):
    \"\"\"Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits).\"\"\"
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def bfp_build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    \"\"\"Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16.\"\"\"
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


bfp_SK2_CONFIGS = [  
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv16_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv8_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler16h_r100', 'unpooler', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
]
bfp_SK2_PICK = {  
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o8h_r88',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8_r176',
    ('qkv_gate', 16): 'qkv16_r176',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16h_r100',
    ('unpooler', 32): 'unpooler32_r144',
}
bfp_SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
bfp_USE_SK2 = True
bfp_WIDE_BF16 = False  


def bfp__sk2_name(layer, m):
    return bfp_SK2_PICK[(layer, bfp__sk2_tokr(m))]


def bfp__sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def bfp__sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(bfp_SK2_K[layer] // (ups * 64)) // CL)   
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   
    rp = R + ((40 - (R % 32)) % 32)
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap = 200
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap = int(tail[1])
    ring = min(cap * 1024, 232448 - extra)
    nstage = max(2, min(16, nst, ring // sst))
    return hl, nstage


def bfp__sk2_src(sm90):
    lines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if bfp_SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    flines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if name != "pooler":
            continue
        for tokr in (8, 16, 32):
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
            flines.append(f"if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                          f"{nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                          f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                          f"rcnt.data_ptr<int>()); return; }}")
    return (bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\\n  ".join(lines))
            .replace("@@SK2F_DISPATCH@@", "\\n  ".join(flines)))


def bfp_sk2_tileunits(w):
    \"\"\"W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9.\"\"\"
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def bfp_sk2_compress(w):
    \"\"\"Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable.\"\"\"
    n, k = w.shape
    tu = bfp_sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "delta": delta.contiguous()}


def bfp_sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    \"\"\"Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows.\"\"\"
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


def bfp_sk2_table(comp, n_cfg, R):
    \"\"\"Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, delta bits).\"\"\"
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["delta"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.view(torch.int32)
    return tab.contiguous(), pmax


def bfp_sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (bfp_USE_SK2 and state["sm90"]):
        return
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = bfp_sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            comp = state.setdefault("sk2_comp", {}).get(layer)
            if comp is None:
                comp = bfp_sk2_compress(w.contiguous())
                state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW)   
            if lkey not in shared:
                shared[lkey] = (bfp_sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW),) + tuple(bfp_sk2_table(comp, n_rows, R))
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)


def bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = bfp__sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def bfp_fused_pooler_router(x, state):
    \"\"\"Pooler and router of one decode step in ONE launch (m <= 32, sm_90): x [m, 8192] bf16 ->
    (pooler [m, 4096] bf16, router [m, 256] fp32). Same numerics as the separate sk2 kernels: 12-bit lossless pooler
    weights with exact escape corrections, router as BF16 hi + lo with FP32 accumulation (quarter-K partials added in
    a fixed order).\"\"\"
    m = x.shape[0]
    assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
    ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
    rstream = state["sk2"]["router"][1]
    if "fused_part" not in state:
        state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
        state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, bfp__sk2_tokr(m))
    return out, outr


bfp_SK2_PDL = 0   


def bfp__mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def bfp_prepare(weights):
    ext = bfp__get_ext()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    old_paths = not (sm90_ and bfp_USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * bfp_w_rows[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * bfp_w_rows[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in bfp_GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    if bfp_GEMM_DIAG:
        state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   
            src = torch.stack([bfp_pack_fragments(hi), bfp_pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   
            nwg_r = bfp_WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
            for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            c = bfp_pack_compressed(w) if name in bfp_COMPRESS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = bfp_pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
            key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
            ckey = (name, splits, rgc, wsn, ks, n_cfg)
            if ckey not in state["cnt"]:
                state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  
            if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
                key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
                if key not in state["streams"]:
                    state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
                ckey = (name, splits, rgc, wsn, ks, n_cfg)
                if ckey not in state["cnt"]:
                    state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
                if (name, splits, rgc, n_cfg) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
            if lname != name or not comp:
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = bfp_build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["old_paths"] = old_paths
    bfp_sk2_prepare(weights, state)
    if bfp_GEMM_DIAG and state["sm90"]:
        state["wtile"], state["diag_cnt"] = {}, {}
        for layer in bfp_DIAG_LAYERS:
            state["wtile"][layer] = bfp_swizzle_tiles(state["raw"][layer], 128)
            for a_s in (0, 1):
                for sp_ in sorted({bfp_GEMM_CONFIGS[(layer, 128)][1], bfp_GEMM_CONFIGS[(layer, 256)][1], 1}):
                    state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(bfp_w_rows[layer] // 16, dtype=torch.int32, device=dev)
    bfp__selftest(weights, state)
    bfp__autotune(weights, state)
    torch.cuda.empty_cache()
    return state


bfp_ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
bfp_WIDE_SM80_TEST = False  
bfp_ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def bfp__route(layer, m, sm90):
    for mmax, r in (bfp_ROUTE_ROUTER if layer == "router" else bfp_ROUTE_BF16):
        if m <= mmax:
            if r == "gemm" and not sm90:
                return "torch"
            if r == "wide" and not (sm90 or bfp_WIDE_SM80_TEST):
                return "torch"
            return r
    return "torch"


def bfp__route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, bfp__sk2_name(layer, m), bfp__sk2_tokr(m))
    if route == "skinny":
        return (layer, route, bfp__mt_key(m))
    if route == "wide":
        return (layer, route) + bfp__wide_key(m)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


bfp_SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def bfp__selftest(weights, state):
    \"\"\"Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on.\"\"\"
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(1234)
    for layer, w in weights.items():
        n, k = w.shape
        for m in bfp_SELFTEST_M:
            nn = 1024 if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in bfp__candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else bfp__route_key(layer, m, ch[0]))
                try:
                    out = bfp__run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError:  
                    err = float("inf")
                if not err < tol:
                    state["disabled"].add(key)


def bfp__run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = bfp__mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = bfp__get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, bfp_w_rows[layer] // 16, ks, dg)
    else:
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, bfp_w_rows[layer] // 16, ks, dg)
    return out


def bfp__run_wide(x, layer, m, n, k, state):
    tok, nz = bfp__wide_key(m)
    nwg, splits, _ = bfp_WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    bfp__get_ext().wide(state["streams"][(layer, 0, rgc, 1, 4, n)], x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def bfp__run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = bfp_GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    bfp__get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def bfp__run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def bfp__run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind == "skinny":
        return bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return bfp__run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return bfp__run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return bfp__run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return bfp__run_gq(x, layer, m, n, k, state)
    return bfp__run_torch(x, layer, m, n, k, state)


def bfp__static_choice(layer, m, state):
    route = bfp__route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (bfp__sk2_name(layer, m) in state.get("sk2", {})
                              and bfp__route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not bfp_WIDE_BF16:
            route = "torch"
    if route != "torch" and bfp__route_key(layer, m, route) in state.get("disabled", ()):
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and bfp__route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = bfp__mt_key(m)
        return ("skinny", bfp_TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def bfp__candidates(layer, m, state):
    \"\"\"Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS.\"\"\"
    c = [bfp__static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in bfp_TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", bfp_TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and bfp__route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and bfp__sk2_name(layer, m) in state.get("sk2", {})
            and bfp__route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (bfp_USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, bfp__gq_bn(m)) in bfp_GQ_CONFIGS
            and bfp__route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


bfp_AUTOTUNE = True
bfp_TUNE_M = (1, 16, 32, 8, 128, 256)   


def bfp__autotune(weights, state):
    \"\"\"Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m).\"\"\"
    state["choice"] = {}
    if not bfp_AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)   
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    reps = 24
    for m in bfp_TUNE_M:
        for layer, w in weights.items():
            n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = bfp__candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                bfp__run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    bfp__run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()


def bfp__nearby_choice(layer, m, state):
    \"\"\"An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket.\"\"\"
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "gq" and (layer, bfp__gq_bn(m)) not in bfp_GQ_CONFIGS:
        return None
    return ch


def bfp_kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or bfp__nearby_choice(layer, m, state) or bfp__static_choice(layer, m, state)
    out = bfp__run(choice, x, layer, m, n, k, state)
    if bfp_GEMM_DIAG:
        bfp__diag(x, layer, m, n, k, state, choice)
    return out


def bfp__diag(x, layer, m, n, k, state, choice):
    \"\"\"Extra kernels for the judge's per-kernel profile (outputs discarded).\"\"\"
    if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
        for dg in (1, 2, 3):   
            bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)


def _bfp_prepare(weights, st):
    \"\"\"Their prepare() on the five dense matrices, guarded: any failure just disables the backend.\"\"\"
    st["bfp"] = None
    try:
        w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        st["bfp"] = bfp_prepare(w5)
    except Exception as ex:  
        print("bfp disabled:", repr(ex)[:160], flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}


BFP_PDL_N = 2   


PCX_CFG = {}   


def _pcx_val(role, T):
    P, d0, g, e = (tuple(PCX_CFG.get((role, T), PCX_CFG.get((role, 0), (BFP_PDL_N, 0, 0, 0)))) + (0,))[:4]
    return (1 << 30) | P | (d0 << 4) | (g << 9) | (e << 14)


def _bfp_pdl(T, role=None):
    \"\"\"Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue.\"\"\"
    global bfp_SK2_PDL
    bfp_SK2_PDL = _pcx_val(role, T) if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
    _bfp_pdl(m, name)
    return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


_MA_MMAX = 384
_MA_EMAX = _MA_MMAX * 8


WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16   
WGS_XROWS = _MA_EMAX + 8 * (32 + _MA_EMAX // 32 + 8)  


def _df_inter(inter):
    \"\"\"DF: inter rows [128, 256) start (and, after every MID launch, end) as bf16 0xFFFF = not yet written.\"\"\"
    inter[128 * 4096:256 * 4096].view(torch.int16).fill_(-1)
    return inter


def _ma_make_scratch(dev):
    return [torch.zeros(8 + 1024 + _MA_EMAX + 4096, dtype=torch.int32, device=dev),  
            torch.zeros(_MA_EMAX + _MA_EMAX * 4096 + WGS_FIXF, dtype=torch.float32, device=dev),
            _df_inter(torch.zeros((_MA_EMAX + 2 * WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev))]  

SCRATCH_GUARD = 1 << 16   
SCRATCH_SENTINEL = 0x5EADBEEF


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
                  lut.contiguous(), weights["post_expert_norm"].contiguous(),  
                  torch.cat([_fat_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_fat_order(x, y) for x, y in zip(wB, sB)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wB, sB)])]
    if "_wgs_on" in globals() and _wgs_on():   
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], _kperm_rows(rowsA), st["ma_sc6"])
                                     for e in range(32)]))
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, st["ma_sc6"])
                                     for e in range(32)]))
    st["ma_scratch"] = [_guarded(t) for t in _ma_make_scratch(dev)]








@triton.jit
def _gemm_splitk(X, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()   
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _gemm_splitk_ng(X, G, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    \"\"\"Like _gemm_splitk with x = rmsnorm_per_64(X) * sigmoid(G) applied on the fly (X bf16, G fp32).\"\"\"
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    g_ptr = G + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr, mask=tmask, other=0.0)
        x3 = tl.reshape(x, [BK // 64, 64, BT])
        r = tl.rsqrt(tl.sum(x3 * x3, 1) / 64.0 + 1e-5)
        xn = tl.reshape(x3 * r[:, None, :], [BK, BT]) * tl.sigmoid(gt)
        acc = tl.dot(w, xn.to(tl.bfloat16), acc)
        w_ptr += BK
        x_ptr += BK
        g_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _normgate(X, G, OUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
    x = x * tl.sigmoid(tl.load(G + t * 1024 + h * 64 + d))
    tl.store(OUT + t * 1024 + h * 64 + d, x.to(tl.bfloat16))


def dense_ng(x, gate, w, T, state=None):
    N, K = w.shape
    if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
        xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
        _normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        return dense(xn, w, T, state, "o")
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    S = 1
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


TUNED_T = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket(T):
    \"\"\"The tuned token count whose choices apply to T (the next one at or above it).\"\"\"
    for t in TUNED_T:
        if T <= t:
            return t
    return TUNED_T[-1]


BFP_MAX_T = 32   


def dense(x, w, T, state=None, name=None):
    \"\"\"Returns (partials [S, T, N] fp32, S). The tuner picks cuBLAS or the split-K Triton kernel per
    (token count, weight matrix): cuBLAS wins on the o projection, the Triton one on the pooler+router.\"\"\"
    N, K = w.shape
    pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
    if pick == "bfp" and T <= BFP_MAX_T and name in _BFP_LAYER and state.get("bfp") is not None:
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
    if pick is True or pick == "bfp":       
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    if T > 32:
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    tiles = (N // BN) * triton.cdiv(T, BT)
    S = 1
    while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
        S *= 2
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


@triton.jit
def _f32_to_e4m3(x):
    \"\"\"FP32 -> E4M3FN bits, RNE; match torch.to(float8_e4m3fn).\"\"\"
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(x)
    scaled = a * 512.0
    lower = scaled.to(tl.int32)
    fraction = scaled - lower.to(tl.float32)
    sub = lower + ((fraction > 0.5) | ((fraction == 0.5) & ((lower & 1) != 0))).to(tl.int32)
    ab = a.to(tl.uint32, bitcast=True)
    r = ab + 0x7FFFF + ((ab >> 20) & 1)
    e = (r >> 23).to(tl.int32) - 120
    m = (r >> 20) & 7
    normal = (e << 3) | m.to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    code = tl.where((a > 464.0) | (a != a), 0x7F, code)
    return code | sign.to(tl.int32)



@triton.jit
def _qk_prep(P, S: tl.constexpr, T, POS, LOC, INVF, KC, VC, KS, VS, QOUT, Q8OUT, GOUT, qk_factor,
             ROPE: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    base = P + t * 2304
    hq = tl.arange(0, 16)[:, None]
    hk = tl.arange(0, 2)[:, None]
    i = tl.arange(0, 32)[None, :]
    d = tl.arange(0, 64)[None, :]
    qe = tl.zeros([16, 32], dtype=tl.float32)
    qo = tl.zeros([16, 32], dtype=tl.float32)
    ke = tl.zeros([2, 32], dtype=tl.float32)
    ko = tl.zeros([2, 32], dtype=tl.float32)
    v = tl.zeros([2, 64], dtype=tl.float32)
    g = tl.zeros([1024], dtype=tl.float32)
    gi = tl.arange(0, 1024)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()
    for s in range(S):
        b = base + s * T * 2304
        qe += tl.load(b + hq * 64 + 2 * i)
        qo += tl.load(b + hq * 64 + 2 * i + 1)
        ke += tl.load(b + 1024 + hk * 64 + 2 * i)
        ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
        v += tl.load(b + 1152 + hk * 64 + d)
        g += tl.load(b + 1280 + gi)
    qe = qe.to(tl.bfloat16).to(tl.float32)
    qo = qo.to(tl.bfloat16).to(tl.float32)
    ke = ke.to(tl.bfloat16).to(tl.float32)
    ko = ko.to(tl.bfloat16).to(tl.float32)
    v = v.to(tl.bfloat16).to(tl.float32)
    g = g.to(tl.bfloat16).to(tl.float32)
    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]
    rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, 1) / 64.0 + 1e-5)[:, None]
    qe = qe * rq
    qo = qo * rq
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        ang = pos * tl.load(INVF + i)
        c = tl.cos(ang)
        sn = tl.sin(ang)
        qe, qo = qe * c - qo * sn, qe * sn + qo * c
        ke, ko = ke * c - ko * sn, ke * sn + ko * c
    qb = QOUT + t * 1024 + hq * 64 + 2 * i
    tl.store(qb, (qe * qk_factor).to(tl.bfloat16))
    tl.store(qb + 1, (qo * qk_factor).to(tl.bfloat16))
    q8b = Q8OUT + t * 1024 + hq * 64 + 2 * i
    tl.store(q8b, _f32_to_e4m3(qe).to(tl.uint8))
    tl.store(q8b + 1, _f32_to_e4m3(qo).to(tl.uint8))
    tl.store(GOUT + t * 1024 + gi, g)
    loc = tl.load(LOC + t).to(tl.int64)
    kinv = 1.0 / tl.load(KS)
    vinv = 1.0 / tl.load(VS)
    kb = KC + loc * 128 + hk * 64 + 2 * i
    tl.store(kb, _f32_to_e4m3(ke * kinv).to(tl.uint8))
    tl.store(kb + 1, _f32_to_e4m3(ko * kinv).to(tl.uint8))
    tl.store(VC + loc * 128 + hk * 64 + d, _f32_to_e4m3(v * vinv).to(tl.uint8))


@triton.jit
def _attn_split(Q, KC, VC, PT, SEQ, KS, ACC, ML, pt_stride, page_size, window, sm_scale,
                L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SLIDING: tl.constexpr):
    \"\"\"One program = one request, both KV heads (128-byte K/V rows), one key split.\"\"\"
    b = tl.program_id(0)
    s = tl.program_id(1)
    seq = tl.load(SEQ + b)
    m_off = tl.arange(0, BM)
    l_idx = m_off // 8
    hh = m_off % 8
    row_ok = m_off < L * 8
    qpos = seq + l_idx
    d = tl.arange(0, 64)
    qrow = (b * L + l_idx) * 16 + hh
    q0 = tl.load(Q + qrow[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    q1 = tl.load(Q + (qrow + 8)[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    total = seq + L
    if SLIDING:
        lo = tl.maximum(seq - window + 1, 0)
    else:
        lo = seq * 0
    n = total - lo
    per = tl.cdiv(tl.cdiv(n, NSPLIT), BN) * BN
    kst = lo + s * per
    ken = tl.minimum(total, kst + per)
    scale = sm_scale * 256.0 * tl.load(KS) * 1.4426950408889634
    m0 = tl.full([BM], float("-inf"), dtype=tl.float32)
    m1 = tl.full([BM], float("-inf"), dtype=tl.float32)
    l0 = tl.zeros([BM], dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    a0 = tl.zeros([BM, 64], dtype=tl.float32)
    a1 = tl.zeros([BM, 64], dtype=tl.float32)
    for kb in range(kst, ken, BN):
        offs = kb + tl.arange(0, BN)
        kv_ok = offs < ken
        page = tl.load(PT + b * pt_stride + offs // page_size, mask=kv_ok, other=0)
        row = page.to(tl.int64) * page_size + offs % page_size
        kptr = row[:, None] * 128 + d[None, :]
        ok = kv_ok[None, :] & (offs[None, :] <= qpos[:, None])
        if SLIDING:
            ok = ok & (offs[None, :] > qpos[:, None] - window)
        k8 = tl.load(KC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16 = (((k8 & 0x7F) << 7) | ((k8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        k8b = tl.load(KC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16b = (((k8b & 0x7F) << 7) | ((k8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8 = tl.load(VC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16 = (((v8 & 0x7F) << 7) | ((v8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8b = tl.load(VC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16b = (((v8b & 0x7F) << 7) | ((v8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        sc = tl.where(ok, tl.dot(q0, tl.trans(k16)) * scale, float("-inf"))
        mn = tl.maximum(m0, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m0 - mu)
        l0 = l0 * al + tl.sum(p, 1)
        a0 = a0 * al[:, None] + tl.dot(p.to(tl.float16), v16)
        m0 = mn
        sc = tl.where(ok, tl.dot(q1, tl.trans(k16b)) * scale, float("-inf"))
        mn = tl.maximum(m1, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m1 - mu)
        l1 = l1 * al + tl.sum(p, 1)
        a1 = a1 * al[:, None] + tl.dot(p.to(tl.float16), v16b)
        m1 = mn
    ob0 = ((b * 2 + 0) * NSPLIT + s) * BM + m_off
    ob1 = ((b * 2 + 1) * NSPLIT + s) * BM + m_off
    tl.store(ACC + ob0[:, None] * 64 + d[None, :], a0)
    tl.store(ACC + ob1[:, None] * 64 + d[None, :], a1)
    tl.store(ML + ob0 * 2, m0)
    tl.store(ML + ob0 * 2 + 1, l0)
    tl.store(ML + ob1 * 2, m1)
    tl.store(ML + ob1 * 2 + 1, l1)


@triton.jit
def _attn_combine(ACC, ML, GATE, VS, OUT, L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr,
                  SB: tl.constexpr, NORMGATE: tl.constexpr = True):
    \"\"\"One program per (token, q head): merge the splits, per-head RMSNorm, * sigmoid(gate).\"\"\"
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = t // L
    l = t % L
    g = h // 8
    m_row = l * 8 + h % 8
    d = tl.arange(0, 64)
    base = (b * 2 + g) * NSPLIT
    M = tl.full([SB], float("-inf"), dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        M = tl.maximum(M, tl.load(ML + ((base + sidx) * BM + m_row) * 2, mask=sm, other=float("-inf")))
    Mx = tl.max(M, 0)
    Mx = tl.where(Mx == float("-inf"), 0.0, Mx)
    den = tl.zeros([SB], dtype=tl.float32)
    num = tl.zeros([64], dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        rows = (base + sidx) * BM + m_row
        m_s = tl.load(ML + rows * 2, mask=sm, other=float("-inf"))
        l_s = tl.load(ML + rows * 2 + 1, mask=sm, other=0.0)
        w = tl.exp2(m_s - Mx)
        den += w * l_s
        a = tl.load(ACC + rows[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0)
        num += tl.sum(w[:, None] * a, 0)
    o = num / tl.sum(den, 0) * (256.0 * tl.load(VS))
    if NORMGATE:
        o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
        gt = tl.load(GATE + t * 1024 + h * 64 + d)
        o = o * tl.sigmoid(gt)
    tl.store(OUT + t * 1024 + h * 64 + d, o.to(tl.bfloat16))


@triton.jit
def _gemm_o(X, G, W, AOUT, SUMS, STREAM, A1, B1, T, N, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
            BK: tl.constexpr, NORMGATE: tl.constexpr):
    \"\"\"o projection without a K split (K = 1024), so each program owns complete rows: it writes a and
    accumulates the four per-token sums the post-attention boundary needs (a^2, (a1 s)^2, (a1 s)(b1 a), (b1 a)^2).\"\"\"
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    w_ptr = W + rn[:, None].to(tl.int64) * K + rk[None, :]
    x_ptr = X + rt[None, :] * K + rk[:, None]
    for _k in range(0, K, BK):
        w = tl.load(w_ptr)
        if NORMGATE:
            xv = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
            gv = tl.load(G + rt[None, :] * K + (rk[:, None] + _k))
            xr = tl.reshape(xv, [BK // 64, 64, BT])
            rs = tl.rsqrt(tl.sum(xr * xr, 1) / 64.0 + 1e-5)
            xv = tl.reshape(xr * rs[:, None, :], [BK, BT]) * tl.sigmoid(gv)
            x = xv.to(tl.bfloat16)
        else:
            x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(AOUT + rt[None, :] * N + rn[:, None], acc, mask=tmask)
    st = tl.load(STREAM + rt[None, :] * N + rn[:, None], mask=tmask, other=0.0)
    a1v, b1v = _gate_ab(A1, rn)
    as_ = a1v[:, None] * st
    ba = b1v[:, None] * acc
    ok = (rt[None, :] < T) + tl.zeros([BN, BT], tl.int1)
    q0 = tl.sum(tl.where(ok, acc * acc, 0.0), 0)
    q1 = tl.sum(tl.where(ok, as_ * as_, 0.0), 0)
    q2 = tl.sum(tl.where(ok, as_ * ba, 0.0), 0)
    q3 = tl.sum(tl.where(ok, ba * ba, 0.0), 0)
    m = rt < T
    tl.atomic_add(SUMS + rt * 4 + 0, q0, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 1, q1, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 2, q2, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 3, q3, mask=m)


@triton.jit
def _bnd1_apply(AOUT, SUMS, STREAM, A1, B1, G, S1OUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    \"\"\"s1 = a1 * stream + b1 * a * rstd(a), h = s1 * rstd(s1) * g, from the four sums (one block per column slice).\"\"\"
    t = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    offs = c0 + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r1 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a = tl.load(AOUT + t * N + offs)
    a1v, b1v = _gate_ab(A1, offs)
    s1 = a1v * tl.load(STREAM + t * N + offs) + b1v * a * r
    tl.store(S1OUT + t * N + offs, s1)
    tl.store(HOUT + t * N + offs, (s1 * r1 * (1.0 + tl.load(G + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_a(P, S: tl.constexpr, T, BIAS, IDS, WTS, LS, LIDS, POOL, PSUM, BC: tl.constexpr):
    \"\"\"Reduce the pooler/router partials: one block per (token, pooled slice); slice 0 also does the top-8.\"\"\"
    t = tl.program_id(0)
    j = tl.program_id(1)
    c = j * BC + tl.arange(0, BC)
    pooled = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        pooled += tl.load(P + (s * T + t) * 4608 + c)
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    tl.store(POOL + t * 4096 + c, pooled)
    tl.atomic_add(PSUM + t, tl.sum(pooled * pooled, 0))
    if j == 0:
        e = tl.arange(0, 256)
        logit = tl.zeros([256], dtype=tl.float32)
        for s in range(S):
            b = P + (s * T + t) * 4608
            logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        ids = tl.zeros([8], dtype=tl.int32)
        wv = tl.zeros([8], dtype=tl.float32)
        for k in tl.static_range(8):
            idx = tl.argmax(sel, 0)
            val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
            ids = tl.where(k8 == k, idx, ids)
            wv = tl.where(k8 == k, val, wv)
            sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _route_b(POOL, PSUM, GPRE, XOUT, BC: tl.constexpr, PERMUTED: tl.constexpr):
    \"\"\"x = rmsnorm(pooled) * (1 + pre_expert_norm), in the expert kernel's layout.\"\"\"
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.rsqrt(tl.load(PSUM + t) / 4096.0 + 1e-5)
    x = tl.load(POOL + t * 4096 + c) * r * (1.0 + tl.load(GPRE + c).to(tl.float32))
    if PERMUTED:
        ch = c // 64
        kk = c % 64
        kb = kk // 16
        rr = kk % 16
        po = ((t * 64 + ch) * 4 + (rr % 8) // 2) * 16 + kb * 4 + (rr // 8) * 2 + (rr % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _bnd2_sums(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, FOUT, SUMS, BC: tl.constexpr):
    \"\"\"f = (unpooler partials summed) * g_postffn, and the four sums for the FFN boundary.\"\"\"
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < NU
    f = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + c, mask=m, other=0.0)
    f = f * (1.0 + tl.load(GPOST + c).to(tl.float32))
    tl.store(FOUT + t * 8192 + c, f)
    st = tl.load(S1 + t * 8192 + c)
    a2v, b2v = _gate_ab(A2, c)
    as_ = a2v * st
    bf = b2v * f
    tl.atomic_add(SUMS + t * 4 + 0, tl.sum(f * f, 0))
    tl.atomic_add(SUMS + t * 4 + 1, tl.sum(as_ * as_, 0))
    tl.atomic_add(SUMS + t * 4 + 2, tl.sum(as_ * bf, 0))
    tl.atomic_add(SUMS + t * 4 + 3, tl.sum(bf * bf, 0))


@triton.jit
def _bnd2_apply(FOUT, SUMS, S1, A2, B2, GNEXT, SOUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r2 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a2v, b2v = _gate_ab(A2, c)
    s2 = a2v * tl.load(S1 + t * N + c) + b2v * tl.load(FOUT + t * N + c) * r
    tl.store(SOUT + t * N + c, s2)
    tl.store(HOUT + t * N + c, (s2 * r2 * (1.0 + tl.load(GNEXT + c).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _gate_ab(G, offs):
    \"\"\"alpha, beta of a residual gate, from the checkpoint's BF16 weight (as SGLang's fused module does).\"\"\"
    w = tl.load(G + offs).to(tl.float32) / 0.3
    beta = tl.sigmoid(w)
    alpha = tl.sqrt(tl.maximum(tl.sigmoid(-w) * (1.0 + beta), 1e-3))
    return alpha, beta


@triton.jit
def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, ROUTER_OUT, WRITE_ROUTER: tl.constexpr = False, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    a1v, b1v = _gate_ab(A1, offs)              
    stv = tl.load(STREAM + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()   
    a = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        a += tl.load(P + (s * T + t) * 8192 + offs)
    a = a.to(tl.bfloat16).to(tl.float32)
    n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
    s1 = a1v * stv + b1v * n.to(tl.bfloat16).to(tl.float32)
    tl.store(S1OUT + t * 8192 + offs, s1)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))
    if WRITE_ROUTER:
        tl.store(ROUTER_OUT + t * 8192 + offs, hh)


@triton.jit
def _route(P, S: tl.constexpr, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    logit = tl.zeros([256], dtype=tl.float32)
    pooled = tl.zeros([4096], dtype=tl.float32)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()   
    for s in range(S):
        b = P + (s * T + t) * 4608
        logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        pooled += tl.load(b + c)
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    \"\"\"Same as _route but the pooler output (bf16) and the router logits (fp32) arrive as separate tensors.\"\"\"
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()   
    logit = tl.load(LOGITS + t * 256 + e)
    pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _pexp_norm_sum(x_ptr, w_ptr, s_ptr, out_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr, eps,
                   BLOCK_H: tl.constexpr):
    \"\"\"post-expert norm (weight first) * routing weight, summed over slots. Same kernel as SGLang's
    fused_post_expert_norm_weighted_sum (Apache 2.0), called directly: its wrapper falls back to a chain of
    ~8 torch ops below 32 tokens, which costs 39-43 us on the judge instead of ~6.\"\"\"
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)   
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        if tl.load(ids_ptr + tok * TOPK + k) >= 0:
            x = tl.load(x_ptr + (tok * TOPK + k) * H + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * w
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _boundary2(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, GNEXT, SOUT, HOUT,
               USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    m = offs < NU
    gp = 1.0 + tl.load(GPOST + offs).to(tl.float32)      
    gn = 1.0 + tl.load(GNEXT + offs).to(tl.float32)      
    a2v, b2v = _gate_ab(A2, offs)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   
        tl.extra.cuda.gdc_wait()   
    s1v = tl.load(S1 + t * 8192 + offs)
    f = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + offs, mask=m, other=0.0)
    f = (f.to(tl.bfloat16).to(tl.float32) * gp).to(tl.bfloat16).to(tl.float32)
    n = f * tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    s2 = a2v * s1v + b2v * n.to(tl.bfloat16).to(tl.float32)
    tl.store(SOUT + t * 8192 + offs, s2)
    hh = s2 * tl.rsqrt(tl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))



def _marlin_bank(weights):
    \"\"\"The local experts in Marlin's layout (SGLang's own loader step), as baseline.py builds them.\"\"\"
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous().clone(),
        w2_weight=weights["w2"].contiguous().clone(),
        w13_weight_scale=weights["w13_sf"].contiguous().clone(),
        w2_weight_scale=weights["w2_sf"].contiguous().clone(),
        w13_weight_scale_2=weights["w13_gs"].contiguous().clone(),
        w2_weight_scale_2=weights["w2_gs"].contiguous().clone(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _marlin_experts(x, topk_weights, topk_ids, state):
    \"\"\"MuseSpark1xFp4Experts._marlin_expert_outputs + post-expert-norm finalize (as baseline.py).
    moe_block_size and use_atomic_add come from the prepare()-time sweep in _tune.\"\"\"
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids  
    mc = state.get("marlin_cfg", {})
    cfgm = mc.get(num_tokens, mc.get(_bucket(num_tokens)))
    if cfgm is None:
        for block_size_m in (8, 16, 32, 48, 64):
            if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
                break
        atomic = False
    else:
        block_size_m, atomic = cfgm
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    m = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _pexp_norm_sum[(num_tokens,)](y, state["g_postexp"], topk_weights, m, local_ids, TOPK=TOP_K,
                                  H=size_k, eps=1e-8, BLOCK_H=4096, num_warps=8, num_stages=2)
    return m


def _graph_time(fn, flush, reps=9):
    \"\"\"Median graph-replay time with a cold L2. The flush is a READ sweep: zeroing a buffer leaves ~120 MB of
    dirty lines to write back, which inflates the next kernel by 15-25% (measured on the judge's H200) and not
    evenly across kernels, so it would skew every choice below.\"\"\"
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        _FLUSH_SINK.copy_(flush.sum())
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


_FLUSH_SINK = None


def _calib_bias(st, dev, g, log=False):
    \"\"\"Give the 32 local experts a skewed bias so that synthetic routing matches the judge's busiest-range load:
    ~1.6 local routes per token and ~15.7 distinct local experts at 32 tokens (reference statistics). Two
    parameters (skew width, bias size) found by a torch simulation of the real router; no input data involved.\"\"\"
    router = st["pr"][D_EXPERT:].float()  
    router = router[:N_EXPERTS] + router[N_EXPERTS:]
    gc = torch.Generator(device=dev).manual_seed(987654321)
    h = torch.randn(512, HIDDEN, generator=gc, device=dev)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
    sc = torch.sigmoid(h.double() @ router.double().T * ROUTER_MULT).float()
    idx = torch.arange(LOCAL_EXPERTS, device=dev).float()

    def stats(shape, delta):
        sel = sc + st["bias"]
        sel[:, :LOCAL_EXPERTS] += delta * shape
        ids = sel.topk(TOP_K, dim=-1).indices
        loc = ids < LOCAL_EXPERTS
        rpt = float(loc.float().sum(-1).mean())
        counts = [torch.unique(ids[i:i + 32][loc[i:i + 32]]).numel() for i in range(0, 512, 32)]
        d32 = sum(counts) / len(counts)
        return rpt, d32

    best = None
    for width in (2.0, 3.0, 4.5, 6.0, 9.0):
        shape = torch.exp(-idx / width)
        lo, hi = 0.0, 3.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if stats(shape, mid)[0] < 1.6:
                lo = mid
            else:
                hi = mid
        rpt, d32 = stats(shape, hi)
        if best is None or abs(d32 - 15.7) < abs(best[3] - 15.7):
            best = (shape, hi, rpt, d32)
    st["bias"][:LOCAL_EXPERTS] += best[1] * best[0]
    if log:
        print(f"calib delta {best[1]:.3f} routes/token {best[2]:.2f} experts@32 {best[3]:.1f}", flush=True)


def _tune(st, log=False):
    \"\"\"prepare()-time choice of the post-attention path per token count: the persistent megakernel or the
    multi-kernel pipeline, and which routed-expert kernel. Uses synthetic activations only (never input data).\"\"\"
    dev = st["qkv"].device
    global _FLUSH_SINK
    flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)  
    _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(12345)
    bias0 = st["bias"].clone()
    _calib_bias(st, dev, g, log)
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        data = {"stream": torch.randn(T, HIDDEN, generator=g, device=dev),
                "local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        if T <= 4:  
            want = max(2, int(round(2.57 * T / 1.0)) if T == 1 else 2 * T)
            for _ in range(40):
                _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                if int(((_ids >= 0) & (_ids < LOCAL_EXPERTS)).sum()) >= min(want, 8 * T):
                    break
                data["stream"] = torch.randn(T, HIDDEN, generator=g, device=dev)
                o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        best, bestt, bestv2 = None, 1e30, True
        times, dfltv2 = {}, False
        msg = []
        nloc = ndis = -1
        for name, w, K in (("qkv", st["qkv"], HIDDEN), ("o", st["o"], 1024), ("pr", st["pr"], HIDDEN),
                           ("un", st["unpool_slice"] if T <= 8 else st["unpool"], D_EXPERT)):
            xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            if st.get("bfp") is not None:
                st["mm_pick"][(T, name)] = "bfp"
                try:
                    if name == "pr":
                        t_bfp = _graph_time(lambda: (_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                     _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                            flush, reps=5)
                    else:
                        nn = w.shape[0]
                        t_bfp = _graph_time(lambda: _bfp_gemm(st, name, xd, T, nn, K), flush, reps=5)
                except Exception:  
                    t_bfp = 1e30
            else:
                t_bfp = 1e30
            st["mm_pick"][(T, name)] = False
            try:
                t_tri = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  
                t_tri = 1e30
            st["mm_pick"][(T, name)] = True
            try:
                t_mm = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  
                t_mm = 1e30
            t_bfpf = 1e30
            if name == "pr" and T <= 32 and st.get("bfp") is not None and "bfp_fused_pooler_router" in globals():
                try:
                    t_bfpf = _graph_time(lambda: bfp_fused_pooler_router(xd, st["bfp"]), flush, reps=5)
                except Exception:  
                    t_bfpf = 1e30
            opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf}
            bestp = min(opts, key=lambda p: opts[p])
            st["mm_pick"][(T, name)] = bestp
            if log:
                msg.append(f"{name}{ {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F'}[bestp] }:{opts[bestp]:.0f}")
        if st["bank"] is not None and T <= 384:
            xm = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            raw = torch.randint(0, N_EXPERTS, (T, 8), generator=g, device=dev, dtype=torch.int32)
            mids = torch.where(raw < LOCAL_EXPERTS, raw, torch.full_like(raw, -1)).contiguous()
            mwt = torch.full((T, 8), 0.125, device=dev)
            bestm, bestmt = None, 1e30
            for bs in (8, 16, 32, 48, 64):
                for atomic in (False, True):
                    st["marlin_cfg"][T] = (bs, atomic)
                    try:
                        tm = _graph_time(lambda: _marlin_experts(xm, mwt, mids, st), flush, reps=5)
                    except Exception:  
                        continue
                    if tm < bestmt:
                        bestm, bestmt = (bs, atomic), tm
            if bestm is not None:
                st["marlin_cfg"][T] = bestm
                if log:
                    msg.append(f"mcfg{bestm[0]}{'a' if bestm[1] else ''}:{bestmt:.0f}")
            else:
                st["marlin_cfg"].pop(T, None)
        cands = [c for c in ("marlin", "agent") if (c != "marlin" or st["bank"] is not None)
                 and (c != "agent" or st["ma_ok"])]
        for moe in cands:
            if moe == "marlin" and st["bank"] is None:
                continue
            if moe == "agent" and not st["ma_ok"]:
                continue
            st["moe_pick"][T] = moe
            for v2 in (False,):  
                st["gluev2"][T] = v2
                try:
                    t = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  
                    if log:
                        msg.append(f"{moe[:3]}{2 if v2 else 1}:X{repr(ex)[:40]}")
                    continue
                if log:
                    msg.append(f"{moe[:3]}{2 if v2 else 1}:{t:.0f}")
                    if nloc < 0:
                        _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                        loc = _ids[(_ids >= 0) & (_ids < LOCAL_EXPERTS)]
                        nloc, ndis = int(loc.numel()), int(torch.unique(loc).numel())
                if t < times.get(moe, 1e30):
                    times[moe] = t
                    if moe == DEFAULT_MOE:
                        dfltv2 = v2
                if t < bestt:
                    best, bestt, bestv2 = moe, t, v2
        if (DEFAULT_MOE in times and best is not None and best != DEFAULT_MOE
                and times[best] > TUNE_MARGIN * times[DEFAULT_MOE]):
            best, bestv2 = DEFAULT_MOE, dfltv2
        st["moe_pick"][T] = best if best is not None else DEFAULT_MOE
        st["gluev2"][T] = bestv2
        if log:
            print(f"T{T}/{nloc}r{ndis}e " + " ".join(msg), flush=True)
    st["bias"].copy_(bias0)
    st["pbmax"] = 0 if FORCE_PBMAX is None else FORCE_PBMAX
    if log:
        print("tune: pbmax", st["pbmax"], "moe", st["moe_pick"], "gluev2", st["gluev2"], flush=True)
    del flush


def prepare(weights):
    dev = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st = {}
    st["attention_scratch"] = {"rows": {}, "cluster": {}, "retained": []}
    st["router_fp32"] = router.contiguous()
    st["qkv"] = weights["qkv_gate"].contiguous()
    st["o"] = weights["o"].contiguous()
    st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
    st["unpool"] = weights["unpooler"].contiguous()
    st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
    st["gate1"] = weights["attn_residual_gate"].contiguous()
    st["gate2"] = weights["ffn_residual_gate"].contiguous()
    st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
    st["g_preexp"] = weights["pre_expert_norm"].contiguous()
    st["g_postexp"] = weights["post_expert_norm"].contiguous()   
    st["g_postffn"] = weights["post_ffn_norm"].contiguous()
    st["g_next"] = weights["next_pre_attn_norm"].contiguous()
    st["bias"] = weights["router_bias"].float().contiguous()
    st["gs13"] = weights["w13_gs"].float().contiguous()
    st["gs2"] = weights["w2_gs"].float().contiguous()
    st["invf0"] = torch.zeros(32, device=dev)
    st["invf"] = (1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM))).contiguous()
    errs = []

    def _build(fn):
        try:
            fn()
        except Exception as ex:  
            errs.append(ex)

    st["ma_sfx"] = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
    st["ma_sc6"] = bool(SC6_WGS and st["ma_sfx"] and _wgs_on() and _sc6_ok(weights["w13_sf"], weights["w2_sf"]))
    print("FORK_MOE_SCALE",st["ma_sfx"],st["ma_sc6"],flush=True)
    st["ma_ext"] = {}
    builders = [_att7, bfp__get_ext, _graft_glue_module, _graft_b2_module, _sl_cluster_module]
    builders.extend(_fork_build_tasks(weights, st))
    ths = [threading.Thread(target=_build, args=(f,)) for f in builders]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    _ma_prepare(weights, st)
    _bfp_prepare(weights, st)
    st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
    st["fa3"] = None
    st["bank"] = None
    st["att_ok"] = True
    st["ma_ok"] = True
    st["pbmax"] = 2
    st["moe_pick"] = {}
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        st["fa3"] = flash_attn_with_kvcache
        st["bank"] = None  # Native experts cover supported shapes; omit unused Marlin packing.
    except Exception as ex:  
        print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
    try:
        _tune(st, log=_TUNE_LOG)
    except Exception as ex:  
        print("tune failed:", repr(ex)[:200], flush=True)
        st["pbmax"] = 2
        st["moe_pick"] = {}
        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
    for k, v in PIN_PICK.items():      
        st.setdefault("mm_pick", {})[k] = v
    torch.cuda.synchronize()
    if st.get("att_ok"):
        _attn_selftest(st)     
    _check_guards()            
    return st


def _attn_cfg(B, L, sliding):
    BM = 16 if L == 1 else 64
    if sliding:
        nsplit = max(1, min(32, 264 // B))
    else:
        nsplit = max(1, min(132, 528 // B))
    return BM, nsplit


_INV_FREQ_CACHE = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    t = _INV_FREQ_CACHE.get(key)
    if t is None:
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
        else:
            t = torch.zeros(32, device=device)
        _INV_FREQ_CACHE[key] = t.float().contiguous()
    return t


def _launch7(data, dbg=None, _mod=None, pdl=None, attention_state=None, prefetch=(0,0), **override):
    assert attention_state is not None, "explicit prepared attention owner required"
    owner = attention_state["attention_scratch"]
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = data["qkv"]
    device = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    page_table = data["page_table"]
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
    plan = _plan7(B, pairs, L, window, page_table.shape[1], P, device)
    tuned = _TUNED7.get((B, L, bool(window), pairs))
    if tuned and not override:
        override = tuned
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = max(1, min(plan["ncl"], 4))
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    if flat:   
        coop = int(plan.get("coop", int(OKC_FLATCOOP and L == 8 and
                                        n_splits * pairs <= 2 * _num_sms7(device))))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                            int(n_splits * B * pairs <= 2 * _num_sms7(device))))
    w0 = int(plan.get("w0", (OKC_W0_L1 if L == 1 else OKC_W0_L8) if (flat and (coop or L == 1)) else
                      (OKC_W0_PR if (coop and not flat and B * pairs == 1) else 1024)))
    use_pdl = OKC_PDL if pdl is None else bool(pdl)
    tma = int(plan.get("tma", 0)) | (coop << 1) | (min(max(w0, 0), 4095) << 2) | (int(use_pdl) << 16)
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws_acc, ws_l, ws_cnt = _workspace7(B * pairs * nc * L * 16, device, owner)
    n_ctas = n_splits * (1 if flat else B) * pairs
    scratch, bar = _cluster_bufs7(n_ctas if cs > 1 else 1, L, device, owner)
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    (_mod or _att7()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                  data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                  ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                  dbg, cs, ncl, scratch, bar, tma, *prefetch)
    return out


def _supported7(cfg):
    P = cfg["page_size"]
    return (cfg["head_dim"] == 64 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["q_len"] in (1, 8) and cfg["batch"] <= 512 and P > 0 and (P & (P - 1)) == 0
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)

ATT_HAS_PDL = True   


def _decode_attention7(data, state, qkv):
    \"\"\"The attention agent's own launcher, lifted verbatim as _launch7: plan, workspace sizes and the coop / w0
    bits always match the kernel version embedded here (v82: my hand copy had drifted -- w0 = 0, no flat coop).
    This layer's config already carries their keys (heads, head_dim, page_size, window, rope_theta, scales).\"\"\"
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   
    bst = state.get("bfp")
    prefetch = (0,0)
    if bst is not None and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1, 8):
        _ps = bst["sk2"]["pooler"][1]
        prefetch = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))
    out = _launch7({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, attention_state=state, prefetch=prefetch, **pdl)
    return out.view(cfg["batch"] * cfg["q_len"], Q_DIM)


_GUARDS = []
_GUARDED_IDS = set()


def _guarded(t):
    nbytes = t.numel() * t.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
    full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
    v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
    _GUARDS.append((full, words))
    _GUARDED_IDS.add(id(v))
    return v


def _check_guards():
    torch.cuda.synchronize()
    for full, words in _GUARDS:
        bad = int((full[words:] != SCRATCH_SENTINEL).sum())
        if bad:
            raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


def _attn_selftest(st):
    \"\"\"Run the attention kernel once per plan family on synthetic inputs (problem constants, no data), so its
    lazily allocated buffers exist and are guarded before any timed call, and the guards see every path.\"\"\"
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(4242)
    nq = st["qkv"].shape[0]
    for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                                 (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                                 (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                                 (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):
        cfg = dict(batch=B, q_len=L, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, page_size=P, window=window,
                   rope_theta=ROPE_THETA if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
        if not _supported7(cfg):
            continue
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        per = (ctx + L + P - 1) // P
        pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
        pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
        loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
        kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
        vc = torch.zeros_like(kc)
        qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=dev)
        _decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                            "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def _att_fits7(data, cfg):
    \"\"\"The attention entry's launcher host-checks these sizes (32-bit index math, OKC_OFF32) and raises otherwise;
    mirrored here so such shapes take SGLang's FA3 instead (fuzz: batch 64 x 150k context exceeds the caches' 4 GB).\"\"\"
    kc, vc = data["k_cache"], data["v_cache"]
    rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
    return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
            and data["page_table"].numel() < (1 << 31)
            and cfg["batch"] * cfg["q_len"] * QKV_ROWS < (1 << 31)
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def _kernel_fn_fa3(data, state):
    if state.get("fa3") is None:
        raise RuntimeError("conservative scaled-verify fallback requires FA3")
    st2 = dict(state)
    st2["att_ok"] = False
    return _attention_fa3_base(data, st2)


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    agent_attn = state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg) and not (L == 8 and cfg.get("kv_scale", 1.0) != 1.0)
    if agent_attn:
        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
        else:
            Pq, Sq = dense(data["attn_input"], state["qkv"], T)
            qkv = Pq[0] if Sq == 1 else Pq.sum(0)
            qkv = qkv.to(torch.bfloat16)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    if L == 8 and cfg.get("kv_scale", 1.0) != 1.0 and state.get("fa3") is None:
        raise RuntimeError("conservative scaled-verify fallback requires FA3")
    dev = data["attn_input"].device
    x = data["attn_input"]
    sliding = cfg["window"] > 0
    use_fa3 = state["fa3"] is not None
    P, S = dense(x, state["qkv"], T)
    q = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
    q8 = torch.empty((T, Q_DIM), device=dev, dtype=torch.uint8)
    gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    _qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                   data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                   ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_fa3:
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                           v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
        window = cfg["window"]
        out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                           v_cache=data["v_cache"], page_table=data["page_table"],
                           cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                           max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                           window_size=(window - 1, 0) if window else (-1, -1),
                           scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(out, tuple):
            out = out[0]
        o_in = out.reshape(T, 1024)
    else:
        BM, NS = _attn_cfg(B, L, sliding)
        acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
        ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
        pt = data["page_table"]
        _attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                             pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                             L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
        o_in = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
        _attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                               SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
    a = o_in.to(torch.bfloat16).view(T, 16, 64).float()
    a = a * torch.rsqrt(a.square().mean(dim=-1, keepdim=True) + 1e-5)
    o_in = (a * torch.sigmoid(gate.view(T, 16, 64).float())).to(torch.bfloat16).view(T, Q_DIM)
    return _after_attention(data, state, o_in, None, T)


_attention_fa3_base = kernel_fn  

def _pick_expert_backend(pick, marlin_available, agent_available):
    \"\"\"The expert-backend fallback, factored out of _after_attention: Marlin
    without its bank falls back to the agent kernel, and vice versa. The
    three steps are kept in order (not a loop): when neither backend is
    available the original code terminates with "agent", and so does this.\"\"\"
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    if pick != "marlin" and not agent_available:
        pick = "marlin"
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    return pick


def _after_attention(data, state, o_in, gate, T):
    \"\"\"o projection onwards: residual boundary, router, experts, unpooler, final boundary.
    gate is None when o_in already carries rmsnorm(o) * sigmoid(gate). state["gluev2"] picks between the
    one-program-per-token boundary kernels and the parallel (atomic-sum) ones; the tuner chooses per T.\"\"\"
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    gv = state.get("gluev2", {})
    v2 = gv.get(T, gv.get(_bucket(T), False)) if FORCE_GLUEV2 is None else FORCE_GLUEV2
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if v2:
        a_out = torch.empty((T, HIDDEN), **f32)
        sums = torch.zeros((T, 4), **f32)
        BN, BT, BK = 64, (16 if T <= 16 else 32), 128
        _gemm_o[(HIDDEN // BN, triton.cdiv(T, BT))](
            o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, HIDDEN,
            K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
        BC1 = 2048
        _bnd1_apply[(T, HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                        s1, h, N=HIDDEN, BC=BC1, num_warps=8)
    else:
        if gate is None:
            P, S = dense(o_in, state["o"], T, state, "o")
        else:
            P, S = dense_ng(o_in, gate, state["o"], T, state)
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                         WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if T <= 8 and v2:
        _router_input_norm[(T,)](s1, state["g_preffn"], hr, num_warps=8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:  
        pooled, logits = P
        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    elif v2:
        pool = torch.empty((T, D_EXPERT), **f32)
        psum = torch.zeros((T,), **f32)
        BCR = 1024
        _route_a[(T, D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                       BC=BCR, num_warps=8)
        _route_b[(T, D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                       PERMUTED=not (use_marlin or use_ma), num_warps=8)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)   
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):      
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])     
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if v2:
        fbuf = torch.zeros((T, HIDDEN), **f32)
        sums2 = torch.zeros((T, 4), **f32)
        BC2 = 1024
        _bnd2_sums[(T, HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                       BC=BC2, num_warps=8)
        _bnd2_apply[(T, HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                        N=HIDDEN, BC=BC2, num_warps=8)
    else:
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids


import ctypes as _C
import hashlib as _hashlib
from functools import lru_cache as _lru_cache
from triton.language.extra.cuda import gdc_wait as _gdc_wait, gdc_launch_dependents as _gdc_launch
from triton.experimental import gluon as _glx
from triton.experimental.gluon import language as _gl

_GRAFT_GLUE_T = (1, 8, 16, 32, 128, 256)      
_GRAFT_LT_T = (128, 256)          
_GRAFT_ROUTE_EARLY = False   

_OKC_VISIBLE_GLUE_0 = r'''
 
 
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF *out;
  float *next, *router;
  int rows;
'''

_OKC_VISIBLE_GLUE_1 = r'''};

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
   
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
template<int V> __device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
template<int V> __device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};"
      ::"l"(p),"f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]):"memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};"
      ::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
  }
}
template<int V> __device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V/2];
  #pragma unroll
  for(int j=0;j<V/2;j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u[j]):"f"(x[2*j+1]),"f"(x[2*j]));
  }
  if constexpr(V==4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};"::"l"(p),"r"(u[0]),"r"(u[1]):"memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u[0]):"memory");
  }
}

template<int N> __device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;j++) q[j]=gf(x[2*j],x[2*j],gm(x[2*j+1],x[2*j+1]));
   
   
  #pragma unroll
  for(int width=N/2;width>1;width/=2) {
    #pragma unroll
    for(int j=0;j<width/2;j++) q[j]=ga(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
  #pragma unroll
  for(int d=first;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm,
                                                cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* p=cluster.map_shared_rank(sm,dst);
      p[rank*warps_per_cta+warp]=sum;
    }
  }
  cluster.sync();
   
   
   
  float all=sm[lane&(NW-1)];
  return butterfly(all,NW/2);
'''

_OKC_VISIBLE_GLUE_2 = r'''}

template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  constexpr int logical_threads=NW*32;
  constexpr int per_cta=logical_threads/8;
  constexpr int N=8192/logical_threads;
  constexpr int repeats=N/V;
  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
'''

_OKC_VISIBLE_GLUE_3 = r'''    glb<V>(a.pre+col,pre+r*V);
    if constexpr(POST) glb<V>(a.post+col,post+r*V);
    if constexpr(SLICED) {
      if(col>=1024) {
        #pragma unroll
        for(int j=0;j<V;j++) {
          int k=r*V+j;
           
           
          ns[k]=gf(alpha[k],s[k],gm(beta[k],gm(0.f,post[k])));
        }
'''

_OKC_VISIBLE_GLUE_4 = r'''      }
    }
  }
   
   
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if constexpr(SLICED) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=0.f;
      if(col<1024) {
        #pragma unroll
        for(int sk=0;sk<8;sk++) {
          float part[V];glf<V>(a.part+(sk*a.rows+row)*1024+col,part);
          #pragma unroll
          for(int j=0;j<V;j++) b[r*V+j]=ga(b[r*V+j],part[j]);
        }
      }
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(b[r*V+j]);
    } else {
      glb<V>(a.branch+row*8192+col,b+r*V);
    }
    if constexpr(POST) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(gm(b[r*V+j],post[r*V+j]));
    }
  }
  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if(!SLICED || col<1024) {
      #pragma unroll
      for(int j=0;j<V;j++) {
        int k=r*V+j;
        float bn=gb(gm(b[k],r1));
        if(SLICED || (PRELOADED_BASE && col<4096))
          ns[k]=gf(alpha[k],s[k],gm(beta[k],bn));
        else
          ns[k]=gf(beta[k],bn,gm(alpha[k],s[k]));
      }
'''

_OKC_VISIBLE_GLUE_5 = r'''      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
  }
}

template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=0;
  cfg.attrs=attrs;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
 
'''

_OKC_VISIBLE_GLUE_6 = r'''  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b1: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(write_router) {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,true,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,true,false>,s);
  } else {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,false,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}
'''

_GRAFT_GLUE_CUDA = (
_OKC_VISIBLE_GLUE_0
 + _OKC_VISIBLE_GLUE_1
 + _OKC_VISIBLE_GLUE_2
 + r'''  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N];
  __shared__ float partials[2][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
'''
 + _OKC_VISIBLE_GLUE_3
 + r'''        if(active) gsf<V>(a.next+row*8192+col,ns+r*V);
'''
 + _OKC_VISIBLE_GLUE_4
 + r'''      if(active) gsf<V>(a.next+row*8192+col,ns+r*V);
    }
  }
  float r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
'''
 + _OKC_VISIBLE_GLUE_5
 + r'''void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0)};
'''
 + _OKC_VISIBLE_GLUE_6
 + r''' 
void b2c(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)next.data_ptr(),(int)stream.size(0)};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b2c: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,true,false,false,true>,s);
  else glue_launch(a,glue_cluster_boundary<4,32,true,false,false,false>,s);
}
'''
)
_GRAFT_B2_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
__device__ __forceinline__ void lf2(const float* p, float* x) {
  float2 v = *reinterpret_cast<const float2*>(p); x[0]=v.x; x[1]=v.y;
}
__device__ __forceinline__ void lb2(const BF* p, float* x) {
  unsigned u = *reinterpret_cast<const unsigned*>(p);
  x[0]=__uint_as_float(u<<16); x[1]=__uint_as_float(u&0xffff0000u);
}
__device__ __forceinline__ void glf2(const float* p, float* x) {
  asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
}
__device__ __forceinline__ void gsf2(float* p, const float* x) {
  asm volatile("st.global.v2.f32 [%0],{%1,%2};"::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
}
__device__ __forceinline__ void gsb2(BF* p, const float* x) {
  unsigned u;
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u):"f"(x[1]),"f"(x[0]));
  asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u):"memory");
}
__device__ __forceinline__ float pair_sq(float x0, float x1) { return gf(x0,x0,gm(x1,x1)); }
__device__ __forceinline__ float butterfly(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}

 
 
template<int P,bool SBF>
__global__ __launch_bounds__(P) void b2_single(const float* part, const float* stream, const BF* post,
    const BF* pre, const float* alpha, const float* beta, BF* out, float* next, int rows) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  constexpr int J=1024/P, JL=512/P, WP=P/32;
  const int t=threadIdx.x, lane=t&31, w=t>>5, row=blockIdx.x;
  __shared__ float sm1[32], sm2[32];
  const float* srow=stream+(size_t)row*8192;
  float* nrow=next+(size_t)row*8192;
  float ns[J][8], pr[J][8];
  float s0[JL][2], al[JL][2], be[JL][2], po[JL][2];
  float q1[JL], q23[JL];
   
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      const int col=2*L+2048*r;
      lb2(pre+col,pr[j]+2*r);
      if(j<JL && r==0) {
        lf2(srow+col,s0[j]); lf2(alpha+col,al[j]); lf2(beta+col,be[j]); lb2(post+col,po[j]);
      } else {
        float s[2],a[2],b[2],p[2];
        lf2(srow+col,s); lf2(alpha+col,a); lf2(beta+col,b); lb2(post+col,p);
        #pragma unroll
        for(int k=0;k<2;k++) ns[j][2*r+k]=gf(a[k],s[k],gm(b[k],gm(0.f,p[k])));
      }
    }
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) if(!(j<JL && r==0)) gsf2(nrow+2*L+2048*r,ns[j]+2*r);
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    if(j<JL) {
      q1[j]=pair_sq(ns[j][2],ns[j][3]);
      q23[j]=ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7]));
    } else {
       
      float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),pair_sq(ns[j][2],ns[j][3])),
                 ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7])));
      v=butterfly(v);
      if(lane==0) { sm2[w+WP*j]=v; sm1[w+WP*j]=0.f; }
    }
  }
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
   
  float b[JL][2];
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    if constexpr(SBF) {
       
      unsigned u;
      asm volatile("ld.global.b32 %0,[%1];":"=r"(u):"l"(reinterpret_cast<const BF*>(part)+(size_t)row*1024+col):"memory");
      b[j][0]=gb(gm(__uint_as_float(u<<16),po[j][0]));
      b[j][1]=gb(gm(__uint_as_float(u&0xffff0000u),po[j][1]));
    } else {
    float pp[8][2];
    #pragma unroll
    for(int sk=0;sk<8;sk++) glf2(part+((size_t)(sk*rows+row))*1024+col,pp[sk]);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float x=0.f;
      #pragma unroll
      for(int sk=0;sk<8;sk++) x=ga(x,pp[sk][k]);
      x=gb(x);
      b[j][k]=gb(gm(x,po[j][k]));
    }
    }
     
    const float z=pair_sq(0.f,0.f);
    float v=ga(ga(pair_sq(b[j][0],b[j][1]),z),ga(z,z));
    v=butterfly(v);
    if(lane==0) sm1[w+WP*j]=v;
  }
  __syncthreads();
  const float r1=gr(butterfly(sm1[lane]),1.e-8f);
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float bn=gb(gm(b[j][k],r1));
      ns[j][k]=gf(al[j][k],s0[j][k],gm(be[j][k],bn));
    }
    gsf2(nrow+col,ns[j]);
    float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),q1[j]),q23[j]);
    v=butterfly(v);
    if(lane==0) sm2[w+WP*j]=v;
  }
  __syncthreads();
  const float r2=gr(butterfly(sm2[lane]),1.e-5f);
  BF* orow=out+(size_t)row*8192;
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      float h[2];
      #pragma unroll
      for(int k=0;k<2;k++) h[k]=gm(gm(ns[j][2*r+k],r2),pr[j][2*r+k]);
      gsb2(orow+2*L+2048*r,h);
    }
  }
}


 
void b2s(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  int rows=(int)stream.size(0);
  if(stream.size(1)!=8192 || branch.size(0)!=rows || branch.size(1)!=1024) throw std::runtime_error("graft b2s: shape");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(rows);cfg.blockDim=dim3(512);cfg.stream=s;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed=0;
  cfg.attrs=attrs;cfg.numAttrs=1;
  auto e=cudaLaunchKernelEx(&cfg,b2_single<512,true>,(const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
      (const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),(const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
      (BF*)out.data_ptr(),(float*)next.data_ptr(),rows);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''


@_lru_cache(None)
def _graft_glue_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='graft_glue_' + _hashlib.md5(_GRAFT_GLUE_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_GRAFT_GLUE_CUDA, functions=['b1', 'b2c'],
                            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@_lru_cache(None)
def _graft_b2_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='graft_b2_' + _hashlib.md5(_GRAFT_B2_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_GRAFT_B2_CUDA, functions=['b2s'],
                            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@_glx.jit
def _graft_row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                        H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    _gdc_wait()
    _gdc_launch()
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.bfloat16).to(_gl.float32)
    if POST:
        w = _gl.load(PostW + c).to(_gl.float32)
        b = (b * w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    s = _gl.load(Stream + row * H + c)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    w = _gl.load(PreW + c).to(_gl.float32)
    h = ns * r * w
    _gl.store(Out + row * H + c, h)


@_glx.jit
def _graft_row_boundary_pre(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                            H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.bfloat16).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    h = ns * r * pw
    _gl.store(Out + row * H + c, h)


@triton.jit
def _graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, PACKED: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if EARLY:
        _gdc_launch()
    _gdc_wait()
    if not EARLY:
        _gdc_launch()
    if kind == 0:
        c = tl.arange(0, 4096)
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c).to(tl.bfloat16).to(tl.float32)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        if PACKED != 0:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if PACKED == 1:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


_GP, _GI, _GSZ = _C.c_void_p, _C.c_int, _C.c_size_t


class _GraftAlgo(_C.Structure):
    _fields_ = [("data", _C.c_uint64 * 8)]


class _GraftHeur(_C.Structure):
    _fields_ = [("algo", _GraftAlgo), ("workspaceSize", _C.c_size_t), ("state", _C.c_int),
                ("wavesCount", _C.c_float), ("reserved", _C.c_int * 4)]


class _GraftLt:
    def __init__(self, transpose_a):
        self.lib = _C.CDLL("libcublasLt.so.13")
        sigs = {
            "cublasLtCreate": [_C.POINTER(_GP)],
            "cublasLtMatmulDescCreate": [_C.POINTER(_GP), _GI, _GI],
            "cublasLtMatmulDescSetAttribute": [_GP, _GI, _GP, _GSZ],
            "cublasLtMatrixLayoutSetAttribute": [_GP, _GI, _GP, _GSZ],
            "cublasLtMatrixLayoutCreate": [_C.POINTER(_GP), _GI, _C.c_uint64, _C.c_uint64, _C.c_int64],
            "cublasLtMatmul": [_GP] * 12 + [_C.POINTER(_GraftAlgo), _GP, _GSZ, _GP],
            "cublasLtMatmulAlgoInit": [_GP, _GI, _GI, _GI, _GI, _GI, _GI, _GI, _C.POINTER(_GraftAlgo)],
            "cublasLtMatmulAlgoConfigSetAttribute": [_C.POINTER(_GraftAlgo), _GI, _GP, _GSZ],
            "cublasLtMatmulAlgoConfigGetAttribute": [_C.POINTER(_GraftAlgo), _GI, _GP, _GSZ, _C.POINTER(_GSZ)],
            "cublasLtMatmulAlgoCheck": [_GP, _GP, _GP, _GP, _GP, _GP, _C.POINTER(_GraftAlgo), _C.POINTER(_GraftHeur)],
        }
        for name, types_ in sigs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types_, _GI
        self.handle = _GP()
        self.call("cublasLtCreate", _C.byref(self.handle))
        self.desc = _GP()
        self.call("cublasLtMatmulDescCreate", _C.byref(self.desc), 68, 0)
        op = _GI(1 if transpose_a else 0)
        self.call("cublasLtMatmulDescSetAttribute", self.desc, 3, _C.byref(op), _C.sizeof(op))
        self.workspace = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
        self.alpha, self.beta = _C.c_float(1), _C.c_float(0)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} status {code}")

    def __deepcopy__(self, memo):
        import copy
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.lib, result.handle, result.desc = self.lib, self.handle, self.desc
        result.alpha, result.beta = self.alpha, self.beta
        result.workspace = copy.deepcopy(self.workspace, memo)
        return result


class _GraftPlan(dict):
    def __deepcopy__(self, memo):
        import copy
        result = type(self)()
        memo[id(self)] = result
        for key, value in self.items():
            result[key] = value if key == "layouts" else copy.deepcopy(value, memo)
        return result



def _graft_layouts(lt, m, n, k, mode, batch, fp32):
    wa = (k, n // batch, k, k * n // batch) if mode == "tn" else (n // batch, k, n, n // batch)
    result = []
    for index, (rows, cols, ld, stride) in enumerate((wa, (k, m, k, 0), (n // batch, m, n, n // batch))):
        desc = _GP()
        lt.call("cublasLtMatrixLayoutCreate", _C.byref(desc), 0 if fp32 and index == 2 else 14, rows, cols, ld)
        count, offset = _GI(batch), _C.c_int64(stride)
        lt.call("cublasLtMatrixLayoutSetAttribute", desc, 5, _C.byref(count), _C.sizeof(count))
        lt.call("cublasLtMatrixLayoutSetAttribute", desc, 6, _C.byref(offset), _C.sizeof(offset))
        result.append(desc)
    return result


_GRAFT_LT_PICKS = {
    ("pr", 256): ('tn', 2, True, [785979015234, 4294967331, 0, 562949953421312, 558779540176897, 60129542158, 292057776128, 0]),
    ("pr", 128): ('tn', 1, True, [575525617730, 4294967331, 0, 1970324836974598, 558779540176897, 60129542158, 292057776128, 0]),
    ("o", 256): ('nn', 2, False, [85899345986, 4294967331, 0, 1407374883553286, 558779540176897, 60129542158, 292058693646, 0]),
    ("un", 256): ('nn', 2, False, [85899345986, 4294967331, 0, 1407374883553287, 558779540176897, 60129542158, 292058693646, 0]),
    ("un", 128): ('nn', 2, False, [77309411394, 4294967331, 0, 1407374883553283, 558779540176897, 60129542158, 292058693646, 0]),
    ("qkv", 128): ('tn', 16, False, [395136991298, 4294967331, 0, 1407374883553282, 558779540176897, 60129542158, 292058693646, 0]),
    ("o", 128): ('nn', 8, False, [77309411394, 4294967331, 0, 1407374883553283, 558779540176897, 60129542158, 292058693646, 0]),
}
_GRAFT_QKV256 = {"id": 66, "tile": 124, "stages": 35, "splitk": 1, "red": 0, "swz": 0, "custom": 0, "cluster": 2}
_GRAFT_QKV256_ATTRS = (("tile", 1, _C.c_uint32), ("splitk", 2, _C.c_int32), ("red", 3, _C.c_uint32),
                       ("swz", 4, _C.c_uint32), ("custom", 5, _C.c_uint32), ("stages", 6, _C.c_uint32),
                       ("cluster", 8, _C.c_uint16))



def _fork_lt_no_split(lt, algo):
    value, written = _GI(), _GSZ()
    lt.call("cublasLtMatmulAlgoConfigGetAttribute", _C.byref(algo), 2,
            _C.byref(value), _C.sizeof(value), _C.byref(written))
    return written.value == _C.sizeof(value) and value.value in (0, 1)

def _graft_lt_plan(st):
    lt_tn, lt_nn = _GraftLt(True), _GraftLt(False)
    plans, status = {}, {}
    src = {"pr": st["pr"], "o": st["o"], "un": st["unpool"], "qkv": st["qkv"]}
    for (name, m), (mode, batch, fp32, words) in _GRAFT_LT_PICKS.items():
        if m not in _GRAFT_LT_T:
            continue
        lt = lt_tn if mode == "tn" else lt_nn
        if mode == "tn":
            weight = src[name]
            n, k = weight.shape
        else:
            key = "graft_t_" + name
            if key not in st:
                st[key] = src[name].t().contiguous()
            weight = st[key]
            k, n = weight.shape
        assert weight.is_contiguous() and weight.data_ptr() % 128 == 0, (name, weight.data_ptr() % 128)
        assert lt.workspace.data_ptr() % 256 == 0 and n % batch == 0 and (n // batch * 2) % 16 == 0
        layouts = _graft_layouts(lt, m, n, k, mode, batch, fp32)
        algo = _GraftAlgo((_C.c_uint64 * 8)(*words))
        heur = _GraftHeur()
        code = lt.lib.cublasLtMatmulAlgoCheck(lt.handle, lt.desc, *layouts, layouts[2], _C.byref(algo), _C.byref(heur))
        status[(name, m)] = (code, heur.workspaceSize)
        if code or heur.workspaceSize > lt.workspace.numel():
            continue
        if not _fork_lt_no_split(lt, algo):
            status[(name,m)] = ("rejected K split", 0)
            continue
        plans[(name, m)] = _GraftPlan(lt=lt, layouts=layouts, algo=algo, weight=weight, n=n, fp32=fp32)
    if 256 in _GRAFT_LT_T:
        algo = _GraftAlgo()
        lt_tn.call("cublasLtMatmulAlgoInit", lt_tn.handle, 68, 0, 14, 14, 14, 14, _GRAFT_QKV256["id"], _C.byref(algo))
        for name, attr, ctype in _GRAFT_QKV256_ATTRS:
            value = ctype(_GRAFT_QKV256[name])
            lt_tn.call("cublasLtMatmulAlgoConfigSetAttribute", _C.byref(algo), attr, _C.byref(value), _C.sizeof(value))
        weight = st["qkv"]
        n, k = weight.shape
        assert weight.is_contiguous() and weight.data_ptr() % 128 == 0
        layouts = _graft_layouts(lt_tn, 256, n, k, "tn", 1, False)
        heur = _GraftHeur()
        code = lt_tn.lib.cublasLtMatmulAlgoCheck(lt_tn.handle, lt_tn.desc, *layouts, layouts[2], _C.byref(algo),
                                                 _C.byref(heur))
        status[("qkv", 256)] = (code, heur.workspaceSize)
        if not code and heur.workspaceSize <= lt_tn.workspace.numel() and _fork_lt_no_split(lt_tn, algo):
            plans[("qkv", 256)] = _GraftPlan(lt=lt_tn, layouts=layouts, algo=algo, weight=weight, n=n, fp32=False)
    st["graft_lt"] = plans
    st["graft_lt_status"] = {f"{k[0]}:{k[1]}": v for k, v in status.items()}


def _graft_lt_run(x, plan):
    x = x.contiguous()
    assert x.data_ptr() % 16 == 0
    out = torch.empty((x.shape[0], plan["n"]), device=x.device,
                      dtype=torch.float32 if plan["fp32"] else torch.bfloat16)
    lt = plan["lt"]
    a, b, c = plan["layouts"]
    lt.call("cublasLtMatmul", lt.handle, lt.desc, _C.byref(lt.alpha), _GP(plan["weight"].data_ptr()), a,
            _GP(x.data_ptr()), b, _C.byref(lt.beta), _GP(out.data_ptr()), c, _GP(out.data_ptr()), c,
            _C.byref(plan["algo"]), _GP(lt.workspace.data_ptr()), lt.workspace.numel(),
            _GP(torch.cuda.current_stream().cuda_stream))
    return out


_vj_dense = dense


def dense(x, w, T, state=None, name=None):
    plans = state.get("graft_lt") if state is not None else None
    plan = plans.get((name, T)) if plans and T > 32 else None
    if plan is not None:
        rq = _rq_gemm(state, name, x, T)
        if rq is not None:
            return rq.view(1, T, rq.shape[1]), 1
        out = _graft_lt_run(x, plan)
        return out.view(1, T, plan["n"]), 1
    return _vj_dense(x, w, T, state, name)


def _graft_al(*ts):
    for t in ts:
        assert t.is_contiguous() and t.data_ptr() % 16 == 0, (t.shape, t.data_ptr() % 16)


def _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):
    \"\"\"Returns True when a grafted kernel handled the boundary, else False (caller runs the base kernel).\"\"\"
    g = st["graft"]
    if S != 1:
        return False
    br = P.view(T, -1)
    width = br.shape[1]
    pfx = "2" if post else "1"
    if T <= 16 and br.dtype == torch.bfloat16:
        _graft_al(br, stream, s_out, h_out)
        if not post and width == HIDDEN:
            _graft_glue_module().b1(br, stream, g["pre1"], g["alpha1"], g["beta1"], h_out, s_out,
                                    router_out if router_out is not None else s_out, router_out is not None)
            return True
        if post and width == 1024 and nu == 1024:
            _graft_b2_module().b2s(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        if post and width == HIDDEN and nu == HIDDEN:
            _graft_glue_module().b2c(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        return False
    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                      g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,
                                      num_warps=32, launch_pdl=False)
        return True
    if T > 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                  num_warps=16 if T > 192 else 32, launch_pdl=False)
        return True
    return False


def _graft_after_attention(data, state, o_in, gate, T):
    \"\"\"Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments.\"\"\"
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if not _graft_boundary(P, S, T, data["stream"], state, False, s1, h, router_out=hr):
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                         WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        if T in (16, 32):
            nw, parts = {16: (8, 1), 32: (8, 1)}[T]
            _sl_tile_route[(T, parts + 1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=True, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=False)
        else:
            _graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=False)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        if T in (16, 32):
            nw, parts = {16: (8, 1), 32: (8, 1)}[T]
            _sl_tile_route[(T, parts + 1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=False, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=False)
        else:
            _graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=False)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if not _graft_boundary(P, S, T, s1, state, True, s2, out, nu=nu):
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids


_vj_after_attention = _after_attention


def _after_attention(data, state, o_in, gate, T):
    if "graft" in state and T in _GRAFT_GLUE_T and state.get("compose_graft_enabled", {}).get(T, T in (1, 8, 128, 256)):
        return _graft_after_attention(data, state, o_in, gate, T)
    return _vj_after_attention(data, state, o_in, gate, T)


_vj_kernel_fn = kernel_fn


def kernel_fn(data, state):
    cfg = data["config"]
    T = cfg["tokens"]
    plan = state.get("graft_lt", {}).get(("qkv", T)) if T in (128, 256) else None
    if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
            and not (cfg["q_len"] == 8 and cfg.get("kv_scale", 1.0) != 1.0)
            and state.get("mm_pick", {}).get((T, "qkv"), True) is True):
        qkv = _rq_gemm(state, "qkv", data["attn_input"], T)
        if qkv is None:
            qkv = _graft_lt_run(data["attn_input"], plan)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    return _vj_kernel_fn(data, state)


_vj_prepare = prepare


def prepare(weights):
    st = _vj_prepare(weights)   
    if _GRAFT_GLUE_T:
        if st.get("bfp") is not None:
            for T in _GRAFT_GLUE_T:
                if T <= 16:
                    st["mm_pick"][(T, "o")] = "bfp"
        g = {}
        for i, name in ((1, "attn_residual_gate"), (2, "ffn_residual_gate")):
            gw = weights[name].float() / 0.3
            beta = torch.sigmoid(gw)
            g["alpha%d" % i] = torch.sqrt(torch.clamp(torch.sigmoid(-gw) * (1. + beta), min=1.e-3)).contiguous()
            g["beta%d" % i] = beta.contiguous()
        g["pre1"] = (weights["pre_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        g["post2"] = (weights["post_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        g["pre2"] = (weights["next_pre_attn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        for v in g.values():
            assert v.data_ptr() % 128 == 0, v.data_ptr() % 128
        st["graft"] = g
        _graft_glue_module()
        _graft_b2_module()
    if _GRAFT_LT_T:
        _graft_lt_plan(st)
    torch.cuda.synchronize()
    return st



_MA_ALT_SRCS['moe_helper0'] = (
r'''
 
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1   
#define OKC_F1EARLY 1   
#define OKC_WGSFINSS 1   
'''
 + _OKC_VISIBLE_MOE_0
 + r'''#define TIDX ((int)((threadIdx.x + NT) % NTL))
'''
 + _OKC_VISIBLE_MOE_1
 + _OKC_VISIBLE_MOE_2
 + _OKC_VISIBLE_MOE_3
 + r'''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
'''
 + _OKC_VISIBLE_MOE_4
 + r'''  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
'''
 + _OKC_VISIBLE_MOE_5
 + r'''constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
'''
 + _OKC_VISIBLE_MOE_6
 + r'''#endif
'''
 + _OKC_VISIBLE_MOE_7
 + r'''  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);
'''
 + _OKC_VISIBLE_MOE_8
 + r'''  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;   
'''
 + _OKC_VISIBLE_MOE_9
 + r'''    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
'''
 + _OKC_VISIBLE_MOE_10
 + r'''  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;   
'''
 + _OKC_VISIBLE_MOE_11
 + r'''    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
'''
 + _OKC_VISIBLE_MOE_12
 + r'''  if (TIDX == 0) {
'''
 + _OKC_VISIBLE_MOE_13
 + r'''  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
'''
 + _OKC_VISIBLE_MOE_14
 + r'''    if (TIDX == 0) S.preissued = 0;
'''
 + _OKC_VISIBLE_MOE_15
 + r'''  if (TIDX == NT - 32) {
'''
 + _OKC_VISIBLE_MOE_16
 + r'''  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;   
'''
 + _OKC_VISIBLE_MOE_17
 + r'''    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
'''
 + _OKC_VISIBLE_MOE_18
 + r'''  const int lane = TIDX & 31;
'''
 + _OKC_VISIBLE_MOE_19
 + r'''constexpr int WGS_PIECE = 9216;                  
'''
 + _OKC_VISIBLE_MOE_20
 + _OKC_VISIBLE_MOE_21
 + r'''#define OKC_WGSFCI 3
'''
 + _OKC_VISIBLE_MOE_22
 + r'''  if ((TIDX & 31) != 0) return;
'''
 + _OKC_VISIBLE_MOE_23
 + r'''      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
'''
 + _OKC_VISIBLE_MOE_24
 + r'''  if ((TIDX & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
  WS_JOIN();   
  if ((TIDX & 31) != 0) return;
'''
 + _OKC_VISIBLE_MOE_25
 + r'''        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }   
      }
       
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }
'''
 + _OKC_VISIBLE_MOE_26
 + r'''  const int tid = TIDX, nj = S.njF + S.njS;
'''
 + _OKC_VISIBLE_MOE_27
 + r'''          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
  if (tid == 0) {   
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
 
 
 
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;
'''
 + _OKC_VISIBLE_MOE_28
 + r'''        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), fork_bf16(d4[h] * gsc) * w0);
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
  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
     
     
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
'''
 + _OKC_VISIBLE_MOE_29
 + r'''  if (!PB) {
    asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
 
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = TIDX, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;   
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];   
   
   
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  int prev = -1;   
'''
 + _OKC_VISIBLE_MOE_30
 + r'''    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      const uint2 scg = scn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else if (lane == 0) {
'''
 + _OKC_VISIBLE_MOE_31
 + r'''          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
'''
 + _OKC_VISIBLE_MOE_32
 + r'''        if (ci == 0) {
          if (prev >= 0) {   
            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
         
         
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
'''
 + _OKC_VISIBLE_MOE_33
 + _OKC_VISIBLE_MOE_34
 + r'''__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
 
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;   
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
'''
 + _OKC_VISIBLE_MOE_35
 + r'''__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
 
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
'''
 + _OKC_VISIBLE_MOE_36
 + r'''  const int tid = TIDX;
'''
 + _OKC_VISIBLE_MOE_37
 + r'''    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
     
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);   
'''
 + _OKC_VISIBLE_MOE_38
 + r'''  {   
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
'''
 + _OKC_VISIBLE_MOE_39
 + _OKC_VISIBLE_MOE_40
 + r'''   
#if OKC_WGS && OKC_WGSFINSS
   
   
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
'''
 + _OKC_VISIBLE_MOE_41
 + r'''          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
'''
 + _OKC_VISIBLE_MOE_42
 + _OKC_VISIBLE_MOE_43
 + r'''  p.ybuf = p.ss + okc::EMAX;
'''
 + _OKC_VISIBLE_MOE_44
 + r'''  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
'''
 + _OKC_VISIBLE_MOE_45
)
MOE_MID_DEFINES += ("@moe_helper0",)



_MA_ALT_SRCS['moe_helper0df'] = _MA_ALT_SRCS['moe_helper0']  
MOE_MID_DEFINES = tuple("@moe_helper0df" if d == "@moe_helper0" else d for d in MOE_MID_DEFINES)


def _compose_times(fns, reps=16):
    graphs, outputs = [], []
    for fn in fns:
        fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out=fn()
        graphs.append(g);outputs.append(out)
    flush=torch.ones((65536,1024),device='cuda',dtype=torch.float32)
    sink=torch.empty(1024,device='cuda',dtype=torch.float32)
    events=[]
    for r in range(reps):
        order=range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))
        for i in order:
            torch.sum(flush,dim=0,out=sink)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[i].replay();b.record()
            events.append((i,a,b))
    torch.cuda.synchronize()
    samples=[[] for _ in graphs]
    for i,a,b in events:samples[i].append(a.elapsed_time(b)*1000)
    return [sorted(s)[len(s)//2] for s in samples]



_stack_unpinned_prepare = prepare

def prepare(weights):
    st = _stack_unpinned_prepare(weights)
    bfp = st.get("bfp")
    if bfp is not None:
        for T in (1, 8, 16, 32):
            for name, layer in _BFP_LAYER.items():
                bfp["choice"][(layer, T)] = bfp__static_choice(layer, T, bfp)
            for name in ("qkv", "o", "un"):
                st["mm_pick"][(T, name)] = "bfp"
            can_fuse = (bfp["choice"][("pooler", T)][0] == "sk2"
                        and bfp["choice"][("router", T)][0] == "sk2")
            st["mm_pick"][(T, "pr")] = "bfpf" if T <= 32 and can_fuse else "bfp"
    for T in (128, 256):
        for name in ("qkv", "o", "pr", "un"):
            st["mm_pick"][(T, name)] = True
    return st

_policy_parent_prepare=prepare


def _policy_variant(st,T,mode):
    ss=dict(st)
    ss['mm_pick']=dict(st['mm_pick'])
    ss['graft_lt']=dict(st.get('graft_lt',{}))
    ss['compose_graft_enabled']=dict(st.get('compose_graft_enabled',{}))
    ss['compose_boundary_warps']=dict(st.get('compose_boundary_warps',{}))
    graft,bw,drops=mode
    ss['compose_graft_enabled'][T]=graft
    ss['compose_boundary_warps'][T]=bw
    for name in drops:ss['graft_lt'].pop((name,T),None)
    return ss


def prepare(weights):
    st=_policy_parent_prepare(weights)
    st['compose_base_mm']=dict(st['mm_pick'])
    st['compose_base_graft_lt']=dict(st['graft_lt'])
    st['compose_active']=set()
    st['compose_selection']=[]
    st['compose_graft_enabled']={T:T in (1,8,128,256) for T in (1,8,16,32,128,256)}
    st['compose_boundary_warps']={}
    gen=torch.Generator(device=st['qkv'].device).manual_seed(1618033)
    for T in (1,8,16,32,128,256):
        original=st['compose_graft_enabled'][T]
        modes=[(original,16,()),(True,16,()),(False,16,()),(False,8,()),(False,4,())]
        modes=list(dict.fromkeys(modes))
        data={'stream':torch.randn(T,HIDDEN,device=st['qkv'].device,generator=gen),
              'local_start':torch.zeros(1,device=st['qkv'].device,dtype=torch.int32)}
        o_in=(.1*torch.randn(T,1024,device=st['qkv'].device,generator=gen)).to(torch.bfloat16)
        ids=_after_attention(data,st,o_in,None,T)[2]
        counts=torch.bincount(ids.flatten().long()//32,minlength=8)
        data['local_start'].fill_(32*int(counts.argmax()))
        variants=[_policy_variant(st,T,mode) for mode in modes]
        ts=_compose_times([lambda ss=ss:_after_attention(data,ss,o_in,None,T) for ss in variants],reps=32)
        best=min(range(len(ts)),key=ts.__getitem__)
        if ts[best]>=ts[0]-.4:best=0
        picked=variants[best]
        st['compose_graft_enabled'][T]=picked['compose_graft_enabled'][T]
        st['compose_boundary_warps'][T]=picked['compose_boundary_warps'][T]
        st['graft_lt']=picked['graft_lt']
        st['compose_selection'].append({'T':T,'region':'chain_policy','us':ts,'modes':modes,'choice':modes[best]})
    return st


_SL_GLUE_CUDA = (
_OKC_VISIBLE_GLUE_0
 + r'''  const float* alpha2;
  float* tail;
'''
 + _OKC_VISIBLE_GLUE_1
 + r'''}

template<int NW,int N>
__device__ __forceinline__ float cluster_squares_pair(const float (&x)[N], const float (&y)[N],
    float* sm, float* smy, cg::cluster_group cluster, int rank, float& tail) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sx=butterfly(local_squares(x),16);
  float sy=butterfly(local_squares(y),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* px=cluster.map_shared_rank(sm,dst);
      volatile float* py=cluster.map_shared_rank(smy,dst);
      px[rank*warps_per_cta+warp]=sx;
      py[rank*warps_per_cta+warp]=sy;
    }
  }
  cluster.sync();
  tail=butterfly(smy[lane&(NW-1)],NW/2);
  return butterfly(sm[lane&(NW-1)],NW/2);
'''
 + _OKC_VISIBLE_GLUE_2
 + r'''  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];
  __shared__ float partials[3][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
    glf<V>(a.alpha2+col,a2+r*V);
'''
 + _OKC_VISIBLE_GLUE_3
 + r'''         
'''
 + _OKC_VISIBLE_GLUE_4
 + r'''       
    }
  }
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    #pragma unroll
    for(int j=0;j<V;j++) {
      int k=r*V+j;
      carry[k]=gm(a2[k],ns[k]);
      masked[k]=(col+j>=1024) ? carry[k] : 0.f;
    }
  }
  float tail;
  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);
  if(rank==0 && tid==0) a.tail[row]=tail;
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      gsf<V>(a.next+row*8192+col,carry+r*V);
'''
 + _OKC_VISIBLE_GLUE_5
 + r'''void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0),(const float*)alpha2.data_ptr(),(float*)tail.data_ptr()};
'''
 + _OKC_VISIBLE_GLUE_6
)

@_lru_cache(None)
def _sl_cluster_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='sl_early_tail_' + _hashlib.md5(_SL_GLUE_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_SL_GLUE_CUDA, functions=['b1'],
                            extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])

@triton.jit
def _sl_cluster_finish(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,
                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0,1024)
    beta = tl.load(BETA+c)
    gp = tl.load(POST+c).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    carry = tl.load(CARRY+t*8192+c)
    tail = tl.load(TAIL+t)
    o = tile*BC+tl.arange(0,BC)
    outcarry = tl.load(CARRY+t*8192+o)
    gn = tl.load(NEXT+o).to(tl.float32)
    f = (tl.load(P+t*1024+c).to(tl.float32)*gp).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(f*f,0)/8192.0+1e-8)
    n = (f*inv).to(tl.bfloat16).to(tl.float32)
    head = carry+beta*n
    invnext = tl.rsqrt((tail+tl.sum(head*head,0))/8192.0+1e-5)
    result = tl.where(o<1024,tl.gather(head,o%1024,0),outcarry)
    tl.store(SOUT+t*8192+o,result)
    tl.store(HOUT+t*8192+o,(result*invnext*gn).to(tl.bfloat16))

SL_CLUSTER_CONFIG={1: (8, 256), 8: (16, 1024)}
def _sl_cluster_after_attention(data, state, o_in, gate, T):
    \"\"\"Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments.\"\"\"
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _graft_after_attention(data,state,o_in,gate,T)
    g=state['graft']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        _sl_graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=False)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        _sl_graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=False)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=False)
    return out, s2, ids


_sl_previous_after_attention=_after_attention

def _after_attention(data,state,o_in,gate,T):
    if T in SL_CLUSTER_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
        return _sl_cluster_after_attention(data,state,o_in,gate,T)
    return _sl_previous_after_attention(data,state,o_in,gate,T)

_sl_previous_prepare=prepare

def prepare(weights):
    st=_sl_previous_prepare(weights)
    _sl_cluster_module()
    return st
@triton.jit
def _sl_graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if EARLY:
        _gdc_launch()
    _gdc_wait()
    if not EARLY:
        _gdc_launch()
    if kind == 0:
        c = tl.arange(0, 4096)
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c).to(tl.bfloat16).to(tl.float32)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

SL_ROUTE_CONFIG={1: (8, 0, 0), 8: (4, 0, 0)}
@triton.jit
def _sl_tile_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if kind<PARTS:
        c=tl.arange(0,4096)
        oc=kind*(4096//PARTS)+tl.arange(0,4096//PARTS)
        if PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        _gdc_launch()
        _gdc_wait()
        if SPLIT:
            pooled=tl.load(POOLED+t*4096+c).to(tl.float32)
        else:
            pooled=tl.load(ALL+t*4608+c).to(tl.bfloat16).to(tl.float32)
        inv=tl.rsqrt(tl.sum(pooled*pooled,0)/4096.0+1e-5)
        if not PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        if PARTS==1:
            vals=pooled
        elif SPLIT:
            vals=tl.load(POOLED+t*4096+oc).to(tl.float32)
        else:
            vals=tl.load(ALL+t*4608+oc).to(tl.bfloat16).to(tl.float32)
        tl.store(XOUT+t*4096+oc,(vals*inv*gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        _gdc_launch()
        _gdc_wait()
        if not PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        k8 = tl.arange(0, 8)
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

SL_TILE_CONFIG={1:(4,1,True),8:(4,4,True)}
def _sl_tile_after_attention(data, state, o_in, gate, T):
    \"\"\"Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments.\"\"\"
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _graft_after_attention(data,state,o_in,gate,T)
    g=state['graft']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=False)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=False)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=False)
    return out, s2, ids


_sl_tile_prior_after=_after_attention
def _after_attention(data,state,o_in,gate,T):
    if T in SL_TILE_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
        return _sl_tile_after_attention(data,state,o_in,gate,T)
    return _sl_tile_prior_after(data,state,o_in,gate,T)

MOE_TINY2_DEFINES += ("@moe_helper0",)
MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1")


_DE_EXT=None
_DE_RUN=bfp__run_sk2
_DE_PREPARE=prepare

def _de_source():
 cfg=('qkv8n_r144','qkv_gate',2304,24,3,1,4,2,2,1)
 _,nstage=bfp__sk2_geom(cfg,8)
 dispatch=f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;'
 source=bfp_SK2_SRC.replace('@@SK2_DISPATCH@@',dispatch).replace('@@SK2F_DISPATCH@@','')
 source=source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)','4').replace('NW + (COMP ? NESC : 0)','NW')
 source=source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");','')
 a=source.index('      int pe = 1;');b=source.index('\\n    }\\n    mbar_arrive(cbar);',a)
 source=source[:a]+r\"\"\"
      for(int pe=1+ew;pe<=cnt;pe+=NESC) {
        const int2 e=ptab_s[pe];const int kk=e.x&65535;
        if(lane<M && kk>=gs0*KST && kk<(gs0+nst)*KST)
          cr[lane*CP+(e.x>>16)] += __int_as_float(e.y)*__bfloat162float(x[(size_t)lane*K+kk]);
      }
\"\"\"+source[b:]
 shared=bfp_CUDA_SRC.replace('@@MBAR@@',bfp_MBAR_HELPERS).replace('@@BODY@@','')
 return shared+bfp_MAP_SRC+source

def _de_ext():
 global _DE_EXT
 if _DE_EXT is None:
  import hashlib
  source=_de_source();tag=hashlib.md5(source.encode()).hexdigest()[:12];source=source.replace('sk2_kernel','sk2_kernel_narrow_'+tag).replace('launch_sk2','launch_sk2_narrow_'+tag);cpp=bfp_CPP_SRC[bfp_CPP_SRC.index('void sk2('):bfp_CPP_SRC.index('void sk2f(')]+'void sk2_set_pdl(int64_t n);'
  _DE_EXT=load_inline('proj_dl_escape_'+hashlib.md5(source.encode()).hexdigest()[:12],'#include <torch/extension.h>\\n'+cpp,
    cuda_sources=source,functions=['sk2','sk2_set_pdl'],no_implicit_headers=True,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
 return _DE_EXT

def prepare(weights):
 _de_ext()
 return _DE_PREPARE(weights)

def bfp__run_sk2(x,layer,m,n,k,state,dg=0,dbg=None):
 if layer!='qkv_gate' or not 2<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)
 name=bfp__sk2_name(layer,m);ci,stream,tab,pmax,b4,n_rows=state['sk2'][name]
 out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 ext=_de_ext();ext.sk2_set_pdl(bfp_SK2_PDL)
 ext.sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,bfp__sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)
 return out

import threading as _GlueContextThreading
_GLUE_CONTEXT_TLS = _GlueContextThreading.local()
_GLUE_CONTEXT_API = None

def _glue_context_api():
    global _GLUE_CONTEXT_API
    if _GLUE_CONTEXT_API is None:
        api = _C.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_C.POINTER(_C.c_void_p)]),
            ("cuCtxGetDevice", [_C.POINTER(_C.c_int)]),
            ("cuDeviceGet", [_C.POINTER(_C.c_int), _C.c_int]),
            ("cuDevicePrimaryCtxRetain", [_C.POINTER(_C.c_void_p), _C.c_int]),
            ("cuCtxSetCurrent", [_C.c_void_p]),
        ):
            function = getattr(api, name)
            function.argtypes, function.restype = args, _C.c_int
        _GLUE_CONTEXT_API = api
    return _GLUE_CONTEXT_API

def _glue_context_check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")

def _glue_bind_context(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_GLUE_CONTEXT_TLS, "device", None) == index:
        return
    api = _glue_context_api()
    context = _C.c_void_p()
    _glue_context_check(api.cuCtxGetCurrent(_C.byref(context)), "cuCtxGetCurrent")
    current_device = _C.c_int(-1)
    if context.value:
        _glue_context_check(api.cuCtxGetDevice(_C.byref(current_device)), "cuCtxGetDevice")
    if not context.value or current_device.value != index:
        device, primary = _C.c_int(), _C.c_void_p()
        _glue_context_check(api.cuDeviceGet(_C.byref(device), index), "cuDeviceGet")
        _glue_context_check(api.cuDevicePrimaryCtxRetain(_C.byref(primary), device), "cuDevicePrimaryCtxRetain")
        _glue_context_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _GLUE_CONTEXT_TLS.device = index

_glue_context_parent_kernel = kernel_fn
def kernel_fn(data, state):
    _glue_bind_context(data["attn_input"])
    return _glue_context_parent_kernel(data, state)



@triton.jit
def _router_input_norm(S1, G, OUT):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    s1 = tl.load(S1 + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(OUT + t * 8192 + offs, hh)


def _pool_router_fp32(h, hr, T, state, ppick):
    if T <= 8:
        if state.get("bfp") is not None and ppick in ("bfp", "bfpf"):
            if ppick == "bfpf":
                _bfp_pdl(T, "pool")
            pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        else:
            pooled = F.linear(h, state["pr"][:D_EXPERT])
        logits = _router_fp32_gemv(hr, state)
        return (pooled, logits), -1
    if ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:
        _bfp_pdl(T, "pool")
        return bfp_fused_pooler_router(h, state["bfp"]), -1
    if ppick == "bfp" and state.get("bfp") is not None:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)
        return (pooled, logits), -1
    return dense(h, state["pr"], T, state, "pr")


_tune_with_bias_restore = _tune


def _tune(st, log=False):
    bias0 = st["bias"].clone()
    try:
        return _tune_with_bias_restore(st, log=log)
    finally:
        st["bias"].copy_(bias0)


_moe_global_cache = _moe


class _RouterExt:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __deepcopy__(self, memo):
        return self


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
    ext = state["ma_ext"].get(key)
    if ext is None:
        ext = _RouterExt(_ma_ext(tuple(defines) + sfx))
        if MOE_HAS_PDL:
            ext.okc_set_pdl(0)
        state["ma_ext"][key] = ext
    return ext


_router_fp32_parent_prepare = prepare


def prepare(weights):
    st = _router_fp32_parent_prepare(weights)
    for T in (1, 8, 16, 32, 128, 256):
        _moe(T, st)
    return st



_ROUTER_FP32_CUDA = r'''
#include <cuda_runtime.h>
#include <stdint.h>

__global__ __launch_bounds__(256) void router_fp32_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int m) {
  const int expert = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = w + (size_t)expert * 8192;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int k = tid * 4; k < 8192; k += 1024) {
    const float4 b = *(const float4*)(row + k);
#pragma unroll
    for (int t = 0; t < 8; ++t) if (t < m) {
      const float4 a = *(const float4*)(x + (size_t)t * 8192 + k);
      acc[t] = fmaf(a.x, b.x, acc[t]);
      acc[t] = fmaf(a.y, b.y, acc[t]);
      acc[t] = fmaf(a.z, b.z, acc[t]);
      acc[t] = fmaf(a.w, b.w, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int d = 16; d; d >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], d);
  }
  __shared__ float partial[8][8];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < 8; ++t) partial[warp][t] = acc[t];
  }
  __syncthreads();
  if (warp < m) {
    float v = lane < 8 ? partial[lane][warp] : 0.f;
#pragma unroll
    for (int d = 4; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (lane == 0) out[(size_t)warp * 256 + expert] = v;
  }
}

int router_fp32_launch(int64_t x, int64_t w, int64_t out, int m, int64_t stream) {
  if (m < 1 || m > 8) return -1;
  router_fp32_kernel<<<256, 256, 0, (cudaStream_t)stream>>>(
      (const float*)x, (const float*)w, (float*)out, m);
  return (int)cudaGetLastError();
}
'''
_ROUTER_FP32_EXT = None


def _router_fp32_ext():
    global _ROUTER_FP32_EXT
    if _ROUTER_FP32_EXT is None:
        import hashlib
        digest = hashlib.md5(_ROUTER_FP32_CUDA.encode()).hexdigest()[:12]
        cpp = '#include <pybind11/pybind11.h>\\n#include <stdint.h>\\nint router_fp32_launch(int64_t,int64_t,int64_t,int,int64_t);'
        _ROUTER_FP32_EXT = load_inline(
            name='router_fp32_' + digest,
            cpp_sources=cpp,
            cuda_sources=_ROUTER_FP32_CUDA,
            functions=['router_fp32_launch'],
            no_implicit_headers=True,
            with_pytorch_error_handling=False,
            verbose=False,
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'])
    return _ROUTER_FP32_EXT


def _router_fp32_gemv(x, state):
    m = x.shape[0]
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    rc = state['router_fp32_ext'].router_fp32_launch(
        x.data_ptr(), state['router_fp32'].data_ptr(), out.data_ptr(), m,
        torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f'FP32 router launch failed: {rc}')
    return out


_router_fp32_native_parent_prepare = prepare


def prepare(weights):
    state = _router_fp32_native_parent_prepare(weights)
    state['router_fp32_ext'] = _RouterExt(_router_fp32_ext())
    return state


_kfix_router_fp32_gemv = _router_fp32_gemv
def _router_fp32_gemv(x, state):
    if state.get('router_fp32_ext') is None:
        state['router_fp32_ext'] = _RouterExt(_router_fp32_ext())
    return _kfix_router_fp32_gemv(x, state)


GLE_FRT = True      
GLE_ILV = 1         
_GLE_EXT = None


def _gle_source():
    cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == bfp_SK2_PICK[("o", 8)])
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    assert (layer, R, NWR * WSK, CL) == ("o", 32, 4, 1), cfg
    _, nstage = bfp__sk2_geom(cfg, 8)
    s = bfp_SK2_SRC
    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\\n"
                  "struct GleEpi { const float *s, *a, *b, *a2; const __nv_bfloat16* p; __nv_bfloat16* uv; float* part; float* uvf; const float *x32, *w32; int fpr, ilv; };\\n"
                  "static thread_local GleEpi g_gle;", 1)
    s = s.replace("           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",
                  "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {", 1)
    pre = r\"\"\"  __shared__ __align__(16) float4 gx_s[64], gx_a[64], gx_b[64], gx_a2[64];
  __shared__ __align__(16) uint2 gx_p[64];
  // the epilogue's input-side vectors (residual stream row, alpha / beta / alpha2 / pre) go to shared memory by
  // cp.async before the wait and are awaited only in the epilogue: no registers held across the main loop (2 CTAs / SM
  // next to attention) and no stall behind the pre-wait weight stages
  // ep.s is data['stream'], the layer's input residual stream: nothing in the call writes it, which is what makes this
  // pre-wait copy safe (keep it that way: any in-call writer of data['stream'] must move this copy after the wait)
  if (!FUSE && tid < 8 * M) {   // o only (this source also holds the fused pooler)
    const size_t col = (size_t)cta * R + (tid & 7) * 4;
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&gx_s[tid]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d0), "l"(ep.s + (size_t)(tid >> 3) * N + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a[tid])), "l"(ep.a + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_b[tid])), "l"(ep.b + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a2[tid])), "l"(ep.a2 + col) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_p[tid])), "l"(ep.p + col) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
\"\"\"
    anchor = '  __syncthreads();\\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread'
    assert s.count(anchor) == 1
    s = s.replace(anchor, pre + anchor, 1)
    epi = r\"\"\"  constexpr int R4 = R / 4;
  if constexpr (!FUSE) {   // o: the gle epilogue; the fused pooler keeps the stock epilogue loop below
  static_assert(R4 == 8 && CL == 1 && sizeof(OutT) == 2, "gle epilogue: o8h geometry, BF16 out");
    const int tt = tid >> 3, row = (tid & 7) * 4;
    const bool on = tid < R4 * M;
    float q[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (on) {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (COMP) {
#pragma unroll
        for (int c = 0; c < NESC; ++c) {
          v.x += corr[c * CSZ + tt * CP + row];
          v.y += corr[c * CSZ + tt * CP + row + 1];
          v.z += corr[c * CSZ + tt * CP + row + 2];
          v.w += corr[c * CSZ + tt * CP + row + 3];
        }
      }
#pragma unroll
      for (int w = 0; w < WSK; ++w) {
        const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
      }
      const size_t col = (size_t)cta * R + row;
      OutT* o = out + (size_t)tt * N + col;
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
      const float pf[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
      asm volatile("cp.async.wait_all;" ::: "memory");   // this thread's own copies
      const uint2 xp = gx_p[tid];
      const float4 xs = gx_s[tid], xa = gx_a[tid], xb = gx_b[tid], xa2 = gx_a2[tid];
      const float4 xas = make_float4(__fmul_rn(xa.x, xs.x), __fmul_rn(xa.y, xs.y), __fmul_rn(xa.z, xs.z), __fmul_rn(xa.w, xs.w));
      // pre = 1 + pre_ffn_norm in FP32 (the reference's (1 + w), not a BF16-rounded copy)
      const float pp[4] = {__fadd_rn(1.f, __uint_as_float(xp.x << 16)), __fadd_rn(1.f, __uint_as_float(xp.x & 0xffff0000u)),
                           __fadd_rn(1.f, __uint_as_float(xp.y << 16)), __fadd_rn(1.f, __uint_as_float(xp.y & 0xffff0000u))};
      const float ex_as[4] = {xas.x, xas.y, xas.z, xas.w}, ex_b[4] = {xb.x, xb.y, xb.z, xb.w};
      const float ex_a2[4] = {xa2.x, xa2.y, xa2.z, xa2.w};
      float uu[4], vv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float as = ex_as[j], bp = __fmul_rn(ex_b[j], pf[j]);
        uu[j] = __fmul_rn(as, pp[j]);
        vv[j] = __fmul_rn(bp, pp[j]);
        q[0] = __fmaf_rn(pf[j], pf[j], q[0]);
        q[1] = __fmaf_rn(as, as, q[1]);
        q[2] = __fmaf_rn(as, bp, q[2]);
        q[3] = __fmaf_rn(bp, bp, q[3]);
        if (col >= 1024) {
          const float x = __fmul_rn(ex_a2[j], as), y = __fmul_rn(ex_a2[j], bp);
          q[4] = __fmaf_rn(x, x, q[4]);
          q[5] = __fmaf_rn(x, y, q[5]);
          q[6] = __fmaf_rn(y, y, q[6]);
        }
      }
      // u, v: FP32 rows for the router (uvf) and BF16 rows for the pooler (uv); rows tt (u) and M + tt (v)
      auto put = [&](const float* f, size_t r) {
        __nv_bfloat162 h0 = __floats2bfloat162_rn(f[0], f[1]), h1 = __floats2bfloat162_rn(f[2], f[3]);
        uint2 hk;
        hk.x = *reinterpret_cast<uint32_t*>(&h0); hk.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(ep.uv + r * N + col) = hk;
        *reinterpret_cast<float4*>(ep.uvf + r * N + col) = make_float4(f[0], f[1], f[2], f[3]);
      };
      put(uu, (size_t)tt);
      put(vv, (size_t)(M + tt));
    }
    // the 8 lanes of one token (one aligned lane group) sum their columns: one partial row per (CTA, token)
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 4);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 2);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 1);
    }
    if (on && (tid & 7) == 0) {
      float4* pq = reinterpret_cast<float4*>(ep.part + ((size_t)cta * M + tt) * 8);
      pq[0] = make_float4(q[0], q[1], q[2], q[3]);
      pq[1] = make_float4(q[4], q[5], q[6], 0.f);
    }
  }
  else
  for (int idx = tid; idx < R4 * M; idx += NC) {
\"\"\"
    anchor = '  constexpr int R4 = R / 4;\\n  for (int idx = tid; idx < R4 * M; idx += NC) {'
    assert s.count(anchor) == 1
    s = s.replace(anchor, epi, 1)
    fpr_stage = (
        "        } else if (ep.fpr) {   // FP32 router stage j = st - nst: columns rq * 2048 + 512 j .. + 511\\n"
        "          const int kb = rq * 2048 + (st - nst) * 512;\\n"
        "          mbar_expect_tx(full0 + 8 * slot, 8 * 2048 + M * 2048);\\n"
        "          for (int r = 0; r < 8; ++r)\\n"
        "            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);\\n"
        "          for (int t = 0; t < M; ++t)\\n"
        "            bulk_g2s(s0 + 16384 + t * 2048, reinterpret_cast<const uint8_t*>(ep.x32 + (size_t)t * 8192 + kb), 2048, full0 + 8 * slot);\\n"
        "        } else {   // router piece: global router stage rq * 4 + j of tile rt")
    fpr_cons = (
        "  if constexpr (FUSE) {\\n    if (ep.fpr) {\\n"
        "      static_assert(!FUSE || (NW == 8 && SST >= 32768), \\"fp32 router stages: 8 consumer warps, 32 KB slots\\");\\n"
        "      // thread: column group cg = tid & 127 (4 of the stage's 512 columns) x all 8 router rows x tokens 4h .. 4h + 3\\n"
        "      // (h = tid >> 7): every shared-memory float4 is read by 2 threads, not 8 warps; FP32 FMAs\\n"
        "      const int cg = tid & 127, th = tid >> 7;\\n"
        "      float af[32];\\n"
        "#pragma unroll\\n"
        "      for (int i = 0; i < 32; ++i) af[i] = 0.f;\\n"
        "      for (int st = nst; st < nstt; ++st) {\\n"
        "        const int slot = st % NSTAGE;\\n"
        "        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\\n"
        "        const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + cg * 4;\\n"
        "        const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384) + cg * 4;\\n"
        "        float4 xa[4];\\n"
        "#pragma unroll\\n"
        "        for (int u = 0; u < 4; ++u)\\n"
        "          xa[u] = (th * 4 + u < M) ? *reinterpret_cast<const float4*>(xs + (th * 4 + u) * 512) : make_float4(0.f, 0.f, 0.f, 0.f);\\n"
        "#pragma unroll\\n"
        "        for (int r = 0; r < 8; ++r) {\\n"
        "          const float4 bw = *reinterpret_cast<const float4*>(ws + r * 512);\\n"
        "#pragma unroll\\n"
        "          for (int u = 0; u < 4; ++u) {\\n"
        "            float v = af[r * 4 + u];\\n"
        "            v = fmaf(xa[u].x, bw.x, v); v = fmaf(xa[u].y, bw.y, v); v = fmaf(xa[u].z, bw.z, v); v = fmaf(xa[u].w, bw.w, v);\\n"
        "            af[r * 4 + u] = v;\\n"
        "          }\\n"
        "        }\\n"
        "        __syncwarp();\\n"
        "        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\\n"
        "      }\\n"
        "      // reduction over the 128 column groups through the (now idle) ring: red[th][cg][33] (static indices only)\\n"
        "      if constexpr (COMP) mbar_wait(cbar, 0);   // as the BF16 router: the escape warps are done with the ring\\n"
        "      named_bar(1, NC);                         // and every consumer is done reading its last stage\\n"
        "      float* red32 = reinterpret_cast<float*>(gbase);\\n"
        "#pragma unroll\\n"
        "      for (int i = 0; i < 32; ++i) red32[(th * 128 + cg) * 33 + i] = af[i];\\n"
        "      named_bar(1, NC);\\n"
        "      float* red2 = red32 + 256 * 33;\\n"
        "      {\\n"
        "        const int o = tid & 63, pq = tid >> 6, h = o >> 5, e = o & 31;\\n"
        "        float v = 0.f;\\n"
        "        for (int c = pq * 32; c < pq * 32 + 32; ++c) v += red32[(h * 128 + c) * 33 + e];\\n"
        "        red2[pq * 64 + o] = v;\\n"
        "      }\\n"
        "      named_bar(1, NC);\\n"
        "      if (tid < 64) {\\n"
        "        const int h = tid >> 5, e = tid & 31, r = e >> 2, t = h * 4 + (e & 3);\\n"
        "        const float v = red2[tid] + red2[64 + tid] + red2[128 + tid] + red2[192 + tid];\\n"
        "        if (t < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + t * 8 + r, v);\\n"
        "      }\\n"
        "      named_bar(1, NC);\\n"
        "    } else {\\n    // router piece:")
    ilv_edits = (
        ("  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\\n",
         "  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\\n"
         "  const bool ilv = FUSE && ep.fpr && ep.ilv && (nstt % 4) == 0 && nstt / 4 > P;\\n"
         "  const int PER = ilv ? nstt / 4 : (1 << 30);\\n"
         "  const int OFF = ilv ? (ep.ilv == 2 ? P : PER - 1) : 0;   // router stage slot inside each period\\n"),
        ("        if (!FUSE || st < nst) {\\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\\n",
         "        const bool isr = ilv ? (st % PER == OFF) : (FUSE && st >= nst);\\n"
         "        const int ps = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\\n"
         "        if (!isr) {\\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + ps) * wstride, WBY, full0 + 8 * slot);\\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + ps) * UPS);\\n"),
        ("          const int kb = rq * 2048 + (st - nst) * 512;\\n",
         "          const int kb = rq * 2048 + (ilv ? st / PER : st - nst) * 512;\\n"),
        ("        if (FUSE && st >= nst) {   // router stages:",
         "        if (ilv ? (st % PER == OFF) : (FUSE && st >= nst)) {   // router stages:"),
        ("        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;\\n",
         "        const int psq = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\\n"
         "        const int kbeg = (gs0 + psq) * KST, kend = kbeg + KST;\\n"),
        ("        if ((st % NESC) != ew) {", "        if ((psq % NESC) != ew) {"),
        ("  for (int st = 0; st < nst; ++st) {\\n    const int slot = st % NSTAGE;\\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\\n",
         "  float afi[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};   // ilv: FP32 router accumulators (warp = router row)\\n"
         "  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\\n    const int slot = st % NSTAGE;\\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\\n"
         "    if constexpr (FUSE) if (ilv && st % PER == OFF) {   // an interleaved FP32 router stage\\n"
         "      const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + warp * 512;\\n"
         "      const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384);\\n"
         "#pragma unroll\\n"
         "      for (int i = 0; i < 4; ++i) {\\n"
         "        const int k = lane * 4 + i * 128;\\n"
         "        const float4 bw = *reinterpret_cast<const float4*>(ws + k);\\n"
         "#pragma unroll\\n"
         "        for (int t = 0; t < 8; ++t) if (t < M) {\\n"
         "          const float4 a = *reinterpret_cast<const float4*>(xs + t * 512 + k);\\n"
         "          afi[t] = fmaf(a.x, bw.x, afi[t]); afi[t] = fmaf(a.y, bw.y, afi[t]);\\n"
         "          afi[t] = fmaf(a.z, bw.z, afi[t]); afi[t] = fmaf(a.w, bw.w, afi[t]);\\n"
         "        }\\n"
         "      }\\n"
         "      __syncwarp();\\n"
         "      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\\n"
         "      continue;\\n"
         "    }\\n"),
        ("  if constexpr (FUSE) {\\n    if (ep.fpr) {\\n",
         "  if constexpr (FUSE) {\\n    if (ilv) {   // interleaved FP32 router: warp butterflies of afi, then the 4-quarter reduction below\\n"
         "#pragma unroll\\n"
         "      for (int t = 0; t < 8; ++t)\\n"
         "#pragma unroll\\n"
         "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);\\n"
         "      float mine = afi[0];\\n"
         "#pragma unroll\\n"
         "      for (int t = 1; t < 8; ++t) mine = lane == t ? afi[t] : mine;\\n"
         "      if (lane < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + lane * 8 + warp, mine);\\n"
         "      if constexpr (COMP) mbar_wait(cbar, 0);\\n"
         "      named_bar(1, NC);\\n"
         "    } else if (ep.fpr) {\\n"))
    for old, new in (
            ("        } else {   // router piece: global router stage rq * 4 + j of tile rt", fpr_stage),
            ("  if constexpr (FUSE) {\\n    // router piece:", fpr_cons),
            ("    named_bar(1, NC);\\n    __shared__ int s_rlast;",
             "    named_bar(1, NC);\\n    }\\n    __shared__ int s_rlast;")):
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new, 1)
    for old, new in ilv_edits:
        assert s.count(old) == 1, old[:70]
        s = s.replace(old, new, 1)
    s = s.replace("wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_gle));", 1)
    s = s.replace("wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_gle);", 1)
    host = (f"void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,\\n"
            f"         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {{")
    assert s.count(host) == 1
    disp = (f"if (tokr == 8 && N == 8192 && K == 1024 && M <= 8) {{ launch_sk2<1, 8, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
            f"{NACC}, {CL}, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, 0, nullptr); return; }}")
    pcfg = next(c for c in bfp_SK2_CONFIGS if c[0] == "pooler")
    flines = []
    for tokr in (8,):   
        _, pns = bfp__sk2_geom(pcfg, tokr)
        _, _, _, pR, pNT, pNWR, pWSK, pUPW, pNACC, pCL = pcfg
        flines.append(f"if (tokr == {tokr}) {{ launch_sk2<1, {tokr}, {pNT}, {pNWR}, {pWSK}, {pUPW}, {pns}, true, false, "
                      f"{pNACC}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                      f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                      f"rcnt.data_ptr<int>()); return; }}")
    s = s.replace("@@SK2_DISPATCH@@", disp).replace("@@SK2F_DISPATCH@@", "\\n  ".join(flines))
    s += r\"\"\"
void gle_o(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
           at::Tensor s, at::Tensor a, at::Tensor b, at::Tensor a2, at::Tensor p, at::Tensor uv, at::Tensor part, at::Tensor uvf) {
  TORCH_CHECK(M >= 1 && M <= 8 && s.size(0) == M && uv.size(0) == 2 * M && uvf.size(0) == 2 * M && part.numel() >= 256 * M * 8, "gle_o: shapes");
  g_gle = GleEpi{s.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), a2.data_ptr<float>(),
                 reinterpret_cast<const __nv_bfloat16*>(p.data_ptr()), reinterpret_cast<__nv_bfloat16*>(uv.data_ptr()),
                 part.data_ptr<float>(), uvf.data_ptr<float>()};
  sk2(Ws, x, out, ptab, pmax, base4, M, 8192, 1024, 0, 8, 0, ptab);
}
void gle_pr(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
            int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, at::Tensor x32, at::Tensor w32, int64_t wpf) {
  // fused pooler (BF16 x rows) + FP32 router (FP32 x32 rows, FP32 weights w32) at M <= 8
  TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x32.size(0) == M && x32.scalar_type() == at::kFloat &&
              w32.scalar_type() == at::kFloat && w32.size(0) == 256 && w32.size(1) == 8192, "gle_pr: shapes");
  g_gle = GleEpi{};
  g_gle.x32 = x32.data_ptr<float>(); g_gle.w32 = w32.data_ptr<float>(); g_gle.fpr = 1; g_gle.ilv = (int)wpf;
  sk2f(Ws, Wsr, x, out, outr, ptab, pmax, base4, part, rcnt, M, 8);
  g_gle = GleEpi{};
}
\"\"\"
    shared = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", "")
    return shared + bfp_MAP_SRC + s


def _gle_ext():
    global _GLE_EXT
    if _GLE_EXT is None:
        source = _gle_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace("sk2_kernel", "sk2_kernel_gle" + tag).replace("launch_sk2", "launch_sk2_gle" + tag)
                  .replace("void sk2(", "void sk2_gle" + tag + "(").replace("  sk2(Ws,", "  sk2_gle" + tag + "(Ws,")
                  .replace("void sk2f(", "void sk2f_gle" + tag + "(").replace("  sk2f(Ws,", "  sk2f_gle" + tag + "(Ws,"))
        cpp = ("void gle_o(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,"
               " int64_t M, torch::Tensor s, torch::Tensor a, torch::Tensor b, torch::Tensor a2, torch::Tensor p,"
               " torch::Tensor uv, torch::Tensor part, torch::Tensor uvf);\\n"
               "void gle_pr(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr,"
               " torch::Tensor ptab, int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M,"
               " torch::Tensor x32, torch::Tensor w32, int64_t wpf);\\nvoid sk2_set_pdl(int64_t n);\\n")
        _GLE_EXT = load_inline("gle_o_" + tag, "#include <torch/extension.h>\\n" + cpp, cuda_sources=source,
                               functions=["gle_o", "gle_pr", "sk2_set_pdl"], no_implicit_headers=True,
                               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
    return _GLE_EXT






def _gle_frt(xb, x32, state):
    \"\"\"Fused BF16 pooler + FP32 router for M = xb.shape[0] <= 8 rows: (pooled bf16 [M, 4096], logits fp32 [M, 256]).\"\"\"
    bst = state["bfp"]
    m = xb.shape[0]
    dev = xb.device
    if "fused_part" not in bst:
        bst["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        bst["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
    pci, pstream, ptab_, ppmax, pb4, _ = bst["sk2"]["pooler"]
    pooled = torch.empty((m, 4096), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((m, 256), device=dev, dtype=torch.float32)
    _bfp_pdl(m, "pool")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_pr(pstream, bst["sk2"]["router"][1], xb, pooled, logits, ptab_, ppmax, pb4, bst["fused_part"], bst["fused_cnt"],
               m, x32, state["router_fp32"], int(GLE_ILV))
    return pooled, logits


_gle_prior_pool_router_fp32 = _pool_router_fp32
def _pool_router_fp32(h, hr, T, state, ppick):
    if (GLE_FRT and T <= 8 and hr is not None and ppick in ("bfp", "bfpf") and state.get("bfp") is not None
            and state.get("router_fp32") is not None and hr.is_contiguous() and h.is_contiguous()):
        return _gle_frt(h, hr, state), -1
    return _gle_prior_pool_router_fp32(h, hr, T, state, ppick)






_gle_prior_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    return _gle_prior_after(data, state, o_in, gate, T)


_gle_prior_prepare = prepare
def prepare(weights):
    return _gle_prior_prepare(weights)


_pin_parent_prepare = prepare
def prepare(weights):
    st = _pin_parent_prepare(weights)
    cge = st.setdefault('compose_graft_enabled', {})
    for T in (1, 8):
        cge[T] = True
    return st


def _rq_gemm(state, name, x, T):
    return None



class _ForkMarlinAdapter:
    def __init__(self,state): self.state=state
    def okc_set_pdl(self,on): pass  
    def okc_run(self,w,s,x,ids,tw,out,lo,nloc,grid):
        if lo!=0 or nloc!=32 or not 1<=x.shape[0]<=384:
            raise RuntimeError('FORK_MOE_FALLBACK local-ID/chunk ABI')
        from sgl_kernel.scalar_type import scalar_types
        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
        bank=self.state['_fork_marlin_bank']; m,h=x.shape
        tw=tw.float().contiguous(); ids=ids.contiguous()
        for bm in (8,16,32,48,64):
            if m*8/32/bm<.9: break
        sorted_ids,experts,padded=moe_align_block_size(ids,bm,256)
        c2=torch.empty((m*8,h),device=x.device,dtype=x.dtype)
        c13=torch.empty((m*8*2*h,),device=x.device,dtype=x.dtype)
        c1=c13.view(-1,2*h); c3=c13[:m*8*h].view(-1,h)
        common=dict(moe_block_size=bm,mul_topk_weights=False,is_ep=True,
            b_q_type=scalar_types.float4_e2m1f,is_k_full=True,use_atomic_add=False,
            use_fp32_reduce=True,is_zp_float=False)
        c1=moe_wna16_marlin_gemm(x,c1,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,
            bank['workspace'],sorted_ids,experts,padded,tw,top_k=8,size_m=m,size_n=2*h,size_k=h,**common)
        silu_and_mul(c1,c2,expert_ids=ids.view(-1),expert_step=1)
        y=moe_wna16_marlin_gemm(c2,c3,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,
            bank['workspace'],sorted_ids,experts,padded,tw,top_k=1,size_m=m*8,size_n=h,size_k=h,**common).view(m,8,h)
        out.copy_(fused_post_expert_norm_weighted_sum(y,tw,norm_weight=self.state['_fork_marlin_gamma'],
            eps=1e-8,weight_first=True,slot_expert_ids=ids))

def _fork_prepare_marlin(weights,st):
    st['_fork_marlin_bank']=_marlin_bank(weights)  # clones before SGLang loader mutates layouts
    st['_fork_marlin_gamma']=(weights['post_expert_norm'].float()+1).to(torch.bfloat16)
    st['_fork_marlin_adapter']=_ForkMarlinAdapter(st)

def _fork_moe_fallback(M,st):
    return st['_fork_marlin_adapter']


def _fork_build_tasks(weights, st):
    dev = weights['w13'].device
    prop = torch.cuda.get_device_properties(dev)
    eligible = bool(st['ma_sfx']) and (prop.major, prop.minor) == (9, 0)
    st['_fork_native_eligible'] = eligible
    st['_fork_build_errors'] = {}
    if not eligible:
        return []
    def build(m):
        with torch.cuda.device(dev):
            try:
                _fork_native_moe(m, st)
            except RuntimeError as exc:
                st['_fork_build_errors'][m] = str(exc)[-300:]
    
    return [lambda m=m: build(m) for m in (1, 8, 16, 32, 128)]

def _fork_install_fallback(weights, st, reason):
    st['_fork_native_eligible'] = False
    st['ma_w'], st['ma_scratch'] = (), ()
    st['_fork_fallback_kind'] = 'Marlin'
    _fork_prepare_marlin(weights, st)
    dev = weights['w13'].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for m in (1,8,16,32,33,128,256):
        x = torch.zeros((m,4096), device=dev, dtype=torch.bfloat16)
        ids = torch.full((m,8), -1, device=dev, dtype=torch.int32)
        ids[:,0] = 0
        tw = torch.full((m,8), .125, device=dev, dtype=torch.float32)
        st['_fork_marlin_adapter'].okc_run((),(),x,ids,tw,torch.empty_like(x),0,32,sms)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    st['_fork_moe_status'] = st['_fork_fallback_kind'] + ': ' + reason
    print('FORK_MOE_ROUTE', st['_fork_moe_status'], flush=True)

def _ma_prepare(weights, st):
    
    torch.cuda.synchronize()
    errors = st.pop('_fork_build_errors')
    if not st['_fork_native_eligible'] or errors:
        reason = 'native build failed '+str(sorted(errors.items())) if errors else 'scale/architecture fallback'
        return _fork_install_fallback(weights, st, reason)
    if set(st['ma_ext']) != {'tiny','tiny2','mid','b4','main'}:
        raise RuntimeError('unexpected native module keys: '+str(sorted(st['ma_ext'])))
    dev = weights['w13'].device
    _fork_native_ma_prepare(weights, st)
    torch.cuda.synchronize()
    for m in (1,8,16,32,33,128,256):
        x=torch.zeros((m,4096),device=dev,dtype=torch.bfloat16)
        ids=torch.full((m,8),-1,device=dev,dtype=torch.int32)
        ids[:,0]=0  
        tw=torch.full((m,8),.125,device=dev,dtype=torch.float32)
        out=torch.empty_like(x)
        torch.cuda.synchronize()  
        try:
            _fork_native_moe(m,st).okc_run(st['ma_w'],st['ma_scratch'],x,ids,tw,out,0,32,
                                         torch.cuda.get_device_properties(dev).multi_processor_count)
        except RuntimeError as exc:
            if 'FORK_MOE_RESOURCE' not in str(exc):
                raise
            return _fork_install_fallback(weights, st, str(exc)[-300:])
        torch.cuda.synchronize()  
    st['_fork_moe_status']='native repaired: cooperative+BF16, MFIN/DF/YBF/PDL off'
    print('FORK_MOE_ROUTE',st['_fork_moe_status'],flush=True)

def _moe(M=None, state=None):
    if state is None:
        return None  
    if not state.get('_fork_native_eligible',False):
        return _fork_moe_fallback(M,state)
    return _fork_native_moe(M,state)

_g1a_prepare=prepare
def prepare(weights):
    state=_g1a_prepare(weights)
    print("G1b4 backend", state["_fork_moe_status"],
          "Lt plans", sorted(state.get("graft_lt", {})), flush=True)
    return state

# 
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
# 
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
#    1. Definitions.
# 
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
# 
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
# 
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
# 
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
# 
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
# 
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
# 
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
# 
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
# 
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
# 
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
# 
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
# 
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
# 
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
# 
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
# 
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
# 
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
# 
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
# 
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
# 
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
# 
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
# 
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
# 
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
# 
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
# 
#    END OF TERMS AND CONDITIONS
# 
#    APPENDIX: How to apply the Apache License to your work.
# 
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
# 
#    Copyright [yyyy] [name of copyright owner]
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# 
# muse-open-kernel-challenge
# Copyright 2026 team keell0renz (Bohdan Agarkov)
# 
# This product includes code from the following projects. Each kernel file keeps the original attribution lines.
# 
# Apache License 2.0:
# - vjeux/okc-2026 (https://github.com/vjeux/okc-2026, commit f35a41e): the starting point of every board.
#   "Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0."
# - alygg77/okc-muse-spark-kernels (commit 6cae49a): decode layer, MoE, prefill bases; pinned cuBLASLt plans and ctypes
#   wrapper in the bf16 projections. "Submitted by alygg77 under the Apache License 2.0 with this notice retained."
# - chenwainuo/okc (muse_bf16_proj/submission.py @ 3d418e1): rs128 GEMVs (bf16 projections, decode layer).
# - ZhimaoL/sglang (okc/x-vfgpdl, okc/cand12pre, okc/x-n72-naw okc_patch kernels): verify-attention kernels (wst, vfg) and the
#   prefill FMHA (fmha7, N72).
# - Madhavan113/ParadigmKernelChallenge (c0f3460, dab77b3, 012d92b): TMA bulk-reduction merge in attention decode.
# - Matsvei (kernarc user 'matvey', agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0): the
#   per-thread CUDA-context bind helper and cuBLASLt plan pins.
# 
# MIT License:
# - NathenFernandes/muse-bf16-proj (submission.py @ 7385e5d), Copyright (c) 2026 Nathen Fernandes: routertail_small and
#   ol_projection kernels in kernels/muse_bf16_proj.py, source unchanged.
# 
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#   persons to whom the Software is furnished to do so, subject to the following conditions:
# 
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
# 
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Added by yelgabo, November 1, 2026: private-state startup validation.
def _release_private_scratch(torch, buffers, sentinel):
    private, tails = [], []
    for original in buffers:
        size = original.numel() * original.element_size()
        words = (size + 3) // 4
        full = torch.full((words + 65536,), sentinel, dtype=torch.int32, device=original.device)
        view = full[:words].view(torch.uint8)[:size].view(original.dtype).view_as(original)
        view.copy_(original)
        private.append(view)
        tails.append(full[words:])
    return private, tails


def _release_routes(torch, size, device):
    rows = torch.arange(size, device=device, dtype=torch.int32)[:, None]
    cols = torch.arange(8, device=device, dtype=torch.int32)[None, :]
    start = 96
    regular = start + (rows + cols) % 32
    hot = (start + cols).expand(size, 8)
    partial = torch.where((cols == 0) & (rows % 3 != 1), start + rows % 4, start + 32 + cols)
    remote = (start + 32 + cols).expand(size, 8)
    return [torch.where((g >= start) & (g < start + 32), g - start, -1).contiguous()
            for g in (regular, hot, partial, remote)]


def _release_verify(torch, actual, expected):
    if actual.dtype != torch.bfloat16 or actual.shape != expected.shape:
        raise RuntimeError('G startup output shape or dtype')
    a, b = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise RuntimeError('G startup nonfinite output')
    diff = a - b
    global_error = float(diff.norm() / b.norm().clamp_min(1e-8))
    row_error = float((diff.norm(dim=1) / b.norm(dim=1).clamp_min(1e-8)).max())
    if global_error > .008 or row_error > .075:
        raise RuntimeError(f'G startup expert comparison failed ({global_error}, {row_error})')


def _release_expert_self_test(torch, weights, production, selector, native_factory,
                              scratch_factory, marlin_prepare, sentinel,
                              sizes=(128, 32, 33, 256, 1, 8, 16), deadline=None):
    import time
    device = weights['w13'].device
    generator = torch.Generator(device=device).manual_seed(768921)
    private = dict(production)
    private['ma_ext'] = {}
    
    private['ma_scratch'], tails = _release_private_scratch(torch, scratch_factory(device), sentinel)
    production_storage = {x.untyped_storage().data_ptr() for x in production.get('ma_scratch', ())}
    if any(x.untyped_storage().data_ptr() in production_storage for x in private['ma_scratch']):
        raise RuntimeError('G startup mutable state alias')
    native = production['_fork_native_eligible']
    reference_state = {}
    marlin_prepare(weights, reference_state)
    reference = reference_state['_fork_marlin_adapter']
    try:
        if not native:
            # Preserve the real production fallback for supported non-native weight layouts.
            marlin_prepare(weights, private)
        helper = selector(1, private)
        checks = 0
        for size in sizes:
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError('G startup self-test exceeded prepare budget')
            ext = selector(size, private)
            if native and (ext is not native_factory(size, private) or helper is not native_factory(1, private)):
                raise RuntimeError('G startup native selection drift')
            routes = _release_routes(torch, size, device)
            banks = torch.randn((4, size, 4096), device=device, dtype=torch.bfloat16, generator=generator)
            scores = torch.randn((4, size, 8), device=device, dtype=torch.float32, generator=generator).softmax(-1)
            x, ids, tw = banks[0].clone(), routes[0].clone(), scores[0].clone()
            out, empty_out = torch.empty_like(x), torch.empty_like(x)
            helper_out = torch.empty_like(x[:1])
            empty_ids = routes[3]
            
            empty_out.fill_(17)
            ext.okc_run(private['ma_w'], private['ma_scratch'], x, empty_ids, tw, empty_out,
                        0, 32, production['nsm'])
            torch.cuda.synchronize()
            if int(torch.count_nonzero(empty_out)):
                raise RuntimeError('G startup first empty route did not write zeros')
            refs = []
            for i in range(3):
                ref = torch.empty_like(x)
                reference.okc_run((), (), banks[i], routes[i], scores[i], ref, 0, 32, production['nsm'])
                refs.append(ref)
            refs.append(torch.zeros_like(x))

            def calls():
                out.fill_(float('nan'))
                ext.okc_run(private['ma_w'], private['ma_scratch'], x, ids, tw, out, 0, 32, production['nsm'])
                empty_out.fill_(17)
                ext.okc_run(private['ma_w'], private['ma_scratch'], x, empty_ids, tw, empty_out, 0, 32, production['nsm'])
                helper_out.fill_(-23)
                helper.okc_run(private['ma_w'], private['ma_scratch'], x[:1], empty_ids[:1], tw[:1],
                               helper_out, 0, 32, production['nsm'])

            calls()
            torch.cuda.synchronize()
            _release_verify(torch, out, refs[0])
            if int(torch.count_nonzero(empty_out)) or int(torch.count_nonzero(helper_out)):
                raise RuntimeError('G startup eager empty route did not write zeros')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                calls()
            
            for i in (3, 1, 2, 0, 2, 3, 0, 1):
                x.copy_(banks[i])
                ids.copy_(routes[i])
                tw.copy_(scores[i])
                graph.replay()
                torch.cuda.synchronize()
                _release_verify(torch, out, refs[i])
                if (i == 3 and int(torch.count_nonzero(out))) or int(torch.count_nonzero(empty_out)) or int(torch.count_nonzero(helper_out)):
                    raise RuntimeError('G startup replay empty route did not write zeros')
                checks += 1
            del graph
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError('G startup conformance incomplete within prepare budget')
        if any(bool((tail != sentinel).any()) for tail in tails):
            raise RuntimeError('G startup expert scratch overrun')
        print('G_RELEASE_STARTUP_OK', 'native' if native else 'Marlin', tuple(sizes), checks, 'verified_replays', flush=True)
    finally:
        reference_state.clear()
        private.clear()


_release_normal_prepare = prepare
def prepare(weights):
    import time
    deadline = time.monotonic() + 295
    state = _release_normal_prepare(weights)
    _release_expert_self_test(torch, weights, state, _moe, _fork_native_moe,
                              _ma_make_scratch, _fork_prepare_marlin, SCRATCH_SENTINEL, deadline=deadline)
    return state
"""
_VERIFY_HYBRID_G = _verify_hybrid_module('_okc_verify_production_g', _VERIFY_HYBRID_SOURCE_G)

_VERIFY_HYBRID_SOURCE_OWNED = """\
# okc: problem=muse_decode_layer

from typing import TypedDict

import hashlib

import threading

import time

import types

import torch

import torch.nn.functional as F

from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sgl_kernel.scalar_type import scalar_types

from torch import nn

from torch.utils.cpp_extension import load_inline

from sglang.kernels.ops.activation.activation import silu_and_mul

from sglang.kernels.ops.kvcache.kvcache import store_cache

from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm

from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting

from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate

from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm

from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)

from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm

from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)

from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

PORT_SK_PDL = True

HIDDEN = 8192

RMS_EPS = 1e-5

POST_NORM_EPS = 1e-8

N_EXPERTS = 256

TOP_K = 8

LOCAL_EXPERTS = 32

D_EXPERT = 4096

ROUTER_MULTIPLIER = 0.1767766952966369

ROUTE_EPS = 1e-15

ROPE_THETA = 500_000.0

_CPP_SRC = r'''#include <torch/extension.h>
std::vector<torch::Tensor> route_fwd(torch::Tensor logits, int64_t lo_off, torch::Tensor x,
                                     torch::Tensor w, torch::Tensor bias, double mult,
                                     double eps, double denom_eps);
std::vector<torch::Tensor> attn_boundary(torch::Tensor stream, torch::Tensor branch,
                                         torch::Tensor alpha, torch::Tensor beta,
                                         torch::Tensor w_next, double eps_post,
                                         double eps_next, bool want_router, int64_t geo);'''

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;

 
 
namespace {

constexpr int kMaxLocal = 32;
constexpr int kMaxRoutes = 4096;
constexpr int kTopK = 8;
constexpr int kExpertHidden = 4096;

 
 
 
template <typename Kern, typename... Args>
void launch_ex(Kern kern, dim3 grid, dim3 block, int smem, cudaStream_t stream, bool pdl,
               Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

template <typename Kern, typename... Args>
void launch_ex_pdl(Kern kern, dim3 grid, dim3 block, int smem, cudaStream_t stream,
                   Args... args) {
    launch_ex(kern, grid, block, smem, stream, true, args...);
}

 
 
constexpr bool kPdlGlue = true;

__device__ __forceinline__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}
constexpr int NORM_N = 4096;
constexpr int NORM_THREADS = 128;
constexpr int NORM_PER = NORM_N / NORM_THREADS / 4;

constexpr int ROUTE_EXPERTS = 256;
constexpr int ROUTE_PER_LANE = ROUTE_EXPERTS / 32;

template <typename InT, bool HILO>
__global__ void __launch_bounds__(NORM_THREADS) route_kernel(
        const float* __restrict__ logits, long long lstride, int lo_off,
        const float* __restrict__ bias, const InT* __restrict__ x, long long row_stride,
        const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
        int* __restrict__ ids, float* __restrict__ wts, float mult, float eps,
        float denom_eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    __shared__ float red[NORM_THREADS / 32];
    __shared__ float mean_sq;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float wv[NORM_PER][4];
#pragma unroll
    for (int i = 0; i < NORM_PER; ++i) {
        const uint2 u = *reinterpret_cast<const uint2*>(w + (tid + i * NORM_THREADS) * 4);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        wv[i][0] = a.x;
        wv[i][1] = a.y;
        wv[i][2] = b.x;
        wv[i][3] = b.y;
    }
    float bv[ROUTE_PER_LANE];
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < ROUTE_PER_LANE; ++j) {
            bv[j] = bias[lane + 32 * j];
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float lg[ROUTE_PER_LANE];
    if (warp == 0) {
        const float* lrow = logits + static_cast<size_t>(blockIdx.x) * lstride;
#pragma unroll
        for (int j = 0; j < ROUTE_PER_LANE; ++j) {
            lg[j] = lrow[lane + 32 * j];
            if constexpr (HILO) {
                lg[j] = lg[j] + lrow[lo_off + lane + 32 * j];
            }
        }
    }
    const InT* row = x + static_cast<size_t>(blockIdx.x) * row_stride;
    float xv[NORM_PER][4];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NORM_PER; ++i) {
        const int c = (tid + i * NORM_THREADS) * 4;
        if constexpr (std::is_same_v<InT, float>) {
            const float4 v = *reinterpret_cast<const float4*>(row + c);
            xv[i][0] = round_bf16(v.x);
            xv[i][1] = round_bf16(v.y);
            xv[i][2] = round_bf16(v.z);
            xv[i][3] = round_bf16(v.w);
        } else {
            const uint2 u = *reinterpret_cast<const uint2*>(row + c);
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            xv[i][0] = a.x;
            xv[i][1] = a.y;
            xv[i][2] = b.x;
            xv[i][3] = b.y;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            ss = fmaf(xv[i][j], xv[i][j], ss);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_down_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
        red[warp] = ss;
    }
    __syncthreads();
    if (tid == 0) {
        mean_sq = ((red[0] + red[2]) + (red[1] + red[3])) / static_cast<float>(NORM_N);
    }
    __syncthreads();
    const float rstd = rsqrtf(mean_sq + eps);
    __nv_bfloat16* orow = out + static_cast<size_t>(blockIdx.x) * NORM_N;
#pragma unroll
    for (int i = 0; i < NORM_PER; ++i) {
        const int c = (tid + i * NORM_THREADS) * 4;
        const __nv_bfloat162 h0 = __floats2bfloat162_rn(wv[i][0] * (rstd * xv[i][0]),
                                                        wv[i][1] * (rstd * xv[i][1]));
        const __nv_bfloat162 h1 = __floats2bfloat162_rn(wv[i][2] * (rstd * xv[i][2]),
                                                        wv[i][3] * (rstd * xv[i][3]));
        *reinterpret_cast<uint2*>(orow + c) = make_uint2(
            *reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
    }
    if (warp != 0) {
        return;
    }
    float sc[ROUTE_PER_LANE];
    float key[ROUTE_PER_LANE];
#pragma unroll
    for (int j = 0; j < ROUTE_PER_LANE; ++j) {
        sc[j] = 1.f / (1.f + expf(-(lg[j] * mult)));
        key[j] = sc[j] + bv[j];
    }
    int pid[8];
    float psc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float best = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < ROUTE_PER_LANE; ++j) {
            if (key[j] > best) {
                best = key[j];
                bi = lane + 32 * j;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        const int jj = bi >> 5;
        float mine = 0.f;
#pragma unroll
        for (int j = 0; j < ROUTE_PER_LANE; ++j) {
            if (j == jj) {
                mine = sc[j];
                if (lane == (bi & 31)) {
                    key[j] = -INFINITY;
                }
            }
        }
        pid[k] = bi;
        psc[k] = __shfl_sync(0xffffffffu, mine, bi & 31);
    }
    if (lane == 0) {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sum += psc[k];
        }
        const float den = sum + denom_eps;
        int4* irow = reinterpret_cast<int4*>(ids + static_cast<size_t>(blockIdx.x) * 8);
        float4* wrow = reinterpret_cast<float4*>(wts + static_cast<size_t>(blockIdx.x) * 8);
        irow[0] = make_int4(pid[0], pid[1], pid[2], pid[3]);
        irow[1] = make_int4(pid[4], pid[5], pid[6], pid[7]);
        wrow[0] = make_float4(psc[0] / den, psc[1] / den, psc[2] / den, psc[3] / den);
        wrow[1] = make_float4(psc[4] / den, psc[5] / den, psc[6] / den, psc[7] / den);
    }
}


}  
std::vector<torch::Tensor> route_fwd(torch::Tensor logits, int64_t lo_off, torch::Tensor x,
                                     torch::Tensor w, torch::Tensor bias, double mult,
                                     double eps, double denom_eps) {
    const int m = x.size(0);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == NORM_N && x.stride(1) == 1 && x.stride(0) % 4 == 0,
                "route_fwd: x must be [T, 4096] with unit column stride and 16-byte rows");
    TORCH_CHECK(w.is_contiguous() && w.numel() == NORM_N && w.scalar_type() == torch::kBFloat16,
                "route_fwd: weight must be BF16 [4096]");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == m && logits.stride(1) == 1
                && logits.scalar_type() == torch::kFloat32
                && logits.size(1) >= ROUTE_EXPERTS + lo_off && lo_off >= 0,
                "route_fwd: logits must be FP32 [T, 256 (+ lo_off)] with unit column stride");
    TORCH_CHECK(bias.is_contiguous() && bias.numel() == ROUTE_EXPERTS
                && bias.scalar_type() == torch::kFloat32, "route_fwd: bias must be FP32 [256]");
    auto out = torch::empty({m, NORM_N}, x.options().dtype(torch::kBFloat16));
    auto ids = torch::empty({m, 8}, x.options().dtype(torch::kInt32));
    auto wts = torch::empty({m, 8}, x.options().dtype(torch::kFloat32));
    if (m == 0) {
        return {out, ids, wts};
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const float* lp = logits.data_ptr<float>();
    const long long ls = logits.stride(0);
    const long long xs = x.stride(0);
    const int lo = static_cast<int>(lo_off);
    const float* bp = bias.data_ptr<float>();
    int* ip = ids.data_ptr<int>();
    float* wtp = wts.data_ptr<float>();
    const float fm = static_cast<float>(mult);
    const float fe = static_cast<float>(eps);
    const float fd = static_cast<float>(denom_eps);
    if (x.scalar_type() == torch::kFloat32) {
        const auto* xp = static_cast<const float*>(x.data_ptr());
        if (lo > 0) {
            launch_ex_pdl(route_kernel<float, true>, dim3(m), dim3(NORM_THREADS), 0, stream,
                          lp, ls, lo, bp, xp, xs, wp, op, ip, wtp, fm, fe, fd);
        } else {
            launch_ex_pdl(route_kernel<float, false>, dim3(m), dim3(NORM_THREADS), 0, stream,
                          lp, ls, lo, bp, xp, xs, wp, op, ip, wtp, fm, fe, fd);
        }
    } else {
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "route_fwd: x must be BF16 or FP32");
        const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
        if (lo > 0) {
            launch_ex_pdl(route_kernel<__nv_bfloat16, true>, dim3(m), dim3(NORM_THREADS), 0,
                          stream, lp, ls, lo, bp, xp, xs, wp, op, ip, wtp, fm, fe, fd);
        } else {
            launch_ex_pdl(route_kernel<__nv_bfloat16, false>, dim3(m), dim3(NORM_THREADS), 0,
                          stream, lp, ls, lo, bp, xp, xs, wp, op, ip, wtp, fm, fe, fd);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, ids, wts};
}
'''

_ONE_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdint>
#include <vector>

 
 
 
 
 
 
 
 

namespace {
namespace one {

constexpr int D = 64;
constexpr int HQ = 16;
constexpr int HKV = 2;
constexpr int GROUP = HQ / HKV;
constexpr int QKV_STRIDE = (2 * HQ + 2 * HKV) * D;
constexpr int GATE_OFF = (HQ + 2 * HKV) * D;
constexpr int BN = 64;
constexpr int ROWB = HKV * D;
constexpr int TILE_BYTES = BN * ROWB;
constexpr int STAGES = 3;
constexpr int NIDX = 8;
constexpr int VT_STRIDE = 5120;
constexpr float EPS = 1e-5f;
constexpr float NEG_INF = -INFINITY;
 
constexpr float OFF_FLOOR = -1e30f;
constexpr int COOP_POLL_LIMIT = 1 << 24;

 
 
 
#ifndef ONE_DIAG
#define ONE_DIAG 0
#endif
 
 
 
 
#ifndef ONE_DIET
#define ONE_DIET 1
#endif
 
 
 
 
 
#ifndef ONE_FIN8
#define ONE_FIN8 1
#endif
 
 
 
 
#ifndef ONE_KVFAST
#define ONE_KVFAST 0
#endif
 
 
 
 
 
 
 
 
#ifndef ONE_QKOV
#define ONE_QKOV 1
#endif
#ifndef ONE_EREL
#define ONE_EREL 1
#endif
#ifndef ONE_PCHK
#define ONE_PCHK 1
#endif
#ifndef ONE_UNR2
#define ONE_UNR2 1
#endif
 
 
 
 
 
 
 
#ifndef ONE_PFAST
#define ONE_PFAST 1
#endif
// Bench-only attribution builds (docs/runs-attention.md, 10-29), never judged:
// ONE_STAMP=1 records per-CTA times and wait cycles into dbg[STAMP_BASE ..] (flat producer
// builds); ONE_PROBE=1 makes the producer's cached-row copies zero-fill without reading the cache
// (consumer-bound time, wrong output); ONE_PROBE=2 drops the consumers' per-tile
// fence.proxy.async after the full barrier (its cost, ordering not guaranteed); ONE_PROBE=3
// (q_len 1 flat) zero-fills as 1 and the consumers release each stage without computing on it
// (the producer-bound pace).
#ifndef ONE_STAMP
#define ONE_STAMP 0
#endif
#ifndef ONE_PROBE
#define ONE_PROBE 0
#endif
 
 
 
 
 
 
#ifndef ONE_WAVE
#define ONE_WAVE 0
#endif
 
 
 
 
 
 
 
 
#ifndef ONE_DCARRY
#define ONE_DCARRY 1
#endif
 
 
 
 
 
#ifndef ONE_PK16
#define ONE_PK16 1
#endif
 
 
 
 
 
 
#ifndef ONE_VHOIST
#define ONE_VHOIST 1
#endif
 
 
 
 
 
 
 
 
 
#ifndef ONE_XPIPE
#define ONE_XPIPE 0
#endif
 
 
 
 
 
 
#ifndef ONE_TRIG
#define ONE_TRIG 0
#endif
 
 
 
 
 
#ifndef ONE_DETM
#define ONE_DETM 0
#endif
 
 
 
 
 
 
 
 
#ifndef ONE_HMRG
#define ONE_HMRG 1
#endif
 
 
 
 
#ifndef ONE_HMRGF
#define ONE_HMRGF 0
#endif
 
 
 
 
 
 
 
#ifndef ONE_F1
#define ONE_F1 0
#endif
 
 
 
#ifndef ONE_T9
#define ONE_T9 0
#endif
 
 
 
 
 
 
 
 
 
 
 
 
#ifndef ONE_PRQ
#define ONE_PRQ 31
#endif
constexpr int STAMP_BASE = 1024;
constexpr int STAMP_W = 16;

 
 
 
 
 
constexpr int M_COOP = 2;
constexpr int M_ONE = 4;
constexpr int M_SPEC = 8;
 
constexpr int M_FLAT = 16;
 
 
constexpr int SEGK1 = 400;
constexpr int SEGK8 = 600;
 
 
constexpr int M_PROD = 32;
 
 
constexpr int M_SOLO = 64;

 
 
 
 
#ifndef ONE_NS1
#define ONE_NS1 3
#endif

// ONE_PBULK (mode 49): the producer's page entries arrive by one bulk copy per tile into a
// shared ring of PB_SLOTS windows, three tiles ahead, instead of four `ld.global.nc` per thread
// per tile held in registers. The mechanism is the public flat q_len 1 kernel's (vjeux,
// okc-2026, Apache-2.0: `fetch_pages` / `load_pages_t`); this is our own code for our ring.
// Bench 4b60264a276c4805: mode 49's producer alone trails the public producer by 4.7 / 8.0 us
// at decode 16 / 32 full without DRAM traffic.
#ifndef ONE_PBULK
#define ONE_PBULK 0
#endif
 
 
#ifndef ONE_PBDBG
#define ONE_PBDBG 0
#endif

__host__ __device__ constexpr bool one_pb(int L, int mode) {
    return ONE_PBULK != 0 && L == 1 && (mode & M_PROD) && !(mode & M_SOLO);
}

__host__ __device__ constexpr int one_ns(int L, int mode) {
    return (mode & M_PROD) && L == 8 ? 4 : ((mode & M_PROD) && L == 1 ? ONE_NS1 : STAGES);
}

__host__ __device__ constexpr int one_threads(int L, int mode) {
    return (L == 8 ? 256 : 128) + ((mode & M_PROD) ? 128 : 0);
}

__host__ __device__ constexpr int one_minb(int L, int mode) {
    return (mode & M_SOLO) ? 1 : (L == 8 || (mode & M_PROD) ? 2 : 4);
}

 
 
 
__host__ __device__ constexpr int one_prod_regs(int L) {
    return L == 8 ? 32 : 56;
}

__host__ __device__ constexpr int one_cons_regs(int L) {
    return L == 8 ? (ONE_XPIPE == 1 ? 112 : 104) : 200;
}

 
 
constexpr int CNT_COOP = 4096;
constexpr int CNT_EPOCH = 8192;
constexpr int CNT_FLAT = 16384;
 
constexpr int CNT_SMR = CNT_FLAT + 4096;
constexpr int CNT_INTS = 32768;

struct Params {
    const __nv_bfloat16* qkv;
    const int* page_table;
    const int* seq_lens;
    const int64_t* cache_loc;
    const float* inv_freq;
    const float* k_scale;
    const float* v_scale;
    uint8_t* k_cache;
    uint8_t* v_cache;
    __nv_bfloat16* out;
    float* ws_acc;
    float* ws_l;
    int* ws_cnt;
    int* dbg;
    int pt_stride;
    int page_size;
    int window;
    int rope;
    int n_splits;
    uint32_t magic;
    int nc;
    float sl2;
     
     
    int phase_limit;
     
    int lp;
     
    int off32;
     
    int wave_n;
};

constexpr int PB_SLOTS = 4;
constexpr int PB_SLOT_BYTES = 320;

template <int L, int NS, bool PB = false>
struct Smem {
    static constexpr int ROWS = HQ * L;
    static constexpr int KV = 0;
    static constexpr int Q = KV + NS * 2 * TILE_BYTES;
    static constexpr int Q_BYTES = L == 8 ? HKV * 4096 : HQ * 128;
    static constexpr int VT = Q + Q_BYTES;
    static constexpr int VT_BYTES = L == 8 ? 4 * VT_STRIDE : 0;
     
    static constexpr int QST = VT;
    static constexpr int QST_BYTES = ROWS * 128;
    static constexpr int KVST = QST + (QST_BYTES > VT_BYTES ? QST_BYTES : VT_BYTES);
    static constexpr int KN = KVST + L * 4 * 128;
    static constexpr int VN = KN + L * ROWB;
    static constexpr int IDX = VN + L * ROWB;
    static constexpr int ROPE = IDX + NIDX * BN * 4;
    static constexpr int CONST = ROPE + L * 2 * 32 * 4;
    static constexpr int FLAG = CONST + 256;
     
    static constexpr int OFF = FLAG + 32;
     
    static constexpr int BAR = OFF + 2304;
     
     
    static constexpr int PSL = (BAR + 2 * NS * 8 + 127) / 128 * 128;
    static constexpr int PBAR = PSL + PB_SLOTS * PB_SLOT_BYTES;
     
    static constexpr int PFREE = PBAR + PB_SLOTS * 8;
    static constexpr int BYTES = PB ? PFREE + PB_SLOTS * 8 : BAR + 2 * NS * 8;
    static_assert(Q % 1024 == 0 && VT % 1024 == 0, "swizzled tiles need 1024-byte bases");
};

 
 
constexpr int C_INVF = 0;
constexpr int C_KS = 128;
constexpr int C_VS = 132;
constexpr int C_EPOCH = 136;
constexpr int C_LOC = 160;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n"
                 :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

 
__device__ __forceinline__ void cp_async16_if(uint32_t dst, const void* src, uint32_t bytes,
                                              bool doit) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %3, 0;\\n"
                 "@p cp.async.cg.shared.global [%0], [%1], 16, %2;\\n}\\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(static_cast<int>(doit)) : "memory");
}

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\\n"
                 :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\\n"
                 :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
}

__device__ __forceinline__ void named_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\\n" :: "r"(id), "r"(count) : "memory");
}

 
__device__ __forceinline__ bool named_sync_or(int id, int count, bool v) {
    uint32_t r;
    asm volatile("{\\n.reg .pred pi, po;\\nsetp.ne.u32 pi, %3, 0;\\n"
                 "bar.red.or.pred po, %1, %2, pi;\\nselp.u32 %0, 1, 0, po;\\n}\\n"
                 : "=r"(r) : "r"(id), "r"(count), "r"(static_cast<uint32_t>(v)) : "memory");
    return r != 0;
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" :: "r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n"
                 :: "r"(bar), "r"(bytes) : "memory");
}

 
 
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];\\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

 
 
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\\n" :: "r"(bar) : "memory");
}

__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\\n.reg .pred p;\\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n"
                 "selp.u32 %0, 1, 0, p;\\n}\\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

 
 
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    if (mbar_try(bar, parity)) {
        return;
    }
    int polls = 0;
    while (!mbar_try(bar, parity)) {
        if (++polls > (1 << 26)) {
            __trap();
        }
    }
}

template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" :: "n"(N));
}

template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\\n" :: "n"(N));
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];\\n" :: "l"(p));
}

__device__ __forceinline__ int ld_nc_s32(const int* p) {
    int v;
    asm volatile("ld.global.nc.s32 %0, [%1];\\n" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n"
                 :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

__device__ __forceinline__ void red_add_f32(float* p, float a) {
    asm volatile("red.global.add.f32 [%0], %1;\\n" :: "l"(p), "f"(a) : "memory");
}

__device__ __forceinline__ int atom_add_acq_rel(int* p, int v) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ float4 ld_cg_f4(const float* p) {
    return __ldcg(reinterpret_cast<const float4*>(p));
}

__device__ __forceinline__ void st_cg_f4(float* p, float4 v) {
    __stcg(reinterpret_cast<float4*>(p), v);
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\\n" : "=f"(y) : "f"(x));
    return y;
}

 
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\\n" : "=h"(lo) : "f"(b), "f"(a));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\\n" : "=h"(hi) : "f"(d), "f"(c));
#if ONE_PK16
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
#else
    return __byte_perm(static_cast<uint32_t>(lo), static_cast<uint32_t>(hi), 0x5410);
#endif
}

__device__ __forceinline__ uint16_t e4m3x2(float a, float b) {
    uint16_t v;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\\n" : "=h"(v) : "f"(b), "f"(a));
    return v;
}

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2(uint32_t two) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\n" : "=r"(r) : "h"(static_cast<uint16_t>(two)));
    return r;
}

__device__ __forceinline__ uint32_t pack_f16(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

 
__device__ __forceinline__ float oct_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v;
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

 
 
 
__device__ __forceinline__ float2 fast_sincos(float x) {
    const float k = rintf(x * 0.15915494309189535f);
    float r = fmaf(-k, 6.2831854820251465f, x);
    r = fmaf(-k, -1.7484555314695172e-7f, r);
    return make_float2(__sinf(r), __cosf(r));
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;\\n" : "=l"(t));
    return t;
}

__device__ __forceinline__ float sigmoid(float x) {
    return __fdividef(1.f, 1.f + __expf(-x));
}

 
 
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1,
                                              uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}

__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2,
                                        uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\\n"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

 
 
 
 
__device__ __forceinline__ uint64_t desc_kmaj(uint32_t addr, int lbo, int sbo) {
    return ((addr & 0x3ffffu) >> 4) | (static_cast<uint64_t>(lbo >> 4) << 16)
           | (static_cast<uint64_t>(sbo >> 4) << 32);
}

__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
    return ((addr & 0x3ffffu) >> 4) | (static_cast<uint64_t>(1) << 16)
           | (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);
}

__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
    return ((addr & 0x3ffffu) >> 4) | (static_cast<uint64_t>(1) << 16)
           | (static_cast<uint64_t>(512 >> 4) << 32) | (static_cast<uint64_t>(2) << 62);
}

__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\\n" :: "n"(N) : "memory");
}

template <int N>
__device__ __forceinline__ void reg_fence(float (&d)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        asm volatile("" : "+f"(d[i]) :: "memory");
    }
}

#define WG_D32 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \\
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"
#define WG_OUT32(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), \\
    "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), \\
    "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \\
    "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), \\
    "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

 
 
template <int ACC>
__device__ __forceinline__ void wgmma_qk(float (&d)[32], uint64_t da, uint64_t db) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %34, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D32
                 ", %32, %33, p, 1, 1;\\n}\\n"
                 : WG_OUT32(d) : "l"(da), "l"(db), "n"(ACC));
}

 
 
__device__ __forceinline__ void wgmma_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                 "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
                 "%31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1;\\n}\\n"
                 : WG_OUT32(d), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

 
 
 
 
template <int L>
__device__ __forceinline__ int k_swz(int chunk, int row) {
    return L == 8 ? chunk ^ (row & 7) : chunk ^ ((row & 1) << 2);
}

template <int L>
__device__ __forceinline__ int v_swz(int chunk, int row) {
    return L == 8 ? chunk ^ (row & 7) : chunk ^ (((row >> 1) & 3) << 1);
}

 
__device__ __forceinline__ int kmaj(int r, int kc) {
    return ((r >> 3) * 4 + kc) * 128 + (r & 7) * 16;
}

#if ONE_DETM
 
constexpr int DET_ROW = 68;

__device__ __forceinline__ size_t det_slot(int b, int G, int s, int w, int head) {
    return ((((static_cast<size_t>(b) * G + s) * 2 + w) * HQ) + head) * DET_ROW;
}
#endif

 
__device__ __forceinline__ int ws_row(int b, int nc, int copy, int L, int j, int head) {
    return ((b * nc + copy) * L + j) * HQ + head;
}


 
 
template <int L, int NT, bool NC8 = false>
__device__ __forceinline__ void finalize_rows(const Params& p, int r_lo, int r_hi, float v_scale,
                                              int tid) {
    constexpr int RPP = NT / 8;
    constexpr int ROWS = HQ * L;
    const int t8 = tid & 7;
    for (int base = r_lo; base < r_hi; base += RPP) {
        const int r = base + (tid >> 3);
        const bool act = r < r_hi;
        const int b = r / ROWS;
        const int lr = r - b * ROWS;
        const int j = lr >> 4;
        const int head = lr & 15;
        const size_t tok = static_cast<size_t>(b) * L + j;
        uint4 graw = make_uint4(0, 0, 0, 0);
        if (act) {
            graw = __ldg(reinterpret_cast<const uint4*>(p.qkv + tok * QKV_STRIDE + GATE_OFF
                                                        + head * D + t8 * 8));
        }
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float lsum = 0.f;
        if (NC8 && p.nc == 8) {
             
             
            float4 a[8], a2[8];
            float lv[8];
#pragma unroll
            for (int copy = 0; copy < 8; ++copy) {
                const int wr = ws_row(b, 8, copy, L, j, head);
                const float* row = p.ws_acc + static_cast<size_t>(wr) * D + t8 * 8;
                a[copy] = act ? ld_cg_f4(row) : make_float4(0.f, 0.f, 0.f, 0.f);
                a2[copy] = act ? ld_cg_f4(row + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
                lv[copy] = act && t8 == 0 ? __ldcg(p.ws_l + wr) : 0.f;
            }
            float lsum8 = 0.f;
#pragma unroll
            for (int copy = 0; copy < 8; ++copy) {
                acc[0] += a[copy].x;
                acc[1] += a[copy].y;
                acc[2] += a[copy].z;
                acc[3] += a[copy].w;
                acc[4] += a2[copy].x;
                acc[5] += a2[copy].y;
                acc[6] += a2[copy].z;
                acc[7] += a2[copy].w;
                lsum8 += lv[copy];
            }
            lsum = __shfl_sync(0xffffffffu, lsum8, (tid & 31) & ~7);
            if (act) {
#pragma unroll
                for (int copy = 0; copy < 8; ++copy) {
                    const int wr = ws_row(b, 8, copy, L, j, head);
                    float* row = p.ws_acc + static_cast<size_t>(wr) * D + t8 * 8;
                    st_cg_f4(row, make_float4(0.f, 0.f, 0.f, 0.f));
                    st_cg_f4(row + 4, make_float4(0.f, 0.f, 0.f, 0.f));
                    if (t8 == 0) {
                        __stcg(p.ws_l + wr, 0.f);
                    }
                }
            }
        }
         
         
        for (int copy = 0; copy < (NC8 && p.nc == 8 ? 0 : p.nc); ++copy) {
            const int wr = ws_row(b, p.nc, copy, L, j, head);
            float* row = p.ws_acc + static_cast<size_t>(wr) * D + t8 * 8;
            float* lp = p.ws_l + wr;
            float lv = 0.f;
            if (act) {
                const float4 a = ld_cg_f4(row);
                const float4 a2 = ld_cg_f4(row + 4);
                acc[0] += a.x;
                acc[1] += a.y;
                acc[2] += a.z;
                acc[3] += a.w;
                acc[4] += a2.x;
                acc[5] += a2.y;
                acc[6] += a2.z;
                acc[7] += a2.w;
                if (t8 == 0) {
                    lv = __ldcg(lp);
                }
            }
            lsum += __shfl_sync(0xffffffffu, lv, (tid & 31) & ~7);
            if (act) {
                st_cg_f4(row, make_float4(0.f, 0.f, 0.f, 0.f));
                st_cg_f4(row + 4, make_float4(0.f, 0.f, 0.f, 0.f));
                if (t8 == 0) {
                    __stcg(lp, 0.f);
                }
            }
        }
        const float inv = lsum > 0.f ? __fdividef(v_scale, lsum) : 0.f;
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc[e] = __bfloat162float(__float2bfloat16_rn(acc[e] * inv));
            ss += acc[e] * acc[e];
        }
        ss = oct_sum(ss);
        const float rn = rsqrtf(ss * (1.f / D) + EPS);
        if (act) {
            const uint32_t gw[4] = {graw.x, graw.y, graw.z, graw.w};
            uint32_t ow[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 gt =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[e]));
                const __nv_bfloat162 o2 = __float22bfloat162_rn(
                    make_float2(acc[2 * e] * rn * sigmoid(gt.x), acc[2 * e + 1] * rn * sigmoid(gt.y)));
                ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
            }
            *reinterpret_cast<uint4*>(p.out + tok * HQ * D + head * D + t8 * 8) =
                make_uint4(ow[0], ow[1], ow[2], ow[3]);
        }
    }
}

 
 
 
 
template <int NC>
__device__ __forceinline__ void coop_barrier(int* cnt, int* epoch, int ep, int mine, int total,
                                             int tid, int* dbg) {
    if (tid == 0) {
        const int old = atom_add_acq_rel(cnt + (ep & 1), mine);
        if (mine > 0 && old + mine == total) {
            __stcg(cnt + ((ep + 1) & 1), 0);
            __stcg(epoch, ep + 1);
        }
        int polls = 0;
        while (ld_acquire(cnt + (ep & 1)) < total) {
            __nanosleep(64);
            if (++polls > COOP_POLL_LIMIT) {
                __trap();
            }
        }
        if (dbg != nullptr) {
            atomicMax(dbg, polls);
        }
    }
    named_sync(1, NC);
}

#if ONE_DETM
 
 
 
template <int SEGK>
__device__ void finalize_det(const Params& p, const int* offs, int B, int G, int b,
                             float v_scale, int tid) {
    const float* part = reinterpret_cast<const float*>(p.dbg);
    const int head = tid >> 3;
    const int t8 = tid & 7;
    const long long total_v = offs[B] - SEGK;
    const int a = offs[b];
    const int n_b = offs[b + 1] - a - SEGK;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    for (int s = 0; s < G; ++s) {
        const int v0 = static_cast<int>(total_v * s / G);
        const int v1 = static_cast<int>(total_v * (s + 1) / G);
        if (!(a + n_b > v0 && a < v1)) {
            continue;
        }
        const int s0 = max(v0, a) - a;
        const int s1 = min(v1, a + n_b) - a;
        const int lo_k = s0 > 0 ? (s0 & ~(BN - 1)) : 0;
        const int hi_k = s1 < n_b ? (s1 & ~(BN - 1)) : n_b;
        if (lo_k >= hi_k) {
            continue;
        }
        for (int w = 0; w < 2; ++w) {
            const float* row = part + det_slot(b, G, s, w, head);
            const float4 x = ld_cg_f4(row + t8 * 8);
            const float4 y = ld_cg_f4(row + t8 * 8 + 4);
            acc[0] += x.x;
            acc[1] += x.y;
            acc[2] += x.z;
            acc[3] += x.w;
            acc[4] += y.x;
            acc[5] += y.y;
            acc[6] += y.z;
            acc[7] += y.w;
            lsum += __ldcg(row + 64);
        }
    }
    const float inv = lsum > 0.f ? __fdividef(v_scale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        acc[e] = __bfloat162float(__float2bfloat16_rn(acc[e] * inv));
        ss += acc[e] * acc[e];
    }
    ss = oct_sum(ss);
    const float rn = rsqrtf(ss * (1.f / D) + EPS);
    const uint4 graw = __ldg(reinterpret_cast<const uint4*>(
        p.qkv + static_cast<size_t>(b) * QKV_STRIDE + GATE_OFF + head * D + t8 * 8));
    const uint32_t gw[4] = {graw.x, graw.y, graw.z, graw.w};
    uint32_t ow[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[e]));
        const __nv_bfloat162 o2 = __float22bfloat162_rn(
            make_float2(acc[2 * e] * rn * sigmoid(gt.x), acc[2 * e + 1] * rn * sigmoid(gt.y)));
        ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
    }
    *reinterpret_cast<uint4*>(p.out + static_cast<size_t>(b) * HQ * D + head * D + t8 * 8) =
        make_uint4(ow[0], ow[1], ow[2], ow[3]);
}
#endif

template <int L, int MODE>
__global__ void __launch_bounds__(one_threads(L, MODE), one_minb(L, MODE))
attn_one_kernel(const Params p) {
    constexpr bool COOP = (MODE & M_COOP) != 0;
    constexpr bool ONE = (MODE & M_ONE) != 0;
    constexpr bool SPEC = (MODE & M_SPEC) != 0;
    constexpr bool FLAT = (MODE & M_FLAT) != 0;
    constexpr bool PROD = (MODE & M_PROD) != 0;
    constexpr bool SOLO = (MODE & M_SOLO) != 0;
    static_assert(!FLAT || !(ONE || SPEC), "flat mode partitions its own ranges");
    static_assert(!PROD || FLAT, "the producer warpgroup belongs to the flat kernel");
    static_assert(!SOLO || PROD, "SOLO is a producer-mode variant");
     
     
     
    constexpr bool DIET = ONE_DIET && !(PROD && L == 8);
     
     
     
    constexpr bool FIN8 = ONE_FIN8 && SPEC && !ONE;
    constexpr bool HMRG = (ONE_HMRG && !FLAT && L == 1) || (ONE_HMRGF && FLAT && L == 1);
    constexpr bool QKOV = ONE_QKOV && PROD && L == 8;
    constexpr bool EREL = ONE_EREL && PROD && L == 8;
    constexpr bool PCHK = ONE_PCHK && PROD && L == 8;
    constexpr int UNR8 = ONE_UNR2 && PROD && L == 8 ? 2 : 1;
    constexpr bool DCARRY = ONE_DCARRY && PROD && L == 8;
    constexpr bool VHOIST = ONE_VHOIST && QKOV;
    constexpr int XPIPE = PROD && L == 8 ? ONE_XPIPE : 0;
    static_assert(XPIPE == 0 || (QKOV && EREL && PCHK && DCARRY && VHOIST),
                  "ONE_XPIPE is written on top of the promoted mode 51 tile loop");
    constexpr bool WAVE = ONE_WAVE > 0 && PROD && !SOLO && L == 8;
    constexpr bool DETM = ONE_DETM && FLAT && !COOP && L == 1;
    constexpr int PRQ = !FLAT && L == 8 ? ONE_PRQ : 0;
    constexpr bool QK2 = (ONE_F1 & 1) && FLAT && L == 1;
    constexpr bool PE16 = (ONE_F1 & 2) && PROD && L == 1;
     
     
    constexpr int NC = L == 8 ? 256 : 128;
    constexpr int NS = one_ns(L, MODE);
    constexpr int ROWS = HQ * L;
    constexpr int SEGK = L == 8 ? SEGK8 : SEGK1;
    constexpr int PF = 3;
    constexpr bool PBULK = one_pb(L, MODE);
    using S = Smem<L, NS, PBULK>;
    extern __shared__ __align__(1024) uint8_t smem[];
    const int split = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int psize = p.page_size;
    const uint32_t smem_base = smem_u32(smem);
    int* idx_ring = reinterpret_cast<int*>(smem + S::IDX);
    int* flag = reinterpret_cast<int*>(smem + S::FLAG);
    const uint32_t full0 = smem_base + S::BAR;
    const uint32_t empty0 = full0 + NS * 8;
     
     
    if constexpr (ONE_TRIG != 0 && (L == 1 || SPEC)) {
        asm volatile("griddepcontrol.launch_dependents;");
    }

     
     
#define PHASE_EXIT(k) \\
    if (ONE_DIAG && p.phase_limit == (k)) { cp_wait<0>(); return false; }
     
     
    auto stamp = [&](int k) {
        if constexpr (ONE_DIAG) {
            if (p.dbg != nullptr && split == 0 && blockIdx.y == 0 && tid == 0) {
                p.dbg[1 + k] = static_cast<int>(globaltimer_ns() / 1000u);
            }
        }
    };
     
     
     
    constexpr bool STAMPS = ONE_STAMP && PROD;
    int* const srow = STAMPS && p.dbg != nullptr ? p.dbg + STAMP_BASE + split * STAMP_W : nullptr;
    auto srec = [&](int k, long long v) {
        if constexpr (STAMPS) {
            if (srow != nullptr) {
                srow[k] = static_cast<int>(v);
            }
        }
    };
    auto snow = [&]() -> long long {
        if constexpr (STAMPS) {
            return static_cast<long long>(globaltimer_ns() & 0xffffffffull);
        }
        return 0;
    };
    long long s_fullw = 0, s_pro = 0, s_loop = 0, s_seg0 = 0, s_l0 = 0;
    const long long s_k0 = STAMPS ? clock64() : 0;
    int s_tiles = 0, s_nseg = 0;
    if constexpr (STAMPS) {
        if (tid == 0) {
            srec(0, snow());
            uint32_t smid;
            asm volatile("mov.u32 %0, %%smid;\\n" : "=r"(smid));
            srec(15, smid);
        }
    }
     
    auto csync = [&]() {
        if constexpr (PROD) {
            named_sync(1, NC);
        } else {
            __syncthreads();
        }
    };

     
    int b = FLAT ? 0 : blockIdx.y;
    const int* pt = p.page_table + static_cast<size_t>(b) * p.pt_stride;
    int seq = 0, start = 0, end = 0, n_keys = 0, lo = 0, hi = 0, ntiles = 0, nact = 0;
    int chunk_keys = 0;
     
     
    const int seq_real = FLAT ? 0 : ld_nc_s32(p.seq_lens + b);
    auto derive = [&]() {
        start = p.window ? max(0, seq - p.window + 1) : 0;
        end = seq + L;
        n_keys = end - start;
        const int ntl = (n_keys + BN - 1) >> 6;
        const int chunk = ONE ? 1 : (p.magic ? static_cast<int>(__umulhi(
                                        static_cast<uint32_t>(ntl + p.n_splits - 1), p.magic))
                                              : ntl);
        chunk_keys = chunk << 6;
        lo = start + split * chunk_keys;
        hi = min(end, lo + chunk_keys);
        ntiles = ONE ? 1 : max(0, (hi - lo + BN - 1) >> 6);
        nact = COOP ? min(p.n_splits, (ntl + chunk - 1) / chunk) : 0;
    };

     
     
     
     
    int* offs = reinterpret_cast<int*>(smem + S::OFF);
    int v0 = 0, v1 = 0, b_first = 0, b_last = -1, keys_all = 0;
    const int B = FLAT ? p.n_splits : 1;
    const int G = gridDim.x;
     
     
     
    auto seg_bounds = [&](int bb) -> bool {
        const int a = offs[bb];
        const int n_b = offs[bb + 1] - a - SEGK;
        const int s0 = max(v0, a) - a;
        const int s1 = min(v1, a + n_b) - a;
        const int lo_k = s0 > 0 ? (s0 & ~(BN - 1)) : 0;
        const int hi_k = s1 < n_b ? (s1 & ~(BN - 1)) : n_b;
        if (lo_k >= hi_k) {
            return false;
        }
        pt = p.page_table + static_cast<size_t>(bb) * p.pt_stride;
        seq = n_b - L;
        start = 0;
        end = n_b;
        n_keys = n_b;
        lo = lo_k;
        hi = hi_k;
        ntiles = (hi - lo + BN - 1) >> 6;
        return true;
    };
    const bool wave_on = WAVE && G == 2 * p.wave_n;
    if constexpr (FLAT) {
        if (wave_on && tid == 32) {
             
             
             
             
             
             
            uint32_t smid;
            asm volatile("mov.u32 %0, %%smid;\\n" : "=r"(smid));
            const int old = atomicAdd(p.ws_cnt + CNT_SMR + smid, 1);
            if (old & 1) {
                atomicExch(p.ws_cnt + CNT_SMR + smid, 0);
            }
            flag[3] = 2 * static_cast<int>(smid) + (old & 1);
        }
        if (warp == 0) {
            int carry = 0;
            for (int base = 0; base < B; base += 32) {
                const int bb = base + lane;
                const int n = bb < B ? __ldg(p.seq_lens + bb) + L + SEGK : 0;
                int inc = n;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int t = __shfl_up_sync(0xffffffffu, inc, o);
                    if (lane >= o) {
                        inc += t;
                    }
                }
                if (bb < B) {
                    offs[bb] = carry + inc - n;
                }
                carry += __shfl_sync(0xffffffffu, inc, 31);
            }
            if (lane == 0) {
                offs[B] = carry;
            }
        }
        if (tid == 0) {
            flag[1] = B;
            flag[2] = -1;
            if constexpr (PROD) {
#pragma unroll
                for (int s = 0; s < NS; ++s) {
                    mbar_init(full0 + 8 * s, 128);
                    mbar_init(empty0 + 8 * s, NC);
                }
                if constexpr (PBULK) {
#pragma unroll
                    for (int s = 0; s < PB_SLOTS; ++s) {
                        mbar_init(smem_base + S::PBAR + 8 * s, 1);
                        mbar_init(smem_base + S::PFREE + 8 * s, 128);
                    }
                }
                asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
            }
        }
        __syncthreads();
        const long long total_v = offs[B] - SEGK;
        if (wave_on) {
            const int slot = flag[3];
            auto wsum = [&](int k) -> long long {
                return static_cast<long long>(k >> 1) * (ONE_WAVE + 1024)
                       + static_cast<long long>(k & 1) * ONE_WAVE;
            };
            const long long wg = wsum(G);
            v0 = static_cast<int>(total_v * wsum(slot) / wg);
            v1 = static_cast<int>(total_v * wsum(slot + 1) / wg);
        } else {
            v0 = static_cast<int>(total_v * split / G);
            v1 = static_cast<int>(total_v * (split + 1) / G);
        }
        keys_all = offs[B] - B * SEGK;
         
        {
            int first = B;
            int last = -1;
            for (int bb = tid; bb < B; bb += blockDim.x) {
                const int a = offs[bb];
                const int e = offs[bb + 1] - SEGK;
                if (e > v0 && a < v1) {
                    first = min(first, bb);
                    last = max(last, bb);
                }
            }
            first = __reduce_min_sync(0xffffffffu, first);
            last = __reduce_max_sync(0xffffffffu, last);
            if (lane == 0) {
                atomicMin(flag + 1, first);
                atomicMax(flag + 2, last);
            }
        }
        __syncthreads();
        b_first = flag[1];
        b_last = flag[2];
        if constexpr (STAMPS) {
            if (tid == 0) {
                srec(1, snow());
            }
        }
    }

     
    if constexpr (PROD) {
        if (tid >= NC) {
            if constexpr (!SOLO) {
                setmaxnreg_dec<one_prod_regs(L)>();
            }
             
             
             
             
            const int ptid = tid - NC;
            const int ch = ptid & 7;
            const int r0 = ptid >> 3;
            int pst = 0;
            uint32_t pph = 0;
            auto entry = [&](int tile, int r) -> int {
                const int pos = lo + tile * BN + r;
                if (tile >= ntiles || pos >= seq) {
                    return 0;
                }
                if constexpr (DIET) {
                    return __ldg(pt + (pos >> p.lp));
                }
                return __ldg(pt + (psize == 1 ? pos : pos / psize));
            };
            constexpr uint32_t CPB = ONE_PROBE == 1 || ONE_PROBE == 3 ? 0u : 16u;
            const bool fast32 = ONE_PFAST && p.off32 != 0;
            long long s_emw = 0;
            const long long s_p0 = STAMPS ? clock64() : 0;
            if constexpr (PBULK) {
                 
                 
                 
                 
                 
                 
                constexpr int PBD = 3;
                static_assert(PBD < PB_SLOTS, "a window is refilled after its last read");
                const long long pt_total = static_cast<long long>(p.n_splits) * p.pt_stride;
                const bool pt_al = (reinterpret_cast<uintptr_t>(p.page_table) & 15u) == 0;
                uint32_t pbits = 0;
                uint32_t fbits = 0;
                int g = 0;
                for (int bb = b_first; bb <= b_last; ++bb) {
                    if (!seg_bounds(bb)) {
                        continue;
                    }
                    const long long rowbase = static_cast<long long>(bb) * p.pt_stride;
                     
                    auto win = [&](int t, long long& w0, int& bytes) -> bool {
                        const int pos0 = lo + t * BN;
                        const int pend = min(pos0 + BN, seq);
                        if (!pt_al || t >= ntiles || pend <= pos0) {
                            return false;
                        }
                        const long long e0 = rowbase + (pos0 >> p.lp);
                        const long long e1 = rowbase + ((pend - 1) >> p.lp) + 1;
                        w0 = e0 & ~3ll;
                        const long long w1 = (e1 + 3) & ~3ll;
                        bytes = static_cast<int>(w1 - w0) * 4;
                        return w1 <= pt_total;
                    };
                    auto fill = [&](int t, int gt) {
                        long long w0 = 0;
                        int bytes = 0;
                        if (ONE_PBDBG != 2 && ptid == 0 && win(t, w0, bytes)) {
                            const int s = gt % PB_SLOTS;
                            const uint32_t bar = smem_base + S::PBAR + 8 * s;
                            if constexpr (ONE_PBULK == 2) {
                                 
                                 
                                mbar_wait(smem_base + S::PFREE + 8 * s, ((fbits >> s) & 1u) ^ 1u);
                                fbits ^= 1u << s;
                            }
                             
                             
                            fence_proxy_async();
                            mbar_expect_tx(bar, bytes);
                            bulk_g2s(smem_base + S::PSL + s * PB_SLOT_BYTES, p.page_table + w0,
                                     bytes, bar);
                        }
                    };
#pragma unroll 1
                    for (int t = 0; t < PBD; ++t) {
                        fill(t, g + t);
                    }
#pragma unroll 1
                    for (int t = 0; t < ntiles; ++t) {
                        fill(t + PBD, g + t + PBD);
                        const int pos0 = lo + t * BN;
                        int pe[4];
                        long long w0 = 0;
                        int bytes = 0;
                        if (ONE_PBDBG != 2 && win(t, w0, bytes)) {
                            const int s = (g + t) % PB_SLOTS;
                            mbar_wait(smem_base + S::PBAR + 8 * s, (pbits >> s) & 1u);
                            pbits ^= 1u << s;
                            const int* sl =
                                reinterpret_cast<const int*>(smem + S::PSL + s * PB_SLOT_BYTES);
                             
                            const int d = static_cast<int>(rowbase - w0);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int pos = pos0 + r0 + 16 * i;
                                pe[i] = pos < seq ? sl[(pos >> p.lp) + d] : 0;
                            }
                            if constexpr (ONE_PBULK == 2) {
                                mbar_arrive(smem_base + S::PFREE + 8 * s);
                            }
                            if constexpr (ONE_PBDBG == 1) {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    pe[i] = entry(t, r0 + 16 * i);
                                }
                            }
                        } else {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                pe[i] = entry(t, r0 + 16 * i);
                            }
                        }
                        const long long s_w0 = STAMPS ? clock64() : 0;
                        mbar_wait(empty0 + 8 * pst, pph ^ 1u);
                        if constexpr (STAMPS) {
                            s_emw += clock64() - s_w0;
                        }
                        const uint32_t kdst = smem_base + S::KV + pst * 2 * TILE_BYTES;
                        const uint32_t vdst = kdst + TILE_BYTES;
                        if (fast32 && pos0 + BN <= seq) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int r = r0 + 16 * i;
                                const uint32_t row = (static_cast<uint32_t>(pe[i]) << p.lp)
                                                     + (static_cast<uint32_t>(pos0 + r) & (psize - 1));
                                const uint32_t off = (row << 7) + ch * 16;
                                cp_async16(kdst + r * ROWB + k_swz<L>(ch, r) * 16, p.k_cache + off,
                                           CPB);
                                cp_async16(vdst + r * ROWB + v_swz<L>(ch, r) * 16, p.v_cache + off,
                                           CPB);
                            }
                        } else {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int r = r0 + 16 * i;
                                const int pos = pos0 + r;
                                const uint32_t kd = kdst + r * ROWB + k_swz<L>(ch, r) * 16;
                                const uint32_t vd = vdst + r * ROWB + v_swz<L>(ch, r) * 16;
                                if (pos < seq) {
                                    int64_t row = pe[i];
                                    if constexpr (DIET) {
                                        row = (row << p.lp) + (pos & (psize - 1));
                                    } else if (psize != 1) {
                                        row = row * psize + pos % psize;
                                    }
                                    const int64_t off = row * ROWB + ch * 16;
                                    cp_async16(kd, p.k_cache + off, CPB);
                                    cp_async16(vd, p.v_cache + off, CPB);
                                } else if (pos >= hi) {
                                    cp_async16(kd, p.k_cache, 0);
                                    cp_async16(vd, p.v_cache, 0);
                                }
                            }
                        }
                        cp_async_mbar_arrive_noinc(full0 + 8 * pst);
                        if (++pst == NS) {
                            pst = 0;
                            pph ^= 1u;
                        }
                        if constexpr (ONE_PBULK != 2) {
                             
                            named_sync(4, 128);
                        }
                    }
                    g += ntiles;
                }
            } else {
             
             
             
            for (int bb = b_first; bb <= b_last; ++bb) {
                if (!seg_bounds(bb)) {
                    continue;
                }
                int pe[PF][4];
#pragma unroll
                for (int u = 0; u < PF; ++u) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        pe[u][i] = entry(u, r0 + 16 * i);
                    }
                }
                for (int base = 0; base < ntiles; base += PF) {
#pragma unroll
                    for (int u = 0; u < PF; ++u) {
                        const int t = base + u;
                        if (t >= ntiles) {
                            break;
                        }
                        const long long s_w0 = STAMPS ? clock64() : 0;
                        mbar_wait(empty0 + 8 * pst, pph ^ 1u);
                        if constexpr (STAMPS) {
                            s_emw += clock64() - s_w0;
                        }
                        const int pos0 = lo + t * BN;
                        const uint32_t kdst = smem_base + S::KV + pst * 2 * TILE_BYTES;
                        const uint32_t vdst = kdst + TILE_BYTES;
                        if (fast32 && pos0 + BN <= seq) {
                             
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int r = r0 + 16 * i;
                                const uint32_t row = (static_cast<uint32_t>(pe[u][i]) << p.lp)
                                                     + (static_cast<uint32_t>(pos0 + r) & (psize - 1));
                                const uint32_t off = (row << 7) + ch * 16;
                                cp_async16(kdst + r * ROWB + k_swz<L>(ch, r) * 16, p.k_cache + off,
                                           CPB);
                                cp_async16(vdst + r * ROWB + v_swz<L>(ch, r) * 16, p.v_cache + off,
                                           CPB);
                            }
                        } else {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int r = r0 + 16 * i;
                                const int pos = pos0 + r;
                                const uint32_t kd = kdst + r * ROWB + k_swz<L>(ch, r) * 16;
                                const uint32_t vd = vdst + r * ROWB + v_swz<L>(ch, r) * 16;
                                if (pos < seq) {
                                    int64_t row = pe[u][i];
                                    if constexpr (DIET) {
                                        row = (row << p.lp) + (pos & (psize - 1));
                                    } else if (psize != 1) {
                                        row = row * psize + pos % psize;
                                    }
                                    const int64_t off = row * ROWB + ch * 16;
                                    cp_async16(kd, p.k_cache + off, CPB);
                                    cp_async16(vd, p.v_cache + off, CPB);
                                } else if (pos >= hi) {
                                    cp_async16(kd, p.k_cache, 0);
                                    cp_async16(vd, p.v_cache, 0);
                                }
                            }
                        }
                        cp_async_mbar_arrive_noinc(full0 + 8 * pst);
                        const int tn = t + PF;
                        if (PE16 && fast32 && p.lp == 0 && tn < ntiles
                            && lo + tn * BN + BN <= seq) {
                            const int* q = pt + static_cast<uint32_t>(lo + tn * BN + r0);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                pe[u][i] = __ldg(q + 16 * i);
                            }
                        } else if (fast32 && tn < ntiles && lo + tn * BN + BN <= seq) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                pe[u][i] = __ldg(pt + ((lo + tn * BN + r0 + 16 * i) >> p.lp));
                            }
                        } else {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                pe[u][i] = entry(tn, r0 + 16 * i);
                            }
                        }
                        if (++pst == NS) {
                            pst = 0;
                            pph ^= 1u;
                        }
                    }
                }
            }
            }
             
             
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                mbar_wait(empty0 + 8 * pst, pph ^ 1u);
                if (++pst == NS) {
                    pst = 0;
                    pph ^= 1u;
                }
            }
            if constexpr (STAMPS) {
                if (ptid == 0) {
                    srec(11, s_emw);
                    srec(12, clock64() - s_p0);
                }
            }
             
             
             
             
             
             
             
            asm volatile("exit;\\n" ::: "memory");
        }
        if constexpr (!SOLO) {
            setmaxnreg_inc<one_cons_regs(L)>();
        }
    }

     
    auto issue_idx = [&](int tile) {
        if constexpr (DIET) {
             
             
            if (tid < BN && tile < ntiles) {
                const int pos = lo + tile * BN + tid;
                const bool ok = pos < seq;
                cp_async4(smem_base + S::IDX + (tile & (NIDX - 1)) * 256 + tid * 4,
                          pt + (ok ? pos >> p.lp : 0), ok ? 4 : 0);
            }
            return;
        }
        if (tid < BN) {
            const int pos = lo + tile * BN + tid;
            const bool ok = tile < ntiles && pos < seq;
            int e = pos;
            if (psize != 1) {
                e = pos / psize;
            }
            cp_async4(smem_base + S::IDX + (tile & (NIDX - 1)) * 256 + tid * 4, pt + (ok ? e : 0),
                      ok ? 4 : 0);
        }
    };
     
     
     
     
    const int ch = tid & 7;
    const int r0 = tid >> 3;
    constexpr int RSTEP = NC / 8;
    auto issue_kv = [&](int tile, int st, bool patch) {
        const int* idx = idx_ring + (tile & (NIDX - 1)) * BN;
        const int pos0 = lo + tile * BN;
        const uint32_t kdst = smem_base + S::KV + st * 2 * TILE_BYTES;
        const uint32_t vdst = kdst + TILE_BYTES;
        if constexpr (DIET) {
            if (!patch) {
                 
                 
#pragma unroll
                for (int i = 0; i < BN / RSTEP; ++i) {
                    const int r = r0 + i * RSTEP;
                    const int pos = pos0 + r;
                    const bool cached = pos < seq;
                    const int64_t row = cached ? (static_cast<int64_t>(idx[r]) << p.lp)
                                                     + (pos & (psize - 1))
                                               : 0;
                    const int64_t off = row * ROWB + ch * 16;
                    const bool doit = cached || pos >= hi;
                    const uint32_t bytes = cached ? 16u : 0u;
                    cp_async16_if(kdst + r * ROWB + k_swz<L>(ch, r) * 16, p.k_cache + off, bytes,
                                  doit);
                    cp_async16_if(vdst + r * ROWB + v_swz<L>(ch, r) * 16, p.v_cache + off, bytes,
                                  doit);
                }
                return;
            }
            if constexpr ((PRQ & 4) != 0) {
                if (p.off32 && pos0 + BN <= seq) {
#pragma unroll
                    for (int i = 0; i < BN / RSTEP; ++i) {
                        const int r = r0 + i * RSTEP;
                        const uint32_t row = (static_cast<uint32_t>(idx[r]) << p.lp)
                                             + (static_cast<uint32_t>(pos0 + r) & (psize - 1));
                        const uint32_t off = (row << 7) + ch * 16;
                        cp_async16(kdst + r * ROWB + k_swz<L>(ch, r) * 16, p.k_cache + off, 16);
                        cp_async16(vdst + r * ROWB + v_swz<L>(ch, r) * 16, p.v_cache + off, 16);
                    }
                    return;
                }
            }
            if (ONE_KVFAST && pos0 + BN <= seq) {
#pragma unroll
                for (int i = 0; i < BN / RSTEP; ++i) {
                    const int r = r0 + i * RSTEP;
                    const int64_t row = (static_cast<int64_t>(idx[r]) << p.lp)
                                        + ((pos0 + r) & (psize - 1));
                    const int64_t off = row * ROWB + ch * 16;
                    cp_async16(kdst + r * ROWB + k_swz<L>(ch, r) * 16, p.k_cache + off, 16);
                    cp_async16(vdst + r * ROWB + v_swz<L>(ch, r) * 16, p.v_cache + off, 16);
                }
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < BN / RSTEP; ++i) {
            const int r = r0 + i * RSTEP;
            const int pos = pos0 + r;
            const uint32_t kd = kdst + r * ROWB + k_swz<L>(ch, r) * 16;
            const uint32_t vd = vdst + r * ROWB + v_swz<L>(ch, r) * 16;
            if (pos < seq) {
                int64_t row = idx[r];
                if constexpr (DIET) {
                    row = (row << p.lp) + (pos & (psize - 1));
                } else if (psize != 1) {
                    row = row * psize + pos % psize;
                }
                const int64_t off = row * ROWB + ch * 16;
                cp_async16(kd, p.k_cache + off, 16);
                cp_async16(vd, p.v_cache + off, 16);
            } else if (pos < hi) {
                if (patch) {
                    const int j = pos - seq;
                    *reinterpret_cast<uint4*>(smem + (kd - smem_base)) =
                        *reinterpret_cast<const uint4*>(smem + S::KN + j * ROWB + ch * 16);
                    *reinterpret_cast<uint4*>(smem + (vd - smem_base)) =
                        *reinterpret_cast<const uint4*>(smem + S::VN + j * ROWB + ch * 16);
                }
            } else {
                cp_async16(kd, p.k_cache, 0);
                cp_async16(vd, p.v_cache, 0);
            }
        }
    };
     
     
    auto patch_tile = [&](int st, int pos0) {
        if (tid < L * 16) {
            const int j = tid >> 4;
            const int cc = tid & 15;
            const int pos = seq + j;
            if (pos >= max(lo, pos0) && pos < min(hi, pos0 + BN)) {
                const int r = pos - pos0;
                const bool is_v = cc >= 8;
                const int c8 = cc & 7;
                const int sw = is_v ? v_swz<L>(c8, r) : k_swz<L>(c8, r);
                *reinterpret_cast<uint4*>(smem + S::KV + st * 2 * TILE_BYTES
                                          + (is_v ? TILE_BYTES : 0) + r * ROWB + sw * 16) =
                    *reinterpret_cast<const uint4*>(smem + (is_v ? S::VN : S::KN) + j * ROWB
                                                    + c8 * 16);
            }
        }
    };
    auto load8 = [&](const uint8_t* src, float (&x)[8]) {
        const uint4 w = *reinterpret_cast<const uint4*>(src);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ws[e]));
            x[2 * e] = f.x;
            x[2 * e + 1] = f.y;
        }
    };
    float* rope_cos = reinterpret_cast<float*>(smem + S::ROPE);
    float* rope_sin = rope_cos + L * 32;
    auto rope8 = [&](float (&x)[8], int j, int t8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float cs = rope_cos[j * 32 + 4 * t8 + i];
            const float sn = rope_sin[j * 32 + 4 * t8 + i];
            const float a = x[2 * i];
            const float bb = x[2 * i + 1];
            x[2 * i] = a * cs - bb * sn;
            x[2 * i + 1] = a * sn + bb * cs;
        }
    };

    float k_scale = 0.f;
    float v_scale = 0.f;
     
    int cst = 0;
    uint32_t cph = 0;

     
     
    auto run_segment = [&]() -> bool {
        if constexpr (STAMPS) {
            s_seg0 = clock64();
        }
        if constexpr (FLAT) {
             
            cp_wait<0>();
            csync();
        }
        if constexpr (!PROD) {
             
#pragma unroll
            for (int t = 0; t < STAGES - 1; ++t) {
                issue_idx(t);
            }
            cp_commit();
        }
        stamp(1);
        PHASE_EXIT(1)
         
         
         
        asm volatile("griddepcontrol.wait;" ::: "memory");
         
         
        {
            const __nv_bfloat16* qbase = p.qkv + static_cast<size_t>(b) * L * QKV_STRIDE;
            if constexpr (DIET) {
                static_assert(ROWS * 8 % NC == 0, "whole passes of the consumer threads");
#pragma unroll
                for (int k = 0; k < ROWS * 8 / NC; ++k) {
                    const int i = tid + k * NC;
                    const int row = i >> 3;
                    const int cc = i & 7;
                    cp_async16(smem_base + S::QST + row * 128 + cc * 16,
                               qbase + (row >> 4) * QKV_STRIDE + (row & 15) * D + cc * 8, 16);
                }
            } else {
                for (int i = tid; i < ROWS * 8; i += NC) {
                    const int row = i >> 3;
                    const int cc = i & 7;
                    cp_async16(smem_base + S::QST + row * 128 + cc * 16,
                               qbase + (row >> 4) * QKV_STRIDE + (row & 15) * D + cc * 8, 16);
                }
            }
            const bool has_new = hi > seq || SPEC;
            if (has_new && tid < L * 32) {
                const int rr = tid >> 3;
                const int cc = tid & 7;
                cp_async16(smem_base + S::KVST + rr * 128 + cc * 16,
                           qbase + (rr >> 2) * QKV_STRIDE + (HQ + (rr & 3)) * D + cc * 8, 16);
            }
            if (tid < 8) {
                cp_async16(smem_base + S::CONST + C_INVF + tid * 16, p.inv_freq + tid * 4, 16);
            } else if (tid == 8) {
                cp_async4(smem_base + S::CONST + C_KS, p.k_scale, 4);
            } else if (tid == 9) {
                cp_async4(smem_base + S::CONST + C_VS, p.v_scale, 4);
            } else if (tid == 10) {
                cp_async4(smem_base + S::CONST + C_EPOCH, p.ws_cnt + CNT_EPOCH + b,
                          COOP && !FLAT ? 4 : 0);
            } else if (tid >= 16 && tid < 16 + L) {
                cp_async8(smem_base + S::CONST + C_LOC + (tid - 16) * 8,
                          p.cache_loc + static_cast<size_t>(b) * L + (tid - 16), 8);
            }
            if ((FLAT ? lo == 0 : split == 0) && tid < ROWS) {
                prefetch_l2(qbase + (tid >> 4) * QKV_STRIDE + GATE_OFF + (tid & 15) * D);
            }
        }
        cp_commit();
        stamp(2);
        PHASE_EXIT(2)

        if constexpr (!PROD) {
            cp_wait<1>();
            stamp(3);
            csync();
            stamp(4);
            PHASE_EXIT(3)
            if constexpr (SPEC) {
                if (seq_real != seq) {
                    cp_wait<0>();
                    __syncthreads();
                    seq = seq_real;
                    derive();
#pragma unroll
                    for (int t = 0; t < STAGES - 1; ++t) {
                        issue_idx(t);
                    }
                    cp_commit();
                    cp_wait<0>();
                    __syncthreads();
                }
                if (lo >= end) {
                    cp_wait<0>();
                    return false;
                }
            }
            stamp(5);
            PHASE_EXIT(4)
             
             
#pragma unroll
            for (int t = 0; t < STAGES - 1; ++t) {
                if (t < ntiles) {
                    issue_kv(t, t, false);
                    issue_idx(t + STAGES - 1);
                }
                cp_commit();
            }
            stamp(6);
            cp_wait<STAGES - 1>();
        } else {
            cp_wait<0>();
        }
        stamp(7);
        csync();
        stamp(8);
        PHASE_EXIT(5)

        k_scale = *reinterpret_cast<const float*>(smem + S::CONST + C_KS);
        v_scale = *reinterpret_cast<const float*>(smem + S::CONST + C_VS);
        const float* invf = reinterpret_cast<const float*>(smem + S::CONST + C_INVF);
        if (p.rope) {
            if (tid < L * 32) {
                const float2 sc =
                    fast_sincos(static_cast<float>(seq + (tid >> 5)) * invf[tid & 31]);
                rope_sin[tid] = sc.x;
                rope_cos[tid] = sc.y;
            }
            csync();
        }
        stamp(9);
        PHASE_EXIT(6)

         
         
         
        auto q_piece = [&](int piece) {
            const int row = piece >> 3;
            const int t8 = piece & 7;
            const int j = row >> 4;
            const int head = row & 15;
            float x[8];
            load8(smem + S::QST + row * 128 + t8 * 16, x);
            float ss = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                ss += x[e] * x[e];
            }
            const float r = rsqrtf(oct_sum(ss) * (1.f / D) + EPS);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] *= r;
            }
            if (p.rope) {
                rope8(x, j, t8);
            }
            if constexpr (L == 1) {
                uint32_t w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    w[e] = pack_f16(x[2 * e] * p.sl2, x[2 * e + 1] * p.sl2);
                }
                *reinterpret_cast<uint4*>(smem + S::Q + head * 128 + t8 * 16) =
                    make_uint4(w[0], w[1], w[2], w[3]);
            } else {
                const int r64 = j * 8 + (head & 7);
                *reinterpret_cast<uint2*>(smem + S::Q + (head >> 3) * 4096 + kmaj(r64, t8 >> 1)
                                          + (t8 & 1) * 8) =
                    make_uint2(e4m3x4(x[0], x[1], x[2], x[3]), e4m3x4(x[4], x[5], x[6], x[7]));
            }
        };
        if constexpr (DIET) {
#pragma unroll
            for (int k = 0; k < ROWS * 8 / NC; ++k) {
                q_piece(tid + k * NC);
            }
        } else {
            for (int piece = tid; piece < ROWS * 8; piece += NC) {
                q_piece(piece);
            }
        }

        stamp(10);
        PHASE_EXIT(7)
         
         
        if (hi > seq && tid < L * 32) {
            const int rr = tid >> 3;
            const int t8 = tid & 7;
            const int j = rr >> 2;
            const bool is_k = (rr & 3) < 2;
            const int hh = rr & 1;
            float x[8];
            load8(smem + S::KVST + rr * 128 + t8 * 16, x);
            float inv = __fdividef(1.f, v_scale);
             
             
             
            float ss = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                ss += x[e] * x[e];
            }
            ss = oct_sum(ss);
            if (is_k) {
                const float r = rsqrtf(ss * (1.f / D) + EPS);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    x[e] *= r;
                }
                if (p.rope) {
                    rope8(x, j, t8);
                }
                inv = __fdividef(1.f, k_scale);
            }
            const uint2 v8 = make_uint2(e4m3x4(x[0] * inv, x[1] * inv, x[2] * inv, x[3] * inv),
                                        e4m3x4(x[4] * inv, x[5] * inv, x[6] * inv, x[7] * inv));
            const int pos = seq + j;
            if (pos >= lo && pos < hi) {
                const int64_t loc =
                    *reinterpret_cast<const int64_t*>(smem + S::CONST + C_LOC + j * 8);
                uint8_t* dst = (is_k ? p.k_cache : p.v_cache) + loc * ROWB + hh * D + t8 * 8;
                *reinterpret_cast<uint2*>(dst) = v8;
            }
            *reinterpret_cast<uint2*>(smem + (is_k ? S::KN : S::VN) + j * ROWB + hh * D + t8 * 8) =
                v8;
        }
        stamp(11);
        csync();
        stamp(12);
        PHASE_EXIT(8)
        if constexpr (!PROD) {
             
            if (hi > seq) {
#pragma unroll
                for (int t = 0; t < STAGES - 1; ++t) {
                    if (t < ntiles) {
                        patch_tile(t, lo + t * BN);
                    }
                }
            }
        }
        if constexpr (L == 8) {
             
             
            if (tid < 128) {
                *reinterpret_cast<uint4*>(smem + S::VT + (tid >> 5) * VT_STRIDE + 64 * 64
                                          + (tid & 31) * 16) =
                    make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
            }
        }

        stamp(13);
        PHASE_EXIT(9)
        const float sl2k = (L == 1 ? 1.f : p.sl2) * k_scale;
        const int copy = p.nc > 1 ? split % p.nc : 0;

         
         
         
         
        auto acquire = [&](int it, int st, bool common = false) {
            if constexpr (PROD) {
                const long long s_w0 = STAMPS ? clock64() : 0;
                mbar_wait(full0 + 8 * st, cph);
                if constexpr (STAMPS) {
                    const long long now = clock64();
                    s_fullw += now - s_w0;
                    if (it == 0) {
                        s_pro += now - s_seg0;
                        s_l0 = now;
                        if (s_nseg == 0 && tid == 0) {
                            srec(2, snow());
                        }
                    }
                }
                if constexpr (L == 8 && ONE_PROBE != 2) {
                    fence_proxy_async();
                }
                const int pos0 = lo + it * BN;
                if (common) {
                    if (it == 0) {
                        csync();
                    }
                } else if (hi > seq && pos0 + BN > seq) {
                    patch_tile(st, pos0);
                    if constexpr (L == 8) {
                        fence_proxy_async();
                    }
                    csync();
                } else if (it == 0) {
                    csync();
                }
            } else {
                cp_wait<STAGES - 2>();
                if constexpr (L == 8) {
                    fence_proxy_async();
                }
                if (it == 0) {
                    stamp(14);
                }
                __syncthreads();
                if (it == 0) {
                    stamp(15);
                }
                if constexpr (!ONE) {
                    if (it + STAGES - 1 < ntiles) {
                        issue_kv(it + STAGES - 1, (st + STAGES - 1) % STAGES, true);
                        issue_idx(it + 2 * STAGES - 2);
                    }
                    cp_commit();
                }
            }
        };
        auto release = [&](int st) {
            if constexpr (PROD) {
                mbar_arrive(empty0 + 8 * st);
                if (++cst == NS) {
                    cst = 0;
                    cph ^= 1u;
                }
            }
        };

        if constexpr (L == 1) {
             
             
            const int hk = warp & 1;
            const int wsub = warp >> 1;
            uint32_t qf[8];
            {
                const uint4* qp =
                    reinterpret_cast<const uint4*>(smem + S::Q + (hk * 8 + g) * 128 + c * 32);
                const uint4 a = qp[0];
                const uint4 bq = qp[1];
                qf[0] = a.x;
                qf[1] = a.y;
                qf[2] = a.z;
                qf[3] = a.w;
                qf[4] = bq.x;
                qf[5] = bq.y;
                qf[6] = bq.z;
                qf[7] = bq.w;
            }
            float o[8][4];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                o[e][0] = o[e][1] = o[e][2] = o[e][3] = 0.f;
            }
            float m = NEG_INF;
            float l = 0.f;
            int st = PROD ? cst : 0;
            for (int it = 0; it < ntiles; ++it) {
                acquire(it, st);
                if constexpr (ONE_PROBE == 3 && PROD) {
                    release(st);
                    st = cst;
                    continue;
                }
                const uint8_t* ks = smem + S::KV + st * 2 * TILE_BYTES;
                const uint8_t* vs = ks + TILE_BYTES;
                const int pos0 = lo + it * BN;
                const bool interior =
                    pos0 + BN - 1 <= seq && (!p.window || pos0 >= seq + L - p.window);
                float s[4][4];
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
                    const int key = wsub * 32 + kb * 8 + g;
                    const uint4 kw = *reinterpret_cast<const uint4*>(
                        ks + key * ROWB + k_swz<1>(hk * 4 + c, key) * 16);
                    const uint32_t kwv[4] = {kw.x, kw.y, kw.z, kw.w};
                    uint32_t kf[8];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        kf[2 * e] = f16x2_from_e4m3x2(kwv[e] & 0xffffu);
                        kf[2 * e + 1] = f16x2_from_e4m3x2(kwv[e] >> 16);
                    }
                    s[kb][0] = s[kb][1] = s[kb][2] = s[kb][3] = 0.f;
                    if constexpr (QK2) {
                        float so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                        for (int j2 = 0; j2 < 4; j2 += 2) {
                            mma_f16(s[kb], qf[2 * j2], 0u, qf[2 * j2 + 1], 0u, kf[2 * j2],
                                    kf[2 * j2 + 1]);
                            mma_f16(so, qf[2 * j2 + 2], 0u, qf[2 * j2 + 3], 0u, kf[2 * j2 + 2],
                                    kf[2 * j2 + 3]);
                        }
                        s[kb][0] += so[0];
                        s[kb][1] += so[1];
                    } else {
#pragma unroll
                        for (int j2 = 0; j2 < 4; ++j2) {
                            mma_f16(s[kb], qf[2 * j2], 0u, qf[2 * j2 + 1], 0u, kf[2 * j2],
                                    kf[2 * j2 + 1]);
                        }
                    }
                }
                float mx = NEG_INF;
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        if (!interior) {
                            const int pos = pos0 + wsub * 32 + kb * 8 + 2 * c + e;
                            if (pos > seq || (p.window && pos < seq - p.window + 1)) {
                                s[kb][e] = NEG_INF;
                            }
                        }
                        mx = fmaxf(mx, s[kb][e]);
                    }
                }
                mx = quad_max(mx) * sl2k;
                const float mu = fmaxf(ONE ? mx : fmaxf(m, mx), OFF_FLOOR);
                float alpha = 1.f;
                if constexpr (!ONE) {
                    alpha = ex2(m - mu);
                }
                m = mu;
                float sum = 0.f;
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        s[kb][e] = ex2(fmaf(s[kb][e], sl2k, -mu));
                        sum += s[kb][e];
                    }
                }
                l = l * alpha + sum;
                if constexpr (!ONE) {
                    if (__any_sync(0xffffffffu, alpha != 1.f)) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            o[e][0] *= alpha;
                            o[e][1] *= alpha;
                        }
                    }
                }
                 
                 
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const uint32_t a0 = pack_f16(s[2 * kk][0], s[2 * kk][1]);
                    const uint32_t a2 = pack_f16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                    uint2 vt[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int key = wsub * 32 + kk * 16 + 2 * c + (k & 1) + (k >> 1) * 8;
                        vt[k] = *reinterpret_cast<const uint2*>(
                            vs + key * ROWB + v_swz<1>(hk * 4 + (g >> 1), key) * 16 + (g & 1) * 8);
                    }
#pragma unroll
                    for (int hw = 0; hw < 2; ++hw) {
                        const uint32_t x0 = hw ? vt[0].y : vt[0].x;
                        const uint32_t x1 = hw ? vt[1].y : vt[1].x;
                        const uint32_t x2 = hw ? vt[2].y : vt[2].x;
                        const uint32_t x3 = hw ? vt[3].y : vt[3].x;
#pragma unroll
                        for (int pr = 0; pr < 2; ++pr) {
                            const uint32_t sel = pr ? 0x7362u : 0x5140u;
                            const uint32_t w01 = __byte_perm(x0, x1, sel);
                            const uint32_t w23 = __byte_perm(x2, x3, sel);
#pragma unroll
                            for (int q = 0; q < 2; ++q) {
                                const int e = hw * 4 + pr * 2 + q;
                                const uint32_t b0 =
                                    f16x2_from_e4m3x2(q ? w01 >> 16 : w01 & 0xffffu);
                                const uint32_t b1 =
                                    f16x2_from_e4m3x2(q ? w23 >> 16 : w23 & 0xffffu);
                                mma_f16(o[e], a0, 0u, a2, 0u, b0, b1);
                            }
                        }
                    }
                }
                release(st);
                st = PROD ? cst : (st == STAGES - 1 ? 0 : st + 1);
            }
            if constexpr (STAMPS) {
                s_loop += clock64() - s_l0;
                s_tiles += ntiles;
                ++s_nseg;
                if (tid == 0) {
                    srec(3, snow());
                    srec(13, clock64() - s_k0);
                }
            }
            stamp(16);
            PHASE_EXIT(10)
             
             
             
            const float f = ex2(fminf(m, 126.f));
#if ONE_DETM
            if constexpr (DETM) {
                float* slot = reinterpret_cast<float*>(p.dbg)
                              + det_slot(b, G, split, wsub, hk * 8 + g);
                st_cg_f4(slot + c * 16, make_float4(o[0][0] * f, o[1][0] * f, o[2][0] * f,
                                                    o[3][0] * f));
                st_cg_f4(slot + c * 16 + 4, make_float4(o[4][0] * f, o[5][0] * f, o[6][0] * f,
                                                        o[7][0] * f));
                st_cg_f4(slot + c * 16 + 8, make_float4(o[0][1] * f, o[1][1] * f, o[2][1] * f,
                                                        o[3][1] * f));
                st_cg_f4(slot + c * 16 + 12, make_float4(o[4][1] * f, o[5][1] * f, o[6][1] * f,
                                                         o[7][1] * f));
                const float lq = quad_sum(l);
                if (c == 0) {
                    __stcg(slot + 64, lq * f);
                }
            } else
#endif
            if constexpr (HMRG) {
                 
                 
                 
                 
                 
                const float lq = quad_sum(l) * f;
                cp_wait<0>();
                csync();
                static_assert(!FLAT || S::Q + 17 * 64 * 4 <= S::KN, "flat handoff bytes");
                float* xs = reinterpret_cast<float*>(smem + (FLAT ? S::Q : S::KV)) + hk * 32
                            + lane;
                if (wsub == 1) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        xs[(2 * e) * 64] = o[e][0] * f;
                        xs[(2 * e + 1) * 64] = o[e][1] * f;
                    }
                    xs[16 * 64] = lq;
                }
                csync();
                if (wsub == 0) {
                    float a[16];
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        a[2 * e] = o[e][0] * f + xs[(2 * e) * 64];
                        a[2 * e + 1] = o[e][1] * f + xs[(2 * e + 1) * 64];
                    }
                    const float la = lq + xs[16 * 64];
                    const int row = ws_row(b, p.nc, copy, 1, 0, hk * 8 + g);
                    float* dst = p.ws_acc + static_cast<size_t>(row) * D + c * 16;
                    red_add_v4(dst, a[0], a[2], a[4], a[6]);
                    red_add_v4(dst + 4, a[8], a[10], a[12], a[14]);
                    red_add_v4(dst + 8, a[1], a[3], a[5], a[7]);
                    red_add_v4(dst + 12, a[9], a[11], a[13], a[15]);
                    if (c == 0) {
                        red_add_f32(p.ws_l + row, la);
                    }
                }
            } else {
                const int row = ws_row(b, p.nc, copy, 1, 0, hk * 8 + g);
                float* dst = p.ws_acc + static_cast<size_t>(row) * D + c * 16;
                red_add_v4(dst, o[0][0] * f, o[1][0] * f, o[2][0] * f, o[3][0] * f);
                red_add_v4(dst + 4, o[4][0] * f, o[5][0] * f, o[6][0] * f, o[7][0] * f);
                red_add_v4(dst + 8, o[0][1] * f, o[1][1] * f, o[2][1] * f, o[3][1] * f);
                red_add_v4(dst + 12, o[4][1] * f, o[5][1] * f, o[6][1] * f, o[7][1] * f);
                const float lq = quad_sum(l);
                if (c == 0) {
                    red_add_f32(p.ws_l + row, lq * f);
                }
            }
        } else {
             
             
             
            const int hk = warp >> 2;
            const int wq = warp & 3;
            uint64_t qd0 = desc_kmaj(smem_base + S::Q + hk * 4096, 128, 512);
            uint64_t qd1 = desc_kmaj(smem_base + S::Q + hk * 4096 + 256, 128, 512);
            uint32_t klo0 = 0;
            int it_c = 0;
            if constexpr (DCARRY) {
                qd0 = (qd0 & ~0xffffffffull)
                      | __shfl_sync(0xffffffffu, static_cast<uint32_t>(qd0), 0);
                qd1 = (qd1 & ~0xffffffffull)
                      | __shfl_sync(0xffffffffu, static_cast<uint32_t>(qd1), 0);
                klo0 = __shfl_sync(0xffffffffu,
                                   static_cast<uint32_t>(desc_sw128(smem_base + S::KV + hk * 64)),
                                   0);
                 
                 
                it_c = __shfl_sync(0xffffffffu, !p.window && seq > lo ? (seq - lo) >> 6 : 0, 0);
            }
            uint32_t vt_cur = 0;
            uint32_t vt_x = 0;
            uint32_t vd_lo = 0;
            uint32_t vd_x = 0;
            uint32_t ld_off = 0;
            uint32_t st_off = 0;
            if constexpr (VHOIST) {
                const uint32_t vt0 = smem_base + S::VT + hk * VT_STRIDE;
                const uint32_t vt1 = smem_base + S::VT + (2 + hk) * VT_STRIDE;
                vt_cur = __shfl_sync(0xffffffffu, vt0, 0);
                vt_x = __shfl_sync(0xffffffffu, vt0 ^ vt1, 0);
                vd_lo = __shfl_sync(0xffffffffu, static_cast<uint32_t>(desc_sw64(vt0)), 0);
                vd_x = vd_lo ^ __shfl_sync(0xffffffffu, static_cast<uint32_t>(desc_sw64(vt1)), 0);
                const int h4 = lane >> 3;
                const int r8 = lane & 7;
                ld_off = smem_base + S::KV + TILE_BYTES + (h4 * 8 + r8) * ROWB
                         + (((hk * 4 + wq) ^ r8) << 4);
                const int n = 8 * (2 * wq + (h4 & 1)) + r8;
                st_off = n * 64 + (((h4 >> 1) ^ ((n >> 1) & 3)) << 4);
                 
                 
                asm volatile("" : "+r"(ld_off), "+r"(st_off));
            }
            float o[36];
#pragma unroll
            for (int i = 0; i < 36; ++i) {
                o[i] = 0.f;
            }
            float off[2] = {OFF_FLOOR, OFF_FLOOR};
            const int qp0 = seq + 2 * wq;
            if constexpr (XPIPE != 0) {
                 
                 
                 
                 
                 
                 
                float s[32];
                uint32_t pa[2][4];
                int st = cst;
                auto qk_issue = [&]() {
                    const uint64_t kd =
                        (desc_sw128(0) & ~0xffffffffull)
                        | (klo0 + static_cast<uint32_t>(st) * (2 * TILE_BYTES >> 4));
                    wg_fence();
                    wgmma_qk<0>(s, qd0, kd);
                    wgmma_qk<1>(s, qd1, kd + 2);
                    wg_commit();
                };
                 
                auto transpose_v = [&](uint32_t vt) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        uint32_t t0, t1, t2, t3;
                        ldsm_x4_trans(ld_off + static_cast<uint32_t>(st) * (2 * TILE_BYTES)
                                          + half * 32 * ROWB,
                                      t0, t1, t2, t3);
                        stsm_x4(vt + (st_off ^ (half ? 32u : 0u)), __byte_perm(t0, t1, 0x6420),
                                __byte_perm(t0, t1, 0x7531), __byte_perm(t2, t3, 0x6420),
                                __byte_perm(t2, t3, 0x7531));
                    }
                };
                auto pv_issue = [&]() {
                    fence_proxy_async();
                    named_sync(2 + hk, 128);
                    const uint64_t vd = (desc_sw64(0) & ~0xffffffffull) | vd_lo;
                    wg_fence();
                    wgmma_pv(o, pa[0], vd);
                    wgmma_pv(o, pa[1], vd + 2);
                    wg_commit();
                };
                 
                 
                 
                auto offset_to_max = [&]() {
                    if constexpr (XPIPE == 1) {
                        wg_wait<0>();
                        reg_fence(o);
                    }
                    float tmax[2];
                    bool move[2];
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        float mx = fmaxf(s[2 * hh], s[2 * hh + 1]);
#pragma unroll
                        for (int i = 1; i < 8; ++i) {
                            mx = fmaxf(mx, fmaxf(s[4 * i + 2 * hh], s[4 * i + 2 * hh + 1]));
                        }
                        tmax[hh] = quad_max(mx) * sl2k;
                        move[hh] = tmax[hh] > off[hh] + 8.f;
                    }
                    if (__any_sync(0xffffffffu, move[0] || move[1])) {
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            const float nxt = move[hh] ? tmax[hh] : off[hh];
                            const float scale = ex2(off[hh] - nxt);
                            off[hh] = nxt;
#pragma unroll
                            for (int i = 0; i < 9; ++i) {
                                o[4 * i + 2 * hh] *= scale;
                                o[4 * i + 2 * hh + 1] *= scale;
                            }
                        }
                    }
                };
                auto pack = [&]() {
                    auto ex = [&](int j) { return ex2(fmaf(s[j], sl2k, -off[(j >> 1) & 1])); };
#pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        const int b0 = 16 * kk;
                        pa[kk][0] = e4m3x4(ex(b0), ex(b0 + 1), ex(b0 + 4), ex(b0 + 5));
                        pa[kk][1] = e4m3x4(ex(b0 + 2), ex(b0 + 3), ex(b0 + 6), ex(b0 + 7));
                        pa[kk][2] = e4m3x4(ex(b0 + 8), ex(b0 + 9), ex(b0 + 12), ex(b0 + 13));
                        pa[kk][3] = e4m3x4(ex(b0 + 10), ex(b0 + 11), ex(b0 + 14), ex(b0 + 15));
                    }
                };
                 
                 
                auto softmax_tile = [&](int it) {
                    const int pos0 = lo + it * BN;
                    const bool interior =
                        it < it_c
                        || (pos0 + BN - 1 <= seq && (!p.window || pos0 >= seq + L - p.window));
                    if (!interior) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                                for (int e = 0; e < 2; ++e) {
                                    const int pos = pos0 + 8 * i + 2 * c + e;
                                    const int qp = qp0 + hh;
                                    if (pos > qp || (p.window && pos < qp - p.window + 1)) {
                                        s[4 * i + 2 * hh + e] = NEG_INF;
                                    }
                                }
                            }
                        }
                    }
                    if (it == 0) {
                        offset_to_max();
                    }
                    pack();
                    if (it != 0) {
                         
                        constexpr uint32_t PB = 0x07070707u;
                        const uint32_t ov =
                            ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) |
                             (pa[0][3] + PB) | (pa[1][0] + PB) | (pa[1][1] + PB) |
                             (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
                        const bool any_ov = XPIPE == 1 ? named_sync_or(2 + hk, 128, ov != 0u)
                                                       : __any_sync(0xffffffffu, ov != 0u);
                        if (any_ov) {
                            offset_to_max();
                            pack();
                        }
                    }
                };
                 
                 
                auto next_qk = [&](int it) {
                    acquire(it, st, it < it_c);
                    qk_issue();
                };
                acquire(0, st, 0 < it_c);
                qk_issue();
                transpose_v(vt_cur);
                wg_wait<0>();
                reg_fence(s);
                release(st);
                st = cst;
#pragma unroll (UNR8)
                for (int it = 0; it < ntiles - 1; ++it) {
                    softmax_tile(it);
                    if constexpr (XPIPE == 1) {
                        next_qk(it + 1);
                         
                         
                        wg_wait<1>();
                        reg_fence(o);
                        pv_issue();
                        transpose_v(vt_cur ^ vt_x);
                        wg_wait<1>();
                    } else {
                        pv_issue();
                        next_qk(it + 1);
                        transpose_v(vt_cur ^ vt_x);
                        wg_wait<0>();
                        reg_fence(o);
                    }
                    reg_fence(s);
                    release(st);
                    st = cst;
                    vt_cur ^= vt_x;
                    vd_lo ^= vd_x;
                }
                softmax_tile(ntiles - 1);
                if constexpr (XPIPE == 1) {
                    wg_wait<0>();
                    reg_fence(o);
                }
                pv_issue();
                wg_wait<0>();
                reg_fence(o);
                vt_cur ^= vt_x;
                vd_lo ^= vd_x;
            } else if constexpr (QKOV || EREL || PCHK || UNR8 > 1) {
                int st = PROD ? cst : 0;
#pragma unroll (UNR8)
                for (int it = 0; it < ntiles; ++it) {
                    const bool common = DCARRY && it < it_c;
                    acquire(it, st, common);
                    const uint8_t* vs = smem + S::KV + st * 2 * TILE_BYTES + TILE_BYTES;
                    const int pos0 = lo + it * BN;
                    const bool interior =
                        common
                        || (pos0 + BN - 1 <= seq && (!p.window || pos0 >= seq + L - p.window));
                     
                     
                     
                     
                     
                    auto vt_of = [&]() { return smem + S::VT + (2 * (it & 1) + hk) * VT_STRIDE; };
                    auto transpose_v = [&]() {
                        if constexpr (VHOIST) {
                             
                             
#pragma unroll
                            for (int half = 0; half < 2; ++half) {
                                uint32_t t0, t1, t2, t3;
                                ldsm_x4_trans(ld_off + static_cast<uint32_t>(st) * (2 * TILE_BYTES)
                                                  + half * 32 * ROWB,
                                              t0, t1, t2, t3);
                                stsm_x4(vt_cur + (st_off ^ (half ? 32u : 0u)),
                                        __byte_perm(t0, t1, 0x6420), __byte_perm(t0, t1, 0x7531),
                                        __byte_perm(t2, t3, 0x6420), __byte_perm(t2, t3, 0x7531));
                            }
                            return;
                        }
                        uint8_t* vt = vt_of();
                        const int h4 = lane >> 3;
                        const int r8 = lane & 7;
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int key = half * 32 + h4 * 8 + r8;
                            uint32_t t0, t1, t2, t3;
                            ldsm_x4_trans(smem_u32(vs + key * ROWB + ((hk * 4 + wq) ^ r8) * 16), t0,
                                          t1, t2, t3);
                            const uint32_t m0 = __byte_perm(t0, t1, 0x6420);
                            const uint32_t m1 = __byte_perm(t0, t1, 0x7531);
                            const uint32_t m2 = __byte_perm(t2, t3, 0x6420);
                            const uint32_t m3 = __byte_perm(t2, t3, 0x7531);
                            const int n = 8 * (2 * wq + (h4 & 1)) + r8;
                            const int chunk = 2 * half + (h4 >> 1);
                            stsm_x4(smem_u32(vt + n * 64 + ((chunk ^ ((n >> 1) & 3)) << 4)), m0, m1,
                                    m2, m3);
                        }
                    };
                    float s[32];
                    {
                        const uint64_t kd =
                            DCARRY ? (desc_sw128(0) & ~0xffffffffull)
                                         | (klo0 + static_cast<uint32_t>(st) * (2 * TILE_BYTES >> 4))
                                   : desc_sw128(smem_base + S::KV + st * 2 * TILE_BYTES + hk * 64);
                        wg_fence();
                        wgmma_qk<0>(s, qd0, kd);
                        wgmma_qk<1>(s, qd1, kd + 2);
                        wg_commit();
                        if constexpr (QKOV) {
                             
                             
                             
                            transpose_v();
                        }
                        wg_wait<0>();
                        reg_fence(s);
                    }
                    if constexpr (EREL) {
                         
                         
                        release(st);
                    }
                    if (!interior) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                                for (int e = 0; e < 2; ++e) {
                                    const int pos = pos0 + 8 * i + 2 * c + e;
                                    const int qp = qp0 + hh;
                                    if (pos > qp || (p.window && pos < qp - p.window + 1)) {
                                        s[4 * i + 2 * hh + e] = NEG_INF;
                                    }
                                }
                            }
                        }
                    }
                     
                     
                    auto offset_to_max = [&]() {
                        float tmax[2];
                        bool move[2];
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            float mx = fmaxf(s[2 * hh], s[2 * hh + 1]);
#pragma unroll
                            for (int i = 1; i < 8; ++i) {
                                mx = fmaxf(mx, fmaxf(s[4 * i + 2 * hh], s[4 * i + 2 * hh + 1]));
                            }
                            tmax[hh] = quad_max(mx) * sl2k;
                            move[hh] = tmax[hh] > off[hh] + 8.f;
                        }
                        if (__any_sync(0xffffffffu, move[0] || move[1])) {
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
                                const float nxt = move[hh] ? tmax[hh] : off[hh];
                                const float scale = ex2(off[hh] - nxt);
                                off[hh] = nxt;
#pragma unroll
                                for (int i = 0; i < 9; ++i) {
                                    o[4 * i + 2 * hh] *= scale;
                                    o[4 * i + 2 * hh + 1] *= scale;
                                }
                            }
                        }
                    };
                     
                     
                     
                     
                    uint32_t pa[2][4];
                    auto pack_exps = [&]() {
#pragma unroll
                        for (int kk = 0; kk < 2; ++kk) {
                            const float* q0 = s + 16 * kk;
                            pa[kk][0] = e4m3x4(q0[0], q0[1], q0[4], q0[5]);
                            pa[kk][1] = e4m3x4(q0[2], q0[3], q0[6], q0[7]);
                            pa[kk][2] = e4m3x4(q0[8], q0[9], q0[12], q0[13]);
                            pa[kk][3] = e4m3x4(q0[10], q0[11], q0[14], q0[15]);
                        }
                    };
                    auto pack = [&]() {
                        auto ex = [&](int j) { return ex2(fmaf(s[j], sl2k, -off[(j >> 1) & 1])); };
#pragma unroll
                        for (int kk = 0; kk < 2; ++kk) {
                            const int b0 = 16 * kk;
                            pa[kk][0] = e4m3x4(ex(b0), ex(b0 + 1), ex(b0 + 4), ex(b0 + 5));
                            pa[kk][1] = e4m3x4(ex(b0 + 2), ex(b0 + 3), ex(b0 + 6), ex(b0 + 7));
                            pa[kk][2] = e4m3x4(ex(b0 + 8), ex(b0 + 9), ex(b0 + 12), ex(b0 + 13));
                            pa[kk][3] = e4m3x4(ex(b0 + 10), ex(b0 + 11), ex(b0 + 14), ex(b0 + 15));
                        }
                    };
                    if constexpr (!PCHK) {
                        offset_to_max();
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                                for (int e = 0; e < 2; ++e) {
                                    s[4 * i + 2 * hh + e] =
                                        ex2(fmaf(s[4 * i + 2 * hh + e], sl2k, -off[hh]));
                                }
                            }
                        }
                        pack_exps();
                    } else {
                        if (it == 0) {
                            offset_to_max();
                        }
                        pack();
                        if (it != 0) {
                             
                             
                             
                            constexpr uint32_t PB = 0x07070707u;
                            const uint32_t ov =
                                ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) |
                                 (pa[0][3] + PB) | (pa[1][0] + PB) | (pa[1][1] + PB) |
                                 (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
                            if (__any_sync(0xffffffffu, ov != 0u)) {
                                offset_to_max();
                                pack();
                            }
                        }
                    }
                    if constexpr (!QKOV) {
                        transpose_v();
                    }
                    fence_proxy_async();
                    named_sync(2 + hk, 128);
                    {
                        const uint64_t vd = VHOIST ? (desc_sw64(0) & ~0xffffffffull) | vd_lo
                                                   : desc_sw64(smem_u32(vt_of()));
                        wg_fence();
                        wgmma_pv(o, pa[0], vd);
                        wgmma_pv(o, pa[1], vd + 2);
                        wg_commit();
                        wg_wait<0>();
                        reg_fence(o);
                    }
                    if constexpr (!EREL) {
                        release(st);
                    }
                    if constexpr (VHOIST) {
                        vt_cur ^= vt_x;
                        vd_lo ^= vd_x;
                    }
                    st = PROD ? cst : (st == STAGES - 1 ? 0 : st + 1);
                }
            } else if constexpr (PRQ != 0) {
                constexpr bool TOV = (PRQ & 1) != 0;
                constexpr bool QKF = (PRQ & 2) != 0;
                constexpr bool SPV = (PRQ & 8) != 0;
                constexpr bool TCHK = (PRQ & 16) != 0;
                int st = 0;
                for (int it = 0; it < ntiles; ++it) {
                     
                     
                    cp_wait<STAGES - 2>();
                    fence_proxy_async();
                    if (it == 0) {
                        stamp(14);
                    }
                    __syncthreads();
                    if (it == 0) {
                        stamp(15);
                    }
                    auto issue_next = [&]() {
                        if constexpr (!ONE) {
                            if (it + STAGES - 1 < ntiles) {
                                issue_kv(it + STAGES - 1, (st + STAGES - 1) % STAGES, true);
                                issue_idx(it + 2 * STAGES - 2);
                            }
                            cp_commit();
                        }
                    };
                    if constexpr (!QKF) {
                        issue_next();
                    }
                    const uint8_t* vs = smem + S::KV + st * 2 * TILE_BYTES + TILE_BYTES;
                    const int pos0 = lo + it * BN;
                    const bool interior =
                        pos0 + BN - 1 <= seq && (!p.window || pos0 >= seq + L - p.window);
                    float s[32];
                    {
                        const uint64_t kd =
                            desc_sw128(smem_base + S::KV + st * 2 * TILE_BYTES + hk * 64);
                        wg_fence();
                        wgmma_qk<0>(s, qd0, kd);
                        wgmma_qk<1>(s, qd1, kd + 2);
                        wg_commit();
                    }
                    if constexpr (QKF) {
                        issue_next();
                    }
                    uint8_t* vt = smem + S::VT + (2 * (it & 1) + hk) * VT_STRIDE;
                    auto transpose = [&]() {
                        const int h4 = lane >> 3;
                        const int r8 = lane & 7;
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int key = half * 32 + h4 * 8 + r8;
                            uint32_t t0, t1, t2, t3;
                            ldsm_x4_trans(smem_u32(vs + key * ROWB + ((hk * 4 + wq) ^ r8) * 16), t0,
                                          t1, t2, t3);
                            const uint32_t m0 = __byte_perm(t0, t1, 0x6420);
                            const uint32_t m1 = __byte_perm(t0, t1, 0x7531);
                            const uint32_t m2 = __byte_perm(t2, t3, 0x6420);
                            const uint32_t m3 = __byte_perm(t2, t3, 0x7531);
                            const int n = 8 * (2 * wq + (h4 & 1)) + r8;
                            const int chunk = 2 * half + (h4 >> 1);
                            stsm_x4(smem_u32(vt + n * 64 + ((chunk ^ ((n >> 1) & 3)) << 4)), m0, m1,
                                    m2, m3);
                        }
                    };
                    if constexpr (TOV) {
                        transpose();
                    }
                    wg_wait<0>();
                    reg_fence(s);
                    if (!interior) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                                for (int e = 0; e < 2; ++e) {
                                    const int pos = pos0 + 8 * i + 2 * c + e;
                                    const int qp = qp0 + hh;
                                    if (pos > qp || (p.window && pos < qp - p.window + 1)) {
                                        s[4 * i + 2 * hh + e] = NEG_INF;
                                    }
                                }
                            }
                        }
                    }
                    float mxr[2];
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        float mx = fmaxf(s[2 * hh], s[2 * hh + 1]);
#pragma unroll
                        for (int i = 1; i < 8; ++i) {
                            mx = fmaxf(mx, fmaxf(s[4 * i + 2 * hh], s[4 * i + 2 * hh + 1]));
                        }
                        mxr[hh] = mx;
                    }
                     
                     
                     
                    bool any_move = false;
                    if constexpr (TCHK) {
                        any_move = __any_sync(0xffffffffu, mxr[0] * sl2k > off[0] + 8.f
                                                               || mxr[1] * sl2k > off[1] + 8.f);
                    }
                    float tmax[2];
                    bool move[2];
                    auto row_moves = [&]() {
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            tmax[hh] = quad_max(mxr[hh]) * sl2k;
                            move[hh] = tmax[hh] > off[hh] + 8.f;
                        }
                    };
                    if constexpr (!TCHK) {
                        row_moves();
                        any_move = __any_sync(0xffffffffu, move[0] || move[1]);
                    }
                    if (any_move) {
                        if constexpr (TCHK) {
                            row_moves();
                        }
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            const float nxt = move[hh] ? tmax[hh] : off[hh];
                            const float scale = ex2(off[hh] - nxt);
                            off[hh] = nxt;
#pragma unroll
                            for (int i = 0; i < 9; ++i) {
                                o[4 * i + 2 * hh] *= scale;
                                o[4 * i + 2 * hh + 1] *= scale;
                            }
                        }
                    }
                    uint32_t pa[2][4];
                    auto pack_half = [&](int kk) {
#pragma unroll
                        for (int i = 4 * kk; i < 4 * kk + 4; ++i) {
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                                for (int e = 0; e < 2; ++e) {
                                    s[4 * i + 2 * hh + e] =
                                        ex2(fmaf(s[4 * i + 2 * hh + e], sl2k, -off[hh]));
                                }
                            }
                        }
                        const float* q0 = s + 16 * kk;
                        pa[kk][0] = e4m3x4(q0[0], q0[1], q0[4], q0[5]);
                        pa[kk][1] = e4m3x4(q0[2], q0[3], q0[6], q0[7]);
                        pa[kk][2] = e4m3x4(q0[8], q0[9], q0[12], q0[13]);
                        pa[kk][3] = e4m3x4(q0[10], q0[11], q0[14], q0[15]);
                    };
                    const uint64_t vd = desc_sw64(smem_u32(vt));
                    if constexpr (SPV) {
                        pack_half(0);
                        if constexpr (!TOV) {
                            transpose();
                        }
                        fence_proxy_async();
                        named_sync(2 + hk, 128);
                        wg_fence();
                        wgmma_pv(o, pa[0], vd);
                        wg_commit();
                        pack_half(1);
                        wg_fence();
                        wgmma_pv(o, pa[1], vd + 2);
                        wg_commit();
                    } else {
                        pack_half(0);
                        pack_half(1);
                        if constexpr (!TOV) {
                            transpose();
                        }
                        fence_proxy_async();
                        named_sync(2 + hk, 128);
                        wg_fence();
                        wgmma_pv(o, pa[0], vd);
                        wgmma_pv(o, pa[1], vd + 2);
                        wg_commit();
                    }
                    wg_wait<0>();
                    reg_fence(o);
                    st = st == STAGES - 1 ? 0 : st + 1;
                }
            } else {
                int st = PROD ? cst : 0;
                for (int it = 0; it < ntiles; ++it) {
                    acquire(it, st);
                    const uint8_t* vs = smem + S::KV + st * 2 * TILE_BYTES + TILE_BYTES;
                    const int pos0 = lo + it * BN;
                    const bool interior =
                        pos0 + BN - 1 <= seq && (!p.window || pos0 >= seq + L - p.window);
                    float s[32];
                    {
                        const uint64_t kd =
                            desc_sw128(smem_base + S::KV + st * 2 * TILE_BYTES + hk * 64);
                        wg_fence();
                        wgmma_qk<0>(s, qd0, kd);
                        wgmma_qk<1>(s, qd1, kd + 2);
                        wg_commit();
                        wg_wait<0>();
                        reg_fence(s);
                    }
                    if (!interior) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
#pragma unroll
                            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                                for (int e = 0; e < 2; ++e) {
                                    const int pos = pos0 + 8 * i + 2 * c + e;
                                    const int qp = qp0 + hh;
                                    if (pos > qp || (p.window && pos < qp - p.window + 1)) {
                                        s[4 * i + 2 * hh + e] = NEG_INF;
                                    }
                                }
                            }
                        }
                    }
                     
                     
                    float tmax[2];
                    bool move[2];
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        float mx = fmaxf(s[2 * hh], s[2 * hh + 1]);
#pragma unroll
                        for (int i = 1; i < 8; ++i) {
                            mx = fmaxf(mx, fmaxf(s[4 * i + 2 * hh], s[4 * i + 2 * hh + 1]));
                        }
                        tmax[hh] = quad_max(mx) * sl2k;
                        move[hh] = tmax[hh] > off[hh] + 8.f;
                    }
                    if (__any_sync(0xffffffffu, move[0] || move[1])) {
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            const float nxt = move[hh] ? tmax[hh] : off[hh];
                            const float scale = ex2(off[hh] - nxt);
                            off[hh] = nxt;
#pragma unroll
                            for (int i = 0; i < 9; ++i) {
                                o[4 * i + 2 * hh] *= scale;
                                o[4 * i + 2 * hh + 1] *= scale;
                            }
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                            for (int e = 0; e < 2; ++e) {
                                s[4 * i + 2 * hh + e] =
                                    ex2(fmaf(s[4 * i + 2 * hh + e], sl2k, -off[hh]));
                            }
                        }
                    }
                     
                     
                     
                     
                    uint32_t pa[2][4];
#pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        const float* q0 = s + 16 * kk;
                        pa[kk][0] = e4m3x4(q0[0], q0[1], q0[4], q0[5]);
                        pa[kk][1] = e4m3x4(q0[2], q0[3], q0[6], q0[7]);
                        pa[kk][2] = e4m3x4(q0[8], q0[9], q0[12], q0[13]);
                        pa[kk][3] = e4m3x4(q0[10], q0[11], q0[14], q0[15]);
                    }
                     
                     
                     
                     
                     
                    uint8_t* vt = smem + S::VT + (2 * (it & 1) + hk) * VT_STRIDE;
                    {
                        const int h4 = lane >> 3;
                        const int r8 = lane & 7;
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int key = half * 32 + h4 * 8 + r8;
                            uint32_t t0, t1, t2, t3;
                            ldsm_x4_trans(smem_u32(vs + key * ROWB + ((hk * 4 + wq) ^ r8) * 16), t0,
                                          t1, t2, t3);
                            const uint32_t m0 = __byte_perm(t0, t1, 0x6420);
                            const uint32_t m1 = __byte_perm(t0, t1, 0x7531);
                            const uint32_t m2 = __byte_perm(t2, t3, 0x6420);
                            const uint32_t m3 = __byte_perm(t2, t3, 0x7531);
                            const int n = 8 * (2 * wq + (h4 & 1)) + r8;
                            const int chunk = 2 * half + (h4 >> 1);
                            stsm_x4(smem_u32(vt + n * 64 + ((chunk ^ ((n >> 1) & 3)) << 4)), m0, m1,
                                    m2, m3);
                        }
                    }
                    fence_proxy_async();
                    named_sync(2 + hk, 128);
                    {
                        const uint64_t vd = desc_sw64(smem_u32(vt));
                        wg_fence();
                        wgmma_pv(o, pa[0], vd);
                        wgmma_pv(o, pa[1], vd + 2);
                        wg_commit();
                        wg_wait<0>();
                        reg_fence(o);
                    }
                    release(st);
                    st = PROD ? cst : (st == STAGES - 1 ? 0 : st + 1);
                }
            }
            if constexpr (STAMPS) {
                s_loop += clock64() - s_l0;
                s_tiles += ntiles;
                ++s_nseg;
            }
            stamp(16);
            PHASE_EXIT(10)
             
             
             
             
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const float f = ex2(fminf(off[hh], 126.f));
                const int row = ws_row(b, p.nc, copy, L, 2 * wq + hh, hk * 8 + g);
                float* dst = p.ws_acc + static_cast<size_t>(row) * D + c * 4;
#pragma unroll
                for (int pp = 0; pp < 4; ++pp) {
                    red_add_v4(dst + 16 * pp, o[8 * pp + 2 * hh] * f, o[8 * pp + 4 + 2 * hh] * f,
                               o[8 * pp + 1 + 2 * hh] * f, o[8 * pp + 5 + 2 * hh] * f);
                }
                if (c == 0) {
                    red_add_f32(p.ws_l + row, o[32 + 2 * hh] * f);
                }
            }
        }
        stamp(17);
        return true;
    };
#undef PHASE_EXIT

     
     
     
    auto ticket = [&]() {
        csync();
        stamp(18);
        const int mine = hi - lo;
        if (tid == 0) {
             
             
            const int old = atom_add_acq_rel(p.ws_cnt + b, mine);
            *flag = old + mine >= n_keys;
        }
        stamp(19);
        csync();
        if (ONE_DIAG && p.phase_limit == 12) {
            return;
        }
        if (*flag) {
#if ONE_DETM
            if constexpr (DETM) {
                finalize_det<SEGK>(p, offs, B, G, b, v_scale, tid);
            } else
#endif
            {
                finalize_rows<L, NC, FIN8>(p, b * ROWS, (b + 1) * ROWS, v_scale, tid);
            }
            if (tid == 0) {
                p.ws_cnt[b] = 0;
            }
        }
        stamp(20);
    };

    if constexpr (!FLAT) {
        seq = SPEC ? p.pt_stride - L : seq_real;
        derive();
        if (!SPEC && lo >= end) {
            return;
        }
        if (!run_segment()) {
            return;
        }
         
        if (ONE_DIAG && p.phase_limit == 11) {
            return;
        }
        if constexpr (!COOP) {
            ticket();
        } else {
             
             
            __syncthreads();
            const int ep = *reinterpret_cast<const int*>(smem + S::CONST + C_EPOCH);
            const int rows_per = (ROWS + nact - 1) / nact;
            coop_barrier<NC>(p.ws_cnt + CNT_COOP + 2 * b, p.ws_cnt + CNT_EPOCH + b, ep, hi - lo,
                             n_keys, tid, p.dbg);
            if (ONE_DIAG && p.phase_limit == 12) {
                return;
            }
            const int r_lo = split * rows_per;
            const int r_hi = min(ROWS, r_lo + rows_per);
            if (r_lo < r_hi) {
                finalize_rows<L, NC, FIN8>(p, b * ROWS + r_lo, b * ROWS + r_hi, v_scale, tid);
            }
        }
    } else {
         
         
        asm volatile("griddepcontrol.wait;" ::: "memory");
         
         
        const int ep = COOP ? __ldcg(p.ws_cnt + CNT_FLAT + 2) : 0;
        const float v_scale_g = __ldg(p.v_scale);
        int mine_total = 0;
        for (b = b_first; b <= b_last; ++b) {
            if (!seg_bounds(b)) {
                continue;
            }
            run_segment();
            if constexpr (!COOP) {
                if constexpr (STAMPS) {
                    if (tid == 0) {
                        srec(4, snow());
                    }
                }
                ticket();
            } else {
                mine_total += hi - lo;
            }
        }
        if constexpr (STAMPS && !COOP) {
            if (tid == 0) {
                srec(5, snow());
                srec(6, s_tiles);
                srec(7, s_nseg);
                srec(8, s_fullw);
                srec(9, s_pro);
                srec(10, s_loop);
                srec(14, clock64() - s_k0);
            }
        }
        if constexpr (COOP) {
            if constexpr (STAMPS) {
                if (tid == 0) {
                    srec(3, snow());
                    srec(6, s_tiles);
                    srec(7, s_nseg);
                    srec(8, s_fullw);
                    srec(9, s_pro);
                    srec(10, s_loop);
                    srec(13, clock64() - s_k0);
                }
            }
            csync();
            coop_barrier<NC>(p.ws_cnt + CNT_FLAT, p.ws_cnt + CNT_FLAT + 2, ep, mine_total,
                             keys_all, tid, p.dbg);
            if constexpr (STAMPS) {
                if (tid == 0) {
                    srec(4, snow());
                }
            }
            const int rows_all = B * ROWS;
            const int rows_per = (rows_all + G - 1) / G;
            const int r_lo = split * rows_per;
            const int r_hi = min(rows_all, r_lo + rows_per);
            if (r_lo < r_hi) {
                finalize_rows<L, NC>(p, r_lo, r_hi, v_scale_g, tid);
            }
            if constexpr (STAMPS) {
                if (tid == 0) {
                    srec(5, snow());
                    srec(14, clock64() - s_k0);
                }
            }
        }
    }
}

 
 
 
 
 
 
template <int L, int MODE>
struct Instance {
    static constexpr int NT = one_threads(L, MODE);
    static constexpr int BYTES = Smem<L, one_ns(L, MODE), one_pb(L, MODE)>::BYTES;
    static int resident;
    static int per_sm;

    static int prepare() {
        if (resident < 0) {
            auto kern = attn_one_kernel<L, MODE>;
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);
            int blocks = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, NT, BYTES);
            int dev = 0;
            cudaGetDevice(&dev);
            int sms = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            per_sm = blocks;
            resident = blocks * sms;
        }
        return resident;
    }

    static cudaError_t launch(const Params& p, int gx, int gy, cudaStream_t stream, bool coop) {
        prepare();
        if (!coop) {
            attn_one_kernel<L, MODE><<<dim3(gx, gy), NT, BYTES, stream>>>(p);
            return cudaGetLastError();
        }
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(gx, gy);
        cfg.blockDim = dim3(NT);
        cfg.dynamicSmemBytes = BYTES;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeCooperative;
        attr[0].val.cooperative = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        const cudaError_t err = cudaLaunchKernelEx(&cfg, attn_one_kernel<L, MODE>, p);
        return err != cudaSuccess ? err : cudaGetLastError();
    }
};

template <int L, int MODE>
int Instance<L, MODE>::resident = -1;

template <int L, int MODE>
int Instance<L, MODE>::per_sm = 0;

__global__ void smid_probe_kernel(int* out) {
    if (threadIdx.x == 0) {
        uint32_t smid;
        asm volatile("mov.u32 %0, %%smid;\\n" : "=r"(smid));
        atomicMax(out, static_cast<int>(smid));
    }
}

 
 
int wave_sms(cudaStream_t stream) {
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cs);
    if (cs != cudaStreamCaptureStatusNone) {
        return 0;
    }
    int dev = 0;
    int n = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    int* d = nullptr;
    int h = 1 << 30;
    if (cudaMalloc(&d, sizeof(int)) == cudaSuccess) {
        cudaMemset(d, 0, sizeof(int));
        smid_probe_kernel<<<16 * n, 32>>>(d);
        if (cudaDeviceSynchronize() != cudaSuccess
            || cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            h = 1 << 30;
        }
        cudaFree(d);
    }
    cudaGetLastError();
    cached = h < n ? n : 0;
    return cached;
}

__global__ void coop_probe_kernel(int* out) {
    if (threadIdx.x == 0) {
        atomicAdd(out, 1);
    }
}

#if ONE_T9
#define ONE_T9_INSTANCES(X) X(1, 9) X(8, 9) X(8, 13)
#else
#define ONE_T9_INSTANCES(X)
#endif
#if ONE_DETM
#define ONE_INSTANCES(X) X(1, 49)
#else
#define ONE_INSTANCES(X) \\
    X(1, 1) X(1, 3) X(1, 11) X(1, 13) X(1, 17) X(1, 49) \\
    X(8, 1) X(8, 3) X(8, 11) X(8, 15) X(8, 19) X(8, 51) X(8, 115) ONE_T9_INSTANCES(X)
#endif

}   
}   
'''

_ONE_HOST_SRC = r'''
namespace {
namespace one {

 
 
 
 
 
 
#define DL_ONE_INSTANCES(X) X(8,3) X(8,51)

template <int L, int MODE>
void one_launch(const Params& p, int gx, int gy, cudaStream_t stream, bool coop, bool pdl) {
    using I = Instance<L, MODE>;
    I::prepare();
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(gx, gy);
    cfg.blockDim = dim3(I::NT);
    cfg.dynamicSmemBytes = I::BYTES;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    unsigned n = 0;
    if (coop) {
        attr[n].id = cudaLaunchAttributeCooperative;
        attr[n].val.cooperative = 1;
        ++n;
    }
    if (pdl) {
        attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[n].val.programmaticStreamSerializationAllowed = 1;
        ++n;
    }
    cfg.attrs = attr;
    cfg.numAttrs = n;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, attn_one_kernel<L, MODE>, p);
    TORCH_CHECK(err == cudaSuccess, "attn_one launch (q_len ", L, ", mode ", MODE, "): ",
                cudaGetErrorString(err));
}

}   
}   

 
 
std::vector<int64_t> attn_one_prepare() {
    using namespace one;
    std::vector<int64_t> out;
#define DL_ONE_PREP(L, M) out.push_back(L); out.push_back(M); \\
    out.push_back(Instance<L, M>::prepare()); out.push_back(Instance<L, M>::BYTES); \\
    out.push_back(Instance<L, M>::per_sm);
    DL_ONE_INSTANCES(DL_ONE_PREP)
#undef DL_ONE_PREP
    return out;
}

 
 
 
 
int64_t attn_one_probe_coop(bool pdl) {
    using namespace one;
    int* d = nullptr;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
     
    bool ok = cudaMemsetAsync(d, 0, sizeof(int), s) == cudaSuccess;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ok = ok && cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) {
        coop_probe_kernel<<<1, 32, 0, s>>>(d);
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(2);
        cfg.blockDim = dim3(32);
        cfg.stream = s;
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeCooperative;
        attr[0].val.cooperative = 1;
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = pdl ? 2 : 1;
        ok = cudaLaunchKernelEx(&cfg, coop_probe_kernel, d) == cudaSuccess;
        ok = (cudaStreamEndCapture(s, &graph) == cudaSuccess) && ok;
    }
    if (ok) {
        ok = cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    }
    if (ok) {
        ok = cudaGraphLaunch(exec, s) == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess;
    }
    int h = 0;
    if (ok) {
        ok = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess && h == 3;
    }
    if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(s);
    cudaFree(d);
    cudaGetLastError();
    return ok ? 1 : 0;
}

 
 
 
 
 
torch::Tensor attn_one(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                       torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                       torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                       torch::Tensor ws_acc, torch::Tensor ws_l, torch::Tensor ws_cnt,
                       int64_t batch, int64_t q_len, int64_t page_size, int64_t window, bool rope,
                       double sl2, int64_t n_splits, int64_t nc, int64_t mode, bool coop_launch,
                       bool pdl, torch::Tensor dbg) {
    using namespace one;
    TORCH_CHECK(q_len == 1 || q_len == 8, "attn_one: q_len must be 1 or 8");
    TORCH_CHECK(qkv.is_contiguous() && qkv.scalar_type() == torch::kBFloat16
                && qkv.dim() == 2 && qkv.size(0) == batch * q_len && qkv.size(1) == QKV_STRIDE,
                "attn_one: qkv must be a contiguous BF16 [batch * q_len, 2304]");
    TORCH_CHECK(page_table.is_contiguous() && page_table.scalar_type() == torch::kInt32
                && page_table.dim() == 2 && page_table.size(0) == batch,
                "attn_one: page_table must be a contiguous int32 [batch, pages]");
    TORCH_CHECK(seq_lens.is_contiguous() && seq_lens.scalar_type() == torch::kInt32
                && seq_lens.numel() == batch, "attn_one: seq_lens must be int32 [batch]");
    TORCH_CHECK(cache_loc.is_contiguous() && cache_loc.scalar_type() == torch::kInt64
                && cache_loc.numel() == batch * q_len,
                "attn_one: cache_loc must be int64 [batch * q_len]");
    TORCH_CHECK(inv_freq.is_contiguous() && inv_freq.scalar_type() == torch::kFloat
                && inv_freq.numel() == D / 2, "attn_one: inv_freq must be fp32 [32]");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat && v_scale.scalar_type() == torch::kFloat,
                "attn_one: scales must be fp32");
     
     
    TORCH_CHECK(batch >= 1 && batch <= 512 && n_splits >= 1 && n_splits <= 65535,
                "attn_one: batch or grid out of range");
    TORCH_CHECK(ws_acc.scalar_type() == torch::kFloat && ws_acc.is_contiguous()
                && ws_acc.dim() == 2 && ws_acc.size(1) == D
                && ws_acc.size(0) >= batch * nc * q_len * HQ && ws_l.numel() >= ws_acc.size(0)
                && ws_cnt.scalar_type() == torch::kInt32 && ws_cnt.numel() >= CNT_INTS,
                "attn_one: workspace too small for this plan");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({batch * q_len, HQ * D}, qkv.options());
    Params p;
    p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
    p.page_table = page_table.data_ptr<int>();
    p.seq_lens = seq_lens.data_ptr<int>();
    p.cache_loc = cache_loc.data_ptr<int64_t>();
    p.inv_freq = inv_freq.data_ptr<float>();
    p.k_scale = k_scale.data_ptr<float>();
    p.v_scale = v_scale.data_ptr<float>();
    p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
    p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.ws_acc = ws_acc.data_ptr<float>();
    p.ws_l = ws_l.data_ptr<float>();
    p.ws_cnt = ws_cnt.data_ptr<int>();
    p.dbg = dbg.numel() ? dbg.data_ptr<int>() : nullptr;
    p.pt_stride = static_cast<int>(page_table.size(1));
    p.page_size = static_cast<int>(page_size);
    p.window = static_cast<int>(window);
    p.rope = rope ? 1 : 0;
    p.n_splits = static_cast<int>(n_splits);
    p.magic = n_splits > 1 ? static_cast<uint32_t>(((1ull << 32) + n_splits - 1) / n_splits) : 0u;
    p.nc = static_cast<int>(nc);
    p.sl2 = static_cast<float>(sl2);
    p.phase_limit = 0;
    p.lp = 0;
    while ((1 << p.lp) < page_size) {
        ++p.lp;
    }
     
    TORCH_CHECK(!ONE_DIET || (1 << p.lp) == page_size, "attn_one: page size ", page_size,
                " is not a power of two");
     
     
     
    const int64_t cache_bytes = std::max<int64_t>(k_cache.numel(), v_cache.numel());
    p.off32 = (1 << p.lp) == page_size && cache_bytes <= (int64_t{1} << 32) ? 1 : 0;
     
    p.wave_n = 0;
     
     
    int gx = static_cast<int>(n_splits);
    int gy = static_cast<int>(batch);
    if (mode & M_FLAT) {
        p.n_splits = static_cast<int>(batch);
        gy = 1;
    }
    const bool coop = coop_launch && (mode & M_COOP) != 0;
    bool hit = false;
#define DL_ONE_LAUNCH(L, M) \\
    if (q_len == L && mode == M) { one_launch<L, M>(p, gx, gy, stream, coop, pdl); hit = true; }
    DL_ONE_INSTANCES(DL_ONE_LAUNCH)
#undef DL_ONE_LAUNCH
    TORCH_CHECK(hit, "attn_one: no instance for q_len ", q_len, " mode ", mode);
    return out;
}

template<int MODE> void qual_one_append(std::vector<int64_t>& out) {
    using I=one::Instance<8,MODE>;I::prepare();auto kernel=one::attn_one_kernel<8,MODE>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,I::BYTES));
    cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    int blocks=0,device=0,sms=0,coop=0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,I::NT,I::BYTES));
    const int64_t row[]={a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,I::BYTES,
                         I::NT,a.maxThreadsPerBlock,blocks,sms,coop,blocks*sms};
    out.insert(out.end(),row,row+10);
}
std::vector<int64_t> qual_one_resources() {
    std::vector<int64_t> out;qual_one_append<3>(out);qual_one_append<51>(out);return out;
}
'''

_BND_SRC = r'''
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace {
namespace bnd {

constexpr int H = 8192;
constexpr int LIVE = 1024;
constexpr int CTAS = 8;
constexpr int THREADS = 256;
constexpr int PER = H / CTAS / THREADS;
static_assert(PER * THREADS == LIVE, "each thread owns the same 4 channels of every slice");

__device__ __forceinline__ float4 bf16x4(const __nv_bfloat16* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float bf16r(float v) {
    return __bfloat162float(__float2bfloat16(v));
}
template <int NT>
__device__ __forceinline__ float block_sum_n(float v, float* red) {
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) {
        red[warp] = v;
    }
    __syncthreads();
    v = lane < NT / 32 ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// attn_boundary_kernel's math at other launch shapes: CL CTAs per token (a cluster, launched
// with the cluster attribute, when CL > 1) of NT threads, each thread owning V float4 groups
// strided by NT so every load is coalesced. prepare() times them against the 8-CTA kernel and
// SGLang's module per token count (_abnd_tune). The 8-CTA kernel fits 124 clusters at once on
// the judge's H200 (check 5285e58d978f42c6), so 128 tokens take two waves and 256 take three;
// there its o + boundary chain was 22.9 / 29.6 us against 17.9 / 22.5 for the best other.
// Only the stream is loaded before the wait: with V groups the gate rows would hold 8 V more
// registers per thread through the wait, and they are L2 hits after the first token.
template <int CL, int NT>
__global__ void __launch_bounds__(NT) attn_boundary_geo_kernel(
    const float* __restrict__ stream, const __nv_bfloat16* __restrict__ branch,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ w_next, float* __restrict__ stream_out,
    __nv_bfloat16* __restrict__ next, float* __restrict__ router, float eps_post,
    float eps_next) {
    asm volatile("griddepcontrol.launch_dependents;");
    constexpr int SLICE = H / CL;
    constexpr int V = SLICE / (4 * NT);
    static_assert(V >= 1 && V * 4 * NT == SLICE, "whole float4 groups per thread");
    namespace cg = cooperative_groups;
    __shared__ float red[NT / 32];
    __shared__ float part[2];
    int rank = 0;
    if constexpr (CL > 1) {
        rank = static_cast<int>(cg::this_cluster().block_rank());
    }
    const size_t row = static_cast<size_t>(blockIdx.y) * H;
    const int base = rank * SLICE + 4 * threadIdx.x;
    float4 s[V];
#pragma unroll
    for (int j = 0; j < V; ++j) {
        s[j] = *reinterpret_cast<const float4*>(stream + row + base + 4 * NT * j);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float4 f[V];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        f[j] = bf16x4(branch + row + base + 4 * NT * j);
        ss += f[j].x * f[j].x + f[j].y * f[j].y + f[j].z * f[j].z + f[j].w * f[j].w;
    }
    ss = block_sum_n<NT>(ss, red);
    if constexpr (CL > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        if (threadIdx.x == 0) {
            part[0] = ss;
        }
        cluster.sync();
        ss = 0.f;
#pragma unroll
        for (int r = 0; r < CL; ++r) {
            ss += cluster.map_shared_rank(part, r)[0];
        }
    }
    const float rb = rsqrtf(ss / H + eps_post);
    float ss2 = 0.f;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        const int c = base + 4 * NT * j;
        const float4 a = *reinterpret_cast<const float4*>(alpha + c);
        const float4 b = *reinterpret_cast<const float4*>(beta + c);
        s[j].x = a.x * s[j].x + b.x * bf16r(f[j].x * rb);
        s[j].y = a.y * s[j].y + b.y * bf16r(f[j].y * rb);
        s[j].z = a.z * s[j].z + b.z * bf16r(f[j].z * rb);
        s[j].w = a.w * s[j].w + b.w * bf16r(f[j].w * rb);
        ss2 += s[j].x * s[j].x + s[j].y * s[j].y + s[j].z * s[j].z + s[j].w * s[j].w;
    }
    ss2 = block_sum_n<NT>(ss2, red);
    if constexpr (CL > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        if (threadIdx.x == 0) {
            part[1] = ss2;
        }
        cluster.sync();
        ss2 = 0.f;
#pragma unroll
        for (int r = 0; r < CL; ++r) {
            ss2 += cluster.map_shared_rank(part, r)[1];
        }
         
        cluster.sync();
    }
    const float rn = rsqrtf(ss2 / H + eps_next);
#pragma unroll
    for (int j = 0; j < V; ++j) {
        const int c = base + 4 * NT * j;
        const float4 nv = bf16x4(w_next + c);
        const float4 o = make_float4(s[j].x * rn * nv.x, s[j].y * rn * nv.y,
                                     s[j].z * rn * nv.z, s[j].w * rn * nv.w);
        *reinterpret_cast<float4*>(stream_out + row + c) = s[j];
        const __nv_bfloat162 h0 = __floats2bfloat162_rn(o.x, o.y);
        const __nv_bfloat162 h1 = __floats2bfloat162_rn(o.z, o.w);
        *reinterpret_cast<uint2*>(next + row + c) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                       *reinterpret_cast<const uint32_t*>(&h1));
        if (router != nullptr) {
            *reinterpret_cast<float4*>(router + row + c) = o;
        }
    }
}

template <typename Kern, typename... Args>
void launch_cl(Kern kern, dim3 grid, dim3 block, int cluster, cudaStream_t stream,
               Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = kPdlGlue ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = cluster;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = cluster > 1 ? 2 : 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

 
constexpr int kGeoCl[] = {8, 2, 4, 1, 1, 1};
constexpr int kGeoNt[] = {256, 512, 256, 256, 512, 1024};
constexpr int kGeos = 6;

template <typename F>
void with_geo(int geo, F fn) {
    TORCH_CHECK(geo==4,"owned verify boundary geometry");
    fn(attn_boundary_geo_kernel<1,512>);
}

}  
}  
std::vector<torch::Tensor> attn_boundary(torch::Tensor stream, torch::Tensor branch,
                                         torch::Tensor alpha, torch::Tensor beta,
                                         torch::Tensor w_next, double eps_post,
                                         double eps_next, bool want_router, int64_t geo) {
    using namespace bnd;
    TORCH_CHECK(geo == 4, "attn_boundary: geometry id");
    const int tokens = stream.size(0);
    TORCH_CHECK(stream.is_contiguous() && stream.scalar_type() == torch::kFloat32
                && stream.size(1) == H, "attn_boundary: stream must be FP32 [T, 8192]");
    TORCH_CHECK(branch.is_contiguous() && branch.scalar_type() == torch::kBFloat16
                && branch.size(0) == tokens && branch.size(1) == H,
                "attn_boundary: branch must be BF16 [T, 8192]");
    TORCH_CHECK(w_next.is_contiguous() && alpha.is_contiguous() && beta.is_contiguous()
                && w_next.scalar_type() == torch::kBFloat16
                && alpha.scalar_type() == torch::kFloat32
                && beta.scalar_type() == torch::kFloat32 && w_next.numel() == H
                && alpha.numel() == H && beta.numel() == H, "attn_boundary: weights");
    auto stream_out = torch::empty_like(stream);
    auto next = torch::empty({tokens, H}, branch.options());
    auto router = torch::empty({want_router ? tokens : 0, H}, stream.options());
    if (tokens == 0) {
        return {next, stream_out, router};
    }
    const float* sp = stream.data_ptr<float>();
    const auto* bp = reinterpret_cast<const __nv_bfloat16*>(branch.data_ptr());
    const auto* np = reinterpret_cast<const __nv_bfloat16*>(w_next.data_ptr());
    auto* op = reinterpret_cast<__nv_bfloat16*>(next.data_ptr());
    float* rp = want_router ? router.data_ptr<float>() : static_cast<float*>(nullptr);
    const auto cs = at::cuda::getCurrentCUDAStream();

        with_geo(static_cast<int>(geo), [&](auto kern) {
            launch_cl(kern, dim3(kGeoCl[geo], tokens), dim3(kGeoNt[geo]), kGeoCl[geo], cs, sp,
                      bp, alpha.data_ptr<float>(), beta.data_ptr<float>(), np,
                      stream_out.data_ptr<float>(), op, rp, static_cast<float>(eps_post),
                      static_cast<float>(eps_next));
        });
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {next, stream_out, router};
}
'''

_MOE1_SRC = r'''


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
namespace moe1 {

constexpr int kH = 4096;
constexpr int kI = 4096;
constexpr int kTopK = 8;
constexpr int kLocalMax = 32;
constexpr int kEmax = 4096;
constexpr int kJmax = kLocalMax + kEmax / 8;
constexpr int kUnitBytes = 36864;
constexpr int kUA = 512;
constexpr int kUB = 256;
constexpr int kNRB = 4;
 
 
 
constexpr int kBarUnit = 1;
constexpr int kBarFull = 2;
constexpr int kBarFree = 6;
 
constexpr long long kSpinLimit = 1ll << 31;

struct Params {
    const __nv_bfloat16* x;
    const int* topk_ids;
    const float* topk_w;
    const unsigned char* w13;
    const unsigned char* w2;
    const float* gs13;
    const float* gs2;
    const float* nw;
    __nv_bfloat16* inter;
    float* ybuf;
    float* ss;
    int* ctr;
    int* done_a;
    __nv_bfloat16* out;
    const unsigned char* w13f;
    const unsigned char* w2f;
    unsigned char* xa;
    unsigned char* xb;
    float* fix;
    int* fixcnt;
    int* xready;
    int* done_as;
    int M;
    int lo;
    int nloc;
    float eps;
     
    int cost[2][9];
     
    const int* lo_p;
    int pdl;
};

template <typename Kern>
void moe1_launch(Kern kern, int grid, int block, int smem, cudaStream_t stream,
                 const Params& p) {
    TORCH_CHECK(!p.pdl, "OWNED_COOP_RESOURCE: PDL overlap unsupported");
    int device=0, resident=0;
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    TORCH_CHECK(prop.major==9 && prop.minor==0 && prop.cooperativeLaunch,
                "OWNED_COOP_RESOURCE: requires SM90 cooperative launch");
    TORCH_CHECK(grid>0 && grid<=prop.multiProcessorCount && block>0 && smem>=0,
                "OWNED_COOP_RESOURCE: grid/block/shared bounds");
    TORCH_CHECK(p.M>0 && p.M<=kEmax/kTopK && p.nloc>=1 && p.nloc<=kLocalMax,
                "OWNED_COOP_RESOURCE: chunk/local expert bounds");
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kern,block,smem));
    TORCH_CHECK(resident>=1 && grid<=resident*prop.multiProcessorCount,
                "OWNED_COOP_RESOURCE: whole grid cannot reside");
    Params launch_params=p;
    void* args[]={&launch_params};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)kern,dim3(grid),dim3(block),
                                              args,smem,stream));
}

__device__ __forceinline__ void prefetch_l2_hint(const void* src, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes)
                 : "memory");
}



__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbar_try(uint64_t* bar, unsigned parity) {
    unsigned ok;
    asm volatile("{\\n.reg .pred p;\\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n"
                 "selp.u32 %0, 1, 0, p;\\n}\\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
    return ok != 0;
}

__device__ __forceinline__ void bulk_load(void* dst, const void* src, unsigned bytes,
                                          uint64_t* bar, uint64_t policy) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1], %2, [%3], %4;"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ uint64_t policy_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

__device__ __forceinline__ void named_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void named_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ int ld_lo(const int* p) {
    int v;
    asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

 
 
 
 
 
 
 
 
__device__ __forceinline__ void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0,
                                         unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned bmul2(unsigned a, unsigned s) {
    const __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),
                                     *reinterpret_cast<__nv_bfloat162*>(&s));
    return *reinterpret_cast<const unsigned*>(&r);
}

 
 
 
 
 
 
__device__ __forceinline__ void dequant(unsigned w, unsigned sg, unsigned sg8, unsigned (&a)[4]) {
    constexpr unsigned kM = 0x81C081C0u;
    const unsigned r0 = w & kM;
    const unsigned r1 = (w << 6) & kM;
    const unsigned r2 = __funnelshift_r(w, w, 6) & kM;
    const unsigned r3 = ((w >> 4) & 0x00400040u) | ((w << 4) & 0x81808180u);
    a[0] = bmul2(r0, sg);
    a[1] = bmul2(r1, sg8);
    a[2] = bmul2(r2, sg);
    a[3] = bmul2(r3, sg8);
}

__device__ __forceinline__ unsigned scale_word(unsigned scw, int byte) {
    return __byte_perm(scw, 0u, 0x4040u + 0x0101u * byte) * 16u + 0x77807780u;
}

 
 
__device__ __forceinline__ unsigned scale_word_k(unsigned scw, int byte, unsigned k16) {
    return __byte_perm(scw, 0u, 0x4040u + 0x0101u * byte) * k16 + 0x77807780u;
}

__device__ __forceinline__ float bf16r(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ unsigned word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ int warp_scan(int v, int lane) {
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, d);
        if (lane >= d) {
            v += u;
        }
    }
    return v;
}


__device__ __forceinline__ bool spin_failed(long long t0, int* ctr, int code) {
    if (clock64() - t0 < kSpinLimit) {
        return false;
    }
    atomicOr(ctr + 8, code);
    asm volatile("trap;" ::: "memory");
    return true;
}


 
 
template <int NWARPS>
__device__ __forceinline__ void position_pass(const signed char* eid_s, short* pos, short* ent,
                                              int (*wc)[kLocalMax], const int* off, int E) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int ch = ((E + NWARPS * 32 - 1) / (NWARPS * 32)) * 32;
    const int i0 = warp * ch;
    const int i1 = min(E, i0 + ch);
    wc[warp][lane] = 0;
    __syncwarp();
    for (int base = i0; base < i1; base += 32) {
        const int i = base + lane;
        const int eid = i < i1 ? eid_s[i] : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, eid);
        if (eid >= 0 && lane == __ffs(peers) - 1) {
            wc[warp][eid] += __popc(peers);
        }
        __syncwarp();
    }
    __syncthreads();
    if (warp == 0) {
        int run = off[lane];
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            const int cw = wc[w][lane];
            wc[w][lane] = run;
            run += cw;
        }
    }
    __syncthreads();
    for (int base = i0; base < i1; base += 32) {
        const int i = base + lane;
        const int eid = i < i1 ? eid_s[i] : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, eid);
        if (eid >= 0) {
            const int q = wc[warp][eid] + __popc(peers & ((1u << lane) - 1u));
            if (ent != nullptr) {
                ent[q] = static_cast<short>(i);
            }
            pos[i] = static_cast<short>(q);
        } else if (i < i1) {
            pos[i] = -1;
        }
        __syncwarp();
        if (eid >= 0 && lane == __ffs(peers) - 1) {
            wc[warp][eid] += __popc(peers);
        }
        __syncwarp();
    }
    __syncthreads();
}

 
 
template <int NWARPS>
__device__ __forceinline__ void position_count(const signed char* eid_s,
                                               int (*wc)[kLocalMax], int E) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ch = ((E + NWARPS * 32 - 1) / (NWARPS * 32)) * 32;
    const int i0 = warp * ch;
    const int i1 = min(E, i0 + ch);
    wc[warp][lane] = 0;
    __syncwarp();
    for (int base = i0; base < i1; base += 32) {
        const int i = base + lane;
        const int eid = i < i1 ? eid_s[i] : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, eid);
        if (eid >= 0 && lane == __ffs(peers) - 1) {
            wc[warp][eid] += __popc(peers);
        }
        __syncwarp();
    }
}

template <int NWARPS>
__device__ __forceinline__ void position_fill(const signed char* eid_s, short* pos,
                                              int (*wc)[kLocalMax], int E) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ch = ((E + NWARPS * 32 - 1) / (NWARPS * 32)) * 32;
    const int i0 = warp * ch;
    const int i1 = min(E, i0 + ch);
    for (int base = i0; base < i1; base += 32) {
        const int i = base + lane;
        const int eid = i < i1 ? eid_s[i] : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, eid);
        if (eid >= 0) {
            pos[i] = static_cast<short>(wc[warp][eid] + __popc(peers & ((1u << lane) - 1u)));
        } else if (i < i1) {
            pos[i] = -1;
        }
        __syncwarp();
        if (eid >= 0 && lane == __ffs(peers) - 1) {
            wc[warp][eid] += __popc(peers);
        }
        __syncwarp();
    }
}

 
 
template <int THREADS>
__device__ __forceinline__ void final_pass(const Params& p, const short* pos, const float* ss,
                                           int cta, int G) {
    constexpr int SEGS = kH / (4 * THREADS);
    const int tid = threadIdx.x;
    const float inv = 1.f / kH;
    for (int it = cta; it < p.M * SEGS; it += G) {
        const int t = it / SEGS;
        const int col = (it % SEGS) * (4 * THREADS) + tid * 4;
        float coef[kTopK];
        int qs[kTopK];
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            qs[s] = pos[t * kTopK + s];
            float c = 0.f;
            if (qs[s] >= 0) {
                const float sv = __ldcg(ss + qs[s]);
                c = __ldg(p.topk_w + t * kTopK + s) * rsqrtf(sv * inv + p.eps);
            }
            coef[s] = c;
        }
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            if (qs[s] >= 0) {
                const float4 y = __ldcg(reinterpret_cast<const float4*>(
                    p.ybuf + static_cast<size_t>(qs[s]) * kH + col));
                acc.x += coef[s] * y.x;
                acc.y += coef[s] * y.y;
                acc.z += coef[s] * y.z;
                acc.w += coef[s] * y.w;
            }
        }
        uint2 o;
        const __nv_bfloat162 lo2 = __floats2bfloat162_rn(acc.x, acc.y);
        const __nv_bfloat162 hi2 = __floats2bfloat162_rn(acc.z, acc.w);
        o.x = *reinterpret_cast<const unsigned*>(&lo2);
        o.y = *reinterpret_cast<const unsigned*>(&hi2);
        *reinterpret_cast<uint2*>(p.out + static_cast<size_t>(t) * kH + col) = o;
    }
}


 
 
 
 
 
 
template <int THREADS>
__device__ __forceinline__ void final_pass_ss(const Params& p, const short* pos, float* red,
                                              int bar_id, int cta, int G) {
    static_assert(THREADS * 8 == kH, "8 columns per thread");
    constexpr int kW = THREADS / 32;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float inv = 1.f / kH;
    int par = 0;
    for (int t = cta; t < p.M; t += G, par ^= 1) {
        float* rb = red + par * (kW + 1) * kTopK;
        int qs[kTopK];
        float ps[kTopK];
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            qs[s] = pos[t * kTopK + s];
            ps[s] = 0.f;
            if (qs[s] >= 0) {
                const float4* src = reinterpret_cast<const float4*>(
                    p.ybuf + static_cast<size_t>(qs[s]) * kH) + tid * 2;
                const float4 a = __ldcg(src);
                const float4 b = __ldcg(src + 1);
                ps[s] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y
                        + b.z * b.z + b.w * b.w;
            }
        }
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            if (qs[s] >= 0) {
#pragma unroll
                for (int d = 16; d > 0; d >>= 1) {
                    ps[s] += __shfl_xor_sync(0xffffffffu, ps[s], d);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int s = 0; s < kTopK; ++s) {
                rb[warp * kTopK + s] = ps[s];
            }
        }
        named_sync(bar_id, THREADS);
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[i] = 0.f;
        }
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            if (qs[s] >= 0) {
                float sv = 0.f;
#pragma unroll
                for (int w = 0; w < kW; ++w) {
                    sv += rb[w * kTopK + s];
                }
                const float c = __ldg(p.topk_w + t * kTopK + s) * rsqrtf(sv * inv + p.eps);
                const float4* src = reinterpret_cast<const float4*>(
                    p.ybuf + static_cast<size_t>(qs[s]) * kH) + tid * 2;
                const float4 a = __ldcg(src);
                const float4 b = __ldcg(src + 1);
                acc[0] += c * a.x;
                acc[1] += c * a.y;
                acc[2] += c * a.z;
                acc[3] += c * a.w;
                acc[4] += c * b.x;
                acc[5] += c * b.y;
                acc[6] += c * b.z;
                acc[7] += c * b.w;
            }
        }
        uint4 o;
        unsigned* ow = reinterpret_cast<unsigned*>(&o);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
            ow[i] = *reinterpret_cast<const unsigned*>(&v);
        }
        *reinterpret_cast<uint4*>(p.out + static_cast<size_t>(t) * kH + tid * 8) = o;
    }
}

__device__ __forceinline__ long long gtimer() {
    long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}










 
 
 
 
constexpr int kNC = 16;
constexpr int kNTM = kNC * 32 + 64;
constexpr int kNWarpsM = kNTM / 32;
constexpr int kNSWMax = 5;
constexpr int kNSX = 3;
 
 
constexpr int kRingBytes = kNSWMax * kUnitBytes + kNSX * 8192;
constexpr int kJmaxM = kLocalMax + kEmax / 64;
constexpr int kUAM = 512;
constexpr int kUBM = 256;
constexpr int kBarWg = 3;
constexpr int kBarGather = 7;
constexpr int kBarFin = 8;
constexpr int kBqMax = 64;
constexpr int kBqEmpty = -2;
 
 
__constant__ int kCostA[9] = {0, 112, 135, 140, 158, 173, 185, 200, 207};
__constant__ int kCostB[9] = {0, 100, 104, 115, 123, 132, 144, 156, 174};
 
 
__constant__ int kCostAm[9] = {0, 90, 100, 115, 130, 150, 170, 190, 210};
__constant__ int kCostBm[9] = {0, 85, 92, 104, 115, 130, 145, 160, 175};

 
 
 
 
__host__ __device__ constexpr int mma_max(int mode) {
    return mode == 6 || mode == 9 || mode == 11 || mode == 12 || mode == 14 || mode == 15
                   || (mode >= 16 && mode <= 29) || (mode >= 31 && mode <= 51)
               ? 2
           : mode == 7 || mode == 10 ? 4
           : mode == 8 ? 8 : 0;
}

 
 
 
 
 
 
 
__host__ __device__ constexpr bool fence_ar(int mode) {
    return mode == 12 || mode == 13 || mode == 14 || (mode >= 15 && mode <= 29)
           || (mode >= 31 && mode <= 51);
}

 
 
 
 
 
 
 
__host__ __device__ constexpr bool dual_chain(int mode) {
    return mode == 14 || (mode >= 16 && mode <= 29) || (mode >= 31 && mode <= 51);
}
__host__ __device__ constexpr bool is_diag(int mode) {
    return mode == 11 || mode == 17 || mode == 22 || mode == 24;
}
 
 
 
__host__ __device__ constexpr bool is_light(int mode) {
    return mode == 25 || mode == 27 || mode == 29 || mode == 31 || mode == 33 || mode == 38
           || mode == 40 || mode == 41 || mode == 42 || mode == 44 || mode == 48 || mode == 49;
}
 
 
 
 
 
 
__host__ __device__ constexpr bool fast_setup(int mode) {
    return (mode >= 26 && mode <= 29) || (mode >= 31 && mode <= 51);
}
 
 
 
 
 
 
 
 
 
 
 
 
 
__host__ __device__ constexpr bool lean_stage(int mode) {
    return mode == 28 || mode == 29 || (mode >= 31 && mode <= 51);
}
 
 
 
 
 
 
 
 
 
__host__ __device__ constexpr bool pa_diag(int mode) {
    return mode == 31 || mode == 33 || mode == 38 || mode == 40 || mode == 41 || mode == 44;
}
 
 
 
__host__ __device__ constexpr bool pb_diag(int mode) {
    return mode == 41 || mode == 44;
}
 
 
 
 
 
 
 
 
 
 
 
__host__ __device__ constexpr bool grp_codes(int mode) {
    return mode == 43 || mode == 44 || (mode >= 47 && mode <= 51);
}
 
 
 
 
__host__ __device__ constexpr bool grp_hoist(int mode) {
    return mode == 43 || mode == 44;
}
 
 
 
 
 
 
 
 
 
 
 
__host__ __device__ constexpr bool skew_st(int mode) {
    return mode == 49;
}
 
 
 
 
 
 
 
__host__ __device__ constexpr bool final_ss(int mode) {
    return mode == 50 || mode == 51;
}
 
 
 
 
constexpr int kFinssMinM = 64;
template <int MODE>
__device__ __forceinline__ bool finss_on(int m) {
    return final_ss(MODE) && (MODE != 51 || m >= kFinssMinM);
}
 
 
 
 
 
 
__host__ __device__ constexpr bool defer_a(int mode) {
    return mode >= 32 && mode <= 35;
}
 
 
 
__host__ __device__ constexpr bool vol_regs(int mode) {
    return mode == 35;
}
 
 
 
 
__host__ __device__ constexpr bool bar_release_a(int mode) {
    return mode == 34 || mode == 37 || (mode >= 39 && mode <= 51);
}
 
 
 
 
 
 
 
 
__host__ __device__ constexpr bool fast_silu(int mode) {
    return mode >= 36 && mode <= 51;
}
 
 
 
 
 
 
__host__ __device__ constexpr bool pair_st(int mode) {
    return mode == 39 || mode == 40;
}
__host__ __device__ constexpr bool a_only(int mode) {
    return mode == 18 || mode == 19 || mode == 45;
}
__host__ __device__ constexpr bool own_a(int mode) {
    return mode == 19 || mode == 20 || (mode >= 22 && mode <= 29) || (mode >= 31 && mode <= 45)
           || (mode >= 47 && mode <= 51);
}
 
 
 
 
 
__host__ __device__ constexpr bool dyn_b(int mode) {
    return (mode >= 23 && mode <= 29) || (mode >= 31 && mode <= 44)
           || (mode >= 46 && mode <= 51);
}
 
 
 
 
 
 
 
 
 
__host__ __device__ constexpr bool b_only(int mode) {
    return mode == 21 || mode == 46;
}

template <bool AR>
__device__ __forceinline__ void fence_gpu() {
    if constexpr (AR) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    } else {
        __threadfence();
    }
}

 
 
 
 
 
 

template <int MODE>
__device__ __forceinline__ int stage_cost(int ph, int ntt) {
    constexpr bool kFlat = MODE >= 8 && MODE <= 10;
    if (kFlat && ntt <= mma_max(MODE)) {
        return ph ? kCostBm[ntt] : kCostAm[ntt];
    }
    return ph ? kCostB[ntt] : kCostA[ntt];
}

constexpr int kSkQ = 256;
constexpr int kSkRow = kNC * 4 + 2;
constexpr int kSkPer = 8 + kSkQ * kSkRow;
constexpr long long kSkOff = 1ll << 20;

__device__ __forceinline__ long long* sk_cta(const Params& p) {
    return reinterpret_cast<long long*>(p.inter) + kSkOff
           + static_cast<long long>(blockIdx.x) * kSkPer;
}

 
 
template <int MODE>
__device__ __forceinline__ void sk_stamp(const Params& p, int q, int k, bool on) {
    if constexpr (skew_st(MODE)) {
        const long long c = clock64();
        long long* a = sk_cta(p) + 8 + static_cast<long long>(q) * kSkRow + k;
        asm volatile("{\\n .reg .pred sp;\\n setp.ne.s32 sp, %2, 0;\\n @sp st.global.b64 [%0], %1;\\n}"
                     :: "l"(a), "l"(c), "r"(static_cast<int>(on && q < kSkQ)));
    }
}

struct __align__(128) SmemMain {
    unsigned char rings[kRingBytes];
    uint64_t wfull[kNSWMax];
    uint64_t wempty[kNSWMax];
    uint64_t xfull[kNSX];
    uint64_t xempty[kNSX];
    int cnt[kLocalMax];
    int off[kLocalMax];
    int job0[kLocalMax];
    int wc[kNWarpsM][kLocalMax];
    int job_e[kJmaxM];
    int job_q[kJmaxM];
    int job_n[kJmaxM];
    int job_ntt[kJmaxM];
    int job_xo[kJmaxM];
    int pa[kJmaxM + 1];
    int pb[kJmaxM + 1];
    short pos[kEmax];
    signed char eid[kEmax];
    int njobs;
    int nsw;
    int xsz;
    int xoff;
    int gsb[2];
    int gse[2];
    int fix_last[4];
    int cost[2][9];
    int align_b;
     
    int ord[kJmaxM];
    int own_n;
     
     
    int bq[kBqMax];
    long long dg[18][8];
    float* ssp;
};

 
 
struct Ring {
    int ws;
    int wlap;
    int xs;
    int xlap;
     
     
    int nsw;
    unsigned xb;
    unsigned xsz;
};

__device__ __forceinline__ void ring_next_w(Ring& r, int nsw) {
    if (++r.ws == nsw) {
        r.ws = 0;
        ++r.wlap;
    }
}

__device__ __forceinline__ void ring_next_x(Ring& r) {
    if (++r.xs == kNSX) {
        r.xs = 0;
        ++r.xlap;
    }
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_arrive_n(uint64_t* bar, unsigned n) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(n)
                 : "memory");
}

 
 
 
template <int MODE>
__device__ __forceinline__ void x_release_warp(uint64_t* bar, int lane) {
    if (lane == 0) {
        if constexpr (lean_stage(MODE)) {
            mbar_arrive_n(bar, 32);
        } else {
            mbar_arrive(bar);
        }
    }
}

__device__ __forceinline__ void mbar_wait_to(uint64_t* bar, unsigned parity, int* ctr, int code) {
    if (mbar_try(bar, parity)) {
        return;
    }
    const long long t0 = clock64();
    while (!mbar_try(bar, parity)) {
        if (spin_failed(t0, ctr, code)) {
            break;
        }
    }
}

 
 
 
 
__device__ __forceinline__ unsigned mbar_wait_cyc(uint64_t* bar, unsigned parity, int* ctr,
                                                  int code) {
    const long long t0 = clock64();
    while (!mbar_try(bar, parity)) {
        if (spin_failed(t0, ctr, code)) {
            break;
        }
    }
    return static_cast<unsigned>(clock64() - t0);
}

template <int MODE, int PH>
__device__ __forceinline__ unsigned mbar_wait_m(uint64_t* bar, unsigned parity, int* ctr,
                                                int code) {
    if constexpr (pa_diag(MODE) && (PH == 0 || pb_diag(MODE))) {
        return mbar_wait_cyc(bar, parity, ctr, code);
    } else {
        mbar_wait_to(bar, parity, ctr, code);
        return 0u;
    }
}

__device__ __forceinline__ void spin_ge(const int* p, int target, int* ctr, int code) {
    if (ld_acquire(p) >= target) {
        return;
    }
    const long long t0 = clock64();
    while (ld_acquire(p) < target) {
        __nanosleep(64);
        if (spin_failed(t0, ctr, code)) {
            break;
        }
    }
}

 
__device__ __forceinline__ void st_release_cta(int* p, int v) {
    asm volatile("st.release.cta.shared::cta.s32 [%0], %1;" :: "r"(smem_u32(p)), "r"(v)
                 : "memory");
}

__device__ __forceinline__ int ld_acquire_cta(const int* p) {
    int v;
    asm volatile("ld.acquire.cta.shared::cta.s32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p))
                 : "memory");
    return v;
}

__device__ __forceinline__ int bq_get(const SmemMain& S, int i, int* ctr) {
    int m = ld_acquire_cta(&S.bq[i]);
    if (m != kBqEmpty) {
        return m;
    }
    const long long t0 = clock64();
    while ((m = ld_acquire_cta(&S.bq[i])) == kBqEmpty) {
        __nanosleep(32);
        if (spin_failed(t0, ctr, 256)) {
            return -1;
        }
    }
    return m;
}

__device__ __forceinline__ void bulk_load_plain(void* dst, const void* src, unsigned bytes,
                                                uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1], %2, [%3];"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

 
__device__ __forceinline__ uint64_t make_desc(const void* ptr, int lbo, int sbo) {
    const uint64_t addr = smem_u32(ptr);
    return ((addr & 0x3ffffu) >> 4) | (static_cast<uint64_t>(lbo >> 4) << 16)
           | (static_cast<uint64_t>(sbo >> 4) << 32);
}

__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void red_add_v4(float4* p, float4 v) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ int atom_add_acqrel(int* p, int v) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v)
                 : "memory");
    return old;
}

__device__ __forceinline__ void red_release_add(int* p, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void wgmma_n8(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, %8, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n16(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
                 "{%8, %9, %10, %11}, %12, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n24(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n32(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
                 "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n40(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, "
                 "{%20, %21, %22, %23}, %24, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n48(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, "
                 "{%24, %25, %26, %27}, %28, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n56(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, "
                 "{%28, %29, %30, %31}, %32, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n64(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

template <int NTT>
__device__ __forceinline__ void wgmma(float (&d)[32], const unsigned (&a)[4], uint64_t desc) {
    if constexpr (NTT == 1) { wgmma_n8(d, a, desc); }
    if constexpr (NTT == 2) { wgmma_n16(d, a, desc); }
    if constexpr (NTT == 3) { wgmma_n24(d, a, desc); }
    if constexpr (NTT == 4) { wgmma_n32(d, a, desc); }
    if constexpr (NTT == 5) { wgmma_n40(d, a, desc); }
    if constexpr (NTT == 6) { wgmma_n48(d, a, desc); }
    if constexpr (NTT == 7) { wgmma_n56(d, a, desc); }
    if constexpr (NTT == 8) { wgmma_n64(d, a, desc); }
}

 
 
template <int NTT>
__device__ __forceinline__ void acc_fence(float (&d)[32]) {
#pragma unroll
    for (int i = 0; i < 4 * NTT; ++i) {
        asm volatile("" : "+f"(d[i]) :: "memory");
    }
}

 
 
 
__device__ __forceinline__ void a_pin(unsigned (&a)[2][4]) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            asm volatile("" : "+r"(a[i][k]) :: "memory");
        }
    }
}

 
 
__device__ __forceinline__ int stage_at(const SmemMain& S, int ph, int k, int G) {
    const int nj = S.njobs;
    const int spj = ph ? kUBM : kUAM;
    if (k >= G) {
        return nj * spj;
    }
    if (k <= 0 || nj == 0) {
        return 0;
    }
    const int* pre = ph ? S.pb : S.pa;
    const unsigned total = static_cast<unsigned>(pre[nj]);
    const unsigned target = total * static_cast<unsigned>(k) / static_cast<unsigned>(G);
    int j = 0;
    while (j + 1 < nj && static_cast<unsigned>(pre[j + 1]) <= target) {
        ++j;
    }
    const int cost = S.cost[ph][S.job_ntt[j]];
    const int s = j * spj + static_cast<int>((target - static_cast<unsigned>(pre[j]))
                                             / static_cast<unsigned>(cost));
     
     
    return ph && S.align_b ? (s + 8) & ~15 : s;
}

__device__ __forceinline__ int seg_count(int b, int e) {
    return b >= e ? 0 : ((e - 1) >> 4) - (b >> 4) + 1;
}

 
 
__device__ __forceinline__ void seg_get(int b, int e, int i, int& gb, int& ge) {
    const int ub = b >> 4;
    const int ue = (e - 1) >> 4;
    if (ub == ue) {
        gb = b;
        ge = e;
        return;
    }
    int k = i;
    if ((e & 15) != 0) {
        if (k == 0) {
            gb = ue << 4;
            ge = e;
            return;
        }
        --k;
    }
    if (k == 0) {
        gb = b;
        ge = (ub + 1) << 4;
        return;
    }
    gb = (ub + k) << 4;
    ge = gb + 16;
}

struct SegInfo {
    int j;
    int u;
    int st_b;
    int st_e;
    int slot;
    bool partial;
     
    int r;
};

 
 
 
__device__ __forceinline__ int own_unit(int r, int cta, int G) {
    return r * G + ((r & 1) ? G - 1 - cta : cta);
}

template <int MODE>
__device__ __forceinline__ int seg_n(const SmemMain& S, int ph) {
    if (own_a(MODE) && ph == 0) {
        return S.own_n;
    }
    return seg_count(S.gsb[ph], S.gse[ph]);
}

template <int MODE>
__device__ __forceinline__ void seg_at(const SmemMain& S, int ph, int i, int cta, int G, int& gb,
                                       int& ge) {
    if (own_a(MODE) && ph == 0) {
        const int k = own_unit(i, cta, G);
        gb = S.ord[k >> 5] * kUAM + (k & 31) * 16;
        ge = gb + 16;
        return;
    }
    seg_get(S.gsb[ph], S.gse[ph], i, gb, ge);
}

 
__device__ __forceinline__ SegInfo seg_info(const SmemMain& S, int ph, int gb, int ge, int cta,
                                            int G) {
    SegInfo si;
    si.j = ph ? (gb >> 8) : (gb >> 9);
    si.u = (gb & (ph ? 255 : 511)) >> 4;
    si.st_b = gb & 15;
    si.st_e = ((ge - 1) & 15) + 1;
    si.partial = si.st_b != 0 || si.st_e != 16;
    si.slot = cta;
    si.r = -1;
    if (si.st_b != 0) {
        const int gs0 = gb & ~15;
        int k = cta;
        while (k > 0 && stage_at(S, ph, k, G) > gs0) {
            --k;
        }
        si.slot = k;
    }
    return si;
}

template <bool FAST>
__device__ __forceinline__ float silu_rcp(float x) {
    if constexpr (FAST) {
        float r;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
        return r;
    } else {
        return __frcp_rn(x);
    }
}

 
 
template <int MODE>
__device__ __forceinline__ void publish_a(const Params& p, int target, int wg, int tid_wg) {
    fence_proxy_async_global();
    if constexpr (!bar_release_a(MODE)) {
        fence_gpu<fence_ar(MODE)>();
    }
    named_sync(kBarWg + wg, 128);
    if (tid_wg == 0) {
        red_release_add(p.done_as + target, 1);
    }
}

 
 
 
template <int NTT, int MODE, bool PUB>
__device__ __forceinline__ int epilogue_a(const Params& p, SmemMain& S,
                                          const float (&acc)[32], int j, int u, int wg,
                                          int w, int tid_wg) {
    const int lane = tid_wg & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int ntok = S.job_n[j];
    const float gsa = __ldg(p.gs13 + S.job_e[j]) * 128.f;
    const int rb = 16 * u + 4 * wg + w;
    const int st = u >> 1;
     
     
    const int kk = 8 * (rb & 31) + g;
    const int step = 4 * (kk >> 6) + ((kk >> 2) & 3);
    const int chunk8 = 2 * step + ((kk & 3) >> 1);
    const int elem = 2 * ((kk >> 4) & 3) + (kk & 1);
    unsigned char* base = p.xb + static_cast<size_t>(S.job_xo[j]) * 65536
                          + static_cast<size_t>(st) * NTT * 4096 + chunk8 * 128
                          + (2 * t4) * 16 + elem * 2;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
        const int tok = 8 * nt + 2 * t4;
        const float g0 = bf16r(acc[4 * nt] * gsa);
        const float g1 = bf16r(acc[4 * nt + 1] * gsa);
        const float u0 = bf16r(acc[4 * nt + 2] * gsa);
        const float u1 = bf16r(acc[4 * nt + 3] * gsa);
        const float h0 = g0 * silu_rcp<fast_silu(MODE)>(1.f + __expf(-g0)) * u0;
        const float h1 = g1 * silu_rcp<fast_silu(MODE)>(1.f + __expf(-g1)) * u1;
        if constexpr (pair_st(MODE)) {
            const bool odd = (g & 1) != 0;
            const float recv = __shfl_xor_sync(0xffffffffu, odd ? h0 : h1, 4);
            if ((odd ? tok + 1 : tok) < ntok) {
                 
                *reinterpret_cast<__nv_bfloat162*>(base + nt * 4096 + (odd ? 14 : 0)) =
                    __floats2bfloat162_rn(odd ? recv : h0, odd ? h1 : recv);
            }
            continue;
        }
        __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(base + nt * 4096);
        if (tok < ntok) {
            d[0] = __float2bfloat16_rn(h0);
        }
        if (tok + 1 < ntok) {
            d[8] = __float2bfloat16_rn(h1);
        }
    }
    if constexpr (PUB) {
        publish_a<MODE>(p, j * 16 + st, wg, tid_wg);
    }
    return j * 16 + st;
}

__device__ __forceinline__ int ld_volatile_s32(const int* p) {
    int v;
    asm volatile("ld.volatile.shared.s32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
    return v;
}

__device__ __forceinline__ float* ld_volatile_ptr(float* const* p) {
    unsigned long long v;
    asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(v) : "r"(smem_u32(p)) : "memory");
    return reinterpret_cast<float*>(v);
}

template <int NTT, bool VOL = false, bool NOSS = false, bool NOSS_BY_M = false>
__device__ __forceinline__ void epilogue_b(const Params& p, SmemMain& S, float* ss_arg,
                                           const float (&acc)[32], int j, int u, int wg,
                                           int w, int tid_wg) {
    float* ss = VOL ? ld_volatile_ptr(&S.ssp) : ss_arg;
    const int lane = tid_wg & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int ntok = S.job_n[j];
    const int q0 = S.job_q[j];
    const float gsb = __ldg(p.gs2 + S.job_e[j]) * 128.f;
    const int rb = 16 * u + 4 * wg + w;
    const int col0 = rb * 16 + g;
    const int col1 = col0 + 8;
    const float w0 = __ldg(p.nw + col0);
    const float w1 = __ldg(p.nw + col1);
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
        const int tok = 8 * nt + 2 * t4;
        const float y00 = bf16r(acc[4 * nt] * gsb) * w0;
        const float y01 = bf16r(acc[4 * nt + 1] * gsb) * w0;
        const float y10 = bf16r(acc[4 * nt + 2] * gsb) * w1;
        const float y11 = bf16r(acc[4 * nt + 3] * gsb) * w1;
        float* dst = p.ybuf + static_cast<size_t>(q0 + tok) * kH;
        if (tok < ntok) {
            dst[col0] = y00;
            dst[col1] = y10;
        }
        if (tok + 1 < ntok) {
            dst[kH + col0] = y01;
            dst[kH + col1] = y11;
        }
        if constexpr (NOSS) {
            if (!NOSS_BY_M || p.M >= kFinssMinM) {
                continue;
            }
        }
        float s0 = y00 * y00 + y10 * y10;
        float s1 = y01 * y01 + y11 * y11;
#pragma unroll
        for (int d = 4; d < 32; d <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, d);
            s1 += __shfl_xor_sync(0xffffffffu, s1, d);
        }
        if (g == 0) {
            if (tok < ntok) {
                atomicAdd(ss + q0 + tok, s0);
            }
            if (tok + 1 < ntok) {
                atomicAdd(ss + q0 + tok + 1, s1);
            }
        }
    }
}

 
 
 
template <int NTT, bool AR>
__device__ __forceinline__ bool fixup(const Params& p, SmemMain& S, const float (&acc)[32],
                                      float (&fin)[32], int ph, int slot, int nst, int wg,
                                      int tid_wg, int G) {
    float4* fp = reinterpret_cast<float4*>(p.fix)
                 + (static_cast<size_t>((ph * G + slot) * 4 + wg) * 128 + tid_wg) * 8;
#pragma unroll
    for (int i = 0; i < NTT; ++i) {
        red_add_v4(fp + i, make_float4(acc[4 * i], acc[4 * i + 1], acc[4 * i + 2],
                                       acc[4 * i + 3]));
    }
    fence_gpu<AR>();
    named_sync(kBarWg + wg, 128);
    int* cp = p.fixcnt + (ph * G + slot) * 4 + wg;
    if (tid_wg == 0) {
        const int old = atom_add_acqrel(cp, nst);
        S.fix_last[wg] = old + nst == 16 ? 1 : 0;
    }
    named_sync(kBarWg + wg, 128);
    if (S.fix_last[wg] == 0) {
        return false;
    }
    fence_gpu<AR>();
#pragma unroll
    for (int i = 0; i < NTT; ++i) {
        const float4 v = __ldcg(fp + i);
        fin[4 * i] = v.x;
        fin[4 * i + 1] = v.y;
        fin[4 * i + 2] = v.z;
        fin[4 * i + 3] = v.w;
        fp[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (tid_wg == 0) {
        *cp = 0;
    }
    return true;
}

 
 
 
__device__ __forceinline__ int own_target(const SmemMain& S, int r, int cta, int G) {
    const int k = own_unit(r, cta, G);
    return S.ord[k >> 5] * 16 + ((k & 31) >> 1);
}

 
 
template <int NTT, int MODE>
__device__ __forceinline__ void epilogue_a_unit(const Params& p, SmemMain& S,
                                                const float (&acc)[32], const SegInfo& si,
                                                int wg, int w, int tid_wg) {
    static_assert(!defer_a(MODE) || (own_a(MODE) && dyn_b(MODE)), "defer_a needs owned units");
    epilogue_a<NTT, MODE, !defer_a(MODE)>(p, S, acc, si.j, si.u, wg, w, tid_wg);
}

 
 
template <int MODE, int PH, int NTT>
__device__ __forceinline__ void pa_record(SmemMain& S, int warp, int lane, unsigned dw,
                                          unsigned dx, long long pseg, long long pdep,
                                          long long ppub, int nst) {
    if constexpr (pa_diag(MODE) && PH == 0) {
        if (lane == 0) {
            long long* d = S.dg[warp];
            d[0] += dw;
            d[1] += dx;
            d[2] += pdep - pseg - ppub;
            d[3] += clock64() - pdep;
            d[4] += nst;
            d[5] += 1;
            d[6] += NTT;
            d[7] += ppub;
        }
    }
}

__device__ __forceinline__ long long* pb_row(const Params& p, int warp) {
    return reinterpret_cast<long long*>(p.inter) + gridDim.x * 18 * 8
           + (blockIdx.x * 16 + warp) * 8;
}

 
template <int MODE, int PH, int NTT>
__device__ __forceinline__ void pb_record(const Params& p, int warp, int lane, unsigned dw,
                                          unsigned dx, long long pseg, long long pdep, int nst) {
    if constexpr (pb_diag(MODE) && PH == 1) {
        if (lane == 0) {
            long long* d = pb_row(p, warp);
            d[0] += dw;
            d[1] += dx;
            d[2] += pdep - pseg;
            d[3] += clock64() - pdep;
            d[4] += nst;
            d[5] += 1;
            d[6] += NTT;
        }
    }
}

 
 
 
template <int NTT, int PH, int MODE>
__device__ __forceinline__ void run_segment(const Params& p, SmemMain& S, float* ss,
                                            const SegInfo& si, float (&acc)[32], Ring& rg,
                                            int& xrel, int G) {
    constexpr int ph = PH;
    constexpr int DEPTH = MODE == 5 ? 3 : 2;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wg = warp >> 2;
    const int w = warp & 3;
    const int tid_wg = tid & 127;
    const int g = lane >> 2;
     
    constexpr bool kDiag = is_diag(MODE);
    constexpr bool kPa = pa_diag(MODE) && (PH == 0 || pb_diag(MODE));
    unsigned dw = 0;
    unsigned dx = 0;
    const unsigned dbeg = kDiag ? static_cast<unsigned>(clock()) : 0u;
    const long long pseg = kPa ? clock64() : 0;
    long long ppub = 0;
    unsigned af[DEPTH][2][4];
#pragma unroll
    for (int i = 0; i < 4 * NTT; ++i) {
        acc[i] = 0.f;
    }
    acc_fence<NTT>(acc);
    constexpr bool kLean = lean_stage(MODE);
     
     
    const unsigned xb0 = kLean ? __shfl_sync(0xffffffffu, rg.xb, 0) : 0u;
    const uint64_t dx0 = ((xb0 & 0x3ffffu) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16)
                         | (static_cast<uint64_t>(4096 >> 4) << 32);
    if constexpr (grp_codes(MODE)) {
        static_assert(kLean && DEPTH == 2 && !defer_a(MODE), "grp_codes extends mode 37");
         
         
        const unsigned char* wbase = S.rings + (wg * 4 + w) * 2048 + lane * 16;
        const unsigned char* sbase = S.rings + 32768 + (wg * 4 + w) * 256 + lane * 8;
        uint4 cb[2];
        unsigned scw[2];
        int ws = rg.ws;
        int xs = rg.xs;
        const uint4* cp = reinterpret_cast<const uint4*>(wbase + ws * kUnitBytes);
        const unsigned* sp = reinterpret_cast<const unsigned*>(sbase + ws * kUnitBytes);
        int sq = skew_st(MODE) ? rg.wlap * rg.nsw + ws : 0;
        if (si.st_b < si.st_e) {
            sk_stamp<MODE>(p, sq, warp * 4, lane == 0);
            dw += mbar_wait_m<MODE, PH>(&S.wfull[ws], rg.wlap & 1, p.ctr, 2);
            sk_stamp<MODE>(p, sq, warp * 4 + 1, lane == 0);
            cb[0] = cp[0];
            scw[0] = sp[0];
            ring_next_w(rg, rg.nsw);
            dx += mbar_wait_m<MODE, PH>(&S.xfull[xs], rg.xlap & 1, p.ctr, 8);
            sk_stamp<MODE>(p, sq, warp * 4 + 2, lane == 0);
        }
#pragma unroll 1
        for (int st = si.st_b; st < si.st_e; ++st) {
            if (!grp_hoist(MODE) && st > si.st_b) {
                ws = rg.ws;
                if constexpr (skew_st(MODE)) {
                    sq = rg.wlap * rg.nsw + ws;
                }
                sk_stamp<MODE>(p, sq, warp * 4, lane == 0);
                dw += mbar_wait_m<MODE, PH>(&S.wfull[ws], rg.wlap & 1, p.ctr, 2);
                sk_stamp<MODE>(p, sq, warp * 4 + 1, lane == 0);
                cp = reinterpret_cast<const uint4*>(wbase + ws * kUnitBytes);
                sp = reinterpret_cast<const unsigned*>(sbase + ws * kUnitBytes);
                cb[0] = cp[0];
                scw[0] = sp[0];
                ring_next_w(rg, rg.nsw);
                dx += mbar_wait_m<MODE, PH>(&S.xfull[xs], rg.xlap & 1, p.ctr, 8);
                sk_stamp<MODE>(p, sq, warp * 4 + 2, lane == 0);
            }
            const uint64_t d0 = dx0 + (__shfl_sync(0xffffffffu,
                                                   static_cast<unsigned>(xs) * rg.xsz, 0) >> 4);
#pragma unroll
            for (int s = 0; s < 16; s += 2) {
                const int c = s >> 2;
                const int b = (s >> 1) % DEPTH;
                if ((s & 3) == 0 && c < 3) {
                    cb[(c + 1) & 1] = cp[(c + 1) * 32];
                    if (c == 1) {
                        scw[1] = sp[1];
                    }
                }
                {
                    const unsigned sg = scale_word(scw[c >> 1], (2 * c) & 3);
                    const unsigned sg8 = scale_word(scw[c >> 1], (2 * c + 1) & 3);
                    dequant(word(cb[c & 1], s & 3), sg, sg8, af[b][0]);
                    dequant(word(cb[c & 1], (s + 1) & 3), sg, sg8, af[b][1]);
                }
                wg_fence();
                wgmma<NTT>(acc, af[b][0], d0 + s * 16);
                wgmma<NTT>(acc, af[b][1], d0 + (s + 1) * 16);
                wg_commit();
                wg_wait<DEPTH - 1>();
                a_pin(af[(b + 1) % DEPTH]);
                if (s == 0 && xrel >= 0) {
                    x_release_warp<MODE>(&S.xempty[xrel], lane);
                    xrel = -1;
                }
                if (s == 10) {
                     
                     
                     
                     
                    fence_proxy_async();
                    mbar_arrive(&S.wempty[ws]);
                    sk_stamp<MODE>(p, sq, warp * 4 + 3, lane == 0);
                }
                if (grp_hoist(MODE) && s == 14 && st + 1 < si.st_e) {
                    ws = rg.ws;
                    dw += mbar_wait_m<MODE, PH>(&S.wfull[ws], rg.wlap & 1, p.ctr, 2);
                    cp = reinterpret_cast<const uint4*>(wbase + ws * kUnitBytes);
                    sp = reinterpret_cast<const unsigned*>(sbase + ws * kUnitBytes);
                    cb[0] = cp[0];
                    scw[0] = sp[0];
                    ring_next_w(rg, rg.nsw);
                    ring_next_x(rg);
                    dx += mbar_wait_m<MODE, PH>(&S.xfull[rg.xs], rg.xlap & 1, p.ctr, 8);
                }
            }
            xrel = xs;
            if (!grp_hoist(MODE) || st + 1 == si.st_e) {
                ring_next_x(rg);
            }
            xs = rg.xs;
        }
    } else {
#pragma unroll 1
        for (int st = si.st_b; st < si.st_e; ++st) {
            if constexpr (defer_a(MODE) && PH == 0) {
                 
                if (st == si.st_b + 1 && si.r > 0) {
                    const long long c = kPa ? clock64() : 0;
                    publish_a<MODE>(p, own_target(S, si.r - 1, blockIdx.x, gridDim.x), wg, tid_wg);
                    if constexpr (kPa) {
                        ppub += clock64() - c;
                    }
                }
            }
            const int ws = rg.ws;
            unsigned dc = kDiag ? static_cast<unsigned>(clock()) : 0u;
            dw += mbar_wait_m<MODE, PH>(&S.wfull[ws], rg.wlap & 1, p.ctr, 2);
            if constexpr (kDiag) {
                dw += static_cast<unsigned>(clock()) - dc;
            }
            const unsigned char* wslot = S.rings + ws * kUnitBytes;
            const uint4* cp = reinterpret_cast<const uint4*>(wslot + (wg * 4 + w) * 2048) + lane;
            uint4 cw[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                cw[i] = cp[i * 32];
            }
            const uint2 sc = *reinterpret_cast<const uint2*>(wslot + 32768 + (wg * 4 + w) * 256
                                                             + lane * 8);
            if constexpr (!kLean) {
                __syncwarp();
                if (lane == 0) {
                    fence_proxy_async();
                    mbar_arrive(&S.wempty[ws]);
                }
            }
            ring_next_w(rg, kLean ? rg.nsw : S.nsw);
            const int xs = rg.xs;
            if constexpr (kDiag) {
                dc = static_cast<unsigned>(clock());
            }
            dx += mbar_wait_m<MODE, PH>(&S.xfull[xs], rg.xlap & 1, p.ctr, 8);
            if constexpr (kDiag) {
                dx += static_cast<unsigned>(clock()) - dc;
            }
             
             
             
            const unsigned xbase = kLean ? 0u
                                         : __shfl_sync(0xffffffffu,
                                                       smem_u32(S.rings + S.xoff + xs * S.xsz), 0);
             
             
             
            const uint64_t d0 = kLean ? dx0 + (__shfl_sync(0xffffffffu,
                                                           static_cast<unsigned>(xs) * rg.xsz, 0)
                                               >> 4)
                                      : ((xbase & 0x3ffffu) >> 4)
                                            | (static_cast<uint64_t>(128 >> 4) << 16)
                                            | (static_cast<uint64_t>(4096 >> 4) << 32);
            if (MODE == 1) {
                 
                acc[0] += __uint_as_float((cw[0].x ^ sc.x) & 0x3f800000u);
                if (xrel >= 0) {
                    if (lane == 0) {
                        mbar_arrive(&S.xempty[xrel]);
                    }
                }
                xrel = xs;
                ring_next_x(rg);
                continue;
            }
#pragma unroll
            for (int s = 0; s < 16; s += 2) {
                const int b = (s >> 1) % DEPTH;
                {
                    const int c = s >> 2;
                    const unsigned scw = c < 2 ? sc.x : sc.y;
                    const unsigned sg = scale_word(scw, (2 * c) & 3);
                    const unsigned sg8 = scale_word(scw, (2 * c + 1) & 3);
                    if (MODE == 3) {
                         
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            af[b][0][i] = word(cw[c], i) ^ sg;
                            af[b][1][i] = word(cw[c], i) ^ sg8;
                        }
                    } else {
                        dequant(word(cw[c], s & 3), sg, sg8, af[b][0]);
                        dequant(word(cw[c], (s + 1) & 3), sg, sg8, af[b][1]);
                    }
                }
                if (MODE == 2) {
                     
                    acc[0] += __uint_as_float((af[b][0][0] ^ af[b][1][3]) & 0x3f800000u);
                    if (s == 2 * (DEPTH - 1) - 2 && xrel >= 0) {
                        if (lane == 0) {
                            mbar_arrive(&S.xempty[xrel]);
                        }
                        xrel = -1;
                    }
                    continue;
                }
                wg_fence();
                wgmma<NTT>(acc, af[b][0], d0 + s * 16);
                wgmma<NTT>(acc, af[b][1], d0 + (s + 1) * 16);
                wg_commit();
                wg_wait<DEPTH - 1>();
                a_pin(af[(b + 1) % DEPTH]);
                 
                 
                if (s == 2 * (DEPTH - 1) - 2 && xrel >= 0) {
                    x_release_warp<MODE>(&S.xempty[xrel], lane);
                    xrel = -1;
                }
                if (kLean && s == 0) {
                     
                     
                    fence_proxy_async();
                    mbar_arrive(&S.wempty[ws]);
                }
            }
            xrel = xs;
            ring_next_x(rg);
        }
    }
    wg_wait<0>();
#pragma unroll
    for (int b = 0; b < DEPTH; ++b) {
        a_pin(af[b]);
    }
    acc_fence<NTT>(acc);
    if (xrel >= 0) {
        x_release_warp<MODE>(&S.xempty[xrel], lane);
        xrel = -1;
    }
    if (MODE == 1) {
        return;
    }
    const unsigned dep = kDiag ? static_cast<unsigned>(clock()) : 0u;
    const long long pdep = kPa ? clock64() : 0;
    if (si.partial) {
        float fin[32];
        if (fixup<NTT, fence_ar(MODE)>(p, S, acc, fin, ph, si.slot, si.st_e - si.st_b, wg, tid_wg, G)) {
            if (ph == 0) {
                epilogue_a<NTT, MODE, true>(p, S, fin, si.j, si.u, wg, w, tid_wg);
            } else {
                epilogue_b<NTT, false, final_ss(MODE), MODE == 51>(p, S, ss, fin, si.j, si.u,
                                                                   wg, w, tid_wg);
            }
        }
    } else if (ph == 0) {
        epilogue_a_unit<NTT, MODE>(p, S, acc, si, wg, w, tid_wg);
    } else {
        epilogue_b<NTT, vol_regs(MODE), final_ss(MODE), MODE == 51>(p, S, ss, acc, si.j, si.u,
                                                                    wg, w, tid_wg);
    }
    pa_record<MODE, PH, NTT>(S, warp, lane, dw, dx, pseg, pdep, ppub, si.st_e - si.st_b);
    pb_record<MODE, PH, NTT>(p, warp, lane, dw, dx, pseg, pdep, si.st_e - si.st_b);
    if constexpr (kDiag) {
        if (lane == 0) {
            long long* d = S.dg[warp];
            d[0] += dw;
            d[1] += dx;
            d[2] += dep - dbeg - dw - dx;
            d[3] += static_cast<unsigned>(clock()) - dep;
            d[4] += si.st_e - si.st_b;
            d[5] += 1;
        }
    }
}

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

 
 
 
 
 
 
template <int NTT, int PH, int MODE>
__device__ __forceinline__ void run_segment_mma(const Params& p, SmemMain& S, float* ss,
                                                const SegInfo& si, Ring& rg, int G) {
    constexpr int ph = PH;
    constexpr bool kDiag = is_diag(MODE);
    constexpr bool kPa = pa_diag(MODE) && (PH == 0 || pb_diag(MODE));
    const long long pseg = kPa ? clock64() : 0;
    long long ppub = 0;
    unsigned dw = 0;
    unsigned dx = 0;
    long long tw = 0;
    long long tx = 0;
    long long tc = 0;
    long long c0 = 0;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wg = warp >> 2;
    const int w = warp & 3;
    const int tid_wg = tid & 127;
     
     
     
    constexpr bool kDual = dual_chain(MODE);
    float acc[32];
    float acc2[kDual ? 4 * NTT : 1];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        acc[i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < (kDual ? 4 * NTT : 1); ++i) {
        acc2[i] = 0.f;
    }
    const unsigned lrow = static_cast<unsigned>((lane >> 3) * 128 + (lane & 7) * 16);
    constexpr bool kLean = lean_stage(MODE);
    const unsigned xb0 = kLean ? rg.xb + lrow : 0u;
#pragma unroll 1
    for (int st = si.st_b; st < si.st_e; ++st) {
        if constexpr (defer_a(MODE) && PH == 0) {
            if (st == si.st_b + 1 && si.r > 0) {
                const long long c = kPa ? clock64() : 0;
                publish_a<MODE>(p, own_target(S, si.r - 1, blockIdx.x, gridDim.x), wg, tid_wg);
                if constexpr (kPa) {
                    ppub += clock64() - c;
                }
            }
        }
        const int ws = rg.ws;
        if constexpr (kDiag) {
            c0 = clock64();
        }
        dw += mbar_wait_m<MODE, PH>(&S.wfull[ws], rg.wlap & 1, p.ctr, 2);
        if constexpr (kDiag) {
            const long long c1 = clock64();
            tw += c1 - c0;
            c0 = c1;
        }
        const unsigned char* wslot = S.rings + ws * kUnitBytes;
        const uint4* cp = reinterpret_cast<const uint4*>(wslot + (wg * 4 + w) * 2048) + lane;
        uint4 cw[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            cw[i] = cp[i * 32];
        }
        const uint2 sc = *reinterpret_cast<const uint2*>(wslot + 32768 + (wg * 4 + w) * 256
                                                         + lane * 8);
        if constexpr (!kLean) {
            __syncwarp();
            if (lane == 0) {
                fence_proxy_async();
                mbar_arrive(&S.wempty[ws]);
            }
        }
        ring_next_w(rg, kLean ? rg.nsw : S.nsw);
        const int xs = rg.xs;
        if constexpr (kDiag) {
            const long long c1 = clock64();
            tc += c1 - c0;
            c0 = c1;
        }
        dx += mbar_wait_m<MODE, PH>(&S.xfull[xs], rg.xlap & 1, p.ctr, 8);
        if constexpr (kDiag) {
            const long long c1 = clock64();
            tx += c1 - c0;
            c0 = c1;
        }
        const unsigned xbase = kLean ? xb0 + static_cast<unsigned>(xs) * rg.xsz
                                     : __shfl_sync(0xffffffffu,
                                                   smem_u32(S.rings + S.xoff + xs * S.xsz), 0)
                                           + lrow;
#pragma unroll
        for (int s = 0; s < 16; s += 2) {
            const int c = s >> 2;
            const unsigned scw = c < 2 ? sc.x : sc.y;
            const unsigned sg = scale_word(scw, (2 * c) & 3);
            const unsigned sg8 = scale_word(scw, (2 * c + 1) & 3);
            unsigned a0[4];
            unsigned a1[4];
            dequant(word(cw[c], s & 3), sg, sg8, a0);
            dequant(word(cw[c], (s + 1) & 3), sg, sg8, a1);
#pragma unroll
            for (int nt = 0; nt < NTT; ++nt) {
                unsigned b[4];
                ldsm_x4(b, xbase + nt * 4096 + s * 256);
                float (&c4)[4] = *reinterpret_cast<float (*)[4]>(acc + 4 * nt);
                mma_bf16(c4, a0, b[0], b[1]);
                if constexpr (kDual) {
                    float (&d4)[4] = *reinterpret_cast<float (*)[4]>(acc2 + 4 * nt);
                    mma_bf16(d4, a1, b[2], b[3]);
                } else {
                    mma_bf16(c4, a1, b[2], b[3]);
                }
            }
            if (kLean && s == 0) {
                 
                fence_proxy_async();
                mbar_arrive(&S.wempty[ws]);
            }
        }
         
        if constexpr (kLean) {
            fence_proxy_async();
            mbar_arrive(&S.xempty[xs]);
        } else {
            __syncwarp();
            if (lane == 0) {
                fence_proxy_async();
                mbar_arrive(&S.xempty[xs]);
            }
        }
        ring_next_x(rg);
        if constexpr (kDiag) {
             
            const long long c1 = clock64() + (acc[0] == 1e30f ? 1 : 0);
            tc += c1 - c0;
        }
    }
    if constexpr (kDual) {
#pragma unroll
        for (int i = 0; i < 4 * NTT; ++i) {
            acc[i] += acc2[i];
        }
    }
    if constexpr (kDiag) {
        c0 = clock64();
    }
    const long long pdep = kPa ? clock64() + (acc[0] == 1e30f ? 1 : 0) : 0;
    if (si.partial) {
        float fin[32];
        if (fixup<NTT, fence_ar(MODE)>(p, S, acc, fin, ph, si.slot, si.st_e - si.st_b, wg, tid_wg, G)) {
            if (ph == 0) {
                epilogue_a<NTT, MODE, true>(p, S, fin, si.j, si.u, wg, w, tid_wg);
            } else {
                epilogue_b<NTT, false, final_ss(MODE), MODE == 51>(p, S, ss, fin, si.j, si.u,
                                                                   wg, w, tid_wg);
            }
        }
    } else if (ph == 0) {
        epilogue_a_unit<NTT, MODE>(p, S, acc, si, wg, w, tid_wg);
    } else {
        epilogue_b<NTT, vol_regs(MODE), final_ss(MODE), MODE == 51>(p, S, ss, acc, si.j, si.u,
                                                                    wg, w, tid_wg);
    }
    pa_record<MODE, PH, NTT>(S, warp, lane, dw, dx, pseg, pdep, ppub, si.st_e - si.st_b);
    pb_record<MODE, PH, NTT>(p, warp, lane, dw, dx, pseg, pdep, si.st_e - si.st_b);
    if constexpr (kDiag) {
        if (lane == 0) {
            long long* d = S.dg[warp];
            d[0] += tw;
            d[1] += tx;
            d[2] += tc;
            d[3] += clock64() - c0;
            d[4] += si.st_e - si.st_b;
            d[5] += 1;
        }
    }
}

 
 
 
template <int NTT, int PH, int MODE>
__device__ __forceinline__ void run_group(const Params& p, SmemMain& S, float* ss,
                                          float (&acc)[32], Ring& rg, int& xrel, int cta,
                                          int G) {
    const int ns = seg_n<MODE>(S, PH);
#pragma unroll 1
    for (int i = 0; i < (vol_regs(MODE) && own_a(MODE) && PH == 0 ? ld_volatile_s32(&S.own_n)
                                                                 : ns); ++i) {
        int gb;
        int ge;
        seg_at<MODE>(S, PH, i, cta, G, gb, ge);
        if (S.job_ntt[PH ? (gb >> 8) : (gb >> 9)] != NTT) {
            continue;
        }
        SegInfo si = seg_info(S, PH, gb, ge, cta, G);
        if (own_a(MODE) && PH == 0) {
            si.r = i;
        }
        if (is_diag(MODE) && threadIdx.x == 0) {
            S.dg[PH * 8 + NTT - 1][6] += si.st_e - si.st_b;
            S.dg[PH * 8 + NTT - 1][7] += 1;
            S.dg[16][1 + PH] += si.partial ? 1 : 0;
        }
         
        const long long cs = is_diag(MODE) ? clock64() : 0;
        if constexpr (NTT <= mma_max(MODE)) {
            run_segment_mma<NTT, PH, MODE>(p, S, ss, si, rg, G);
        } else {
            run_segment<NTT, PH, MODE>(p, S, ss, si, acc, rg, xrel, G);
        }
        if (is_diag(MODE) && threadIdx.x == 0) {
            S.dg[17][5 + PH] += clock64() - cs;
        }
    }
}

template <int PH, int MODE>
__device__ __forceinline__ void consumer_phase(const Params& p, SmemMain& S, float* ss,
                                               float (&acc)[32], Ring& rg, int& xrel, int cta,
                                               int G) {
    if constexpr (dyn_b(MODE)) {
        run_group<8, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<7, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<6, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<5, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<4, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<3, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<2, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<1, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
    } else {
        run_group<1, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<2, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<3, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<4, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<5, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<6, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<7, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
        run_group<8, PH, MODE>(p, S, ss, acc, rg, xrel, cta, G);
    }
}

 
 
template <int NTT, int MODE>
__device__ __forceinline__ void run_group_dyn(const Params& p, SmemMain& S, float* ss,
                                              float (&acc)[32], Ring& rg, int& xrel, int& qi,
                                              int cta, int G) {
#pragma unroll 1
    while (qi < kBqMax) {
        const long long cq = pb_diag(MODE) ? clock64() : 0;
        const int m = bq_get(S, qi, p.ctr);
        if (pb_diag(MODE) && (threadIdx.x & 31) == 0) {
            pb_row(p, threadIdx.x >> 5)[7] += clock64() - cq;
        }
        if (m < 0) {
            return;
        }
        const int j = S.ord[m >> 4];
        if (S.job_ntt[j] != NTT) {
            return;
        }
        SegInfo si;
        si.j = j;
        si.u = m & 15;
        si.st_b = 0;
        si.st_e = 16;
        si.slot = cta;
        si.partial = false;
        si.r = -1;
        if (is_diag(MODE) && threadIdx.x == 0) {
            S.dg[8 + NTT - 1][6] += 16;
            S.dg[8 + NTT - 1][7] += 1;
        }
        const long long cs = is_diag(MODE) ? clock64() : 0;
        if constexpr (NTT <= mma_max(MODE)) {
            run_segment_mma<NTT, 1, MODE>(p, S, ss, si, rg, G);
        } else {
            run_segment<NTT, 1, MODE>(p, S, ss, si, acc, rg, xrel, G);
        }
        if (is_diag(MODE) && threadIdx.x == 0) {
            S.dg[17][6] += clock64() - cs;
        }
        ++qi;
    }
}

template <int MODE>
__device__ __forceinline__ void consumer_dyn_b(const Params& p, SmemMain& S, float* ss,
                                               float (&acc)[32], Ring& rg, int& xrel, int cta,
                                               int G) {
    int qi = 0;
    run_group_dyn<8, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_dyn<7, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_dyn<6, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_dyn<5, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_dyn<4, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_dyn<3, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_dyn<2, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_dyn<1, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
}

template <int MODE>
__device__ __forceinline__ void consumers(const Params& p, SmemMain& S, float* ss, int cta,
                                          int G) {
    Ring rg = {0, 0, 0, 0, 0, 0u, 0u};
    if constexpr (lean_stage(MODE)) {
        rg.nsw = S.nsw;
        rg.xb = smem_u32(S.rings + S.xoff);
        rg.xsz = static_cast<unsigned>(S.xsz);
    }
    int xrel = -1;
    float acc[32];
    if (pb_diag(MODE) && (threadIdx.x & 31) == 0) {
         
        for (int c = 0; c < 8; ++c) {
            pb_row(p, threadIdx.x >> 5)[c] = 0;
        }
    }
    const long long pc0 = pa_diag(MODE) ? clock64() : 0;
    consumer_phase<0, MODE>(p, S, ss, acc, rg, xrel, cta, G);
    if constexpr (defer_a(MODE)) {
         
        if (S.own_n > 0) {
            const long long c = pa_diag(MODE) ? clock64() : 0;
            publish_a<MODE>(p, own_target(S, S.own_n - 1, cta, G), threadIdx.x >> 7,
                            threadIdx.x & 127);
            if (pa_diag(MODE) && (threadIdx.x & 31) == 0) {
                S.dg[threadIdx.x >> 5][7] += clock64() - c;
            }
        }
    }
    if (pa_diag(MODE) && threadIdx.x == 0) {
        S.dg[17][7] = clock64() - pc0;
    }
    if (is_diag(MODE) && threadIdx.x == 0) {
        S.dg[17][7] = gtimer();
    }
    if (is_light(MODE) && threadIdx.x == 0) {
        S.dg[16][3] = gtimer();
    }
    if constexpr (dyn_b(MODE)) {
        consumer_dyn_b<MODE>(p, S, ss, acc, rg, xrel, cta, G);
    } else if constexpr (!a_only(MODE)) {
         
         
        consumer_phase<1, MODE>(p, S, ss, acc, rg, xrel, cta, G);
    }
}

template <int MODE>
__device__ __forceinline__ void wproducer(const Params& p, SmemMain& S, int cta, int G,
                                          uint64_t pol) {
    long long twe = 0;
    long long pwe = 0;
    Ring rg = {0, 0, 0, 0};
    for (int ph = 0; ph < (dyn_b(MODE) || a_only(MODE) ? 1 : 2); ++ph) {
        const int ns = seg_n<MODE>(S, ph);
        for (int n = 1; n <= 8; ++n) {
        const int ntt = dyn_b(MODE) ? 9 - n : n;
        for (int i = 0; i < ns; ++i) {
            int gb;
            int ge;
            seg_at<MODE>(S, ph, i, cta, G, gb, ge);
            const int j = ph ? (gb >> 8) : (gb >> 9);
            if (S.job_ntt[j] != ntt) {
                continue;
            }
            const int u = (gb & (ph ? 255 : 511)) >> 4;
            const int e = S.job_e[j];
            const unsigned char* base =
                ph ? p.w2f + static_cast<size_t>(e * 16 + u) * 16 * kUnitBytes
                   : p.w13f + static_cast<size_t>(e * 32 + u) * 16 * kUnitBytes;
            const int st_e = ((ge - 1) & 15) + 1;
            for (int st = gb & 15; st < st_e; ++st) {
                const int ws = rg.ws;
                if (rg.wlap > 0) {
                    const long long c0 = is_diag(MODE) ? clock64() : 0;
                    pwe += mbar_wait_m<MODE, 0>(&S.wempty[ws], (rg.wlap - 1) & 1, p.ctr, 16);
                    if (is_diag(MODE)) {
                        twe += clock64() - c0;
                    }
                }
                sk_stamp<MODE>(p, rg.wlap * S.nsw + ws, kNC * 4, true);
                mbar_expect_tx(&S.wfull[ws], kUnitBytes);
                bulk_load(S.rings + ws * kUnitBytes, base + static_cast<size_t>(st) * kUnitBytes,
                          kUnitBytes, &S.wfull[ws], pol);
                ring_next_w(rg, S.nsw);
            }
        }
        }
    }
    if constexpr (pa_diag(MODE)) {
        S.dg[16][7] = pwe;
    }
    if constexpr (dyn_b(MODE)) {
         
         
         
         
        int* claim = p.ctr + 10;
        const int units = S.njobs * 16;
        for (int i = 0; i < kBqMax; ++i) {
            if (rg.wlap > 0) {
                const long long c0 = is_diag(MODE) ? clock64() : 0;
                mbar_wait_to(&S.wempty[rg.ws], (rg.wlap - 1) & 1, p.ctr, 16);
                if (is_diag(MODE)) {
                    twe += clock64() - c0;
                }
            }
            int m = i < kBqMax - 1 ? atomicAdd(claim, 1) : units;
            if (m >= units) {
                m = -1;
            }
            st_release_cta(&S.bq[i], m);
            if (m < 0) {
                break;
            }
            const int j = S.ord[m >> 4];
            const unsigned char* base =
                p.w2f + static_cast<size_t>(S.job_e[j] * 16 + (m & 15)) * 16 * kUnitBytes;
            for (int st = 0; st < 16; ++st) {
                const int ws = rg.ws;
                if (st > 0 && rg.wlap > 0) {
                    const long long c0 = is_diag(MODE) ? clock64() : 0;
                    mbar_wait_to(&S.wempty[ws], (rg.wlap - 1) & 1, p.ctr, 16);
                    if (is_diag(MODE)) {
                        twe += clock64() - c0;
                    }
                }
                sk_stamp<MODE>(p, rg.wlap * S.nsw + ws, kNC * 4, true);
                mbar_expect_tx(&S.wfull[ws], kUnitBytes);
                bulk_load(S.rings + ws * kUnitBytes, base + static_cast<size_t>(st) * kUnitBytes,
                          kUnitBytes, &S.wfull[ws], pol);
                ring_next_w(rg, S.nsw);
            }
        }
    }
    if (is_diag(MODE)) {
        S.dg[16][0] = twe;
    }
}

 
 
template <int MODE>
__device__ __forceinline__ void xproducer(const Params& p, SmemMain& S, int cta, int G) {
    Ring rg = {0, 0, 0, 0};
    long long txe = 0;
    long long tda = 0;
    long long txr = 0;
    long long pxe = 0;
    for (int ph = 0; ph < (dyn_b(MODE) || a_only(MODE) ? 1 : 2); ++ph) {
        const int ns = seg_n<MODE>(S, ph);
        for (int n = 1; n <= 8; ++n) {
        const int ntt = dyn_b(MODE) ? 9 - n : n;
        for (int i = 0; i < ns; ++i) {
            int gb;
            int ge;
            seg_at<MODE>(S, ph, i, cta, G, gb, ge);
            const int j = ph ? (gb >> 8) : (gb >> 9);
            if (S.job_ntt[j] != ntt) {
                continue;
            }
            constexpr bool kDiag = is_diag(MODE);
            const unsigned bytes = ntt * 4096;
            const unsigned char* base = (ph ? p.xb : p.xa)
                                        + static_cast<size_t>(S.job_xo[j]) * 65536;
            long long c0 = kDiag ? clock64() : 0;
            if (ph == 0 && MODE != 1 && MODE != 4) {
                const long long c = pa_diag(MODE) ? clock64() : 0;
                spin_ge(p.xready + j, S.job_n[j], p.ctr, 32);
                if (pa_diag(MODE)) {
                    pxe += clock64() - c;
                }
                fence_proxy_async_global();
            }
            if (kDiag) {
                txr += clock64() - c0;
            }
            const int st_e = ((ge - 1) & 15) + 1;
            for (int st = gb & 15; st < st_e; ++st) {
                const int xs = rg.xs;
                if (kDiag) {
                    c0 = clock64();
                }
                if (rg.xlap > 0) {
                    pxe += mbar_wait_m<MODE, 0>(&S.xempty[xs], (rg.xlap - 1) & 1, p.ctr, 64);
                }
                if (kDiag) {
                    const long long c1 = clock64();
                    txe += c1 - c0;
                    c0 = c1;
                }
                if (ph == 1 && MODE != 1 && MODE != 4) {
                    if (!b_only(MODE)) {
                        spin_ge(p.done_as + j * 16 + st, 8, p.ctr, 128);
                    }
                    fence_proxy_async_global();
                }
                if (kDiag) {
                    tda += clock64() - c0;
                }
                sk_stamp<MODE>(p, rg.xlap * kNSX + xs, kNC * 4 + 1, true);
                mbar_expect_tx(&S.xfull[xs], bytes);
                bulk_load_plain(S.rings + S.xoff + xs * S.xsz,
                                base + static_cast<size_t>(st) * bytes, bytes, &S.xfull[xs]);
                ring_next_x(rg);
            }
        }
        }
    }
    if constexpr (pa_diag(MODE)) {
        S.dg[17][6] = pxe;
    }
    if constexpr (dyn_b(MODE)) {
        for (int i = 0; i < kBqMax; ++i) {
            const int m = bq_get(S, i, p.ctr);
            if (m < 0) {
                break;
            }
            const int j = S.ord[m >> 4];
            const unsigned bytes = S.job_ntt[j] * 4096;
            const unsigned char* base = p.xb + static_cast<size_t>(S.job_xo[j]) * 65536;
            for (int st = 0; st < 16; ++st) {
                const int xs = rg.xs;
                long long c0 = is_diag(MODE) ? clock64() : 0;
                if (rg.xlap > 0) {
                    mbar_wait_to(&S.xempty[xs], (rg.xlap - 1) & 1, p.ctr, 64);
                }
                if (is_diag(MODE)) {
                    const long long c1 = clock64();
                    txe += c1 - c0;
                    c0 = c1;
                }
                if (!b_only(MODE)) {
                    spin_ge(p.done_as + j * 16 + st, 8, p.ctr, 128);
                }
                fence_proxy_async_global();
                if (is_diag(MODE)) {
                    tda += clock64() - c0;
                }
                sk_stamp<MODE>(p, rg.xlap * kNSX + xs, kNC * 4 + 1, true);
                mbar_expect_tx(&S.xfull[xs], bytes);
                bulk_load_plain(S.rings + S.xoff + xs * S.xsz,
                                base + static_cast<size_t>(st) * bytes, bytes, &S.xfull[xs]);
                ring_next_x(rg);
            }
        }
    }
    if (is_diag(MODE)) {
        S.dg[17][0] = txe;
        S.dg[17][1] = tda;
        S.dg[17][2] = txr;
    }
}

template <int MODE>
__global__ void __launch_bounds__(kNTM, 1) moe_main_kernel(const Params p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    SmemMain& S = *reinterpret_cast<SmemMain*>(smem_raw);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int G = gridDim.x;
    const int cta = blockIdx.x;
    const int E = p.M * kTopK;
    if (p.pdl) {
        if (warp == kNWarpsM - 2 && lane == 0) {
            constexpr unsigned kRelP = lean_stage(MODE) ? kNC * 32 : kNC;
#pragma unroll
            for (int s = 0; s < kNSWMax; ++s) {
                mbar_init(&S.wfull[s], 1);
                mbar_init(&S.wempty[s], kRelP);
            }
#pragma unroll
            for (int s = 0; s < kNSX; ++s) {
                mbar_init(&S.xfull[s], 1);
                mbar_init(&S.xempty[s], kRelP);
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        if (tid == 0) {
            for (int i = cta; i < p.nloc * 16; i += G) {
                prefetch_l2_hint(p.w13f + (static_cast<size_t>(i % p.nloc) * 32 * 16
                                           + i / p.nloc) * kUnitBytes, kUnitBytes);
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    constexpr bool kDiag = is_diag(MODE);
    constexpr bool kLight = is_light(MODE);
    long long dt[5] = {0, 0, 0, 0, 0};
    long long dc0 = 0;
    if (pa_diag(MODE) && tid < 16 * 8) {
        (&S.dg[0][0])[tid] = 0;
    }
    if (kLight && tid == 0) {
        S.dg[16][0] = gtimer();
    }
    if constexpr (kDiag) {
        dt[0] = gtimer();
        dc0 = clock64();
        if (tid < 18 * 8) {
            (&S.dg[0][0])[tid] = 0;
        }
    }
    const int gen = *reinterpret_cast<volatile int*>(p.ctr + 3);
    float* ss = p.ss + (gen & 1) * kEmax;
    if (vol_regs(MODE) && tid == 0) {
        S.ssp = ss;
    }
    const uint64_t pol = policy_evict_first();

     
     
     
    {
        constexpr int kLd = (kEmax / 4 + kNTM - 1) / kNTM;
        int4 v[kLd];
#pragma unroll
        for (int k = 0; k < kLd; ++k) {
            const int i4 = tid + k * kNTM;
            v[k] = i4 < (E >> 2) ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4)
                                 : make_int4(-1, -1, -1, -1);
        }
#pragma unroll
        for (int k = 0; k < kLd; ++k) {
            const int i4 = tid + k * kNTM;
            if (i4 < (E >> 2)) {
                const int lo = ld_lo(p.lo_p);
                const int ev[4] = {v[k].x - lo, v[k].y - lo, v[k].z - lo, v[k].w - lo};
                char4 c;
                c.x = static_cast<signed char>(ev[0] >= 0 && ev[0] < p.nloc ? ev[0] : -1);
                c.y = static_cast<signed char>(ev[1] >= 0 && ev[1] < p.nloc ? ev[1] : -1);
                c.z = static_cast<signed char>(ev[2] >= 0 && ev[2] < p.nloc ? ev[2] : -1);
                c.w = static_cast<signed char>(ev[3] >= 0 && ev[3] < p.nloc ? ev[3] : -1);
                *reinterpret_cast<char4*>(S.eid + 4 * i4) = c;
            }
        }
    }
    __syncthreads();
    if (kLight && tid == 0) {
        S.dg[17][0] = gtimer();
    }
    if constexpr (fast_setup(MODE)) {
        position_count<kNWarpsM>(S.eid, S.wc, E);
        __syncthreads();
    }

    if (warp == kNWarpsM - 1) {
        if (lane < 9) {
            const bool given = p.cost[0][1] > 0;
            S.cost[0][lane] = given ? p.cost[0][lane] : stage_cost<MODE>(0, lane);
            S.align_b = MODE == 15 || (MODE >= 16 && MODE <= 29) || (MODE >= 31 && MODE <= 51)
                        ? 1 : 0;
            S.cost[1][lane] = given ? p.cost[1][lane] : stage_cost<MODE>(1, lane);
        }
        if constexpr (fast_setup(MODE)) {
            int c = 0;
#pragma unroll
            for (int w = 0; w < kNWarpsM; ++w) {
                c += S.wc[w][lane];
            }
            S.cnt[lane] = c;
        } else {
            S.cnt[lane] = 0;
            __syncwarp();
            for (int base = 0; base < E; base += 128) {
                const int i = base + 4 * lane;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int eid = i < E ? S.eid[i + k] : -1;
                    const unsigned peers = __match_any_sync(0xffffffffu, eid);
                    if (eid >= 0 && lane == __ffs(peers) - 1) {
                        S.cnt[eid] += __popc(peers);
                    }
                    __syncwarp();
                }
            }
        }
        const int tot = lane < p.nloc ? S.cnt[lane] : 0;
        const int off = warp_scan(tot, lane) - tot;
        S.off[lane] = off;
        if (kLight && lane == 0) {
            S.dg[17][3] = gtimer();
        }
        if constexpr (fast_setup(MODE)) {
             
            int run = off;
#pragma unroll
            for (int w = 0; w < kNWarpsM; ++w) {
                const int cw = S.wc[w][lane];
                S.wc[w][lane] = run;
                run += cw;
            }
        }
        const int nj = (tot + 63) >> 6;
        const int jincl = warp_scan(nj, lane);
        const int j0 = jincl - nj;
        S.job0[lane] = j0;
        int xo_sum = 0;
        int ca_sum = 0;
        int cb_sum = 0;
        for (int k = 0; k < nj; ++k) {
            const int ntt = (min(64, tot - 64 * k) + 7) >> 3;
            xo_sum += ntt;
            ca_sum += kUAM * S.cost[0][ntt];
            cb_sum += kUBM * S.cost[1][ntt];
        }
        const int xo_incl = warp_scan(xo_sum, lane);
        const int ca_incl = warp_scan(ca_sum, lane);
        const int cb_incl = warp_scan(cb_sum, lane);
        int xo = xo_incl - xo_sum;
        int ca = ca_incl - ca_sum;
        int cb = cb_incl - cb_sum;
        for (int k = 0; k < nj; ++k) {
            const int j = j0 + k;
            const int n = min(64, tot - 64 * k);
            const int ntt = (n + 7) >> 3;
            S.job_e[j] = lane;
            S.job_q[j] = off + 64 * k;
            S.job_n[j] = n;
            S.job_ntt[j] = ntt;
            S.job_xo[j] = xo;
            S.pa[j] = ca;
            S.pb[j] = cb;
            xo += ntt;
            ca += kUAM * S.cost[0][ntt];
            cb += kUBM * S.cost[1][ntt];
        }
        if (kLight && lane == 0) {
            S.dg[17][4] = gtimer();
        }
        if constexpr (lean_stage(MODE)) {
             
             
             
             
             
            const int tl = nj > 0 ? (min(64, tot - 64 * (nj - 1)) + 7) >> 3 : 0;
            const int n8 = tl == 8 ? nj : nj - (nj > 0 ? 1 : 0);
            const int n8_incl = warp_scan(n8, lane);
            const int all8 = __shfl_sync(0xffffffffu, n8_incl, 31);
            for (int k = 0; k < n8; ++k) {
                S.ord[n8_incl - n8 + k] = j0 + k;
            }
            int above = 0;
            unsigned same = 0;
#pragma unroll
            for (int t = 7; t >= 1; --t) {
                const unsigned b = __ballot_sync(0xffffffffu, tl == t);
                above += t > tl ? __popc(b) : 0;
                same = t == tl ? b : same;
            }
            if (tl > 0 && tl < 8) {
                S.ord[all8 + above + __popc(same & ((1u << lane) - 1u))] = j0 + nj - 1;
            }
        } else if constexpr (own_a(MODE)) {
             
             
            int base = 0;
            for (int t = 8; t >= 1; --t) {
                int c = 0;
                for (int k = 0; k < nj; ++k) {
                    c += ((min(64, tot - 64 * k) + 7) >> 3) == t ? 1 : 0;
                }
                const int incl = warp_scan(c, lane);
                int r = base + incl - c;
                for (int k = 0; k < nj; ++k) {
                    if (((min(64, tot - 64 * k) + 7) >> 3) == t) {
                        S.ord[r++] = j0 + k;
                    }
                }
                base += __shfl_sync(0xffffffffu, incl, 31);
            }
        }
        if (kLight && lane == 0) {
            S.dg[17][5] = gtimer();
        }
        int mx = tot > 0 ? (min(64, tot) + 7) >> 3 : 0;
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, d));
        }
        if (lane == 31) {
            S.njobs = jincl;
            S.pa[jincl] = ca_incl;
            S.pb[jincl] = cb_incl;
            const int xsz = mx <= 2 ? 8192 : mx <= 4 ? 16384 : 32768;
            S.xsz = xsz;
            S.nsw = (kRingBytes - kNSX * xsz) / kUnitBytes;
            S.xoff = S.nsw * kUnitBytes;
        }
    } else if (warp == kNWarpsM - 2) {
        if (lane == 0 && !p.pdl) {
            constexpr unsigned kRel = lean_stage(MODE) ? kNC * 32 : kNC;
#pragma unroll
            for (int s = 0; s < kNSWMax; ++s) {
                mbar_init(&S.wfull[s], 1);
                mbar_init(&S.wempty[s], kRel);
            }
#pragma unroll
            for (int s = 0; s < kNSX; ++s) {
                mbar_init(&S.xfull[s], 1);
                mbar_init(&S.xempty[s], kRel);
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    } else {
        const int ctid = cta * (kNC * 32) + tid;
        const int cstride = G * (kNC * 32);
        float* ss_next = p.ss + ((gen + 1) & 1) * kEmax;
        for (int i = ctid; i < kEmax; i += cstride) {
            ss_next[i] = 0.f;
        }
        if (ctid == 0) {
            p.ctr[4 + ((gen + 1) & 1)] = 0;
        }
        if (dyn_b(MODE) && tid < kBqMax) {
            S.bq[tid] = kBqEmpty;
        }
    }
    __syncthreads();
    if (kLight && tid == 0) {
        S.dg[17][1] = gtimer();
    }
    if constexpr (fast_setup(MODE)) {
        position_fill<kNWarpsM>(S.eid, S.pos, S.wc, E);
    } else {
        position_pass<kNWarpsM>(S.eid, S.pos, nullptr, S.wc, S.off, E);
    }
    if (kLight && tid == 0) {
        S.dg[17][2] = gtimer();
    }
    if (tid == 0) {
         
         
         
#pragma unroll
        for (int ph = 0; ph < 2; ++ph) {
            S.gsb[ph] = fast_setup(MODE) ? 0 : stage_at(S, ph, cta, G);
            S.gse[ph] = fast_setup(MODE) ? 0 : stage_at(S, ph, cta + 1, G);
        }
        if (a_only(MODE)) {
            S.gsb[1] = 0;
            S.gse[1] = 0;
        }
        if (b_only(MODE)) {
            S.gsb[0] = 0;
            S.gse[0] = 0;
        }
        if (own_a(MODE)) {
            const int units = S.njobs * (kUAM / 16);
            int n = 0;
            while (own_unit(n, cta, G) < units) {
                ++n;
            }
            S.own_n = n;
        }
    }
     
     
    __syncthreads();
    if (kLight && tid == 0) {
        S.dg[16][1] = gtimer();
    }
    if (warp == kNC) {
        if (lane == 0) {
            wproducer<MODE>(p, S, cta, G, pol);
        }
    } else {
    const int gt = warp < kNC ? tid : tid - 32;
    constexpr int kNG = kNTM - 32;
    if constexpr (!b_only(MODE)) {
         
         
        if constexpr (fast_setup(MODE)) {
             
             
             
            constexpr int kGB = 8;
            for (int ib = cta; ib < E; ib += kGB * G) {
                uint4 v[kGB];
                unsigned char* dst[kGB];
#pragma unroll
                for (int k = 0; k < kGB; ++k) {
                    const int i = ib + k * G;
                    dst[k] = nullptr;
                    const int q = i < E ? S.pos[i] : -1;
                    if (q >= 0 && gt < 512) {
                        const int e = S.eid[i];
                        const int r = q - S.off[e];
                        const int jj = S.job0[e] + (r >> 6);
                        const int n = r & 63;
                        dst[k] = p.xa + static_cast<size_t>(S.job_xo[jj]) * 65536
                                 + ((4 * gt) >> 7) * S.job_ntt[jj] * 4096 + (n >> 3) * 4096
                                 + (n & 7) * 16;
                        v[k] = __ldg(reinterpret_cast<const uint4*>(
                                         p.x + static_cast<size_t>(i >> 3) * kH) + gt);
                    }
                }
#pragma unroll
                for (int k = 0; k < kGB; ++k) {
                    if (dst[k] != nullptr) {
                        const unsigned w4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
                        for (int h = 0; h < 4; ++h) {
                            const int kk = ((4 * gt + h) & 127) * 2;
                            const int step = 4 * (kk >> 6) + ((kk >> 2) & 3);
                            const int chunk8 = 2 * step + ((kk & 3) >> 1);
                            const int elem = 2 * ((kk >> 4) & 3);
                            *reinterpret_cast<unsigned*>(dst[k] + chunk8 * 128 + elem * 2) = w4[h];
                        }
                    }
                }
            }
        }
        for (int i = cta; i < (fast_setup(MODE) ? 0 : E); i += G) {
            const int q = S.pos[i];
            if (q < 0) {
                continue;
            }
            const int e = S.eid[i];
            const int r = q - S.off[e];
            const int jj = S.job0[e] + (r >> 6);
            const int n = r & 63;
            const int ntt = S.job_ntt[jj];
            const unsigned* src =
                reinterpret_cast<const unsigned*>(p.x + static_cast<size_t>(i >> 3) * kH);
            unsigned char* dst = p.xa + static_cast<size_t>(S.job_xo[jj]) * 65536 + (n >> 3) * 4096
                                 + (n & 7) * 16;
             
             
            for (int q = gt; q < 2048; q += kNG) {
                const int kk = (q & 127) * 2;
                const int step = 4 * (kk >> 6) + ((kk >> 2) & 3);
                const int chunk8 = 2 * step + ((kk & 3) >> 1);
                const int elem = 2 * ((kk >> 4) & 3);
                *reinterpret_cast<unsigned*>(dst + (q >> 7) * ntt * 4096 + chunk8 * 128
                                             + elem * 2) = __ldg(src + q);
            }
        }
        fence_proxy_async_global();
        __threadfence();
        named_sync(kBarGather, kNG);
         
         
         
        for (int i = cta + gt * G; i < E; i += kNG * G) {
            if (S.pos[i] >= 0) {
                const int e = S.eid[i];
                red_release_add(p.xready + S.job0[e] + ((S.pos[i] - S.off[e]) >> 6), 1);
            }
        }
    }

    if constexpr (kDiag) {
        dt[1] = gtimer();
    }
    if (kLight && tid == 0) {
        S.dg[16][2] = gtimer();
    }
    if (skew_st(MODE) && tid == 0) {
        long long* h = sk_cta(p);
        h[0] = S.nsw;
        h[1] = clock64();
        h[2] = gtimer();
    }
    if (warp < kNC) {
        consumers<MODE>(p, S, ss, cta, G);
    } else if (lane == 0) {
        xproducer<MODE>(p, S, cta, G);
    }
    }
    __syncthreads();
    if constexpr (kDiag) {
        dt[2] = gtimer();
    }
    if (kLight && tid == 0) {
        S.dg[16][4] = gtimer();
    }

    if (tid == 0) {
        int* bc = p.ctr + 4 + (gen & 1);
        __threadfence();
        const int old = atomicAdd(bc, 1);
        if (old == G - 1) {
            atomicAdd(p.ctr + 3, 1);
        }
        if (ld_acquire(bc) < G) {
            const long long t0 = clock64();
            while (ld_acquire(bc) < G) {
                __nanosleep(64);
                if (spin_failed(t0, p.ctr, 4)) {
                    break;
                }
            }
        }
    }
    __syncthreads();
    if constexpr (kDiag) {
        dt[3] = gtimer();
    }
    if (kLight && tid == 0) {
        S.dg[16][5] = gtimer();
    }
    for (int i = cta * kNTM + tid; i < kJmaxM * 16; i += G * kNTM) {
        p.done_as[i] = 0;
    }
    for (int i = cta * kNTM + tid; i < kJmaxM; i += G * kNTM) {
        p.xready[i] = 0;
    }
    for (int i = cta * kNTM + tid; i < 2 * G * 4; i += G * kNTM) {
        p.fixcnt[i] = 0;
    }
    if (dyn_b(MODE) && cta == 0 && tid == 0) {
        p.ctr[10] = 0;
    }
    if (warp < kNC && !a_only(MODE)) {
        if constexpr (final_ss(MODE)) {
            if (finss_on<MODE>(p.M)) {
                 
                 
                final_pass_ss<kNC * 32>(p, S.pos, reinterpret_cast<float*>(S.rings), kBarFin,
                                        cta, G);
            } else {
                final_pass<kNC * 32>(p, S.pos, ss, cta, G);
            }
        } else {
            final_pass<kNC * 32>(p, S.pos, ss, cta, G);
        }
    }
    if constexpr (kDiag) {
        __syncthreads();
        if (tid == 0) {
            dt[4] = gtimer();
            for (int i = 0; i < 5; ++i) {
                S.dg[16][3 + i] = dt[i];
            }
            S.dg[17][3] = dc0;
            S.dg[17][4] = clock64();
        }
        __syncthreads();
        long long* out = reinterpret_cast<long long*>(p.inter) + cta * 18 * 8;
        if (tid < 18 * 8) {
            out[tid] = (&S.dg[0][0])[tid];
        }
    }
    if constexpr (kLight) {
        __syncthreads();
        if (tid == 0) {
            S.dg[16][6] = gtimer();
            if (skew_st(MODE)) {
                long long* h = sk_cta(p);
                h[3] = clock64();
                h[4] = gtimer();
            }
        }
        __syncthreads();
        if (pa_diag(MODE)) {
            if (tid < 18 * 8) {
                reinterpret_cast<long long*>(p.inter)[cta * 18 * 8 + tid] = (&S.dg[0][0])[tid];
            }
        } else if (tid < 16) {
            reinterpret_cast<long long*>(p.inter)[cta * 16 + tid] = S.dg[16 + tid / 8][tid % 8];
        }
    }
}

template <int MODE>
void launch_main(const Params& p, int sms, cudaStream_t stream) {
    static bool ready = false;
    if (!ready) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(moe_main_kernel<MODE>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(sizeof(SmemMain))));
        ready = true;
    }
    moe1_launch(moe_main_kernel<MODE>, sms, kNTM, static_cast<int>(sizeof(SmemMain)), stream,
                p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}   
}   
'''

_MOE1_HOST_SRC = r'''
namespace {
namespace moe1 {

void moe1_dispatch(int variant, const Params& p, int sms, cudaStream_t stream) {
    switch (variant) {
//@MOE1_DISPATCH@
        default: TORCH_CHECK(false, "moe_fused: no variant ", variant);
    }
}

}   
}   

 
 
 
 
 
 
 
 
torch::Tensor moe_fused_run(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w,
                            torch::Tensor local_start, torch::Tensor w13, torch::Tensor w2,
                            torch::Tensor gs13, torch::Tensor gs2, torch::Tensor nw,
                            torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss,
                            torch::Tensor ctr, torch::Tensor done_a, torch::Tensor w13f,
                            torch::Tensor w2f, torch::Tensor xa, torch::Tensor xb,
                            torch::Tensor fix, torch::Tensor fixcnt, torch::Tensor xready,
                            torch::Tensor done_as, int64_t nloc, double eps, int64_t variant,
                            bool pdl) {
    using namespace moe1;
    static int sms = 0;
    if (sms == 0) {
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                              x.get_device()));
    }
    const int64_t m = x.size(0);
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2
                && x.size(1) == kH, "moe_fused: x must be a contiguous BF16 [M, 4096]");
    TORCH_CHECK(topk_ids.is_contiguous() && topk_ids.scalar_type() == torch::kInt32
                && topk_ids.dim() == 2 && topk_ids.size(0) == m && topk_ids.size(1) == kTopK
                && reinterpret_cast<uintptr_t>(topk_ids.data_ptr()) % 16 == 0,
                "moe_fused: topk_ids must be a 16-byte aligned contiguous int32 [M, 8]");
    TORCH_CHECK(topk_w.is_contiguous() && topk_w.scalar_type() == torch::kFloat
                && topk_w.sizes() == topk_ids.sizes(),
                "moe_fused: topk_weights must be a contiguous fp32 [M, 8]");
    TORCH_CHECK(local_start.scalar_type() == torch::kInt32 && local_start.numel() == 1,
                "moe_fused: local_start must be one int32 on the device");
    TORCH_CHECK(nloc >= 1 && nloc <= kLocalMax && w13.size(0) >= nloc && w2.size(0) >= nloc
                && w13f.size(0) >= nloc && w2f.size(0) >= nloc && gs13.numel() >= nloc
                && gs2.numel() >= nloc, "moe_fused: weights cover fewer experts than nloc");
    TORCH_CHECK(nw.scalar_type() == torch::kFloat && nw.numel() == kH
                && gs13.scalar_type() == torch::kFloat && gs2.scalar_type() == torch::kFloat,
                "moe_fused: scales and the norm weight must be fp32");
    const int64_t xbytes = static_cast<int64_t>(kEmax / 8 + kJmaxM) * 65536;
    TORCH_CHECK(inter.numel() >= static_cast<int64_t>(kEmax) * kI
                && ybuf.numel() >= static_cast<int64_t>(kEmax) * kH && ss.numel() >= 2 * kEmax
                && ctr.numel() >= 64 && done_a.numel() >= kJmax && xready.numel() >= kJmaxM
                && done_as.numel() >= kJmaxM * 16 && fixcnt.numel() >= 2 * sms * 4
                && fix.numel() >= static_cast<int64_t>(2 * sms * 4) * 128 * 32
                && xa.numel() >= xbytes && xb.numel() >= xbytes, "moe_fused: scratch too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({m, kH}, x.options());
    Params p;
    p.w13 = static_cast<const unsigned char*>(w13.data_ptr());
    p.w2 = static_cast<const unsigned char*>(w2.data_ptr());
    p.gs13 = gs13.data_ptr<float>();
    p.gs2 = gs2.data_ptr<float>();
    p.nw = nw.data_ptr<float>();
    p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>());
    p.ybuf = ybuf.data_ptr<float>();
    p.ss = ss.data_ptr<float>();
    p.ctr = ctr.data_ptr<int>();
    p.done_a = done_a.data_ptr<int>();
    p.w13f = static_cast<const unsigned char*>(w13f.data_ptr());
    p.w2f = static_cast<const unsigned char*>(w2f.data_ptr());
    p.xa = static_cast<unsigned char*>(xa.data_ptr());
    p.xb = static_cast<unsigned char*>(xb.data_ptr());
    p.fix = fix.data_ptr<float>();
    p.fixcnt = fixcnt.data_ptr<int>();
    p.xready = xready.data_ptr<int>();
    p.done_as = done_as.data_ptr<int>();
    p.lo = 0;
    p.lo_p = local_start.data_ptr<int>();
    p.pdl = 0;
    p.nloc = static_cast<int>(nloc);
    p.eps = static_cast<float>(eps);
     
     
    for (int i = 0; i < 18; ++i) {
        p.cost[i / 9][i % 9] = 0;
    }
    for (int64_t m0 = 0; m0 < m; m0 += kEmax / kTopK) {
        p.M = static_cast<int>(std::min<int64_t>(kEmax / kTopK, m - m0));
        p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>())
              + static_cast<size_t>(m0) * kH;
        p.topk_ids = topk_ids.data_ptr<int>() + static_cast<size_t>(m0) * kTopK;
        p.topk_w = topk_w.data_ptr<float>() + static_cast<size_t>(m0) * kTopK;
        p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>())
                + static_cast<size_t>(m0) * kH;
        p.pdl = pdl && m0 == 0 ? 1 : 0;
        moe1_dispatch(static_cast<int>(variant), p, sms, stream);
    }
    return out;
}

std::vector<int64_t> qual_moe_resources() {
    auto kernel=moe1::moe_main_kernel<51>;
    const int bytes=sizeof(moe1::SmemMain),threads=moe1::kNTM;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    int blocks=0,device=0,sms=0,coop=0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,bytes));
    return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,threads,
            a.maxThreadsPerBlock,blocks,sms,coop,blocks*sms};
}
'''

_MOE1_CPP_SRC = r'''
#include <torch/extension.h>
torch::Tensor moe_fused_run(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w,
                            torch::Tensor local_start, torch::Tensor w13, torch::Tensor w2,
                            torch::Tensor gs13, torch::Tensor gs2, torch::Tensor nw,
                            torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss,
                            torch::Tensor ctr, torch::Tensor done_a, torch::Tensor w13f,
                            torch::Tensor w2f, torch::Tensor xa, torch::Tensor xb,
                            torch::Tensor fix, torch::Tensor fixcnt, torch::Tensor xready,
                            torch::Tensor done_as, int64_t nloc, double eps, int64_t variant,
                            bool pdl);

std::vector<int64_t> qual_moe_resources();
'''

_SKP_SRC = r'''

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef SK_DEVICE_ONLY
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#endif

 
 
 
#ifndef SK_PDL
#define SK_PDL 0
#endif

namespace {

constexpr uint32_t SUSPEND_HINT = 0x989680u;
 
 
constexpr long long WAIT_LIMIT = 1ll << 28;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}

__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\\n.reg .pred p;\\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\nselp.u32 %0, 1, 0, p;\\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

 
 
#ifndef OKC_WAIT_SLEEP
#define OKC_WAIT_SLEEP 1
#endif

__device__ __forceinline__ bool mbar_test_sleep(uint32_t bar, uint32_t parity) {
#if OKC_WAIT_SLEEP == 0
    return mbar_test(bar, parity);
#endif
    uint32_t ok;
    asm volatile("{\\n.reg .pred p;\\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\\nselp.u32 %0, 1, 0, p;\\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(SUSPEND_HINT) : "memory");
    return ok != 0;
}

__device__ __forceinline__ bool mbar_test_cluster(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\\n.reg .pred p;\\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\\n"
                 "selp.u32 %0, 1, 0, p;\\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

 
__device__ __forceinline__ bool mbar_wait_cluster(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead) {
        return false;
    }
    const long long t0 = clock64();
    while (!mbar_test_cluster(bar, parity)) {
        if (clock64() - t0 > WAIT_LIMIT) {
            dead = true;
            if (diag != nullptr) {
                atomicOr(diag, code);
            }
            return false;
        }
    }
    return true;
}

template <bool SLEEP>
__device__ __forceinline__ bool mbar_wait(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead) {
        return false;
    }
    if (SLEEP ? mbar_test_sleep(bar, parity) : mbar_test(bar, parity)) {
        return true;
    }
    const long long t0 = clock64();
    while (!(SLEEP ? mbar_test_sleep(bar, parity) : mbar_test(bar, parity))) {
        if (clock64() - t0 > WAIT_LIMIT) {
            dead = true;
            if (diag != nullptr) {
                atomicOr(diag, code);
            }
            return false;
        }
    }
    return true;
}

 
 
__device__ __forceinline__ bool mbar_wait_lite(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead) {
        return false;
    }
    if (mbar_test(bar, parity)) {
        return true;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return false;
            }
        }
    }
    return true;
}

 
 
 
 
__device__ __forceinline__ void mbar_wait_novote(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead || mbar_test(bar, parity)) {
        return;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return;
            }
        }
    }
}

 
 
__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void named_bar(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5}], [%2];"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ uint4 lds128(uint32_t a) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)
                 : "memory");
    return v;
}

__device__ __forceinline__ uint2 lds64(uint32_t a) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory");
    return v;
}

__device__ __forceinline__ float lds_bf16(uint32_t a) {
    uint16_t v;
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory");
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ __forceinline__ void ldsm_x4(uint32_t a, uint32_t* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}

 
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}

 
 
 
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t codes, uint32_t base4, uint32_t* r) {
    const uint32_t ha = ((codes & 0x07070707u) + base4) | ((codes << 4) & 0x80808080u);
    const uint32_t hb = (((codes >> 4) & 0x07070707u) + base4) | (codes & 0x80808080u);
    r[0] = prmt(lw0, ha, 0x5140);
    r[1] = prmt(lw0, ha, 0x7362);
    r[2] = prmt(lw1, hb, 0x5140);
    r[3] = prmt(lw1, hb, 0x7362);
}


#ifndef SK_DEVICE_ONLY
PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
    if (fn == nullptr) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 12050
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
        TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
        fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
    }
    return fn;
}
#endif
}  
'''

_SKP_WD_SRC = r'''

namespace {

template <int BN>
__device__ __forceinline__ void wgmma_ss(float* d, uint64_t a, uint64_t b);
template <> __device__ __forceinline__ void wgmma_ss<128>(float* d, uint64_t a, uint64_t b) {
    asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %66, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "%64, %65, p, 1, 1, 0, 0;\\n}\\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "l"(a), "l"(b), "r"(1) : "memory");
}


 
 
 
 
__device__ __forceinline__ uint64_t wg_desc(uint32_t addr) {
    return (uint64_t)((addr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32)
           | ((uint64_t)1 << 62);
}

__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}

 
 
template <int R>
__device__ __forceinline__ void wg_fence_acc(float* d) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
        asm volatile("" : "+f"(d[i]) :: "memory");
    }
}

__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1,
                                           uint64_t pol) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%3, %4}], [%2], %5;"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}

 
 
 
__device__ __forceinline__ uint64_t l2_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

__device__ __forceinline__ uint64_t l2_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

__device__ __forceinline__ uint32_t wd_cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ uint32_t wd_mapa(uint32_t addr, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

__device__ __forceinline__ float4 wd_ld_cluster(uint32_t addr) {
    float4 v;
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ float4 wd_ld_local(uint32_t addr) {
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(addr) : "memory");
    return v;
}

 
__device__ __forceinline__ void wd_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

 
 
template <bool RELAXED>
__device__ __forceinline__ void tail_arrive() {
    if constexpr (RELAXED) {
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    } else {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    }
}

__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1,
                                          uint16_t mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%3, %4}], [%2], %5;"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory");
}

 
 
 
 
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_addr) {
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(cluster_addr) : "memory");
}

 
 
 
 
 
 
 
 
 
 
constexpr int WV_NOPROD = 1;      
constexpr int WV_PACK = 2;        
constexpr int WV_PROMO64 = 4;     
constexpr int WV_PUSH = 8;        
constexpr int WV_SPIN = 16;       
constexpr int WV_EARLY = 32;      
constexpr int WV_DUP = 64;        
constexpr int WV_TAILSYNC = 128;  
constexpr int WV_NOSLEEP = 256;   
 
 
 
constexpr int WV_NOEMPTY = 512;
 
 
 
constexpr int WV_XA = 1024;
 
 
 
constexpr int WV_TAILSPLIT = 2048;
constexpr int WV_TAILRLX = 4096;
 
 
 
 
constexpr int WV_PGROUP = 8192;
constexpr int WV_ACQCL = 16384;
constexpr int WV_SCALAR = 32768;
// The K split without a cluster: CTAs of one output tile add their FP32 partials into a
// zero-kept global accumulator with red.add, and the last of them (a per-tile counter) copies
// the tile to out and zeroes it again. No co-residency is needed, so the split can exceed the
// 8-CTA portable cluster and the grid can exceed one wave. Mechanism of the rk router in
// alygg77's kernels/muse_bf16_proj.py, re-implemented here with a last-arriver join instead of
// its grid-wide barrier; ordering as CUDA's threadFenceReduction sample (docs/proj-port-guide.md,
// "Global split-K join").
constexpr int WV_GRED = 65536;
 
 
 
constexpr int WV_GZERO = 131072;
constexpr int GRED_TOK = 256;
constexpr int GRED_N = 256;
constexpr int GRED_TILES = 64;
__device__ __align__(16) float wd_gred_acc[GRED_TOK * GRED_N];
__device__ unsigned int wd_gred_cnt[GRED_TILES];

__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ void mbar_spin_cluster(uint32_t bar, uint32_t parity) {
    asm volatile("{\\n.reg .pred p;\\nwd_spin:\\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\\n"
                 "@!p bra wd_spin;\\n}" :: "r"(bar), "r"(parity) : "memory");
}

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, int VAR = 0>
struct WdCfg {
    static constexpr bool NOPROD = (VAR & WV_NOPROD) != 0;
    static constexpr bool PACK = (VAR & WV_PACK) != 0;
    static constexpr bool PUSH = (VAR & WV_PUSH) != 0;
    static constexpr int NCT = WG * 128;
     
     
    static constexpr int TPR = BN / CLK;
    static constexpr int JPR = TPR / 8;
    static constexpr bool XA = (VAR & WV_XA) != 0;
     
     
    static constexpr int XUNITS = NCT * 8 / CLK;
    static constexpr int RECV = PUSH ? (XA ? CLK * XUNITS * 16 : CLK * JPR * NCT * 16) : 0;
    static constexpr int NACC = XA ? 64 : BN / 2;
    static constexpr int CL = CLK * CLR;
    static constexpr int BM = 64 * WG;
    static constexpr int ATILE = 64 * 128;
    static constexpr int NA = HL ? 2 : 1;
    static constexpr int ABY = WG * NA * ATILE;
    static constexpr int BBY = BN * 128;
    static constexpr int XS = BN / CLR;
    static constexpr int SB = ABY + BBY;
    static constexpr int RING = NST * SB;
     
     
    static constexpr int RS = BM + 4;
    static constexpr int PBY = BN * RS * 4;
    static constexpr bool PGROUP = (VAR & WV_PGROUP) != 0;
    static constexpr bool GRED = (VAR & WV_GRED) != 0;
     
    static constexpr int LCL = GRED ? 1 : CLK * CLR;
    static constexpr int THREADS = NCT + (NOPROD ? 0 : (PGROUP ? 128 : 32));
    static constexpr int SMEM = 1024 + RING + 16 * NST + (PUSH ? 16 + RECV : 0);
    static_assert(!PUSH || (CLR == 1 && CLK > 1 && TPR % 8 == 0 && HL), "wd: push needs CLR 1, 8-token owners, f32 out");
    static_assert(!PACK || HL, "wd: packed weights are the router's hi/lo pairs");
    static_assert(!XA || (PACK && PUSH && WG == 1 && BN == 64 && (TPR == 8 || TPR % 16 == 0)), "wd: bad XA config");
    static_assert(!PGROUP || (!NOPROD && PUSH), "wd: the producer group is built for the push router only");
    static_assert((VAR & WV_SCALAR) == 0 || XA, "wd: SCALAR changes only the XA epilogue");
    static_assert(PBY <= RING, "wd: partial tile does not fit in the ring");
    static_assert(BN % 8 == 0 && BN <= 256, "wd: bad BN");
     
     
    static_assert(BN % CLR == 0 && XS % 8 == 0, "wd: bad x slice");
    static_assert(LCL <= 16, "wd: cluster too large");
    static_assert(!GRED || (HL && CLR == 1 && !PUSH && !NOPROD), "wd: GRED is built for the f32 router only");
    static_assert(SMEM <= 227 * 1024, "wd: shared memory over budget");
};

__device__ __forceinline__ void wd_st_async(uint32_t cluster_addr, float4 v, uint32_t cluster_bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                 :: "r"(cluster_addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(cluster_bar) : "memory");
}

__device__ __forceinline__ void wd_st_local(uint32_t addr, float4 v) {
    asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL, bool POL, int VAR = 0>
__global__ void __launch_bounds__((WdCfg<WG, BN, CLK, CLR, HL, NST, VAR>::THREADS), 1)
wd_kernel(const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap, OutT* __restrict__ out,
          int M, int N, int K, int* diag) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST, VAR>;
    constexpr int CL = C::CL;
    constexpr bool NOPROD = C::NOPROD, PACK = C::PACK, PUSH = C::PUSH, XA = C::XA;
    constexpr int NACC = C::NACC;
    constexpr bool SPIN = (VAR & WV_SPIN) != 0, EARLY = (VAR & WV_EARLY) != 0;
    constexpr bool TSPLIT = (VAR & WV_TAILSPLIT) != 0;
    constexpr bool ACQCL = (VAR & WV_ACQCL) != 0, SCALAR = (VAR & WV_SCALAR) != 0;
    static_assert(!PUSH || sizeof(OutT) == 4, "wd: push writes f32");
    static_assert(!TSPLIT || (PUSH && (VAR & WV_TAILSYNC) == 0), "wd: split exit barrier replaces TAILSYNC");
    static_assert((VAR & WV_TAILRLX) == 0 || TSPLIT, "wd: TAILRLX needs TAILSPLIT");
    extern __shared__ __align__(1024) uint8_t wd_smem[];
    const uint32_t raw = smem_u32(wd_smem);
    const uint32_t sbase = (raw + 1023u) & ~1023u;
    float* part = reinterpret_cast<float*>(wd_smem + (sbase - raw));
    const uint32_t full0 = sbase + C::RING;
    const uint32_t empty0 = full0 + 8 * NST;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rank = C::GRED ? (int)(blockIdx.x % CL) : (CL > 1 ? (int)wd_cluster_rank() : 0);
    const int rk = rank % CLK;
    const int rr = rank / CLK;
    const int rowbase = ((int)(blockIdx.x / CL) * CLR + rr) * C::BM;
    const int tok0 = (int)blockIdx.y * BN;
     
    const int kt0 = rk * (K / 64) / CLK;
    const int ktl = (rk + 1) * (K / 64) / CLK - kt0;
    bool dead = false;
    const uint32_t rbar = empty0 + 8 * NST;
    const uint32_t recv = rbar + 16;
     
    const int pt = NOPROD ? 0 : C::NCT;
    const int npro = ktl < NST ? ktl : NST;

    auto issue = [&](int i, uint64_t wpol, uint64_t xpol) {
        uint16_t mask = 0;
#pragma unroll
        for (int r = 0; r < CLR; ++r) {
            mask |= (uint16_t)(1u << (rk + CLK * r));
        }
        const int s = i % NST;
        const uint32_t fb = full0 + 8 * s;
        mbar_expect_tx(fb, C::SB);
        const uint32_t st = sbase + s * C::SB;
        const int kc = (kt0 + i) * 64;
#pragma unroll
        for (int g = 0; g < WG; ++g) {
            if constexpr (PACK) {
                 
                tma_2d(st + g * C::NA * C::ATILE, &wmap, fb, 0, ((rowbase / 64 + g) * (K / 64) + kt0 + i) * 128);
            } else {
                if constexpr (POL) {
                    tma_2d_pol(st + g * C::NA * C::ATILE, &wmap, fb, kc, rowbase + 64 * g, wpol);
                } else {
                    tma_2d(st + g * C::NA * C::ATILE, &wmap, fb, kc, rowbase + 64 * g);
                }
                if constexpr (HL) {
                    if constexpr (POL) {
                        tma_2d_pol(st + (g * C::NA + 1) * C::ATILE, &wmap, fb, kc, N + rowbase + 64 * g, wpol);
                    } else {
                        tma_2d(st + (g * C::NA + 1) * C::ATILE, &wmap, fb, kc, N + rowbase + 64 * g);
                    }
                }
            }
        }
        if constexpr (CLR > 1) {
            tma_2d_mc(st + C::ABY + rr * C::XS * 128, &xmap, fb, kc, tok0 + rr * C::XS, mask);
        } else if constexpr (POL) {
            tma_2d_pol(st + C::ABY, &xmap, fb, kc, tok0, xpol);
        } else {
            tma_2d(st + C::ABY, &xmap, fb, kc, tok0);
        }
    };

    if (tid == pt) {
        for (int s = 0; s < NST; ++s) {
            mbar_init(full0 + 8 * s, 1);
            mbar_init(empty0 + 8 * s, 4 * WG * CLR);
        }
        if constexpr (PUSH) {
            mbar_init(rbar, 1);
        }
        fence_mbar_init();
        if constexpr (PUSH) {
             
             
             
            mbar_expect_tx(rbar, XA ? (CLK - 1) * C::XUNITS * 16 : (CLK - 1) * C::JPR * C::NCT * 16);
        }
         
         
        if constexpr (EARLY && CLR == 1) {
            for (int i = 0; i < npro; ++i) {
                issue(i, POL ? l2_evict_first() : 0, POL ? l2_evict_last() : 0);
            }
        }
    }
     
     
    if constexpr (CLR > 1) {
        wd_cluster_sync();
    } else {
        __syncthreads();
    }
    if constexpr (PUSH) {
         
         
        cluster_arrive();
    }
    if constexpr (NOPROD && !(EARLY && CLR == 1)) {
        if (tid == pt) {
            for (int i = 0; i < npro; ++i) {
                issue(i, POL ? l2_evict_first() : 0, POL ? l2_evict_last() : 0);
            }
        }
    }

    float acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
        acc[j] = 0.f;
    }
    if (!NOPROD && (C::PGROUP ? warp >= 4 * WG : warp == 4 * WG)) {
        if ((!C::PGROUP || warp == 4 * WG) && lane == 0) {
            const uint64_t wpol = POL ? l2_evict_first() : 0;
            const uint64_t xpol = POL ? l2_evict_last() : 0;
            for (int i = (EARLY && CLR == 1) ? npro : 0; i < ktl; ++i) {
                if (i >= NST) {
                    const uint32_t eb = empty0 + 8 * (i % NST);
                    const uint32_t ep = ((i / NST) - 1) & 1;
                    if constexpr (ACQCL && SPIN) {
                        mbar_spin_cluster(eb, ep);
                    } else if constexpr (ACQCL) {
                        mbar_wait_cluster(eb, ep, dead, diag, 1);
                    } else {
                        mbar_wait<(VAR & WV_NOSLEEP) == 0>(eb, ep, dead, diag, 1);
                    }
                }
                issue(i, wpol, xpol);
            }
        }
        __syncwarp();
    } else {
        const int g = warp >> 2;
        for (int i = 0; i < ktl; ++i) {
            const int s = i % NST;
            if constexpr (ACQCL && SPIN) {
                mbar_spin_cluster(full0 + 8 * s, (i / NST) & 1);
            } else if constexpr (ACQCL) {
                mbar_wait_cluster(full0 + 8 * s, (i / NST) & 1, dead, diag, 2);
            } else if constexpr (SPIN) {
                while (!mbar_test(full0 + 8 * s, (i / NST) & 1)) {
                }
            } else {
                mbar_wait<false>(full0 + 8 * s, (i / NST) & 1, dead, diag, 2);
            }
            const uint32_t a0 = sbase + s * C::SB + g * C::NA * C::ATILE;
            const uint32_t b0 = sbase + s * C::SB + C::ABY;
            wg_fence_acc<NACC>(acc);
            wg_fence();
#pragma unroll
            for (int kk = 0; kk < ((ABL & 1) ? 0 : 4); ++kk) {
                if constexpr (XA) {
                    wgmma_ss<128>(acc, wg_desc(b0 + 32 * kk), wg_desc(a0 + 32 * kk));
                } else {
                    wgmma_ss<BN>(acc, wg_desc(a0 + 32 * kk), wg_desc(b0 + 32 * kk));
                    if constexpr (HL) {
                        wgmma_ss<BN>(acc, wg_desc(a0 + C::ATILE + 32 * kk), wg_desc(b0 + 32 * kk));
                    }
                }
            }
            wg_commit();
            wg_fence_acc<NACC>(acc);
             
             
            wg_wait<1>();
            if (i > 0 && lane == 0) {
                const uint32_t eb = empty0 + 8 * ((i - 1) % NST);
                if constexpr (CLR > 1) {
#pragma unroll
                    for (int r = 0; r < CLR; ++r) {
                        mbar_arrive_remote(wd_mapa(eb, (uint32_t)(rk + CLK * r)));
                    }
                } else {
                    mbar_arrive(eb);
                }
            }
            if constexpr (NOPROD) {
                 
                 
                const int j = i - 1 + NST;
                if (tid == pt && i > 0 && j < ktl) {
                    if constexpr ((VAR & WV_NOEMPTY) == 0) {
                        mbar_wait<false>(empty0 + 8 * (j % NST), ((j / NST) - 1) & 1, dead, diag, 1);
                    }
                    issue(j, POL ? l2_evict_first() : 0, POL ? l2_evict_last() : 0);
                }
            }
        }
        wg_wait<0>();
        wg_fence_acc<NACC>(acc);
        if constexpr (!PUSH) {
             
            named_bar(1, WG * 128);
            const int row = g * 64 + (warp & 3) * 16 + (lane >> 2);
            const int q = lane & 3;
#pragma unroll
            for (int j = 0; j < BN / 8; ++j) {
                const int t = 8 * j + 2 * q;
                part[t * C::RS + row] = acc[4 * j];
                part[(t + 1) * C::RS + row] = acc[4 * j + 1];
                part[t * C::RS + row + 8] = acc[4 * j + 2];
                part[(t + 1) * C::RS + row + 8] = acc[4 * j + 3];
            }
        }
    }

    if constexpr (PUSH) {
         
         
         
         
         
         
         
        cluster_wait();
        if constexpr (XA) {
             
             
             
             
             
             
             
             
             
            constexpr int JW = C::TPR >= 16 ? C::TPR / 16 : 1;
            float s[32];
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                s[e] = acc[e] + acc[32 + e];
            }
            if (tid < C::NCT) {
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    int dst, slot;
                    float4 v;
                    if constexpr (C::TPR >= 16) {
                        dst = warp * 16 / C::TPR;
                        slot = ((warp % JW) * 32 + lane) * 8 + u;
                        v = make_float4(s[4 * u], s[4 * u + 1], s[4 * u + 2], s[4 * u + 3]);
                    } else {
                        const int h = u >> 2;
                        const int j0 = 2 * (u & 3);
                        dst = 2 * warp + h;
                        slot = lane * 4 + (u & 3);
                        v = make_float4(s[4 * j0 + 2 * h], s[4 * j0 + 2 * h + 1], s[4 * j0 + 4 + 2 * h],
                                        s[4 * j0 + 5 + 2 * h]);
                    }
                    const uint32_t a = recv + (uint32_t)(rk * C::XUNITS + slot) * 16u;
                    if (dst == rk) {
                        wd_st_local(a, v);
                    } else {
                        wd_st_async(wd_mapa(a, (uint32_t)dst), v, wd_mapa(rbar, (uint32_t)dst));
                    }
                }
                named_bar(1, C::NCT);
                mbar_wait_cluster(rbar, 0, dead, diag, 4);
                if constexpr (TSPLIT) {
                    __syncwarp();
                    tail_arrive<(VAR & WV_TAILRLX) != 0>();
                }
                for (int it = tid; it < C::XUNITS; it += C::NCT) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                    for (int p = 0; p < CLK; ++p) {
                        const float4 q4 = wd_ld_local(recv + (uint32_t)(p * C::XUNITS + it) * 16u);
                        v.x += q4.x;
                        v.y += q4.y;
                        v.z += q4.z;
                        v.w += q4.w;
                    }
                    int t0, t1, c0, c1;
                    if constexpr (C::TPR >= 16) {
                        const int l = (it >> 3) & 31;
                        const int w = rk * JW + it / 256;
                        t0 = 16 * w + (l >> 2);
                        t1 = t0 + 8;
                        c0 = c1 = 8 * (it & 7) + 2 * (l & 3);
                    } else {
                        const int l = it >> 2;
                        t0 = t1 = 16 * (rk >> 1) + 8 * (rk & 1) + (l >> 2);
                        c0 = 16 * (it & 3) + 2 * (l & 3);
                        c1 = c0 + 8;
                    }
                    if constexpr (SCALAR) {
                        float* o0 = out + (size_t)(tok0 + t0) * N + rowbase + c0;
                        float* o1 = out + (size_t)(tok0 + t1) * N + rowbase + c1;
                        if (tok0 + t0 < M) {
                            o0[0] = v.x;
                            o0[1] = v.y;
                        }
                        if (tok0 + t1 < M) {
                            o1[0] = v.z;
                            o1[1] = v.w;
                        }
                    } else {
                        if (tok0 + t0 < M) {
                            *reinterpret_cast<float2*>(out + (size_t)(tok0 + t0) * N + rowbase + c0) = make_float2(v.x, v.y);
                        }
                        if (tok0 + t1 < M) {
                            *reinterpret_cast<float2*>(out + (size_t)(tok0 + t1) * N + rowbase + c1) = make_float2(v.z, v.w);
                        }
                    }
                }
            }
        } else if (tid < C::NCT) {
#pragma unroll
            for (int j = 0; j < BN / 8; ++j) {
                const int dst = j / C::JPR;
                const uint32_t a = recv + (uint32_t)((rk * C::JPR + j % C::JPR) * C::NCT + tid) * 16u;
                const float4 v = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
                if (dst == rk) {
                    wd_st_local(a, v);
                } else {
                    wd_st_async(wd_mapa(a, (uint32_t)dst), v, wd_mapa(rbar, (uint32_t)dst));
                }
            }
             
            named_bar(1, C::NCT);
            mbar_wait_cluster(rbar, 0, dead, diag, 4);
            if constexpr (TSPLIT) {
                __syncwarp();
                tail_arrive<(VAR & WV_TAILRLX) != 0>();
            }
            for (int it = tid; it < C::JPR * C::NCT; it += C::NCT) {
                const int jl = it / C::NCT;
                const int c = it % C::NCT;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                 
#pragma unroll
                for (int p = 0; p < CLK; ++p) {
                    const float4 u = wd_ld_local(recv + (uint32_t)((p * C::JPR + jl) * C::NCT + c) * 16u);
                    v.x += u.x;
                    v.y += u.y;
                    v.z += u.z;
                    v.w += u.w;
                }
                const int row = rowbase + (c >> 7) * 64 + ((c >> 5) & 3) * 16 + ((c & 31) >> 2);
                const int gt = tok0 + 8 * (rk * C::JPR + jl) + 2 * (c & 3);
                if (gt < M) {
                    out[(size_t)gt * N + row] = v.x;
                    out[(size_t)gt * N + row + 8] = v.z;
                }
                if (gt + 1 < M) {
                    out[(size_t)(gt + 1) * N + row] = v.y;
                    out[(size_t)(gt + 1) * N + row + 8] = v.w;
                }
            }
        }
        if constexpr ((VAR & WV_TAILSYNC) != 0) {
            wd_cluster_sync();
        }
        if constexpr (TSPLIT) {
            if (tid >= C::NCT) {
                tail_arrive<(VAR & WV_TAILRLX) != 0>();
            }
            cluster_wait();
        }
        return;
    }

     
     
    if constexpr (CL > 1 && !C::GRED) {
        wd_cluster_sync();
    } else {
        __syncthreads();
    }

    if constexpr (C::GRED && (VAR & WV_GZERO) != 0) {
        constexpr int C4G = C::BM / 4;
        for (int it = tid; it < ((ABL & 2) ? 0 : BN * C4G); it += C::THREADS) {
            const int t = it / C4G;
            const int c4 = it % C4G;
            if (tok0 + t < M) {
                red_add_v4(reinterpret_cast<float*>(out) + (size_t)(tok0 + t) * N + rowbase + 4 * c4,
                           wd_ld_local(sbase + (uint32_t)(t * C::RS + 4 * c4) * 4u));
            }
        }
        return;
    }
    if constexpr (C::GRED) {
        constexpr int C4G = C::BM / 4;
        __shared__ int gred_last;
        for (int it = tid; it < ((ABL & 2) ? 0 : BN * C4G); it += C::THREADS) {
            const int t = it / C4G;
            const int c4 = it % C4G;
            if (tok0 + t < M) {
                red_add_v4(wd_gred_acc + (size_t)(tok0 + t) * N + rowbase + 4 * c4,
                           wd_ld_local(sbase + (uint32_t)(t * C::RS + 4 * c4) * 4u));
            }
        }
         
         
         
        __threadfence();
        __syncthreads();
        const int tile = (int)blockIdx.y * (N / C::BM) + rowbase / C::BM;
        if (tid == 0) {
            gred_last = atomicInc(&wd_gred_cnt[tile], (unsigned)(CLK - 1)) == (unsigned)(CLK - 1);
        }
        __syncthreads();
        if (gred_last) {
            __threadfence();
            for (int it = tid; it < BN * C4G; it += C::THREADS) {
                const int t = it / C4G;
                const int c4 = it % C4G;
                if (tok0 + t < M) {
                    float4* a = reinterpret_cast<float4*>(wd_gred_acc + (size_t)(tok0 + t) * N + rowbase + 4 * c4);
                    const float4 v = __ldcg(a);
                    __stcg(a, make_float4(0.f, 0.f, 0.f, 0.f));
                    *reinterpret_cast<float4*>(out + (size_t)(tok0 + t) * N + rowbase + 4 * c4) = v;
                }
            }
        }
        return;
    }

    const int tb = rk * BN / CLK;
    constexpr int C4 = C::BM / 4;
    for (int it = tid; it < ((ABL & 2) ? 0 : (((rk + 1) * BN / CLK) - tb) * C4); it += C::THREADS) {
        const int t = tb + it / C4;
        const int c4 = it % C4;
        const int gt = tok0 + t;
        if (gt >= M) {
            continue;
        }
        const uint32_t a = sbase + (uint32_t)(t * C::RS + 4 * c4) * 4u;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int p = 0; p < CLK; ++p) {
            const float4 u = CLK > 1 ? wd_ld_cluster(wd_mapa(a, (uint32_t)(p + CLK * rr))) : wd_ld_local(a);
            v.x += u.x;
            v.y += u.y;
            v.z += u.z;
            v.w += u.w;
        }
        OutT* o = out + (size_t)gt * N + rowbase + 4 * c4;
        if constexpr (sizeof(OutT) == 4) {
            *reinterpret_cast<float4*>(o) = v;
        } else {
            __nv_bfloat162 lo2 = __floats2bfloat162_rn(v.x, v.y);
            __nv_bfloat162 hi2 = __floats2bfloat162_rn(v.z, v.w);
            uint2 pk;
            pk.x = *reinterpret_cast<uint32_t*>(&lo2);
            pk.y = *reinterpret_cast<uint32_t*>(&hi2);
            *reinterpret_cast<uint2*>(o) = pk;
        }
    }

     
    if constexpr (CL > 1) {
        wd_cluster_sync();
    }
}

#ifndef SK_DEVICE_ONLY
 
 
CUtensorMap make_2d_map(const void* p, uint64_t rows, uint64_t k, uint32_t box_rows,
                        CUtensorMapL2promotion promo = CU_TENSOR_MAP_L2_PROMOTION_L2_256B) {
    CUtensorMap map;
    cuuint64_t dims[2] = {k, rows};
    cuuint64_t strides[1] = {k * 2};
    cuuint32_t box[2] = {64, box_rows};
    cuuint32_t estr[2] = {1, 1};
    const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, promo,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return map;
}

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL, bool POL, int VAR = 0>
void launch_wd(const at::Tensor& w, const at::Tensor& x, at::Tensor& out, int N, int K, int* diag) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST, VAR>;
    auto kern = wd_kernel<WG, BN, CLK, CLR, HL, NST, OutT, ABL, POL, VAR>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        if (C::LCL > 8) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        }
        configured = true;
    }
    const int M = (int)x.size(0);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2
                && x.size(1) == K, "wd: x must be a contiguous bf16 [M, K] tensor");
    if constexpr (C::PACK) {
        TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                    && w.size(1) == 64 && w.size(0) == (int64_t)(N / 64) * (K / 64) * 128,
                    "wd: packed w must be a contiguous bf16 [N/64 * K/64 * 128, 64] tensor");
    } else {
        TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                    && w.size(1) == K && w.size(0) >= (HL ? 2 : 1) * N,
                    "wd: w must be a contiguous bf16 [rows, K] tensor");
    }
    TORCH_CHECK(out.is_contiguous() && out.size(0) == M && out.size(1) == N && out.element_size() == sizeof(OutT),
                "wd: bad out");
    TORCH_CHECK(M >= 1 && N % (C::BM * CLR) == 0 && K % 64 == 0 && K / 64 >= CLK, "wd: shape does not tile");
    const CUtensorMapL2promotion promo =
        (VAR & WV_PROMO64) ? CU_TENSOR_MAP_L2_PROMOTION_L2_64B : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    const CUtensorMap wmap = C::PACK ? make_2d_map(w.data_ptr(), (uint64_t)w.size(0), 64, 128, promo)
                                     : make_2d_map(w.data_ptr(), (uint64_t)w.size(0), (uint64_t)K, 64, promo);
    const CUtensorMap xmap = make_2d_map(x.data_ptr(), (uint64_t)M, (uint64_t)K, C::XS, promo);
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    cudaLaunchConfig_t lc = {};
    if constexpr (C::GRED) {
        TORCH_CHECK(M <= GRED_TOK && N <= GRED_N && (N / C::BM) * ((M + BN - 1) / BN) <= GRED_TILES,
                    "wd: GRED workspace too small for this shape");
    }
    lc.gridDim = dim3((unsigned)(N / (C::BM * CLR) * C::CL), (unsigned)((M + BN - 1) / BN));
    lc.blockDim = dim3(C::THREADS);
    lc.dynamicSmemBytes = C::SMEM;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C::LCL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    lc.attrs = attr;
    lc.numAttrs = C::LCL > 1 ? 1 : 0;
    if constexpr ((VAR & WV_GZERO) != 0) {
        static_assert(C::GRED, "wd: GZERO needs GRED");
        C10_CUDA_CHECK(cudaMemsetAsync(op, 0, (size_t)M * N * sizeof(OutT), lc.stream));
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, wmap, xmap, op, M, N, K, diag));
}
#endif

}   

#ifdef SK_DEVICE_ONLY
namespace {
template __global__ void wd_kernel<1, 128, 2, 1, false, 8, __nv_bfloat16, 0, true, 0>(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*, int, int, int, int*);
template __global__ void wd_kernel<1, 128, 3, 1, false, 8, __nv_bfloat16, 0, true, 0>(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*, int, int, int, int*);
}
#else
void wd_fwd(int64_t cfg, at::Tensor w, at::Tensor x, at::Tensor out, int64_t n, int64_t k, at::Tensor diag) {
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
    switch (cfg) {
    case 0: launch_wd<1, 128, 2, 1, false, 8, __nv_bfloat16, 0, true, 0>(w, x, out, (int)n, (int)k, dg); break;
    case 1: launch_wd<1, 128, 3, 1, false, 8, __nv_bfloat16, 0, true, 0>(w, x, out, (int)n, (int)k, dg); break;
    default:
        TORCH_CHECK(false, "wd: unknown config ", cfg);
    }
}
#endif
'''

_SKP_WD_GEOM = {
    'b1n128c2s8h': (1, 128, 2, 1, False, 8),
    'b1n128c3s8h': (1, 128, 3, 1, False, 8),
}

_SKP_CPP_SRC = r\"\"\"
#include <torch/extension.h>
void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor diag);
void skf_fwd(int64_t cfg, at::Tensor ws, at::Tensor rws, at::Tensor x, at::Tensor out, at::Tensor rout, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor rpart, at::Tensor rcount, at::Tensor diag);

void cgemv_fwd(at::Tensor stream, at::Tensor table, int64_t pmax, int64_t base4, at::Tensor x, at::Tensor out, int64_t nthreads);
void rgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out);
void sgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out);
void wd_fwd(int64_t cfg, at::Tensor w, at::Tensor x, at::Tensor out, int64_t n, int64_t k, at::Tensor diag);
\"\"\"

W4_BITPOS = (
    (6, 7, 8, 15), (22, 23, 24, 31),
    (0, 1, 2, 9), (16, 17, 18, 25),
    (12, 13, 14, 21), (28, 29, 30, 5),
    (10, 3, 4, 11), (26, 19, 20, 27),
)

def _perm(a, dims):
    return a.permute(*dims)

def _stack_last(a, b):
    return torch.stack((a, b), dim=-1)

def _to_i64(a):
    return a.to(torch.int64)

def _w4_scales_ok(sf_bytes):
    pass
    b = sf_bytes.to(torch.int32)
    exp = (b >> 3) & 15
    return bool(((b & 0x80) == 0).all() and (exp > 0).all() and (b != 0x7F).all())

def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)

def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)

def _norm(cls, w, size, eps, **kw):
    pass
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm

def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate

def _moe1_load():
    return load_inline(
        name='okc_decode_layer_moe1_coop_trap_v1_startup_attrs_v1',
        cpp_sources=_MOE1_CPP_SRC,
        cuda_sources=[_MOE1_SRC, _moe1_source()],
        functions=['moe_fused_run', 'qual_moe_resources'],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    )

def moe1_unit_rows(a, kind):
    pass
    n = a.shape[0]
    if kind == 1:
        return _perm(a.reshape(2, n // 16, 8, *a.shape[1:]), (1, 0, 2, 3))
    return a.reshape(n // 16, 2, 8, *a.shape[1:])

def moe1_repack_expert(packed, sf, kind):
    pass
    n, k2 = packed.shape
    k = 2 * k2
    codes = _stack_last(packed & 15, packed >> 4).reshape(n, k)
    c = moe1_unit_rows(codes, kind)
    u = c.shape[0]
    
    
    c = c.reshape(u, 2, 8, 8, 8, 4, 4, 2, 2)
    c = _to_i64(_perm(c, (0, 3, 4, 2, 5, 6, 7, 1, 8)).reshape(-1, 8))
    words = c[:, 0] * 0
    for v in range(8):
        for b in range(4):
            words = words | (((c[:, v] >> b) & 1) << W4_BITPOS[v][b])
    words = _to_u8_words(words).reshape(u, 32768)
    s = moe1_unit_rows(sf, kind).reshape(u, 2, 8, 8, 8, 4)
    s = _perm(s, (0, 3, 2, 5, 4, 1)).reshape(u, 4096)
    return _cat_last(words, s)

def moe1_repack_expert_fat(packed, sf, kind):
    pass
    n, k2 = packed.shape
    k = 2 * k2
    codes = _stack_last(packed & 15, packed >> 4).reshape(n, k)
    c = moe1_unit_rows(codes, kind)
    nu = c.shape[0] // 16
    
    c = c.reshape(nu, 4, 4, 2, 8, 16, 4, 4, 4, 2, 2)
    c = _to_i64(_perm(c, (0, 5, 1, 2, 6, 4, 7, 8, 9, 3, 10)).reshape(-1, 8))
    words = c[:, 0] * 0
    for v in range(8):
        for b in range(4):
            words = words | (((c[:, v] >> b) & 1) << W4_BITPOS[v][b])
    words = _to_u8_words(words).reshape(nu * 16, 32768)
    
    sc = moe1_unit_rows(sf, kind).reshape(nu, 4, 4, 2, 8, 16, 4, 4)
    sc = _perm(sc, (0, 5, 1, 2, 4, 7, 6, 3)).reshape(nu * 16, 4096)
    return _cat_last(words, sc).reshape(nu, 16, 36864)

def _to_u8_words(words):
    pass
    w = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return w.contiguous().view(torch.uint8)

def _cat_last(a, b):
    return torch.cat((a, b), dim=-1).contiguous()

MOE1_TOKENS = 512

_MOE1_EMAX = MOE1_TOKENS * TOP_K

_MOE1_JMAX = LOCAL_EXPERTS + _MOE1_EMAX // 8

_MOE1_JMAX_MAIN = LOCAL_EXPERTS + _MOE1_EMAX // 64

def _moe1_scratch(device):
    pass
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    xbytes = (_MOE1_EMAX // 8 + _MOE1_JMAX_MAIN) * 65536
    return {
        "inter": torch.zeros((_MOE1_EMAX, D_EXPERT), dtype=torch.bfloat16, device=device),
        "ybuf": torch.zeros((_MOE1_EMAX, D_EXPERT), dtype=torch.float32, device=device),
        "ss": torch.zeros((2, _MOE1_EMAX), dtype=torch.float32, device=device),
        "ctr": torch.zeros((64,), dtype=torch.int32, device=device),
        "done_a": torch.zeros((_MOE1_JMAX,), dtype=torch.int32, device=device),
        "xa": torch.zeros((xbytes,), dtype=torch.uint8, device=device),
        "xb": torch.zeros((xbytes,), dtype=torch.uint8, device=device),
        "fix": torch.zeros((2 * sms * 4 * 128 * 32,), dtype=torch.float32, device=device),
        "fixcnt": torch.zeros((2 * sms * 4,), dtype=torch.int32, device=device),
        "xready": torch.zeros((_MOE1_JMAX_MAIN,), dtype=torch.int32, device=device),
        "done_as": torch.zeros((_MOE1_JMAX_MAIN * 16,), dtype=torch.int32, device=device),
    }

def _moe1_prepare(weights):
    pass
    out = {}
    for name, kind in (("w13", 1), ("w2", 2)):
        packed = weights[name]
        sf = weights[name + "_sf"].view(torch.uint8)
        if not _w4_scales_ok(sf):
            return None
        out[name] = torch.stack([moe1_repack_expert(packed[e], sf[e], kind)
                                 for e in range(packed.shape[0])])
        out[name + "f"] = torch.stack([moe1_repack_expert_fat(packed[e], sf[e], kind)
                                       for e in range(packed.shape[0])])
        out[name + "_gs"] = weights[name + "_gs"].float().contiguous()
    torch.cuda.synchronize()
    return out

def _moe1_run(x, topk_weights, topk_ids, local_start, state, variant, pdl=False):
    pass
    st = state["moe1"]
    w, s = st["w"], st["scratch"]
    return _moe1_ext().moe_fused_run(
        x, topk_ids, topk_weights, local_start, w["w13"], w["w2"], w["w13_gs"], w["w2_gs"],
        st["nw"], s["inter"], s["ybuf"], s["ss"], s["ctr"], s["done_a"], w["w13f"], w["w2f"],
        s["xa"], s["xb"], s["fix"], s["fixcnt"], s["xready"], s["done_as"], LOCAL_EXPERTS,
        POST_NORM_EPS, variant, pdl)

def _skp_load():
    src = _SKP_SRC + _SKP_WD_SRC
    kwargs = dict(name="okc_decode_layer_skp", cpp_sources=_SKP_CPP_SRC, cuda_sources=src,
                  functions=["wd_fwd"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
                  + (["-DSK_PDL=1"] if PORT_SK_PDL else []))
    try:
        return load_inline(no_implicit_headers=True, **kwargs)
    except TypeError:
        
        kwargs["cuda_sources"] = src.replace("#include <ATen/core/Tensor.h>",
                                             "#include <torch/types.h>")
        return load_inline(**kwargs)

def _skp_wd_runner(ext, name, w, diag):
    cid = list(_SKP_WD_GEOM).index(name)
    n, k = w.shape

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run

LOG2E = 1.4426950408889634

_ONE_CPP_SRC = r'''
#include <torch/extension.h>
std::vector<int64_t> attn_one_prepare();
int64_t attn_one_probe_coop(bool pdl);
torch::Tensor attn_one(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                       torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                       torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                       torch::Tensor ws_acc, torch::Tensor ws_l, torch::Tensor ws_cnt,
                       int64_t batch, int64_t q_len, int64_t page_size, int64_t window, bool rope,
                       double sl2, int64_t n_splits, int64_t nc, int64_t mode, bool coop_launch,
                       bool pdl, torch::Tensor dbg);

std::vector<int64_t> qual_one_resources();
'''

def _one_load():
    return load_inline(
        name='okc_decode_layer_one_startup_attrs_v1',
        cpp_sources=_ONE_CPP_SRC,
        cuda_sources=[_ONE_SRC, _ONE_HOST_SRC],
        functions=['attn_one', 'attn_one_prepare', 'attn_one_probe_coop', 'qual_one_resources'],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
        + (["-DONE_TRIG=1"] if PORT_SK_PDL else []),
    )

_ONE_WS_ROWS = 65536

_ONE_CNT_INTS = 32768

_ONE_TARGET = 264

_ONE_BUILT = {1: (1, 3, 11, 13, 49), 8: (1, 3, 11, 15, 19, 51)}

_ONE_FLAT = True

_ONE_PROD = True

_ONE_COOP_LAUNCH = True

_ONE_PDL = False

def _one_setup(state):
    pass
    assert not torch.cuda.is_current_stream_capturing(), "attn_one setup inside a capture"
    ext = _one_ext()
    device = state["inv_freq"].device
    lim = [int(v) for v in ext.attn_one_prepare()]
    pdl = _ONE_PDL
    coop = False
    if _ONE_COOP_LAUNCH:
        coop = bool(ext.attn_one_probe_coop(pdl))
        if not coop and pdl:
            pdl = False
            coop = bool(ext.attn_one_probe_coop(False))
    resident = {}
    for i in range(0, len(lim), 5):
        length, mode, res, _, per_sm = lim[i:i + 5]
        
        
        resident[(length, mode)] = res if (coop or not mode & 2) else max(0, res - per_sm)
    state["one"] = {
        "acc": torch.zeros((_ONE_WS_ROWS, 64), dtype=torch.float32, device=device),
        "l": torch.zeros((_ONE_WS_ROWS,), dtype=torch.float32, device=device),
        "cnt": torch.zeros((_ONE_CNT_INTS,), dtype=torch.int32, device=device),
        "resident": resident, "coop_launch": coop, "pdl": pdl, "limits": lim,
        "dbg": torch.zeros(32, dtype=torch.int32, device=device),
        "nodbg": torch.empty(0, dtype=torch.int32, device=device),
    }
    return state["one"]

def _one_plan(cfg, pt_width, resident):
    pass
    batch, length, psize, window = cfg["batch"], cfg["q_len"], cfg["page_size"], cfg["window"]
    kmax = (window + length - 1) if window else pt_width * psize
    tiles = (kmax + 63) // 64
    nc = 1
    if not window and batch > 1 and _ONE_FLAT:
        
        
        for mode in ((49, 17) if length == 1 else (51, 19)) if _ONE_PROD else \\
                ((17,) if length == 1 else (19,)):
            grid = min(_ONE_TARGET, resident.get((length, mode), 0))
            if grid >= 2 * batch:
                return grid, 1, mode
    if window:
        if length == 1:
            n = min(tiles, max(1, _ONE_TARGET // batch))
            coop = False
        else:
            n = min(tiles, max(1, (264 if batch >= 24 else 132) // batch))
            if batch == 32:
                n = min(tiles, 4)
            coop = True
    elif batch == 1:
        n = min(tiles, 252)
        nc = 8
        coop = True
    else:
        n = min(tiles, max(1, _ONE_TARGET // batch))
        coop = length == 8
    nc = max(1, min(nc, _ONE_WS_ROWS // (batch * length * 16)))
    one = n >= tiles
    spec = batch == 1 and psize == 1
    built = _ONE_BUILT[length]
    candidates = []
    for cp in ((True, False) if coop else (False,)):
        for on in ((True, False) if one else (False,)):
            for sp in ((True, False) if spec else (False,)):
                m = (3 if cp else 1) | (4 if on else 0) | (8 if sp else 0)
                if m in built and (not cp or n * batch <= resident.get((length, m), 0)):
                    candidates.append(m)
    return n, nc, candidates[0]

def _attn_one(qkv, data, state, dbg=False):
    pass
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    st = state["one"]
    n, c, m = _one_plan(cfg, data["page_table"].shape[1], st["resident"])
    return _one_ext().attn_one(
        qkv.contiguous(), data["page_table"], data["seq_lens"], data["cache_loc"],
        state["inv_freq"], data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"],
        st["acc"], st["l"], st["cnt"], batch, length, cfg["page_size"], cfg["window"],
        bool(cfg["rope_theta"]), cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, n, c, m,
        st["coop_launch"], st["pdl"], st["dbg"] if dbg else st["nodbg"])

ABND_GEOS = ("c8x256", "c2x512", "c4x256", "c1x256", "c1x512", "c1x1024")

def _abnd(stream, branch, want_router, state, geo="c8x256"):
    if geo == "sglang":
        return state["attn_boundary"](stream, branch, branch_dtype=torch.bfloat16,
                                      router_dtype=torch.float32 if want_router else None)
    abnd = state["abnd"]
    nxt, out, router = _ext().attn_boundary(stream.contiguous(), branch.contiguous(),
                                            abnd["alpha"], abnd["beta"], abnd["w_next"],
                                            abnd["eps_post"], abnd["eps_next"], want_router,
                                            ABND_GEOS.index(geo))
    return nxt, out, (router if want_router else None)

def _route_fwd(logits, lo_off, pooled, state, bias=None):
    return _ext().route_fwd(logits, lo_off, pooled, state["pre_expert_w"],
                            state["route_bias"] if bias is None else bias,
                            ROUTER_MULTIPLIER, RMS_EPS, ROUTE_EPS)

_DLPLT_CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

 
 
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <vector>

namespace {

using LtCfg = std::array<int64_t, 9>;

 
 
 
cublasLtMatmulAlgoConfigAttributes_t cfg_attr(int i) {
    return (cublasLtMatmulAlgoConfigAttributes_t)i;
}

LtCfg cfg_of(const cublasLtMatmulAlgo_t& a) {
    LtCfg c{};
    for (int i = 0; i < 9; ++i) {
        size_t wr = 0;
        if (i >= 7) {
            uint16_t v = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&a, cfg_attr(i), &v, sizeof(v), &wr);
            c[i] = v;
        } else {
            uint32_t v = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&a, cfg_attr(i), &v, sizeof(v), &wr);
            c[i] = v;
        }
    }
    return c;
}

bool cfg_set(cublasLtMatmulAlgo_t& a, int i, int64_t value) {
    if (i >= 7) {
        uint16_t v = (uint16_t)value;
        return cublasLtMatmulAlgoConfigSetAttribute(&a, cfg_attr(i), &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
    }
    uint32_t v = (uint32_t)value;
    return cublasLtMatmulAlgoConfigSetAttribute(&a, cfg_attr(i), &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
}

uint32_t cap_u32(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
    uint32_t v = 0;
    size_t wr = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, &v, sizeof(v), &wr) != CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    return v;
}

std::vector<uint32_t> cap_list(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
    size_t bytes = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, nullptr, 0, &bytes) != CUBLAS_STATUS_SUCCESS || bytes == 0) {
        return {};
    }
    std::vector<uint32_t> v(bytes / sizeof(uint32_t));
    if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, v.data(), bytes, &bytes) != CUBLAS_STATUS_SUCCESS) {
        return {};
    }
    return v;
}

cublasLtHandle_t g_handle = nullptr;

struct LtPlan {
    int m = 0, n = 0, k = 0, layout = 0;
    size_t ws_cap = 0;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    std::vector<cublasLtMatmulAlgo_t> algos;
    std::vector<size_t> ws;
    std::vector<float> waves;
    std::set<LtCfg> seen;
    int rejected = 0;

    ~LtPlan() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (la) cublasLtMatrixLayoutDestroy(la);
        if (lb) cublasLtMatrixLayoutDestroy(lb);
        if (ld) cublasLtMatrixLayoutDestroy(ld);
    }

    // Accepts an algorithm only with FP32 accumulation and either no split-K or a split-K
    // reduced in the compute type: the in-place and output-type schemes sum BF16-rounded
    // partials, which SGLang's own split-K (splitKreduce in FP32) does not.
    bool add(const cublasLtMatmulAlgo_t& a) {
        LtCfg c = cfg_of(a);
        if (!seen.insert(c).second) return false;
        if (c[2] > 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) { ++rejected; return false; }
        if (c[2] <= 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_NONE && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) {
            ++rejected;
            return false;
        }
        uint64_t flags = 0;
        size_t wr = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &flags, sizeof(flags), &wr)
            != CUBLAS_STATUS_SUCCESS) {
            ++rejected;
            return false;
        }
        if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F
            || (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)) {
            ++rejected;
            return false;
        }
        cublasLtMatmulHeuristicResult_t r;
        std::memset(&r, 0, sizeof(r));
        if (cublasLtMatmulAlgoCheck(g_handle, op, la, lb, ld, ld, &a, &r) != CUBLAS_STATUS_SUCCESS
            || r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > ws_cap) {
            ++rejected;
            return false;
        }
        algos.push_back(a);
        ws.push_back(r.workspaceSize);
        waves.push_back(r.wavesCount);
        return true;
    }
};

std::vector<std::unique_ptr<LtPlan>> g_plans;

LtPlan* plan_at(int64_t p) {
    return (p >= 0 && p < (int64_t)g_plans.size()) ? g_plans[p].get() : nullptr;
}

}   

int64_t lt_version() {
    return (int64_t)cublasLtGetVersion();
}

 
 
 
 
 
 
int64_t lt_plan_new(int64_t m, int64_t n, int64_t k, int64_t layout, int64_t ws_cap) {
    if (!g_handle && cublasLtCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return -1;
    auto p = std::make_unique<LtPlan>();
    p->m = (int)m;
    p->n = (int)n;
    p->k = (int)k;
    p->layout = (int)layout;
    p->ws_cap = (size_t)ws_cap;
    if (cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return -2;
    const bool row = layout >= 2;
    const bool packed = layout % 2 == 1;
    cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_N;
    if (!row) {
        ta = packed ? CUBLAS_OP_N : CUBLAS_OP_T;
        if (packed) {
            cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, n, k, n);
        } else {
            cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, k, n, k);
        }
        cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, k, m, k);
        cublasLtMatrixLayoutCreate(&p->ld, CUDA_R_16BF, n, m, n);
    } else {
        tb = packed ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, m, k, k);
        if (packed) {
            cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, k, n, n);
        } else {
            cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, n, k, k);
        }
        cublasLtMatrixLayoutCreate(&p->ld, CUDA_R_16BF, m, n, n);
        cublasLtOrder_t ord = CUBLASLT_ORDER_ROW;
        for (auto l : {p->la, p->lb, p->ld}) {
            if (!l || cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &ord, sizeof(ord))
                       != CUBLAS_STATUS_SUCCESS) {
                return -3;
            }
        }
    }
    if (!p->la || !p->lb || !p->ld) return -4;
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) != CUBLAS_STATUS_SUCCESS) {
        return -5;
    }
    g_plans.push_back(std::move(p));
    return (int64_t)g_plans.size() - 1;
}

 
int64_t lt_heuristic(int64_t plan, int64_t ws_bytes, int64_t req) {
    LtPlan* p = plan_at(plan);
    if (!p) return -1;
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return -2;
    uint64_t wsb = (uint64_t)std::min<int64_t>(ws_bytes, (int64_t)p->ws_cap);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
    std::vector<cublasLtMatmulHeuristicResult_t> res((size_t)req);
    int got = 0;
    int64_t added = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_handle, p->op, p->la, p->lb, p->ld, p->ld, pref, (int)req, res.data(), &got)
        == CUBLAS_STATUS_SUCCESS) {
        for (int i = 0; i < got; ++i) {
            if (res[i].state == CUBLAS_STATUS_SUCCESS && p->add(res[i].algo)) ++added;
        }
    }
    cublasLtMatmulPreferenceDestroy(pref);
    return added;
}

 
 
int64_t lt_expand(int64_t plan, int64_t idx, int64_t max_add) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return -1;
    const cublasLtMatmulAlgo_t base = p->algos[(size_t)idx];
    const size_t before = p->algos.size();
    auto room = [&]() { return (int64_t)(p->algos.size() - before) < max_add; };
    for (uint32_t s : cap_list(base, CUBLASLT_ALGO_CAP_STAGES_IDS)) {
        if (!room()) break;
        auto a = base;
        if (cfg_set(a, 6, s)) p->add(a);
    }
    int32_t sk = 0;
    size_t wr = 0;
    cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &sk, sizeof(sk), &wr);
    if (sk && (cap_u32(base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for (int s : {2, 3, 4, 6, 8}) {
            if (!room()) break;
            auto a = base;
            if (cfg_set(a, 2, s) && cfg_set(a, 3, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) p->add(a);
        }
    }
    const int swz = (int)cap_u32(base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
    const int cust = (int)std::min<uint32_t>(cap_u32(base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX), 15u);
    for (int z = 0; z <= swz; ++z) {
        for (int c = 0; c <= cust; ++c) {
            if (!room()) break;
            auto a = base;
            if (cfg_set(a, 4, z) && cfg_set(a, 5, c)) p->add(a);
        }
    }
    for (int cl = 0; cl <= 24; ++cl) {
        if (!room()) break;
        auto a = base;
        if (cfg_set(a, 8, cl)) p->add(a);
    }
    return (int64_t)(p->algos.size() - before);
}

 
 
int64_t lt_sweep(int64_t plan, int64_t idx, std::vector<int64_t> clusters, int64_t max_add) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return -1;
    cublasLtMatmulAlgo_t base = p->algos[(size_t)idx];
    cfg_set(base, 2, 1);
    cfg_set(base, 3, CUBLASLT_REDUCTION_SCHEME_NONE);
    cfg_set(base, 4, 0);
    cfg_set(base, 5, 0);
    const size_t before = p->algos.size();
    for (uint32_t t : cap_list(base, CUBLASLT_ALGO_CAP_TILE_IDS)) {
        for (int64_t cl : clusters) {
            if ((int64_t)(p->algos.size() - before) >= max_add) return (int64_t)(p->algos.size() - before);
            auto a = base;
            if (cfg_set(a, 1, t) && cfg_set(a, 8, cl)) p->add(a);
        }
    }
    return (int64_t)(p->algos.size() - before);
}

 
int64_t lt_add_config(int64_t plan, std::vector<int64_t> c) {
    LtPlan* p = plan_at(plan);
    if (!p || c.size() != 9) return -1;
    cublasLtMatmulAlgo_t a;
    std::memset(&a, 0, sizeof(a));
    if (cublasLtMatmulAlgoInit(g_handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                               CUDA_R_16BF, (int)c[0], &a) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    for (int i = 1; i < 9; ++i) {
        if (!cfg_set(a, i, c[i]) && c[i] != 0) return -1;
    }
    LtCfg want{};
    for (int i = 0; i < 9; ++i) want[i] = c[i];
    for (size_t j = 0; j < p->algos.size(); ++j) {
        if (cfg_of(p->algos[j]) == want) return (int64_t)j;
    }
    if (!p->add(a)) return -1;
    return cfg_of(p->algos.back()) == want ? (int64_t)p->algos.size() - 1 : -1;
}

int64_t lt_count(int64_t plan) {
    LtPlan* p = plan_at(plan);
    return p ? (int64_t)p->algos.size() : -1;
}

int64_t lt_rejected(int64_t plan) {
    LtPlan* p = plan_at(plan);
    return p ? (int64_t)p->rejected : -1;
}

// The nine config fields, then workspace bytes and 1000 x the reported waves.
std::vector<int64_t> lt_info(int64_t plan, int64_t idx) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return {};
    LtCfg c = cfg_of(p->algos[(size_t)idx]);
    std::vector<int64_t> v(c.begin(), c.end());
    v.push_back((int64_t)p->ws[(size_t)idx]);
    v.push_back((int64_t)(1000.0f * p->waves[(size_t)idx]));
    return v;
}

 
int64_t lt_run_raw(int64_t plan, int64_t idx, int64_t w, int64_t x, int64_t out, int64_t ws, int64_t ws_bytes,
                   int64_t stream) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return -1;
    if ((int64_t)p->ws[(size_t)idx] > ws_bytes) return -2;
    const float alpha = 1.0f, beta = 0.0f;
    const bool row = p->layout >= 2;
    const void* a = row ? (const void*)x : (const void*)w;
    const void* b = row ? (const void*)w : (const void*)x;
    return (int64_t)cublasLtMatmul(g_handle, p->op, &alpha, a, p->la, b, p->lb, &beta, (void*)out, p->ld, (void*)out,
                                   p->ld, &p->algos[(size_t)idx], (void*)ws, p->ws[(size_t)idx],
                                   (cudaStream_t)stream);
}

int64_t lt_run(int64_t plan, int64_t idx, at::Tensor w, at::Tensor x, at::Tensor out, at::Tensor ws) {
  return lt_run_raw(plan, idx, (int64_t)w.data_ptr(), (int64_t)x.data_ptr(), (int64_t)out.data_ptr(),
                    (int64_t)ws.data_ptr(), ws.numel(), (int64_t)at::cuda::getCurrentCUDAStream().stream());
}
'''

_DLPLT_FUNCS = ['lt_version', 'lt_plan_new', 'lt_heuristic', 'lt_expand', 'lt_sweep', 'lt_add_config', 'lt_count', 'lt_rejected', 'lt_info', 'lt_run']

def _dlplt_build():
    
    errs = []
    for i, flag in enumerate(("-l:libcublasLt.so.13", "-lcublasLt", "-l:libcublasLt.so.12")):
        try:
            return load_inline(name=f"okc_dlplt_v1_{i}", cpp_sources=_DLPLT_CPP, functions=_DLPLT_FUNCS, with_cuda=True,
                               extra_cflags=["-O2", "-std=c++20"], extra_ldflags=[flag], verbose=False)
        except Exception as exc:
            msg = str(exc)
            lines = [ln.strip() for ln in msg.splitlines() if "error" in ln.lower() or "cannot find" in ln]
            errs.append(f"{flag}: " + " | ".join(lines[:4])[:500] + " ... " + msg[-200:])
    raise RuntimeError(" || ".join(errs))

_slice_main=_slice_moe=_slice_one=None

def _ext():
    global _slice_main
    if _slice_main is None:
        _slice_main = load_inline(name="okc_owned_slice_main", cpp_sources=_CPP_SRC,
            cuda_sources=[_CUDA_SRC, _BND_SRC], functions=["route_fwd", "attn_boundary"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    return _slice_main

def _moe1_ext():
    global _slice_moe
    if _slice_moe is None:
        _slice_moe = _moe1_load()
    return _slice_moe

def _one_ext():
    global _slice_one
    if _slice_one is None:
        _slice_one = _one_load()
    return _slice_one

def _moe1_source():
    
    return _MOE1_HOST_SRC.replace("//@MOE1_DISPATCH@",
        "        case 10: launch_main<51>(p, sms, stream); break;")

def _slice_proj(x, layer, state):
    if layer == "qkv_gate" and x.shape[0] == 128:
        return state["verify_qkv128"](x.contiguous())
    entry = state["dlplt"].get((layer, x.shape[0]))
    w = state["dense"][layer]
    if entry is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 256 == 0:
        return entry(x)
    return _gemm(x, w)

def _slice_lt(state):
    lt = _dlplt_build()
    state["dlplt"] = {}
    for layer, m, layout, cfg in (('unpooler', 128, 3, [66, 18, 1, 0, 0, 2, 35, 0, 5]), ('unpooler', 256, 1, [66, 20, 1, 0, 0, 4, 35, 0, 3]), ('qkv_gate', 256, 0, [66, 124, 1, 0, 0, 0, 35, 0, 2])):
        w = state["dense"][layer]; n,k = w.shape
        ws = torch.zeros(32 << 20,dtype=torch.uint8,device=w.device)
        wa = w.t().contiguous() if layout % 2 else w
        plan = lt.lt_plan_new(m,n,k,layout,ws.numel())
        idx = lt.lt_add_config(plan,list(cfg)) if plan >= 0 else -1
        assert idx >= 0, ("verify slice Lt pin rejected", layer, m)
        def run(x, plan=plan, idx=idx, wa=wa, ws=ws, n=n, m=m):
            out = torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
            rc = lt.lt_run(plan,idx,wa,x,out,ws)
            if rc: raise RuntimeError(f"verify slice Lt status {rc}")
            return out
        state["dlplt"][layer,m] = run

def prepare(weights):
    device = weights["qkv_gate"].device
    pre = _norm(MuseSpark1xRMSNorm,weights["pre_ffn_norm"],HIDDEN,RMS_EPS)
    nxt = _norm(MuseSpark1xRMSNorm,weights["next_pre_attn_norm"],HIDDEN,RMS_EPS)
    post = _norm(MuseSpark1xRMSNorm,None,HIDDEN,POST_NORM_EPS,is_post_norm=True,
                 post_norm_gain_center_type="one",residual_id=1)
    postffn = _norm(MuseSpark1xRMSNormWithInputScale,weights["post_ffn_norm"],HIDDEN,
        POST_NORM_EPS,is_post_norm=True,post_norm_gain_center_type="one",residual_id=2)
    poste = _norm(MuseSpark1xRMSNormWithInputScale,weights["post_expert_norm"],D_EXPERT,
        POST_NORM_EPS,is_post_norm=True,post_norm_gain_center_type="one",residual_id=2)
    pren = _norm(MuseSpark1xRMSNorm,weights["pre_expert_norm"],D_EXPERT,RMS_EPS)
    ar = _residual(weights["attn_residual_gate"])
    fr = _residual(weights["ffn_residual_gate"])
    router = weights["router"]; hi=router.to(torch.bfloat16); lo=(router-hi.float()).to(torch.bfloat16)
    state={"dense":{k:weights[k].contiguous() for k in ("qkv_gate","o","pooler","unpooler")},
      "inv_freq":1.0/(ROPE_THETA**(torch.arange(0,64,2,dtype=torch.float,device=device)/64)),
      "pre_expert_w":pren._get_effective_weight().detach().to(torch.bfloat16).contiguous(),
      "route_bias":weights["router_bias"].float().contiguous(),
      "pool_router":torch.cat((weights["pooler"],hi,lo),0).contiguous(),
      "abnd":{"alpha":ar._alpha.float().reshape(-1).contiguous(),
              "beta":ar._beta.float().reshape(-1).contiguous(),
              "w_next":pre._get_effective_weight().detach().to(torch.bfloat16).contiguous(),
              "eps_post":float(post.eps),"eps_next":float(pre.eps)},
      "attn_boundary":FusedPostNormResidualPreNorm(post,ar,pre),
      "ffn_boundary":FusedPostNormResidualPreNorm(postffn,fr,nxt),
      "_modules":(pre,nxt,post,postffn,poste,pren,ar,fr)}
    _ext(); _one_setup(state)
    state["verify_diag"] = torch.zeros(4,dtype=torch.int32,device=device)
    state["verify_qkv128"] = _skp_wd_runner(_skp_load(),"b1n128c3s8h",
        state["dense"]["qkv_gate"],state["verify_diag"])
    for batch in (16, 32):
        for window in (0,2048):
            expected = (264,1,51) if not window else (8 if batch==16 else 4,1,3)
            assert _one_plan({"batch":batch,"q_len":8,"page_size":1,"window":window},86008,
                state["one"]["resident"]) == expected, "verify slice attention selection drift"
    w=_moe1_prepare(weights)
    if w is None: raise RuntimeError("slice: unsupported expert scales; no fallback")
    state["moe1"]={"w":w,"nw":poste._get_effective_weight().detach().to(torch.bfloat16).float().contiguous(),
                   "scratch":_moe1_scratch(device)}
    _moe1_ext(); _slice_lt(state)
    torch.cuda.synchronize()
    print("OWNED_VERIFY_SLICE verify16/32 sliding/full, QKV128 WD, SG128 boundary, MOE51, three Lt pins; no autotuning",flush=True)
    return state

def kernel_fn(data,state):
    c=data["config"]
    batch=c.get("batch"); window=c.get("window")
    assert batch in (16, 32) and window in (0,2048), "unsupported verify slice config"
    assert all(c.get(k)==v for k,v in {"batch":batch,"q_len":8,"tokens":batch*8,"page_size":1,
        "window":window,"hidden":8192,"num_q_heads":16,"num_kv_heads":2,"head_dim":64,
        "n_experts":256,"top_k":8,"local_experts":32,"kv_scale":1.0}.items()), "verify slice geometry drift"
    assert data["page_table"].shape==(batch,86008), "unsupported verify slice context capacity"
    qkv=_slice_proj(data["attn_input"],"qkv_gate",state)
    o=_attn_one(qkv,data,state)
    attn_output=_slice_proj(o,"o",state)
    ffn,stream,_=_abnd(data["stream"],attn_output,False,state,"sglang" if batch==16 else "c1x512")
    z=torch.mm(ffn,state["pool_router"].t(),out_dtype=torch.float32)
    x,ids,wts=_route_fwd(z[:,D_EXPERT:],N_EXPERTS,z[:,:D_EXPERT],state)
    expert=_moe1_run(x,wts,ids,data["local_start"],state,10,False)
    out=_slice_proj(expert,"unpooler",state)
    nxt,stream,_=state["ffn_boundary"](stream,out,branch_dtype=torch.bfloat16)
    return nxt,stream,ids
"""
_VERIFY_HYBRID_OWNED = _verify_hybrid_module('_okc_verify_production_owned', _VERIFY_HYBRID_SOURCE_OWNED)

"""Production shape dispatcher; constants and modules are supplied by the builder.

The mask is fixed in the emitted file, never selected by timings or input values.
Each bit chooses one complete owned verify pipeline, including its BF16 boundaries.
"""


def _verify_owned_metadata(data, batch):
    # Raw native ABIs assume these layouts. Inspect metadata only; tensor values
    # (including scales, routing and cache lengths) remain runtime inputs.
    device = data["attn_input"].device
    specs = {"attn_input": ("torch.bfloat16", (batch * 8, 8192)),
             "stream": ("torch.float32", (batch * 8, 8192)),
             "page_table": ("torch.int32", (batch, 86008)),
             "seq_lens": ("torch.int32", (batch,)),
             "cache_loc": ("torch.int64", (batch * 8,)),
             "local_start": ("torch.int32", (1,)),
             "k_scale": ("torch.float32", ()),
             "v_scale": ("torch.float32", ())}
    for name, (dtype, shape) in specs.items():
        value = data[name]
        if (str(value.dtype) != dtype or value.shape != shape or
                value.device != device or not value.is_contiguous()):
            return False
    k, v = data["k_cache"], data["v_cache"]
    return (k.shape == v.shape and len(k.shape) == 4 and k.shape[0] > 0 and
            k.shape[1:] == (1, 2, 64) and
            all(str(value.dtype) == "torch.float8_e4m3fn" and
                value.device == device and value.is_contiguous() for value in (k, v)))


def _verify_override_bit(data):
    config = data["config"]
    batch = config.get("batch")
    window = config.get("window")
    if batch not in (16, 32) or window not in (0, 2048):
        return 0
    expected = {"batch": batch, "q_len": 8, "tokens": batch * 8,
                "window": window, "page_size": 1, "hidden": 8192,
                "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64,
                "n_experts": 256, "local_experts": 32, "top_k": 8,
                "kv_scale": 1.0}
    if any(config.get(key) != value for key, value in expected.items()):
        return 0
    if data["page_table"].shape != (batch, 86008):
        return 0
    if config.get("rope_theta") != (500000.0 if window else 0.0):
        return 0
    index = (0 if batch == 16 else 2) + (0 if window == 2048 else 1)
    bit = (1 << index) & _VERIFY_OVERRIDE_MASK
    return bit if bit and _verify_owned_metadata(data, batch) else 0


def _verify_owned_weights_eligible(weights):
    # Exactly the rejection predicate in the frozen owned _moe1_prepare.
    # This reads immutable weights during prepare, never timed input values.
    owned = _VERIFY_HYBRID_OWNED
    return all(owned._w4_scales_ok(weights[name + "_sf"].view(owned.torch.uint8))
               for name in ("w13", "w2"))


def prepare(weights):
    import time
    deadline = time.monotonic() + 295
    # These states share only immutable input weights/code. Each backend creates
    # its own scratch, module buffers, repacked weights and Lt workspace closures.
    state = {"G": _VERIFY_HYBRID_G.prepare(weights)}
    if time.monotonic() >= deadline:
        raise RuntimeError("verify hybrid incumbent preparation exceeded budget")
    if _VERIFY_OVERRIDE_MASK and _verify_owned_weights_eligible(weights):
        state["owned"] = _VERIFY_HYBRID_OWNED.prepare(weights)
    if time.monotonic() >= deadline:
        raise RuntimeError("verify hybrid preparation exceeded budget")
    return state


def kernel_fn(data, state):
    if "owned" in state and _verify_override_bit(data):
        return _VERIFY_HYBRID_OWNED.kernel_fn(data, state["owned"])
    return _VERIFY_HYBRID_G.kernel_fn(data, state["G"])

# Single-arm startup checks; timed kernel_fn unchanged.
import torch,time,math

import time
FP8=torch.float8_e4m3fn
RMS_EPS=1e-5
POST_NORM_EPS=1e-8
ROUTER_MULTIPLIER=0.1767766952966369
ROUTE_EPS=1e-15
SLICED_UNPOOL_MAX_TOKENS=8
UNPOOL_SLICE=1024
_E2M1=[0.,.5,1.,1.5,2.,3.,4.,6.]
_AB_DEQ={}
_REF_STREAM_SEED,_REF_STREAM_MEAN=5354,0.7
_REF_STREAM_SHARED,_REF_STREAM_RANK,_REF_DRAFT_SHARE=.9,4,.6

HIDDEN=8192
D_EXPERT=4096
LOCAL_EXPERTS=32
N_EXPERTS=256
TOP_K=8
ROPE_THETA=500000.
def _ab_inputs(batch, q_len, layer, seed):
    """reference.py's generate_input for a scored case (ctx 80000, page 1, dist soft, spread
    agentic above batch 1, kv_scale 1), local_start 0."""
    import math
    dev = "cuda"
    ctx, page = 80000, 1
    gen = torch.Generator(device=dev).manual_seed(seed)
    tokens = batch * q_len
    if batch > 1:
        longest = math.ceil(ctx * 1.075)
        u = torch.rand((batch,), generator=gen, device=dev)
        seq_lens = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, longest)
    else:
        longest = ctx
        seq_lens = torch.full((batch,), ctx, dtype=torch.int64, device=dev)
    pages_per = (seq_lens + q_len + page - 1) // page
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=dev).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=dev)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page
                 + pos % page).reshape(-1)
    shape = (total, page, 2, 64)
    soft = torch.Generator(device=dev).manual_seed(seed + 0x50F7)
    uu = torch.randn((2, 64), generator=soft, device=dev)
    uu = uu * torch.rsqrt(uu.pow(2).mean(-1, keepdim=True))
    lo, hi = (0.3, 0.75) if layer == "sliding" else (0.5, 0.62)
    c2 = lo + (hi - lo) * torch.rand((2, 1), generator=soft, device=dev)
    mix = (c2.sqrt() * uu, (1 - c2).sqrt())
    k_cache = torch.empty(shape, dtype=FP8, device=dev)
    v_cache = torch.empty(shape, dtype=FP8, device=dev)
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, 2, 64)
        for at in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - at)
            xr = torch.randn((n, 2, 64), generator=gen, device=dev)
            if keys:
                xr = mix[0] + mix[1] * xr
            flat[at:at + n] = xr.to(FP8)
    g0 = torch.Generator().manual_seed(_REF_STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=g0).to(dev)
    shared = torch.randn((_REF_STREAM_RANK, HIDDEN), generator=g0).to(dev)
    coef = torch.randn((tokens, _REF_STREAM_RANK), generator=gen, device=dev)
    scale = torch.ones((batch, q_len, 1), device=dev)
    if 1 < q_len <= 8:
        scale[:, 1:] = _REF_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (_REF_STREAM_MEAN * mean
                       + math.sqrt(_REF_STREAM_SHARED / _REF_STREAM_RANK) * coef @ shared)
              + (1 - _REF_STREAM_SHARED * scale * scale).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=dev))
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=dev)
    sched = None
    if layer == "full" and q_len == 1:
        try:
            from sgl_kernel.flash_attn import get_scheduler_metadata
            sched = get_scheduler_metadata(
                batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()),
                num_heads=16, num_heads_k=2, headdim=64, cache_seqlens=cache_seqlens,
                qkv_dtype=FP8, cu_seqlens_q=cu_seqlens_q, page_size=page, causal=True,
                num_splits=0)
        except Exception:
            sched = None
    attn_input = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS))
    return {
        "stream": stream, "attn_input": attn_input.to(torch.bfloat16), "positions": positions,
        "cache_loc": cache_loc, "cache_seqlens": cache_seqlens, "cu_seqlens_q": cu_seqlens_q,
        "scheduler_metadata": sched, "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(1.0, dtype=torch.float32, device=dev),
        "v_scale": torch.tensor(1.0, dtype=torch.float32, device=dev),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "local_start": torch.tensor([0], dtype=torch.int32, device=dev),
        "config": {"batch": batch, "q_len": q_len, "tokens": tokens, "hidden": HIDDEN,
                   "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64, "page_size": page,
                   "window": 2048 if layer == "sliding" else 0,
                   "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,
                   "qk_scale_factor": 5.4730064863875, "sm_scale": 64 ** -0.5,
                   "kv_scale": 1.0, "n_experts": N_EXPERTS, "top_k": TOP_K,
                   "local_experts": LOCAL_EXPERTS},
    }

def _ab_rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def _ab_rope(x, pos, theta):
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, device=x.device).float() / 64))
    ang = pos.float()[:, None] * inv
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)

def _ab_coeffs(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha, beta

def _ab_attention(data, x, weights, qdtype):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    qkv = x @ weights["qkv_gate"].float().T
    q, k, v, gate = qkv.split([1024, 128, 128, 1024], dim=-1)
    pos = (data["seq_lens"].long()[:, None] + torch.arange(length, device=x.device)).reshape(-1)
    q = _ab_rms(q.view(-1, 16, 64))
    k = _ab_rms(k.view(-1, 2, 64))
    if cfg["rope_theta"]:
        q, k = _ab_rope(q, pos, cfg["rope_theta"]), _ab_rope(k, pos, cfg["rope_theta"])
    if qdtype == torch.bfloat16:
        q = (q * cfg["qk_scale_factor"]).to(torch.bfloat16).float()
    else:
        q = q.to(qdtype).float() * cfg["qk_scale_factor"]
    q = q.view(batch, length, 16, 64)
    ks, vs = data["k_scale"], data["v_scale"]
    k_new = (k.float() / ks).to(FP8).view(batch, length, 2, 64)
    v_new = (v.view(-1, 2, 64).float() / vs).to(FP8).view(batch, length, 2, 64)
    k_flat = data["k_cache"].view(-1, 2, 64)
    v_flat = data["v_cache"].view(-1, 2, 64)
    window = cfg["window"]
    out = torch.empty((batch, length, 16, 64), device=x.device)
    steps = torch.arange(length, device=x.device)
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        first = max(0, seq - window + 1) if window else 0
        past = torch.arange(first, seq, device=x.device)
        size = cfg["page_size"]
        rows = data["page_table"][b].long()[past // size] * size + past % size
        keys = torch.cat([k_flat[rows].float() * ks, k_new[b].float() * ks])
        vals = torch.cat([v_flat[rows].float() * vs, v_new[b].float() * vs])
        key_pos = torch.cat([past, seq + steps])
        query_pos = seq + steps
        visible = key_pos[None, :] <= query_pos[:, None]
        if window:
            visible &= key_pos[None, :] > query_pos[:, None] - window
        att = torch.einsum("lkgd,tkd->lkgt", q[b].reshape(length, 2, 8, 64), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgt,tkd->lkgd", att, vals).reshape(length, 16, 64)
    out = _ab_rms(out).reshape(-1, 1024) * torch.sigmoid(gate)
    return out, k_new, v_new

def _ab_ref_front(data, weights, qdtype=torch.bfloat16):
    o, k_new, v_new = _ab_attention(data, data["attn_input"].float(), weights, qdtype)
    a = o @ weights["o"].float().T
    alpha1, beta1 = _ab_coeffs(weights["attn_residual_gate"])
    stream1 = alpha1 * data["stream"].float() + beta1 * _ab_rms(a, POST_NORM_EPS)
    h = _ab_rms(stream1) * (1.0 + weights["pre_ffn_norm"].float())
    logits = h.double() @ weights["router"].double().T
    scores = torch.sigmoid(logits * ROUTER_MULTIPLIER).float()
    return {"stream1": stream1, "h": h, "scores": scores,
            "biased": scores + weights["router_bias"].float(), "k_new": k_new, "v_new": v_new,
            "local_start": int(data["local_start"])}

def _ab_deq(weights, e):
    if e not in _AB_DEQ:
        def dq(packed, sf, gs):
            p = packed.view(torch.uint8)
            codes = torch.stack((p & 0x0F, (p >> 4) & 0x0F), dim=-1).flatten(-2)
            grid = torch.tensor(_E2M1, dtype=torch.float32, device=p.device)
            mag = grid[(codes & 0x7).long()]
            vals = torch.where((codes & 0x8) != 0, -mag, mag)
            blocks = vals.view(*vals.shape[:-1], vals.shape[-1] // 16, 16)
            return (blocks * (sf.float() * gs.float()).unsqueeze(-1)).view_as(vals)
        _AB_DEQ[e] = (dq(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]),
                      dq(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]))
    return _AB_DEQ[e]

def _ab_ref_back(front, weights, ids):
    picked = front["scores"].gather(1, ids.long())
    tw = picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS)
    p = front["h"] @ weights["pooler"].float().T
    p = _ab_rms(p) * (1.0 + weights["pre_expert_norm"].float())
    local = (ids.long() - front["local_start"]) % N_EXPERTS
    moe = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
    for e in torch.unique(local).tolist():
        if not 0 <= e < LOCAL_EXPERTS:
            continue
        tok, slot = (local == e).nonzero(as_tuple=True)
        w13, w2 = _ab_deq(weights, e)
        hh = p[tok] @ w13.T
        inter = torch.nn.functional.silu(hh[:, :D_EXPERT]) * hh[:, D_EXPERT:]
        y = (inter @ w2.T) * (1.0 + weights["post_expert_norm"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        moe.index_add_(0, tok, y * tw[tok, slot, None])
    if len(moe) <= SLICED_UNPOOL_MAX_TOKENS:
        f = torch.zeros_like(front["stream1"])
        f[:, :UNPOOL_SLICE] = moe @ weights["unpooler"][:UNPOOL_SLICE].float().T
    else:
        f = moe @ weights["unpooler"].float().T
    f = f * (1.0 + weights["post_ffn_norm"].float())
    alpha2, beta2 = _ab_coeffs(weights["ffn_residual_gate"])
    stream2 = alpha2 * front["stream1"] + beta2 * _ab_rms(f, POST_NORM_EPS)
    hidden = _ab_rms(stream2) * (1.0 + weights["next_pre_attn_norm"].float())
    return hidden.to(torch.bfloat16), stream2

def _ab_local(ids, start):
    local = ids[(ids - start) % N_EXPERTS < LOCAL_EXPERTS]
    return int(local.numel()), int(torch.unique(local).numel())

def _ab_check_routing(ids, biased, start):
    if ids.min() < 0 or ids.max() >= N_EXPERTS:
        return "expert id out of range"
    ranked = ids.long().sort(-1).values
    if (ranked[:, 1:] == ranked[:, :-1]).any():
        return "a token routed to the same expert twice"
    best = biased.topk(TOP_K, dim=-1)
    gap = best.values[:, -1:] - biased.gather(1, ids.long())
    if gap.max() > 5e-2:
        return f"pick {float(gap.max()):.2e} below the 8th best"
    ninth = biased.topk(TOP_K + 1, dim=-1).values[:, -1:]
    keep_tol = 0.03 if len(ids) <= 256 else 0.04
    kept = (best.indices[:, :, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices - start) % N_EXPERTS < LOCAL_EXPERTS
    if (local & ~kept & (best.values - ninth > keep_tol)).any():
        return "dropped a clear local expert"
    routes, distinct = _ab_local(ids.long(), start)
    rr, rd = _ab_local(best.indices, start)
    if routes < rr * 0.92 - 4:
        return f"{routes} local routes against {rr}"
    if distinct < rd * 0.85 - 2:
        return f"{distinct} local experts against {rd}"
    return None

def _ab_compare(out, ref, tol):
    if out.shape != ref.shape or out.dtype != ref.dtype:
        return False, f"shape/dtype {tuple(out.shape)} {out.dtype}"
    o, r = out.float(), ref.float()
    if not torch.isfinite(o).all():
        return False, "NaN or Inf"
    d = o - r
    err = float(d.norm() / r.norm().clamp(min=1e-12))
    rows_d = d.reshape(-1, r.shape[-1]).norm(dim=-1)
    rows_r = r.reshape(-1, r.shape[-1]).norm(dim=-1)
    worst = float((rows_d / torch.maximum(rows_r, (0.1 * rows_r.mean()).clamp(min=1e-12))).max())
    return err <= tol and worst <= 5 * tol, f"{err:.2e}/{worst:.2e}<={tol:.3f}"

def _ab_validate(base, out, arm_data, weights, front, front8):
    """The judge's check_implementation on one arm's outputs: (ok, message)."""
    hidden, stream, ids = out
    err = _ab_check_routing(ids, front["biased"], front["local_start"])
    if err:
        return False, f"routing: {err}"
    ref_h, ref_s = _ab_ref_back(front, weights, ids)
    e8_h, e8_s = _ab_ref_back(front8, weights, ids)
    a1, _ = _ab_coeffs(weights["attn_residual_gate"])
    a2, _ = _ab_coeffs(weights["ffn_residual_gate"])
    carried = a1 * a2 * base["stream"].float()

    def bound(e):
        return min(0.15, max(0.03, 2.5 * e))

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))

    ok1, m1 = _ab_compare(stream - carried, ref_s - carried, bound(rel(e8_s - carried, ref_s - carried)))
    ok2, m2 = _ab_compare(hidden, ref_h, bound(rel(e8_h, ref_h)))
    cfg = base["config"]
    pos = base["seq_lens"].long()[:, None] + torch.arange(cfg["q_len"], device=ids.device)
    slots = (torch.gather(base["page_table"].long(), 1, pos // cfg["page_size"])
             * cfg["page_size"] + pos % cfg["page_size"])
    ok3, m3 = True, ""
    for name, cache, want, sc in (("k", "k_cache", front["k_new"], base["k_scale"]),
                                  ("v", "v_cache", front["v_new"], base["v_scale"])):
        got = arm_data[cache].view(-1, 2, 64)[slots]
        ok, m = _ab_compare(got.float() * sc, want.float() * sc, 8e-2)
        ok3, m3 = ok3 and ok, m3 + f" {name} {m}"
    return ok1 and ok2 and ok3, f"update {m1} hidden {m2} cache{m3}"

def _qual_oracle(base, out, arm_data, weights, front, front8):
    """The judge's check_implementation on one arm's outputs: (ok, message)."""
    hidden, stream, ids = out
    err = _ab_check_routing(ids, front['biased'], front['local_start'])
    if err:
        return (False, f'routing: {err}')
    ref_h, ref_s = _ab_ref_back(front, weights, ids)
    e8_h, e8_s = _ab_ref_back(front8, weights, ids)
    a1, _ = _ab_coeffs(weights['attn_residual_gate'])
    a2, _ = _ab_coeffs(weights['ffn_residual_gate'])
    carried = a1 * a2 * base['stream'].float()

    def bound(e):
        assert math.isfinite(e), 'QUAL_NONFINITE_CALIBRATION'
        return min(0.15, max(0.03, 2.5 * e))

    def rel(a, b):
        value = float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))
        assert math.isfinite(value), 'QUAL_NONFINITE_RELATIVE_METRIC'
        return value
    ok1, m1 = _ab_compare(stream - carried, ref_s - carried, bound(rel(e8_s - carried, ref_s - carried)))
    ok2, m2 = _ab_compare(hidden, ref_h, bound(rel(e8_h, ref_h)))
    cfg = base['config']
    pos = base['seq_lens'].long()[:, None] + torch.arange(cfg['q_len'], device=ids.device)
    slots = torch.gather(base['page_table'].long(), 1, pos // cfg['page_size']) * cfg['page_size'] + pos % cfg['page_size']
    ok3, m3 = (True, '')
    for name, cache, want, sc in (('k', 'k_cache', front['k_new'], base['k_scale']), ('v', 'v_cache', front['v_new'], base['v_scale'])):
        got = arm_data[cache].view(-1, 2, 64)[slots]
        ok, m = _ab_compare(got.float() * sc, want.float() * sc, 0.08)
        ok3, m3 = (ok3 and ok, m3 + f' {name} {m}')
    return (ok1 and ok2 and ok3, f'update {m1} hidden {m2} cache{m3}')


_QUAL_NORMAL_PREPARE=prepare
_QUAL_BACK_ORIGINAL=_ab_ref_back

def _qual_budget(start):
    assert time.monotonic()-start<295,'QUAL_PREPARE_BUDGET'

def _qual_finite(value,label):
    assert bool(torch.isfinite(value.float()).all()),('QUAL_NONFINITE',label)
    assert bool(torch.isfinite(value.float().norm())),('QUAL_NONFINITE_NORM',label)

def _qual_ref_back(front,weights,ids):
    result=_QUAL_BACK_ORIGINAL(front,weights,ids)
    for i,value in enumerate(result):_qual_finite(value,'reference_back_'+str(i))
    return result

_ab_ref_back=_qual_ref_back

def _qual_storage(t):
    s=t.untyped_storage();return (str(t.device),s.data_ptr(),s.nbytes())

def _qual_overlap(a,b):
    return any(da==db and pa<pb+nb and pb<pa+na for da,pa,na in a for db,pb,nb in b if na and nb)

def _qual_owners(first,second,weights):
    def tensors(value,seen):
        if id(value) in seen:return []
        seen.add(id(value))
        if isinstance(value,torch.Tensor):return [value]
        if isinstance(value,dict):items=value.values()
        elif isinstance(value,(tuple,list)):items=value
        elif isinstance(value,torch.nn.Module) or type(value).__module__.startswith('sglang'):items=vars(value).values()
        else:return []
        return [t for item in items for t in tensors(item,seen)]
    readonly={_qual_storage(t) for t in weights.values()}
    groups=[[_qual_storage(t) for t in tensors(state,set()) if _qual_storage(t) not in readonly] for state in (first,second)]
    assert not _qual_overlap(*groups),'QUAL_OWNED_STATE_ALIAS'
    workspace=[]
    for state in (first,second):
        assert set(state['dlplt'])=={('unpooler',128),('unpooler',256),('qkv_gate',256)},'QUAL_LT_COVERAGE'
        current=[]
        for fn in state['dlplt'].values():
            names=fn.__code__.co_varnames[1:fn.__code__.co_argcount]
            defaults=dict(zip(names,fn.__defaults__))
            assert isinstance(defaults['ws'],torch.Tensor),'QUAL_LT_WORKSPACE'
            current.append(_qual_storage(defaults['ws']))
        workspace.append(current)
    assert not _qual_overlap(*workspace),'QUAL_LT_ALIAS'

def _qual_inputs(batch,layer,seed,weights):
    data=_ab_inputs(batch,8,layer,seed)
    refs=(_ab_ref_front(data,weights),_ab_ref_front(data,weights,FP8))
    counts=torch.bincount((refs[0]['biased'].topk(8,dim=-1).indices//32).flatten(),minlength=8)
    local=32*int(counts.argmax());assert int(counts.max())>0,'QUAL_EMPTY_REFERENCE_ROUTES'
    data['local_start'].fill_(local)
    for front in refs:front['local_start']=local
    return data,refs

def _qual_restore(dst,src):
    for name,value in src.items():
        if isinstance(value,torch.Tensor):
            assert dst[name].shape==value.shape and dst[name].dtype==value.dtype,'QUAL_INPUT_ABI'
            dst[name].copy_(value)
        else:assert dst[name]==value,'QUAL_INPUT_METADATA'

def _qual_poison(data,output=None):
    if output is not None:
        for value in output:
            value.fill_(float('nan') if value.is_floating_point() else -777)
    rows=data['cache_loc'].long().reshape(-1)
    assert torch.unique(rows).numel()==rows.numel(),'QUAL_CACHE_LOC_ALIAS'
    for name in ('k_cache','v_cache'):
        data[name].view(torch.uint8).reshape(-1,2,64)[rows]=127

def _qual_validate(base,data,output,refs,weights):
    for i,value in enumerate(output[:2]):_qual_finite(value,'output_'+str(i))
    for front in refs:
        for name,value in front.items():
            if isinstance(value,torch.Tensor):_qual_finite(value,'front_'+name)
    ok,message=_qual_oracle(base,output,data,weights,*refs)
    assert ok,('QUAL_MODEL_ROUTER_CACHE',message)
    routes,distinct=_ab_local(output[2].long(),int(base['local_start']))
    assert routes>0 and distinct>0,'QUAL_NATIVE_EXPERT_WORK'

def _qual_diagnostics(state):
    assert int(torch.count_nonzero(state['verify_diag']))==0,'QUAL_WD_DIAGNOSTIC'
    assert int(state['moe1']['scratch']['ctr'][8])==0,'QUAL_EXPERT_DIAGNOSTIC'

def _qual_resources(state):
    owned=_VERIFY_HYBRID_OWNED
    moe=list(owned._moe1_ext().qual_moe_resources())
    one=list(owned._one_ext().qual_one_resources())
    assert len(moe)==10 and len(one)==20,'QUAL_RESOURCE_ABI'
    rows=[('moe51',moe),('one8_mode3',one[:10]),('one8_mode51',one[10:])]
    for name,r in rows:
        # regs,local,static,dynamic,threads,maxthreads,blocks/SM,SMs,coop,resident
        assert all(isinstance(v,int) and v>=0 for v in r),'QUAL_RESOURCE_VALUES'
        assert r[1]==0,'QUAL_RESOURCE_LOCAL'
        assert r[4]>0 and r[4]<=r[5] and r[6]>=1 and r[7]>0 and r[8]==1,'QUAL_RESOURCE_ADMISSION'
        assert r[9]==r[6]*r[7],'QUAL_RESOURCE_RESIDENT'
        assert r[3]<=227*1024,'QUAL_RESOURCE_SHARED'
    assert moe[4]==576 and moe[9]>=moe[7],'QUAL_MOE_GRID'
    assert state['one']['coop_launch'],'QUAL_ONE_COOPERATIVE_REQUIRED'
    assert 0<state['one']['resident'][8,3]<=one[9] and 0<state['one']['resident'][8,51]<=one[19],'QUAL_ONE_RESIDENCY'
    return rows

def _qual_empty_check(out,state,m):
    assert out.dtype==torch.bfloat16 and out.shape==(m,D_EXPERT),'QUAL_EMPTY_ABI'
    assert int(torch.count_nonzero(out))==0,'QUAL_EMPTY_OUTPUT'
    _qual_diagnostics(state)

def _qual_empty_capture(state,m,start):
    _qual_budget(start);device=state['verify_diag'].device
    gen=torch.Generator(device=device).manual_seed(970001+m)
    x=torch.randn((m,D_EXPERT),generator=gen,device=device).to(torch.bfloat16)
    ids=torch.arange(32,40,dtype=torch.int32,device=device).expand(m,8).contiguous()
    weights=torch.full((m,8),.125,dtype=torch.float32,device=device)
    local=torch.zeros(1,dtype=torch.int32,device=device)
    def run():return _VERIFY_HYBRID_OWNED._moe1_run(x,weights,ids,local,state,10,False)
    out=run();torch.cuda.synchronize();_qual_empty_check(out,state,m)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run()
    graph._qual_inputs=(x,weights,ids,local)
    graph.replay();torch.cuda.synchronize();_qual_empty_check(out,state,m)
    _qual_budget(start)
    return graph,out

def _qual_prepare(weights,production,start):
    import copy
    _qual_budget(start)
    if 'owned' not in production:
        print('HYBRID_QUAL_DONE selected=0 fallback=unsupported_weights checks=0 captures=0 fresh=0',flush=True)
        _qual_budget(start);return production
    private_owned=_VERIFY_HYBRID_OWNED.prepare(weights)
    torch.cuda.synchronize();_qual_budget(start)
    _qual_owners(production['owned'],private_owned,weights)
    private={'G':production['G'],'owned':private_owned}
    resources=_qual_resources(private_owned)
    checks=0;captures=0;fresh_checks=0;covered=set();empty_checks=0;empty_captures=0;restored=0
    for batch,layer in ((16,'sliding'),(16,'full'),(32,'sliding'),(32,'full')):
        for seed in (938101,938123):
            _qual_budget(start)
            base,refs=_qual_inputs(batch,layer,seed,weights)
            fresh,fresh_refs=_qual_inputs(batch,layer,seed+101,weights)
            data=copy.deepcopy(base)
            assert _verify_override_bit(data)==(1<<((0 if batch==16 else 2)+(layer=='full'))),'QUAL_ACTUAL_DISPATCH'
            a=[_qual_storage(v) for v in data.values() if isinstance(v,torch.Tensor)]
            b=[_qual_storage(v) for v in base.values() if isinstance(v,torch.Tensor)]
            assert not _qual_overlap(a,b),'QUAL_INPUT_OWNER_ALIAS'
            empty_graph,empty_out=_qual_empty_capture(private_owned,batch*8,start)
            empty_checks+=2;empty_captures+=1
            warm=torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for i in range(2):
                    _qual_budget(start);_qual_restore(data,base);_qual_poison(data)
                    output=kernel_fn(data,private);torch.cuda.synchronize()
                    _qual_validate(base,data,output,refs,weights);_qual_diagnostics(private_owned);checks+=1
                    if i==0:restored+=1
            torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
            _qual_restore(data,base);_qual_poison(data)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=kernel_fn(data,private)
            graph._qual_inputs=data
            graph.replay();torch.cuda.synchronize()
            _qual_validate(base,data,output,refs,weights);_qual_diagnostics(private_owned);checks+=1;captures+=1
            for source,want in ((fresh,fresh_refs),(base,refs),(fresh,fresh_refs)):
                _qual_budget(start)
                empty_out.fill_(float('nan'));empty_graph.replay();torch.cuda.synchronize()
                _qual_empty_check(empty_out,private_owned,batch*8);empty_checks+=1
                for value in output:value.fill_(float('nan') if value.is_floating_point() else -777)
                # Restore after output poison, including any legitimate input/output alias.
                _qual_restore(data,source);_qual_poison(data)
                graph.replay();torch.cuda.synchronize()
                _qual_validate(source,data,output,want,weights);_qual_diagnostics(private_owned)
                checks+=1;fresh_checks+=1;restored+=1
            covered.add((batch,layer))
            del graph,output,data,base,fresh,refs,fresh_refs,source,want,empty_graph,empty_out,warm
            _qual_budget(start)
    assert len(covered)==4 and checks==48 and captures==8 and fresh_checks==24,'QUAL_INCOMPLETE'
    assert empty_checks==40 and empty_captures==8 and restored==32,'QUAL_EMPTY_RESTORE_COVERAGE'
    _qual_owners(production['owned'],private_owned,weights);_qual_diagnostics(private_owned)
    for name,vector in resources:print('HYBRID_QUAL_RESOURCE',name,vector,flush=True)
    _qual_budget(start)
    print('HYBRID_QUAL_DONE selected=4 fallback=0 checks=48 captures=8 fresh=24 empty_checks=40 empty_captures=8 restored=32',flush=True)
    _qual_budget(start)
    return production

def prepare(weights):
    start=time.monotonic()
    production=_QUAL_NORMAL_PREPARE(weights)
    _qual_budget(start)
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=False
        return _qual_prepare(weights,production,start)
    finally:
        _AB_DEQ.clear();torch.backends.cuda.matmul.allow_tf32=old_tf32
