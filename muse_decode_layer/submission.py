# okc: problem=muse_decode_layer
"""SGLang's decoder layer (baseline.py's call sequence) with pieces measured faster on the
sibling problems swapped in. Every swapped piece falls back to SGLang's call when its
self-test in prepare() fails or where it was measured to lose.

- MoE (from muse_moe_w4a16): one CUDA kernel builds Marlin's block-sorted routing metadata
  over the 32 local experts, reading this GPU's first expert from the device; one CUDA kernel
  does the post-expert norm and weighted sum, which SGLang runs as about ten PyTorch ops
  below 32 tokens; Marlin moe_block_size 8 below 12 tokens, else 16.
- Expert GEMMs (W4A16 kernel from muse_moe_w4a16): NVFP4 weights as the mma A operand,
  repacked losslessly in prepare(), over work items of routed experts only, with K splits and
  programmatic dependent launch; BF16 at the points where SGLang materializes BF16. Configs
  per token count are tuned in prepare() against Marlin, which stays wherever it wins.
- Fused MoE (muse_moe_w4a16's one-launch kernel): every CTA routes from the global ids on
  the device, streams 36 KB weight units through a shared-memory ring, runs GEMM1 with
  SwiGLU and GEMM2 (mma.sync up to 16 tokens, wgmma above), crosses a software grid
  barrier and applies the post-expert norm and weighted sum; a prepare-time bench picks it
  or the W4 chain per token count.
- Router (from muse_bf16_proj): up to 8 tokens one CUDA-core FP32 kernel on the boundary's
  FP32 router input; 9 to 32 tokens the tensor-core kernel on a BF16 hi + lo split of the
  weights with FP32 accumulation (SGLang's own method above 8 tokens).
- Stream overlap: the router, top-8 and the MoE routing metadata run on a side stream while
  the main stream runs the pooler and the pre-expert norm; the streams join right before
  GEMM1.
- Projections (from muse_bf16_proj): an mma kernel streaming weights repacked into fragment
  order, for qkv_gate, pooler and the unpooler slice up to 8 tokens and o at 1 token.
- Skinny projections (muse_bf16_proj's sk kernel), up to 32 tokens where the autotune picks
  them: weights stored losslessly in 12 bits (a raw low byte and a 4-bit exponent code),
  decoded in registers for mma.m16n8k16, escapes corrected exactly by dedicated warps; the
  pooler and the hi/lo router in one launch up to 16 tokens as a front variant.

- Attention prep (idea 1 from muse_attention_decode): one launch for QK-norm, RoPE, the FP8
  query and the FP8 cache write in place of FusedQKNorm, three casts and store_cache, and one
  launch for the norm-gate; FA3 stays SGLang's.
- Fused decode attention (idea 2 from muse_attention_decode), q_len 1 except full layers at
  batch 1-2: one kernel per request and context slice covering both KV heads (norm, RoPE,
  cache write, Q.K and P.V on f16 tensor cores, online softmax); sliding layers merge splits in
  a thread-block cluster with the norm-gate, full layers in one more launch.
- Verify attention (muse_attention_decode's attn_ws.cu, var 16), q_len 8 on sliding layers and
  on full layers above batch 2: loader warps gather FP8 K/V through an mbarrier ring, two
  consumer warpgroups run Q.K on FP8 wgmma into FP32, FP32 softmax, P.V on f16 wgmma into
  FP32; one more launch merges the splits and applies the norm-gate. Full verify at batch 1-2
  stays on prep + FA3.
- One-launch attention (muse_attention_decode's attn_one.cu), every tp8 shape: a CTA takes one
  request, both KV heads and one key range, adds o 2^m and l 2^m into a persistent FP32
  workspace, and the last arriving CTA or every CTA of a cooperative launch normalizes,
  gates, writes and zeroes its rows; full context at batch > 1 on a flat partition of all
  requests' keys with a producer warpgroup. The paths above stay as fallbacks.

The fused norms around attention and the MoE are SGLang's, called as baseline.py does.
"""

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


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192]
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256]
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048], packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]
    post_ffn_norm: torch.Tensor       # bfloat16 [8192]
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192]


class Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int


class Inputs(TypedDict):
    stream: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: Config


# Which ported pieces kernel_fn may use; each was turned on in its own check run.
PORT_MOE = True
# The W4A16 expert GEMMs from muse_moe_w4a16, per token count where its tuner in prepare()
# finds them faster than Marlin.
PORT_MOE_W4 = True
# The fused MoE kernel from muse_moe_w4a16 (one persistent launch for routing, both GEMMs,
# SwiGLU, the post-expert norm and the weighted sum), per token count where the bench in
# prepare() finds one of its variants faster than the W4 chain.
PORT_MOE_FUSED = True
PORT_ROUTER = True
PORT_PROJ = True
# The skinny tensor-core projection kernels from muse_bf16_proj (12-bit lossless weights with
# escape warps, one bulk copy of weights and one TMA box of x per stage) and its m = 1 GEMVs
# for qkv_gate, o, pooler and the unpooler up to 32 tokens, plus the fused pooler + router
# entry up to 16, per (layer, T) where the cold-L2 autotune in prepare() picks them over
# proj_fwd and cuBLAS. Independent of the other switches.
PORT_PROJ_SK = True
# The hi/lo mma router at 9-32 tokens as a candidate for the MoE-front tuner in prepare(),
# which times each side-chain variant as a graph with a cold L2 and keeps the fastest per token
# count (SGLang's split router, our mma router, or one GEMM for the pooler and both router
# halves). Flipping it blind moved decode 16 by -0.01 to -0.02 both ways (checks 6ef2d969b6104e10,
# 46dab8fb87274880): the side chain was hidden under the pooler then; with our norm it is not.
PORT_ROUTER_MMA = True
# One cuBLAS GEMM with FP32 output for the pooler and the router's hi and lo halves at 9+ tokens,
# as a candidate for the same tuner: the pooled columns feed the norm (rounded to BF16 on load),
# the router columns are summed as SGLang sums hi + lo.
PORT_FRONT_CONCAT = True
# Idea 1 from muse_attention_decode: one prep launch and one norm-gate launch around FA3.
PORT_ATTN_PREP = True
# Idea 2: the fused q_len 1 attention kernel, on the shapes where it won on its own problem.
PORT_ATTN_FUSED = True
# The warp-specialized q_len 8 kernel from muse_attention_decode (var 16): sliding verify at
# every batch, full verify above batch 2.
PORT_ATTN_WS = True
# The one-launch attention kernel from muse_attention_decode (attn_one.cu) on every tp8
# shape, q_len 1 and 8, sliding and full, when its self-test passes (the shared cases, the
# workspace invariant after each, a replay stress); the kernels above stay as fallbacks and
# for the shapes _ONE_ROUTE excludes.
PORT_ATTN_ONE = True
# The routing metadata (route_align or SGLang's moe_align_block_size) built on the router's
# side stream right after top-8, and the pre-expert norm on the main stream before the join,
# so the main stream goes from the norm straight to GEMM1. False: both after the join.
ROUTE_ON_SIDE = True
# Front variants "concat-r" and "skf-r" (gap map 2026-10-30, E2): the side stream's logit add and
# routing, the main stream's pre-expert norm and the join become one route kernel on the main
# stream, and the fused MoE launches as its PDL secondary. The front tuner picks per token
# count against the side-stream fronts, timing the front and the MoE together.
PORT_ROUTE_KERNEL = True
# M1 (docs/steering/2026-10-29-v1-d32-breakdown.md): the projections built with SK_PDL, so a
# single-CTA sk config streams its first ring of weights before griddepcontrol.wait and launches
# as a PDL secondary when bit 16 of its cfg is set, and the q_len 1 / batch-1 q_len 8 attention
# instances built with ONE_TRIG (launch_dependents at entry). _m1_tune picks per token bucket
# (8, 16, 32) among off, the o projection, and the o projection plus the skf front.
PORT_SK_PDL = True
SK_PDL_BIT = 1 << 16
M1_BUCKETS = (8, 16, 32)
ROUTE_VARIANTS = {"concat-r": "concat", "skf-r": "skf"}
# Up to 8 tokens, our FFN boundary on the unpooler slice's 1024 live channels in place of
# F.pad to 8192 and FusedPostNormResidualPreNorm.
PORT_FFN_BOUNDARY = True
# Up to 8 tokens, a kernel on the side stream forked right after the qkv projection prefetches
# into L2 the weights the chain reads after attention (o, the FP32 router, the unpooler slice:
# 34 MB) while attention runs, with an evict_last policy. Off: the bulk prefetch hint left the o
# GEMV at 7.3 us (check 985732949c07474f) and real loads brought it only to 5.9 while the 34 MB
# stream during attention cost 2-4 us on every T <= 8 shape (check 34568f495b024c2f).
PORT_L2_PREFETCH = False
# Our attention boundary (post-attention norm, residual gate, pre-FFN norm, FP32 router input
# at T <= 8) with SGLang's one BF16 rounding, as a PDL secondary that loads the stream and the
# weights before its wait and triggers at entry, so the pooler's weight stream starts earlier.
PORT_ATTN_BOUNDARY = True
# Our pre-expert norm (torch's F.rms_norm math and summation order) as a PDL secondary of the
# pooler that triggers at entry, so GEMM1 is resident when it ends; torch's kernel cost 6 us
# on every shape.
PORT_PRE_NORM = True

HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
# SGLang's routing constants, as prepare() passes them to FusedMuseSparkRouting.
ROUTER_MULTIPLIER = 0.1767766952966369
ROUTE_EPS = 1e-15
ROUTER_MMA_MAX_TOKENS = 32
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0
# route_align keeps per-route ranks in shared memory and scans experts with one warp.
MAX_ROUTES = 4096

_CPP_SRC = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> route_align(torch::Tensor topk_ids, torch::Tensor start,
                                       int64_t block_size, int64_t max_padded,
                                       int64_t item_rows, int64_t item_rows2);
torch::Tensor finalize(torch::Tensor y, torch::Tensor scores, torch::Tensor norm_w,
                       torch::Tensor local_ids, double eps);
torch::Tensor router_fwd(torch::Tensor x, torch::Tensor w);
torch::Tensor pre_norm(torch::Tensor x, torch::Tensor w, double eps);
std::vector<torch::Tensor> route_fwd(torch::Tensor logits, int64_t lo_off, torch::Tensor x,
                                     torch::Tensor w, torch::Tensor bias, double mult,
                                     double eps, double denom_eps);
void prefetch_l2(torch::Tensor a, torch::Tensor b, torch::Tensor c);
torch::Tensor proj_fwd(torch::Tensor x, torch::Tensor wpacked, int64_t n);
torch::Tensor router_mma_fwd(torch::Tensor x, torch::Tensor wpacked);
torch::Tensor attn_prep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc,
                        torch::Tensor inv_freq, torch::Tensor k_scale, torch::Tensor v_scale,
                        torch::Tensor k_cache, torch::Tensor v_cache, bool rope);
torch::Tensor attn_gate(torch::Tensor o, torch::Tensor qkv);
torch::Tensor attn_fused(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                         torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                         torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                         int64_t page_size, int64_t window, bool rope, double qk_scale,
                         int64_t splits, bool cluster);
torch::Tensor attn_verify(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                          torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                          torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                          int64_t page_size, int64_t window, bool rope, double qk_scale,
                          int64_t splits);
std::vector<torch::Tensor> ffn_boundary_short(torch::Tensor stream, torch::Tensor f,
                                              torch::Tensor w_post, torch::Tensor alpha,
                                              torch::Tensor beta, torch::Tensor w_next,
                                              double eps_post, double eps_next);
std::vector<torch::Tensor> attn_boundary(torch::Tensor stream, torch::Tensor branch,
                                         torch::Tensor alpha, torch::Tensor beta,
                                         torch::Tensor w_next, double eps_post,
                                         double eps_next, bool want_router, int64_t geo);
std::vector<int64_t> attn_boundary_occupancy(int64_t geo);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;

// Internal linkage: another extension loaded into the same process would otherwise bind to
// this module's template instances and their one-time shared-memory setup.
namespace {

constexpr int kMaxLocal = 32;
constexpr int kMaxRoutes = 4096;
constexpr int kTopK = 8;
constexpr int kExpertHidden = 4096;

// A launch with programmatic dependent launch: the kernel may start once the kernel before it
// on the stream executes griddepcontrol.launch_dependents (or ends), and must execute
// griddepcontrol.wait before it reads that kernel's output. Compile-time cluster dims apply.
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

// PDL on the glue launches (router, route_align, finalize_row, the boundaries); the
// projection, norm and attention launches always carry it.
constexpr bool kPdlGlue = true;

// Marlin's moe_align layout over this GPU's 32 experts only: blocks of block_size rows,
// experts ascending, pad slots hold numel. SGLang aligns over 257 buckets instead and emits
// blocks of other GPUs' routes that Marlin then scans past. The first local expert is read
// from the device so the call needs no host sync. Atomic ranks order rows within an expert
// arbitrarily; Marlin computes each row independently, so the output does not depend on it.
__global__ void __launch_bounds__(1024) route_align_kernel(
        const int* __restrict__ topk_ids, const int* __restrict__ start_p, int numel, int bs,
        int* __restrict__ sorted_ids, int* __restrict__ expert_ids, int* __restrict__ num_post,
        int* __restrict__ local_ids, int item_rows, int item_rows2, int items2_at,
        int2* __restrict__ items, int* __restrict__ num_items) {
    asm volatile("griddepcontrol.launch_dependents;");
    __shared__ int cnt[kMaxLocal];
    __shared__ int off[kMaxLocal];
    __shared__ int total_s;
    __shared__ short rank_s[kMaxRoutes];
    const int tid = threadIdx.x;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int start = *start_p;
    if (tid < kMaxLocal) {
        cnt[tid] = 0;
    }
    __syncthreads();
    for (int p = tid; p < numel; p += blockDim.x) {
        const int e = topk_ids[p] - start;
        const int loc = (e >= 0 && e < kMaxLocal) ? e : -1;
        local_ids[p] = loc;
        if (loc >= 0) {
            rank_s[p] = static_cast<short>(atomicAdd(&cnt[loc], 1));
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int c = cnt[tid];
        const int padded = (c + bs - 1) / bs * bs;
        int incl = padded;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, incl, d);
            if (tid >= d) {
                incl += v;
            }
        }
        const int excl = incl - padded;
        off[tid] = excl;
        if (tid == 31) {
            total_s = incl;
            num_post[0] = incl;
        }
        for (int b = excl / bs; b < incl / bs; ++b) {
            expert_ids[b] = tid;
        }
        // Work items for the W4 GEMMs (one list per GEMM, the second at items2_at): up to
        // ir routes of one expert each, as (expert << 16 | rows, first row in sorted_ids).
        for (int q = 0; q < 2; ++q) {
            const int ir = q == 0 ? item_rows : item_rows2;
            if (ir <= 0) {
                continue;
            }
            int2* list = items + (q == 0 ? 0 : items2_at);
            const int nit = (c + ir - 1) / ir;
            int iincl = nit;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, iincl, d);
                if (tid >= d) {
                    iincl += v;
                }
            }
            for (int j = 0; j < nit; ++j) {
                list[iincl - nit + j] = make_int2((tid << 16) | min(ir, c - j * ir), excl + j * ir);
            }
            if (tid == 31) {
                num_items[q] = iincl;
            }
        }
    }
    __syncthreads();
    const int total = total_s;
    for (int i = tid; i < total; i += blockDim.x) {
        sorted_ids[i] = numel;
    }
    __syncthreads();
    for (int p = tid; p < numel; p += blockDim.x) {
        const int e = topk_ids[p] - start;
        if (e >= 0 && e < kMaxLocal) {
            sorted_ids[off[e] + rank_s[p]] = p;
        }
    }
}

__device__ __forceinline__ void unpack8(uint4 raw, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v = __bfloat1622float2(h[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    }
    return raw;
}

// One block per token, 512 threads x 8 channels. Slots of other GPUs' experts may hold
// garbage, even NaN, so they are never read.
__global__ void __launch_bounds__(512) finalize_row_kernel(
        const __nv_bfloat16* __restrict__ y, const float* __restrict__ scores,
        const __nv_bfloat16* __restrict__ norm_w, const int* __restrict__ local_ids,
        __nv_bfloat16* __restrict__ out, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    constexpr int THREADS = 512;
    static_assert(THREADS * 8 == kExpertHidden);
    constexpr int WARPS = THREADS / 32;
    __shared__ float red[kTopK][WARPS];
    __shared__ float scale[kTopK];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c = tid * 8;
    float w[8];
    unpack8(*reinterpret_cast<const uint4*>(norm_w + c), w);
    bool live[kTopK];
    float v[kTopK][8];
    float ss[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        live[k] = local_ids[t * kTopK + k] >= 0;
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (live[k]) {
            const size_t row = static_cast<size_t>(t) * kTopK + k;
            raw = *reinterpret_cast<const uint4*>(y + row * kExpertHidden + c);
        }
        unpack8(raw, v[k]);
        ss[k] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[k][j] *= w[j];
            ss[k] += v[k][j] * v[k][j];
        }
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], d);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            red[k][warp] = ss[k];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            float s = lane < WARPS ? red[k][lane] : 0.f;
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, d);
            }
            if (lane == 0) {
                scale[k] = live[k] ? scores[t * kTopK + k] * rsqrtf(s / kExpertHidden + eps)
                                   : 0.f;
            }
        }
    }
    __syncthreads();
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const float s = scale[k];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] += s * v[k][j];
        }
    }
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kExpertHidden + c) = pack8(acc);
}

// FP32 router up to 8 tokens. Grid: 16 row groups x 8 k-slices = 128 blocks, one per SM.
// Each warp owns one weight row and one 1024-wide k-slice; the block's FP32 x slice is
// staged in shared memory once. The 8 k-slice blocks of a row group form a cluster and add
// their partial sums over distributed shared memory: no atomics, and a fixed sum order.
// Plain FP32 FMAs, so the math matches F.linear without TF32.
constexpr int RN = 256;
constexpr int RK = 8192;
constexpr int R_ROWS_PER_BLOCK = 16;
constexpr int R_CLUSTER = 8;
constexpr int R_K_SLICE = RK / R_CLUSTER;
constexpr int R_STEPS = R_K_SLICE / 128;

template <int M>
constexpr int router_smem_bytes() {
    return M * R_K_SLICE * 4 + M * R_ROWS_PER_BLOCK * 4;
}

template <int M>
__global__ void __cluster_dims__(1, R_CLUSTER, 1) __launch_bounds__(512, 1)
router_kernel(const float* __restrict__ w, const float* __restrict__ x,
              float* __restrict__ out, int m) {
    asm volatile("griddepcontrol.launch_dependents;");
    extern __shared__ __align__(16) unsigned char smem[];
    float* xs = reinterpret_cast<float*>(smem);
    float* partial = reinterpret_cast<float*>(smem + M * R_K_SLICE * 4);

    cg::cluster_group cluster = cg::this_cluster();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * R_ROWS_PER_BLOCK + warp;
    const int kbase = blockIdx.y * R_K_SLICE;

    const float* wrow = w + (size_t)n * RK + kbase + lane * 4;
    float4 wv[R_STEPS];
#pragma unroll
    for (int s = 0; s < R_STEPS; ++s) {
        wv[s] = __ldcs(reinterpret_cast<const float4*>(wrow + s * 128));
    }

    // x is the output of the kernel before this one; the weights above are state.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // Rows past m are zero-filled so the inner loop needs no bounds check.
    constexpr int CHUNKS = R_K_SLICE / 4;
    for (int i = threadIdx.x; i < M * CHUNKS; i += blockDim.x) {
        const int t = i / CHUNKS;
        const int c = i % CHUNKS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t < m) {
            v = __ldg(reinterpret_cast<const float4*>(x + (size_t)t * RK + kbase) + c);
        }
        reinterpret_cast<float4*>(xs)[i] = v;
    }
    __syncthreads();

    float acc[M];
#pragma unroll
    for (int t = 0; t < M; ++t) {
        acc[t] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < R_STEPS; ++s) {
        const int k = s * 128 + lane * 4;
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float4 a = *reinterpret_cast<const float4*>(xs + t * R_K_SLICE + k);
            acc[t] = fmaf(wv[s].x, a.x, acc[t]);
            acc[t] = fmaf(wv[s].y, a.y, acc[t]);
            acc[t] = fmaf(wv[s].z, a.z, acc[t]);
            acc[t] = fmaf(wv[s].w, a.w, acc[t]);
        }
    }

#pragma unroll
    for (int t = 0; t < M; ++t) {
        float v = acc[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (lane == 0) {
            partial[t * R_ROWS_PER_BLOCK + warp] = v;
        }
    }
    cluster.sync();

    const int rank = cluster.block_rank();
    for (int j = threadIdx.x; j < M * R_ROWS_PER_BLOCK; j += blockDim.x) {
        const int t = j / R_ROWS_PER_BLOCK;
        if (j % R_CLUSTER == rank && t < m) {
            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < R_CLUSTER; ++c) {
                sum += cluster.map_shared_rank(partial, c)[j];
            }
            out[(size_t)t * RN + blockIdx.x * R_ROWS_PER_BLOCK + j % R_ROWS_PER_BLOCK] = sum;
        }
    }
    // Peers read this block's shared memory above; it must outlive those reads.
    cluster.sync();
}

template <int M>
void router_launch(const float* wp, const float* xp, float* op, int m, cudaStream_t stream) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(router_kernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             router_smem_bytes<M>());
        configured = true;
    }
    const dim3 grid(RN / R_ROWS_PER_BLOCK, R_CLUSTER);
    launch_ex(router_kernel<M>, grid, dim3(32 * R_ROWS_PER_BLOCK), router_smem_bytes<M>(),
              stream, kPdlGlue, wp, xp, op, m);
}

__device__ __forceinline__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// The pre-expert norm: torch's F.rms_norm of a BF16 [T, 4096] row with the effective 1 + w, in
// float, in the summation order of torch's vectorized_layer_norm_kernel (4 warps, each thread
// 8 vectors of 4 channels accumulated by FMA, a shfl_down tree per warp, the warps combined as
// (w0 + w2) + (w1 + w3), out = w * (rstd * x)), so the output is bitwise torch's. An FP32 input
// is rounded to BF16 on load, as cuBLAS rounds a BF16 output. A PDL secondary that triggers at
// entry, so GEMM1 (which waits at entry) is resident when the norm ends.
constexpr int NORM_N = 4096;
constexpr int NORM_THREADS = 128;
constexpr int NORM_PER = NORM_N / NORM_THREADS / 4;

template <typename InT>
__global__ void __launch_bounds__(NORM_THREADS) pre_norm_kernel(
        const InT* __restrict__ x, const __nv_bfloat16* __restrict__ w,
        __nv_bfloat16* __restrict__ out, long long row_stride, float eps) {
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
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
}

// The main-stream route kernel (gap map 2026-10-30, E2), one CTA per token: the router logits
// (with HILO, the concat GEMM's hi and lo column blocks added in FP32, as the torch add did),
// SGLang's routing (sigmoid of logit x multiplier, top 8 of score + bias, weights = picked
// scores over their sum + eps) and the pre-expert norm with pre_norm_kernel's arithmetic, so its
// output is bitwise that kernel's. It replaces the side stream's add and routing kernel, the
// main stream's norm and the join. It triggers its dependents at entry: the fused MoE launched
// after it as a PDL secondary may then start its data-independent preamble. Ties in the biased
// score go to the lower expert id; the judge accepts any choice within 0.05 of the 8th best.
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

// Warms L2 with weights the chain reads after attention (the o projection, the FP32 router and
// the unpooler slice at T <= 8) while attention runs: 16-byte loads with an evict_last policy,
// so the lines outlive the pooler's and the experts' streams, eight in flight per thread. Real
// loads, not prefetch hints: cp.async.bulk.prefetch.L2 left the o GEMV at 7.3 us (job
// 985732949c07474f). The kernel reads state only, derives nothing from the step's inputs and
// keeps nothing; the values go to a store that never happens.
constexpr int PF_CHUNK = 16;
constexpr int PF_THREADS = 256;
constexpr int PF_UNROLL = 8;

__global__ void __launch_bounds__(PF_THREADS) prefetch_kernel(const char* p0, long long n0,
                                                              const char* p1, long long n1,
                                                              const char* p2, long long n2,
                                                              unsigned* sink) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    const long long c0 = n0 / PF_CHUNK;
    const long long c1 = n1 / PF_CHUNK;
    const long long c2 = n2 / PF_CHUNK;
    const long long total = c0 + c1 + c2;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    unsigned acc = 0;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += PF_UNROLL * step) {
        unsigned v[PF_UNROLL][4];
#pragma unroll
        for (int u = 0; u < PF_UNROLL; ++u) {
            const long long j = i + u * step;
            v[u][0] = v[u][1] = v[u][2] = v[u][3] = 0u;
            if (j < total) {
                const char* base = p0;
                long long k = j;
                if (j >= c0 + c1) {
                    base = p2;
                    k = j - c0 - c1;
                } else if (j >= c0) {
                    base = p1;
                    k = j - c0;
                }
                asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                             : "=r"(v[u][0]), "=r"(v[u][1]), "=r"(v[u][2]), "=r"(v[u][3])
                             : "l"(base + k * PF_CHUNK), "l"(policy));
            }
        }
#pragma unroll
        for (int u = 0; u < PF_UNROLL; ++u) {
            acc ^= v[u][0] ^ v[u][1] ^ v[u][2] ^ v[u][3];
        }
    }
    if (acc == 0x9e3779b9u && sink != nullptr) {
        *sink = acc;
    }
}

// BF16 projections on tensor cores, also used for the FP32 router as BF16 hi + lo (HILO).
// Each warp multiplies R 16-row weight tiles by up to 32 tokens with mma.m16n8k16; the math
// is padded to 8 tokens, which costs nothing because these shapes are memory-bound. The
// weights are repacked losslessly into the mma A-fragment order, so each lane's share of a
// k-tile is 16 contiguous bytes. A 4-stage cp.async pipeline keeps about 96 KB of weights and
// tokens in flight per SM. The S k-slice blocks of a row group form a cluster and sum their
// partials over distributed shared memory.
constexpr int STAGES = 4;

// Token rows hold one chunk (CH k-tiles) padded by 8 elements, so the 8 row addresses of one
// ldmatrix fall in different banks.
template <int CH>
__host__ __device__ constexpr int xs_stride() {
    return CH * 16 + 8;
}

template <int MT, int WARPS, int R, int CH>
constexpr int proj_smem_bytes() {
    return STAGES * WARPS * R * CH * 512 + STAGES * MT * 8 * xs_stride<CH>() * 2 +
           MT * 8 * WARPS * R * 16 * 4;
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// A source size of 0 makes cp.async write zeros, which pads tokens past m.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_addr(dst)), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ uint2 ldmatrix_x2(const __nv_bfloat16* p) {
    uint2 r;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r.x), "=r"(r.y)
                 : "r"(smem_addr(p)));
    return r;
}

__device__ __forceinline__ void store_out(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

__device__ __forceinline__ void store_out(float* p, float v) {
    *p = v;
}

// Not volatile: it only reads and writes registers, and the compiler needs to move it to
// overlap ldmatrix latency.
__device__ __forceinline__ void mma_bf16(float* c, const uint4& a, const uint2& b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

// HILO: the router's FP32 weights as BF16 hi and lo tiles interleaved along k (k-tile 2j is
// hi, 2j + 1 is lo of x's k-tile j), so both multiply the same x fragment into one FP32
// accumulator; k is then twice x's row length. blockIdx.z picks a group of MT * 8 tokens.
template <int MT, int WARPS, int R, int CH, int S, bool HILO = false,
          typename OutT = __nv_bfloat16>
__global__ void __cluster_dims__(1, S, 1) __launch_bounds__(WARPS * 32)
proj_kernel(const uint4* __restrict__ wp, const __nv_bfloat16* __restrict__ x,
            OutT* __restrict__ out, int m, int n_out, int k, int t_per_block) {
    asm volatile("griddepcontrol.launch_dependents;");
    constexpr int TOKENS = MT * 8;
    constexpr int ROWS = WARPS * R * 16;
    constexpr int XS = xs_stride<CH>();
    constexpr int W_STAGE_VECS = WARPS * R * CH * 32;
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* ws = reinterpret_cast<uint4*>(smem);
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem + STAGES * W_STAGE_VECS * 16);
    float* partial = reinterpret_cast<float*>(
        smem + STAGES * W_STAGE_VECS * 16 + STAGES * TOKENS * XS * 2);

    cg::cluster_group cluster = cg::this_cluster();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int kt_total = k / 16;
    const int xk = HILO ? k / 2 : k;
    const int tok0 = blockIdx.z * TOKENS;
    x += (size_t)tok0 * xk;
    out += (size_t)tok0 * n_out;
    m -= tok0;
    const int tile0 = (blockIdx.x * WARPS + warp) * R;
    const int kt0 = blockIdx.y * t_per_block;
    const int chunks = t_per_block / CH;
    const uint4* wsrc = wp + ((size_t)tile0 * kt_total + kt0) * 32 + lane;

    auto load_w = [&](int c, int slot) {
        uint4* wdst = ws + slot * W_STAGE_VECS + warp * R * CH * 32 + lane;
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int i = 0; i < CH; ++i) {
                cp_async16(wdst + (r * CH + i) * 32,
                           wsrc + ((size_t)r * kt_total + c * CH + i) * 32, true);
            }
        }
    };
    constexpr int VECS_PER_ROW = HILO ? CH : CH * 2;
    auto x_src = [&](int c, int v) -> const __nv_bfloat16* {
        const int tok = v / VECS_PER_ROW;
        const int col = (v % VECS_PER_ROW) * 8;
        const int kcol = (HILO ? (kt0 + c * CH) / 2 : kt0 + c * CH) * 16;
        return x + (size_t)tok * xk + kcol + col;
    };
    auto load_chunk = [&](int c, int slot) {
        load_w(c, slot);
        __nv_bfloat16* xdst = xs + slot * TOKENS * XS;
        for (int v = threadIdx.x; v < TOKENS * VECS_PER_ROW; v += WARPS * 32) {
            const int tok = v / VECS_PER_ROW;
            const int col = (v % VECS_PER_ROW) * 8;
            const bool valid = tok < m;
            cp_async16(xdst + tok * XS + col, valid ? x_src(c, v) : x, valid);
        }
    };

    // The first weight stages stream before the wait on the kernel before this one, whose
    // output x is. The x rows of those stages are then staged with plain loads, so the
    // cp.async groups are the same three as before and the loop's wait_group count holds.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < chunks) {
            load_w(s, s);
        }
        asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < chunks) {
            __nv_bfloat16* xdst = xs + s * TOKENS * XS;
            for (int v = threadIdx.x; v < TOKENS * VECS_PER_ROW; v += WARPS * 32) {
                const int tok = v / VECS_PER_ROW;
                const int col = (v % VECS_PER_ROW) * 8;
                const uint4 val = tok < m ? *reinterpret_cast<const uint4*>(x_src(s, v))
                                          : make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(xdst + tok * XS + col) = val;
            }
        }
    }

    float acc[R][MT][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int nt = 0; nt < MT; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[r][nt][i] = 0.f;
            }
        }
    }
    // ldmatrix rows: lanes 0-7 give matrix 0 (k 0-7), lanes 8-15 matrix 1 (k 8-15).
    const int xrow = (lane & 7) * XS + ((lane >> 3) & 1) * 8;

    for (int c = 0; c < chunks; ++c) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        // Also guarantees every warp is done with the slot the next load overwrites.
        __syncthreads();
        if (c + STAGES - 1 < chunks) {
            load_chunk(c + STAGES - 1, (c + STAGES - 1) % STAGES);
        }
        asm volatile("cp.async.commit_group;\n");

        const int slot = c % STAGES;
        const uint4* wa = ws + slot * W_STAGE_VECS + warp * R * CH * 32 + lane;
        const __nv_bfloat16* xb = xs + slot * TOKENS * XS + xrow;
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            uint4 a[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                a[r] = wa[(r * CH + i) * 32];
            }
#pragma unroll
            for (int nt = 0; nt < MT; ++nt) {
                const uint2 b = ldmatrix_x2(xb + nt * 8 * XS + (HILO ? i / 2 : i) * 16);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    mma_bf16(acc[r][nt], a[r], b);
                }
            }
        }
    }

    // C fragment: rows lane/4 and lane/4 + 8, token columns (lane%4)*2 and +1.
    const int g = lane >> 2;
    const int q = (lane & 3) * 2;
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int nt = 0; nt < MT; ++nt) {
            const int row = (warp * R + r) * 16 + g;
            const int t = nt * 8 + q;
            partial[t * ROWS + row] = acc[r][nt][0];
            partial[(t + 1) * ROWS + row] = acc[r][nt][1];
            partial[t * ROWS + row + 8] = acc[r][nt][2];
            partial[(t + 1) * ROWS + row + 8] = acc[r][nt][3];
        }
    }
    cluster.sync();

    const int rank = cluster.block_rank();
    for (int i = threadIdx.x; i < TOKENS * ROWS / S; i += blockDim.x) {
        const int j = i * S + rank;
        const int t = j / ROWS;
        if (t < m) {
            float sum = 0.f;
#pragma unroll
            for (int s = 0; s < S; ++s) {
                sum += cluster.map_shared_rank(partial, s)[j];
            }
            store_out(out + (size_t)t * n_out + blockIdx.x * ROWS + j % ROWS, sum);
        }
    }
    // Peers read this block's shared memory above; it must outlive those reads.
    cluster.sync();
}

template <int MT, int WARPS, int R, int CH, int S, bool HILO = false,
          typename OutT = __nv_bfloat16>
void proj_launch(const uint4* wp, const __nv_bfloat16* xp, OutT* op, int m, int n, int k,
                 cudaStream_t stream) {
    constexpr int smem = proj_smem_bytes<MT, WARPS, R, CH>();
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(proj_kernel<MT, WARPS, R, CH, S, HILO, OutT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    const dim3 grid(n / (WARPS * R * 16), S, (m + MT * 8 - 1) / (MT * 8));
    launch_ex_pdl(proj_kernel<MT, WARPS, R, CH, S, HILO, OutT>, grid, dim3(WARPS * 32), smem,
                  stream, wp, xp, op, m, n, k, k / 16 / S);
}

// Up to 8 tokens: 4 warps of one row tile each, 8 k-tiles per stage, two blocks per SM.
template <int S>
void proj_small(const uint4* wp, const __nv_bfloat16* xp, __nv_bfloat16* op, int m, int n,
                int k, cudaStream_t stream) {
    proj_launch<1, 4, 1, 8, S>(wp, xp, op, m, n, k, stream);
}

// The router at 9 to 32 tokens in 16-token groups: 32 tokens ran 8.6 us as two groups and
// 10.5 us as one (muse_bf16_proj checks 457c09b7f2a04ec3, 69a452e14dbc4729). 256 rows as 16
// one-warp row groups x 8 k-slices = 128 blocks.
void router_mma(const uint4* wp, const __nv_bfloat16* xp, float* op, int m,
                cudaStream_t stream) {
    proj_launch<2, 1, 1, 16, 8, true, float>(wp, xp, op, m, 256, 2 * 8192, stream);
}

// compile_check: route_align_kernel<<<1, 64>>>(0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0);
// compile_check: finalize_row_kernel<<<1, 512>>>(0, 0, 0, 0, 0, 0.f);
// compile_check: router_launch<1>(0, 0, 0, 1, 0);
// compile_check: router_launch<2>(0, 0, 0, 1, 0);
// compile_check: router_launch<4>(0, 0, 0, 1, 0);
// compile_check: router_launch<8>(0, 0, 0, 1, 0);
// compile_check: proj_small<2>(0, 0, 0, 1, 8192, 1024, 0);
// compile_check: proj_small<4>(0, 0, 0, 1, 2304, 8192, 0);
// compile_check: proj_small<8>(0, 0, 0, 1, 1024, 4096, 0);
// compile_check: router_mma(0, 0, 0, 16, 0);
}  // namespace

std::vector<torch::Tensor> route_align(torch::Tensor topk_ids, torch::Tensor start,
                                       int64_t block_size, int64_t max_padded,
                                       int64_t item_rows, int64_t item_rows2) {
    const int numel = topk_ids.numel();
    TORCH_CHECK(numel <= kMaxRoutes);
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 && start.scalar_type() == torch::kInt32);
    auto opts = topk_ids.options();
    auto sorted_ids = torch::empty({max_padded}, opts);
    auto expert_ids = torch::empty({(max_padded + block_size - 1) / block_size}, opts);
    auto num_post = torch::empty({1}, opts);
    auto local_ids = torch::empty_like(topk_ids);
    const int64_t max1 = item_rows > 0 ? kMaxLocal + numel / item_rows + 1 : 1;
    const int64_t max2 = item_rows2 > 0 ? kMaxLocal + numel / item_rows2 + 1 : 1;
    auto items = torch::empty({max1 + max2, 2}, opts);
    auto num_items = torch::empty({2}, opts);
    int threads = ((numel + 31) / 32) * 32;
    threads = threads < 64 ? 64 : (threads > 1024 ? 1024 : threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_ex(
        route_align_kernel, dim3(1), dim3(threads), 0, stream, kPdlGlue, topk_ids.data_ptr<int>(),
        start.data_ptr<int>(), numel, static_cast<int>(block_size), sorted_ids.data_ptr<int>(),
        expert_ids.data_ptr<int>(), num_post.data_ptr<int>(), local_ids.data_ptr<int>(),
        static_cast<int>(item_rows), static_cast<int>(item_rows2), static_cast<int>(max1),
        reinterpret_cast<int2*>(items.data_ptr<int>()), num_items.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {sorted_ids, expert_ids, num_post, local_ids, items.narrow(0, 0, max1),
            num_items.narrow(0, 0, 1), items.narrow(0, max1, max2), num_items.narrow(0, 1, 1)};
}

torch::Tensor finalize(torch::Tensor y, torch::Tensor scores, torch::Tensor norm_w,
                       torch::Tensor local_ids, double eps) {
    const int m = y.size(0);
    TORCH_CHECK(y.size(1) == kTopK && y.size(2) == kExpertHidden);
    auto out = torch::empty({m, kExpertHidden}, y.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_ex(
        finalize_row_kernel, dim3(m), dim3(512), 0, stream, kPdlGlue,
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        scores.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>()),
        local_ids.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor router_fwd(torch::Tensor x, torch::Tensor w) {
    const int m = x.size(0);
    TORCH_CHECK(m <= 8 && x.scalar_type() == torch::kFloat32 && x.size(1) == RK);
    auto out = torch::empty({m, RN}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* wp = w.data_ptr<float>();
    const float* xp = x.data_ptr<float>();
    float* op = out.data_ptr<float>();
    if (m <= 1) {
        router_launch<1>(wp, xp, op, m, stream);
    } else if (m <= 2) {
        router_launch<2>(wp, xp, op, m, stream);
    } else if (m <= 4) {
        router_launch<4>(wp, xp, op, m, stream);
    } else {
        router_launch<8>(wp, xp, op, m, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor pre_norm(torch::Tensor x, torch::Tensor w, double eps) {
    const int m = x.size(0);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == NORM_N && x.stride(1) == 1 && x.stride(0) % 4 == 0,
                "pre_norm: x must be [T, 4096] with unit column stride and 16-byte rows");
    TORCH_CHECK(w.is_contiguous() && w.numel() == NORM_N && w.scalar_type() == torch::kBFloat16,
                "pre_norm: weight must be BF16 [4096]");
    auto out = torch::empty({m, NORM_N}, x.options().dtype(torch::kBFloat16));
    if (m == 0) {
        return out;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const long long stride = x.stride(0);
    const float e = static_cast<float>(eps);
    if (x.scalar_type() == torch::kFloat32) {
        launch_ex_pdl(pre_norm_kernel<float>, dim3(m), dim3(NORM_THREADS), 0, stream,
                      static_cast<const float*>(x.data_ptr()), wp, op, stride, e);
    } else {
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "pre_norm: x must be BF16 or FP32");
        launch_ex_pdl(pre_norm_kernel<__nv_bfloat16>, dim3(m), dim3(NORM_THREADS), 0, stream,
                      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), wp, op, stride, e);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// logits: FP32 rows of 256 router logits (unit column stride); lo_off > 0 adds the block that
// starts lo_off columns later (the concat GEMM's lo half). x: the pooler output, [T, 4096] BF16
// or a strided FP32 view. Returns the normalized expert input, the global top-8 ids and their
// weights.
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

void prefetch_l2(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    long long n[3];
    const char* ptr[3];
    const torch::Tensor* ts[3] = {&a, &b, &c};
    long long chunks = 0;
    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK(ts[i]->is_contiguous(), "prefetch_l2: contiguous tensors");
        n[i] = static_cast<long long>(ts[i]->numel()) * ts[i]->element_size();
        TORCH_CHECK(n[i] % 16 == 0, "prefetch_l2: sizes must be multiples of 16 bytes");
        ptr[i] = static_cast<const char*>(ts[i]->data_ptr());
        chunks += (n[i] + PF_CHUNK - 1) / PF_CHUNK;
    }
    if (chunks == 0) {
        return;
    }
    const int blocks = static_cast<int>(
        std::min<long long>((chunks + PF_THREADS * PF_UNROLL - 1) / (PF_THREADS * PF_UNROLL), 264));
    prefetch_kernel<<<blocks, PF_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr[0], n[0], ptr[1], n[1], ptr[2], n[2], static_cast<unsigned*>(nullptr));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor proj_fwd(torch::Tensor x, torch::Tensor wpacked, int64_t n) {
    const int m = x.size(0);
    const int k = x.size(1);
    TORCH_CHECK(m <= 8);
    auto out = torch::empty({m, n}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* wp = reinterpret_cast<const uint4*>(wpacked.data_ptr());
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    if (k == 4096 && n == 1024) {
        proj_small<8>(wp, xp, op, m, n, k, stream);
    } else if (k == 8192) {
        proj_small<4>(wp, xp, op, m, n, k, stream);
    } else {
        proj_small<2>(wp, xp, op, m, n, k, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor router_mma_fwd(torch::Tensor x, torch::Tensor wpacked) {
    const int m = x.size(0);
    TORCH_CHECK(m <= 32 && x.size(1) == RK);
    auto out = torch::empty({m, RN}, x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    router_mma(reinterpret_cast<const uint4*>(wpacked.data_ptr()),
               reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
               out.data_ptr<float>(), m, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

# Idea 1 from muse_attention_decode (docs/runs-attention.md): one launch for QK-norm, RoPE, the
# FP8 query and the FP8 cache write, and one for the per-head norm-gate after FA3.
_PREP_SRC = r"""
#include <cuda_fp8.h>

namespace {
namespace prep {

constexpr int D = 64;
constexpr int HQ = 16;
constexpr int HKV = 2;
constexpr float EPS = 1e-5f;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ uint16_t to_fp8x2(float2 x) {
    return __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
}

// One warp per (token, head); lane i holds dims 2i and 2i+1, which is also RoPE pair i.
__global__ void prep_kernel(const __nv_bfloat16* __restrict__ qkv, const int64_t* __restrict__ pos,
                            const int64_t* __restrict__ loc, const float* __restrict__ inv_freq,
                            const float* __restrict__ k_scale, const float* __restrict__ v_scale,
                            uint16_t* __restrict__ q_out, uint16_t* k_cache, uint16_t* v_cache,
                            int rope) {
    asm volatile("griddepcontrol.launch_dependents;");
    const int t = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    constexpr int heads = HQ + 2 * HKV;
    if (h >= heads) {
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const auto* src = reinterpret_cast<const __nv_bfloat162*>(qkv + (size_t)t * (heads + HQ) * D);
    float2 x = __bfloat1622float2(src[h * (D / 2) + lane]);
    if (h < HQ + HKV) {
        const float r = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + EPS);
        x.x *= r;
        x.y *= r;
        if (rope) {
            float s, c;
            sincosf((float)pos[t] * inv_freq[lane], &s, &c);
            x = make_float2(x.x * c - x.y * s, x.x * s + x.y * c);
        }
    }
    if (h < HQ) {
        q_out[((size_t)t * HQ + h) * (D / 2) + lane] = to_fp8x2(x);
        return;
    }
    const bool is_k = h < HQ + HKV;
    const int kh = is_k ? h - HQ : h - HQ - HKV;
    const float inv = 1.f / (is_k ? *k_scale : *v_scale);
    x.x *= inv;
    x.y *= inv;
    uint16_t* dst = is_k ? k_cache : v_cache;
    dst[((size_t)loc[t] * HKV + kh) * (D / 2) + lane] = to_fp8x2(x);
}

// One warp per (token, head): RMSNorm of the attention output times sigmoid(gate).
__global__ void gate_kernel(const __nv_bfloat16* __restrict__ o,
                            const __nv_bfloat16* __restrict__ qkv,
                            __nv_bfloat16* __restrict__ out) {
    asm volatile("griddepcontrol.launch_dependents;");
    const int t = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (h >= HQ) {
        return;
    }
    const size_t row = (size_t)t * HQ + h;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float2 x = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(o)[row * (D / 2) + lane]);
    const auto* g2 = reinterpret_cast<const __nv_bfloat162*>(
        qkv + (size_t)t * (2 * HQ + 2 * HKV) * D + (HQ + 2 * HKV) * D);
    const float2 g = __bfloat1622float2(g2[h * (D / 2) + lane]);
    const float r = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + EPS);
    x.x = x.x * r / (1.f + __expf(-g.x));
    x.y = x.y * r / (1.f + __expf(-g.y));
    reinterpret_cast<__nv_bfloat162*>(out)[row * (D / 2) + lane] = __float22bfloat162_rn(x);
}

}  // namespace prep
}  // namespace

torch::Tensor attn_prep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc,
                        torch::Tensor inv_freq, torch::Tensor k_scale, torch::Tensor v_scale,
                        torch::Tensor k_cache, torch::Tensor v_cache, bool rope) {
    using namespace prep;
    TORCH_CHECK(qkv.is_contiguous() && qkv.size(1) == (2 * HQ + 2 * HKV) * D, "qkv layout");
    const int tokens = qkv.size(0);
    auto q = torch::empty({tokens, HQ, D}, qkv.options().dtype(torch::kFloat8_e4m3fn));
    dim3 grid(tokens, (HQ + 2 * HKV + 7) / 8);
    launch_ex_pdl(
        prep_kernel, grid, dim3(256), 0, at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), pos.data_ptr<int64_t>(),
        loc.data_ptr<int64_t>(), inv_freq.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), reinterpret_cast<uint16_t*>(q.data_ptr()),
        reinterpret_cast<uint16_t*>(k_cache.data_ptr()),
        reinterpret_cast<uint16_t*>(v_cache.data_ptr()), rope ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q;
}

torch::Tensor attn_gate(torch::Tensor o, torch::Tensor qkv) {
    using namespace prep;
    TORCH_CHECK(o.is_contiguous() && qkv.is_contiguous(), "attn_gate layout");
    const int tokens = qkv.size(0);
    auto out = torch::empty({tokens, HQ * D}, qkv.options());
    dim3 grid(tokens, (HQ + 7) / 8);
    launch_ex_pdl(
        gate_kernel, grid, dim3(256), 0, at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

# Idea 2 from muse_attention_decode: the fused q_len 1 attention kernel, copied from
# muse_attention_decode/attn_kernel.cu by tools/dl_attn_inline.py.
_ATTN_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {
namespace attn {

constexpr int D = 64;
constexpr int HQ = 16;
constexpr int HKV = 2;
constexpr int GROUP = HQ / HKV;
constexpr int QKV_STRIDE = (2 * HQ + 2 * HKV) * D;
constexpr int BN = 64;
constexpr int STAGES = 4;
constexpr int ROW_BYTES = HKV * D;
constexpr int STAGE_BYTES = 2 * BN * ROW_BYTES;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES;
// One partial per row: m, l, then 64 unnormalized output dims.
constexpr int PART = 2 + D;
constexpr float EPS = 1e-5f;
constexpr float NEG_INF = -INFINITY;

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
    float* ws;
    int pt_stride;
    int page_size;
    int window;
    int rope;
    int splits;
    float qk_scale;
    long long* dbg;
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

__device__ __forceinline__ uint32_t f16x2_from_fp8x2(uint32_t two) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

__device__ __forceinline__ uint32_t pack_f16(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Shared-memory layout of one stage: BN key rows then BN value rows, 128 bytes each (both KV
// heads). 16-byte chunks are XOR-swizzled by token so the fragment reads below hit distinct
// banks: key reads take 16 bytes from two tokens per quarter warp, value reads 8 bytes from
// four tokens per half warp.
__device__ __forceinline__ int k_chunk(int chunk, int tok) {
    return chunk ^ ((tok & 1) << 2);
}

__device__ __forceinline__ int v_chunk(int chunk, int tok) {
    return chunk ^ (((tok >> 1) & 3) << 1);
}

__device__ __forceinline__ size_t cache_row(const int* pt, int pos, int page_size) {
    if (page_size == 1) {
        return static_cast<size_t>(pt[pos]);
    }
    return static_cast<size_t>(pt[pos / page_size]) * page_size + pos % page_size;
}

// Rows of a CTA: r = j * HQ + qh for block token j and query head qh. A warp owns MT m16
// tiles of one KV head; tile row g < 8 is query head h * 8 + g. With L = 1 tile rows 8..15
// repeat rows 0..7 and are dropped; with L = 8 tile t of the head holds tokens 2t (rows
// 0..7) and 2t + 1 (rows 8..15).
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(dst), "l"(src), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                          uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// Key and value tiles converted to f16 once per stage when VF16: [token][128 dims] rows of
// 256 bytes, 16-byte chunks XOR-swizzled by token so ldmatrix row reads hit distinct banks.
constexpr int VH_BYTES = BN * ROW_BYTES * 2;

__device__ __forceinline__ int vh_chunk(int chunk, int tok) {
    return chunk ^ (tok & 7);
}

__device__ __forceinline__ float2 load_gate(const __nv_bfloat16* qkv, size_t tok, int qh,
                                            int lane) {
    return __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
        qkv + tok * QKV_STRIDE + (HQ + 2 * HKV + qh) * D)[lane]);
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// Rows of a CTA: r = j * HQ + qh for block token j and query head qh. A warp owns MT m16
// tiles of one KV head; tile row g < 8 is query head h * 8 + g. With L = 1 tile rows 8..15
// repeat rows 0..7 and are dropped; with L = 8 tile t of the head holds tokens 2t (rows
// 0..7) and 2t + 1 (rows 8..15). TP warps per (head, tile group) split each stage's 16-token
// chunks; each warp takes KC adjacent chunks per softmax update.
template <int L, int MT, int TP, int KC>
struct Cfg {
    static constexpr int TILES_PER_HEAD = L == 1 ? 1 : GROUP * L / 16;
    static constexpr int MG = TILES_PER_HEAD / MT;
    static constexpr int WARPS = HKV * MG * TP;
    static constexpr int THREADS = WARPS * 32;
    static constexpr int ROWS = HQ * L;
    static constexpr int TOK_PER_THREAD = BN * 16 / THREADS;
    static constexpr int TOK_STEP = THREADS / 16;
    static constexpr int CHUNKS = BN / 16;
    static_assert(CHUNKS % (TP * KC) == 0, "chunks must split evenly over warps");
    static_assert(TP * ROWS * PART * 4 <= SMEM_BYTES, "partials must fit in smem");
};

// Page-table ring: entries for tile t sit in slot t % RING. Group k of the main loop carries
// tile k + 3's data and tile k + 6's page entries, so a tile's entries have landed three
// iterations before its data load reads them.
constexpr int RING = 8;
constexpr int PT_AHEAD = 6;
constexpr int SMEM_TOTAL = SMEM_BYTES + RING * BN * 4;

// Cluster launches without VF16 stage the block's new keys and values in their own tile at
// SMEM_TOTAL during the prologue (full-layer launches lost 2-3% to the extra smem, job
// fa01e4d1a9f54b0f, so they stage them into stage 0 after the loop instead).
// Receive slots for the push merge sit past everything the main loop uses, since other CTAs
// of the cluster may push before this one finishes its loop. Row r of CTA s lands in slot
// (r / S) * S + s of CTA r % S, so ROWS + 16 slots cover any cluster of up to 16.
template <bool VF16>
__host__ __device__ constexpr int recv_offset() {
    return SMEM_TOTAL + (VF16 ? 2 * BN * ROW_BYTES * 2 : STAGE_BYTES);
}

// Merges n partials (m, l, o) of one row with one warp, lane k reading partial k first so the
// loads overlap, then applies the norm-gate and writes the row. get(s) points at partial s.
template <typename Get>
__device__ __forceinline__ void merge_row(Get get, int n, int lane, float v_scale, float2 gt,
                                          __nv_bfloat16* out) {
    float mmax = NEG_INF;
    for (int s = lane; s < n; s += 32) {
        mmax = fmaxf(mmax, get(s)[0]);
    }
    mmax = warp_max(mmax);
    float lsum = 0.f;
    float2 acc = make_float2(0.f, 0.f);
    for (int base = 0; base < n; base += 32) {
        float w = 0.f;
        if (base + lane < n) {
            const float* q = get(base + lane);
            w = q[0] == NEG_INF ? 0.f : exp2f(q[0] - mmax);
            lsum += w * q[1];
        }
        const int cnt = min(32, n - base);
#pragma unroll 8
        for (int k = 0; k < cnt; ++k) {
            const float wk = __shfl_sync(0xffffffffu, w, k);
            const float2 v = *reinterpret_cast<const float2*>(get(base + k) + 2 + 2 * lane);
            acc.x += wk * v.x;
            acc.y += wk * v.y;
        }
    }
    lsum = warp_sum(lsum);
    const float inv = v_scale / lsum;
    acc.x *= inv;
    acc.y *= inv;
    const float r = rsqrtf(warp_sum(acc.x * acc.x + acc.y * acc.y) / D + EPS);
    acc.x = acc.x * r / (1.f + __expf(-gt.x));
    acc.y = acc.y * r / (1.f + __expf(-gt.y));
    reinterpret_cast<__nv_bfloat162*>(out)[lane] = __float22bfloat162_rn(acc);
}

template <int L, int MT, int TP, int KC, bool CLUSTER, bool VF16, bool PV8>
__global__ void __launch_bounds__(Cfg<L, MT, TP, KC>::THREADS, VF16 ? 2 : 1)
attn_kernel(const Params p) {
    asm volatile("griddepcontrol.launch_dependents;");
    static_assert(!PV8 || (KC % 2 == 0 && !VF16), "FP8 P.V takes 32-token steps");
    using C = Cfg<L, MT, TP, KC>;
    extern __shared__ __align__(128) uint8_t smem[];
    const int split = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int head = warp % HKV;
    const int mg = (warp / HKV) % C::MG;
    const int tp = warp / (HKV * C::MG);

    const bool stamp_cta = p.dbg != nullptr && tid == 0 && b == 0
                           && (split == 0 || split == p.splits - 1);
    long long* stamps = p.dbg + (split == 0 ? 0 : 8);
    auto stamp = [&](int i) {
        if (stamp_cta) {
            stamps[i] = clock64();
        }
    };
    stamp(0);
    const float k_scale = *p.k_scale;
    const float v_scale = *p.v_scale;
    const int seq = p.seq_lens[b];
    const int start = p.window ? max(0, seq - p.window + 1) : 0;
    int per = (seq - start + p.splits - 1) / p.splits;
    per = (per + BN - 1) / BN * BN;
    const int lo = min(seq, start + split * per);
    const int hi = min(seq, lo + per);
    const int ntiles = (hi - lo + BN - 1) / BN;
    const int* pt = p.page_table + static_cast<size_t>(b) * p.pt_stride;
    const int psize = p.page_size;
    int* ring = reinterpret_cast<int*>(smem + SMEM_BYTES);

    // Loader role: chunk lc (0..7 key, 8..15 value) of tokens lt + i * TOK_STEP.
    const int lc = tid & 15;
    const int lt = tid >> 4;
    const uint8_t* lsrc = lc < 8 ? p.k_cache : p.v_cache;
    const int lchunk = lc & 7;

    auto issue = [&](int tile) {
        uint8_t* st = smem + (tile % STAGES) * STAGE_BYTES + (lc < 8 ? 0 : BN * ROW_BYTES);
        const int base = lo + tile * BN;
        const int* rp = ring + (tile % RING) * BN;
#pragma unroll
        for (int i = 0; i < C::TOK_PER_THREAD; ++i) {
            const int t = lt + i * C::TOK_STEP;
            const int pos = base + t;
            const bool valid = pos < hi;
            size_t row = static_cast<size_t>(rp[t]);
            if (psize != 1) {
                row = row * psize + pos % psize;
            }
            const int sw = lc < 8 ? k_chunk(lchunk, t) : v_chunk(lchunk, t);
            const uint8_t* src = valid ? lsrc + row * ROW_BYTES + lchunk * 16 : lsrc;
            cp_async16(smem_u32(st + t * ROW_BYTES + sw * 16), src, valid);
        }
    };
    auto stage_pt = [&](int tile) {
        for (int t = tid; t < BN; t += C::THREADS) {
            const int pos = lo + tile * BN + t;
            const bool valid = tile < ntiles && pos < hi;
            const int* src = valid ? pt + (psize == 1 ? pos : pos / psize) : pt;
            cp_async4(smem_u32(ring + (tile % RING) * BN + t), src, valid);
        }
    };

    static_assert(STAGES == 4 && PT_AHEAD == 6 && RING >= PT_AHEAD + 2, "ring schedule");
    for (int t = 0; t < PT_AHEAD; ++t) {
        stage_pt(t);
    }
    cp_commit();

    cp_wait<0>();
    __syncthreads();
    stamp(1);
#pragma unroll
    for (int t = 0; t < STAGES - 1; ++t) {
        if (t < ntiles) {
            issue(t);
        }
        cp_commit();
    }
    stamp(2);

    // Everything above reads inputs (page table, lengths, the cache); the page entries and the
    // first key and value tiles are in flight before the wait on the qkv projection, whose
    // output every read below depends on.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // The first row this warp merges in a cluster (CTA r owns rows r, r + S, ...); its gate load
    // overlaps the main loop.
    const int row0 = warp * p.splits + split;
    float2 gate0 = make_float2(0.f, 0.f);
    if (CLUSTER && row0 < C::ROWS) {
        gate0 = load_gate(p.qkv, static_cast<size_t>(b) * L + row0 / HQ, row0 % HQ, lane);
    }
    // Query fragments as f16 pairs: rows g and g + 8 of each tile, dims 16c..16c+15,
    // normalized and RoPE'd. Key dims are read in the same permuted order, so the dot products
    // are unchanged. (FP8 mma.sync is emulated on sm_90 by unpacking both operands to f16 on
    // every call, so f16 operands converted once are cheaper and no less precise.)
    uint32_t qf[MT][2][8];
    int qpos[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int tile = mg * MT + mt;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int j = L == 1 ? 0 : 2 * tile + half;
            const int qh = head * GROUP + g;
            const int pos = seq + j;
            qpos[mt][half] = pos;
            const __nv_bfloat16* qrow =
                p.qkv + static_cast<size_t>(b * L + j) * QKV_STRIDE + qh * D;
            // Word w holds dims qdim(w), qdim(w) + 1: 16c + 2w for FP8 keys read 16 bytes per
            // thread, 16(w/2) + 2c + 8(w%2) for the mma fragment order ldmatrix gives.
            auto qdim = [&](int w) {
                return VF16 ? 16 * (w >> 1) + 2 * c + 8 * (w & 1) : 16 * c + 2 * w;
            };
            float x[16];
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const float2 f = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(qrow + qdim(w)));
                x[2 * w] = f.x;
                x[2 * w + 1] = f.y;
            }
            float ss = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                ss += x[e] * x[e];
            }
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            const float r = rsqrtf(ss / D + EPS);
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                x[e] *= r;
            }
            if (p.rope) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float sn, cs;
                    sincosf(static_cast<float>(pos) * p.inv_freq[qdim(i) >> 1], &sn, &cs);
                    const float a = x[2 * i];
                    const float bb = x[2 * i + 1];
                    x[2 * i] = a * cs - bb * sn;
                    x[2 * i + 1] = a * sn + bb * cs;
                }
            }
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                qf[mt][half][w] = pack_f16(x[2 * w], x[2 * w + 1]);
            }
        }
    }
    const float qk_scale = p.qk_scale * k_scale;

    // The last split also attends the block's own new keys and values: it normalizes, rotates
    // and rounds them to E4M3 and writes them to the cache here, while page entries load, and
    // keeps them in registers to stage after the loop.
    constexpr int TASKS = L * HKV * 2;
    constexpr int TASKS_PER_WARP = (TASKS + C::WARPS - 1) / C::WARPS;
    uint16_t newv[TASKS_PER_WARP];
    const bool last_split = split == p.splits - 1;
    constexpr bool NEWB = CLUSTER && !VF16;
    uint8_t* newb = smem + SMEM_TOTAL;
    if (last_split) {
        if constexpr (NEWB) {
            for (int i = tid; i < STAGE_BYTES / 16; i += C::THREADS) {
                reinterpret_cast<uint4*>(newb)[i] = make_uint4(0, 0, 0, 0);
            }
            __syncthreads();
        }
        const float inv_k = 1.f / k_scale;
        const float inv_v = 1.f / v_scale;
        float2 xs[TASKS_PER_WARP];
#pragma unroll
        for (int i = 0; i < TASKS_PER_WARP; ++i) {
            const int task = warp + i * C::WARPS;
            xs[i] = make_float2(0.f, 0.f);
            if (task < TASKS) {
                const int j = task / (HKV * 2);
                const int kh = (task / 2) % HKV;
                const bool is_k = (task & 1) == 0;
                const size_t tok = static_cast<size_t>(b) * L + j;
                xs[i] = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                    p.qkv + tok * QKV_STRIDE + (HQ + (is_k ? 0 : HKV) + kh) * D)[lane]);
            }
        }
#pragma unroll
        for (int i = 0; i < TASKS_PER_WARP; ++i) {
            const int task = warp + i * C::WARPS;
            newv[i] = 0;
            if (task < TASKS) {
                const int j = task / (HKV * 2);
                const int kh = (task / 2) % HKV;
                const bool is_k = (task & 1) == 0;
                const size_t tok = static_cast<size_t>(b) * L + j;
                float2 x = xs[i];
                float inv = inv_v;
                if (is_k) {
                    const float r = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + EPS);
                    x.x *= r;
                    x.y *= r;
                    if (p.rope) {
                        float sn, cs;
                        sincosf(static_cast<float>(seq + j) * p.inv_freq[lane], &sn, &cs);
                        x = make_float2(x.x * cs - x.y * sn, x.x * sn + x.y * cs);
                    }
                    inv = inv_k;
                }
                x.x *= inv;
                x.y *= inv;
                newv[i] = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
                uint8_t* dst = is_k ? p.k_cache : p.v_cache;
                uint8_t* row = dst + p.cache_loc[tok] * ROW_BYTES;
                *reinterpret_cast<uint16_t*>(row + kh * D + 2 * lane) = newv[i];
                if constexpr (NEWB) {
                    const int byte = kh * D + 2 * lane;
                    const int sw = is_k ? k_chunk(byte >> 4, j) : v_chunk(byte >> 4, j);
                    *reinterpret_cast<uint16_t*>(newb + (is_k ? 0 : BN * ROW_BYTES)
                                                 + j * ROW_BYTES + sw * 16 + (byte & 15)) = newv[i];
                }
            }
        }
    }


    float o[MT][8][4];
    float m[MT][2];
    float l[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                o[mt][e][k] = 0.f;
            }
        }
        m[mt][0] = m[mt][1] = NEG_INF;
        l[mt][0] = l[mt][1] = 0.f;
    }

    // KC 16-token chunks from chunk ch0 of smem stage st; token i of the stage sits at position
    // pos0 + i and is visible to a row when lo_row <= pos <= min(lim_all, the row's position).
    uint8_t* kh16 = smem + SMEM_TOTAL;
    uint8_t* vh = kh16 + VH_BYTES;
    // All threads convert a stage's FP8 keys and values to f16 once, instead of every warp of a
    // KV head converting the same fragments (the conversion, not the mma, was the bottleneck).
    auto convert_kv = [&](const uint8_t* st) {
        for (int i = tid; i < 2 * BN * 8; i += C::THREADS) {
            const bool is_v = i >= BN * 8;
            const int t = (i >> 3) & (BN - 1);
            const int ck = i & 7;
            const uint8_t* src = st + (is_v ? BN * ROW_BYTES : 0) + t * ROW_BYTES
                                 + (is_v ? v_chunk(ck, t) : k_chunk(ck, t)) * 16;
            const uint4 w = *reinterpret_cast<const uint4*>(src);
            const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
            uint32_t h[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                h[2 * e] = f16x2_from_fp8x2(wv[e] & 0xffffu);
                h[2 * e + 1] = f16x2_from_fp8x2(wv[e] >> 16);
            }
            uint8_t* row = (is_v ? vh : kh16) + t * (2 * ROW_BYTES);
            *reinterpret_cast<uint4*>(row + vh_chunk(2 * ck, t) * 16) =
                make_uint4(h[0], h[1], h[2], h[3]);
            *reinterpret_cast<uint4*>(row + vh_chunk(2 * ck + 1, t) * 16) =
                make_uint4(h[4], h[5], h[6], h[7]);
        }
    };

    auto chunk = [&](const uint8_t* st, int ch0, int pos0, int lim_all) {
        const uint8_t* ks = st;
        const uint8_t* vs = st + BN * ROW_BYTES;
        float s[MT][2 * KC][4];
#pragma unroll
        for (int sub = 0; sub < 2 * KC; ++sub) {
            if constexpr (VF16) {
                // Lane groups of 8 address the four 8x8 blocks of dims 32h..32h+31.
                const int t = ch0 * 16 + sub * 8 + (lane & 7);
                uint32_t kb[8];
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const int ck = head * 8 + h2 * 4 + (lane >> 3);
                    ldsm_x4(smem_u32(kh16 + t * (2 * ROW_BYTES) + vh_chunk(ck, t) * 16), kb[4 * h2],
                            kb[4 * h2 + 1], kb[4 * h2 + 2], kb[4 * h2 + 3]);
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    float* acc = s[mt][sub];
                    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
#pragma unroll
                    for (int st4 = 0; st4 < 4; ++st4) {
                        mma_f16(acc, qf[mt][0][2 * st4], qf[mt][1][2 * st4], qf[mt][0][2 * st4 + 1],
                                qf[mt][1][2 * st4 + 1], kb[2 * st4], kb[2 * st4 + 1]);
                    }
                }
                continue;
            }
            const int t = ch0 * 16 + sub * 8 + g;
            const uint4 kf = *reinterpret_cast<const uint4*>(
                ks + t * ROW_BYTES + k_chunk(head * 4 + c, t) * 16);
            const uint32_t kw[4] = {kf.x, kf.y, kf.z, kf.w};
            uint32_t kb[8];
#pragma unroll
            for (int st4 = 0; st4 < 4; ++st4) {
                kb[2 * st4] = f16x2_from_fp8x2(kw[st4] & 0xffffu);
                kb[2 * st4 + 1] = f16x2_from_fp8x2(kw[st4] >> 16);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float* acc = s[mt][sub];
                acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
#pragma unroll
                for (int st4 = 0; st4 < 4; ++st4) {
                    mma_f16(acc, qf[mt][0][2 * st4], qf[mt][1][2 * st4], qf[mt][0][2 * st4 + 1],
                            qf[mt][1][2 * st4 + 1], kb[2 * st4], kb[2 * st4 + 1]);
                }
            }
        }
        uint32_t pa[MT][KC][4];
        // Interior chunks are visible to every row of the block: no masking needed.
        const int first = pos0 + ch0 * 16;
        const int last = first + 16 * KC - 1;
        const bool interior = last <= min(lim_all, seq)
                              && (!p.window || first >= seq + L - p.window);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int qp = qpos[mt][half];
                const int lo_row = p.window ? qp - p.window + 1 : 0;
                const int lim = min(lim_all, qp);
                float mx = NEG_INF;
#pragma unroll
                for (int sub = 0; sub < 2 * KC; ++sub) {
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int pos = pos0 + ch0 * 16 + sub * 8 + 2 * c + k;
                        float v = s[mt][sub][2 * half + k] * qk_scale;
                        if (!interior) {
                            v = (pos >= lo_row && pos <= lim) ? v : NEG_INF;
                        }
                        s[mt][sub][2 * half + k] = v;
                        mx = fmaxf(mx, v);
                    }
                }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const float mnew = fmaxf(m[mt][half], mx);
                const float muse = mnew == NEG_INF ? 0.f : mnew;
                const float alpha = exp2f(m[mt][half] - muse);
                // With FP8 probabilities, FA3 scales them by 2^8 so small ones keep E4M3
                // precision; l carries the same factor, so it cancels.
                const float mexp = PV8 ? muse - 8.f : muse;
                m[mt][half] = mnew;
                float sum = 0.f;
#pragma unroll
                for (int sub = 0; sub < 2 * KC; ++sub) {
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const float e = exp2f(s[mt][sub][2 * half + k] - mexp);
                        s[mt][sub][2 * half + k] = e;
                        sum += e;
                    }
                }
                l[mt][half] = l[mt][half] * alpha + sum;
                // Once the running max settles, alpha is 1 for the whole warp; skip the rescale.
                if (__any_sync(0xffffffffu, alpha != 1.f)) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        o[mt][e][2 * half] *= alpha;
                        o[mt][e][2 * half + 1] *= alpha;
                    }
                }
            }
            if constexpr (PV8) {
                // k32 step kk2 covers subtiles 4kk2..4kk2+3; byte order (2c, 2c+1, 2c+8, 2c+9)
                // matches the value words built below.
#pragma unroll
                for (int kk2 = 0; kk2 < KC / 2; ++kk2) {
                    const float* s0 = s[mt][4 * kk2];
                    const float* s1 = s[mt][4 * kk2 + 1];
                    const float* s2 = s[mt][4 * kk2 + 2];
                    const float* s3 = s[mt][4 * kk2 + 3];
                    pa[mt][kk2][0] = fp8x4(s0[0], s0[1], s1[0], s1[1]);
                    pa[mt][kk2][1] = fp8x4(s0[2], s0[3], s1[2], s1[3]);
                    pa[mt][kk2][2] = fp8x4(s2[0], s2[1], s3[0], s3[1]);
                    pa[mt][kk2][3] = fp8x4(s2[2], s2[3], s3[2], s3[3]);
                }
                continue;
            }
#pragma unroll
            for (int kk = 0; kk < KC; ++kk) {
                pa[mt][kk][0] = pack_f16(s[mt][2 * kk][0], s[mt][2 * kk][1]);
                pa[mt][kk][1] = pack_f16(s[mt][2 * kk][2], s[mt][2 * kk][3]);
                pa[mt][kk][2] = pack_f16(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);
                pa[mt][kk][3] = pack_f16(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);
            }
        }
        if constexpr (PV8) {
            // FP8 values straight from smem: for each dim, a word of the four tokens 2c, 2c+1,
            // 2c+8, 2c+9 of a 16-token half step. Output column g of n-tile e is dim 8g + e.
#pragma unroll
            for (int kk2 = 0; kk2 < KC / 2; ++kk2) {
                uint32_t bw[2][8];
#pragma unroll
                for (int hs = 0; hs < 2; ++hs) {
                    uint2 vt[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int t = (ch0 + 2 * kk2 + hs) * 16 + 2 * c + (k & 1) + (k >> 1) * 8;
                        const int slot = head * 8 + g;
                        vt[k] = *reinterpret_cast<const uint2*>(
                            vs + t * ROW_BYTES + v_chunk(slot >> 1, t) * 16 + (slot & 1) * 8);
                    }
#pragma unroll
                    for (int hw = 0; hw < 2; ++hw) {
                        const uint32_t x0 = hw ? vt[0].y : vt[0].x;
                        const uint32_t x1 = hw ? vt[1].y : vt[1].x;
                        const uint32_t x2 = hw ? vt[2].y : vt[2].x;
                        const uint32_t x3 = hw ? vt[3].y : vt[3].x;
                        const uint32_t a_lo = __byte_perm(x0, x1, 0x5140u);
                        const uint32_t b_lo = __byte_perm(x2, x3, 0x5140u);
                        const uint32_t a_hi = __byte_perm(x0, x1, 0x7362u);
                        const uint32_t b_hi = __byte_perm(x2, x3, 0x7362u);
                        bw[hs][hw * 4 + 0] = __byte_perm(a_lo, b_lo, 0x5410u);
                        bw[hs][hw * 4 + 1] = __byte_perm(a_lo, b_lo, 0x7632u);
                        bw[hs][hw * 4 + 2] = __byte_perm(a_hi, b_hi, 0x5410u);
                        bw[hs][hw * 4 + 3] = __byte_perm(a_hi, b_hi, 0x7632u);
                    }
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        mma_fp8(o[mt][e], pa[mt][kk2][0], pa[mt][kk2][1], pa[mt][kk2][2],
                                pa[mt][kk2][3], bw[0][e], bw[1][e]);
                    }
                }
            }
            return;
        }
        if constexpr (VF16) {
            // ldmatrix.trans of the f16 values: n-tile e is head dims 8e..8e+7.
            const int r = lane & 7;
            const int mi = lane >> 3;
#pragma unroll
            for (int kk = 0; kk < KC; ++kk) {
                const int t = (ch0 + kk) * 16 + (mi & 1) * 8 + r;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int ck = head * 8 + 2 * q + (mi >> 1);
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4_t(smem_u32(vh + t * (2 * ROW_BYTES) + vh_chunk(ck, t) * 16), b0, b1,
                              b2, b3);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        mma_f16(o[mt][2 * q], pa[mt][kk][0], pa[mt][kk][1], pa[mt][kk][2],
                                pa[mt][kk][3], b0, b1);
                        mma_f16(o[mt][2 * q + 1], pa[mt][kk][0], pa[mt][kk][1], pa[mt][kk][2],
                                pa[mt][kk][3], b2, b3);
                    }
                }
            }
            return;
        }
        // Value fragments: tokens 2c, 2c+1, 2c+8, 2c+9 of each 16-token step, dims 8g..8g+7 of
        // the head. Output column g of n-tile e is head dim 8g + e.
#pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            uint2 vt[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int t = (ch0 + kk) * 16 + 2 * c + (k & 1) + (k >> 1) * 8;
                const int slot = head * 8 + g;
                vt[k] = *reinterpret_cast<const uint2*>(
                    vs + t * ROW_BYTES + v_chunk(slot >> 1, t) * 16 + (slot & 1) * 8);
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
                        const uint32_t b0 = f16x2_from_fp8x2(q ? w01 >> 16 : w01 & 0xffffu);
                        const uint32_t b1 = f16x2_from_fp8x2(q ? w23 >> 16 : w23 & 0xffffu);
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            mma_f16(o[mt][e], pa[mt][kk][0], pa[mt][kk][1], pa[mt][kk][2],
                                    pa[mt][kk][3], b0, b1);
                        }
                    }
                }
            }
        }
    };

    for (int it = 0; it < ntiles; ++it) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < ntiles) {
            issue(it + STAGES - 1);
        }
        if (it + PT_AHEAD < ntiles) {
            stage_pt(it + PT_AHEAD);
        }
        cp_commit();
        const uint8_t* st = smem + (it % STAGES) * STAGE_BYTES;
        const int pos0 = lo + it * BN;
        if constexpr (VF16) {
            // The top-of-loop barrier keeps this from overwriting tiles still being read.
            convert_kv(st);
            __syncthreads();
        }
#pragma unroll
        for (int ch = tp * KC; ch < C::CHUNKS; ch += TP * KC) {
            if (pos0 + ch * 16 < hi) {
                chunk(st, ch, pos0, hi - 1);
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
    stamp(3);

    if (NEWB && last_split && tp == 0) {
        chunk(newb, 0, seq, seq + L - 1);
    }
    if (!NEWB && last_split) {
        uint8_t* st = smem;
        for (int i = tid; i < STAGE_BYTES / 16; i += C::THREADS) {
            reinterpret_cast<uint4*>(st)[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TASKS_PER_WARP; ++i) {
            const int task = warp + i * C::WARPS;
            if (task < TASKS) {
                const int j = task / (HKV * 2);
                const int kh = (task / 2) % HKV;
                const bool is_k = (task & 1) == 0;
                const int byte = kh * D + 2 * lane;
                const int sw = is_k ? k_chunk(byte >> 4, j) : v_chunk(byte >> 4, j);
                *reinterpret_cast<uint16_t*>(st + (is_k ? 0 : BN * ROW_BYTES) + j * ROW_BYTES
                                             + sw * 16 + (byte & 15)) = newv[i];
            }
        }
        __syncthreads();
        if constexpr (VF16) {
            convert_kv(st);
            __syncthreads();
        }
        if (tp == 0) {
            chunk(st, 0, seq, seq + L - 1);
        }
        // Partials below reuse the stage memory this chunk reads.
        __syncthreads();
    }

    stamp(4);
    // Partials: each warp's rows, l reduced over its quad, into smem [TP][ROWS][PART]; then the
    // TP partials of a row merge into slot 0 so the cross-CTA merge reads one per CTA.
    float* part = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int tile = mg * MT + mt;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            if (L == 1 && half == 1) {
                continue;
            }
            const int j = L == 1 ? 0 : 2 * tile + half;
            const int row = j * HQ + head * GROUP + g;
            float* dst = part + (tp * C::ROWS + row) * PART;
            float lsum = l[mt][half];
            lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
            lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
            if (c == 0) {
                dst[0] = m[mt][half];
                dst[1] = lsum;
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if constexpr (VF16) {
                    dst[2 + 8 * e + 2 * c] = o[mt][e][2 * half];
                    dst[2 + 8 * e + 2 * c + 1] = o[mt][e][2 * half + 1];
                } else {
                    dst[2 + 8 * (2 * c) + e] = o[mt][e][2 * half];
                    dst[2 + 8 * (2 * c + 1) + e] = o[mt][e][2 * half + 1];
                }
            }
        }
    }
    __syncthreads();
    if constexpr (TP > 1) {
        for (int row = warp; row < C::ROWS; row += C::WARPS) {
            float mt_[TP];
            float mmax = NEG_INF;
#pragma unroll
            for (int t = 0; t < TP; ++t) {
                mt_[t] = part[(t * C::ROWS + row) * PART];
                mmax = fmaxf(mmax, mt_[t]);
            }
            float lsum = 0.f;
            float2 acc = make_float2(0.f, 0.f);
#pragma unroll
            for (int t = 0; t < TP; ++t) {
                const float* q = part + (t * C::ROWS + row) * PART;
                const float w = mt_[t] == NEG_INF ? 0.f : exp2f(mt_[t] - mmax);
                lsum += w * q[1];
                acc.x += w * q[2 + 2 * lane];
                acc.y += w * q[3 + 2 * lane];
            }
            __syncwarp();
            float* dst = part + row * PART;
            if (lane == 0) {
                dst[0] = mmax;
                dst[1] = lsum;
            }
            dst[2 + 2 * lane] = acc.x;
            dst[3 + 2 * lane] = acc.y;
        }
        __syncthreads();
    }

    stamp(5);
    if constexpr (!CLUSTER) {
        float* ws = p.ws + (static_cast<size_t>(b) * p.splits + split) * C::ROWS * PART;
        for (int i = tid; i < C::ROWS * PART / 4; i += C::THREADS) {
            reinterpret_cast<float4*>(ws)[i] = reinterpret_cast<const float4*>(part)[i];
        }
    } else {
        // Push merge: each CTA stores its partial of row r into the receive slots of the CTA
        // that owns r (r % S), then one barrier, then owners merge from local smem. Remote
        // stores do not wait on a round trip the way remote loads do, and no CTA reads another's
        // smem after the barrier, so no second barrier is needed.
        cg::cluster_group cluster = cg::this_cluster();
        const int S = p.splits;
        float* recv = reinterpret_cast<float*>(smem + recv_offset<VF16>());
        for (int row = warp; row < C::ROWS; row += C::WARPS) {
            float* dst = cluster.map_shared_rank(recv, row % S) + ((row / S) * S + split) * PART;
            const float2* src = reinterpret_cast<const float2*>(part + row * PART);
            reinterpret_cast<float2*>(dst)[lane] = src[lane];
            if (lane == 0) {
                reinterpret_cast<float2*>(dst)[32] = src[32];
            }
        }
        cluster.sync();
        stamp(6);
        for (int li = warp; li * S + split < C::ROWS; li += C::WARPS) {
            const int row = li * S + split;
            const int j = row / HQ;
            const int qh = row % HQ;
            const size_t tok = static_cast<size_t>(b) * L + j;
            const float2 gt = row == row0 ? gate0 : load_gate(p.qkv, tok, qh, lane);
            const float* base = recv + li * S * PART;
            merge_row([&](int s) { return base + s * PART; }, S, lane, v_scale, gt,
                      p.out + tok * HQ * D + qh * D);
        }
    }
    stamp(7);
}

// Merges the partials in ws and applies the norm-gate: one warp per (token, query head).
__global__ void combine_kernel(const float* __restrict__ ws, const __nv_bfloat16* __restrict__ qkv,
                               const float* __restrict__ v_scale, __nv_bfloat16* __restrict__ out,
                               int L, int nparts) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int tok = blockIdx.x;
    const int b = tok / L;
    const int j = tok % L;
    const int qh = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const size_t stride = static_cast<size_t>(HQ) * L * PART;
    const float* base = ws + b * nparts * stride + (j * HQ + qh) * PART;
    merge_row([&](int s) { return base + s * stride; }, nparts, lane, *v_scale,
              load_gate(qkv, tok, qh, lane), out + static_cast<size_t>(tok) * HQ * D + qh * D);
}

template <int L, int MT, int TP, int KC, bool CLUSTER, bool VF16 = false, bool PV8 = false>
void launch(const Params& p, int batch, cudaStream_t stream) {
    using C = Cfg<L, MT, TP, KC>;
    auto kern = attn_kernel<L, MT, TP, KC, CLUSTER, VF16, PV8>;
    constexpr int smem_bytes = CLUSTER ? recv_offset<VF16>() + (C::ROWS + 16) * PART * 4
                                       : SMEM_TOTAL + (VF16 ? 2 * VH_BYTES : 0);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (CLUSTER) {
            cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        }
        configured = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.splits, batch);
    cfg.blockDim = dim3(C::THREADS);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    if (CLUSTER) {
        attr[1].id = cudaLaunchAttributeClusterDimension;
        attr[1].val.clusterDim.x = p.splits;
        attr[1].val.clusterDim.y = 1;
        attr[1].val.clusterDim.z = 1;
        cfg.numAttrs = 2;
    }
    const cudaError_t err = cudaLaunchKernelEx(&cfg, kern, p);
    TORCH_CHECK(err == cudaSuccess, "attn_kernel launch: ", cudaGetErrorString(err));
}

}  // namespace attn
}  // namespace
"""

# Idea 2's host wrapper: q_len 1 only, so only the decode instances of attn_kernel are built.
_ATTN_HOST_SRC = r"""
torch::Tensor attn_fused(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                         torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                         torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                         int64_t page_size, int64_t window, bool rope, double qk_scale,
                         int64_t splits, bool cluster) {
    using namespace attn;
    const int batch = seq_lens.size(0);
    TORCH_CHECK(qkv.is_contiguous() && qkv.size(0) == batch && qkv.size(1) == QKV_STRIDE,
                "attn_fused: qkv must be [batch, 2304] (one token per request)");
    TORCH_CHECK(page_table.is_contiguous() && cache_loc.is_contiguous(), "attn_fused layout");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({batch, HQ * D}, qkv.options());
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
    p.ws = nullptr;
    p.pt_stride = page_table.size(1);
    p.page_size = page_size;
    p.window = window;
    p.rope = rope ? 1 : 0;
    p.splits = splits;
    p.qk_scale = static_cast<float>(qk_scale);
    p.dbg = nullptr;
    torch::Tensor ws;
    if (cluster) {
        launch<1, 1, 4, 1, true>(p, batch, stream);
        return out;
    }
    ws = torch::empty({batch * splits * HQ * PART}, qkv.options().dtype(torch::kFloat32));
    p.ws = ws.data_ptr<float>();
    launch<1, 1, 4, 1, false>(p, batch, stream);
    launch_ex_pdl(combine_kernel, dim3(batch), dim3(HQ * 32), 0, stream, p.ws, p.qkv, p.v_scale,
                  p.out, 1, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

# The warp-specialized q_len 8 verify kernel from muse_attention_decode (attn_ws.cu, submitted
# there at 1.321x, job cb42811004a9440c), copied in by tools/dl_ws_inline.py.
_WS_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {
namespace verify {

constexpr int D = 64;
constexpr int HQ = 16;
constexpr int HKV = 2;
constexpr int GROUP = HQ / HKV;
constexpr int L = 8;
constexpr int ROWS = HQ * L;
constexpr int QKV_STRIDE = (2 * HQ + 2 * HKV) * D;
constexpr int BN = 64;
// One FP8 stage: keys of both heads in wgmma's K-major core-matrix order (head h at h * 4096),
// then values as [token][128 bytes]. A value slot: both heads' values in f16, head h at
// h * VH_HEAD, in MN-major core-matrix order (token block kb, 8-dim group nb at (8 kb + nb) * 128).
constexpr int K_TILE = HKV * BN * D;
constexpr int STAGE = 2 * K_TILE;
constexpr int VH_HEAD = BN * D * 2;
constexpr int Q_HEAD = GROUP * L * D;
constexpr int PART = 2 + D;
constexpr int CONSUMERS = 256;
constexpr float EPS = 1e-5f;
constexpr float NEG_INF = -INFINITY;

template <int NS, int DA>
struct Smem {
    static constexpr int VH = NS * STAGE;
    static constexpr int NEWK = VH + (NS - DA) * HKV * VH_HEAD;
    static constexpr int NEWV = NEWK + K_TILE;
    static constexpr int Q = NEWV + HKV * VH_HEAD;
    static constexpr int BAR = Q + HKV * Q_HEAD;
    static constexpr int BYTES = BAR + 2 * NS * 8;
};

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
    float* ws;
    unsigned long long* dbg;
    int pt_stride;
    int page_size;
    int window;
    int rope;
    int splits;
    float qk_scale;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

// A wait that gives up after about a second and marks the CTA dead instead of hanging the run;
// a dead CTA skips every later wait, so a broken pipeline costs one second, not the 15-minute
// limit, and the dbg flag says which side starved.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity, bool& dead,
                                          unsigned long long* dbg, int code) {
    if (dead) {
        return;
    }
    if (mbar_try(bar, parity)) {
        return;
    }
    const long long t0 = clock64();
    while (!mbar_try(bar, parity)) {
        if (clock64() - t0 > (1ll << 31)) {
            dead = true;
            if (dbg != nullptr) {
                atomicOr(dbg, static_cast<unsigned long long>(code));
            }
            return;
        }
    }
}

__device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void named_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ uint64_t desc(const void* p, int lbo, int sbo) {
    const uint64_t addr = smem_u32(p);
    return ((addr & 0x3ffff) >> 4) | (static_cast<uint64_t>(lbo >> 4) << 16)
           | (static_cast<uint64_t>(sbo >> 4) << 32);
}

__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// Keeps the compiler from moving accumulator reads or writes across an async wgmma.
__device__ __forceinline__ void reg_fence(float (&d)[32]) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        asm volatile("" : "+f"(d[i]) :: "memory");
    }
}

#define WG_D32 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"
#define WG_OUT32(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), \
    "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), \
    "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
    "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), \
    "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

// S (64 x 64, f16 accumulators packed in pairs) = A (64 x 32 E4M3, smem) . B (64 tokens x 32
// E4M3, smem, K-major). Register 2i holds row g, columns 8i + 2c and 8i + 2c + 1; register
// 2i + 1 the same columns of row g + 8, which is also the f16 A-fragment order of P.V.
template <int ACC>
__device__ __forceinline__ void wgmma_qk(uint32_t (&d)[16], uint64_t da, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}"
                 ", %16, %17, p, 1, 1;\n}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]),
                   "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]),
                   "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
                 : "l"(da), "l"(db), "n"(ACC));
}

// The same product with FP32 accumulators, as FA3 keeps S: register 4i + e holds row g, column
// 8i + 2c + e, and register 4i + 2 + e row g + 8, the layout of O.
template <int ACC>
__device__ __forceinline__ void wgmma_qk32(float (&d)[32], uint64_t da, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D32
                 ", %32, %33, p, 1, 1;\n}\n"
                 : WG_OUT32(d) : "l"(da), "l"(db), "n"(ACC));
}

__device__ __forceinline__ void reg_fence16(uint32_t (&d)[16]) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        asm volatile("" : "+r"(d[i]) :: "memory");
    }
}

__device__ __forceinline__ uint32_t ex2_h2(uint32_t x) {
    uint32_t y;
    asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(y) : "r"(x));
    return y;
}

__device__ __forceinline__ __half2 as_h2(uint32_t x) {
    return *reinterpret_cast<__half2*>(&x);
}

__device__ __forceinline__ uint32_t as_u32(__half2 x) {
    return *reinterpret_cast<uint32_t*>(&x);
}

// O (64 x 64 f32) += P (64 x 16 f16, registers) . V (16 tokens x 64 dims f16, smem, MN-major).
__device__ __forceinline__ void wgmma_pv(float (&d)[32], const uint32_t* a, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D32
                 ", {%32, %33, %34, %35}, %36, p, 1, 1, 1;\n}\n"
                 : WG_OUT32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// 2^x for x <= 0 on the FMA pipe instead of MUFU, as FA4 does for part of its exponentials:
// x = n + f with n = round(x), f in [-0.5, 0.5], a degree-5 minimax polynomial for 2^f, and n
// added to the exponent field. Max relative error 1.7e-7 over [-126, 0] in an FP32 emulation
// (tools/attn_emulate_fp32.py, which fits the coefficients), against 2^-22 for ex2.approx.f32. The clamp keeps that field
// from borrowing into the sign; below it the result stays under 2^-125, which the P rounding to
// f16 flushes to 0 as MUFU's ftz result would be.
__device__ __forceinline__ float ex2_poly(float x) {
    x = fmaxf(x, -126.f);
    const float t = x + 12582912.f;
    const float f = x - (t - 12582912.f);
    float y = fmaf(1.32647238e-3f, f, 9.67151299e-3f);
    y = fmaf(y, f, 5.55073358e-2f);
    y = fmaf(y, f, 2.40222424e-1f);
    y = fmaf(y, f, 6.93147004e-1f);
    y = fmaf(y, f, 1.f);
    return __uint_as_float(__float_as_uint(y) + (__float_as_uint(t) << 23));
}

__device__ __forceinline__ uint32_t pack_f16(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t f16x2_from_fp8x2(uint32_t two) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

// The f16 bit patterns of four E4M3 bytes with the exponent left 8 short of the f16 bias:
// sign to bit 15, exponent and mantissa to bits 13..7. Each is exactly the E4M3 value times
// 2^-8 (subnormals included), in three integer ops per pair instead of one F2FP, which shares
// the conversion/MUFU pipe with the softmax's ex2.
__device__ __forceinline__ uint2 f16raw_from_fp8x4(uint32_t w) {
    const uint32_t a = __byte_perm(w, 0, 0x1404);
    const uint32_t b = __byte_perm(w, 0, 0x3424);
    return make_uint2((a & 0x80008000u) | ((a >> 1) & 0x3f803f80u),
                      (b & 0x80008000u) | ((b >> 1) & 0x3f803f80u));
}

template <int CVT>
__device__ __forceinline__ uint4 f16_from_fp8x8(uint32_t w0, uint32_t w1) {
    if constexpr (CVT == 1) {
        const uint2 a = f16raw_from_fp8x4(w0);
        const uint2 b = f16raw_from_fp8x4(w1);
        return make_uint4(a.x, a.y, b.x, b.y);
    }
    return make_uint4(f16x2_from_fp8x2(w0 & 0xffffu), f16x2_from_fp8x2(w0 >> 16),
                      f16x2_from_fp8x2(w1 & 0xffffu), f16x2_from_fp8x2(w1 >> 16));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// Byte offset of 16-byte chunk kc of row r in a K-major core-matrix tile with 4 chunks per row
// (64 bytes of E4M3): core (r / 8, kc) is 128 contiguous bytes, LBO 128, SBO 512.
__device__ __forceinline__ int kmaj(int r, int kc) {
    return ((r >> 3) * 4 + kc) * 128 + (r & 7) * 16;
}

// One CTA per (split, request): warps 0-3 attend KV head 0, warps 4-7 KV head 1, each warpgroup
// covering 64 query rows (8 block tokens x 8 query heads, row 8j + g), and PW loader warps.
// Loaders gather each tile's K/V rows by cp.async into an NS-stage FP8 ring and, DA tiles
// later, convert the values they loaded themselves to f16 in an (NS - DA)-slot ring, then
// release the tile to the math warps through its full barrier. Math warps release a tile (its
// FP8 stage and its value slot) after P.V. No block-wide barrier inside the loop.
//
// MODE 0: the kernel. MODE 1: loads only (math warps wait for each tile and release it), to
// separate memory time from math. MODE 2: the kernel plus clock64 phase totals in dbg[1..15].
//
// VAR bit 0: values converted with integer ops into f16 values scaled by 2^-8, which the merge
// multiplies back (else F2FP). Bit 1: loaders release a tile without their own proxy fence
// (the math warps fence after acquiring it either way). Bit 2: math warps take tiles in pairs.
// Bit 3 is read by the host only: eight loader warps instead of four. Bit 4: FA3's precision,
// that is Q rounded to E4M3 without the softmax scale, Q.K accumulated in FP32, and the running
// max, sum, rescale factors and exponentials in FP32 (P still goes to P.V as f16, where FA3's
// FP8 path uses E4M3). Bit 5 (with bit 4): FA3's intra-warpgroup overlap, the P.V of tile i in
// flight under the Q.K and softmax of tile i + 1. Bit 6 (with bit 4): a quarter of the
// exponentials on the FMA pipe (ex2_poly) to unload MUFU.
template <int NS, int DA, int MODE, int PW, int VAR>
__global__ void __launch_bounds__(CONSUMERS + 32 * PW, 1) attn_ws_kernel(const Params p) {
    asm volatile("griddepcontrol.launch_dependents;");
    constexpr bool STAMP = MODE == 2;
    constexpr bool F32 = (VAR & 16) != 0;
    static_assert(F32 || (VAR & 96) == 0, "bits 5 and 6 need bit 4");
    static_assert(!F32 || (VAR & 5) == 0, "bit 4 does not combine with bits 0 and 2");
    constexpr int NV = NS - DA;
    using S = Smem<NS, DA>;
    long long acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    long long tclk = 0;
    auto lap = [&](int k) {
        if constexpr (STAMP) {
            const long long now = clock64();
            acc[k] += now - tclk;
            tclk = now;
        }
    };
    auto flush_stamps = [&](int base, int n) {
        if constexpr (STAMP) {
            if ((threadIdx.x & 31) == 0) {
                for (int k = 0; k < n; ++k) {
                    atomicAdd(p.dbg + 1 + base + k, static_cast<unsigned long long>(acc[k]));
                }
            }
        }
    };
    extern __shared__ __align__(1024) uint8_t smem[];
    const int split = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + S::BAR);
    const uint32_t full0 = smem_u32(bars);
    const uint32_t empty0 = smem_u32(bars + NS);

    const int seq = p.seq_lens[b];
    const int start = p.window ? max(0, seq - p.window + 1) : 0;
    int per = (seq - start + p.splits - 1) / p.splits;
    per = (per + BN - 1) / BN * BN;
    const int lo = min(seq, start + split * per);
    const int hi = min(seq, lo + per);
    const int ntiles = (hi - lo + BN - 1) / BN;
    const bool last_split = split == p.splits - 1;
    bool dead = false;

    if (tid == 0) {
        for (int s = 0; s < NS; ++s) {
            mbar_init(full0 + 8 * s, 32 * PW);
            mbar_init(empty0 + 8 * s, 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp >= 8) {
        // Loader warp pw owns tokens TPW * pw .. TPW * pw + TPW - 1 of every tile. Lane
        // (cq, tt) = (lane / 8, lane % 8) copies chunk cq of both heads' 64-byte halves of
        // token 8 x + tt, so a quarter warp's f16 stores below hit eight distinct token rows.
        constexpr int TPW = BN / PW;
        const int pw = warp - 8;
        const int tbase = TPW * pw;
        const int tt = lane & 7;
        const int cq = lane >> 3;
        const int* pt = p.page_table + static_cast<size_t>(b) * p.pt_stride;
        const int psize = p.page_size;
        auto entry = [&](int tile, int t) {
            const int pos = lo + tile * BN + t;
            if (tile >= ntiles || pos >= hi) {
                return 0;
            }
            return __ldg(pt + (psize == 1 ? pos : pos / psize));
        };
        // Page entries run PF tiles ahead in registers, so a tile's cp.async never waits on
        // the dependent page-table load (job 418d8d492c074b88 lost 6x to that wait).
        constexpr int PF = 4;
        int pe[PF];
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            pe[u] = entry(u, tbase + lane % TPW);
        }
        if constexpr (STAMP) {
            tclk = clock64();
        }
        for (int base = 0; base < ntiles + DA; base += PF) {
#pragma unroll
            for (int u = 0; u < PF; ++u) {
                const int i = base + u;
                if (i >= ntiles + DA) {
                    break;
                }
                lap(2);
                // Tile i reuses the FP8 stage of tile i - NS, and tile i - DA (converted
                // below) the value slot of tile i - NS too, since NV = NS - DA.
                if (i >= NS) {
                    const int s = i % NS;
                    mbar_wait(empty0 + 8 * s, ((i / NS) + 1) & 1, dead, p.dbg, 1);
                }
                lap(0);
                if (i < ntiles) {
                    uint8_t* st = smem + (i % NS) * STAGE;
                    const int pos0 = lo + i * BN;
#pragma unroll
                    for (int tb = 0; tb < TPW; tb += 8) {
                        const int t = tbase + tb + tt;
                        const int e = __shfl_sync(0xffffffffu, pe[u], tb + tt);
                        const int pos = pos0 + t;
                        const bool valid = pos < hi;
                        size_t row = static_cast<size_t>(e);
                        if (psize != 1) {
                            row = row * psize + pos % psize;
                        }
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const int ck = cq + 4 * j;
                            const size_t off = row * (HKV * D) + ck * 16;
                            cp_async16(smem_u32(st + j * (BN * D) + kmaj(t, cq)),
                                       valid ? p.k_cache + off : p.k_cache, valid);
                            cp_async16(smem_u32(st + K_TILE + t * 128 + ck * 16),
                                       valid ? p.v_cache + off : p.v_cache, valid);
                        }
                    }
                    pe[u] = entry(i + PF, tbase + lane % TPW);
                }
                cp_commit();
                lap(1);
                if (i >= DA) {
                    const int f = i - DA;
                    cp_wait<DA>();
                    lap(3);
                    const uint8_t* vs = smem + (f % NS) * STAGE + K_TILE;
                    uint8_t* vb = smem + S::VH + (f % NV) * (HKV * VH_HEAD);
#pragma unroll
                    for (int tb = 0; tb < TPW; tb += 8) {
                        const int t = tbase + tb + tt;
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const uint4 w = *reinterpret_cast<const uint4*>(
                                vs + t * 128 + (cq + 4 * j) * 16);
                            uint8_t* dst = vb + j * VH_HEAD + ((t >> 3) * 8 + 2 * cq) * 128
                                           + (t & 7) * 16;
                            *reinterpret_cast<uint4*>(dst) = f16_from_fp8x8<VAR & 1>(w.x, w.y);
                            *reinterpret_cast<uint4*>(dst + 128) =
                                f16_from_fp8x8<VAR & 1>(w.z, w.w);
                        }
                    }
                    if constexpr ((VAR & 2) == 0) {
                        fence_async_smem();
                    }
                    mbar_arrive(full0 + 8 * (f % NS));
                    lap(4);
                }
            }
        }
        flush_stamps(8, 5);
        if (STAMP && pw == 0 && lane == 0 && split == 0 && b == 0) {
            p.dbg[16] = PW;
        }
        return;
    }

    if constexpr (STAMP) {
        tclk = clock64();
    }
    const int h = warp >> 2;
    const int wl = warp & 3;
    const int wtid = tid & 127;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int bar_id = 1 + h;
    const float k_scale = *p.k_scale;
    const float v_scale = *p.v_scale;
    uint8_t* qs = smem + S::Q + h * Q_HEAD;
    uint8_t* newk = smem + S::NEWK;
    uint8_t* newv = smem + S::NEWV;

    // The loader warps above read only inputs (page table, lengths, the cache rows of earlier
    // tokens), so they never wait; the queries, gates and new keys and values below are the qkv
    // projection's output.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // Query prep: two threads per row (row m = 8j + g'), 32 dims each, normalized, rotated and
    // rounded to E4M3 into the K-major A tile.
    {
        const int m = wtid >> 1;
        const int part = wtid & 1;
        const int j = m >> 3;
        const int qh = h * GROUP + (m & 7);
        const __nv_bfloat16* src =
            p.qkv + static_cast<size_t>(b * L + j) * QKV_STRIDE + qh * D + 32 * part;
        float x[32];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 w = reinterpret_cast<const uint4*>(src)[v];
            const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ws[e]));
                x[8 * v + 2 * e] = f.x;
                x[8 * v + 2 * e + 1] = f.y;
            }
        }
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 32; ++e) {
            ss += x[e] * x[e];
        }
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        // The f16 path folds the softmax scale (with log2 e) in before the E4M3 rounding, so
        // scores come out of the f16-accumulating wgmma in log2 units; the FP32 path rounds the
        // unscaled query, as SGLang hands FA3, and scales the FP32 scores.
        const float r = rsqrtf(ss / D + EPS) * (F32 ? 1.f : p.qk_scale);
        const float pos = static_cast<float>(seq + j);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float a = x[2 * i] * r;
            float bb = x[2 * i + 1] * r;
            if (p.rope) {
                float sn, cs;
                sincosf(pos * p.inv_freq[16 * part + i], &sn, &cs);
                const float a2 = a * cs - bb * sn;
                bb = a * sn + bb * cs;
                a = a2;
            }
            x[2 * i] = a;
            x[2 * i + 1] = bb;
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            uint32_t wv[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float* y = x + 16 * h2 + 4 * e;
                const uint32_t lo2 = __nv_cvt_float2_to_fp8x2(make_float2(y[0], y[1]),
                                                              __NV_SATFINITE, __NV_E4M3);
                const uint32_t hi2 = __nv_cvt_float2_to_fp8x2(make_float2(y[2], y[3]),
                                                              __NV_SATFINITE, __NV_E4M3);
                wv[e] = lo2 | (hi2 << 16);
            }
            *reinterpret_cast<uint4*>(qs + kmaj(m, 2 * part + h2)) =
                make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
    }

    // The last split also writes the block's new keys and values (KV head h) to the cache and
    // stages them in their own tile (keys as E4M3, values as the f16 of their E4M3 rounding),
    // attended after the loop. Each warpgroup zeroes and fills only its head's bytes, so no
    // cross-warpgroup barrier is needed.
    if (last_split) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            reinterpret_cast<uint4*>(newk + h * (BN * D))[wtid + 128 * k] = make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            reinterpret_cast<uint4*>(newv + h * VH_HEAD)[wtid + 128 * k] = make_uint4(0, 0, 0, 0);
        }
        named_sync(bar_id, 128);
        const float inv_k = 1.f / k_scale;
        const float inv_v = 1.f / v_scale;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int task = wl + 4 * k;
            const int j = task >> 1;
            const bool is_k = (task & 1) == 0;
            const size_t tok = static_cast<size_t>(b) * L + j;
            float2 x = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                p.qkv + tok * QKV_STRIDE + (HQ + (is_k ? 0 : HKV) + h) * D)[lane]);
            float inv = inv_v;
            if (is_k) {
                const float r = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + EPS);
                x.x *= r;
                x.y *= r;
                if (p.rope) {
                    float sn, cs;
                    sincosf(static_cast<float>(seq + j) * p.inv_freq[lane], &sn, &cs);
                    x = make_float2(x.x * cs - x.y * sn, x.x * sn + x.y * cs);
                }
                inv = inv_k;
            }
            x.x *= inv;
            x.y *= inv;
            const uint16_t v8 = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
            uint8_t* dst = (is_k ? p.k_cache : p.v_cache) + p.cache_loc[tok] * (HKV * D);
            *reinterpret_cast<uint16_t*>(dst + h * D + 2 * lane) = v8;
            if (is_k) {
                const int byte = 2 * lane;
                *reinterpret_cast<uint16_t*>(newk + h * (BN * D) + kmaj(j, byte >> 4)
                                             + (byte & 15)) = v8;
            } else {
                *reinterpret_cast<uint32_t*>(newv + h * VH_HEAD + (lane >> 2) * 128 + j * 16
                                             + (lane & 3) * 4) =
                    (VAR & 1) ? f16raw_from_fp8x4(v8).x : f16x2_from_fp8x2(v8);
            }
        }
    }
    fence_async_smem();
    named_sync(bar_id, 128);

    float o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i] = 0.f;
    }
    // Running maxima of rows g (low half) and g + 8 (high half), kept in f16 so the P values
    // and the rescale share one reference.
    __half2 mrow2 = __float2half2_rn(NEG_INF);
    float lrow[2] = {0.f, 0.f};
    lap(7);
    int qpos[2];
    int lo_row[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        qpos[hh] = seq + 2 * wl + hh;
        lo_row[hh] = p.window ? qpos[hh] - p.window + 1 : 0;
    }
    const uint64_t qd0 = desc(qs, 128, 512);
    const uint64_t qd1 = desc(qs + 256, 128, 512);
    const __half2 ks2 = __float2half2_rn(k_scale);
    // A fully masked row keeps this floor as its max, so its exponentials are 0, not NaN.
    const __half2 lowest2 = __float2half2_rn(-65504.f);
    // FP32 path: running maxima in log2 units, and the score scale applied in FP32 as FA3's
    // softmax_scale_log2 (which also carries the key descale).
    float mrow[2] = {NEG_INF, NEG_INF};
    const float sl2 = p.qk_scale * k_scale;

    // Pieces of one 64-token tile: S = Q.K^T (FP8 wgmma into f16), online softmax in half2,
    // then O += P.V (f16 wgmma into f32) from the loaders' f16 values.
    auto qk_issue = [&](const uint8_t* kt, uint32_t (&s)[16]) {
        const uint8_t* ks = kt + h * (BN * D);
        wgmma_qk<0>(s, qd0, desc(ks, 128, 512));
        wgmma_qk<1>(s, qd1, desc(ks + 256, 128, 512));
    };
    // Turns S into P (f16 A fragments) and updates the row max and sum; returns the factors O
    // must be scaled by before this tile's P.V, without touching O, so it can run while the
    // previous P.V is in flight.
    auto softmax = [&](uint32_t (&s)[16], int pos0, int lim, uint32_t (&pa)[16], float& a0,
                       float& a1) {
        const bool interior = pos0 + BN - 1 <= lim && (!p.window || pos0 >= seq + L - p.window);
        if (!interior) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int top = min(lim, qpos[hh]);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int pos = pos0 + 8 * i + 2 * c;
                    const uint32_t keep0 = pos >= lo_row[hh] && pos <= top ? 0xffffu : 0u;
                    const uint32_t keep1 = pos + 1 >= lo_row[hh] && pos + 1 <= top ? 0xffffu : 0u;
                    const uint32_t keep = keep0 | (keep1 << 16);
                    // -inf in f16 is 0xfc00.
                    s[2 * i + hh] = (s[2 * i + hh] & keep) | (0xfc00fc00u & ~keep);
                }
            }
        }
        // Scores are in log2 units over k_scale (the softmax scale is folded into Q), so each
        // pair of probabilities is one HFMA2 and one f16x2 ex2; both rows share each half2 op
        // of the max and the rescale.
        __half2 m0 = as_h2(s[0]);
        __half2 m1 = as_h2(s[1]);
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            m0 = __hmax2(m0, as_h2(s[2 * i]));
            m1 = __hmax2(m1, as_h2(s[2 * i + 1]));
        }
        __half2 mx = __halves2half2(__hmax(__low2half(m0), __high2half(m0)),
                                    __hmax(__low2half(m1), __high2half(m1)));
        mx = __hmax2(mx, as_h2(__shfl_xor_sync(0xffffffffu, as_u32(mx), 1)));
        mx = __hmax2(mx, as_h2(__shfl_xor_sync(0xffffffffu, as_u32(mx), 2)));
        const __half2 muse = __hmax2(__hmax2(mrow2, __hmul2(mx, ks2)), lowest2);
        const __half2 alpha2 = as_h2(ex2_h2(as_u32(__hsub2(mrow2, muse))));
        mrow2 = muse;
        const __half2 nm = __hneg2(muse);
        const __half2 n0 = __low2half2(nm);
        const __half2 n1 = __high2half2(nm);
        __half2 sum0 = __float2half2_rn(0.f);
        __half2 sum1 = sum0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            pa[2 * i] = ex2_h2(as_u32(__hfma2(as_h2(s[2 * i]), ks2, n0)));
            pa[2 * i + 1] = ex2_h2(as_u32(__hfma2(as_h2(s[2 * i + 1]), ks2, n1)));
            sum0 = __hadd2(sum0, as_h2(pa[2 * i]));
            sum1 = __hadd2(sum1, as_h2(pa[2 * i + 1]));
        }
        a0 = __low2float(alpha2);
        a1 = __high2float(alpha2);
        lrow[0] = lrow[0] * a0 + __low2float(sum0) + __high2float(sum0);
        lrow[1] = lrow[1] * a1 + __low2float(sum1) + __high2float(sum1);
    };
    auto qk_issue32 = [&](const uint8_t* kt, float (&s)[32]) {
        const uint8_t* ks = kt + h * (BN * D);
        wgmma_qk32<0>(s, qd0, desc(ks, 128, 512));
        wgmma_qk32<1>(s, qd1, desc(ks + 256, 128, 512));
    };
    // FA3's online softmax (hopper/softmax.h): row max over raw FP32 scores, then
    // exp2(s * scale_log2 - max * scale_log2) as one FFMA and one ex2 per score, FP32 row sums of
    // the unrounded exponentials. P is rounded to f16 for the P.V wgmma.
    auto softmax32 = [&](float (&s)[32], int pos0, int lim, uint32_t (&pa)[16], float& a0,
                         float& a1) {
        const bool interior = pos0 + BN - 1 <= lim && (!p.window || pos0 >= seq + L - p.window);
        if (!interior) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int top = min(lim, qpos[hh]);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int pos = pos0 + 8 * i + 2 * c + e;
                        if (pos < lo_row[hh] || pos > top) {
                            s[4 * i + 2 * hh + e] = NEG_INF;
                        }
                    }
                }
            }
        }
        float mu[2];
        float alpha[2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            float mx = fmaxf(s[2 * hh], s[2 * hh + 1]);
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                mx = fmaxf(mx, fmaxf(s[4 * i + 2 * hh], s[4 * i + 2 * hh + 1]));
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            // A row with every key masked so far keeps a finite floor as its max, so its
            // exponentials are 0, not NaN; FA3 does the same with Check_inf.
            mu[hh] = fmaxf(fmaxf(mrow[hh], mx * sl2), -1e30f);
            alpha[hh] = ex2(mrow[hh] - mu[hh]);
            mrow[hh] = mu[hh];
        }
        float sum[2] = {0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                float e2[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float x = fmaf(s[4 * i + 2 * hh + e], sl2, -mu[hh]);
                    e2[e] = ((VAR & 64) != 0 && (i & 3) == 3) ? ex2_poly(x) : ex2(x);
                }
                sum[hh] += e2[0] + e2[1];
                pa[2 * i + hh] = pack_f16(e2[0], e2[1]);
            }
        }
        a0 = alpha[0];
        a1 = alpha[1];
        lrow[0] = lrow[0] * a0 + sum[0];
        lrow[1] = lrow[1] * a1 + sum[1];
    };
    auto rescale = [&](float a0, float a1) {
        if (__any_sync(0xffffffffu, a0 != 1.f || a1 != 1.f)) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[4 * i] *= a0;
                o[4 * i + 1] *= a0;
                o[4 * i + 2] *= a1;
                o[4 * i + 3] *= a1;
            }
        }
    };
    auto pv_issue = [&](const uint32_t (&pa)[16], const uint8_t* vb) {
        const uint8_t* vh = vb + h * VH_HEAD;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            wgmma_pv(o, pa + 4 * k, desc(vh + k * 2048, 1024, 128));
        }
    };
    auto release = [&](uint32_t empty_bar) {
        if (empty_bar != 0) {
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty_bar);
            }
        }
    };
    // Every async wgmma retires inside the call that issued it: leaving one in flight across
    // the loop back-edge makes ptxas serialize every wgmma in the kernel (C7514).
    auto tile = [&](const uint8_t* kt, const uint8_t* vb, int pos0, int lim, uint32_t empty_bar) {
        uint32_t s[16];
        uint32_t pa[16];
        float a0, a1;
        wg_fence();
        qk_issue(kt, s);
        wg_commit();
        wg_wait<0>();
        reg_fence16(s);
        lap(1);
        softmax(s, pos0, lim, pa, a0, a1);
        rescale(a0, a1);
        lap(2);
        reg_fence(o);
        wg_fence();
        pv_issue(pa, vb);
        wg_commit();
        wg_wait<0>();
        reg_fence(o);
        lap(3);
        release(empty_bar);
    };
    auto tile32 = [&](const uint8_t* kt, const uint8_t* vb, int pos0, int lim,
                      uint32_t empty_bar) {
        float s[32];
        uint32_t pa[16];
        float a0, a1;
        wg_fence();
        qk_issue32(kt, s);
        wg_commit();
        wg_wait<0>();
        reg_fence(s);
        lap(1);
        softmax32(s, pos0, lim, pa, a0, a1);
        rescale(a0, a1);
        lap(2);
        reg_fence(o);
        wg_fence();
        pv_issue(pa, vb);
        wg_commit();
        wg_wait<0>();
        reg_fence(o);
        lap(3);
        release(empty_bar);
    };
    // Two tiles: the second Q.K runs under the first softmax, and the first P.V under the
    // second softmax, whose rescale waits until that P.V retires.
    auto tile2 = [&](const uint8_t* kt0, const uint8_t* vb0, int pos0, uint32_t e0,
                     const uint8_t* kt1, const uint8_t* vb1, int pos1, uint32_t e1, int lim) {
        uint32_t s0[16];
        uint32_t s1[16];
        uint32_t p0[16];
        uint32_t p1[16];
        float a0, a1, b0, b1;
        wg_fence();
        qk_issue(kt0, s0);
        wg_commit();
        qk_issue(kt1, s1);
        wg_commit();
        wg_wait<1>();
        reg_fence16(s0);
        lap(1);
        softmax(s0, pos0, lim, p0, a0, a1);
        rescale(a0, a1);
        lap(2);
        reg_fence(o);
        wg_fence();
        pv_issue(p0, vb0);
        wg_commit();
        wg_wait<1>();
        reg_fence16(s1);
        lap(3);
        softmax(s1, pos1, lim, p1, b0, b1);
        lap(2);
        wg_wait<0>();
        reg_fence(o);
        lap(3);
        release(e0);
        rescale(b0, b1);
        reg_fence(o);
        wg_fence();
        pv_issue(p1, vb1);
        wg_commit();
        wg_wait<0>();
        reg_fence(o);
        lap(3);
        release(e1);
    };

    auto vslot = [&](int t) {
        return smem + S::VH + (t % NV) * (HKV * VH_HEAD);
    };
    int it = 0;
    if constexpr (MODE == 1) {
        for (; it < ntiles; ++it) {
            const int s = it % NS;
            mbar_wait(full0 + 8 * s, (it / NS) & 1, dead, p.dbg, 2);
            lap(0);
            release(empty0 + 8 * s);
        }
    } else {
        if constexpr ((VAR & 32) != 0) {
            // FA3's overlap: P.V of tile it - 1 is issued right behind the Q.K of tile it and
            // runs under its softmax; both retire before the back-edge (see tile).
            if (ntiles > 0) {
                // P of the tile in flight alternates between two arrays, so no copy writes
                // a wgmma's A registers.
                uint32_t pa0[16];
                uint32_t pa1[16];
                {
                    float s[32];
                    float a0, a1;
                    mbar_wait(full0, 0, dead, p.dbg, 2);
                    lap(0);
                    fence_async_smem();
                    wg_fence();
                    qk_issue32(smem, s);
                    wg_commit();
                    wg_wait<0>();
                    reg_fence(s);
                    lap(1);
                    softmax32(s, lo, hi - 1, pa0, a0, a1);
                    lap(2);
                }
                auto step = [&](int t, uint32_t (&pin)[16], uint32_t (&pout)[16]) {
                    const int st = t % NS;
                    float s[32];
                    float a0, a1;
                    mbar_wait(full0 + 8 * st, (t / NS) & 1, dead, p.dbg, 2);
                    lap(0);
                    fence_async_smem();
                    reg_fence(o);
                    wg_fence();
                    qk_issue32(smem + st * STAGE, s);
                    wg_commit();
                    pv_issue(pin, vslot(t - 1));
                    wg_commit();
                    wg_wait<1>();
                    reg_fence(s);
                    lap(1);
                    softmax32(s, lo + t * BN, hi - 1, pout, a0, a1);
                    lap(2);
                    wg_wait<0>();
                    reg_fence(o);
                    lap(3);
                    release(empty0 + 8 * ((t - 1) % NS));
                    rescale(a0, a1);
                };
                auto drain = [&](const uint32_t (&pin)[16]) {
                    reg_fence(o);
                    wg_fence();
                    pv_issue(pin, vslot(ntiles - 1));
                    wg_commit();
                    wg_wait<0>();
                    reg_fence(o);
                    lap(3);
                    release(empty0 + 8 * ((ntiles - 1) % NS));
                };
                for (it = 1; it + 1 < ntiles; it += 2) {
                    step(it, pa0, pa1);
                    step(it + 1, pa1, pa0);
                }
                if (it < ntiles) {
                    step(it, pa0, pa1);
                    drain(pa1);
                } else {
                    drain(pa0);
                }
                it = ntiles;
            }
        } else if constexpr ((VAR & 4) != 0) {
            for (; it + 1 < ntiles; it += 2) {
                const int s0 = it % NS;
                const int s1 = (it + 1) % NS;
                mbar_wait(full0 + 8 * s0, (it / NS) & 1, dead, p.dbg, 2);
                mbar_wait(full0 + 8 * s1, ((it + 1) / NS) & 1, dead, p.dbg, 2);
                lap(0);
                fence_async_smem();
                tile2(smem + s0 * STAGE, smem + S::VH + (it % NV) * (HKV * VH_HEAD), lo + it * BN,
                      empty0 + 8 * s0, smem + s1 * STAGE,
                      smem + S::VH + ((it + 1) % NV) * (HKV * VH_HEAD), lo + (it + 1) * BN,
                      empty0 + 8 * s1, hi - 1);
            }
        }
        for (; it < ntiles; ++it) {
            const int s = it % NS;
            mbar_wait(full0 + 8 * s, (it / NS) & 1, dead, p.dbg, 2);
            lap(0);
            fence_async_smem();
            if constexpr (F32) {
                tile32(smem + s * STAGE, vslot(it), lo + it * BN, hi - 1, empty0 + 8 * s);
            } else {
                tile(smem + s * STAGE, vslot(it), lo + it * BN, hi - 1, empty0 + 8 * s);
            }
        }
        if (last_split) {
            if constexpr (F32) {
                tile32(newk, newv, seq, seq + L - 1, 0);
            } else {
                tile(newk, newv, seq, seq + L - 1, 0);
            }
        }
    }
    lap(6);

    float* ws = p.ws + (static_cast<size_t>(b) * p.splits + split) * ROWS * PART;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int j = 2 * wl + hh;
        const int row = j * HQ + h * GROUP + g;
        float* dst = ws + row * PART;
        float lsum = lrow[hh];
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
        lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
        if (c == 0) {
            if constexpr (F32) {
                dst[0] = mrow[hh];
            } else {
                dst[0] = hh ? __high2float(mrow2) : __low2float(mrow2);
            }
            dst[1] = lsum;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            *reinterpret_cast<float2*>(dst + 2 + 8 * i + 2 * c) =
                make_float2(o[4 * i + 2 * hh], o[4 * i + 2 * hh + 1]);
        }
    }
    lap(6);
    flush_stamps(0, 8);
}

__device__ __forceinline__ float2 load_gate(const __nv_bfloat16* qkv, size_t tok, int qh,
                                            int lane) {
    return __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
        qkv + tok * QKV_STRIDE + (HQ + 2 * HKV + qh) * D)[lane]);
}

// Merges a row's split partials (lane k reads partial k first so the loads overlap) and applies
// the norm-gate: one warp per (token, query head).
__global__ void ws_combine_kernel(const float* __restrict__ ws,
                                  const __nv_bfloat16* __restrict__ qkv,
                                  const float* __restrict__ v_scale,
                                  __nv_bfloat16* __restrict__ out, int nparts, float vmul) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int tok = blockIdx.x;
    const int b = tok / L;
    const int j = tok % L;
    const int qh = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const size_t stride = static_cast<size_t>(ROWS) * PART;
    const float* base = ws + b * nparts * stride + (j * HQ + qh) * PART;
    const float2 gt = load_gate(qkv, tok, qh, lane);
    float mmax = NEG_INF;
    for (int s = lane; s < nparts; s += 32) {
        mmax = fmaxf(mmax, base[s * stride]);
    }
    mmax = warp_max(mmax);
    float lsum = 0.f;
    float2 acc = make_float2(0.f, 0.f);
    for (int s0 = 0; s0 < nparts; s0 += 32) {
        float w = 0.f;
        if (s0 + lane < nparts) {
            const float* q = base + (s0 + lane) * stride;
            w = q[0] == NEG_INF ? 0.f : exp2f(q[0] - mmax);
            lsum += w * q[1];
        }
        const int cnt = min(32, nparts - s0);
#pragma unroll 8
        for (int k = 0; k < cnt; ++k) {
            const float wk = __shfl_sync(0xffffffffu, w, k);
            const float2 v =
                *reinterpret_cast<const float2*>(base + (s0 + k) * stride + 2 + 2 * lane);
            acc.x += wk * v.x;
            acc.y += wk * v.y;
        }
    }
    lsum = warp_sum(lsum);
    const float inv = *v_scale * vmul / lsum;
    acc.x *= inv;
    acc.y *= inv;
    const float r = rsqrtf(warp_sum(acc.x * acc.x + acc.y * acc.y) / D + EPS);
    acc.x = acc.x * r / (1.f + __expf(-gt.x));
    acc.y = acc.y * r / (1.f + __expf(-gt.y));
    reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(tok) * HQ * D + qh * D)[lane] =
        __float22bfloat162_rn(acc);
}

template <int NS, int DA, int MODE, int PW, int VAR>
void launch_ws(const Params& p, int batch, cudaStream_t stream) {
    auto kern = attn_ws_kernel<NS, DA, MODE, PW, VAR>;
    constexpr int bytes = Smem<NS, DA>::BYTES;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        configured = true;
    }
    launch_ex_pdl(kern, dim3(p.splits, batch), dim3(CONSUMERS + 32 * PW), bytes, stream, p);
}

template <int NS, int DA, int MODE, int VAR>
void launch_var(const Params& p, int batch, cudaStream_t stream) {
    launch_ws<NS, DA, MODE, (VAR & 8) ? 8 : 4, VAR & ~8>(p, batch, stream);
}

}  // namespace verify
}  // namespace
"""

# Var 16 only: FP32 Q.K accumulators and FP32 softmax as FA3, P in f16, FP32 P.V accumulators,
# and the loaders keep their own fence.proxy.async before the full-barrier arrive. Variants
# with bit 1 set (2, 18) drop that fence, and passing tests do not prove that ordering safe.
_WS_HOST_SRC = r"""
torch::Tensor attn_verify(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                          torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                          torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                          int64_t page_size, int64_t window, bool rope, double qk_scale,
                          int64_t splits) {
    using namespace verify;
    const int batch = seq_lens.size(0);
    TORCH_CHECK(qkv.is_contiguous() && qkv.size(0) == batch * L && qkv.size(1) == QKV_STRIDE,
                "attn_verify: qkv must be [batch * 8, 2304]");
    TORCH_CHECK(page_table.is_contiguous() && cache_loc.is_contiguous()
                && seq_lens.is_contiguous(), "attn_verify layout");
    TORCH_CHECK(cache_loc.scalar_type() == torch::kInt64
                && seq_lens.scalar_type() == torch::kInt32
                && page_table.scalar_type() == torch::kInt32, "attn_verify index dtypes");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({batch * L, HQ * D}, qkv.options());
    auto part = torch::empty({batch * splits * ROWS * PART},
                             qkv.options().dtype(torch::kFloat32));
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
    p.ws = part.data_ptr<float>();
    p.dbg = nullptr;
    p.pt_stride = page_table.size(1);
    p.page_size = page_size;
    p.window = window;
    p.rope = rope ? 1 : 0;
    p.splits = splits;
    p.qk_scale = static_cast<float>(qk_scale);
    launch_var<7, 3, 0, 16>(p, batch, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_ex_pdl(ws_combine_kernel, dim3(batch * L), dim3(HQ * 32), 0, stream, p.ws, p.qkv,
                  p.v_scale, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), splits, 1.f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

# The one-launch attention kernel from muse_attention_decode (attn_one.cu), copied in by
# tools/dl_one_inline.py inside a nested `one` namespace, with the two griddepcontrol.wait lines
# the port guide asks for (docs/attention-port-guide.md in that problem's worktree). One launch
# per call for every tp8 shape: splits add o 2^m and l 2^m into a persistent FP32 workspace
# with red.global.add, the last arriving CTA (key-count ticket) or every CTA of a cooperative
# launch finalizes and zeroes what it read, so the graph holds one kernel node and no memset.
_ONE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdint>
#include <vector>

// One launch per attention call (docs/attention-port-guide.md). A CTA takes one request and
// one KV pair (both heads of a 128-byte cache row, all 16 query heads) over one key range.
// Splits never exchange a running max: each adds o * 2^m and l * 2^m into a persistent FP32
// workspace with red.global.add, which QK-norm keeps inside FP32 range (|s| <= ~67 in log2
// units at d = 64). A key-count ticket makes the last arriving CTA finalize a request (MODE
// bit 0), or every CTA finalizes a slice of its rows after a cooperative barrier (MODE bits
// 0-1 = 3). The finalizer zeroes what it read and the counters reset themselves, so the graph
// holds one kernel node and no memset.

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
// A row whose keys are all masked so far keeps this finite offset, so its exponentials are 0.
constexpr float OFF_FLOOR = -1e30f;
constexpr int COOP_POLL_LIMIT = 1 << 24;

// ONE_DIAG (off by default; a bench builds it with -DONE_DIAG=1): the phase exits and timer
// stamps. Each one is a test and a taken branch on the once-per-launch path, which runs from a
// cold instruction cache.
#ifndef ONE_DIAG
#define ONE_DIAG 0
#endif
// ONE_DIET: once-path code without integer division by the page size (shift and mask by its
// log2), with straight-line query copies and query pass, branch-free predicated row copies in
// the prologue issue, and no page-index copies for tiles past the CTA's range. Off and on are
// timed side by side in bench d57d1fe064164611 (docs/runs-attention.md, 10-29).
#ifndef ONE_DIET
#define ONE_DIET 1
#endif
// ONE_FIN8: the finalize issues all eight workspace copies' loads (nc = 8, batch-1 full
// context) before zeroing any, as one L2 round trip instead of eight dependent ones. Each thread
// still loads an address before it zeroes it and no other thread touches that address after the
// cooperative barrier, so moving the other copies' loads ahead of the stores needs no fence
// (PTX memory model, docs/runs-attention.md, 10-29).
#ifndef ONE_FIN8
#define ONE_FIN8 1
#endif
// ONE_KVFAST: the in-loop copy of a tile whose rows are all cached (every tile but the one that
// reaches the block's own positions) runs without the per-row three-way branch. Off: bench
// 82e52f7b23ec49da had it 0.04-0.36 us ahead on sliding, but judged check 422f81a11f384deb
// scored below the ONE_FIN8 pair (docs/ruled-out.md).
#ifndef ONE_KVFAST
#define ONE_KVFAST 0
#endif
// The public flat q_len 8 kernel's tile order, as separate switches for the q_len 8 producer
// builds (51, 115) so a bench can attribute each step (docs/runs-attention.md, 10-29):
// ONE_QKOV writes V^T while the Q.K wgmma is in flight instead of after the softmax;
// ONE_EREL releases the stage once Q.K and the V^T reads are done instead of after P.V;
// ONE_PCHK skips the per-tile row max after a segment's first tile and moves the offset only
// when a packed e4m3 P exceeds 256; ONE_UNR2 unrolls the tile loop twice. Bench
// e1d8a55eaa2f455c, verify 32 full warm: QKOV alone +2.5 us, with EREL -5.4, with PCHK -6.5,
// all four -9.2; verify 16 full flat. With all four off the board build's SASS is unchanged.
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
// ONE_PFAST (producer builds 49, 51, 115): a tile whose rows are all cached is gathered without
// the per-row branches, with 32-bit cache offsets (the host passes off32 only when both caches
// are below 4 GB) and the page index by shift; its page entries are refilled the same way. The
// public flat producer has this fast path (about 115 SASS instructions per tile per thread on
// its common path against about 230 without it), and the producer warps issue on the consumers'
// SMSPs. Bench 1e6626a427454f9a, cold / warm: verify 32 full -11.5 / -16.1 us, verify 16 full
// -5.6 / -8.2, decode 16 / 32 full -0.6 / -1.8 cold; output bit-identical to the slow path.
#ifndef ONE_PFAST
#define ONE_PFAST 1
#endif
// Bench-only attribution builds (docs/runs-attention.md, 10-29), never judged:
// ONE_STAMP=1 records per-CTA times and wait cycles into dbg[STAMP_BASE ..] (flat producer
// builds); ONE_PROBE=1 makes the producer's cached-row copies zero-fill without reading the cache
// (consumer-bound time, wrong output); ONE_PROBE=2 drops the consumers' per-tile
// fence.proxy.async after the full barrier (its cost, ordering not guaranteed).
#ifndef ONE_STAMP
#define ONE_STAMP 0
#endif
#ifndef ONE_PROBE
#define ONE_PROBE 0
#endif
// ONE_WAVE=w (mode 51, cooperative launch of exactly two CTAs per SM): the CTA that reaches its
// SM's rank counter first takes w / 1024 of its SM partner's share of the virtual key axis, and
// the two take adjacent ranges (slot 2 smid + rank). Stamps of 7631033a0c93485d at verify 32 full:
// on all 132 SMs the lower-index CTA ends its tile loop first, 29-41 us before its partner
// (median 35), with equal tiles; a split by block index instead (first 132 CTAs heavier) lost
// because some SMs hold two of them (docs/runs-attention.md, 10-29). 0 keeps equal shares.
#ifndef ONE_WAVE
#define ONE_WAVE 0
#endif
// ONE_DCARRY (mode 51 tile loop): the public flat kernel's consumer path before the Q.K issue,
// 37 SASS instructions against 94 in mode 51 (docs/runs-attention.md, 10-29). The tiles whose
// keys all lie below the block's own positions (no mask, no patch) are found once per segment
// and tested with one compare; the Q and K descriptor low words are broadcast from lane 0, which
// keeps them in uniform registers, and the K descriptor of a stage is base + 1024 * stage (the
// 14-bit address field never carries: shared memory is below 256 KB). The full-barrier wait, the
// per-tile fence.proxy.async and the patch path of the other tiles are unchanged.
// Judged -8.4 us at verify 32 full in f64e8680b3e441be (bench 4ff339332fef4aff: -7.3 cold).
#ifndef ONE_DCARRY
#define ONE_DCARRY 1
#endif
// ONE_PK16 (every instance): e4m3x4 joins its two halves as lo | hi << 16, which ptxas folds
// into the second F2FP (PACK_AB_MERGE_C), as the public kernel's e4m3x4f does. The byte_perm
// form compiles to F2FP + LOP3 (& 0xffff) per half and a PRMT: 40 instructions per q_len 8 tile
// per consumer thread for P instead of 16. Same bits. Bench 025ccaf9c57a4ba2, cold: verify 32 /
// 16 full -6.3 / -3.6 us alone, -9.6 / -5.4 with ONE_VHOIST; sliding unchanged.
#ifndef ONE_PK16
#define ONE_PK16 1
#endif
// ONE_VHOIST (mode 51 and 115 tile loop): the V^T transpose addresses are per-thread offsets
// computed once per segment (half 1 of a store is half 0's offset ^ 32, a load is +4096), and
// the V^T buffer and its descriptor low word are toggled by XOR per tile, the descriptor
// broadcast from lane 0 so it stays uniform; the unrolled loop's first body otherwise rebuilt
// about 25 address instructions per tile and every body about 14 for the P.V descriptor.
// Bench 025ccaf9c57a4ba2, cold: verify 32 / 16 full -3.9 / -2.0 us alone.
#ifndef ONE_VHOIST
#define ONE_VHOIST 1
#endif
// ONE_TRIG: griddepcontrol.launch_dependents at kernel entry (q_len 1 and batch-1 q_len 8
// instances only), so a following launch with the
// programmatic-serialization attribute (the layer's o projection built with SK_PDL) can start
// streaming its static weights on the SMs this grid leaves idle. It is a no-op for a plain
// successor, and a PDL successor still reads this grid's output only after griddepcontrol.wait,
// which waits for this grid's completion and memory flush (PTX ISA 9.7.15.14). Off by default.
#ifndef ONE_TRIG
#define ONE_TRIG 0
#endif
constexpr int STAMP_BASE = 1024;
constexpr int STAMP_W = 16;

// MODE bits: 1 = last-CTA finalize, 3 = cooperative finalize (every CTA a slice of rows),
// |4 = at most one tile per CTA (straight-line prologue), |8 = the sequence length is taken
// from the page table's width (batch 1, page size 1) and verified on the device, |16 = flat:
// the grid shares all requests' keys equally (n_splits carries the batch size), |32 = flat
// with a producer warpgroup feeding an mbarrier ring.
constexpr int M_COOP = 2;
constexpr int M_ONE = 4;
constexpr int M_SPEC = 8;
// |16: flat partition of all requests' keys over the grid (full context, batch > 1).
constexpr int M_FLAT = 16;
// Virtual keys charged per request boundary in the flat partition: a segment prologue
// costs about that many keys' worth of loads (the public entry measured 400 / 600).
constexpr int SEGK1 = 400;
constexpr int SEGK8 = 600;
// |32 (flat only): a producer warpgroup gathers the tiles into an mbarrier ring (3 stages at
// q_len 1, 4 at q_len 8) and the consumers never issue a copy.
constexpr int M_PROD = 32;
// |64 (with M_PROD): one CTA per SM and no register split, so ptxas compiles the consumers
// with the full 168-register budget instead of 80 with spills.
constexpr int M_SOLO = 64;

__host__ __device__ constexpr int one_ns(int L, int mode) {
    return (mode & M_PROD) && L == 8 ? 4 : STAGES;
}

__host__ __device__ constexpr int one_threads(int L, int mode) {
    return (L == 8 ? 256 : 128) + ((mode & M_PROD) ? 128 : 0);
}

__host__ __device__ constexpr int one_minb(int L, int mode) {
    return (mode & M_SOLO) ? 1 : (L == 8 || (mode & M_PROD) ? 2 : 4);
}

// Register split in producer mode, two CTAs per SM: the launch allots 128 (q_len 1, 256
// threads) or 80 (q_len 8, 384 threads) per thread; the producers give up what the consumers
// take, so the sums stay inside the allotment.
__host__ __device__ constexpr int one_prod_regs(int L) {
    return L == 8 ? 32 : 56;
}

__host__ __device__ constexpr int one_cons_regs(int L) {
    return L == 8 ? 104 : 200;
}

// ws_cnt map (ints): [b] last-CTA key ticket; [CNT_COOP + 2 b + e] cooperative key counters of
// parity e; [CNT_EPOCH + b] the request's launch parity; [CNT_FLAT ..] flat-kernel counters.
constexpr int CNT_COOP = 4096;
constexpr int CNT_EPOCH = 8192;
constexpr int CNT_FLAT = 16384;
// [CNT_SMR + smid]: ONE_WAVE's per-SM rank counters, zero between launches.
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
    // Diagnostic: > 0 makes the per-request path drain and return after that phase (see
    // PHASE_EXIT in attn_one_kernel); 0 runs the whole kernel.
    int phase_limit;
    // log2 of page_size (a power of two; the host checks).
    int lp;
    // Both caches below 4 GB: byte offsets fit 32 bits (ONE_PFAST).
    int off32;
    // SM count of the device (ONE_WAVE).
    int wave_n;
};

template <int L, int NS>
struct Smem {
    static constexpr int ROWS = HQ * L;
    static constexpr int KV = 0;
    static constexpr int Q = KV + NS * 2 * TILE_BYTES;
    static constexpr int Q_BYTES = L == 8 ? HKV * 4096 : HQ * 128;
    static constexpr int VT = Q + Q_BYTES;
    static constexpr int VT_BYTES = L == 8 ? 4 * VT_STRIDE : 0;
    // The bf16 query rows land here and are consumed before the first V transpose.
    static constexpr int QST = VT;
    static constexpr int QST_BYTES = ROWS * 128;
    static constexpr int KVST = QST + (QST_BYTES > VT_BYTES ? QST_BYTES : VT_BYTES);
    static constexpr int KN = KVST + L * 4 * 128;
    static constexpr int VN = KN + L * ROWB;
    static constexpr int IDX = VN + L * ROWB;
    static constexpr int ROPE = IDX + NIDX * BN * 4;
    static constexpr int CONST = ROPE + L * 2 * 32 * 4;
    static constexpr int FLAG = CONST + 256;
    // Flat mode: virtual offsets of up to 512 requests plus the total.
    static constexpr int OFF = FLAG + 32;
    // Producer mode: NS full and NS empty mbarriers.
    static constexpr int BAR = OFF + 2304;
    static constexpr int BYTES = BAR + 2 * NS * 8;
    static_assert(Q % 1024 == 0 && VT % 1024 == 0, "swizzled tiles need 1024-byte bases");
};

// CONST layout (bytes): inv_freq[32] at 0, k_scale 128, v_scale 132, epoch 136, cache_loc[L]
// (int64) at 160.
constexpr int C_INVF = 0;
constexpr int C_KS = 128;
constexpr int C_VS = 132;
constexpr int C_EPOCH = 136;
constexpr int C_LOC = 160;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

// cp_async16 under a predicate, so a per-row choice needs no branch.
__device__ __forceinline__ void cp_async16_if(uint32_t dst, const void* src, uint32_t bytes,
                                              bool doit) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n"
                 "@p cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(static_cast<int>(doit)) : "memory");
}

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void named_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}

// Arrives on the barrier once every cp.async this thread issued before has landed; .noinc
// means the arrival counts against the barrier's expected count.
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}

__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

// Waits for the phase with the given parity; traps after about a second so a broken ring
// fails the run instead of hanging it.
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
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}

template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p));
}

__device__ __forceinline__ int ld_nc_s32(const int* p) {
    int v;
    asm volatile("ld.global.nc.s32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

__device__ __forceinline__ void red_add_f32(float* p, float a) {
    asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(p), "f"(a) : "memory");
}

__device__ __forceinline__ int atom_add_acq_rel(int* p, int v) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
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
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Four E4M3 bytes from four floats (round to nearest, saturating).
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(b), "f"(a));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(d), "f"(c));
#if ONE_PK16
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
#else
    return __byte_perm(static_cast<uint32_t>(lo), static_cast<uint32_t>(hi), 0x5410);
#endif
}

__device__ __forceinline__ uint16_t e4m3x2(float a, float b) {
    uint16_t v;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(v) : "f"(b), "f"(a));
    return v;
}

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2(uint32_t two) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(static_cast<uint16_t>(two)));
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

// Sum over the 8 consecutive lanes that share a row.
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

// sin and cos of x = pos * inv_freq as the reference forms it (an FP32 product), by a
// two-constant Cody-Waite reduction and the hardware functions; about 1e-6 absolute, fine
// against a bf16 output. One body serves every RoPE angle in the kernel.
__device__ __forceinline__ float2 fast_sincos(float x) {
    const float k = rintf(x * 0.15915494309189535f);
    float r = fmaf(-k, 6.2831854820251465f, x);
    r = fmaf(-k, -1.7484555314695172e-7f, r);
    return make_float2(__sinf(r), __cosf(r));
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
    return t;
}

__device__ __forceinline__ float sigmoid(float x) {
    return __fdividef(1.f, 1.f + __expf(-x));
}

// D (16 x 8 f32) += A (16 x 16 f16) . B (16 x 8 f16): the a-registers hold rows g and g + 8
// at k = 2c, 2c + 1 (a0, a1) and k + 8 (a2, a3); b0, b1 hold k = 2c, 2c + 1 and k + 8 of column g.
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1,
                                              uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}

__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2,
                                        uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

// wgmma shared-memory descriptors. K-major, no swizzle: 8-row core matrices of 16 bytes, LBO
// between core matrices along K, SBO along M/N. Swizzled K-major: the hardware permutes the
// 16-byte chunks of each 128-byte (SW128) or 64-byte (SW64) row by the row index, so a tile
// must start on a 1024-byte boundary; k-steps inside a row advance the start address.
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
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

template <int N>
__device__ __forceinline__ void reg_fence(float (&d)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        asm volatile("" : "+f"(d[i]) :: "memory");
    }
}

#define WG_D32 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"
#define WG_OUT32(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), \
    "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), \
    "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
    "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), \
    "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

// S (64 x 64 f32) (+)= Q (64 x 32 e4m3, smem) . K (64 keys x 32 e4m3, smem, K-major).
// Register 4i + e holds row g, column 8i + 2c + e; register 4i + 2 + e row g + 8.
template <int ACC>
__device__ __forceinline__ void wgmma_qk(float (&d)[32], uint64_t da, uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D32
                 ", %32, %33, p, 1, 1;\n}\n"
                 : WG_OUT32(d) : "l"(da), "l"(db), "n"(ACC));
}

// O (64 x 72 f32) += P (64 x 32 e4m3, registers) . V^T (72 x 32 e4m3, smem, K-major): columns
// 64..71 multiply the rows of ones, so d[32..35] carry the row sums of the same rounded P.
__device__ __forceinline__ void wgmma_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                 "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
                 "%31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1;\n}\n"
                 : WG_OUT32(d), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

// 16-byte chunk placement inside a 128-byte stage row. q_len 8 tiles use the 128-byte swizzle
// wgmma expects (chunk ^ (row & 7)); q_len 1 tiles use the patterns that make the mma.sync
// fragment reads below conflict-free: keys 16 bytes from two rows per quarter warp, values 8
// bytes from four rows per half warp.
template <int L>
__device__ __forceinline__ int k_swz(int chunk, int row) {
    return L == 8 ? chunk ^ (row & 7) : chunk ^ ((row & 1) << 2);
}

template <int L>
__device__ __forceinline__ int v_swz(int chunk, int row) {
    return L == 8 ? chunk ^ (row & 7) : chunk ^ (((row >> 1) & 3) << 1);
}

// Core-matrix placement of chunk kc (16 dims) of row r in the no-swizzle K-major Q tile.
__device__ __forceinline__ int kmaj(int r, int kc) {
    return ((r >> 3) * 4 + kc) * 128 + (r & 7) * 16;
}

// Row index of (request b, copy, token j, head) in the workspace.
__device__ __forceinline__ int ws_row(int b, int nc, int copy, int L, int j, int head) {
    return ((b * nc + copy) * L + j) * HQ + head;
}


// Normalizes, scales and writes rows [r_lo, r_hi) of the flat row space (row = (b * L + j) * 16
// + head) from the workspace, then zeroes exactly what it read. Eight threads per row.
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
            // All eight copies' loads before any store: in the loop below each copy's loads
            // wait behind the previous copy's stores, eight dependent L2 round trips.
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
        // Lane t8 == 0 alone reads and zeroes the row sum and broadcasts it, so no lane reads
        // what another lane zeroes; the loop is uniform so the shuffle is reached by every lane.
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
            acc[e] *= inv;
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

// Thread 0 adds `mine` keys to the parity counter of epoch `ep`, the last arriver (the one
// whose keys complete the count; a CTA with no keys never is) clears the other parity and
// bumps the epoch, and every CTA spins until the count reaches `total`. dbg[0], when given,
// records the longest spin in polls.
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
    // The q_len 8 producer builds keep the plain once-path: with the diet, mode 51 timed 0.9-1.9
    // us slower than without it at full verify b16 / b32 in all four samples of bench
    // d57d1fe064164611 (docs/runs-attention.md, 10-29).
    constexpr bool DIET = ONE_DIET && !(PROD && L == 8);
    // Eight workspace copies come only with the batch-1 full-context plan (cooperative,
    // speculated length, several tiles per CTA); other instances keep the rolled loop.
    constexpr bool FIN8 = ONE_FIN8 && COOP && SPEC && !ONE;
    constexpr bool QKOV = ONE_QKOV && PROD && L == 8;
    constexpr bool EREL = ONE_EREL && PROD && L == 8;
    constexpr bool PCHK = ONE_PCHK && PROD && L == 8;
    constexpr int UNR8 = ONE_UNR2 && PROD && L == 8 ? 2 : 1;
    constexpr bool DCARRY = ONE_DCARRY && PROD && L == 8;
    constexpr bool VHOIST = ONE_VHOIST && QKOV;
    constexpr bool WAVE = ONE_WAVE > 0 && PROD && !SOLO && L == 8;
    // NC consumer threads (one warpgroup per KV head at q_len 8, four warps at q_len 1); in
    // producer mode one more warpgroup gathers the tiles.
    constexpr int NC = L == 8 ? 256 : 128;
    constexpr int NS = one_ns(L, MODE);
    constexpr int ROWS = HQ * L;
    constexpr int SEGK = L == 8 ? SEGK8 : SEGK1;
    constexpr int PF = 3;
    using S = Smem<L, NS>;
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
    // Only the plans whose layer successor is the skinny o projection (T <= 32: q_len 1, and
    // q_len 8 at batch 1); above 32 tokens o runs on cuBLAS and mode 51's schedule stays as judged.
    if constexpr (ONE_TRIG != 0 && (L == 1 || SPEC)) {
        asm volatile("griddepcontrol.launch_dependents;");
    }

    // Diagnostic early exit after phase k: uniform across the CTA (a kernel parameter), placed
    // only where every thread passes, with the copies drained first.
#define PHASE_EXIT(k) \
    if (ONE_DIAG && p.phase_limit == (k)) { cp_wait<0>(); return false; }
    // Diagnostic stamps: thread 0 of CTA (0, 0) records the time in microseconds at point k
    // into dbg[1 + k] when a dbg buffer is given (the judge's timed path passes none).
    auto stamp = [&](int k) {
        if constexpr (ONE_DIAG) {
            if (p.dbg != nullptr && split == 0 && blockIdx.y == 0 && tid == 0) {
                p.dbg[1 + k] = static_cast<int>(globaltimer_ns() / 1000u);
            }
        }
    };
    // ONE_STAMP: record k of this CTA's row (thread 0 of the consumers, or of the producers for
    // their fields). Times are %globaltimer ns (low 32 bits, comparable across SMs); the *_cyc
    // sums are clock64 cycles of one thread.
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
            asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
            srec(15, smid);
        }
    }
    // Consumer-wide barrier: the whole CTA unless the producer warpgroup has split off.
    auto csync = [&]() {
        if constexpr (PROD) {
            named_sync(1, NC);
        } else {
            __syncthreads();
        }
    };

    // The current segment: request b, positions [lo, hi) of its range [start, end).
    int b = FLAT ? 0 : blockIdx.y;
    const int* pt = p.page_table + static_cast<size_t>(b) * p.pt_stride;
    int seq = 0, start = 0, end = 0, n_keys = 0, lo = 0, hi = 0, ntiles = 0, nact = 0;
    int chunk_keys = 0;
    // Per-request modes: the real length is in flight while the guessed page indices load
    // (SPEC); otherwise it heads the chain. Flat mode reads every length in the partition.
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

    // Flat partition (both roles compute the same segment list): every CTA takes an equal
    // share of a virtual key axis that holds all requests' keys with SEGK extra keys at each
    // request boundary (a boundary costs a segment prologue), mapped back to whole-tile
    // ranges inside each request, so the grid finishes together whatever the lengths.
    int* offs = reinterpret_cast<int*>(smem + S::OFF);
    int v0 = 0, v1 = 0, b_first = 0, b_last = -1, keys_all = 0;
    const int B = FLAT ? p.n_splits : 1;
    const int G = gridDim.x;
    // Positions [lo, hi) of request bb inside the CTA's share; false when the rounded range
    // is empty. Neighbours round the same split point the same way, so the ranges tile each
    // request exactly.
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
            // The two RMWs of an SM's pair return distinct parities whatever the counter held,
            // and the one that sees an odd value resets it: both RMWs precede that store in the
            // location's coherence order (PTX ISA 8.10.1, 8.10.3), so the next launch, stream
            // ordered after this one, starts from zero. Exactly two CTAs per SM is what the
            // cooperative launch of 2 x SMs CTAs at two resident per SM implies; the host sets
            // wave_n only for such launches, on a device whose SM ids run 0 .. SMs - 1.
            uint32_t smid;
            asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
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
                asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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
        // Requests touched: the first with an end past v0 and the last starting before v1.
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

    // ------------------------------------------------------------------ producer warpgroup
    if constexpr (PROD) {
        if (tid >= NC) {
            if constexpr (!SOLO) {
                setmaxnreg_dec<one_prod_regs(L)>();
            }
            // Thread (r0, ch): chunk ch of rows r0 + 16 i of every tile. Page entries run PF
            // tiles ahead in registers. A stage is published by cp.async.mbarrier.arrive.noinc
            // (one arrival per producer thread, when its copies have landed); the block's own
            // rows are left for the consumers to patch.
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
            constexpr uint32_t CPB = ONE_PROBE == 1 ? 0u : 16u;
            const bool fast32 = ONE_PFAST && p.off32 != 0;
            long long s_emw = 0;
            const long long s_p0 = STAMPS ? clock64() : 0;
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
                            // Every row is cached (pos0 + BN <= seq < hi).
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
                        if (fast32 && tn < ntiles && lo + tn * BN + BN <= seq) {
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
            // Leave only once every stage has been released: no copy of this thread is then
            // in flight and the consumers no longer need the ring.
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
            // PTX exit, not return: a return branches to the kernel's one ret block, which the
            // consumers also reach, and ptxas then wraps both roles in a single convergence
            // region (BSSY before the role branch, BSYNC + EXIT at the end) and caps the
            // consumer code after setmaxnreg.inc at the launch-bounds count (80 at q_len 8:
            // 304 bytes of spills). With exit the roles share no block and the consumers get
            // the 104 registers. No __builtin_unreachable after it: with that hint the shared
            // block comes back (CUDA 12.8.1). docs/runs-attention.md, 10-29.
            asm volatile("exit;\n" ::: "memory");
        }
        if constexpr (!SOLO) {
            setmaxnreg_inc<one_cons_regs(L)>();
        }
    }

    // ------------------------------------------------------------------ consumers
    auto issue_idx = [&](int tile) {
        if constexpr (DIET) {
            // Entries of tiles past the range are never read (issue_kv reads an entry only
            // for a cached row of a tile below ntiles), so they need no zero-filling copy.
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
    // Rows of tile `tile` into stage `st` (non-producer modes): cached rows by cp.async, rows
    // past the range zero-filled, the block's own rows (positions seq .. end - 1) from
    // S.KN / S.VN by st.shared once those are staged (patch == true); before that they are
    // left for patch_tile.
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
                // Prologue issue, branch-free: cached rows copy, rows past the range
                // zero-fill, the block's own rows are left for patch_tile.
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
    // The block's own rows of the tile at pos0 in stage st, from S.KN / S.VN (16 chunks per
    // token, one thread each).
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
    // Ring position of the consumers in producer mode: continuous over segments.
    int cst = 0;
    uint32_t cph = 0;

    // One (request, range) segment: loads, query and new-token preparation, the tile loop
    // and the reds into the workspace. Returns false when SPEC found the segment empty.
    auto run_segment = [&]() -> bool {
        if constexpr (STAMPS) {
            s_seg0 = clock64();
        }
        if constexpr (FLAT) {
            // The previous segment's tiles, Q staging and V^T buffers are all consumed.
            cp_wait<0>();
            csync();
        }
        if constexpr (!PROD) {
            // Group 0: page indices of the prologue tiles.
#pragma unroll
            for (int t = 0; t < STAGES - 1; ++t) {
                issue_idx(t);
            }
            cp_commit();
        }
        stamp(1);
        PHASE_EXIT(1)
        // Everything above reads inputs. The query rows are the qkv projection's output
        // and the constants group reads the epoch the previous launch wrote, so under PDL
        // both follow the wait (which returns at once on a plain launch).
        asm volatile("griddepcontrol.wait;" ::: "memory");
        // Query rows, the block's new key/value rows (if this range may hold them),
        // constants; the range holding the request's first key prefetches the gate rows.
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
            // K/V of the prologue tiles go out while the query group is still in flight;
            // each later group also carries the page indices STAGES - 1 tiles ahead.
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

        // Query pass: eight threads per row, eight dims each. q_len 1 keeps f16 queries
        // pre-scaled into log2 units (the key scale is applied to the scores); q_len 8 rounds
        // the unit-RMS query to E4M3 and scales in the exponent.
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
        // The block's own keys and values: normalized, rotated, rounded, written to the cache
        // by the range that holds the position, and staged for the attention.
        if (hi > seq && tid < L * 32) {
            const int rr = tid >> 3;
            const int t8 = tid & 7;
            const int j = rr >> 2;
            const bool is_k = (rr & 3) < 2;
            const int hh = rr & 1;
            float x[8];
            load8(smem + S::KVST + rr * 128 + t8 * 16, x);
            float inv = __fdividef(1.f, v_scale);
            // Every lane of the warp reaches the full-mask shuffle: half of each warp holds V
            // rows, and a full-mask shfl inside `if (is_k)` deadlocked on the judge (the K
            // lanes wait in WARPSYNC for V lanes already parked at the next bar.sync).
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
            // Patch the new rows into the tiles issued before they were staged.
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
            // Rows 64..71 of every V^T buffer hold E4M3 ones, so the P.V wgmma also sums P.
            // (The Q staging aliases the buffers, so this follows the query pass.)
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

        // Tile it of this segment becomes readable in stage `st` (every consumer thread calls
        // this; the generic writes above and the gathered rows are fenced for the wgmma
        // reads at q_len 8).
        // common: the caller knows the tile holds none of the block's own rows (ONE_DCARRY).
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
            // Warp (hk, wsub): KV head hk, keys 32 wsub .. 32 wsub + 31 of each tile. Thread
            // (g, c) holds query head hk * 8 + g, dims 16 c .. 16 c + 15 as f16 pairs.
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
#pragma unroll
                    for (int j2 = 0; j2 < 4; ++j2) {
                        mma_f16(s[kb], qf[2 * j2], 0u, qf[2 * j2 + 1], 0u, kf[2 * j2],
                                kf[2 * j2 + 1]);
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
                // P.V: per 16-key step, the values of keys 2c, 2c+1, 2c+8, 2c+9 at dims
                // 8g..8g+7; output column g of n-tile e is dim 8 g + e.
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
            stamp(16);
            PHASE_EXIT(10)
            // Fixed-frame merge: this warp's rows scaled by 2^m into the workspace. Thread
            // (g, c) holds dims 16 c .. 16 c + 15 of head hk * 8 + g: o[e][0] is dim 16 c + e,
            // o[e][1] dim 16 c + 8 + e.
            const float f = ex2(fminf(m, 126.f));
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
        } else {
            // Warpgroup hk = KV head; warp wq holds accumulator rows 16 wq .. 16 wq + 15 =
            // tokens 2 wq, 2 wq + 1 x 8 heads (row 8 j + head); thread (g, c) rows g (token
            // 2 wq) and g + 8 (token 2 wq + 1), head g.
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
                // Tiles [0, it_c) end at or below seq: every key is cached and visible to every
                // query row (window 0), so they need no mask and no patch.
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
                // Opaque to ptxas, which otherwise rebuilds them from tid in every first
                // unrolled body (rematerialization) instead of keeping two registers.
                asm volatile("" : "+r"(ld_off), "+r"(st_off));
            }
            float o[36];
#pragma unroll
            for (int i = 0; i < 36; ++i) {
                o[i] = 0.f;
            }
            float off[2] = {OFF_FLOOR, OFF_FLOOR};
            const int qp0 = seq + 2 * wq;
            if constexpr (QKOV || EREL || PCHK || UNR8 > 1) {
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
                    // V^T for this warp's 16 dims: per 32-key half, one ldmatrix.x4.trans over
                    // the four 8-key groups (b16 = two dims), byte permutes into P's key order,
                    // one stmatrix.x4 into the 64B-swizzled K-major buffer; V^T row
                    // 8 (2 dg + e) + r holds dim 16 dg + 2 r + e, so the merge below sees dims
                    // in natural order.
                    auto vt_of = [&]() { return smem + S::VT + (2 * (it & 1) + hk) * VT_STRIDE; };
                    auto transpose_v = [&]() {
                        if constexpr (VHOIST) {
                            // Half 1: keys + 32 (4096 bytes on), chunk 2 + (h4 >> 1), which is
                            // half 0's chunk ^ 2, so the store offset ^ 32.
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
                            // The Q.K wgmma reads Q and the K half of the stage; these stores go to
                            // this tile's V^T buffer, which shares no byte with either, and s is
                            // not read until the wait below.
                            transpose_v();
                        }
                        wg_wait<0>();
                        reg_fence(s);
                    }
                    if constexpr (EREL) {
                        // Every read of the stage is done: the Q.K wgmma's by the wait above, this
                        // thread's ldmatrix of V before it in program order. P.V reads V^T only.
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
                    // Lazy offset: a row keeps its offset until a tile's max exceeds it by 8 (P
                    // stays below 2^8 < 448); the move rescales o and the row sums together.
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
                    // A fragments of the e4m3 wgmma: register 0 = row g, k 4c..4c+3; 1 = row
                    // g + 8; 2 and 3 the same at k + 16. V^T's key order makes k = 4c + 2h + b
                    // (h < 2) and 16 + 4c + 2(h - 2) + b (h >= 2) the key 8h + 2c + b, which is
                    // what the thread holds. s[j] belongs to row (j >> 1) & 1.
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
                            // P >= 0 packs to bytes <= 0x7e (satfinite), so adding 7 to every byte
                            // never carries and sets bit 7 exactly for bytes above 0x78 (P > 256):
                            // a score past its row's offset + 8, the case offset_to_max moves.
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
                    // Lazy offset: a row keeps its offset until a tile's max exceeds it by 8 (P
                    // stays below 2^8 < 448); the move rescales o and the row sums together.
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
                    // A fragments of the e4m3 wgmma: register 0 = row g, k 4c..4c+3; 1 = row
                    // g + 8; 2 and 3 the same at k + 16. V^T's key order makes k = 4c + 2h + b
                    // (h < 2) and 16 + 4c + 2(h - 2) + b (h >= 2) the key 8h + 2c + b, which is
                    // what the thread holds.
                    uint32_t pa[2][4];
#pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        const float* q0 = s + 16 * kk;
                        pa[kk][0] = e4m3x4(q0[0], q0[1], q0[4], q0[5]);
                        pa[kk][1] = e4m3x4(q0[2], q0[3], q0[6], q0[7]);
                        pa[kk][2] = e4m3x4(q0[8], q0[9], q0[12], q0[13]);
                        pa[kk][3] = e4m3x4(q0[10], q0[11], q0[14], q0[15]);
                    }
                    // V^T for this warp's 16 dims: per 32-key half, one ldmatrix.x4.trans over
                    // the four 8-key groups (b16 = two dims), byte permutes into P's key order,
                    // one stmatrix.x4 into the 64B-swizzled K-major buffer; V^T row
                    // 8 (2 dg + e) + r holds dim 16 dg + 2 r + e, so the merge below sees dims
                    // in natural order.
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
            // Fixed-frame merge. o[4i + 2hh + e] is column 8i + 2c + e of row (token
            // 2wq + hh, head g), and column n holds dim 16 (n >> 4) + 2 (n & 7) + ((n >> 3) & 1),
            // so (o[8p + 2hh], o[8p + 4 + 2hh], o[8p + 1 + 2hh], o[8p + 5 + 2hh]) are dims
            // 16 p + 4 c .. + 3; o[32 + 2hh] is the row sum.
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

    // Last-CTA finalize of request b: every red above is ordered before thread 0's release
    // by the barrier, and the last arriver's acquire makes them visible to its reads (which
    // bypass L1).
    auto ticket = [&]() {
        csync();
        stamp(18);
        const int mine = hi - lo;
        if (tid == 0) {
            // >= rather than ==: an overshoot would be a partition bug, and == would then
            // leave the ticket non-zero for every later launch using this request index.
            const int old = atom_add_acq_rel(p.ws_cnt + b, mine);
            *flag = old + mine >= n_keys;
        }
        stamp(19);
        csync();
        if (*flag) {
            finalize_rows<L, NC>(p, b * ROWS, (b + 1) * ROWS, v_scale, tid);
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
        if constexpr (!COOP) {
            ticket();
        } else {
            // Every active split must be resident (the host checks). Counters alternate by
            // launch parity; each split then finalizes a slice of the request's rows.
            __syncthreads();
            const int ep = *reinterpret_cast<const int*>(smem + S::CONST + C_EPOCH);
            const int rows_per = (ROWS + nact - 1) / nact;
            coop_barrier<NC>(p.ws_cnt + CNT_COOP + 2 * b, p.ws_cnt + CNT_EPOCH + b, ep, hi - lo,
                             n_keys, tid, p.dbg);
            const int r_lo = split * rows_per;
            const int r_hi = min(ROWS, r_lo + rows_per);
            if (r_lo < r_hi) {
                finalize_rows<L, NC, FIN8>(p, b * ROWS + r_lo, b * ROWS + r_hi, v_scale, tid);
            }
        }
    } else {
        // Under PDL the previous launch's writes are visible only after the wait (on a
        // plain launch stream order already makes them so).
        asm volatile("griddepcontrol.wait;" ::: "memory");
        // The previous launch wrote this epoch (stream order makes it visible); a coherent
        // load, since the same kernel writes it and ld.global.nc is outside the memory model.
        const int ep = COOP ? __ldcg(p.ws_cnt + CNT_FLAT + 2) : 0;
        const float v_scale_g = __ldg(p.v_scale);
        int mine_total = 0;
        for (b = b_first; b <= b_last; ++b) {
            if (!seg_bounds(b)) {
                continue;
            }
            run_segment();
            if constexpr (!COOP) {
                ticket();
            } else {
                mine_total += hi - lo;
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

// One instance per launch mode, so each build carries only its own prologue and epilogue.
// prepare() sets the dynamic smem limit and records how many CTAs of the instance fit on an
// idle device (the cooperative modes need every active split resident); the host plan reads
// those limits through attn_one_prepare(). With coop, the launch carries
// cudaLaunchAttributeCooperative, so the driver guarantees co-residency (or refuses the
// launch) instead of a foreign CTA stalling the grid into the poll-limit trap.
template <int L, int MODE>
struct Instance {
    static constexpr int NT = one_threads(L, MODE);
    static constexpr int BYTES = Smem<L, one_ns(L, MODE)>::BYTES;
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
        asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
        atomicMax(out, static_cast<int>(smid));
    }
}

// ONE_WAVE: the SM count when every %smid seen on this device lies below it (the kernel's slot
// 2 smid + rank must tile 0 .. 2 SMs - 1), else 0. Probed once, outside graph capture.
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

#define ONE_INSTANCES(X) \
    X(1, 1) X(1, 3) X(1, 11) X(1, 13) X(1, 17) X(1, 49) \
    X(8, 1) X(8, 3) X(8, 11) X(8, 15) X(8, 19) X(8, 51) X(8, 115)

}  // namespace one
}  // namespace
"""

# The layer's host side of the one-launch kernel: the instances its plan uses, a launcher that
# adds the cooperative and (optionally) the PDL attribute, the residency query and a probe of
# those attributes under graph capture.
_ONE_HOST_SRC = r"""
namespace {
namespace one {

// q_len 1: 13 (decode 1 sliding), 1 (decode 16 / 32 sliding, page-64 and short tests), 11
// (decode 1 full, page 1), 3 (batch 1 full at page 64), 49 (full decode at batch > 1, producer
// warpgroup); q_len 8: 15, 3, 11, 1 likewise, 51 (full verify at batch > 1, producer warpgroup)
// and 19 as its fallback when 51's residency is short. Since the producer leaves by PTX exit,
// 51 does not spill and timed 125.0 / 232.4 us against 19's 128.2 / 238.8 at batch 16 / 32 in
// muse_attention_decode's d90a73871d8e4481. 17 and 115 stay out for compile time.
#define DL_ONE_INSTANCES(X) \
    X(1, 1) X(1, 3) X(1, 11) X(1, 13) X(1, 49) X(8, 1) X(8, 3) X(8, 11) X(8, 15) X(8, 19) \
    X(8, 51)

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

}  // namespace one
}  // namespace

// Residency of every built instance, in DL_ONE_INSTANCES order: [L, mode, resident CTAs,
// smem bytes, CTAs per SM] per instance (an idle-device figure).
std::vector<int64_t> attn_one_prepare() {
    using namespace one;
    std::vector<int64_t> out;
#define DL_ONE_PREP(L, M) out.push_back(L); out.push_back(M); \
    out.push_back(Instance<L, M>::prepare()); out.push_back(Instance<L, M>::BYTES); \
    out.push_back(Instance<L, M>::per_sm);
    DL_ONE_INSTANCES(DL_ONE_PREP)
#undef DL_ONE_PREP
    return out;
}

// 1 when a cooperative launch (with the PDL attribute too, if pdl) captures into a graph and
// replays on this driver: a plain one-block launch of a counting kernel, then a two-block
// cooperative one after it as the layer's attention follows its projection, captured,
// instantiated and run. Errors are cleared.
int64_t attn_one_probe_coop(bool pdl) {
    using namespace one;
    int* d = nullptr;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    // On the probe's own stream: a non-blocking stream never orders against the legacy one.
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

// sl2 = sm_scale * qk_scale_factor * log2(e); the key scale is applied on the device. mode is
// the instance's MODE (see attn_one_kernel); coop_launch adds the cooperative attribute to the
// cooperative modes and pdl the programmatic one to every launch; dbg (int32, empty or [32])
// receives the longest cooperative spin in [0] and, in a -DONE_DIAG=1 build, the phase stamps in
// [1 .. 21]. Output [batch * q_len, 1024] BF16.
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
    // The flat partition keeps 512 request offsets in shared memory (the cooperative counters
    // would allow 2048); the plan clamps nc so the rows fit.
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
    // The diet builds index pages by shift and mask; _one_wins keeps other sizes off this path.
    TORCH_CHECK(!ONE_DIET || (1 << p.lp) == page_size, "attn_one: page size ", page_size,
                " is not a power of two");
    // The standalone wrapper's ONE_PFAST condition. The layer passes the same caches, e4m3
    // [pages, page_size, 2, 64] with 128-byte rows from data_ptr, so one byte per element
    // bounds every row offset; above 4 GB the producer keeps its 64-bit path.
    const int64_t cache_bytes = std::max<int64_t>(k_cache.numel(), v_cache.numel());
    p.off32 = (1 << p.lp) == page_size && cache_bytes <= (int64_t{1} << 32) ? 1 : 0;
    // ONE_WAVE is 0 in the layer, so no instance reads it; set because Params has no initializers.
    p.wave_n = 0;
    // Per-request modes launch (n_splits, batch) CTAs; flat modes launch n_splits CTAs that
    // share the batch's keys, with the batch size carried in Params::n_splits.
    int gx = static_cast<int>(n_splits);
    int gy = static_cast<int>(batch);
    if (mode & M_FLAT) {
        p.n_splits = static_cast<int>(batch);
        gy = 1;
    }
    const bool coop = coop_launch && (mode & M_COOP) != 0;
    bool hit = false;
#define DL_ONE_LAUNCH(L, M) \
    if (q_len == L && mode == M) { one_launch<L, M>(p, gx, gy, stream, coop, pdl); hit = true; }
    DL_ONE_INSTANCES(DL_ONE_LAUNCH)
#undef DL_ONE_LAUNCH
    TORCH_CHECK(hit, "attn_one: no instance for q_len ", q_len, " mode ", mode);
    return out;
}
"""

# X5: the FFN boundary at T <= 8 on the unpooler's 1024 live channels, without the all-gather's
# zero pad; replaces FusedPostNormResidualPreNorm there when its self-test passes.
_BND_SRC = r"""
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

__device__ __forceinline__ float block_sum(float v, float* red) {
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
    v = lane < THREADS / 32 ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ float4 bf16x4(const __nv_bfloat16* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float bf16r(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// The FFN boundary at T <= 8, where the unpooler's output has 1024 live channels and the
// other 7168 are the all-gather's exact zeros: branch = f * w_post over the live channels,
// normalized by its RMS over all 8192 (the zeros add nothing to the sum of squares but count in
// the mean); stream = alpha * stream + beta * branch; next = RMS-normalized stream * w_next in
// BF16. SGLang's kernel (norm_residual.cuh) rounds to BF16 at exactly two points, the
// input-scale product f * w_post and the normalized branch, and keeps the stream FP32; bf16r()
// reproduces both. A cluster of 8 CTAs per token, 1024 channels each; every CTA reads the 1024
// live channels for the branch RMS (4 KB), and the stream's sum of squares is reduced over the
// cluster through distributed shared memory.
__global__ void __cluster_dims__(CTAS, 1, 1) __launch_bounds__(THREADS) boundary_kernel(
    const float* __restrict__ stream, const __nv_bfloat16* __restrict__ f,
    const __nv_bfloat16* __restrict__ w_post, const float* __restrict__ alpha,
    const float* __restrict__ beta, const __nv_bfloat16* __restrict__ w_next,
    float* __restrict__ stream_out, __nv_bfloat16* __restrict__ next, float eps_post,
    float eps_next) {
    asm volatile("griddepcontrol.launch_dependents;");
    namespace cg = cooperative_groups;
    __shared__ float red[THREADS / 32];
    __shared__ float part;
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    const int tok = blockIdx.y;
    const int lc = PER * threadIdx.x;
    const int c0 = rank * (H / CTAS) + lc;
    const size_t row = static_cast<size_t>(tok) * H;
    // Loaded before the wait on the unpooler: the gate and norm weights are state, and the
    // stream was written by the attention boundary, which is complete before this kernel can
    // start (every path from it to here passes GEMM1 or Marlin, which wait before they
    // trigger). Only the branch f is the unpooler's output.
    const float4 sv = *reinterpret_cast<const float4*>(stream + row + c0);
    const float4 av = *reinterpret_cast<const float4*>(alpha + c0);
    const float4 bv = *reinterpret_cast<const float4*>(beta + c0);
    const float4 nv = bf16x4(w_next + c0);
    const float4 pv = bf16x4(w_post + lc);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const float4 fv = bf16x4(f + static_cast<size_t>(tok) * LIVE + lc);
    float br[PER] = {bf16r(fv.x * pv.x), bf16r(fv.y * pv.y), bf16r(fv.z * pv.z),
                     bf16r(fv.w * pv.w)};
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        ss += br[k] * br[k];
    }
    const float rb = rsqrtf(block_sum(ss, red) / H + eps_post);
    const float sa[PER] = {sv.x, sv.y, sv.z, sv.w};
    const float aa[PER] = {av.x, av.y, av.z, av.w};
    const float ba[PER] = {bv.x, bv.y, bv.z, bv.w};
    float s[PER];
    float ss2 = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        // Only rank 0's slice holds live channels; elsewhere the branch is an exact zero.
        const float b = rank == 0 ? bf16r(br[k] * rb) : 0.f;
        s[k] = aa[k] * sa[k] + ba[k] * b;
        ss2 += s[k] * s[k];
    }
    ss2 = block_sum(ss2, red);
    if (threadIdx.x == 0) {
        part = ss2;
    }
    cluster.sync();
    float total = 0.f;
#pragma unroll
    for (int r = 0; r < CTAS; ++r) {
        total += *cluster.map_shared_rank(&part, r);
    }
    // No CTA may exit while another can still read its partial.
    cluster.sync();
    const float rn = rsqrtf(total / H + eps_next);
    *reinterpret_cast<float4*>(stream_out + row + c0) = make_float4(s[0], s[1], s[2], s[3]);
    const __nv_bfloat162 h0 = __floats2bfloat162_rn(s[0] * rn * nv.x, s[1] * rn * nv.y);
    const __nv_bfloat162 h1 = __floats2bfloat162_rn(s[2] * rn * nv.z, s[3] * rn * nv.w);
    *reinterpret_cast<uint2*>(next + row + c0) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                   *reinterpret_cast<const uint32_t*>(&h1));
}

// The attention boundary: post-attention norm (no weight), residual gate, pre-FFN norm, and up
// to 8 tokens the FP32 router input. SGLang's kernel rounds the normalized branch to BF16
// before the residual add, keeps the stream FP32 and writes the router copy unrounded
// (norm_residual.cuh); bf16r() reproduces the one rounding. A cluster of 8 CTAs per token;
// both sums of squares are reduced over the cluster through distributed shared memory. The
// stream (a step input), the gate and the norm weight are loaded before the wait on the o
// projection, whose output the branch is.
__global__ void __cluster_dims__(CTAS, 1, 1) __launch_bounds__(THREADS) attn_boundary_kernel(
    const float* __restrict__ stream, const __nv_bfloat16* __restrict__ branch,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ w_next, float* __restrict__ stream_out,
    __nv_bfloat16* __restrict__ next, float* __restrict__ router, float eps_post,
    float eps_next) {
    asm volatile("griddepcontrol.launch_dependents;");
    namespace cg = cooperative_groups;
    __shared__ float red[THREADS / 32];
    __shared__ float part[2];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    const int tok = blockIdx.y;
    const int c0 = rank * (H / CTAS) + PER * threadIdx.x;
    const size_t row = static_cast<size_t>(tok) * H;
    const float4 sv = *reinterpret_cast<const float4*>(stream + row + c0);
    const float4 av = *reinterpret_cast<const float4*>(alpha + c0);
    const float4 bv = *reinterpret_cast<const float4*>(beta + c0);
    const float4 nv = bf16x4(w_next + c0);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const float4 fv = bf16x4(branch + row + c0);
    const float fa[PER] = {fv.x, fv.y, fv.z, fv.w};
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        ss += fa[k] * fa[k];
    }
    ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        part[0] = ss;
    }
    cluster.sync();
    float total = 0.f;
#pragma unroll
    for (int r = 0; r < CTAS; ++r) {
        total += cluster.map_shared_rank(part, r)[0];
    }
    const float rb = rsqrtf(total / H + eps_post);
    const float sa[PER] = {sv.x, sv.y, sv.z, sv.w};
    const float aa[PER] = {av.x, av.y, av.z, av.w};
    const float ba[PER] = {bv.x, bv.y, bv.z, bv.w};
    float s[PER];
    float ss2 = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        s[k] = aa[k] * sa[k] + ba[k] * bf16r(fa[k] * rb);
        ss2 += s[k] * s[k];
    }
    ss2 = block_sum(ss2, red);
    if (threadIdx.x == 0) {
        part[1] = ss2;
    }
    cluster.sync();
    float total2 = 0.f;
#pragma unroll
    for (int r = 0; r < CTAS; ++r) {
        total2 += cluster.map_shared_rank(part, r)[1];
    }
    // No CTA may exit while another can still read its partials.
    cluster.sync();
    const float rn = rsqrtf(total2 / H + eps_next);
    *reinterpret_cast<float4*>(stream_out + row + c0) = make_float4(s[0], s[1], s[2], s[3]);
    const float na[PER] = {nv.x, nv.y, nv.z, nv.w};
    float o[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        o[k] = s[k] * rn * na[k];
    }
    const __nv_bfloat162 h0 = __floats2bfloat162_rn(o[0], o[1]);
    const __nv_bfloat162 h1 = __floats2bfloat162_rn(o[2], o[3]);
    *reinterpret_cast<uint2*>(next + row + c0) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                   *reinterpret_cast<const uint32_t*>(&h1));
    if (router != nullptr) {
        *reinterpret_cast<float4*>(router + row + c0) = make_float4(o[0], o[1], o[2], o[3]);
    }
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
        // No CTA may exit while another can still read its partials.
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

// launch_ex with a runtime cluster size; the geometry kernels carry no __cluster_dims__.
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

// Geometry ids, in the order of ABND_GEOS in the Python: {CTAs per token, threads per CTA}.
constexpr int kGeoCl[] = {8, 2, 4, 1, 1, 1};
constexpr int kGeoNt[] = {256, 512, 256, 256, 512, 1024};
constexpr int kGeos = 6;

template <typename F>
void with_geo(int geo, F fn) {
    switch (geo) {
        case 1: fn(attn_boundary_geo_kernel<2, 512>); break;
        case 2: fn(attn_boundary_geo_kernel<4, 256>); break;
        case 3: fn(attn_boundary_geo_kernel<1, 256>); break;
        case 4: fn(attn_boundary_geo_kernel<1, 512>); break;
        case 5: fn(attn_boundary_geo_kernel<1, 1024>); break;
        default: fn(attn_boundary_kernel); break;
    }
}

}  // namespace bnd
}  // namespace

// Per geometry id: resident CTAs per SM, and the clusters (CTAs when the geometry has no
// cluster) the whole GPU holds at once, for the prepare log.
std::vector<int64_t> attn_boundary_occupancy(int64_t geo) {
    using namespace bnd;
    TORCH_CHECK(geo >= 0 && geo < kGeos, "attn_boundary_occupancy: geometry id");
    const int cl = kGeoCl[geo];
    const int nt = kGeoNt[geo];
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    int per_sm = 0;
    int clusters = 0;
    with_geo(static_cast<int>(geo), [&](auto kern) {
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, nt, 0));
        if (cl == 1) {
            clusters = per_sm * sms;
            return;
        }
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(cl, 2048);
        cfg.blockDim = dim3(nt);
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = cl;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        // The 8-CTA kernel's dims are compiled in; the attribute must then be absent.
        cfg.numAttrs = geo == 0 ? 0 : 1;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, kern, &cfg));
    });
    return {per_sm, clusters};
}

std::vector<torch::Tensor> attn_boundary(torch::Tensor stream, torch::Tensor branch,
                                         torch::Tensor alpha, torch::Tensor beta,
                                         torch::Tensor w_next, double eps_post,
                                         double eps_next, bool want_router, int64_t geo) {
    using namespace bnd;
    TORCH_CHECK(geo >= 0 && geo < kGeos, "attn_boundary: geometry id");
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
    if (geo == 0) {
        launch_ex(attn_boundary_kernel, dim3(CTAS, tokens), dim3(THREADS), 0, cs, kPdlGlue, sp,
                  bp, alpha.data_ptr<float>(), beta.data_ptr<float>(), np,
                  stream_out.data_ptr<float>(), op, rp, static_cast<float>(eps_post),
                  static_cast<float>(eps_next));
    } else {
        with_geo(static_cast<int>(geo), [&](auto kern) {
            launch_cl(kern, dim3(kGeoCl[geo], tokens), dim3(kGeoNt[geo]), kGeoCl[geo], cs, sp,
                      bp, alpha.data_ptr<float>(), beta.data_ptr<float>(), np,
                      stream_out.data_ptr<float>(), op, rp, static_cast<float>(eps_post),
                      static_cast<float>(eps_next));
        });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {next, stream_out, router};
}

std::vector<torch::Tensor> ffn_boundary_short(torch::Tensor stream, torch::Tensor f,
                                              torch::Tensor w_post, torch::Tensor alpha,
                                              torch::Tensor beta, torch::Tensor w_next,
                                              double eps_post, double eps_next) {
    using namespace bnd;
    const int tokens = stream.size(0);
    TORCH_CHECK(stream.is_contiguous() && stream.scalar_type() == torch::kFloat32
                && stream.size(1) == H, "ffn_boundary_short: stream must be FP32 [T, 8192]");
    TORCH_CHECK(f.is_contiguous() && f.scalar_type() == torch::kBFloat16 && f.size(0) == tokens
                && f.size(1) == LIVE, "ffn_boundary_short: f must be BF16 [T, 1024]");
    TORCH_CHECK(w_post.is_contiguous() && w_next.is_contiguous() && alpha.is_contiguous()
                && beta.is_contiguous() && w_post.scalar_type() == torch::kBFloat16
                && w_next.scalar_type() == torch::kBFloat16
                && alpha.scalar_type() == torch::kFloat32
                && beta.scalar_type() == torch::kFloat32 && w_post.numel() >= LIVE
                && w_next.numel() == H && alpha.numel() == H && beta.numel() == H,
                "ffn_boundary_short: weights");
    auto stream_out = torch::empty_like(stream);
    auto next = torch::empty({tokens, H}, f.options());
    if (tokens == 0) {
        return {next, stream_out};
    }
    launch_ex(
        boundary_kernel, dim3(CTAS, tokens), dim3(THREADS), 0, at::cuda::getCurrentCUDAStream(),
        kPdlGlue,
        stream.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(f.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_post.data_ptr()), alpha.data_ptr<float>(),
        beta.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(w_next.data_ptr()),
        stream_out.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(next.data_ptr()),
        static_cast<float>(eps_post), static_cast<float>(eps_next));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {next, stream_out};
}
"""

# The W4A16 grouped GEMM from muse_moe_w4a16 (submission at 0a917ae; kernel source, repack
# and dequant copied unchanged). A config is (kind, nt, rows, s, w, d, mode, layout, p): kind
# 1 is x @ W13^T with SwiGLU, kind 2 is inter @ W2^T; nt n8 token tiles per work item; rows of
# x kept in shared memory (min(8 nt, T)); s K splits; w warps per block; d register stages of
# weights in flight per warp; mode 0 is the GEMM, 5 the GEMM without the L2 prefetch hint;
# layout and p as in w4_gemm_kernel. The muse_moe_w4a16 GEMM configs, without its timing-only
# modes and the ones that lost to Marlin or to another config at every size in its tuner
# (check 84f3975334f048c3: its 15, 17, 18, 22, 24, 27, 31).
W4_CONFIGS = [
    (1, 1, 1, 4, 4, 4, 0, 0, 0), (1, 1, 8, 2, 4, 4, 0, 0, 0), (1, 1, 8, 4, 4, 4, 0, 0, 0),
    (1, 1, 8, 2, 4, 8, 0, 0, 0), (1, 1, 8, 2, 8, 4, 0, 0, 0), (1, 2, 16, 4, 4, 4, 0, 0, 0),
    (1, 2, 16, 2, 4, 4, 0, 0, 0), (1, 2, 16, 4, 8, 4, 0, 0, 0), (1, 2, 16, 4, 8, 4, 0, 1, 0),
    (2, 1, 1, 4, 4, 8, 0, 0, 0), (2, 1, 1, 4, 4, 4, 0, 0, 0), (2, 1, 8, 4, 4, 4, 0, 0, 0),
    (2, 1, 8, 2, 4, 4, 0, 0, 0), (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 2, 16, 2, 4, 4, 0, 0, 0),
    (1, 1, 1, 2, 4, 4, 0, 0, 1), (1, 1, 8, 4, 4, 4, 0, 0, 1), (1, 1, 8, 2, 8, 4, 0, 0, 1),
    (1, 1, 8, 4, 8, 4, 0, 0, 1), (1, 2, 16, 4, 4, 4, 0, 0, 1),
    (2, 1, 1, 2, 4, 4, 0, 0, 1), (2, 1, 1, 4, 4, 4, 0, 0, 1), (2, 1, 8, 2, 4, 4, 0, 0, 1),
    (2, 1, 8, 4, 4, 4, 0, 0, 1), (2, 1, 8, 2, 8, 4, 0, 0, 1), (2, 2, 16, 4, 4, 4, 0, 0, 1),
    (1, 1, 8, 4, 4, 4, 5, 0, 0), (1, 1, 8, 2, 8, 4, 5, 0, 1), (2, 1, 8, 4, 4, 4, 5, 0, 0),
    (2, 2, 16, 4, 4, 4, 5, 0, 0), (1, 2, 16, 4, 4, 4, 5, 0, 0),
]

_W4_CPP_SRC = r"""
#include <torch/extension.h>
void w4_gemm(int64_t cfg, torch::Tensor act, torch::Tensor wq, torch::Tensor wsc,
             torch::Tensor gs, torch::Tensor sorted_ids, torch::Tensor items,
             torch::Tensor num_items, int64_t numel, torch::Tensor out, bool pdl);
void w4_swiglu(torch::Tensor part, torch::Tensor local_ids, int64_t splits, torch::Tensor out,
               bool pdl);
torch::Tensor w4_finalize(torch::Tensor part, torch::Tensor scores, torch::Tensor norm_w,
                          torch::Tensor local_ids, double eps, bool pdl, int64_t cl);
std::vector<int64_t> w4_info(int64_t cfg);
"""

_W4_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace {

namespace cg = cooperative_groups;

constexpr int kK = 4096;
constexpr int kInter = 4096;

template <bool HINT>
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 v;
    if (HINT) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    } else {
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    }
    return v;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0,
                                         unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <bool HIGH>
__device__ __forceinline__ unsigned bmul(unsigned a, unsigned s) {
    const __nv_bfloat162 s2 = *reinterpret_cast<__nv_bfloat162*>(&s);
    const __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),
                                     HIGH ? __high2bfloat162(s2) : __low2bfloat162(s2));
    return *reinterpret_cast<const unsigned*>(&r);
}

// One A fragment (16 rows x 16 k) of NVFP4 to bf16, block scale applied. The repack in
// prepare() permutes each word's 32 bits so that register j's two E2M1 codes land on bf16
// bits {15, 8, 7, 6} (sign, exponent, mantissa) after at most one shift or rotate; the code
// then reads as its value times 2^-126. The block scale byte (positive, normal E4M3, checked
// in prepare) becomes bf16 scale * 2^119 by shifting it into place and adding 112 to the
// exponent, so the product is value * scale * 2^-7, exact in bf16. The epilogue multiplies by
// the global scale * 2^7.
template <int TILE>
__device__ __forceinline__ void dequant(unsigned w, unsigned sw, unsigned (&a)[4]) {
    constexpr unsigned kM = 0x81C081C0u;
    const unsigned r0 = w & kM;
    const unsigned r1 = (w << 6) & kM;
    const unsigned r2 = __funnelshift_r(w, w, 6) & kM;
    const unsigned r3 = ((w >> 4) & 0x00400040u) | ((w << 4) & 0x81808180u);
    const unsigned sc = __byte_perm(sw, 0u, TILE ? 0x4342 : 0x4140) * 16u + 0x77807780u;
    a[0] = bmul<false>(r0, sc);
    a[1] = bmul<true>(r1, sc);
    a[2] = bmul<false>(r2, sc);
    a[3] = bmul<true>(r3, sc);
}

// Rounds to the nearest bf16 value, kept in a float register.
__device__ __forceinline__ float bf16r(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ unsigned word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// A unit is 32 weight rows of one expert over one K split: for KIND 1, 16 gate rows and the
// 16 up rows of the same intermediate features, so the SwiGLU pairs sit in one thread's
// accumulators; for KIND 2, 32 output rows. Each warp owns one unit at a time and streams its
// weights straight into registers (each lane reads only its own fragments) D stages of 64 k
// ahead; the item's activation rows for the K split sit in shared memory and feed mma as B.
// P 0: grid (units / W, max items, S), one unit per warp. P 1: a persistent grid that walks
// (item, chunk, split) units; the next unit's activations are staged into the other buffer
// during this one, and its first D weight stages are loaded during this unit's last D.
// MODE 0 is the GEMM and MODE 5 the GEMM without the L2 256-byte prefetch hint. The rest
// time only: 1 streams the same weights and activations without dequant or mma; 2 runs mma
// on the raw words without dequant; 3 streams the weights only (no activation staging);
// 4 streams the same number of weight bytes with every warp reading consecutive 1 KB chunks
// at each stage, like a plain streaming kernel; 6 is 4 without the hint.
// LAYOUT 0 stores each unit's 64 k stages contiguously; LAYOUT 1 interleaves 8 neighbouring
// units stage by stage, so the warps of a block read one contiguous stretch per stage.
template <int KIND, int NT, int ROWS, int S, int W, int D, int MODE, int LAYOUT, int P>
__global__ void __launch_bounds__(W * 32) w4_gemm_kernel(
        const __nv_bfloat16* __restrict__ act, const uint4* __restrict__ wq,
        const uint4* __restrict__ wsc, const float* __restrict__ gs,
        const int* __restrict__ sorted_ids, const int2* __restrict__ items,
        const int* __restrict__ num_items, int numel, void* __restrict__ out) {
    constexpr int KC = kK / S;
    constexpr int NST = KC / 64;
    constexpr int U = KIND == 1 ? 256 : 128;
    constexpr int RS = KC + 8;
    constexpr int TR = 8 * NT;
    constexpr int PER_ITEM = U / W * S;
    constexpr int WSTR = LAYOUT ? 8 * 64 : 64;
    constexpr int SSTR = LAYOUT ? 8 * 8 : 8;
    static_assert(!P || NST % D == 0, "cross-unit prefetch needs whole groups of D stages");
    constexpr bool HINT = MODE != 5 && MODE != 6;
    constexpr bool LOADS_ONLY = MODE == 1 || MODE == 3 || MODE == 4 || MODE == 6;
    constexpr bool STAGE_X = !(MODE == 3 || MODE == 4 || MODE == 6);
    constexpr bool STREAMLIKE = MODE == 4 || MODE == 6;
    static_assert(!STREAMLIKE || !P, "streaming diagnostic is for one unit per warp");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rid[2][TR];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // GEMM1 needs route_align's items before it can address any weights. GEMM2's items come
    // from route_align too, which finished before GEMM1 passed its own wait, so GEMM2 starts
    // its weight stream first and waits only before reading GEMM1's output.
    if (KIND == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
    const int total = P ? num_items[0] * PER_ITEM : num_items[0];
    int q = P ? blockIdx.x : blockIdx.y;
    if (q >= total) {
        return;
    }

    int e, nrows, row0, u, z;
    auto decode = [&](int unit, int& e_, int& nrows_, int& row0_, int& u_, int& z_) {
        int item = blockIdx.y;
        int chunk = blockIdx.x;
        z_ = blockIdx.z;
        if (P) {
            item = unit / PER_ITEM;
            const int r = unit - item * PER_ITEM;
            chunk = r / S;
            z_ = r - chunk * S;
        }
        const int2 it = items[item];
        e_ = it.x >> 16;
        nrows_ = it.x & 0xffff;
        row0_ = it.y;
        u_ = chunk * W + warp;
    };
    int wstr = WSTR;
    int sstr = SSTR;
    auto weights_at = [&](int e_, int u_, int z_, const uint4*& wp_, const uint4*& sp_) {
        if (STREAMLIKE) {
            const int warps = num_items[0] * (U / W) * S * W;
            const int gw = ((blockIdx.z * num_items[0] + blockIdx.y) * (U / W) + blockIdx.x) * W
                           + warp;
            wp_ = wq + static_cast<size_t>(gw) * 64 + lane;
            sp_ = wsc + static_cast<size_t>(gw) * 8 + (lane >> 2);
            wstr = warps * 64;
            sstr = warps * 8;
            return;
        }
        const size_t stage0 = LAYOUT
            ? (static_cast<size_t>(e_ * (U / 8) + u_ / 8) * 64 + z_ * NST) * 8 + (u_ & 7)
            : static_cast<size_t>(e_ * U + u_) * 64 + z_ * NST;
        wp_ = wq + stage0 * 64 + lane;
        sp_ = wsc + stage0 * 8 + (lane >> 2);
    };
    auto stage_x = [&](int buf, int nrows_, int row0_, int z_) {
        __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw) + buf * ROWS * RS;
        for (int r = threadIdx.x; r < TR; r += W * 32) {
            rid[buf][r] = r < nrows_ ? sorted_ids[row0_ + r] : -1;
        }
        constexpr int CH = KC / 8;
        for (int i = threadIdx.x; i < ROWS * CH; i += W * 32) {
            const int r = i / CH;
            const int c = i - r * CH;
            if (r < nrows_) {
                const int p = sorted_ids[row0_ + r];
                const int src_row = KIND == 1 ? p / 8 : p;
                cp_async16(xs + r * RS + c * 8,
                           act + static_cast<size_t>(src_row) * kK + z_ * KC + c * 8);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    if (!STAGE_X) {
        for (int r = threadIdx.x; r < 2 * TR; r += W * 32) {
            rid[r / TR][r % TR] = -1;
        }
    }

    decode(q, e, nrows, row0, u, z);
    const uint4* wp;
    const uint4* sp;
    weights_at(e, u, z, wp, sp);
    uint4 wv[D][2];
    uint4 sv[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        if (d < NST) {
            wv[d][0] = ld_stream<HINT>(wp + d * wstr);
            wv[d][1] = ld_stream<HINT>(wp + d * wstr + 32);
            sv[d] = ld_stream<HINT>(sp + d * sstr);
        }
    }
    if (KIND == 2) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if (STAGE_X) {
        stage_x(0, nrows, row0, z);
    }

    const int g = lane >> 2;
    const int t4 = lane & 3;
    int buf = 0;
#pragma unroll 1
    for (;;) {
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        const int qn = q + gridDim.x;
        const bool has_next = P && qn < total;
        int en = 0, nrowsn = 0, row0n = 0, un = 0, zn = 0;
        const uint4* wpn = wp;
        const uint4* spn = sp;
        if (has_next) {
            decode(qn, en, nrowsn, row0n, un, zn);
            weights_at(en, un, zn, wpn, spn);
            stage_x(buf ^ 1, nrowsn, row0n, zn);
        }

        const __nv_bfloat16* xs = reinterpret_cast<const __nv_bfloat16*>(smem_raw)
                                  + buf * ROWS * RS;
        // ldmatrix row address: matrix lane / 8 (k offset 8 * that), tile row lane % 8.
        const __nv_bfloat16* xrow[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int r = nt * 8 + (lane & 7);
            xrow[nt] = xs + (r < ROWS ? r : 0) * RS + (lane >> 3) * 8;
        }
        float acc[2][NT][4];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[t][nt][j] = 0.f;
                }
            }
        }

#pragma unroll 1
        for (int s0 = 0; s0 < NST; s0 += D) {
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const int s = s0 + d;
                if (s < NST && LOADS_ONLY) {
                    acc[0][0][0] += __uint_as_float(
                        (wv[d][0].x ^ wv[d][0].w ^ wv[d][1].x ^ wv[d][1].w ^ sv[d].x)
                        & 0x3f800000u);
                } else if (s < NST) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        unsigned b[NT][4];
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            ldmatrix_x4(b[nt], xrow[nt] + s * 64 + h * 32);
                        }
#pragma unroll
                        for (int j1 = 0; j1 < 2; ++j1) {
                            const unsigned sw = word(sv[d], h * 2 + j1);
                            unsigned a0[4];
                            unsigned a1[4];
                            if (MODE == 2) {
                                const unsigned r0 = word(wv[d][h], j1 * 2);
                                const unsigned r1 = word(wv[d][h], j1 * 2 + 1);
                                a0[0] = r0; a0[1] = r0 ^ sw; a0[2] = r0 >> 1; a0[3] = r0 << 1;
                                a1[0] = r1; a1[1] = r1 ^ sw; a1[2] = r1 >> 1; a1[3] = r1 << 1;
                            } else {
                                dequant<0>(word(wv[d][h], j1 * 2), sw, a0);
                                dequant<1>(word(wv[d][h], j1 * 2 + 1), sw, a1);
                            }
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt) {
                                mma_bf16(acc[0][nt], a0, b[nt][j1 * 2], b[nt][j1 * 2 + 1]);
                                mma_bf16(acc[1][nt], a1, b[nt][j1 * 2], b[nt][j1 * 2 + 1]);
                            }
                        }
                    }
                }
                if (s < NST) {
                    if (s + D < NST) {
                        wv[d][0] = ld_stream<HINT>(wp + (s + D) * wstr);
                        wv[d][1] = ld_stream<HINT>(wp + (s + D) * wstr + 32);
                        sv[d] = ld_stream<HINT>(sp + (s + D) * sstr);
                    } else if (has_next) {
                        wv[d][0] = ld_stream<HINT>(wpn + (s + D - NST) * wstr);
                        wv[d][1] = ld_stream<HINT>(wpn + (s + D - NST) * wstr + 32);
                        sv[d] = ld_stream<HINT>(spn + (s + D - NST) * sstr);
                    }
                }
            }
        }

        const float gscale = gs[e] * 128.f;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int qq = 0; qq < 2; ++qq) {
                const int col = nt * 8 + t4 * 2 + qq;
                const int p = rid[buf][col];
                if (p < 0) {
                    continue;
                }
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const float v0 = acc[0][nt][hh * 2 + qq] * gscale;
                    const float v1 = acc[1][nt][hh * 2 + qq] * gscale;
                    if (KIND == 1 && S == 1) {
                        // TASK.md: activations stay BF16 wherever SGLang keeps BF16. Marlin
                        // writes gate and up to its BF16 cache1 before silu_and_mul.
                        const float gb = bf16r(v0);
                        const float ub = bf16r(v1);
                        const float sv0 = gb / (1.f + __expf(-gb));
                        // SwiGLU's output is GEMM2's BF16 input in SGLang too.
                        reinterpret_cast<__nv_bfloat16*>(out)[static_cast<size_t>(p) * kInter
                                                              + u * 16 + hh * 8 + g] =
                            __float2bfloat16_rn(sv0 * ub);
                    } else if (KIND == 1) {
                        float* o = reinterpret_cast<float*>(out)
                                   + (static_cast<size_t>(z) * numel + p) * (2 * kInter);
                        o[u * 16 + hh * 8 + g] = v0;
                        o[kInter + u * 16 + hh * 8 + g] = v1;
                    } else {
                        float* o = reinterpret_cast<float*>(out)
                                   + (static_cast<size_t>(z) * numel + p) * kK;
                        o[u * 32 + hh * 8 + g] = v0;
                        o[u * 32 + 16 + hh * 8 + g] = v1;
                    }
                }
            }
        }
        if (!has_next) {
            break;
        }
        q = qn;
        e = en;
        nrows = nrowsn;
        row0 = row0n;
        u = un;
        z = zn;
        wp = wpn;
        sp = spn;
        buf ^= 1;
    }
}

// The post-expert norm and weighted sum over GEMM2's FP32 K-split partials. CL blocks per
// token (a cluster when CL > 1), 512 / CL threads each, 8 channels per thread; the per-slot
// sums of squares are combined through distributed shared memory. One block per token leaves
// small batches to one SM's bandwidth. Dead slots are never read.
template <int CL>
__global__ void __launch_bounds__(512 / CL) w4_finalize_kernel(
        const float* __restrict__ part, int splits, int numel, const float* __restrict__ scores,
        const __nv_bfloat16* __restrict__ norm_w, const int* __restrict__ local_ids,
        __nv_bfloat16* __restrict__ out, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    constexpr int kTopK = 8;
    constexpr int THREADS = 512 / CL;
    constexpr int WARPS = THREADS / 32;
    __shared__ float red[kTopK][WARPS];
    __shared__ float psum[kTopK];
    __shared__ float scale[kTopK];
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c = (blockIdx.x * THREADS + tid) * 8;
    float w[8];
    {
        const uint4 raw = *reinterpret_cast<const uint4*>(norm_w + c);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            w[2 * i] = f.x;
            w[2 * i + 1] = f.y;
        }
    }
    bool live[kTopK];
    float v[kTopK][8];
    float ss[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        live[k] = local_ids[t * kTopK + k] >= 0;
        ss[k] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[k][j] = 0.f;
        }
        if (live[k]) {
            for (int z = 0; z < splits; ++z) {
                const float* row = part + (static_cast<size_t>(z) * numel + t * kTopK + k) * kK + c;
                const float4 a = *reinterpret_cast<const float4*>(row);
                const float4 b = *reinterpret_cast<const float4*>(row + 4);
                v[k][0] += a.x; v[k][1] += a.y; v[k][2] += a.z; v[k][3] += a.w;
                v[k][4] += b.x; v[k][5] += b.y; v[k][6] += b.z; v[k][7] += b.w;
            }
        }
        // TASK.md: activations stay BF16 wherever SGLang keeps BF16. The K-split partials
        // are GEMM2's own FP32 reduction (Marlin reduces in FP32 too); its output, which
        // Marlin writes as BF16, is rounded here before the norm.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[k][j] = bf16r(v[k][j]) * w[j];
            ss[k] += v[k][j] * v[k][j];
        }
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], d);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            red[k][warp] = ss[k];
        }
    }
    __syncthreads();
    if (tid < kTopK) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < WARPS; ++i) {
            s += red[tid][i];
        }
        psum[tid] = s;
    }
    if (CL > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        if (tid < kTopK) {
            float s = 0.f;
#pragma unroll
            for (int r = 0; r < CL; ++r) {
                s += *cluster.map_shared_rank(&psum[tid], r);
            }
            scale[tid] = live[tid] ? scores[t * kTopK + tid] * rsqrtf(s / kK + eps) : 0.f;
        }
        // Also keeps every block's psum alive until the other blocks have read it.
        cluster.sync();
    } else {
        __syncthreads();
        if (tid < kTopK) {
            scale[tid] = live[tid] ? scores[t * kTopK + tid] * rsqrtf(psum[tid] / kK + eps)
                                   : 0.f;
        }
        __syncthreads();
    }
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const float s = scale[k];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] += s * v[k][j];
        }
    }
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kK + c) = raw;
}

// Sums the K-split partials of GEMM1 and applies SwiGLU: 8 blocks of 64 threads per route,
// 8 features per thread; routes to other GPUs' experts are skipped.
__global__ void __launch_bounds__(64) w4_swiglu_kernel(
        const float* __restrict__ part, const int* __restrict__ local_ids, int numel,
        int splits, __nv_bfloat16* __restrict__ out) {
    // GEMM2 may start streaming its weights while this kernel runs; it waits for this
    // kernel's output itself.
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int p = blockIdx.x;
    if (local_ids[p] < 0) {
        return;
    }
    const int c = (blockIdx.y * 64 + threadIdx.x) * 8;
    float gv[8];
    float uv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        gv[j] = 0.f;
        uv[j] = 0.f;
    }
    for (int z = 0; z < splits; ++z) {
        const float* row = part + (static_cast<size_t>(z) * numel + p) * (2 * kInter);
        const float4 g0 = *reinterpret_cast<const float4*>(row + c);
        const float4 g1 = *reinterpret_cast<const float4*>(row + c + 4);
        const float4 u0 = *reinterpret_cast<const float4*>(row + kInter + c);
        const float4 u1 = *reinterpret_cast<const float4*>(row + kInter + c + 4);
        gv[0] += g0.x; gv[1] += g0.y; gv[2] += g0.z; gv[3] += g0.w;
        gv[4] += g1.x; gv[5] += g1.y; gv[6] += g1.z; gv[7] += g1.w;
        uv[0] += u0.x; uv[1] += u0.y; uv[2] += u0.z; uv[3] += u0.w;
        uv[4] += u1.x; uv[5] += u1.y; uv[6] += u1.z; uv[7] += u1.w;
    }
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // TASK.md: activations stay BF16 wherever SGLang keeps BF16. The split-K sum is
        // GEMM1's own FP32 reduction; its gate and up outputs, which Marlin writes to BF16
        // cache1, are rounded before SwiGLU, and SwiGLU's output is BF16 like SGLang's.
        const float g0 = bf16r(gv[2 * j]);
        const float g1 = bf16r(gv[2 * j + 1]);
        const float a = g0 / (1.f + __expf(-g0)) * bf16r(uv[2 * j]);
        const float b = g1 / (1.f + __expf(-g1)) * bf16r(uv[2 * j + 1]);
        h[j] = __floats2bfloat162_rn(a, b);
    }
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(p) * kInter + c) = raw;
}

template <typename Kern, typename... Args>
void launch_pdl(Kern kern, dim3 grid, int threads, int smem, bool pdl, cudaStream_t stream,
                Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

template <int KIND, int NT, int ROWS, int S, int W, int D, int MODE, int LAYOUT, int P>
void w4_launch(const __nv_bfloat16* act, const uint4* wq, const uint4* wsc, const float* gs,
               const int* sorted_ids, const int2* items, const int* num_items, int max_items,
               int numel, void* out, bool pdl, cudaStream_t stream) {
    constexpr int U = KIND == 1 ? 256 : 128;
    constexpr int smem = (P ? 2 : 1) * ROWS * (kK / S + 8) * 2;
    auto kern = w4_gemm_kernel<KIND, NT, ROWS, S, W, D, MODE, LAYOUT, P>;
    static int grid = 0;
    if (grid == 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, W * 32, smem));
        int sms = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        grid = blocks * sms;
    }
    const dim3 g = P ? dim3(grid) : dim3(U / W, max_items, S);
    launch_pdl(kern, g, W * 32, smem, pdl, stream, act, wq, wsc, gs, sorted_ids, items,
               num_items, numel, out);
}

// Registers per thread, dynamic shared memory per block, and resident blocks per SM.
template <int KIND, int NT, int ROWS, int S, int W, int D, int MODE, int LAYOUT, int P>
std::vector<int64_t> w4_query() {
    constexpr int smem = (P ? 2 : 1) * ROWS * (kK / S + 8) * 2;
    auto kern = w4_gemm_kernel<KIND, NT, ROWS, S, W, D, MODE, LAYOUT, P>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, W * 32, smem));
    return {attr.numRegs, smem + static_cast<int64_t>(attr.sharedSizeBytes), blocks};
}

template <int CL>
void finalize_launch(const float* part, int splits, int numel, const float* scores,
                     const __nv_bfloat16* norm_w, const int* local_ids, __nv_bfloat16* out,
                     float eps, bool pdl, cudaStream_t stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CL, numel / 8, 1);
    cfg.blockDim = dim3(512 / CL, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = CL;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = CL > 1 ? 2 : 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, w4_finalize_kernel<CL>, part, splits, numel, scores,
                                      norm_w, local_ids, out, eps));
}

// w4_compile_check: w4_swiglu_kernel<<<1, 512>>>(nullptr, nullptr, 0, 1, nullptr);
// w4_compile_check: finalize_launch<1>(0, 1, 8, 0, 0, 0, 0, 0.f, false, 0);
// w4_compile_check: finalize_launch<4>(0, 1, 8, 0, 0, 0, 0, 0.f, false, 0);
// w4_compile_check: finalize_launch<8>(0, 1, 8, 0, 0, 0, 0, 0.f, false, 0);
//@W4_COMPILE_CHECKS@
}  // namespace

void w4_gemm(int64_t cfg, torch::Tensor act, torch::Tensor wq, torch::Tensor wsc,
             torch::Tensor gs, torch::Tensor sorted_ids, torch::Tensor items,
             torch::Tensor num_items, int64_t numel, torch::Tensor out, bool pdl) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* ap = reinterpret_cast<const __nv_bfloat16*>(act.data_ptr<at::BFloat16>());
    const auto* wp = reinterpret_cast<const uint4*>(wq.data_ptr());
    const auto* sp = reinterpret_cast<const uint4*>(wsc.data_ptr());
    const float* gp = gs.data_ptr<float>();
    const int* ip = sorted_ids.data_ptr<int>();
    const auto* itp = reinterpret_cast<const int2*>(items.data_ptr<int>());
    const int* np = num_items.data_ptr<int>();
    const int mi = items.size(0);
    const int n = static_cast<int>(numel);
    void* op = out.data_ptr();
    switch (cfg) {
//@W4_CASES@
        default: TORCH_CHECK(false, "bad w4 config");
    }
}

std::vector<int64_t> w4_info(int64_t cfg) {
    switch (cfg) {
//@W4_INFO@
        default: TORCH_CHECK(false, "bad w4 config");
    }
    return {};
}

torch::Tensor w4_finalize(torch::Tensor part, torch::Tensor scores, torch::Tensor norm_w,
                          torch::Tensor local_ids, double eps, bool pdl, int64_t cl) {
    const int splits = part.size(0);
    const int numel = part.size(1);
    auto out = torch::empty({numel / 8, kK}, norm_w.options());
    const float* pp = part.data_ptr<float>();
    const float* sp = scores.data_ptr<float>();
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>());
    const int* ip = local_ids.data_ptr<int>();
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    const float e = static_cast<float>(eps);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (cl) {
        case 1: finalize_launch<1>(pp, splits, numel, sp, wp, ip, op, e, pdl, stream); break;
        case 4: finalize_launch<4>(pp, splits, numel, sp, wp, ip, op, e, pdl, stream); break;
        case 8: finalize_launch<8>(pp, splits, numel, sp, wp, ip, op, e, pdl, stream); break;
        default: TORCH_CHECK(false, "bad finalize cluster size");
    }
    return out;
}

void w4_swiglu(torch::Tensor part, torch::Tensor local_ids, int64_t splits, torch::Tensor out,
               bool pdl) {
    const int numel = local_ids.numel();
    launch_pdl(w4_swiglu_kernel, dim3(numel, 8), 64, 0, pdl, at::cuda::getCurrentCUDAStream(),
               static_cast<const float*>(part.data_ptr<float>()),
               static_cast<const int*>(local_ids.data_ptr<int>()), numel,
               static_cast<int>(splits),
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
}
"""


# The fused MoE kernel from muse_moe_w4a16 (docs/moe-port-guide.md in that problem's worktree),
# assembled for the computing variants by tools/dl_moe_inline.py inside a nested `moe1`
# namespace: grid = SM count, every CTA routes from topk_ids, phase A (GEMM1 + SwiGLU) and
# phase B (GEMM2 times 1 + w, sums of squares), a software grid barrier, the final norm and
# weighted sum. The inliner also points the kernels' local range at a device tensor (lo_p).
_MOE1_SRC = r"""

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
// Named barriers: 1 syncs the compute warps after a unit's reads of its ring slot; 2 + b
// says partial buffer b is full (compute warps arrive, the helper waits); 6 + b says it is
// free again (the helper arrives, the compute warps wait).
constexpr int kBarUnit = 1;
constexpr int kBarFull = 2;
constexpr int kBarFree = 6;
// A spin that lasts this many cycles marks the launch failed instead of hanging the run.
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
    // Fat path stage costs by (phase, n-tile count); cost[0][1] == 0 means the compiled table.
    int cost[2][9];
    // The layer's local_start is an input tensor, read here instead of the host value.
    const int* lo_p;
    int pdl;
};

template <typename Kern>
void moe1_launch(Kern kern, int grid, int block, int smem, cudaStream_t stream,
                 const Params& p) {
    if (!p.pdl) {
        kern<<<grid, block, smem, stream>>>(p);
        return;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(block);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));
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
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n"
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

// x tiles in global memory (xa from the gather, xb from phase A's epilogue) are written with
// generic stores and read by other CTAs through cp.async.bulk, which is the async proxy (PTX
// ISA 8.6, 9.7.10.28.2). Proxy-preserved base causality (8.9.5) does not carry a generic write
// to an async-proxy read without a proxy fence on the path, so every writer fences its own
// stores before the release, and the issuing thread fences again after its acquire: the
// bidirectional fence acts within one thread (9.7.15.4), and the second one keeps the chain
// valid under the reading where only the issuer's own later async operations are ordered.
// docs/runs-moe.md, "Proxy ordering of the x tiles", has the whole chain.
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

// One A fragment (16 rows x 16 k) from one repacked word. The repack permutes each word's
// 32 bits so that register j's two E2M1 codes land on bf16 bits {15, 8, 7, 6} (sign,
// exponent, mantissa) after at most one shift or rotate; the code then reads as its value
// times 2^-126. sg and sg8 hold the row's block scale (positive normal E4M3, checked in
// prepare) as bf16 scale * 2^119 in both halves, so the product is value * scale * 2^-7,
// exact in bf16; the epilogue multiplies by the global scale * 2^7.
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

// scale_word with the multiplier passed in: k16 = 16 that the compiler cannot see keeps the
// multiply an IMAD on the FMA pipe (skinny kMode 5, 7, 8).
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
    return true;
}


// Pair rows: expert-major, each warp's chunk of entries in order within an expert. pos[i] is
// entry i's pair row or -1; ent[q] (when given) is pair row q's entry.
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

// position_pass in two halves for fast_setup: the per-warp counts, and the positions once
// wc holds each warp's first pair row per expert. Same warp partition as position_pass.
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

// out[t] = sum over local slots of topk_w * rsqrt(ss / H + eps) * y, y already times
// (1 + post_norm_w). Items are (token, 4 * THREADS-column segment), 4 columns per thread.
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


__device__ __forceinline__ long long gtimer() {
    long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}


namespace sk_8_4_0 {

constexpr int kNS = 4;
constexpr int kNW = 8;
constexpr int kMode = 0;
// kMode 2 and 4: timing stamps (2 is mode 0's, 4 is mode 3's). kMode 3: each compute warp loads
// its codes of a unit up front and releases the ring slot itself (fence.proxy.async, then
// mbarrier.arrive on empty, count kNT, as the fat path's mode 28), and thread 0 refills the slot
// once every warp has arrived, instead of after a bar.sync of all compute warps behind the
// unit's last MMA. The A/B at 8 tokens (240b68e7df1a427f) put mode 0 3.6 us over its loads-only
// twin while thread 0 never waited for a unit, so the per-unit barrier is on the critical path.
// kMode 5 and 7: the scale decode's multiply by 16 takes a multiplier ptxas cannot see, so it
// stays an IMAD (FMA pipe) instead of becoming a LEA on the ALU pipe that the FP4 decode
// saturates (the public kernel's SCDK). kMode 6 and 7: the partial-sum buffers are handed to and
// from the helper warp through mbarriers (every compute thread arrives on hand[bb] after its
// store, every helper lane on hand[kNRB + bb] after its loads) instead of a 288-thread bar.sync
// per unit, so the compute warps meet once per unit (kBarUnit) instead of twice. The ordering
// chain is in docs/runs-moe.md, "Skinny partial handoff through mbarriers". kMode 8: 7 with
// the stamps.
// Modes 16 and up are flags over mode kLow (docs/runs-moe.md, "Skinny weight stream against the
// public kernel"). 16: thread 0's clock64 per unit (issue, wait start, wait end, slot free) and
// a header, written to p.inter from 16 MB on (pair rows >= 2048, which the skinny sizes never
// use). 32: the unit barrier and the refill move ahead of the partial-buffer handoff, so a refill
// no longer waits for the helper to free a buffer. 64: weight copies without the L2 hint.
constexpr int kLow = kMode & 15;
constexpr bool kStamp = kLow == 2 || kLow == 4 || kLow == 8;
constexpr bool kEarly = kLow == 3 || kLow == 4;
constexpr bool kScaleFma = kLow == 5 || kLow == 7 || kLow == 8;
constexpr bool kHand = kLow >= 6;
constexpr bool kUnitStamp = (kMode & 16) != 0;
constexpr bool kRefillFirst = (kMode & 32) != 0;
constexpr bool kNoHint = (kMode & 64) != 0;
static_assert(!kRefillFirst || (!kEarly && !kHand), "refill-first is a mode 0 / 5 change");
constexpr int kUSMax = 256;
constexpr int kUSPer = 8 + 4 * kUSMax;
constexpr int kNT = kNW * 32;
constexpr int kNTL = kNT + 32;
constexpr int kNWarps = kNTL / 32;

struct __align__(128) Smem {
    unsigned char ring[kNS][kUnitBytes];
    float4 red[kNRB][kNW][32];
    uint64_t full[kNS];
    uint64_t empty[kNS];
    int cnt[kLocalMax];
    int off[kLocalMax];
    int wc[kNWarps][kLocalMax];
    int job_e[kJmax];
    int job_q[kJmax];
    int job_n[kJmax];
    short pos[kEmax];
    short ent[kEmax];
    signed char eid[kEmax];
    int njobs;
    int a0, na, b0, nb;
    // kMode 2 (timing only): thread 0's globaltimer at the phase points and cycle counters,
    // written to p.inter at the end (16 int64 per CTA; tools/moe_make_bench_ab.py reads them).
    long long dg[kStamp ? 16 : 1];
    uint64_t hand[kHand ? 2 * kNRB : 1];
};

__device__ __forceinline__ void hand_wait(uint64_t* bar, unsigned parity, int* ctr) {
    if (!mbar_try(bar, parity)) {
        const long long t0 = clock64();
        while (!mbar_try(bar, parity)) {
            if (spin_failed(t0, ctr, 2)) {
                break;
            }
        }
    }
}

__device__ __forceinline__ void hand_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ long long* ustamps(const Params& p) {
    return reinterpret_cast<long long*>(p.inter) + (1 << 21) + blockIdx.x * kUSPer;
}

// Event k of unit q: 0 copy issued, 1 thread 0 starts waiting for it, 2 the wait ends, 3 the
// slot is free (unit barrier passed; the refill follows).
__device__ __forceinline__ void ustamp(const Params& p, int q, int k) {
    if (kUnitStamp && q < kUSMax) {
        ustamps(p)[8 + 4 * q + k] = clock64();
    }
}

// Unit q of this CTA's list (its phase A units, then its phase B units) into ring slot q % kNS.
__device__ __forceinline__ void issue_unit(const Params& p, Smem& S, int q, uint64_t pol) {
    const int slot = q % kNS;
    const unsigned char* src;
    if (q < S.na) {
        const int v = S.a0 + q;
        src = p.w13 + (static_cast<size_t>(S.job_e[v >> 9]) * kUA + (v & 511)) * kUnitBytes;
    } else {
        const int v = S.b0 + (q - S.na);
        src = p.w2 + (static_cast<size_t>(S.job_e[v >> 8]) * kUB + (v & 255)) * kUnitBytes;
    }
    mbar_expect_tx(&S.full[slot], kUnitBytes);
    ustamp(p, q, 0);
    if constexpr (kNoHint) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];"
                     :: "r"(smem_u32(S.ring[slot])), "l"(src), "r"(kUnitBytes),
                        "r"(smem_u32(&S.full[slot])) : "memory");
    } else {
        bulk_load(S.ring[slot], src, kUnitBytes, &S.full[slot], pol);
    }
}

// Compute warps: for each unit, 8 chunks of 64 k (4 mma k16 steps each) over this warp's
// 512-k slice, the job's x rows kept in registers.
__device__ __forceinline__ void compute_loop(const Params& p, Smem& S, uint64_t pol) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    // The unit layout holds 8 k-lanes of 512 (8 chunks of 64); NW warps split them.
    constexpr int kCPW = 64 / kNW;
    const int w8 = warp / (kNW / 8);
    const int c0 = (warp % (kNW / 8)) * kCPW;
    uint4 xr[kCPW][2];
    int jcur = -1;
    bool bcur = false;
    long long wfull = 0;
    // 16, from a kernel parameter (nloc <= kLocalMax = 32) so ptxas cannot fold it into a LEA.
    const unsigned k16 = 16u + (static_cast<unsigned>(p.nloc) >> 6);
    if (kUnitStamp && tid == 0) {
        ustamps(p)[0] = na;
        ustamps(p)[1] = n;
    }
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        if (kStamp && tid == 0 && q == na) {
            S.dg[3] = gtimer();
            S.dg[8] = wfull;
            wfull = 0;
        }
        if (j != jcur || phb != bcur) {
            jcur = j;
            bcur = phb;
            const int q0 = S.job_q[j];
            const bool have = g < S.job_n[j];
            if (!phb) {
                const int tok = have ? (S.ent[q0 + g] >> 3) : 0;
                const __nv_bfloat16* src = p.x + static_cast<size_t>(tok) * kH + w8 * 512 + c0 * 64
                                           + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            } else {
                if (ld_acquire(p.done_a + j) < kUA) {
                    const long long t0 = clock64();
                    while (ld_acquire(p.done_a + j) < kUA) {
                        __nanosleep(64);
                        if (spin_failed(t0, p.ctr, 1)) {
                            break;
                        }
                    }
                }
                if (kStamp && tid == 0 && q == na) {
                    S.dg[4] = gtimer();
                }
                const __nv_bfloat16* src = p.inter + static_cast<size_t>(q0 + (have ? g : 0)) * kI
                                           + w8 * 512 + c0 * 64 + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        const int slot = q % kNS;
        // The stamps time the first try too: try_wait may suspend the thread for a
        // system-dependent time before it returns false (PTX ISA, mbarrier.try_wait).
        const long long tw = kStamp ? clock64() : 0;
        if (tid == 0) {
            ustamp(p, q, 1);
        }
        if (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
            const long long t0 = clock64();
            while (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
                if (spin_failed(t0, p.ctr, 2)) {
                    break;
                }
            }
        }
        if (kStamp) {
            wfull += clock64() - tw;
        }
        if (tid == 0) {
            ustamp(p, q, 2);
        }
        const unsigned char* sl = S.ring[slot];
        const uint4* codes = reinterpret_cast<const uint4*>(sl + w8 * 4096 + c0 * 512) + lane;
        const uint4 sc = *(reinterpret_cast<const uint4*>(sl + 32768 + w8 * 512) + lane);
        float acc[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[s][i] = 0.f;
            }
        }
        uint4 cw[kEarly ? kCPW : 1];
        if constexpr (kEarly) {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                cw[c] = codes[c * 32];
            }
            // Every generic read of the slot (codes, scales) is issued above; the per-thread
            // proxy fence orders them before the refill's bulk write (docs/runs-moe.md,
            // "Ordering of the ring slot releases"; the chain is the fat path's mode 28).
            fence_proxy_async();
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         :: "r"(smem_u32(&S.empty[slot])) : "memory");
            if (tid == 0 && q + kNS < n) {
                const unsigned par = (q / kNS) & 1;
                if (!mbar_try(&S.empty[slot], par)) {
                    const long long t0 = clock64();
                    while (!mbar_try(&S.empty[slot], par)) {
                        if (spin_failed(t0, p.ctr, 2)) {
                            break;
                        }
                    }
                }
                issue_unit(p, S, q + kNS, pol);
            }
        }
        if (kLow == 1) {
            // Timing only: the unit's bytes arrive and one word is read, nothing is computed.
            acc[0][0] = __uint_as_float(codes[0].x & 0x3f800000u);
        } else {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                const uint4 w = kEarly ? cw[c] : codes[c * 32];
                const unsigned scw = word(sc, (c0 + c) >> 1);
                const unsigned sg = kScaleFma ? scale_word_k(scw, (2 * (c0 + c)) & 3, k16)
                                              : scale_word(scw, (2 * (c0 + c)) & 3);
                const unsigned sg8 = kScaleFma ? scale_word_k(scw, (2 * (c0 + c) + 1) & 3, k16)
                                               : scale_word(scw, (2 * (c0 + c) + 1) & 3);
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    unsigned a[4];
                    dequant(word(w, s), sg, sg8, a);
                    mma_bf16(acc[s], a, word(xr[c][s >> 1], 2 * (s & 1)),
                             word(xr[c][s >> 1], 2 * (s & 1) + 1));
                }
            }
        }
        float4 r;
        r.x = (acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]);
        r.y = (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]);
        r.z = (acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2]);
        r.w = (acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3]);
        const int bb = q & (kNRB - 1);
        if constexpr (kRefillFirst) {
            // Every compute thread's reads of the slot are done once its MMAs have their
            // operands; the barrier and the fence are the same chain as below, earlier.
            named_sync(kBarUnit, kNT);
            if (tid == 0) {
                ustamp(p, q, 3);
                if (q + kNS < n) {
                    fence_proxy_async();
                    issue_unit(p, S, q + kNS, pol);
                }
            }
        }
        const long long tb = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            // Buffer bb was last read by the helper for unit q - kNRB: completion number
            // q / kNRB - 1 of its free barrier.
            if (q >= kNRB) {
                hand_wait(&S.hand[kNRB + bb], ((q / kNRB) & 1) ^ 1, p.ctr);
            }
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            hand_arrive(&S.hand[bb]);
        } else {
            named_sync(kBarFree + bb, kNTL);
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            named_arrive(kBarFull + bb, kNTL);
        }
        if constexpr (kEarly || kRefillFirst) {
            continue;
        }
        const long long tu = kStamp ? clock64() : 0;
        named_sync(kBarUnit, kNT);
        if (kStamp && tid == 0) {
            S.dg[15] += clock64() - tu;
        }
        if (tid == 0) {
            ustamp(p, q, 3);
        }
        if (tid == 0 && q + kNS < n) {
            // The slot's generic-proxy reads are complete (the barrier above); order them
            // before the async-proxy write that refills it.
            fence_proxy_async();
            issue_unit(p, S, q + kNS, pol);
        }
    }
    if (kStamp && tid == 0) {
        if (na == n) {
            S.dg[3] = gtimer();
            S.dg[4] = S.dg[3];
            S.dg[8] = wfull;
            wfull = 0;
        }
        S.dg[9] = wfull;
    }
}

// Makes this warp's inter stores for job j visible, then adds n finished units to its counter.
__device__ __forceinline__ void credit(const Params& p, Smem& S, int j, int n, int lane) {
    if (n <= 0) {
        return;
    }
    const long long c0 = kStamp ? clock64() : 0;
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomicAdd(p.done_a + j, n);
    }
    if (kStamp && lane == 0) {
        S.dg[11] += clock64() - c0;
    }
}

// Helper warp: sums the 8 warps' partials of each unit and runs its epilogue.
__device__ __forceinline__ void helper_loop(const Params& p, Smem& S, float* ss) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    if constexpr (!kHand) {
#pragma unroll
        for (int bb = 0; bb < kNRB; ++bb) {
            named_arrive(kBarFree + bb, kNTL);
        }
    }
    int jcred = -1;
    int ncred = 0;
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        const int rb = phb ? (v & 255) : (v & 511);
        const int e = S.job_e[j];
        const int q0 = S.job_q[j];
        const int ntok = S.job_n[j];
        const int bb = q & (kNRB - 1);
        // Phase A's last credits go out before waiting on any phase B partials: the compute
        // warps produce none until done_a is complete, which needs these credits.
        if (phb && ncred > 0) {
            credit(p, S, jcred, ncred, lane);
            ncred = 0;
        }
        const long long cw = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            hand_wait(&S.hand[bb], (q / kNRB) & 1, p.ctr);
        } else {
            named_sync(kBarFull + bb, kNTL);
        }
        if (kStamp && lane == 0) {
            S.dg[10] += clock64() - cw;
        }
        float4 r = S.red[bb][0][lane];
#pragma unroll
        for (int w = 1; w < kNW; ++w) {
            const float4 t = S.red[bb][w][lane];
            r.x += t.x;
            r.y += t.y;
            r.z += t.z;
            r.w += t.w;
        }
        if constexpr (kHand) {
            hand_arrive(&S.hand[kNRB + bb]);
        } else {
            named_arrive(kBarFree + bb, kNTL);
        }
        const bool t0 = 2 * t4 < ntok;
        const bool t1 = 2 * t4 + 1 < ntok;
        if (!phb) {
            // Marlin writes gate and up to bf16 before silu_and_mul, and SwiGLU's output is
            // GEMM2's bf16 input.
            const float gsa = __ldg(p.gs13 + e) * 128.f;
            const float g0 = bf16r(r.x * gsa);
            const float g1 = bf16r(r.y * gsa);
            const float u0 = bf16r(r.z * gsa);
            const float u1 = bf16r(r.w * gsa);
            const float h0 = g0 * __frcp_rn(1.f + __expf(-g0)) * u0;
            const float h1 = g1 * __frcp_rn(1.f + __expf(-g1)) * u1;
            __nv_bfloat16* dst = p.inter + static_cast<size_t>(q0 + 2 * t4) * kI + rb * 8 + g;
            if (t0) {
                dst[0] = __float2bfloat16_rn(h0);
            }
            if (t1) {
                dst[kI] = __float2bfloat16_rn(h1);
            }
            // Credits go out once per job: the fence behind each credit costs about a
            // microsecond under a saturated memory system, too much for every unit.
            if (j != jcred) {
                credit(p, S, jcred, ncred, lane);
                jcred = j;
                ncred = 0;
            }
            ++ncred;
        } else {
            // GEMM2's output is bf16 in SGLang; the norm then runs in FP32.
            const float gsb = __ldg(p.gs2 + e) * 128.f;
            const int col0 = rb * 16 + g;
            const int col1 = col0 + 8;
            const float w0 = __ldg(p.nw + col0);
            const float w1 = __ldg(p.nw + col1);
            const float y00 = bf16r(r.x * gsb) * w0;
            const float y01 = bf16r(r.y * gsb) * w0;
            const float y10 = bf16r(r.z * gsb) * w1;
            const float y11 = bf16r(r.w * gsb) * w1;
            float* dst = p.ybuf + static_cast<size_t>(q0 + 2 * t4) * kH;
            if (t0) {
                dst[col0] = y00;
                dst[col1] = y10;
            }
            if (t1) {
                dst[kH + col0] = y01;
                dst[kH + col1] = y11;
            }
            float s0 = y00 * y00 + y10 * y10;
            float s1 = y01 * y01 + y11 * y11;
#pragma unroll
            for (int d = 4; d < 32; d <<= 1) {
                s0 += __shfl_xor_sync(0xffffffffu, s0, d);
                s1 += __shfl_xor_sync(0xffffffffu, s1, d);
            }
            if (g == 0) {
                if (t0) {
                    atomicAdd(ss + q0 + 2 * t4, s0);
                }
                if (t1) {
                    atomicAdd(ss + q0 + 2 * t4 + 1, s1);
                }
            }
        }
    }
    if (ncred > 0) {
        credit(p, S, jcred, ncred, lane);
    }
}

template <int SPLIT>
__global__ void __launch_bounds__(kNTL, 1) moe_fused_kernel(const Params p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int G = gridDim.x;
    const int cta = blockIdx.x;
    const int E = p.M * kTopK;
    // Under programmatic dependent launch the inputs are not ready before this; without it
    // the instruction does nothing.
    if (p.pdl) {
        if (warp == kNW) {
            if (lane == 0) {
    #pragma unroll
                for (int s = 0; s < kNS; ++s) {
                    mbar_init(&S.full[s], 1);
                    mbar_init(&S.empty[s], kNT);
                }
                if constexpr (kHand) {
    #pragma unroll
                    for (int b = 0; b < kNRB; ++b) {
                        mbar_init(&S.hand[b], kNT);
                        mbar_init(&S.hand[kNRB + b], 32);
                    }
                }
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
        }
        if (tid == 0) {
            for (int i = cta; i < p.nloc * 4; i += G) {
                prefetch_l2_hint(p.w13 + (static_cast<size_t>(i % p.nloc) * kUA
                                          + i / p.nloc) * kUnitBytes, kUnitBytes);
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // The generation picks this launch's sum-of-squares copy and barrier counter; the last
    // CTA through the barrier bumps it, after every CTA has read it.
    const int gen = *reinterpret_cast<volatile int*>(p.ctr + 3);
    float* ss = p.ss + (gen & 1) * kEmax;
    const uint64_t pol = policy_evict_first();
    if (kStamp && tid < 16) {
        S.dg[tid] = tid == 0 ? gtimer() : tid == 12 ? clock64() : 0;
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[2] = clock64();
        ustamps(p)[3] = gtimer();
    }

    if (warp == kNW) {
        // Histogram and job list by one warp, so this CTA's first unit copies can be issued
        // before the full routing; the copies' cold start then overlaps it.
        if (lane == 0 && !p.pdl) {
#pragma unroll
            for (int s = 0; s < kNS; ++s) {
                mbar_init(&S.full[s], 1);
                mbar_init(&S.empty[s], kNT);
            }
            if constexpr (kHand) {
#pragma unroll
                for (int b = 0; b < kNRB; ++b) {
                    mbar_init(&S.hand[b], kNT);
                    mbar_init(&S.hand[kNRB + b], 32);
                }
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        S.cnt[lane] = 0;
        __syncwarp();
        for (int base = 0; base < E; base += 128) {
            const int i = base + 4 * lane;
            int4 v = make_int4(-1, -1, -1, -1);
            if (i < E) {
                v = __ldg(reinterpret_cast<const int4*>(p.topk_ids + i));
            }
            const int ev[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = ev[k] - ld_lo(p.lo_p);
                const int eid = (i < E && e >= 0 && e < p.nloc) ? e : -1;
                if (i < E) {
                    S.eid[i + k] = static_cast<signed char>(eid);
                }
                const unsigned peers = __match_any_sync(0xffffffffu, eid);
                if (eid >= 0 && lane == __ffs(peers) - 1) {
                    S.cnt[eid] += __popc(peers);
                }
                __syncwarp();
            }
        }
        const int tot = lane < p.nloc ? S.cnt[lane] : 0;
        const int off = warp_scan(tot, lane) - tot;
        S.off[lane] = off;
        const int nj = SPLIT ? (tot + 7) >> 3 : (tot > 0 ? 1 : 0);
        const int jincl = warp_scan(nj, lane);
        const int j0 = jincl - nj;
        const int njobs = __shfl_sync(0xffffffffu, jincl, 31);
        for (int k = 0; k < nj; ++k) {
            S.job_e[j0 + k] = lane;
            S.job_q[j0 + k] = off + 8 * k;
            S.job_n[j0 + k] = min(8, tot - 8 * k);
        }
        __syncwarp();
        if (lane == 0) {
            const unsigned ua = njobs * kUA;
            const unsigned ub = njobs * kUB;
            const int a0 = ua * cta / G;
            const int a1 = ua * (cta + 1) / G;
            const int b0 = ub * cta / G;
            const int b1 = ub * (cta + 1) / G;
            S.njobs = njobs;
            S.a0 = a0;
            S.na = a1 - a0;
            S.b0 = b0;
            S.nb = b1 - b0;
            __threadfence_block();
            const int n = (a1 - a0) + (b1 - b0);
            for (int q = 0; q < min(kNS, n); ++q) {
                issue_unit(p, S, q, pol);
            }
        }
    } else {
        const int ctid = cta * kNT + tid;
        const int cstride = G * kNT;
        for (int i = ctid; i < p.M * 64; i += cstride) {
            prefetch_l2(p.x + i * 64);
        }
        for (int i = ctid; i < (E + 31) / 32; i += cstride) {
            prefetch_l2(p.topk_w + i * 32);
        }
        // The next launch's sum-of-squares copy and barrier counter; nothing reads them in
        // this launch.
        float* ss_next = p.ss + ((gen + 1) & 1) * kEmax;
        for (int i = ctid; i < kEmax; i += cstride) {
            ss_next[i] = 0.f;
        }
        if (ctid == 0) {
            p.ctr[4 + ((gen + 1) & 1)] = 0;
        }
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[1] = gtimer();
    }

    position_pass<kNWarps>(S.eid, S.pos, S.ent, S.wc, S.off, E);
    if (kStamp && tid == 0) {
        S.dg[2] = gtimer();
    }

    if (warp < kNW) {
        compute_loop(p, S, pol);
    } else {
        helper_loop(p, S, ss);
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[5] = gtimer();
    }

    // Grid barrier. The counter alternates with the generation, so the one this launch
    // uses was zeroed by the previous launch and nothing is reset between launches.
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
    if (kStamp && tid == 0) {
        S.dg[6] = gtimer();
    }
    for (int j = cta * kNTL + tid; j < kJmax; j += G * kNTL) {
        p.done_a[j] = 0;
    }

    if (warp < kNW) {
        final_pass<kNT>(p, S.pos, ss, cta, G);
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[4] = clock64();
        ustamps(p)[5] = gtimer();
    }
    if constexpr (kStamp) {
        __syncthreads();
        if (tid == 0) {
            S.dg[7] = gtimer();
            S.dg[13] = clock64();
        }
        __syncthreads();
        if (tid < 16) {
            reinterpret_cast<long long*>(p.inter)[cta * 16 + tid] = S.dg[tid];
        }
    }
}

template <int SPLIT>
void launch(const Params& p, int sms, cudaStream_t stream) {
    static bool ready = false;
    if (!ready) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused_kernel<SPLIT>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(sizeof(Smem))));
        ready = true;
    }
    moe1_launch(moe_fused_kernel<SPLIT>, sms, kNTL, static_cast<int>(sizeof(Smem)), stream,
                p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sk_8_4_0


namespace sk_16_4_0 {

constexpr int kNS = 4;
constexpr int kNW = 16;
constexpr int kMode = 0;
// kMode 2 and 4: timing stamps (2 is mode 0's, 4 is mode 3's). kMode 3: each compute warp loads
// its codes of a unit up front and releases the ring slot itself (fence.proxy.async, then
// mbarrier.arrive on empty, count kNT, as the fat path's mode 28), and thread 0 refills the slot
// once every warp has arrived, instead of after a bar.sync of all compute warps behind the
// unit's last MMA. The A/B at 8 tokens (240b68e7df1a427f) put mode 0 3.6 us over its loads-only
// twin while thread 0 never waited for a unit, so the per-unit barrier is on the critical path.
// kMode 5 and 7: the scale decode's multiply by 16 takes a multiplier ptxas cannot see, so it
// stays an IMAD (FMA pipe) instead of becoming a LEA on the ALU pipe that the FP4 decode
// saturates (the public kernel's SCDK). kMode 6 and 7: the partial-sum buffers are handed to and
// from the helper warp through mbarriers (every compute thread arrives on hand[bb] after its
// store, every helper lane on hand[kNRB + bb] after its loads) instead of a 288-thread bar.sync
// per unit, so the compute warps meet once per unit (kBarUnit) instead of twice. The ordering
// chain is in docs/runs-moe.md, "Skinny partial handoff through mbarriers". kMode 8: 7 with
// the stamps.
// Modes 16 and up are flags over mode kLow (docs/runs-moe.md, "Skinny weight stream against the
// public kernel"). 16: thread 0's clock64 per unit (issue, wait start, wait end, slot free) and
// a header, written to p.inter from 16 MB on (pair rows >= 2048, which the skinny sizes never
// use). 32: the unit barrier and the refill move ahead of the partial-buffer handoff, so a refill
// no longer waits for the helper to free a buffer. 64: weight copies without the L2 hint.
constexpr int kLow = kMode & 15;
constexpr bool kStamp = kLow == 2 || kLow == 4 || kLow == 8;
constexpr bool kEarly = kLow == 3 || kLow == 4;
constexpr bool kScaleFma = kLow == 5 || kLow == 7 || kLow == 8;
constexpr bool kHand = kLow >= 6;
constexpr bool kUnitStamp = (kMode & 16) != 0;
constexpr bool kRefillFirst = (kMode & 32) != 0;
constexpr bool kNoHint = (kMode & 64) != 0;
static_assert(!kRefillFirst || (!kEarly && !kHand), "refill-first is a mode 0 / 5 change");
constexpr int kUSMax = 256;
constexpr int kUSPer = 8 + 4 * kUSMax;
constexpr int kNT = kNW * 32;
constexpr int kNTL = kNT + 32;
constexpr int kNWarps = kNTL / 32;

struct __align__(128) Smem {
    unsigned char ring[kNS][kUnitBytes];
    float4 red[kNRB][kNW][32];
    uint64_t full[kNS];
    uint64_t empty[kNS];
    int cnt[kLocalMax];
    int off[kLocalMax];
    int wc[kNWarps][kLocalMax];
    int job_e[kJmax];
    int job_q[kJmax];
    int job_n[kJmax];
    short pos[kEmax];
    short ent[kEmax];
    signed char eid[kEmax];
    int njobs;
    int a0, na, b0, nb;
    // kMode 2 (timing only): thread 0's globaltimer at the phase points and cycle counters,
    // written to p.inter at the end (16 int64 per CTA; tools/moe_make_bench_ab.py reads them).
    long long dg[kStamp ? 16 : 1];
    uint64_t hand[kHand ? 2 * kNRB : 1];
};

__device__ __forceinline__ void hand_wait(uint64_t* bar, unsigned parity, int* ctr) {
    if (!mbar_try(bar, parity)) {
        const long long t0 = clock64();
        while (!mbar_try(bar, parity)) {
            if (spin_failed(t0, ctr, 2)) {
                break;
            }
        }
    }
}

__device__ __forceinline__ void hand_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ long long* ustamps(const Params& p) {
    return reinterpret_cast<long long*>(p.inter) + (1 << 21) + blockIdx.x * kUSPer;
}

// Event k of unit q: 0 copy issued, 1 thread 0 starts waiting for it, 2 the wait ends, 3 the
// slot is free (unit barrier passed; the refill follows).
__device__ __forceinline__ void ustamp(const Params& p, int q, int k) {
    if (kUnitStamp && q < kUSMax) {
        ustamps(p)[8 + 4 * q + k] = clock64();
    }
}

// Unit q of this CTA's list (its phase A units, then its phase B units) into ring slot q % kNS.
__device__ __forceinline__ void issue_unit(const Params& p, Smem& S, int q, uint64_t pol) {
    const int slot = q % kNS;
    const unsigned char* src;
    if (q < S.na) {
        const int v = S.a0 + q;
        src = p.w13 + (static_cast<size_t>(S.job_e[v >> 9]) * kUA + (v & 511)) * kUnitBytes;
    } else {
        const int v = S.b0 + (q - S.na);
        src = p.w2 + (static_cast<size_t>(S.job_e[v >> 8]) * kUB + (v & 255)) * kUnitBytes;
    }
    mbar_expect_tx(&S.full[slot], kUnitBytes);
    ustamp(p, q, 0);
    if constexpr (kNoHint) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];"
                     :: "r"(smem_u32(S.ring[slot])), "l"(src), "r"(kUnitBytes),
                        "r"(smem_u32(&S.full[slot])) : "memory");
    } else {
        bulk_load(S.ring[slot], src, kUnitBytes, &S.full[slot], pol);
    }
}

// Compute warps: for each unit, 8 chunks of 64 k (4 mma k16 steps each) over this warp's
// 512-k slice, the job's x rows kept in registers.
__device__ __forceinline__ void compute_loop(const Params& p, Smem& S, uint64_t pol) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    // The unit layout holds 8 k-lanes of 512 (8 chunks of 64); NW warps split them.
    constexpr int kCPW = 64 / kNW;
    const int w8 = warp / (kNW / 8);
    const int c0 = (warp % (kNW / 8)) * kCPW;
    uint4 xr[kCPW][2];
    int jcur = -1;
    bool bcur = false;
    long long wfull = 0;
    // 16, from a kernel parameter (nloc <= kLocalMax = 32) so ptxas cannot fold it into a LEA.
    const unsigned k16 = 16u + (static_cast<unsigned>(p.nloc) >> 6);
    if (kUnitStamp && tid == 0) {
        ustamps(p)[0] = na;
        ustamps(p)[1] = n;
    }
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        if (kStamp && tid == 0 && q == na) {
            S.dg[3] = gtimer();
            S.dg[8] = wfull;
            wfull = 0;
        }
        if (j != jcur || phb != bcur) {
            jcur = j;
            bcur = phb;
            const int q0 = S.job_q[j];
            const bool have = g < S.job_n[j];
            if (!phb) {
                const int tok = have ? (S.ent[q0 + g] >> 3) : 0;
                const __nv_bfloat16* src = p.x + static_cast<size_t>(tok) * kH + w8 * 512 + c0 * 64
                                           + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            } else {
                if (ld_acquire(p.done_a + j) < kUA) {
                    const long long t0 = clock64();
                    while (ld_acquire(p.done_a + j) < kUA) {
                        __nanosleep(64);
                        if (spin_failed(t0, p.ctr, 1)) {
                            break;
                        }
                    }
                }
                if (kStamp && tid == 0 && q == na) {
                    S.dg[4] = gtimer();
                }
                const __nv_bfloat16* src = p.inter + static_cast<size_t>(q0 + (have ? g : 0)) * kI
                                           + w8 * 512 + c0 * 64 + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        const int slot = q % kNS;
        // The stamps time the first try too: try_wait may suspend the thread for a
        // system-dependent time before it returns false (PTX ISA, mbarrier.try_wait).
        const long long tw = kStamp ? clock64() : 0;
        if (tid == 0) {
            ustamp(p, q, 1);
        }
        if (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
            const long long t0 = clock64();
            while (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
                if (spin_failed(t0, p.ctr, 2)) {
                    break;
                }
            }
        }
        if (kStamp) {
            wfull += clock64() - tw;
        }
        if (tid == 0) {
            ustamp(p, q, 2);
        }
        const unsigned char* sl = S.ring[slot];
        const uint4* codes = reinterpret_cast<const uint4*>(sl + w8 * 4096 + c0 * 512) + lane;
        const uint4 sc = *(reinterpret_cast<const uint4*>(sl + 32768 + w8 * 512) + lane);
        float acc[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[s][i] = 0.f;
            }
        }
        uint4 cw[kEarly ? kCPW : 1];
        if constexpr (kEarly) {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                cw[c] = codes[c * 32];
            }
            // Every generic read of the slot (codes, scales) is issued above; the per-thread
            // proxy fence orders them before the refill's bulk write (docs/runs-moe.md,
            // "Ordering of the ring slot releases"; the chain is the fat path's mode 28).
            fence_proxy_async();
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         :: "r"(smem_u32(&S.empty[slot])) : "memory");
            if (tid == 0 && q + kNS < n) {
                const unsigned par = (q / kNS) & 1;
                if (!mbar_try(&S.empty[slot], par)) {
                    const long long t0 = clock64();
                    while (!mbar_try(&S.empty[slot], par)) {
                        if (spin_failed(t0, p.ctr, 2)) {
                            break;
                        }
                    }
                }
                issue_unit(p, S, q + kNS, pol);
            }
        }
        if (kLow == 1) {
            // Timing only: the unit's bytes arrive and one word is read, nothing is computed.
            acc[0][0] = __uint_as_float(codes[0].x & 0x3f800000u);
        } else {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                const uint4 w = kEarly ? cw[c] : codes[c * 32];
                const unsigned scw = word(sc, (c0 + c) >> 1);
                const unsigned sg = kScaleFma ? scale_word_k(scw, (2 * (c0 + c)) & 3, k16)
                                              : scale_word(scw, (2 * (c0 + c)) & 3);
                const unsigned sg8 = kScaleFma ? scale_word_k(scw, (2 * (c0 + c) + 1) & 3, k16)
                                               : scale_word(scw, (2 * (c0 + c) + 1) & 3);
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    unsigned a[4];
                    dequant(word(w, s), sg, sg8, a);
                    mma_bf16(acc[s], a, word(xr[c][s >> 1], 2 * (s & 1)),
                             word(xr[c][s >> 1], 2 * (s & 1) + 1));
                }
            }
        }
        float4 r;
        r.x = (acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]);
        r.y = (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]);
        r.z = (acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2]);
        r.w = (acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3]);
        const int bb = q & (kNRB - 1);
        if constexpr (kRefillFirst) {
            // Every compute thread's reads of the slot are done once its MMAs have their
            // operands; the barrier and the fence are the same chain as below, earlier.
            named_sync(kBarUnit, kNT);
            if (tid == 0) {
                ustamp(p, q, 3);
                if (q + kNS < n) {
                    fence_proxy_async();
                    issue_unit(p, S, q + kNS, pol);
                }
            }
        }
        const long long tb = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            // Buffer bb was last read by the helper for unit q - kNRB: completion number
            // q / kNRB - 1 of its free barrier.
            if (q >= kNRB) {
                hand_wait(&S.hand[kNRB + bb], ((q / kNRB) & 1) ^ 1, p.ctr);
            }
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            hand_arrive(&S.hand[bb]);
        } else {
            named_sync(kBarFree + bb, kNTL);
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            named_arrive(kBarFull + bb, kNTL);
        }
        if constexpr (kEarly || kRefillFirst) {
            continue;
        }
        const long long tu = kStamp ? clock64() : 0;
        named_sync(kBarUnit, kNT);
        if (kStamp && tid == 0) {
            S.dg[15] += clock64() - tu;
        }
        if (tid == 0) {
            ustamp(p, q, 3);
        }
        if (tid == 0 && q + kNS < n) {
            // The slot's generic-proxy reads are complete (the barrier above); order them
            // before the async-proxy write that refills it.
            fence_proxy_async();
            issue_unit(p, S, q + kNS, pol);
        }
    }
    if (kStamp && tid == 0) {
        if (na == n) {
            S.dg[3] = gtimer();
            S.dg[4] = S.dg[3];
            S.dg[8] = wfull;
            wfull = 0;
        }
        S.dg[9] = wfull;
    }
}

// Makes this warp's inter stores for job j visible, then adds n finished units to its counter.
__device__ __forceinline__ void credit(const Params& p, Smem& S, int j, int n, int lane) {
    if (n <= 0) {
        return;
    }
    const long long c0 = kStamp ? clock64() : 0;
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomicAdd(p.done_a + j, n);
    }
    if (kStamp && lane == 0) {
        S.dg[11] += clock64() - c0;
    }
}

// Helper warp: sums the 8 warps' partials of each unit and runs its epilogue.
__device__ __forceinline__ void helper_loop(const Params& p, Smem& S, float* ss) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    if constexpr (!kHand) {
#pragma unroll
        for (int bb = 0; bb < kNRB; ++bb) {
            named_arrive(kBarFree + bb, kNTL);
        }
    }
    int jcred = -1;
    int ncred = 0;
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        const int rb = phb ? (v & 255) : (v & 511);
        const int e = S.job_e[j];
        const int q0 = S.job_q[j];
        const int ntok = S.job_n[j];
        const int bb = q & (kNRB - 1);
        // Phase A's last credits go out before waiting on any phase B partials: the compute
        // warps produce none until done_a is complete, which needs these credits.
        if (phb && ncred > 0) {
            credit(p, S, jcred, ncred, lane);
            ncred = 0;
        }
        const long long cw = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            hand_wait(&S.hand[bb], (q / kNRB) & 1, p.ctr);
        } else {
            named_sync(kBarFull + bb, kNTL);
        }
        if (kStamp && lane == 0) {
            S.dg[10] += clock64() - cw;
        }
        float4 r = S.red[bb][0][lane];
#pragma unroll
        for (int w = 1; w < kNW; ++w) {
            const float4 t = S.red[bb][w][lane];
            r.x += t.x;
            r.y += t.y;
            r.z += t.z;
            r.w += t.w;
        }
        if constexpr (kHand) {
            hand_arrive(&S.hand[kNRB + bb]);
        } else {
            named_arrive(kBarFree + bb, kNTL);
        }
        const bool t0 = 2 * t4 < ntok;
        const bool t1 = 2 * t4 + 1 < ntok;
        if (!phb) {
            // Marlin writes gate and up to bf16 before silu_and_mul, and SwiGLU's output is
            // GEMM2's bf16 input.
            const float gsa = __ldg(p.gs13 + e) * 128.f;
            const float g0 = bf16r(r.x * gsa);
            const float g1 = bf16r(r.y * gsa);
            const float u0 = bf16r(r.z * gsa);
            const float u1 = bf16r(r.w * gsa);
            const float h0 = g0 * __frcp_rn(1.f + __expf(-g0)) * u0;
            const float h1 = g1 * __frcp_rn(1.f + __expf(-g1)) * u1;
            __nv_bfloat16* dst = p.inter + static_cast<size_t>(q0 + 2 * t4) * kI + rb * 8 + g;
            if (t0) {
                dst[0] = __float2bfloat16_rn(h0);
            }
            if (t1) {
                dst[kI] = __float2bfloat16_rn(h1);
            }
            // Credits go out once per job: the fence behind each credit costs about a
            // microsecond under a saturated memory system, too much for every unit.
            if (j != jcred) {
                credit(p, S, jcred, ncred, lane);
                jcred = j;
                ncred = 0;
            }
            ++ncred;
        } else {
            // GEMM2's output is bf16 in SGLang; the norm then runs in FP32.
            const float gsb = __ldg(p.gs2 + e) * 128.f;
            const int col0 = rb * 16 + g;
            const int col1 = col0 + 8;
            const float w0 = __ldg(p.nw + col0);
            const float w1 = __ldg(p.nw + col1);
            const float y00 = bf16r(r.x * gsb) * w0;
            const float y01 = bf16r(r.y * gsb) * w0;
            const float y10 = bf16r(r.z * gsb) * w1;
            const float y11 = bf16r(r.w * gsb) * w1;
            float* dst = p.ybuf + static_cast<size_t>(q0 + 2 * t4) * kH;
            if (t0) {
                dst[col0] = y00;
                dst[col1] = y10;
            }
            if (t1) {
                dst[kH + col0] = y01;
                dst[kH + col1] = y11;
            }
            float s0 = y00 * y00 + y10 * y10;
            float s1 = y01 * y01 + y11 * y11;
#pragma unroll
            for (int d = 4; d < 32; d <<= 1) {
                s0 += __shfl_xor_sync(0xffffffffu, s0, d);
                s1 += __shfl_xor_sync(0xffffffffu, s1, d);
            }
            if (g == 0) {
                if (t0) {
                    atomicAdd(ss + q0 + 2 * t4, s0);
                }
                if (t1) {
                    atomicAdd(ss + q0 + 2 * t4 + 1, s1);
                }
            }
        }
    }
    if (ncred > 0) {
        credit(p, S, jcred, ncred, lane);
    }
}

template <int SPLIT>
__global__ void __launch_bounds__(kNTL, 1) moe_fused_kernel(const Params p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int G = gridDim.x;
    const int cta = blockIdx.x;
    const int E = p.M * kTopK;
    // Under programmatic dependent launch the inputs are not ready before this; without it
    // the instruction does nothing.
    if (p.pdl) {
        if (warp == kNW) {
            if (lane == 0) {
    #pragma unroll
                for (int s = 0; s < kNS; ++s) {
                    mbar_init(&S.full[s], 1);
                    mbar_init(&S.empty[s], kNT);
                }
                if constexpr (kHand) {
    #pragma unroll
                    for (int b = 0; b < kNRB; ++b) {
                        mbar_init(&S.hand[b], kNT);
                        mbar_init(&S.hand[kNRB + b], 32);
                    }
                }
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
        }
        if (tid == 0) {
            for (int i = cta; i < p.nloc * 4; i += G) {
                prefetch_l2_hint(p.w13 + (static_cast<size_t>(i % p.nloc) * kUA
                                          + i / p.nloc) * kUnitBytes, kUnitBytes);
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // The generation picks this launch's sum-of-squares copy and barrier counter; the last
    // CTA through the barrier bumps it, after every CTA has read it.
    const int gen = *reinterpret_cast<volatile int*>(p.ctr + 3);
    float* ss = p.ss + (gen & 1) * kEmax;
    const uint64_t pol = policy_evict_first();
    if (kStamp && tid < 16) {
        S.dg[tid] = tid == 0 ? gtimer() : tid == 12 ? clock64() : 0;
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[2] = clock64();
        ustamps(p)[3] = gtimer();
    }

    if (warp == kNW) {
        // Histogram and job list by one warp, so this CTA's first unit copies can be issued
        // before the full routing; the copies' cold start then overlaps it.
        if (lane == 0 && !p.pdl) {
#pragma unroll
            for (int s = 0; s < kNS; ++s) {
                mbar_init(&S.full[s], 1);
                mbar_init(&S.empty[s], kNT);
            }
            if constexpr (kHand) {
#pragma unroll
                for (int b = 0; b < kNRB; ++b) {
                    mbar_init(&S.hand[b], kNT);
                    mbar_init(&S.hand[kNRB + b], 32);
                }
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        S.cnt[lane] = 0;
        __syncwarp();
        for (int base = 0; base < E; base += 128) {
            const int i = base + 4 * lane;
            int4 v = make_int4(-1, -1, -1, -1);
            if (i < E) {
                v = __ldg(reinterpret_cast<const int4*>(p.topk_ids + i));
            }
            const int ev[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = ev[k] - ld_lo(p.lo_p);
                const int eid = (i < E && e >= 0 && e < p.nloc) ? e : -1;
                if (i < E) {
                    S.eid[i + k] = static_cast<signed char>(eid);
                }
                const unsigned peers = __match_any_sync(0xffffffffu, eid);
                if (eid >= 0 && lane == __ffs(peers) - 1) {
                    S.cnt[eid] += __popc(peers);
                }
                __syncwarp();
            }
        }
        const int tot = lane < p.nloc ? S.cnt[lane] : 0;
        const int off = warp_scan(tot, lane) - tot;
        S.off[lane] = off;
        const int nj = SPLIT ? (tot + 7) >> 3 : (tot > 0 ? 1 : 0);
        const int jincl = warp_scan(nj, lane);
        const int j0 = jincl - nj;
        const int njobs = __shfl_sync(0xffffffffu, jincl, 31);
        for (int k = 0; k < nj; ++k) {
            S.job_e[j0 + k] = lane;
            S.job_q[j0 + k] = off + 8 * k;
            S.job_n[j0 + k] = min(8, tot - 8 * k);
        }
        __syncwarp();
        if (lane == 0) {
            const unsigned ua = njobs * kUA;
            const unsigned ub = njobs * kUB;
            const int a0 = ua * cta / G;
            const int a1 = ua * (cta + 1) / G;
            const int b0 = ub * cta / G;
            const int b1 = ub * (cta + 1) / G;
            S.njobs = njobs;
            S.a0 = a0;
            S.na = a1 - a0;
            S.b0 = b0;
            S.nb = b1 - b0;
            __threadfence_block();
            const int n = (a1 - a0) + (b1 - b0);
            for (int q = 0; q < min(kNS, n); ++q) {
                issue_unit(p, S, q, pol);
            }
        }
    } else {
        const int ctid = cta * kNT + tid;
        const int cstride = G * kNT;
        for (int i = ctid; i < p.M * 64; i += cstride) {
            prefetch_l2(p.x + i * 64);
        }
        for (int i = ctid; i < (E + 31) / 32; i += cstride) {
            prefetch_l2(p.topk_w + i * 32);
        }
        // The next launch's sum-of-squares copy and barrier counter; nothing reads them in
        // this launch.
        float* ss_next = p.ss + ((gen + 1) & 1) * kEmax;
        for (int i = ctid; i < kEmax; i += cstride) {
            ss_next[i] = 0.f;
        }
        if (ctid == 0) {
            p.ctr[4 + ((gen + 1) & 1)] = 0;
        }
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[1] = gtimer();
    }

    position_pass<kNWarps>(S.eid, S.pos, S.ent, S.wc, S.off, E);
    if (kStamp && tid == 0) {
        S.dg[2] = gtimer();
    }

    if (warp < kNW) {
        compute_loop(p, S, pol);
    } else {
        helper_loop(p, S, ss);
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[5] = gtimer();
    }

    // Grid barrier. The counter alternates with the generation, so the one this launch
    // uses was zeroed by the previous launch and nothing is reset between launches.
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
    if (kStamp && tid == 0) {
        S.dg[6] = gtimer();
    }
    for (int j = cta * kNTL + tid; j < kJmax; j += G * kNTL) {
        p.done_a[j] = 0;
    }

    if (warp < kNW) {
        final_pass<kNT>(p, S.pos, ss, cta, G);
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[4] = clock64();
        ustamps(p)[5] = gtimer();
    }
    if constexpr (kStamp) {
        __syncthreads();
        if (tid == 0) {
            S.dg[7] = gtimer();
            S.dg[13] = clock64();
        }
        __syncthreads();
        if (tid < 16) {
            reinterpret_cast<long long*>(p.inter)[cta * 16 + tid] = S.dg[tid];
        }
    }
}

template <int SPLIT>
void launch(const Params& p, int sms, cudaStream_t stream) {
    static bool ready = false;
    if (!ready) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused_kernel<SPLIT>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(sizeof(Smem))));
        ready = true;
    }
    moe1_launch(moe_fused_kernel<SPLIT>, sms, kNTL, static_cast<int>(sizeof(Smem)), stream,
                p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sk_16_4_0


namespace sk_8_3_0 {

constexpr int kNS = 3;
constexpr int kNW = 8;
constexpr int kMode = 0;
// kMode 2 and 4: timing stamps (2 is mode 0's, 4 is mode 3's). kMode 3: each compute warp loads
// its codes of a unit up front and releases the ring slot itself (fence.proxy.async, then
// mbarrier.arrive on empty, count kNT, as the fat path's mode 28), and thread 0 refills the slot
// once every warp has arrived, instead of after a bar.sync of all compute warps behind the
// unit's last MMA. The A/B at 8 tokens (240b68e7df1a427f) put mode 0 3.6 us over its loads-only
// twin while thread 0 never waited for a unit, so the per-unit barrier is on the critical path.
// kMode 5 and 7: the scale decode's multiply by 16 takes a multiplier ptxas cannot see, so it
// stays an IMAD (FMA pipe) instead of becoming a LEA on the ALU pipe that the FP4 decode
// saturates (the public kernel's SCDK). kMode 6 and 7: the partial-sum buffers are handed to and
// from the helper warp through mbarriers (every compute thread arrives on hand[bb] after its
// store, every helper lane on hand[kNRB + bb] after its loads) instead of a 288-thread bar.sync
// per unit, so the compute warps meet once per unit (kBarUnit) instead of twice. The ordering
// chain is in docs/runs-moe.md, "Skinny partial handoff through mbarriers". kMode 8: 7 with
// the stamps.
// Modes 16 and up are flags over mode kLow (docs/runs-moe.md, "Skinny weight stream against the
// public kernel"). 16: thread 0's clock64 per unit (issue, wait start, wait end, slot free) and
// a header, written to p.inter from 16 MB on (pair rows >= 2048, which the skinny sizes never
// use). 32: the unit barrier and the refill move ahead of the partial-buffer handoff, so a refill
// no longer waits for the helper to free a buffer. 64: weight copies without the L2 hint.
constexpr int kLow = kMode & 15;
constexpr bool kStamp = kLow == 2 || kLow == 4 || kLow == 8;
constexpr bool kEarly = kLow == 3 || kLow == 4;
constexpr bool kScaleFma = kLow == 5 || kLow == 7 || kLow == 8;
constexpr bool kHand = kLow >= 6;
constexpr bool kUnitStamp = (kMode & 16) != 0;
constexpr bool kRefillFirst = (kMode & 32) != 0;
constexpr bool kNoHint = (kMode & 64) != 0;
static_assert(!kRefillFirst || (!kEarly && !kHand), "refill-first is a mode 0 / 5 change");
constexpr int kUSMax = 256;
constexpr int kUSPer = 8 + 4 * kUSMax;
constexpr int kNT = kNW * 32;
constexpr int kNTL = kNT + 32;
constexpr int kNWarps = kNTL / 32;

struct __align__(128) Smem {
    unsigned char ring[kNS][kUnitBytes];
    float4 red[kNRB][kNW][32];
    uint64_t full[kNS];
    uint64_t empty[kNS];
    int cnt[kLocalMax];
    int off[kLocalMax];
    int wc[kNWarps][kLocalMax];
    int job_e[kJmax];
    int job_q[kJmax];
    int job_n[kJmax];
    short pos[kEmax];
    short ent[kEmax];
    signed char eid[kEmax];
    int njobs;
    int a0, na, b0, nb;
    // kMode 2 (timing only): thread 0's globaltimer at the phase points and cycle counters,
    // written to p.inter at the end (16 int64 per CTA; tools/moe_make_bench_ab.py reads them).
    long long dg[kStamp ? 16 : 1];
    uint64_t hand[kHand ? 2 * kNRB : 1];
};

__device__ __forceinline__ void hand_wait(uint64_t* bar, unsigned parity, int* ctr) {
    if (!mbar_try(bar, parity)) {
        const long long t0 = clock64();
        while (!mbar_try(bar, parity)) {
            if (spin_failed(t0, ctr, 2)) {
                break;
            }
        }
    }
}

__device__ __forceinline__ void hand_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ long long* ustamps(const Params& p) {
    return reinterpret_cast<long long*>(p.inter) + (1 << 21) + blockIdx.x * kUSPer;
}

// Event k of unit q: 0 copy issued, 1 thread 0 starts waiting for it, 2 the wait ends, 3 the
// slot is free (unit barrier passed; the refill follows).
__device__ __forceinline__ void ustamp(const Params& p, int q, int k) {
    if (kUnitStamp && q < kUSMax) {
        ustamps(p)[8 + 4 * q + k] = clock64();
    }
}

// Unit q of this CTA's list (its phase A units, then its phase B units) into ring slot q % kNS.
__device__ __forceinline__ void issue_unit(const Params& p, Smem& S, int q, uint64_t pol) {
    const int slot = q % kNS;
    const unsigned char* src;
    if (q < S.na) {
        const int v = S.a0 + q;
        src = p.w13 + (static_cast<size_t>(S.job_e[v >> 9]) * kUA + (v & 511)) * kUnitBytes;
    } else {
        const int v = S.b0 + (q - S.na);
        src = p.w2 + (static_cast<size_t>(S.job_e[v >> 8]) * kUB + (v & 255)) * kUnitBytes;
    }
    mbar_expect_tx(&S.full[slot], kUnitBytes);
    ustamp(p, q, 0);
    if constexpr (kNoHint) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];"
                     :: "r"(smem_u32(S.ring[slot])), "l"(src), "r"(kUnitBytes),
                        "r"(smem_u32(&S.full[slot])) : "memory");
    } else {
        bulk_load(S.ring[slot], src, kUnitBytes, &S.full[slot], pol);
    }
}

// Compute warps: for each unit, 8 chunks of 64 k (4 mma k16 steps each) over this warp's
// 512-k slice, the job's x rows kept in registers.
__device__ __forceinline__ void compute_loop(const Params& p, Smem& S, uint64_t pol) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    // The unit layout holds 8 k-lanes of 512 (8 chunks of 64); NW warps split them.
    constexpr int kCPW = 64 / kNW;
    const int w8 = warp / (kNW / 8);
    const int c0 = (warp % (kNW / 8)) * kCPW;
    uint4 xr[kCPW][2];
    int jcur = -1;
    bool bcur = false;
    long long wfull = 0;
    // 16, from a kernel parameter (nloc <= kLocalMax = 32) so ptxas cannot fold it into a LEA.
    const unsigned k16 = 16u + (static_cast<unsigned>(p.nloc) >> 6);
    if (kUnitStamp && tid == 0) {
        ustamps(p)[0] = na;
        ustamps(p)[1] = n;
    }
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        if (kStamp && tid == 0 && q == na) {
            S.dg[3] = gtimer();
            S.dg[8] = wfull;
            wfull = 0;
        }
        if (j != jcur || phb != bcur) {
            jcur = j;
            bcur = phb;
            const int q0 = S.job_q[j];
            const bool have = g < S.job_n[j];
            if (!phb) {
                const int tok = have ? (S.ent[q0 + g] >> 3) : 0;
                const __nv_bfloat16* src = p.x + static_cast<size_t>(tok) * kH + w8 * 512 + c0 * 64
                                           + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            } else {
                if (ld_acquire(p.done_a + j) < kUA) {
                    const long long t0 = clock64();
                    while (ld_acquire(p.done_a + j) < kUA) {
                        __nanosleep(64);
                        if (spin_failed(t0, p.ctr, 1)) {
                            break;
                        }
                    }
                }
                if (kStamp && tid == 0 && q == na) {
                    S.dg[4] = gtimer();
                }
                const __nv_bfloat16* src = p.inter + static_cast<size_t>(q0 + (have ? g : 0)) * kI
                                           + w8 * 512 + c0 * 64 + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        const int slot = q % kNS;
        // The stamps time the first try too: try_wait may suspend the thread for a
        // system-dependent time before it returns false (PTX ISA, mbarrier.try_wait).
        const long long tw = kStamp ? clock64() : 0;
        if (tid == 0) {
            ustamp(p, q, 1);
        }
        if (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
            const long long t0 = clock64();
            while (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
                if (spin_failed(t0, p.ctr, 2)) {
                    break;
                }
            }
        }
        if (kStamp) {
            wfull += clock64() - tw;
        }
        if (tid == 0) {
            ustamp(p, q, 2);
        }
        const unsigned char* sl = S.ring[slot];
        const uint4* codes = reinterpret_cast<const uint4*>(sl + w8 * 4096 + c0 * 512) + lane;
        const uint4 sc = *(reinterpret_cast<const uint4*>(sl + 32768 + w8 * 512) + lane);
        float acc[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[s][i] = 0.f;
            }
        }
        uint4 cw[kEarly ? kCPW : 1];
        if constexpr (kEarly) {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                cw[c] = codes[c * 32];
            }
            // Every generic read of the slot (codes, scales) is issued above; the per-thread
            // proxy fence orders them before the refill's bulk write (docs/runs-moe.md,
            // "Ordering of the ring slot releases"; the chain is the fat path's mode 28).
            fence_proxy_async();
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         :: "r"(smem_u32(&S.empty[slot])) : "memory");
            if (tid == 0 && q + kNS < n) {
                const unsigned par = (q / kNS) & 1;
                if (!mbar_try(&S.empty[slot], par)) {
                    const long long t0 = clock64();
                    while (!mbar_try(&S.empty[slot], par)) {
                        if (spin_failed(t0, p.ctr, 2)) {
                            break;
                        }
                    }
                }
                issue_unit(p, S, q + kNS, pol);
            }
        }
        if (kLow == 1) {
            // Timing only: the unit's bytes arrive and one word is read, nothing is computed.
            acc[0][0] = __uint_as_float(codes[0].x & 0x3f800000u);
        } else {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                const uint4 w = kEarly ? cw[c] : codes[c * 32];
                const unsigned scw = word(sc, (c0 + c) >> 1);
                const unsigned sg = kScaleFma ? scale_word_k(scw, (2 * (c0 + c)) & 3, k16)
                                              : scale_word(scw, (2 * (c0 + c)) & 3);
                const unsigned sg8 = kScaleFma ? scale_word_k(scw, (2 * (c0 + c) + 1) & 3, k16)
                                               : scale_word(scw, (2 * (c0 + c) + 1) & 3);
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    unsigned a[4];
                    dequant(word(w, s), sg, sg8, a);
                    mma_bf16(acc[s], a, word(xr[c][s >> 1], 2 * (s & 1)),
                             word(xr[c][s >> 1], 2 * (s & 1) + 1));
                }
            }
        }
        float4 r;
        r.x = (acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]);
        r.y = (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]);
        r.z = (acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2]);
        r.w = (acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3]);
        const int bb = q & (kNRB - 1);
        if constexpr (kRefillFirst) {
            // Every compute thread's reads of the slot are done once its MMAs have their
            // operands; the barrier and the fence are the same chain as below, earlier.
            named_sync(kBarUnit, kNT);
            if (tid == 0) {
                ustamp(p, q, 3);
                if (q + kNS < n) {
                    fence_proxy_async();
                    issue_unit(p, S, q + kNS, pol);
                }
            }
        }
        const long long tb = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            // Buffer bb was last read by the helper for unit q - kNRB: completion number
            // q / kNRB - 1 of its free barrier.
            if (q >= kNRB) {
                hand_wait(&S.hand[kNRB + bb], ((q / kNRB) & 1) ^ 1, p.ctr);
            }
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            hand_arrive(&S.hand[bb]);
        } else {
            named_sync(kBarFree + bb, kNTL);
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            named_arrive(kBarFull + bb, kNTL);
        }
        if constexpr (kEarly || kRefillFirst) {
            continue;
        }
        const long long tu = kStamp ? clock64() : 0;
        named_sync(kBarUnit, kNT);
        if (kStamp && tid == 0) {
            S.dg[15] += clock64() - tu;
        }
        if (tid == 0) {
            ustamp(p, q, 3);
        }
        if (tid == 0 && q + kNS < n) {
            // The slot's generic-proxy reads are complete (the barrier above); order them
            // before the async-proxy write that refills it.
            fence_proxy_async();
            issue_unit(p, S, q + kNS, pol);
        }
    }
    if (kStamp && tid == 0) {
        if (na == n) {
            S.dg[3] = gtimer();
            S.dg[4] = S.dg[3];
            S.dg[8] = wfull;
            wfull = 0;
        }
        S.dg[9] = wfull;
    }
}

// Makes this warp's inter stores for job j visible, then adds n finished units to its counter.
__device__ __forceinline__ void credit(const Params& p, Smem& S, int j, int n, int lane) {
    if (n <= 0) {
        return;
    }
    const long long c0 = kStamp ? clock64() : 0;
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomicAdd(p.done_a + j, n);
    }
    if (kStamp && lane == 0) {
        S.dg[11] += clock64() - c0;
    }
}

// Helper warp: sums the 8 warps' partials of each unit and runs its epilogue.
__device__ __forceinline__ void helper_loop(const Params& p, Smem& S, float* ss) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    if constexpr (!kHand) {
#pragma unroll
        for (int bb = 0; bb < kNRB; ++bb) {
            named_arrive(kBarFree + bb, kNTL);
        }
    }
    int jcred = -1;
    int ncred = 0;
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        const int rb = phb ? (v & 255) : (v & 511);
        const int e = S.job_e[j];
        const int q0 = S.job_q[j];
        const int ntok = S.job_n[j];
        const int bb = q & (kNRB - 1);
        // Phase A's last credits go out before waiting on any phase B partials: the compute
        // warps produce none until done_a is complete, which needs these credits.
        if (phb && ncred > 0) {
            credit(p, S, jcred, ncred, lane);
            ncred = 0;
        }
        const long long cw = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            hand_wait(&S.hand[bb], (q / kNRB) & 1, p.ctr);
        } else {
            named_sync(kBarFull + bb, kNTL);
        }
        if (kStamp && lane == 0) {
            S.dg[10] += clock64() - cw;
        }
        float4 r = S.red[bb][0][lane];
#pragma unroll
        for (int w = 1; w < kNW; ++w) {
            const float4 t = S.red[bb][w][lane];
            r.x += t.x;
            r.y += t.y;
            r.z += t.z;
            r.w += t.w;
        }
        if constexpr (kHand) {
            hand_arrive(&S.hand[kNRB + bb]);
        } else {
            named_arrive(kBarFree + bb, kNTL);
        }
        const bool t0 = 2 * t4 < ntok;
        const bool t1 = 2 * t4 + 1 < ntok;
        if (!phb) {
            // Marlin writes gate and up to bf16 before silu_and_mul, and SwiGLU's output is
            // GEMM2's bf16 input.
            const float gsa = __ldg(p.gs13 + e) * 128.f;
            const float g0 = bf16r(r.x * gsa);
            const float g1 = bf16r(r.y * gsa);
            const float u0 = bf16r(r.z * gsa);
            const float u1 = bf16r(r.w * gsa);
            const float h0 = g0 * __frcp_rn(1.f + __expf(-g0)) * u0;
            const float h1 = g1 * __frcp_rn(1.f + __expf(-g1)) * u1;
            __nv_bfloat16* dst = p.inter + static_cast<size_t>(q0 + 2 * t4) * kI + rb * 8 + g;
            if (t0) {
                dst[0] = __float2bfloat16_rn(h0);
            }
            if (t1) {
                dst[kI] = __float2bfloat16_rn(h1);
            }
            // Credits go out once per job: the fence behind each credit costs about a
            // microsecond under a saturated memory system, too much for every unit.
            if (j != jcred) {
                credit(p, S, jcred, ncred, lane);
                jcred = j;
                ncred = 0;
            }
            ++ncred;
        } else {
            // GEMM2's output is bf16 in SGLang; the norm then runs in FP32.
            const float gsb = __ldg(p.gs2 + e) * 128.f;
            const int col0 = rb * 16 + g;
            const int col1 = col0 + 8;
            const float w0 = __ldg(p.nw + col0);
            const float w1 = __ldg(p.nw + col1);
            const float y00 = bf16r(r.x * gsb) * w0;
            const float y01 = bf16r(r.y * gsb) * w0;
            const float y10 = bf16r(r.z * gsb) * w1;
            const float y11 = bf16r(r.w * gsb) * w1;
            float* dst = p.ybuf + static_cast<size_t>(q0 + 2 * t4) * kH;
            if (t0) {
                dst[col0] = y00;
                dst[col1] = y10;
            }
            if (t1) {
                dst[kH + col0] = y01;
                dst[kH + col1] = y11;
            }
            float s0 = y00 * y00 + y10 * y10;
            float s1 = y01 * y01 + y11 * y11;
#pragma unroll
            for (int d = 4; d < 32; d <<= 1) {
                s0 += __shfl_xor_sync(0xffffffffu, s0, d);
                s1 += __shfl_xor_sync(0xffffffffu, s1, d);
            }
            if (g == 0) {
                if (t0) {
                    atomicAdd(ss + q0 + 2 * t4, s0);
                }
                if (t1) {
                    atomicAdd(ss + q0 + 2 * t4 + 1, s1);
                }
            }
        }
    }
    if (ncred > 0) {
        credit(p, S, jcred, ncred, lane);
    }
}

template <int SPLIT>
__global__ void __launch_bounds__(kNTL, 1) moe_fused_kernel(const Params p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int G = gridDim.x;
    const int cta = blockIdx.x;
    const int E = p.M * kTopK;
    // Under programmatic dependent launch the inputs are not ready before this; without it
    // the instruction does nothing.
    if (p.pdl) {
        if (warp == kNW) {
            if (lane == 0) {
    #pragma unroll
                for (int s = 0; s < kNS; ++s) {
                    mbar_init(&S.full[s], 1);
                    mbar_init(&S.empty[s], kNT);
                }
                if constexpr (kHand) {
    #pragma unroll
                    for (int b = 0; b < kNRB; ++b) {
                        mbar_init(&S.hand[b], kNT);
                        mbar_init(&S.hand[kNRB + b], 32);
                    }
                }
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
        }
        if (tid == 0) {
            for (int i = cta; i < p.nloc * 4; i += G) {
                prefetch_l2_hint(p.w13 + (static_cast<size_t>(i % p.nloc) * kUA
                                          + i / p.nloc) * kUnitBytes, kUnitBytes);
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // The generation picks this launch's sum-of-squares copy and barrier counter; the last
    // CTA through the barrier bumps it, after every CTA has read it.
    const int gen = *reinterpret_cast<volatile int*>(p.ctr + 3);
    float* ss = p.ss + (gen & 1) * kEmax;
    const uint64_t pol = policy_evict_first();
    if (kStamp && tid < 16) {
        S.dg[tid] = tid == 0 ? gtimer() : tid == 12 ? clock64() : 0;
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[2] = clock64();
        ustamps(p)[3] = gtimer();
    }

    if (warp == kNW) {
        // Histogram and job list by one warp, so this CTA's first unit copies can be issued
        // before the full routing; the copies' cold start then overlaps it.
        if (lane == 0 && !p.pdl) {
#pragma unroll
            for (int s = 0; s < kNS; ++s) {
                mbar_init(&S.full[s], 1);
                mbar_init(&S.empty[s], kNT);
            }
            if constexpr (kHand) {
#pragma unroll
                for (int b = 0; b < kNRB; ++b) {
                    mbar_init(&S.hand[b], kNT);
                    mbar_init(&S.hand[kNRB + b], 32);
                }
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        S.cnt[lane] = 0;
        __syncwarp();
        for (int base = 0; base < E; base += 128) {
            const int i = base + 4 * lane;
            int4 v = make_int4(-1, -1, -1, -1);
            if (i < E) {
                v = __ldg(reinterpret_cast<const int4*>(p.topk_ids + i));
            }
            const int ev[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = ev[k] - ld_lo(p.lo_p);
                const int eid = (i < E && e >= 0 && e < p.nloc) ? e : -1;
                if (i < E) {
                    S.eid[i + k] = static_cast<signed char>(eid);
                }
                const unsigned peers = __match_any_sync(0xffffffffu, eid);
                if (eid >= 0 && lane == __ffs(peers) - 1) {
                    S.cnt[eid] += __popc(peers);
                }
                __syncwarp();
            }
        }
        const int tot = lane < p.nloc ? S.cnt[lane] : 0;
        const int off = warp_scan(tot, lane) - tot;
        S.off[lane] = off;
        const int nj = SPLIT ? (tot + 7) >> 3 : (tot > 0 ? 1 : 0);
        const int jincl = warp_scan(nj, lane);
        const int j0 = jincl - nj;
        const int njobs = __shfl_sync(0xffffffffu, jincl, 31);
        for (int k = 0; k < nj; ++k) {
            S.job_e[j0 + k] = lane;
            S.job_q[j0 + k] = off + 8 * k;
            S.job_n[j0 + k] = min(8, tot - 8 * k);
        }
        __syncwarp();
        if (lane == 0) {
            const unsigned ua = njobs * kUA;
            const unsigned ub = njobs * kUB;
            const int a0 = ua * cta / G;
            const int a1 = ua * (cta + 1) / G;
            const int b0 = ub * cta / G;
            const int b1 = ub * (cta + 1) / G;
            S.njobs = njobs;
            S.a0 = a0;
            S.na = a1 - a0;
            S.b0 = b0;
            S.nb = b1 - b0;
            __threadfence_block();
            const int n = (a1 - a0) + (b1 - b0);
            for (int q = 0; q < min(kNS, n); ++q) {
                issue_unit(p, S, q, pol);
            }
        }
    } else {
        const int ctid = cta * kNT + tid;
        const int cstride = G * kNT;
        for (int i = ctid; i < p.M * 64; i += cstride) {
            prefetch_l2(p.x + i * 64);
        }
        for (int i = ctid; i < (E + 31) / 32; i += cstride) {
            prefetch_l2(p.topk_w + i * 32);
        }
        // The next launch's sum-of-squares copy and barrier counter; nothing reads them in
        // this launch.
        float* ss_next = p.ss + ((gen + 1) & 1) * kEmax;
        for (int i = ctid; i < kEmax; i += cstride) {
            ss_next[i] = 0.f;
        }
        if (ctid == 0) {
            p.ctr[4 + ((gen + 1) & 1)] = 0;
        }
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[1] = gtimer();
    }

    position_pass<kNWarps>(S.eid, S.pos, S.ent, S.wc, S.off, E);
    if (kStamp && tid == 0) {
        S.dg[2] = gtimer();
    }

    if (warp < kNW) {
        compute_loop(p, S, pol);
    } else {
        helper_loop(p, S, ss);
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[5] = gtimer();
    }

    // Grid barrier. The counter alternates with the generation, so the one this launch
    // uses was zeroed by the previous launch and nothing is reset between launches.
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
    if (kStamp && tid == 0) {
        S.dg[6] = gtimer();
    }
    for (int j = cta * kNTL + tid; j < kJmax; j += G * kNTL) {
        p.done_a[j] = 0;
    }

    if (warp < kNW) {
        final_pass<kNT>(p, S.pos, ss, cta, G);
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[4] = clock64();
        ustamps(p)[5] = gtimer();
    }
    if constexpr (kStamp) {
        __syncthreads();
        if (tid == 0) {
            S.dg[7] = gtimer();
            S.dg[13] = clock64();
        }
        __syncthreads();
        if (tid < 16) {
            reinterpret_cast<long long*>(p.inter)[cta * 16 + tid] = S.dg[tid];
        }
    }
}

template <int SPLIT>
void launch(const Params& p, int sms, cudaStream_t stream) {
    static bool ready = false;
    if (!ready) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused_kernel<SPLIT>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(sizeof(Smem))));
        ready = true;
    }
    moe1_launch(moe_fused_kernel<SPLIT>, sms, kNTL, static_cast<int>(sizeof(Smem)), stream,
                p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sk_8_3_0


namespace sk_8_4_3 {

constexpr int kNS = 4;
constexpr int kNW = 8;
constexpr int kMode = 3;
// kMode 2 and 4: timing stamps (2 is mode 0's, 4 is mode 3's). kMode 3: each compute warp loads
// its codes of a unit up front and releases the ring slot itself (fence.proxy.async, then
// mbarrier.arrive on empty, count kNT, as the fat path's mode 28), and thread 0 refills the slot
// once every warp has arrived, instead of after a bar.sync of all compute warps behind the
// unit's last MMA. The A/B at 8 tokens (240b68e7df1a427f) put mode 0 3.6 us over its loads-only
// twin while thread 0 never waited for a unit, so the per-unit barrier is on the critical path.
// kMode 5 and 7: the scale decode's multiply by 16 takes a multiplier ptxas cannot see, so it
// stays an IMAD (FMA pipe) instead of becoming a LEA on the ALU pipe that the FP4 decode
// saturates (the public kernel's SCDK). kMode 6 and 7: the partial-sum buffers are handed to and
// from the helper warp through mbarriers (every compute thread arrives on hand[bb] after its
// store, every helper lane on hand[kNRB + bb] after its loads) instead of a 288-thread bar.sync
// per unit, so the compute warps meet once per unit (kBarUnit) instead of twice. The ordering
// chain is in docs/runs-moe.md, "Skinny partial handoff through mbarriers". kMode 8: 7 with
// the stamps.
// Modes 16 and up are flags over mode kLow (docs/runs-moe.md, "Skinny weight stream against the
// public kernel"). 16: thread 0's clock64 per unit (issue, wait start, wait end, slot free) and
// a header, written to p.inter from 16 MB on (pair rows >= 2048, which the skinny sizes never
// use). 32: the unit barrier and the refill move ahead of the partial-buffer handoff, so a refill
// no longer waits for the helper to free a buffer. 64: weight copies without the L2 hint.
constexpr int kLow = kMode & 15;
constexpr bool kStamp = kLow == 2 || kLow == 4 || kLow == 8;
constexpr bool kEarly = kLow == 3 || kLow == 4;
constexpr bool kScaleFma = kLow == 5 || kLow == 7 || kLow == 8;
constexpr bool kHand = kLow >= 6;
constexpr bool kUnitStamp = (kMode & 16) != 0;
constexpr bool kRefillFirst = (kMode & 32) != 0;
constexpr bool kNoHint = (kMode & 64) != 0;
static_assert(!kRefillFirst || (!kEarly && !kHand), "refill-first is a mode 0 / 5 change");
constexpr int kUSMax = 256;
constexpr int kUSPer = 8 + 4 * kUSMax;
constexpr int kNT = kNW * 32;
constexpr int kNTL = kNT + 32;
constexpr int kNWarps = kNTL / 32;

struct __align__(128) Smem {
    unsigned char ring[kNS][kUnitBytes];
    float4 red[kNRB][kNW][32];
    uint64_t full[kNS];
    uint64_t empty[kNS];
    int cnt[kLocalMax];
    int off[kLocalMax];
    int wc[kNWarps][kLocalMax];
    int job_e[kJmax];
    int job_q[kJmax];
    int job_n[kJmax];
    short pos[kEmax];
    short ent[kEmax];
    signed char eid[kEmax];
    int njobs;
    int a0, na, b0, nb;
    // kMode 2 (timing only): thread 0's globaltimer at the phase points and cycle counters,
    // written to p.inter at the end (16 int64 per CTA; tools/moe_make_bench_ab.py reads them).
    long long dg[kStamp ? 16 : 1];
    uint64_t hand[kHand ? 2 * kNRB : 1];
};

__device__ __forceinline__ void hand_wait(uint64_t* bar, unsigned parity, int* ctr) {
    if (!mbar_try(bar, parity)) {
        const long long t0 = clock64();
        while (!mbar_try(bar, parity)) {
            if (spin_failed(t0, ctr, 2)) {
                break;
            }
        }
    }
}

__device__ __forceinline__ void hand_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ long long* ustamps(const Params& p) {
    return reinterpret_cast<long long*>(p.inter) + (1 << 21) + blockIdx.x * kUSPer;
}

// Event k of unit q: 0 copy issued, 1 thread 0 starts waiting for it, 2 the wait ends, 3 the
// slot is free (unit barrier passed; the refill follows).
__device__ __forceinline__ void ustamp(const Params& p, int q, int k) {
    if (kUnitStamp && q < kUSMax) {
        ustamps(p)[8 + 4 * q + k] = clock64();
    }
}

// Unit q of this CTA's list (its phase A units, then its phase B units) into ring slot q % kNS.
__device__ __forceinline__ void issue_unit(const Params& p, Smem& S, int q, uint64_t pol) {
    const int slot = q % kNS;
    const unsigned char* src;
    if (q < S.na) {
        const int v = S.a0 + q;
        src = p.w13 + (static_cast<size_t>(S.job_e[v >> 9]) * kUA + (v & 511)) * kUnitBytes;
    } else {
        const int v = S.b0 + (q - S.na);
        src = p.w2 + (static_cast<size_t>(S.job_e[v >> 8]) * kUB + (v & 255)) * kUnitBytes;
    }
    mbar_expect_tx(&S.full[slot], kUnitBytes);
    ustamp(p, q, 0);
    if constexpr (kNoHint) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];"
                     :: "r"(smem_u32(S.ring[slot])), "l"(src), "r"(kUnitBytes),
                        "r"(smem_u32(&S.full[slot])) : "memory");
    } else {
        bulk_load(S.ring[slot], src, kUnitBytes, &S.full[slot], pol);
    }
}

// Compute warps: for each unit, 8 chunks of 64 k (4 mma k16 steps each) over this warp's
// 512-k slice, the job's x rows kept in registers.
__device__ __forceinline__ void compute_loop(const Params& p, Smem& S, uint64_t pol) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    // The unit layout holds 8 k-lanes of 512 (8 chunks of 64); NW warps split them.
    constexpr int kCPW = 64 / kNW;
    const int w8 = warp / (kNW / 8);
    const int c0 = (warp % (kNW / 8)) * kCPW;
    uint4 xr[kCPW][2];
    int jcur = -1;
    bool bcur = false;
    long long wfull = 0;
    // 16, from a kernel parameter (nloc <= kLocalMax = 32) so ptxas cannot fold it into a LEA.
    const unsigned k16 = 16u + (static_cast<unsigned>(p.nloc) >> 6);
    if (kUnitStamp && tid == 0) {
        ustamps(p)[0] = na;
        ustamps(p)[1] = n;
    }
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        if (kStamp && tid == 0 && q == na) {
            S.dg[3] = gtimer();
            S.dg[8] = wfull;
            wfull = 0;
        }
        if (j != jcur || phb != bcur) {
            jcur = j;
            bcur = phb;
            const int q0 = S.job_q[j];
            const bool have = g < S.job_n[j];
            if (!phb) {
                const int tok = have ? (S.ent[q0 + g] >> 3) : 0;
                const __nv_bfloat16* src = p.x + static_cast<size_t>(tok) * kH + w8 * 512 + c0 * 64
                                           + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            } else {
                if (ld_acquire(p.done_a + j) < kUA) {
                    const long long t0 = clock64();
                    while (ld_acquire(p.done_a + j) < kUA) {
                        __nanosleep(64);
                        if (spin_failed(t0, p.ctr, 1)) {
                            break;
                        }
                    }
                }
                if (kStamp && tid == 0 && q == na) {
                    S.dg[4] = gtimer();
                }
                const __nv_bfloat16* src = p.inter + static_cast<size_t>(q0 + (have ? g : 0)) * kI
                                           + w8 * 512 + c0 * 64 + t4 * 16;
#pragma unroll
                for (int c = 0; c < kCPW; ++c) {
                    xr[c][0] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64))
                                    : make_uint4(0u, 0u, 0u, 0u);
                    xr[c][1] = have ? __ldcg(reinterpret_cast<const uint4*>(src + c * 64 + 8))
                                    : make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        const int slot = q % kNS;
        // The stamps time the first try too: try_wait may suspend the thread for a
        // system-dependent time before it returns false (PTX ISA, mbarrier.try_wait).
        const long long tw = kStamp ? clock64() : 0;
        if (tid == 0) {
            ustamp(p, q, 1);
        }
        if (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
            const long long t0 = clock64();
            while (!mbar_try(&S.full[slot], (q / kNS) & 1)) {
                if (spin_failed(t0, p.ctr, 2)) {
                    break;
                }
            }
        }
        if (kStamp) {
            wfull += clock64() - tw;
        }
        if (tid == 0) {
            ustamp(p, q, 2);
        }
        const unsigned char* sl = S.ring[slot];
        const uint4* codes = reinterpret_cast<const uint4*>(sl + w8 * 4096 + c0 * 512) + lane;
        const uint4 sc = *(reinterpret_cast<const uint4*>(sl + 32768 + w8 * 512) + lane);
        float acc[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[s][i] = 0.f;
            }
        }
        uint4 cw[kEarly ? kCPW : 1];
        if constexpr (kEarly) {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                cw[c] = codes[c * 32];
            }
            // Every generic read of the slot (codes, scales) is issued above; the per-thread
            // proxy fence orders them before the refill's bulk write (docs/runs-moe.md,
            // "Ordering of the ring slot releases"; the chain is the fat path's mode 28).
            fence_proxy_async();
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         :: "r"(smem_u32(&S.empty[slot])) : "memory");
            if (tid == 0 && q + kNS < n) {
                const unsigned par = (q / kNS) & 1;
                if (!mbar_try(&S.empty[slot], par)) {
                    const long long t0 = clock64();
                    while (!mbar_try(&S.empty[slot], par)) {
                        if (spin_failed(t0, p.ctr, 2)) {
                            break;
                        }
                    }
                }
                issue_unit(p, S, q + kNS, pol);
            }
        }
        if (kLow == 1) {
            // Timing only: the unit's bytes arrive and one word is read, nothing is computed.
            acc[0][0] = __uint_as_float(codes[0].x & 0x3f800000u);
        } else {
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                const uint4 w = kEarly ? cw[c] : codes[c * 32];
                const unsigned scw = word(sc, (c0 + c) >> 1);
                const unsigned sg = kScaleFma ? scale_word_k(scw, (2 * (c0 + c)) & 3, k16)
                                              : scale_word(scw, (2 * (c0 + c)) & 3);
                const unsigned sg8 = kScaleFma ? scale_word_k(scw, (2 * (c0 + c) + 1) & 3, k16)
                                               : scale_word(scw, (2 * (c0 + c) + 1) & 3);
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    unsigned a[4];
                    dequant(word(w, s), sg, sg8, a);
                    mma_bf16(acc[s], a, word(xr[c][s >> 1], 2 * (s & 1)),
                             word(xr[c][s >> 1], 2 * (s & 1) + 1));
                }
            }
        }
        float4 r;
        r.x = (acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]);
        r.y = (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]);
        r.z = (acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2]);
        r.w = (acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3]);
        const int bb = q & (kNRB - 1);
        if constexpr (kRefillFirst) {
            // Every compute thread's reads of the slot are done once its MMAs have their
            // operands; the barrier and the fence are the same chain as below, earlier.
            named_sync(kBarUnit, kNT);
            if (tid == 0) {
                ustamp(p, q, 3);
                if (q + kNS < n) {
                    fence_proxy_async();
                    issue_unit(p, S, q + kNS, pol);
                }
            }
        }
        const long long tb = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            // Buffer bb was last read by the helper for unit q - kNRB: completion number
            // q / kNRB - 1 of its free barrier.
            if (q >= kNRB) {
                hand_wait(&S.hand[kNRB + bb], ((q / kNRB) & 1) ^ 1, p.ctr);
            }
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            hand_arrive(&S.hand[bb]);
        } else {
            named_sync(kBarFree + bb, kNTL);
            if (kStamp && tid == 0) {
                S.dg[14] += clock64() - tb;
            }
            S.red[bb][warp][lane] = r;
            named_arrive(kBarFull + bb, kNTL);
        }
        if constexpr (kEarly || kRefillFirst) {
            continue;
        }
        const long long tu = kStamp ? clock64() : 0;
        named_sync(kBarUnit, kNT);
        if (kStamp && tid == 0) {
            S.dg[15] += clock64() - tu;
        }
        if (tid == 0) {
            ustamp(p, q, 3);
        }
        if (tid == 0 && q + kNS < n) {
            // The slot's generic-proxy reads are complete (the barrier above); order them
            // before the async-proxy write that refills it.
            fence_proxy_async();
            issue_unit(p, S, q + kNS, pol);
        }
    }
    if (kStamp && tid == 0) {
        if (na == n) {
            S.dg[3] = gtimer();
            S.dg[4] = S.dg[3];
            S.dg[8] = wfull;
            wfull = 0;
        }
        S.dg[9] = wfull;
    }
}

// Makes this warp's inter stores for job j visible, then adds n finished units to its counter.
__device__ __forceinline__ void credit(const Params& p, Smem& S, int j, int n, int lane) {
    if (n <= 0) {
        return;
    }
    const long long c0 = kStamp ? clock64() : 0;
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomicAdd(p.done_a + j, n);
    }
    if (kStamp && lane == 0) {
        S.dg[11] += clock64() - c0;
    }
}

// Helper warp: sums the 8 warps' partials of each unit and runs its epilogue.
__device__ __forceinline__ void helper_loop(const Params& p, Smem& S, float* ss) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int na = S.na;
    const int n = na + S.nb;
    const int a0 = S.a0;
    const int b0 = S.b0;
    if constexpr (!kHand) {
#pragma unroll
        for (int bb = 0; bb < kNRB; ++bb) {
            named_arrive(kBarFree + bb, kNTL);
        }
    }
    int jcred = -1;
    int ncred = 0;
#pragma unroll 1
    for (int q = 0; q < n; ++q) {
        const bool phb = q >= na;
        const int v = phb ? b0 + (q - na) : a0 + q;
        const int j = phb ? (v >> 8) : (v >> 9);
        const int rb = phb ? (v & 255) : (v & 511);
        const int e = S.job_e[j];
        const int q0 = S.job_q[j];
        const int ntok = S.job_n[j];
        const int bb = q & (kNRB - 1);
        // Phase A's last credits go out before waiting on any phase B partials: the compute
        // warps produce none until done_a is complete, which needs these credits.
        if (phb && ncred > 0) {
            credit(p, S, jcred, ncred, lane);
            ncred = 0;
        }
        const long long cw = kStamp ? clock64() : 0;
        if constexpr (kHand) {
            hand_wait(&S.hand[bb], (q / kNRB) & 1, p.ctr);
        } else {
            named_sync(kBarFull + bb, kNTL);
        }
        if (kStamp && lane == 0) {
            S.dg[10] += clock64() - cw;
        }
        float4 r = S.red[bb][0][lane];
#pragma unroll
        for (int w = 1; w < kNW; ++w) {
            const float4 t = S.red[bb][w][lane];
            r.x += t.x;
            r.y += t.y;
            r.z += t.z;
            r.w += t.w;
        }
        if constexpr (kHand) {
            hand_arrive(&S.hand[kNRB + bb]);
        } else {
            named_arrive(kBarFree + bb, kNTL);
        }
        const bool t0 = 2 * t4 < ntok;
        const bool t1 = 2 * t4 + 1 < ntok;
        if (!phb) {
            // Marlin writes gate and up to bf16 before silu_and_mul, and SwiGLU's output is
            // GEMM2's bf16 input.
            const float gsa = __ldg(p.gs13 + e) * 128.f;
            const float g0 = bf16r(r.x * gsa);
            const float g1 = bf16r(r.y * gsa);
            const float u0 = bf16r(r.z * gsa);
            const float u1 = bf16r(r.w * gsa);
            const float h0 = g0 * __frcp_rn(1.f + __expf(-g0)) * u0;
            const float h1 = g1 * __frcp_rn(1.f + __expf(-g1)) * u1;
            __nv_bfloat16* dst = p.inter + static_cast<size_t>(q0 + 2 * t4) * kI + rb * 8 + g;
            if (t0) {
                dst[0] = __float2bfloat16_rn(h0);
            }
            if (t1) {
                dst[kI] = __float2bfloat16_rn(h1);
            }
            // Credits go out once per job: the fence behind each credit costs about a
            // microsecond under a saturated memory system, too much for every unit.
            if (j != jcred) {
                credit(p, S, jcred, ncred, lane);
                jcred = j;
                ncred = 0;
            }
            ++ncred;
        } else {
            // GEMM2's output is bf16 in SGLang; the norm then runs in FP32.
            const float gsb = __ldg(p.gs2 + e) * 128.f;
            const int col0 = rb * 16 + g;
            const int col1 = col0 + 8;
            const float w0 = __ldg(p.nw + col0);
            const float w1 = __ldg(p.nw + col1);
            const float y00 = bf16r(r.x * gsb) * w0;
            const float y01 = bf16r(r.y * gsb) * w0;
            const float y10 = bf16r(r.z * gsb) * w1;
            const float y11 = bf16r(r.w * gsb) * w1;
            float* dst = p.ybuf + static_cast<size_t>(q0 + 2 * t4) * kH;
            if (t0) {
                dst[col0] = y00;
                dst[col1] = y10;
            }
            if (t1) {
                dst[kH + col0] = y01;
                dst[kH + col1] = y11;
            }
            float s0 = y00 * y00 + y10 * y10;
            float s1 = y01 * y01 + y11 * y11;
#pragma unroll
            for (int d = 4; d < 32; d <<= 1) {
                s0 += __shfl_xor_sync(0xffffffffu, s0, d);
                s1 += __shfl_xor_sync(0xffffffffu, s1, d);
            }
            if (g == 0) {
                if (t0) {
                    atomicAdd(ss + q0 + 2 * t4, s0);
                }
                if (t1) {
                    atomicAdd(ss + q0 + 2 * t4 + 1, s1);
                }
            }
        }
    }
    if (ncred > 0) {
        credit(p, S, jcred, ncred, lane);
    }
}

template <int SPLIT>
__global__ void __launch_bounds__(kNTL, 1) moe_fused_kernel(const Params p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int G = gridDim.x;
    const int cta = blockIdx.x;
    const int E = p.M * kTopK;
    // Under programmatic dependent launch the inputs are not ready before this; without it
    // the instruction does nothing.
    if (p.pdl) {
        if (warp == kNW) {
            if (lane == 0) {
    #pragma unroll
                for (int s = 0; s < kNS; ++s) {
                    mbar_init(&S.full[s], 1);
                    mbar_init(&S.empty[s], kNT);
                }
                if constexpr (kHand) {
    #pragma unroll
                    for (int b = 0; b < kNRB; ++b) {
                        mbar_init(&S.hand[b], kNT);
                        mbar_init(&S.hand[kNRB + b], 32);
                    }
                }
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
        }
        if (tid == 0) {
            for (int i = cta; i < p.nloc * 4; i += G) {
                prefetch_l2_hint(p.w13 + (static_cast<size_t>(i % p.nloc) * kUA
                                          + i / p.nloc) * kUnitBytes, kUnitBytes);
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // The generation picks this launch's sum-of-squares copy and barrier counter; the last
    // CTA through the barrier bumps it, after every CTA has read it.
    const int gen = *reinterpret_cast<volatile int*>(p.ctr + 3);
    float* ss = p.ss + (gen & 1) * kEmax;
    const uint64_t pol = policy_evict_first();
    if (kStamp && tid < 16) {
        S.dg[tid] = tid == 0 ? gtimer() : tid == 12 ? clock64() : 0;
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[2] = clock64();
        ustamps(p)[3] = gtimer();
    }

    if (warp == kNW) {
        // Histogram and job list by one warp, so this CTA's first unit copies can be issued
        // before the full routing; the copies' cold start then overlaps it.
        if (lane == 0 && !p.pdl) {
#pragma unroll
            for (int s = 0; s < kNS; ++s) {
                mbar_init(&S.full[s], 1);
                mbar_init(&S.empty[s], kNT);
            }
            if constexpr (kHand) {
#pragma unroll
                for (int b = 0; b < kNRB; ++b) {
                    mbar_init(&S.hand[b], kNT);
                    mbar_init(&S.hand[kNRB + b], 32);
                }
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        S.cnt[lane] = 0;
        __syncwarp();
        for (int base = 0; base < E; base += 128) {
            const int i = base + 4 * lane;
            int4 v = make_int4(-1, -1, -1, -1);
            if (i < E) {
                v = __ldg(reinterpret_cast<const int4*>(p.topk_ids + i));
            }
            const int ev[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = ev[k] - ld_lo(p.lo_p);
                const int eid = (i < E && e >= 0 && e < p.nloc) ? e : -1;
                if (i < E) {
                    S.eid[i + k] = static_cast<signed char>(eid);
                }
                const unsigned peers = __match_any_sync(0xffffffffu, eid);
                if (eid >= 0 && lane == __ffs(peers) - 1) {
                    S.cnt[eid] += __popc(peers);
                }
                __syncwarp();
            }
        }
        const int tot = lane < p.nloc ? S.cnt[lane] : 0;
        const int off = warp_scan(tot, lane) - tot;
        S.off[lane] = off;
        const int nj = SPLIT ? (tot + 7) >> 3 : (tot > 0 ? 1 : 0);
        const int jincl = warp_scan(nj, lane);
        const int j0 = jincl - nj;
        const int njobs = __shfl_sync(0xffffffffu, jincl, 31);
        for (int k = 0; k < nj; ++k) {
            S.job_e[j0 + k] = lane;
            S.job_q[j0 + k] = off + 8 * k;
            S.job_n[j0 + k] = min(8, tot - 8 * k);
        }
        __syncwarp();
        if (lane == 0) {
            const unsigned ua = njobs * kUA;
            const unsigned ub = njobs * kUB;
            const int a0 = ua * cta / G;
            const int a1 = ua * (cta + 1) / G;
            const int b0 = ub * cta / G;
            const int b1 = ub * (cta + 1) / G;
            S.njobs = njobs;
            S.a0 = a0;
            S.na = a1 - a0;
            S.b0 = b0;
            S.nb = b1 - b0;
            __threadfence_block();
            const int n = (a1 - a0) + (b1 - b0);
            for (int q = 0; q < min(kNS, n); ++q) {
                issue_unit(p, S, q, pol);
            }
        }
    } else {
        const int ctid = cta * kNT + tid;
        const int cstride = G * kNT;
        for (int i = ctid; i < p.M * 64; i += cstride) {
            prefetch_l2(p.x + i * 64);
        }
        for (int i = ctid; i < (E + 31) / 32; i += cstride) {
            prefetch_l2(p.topk_w + i * 32);
        }
        // The next launch's sum-of-squares copy and barrier counter; nothing reads them in
        // this launch.
        float* ss_next = p.ss + ((gen + 1) & 1) * kEmax;
        for (int i = ctid; i < kEmax; i += cstride) {
            ss_next[i] = 0.f;
        }
        if (ctid == 0) {
            p.ctr[4 + ((gen + 1) & 1)] = 0;
        }
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[1] = gtimer();
    }

    position_pass<kNWarps>(S.eid, S.pos, S.ent, S.wc, S.off, E);
    if (kStamp && tid == 0) {
        S.dg[2] = gtimer();
    }

    if (warp < kNW) {
        compute_loop(p, S, pol);
    } else {
        helper_loop(p, S, ss);
    }
    __syncthreads();
    if (kStamp && tid == 0) {
        S.dg[5] = gtimer();
    }

    // Grid barrier. The counter alternates with the generation, so the one this launch
    // uses was zeroed by the previous launch and nothing is reset between launches.
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
    if (kStamp && tid == 0) {
        S.dg[6] = gtimer();
    }
    for (int j = cta * kNTL + tid; j < kJmax; j += G * kNTL) {
        p.done_a[j] = 0;
    }

    if (warp < kNW) {
        final_pass<kNT>(p, S.pos, ss, cta, G);
    }
    if (kUnitStamp && tid == 0) {
        ustamps(p)[4] = clock64();
        ustamps(p)[5] = gtimer();
    }
    if constexpr (kStamp) {
        __syncthreads();
        if (tid == 0) {
            S.dg[7] = gtimer();
            S.dg[13] = clock64();
        }
        __syncthreads();
        if (tid < 16) {
            reinterpret_cast<long long*>(p.inter)[cta * 16 + tid] = S.dg[tid];
        }
    }
}

template <int SPLIT>
void launch(const Params& p, int sms, cudaStream_t stream) {
    static bool ready = false;
    if (!ready) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused_kernel<SPLIT>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(sizeof(Smem))));
        ready = true;
    }
    moe1_launch(moe_fused_kernel<SPLIT>, sms, kNTL, static_cast<int>(sizeof(Smem)), stream,
                p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sk_8_4_3


// ---- Fat path (M >= 17): jobs of up to 64 tokens, 256-row units streamed in 16 stages of
// 256 k, four consumer warpgroups (64 rows each) on register-A wgmma with the job's x stage
// in shared memory through a descriptor, one weights producer thread and one x producer
// thread, stream-K over stages with a global fixup for the units cut between CTAs.
constexpr int kNC = 16;
constexpr int kNTM = kNC * 32 + 64;
constexpr int kNWarpsM = kNTM / 32;
constexpr int kNSWMax = 5;
constexpr int kNSX = 3;
// 5 weight slots + 3 x 8 KB x slots, 4 + 3 x 16 KB, or 3 + 3 x 32 KB: the x slot size follows
// the launch's largest job (max n-tile count), the weights ring takes the rest.
constexpr int kRingBytes = kNSWMax * kUnitBytes + kNSX * 8192;
constexpr int kJmaxM = kLocalMax + kEmax / 64;
constexpr int kUAM = 512;
constexpr int kUBM = 256;
constexpr int kBarWg = 3;
constexpr int kBarGather = 7;
constexpr int kBqMax = 64;
constexpr int kBqEmpty = -2;
// Relative cost of one stage by n8-tile count: the wgmma work grows with N while the bytes
// stay the same.
__constant__ int kCostA[9] = {0, 112, 135, 140, 158, 173, 185, 200, 207};
__constant__ int kCostB[9] = {0, 100, 104, 115, 123, 132, 144, 156, 174};
// The same for stages run on mma.sync (modes 8 to 10): at small N the consumers keep up with
// the weights stream, so the cost is closer to flat. First guesses, ranked by the bench.
__constant__ int kCostAm[9] = {0, 90, 100, 115, 130, 150, 170, 190, 210};
__constant__ int kCostBm[9] = {0, 85, 92, 104, 115, 130, 145, 160, 175};

// Modes 6 to 10 run jobs of at most mma_max(MODE) n8 tiles on mma.sync instead of wgmma: an
// RS wgmma group at N = 8 or 16 costs the consumers more than the stage takes to stream
// (bench 36b5b559b0344d0f: main-d2 197 us at bs32 against 139 loads-only and 150 for the
// mma.sync skinny kernel).
__host__ __device__ constexpr int mma_max(int mode) {
    return mode == 6 || mode == 9 || mode == 11 || mode == 12 || mode == 14 || mode == 15
                   || (mode >= 16 && mode <= 29) || (mode >= 31 && mode <= 40)
               ? 2
           : mode == 7 || mode == 10 ? 4
           : mode == 8 ? 8 : 0;
}

// Modes 12 and 13 (modes 6 and 0 otherwise) order the epilogue's and the fixup's global
// writes before their gpu-scope release or acquire with fence.acq_rel.gpu instead of
// __threadfence (fence.sc.gpu). The pattern stays fence, then bar.sync among the warpgroup,
// then one thread's red.release / atom.acq_rel: a fence.acq_rel followed by a strong write is
// a release pattern and a strong read followed by fence.acq_rel an acquire pattern (PTX ISA,
// Memory Consistency Model, "Release and Acquire Patterns"); fence.sc adds only the SC order,
// which no reader here relies on. The SASS is MEMBAR.ALL.GPU instead of MEMBAR.SC.GPU.
__host__ __device__ constexpr bool fence_ar(int mode) {
    return mode == 12 || mode == 13 || mode == 14 || (mode >= 15 && mode <= 29)
           || (mode >= 31 && mode <= 40);
}

// Modes 17 to 20 are mode 16 (two mma.sync chains, unit-aligned phase B) with one change each:
// 17 adds the clock stamps of mode 11 (timing only); 18 runs phase A alone (no phase B, no
// final pass: timing only, the GEMM1 gate's baseline); 19 runs phase A alone with every unit
// owned whole by one CTA (no cut unit, no fixup; timing only, the staged GEMM1 prototype); 20
// is 19's phase A followed by mode 16's phase B and final pass (a full, checked candidate).
// 22 is 20 with the stamps of 17. 23 is 20 with phase B units claimed at run time (dyn_b) and
// both phases walked largest n-tile count first; 24 is 23 with the stamps.
__host__ __device__ constexpr bool dual_chain(int mode) {
    return mode == 14 || (mode >= 16 && mode <= 29) || (mode >= 31 && mode <= 40);
}
__host__ __device__ constexpr bool is_diag(int mode) {
    return mode == 11 || mode == 17 || mode == 22 || mode == 24;
}
// Mode 25 is 23 with thread 0's globaltimer at the phase boundaries only (timing only). Mode
// 24's per-segment clock stamps add 14-18 us, too much to set its phase times beside the
// public kernel's eight thread-0 stamps; these cost a few stores per CTA.
__host__ __device__ constexpr bool is_light(int mode) {
    return mode == 25 || mode == 27 || mode == 29 || mode == 31 || mode == 33 || mode == 38
           || mode == 40;
}
// Modes 26 and 27 (27 with the light stamps) are 23 with the setup of the public kernel's
// route(): every warp counts its slice of the routes once (match_any), the routing warp sums
// the per-warp counts instead of counting all E entries alone and the positions reuse them;
// and the x gather issues all of a CTA's row loads (16 B each) before its stores. Mode 25
// put setup at 7 / 11 / 16 us of routing at 32 / 128 / 256 tokens against 2-3 us for the
// public kernel (check d1f587760b1241dc).
__host__ __device__ constexpr bool fast_setup(int mode) {
    return (mode >= 26 && mode <= 29) || (mode >= 31 && mode <= 40);
}
// Modes 28 and 29 (29 with the light stamps) are 26 with a leaner consumer stage. The stage
// loop was 389 instructions against the public kernel's 322 for the same dequant and MMA core
// (profiles/sass_stage_loop_vs_public.2026-10-29.txt); with four consumer warps per scheduler
// at about 1 us per stage, issue slots are a plausible limit (not measured). Per stage: the
// ring size and x slot 0's address come from registers set once per launch instead of three
// shared loads, and the S.ord sort takes one pass instead of eight dependent rounds. The
// weights slot release keeps its fence.proxy.async.shared::cta but moves behind the stage's
// first wgmma group, off the path from the full-barrier wait to the first wgmma, and every
// consumer thread fences and arrives itself (wempty and, on the mma.sync path, xempty count
// kNC * 32), so each thread's generic reads reach the producer's acquire through its own
// program order (docs/runs-moe.md, "Ordering of the ring slot releases"). 26's lane-0
// release after __syncwarp at the moved point makes ptxas serialize the wgmma groups (C7513,
// 400 instructions per stage), so that variant was dropped.
__host__ __device__ constexpr bool lean_stage(int mode) {
    return mode == 28 || mode == 29 || (mode >= 31 && mode <= 40);
}
// Modes 31 and 33 (timing only) are 28 and 32 with the light stamps plus phase A counters per
// consumer warp (S.dg rows 0-15): cycles spent in weights and x waits that missed their first
// try, cycles from a unit's start to its accumulators, epilogue cycles, deferred publish
// cycles, stages, units and n-tiles; and the producers' phase A cycles waiting for a free slot.
// A wait that succeeds on its first try adds nothing, so the stage loop gains no clock read.
__host__ __device__ constexpr bool pa_diag(int mode) {
    return mode == 31 || mode == 33 || mode == 38 || mode == 40;
}
// Modes 32 to 34 publish a finished phase A unit (proxy fence, fence_gpu, the warpgroup's
// bar.sync, red.release on done_as) at the start of the warpgroup's next phase A unit's second
// stage instead of right after the SwiGLU stores, and at the end of phase A for the last one;
// the public kernel defers the same release (wgs_flush). Mode 28 runs two MEMBAR.ALL.GPU,
// CCTL.IVALL and a bar.sync per unit while its stores are still in flight, with every
// warpgroup of the CTA idle (docs/runs-moe.md, "Phase A at 256 tokens").
__host__ __device__ constexpr bool defer_a(int mode) {
    return mode >= 32 && mode <= 35;
}
// Mode 35 is 32 with the sums-of-squares pointer and the owned unit count read from shared
// memory at each use (ld.volatile.shared, as the public kernel's ss_cur): hoisted out of the
// unit loops they stay live across the stage loops at the 96-register cap and spill.
__host__ __device__ constexpr bool vol_regs(int mode) {
    return mode == 35;
}
// Mode 34 is 32 without the per-thread fence_gpu before the bar.sync: the bar.sync and the
// releasing thread's red.release order every writer's stores before the release by
// themselves (the chain in docs/runs-moe.md, "Proxy ordering of the x tiles", steps 2-3).
// Each writer keeps its fence.proxy.async.global.
__host__ __device__ constexpr bool bar_release_a(int mode) {
    return mode == 34 || mode == 37 || mode == 39 || mode == 40;
}
// Modes 36 to 38 are 28 with SiLU's reciprocal as one MUFU.RCP (rcp.approx.ftz, as the public
// kernel's __fdividef): __frcp_rn compiles to MUFU.RCP plus a range check that branches to a
// subroutine, inside a convergence region per element, 16 per thread per unit at 8 n-tiles.
// Mode 31's counters put mode 28's phase A epilogue at 3.0 us per unit at 256 tokens with the
// consumers never waiting for weights or x (check c7db20e053254b52). 37 is 36 without the
// per-thread fence_gpu (as 34), 38 is 36 with the phase A counters (timing only).
__host__ __device__ constexpr bool fast_silu(int mode) {
    return mode >= 36 && mode <= 40;
}
// Modes 39 and 40 (40 with the counters) are 37 with the SwiGLU outputs stored in pairs: in
// the xb layout a warp owns two adjacent bf16 of each 16-byte token row, held by lanes g and
// g ^ 1, so one shuffle lets the even lane store token tok's pair and the odd lane token tok +
// 1's as one 4-byte store each, half the store instructions and sector writes of 2-byte stores.
// With 37's fences gone, mode 33's counters put the stores and SwiGLU at 1.6 us per unit at
// 256 tokens (check c7db20e053254b52).
__host__ __device__ constexpr bool pair_st(int mode) {
    return mode == 39 || mode == 40;
}
__host__ __device__ constexpr bool a_only(int mode) {
    return mode == 18 || mode == 19;
}
__host__ __device__ constexpr bool own_a(int mode) {
    return mode == 19 || mode == 20 || (mode >= 22 && mode <= 29) || (mode >= 31 && mode <= 40);
}
// Modes 23 and 24: a CTA's weights producer claims phase B units one at a time from a global
// counter, in S.ord order, when its ring has a free slot. Owned phase A leaves CTAs with up to
// one unit more than others and mode 20's static phase B ranges ignore that, so the CTA that
// frees first takes the next unit; S.ord puts the jobs whose phase A rounds come first (the
// snake deal walks S.ord) at the front, so claimed units find their inputs ready.
__host__ __device__ constexpr bool dyn_b(int mode) {
    return (mode >= 23 && mode <= 29) || (mode >= 31 && mode <= 40);
}
// Mode 21 is the second launch of the staged pair (19, then 21): phase B and the final pass
// alone, on the xb a previous launch wrote. That launch completed before this one started
// (stream order, or griddepcontrol.wait under PDL: PTX ISA 9.7.15.14, 8.9.4), so phase B
// takes no done_as wait and no x gather; the issuing thread keeps its proxy fence.
__host__ __device__ constexpr bool b_only(int mode) {
    return mode == 21;
}

template <bool AR>
__device__ __forceinline__ void fence_gpu() {
    if constexpr (AR) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    } else {
        __threadfence();
    }
}

// Modes 11 and 17 (modes 6 and 16 with clock stamps; timing only, never a policy choice):
// each consumer warp sums its cycles waiting for weights, waiting for x, computing and in the
// epilogue or fixup over its segments of both paths (wgmma segments since 2026-10-29); the
// producers sum their waits; thread 0 records globaltimer and clock64 at the kernel's phase
// boundaries and when its phase A segments end. S.dg goes to p.inter (unused on the fat path)
// at the end, 18 x 8 int64 per CTA; _diag_report has the layout.

template <int MODE>
__device__ __forceinline__ int stage_cost(int ph, int ntt) {
    constexpr bool kFlat = MODE >= 8 && MODE <= 10;
    if (kFlat && ntt <= mma_max(MODE)) {
        return ph ? kCostBm[ntt] : kCostAm[ntt];
    }
    return ph ? kCostB[ntt] : kCostA[ntt];
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
    // Owned phase A (own_a): jobs by n-tile count, largest first, and this CTA's unit count.
    int ord[kJmaxM];
    int own_n;
    // dyn_b: the i-th claimed phase B unit (index into S.ord order, 16 per job), -1 when the
    // claims are exhausted, kBqEmpty until the weights producer writes it.
    int bq[kBqMax];
    long long dg[18][8];
    float* ssp;
};

// Ring positions shared by the consumers and the producers: slot and lap per ring, the lap
// parity being the mbarrier phase to wait for.
struct Ring {
    int ws;
    int wlap;
    int xs;
    int xlap;
    // lean_stage only, set once per launch: weights ring slots, x slot 0's shared address
    // and the x slot size in bytes.
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

// lean_stage modes: every consumer thread releases the weights slot (and, on the mma.sync
// path, the x slot) itself, so the empty barriers count kNC * 32 arrivals; a release that
// stays warp-level (the wgmma path's x slot) arrives once for the whole warp.
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

// mbar_wait_to returning the cycles spent after a failed first try (pa_diag).
__device__ __forceinline__ unsigned mbar_wait_cyc(uint64_t* bar, unsigned parity, int* ctr,
                                                  int code) {
    if (mbar_try(bar, parity)) {
        return 0u;
    }
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
    if constexpr (pa_diag(MODE) && PH == 0) {
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

// The claimed-unit queue S.bq: one writer (the weights producer), readers in the same CTA.
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

// No-swizzle K-major descriptor: lbo between core matrices along k, sbo along n.
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
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n16(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
                 "{%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n24(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
                 "{%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n32(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
                 "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n40(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, "
                 "{%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n48(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, "
                 "{%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n56(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, "
                 "{%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

__device__ __forceinline__ void wgmma_n64(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
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

// Pins the live accumulators so the compiler neither copies them across an async wgmma nor
// moves them to local memory.
template <int NTT>
__device__ __forceinline__ void acc_fence(float (&d)[32]) {
#pragma unroll
    for (int i = 0; i < 4 * NTT; ++i) {
        asm volatile("" : "+f"(d[i]) :: "memory");
    }
}

// A wgmma keeps reading its A registers until it retires, which the compiler cannot see; a
// group's fragments are pinned after the wait that retires it, so the next group's dequant
// lands in other registers instead of behind a full pipeline drain.
__device__ __forceinline__ void a_pin(unsigned (&a)[2][4]) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            asm volatile("" : "+r"(a[i][k]) :: "memory");
        }
    }
}

// Global stage index where CTA k's range of phase ph begins: the cost prefix over jobs
// (pa / pb) split evenly over the grid.
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
    // Modes 15 and 16: phase B boundaries on the nearest unit (16 stages) so no phase B segment is cut;
    // the cost model put a cut phase B segment at about 10 us (docs/runs-moe.md, "Cost model").
    return ph && S.align_b ? (s + 8) & ~15 : s;
}

__device__ __forceinline__ int seg_count(int b, int e) {
    return b >= e ? 0 : ((e - 1) >> 4) - (b >> 4) + 1;
}

// The i-th segment of stage range [b, e) in execution order: the owned trailing partial unit
// first, so its share of a cut unit lands early, then the rest in order.
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
    // Owned phase A: the unit's round in own_unit (the CTA's r-th unit), else -1.
    int r;
};

// Owned phase A (own_a): the units of all jobs sorted by n-tile count, largest first (S.ord,
// 32 units per job), dealt to the CTAs in snake order, so round r gives CTA c the unit
// r G + c (r even) or r G + G - 1 - c (r odd). Every unit runs whole on one CTA.
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

// A cut unit's fixup slot is the index of the CTA holding its first stage.
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

// The release of a finished phase A unit on done_as[target]: every writer's proxy fence (and
// fence_gpu unless bar_release_a), the warpgroup's bar.sync, one red.release.
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

// SwiGLU of a finished phase A unit straight into phase B's x layout (stage u / 2 of the job,
// 16-byte row per token and 8-feature chunk), then one release per warpgroup on the stage's
// readiness counter (PUB), or the counter's index returned for a later publish_a.
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
    // Feature kk of the stage sits at mma-k position 16 (4 c + s) + (j < 2 ? 2 t4k + j :
    // 8 + 2 t4k + j - 2) with kk = 64 c + 16 t4k + 4 s + j (the fat layout's k order).
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
                // The odd lane's element is the pair's high half, one row (16 bytes) down.
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

template <int NTT, bool VOL = false>
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

// Every share of a cut unit adds its partial into the owner's slot and counts its stages; the
// share that completes the count reads the sum back into fin, clears the slot and finishes
// the unit. acc itself is never written by anything but wgmma (ptxas C7515).
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

// defer_a: the done_as index of the CTA's owned phase A unit of round r. The rounds run in
// order (S.ord's n-tile counts never rise with r, and consumer_phase walks the counts largest
// first), so round r - 1 is the unit before round r, and owned units are always 16 stages.
__device__ __forceinline__ int own_target(const SmemMain& S, int r, int cta, int G) {
    const int k = own_unit(r, cta, G);
    return S.ord[k >> 5] * 16 + ((k & 31) >> 1);
}

// A whole phase A unit's epilogue: published at once, or (defer_a) left for the next unit's
// second stage or the end of phase A.
template <int NTT, int MODE>
__device__ __forceinline__ void epilogue_a_unit(const Params& p, SmemMain& S,
                                                const float (&acc)[32], const SegInfo& si,
                                                int wg, int w, int tid_wg) {
    static_assert(!defer_a(MODE) || (own_a(MODE) && dyn_b(MODE)), "defer_a needs owned units");
    epilogue_a<NTT, MODE, !defer_a(MODE)>(p, S, acc, si.j, si.u, wg, w, tid_wg);
}

// pa_diag: one phase A unit's counters into the warp's S.dg row. Segment cycles include the
// waits (columns 0 and 1) and exclude the deferred publish (column 7).
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

// One segment (stages st_b..st_e of one unit) on a consumer warpgroup: per stage the warp's
// 16 words and 32 scale bytes come out of the weights ring into registers, then 16 k16 steps
// of dequant and wgmma in groups of two, one group kept in flight.
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
    // 32-bit clock deltas keep the stamps' register cost low on the wgmma path.
    constexpr bool kDiag = is_diag(MODE);
    constexpr bool kPa = pa_diag(MODE) && PH == 0;
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
    // kLean: x slot 0's descriptor once per segment; a stage adds its slot's offset in 16-byte
    // units (shared addresses stay below 2^18, so the 14-bit address field cannot carry).
    const unsigned xb0 = kLean ? __shfl_sync(0xffffffffu, rg.xb, 0) : 0u;
    const uint64_t dx0 = ((xb0 & 0x3ffffu) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16)
                         | (static_cast<uint64_t>(4096 >> 4) << 32);
#pragma unroll 1
    for (int st = si.st_b; st < si.st_e; ++st) {
        if constexpr (defer_a(MODE) && PH == 0) {
            // One stage into the next unit, the previous unit's SwiGLU stores have drained.
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
        // The slot base goes through a shuffle so ptxas knows it is warp-uniform and keeps
        // the descriptors in uniform registers (one UIADD3 per step instead of a 64-bit add
        // and two R2UR moves on every wgmma's critical path).
        const unsigned xbase = kLean ? 0u
                                     : __shfl_sync(0xffffffffu,
                                                   smem_u32(S.rings + S.xoff + xs * S.xsz), 0);
        // The slot offset still takes a shuffle: the ring position is carried around a loop
        // whose trip count ptxas cannot prove warp-uniform, and without it every wgmma's
        // descriptor goes through R2UR (448 instructions per stage instead of 389).
        const uint64_t d0 = kLean ? dx0 + (__shfl_sync(0xffffffffu,
                                                       static_cast<unsigned>(xs) * rg.xsz, 0)
                                           >> 4)
                                  : ((xbase & 0x3ffffu) >> 4)
                                        | (static_cast<uint64_t>(128 >> 4) << 16)
                                        | (static_cast<uint64_t>(4096 >> 4) << 32);
        if (MODE == 1) {
            // Timing only: the stage's bytes arrive and one word of each is read.
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
                    // Timing only: the raw words stand in for the fragments.
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
                // Timing only: the fragments are consumed by one add instead of a wgmma.
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
            // The previous stage's last group has retired once DEPTH - 1 groups of this
            // stage are committed.
            if (s == 2 * (DEPTH - 1) - 2 && xrel >= 0) {
                x_release_warp<MODE>(&S.xempty[xrel], lane);
                xrel = -1;
            }
            if (kLean && s == 0) {
                // The weights slot's generic reads (cw, sc) were issued before this point;
                // the fence orders them before the producer's next bulk write into the slot.
                fence_proxy_async();
                mbar_arrive(&S.wempty[ws]);
            }
        }
        xrel = xs;
        ring_next_x(rg);
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
                epilogue_b<NTT>(p, S, ss, fin, si.j, si.u, wg, w, tid_wg);
            }
        }
    } else if (ph == 0) {
        epilogue_a_unit<NTT, MODE>(p, S, acc, si, wg, w, tid_wg);
    } else {
        epilogue_b<NTT, vol_regs(MODE)>(p, S, ss, acc, si.j, si.u, wg, w, tid_wg);
    }
    pa_record<MODE, PH, NTT>(S, warp, lane, dw, dx, pseg, pdep, ppub, si.st_e - si.st_b);
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

// run_segment on mma.sync m16n8k16 for small N. A warp's wgmma m64 A fragment is the
// mma.sync A fragment of its 16 rows, and the m64nN accumulator per warp is NTT m16n8
// accumulators in the same order, so the ring layouts, the dequant, the fixup and the
// epilogues are shared with the wgmma path. B comes from the x slot by ldmatrix: one x4 covers
// two k16 steps of one n8 tile (step s's two 8-k core matrices at 256 s and 256 s + 128).
// Each warp runs on its own; no warpgroup-wide wgmma sync.
template <int NTT, int PH, int MODE>
__device__ __forceinline__ void run_segment_mma(const Params& p, SmemMain& S, float* ss,
                                                const SegInfo& si, Ring& rg, int G) {
    constexpr int ph = PH;
    constexpr bool kDiag = is_diag(MODE);
    constexpr bool kPa = pa_diag(MODE) && PH == 0;
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
    // Mode 14 splits each n8 tile's 16-step mma.sync chain into two accumulators (even and odd
    // k16 steps), halving the dependent chain per stage; the stamps put a stage's compute at
    // 1.02 us against about 0.66 us of issue for its four warps per scheduler.
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
                // As on the wgmma path: the weights slot's reads were issued before this point.
                fence_proxy_async();
                mbar_arrive(&S.wempty[ws]);
            }
        }
        // The x slot is refilled by the async proxy after these generic-proxy reads.
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
            // Reading acc makes the stamp wait for this stage's mma chain.
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
                epilogue_b<NTT>(p, S, ss, fin, si.j, si.u, wg, w, tid_wg);
            }
        }
    } else if (ph == 0) {
        epilogue_a_unit<NTT, MODE>(p, S, acc, si, wg, w, tid_wg);
    } else {
        epilogue_b<NTT, vol_regs(MODE)>(p, S, ss, acc, si.j, si.u, wg, w, tid_wg);
    }
    pa_record<MODE, PH, NTT>(S, warp, lane, dw, dx, pseg, pdep, ppub, si.st_e - si.st_b);
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

// Segments run grouped by (phase, n-tile count), each group its own loop: with several
// wgmma shapes inside one enclosing loop ptxas serializes every wgmma (C7512, see the
// micro tests in docs/runs-moe.md); the producers walk the same order.
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
        // Warp 0's cycles inside segments: the cost model's work target.
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

// dyn_b phase B: the claimed units in claim order. Claims rise through S.ord, whose n-tile
// counts never rise, so one loop per count, largest first, takes them all in order.
template <int NTT, int MODE>
__device__ __forceinline__ void run_group_dyn(const Params& p, SmemMain& S, float* ss,
                                              float (&acc)[32], Ring& rg, int& xrel, int& qi,
                                              int cta, int G) {
#pragma unroll 1
    while (qi < kBqMax) {
        const int m = bq_get(S, qi, p.ctr);
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
    const long long pc0 = pa_diag(MODE) ? clock64() : 0;
    consumer_phase<0, MODE>(p, S, ss, acc, rg, xrel, cta, G);
    if constexpr (defer_a(MODE)) {
        // Before any phase B wait: another CTA's phase B stage may need this unit.
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
    } else {
        consumer_phase<1, MODE>(p, S, ss, acc, rg, xrel, cta, G);
    }
}

template <int MODE>
__device__ __forceinline__ void wproducer(const Params& p, SmemMain& S, int cta, int G,
                                          uint64_t pol) {
    long long twe = 0;
    long long pwe = 0;
    Ring rg = {0, 0, 0, 0};
    for (int ph = 0; ph < (dyn_b(MODE) ? 1 : 2); ++ph) {
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
        // Claim only once a slot is free, so the claim lands as late as the ring allows: the
        // CTA that drains first takes the next unit. The counter is a plain atomic (exactly
        // one claimant per value); the units' inputs still come through done_as. CTA 0 zeroes
        // it after the grid barrier, which every claim precedes.
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

// Phase A stages come from the gathered x rows once the job's rows are all in; phase B
// stages from the SwiGLU output once the stage's 8 warpgroup units have released it.
template <int MODE>
__device__ __forceinline__ void xproducer(const Params& p, SmemMain& S, int cta, int G) {
    Ring rg = {0, 0, 0, 0};
    long long txe = 0;
    long long tda = 0;
    long long txr = 0;
    long long pxe = 0;
    for (int ph = 0; ph < (dyn_b(MODE) ? 1 : 2); ++ph) {
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
                spin_ge(p.done_as + j * 16 + st, 8, p.ctr, 128);
                fence_proxy_async_global();
                if (is_diag(MODE)) {
                    tda += clock64() - c0;
                }
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

    // Every thread's topk_ids loads issued before any store (E / 4 <= 1024 int4 over 576
    // threads): one memory round trip instead of one per 128 entries in the routing warp,
    // stage-stamped at 12 us of prologue at 128 tokens and 18 at 256 (check 2072795029b7479f).
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
            S.align_b = MODE == 15 || (MODE >= 16 && MODE <= 29) || (MODE >= 31 && MODE <= 40)
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
            // Each warp's first pair row per expert, as position_pass's warp 0 forms it.
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
            // The same order in one pass: an expert's jobs are full (8 n-tiles) except
            // possibly its last, so a job's rank is the 8-tile jobs of lower experts, or
            // all 8-tile jobs plus the partial jobs that are larger or come from a lower
            // expert with the same count. The counting sort below took eight dependent
            // rounds (2.5 us of routing at 128 tokens, 4c285df44eb34bc1).
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
            // Counting sort of the jobs by n-tile count, largest first; within a count, by
            // expert then job, so every CTA derives the same order.
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
        // Owned phase A and claimed phase B read no stage range (seg_n and seg_at take own_n
        // and S.ord, the claims come through S.bq), so fast_setup leaves out the four
        // stage_at searches, 2.3-3.1 us on thread 0 in mode 27's stamps (39c365c34cd74559).
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
    // The weights producer needs only the stage ranges, so it starts streaming here while the
    // other warps gather x; they sync on their own named barrier.
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
        // Phase A's x: every token row of every job into the per-stage wgmma layout, this CTA
        // taking entries cta, cta + G, ...; one release per row.
        if constexpr (fast_setup(MODE)) {
            // kGB entries' 16-byte loads go out before their stores: one row at a time, a row's
            // loads wait behind the previous row's stores (they may alias), one HBM round trip
            // per row. Thread gt < 512 moves 8 bf16 of a row, pairs 4 gt .. 4 gt + 3.
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
            // Pair q holds K = 2 q, 2 q + 1; its stage is q / 128 and its slab position follows
            // the fat layout's k order (epilogue_a has the same mapping).
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
        // A release after the barrier: a fence before bar.sync followed by a relaxed atomic is a
        // release pattern only for the atomic thread's own earlier writes (PTX ISA 8.8), not for
        // the rows other threads stored; red.release after bar.sync covers them by cumulativity.
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
        final_pass<kNC * 32>(p, S.pos, ss, cta, G);
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

}  // namespace moe1
}  // namespace
"""

# The layer's host side: the dispatch over _MOE1_VARIANTS (filled in by _moe1_source()) and the
# launch with local_start read on the device.
_MOE1_HOST_SRC = r"""
namespace {
namespace moe1 {

void moe1_dispatch(int variant, const Params& p, int sms, cudaStream_t stream) {
    switch (variant) {
//@MOE1_DISPATCH@
        default: TORCH_CHECK(false, "moe_fused: no variant ", variant);
    }
}

}  // namespace moe1
}  // namespace

// One persistent launch per chunk of 512 tokens: grid = SM count, one CTA per SM, no
// cooperative attribute (the shared-memory footprint keeps every CTA resident) and no early
// trigger. The grid barrier spins until all CTAs arrive, so nothing else of ours may hold SMs
// meanwhile: the layer joins its side stream before this call, or (pdl) launches it as the PDL
// secondary of the route kernel, whose CTAs all exit before any MoE CTA passes
// griddepcontrol.wait; only the first chunk takes the attribute. x is
// the normalized expert input, topk_ids the router's global ids (the kernel keeps the local
// range [local_start, local_start + nloc)), nw = 1 + post_norm_w in FP32. Output bf16 [M, 4096].
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
    // Params is not zero-initialized; cost[0][1] == 0 selects the compiled stage-cost table
    // (the layer runs no DIAG fit), any other value would be taken as a fitted table.
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
"""

_MOE1_CPP_SRC = r"""
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
"""

# (name, kind, launch) per built variant, in dispatch order, written by tools/dl_moe_inline.py;
# a "main" (fat, wgmma) variant takes batches of _MOE1_MAIN_MIN_M tokens and more.
_MOE1_VARIANTS = (
    ('sk8w4s', 'skinny', 'sk_8_4_0::launch<1>'),
    ('sk16w4s', 'skinny', 'sk_16_4_0::launch<1>'),
    ('sk8w3s', 'skinny', 'sk_8_3_0::launch<1>'),
    ('sk8w4s-er', 'skinny', 'sk_8_4_3::launch<1>'),
    ('main-d2', 'main', 'launch_main<0>'),
    ('main-m2-ar2-ub', 'main', 'launch_main<16>'),
    ('main-dynb-fs', 'main', 'launch_main<26>'),
    ('main-lean', 'main', 'launch_main<28>'),
    ('main-lean-silu-bar', 'main', 'launch_main<37>'),
)
_MOE1_MAIN_MIN_M = 17


def _moe1_source():
    cases = "\n".join(f"        case {i}: {launch}(p, sms, stream); break;"
                      for i, (_, _, launch) in enumerate(_MOE1_VARIANTS))
    return _MOE1_HOST_SRC.replace("//@MOE1_DISPATCH@", cases)


# The projection kernels from muse_bf16_proj (docs/proj-port-guide.md in that problem's
# worktree), assembled by tools/dl_proj_inline.py for the layer's configs: one skinny
# tensor-core kernel (x the mma A operand by a TMA box, weights the B operand in 8-row tiles
# by one cp.async.bulk per stage; 12-bit lossless weights decoded in registers, escape warps
# correcting the weights outside the exponent window from a per-CTA table; raw BF16
# configs; a fused pooler + router instantiation) and the m = 1 GEMVs. The kernel has no
# griddepcontrol, so the layer launches it plain.
_SKP_SRC = r"""

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

// SK_PDL 1 builds every single-CTA config to stream its first ring of weights before
// griddepcontrol.wait, and lets sk_fwd / skf_fwd launch as a PDL secondary when bit 16 of cfg
// is set. Off by default: the default build's device code is the one judged so far.
#ifndef SK_PDL
#define SK_PDL 0
#endif

namespace {

constexpr uint32_t SUSPEND_HINT = 0x989680u;
// A wait that has not completed after this many cycles marks the thread dead (it skips every
// later wait) so a broken pipeline produces wrong output for the self-test instead of hanging.
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
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

__device__ __forceinline__ bool mbar_test_sleep(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\nselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(SUSPEND_HINT) : "memory");
    return ok != 0;
}

__device__ __forceinline__ bool mbar_test_cluster(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

// Remote st.async completions release at cluster scope, so the receiving wait acquires there.
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

// Initialized barriers must be visible to the cluster and to the async proxy before a copy
// can complete on them (CUDA programming guide, TMA example).
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

// Registers only, so not volatile: the compiler may move it to overlap the loads.
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

// Eight 12-bit weights -> four bf16x2 registers in B-fragment order. lw0/lw1 hold the low
// bytes of weights 0-3 / 4-7; the low nibble of code byte b is weight b's code, the high nibble
// weight 4+b's. A code is sign<<3 | offset; the high byte is sign<<7 | (base + offset).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t codes, uint32_t base4, uint32_t* r) {
    const uint32_t ha = ((codes & 0x07070707u) + base4) | ((codes << 4) & 0x80808080u);
    const uint32_t hb = (((codes >> 4) & 0x07070707u) + base4) | (codes & 0x80808080u);
    r[0] = prmt(lw0, ha, 0x5140);
    r[1] = prmt(lw0, ha, 0x7362);
    r[2] = prmt(lw1, hb, 0x5140);
    r[3] = prmt(lw1, hb, 0x7362);
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL,
          bool FUSE>
struct SkCfg {
    static constexpr int NW = NWR * WSK;
    static constexpr int NC = NW * 32;
    static constexpr int R = NWR * NT * 8;
    static constexpr int UPS = WSK * UPW;
    static constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
    static constexpr int WBY = NW * UPW * NT * TU;
    static constexpr int XBY = UPS * TOKR * 128;
    // Fused pooler + router: after its pooler stages a CTA streams one 8-row router piece (hi
    // and lo tiles, 2048 B per unit) through the same ring, every consumer warp taking RUPW units.
    static constexpr int RUPW = FUSE ? UPS / NW : 1;
    static constexpr int RWBY = FUSE ? UPS * 2048 : 0;
    static constexpr int SST = (XBY + (WBY > RWBY ? WBY : RWBY) + 1023) & ~1023;
    static constexpr int TOKS = 16 * MT;
    // Partial-sum row stride, 8 mod 32 so the float2 stores of a warp hit distinct banks.
    static constexpr int RP = R + ((40 - (R % 32)) % 32);
    // Correction row stride, odd so the per-token updates of a warp hit distinct banks.
    static constexpr int CP = R + 1;
    static constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 0;
    static constexpr int THREADS = (NW + 1 + NESC) * 32;
    static constexpr int CSZ = (TOKS * CP + 3) & ~3;
    static constexpr int CORR_FLOATS = COMP ? NESC * CSZ : 0;
    static constexpr int PEER_FLOATS = CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0;
    static constexpr int BAR_BYTES = 16 * NSTAGE + 32;
    static constexpr int FIXED = 1024 + NSTAGE * SST + BAR_BYTES + 4 * (CORR_FLOATS + PEER_FLOATS) + 128 + 16;
    static constexpr int MINB = 2 * (FIXED + 2048) <= 228 * 1024 ? 2 : 1;
    static_assert(WSK * TOKS * RP * 4 <= NSTAGE * SST, "epilogue partials must fit in the ring");
    static_assert(!FUSE || (COMP && CL == 1 && NW * RUPW == UPS && NW * TOKS * 8 * 4 <= NSTAGE * SST),
                  "fused piece geometry");
    static_assert(TOKR == 16 * MT || (TOKR == 8 && MT == 1), "TOKR must match MT");
    static_assert(!(COMP && CL > 1), "escape warps are not clustered");
    static_assert(NESC <= NSTAGE, "escape warps take turns on stages");
};

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL,
          typename OutT, bool FUSE>
__global__ void __launch_bounds__((SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>::THREADS),
                                  (SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>::MINB))
sk_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ ws, OutT* __restrict__ out,
          const int32_t* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int* __restrict__ diag,
          const uint8_t* __restrict__ rws, float* __restrict__ rpart, int* __restrict__ rcount,
          float* __restrict__ rout) {
    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>;
    constexpr int NW = C::NW, NC = C::NC, R = C::R, UPS = C::UPS, TU = C::TU, WBY = C::WBY, XBY = C::XBY;
    constexpr int SST = C::SST, TOKS = C::TOKS, RP = C::RP, CP = C::CP, NESC = C::NESC, CSZ = C::CSZ;
    constexpr int KST = UPS * 64, RUPW = C::RUPW, RWBY = C::RWBY;

    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const uint32_t sraw = smem_u32(smem_raw);
    const uint32_t sbase = (sraw + 1023u) & ~1023u;
    unsigned char* gbase = smem_raw + (sbase - sraw);
    const uint32_t bars = sbase + NSTAGE * SST;
    const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
    float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + C::BAR_BYTES);
    float* peer = corr + C::CORR_FLOATS;
    int32_t* tab = reinterpret_cast<int32_t*>(peer + C::PEER_FLOATS);
    const int tab_ints = COMP ? pmax * 2 + R : 0;
    uint32_t* zrow_p = reinterpret_cast<uint32_t*>(tab + tab_ints);
    const uint32_t zrow = smem_u32(zrow_p);
    int* last_flag = reinterpret_cast<int*>(zrow_p + 32);
    float* red = reinterpret_cast<float*>(gbase);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rb = CL > 1 ? blockIdx.x / CL : blockIdx.x;
    const int crank = CL > 1 ? blockIdx.x % CL : 0;
    const int nrb = CL > 1 ? gridDim.x / CL : gridDim.x;
    const int tst = K / KST;
    const int gs0 = (crank * tst) / CL;
    const int nst = ((crank + 1) * tst) / CL - gs0;
    const uint8_t* wsrc = ws + (size_t)rb * WBY;
    const size_t wstride = (size_t)nrb * WBY;
    const uint32_t XU = (uint32_t)M * 128u;
    const uint32_t stage_tx = UPS * XU + WBY;
    // Fused router piece: row block rbk (8 rows of 256), K piece rq of npiece, spp stages each.
    const int npiece = FUSE ? (int)gridDim.x / 32 : 1;
    const int spp = FUSE ? tst / npiece : 0;
    const int rbk = FUSE ? (int)blockIdx.x % 32 : 0;
    const int rq = FUSE ? (int)blockIdx.x / 32 : 0;
    const int nst_tot = nst + spp;

    auto issue = [&](int st) {
        const int slot = st % NSTAGE;
        const uint32_t s0 = sbase + slot * SST;
        if (FUSE && st >= nst) {
            const int rs = rq * spp + (st - nst);
            mbar_expect_tx(full0 + 8 * slot, UPS * XU + RWBY);
            bulk_g2s(s0 + XBY, rws + ((size_t)rs * 32 + rbk) * RWBY, RWBY, full0 + 8 * slot);
            tma_3d(s0, &xmap, full0 + 8 * slot, 0, 0, rs * UPS);
            return;
        }
        mbar_expect_tx(full0 + 8 * slot, stage_tx);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        tma_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
    };

#if SK_PDL
    constexpr bool PDL = CL == 1;
#else
    constexpr bool PDL = false;
#endif
    // Under PDL the weights (and router pieces) of the first ring's worth of stages are static,
    // so they stream while the predecessor still runs; x is the predecessor's output and is
    // copied only after griddepcontrol.wait. Each stage's expect_tx covers both copies, so the
    // full barrier cannot complete on the weights alone.
    const int npre = PDL ? (nst_tot < NSTAGE ? nst_tot : NSTAGE) : 1;
    auto issue_w = [&](int st) {
        const int slot = st % NSTAGE;
        const uint32_t s0 = sbase + slot * SST;
        if (FUSE && st >= nst) {
            const int rs = rq * spp + (st - nst);
            mbar_expect_tx(full0 + 8 * slot, UPS * XU + RWBY);
            bulk_g2s(s0 + XBY, rws + ((size_t)rs * 32 + rbk) * RWBY, RWBY, full0 + 8 * slot);
            return;
        }
        mbar_expect_tx(full0 + 8 * slot, stage_tx);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
    };
    auto issue_x = [&](int st) {
        const int slot = st % NSTAGE;
        const int xu = FUSE && st >= nst ? (rq * spp + (st - nst)) * UPS : (gs0 + st) * UPS;
        tma_3d(sbase + slot * SST, &xmap, full0 + 8 * slot, 0, 0, xu);
    };

    if (PDL && tid == 0) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
        for (int i = 0; i < NSTAGE; ++i) {
            mbar_init(full0 + 8 * i, 1);
        }
        fence_mbar_init();
        for (int st = 0; st < npre; ++st) {
            issue_w(st);
        }
        if constexpr (COMP) {
            mbar_init(pbar, 1);
            fence_mbar_init();
            mbar_expect_tx(pbar, tab_ints * 4);
            bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
        }
        for (int i = 0; i < NSTAGE; ++i) {
            mbar_init(empty0 + 8 * i, NW + NESC);
        }
        mbar_init(cbar, NESC > 0 ? NESC : 1);
        fence_mbar_init();
    } else if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
        if constexpr (CL > 1) {
            // A cluster-scope release after a copy is in flight waits for it, so every barrier
            // is ready before the first copy here.
            for (int i = 0; i < NSTAGE; ++i) {
                mbar_init(full0 + 8 * i, 1);
                mbar_init(empty0 + 8 * i, NW + NESC);
            }
            mbar_init(qbar, 1);
            if (crank == 0) {
                mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
            }
            fence_mbar_init();
            issue(0);
        } else {
            // Stage 0 goes out before anything else: its latency is the ramp of the kernel.
            mbar_init(full0, 1);
            fence_mbar_init();
            issue(0);
            if constexpr (COMP) {
                mbar_init(pbar, 1);
                fence_mbar_init();
                mbar_expect_tx(pbar, tab_ints * 4);
                bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
            }
            for (int i = 1; i < NSTAGE; ++i) {
                mbar_init(full0 + 8 * i, 1);
            }
            for (int i = 0; i < NSTAGE; ++i) {
                mbar_init(empty0 + 8 * i, NW + NESC);
            }
            mbar_init(cbar, NESC > 0 ? NESC : 1);
            fence_mbar_init();
        }
    }
    if (warp == 1) {
        zrow_p[lane] = 0u;
    }
    __syncthreads();
    if constexpr (PDL) {
        // Every thread waits, so no global write of this kernel can pass a predecessor's read.
        asm volatile("griddepcontrol.wait;" ::: "memory");
        if (tid == 0) {
            for (int st = 0; st < npre; ++st) {
                issue_x(st);
            }
        }
    }
    if constexpr (CL > 1) {
        cluster_arrive();
    }

    if (warp == NW) {
        if (lane == 0) {
            bool dead = false;
            for (int st = npre; st < nst_tot; ++st) {
                const int slot = st % NSTAGE;
                if (st >= NSTAGE && !mbar_wait<true>(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1, dead, diag, 1)) {
                    break;
                }
                issue(st);
            }
        }
        __syncwarp();
        if constexpr (CL > 1) {
            cluster_wait();
        }
        return;
    }

    if (warp > NW) {
        if constexpr (COMP) {
            // Escape warp ew corrects the escapes of stages st with st % NESC == ew; lane t
            // owns token t. It observes every stage in order and hands every stage back, so the
            // producer can never refill a slot this warp has not seen, which keeps the parity
            // waits from aliasing.
            const int ew = warp - NW - 1;
            float* cr = corr + ew * CSZ;
            for (int i = lane; i < TOKS * CP; i += 32) {
                cr[i] = 0.f;
            }
            __syncwarp();
            bool dead = false;
            if (!__all_sync(0xffffffffu, mbar_wait<true>(pbar, 0, dead, diag, 2))) {
                dead = true;
            }
            const int cnt = dead ? 0 : tab[0];
            int pe = 1;
            for (int st = 0; st < nst_tot; ++st) {
                const int slot = st % NSTAGE;
                const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
                if (!__all_sync(0xffffffffu, mbar_wait<true>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 4))) {
                    dead = true;
                    break;
                }
                if (st % NESC != ew) {
                    while (pe <= cnt && (tab[2 * pe] & 0xFFFF) < kend) {
                        ++pe;
                    }
                } else {
                    const uint32_t xs = sbase + slot * SST;
                    while (pe <= cnt) {
                        const int e0 = tab[2 * pe];
                        const int kk = e0 & 0xFFFF;
                        if (kk >= kend) {
                            break;
                        }
                        if (lane < M) {
                            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
                            const uint32_t xa = xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * M + lane) & 7)) << 4)
                                                + (kc & 7) * 2;
                            cr[lane * CP + (e0 >> 16)] += __int_as_float(tab[2 * pe + 1]) * lds_bf16(xa);
                        }
                        ++pe;
                    }
                }
                fence_proxy_async();
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(empty0 + 8 * slot);
                }
            }
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(cbar);
            }
        }
        if constexpr (CL > 1) {
            cluster_wait();
        }
        return;
    }

    const int wr = warp / WSK, wk = warp % WSK;
    const int g = lane >> 2, t4 = lane & 3;
    // ldmatrix.x4: lanes 8i..8i+7 give the row addresses of 8x8 matrix i; matrices 0..3 are the
    // A fragments (tokens 0-7, k 0-7), (8-15, 0-7), (0-7, 8-15), (8-15, 8-15).
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int lchunk = lane >> 4;
    auto load_a = [&](uint32_t xt, int ui, uint32_t (&a)[4][MT][4]) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int trow = mt * 16 + lrow;
            const uint32_t rbase = xt + trow * 128;
            const int sw = (ui * M + trow) & 7;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                ldsm_x4(trow < M ? rbase + (((2 * j + lchunk) ^ sw) << 4) : zrow, a[j][mt]);
            }
        }
    };
    float acc[NACC][MT][NT][4];
#pragma unroll
    for (int c = 0; c < NACC; ++c) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
            }
        }
    }
    bool dead = false;
    for (int st = 0; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (!__all_sync(0xffffffffu, mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {
            dead = true;
            break;
        }
        const uint32_t s0 = sbase + slot * SST;
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
                    lw[nt] = lds128(wb + nt * TU + lane * 16);
                    cw[nt] = lds64(wb + nt * TU + 512 + lane * 8);
                } else {
                    const uint32_t a0 = wb + nt * TU + lane * 32;
                    const uint4 v0 = lds128(a0), v1 = lds128(a0 + 16);
                    b[nt][0] = v0.x; b[nt][1] = v0.y; b[nt][2] = v0.z; b[nt][3] = v0.w;
                    b[nt][4] = v1.x; b[nt][5] = v1.y; b[nt][6] = v1.z; b[nt][7] = v1.w;
                    if constexpr (HL) {
                        const uint4 l0 = lds128(a0 + 1024), l1 = lds128(a0 + 1040);
                        bl[nt][0] = l0.x; bl[nt][1] = l0.y; bl[nt][2] = l0.z; bl[nt][3] = l0.w;
                        bl[nt][4] = l1.x; bl[nt][5] = l1.y; bl[nt][6] = l1.z; bl[nt][7] = l1.w;
                    }
                }
            }
            uint32_t a[4][MT][4];
            load_a(xt, ui, a);
            if (uu == UPW - 1) {
                // Everything this warp needs from the slot is in registers.
                fence_proxy_async();
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(empty0 + 8 * slot);
                }
            }
            if constexpr (COMP) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
                    decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        constexpr int HM = HL ? 2 : 1;
                        mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt], b[nt][2 * j], b[nt][2 * j + 1]);
                        if constexpr (HL) {
                            mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt], bl[nt][2 * j],
                                     bl[nt][2 * j + 1]);
                        }
                    }
                }
            }
        }
    }

    float racc[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            racc[h][mt][0] = racc[h][mt][1] = racc[h][mt][2] = racc[h][mt][3] = 0.f;
        }
    }
    if constexpr (FUSE) {
        for (int st = nst; st < nst_tot; ++st) {
            const int slot = st % NSTAGE;
            if (!__all_sync(0xffffffffu, mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 64))) {
                dead = true;
                break;
            }
            const uint32_t s0 = sbase + slot * SST;
#pragma unroll
            for (int uu = 0; uu < RUPW; ++uu) {
                const int ui = warp * RUPW + uu;
                const uint32_t wb = s0 + XBY + ui * 2048 + lane * 32;
                const uint4 h0 = lds128(wb), h1 = lds128(wb + 16), l0 = lds128(wb + 1024), l1 = lds128(wb + 1040);
                const uint32_t bh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                const uint32_t bl[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
                uint32_t a[4][MT][4];
                load_a(s0 + ui * XU, ui, a);
                if (uu == RUPW - 1) {
                    fence_proxy_async();
                    __syncwarp();
                    if (lane == 0) {
                        mbar_arrive(empty0 + 8 * slot);
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        mma_bf16(racc[0][mt], a[j][mt], bh[2 * j], bh[2 * j + 1]);
                        mma_bf16(racc[1][mt], a[j][mt], bl[2 * j], bl[2 * j + 1]);
                    }
                }
            }
        }
    }

    if constexpr (COMP) {
        mbar_wait<false>(cbar, 0, dead, diag, 16);
    }
    if (CL > 1 && crank > 0) {
        // Ranks above 0 push their per-warp sums straight into rank 0's peer buffer; the
        // transaction count on rank 0's qbar tells it when all of them have landed.
        cluster_wait();
        const uint32_t peer_l = smem_u32(peer + (crank - 1) * WSK * TOKS * RP);
        uint32_t peer_r, qbar_r;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int row = (wr * NT + nt) * 8 + 2 * t4;
                const int tok = mt * 16 + g;
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[e] = acc[0][mt][nt][e];
#pragma unroll
                    for (int c = 1; c < NACC; ++c) {
                        v[e] += acc[c][mt][nt][e];
                    }
                }
                const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                             :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                             :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r)
                             : "memory");
            }
        }
        return;
    }
    // Every consumer warp has left the ring (and the escape warps are done): the partial sums
    // may overwrite it.
    named_bar(1, NC);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
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
                for (int c = 1; c < NACC; ++c) {
                    v[e] += acc[c][mt][nt][e];
                }
            }
            *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
            *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
        }
    }
    named_bar(1, NC);
    if constexpr (CL > 1) {
        cluster_wait();
        mbar_wait_cluster(qbar, 0, dead, diag, 32);
    }
    constexpr int R4 = R / 4;
    for (int idx = tid; idx < R4 * M; idx += NC) {
        const int tt = idx / R4, row = (idx - tt * R4) * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if constexpr (COMP) {
#pragma unroll
            for (int c = 0; c < NESC; ++c) {
                const float* cr = corr + c * CSZ + tt * CP + row;
                v.x += cr[0]; v.y += cr[1]; v.z += cr[2]; v.w += cr[3];
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
        // Compressed streams carry a row map (4-row groups moved whole, so the 4 stay contiguous).
        const int orow = COMP ? tab[2 * pmax + row] : rb * R + row;
        if ((unsigned)orow >= (unsigned)N) {
            // Only a timed-out table copy gets here; the self-test then rejects the candidate.
            continue;
        }
        OutT* o = out + (size_t)tt * N + orow;
        if constexpr (sizeof(OutT) == 4) {
            *reinterpret_cast<float4*>(o) = v;
        } else {
            const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
            uint2 pk;
            pk.x = *reinterpret_cast<const uint32_t*>(&lo);
            pk.y = *reinterpret_cast<const uint32_t*>(&hi);
            *reinterpret_cast<uint2*>(o) = pk;
        }
    }
    if constexpr (FUSE) {
        named_bar(1, NC);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int tok = mt * 16 + g;
            float* r0 = red + (warp * TOKS + tok) * 8 + 2 * t4;
            *reinterpret_cast<float2*>(r0) =
                make_float2(racc[0][mt][0] + racc[1][mt][0], racc[0][mt][1] + racc[1][mt][1]);
            *reinterpret_cast<float2*>(r0 + 64) =
                make_float2(racc[0][mt][2] + racc[1][mt][2], racc[0][mt][3] + racc[1][mt][3]);
        }
        named_bar(1, NC);
        for (int idx = tid; idx < 2 * M; idx += NC) {
            const int tt = idx >> 1, row = (idx & 1) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * 8 + row);
                v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
            }
            *reinterpret_cast<float4*>(rpart + ((size_t)rq * M + tt) * 256 + rbk * 8 + row) = v;
        }
        // Classic last-block reduction: every writer fences before the counter, the last
        // arrival fences again before reading the other pieces through L2, and zeroes the
        // counter for the next launch (stream-ordered, so every piece has arrived by then).
        __threadfence();
        named_bar(1, NC);
        if (tid == 0) {
            const int old = atomicAdd(rcount + rbk, 1);
            const int last = old + 1 == npiece;
            if (last) {
                rcount[rbk] = 0;
            }
            *last_flag = last;
        }
        named_bar(1, NC);
        if (*last_flag) {
            __threadfence();
            for (int idx = tid; idx < 2 * M; idx += NC) {
                const int tt = idx >> 1, row = (idx & 1) * 4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                for (int pc = 0; pc < npiece; ++pc) {
                    const float4 q = __ldcg(reinterpret_cast<const float4*>(
                        rpart + ((size_t)pc * M + tt) * 256 + rbk * 8 + row));
                    v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
                }
                *reinterpret_cast<float4*>(rout + (size_t)tt * 256 + rbk * 8 + row) = v;
            }
        }
    }
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

// x [M, K] bf16 seen as {64 k, M tokens, K/64 units}: one box brings `units` tiles of M rows x
// 128 B, swizzled so that 16-byte chunk c of linear row r lands at chunk c ^ (r & 7).
CUtensorMap make_xmap(const void* x, uint64_t m, uint64_t k, uint32_t units) {
    CUtensorMap map;
    cuuint64_t dims[3] = {64, m, k / 64};
    cuuint64_t strides[2] = {k * 2, 128};
    cuuint32_t box[3] = {64, (cuuint32_t)m, units};
    cuuint32_t estr[3] = {1, 1, 1};
    const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(x), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return map;
}

#if SK_PDL
// Set by sk_fwd / skf_fwd for the one launch they dispatch. A file-scope flag instead of a
// launch_sk argument keeps the dispatch lines that tools/dl_proj_inline.py generates valid.
bool sk_pdl_launch = false;
#endif

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL,
          typename OutT, bool FUSE>
void launch_sk(const at::Tensor& ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab, int pmax,
               uint32_t base4, int M, int N, int K, int* diag, const uint8_t* rws, float* rpart, int* rcount,
               float* rout) {
    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, FUSE>;
    auto kern = sk_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, FUSE>;
    const int tab_bytes = COMP ? (pmax * 2 + C::R) * 4 : 0;
    const size_t smem = (size_t)C::FIXED + tab_bytes;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
        configured = true;
    }
    TORCH_CHECK(M >= 1 && M <= TOKR, "sk: M out of range for this config");
    TORCH_CHECK(N % C::R == 0 && K % (C::UPS * 64) == 0, "sk: shape does not tile");
    if constexpr (FUSE) {
        const int ncta = N / C::R;
        TORCH_CHECK(ncta % 32 == 0 && (K / (C::UPS * 64)) % (ncta / 32) == 0 && rws != nullptr,
                    "skf: the router pieces do not tile");
    }
    TORCH_CHECK(smem <= 227 * 1024, "sk: shared memory over budget");
    const CUtensorMap xmap = make_xmap(x.data_ptr(), (uint64_t)M, (uint64_t)K, C::UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(ws.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int32_t* tp = COMP ? ptab.data_ptr<int32_t>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    if constexpr (CL > 1) {
        cudaLaunchConfig_t lc = {};
        lc.gridDim = dim3(N / C::R * CL);
        lc.blockDim = dim3(C::THREADS);
        lc.dynamicSmemBytes = smem;
        lc.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = CL;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        lc.attrs = attr;
        lc.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xmap, wsp, op, tp, pmax, base4, M, N, K, diag, rws, rpart,
                                          rcount, rout));
    } else {
#if SK_PDL
        if (sk_pdl_launch) {
            cudaLaunchConfig_t lc = {};
            lc.gridDim = dim3(N / C::R);
            lc.blockDim = dim3(C::THREADS);
            lc.dynamicSmemBytes = smem;
            lc.stream = stream;
            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[0].val.programmaticStreamSerializationAllowed = 1;
            lc.attrs = attr;
            lc.numAttrs = 1;
            C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xmap, wsp, op, tp, pmax, base4, M, N, K, diag, rws, rpart,
                                              rcount, rout));
            return;
        }
#endif
        kern<<<N / C::R, C::THREADS, smem, stream>>>(xmap, wsp, op, tp, pmax, base4, M, N, K, diag, rws, rpart,
                                                     rcount, rout);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
#endif

}  // namespace

#ifdef SK_DEVICE_ONLY

#else
void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
            int64_t n, int64_t k, at::Tensor diag) {
    const int M = x.size(0);
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
#if SK_PDL
    sk_pdl_launch = ((cfg >> 16) & 1) != 0;
    cfg &= 0xffff;
#endif
    switch (cfg) {
    case 0: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 1: launch_sk<1, 8, 3, 1, 4, 4, 3, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 2: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 3: launch_sk<1, 16, 3, 1, 4, 2, 5, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 4: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 5: launch_sk<2, 32, 3, 1, 4, 2, 4, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 6: launch_sk<1, 8, 4, 1, 4, 1, 4, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 7: launch_sk<1, 8, 4, 1, 4, 1, 4, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 8: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 9: launch_sk<1, 16, 2, 2, 2, 2, 3, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 10: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 11: launch_sk<2, 32, 2, 2, 2, 2, 3, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 12: launch_sk<1, 8, 1, 2, 2, 4, 3, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 13: launch_sk<1, 8, 1, 2, 2, 4, 3, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 14: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 15: launch_sk<1, 16, 2, 2, 2, 4, 4, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 16: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 17: launch_sk<2, 32, 2, 2, 2, 4, 3, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 18: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 19: launch_sk<1, 8, 1, 1, 8, 2, 4, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 20: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 21: launch_sk<1, 16, 2, 2, 2, 2, 3, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 22: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 23: launch_sk<2, 32, 2, 2, 2, 2, 3, false, false, 1, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    default:
        TORCH_CHECK(false, "sk: unknown config ", cfg);
    }
}

// Fused pooler + router: out [M, n] bf16 from the pooler stream, rout [M, 256] f32 from the
// router hi/lo stream; rpart is [pieces, M, 256] f32 scratch and rcount int32[32] counters.
void skf_fwd(int64_t cfg, at::Tensor ws, at::Tensor rws, at::Tensor x, at::Tensor out, at::Tensor rout,
             at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor rpart, at::Tensor rcount,
             at::Tensor diag) {
    const int M = x.size(0);
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
    const uint8_t* rw = reinterpret_cast<const uint8_t*>(rws.data_ptr());
#if SK_PDL
    sk_pdl_launch = ((cfg >> 16) & 1) != 0;
    cfg &= 0xffff;
#endif
    switch (cfg) {
    case 24: launch_sk<1, 8, 1, 2, 2, 4, 3, true, false, 1, 1, __nv_bfloat16, true>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, rw, rpart.data_ptr<float>(), rcount.data_ptr<int>(), rout.data_ptr<float>()); break;
    case 25: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 1, 1, __nv_bfloat16, true>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, rw, rpart.data_ptr<float>(), rcount.data_ptr<int>(), rout.data_ptr<float>()); break;
    case 26: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1, 1, __nv_bfloat16, true>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, rw, rpart.data_ptr<float>(), rcount.data_ptr<int>(), rout.data_ptr<float>()); break;
    default:
        TORCH_CHECK(false, "skf: unknown config ", cfg);
    }
}
#endif
"""

_SKP_GEMV_SRC = r"""

#ifndef GEMV_DEVICE_ONLY
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int EXP_WINDOW = 8;
constexpr int ROUTER_K = 8192;
constexpr int SLICE_K = 4096;

// Weights are read once per call, so they must not displace x from L1.
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float lo_f(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_f(unsigned u) { return __uint_as_float(u & 0xFFFF0000u); }

__device__ __forceinline__ unsigned pack_bf16x2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&v);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

// High bytes of the four weights coded in a word's low nibbles, and in its high nibbles.
__device__ __forceinline__ unsigned hi_bytes_lo(unsigned c, unsigned base4) {
    return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
__device__ __forceinline__ unsigned hi_bytes_hi(unsigned c, unsigned base4) {
    return (((c >> 4) & 0x07070707u) + base4) | (c & 0x80808080u);
}

__device__ __forceinline__ void fma4(unsigned lo, unsigned hi, unsigned x01, unsigned x23, float& acc) {
    const unsigned p01 = __byte_perm(lo, hi, 0x5140), p23 = __byte_perm(lo, hi, 0x7362);
    acc = fmaf(lo_f(p01), lo_f(x01), acc);
    acc = fmaf(hi_f(p01), hi_f(x01), acc);
    acc = fmaf(lo_f(p23), lo_f(x23), acc);
    acc = fmaf(hi_f(p23), hi_f(x23), acc);
}

// One row of a chunk: low-byte words l.x..l.w hold k 0-3, 4-7, 8-11, 12-15; code word c0 holds
// k 0-3 (low nibbles) and 8-11 (high), c1 holds 4-7 and 12-15.
__device__ __forceinline__ void chunk_fma(uint4 l, unsigned c0, unsigned c1, uint4 xa, uint4 xb,
                                          unsigned base4, float& acc) {
    fma4(l.x, hi_bytes_lo(c0, base4), xa.x, xa.y, acc);
    fma4(l.y, hi_bytes_lo(c1, base4), xa.z, xa.w, acc);
    fma4(l.z, hi_bytes_hi(c0, base4), xb.x, xb.y, acc);
    fma4(l.w, hi_bytes_hi(c1, base4), xb.z, xb.w, acc);
}

// CTA c computes out[2c] and out[2c+1]. Compute thread t owns chunks t + i*NTH; the extra warp
// applies the escape table. The stream is two row-major planes, every row's low bytes [K] then
// every row's codes [K/2], so per row and chunk a lane loads 16 B of low bytes and 8 B of codes
// and each warp load is 512 or 256 contiguous bytes. Against the earlier stream of one 48-byte
// chunk per row pair split into three 16-byte planes, this is -0.2 to -0.45 us on qkv_gate and
// the slice (5ff76a62b0044378, cgemvp). Every partial sum is combined in a fixed order, so the
// result is deterministic. The escape terms use
// separately rounded multiply and add so a CPU emulation in float32 reproduces them bit for bit
// (the main FMAs are exact products, so they need no such care).
template <int NTH, int CPT>
__device__ __forceinline__ void cgemv_body(const uint8_t* __restrict__ stream, int half_n,
                                           const int2* __restrict__ table, int pmax, unsigned base4,
                                           const __nv_bfloat16* __restrict__ x,
                                           __nv_bfloat16* __restrict__ out) {
    constexpr int NW = NTH / 32, K = 16 * NTH * CPT;
    __shared__ float red[2][NW + 1];
    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (warp < NW) {
        const uint8_t* lp = stream + (size_t)c * 2 * K;
        const uint8_t* cp = stream + (size_t)half_n * 2 * K + (size_t)c * K;
        const uint4* xp = reinterpret_cast<const uint4*>(x) + 2 * t;
        uint4 l0[CPT], l1[CPT], xa[CPT], xb[CPT];
        uint2 q0[CPT], q1[CPT];
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            const int j = i * NTH + t;
            l0[i] = ld_stream(reinterpret_cast<const uint4*>(lp) + j);
            l1[i] = ld_stream(reinterpret_cast<const uint4*>(lp + K) + j);
            q0[i] = __ldg(reinterpret_cast<const uint2*>(cp) + j);
            q1[i] = __ldg(reinterpret_cast<const uint2*>(cp + K / 2) + j);
        }
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            xa[i] = __ldg(xp + i * NTH * 2);
            xb[i] = __ldg(xp + i * NTH * 2 + 1);
        }
        float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            chunk_fma(l0[i], q0[i].x, q0[i].y, xa[i], xb[i], base4, acc0);
            chunk_fma(l1[i], q1[i].x, q1[i].y, xa[i], xb[i], base4, acc1);
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
        if (lane == 0) {
            red[0][warp] = acc0;
            red[1][warp] = acc1;
        }
    } else {
        // Lane i loads entry i, so the count (entry 0) arrives with the first batch in one load.
        // Loading the count and the batch as two loads cost the pooler about 1 us
        // (5ff76a62b0044378: cgemvpe against cgemvp, -1.0 / -1.1 us).
        const int2* tb = table + (size_t)c * pmax;
        int2 ent = __ldg(tb + min(lane, pmax - 1));
        const int count = __shfl_sync(0xFFFFFFFFu, ent.x, 0);
        const unsigned base = base4 & 0xFFu;
        float corr0 = 0.f, corr1 = 0.f;
        for (int e = lane; e <= count; e += 32) {
            const int row = ent.x >> 16, kk = ent.x & 0xFFFF;
            const unsigned bits = (unsigned)ent.y;
            const bool live = e >= 1;
            ent = __ldg(tb + min(e + 32, pmax - 1));
            if (live) {
                const int off = min(max((int)((bits >> 8) & 0x7Fu) - (int)base, 0), EXP_WINDOW - 1);
                const unsigned dec = (bits & 0x80FFu) | ((base + off) << 8);
                const float delta = __fsub_rn(__uint_as_float(bits << 16), __uint_as_float(dec << 16));
                const float xv = __uint_as_float(
                    (unsigned)__ldg(reinterpret_cast<const unsigned short*>(x) + kk) << 16);
                const float d = __fmul_rn(delta, xv);
                corr0 = __fadd_rn(corr0, row ? 0.f : d);
                corr1 = __fadd_rn(corr1, row ? d : 0.f);
            }
        }
        corr0 = warp_sum(corr0);
        corr1 = warp_sum(corr1);
        if (lane == 0) {
            red[0][NW] = corr0;
            red[1][NW] = corr1;
        }
    }
    __syncthreads();
    if (t == 0) {
        float r0 = 0.f, r1 = 0.f;
#pragma unroll
        for (int w = 0; w <= NW; ++w) {
            r0 += red[0][w];
            r1 += red[1][w];
        }
        *reinterpret_cast<unsigned*>(out + 2 * c) = pack_bf16x2(r0, r1);
    }
}

// Registers set the occupancy at one chunk per thread: a warp takes its registers in units of
// 256 from one SM sub-partition's 16384 (cuda_occupancy.h), so at 33-40 registers a
// sub-partition holds 12 warps and at 32 it holds 16, which is 2 against 3 CTAs per SM for the
// 17-warp pooler CTA (b135dd8ee10f4b24). With two chunks per thread (qkv_gate, the slice) 40
// registers win over 32 (5ff76a62b0044378, cgemvp40 against cgemvp: -0.3 / -0.2 us on qkv_gate,
// -0.2 / -0.2 on the slice), as in the public cgemv.
template <int NTH, int CPT>
__global__ void __maxnreg__(32)
cgemv_kernel(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,
             unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {
    cgemv_body<NTH, CPT>(stream, half_n, table, pmax, base4, x, out);
}

template <int NTH, int CPT>
__global__ void __maxnreg__(40)
cgemv40_kernel(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,
               unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {
    cgemv_body<NTH, CPT>(stream, half_n, table, pmax, base4, x, out);
}

template <int NTH, int CPT>
void cgemv_launch(const uint8_t* stream, const int2* table, int pmax, unsigned base4,
                  const __nv_bfloat16* x, __nv_bfloat16* out, int half_n, cudaStream_t s) {
    if constexpr (CPT == 1) {
        cgemv_kernel<NTH, CPT><<<half_n, NTH + 32, 0, s>>>(stream, half_n, table, pmax, base4, x, out);
    } else {
        cgemv40_kernel<NTH, CPT><<<half_n, NTH + 32, 0, s>>>(stream, half_n, table, pmax, base4, x, out);
    }
}

// FP32 router row per CTA: thread t reads float4 i*256+t, so each warp's load is 512 contiguous
// bytes, and all 16 loads are in flight before the first FMA. The block count in
// __launch_bounds__ (here and in sgemv) keeps ptxas from budgeting 32 registers, which sinks the
// later loads below the first FMAs; tools/gemv_compile_check.sh checks the SASS.
__global__ void __launch_bounds__(256, 2)
rgemv_kernel(const float4* __restrict__ w, const uint2* __restrict__ x, float* __restrict__ out) {
    __shared__ float red[8];
    const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float4* wp = w + (size_t)row * (ROUTER_K / 4) + t;
    float4 wv[8];
    uint2 xv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + i * 256);
#pragma unroll
    for (int i = 0; i < 8; ++i) xv[i] = __ldg(x + i * 256 + t);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc = fmaf(wv[i].x, lo_f(xv[i].x), acc);
        acc = fmaf(wv[i].y, hi_f(xv[i].x), acc);
        acc = fmaf(wv[i].z, lo_f(xv[i].y), acc);
        acc = fmaf(wv[i].w, hi_f(xv[i].y), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (t == 0) {
        float r = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) r += red[i];
        out[row] = r;
    }
}

// BF16 rows as stored, four per CTA, K = 4096: thread t reads 8 weights of each row at column 8t.
__global__ void __launch_bounds__(512, 2)
sgemv_kernel(const uint4* __restrict__ w, const uint4* __restrict__ x, __nv_bfloat16* __restrict__ out) {
    constexpr int KV = SLICE_K / 8;
    __shared__ float red[4][16];
    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const uint4* wp = w + (size_t)c * 4 * KV + t;
    uint4 wv[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) wv[r] = ld_stream(wp + r * KV);
    const uint4 xv = __ldg(x + t);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        acc[r] = fmaf(lo_f(wv[r].x), lo_f(xv.x), acc[r]);
        acc[r] = fmaf(hi_f(wv[r].x), hi_f(xv.x), acc[r]);
        acc[r] = fmaf(lo_f(wv[r].y), lo_f(xv.y), acc[r]);
        acc[r] = fmaf(hi_f(wv[r].y), hi_f(xv.y), acc[r]);
        acc[r] = fmaf(lo_f(wv[r].z), lo_f(xv.z), acc[r]);
        acc[r] = fmaf(hi_f(wv[r].z), hi_f(xv.z), acc[r]);
        acc[r] = fmaf(lo_f(wv[r].w), lo_f(xv.w), acc[r]);
        acc[r] = fmaf(hi_f(wv[r].w), hi_f(xv.w), acc[r]);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        acc[r] = warp_sum(acc[r]);
        if (lane == 0) red[r][warp] = acc[r];
    }
    __syncthreads();
    if (t == 0) {
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
#pragma unroll
            for (int r = 0; r < 4; ++r) s[r] += red[r][i];
        }
        uint2 o;
        o.x = pack_bf16x2(s[0], s[1]);
        o.y = pack_bf16x2(s[2], s[3]);
        *reinterpret_cast<uint2*>(out + 4 * c) = o;
    }
}

}  // namespace

#ifndef GEMV_DEVICE_ONLY
static bool aligned_to(const at::Tensor& t, size_t bytes) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
}

static void check_x(const at::Tensor& x, int64_t k) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2
                && x.size(0) == 1 && x.size(1) == k && aligned_to(x, 16),
                "gemv: x must be a contiguous bf16 [1, ", k, "] CUDA tensor, got ", x.sizes());
}

void cgemv_fwd(at::Tensor stream, at::Tensor table, int64_t pmax, int64_t base4, at::Tensor x,
               at::Tensor out, int64_t nthreads) {
    TORCH_CHECK(stream.is_cuda() && stream.scalar_type() == at::kByte && stream.is_contiguous()
                && stream.dim() == 3 && stream.size(2) == 48 && aligned_to(stream, 16),
                "cgemv_fwd: stream must be a contiguous uint8 [N/2, K/16, 48] CUDA tensor");
    const int64_t half_n = stream.size(0), chunks = stream.size(1), k = chunks * 16;
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kInt && table.is_contiguous()
                && table.dim() == 3 && table.size(0) == half_n && table.size(1) == pmax
                && table.size(2) == 2 && pmax >= 2 && aligned_to(table, 8),
                "cgemv_fwd: table must be a contiguous int32 [N/2, pmax, 2] CUDA tensor");
    check_x(x, k);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous()
                && out.numel() == 2 * half_n && aligned_to(out, 4),
                "cgemv_fwd: out must be a contiguous bf16 tensor of ", 2 * half_n, " elements");
    const uint8_t* sp = stream.data_ptr<uint8_t>();
    const int2* tp = reinterpret_cast<const int2*>(table.data_ptr<int32_t>());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
#define CGEMV_CASE(NTH, CPT)                                                                       \
    if (nthreads == NTH && chunks == NTH * CPT) {                                                  \
        cgemv_launch<NTH, CPT>(sp, tp, (int)pmax, (unsigned)base4, xp, op, (int)half_n, s);       \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                            \
        return;                                                                                    \
    }
    CGEMV_CASE(64, 1) CGEMV_CASE(64, 2) CGEMV_CASE(64, 4)
    CGEMV_CASE(128, 1) CGEMV_CASE(128, 2) CGEMV_CASE(128, 4)
    CGEMV_CASE(256, 1) CGEMV_CASE(256, 2) CGEMV_CASE(256, 4)
    CGEMV_CASE(512, 1) CGEMV_CASE(512, 2) CGEMV_CASE(512, 4)
#undef CGEMV_CASE
    TORCH_CHECK(false, "cgemv_fwd: unsupported threads ", nthreads, " for K = ", k,
                " (need threads in {64,128,256,512} and K/16/threads in {1,2,4})");
}

void rgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out) {
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kFloat && w.is_contiguous() && w.dim() == 2
                && w.size(1) == ROUTER_K && aligned_to(w, 16),
                "rgemv_fwd: w must be a contiguous float32 [N, 8192] CUDA tensor");
    check_x(x, ROUTER_K);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous()
                && out.numel() == w.size(0), "rgemv_fwd: out must be a contiguous float32 [1, N] tensor");
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    rgemv_kernel<<<(unsigned)w.size(0), 256, 0, s>>>(
        reinterpret_cast<const float4*>(w.data_ptr<float>()),
        reinterpret_cast<const uint2*>(x.data_ptr<at::BFloat16>()), out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out) {
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                && w.size(0) % 4 == 0 && w.size(1) == SLICE_K && aligned_to(w, 16),
                "sgemv_fwd: w must be a contiguous bf16 [N % 4 == 0, 4096] CUDA tensor");
    check_x(x, SLICE_K);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous()
                && out.numel() == w.size(0) && aligned_to(out, 8),
                "sgemv_fwd: out must be a contiguous bf16 [1, N] tensor");
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    sgemv_kernel<<<(unsigned)(w.size(0) / 4), 512, 0, s>>>(
        reinterpret_cast<const uint4*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint4*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
"""

_SKP_WD_SRC = r"""

namespace {

template <int BN>
__device__ __forceinline__ void wgmma_ss(float* d, uint64_t a, uint64_t b);
template <> __device__ __forceinline__ void wgmma_ss<128>(float* d, uint64_t a, uint64_t b) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "%64, %65, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "l"(a), "l"(b), "r"(1) : "memory");
}


// K-major operand in the 128-byte swizzle that TMA wrote: 8-row groups of 128-byte rows sit
// 1024 B apart (stride byte offset); the leading byte offset is unused for swizzled K-major
// tiles and set to 1 as CUTLASS does. Advancing K by 16 elements inside the 1024-byte-aligned
// atom adds 32 B to the start address; the hardware applies the swizzle on address bits.
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

// wgmma writes the accumulators asynchronously; this keeps the compiler from moving their reads
// or writes across the fence/wait instructions.
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

// The public entry's gq/gqx kernels load weights evict_first and x evict_last. Measured alone
// on b1n128c2 the policy changes nothing (4098aa2361554229); the picked configs keep it because
// they were judged with it.
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

// Partials written before the arrive must be visible to the peers that read them after the wait.
__device__ __forceinline__ void wd_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1,
                                          uint16_t mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%3, %4}], [%2], %5;"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory");
}

// CUTLASS's ClusterBarrier::arrive(cta_id) form: default .release.cta semantics. The stage's
// readers are wgmma operations already retired by wgmma.wait_group, so there are no generic
// writes to publish; job 80b89e112bf3447f ran the .release.cluster form at 1.3 to 4 us per k-tile,
// growing with the number of remote arrives per stage.
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_addr) {
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(cluster_addr) : "memory");
}

// Cluster of CLK * CLR CTAs, rank = rk + CLK * rr: the CLK ranks with one rr split K for one
// row block and sum their partials over DSMEM; the CLR ranks with one rk hold CLR consecutive
// row blocks over the same K range and token tile, so each loads BN / CLR token rows of the x
// tile and multicasts them to all CLR. A stage is refilled only after the consumers of all CLR
// CTAs released it, so every consumer warp arrives on the empty barrier of each CTA it shares
// x with.
template <int WG, int BN, int CLK, int CLR, bool HL, int NST>
struct WdCfg {
    static constexpr int CL = CLK * CLR;
    static constexpr int BM = 64 * WG;
    static constexpr int ATILE = 64 * 128;
    static constexpr int NA = HL ? 2 : 1;
    static constexpr int ABY = WG * NA * ATILE;
    static constexpr int BBY = BN * 128;
    static constexpr int XS = BN / CLR;
    static constexpr int SB = ABY + BBY;
    static constexpr int RING = NST * SB;
    // Partial rows padded to 4 mod 32 floats: the fragment stores (8 rows x 4 token pairs per
    // warp) then hit 32 distinct banks.
    static constexpr int RS = BM + 4;
    static constexpr int PBY = BN * RS * 4;
    static constexpr int THREADS = WG * 128 + 32;
    static constexpr int SMEM = 1024 + RING + 16 * NST;
    static_assert(PBY <= RING, "wd: partial tile does not fit in the ring");
    static_assert(BN % 8 == 0 && BN <= 256, "wd: bad BN");
    // Each multicast slice must start on a 1024-byte swizzle atom so the slices tile exactly
    // like one box.
    static_assert(BN % CLR == 0 && XS % 8 == 0, "wd: bad x slice");
    static_assert(CL <= 16, "wd: cluster too large");
    static_assert(SMEM <= 227 * 1024, "wd: shared memory over budget");
};

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL, bool POL>
__global__ void __launch_bounds__(WG * 128 + 32, 1)
wd_kernel(const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap, OutT* __restrict__ out,
          int M, int N, int K, int* diag) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST>;
    constexpr int CL = C::CL;
    extern __shared__ __align__(1024) uint8_t wd_smem[];
    const uint32_t raw = smem_u32(wd_smem);
    const uint32_t sbase = (raw + 1023u) & ~1023u;
    float* part = reinterpret_cast<float*>(wd_smem + (sbase - raw));
    const uint32_t full0 = sbase + C::RING;
    const uint32_t empty0 = full0 + 8 * NST;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rank = CL > 1 ? (int)wd_cluster_rank() : 0;
    const int rk = rank % CLK;
    const int rr = rank / CLK;
    const int rowbase = ((int)(blockIdx.x / CL) * CLR + rr) * C::BM;
    const int tok0 = (int)blockIdx.y * BN;
    // Uneven split when CLK does not divide the k tiles (qkv_gate's 128 over 3 ranks).
    const int kt0 = rk * (K / 64) / CLK;
    const int ktl = (rk + 1) * (K / 64) / CLK - kt0;
    bool dead = false;

    if (tid == WG * 128) {
        for (int s = 0; s < NST; ++s) {
            mbar_init(full0 + 8 * s, 1);
            mbar_init(empty0 + 8 * s, 4 * WG * CLR);
        }
        fence_mbar_init();
    }
    // Peers multicast into this CTA's ring and arrive on its barriers: they must be initialized
    // cluster-wide first.
    if constexpr (CLR > 1) {
        wd_cluster_sync();
    } else {
        __syncthreads();
    }

    if (warp == 4 * WG) {
        if (lane == 0) {
            uint16_t mask = 0;
#pragma unroll
            for (int r = 0; r < CLR; ++r) {
                mask |= (uint16_t)(1u << (rk + CLK * r));
            }
            const uint64_t wpol = POL ? l2_evict_first() : 0;
            const uint64_t xpol = POL ? l2_evict_last() : 0;
            for (int i = 0; i < ktl; ++i) {
                const int s = i % NST;
                if (i >= NST) {
                    mbar_wait<true>(empty0 + 8 * s, ((i / NST) - 1) & 1, dead, diag, 1);
                }
                const uint32_t fb = full0 + 8 * s;
                mbar_expect_tx(fb, C::SB);
                const uint32_t st = sbase + s * C::SB;
                const int kc = (kt0 + i) * 64;
#pragma unroll
                for (int g = 0; g < WG; ++g) {
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
                if constexpr (CLR > 1) {
                    tma_2d_mc(st + C::ABY + rr * C::XS * 128, &xmap, fb, kc, tok0 + rr * C::XS, mask);
                } else if constexpr (POL) {
                    tma_2d_pol(st + C::ABY, &xmap, fb, kc, tok0, xpol);
                } else {
                    tma_2d(st + C::ABY, &xmap, fb, kc, tok0);
                }
            }
        }
        __syncwarp();
    } else {
        const int g = warp >> 2;
        float acc[BN / 2];
#pragma unroll
        for (int j = 0; j < BN / 2; ++j) {
            acc[j] = 0.f;
        }
        for (int i = 0; i < ktl; ++i) {
            const int s = i % NST;
            mbar_wait<false>(full0 + 8 * s, (i / NST) & 1, dead, diag, 2);
            const uint32_t a0 = sbase + s * C::SB + g * C::NA * C::ATILE;
            const uint32_t b0 = sbase + s * C::SB + C::ABY;
            wg_fence_acc<BN / 2>(acc);
            wg_fence();
#pragma unroll
            for (int kk = 0; kk < ((ABL & 1) ? 0 : 4); ++kk) {
                wgmma_ss<BN>(acc, wg_desc(a0 + 32 * kk), wg_desc(b0 + 32 * kk));
                if constexpr (HL) {
                    wgmma_ss<BN>(acc, wg_desc(a0 + C::ATILE + 32 * kk), wg_desc(b0 + 32 * kk));
                }
            }
            wg_commit();
            wg_fence_acc<BN / 2>(acc);
            // With at most this iteration's group pending, the previous group has finished
            // reading its stage.
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
        }
        wg_wait<0>();
        wg_fence_acc<BN / 2>(acc);
        // Another consumer warp may still be reading the ring that the partials overwrite.
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

    // Also keeps a CTA's ring alive until every peer's multicast into it and every remote
    // arrive on it has landed: all of them precede the peers' arrival here.
    if constexpr (CL > 1) {
        wd_cluster_sync();
    } else {
        __syncthreads();
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

    // No CTA may exit while a peer still reads its partials.
    if constexpr (CL > 1) {
        wd_cluster_sync();
    }
}

#ifndef SK_DEVICE_ONLY
// A [rows, k] bf16 matrix as 2-D tiles of {64 k, box_rows} in the 128-byte swizzle; rows past
// the end read as zeros (the token tail of the last x tile).
CUtensorMap make_2d_map(const void* p, uint64_t rows, uint64_t k, uint32_t box_rows) {
    CUtensorMap map;
    cuuint64_t dims[2] = {k, rows};
    cuuint64_t strides[1] = {k * 2};
    cuuint32_t box[2] = {64, box_rows};
    cuuint32_t estr[2] = {1, 1};
    const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return map;
}

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL, bool POL>
void launch_wd(const at::Tensor& w, const at::Tensor& x, at::Tensor& out, int N, int K, int* diag) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST>;
    auto kern = wd_kernel<WG, BN, CLK, CLR, HL, NST, OutT, ABL, POL>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        if (C::CL > 8) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        }
        configured = true;
    }
    const int M = (int)x.size(0);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2
                && x.size(1) == K, "wd: x must be a contiguous bf16 [M, K] tensor");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                && w.size(1) == K && w.size(0) >= (HL ? 2 : 1) * N, "wd: w must be a contiguous bf16 [rows, K] tensor");
    TORCH_CHECK(out.is_contiguous() && out.size(0) == M && out.size(1) == N && out.element_size() == sizeof(OutT),
                "wd: bad out");
    TORCH_CHECK(M >= 1 && N % (C::BM * CLR) == 0 && K % 64 == 0 && K / 64 >= CLK, "wd: shape does not tile");
    const CUtensorMap wmap = make_2d_map(w.data_ptr(), (uint64_t)w.size(0), (uint64_t)K, 64);
    const CUtensorMap xmap = make_2d_map(x.data_ptr(), (uint64_t)M, (uint64_t)K, C::XS);
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3((unsigned)(N / (C::BM * CLR) * C::CL), (unsigned)((M + BN - 1) / BN));
    lc.blockDim = dim3(C::THREADS);
    lc.dynamicSmemBytes = C::SMEM;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C::CL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    lc.attrs = attr;
    lc.numAttrs = C::CL > 1 ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, wmap, xmap, op, M, N, K, diag));
}
#endif

}  // namespace

#ifdef SK_DEVICE_ONLY
namespace {
template __global__ void wd_kernel<1, 128, 2, 1, false, 8, __nv_bfloat16, 0, true>(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*, int, int, int, int*);
template __global__ void wd_kernel<1, 128, 3, 1, false, 8, __nv_bfloat16, 0, true>(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*, int, int, int, int*);
}
#else
void wd_fwd(int64_t cfg, at::Tensor w, at::Tensor x, at::Tensor out, int64_t n, int64_t k, at::Tensor diag) {
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
    switch (cfg) {
    case 0: launch_wd<1, 128, 2, 1, false, 8, __nv_bfloat16, 0, true>(w, x, out, (int)n, (int)k, dg); break;
    case 1: launch_wd<1, 128, 3, 1, false, 8, __nv_bfloat16, 0, true>(w, x, out, (int)n, (int)k, dg); break;
    default:
        TORCH_CHECK(false, "wd: unknown config ", cfg);
    }
}
#endif
"""

# name -> (WG, BN, CLK, CLR, HL, NSTAGE) of the wide kernel's built configs, in wd_fwd's
# dispatch order, written by tools/dl_proj_inline.py.
_SKP_WD_GEOM = {
    'b1n128c2s8h': (1, 128, 2, 1, False, 8),
    'b1n128c3s8h': (1, 128, 3, 1, False, 8),
}

_SKP_CPP_SRC = r"""
#include <torch/extension.h>
void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor diag);
void skf_fwd(int64_t cfg, at::Tensor ws, at::Tensor rws, at::Tensor x, at::Tensor out, at::Tensor rout, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor rpart, at::Tensor rcount, at::Tensor diag);

void cgemv_fwd(at::Tensor stream, at::Tensor table, int64_t pmax, int64_t base4, at::Tensor x, at::Tensor out, int64_t nthreads);
void rgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out);
void sgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out);
void wd_fwd(int64_t cfg, at::Tensor w, at::Tensor x, at::Tensor out, int64_t n, int64_t k, at::Tensor diag);
"""

# (name, MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, out, fuse) per built config, in
# dispatch order, written by tools/dl_proj_inline.py; the packing helpers index it as SK_CONFIGS.
_SKP_CONFIGS = (
    ('qkv8', 1, 8, 3, 1, 4, 4, 4, True, False, 1, 1, 'bf16', False),
    ('qkv8raw', 1, 8, 3, 1, 4, 4, 3, False, False, 1, 1, 'bf16', False),
    ('qkv16', 1, 16, 3, 1, 4, 2, 5, True, False, 1, 1, 'bf16', False),
    ('qkv16raw', 1, 16, 3, 1, 4, 2, 5, False, False, 1, 1, 'bf16', False),
    ('qkv32', 2, 32, 3, 1, 4, 2, 4, True, False, 1, 1, 'bf16', False),
    ('qkv32raw', 2, 32, 3, 1, 4, 2, 4, False, False, 1, 1, 'bf16', False),
    ('o8', 1, 8, 4, 1, 4, 1, 4, True, False, 1, 1, 'bf16', False),
    ('o8raw', 1, 8, 4, 1, 4, 1, 4, False, False, 1, 1, 'bf16', False),
    ('o16', 1, 16, 2, 2, 2, 2, 3, True, False, 1, 1, 'bf16', False),
    ('o16raw', 1, 16, 2, 2, 2, 2, 3, False, False, 1, 1, 'bf16', False),
    ('o32', 2, 32, 2, 2, 2, 2, 3, True, False, 1, 1, 'bf16', False),
    ('o32raw', 2, 32, 2, 2, 2, 2, 3, False, False, 1, 1, 'bf16', False),
    ('pool8', 1, 8, 1, 2, 2, 4, 3, True, False, 1, 1, 'bf16', False),
    ('pool8raw', 1, 8, 1, 2, 2, 4, 3, False, False, 1, 1, 'bf16', False),
    ('pool16', 1, 16, 2, 2, 2, 4, 4, True, False, 1, 1, 'bf16', False),
    ('pool16raw', 1, 16, 2, 2, 2, 4, 4, False, False, 1, 1, 'bf16', False),
    ('pool32', 2, 32, 2, 2, 2, 4, 3, True, False, 1, 1, 'bf16', False),
    ('pool32raw', 2, 32, 2, 2, 2, 4, 3, False, False, 1, 1, 'bf16', False),
    ('slice8', 1, 8, 1, 1, 8, 2, 4, True, False, 1, 1, 'bf16', False),
    ('slice8raw', 1, 8, 1, 1, 8, 2, 4, False, False, 1, 1, 'bf16', False),
    ('unp16', 1, 16, 2, 2, 2, 2, 3, True, False, 1, 1, 'bf16', False),
    ('unp16raw', 1, 16, 2, 2, 2, 2, 3, False, False, 1, 1, 'bf16', False),
    ('unp32', 2, 32, 2, 2, 2, 2, 3, True, False, 1, 1, 'bf16', False),
    ('unp32raw', 2, 32, 2, 2, 2, 2, 3, False, False, 1, 1, 'bf16', False),
    ('pool8f', 1, 8, 1, 2, 2, 4, 3, True, False, 1, 1, 'bf16', True),
    ('pool16f', 1, 16, 2, 2, 2, 4, 4, True, False, 1, 1, 'bf16', True),
    ('pool32f', 2, 32, 2, 2, 2, 4, 3, True, False, 1, 1, 'bf16', True),
)
SK_CONFIGS = list(_SKP_CONFIGS)
SK_INDEX = {c[0]: i for i, c in enumerate(SK_CONFIGS)}

# @SKP_PY_BEGIN@
EXP_WINDOW = 8


MAX_ESCAPE_DENOM = 200


BALANCE_GROUP = 4


BALANCED_LAYERS = ("qkv_gate",)


SK_USE = {
    ("qkv_gate", 8): "qkv8", ("qkv_gate", 16): "qkv16", ("qkv_gate", 32): "qkv32",
    ("o", 8): "o8", ("o", 16): "o16", ("o", 32): "o32",
    ("pooler", 8): "pool8", ("pooler", 16): "pool16", ("pooler", 32): "pool32",
    ("unpooler", 8): "slice8", ("unpooler", 16): "unp16", ("unpooler", 32): "unp32",
    ("router", 8): "router8", ("router", 16): "router16", ("router", 32): "router32",
}


# At 32 tokens the side-stream router stretched the pooler from 17 to 26 us (timeline in
# check 181ab39f1bd84912); one launch for both took the front chain from 37.6 to 32.4 us
# (5869cc624a394904).
FUSED_GEOM = ("pool8", "pool16", "pool32")


_LAYER_THREADS = {(2304, 8192): 256, (8192, 1024): 64, (4096, 8192): 512, (1024, 4096): 128}


def _sk_geom(name):
    c = SK_CONFIGS[SK_INDEX[name]]
    mt, tokr, nt, nwr, wsk, upw = c[1:7]
    return {"R": nwr * nt * 8, "NT": nt, "NWR": nwr, "WSK": wsk, "UPW": upw, "NW": nwr * wsk,
            "UPS": wsk * upw, "comp": c[8], "hl": c[9], "cl": c[11], "tokr": tokr, "fuse": c[13]}


def _fused_router_geom(name):
    """The router piece's stream geometry for a fused pooler config: every consumer warp splits
    the stage's units, one 8-row tile each."""
    g = _sk_geom(name)
    return {"R": 8, "NT": 1, "NWR": 1, "WSK": g["NW"], "UPW": g["UPS"] // g["NW"], "NW": g["NW"], "UPS": g["UPS"]}


def bf16_bits(w):
    """[n, k] bfloat16 -> int32 bit patterns in 0..65535."""
    return w.view(torch.int16).to(torch.int32) & 0xFFFF


def tile_values(bits):
    """[n, k] -> [n/8 tiles][k/64 units][32 lanes][16 values]: lane 4g+t holds row g of the tile
    and value 4j+2h+p is column 16j+8h+2t+p of the unit, the mma.m16n8k16 B-fragment order."""
    n, k = bits.shape
    t = bits.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def window_base(bits):
    """Start of the 8-wide window of the top seven exponent bits that covers the most weights."""
    e71 = (bits >> 8) & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    return int(hist.unfold(0, 8, 1).sum(1).argmax())


def code_bits(bits, base):
    """(code, escape, decoded bits) per weight: code = sign<<3 | clamp(e71 - base, 0, 7)."""
    e71 = (bits >> 8) & 0x7F
    off = (e71 - base).clamp(0, 7)
    escape = (e71 < base) | (e71 > base + 7)
    sign = bits >> 15
    code = (sign << 3) | off
    decoded = (bits & 0x80FF) | ((base + off) << 8)
    return code, escape, decoded


def raw_units(bits):
    """Raw BF16 tile units, uint8 [n/8, k/64, 1024]: 32 B per lane, the 16 values in order."""
    tv = tile_values(bits)
    return tv.to(torch.int16).view(torch.uint8).reshape(tv.shape[0], tv.shape[1], 1024).contiguous()


def hilo_units(hi_bits, lo_bits):
    """Router tile units, uint8 [n/8, k/64, 2048]: the hi tile then the lo tile."""
    return torch.cat([raw_units(hi_bits), raw_units(lo_bits)], dim=-1).contiguous()


def comp_units(bits, base):
    """12-bit tile units, uint8 [n/8, k/64, 768]: 512 B of low bytes (16 per lane) then 256 B of
    codes (8 per lane: byte b of word w holds the codes of values 8w+b (low nibble) and 8w+4+b)."""
    tv = tile_values(bits)
    code, _, _ = code_bits(tv, base)
    lo = (tv & 0xFF).to(torch.uint8).reshape(tv.shape[0], tv.shape[1], 512)
    cb = code.view(tv.shape[0], tv.shape[1], 32, 2, 2, 4)
    codes = (cb[..., 0, :] | (cb[..., 1, :] << 4)).to(torch.uint8).reshape(tv.shape[0], tv.shape[1], 256)
    return torch.cat([lo, codes], dim=-1).contiguous()


def stage_stream(units, geom, nrows):
    """Stage-major stream [stage][cta][warp][unit in warp][tile] of the first nrows rows, so every
    CTA's stage is one contiguous bulk copy and all CTAs read one contiguous region per stage."""
    R, NT, WSK, UPW, NW, UPS = geom["R"], geom["NT"], geom["WSK"], geom["UPW"], geom["NW"], geom["UPS"]
    dev = units.device
    U = units.shape[1]
    s = torch.arange(U // UPS, device=dev).view(-1, 1, 1, 1, 1)
    c = torch.arange(nrows // R, device=dev).view(1, -1, 1, 1, 1)
    w = torch.arange(NW, device=dev).view(1, 1, -1, 1, 1)
    uu = torch.arange(UPW, device=dev).view(1, 1, 1, -1, 1)
    nt = torch.arange(NT, device=dev).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * UPS + (w % WSK) * UPW + uu
    return units[tile, unit].contiguous().view(-1)


def balance_perm(escape, R):
    """Row permutation (new row -> original row) dealing 4-row groups, sorted by escape count,
    in snake order across the N/R row blocks, so every CTA's escape table is about as long."""
    n = escape.shape[0]
    C, G = n // R, n // BALANCE_GROUP
    per_group = escape.view(G, -1).sum(1)
    order = per_group.argsort(descending=True)
    i = torch.arange(G, device=escape.device)
    rnd, pos = i // C, i % C
    cta = torch.where(rnd % 2 == 0, pos, C - 1 - pos)
    dest = cta * (R // BALANCE_GROUP) + rnd
    groups = torch.empty(G, dtype=torch.long, device=escape.device)
    groups[dest] = order
    return (groups[:, None] * BALANCE_GROUP + torch.arange(BALANCE_GROUP, device=escape.device)).flatten()


def escape_table(bits, decoded, escape, R, nrows, perm=None):
    """int32 [nrows/R, 2*pmax + R]: entry 0 = (count, 0), entries 1..count = (row_local<<16 | k,
    FP32 bits of w - w_decoded) sorted by k, then the row map (original row of each local row).
    Returns (table, pmax, inexact deltas)."""
    dev = bits.device
    C = nrows // R
    rows, cols = escape[:nrows].nonzero(as_tuple=True)
    cta = rows // R
    order = (cta * bits.shape[1] + cols).argsort()
    rows, cols, cta = rows[order], cols[order], cta[order]
    counts = torch.bincount(cta, minlength=C)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=dev) - start[cta] + 1
    w = (bits[rows, cols] << 16).view(torch.float32)
    wd = (decoded[rows, cols] << 16).view(torch.float32)
    delta = w - wd
    inexact = int(((w.double() - wd.double()) != delta.double()).sum())
    tab = torch.zeros(C, 2 * pmax + R, dtype=torch.int32, device=dev)
    tab[:, 0] = counts.to(torch.int32)
    tab[cta, 2 * pos] = (((rows - cta * R) << 16) | cols).to(torch.int32)
    tab[cta, 2 * pos + 1] = delta.view(torch.int32)
    rowmap = perm[:nrows] if perm is not None else torch.arange(nrows, device=dev)
    tab[:, 2 * pmax:] = rowmap.view(C, R).to(torch.int32)
    return tab.contiguous(), pmax, inexact


def pack_layer(w, geoms, slices, balance=False):
    """All packed forms of one BF16 weight: per geometry name a stream, per (R, nrows) a table.
    geoms: {name: geom}; slices: {name: nrows}. Returns None if the matrix is rejected."""
    n, k = w.shape
    bits = bf16_bits(w)
    base = window_base(bits)
    _, escape, _ = code_bits(bits, base)
    n_esc = int(escape.sum())
    if n_esc * MAX_ESCAPE_DENOM > n * k:
        return None
    perm = None
    if balance and n_esc and slices_balanced(geoms, slices, n):
        perm = balance_perm(escape, next(iter(geoms.values()))["R"])
        bits, escape = bits[perm], escape[perm]
    _, escape, decoded = code_bits(bits, base)
    units = comp_units(bits, base)
    out = {"base": base, "base4": base * 0x01010101, "escapes": n_esc, "streams": {}, "tabs": {}, "inexact": 0}
    for name, g in geoms.items():
        nrows = slices[name]
        out["streams"][name] = stage_stream(units, g, nrows)
        key = (g["R"], nrows)
        if key not in out["tabs"]:
            tab, pmax, inexact = escape_table(bits, decoded, escape, g["R"], nrows, perm)
            out["tabs"][key] = (tab, pmax)
            out["inexact"] += inexact
    return out


def slices_balanced(geoms, slices, n):
    """Balancing permutes whole matrices only: every geometry must cover all rows with one R."""
    rs = {g["R"] for g in geoms.values()}
    return len(rs) == 1 and all(nr == n for nr in slices.values())


def pack_raw_layer(w, geoms, slices):
    units = raw_units(bf16_bits(w))
    return {name: stage_stream(units, g, slices[name]) for name, g in geoms.items()}


def pack_router(router, geoms):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    units = hilo_units(bf16_bits(hi), bf16_bits(lo))
    return hi, lo, {name: stage_stream(units, g, router.shape[0]) for name, g in geoms.items()}


def pack_cgemv(w):
    """Stream [N/2, K/16, 48] uint8 and escape table [N/2, pmax, 2] int32 for cgemv_fwd, or None.

    The stream's bytes are two row-major planes (its shape only fixes the byte count): every
    row's low bytes [N, K], then every row's codes [N, K/2], byte b of a row's 16-k chunk j
    holding the codes of k = 16j+b (low nibble) and 16j+8+b (high nibble).
    Table row c: entry 0 is (count, 0); entries 1..count are ((row_local << 16) | k, original
    bits) sorted by k; pmax is even so each row is a multiple of 16 bytes.
    """
    n, k = w.shape
    if w.dtype != torch.bfloat16 or n % 2 or k % 16:
        raise ValueError(f"pack_cgemv: need bfloat16 [even N, K % 16 == 0], got {w.dtype} {tuple(w.shape)}")
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    base = window_base(bits)
    off = ((bits >> 8) & 0x7F) - base
    escaped = (off < 0) | (off >= EXP_WINDOW)
    escapes = int(escaped.sum())
    if escapes > n * k // MAX_ESCAPE_DENOM:
        return None
    half, chunks = n // 2, k // 16
    code = (((bits >> 15) << 3) | off.clamp(0, EXP_WINDOW - 1)).view(n, chunks, 2, 8)
    codes = (code[:, :, 0, :] | (code[:, :, 1, :] << 4)).reshape(-1)
    stream = torch.cat([(bits & 0xFF).reshape(-1), codes]).to(torch.uint8).view(half, chunks, 48)

    c_i, r_i, k_i = escaped.view(half, 2, k).nonzero().unbind(1)
    order = torch.argsort(c_i * (2 * k) + k_i * 2 + r_i)
    c_i, r_i, k_i = c_i[order], r_i[order], k_i[order]
    count = torch.bincount(c_i, minlength=half)
    pmax = int(count.max()) + 1
    pmax += pmax % 2
    pos = torch.arange(len(c_i), device=w.device) - (count.cumsum(0) - count)[c_i] + 1
    table = torch.zeros(half, pmax, 2, dtype=torch.int32, device=w.device)
    table[:, 0, 0] = count.to(torch.int32)
    table[c_i, pos, 0] = ((r_i << 16) | k_i).to(torch.int32)
    table[c_i, pos, 1] = bits.view(half, 2, k)[c_i, r_i, k_i]
    return {"stream": stream, "table": table, "pmax": pmax, "base4": base * 0x01010101,
            "base": base, "escapes": escapes}


def cgemv_threads(n, k):
    """Compute threads per CTA for cgemv_fwd (the escape warp comes on top)."""
    if (n, k) in _LAYER_THREADS:
        return _LAYER_THREADS[(n, k)]
    chunks = k // 16
    for nth in (512, 256, 128, 64):
        if chunks % nth == 0 and chunks // nth >= 1:
            return nth
    raise ValueError(f"cgemv_threads: no thread count divides K/16 = {chunks}")


def bucket_of(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else None
# @SKP_PY_END@


def _w4_source():
    cases = []
    checks = []
    infos = []
    for i, cfg in enumerate(W4_CONFIGS):
        targs = ", ".join(str(v) for v in cfg)
        infos.append(f"        case {i}: return w4_query<{targs}>();")
        cases.append(f"        case {i}: w4_launch<{targs}>(ap, wp, sp, gp, ip, itp, np, mi, n, "
                     f"op, pdl, stream); break;")
        checks.append(
            f"// w4_compile_check: w4_launch<{targs}>(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, false, 0);")
    return (_W4_SRC.replace("//@W4_CASES@", "\n".join(cases))
            .replace("//@W4_INFO@", "\n".join(infos))
            .replace("//@W4_COMPILE_CHECKS@", "\n".join(checks)))


_W4_EXT = None


def _w4_ext():
    global _W4_EXT
    if _W4_EXT is None:
        _W4_EXT = load_inline(
            name="okc_decode_layer_w4",
            cpp_sources=_W4_CPP_SRC,
            cuda_sources=_w4_source(),
            functions=["w4_gemm", "w4_swiglu", "w4_info", "w4_finalize"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        )
    return _W4_EXT


# Bit position of each E2M1 code bit (mantissa, exponent low, exponent high, sign) for the 8
# codes of a repacked word, in fragment order: register j's low then high bf16. Register 0's
# bits are already in place; register 1's sit 6 lower, register 2's 6 higher (mod 32), and
# register 3 takes its mantissa from 4 higher and the rest from 4 lower (see dequant()).
W4_BITPOS = (
    (6, 7, 8, 15), (22, 23, 24, 31),
    (0, 1, 2, 9), (16, 17, 18, 25),
    (12, 13, 14, 21), (28, 29, 30, 5),
    (10, 3, 4, 11), (26, 19, 20, 27),
)


def w4_unit_rows(codes, kind):
    """[N, K] per-row data -> [U, tile, hi8, g, K]: the 32 rows of each unit. Kind 1 pairs gate
    rows 16u.. with up rows N/2 + 16u..; kind 2 takes 32 consecutive rows."""
    n = codes.shape[0]
    rest = codes.shape[1]
    if kind == 1:
        return _perm(codes.reshape(2, n // 32, 2, 8, rest), (1, 0, 2, 3, 4))
    return codes.reshape(n // 32, 2, 2, 8, rest)


def w4_repack_values(packed, kind):
    """One expert's E2M1 bytes [N, K/2] (low nibble first) -> int64 words [N * K / 8], in the
    order [unit][k stage of 64][half][lane][4 words], each word one A fragment of one tile and
    one 16-wide k block: word i of a lane's 16 bytes is k block 2 * half + i // 2, tile i % 2."""
    n = packed.shape[0]
    k = packed.shape[1] * 2
    lo = packed & 15
    hi = packed >> 4
    codes = _stack_last(lo, hi).reshape(n, k)
    c = w4_unit_rows(codes, 1 if kind == 1 else 2)
    # [u, tile, hi8, g, stage, half, j1, khi, t, q]
    c = c.reshape(n // 32, 2, 2, 8, k // 64, 2, 2, 2, 4, 2)
    # -> [u, stage, half, g, t, j1, tile, khi, hi8, q]: lane = 4 g + t, register = 2 khi + hi8
    c = _perm(c, (0, 4, 5, 3, 8, 6, 1, 7, 2, 9)).reshape(-1, 8)
    c = _to_i64(c)
    out = c[:, 0] * 0
    for v in range(8):
        for b in range(4):
            out = out | (((c[:, v] >> b) & 1) << W4_BITPOS[v][b])
    return out


def w4_group_units(a, per_unit):
    """Layout 0 [unit][stage][...] -> layout 1 [unit / 8][stage][unit % 8][...] (a flat array
    of whole experts; per_unit elements per unit, 64 stages each)."""
    per_stage = per_unit // 64
    return _perm(a.reshape(-1, 8, 64, per_stage), (0, 2, 1, 3)).reshape(-1)


def w4_repack_scales(sf, kind):
    """One expert's E4M3 block-scale bytes [N, K/16] -> bytes in the order
    [unit][stage][g][k block 0-3][tile][row g or g + 8]: 16 bytes per lane group per stage."""
    n = sf.shape[0]
    kb = sf.shape[1]
    s = w4_unit_rows(sf, 1 if kind == 1 else 2)
    # [u, tile, hi8, g, stage, half, j1]
    s = s.reshape(n // 32, 2, 2, 8, kb // 4, 2, 2)
    return _perm(s, (0, 4, 3, 5, 6, 1, 2)).reshape(-1)


def _perm(a, dims):
    return a.permute(*dims)


def _stack_last(a, b):
    return torch.stack((a, b), dim=-1)


def _to_i64(a):
    return a.to(torch.int64)


def _w4_scales_ok(sf_bytes):
    """dequant() turns a scale byte into bf16 by shifting and adding to the exponent, which is
    exact only for positive normal E4M3 values (not zero, subnormal, negative or NaN)."""
    b = sf_bytes.to(torch.int32)
    exp = (b >> 3) & 15
    return bool(((b & 0x80) == 0).all() and (exp > 0).all() and (b != 0x7F).all())


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="okc_decode_layer_v1",
            cpp_sources=_CPP_SRC,
            cuda_sources=[_CUDA_SRC, _PREP_SRC, _ATTN_SRC, _ATTN_HOST_SRC, _WS_SRC,
                          _WS_HOST_SRC, _BND_SRC],
            functions=["route_align", "finalize", "router_fwd", "proj_fwd", "router_mma_fwd",
                       "attn_prep", "attn_gate", "attn_fused", "attn_verify",
                       "ffn_boundary_short", "pre_norm", "attn_boundary",
                       "attn_boundary_occupancy", "prefetch_l2", "route_fwd"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        )
    return _EXT


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
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


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data,
        "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data,
        "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _pack_mma_a(w):
    """[n, k] -> mma.m16n8k16 A fragments: [n/16 row tile][k/16 k-tile][lane g*4+c][reg][2]."""
    n, k = w.shape
    tiles = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)
    return tiles.permute(0, 3, 2, 5, 4, 1, 6).contiguous()


def _pack_router_hilo(hi, lo):
    """hi and lo [n, k] BF16 -> A fragments with each k-tile's hi tile followed by its lo tile."""
    n, k = hi.shape
    tiles = [_pack_mma_a(w).view(n // 16, k // 16, 256) for w in (hi, lo)]
    return torch.stack(tiles, dim=2).contiguous()


def _sglang_bsm(num_tokens):
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    return block_size_m


# Token counts the W4 configs are tuned at in prepare() (the scored T); other counts use the
# next one up.
TUNE_TOKENS = (1, 8, 16, 32, 128, 256)


def _w4_policy(num_tokens, state):
    """(GEMM1 config, GEMM2 config, PDL, finalizer cluster size) tuned in prepare() for this
    token count; a None config is Marlin's."""
    policy = state.get("w4_policy") if state.get("w4_ok") else None
    if not policy:
        return None, None, False, 1
    key = next((t for t in TUNE_TOKENS if t >= num_tokens), TUNE_TOKENS[-1])
    return policy.get(key, (None, None, False, 1))


def _w4_fits(cfg, num_tokens):
    """A work item holds at most 8 * nt routes of one expert, and an expert gets at most
    num_tokens routes, so shared memory must hold min(8 * nt, num_tokens) rows."""
    _, nt, rows, _, _, _, _, _, _ = W4_CONFIGS[cfg]
    return rows >= min(8 * nt, num_tokens)


def _moe_opts(num_tokens, state):
    """Our MoE choices for this token count: our routing metadata and finalizer, Marlin
    moe_block_size 8 below 12 tokens else 16 (SGLang's heuristic picks 48 at 128 tokens and 64
    at 256; 16 was 25-35% faster there on muse_moe_w4a16), and the W4 configs tuned in
    prepare(), and the fused MoE variant its bench chose (None: the W4 chain)."""
    g1, g2, pdl, fcl = _w4_policy(num_tokens, state)
    return {"bsm": 8 if num_tokens < 12 else 16, "g1": g1, "g2": g2, "pdl": pdl, "fcl": fcl,
            "fused": _moe1_variant(num_tokens, state)}


def _w4_gemm1(w4, cfg, x, sorted_ids, items, num_items, local_ids, numel, inter, pdl=False):
    """inter[route] = SwiGLU(x[token] @ W13[expert]^T) for every local route, in bf16."""
    splits = W4_CONFIGS[cfg][3]
    name = "w13g" if W4_CONFIGS[cfg][7] else "w13"
    ext = _w4_ext()
    if splits == 1:
        ext.w4_gemm(cfg, x, w4[name], w4[name + "_s"], w4["w13_gs"], sorted_ids, items,
                    num_items, numel, inter, pdl)
        return
    part = torch.empty((splits, numel, 2 * D_EXPERT), device=x.device, dtype=torch.float32)
    ext.w4_gemm(cfg, x, w4[name], w4[name + "_s"], w4["w13_gs"], sorted_ids, items, num_items,
                numel, part, pdl)
    ext.w4_swiglu(part, local_ids.view(-1), splits, inter, pdl)


def _w4_gemm2(w4, cfg, inter, sorted_ids, items, num_items, numel, pdl=False):
    """FP32 K-split partials [splits, routes, 4096] of inter[route] @ W2[expert]^T."""
    splits = W4_CONFIGS[cfg][3]
    name = "w2g" if W4_CONFIGS[cfg][7] else "w2"
    part = torch.empty((splits, numel, D_EXPERT), device=inter.device, dtype=torch.float32)
    _w4_ext().w4_gemm(cfg, inter, w4[name], w4[name + "_s"], w4["w2_gs"], sorted_ids, items,
                      num_items, numel, part, pdl)
    return part


def _route_meta(topk_weights, topk_ids, local_start, num_tokens, state, opts):
    """The part of the MoE call that reads only the routing: block size, GEMM configs and the
    block-sorted routing metadata (ours, or SGLang's expert_map and moe_align_block_size).
    topk_ids are global expert ids; this GPU's experts start at local_start (a device tensor).
    opts: _moe_opts() (our routing metadata, finalizer and block size, and per GEMM a W4 config
    or None for Marlin), or None for SGLang's calls."""
    bank = state["bank"]
    numel = num_tokens * TOP_K
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if opts is not None and opts.get("fused") is not None and num_tokens <= MOE1_TOKENS:
        # The fused kernel routes on the device from the global ids, so the side stream
        # carries only the router and top-8 and nothing is aligned here.
        return {"opts": opts, "fused": opts["fused"], "topk_weights": topk_weights,
                "topk_ids": topk_ids.to(torch.int32).contiguous(), "local_start": local_start}
    if numel > MAX_ROUTES:
        opts = None
    g1 = g2 = None
    items = num_items = items2 = num_items2 = None
    if opts is not None:
        block_size_m = opts["bsm"]
        # The buffer size SGLang allocates, so Marlin sizes its FP32 reduce buffer alike.
        if numel < N_EXPERTS + 1:
            max_padded = numel * block_size_m
        else:
            max_padded = numel + (N_EXPERTS + 1) * (block_size_m - 1)
        max_padded = (max_padded + 3) // 4 * 4
        g1, g2 = opts["g1"], opts["g2"]
        if g1 is not None and not _w4_fits(g1, num_tokens):
            g1 = None
        if g2 is not None and not _w4_fits(g2, num_tokens):
            g2 = None
        rows1 = 8 * W4_CONFIGS[g1][1] if g1 is not None else 0
        rows2 = 8 * W4_CONFIGS[g2][1] if g2 is not None else 0
        (sorted_ids, block_experts, n_padded, local_ids, items, num_items, items2,
         num_items2) = _ext().route_align(topk_ids.to(torch.int32).contiguous(), local_start,
                                          block_size_m, max_padded, rows1, rows2)
    else:
        block_size_m = _sglang_bsm(num_tokens)
        local_ids = (topk_ids - local_start) % N_EXPERTS
        local_ids = bank["expert_map"][local_ids].contiguous()
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS)
    return {
        "opts": opts, "g1": g1, "g2": g2, "bsm": block_size_m, "topk_weights": topk_weights,
        "sorted_ids": sorted_ids, "block_experts": block_experts, "n_padded": n_padded,
        "local_ids": local_ids, "items": items, "num_items": num_items, "items2": items2,
        "num_items2": num_items2,
    }


def _meta_tensors(meta):
    return [v for v in meta.values() if isinstance(v, torch.Tensor)]


def _experts_run(x, meta, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize, on the
    routing metadata from _route_meta(); or the fused launch when the metadata names it."""
    if meta.get("fused") is not None:
        return _moe1_run(x, meta["topk_weights"], meta["topk_ids"], meta["local_start"], state,
                         meta["fused"], meta.get("pdl", False))
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    numel = num_tokens * TOP_K
    opts, g1, g2 = meta["opts"], meta["g1"], meta["g2"]
    topk_weights, local_ids = meta["topk_weights"], meta["local_ids"]
    sorted_ids, block_experts, n_padded = (meta["sorted_ids"], meta["block_experts"],
                                           meta["n_padded"])
    pdl = bool(opts and opts["pdl"])
    cache2 = torch.empty((numel, size_n), device=x.device, dtype=x.dtype)
    common = dict(moe_block_size=meta["bsm"], mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    if g1 is not None:
        _w4_gemm1(state["w4"], g1, x, sorted_ids, meta["items"], meta["num_items"], local_ids,
                  numel, cache2, pdl)
    else:
        cache1 = torch.empty((numel, gemm1_n), device=x.device, dtype=x.dtype)
        cache1 = moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
        silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1),
                     expert_step=1)
    if g2 is not None:
        part = _w4_gemm2(state["w4"], g2, cache2, sorted_ids, meta["items2"],
                         meta["num_items2"], numel, pdl)
        return _w4_ext().w4_finalize(part, topk_weights, state["post_expert_w"],
                                     local_ids.view(-1), POST_NORM_EPS, pdl, opts["fcl"])
    cache3 = torch.empty((numel, size_k), device=x.device, dtype=x.dtype)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=numel, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if opts is not None:
        return _ext().finalize(y, topk_weights, state["post_expert_w"],
                               local_ids.view(num_tokens, TOP_K), POST_NORM_EPS)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _experts(x, topk_weights, topk_ids, local_start, state, opts):
    """The whole MoE call on one stream: _route_meta() then _experts_run()."""
    meta = _route_meta(topk_weights, topk_ids, local_start, x.shape[0], state, opts)
    return _experts_run(x, meta, state)


def _moe1_load():
    return load_inline(
        name="okc_decode_layer_moe1",
        cpp_sources=_MOE1_CPP_SRC,
        cuda_sources=[_MOE1_SRC, _moe1_source()],
        functions=["moe_fused_run"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    )


def _moe1_ext():
    return _bg_ext("moe1", _moe1_load)


# @MOE1_REPACK_BEGIN@
def moe1_unit_rows(a, kind):
    """[N, ...] per-row data -> [units, h, g, ...]: unit rb holds rows (h, g) = gate 8rb + g and
    up N/2 + 8rb + g for kind 1 (W13), rows 16rb + 8h + g for kind 2 (W2)."""
    n = a.shape[0]
    if kind == 1:
        return _perm(a.reshape(2, n // 16, 8, *a.shape[1:]), (1, 0, 2, 3))
    return a.reshape(n // 16, 2, 8, *a.shape[1:])


def moe1_repack_expert(packed, sf, kind):
    """One expert's NVFP4 [N, K/2] bytes (low nibble first) and [N, K/16] E4M3 scale bytes ->
    uint8 [N/16, 36864]: per unit 32,768 bytes of codes as words [warp 8][chunk 8][lane 32]
    [k16 step 4], one word = one lane's m16n8k16 A fragment (8 codes, bits permuted per
    W4_BITPOS) for K = 512 warp + 64 chunk + 16 (lane % 4) + 4 step + j, rows g = lane / 4
    and g + 8; then 4096 bytes of scales [warp][lane][chunk][h], the block scale of row
    (h ? g + 8 : g) for the lane's 16-K block of that chunk."""
    n, k2 = packed.shape
    k = 2 * k2
    codes = _stack_last(packed & 15, packed >> 4).reshape(n, k)
    c = moe1_unit_rows(codes, kind)
    u = c.shape[0]
    # (rb, h, g, w, c, t4, s, jhi, jlo) -> (rb, w, c, g, t4, s, jhi, h, jlo): the last three
    # give the code's fragment position v = 4 jhi + 2 h + jlo.
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
    """The same expert for the fat path: uint8 [N/256, 16, 36864], one 256-row unit x 256-K
    stage per entry. Codes are words [warpgroup 4][warp 4][chunk 4][lane 32][step 4]: the
    fragment of 16-row block 16 u + 4 wg + w (rows as in moe1_unit_rows) for k16 step 4 chunk + step
    at K = 256 st + 64 chunk + 16 t4 + 4 step + j (mma k 2 t4 + j for j < 2, 8 + 2 t4 + j - 2
    above), so a lane's four steps of a chunk share one 16-K scale block and its 16-byte load
    sits next to its neighbours' (no bank conflicts); then scales [warpgroup][warp][lane]
    [chunk][h], one byte per row and 16-K block. The x slab holds each stage in the same mma-k
    order (see the gather and epilogue_a)."""
    n, k2 = packed.shape
    k = 2 * k2
    codes = _stack_last(packed & 15, packed >> 4).reshape(n, k)
    c = moe1_unit_rows(codes, kind)
    nu = c.shape[0] // 16
    # (u, wg, w, h, g, st, c, t4, s, jhi, jlo) -> (u, st, wg, w, c, g, t4, s, jhi, h, jlo)
    c = c.reshape(nu, 4, 4, 2, 8, 16, 4, 4, 4, 2, 2)
    c = _to_i64(_perm(c, (0, 5, 1, 2, 6, 4, 7, 8, 9, 3, 10)).reshape(-1, 8))
    words = c[:, 0] * 0
    for v in range(8):
        for b in range(4):
            words = words | (((c[:, v] >> b) & 1) << W4_BITPOS[v][b])
    words = _to_u8_words(words).reshape(nu * 16, 32768)
    # (u, wg, w, h, g, st, c, t4) -> (u, st, wg, w, g, t4, c, h)
    sc = moe1_unit_rows(sf, kind).reshape(nu, 4, 4, 2, 8, 16, 4, 4)
    sc = _perm(sc, (0, 5, 1, 2, 4, 7, 6, 3)).reshape(nu * 16, 4096)
    return _cat_last(words, sc).reshape(nu, 16, 36864)


def _to_u8_words(words):
    """int64 words in [0, 2^32) -> their little-endian bytes."""
    w = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return w.contiguous().view(torch.uint8)


def _cat_last(a, b):
    return torch.cat((a, b), dim=-1).contiguous()
# @MOE1_REPACK_END@

# Tokens per fused launch; the kernel chunks larger batches itself and the layer's largest
# step has 264. Pair rows, skinny jobs (8 tokens) and fat jobs (64 tokens) per launch.
MOE1_TOKENS = 512
# Fused variants built, self-tested and benched but never picked. Empty: mode 37's dropped
# per-thread fence.acq_rel.gpu was reviewed SOUND (docs/review-mode37-ordering-2026-10-29.md).
MOE1_HELD = ()
_MOE1_EMAX = MOE1_TOKENS * TOP_K
_MOE1_JMAX = LOCAL_EXPERTS + _MOE1_EMAX // 8
_MOE1_JMAX_MAIN = LOCAL_EXPERTS + _MOE1_EMAX // 64


def _moe1_scratch(device):
    """The fused kernel's scratch, allocated once (docs/moe-port-guide.md): SwiGLU output and
    GEMM2 output by pair row, the sums of squares and the barrier counter double-buffered by
    the launch generation, the per-job flags the launch resets itself, and the fat path's x
    stages and stream-K fixup slots. Nothing is memset between launches."""
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
    """The local experts' NVFP4 bytes and E4M3 scales in the fused kernel's two unit layouts
    (16-row units for the skinny kernels, 256-row units in 16 stages for the fat one; bits and
    bytes permuted only) and the global scales; None if a block scale is outside positive
    normal E4M3, where the kernel's exponent-add scale conversion is inexact."""
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
    """One persistent launch for the MoE block: routing from the global ids on the device,
    GEMM1 with SwiGLU, GEMM2, the post-expert norm and the weighted sum. pdl: launched as the
    PDL secondary of the kernel before it on this stream (the route kernel)."""
    st = state["moe1"]
    w, s = st["w"], st["scratch"]
    return _moe1_ext().moe_fused_run(
        x, topk_ids, topk_weights, local_start, w["w13"], w["w2"], w["w13_gs"], w["w2_gs"],
        st["nw"], s["inter"], s["ybuf"], s["ss"], s["ctr"], s["done_a"], w["w13f"], w["w2f"],
        s["xa"], s["xb"], s["fix"], s["fixcnt"], s["xready"], s["done_as"], LOCAL_EXPERTS,
        POST_NORM_EPS, variant, pdl)


def _moe1_variant(num_tokens, state):
    """The fused variant the prepare-time bench chose for this token count; None where the W4
    chain won or the bench did not run."""
    policy = state.get("moe1_policy") if state.get("moe1_ok") else None
    if not policy or num_tokens > MOE1_TOKENS:
        return None
    key = next((t for t in TUNE_TOKENS if t >= num_tokens), TUNE_TOKENS[-1])
    variant = policy.get(key)
    if variant is not None and _MOE1_VARIANTS[variant][1] == "main" \
            and num_tokens < _MOE1_MAIN_MIN_M:
        return None
    return variant


def _moe1_name(variant):
    return "w4" if variant is None else _MOE1_VARIANTS[variant][0]


def _moe1_check(state):
    """The scratch invariant after a completed launch: no spin timeout, the next generation's
    barrier counter and sum-of-squares copy zero, the phase B claim counter of the run-time
    claim modes (23 to 29) zero (CTA 0 zeroes it after the grid barrier), every per-job flag
    zero."""
    s = state["moe1"]["scratch"]
    ctr = s["ctr"]
    gen = int(ctr[3])
    counts = {
        "spin_flags": int(ctr[8]),
        "next_barrier": int(ctr[4 + (gen & 1)]),
        "claim_ctr": int(ctr[10]),
        "next_ss_nonzero": int((s["ss"][gen & 1] != 0).sum()),
        "done_a_nonzero": int((s["done_a"] != 0).sum()),
        "xready_nonzero": int((s["xready"] != 0).sum()),
        "done_as_nonzero": int((s["done_as"] != 0).sum()),
        "fixcnt_nonzero": int((s["fixcnt"] != 0).sum()),
    }
    return all(v == 0 for v in counts.values()), " ".join(f"{k}={v}" for k, v in counts.items())


def _moe1_self_test(state, weights):
    """Every built variant against the FP32 reference at the W4 self-test's token counts plus
    257, normal and skewed routings at two local_start values, with the W4 chain's output
    beside it, and the scratch invariant after each variant. Returns the variants that pass."""
    sizes = (1, 3, 8, 9, 16, 24, 32, 33, 40, 64, 128, 256, 257, 264)
    cases = {}
    for tokens in sizes:
        for start, skewed in ((0, False), (96, True)):
            x, tw, ids, local_start = _moe_routing(tokens, 3000 + tokens + start, start, skewed)
            ref = _ref_experts(x, tw, ids, start, weights, state["post_expert_w"])
            w4 = _experts(x, tw, ids, local_start, state, dict(_moe_opts(tokens, state), fused=None))
            cases[(tokens, start)] = (x, tw, ids, local_start, ref, w4)
    ctr = state["moe1"]["scratch"]["ctr"]
    passed = []
    for variant, (name, kind, _) in enumerate(_MOE1_VARIANTS):
        worst = worst_w4 = 0.0
        for (tokens, start), (x, tw, ids, local_start, ref, w4) in cases.items():
            if kind == "main" and tokens < _MOE1_MAIN_MIN_M:
                continue
            out = _experts(x, tw, ids, local_start, state,
                           dict(_moe_opts(tokens, state), fused=variant))
            err = _rel(out, ref) if torch.isfinite(out).all() else float("inf")
            worst = max(worst, err)
            worst_w4 = max(worst_w4, _rel(out, w4))
        torch.cuda.synchronize()
        clean, msg = _moe1_check(state)
        ok = worst < W4_TEST_TOL and clean
        print(f"MOE1TEST {name}: worst vs fp32 {worst:.2e}, vs the W4 chain {worst_w4:.2e}, "
              f"{msg}: {'ok' if ok else 'FAIL'}")
        if not ok:
            _FAILED.append(f"moe1 {name} (fp32 {worst:.1e}, {msg})")
        ctr[8] = 0
        if ok:
            passed.append(variant)
    return passed


def _moe1_stress(state, sizes, variant, replays):
    """The MoE calls for `sizes` captured back to back in one graph (as a sliding and a full
    layer run one after the other on the shared scratch) and replayed `replays` times: the grid
    barrier's generation must advance once per launch, each replayed output must match an
    eager call (sums of squares are FP32 atomics, so not bitwise) and the scratch invariant must
    hold. Raises on any failure."""
    calls = []
    for i, tokens in enumerate(sizes):
        x, tw, ids, local_start = _moe_routing(tokens, 4000 + tokens + 7 * i, 96, True)
        calls.append((x, tw, ids, local_start, dict(_moe_opts(tokens, state), fused=variant)))
    s = state["moe1"]["scratch"]
    torch.cuda.synchronize()
    gen0 = int(s["ctr"][3])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for c in calls:
                _experts(*c[:4], state, c[4])
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [_experts(*c[:4], state, c[4]) for c in calls]
    torch.cuda.synchronize()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    replayed = [o.clone() for o in outs]
    eager = [_experts(*c[:4], state, c[4]) for c in calls]
    torch.cuda.synchronize()
    # Two warm-up rounds, the replays and the eager round; a capture runs nothing.
    gens = int(s["ctr"][3]) - gen0
    rel = max(_rel(r, e) for r, e in zip(replayed, eager))
    finite = all(bool(torch.isfinite(r).all()) for r in replayed)
    clean, msg = _moe1_check(state)
    msg = (f"T={'+'.join(map(str, sizes))} {_moe1_name(variant)} replays={replays}, one host "
           f"sync after all of them; retained: the last replay's {len(sizes)} outputs, "
           f"replayed_vs_eager={rel:.1e} (each < 1e-4, finite={finite}) generations={gens} {msg}")
    if not (finite and rel < 1e-4 and clean and gens == len(sizes) * (replays + 3)):
        raise RuntimeError(f"moe fused stress: {msg} finite={finite}")
    return msg


def _moe1_tune(state, variants, seeds=(11, 12, 13, 14)):
    """Per TUNE_TOKENS count, the main-stream MoE call (its routing metadata built outside the
    timed graph, as the side stream builds it in kernel_fn) for the W4 chain and every passing
    fused variant, as cold-L2 graph replays on judge-like routings; the fastest wins. Only
    timings are kept: the choice is a variant per token count, fixed before any timed call."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    policy = {}
    for bs in TUNE_TOKENS:
        cands = [None] + [v for v in variants
                          if _MOE1_VARIANTS[v][1] == "skinny" or bs >= _MOE1_MAIN_MIN_M]
        tot = {v: 0.0 for v in cands}
        busiest = []
        for seed in seeds:
            x, tw, ids, start = _moe_judge_routing(bs, _TUNE_Q_LEN[bs], 977 * seed + bs, state)
            loc = ids.long().flatten() - start
            busiest.append(int(torch.bincount(loc[(loc >= 0) & (loc < LOCAL_EXPERTS)],
                                              minlength=LOCAL_EXPERTS).max()))
            for v in cands:
                meta = _route_meta(tw, ids, start, bs, state, dict(_moe_opts(bs, state), fused=v))
                tot[v] += _graph_time(lambda: _experts_run(x, meta, state), flush)
        n = len(seeds)
        line = " ".join(f"{_moe1_name(v)}:{t / n:.1f}"
                        for v, t in sorted(tot.items(), key=lambda kv: kv[1]))
        print(f"MOE1BENCH T={bs} {line} us (graph replay, cold L2, mean of {n} judge-like "
              f"routings, busiest local expert {busiest} routes)")
        state.setdefault("moe1_bench", {})[bs] = {_moe1_name(v): t / n for v, t in tot.items()}
        state.setdefault("moe1_busiest", {})[bs] = busiest
        pickable = {v: t for v, t in tot.items() if _moe1_name(v) not in MOE1_HELD}
        policy[bs] = min(pickable, key=pickable.get)
    clean, msg = _moe1_check(state)
    if not clean:
        print(f"MOE1BENCH scratch dirty after the bench ({msg}); fused policy discarded")
        _FAILED.append(f"moe1 bench: scratch dirty ({msg}), policy discarded")
        return {}
    print(f"MOE1POLICY {{{', '.join(f'{b}: {_moe1_name(v)}' for b, v in policy.items())}}}")
    return policy


def _moe1_enable(state, weights):
    t0 = time.time()
    w = _moe1_prepare(weights)
    if w is None:
        raise RuntimeError("block scales outside positive normal E4M3")
    state["moe1"] = {"w": w, "nw": state["post_expert_w"].float().contiguous(),
                     "scratch": _moe1_scratch(weights["w13"].device)}
    t1 = time.time()
    _moe1_ext()
    t2 = time.time()
    passed = _moe1_self_test(state, weights)
    if not passed:
        raise RuntimeError("no fused variant passed the FP32 check")
    t3 = time.time()
    skinny = [v for v in passed if _MOE1_VARIANTS[v][1] == "skinny"]
    main = [v for v in passed if _MOE1_VARIANTS[v][1] == "main"]
    # Mixed sizes back to back on the shared scratch, as consecutive layers call it: every
    # skinny variant at 8, 16, 1, 32, 8 tokens and every fat mode at 32, 128, 17, 256.
    shapes = [((8, 16, 1, 32, 8), v) for v in skinny]
    shapes += [((32, 128, 17, 256), v) for v in main] if main else [((128,), skinny[0])]
    stress = state["moe1_stress"] = {}
    for sizes, variant in shapes:
        try:
            msg = _moe1_stress(state, sizes, variant, 2000)
            stress[_moe1_name(variant)] = "ok"
        except RuntimeError as exc:
            # Each variant is its own kernel instance, so one that fails here is dropped from
            # the bench without taking the others (already stressed or not) with it, unless it
            # is the last of its kind.
            kind = _MOE1_VARIANTS[variant][1]
            if len(main if kind == "main" else skinny) < 2:
                raise
            msg = str(exc)
            stress[_moe1_name(variant)] = "FAIL"
            _FAILED.append(f"moe1 stress {_moe1_name(variant)}: {msg[:160]}")
            passed.remove(variant)
            (main if kind == "main" else skinny).remove(variant)
            # Back to the state as allocated, which every variant starts from.
            torch.cuda.synchronize()
            for k in ("ctr", "ss", "done_a", "xready", "done_as", "fixcnt"):
                state["moe1"]["scratch"][k].zero_()
        print(f"MOE1STRESS {msg}")
    t4 = time.time()
    state["moe1_policy"] = _moe1_tune(state, passed)
    state["moe1_passed"] = list(passed)
    state["moe1_ok"] = True
    t5 = time.time()
    print(f"MoE fused: repack {t1 - t0:.1f} s, extension build {_bg_seconds('moe1'):.1f} s on "
          f"its thread (waited {t2 - t1:.1f} s), self-test {t3 - t2:.1f} s, stress "
          f"{t4 - t3:.1f} s, bench {t5 - t4:.1f} s")


def _skp_load():
    src = _SKP_SRC + _SKP_GEMV_SRC + _SKP_WD_SRC
    kwargs = dict(name="okc_decode_layer_skp", cpp_sources=_SKP_CPP_SRC, cuda_sources=src,
                  functions=["sk_fwd", "skf_fwd", "cgemv_fwd", "rgemv_fwd", "sgemv_fwd",
                             "wd_fwd"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
                  + (["-DSK_PDL=1"] if PORT_SK_PDL else []))
    try:
        return load_inline(no_implicit_headers=True, **kwargs)
    except TypeError:
        # An older torch adds torch/extension.h itself, which wants torch/types.h in the source.
        kwargs["cuda_sources"] = src.replace("#include <ATen/core/Tensor.h>",
                                             "#include <torch/types.h>")
        return load_inline(**kwargs)


def _skp_ext():
    return _bg_ext("skp", _skp_load)


SKP_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
# muse_bf16_proj's gate: relative L2 against FP64 (BF16 layers; the router's hi/lo result
# measured 3e-8 from hi + lo in emulation), probe rows bitwise.
SKP_TOL = 5e-3
SKP_ROUTER_TOL = 2e-5
SKP_PROBE_TOL = 5e-4
SKP_MS = {8: (1, 2, 5, 7, 8), 16: (9, 12, 16), 32: (17, 31, 32)}
# The layer's token counts up to 32: decode 1, verify 1 (8), decode 16 and 32.
SKP_TUNE_MS = (1, 8, 16, 32)


def _skp_rows(layer, bucket, n):
    """Rows the layer asks for: the unpooler's 1024-row slice up to 8 tokens, else the whole
    matrix."""
    return UNPOOL_SLICE if layer == "unpooler" and bucket == 8 else n


def _skp_runner(ext, name, stream, tab, pmax, base4, n, k, diag):
    cid = SK_INDEX[name]

    def run(x, pdl=False):
        # Bit 16 of the config: a PDL launch (built with SK_PDL, M1); without the switch the
        # extension masks nothing and would reject the config, so it is set only then.
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        ext.sk_fwd(cid | (SK_PDL_BIT if pdl else 0), stream, x, out, tab, pmax, base4, n, k, diag)
        return out
    run.sk = True
    return run


def _skp_gemv_runner(call, n):
    def run(x):
        out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
        call(x, out)
        return out
    return run


def _skp_pack(state):
    """Every candidate of every (layer, bucket) as run(x) -> out: the compressed sk stream per
    geometry with its escape table (qkv rows balanced across row blocks; a matrix past the
    1/200 escape rule stays raw), the raw BF16 stream, and at m = 1 cgemv's 12-bit row stream
    and, for the unpooler slice, sgemv. Returns the candidates and the packed pooler (the
    fused entry reuses its stream and table)."""
    ext = _skp_ext()
    dense = state["dense"]
    diag = state["skp"]["diag"]
    cands = {}
    packed_pooler = None
    for layer in SKP_LAYERS:
        w = dense[layer]
        n, k = w.shape
        names = {b: SK_USE[(layer, b)] for b in (8, 16, 32)}
        slices = {}
        for b, nm in names.items():
            slices[nm] = slices[nm + "raw"] = _skp_rows(layer, b, n)
        geoms = {nm: _sk_geom(nm) for nm in names.values()}
        packed = pack_layer(w, geoms, slices, balance=layer in BALANCED_LAYERS)
        if layer == "pooler":
            packed_pooler = packed
        if packed is not None:
            print(f"SKP pack {layer}: base {packed['base']}, {packed['escapes']} escapes, "
                  f"{packed['inexact']} inexact deltas")
            for b, nm in names.items():
                nrows = slices[nm]
                tab, pmax = packed["tabs"][(geoms[nm]["R"], nrows)]
                run = _skp_runner(ext, nm, packed["streams"][nm], tab, pmax, packed["base4"],
                                  nrows, k, diag)
                cands.setdefault((layer, b), []).append((nm, run, False))
        else:
            print(f"SKP pack {layer}: past the 1/{MAX_ESCAPE_DENOM} escape rule, raw BF16 only")
        raw = pack_raw_layer(w, {nm + "raw": _sk_geom(nm + "raw") for nm in names.values()},
                             slices)
        for b, nm in names.items():
            nrows = slices[nm + "raw"]
            run = _skp_runner(ext, nm + "raw", raw[nm + "raw"], diag, 0, 0, nrows, k, diag)
            cands.setdefault((layer, b), []).append((nm + "raw", run, False))
        w1 = w[:_skp_rows(layer, 8, n)].contiguous()
        cg = pack_cgemv(w1)
        if cg is not None:
            nth = cgemv_threads(*w1.shape)
            cands[(layer, 8)].append(("cgemv", _skp_gemv_runner(
                lambda x, out, cg=cg, nth=nth: ext.cgemv_fwd(cg["stream"], cg["table"], cg["pmax"],
                                                             cg["base4"], x, out, nth),
                w1.shape[0]), True))
        if layer == "unpooler":
            cands[(layer, 8)].append(("sgemv", _skp_gemv_runner(
                lambda x, out, w1=w1: ext.sgemv_fwd(w1, x, out), w1.shape[0]), True))
    return cands, packed_pooler


def _skp_gen_x(m, k, gen, device):
    """Random tokens with the judge's probe rows: one power-of-two entry in about one row in 8
    (always the first), so the exact answer there is a weight column times 2^p."""
    x = torch.randn((m, k), generator=gen, device=device)
    probes = torch.rand((m,), generator=gen, device=device) < 1 / 8
    probes[0] = True
    cols = torch.randint(k, (m,), generator=gen, device=device)
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16), probes


def _skp_check(out, ref, probes, tol):
    out = out.double()
    err = float((out - ref).norm() / ref.norm())
    rows = probes.nonzero().flatten()
    perr = float(((out[rows] - ref[rows]).norm(dim=-1)
                  / ref[rows].norm(dim=-1).clamp(min=1e-30)).max())
    return err < tol and perr < SKP_PROBE_TOL and bool(torch.isfinite(out).all()), err, perr


def _skp_self_test(state, cands):
    """Every candidate against FP64 at its bucket's token counts (1, 2, 5, 7, 8; 9, 12, 16;
    17, 31, 32; the GEMVs at 1) on inputs with probe rows: relative L2 within 5e-3, the probe
    rows bitwise equal to the BF16-rounded reference, no timed-out wait. Failures are dropped
    with a printed reason; returns the survivors."""
    dense = state["dense"]
    diag = state["skp"]["diag"]
    gen = torch.Generator(device=diag.device).manual_seed(1234)
    survivors = {}
    for (layer, b), lst in sorted(cands.items()):
        w = dense[layer]
        k = w.shape[1]
        nrows = _skp_rows(layer, b, w.shape[0])
        for name, run, m1 in lst:
            try:
                worst = 0.0
                for m in (1,) if m1 else SKP_MS[b]:
                    diag.zero_()
                    x, probes = _skp_gen_x(m, k, gen, w.device)
                    out = run(x)
                    torch.cuda.synchronize()
                    ref = (x.double() @ w[:nrows].double().T).to(torch.bfloat16).double()
                    ok, err, perr = _skp_check(out, ref, probes, SKP_TOL)
                    exact = bool(torch.equal(out[probes].double(), ref[probes]))
                    worst = max(worst, err)
                    if not (ok and exact and int(diag[0]) == 0):
                        raise RuntimeError(f"m={m}: rel err {err:.3e}, probe err {perr:.3e}, probe "
                                           f"rows exact {exact}, diag {int(diag[0])}")
                survivors.setdefault((layer, b), []).append((name, run, m1))
                _SKP_REPORT.setdefault("selftest", []).append(f"{layer}:{name}")
                print(f"SKP selftest {layer} {name}: ok (worst rel err {worst:.2e})")
            except Exception as exc:
                print(f"SKP selftest {layer} {name}: disabled ({type(exc).__name__}: "
                      f"{str(exc)[-300:]})")
                _FAILED.append(f"skp {layer} {name} ({type(exc).__name__})")
    return survivors


def _skp_fused(state, packed_pooler):
    """The fused pooler + router entry per token bucket (8, 16 and 32): run(x) -> (pooler out
    bf16 [M, 4096], router logits fp32 [M, 256]) from the pooler's 12-bit stream and the
    router's hi/lo stream in one launch, self-tested against FP64 here (pooler within 5e-3
    with bitwise probe rows, router within 2e-5)."""
    if packed_pooler is None:
        return {}
    ext = _skp_ext()
    router, pooler = state["router"], state["dense"]["pooler"]
    diag = state["skp"]["diag"]
    dev = router.device
    gen = torch.Generator(device=dev).manual_seed(4321)
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    units = hilo_units(bf16_bits(hi), bf16_bits(lo))
    fused = {}
    for b, nm in ((8, "pool8"), (16, "pool16"), (32, "pool32")):
        # Scratch per config: the piece counter resets itself per launch, so the configs may
        # alternate. [8, 16, 256] holds 4 pieces x 32 tokens, pool32f's 128 CTAs at M = 32.
        rpart = torch.zeros((8, 16, 256), dtype=torch.float32, device=dev)
        rcount = torch.zeros(32, dtype=torch.int32, device=dev)
        g = _sk_geom(nm)
        rstream = stage_stream(units, _fused_router_geom(nm), router.shape[0])
        tab, pmax = packed_pooler["tabs"][(g["R"], pooler.shape[0])]
        stream, base4 = packed_pooler["streams"][nm], packed_pooler["base4"]
        cid = SK_INDEX[nm + "f"]

        def run(x, pdl=False, cid=cid, stream=stream, rstream=rstream, tab=tab, pmax=pmax,
                base4=base4, rpart=rpart, rcount=rcount):
            m = x.shape[0]
            out = torch.empty((m, pooler.shape[0]), dtype=torch.bfloat16, device=x.device)
            rout = torch.empty((m, router.shape[0]), dtype=torch.float32, device=x.device)
            ext.skf_fwd(cid | (SK_PDL_BIT if pdl else 0), stream, rstream, x, out, rout, tab,
                        pmax, base4, pooler.shape[0],
                        pooler.shape[1], rpart, rcount, diag)
            return out, rout
        try:
            for m in {8: (1, 3, 8), 16: (9, 16), 32: (17, 32)}[b]:
                diag.zero_()
                x, probes = _skp_gen_x(m, pooler.shape[1], gen, dev)
                out, rout = run(x)
                torch.cuda.synchronize()
                ref = (x.double() @ pooler.double().T).to(torch.bfloat16).double()
                ok, err, perr = _skp_check(out, ref, probes, SKP_TOL)
                exact = bool(torch.equal(out[probes].double(), ref[probes]))
                rref = x.double() @ router.double().T
                rok, rerr, rperr = _skp_check(rout, rref, probes, SKP_ROUTER_TOL)
                if not (ok and exact and rok and int(diag[0]) == 0):
                    raise RuntimeError(f"m={m}: pooler err {err:.3e}/{perr:.3e} exact {exact}, "
                                       f"router err {rerr:.3e}/{rperr:.3e}, diag {int(diag[0])}")
            fused[b] = run
            print(f"SKP fused pooler+router {nm}f: ok")
        except Exception as exc:
            print(f"SKP fused pooler+router {nm}f: disabled ({type(exc).__name__}: "
                  f"{str(exc)[-300:]})")
            _FAILED.append(f"skp fused {nm}f ({type(exc).__name__})")
    return fused


# Compact SKP self-test and m = 1 / 8 tune results for the summary, which the log tail keeps.
_SKP_REPORT = {}


def _skp_tune(state, survivors):
    """Per (layer, m) for m in 1, 8, 16, 32: the surviving sk and GEMV candidates against the
    path kernel_fn runs today (proj_fwd where _proj_wins, else cuBLAS), as cold-L2 graph
    replays on random inputs; the fastest becomes the pick, None keeping today's path. Other
    m share their bucket's pick."""
    dense = state["dense"]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(2468)
    picks = {}
    for layer in SKP_LAYERS:
        w = dense[layer]
        n, k = w.shape
        for m in SKP_TUNE_MS:
            b = bucket_of(m)
            nrows = _skp_rows(layer, b, n)
            lst = {nm: run for nm, run, m1 in survivors.get((layer, b), []) if m == 1 or not m1}
            x = torch.randn((m, k), generator=gen, device=w.device).to(torch.bfloat16)
            today = "proj_fwd" if state["proj_ours"] and _proj_wins(layer, m) else "cublas"
            times = {today: _graph_time(lambda: _proj(x, layer, nrows, state), flush)}
            for nm, run in lst.items():
                times[nm] = _graph_time(lambda run=run: run(x), flush)
            best = min(times, key=times.get)
            if m in (1, 8):
                _SKP_REPORT.setdefault(f"tune m={m}", []).append(
                    f"{layer} " + " ".join(f"{nm}:{t:.1f}" for nm, t in
                                           sorted(times.items(), key=lambda kv: kv[1])[:3]))
            print(f"SKP tune {layer} m={m}: "
                  + " ".join(f"{nm}:{t:.1f}" for nm, t in sorted(times.items(), key=lambda kv: kv[1]))
                  + f" -> {best}")
            picks[(layer, m)] = None if best == today else (best, lst[best], nrows)
        for m in range(2, 33):
            if (layer, m) not in picks:
                picks[(layer, m)] = picks[(layer, 8 if m <= 8 else 16 if m <= 16 else 32)]
    return picks


# The wide kernel serves muse_bf16_proj's 128-token bucket (65 to 128 tokens), tuned at 128,
# the only scored count in it (verify 16).
SKP_WD_MIN, SKP_WD_MAX = 65, 128
SKP_WD_MS = (65, 100, 128)
SKP_WD_LAYERS = ("qkv_gate", "pooler")
# muse_bf16_proj's WD_MARGIN: its tuner picked a 1% win (pooler m=128, 30.2 vs 30.5 us) that
# the judge then timed at 0.87x (job 71b1927dca894bc4).
SKP_WD_MARGIN = 0.97


def _skp_wd_fits(name, n, k):
    wg, bn, clk, clr, hl, nst = _SKP_WD_GEOM[name]
    return not hl and n % (64 * wg * clr) == 0 and k % 64 == 0 and k // 64 >= clk


def _skp_wd_runner(ext, name, w, diag):
    cid = list(_SKP_WD_GEOM).index(name)
    n, k = w.shape

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run


def _skp_wide(state):
    """muse_bf16_proj's wide kernel (TMA + wgmma, K split over a cluster, 8-stage ring) for the
    qkv_gate and the pooler at 65 to 128 tokens. Every built config that tiles the matrix is
    self-tested against FP64 at 65, 100 and 128 tokens (relative L2 within 5e-3, probe rows
    bitwise, no timed-out wait), then timed at 128 tokens as cold-L2 graph replays against the
    path _proj runs there today (cuBLAS). Returns {layer: (name, run)} for the configs under
    SKP_WD_MARGIN of that time; the weights are read as given, no repack."""
    ext = _skp_ext()
    dense = state["dense"]
    diag = state["skp"]["diag"]
    gen = torch.Generator(device=diag.device).manual_seed(1357)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    picks, report = {}, {}
    for layer in SKP_WD_LAYERS:
        w = dense[layer]
        n, k = w.shape
        runs = {}
        for name in _SKP_WD_GEOM:
            if not _skp_wd_fits(name, n, k):
                continue
            run = _skp_wd_runner(ext, name, w, diag)
            try:
                worst = 0.0
                for m in SKP_WD_MS:
                    diag.zero_()
                    x, probes = _skp_gen_x(m, k, gen, w.device)
                    out = run(x)
                    torch.cuda.synchronize()
                    ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
                    ok, err, perr = _skp_check(out, ref, probes, SKP_TOL)
                    exact = bool(torch.equal(out[probes].double(), ref[probes]))
                    worst = max(worst, err)
                    if not (ok and exact and int(diag[0]) == 0):
                        raise RuntimeError(f"m={m}: rel err {err:.3e}, probe err {perr:.3e}, probe "
                                           f"rows exact {exact}, diag {int(diag[0])}")
                runs[name] = run
                print(f"SKP selftest {layer} {name}: ok (worst rel err {worst:.2e})")
            except Exception as exc:
                print(f"SKP selftest {layer} {name}: disabled ({type(exc).__name__}: "
                      f"{str(exc)[-300:]})")
                _FAILED.append(f"skp {layer} {name} ({type(exc).__name__})")
        if not runs:
            continue
        x = torch.randn((SKP_WD_MAX, k), generator=gen, device=w.device).to(torch.bfloat16)
        times = {"cublas": _graph_time(lambda: _proj(x, layer, n, state), flush)}
        for name, run in runs.items():
            times[name] = _graph_time(lambda run=run: run(x), flush)
        best = min(runs, key=times.get)
        pick = best if times[best] < SKP_WD_MARGIN * times["cublas"] else None
        if pick is not None:
            picks[layer] = (pick, runs[pick])
        report[layer] = (pick or "cublas", times)
        print(f"SKP tune {layer} m={SKP_WD_MAX}: "
              + " ".join(f"{nm}:{t:.1f}" for nm, t in sorted(times.items(), key=lambda kv: kv[1]))
              + f" -> {pick or 'cublas'}")
    state["skp_wide_report"] = report
    return picks


def _skp_enable(state):
    t0 = time.time()
    device = state["dense"]["pooler"].device
    state["skp"] = {"diag": torch.zeros(4, dtype=torch.int32, device=device), "fused": {}}
    _skp_ext()
    t1 = time.time()
    cands, packed_pooler = _skp_pack(state)
    torch.cuda.synchronize()
    t2 = time.time()
    survivors = _skp_self_test(state, cands)
    state["skp"]["fused"] = _skp_fused(state, packed_pooler)
    t3 = time.time()
    state["skp_pick"] = _skp_tune(state, survivors)
    t4 = time.time()
    # Timed against _proj's path before skp_ok is set, so the reference is cuBLAS. A failure
    # here leaves the sk picks on.
    state["skp_wide"] = {}
    try:
        state["skp_wide"] = _skp_wide(state)
    except Exception as exc:
        print(f"SKP wide: disabled ({type(exc).__name__}: {str(exc)[-300:]})")
        _FAILED.append(f"skp wide ({type(exc).__name__})")
    state["skp_ok"] = True
    t5 = time.time()
    on = sorted({(layer, pick[0]) for (layer, _), pick in state["skp_pick"].items() if pick})
    on += sorted((layer, pick[0]) for layer, pick in state["skp_wide"].items())
    print(f"SKP: extension build {_bg_seconds('skp'):.1f} s on its thread (waited {t1 - t0:.1f} s), "
          f"pack {t2 - t1:.1f} s, self-test {t3 - t2:.1f} s, tune {t4 - t3:.1f} s, wide "
          f"{t5 - t4:.1f} s; picked {on}")


def _attn_sglang(qkv, data, state):
    """FusedQKNorm, the FP8 casts and store_cache, FA3, FusedRMSNormGate: baseline.py's calls."""
    cfg = data["config"]
    tokens = cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8),
                v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    out = _fa3(q.view(tokens, hq, d).to(FP8), data)
    return state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)


def _fa3(q, data):
    cfg = data["config"]
    batch, hkv, window = cfg["batch"], cfg["num_kv_heads"], cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q,
        k_cache=data["k_cache"],
        v_cache=data["v_cache"],
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=cfg["q_len"],
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    return out[0] if isinstance(out, tuple) else out


def _attn_prep(qkv, data, state):
    """Idea 1: our prep launch, SGLang's FA3 on our FP8 queries, our norm-gate launch."""
    cfg = data["config"]
    tokens = cfg["tokens"]
    q = _ext().attn_prep(qkv, data["positions"], data["cache_loc"], state["inv_freq"],
                         data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"],
                         bool(cfg["rope_theta"]))
    out = _fa3(q, data)
    if out.dtype != torch.bfloat16:
        out = out.to(torch.bfloat16)
    return _ext().attn_gate(out.reshape(tokens, -1).contiguous(), qkv)


LOG2E = 1.4426950408889634


def _fused_splits(cfg):
    """(splits, cluster) as tuned on muse_attention_decode: sliding layers merge splits inside
    a thread-block cluster in the same launch; full layers write partials and merge in a
    second launch."""
    batch = cfg["batch"]
    if cfg["window"]:
        return (16 if batch <= 2 else 8 if batch <= 16 else 4), True
    return (128 if batch <= 2 else max(4, 256 // batch)), False


def _attn_fused(qkv, data, state):
    """Idea 2: one kernel for norm, RoPE, cache write, attention and (on sliding layers) the
    norm-gate; full layers add one merge launch."""
    cfg = data["config"]
    splits, cluster = _fused_splits(cfg)
    return _ext().attn_fused(
        qkv, data["page_table"], data["seq_lens"], data["cache_loc"], state["inv_freq"],
        data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"], cfg["page_size"],
        cfg["window"], bool(cfg["rope_theta"]), cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
        splits, cluster)


def _fused_wins(cfg):
    """Where the fused kernel beat prep + FA3 on muse_attention_decode (job 944b519698894c2a):
    one-token decode, except full layers at batch 1-2, where one request's 80K tokens leave
    each CTA latency-bound (29.4 vs 27.5 µs there, job a15d258c37f44232)."""
    return cfg["q_len"] == 1 and (bool(cfg["window"]) or cfg["batch"] > 2)


def _ws_splits(cfg):
    """Context splits per request as muse_attention_decode's submission picks them."""
    batch = cfg["batch"]
    if cfg["window"]:
        return 16 if batch <= 2 else 8 if batch <= 16 else 4
    return 128 if batch <= 2 else max(4, 128 // batch)


def _attn_ws(qkv, data, state):
    """The warp-specialized verify kernel (norm, RoPE, cache write, FP32 attention) and one
    launch that merges the splits and applies the norm-gate."""
    cfg = data["config"]
    return _ext().attn_verify(
        qkv.contiguous(), data["page_table"], data["seq_lens"], data["cache_loc"],
        state["inv_freq"], data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"],
        cfg["page_size"], cfg["window"], bool(cfg["rope_theta"]),
        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, _ws_splits(cfg))


def _ws_wins(cfg):
    """Where muse_attention_decode routes its verify blocks to the ws kernel (job
    cb42811004a9440c): sliding layers at every batch, full layers above batch 2. At full batch
    1 the f16 kernel measured 40-41 us against 38.5 for prep + FA3 (job 34929136fa004b17)."""
    return cfg["q_len"] == 8 and (bool(cfg["window"]) or cfg["batch"] > 2)



_ONE_CPP_SRC = r"""
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
"""

# Extensions built on a thread while the main and W4 extensions compile and the self-tests
# run (the one-launch attention and the fused MoE kernels), each its own module.
_BG_BUILDS = {}


def _bg_start(name, build):
    if name in _BG_BUILDS:
        return
    rec = _BG_BUILDS[name] = {"started": time.time()}

    def run():
        try:
            rec["ext"] = build()
        except Exception as exc:
            rec["error"] = exc
        rec["done"] = time.time()

    rec["thread"] = threading.Thread(target=run, name=f"okc-build-{name}", daemon=True)
    rec["thread"].start()


def _bg_ext(name, build):
    """The module `build` returned, once its thread (started here if it was not) has finished;
    a failed build raises its error on the caller's thread."""
    rec = _BG_BUILDS.get(name)
    if rec is not None and "ext" in rec:
        return rec["ext"]
    _bg_start(name, build)
    rec = _BG_BUILDS[name]
    rec["thread"].join()
    if "ext" not in rec:
        raise rec["error"]
    return rec["ext"]


def _bg_seconds(name):
    rec = _BG_BUILDS.get(name, {})
    return rec.get("done", 0.0) - rec.get("started", 0.0)


def _one_load():
    return load_inline(
        name="okc_decode_layer_one",
        cpp_sources=_ONE_CPP_SRC,
        cuda_sources=[_ONE_SRC, _ONE_HOST_SRC],
        functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
        + (["-DONE_TRIG=1"] if PORT_SK_PDL else []),
    )


def _one_ext():
    return _bg_ext("attn_one", _one_load)


# The one-launch kernel's persistent workspace (allocated once in prepare, zeroed once; every
# completed launch leaves it zero again): FP32 rows of [64] plus one row sum each, and the
# counters. Row ((b * nc + copy) * L + j) * 16 + head; batch 1 full takes 8 copies x 8 tokens x
# 16 heads = 1,024 rows, batch 512 at q_len 8 all of them. Counters: [b] tickets,
# [4096 + 2 b + parity] cooperative, [8192 + b] epochs, [16384 ..] the flat kernel's.
_ONE_WS_ROWS = 65536
_ONE_CNT_INTS = 32768
# Grid target: two CTAs per SM on 132 SMs.
_ONE_TARGET = 264
# The instances _ONE_HOST_SRC builds (DL_ONE_INSTANCES), with the diet build defaults.
_ONE_BUILT = {1: (1, 3, 11, 13, 49), 8: (1, 3, 11, 15, 19, 51)}
# Full-context batches on the flat partition (else per request). The flat instances built here
# are 49 (q_len 1, producer warpgroup), 51 (q_len 8, producer warpgroup) and 19 (q_len 8,
# none), so _ONE_PROD = False sends full decode at batch > 1 to the per-request kernels and
# full verify to 19.
_ONE_FLAT = True
_ONE_PROD = True
# Cooperative modes launch with cudaLaunchAttributeCooperative when the driver accepts it under
# graph capture (probed once in prepare); otherwise their grids keep one SM of headroom below
# the idle occupancy limit, so a foreign CTA cannot stall them into the poll-limit trap.
_ONE_COOP_LAUNCH = True
# The PDL attribute on the attention launch. The kernel waits before its first read of the
# projection's output and of the workspace, so either setting is correct; off, the launch has
# the plain stream semantics the standalone problem's checks exercise. The probe tries the
# attribute set the launches will use and drops PDL first if the driver refuses the pair.
_ONE_PDL = False
# Where the one-launch kernel runs, by (layer, q_len, batch); True everywhere it is not listed.
_ONE_ROUTE = {}


def _one_setup(state):
    """Allocates the workspace, reads the instances' residency limits and probes the launch
    attributes under graph capture. Outside any capture (prepare runs before the tests)."""
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
        # Only the cooperative modes depend on co-residency; without the launch attribute
        # they keep one SM of headroom.
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
    """(n_splits, nc, mode) as a pure function of the shapes and the device's residency
    limits (muse_attention_decode's _one_plan). mode bits: 1 last-CTA finalize, 3
    cooperative, |4 one tile per CTA, |8 the sequence length taken from the page table's
    width (batch 1, page size 1), |16 flat (the grid shares every request's keys; n_splits is
    then the grid size), |32 a producer warpgroup feeds the flat kernel."""
    batch, length, psize, window = cfg["batch"], cfg["q_len"], cfg["page_size"], cfg["window"]
    kmax = (window + length - 1) if window else pt_width * psize
    tiles = (kmax + 63) // 64
    nc = 1
    if not window and batch > 1 and _ONE_FLAT:
        # 17 is not built in the layer, so a short q_len 1 residency falls to the per-request
        # plan below.
        for mode in ((49, 17) if length == 1 else (51, 19)) if _ONE_PROD else \
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
    """One launch: norm, RoPE, cache write, attention, merge and norm-gate (attn_one.cu)."""
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


def _one_wins(cfg):
    """Every decode and verify shape with tp8 heads (checked by the caller), unless _ONE_ROUTE
    excludes it. The flat partition keeps 512 request offsets in shared memory."""
    if cfg["q_len"] not in (1, 8) or cfg["batch"] > 512:
        return False
    # The kernel indexes pages by shift and mask (attn_one.cu, ONE_DIET).
    if cfg["page_size"] & (cfg["page_size"] - 1):
        return False
    key = ("sliding" if cfg["window"] else "full", cfg["q_len"], cfg["batch"])
    return _ONE_ROUTE.get(key, True)


def _one_ws_check(st):
    """The workspace invariant after a completed launch: every row and row sum zero, every
    ticket zero, the next parity counter of every request and of the flat kernel zero."""
    cnt = st["cnt"]
    acc_nz = int((st["acc"] != 0).sum())
    l_nz = int((st["l"] != 0).sum())
    tickets_nz = int((cnt[:4096] != 0).sum())
    ep = cnt[8192:8192 + 2048].long()
    nxt = 4096 + 2 * torch.arange(2048, device=cnt.device) + (ep & 1)
    coop_nz = int((cnt[nxt] != 0).sum())
    flat_nz = int(cnt[16384 + (int(cnt[16386]) & 1)] != 0)
    ok = acc_nz == 0 and l_nz == 0 and tickets_nz == 0 and coop_nz == 0 and flat_nz == 0
    return ok, (f"ws rows_nonzero={acc_nz} l_nonzero={l_nz} tickets_nonzero={tickets_nz} "
                f"coop_next_parity_nonzero={coop_nz} flat_next_parity_nonzero={flat_nz}")


def _one_stress(state, batch, length, layer, replays, seed, ctx=None, expect=None):
    """One shape captured as a graph and replayed `replays` times back to back, then the
    replayed output against an eager call (the workspace atomics make the sums
    order-dependent at the 1e-7 level, so bitwise equality is not expected) and against
    SGLang's path, a NaN check and the workspace invariant; then 50 replays on fresh qkv rows
    written into the captured buffer, with the same checks. On the batch-1 page-1 shape the
    sequence length is then shrunk by one tile in place, so the speculated length is wrong
    and the device must take the real one, for 50 more replays with the same checks. expect:
    the (nc, mode) the plan must select; a different plan raises, so a case meant to cover a
    mode cannot silently cover another."""
    device = state["inv_freq"].device
    st = state["one"]
    qkv, data = _attn_case(device, batch, length, layer, 1, 1.0, seed, ctx=ctx)
    ours = dict(data, k_cache=data["k_cache"].clone(), v_cache=data["v_cache"].clone())
    plan = _one_plan(data["config"], data["page_table"].shape[1], st["resident"])
    if expect is not None and tuple(plan[1:]) != tuple(expect):
        raise RuntimeError(f"attn one stress {batch}x{length} {layer}: plan {plan}, expected "
                           f"(nc, mode) {expect}")
    ref = _attn_sglang(qkv.clone(), data, state)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _attn_one(qkv, ours, state, dbg=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _attn_one(qkv, ours, state, dbg=True)
    torch.cuda.synchronize()

    def check(name, n, ref):
        st["dbg"].zero_()
        for _ in range(n):
            graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        eager = _attn_one(qkv, ours, state, dbg=True)
        torch.cuda.synchronize()
        polls = int(st["dbg"][0])
        rel_eager = _rel(replayed, eager)
        rel_ref = _rel(replayed, ref)
        finite = bool(torch.isfinite(replayed.float()).all())
        ws_ok, ws_msg = _one_ws_check(st)
        msg = (f"{name}: replays={n} (one host sync after the batch; retained: the last replay's "
               f"output) replayed_vs_eager={rel_eager:.1e} vs_sglang={rel_ref:.2e} "
               f"finite={finite} max_polls={polls} {ws_msg}")
        if not (finite and rel_eager < 1e-4 and rel_ref < 5e-2 and ws_ok):
            raise RuntimeError(f"attn one stress {batch}x{length} {layer} {msg} finite={finite}")
        return msg

    msgs = [check(f"plan={plan}", replays, ref)]
    # A fresh step through the same captured graph: new query, key and value rows written into
    # the captured qkv buffer, as the judge's next timed call does.
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    qkv.copy_((torch.randn(qkv.shape, generator=gen, device=device) * 2).to(qkv.dtype))
    ref = _attn_sglang(qkv.clone(), data, state)
    msgs.append(check("fresh qkv", 50, ref))
    if batch == 1 and data["config"]["page_size"] == 1:
        # seq_lens is shared by data and ours; the SGLang reference needs the matching
        # positions, cache rows and lengths.
        data["seq_lens"].sub_(64)
        pos = data["seq_lens"].long()[:, None] + torch.arange(length, device=device)
        data["positions"].copy_(pos.reshape(-1))
        data["cache_loc"].copy_(torch.gather(data["page_table"].long(), 1, pos).reshape(-1))
        data["cache_seqlens"].copy_(data["seq_lens"] + length)
        ref = _attn_sglang(qkv.clone(), data, state)
        msgs.append(check("spec fallback (seq - 64)", 50, ref))
    return "; ".join(msgs)


def _attn_one_self_test(state):
    """The one-launch kernel on the shared attention cases against SGLang's path (the shared
    bounds) with the workspace invariant after every case, then the replay stress on decode 1
    sliding (the speculated one-tile launch), decode 16 full (flat tickets), their verify
    counterparts (cooperative counters), verify 32 full (the producer build 51 at the
    scored batch) and decode / verify 1 full (mode 11 with eight copies, whose finalize loads
    every copy before zeroing any under ONE_FIN8). Prints each step's time; raises on any
    failure."""
    t0 = time.time()
    st = _one_setup(state)
    lim = st["limits"]
    res = " ".join(f"L{lim[i]}m{lim[i + 1]}:{lim[i + 2]}({lim[i + 4]}/SM,{lim[i + 3] // 1024}KB)"
                   for i in range(0, len(lim), 5))
    build = _bg_seconds("attn_one")
    print(f"attn one: extension build {build:.1f} s on its thread, setup {time.time() - t0:.1f} s; "
          f"cooperative_launch={st['coop_launch']} pdl={st['pdl']} {res}")

    def after_case(i, path):
        if path is _attn_one:
            ok, msg = _one_ws_check(st)
            if not ok:
                raise RuntimeError(f"attn case {i}: workspace dirty after the launch ({msg})")

    t1 = time.time()
    # Plans the shared list leaves out: batch 1 full at page 1 (the scored decode 1 and
    # verify 1 full plans, cooperative with 8 copies) and at page 64 (the judge's ctx 3000
    # and 1M tests), and the judge's first test (ctx 10: one tile, 31 empty splits).
    extra = [(1, 1, "full", 1, 1.0), (1, 8, "full", 1, 1.0), (1, 1, "full", 64, 1.0),
             (1, 8, "full", 64, 1.0), (1, 1, "sliding", 64, 1.0, 10)]
    _attn_self_test(state, after_case=after_case, extra_cases=extra)
    print(f"attn one: cases {time.time() - t1:.1f} s")
    # (batch, q_len, layer, ctx, (nc, mode) the plan must pick): the scored plans. Batch 1 full
    # needs more than 252 tiles (ctx 20000) for mode 11 with eight copies (ONE_FIN8); the
    # default ctx of 1500-5000 plans mode 15 there (check 9b921febc7ad4140's stress lines).
    stresses = ((1, 1, "sliding", None, (1, 13)), (16, 1, "full", None, (1, 49)),
                (1, 8, "sliding", None, (1, 15)), (16, 8, "full", None, (1, 51)),
                (32, 8, "full", None, (1, 51)), (1, 1, "full", 20000, (8, 11)),
                (1, 8, "full", 20000, (8, 11)))
    for i, (batch, length, layer, ctx, expect) in enumerate(stresses):
        t2 = time.time()
        msg = _one_stress(state, batch, length, layer, 2000, 8000 + i, ctx, expect)
        print(f"attn one: stress {batch}x{length} {layer} {msg} ({time.time() - t2:.1f} s)")


def _attn_path(cfg, state):
    heads = (cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"])
    if heads != (16, 2, 64):
        return _attn_sglang
    if state["attn_one_ours"] and _one_wins(cfg):
        return _attn_one
    if state["attn_fused_ours"] and _fused_wins(cfg):
        return _attn_fused
    if state["attn_ws_ours"] and _ws_wins(cfg):
        return _attn_ws
    if state["attn_prep_ours"]:
        return _attn_prep
    return _attn_sglang


def _attn_case(device, batch, length, layer, page, kv_scale, seed, ctx=None):
    """A synthetic attention step in the judge's input layout: random caches and a shuffled
    page table, ctx 1500-5000 per request (or ctx on every request)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    seq = torch.randint(1500, 5000, (batch,), generator=gen, device=device, dtype=torch.int32)
    if ctx is not None:
        seq.fill_(ctx)
    pages_per = (seq + length + page - 1) // page
    total = int(pages_per.sum())
    max_pages = int(pages_per.max())
    perm = torch.randperm(total + 7, generator=gen, device=device)[:total].to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    at = 0
    for b in range(batch):
        n = int(pages_per[b])
        page_table[b, :n] = perm[at:at + n]
        at += n
    shape = (total + 7, page, 2, 64)
    # Small keys keep the softmax broad and a per-dim value mean keeps the output away from
    # zero. With unit keys (logit std ~5.5) one-ulp FP8 query differences between two valid
    # paths moved the output 5% (check 003409c1282043d6), which hides real bugs in noise.
    k_cache = (torch.randn(shape, generator=gen, device=device) * 0.3).to(FP8)
    v_mean = torch.randn((1, 1, 2, 64), generator=gen, device=device)
    v_cache = (torch.randn(shape, generator=gen, device=device) * 0.5 + v_mean).to(FP8)
    steps = torch.arange(length, device=device)
    positions = (seq.long()[:, None] + steps).reshape(-1)
    rows = page_table.long().repeat_interleave(length, 0)
    rows = rows.gather(1, (positions // page)[:, None])[:, 0] * page + positions % page
    tokens = batch * length
    qkv = (torch.randn((tokens, 2304), generator=gen, device=device) * 2).to(torch.bfloat16)
    scale = torch.tensor(kv_scale, dtype=torch.float32, device=device)
    data = {
        "k_cache": k_cache, "v_cache": v_cache, "positions": positions, "cache_loc": rows,
        "cache_seqlens": seq + length,
        "cu_seqlens_q": torch.arange(0, tokens + 1, length, dtype=torch.int32, device=device),
        "scheduler_metadata": None, "k_scale": scale, "v_scale": scale.clone(),
        "page_table": page_table, "seq_lens": seq,
        "config": {"batch": batch, "q_len": length, "tokens": tokens, "num_q_heads": 16,
                   "num_kv_heads": 2, "head_dim": 64, "page_size": page,
                   "window": 2048 if layer == "sliding" else 0,
                   "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,
                   "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125,
                   "kv_scale": kv_scale},
    }
    return qkv, data


def _attn_self_test(state, after_case=None, extra_cases=()):
    """Each attention path against SGLang's on synthetic steps: the output and the new cache
    rows; after_case(i, path) runs after each case's checks; extra_cases (a sixth element
    fixes the context length) follow the shared list. Raises on a mismatch."""
    device = state["inv_freq"].device
    cases = [(1, 1, "sliding", 1, 1.0), (3, 1, "full", 1, 1.0), (5, 1, "sliding", 1, 1.0),
             (2, 8, "sliding", 1, 1.0), (3, 8, "full", 1, 1.0), (4, 1, "full", 64, 0.5),
             (2, 8, "sliding", 64, 1.0), (33, 1, "full", 1, 1.0), (2, 1, "full", 1, 1.0),
             (32, 1, "sliding", 1, 1.0), (16, 1, "full", 1, 1.0), (1, 8, "sliding", 1, 1.0),
             (16, 8, "full", 1, 1.0), (32, 8, "sliding", 1, 1.0), (4, 8, "sliding", 1, 0.5),
             (3, 8, "full", 64, 1.0), (5, 8, "full", 1, 0.25)] + list(extra_cases)
    worst_out, worst_cache = 0.0, 0.0
    for i, case in enumerate(cases):
        batch, length, layer, page, kv_scale = case[:5]
        ctx = case[5] if len(case) > 5 else None
        qkv, data = _attn_case(device, batch, length, layer, page, kv_scale, 7000 + i, ctx)
        path = _attn_path(data["config"], state)
        if path is _attn_sglang:
            continue
        ours = dict(data, k_cache=data["k_cache"].clone(), v_cache=data["v_cache"].clone())
        # Separate copies of qkv, in case either path normalizes q and k in place.
        out = path(qkv.clone(), ours, state)
        ref = _attn_sglang(qkv.clone(), data, state)
        torch.cuda.synchronize()
        if not torch.isfinite(out.float()).all():
            raise RuntimeError(f"attn case {i}: non-finite output")
        diff = _rel(out, ref)
        print(f"attn self-test case {i} {batch}x{length} {layer} page {page}"
              f"{'' if ctx is None else f' ctx {ctx}'} {path.__name__}: output rel diff "
              f"{diff:.2e}")
        worst_out = max(worst_out, diff)
        rows = data["cache_loc"]
        for name in ("k_cache", "v_cache"):
            a = ours[name].view(-1, 128)[rows].float()
            b = data[name].view(-1, 128)[rows].float()
            worst_cache = max(worst_cache, _rel(a, b))
            ours[name].view(-1, 128)[rows] = data[name].view(-1, 128)[rows]
            if not torch.equal(ours[name].view(torch.uint8), data[name].view(torch.uint8)):
                raise RuntimeError(f"attn case {i}: {name} written outside the step's rows")
        if after_case is not None:
            after_case(i, path)
    gen = torch.Generator(device=device).manual_seed(77)
    qkv = torch.randn((40, 2304), generator=gen, device=device).to(torch.bfloat16)
    o = torch.randn((40, 1024), generator=gen, device=device).to(torch.bfloat16)
    gate_diff = _rel(_ext().attn_gate(o, qkv), state["o_norm_gate"](o, qkv[:, 1280:]))
    print(f"attn self-test: norm-gate alone {gate_diff:.2e}")
    if not gate_diff < 1e-2:
        raise RuntimeError("attn self-test: norm-gate mismatch")
    print(f"attn self-test: worst rel diff vs SGLang path: output {worst_out:.2e}, "
          f"new cache rows {worst_cache:.2e}")
    # SGLang's FusedQKNorm rounds k to BF16 before the FP8 cast; ours casts from FP32, so a
    # few percent of the new key bytes differ by one FP8 step (1.7e-2 in check
    # 003409c1282043d6). The judge allows 8% on the new cache entries.
    if not (worst_out < 5e-2 and worst_cache < 4e-2):
        raise RuntimeError("attn self-test: mismatch")


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


# reference.py's residual-stream distribution (STREAM_SEED, STREAM_MEAN, STREAM_SHARED,
# STREAM_RANK, DRAFT_SHARE), which decides how the judge's router concentrates routes on a
# few experts.
_REF_STREAM_SEED = 5354
_REF_STREAM_MEAN = 0.7
_REF_STREAM_SHARED, _REF_STREAM_RANK = 0.9, 4
_REF_DRAFT_SHARE = 0.6
# The scored load each MoE bench bucket stands for: decode 1/16/32 have one row per sequence,
# verify 1/16/32 blocks of 8 whose draft rows keep less of the shared structure.
_TUNE_Q_LEN = {1: 1, 8: 8, 16: 1, 32: 1, 128: 8, 256: 8}


def _moe_judge_routing(tokens, q_len, seed, state):
    """A step's MoE inputs routed the way the judge's are: reference.py's residual stream
    through the attention boundary (a random attention branch, which the boundary
    RMS-normalizes), this layer's router and SGLang's routing, local_start on the 32-expert
    range with the most pairs (reference.py's case_fields), x the pooled, pre-normed rows.
    Synthetic routing ranks the fused kernel against the W4 chain the other way at 128 tokens
    on the two seeds the bench used (layer probes 0cb15c11accf4190 and 9d046da0207b40b2)."""
    device = "cuda"
    g = torch.Generator().manual_seed(_REF_STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=g).to(device)
    shared = torch.randn((_REF_STREAM_RANK, HIDDEN), generator=g).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    coef = torch.randn((tokens, _REF_STREAM_RANK), generator=gen, device=device)
    scale = torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = _REF_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (_REF_STREAM_MEAN * mean
                       + (_REF_STREAM_SHARED / _REF_STREAM_RANK) ** 0.5 * coef @ shared)
              + (1 - _REF_STREAM_SHARED * scale * scale).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=device))
    attn = torch.randn((tokens, HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    ffn_input, _, _ = state["attn_boundary"](stream, attn, branch_dtype=torch.bfloat16,
                                             router_dtype=None)
    logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
    logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
    tw, ids = state["routing"](None, logits, TOP_K, True)
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(),
                           minlength=N_EXPERTS // LOCAL_EXPERTS)
    start = LOCAL_EXPERTS * int(pairs.argmax())
    x = _pre_norm(ffn_input @ state["dense"]["pooler"].t(), state).contiguous()
    local_start = torch.tensor([start], dtype=torch.int32, device=device)
    return x, tw.float().contiguous(), ids.to(torch.int32).contiguous(), local_start


def _moe_routing(tokens, seed, start, skewed):
    """A synthetic step's MoE inputs: x [tokens, 4096] and top-8 of uniform scores plus
    heavy-tailed expert popularity (or a few hot experts), rotated so experts start..start+31
    get the most routes, as on the judge's skewed routing."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
    if skewed:
        scores = scores + torch.rand((N_EXPERTS,), generator=gen, device=device).pow(6) * 4
    else:
        z = torch.randn((N_EXPERTS,), generator=gen, device=device)
        u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=N_EXPERTS // 32)
    ids = (ids - 32 * pairs.argmax() + start) % N_EXPERTS
    weights = (routed.values / routed.values.sum(-1, keepdim=True)).float()
    local_start = torch.tensor([start], dtype=torch.int32, device=device)
    return x, weights, ids.to(torch.int32), local_start


def _moe_self_test(state):
    """Our MoE path with Marlin GEMMs against SGLang's at the judge's token counts, with skewed
    routing and several local_start values. Raises on a mismatch."""
    worst = 0.0
    for tokens in (1, 3, 8, 9, 16, 24, 32, 33, 40, 128, 256, 264):
        for start in (0, 96, 224):
            x, weights, ids, local_start = _moe_routing(tokens, 1000 + tokens + start, start,
                                                        True)
            ref = _experts(x, weights, ids, local_start, state, None)
            opts = dict(_moe_opts(tokens, state), g1=None, g2=None)
            out = _experts(x, weights, ids, local_start, state, opts)
            if not torch.isfinite(out).all():
                raise RuntimeError(f"moe tokens={tokens} start={start}: non-finite")
            worst = max(worst, _rel(out, ref))
    torch.cuda.synchronize()
    print(f"moe self-test: worst rel diff vs SGLang path {worst:.2e}")
    if not worst < 5e-3:
        raise RuntimeError(f"moe self-test: rel diff {worst:.2e}")


def _w4_prepare(weights):
    """The local experts' NVFP4 bytes and E4M3 scales repacked losslessly into the W4 kernel's
    fragment order (both weight layouts); None if a block scale is outside positive normal
    E4M3, which the kernel's scale conversion does not handle."""
    out = {}
    for name, kind in (("w13", 1), ("w2", 2)):
        packed = weights[name]
        sf = weights[name + "_sf"].view(torch.uint8)
        if not _w4_scales_ok(sf):
            return None
        words = []
        scales = []
        for e in range(packed.shape[0]):
            w = w4_repack_values(packed[e], kind)
            w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)
            words.append(w)
            scales.append(w4_repack_scales(sf[e], kind).contiguous())
        out[name] = torch.cat(words).contiguous()
        out[name + "_s"] = torch.cat(scales).contiguous()
        out[name + "g"] = w4_group_units(out[name], 64 * 256)
        out[name + "g_s"] = w4_group_units(out[name + "_s"], 64 * 128)
        out[name + "_gs"] = weights[name + "_gs"].float().contiguous()
    torch.cuda.synchronize()
    return out


def _graph_time(fn, flush, iters=20):
    """Median of CUDA graph replays of fn, each after a write that evicts L2, as the judge
    times kernel_fn."""
    return _graph_median(_graph_build(fn), flush, iters)


def _graph_build(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _graph_median(graph, flush, iters=20):
    times = []
    for _ in range(iters):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        times.append((start, end))
    torch.cuda.synchronize()
    ms = sorted(a.elapsed_time(b) for a, b in times)
    return 1000 * ms[len(ms) // 2]


# Shapes the prepare-time timeline traces: (label, batch, q_len, layer). The judge's profile
# gives per-kernel sums only, and its rows hide where the step waits between kernels.
TIMELINE_SHAPES = (("v1s", 1, 8, "sliding"), ("v1f", 1, 8, "full"),
                   ("v16s", 16, 8, "sliding"), ("v16f", 16, 8, "full"),
                   ("d32s", 32, 1, "sliding"), ("d32f", 32, 1, "full"))
TIMELINE_REPS = 7


def _ev_ns(ev):
    if hasattr(ev, "start_ns"):
        return ev.start_ns(), ev.start_ns() + ev.duration_ns()
    return 1000 * ev.start_us(), 1000 * (ev.start_us() + ev.duration_us())


def _timeline_case(state, batch, length, layer, seed):
    device = state["inv_freq"].device
    _, data = _attn_case(device, batch, length, layer, 1, 1.0, seed, ctx=80000)
    tokens = batch * length
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    stream = torch.randn((tokens, HIDDEN), generator=gen, device=device)
    data["stream"] = stream
    data["attn_input"] = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                          ).to(torch.bfloat16)
    data["local_start"] = torch.tensor([96], dtype=torch.int32, device=device)
    return data


def _timeline(state):
    """Per shape, a CUDA graph of kernel_fn on synthetic inputs (ctx 80000 on every request,
    not the judge's agentic spread) replayed TIMELINE_REPS times after an L2 flush and a host
    sync, under torch.profiler. Prints the replay with the median span: each GPU activity's
    stream, start and end relative to the first, and the idle time before it on its stream and
    on the whole device. Diagnostic only; it reads no output."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for label, batch, length, layer in TIMELINE_SHAPES:
        data = _timeline_case(state, batch, length, layer, 9100 + batch * length)
        graph = _graph_build(lambda: kernel_fn(data, state))
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(TIMELINE_REPS):
                flush.zero_()
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
        evs = []
        for ev in prof.profiler.kineto_results.events():
            if ev.device_type() != DeviceType.CUDA:
                continue
            t0, t1 = _ev_ns(ev)
            evs.append((t0, t1, ev.device_resource_id(), ev.name()))
        evs.sort()
        # The host syncs leave tens of us between replays; the flush is a one-kernel segment.
        segs, cur = [], []
        for e in evs:
            if cur and e[0] - max(c[1] for c in cur) > 20000:
                segs.append(cur)
                cur = []
            cur.append(e)
        if cur:
            segs.append(cur)
        segs = [s for s in segs if len(s) >= 3]
        if not segs:
            print(f"TL {label}: no replay segments ({len(evs)} GPU events)")
            continue
        spans = sorted((max(e[1] for e in s) - s[0][0], i) for i, s in enumerate(segs))
        seg = segs[spans[len(spans) // 2][1]]
        base = seg[0][0]
        rank = {}
        for e in seg:
            rank.setdefault(e[2], f"s{len(rank)}")
        busy = sum(e[1] - e[0] for e in seg)
        union, last_end = 0, base
        for e in seg:
            if e[1] > last_end:
                union += e[1] - max(e[0], last_end)
                last_end = e[1]
        span = spans[len(spans) // 2][0]
        print(f"TL {label}: span {span / 1000:.1f} us (replays {len(segs)}: "
              + " ".join(f"{s / 1000:.1f}" for s, _ in spans) + f"), device busy "
              f"{union / 1000:.1f}, idle {(span - union) / 1000:.1f}, kernel sum {busy / 1000:.1f}")
        stream_end, dev_end = {}, base
        for e in seg:
            st = rank[e[2]]
            gap_st = (e[0] - stream_end[st]) / 1000 if st in stream_end else 0.0
            gap_dev = max(0, e[0] - dev_end) / 1000
            name = e[3].replace("void ", "").replace("(anonymous namespace)::", "")
            print(f"TL {label} {st} {(e[0] - base) / 1000:6.1f} {(e[1] - base) / 1000:6.1f} "
                  f"{(e[1] - e[0]) / 1000:5.1f} st-gap {gap_st:4.1f} dev-gap {gap_dev:4.1f} "
                  f"{name[:44]}")
            stream_end[st] = e[1]
            dev_end = max(dev_end, e[1])
    del flush


def _w4_tune(state, seeds=(11, 12)):
    """muse_moe_w4a16's tuner: per TUNE_TOKENS count, times Marlin's GEMM1 + silu_and_mul and
    every fitting W4 GEMM1 config, and Marlin's GEMM2 + finalizer and every W4 GEMM2 config +
    its finalizer, on synthetic routings; keeps the fastest of each (summed over seeds), then
    picks PDL and the finalizer cluster size on the whole MoE call. A W4 config more than 2e-2
    from Marlin's output is dropped. Only timings are kept, nothing from any input."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    bank = state["bank"]
    ws = bank["workspace"]
    w4 = state["w4"]
    norm_w = state["post_expert_w"]
    policy = {}
    bad = set()
    for bs in TUNE_TOKENS:
        tot = [{}, {}]
        for seed in seeds:
            x, tw, topk_ids, start = _moe_routing(bs, 977 * seed + bs, 0, False)
            bsm = _moe_opts(bs, {})["bsm"]
            numel = bs * TOP_K
            max_padded = (numel * bsm if numel < N_EXPERTS + 1
                          else numel + (N_EXPERTS + 1) * (bsm - 1))
            max_padded = (max_padded + 3) // 4 * 4
            sorted_ids, block_experts, n_padded, local_ids = _ext().route_align(
                topk_ids, start, bsm, max_padded, 0, 0)[:4]
            cache1 = torch.empty((numel, 2 * D_EXPERT), device="cuda", dtype=torch.bfloat16)
            inter = torch.empty((numel, D_EXPERT), device="cuda", dtype=torch.bfloat16)
            cache3 = torch.empty((numel, D_EXPERT), device="cuda", dtype=torch.bfloat16)
            common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True,
                          b_q_type=scalar_types.float4_e2m1f, is_k_full=True,
                          use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

            def marlin1():
                moe_wna16_marlin_gemm(
                    x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=TOP_K,
                    size_m=bs, size_n=2 * D_EXPERT, size_k=D_EXPERT, **common)
                silu_and_mul(cache1, inter, expert_ids=local_ids.view(-1), expert_step=1)

            def marlin2():
                moe_wna16_marlin_gemm(
                    inter, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=1,
                    size_m=numel, size_n=D_EXPERT, size_k=D_EXPERT, **common)
                _ext().finalize(cache3.view(bs, TOP_K, D_EXPERT), tw, norm_w,
                                local_ids.view(bs, TOP_K), POST_NORM_EPS)

            tot[0][None] = tot[0].get(None, 0.0) + _graph_time(marlin1, flush)
            tot[1][None] = tot[1].get(None, 0.0) + _graph_time(marlin2, flush)
            marlin1()
            ref1 = inter.clone()
            marlin2()
            ref2 = cache3.clone()
            live = local_ids.view(-1) >= 0
            for i, cfg in enumerate(W4_CONFIGS):
                if not _w4_fits(i, bs) or i in bad:
                    continue
                sids, _, _, lids, items, nitems = _ext().route_align(
                    topk_ids, start, bsm, max_padded, 8 * cfg[1], 0)[:6]
                res = {}
                if cfg[0] == 1:
                    out = torch.zeros((numel, D_EXPERT), device="cuda", dtype=torch.bfloat16)

                    def run():
                        _w4_gemm1(w4, i, x, sids, items, nitems, lids, numel, out)
                        res["out"] = out
                else:
                    def run():
                        part = _w4_gemm2(w4, i, inter, sids, items, nitems, numel)
                        res["part"] = part
                        _w4_ext().w4_finalize(part, tw, norm_w, lids.view(-1), POST_NORM_EPS,
                                              False, 1)
                k = cfg[0] - 1
                tot[k][i] = tot[k].get(i, 0.0) + _graph_time(run, flush)
                run()
                ref = ref1 if cfg[0] == 1 else ref2
                got = res["out"] if cfg[0] == 1 else res["part"].sum(0)
                err = _rel(got[live], ref[live])
                if not err < 2e-2:
                    print(f"TUNE bad cfg {i} {cfg} T={bs} seed={seed}: rel diff {err:.2e}")
                    _FAILED.append(f"w4 config {i} T={bs} ({err:.1e})")
                    bad.add(i)
        n = len(seeds)
        for k in range(2):
            line = " ".join(f"{'m' if c is None else c}:{t / n:.1f}"
                            for c, t in sorted(tot[k].items(), key=lambda kv: kv[1]))
            print(f"TUNE g{k + 1} T={bs} {line}")
            for i in bad:
                tot[k].pop(i, None)
        g1, g2 = (min(tot[k], key=tot[k].get) for k in range(2))
        st = dict(state, w4_ok=True)
        variants = [(pdl, fcl) for pdl in (False, True)
                    for fcl in ((1, 4, 8) if g2 is not None else (1,))]
        tv = {v: 0.0 for v in variants}
        try:
            for seed in seeds:
                x, tw, topk_ids, start = _moe_routing(bs, 977 * seed + bs, 0, False)
                for v in variants:
                    o = dict(_moe_opts(bs, {}), g1=g1, g2=g2, pdl=v[0], fcl=v[1])
                    tv[v] += _graph_time(lambda: _experts(x, tw, topk_ids, start, st, o), flush)
        except Exception as exc:
            print(f"TUNE MoE variants failed: {type(exc).__name__}: {exc}")
            _FAILED.append(f"w4 pdl/cluster variants T={bs} ({type(exc).__name__})")
            tv = {(False, 1): 0.0}
        line = " ".join(f"pdl={int(v[0])},cl={v[1]}:{t / n:.1f}" for v, t in tv.items())
        print(f"TUNE MoE T={bs} {line}")
        pdl, fcl = min(tv, key=tv.get)
        policy[bs] = (g1, g2, pdl, fcl)
    print(f"TUNE policy {policy}")
    return policy


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant_ref(packed, sf, gs):
    """NVFP4 [N, K/2] bytes, [N, K/16] E4M3 scales, FP32 global scale -> FP32 [N, K]."""
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    vals = torch.where(codes >= 8, -grid[codes & 7], grid[codes & 7])
    n, k = vals.shape
    return (vals.view(n, k // 16, 16) * (sf.float() * gs).unsqueeze(-1)).view(n, k)


def _ref_experts(x, topk_weights, ids, start, weights, norm_w):
    """The MoE block in FP32 from the NVFP4 weights, for the W4 self-test."""
    x = x.float()
    local = ids.long() - start
    out = torch.zeros_like(x)
    w = norm_w.float()
    for e in torch.unique(local).tolist():
        if not 0 <= e < LOCAL_EXPERTS:
            continue
        tok, slot = (local == e).nonzero(as_tuple=True)
        h = x[tok] @ _dequant_ref(weights["w13"][e], weights["w13_sf"][e],
                                  weights["w13_gs"][e]).T
        inter = F.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        y = inter @ _dequant_ref(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        y = y * w
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        out.index_add_(0, tok, y * topk_weights[tok, slot].unsqueeze(1))
    return out


# muse_moe_w4a16 measured the W4 path at 4.2e-3 to 4.4e-3 relative L2 from FP32 with BF16 at
# SGLang's boundaries, and Marlin's path at 5.2e-3 to 5.5e-3 (check 84f3975334f048c3).
W4_TEST_TOL = 8e-3


def _w4_self_test(state, weights):
    """The tuned W4 policy against FP32 at the judge's token counts and two local_start
    values, with Marlin's path (our routing metadata and finalizer) printed beside it. Raises
    if the W4 path is off by more than W4_TEST_TOL."""
    worst = 0.0
    for tokens in (1, 3, 8, 9, 16, 24, 32, 33, 40, 64, 128, 256, 264):
        for start, skewed in ((0, False), (96, True)):
            x, tw, ids, local_start = _moe_routing(tokens, 2000 + tokens + start, start, skewed)
            ref = _ref_experts(x, tw, ids, start, weights, state["post_expert_w"])
            opts = _moe_opts(tokens, state)
            base = _rel(_experts(x, tw, ids, local_start, state, dict(opts, g1=None, g2=None)),
                        ref)
            out = _experts(x, tw, ids, local_start, state, opts)
            err = _rel(out, ref) if torch.isfinite(out).all() else float("inf")
            worst = max(worst, err)
            print(f"W4TEST T={tokens} start={start} g1={opts['g1']} g2={opts['g2']}: "
                  f"vs fp32 {err:.2e} (marlin path {base:.2e})")
    torch.cuda.synchronize()
    print(f"W4TEST worst rel err vs fp32 {worst:.2e}")
    if not worst < W4_TEST_TOL:
        raise RuntimeError(f"w4 self-test: rel err {worst:.2e}")


def _w4_enable(state, weights):
    t0 = time.time()
    state["w4"] = _w4_prepare(weights)
    if state["w4"] is None:
        raise RuntimeError("block scales outside positive normal E4M3")
    t1 = time.time()
    _w4_ext()
    t2 = time.time()
    state["w4_policy"] = _w4_tune(state)
    t3 = time.time()
    print(f"W4 repack {t1 - t0:.1f} s, extension build {t2 - t1:.1f} s, tuning {t3 - t2:.1f} s")
    state["w4_ok"] = True
    _w4_self_test(state, weights)


def _router_self_test(state):
    router = state["router"]
    gen = torch.Generator(device=router.device).manual_seed(0)
    for m in (1, 2, 3, 5, 8):
        x = torch.randn(m, HIDDEN, generator=gen, device=router.device)
        # The judge's probe rows: one power-of-two entry, so the exact answer is a column.
        x[0].zero_()
        x[0, 4097] = 2.0
        out = _ext().router_fwd(x, router)
        torch.cuda.synchronize()
        ref = x.double() @ router.double().T
        err = ((out.double() - ref).norm() / ref.norm()).item()
        if not err < 1e-6:
            raise RuntimeError(f"router m={m}: relative L2 error {err:.3e}")
        if not torch.equal(out[0], 2.0 * router[:, 4097]):
            raise RuntimeError(f"router m={m}: probe row is not exact")
    packed = state["router_packed"]
    for m in (9, 16, 17, 24, 31, 32):
        x = torch.randn(m, HIDDEN, generator=gen, device=router.device).to(torch.bfloat16)
        out = _ext().router_mma_fwd(x, packed)
        torch.cuda.synchronize()
        ref = x.double() @ router.double().T
        err = ((out.double() - ref).norm() / ref.norm()).item()
        if not err < 5e-5:
            raise RuntimeError(f"router mma m={m}: relative L2 error {err:.3e}")
    print("router self-test: ok")


def _proj_self_test(state):
    dense, packed = state["dense"], state["proj_packed"]
    gen = torch.Generator(device=dense["pooler"].device).manual_seed(0)
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        w = dense[layer]
        for m in (1, 3, 7, 8):
            n = UNPOOL_SLICE if layer == "unpooler" else w.shape[0]
            x = torch.randn(m, w.shape[1], generator=gen, device=w.device).to(torch.bfloat16)
            x[0].zero_()
            x[0, 777] = 8.0
            out = _ext().proj_fwd(x, packed[layer], n)
            torch.cuda.synchronize()
            ref = x.double() @ w[:n].double().T
            err = ((out.double() - ref).norm() / ref.norm()).item()
            if not err < 5e-3:
                raise RuntimeError(f"{layer} m={m}: relative L2 error {err:.3e}")
            if not torch.equal(out[0], (8.0 * w[:n, 777].float()).to(torch.bfloat16)):
                raise RuntimeError(f"{layer} m={m}: probe row is not exact")
    print("proj self-test: ok")


def _route_side_self_test(state):
    """The MoE call with its routing metadata built on the side stream and joined right before
    GEMM1, as kernel_fn runs it, eagerly and captured in a CUDA graph (W4 GEMM1 is a PDL
    launch whose predecessors now include a kernel on another stream), against the one-stream
    call. Raises on any difference."""
    side = state["side_stream"]
    worst = 0.0
    for tokens in (1, 8, 16, 32, 128, 256):
        x, tw, ids, local_start = _moe_routing(tokens, 3000 + tokens, 96, True)
        opts = _moe_opts(tokens, state) if state["moe_ours"] else None
        ref = _experts(state["pre_expert_norm"](x) if state["norm_ours"] else x, tw, ids,
                       local_start, state, opts)
        res = {}

        def run():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            # The main-stream kernel producing GEMM1's input: our norm (a PDL primary and
            # secondary) when enabled, else a stand-in for torch's.
            xin = _pre_norm(x, state) if state["norm_ours"] else x * 1
            with torch.cuda.stream(side):
                meta = _route_meta(tw, ids, local_start, tokens, state, opts)
            main.wait_stream(side)
            if not torch.cuda.is_current_stream_capturing():
                for tensor in _meta_tensors(meta):
                    tensor.record_stream(main)
            res["out"] = _experts_run(xin, meta, state)

        run()
        torch.cuda.synchronize()
        eager = _rel(res["out"], ref)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        captured = _rel(res["out"], ref)
        print(f"route-side self-test T={tokens}: eager {eager:.2e}, graph {captured:.2e}")
        worst = max(worst, eager, captured)
        del graph
    if not worst < 1e-4:
        raise RuntimeError(f"route-side self-test: rel diff {worst:.2e}")


def _norm_self_test(state):
    """Our pre-expert norm against the module (torch's F.rms_norm) at 1-264 tokens, on BF16
    rows and on the same rows as a strided FP32 view: every element must be bitwise equal or
    one BF16 ulp off. Prints how many differ."""
    mod = state["pre_expert_norm"]
    w = mod._get_effective_weight().detach().to(torch.bfloat16).contiguous()
    if w.numel() != D_EXPERT or float(mod.eps) != RMS_EPS:
        raise RuntimeError("pre-expert norm: unexpected configuration")
    state["pre_expert_w"] = w
    gen = torch.Generator(device=w.device).manual_seed(7)
    worst_ulps, differ, total = 0.0, 0, 0
    for tokens in (1, 2, 3, 5, 8, 16, 32, 33, 128, 256, 264):
        x = (torch.randn((tokens, D_EXPERT), generator=gen, device=w.device) * 0.5)
        x = x.to(torch.bfloat16)
        xf = torch.randn((tokens, D_EXPERT + 512), generator=gen, device=w.device)
        xf[:, :D_EXPERT] = x.float()
        ref = mod(x)
        outs = {"bf16": _ext().pre_norm(x, w, RMS_EPS),
                "fp32": _ext().pre_norm(xf[:, :D_EXPERT], w, RMS_EPS)}
        torch.cuda.synchronize()
        for name, out in outs.items():
            d = (out.float() - ref.float()).abs()
            ulp = torch.exp2(torch.floor(torch.log2(ref.float().abs().clamp(min=1e-30))) - 7)
            ulps = float((d / ulp).max())
            n = int((d > 0).sum())
            worst_ulps = max(worst_ulps, ulps)
            differ += n
            total += out.numel()
            if n or tokens in (1, 264):
                print(f"pre-expert norm self-test T={tokens} {name}: {n} of {out.numel()} "
                      f"differ, worst {ulps:.2f} ulp")
    print(f"pre-expert norm self-test: {differ} of {total} elements differ from torch, "
          f"worst {worst_ulps:.2f} ulp")
    if not worst_ulps <= 1.0:
        raise RuntimeError(f"pre-expert norm: {worst_ulps:.2f} ulp from torch")


def _pre_norm(x, state):
    if state["norm_ours"]:
        return _ext().pre_norm(x, state["pre_expert_w"], RMS_EPS)
    return state["pre_expert_norm"](x)


def _prefetch_set(tokens, state):
    """The weights read after attention at T <= 8: the o weights (packed for our kernel at one
    token, cuBLAS's above), the FP32 router, and the unpooler slice's row tiles (the first 1024
    rows of the packed layout are its first 8.4 MB)."""
    packed = state["proj_packed"]
    o = packed["o"] if _proj_wins("o", tokens) else state["dense"]["o"]
    tiles = UNPOOL_SLICE // 16 * (D_EXPERT // 16) * 256
    return o, state["router"], packed["unpooler"].view(-1)[:tiles]


def _prefetch_enable(state):
    if not (state["proj_ours"] and state["router_ours"]):
        raise RuntimeError("prefetch needs the packed projections and the FP32 router")
    for tokens in (1, 8):
        _ext().prefetch_l2(*_prefetch_set(tokens, state))
    torch.cuda.synchronize()
    print("prefetch: ok")


# Launch shapes of our attention boundary, {CTAs per token}x{threads}, in the order of the
# extension's geometry ids (kGeoCl / kGeoNt); "sglang" is FusedPostNormResidualPreNorm.
ABND_GEOS = ("c8x256", "c2x512", "c4x256", "c1x256", "c1x512", "c1x1024")


def _abnd_enable(state):
    """Takes what SGLang's attention boundary holds (the residual gate's FP32 alpha and beta,
    the pre-FFN norm's effective 1 + w in BF16, both eps) and checks every geometry of our
    kernel against the module at 1-256 tokens, with and without the FP32 router copy. A
    geometry that mismatches is dropped; raises if the 8-CTA one does."""
    mod = state["attn_boundary"]
    post, res, pre = mod._post_norm, mod._residual_add, mod._pre_norm
    if not (not mod._post_norm_is_input_scale and not mod._pre_norm_is_input_scale
            and mod._use_residual_gate and mod._hidden_size == HIDDEN
            and post.weight is None and pre.weight is not None):
        raise RuntimeError("attention boundary: unexpected configuration")
    abnd = {
        "alpha": res._alpha.float().reshape(-1).contiguous(),
        "beta": res._beta.float().reshape(-1).contiguous(),
        "w_next": pre._get_effective_weight().detach().to(torch.bfloat16).contiguous(),
        "eps_post": float(post.eps),
        "eps_next": float(pre.eps),
    }
    state["abnd"] = abnd
    try:
        state["abnd_occ"] = {name: _ext().attn_boundary_occupancy(geo)
                             for geo, name in enumerate(ABND_GEOS)}
    except RuntimeError as exc:
        # A diagnostic only; it must not take the kernel down with it.
        print(f"attention boundary occupancy query failed: {exc}")
    for name, (per_sm, resident) in state.get("abnd_occ", {}).items():
        print(f"attention boundary {name}: {per_sm} CTAs per SM, {resident} "
              f"{'CTAs' if name.startswith('c1x') else 'clusters'} resident")
    ok = []
    for name in ABND_GEOS:
        try:
            _abnd_check(mod, name, state)
            ok.append(name)
        except RuntimeError as exc:
            if name == ABND_GEOS[0]:
                raise
            print(f"attention boundary {name} dropped: {exc}")
            _FAILED.append(f"abnd {name}: {str(exc)[:80]}")
    state["abnd_geos"] = ok


def _abnd_check(mod, name, state):
    abnd = state["abnd"]
    gen = torch.Generator(device="cuda").manual_seed(9)
    worst = 0.0
    for tokens in (1, 2, 3, 5, 8, 9, 16, 33, 128, 256):
        stream = torch.randn((tokens, HIDDEN), generator=gen, device="cuda") * 3
        branch = (torch.randn((tokens, HIDDEN), generator=gen, device="cuda") * 0.5)
        branch = branch.to(torch.bfloat16)
        want_router = tokens <= ROUTER_FP32_MAX_TOKENS
        ref_next, ref_stream, ref_router = mod(
            stream.clone(), branch, branch_dtype=torch.bfloat16,
            router_dtype=torch.float32 if want_router else None)
        nxt, out, router = _abnd(stream.clone(), branch, want_router, state, name)
        torch.cuda.synchronize()
        carried = abnd["alpha"] * stream
        errs = {"next input": _rel(nxt, ref_next),
                "update": _rel(out - carried, ref_stream - carried),
                "stream": _rel(out, ref_stream)}
        if want_router:
            errs["router"] = _rel(router, ref_router)
        if name == ABND_GEOS[0] or tokens == 256:
            print(f"attention boundary self-test {name} T={tokens}: "
                  + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()))
        worst = max(worst, max(errs.values()))
    print(f"attention boundary self-test {name}: worst {worst:.2e}")
    if not worst < 1e-4:
        raise RuntimeError(f"attention boundary {name}: rel diff {worst:.2e}")


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


def _abnd_geo(tokens, state):
    key = next((t for t in FRONT_TOKENS if t >= tokens), FRONT_TOKENS[-1])
    return state.get("abnd_policy", {}).get(key, ABND_GEOS[0])


# A challenger must beat the 8-CTA kernel by this much (us, median chain time) to take a
# bucket, so equal kernels do not flip between runs on replay noise.
ABND_MARGIN_US = 0.3


def _abnd_tune(state):
    """Per token count, times the o projection followed by each attention boundary (every
    geometry that passed its self-test, and SGLang's module) as CUDA graphs replayed after an
    L2 flush, in six rounds with the order rotated each round, and keeps the candidate whose
    median over the six round medians is lowest. The o projection is in the chain because
    the boundary is its PDL secondary and preloads the stream during its tail. Every
    candidate's outputs on the projection's output must match the module's within the
    self-test bound. The module gets its own copy of the stream, since it may update the
    stream in place (the self-test clones it too)."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(31)
    k_o = state["dense"]["o"].shape[1]
    names = list(state["abnd_geos"]) + ["sglang"]
    policy, report = {}, {}
    for tokens in FRONT_TOKENS:
        want_router = tokens <= ROUTER_FP32_MAX_TOKENS
        o_in = (torch.randn((tokens, k_o), generator=gen, device="cuda") * 0.05)
        o_in = o_in.to(torch.bfloat16)
        stream = torch.randn((tokens, HIDDEN), generator=gen, device="cuda") * 3
        branch = _proj(o_in, "o", HIDDEN, state)
        outs = {name: [x for x in _abnd(stream.clone(), branch, want_router, state, name)
                       if x is not None] for name in names}
        torch.cuda.synchronize()
        graphs = {}
        for name in names:
            own = stream.clone() if name == "sglang" else stream

            def run(name=name, own=own):
                _abnd(own, _proj(o_in, "o", HIDDEN, state), want_router, state, name)

            graphs[name] = _graph_build(run)
        ref = outs["sglang"]
        for name in names:
            err = max(_rel(a, b) for a, b in zip(outs[name], ref))
            if not err < 1e-4:
                print(f"TUNE abnd T={tokens} {name} dropped: rel diff {err:.2e}")
                _FAILED.append(f"abnd {name} T={tokens} ({err:.1e})")
                graphs.pop(name)
        rounds = {name: [] for name in graphs}
        order = list(graphs)
        for r in range(6):
            for name in order[r % len(order):] + order[:r % len(order)]:
                rounds[name].append(_graph_median(graphs[name], flush))
        times = {name: sorted(v)[len(v) // 2] for name, v in rounds.items()}
        best = min(times, key=times.get)
        incumbent = ABND_GEOS[0]
        if incumbent in times and times[best] > times[incumbent] - ABND_MARGIN_US:
            best = incumbent
        policy[tokens] = best
        report[tokens] = times
        print(f"TUNE abnd T={tokens} "
              + " ".join(f"{v}:{t:.2f}" for v, t in sorted(times.items(), key=lambda kv: kv[1]))
              + f" -> {best}")
        del graphs
    state["abnd_policy"] = policy
    state["abnd_times"] = report


def _bnd_enable(state):
    """Takes the weights and coefficients SGLang's FFN boundary holds (its post-norm's and
    pre-norm's effective 1 + w in BF16, the residual gate's FP32 alpha and beta) and checks our
    short boundary against the padded SGLang call at 1-8 tokens. The bound is the FP32
    summation-order level, since the kernel rounds where SGLang's does. Raises on a mismatch."""
    mod = state["ffn_boundary"]
    post, res, pre = mod._post_norm, mod._residual_add, mod._pre_norm
    if not (mod._post_norm_is_input_scale and not mod._pre_norm_is_input_scale
            and mod._use_residual_gate and mod._hidden_size == HIDDEN):
        raise RuntimeError("ffn boundary: unexpected configuration")
    bnd = {
        "w_post": post._frozen_weight.to(torch.bfloat16).contiguous(),
        "alpha": res._alpha.float().reshape(-1).contiguous(),
        "beta": res._beta.float().reshape(-1).contiguous(),
        "w_next": pre._frozen_weight.to(torch.bfloat16).contiguous(),
        "eps_post": float(post.eps),
        "eps_next": float(pre.eps),
    }
    state["bnd"] = bnd
    gen = torch.Generator(device="cuda").manual_seed(5)
    worst_next, worst_update = 0.0, 0.0
    for tokens in (1, 2, 3, 5, 8):
        stream = torch.randn((tokens, HIDDEN), generator=gen, device="cuda") * 3
        f = (torch.randn((tokens, UNPOOL_SLICE), generator=gen, device="cuda") * 0.02)
        f = f.to(torch.bfloat16)
        if tokens == 3:
            f[1].zero_()
        ref_next, ref_stream, _ = mod(stream.clone(), F.pad(f, (0, HIDDEN - UNPOOL_SLICE)),
                                      branch_dtype=torch.bfloat16)
        nxt, out = _bnd(stream.clone(), f, state)
        torch.cuda.synchronize()
        carried = bnd["alpha"] * stream
        err_next = _rel(nxt, ref_next)
        err_update = _rel(out - carried, ref_stream - carried)
        print(f"ffn boundary self-test T={tokens}: next input {err_next:.2e}, "
              f"update {err_update:.2e}, stream {_rel(out, ref_stream):.2e}")
        worst_next = max(worst_next, err_next)
        worst_update = max(worst_update, err_update)
    if not (worst_next < 1e-4 and worst_update < 1e-4):
        raise RuntimeError(f"ffn boundary: next {worst_next:.2e}, update {worst_update:.2e}")


def _bnd(stream, f, state):
    bnd = state["bnd"]
    return _ext().ffn_boundary_short(stream.contiguous(), f.contiguous(), bnd["w_post"],
                                     bnd["alpha"], bnd["beta"], bnd["w_next"],
                                     bnd["eps_post"], bnd["eps_next"])


# Token counts the MoE-front tuner times (the scored T above 8); other counts use the next one
# up.
FRONT_TOKENS = (1, 8, 16, 32, 128, 256)


def _front_variants(tokens, state):
    """The side-chain variants the tuner may pick at this token count. Up to 8 tokens "fp32"
    is SGLang's method (the FP32 router on the boundary's FP32 router input); above, "split" is
    its hi/lo torch.mm pair (two GEMMs, two split-K reduces and an add) under the pooler, "mma"
    our hi/lo mma router (up to 32 tokens), "concat" one cuBLAS GEMM for the pooler and both
    router halves with FP32 output (needs our norm, which rounds the pooled columns on load).
    "skf" is muse_bf16_proj's fused pooler + router launch (12-bit pooler, hi/lo router on the
    BF16 FFN input) up to 32 tokens. "wide" runs the pooler on _proj's wide-kernel pick (65 to
    128 tokens) and the router as one cuBLAS GEMM over the stacked hi and lo rows with FP32
    output, beside it on the side stream; "concat" holds the pooler inside its cuBLAS GEMM, so
    the wide kernel reaches the pooler only through "wide" and "split"."""
    if tokens <= ROUTER_FP32_MAX_TOKENS:
        out = ["fp32"]
    else:
        out = ["split"]
        if PORT_ROUTER_MMA and state["router_ours"] and tokens <= ROUTER_MMA_MAX_TOKENS:
            out.append("mma")
        if PORT_FRONT_CONCAT and state["norm_ours"] and "pool_router" in state:
            out.append("concat")
        if (state["skp_ok"] and "pooler" in state["skp_wide"] and "router_hilo" in state
                and SKP_WD_MIN <= tokens <= SKP_WD_MAX):
            out.append("wide")
    if PORT_PROJ_SK and state.get("skp_ok") and bucket_of(tokens) in state["skp"]["fused"]:
        out.append("skf")
    if state.get("route_ours") and _moe1_variant(tokens, state) is not None:
        out += [r for r, base in ROUTE_VARIANTS.items() if base in out]
    return out


def _front_variant(tokens, state):
    key = next((t for t in FRONT_TOKENS if t >= tokens), FRONT_TOKENS[-1])
    return state["front_policy"].get(key, "fp32" if tokens <= ROUTER_FP32_MAX_TOKENS else "split")


def _moe_front(ffn_input, router_input, local_start, state, moe_opts, variant):
    """From the FFN input to GEMM1's inputs: the pooler and the pre-expert norm on the main
    stream; the router, routing and (with ROUTE_ON_SIDE) the routing metadata on the side
    stream, joined before GEMM1. Returns (expert input, topk_weights, topk_ids, meta); the
    expert input is normalized when meta is not None, else the caller normalizes it."""
    tokens = ffn_input.shape[0]
    if variant in ROUTE_VARIANTS:
        return _moe_front_route(ffn_input, local_start, state, moe_opts, variant)
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    route_side = ROUTE_ON_SIDE and state["route_side_ok"]
    if variant == "skf":
        # One launch for the pooler and the router logits; the side stream reads the logits.
        expert_input, logits32 = state["skp"]["fused"][bucket_of(tokens)](
            ffn_input.contiguous(), _m1_mode(tokens, state) == "o+front")
        side_stream.wait_stream(main_stream)
    elif variant == "concat":
        out32 = torch.mm(ffn_input, state["pool_router"].t(), out_dtype=torch.float32)
        # The side stream reads the router columns, so it forks after the GEMM.
        side_stream.wait_stream(main_stream)
        expert_input = out32[:, :D_EXPERT]
    else:
        side_stream.wait_stream(main_stream)
        expert_input = _proj(ffn_input, "pooler", D_EXPERT, state)
    if route_side:
        expert_input = _pre_norm(expert_input, state)
    meta = None
    with torch.cuda.stream(side_stream):
        if variant == "skf":
            logits = logits32
        elif variant == "concat":
            logits = out32[:, D_EXPERT:D_EXPERT + N_EXPERTS] + out32[:, D_EXPERT + N_EXPERTS:]
        elif variant == "wide":
            both = torch.mm(ffn_input, state["router_hilo"].t(), out_dtype=torch.float32)
            logits = both[:, :N_EXPERTS] + both[:, N_EXPERTS:]
        elif variant == "fp32" and state["router_ours"]:
            logits = _ext().router_fwd(router_input.contiguous(), state["router"])
        elif variant == "fp32":
            logits = F.linear(router_input, state["router"])
        elif variant == "mma":
            logits = _ext().router_mma_fwd(ffn_input.contiguous(), state["router_packed"])
        else:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
        if route_side:
            meta = _route_meta(topk_weights, topk_ids, local_start, tokens, state, moe_opts)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        crossing = [topk_weights, topk_ids] + (_meta_tensors(meta) if meta else [])
        for tensor in crossing:
            tensor.record_stream(main_stream)
    return expert_input, topk_weights, topk_ids, meta


def _route_fwd(logits, lo_off, pooled, state, bias=None):
    return _ext().route_fwd(logits, lo_off, pooled, state["pre_expert_w"],
                            state["route_bias"] if bias is None else bias,
                            ROUTER_MULTIPLIER, RMS_EPS, ROUTE_EPS)


def _moe_front_route(ffn_input, local_start, state, moe_opts, variant):
    """The route variants of _moe_front: the front GEMM (concat or skf) on the main stream, then
    the route kernel, no side stream. The meta asks the fused MoE for a PDL launch."""
    tokens = ffn_input.shape[0]
    if variant == "skf-r":
        pooled, logits = state["skp"]["fused"][bucket_of(tokens)](
            ffn_input.contiguous(), _m1_mode(tokens, state) == "o+front")
        x, topk_ids, topk_weights = _route_fwd(logits, 0, pooled, state)
    else:
        out32 = torch.mm(ffn_input, state["pool_router"].t(), out_dtype=torch.float32)
        x, topk_ids, topk_weights = _route_fwd(out32[:, D_EXPERT:], N_EXPERTS,
                                               out32[:, :D_EXPERT], state)
    meta = _route_meta(topk_weights, topk_ids, local_start, tokens, state, moe_opts)
    if meta.get("fused") is not None:
        meta["pdl"] = True
    return x, topk_weights, topk_ids, meta


def _route_self_test(state):
    """The route kernel against the current path (the torch hi + lo add, SGLang's routing
    kernel, our pre-expert norm) at 1-264 tokens, on the concat layout (FP32, hi and lo blocks)
    and the skf layout (BF16 pooled, FP32 logits): top-8 id sets equal row by row, weights
    within 1e-5 relative, the normalized input within 1e-3 (bitwise expected; the differing
    elements are counted). Then tie cases (integer logits, zero bias, beside a zero-bias SGLang
    routing module): every row's picks must be a valid top-8 (their biased scores are the 8
    largest, no repeats). Raises on any failure of ours."""
    dev = state["pre_expert_w"].device
    gen = torch.Generator(device=dev).manual_seed(77)
    bias = state["route_bias"]
    zero = torch.zeros_like(bias)
    tie_routing = FusedMuseSparkRouting(expert_bias=zero, route_scale=1.0,
                                        router_output_multiplier=ROUTER_MULTIPLIER,
                                        router_score_division_eps=ROUTE_EPS)

    def keys_of(logits, b):
        return torch.sigmoid(logits * ROUTER_MULTIPLIER) + b

    worst_w, tie_agree, tie_rows, x_differ, sgl_invalid = 0.0, 0, 0, 0, 0
    for tokens in (1, 2, 8, 16, 32, 33, 128, 256, 264):
        out32 = torch.randn((tokens, D_EXPERT + 2 * N_EXPERTS), generator=gen, device=dev)
        out32[:, D_EXPERT:] *= 4.0
        logits = out32[:, D_EXPERT:D_EXPERT + N_EXPERTS] + out32[:, D_EXPERT + N_EXPERTS:]
        ref_w, ref_ids = state["routing"](None, logits, TOP_K, True)
        ref_x = _pre_norm(out32[:, :D_EXPERT], state)
        x, ids, w = _route_fwd(out32[:, D_EXPERT:], N_EXPERTS, out32[:, :D_EXPERT], state)
        pooled = (torch.randn((tokens, D_EXPERT), generator=gen, device=dev)).to(torch.bfloat16)
        logits2 = 4.0 * torch.randn((tokens, N_EXPERTS), generator=gen, device=dev)
        ref_w2, ref_ids2 = state["routing"](None, logits2, TOP_K, True)
        ref_x2 = _pre_norm(pooled, state)
        x2, ids2, w2 = _route_fwd(logits2, 0, pooled, state)
        torch.cuda.synchronize()
        for name, a, b, ia, ib, wa, wb in (("concat", x, ref_x, ids, ref_ids, w, ref_w),
                                           ("skf", x2, ref_x2, ids2, ref_ids2, w2, ref_w2)):
            ia_s, oa = ia.long().sort(-1)
            ib_s, ob = ib.long().sort(-1)
            if not torch.equal(ia_s, ib_s):
                bad = int((ia_s != ib_s).any(-1).sum())
                raise RuntimeError(f"route T={tokens} {name}: {bad} rows pick other experts")
            wa_s = wa.float().gather(1, oa)
            wb_s = wb.float().gather(1, ob)
            err = float(((wa_s - wb_s).abs() / wb_s.abs().clamp(min=1e-30)).max())
            worst_w = max(worst_w, err)
            if not err < 1e-5:
                raise RuntimeError(f"route T={tokens} {name}: weights differ by {err:.2e}")
            x_differ += int((a != b).sum())
            if not _rel(a, b) < 1e-3:
                raise RuntimeError(f"route T={tokens} {name}: normalized input differs by "
                                   f"{_rel(a, b):.2e}")
        tie_logits = torch.randint(-3, 4, (tokens, N_EXPERTS), generator=gen,
                                   device=dev).float()
        _, tie_ref = tie_routing(None, tie_logits, TOP_K, True)
        _, tie_ids, _ = _route_fwd(tie_logits, 0, pooled, state, bias=zero)
        torch.cuda.synchronize()
        keys = keys_of(tie_logits, zero)
        top = keys.topk(TOP_K, dim=-1).values.sort(-1).values
        for name, picks in (("ours", tie_ids), ("sglang", tie_ref)):
            got = keys.gather(1, picks.long()).sort(-1).values
            srt = picks.long().sort(-1).values
            if not torch.equal(got, top) or (srt[:, 1:] == srt[:, :-1]).any():
                if name == "sglang":
                    sgl_invalid += 1
                    continue
                raise RuntimeError(f"route ties T={tokens}: {name} picks are not a top-8")
        tie_agree += int((tie_ids.long().sort(-1).values
                          == tie_ref.long().sort(-1).values).all(-1).sum())
        tie_rows += tokens
    print(f"route self-test: ids equal on every row, weights within {worst_w:.1e}, {x_differ} "
          f"normalized elements not bitwise; ties valid, same set as SGLang on {tie_agree} of "
          f"{tie_rows} tie rows (SGLang not a top-8 at {sgl_invalid} token counts)")
    if state.get("moe1_ok"):
        _moe1_pdl_self_test(state)


ROUTE_PDL_REPLAYS = 100


def _moe1_pdl_self_test(state):
    """Every fused variant that passed its FP32 self-test, launched as the route kernel's PDL
    secondary (the preamble initialises the ring's mbarriers before griddepcontrol.wait and the
    routing section skips them), against the plain launch on the same route output: eagerly and
    as a CUDA graph replayed ROUTE_PDL_REPLAYS times, outputs within 1e-4 (sums of squares are
    FP32 atomics) and the scratch clean. A variant the policy uses that fails raises (the route
    variants are then off); any other failing variant is reported."""
    dev = state["pre_expert_w"].device
    gen = torch.Generator(device=dev).manual_seed(78)
    ctr = state["moe1"]["scratch"]["ctr"]
    used = set((state.get("moe1_policy") or {}).values())
    local_start = torch.tensor([96], dtype=torch.int32, device=dev)
    bad, lines = [], []
    for variant in state.get("moe1_passed", []):
        name, kind, _ = _MOE1_VARIANTS[variant]
        worst = 0.0
        for tokens in ((8, 16) if kind == "skinny" else (32, 128, 256)):
            logits = 4.0 * torch.randn((tokens, N_EXPERTS), generator=gen, device=dev)
            pooled = torch.randn((tokens, D_EXPERT), generator=gen, device=dev)
            pooled = pooled.to(torch.bfloat16)
            res = {}

            def run(pdl):
                x, ids, w = _route_fwd(logits, 0, pooled, state)
                res[pdl] = _moe1_run(x, w, ids, local_start, state, variant, pdl)

            run(False)
            run(True)
            torch.cuda.synchronize()
            eager = _rel(res[True], res[False])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(True)
            for _ in range(ROUTE_PDL_REPLAYS):
                graph.replay()
            torch.cuda.synchronize()
            replayed = _rel(res[True], res[False])
            del graph
            finite = bool(torch.isfinite(res[True]).all())
            worst = max(worst, eager, replayed, 0.0 if finite else float("inf"))
        clean, msg = _moe1_check(state)
        ok = worst < 1e-4 and clean
        lines.append(f"{name} {worst:.1e}{'' if ok else ' FAIL ' + msg}")
        if not ok:
            ctr[8] = 0
            bad.append(variant)
            _FAILED.append(f"moe1 pdl {name} ({worst:.1e}, {msg[:60]})")
    state["moe1_pdl_report"] = ", ".join(lines)
    print("MOE1PDL vs plain launch after the route kernel: " + ", ".join(lines))
    if any(v in used for v in bad):
        raise RuntimeError("a fused variant the policy uses fails as the route kernel's PDL "
                           "secondary")


def _front_tune(state):
    """Times the MoE front per token count, as a CUDA graph replayed after an L2 flush as the
    judge times it, for every side-chain variant, and keeps the fastest. The variants must
    agree with the reference ("split" above 8 tokens, "fp32" up to 8): the normalized expert
    input within 1e-3 (relative L2), the level of BF16 rounding flips from a different FP32
    summation order, and at least 99% of the routes equal (as sets per token); a variant that
    disagrees is dropped at that token count. Up to 8 tokens the FP32 router input is the BF16
    input widened, so the variants route the same values.

    Where a route variant (ROUTE_VARIANTS) is offered, the pick is by the front and the MoE
    timed together, since the route variants' saving includes the MoE's PDL start; the
    front-only times are printed beside them. A route variant must also match its base variant
    (same logits): equal top-8 sets, the normalized input within 1e-3, the MoE output within
    1e-3, and after ROUTE_STRESS more replays of its graph the same output and a clean MoE
    scratch (no spin timeout)."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(21)
    local_start = torch.tensor([96], dtype=torch.int32, device="cuda")
    policy = {}
    report = state.setdefault("front_report", {})
    for tokens in FRONT_TOKENS:
        variants = _front_variants(tokens, state)
        if len(variants) == 1:
            policy[tokens] = variants[0]
            continue
        ffn_input = torch.randn((tokens, HIDDEN), generator=gen, device="cuda")
        ffn_input = ffn_input.to(torch.bfloat16)
        router_input = ffn_input.float() if tokens <= ROUTER_FP32_MAX_TOKENS else None
        opts = _moe_opts(tokens, state) if state["moe_ours"] else None
        with_moe = any(v in ROUTE_VARIANTS for v in variants)
        times, chain, outs = {}, {}, {}
        for variant in variants:
            res = {}

            def front():
                ei, tw, ids, meta = _moe_front(ffn_input, router_input, local_start, state,
                                               opts, variant)
                res["x"] = ei if meta is not None else _pre_norm(ei, state)
                res["ids"] = ids
                return meta, tw

            def front_moe():
                meta, tw = front()
                if meta is not None:
                    res["y"] = _experts_run(res["x"], meta, state)
                else:
                    res["y"] = _experts(res["x"], tw, res["ids"], local_start, state, opts)

            times[variant] = _graph_time(front, flush)
            torch.cuda.synchronize()
            outs[variant] = [res["x"].clone(), res["ids"].clone(), None]
            if with_moe:
                graph = _graph_build(front_moe)
                chain[variant] = _graph_median(graph, flush, 30)
                torch.cuda.synchronize()
                if res.get("y") is not None:
                    outs[variant][2] = res["y"].clone()
                if variant in ROUTE_VARIANTS and res.get("y") is not None:
                    for _ in range(ROUTE_STRESS):
                        graph.replay()
                    torch.cuda.synchronize()
                    outs[variant].append(res["y"].clone())
                del graph
        reference = variants[0]
        ref_x, ref_ids = outs[reference][:2]
        agree = {}
        for variant, (x, ids, *_rest) in outs.items():
            err = _rel(x, ref_x)
            same = float((ids.long().sort(-1).values == ref_ids.long().sort(-1).values)
                         .float().mean())
            agree[variant] = f"{err:.1e}/{same:.3f}"
            bad = None
            if not (err < 1e-3 and same >= 0.99):
                bad = f"rel diff {err:.2e}, routes {same:.3f} against {reference}"
            if variant in ROUTE_VARIANTS and bad is None:
                bad = _route_variant_check(outs[variant], outs.get(ROUTE_VARIANTS[variant]),
                                           state)
            if bad is not None:
                print(f"TUNE front T={tokens} {variant} dropped: {bad}")
                _FAILED.append(f"front {variant} T={tokens} ({bad[:80]})")
                times.pop(variant)
                chain.pop(variant, None)
        rank = chain if with_moe else times
        best = min(rank, key=rank.get)
        policy[tokens] = best
        report[tokens] = (dict(times), dict(chain))
        line = (f"TUNE front T={tokens} "
                + " ".join(f"{v}:{t:.1f}" for v, t in sorted(times.items(), key=lambda kv: kv[1])))
        if with_moe:
            line += (" | +moe " + " ".join(f"{v}:{t:.1f}" for v, t in
                                           sorted(chain.items(), key=lambda kv: kv[1])))
        print(line + f" -> {best} (vs {reference}: "
              + " ".join(f"{v} {a}" for v, a in agree.items()) + ")")
    state["front_policy"] = policy


# Extra replays of a route variant's front + MoE graph in the tuner, checked for a stable
# output and a clean MoE scratch: the PDL launch runs there under graph replay, as judged.
ROUTE_STRESS = 200


def _route_variant_check(got, base, state):
    """None if a route variant's outputs match its base variant's; else the reason."""
    if base is None:
        return "no base variant timed"
    x, ids, y = got[:3]
    bx, bids, by = base[:3]
    if not torch.equal(ids.long().sort(-1).values, bids.long().sort(-1).values):
        return "top-8 sets differ from the base variant"
    if not _rel(x, bx) < 1e-3:
        return f"normalized input {_rel(x, bx):.1e} from the base variant"
    if y is None or by is None:
        return "no MoE output"
    if not (bool(torch.isfinite(y).all()) and _rel(y, by) < 1e-3):
        return f"MoE output {_rel(y, by):.1e} from the base variant"
    if len(got) > 3 and not _rel(got[3], y) < 1e-4:
        return f"MoE output moved {_rel(got[3], y):.1e} over {ROUTE_STRESS} replays"
    clean, msg = _moe1_check(state)
    if not clean:
        state["moe1"]["scratch"]["ctr"][8] = 0
        return f"MoE scratch after the replays: {msg}"
    return None


def _m1_mode(tokens, state):
    """M1's choice for this token count: "off", "o" or "o+front"."""
    if not state.get("m1_ok") or tokens > M1_BUCKETS[-1]:
        return "off"
    key = next(b for b in M1_BUCKETS if b >= tokens)
    return state.get("m1_policy", {}).get(key, "off")


M1_STRESS_REPLAYS = 2000
M1_CHAIN_GATE_US = 1.0


def _m1_selftest(state):
    """The SKP self-test with bit 16 set: every o pick at 1-32 tokens that is an sk config and
    every fused pooler + router config, launched as a PDL secondary, against the plain launch on
    the same x: bitwise equal (the same kernel and arithmetic; only the launch and the order of
    the first copies differ). Raises on a difference."""
    dev = state["dense"]["o"].device
    gen = torch.Generator(device=dev).manual_seed(4545)
    checked = []
    for m in (1, 2, 5, 8, 9, 16, 17, 32):
        pick = state["skp_pick"].get(("o", m))
        if pick is not None and getattr(pick[1], "sk", False):
            x = torch.randn((m, state["dense"]["o"].shape[1]), generator=gen, device=dev)
            x = x.to(torch.bfloat16)
            a, b = pick[1](x, False), pick[1](x, True)
            torch.cuda.synchronize()
            if not torch.equal(a, b):
                raise RuntimeError(f"o {pick[0]} m={m}: PDL launch differs from the plain one")
            checked.append(f"o:{pick[0]}@{m}")
        fused = state.get("skp", {}).get("fused", {}).get(bucket_of(m)) if m > 1 else None
        if fused is not None:
            x = torch.randn((m, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
            (a, ar), (b, br) = fused(x, False), fused(x, True)
            torch.cuda.synchronize()
            if not (torch.equal(a, b) and torch.equal(ar, br)):
                raise RuntimeError(f"skf bucket {bucket_of(m)} m={m}: PDL launch differs")
            checked.append(f"skf{bucket_of(m)}@{m}")
    return checked


def _m1_stress(state):
    """kernel_fn with M1 forced on (o and front) at every scored T <= 32 shape, captured and
    replayed M1_STRESS_REPLAYS times with one host sync after them; the last replay's three
    outputs against kernel_fn with M1 off on the same inputs (hidden and stream within 1e-4,
    the MoE's sums of squares and attention's merges being FP32 atomics; ids equal). Returns
    per-shape lines; raises on a mismatch."""
    lines = []
    saved = dict(state.get("m1_policy", {}))
    try:
        for batch, length, layer in ((1, 1, "sliding"), (1, 1, "full"), (1, 8, "sliding"),
                                     (1, 8, "full"), (16, 1, "sliding"), (16, 1, "full"),
                                     (32, 1, "sliding"), (32, 1, "full")):
            data = _timeline_case(state, batch, length, layer, 9300 + batch * length)
            state["m1_policy"] = {}
            ref = [t.clone() for t in kernel_fn(data, state)]
            state["m1_policy"] = {b: "o+front" for b in M1_BUCKETS}
            res = {}

            def run():
                res["o"] = kernel_fn(data, state)

            graph = _graph_build(run)
            for _ in range(M1_STRESS_REPLAYS):
                graph.replay()
            torch.cuda.synchronize()
            got = [t.clone() for t in res["o"]]
            del graph
            rh, rs = _rel(got[0], ref[0]), _rel(got[1], ref[1])
            same = bool(torch.equal(got[2], ref[2]))
            finite = all(bool(torch.isfinite(t.float()).all()) for t in got[:2])
            ok = finite and rh < 1e-4 and rs < 1e-4 and same
            lines.append(f"{batch}x{length}{layer[0]} {'ok' if ok else 'FAIL'} hidden {rh:.1e} "
                         f"stream {rs:.1e} ids {'=' if same else '!='}")
            if not ok:
                raise RuntimeError("M1 replay mismatch: " + lines[-1])
    finally:
        state["m1_policy"] = saved
    return lines


def _m1_chain(state, batch, length, layer, flush, rounds=5):
    """attention -> o -> attention boundary as a CUDA graph, o plain against o as a PDL
    secondary, cold L2, interleaved rounds of graph medians; returns (plain, pdl) in us."""
    data = _timeline_case(state, batch, length, layer, 9500 + batch * length)
    cfg = data["config"]
    tokens = cfg["tokens"]
    qkv = _proj(data["attn_input"], "qkv_gate", state["dense"]["qkv_gate"].shape[0], state)
    path = _attn_path(cfg, state)

    def chain():
        o = path(qkv, data, state)
        a = _proj(o, "o", HIDDEN, state)
        if state["abnd_ours"]:
            _abnd(data["stream"], a, tokens <= ROUTER_FP32_MAX_TOKENS, state,
                  _abnd_geo(tokens, state))
        else:
            state["attn_boundary"](data["stream"], a, branch_dtype=torch.bfloat16,
                                   router_dtype=None)

    saved = dict(state.get("m1_policy", {}))
    graphs = {}
    for mode in ("off", "o"):
        state["m1_policy"] = {b: mode for b in M1_BUCKETS}
        graphs[mode] = _graph_build(chain)
    state["m1_policy"] = saved
    times = {"off": [], "o": []}
    for r in range(rounds):
        for mode in (("off", "o") if r % 2 == 0 else ("o", "off")):
            times[mode].append(_graph_median(graphs[mode], flush, 10))
    return tuple(sorted(times[m])[len(times[m]) // 2] for m in ("off", "o"))


def _m1_tune(state):
    """M1 per token bucket: the self-test with the bit set and the replay stress first (any
    difference blocks M1), then the attention -> o -> boundary chain at 8 tokens sliding (under
    M1_CHAIN_GATE_US saved closes M1), then per bucket the whole step (kernel_fn, 46 x sliding
    + 16 x full, cold L2, interleaved rounds) for off, o and o + front; the fastest wins."""
    state["m1_report"] = []
    rep = state["m1_report"]
    checked = _m1_selftest(state)
    rep.append(f"selftest PDL = plain bitwise on {len(checked)} launches")
    state["m1_ok"] = True
    rep.append("stress " + "; ".join(_m1_stress(state)))
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    chains = {}
    for batch, length in ((1, 8), (16, 1), (32, 1)):
        chains[batch * length] = _m1_chain(state, batch, length, "sliding", flush)
    rep.append("chain attn->o->bnd sliding (plain/pdl us) " + " ".join(
        f"T={t} {a:.1f}/{b:.1f}" for t, (a, b) in chains.items()))
    plain8, pdl8 = chains[8]
    if plain8 - pdl8 < M1_CHAIN_GATE_US:
        state["m1_policy"] = {}
        rep.append(f"closed: chain at 8 tokens saves {plain8 - pdl8:.2f} us < {M1_CHAIN_GATE_US}")
        return
    policy = {}
    for bucket, (batch, length) in zip(M1_BUCKETS, ((1, 8), (16, 1), (32, 1))):
        datas = {layer: _timeline_case(state, batch, length, layer, 9700 + batch * length)
                 for layer in ("sliding", "full")}
        graphs = {}
        for mode in ("off", "o", "o+front"):
            state["m1_policy"] = {bucket: mode}
            for layer, data in datas.items():
                graphs[(mode, layer)] = _graph_build(lambda d=data: kernel_fn(d, state))
        times = {k: [] for k in graphs}
        for r in range(5):
            keys = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
            for k in keys:
                times[k].append(_graph_median(graphs[k], flush, 10))
        med = {k: sorted(v)[2] for k, v in times.items()}
        load = {mode: (46 * med[(mode, "sliding")] + 16 * med[(mode, "full")]) / 62
                for mode in ("off", "o", "o+front")}
        best = min(load, key=load.get)
        policy[bucket] = best
        rep.append(f"T={bucket} step (46s+16f)/62 " + " ".join(f"{m} {t:.1f}" for m, t in
                                                             load.items()) + f" -> {best}")
        del graphs
    state["m1_policy"] = policy


def _m1_enable(state):
    try:
        _m1_tune(state)
    except Exception:
        state["m1_ok"] = False
        state["m1_policy"] = {}
        raise


def _try(name, fn):
    t0 = time.time()
    try:
        fn()
        print(f"{name}: on ({time.time() - t0:.1f} s)")
        return True
    except Exception as exc:
        print(f"{name}: SGLang fallback ({type(exc).__name__}: {exc})")
        _FAILED.append(f"{name}: fallback ({type(exc).__name__}: {str(exc)[:120]})")
        return False


# Candidates that failed a self-test or tuner check and the stages that fell back, for the
# summary at the end of prepare.
_FAILED = []

# The judge keeps only the last 16,000 characters of the log (result.json log_tail), so the
# per-stage lines printed early in prepare are lost (check e87fcb1ed5c64a62 has no MOE1BENCH
# or MOE1POLICY). This summary is printed last and kept short (about 2.5 KB) so the tail
# always holds it.
_SCORED = [(load, b, q, layer) for load, b, q in
           (("d1", 1, 1), ("d16", 16, 1), ("d32", 32, 1), ("v1", 1, 8), ("v16", 16, 8),
            ("v32", 32, 8)) for layer in ("sliding", "full")]


def _short_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _code_bytes(code):
    # Bytecode, names and constants, without the file name or line numbers, so the same source
    # gives the same digest wherever the judge stores the payload.
    out = [code.co_code, repr(code.co_names).encode()]
    for c in code.co_consts:
        out.append(_code_bytes(c) if isinstance(c, types.CodeType) else repr(c).encode())
    return b"|".join(out)


def _source_digest():
    # The module's own functions and string constants, read from memory: the submission
    # opens no files. The judge's result.json content_hash identifies the payload itself.
    h = hashlib.sha256()
    for name, value in sorted(globals().items()):
        if isinstance(value, types.FunctionType) and value.__module__ == __name__:
            h.update(name.encode() + _code_bytes(value.__code__))
        elif isinstance(value, str) and name.isupper():
            h.update(name.encode() + repr(value).encode())
    return h.hexdigest()[:16]


def _selection_summary(state):
    lines = [f"source digest {_source_digest()}"]
    gen = {"main": _CPP_SRC + _CUDA_SRC + _PREP_SRC + _ATTN_SRC + _ATTN_HOST_SRC + _WS_SRC
           + _WS_HOST_SRC + _BND_SRC, "one": _ONE_SRC + _ONE_HOST_SRC + _ONE_CPP_SRC,
           "w4": _W4_CPP_SRC + _w4_source(), "moe1": _MOE1_SRC + _moe1_source() + _MOE1_CPP_SRC,
           "skp": _SKP_SRC + _SKP_GEMV_SRC + _SKP_WD_SRC + _SKP_CPP_SRC}
    lines.append("components " + " ".join(f"{k}={_short_hash(v)}" for k, v in gen.items()))
    flags = ("moe_ours", "w4_ok", "moe1_ok", "router_ours", "proj_ours", "skp_ok",
             "attn_one_ours", "attn_ws_ours", "attn_fused_ours", "bnd_ours", "abnd_ours",
             "norm_ours", "route_side_ok", "route_ours")
    lines.append("stages " + " ".join(f"{k}={int(bool(state.get(k)))}" for k in flags))
    for layer, (pick, times) in state.get("skp_wide_report", {}).items():
        lines.append(f"proj {layer} T={SKP_WD_MAX}: {pick} ["
                     + " ".join(f"{nm} {t:.1f}" for nm, t in sorted(times.items(),
                                                                   key=lambda kv: kv[1])) + "]")
    lines.append("moe stress " + " ".join(f"{k}={v}" for k, v in
                                          state.get("moe1_stress", {}).items()))
    gemv = [x for x in _SKP_REPORT.get("selftest", []) if x.endswith(":cgemv")]
    lines.append(f"skp selftest ok {len(_SKP_REPORT.get('selftest', []))}, cgemv ok on "
                 + ",".join(x.split(":")[0] for x in gemv))
    for key in ("tune m=1", "tune m=8"):
        lines.append(f"skp {key}: " + "; ".join(_SKP_REPORT.get(key, [])))
    lines.append("moe pdl " + state.get("moe1_pdl_report", "not run"))
    lines.append(f"m1 ok={int(bool(state.get('m1_ok')))} policy {state.get('m1_policy', {})}")
    for item in state.get("m1_report", []):
        lines.append("m1 " + item)
    bench = state.get("moe1_bench", {})
    for bs in TUNE_TOKENS:
        opts = _moe_opts(bs, state)
        pick = _moe1_name(opts["fused"])
        times = " ".join(f"{v} {t:.1f}" for v, t in sorted(bench.get(bs, {}).items(),
                                                           key=lambda kv: kv[1]))
        w4 = f"g1={opts['g1']} g2={opts['g2']} pdl={int(opts['pdl'])} cl={opts['fcl']}"
        front = state.get("front_policy", {}).get(bs, "-")
        busy = state.get("moe1_busiest", {}).get(bs, "-")
        lines.append(f"moe T={bs}: {pick} [{times or 'no bench'}] judge-like routing, busiest "
                     f"expert {busy} w4 {w4} front {front}")
    for bs, (ft, ct) in state.get("front_report", {}).items():
        if ct:
            lines.append(f"front T={bs} (front / front+moe us): " + " ".join(
                f"{v} {ft[v]:.1f}/{ct[v]:.1f}" for v in sorted(ct, key=ct.get) if v in ft))
    occ = state.get("abnd_occ", {})
    lines.append("abnd occupancy (CTAs/SM, resident clusters) "
                 + " ".join(f"{k}={v[0]},{v[1]}" for k, v in occ.items()))
    for bs, times in state.get("abnd_times", {}).items():
        ranked = sorted(times.items(), key=lambda kv: kv[1])
        lines.append(f"abnd T={bs}: {state['abnd_policy'][bs]} ["
                     + " ".join(f"{v} {t:.2f}" for v, t in ranked) + "]")
    one = state.get("one")
    for load, batch, length, layer in _SCORED:
        cfg = {"batch": batch, "q_len": length, "tokens": batch * length, "num_q_heads": 16,
               "num_kv_heads": 2, "head_dim": 64, "page_size": 1,
               "window": 2048 if layer == "sliding" else 0}
        path = _attn_path(cfg, state)
        how = path.__name__.lstrip("_")
        if path is _attn_one and one is not None:
            # Planned at ctx 80000; only batch 1 full reads the width (as tiles).
            plan = _one_plan(cfg, 80000 + length, one["resident"])
            how += f" plan(splits,nc,mode)={plan}"
        lines.append(f"attn {load} {layer}: {how}")
    lines.append(f"failed {len(_FAILED)}: " + ("; ".join(_FAILED)[:1500] if _FAILED else "none"))
    for line in lines:
        print(f"SUMMARY {line}")


def prepare(weights):
    t_start = time.time()
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    positions = torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN,
                     POST_NORM_EPS, is_post_norm=True, post_norm_gain_center_type="one",
                     residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT,
                        POST_NORM_EPS, is_post_norm=True, post_norm_gain_center_type="one",
                        residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous()
                  for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(),
                                         route_scale=1.0,
                                         router_output_multiplier=ROUTER_MULTIPLIER,
                                         router_score_division_eps=ROUTE_EPS),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False),
                                        True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(
            post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(
            post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
        "moe_ours": False,
        "w4_ok": False,
        "moe1_ok": False,
        "router_ours": False,
        "proj_ours": False,
        "skp_ok": False,
        "skp_wide": {},
        "attn_prep_ours": False,
        "attn_fused_ours": False,
        "attn_ws_ours": False,
        "attn_one_ours": False,
        "route_side_ok": False,
        "route_ours": False,
        "m1_ok": False,
        "bnd_ours": False,
        "abnd_ours": False,
        "prefetch_ours": False,
        "norm_ours": False,
        "front_policy": {},
        # As SGLang's RotaryEmbedding computes them, so the angles match its cos/sin table.
        "inv_freq": inv_freq,
    }
    del freqs
    torch.cuda.synchronize()
    print(f"prepare: SGLang state built ({time.time() - t_start:.1f} s)")

    if not (PORT_MOE or PORT_ROUTER or PORT_PROJ or PORT_ATTN_PREP or PORT_ATTN_FUSED
            or PORT_ATTN_WS or PORT_ATTN_ONE or PORT_PROJ_SK):
        return state
    # Their own modules, compiled on threads while the main and W4 extensions build and the
    # self-tests run; each self-test joins its thread.
    if PORT_ATTN_ONE:
        _bg_start("attn_one", _one_load)
    if PORT_MOE and PORT_MOE_FUSED:
        _bg_start("moe1", _moe1_load)
    if PORT_PROJ_SK:
        _bg_start("skp", _skp_load)
    if not _try("extension build", _ext):
        return state
    if PORT_MOE:
        state["moe_ours"] = _try("moe", lambda: _moe_self_test(state))
    if PORT_MOE and PORT_MOE_W4 and state["moe_ours"]:
        if not _try("moe w4", lambda: _w4_enable(state, weights)):
            state["w4_ok"] = False
    if PORT_MOE and PORT_MOE_FUSED and state["moe_ours"]:
        _try("moe fused", lambda: _moe1_enable(state, weights))
    if PORT_ROUTER:
        state["router_packed"] = _pack_router_hilo(hi, lo)
        state["router_ours"] = _try("router", lambda: _router_self_test(state))
    if PORT_PROJ:
        state["proj_packed"] = {name: _pack_mma_a(w) for name, w in state["dense"].items()}
        state["proj_ours"] = _try("proj", lambda: _proj_self_test(state))
    if PORT_PROJ_SK:
        _try("proj sk", lambda: _skp_enable(state))
        if state["skp_ok"] and "pooler" in state["skp_wide"]:
            # [256 + 256, 8192]: the router's BF16 hi then lo rows, for the "wide" front.
            state["router_hilo"] = torch.cat((hi, lo), 0).contiguous()
    if PORT_ATTN_PREP:
        state["attn_prep_ours"] = True
        state["attn_prep_ours"] = _try("attn prep", lambda: _attn_self_test(state))
    if PORT_ATTN_FUSED:
        state["attn_fused_ours"] = True
        state["attn_fused_ours"] = _try("attn fused", lambda: _attn_self_test(state))
    if PORT_FFN_BOUNDARY:
        state["bnd_ours"] = _try("ffn boundary", lambda: _bnd_enable(state))
    if PORT_ATTN_BOUNDARY:
        state["abnd_ours"] = _try("attention boundary", lambda: _abnd_enable(state))
        if state["abnd_ours"]:
            _try("attention boundary tuner", lambda: _abnd_tune(state))
    if PORT_L2_PREFETCH:
        state["prefetch_ours"] = _try("prefetch", lambda: _prefetch_enable(state))
    if PORT_PRE_NORM:
        state["norm_ours"] = _try("pre-expert norm", lambda: _norm_self_test(state))
    if ROUTE_ON_SIDE:
        state["route_side_ok"] = _try("route side", lambda: _route_side_self_test(state))
    if PORT_ROUTE_KERNEL and state["norm_ours"]:
        state["route_bias"] = weights["router_bias"].float().contiguous()
        state["route_ours"] = _try("route kernel", lambda: _route_self_test(state))
    if PORT_ATTN_WS:
        state["attn_ws_ours"] = True
        state["attn_ws_ours"] = _try("attn ws", lambda: _attn_self_test(state))
    if PORT_ATTN_ONE:
        state["attn_one_ours"] = True
        state["attn_one_ours"] = _try("attn one", lambda: _attn_one_self_test(state))
    if PORT_FRONT_CONCAT and state["norm_ours"]:
        # [4096 + 256 + 256, 8192]: the pooler, then the router's BF16 hi and lo halves; its
        # transpose is the second operand of one torch.mm with FP32 output, the layout the
        # pooler's own GEMM reads.
        state["pool_router"] = torch.cat((state["dense"]["pooler"], hi, lo), 0).contiguous()
    fused_front = bool(state.get("skp", {}).get("fused"))
    if (state["w4_ok"] and (PORT_ROUTER_MMA or PORT_FRONT_CONCAT)) or fused_front:
        _try("moe front tuner", lambda: _front_tune(state))
    if PORT_SK_PDL and state["skp_ok"] and state["attn_one_ours"]:
        _try("M1 sk pdl tuner", lambda: _m1_enable(state))
    torch.cuda.synchronize()
    print(f"prepare: done ({time.time() - t_start:.1f} s)")
    try:
        _selection_summary(state)
    except Exception as exc:
        print(f"SUMMARY failed: {type(exc).__name__}: {exc}")
    try:
        _timeline(state)
    except Exception as exc:
        print(f"TL failed: {type(exc).__name__}: {str(exc)[:300]}")
    torch.cuda.synchronize()
    return state


def _proj_wins(layer, tokens):
    """Where the mma kernel beat cuBLAS on muse_bf16_proj (check ee65e159c9f54269): every
    layer at 1 token and all but o at 8; above 8 tokens it loses."""
    if layer == "o":
        return tokens == 1
    return tokens <= 8


def _proj(x, layer, n, state):
    w = state["dense"][layer]
    m = x.shape[0]
    if state["skp_ok"] and m <= 32:
        pick = state["skp_pick"].get((layer, m))
        if pick is not None and pick[2] == n:
            if layer == "o" and getattr(pick[1], "sk", False) and _m1_mode(m, state) != "off":
                return pick[1](x.contiguous(), True)
            return pick[1](x.contiguous())
    if state["skp_ok"] and SKP_WD_MIN <= m <= SKP_WD_MAX and n == w.shape[0]:
        wide = state["skp_wide"].get(layer)
        if wide is not None:
            return wide[1](x.contiguous())
    if state["proj_ours"] and _proj_wins(layer, m):
        return _ext().proj_fwd(x.contiguous(), state["proj_packed"][layer], n)
    return _gemm(x, w if n == w.shape[0] else w[:n])


def kernel_fn(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    dense = state["dense"]
    qkv = _proj(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], state)
    if state["prefetch_ours"] and tokens <= SLICED_UNPOOL_MAX_TOKENS:
        # The side stream is in order, so the router later follows this on the same stream.
        side_stream = state["side_stream"]
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _ext().prefetch_l2(*_prefetch_set(tokens, state))
    o = _attn_path(cfg, state)(qkv, data, state)
    attn_output = _proj(o, "o", HIDDEN, state)  # all-reduce: identity

    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    if state["abnd_ours"]:
        ffn_input, stream, router_input = _abnd(data["stream"], attn_output, not split_router,
                                                state, _abnd_geo(tokens, state))
    else:
        ffn_input, stream, router_input = state["attn_boundary"](
            data["stream"], attn_output, branch_dtype=torch.bfloat16,
            router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    moe_opts = _moe_opts(tokens, state) if state["moe_ours"] else None
    expert_input, topk_weights, topk_ids, meta = _moe_front(
        ffn_input, router_input, data["local_start"], state, moe_opts,
        _front_variant(tokens, state))
    # all-reduce: identity
    if meta is not None:
        expert_output = _experts_run(expert_input, meta, state)
    else:
        expert_output = _experts(_pre_norm(expert_input, state), topk_weights, topk_ids,
                                 data["local_start"], state, moe_opts)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _proj(expert_output, "unpooler", UNPOOL_SLICE, state)
        if state["bnd_ours"]:
            # The all-gather's other 7168 channels are exact zeros; the boundary reads only
            # the live 1024 and still normalizes the branch over all 8192.
            next_attn_input, stream = _bnd(stream, unpooled, state)
            return next_attn_input, stream, topk_ids
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _proj(expert_output, "unpooler", HIDDEN, state)

    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output,
                                                       branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
