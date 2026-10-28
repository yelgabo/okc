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
- Router (from muse_bf16_proj): up to 8 tokens one CUDA-core FP32 kernel on the boundary's
  FP32 router input; 9 to 32 tokens the tensor-core kernel on a BF16 hi + lo split of the
  weights with FP32 accumulation (SGLang's own method above 8 tokens).
- Stream overlap: the router, top-8 and the MoE routing metadata run on a side stream while
  the main stream runs the pooler and the pre-expert norm; the streams join right before
  GEMM1.
- Projections (from muse_bf16_proj): an mma kernel streaming weights repacked into fragment
  order, for qkv_gate, pooler and the unpooler slice up to 8 tokens and o at 1 token.

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
PORT_ROUTER = True
PORT_PROJ = True
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
                                         double eps_next, bool want_router);
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
    return __byte_perm(static_cast<uint32_t>(lo), static_cast<uint32_t>(hi), 0x5410);
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
template <int L, int NT>
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
        // Lane t8 == 0 alone reads and zeroes the row sum and broadcasts it, so no lane reads
        // what another lane zeroes; the loop is uniform so the shuffle is reached by every lane.
        for (int copy = 0; copy < p.nc; ++copy) {
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
    if constexpr (FLAT) {
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
        v0 = static_cast<int>(total_v * split / G);
        v1 = static_cast<int>(total_v * (split + 1) / G);
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
                return __ldg(pt + (psize == 1 ? pos : pos / psize));
            };
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
                        mbar_wait(empty0 + 8 * pst, pph ^ 1u);
                        const int pos0 = lo + t * BN;
                        const uint32_t kdst = smem_base + S::KV + pst * 2 * TILE_BYTES;
                        const uint32_t vdst = kdst + TILE_BYTES;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int r = r0 + 16 * i;
                            const int pos = pos0 + r;
                            const uint32_t kd = kdst + r * ROWB + k_swz<L>(ch, r) * 16;
                            const uint32_t vd = vdst + r * ROWB + v_swz<L>(ch, r) * 16;
                            if (pos < seq) {
                                int64_t row = pe[u][i];
                                if (psize != 1) {
                                    row = row * psize + pos % psize;
                                }
                                const int64_t off = row * ROWB + ch * 16;
                                cp_async16(kd, p.k_cache + off, 16);
                                cp_async16(vd, p.v_cache + off, 16);
                            } else if (pos >= hi) {
                                cp_async16(kd, p.k_cache, 0);
                                cp_async16(vd, p.v_cache, 0);
                            }
                        }
                        cp_async_mbar_arrive_noinc(full0 + 8 * pst);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            pe[u][i] = entry(t + PF, r0 + 16 * i);
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
            return;
        }
        if constexpr (!SOLO) {
            setmaxnreg_inc<one_cons_regs(L)>();
        }
    }

    // ------------------------------------------------------------------ consumers
    auto issue_idx = [&](int tile) {
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
#pragma unroll
        for (int i = 0; i < BN / RSTEP; ++i) {
            const int r = r0 + i * RSTEP;
            const int pos = pos0 + r;
            const uint32_t kd = kdst + r * ROWB + k_swz<L>(ch, r) * 16;
            const uint32_t vd = vdst + r * ROWB + v_swz<L>(ch, r) * 16;
            if (pos < seq) {
                int64_t row = idx[r];
                if (psize != 1) {
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
        // Everything above reads inputs. The query rows are the qkv projection's output
        // and the constants group reads the epoch the previous launch wrote, so under PDL
        // both follow the wait (which returns at once on a plain launch).
        asm volatile("griddepcontrol.wait;" ::: "memory");
        // Query rows, the block's new key/value rows (if this range may hold them),
        // constants; the range holding the request's first key prefetches the gate rows.
        {
            const __nv_bfloat16* qbase = p.qkv + static_cast<size_t>(b) * L * QKV_STRIDE;
            for (int i = tid; i < ROWS * 8; i += NC) {
                const int row = i >> 3;
                const int cc = i & 7;
                cp_async16(smem_base + S::QST + row * 128 + cc * 16,
                           qbase + (row >> 4) * QKV_STRIDE + (row & 15) * D + cc * 8, 16);
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

        if constexpr (!PROD) {
            cp_wait<1>();
            csync();
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
            cp_wait<STAGES - 1>();
        } else {
            cp_wait<0>();
        }
        csync();

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

        // Query pass: eight threads per row, eight dims each. q_len 1 keeps f16 queries
        // pre-scaled into log2 units (the key scale is applied to the scores); q_len 8 rounds
        // the unit-RMS query to E4M3 and scales in the exponent.
        for (int piece = tid; piece < ROWS * 8; piece += NC) {
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
        }

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
            if (is_k) {
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
        csync();
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

        const float sl2k = (L == 1 ? 1.f : p.sl2) * k_scale;
        const int copy = p.nc > 1 ? split % p.nc : 0;

        // Tile it of this segment becomes readable in stage `st` (every consumer thread calls
        // this; the generic writes above and the gathered rows are fenced for the wgmma
        // reads at q_len 8).
        auto acquire = [&](int it, int st) {
            if constexpr (PROD) {
                mbar_wait(full0 + 8 * st, cph);
                if constexpr (L == 8) {
                    fence_proxy_async();
                }
                const int pos0 = lo + it * BN;
                if (hi > seq && pos0 + BN > seq) {
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
                __syncthreads();
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
            const uint64_t qd0 = desc_kmaj(smem_base + S::Q + hk * 4096, 128, 512);
            const uint64_t qd1 = desc_kmaj(smem_base + S::Q + hk * 4096 + 256, 128, 512);
            float o[36];
#pragma unroll
            for (int i = 0; i < 36; ++i) {
                o[i] = 0.f;
            }
            float off[2] = {OFF_FLOOR, OFF_FLOOR};
            const int qp0 = seq + 2 * wq;
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
        return true;
    };

    // Last-CTA finalize of request b: every red above is ordered before thread 0's release
    // by the barrier, and the last arriver's acquire makes them visible to its reads (which
    // bypass L1).
    auto ticket = [&]() {
        csync();
        const int mine = hi - lo;
        if (tid == 0) {
            // >= rather than ==: an overshoot would be a partition bug, and == would then
            // leave the ticket non-zero for every later launch using this request index.
            const int old = atom_add_acq_rel(p.ws_cnt + b, mine);
            *flag = old + mine >= n_keys;
        }
        csync();
        if (*flag) {
            finalize_rows<L, NC>(p, b * ROWS, (b + 1) * ROWS, v_scale, tid);
            if (tid == 0) {
                p.ws_cnt[b] = 0;
            }
        }
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
                finalize_rows<L, NC>(p, b * ROWS + r_lo, b * ROWS + r_hi, v_scale, tid);
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
            csync();
            coop_barrier<NC>(p.ws_cnt + CNT_FLAT, p.ws_cnt + CNT_FLAT + 2, ep, mine_total,
                             keys_all, tid, p.dbg);
            const int rows_all = B * ROWS;
            const int rows_per = (rows_all + G - 1) / G;
            const int r_lo = split * rows_per;
            const int r_hi = min(rows_all, r_lo + rows_per);
            if (r_lo < r_hi) {
                finalize_rows<L, NC>(p, r_lo, r_hi, v_scale_g, tid);
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

    static void launch(const Params& p, int gx, int gy, cudaStream_t stream, bool coop) {
        prepare();
        if (!coop) {
            attn_one_kernel<L, MODE><<<dim3(gx, gy), NT, BYTES, stream>>>(p);
            return;
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
        TORCH_CHECK(err == cudaSuccess, "attn_one cooperative launch: ", cudaGetErrorString(err));
    }
};

template <int L, int MODE>
int Instance<L, MODE>::resident = -1;

template <int L, int MODE>
int Instance<L, MODE>::per_sm = 0;

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
// warpgroup); q_len 8: 15, 3, 11, 1 and 51 likewise. The |64 and non-producer flat variants
// (115, 17, 19) stay out: the plan falls back to the per-request instances without them.
#define DL_ONE_INSTANCES(X) \
    X(1, 1) X(1, 3) X(1, 11) X(1, 13) X(1, 49) X(8, 1) X(8, 3) X(8, 11) X(8, 15) X(8, 51)

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
// cooperative modes and pdl the programmatic one to every launch; dbg (int32, empty or [4])
// receives the longest cooperative spin. Output [batch * q_len, 1024] BF16.
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

}  // namespace bnd
}  // namespace

std::vector<torch::Tensor> attn_boundary(torch::Tensor stream, torch::Tensor branch,
                                         torch::Tensor alpha, torch::Tensor beta,
                                         torch::Tensor w_next, double eps_post,
                                         double eps_next, bool want_router) {
    using namespace bnd;
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
    launch_ex(
        attn_boundary_kernel, dim3(CTAS, tokens), dim3(THREADS), 0,
        at::cuda::getCurrentCUDAStream(), kPdlGlue, stream.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(branch.data_ptr()), alpha.data_ptr<float>(),
        beta.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(w_next.data_ptr()),
        stream_out.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(next.data_ptr()),
        want_router ? router.data_ptr<float>() : static_cast<float*>(nullptr),
        static_cast<float>(eps_post), static_cast<float>(eps_next));
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
                       "ffn_boundary_short", "pre_norm", "attn_boundary", "prefetch_l2"],
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
    prepare()."""
    g1, g2, pdl, fcl = _w4_policy(num_tokens, state)
    return {"bsm": 8 if num_tokens < 12 else 16, "g1": g1, "g2": g2, "pdl": pdl, "fcl": fcl}


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
    routing metadata from _route_meta()."""
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

_ONE_EXT = None
_ONE_BUILD = {}


def _one_build():
    """Builds the one-launch extension: its own module, so the build runs on a thread while
    the main and W4 extensions compile and their self-tests run."""
    global _ONE_EXT
    try:
        _ONE_EXT = load_inline(
            name="okc_decode_layer_one",
            cpp_sources=_ONE_CPP_SRC,
            cuda_sources=[_ONE_SRC, _ONE_HOST_SRC],
            functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        )
    except Exception as exc:
        _ONE_BUILD["error"] = exc
    _ONE_BUILD["done"] = time.time()


def _one_build_start():
    if "thread" not in _ONE_BUILD:
        _ONE_BUILD["started"] = time.time()
        thread = threading.Thread(target=_one_build, name="okc-attn-one-build", daemon=True)
        thread.start()
        _ONE_BUILD["thread"] = thread


def _one_ext():
    if _ONE_EXT is None:
        if "thread" in _ONE_BUILD:
            _ONE_BUILD["thread"].join()
        else:
            _one_build()
        if _ONE_EXT is None:
            raise _ONE_BUILD["error"]
    return _ONE_EXT


# The one-launch kernel's persistent workspace (allocated once in prepare, zeroed once; every
# completed launch leaves it zero again): FP32 rows of [64] plus one row sum each, and the
# counters. Row ((b * nc + copy) * L + j) * 16 + head; batch 1 full takes 8 copies x 8 tokens x
# 16 heads = 1,024 rows, batch 512 at q_len 8 all of them. Counters: [b] tickets,
# [4096 + 2 b + parity] cooperative, [8192 + b] epochs, [16384 ..] the flat kernel's.
_ONE_WS_ROWS = 65536
_ONE_CNT_INTS = 32768
# Grid target: two CTAs per SM on 132 SMs.
_ONE_TARGET = 264
# The instances _ONE_HOST_SRC builds (DL_ONE_INSTANCES).
_ONE_BUILT = {1: (1, 3, 11, 13, 49), 8: (1, 3, 11, 15, 51)}
# Full-context batches on the flat partition (else per request). Only the producer-warpgroup
# flat instances (49, 51) are built here, so _ONE_PROD = False also sends those batches to
# the per-request kernels.
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
        "dbg": torch.zeros(4, dtype=torch.int32, device=device),
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


def _one_stress(state, batch, length, layer, replays, seed):
    """One shape captured as a graph and replayed `replays` times back to back, then the
    replayed output against an eager call (the workspace atomics make the sums
    order-dependent at the 1e-7 level, so bitwise equality is not expected) and against
    SGLang's path, a NaN check and the workspace invariant. On the batch-1 page-1 shape the
    sequence length is then shrunk by one tile in place, so the speculated length is wrong
    and the device must take the real one, for 50 more replays with the same checks."""
    device = state["inv_freq"].device
    st = state["one"]
    qkv, data = _attn_case(device, batch, length, layer, 1, 1.0, seed)
    ours = dict(data, k_cache=data["k_cache"].clone(), v_cache=data["v_cache"].clone())
    plan = _one_plan(data["config"], data["page_table"].shape[1], st["resident"])
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
        msg = (f"{name}: replays={n} replayed_vs_eager={rel_eager:.1e} vs_sglang={rel_ref:.2e} "
               f"max_polls={polls} {ws_msg}")
        if not (finite and rel_eager < 1e-4 and rel_ref < 5e-2 and ws_ok):
            raise RuntimeError(f"attn one stress {batch}x{length} {layer} {msg} finite={finite}")
        return msg

    msgs = [check(f"plan={plan}", replays, ref)]
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
    sliding (the speculated one-tile launch), decode 16 full (flat tickets) and their verify
    counterparts (cooperative counters). Prints each step's time; raises on any failure."""
    t0 = time.time()
    st = _one_setup(state)
    lim = st["limits"]
    res = " ".join(f"L{lim[i]}m{lim[i + 1]}:{lim[i + 2]}({lim[i + 4]}/SM,{lim[i + 3] // 1024}KB)"
                   for i in range(0, len(lim), 5))
    build = _ONE_BUILD.get("done", t0) - _ONE_BUILD.get("started", t0)
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
    for i, (batch, length, layer) in enumerate(((1, 1, "sliding"), (16, 1, "full"),
                                                (1, 8, "sliding"), (16, 8, "full"))):
        t2 = time.time()
        msg = _one_stress(state, batch, length, layer, 2000, 8000 + i)
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
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
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


def _abnd_enable(state):
    """Takes what SGLang's attention boundary holds (the residual gate's FP32 alpha and beta,
    the pre-FFN norm's effective 1 + w in BF16, both eps) and checks our kernel against the
    module at 1-256 tokens, with and without the FP32 router copy. Raises on a mismatch."""
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
    gen = torch.Generator(device="cuda").manual_seed(9)
    worst = 0.0
    for tokens in (1, 2, 3, 5, 8, 9, 16, 33, 256):
        stream = torch.randn((tokens, HIDDEN), generator=gen, device="cuda") * 3
        branch = (torch.randn((tokens, HIDDEN), generator=gen, device="cuda") * 0.5)
        branch = branch.to(torch.bfloat16)
        want_router = tokens <= ROUTER_FP32_MAX_TOKENS
        ref_next, ref_stream, ref_router = mod(
            stream.clone(), branch, branch_dtype=torch.bfloat16,
            router_dtype=torch.float32 if want_router else None)
        nxt, out, router = _abnd(stream.clone(), branch, want_router, state)
        torch.cuda.synchronize()
        carried = abnd["alpha"] * stream
        errs = {"next input": _rel(nxt, ref_next),
                "update": _rel(out - carried, ref_stream - carried),
                "stream": _rel(out, ref_stream)}
        if want_router:
            errs["router"] = _rel(router, ref_router)
        print(f"attention boundary self-test T={tokens}: "
              + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()))
        worst = max(worst, max(errs.values()))
    if not worst < 1e-4:
        raise RuntimeError(f"attention boundary: rel diff {worst:.2e}")


def _abnd(stream, branch, want_router, state):
    abnd = state["abnd"]
    nxt, out, router = _ext().attn_boundary(stream.contiguous(), branch.contiguous(),
                                            abnd["alpha"], abnd["beta"], abnd["w_next"],
                                            abnd["eps_post"], abnd["eps_next"], want_router)
    return nxt, out, (router if want_router else None)


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
FRONT_TOKENS = (16, 32, 128, 256)


def _front_variants(tokens, state):
    """The side-chain variants the tuner may pick at this token count: "split" is SGLang's hi/lo
    torch.mm pair (two GEMMs, two split-K reduces and an add) under the pooler, "mma" our hi/lo
    mma router (up to 32 tokens), "concat" one cuBLAS GEMM for the pooler and both router halves
    with FP32 output (needs our norm, which rounds the pooled columns on load)."""
    out = ["split"]
    if PORT_ROUTER_MMA and state["router_ours"] and tokens <= ROUTER_MMA_MAX_TOKENS:
        out.append("mma")
    if PORT_FRONT_CONCAT and state["norm_ours"] and "pool_router" in state:
        out.append("concat")
    return out


def _front_variant(tokens, state):
    if tokens <= ROUTER_FP32_MAX_TOKENS:
        return "fp32"
    key = next((t for t in FRONT_TOKENS if t >= tokens), FRONT_TOKENS[-1])
    return state["front_policy"].get(key, "split")


def _moe_front(ffn_input, router_input, local_start, state, moe_opts, variant):
    """From the FFN input to GEMM1's inputs: the pooler and the pre-expert norm on the main
    stream; the router, routing and (with ROUTE_ON_SIDE) the routing metadata on the side
    stream, joined before GEMM1. Returns (expert input, topk_weights, topk_ids, meta); the
    expert input is normalized when meta is not None, else the caller normalizes it."""
    tokens = ffn_input.shape[0]
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    route_side = ROUTE_ON_SIDE and state["route_side_ok"]
    if variant == "concat":
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
        if variant == "concat":
            logits = out32[:, D_EXPERT:D_EXPERT + N_EXPERTS] + out32[:, D_EXPERT + N_EXPERTS:]
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


def _front_tune(state):
    """Times the MoE front per token count above 8, as a CUDA graph replayed after an L2 flush
    as the judge times it, for every side-chain variant, and keeps the fastest. The variants
    must agree: the normalized expert input within 1e-3 (relative L2) of the split variant's,
    which is the level of BF16 rounding flips from a different FP32 summation order, and at
    least 99% of the routes equal."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(21)
    local_start = torch.tensor([96], dtype=torch.int32, device="cuda")
    policy = {}
    for tokens in FRONT_TOKENS:
        ffn_input = torch.randn((tokens, HIDDEN), generator=gen, device="cuda")
        ffn_input = ffn_input.to(torch.bfloat16)
        opts = _moe_opts(tokens, state) if state["moe_ours"] else None
        times, outs = {}, {}
        for variant in _front_variants(tokens, state):
            res = {}

            def run():
                ei, _, ids, meta = _moe_front(ffn_input, None, local_start, state, opts,
                                              variant)
                res["x"] = ei if meta is not None else _pre_norm(ei, state)
                res["ids"] = ids

            times[variant] = _graph_time(run, flush)
            torch.cuda.synchronize()
            outs[variant] = (res["x"].clone(), res["ids"].clone())
        ref_x, ref_ids = outs["split"]
        agree = {}
        for variant, (x, ids) in outs.items():
            err = _rel(x, ref_x)
            same = float((ids == ref_ids).float().mean())
            agree[variant] = f"{err:.1e}/{same:.3f}"
            if not (err < 1e-3 and same >= 0.99):
                raise RuntimeError(f"front T={tokens} {variant}: rel diff {err:.2e}, "
                                   f"routes {same:.3f}")
        best = min(times, key=times.get)
        policy[tokens] = best
        print(f"TUNE front T={tokens} "
              + " ".join(f"{v}:{t:.1f}" for v, t in sorted(times.items(), key=lambda kv: kv[1]))
              + f" -> {best} (vs split: "
              + " ".join(f"{v} {a}" for v, a in agree.items()) + ")")
    state["front_policy"] = policy


def _try(name, fn):
    t0 = time.time()
    try:
        fn()
        print(f"{name}: on ({time.time() - t0:.1f} s)")
        return True
    except Exception as exc:
        print(f"{name}: SGLang fallback ({type(exc).__name__}: {exc})")
        return False


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
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
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
        "router_ours": False,
        "proj_ours": False,
        "attn_prep_ours": False,
        "attn_fused_ours": False,
        "attn_ws_ours": False,
        "attn_one_ours": False,
        "route_side_ok": False,
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
            or PORT_ATTN_WS or PORT_ATTN_ONE):
        return state
    if PORT_ATTN_ONE:
        # Its own module, compiled on a thread while the main and W4 extensions build and
        # the self-tests run; its self-test joins the thread.
        _one_build_start()
    if not _try("extension build", _ext):
        return state
    if PORT_MOE:
        state["moe_ours"] = _try("moe", lambda: _moe_self_test(state))
    if PORT_MOE and PORT_MOE_W4 and state["moe_ours"]:
        if not _try("moe w4", lambda: _w4_enable(state, weights)):
            state["w4_ok"] = False
    if PORT_ROUTER:
        state["router_packed"] = _pack_router_hilo(hi, lo)
        state["router_ours"] = _try("router", lambda: _router_self_test(state))
    if PORT_PROJ:
        state["proj_packed"] = {name: _pack_mma_a(w) for name, w in state["dense"].items()}
        state["proj_ours"] = _try("proj", lambda: _proj_self_test(state))
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
    if PORT_L2_PREFETCH:
        state["prefetch_ours"] = _try("prefetch", lambda: _prefetch_enable(state))
    if PORT_PRE_NORM:
        state["norm_ours"] = _try("pre-expert norm", lambda: _norm_self_test(state))
    if ROUTE_ON_SIDE:
        state["route_side_ok"] = _try("route side", lambda: _route_side_self_test(state))
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
    if state["w4_ok"] and (PORT_ROUTER_MMA or PORT_FRONT_CONCAT):
        _try("moe front tuner", lambda: _front_tune(state))
    torch.cuda.synchronize()
    print(f"prepare: done ({time.time() - t_start:.1f} s)")
    return state


def _proj_wins(layer, tokens):
    """Where the mma kernel beat cuBLAS on muse_bf16_proj (check ee65e159c9f54269): every
    layer at 1 token and all but o at 8; above 8 tokens it loses."""
    if layer == "o":
        return tokens == 1
    return tokens <= 8


def _proj(x, layer, n, state):
    w = state["dense"][layer]
    if state["proj_ours"] and _proj_wins(layer, x.shape[0]):
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
                                                state)
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
        ffn_output = _gemm(expert_output, dense["unpooler"])

    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output,
                                                       branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
