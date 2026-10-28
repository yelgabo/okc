# okc: problem=muse_decode_layer
"""SGLang's decoder layer (baseline.py's call sequence) with pieces measured faster on the
sibling problems swapped in. Every swapped piece falls back to SGLang's call when its
self-test in prepare() fails or where it was measured to lose.

- MoE (from muse_moe_w4a16): one CUDA kernel builds Marlin's block-sorted routing metadata
  over the 32 local experts, reading this GPU's first expert from the device; one CUDA kernel
  does the post-expert norm and weighted sum, which SGLang runs as about ten PyTorch ops
  below 32 tokens; Marlin moe_block_size 8 below 12 tokens, else 16.
- Router (from muse_bf16_proj): up to 8 tokens one CUDA-core FP32 kernel on the boundary's
  FP32 router input; 9 to 32 tokens the tensor-core kernel on a BF16 hi + lo split of the
  weights with FP32 accumulation (SGLang's own method above 8 tokens).
- Projections (from muse_bf16_proj): an mma kernel streaming weights repacked into fragment
  order, for qkv_gate, pooler and the unpooler slice up to 8 tokens and o at 1 token.

- Attention prep (idea 1 from muse_attention_decode): one launch for QK-norm, RoPE, the FP8
  query and the FP8 cache write in place of FusedQKNorm, three casts and store_cache, and one
  launch for the norm-gate; FA3 stays SGLang's.
- Fused decode attention (idea 2 from muse_attention_decode), q_len 1 except full layers at
  batch 1-2: one kernel per request and context slice covering both KV heads (norm, RoPE,
  cache write, Q.K and P.V on f16 tensor cores, online softmax); sliding layers merge splits in
  a thread-block cluster with the norm-gate, full layers in one more launch.

The fused norms around attention and the MoE are SGLang's, called as baseline.py does.
"""

from typing import TypedDict

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
PORT_ROUTER = True
PORT_PROJ = True
# The hi/lo mma router at 9-32 tokens runs on the side stream under the pooler: switching it
# on and off moved decode 16 by -0.01 to -0.02 both times, within noise (checks 6ef2d969b6104e10,
# 46dab8fb87274880), so SGLang's split router stays there.
PORT_ROUTER_MMA = False
# Idea 1 from muse_attention_decode: one prep launch and one norm-gate launch around FA3.
PORT_ATTN_PREP = True
# Idea 2: the fused q_len 1 attention kernel, on the shapes where it won on its own problem.
PORT_ATTN_FUSED = True

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
                                       int64_t block_size, int64_t max_padded);
torch::Tensor finalize(torch::Tensor y, torch::Tensor scores, torch::Tensor norm_w,
                       torch::Tensor local_ids, double eps);
torch::Tensor router_fwd(torch::Tensor x, torch::Tensor w);
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

// Marlin's moe_align layout over this GPU's 32 experts only: blocks of block_size rows,
// experts ascending, pad slots hold numel. SGLang aligns over 257 buckets instead and emits
// blocks of other GPUs' routes that Marlin then scans past. The first local expert is read
// from the device so the call needs no host sync. Atomic ranks order rows within an expert
// arbitrarily; Marlin computes each row independently, so the output does not depend on it.
__global__ void __launch_bounds__(1024) route_align_kernel(
        const int* __restrict__ topk_ids, const int* __restrict__ start_p, int numel, int bs,
        int* __restrict__ sorted_ids, int* __restrict__ expert_ids, int* __restrict__ num_post,
        int* __restrict__ local_ids) {
    __shared__ int cnt[kMaxLocal];
    __shared__ int off[kMaxLocal];
    __shared__ int total_s;
    __shared__ short rank_s[kMaxRoutes];
    const int tid = threadIdx.x;
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
    router_kernel<M><<<grid, 32 * R_ROWS_PER_BLOCK, router_smem_bytes<M>(), stream>>>(
        wp, xp, op, m);
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

    auto load_chunk = [&](int c, int slot) {
        uint4* wdst = ws + slot * W_STAGE_VECS + warp * R * CH * 32 + lane;
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int i = 0; i < CH; ++i) {
                cp_async16(wdst + (r * CH + i) * 32,
                           wsrc + ((size_t)r * kt_total + c * CH + i) * 32, true);
            }
        }
        __nv_bfloat16* xdst = xs + slot * TOKENS * XS;
        const int kcol = (HILO ? (kt0 + c * CH) / 2 : kt0 + c * CH) * 16;
        constexpr int VECS_PER_ROW = HILO ? CH : CH * 2;
        for (int v = threadIdx.x; v < TOKENS * VECS_PER_ROW; v += WARPS * 32) {
            const int tok = v / VECS_PER_ROW;
            const int col = (v % VECS_PER_ROW) * 8;
            const bool valid = tok < m;
            cp_async16(xdst + tok * XS + col, valid ? x + (size_t)tok * xk + kcol + col : x,
                       valid);
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < chunks) {
            load_chunk(s, s);
        }
        asm volatile("cp.async.commit_group;\n");
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
    proj_kernel<MT, WARPS, R, CH, S, HILO, OutT><<<grid, WARPS * 32, smem, stream>>>(
        wp, xp, op, m, n, k, k / 16 / S);
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

// compile_check: route_align_kernel<<<1, 64>>>(0, 0, 0, 8, 0, 0, 0, 0);
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
                                       int64_t block_size, int64_t max_padded) {
    const int numel = topk_ids.numel();
    TORCH_CHECK(numel <= kMaxRoutes);
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 && start.scalar_type() == torch::kInt32);
    auto opts = topk_ids.options();
    auto sorted_ids = torch::empty({max_padded}, opts);
    auto expert_ids = torch::empty({(max_padded + block_size - 1) / block_size}, opts);
    auto num_post = torch::empty({1}, opts);
    auto local_ids = torch::empty_like(topk_ids);
    int threads = ((numel + 31) / 32) * 32;
    threads = threads < 64 ? 64 : (threads > 1024 ? 1024 : threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    route_align_kernel<<<1, threads, 0, stream>>>(
        topk_ids.data_ptr<int>(), start.data_ptr<int>(), numel, static_cast<int>(block_size),
        sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(), num_post.data_ptr<int>(),
        local_ids.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {sorted_ids, expert_ids, num_post, local_ids};
}

torch::Tensor finalize(torch::Tensor y, torch::Tensor scores, torch::Tensor norm_w,
                       torch::Tensor local_ids, double eps) {
    const int m = y.size(0);
    TORCH_CHECK(y.size(1) == kTopK && y.size(2) == kExpertHidden);
    auto out = torch::empty({m, kExpertHidden}, y.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    finalize_row_kernel<<<m, 512, 0, stream>>>(
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
    const int t = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    constexpr int heads = HQ + 2 * HKV;
    if (h >= heads) {
        return;
    }
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
    const int t = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (h >= HQ) {
        return;
    }
    const size_t row = (size_t)t * HQ + h;
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
    prep_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    gate_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    // The first row this warp merges in a cluster (CTA r owns rows r, r + S, ...); its gate load
    // overlaps the whole kernel.
    const int row0 = warp * p.splits + split;
    float2 gate0 = make_float2(0.f, 0.f);
    if (CLUSTER && row0 < C::ROWS) {
        gate0 = load_gate(p.qkv, static_cast<size_t>(b) * L + row0 / HQ, row0 % HQ, lane);
    }
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
    cudaLaunchAttribute attr[1];
    cfg.numAttrs = 0;
    if (CLUSTER) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = p.splits;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
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
    combine_kernel<<<batch, HQ * 32, 0, stream>>>(p.ws, p.qkv, p.v_scale, p.out, 1, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="okc_decode_layer_v1",
            cpp_sources=_CPP_SRC,
            cuda_sources=[_CUDA_SRC, _PREP_SRC, _ATTN_SRC, _ATTN_HOST_SRC],
            functions=["route_align", "finalize", "router_fwd", "proj_fwd", "router_mma_fwd",
                       "attn_prep", "attn_gate", "attn_fused"],
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


def _experts(x, topk_weights, topk_ids, local_start, state, ours):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize.
    topk_ids are global expert ids; this GPU's experts start at local_start (a device tensor).
    ours: our routing metadata, finalizer and block size; else SGLang's calls."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    numel = num_tokens * TOP_K
    topk_weights = topk_weights.to(torch.float32).contiguous()
    ours = ours and numel <= MAX_ROUTES
    if ours:
        # SGLang's heuristic picks 48 at 128 tokens and 64 at 256; 16 was 25-35% faster
        # there on muse_moe_w4a16 (docs/runs-moe.md).
        block_size_m = 8 if num_tokens < 12 else 16
        # The buffer size SGLang allocates, so Marlin sizes its FP32 reduce buffer alike.
        if numel < N_EXPERTS + 1:
            max_padded = numel * block_size_m
        else:
            max_padded = numel + (N_EXPERTS + 1) * (block_size_m - 1)
        max_padded = (max_padded + 3) // 4 * 4
        sorted_ids, block_experts, n_padded, local_ids = _ext().route_align(
            topk_ids.to(torch.int32).contiguous(), local_start, block_size_m, max_padded)
    else:
        block_size_m = _sglang_bsm(num_tokens)
        local_ids = (topk_ids - local_start) % N_EXPERTS
        local_ids = bank["expert_map"][local_ids].contiguous()
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((numel, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((numel * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: numel * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: numel * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
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
        size_m=numel, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if ours:
        return _ext().finalize(y, topk_weights, state["post_expert_w"],
                               local_ids.view(num_tokens, TOP_K), POST_NORM_EPS)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


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


def _attn_path(cfg, state):
    heads = (cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"])
    if heads != (16, 2, 64):
        return _attn_sglang
    if state["attn_fused_ours"] and _fused_wins(cfg):
        return _attn_fused
    if state["attn_prep_ours"]:
        return _attn_prep
    return _attn_sglang


def _attn_case(device, batch, length, layer, page, kv_scale, seed):
    """A synthetic attention step in the judge's input layout: random caches and a shuffled
    page table, ctx 1500-5000 per request."""
    gen = torch.Generator(device=device).manual_seed(seed)
    seq = torch.randint(1500, 5000, (batch,), generator=gen, device=device, dtype=torch.int32)
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


def _attn_self_test(state):
    """Each attention path against SGLang's on synthetic steps: the output and the new cache
    rows. Raises on a mismatch."""
    device = state["inv_freq"].device
    cases = [(1, 1, "sliding", 1, 1.0), (3, 1, "full", 1, 1.0), (5, 1, "sliding", 1, 1.0),
             (2, 8, "sliding", 1, 1.0), (3, 8, "full", 1, 1.0), (4, 1, "full", 64, 0.5),
             (2, 8, "sliding", 64, 1.0), (33, 1, "full", 1, 1.0), (2, 1, "full", 1, 1.0),
             (32, 1, "sliding", 1, 1.0), (16, 1, "full", 1, 1.0)]
    worst_out, worst_cache = 0.0, 0.0
    for i, (batch, length, layer, page, kv_scale) in enumerate(cases):
        qkv, data = _attn_case(device, batch, length, layer, page, kv_scale, 7000 + i)
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
        print(f"attn self-test case {i} {batch}x{length} {layer} page {page} "
              f"{path.__name__}: output rel diff {diff:.2e}")
        worst_out = max(worst_out, diff)
        rows = data["cache_loc"]
        for name in ("k_cache", "v_cache"):
            a = ours[name].view(-1, 128)[rows].float()
            b = data[name].view(-1, 128)[rows].float()
            worst_cache = max(worst_cache, _rel(a, b))
            ours[name].view(-1, 128)[rows] = data[name].view(-1, 128)[rows]
            if not torch.equal(ours[name].view(torch.uint8), data[name].view(torch.uint8)):
                raise RuntimeError(f"attn case {i}: {name} written outside the step's rows")
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


def _moe_self_test(state):
    """Our MoE path against SGLang's at the judge's token counts, with skewed routing and
    several local_start values. Raises on a mismatch."""
    device = state["bank"]["w13"].device
    worst = 0.0
    for tokens in (1, 3, 8, 9, 16, 24, 32, 33, 40, 128, 256, 264):
        for start in (0, 96, 224):
            gen = torch.Generator(device=device).manual_seed(1000 + tokens + start)
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
            scores = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
            hot = torch.rand((N_EXPERTS,), generator=gen, device=device).pow(6) * 4
            routed = (scores + hot).topk(TOP_K, dim=-1)
            ids = routed.indices.to(torch.int32)
            weights = (routed.values / routed.values.sum(-1, keepdim=True)).float()
            local_start = torch.tensor([start], dtype=torch.int32, device=device)
            ref = _experts(x, weights, ids, local_start, state, False)
            out = _experts(x, weights, ids, local_start, state, True)
            if not torch.isfinite(out).all():
                raise RuntimeError(f"moe tokens={tokens} start={start}: non-finite")
            worst = max(worst, _rel(out, ref))
    torch.cuda.synchronize()
    print(f"moe self-test: worst rel diff vs SGLang path {worst:.2e}")
    if not worst < 5e-3:
        raise RuntimeError(f"moe self-test: rel diff {worst:.2e}")


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
        "router_ours": False,
        "proj_ours": False,
        "attn_prep_ours": False,
        "attn_fused_ours": False,
        # As SGLang's RotaryEmbedding computes them, so the angles match its cos/sin table.
        "inv_freq": inv_freq,
    }
    del freqs
    torch.cuda.synchronize()
    print(f"prepare: SGLang state built ({time.time() - t_start:.1f} s)")

    if not (PORT_MOE or PORT_ROUTER or PORT_PROJ or PORT_ATTN_PREP or PORT_ATTN_FUSED):
        return state
    if not _try("extension build", _ext):
        return state
    if PORT_MOE:
        state["moe_ours"] = _try("moe", lambda: _moe_self_test(state))
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
    o = _attn_path(cfg, state)(qkv, data, state)
    attn_output = _proj(o, "o", HIDDEN, state)  # all-reduce: identity

    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16,
        router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _proj(ffn_input, "pooler", D_EXPERT, state)
    with torch.cuda.stream(side_stream):
        if not split_router and state["router_ours"]:
            logits = _ext().router_fwd(router_input.contiguous(), state["router"])
        elif not split_router:
            logits = F.linear(router_input, state["router"])
        elif state["router_ours"] and PORT_ROUTER_MMA and tokens <= ROUTER_MMA_MAX_TOKENS:
            logits = _ext().router_mma_fwd(ffn_input.contiguous(), state["router_packed"])
        else:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, topk_ids,
                             data["local_start"], state, state["moe_ours"])
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _proj(expert_output, "unpooler", UNPOOL_SLICE, state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output,
                                                       branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
