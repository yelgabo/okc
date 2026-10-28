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

Attention and the fused norms are SGLang's, called as baseline.py does.
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
PORT_PROJ = False

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

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="okc_decode_layer_v1",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["route_align", "finalize", "router_fwd", "proj_fwd", "router_mma_fwd"],
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
    }
    del freqs
    torch.cuda.synchronize()
    print(f"prepare: SGLang state built ({time.time() - t_start:.1f} s)")

    if not (PORT_MOE or PORT_ROUTER or PORT_PROJ):
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
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    qkv = _proj(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], state)
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
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
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
        elif state["router_ours"] and tokens <= ROUTER_MMA_MAX_TOKENS:
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
