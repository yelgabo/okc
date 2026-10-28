# okc: problem=muse_moe_w4a16
"""SGLang's Marlin NVFP4 expert path with its overheads cut and a better Marlin tile.

- Routing metadata: one CUDA kernel maps topk_ids to local expert ids and builds Marlin's
  block-sorted layout over the 32 local experts only. SGLang gathers through an expert map
  and then aligns over 257 buckets (one per global expert plus the filtered -1 bucket),
  which skips its single-token fast path and emits blocks of filtered routes that Marlin
  then scans past.
- Finalizer: SGLang's fused post-expert norm + weighted sum falls back to about ten
  PyTorch ops below 32 tokens. One CUDA kernel does it in a single pass.
- Marlin moe_block_size: 8 below 12 tokens, else 16. SGLang's heuristic picks 48 or 64 at
  128 and 256 tokens, which pads the many lightly loaded experts' blocks.

The Marlin GEMMs and silu_and_mul are SGLang's, called as baseline.py does.
"""

from typing import TypedDict

import time
import types

import torch
from sgl_kernel.scalar_type import scalar_types
from torch.utils.cpp_extension import load_inline

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8
# route_align keeps per-route ranks in shared memory and scans experts with one warp.
MAX_ROUTES = 4096
MAX_LOCAL = 32

_CPP_SRC = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> route_align(torch::Tensor topk_ids, int64_t start, int64_t count,
                                       int64_t block_size, int64_t max_padded,
                                       int64_t item_rows, int64_t item_rows2, int64_t big_rows,
                                       int64_t big_rows2, int64_t big_at, int64_t big_at2);
torch::Tensor finalize(torch::Tensor y, torch::Tensor scores, torch::Tensor norm_w,
                       torch::Tensor local_ids, int64_t variant, double eps);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kMaxLocal = 32;
constexpr int kMaxRoutes = 4096;
constexpr int kTopK = 8;
constexpr int kHidden = 4096;

// Work-item lists for the W4 GEMMs: 0 and 1 are GEMM1's and GEMM2's, 2 and 3 their large-tile
// lists. An expert with lo <= routes < hi gets items of up to rows routes in that list.
struct ItemLists {
    int rows[4];
    int lo[4];
    int hi[4];
    int off[4];
};

// Marlin's moe_align layout over local experts only: blocks of block_size rows, experts
// ascending, pad slots hold numel. Atomic ranks order rows within an expert arbitrarily;
// Marlin computes each row independently, so the output does not depend on that order.
__global__ void __launch_bounds__(1024) route_align_kernel(
        const int* __restrict__ topk_ids, int numel, int start, int count, int bs,
        int* __restrict__ sorted_ids, int* __restrict__ expert_ids, int* __restrict__ num_post,
        int* __restrict__ local_ids, ItemLists lists, int2* __restrict__ items,
        int* __restrict__ num_items) {
    __shared__ int cnt[kMaxLocal];
    __shared__ int off[kMaxLocal];
    __shared__ int total_s;
    __shared__ short rank_s[kMaxRoutes];
    const int tid = threadIdx.x;
    if (tid < kMaxLocal) {
        cnt[tid] = 0;
    }
    __syncthreads();
    for (int p = tid; p < numel; p += blockDim.x) {
        const int e = topk_ids[p] - start;
        const int loc = (e >= 0 && e < count) ? e : -1;
        local_ids[p] = loc;
        if (loc >= 0) {
            rank_s[p] = static_cast<short>(atomicAdd(&cnt[loc], 1));
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int c = tid < count ? cnt[tid] : 0;
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
        // Work items for the W4 GEMMs: up to ir routes of one expert each, as
        // (expert << 16 | rows, first row in sorted_ids).
        for (int q = 0; q < 4; ++q) {
            const int ir = lists.rows[q];
            if (ir <= 0) {
                continue;
            }
            int2* list = items + lists.off[q];
            const int nit = c >= lists.lo[q] && c < lists.hi[q] ? (c + ir - 1) / ir : 0;
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
        if (e >= 0 && e < count) {
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

// Loads this thread's 8 channels of every live slot (dead slots may hold garbage, even NaN,
// so they are never read) and returns the per-slot partial sums of squares.
__device__ __forceinline__ void load_slots(
        const __nv_bfloat16* __restrict__ y, const int* __restrict__ local_ids,
        const float* w, int t, int c, bool* live, float (*v)[8], float* ss) {
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        live[k] = local_ids[t * kTopK + k] >= 0;
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (live[k]) {
            const size_t row = static_cast<size_t>(t) * kTopK + k;
            raw = *reinterpret_cast<const uint4*>(y + row * kHidden + c);
        }
        unpack8(raw, v[k]);
        ss[k] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[k][j] *= w[j];
            ss[k] += v[k][j] * v[k][j];
        }
    }
}

// One block per token: 512 threads x 8 channels.
template <int THREADS>
__global__ void __launch_bounds__(THREADS) finalize_row_kernel(
        const __nv_bfloat16* __restrict__ y, const float* __restrict__ scores,
        const __nv_bfloat16* __restrict__ norm_w, const int* __restrict__ local_ids,
        __nv_bfloat16* __restrict__ out, float eps) {
    static_assert(THREADS * 8 == kHidden);
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
    load_slots(y, local_ids, w, t, c, live, v, ss);
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
                scale[k] = live[k] ? scores[t * kTopK + k] * rsqrtf(s / kHidden + eps) : 0.f;
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
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kHidden + c) = pack8(acc);
}

// S blocks per token in a cluster, each over kHidden / S channels; the per-slot sums of
// squares are exchanged through distributed shared memory.
template <int S>
__global__ void __launch_bounds__(kHidden / 8 / S) finalize_cluster_kernel(
        const __nv_bfloat16* __restrict__ y, const float* __restrict__ scores,
        const __nv_bfloat16* __restrict__ norm_w, const int* __restrict__ local_ids,
        __nv_bfloat16* __restrict__ out, float eps) {
    constexpr int THREADS = kHidden / 8 / S;
    constexpr int WARPS = THREADS / 32;
    __shared__ float red[kTopK][WARPS];
    __shared__ float part[kTopK];
    __shared__ float scale[kTopK];
    cg::cluster_group cluster = cg::this_cluster();
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c = (blockIdx.x * THREADS + tid) * 8;
    float w[8];
    unpack8(*reinterpret_cast<const uint4*>(norm_w + c), w);
    bool live[kTopK];
    float v[kTopK][8];
    float ss[kTopK];
    load_slots(y, local_ids, w, t, c, live, v, ss);
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
        part[tid] = s;
    }
    cluster.sync();
    if (tid < kTopK) {
        float s = 0.f;
#pragma unroll
        for (int r = 0; r < S; ++r) {
            s += *cluster.map_shared_rank(&part[tid], r);
        }
        scale[tid] = live[tid] ? scores[t * kTopK + tid] * rsqrtf(s / kHidden + eps) : 0.f;
    }
    // Also keeps every block's part[] alive until the other blocks have read it.
    cluster.sync();
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
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kHidden + c) = pack8(acc);
}

template <int S>
void launch_cluster(const __nv_bfloat16* y, const float* scores, const __nv_bfloat16* w,
                    const int* ids, __nv_bfloat16* out, int m, float eps, cudaStream_t stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(S, m, 1);
    cfg.blockDim = dim3(kHidden / 8 / S, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = S;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(
        cudaLaunchKernelEx(&cfg, finalize_cluster_kernel<S>, y, scores, w, ids, out, eps));
}

// compile_check: route_align_kernel<<<1, 64>>>(nullptr, 0, 0, 0, 8, 0, 0, 0, 0, {}, 0, 0);
// compile_check: finalize_row_kernel<512><<<1, 512>>>(0, 0, 0, 0, 0, 0.f);
// compile_check: launch_cluster<2>(0, 0, 0, 0, 0, 1, 0.f, 0);
// compile_check: launch_cluster<4>(0, 0, 0, 0, 0, 1, 0.f, 0);
// compile_check: launch_cluster<8>(0, 0, 0, 0, 0, 1, 0.f, 0);
}  // namespace

std::vector<torch::Tensor> route_align(torch::Tensor topk_ids, int64_t start, int64_t count,
                                       int64_t block_size, int64_t max_padded,
                                       int64_t item_rows, int64_t item_rows2, int64_t big_rows,
                                       int64_t big_rows2, int64_t big_at, int64_t big_at2) {
    const int numel = topk_ids.numel();
    TORCH_CHECK(numel <= kMaxRoutes && count <= kMaxLocal);
    auto opts = topk_ids.options();
    auto sorted_ids = torch::empty({max_padded}, opts);
    auto expert_ids = torch::empty({(max_padded + block_size - 1) / block_size}, opts);
    auto num_post = torch::empty({1}, opts);
    auto local_ids = torch::empty_like(topk_ids);
    const int64_t rows[4] = {item_rows, item_rows2, big_rows, big_rows2};
    const int64_t big_from[2] = {big_at, big_at2};
    ItemLists lists;
    int64_t len[4];
    int64_t total_items = 0;
    for (int q = 0; q < 4; ++q) {
        const bool big = q >= 2;
        const bool split = rows[q % 2 + 2] > 0;
        lists.rows[q] = static_cast<int>(rows[q]);
        lists.lo[q] = big ? static_cast<int>(big_from[q % 2]) : 0;
        lists.hi[q] = !big && split ? static_cast<int>(big_from[q % 2]) : 1 << 30;
        lists.off[q] = static_cast<int>(total_items);
        len[q] = rows[q] > 0 ? count + numel / rows[q] + 1 : 1;
        total_items += len[q];
    }
    auto items = torch::empty({total_items, 2}, opts);
    auto num_items = torch::empty({4}, opts);
    int threads = ((numel + 31) / 32) * 32;
    threads = threads < 64 ? 64 : (threads > 1024 ? 1024 : threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    route_align_kernel<<<1, threads, 0, stream>>>(
        topk_ids.data_ptr<int>(), numel, static_cast<int>(start), static_cast<int>(count),
        static_cast<int>(block_size), sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
        num_post.data_ptr<int>(), local_ids.data_ptr<int>(), lists,
        reinterpret_cast<int2*>(items.data_ptr<int>()), num_items.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::vector<torch::Tensor> res = {sorted_ids, expert_ids, num_post, local_ids};
    for (int q = 0; q < 4; ++q) {
        res.push_back(items.narrow(0, lists.off[q], len[q]));
        res.push_back(num_items.narrow(0, q, 1));
    }
    return res;
}

torch::Tensor finalize(torch::Tensor y, torch::Tensor scores, torch::Tensor norm_w,
                       torch::Tensor local_ids, int64_t variant, double eps) {
    const int m = y.size(0);
    TORCH_CHECK(y.size(1) == kTopK && y.size(2) == kHidden);
    auto out = torch::empty({m, kHidden}, y.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* yp = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>());
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    const float* sp = scores.data_ptr<float>();
    const int* ip = local_ids.data_ptr<int>();
    const float e = static_cast<float>(eps);
    switch (variant) {
        case 0:
            finalize_row_kernel<512><<<m, 512, 0, stream>>>(yp, sp, wp, ip, op, e);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            break;
        case 2: launch_cluster<2>(yp, sp, wp, ip, op, m, e, stream); break;
        case 4: launch_cluster<4>(yp, sp, wp, ip, op, m, e, stream); break;
        case 8: launch_cluster<8>(yp, sp, wp, ip, op, m, e, stream); break;
        default: TORCH_CHECK(false, "bad finalize variant");
    }
    return out;
}
"""

# Choices measured with okc check (see docs/runs-moe.md). align: "ours" or "sgl"; fin: a
# finalize() variant (0 = one block per token, 2/4/8 = cluster split) or "sgl"; bsm: Marlin
# moe_block_size, or None for SGLang's heuristic; g1: a W4_CONFIGS index for GEMM1, or None for
# Marlin's.
def _opts_for(num_tokens, state=None):
    # SGLang's heuristic picks 48 at 128 tokens and 64 at 256; 16 was 25-35% faster there.
    bsm = 8 if num_tokens < 12 else 16
    g1, g2, pdl, fcl = _w4_policy(num_tokens, state)
    return {"align": "ours", "fin": 0, "bsm": bsm, "g1": g1, "g2": g2, "pdl": pdl, "fcl": fcl}


# Batch sizes the W4 configs are tuned at in prepare(); other sizes use the next one up.
TUNE_BS = (1, 8, 16, 32, 128, 256)


def _w4_policy(num_tokens, state):
    """(GEMM1 config, GEMM2 config, PDL, finalizer cluster size) tuned in prepare() for this
    batch size; a None config is Marlin's."""
    policy = (state or {}).get("w4_policy")
    if not policy:
        return None, None, False, 1
    key = next((b for b in TUNE_BS if b >= num_tokens), TUNE_BS[-1])
    return policy.get(key, (None, None, False, 1))


def _w4_fits(cfg, num_tokens):
    """A work item holds at most 8 * nt routes of one expert, and an expert gets at most
    num_tokens routes, so shared memory must hold min(8 * nt, num_tokens) rows."""
    _, nt, rows, _, _, _, _, _, _ = W4_CONFIGS[cfg]
    return rows >= min(8 * nt, num_tokens)


BASELINE_OPTS = {"align": "sgl", "fin": "sgl", "bsm": None}

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="okc_moe_ext",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["route_align", "finalize"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        )
    return _EXT


# W4A16 grouped GEMM for the routed experts (see _W4_SRC). A config is
# (kind, nt, rows, s, w, d, mode, layout, p): kind 1 is x @ W13^T with SwiGLU, kind 2 is
# inter @ W2^T; nt n8 token tiles per work item; rows of x kept in shared memory
# (min(8 nt, bs)); s K splits; w warps per block; d register stages of weights in flight per
# warp; mode, layout and p as in w4_gemm_kernel (mode 0 is the GEMM, the others time only).
# Mode 7 is the GEMM with weights brought in by bulk copies (w4_tma_kernel), d shared-memory
# stages per block. A large-tile config runs as a second launch over the work items of the
# heavily routed experts (opts g1b/g2b, threshold t1/t2 routes).
# Checks 42924d2c6e314921, 05924173c44046e9 and 0f478ae5371a4b75 pruned the losers (no K split
# without persistence, 2 warps per block, 2 stages in flight, 32 rows, most of layout 1).
W4_CONFIGS = [
    (1, 1, 1, 4, 4, 4, 0, 0, 0), (1, 1, 8, 2, 4, 4, 0, 0, 0), (1, 1, 8, 4, 4, 4, 0, 0, 0),
    (1, 1, 8, 2, 4, 8, 0, 0, 0), (1, 1, 8, 2, 8, 4, 0, 0, 0), (1, 2, 16, 4, 4, 4, 0, 0, 0),
    (1, 2, 16, 2, 4, 4, 0, 0, 0), (1, 2, 16, 4, 8, 4, 0, 0, 0), (1, 2, 16, 4, 8, 4, 0, 1, 0),
    (2, 1, 1, 4, 4, 8, 0, 0, 0), (2, 1, 1, 4, 4, 4, 0, 0, 0), (2, 1, 8, 4, 4, 4, 0, 0, 0),
    (2, 1, 8, 2, 4, 4, 0, 0, 0), (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 2, 16, 2, 4, 4, 0, 0, 0),
    (1, 1, 1, 1, 4, 4, 0, 0, 1), (1, 1, 1, 2, 4, 4, 0, 0, 1), (1, 1, 8, 1, 8, 4, 0, 0, 1),
    (1, 1, 8, 2, 4, 4, 0, 0, 1), (1, 1, 8, 4, 4, 4, 0, 0, 1), (1, 1, 8, 2, 8, 4, 0, 0, 1),
    (1, 1, 8, 4, 8, 4, 0, 0, 1), (1, 2, 16, 2, 4, 4, 0, 0, 1), (1, 2, 16, 4, 4, 4, 0, 0, 1),
    (1, 2, 16, 2, 8, 4, 0, 0, 1),
    (2, 1, 1, 2, 4, 4, 0, 0, 1), (2, 1, 1, 4, 4, 4, 0, 0, 1), (2, 1, 8, 1, 4, 4, 0, 0, 1),
    (2, 1, 8, 2, 4, 4, 0, 0, 1), (2, 1, 8, 4, 4, 4, 0, 0, 1), (2, 1, 8, 2, 8, 4, 0, 0, 1),
    (2, 2, 16, 2, 4, 4, 0, 0, 1), (2, 2, 16, 4, 4, 4, 0, 0, 1),
    # Without the L2 prefetch hint.
    (1, 1, 8, 4, 4, 4, 5, 0, 0), (1, 1, 8, 2, 8, 4, 5, 0, 1), (2, 1, 8, 4, 4, 4, 5, 0, 0),
    (2, 2, 16, 4, 4, 4, 5, 0, 0), (1, 2, 16, 4, 4, 4, 5, 0, 0),
    # Timing only (see MODE in w4_gemm_kernel).
    (1, 1, 8, 4, 4, 4, 1, 0, 0), (1, 1, 8, 4, 4, 4, 3, 0, 0), (1, 1, 8, 4, 4, 4, 4, 0, 0),
    (1, 1, 8, 4, 4, 4, 6, 0, 0), (1, 1, 8, 4, 8, 4, 4, 0, 0),
    # Benchmarked only (W4_TUNED). Mode 7, layout 1: weights by bulk copy (TMA).
    (1, 1, 8, 4, 4, 4, 7, 1, 0), (1, 1, 8, 4, 4, 8, 7, 1, 0), (1, 1, 8, 4, 8, 4, 7, 1, 0),
    (1, 1, 8, 2, 8, 4, 7, 1, 0), (1, 1, 8, 2, 4, 8, 7, 1, 0), (1, 2, 16, 4, 4, 8, 7, 1, 0),
    (1, 2, 16, 4, 8, 4, 7, 1, 0), (1, 1, 1, 4, 4, 8, 7, 1, 0),
    (2, 1, 8, 4, 4, 8, 7, 1, 0), (2, 1, 8, 4, 8, 4, 7, 1, 0), (2, 1, 8, 2, 4, 8, 7, 1, 0),
    (2, 2, 16, 4, 4, 8, 7, 1, 0), (2, 2, 16, 4, 8, 4, 7, 1, 0), (2, 1, 1, 4, 4, 8, 7, 1, 0),
    # Large token tiles (32 or 64 routes per item) for heavily routed experts.
    (1, 4, 32, 4, 4, 4, 0, 0, 0), (1, 8, 64, 4, 8, 4, 0, 0, 0), (1, 4, 32, 2, 8, 4, 0, 0, 0),
    (1, 8, 64, 4, 4, 2, 0, 0, 0),
    (2, 4, 32, 4, 4, 4, 0, 0, 0), (2, 8, 64, 4, 8, 4, 0, 0, 0), (2, 8, 64, 4, 4, 2, 0, 0, 0),
]
# The tuner in prepare() picks among the configs before this one.
W4_TUNED = W4_CONFIGS.index((1, 1, 8, 4, 4, 4, 7, 1, 0))

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

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_addr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_addr(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile("{\n\t.reg .pred p;\n"
                 "WAIT:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra WAIT;\n}"
                 :: "r"(smem_addr(bar)), "r"(parity) : "memory");
}

__device__ __forceinline__ void bulk_load(void* dst, const void* src, unsigned bytes,
                                          uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 :: "r"(smem_addr(dst)), "l"(src), "r"(bytes), "r"(smem_addr(bar)) : "memory");
}

// w4_gemm_kernel (MODE 0, LAYOUT 1, P 0) with the weights brought into shared memory by bulk
// copies instead of each lane's own loads, testing whether the register-held stages limit
// resident warps. Layout 1 makes one block's stage of W units one contiguous W KB of values and
// W * 128 B of scales, so one thread of an extra producer warp issues two copies per stage
// into a ring of D stages; full barriers count the bytes, empty barriers the consumer lanes.
template <int KIND, int NT, int ROWS, int S, int W, int D>
__global__ void __launch_bounds__((W + 1) * 32) w4_tma_kernel(
        const __nv_bfloat16* __restrict__ act, const uint4* __restrict__ wq,
        const uint4* __restrict__ wsc, const float* __restrict__ gs,
        const int* __restrict__ sorted_ids, const int2* __restrict__ items,
        const int* __restrict__ num_items, int numel, void* __restrict__ out) {
    constexpr int KC = kK / S;
    constexpr int NST = KC / 64;
    constexpr int U = KIND == 1 ? 256 : 128;
    constexpr int RS = KC + 8;
    constexpr int TR = 8 * NT;
    constexpr int VB = W * 1024;
    constexpr int SB = W * 128;
    static_assert(8 % W == 0 && D <= NST, "a block's units must share one 8-unit group");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rid[TR];
    __shared__ __align__(8) uint64_t full[D];
    __shared__ __align__(8) uint64_t empty[D];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (KIND == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
    if (static_cast<int>(blockIdx.y) >= num_items[0]) {
        return;
    }
    const int2 it = items[blockIdx.y];
    const int e = it.x >> 16;
    const int nrows = it.x & 0xffff;
    const int row0 = it.y;
    const int chunk = blockIdx.x;
    const int z = blockIdx.z;
    unsigned char* wst = smem_raw + ROWS * RS * 2;
    unsigned char* sst = wst + D * VB;

    if (threadIdx.x == 0) {
        for (int i = 0; i < D; ++i) {
            mbar_init(&full[i], 1);
            mbar_init(&empty[i], W * 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (warp == W) {
        if (lane == 0) {
            const size_t g0 = (static_cast<size_t>(e * (U / 8) + chunk * W / 8) * 64 + z * NST)
                              * 8 + (chunk * W) % 8;
#pragma unroll 1
            for (int s = 0; s < NST; ++s) {
                const int slot = s % D;
                if (s >= D) {
                    mbar_wait(&empty[slot], (s / D - 1) & 1);
                }
                mbar_expect_tx(&full[slot], VB + SB);
                const size_t st = g0 + static_cast<size_t>(s) * 8;
                bulk_load(wst + slot * VB, wq + st * 64, VB, &full[slot]);
                bulk_load(sst + slot * SB, wsc + st * 8, SB, &full[slot]);
            }
        }
        return;
    }

    if (KIND == 2) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    for (int r = threadIdx.x; r < TR; r += W * 32) {
        rid[r] = r < nrows ? sorted_ids[row0 + r] : -1;
    }
    {
        constexpr int CH = KC / 8;
        for (int i = threadIdx.x; i < ROWS * CH; i += W * 32) {
            const int r = i / CH;
            const int c = i - r * CH;
            if (r < nrows) {
                const int p = sorted_ids[row0 + r];
                const int src_row = KIND == 1 ? p / 8 : p;
                cp_async16(xs + r * RS + c * 8,
                           act + static_cast<size_t>(src_row) * kK + z * KC + c * 8);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_all;" ::: "memory");
    }
    // Consumer warps only: the producer warp has moved on.
    asm volatile("bar.sync 1, %0;" :: "n"(W * 32) : "memory");

    const int u = chunk * W + warp;
    const int g = lane >> 2;
    const int t4 = lane & 3;
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
    for (int s = 0; s < NST; ++s) {
        const int slot = s % D;
        mbar_wait(&full[slot], (s / D) & 1);
        const uint4* vp = reinterpret_cast<const uint4*>(wst + slot * VB) + warp * 64 + lane;
        uint4 wv[2];
        wv[0] = vp[0];
        wv[1] = vp[32];
        const uint4 sv = reinterpret_cast<const uint4*>(sst + slot * SB)[warp * 8 + (lane >> 2)];
        mbar_arrive(&empty[slot]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned b[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                ldmatrix_x4(b[nt], xrow[nt] + s * 64 + h * 32);
            }
#pragma unroll
            for (int j1 = 0; j1 < 2; ++j1) {
                const unsigned sw = word(sv, h * 2 + j1);
                unsigned a0[4];
                unsigned a1[4];
                dequant<0>(word(wv[h], j1 * 2), sw, a0);
                dequant<1>(word(wv[h], j1 * 2 + 1), sw, a1);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    mma_bf16(acc[0][nt], a0, b[nt][j1 * 2], b[nt][j1 * 2 + 1]);
                    mma_bf16(acc[1][nt], a1, b[nt][j1 * 2], b[nt][j1 * 2 + 1]);
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
            const int p = rid[col];
            if (p < 0) {
                continue;
            }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const float v0 = acc[0][nt][hh * 2 + qq] * gscale;
                const float v1 = acc[1][nt][hh * 2 + qq] * gscale;
                if (KIND == 1 && S == 1) {
                    // As in w4_gemm_kernel: BF16 gate and up before SwiGLU, BF16 output.
                    const float gb = bf16r(v0);
                    const float ub = bf16r(v1);
                    const float sv0 = gb / (1.f + __expf(-gb));
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
    if constexpr (MODE == 7) {
        static_assert(LAYOUT == 1 && P == 0, "the bulk-copy path reads layout 1, one unit a warp");
        constexpr int smem = ROWS * (kK / S + 8) * 2 + D * W * 1152;
        auto kern = w4_tma_kernel<KIND, NT, ROWS, S, W, D>;
        static bool ready = false;
        if (!ready) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            ready = true;
        }
        launch_pdl(kern, dim3(U / W, max_items, S), (W + 1) * 32, smem, pdl, stream, act, wq,
                   wsc, gs, sorted_ids, items, num_items, numel, out);
    } else {
        constexpr int smem = (P ? 2 : 1) * ROWS * (kK / S + 8) * 2;
        auto kern = w4_gemm_kernel<KIND, NT, ROWS, S, W, D, MODE, LAYOUT, P>;
        static int grid = 0;
        if (grid == 0) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            int blocks = 0;
            C10_CUDA_CHECK(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, W * 32, smem));
            int sms = 0;
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
            grid = blocks * sms;
        }
        const dim3 g = P ? dim3(grid) : dim3(U / W, max_items, S);
        launch_pdl(kern, g, W * 32, smem, pdl, stream, act, wq, wsc, gs, sorted_ids, items,
                   num_items, numel, out);
    }
}

// Registers per thread, dynamic shared memory per block, and resident blocks per SM.
template <int KIND, int NT, int ROWS, int S, int W, int D, int MODE, int LAYOUT, int P>
std::vector<int64_t> w4_query() {
    constexpr bool tma = MODE == 7;
    constexpr int smem = tma ? ROWS * (kK / S + 8) * 2 + D * W * 1152
                             : (P ? 2 : 1) * ROWS * (kK / S + 8) * 2;
    constexpr int threads = (tma ? W + 1 : W) * 32;
    auto kern = [] {
        if constexpr (tma) {
            return w4_tma_kernel<KIND, NT, ROWS, S, W, D>;
        } else {
            return w4_gemm_kernel<KIND, NT, ROWS, S, W, D, MODE, LAYOUT, P>;
        }
    }();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, threads, smem));
    return {attr.numRegs, smem + static_cast<int64_t>(attr.sharedSizeBytes), blocks, threads,
            static_cast<int64_t>(attr.localSizeBytes)};
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
        checks.append(f"// w4_compile_check: (void)w4_query<{targs}>();")
    return (_W4_SRC.replace("//@W4_CASES@", "\n".join(cases))
            .replace("//@W4_INFO@", "\n".join(infos))
            .replace("//@W4_COMPILE_CHECKS@", "\n".join(checks)))


_W4_EXT = None


def _w4_ext():
    global _W4_EXT
    if _W4_EXT is None:
        _W4_EXT = load_inline(
            name="okc_moe_w4",
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


def _w4_prepare(weights, start, count):
    device = weights["w13"].device
    out = {}
    for name, kind in (("w13", 1), ("w2", 2)):
        packed = weights[name][start:start + count]
        sf = weights[name + "_sf"][start:start + count].view(torch.uint8)
        if not _w4_scales_ok(sf):
            return None
        words = []
        scales = []
        for e in range(count):
            w = w4_repack_values(packed[e], kind)
            w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)
            words.append(w)
            scales.append(w4_repack_scales(sf[e], kind).contiguous())
        out[name] = torch.cat(words).contiguous()
        out[name + "_s"] = torch.cat(scales).contiguous()
        out[name + "g"] = w4_group_units(out[name], 64 * 256)
        out[name + "g_s"] = w4_group_units(out[name + "_s"], 64 * 128)
        out[name + "_gs"] = weights[name + "_gs"][start:start + count].float().contiguous()
    torch.cuda.synchronize(device)
    return out


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _prepare_base(weights):
    device = weights["w13"].device
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data,
            "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data,
            "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count,
            "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "ours_ok": False,
    }


def _heuristic_bsm(num_tokens, count):
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / count / block_size_m < 0.9:
            break
    return block_size_m


def _forward(data, state, opts):
    cfg = data["config"]
    start, count = cfg["local_start"], cfg["local_count"]
    bank = state["banks"][(start, count)]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]
    topk_ids = data["topk_ids"].to(torch.int32).contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    block_size_m = opts["bsm"] or _heuristic_bsm(num_tokens, bank["count"])
    numel = num_tokens * TOP_K

    ours = (opts["align"] == "ours" and expert_map is not None and numel <= MAX_ROUTES
            and count <= MAX_LOCAL)
    if ours:
        # Same buffer size SGLang allocates, so Marlin sizes its FP32 reduce buffer alike.
        if numel < N_EXPERTS + 1:
            max_padded = numel * block_size_m
        else:
            max_padded = numel + (N_EXPERTS + 1) * (block_size_m - 1)
        max_padded = (max_padded + 3) // 4 * 4
        g1, g2 = opts.get("g1"), opts.get("g2")
        if g1 is not None and not (state.get("w4_ok") and _w4_fits(g1, num_tokens)):
            g1 = None
        if g2 is not None and not (state.get("w4_ok") and _w4_fits(g2, num_tokens)):
            g2 = None
        g1b, g2b = (opts.get("g1b") if g1 is not None else None,
                    opts.get("g2b") if g2 is not None else None)
        rows1 = 8 * W4_CONFIGS[g1][1] if g1 is not None else 0
        rows2 = 8 * W4_CONFIGS[g2][1] if g2 is not None else 0
        big1 = 8 * W4_CONFIGS[g1b][1] if g1b is not None else 0
        big2 = 8 * W4_CONFIGS[g2b][1] if g2b is not None else 0
        (sorted_ids, block_experts, n_padded, local_ids, items, num_items, items2, num_items2,
         items_b1, num_b1, items_b2, num_b2) = _ext().route_align(
            topk_ids, start, count, block_size_m, max_padded, rows1, rows2, big1, big2,
            opts.get("t1", 1), opts.get("t2", 1))
        big_1 = (g1b, items_b1, num_b1) if g1b is not None else None
        big_2 = (g2b, items_b2, num_b2) if g2b is not None else None
    else:
        g1 = g2 = None
        local_ids = topk_ids
        if expert_map is not None:
            local_ids = expert_map[local_ids]
        local_ids = local_ids.contiguous()
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((numel, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((numel * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: numel * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: numel * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False,
                  is_ep=expert_map is not None, b_q_type=scalar_types.float4_e2m1f,
                  is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    pdl = bool(opts.get("pdl"))
    if g1 is not None:
        _w4_gemm1(state["w4"], g1, x, sorted_ids, items, num_items, local_ids, numel, cache2,
                  pdl, big_1)
    else:
        cache1 = moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
        silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1),
                     expert_step=1)
    if g2 is not None:
        part = _w4_gemm2(state["w4"], g2, cache2, sorted_ids, items2, num_items2, numel, pdl,
                         big_2)
        return _w4_ext().w4_finalize(part, topk_weights, state["norm_w"], local_ids.view(-1),
                                     POST_NORM_EPS, pdl, opts.get("fcl", 1))
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=numel, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if opts["fin"] != "sgl" and size_k == 4096:
        return _ext().finalize(y, topk_weights, state["norm_w"], local_ids.view(num_tokens, TOP_K),
                               opts["fin"], POST_NORM_EPS)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _w4_launches(cfg, items, num_items, big):
    """The GEMM launches for one GEMM: big = (config, items, num_items) of the large-tile list,
    run first, or None. Both write the same K-split partials, so their splits must match."""
    if big is None:
        return [(cfg, items, num_items)]
    if W4_CONFIGS[big[0]][3] != W4_CONFIGS[cfg][3]:
        raise ValueError("large-tile config must use the same K splits")
    return [big, (cfg, items, num_items)]


def _w4_gemm1(w4, cfg, x, sorted_ids, items, num_items, local_ids, numel, inter, pdl=False,
              big=None):
    """inter[route] = SwiGLU(x[token] @ W13[expert]^T) for every local route, in bf16."""
    splits = W4_CONFIGS[cfg][3]
    ext = _w4_ext()
    out = inter
    if splits > 1:
        out = torch.empty((splits, numel, 2 * 4096), device=x.device, dtype=torch.float32)
    for c, its, n in _w4_launches(cfg, items, num_items, big):
        name = "w13g" if W4_CONFIGS[c][7] else "w13"
        ext.w4_gemm(c, x, w4[name], w4[name + "_s"], w4["w13_gs"], sorted_ids, its, n, numel,
                    out, pdl)
    if splits > 1:
        ext.w4_swiglu(out, local_ids.view(-1), splits, inter, pdl)


def _w4_gemm2(w4, cfg, inter, sorted_ids, items, num_items, numel, pdl=False, big=None):
    """FP32 K-split partials [splits, routes, 4096] of inter[route] @ W2[expert]^T."""
    splits = W4_CONFIGS[cfg][3]
    part = torch.empty((splits, numel, 4096), device=inter.device, dtype=torch.float32)
    for c, its, n in _w4_launches(cfg, items, num_items, big):
        name = "w2g" if W4_CONFIGS[c][7] else "w2"
        _w4_ext().w4_gemm(c, inter, w4[name], w4[name + "_s"], w4["w2_gs"], sorted_ids, its, n,
                          numel, part, pdl)
    return part


def kernel_fn(data, state):
    opts = BASELINE_OPTS
    if state["ours_ok"]:
        opts = _opts_for(data["hidden_states"].shape[0], state)
    return _forward(data, state, opts)


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
    ms = sorted(s.elapsed_time(e) for s, e in times)
    return 1000 * ms[len(ms) // 2]


def _w4_tune(state, seeds=(11, 12)):
    """Per TUNE_BS batch size, times Marlin's GEMM1 + silu_and_mul and every fitting W4 GEMM1
    config, and Marlin's GEMM2 + finalizer and every W4 GEMM2 config + its finalizer, on
    synthetic routings like the judge's; keeps the fastest of each (summed over seeds). Only
    timings are kept, nothing from any input."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    bank = state["banks"][(0, 32)]
    ws = state["workspace"]
    w4 = state["w4"]
    policy = {}
    bad = set()
    for bs in TUNE_BS:
        tot = [{}, {}, {}, {}]
        for seed in seeds:
            data = _make_inputs(bs, 977 * seed + bs)
            bsm = _opts_for(bs)["bsm"]
            numel = bs * TOP_K
            max_padded = (numel * bsm if numel < N_EXPERTS + 1
                          else numel + (N_EXPERTS + 1) * (bsm - 1))
            max_padded = (max_padded + 3) // 4 * 4
            topk_ids = data["topk_ids"].to(torch.int32).contiguous()
            tw = data["topk_weights"].float().contiguous()
            x = data["hidden_states"].contiguous()
            sorted_ids, block_experts, n_padded, local_ids = _ext().route_align(
                topk_ids, 0, 32, bsm, max_padded, 0, 0, 0, 0, 1, 1)[:4]
            cache1 = torch.empty((numel, 8192), device="cuda", dtype=torch.bfloat16)
            inter = torch.empty((numel, 4096), device="cuda", dtype=torch.bfloat16)
            cache3 = torch.empty((numel, 4096), device="cuda", dtype=torch.bfloat16)
            common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True,
                          b_q_type=scalar_types.float4_e2m1f, is_k_full=True,
                          use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

            def marlin1():
                moe_wna16_marlin_gemm(
                    x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=TOP_K,
                    size_m=bs, size_n=8192, size_k=4096, **common)
                silu_and_mul(cache1, inter, expert_ids=local_ids.view(-1), expert_step=1)

            def marlin2():
                moe_wna16_marlin_gemm(
                    inter, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=1,
                    size_m=numel, size_n=4096, size_k=4096, **common)
                _ext().finalize(cache3.view(bs, TOP_K, 4096), tw, state["norm_w"],
                                local_ids.view(bs, TOP_K), 0, POST_NORM_EPS)

            tot[0][None] = tot[0].get(None, 0.0) + _graph_time(marlin1, flush)
            tot[1][None] = tot[1].get(None, 0.0) + _graph_time(marlin2, flush)
            marlin1()
            ref1 = inter.clone()
            marlin2()
            ref2 = cache3.clone()
            live = local_ids.view(-1) >= 0
            for i, cfg in enumerate(W4_CONFIGS[:W4_TUNED]):
                if not _w4_fits(i, bs) or i in bad:
                    continue
                sids, _, _, lids, items, nitems = _ext().route_align(
                    topk_ids, 0, 32, bsm, max_padded, 8 * cfg[1], 0, 0, 0, 1, 1)[:6]
                res = {}
                if cfg[0] == 1:
                    out = torch.zeros((numel, 4096), device="cuda", dtype=torch.bfloat16)

                    def run():
                        _w4_gemm1(w4, i, x, sids, items, nitems, lids, numel, out)
                        res["out"] = out
                else:
                    def run():
                        part = _w4_gemm2(w4, i, inter, sids, items, nitems, numel)
                        res["part"] = part
                        _w4_ext().w4_finalize(part, tw, state["norm_w"], lids.view(-1),
                                              POST_NORM_EPS, False, 1)
                k = cfg[0] - 1 + (0 if cfg[6] in (0, 5) else 2)
                tot[k][i] = tot[k].get(i, 0.0) + _graph_time(run, flush)
                if cfg[6] not in (0, 5):
                    continue
                run()
                ref = ref1 if cfg[0] == 1 else ref2
                got = res["out"] if cfg[0] == 1 else res["part"].sum(0)
                err = _rel(got[live], ref[live])
                if not err < 2e-2:
                    print(f"TUNE bad cfg {i} {cfg} bs={bs} seed={seed}: rel diff {err:.2e}")
                    bad.add(i)
        n = len(seeds)
        for k in range(4):
            line = " ".join(f"{'m' if c is None else c}:{t / n:.1f}"
                            for c, t in sorted(tot[k].items(), key=lambda kv: kv[1]))
            print(f"TUNE {'diag ' if k >= 2 else ''}g{k % 2 + 1} bs={bs} {line}")
        for k in range(2):
            for i in bad:
                tot[k].pop(i, None)
        g1, g2 = (min(tot[k], key=tot[k].get) for k in range(2))
        # Whole forward: programmatic dependent launch off or on, and the finalizer's cluster
        # size (only with the W4 GEMM2, whose FP32 partials it reads).
        st = dict(state, w4_ok=True)
        variants = [(pdl, fcl) for pdl in (False, True) for fcl in ((1, 4, 8) if g2 is not None
                                                                     else (1,))]
        tv = {v: 0.0 for v in variants}
        try:
            for seed in seeds:
                data = _make_inputs(bs, 977 * seed + bs)
                for v in variants:
                    o = dict(_opts_for(bs), g1=g1, g2=g2, pdl=v[0], fcl=v[1])
                    tv[v] += _graph_time(lambda: _forward(data, st, o), flush)
        except Exception as exc:
            print(f"TUNE forward variants failed: {type(exc).__name__}: {exc}")
            tv = {(False, 1): 0.0}
        line = " ".join(f"pdl={int(v[0])},cl={v[1]}:{t / n:.1f}" for v, t in tv.items())
        print(f"TUNE forward bs={bs} {line}")
        pdl, fcl = min(tv, key=tv.get)
        policy[bs] = (g1, g2, pdl, fcl)
    print(f"TUNE policy {policy}")
    return policy


def _make_inputs(bs, seed, dist="normal", device="cuda"):
    """Routing like the judge's: top-8 of uniform scores plus heavy-tailed expert popularity
    (or a few hot experts), rotated so experts 0-31 get the most routes."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((N_EXPERTS,), generator=gen, device=device)
        u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((N_EXPERTS,), generator=gen, device=device).pow(6) * 4
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=N_EXPERTS // 32)
    ids = (ids - 32 * pairs.argmax()) % N_EXPERTS
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": N_EXPERTS,
                   "local_start": 0, "local_count": 32, "top_k": TOP_K, "sharding": "ep8"},
    }


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def _self_test(state):
    """Our path against SGLang's on the judge's test sizes; any mismatch keeps SGLang's."""
    worst = 0.0
    for bs in (1, 4, 8, 16, 32, 33, 128, 256, 257):
        for dist in ("normal", "skewed"):
            data = _make_inputs(bs, 100 + bs, dist)
            ref = _forward(data, state, BASELINE_OPTS)
            out = _forward(data, state, _opts_for(bs))
            if not torch.isfinite(out).all():
                print(f"SELFTEST bs={bs} {dist}: non-finite")
                return False
            worst = max(worst, _rel(out, ref))
    torch.cuda.synchronize()
    print(f"SELFTEST worst rel diff vs sglang path {worst:.2e}")
    return worst < 5e-3


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant_ref(packed, sf, gs):
    """NVFP4 [N, K/2] bytes, [N, K/16] E4M3 scales, FP32 global scale -> FP32 [N, K]."""
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    vals = torch.where(codes >= 8, -grid[codes & 7], grid[codes & 7])
    n, k = vals.shape
    return (vals.view(n, k // 16, 16) * (sf.float() * gs).unsqueeze(-1)).view(n, k)


def _ref_forward(data, weights):
    """The FP32 computation the judge checks against, for the self-test."""
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    out = torch.zeros_like(x)
    w = weights["post_norm_w"].float()
    for e in torch.unique(ids).tolist():
        if not 0 <= e < 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _dequant_ref(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = inter @ _dequant_ref(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        y = y * (1.0 + w)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].unsqueeze(1))
    return out


# The judge allows 1.5e-2 relative L2 against FP32; Marlin's path measured 5.2e-3 to 5.5e-3 on
# the self-test inputs (check 5dea75d7c668473b), ours 3.7e-3 to 3.8e-3.
W4_TEST_TOL = 8e-3


def _w4_self_test(state, weights, extra=()):
    """Each W4 config the policy uses (and any in extra) against FP32, at the judge's test
    sizes; the Marlin path's error is printed beside it."""
    worst = 0.0
    for bs in (1, 4, 8, 16, 32, 33, 128, 257):
        for dist in ("normal", "skewed"):
            data = _make_inputs(bs, 200 + bs, dist)
            ref = _ref_forward(data, weights)
            opts = _opts_for(bs, state)
            base = _rel(_forward(data, state, dict(opts, g1=None, g2=None)), ref)
            variants = [("policy", opts)] + [(f"g2={c}", dict(opts, g2=c)) for c in extra
                                             if _w4_fits(c, bs)]
            for name, o in variants:
                out = _forward(data, state, o)
                err = _rel(out, ref) if torch.isfinite(out).all() else float("inf")
                worst = max(worst, err)
                print(f"W4TEST bs={bs} {dist} {name}: vs fp32 {err:.2e} (marlin path {base:.2e})")
    torch.cuda.synchronize()
    print(f"W4TEST worst rel err vs fp32 {worst:.2e}")
    return worst < W4_TEST_TOL


def prepare(weights):
    state = _prepare_base(weights)
    try:
        _ext()
        state["ours_ok"] = True
        state["ours_ok"] = _self_test(state)
    except Exception as exc:
        state["ours_ok"] = False
        print(f"OURS disabled: {type(exc).__name__}: {exc}")
    state["w4_ok"] = False
    if state["ours_ok"]:
        try:
            t0 = time.time()
            state["w4"] = _w4_prepare(weights, 0, 32)
            t1 = time.time()
            if state["w4"] is None:
                print("W4 disabled: block scales outside positive normal E4M3")
            else:
                _w4_ext()
                t2 = time.time()
                state["w4_policy"] = _w4_tune(state)
                print(f"W4 repack {t1 - t0:.1f} s, extension build {t2 - t1:.1f} s, "
                      f"tuning {time.time() - t2:.1f} s")
                state["w4_ok"] = True
                state["w4_ok"] = _w4_self_test(state, weights)
        except Exception as exc:
            state["w4_ok"] = False
            print(f"W4 disabled: {type(exc).__name__}: {exc}")
    torch.cuda.synchronize()
    return state


# Generated by tools/moe_make_bench.py: measurements timed inside prepare().
_STREAM_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

__global__ void __launch_bounds__(512) stream_kernel(const uint4* __restrict__ p, size_t n,
                                                     unsigned* __restrict__ out) {
    unsigned acc = 0;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                         : "l"(p + i + u * stride));
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
    }
    for (; i < n; i += stride) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) {
        out[0] = acc;
    }
}

}  // namespace

void stream_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads) {
    const size_t n = buf.numel() / 16;
    stream_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(buf.data_ptr()), n,
        reinterpret_cast<unsigned*>(out.data_ptr()));
}
"""


def _graph_time(fn, flush, iters=30):
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
    ms = sorted(s.elapsed_time(e) for s, e in times)
    return 1000 * ms[len(ms) // 2]


def _empty_graph_us(flush):
    x = torch.zeros(1, device="cuda")
    return _graph_time(lambda: x.add_(1), flush)


def _stream_bench(flush):
    ext = load_inline(name="okc_moe_stream", cpp_sources=(
        "#include <torch/extension.h>\n"
        "void stream_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads);"),
        cuda_sources=_STREAM_SRC, functions=["stream_read"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    big = torch.randint(0, 255, (640 << 20,), dtype=torch.uint8, device="cuda")
    base = _empty_graph_us(flush)
    print(f"STREAM empty graph replay {base:.1f} us (subtract from the times below)")
    for mb in (9, 18, 28, 57, 113, 170, 283, 453, 640):
        buf = big[: mb << 20]
        best = None
        line = f"STREAM {mb:4d} MB"
        for blocks, threads in ((132, 512), (264, 512), (528, 512), (1056, 256), (2112, 256)):
            us = _graph_time(lambda: ext.stream_read(buf, out, blocks, threads), flush) - base
            line += f" {blocks}x{threads}:{us:6.1f}us/{mb * 1.048576 / us:4.2f}TB/s"
            best = us if best is None else min(best, us)
        print(line + f" best {mb * 1.048576 / best:4.2f} TB/s")
    del big


_W13_EXPERT_BYTES = 8192 * 4096 // 2 + 8192 * 4096 // 16
_W2_EXPERT_BYTES = 4096 * 4096 // 2 + 4096 * 4096 // 16


def _split_bench(state, flush, base):
    bank = state["banks"][(0, 32)]
    size_n = bank["size_n"]
    ws = state["workspace"]
    for bs in (1, 16, 32, 8, 128, 256):
        for seed in (11, 12, 13):
            data = _make_inputs(bs, 977 * seed + bs)
            ids = data["topk_ids"]
            local = ids[(ids >= 0) & (ids < 32)]
            counts = torch.bincount(local.long(), minlength=32)
            active = int((counts > 0).sum())
            routes = int(counts.sum())
            top = sorted(counts.tolist(), reverse=True)[:6]
            opts = _opts_for(bs)
            bsm = opts["bsm"]
            numel = bs * TOP_K
            max_padded = numel * bsm if numel < N_EXPERTS + 1 else numel + (N_EXPERTS + 1) * (bsm - 1)
            max_padded = (max_padded + 3) // 4 * 4
            topk_ids = ids.to(torch.int32).contiguous()
            tw = data["topk_weights"].float().contiguous()
            sorted_ids, block_experts, n_padded, local_ids = _ext().route_align(
                topk_ids, 0, 32, bsm, max_padded, 0, 0)[:4]
            padded_rows = int(n_padded[0])
            x = data["hidden_states"].contiguous()
            cache1 = torch.empty((numel, 2 * size_n), device="cuda", dtype=torch.bfloat16)
            cache2 = torch.empty((numel, size_n), device="cuda", dtype=torch.bfloat16)
            cache3 = torch.empty((numel, 4096), device="cuda", dtype=torch.bfloat16)
            common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True,
                          b_q_type=scalar_types.float4_e2m1f, is_k_full=True,
                          use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

            def g1():
                moe_wna16_marlin_gemm(
                    x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=TOP_K,
                    size_m=bs, size_n=2 * size_n, size_k=4096, **common)

            def act():
                silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1)

            def g2():
                moe_wna16_marlin_gemm(
                    cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=1,
                    size_m=numel, size_n=4096, size_k=size_n, **common)

            def fin():
                _ext().finalize(cache3.view(bs, TOP_K, 4096), tw, state["norm_w"],
                                local_ids.view(bs, TOP_K), 0, POST_NORM_EPS)

            t1 = _graph_time(g1, flush) - base
            ta = _graph_time(act, flush) - base
            t2 = _graph_time(g2, flush) - base
            tf = _graph_time(fin, flush) - base
            tall = _graph_time(lambda: _forward(data, state, opts), flush) - base
            b1 = active * _W13_EXPERT_BYTES
            b2 = active * _W2_EXPERT_BYTES
            print(f"SPLIT bs={bs:3d} seed={seed} active={active:2d} routes={routes:4d} "
                  f"top={top} padded_rows={padded_rows} | g1 {t1:6.1f}us "
                  f"{b1 / t1 / 1e6:4.2f}TB/s | act {ta:5.1f} | g2 {t2:6.1f}us "
                  f"{b2 / t2 / 1e6:4.2f}TB/s | fin {tf:5.1f} | all {tall:6.1f} | "
                  f"floor@4.8 g1 {b1 / 4.8e6:5.1f} g2 {b2 / 4.8e6:5.1f}")


def _w4_bench(state, flush, base, sizes=(1, 16, 32, 8, 128, 256), seeds=(11, 12)):
    """GEMM1 + SwiGLU per shape: Marlin's GEMM1 and silu_and_mul against every W4 config that
    fits, on the same routing, then the whole forward with Marlin's and with the chosen W4."""
    if not state.get("w4_ok"):
        print("W4BENCH skipped: w4 path disabled")
        return
    ext = _w4_ext()
    for i, cfg in enumerate(W4_CONFIGS):
        regs, smem, blocks = ext.w4_info(i)
        print(f"W4CFG {i:2d} kind,nt,rows,s,w,d={cfg} regs={regs} smem={smem} "
              f"blocks/SM={blocks} warps/SM={blocks * cfg[4]}")
    bank = state["banks"][(0, 32)]
    size_n = bank["size_n"]
    ws = state["workspace"]
    w4 = state["w4"]
    for bs in sizes:
        for seed in seeds:
            data = _make_inputs(bs, 977 * seed + bs)
            ids = data["topk_ids"]
            counts = torch.bincount(ids[(ids >= 0) & (ids < 32)].long(), minlength=32)
            active = int((counts > 0).sum())
            top = sorted(counts.tolist(), reverse=True)[:4]
            opts = _opts_for(bs)
            bsm = opts["bsm"]
            numel = bs * TOP_K
            max_padded = numel * bsm if numel < N_EXPERTS + 1 else numel + (N_EXPERTS + 1) * (bsm - 1)
            max_padded = (max_padded + 3) // 4 * 4
            topk_ids = ids.to(torch.int32).contiguous()
            tw = data["topk_weights"].float().contiguous()
            x = data["hidden_states"].contiguous()
            sorted_ids, block_experts, n_padded, local_ids = _ext().route_align(
                topk_ids, 0, 32, bsm, max_padded, 0, 0)[:4]
            cache1 = torch.empty((numel, 2 * size_n), device="cuda", dtype=torch.bfloat16)
            cache2 = torch.empty((numel, size_n), device="cuda", dtype=torch.bfloat16)
            common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True,
                          b_q_type=scalar_types.float4_e2m1f, is_k_full=True,
                          use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

            def marlin():
                moe_wna16_marlin_gemm(
                    x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=TOP_K,
                    size_m=bs, size_n=2 * size_n, size_k=4096, **common)
                silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1)

            b1 = active * _W13_EXPERT_BYTES
            tm = _graph_time(marlin, flush) - base
            ref_inter = cache2.clone()
            line = (f"W4BENCH bs={bs:3d} seed={seed} active={active:2d} top={top} "
                    f"floor@4.8={b1 / 4.8e6:5.1f} marlin+act={tm:6.1f}us({b1 / tm / 1e6:4.2f}TB/s)")
            print(line)
            live = local_ids.view(-1) >= 0
            results = []
            for i, cfg in enumerate(W4_CONFIGS):
                if cfg[0] != 1 or not _w4_fits(i, bs):
                    continue
                sids, _, _, lids, items, nitems = _ext().route_align(
                    topk_ids, 0, 32, bsm, max_padded, 8 * cfg[1], 0)[:6]
                inter = torch.zeros((numel, size_n), device="cuda", dtype=torch.bfloat16)

                def run():
                    _w4_gemm1(w4, i, x, sids, items, nitems, lids, numel, inter)

                t = _graph_time(run, flush) - base
                err = _rel(inter[live], ref_inter[live])
                results.append((t, i))
                print(f"W4BENCH   cfg {i:2d} {cfg} {t:6.1f}us ({b1 / t / 1e6:4.2f}TB/s) "
                      f"x{tm / t:4.2f} vs marlin  rel diff {err:.1e}")
            best = min(results)

            # GEMM2 + finalizer on Marlin's intermediate: Marlin GEMM2 and the bf16 finalizer
            # against each W4 GEMM2 config and the FP32-partials finalizer.
            cache3 = torch.empty((numel, 4096), device="cuda", dtype=torch.bfloat16)

            def marlin2():
                moe_wna16_marlin_gemm(
                    ref_inter, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None,
                    None, ws, sorted_ids, block_experts, n_padded, tw, top_k=1,
                    size_m=numel, size_n=4096, size_k=size_n, **common)
                return _ext().finalize(cache3.view(bs, TOP_K, 4096), tw, state["norm_w"],
                                       local_ids.view(bs, TOP_K), 0, POST_NORM_EPS)

            b2 = active * _W2_EXPERT_BYTES
            tm2 = _graph_time(marlin2, flush) - base
            ref_out = marlin2()
            print(f"W4BENCH2 bs={bs:3d} seed={seed} floor@4.8={b2 / 4.8e6:5.1f} "
                  f"marlin2+fin={tm2:6.1f}us({b2 / tm2 / 1e6:4.2f}TB/s)")
            results2 = []
            for i, cfg in enumerate(W4_CONFIGS):
                if cfg[0] != 2 or not _w4_fits(i, bs):
                    continue
                sids, _, _, lids, items, nitems = _ext().route_align(
                    topk_ids, 0, 32, bsm, max_padded, 8 * cfg[1], 0)[:6]
                outs = []

                def run2():
                    part = _w4_gemm2(w4, i, ref_inter, sids, items, nitems, numel)
                    outs.append(_w4_ext().w4_finalize(part, tw, state["norm_w"], lids.view(-1),
                                                      POST_NORM_EPS, False, 1))

                t = _graph_time(run2, flush) - base
                run2()
                err = _rel(outs[-1], ref_out)
                results2.append((t, i))
                print(f"W4BENCH2   cfg {i:2d} {cfg} {t:6.1f}us ({b2 / t / 1e6:4.2f}TB/s) "
                      f"x{tm2 / t:4.2f} vs marlin  rel diff {err:.1e}")
            best2 = min(results2)
            tcur = _graph_time(lambda: _forward(data, state, dict(opts, g1=None, g2=None)),
                               flush) - base
            tpol = _graph_time(lambda: _forward(data, state, opts), flush) - base
            both = dict(opts, g1=best[1], g2=best2[1])
            tbest = _graph_time(lambda: _forward(data, state, both), flush) - base
            ref = _ref_forward(data, state["_weights"]) if "_weights" in state else None
            err = _rel(_forward(data, state, both), ref) if ref is not None else float("nan")
            print(f"W4BENCH   full forward: marlin both {tcur:6.1f}us, policy g1={opts['g1']} "
                  f"g2={opts['g2']} {tpol:6.1f}us, best g1={best[1]} g2={best2[1]} "
                  f"{tbest:6.1f}us (rel err vs fp32 {err:.2e})")


BENCH_PARTS = ("w4",)


def _run_bench(state):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    if "stream" in BENCH_PARTS:
        _stream_bench(flush)
    base = _empty_graph_us(flush)
    print(f"BENCH empty graph replay {base:.1f} us, subtracted below")
    if "split" in BENCH_PARTS:
        _split_bench(state, flush, base)
    if "w4" in BENCH_PARTS:
        _w4_bench(state, flush, base)


_prepare_without_bench = prepare


def prepare(weights):
    import time
    t0 = time.time()
    state = _prepare_without_bench(weights)
    print(f"PREP base {time.time() - t0:.1f} s")
    try:
        state["_weights"] = weights
        _run_bench(state)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"BENCH failed: {type(exc).__name__}: {exc}")
    state.pop("_weights", None)
    print(f"PREP total {time.time() - t0:.1f} s")
    return state

# Appended to bench.py by `python3 tools/moe_make_bench.py tools/moe_bench_w4x.py`: the W4
# GEMMs with weights by bulk copy (TMA, mode 7) and with large token tiles for heavily routed
# experts, against the policy prepare() tuned, per batch size, GEMM and routing seed. Every
# time is the median of 20 CUDA graph replays, each after a write that evicts L2, and includes
# the graph replay overhead (XB empty), the SwiGLU reduce (GEMM1 with K splits) or the
# finalizer (GEMM2), as the tuner times them.

XB_SEEDS = (("normal", 11), ("normal", 12), ("normal", 13), ("skewed", 21), ("skewed", 22))
XB_BS = (1, 8, 16, 32, 128, 256)


def _xb_cfg(t):
    return W4_CONFIGS.index(t)


# (label, small config, large-tile config or None, threshold in routes, smallest bs)
XB_G1 = [
    ("r8a", (1, 1, 8, 4, 4, 4, 0, 0, 0), None, 0, 1),
    ("r16w8l1", (1, 2, 16, 4, 8, 4, 0, 1, 0), None, 0, 1),
    ("tA", (1, 1, 8, 4, 4, 4, 7, 1, 0), None, 0, 1),
    ("tB", (1, 1, 8, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tC", (1, 1, 8, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tD", (1, 1, 8, 2, 8, 4, 7, 1, 0), None, 0, 1),
    ("tE", (1, 1, 8, 2, 4, 8, 7, 1, 0), None, 0, 1),
    ("tF", (1, 2, 16, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tG", (1, 2, 16, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tH", (1, 1, 1, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("u32", (1, 4, 32, 4, 4, 4, 0, 0, 0), None, 0, 32),
    ("u64", (1, 8, 64, 4, 8, 4, 0, 0, 0), None, 0, 32),
    ("h8+32", (1, 1, 8, 4, 4, 4, 0, 0, 0), (1, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+32", (1, 2, 16, 4, 4, 4, 0, 0, 0), (1, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+64", (1, 2, 16, 4, 4, 4, 0, 0, 0), (1, 8, 64, 4, 8, 4, 0, 0, 0), 64, 32),
    ("h16+64d2", (1, 2, 16, 4, 4, 4, 0, 0, 0), (1, 8, 64, 4, 4, 2, 0, 0, 0), 64, 32),
    ("h8s2+32", (1, 1, 8, 2, 8, 4, 0, 0, 0), (1, 4, 32, 2, 8, 4, 0, 0, 0), 32, 32),
]
XB_G2 = [
    ("r8a", (2, 1, 8, 4, 4, 4, 0, 0, 0), None, 0, 1),
    ("r16", (2, 2, 16, 4, 4, 4, 0, 0, 0), None, 0, 1),
    ("tA", (2, 1, 8, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tB", (2, 1, 8, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tC", (2, 1, 8, 2, 4, 8, 7, 1, 0), None, 0, 1),
    ("tD", (2, 2, 16, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("tE", (2, 2, 16, 4, 8, 4, 7, 1, 0), None, 0, 1),
    ("tF", (2, 1, 1, 4, 4, 8, 7, 1, 0), None, 0, 1),
    ("u32", (2, 4, 32, 4, 4, 4, 0, 0, 0), None, 0, 32),
    ("u64", (2, 8, 64, 4, 8, 4, 0, 0, 0), None, 0, 32),
    ("h8+32", (2, 1, 8, 4, 4, 4, 0, 0, 0), (2, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+32", (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 4, 32, 4, 4, 4, 0, 0, 0), 32, 32),
    ("h16+64", (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 8, 64, 4, 8, 4, 0, 0, 0), 64, 32),
    ("h16+64d2", (2, 2, 16, 4, 4, 4, 0, 0, 0), (2, 8, 64, 4, 4, 2, 0, 0, 0), 64, 32),
]


def _xb_fmt(t):
    return f"{t:.0f}" if t >= 100 else f"{t:.1f}"


def _xb_info():
    ext = _w4_ext()
    seen = set()
    for lst in (XB_G1, XB_G2):
        for _, small, big, _, _ in lst:
            for t in (small, big):
                if t is None or t in seen:
                    continue
                seen.add(t)
                regs, smem, blocks, threads, local = ext.w4_info(_xb_cfg(t))
                print(f"XINFO {t}: regs {regs} smem {smem} blocks/SM {blocks} "
                      f"warps/SM {blocks * threads // 32} local {local}")


def _xb_run_bench(state):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    print(f"XB empty graph replay {_empty_graph_us(flush):.1f} us")
    _xb_info()
    w4 = state["w4"]
    ext = _w4_ext()
    policy = state["w4_policy"]
    worst = {}
    for bs in XB_BS:
        g1p, g2p = policy[_w4_key(bs)][:2]
        numel = bs * TOP_K
        bsm = _opts_for(bs)["bsm"]
        max_padded = (numel * bsm if numel < N_EXPERTS + 1
                      else numel + (N_EXPERTS + 1) * (bsm - 1))
        max_padded = (max_padded + 3) // 4 * 4
        times = [{}, {}]
        route = []
        for dist, seed in XB_SEEDS:
            data = _make_inputs(bs, 977 * seed + bs, dist)
            topk_ids = data["topk_ids"].to(torch.int32).contiguous()
            tw = data["topk_weights"].float().contiguous()
            x = data["hidden_states"].contiguous()
            ids = topk_ids.view(-1)
            cnt = torch.bincount(ids[ids < 32].long(), minlength=32).tolist()
            live_c = [c for c in cnt if c]
            route.append(f"{dist[0]}{seed}:{len(live_c)}e/max{max(live_c)}/"
                         + "/".join(f"i{r}={sum((c + r - 1) // r for c in live_c)}"
                                    for r in (8, 16, 32, 64)))

            def lists(small, big, at, gemm):
                rows = 8 * small[1]
                brow = 8 * big[1] if big is not None else 0
                r = ext.route_align(topk_ids, 0, 32, bsm, max_padded,
                                    rows if gemm == 1 else 0, rows if gemm == 2 else 0,
                                    brow if gemm == 1 else 0, brow if gemm == 2 else 0,
                                    max(at, 1), max(at, 1))
                sids, lids = r[0], r[3]
                its, n = (r[4], r[5]) if gemm == 1 else (r[6], r[7])
                bits, bn = (r[8], r[9]) if gemm == 1 else (r[10], r[11])
                b = (_xb_cfg(big), bits, bn) if big is not None else None
                return sids, lids, its, n, b

            inter_ref = torch.empty((numel, 4096), device="cuda", dtype=torch.bfloat16)
            cur1 = ("cur", W4_CONFIGS[g1p], None, 0, 1)
            cur2 = ("cur", W4_CONFIGS[g2p], None, 0, 1)
            for gemm, variants in ((1, [cur1] + XB_G1), (2, [cur2] + XB_G2)):
                ref = None
                for label, small, big, at, min_bs in variants:
                    if bs < min_bs or not _w4_fits(_xb_cfg(small), bs):
                        continue
                    if big is not None and not _w4_fits(_xb_cfg(big), bs):
                        continue
                    sids, lids, its, n, b = lists(small, big, at, gemm)
                    live = lids.view(-1) >= 0
                    ci = _xb_cfg(small)
                    res = {}
                    if gemm == 1:
                        out = inter_ref if label == "cur" else torch.zeros_like(inter_ref)

                        def run():
                            _w4_gemm1(w4, ci, x, sids, its, n, lids, numel, out, False, b)
                            res["o"] = out
                    else:
                        def run():
                            part = _w4_gemm2(w4, ci, inter_ref, sids, its, n, numel, False, b)
                            res["o"] = ext.w4_finalize(part, tw, state["norm_w"], lids.view(-1),
                                                       POST_NORM_EPS, False, 1)
                    try:
                        t = _graph_time(run, flush)
                        run()
                        got = res["o"][live] if gemm == 1 else res["o"]
                    except Exception as exc:
                        print(f"XB fail g{gemm} bs={bs} {label}: {type(exc).__name__}: {exc}")
                        continue
                    times[gemm - 1].setdefault(label, []).append(t)
                    if label == "cur":
                        ref = got.clone()
                    elif ref is not None:
                        err = _rel(got, ref) if torch.isfinite(got).all() else float("inf")
                        key = (gemm, label)
                        worst[key] = max(worst.get(key, 0.0), err)
        print(f"XROUTE bs={bs} " + " ".join(route))
        for k, cur in ((0, g1p), (1, g2p)):
            line = " ".join(f"{lab}:" + "/".join(_xb_fmt(t) for t in ts)
                            for lab, ts in times[k].items())
            print(f"XB g{k + 1} bs={bs} cur={cur} {line}")
    print("XERR " + " ".join(f"g{g}:{lab}:{e:.1e}" for (g, lab), e in worst.items()))


def _w4_key(bs):
    return next((b for b in TUNE_BS if b >= bs), TUNE_BS[-1])


def _run_bench(state):
    if not state.get("w4_ok"):
        print("XB skipped: W4 path off")
        return
    _xb_run_bench(state)
