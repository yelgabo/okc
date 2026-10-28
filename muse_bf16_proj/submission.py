# okc: problem=muse_bf16_proj
"""SGLang's projections, with the FP32 router replaced by one fused CUDA kernel up to 32 tokens.

At m <= 32 SGLang runs the router as FP32 F.linear (m <= 8) or two BF16 GEMMs plus an add,
taking 10 to 26 us against a 1.8 us memory floor. The kernel here reads the 8 MB FP32 weight
once and does FP32 FMAs on CUDA cores, so the math matches F.linear without TF32.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


ROUTER_N = 256
ROUTER_K = 8192
FUSED_MAX_TOKENS = 32

# Grid: 16 row groups x 8 k-slices = 128 blocks, one per SM, all loads in flight at once.
# Each warp owns one weight row and one 1024-wide k-slice. The block's x slice is staged in
# shared memory once, so the 16 warps stop re-reading it from L1. The 8 k-slice blocks of a row
# group form a cluster and add their partial sums over distributed shared memory: no atomics,
# so the output needs no zeroing and the sum order is fixed.
_CUDA_SRC = r"""
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

constexpr int N = 256;
constexpr int K = 8192;
constexpr int ROWS_PER_BLOCK = 16;
constexpr int CLUSTER = 8;
constexpr int K_SLICE = K / CLUSTER;
constexpr int STEPS = K_SLICE / 128;

template <int M>
constexpr int smem_bytes() {
    return M * K_SLICE * 2 + M * ROWS_PER_BLOCK * 4;
}

template <int M>
__global__ void __cluster_dims__(1, CLUSTER, 1) __launch_bounds__(512, 1)
router_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
              float* __restrict__ out, int m) {
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    float* partial = reinterpret_cast<float*>(smem + M * K_SLICE * 2);

    cg::cluster_group cluster = cg::this_cluster();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * ROWS_PER_BLOCK + warp;
    const int kbase = blockIdx.y * K_SLICE;

    const float* wrow = w + (size_t)n * K + kbase + lane * 4;
    float4 wv[STEPS];
#pragma unroll
    for (int s = 0; s < STEPS; ++s) {
        wv[s] = __ldcs(reinterpret_cast<const float4*>(wrow + s * 128));
    }

    // Rows past m are zero-filled so the inner loop needs no bounds check.
    constexpr int CHUNKS = K_SLICE / 8;
    for (int i = threadIdx.x; i < M * CHUNKS; i += blockDim.x) {
        const int t = i / CHUNKS;
        const int c = i % CHUNKS;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t < m) {
            v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * K + kbase) + c);
        }
        reinterpret_cast<uint4*>(xs)[i] = v;
    }
    __syncthreads();

    float acc[M];
#pragma unroll
    for (int t = 0; t < M; ++t) {
        acc[t] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < STEPS; ++s) {
        const int k = s * 128 + lane * 4;
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const uint2 raw = *reinterpret_cast<const uint2*>(xs + t * K_SLICE + k);
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
            acc[t] = fmaf(wv[s].x, a.x, acc[t]);
            acc[t] = fmaf(wv[s].y, a.y, acc[t]);
            acc[t] = fmaf(wv[s].z, b.x, acc[t]);
            acc[t] = fmaf(wv[s].w, b.y, acc[t]);
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
            partial[t * ROWS_PER_BLOCK + warp] = v;
        }
    }
    cluster.sync();

    const int rank = cluster.block_rank();
    for (int j = threadIdx.x; j < M * ROWS_PER_BLOCK; j += blockDim.x) {
        const int t = j / ROWS_PER_BLOCK;
        if (j % CLUSTER == rank && t < m) {
            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < CLUSTER; ++c) {
                sum += cluster.map_shared_rank(partial, c)[j];
            }
            out[(size_t)t * N + blockIdx.x * ROWS_PER_BLOCK + j % ROWS_PER_BLOCK] = sum;
        }
    }
    // Peers read this block's shared memory above; it must outlive those reads.
    cluster.sync();
}

template <int M>
void launch(const float* wp, const __nv_bfloat16* xp, float* op, int m, cudaStream_t stream) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(router_kernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes<M>());
        configured = true;
    }
    const dim3 grid(N / ROWS_PER_BLOCK, CLUSTER);
    router_kernel<M><<<grid, 32 * ROWS_PER_BLOCK, smem_bytes<M>(), stream>>>(wp, xp, op, m);
}

torch::Tensor router_fwd(torch::Tensor x, torch::Tensor w) {
    const int m = x.size(0);
    auto out = torch::empty({m, N}, x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* wp = w.data_ptr<float>();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    float* op = out.data_ptr<float>();
    if (m <= 1) {
        launch<1>(wp, xp, op, m, stream);
    } else if (m <= 2) {
        launch<2>(wp, xp, op, m, stream);
    } else if (m <= 4) {
        launch<4>(wp, xp, op, m, stream);
    } else if (m <= 8) {
        launch<8>(wp, xp, op, m, stream);
    } else if (m <= 16) {
        launch<16>(wp, xp, op, m, stream);
    } else {
        launch<32>(wp, xp, op, m, stream);
    }
    return out;
}
"""

_CPP_SRC = "torch::Tensor router_fwd(torch::Tensor x, torch::Tensor w);"


def _router_split2(x, state):
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _router_baseline(x, state):
    if x.shape[0] <= 8:
        return F.linear(x.float(), state["router"])
    return _router_split2(x, state)


def _self_test(fused, w):
    gen = torch.Generator(device=w.device).manual_seed(0)
    for m in (1, 3, 8, 9, 16, 31, 32):
        x = torch.randn(m, ROUTER_K, generator=gen, device=w.device).to(torch.bfloat16)
        # The judge's probe rows: one power-of-two entry, so the exact answer is a weight column.
        x[0].zero_()
        x[0, 4097] = 2.0
        out = fused(x, w)
        torch.cuda.synchronize()
        ref = x.double() @ w.double().T
        err = ((out.double() - ref).norm() / ref.norm()).item()
        if not err < 1e-5:
            raise RuntimeError(f"m={m}: relative L2 error {err:.3e}")
        if not torch.equal(out[0], (2.0 * w[:, 4097]).float()):
            raise RuntimeError(f"m={m}: probe row is not exact")


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()

    state["router_fused"] = None
    try:
        ext = load_inline(
            name="okc_router_v2",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["router_fwd"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        )
        _self_test(ext.router_fwd, router)
        state["router_fused"] = ext.router_fwd
        print("router kernel: fused")
    except Exception as exc:
        print(f"router kernel: baseline ({type(exc).__name__}: {exc})")
    return state


def kernel_fn(data, state):
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][: data["config"]["n"]], None)
    fused = state["router_fused"]
    if fused is not None and x.shape[0] <= FUSED_MAX_TOKENS:
        return fused(x.contiguous(), state["router"])
    return _router_baseline(x, state)
