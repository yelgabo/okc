# okc: problem=muse_moe_w4a16
"""The routed experts as one persistent CUDA launch.

Grid = SM count, one CTA per SM. Every CTA runs the routing prologue redundantly from
topk_ids (histogram, pair-row offsets, a job list of up to 8 tokens of one expert each), then
walks a static contiguous slice of the unit list: phase A units (16 W13 rows, 8 gate rows and
their 8 up rows, over the full K of 4096) produce SwiGLU output in bf16 `inter`; phase B units
(16 W2 rows) produce FP32 y times (1 + post_norm_w) and per-route sums of squares; a software
grid barrier; then the norm and weighted sum per token.

A unit is one 36,864-byte bulk copy (codes plus E4M3 scales, repacked in prepare) into a
3-slot shared-memory ring. Eight compute warps split K (512 each) with mma.sync m16n8k16, the
weights as A dequantized in registers and the job's x rows held in registers for the whole
job; a ninth warp sums the warps' partials and runs the epilogue. Phase B units of a job wait
for its phase A units through a per-job counter, so GEMM2 starts while GEMM1 drains.

Numerics follow SGLang's Marlin path: gate and up are rounded to bf16 before SiLU, SwiGLU's
output is bf16, GEMM2's output is rounded to bf16 before the norm, FP32 everywhere else.
"""

from typing import TypedDict

import time

import torch
from torch.utils.cpp_extension import load_inline


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
HIDDEN = 4096
POST_NORM_EPS = 1e-8
MAX_LOCAL = 32
# Tokens per launch; larger batches run as several launches.
CHUNK_TOKENS = 512
EMAX = CHUNK_TOKENS * TOP_K
JMAX = MAX_LOCAL + EMAX // 8
JMAX_MAIN = MAX_LOCAL + EMAX // 64
UNIT_BYTES = 36864
# Batches of this many tokens and more run the fat (wgmma) path.
MAIN_MIN_M = 17

_CPP_SRC = r"""
#include <torch/extension.h>
void moe_run(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor w13,
             torch::Tensor w2, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor nw,
             torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
             torch::Tensor done_a, torch::Tensor out, torch::Tensor w13f, torch::Tensor w2f,
             torch::Tensor xa, torch::Tensor xb, torch::Tensor fix, torch::Tensor fixcnt,
             torch::Tensor xready, torch::Tensor done_as, int64_t lo, int64_t nloc, double eps,
             int64_t variant, std::vector<int64_t> cost);
int64_t moe_smem_bytes();
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

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
};


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


// final_ss: out[t] as final_pass, with each route's sum of squares taken from its ybuf row here.
// One token per item; thread tid holds columns 8 tid .. 8 tid + 7. Pass 1 sums y^2 per local
// slot (warp shuffles, then the warps' partials through red), pass 2 re-reads the rows. ybuf is
// read only after the grid barrier, as final_pass reads it. red alternates between two halves
// by item parity, so one barrier per item orders the next item's writes after this item's
// reads of the other half.
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

//@SKINNY@

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
constexpr int kBarFin = 8;
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
                   || (mode >= 16 && mode <= 29) || (mode >= 31 && mode <= 75)
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
           || (mode >= 31 && mode <= 75);
}

// Modes 17 to 20 are mode 16 (two mma.sync chains, unit-aligned phase B) with one change each:
// 17 adds the clock stamps of mode 11 (timing only); 18 runs phase A alone (no phase B, no
// final pass: timing only, the GEMM1 gate's baseline); 19 runs phase A alone with every unit
// owned whole by one CTA (no cut unit, no fixup; timing only, the staged GEMM1 prototype); 20
// is 19's phase A followed by mode 16's phase B and final pass (a full, checked candidate).
// 22 is 20 with the stamps of 17. 23 is 20 with phase B units claimed at run time (dyn_b) and
// both phases walked largest n-tile count first; 24 is 23 with the stamps.
__host__ __device__ constexpr bool dual_chain(int mode) {
    return mode == 14 || (mode >= 16 && mode <= 29) || (mode >= 31 && mode <= 75);
}
__host__ __device__ constexpr bool is_diag(int mode) {
    return mode == 11 || mode == 17 || mode == 22 || mode == 24;
}
// Mode 25 is 23 with thread 0's globaltimer at the phase boundaries only (timing only). Mode
// 24's per-segment clock stamps add 14-18 us, too much to set its phase times beside the
// public kernel's eight thread-0 stamps; these cost a few stores per CTA.
__host__ __device__ constexpr bool is_light(int mode) {
    return mode == 25 || mode == 27 || mode == 29 || mode == 31 || mode == 33 || mode == 38
           || mode == 40 || mode == 41 || mode == 42 || mode == 44 || mode == 48 || mode == 49
           || mode == 54 || mode == 55 || mode == 56 || mode == 58 || mode == 61 || mode == 64
           || mode == 65 || mode == 67 || mode == 70 || mode == 71 || mode == 73 || mode == 75;
}
// Modes 26 and 27 (27 with the light stamps) are 23 with the setup of the public kernel's
// route(): every warp counts its slice of the routes once (match_any), the routing warp sums
// the per-warp counts instead of counting all E entries alone and the positions reuse them;
// and the x gather issues all of a CTA's row loads (16 B each) before its stores. Mode 25
// put setup at 7 / 11 / 16 us of routing at 32 / 128 / 256 tokens against 2-3 us for the
// public kernel (check d1f587760b1241dc).
__host__ __device__ constexpr bool fast_setup(int mode) {
    return (mode >= 26 && mode <= 29) || (mode >= 31 && mode <= 75);
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
    return mode == 28 || mode == 29 || (mode >= 31 && mode <= 75);
}
// Modes 31 and 33 (timing only) are 28 and 32 with the light stamps plus phase A counters per
// consumer warp (S.dg rows 0-15): cycles spent in weights and x waits, cycles from a unit's
// start to its accumulators, epilogue cycles, deferred publish cycles, stages, units and
// n-tiles; and the producers' phase A cycles waiting for a free slot. Since 2026-10-29 a wait
// is timed from before its first try_wait (mbar_wait_cyc); before that, a wait that succeeded
// on its first try counted zero and a failed first try's time was dropped.
// Mode 41 is 37 with these counters plus the same per-warp counters over phase B (pb_diag),
// mode 42 is 37 with the light stamps only (timing only, the overhead baseline for 41).
// Mode 44 is 43 with mode 41's counters.
__host__ __device__ constexpr bool pa_diag(int mode) {
    return mode == 31 || mode == 33 || mode == 38 || mode == 40 || mode == 41 || mode == 44;
}
// pb_diag: phase B counters per consumer warp go to global memory after S.dg's 18 x 8 int64
// per CTA (SmemMain stays the size the production modes have): p.inter as int64 from G * 144,
// 16 x 8 per CTA, columns as pa_record's, with column 7 the cycles waiting for a claimed unit.
__host__ __device__ constexpr bool pb_diag(int mode) {
    return mode == 41 || mode == 44;
}
// Modes 43 and 44 (44 with mode 41's counters) are 37 with the public kernel's stage structure
// on the wgmma path: a warp holds one 4-step code group (one uint4 and its scale word) and the
// next, not the stage's 16 code registers, and issues the next stage's weights and x waits and
// its first code group under the current stage's last wgmma group. Mode 41 put the consumers'
// waits at about 20% of the stage loops, 210-230 cycles each, while the producers found the ring
// full 70% of phase A (check ca43908f02874886), so the wait looks like a serial cost at the
// stage start rather than a stream that falls behind. The weights slot is released one wgmma
// group after the stage's last code load is issued (behind the sixth of eight groups, not the
// first), with mode 28's per-thread fence.proxy.async.shared::cta and arrive (docs/runs-moe.md,
// "Ordering of the ring slot releases"; the moved release under "Code groups with the next
// stage hoisted").
__host__ __device__ constexpr bool grp_codes(int mode) {
    return mode == 43 || mode == 44 || (mode >= 47 && mode <= 75);
}
// Mode 47 is 43 without the hoist: the code groups as 43, the waits and the first group at the
// stage's start as 37. In check 9d94ac75f4724b52, 43 beat 37 by 2.7 us at 256 tokens while
// 44's cycles per wait did not fall (239 against 41's 223) and its epilogue got shorter, which
// points at 43's smaller spill (8 / 100 B against 156 / 416) rather than the hoist.
__host__ __device__ constexpr bool grp_hoist(int mode) {
    return mode == 43 || mode == 44;
}
// Modes 48 and 49 are timing-only twins of 47: 48 with the light stamps, 49 with the light stamps
// and per-warp stage stamps (skew_st). Mode 41's counters put the consumers' waits at 210-240
// cycles at every stage boundary and 43's hoist did not shorten them, so the open question is
// whether the 16 warps of a CTA drift apart and a slot refills only behind the slowest one.
// skew_st: lane 0 of every consumer warp writes clock64 at each wgmma stage's start, after its
// weights wait, after its x wait and after its weights-slot arrive; the producers write their
// issue time per stage. Stage q is the weights ring position (lap * nsw + slot), which the x
// ring position equals. p.inter as int64 from kSkOff, kSkPer words per CTA: 8 header words
// (nsw, clock64 and globaltimer at the consumers' start and at the CTA's end), then per stage
// q < kSkQ 16 x 4 consumer words and 2 producer words. The mma.sync stages (jobs of 1 or 2
// n-tiles) are not stamped; the bench zeroes the region before the stamped launch.
__host__ __device__ constexpr bool skew_st(int mode) {
    return mode == 49 || mode == 56;
}
// Modes 55 and 56 are timing-only twins of 51 (55 the light stamps, 56 the light and per-warp
// stage stamps) with three more light stamps around the x gather: thread 0's globaltimer after its
// row stores (S.dg[17][6]) and after the gather's named barrier (S.dg[17][7]), and the x
// producer's globaltimer when the xready wait of its first phase A job ends (S.dg[16][7]). The
// gather and its release are 3.4-3.5 us behind the public kernel's at 128 and 256 tokens (48's
// stamps, d47c78bdc130424b) with no split between the stores, the fences and the release.
__host__ __device__ constexpr bool gather_st(int mode) {
    return mode == 55 || mode == 56 || mode == 58 || mode == 61 || mode == 64 || mode == 65
           || mode == 67 || mode == 70 || mode == 71 || mode == 73 || mode == 75;
}
// Modes 60 and 61 (61 with 55's light and gather stamps, timing only) are 51 without the
// gathering threads' __threadfence (fence.sc.gpu) between their fence.proxy.async.global and the
// gather barrier. The bar.sync and the red.release after it order every writer's row stores
// before the xready release by themselves, the chain of mode 37's phase A publish; each writer
// keeps its proxy fence, the barrier and the releases stay (docs/runs-moe.md, "Gather publish
// without the per-thread fence"). In 55's stamps the fences and the barrier took thread 0 4.3 /
// 2.1 us at 128 / 256 tokens against a public gather of 3.3 / 4.8 us in all, which has no
// per-thread fence (7063f86352344efa). Not built: in 48b9bf9d77d146ac 60 minus 51 was -0.5 to +0.0
// us at 32, 128 and 256 tokens; the barrier wait fell but the first red.release's MEMBAR.ALL.GPU
// took over the stores' drain.
__host__ __device__ constexpr bool gather_nofence(int mode) {
    return mode == 60 || mode == 61;
}
// Modes 57 to 59 are 51 with a cheaper x gather; no fence, barrier or release is removed or moved.
// In 55's stamps (7063f86352344efa) thread 0's row loads and stores took 1.7 / 5.6 us at 128 / 256
// tokens (two dependent batches at 256: the entries cta + k G run past kGB G) and its fences and the
// gather barrier 4.3 / 2.1, against 3.3 / 4.8 us for the public kernel's whole gather.
// gx_pre (57, 58, 59): right after the routing ids are in shared memory, every gathering thread
// loads its 16 bytes of the CTA's first kPreRows local entries' x rows, so the HBM round trip runs
// under the routing work; the batched loop takes only the entries after them.
// gx_st16 (57, 58): each lane stores one 16-byte group of the row's per-stage wgmma layout instead
// of four 4-byte words: the four words of a group come from lanes l, l ^ 2, l ^ 4, l ^ 6 and are
// exchanged with three shuffles, a quarter of the store instructions (the sector writes are the
// same: four lanes of a group already wrote adjacent words in one 4-byte store).
// 58 is 57 with the light and gather stamps (timing only), 59 is gx_pre alone (diagnostic arm).
// Not built: in e7e75c9510994c08, 57 minus 51 was +1.1 to +1.2 us and 59 -0.3 to +0.6 at 32, 128
// and 256 tokens; thread 0's gather got 0.8-1.8 us shorter, but each job's x readiness follows
// its slowest gathering CTA and moved 0.2-0.6 us.
constexpr int kPreRows = 4;
__host__ __device__ constexpr bool gx_pre(int mode) {
    return mode >= 57 && mode <= 59;
}
__host__ __device__ constexpr bool gx_st16(int mode) {
    return mode == 57 || mode == 58;
}
// Mode 50 is 47 with each route's sum of squares formed in the final pass from its finished ybuf
// row (the public kernel's OKC_WGSFINSS) instead of in the phase B epilogue: the epilogue keeps
// its ybuf stores and drops its three shuffle rounds and two FP32 atomics per n-tile. Mode 41
// put the phase B epilogue at 3.76 us per unit at 256 tokens against phase A's 1.71
// (ca43908f02874886), and in d47c78bdc130424b phase B is the largest piece of the same-process
// gap to the public kernel summed over 128 and 256 tokens (+2.9 and +7.3 us). The final pass
// then takes one token per CTA iteration and reads its rows twice, the second time from L2.
__host__ __device__ constexpr bool final_ss(int mode) {
    return mode >= 50 && mode <= 75;
}
// Mode 51 is 50 with the final-pass sums only from kFinssMinM tokens up, 47's path below. In
// 50's gate (8a7051fa926e43e1) 50 minus 47 was -4.1 to -8.3 us at 128 and 256 tokens but +0.7 at
// 32 on layer routing: one token per final-pass item leaves 100 of 132 CTAs idle at 32 tokens.
// The choice is uniform per launch (p.M), so the epilogue and the final pass agree on it.
constexpr int kFinssMinM = 64;
__host__ __device__ constexpr bool finss_by_m(int mode) {
    return mode >= 51 && mode <= 75;
}
template <int MODE>
__device__ __forceinline__ bool finss_on(int m) {
    return final_ss(MODE) && (!finss_by_m(MODE) || m >= kFinssMinM);
}
// Modes 52 and 53 are 51 with the last phase B units claimed in K halves of 8 stages: the claim
// values past units - xs map to the two halves of each of the last xs units, xs = G / 2 (52) or G
// (53), at most the unit count. At 32 tokens on standalone routing the last CTA's phase B ended
// 10.8 us after the CTAs' mean against 4.8 for the public kernel's stream-K split, with 1.7 whole
// units of about 22 us per CTA (1fbb9fbf524f4ed7); halves end that last wave in half the time. A
// half adds its partial into a phase B fixup slot (fixup, the cut-unit path of modes 12-16,
// indexed by the unit's place in the split tail, below G) and the second half to finish runs the
// epilogue. Mode 54 is 52 with the light stamps (timing only). Both lost at every size in
// 209cf29bf91c4ea2 (+3.3 to +8.5 us against 51): the halves' fixup costs more than the shorter
// last wave saves, and 54's tail did not shrink (7.9 against 48's 7.1 us at 32 tokens).
__host__ __device__ constexpr bool split_tail(int mode) {
    return mode >= 52 && mode <= 54;
}
template <int MODE>
__device__ __forceinline__ int tail_xs(int units, int G) {
    return split_tail(MODE) ? min(units, MODE == 53 ? G : G / 2) : 0;
}
struct Claim {
    int unit;
    int st_b;
    int st_e;
    int slot;
};
template <int MODE>
__device__ __forceinline__ Claim claim_of(int m, int units, int xs) {
    if constexpr (split_tail(MODE)) {
        const int base = units - xs;
        if (m >= base) {
            const int k = m - base;
            return {base + (k >> 1), (k & 1) * 8, (k & 1) * 8 + 8, k >> 1};
        }
    }
    return {m, 0, 16, -1};
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
    return mode == 34 || mode == 37 || (mode >= 39 && mode <= 75);
}
// Modes 36 to 38 are 28 with SiLU's reciprocal as one MUFU.RCP (rcp.approx.ftz, as the public
// kernel's __fdividef): __frcp_rn compiles to MUFU.RCP plus a range check that branches to a
// subroutine, inside a convergence region per element, 16 per thread per unit at 8 n-tiles.
// Mode 31's counters put mode 28's phase A epilogue at 3.0 us per unit at 256 tokens with the
// consumers never waiting for weights or x (check c7db20e053254b52). [Corrected 2026-10-29: that
// counter skipped the first try_wait; timed in full the consumers wait about 20% of the stage
// loops, check ca43908f02874886.] 37 is 36 without the per-thread fence_gpu (as 34), 38 is 36
// with the phase A counters (timing only).
__host__ __device__ constexpr bool fast_silu(int mode) {
    return mode >= 36 && mode <= 73;
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
    return mode == 18 || mode == 19 || mode == 45;
}
__host__ __device__ constexpr bool own_a(int mode) {
    return mode == 19 || mode == 20 || (mode >= 22 && mode <= 29) || (mode >= 31 && mode <= 45)
           || (mode >= 47 && mode <= 71);
}
// Modes 72 and 73 (73 with 55's light and gather stamps plus fix_st, timing only) are 51 with
// phase A dealt as mode 16 deals it: each CTA a contiguous range of equal cost under the stage
// cost table, units cut at range ends finished through the fixup slots, phase B claimed as in
// 51. Owned units (mode 20) beat 16 by 2-4 us at 128 and 256 tokens on the stage loop of 2026-10-29
// (before 37, 47 and 51 made stages and epilogues cheaper); the public kernel deals phase A this
// way, about 1.85 cut units per CTA, and its last CTA leaves phase A 5.2-9.1 us after the mean
// CTA against 9.5-14.2 for 51 (48b9bf9d77d146ac). The range boundaries are searched by a warp
// in setup instead of serially by thread 0 (mode 27 put four serial searches at 2.3-3.1 us), the
// leading cut unit's fixup slot is found there once, and a phase A segment runs one epilogue
// call whether it is whole or the last share of a cut unit, so the epilogue is not compiled twice.
__host__ __device__ constexpr bool stream_a(int mode) {
    return mode >= 72 && mode <= 75;
}
// Modes 74 and 75 (75 with 73's stamps plus the finisher's wait, timing only) are 72 with the
// public kernel's designated-finisher fixup in place of the last arrival. 73's stamps put 72's
// fixup at 3.3 us per share to its count and 6-9 us per finishing read-back (098949b28fe84bd1).
// Here every share adds its partial, every share but one then fences and counts with one
// red.release (no atom round trip, no second barrier), and the designated share waits for the
// others' count and reads the sum once, as the public kernel's does. The public kernel
// designates the share holding stage 15 and walks a range trailing cut share first, so that
// share runs last; our walk (n-tile count largest first) runs shares in another order, and a CPU
// model of it put the stage-15 rule's waits at p99 73 stages at 132 CTAs. So setup designates the
// share that ends last in cost time under our walk (dfin_flags; modelled wait zero in 99.9% of
// cut units). A
// walk in range order with one switch over n-tile counts made ptxas serialize the wgmmas
// (C7512). The slot is laid out [n-tile][thread] so a warp's 16-byte adds and reads cover whole
// lines (72's [thread][n-tile] touched 32 lines per instruction).
__host__ __device__ constexpr bool dfin_a(int mode) {
    return mode == 74 || mode == 75;
}
// fix_st: per CTA, thread 0's phase A fixup cycles and counts (fixup's ST words 0-3) and clock64
// at the consumers' start and at phase A's end (6, 7) to convert the cycles. Written to p.inter
// as int64 after the light words: G * 16 + cta * 8. Under dfin_a, word 4 holds the finisher's
// cycles waiting for the other shares' count (inside word 2).
__host__ __device__ constexpr bool fix_st(int mode) {
    return mode == 73 || mode == 75;
}
// Modes 23 and 24: a CTA's weights producer claims phase B units one at a time from a global
// counter, in S.ord order, when its ring has a free slot. Owned phase A leaves CTAs with up to
// one unit more than others and mode 20's static phase B ranges ignore that, so the CTA that
// frees first takes the next unit; S.ord puts the jobs whose phase A rounds come first (the
// snake deal walks S.ord) at the front, so claimed units find their inputs ready.
__host__ __device__ constexpr bool dyn_b(int mode) {
    return (mode >= 23 && mode <= 29) || (mode >= 31 && mode <= 44)
           || (mode >= 46 && mode <= 75);
}
// Modes 62 to 65 are 51 with phase A's last round(s) claimed at run time: the snake deal keeps
// the first ceil(units / G) - claim_rounds rounds, and the remaining units (S.ord order) go out
// through ctr[10] ahead of the phase B units, claim value m < na being phase A unit a_static + m
// and m >= na phase B unit m - na. Units stay whole on one CTA. With about 8.3 units per CTA at
// 128 tokens the deal gives a third of the CTAs one unit more whatever their speed; in 61's
// stamps the last CTA left phase A 9.5-14.2 us after the mean CTA against the public kernel's
// 5.2-9.1 (48b9bf9d77d146ac). 62 claims the last round, 63 the last two; 64 and 65 are 62 and 63
// with 55's light and gather stamps (timing only). Not built: in b7e43ee6a517487a 62 and 63 were
// +3.9 to +9.2 us against 51 at 32, 128 and 256 tokens; the end spread at 128 narrowed by 2-3 us
// but every CTA's phase A grew by about 5 us, also at 32 tokens where 6 units are claimed.
// 66 (67 its stamped twin) runs the same code with no round claimed (K = 0): the queue is read in
// phase A as in 62 but holds only phase B units, so 66 against 51 is the cost of the code shape.
// In ed4591a2477041c2 66 was +7.0 to +8.8 us against 51 at 128 and 256 tokens and tied 62: the
// loss is this code shape, not the claims. Not built.
__host__ __device__ constexpr bool claim_a(int mode) {
    return mode >= 62 && mode <= 67;
}
// Modes 68 to 71 claim phase A's last round(s) like 62 without 62's code shape. The consumers run
// 51's phase A loops unchanged over their dealt units, then the claimed units in a loop of their
// own per n-tile count. The weights producer claims and decodes, and puts a claimed phase A unit
// in S.bq as kBqA plus its first stage index (job * kUAM + unit * 16) and a phase B claim as its
// phase B unit index, so the x producer and the consumers decode a phase B entry as 51 does.
// 68 claims the last round, 69 none (K = 0, the null that prices the code shape: in
// ed4591a2477041c2 62's code with nothing claimed lost 7.0-8.8 us to 51); 70 and 71 are 68 and
// 69 with 55's light and gather stamps (timing only). Not built: in b2c92a71600d488b the null 69
// was +3.9 to +5.2 us against 51 at 128 and 256 tokens, phase B included, so 68 was not timed.
__host__ __device__ constexpr bool claim_lean(int mode) {
    return mode >= 68 && mode <= 71;
}
__host__ __device__ constexpr int claim_rounds(int mode) {
    return mode == 63 || mode == 65 ? 2
           : mode == 66 || mode == 67 || mode == 69 || mode == 71 ? 0 : 1;
}
constexpr int kBqA = 1 << 24;
// Phase A units (S.ord order, 32 per job) below this index are dealt; a multiple of G, so every
// CTA owns the same number of rounds.
template <int MODE>
__device__ __forceinline__ int a_static(int njobs, int G) {
    const int units = njobs * (kUAM / 16);
    if constexpr (claim_a(MODE)) {
        return min(units, max(0, (units + G - 1) / G - claim_rounds(MODE)) * G);
    }
    return units;
}
// a_static for the claim_lean modes, used only by the setup's own_n and the weights producer:
// with claim_a false a_static keeps the consumers' and the x producer's phase B decode at 51's
// (na folds to 0).
template <int MODE>
__device__ __forceinline__ int a_dealt(int njobs, int G) {
    const int units = njobs * (kUAM / 16);
    if constexpr (claim_lean(MODE)) {
        return min(units, max(0, (units + G - 1) / G - claim_rounds(MODE)) * G);
    }
    return a_static<MODE>(njobs, G);
}
// Mode 21 is the second launch of the staged pair (19, then 21): phase B and the final pass
// alone, on the xb a previous launch wrote. That launch completed before this one started
// (stream order, or griddepcontrol.wait under PDL: PTX ISA 9.7.15.14, 8.9.4), so phase B
// takes no done_as wait and no x gather; the issuing thread keeps its proxy fence.
// Modes 45 and 46 are the staged pair rebuilt on mode 37's components (lean stage, setup,
// SiLU, publish), its revisit trigger in docs/ruled-out.md: 45 is 19's phase A alone on owned
// units, 46 is 21's phase B and final pass alone with 37's claimed phase B units. 19 and 21
// lost to the fused 16 and 20 by 24-30 us (checks 9332e2a49aff4ff2, e221858b020c4dee), before
// 23, 26, 28 and 37 made the fused kernel's phase B, setup and stage loop cheaper.
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

constexpr int kSkQ = 256;
constexpr int kSkRow = kNC * 4 + 2;
constexpr int kSkPer = 8 + kSkQ * kSkRow;
constexpr long long kSkOff = 1ll << 20;

__device__ __forceinline__ long long* sk_cta(const Params& p) {
    return reinterpret_cast<long long*>(p.inter) + kSkOff
           + static_cast<long long>(blockIdx.x) * kSkPer;
}

// A predicated store rather than a lane-0 branch: a divergent block between the wgmma groups of
// a stage makes ptxas serialize them (C7513, docs/ruled-out.md).
template <int MODE>
__device__ __forceinline__ void sk_stamp(const Params& p, int q, int k, bool on) {
    if constexpr (skew_st(MODE)) {
        const long long c = clock64();
        long long* a = sk_cta(p) + 8 + static_cast<long long>(q) * kSkRow + k;
        asm volatile("{\n .reg .pred sp;\n setp.ne.s32 sp, %2, 0;\n @sp st.global.b64 [%0], %1;\n}"
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
    // Owned phase A (own_a): jobs by n-tile count, largest first, and this CTA's unit count.
    int ord[kJmaxM];
    int own_n;
    // dyn_b: the i-th claimed phase B unit (index into S.ord order, 16 per job), -1 when the
    // claims are exhausted, kBqEmpty until the weights producer writes it. claim_lean also puts
    // claimed phase A units here, as kBqA plus the unit's first stage index.
    int bq[kBqMax];
    long long dg[18][8];
    float* ssp;
    // stream_a: the fixup slot of this CTA's leading cut phase A unit (the CTA holding its first
    // stage). Last, so the production modes' field offsets stay as they were.
    int lead_slot;
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

// mbar_wait_to returning the cycles of the whole wait (pa_diag). The clock starts before the
// first try: try_wait is potentially blocking (PTX ISA 9.4, mbarrier.test_wait /
// mbarrier.try_wait), so a first try that returns true may have suspended the thread until the
// phase completed, and one that returns false may have suspended it for a time limit first.
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
//@WGMMA@

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
    // dfin_a: this cut share is its unit's designated finisher.
    bool fin;
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

// stage_at for phase A by a whole warp: the cost prefix S.pa never falls, so the jobs with
// pa[j] <= target are a prefix and their count less one is the j stage_at's scan stops at.
__device__ __forceinline__ int stage_at_a_warp(const SmemMain& S, int k, int G, int lane) {
    const int nj = S.njobs;
    if (k >= G) {
        return nj * kUAM;
    }
    if (k <= 0 || nj == 0) {
        return 0;
    }
    const unsigned total = static_cast<unsigned>(S.pa[nj]);
    const unsigned target = total * static_cast<unsigned>(k) / static_cast<unsigned>(G);
    int c = 0;
    for (int b = 0; b < nj; b += 32) {
        const int jj = b + lane;
        c += __popc(__ballot_sync(0xffffffffu,
                                  jj < nj && static_cast<unsigned>(S.pa[jj]) <= target));
    }
    const int j = c - 1;
    const int cost = S.cost[0][S.job_ntt[j]];
    return j * kUAM + static_cast<int>((target - static_cast<unsigned>(S.pa[j]))
                                       / static_cast<unsigned>(cost));
}

// dfin_a: the cost time at which the share of unit u in range [b, e) ends in the consumers' walk
// (n-tile count largest first, then seg_get's order) when nothing waits. Whole warp.
__device__ __forceinline__ int share_end(const SmemMain& S, int b, int e, int u, int lane) {
    const int ub = b >> 4;
    const int ue = (e - 1) >> 4;
    const int idx = (ub != ue && u == ub && (e & 15) != 0) ? 1 : 0;
    const int n_s = S.job_ntt[u >> 5];
    const int ns = seg_count(b, e);
    unsigned t = 0;
    for (int i = lane; i < ns; i += 32) {
        int gb;
        int ge;
        seg_get(b, e, i, gb, ge);
        const int n = S.job_ntt[gb >> 9];
        if (n > n_s || (n == n_s && i <= idx)) {
            t += static_cast<unsigned>((ge - gb) * S.cost[0][n]);
        }
    }
    return static_cast<int>(__reduce_add_sync(0xffffffffu, t));
}

// dfin_a: bit 16 when this CTA's cut share with st_b != 0 finishes its unit, bit 17 when its cut
// share with st_b == 0 does (the trailing one, or a range inside one unit starting on its first
// stage). A unit's finisher is the CTA whose share ends last by share_end (ties to the later
// CTA), so its wait for the other shares' counts is modelled at zero; every CTA derives it from
// the same shared values, so the CTAs holding the unit agree. lead is the CTA holding the first
// stage of the unit at b. One loop and one call site each, to keep the kernel small. Whole warp.
__device__ __forceinline__ int dfin_flags(const SmemMain& S, int b, int e, int lead, int cta,
                                          int G, int lane) {
    if (b >= e) {
        return 0;
    }
    const int ub = b >> 4;
    const int ue = (e - 1) >> 4;
    int f = 0;
#pragma unroll 1
    for (int q = 0; q < 2; ++q) {
        // q 0: the share with st_b != 0 (unit ub, from lead); q 1: the one with st_b == 0.
        const bool has = q == 0 ? (b & 15) != 0 : (ub == ue ? (b & 15) == 0 && (e & 15) != 0
                                                             : (e & 15) != 0);
        if (!has) {
            continue;
        }
        const int u = q == 0 ? ub : ue;
        const int k0 = q == 0 ? lead : cta;
        int bk = 0;
        int best = -1;
        int fc = k0;
#pragma unroll 1
        for (int k = k0 - 1; k < G; ++k) {
            const int ek = stage_at_a_warp(S, k + 1, G, lane);
            if (k >= k0 && bk < ek) {
                const int t = share_end(S, bk, ek, u, lane);
                if (t >= best) {
                    best = t;
                    fc = k;
                }
            }
            if (ek > u * 16 + 15) {
                break;
            }
            bk = ek;
        }
        if (fc == cta) {
            f |= 1 << (16 + q);
        }
    }
    return f;
}

// seg_info with stream_a's leading-cut slot from setup instead of seg_info's per-segment search.
template <int MODE>
__device__ __forceinline__ SegInfo seg_info_m(const SmemMain& S, int ph, int gb, int ge, int cta,
                                              int G) {
    if constexpr (stream_a(MODE)) {
        if (ph == 0) {
            SegInfo si;
            si.j = gb >> 9;
            si.u = (gb & 511) >> 4;
            si.st_b = gb & 15;
            si.st_e = ((ge - 1) & 15) + 1;
            si.partial = si.st_b != 0 || si.st_e != 16;
            const int ls = S.lead_slot;
            si.slot = si.st_b != 0 ? (dfin_a(MODE) ? ls & 0xffff : ls) : cta;
            si.fin = dfin_a(MODE) && ((ls >> (si.st_b != 0 ? 16 : 17)) & 1) != 0;
            si.r = -1;
            return si;
        }
    }
    return seg_info(S, ph, gb, ge, cta, G);
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

// Every share of a cut unit adds its partial into the owner's slot and counts its stages; the
// share that completes the count reads the sum back into fin, clears the slot and finishes
// the unit. acc itself is never written by anything but wgmma (ptxas C7515).
// ST (fix_st): warpgroup 0's thread 0 adds, in the branches it already takes, its cycles from
// the fixup's start to its count update and the count of such shares (S.dg[0][0], [1]), and for
// a share that finishes the unit the cycles to the end of the read-back ([2], [3]). Any stamp
// code placed after the fixup returned made mode 73 spill 288 B.
template <int NTT, bool AR, bool ST = false>
__device__ __forceinline__ bool fixup(const Params& p, SmemMain& S, const float (&acc)[32],
                                      float (&fin)[32], int ph, int slot, int nst, int wg,
                                      int tid_wg, int G) {
    const unsigned c0 = ST ? static_cast<unsigned>(clock()) : 0u;
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
        if (ST && wg == 0) {
            S.dg[0][0] += static_cast<unsigned>(clock()) - c0;
            S.dg[0][1] += 1;
        }
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
        if (ST && wg == 0) {
            S.dg[0][2] += static_cast<unsigned>(clock()) - c0;
            S.dg[0][3] += 1;
        }
    }
    return true;
}

// dfin_a: every share adds its partial into the slot; the designated share (is_fin, from
// setup's dfin_flags) does not count, waits for the others' count (16 less its own stages) and
// reads the whole sum back, its own adds included (same thread, same addresses: program order).
// Keeping its own accumulators out of the read-back keeps acc dead after the adds (74 spilled
// 20 / 216 B when the finisher added them in registers). Same slot and count words as fixup
// (phase A), the slot laid out [n-tile][thread] in float4.
// ST: words 0 and 1 as fixup's for the counting shares; 2 and 3 the finisher's cycles to the end
// of its read-back and its count; 4 its cycles waiting for the count (inside word 2).
template <int NTT, bool AR, bool ST = false>
__device__ __forceinline__ bool fixup_df(const Params& p, SmemMain& S, const float (&acc)[32],
                                         float (&fin)[32], int slot, int st_b, int st_e,
                                         bool is_fin, int wg, int tid_wg) {
    const unsigned c0 = ST ? static_cast<unsigned>(clock()) : 0u;
    float4* fp = reinterpret_cast<float4*>(p.fix) + static_cast<size_t>(slot * 4 + wg) * 1024
                 + tid_wg;
    int* cp = p.fixcnt + slot * 4 + wg;
#pragma unroll
    for (int i = 0; i < NTT; ++i) {
        red_add_v4(fp + i * 128, make_float4(acc[4 * i], acc[4 * i + 1], acc[4 * i + 2],
                                             acc[4 * i + 3]));
    }
    if (!is_fin) {
        fence_gpu<AR>();
        named_sync(kBarWg + wg, 128);
        if (tid_wg == 0) {
            red_release_add(cp, st_e - st_b);
            if (ST && wg == 0) {
                S.dg[0][0] += static_cast<unsigned>(clock()) - c0;
                S.dg[0][1] += 1;
            }
        }
        return false;
    }
    if (tid_wg == 0) {
        spin_ge(cp, 16 - (st_e - st_b), p.ctr, 512);
        if (ST && wg == 0) {
            S.dg[0][4] += static_cast<unsigned>(clock()) - c0;
        }
    }
    named_sync(kBarWg + wg, 128);
    fence_gpu<AR>();
#pragma unroll
    for (int i = 0; i < NTT; ++i) {
        const float4 v = __ldcg(fp + i * 128);
        fin[4 * i] = v.x;
        fin[4 * i + 1] = v.y;
        fin[4 * i + 2] = v.z;
        fin[4 * i + 3] = v.w;
        fp[i * 128] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (tid_wg == 0) {
        *cp = 0;
        if (ST && wg == 0) {
            S.dg[0][2] += static_cast<unsigned>(clock()) - c0;
            S.dg[0][3] += 1;
        }
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

// stream_a: a phase A segment's epilogue, called once whether the segment is a whole unit or a
// share of a cut unit, so the epilogue is compiled once per n-tile count, not twice. acc is only
// read here (ptxas C7515 forbids writing it outside wgmma).
template <int NTT, int MODE>
__device__ __forceinline__ void epilogue_a_seg(const Params& p, SmemMain& S,
                                               const float (&acc)[32], const SegInfo& si, int wg,
                                               int w, int tid_wg, int G) {
    float fin[32];
    bool fin_ok = true;
    if (si.partial) {
        if constexpr (dfin_a(MODE)) {
            fin_ok = fixup_df<NTT, fence_ar(MODE), fix_st(MODE)>(p, S, acc, fin, si.slot,
                                                                 si.st_b, si.st_e, si.fin, wg,
                                                                 tid_wg);
        } else {
            fin_ok = fixup<NTT, fence_ar(MODE), fix_st(MODE)>(p, S, acc, fin, 0, si.slot,
                                                              si.st_e - si.st_b, wg, tid_wg, G);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4 * NTT; ++i) {
            fin[i] = acc[i];
        }
    }
    if (fin_ok) {
        epilogue_a<NTT, MODE, true>(p, S, fin, si.j, si.u, wg, w, tid_wg);
    }
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

__device__ __forceinline__ long long* pb_row(const Params& p, int warp) {
    return reinterpret_cast<long long*>(p.inter) + gridDim.x * 18 * 8
           + (blockIdx.x * 16 + warp) * 8;
}

// pb_diag: one phase B unit's counters into the warp's global row (pb_row).
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
    // kLean: x slot 0's descriptor once per segment; a stage adds its slot's offset in 16-byte
    // units (shared addresses stay below 2^18, so the 14-bit address field cannot carry).
    const unsigned xb0 = kLean ? __shfl_sync(0xffffffffu, rg.xb, 0) : 0u;
    const uint64_t dx0 = ((xb0 & 0x3ffffu) >> 4) | (static_cast<uint64_t>(128 >> 4) << 16)
                         | (static_cast<uint64_t>(4096 >> 4) << 32);
    if constexpr (grp_codes(MODE)) {
        static_assert(kLean && DEPTH == 2 && !defer_a(MODE), "grp_codes extends mode 37");
        // Code group c (steps 4c to 4c + 3) sits in cb[c & 1]; scw[0] holds the scale bytes of
        // groups 0 and 1, scw[1] those of groups 2 and 3.
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
                    // Every generic read of this slot (groups 0 to 3, both scale words) is in
                    // this thread's program order before the fence. The last one was issued
                    // a whole group earlier (s == 8) so that the fence's MEMBAR does not find
                    // it still in flight.
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
    if constexpr (stream_a(MODE) && PH == 0) {
        epilogue_a_seg<NTT, MODE>(p, S, acc, si, wg, w, tid_wg, G);
    } else if (si.partial) {
        float fin[32];
        if (fixup<NTT, fence_ar(MODE)>(p, S, acc, fin, ph, si.slot, si.st_e - si.st_b, wg, tid_wg, G)) {
            if (ph == 0) {
                epilogue_a<NTT, MODE, true>(p, S, fin, si.j, si.u, wg, w, tid_wg);
            } else {
                epilogue_b<NTT, false, final_ss(MODE), finss_by_m(MODE)>(p, S, ss, fin, si.j, si.u,
                                                                   wg, w, tid_wg);
            }
        }
    } else if (ph == 0) {
        epilogue_a_unit<NTT, MODE>(p, S, acc, si, wg, w, tid_wg);
    } else {
        epilogue_b<NTT, vol_regs(MODE), final_ss(MODE), finss_by_m(MODE)>(p, S, ss, acc, si.j, si.u,
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
    if constexpr (stream_a(MODE) && PH == 0) {
        epilogue_a_seg<NTT, MODE>(p, S, acc, si, wg, w, tid_wg, G);
    } else if (si.partial) {
        float fin[32];
        if (fixup<NTT, fence_ar(MODE)>(p, S, acc, fin, ph, si.slot, si.st_e - si.st_b, wg, tid_wg, G)) {
            if (ph == 0) {
                epilogue_a<NTT, MODE, true>(p, S, fin, si.j, si.u, wg, w, tid_wg);
            } else {
                epilogue_b<NTT, false, final_ss(MODE), finss_by_m(MODE)>(p, S, ss, fin, si.j, si.u,
                                                                   wg, w, tid_wg);
            }
        }
    } else if (ph == 0) {
        epilogue_a_unit<NTT, MODE>(p, S, acc, si, wg, w, tid_wg);
    } else {
        epilogue_b<NTT, vol_regs(MODE), final_ss(MODE), finss_by_m(MODE)>(p, S, ss, acc, si.j, si.u,
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

// Segments run grouped by (phase, n-tile count), each group its own loop: with several
// wgmma shapes inside one enclosing loop ptxas serializes every wgmma (C7512, see the
// micro tests in docs/runs-moe.md); the producers walk the same order.
// claim_a phase A: the CTA's dealt units of this n-tile count, then its claimed units of this
// count from S.bq. Dealt units come before claimed ones in S.ord, so their counts are never
// smaller; the queue is read only once no dealt unit of a smaller count is left (the producers
// claim after their last dealt stage, which needs the consumers to have taken the earlier ones).
template <int NTT, int MODE>
__device__ __forceinline__ void run_group_ca(const Params& p, SmemMain& S, float* ss,
                                             float (&acc)[32], Ring& rg, int& xrel, int& qi,
                                             int cta, int G) {
    const int ns = S.own_n;
    const int s0 = a_static<MODE>(S.njobs, G);
    const int na = S.njobs * (kUAM / 16) - s0;
    const bool claims =
        ns == 0 || S.job_ntt[S.ord[own_unit(ns - 1, cta, G) >> 5]] >= NTT;
#pragma unroll 1
    for (int i = 0;; ++i) {
        int gb;
        if (i < ns) {
            int ge;
            seg_at<MODE>(S, 0, i, cta, G, gb, ge);
            if (S.job_ntt[gb >> 9] != NTT) {
                continue;
            }
        } else {
            if (!claims || qi >= kBqMax) {
                return;
            }
            const int m = bq_get(S, qi, p.ctr);
            if (m < 0 || m >= na) {
                return;
            }
            const int k = s0 + m;
            const int j = S.ord[k >> 5];
            if (S.job_ntt[j] != NTT) {
                return;
            }
            gb = j * kUAM + (k & 31) * 16;
            ++qi;
        }
        SegInfo si = seg_info(S, 0, gb, gb + 16, cta, G);
        if constexpr (NTT <= mma_max(MODE)) {
            run_segment_mma<NTT, 0, MODE>(p, S, ss, si, rg, G);
        } else {
            run_segment<NTT, 0, MODE>(p, S, ss, si, acc, rg, xrel, G);
        }
    }
}

template <int NTT, int PH, int MODE>
__device__ __forceinline__ void run_group(const Params& p, SmemMain& S, float* ss,
                                          float (&acc)[32], Ring& rg, int& xrel, int& qi,
                                          int cta, int G) {
    if constexpr (claim_a(MODE) && PH == 0) {
        run_group_ca<NTT, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        return;
    }
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
        SegInfo si = seg_info_m<MODE>(S, PH, gb, ge, cta, G);
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
                                               float (&acc)[32], Ring& rg, int& xrel, int& qi,
                                               int cta, int G) {
    if constexpr (dyn_b(MODE)) {
        run_group<8, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<7, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<6, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<5, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<4, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<3, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<2, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<1, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    } else {
        run_group<1, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<2, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<3, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<4, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<5, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<6, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<7, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
        run_group<8, PH, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    }
}

// claim_lean phase A: after every dealt unit, the claimed units of this n-tile count in claim
// order. Claims rise through S.ord past the dealt units, so their counts never rise and never
// exceed a dealt count: the producers' order (dealt units, then claims) is the consumers'.
template <int NTT, int MODE>
__device__ __forceinline__ void run_group_cla(const Params& p, SmemMain& S, float* ss,
                                              float (&acc)[32], Ring& rg, int& xrel, int& qi,
                                              int cta, int G) {
#pragma unroll 1
    while (qi < kBqMax) {
        const int m = bq_get(S, qi, p.ctr);
        if (m < kBqA) {
            return;
        }
        const int gb = m - kBqA;
        if (S.job_ntt[gb >> 9] != NTT) {
            return;
        }
        SegInfo si = seg_info(S, 0, gb, gb + 16, cta, G);
        if constexpr (NTT <= mma_max(MODE)) {
            run_segment_mma<NTT, 0, MODE>(p, S, ss, si, rg, G);
        } else {
            run_segment<NTT, 0, MODE>(p, S, ss, si, acc, rg, xrel, G);
        }
        ++qi;
    }
}

template <int MODE>
__device__ __forceinline__ void consumer_claim_a(const Params& p, SmemMain& S, float* ss,
                                                 float (&acc)[32], Ring& rg, int& xrel, int& qi,
                                                 int cta, int G) {
    run_group_cla<8, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_cla<7, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_cla<6, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_cla<5, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_cla<4, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_cla<3, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_cla<2, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    run_group_cla<1, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
}

// dyn_b phase B: the claimed units in claim order. Claims rise through S.ord, whose n-tile
// counts never rise, so one loop per count, largest first, takes them all in order.
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
        const int na = S.njobs * (kUAM / 16) - a_static<MODE>(S.njobs, G);
        const Claim c =
            claim_of<MODE>(m - na, S.njobs * 16, tail_xs<MODE>(S.njobs * 16, G));
        const int j = S.ord[c.unit >> 4];
        if (S.job_ntt[j] != NTT) {
            return;
        }
        SegInfo si;
        si.j = j;
        si.u = c.unit & 15;
        si.st_b = c.st_b;
        si.st_e = c.st_e;
        si.slot = c.slot >= 0 ? c.slot : cta;
        si.partial = c.slot >= 0;
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
                                               float (&acc)[32], Ring& rg, int& xrel, int qi,
                                               int cta, int G) {
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
        // Lane 0 zeroes the row it alone adds to later, so program order covers the reuse.
        for (int c = 0; c < 8; ++c) {
            pb_row(p, threadIdx.x >> 5)[c] = 0;
        }
    }
    const long long pc0 = pa_diag(MODE) ? clock64() : 0;
    int qi = 0;
    consumer_phase<0, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    if constexpr (claim_lean(MODE)) {
        consumer_claim_a<MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    }
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
        if (fix_st(MODE)) {
            S.dg[0][7] = clock64();
        }
    }
    if constexpr (dyn_b(MODE)) {
        consumer_dyn_b<MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
    } else if constexpr (!a_only(MODE)) {
        // a_only: phase B has no segments; leaving its loops out keeps them off the register
        // allocation of the phase A loops (mode 45 spilled 268 / 656 B with them).
        consumer_phase<1, MODE>(p, S, ss, acc, rg, xrel, qi, cta, G);
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
        // Claim only once a slot is free, so the claim lands as late as the ring allows: the
        // CTA that drains first takes the next unit. The counter is a plain atomic (exactly
        // one claimant per value); the units' inputs still come through done_as. CTA 0 zeroes
        // it after the grid barrier, which every claim precedes.
        // claim_a: the values below na are phase A units, so every claimed phase A unit is
        // taken before any phase B unit, and a CTA runs its claims in claim order.
        int* claim = p.ctr + 10;
        const int units = S.njobs * 16;
        const int xs = tail_xs<MODE>(units, G);
        const int s0 = claim_lean(MODE) ? a_dealt<MODE>(S.njobs, G) : a_static<MODE>(S.njobs, G);
        const int na = S.njobs * (kUAM / 16) - s0;
        for (int i = 0; i < kBqMax; ++i) {
            if (rg.wlap > 0) {
                const long long c0 = is_diag(MODE) ? clock64() : 0;
                mbar_wait_to(&S.wempty[rg.ws], (rg.wlap - 1) & 1, p.ctr, 16);
                if (is_diag(MODE)) {
                    twe += clock64() - c0;
                }
            }
            int m = i < kBqMax - 1 ? atomicAdd(claim, 1) : na + units + xs;
            if (m >= na + units + xs) {
                m = -1;
            }
            if constexpr (claim_lean(MODE)) {
                if (m >= na) {
                    m -= na;
                } else if (m >= 0) {
                    const int k = s0 + m;
                    m = kBqA + S.ord[k >> 5] * kUAM + (k & 31) * 16;
                }
            }
            st_release_cta(&S.bq[i], m);
            if (m < 0) {
                break;
            }
            Claim c;
            const unsigned char* base;
            if (claim_lean(MODE) && m >= kBqA) {
                const int gb = m - kBqA;
                c = {gb, 0, 16, -1};
                base = p.w13f
                       + static_cast<size_t>(S.job_e[gb >> 9] * 32 + ((gb & 511) >> 4)) * 16
                             * kUnitBytes;
            } else if (claim_lean(MODE)) {
                c = claim_of<MODE>(m, units, xs);
                const int j = S.ord[c.unit >> 4];
                base = p.w2f
                       + static_cast<size_t>(S.job_e[j] * 16 + (c.unit & 15)) * 16 * kUnitBytes;
            } else if (claim_a(MODE) && m < na) {
                const int k = s0 + m;
                c = {k, 0, 16, -1};
                base = p.w13f
                       + static_cast<size_t>(S.job_e[S.ord[k >> 5]] * 32 + (k & 31)) * 16
                             * kUnitBytes;
            } else {
                c = claim_of<MODE>(m - na, units, xs);
                const int j = S.ord[c.unit >> 4];
                base = p.w2f
                       + static_cast<size_t>(S.job_e[j] * 16 + (c.unit & 15)) * 16 * kUnitBytes;
            }
            for (int st = c.st_b; st < c.st_e; ++st) {
                const int ws = rg.ws;
                if (st > c.st_b && rg.wlap > 0) {
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

// Phase A stages come from the gathered x rows once the job's rows are all in; phase B
// stages from the SwiGLU output once the stage's 8 warpgroup units have released it.
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
                if (gather_st(MODE) && S.dg[16][7] == 0) {
                    S.dg[16][7] = gtimer();
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
        const int s0 = a_static<MODE>(S.njobs, G);
        const int na = S.njobs * (kUAM / 16) - s0;
        for (int i = 0; i < kBqMax; ++i) {
            const int m = bq_get(S, i, p.ctr);
            if (m < 0) {
                break;
            }
            // claim_a, claim_lean: a claimed phase A unit takes its x as a dealt one does, after
            // the job's rows are gathered (xready acquire, then the proxy fence), with no done_as
            // wait.
            const bool ca = claim_lean(MODE) ? m >= kBqA : claim_a(MODE) && m < na;
            Claim c;
            int j;
            if (ca) {
                if constexpr (claim_lean(MODE)) {
                    c = {m - kBqA, 0, 16, -1};
                    j = c.unit >> 9;
                } else {
                    c = {s0 + m, 0, 16, -1};
                    j = S.ord[c.unit >> 5];
                }
                spin_ge(p.xready + j, S.job_n[j], p.ctr, 32);
                fence_proxy_async_global();
            } else {
                c = claim_of<MODE>(m - na, S.njobs * 16, tail_xs<MODE>(S.njobs * 16, G));
                j = S.ord[c.unit >> 4];
            }
            const unsigned bytes = S.job_ntt[j] * 4096;
            const unsigned char* base = (ca ? p.xa : p.xb)
                                        + static_cast<size_t>(S.job_xo[j]) * 65536;
            for (int st = c.st_b; st < c.st_e; ++st) {
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
                if (!ca) {
                    if (!b_only(MODE)) {
                        spin_ge(p.done_as + j * 16 + st, 8, p.ctr, 128);
                    }
                    fence_proxy_async_global();
                }
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

// gx_st16: lane l holds bf16 pairs 4 l .. 4 l + 3 of its warp's 256-k stage; pair h lands in
// 16-byte group chunk8 = 8 (l >> 3) + 4 (l & 1) + h at word e = (l >> 1) & 3. Lane l stores group
// h = e, whose word e' it receives from lane l ^ 2 (e ^ e'). The whole warp must call this.
__device__ __forceinline__ void gx_store16(unsigned char* row, const uint4& v, int lane) {
    const unsigned w[4] = {v.x, v.y, v.z, v.w};
    const int e = (lane >> 1) & 3;
    unsigned o[4] = {w[0], w[1], w[2], w[3]};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int t = e ^ r;
        const unsigned send = t == 0 ? w[0] : t == 1 ? w[1] : t == 2 ? w[2] : w[3];
        const unsigned got = r == 0 ? send : __shfl_xor_sync(0xffffffffu, send, r << 1);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            o[k] = k == t ? got : o[k];
        }
    }
    *reinterpret_cast<uint4*>(row + (8 * (lane >> 3) + 4 * (lane & 1) + e) * 128) =
        make_uint4(o[0], o[1], o[2], o[3]);
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
        if (gather_st(MODE)) {
            S.dg[16][7] = 0;
        }
    }
    if (fix_st(MODE) && tid < 8) {
        S.dg[0][tid] = 0;
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
                const int ev[4] = {v[k].x - p.lo, v[k].y - p.lo, v[k].z - p.lo, v[k].w - p.lo};
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
    // Every thread finds the same entries, so ipf and the batched loop's start agree CTA-wide.
    uint4 xpf[kPreRows];
    int ipf[kPreRows];
    if constexpr (gx_pre(MODE)) {
        int i = cta;
#pragma unroll
        for (int c = 0; c < kPreRows; ++c) {
            while (i < E && S.eid[i] < 0) {
                i += G;
            }
            ipf[c] = i < E ? i : -1;
            if (i < E && warp < kNC) {
                xpf[c] = __ldg(reinterpret_cast<const uint4*>(p.x + static_cast<size_t>(i >> 3) * kH)
                               + tid);
            }
            i += G;
        }
    }
    if constexpr (fast_setup(MODE)) {
        position_count<kNWarpsM>(S.eid, S.wc, E);
        __syncthreads();
    }

    if (warp == kNWarpsM - 1) {
        if (lane < 9) {
            const bool given = p.cost[0][1] > 0;
            S.cost[0][lane] = given ? p.cost[0][lane] : stage_cost<MODE>(0, lane);
            S.align_b = MODE == 15 || (MODE >= 16 && MODE <= 29) || (MODE >= 31 && MODE <= 75)
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
        if (lane == 0) {
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
    if constexpr (stream_a(MODE)) {
        if (warp == 0) {
            const int b = stage_at_a_warp(S, cta, G, lane);
            const int e = stage_at_a_warp(S, cta + 1, G, lane);
            // seg_info's search: the last CTA whose range starts at or before the cut unit's
            // first stage holds that stage.
            int k = cta;
            if ((b & 15) != 0) {
                --k;
                while (k > 0 && stage_at_a_warp(S, k, G, lane) > (b & ~15)) {
                    --k;
                }
            }
            if constexpr (dfin_a(MODE)) {
                k |= dfin_flags(S, b, e, k, cta, G, lane);
            }
            if (lane == 0) {
                S.gsb[0] = b;
                S.gse[0] = e;
                S.lead_slot = k;
            }
        }
    }
    if (tid == 0) {
        // Owned phase A and claimed phase B read no stage range (seg_n and seg_at take own_n
        // and S.ord, the claims come through S.bq), so fast_setup leaves out the four
        // stage_at searches, 2.3-3.1 us on thread 0 in mode 27's stamps (39c365c34cd74559).
#pragma unroll
        for (int ph = stream_a(MODE) ? 1 : 0; ph < 2; ++ph) {
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
            const int units = a_dealt<MODE>(S.njobs, G);
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
            int ib0 = cta;
            if constexpr (gx_pre(MODE)) {
                ib0 = ipf[kPreRows - 1] < 0 ? E : ipf[kPreRows - 1] + G;
                if (warp < kNC) {
#pragma unroll
                    for (int c = 0; c < kPreRows; ++c) {
                        if (ipf[c] >= 0) {
                            const int i = ipf[c];
                            const int e = S.eid[i];
                            const int r = S.pos[i] - S.off[e];
                            const int jj = S.job0[e] + (r >> 6);
                            const int n = r & 63;
                            unsigned char* row = p.xa + static_cast<size_t>(S.job_xo[jj]) * 65536
                                                 + warp * S.job_ntt[jj] * 4096 + (n >> 3) * 4096
                                                 + (n & 7) * 16;
                            if constexpr (gx_st16(MODE)) {
                                gx_store16(row, xpf[c], lane);
                            } else {
                                const unsigned w4[4] = {xpf[c].x, xpf[c].y, xpf[c].z, xpf[c].w};
#pragma unroll
                                for (int h = 0; h < 4; ++h) {
                                    const int kk = ((4 * tid + h) & 127) * 2;
                                    const int step = 4 * (kk >> 6) + ((kk >> 2) & 3);
                                    const int chunk8 = 2 * step + ((kk & 3) >> 1);
                                    const int elem = 2 * ((kk >> 4) & 3);
                                    *reinterpret_cast<unsigned*>(row + chunk8 * 128 + elem * 2) =
                                        w4[h];
                                }
                            }
                        }
                    }
                }
            }
            for (int ib = ib0; ib < E; ib += kGB * G) {
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
                    if (gx_st16(MODE) && dst[k] != nullptr) {
                        gx_store16(dst[k], v[k], lane);
                    } else if (dst[k] != nullptr) {
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
        if (gather_st(MODE) && tid == 0) {
            S.dg[17][6] = gtimer();
        }
        fence_proxy_async_global();
        if constexpr (!gather_nofence(MODE)) {
            __threadfence();
        }
        named_sync(kBarGather, kNG);
        if (gather_st(MODE) && tid == 0) {
            S.dg[17][7] = gtimer();
        }
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
        if (fix_st(MODE)) {
            S.dg[0][6] = clock64();
        }
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
                // The rings are idle here: every bulk copy was waited on by a consumer before
                // the __syncthreads ahead of the grid barrier.
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
        if (fix_st(MODE) && tid < 8) {
            reinterpret_cast<long long*>(p.inter)[G * 16 + cta * 8 + tid] = S.dg[0][tid];
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
    moe_main_kernel<MODE><<<sms, kNTM, sizeof(SmemMain), stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}  // namespace

int64_t moe_smem_bytes() {
    return sizeof(SmemMain);
}

void moe_run(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor w13,
             torch::Tensor w2, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor nw,
             torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
             torch::Tensor done_a, torch::Tensor out, torch::Tensor w13f, torch::Tensor w2f,
             torch::Tensor xa, torch::Tensor xb, torch::Tensor fix, torch::Tensor fixcnt,
             torch::Tensor xready, torch::Tensor done_as, int64_t lo, int64_t nloc, double eps,
             int64_t variant, std::vector<int64_t> cost) {
    static int sms = 0;
    if (sms == 0) {
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device()));
    }
    const int m = x.size(0);
    TORCH_CHECK(x.size(1) == kH && topk_ids.size(1) == kTopK && nloc <= kLocalMax);
    auto stream = at::cuda::getCurrentCUDAStream();
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
    TORCH_CHECK(fixcnt.numel() >= 2 * sms * 4 && fix.numel() >= static_cast<int64_t>(2 * sms * 4) * 128 * 32);
    p.lo = static_cast<int>(lo);
    p.nloc = static_cast<int>(nloc);
    p.eps = static_cast<float>(eps);
    TORCH_CHECK(cost.empty() || cost.size() == 18);
    for (int i = 0; i < 18; ++i) {
        p.cost[i / 9][i % 9] = cost.empty() ? 0 : static_cast<int>(cost[i]);
    }
    for (int m0 = 0; m0 < m; m0 += kEmax / kTopK) {
        p.M = std::min(kEmax / kTopK, m - m0);
        p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>())
              + static_cast<size_t>(m0) * kH;
        p.topk_ids = topk_ids.data_ptr<int>() + static_cast<size_t>(m0) * kTopK;
        p.topk_w = topk_w.data_ptr<float>() + static_cast<size_t>(m0) * kTopK;
        p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>())
                + static_cast<size_t>(m0) * kH;
        switch (variant) {
//@DISPATCH@
            default: TORCH_CHECK(false, "bad variant");
        }
    }
}
"""

_SKINNY_BLOCK = r"""
namespace sk_@TAG@ {

constexpr int kNS = @NS@;
constexpr int kNW = @NW@;
constexpr int kMode = @MODE@;
// kMode 2 and 4: timing stamps (2 is mode 0's, 4 is mode 3's). kMode 3: each compute warp loads
// its codes of a unit up front and releases the ring slot itself (fence.proxy.async, then
// mbarrier.arrive on empty, count kNT, as the fat path's mode 28), and thread 0 refills the slot
// once every warp has arrived, instead of after a bar.sync of all compute warps behind the
// unit's last MMA. The A/B at 8 tokens (240b68e7df1a427f) put mode 0 3.6 us over its loads-only
// twin while thread 0 never waited for a unit, so the per-unit barrier is on the critical path.
// [Corrected 2026-10-29: those stamps skipped the first try_wait; timed in full, thread 0 does
// wait (docs/runs-moe.md, "Two stamp facts that change the attribution").]
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
                const int e = ev[k] - p.lo;
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
    moe_fused_kernel<SPLIT><<<sms, kNTL, sizeof(Smem), stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sk_@TAG@
"""


def _wgmma_code():
    """wgmma.mma_async m64nNk16 bf16 with A in registers, N = 8..64, FP32 accumulators,
    always accumulating: a runtime scale-d inside the stage loop makes ptxas serialize every
    wgmma (C7515), so a segment zeroes its accumulators before its first wgmma.fence instead."""
    out = []
    for ntt in range(1, 9):
        n = 8 * ntt
        nd = 4 * ntt
        dlist = ", ".join(f"%{i}" for i in range(nd))
        alist = ", ".join(f"%{nd + i}" for i in range(4))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(nd))
        out.append(f"""
__device__ __forceinline__ void wgmma_n{n}(float (&d)[32], const unsigned (&a)[4],
                                           uint64_t desc) {{
    asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dlist}}}, "
                 "{{{alist}}}, %{nd + 4}, p, 1, 1, 0;\\n}}\\n"
                 : {outs}
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}}""")
    cases = "\n".join(f"    if constexpr (NTT == {ntt}) {{ wgmma_n{8 * ntt}(d, a, desc); }}"
                      for ntt in range(1, 9))
    out.append(f"""
template <int NTT>
__device__ __forceinline__ void wgmma(float (&d)[32], const unsigned (&a)[4], uint64_t desc) {{
{cases}
}}

// Pins the live accumulators so the compiler neither copies them across an async wgmma nor
// moves them to local memory.
template <int NTT>
__device__ __forceinline__ void acc_fence(float (&d)[32]) {{
#pragma unroll
    for (int i = 0; i < 4 * NTT; ++i) {{
        asm volatile("" : "+f"(d[i]) :: "memory");
    }}
}}

// A wgmma keeps reading its A registers until it retires, which the compiler cannot see; a
// group's fragments are pinned after the wait that retires it, so the next group's dequant
// lands in other registers instead of behind a full pipeline drain.
__device__ __forceinline__ void a_pin(unsigned (&a)[2][4]) {{
#pragma unroll
    for (int i = 0; i < 2; ++i) {{
#pragma unroll
        for (int k = 0; k < 4; ++k) {{
            asm volatile("" : "+r"(a[i][k]) :: "memory");
        }}
    }}
}}""")
    return "\n".join(out)


# Skinny kernel variants (compute warps, ring slots, mode: 0 computes, 1 streams only), each
# a namespace of its own in the source, and fat kernel modes; a variant id indexes this table.
SKINNY_VARIANTS = ((8, 4, 0), (16, 4, 0), (8, 4, 1), (8, 3, 0))
# (8, 3, 0), sk8w3s: the 3-slot ring. At saturated HBM bandwidth a fourth copy in flight only
# lengthens each copy's issue-to-arrival time (4.7 against 3.2 us at 8 tokens), and the consumer
# waits for it; 36 of 36 paired seeds faster than sk8w4s at 1 / 8 / 16 tokens (run
# 75bca9775f6f4b0b, docs/runs-moe.md "Skinny weight stream against the public kernel").
# Flag modes 16 (unit stamps), 32 (refill ahead of the handoff) and 64 (no L2 hint) are built
# only through the bench tool's --skextra.
# Mode 2 (timing only, tools/moe_make_bench_ab.py --small appends (8, 4, 2)): mode 0 with thread
# 0's phase stamps and cycle counters in p.inter. Modes 3 and 4 (er, its stamps), 5 (sf: scale
# decode on the FMA pipe), 6 (hand: partial buffers through mbarriers), 7 (sfh: 5 and 6) and 8
# (7 with stamps) are built only through the bench tool's --skextra; see the kernel's kMode notes.
# 0: depth-2 wgmma pipeline; 6: jobs of up to 2 n8 tiles on mma.sync, the rest on wgmma; 12
# and 13: modes 6 and 0 with fence.acq_rel in place of fence.sc (fence_ar in the source); 14:
# mode 12 with two accumulator chains per n8 tile on mma.sync; 15: mode 12 with phase B CTA
# boundaries on unit boundaries (no cut phase B segments); 16: mode 14 with the same.
# Timing-only diagnostics: 1 streams the weights only (the ring's ceiling), 11 is mode 6 with
# clock stamps (_diag_report). Also in the source, not built: 7 to 10 (more jobs on mma.sync,
# flatter stage costs; all lost in submit 0400cfbdfa2845fe). Also in the source, not built: 2 dequantizes without wgmma, 3 runs wgmma on
# the raw words (run 36b5b559b0344d0f has their numbers), depth 3 (mode 5) and no readiness
# waits (mode 4), which run bab82f5f5b974594 showed do not move the fat path.
# 17 to 20 (2026-10-29, the staged-GEMM1 gate): 17 is mode 16 with stamps; 18 and 19 run phase
# A alone (timing only), 18 on mode 16's stream-K stages, 19 with every unit owned whole by one
# CTA; 20 is 19's phase A with mode 16's phase B; 21 runs phase B and the final pass alone on
# a previous launch's xb (timing only alone; VARIANT_STAGED runs 19 then 21). Modes 6 and 11 to
# 15 are not built for these checks (the build would pass the 300 s prepare cap with them); 16
# won 128 and 256 tokens over all of them in check 9a9fb08c134c4a16. 17 ran in 9332e2a49aff4ff2.
# The shipped build: 18, 19, 21 and the staged pair lost or are timing only (checks
# 9332e2a49aff4ff2, e221858b020c4dee; docs/runs-moe.md "Staged GEMM1 gate"); list them here to
# rerun the gate.
# 22 to 24 (2026-10-29, phase B tail): 22 is 20 with stamps, 23 claims phase B units at run time
# and walks both phases largest n-tile count first, 24 is 23 with stamps. 22 and 24 ran in check
# a9dbec1d98334943 (_diag_report prints them when listed here); the shipped build leaves them out.
# 25 to 27 (2026-10-29, setup): 25 is 23 with thread-0 phase stamps only, 26 is 23 with every warp
# counting the routes, a batched x gather and no stage-range searches, 27 is 26 with the stamps.
# 25 and 27 are timing only (tools/moe_make_bench_ab.py builds them).
# 28 and 29 (2026-10-29, the consumer stage): 28 is 26 with a leaner stage loop (ring constants in
# registers, the weights slot released per thread behind the first wgmma group) and a one-pass
# S.ord sort, 29 is 28 with the stamps (timing only).
# 31 to 40 (2026-10-29, phase A epilogue): 31, 33, 38 and 40 are 28, 32, 36 and 39 with phase A
# counters (timing only); 32 defers the phase A publish by one stage, 34 also drops the
# per-thread fence_gpu, 35 reloads ss and own_n from shared memory (all three lost or tied, checks
# c7db20e053254b52 and 0b684d5db0834dbc); 36 takes SiLU's reciprocal as one MUFU.RCP, 37 is 36
# without the per-thread fence_gpu (the shipped mode), 39 is 37 with paired SwiGLU stores (tied
# with 37, 473d87d03de94e96). docs/runs-moe.md, "Phase A against the public kernel".
# 41 and 42 (timing only): 37 with the phase A and phase B counters (whole waits, first try
# included) and 37 with the light stamps alone; docs/runs-moe.md, "Fat-path waits with the
# first try timed".
# 43 (candidate) and 44 (43 with 41's counters, timing only): one code group per 4 steps and the
# next stage's waits and first group under the last wgmma group; docs/runs-moe.md, "Code groups
# with the next stage hoisted".
# 45 and 46 (timing only): the staged pair (19 then 21) on 37's components, run as VARIANT_STAGED2;
# docs/runs-moe.md, "Staged pair on current components".
# 47 (candidate): 43 without the hoist (code groups only). Its pre-registered gate against 37,
# check 5f381b8b6946492d: -2.8 to -3.7 us at 256 tokens on layer, standalone and skewed routing
# (18 of 18 seeds lower), -1.0 to -2.8 at 32 and 128; docs/runs-moe.md, "Mode 47 gate".
# 48 and 49 (timing only): 47 with the light stamps, and with per-warp stage stamps (warp skew);
# 50 (lost at 32 tokens, 8a7051fa926e43e1): 47 with the sums of squares formed in the final pass.
# 51 (candidate): 50 from 64 tokens up, 47 below. Its pre-registered gate against 47, check
# c6b0e500c3a84c9a: -3.8 to -7.7 us at 128 and 256 tokens on layer, standalone and skewed routing
# (35 of 36 seeds lower), -1.0 to +0.0 at 32; docs/runs-moe.md, "Mode 51".
# 52 and 53 (lost at 32, 128 and 256 tokens, 209cf29bf91c4ea2: +3.3 to +8.5 us against 51) and 54
# (52 with the light stamps): the last phase B units claimed in K halves through the fixup;
# docs/runs-moe.md, "Phase attribution at 32 tokens against the public kernel". Not built.
# 55 and 56 (timing only): 51 with the light and x-gather stamps, 56 also with the skew stamps.
# 57 to 59 (57 and 59 lost or tied against 51, e7e75c9510994c08; 58 is 57's stamped twin): the x
# gather's rows loaded under the routing work, with and without 16-byte stores; docs/runs-moe.md,
# "The x gather or the warpgroup skew". Not built.
# 60 and 61 (tied with 51, 48b9bf9d77d146ac; 61 is 60's stamped twin): the x gather without the
# per-thread __threadfence; docs/runs-moe.md, "Gather publish without the per-thread fence". Not built.
# 62 and 63 (lost, +3.9 to +9.2 us against 51, b7e43ee6a517487a): 51 with phase A's last one or two
# rounds claimed at run time through ctr[10] ahead of the phase B units; 64 and 65 are their stamped
# twins (timing only); 66 (67 stamped) is the same code with nothing claimed, which lost as much
# (ed4591a2477041c2); docs/runs-moe.md, "Phase A's last round claimed at run time". Not built.
# 68 and 69 (70 and 71 stamped, timing only): the lean build of the same claims, 51's phase A loops
# unchanged and the claimed units in a loop of their own after them; 69 claims nothing (the null);
# docs/runs-moe.md, "Lean phase A claims". The null lost (+3.9 to +5.2 us, b2c92a71600d488b). Not built.
# 72 (73 stamped, timing only): 51 with phase A dealt as stream-K ranges and cut units finished
# through the fixup (mode 16's partition on 51's components); docs/runs-moe.md, "Stream-K phase A
# on 51's components". Not built: in 098949b28fe84bd1 72 lost to 51 by 6.6-13.5 us at 32, 128 and
# 256 tokens; its fixups took 9-11 us per CTA and widened the phase A end spread.
MAIN_MODES = (0, 16, 26, 28, 37, 47, 51)
MAIN_CANDIDATES = (0, 6, 12, 13, 14, 15, 16, 20, 23, 26, 28, 32, 34, 35, 36, 37, 39, 43, 47, 51)
VARIANT_MAIN0 = len(SKINNY_VARIANTS)


def _skinny_tag(nw, ns, mode):
    return f"{nw}_{ns}_{mode}"


def _cuda_source():
    block = _SKINNY_BLOCK
    skinny = "\n".join(block.replace("@TAG@", _skinny_tag(nw, ns, mode)).replace("@NW@", str(nw))
                       .replace("@NS@", str(ns)).replace("@MODE@", str(mode))
                       for nw, ns, mode in SKINNY_VARIANTS)
    cases = [f"            case {i}: sk_{_skinny_tag(nw, ns, mode)}::launch<1>(p, sms, stream); break;"
             for i, (nw, ns, mode) in enumerate(SKINNY_VARIANTS)]
    cases += [f"            case {VARIANT_MAIN0 + i}: launch_main<{m}>(p, sms, stream); break;"
              for i, m in enumerate(MAIN_MODES)]
    return (_CUDA_SRC.replace("//@WGMMA@", _wgmma_code()).replace("//@SKINNY@", skinny)
            .replace("//@DISPATCH@", "\n".join(cases)))


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="okc_moe_fused",
            cpp_sources=_CPP_SRC,
            cuda_sources=_cuda_source(),
            functions=["moe_run", "moe_smem_bytes"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        )
    return _EXT


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


def unit_rows(a, kind):
    """[N, ...] per-row data -> [units, h, g, ...]: unit rb holds rows (h, g) = gate 8rb + g and
    up N/2 + 8rb + g for kind 1 (W13), rows 16rb + 8h + g for kind 2 (W2)."""
    n = a.shape[0]
    if kind == 1:
        return _perm(a.reshape(2, n // 16, 8, *a.shape[1:]), (1, 0, 2, 3))
    return a.reshape(n // 16, 2, 8, *a.shape[1:])


def repack_expert(packed, sf, kind):
    """One expert's NVFP4 [N, K/2] bytes (low nibble first) and [N, K/16] E4M3 scale bytes ->
    uint8 [N/16, 36864]: per unit 32,768 bytes of codes as words [warp 8][chunk 8][lane 32]
    [k16 step 4], one word = one lane's m16n8k16 A fragment (8 codes, bits permuted per
    W4_BITPOS) for K = 512 warp + 64 chunk + 16 (lane % 4) + 4 step + j, rows g = lane / 4
    and g + 8; then 4096 bytes of scales [warp][lane][chunk][h], the block scale of row
    (h ? g + 8 : g) for the lane's 16-K block of that chunk."""
    n, k2 = packed.shape
    k = 2 * k2
    codes = _stack_last(packed & 15, packed >> 4).reshape(n, k)
    c = unit_rows(codes, kind)
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
    s = unit_rows(sf, kind).reshape(u, 2, 8, 8, 8, 4)
    s = _perm(s, (0, 3, 2, 5, 4, 1)).reshape(u, 4096)
    return _cat_last(words, s)


def repack_expert_fat(packed, sf, kind):
    """The same expert for the fat path: uint8 [N/256, 16, 36864], one 256-row unit x 256-K
    stage per entry. Codes are words [warpgroup 4][warp 4][chunk 4][lane 32][step 4]: the
    fragment of 16-row block 16 u + 4 wg + w (rows as in unit_rows) for k16 step 4 chunk + step
    at K = 256 st + 64 chunk + 16 t4 + 4 step + j (mma k 2 t4 + j for j < 2, 8 + 2 t4 + j - 2
    above), so a lane's four steps of a chunk share one 16-K scale block and its 16-byte load
    sits next to its neighbours' (no bank conflicts); then scales [warpgroup][warp][lane]
    [chunk][h], one byte per row and 16-K block. The x slab holds each stage in the same mma-k
    order (see the gather and epilogue_a)."""
    n, k2 = packed.shape
    k = 2 * k2
    codes = _stack_last(packed & 15, packed >> 4).reshape(n, k)
    c = unit_rows(codes, kind)
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
    sc = unit_rows(sf, kind).reshape(nu, 4, 4, 2, 8, 16, 4, 4)
    sc = _perm(sc, (0, 5, 1, 2, 4, 7, 6, 3)).reshape(nu * 16, 4096)
    return _cat_last(words, sc).reshape(nu, 16, 36864)


def _perm(a, dims):
    return a.permute(*dims)


def _stack_last(a, b):
    return torch.stack((a, b), dim=-1)


def _to_i64(a):
    return a.to(torch.int64)


def _to_u8_words(words):
    """int64 words in [0, 2^32) -> their little-endian bytes."""
    w = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return w.contiguous().view(torch.uint8)


def _cat_last(a, b):
    return torch.cat((a, b), dim=-1).contiguous()


def scales_ok(sf_bytes):
    """dequant() turns a scale byte into bf16 by shifting and adding to the exponent, which is
    exact only for positive normal E4M3 values (not zero, subnormal, negative or NaN)."""
    b = sf_bytes.to(torch.int32)
    exp = (b >> 3) & 15
    return bool(((b & 0x80) == 0).all() and (exp > 0).all() and (b != 0x7F).all())


def _repack(weights, start, count):
    out = {}
    for name, kind in (("w13", 1), ("w2", 2)):
        packed = weights[name][start:start + count]
        sf = weights[name + "_sf"][start:start + count].view(torch.uint8)
        if not scales_ok(sf):
            raise ValueError(f"{name} block scales outside positive normal E4M3")
        out[name] = torch.stack([repack_expert(packed[e], sf[e], kind) for e in range(count)])
        out[name + "f"] = torch.stack([repack_expert_fat(packed[e], sf[e], kind)
                                       for e in range(count)])
        out[name + "_gs"] = weights[name + "_gs"][start:start + count].float().contiguous()
    return out


# ---- The previous kernel (check a49578e5c30348bb, 1.691x): per-warp register-path mma.sync
# GEMMs with K splits, a SwiGLU reduce and an FP32 finalizer, as fallback candidates where
# the fused kernel loses. Only the configs its tuner picked at 32, 128 and 256 tokens are
# built: (kind, nt, rows, s, w, d, mode, layout, p) as in w4_gemm_kernel.
LEGACY_CONFIGS = [
    (1, 1, 8, 4, 4, 4, 5, 0, 0), (2, 1, 8, 4, 4, 4, 5, 0, 0),
    (1, 1, 8, 2, 8, 4, 0, 0, 0), (2, 2, 16, 4, 4, 4, 0, 0, 0),
    (1, 2, 16, 4, 4, 4, 5, 0, 0), (2, 2, 16, 4, 4, 4, 5, 0, 0),
]
# (GEMM1 config, GEMM2 config, finalizer cluster size), the tuned policy at 32, 128, 256.
LEGACY_POLICIES = ((0, 1, 8), (2, 3, 1), (4, 5, 1))
VARIANT_LEGACY0 = 100
# A fat variant VARIANT_MAIN0 + i run with the stage costs of the validated cost model
# (_cost_model in prepare, state["fit_cost"]) is VARIANT_FIT0 + i.
VARIANT_FIT0 = 200
# The staged pair: mode 19 (routing, x gather, GEMM1 + SwiGLU on owned units) as one launch,
# then mode 21 (GEMM2, the post-expert norm and the weighted sum) as a second.
VARIANT_STAGED = 300
# The staged pair on mode 37's components: 45 (phase A alone) then 46 (phase B and the final pass
# alone, claimed units). Timing only; never a policy candidate.
VARIANT_STAGED2 = 301
# Modes the fitted costs are offered for: the stamps time mode 6's segments.
FIT_MODES = (6, 12, 14)

_LEGACY_ROUTE_CPP = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> route_align(torch::Tensor topk_ids, int64_t start, int64_t count,
                                       int64_t block_size, int64_t max_padded,
                                       int64_t item_rows, int64_t item_rows2);
torch::Tensor finalize(torch::Tensor y, torch::Tensor scores, torch::Tensor norm_w,
                       torch::Tensor local_ids, int64_t variant, double eps);
"""

_LEGACY_ROUTE_SRC = r"""
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

// Marlin's moe_align layout over local experts only: blocks of block_size rows, experts
// ascending, pad slots hold numel. Atomic ranks order rows within an expert arbitrarily;
// Marlin computes each row independently, so the output does not depend on that order.
__global__ void __launch_bounds__(1024) route_align_kernel(
        const int* __restrict__ topk_ids, int numel, int start, int count, int bs,
        int* __restrict__ sorted_ids, int* __restrict__ expert_ids, int* __restrict__ num_post,
        int* __restrict__ local_ids, int item_rows, int item_rows2, int items2_at,
        int2* __restrict__ items, int* __restrict__ num_items) {
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

// compile_check: route_align_kernel<<<1, 64>>>(nullptr, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0);
// compile_check: finalize_row_kernel<512><<<1, 512>>>(0, 0, 0, 0, 0, 0.f);
// compile_check: launch_cluster<2>(0, 0, 0, 0, 0, 1, 0.f, 0);
// compile_check: launch_cluster<4>(0, 0, 0, 0, 0, 1, 0.f, 0);
// compile_check: launch_cluster<8>(0, 0, 0, 0, 0, 1, 0.f, 0);
}  // namespace

std::vector<torch::Tensor> route_align(torch::Tensor topk_ids, int64_t start, int64_t count,
                                       int64_t block_size, int64_t max_padded,
                                       int64_t item_rows, int64_t item_rows2) {
    const int numel = topk_ids.numel();
    TORCH_CHECK(numel <= kMaxRoutes && count <= kMaxLocal);
    auto opts = topk_ids.options();
    auto sorted_ids = torch::empty({max_padded}, opts);
    auto expert_ids = torch::empty({(max_padded + block_size - 1) / block_size}, opts);
    auto num_post = torch::empty({1}, opts);
    auto local_ids = torch::empty_like(topk_ids);
    const int64_t max1 = item_rows > 0 ? count + numel / item_rows + 1 : 1;
    const int64_t max2 = item_rows2 > 0 ? count + numel / item_rows2 + 1 : 1;
    auto items = torch::empty({max1 + max2, 2}, opts);
    auto num_items = torch::empty({2}, opts);
    int threads = ((numel + 31) / 32) * 32;
    threads = threads < 64 ? 64 : (threads > 1024 ? 1024 : threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    route_align_kernel<<<1, threads, 0, stream>>>(
        topk_ids.data_ptr<int>(), numel, static_cast<int>(start), static_cast<int>(count),
        static_cast<int>(block_size), sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
        num_post.data_ptr<int>(), local_ids.data_ptr<int>(), static_cast<int>(item_rows),
        static_cast<int>(item_rows2), static_cast<int>(max1),
        reinterpret_cast<int2*>(items.data_ptr<int>()), num_items.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {sorted_ids, expert_ids, num_post, local_ids, items.narrow(0, 0, max1),
            num_items.narrow(0, 0, 1), items.narrow(0, max1, max2), num_items.narrow(0, 1, 1)};
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

_LEGACY_W4_CPP = r"""
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

_LEGACY_W4_SRC = r"""
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


def _legacy_w4_source():
    cases = []
    checks = []
    infos = []
    for i, cfg in enumerate(LEGACY_CONFIGS):
        targs = ", ".join(str(v) for v in cfg)
        infos.append(f"        case {i}: return w4_query<{targs}>();")
        cases.append(f"        case {i}: w4_launch<{targs}>(ap, wp, sp, gp, ip, itp, np, mi, n, "
                     f"op, pdl, stream); break;")
        checks.append(
            f"// w4_compile_check: w4_launch<{targs}>(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, false, 0);")
    return (_LEGACY_W4_SRC.replace("//@W4_CASES@", "\n".join(cases))
            .replace("//@W4_INFO@", "\n".join(infos))
            .replace("//@W4_COMPILE_CHECKS@", "\n".join(checks)))


_LEGACY_EXTS = {}


def _legacy_route_ext():
    if "route" not in _LEGACY_EXTS:
        _LEGACY_EXTS["route"] = load_inline(
            name="okc_moe_legacy_route", cpp_sources=_LEGACY_ROUTE_CPP,
            cuda_sources=_LEGACY_ROUTE_SRC, functions=["route_align", "finalize"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    return _LEGACY_EXTS["route"]


def _legacy_w4_ext():
    if "w4" not in _LEGACY_EXTS:
        _LEGACY_EXTS["w4"] = load_inline(
            name="okc_moe_legacy_w4", cpp_sources=_LEGACY_W4_CPP, cuda_sources=_legacy_w4_source(),
            functions=["w4_gemm", "w4_swiglu", "w4_info", "w4_finalize"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    return _LEGACY_EXTS["w4"]


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


def w4_repack_scales(sf, kind):
    """One expert's E4M3 block-scale bytes [N, K/16] -> bytes in the order
    [unit][stage][g][k block 0-3][tile][row g or g + 8]: 16 bytes per lane group per stage."""
    n = sf.shape[0]
    kb = sf.shape[1]
    s = w4_unit_rows(sf, 1 if kind == 1 else 2)
    # [u, tile, hi8, g, stage, half, j1]
    s = s.reshape(n // 32, 2, 2, 8, kb // 4, 2, 2)
    return _perm(s, (0, 4, 3, 5, 6, 1, 2)).reshape(-1)


def _legacy_prepare(weights, start, count):
    """The previous kernel's weight layout (32-row units, 64-k stages, per-lane fragments)."""
    out = {}
    for name, kind in (("w13", 1), ("w2", 2)):
        packed = weights[name][start:start + count]
        sf = weights[name + "_sf"][start:start + count].view(torch.uint8)
        words = []
        scales = []
        for e in range(count):
            w = w4_repack_values(packed[e], kind)
            w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)
            words.append(w)
            scales.append(w4_repack_scales(sf[e], kind).contiguous())
        out[name] = torch.cat(words).contiguous()
        out[name + "_s"] = torch.cat(scales).contiguous()
        out[name + "_gs"] = weights[name + "_gs"][start:start + count].float().contiguous()
    return out


def _w4_gemm1(w4, cfg, x, sorted_ids, items, num_items, local_ids, numel, inter, pdl=False):
    """inter[route] = SwiGLU(x[token] @ W13[expert]^T) for every local route, in bf16."""
    splits = LEGACY_CONFIGS[cfg][3]
    name = "w13"
    ext = _legacy_w4_ext()
    if splits == 1:
        ext.w4_gemm(cfg, x, w4[name], w4[name + "_s"], w4["w13_gs"], sorted_ids, items,
                    num_items, numel, inter, pdl)
        return
    part = torch.empty((splits, numel, 2 * 4096), device=x.device, dtype=torch.float32)
    ext.w4_gemm(cfg, x, w4[name], w4[name + "_s"], w4["w13_gs"], sorted_ids, items, num_items,
                numel, part, pdl)
    ext.w4_swiglu(part, local_ids.view(-1), splits, inter, pdl)


def _w4_gemm2(w4, cfg, inter, sorted_ids, items, num_items, numel, pdl=False):
    """FP32 K-split partials [splits, routes, 4096] of inter[route] @ W2[expert]^T."""
    splits = LEGACY_CONFIGS[cfg][3]
    name = "w2"
    part = torch.empty((splits, numel, 4096), device=inter.device, dtype=torch.float32)
    _legacy_w4_ext().w4_gemm(cfg, inter, w4[name], w4[name + "_s"], w4["w2_gs"], sorted_ids, items,
                      num_items, numel, part, pdl)
    return part


def _run_legacy(data, state, pol):
    """route_align, W4 GEMM1 with its SwiGLU reduce, W4 GEMM2, FP32 finalizer; PDL on."""
    g1, g2, fcl = pol
    cfg = data["config"]
    start, count = cfg["local_start"], cfg["local_count"]
    x = data["hidden_states"]
    if not x.is_contiguous():
        x = x.contiguous()
    m = x.shape[0]
    numel = m * TOP_K
    ids = data["topk_ids"]
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        ids = ids.to(torch.int32).contiguous()
    tw = data["topk_weights"]
    if tw.dtype != torch.float32 or not tw.is_contiguous():
        tw = tw.to(torch.float32).contiguous()
    bsm = 16
    max_padded = numel * bsm if numel < N_EXPERTS + 1 else numel + (N_EXPERTS + 1) * (bsm - 1)
    max_padded = (max_padded + 3) // 4 * 4
    rows1 = 8 * LEGACY_CONFIGS[g1][1]
    rows2 = 8 * LEGACY_CONFIGS[g2][1]
    (sorted_ids, _, _, local_ids, items, num_items, items2,
     num_items2) = _legacy_route_ext().route_align(ids, start, count, bsm, max_padded, rows1, rows2)
    w4 = state["legacy"]
    inter = torch.empty((numel, HIDDEN), device=x.device, dtype=torch.bfloat16)
    _w4_gemm1(w4, g1, x, sorted_ids, items, num_items, local_ids, numel, inter, True)
    part = _w4_gemm2(w4, g2, inter, sorted_ids, items2, num_items2, numel, True)
    return _legacy_w4_ext().w4_finalize(part, tw, state["nw_bf16"], local_ids.view(-1),
                                        POST_NORM_EPS, True, fcl)


def _scratch(device):
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    xbytes = (EMAX // 8 + JMAX_MAIN) * 65536
    return {
        "inter": torch.zeros((EMAX, HIDDEN), dtype=torch.bfloat16, device=device),
        "ybuf": torch.zeros((EMAX, HIDDEN), dtype=torch.float32, device=device),
        "ss": torch.zeros((2, EMAX), dtype=torch.float32, device=device),
        "ctr": torch.zeros((64,), dtype=torch.int32, device=device),
        "done_a": torch.zeros((JMAX,), dtype=torch.int32, device=device),
        "xa": torch.zeros((xbytes,), dtype=torch.uint8, device=device),
        "xb": torch.zeros((xbytes,), dtype=torch.uint8, device=device),
        "fix": torch.zeros((2 * sms * 4 * 128 * 32,), dtype=torch.float32, device=device),
        "fixcnt": torch.zeros((2 * sms * 4,), dtype=torch.int32, device=device),
        "xready": torch.zeros((JMAX_MAIN,), dtype=torch.int32, device=device),
        "done_as": torch.zeros((JMAX_MAIN * 16,), dtype=torch.int32, device=device),
    }


def _run(data, state, variant=None, cost=None):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    ids = data["topk_ids"]
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        ids = ids.to(torch.int32).contiguous()
    tw = data["topk_weights"]
    if tw.dtype != torch.float32 or not tw.is_contiguous():
        tw = tw.to(torch.float32).contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=x.device)
    w = state["w"]
    s = state["scratch"]
    if variant is None:
        variant = _policy_variant(state, m)
    if cost is None:
        cost = []
    launches = [variant]
    if variant == VARIANT_STAGED and m > CHUNK_TOKENS:
        # moe_run splits M into chunks that share one xb, so the pair cannot run chunk by
        # chunk; mode 20 is the same work in one launch.
        launches = [VARIANT_MAIN0 + MAIN_MODES.index(20)]
    elif variant == VARIANT_STAGED:
        launches = [VARIANT_MAIN0 + MAIN_MODES.index(19), VARIANT_MAIN0 + MAIN_MODES.index(21)]
    elif variant == VARIANT_STAGED2 and m > CHUNK_TOKENS:
        launches = [VARIANT_MAIN0 + MAIN_MODES.index(37)]
    elif variant == VARIANT_STAGED2:
        launches = [VARIANT_MAIN0 + MAIN_MODES.index(45), VARIANT_MAIN0 + MAIN_MODES.index(46)]
    elif variant >= VARIANT_FIT0:
        cost = state["fit_cost"]
        launches = [VARIANT_MAIN0 + variant - VARIANT_FIT0]
    elif variant >= VARIANT_LEGACY0:
        return _run_legacy(data, state, LEGACY_POLICIES[variant - VARIANT_LEGACY0])
    for v in launches:
        _ext().moe_run(x, ids, tw, w["w13"], w["w2"], w["w13_gs"], w["w2_gs"], state["nw"],
                       s["inter"], s["ybuf"], s["ss"], s["ctr"], s["done_a"], out,
                       w["w13f"], w["w2f"], s["xa"], s["xb"], s["fix"], s["fixcnt"],
                       s["xready"], s["done_as"], cfg["local_start"], cfg["local_count"],
                       POST_NORM_EPS, v, cost)
    return out


def kernel_fn(data, state):
    return _run(data, state)


# Batch sizes the policy is keyed by; a batch uses the entry of the next size up.
POLICY_BS = (1, 8, 16, 32, 128, 256)
SKINNY_DEFAULT = 0
# Computing variants only: the loads-only ones time the rings and are never a policy choice.
SKINNY_CANDIDATES = tuple(i for i, v in enumerate(SKINNY_VARIANTS) if v[2] in (0, 3, 5, 6, 7))


def _policy_variant(state, m):
    key = next((b for b in POLICY_BS if b >= m), POLICY_BS[-1])
    policy = state.get("policy") or {}
    if key in policy:
        return policy[key]
    if m >= MAIN_MIN_M and state.get("main_ok"):
        return VARIANT_MAIN0
    return SKINNY_DEFAULT


def _make_inputs(bs, seed, dist="normal", device="cuda"):
    """Routing like the judge's: top-8 of uniform scores plus heavy-tailed expert popularity
    (or a few hot experts), rotated so experts 0-31 get the most routes."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, HIDDEN), generator=gen, device=device).to(torch.bfloat16)
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
        "config": {"bs": bs, "d_hidden": HIDDEN, "d_expert": HIDDEN, "n_experts": N_EXPERTS,
                   "local_start": 0, "local_count": 32, "top_k": TOP_K, "sharding": "ep8"},
    }


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant_ref(packed, sf, gs):
    """NVFP4 [N, K/2] bytes, [N, K/16] E4M3 scales, FP32 global scale -> FP32 [N, K]."""
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    vals = torch.where(codes >= 8, -grid[codes & 7], grid[codes & 7])
    n, k = vals.shape
    return (vals.view(n, k // 16, 16) * (sf.float() * gs).unsqueeze(-1)).view(n, k)


def _ref_forward(data, weights):
    """The FP32 computation the judge checks against."""
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


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


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


# The judge allows 1.5e-2 relative L2 against FP32; the Marlin path measures 5.2e-3 to 5.5e-3
# and the previous W4 kernels 4.2e-3 to 4.4e-3 on these inputs.
TEST_TOL = 8e-3


# FP32 references of the self-test inputs, shared by every variant's test in prepare and
# cleared at its end (the reference dequantizes whole experts; recomputing it per variant
# would cost about as much as the build).
_REF_CACHE = {}


def _self_test(state, weights, sizes, label, variant=None):
    """Every size against the FP32 reference on normal and skewed routings; a spin timeout in
    any launch (ctr[8]) fails the test too."""
    worst = 0.0
    for bs in sizes:
        for dist in ("normal", "skewed"):
            if (bs, dist) not in _REF_CACHE:
                data = _make_inputs(bs, 200 + bs, dist)
                _REF_CACHE[bs, dist] = (data, _ref_forward(data, weights))
            data, ref = _REF_CACHE[bs, dist]
            out = _run(data, state, variant)
            err = _rel(out, ref) if torch.isfinite(out).all() else float("inf")
            rows = (out.float() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(
                min=0.1 * ref.norm(dim=-1).mean())
            worst = max(worst, err)
            print(f"TEST {label} bs={bs} {dist}: vs fp32 {err:.2e}, "
                  f"worst row {float(rows.max()):.2e}")
    torch.cuda.synchronize()
    ctr = state["scratch"]["ctr"]
    flags = int(ctr[8].item())
    ctr[8] = 0
    print(f"TEST {label} worst rel err vs fp32 {worst:.2e}, spin flags {flags}")
    return worst < TEST_TOL and flags == 0


def _bench(state, main_ok, seeds=(11, 12)):
    """Times every variant per batch size (graph replay, cold L2, judge-like routings) and
    returns the policy: the fastest computing variant per size. Loads-only variants are
    timed for the record only. Nothing from any input is kept."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    policy = {}
    for bs in POLICY_BS:
        cands = list(SKINNY_CANDIDATES)
        diag = [i for i, v in enumerate(SKINNY_VARIANTS) if v[2] == 1]
        if main_ok and bs >= MAIN_MIN_M:
            cands += [VARIANT_MAIN0 + i for i, m in enumerate(MAIN_MODES)
                      if m in MAIN_CANDIDATES and VARIANT_MAIN0 + i not in state["main_bad"]]
            diag += [VARIANT_MAIN0 + i for i, m in enumerate(MAIN_MODES) if m not in MAIN_CANDIDATES]
            cands += [v for v in state.get("fit_ok", ())]
            if state.get("staged_ok"):
                cands.append(VARIANT_STAGED)
        if state.get("legacy_ok") and bs >= 16:
            cands += [VARIANT_LEGACY0 + i for i in range(len(LEGACY_POLICIES))]
        tot = {}
        for seed in seeds:
            data = _make_inputs(bs, 977 * seed + bs)
            for v in cands + diag:
                tot[v] = tot.get(v, 0.0) + _graph_time(lambda: _run(data, state, v), flush)
        n = len(seeds)
        line = " ".join(f"{_vname(v)}:{tot[v] / n:.1f}" for v in sorted(tot, key=tot.get))
        print(f"BENCH bs={bs} {line} us (graph replay, cold L2, mean of {n} routings)")
        policy[bs] = min(cands, key=lambda v: tot[v])
    print(f"POLICY {{{', '.join(f'{b}: {_vname(v)}' for b, v in policy.items())}}}")
    flags = int(state["scratch"]["ctr"][8].item())
    state["scratch"]["ctr"][8] = 0
    if flags:
        print(f"BENCH spin flags {flags}; policy discarded")
        return {}
    return policy


def _diag_launch(state, data, cost, flush, mode=11):
    """One cold-L2 launch of a stamped mode (11 or 17) with the given stage-cost table. Per
    CTA, S.dg (int64
    [18][8]) lands in scratch `inter`: rows 0-15 per consumer warp in columns 0-5 (cycles
    waiting for weights, waiting for x, computing, epilogue or fixup; stages; segments) and
    per class (phase * 8 + n-tiles - 1) in columns 6-7 (stages, segments); row 16 [0] the
    weights producer's cycles waiting for a free slot, [1..2] cut (partial) segments per
    phase, [3..7] globaltimer at start, consumers start, consumers done, barrier done, end;
    row 17 [0..2] the x producer's cycles waiting for a free slot, for done_as, for xready,
    [3..4] clock64 at start and end, [5..6] warp 0's cycles inside segments per phase, [7]
    globaltimer when thread 0's phase A segments end.
    Returns the stamps and the clock in cycles per us."""
    v = VARIANT_MAIN0 + MAIN_MODES.index(mode)
    inter = state["scratch"]["inter"]
    _run(data, state, v, cost=cost)
    flush.zero_()
    torch.cuda.synchronize()
    _run(data, state, v, cost=cost)
    torch.cuda.synchronize()
    g = torch.cuda.get_device_properties(0).multi_processor_count
    d = inter.reshape(-1).view(torch.int64)[: g * 144].view(g, 18, 8).double().cpu()
    ghz = (d[:, 17, 4] - d[:, 17, 3]) / (d[:, 16, 7] - d[:, 16, 3])
    return d, ghz * 1e3


def _diag_report(state, bs, seed=11, mode=11):
    """Where the fat path's time goes on the bench's routing with the compiled costs."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    d, cyc = _diag_launch(state, _make_inputs(bs, 977 * seed + bs), [], flush, mode)
    cyc_us = cyc.unsqueeze(1)
    cw = d[:, :16, :4] / cyc_us.unsqueeze(2)
    t = d[:, 16, 3:8]
    t0 = float(t[:, 0].min())
    stages = d[:, :16, 4].mean(1)

    def mm(x):
        return f"{float(x.mean()):.1f}/{float(x.max()):.1f}"

    tag = f"DIAG{mode} bs={bs} seed={seed}"
    done = (t[:, 2] - t0) / 1e3
    adone = (d[:, 17, 7] - t0) / 1e3
    print(f"{tag} critical path us: phase A done (thread 0) mean/max {mm(adone)}, consumers "
          f"done mean/max {mm(done)}, completion tail (last consumers minus mean) "
          f"{float(done.max() - done.mean()):.1f}, end {float(((t[:, 4] - t0) / 1e3).max()):.1f}; "
          f"cut segments per CTA A/B {float(d[:, 16, 1].mean()):.2f}/"
          f"{float(d[:, 16, 2].mean()):.2f}")
    print(f"{tag} clock {float(cyc.mean()) / 1e3:.2f} GHz; per CTA mean/max us: "
          f"route+gather {mm((t[:, 1] - t[:, 0]) / 1e3)}, consumers {mm((t[:, 2] - t[:, 1]) / 1e3)}, "
          f"barrier wait {mm((t[:, 3] - t[:, 2]) / 1e3)}, final {mm((t[:, 4] - t[:, 3]) / 1e3)}; "
          f"consumers done at {mm((t[:, 2] - t0) / 1e3)} after the first start, "
          f"end {mm((t[:, 4] - t0) / 1e3)}")
    print(f"{tag} segments per warp mean/max us: wait weights "
          f"{mm(cw[:, :, 0].mean(1))}, wait x {mm(cw[:, :, 1].mean(1))}, compute "
          f"{mm(cw[:, :, 2].mean(1))}, epilogue+fixup {mm(cw[:, :, 3].mean(1))}; stages "
          f"{mm(stages)}, segments {mm(d[:, :16, 5].mean(1))}")
    print(f"{tag} producers mean/max us: weights wait for a free slot "
          f"{mm(d[:, 16, 0] / cyc_us[:, 0])}, x wait for a free slot {mm(d[:, 17, 0] / cyc_us[:, 0])}, "
          f"x wait done_as {mm(d[:, 17, 1] / cyc_us[:, 0])}, x wait xready {mm(d[:, 17, 2] / cyc_us[:, 0])}")
    sa = d[:, 0:8, 6].sum(1)
    sb = d[:, 8:16, 6].sum(1)
    wda = d[:, 17, 1] / cyc_us[:, 0]
    after = done - adone
    print(f"{tag} phase B after phase A (consumers done minus own phase A done) mean/max "
          f"{mm(after)}; last phase A done {float(adone.max()):.1f}, last consumers done minus it "
          f"{float(done.max() - adone.max()):.1f}; stages per CTA A {mm(sa)} B {mm(sb)}")
    for c in torch.argsort(done, descending=True)[:4].tolist():
        print(f"{tag} slow CTA {c}: A stages {int(sa[c])} done {float(adone[c]):.1f}, B stages "
              f"{int(sb[c])} done {float(done[c]):.1f} (B span {float(after[c]):.1f}), x wait "
              f"done_as {float(wda[c]):.1f}")
    for k in sorted(set(sa.tolist())):
        sel = sa == k
        print(f"{tag} CTAs with {int(k)} A stages: {int(sel.sum())}, A done mean {float(adone[sel].mean()):.1f}, "
              f"B stages mean {float(sb[sel].mean()):.1f}, consumers done mean/max {mm(done[sel])}")



# Seeds 11 to 16 were the gate's in checks up to a9dbec1d98334943; 17 to 22 are held out from
# them, and the skewed routings (reference.py dist="skewed") are a second distribution.
GATE_RUNS = (("normal", (17, 18, 19, 20, 21, 22)), ("skewed", (11, 12, 13)))


def _gate_report(state):
    """The staged-GEMM1 gate (docs/plan-2026-10-29-steering.md, step 5) and the staged pair:
    phase A alone on mode 16's stream-K stages (18) against phase A alone on owned units (19,
    the GEMM1-only prototype); phase B and the final pass alone (21, after a full launch on the
    same inputs left xb); the staged pair (19 then 21), mode 20 and mode 16. Same process, same
    inputs, variants interleaved per seed, graph replay after an L2 flush. Also the SwiGLU
    output of 19 against 18 (xb, bf16) and the final outputs of 20 and the staged pair against
    16. Since 2026-10-29 also mode 23 (claimed phase B units) against 20 and 16; the variant
    order is reversed on every other seed so neither side always runs first."""
    names = {16: None, 18: None, 19: None, 20: None, 21: None, 23: None, 26: None, 28: None,
             37: None, 47: None, 51: None}
    vs = {m: VARIANT_MAIN0 + MAIN_MODES.index(m) for m in names if m in MAIN_MODES}
    if state.get("staged_ok"):
        vs["staged"] = VARIANT_STAGED
    ctr = state["scratch"]["ctr"]
    for m, v in list(vs.items()):
        _run(_make_inputs(128, 5), state, v)
        torch.cuda.synchronize()
        flags = int(ctr[8].item())
        if flags:
            ctr[8] = 0
            print(f"GATE {_vname(v)}: spin flags {flags}; dropped")
            vs.pop(m)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    xb = state["scratch"]["xb"]
    for dist, seeds in GATE_RUNS:
      for bs in (128, 256):
        tag = f"bs={bs}" + ("" if dist == "normal" else f" {dist}")
        tim = {m: [] for m in vs}
        xbd = []
        outd = {m: [] for m in (20, 23, 26, 28, 37, 47, 51, "staged") if m in vs}
        for si, seed in enumerate(seeds):
            data = _make_inputs(bs, 977 * seed + bs, dist)
            order = list(vs.items())
            if si % 2:
                order.reverse()
            for m, v in order:
                if m == 21 and 16 in vs:
                    _run(data, state, vs[16])
                tim[m].append(_graph_time(lambda: _run(data, state, v), flush))
            if 18 in vs and 19 in vs:
                _run(data, state, vs[18])
                ref = xb.view(torch.bfloat16).float().clone()
                _run(data, state, vs[19])
                xbd.append(_rel(xb.view(torch.bfloat16).float(), ref))
            if 16 in vs:
                ref = _run(data, state, vs[16]).float()
                for m in outd:
                    outd[m].append(_rel(_run(data, state, vs[m]).float(), ref))
        n = len(seeds)
        line = " ".join(f"{_vname(v)}:{sum(tim[m]) / n:.1f}" for m, v in vs.items())
        print(f"GATE {tag} {line} us (mean of {n} seeds {seeds})")
        for a, b in ((19, 18), (20, 16), (23, 20), (23, 16), (26, 23), (28, 26), (37, 28),
                     (47, 37), (51, 47), ("staged", 16), ("staged", 20)):
            if a in vs and b in vs:
                d = [x - y for x, y in zip(tim[a], tim[b])]
                print(f"GATE {tag} {_vname(vs[a])} minus {_vname(vs[b])} per seed "
                      + " ".join(f"{x:+.1f}" for x in d) + f" (mean {sum(d) / n:+.1f})")
        if xbd:
            print(f"GATE {tag} SwiGLU output (xb) of 19 vs 18, rel L2 per seed "
                  + " ".join(f"{x:.1e}" for x in xbd))
        for m, d in outd.items():
            print(f"GATE {tag} output of {_vname(vs[m])} vs 16, rel L2 per seed "
                  + " ".join(f"{x:.1e}" for x in d))
    flags = int(ctr[8].item())
    ctr[8] = 0
    print(f"GATE spin flags {flags}")


# The compiled stage costs of modes 0 and 6 (kCostA / kCostB in the CUDA source).
COST_DEFAULT = ((0, 112, 135, 140, 158, 173, 185, 200, 207),
                (0, 100, 104, 115, 123, 132, 144, 156, 174))
# Tables the DIAG launches partition with. The compiled table gives every CTA nearly the same
# predicted work, so on its own it leaves the terms collinear (the 2026-10-28 least-squares
# fit came out negative); flat and steep tables put other mixes of n-tile classes and cuts on
# each CTA.
DIAG_TABLES = (
    ("compiled", []),
    ("flat", [0] + [100] * 8 + [0] + [100] * 8),
    ("steep", [0] + [20 + 25 * n for n in range(1, 9)] + [0] + [20 + 25 * n for n in range(1, 9)]),
)
DIAG_SIZES = (32, 128, 256)
DIAG_TRAIN_SEEDS = (11, 12)
DIAG_HOLD_SEEDS = (13, 14)
# One term per kind of work on a CTA: per stage of each phase (the weights a stage streams
# and dequantizes), per n8 tile of each stage (the MMAs), per segment (the epilogue at a
# unit's end), per cut segment (the fixup's global adds and count), and a fixed CTA cost.
COST_TERMS = ("A stage", "A tile", "A seg", "A cut", "B stage", "B tile", "B seg", "B cut", "CTA")


def _cost_features(d):
    """Per CTA, the COST_TERMS counts from the stamps."""
    ntt = torch.arange(1, 9, dtype=torch.float64)
    cols = []
    for ph in range(2):
        st = d[:, 8 * ph: 8 * ph + 8, 6]
        cols += [st.sum(1), (st * ntt).sum(1), d[:, 8 * ph: 8 * ph + 8, 7].sum(1), d[:, 16, 1 + ph]]
    cols.append(torch.ones(d.shape[0], dtype=torch.float64))
    return torch.stack(cols, 1)


def _fixed_feature(d):
    """Per CTA, the work the compiled table assigns it (its stages times their compiled cost)."""
    c = torch.tensor([COST_DEFAULT[c // 8][c % 8 + 1] for c in range(16)], dtype=torch.float64)
    return torch.stack([(d[:, :16, 6] * c).sum(1), torch.ones(d.shape[0], dtype=torch.float64)], 1)


def _nnls(x, y):
    """Least squares with nonnegative coefficients, exact by trying every support (2^9 at
    most): the best fit over the supports whose unconstrained solution is nonnegative."""
    k = x.shape[1]
    best = (float("inf"), torch.zeros(k, dtype=torch.float64))
    for mask in range(1, 1 << k):
        idx = [i for i in range(k) if mask >> i & 1]
        sub = x[:, idx]
        if int(torch.linalg.matrix_rank(sub)) < len(idx):
            continue
        sol = torch.linalg.lstsq(sub, y.unsqueeze(1)).solution.squeeze(1)
        if bool((sol < 0).any()):
            continue
        err = float((sub @ sol - y).pow(2).sum())
        if err < best[0]:
            full = torch.zeros(k, dtype=torch.float64)
            full[idx] = sol
            best = (err, full)
    return best[1]


def _spearman(a, b):
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float((ra.pow(2).sum() * rb.pow(2).sum()).sqrt())
    return float((ra * rb).sum()) / den if den > 0 else 0.0


def _cost_model(state):
    """DIAG launches over DIAG_SIZES x {normal, skewed} x seeds x DIAG_TABLES; a nonnegative
    fit of each CTA's consumer work (warp 0's time inside segments, both paths) on
    COST_TERMS from the training seeds; validation on the held-out seeds against the
    compiled table's prediction (same nonnegative scale + offset fit). Ranks: per launch, the
    Spearman correlation of predicted work with each CTA's measured finish time; per routing
    case, whether the predicted makespans order the three tables as the measured ones do.
    Returns the 18-entry stage-cost table moe_run takes when the model ranks better on both,
    else []."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    runs = []
    for bs in DIAG_SIZES:
        for dist in ("normal", "skewed"):
            for seed in DIAG_TRAIN_SEEDS + DIAG_HOLD_SEEDS:
                data = _make_inputs(bs, 977 * seed + bs, dist)
                for name, table in DIAG_TABLES:
                    d, cyc = _diag_launch(state, data, table, flush)
                    t = d[:, 16, 3:8]
                    work = (d[:, 17, 5] + d[:, 17, 6]) / cyc
                    finish = (t[:, 2] - t[:, 0].min()) / 1e3
                    runs.append({"bs": bs, "dist": dist, "seed": seed, "table": name,
                                 "x": _cost_features(d), "f": _fixed_feature(d), "work": work,
                                 "finish": finish, "end": float((t[:, 4] - t[:, 0].min()).max()) / 1e3})
    flags = int(state["scratch"]["ctr"][8].item())
    if flags:
        state["scratch"]["ctr"][8] = 0
        print(f"COSTMODEL spin flags {flags}; no model")
        return []
    train = [r for r in runs if r["seed"] in DIAG_TRAIN_SEEDS]
    hold = [r for r in runs if r["seed"] in DIAG_HOLD_SEEDS]
    x = torch.cat([r["x"] for r in train])
    y = torch.cat([r["work"] for r in train])
    coef = _nnls(x, y)
    fcoef = _nnls(torch.cat([r["f"] for r in train]), y)
    print("COSTMODEL us per unit " + " ".join(f"{n}:{float(c):.3f}" for n, c in zip(COST_TERMS, coef))
          + f"; fixed table scale {float(fcoef[0]):.5f} offset {float(fcoef[1]):.1f}; "
          f"train {len(train)} launches, {len(y)} CTAs, work mean {float(y.mean()):.1f} us")

    def rms(rs, key, c):
        e = torch.cat([r[key] @ c - r["work"] for r in rs])
        return float(e.pow(2).mean().sqrt())

    result = {}
    for dist in ("normal", "skewed", "all"):
        rs = [r for r in hold if dist in ("all", r["dist"])]
        rho_m = sum(_spearman(r["x"] @ coef, r["finish"]) for r in rs) / len(rs)
        rho_f = sum(_spearman(r["f"] @ fcoef, r["finish"]) for r in rs) / len(rs)
        pairs = right_m = right_f = 0
        for bs in DIAG_SIZES:
            for seed in DIAG_HOLD_SEEDS:
                for dd in ("normal", "skewed"):
                    if dist not in ("all", dd):
                        continue
                    case = [r for r in rs if r["bs"] == bs and r["seed"] == seed and r["dist"] == dd]
                    for i in range(len(case)):
                        for j in range(i + 1, len(case)):
                            a, b = case[i], case[j]
                            meas = float(a["finish"].max() - b["finish"].max())
                            if abs(meas) < 1.0:
                                continue
                            pairs += 1
                            pm = float((a["x"] @ coef).max() - (b["x"] @ coef).max())
                            pf = float((a["f"] @ fcoef).max() - (b["f"] @ fcoef).max())
                            right_m += (pm > 0) == (meas > 0)
                            right_f += (pf > 0) == (meas > 0)
        result[dist] = (rho_m, rho_f, right_m, right_f, pairs)
        print(f"COSTMODEL held-out {dist}: CTA finish rank rho model {rho_m:.3f} fixed {rho_f:.3f}; "
              f"table order (makespans >= 1 us apart) model {right_m}/{pairs} fixed {right_f}/{pairs}; "
              f"work rms model {rms(rs, 'x', coef):.1f} fixed {rms(rs, 'f', fcoef):.1f} us")
    for r in runs:
        if r["seed"] in DIAG_HOLD_SEEDS:
            print(f"COSTMODEL run bs={r['bs']} {r['dist']} seed {r['seed']} {r['table']}: finish "
                  f"max/mean {float(r['finish'].max()):.1f}/{float(r['finish'].mean()):.1f} us, end "
                  f"{r['end']:.1f}; predicted work max/mean model "
                  f"{float((r['x'] @ coef).max()):.1f}/{float((r['x'] @ coef).mean()):.1f} fixed "
                  f"{float((r['f'] @ fcoef).max()):.1f}/{float((r['f'] @ fcoef).mean()):.1f}")
    # Better CTA ranking on both routings and no worse table ordering: with three tables per
    # case the ordering often ties at all pairs right.
    ok = all(result[k][0] > result[k][1] and result[k][2] >= result[k][3]
             for k in ("normal", "skewed"))
    if not ok:
        print("COSTMODEL gate: the model does not rank held-out cases better than the compiled "
              "table on both routings; it does not drive the partition")
        return []
    # A stage's cost as the partition sees it: its stage and tile terms plus one sixteenth of a
    # segment (a unit is 16 stages and ends in one epilogue). Cuts and the CTA term do not
    # depend on how a phase's stages are divided, so they stay out of the table.
    per = [[float(coef[4 * ph] + coef[4 * ph + 1] * n + coef[4 * ph + 2] / 16) for n in range(9)]
           for ph in range(2)]
    # Each phase is split by its own prefix sums, so only ratios within a phase matter; each is
    # scaled to a top cost of 200, inside the compiled table's range (stage_at's unsigned
    # products stay below 2^32).
    table = [0] * 18
    for ph in range(2):
        top = max(per[ph][1:])
        if top <= 0:
            return []
        for n in range(1, 9):
            table[9 * ph + n] = max(1, int(round(200 * per[ph][n] / top)))
    print(f"COSTMODEL gate passed; table A {table[1:9]} B {table[10:18]}")
    return table


def _vname(v):
    if v == VARIANT_STAGED:
        return "staged-own"
    if v == VARIANT_STAGED2:
        return "staged-lean"
    if v >= VARIANT_FIT0:
        return _vname(VARIANT_MAIN0 + v - VARIANT_FIT0) + "-nn"
    if v >= VARIANT_LEGACY0:
        return f"legacy{(32, 128, 256)[v - VARIANT_LEGACY0]}"
    if v >= VARIANT_MAIN0:
        m = MAIN_MODES[v - VARIANT_MAIN0]
        return {0: "main-d2", 5: "main-d3", 1: "main-loads", 4: "main-nogate", 2: "main-nowgmma",
                3: "main-nodequant", 6: "main-m2", 7: "main-m4", 8: "main-m8f", 9: "main-m2f",
                10: "main-m4f", 11: "main-m2-stamps", 12: "main-m2-ar", 13: "main-d2-ar",
                14: "main-m2-ar2", 15: "main-m2-ar-ub",
                16: "main-m2-ar2-ub", 17: "main-ub-stamps", 18: "main-ub-aonly",
                19: "main-own-aonly", 20: "main-own", 21: "main-bonly",
                22: "main-own-stamps", 23: "main-own-dynb", 24: "main-dynb-stamps",
                25: "main-dynb-light", 26: "main-dynb-fs", 27: "main-dynb-fs-light",
                28: "main-lean", 29: "main-lean-light", 31: "main-lean-pa",
                32: "main-lean-dfa", 33: "main-lean-dfa-pa", 34: "main-lean-dfa-bar",
                35: "main-lean-dfa-vol", 36: "main-lean-silu", 37: "main-lean-silu-bar",
                38: "main-lean-silu-pa", 39: "main-lean-epi", 40: "main-lean-epi-pa",
                41: "main-lean-silu-bar-pab", 42: "main-lean-silu-bar-light",
                43: "main-lean-grp", 44: "main-lean-grp-pab", 45: "main-lean-aonly",
                46: "main-lean-bonly", 47: "main-lean-grp-nohoist",
                48: "main-lean-grp-nohoist-light", 49: "main-lean-grp-nohoist-skew",
                50: "main-lean-grp-finss", 51: "main-lean-grp-finss64",
                52: "main-finss64-tail2", 53: "main-finss64-tailg",
                54: "main-finss64-tail2-light", 55: "main-finss64-light",
                56: "main-finss64-skew", 57: "main-finss64-gx", 58: "main-finss64-gx-light",
                59: "main-finss64-gxpre", 60: "main-finss64-gnf",
                61: "main-finss64-gnf-light", 62: "main-finss64-ca1", 63: "main-finss64-ca2",
                64: "main-finss64-ca1-light", 65: "main-finss64-ca2-light",
                66: "main-finss64-ca0", 67: "main-finss64-ca0-light",
                68: "main-finss64-cl1", 69: "main-finss64-cl0", 70: "main-finss64-cl1-light",
                71: "main-finss64-cl0-light", 72: "main-finss64-sk",
                73: "main-finss64-sk-light", 74: "main-finss64-skdf",
                75: "main-finss64-skdf-light"}[m]
    nw, ns, mode = SKINNY_VARIANTS[v]
    tags = ('', '-loads', '-stamps', '-er', '-er-stamps', '-sf', '-hand', '-sfh', '-sfh-stamps')
    flags = "".join(t for b, t in ((32, "-rf"), (64, "-nh"), (16, "-us")) if mode & b)
    return f"sk{nw}w{ns}s{tags[mode & 15]}{flags}"


def prepare(weights):
    device = weights["w13"].device
    t0 = time.time()
    state = {
        "w": _repack(weights, 0, MAX_LOCAL),
        "nw": (weights["post_norm_w"].float() + 1.0).contiguous(),
        "nw_bf16": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).contiguous(),
        "scratch": _scratch(device),
        "legacy": _legacy_prepare(weights, 0, MAX_LOCAL),
        "legacy_ok": False,
    }
    torch.cuda.synchronize()
    t1 = time.time()
    _ext()
    try:
        _legacy_route_ext()
        _legacy_w4_ext()
        state["legacy_ok"] = True
    except Exception as exc:
        print(f"LEGACY build failed: {type(exc).__name__}: {exc}")
    t2 = time.time()
    state["main_ok"] = False
    state["policy"] = {}
    # Every computing skinny variant on the small sizes, the fat path on the large ones.
    ok = True
    for v in SKINNY_CANDIDATES:
        ok = _self_test(state, weights, (1, 4, 8, 16), _vname(v), v) and ok
    # main_ok is mode 0's result (kernel_fn's fallback above 16 tokens); another mode that
    # fails only leaves the bench.
    main_ok = True
    state["main_bad"] = set()
    for i, m in enumerate(MAIN_MODES):
        if m in MAIN_CANDIDATES:
            v = VARIANT_MAIN0 + i
            if not _self_test(state, weights, (32, 33, 128, 257), _vname(v), v):
                state["main_bad"].add(v)
                main_ok = main_ok and m != 0
    state["main_ok"] = main_ok
    state["staged_ok"] = False
    if main_ok and 19 in MAIN_MODES and 21 in MAIN_MODES:
        state["staged_ok"] = _self_test(state, weights, (32, 33, 128, 257),
                                        _vname(VARIANT_STAGED), VARIANT_STAGED)
    state["staged2_ok"] = False
    if main_ok and 45 in MAIN_MODES and 46 in MAIN_MODES:
        state["staged2_ok"] = _self_test(state, weights, (32, 33, 128, 257),
                                         _vname(VARIANT_STAGED2), VARIANT_STAGED2)
    if state["legacy_ok"]:
        for i in range(len(LEGACY_POLICIES)):
            state["legacy_ok"] = _self_test(state, weights, (32, 33, 128, 257),
                                            _vname(VARIANT_LEGACY0 + i), VARIANT_LEGACY0 + i) \
                and state["legacy_ok"]
    if not main_ok:
        # The skinny path handles every size (split jobs of 8 tokens); it is the fallback.
        ok = _self_test(state, weights, (32, 33, 128, 257), "fallback", SKINNY_DEFAULT) and ok
    # Stamps of mode 11 (where the fat path's time goes, printed), the nonnegative cost model
    # fitted and validated on them, and its table offered to the bench on FIT_MODES after its
    # own self-test, only when it ranks held-out routings better than the compiled table.
    state["fit_ok"] = []
    for dm in (17, 22, 24):
        if main_ok and dm in MAIN_MODES:
            for bs in (128, 256):
                for seed in (11, 12):
                    try:
                        _diag_report(state, bs, seed, dm)
                    except Exception as exc:
                        print(f"DIAG{dm} bs={bs} failed: {type(exc).__name__}: {exc}")
    if main_ok:
        try:
            _gate_report(state)
        except Exception as exc:
            print(f"GATE failed: {type(exc).__name__}: {exc}")
    if main_ok and 11 in MAIN_MODES:
        for bs in DIAG_SIZES:
            try:
                _diag_report(state, bs)
            except Exception as exc:
                print(f"DIAG bs={bs} failed: {type(exc).__name__}: {exc}")
        table = []
        try:
            table = _cost_model(state)
        except Exception as exc:
            print(f"COSTMODEL failed: {type(exc).__name__}: {exc}")
        if table:
            state["fit_cost"] = table
            for i, m in enumerate(MAIN_MODES):
                v = VARIANT_FIT0 + i
                if m in FIT_MODES and VARIANT_MAIN0 + i not in state["main_bad"]:
                    if _self_test(state, weights, (32, 33, 128, 257), _vname(v), v):
                        state["fit_ok"].append(v)
    t3 = time.time()
    state["policy"] = _bench(state, main_ok)
    t4 = time.time()
    # The chosen variants once more at every test size, on the policy as kernel_fn runs it.
    ok = _self_test(state, weights, (1, 4, 8, 16, 32, 33, 128, 257), "policy") and ok
    _REF_CACHE.clear()
    print(f"PREPARE repack {t1 - t0:.1f} s, build {t2 - t1:.1f} s, self-test {t3 - t2:.1f} s, "
          f"bench {t4 - t3:.1f} s, smem {_ext().moe_smem_bytes()} B, ok {ok}, main {main_ok}")
    torch.cuda.synchronize()
    return state
