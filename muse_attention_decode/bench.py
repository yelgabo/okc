# okc: problem=muse_attention_decode
"""Gated attention decode for Muse Spark Open on one H200, three paths by shape:

  q_len 1, except full layers at batch 1: one fused kernel (attn_fused). Each CTA takes one
      request and a slice of its context and covers both KV heads, so every gathered 128-byte
      cache row serves all 16 query heads. It normalizes and rotates its queries, gathers FP8
      K/V rows through a cp.async pipeline (page entries staged 6 tiles ahead), runs Q.K and
      P.V on f16 tensor cores and keeps an online softmax. The last split also normalizes,
      rotates and writes the block's new K/V to the cache and attends them. Sliding layers
      merge the splits inside a thread-block cluster and apply the norm-gate in the same
      launch; full layers write partials and merge with the norm-gate in a second launch.
  everything else (q_len 8, full layers at batch 1, other head counts): one prep launch (norm,
      RoPE, FP8 query, cache write), SGLang's FA3 call, one norm-gate launch.
  _sglang: SGLang's own sequence, kept as the fallback if an extension fails to build.

Run log and measurements: docs/runs-attention.md."""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: q (Hq*64), keys (Hkv*64),
                               # values (Hkv*64), the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _sglang(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
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
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


_PREP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int D = 64;
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
                            int hq, int hkv, int rope) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int heads = hq + 2 * hkv;
    if (h >= heads) {
        return;
    }
    const auto* src = reinterpret_cast<const __nv_bfloat162*>(qkv + (size_t)t * (heads + hq) * D);
    float2 x = __bfloat1622float2(src[h * (D / 2) + lane]);
    if (h < hq + hkv) {
        const float r = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + EPS);
        x.x *= r;
        x.y *= r;
        if (rope) {
            float s, c;
            sincosf((float)pos[t] * inv_freq[lane], &s, &c);
            x = make_float2(x.x * c - x.y * s, x.x * s + x.y * c);
        }
    }
    if (h < hq) {
        q_out[((size_t)t * hq + h) * (D / 2) + lane] = to_fp8x2(x);
        return;
    }
    const bool is_k = h < hq + hkv;
    const int kh = is_k ? h - hq : h - hq - hkv;
    const float inv = 1.f / (is_k ? *k_scale : *v_scale);
    x.x *= inv;
    x.y *= inv;
    uint16_t* dst = is_k ? k_cache : v_cache;
    dst[((size_t)loc[t] * hkv + kh) * (D / 2) + lane] = to_fp8x2(x);
}

// One warp per (token, head): RMSNorm of the attention output times sigmoid(gate).
__global__ void gate_kernel(const __nv_bfloat16* __restrict__ o,
                            const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
                            int hq, int hkv) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (h >= hq) {
        return;
    }
    const size_t row = (size_t)t * hq + h;
    float2 x = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(o)[row * (D / 2) + lane]);
    const auto* g2 = reinterpret_cast<const __nv_bfloat162*>(
        qkv + (size_t)t * (2 * hq + 2 * hkv) * D + (hq + 2 * hkv) * D);
    const float2 g = __bfloat1622float2(g2[h * (D / 2) + lane]);
    const float r = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) / D + EPS);
    x.x = x.x * r / (1.f + __expf(-g.x));
    x.y = x.y * r / (1.f + __expf(-g.y));
    reinterpret_cast<__nv_bfloat162*>(out)[row * (D / 2) + lane] = __float22bfloat162_rn(x);
}

}  // namespace

torch::Tensor attn_prep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc,
                        torch::Tensor inv_freq, torch::Tensor k_scale, torch::Tensor v_scale,
                        torch::Tensor k_cache, torch::Tensor v_cache, int64_t hq, int64_t hkv,
                        bool rope) {
    const int tokens = qkv.size(0);
    auto q = torch::empty({tokens, hq, D}, qkv.options().dtype(torch::kFloat8_e4m3fn));
    const int heads = hq + 2 * hkv;
    dim3 grid(tokens, (heads + 7) / 8);
    prep_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), pos.data_ptr<int64_t>(),
        loc.data_ptr<int64_t>(), inv_freq.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), reinterpret_cast<uint16_t*>(q.data_ptr()),
        reinterpret_cast<uint16_t*>(k_cache.data_ptr()),
        reinterpret_cast<uint16_t*>(v_cache.data_ptr()), hq, hkv, rope ? 1 : 0);
    return q;
}

torch::Tensor attn_gate(torch::Tensor o, torch::Tensor qkv, int64_t hq, int64_t hkv) {
    const int tokens = qkv.size(0);
    auto out = torch::empty({tokens, hq * D}, qkv.options());
    dim3 grid(tokens, (hq + 7) / 8);
    gate_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), hq, hkv);
    return out;
}
"""

_PREP_CPP = """
torch::Tensor attn_prep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc,
                        torch::Tensor inv_freq, torch::Tensor k_scale, torch::Tensor v_scale,
                        torch::Tensor k_cache, torch::Tensor v_cache, int64_t hq, int64_t hkv,
                        bool rope);
torch::Tensor attn_gate(torch::Tensor o, torch::Tensor qkv, int64_t hq, int64_t hkv);
"""

_INV_FREQ = {}


def _inv_freq(theta, d, device):
    """RoPE frequencies computed exactly as SGLang's RotaryEmbedding does, so angles match."""
    key = (theta, d)
    if key not in _INV_FREQ:
        exps = torch.arange(0, d, 2, dtype=torch.float, device=device) / d
        _INV_FREQ[key] = 1.0 / (theta ** exps)
    return _INV_FREQ[key]


try:
    _PREP = load_inline(
        name="okc_attn_prep_v1",
        cpp_sources=_PREP_CPP,
        cuda_sources=_PREP_SRC,
        functions=["attn_prep", "attn_gate"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    )
    print("attn prep kernels: loaded")
except Exception as _exc:
    _PREP = None
    print(f"attn prep kernels: unavailable ({type(_exc).__name__}: {_exc})")


def _fused_prep(data):
    """One launch for QK-norm, RoPE, the FP8 cache write and the FP8 query, then FA3, then one
    launch for the norm-gate."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    theta = cfg["rope_theta"]
    inv_freq = _inv_freq(theta if theta else 1.0, d, qkv.device)
    q = _PREP.attn_prep(qkv, data["positions"], data["cache_loc"], inv_freq, data["k_scale"],
                        data["v_scale"], data["k_cache"], data["v_cache"], hq, hkv, bool(theta))
    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    if out.dtype != torch.bfloat16:
        out = out.to(torch.bfloat16)
    o = _PREP.attn_gate(out.reshape(tokens, hq * d), qkv, hq, hkv)
    return o.view(batch, length, hq * d)


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

// INSTANTIATE: template void launch<1, 1, 4, 1, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<1, 1, 4, 1, false>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 2, 1, 2, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 2, 1, 2, false>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 4, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 4, false>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<1, 1, 2, 2, false>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 4, true, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 4, false, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 2, 1, 2, false, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 2, false, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 4, true, false, true>(const Params&, int,
// INSTANTIATE:     cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 4, false, false, true>(const Params&, int,
// INSTANTIATE:     cudaStream_t);
// INSTANTIATE: template void launch<8, 2, 1, 2, false, false, true>(const Params&, int,
// INSTANTIATE:     cudaStream_t);

}  // namespace

// variant: q_len 1: 0 = 8 warps, 1 chunk each; 1 = 4 warps, 2 chunks each.
//          q_len 8: 0 = 2 m-tiles x 2 chunks per warp (4 warps); 1 = 1 m-tile x 4 chunks
//          (8 warps); 2, 3, 4 = values converted to f16 once per stage, with 1 m-tile x 4
//          chunks (8 warps), 2 m-tiles x 2 chunks (4 warps), 1 m-tile x 2 chunks (8 warps);
//          5, 6 = FP8 probabilities and values (P scaled by 2^8, as FA3 does) with 1 m-tile x
//          4 chunks (8 warps), 2 m-tiles x 2 chunks (4 warps).
torch::Tensor attn_fused(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                         torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                         torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                         int64_t batch, int64_t q_len, int64_t page_size, int64_t window,
                         bool rope, double qk_scale, int64_t splits, bool cluster,
                         int64_t variant, torch::Tensor dbg) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({batch, q_len, HQ * D}, qkv.options());
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
    p.dbg = dbg.numel() ? reinterpret_cast<long long*>(dbg.data_ptr<int64_t>()) : nullptr;
    if (q_len == 8 && (variant == 3 || variant == 4 || variant == 6)) {
        cluster = false;
    }
    torch::Tensor ws;
    if (!cluster) {
        ws = torch::empty({batch * splits * HQ * q_len * PART},
                          qkv.options().dtype(torch::kFloat32));
        p.ws = ws.data_ptr<float>();
    }
    if (q_len == 1) {
        if (cluster) {
            launch<1, 1, 4, 1, true>(p, batch, stream);
        } else if (variant == 1) {
            launch<1, 1, 2, 2, false>(p, batch, stream);
        } else {
            launch<1, 1, 4, 1, false>(p, batch, stream);
        }
    } else if (variant == 1) {
        if (cluster) {
            launch<8, 1, 1, 4, true>(p, batch, stream);
        } else {
            launch<8, 1, 1, 4, false>(p, batch, stream);
        }
    } else if (variant == 2) {
        if (cluster) {
            launch<8, 1, 1, 4, true, true>(p, batch, stream);
        } else {
            launch<8, 1, 1, 4, false, true>(p, batch, stream);
        }
    } else if (variant == 5) {
        if (cluster) {
            launch<8, 1, 1, 4, true, false, true>(p, batch, stream);
        } else {
            launch<8, 1, 1, 4, false, false, true>(p, batch, stream);
        }
    } else if (variant == 6) {
        launch<8, 2, 1, 2, false, false, true>(p, batch, stream);
    } else if (variant == 3) {
        launch<8, 2, 1, 2, false, true>(p, batch, stream);
    } else if (variant == 4) {
        launch<8, 1, 1, 2, false, true>(p, batch, stream);
    } else if (cluster) {
        launch<8, 2, 1, 2, true>(p, batch, stream);
    } else {
        launch<8, 2, 1, 2, false>(p, batch, stream);
    }
    if (!cluster) {
        combine_kernel<<<batch * q_len, HQ * 32, 0, stream>>>(p.ws, p.qkv, p.v_scale, p.out, q_len,
                                                               splits);
    }
    return out;
}
"""

_ATTN_CPP = """
torch::Tensor attn_fused(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                         torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                         torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                         int64_t batch, int64_t q_len, int64_t page_size, int64_t window,
                         bool rope, double qk_scale, int64_t splits, bool cluster,
                         int64_t variant, torch::Tensor dbg);
"""

try:
    _ATTN = load_inline(
        name="okc_attn_fused_v1",
        cpp_sources=_ATTN_CPP,
        cuda_sources=_ATTN_SRC,
        functions=["attn_fused"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    )
    print("attn fused kernel: loaded")
except Exception as _exc:
    _ATTN = None
    print(f"attn fused kernel: unavailable ({type(_exc).__name__}: {_exc})")

LOG2E = 1.4426950408889634


def _splits(cfg):
    """(splits, cluster): sliding layers merge splits inside a thread-block cluster in the same
    launch; full layers write partials and merge in a second launch."""
    batch = cfg["batch"]
    if cfg["window"]:
        return (16 if batch <= 2 else 8 if batch <= 16 else 4), True
    return (128 if batch <= 2 else max(4, 256 // batch)), False


_NO_DBG = {}


def _fused(data, splits=None, cluster=None, variant=0, dbg=None):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    qkv = data["qkv"].view(batch * length, -1)
    theta = cfg["rope_theta"]
    default_splits, default_cluster = _splits(cfg)
    splits = splits or default_splits
    if cluster is None:
        cluster = default_cluster and splits <= 16
    return _ATTN.attn_fused(
        qkv, data["page_table"], data["seq_lens"], data["cache_loc"],
        _inv_freq(theta if theta else 1.0, cfg["head_dim"], qkv.device), data["k_scale"],
        data["v_scale"], data["k_cache"], data["v_cache"], batch, length, cfg["page_size"],
        cfg["window"], bool(theta), cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, splits,
        cluster, variant, _no_dbg(qkv.device) if dbg is None else dbg)


def _no_dbg(device):
    if device not in _NO_DBG:
        _NO_DBG[device] = torch.empty(0, dtype=torch.int64, device=device)
    return _NO_DBG[device]


_WS_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

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
// CVT 0: values converted with F2FP. CVT 1: with integer ops into f16 values scaled by 2^-8,
// which the merge multiplies back.
template <int NS, int DA, int MODE, int PW, int CVT>
__global__ void __launch_bounds__(CONSUMERS + 32 * PW, 1) attn_ws_kernel(const Params p) {
    constexpr bool STAMP = MODE == 2;
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
                            *reinterpret_cast<uint4*>(dst) = f16_from_fp8x8<CVT>(w.x, w.y);
                            *reinterpret_cast<uint4*>(dst + 128) = f16_from_fp8x8<CVT>(w.z, w.w);
                        }
                    }
                    fence_async_smem();
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
                const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ws[e]));
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
        // The softmax scale (with log2 e) is folded in before the E4M3 rounding, so scores
        // come out of the f16-accumulating wgmma in log2 units.
        const float r = rsqrtf(ss / D + EPS) * p.qk_scale;
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
                    CVT == 1 ? f16raw_from_fp8x4(v8).x : f16x2_from_fp8x2(v8);
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

    // One 64-token tile: S = Q.K^T (FP8 wgmma into f16), online softmax in half2, then
    // O += P.V (f16 wgmma into f32) from the loaders' f16 values.
    auto tile = [&](const uint8_t* kt, const uint8_t* vb, int pos0, int lim, uint32_t empty_bar) {
        uint32_t s[16];
        wg_fence();
        const uint8_t* ks = kt + h * (BN * D);
        wgmma_qk<0>(s, qd0, desc(ks, 128, 512));
        wgmma_qk<1>(s, qd1, desc(ks + 256, 128, 512));
        wg_commit();
        wg_wait<0>();
        reg_fence16(s);
        lap(1);
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
        uint32_t pa[16];
        __half2 sum0 = __float2half2_rn(0.f);
        __half2 sum1 = sum0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            pa[2 * i] = ex2_h2(as_u32(__hfma2(as_h2(s[2 * i]), ks2, n0)));
            pa[2 * i + 1] = ex2_h2(as_u32(__hfma2(as_h2(s[2 * i + 1]), ks2, n1)));
            sum0 = __hadd2(sum0, as_h2(pa[2 * i]));
            sum1 = __hadd2(sum1, as_h2(pa[2 * i + 1]));
        }
        const float a0 = __low2float(alpha2);
        const float a1 = __high2float(alpha2);
        lrow[0] = lrow[0] * a0 + __low2float(sum0) + __high2float(sum0);
        lrow[1] = lrow[1] * a1 + __low2float(sum1) + __high2float(sum1);
        if (__any_sync(0xffffffffu, a0 != 1.f || a1 != 1.f)) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[4 * i] *= a0;
                o[4 * i + 1] *= a0;
                o[4 * i + 2] *= a1;
                o[4 * i + 3] *= a1;
            }
        }
        lap(2);
        reg_fence(o);
        wg_fence();
        const uint8_t* vh = vb + h * VH_HEAD;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            wgmma_pv(o, pa + 4 * k, desc(vh + k * 2048, 1024, 128));
        }
        wg_commit();
        // Leaving P.V in flight across the loop back-edge makes ptxas serialize every wgmma
        // in the kernel (C7514), so it retires here.
        wg_wait<0>();
        reg_fence(o);
        lap(3);
        if (empty_bar != 0) {
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty_bar);
            }
        }
    };

    int it = 0;
    for (; it < ntiles; ++it) {
        const int s = it % NS;
        mbar_wait(full0 + 8 * s, (it / NS) & 1, dead, p.dbg, 2);
        lap(0);
        if constexpr (MODE == 1) {
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty0 + 8 * s);
            }
            continue;
        }
        fence_async_smem();
        tile(smem + s * STAGE, smem + S::VH + (it % NV) * (HKV * VH_HEAD), lo + it * BN, hi - 1,
             empty0 + 8 * s);
    }
    if (MODE != 1 && last_split) {
        tile(newk, newv, seq, seq + L - 1, 0);
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
            dst[0] = hh ? __high2float(mrow2) : __low2float(mrow2);
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
                                  const float* __restrict__ v_scale, __nv_bfloat16* __restrict__ out,
                                  int nparts, float vmul) {
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
            const float2 v = *reinterpret_cast<const float2*>(base + (s0 + k) * stride + 2 + 2 * lane);
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

template <int NS, int DA, int MODE, int PW, int CVT>
void launch_ws(const Params& p, int batch, cudaStream_t stream) {
    auto kern = attn_ws_kernel<NS, DA, MODE, PW, CVT>;
    constexpr int bytes = Smem<NS, DA>::BYTES;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        configured = true;
    }
    kern<<<dim3(p.splits, batch), CONSUMERS + 32 * PW, bytes, stream>>>(p);
}

template <int NS, int DA, int MODE, int CVT>
void launch_pw(const Params& p, int batch, int pw, cudaStream_t stream) {
    if (pw == 8) {
        launch_ws<NS, DA, MODE, 8, CVT>(p, batch, stream);
    } else {
        launch_ws<NS, DA, MODE, 4, CVT>(p, batch, stream);
    }
}

// INSTANTIATE: template void launch_pw<7, 3, 0, 0>(const Params&, int, int, cudaStream_t);
// INSTANTIATE: template void launch_pw<6, 2, 0, 1>(const Params&, int, int, cudaStream_t);
// INSTANTIATE: template void launch_pw<7, 3, 0, 1>(const Params&, int, int, cudaStream_t);
// INSTANTIATE: template void launch_pw<8, 4, 0, 1>(const Params&, int, int, cudaStream_t);
// INSTANTIATE: template void launch_pw<7, 3, 1, 1>(const Params&, int, int, cudaStream_t);
// INSTANTIATE: template void launch_pw<7, 3, 2, 1>(const Params&, int, int, cudaStream_t);

}  // namespace

// stages: FP8 stages, 6, 7 or 8 (anything else runs 7), with values converted 2, 3 or 4 tiles
// behind their loads; mode: see attn_ws_kernel (1 and 2 with 7 stages); pw: loader warps, 4 or 8;
// cvt: value conversion, see attn_ws_kernel (0 only with 7 stages and mode 0).
// dbg: int64 [17] or empty. dbg[0] bit 0: a loader wait timed out; bit 1: a math-warp wait
// timed out. With mode 2, dbg[1..8] sum math-warp cycles per phase (wait for data, Q.K, mask
// and softmax, P.V, unused, unused, tail, prologue) and dbg[9..13] loader cycles (wait for a
// free stage, cp.async issue, page-entry loads, wait for own data, value convert and release),
// over all warps; dbg[16] is the loader warp count.
torch::Tensor attn_ws(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                      torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                      torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                      int64_t batch, int64_t page_size, int64_t window, bool rope,
                      double qk_scale, int64_t splits, int64_t stages, int64_t mode,
                      int64_t pw, int64_t cvt, torch::Tensor dbg) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({batch, L, HQ * D}, qkv.options());
    auto ws = torch::empty({batch * splits * ROWS * PART}, qkv.options().dtype(torch::kFloat32));
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
    p.ws = ws.data_ptr<float>();
    p.dbg = dbg.numel() ? reinterpret_cast<unsigned long long*>(dbg.data_ptr<int64_t>()) : nullptr;
    p.pt_stride = page_table.size(1);
    p.page_size = page_size;
    p.window = window;
    p.rope = rope ? 1 : 0;
    p.splits = splits;
    p.qk_scale = static_cast<float>(qk_scale);
    if (cvt == 0) {
        launch_pw<7, 3, 0, 0>(p, batch, pw, stream);
    } else if (mode == 1) {
        launch_pw<7, 3, 1, 1>(p, batch, pw, stream);
    } else if (mode == 2) {
        launch_pw<7, 3, 2, 1>(p, batch, pw, stream);
    } else if (stages == 6) {
        launch_pw<6, 2, 0, 1>(p, batch, pw, stream);
    } else if (stages == 8) {
        launch_pw<8, 4, 0, 1>(p, batch, pw, stream);
    } else {
        launch_pw<7, 3, 0, 1>(p, batch, pw, stream);
    }
    ws_combine_kernel<<<batch * L, HQ * 32, 0, stream>>>(p.ws, p.qkv, p.v_scale,
                                                          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                                          splits, cvt == 0 ? 1.f : 256.f);
    return out;
}
"""

_WS_CPP = """
torch::Tensor attn_ws(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                      torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                      torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                      int64_t batch, int64_t page_size, int64_t window, bool rope,
                      double qk_scale, int64_t splits, int64_t stages, int64_t mode,
                      int64_t pw, int64_t cvt, torch::Tensor dbg);
"""

try:
    _WS = load_inline(
        name="okc_attn_ws_v1",
        cpp_sources=_WS_CPP,
        cuda_sources=_WS_SRC,
        functions=["attn_ws"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    )
    print("attn ws kernel: loaded")
except Exception as _exc:
    _WS = None
    print(f"attn ws kernel: unavailable ({type(_exc).__name__}: {_exc})")


def _ws_splits(cfg):
    batch = cfg["batch"]
    if cfg["window"]:
        return 16 if batch <= 2 else 8 if batch <= 16 else 4
    return 128 if batch <= 2 else max(4, 128 // batch)


def _ws(data, splits=None, stages=7, mode=0, pw=4, cvt=1, dbg=None):
    """q_len 8 on the warp-specialized kernel (attn_ws.cu), then a merge + norm-gate launch."""
    cfg = data["config"]
    batch = cfg["batch"]
    qkv = data["qkv"].view(batch * cfg["q_len"], -1)
    theta = cfg["rope_theta"]
    return _WS.attn_ws(
        qkv, data["page_table"], data["seq_lens"], data["cache_loc"],
        _inv_freq(theta if theta else 1.0, cfg["head_dim"], qkv.device), data["k_scale"],
        data["v_scale"], data["k_cache"], data["v_cache"], batch, cfg["page_size"],
        cfg["window"], bool(theta), cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
        splits or _ws_splits(cfg), stages, mode, pw, cvt,
        _no_dbg(qkv.device) if dbg is None else dbg)


def _fused_ok(cfg):
    """Where the fused kernel measured faster than the prep + FA3 path on the judge: one-token
    decode, except full layers at batch 1 (29.4 vs 27.5 µs, jobs a15d258c37f44232 and
    c35deb23dada45d4), where 80K tokens of one request leave each CTA latency-bound."""
    if _ATTN is None or (cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]) != (16, 2, 64):
        return False
    return cfg["q_len"] == 1 and (bool(cfg["window"]) or cfg["batch"] > 2)


def _ws_ok(cfg):
    """Verify blocks on the warp-specialized kernel, except full layers at batch 1, where it
    measured 40-41 µs against 38.5 for prep + FA3 (bench, job 34929136fa004b17)."""
    if _WS is None or (cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]) != (16, 2, 64):
        return False
    return cfg["q_len"] == 8 and (bool(cfg["window"]) or cfg["batch"] > 2)


def kernel_fn(data):
    if _fused_ok(data["config"]):
        return _fused(data)
    if _ws_ok(data["config"]):
        return _ws(data)
    if _PREP is not None and data["config"]["head_dim"] == 64:
        return _fused_prep(data)
    return _sglang(data)


# Generated by tools/attn_make_bench.py: a timing sweep run at import.
_BENCH_VARIANTS = []
_BENCH_STAMPS = []
_BENCH_SHAPES = [
    (1, "sliding", 1), (1, "full", 1), (16, "sliding", 1), (16, "full", 1),
    (32, "sliding", 1), (32, "full", 1), (1, "sliding", 8), (1, "full", 8),
    (16, "sliding", 8), (16, "full", 8), (32, "sliding", 8), (32, "full", 8),
]


def _bench_inputs(batch, layer, q_len, ctx=80000, seed=1):
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    hq, hkv, d = 16, 2, 64
    u = torch.rand((batch,), generator=gen, device=dev)
    seq_lens = (ctx * (1 + 0.075 * (2 * u - 1))).long()
    longest = int(ctx * 1.075) + 1
    max_pages = longest + q_len
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=dev).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=dev)
    lens = (seq_lens + q_len).tolist()
    at = 0
    for b in range(batch):
        page_table[b, :lens[b]] = perm[at:at + lens[b]]
        at += lens[b]
    k_cache = torch.empty((total, 1, hkv, d), dtype=FP8, device=dev)
    v_cache = torch.empty((total, 1, hkv, d), dtype=FP8, device=dev)
    soft_dir = torch.randn((hkv, d), generator=gen, device=dev)
    for c in (k_cache, v_cache):
        flat = c.view(-1, hkv, d)
        for i in range(0, total, 1 << 20):
            n = min(1 << 20, total - i)
            x = torch.randn((n, hkv, d), generator=gen, device=dev)
            if c is k_cache:
                # Keys share a direction per head, as the judge's "soft" inputs do, so
                # attention spreads over many keys and FP8 rounding differences stay small.
                x = 0.8 * soft_dir + 0.6 * x
            flat[i:i + n] = x.to(FP8)
    qkv = torch.randn((batch, q_len, (2 * hq + 2 * hkv) * d), generator=gen, device=dev)
    qkv = (2 * qkv).to(torch.bfloat16)
    positions = (seq_lens[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    cache_loc = torch.gather(page_table.long(), 1, positions.view(batch, q_len)).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=dev)
    meta = None
    if layer == "full" and q_len == 1:
        from sgl_kernel.flash_attn import get_scheduler_metadata
        meta = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=hq,
            num_heads_k=hkv, headdim=d, cache_seqlens=cache_seqlens, qkv_dtype=FP8,
            cu_seqlens_q=cu, page_size=1, causal=True, num_splits=0)
    return {
        "qkv": qkv, "positions": positions, "cache_loc": cache_loc,
        "cache_seqlens": cache_seqlens, "cu_seqlens_q": cu, "scheduler_metadata": meta,
        "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "config": {
            "batch": batch, "q_len": q_len, "num_q_heads": hq, "num_kv_heads": hkv,
            "head_dim": d, "page_size": 1, "window": 2048 if layer == "sliding" else 0,
            "rope_theta": 500000.0 if layer == "sliding" else 0.0,
            "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0,
        },
    }


def _probe_empty(data):
    return data["qkv"].view(-1)[:1] + 1


def _probe_prep(data):
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"].view(tokens, -1)
    if cfg["rope_theta"]:
        cos_sin = _cos_sin(cfg["rope_theta"], d, qkv.device)
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], cos_sin)
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    store_cache(k.reshape(tokens, hkv * d).to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                data["k_cache"].view(-1, hkv * d).view(torch.uint8),
                data["v_cache"].view(-1, hkv * d).view(torch.uint8), data["cache_loc"],
                reserved_skip_index=-1)
    return q.view(tokens, hq, d).to(FP8)


def _probe_fa3(data):
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    q = data["qkv"].view(tokens, -1)[:, :16 * 64].reshape(tokens, 16, 64).to(FP8)
    window = cfg["window"]
    out = flash_attn_with_kvcache(
        q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=cfg["q_len"], softmax_scale=0.684, causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"])
    return out[0] if isinstance(out, tuple) else out


def _probe_gate(data):
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    qkv = data["qkv"].view(tokens, -1)
    return rms_norm_gate(qkv[:, :1024].contiguous(), qkv[:, 1280:], None, head_dim=64, eps=RMS_EPS,
                         has_norm=True)


def _bench_time(graph, flush, iters=40):
    events = []
    for _ in range(iters):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in events)
    return 1000 * ms[len(ms) // 2]


def _bench_graph(fn, data):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn(data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data)
    return graph, out


def _run_bench():
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    import functools
    fns = {}
    for spec in _BENCH_VARIANTS:
        name, _, args = spec.partition("@")
        keys = {"s": "splits", "c": "cluster", "v": "variant"}
        kw = {keys[k]: int(v) for k, v in (a.split("=") for a in args.split(",") if a)}
        if "cluster" in kw:
            kw["cluster"] = bool(kw["cluster"])
        fns[spec] = functools.partial(globals()[name], **kw)
    for batch, layer, q_len in _BENCH_SHAPES:
        data = _bench_inputs(batch, layer, q_len)
        ref = None
        row = []
        for name, fn in fns.items():
            try:
                graph, out = _bench_graph(fn, data)
                us = _bench_time(graph, flush)
                graph.replay()
                torch.cuda.synchronize()
                err = ""
                if ref is None:
                    ref = out.float().clone()
                elif out.shape == ref.shape:
                    e = ((out.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()
                    err = f" e={e:.1e}"
                row.append(f"{name}={us:.1f}{err}")
                del graph
            except Exception as exc:
                row.append(f"{name}=FAIL({type(exc).__name__}: {str(exc)[:200]})")
        print(f"BENCH b={batch:2d} {layer:7s} L={q_len} " + " ".join(row), flush=True)
        for spec in _BENCH_STAMPS:
            try:
                name, _, args = spec.partition("@")
                kw = {keys[k]: int(v) for k, v in (a.split("=") for a in args.split(",") if a)}
                dbg = torch.zeros(16, dtype=torch.int64, device="cuda")
                _fused(data, dbg=dbg, **kw)
                torch.cuda.synchronize()
                t = dbg.tolist()
                last = [t[8 + i] - t[8] if t[8 + i] else -1 for i in range(8)]
                first = [t[i] - t[0] if t[i] else -1 for i in range(8)]
                print(f"STAMPS b={batch:2d} {layer:7s} L={q_len} {spec} last={last} "
                      f"first={first}", flush=True)
            except Exception as exc:
                print(f"STAMPS {spec} failed: {exc}")
        del data
        torch.cuda.empty_cache()



# Generated by tools/attn_make_bench_ws.py.
_WSB_VARIANTS = ['_ref32', '_fused_prep', '_probe_fa3', '_ws', '_ws@x=0', '_ws@n=6', '_ws@n=8', '_ws@p=8', '_ws@m=1', '_ws@m=1,p=8']
_WSB_GATHER = False
_WSB_SHAPES = [
    (32, "full", 8), (16, "full", 8), (1, "full", 8), (32, "sliding", 8), (16, "sliding", 8),
    (1, "sliding", 8), (32, "full", 1),
]
_GATHER_SRC = "\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n\nnamespace {\n\n// Each CTA reads its slice of one request's rows, as an attention split would. With ROWB 64,\n// grid.z picks the KV head and the CTA reads that head's half of each row.\ntemplate <int ROWB, int U, bool PREF>\n__global__ void __launch_bounds__(256) gather_kernel(const int* __restrict__ pt, int pt_stride,\n                                                     const int* __restrict__ seq_lens,\n                                                     const uint4* __restrict__ kc,\n                                                     const uint4* __restrict__ vc, int splits,\n                                                     unsigned* out) {\n    constexpr int CH = ROWB / 16;\n    constexpr int RPI = 256 / CH;\n    const int split = blockIdx.x;\n    const int b = blockIdx.y;\n    const int head = blockIdx.z;\n    const int seq = seq_lens[b];\n    const int per = (seq + splits - 1) / splits;\n    const int lo = min(seq, split * per);\n    const int hi = min(seq, lo + per);\n    const int sub = threadIdx.x / CH;\n    const int ck = threadIdx.x % CH + (ROWB == 64 ? head * 4 : 0);\n    const int* ptb = pt + static_cast<size_t>(b) * pt_stride;\n    uint32_t x = 0;\n    int nxt[U];\n    if (PREF) {\n#pragma unroll\n        for (int u = 0; u < U; ++u) {\n            const int pos = lo + u * RPI + sub;\n            nxt[u] = pos < hi ? __ldg(ptb + pos) : -1;\n        }\n    }\n    for (int base = lo; base < hi; base += RPI * U) {\n        int e[U];\n#pragma unroll\n        for (int u = 0; u < U; ++u) {\n            if (PREF) {\n                e[u] = nxt[u];\n                const int pos = base + RPI * U + u * RPI + sub;\n                nxt[u] = pos < hi ? __ldg(ptb + pos) : -1;\n            } else {\n                const int pos = base + u * RPI + sub;\n                e[u] = pos < hi ? __ldg(ptb + pos) : -1;\n            }\n        }\n        uint4 kv[U][2];\n#pragma unroll\n        for (int u = 0; u < U; ++u) {\n            if (e[u] >= 0) {\n                kv[u][0] = __ldcg(kc + static_cast<size_t>(e[u]) * 8 + ck);\n                kv[u][1] = __ldcg(vc + static_cast<size_t>(e[u]) * 8 + ck);\n            } else {\n                kv[u][0] = kv[u][1] = make_uint4(0, 0, 0, 0);\n            }\n        }\n#pragma unroll\n        for (int u = 0; u < U; ++u) {\n            x ^= kv[u][0].x ^ kv[u][0].w ^ kv[u][1].y ^ kv[u][1].z;\n        }\n    }\n    if (x == 0x9e3779b9u) {\n        out[0] = x;\n    }\n}\n\n// One thread walks n steps. ROW_ONLY 0: page entry at position 32 i (a fresh 128-byte line each\n// step), then its K row, each address depending on the previous row's data. ROW_ONLY 1: only\n// the row load, at a hashed row.\ntemplate <int ROW_ONLY>\n__global__ void chase_kernel(const int* __restrict__ pt, const uint32_t* kc, int rows, int n,\n                             int64_t* out) {\n    uint32_t x = 0;\n    int pos = 0;\n    const long long t0 = clock64();\n    for (int i = 0; i < n; ++i) {\n        int e;\n        if (ROW_ONLY) {\n            e = static_cast<int>((static_cast<uint32_t>(i) * 2654435761u + (x & 1)) % rows);\n        } else {\n            e = *(volatile const int*)(pt + pos);\n        }\n        const uint32_t v = *(volatile const uint32_t*)(kc + static_cast<size_t>(e) * 32);\n        x += v;\n        pos += 32 + (v == 0x80000000u ? 1 : 0);\n    }\n    out[0] = clock64() - t0;\n    out[1] = x;\n}\n\n}  // namespace\n\ntemplate <int R, int UU, bool P>\nvoid run_gather(torch::Tensor& pt, torch::Tensor& seq_lens, torch::Tensor& kc, torch::Tensor& vc,\n                int splits, torch::Tensor& out) {\n    const int batch = seq_lens.size(0);\n    dim3 grid(splits, batch, R == 64 ? 2 : 1);\n    gather_kernel<R, UU, P><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(\n        pt.data_ptr<int>(), static_cast<int>(pt.size(1)), seq_lens.data_ptr<int>(),\n        reinterpret_cast<const uint4*>(kc.data_ptr()), reinterpret_cast<const uint4*>(vc.data_ptr()),\n        splits, reinterpret_cast<unsigned*>(out.data_ptr<int>()));\n}\n\ntemplate <int R, bool P>\nvoid gather_u(torch::Tensor& pt, torch::Tensor& seq_lens, torch::Tensor& kc, torch::Tensor& vc,\n              int64_t u, int splits, torch::Tensor& out) {\n    if (u == 1) {\n        run_gather<R, 1, P>(pt, seq_lens, kc, vc, splits, out);\n    } else if (u == 2) {\n        run_gather<R, 2, P>(pt, seq_lens, kc, vc, splits, out);\n    } else if (u == 4) {\n        run_gather<R, 4, P>(pt, seq_lens, kc, vc, splits, out);\n    } else {\n        run_gather<R, 8, P>(pt, seq_lens, kc, vc, splits, out);\n    }\n}\n\nvoid gather(torch::Tensor pt, torch::Tensor seq_lens, torch::Tensor kc, torch::Tensor vc,\n            int64_t rowb, int64_t u, bool pref, int64_t splits, torch::Tensor out) {\n    const int sp = static_cast<int>(splits);\n    if (rowb == 128) {\n        if (pref) {\n            gather_u<128, true>(pt, seq_lens, kc, vc, u, sp, out);\n        } else {\n            gather_u<128, false>(pt, seq_lens, kc, vc, u, sp, out);\n        }\n    } else {\n        if (pref) {\n            gather_u<64, true>(pt, seq_lens, kc, vc, u, sp, out);\n        } else {\n            gather_u<64, false>(pt, seq_lens, kc, vc, u, sp, out);\n        }\n    }\n}\n\nvoid chase(torch::Tensor pt, torch::Tensor kc, int64_t rows, int64_t n, bool row_only,\n           torch::Tensor out) {\n    auto stream = at::cuda::getCurrentCUDAStream();\n    const auto* k = reinterpret_cast<const uint32_t*>(kc.data_ptr());\n    if (row_only) {\n        chase_kernel<1><<<1, 1, 0, stream>>>(pt.data_ptr<int>(), k, rows, n, out.data_ptr<int64_t>());\n    } else {\n        chase_kernel<0><<<1, 1, 0, stream>>>(pt.data_ptr<int>(), k, rows, n, out.data_ptr<int64_t>());\n    }\n}\n"
_GATHER_CPP = '\nvoid gather(torch::Tensor pt, torch::Tensor seq_lens, torch::Tensor kc, torch::Tensor vc,\n            int64_t rowb, int64_t u, bool pref, int64_t splits, torch::Tensor out);\nvoid chase(torch::Tensor pt, torch::Tensor kc, int64_t rows, int64_t n, bool row_only,\n           torch::Tensor out);\n'


def _ref32(data):
    """FP32 attention with unrounded queries over the cache after this step's own write (done
    here the way the prep kernel does it), for the error column: graph-capturable torch."""
    cfg = data["config"]
    batch, lq = cfg["batch"], cfg["q_len"]
    hq, hkv, d = 16, 2, 64
    tokens = batch * lq
    qkv = data["qkv"].view(tokens, -1).float()
    q = qkv[:, :hq * d].view(tokens, hq, d)
    k = qkv[:, hq * d:(hq + hkv) * d].view(tokens, hkv, d)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d].view(tokens, hkv, d)
    gate = qkv[:, (hq + 2 * hkv) * d:].view(tokens, hq, d)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    pos = data["positions"]
    if cfg["rope_theta"]:
        ang = pos.float()[:, None] * _inv_freq(cfg["rope_theta"], d, qkv.device)[None, :]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]

        def rope(x):
            a, b2 = x[..., 0::2], x[..., 1::2]
            return torch.stack((a * cos - b2 * sin, a * sin + b2 * cos), -1).flatten(-2)
        q, k = rope(q), rope(k)
    kc = data["k_cache"].view(-1, hkv, d)
    vc = data["v_cache"].view(-1, hkv, d)
    loc = data["cache_loc"]
    kc[loc] = (k / data["k_scale"]).to(FP8)
    vc[loc] = (v / data["v_scale"]).to(FP8)
    pt = data["page_table"].long().clamp(min=0)
    kk = kc[pt].float() * data["k_scale"]
    vv = vc[pt].float() * data["v_scale"]
    qq = q.view(batch, lq, hkv, hq // hkv, d)
    sc = torch.einsum("bjhgd,bphd->bjhgp", qq, kk) * (cfg["sm_scale"] * cfg["qk_scale_factor"])
    p_idx = torch.arange(pt.shape[1], device=qkv.device)
    qpos = pos.view(batch, lq)
    ok = p_idx[None, None, :] <= qpos[:, :, None]
    if cfg["window"]:
        ok = ok & (p_idx[None, None, :] > qpos[:, :, None] - cfg["window"])
    sc = sc.masked_fill(~ok[:, :, None, None, :], float("-inf"))
    a = torch.einsum("bjhgp,bphd->bjhgd", sc.softmax(-1), vv).reshape(tokens, hq, d)
    a = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + RMS_EPS) * torch.sigmoid(gate)
    return a.to(torch.bfloat16).view(batch, lq, hq * d)


def _event_us(fn, flush, iters=7):
    times = []
    for _ in range(iters):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        z = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        z.record()
        torch.cuda.synchronize()
        times.append(1000 * a.elapsed_time(z))
    return sorted(times)[len(times) // 2]


def _gather_bench(flush):
    ext = load_inline(name="okc_attn_gather_bench_v1", cpp_sources=_GATHER_CPP,
                      cuda_sources=_GATHER_SRC, functions=["gather", "chase"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    empty = _event_us(lambda: out.add_(1), flush)
    print(f"GATHER empty-launch {empty:.1f} us (subtract for small reads)", flush=True)
    for batch, split_list in ((32, (4, 8, 16, 32)), (16, (8, 16, 32)), (1, (64, 132, 264))):
        data = _bench_inputs(batch, "full", 1)
        seq = data["seq_lens"]
        pt = data["page_table"]
        kc = data["k_cache"].view(torch.uint8)
        vc = data["v_cache"].view(torch.uint8)
        nbytes = int(seq.sum()) * 256
        row = []
        for rowb in (128, 64):
            for pref in (False, True):
                for splits in split_list:
                    for u in (1, 2, 4, 8):
                        us = _event_us(lambda: ext.gather(pt, seq, kc, vc, rowb, u, pref, splits,
                                                          out), flush)
                        row.append((rowb, pref, splits, u, us))
        best = min(row, key=lambda r: r[4])
        print(f"GATHER b={batch} bytes={nbytes / 1e6:.0f}MB floor@4.8TB/s="
              f"{nbytes / 4.8e6:.1f}us best={best[4]:.1f}us "
              f"({nbytes / best[4] / 1e6:.2f} TB/s rowb={best[0]} pref={int(best[1])} "
              f"splits={best[2]} u={best[3]})", flush=True)
        for rowb, pref, splits, u, us in row:
            print(f"GATHER b={batch} rowb={rowb} pref={int(pref)} splits={splits} u={u} "
                  f"{us:.1f}us {nbytes / us / 1e6:.2f}TB/s", flush=True)
        if batch == 32:
            res = torch.zeros(2, dtype=torch.int64, device="cuda")
            rows = kc.numel() // 128
            for row_only in (False, True):
                n = 2000
                flush.zero_()
                ext.chase(pt[0], kc, rows, n, row_only, res)
                torch.cuda.synchronize()
                cyc = res[0].item() / n
                print(f"GATHER latency {'row' if row_only else 'entry+row'}: {cyc:.0f} "
                      f"cycles per dependent step (cold, one thread)", flush=True)
        del data
        torch.cuda.empty_cache()


_WS_PHASES = ["wait-data", "qk", "softmax", "pv", "-", "-", "tail", "prologue"]
_WS_LOADER = ["wait-free-stage", "cp.async-issue", "page-entries", "wait-own-data",
              "v-convert+release"]


def _ws_stamps(data, cfg_splits, pw=4):
    dbg = torch.zeros(17, dtype=torch.int64, device="cuda")
    _ws(data, splits=cfg_splits, mode=2, pw=pw, dbg=dbg)
    torch.cuda.synchronize()
    t = dbg.tolist()
    cfg = data["config"]
    seq = data["seq_lens"].tolist()
    splits = cfg_splits or _ws_splits(cfg)
    start = [max(0, s - cfg["window"] + 1) if cfg["window"] else 0 for s in seq]
    tiles = sum((s - a + 63) // 64 for s, a in zip(seq, start)) + cfg["batch"] * 0
    ctas = splits * cfg["batch"]
    math = " ".join(f"{n}={t[1 + k] / 8 / max(tiles, 1):.0f}" for k, n in enumerate(_WS_PHASES))
    nload = max(t[16], 1)
    load = " ".join(f"{n}={t[9 + k] / nload / max(tiles, 1):.0f}"
                    for k, n in enumerate(_WS_LOADER))
    return (f"pw={t[16]} timeout_flags={t[0]} tiles={tiles} ctas={ctas} cycles per tile per "
            f"math warp: {math} | per loader warp per tile: {load}")


def _ws_bench():
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    try:
        if _WSB_GATHER:
            _gather_bench(flush)
    except Exception as exc:
        msg = str(exc)
        errs = [ln for ln in msg.splitlines() if "error" in ln.lower()]
        print(f"GATHER failed: {type(exc).__name__}: {msg[:300]} ... "
              f"{' | '.join(errs)[:3000]}", flush=True)
    import functools
    keys = {"s": "splits", "n": "stages", "m": "mode", "v": "variant", "c": "cluster",
            "p": "pw", "x": "cvt"}
    fns = {}
    for spec in _WSB_VARIANTS:
        name, _, args = spec.partition("@")
        kw = {keys[k]: int(v) for k, v in (a.split("=") for a in args.split(",") if a)}
        if "cluster" in kw:
            kw["cluster"] = bool(kw["cluster"])
        fns[spec] = functools.partial(globals()[name], **kw)
    for batch, layer, q_len in _WSB_SHAPES:
        data = _bench_inputs(batch, layer, q_len)
        ref = None
        row = []
        for name, fn in fns.items():
            if q_len == 1 and name.startswith("_ws"):
                continue
            try:
                graph, out = _bench_graph(fn, data)
                us = _bench_time(graph, flush)
                graph.replay()
                torch.cuda.synchronize()
                err = ""
                if ref is None:
                    ref = out.float().clone()
                elif out.shape == ref.shape:
                    e = ((out.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()
                    err = f" e={e:.1e}"
                row.append(f"{name}={us:.1f}{err}")
                del graph
            except Exception as exc:
                row.append(f"{name}=FAIL({type(exc).__name__}: {str(exc)[:200]})")
        print(f"BENCH b={batch:2d} {layer:7s} L={q_len} " + " ".join(row), flush=True)
        if q_len == 8 and _WS is not None:
            for pw in (4, 8):
                try:
                    print(f"STAMPS b={batch:2d} {layer:7s} L={q_len} "
                          + _ws_stamps(data, None, pw), flush=True)
                except Exception as exc:
                    print(f"STAMPS failed: {type(exc).__name__}: {exc}", flush=True)
        del data
        torch.cuda.empty_cache()


import os as _os

if not _os.environ.get("ATTN_BENCH_DONE"):
    _os.environ["ATTN_BENCH_DONE"] = "1"
    try:
        _ws_bench()
    except Exception as _exc:
        print(f"BENCH failed: {type(_exc).__name__}: {_exc}")
