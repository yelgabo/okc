# okc: problem=muse_attention_decode
"""SGLang's attention for Muse Spark Open on H200 (FA3 backend, FP8 E4M3 KV cache), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: the kernels between the QKV and output projections
(MuseSpark1xAttention.gated_attention), in order:

  FusedQKNorm (Triton)                 per-head RMSNorm of q and k from the fused projection
                                       output, with interleaved RoPE fused in on sliding layers
  set_kv_buffer                        k and v cast to E4M3 (divided by the cache scales when
                                       set), stored at the step's cache rows (store_cache)
  q.to(float8_e4m3fn)                  as prepare_paged_mha_query does for an FP8 cache
  flash_attn_with_kvcache (FA3)        paged, causal, window (2047, 0) on sliding layers,
                                       5.473/8 softmax scale
  rms_norm_gate (JIT CUDA)             per-head RMSNorm x sigmoid(gate)

Differs from SGLang: the step's positions, cache rows, FA3 lengths and FA3 scheduler metadata
come in with the inputs (SGLang builds them once per step for all layers)."""

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
__global__ void gate_kernel(const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ qkv,
                            __nv_bfloat16* __restrict__ out, int hq, int hkv) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (h >= hq) {
        return;
    }
    const size_t row = (size_t)t * hq + h;
    float2 x = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(o)[row * (D / 2) + lane]);
    const auto* g2 = reinterpret_cast<const __nv_bfloat162*>(qkv + (size_t)t * (2 * hq + 2 * hkv) * D
                                                              + (hq + 2 * hkv) * D);
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
        _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
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

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                          uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// Value tile converted to f16 once per stage when VF16: [token][128 dims] rows of 256 bytes,
// 16-byte chunks XOR-swizzled by token so ldmatrix row reads hit distinct banks.
constexpr int VH_BYTES = BN * ROW_BYTES * 2;

__device__ __forceinline__ int vh_chunk(int chunk, int tok) {
    return chunk ^ (tok & 7);
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

// Merges n partials (m, l, o) of one row with one warp, lane k reading partial k first so the
// loads overlap, then applies the norm-gate and writes the row. get(s) points at partial s.
template <typename Get>
__device__ __forceinline__ void merge_row(Get get, int n, int lane, float v_scale,
                                          const __nv_bfloat16* gate, __nv_bfloat16* out) {
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
    const float2 gt = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(gate)[lane]);
    acc.x = acc.x * r / (1.f + __expf(-gt.x));
    acc.y = acc.y * r / (1.f + __expf(-gt.y));
    reinterpret_cast<__nv_bfloat162*>(out)[lane] = __float22bfloat162_rn(acc);
}

template <int L, int MT, int TP, int KC, bool CLUSTER, bool VF16, bool PV8>
__global__ void __launch_bounds__(Cfg<L, MT, TP, KC>::THREADS)
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

    // Query fragments: rows g and g + 8 of each tile, dims 16c..16c+15, normalized, RoPE'd and
    // rounded to E4M3 as FA3's path does. Key dims are read in the same permuted order, so the
    // dot products are unchanged.
    uint4 qf[MT][2];
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
            const uint4* src = reinterpret_cast<const uint4*>(
                p.qkv + static_cast<size_t>(b * L + j) * QKV_STRIDE + qh * D + 16 * c);
            float x[16];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const uint4 w = src[v];
                const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv[e]));
                    x[v * 8 + 2 * e] = f.x;
                    x[v * 8 + 2 * e + 1] = f.y;
                }
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
                    sincosf(static_cast<float>(pos) * p.inv_freq[8 * c + i], &sn, &cs);
                    const float a = x[2 * i];
                    const float bb = x[2 * i + 1];
                    x[2 * i] = a * cs - bb * sn;
                    x[2 * i + 1] = a * sn + bb * cs;
                }
            }
            qf[mt][half] = make_uint4(fp8x4(x[0], x[1], x[2], x[3]), fp8x4(x[4], x[5], x[6], x[7]),
                                      fp8x4(x[8], x[9], x[10], x[11]),
                                      fp8x4(x[12], x[13], x[14], x[15]));
        }
    }
    const float qk_scale = p.qk_scale * *p.k_scale;
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
    uint8_t* vh = smem + SMEM_TOTAL;
    // All threads convert a stage's FP8 values to f16 once, instead of each warp converting
    // the fragments it needs.
    auto convert_v = [&](const uint8_t* st) {
        const uint8_t* vs = st + BN * ROW_BYTES;
        for (int i = tid; i < BN * 8; i += C::THREADS) {
            const int t = i >> 3;
            const int ck = i & 7;
            const uint4 w = *reinterpret_cast<const uint4*>(vs + t * ROW_BYTES + v_chunk(ck, t) * 16);
            const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
            uint32_t h[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                h[2 * e] = f16x2_from_fp8x2(wv[e] & 0xffffu);
                h[2 * e + 1] = f16x2_from_fp8x2(wv[e] >> 16);
            }
            uint8_t* row = vh + t * (2 * ROW_BYTES);
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
            const int t = ch0 * 16 + sub * 8 + g;
            const uint4 kf = *reinterpret_cast<const uint4*>(
                ks + t * ROW_BYTES + k_chunk(head * 4 + c, t) * 16);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float* acc = s[mt][sub];
                acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
                mma_fp8(acc, qf[mt][0].x, qf[mt][1].x, qf[mt][0].y, qf[mt][1].y, kf.x, kf.y);
                mma_fp8(acc, qf[mt][0].z, qf[mt][1].z, qf[mt][0].w, qf[mt][1].w, kf.z, kf.w);
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
            // The top-of-loop barrier keeps this from overwriting values still being read.
            convert_v(st);
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

    // The last split also attends the block's own new keys and values, which it normalizes,
    // rotates, rounds to E4M3, writes to the cache and stages in smem.
    if (split == p.splits - 1) {
        uint8_t* st = smem;
        for (int i = tid; i < STAGE_BYTES / 16; i += C::THREADS) {
            reinterpret_cast<uint4*>(st)[i] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        const float inv_k = 1.f / *p.k_scale;
        const float inv_v = 1.f / *p.v_scale;
        for (int task = warp; task < L * HKV * 2; task += C::WARPS) {
            const int j = task / (HKV * 2);
            const int kh = (task / 2) % HKV;
            const bool is_k = (task & 1) == 0;
            const size_t tok = static_cast<size_t>(b) * L + j;
            const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
                p.qkv + tok * QKV_STRIDE + (HQ + (is_k ? 0 : HKV) + kh) * D);
            float2 x = __bfloat1622float2(src[lane]);
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
            const int byte = kh * D + 2 * lane;
            uint8_t* dst = is_k ? p.k_cache : p.v_cache;
            *reinterpret_cast<uint16_t*>(dst + p.cache_loc[tok] * ROW_BYTES + byte) = v8;
            const int sw = is_k ? k_chunk(byte >> 4, j) : v_chunk(byte >> 4, j);
            *reinterpret_cast<uint16_t*>(st + (is_k ? 0 : BN * ROW_BYTES) + j * ROW_BYTES
                                         + sw * 16 + (byte & 15)) = v8;
        }
        __syncthreads();
        if constexpr (VF16) {
            convert_v(st);
            __syncthreads();
        }
        if (tp == 0) {
            chunk(st, 0, seq, seq + L - 1);
        }
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
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        stamp(6);
        const float v_scale = *p.v_scale;
        const int nwarps = p.splits * C::WARPS;
        for (int row = split * C::WARPS + warp; row < C::ROWS; row += nwarps) {
            const int j = row / HQ;
            const int qh = row % HQ;
            const size_t tok = static_cast<size_t>(b) * L + j;
            merge_row([&](int s) { return cluster.map_shared_rank(part, s) + row * PART; },
                      p.splits, lane, v_scale, p.qkv + tok * QKV_STRIDE + (HQ + 2 * HKV + qh) * D,
                      p.out + tok * HQ * D + qh * D);
        }
        cluster.sync();
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
              qkv + static_cast<size_t>(tok) * QKV_STRIDE + (HQ + 2 * HKV + qh) * D,
              out + static_cast<size_t>(tok) * HQ * D + qh * D);
}

template <int L, int MT, int TP, int KC, bool CLUSTER, bool VF16 = false, bool PV8 = false>
void launch(const Params& p, int batch, cudaStream_t stream) {
    using C = Cfg<L, MT, TP, KC>;
    auto kern = attn_kernel<L, MT, TP, KC, CLUSTER, VF16, PV8>;
    constexpr int smem_bytes = SMEM_TOTAL + (VF16 ? VH_BYTES : 0);
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
// INSTANTIATE: template void launch<8, 1, 1, 4, true, false, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 1, 1, 4, false, false, true>(const Params&, int, cudaStream_t);
// INSTANTIATE: template void launch<8, 2, 1, 2, false, false, true>(const Params&, int, cudaStream_t);

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
        ws = torch::empty({batch * splits * HQ * q_len * PART}, qkv.options().dtype(torch::kFloat32));
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


def _fused_ok(cfg):
    """Where the fused kernel measured faster than the prep + FA3 path (job 4b52fbd2aaf24247):
    one-token decode, except full layers at batch 1, where 80K tokens over too few requests
    leave each CTA latency-bound."""
    if _ATTN is None or (cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]) != (16, 2, 64):
        return False
    if cfg["q_len"] != 1:
        return False
    return bool(cfg["window"]) or cfg["batch"] > 2


def prepare(*args, **kwargs):
    print(f"prepare called with {len(args)} args, keys {list(kwargs)}")
    return None


def kernel_fn(data, state=None):
    if _fused_ok(data["config"]):
        return _fused(data)
    if _PREP is not None and data["config"]["head_dim"] == 64:
        return _fused_prep(data)
    return _sglang(data)
