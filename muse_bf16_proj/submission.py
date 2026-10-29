# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open at 1 to 32 tokens, cuBLAS above.

Design (docs/proj-port-guide.md has the full host interface and formats):

- One skinny tensor-core kernel (sk) for m <= 32: x is the mma.m16n8k16 A operand (16 tokens
  per m-tile), the weights are the B operand in 8-row tiles. Each CTA owns R rows over the
  whole K; K is split only across the warps of the CTA and summed through the idle ring at the
  end, so there is no cross-CTA reduction. Weights arrive by one cp.async.bulk per stage from a
  stage-major stream, x by one 3-D TMA box per stage, both on one mbarrier.
- Weights are stored losslessly in 12 bits: the raw low byte plus a 4-bit code (sign and a
  3-bit offset of the top seven exponent bits into an 8-value window chosen per matrix).
  Weights outside the window decode with a clamped exponent and are corrected exactly from a
  per-CTA table by dedicated escape warps while the stream runs.
- The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation on the same kernel, a
  3-CTA cluster summing its K pieces over distributed shared memory.
- m = 1 has CUDA-core GEMV candidates (12-bit cgemv, FP32 rgemv, BF16 sgemv).
- prepare() self-tests every candidate against FP64 and then times the survivors by CUDA-graph
  replay from a cold L2; kernel_fn runs the pick for the shape, cuBLAS where nothing won.
"""

import statistics
import time
from typing import TypedDict

import torch
import torch.nn.functional as F

try:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
except ImportError:  # local tools import this file without sglang
    _bf16_gemm_dispatch_impl = None
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


UNPOOL_SLICE = 1024
BF16_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
# The 12-bit code covers 8 consecutive values of the top seven exponent bits; a matrix whose
# escapes exceed 1/MAX_ESCAPE_DENOM of its weights stays raw BF16.
EXP_WINDOW = 8
MAX_ESCAPE_DENOM = 200
# Rows are dealt to CTAs in groups of this size so that escapes balance across row blocks.
BALANCE_GROUP = 4
BALANCED_LAYERS = ("qkv_gate",)
SELFTEST_TOL = {"bf16": 5e-3, "router": 5e-5}
PROBE_TOL = 5e-4
TUNE_REPS = 24
FLUSH_BYTES = 256 << 20


# --------------------------------------------------------------------------------------------
# Skinny kernel configs: name -> (MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL).
# R = NWR * NT * 8 rows per CTA, NW = NWR * WSK consumer warps, UPS = WSK * UPW 64-k units per
# stage. TOKR is the token rows reserved per unit in the ring (8, 16 or 32; MT = 16-token
# m-tiles). The geometry follows the public design memo's per-shape table; NSTAGE is capped by
# the shared-memory budget at build time.
# --------------------------------------------------------------------------------------------
SK_GEOM = {
    "qkv8": (1, 8, 3, 1, 4, 2, 5),
    "qkv16": (1, 16, 3, 1, 4, 2, 5),
    "qkv32": (2, 32, 3, 1, 4, 2, 4),
    "o8": (1, 8, 4, 1, 4, 1, 4),
    "o16": (1, 16, 2, 2, 1, 4, 3),
    "o32": (2, 32, 2, 2, 2, 2, 3),
    "pool8": (1, 8, 1, 2, 2, 4, 3),
    "pool16": (1, 16, 2, 2, 2, 4, 4),
    "pool32": (2, 32, 2, 2, 2, 2, 5),
    "slice8": (1, 8, 1, 1, 8, 2, 4),
    "unp16": (1, 16, 2, 2, 2, 2, 3),
    "unp32": (2, 32, 4, 2, 2, 2, 3),
    # An optional eighth field CL splits K across a cluster of CL CTAs that share one row block
    # (DSMEM reduction into rank 0). At 32 tokens each CTA then streams 1/CL of x for twice the
    # rows: the per-CTA x bytes (512 KB for the pooler, more than its 384 KB of weights) halve.
    "pool32c": (2, 32, 4, 2, 2, 2, 5, 2),
    "pool32cw": (2, 32, 4, 2, 4, 1, 5, 2),
    "unp32c": (2, 32, 8, 2, 2, 1, 4, 2),
}
ROUTER_GEOM = {
    "router8": (1, 8, 1, 1, 8, 1, 6),
    "router16": (1, 16, 1, 1, 8, 1, 5),
    "router32": (2, 32, 1, 1, 8, 1, 4),
}
# Pooler geometries that also get a fused pooler + router instantiation (name + "f").
FUSED_GEOM = ("pool8", "pool16")
# Which geometry each (layer, token bucket) uses; buckets are 8 (m <= 8), 16 (9..16), 32 (17..32).
SK_USE = {
    ("qkv_gate", 8): "qkv8", ("qkv_gate", 16): "qkv16", ("qkv_gate", 32): "qkv32",
    ("o", 8): "o8", ("o", 16): "o16", ("o", 32): "o32",
    ("pooler", 8): "pool8", ("pooler", 16): "pool16", ("pooler", 32): "pool32",
    ("unpooler", 8): "slice8", ("unpooler", 16): "unp16", ("unpooler", 32): "unp32",
    ("router", 8): "router8", ("router", 16): "router16", ("router", 32): "router32",
}
# tools/make_bench.py adds alternative geometries here; prepare() times them beside the picks.
SK_EXTRA = {("pooler", 32): ["pool32c", "pool32cw"], ("unpooler", 32): ["unp32c"]}
# Diagnostic timings printed by prepare(): (layer, ((config, bucket, token counts), ...)).
SK_SWEEP = (
    ("pooler", (("pool32", 32, (1, 8, 16, 24, 32)), ("pool32raw", 32, (16, 32)), ("pool16", 16, (16,)),
                ("pool32c", 32, (16, 32)), ("pool32craw", 32, (16, 32)), ("pool32cw", 32, (32,)))),
    ("unpooler", (("unp32", 32, (16, 32)), ("unp16", 16, (16,)), ("unp32c", 32, (16, 32)))),
    ("qkv_gate", (("qkv32", 32, (1, 16, 32)), ("qkv32raw", 32, (16, 32)), ("qkv16", 16, (16,)))),
)
SMEM_BUDGET = 227 * 1024
SMEM_PER_SM = 228 * 1024


def _sk_configs():
    """Every kernel instantiation: (name, MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, out, fuse)."""
    cfgs = []
    for name, geom in SK_GEOM.items():
        mt, tokr, nt, nwr, wsk, upw, nstage = geom[:7]
        cl = geom[7] if len(geom) > 7 else 1
        for comp in (True, False):
            cname = name if comp else name + "raw"
            cfgs.append((cname, mt, tokr, nt, nwr, wsk, upw,
                         _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, comp, False, cl), comp, False, 1, cl,
                         "bf16", False))
    for name, (mt, tokr, nt, nwr, wsk, upw, nstage) in ROUTER_GEOM.items():
        cfgs.append((name, mt, tokr, nt, nwr, wsk, upw,
                     _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, False, True, 3), False, True, 2, 3, "f32",
                     False))
    for name in FUSED_GEOM:
        mt, tokr, nt, nwr, wsk, upw, nstage = SK_GEOM[name][:7]
        cfgs.append((name + "f", mt, tokr, nt, nwr, wsk, upw,
                     _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, True, False, 1, True), True, False, 1, 1,
                     "bf16", True))
    return cfgs


def _sk_smem(mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, cl, fuse=False, pmax=64):
    """Mirror of the kernel's constexpr layout, with a pmax guess for the escape table."""
    nw, r, ups = nwr * wsk, nwr * nt * 8, wsk * upw
    tu = 768 if comp else (2048 if hl else 1024)
    wby, xby = nw * upw * nt * tu, ups * tokr * 128
    sst = (xby + max(wby, ups * 2048 if fuse else 0) + 1023) & ~1023
    toks, rp, cp = 16 * mt, r + ((40 - (r % 32)) % 32), r + 1
    nesc = (4 if nstage >= 4 else nstage) if comp else 0
    corr = nesc * ((toks * cp + 3) & ~3) * 4 if comp else 0
    peer = (cl - 1) * wsk * toks * rp * 4 if cl > 1 else 0
    tab = (pmax * 8 + r * 4) if comp else 0
    return 1024 + nstage * sst + 16 * nstage + 32 + corr + peer + tab + 128 + 16


def _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, cl, fuse=False):
    while nstage > 2 and _sk_smem(mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, cl, fuse) > SMEM_BUDGET:
        nstage -= 1
    return nstage


SK_CONFIGS = _sk_configs()
SK_INDEX = {c[0]: i for i, c in enumerate(SK_CONFIGS)}


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


_SK_SRC = r"""
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

    if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
        if constexpr (CL > 1) {
            // A cluster-scope release after a copy is in flight waits for it, so every barrier
            // is ready before the first copy here.
            for (int i = 0; i < NSTAGE; ++i) {
                mbar_init(full0 + 8 * i, 1);
                mbar_init(empty0 + 8 * i, NW + NESC);
            }
            mbar_init(qbar, 1);
            if constexpr (COMP) {
                mbar_init(pbar, 1);
                mbar_init(cbar, NESC);
            }
            if (crank == 0) {
                mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
            }
            fence_mbar_init();
            issue(0);
            if constexpr (COMP) {
                // Both ranks of a row block load the same table; each corrects its own K range.
                mbar_expect_tx(pbar, tab_ints * 4);
                bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
            }
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
    if constexpr (CL > 1) {
        cluster_arrive();
    }

    if (warp == NW) {
        if (lane == 0) {
            bool dead = false;
            for (int st = 1; st < nst_tot; ++st) {
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
            while (pe <= cnt && (tab[2 * pe] & 0xFFFF) < gs0 * KST) {
                ++pe;
            }
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
                if constexpr (COMP) {
                    // Rank 0 sums only its own escape warps, so this rank's corrections ride
                    // along with its partials, once per row (the wk == 0 warp).
                    if (wk == 0) {
#pragma unroll
                        for (int c = 0; c < NESC; ++c) {
                            const float* cr = corr + c * CSZ + tok * CP + row;
                            v[0] += cr[0];
                            v[1] += cr[1];
                            v[2] += cr[8 * CP];
                            v[3] += cr[8 * CP + 1];
                        }
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
        kern<<<N / C::R, C::THREADS, smem, stream>>>(xmap, wsp, op, tp, pmax, base4, M, N, K, diag, rws, rpart,
                                                     rcount, rout);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
#endif

}  // namespace

#ifdef SK_DEVICE_ONLY
@@INSTANCES@@
#else
void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
            int64_t n, int64_t k, at::Tensor diag) {
    const int M = x.size(0);
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
    switch (cfg) {
@@DISPATCH@@
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
    switch (cfg) {
@@FDISPATCH@@
    default:
        TORCH_CHECK(false, "skf: unknown config ", cfg);
    }
}
#endif
"""

_SK_CPP = (
    "void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, "
    "int64_t base4, int64_t n, int64_t k, at::Tensor diag);\n"
    "void skf_fwd(int64_t cfg, at::Tensor ws, at::Tensor rws, at::Tensor x, at::Tensor out, at::Tensor rout, "
    "at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor rpart, at::Tensor rcount, "
    "at::Tensor diag);\n")


def _sk_targs(c):
    name, mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, nacc, cl, outt, fuse = c
    ot = "float" if outt == "f32" else "__nv_bfloat16"
    return (f"{mt}, {tokr}, {nt}, {nwr}, {wsk}, {upw}, {nstage}, {str(comp).lower()}, {str(hl).lower()}, {nacc}, "
            f"{cl}, {ot}, {str(fuse).lower()}")


def sk_source(device_only=False):
    """The kernel source with the config table expanded; device_only adds explicit
    instantiations for a host-less compile (tools/compile_check.sh)."""
    dispatch = "\n".join(
        f"    case {i}: launch_sk<{_sk_targs(c)}>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, "
        f"nullptr, nullptr, nullptr, nullptr); break;"
        for i, c in enumerate(SK_CONFIGS) if not c[13])
    fdispatch = "\n".join(
        f"    case {i}: launch_sk<{_sk_targs(c)}>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, "
        f"rw, rpart.data_ptr<float>(), rcount.data_ptr<int>(), rout.data_ptr<float>()); break;"
        for i, c in enumerate(SK_CONFIGS) if c[13])
    inst = "\n".join(
        f"template __global__ void sk_kernel<{targs}>(const __grid_constant__ CUtensorMap, const uint8_t*, "
        f"{'float' if c[12] == 'f32' else '__nv_bfloat16'}*, const int32_t*, int, uint32_t, int, int, int, int*, "
        f"const uint8_t*, float*, int*, float*);"
        for targs, c in {_sk_targs(c): c for c in SK_CONFIGS}.items())
    # Explicit instantiation must name the template at namespace scope, so the device-only
    # build reopens the anonymous namespace around it.
    src = (_SK_SRC.replace("@@DISPATCH@@", dispatch).replace("@@FDISPATCH@@", fdispatch)
           .replace("@@INSTANCES@@", "namespace {\n" + inst + "\n}"))
    if not device_only:
        return src
    ginst = "".join(
        f"template void cgemv_launch<{nth}, {cpt}>(const uint4*, const int2*, int, unsigned, "
        f"const __nv_bfloat16*, __nv_bfloat16*, int, cudaStream_t);\n"
        for nth, cpt in ((256, 2), (64, 1), (512, 1), (128, 2)))
    gemv = GEMV_CUDA_SRC.replace("}  // namespace", ginst + "}  // namespace", 1)
    return "#define SK_DEVICE_ONLY 1\n#define GEMV_DEVICE_ONLY 1\n" + src + gemv + wd_source()


# --------------------------------------------------------------------------------------------
# Packing: tile units, the 12-bit code, stage-major streams and escape tables.
# --------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------
# m = 1 GEMV candidates: cgemv on a 12-bit row stream (2 rows per CTA), rgemv on the FP32
# router, sgemv on the raw BF16 unpooler slice. The 12-bit code is the one above, laid out per
# 16-k chunk of a row pair instead of per tile.
# --------------------------------------------------------------------------------------------
_LAYER_THREADS = {(2304, 8192): 256, (8192, 1024): 64, (4096, 8192): 512, (1024, 4096): 128}


def pack_cgemv(w):
    """Stream [N/2, K/16, 48] uint8 and escape table [N/2, pmax, 2] int32 for cgemv_fwd, or None.

    Chunk (c, j) covers rows 2c, 2c+1 and k in [16j, 16j+16): the two rows' low bytes in k
    order, then each row's 8 code bytes, byte b holding the codes of k = 16j+b (low nibble) and
    16j+8+b (high nibble). Table row c: entry 0 is (count, 0); entries 1..count are
    ((row_local << 16) | k, original bits) sorted by k; pmax is even so each row is a multiple
    of 16 bytes.
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
    code = ((bits >> 15) << 3) | off.clamp(0, EXP_WINDOW - 1)
    lo = (bits & 0xFF).view(half, 2, chunks, 16).transpose(1, 2).reshape(half, chunks, 32)
    code = code.view(half, 2, chunks, 2, 8)
    codes = (code[..., 0, :] | (code[..., 1, :] << 4)).transpose(1, 2).reshape(half, chunks, 16)
    stream = torch.cat([lo, codes], -1).to(torch.uint8).contiguous()

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



GEMV_CPP_DECLS = """
void cgemv_fwd(at::Tensor stream, at::Tensor table, int64_t pmax, int64_t base4, at::Tensor x, at::Tensor out, int64_t nthreads);
void rgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out);
void sgemv_fwd(at::Tensor w, at::Tensor x, at::Tensor out);
"""

GEMV_CUDA_SRC = r"""
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
// applies the escape table. Every partial sum is combined in a fixed order, so the result is
// deterministic. The escape terms use separately rounded multiply and add so a CPU emulation
// in float32 reproduces them bit for bit (the main FMAs are exact products, so they need no such
// care).
// The block count in __launch_bounds__ (here and below) is for ptxas: without it ptxas budgets
// 32 registers for full occupancy and sinks the later loads below the first FMAs; with it every
// load is in flight before the first FMA. tools/gemv_compile_check.sh checks that in the SASS.
template <int NTH, int CPT>
__global__ void __launch_bounds__(NTH + 32, 2)
cgemv_kernel(const uint4* __restrict__ stream, const int2* __restrict__ table, int pmax,
             unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {
    constexpr int NW = NTH / 32, CHUNKS = NTH * CPT;
    __shared__ float red[2][NW + 1];
    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (warp < NW) {
        const uint4* wp = stream + ((size_t)c * CHUNKS + t) * 3;
        const uint4* xp = reinterpret_cast<const uint4*>(x) + 2 * t;
        uint4 wa[CPT], wb[CPT], wc[CPT], xa[CPT], xb[CPT];
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            wa[i] = ld_stream(wp + i * NTH * 3);
            wb[i] = ld_stream(wp + i * NTH * 3 + 1);
            wc[i] = ld_stream(wp + i * NTH * 3 + 2);
        }
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            xa[i] = __ldg(xp + i * NTH * 2);
            xb[i] = __ldg(xp + i * NTH * 2 + 1);
        }
        float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            chunk_fma(wa[i], wc[i].x, wc[i].y, xa[i], xb[i], base4, acc0);
            chunk_fma(wb[i], wc[i].z, wc[i].w, xa[i], xb[i], base4, acc1);
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
        if (lane == 0) {
            red[0][warp] = acc0;
            red[1][warp] = acc1;
        }
    } else {
        const int2* tb = table + (size_t)c * pmax;
        const int count = __ldg(tb).x;
        // Entries 1..pmax-1 exist for every CTA, so the first batch loads alongside the count.
        int2 ent = __ldg(tb + min(lane + 1, pmax - 1));
        const unsigned base = base4 & 0xFFu;
        float corr0 = 0.f, corr1 = 0.f;
        for (int e = lane + 1; e <= count; e += 32) {
            const int row = ent.x >> 16, kk = ent.x & 0xFFFF;
            const unsigned bits = (unsigned)ent.y;
            ent = __ldg(tb + min(e + 32, pmax - 1));
            const int off = min(max((int)((bits >> 8) & 0x7Fu) - (int)base, 0), EXP_WINDOW - 1);
            const unsigned dec = (bits & 0x80FFu) | ((base + off) << 8);
            const float delta = __fsub_rn(__uint_as_float(bits << 16), __uint_as_float(dec << 16));
            const float xv = __uint_as_float(
                (unsigned)__ldg(reinterpret_cast<const unsigned short*>(x) + kk) << 16);
            const float d = __fmul_rn(delta, xv);
            corr0 = __fadd_rn(corr0, row ? 0.f : d);
            corr1 = __fadd_rn(corr1, row ? d : 0.f);
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

template <int NTH, int CPT>
void cgemv_launch(const uint4* stream, const int2* table, int pmax, unsigned base4,
                  const __nv_bfloat16* x, __nv_bfloat16* out, int half_n, cudaStream_t s) {
    cgemv_kernel<NTH, CPT><<<half_n, NTH + 32, 0, s>>>(stream, table, pmax, base4, x, out);
}

// FP32 router row per CTA: thread t reads float4 i*256+t, so each warp's load is 512 contiguous
// bytes, and all 16 loads are in flight before the first FMA.
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
    const uint4* sp = reinterpret_cast<const uint4*>(stream.data_ptr<uint8_t>());
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



# --------------------------------------------------------------------------------------------
# Wide kernel (wd) for 33 to 256 tokens: TMA + wgmma, swap-AB. A 64-row weight tile per
# consumer warpgroup is the wgmma A operand and a BN-token x tile is the B operand, both K-major
# and 128B-swizzled straight from TMA, so the weights need no repack. K is split across the CTAs
# of a cluster; their FP32 partials are summed over distributed shared memory, each rank
# writing BN/CLK tokens; CLR row blocks of a cluster share one multicast x tile. The router runs
# on hi and lo tiles into one FP32 accumulator.
# name -> (WG, BN, CLK, CLR, HL, NSTAGE). Rows per CTA = 64 * WG; cluster = CLK * CLR CTAs.
# --------------------------------------------------------------------------------------------
WD_GEOM = {
    "r64c8": (1, 64, 8, 1, True, 4),
    "r64c16": (1, 64, 16, 1, True, 4),
    "r128c8": (1, 128, 8, 1, True, 4),
    "r128c16": (1, 128, 16, 1, True, 4),
    "b1n64c2": (1, 64, 2, 1, False, 6),
    "b1n128c2": (1, 128, 2, 1, False, 4),
    "b2n256c2": (2, 256, 2, 1, False, 4),
    # x multicast across CLR row blocks (job 6131a4a220d5486e: x re-reads dominate L2 traffic).
    "m64k2r4": (1, 64, 2, 4, False, 6),
    "m64k1r8": (1, 64, 1, 8, False, 6),
    "m128k2r4": (1, 128, 2, 4, False, 4),
    "m128k1r4": (1, 128, 1, 4, False, 4),
    "m128k1r8": (1, 128, 1, 8, False, 4),
    "m128k2r8": (1, 128, 2, 8, False, 4),
    "m128k4r2": (1, 128, 4, 2, False, 4),
    "m256k2r4": (1, 256, 2, 4, False, 4),
    "m256k1r4": (1, 256, 1, 4, False, 4),
    "m256k1r8": (1, 256, 1, 8, False, 4),
    "m256k2r8": (1, 256, 2, 8, False, 4),
    "m256k2r2": (1, 256, 2, 2, False, 4),
    "rm64k4r4": (1, 64, 4, 4, True, 6),
    "rm128k4r4": (1, 128, 4, 4, True, 4),
}
# Diagnostic builds timed in prepare(), never dispatched: ABL bit 1 skips the wgmma (memory
# pipeline alone), bit 2 skips the cluster reduction and the stores.
WD_ABLATE = {("pooler", 128): "m128k2r4", ("o", 256): "m256k1r4"}
WD_ABL = {f"{base}_a{a}": (base, a) for base in WD_ABLATE.values() for a in (1, 3)}
WD_NAMES = list(WD_GEOM) + list(WD_ABL)
WD_INDEX = {nm: i for i, nm in enumerate(WD_NAMES)}
# Wide buckets: the pick tuned at the bucket's top m serves every m in (previous top, top].
WD_BUCKETS = (64, 128, 256)
WD_SELFTEST_MS = {64: (33, 47, 64), 128: (65, 100, 128), 256: (129, 130, 200, 256)}
WD_BN = {64: (64,), 128: (64, 128), 256: (128, 256)}
# A wide kernel replaces cuBLAS only if it is this much faster in the autotune: the tuner's
# medians carry about 1% noise, and job 71b1927dca894bc4 picked pooler m=128 on 30.2 vs 30.5 us
# that the judge then timed at 0.87x.
WD_MARGIN = 0.97


def wd_smem(wg, bn, clk, clr, hl, nst):
    sb = wg * (2 if hl else 1) * 8192 + bn * 128
    return 1024 + nst * sb + 16 * nst


def wd_geom(name):
    return WD_GEOM[WD_ABL[name][0]] if name in WD_ABL else WD_GEOM[name]


def wd_fits(name, n, k):
    wg, bn, clk, clr, hl, nst = wd_geom(name)
    return n % (64 * wg * clr) == 0 and k % 64 == 0 and (k // 64) % clk == 0


def _wgmma_spec(bn):
    nreg = bn // 2
    regs = ", ".join(f"%{i}" for i in range(nreg))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    return (f"template <> __device__ __forceinline__ void wgmma_ss<{bn}>(float* d, uint64_t a, uint64_t b) {{\n"
            f"    asm volatile(\"{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nreg + 2}, 0;\\n\"\n"
            f"                 \"wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 {{{regs}}}, \"\n"
            f"                 \"%{nreg}, %{nreg + 1}, p, 1, 1, 0, 0;\\n}}\\n\"\n"
            f"                 : {outs}\n"
            f"                 : \"l\"(a), \"l\"(b), \"r\"(1) : \"memory\");\n}}\n")


_WD_SRC = r"""
namespace {

template <int BN>
__device__ __forceinline__ void wgmma_ss(float* d, uint64_t a, uint64_t b);
@@WGMMA@@

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
    static_assert(BN % CLK == 0 && BN % 8 == 0 && BN <= 256, "wd: bad BN");
    // Each multicast slice must start on a 1024-byte swizzle atom so the slices tile exactly
    // like one box.
    static_assert(BN % CLR == 0 && XS % 8 == 0, "wd: bad x slice");
    static_assert(CL <= 16, "wd: cluster too large");
    static_assert(SMEM <= 227 * 1024, "wd: shared memory over budget");
};

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL>
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
    const int ktl = (K / 64) / CLK;
    const int kt0 = rk * ktl;
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
                    tma_2d(st + g * C::NA * C::ATILE, &wmap, fb, kc, rowbase + 64 * g);
                    if constexpr (HL) {
                        tma_2d(st + (g * C::NA + 1) * C::ATILE, &wmap, fb, kc, N + rowbase + 64 * g);
                    }
                }
                if constexpr (CLR > 1) {
                    tma_2d_mc(st + C::ABY + rr * C::XS * 128, &xmap, fb, kc, tok0 + rr * C::XS, mask);
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

    constexpr int TPR = BN / CLK;
    constexpr int C4 = C::BM / 4;
    for (int it = tid; it < ((ABL & 2) ? 0 : TPR * C4); it += C::THREADS) {
        const int t = rk * TPR + it / C4;
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

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL>
void launch_wd(const at::Tensor& w, const at::Tensor& x, at::Tensor& out, int N, int K, int* diag) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST>;
    auto kern = wd_kernel<WG, BN, CLK, CLR, HL, NST, OutT, ABL>;
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
    TORCH_CHECK(M >= 1 && N % (C::BM * CLR) == 0 && K % 64 == 0 && (K / 64) % CLK == 0, "wd: shape does not tile");
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
@@WDINSTANCES@@
#else
void wd_fwd(int64_t cfg, at::Tensor w, at::Tensor x, at::Tensor out, int64_t n, int64_t k, at::Tensor diag) {
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
    switch (cfg) {
@@WDDISPATCH@@
    default:
        TORCH_CHECK(false, "wd: unknown config ", cfg);
    }
}
#endif
"""

_WD_CPP = "void wd_fwd(int64_t cfg, at::Tensor w, at::Tensor x, at::Tensor out, int64_t n, int64_t k, at::Tensor diag);\n"


def _wd_targs(name):
    wg, bn, clk, clr, hl, nst = wd_geom(name)
    abl = WD_ABL[name][1] if name in WD_ABL else 0
    return f"{wg}, {bn}, {clk}, {clr}, {str(hl).lower()}, {nst}, {'float' if hl else '__nv_bfloat16'}, {abl}"


def wd_source():
    """The wide kernel with its wgmma shapes and config table expanded; under SK_DEVICE_ONLY it
    emits explicit instantiations instead of the host dispatch."""
    for name, (wg, bn, clk, clr, hl, nst) in WD_GEOM.items():
        assert wd_smem(wg, bn, clk, clr, hl, nst) <= SMEM_BUDGET and clk * clr <= 16, name
        assert bn % clk == 0 and bn % clr == 0 and (bn // clr) % 8 == 0, name
        assert bn * (64 * wg + 4) * 4 <= nst * (wg * (2 if hl else 1) * 8192 + bn * 128), name
    wgmma = "".join(_wgmma_spec(bn) for bn in sorted({g[1] for g in WD_GEOM.values()}))
    dispatch = "\n".join(
        f"    case {i}: launch_wd<{_wd_targs(nm)}>(w, x, out, (int)n, (int)k, dg); break;"
        for i, nm in enumerate(WD_NAMES))
    inst = "\n".join(
        f"template __global__ void wd_kernel<{_wd_targs(nm)}>(const __grid_constant__ CUtensorMap, "
        f"const __grid_constant__ CUtensorMap, {'float' if wd_geom(nm)[4] else '__nv_bfloat16'}*, int, int, int, int*);"
        for nm in WD_NAMES)
    return (_WD_SRC.replace("@@WGMMA@@", wgmma).replace("@@WDDISPATCH@@", dispatch)
            .replace("@@WDINSTANCES@@", "namespace {\n" + inst + "\n}"))


# --------------------------------------------------------------------------------------------
# Candidates, self-test, autotune, dispatch.
# --------------------------------------------------------------------------------------------
def bucket_of(m):
    if m <= 32:
        return 8 if m <= 8 else 16 if m <= 16 else 32
    return next((b for b in WD_BUCKETS if m <= b), None)


def _gen_x(m, k, gen, device):
    """Random tokens with the judge's probe rows: one power-of-two entry in about one row in 8."""
    x = torch.randn((m, k), generator=gen, device=device)
    probes = torch.rand((m,), generator=gen, device=device) < 1 / 8
    probes[0] = True
    cols = torch.randint(k, (m,), generator=gen, device=device)
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16), probes


def _check(out, ref, probes, tol):
    out = out.double()
    err = float((out - ref).norm() / ref.norm())
    rows = probes.nonzero().flatten()
    perr = float(((out[rows] - ref[rows]).norm(dim=-1) / ref[rows].norm(dim=-1).clamp(min=1e-30)).max())
    return err < tol and perr < PROBE_TOL and bool(torch.isfinite(out).all()), err, perr


def _self_test(name, fn, w, nrows, ms, tol, gen, diag):
    """Every m in ms against FP64 on random inputs with probe rows; raises on the first failure."""
    k = w.shape[1]
    for m in ms:
        diag.zero_()
        x, probes = _gen_x(m, k, gen, w.device)
        out = fn(x)
        torch.cuda.synchronize()
        ref = (x.double() @ w[:nrows].double().T)
        if out.dtype == torch.bfloat16:
            ref = ref.to(torch.bfloat16).double()
        ok, err, perr = _check(out, ref, probes, tol)
        if not ok:
            raise RuntimeError(f"{name} m={m}: rel err {err:.3e}, probe err {perr:.3e}, diag {int(diag[0])}")


def _match_twin(fn, twin, ms, k, gen, dev):
    """A K-split config against the unsplit config on the same stream format: only the FP32
    summation order differs, so every element must agree to within two bf16 ulps. The FP64
    self-test's relative-L2 tolerance could miss a few dropped escape corrections; this cannot."""
    for m in ms:
        x, _ = _gen_x(m, k, gen, dev)
        a, b = fn(x).double(), twin(x).double()
        torch.cuda.synchronize()
        bad = int(((a - b).abs() > 2 ** -6 * torch.maximum(a.abs(), b.abs()) + 1e-5).sum())
        if bad:
            raise RuntimeError(f"m={m}: {bad} elements differ from the unsplit twin by more than 2 ulps")


def _time_candidates(fns, x, flush):
    """Median microseconds per candidate over TUNE_REPS CUDA-graph replays from a cold L2,
    candidates interleaved in alternating order so drift cancels."""
    graphs = []
    for fn in fns:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn(x)
        graphs.append(g)
    times = [[] for _ in fns]
    for rep in range(TUNE_REPS):
        order = range(len(fns)) if rep % 2 == 0 else range(len(fns) - 1, -1, -1)
        for i in order:
            flush.sum()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[i].replay()
            e1.record()
            torch.cuda.synchronize()
            times[i].append(1000 * e0.elapsed_time(e1))
    return [statistics.median(t) for t in times]


def _cublas(x, w):
    if _bf16_gemm_dispatch_impl is not None:
        return _bf16_gemm_dispatch_impl(x, w, None)
    return x @ w.T


def _router_baseline(x, state):
    if x.shape[0] <= 8:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _sk_runner(ext, cfg_name, stream, tab, pmax, base4, n, k, out_dtype, diag):
    cid = SK_INDEX[cfg_name]

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
        ext.sk_fwd(cid, stream, x, out, tab, pmax, base4, n, k, diag)
        return out
    return run


def _wd_runner(ext, name, w, n, k, out_dtype, diag):
    cid = WD_INDEX[name]

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run


def _build_ext():
    src = sk_source() + GEMV_CUDA_SRC + wd_source()
    kwargs = dict(name="okc_sk_v2", cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + GEMV_CPP_DECLS + _WD_CPP,
                  cuda_sources=src,
                  functions=["sk_fwd", "skf_fwd", "cgemv_fwd", "rgemv_fwd", "sgemv_fwd", "wd_fwd"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    try:
        return load_inline(no_implicit_headers=True, **kwargs)
    except TypeError:
        kwargs["cuda_sources"] = src.replace("#include <ATen/core/Tensor.h>", "#include <torch/types.h>")
        return load_inline(**kwargs)


def _build_fused(ext, state, packed_pooler, gen, diag):
    """The layer's fused pooler + router entry per token bucket (8 and 16), self-tested here:
    run(x) -> (pooler out [M, 4096] bf16, router out [M, 256] f32)."""
    if packed_pooler is None:
        return {}
    router, pooler = state["router"], state["pooler"]
    dev = router.device
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    units = hilo_units(bf16_bits(hi), bf16_bits(lo))
    fused = {}
    for b, nm in ((8, "pool8"), (16, "pool16")):
        rpart = torch.zeros((8, 16, 256), dtype=torch.float32, device=dev)
        rcount = torch.zeros(32, dtype=torch.int32, device=dev)
        g = _sk_geom(nm)
        rstream = stage_stream(units, _fused_router_geom(nm), router.shape[0])
        tab, pmax = packed_pooler["tabs"][(g["R"], pooler.shape[0])]
        stream, base4 = packed_pooler["streams"][nm], packed_pooler["base4"]
        cid = SK_INDEX[nm + "f"]

        def run(x, cid=cid, stream=stream, rstream=rstream, tab=tab, pmax=pmax, base4=base4, rpart=rpart,
                rcount=rcount):
            m = x.shape[0]
            out = torch.empty((m, pooler.shape[0]), dtype=torch.bfloat16, device=x.device)
            rout = torch.empty((m, router.shape[0]), dtype=torch.float32, device=x.device)
            ext.skf_fwd(cid, stream, rstream, x, out, rout, tab, pmax, base4, pooler.shape[0], pooler.shape[1],
                        rpart, rcount, diag)
            return out, rout
        try:
            for m in ((1, 3, 8) if b == 8 else (9, 16)):
                diag.zero_()
                x, probes = _gen_x(m, pooler.shape[1], gen, dev)
                out, rout = run(x)
                torch.cuda.synchronize()
                ref = (x.double() @ pooler.double().T).to(torch.bfloat16).double()
                ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["bf16"])
                rref = x.double() @ router.double().T
                rok, rerr, rperr = _check(rout, rref, probes, SELFTEST_TOL["router"])
                if not (ok and rok):
                    raise RuntimeError(f"m={m}: pooler err {err:.3e}/{perr:.3e}, router err {rerr:.3e}/{rperr:.3e},"
                                       f" diag {int(diag[0])}")
            fused[b] = run
            print(f"fused pooler+router {nm}f: ok")
        except Exception as exc:
            print(f"fused pooler+router {nm}f: disabled ({type(exc).__name__}: {str(exc)[-300:]})")
    return fused


def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["pick"] = {}
    dev = router.device
    try:
        ext = _build_ext()
    except Exception as exc:
        print(f"sk build failed, cuBLAS everywhere ({type(exc).__name__}: {str(exc)[-800:]})")
        return state
    print(f"sk build {time.time() - t0:.1f} s")
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(1234)
    cands = {}

    def add(layer, bucket, name, fn, w, nrows, ms, tol):
        try:
            _self_test(f"{layer} {name}", fn, w, nrows, ms, tol, gen, diag)
            cands.setdefault((layer, bucket), []).append((name, fn, ms == (1,)))
        except Exception as exc:
            print(f"selftest {layer} {name}: disabled ({type(exc).__name__}: {str(exc)[-300:]})")

    def gemv_runner(call, n, dtype):
        def run(x):
            out = torch.empty((1, n), dtype=dtype, device=x.device)
            call(x, out)
            return out
        return run

    ms_of = {8: (1, 2, 5, 7, 8), 16: (9, 12, 16), 32: (17, 31, 32)}

    def names_of(layer, b):
        return [SK_USE[(layer, b)]] + list(SK_EXTRA.get((layer, b), []))

    for layer in BF16_LAYERS:
        w = state[layer]
        n, k = w.shape
        names = {b: names_of(layer, b) for b in (8, 16, 32)}
        comp_names = [nm for lst in names.values() for nm in lst if not nm.endswith("raw")]
        raw_names = [nm for lst in names.values() for nm in lst if nm.endswith("raw")]
        raw_names += [nm + "raw" for nm in comp_names]
        rows_of = lambda b: UNPOOL_SLICE if layer == "unpooler" and b == 8 else n
        slices = {nm: rows_of(b) for b, lst in names.items() for nm in lst}
        slices.update({nm + "raw": rows_of(b) for b, lst in names.items() for nm in lst})
        geoms = {nm: _sk_geom(nm) for nm in comp_names}
        packed = pack_layer(w, geoms, slices, balance=layer in BALANCED_LAYERS) if comp_names else None
        if layer == "pooler":
            packed_pooler = packed
        if packed is not None:
            print(f"pack {layer}: base {packed['base']}, {packed['escapes']} escapes, "
                  f"{packed['inexact']} inexact deltas, {time.time() - t0:.1f} s")
            for b, lst in names.items():
                twin = None
                for nm in lst:
                    if nm not in packed["streams"]:
                        continue
                    nrows = slices[nm]
                    tab, pmax = packed["tabs"][(geoms[nm]["R"], nrows)]
                    fn = _sk_runner(ext, nm, packed["streams"][nm], tab, pmax, packed["base4"], nrows, k,
                                    torch.bfloat16, diag)
                    if twin is None:
                        twin = fn
                    elif geoms[nm]["cl"] > 1:
                        try:
                            _match_twin(fn, twin, ms_of[b], k, gen, dev)
                        except Exception as exc:
                            print(f"selftest {layer} {nm}: disabled ({type(exc).__name__}: {str(exc)[-300:]})")
                            continue
                    add(layer, b, nm, fn, w, nrows, ms_of[b], SELFTEST_TOL["bf16"])
        elif comp_names:
            print(f"pack {layer}: rejected, raw BF16 only")
        raw = pack_raw_layer(w, {nm: _sk_geom(nm) for nm in raw_names}, slices)
        for b, lst in names.items():
            for nm in lst:
                rnm = nm if nm.endswith("raw") else nm + "raw"
                nrows = slices[rnm]
                fn = _sk_runner(ext, rnm, raw[rnm], diag, 0, 0, nrows, k, torch.bfloat16, diag)
                add(layer, b, rnm, fn, w, nrows, ms_of[b], SELFTEST_TOL["bf16"])
        w1 = w[:rows_of(8)]
        cg = pack_cgemv(w1)
        if cg is not None:
            nth = cgemv_threads(*w1.shape)
            fn = gemv_runner(lambda x, out, cg=cg, nth=nth: ext.cgemv_fwd(
                cg["stream"], cg["table"], cg["pmax"], cg["base4"], x, out, nth), w1.shape[0], torch.bfloat16)
            add(layer, 8, "cgemv", fn, w, w1.shape[0], (1,), SELFTEST_TOL["bf16"])
        if layer == "unpooler":
            fn = gemv_runner(lambda x, out, w1=w1: ext.sgemv_fwd(w1, x, out), w1.shape[0], torch.bfloat16)
            add(layer, 8, "sgemv", fn, w, w1.shape[0], (1,), SELFTEST_TOL["bf16"])
        for b in (8, 16, 32):
            nrows = rows_of(b)
            cands.setdefault((layer, b), []).append(("cublas", lambda x, w=w, nrows=nrows: _cublas(x, w[:nrows]), False))
    fn = gemv_runner(lambda x, out: ext.rgemv_fwd(router, x, out), router.shape[0], torch.float32)
    add("router", 8, "rgemv", fn, router, router.shape[0], (1,), SELFTEST_TOL["router"])
    state["fused"] = _build_fused(ext, state, packed_pooler, gen, diag)
    rnames = {b: names_of("router", b) for b in (8, 16, 32)}
    rgeoms = {nm: _sk_geom(nm) for lst in rnames.values() for nm in lst}
    _, _, rstreams = pack_router(router, rgeoms)
    for b, lst in rnames.items():
        for nm in lst:
            fn = _sk_runner(ext, nm, rstreams[nm], diag, 0, 0, router.shape[0], router.shape[1], torch.float32, diag)
            add("router", b, nm, fn, router, router.shape[0], ms_of[b], SELFTEST_TOL["router"])
        cands.setdefault(("router", b), []).append(("baseline", lambda x: _router_baseline(x, state), False))
    state["router_hilo"] = torch.cat([hi, lo], 0).contiguous()
    for layer in BF16_LAYERS + ("router",):
        is_router = layer == "router"
        w = state[layer]
        n, k = w.shape
        wk = state["router_hilo"] if is_router else w
        for b in WD_BUCKETS:
            # cuBLAS first: it stands in if timing fails.
            if is_router:
                cands.setdefault((layer, b), []).append(("baseline", lambda x: _router_baseline(x, state), False))
            else:
                cands.setdefault((layer, b), []).append(("cublas", lambda x, w=w: _cublas(x, w), False))
            for nm in WD_GEOM:
                wg, bn, clk, clr, hl, nst = WD_GEOM[nm]
                if hl != is_router or bn not in WD_BN[b] or not wd_fits(nm, n, k):
                    continue
                fn = _wd_runner(ext, nm, wk, n, k, torch.float32 if is_router else torch.bfloat16, diag)
                add(layer, b, nm, fn, w, n, WD_SELFTEST_MS[b], SELFTEST_TOL["router" if is_router else "bf16"])
    print(f"selftests done {time.time() - t0:.1f} s")

    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    for (layer, b), all_cands in sorted(cands.items()):
        k = state[layer].shape[1]
        for m in ((1, 8) if b == 8 else (b,)):
            lst = [c for c in all_cands if m == 1 or not c[2]]
            x, _ = _gen_x(m, k, gen, dev)
            best = 0
            if len(lst) > 1:
                try:
                    us = _time_candidates([fn for _, fn, _ in lst], x, flush)
                    best = min(range(len(lst)), key=lambda i: us[i])
                    if b in WD_BUCKETS and us[best] > WD_MARGIN * us[0]:
                        best = 0
                    print(f"tune {layer:9s} m={m:2d}: " + ", ".join(f"{c[0]} {t:.1f}" for c, t in zip(lst, us))
                          + f" -> {lst[best][0]}")
                except Exception as exc:
                    # Candidates are listed fastest-expected first, so the first one stands in.
                    torch.cuda.synchronize()
                    print(f"tune {layer:9s} m={m:2d}: timing failed ({type(exc).__name__}: {str(exc)[-200:]}),"
                          f" using {lst[0][0]}")
            state["pick"][(layer, m)] = lst[best][1]
        if b != 8:
            continue
        for m in (2, 3, 4, 5, 6, 7):
            state["pick"][(layer, m)] = state["pick"][(layer, 8)]
    for (layer, b), lst in cands.items():
        for m in range(9, WD_BUCKETS[-1] + 1):
            if (layer, m) not in state["pick"]:
                state["pick"][(layer, m)] = state["pick"][(layer, bucket_of(m))]
    for layer, entries in SK_SWEEP:
        # One compiled kernel at several M runs the same instructions (rows >= M read the zero
        # row), so its time against M isolates the x bytes each CTA streams.
        fns, labels = [], []
        for nm, b, ms in entries:
            fn = next((c[1] for c in cands.get((layer, b), []) if c[0] == nm), None)
            if fn is None:
                continue
            for m in ms:
                fns.append(lambda x, fn=fn, m=m: fn(x[:m]))
                labels.append(f"{nm}@{m}")
        try:
            x, _ = _gen_x(32, state[layer].shape[1], gen, dev)
            us = _time_candidates(fns, x, flush)
            print(f"sweep {layer}: " + ", ".join(f"{lb} {t:.1f}" for lb, t in zip(labels, us)))
        except Exception as exc:
            torch.cuda.synchronize()
            print(f"sweep {layer}: failed ({type(exc).__name__}: {str(exc)[-200:]})")
    for (layer, m), base in WD_ABLATE.items():
        is_router = layer == "router"
        w = state["router_hilo"] if is_router else state[layer]
        n, k = state[layer].shape
        dt = torch.float32 if is_router else torch.bfloat16
        names = [base] + [nm for nm, (b, _) in WD_ABL.items() if b == base]
        try:
            x, _ = _gen_x(m, k, gen, dev)
            us = _time_candidates([_wd_runner(ext, nm, w, n, k, dt, diag) for nm in names], x, flush)
            print(f"ablate {layer} m={m}: " + ", ".join(f"{nm} {t:.1f}" for nm, t in zip(names, us)))
        except Exception as exc:
            torch.cuda.synchronize()
            print(f"ablate {layer} m={m}: failed ({type(exc).__name__}: {str(exc)[-200:]})")
    print(f"prepare done {time.time() - t0:.1f} s")
    return state


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    run = state["pick"].get((layer, m))
    if run is not None:
        return run(x.contiguous())
    if layer == "router":
        return _router_baseline(x, state)
    return _cublas(x, state[layer][:n])
