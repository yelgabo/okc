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
- m = 33 to 256 has the wd TMA + wgmma kernel, and pooler m = 129 to 256 a CUTLASS pingpong
  GEMM (CUT_SRC) built from the CUTLASS headers in the image's FlashInfer package.
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
    # Four units per warp per stage: 8 stages instead of 16. Every stage pays the per-lane proxy
    # fence (a MEMBAR.ALL.CTA in SASS) before the empty arrive; halving the stages was -0.69 /
    # -0.66 us against two units per warp and ties the public entry (bench 2b42507273c14aab).
    "qkv8": (1, 8, 3, 1, 4, 4, 4),
    "qkv16": (1, 16, 3, 1, 4, 2, 5),
    "qkv32": (2, 32, 3, 1, 4, 2, 4),
    "o8": (1, 8, 4, 1, 4, 1, 4),
    # Four consumer warps (K split 2 ways) instead of two: K = 1024 is only four stages, and
    # -0.22 / -0.43 / -0.38 / -0.35 us paired (benches 01703396ea1a409e, 3678bfd3ef1c4ce6).
    "o16": (1, 16, 2, 2, 2, 2, 3),
    "o32": (2, 32, 2, 2, 2, 2, 3),
    "pool8": (1, 8, 1, 2, 2, 4, 3),
    "pool16": (1, 16, 2, 2, 2, 4, 4),
    # Four units per consumer warp per stage. With two (32 stages) the consumers are busy ~75%
    # of the kernel and each extra stage costs ~120 ns of fixed path (wait, operand loads, proxy
    # fence, arrive), so the old geometry ran 2.8 us slower at any token count (checks
    # 105c1beba7284938, 748b9d6885324f40, 73aa65b0ccec4745).
    "pool32": (2, 32, 2, 2, 2, 4, 4),
    "slice8": (1, 8, 1, 1, 8, 2, 4),
    "unp16": (1, 16, 2, 2, 2, 2, 3),
    # 32 rows per CTA, 256 CTAs at 2 per SM (98 KB) instead of 64 rows at 1: -0.54 / -0.16 /
    # -0.32 / -0.27 us paired (benches 01703396ea1a409e, 3678bfd3ef1c4ce6).
    "unp32": (2, 32, 2, 2, 2, 2, 3),
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
SK_EXTRA = {}
# name -> NACC template value for SK_GEOM configs (default 1): bits 0-3 the accumulator count,
# SK_ESCG the global-x escape corrections (SkCfg::ESCG).
SK_ESCG = 16
SK_NOVOTE = 1024
# Every compressed sk config waits on its full barriers per thread without the warp vote and
# with the clock read once per 256 polls (mbar_wait_novote; still fails closed through diag).
# Paired against the voted per-poll wait in benches 8712eb94a71e4075 / eebd764690d5470f
# (F-PROJ-1, docs/runs.md): qkv32 -0.45, other cases 0 to -0.16 alone, more with FULL below.
SK_NACC = {nm: 1 | SK_NOVOTE for nm in SK_GEOM}
# Compressed configs that also get a twin compiled for M == TOKR (name + "F", SkCfg::FULL:
# compile-time x-tile addressing, the donor sk2's packed-off layout at full tiles); _sk_runner
# dispatches to it when the call has exactly TOKR tokens. With NOVOTE: pool32 -0.45, pool16
# -0.34, qkv16 / unp16 / qkv8 / pool8 -0.18 / -0.18 / -0.18 / -0.19 us (eebd764690d5470f);
# qkv32 is left out (its FULL twin lost 0.2 us to the plain NOVOTE form there).
SK_FULL = {nm for nm in SK_GEOM if nm != "qkv32"}
SMEM_BUDGET = 227 * 1024
SMEM_PER_SM = 228 * 1024


def _sk_configs():
    """Every kernel instantiation: (name, MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, out, fuse)."""
    cfgs = []
    for name, (mt, tokr, nt, nwr, wsk, upw, nstage) in SK_GEOM.items():
        for comp in (True, False):
            cname = name if comp else name + "raw"
            nacc = SK_NACC.get(name, 1) if comp else SK_NACC.get(name, 1) & 15
            nst = _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, comp, False, 1)
            cfgs.append((cname, mt, tokr, nt, nwr, wsk, upw, nst, comp, False, nacc, 1, "bf16", False))
            if comp and name in SK_FULL:
                cfgs.append((name + "F", mt, tokr, nt, nwr, wsk, upw, nst, comp, False, nacc | 32, 1, "bf16", False))
    for name, (mt, tokr, nt, nwr, wsk, upw, nstage) in ROUTER_GEOM.items():
        cfgs.append((name, mt, tokr, nt, nwr, wsk, upw,
                     _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, False, True, 3), False, True, 2, 3, "f32",
                     False))
    for name in FUSED_GEOM:
        mt, tokr, nt, nwr, wsk, upw, nstage = SK_GEOM[name]
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

// OKC_WAIT_SLEEP 0 drops the suspend-time hint from every producer and escape-warp wait (bench
// builds only; the judged builds use 1).
#ifndef OKC_WAIT_SLEEP
#define OKC_WAIT_SLEEP 1
#endif

__device__ __forceinline__ bool mbar_test_sleep(uint32_t bar, uint32_t parity) {
#if OKC_WAIT_SLEEP == 0
    return mbar_test(bar, parity);
#endif
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

// The same wait with the clock read once per 256 polls instead of every poll; still fails
// closed after WAIT_LIMIT cycles (sk consumer variant LITEWD, bench round f4).
__device__ __forceinline__ bool mbar_wait_lite(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead) {
        return false;
    }
    if (mbar_test(bar, parity)) {
        return true;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return false;
            }
        }
    }
    return true;
}

// Per-thread wait without a warp vote: once a thread times out it marks itself dead and skips
// every later wait, so all lanes still run the same stage loop (the mma.sync stays converged)
// and the result is garbage with diag set, which the self-test rejects (fail closed). The
// clock is read once per 256 polls. Measured against the voted per-poll form in round f4.
__device__ __forceinline__ void mbar_wait_novote(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead || mbar_test(bar, parity)) {
        return;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return;
            }
        }
    }
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
    // NACC bit 4 (SK_ESCG in Python) selects escape corrections read from x in global memory:
    // the escape warps then never touch the ring, so they take table entries round-robin instead
    // of stages and are not counted on the empty barriers (the scheme of alygg77's
    // kernels/muse_bf16_proj.py `_escape_source`, re-implemented here). Bits 0-3 are the
    // accumulator count.
    static constexpr bool ESCG = (NACC & 16) != 0;
    static constexpr int NA = NACC & 15;
    // Bench variants (bits 5-8): FULL compiles the x-tile addressing for M == TOKR (the launch
    // checks it); ESCFAST reads escape entries and x with plain loads that the compiler may
    // overlap across entries; NOWD drops the consumer full-wait watchdog and vote; ESCOFF skips
    // the corrections (attribution only, outputs inexact, never dispatched by prepare()).
    static constexpr bool FULL = (NACC & 32) != 0;
    static constexpr bool ESCFAST = (NACC & 64) != 0;
    static constexpr bool NOWD = (NACC & 128) != 0;
    static constexpr bool ESCOFF = (NACC & 256) != 0;
    // LITEWD: the consumer full wait polls the clock once per 256 polls (still fails closed).
    static constexpr bool LITEWD = (NACC & 512) != 0;
    // NOVOTE: the consumer full wait is mbar_wait_novote (no vote, no break, fails closed).
    static constexpr bool NOVOTE = (NACC & 1024) != 0;
    static constexpr int NESC = COMP ? (ESCG ? 4 : (NSTAGE >= 4 ? 4 : NSTAGE)) : 0;
    static constexpr int EMPTY_COUNT = NW + (ESCG ? 0 : NESC);
    static constexpr int THREADS = (NW + 1 + NESC) * 32;
    static constexpr int CSZ = (TOKS * CP + 3) & ~3;
    static constexpr int CORR_FLOATS = COMP ? NESC * CSZ : 0;
    static constexpr int PEER_FLOATS = CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0;
    static constexpr int BAR_BYTES = 16 * NSTAGE + 32;
    static constexpr int FIXED = 1024 + NSTAGE * SST + BAR_BYTES + 4 * (CORR_FLOATS + PEER_FLOATS) + 128 + 16;
    static constexpr int MINB = 2 * (FIXED + 2048) <= 228 * 1024 ? 2 : 1;
    // Release a slot after the stage's MMAs instead of after its last operand loads (the public
    // entry's order; same fence). At 32 tokens with two units per warp this was -0.38 / -0.29 us
    // on qkv_gate, -0.26 / -0.19 on o, -0.13 / -0.16 on the unpooler, but +0.26 / +0.26 on the
    // pooler (UPW 4) and +0.10 / +0.16 on the m = 8 slice (benches 807464e240ec4f6c,
    // c991867992d34fb2), so it is limited to MT 2 with UPW 2.
    static constexpr bool LATE_ARRIVE = MT == 2 && UPW == 2;
    static_assert(WSK * TOKS * RP * 4 <= NSTAGE * SST, "epilogue partials must fit in the ring");
    static_assert(!FUSE || (COMP && CL == 1 && NW * RUPW == UPS && NW * TOKS * 8 * 4 <= NSTAGE * SST),
                  "fused piece geometry");
    static_assert(TOKR == 16 * MT || (TOKR == 8 && MT == 1), "TOKR must match MT");
    static_assert(!(COMP && CL > 1), "escape warps are not clustered");
    static_assert(ESCG || NESC <= NSTAGE, "escape warps take turns on stages");
    static_assert(!ESCG || (COMP && !FUSE && CL == 1), "global-x escapes: compressed, unfused, unclustered");
    static_assert(!FULL || TOKR == 16 * MT || TOKR == 8, "FULL: bad TOKR");
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
    if constexpr (C::FULL) {
        M = TOKR;
    }
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
            mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
                mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
                mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
            const int cnt = (dead || C::ESCOFF) ? 0 : tab[0];
            if constexpr (C::ESCG) {
                // x is the kernel's input in global memory (written before launch, read-only
                // here), so these reads need no ordering with the ring; rws carries its pointer.
                const __nv_bfloat16* xg = reinterpret_cast<const __nv_bfloat16*>(rws);
                const int kbeg = gs0 * KST, kend = (gs0 + nst) * KST;
                for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
                    const int e0 = tab[2 * pe];
                    const int kk = e0 & 0xFFFF;
                    if (lane < M && kk >= kbeg && kk < kend) {
                        cr[lane * CP + (e0 >> 16)] +=
                            __int_as_float(tab[2 * pe + 1]) * __bfloat162float(xg[(size_t)lane * K + kk]);
                    }
                }
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(cbar);
                }
                return;
            }
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
                    if constexpr (C::ESCFAST) {
                        // Plain loads: the full-barrier wait above and the proxy fence below
                        // are asm with memory clobbers, so nothing moves across them.
                        const int2* te = reinterpret_cast<const int2*>(tab);
                        const unsigned char* xg = gbase + (xs - sbase);
                        while (pe <= cnt) {
                            const int2 e = te[pe];
                            const int kk = e.x & 0xFFFF;
                            if (kk >= kend) {
                                break;
                            }
                            if (lane < M) {
                                const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
                                const uint32_t off = u * XU + lane * 128 + (((kc >> 3) ^ ((u * M + lane) & 7)) << 4)
                                                     + (kc & 7) * 2;
                                const float xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xg + off));
                                cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * xv;
                            }
                            ++pe;
                        }
                    }
                    while (!C::ESCFAST && pe <= cnt) {
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
    constexpr int NA = C::NA;
    float acc[NA][MT][NT][4];
#pragma unroll
    for (int c = 0; c < NA; ++c) {
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
        if constexpr (C::NOWD) {
            while (!mbar_test(full0 + 8 * slot, (st / NSTAGE) & 1)) {
            }
        } else if constexpr (C::NOVOTE) {
            mbar_wait_novote(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8);
        } else if constexpr (C::LITEWD) {
            if (!__all_sync(0xffffffffu, mbar_wait_lite(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {
                dead = true;
                break;
            }
        } else if (!__all_sync(0xffffffffu, mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {
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
            if (!C::LATE_ARRIVE && uu == UPW - 1) {
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
                        mma_bf16(acc[((uu * 4 + j) * HM) % NA][mt][nt], a[j][mt], b[nt][2 * j], b[nt][2 * j + 1]);
                        if constexpr (HL) {
                            mma_bf16(acc[((uu * 4 + j) * HM + 1) % NA][mt][nt], a[j][mt], bl[nt][2 * j],
                                     bl[nt][2 * j + 1]);
                        }
                    }
                }
            }
        }
        if constexpr (C::LATE_ARRIVE) {
            fence_proxy_async();
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty0 + 8 * slot);
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
                    for (int c = 1; c < NA; ++c) {
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
                for (int c = 1; c < NA; ++c) {
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
    if constexpr (C::FULL) {
        TORCH_CHECK(M == TOKR, "sk: FULL configs need M == TOKR");
    }
    if constexpr (C::ESCG) {
        TORCH_CHECK(x.is_contiguous(), "sk: global-x escapes need a contiguous x");
        rws = reinterpret_cast<const uint8_t*>(x.data_ptr());
    }
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
@@INSTANCES@@
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
#if SK_PDL
    sk_pdl_launch = ((cfg >> 16) & 1) != 0;
    cfg &= 0xffff;
#endif
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
    # Parameter types and (NTH, CPT) pairs come from the source so the instantiations cannot go
    # stale when cgemv_launch changes (they did once: a const uint4* stream argument).
    import re
    params = re.search(r"void cgemv_launch\(([^)]*)\)", GEMV_CUDA_SRC).group(1)
    ptypes = ", ".join(re.sub(r"\s*\b\w+$", "", p.strip()) for p in params.split(","))
    pairs = sorted({(int(a), int(b)) for a, b in re.findall(r"CGEMV_CASE\((\d+), (\d+)\)", GEMV_CUDA_SRC)})
    ginst = "".join(f"template void cgemv_launch<{nth}, {cpt}>({ptypes});\n" for nth, cpt in pairs)
    gemv = GEMV_CUDA_SRC.replace("}  // namespace", ginst + "}  // namespace", 1)
    return ("#define SK_DEVICE_ONLY 1\n#define GEMV_DEVICE_ONLY 1\n" + src + gemv + wd_source() + rt_source()
            + rd_source())


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
# router, sgemv on the raw BF16 unpooler slice. The 12-bit code is the one above, laid out in
# row-major planes instead of per tile.
# --------------------------------------------------------------------------------------------
_LAYER_THREADS = {(2304, 8192): 256, (8192, 1024): 64, (4096, 8192): 512, (1024, 4096): 128}


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
    # 64-token tiles and an 8-stage ring at m = 128 / 256 (6 stages at 128 tokens): the
    # independent reference's router runs 64 rows x 64 tokens per CTA, K split 8, 8 stages.
    "r64c8s8": (1, 64, 8, 1, True, 8),
    "r64c4s8": (1, 64, 4, 1, True, 8),
    "r128c8s6": (1, 128, 8, 1, True, 6),
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
    # The public entry's gq (pooler) and gqx (qkv_gate) recipes at m=128: b1n128c2 with an
    # 8-stage ring instead of 4, and a 3-way K split for qkv_gate. Ring depth is what wins:
    # pooler m=128 26.5 us at 8 stages vs 31.1 at 4, with or without the L2 policy (31.1 both),
    # cuBLAS 28.9 (4098aa2361554229).
    "b1n128c2s8h": (1, 128, 2, 1, False, 8),
    "b1n128c3s8h": (1, 128, 3, 1, False, 8),
}
# Configs whose TMA loads carry L2 policies: weights evict_first, x evict_last.
WD_POL = {nm for nm in WD_GEOM if nm.endswith("h")}
# name -> VAR bits of the kernel (WV_* in _WD_SRC), 0 when absent. WV_PACK configs take the
# packed router weights (router_pack()).
WV_NOPROD, WV_PACK, WV_PROMO64, WV_PUSH, WV_EARLY, WV_TAILSYNC = 1, 2, 4, 8, 32, 128
WV_NOSLEEP, WV_NOEMPTY, WV_XA, WV_TAILSPLIT, WV_TAILRLX = 256, 512, 1024, 2048, 4096
WV_PGROUP, WV_ACQCL, WV_SCALAR, WV_GRED, WV_GZERO = 8192, 16384, 32768, 65536, 131072
# Packed hi/lo box, 64 B promotion, push reduction and early prologue (KLUE) on the two router
# picks; paired against the pick in a91f2152d8974458 / ba85985b8c5e4969: -0.77 / -0.90 us at
# m = 128, -0.90 / -0.90 at 256, the push reduction alone -0.46 / -0.54 and -0.48 / -0.66.
# Dropping the producer warp lost (+1.74 / +1.25); the producer without the suspend-time hint
# was a null on these two (+0.10 / -0.05). r64c4s8a is the reference's m = 256 orientation
# (XA: x as the A operand, one m64n128k16 for hi and lo) on its geometry, without the hint
# (-0.37 on r64c4s8 KLUE in ba85985b8c5e4969).
# WV_TAILSYNC: a cluster barrier before exit, so no CTA exits with an outgoing st.async in flight
# (decision 2026-10-30: the PTX ISA does not say that is allowed; cost about +0.3 us in
# a91f2152d8974458; +0.58 us at m = 128 on the judge). WV_TAILSPLIT keeps that barrier but arrives
# right after the receive wait, so the output stores no longer drain inside a release before
# it (argument: docs/proj-port-guide.md, "Split exit barrier").
_WD_KLUE = WV_PACK | WV_PROMO64 | WV_PUSH | WV_EARLY | WV_TAILSPLIT | WV_TAILRLX
WD_GEOM.update({"r64c8s8k": WD_GEOM["r64c8s8"], "r128c8s6k": WD_GEOM["r128c8s6"], "r64c4s8a": WD_GEOM["r64c4s8"]})
WD_VAR = {"r64c8s8k": _WD_KLUE, "r128c8s6k": _WD_KLUE, "r64c4s8a": _WD_KLUE | WV_NOSLEEP | WV_XA}
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

# --------------------------------------------------------------------------------------------
# CUTLASS sm90 GEMM (header-only templates from the CUTLASS copy in the FlashInfer package),
# swap-AB: D^T = W x^T with A = W [N, K] row-major, B = x as a K x M column-major matrix, D
# column-major [N, M] = out [M, N] row-major. Pingpong, persistent scheduler, tile 128 W rows x
# 64 tokens x 64, cluster 1: the public entry's pooler m=256 kernel. The sweep in job
# 8bea43d605c74546 (11 tiles / schedules / tile schedulers, stream-K and split-K included, at
# T = 128 and 256 on every layer shape plus the decode layer's FP32 pooler+router concat) found
# no other shape where any CUTLASS config beats cuBLAS. Pooler m=256 has 4096 / 128 * 256 / 64
# = 128 tiles, one wave on 132 SMs.
# name -> {(layer, bucket)} it is offered for.
CUT_USE = {"cut128x64": {("pooler", 256)}}
CUT_SRC = r"""
// Instantiates NVIDIA CUTLASS 3.x templates (header-only, BSD-3-Clause), composed as in CUTLASS's
// examples/49_hopper_gemm_with_collective_builder.
#include <cuda_runtime.h>
#include <cstdint>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
namespace {
using EA = cutlass::bfloat16_t; using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t; using LB = cutlass::layout::ColumnMajor;
using EC = cutlass::bfloat16_t; using LC = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, EC, LC, 8, EC, LC, 8,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, 8, EB, LB, 8, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi,
                                                    cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}  // namespace

// Kept free of torch headers: with them this unit must follow torch's C++ standard and ran out of
// memory in a 6 GB compile check. The torch wrapper is CUT_CPP.
int cut_run(const void* W, const void* x, void* out, void* ws, int64_t wsb, int N, int M, int K,
            cudaStream_t st) {
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(N, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(M, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(N, M, 1));
  auto sD = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, cute::make_shape(N, M, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {N, M, K, 1},
      {static_cast<const EA*>(W), sA, static_cast<const EB*>(x), sB},
      {{1.f, 0.f}, static_cast<const EC*>(out), sC, static_cast<EC*>(out), sD}};
  Gemm gemm;
  if ((int64_t)Gemm::get_workspace_size(args) > wsb) return 1;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 2;
  if (gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) return 3;
  if (gemm.run(st) != cutlass::Status::kSuccess) return 4;
  return 0;
}
"""
CUT_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int cut_run(const void* W, const void* x, void* out, void* ws, int64_t wsb, int N, int M, int K,
            cudaStream_t st);
void cut_fwd(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws) {
  TORCH_CHECK(W.is_contiguous() && x.is_contiguous() && out.is_contiguous() && ws.is_contiguous(),
              "cut_fwd: layout");
  const int N = W.size(0), K = W.size(1), M = x.size(0);
  TORCH_CHECK(x.size(1) == K && out.size(0) == M && out.size(1) == N, "cut_fwd: shapes");
  int rc = cut_run(W.data_ptr(), x.data_ptr(), out.data_ptr(), ws.data_ptr(), ws.numel(), N, M, K,
                   at::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(rc == 0, "cut_fwd: CUTLASS status ", rc);
}
"""

# --------------------------------------------------------------------------------------------
# Direct cuBLASLt at m = 128 / 256 on the four BF16 layers. SGLang's dispatch (our "cublas"
# candidate) takes the library heuristic's first answer for one layout; here prepare() lists the
# heuristic answers for four layouts (W or a K-major copy, column or row order) under several
# workspace caps, screens them from a cold L2, expands the best few (stages, FP32-reduced
# split-K, swizzle, custom option, cluster shape), self-tests the fastest against FP64 and
# times them against the existing pick. Only FP32 accumulation, and split-K only when reduced
# in FP32. Plans are fixed to one m, so they serve exactly m = 128 and 256.
# --------------------------------------------------------------------------------------------
# One host-compiled translation unit: torch headers, cublasLt.h and the plan code together.
LT_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Direct cuBLASLt plans for out[m, n] = x[m, k] W[n, k]^T, BF16 in and out, FP32 compute.
// Host code only; the part above lt_run compiles without torch for checking.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <vector>

namespace {

using LtCfg = std::array<int64_t, 9>;

// Config attributes 0-8 in cublasLt.h order: algo id, tile, split-k, reduction scheme, swizzle,
// custom option, stages, inner shape, cluster shape (the last two uint16). Cast from the
// number, not named, because the inner and cluster names are absent from some headers.
cublasLtMatmulAlgoConfigAttributes_t cfg_attr(int i) {
    return (cublasLtMatmulAlgoConfigAttributes_t)i;
}

LtCfg cfg_of(const cublasLtMatmulAlgo_t& a) {
    LtCfg c{};
    for (int i = 0; i < 9; ++i) {
        size_t wr = 0;
        if (i >= 7) {
            uint16_t v = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&a, cfg_attr(i), &v, sizeof(v), &wr);
            c[i] = v;
        } else {
            uint32_t v = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&a, cfg_attr(i), &v, sizeof(v), &wr);
            c[i] = v;
        }
    }
    return c;
}

bool cfg_set(cublasLtMatmulAlgo_t& a, int i, int64_t value) {
    if (i >= 7) {
        uint16_t v = (uint16_t)value;
        return cublasLtMatmulAlgoConfigSetAttribute(&a, cfg_attr(i), &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
    }
    uint32_t v = (uint32_t)value;
    return cublasLtMatmulAlgoConfigSetAttribute(&a, cfg_attr(i), &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
}

uint32_t cap_u32(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
    uint32_t v = 0;
    size_t wr = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, &v, sizeof(v), &wr) != CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    return v;
}

std::vector<uint32_t> cap_list(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
    size_t bytes = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, nullptr, 0, &bytes) != CUBLAS_STATUS_SUCCESS || bytes == 0) {
        return {};
    }
    std::vector<uint32_t> v(bytes / sizeof(uint32_t));
    if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, v.data(), bytes, &bytes) != CUBLAS_STATUS_SUCCESS) {
        return {};
    }
    return v;
}

cublasLtHandle_t g_handle = nullptr;

struct LtPlan {
    int m = 0, n = 0, k = 0, layout = 0;
    size_t ws_cap = 0;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    std::vector<cublasLtMatmulAlgo_t> algos;
    std::vector<size_t> ws;
    std::vector<float> waves;
    std::set<LtCfg> seen;
    int rejected = 0;

    ~LtPlan() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (la) cublasLtMatrixLayoutDestroy(la);
        if (lb) cublasLtMatrixLayoutDestroy(lb);
        if (ld) cublasLtMatrixLayoutDestroy(ld);
    }

    // Accepts an algorithm only with FP32 accumulation and either no split-K or a split-K
    // reduced in the compute type: the in-place and output-type schemes sum BF16-rounded
    // partials, which SGLang's own split-K (splitKreduce in FP32) does not.
    bool add(const cublasLtMatmulAlgo_t& a) {
        LtCfg c = cfg_of(a);
        if (!seen.insert(c).second) return false;
        if (c[2] > 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) { ++rejected; return false; }
        if (c[2] <= 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_NONE && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) {
            ++rejected;
            return false;
        }
        uint64_t flags = 0;
        size_t wr = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &flags, sizeof(flags), &wr)
            != CUBLAS_STATUS_SUCCESS) {
            ++rejected;
            return false;
        }
        if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F
            || (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)) {
            ++rejected;
            return false;
        }
        cublasLtMatmulHeuristicResult_t r;
        std::memset(&r, 0, sizeof(r));
        if (cublasLtMatmulAlgoCheck(g_handle, op, la, lb, ld, ld, &a, &r) != CUBLAS_STATUS_SUCCESS
            || r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > ws_cap) {
            ++rejected;
            return false;
        }
        algos.push_back(a);
        ws.push_back(r.workspaceSize);
        waves.push_back(r.wavesCount);
        return true;
    }
};

std::vector<std::unique_ptr<LtPlan>> g_plans;

LtPlan* plan_at(int64_t p) {
    return (p >= 0 && p < (int64_t)g_plans.size()) ? g_plans[p].get() : nullptr;
}

}  // namespace

int64_t lt_version() {
    return (int64_t)cublasLtGetVersion();
}

// Layouts (cuBLASLt is column-major unless ORDER_ROW is set):
//   0  column-major, D^T[n, m] = op_T(W as k x n) * x^T(k x m)      W as given
//   1  column-major, D^T[n, m] = Wt(n x k, ld n) * x^T(k x m)        Wt = W^T contiguous [k, n]
//   2  row order,    D[m, n] = x(m x k) * op_T(W as n x k)            W as given
//   3  row order,    D[m, n] = x(m x k) * Wt(k x n)                   Wt
// D is the same row-major out[m, n] buffer in all four.
int64_t lt_plan_new(int64_t m, int64_t n, int64_t k, int64_t layout, int64_t ws_cap) {
    if (!g_handle && cublasLtCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return -1;
    auto p = std::make_unique<LtPlan>();
    p->m = (int)m;
    p->n = (int)n;
    p->k = (int)k;
    p->layout = (int)layout;
    p->ws_cap = (size_t)ws_cap;
    if (cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return -2;
    const bool row = layout >= 2;
    const bool packed = layout % 2 == 1;
    cublasOperation_t ta = CUBLAS_OP_N, tb = CUBLAS_OP_N;
    if (!row) {
        ta = packed ? CUBLAS_OP_N : CUBLAS_OP_T;
        if (packed) {
            cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, n, k, n);
        } else {
            cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, k, n, k);
        }
        cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, k, m, k);
        cublasLtMatrixLayoutCreate(&p->ld, CUDA_R_16BF, n, m, n);
    } else {
        tb = packed ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, m, k, k);
        if (packed) {
            cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, k, n, n);
        } else {
            cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, n, k, k);
        }
        cublasLtMatrixLayoutCreate(&p->ld, CUDA_R_16BF, m, n, n);
        cublasLtOrder_t ord = CUBLASLT_ORDER_ROW;
        for (auto l : {p->la, p->lb, p->ld}) {
            if (!l || cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &ord, sizeof(ord))
                       != CUBLAS_STATUS_SUCCESS) {
                return -3;
            }
        }
    }
    if (!p->la || !p->lb || !p->ld) return -4;
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) != CUBLAS_STATUS_SUCCESS) {
        return -5;
    }
    g_plans.push_back(std::move(p));
    return (int64_t)g_plans.size() - 1;
}

// Heuristic candidates under a workspace cap; returns how many new ones were accepted.
int64_t lt_heuristic(int64_t plan, int64_t ws_bytes, int64_t req) {
    LtPlan* p = plan_at(plan);
    if (!p) return -1;
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return -2;
    uint64_t wsb = (uint64_t)std::min<int64_t>(ws_bytes, (int64_t)p->ws_cap);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
    std::vector<cublasLtMatmulHeuristicResult_t> res((size_t)req);
    int got = 0;
    int64_t added = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_handle, p->op, p->la, p->lb, p->ld, p->ld, pref, (int)req, res.data(), &got)
        == CUBLAS_STATUS_SUCCESS) {
        for (int i = 0; i < got; ++i) {
            if (res[i].state == CUBLAS_STATUS_SUCCESS && p->add(res[i].algo)) ++added;
        }
    }
    cublasLtMatmulPreferenceDestroy(pref);
    return added;
}

// Neighbours of accepted candidate idx: its other stage counts, split-K 2-8 reduced in FP32,
// swizzle and custom options, cluster shapes. Returns how many were accepted.
int64_t lt_expand(int64_t plan, int64_t idx, int64_t max_add) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return -1;
    const cublasLtMatmulAlgo_t base = p->algos[(size_t)idx];
    const size_t before = p->algos.size();
    auto room = [&]() { return (int64_t)(p->algos.size() - before) < max_add; };
    for (uint32_t s : cap_list(base, CUBLASLT_ALGO_CAP_STAGES_IDS)) {
        if (!room()) break;
        auto a = base;
        if (cfg_set(a, 6, s)) p->add(a);
    }
    int32_t sk = 0;
    size_t wr = 0;
    cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &sk, sizeof(sk), &wr);
    if (sk && (cap_u32(base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for (int s : {2, 3, 4, 6, 8}) {
            if (!room()) break;
            auto a = base;
            if (cfg_set(a, 2, s) && cfg_set(a, 3, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) p->add(a);
        }
    }
    const int swz = (int)cap_u32(base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
    const int cust = (int)std::min<uint32_t>(cap_u32(base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX), 15u);
    for (int z = 0; z <= swz; ++z) {
        for (int c = 0; c <= cust; ++c) {
            if (!room()) break;
            auto a = base;
            if (cfg_set(a, 4, z) && cfg_set(a, 5, c)) p->add(a);
        }
    }
    for (int cl = 0; cl <= 24; ++cl) {
        if (!room()) break;
        auto a = base;
        if (cfg_set(a, 8, cl)) p->add(a);
    }
    return (int64_t)(p->algos.size() - before);
}

// Every tile of candidate idx's algorithm, at its stage count with no split-K, swizzle or
// custom option, under each cluster shape id in clusters. Returns how many were accepted.
int64_t lt_sweep(int64_t plan, int64_t idx, std::vector<int64_t> clusters, int64_t max_add) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return -1;
    cublasLtMatmulAlgo_t base = p->algos[(size_t)idx];
    cfg_set(base, 2, 1);
    cfg_set(base, 3, CUBLASLT_REDUCTION_SCHEME_NONE);
    cfg_set(base, 4, 0);
    cfg_set(base, 5, 0);
    const size_t before = p->algos.size();
    for (uint32_t t : cap_list(base, CUBLASLT_ALGO_CAP_TILE_IDS)) {
        for (int64_t cl : clusters) {
            if ((int64_t)(p->algos.size() - before) >= max_add) return (int64_t)(p->algos.size() - before);
            auto a = base;
            if (cfg_set(a, 1, t) && cfg_set(a, 8, cl)) p->add(a);
        }
    }
    return (int64_t)(p->algos.size() - before);
}

// Adds one config by value (all nine fields); returns its index or -1.
int64_t lt_add_config(int64_t plan, std::vector<int64_t> c) {
    LtPlan* p = plan_at(plan);
    if (!p || c.size() != 9) return -1;
    cublasLtMatmulAlgo_t a;
    std::memset(&a, 0, sizeof(a));
    if (cublasLtMatmulAlgoInit(g_handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                               CUDA_R_16BF, (int)c[0], &a) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    for (int i = 1; i < 9; ++i) {
        if (!cfg_set(a, i, c[i]) && c[i] != 0) return -1;
    }
    LtCfg want{};
    for (int i = 0; i < 9; ++i) want[i] = c[i];
    for (size_t j = 0; j < p->algos.size(); ++j) {
        if (cfg_of(p->algos[j]) == want) return (int64_t)j;
    }
    if (!p->add(a)) return -1;
    return cfg_of(p->algos.back()) == want ? (int64_t)p->algos.size() - 1 : -1;
}

int64_t lt_count(int64_t plan) {
    LtPlan* p = plan_at(plan);
    return p ? (int64_t)p->algos.size() : -1;
}

int64_t lt_rejected(int64_t plan) {
    LtPlan* p = plan_at(plan);
    return p ? (int64_t)p->rejected : -1;
}

// The nine config fields, then workspace bytes and 1000 x the reported waves.
std::vector<int64_t> lt_info(int64_t plan, int64_t idx) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return {};
    LtCfg c = cfg_of(p->algos[(size_t)idx]);
    std::vector<int64_t> v(c.begin(), c.end());
    v.push_back((int64_t)p->ws[(size_t)idx]);
    v.push_back((int64_t)(1000.0f * p->waves[(size_t)idx]));
    return v;
}

// w is W [n, k] for layouts 0 / 2 and Wt [k, n] for 1 / 3. Returns the cuBLAS status.
int64_t lt_run_raw(int64_t plan, int64_t idx, int64_t w, int64_t x, int64_t out, int64_t ws, int64_t ws_bytes,
                   int64_t stream) {
    LtPlan* p = plan_at(plan);
    if (!p || idx < 0 || idx >= (int64_t)p->algos.size()) return -1;
    if ((int64_t)p->ws[(size_t)idx] > ws_bytes) return -2;
    const float alpha = 1.0f, beta = 0.0f;
    const bool row = p->layout >= 2;
    const void* a = row ? (const void*)x : (const void*)w;
    const void* b = row ? (const void*)w : (const void*)x;
    return (int64_t)cublasLtMatmul(g_handle, p->op, &alpha, a, p->la, b, p->lb, &beta, (void*)out, p->ld, (void*)out,
                                   p->ld, &p->algos[(size_t)idx], (void*)ws, p->ws[(size_t)idx],
                                   (cudaStream_t)stream);
}

int64_t lt_run(int64_t plan, int64_t idx, at::Tensor w, at::Tensor x, at::Tensor out, at::Tensor ws) {
  return lt_run_raw(plan, idx, (int64_t)w.data_ptr(), (int64_t)x.data_ptr(), (int64_t)out.data_ptr(),
                    (int64_t)ws.data_ptr(), ws.numel(), (int64_t)at::cuda::getCurrentCUDAStream().stream());
}
"""
LT_FUNCS = ["lt_version", "lt_plan_new", "lt_heuristic", "lt_expand", "lt_sweep", "lt_add_config", "lt_count", "lt_rejected",
            "lt_info", "lt_run"]
LT_TARGETS = tuple((layer, m) for m in (128, 256) for layer in BF16_LAYERS)
LT_LAYOUTS = (0, 1, 2, 3)
LT_WS_BYTES = 32 << 20
LT_WS_CAPS = (0, 1 << 20, LT_WS_BYTES)
LT_HEUR_REQ = 32
LT_SCREEN_REPS = 6
LT_EXPAND_TOP = 4
LT_EXPAND_MAX = 48
LT_FINALISTS = 5
# The 6-rep screen's ranking among near-equal configs is noise (pooler m=128 picks moved
# 1.4 us between preparations in d6eaf1e73db047a5); the leaders are re-timed at more reps.
LT_RESCREEN = 24
LT_RESCREEN_REPS = 12
# Tile sweep: the best LT_SWEEP_TOP plans' algorithms over all their tiles under these cluster
# shape ids (2 = 1x1x1, 3 = 2x1x1, 4 = 4x1x1, 5 = 1x2x1, 0 = automatic), then the best
# LT_SWEEP_TILES tiles under every id up to 24.
LT_SWEEP_TOP = 2
LT_SWEEP_CLUSTERS = (0, 2, 3, 5)
LT_SWEEP_MAX = 640
LT_SWEEP_TILES = 4
# Stable final choice (docs/plan-2026-10-30-tuner-stability.md, design 1): the pooler m = 128
# pick moved by up to 1.1 us between preparations of one file (44fa08c6d2f44b44), because a
# one-pass median picked among near-equal configs. Now a pinned preference per shape, the
# configuration that led most earlier preparations (bf6c825b..6cbd8b98 and the P1 benches;
# "base" where cuBLASLt never won clearly), is kept unless another candidate saves more than
# max(STABLE_MARGIN_US, 2 x the preference's round MAD) in median paired difference and is
# faster in all rounds but one. Values: (layout, the nine lt_info config fields).
LT_PREF = {
    # The donor's pin (alygg77 kernels/muse_bf16_proj.py _KLARGE_CONFIGS: "nn" on a K-major copy,
    # tile 15 = 64x64, custom 4, cluster id 2): -0.78 / -1.36 / -1.22 us against the old
    # preference (L0 tile 134, cluster 7) in a63fa0c3 / 6d5554de / a8de7b16.
    ("pooler", 128): (1, [66, 15, 1, 0, 0, 4, 35, 0, 2]),
    ("unpooler", 128): (3, [66, 18, 1, 0, 0, 2, 35, 0, 5]),
    ("qkv_gate", 256): (0, [66, 124, 1, 0, 0, 0, 35, 0, 2]),
    ("pooler", 256): (1, [66, 18, 1, 0, 0, 0, 35, 0, 2]),
    ("unpooler", 256): (1, [66, 20, 1, 0, 0, 4, 35, 0, 3]),
    # o 256 flipped between base and an Lt config in two preparations of one file
    # (6cc72e840403402b: +0.48 us in 8 of 9 rounds against +0.26 in 8 of 9, margin 0.30-0.32).
    ("o", 256): (3, [66, 20, 1, 0, 0, 4, 35, 0, 3]),
}
STABLE_ROUNDS = 9
STABLE_REPS = 3
STABLE_MARGIN_US = 0.3
# cuBLASLt's heuristic assumes 256-byte aligned operands unless told otherwise.
LT_ALIGN = 256


def wd_smem(wg, bn, clk, clr, hl, nst, var=0):
    sb = wg * (2 if hl else 1) * 8192 + bn * 128
    recv = 16 + bn * wg * 128 * 2 if var & 8 else 0
    return 1024 + nst * sb + 16 * nst + recv


def router_pack(hi, lo):
    """Router hi / lo [N, K] bf16 as [N/64 * K/64 * 128, 64]: per (64-row block, 64-wide k tile)
    the 64 hi rows then the 64 lo rows, so one 128-row TMA box loads both (WV_PACK)."""
    n, k = hi.shape
    blocks = [p.view(n // 64, 64, k // 64, 64).permute(0, 2, 1, 3) for p in (hi, lo)]
    return torch.stack(blocks, dim=2).contiguous().view(-1, 64)


def wd_geom(name):
    return WD_GEOM[WD_ABL[name][0]] if name in WD_ABL else WD_GEOM[name]


def wd_fits(name, n, k):
    wg, bn, clk, clr, hl, nst = wd_geom(name)
    return n % (64 * wg * clr) == 0 and k % 64 == 0 and k // 64 >= clk


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

// A receiver's arrive must follow its receive wait, so a peer's exit wait orders every push into
// this CTA before that peer exits (docs/proj-port-guide.md, split exit barrier).
template <bool RELAXED>
__device__ __forceinline__ void tail_arrive() {
    if constexpr (RELAXED) {
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    } else {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    }
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
//
// VAR bits change one mechanism each (0 is the build judged through 6cbd8b981a5d4c3f); they
// isolate the design differences against the independent reference's router (docs/runs.md,
// 2026-10-30 round 2).
constexpr int WV_NOPROD = 1;     // no producer warp: thread 0 refills a stage after its empty barrier
constexpr int WV_PACK = 2;       // router hi and lo rows packed so one 128-row TMA box carries both
constexpr int WV_PROMO64 = 4;    // 64-byte L2 promotion on both tensor maps (host side)
constexpr int WV_PUSH = 8;       // partials pushed by st.async to the owning rank, no pull or exit barrier
constexpr int WV_SPIN = 16;      // full-barrier waits without the clock64 watchdog
constexpr int WV_EARLY = 32;     // the init thread issues the prologue loads before the CTA barrier
constexpr int WV_DUP = 64;       // no effect: a separately compiled copy, for an independent null
constexpr int WV_TAILSYNC = 128; // with PUSH: keep a cluster barrier before exit
constexpr int WV_NOSLEEP = 256;  // producer empty-barrier waits without the suspend-time hint
// Bench diagnostic only, never dispatched by prepare(): with NOPROD, thread 0 refills a stage
// right after its own wgmma.wait_group, without the empty barrier (the reference's order). It
// drops a barrier, so it may not ship without a PTX memory-model argument and review.
constexpr int WV_NOEMPTY = 512;
// x tile as the wgmma A operand (64 tokens) and the packed [hi; lo] box as one 128-wide B
// operand: one m64n128k16 per k16 step instead of two m64n64k16, hi and lo summed once at the
// end (the reference's m = 256 kernel orientation). Needs PACK, PUSH, WG 1, BN 64.
constexpr int WV_XA = 1024;
// With PUSH, instead of TAILSYNC: each consumer arrives on the exit cluster barrier as soon as its
// receive wait completes and waits on it just before exit, so the reduction and output stores
// overlap the barrier; TAILRLX makes that arrive relaxed.
constexpr int WV_TAILSPLIT = 2048;
constexpr int WV_TAILRLX = 4096;
// Bench-only toggles of the reference router's remaining differences (docs/runs.md, round 2):
// PGROUP a 4-warp producer group (256-thread CTA at WG 1) of which one thread issues; ACQCL the
// consumer full and producer empty waits as try_wait.parity.acquire.cluster (with SPIN, the
// reference's one-instruction branch loop); SCALAR the XA epilogue's float2 stores as scalars.
constexpr int WV_PGROUP = 8192;
constexpr int WV_ACQCL = 16384;
constexpr int WV_SCALAR = 32768;
// The K split without a cluster: CTAs of one output tile add their FP32 partials into a
// zero-kept global accumulator with red.add, and the last of them (a per-tile counter) copies
// the tile to out and zeroes it again. No co-residency is needed, so the split can exceed the
// 8-CTA portable cluster and the grid can exceed one wave. Mechanism of the rk router in
// alygg77's kernels/muse_bf16_proj.py, re-implemented here with a last-arriver join instead of
// its grid-wide barrier; ordering as CUDA's threadFenceReduction sample (docs/proj-port-guide.md,
// "Global split-K join").
constexpr int WV_GRED = 65536;
// With GRED: out is zeroed by a memset node ahead of the kernel (stream order) and the partials
// are added into it directly, so the reds are the CTA's last memory operations and there is no
// counter, fence or last-arriver pass.
constexpr int WV_GZERO = 131072;
constexpr int GRED_TOK = 256;
constexpr int GRED_N = 256;
constexpr int GRED_TILES = 64;
__device__ __align__(16) float wd_gred_acc[GRED_TOK * GRED_N];
__device__ unsigned int wd_gred_cnt[GRED_TILES];

__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ void mbar_spin_cluster(uint32_t bar, uint32_t parity) {
    asm volatile("{\n.reg .pred p;\nwd_spin:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra wd_spin;\n}" :: "r"(bar), "r"(parity) : "memory");
}

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, int VAR = 0>
struct WdCfg {
    static constexpr bool NOPROD = (VAR & WV_NOPROD) != 0;
    static constexpr bool PACK = (VAR & WV_PACK) != 0;
    static constexpr bool PUSH = (VAR & WV_PUSH) != 0;
    static constexpr int NCT = WG * 128;
    // With PUSH each rank owns TPR consecutive tokens and receives one float4 per consumer thread
    // per owned 8-token fragment column from every rank: CLK * (TPR / 8) * NCT * 16 bytes.
    static constexpr int TPR = BN / CLK;
    static constexpr int JPR = TPR / 8;
    static constexpr bool XA = (VAR & WV_XA) != 0;
    // XA: each consumer thread holds 32 summed floats as 8 float4 units; every rank receives
    // NCT * 8 / CLK units from every rank (slot order in wd_kernel's XA epilogue).
    static constexpr int XUNITS = NCT * 8 / CLK;
    static constexpr int RECV = PUSH ? (XA ? CLK * XUNITS * 16 : CLK * JPR * NCT * 16) : 0;
    static constexpr int NACC = XA ? 64 : BN / 2;
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
    static constexpr bool PGROUP = (VAR & WV_PGROUP) != 0;
    static constexpr bool GRED = (VAR & WV_GRED) != 0;
    // Cluster size actually launched: GRED splits K without one.
    static constexpr int LCL = GRED ? 1 : CLK * CLR;
    static constexpr int THREADS = NCT + (NOPROD ? 0 : (PGROUP ? 128 : 32));
    static constexpr int SMEM = 1024 + RING + 16 * NST + (PUSH ? 16 + RECV : 0);
    static_assert(!PUSH || (CLR == 1 && CLK > 1 && TPR % 8 == 0 && HL), "wd: push needs CLR 1, 8-token owners, f32 out");
    static_assert(!PACK || HL, "wd: packed weights are the router's hi/lo pairs");
    static_assert(!XA || (PACK && PUSH && WG == 1 && BN == 64 && (TPR == 8 || TPR % 16 == 0)), "wd: bad XA config");
    static_assert(!PGROUP || (!NOPROD && PUSH), "wd: the producer group is built for the push router only");
    static_assert((VAR & WV_SCALAR) == 0 || XA, "wd: SCALAR changes only the XA epilogue");
    static_assert(PBY <= RING, "wd: partial tile does not fit in the ring");
    static_assert(BN % 8 == 0 && BN <= 256, "wd: bad BN");
    // Each multicast slice must start on a 1024-byte swizzle atom so the slices tile exactly
    // like one box.
    static_assert(BN % CLR == 0 && XS % 8 == 0, "wd: bad x slice");
    static_assert(LCL <= 16, "wd: cluster too large");
    static_assert(!GRED || (HL && CLR == 1 && !PUSH && !NOPROD), "wd: GRED is built for the f32 router only");
    static_assert(SMEM <= 227 * 1024, "wd: shared memory over budget");
};

__device__ __forceinline__ void wd_st_async(uint32_t cluster_addr, float4 v, uint32_t cluster_bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                 :: "r"(cluster_addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(cluster_bar) : "memory");
}

__device__ __forceinline__ void wd_st_local(uint32_t addr, float4 v) {
    asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL, bool POL, int VAR = 0>
__global__ void __launch_bounds__((WdCfg<WG, BN, CLK, CLR, HL, NST, VAR>::THREADS), 1)
wd_kernel(const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap xmap, OutT* __restrict__ out,
          int M, int N, int K, int* diag) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST, VAR>;
    constexpr int CL = C::CL;
    constexpr bool NOPROD = C::NOPROD, PACK = C::PACK, PUSH = C::PUSH, XA = C::XA;
    constexpr int NACC = C::NACC;
    constexpr bool SPIN = (VAR & WV_SPIN) != 0, EARLY = (VAR & WV_EARLY) != 0;
    constexpr bool TSPLIT = (VAR & WV_TAILSPLIT) != 0;
    constexpr bool ACQCL = (VAR & WV_ACQCL) != 0, SCALAR = (VAR & WV_SCALAR) != 0;
    static_assert(!PUSH || sizeof(OutT) == 4, "wd: push writes f32");
    static_assert(!TSPLIT || (PUSH && (VAR & WV_TAILSYNC) == 0), "wd: split exit barrier replaces TAILSYNC");
    static_assert((VAR & WV_TAILRLX) == 0 || TSPLIT, "wd: TAILRLX needs TAILSPLIT");
    extern __shared__ __align__(1024) uint8_t wd_smem[];
    const uint32_t raw = smem_u32(wd_smem);
    const uint32_t sbase = (raw + 1023u) & ~1023u;
    float* part = reinterpret_cast<float*>(wd_smem + (sbase - raw));
    const uint32_t full0 = sbase + C::RING;
    const uint32_t empty0 = full0 + 8 * NST;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rank = C::GRED ? (int)(blockIdx.x % CL) : (CL > 1 ? (int)wd_cluster_rank() : 0);
    const int rk = rank % CLK;
    const int rr = rank / CLK;
    const int rowbase = ((int)(blockIdx.x / CL) * CLR + rr) * C::BM;
    const int tok0 = (int)blockIdx.y * BN;
    // Uneven split when CLK does not divide the k tiles (qkv_gate's 128 over 3 ranks).
    const int kt0 = rk * (K / 64) / CLK;
    const int ktl = (rk + 1) * (K / 64) / CLK - kt0;
    bool dead = false;
    const uint32_t rbar = empty0 + 8 * NST;
    const uint32_t recv = rbar + 16;
    // The thread that initializes the barriers and issues every TMA load.
    const int pt = NOPROD ? 0 : C::NCT;
    const int npro = ktl < NST ? ktl : NST;

    auto issue = [&](int i, uint64_t wpol, uint64_t xpol) {
        uint16_t mask = 0;
#pragma unroll
        for (int r = 0; r < CLR; ++r) {
            mask |= (uint16_t)(1u << (rk + CLK * r));
        }
        const int s = i % NST;
        const uint32_t fb = full0 + 8 * s;
        mbar_expect_tx(fb, C::SB);
        const uint32_t st = sbase + s * C::SB;
        const int kc = (kt0 + i) * 64;
#pragma unroll
        for (int g = 0; g < WG; ++g) {
            if constexpr (PACK) {
                // Packed rows: block (64-row block, k tile) is 64 hi rows then 64 lo rows of 64.
                tma_2d(st + g * C::NA * C::ATILE, &wmap, fb, 0, ((rowbase / 64 + g) * (K / 64) + kt0 + i) * 128);
            } else {
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
        }
        if constexpr (CLR > 1) {
            tma_2d_mc(st + C::ABY + rr * C::XS * 128, &xmap, fb, kc, tok0 + rr * C::XS, mask);
        } else if constexpr (POL) {
            tma_2d_pol(st + C::ABY, &xmap, fb, kc, tok0, xpol);
        } else {
            tma_2d(st + C::ABY, &xmap, fb, kc, tok0);
        }
    };

    if (tid == pt) {
        for (int s = 0; s < NST; ++s) {
            mbar_init(full0 + 8 * s, 1);
            mbar_init(empty0 + 8 * s, 4 * WG * CLR);
        }
        if constexpr (PUSH) {
            mbar_init(rbar, 1);
        }
        fence_mbar_init();
        if constexpr (PUSH) {
            // A peer's complete_tx may land before this expect (the relaxed cluster arrive does
            // not order it); the tx count then runs negative, which the mbarrier allows, and the
            // phase cannot complete before this arrive because the pending count is 1 until then.
            mbar_expect_tx(rbar, XA ? (CLK - 1) * C::XUNITS * 16 : (CLK - 1) * C::JPR * C::NCT * 16);
        }
        // Barriers this thread just initialized and fenced for the async proxy; consumers see
        // them after the CTA barrier below.
        if constexpr (EARLY && CLR == 1) {
            for (int i = 0; i < npro; ++i) {
                issue(i, POL ? l2_evict_first() : 0, POL ? l2_evict_last() : 0);
            }
        }
    }
    // Peers multicast into this CTA's ring and arrive on its barriers: they must be initialized
    // cluster-wide first.
    if constexpr (CLR > 1) {
        wd_cluster_sync();
    } else {
        __syncthreads();
    }
    if constexpr (PUSH) {
        // Pairs with the wait before the pushes: the receive barrier is initialized cluster-wide
        // (fence.mbarrier_init above, then relaxed arrive / acquire wait, as CUTLASS orders it).
        cluster_arrive();
    }
    if constexpr (NOPROD && !(EARLY && CLR == 1)) {
        if (tid == pt) {
            for (int i = 0; i < npro; ++i) {
                issue(i, POL ? l2_evict_first() : 0, POL ? l2_evict_last() : 0);
            }
        }
    }

    float acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
        acc[j] = 0.f;
    }
    if (!NOPROD && (C::PGROUP ? warp >= 4 * WG : warp == 4 * WG)) {
        if ((!C::PGROUP || warp == 4 * WG) && lane == 0) {
            const uint64_t wpol = POL ? l2_evict_first() : 0;
            const uint64_t xpol = POL ? l2_evict_last() : 0;
            for (int i = (EARLY && CLR == 1) ? npro : 0; i < ktl; ++i) {
                if (i >= NST) {
                    const uint32_t eb = empty0 + 8 * (i % NST);
                    const uint32_t ep = ((i / NST) - 1) & 1;
                    if constexpr (ACQCL && SPIN) {
                        mbar_spin_cluster(eb, ep);
                    } else if constexpr (ACQCL) {
                        mbar_wait_cluster(eb, ep, dead, diag, 1);
                    } else {
                        mbar_wait<(VAR & WV_NOSLEEP) == 0>(eb, ep, dead, diag, 1);
                    }
                }
                issue(i, wpol, xpol);
            }
        }
        __syncwarp();
    } else {
        const int g = warp >> 2;
        for (int i = 0; i < ktl; ++i) {
            const int s = i % NST;
            if constexpr (ACQCL && SPIN) {
                mbar_spin_cluster(full0 + 8 * s, (i / NST) & 1);
            } else if constexpr (ACQCL) {
                mbar_wait_cluster(full0 + 8 * s, (i / NST) & 1, dead, diag, 2);
            } else if constexpr (SPIN) {
                while (!mbar_test(full0 + 8 * s, (i / NST) & 1)) {
                }
            } else {
                mbar_wait<false>(full0 + 8 * s, (i / NST) & 1, dead, diag, 2);
            }
            const uint32_t a0 = sbase + s * C::SB + g * C::NA * C::ATILE;
            const uint32_t b0 = sbase + s * C::SB + C::ABY;
            wg_fence_acc<NACC>(acc);
            wg_fence();
#pragma unroll
            for (int kk = 0; kk < ((ABL & 1) ? 0 : 4); ++kk) {
                if constexpr (XA) {
                    wgmma_ss<128>(acc, wg_desc(b0 + 32 * kk), wg_desc(a0 + 32 * kk));
                } else {
                    wgmma_ss<BN>(acc, wg_desc(a0 + 32 * kk), wg_desc(b0 + 32 * kk));
                    if constexpr (HL) {
                        wgmma_ss<BN>(acc, wg_desc(a0 + C::ATILE + 32 * kk), wg_desc(b0 + 32 * kk));
                    }
                }
            }
            wg_commit();
            wg_fence_acc<NACC>(acc);
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
            if constexpr (NOPROD) {
                // The stage of iteration i - 1 is refilled only after all consumer warps have
                // released it on its empty barrier, as with the producer warp.
                const int j = i - 1 + NST;
                if (tid == pt && i > 0 && j < ktl) {
                    if constexpr ((VAR & WV_NOEMPTY) == 0) {
                        mbar_wait<false>(empty0 + 8 * (j % NST), ((j / NST) - 1) & 1, dead, diag, 1);
                    }
                    issue(j, POL ? l2_evict_first() : 0, POL ? l2_evict_last() : 0);
                }
            }
        }
        wg_wait<0>();
        wg_fence_acc<NACC>(acc);
        if constexpr (!PUSH) {
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
    }

    if constexpr (PUSH) {
        // Rank rk owns tokens [rk * TPR, (rk + 1) * TPR) of the tile. Fragment column j (8
        // tokens) of consumer thread c goes to rank j / JPR, slot ((src * JPR + j % JPR) * NCT
        // + c), so every slot has one writer. Remote accesses to a CTA's shared memory are only
        // these pushes into recv (never the ring) and their complete_tx on rbar; the CTA's
        // consumers wait for all (CLK - 1) peers' bytes before reading recv and before exiting,
        // so the CTA outlives every access to it. A peer pushes only after the cluster wait,
        // which follows this CTA's rbar init and expect_tx.
        cluster_wait();
        if constexpr (XA) {
            // Accumulator of thread (warp w, lane l): token rows r = 16 w + l / 4 and r + 8,
            // columns 8 j + 2 (l % 4) + {0, 1}; j < 8 are the hi products of weight row 8 j + .,
            // j >= 8 the lo products of the same rows. Unit u of the summed 32 floats:
            //   TPR >= 16: u = j, float4 (r, c), (r, c + 1), (r + 8, c), (r + 8, c + 1), all to
            //     rank 16 w / TPR, slot ((w % JW) * 32 + l) * 8 + u, JW = TPR / 16;
            //   TPR == 8: u < 4 holds token r at columns of j = 2u, 2u + 1 and goes to rank 2 w;
            //     u >= 4 holds token r + 8 (j = 2(u - 4), + 1) and goes to rank 2 w + 1; slot
            //     l * 4 + u % 4.
            // Every slot has one writer; recv is [source rank][XUNITS] float4.
            constexpr int JW = C::TPR >= 16 ? C::TPR / 16 : 1;
            float s[32];
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                s[e] = acc[e] + acc[32 + e];
            }
            if (tid < C::NCT) {
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    int dst, slot;
                    float4 v;
                    if constexpr (C::TPR >= 16) {
                        dst = warp * 16 / C::TPR;
                        slot = ((warp % JW) * 32 + lane) * 8 + u;
                        v = make_float4(s[4 * u], s[4 * u + 1], s[4 * u + 2], s[4 * u + 3]);
                    } else {
                        const int h = u >> 2;
                        const int j0 = 2 * (u & 3);
                        dst = 2 * warp + h;
                        slot = lane * 4 + (u & 3);
                        v = make_float4(s[4 * j0 + 2 * h], s[4 * j0 + 2 * h + 1], s[4 * j0 + 4 + 2 * h],
                                        s[4 * j0 + 5 + 2 * h]);
                    }
                    const uint32_t a = recv + (uint32_t)(rk * C::XUNITS + slot) * 16u;
                    if (dst == rk) {
                        wd_st_local(a, v);
                    } else {
                        wd_st_async(wd_mapa(a, (uint32_t)dst), v, wd_mapa(rbar, (uint32_t)dst));
                    }
                }
                named_bar(1, C::NCT);
                mbar_wait_cluster(rbar, 0, dead, diag, 4);
                if constexpr (TSPLIT) {
                    __syncwarp();
                    tail_arrive<(VAR & WV_TAILRLX) != 0>();
                }
                for (int it = tid; it < C::XUNITS; it += C::NCT) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                    for (int p = 0; p < CLK; ++p) {
                        const float4 q4 = wd_ld_local(recv + (uint32_t)(p * C::XUNITS + it) * 16u);
                        v.x += q4.x;
                        v.y += q4.y;
                        v.z += q4.z;
                        v.w += q4.w;
                    }
                    int t0, t1, c0, c1;
                    if constexpr (C::TPR >= 16) {
                        const int l = (it >> 3) & 31;
                        const int w = rk * JW + it / 256;
                        t0 = 16 * w + (l >> 2);
                        t1 = t0 + 8;
                        c0 = c1 = 8 * (it & 7) + 2 * (l & 3);
                    } else {
                        const int l = it >> 2;
                        t0 = t1 = 16 * (rk >> 1) + 8 * (rk & 1) + (l >> 2);
                        c0 = 16 * (it & 3) + 2 * (l & 3);
                        c1 = c0 + 8;
                    }
                    if constexpr (SCALAR) {
                        float* o0 = out + (size_t)(tok0 + t0) * N + rowbase + c0;
                        float* o1 = out + (size_t)(tok0 + t1) * N + rowbase + c1;
                        if (tok0 + t0 < M) {
                            o0[0] = v.x;
                            o0[1] = v.y;
                        }
                        if (tok0 + t1 < M) {
                            o1[0] = v.z;
                            o1[1] = v.w;
                        }
                    } else {
                        if (tok0 + t0 < M) {
                            *reinterpret_cast<float2*>(out + (size_t)(tok0 + t0) * N + rowbase + c0) = make_float2(v.x, v.y);
                        }
                        if (tok0 + t1 < M) {
                            *reinterpret_cast<float2*>(out + (size_t)(tok0 + t1) * N + rowbase + c1) = make_float2(v.z, v.w);
                        }
                    }
                }
            }
        } else if (tid < C::NCT) {
#pragma unroll
            for (int j = 0; j < BN / 8; ++j) {
                const int dst = j / C::JPR;
                const uint32_t a = recv + (uint32_t)((rk * C::JPR + j % C::JPR) * C::NCT + tid) * 16u;
                const float4 v = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
                if (dst == rk) {
                    wd_st_local(a, v);
                } else {
                    wd_st_async(wd_mapa(a, (uint32_t)dst), v, wd_mapa(rbar, (uint32_t)dst));
                }
            }
            // This thread's own slot stores are read by other consumer threads below.
            named_bar(1, C::NCT);
            mbar_wait_cluster(rbar, 0, dead, diag, 4);
            if constexpr (TSPLIT) {
                __syncwarp();
                tail_arrive<(VAR & WV_TAILRLX) != 0>();
            }
            for (int it = tid; it < C::JPR * C::NCT; it += C::NCT) {
                const int jl = it / C::NCT;
                const int c = it % C::NCT;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                // Ranks in order 0 .. CLK-1, as the pull reduction sums them.
#pragma unroll
                for (int p = 0; p < CLK; ++p) {
                    const float4 u = wd_ld_local(recv + (uint32_t)((p * C::JPR + jl) * C::NCT + c) * 16u);
                    v.x += u.x;
                    v.y += u.y;
                    v.z += u.z;
                    v.w += u.w;
                }
                const int row = rowbase + (c >> 7) * 64 + ((c >> 5) & 3) * 16 + ((c & 31) >> 2);
                const int gt = tok0 + 8 * (rk * C::JPR + jl) + 2 * (c & 3);
                if (gt < M) {
                    out[(size_t)gt * N + row] = v.x;
                    out[(size_t)gt * N + row + 8] = v.z;
                }
                if (gt + 1 < M) {
                    out[(size_t)(gt + 1) * N + row] = v.y;
                    out[(size_t)(gt + 1) * N + row + 8] = v.w;
                }
            }
        }
        if constexpr ((VAR & WV_TAILSYNC) != 0) {
            wd_cluster_sync();
        }
        if constexpr (TSPLIT) {
            if (tid >= C::NCT) {
                tail_arrive<(VAR & WV_TAILRLX) != 0>();
            }
            cluster_wait();
        }
        return;
    }

    // Also keeps a CTA's ring alive until every peer's multicast into it and every remote
    // arrive on it has landed: all of them precede the peers' arrival here.
    if constexpr (CL > 1 && !C::GRED) {
        wd_cluster_sync();
    } else {
        __syncthreads();
    }

    if constexpr (C::GRED && (VAR & WV_GZERO) != 0) {
        constexpr int C4G = C::BM / 4;
        for (int it = tid; it < ((ABL & 2) ? 0 : BN * C4G); it += C::THREADS) {
            const int t = it / C4G;
            const int c4 = it % C4G;
            if (tok0 + t < M) {
                red_add_v4(reinterpret_cast<float*>(out) + (size_t)(tok0 + t) * N + rowbase + 4 * c4,
                           wd_ld_local(sbase + (uint32_t)(t * C::RS + 4 * c4) * 4u));
            }
        }
        return;
    }
    if constexpr (C::GRED) {
        constexpr int C4G = C::BM / 4;
        __shared__ int gred_last;
        for (int it = tid; it < ((ABL & 2) ? 0 : BN * C4G); it += C::THREADS) {
            const int t = it / C4G;
            const int c4 = it % C4G;
            if (tok0 + t < M) {
                red_add_v4(wd_gred_acc + (size_t)(tok0 + t) * N + rowbase + 4 * c4,
                           wd_ld_local(sbase + (uint32_t)(t * C::RS + 4 * c4) * 4u));
            }
        }
        // Each thread's adds are ordered before thread 0's counter increment (fence, then CTA
        // barrier); the CTA that takes the counter to CLK - 1 sees every add of the tile after
        // its own fence. atomicInc wraps the counter back to 0 for the next launch.
        __threadfence();
        __syncthreads();
        const int tile = (int)blockIdx.y * (N / C::BM) + rowbase / C::BM;
        if (tid == 0) {
            gred_last = atomicInc(&wd_gred_cnt[tile], (unsigned)(CLK - 1)) == (unsigned)(CLK - 1);
        }
        __syncthreads();
        if (gred_last) {
            __threadfence();
            for (int it = tid; it < BN * C4G; it += C::THREADS) {
                const int t = it / C4G;
                const int c4 = it % C4G;
                if (tok0 + t < M) {
                    float4* a = reinterpret_cast<float4*>(wd_gred_acc + (size_t)(tok0 + t) * N + rowbase + 4 * c4);
                    const float4 v = __ldcg(a);
                    __stcg(a, make_float4(0.f, 0.f, 0.f, 0.f));
                    *reinterpret_cast<float4*>(out + (size_t)(tok0 + t) * N + rowbase + 4 * c4) = v;
                }
            }
        }
        return;
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
CUtensorMap make_2d_map(const void* p, uint64_t rows, uint64_t k, uint32_t box_rows,
                        CUtensorMapL2promotion promo = CU_TENSOR_MAP_L2_PROMOTION_L2_256B) {
    CUtensorMap map;
    cuuint64_t dims[2] = {k, rows};
    cuuint64_t strides[1] = {k * 2};
    cuuint32_t box[2] = {64, box_rows};
    cuuint32_t estr[2] = {1, 1};
    const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, promo,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return map;
}

template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL, bool POL, int VAR = 0>
void launch_wd(const at::Tensor& w, const at::Tensor& x, at::Tensor& out, int N, int K, int* diag) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST, VAR>;
    auto kern = wd_kernel<WG, BN, CLK, CLR, HL, NST, OutT, ABL, POL, VAR>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        if (C::LCL > 8) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        }
        configured = true;
    }
    const int M = (int)x.size(0);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2
                && x.size(1) == K, "wd: x must be a contiguous bf16 [M, K] tensor");
    if constexpr (C::PACK) {
        TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                    && w.size(1) == 64 && w.size(0) == (int64_t)(N / 64) * (K / 64) * 128,
                    "wd: packed w must be a contiguous bf16 [N/64 * K/64 * 128, 64] tensor");
    } else {
        TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                    && w.size(1) == K && w.size(0) >= (HL ? 2 : 1) * N,
                    "wd: w must be a contiguous bf16 [rows, K] tensor");
    }
    TORCH_CHECK(out.is_contiguous() && out.size(0) == M && out.size(1) == N && out.element_size() == sizeof(OutT),
                "wd: bad out");
    TORCH_CHECK(M >= 1 && N % (C::BM * CLR) == 0 && K % 64 == 0 && K / 64 >= CLK, "wd: shape does not tile");
    const CUtensorMapL2promotion promo =
        (VAR & WV_PROMO64) ? CU_TENSOR_MAP_L2_PROMOTION_L2_64B : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    const CUtensorMap wmap = C::PACK ? make_2d_map(w.data_ptr(), (uint64_t)w.size(0), 64, 128, promo)
                                     : make_2d_map(w.data_ptr(), (uint64_t)w.size(0), (uint64_t)K, 64, promo);
    const CUtensorMap xmap = make_2d_map(x.data_ptr(), (uint64_t)M, (uint64_t)K, C::XS, promo);
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    cudaLaunchConfig_t lc = {};
    if constexpr (C::GRED) {
        TORCH_CHECK(M <= GRED_TOK && N <= GRED_N && (N / C::BM) * ((M + BN - 1) / BN) <= GRED_TILES,
                    "wd: GRED workspace too small for this shape");
    }
    lc.gridDim = dim3((unsigned)(N / (C::BM * CLR) * C::CL), (unsigned)((M + BN - 1) / BN));
    lc.blockDim = dim3(C::THREADS);
    lc.dynamicSmemBytes = C::SMEM;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C::LCL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    lc.attrs = attr;
    lc.numAttrs = C::LCL > 1 ? 1 : 0;
    if constexpr ((VAR & WV_GZERO) != 0) {
        static_assert(C::GRED, "wd: GZERO needs GRED");
        C10_CUDA_CHECK(cudaMemsetAsync(op, 0, (size_t)M * N * sizeof(OutT), lc.stream));
    }
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
    pol = (WD_ABL[name][0] if name in WD_ABL else name) in WD_POL
    # globals().get: the layer's inliner (tools/dl_proj_inline.py) copies only the tables it knows.
    var = globals().get("WD_VAR", {}).get(name, 0)
    return (f"{wg}, {bn}, {clk}, {clr}, {str(hl).lower()}, {nst}, {'float' if hl else '__nv_bfloat16'}, {abl}, "
            f"{str(pol).lower()}, {var}")


def wd_source():
    """The wide kernel with its wgmma shapes and config table expanded; under SK_DEVICE_ONLY it
    emits explicit instantiations instead of the host dispatch."""
    for name, (wg, bn, clk, clr, hl, nst) in WD_GEOM.items():
        var = globals().get("WD_VAR", {}).get(name, 0)
        assert wd_smem(wg, bn, clk, clr, hl, nst, var) <= SMEM_BUDGET and clk * clr <= 16, name
        assert bn % clr == 0 and (bn // clr) % 8 == 0, name
        assert bn * (64 * wg + 4) * 4 <= nst * (wg * (2 if hl else 1) * 8192 + bn * 128), name
    # XA (VAR bit 1024) configs issue m64n128k16 whatever their BN.
    xa = any(globals().get("WD_VAR", {}).get(nm, 0) & 1024 for nm in WD_GEOM)
    wgmma = "".join(_wgmma_spec(bn) for bn in sorted({g[1] for g in WD_GEOM.values()} | ({128} if xa else set())))
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
# rt: the FP32 router at 1 to 32 tokens as BF16 hi + lo with FP32 accumulation. Every weight
# fragment goes from global memory straight into mma.sync A registers in one round trip (no
# shared-memory ring); K is split over the WK warps of a CTA and the S CTAs of a cluster; the
# per-CTA sums are pushed by st.async to P owner ranks, which add the S pieces in rank order.
# The design is that of the routertail_small kernel in NathenFernandes/muse-bf16-proj v18
# (commit 86aff75), MIT License, Copyright (c) 2026 Nathen Fernandes; this is our own
# implementation of it (packing, indexing and synchronization written here).
# name -> (S cluster ranks, WK warps per CTA, P owner ranks, exit cluster barrier mode: 0 none,
# 1 arrive after the stores, 2 / 3 arrive right after the receive wait with a release / relaxed
# arrive).
# --------------------------------------------------------------------------------------------
RT_GEOM = {
    # bench a91f2152d8974458, paired against the sk router picks: rt8w8 -1.39 us at m = 8, rt16w8
    # -0.83 at 16, rt16w8p8 -0.54 at 32 (null 0.02-0.08); the tuner picks per bucket.
    # Every shipped config keeps an exit cluster barrier: no CTA exits with an outgoing st.async
    # in flight (decision 2026-10-30, docs/proj-port-guide.md).
    "rt8w8": (8, 8, 4, 3),
    "rt16w8": (16, 8, 4, 3),
    "rt16w8p8": (16, 8, 8, 3),
}
RT_MT = (1, 2, 4)
RT_NAMES = list(RT_GEOM)

_RT_SRC = r"""
namespace {

__device__ __forceinline__ uint4 rt_ld_weights(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 rt_ld_x(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ void rt_st_async(uint32_t cluster_addr, float4 v, uint32_t cluster_bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                 :: "r"(cluster_addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(cluster_bar) : "memory");
}

__device__ __forceinline__ uint32_t rt_mapa(uint32_t addr, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

// Weights: rt_pack() order, per (16-row block, 64-wide k tile) 8 fragments x 32 lanes of uint4:
// fragment j < 4 is the hi tile of k chunk j, j >= 4 the lo tile of chunk j - 4. Within a tile
// the k order is permuted so a lane's x operand for all four chunks is 16 contiguous bf16
// (k = 16 t + 4 q + {0, 1} for b0, + {2, 3} for b1, t = lane & 3); the weights carry the same
// permutation, so each product pairs the same k.
template <int MT, int S, int WK, int P, int TAIL>
__global__ void __launch_bounds__(WK * 32)
rt_kernel(const uint4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int M,
          int* diag) {
    constexpr int KU = 128 / (S * WK);
    constexpr int NI = MT * 32;
    constexpr int OWN = NI / P;
    static_assert(KU >= 1 && 128 % (S * WK) == 0 && NI % P == 0 && P <= S, "rt: bad geometry");
    static_assert(TAIL >= 0 && TAIL <= 3, "rt: exit barrier mode 0 to 3");
    extern __shared__ __align__(16) unsigned char rt_smem[];
    float4* recv = reinterpret_cast<float4*>(rt_smem);
    float4* part = recv + S * OWN;
    const uint32_t rbar = smem_u32(part + WK * NI);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int rank = (int)blockIdx.x % S;
    const int rb = (int)blockIdx.x / S;
    bool dead = false;

    uint4 w[KU][8];
    uint4 xv[KU][MT][2];
#pragma unroll
    for (int u = 0; u < KU; ++u) {
        const int kt = (rank * WK + warp) * KU + u;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[u][j] = rt_ld_weights(wp + ((size_t)(rb * 128 + kt) * 8 + j) * 32 + lane);
        }
#pragma unroll
        for (int n = 0; n < MT; ++n) {
            // Rows past M read row M - 1; their products land in output columns never stored.
            const int t = min(n * 8 + (lane >> 2), M - 1);
            const __nv_bfloat16* p = x + (size_t)t * 8192 + kt * 64 + (lane & 3) * 16;
            xv[u][n][0] = rt_ld_x(p);
            xv[u][n][1] = rt_ld_x(p + 8);
        }
    }
    if (rank < P && tid == 0) {
        mbar_init(rbar, 1);
        fence_mbar_init();
        // The (S - 1) peers' pushes; this rank's own share is a local store. A push may land
        // before this expect; the phase still cannot complete before it (pending count 1 until
        // this arrive, negative tx count allowed).
        mbar_expect_tx(rbar, (S - 1) * OWN * 16);
    }
    // .aligned below needs the warp converged after the tid == 0 branch.
    __syncwarp();
    // The receive barrier is initialized cluster-wide before any push (relaxed arrive after the
    // mbarrier_init fence, acquire wait before the pushes, as CUTLASS orders it).
    cluster_arrive();

    float c[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                c[h][n][e] = 0.f;
            }
        }
    }
#pragma unroll
    for (int u = 0; u < KU; ++u) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
            const uint32_t b[8] = {xv[u][n][0].x, xv[u][n][0].y, xv[u][n][0].z, xv[u][n][0].w,
                                   xv[u][n][1].x, xv[u][n][1].y, xv[u][n][1].z, xv[u][n][1].w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                mma_bf16(c[0][n], reinterpret_cast<const uint32_t*>(&w[u][q]), b[2 * q], b[2 * q + 1]);
                mma_bf16(c[1][n], reinterpret_cast<const uint32_t*>(&w[u][4 + q]), b[2 * q], b[2 * q + 1]);
            }
        }
    }
#pragma unroll
    for (int n = 0; n < MT; ++n) {
        part[warp * NI + n * 32 + lane] = make_float4(c[0][n][0] + c[1][n][0], c[0][n][1] + c[1][n][1],
                                                      c[0][n][2] + c[1][n][2], c[0][n][3] + c[1][n][3]);
    }
    __syncthreads();
    cluster_wait();
    if (tid < NI) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < WK; ++k) {
            const float4 u = part[k * NI + tid];
            v.x += u.x;
            v.y += u.y;
            v.z += u.z;
            v.w += u.w;
        }
        // Slot (rank, tid % OWN) of owner tid / OWN has this one writer. Remote accesses to a
        // CTA's shared memory are only these pushes into recv and their complete_tx on rbar.
        // The own-rank share is a plain shared store: the PTX ISA does not say st.async may
        // target the issuing CTA, so that is not relied on (decision 2026-10-30).
        const int dst = tid / OWN;
        if (dst == rank) {
            recv[rank * OWN + tid % OWN] = v;
        } else {
            const uint32_t a = smem_u32(recv + rank * OWN + tid % OWN);
            rt_st_async(rt_mapa(a, (uint32_t)dst), v, rt_mapa(rbar, (uint32_t)dst));
        }
    }
    // The own-rank slots stored above are read by other threads (tid < OWN) below.
    __syncthreads();
    if constexpr (TAIL >= 2) {
        if (rank < P && tid < OWN) {
            mbar_wait_cluster(rbar, 0, dead, diag, 8);
        }
        __syncwarp();
        tail_arrive<TAIL == 3>();
    }
    if (rank < P && tid < OWN) {
        if constexpr (TAIL < 2) {
            mbar_wait_cluster(rbar, 0, dead, diag, 8);
        }
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int r = 0; r < S; ++r) {
            const float4 u = recv[r * OWN + tid];
            v.x += u.x;
            v.y += u.y;
            v.z += u.z;
            v.w += u.w;
        }
        // Item i = n * 32 + l is the m16n8 fragment of lane l in token group n: rows
        // 16 rb + (l >> 2) and + 8, tokens 8 n + 2 (l & 3) and + 1.
        const int i = rank * OWN + tid;
        const int l = i & 31;
        const int row = rb * 16 + (l >> 2);
        const int t = (i >> 5) * 8 + 2 * (l & 3);
        if (t < M) {
            out[(size_t)t * 256 + row] = v.x;
            out[(size_t)t * 256 + row + 8] = v.z;
        }
        if (t + 1 < M) {
            out[(size_t)(t + 1) * 256 + row] = v.y;
            out[(size_t)(t + 1) * 256 + row + 8] = v.w;
        }
    }
    if constexpr (TAIL >= 2) {
        cluster_wait();
    }
    if constexpr (TAIL == 1) {
        // No CTA exits with an outgoing st.async in flight: each owner arrives here only after
        // its receive wait saw every pushed byte, so passing this barrier implies all of this
        // CTA's pushes completed. Exiting earlier is undocumented (decision 2026-10-30: ship
        // the conforming form; +0.10 to +0.21 us in a91f2152d8974458).
        cluster_arrive();
        cluster_wait();
    }
}

template <int MT, int S, int WK, int P>
constexpr int rt_smem_bytes() {
    return (S * (MT * 32 / P) + WK * MT * 32) * 16 + 16;
}

#ifndef SK_DEVICE_ONLY
template <int MT, int S, int WK, int P, int TAIL>
void launch_rt(const at::Tensor& wp, const at::Tensor& x, at::Tensor& out, int* diag) {
    auto kern = rt_kernel<MT, S, WK, P, TAIL>;
    constexpr int SMEM = rt_smem_bytes<MT, S, WK, P>();
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if (S > 8) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        }
        configured = true;
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(16 * S);
    lc.blockDim = dim3(WK * 32);
    lc.dynamicSmemBytes = SMEM;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = S;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const uint4*>(wp.data_ptr()),
                                      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(),
                                      (int)x.size(0), diag));
}
#endif

}  // namespace

#ifdef SK_DEVICE_ONLY
@@RTINSTANCES@@
#else
void rt_fwd(int64_t cfg, at::Tensor wp, at::Tensor x, at::Tensor out, at::Tensor diag) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2
                && x.size(1) == 8192 && x.size(0) >= 1 && x.size(0) <= 32, "rt: x must be a contiguous bf16 [1..32, 8192]");
    TORCH_CHECK(wp.is_cuda() && wp.is_contiguous() && wp.numel() * wp.element_size() == 256 * 8192 * 4,
                "rt: wp must be the packed hi / lo router (8 MB)");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.size(0) == x.size(0)
                && out.size(1) == 256, "rt: out must be f32 [M, 256]");
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
    const int mt = x.size(0) <= 8 ? 1 : x.size(0) <= 16 ? 2 : 4;
    switch (cfg * 8 + mt) {
@@RTDISPATCH@@
    default:
        TORCH_CHECK(false, "rt: unknown config ", cfg);
    }
}
#endif
"""

_RT_CPP = "void rt_fwd(int64_t cfg, at::Tensor wp, at::Tensor x, at::Tensor out, at::Tensor diag);\n"


def rt_source():
    def targs(name, mt):
        s, wk, p, tail = RT_GEOM[name]
        return f"{mt}, {s}, {wk}, {p}, {int(tail)}"
    dispatch = "\n".join(f"    case {i * 8 + mt}: launch_rt<{targs(nm, mt)}>(wp, x, out, dg); break;"
                         for i, nm in enumerate(RT_NAMES) for mt in RT_MT)
    inst = "\n".join(f"template __global__ void rt_kernel<{targs(nm, mt)}>(const uint4*, const __nv_bfloat16*, "
                     f"float*, int, int*);" for nm in RT_NAMES for mt in RT_MT)
    return (_RT_SRC.replace("@@RTDISPATCH@@", dispatch)
            .replace("@@RTINSTANCES@@", "namespace {\n" + inst + "\n}"))


def rt_pack(hi, lo):
    """Router hi / lo [256, 8192] bf16 in rt_kernel's fragment order (see _RT_SRC): per
    (16-row block, 64-wide k tile), fragment j = hl * 4 + q, lane = g * 4 + t, and the uint4 of a
    lane holds W[g][c], W[g][c+1], W[g+8][c], W[g+8][c+1], W[g][c+2], W[g][c+3], W[g+8][c+2],
    W[g+8][c+3] with c = 16 t + 4 q (the mma A fragment a0..a3 under the k permutation)."""
    def frag(p):
        # rows = rb * 16 + h * 8 + g, k = kt * 64 + t * 16 + q * 4 + pp * 2 + e
        v = p.view(16, 2, 8, 128, 4, 4, 2, 2)
        return v.permute(0, 3, 5, 2, 4, 6, 1, 7)
    return torch.stack([frag(hi), frag(lo)], dim=2).contiguous().view(-1)


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
    full = SK_INDEX.get(cfg_name + "F")
    tokr = SK_CONFIGS[cid][2]

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
        c = full if full is not None and x.shape[0] == tokr else cid
        ext.sk_fwd(c, stream, x, out, tab, pmax, base4, n, k, diag)
        return out
    return run


def _wd_runner(ext, name, w, n, k, out_dtype, diag):
    cid = WD_INDEX[name]

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run


def _rt_runner(ext, name, wp, diag):
    cid = RT_NAMES.index(name)

    def run(x):
        out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
        ext.rt_fwd(cid, wp, x, out, diag)
        return out
    return run


def _build_cut():
    import importlib.util
    import os
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("flashinfer (CUTLASS headers) not installed")
    root = os.path.join(list(spec.submodule_search_locations)[0], "data", "cutlass")
    inc = [os.path.join(root, "include"), os.path.join(root, "tools", "util", "include")]
    if not all(os.path.isdir(d) for d in inc):
        raise RuntimeError(f"CUTLASS headers missing under {root}")
    return load_inline(
        name="okc_cut_v2",
        cpp_sources=CUT_CPP, cuda_sources=CUT_SRC, functions=["cut_fwd"], extra_include_paths=inc,
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a",
                           "-DNDEBUG"],
        verbose=False)


def _cut_runner(ext, w, ws):
    def run(x):
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        ext.cut_fwd(w, x, out, ws)
        return out
    return run


def _build_lt():
    # The image's cuBLASLt may be a runtime-only install (no unversioned symlink).
    errs = []
    for i, flag in enumerate(("-l:libcublasLt.so.13", "-lcublasLt", "-l:libcublasLt.so.12")):
        try:
            return load_inline(name=f"okc_lt_v2_{i}", cpp_sources=LT_CPP, functions=LT_FUNCS, with_cuda=True,
                               extra_cflags=["-O2", "-std=c++20"], extra_ldflags=[flag], verbose=False)
        except Exception as exc:
            msg = str(exc)
            lines = [ln.strip() for ln in msg.splitlines() if "error" in ln.lower() or "cannot find" in ln]
            errs.append(f"{flag}: " + " | ".join(lines[:4])[:500] + " ... " + msg[-200:])
    raise RuntimeError(" || ".join(errs))


def _lt_cfg_str(info):
    a, t, sk, red, swz, cust, st, inner, cl, ws, waves = info
    return f"a{a}t{t}s{st}k{sk}r{red}z{swz}c{cust}i{inner}g{cl}w{ws >> 10}v{waves / 1000:.2f}"


def _lt_runner(lt, plan, idx, warg, m, n, ws, fallback):
    def run(x):
        # Plans are built for one m, and the heuristic assumed 256-byte aligned operands.
        if x.shape[0] != m or x.data_ptr() % LT_ALIGN:
            return fallback(x)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        rc = lt.lt_run(plan, idx, warg, x, out, ws)
        if rc:
            raise RuntimeError(f"cublasLt status {rc}")
        return out
    return run


def _lt_screen(fns, x, flush, reps):
    """Median microseconds per function over reps cold-L2 direct launches (no graph): the 64 us
    flush ahead of each launch covers the host launch cost. Failures time as infinity."""
    times = [[] for _ in fns]
    alive = [True] * len(fns)
    for rep in range(reps):
        order = range(len(fns)) if rep % 2 == 0 else range(len(fns) - 1, -1, -1)
        evs = []
        for i in order:
            if not alive[i]:
                continue
            flush.sum()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            try:
                fns[i](x)
            except Exception:
                alive[i] = False
            e1.record()
            evs.append((i, e0, e1))
        torch.cuda.synchronize()
        for i, e0, e1 in evs:
            times[i].append(1000 * e0.elapsed_time(e1))
    return [statistics.median(t) if alive[i] and t else float("inf") for i, t in enumerate(times)]


def _lt_pref(lt, state, layer, m, gen, diag, fallback):
    """The pinned LT_PREF config for (layer, m) as (info, run), or None if absent, rejected by
    this library or failing the self-test."""
    ent = LT_PREF.get((layer, m))
    if ent is None:
        return None
    layout, cfg = ent
    w = state[layer]
    n, k = w.shape
    wts = state.setdefault("lt_wt", {})
    if layer not in wts:
        wts[layer] = w.t().contiguous()
    p = lt.lt_plan_new(m, n, k, layout, state["lt_ws"].numel())
    idx = lt.lt_add_config(p, list(cfg)) if p >= 0 else -1
    if idx < 0:
        print(f"lt {layer} m={m}: preferred L{layout} {cfg} not accepted")
        return None
    fn = _lt_runner(lt, p, idx, wts[layer] if layout % 2 else w, m, n, state["lt_ws"], fallback)
    try:
        _self_test(f"{layer} lt pref", fn, w, n, (m,), SELFTEST_TOL["bf16"], gen, diag)
    except Exception as exc:
        print(f"lt {layer} m={m}: preferred config failed self-test ({str(exc)[-120:]})")
        return None
    return [layout] + list(lt.lt_info(p, idx)), fn


def _time_rounds(fns, x, flush, rounds, reps):
    """Per candidate, its median over reps cold-L2 CUDA-graph replays in each of `rounds`
    rounds; the candidate order rotates every round so drift spreads over all of them."""
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
    out = [[] for _ in fns]
    n = len(fns)
    for r in range(rounds):
        for j in range(n):
            i = (r + j) % n
            ts = []
            for _ in range(reps):
                flush.sum()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[i].replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(1000 * e0.elapsed_time(e1))
            out[i].append(statistics.median(ts))
    return out


def _stable_pick(rounds, preferred):
    """rounds: name -> per-round times. The preferred candidate stays unless another one's
    median paired saving against it exceeds max(STABLE_MARGIN_US, 2 x the MAD of the
    preferred's rounds) and it is faster in all rounds but one; of those the largest saving
    wins. Returns (pick, note)."""
    base = rounds[preferred]
    mid = statistics.median(base)
    margin = max(STABLE_MARGIN_US, 2 * statistics.median([abs(t - mid) for t in base]))
    best, gain, parts = preferred, 0.0, []
    for k, v in rounds.items():
        if k == preferred:
            continue
        d = [a - b for a, b in zip(base, v)]
        saving = statistics.median(d)
        wins = sum(x > 0 for x in d)
        parts.append(f"{k} {saving:+.2f}/{wins}")
        if saving > margin and wins >= len(d) - 1 and saving > gain:
            best, gain = k, saving
    return best, f"pref {preferred} margin {margin:.2f} " + " ".join(parts) + f" -> {best}"


def _lt_search(lt, state, layer, m, ws, flush, gen, diag, fallback):
    """Validated cuBLASLt finalists for (layer, m), fastest first: [(info, run)]."""
    w = state[layer]
    n, k = w.shape
    dev = w.device
    wts = state.setdefault("lt_wt", {})
    if layer not in wts:
        wts[layer] = w.t().contiguous()
    plans = []
    for layout in LT_LAYOUTS:
        p = lt.lt_plan_new(m, n, k, layout, ws.numel())
        if p < 0:
            continue
        for cap in LT_WS_CAPS:
            lt.lt_heuristic(p, cap, LT_HEUR_REQ)
        plans.append((p, layout))
    x, _ = _gen_x(m, k, gen, dev)

    def runner(p, layout, i):
        return _lt_runner(lt, p, i, wts[layer] if layout % 2 else w, m, n, ws, fallback)

    seen = {}

    def screen():
        new = [(p, layout, i) for p, layout in plans for i in range(lt.lt_count(p)) if (p, i) not in seen]
        us = _lt_screen([runner(*c) for c in new], x, flush, LT_SCREEN_REPS)
        for c, t in zip(new, us):
            seen[(c[0], c[2])] = (t, c[1])

    screen()
    first = len(seen)
    for (p, i), (t, layout) in sorted(seen.items(), key=lambda kv: kv[1][0])[:LT_EXPAND_TOP]:
        lt.lt_expand(p, i, LT_EXPAND_MAX)
    screen()
    done = set()
    for (p, i), (t, layout) in sorted(seen.items(), key=lambda kv: kv[1][0]):
        if len(done) == LT_SWEEP_TOP:
            break
        if p in done:
            continue
        done.add(p)
        lt.lt_sweep(p, i, list(LT_SWEEP_CLUSTERS), LT_SWEEP_MAX)
    after_sweep = len(seen)
    screen()
    swept = len(seen) - after_sweep
    for (p, i), (t, layout) in sorted(seen.items(), key=lambda kv: kv[1][0])[:LT_SWEEP_TILES]:
        lt.lt_sweep(p, i, list(range(25)), 25)
        lt.lt_expand(p, i, LT_EXPAND_MAX)
    screen()
    # The same config is a different kernel launch on another layout; the screen only swept
    # the leading layouts.
    for (p, i), (t, layout) in sorted(seen.items(), key=lambda kv: kv[1][0])[:LT_SWEEP_TILES]:
        cfg = list(lt.lt_info(p, i))[:9]
        for q, _ in plans:
            if q != p:
                lt.lt_add_config(q, cfg)
    screen()
    top = sorted(seen.items(), key=lambda kv: kv[1][0])[:LT_RESCREEN]
    us = _lt_screen([runner(p, layout, i) for (p, i), (t, layout) in top], x, flush, LT_RESCREEN_REPS)
    for ((p, i), (t, layout)), u in zip(top, us):
        seen[(p, i)] = (u, layout)
    ranked = sorted(seen.items(), key=lambda kv: kv[1][0])
    out = []
    for (p, i), (t, layout) in ranked:
        if len(out) == LT_FINALISTS or t == float("inf"):
            break
        info = [layout] + list(lt.lt_info(p, i))
        fn = runner(p, layout, i)
        try:
            _self_test(f"{layer} lt", fn, w, n, (m,), SELFTEST_TOL["bf16"], gen, diag)
            out.append((info, fn, t))
        except Exception as exc:
            print(f"lt {layer} m={m}: L{layout} {_lt_cfg_str(info[1:])} failed self-test ({str(exc)[-120:]})")
    print(f"lt {layer} m={m}: {first} heuristic, {swept} swept, {len(seen) - first - swept} other, screen best "
          + ", ".join(f"L{info[0]}:{_lt_cfg_str(info[1:])} {t:.1f}" for info, _, t in out))
    return out



# --------------------------------------------------------------------------------------------
# rd: register-direct 12-bit GEMM for m <= 16 (mechanism of the smallm2 / v18s / decodediet
# routes in alygg77 okc-muse-spark-kernels @ 6cae49a, kernels/muse_bf16_proj.py, which carry
# Nathen Fernandes' MIT routes; re-implemented here). The weights are the 16-row A operand of
# mma.m16n8k16 and come from global memory straight into registers, decoded there; x is the
# n8 B operand, two 16-byte loads per lane per 64-k unit. No shared-memory ring, no barriers in
# the main loop, no escape warps: escapes are patched by XOR into the decoded A register.
#
# Unit = 16 rows x 64 k, 1536 B, three 512-byte planes read lane-major (16 B per lane): low
# bytes of row g, low bytes of row g + 8, then the 4-bit codes (8 B for row g, 8 B for row g + 8;
# byte b of word w holds values 8w + b and 8w + 4 + b, as decode8 expects). Lane 4g + a holds the
# 16 contiguous k 16a .. 16a + 15 of its two rows: k16 step b uses k 16a + 4b .. 16a + 4b + 3 for
# the A positions (2a, 2a + 1, 2a + 8, 2a + 9) and the same k for the B positions, so the
# product is unchanged and every x load is contiguous. Stream order: unit u of all row groups
# together ([u][row group][1536 B]), so the GPU reads one contiguous region at a time.
# Escapes: per warp at most RD_ESC entries, sorted by unit, each
#   seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7
# (reg 0-7 row g pairs, 8-15 row g + 8 pairs; xor7 the difference of the top seven exponent
# bits), padded with seq = 127. A matrix whose escapes overflow a warp's list is not offered.
# RD_GEOM name -> (MT n8 token tiles, WR row groups per CTA, WK warps splitting K, KU units per
# warp, PF units of weights in flight ahead of the one being computed); K = 64 * WK * KU.
# --------------------------------------------------------------------------------------------
RD_ESC = 32
RD_GEOM = {
    # Bench f6 (bcabe48caa884e67), paired against C1's sk picks: rdp8 -0.54 / -0.62 us (two
    # extensions), rdo8 -0.38 / -0.34; rdp8w8 +1.2, rdp16 / rdp16w16 at PF 1 +1.1.
    "rdp8": (1, 2, 16, 8, 1),
    "rdo8": (1, 1, 8, 2, 1),
}
# Which layer / bucket gets which rd candidates (the tuner times them beside the sk pick).
RD_USE = {("pooler", 8): ["rdp8"], ("o", 8): ["rdo8"]}
RD_NAMES = list(RD_GEOM)

_RD_SRC = r"""
namespace {

__device__ __forceinline__ uint4 rd_ldw(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 rd_ldx(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <int MT, int WR, int WK, int KU, int PF>
__global__ void __launch_bounds__(WR * WK * 32) rd_kernel(const uint8_t* __restrict__ ws,
                                                          const uint32_t* __restrict__ esc,
                                                          const __nv_bfloat16* __restrict__ x,
                                                          __nv_bfloat16* __restrict__ out, int M, int N, int K,
                                                          uint32_t base4) {
    constexpr int NWARP = WR * WK;
    __shared__ float4 red[NWARP][MT][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wr = warp / WK, wk = warp % WK;
    const int nrg = N / 16;
    const int rg = blockIdx.x * WR + wr;
    const int g = lane >> 2, a = lane & 3;
    const uint32_t mine = esc[((size_t)rg * WK + wk) * RD_ESCN + lane];
    int nxt = 0;
    uint32_t e = __shfl_sync(0xffffffffu, mine, 0);
    int tok[MT];
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        tok[nt] = min(nt * 8 + g, M - 1);
    }
    const size_t ustride = (size_t)nrg * 1536;
    const uint8_t* wp = ws + ((size_t)(wk * KU) * nrg + rg) * 1536 + lane * 16;
    float acc[MT][4];
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
    // A ring of PF + 1 units in registers; the loop is unrolled so every ring index is constant.
    constexpr int RG = PF + 1;
    uint4 rl0[RG], rl1[RG], rcw[RG];
#pragma unroll
    for (int p = 0; p < PF && p < KU; ++p) {
        const uint8_t* q = wp + (size_t)p * ustride;
        rl0[p] = rd_ldw(q);
        rl1[p] = rd_ldw(q + 512);
        rcw[p] = rd_ldw(q + 1024);
    }
#pragma unroll
    for (int s = 0; s < KU; ++s) {
        if (s + PF < KU) {
            const uint8_t* q = wp + (size_t)(s + PF) * ustride;
            rl0[(s + PF) % RG] = rd_ldw(q);
            rl1[(s + PF) % RG] = rd_ldw(q + 512);
            rcw[(s + PF) % RG] = rd_ldw(q + 1024);
        }
        const uint4 l0 = rl0[s % RG], l1 = rl1[s % RG], cw = rcw[s % RG];
        const int k0 = (wk * KU + s) * 64 + 16 * a;
        uint4 xb[MT][2];
#pragma unroll
        for (int nt = 0; nt < MT; ++nt) {
            const __nv_bfloat16* xp = x + (size_t)tok[nt] * K + k0;
            xb[nt][0] = rd_ldx(xp);
            xb[nt][1] = rd_ldx(xp + 8);
        }
        uint32_t r0[8], r8[8];
        decode8(l0.x, l0.y, cw.x, base4, &r0[0]);
        decode8(l0.z, l0.w, cw.y, base4, &r0[4]);
        decode8(l1.x, l1.y, cw.z, base4, &r8[0]);
        decode8(l1.z, l1.w, cw.w, base4, &r8[4]);
        while ((int)(e >> 17) == s) {
            if (lane == (int)((e >> 12) & 31)) {
                const int reg = (e >> 8) & 15;
                const uint32_t mask = (e & 127u) << (8 + 16 * ((e >> 7) & 1));
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    r0[i] ^= reg == i ? mask : 0u;
                    r8[i] ^= reg == 8 + i ? mask : 0u;
                }
            }
            ++nxt;
            e = __shfl_sync(0xffffffffu, mine, nxt & 31);
            if (nxt >= RD_ESCN) {
                e = 127u << 17;
            }
        }
        const uint32_t* xw[MT];
#pragma unroll
        for (int nt = 0; nt < MT; ++nt) {
            xw[nt] = reinterpret_cast<const uint32_t*>(&xb[nt][0]);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t av[4] = {r0[2 * b], r8[2 * b], r0[2 * b + 1], r8[2 * b + 1]};
#pragma unroll
            for (int nt = 0; nt < MT; ++nt) {
                mma_bf16(acc[nt], av, xw[nt][2 * b], xw[nt][2 * b + 1]);
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < MT; ++nt) {
        red[warp][nt][lane] = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < WR * MT * 32; i += NWARP * 32) {
        const int ln = i & 31, nt = (i >> 5) % MT, w = i / (32 * MT);
        float4 v = red[w * WK][nt][ln];
#pragma unroll
        for (int j = 1; j < WK; ++j) {
            const float4 u = red[w * WK + j][nt][ln];
            v.x += u.x;
            v.y += u.y;
            v.z += u.z;
            v.w += u.w;
        }
        const int row = (blockIdx.x * WR + w) * 16 + (ln >> 2);
        const int t = nt * 8 + 2 * (ln & 3);
        if (t < M) {
            out[(size_t)t * N + row] = __float2bfloat16_rn(v.x);
            out[(size_t)t * N + row + 8] = __float2bfloat16_rn(v.z);
        }
        if (t + 1 < M) {
            out[(size_t)(t + 1) * N + row] = __float2bfloat16_rn(v.y);
            out[(size_t)(t + 1) * N + row + 8] = __float2bfloat16_rn(v.w);
        }
    }
}

#ifndef SK_DEVICE_ONLY
template <int MT, int WR, int WK, int KU, int PF>
void launch_rd(const at::Tensor& ws, const at::Tensor& esc, const at::Tensor& x, at::Tensor& out, int N, int K,
               uint32_t base4) {
    const int M = (int)x.size(0);
    TORCH_CHECK(M >= 1 && M <= 8 * MT && x.is_contiguous() && x.size(1) == K && x.data_ptr() != nullptr
                && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "rd: bad x");
    TORCH_CHECK(N % (16 * WR) == 0 && K == 64 * WK * KU, "rd: shape does not tile");
    TORCH_CHECK(esc.numel() >= (int64_t)(N / 16) * WK * RD_ESCN, "rd: escape lists too short");
    rd_kernel<MT, WR, WK, KU, PF><<<N / (16 * WR), WR * WK * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(ws.data_ptr()), reinterpret_cast<const uint32_t*>(esc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N,
        K, base4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

}  // namespace

#ifdef SK_DEVICE_ONLY
@@RDINSTANCES@@
#else
void rd_fwd(int64_t cfg, at::Tensor ws, at::Tensor esc, at::Tensor x, at::Tensor out, int64_t n, int64_t k,
            int64_t base4) {
    switch (cfg) {
@@RDDISPATCH@@
    default:
        TORCH_CHECK(false, "rd: unknown config ", cfg);
    }
}
#endif
"""

_RD_CPP = ("void rd_fwd(int64_t cfg, at::Tensor ws, at::Tensor esc, at::Tensor x, at::Tensor out, int64_t n, "
           "int64_t k, int64_t base4);\n")


def rd_source():
    targs = {nm: ", ".join(str(v) for v in g) for nm, g in RD_GEOM.items()}
    dispatch = "\n".join(f"    case {i}: launch_rd<{targs[nm]}>(ws, esc, x, out, (int)n, (int)k, (uint32_t)base4); break;"
                         for i, nm in enumerate(RD_NAMES))
    inst = "\n".join(f"template __global__ void rd_kernel<{t}>(const uint8_t*, const uint32_t*, const __nv_bfloat16*, "
                     f"__nv_bfloat16*, int, int, int, uint32_t);" for t in dict.fromkeys(targs.values()))
    return (_RD_SRC.replace("RD_ESCN", str(RD_ESC)).replace("@@RDDISPATCH@@", dispatch)
            .replace("@@RDINSTANCES@@", "namespace {\n" + inst + "\n}"))


def rd_pack(w, geom_name):
    """Stream (uint8) and escape lists (int32 [N/16 * WK, RD_ESC]) for rd_fwd, or None when a
    warp's escapes overflow RD_ESC. Uses the 12-bit code's window base (window_base)."""
    mt, wr, wk, ku, _ = RD_GEOM[geom_name]
    n, k = w.shape
    if n % (16 * wr) or k != 64 * wk * ku:
        return None
    bits = bf16_bits(w)
    base = window_base(bits)
    code, escape, _ = code_bits(bits, base)
    nrg, nu = n // 16, k // 64
    # [rg, h, g, u, a, j]: row 16 rg + 8 h + g, k 64 u + 16 a + j
    b6 = bits.view(nrg, 2, 8, nu, 4, 16)
    c6 = code.view(nrg, 2, 8, nu, 4, 16)
    lo = (b6 & 0xFF).permute(3, 0, 1, 2, 4, 5)  # [u, rg, h, g, a, j]
    cd = c6.permute(3, 0, 1, 2, 4, 5)
    # codes: byte b of word w: value 8w + b (low nibble), 8w + 4 + b (high nibble)
    cw = cd.reshape(nu, nrg, 2, 8, 4, 2, 2, 4)  # j = 8w + 4hi + b
    cbytes = (cw[..., 0, :] | (cw[..., 1, :] << 4))  # [u, rg, h, g, a, w, b]
    lo_planes = lo.reshape(nu, nrg, 2, 32, 16)  # lane = 4g + a
    code_plane = cbytes.permute(0, 1, 3, 4, 2, 5, 6).reshape(nu, nrg, 32, 16)  # per lane: row g 8 B, row g+8 8 B
    unit = torch.cat([lo_planes[:, :, 0], lo_planes[:, :, 1], code_plane], dim=2)  # [u, rg, 96 (3 planes x 32 lanes), 16]
    stream = unit.to(torch.uint8).reshape(-1).contiguous()
    # escapes
    rows, cols = escape.nonzero(as_tuple=True)
    tab = torch.full((nrg * wk, RD_ESC), 127 << 17, dtype=torch.int64, device=w.device)
    if rows.numel():
        rgi, rr = rows // 16, rows % 16
        h, gg = rr // 8, rr % 8
        u, kk = cols // 64, cols % 64
        a, j = kk // 16, kk % 16
        wki, seq = u // ku, u % ku
        lane = 4 * gg + a
        reg = 8 * h + 2 * (j // 4) + (j % 4) // 2
        half = j % 2
        e71 = (bits[rows, cols] >> 8) & 0x7F
        dec = base + (e71 - base).clamp(0, 7)
        xor7 = e71 ^ dec
        warp = rgi * wk + wki
        key = warp * 4096 + seq
        order = key.argsort()
        warp, seq, lane, reg, half, xor7 = (t[order] for t in (warp, seq, lane, reg, half, xor7))
        cnt = torch.bincount(warp, minlength=nrg * wk)
        if int(cnt.max()) > RD_ESC:
            return None
        start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(warp.numel(), device=w.device) - start[warp]
        tab[warp, pos] = (seq << 17) | (lane << 12) | (reg << 8) | (half << 7) | xor7
    return {"stream": stream, "esc": tab.to(torch.int32).contiguous(), "base4": base * 0x01010101,
            "escapes": int(rows.numel()), "max_esc": int(cnt.max()) if rows.numel() else 0}


def _build_ext():
    src = sk_source() + GEMV_CUDA_SRC + wd_source() + rt_source() + rd_source()
    kwargs = dict(name="okc_sk_v2", cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + GEMV_CPP_DECLS + _WD_CPP + _RT_CPP
                  + _RD_CPP,
                  cuda_sources=src,
                  functions=["sk_fwd", "skf_fwd", "cgemv_fwd", "rgemv_fwd", "sgemv_fwd", "wd_fwd", "rt_fwd", "rd_fwd"],
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
    state["wd_ext"] = ext
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
                for nm in lst:
                    if nm not in packed["streams"]:
                        continue
                    nrows = slices[nm]
                    tab, pmax = packed["tabs"][(geoms[nm]["R"], nrows)]
                    fn = _sk_runner(ext, nm, packed["streams"][nm], tab, pmax, packed["base4"], nrows, k,
                                    torch.bfloat16, diag)
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
        for (lname, b), rnames in RD_USE.items():
            if lname != layer:
                continue
            for rnm in rnames:
                pk = rd_pack(w[:rows_of(b)], rnm)
                if pk is None:
                    print(f"rd {layer} {rnm}: not offered (shape or escape list overflow)")
                    continue
                cid = RD_NAMES.index(rnm)
                nrows = rows_of(b)

                # Every loop variable is bound here: prepare() calls these after the loop ends.
                def rd_run(x, pk=pk, cid=cid, nrows=nrows, k=k):
                    # rd's 16-byte vector loads of x need an aligned, contiguous row-major x.
                    if x.data_ptr() % 16 or not x.is_contiguous():
                        x = x.clone(memory_format=torch.contiguous_format)
                    out = torch.empty((x.shape[0], nrows), dtype=torch.bfloat16, device=x.device)
                    ext.rd_fwd(cid, pk["stream"], pk["esc"], x, out, nrows, k, pk["base4"])
                    return out
                add(layer, b, rnm, rd_run, w, nrows, ms_of[b], SELFTEST_TOL["bf16"])
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
    state["router_rt"] = rt_pack(hi, lo)
    rnames = {b: names_of("router", b) for b in (8, 16, 32)}
    rgeoms = {nm: _sk_geom(nm) for lst in rnames.values() for nm in lst}
    _, _, rstreams = pack_router(router, rgeoms)
    for b, lst in rnames.items():
        for nm in lst:
            fn = _sk_runner(ext, nm, rstreams[nm], diag, 0, 0, router.shape[0], router.shape[1], torch.float32, diag)
            add("router", b, nm, fn, router, router.shape[0], ms_of[b], SELFTEST_TOL["router"])
        for nm in RT_NAMES:
            add("router", b, nm, _rt_runner(ext, nm, state["router_rt"], diag), router, router.shape[0], ms_of[b],
                SELFTEST_TOL["router"])
        cands.setdefault(("router", b), []).append(("baseline", lambda x: _router_baseline(x, state), False))
    state["router_hilo"] = torch.cat([hi, lo], 0).contiguous()
    if any(v & WV_PACK for v in WD_VAR.values()):
        state["router_pack"] = router_pack(hi, lo)
    cut = None
    try:
        t1 = time.time()
        cut = _build_cut()
        # The persistent scheduler needs a few KB at most; cut_fwd checks the size.
        state["cut_ws"] = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
        print(f"cut build {time.time() - t1:.1f} s")
    except Exception as exc:
        print(f"cut build failed, route off ({type(exc).__name__}: {str(exc)[-400:]})")
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
                bn_ok = bn in WD_BN[b] or (is_router and b == 256 and bn == 64)
                if hl != is_router or not bn_ok or not wd_fits(nm, n, k) or WD_VAR.get(nm, 0) & WV_NOEMPTY:
                    continue
                wn = state["router_pack"] if WD_VAR.get(nm, 0) & WV_PACK else wk
                fn = _wd_runner(ext, nm, wn, n, k, torch.float32 if is_router else torch.bfloat16, diag)
                add(layer, b, nm, fn, w, n, WD_SELFTEST_MS[b], SELFTEST_TOL["router" if is_router else "bf16"])
            for nm, uses in CUT_USE.items():
                if cut is not None and (layer, b) in uses:
                    add(layer, b, nm, _cut_runner(cut, w, state["cut_ws"]), w, n, WD_SELFTEST_MS[b],
                        SELFTEST_TOL["bf16"])
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
    state["pick_base"] = dict(state["pick"])
    state["lt_pick"] = {}
    lt = None
    try:
        t1 = time.time()
        lt = _build_lt()
        state["lt_ws"] = torch.zeros(LT_WS_BYTES, dtype=torch.uint8, device=dev)
        print(f"lt build {time.time() - t1:.1f} s, cublasLt {lt.lt_version()}, torch cuda {torch.version.cuda}")
    except Exception as exc:
        print(f"lt build failed, route off ({type(exc).__name__}: {str(exc)[:1500]})")
    if lt is not None:
        state["lt_ext"] = lt
        for layer, m in LT_TARGETS:
            base = state["pick"].get((layer, m))
            if base is None:
                continue
            try:
                fins = _lt_search(lt, state, layer, m, state["lt_ws"], flush, gen, diag, base)
                pref = _lt_pref(lt, state, layer, m, gen, diag, base)
                cands = [("base", None, base)] + ([("pref",) + pref] if pref else [])
                cands += [(f"f{i}", f[0], f[1]) for i, f in enumerate(fins)]
                if len(cands) == 1:
                    continue
                x, _ = _gen_x(m, state[layer].shape[1], gen, dev)
                rounds = _time_rounds([c[2] for c in cands], x, flush, STABLE_ROUNDS, STABLE_REPS)
                pick, note = _stable_pick({c[0]: r for c, r in zip(cands, rounds)}, "pref" if pref else "base")
                chosen = next(c for c in cands if c[0] == pick)
                if pick != "base":
                    state["pick"][(layer, m)] = chosen[2]
                    state["lt_pick"][(layer, m)] = chosen[1]
                print(f"ltune {layer:9s} m={m}: "
                      + ", ".join(f"{c[0]}" + (f"=L{c[1][0]}:{_lt_cfg_str(c[1][1:])}" if c[1] else "")
                                  + f" {statistics.median(r):.1f}" for c, r in zip(cands, rounds))
                      + f"; {note}")
            except Exception as exc:
                torch.cuda.synchronize()
                print(f"ltune {layer} m={m}: failed ({type(exc).__name__}: {str(exc)[-200:]})")
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

# Authorized scored control 2026-10-31 repeat1.

# Owned sparse E7-XOR prototype; normal C2 entry is unchanged.
_XOR_CUDA = r'''
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

// OKC_WAIT_SLEEP 0 drops the suspend-time hint from every producer and escape-warp wait (bench
// builds only; the judged builds use 1).
#ifndef OKC_WAIT_SLEEP
#define OKC_WAIT_SLEEP 1
#endif

__device__ __forceinline__ bool mbar_test_sleep(uint32_t bar, uint32_t parity) {
#if OKC_WAIT_SLEEP == 0
    return mbar_test(bar, parity);
#endif
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

// The same wait with the clock read once per 256 polls instead of every poll; still fails
// closed after WAIT_LIMIT cycles (sk consumer variant LITEWD, bench round f4).
__device__ __forceinline__ bool mbar_wait_lite(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead) {
        return false;
    }
    if (mbar_test(bar, parity)) {
        return true;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return false;
            }
        }
    }
    return true;
}

// Per-thread wait without a warp vote: once a thread times out it marks itself dead and skips
// every later wait, so all lanes still run the same stage loop (the mma.sync stays converged)
// and the result is garbage with diag set, which the self-test rejects (fail closed). The
// clock is read once per 256 polls. Measured against the voted per-poll form in round f4.
__device__ __forceinline__ void mbar_wait_novote(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead || mbar_test(bar, parity)) {
        return;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return;
            }
        }
    }
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


__device__ __forceinline__ uint32_t sx_lds32(uint32_t address) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(address) : "memory");
    return v;
}
// Header and all packed operands have been copied from the ring before release.
// Only immutable global records are read here. Every lane executes the same
// count/loop; lane ownership masks XOR without divergent MMA participation.
__device__ __forceinline__ void sx_repair(uint32_t (&b)[8], uint32_t h,
                                         const uint16_t* records, int lane) {
    const unsigned count=h&1023u, start=h>>10;
    for(unsigned i=0;i<count;++i){
        const unsigned rec=records[start+i];
        const unsigned reg=(rec>>5)&7u;
        const uint32_t x=((rec>>9)&127u)<<(8+16*((rec>>8)&1u));
        const uint32_t mask=(rec&31u)==(unsigned)lane ? x : 0u;
#pragma unroll
        for(int r=0;r<8;++r)b[r]^=reg==(unsigned)r ? mask : 0u;
    }
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
    static constexpr bool SX=(NACC & 4096)!=0;
    static constexpr int NW = NWR * WSK;
    static constexpr int NC = NW * 32;
    static constexpr int R = NWR * NT * 8;
    static constexpr int UPS = WSK * UPW;
    static constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
    static constexpr int PAYLOAD = NW * UPW * NT * TU;
    static constexpr int HEADERS = SX ? ((NW*UPW*NT*4+15)&~15) : 0;
    static constexpr int WBY = PAYLOAD + HEADERS;
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
    // NACC bit 4 (SK_ESCG in Python) selects escape corrections read from x in global memory:
    // the escape warps then never touch the ring, so they take table entries round-robin instead
    // of stages and are not counted on the empty barriers (the scheme of alygg77's
    // kernels/muse_bf16_proj.py `_escape_source`, re-implemented here). Bits 0-3 are the
    // accumulator count.
    static constexpr bool ESCG = (NACC & 16) != 0;
    static constexpr int NA = NACC & 15;
    // Bench variants (bits 5-8): FULL compiles the x-tile addressing for M == TOKR (the launch
    // checks it); ESCFAST reads escape entries and x with plain loads that the compiler may
    // overlap across entries; NOWD drops the consumer full-wait watchdog and vote; ESCOFF skips
    // the corrections (attribution only, outputs inexact, never dispatched by prepare()).
    static constexpr bool FULL = (NACC & 32) != 0;
    static constexpr bool ESCFAST = (NACC & 64) != 0;
    static constexpr bool NOWD = (NACC & 128) != 0;
    static constexpr bool ESCOFF = (NACC & 256) != 0;
    // LITEWD: the consumer full wait polls the clock once per 256 polls (still fails closed).
    static constexpr bool LITEWD = (NACC & 512) != 0;
    // NOVOTE: the consumer full wait is mbar_wait_novote (no vote, no break, fails closed).
    static constexpr bool NOVOTE = (NACC & 1024) != 0;
    static constexpr int NESC = (COMP && !SX) ? (ESCG ? 4 : (NSTAGE >= 4 ? 4 : NSTAGE)) : 0;
    static constexpr int EMPTY_COUNT = NW + (ESCG ? 0 : NESC);
    static constexpr int THREADS = (NW + 1 + NESC) * 32;
    static constexpr int CSZ = (TOKS * CP + 3) & ~3;
    static constexpr int CORR_FLOATS = COMP ? NESC * CSZ : 0;
    static constexpr int PEER_FLOATS = CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0;
    static constexpr int BAR_BYTES = 16 * NSTAGE + 32;
    static constexpr int FIXED = 1024 + NSTAGE * SST + BAR_BYTES + 4 * (CORR_FLOATS + PEER_FLOATS) + 128 + 16;
    static constexpr int MINB = 2 * (FIXED + 2048) <= 228 * 1024 ? 2 : 1;
    // Release a slot after the stage's MMAs instead of after its last operand loads (the public
    // entry's order; same fence). At 32 tokens with two units per warp this was -0.38 / -0.29 us
    // on qkv_gate, -0.26 / -0.19 on o, -0.13 / -0.16 on the unpooler, but +0.26 / +0.26 on the
    // pooler (UPW 4) and +0.10 / +0.16 on the m = 8 slice (benches 807464e240ec4f6c,
    // c991867992d34fb2), so it is limited to MT 2 with UPW 2.
    static constexpr bool LATE_ARRIVE = MT == 2 && UPW == 2;
    static_assert(WSK * TOKS * RP * 4 <= NSTAGE * SST, "epilogue partials must fit in the ring");
    static_assert(!FUSE || (COMP && CL == 1 && NW * RUPW == UPS && NW * TOKS * 8 * 4 <= NSTAGE * SST),
                  "fused piece geometry");
    static_assert(TOKR == 16 * MT || (TOKR == 8 && MT == 1), "TOKR must match MT");
    static_assert(!SX || (COMP && !HL && CL==1 && !FUSE && !ESCG),"bounded sparse XOR");
    static_assert(!(COMP && CL > 1), "escape warps are not clustered");
    static_assert(ESCG || NESC <= NSTAGE, "escape warps take turns on stages");
    static_assert(!ESCG || (COMP && !FUSE && CL == 1), "global-x escapes: compressed, unfused, unclustered");
    static_assert(!FULL || TOKR == 16 * MT || TOKR == 8, "FULL: bad TOKR");
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
    if constexpr (C::FULL) {
        M = TOKR;
    }
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
    const int tab_ints = (COMP && !C::SX) ? pmax * 2 + R : 0;
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
        if constexpr (COMP && !C::SX) {
            mbar_init(pbar, 1);
            fence_mbar_init();
            mbar_expect_tx(pbar, tab_ints * 4);
            bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
        }
        for (int i = 0; i < NSTAGE; ++i) {
            mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
                mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
            if constexpr (COMP && !C::SX) {
                mbar_init(pbar, 1);
                fence_mbar_init();
                mbar_expect_tx(pbar, tab_ints * 4);
                bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
            }
            for (int i = 1; i < NSTAGE; ++i) {
                mbar_init(full0 + 8 * i, 1);
            }
            for (int i = 0; i < NSTAGE; ++i) {
                mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
        if constexpr (COMP && !C::SX) {
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
            const int cnt = (dead || C::ESCOFF) ? 0 : tab[0];
            if constexpr (C::ESCG) {
                // x is the kernel's input in global memory (written before launch, read-only
                // here), so these reads need no ordering with the ring; rws carries its pointer.
                const __nv_bfloat16* xg = reinterpret_cast<const __nv_bfloat16*>(rws);
                const int kbeg = gs0 * KST, kend = (gs0 + nst) * KST;
                for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
                    const int e0 = tab[2 * pe];
                    const int kk = e0 & 0xFFFF;
                    if (lane < M && kk >= kbeg && kk < kend) {
                        cr[lane * CP + (e0 >> 16)] +=
                            __int_as_float(tab[2 * pe + 1]) * __bfloat162float(xg[(size_t)lane * K + kk]);
                    }
                }
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(cbar);
                }
                return;
            }
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
                    if constexpr (C::ESCFAST) {
                        // Plain loads: the full-barrier wait above and the proxy fence below
                        // are asm with memory clobbers, so nothing moves across them.
                        const int2* te = reinterpret_cast<const int2*>(tab);
                        const unsigned char* xg = gbase + (xs - sbase);
                        while (pe <= cnt) {
                            const int2 e = te[pe];
                            const int kk = e.x & 0xFFFF;
                            if (kk >= kend) {
                                break;
                            }
                            if (lane < M) {
                                const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
                                const uint32_t off = u * XU + lane * 128 + (((kc >> 3) ^ ((u * M + lane) & 7)) << 4)
                                                     + (kc & 7) * 2;
                                const float xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xg + off));
                                cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * xv;
                            }
                            ++pe;
                        }
                    }
                    while (!C::ESCFAST && pe <= cnt) {
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
    constexpr int NA = C::NA;
    float acc[NA][MT][NT][4];
#pragma unroll
    for (int c = 0; c < NA; ++c) {
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
        if constexpr (C::NOWD) {
            while (!mbar_test(full0 + 8 * slot, (st / NSTAGE) & 1)) {
            }
        } else if constexpr (C::NOVOTE) {
            mbar_wait_novote(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8);
        } else if constexpr (C::LITEWD) {
            if (!__all_sync(0xffffffffu, mbar_wait_lite(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {
                dead = true;
                break;
            }
        } else if (!__all_sync(0xffffffffu, mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {
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
            uint32_t sxh[NT];
            uint32_t b[NT][8];
            uint32_t bl[HL ? NT : 1][8];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                if constexpr (COMP) {
                    lw[nt] = lds128(wb + nt * TU + lane * 16);
                    cw[nt] = lds64(wb + nt * TU + 512 + lane * 8);
                    if constexpr(C::SX) sxh[nt]=sx_lds32(s0+XBY+C::PAYLOAD+((warp*UPW+uu)*NT+nt)*4);
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
            if constexpr(C::SX) {
#pragma unroll
                for(int nt=0;nt<NT;++nt){
                    decode8(lw[nt].x,lw[nt].y,cw[nt].x,base4,&b[nt][0]);
                    decode8(lw[nt].z,lw[nt].w,cw[nt].y,base4,&b[nt][4]);
                    sx_repair(b[nt],sxh[nt],reinterpret_cast<const uint16_t*>(ptab),lane);
                }
            }
            uint32_t a[4][MT][4];
            load_a(xt, ui, a);
            if (!C::LATE_ARRIVE && uu == UPW - 1) {
                // Everything this warp needs from the slot is in registers.
                fence_proxy_async();
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(empty0 + 8 * slot);
                }
            }
            if constexpr (COMP && !C::SX) {
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
                        mma_bf16(acc[((uu * 4 + j) * HM) % NA][mt][nt], a[j][mt], b[nt][2 * j], b[nt][2 * j + 1]);
                        if constexpr (HL) {
                            mma_bf16(acc[((uu * 4 + j) * HM + 1) % NA][mt][nt], a[j][mt], bl[nt][2 * j],
                                     bl[nt][2 * j + 1]);
                        }
                    }
                }
            }
        }
        if constexpr (C::LATE_ARRIVE) {
            fence_proxy_async();
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty0 + 8 * slot);
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

    if constexpr (COMP && !C::SX) {
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
                    for (int c = 1; c < NA; ++c) {
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
                for (int c = 1; c < NA; ++c) {
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
        if constexpr (COMP && !C::SX) {
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
        const int orow = C::SX ? ptab[pmax+rb*R+row] : (COMP ? tab[2 * pmax + row] : rb * R + row);
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
    const int tab_bytes = (COMP && !C::SX) ? (pmax * 2 + C::R) * 4 : 0;
    if constexpr(C::SX) TORCH_CHECK(pmax>=0 && ptab.numel()>=(int64_t)pmax+N,"sx: patch/rowmap bounds");
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
    if constexpr (C::FULL) {
        TORCH_CHECK(M == TOKR, "sk: FULL configs need M == TOKR");
    }
    if constexpr (C::ESCG) {
        TORCH_CHECK(x.is_contiguous(), "sk: global-x escapes need a contiguous x");
        rws = reinterpret_cast<const uint8_t*>(x.data_ptr());
    }
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
namespace {
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
}
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
    case 0: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 1: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 2: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 3: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 4: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 5: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 6: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 7: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 8: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 9: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 10: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 11: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 12: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 13: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 14: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 15: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 16: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 17: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 18: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 19: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 20: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 21: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 22: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 23: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 24: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 25: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 26: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 27: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 28: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 29: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 30: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 31: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 32: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 33: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 34: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 35: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 36: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 37: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 38: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 39: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
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

    default:
        TORCH_CHECK(false, "skf: unknown config ", cfg);
    }
}
#endif

#ifndef SK_DEVICE_ONLY
std::vector<int64_t> sx_resources(int64_t cfg,int64_t pmax){switch(cfg){
case 0: { using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, false>; auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 1: { using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, false>; auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 2: { using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, false>; auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 3: { using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, false>; auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 4: { using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, false>; auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 5: { using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, false>; auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 6: { using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, false>; auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 7: { using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, false>; auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 8: { using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, false>; auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 9: { using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, false>; auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 10: { using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, false>; auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 11: { using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, false>; auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 12: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 13: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 14: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 15: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 16: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 17: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 18: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 19: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 20: { using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 21: { using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 22: { using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 23: { using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 24: { using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 25: { using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 26: { using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 27: { using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 28: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 29: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 30: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 31: { using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, false>; auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 32: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 33: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 34: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 35: { using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, false>; auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 36: { using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, false>; auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 37: { using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, false>; auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 38: { using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, false>; auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
case 39: { using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, false>; auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>;
cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k)); int blocks=0;
const size_t bytes=C::FIXED+((C::SX)?0:(pmax*2+C::R)*4);
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks}; }
default: TORCH_CHECK(false,"sx cfg");}}
#endif
'''
_XOR_CPP = r'''#include <torch/extension.h>
void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor diag);
void skf_fwd(int64_t cfg, at::Tensor ws, at::Tensor rws, at::Tensor x, at::Tensor out, at::Tensor rout, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor rpart, at::Tensor rcount, at::Tensor diag);

std::vector<int64_t> sx_resources(int64_t cfg,int64_t pmax);
'''
_XOR_CONFIG = {'qkv8F': {'ids': [[0, 1], [2, 3]], 'geom': {'R': 24, 'NT': 3, 'NWR': 1, 'WSK': 4, 'UPW': 4, 'NW': 4, 'UPS': 16, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 8, 'fuse': False}, 'original_targs': '1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('qkv8F', 1, 8, 3, 1, 4, 4, 4, True, False, 1057, 1, 'bf16', False)}, 'qkv16F': {'ids': [[4, 5], [6, 7]], 'geom': {'R': 24, 'NT': 3, 'NWR': 1, 'WSK': 4, 'UPW': 2, 'NW': 4, 'UPS': 8, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 16, 'fuse': False}, 'original_targs': '1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('qkv16F', 1, 16, 3, 1, 4, 2, 5, True, False, 1057, 1, 'bf16', False)}, 'qkv32': {'ids': [[8, 9], [10, 11]], 'geom': {'R': 24, 'NT': 3, 'NWR': 1, 'WSK': 4, 'UPW': 2, 'NW': 4, 'UPS': 8, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 32, 'fuse': False}, 'original_targs': '2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false', 'original_cfg': ('qkv32', 2, 32, 3, 1, 4, 2, 4, True, False, 1025, 1, 'bf16', False)}, 'o16F': {'ids': [[12, 13], [14, 15]], 'geom': {'R': 32, 'NT': 2, 'NWR': 2, 'WSK': 2, 'UPW': 2, 'NW': 4, 'UPS': 4, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 16, 'fuse': False}, 'original_targs': '1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('o16F', 1, 16, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'o32F': {'ids': [[16, 17], [18, 19]], 'geom': {'R': 32, 'NT': 2, 'NWR': 2, 'WSK': 2, 'UPW': 2, 'NW': 4, 'UPS': 4, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 32, 'fuse': False}, 'original_targs': '2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('o32F', 2, 32, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'pool16F': {'ids': [[20, 21], [22, 23]], 'geom': {'R': 32, 'NT': 2, 'NWR': 2, 'WSK': 2, 'UPW': 4, 'NW': 4, 'UPS': 8, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 16, 'fuse': False}, 'original_targs': '1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('pool16F', 1, 16, 2, 2, 2, 4, 4, True, False, 1057, 1, 'bf16', False)}, 'pool32F': {'ids': [[24, 25], [26, 27]], 'geom': {'R': 32, 'NT': 2, 'NWR': 2, 'WSK': 2, 'UPW': 4, 'NW': 4, 'UPS': 8, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 32, 'fuse': False}, 'original_targs': '2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('pool32F', 2, 32, 2, 2, 2, 4, 3, True, False, 1057, 1, 'bf16', False)}, 'unp16F': {'ids': [[28, 29], [30, 31]], 'geom': {'R': 32, 'NT': 2, 'NWR': 2, 'WSK': 2, 'UPW': 2, 'NW': 4, 'UPS': 4, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 16, 'fuse': False}, 'original_targs': '1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('unp16F', 1, 16, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'unp32F': {'ids': [[32, 33], [34, 35]], 'geom': {'R': 32, 'NT': 2, 'NWR': 2, 'WSK': 2, 'UPW': 2, 'NW': 4, 'UPS': 4, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 32, 'fuse': False}, 'original_targs': '2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('unp32F', 2, 32, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'slice8F': {'ids': [[36, 37], [38, 39]], 'geom': {'R': 8, 'NT': 1, 'NWR': 1, 'WSK': 8, 'UPW': 2, 'NW': 8, 'UPS': 16, 'comp': True, 'hl': False, 'cl': 1, 'tokr': 8, 'fuse': False}, 'original_targs': '1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('slice8F', 1, 8, 1, 1, 8, 2, 4, True, False, 1057, 1, 'bf16', False)}}

def xor_expected_layout(name,variant,pmax):
    assert name in _XOR_CONFIG and variant in (0,1) and pmax>=0
    c=_XOR_CONFIG[name]['original_cfg']
    _,mt,tokr,nt,nwr,wsk,upw,nstage,comp,hl,nacc,cl,outt,fuse=c
    assert comp and not hl and cl==1 and not fuse and not(nacc&16)
    nw=nwr*wsk;R=nwr*nt*8;ups=wsk*upw
    payload=nw*upw*nt*768;headers=((nw*upw*nt*4+15)&~15) if variant else 0
    xbytes=ups*tokr*128;stage=(xbytes+payload+headers+1023)&~1023
    nesc=0 if variant else (4 if nstage>=4 else nstage)
    correction=nesc*((16*mt*(R+1)+3)&~3)*4
    fixed=1024+nstage*stage+16*nstage+32+correction+128+16
    dynamic=fixed+(0 if variant else pmax*8+R*4)
    return dict(static=0,dynamic=dynamic,threads=(nw+1+nesc)*32,
        stage_stride=stage,weight_payload=payload,header_bytes=headers,
        weight_bytes=payload+headers,x_reservation=xbytes,stages=nstage,
        correction_bytes=correction,escape_readers=nesc,empty_arrivals=nw+nesc)

def xor_build(tag):
    return load_inline(name='okc_owned_sx_'+str(tag),cpp_sources=_XOR_CPP,
        cuda_sources=_XOR_CUDA,functions=['sk_fwd','sx_resources'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],
        verbose=False,no_implicit_headers=True)

def xor_pack(name,weight,variant):
    assert name in _XOR_CONFIG and variant in (0,1)
    nrows=1024 if name=='slice8F' else weight.shape[0]
    g=_XOR_CONFIG[name]['geom'];R=g['R']
    # Original finite matrix/rejection domain. Bit-transport CPU proof is broader.
    if weight.dtype!=torch.bfloat16 or not bool(torch.isfinite(weight.float()).all()):return None
    if not variant:
        pk=pack_layer(weight,{name:g},{name:nrows},balance=name.startswith('qkv'))
        if pk is None:return None
        tab,pmax=pk['tabs'][R,nrows]
        return dict(stream=pk['streams'][name],table=tab,pmax=pmax,base4=pk['base4'],n=nrows,k=weight.shape[1],
                    immutable=[pk['streams'][name],tab],variant=0,name=name,escapes=pk['escapes'])
    bits=bf16_bits(weight);base=window_base(bits);_,escape,_=code_bits(bits,base)
    total=int(escape.sum())
    if total*MAX_ESCAPE_DENOM>bits.numel():return None
    perm=balance_perm(escape,R) if name.startswith('qkv') and total else torch.arange(weight.shape[0],device=weight.device)
    return _xor_pack_bits(name,bits,base,perm,nrows,total)

def _xor_pack_bits(name,bits,base,perm,nrows,total):
    # Exact bit-transport core. Public xor_pack alone enforces finite/sparse offer.
    g=_XOR_CONFIG[name]['geom'];R=g['R']
    bits=bits[perm][:nrows].contiguous();tv=tile_values(bits)
    _,_,decoded=code_bits(tv,base);bad=(tv!=decoded)
    tile,unit,lane,value=bad.nonzero(as_tuple=True)
    ntile,nu=tv.shape[:2];linear=tile*nu+unit
    count=torch.bincount(linear,minlength=ntile*nu).to(torch.int64)
    start=count.cumsum(0)-count
    if int(count.max())>512 or (len(linear) and int(start.max())>=(1<<22)):return None
    assert len(linear)<(1<<22),'SX_PATCH_OFFSET_OVERFLOW'
    xor=((tv[tile,unit,lane,value]^decoded[tile,unit,lane,value])>>8)&127
    records=(lane|((value//2)<<5)|((value%2)<<8)|(xor<<9)).to(torch.int16)
    # nonzero returns lexicographic tile/unit/lane/value order: same unit header
    # selects a unique finite list, no rank/half collision and no padding reader.
    pad=torch.zeros(((len(records)+1)//2)*2,dtype=torch.int16,device=bits.device)
    pad[:len(records)]=records
    patchwords=pad.view(torch.int32);pmax=patchwords.numel()
    table=torch.cat((patchwords,perm[:nrows].to(torch.int32))).contiguous()
    headers=((start<<10)|count).to(torch.int32).reshape(ntile,nu)
    payload=stage_stream(comp_units(bits,base),g,nrows)
    NW,UPW,NT,WSK=g['NW'],g['UPW'],g['NT'],g['WSK'];UPS=g['UPS']
    ns=nu//UPS;nc=nrows//R
    st=torch.arange(ns,device=bits.device).view(-1,1,1,1,1)
    c=torch.arange(nc,device=bits.device).view(1,-1,1,1,1)
    w=torch.arange(NW,device=bits.device).view(1,1,-1,1,1)
    u=torch.arange(UPW,device=bits.device).view(1,1,1,-1,1)
    nt=torch.arange(NT,device=bits.device).view(1,1,1,1,-1)
    tile_index=c*(R//8)+(w//WSK)*NT+nt;unit_index=st*UPS+(w%WSK)*UPW+u
    h=headers[tile_index,unit_index].contiguous().view(torch.uint8).reshape(ns,nc,-1)
    hbytes=(h.shape[-1]+15)&~15
    hp=torch.zeros(ns,nc,hbytes,dtype=torch.uint8,device=bits.device);hp[:,:,:h.shape[-1]]=h
    stream=torch.cat((payload.view(ns,nc,-1),hp),-1).contiguous().view(-1)
    return dict(stream=stream,table=table,pmax=pmax,base4=base*0x01010101,n=nrows,k=bits.shape[1],
        immutable=[stream,table],variant=1,name=name,escapes=total,records=len(records),
        header_bytes=hp.numel(),patch_bytes=patchwords.numel()*4,rowmap_bytes=nrows*4)

def xor_runner(module,name,weight,variant,*,nearby=False,packed=None):
    pk=xor_pack(name,weight,variant) if packed is None else packed
    if pk is None:return None
    assert pk['name']==name and pk['variant']==variant
    cfg=_XOR_CONFIG[name]['ids'][variant][int(nearby)]
    diag=torch.zeros(4,dtype=torch.int32,device=weight.device)
    def run(x):
        assert x.dtype==torch.bfloat16 and x.shape[1]==pk['k'] and x.is_contiguous()
        out=torch.empty((x.shape[0],pk['n']),dtype=torch.bfloat16,device=x.device)
        module.sk_fwd(cfg,pk['stream'],x,out,pk['table'],pk['pmax'],pk['base4'],pk['n'],pk['k'],diag)
        return out
    info=dict(cfg=cfg,name=name,variant=variant,nearby=nearby,private_diag=diag,immutable=pk['immutable'],
        pack=pk,module_name=module.__name__,resources=list(module.sx_resources(cfg,pk['pmax'])),
        expected=xor_expected_layout(name,variant,pk['pmax']))
    return run,info

# OFF-by-default stage-local record prototype.
_SXS_CUDA = r'''
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

// OKC_WAIT_SLEEP 0 drops the suspend-time hint from every producer and escape-warp wait (bench
// builds only; the judged builds use 1).
#ifndef OKC_WAIT_SLEEP
#define OKC_WAIT_SLEEP 1
#endif

__device__ __forceinline__ bool mbar_test_sleep(uint32_t bar, uint32_t parity) {
#if OKC_WAIT_SLEEP == 0
    return mbar_test(bar, parity);
#endif
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

// The same wait with the clock read once per 256 polls instead of every poll; still fails
// closed after WAIT_LIMIT cycles (sk consumer variant LITEWD, bench round f4).
__device__ __forceinline__ bool mbar_wait_lite(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead) {
        return false;
    }
    if (mbar_test(bar, parity)) {
        return true;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return false;
            }
        }
    }
    return true;
}

// Per-thread wait without a warp vote: once a thread times out it marks itself dead and skips
// every later wait, so all lanes still run the same stage loop (the mma.sync stays converged)
// and the result is garbage with diag set, which the self-test rejects (fail closed). The
// clock is read once per 256 polls. Measured against the voted per-poll form in round f4.
__device__ __forceinline__ void mbar_wait_novote(uint32_t bar, uint32_t parity, bool& dead, int* diag, int code) {
    if (dead || mbar_test(bar, parity)) {
        return;
    }
    const long long t0 = clock64();
    int n = 0;
    while (!mbar_test(bar, parity)) {
        if (++n == 256) {
            n = 0;
            if (clock64() - t0 > WAIT_LIMIT) {
                dead = true;
                if (diag != nullptr) {
                    atomicOr(diag, code);
                }
                return;
            }
        }
    }
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


__device__ __forceinline__ uint32_t sx_lds32(uint32_t address) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(address) : "memory");
    return v;
}
// Header and all packed operands have been copied from the ring before release.
// Only immutable global records are read here. Every lane executes the same
// count/loop; lane ownership masks XOR without divergent MMA participation.
__device__ __forceinline__ void sx_repair(uint32_t (&b)[8], uint32_t h,
                                         const uint16_t* records, int lane) {
    const unsigned count=h&1023u, start=h>>10;
    for(unsigned i=0;i<count;++i){
        const unsigned rec=records[start+i];
        const unsigned reg=(rec>>5)&7u;
        const uint32_t x=((rec>>9)&127u)<<(8+16*((rec>>8)&1u));
        const uint32_t mask=(rec&31u)==(unsigned)lane ? x : 0u;
#pragma unroll
        for(int r=0;r<8;++r)b[r]^=reg==(unsigned)r ? mask : 0u;
    }
}


__device__ __forceinline__ unsigned sxs_lds16(uint32_t address) {
    unsigned value;
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}
// Same warp-uniform record order/masks as the global variant. Every record
// access is now part of the credited ring stage, and precedes its empty arrive.
__device__ __forceinline__ void sxs_repair(uint32_t (&b)[8],uint32_t h,
                                          uint32_t arena,int lane) {
    const unsigned count=h&1023u,start=h>>10;
    for(unsigned i=0;i<count;++i){
        const unsigned rec=sxs_lds16(arena+2*(start+i));
        const unsigned reg=(rec>>5)&7u;
        const uint32_t x=((rec>>9)&127u)<<(8+16*((rec>>8)&1u));
        const uint32_t mask=(rec&31u)==(unsigned)lane?x:0u;
#pragma unroll
        for(int r=0;r<8;++r)b[r]^=reg==(unsigned)r?mask:0u;
    }
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
    static constexpr bool SX=(NACC & 4096)!=0;
    static constexpr bool STAGE=(NACC & 8192)!=0;
    static constexpr int NW = NWR * WSK;
    static constexpr int NC = NW * 32;
    static constexpr int R = NWR * NT * 8;
    static constexpr int UPS = WSK * UPW;
    static constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
    static constexpr int PAYLOAD = NW * UPW * NT * TU;
    static constexpr int HEADERS = SX ? ((NW*UPW*NT*4+15)&~15) : 0;
    static constexpr int WBY = PAYLOAD + HEADERS;
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
    // NACC bit 4 (SK_ESCG in Python) selects escape corrections read from x in global memory:
    // the escape warps then never touch the ring, so they take table entries round-robin instead
    // of stages and are not counted on the empty barriers (the scheme of alygg77's
    // kernels/muse_bf16_proj.py `_escape_source`, re-implemented here). Bits 0-3 are the
    // accumulator count.
    static constexpr bool ESCG = (NACC & 16) != 0;
    static constexpr int NA = NACC & 15;
    // Bench variants (bits 5-8): FULL compiles the x-tile addressing for M == TOKR (the launch
    // checks it); ESCFAST reads escape entries and x with plain loads that the compiler may
    // overlap across entries; NOWD drops the consumer full-wait watchdog and vote; ESCOFF skips
    // the corrections (attribution only, outputs inexact, never dispatched by prepare()).
    static constexpr bool FULL = (NACC & 32) != 0;
    static constexpr bool ESCFAST = (NACC & 64) != 0;
    static constexpr bool NOWD = (NACC & 128) != 0;
    static constexpr bool ESCOFF = (NACC & 256) != 0;
    // LITEWD: the consumer full wait polls the clock once per 256 polls (still fails closed).
    static constexpr bool LITEWD = (NACC & 512) != 0;
    // NOVOTE: the consumer full wait is mbar_wait_novote (no vote, no break, fails closed).
    static constexpr bool NOVOTE = (NACC & 1024) != 0;
    static constexpr int NESC = (COMP && !SX) ? (ESCG ? 4 : (NSTAGE >= 4 ? 4 : NSTAGE)) : 0;
    static constexpr int EMPTY_COUNT = NW + (ESCG ? 0 : NESC);
    static constexpr int THREADS = (NW + 1 + NESC) * 32;
    static constexpr int CSZ = (TOKS * CP + 3) & ~3;
    static constexpr int CORR_FLOATS = COMP ? NESC * CSZ : 0;
    static constexpr int PEER_FLOATS = CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0;
    static constexpr int BAR_BYTES = 16 * NSTAGE + 32;
    static constexpr int FIXED = 1024 + NSTAGE * SST + BAR_BYTES + 4 * (CORR_FLOATS + PEER_FLOATS) + 128 + 16;
    static constexpr int MINB = 2 * (FIXED + 2048) <= 228 * 1024 ? 2 : 1;
    // Release a slot after the stage's MMAs instead of after its last operand loads (the public
    // entry's order; same fence). At 32 tokens with two units per warp this was -0.38 / -0.29 us
    // on qkv_gate, -0.26 / -0.19 on o, -0.13 / -0.16 on the unpooler, but +0.26 / +0.26 on the
    // pooler (UPW 4) and +0.10 / +0.16 on the m = 8 slice (benches 807464e240ec4f6c,
    // c991867992d34fb2), so it is limited to MT 2 with UPW 2.
    static constexpr bool LATE_ARRIVE = MT == 2 && UPW == 2;
    static_assert(WSK * TOKS * RP * 4 <= NSTAGE * SST, "epilogue partials must fit in the ring");
    static_assert(!FUSE || (COMP && CL == 1 && NW * RUPW == UPS && NW * TOKS * 8 * 4 <= NSTAGE * SST),
                  "fused piece geometry");
    static_assert(TOKR == 16 * MT || (TOKR == 8 && MT == 1), "TOKR must match MT");
    static_assert(!STAGE || SX,"stage records require sparse XOR");
    static_assert(!SX || (COMP && !HL && CL==1 && !FUSE && !ESCG),"bounded sparse XOR");
    static_assert(!(COMP && CL > 1), "escape warps are not clustered");
    static_assert(ESCG || NESC <= NSTAGE, "escape warps take turns on stages");
    static_assert(!ESCG || (COMP && !FUSE && CL == 1), "global-x escapes: compressed, unfused, unclustered");
    static_assert(!FULL || TOKR == 16 * MT || TOKR == 8, "FULL: bad TOKR");
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
    if constexpr (C::FULL) {
        M = TOKR;
    }
    constexpr int NW = C::NW, NC = C::NC, R = C::R, UPS = C::UPS, TU = C::TU, XBY = C::XBY;
    const int WBY=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
    const int SST=C::STAGE?((XBY+WBY+1023)&~1023):C::SST;
    constexpr int TOKS = C::TOKS, RP = C::RP, CP = C::CP, NESC = C::NESC, CSZ = C::CSZ;
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
    const int tab_ints = (COMP && !C::SX) ? pmax * 2 + R : 0;
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
        if constexpr (COMP && !C::SX) {
            mbar_init(pbar, 1);
            fence_mbar_init();
            mbar_expect_tx(pbar, tab_ints * 4);
            bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
        }
        for (int i = 0; i < NSTAGE; ++i) {
            mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
                mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
            if constexpr (COMP && !C::SX) {
                mbar_init(pbar, 1);
                fence_mbar_init();
                mbar_expect_tx(pbar, tab_ints * 4);
                bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);
            }
            for (int i = 1; i < NSTAGE; ++i) {
                mbar_init(full0 + 8 * i, 1);
            }
            for (int i = 0; i < NSTAGE; ++i) {
                mbar_init(empty0 + 8 * i, C::EMPTY_COUNT);
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
        if constexpr (COMP && !C::SX) {
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
            const int cnt = (dead || C::ESCOFF) ? 0 : tab[0];
            if constexpr (C::ESCG) {
                // x is the kernel's input in global memory (written before launch, read-only
                // here), so these reads need no ordering with the ring; rws carries its pointer.
                const __nv_bfloat16* xg = reinterpret_cast<const __nv_bfloat16*>(rws);
                const int kbeg = gs0 * KST, kend = (gs0 + nst) * KST;
                for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
                    const int e0 = tab[2 * pe];
                    const int kk = e0 & 0xFFFF;
                    if (lane < M && kk >= kbeg && kk < kend) {
                        cr[lane * CP + (e0 >> 16)] +=
                            __int_as_float(tab[2 * pe + 1]) * __bfloat162float(xg[(size_t)lane * K + kk]);
                    }
                }
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(cbar);
                }
                return;
            }
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
                    if constexpr (C::ESCFAST) {
                        // Plain loads: the full-barrier wait above and the proxy fence below
                        // are asm with memory clobbers, so nothing moves across them.
                        const int2* te = reinterpret_cast<const int2*>(tab);
                        const unsigned char* xg = gbase + (xs - sbase);
                        while (pe <= cnt) {
                            const int2 e = te[pe];
                            const int kk = e.x & 0xFFFF;
                            if (kk >= kend) {
                                break;
                            }
                            if (lane < M) {
                                const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
                                const uint32_t off = u * XU + lane * 128 + (((kc >> 3) ^ ((u * M + lane) & 7)) << 4)
                                                     + (kc & 7) * 2;
                                const float xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xg + off));
                                cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * xv;
                            }
                            ++pe;
                        }
                    }
                    while (!C::ESCFAST && pe <= cnt) {
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
    constexpr int NA = C::NA;
    float acc[NA][MT][NT][4];
#pragma unroll
    for (int c = 0; c < NA; ++c) {
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
        if constexpr (C::NOWD) {
            while (!mbar_test(full0 + 8 * slot, (st / NSTAGE) & 1)) {
            }
        } else if constexpr (C::NOVOTE) {
            mbar_wait_novote(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8);
        } else if constexpr (C::LITEWD) {
            if (!__all_sync(0xffffffffu, mbar_wait_lite(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {
                dead = true;
                break;
            }
        } else if (!__all_sync(0xffffffffu, mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {
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
            uint32_t sxh[NT];
            uint32_t b[NT][8];
            uint32_t bl[HL ? NT : 1][8];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                if constexpr (COMP) {
                    lw[nt] = lds128(wb + nt * TU + lane * 16);
                    cw[nt] = lds64(wb + nt * TU + 512 + lane * 8);
                    if constexpr(C::SX) sxh[nt]=sx_lds32(s0+XBY+C::PAYLOAD+((warp*UPW+uu)*NT+nt)*4);
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
            if constexpr(C::SX) {
#pragma unroll
                for(int nt=0;nt<NT;++nt){
                    decode8(lw[nt].x,lw[nt].y,cw[nt].x,base4,&b[nt][0]);
                    decode8(lw[nt].z,lw[nt].w,cw[nt].y,base4,&b[nt][4]);
                    if constexpr(C::STAGE) sxs_repair(b[nt],sxh[nt],s0+XBY+C::PAYLOAD+C::HEADERS,lane);
                    else sx_repair(b[nt],sxh[nt],reinterpret_cast<const uint16_t*>(ptab),lane);
                }
            }
            uint32_t a[4][MT][4];
            load_a(xt, ui, a);
            if (!C::LATE_ARRIVE && uu == UPW - 1) {
                // Everything this warp needs from the slot is in registers.
                fence_proxy_async();
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(empty0 + 8 * slot);
                }
            }
            if constexpr (COMP && !C::SX) {
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
                        mma_bf16(acc[((uu * 4 + j) * HM) % NA][mt][nt], a[j][mt], b[nt][2 * j], b[nt][2 * j + 1]);
                        if constexpr (HL) {
                            mma_bf16(acc[((uu * 4 + j) * HM + 1) % NA][mt][nt], a[j][mt], bl[nt][2 * j],
                                     bl[nt][2 * j + 1]);
                        }
                    }
                }
            }
        }
        if constexpr (C::LATE_ARRIVE) {
            fence_proxy_async();
            __syncwarp();
            if (lane == 0) {
                mbar_arrive(empty0 + 8 * slot);
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

    if constexpr (COMP && !C::SX) {
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
                    for (int c = 1; c < NA; ++c) {
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
                for (int c = 1; c < NA; ++c) {
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
        if constexpr (COMP && !C::SX) {
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
        const int orow=C::STAGE?ptab[rb*R+row]:(C::SX?ptab[pmax+rb*R+row]:(COMP?tab[2*pmax+row]:rb*R+row));
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
    const int tab_bytes = (COMP && !C::SX) ? (pmax * 2 + C::R) * 4 : 0;
    if constexpr(C::SX) TORCH_CHECK(pmax>=0 && ptab.numel()>=(int64_t)(C::STAGE?0:pmax)+N,"sx: record/rowmap bounds");
    if constexpr(C::STAGE) TORCH_CHECK(pmax<=C::NW*UPW*NT*512,"sxs: stage capacity bounds");
    const int sxs_wbytes=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
    const int sxs_sst=C::STAGE?((C::XBY+sxs_wbytes+1023)&~1023):C::SST;
    const size_t smem=(size_t)C::FIXED+NSTAGE*(sxs_sst-C::SST)+tab_bytes;
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
    if constexpr (C::FULL) {
        TORCH_CHECK(M == TOKR, "sk: FULL configs need M == TOKR");
    }
    if constexpr (C::ESCG) {
        TORCH_CHECK(x.is_contiguous(), "sk: global-x escapes need a contiguous x");
        rws = reinterpret_cast<const uint8_t*>(x.data_ptr());
    }
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
namespace {
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 13345, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 13345, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 13345, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 13345, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 13345, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
}
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
    case 0: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 1: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 2: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 3: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 4: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 5: launch_sk<1, 8, 3, 1, 4, 4, 4, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 6: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 7: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 8: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 9: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 10: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 11: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 12: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 13: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 14: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 15: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 16: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 17: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 18: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 19: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 20: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 21: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 22: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 23: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 24: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 25: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 26: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 27: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 28: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 29: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 30: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 31: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 32: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 33: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 34: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 35: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 36: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 37: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 38: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 39: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 40: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 41: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 42: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 43: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 44: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 45: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 46: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 47: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 48: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 49: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 50: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 51: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 52: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 53: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 54: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 55: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 56: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 57: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 58: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 13345, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 59: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
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

    default:
        TORCH_CHECK(false, "skf: unknown config ", cfg);
    }
}
#endif

#ifndef SK_DEVICE_ONLY
std::vector<int64_t> sxs_resources(int64_t cfg,int64_t pmax){switch(cfg){
case 0:{using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, false>;auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 1:{using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, false>;auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 2:{using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, false>;auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 3:{using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, false>;auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 4:{using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 13345, 1, false>;auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 5:{using C=SkCfg<1, 8, 3, 1, 4, 4, 4, true, false, 13313, 1, false>;auto k=sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 6:{using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, false>;auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+5*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 7:{using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, false>;auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+5*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 8:{using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, false>;auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+5*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 9:{using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, false>;auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+5*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 10:{using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 13345, 1, false>;auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+5*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 11:{using C=SkCfg<1, 16, 3, 1, 4, 2, 5, true, false, 13313, 1, false>;auto k=sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+5*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 12:{using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, false>;auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 13:{using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, false>;auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 14:{using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, false>;auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 15:{using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, false>;auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 16:{using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 13313, 1, false>;auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 17:{using C=SkCfg<2, 32, 3, 1, 4, 2, 4, true, false, 13313, 1, false>;auto k=sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 18:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 19:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 20:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 21:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 22:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 13345, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 23:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 13313, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 24:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 25:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 26:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 27:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 28:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 13345, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 29:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 13313, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 30:{using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 31:{using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 32:{using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 33:{using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 34:{using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 13345, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 35:{using C=SkCfg<1, 16, 2, 2, 2, 4, 4, true, false, 13313, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 36:{using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 37:{using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 38:{using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 39:{using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 40:{using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 13345, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 41:{using C=SkCfg<2, 32, 2, 2, 2, 4, 3, true, false, 13313, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 42:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 43:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 44:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 45:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 46:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 13345, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 47:{using C=SkCfg<1, 16, 2, 2, 2, 2, 3, true, false, 13313, 1, false>;auto k=sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 48:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 49:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 50:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 51:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 52:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 13345, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 53:{using C=SkCfg<2, 32, 2, 2, 2, 2, 3, true, false, 13313, 1, false>;auto k=sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+3*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 54:{using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, false>;auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 55:{using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, false>;auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 56:{using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, false>;auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5153, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 57:{using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, false>;auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 5121, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 58:{using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 13345, 1, false>;auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 13345, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
case 59:{using C=SkCfg<1, 8, 1, 1, 8, 2, 4, true, false, 13313, 1, false>;auto k=sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 13313, 1, __nv_bfloat16, false>;
const int wb=C::WBY+(C::STAGE?((pmax*2+15)&~15):0);
const int ss=C::STAGE?((C::XBY+wb+1023)&~1023):C::SST;
const size_t bytes=C::FIXED+4*(ss-C::SST)+(C::SX?0:(pmax*2+C::R)*4);
TORCH_CHECK(pmax>=0 && bytes<=227*1024,"sxs resource extent");
cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,227*1024));
C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));int blocks=0;
C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,bytes));
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)bytes,C::THREADS,a.maxThreadsPerBlock,blocks};}
default:TORCH_CHECK(false,"sxs cfg");}}
#endif
'''
_SXS_CPP = r'''#include <torch/extension.h>
void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor diag);
void skf_fwd(int64_t cfg, at::Tensor ws, at::Tensor rws, at::Tensor x, at::Tensor out, at::Tensor rout, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor rpart, at::Tensor rcount, at::Tensor diag);

std::vector<int64_t> sxs_resources(int64_t cfg,int64_t pmax);
'''
_SXS_CONFIG = {'qkv8F': {'ids': [[0, 1], [2, 3], [4, 5]], 'original_targs': '1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('qkv8F', 1, 8, 3, 1, 4, 4, 4, True, False, 1057, 1, 'bf16', False)}, 'qkv16F': {'ids': [[6, 7], [8, 9], [10, 11]], 'original_targs': '1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('qkv16F', 1, 16, 3, 1, 4, 2, 5, True, False, 1057, 1, 'bf16', False)}, 'qkv32': {'ids': [[12, 13], [14, 15], [16, 17]], 'original_targs': '2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false', 'original_cfg': ('qkv32', 2, 32, 3, 1, 4, 2, 4, True, False, 1025, 1, 'bf16', False)}, 'o16F': {'ids': [[18, 19], [20, 21], [22, 23]], 'original_targs': '1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('o16F', 1, 16, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'o32F': {'ids': [[24, 25], [26, 27], [28, 29]], 'original_targs': '2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('o32F', 2, 32, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'pool16F': {'ids': [[30, 31], [32, 33], [34, 35]], 'original_targs': '1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('pool16F', 1, 16, 2, 2, 2, 4, 4, True, False, 1057, 1, 'bf16', False)}, 'pool32F': {'ids': [[36, 37], [38, 39], [40, 41]], 'original_targs': '2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('pool32F', 2, 32, 2, 2, 2, 4, 3, True, False, 1057, 1, 'bf16', False)}, 'unp16F': {'ids': [[42, 43], [44, 45], [46, 47]], 'original_targs': '1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('unp16F', 1, 16, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'unp32F': {'ids': [[48, 49], [50, 51], [52, 53]], 'original_targs': '2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('unp32F', 2, 32, 2, 2, 2, 2, 3, True, False, 1057, 1, 'bf16', False)}, 'slice8F': {'ids': [[54, 55], [56, 57], [58, 59]], 'original_targs': '1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false', 'original_cfg': ('slice8F', 1, 8, 1, 1, 8, 2, 4, True, False, 1057, 1, 'bf16', False)}}

def sxs_expected_layout(name,variant,pmax):
    assert variant in (0,1,2) and pmax>=0
    expected=xor_expected_layout(name,min(variant,1),pmax)
    if variant==2:
        expected=dict(expected)
        assert pmax<=_XOR_CONFIG[name]['geom']['NW']*_XOR_CONFIG[name]['geom']['UPW']*_XOR_CONFIG[name]['geom']['NT']*512
        records=(pmax*2+15)&~15
        wbytes=expected['weight_bytes']+records
        stride=(expected['x_reservation']+wbytes+1023)&~1023
        expected['dynamic']+=expected['stages']*(stride-expected['stage_stride'])
        expected.update(stage_stride=stride,weight_bytes=wbytes,record_bytes=records,stage_capacity=pmax)
    return expected

def sxs_build(tag):
    return load_inline(name='okc_owned_sxs_'+str(tag),cpp_sources=_SXS_CPP,cuda_sources=_SXS_CUDA,
        functions=['sk_fwd','sxs_resources'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],
        verbose=False,no_implicit_headers=True)

def sxs_repack(pk,name):
    # Derive a fixed arena from the actual maximum of the complete stage/CTA
    # population. No bounded-cap search, truncation or special-case omission.
    g=_XOR_CONFIG[name]['geom'];units=g['NW']*g['UPW']*g['NT']
    payload=units*768;headers=(units*4+15)&~15
    n,k=pk['n'],pk['k'];ns=(k//64)//g['UPS'];nc=n//g['R']
    old=pk['stream'].view(ns*nc,payload+headers)
    h=old[:,payload:payload+units*4].contiguous().view(torch.int32).to(torch.int64)&0xffffffff
    count=h&1023;start=h>>10
    assert bool((count<=512).all()),'SXS_UNIT_COUNT'
    local_start=count.cumsum(1)-count
    totals=count.sum(1);capacity=int(totals.max())
    assert capacity<=units*512,'SXS_CAPACITY'
    assert bool((local_start+count<=capacity).all()),'SXS_ARENA_BOUNDS'
    arena_bytes=(capacity*2+15)&~15
    flat_count=count.reshape(-1)
    expanded=torch.repeat_interleave(torch.arange(flat_count.numel(),device=old.device),flat_count)
    beginning=flat_count.cumsum(0)-flat_count
    rank=torch.arange(expanded.numel(),device=old.device)-beginning[expanded]
    source=start.reshape(-1)[expanded]+rank
    global_records=pk['table'][:pk['pmax']].view(torch.int16)
    assert bool((source<global_records.numel()).all()),'SXS_GLOBAL_RECORD_BOUNDS'
    # Pad only for the bulk-copy alignment. Local headers can never read padding.
    arena=torch.zeros(ns*nc,arena_bytes//2,dtype=torch.int16,device=old.device)
    destination=local_start.reshape(-1)[expanded]+rank
    if expanded.numel():arena[expanded//units,destination]=global_records[source]
    newh=((local_start<<10)|count).to(torch.int32).contiguous().view(torch.uint8)
    hp=torch.zeros(ns*nc,headers,dtype=torch.uint8,device=old.device);hp[:,:units*4]=newh
    stream=torch.cat((old[:,:payload],hp,arena.view(torch.uint8)),1).contiguous().view(-1)
    rowmap=pk['table'][pk['pmax']:].clone().contiguous()
    assert rowmap.numel()==n,'SXS_ROWMAP'
    return dict(stream=stream,table=rowmap,pmax=capacity,base4=pk['base4'],n=n,k=k,
        name=name,variant=2,immutable=[stream,rowmap],records=expanded.numel(),
        stage_capacity=capacity,stage_record_bytes=arena_bytes,stage_groups=ns*nc,
        header_bytes=ns*nc*headers,padded_record_bytes=ns*nc*arena_bytes,
        max_unit_count=int(count.max()),minimum_stage_count=int(totals.min()),
        global_source_record_words=pk['pmax'])

def sxs_pack(name,weight,variant):
    assert variant in (0,1,2)
    pk=xor_pack(name,weight,min(variant,1))
    if pk is None:return None
    return sxs_repack(pk,name) if variant==2 else pk

def sxs_runner(module,name,weight,variant,*,nearby=False,packed=None):
    pk=sxs_pack(name,weight,variant) if packed is None else packed
    if pk is None:return None
    assert pk['name']==name and pk['variant']==variant
    cfg=_SXS_CONFIG[name]['ids'][variant][int(nearby)]
    diag=torch.zeros(4,dtype=torch.int32,device=weight.device)
    def run(x):
        assert x.dtype==torch.bfloat16 and x.shape[1]==pk['k'] and x.is_contiguous()
        out=torch.empty((x.shape[0],pk['n']),dtype=torch.bfloat16,device=x.device)
        module.sk_fwd(cfg,pk['stream'],x,out,pk['table'],pk['pmax'],pk['base4'],pk['n'],pk['k'],diag)
        return out
    return run,dict(cfg=cfg,name=name,variant=variant,nearby=nearby,private_diag=diag,
        immutable=pk['immutable'],pack=pk,module_name=module.__name__,
        resources=list(module.sxs_resources(cfg,pk['pmax'])),expected=sxs_expected_layout(name,variant,pk['pmax']))

# Fixed single-arm stage-local sparse repair; prepare gates only.
_SXSR_PINS = {'control_raw': '5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4', 'control_lf': '7f4fd58cdd940f36c8d4d355f3de85687bc7f6ab04981842ce936db76d78adc2', 'prototype': 'b726aebd42c0749d3ff736fdb72d667d1143dce90819fc45b785580a1832bc2e', 'helper_body': '92b16040b139ead412cc40c1d81e1f2ea36137ea7f22870036e5d6f06c1c1ba0', 'release_body': 'b9974907761268d615e42f57a81165627caa02f027ac071d4f59ff261c51b8d2'}
_SX_TARGET = {('qkv_gate',8):'qkv8F', ('qkv_gate',16):'qkv16F',
 ('qkv_gate',32):'qkv32', ('o',16):'o16F', ('o',32):'o32F',
 ('pooler',16):'pool16F', ('pooler',32):'pool32F',
 ('unpooler',8):'slice8F', ('unpooler',16):'unp16F', ('unpooler',32):'unp32F'}

def _sx_guard(start):
    assert time.monotonic()-start < 295, 'SX incomplete: prepare deadline295s'

def _sx_storage(t):
    assert isinstance(t,torch.Tensor)
    if t.numel()==0:return None
    # Full backing-storage interval rejects aliases/views, conservatively.
    s=t.untyped_storage()
    return (str(t.device),s.data_ptr(),s.data_ptr()+s.nbytes())

def _sx_disjoint(groups):
    intervals=[]
    for owner,ts in groups.items():
        for t in ts:
            x=_sx_storage(t)
            if x is None:continue
            for old_owner,old in intervals:
                assert owner==old_owner or x[0]!=old[0] or x[2]<=old[1] or old[2]<=x[1], ('SX mutable alias',owner,old_owner)
            intervals.append((owner,x))

def _sx_reference(x,w,n,router=False):
    # Official reference output boundary: BF16 projections, FP32 router.
    return (x.double()@w[:n].double().T).to(torch.float32 if router else torch.bfloat16)

def _sx_check(out,ref,x,router=False):
    assert out.shape==ref.shape and out.dtype==ref.dtype and torch.isfinite(out).all(), 'SX shape/dtype/finite'
    d=(out.double()-ref.double());r=ref.double()
    tol=1e-4 if router else 1e-2
    rel=(d.norm()/r.norm().clamp_min(1e-30)).item()
    row=(d.norm(dim=1)/r.norm(dim=1).clamp_min(1e-30)).max().item()
    assert rel<=tol and row<=5*tol, ('SX official math',rel,row,tol)
    probes=(x!=0).sum(1)==1
    if probes.any():
        p=(d[probes].norm(dim=1)/r[probes].norm(dim=1).clamp_min(1e-30)).max().item()
        assert p<=1e-3, ('SX onehot',p)
    return [rel,row]

def _sx_x(m,k,gen,dev,onehot=False,outlier=False):
    x=torch.randn((m,k),generator=gen,device=dev)
    if outlier:x[:,torch.randperm(k,generator=gen,device=dev)[:max(1,k//100)]]*=30
    if onehot:
        x.zero_()
        cols=(0,15,16,63,64,k-1)
        for i in range(m):x[i,cols[i%len(cols)]]=(-1 if i%2 else 1)*2**(3+i%5)
    else:
        # Fixed probes reproduce the official need for weight precision.
        for i in range(0,m,8):
            x[i].zero_();x[i,(17+997*i)%k]=2**(3+i%5)
    return x.to(torch.bfloat16)

def _sx_pack_key(name,weight,variant):
    g=_XOR_CONFIG[name]['geom']
    # stage_stream/header byte order depends on these fields, not TOKR/NSTAGE.
    # Identity requires the same actual supplied weight object and row-balancing rule.
    nrows=1024 if name=='slice8F' else weight.shape[0]
    assert g['comp'] and not g['hl'] and g['cl']==1 and not g['fuse']
    return (id(weight),variant,nrows,name.startswith('qkv'))+tuple(g[k] for k in ('R','NW','UPW','NT','WSK','UPS'))

def _sx_layout(name,variant,pmax):
    c=SK_CONFIGS[SK_INDEX[name]]
    _,mt,tokr,nt,nwr,wsk,upw,nstage,comp,hl,nacc,cl,_,fuse=c
    assert comp and not hl and cl==1 and not fuse and not (nacc&16)
    nw=nwr*wsk;r=nwr*nt*8;ups=wsk*upw
    payload=nw*upw*nt*768
    headers=((nw*upw*nt*4+15)&~15) if variant else 0
    xby=ups*tokr*128;sst=(xby+payload+headers+1023)&~1023
    nesc=0 if variant else min(4,nstage)
    csz=(16*mt*(r+1)+3)&~3
    fixed=1024+nstage*sst+(16*nstage+32)+4*nesc*csz+128+16
    dynamic=fixed+(0 if variant else (2*pmax+r)*4)
    return {'static':0,'dynamic':dynamic,'threads':32*(nw+1+nesc),
            'headers_per_stage':headers,'stage_stride':sst,'fixed':fixed,'escape_warps':nesc}

def _sx_capture(fn,x):
    stream=torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(x);fn(x)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):out=fn(x)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g.replay();torch.cuda.synchronize()
    return g,out,stream


_SXSR_NORMAL_PREPARE=prepare

def _sxsr_math(out,ref,x,candidate=False):
    assert bool(torch.isfinite(ref).all()),'SXSRELEASE_REFERENCE_NONFINITE'
    metrics=_sx_check(out,ref,x)
    assert all(__import__('math').isfinite(v) for v in metrics),'SXSRELEASE_METRIC_NONFINITE'
    assert metrics[0]<=SELFTEST_TOL['bf16'] and metrics[1]<=5*SELFTEST_TOL['bf16'],'SXSRELEASE_C2_MATH_GATE'
    probes=(x!=0).sum(1)==1
    if bool(probes.any()):
        pe=((out[probes].double()-ref[probes].double()).norm(dim=1)/ref[probes].double().norm(dim=1).clamp_min(1e-30)).max()
        assert float(pe)<=PROBE_TOL,'SXSRELEASE_C2_PROBE_GATE'
        if candidate:assert torch.equal(out[probes],ref[probes]),'SXSRELEASE_EXACT_PROBE'
    return metrics

def _sxsr_resource(name,info,control):
    r=info['resources'];expected=sxs_expected_layout(name,info['variant'],info['pack']['pmax'])
    assert len(r)==7 and r[1]==0 and r[6]>=1,('SXSRELEASE_RESOURCE',name,r)
    assert expected['dynamic']<=227*1024,('SXSRELEASE_DYNAMIC_EXTENT',name,expected)
    assert r[2]==expected['static'] and r[3]==expected['dynamic'] and r[4]==r[5]==expected['threads'],('SXSRELEASE_LAYOUT',name,r,expected)
    if control is not None:
        assert r[6]>=control['resources'][6],('SXSRELEASE_ADMISSION',name,r,control['resources'])
    return list(r)

def _sxsr_qualify(weights,state,start):
    _sx_guard(start)
    module=sxs_build('release_sk10_stage_e7xor1')
    _sx_guard(start)
    originals=dict(state['pick']);ready={};cache={};rows=[];diagnostics={}
    checks=0;captures=0;fresh_checks=0;qualified=0
    for index,((layer,m),name) in enumerate(_SX_TARGET.items()):
        _sx_guard(start);weight=weights[layer];packs=[]
        # Same twelve immutable formats as the audited paired harness. Failed
        # finite/sparse offer is a legal incumbent fallback, not qualification.
        for variant in (2,0):
            key=_sx_pack_key(name,weight,variant)
            if key not in cache:cache[key]=sxs_pack(name,weight,variant)
            pk=cache[key]
            packs.append(None if pk is None else dict(pk,name=name))
            if pk is None:break
        if len(packs)!=2 or any(p is None for p in packs):
            rows.append(dict(layer=layer,m=m,name=name,status='unsupported_pack_original_C2',checks=0))
            continue
        candidate_pack,control_pack=packs
        # Reject unsupported resource extents before any test kernel can launch.
        # Weight-representation offer rejection above alone permits fallback.
        for variant,pk in ((2,candidate_pack),(0,control_pack)):
            layout=sxs_expected_layout(name,variant,pk['pmax'])
            assert 0<=layout['dynamic']<=227*1024,('SXSRELEASE_DYNAMIC_EXTENT',name,layout)
        assert candidate_pack['table'].numel()==candidate_pack['n'],'SXSRELEASE_ROWMAP_ONLY'
        control,ci=sxs_runner(module,name,weight,0,packed=control_pack)
        candidate,ti=sxs_runner(module,name,weight,2,packed=candidate_pack)
        control_nf,cn=sxs_runner(module,name,weight,0,nearby=True,packed=control_pack)
        candidate_nf,tn=sxs_runner(module,name,weight,2,nearby=True,packed=candidate_pack)
        vectors=[_sxsr_resource(name,ci,None),_sxsr_resource(name,ti,ci),
                 _sxsr_resource(name,cn,None),_sxsr_resource(name,tn,cn)]
        for label,info in zip(('control','candidate','control_nf','candidate_nf'),(ci,ti,cn,tn)):
            diagnostics[name+':'+label]=[info['private_diag']]
        _sx_disjoint(diagnostics)
        gen=torch.Generator(device=weight.device).manual_seed(914301+index*97+m)
        n=candidate_pack['n'];k=weight.shape[1];case_checks=0
        for mm,functions in ((m,(control,candidate)),(max(1,m-1),(control_nf,candidate_nf))):
            for bank in ('normal','onehot','outlier'):
                _sx_guard(start)
                x=_sx_x(mm,k,gen,weight.device,onehot=bank=='onehot',outlier=bank=='outlier')
                ref=_sx_reference(x,weight,n)
                for variant,fn in enumerate(functions):
                    out=fn(x);torch.cuda.synchronize()
                    _sxsr_math(out,ref,x,bool(variant));case_checks+=1
                del x,ref,out
        # Only the single treatment is captured here. No events/latency samples.
        x=_sx_x(m,k,gen,weight.device);ref=_sx_reference(x,weight,n)
        graph,out,stream=_sx_capture(candidate,x);captures+=1
        _sxsr_math(out,ref,x,True);case_checks+=1
        for replay in range(3):
            _sx_guard(start)
            fresh=_sx_x(m,k,gen,weight.device)
            ref=_sx_reference(fresh,weight,n)
            x.copy_(fresh);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            _sxsr_math(out,ref,x,True);case_checks+=1;fresh_checks+=1
        for info in (ci,ti,cn,tn):
            assert int(torch.count_nonzero(info['private_diag']))==0,('SXSRELEASE_DEVICE_DIAGNOSTIC',name)
        assert case_checks==16,('SXSRELEASE_CASE_COVERAGE',name,case_checks)
        checks+=case_checks;qualified+=1
        first=2 if m==8 else 9 if m==16 else 17
        for nearby_m in range(first,m):ready[layer,nearby_m]=candidate_nf
        ready[layer,m]=candidate
        rows.append(dict(layer=layer,m=m,name=name,status='override',checks=case_checks,
            stage_capacity=candidate_pack['pmax'],stage_record_bytes=candidate_pack['stage_record_bytes'],
            weight_bytes=sxs_expected_layout(name,2,candidate_pack['pmax'])['weight_bytes'],
            installed_range=[first,m],resources=vectors,register_thread_products=[r[0]*r[4] for r in vectors]))
        del graph,out,stream,x,ref,fresh,control,control_nf,candidate_nf,ci,ti,cn,tn
        _sx_guard(start)
    allowed={(layer,mm) for layer,m in _SX_TARGET for mm in range(2 if m==8 else 9 if m==16 else 17,m+1)}
    assert len(rows)==10 and set(ready).issubset(allowed),'SXSRELEASE_COVERAGE'
    assert checks==16*qualified and captures==qualified and fresh_checks==3*qualified,'SXSRELEASE_INCOMPLETE'
    for buffers in diagnostics.values():
        for diag in buffers:assert int(torch.count_nonzero(diag))==0,'SXSRELEASE_FINAL_DIAGNOSTIC'
    _sx_guard(start)
    # Atomic host publication after every required check has passed. Untargeted
    # keys, pick_base, general-M fallbacks and normal kernel_fn stay incumbent.
    assert state['pick']==originals,'SXSRELEASE_INCUMBENT_MUTATED'
    state['pick'].update(ready)
    assert all(state['pick'][key] is fn for key,fn in originals.items() if key not in ready),'SXSRELEASE_UNTARGETED_MUTATED'
    state['stage_xor_release']={'source':_SXSR_PINS,'overrides':qualified,'fallbacks':10-qualified,
        'installed_pick_keys':len(ready),'nearby_pick_keys':len(ready)-qualified,
        'math_checks':checks,'captures':captures,'fresh_checks':fresh_checks,'routes':rows}
    _sx_guard(start)
    for row in rows:
        print('SXSRELEASE_ROUTE',row['layer'],row['m'],row['status'],'checks='+str(row['checks']),
              'capacity='+str(row.get('stage_capacity',0)),'arena_bytes='+str(row.get('stage_record_bytes',0)),
              'resources='+str(row.get('resources',[])),flush=True)
    print('SXSRELEASE_DONE overrides='+str(qualified)+' fallbacks='+str(10-qualified)+
          ' pick_keys='+str(len(ready))+' checks='+str(checks)+' captures='+str(captures)+' fresh='+str(fresh_checks),flush=True)
    _sx_guard(start)
    # Only immutable weight packs, selected runners and their diagnostics survive.
    return state

def prepare(weights):
    start=time.monotonic()
    state=_SXSR_NORMAL_PREPARE(weights)
    _sx_guard(start)
    return _sxsr_qualify(weights,state,start)
