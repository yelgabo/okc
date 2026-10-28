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

import math
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
# A matrix whose escapes exceed this share of its weights stays raw BF16.
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
}
ROUTER_GEOM = {
    "router8": (1, 8, 1, 1, 8, 1, 6),
    "router16": (1, 16, 1, 1, 8, 1, 5),
    "router32": (2, 32, 1, 1, 8, 1, 4),
}
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
SMEM_BUDGET = 227 * 1024
SMEM_PER_SM = 228 * 1024


def _sk_configs():
    """Every kernel instantiation: (name, MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, out)."""
    cfgs = []
    for name, (mt, tokr, nt, nwr, wsk, upw, nstage) in SK_GEOM.items():
        for comp in (True, False):
            cname = name if comp else name + "raw"
            cfgs.append((cname, mt, tokr, nt, nwr, wsk, upw, _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, comp, False, 1),
                         comp, False, 1, 1, "bf16"))
    for name, (mt, tokr, nt, nwr, wsk, upw, nstage) in ROUTER_GEOM.items():
        cfgs.append((name, mt, tokr, nt, nwr, wsk, upw, _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, False, True, 3),
                     False, True, 2, 3, "f32"))
    return cfgs


def _sk_smem(mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, cl, pmax=64):
    """Mirror of the kernel's constexpr layout, with a pmax guess for the escape table."""
    nw, r, ups = nwr * wsk, nwr * nt * 8, wsk * upw
    tu = 768 if comp else (2048 if hl else 1024)
    wby, xby = nw * upw * nt * tu, ups * tokr * 128
    sst = (xby + wby + 1023) & ~1023
    toks, rp, cp = 16 * mt, r + ((40 - (r % 32)) % 32), r + 1
    nesc = (4 if nstage >= 4 else nstage) if comp else 0
    corr = nesc * ((toks * cp + 3) & ~3) * 4 if comp else 0
    peer = (cl - 1) * wsk * toks * rp * 4 if cl > 1 else 0
    tab = (pmax * 8 + r * 4) if comp else 0
    return 1024 + nstage * sst + 16 * nstage + 32 + corr + peer + tab + 128


def _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, cl):
    while nstage > 2 and _sk_smem(mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, cl) > SMEM_BUDGET:
        nstage -= 1
    return nstage


SK_CONFIGS = _sk_configs()
SK_INDEX = {c[0]: i for i, c in enumerate(SK_CONFIGS)}


def _sk_geom(name):
    c = SK_CONFIGS[SK_INDEX[name]]
    mt, tokr, nt, nwr, wsk, upw = c[1:7]
    return {"R": nwr * nt * 8, "NT": nt, "NWR": nwr, "WSK": wsk, "UPW": upw, "NW": nwr * wsk,
            "UPS": wsk * upw, "comp": c[8], "hl": c[9], "cl": c[11], "tokr": tokr}


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

__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
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
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
    return v;
}

__device__ __forceinline__ uint2 lds64(uint32_t a) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
    return v;
}

__device__ __forceinline__ float lds_bf16(uint32_t a) {
    uint16_t v;
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ __forceinline__ void ldsm_x4(uint32_t a, uint32_t* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
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

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL>
struct SkCfg {
    static constexpr int NW = NWR * WSK;
    static constexpr int NC = NW * 32;
    static constexpr int R = NWR * NT * 8;
    static constexpr int UPS = WSK * UPW;
    static constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
    static constexpr int WBY = NW * UPW * NT * TU;
    static constexpr int XBY = UPS * TOKR * 128;
    static constexpr int SST = (XBY + WBY + 1023) & ~1023;
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
    static constexpr int FIXED = 1024 + NSTAGE * SST + BAR_BYTES + 4 * (CORR_FLOATS + PEER_FLOATS) + 128;
    static constexpr int MINB = 2 * (FIXED + 2048) <= 228 * 1024 ? 2 : 1;
    static_assert(WSK * TOKS * RP * 4 <= NSTAGE * SST, "epilogue partials must fit in the ring");
    static_assert(TOKR == 16 * MT || (TOKR == 8 && MT == 1), "TOKR must match MT");
    static_assert(!(COMP && CL > 1), "escape warps are not clustered");
    static_assert(NESC <= NSTAGE, "escape warps take turns on stages");
};

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL,
          typename OutT>
__global__ void __launch_bounds__((SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL>::THREADS),
                                  (SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL>::MINB))
sk_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ ws, OutT* __restrict__ out,
          const int32_t* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int* __restrict__ diag) {
    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL>;
    constexpr int NW = C::NW, NC = C::NC, R = C::R, UPS = C::UPS, TU = C::TU, WBY = C::WBY, XBY = C::XBY;
    constexpr int SST = C::SST, TOKS = C::TOKS, RP = C::RP, CP = C::CP, NESC = C::NESC, CSZ = C::CSZ;
    constexpr int KST = UPS * 64;

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

    auto issue = [&](int st) {
        const int slot = st % NSTAGE;
        const uint32_t s0 = sbase + slot * SST;
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
            for (int st = 1; st < nst; ++st) {
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
            mbar_wait<true>(pbar, 0, dead, diag, 2);
            const int cnt = dead ? 0 : tab[0];
            int pe = 1;
            for (int st = 0; st < nst; ++st) {
                const int slot = st % NSTAGE;
                const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
                if (!mbar_wait<true>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 4)) {
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
        if (!mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8)) {
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
        mbar_wait<false>(qbar, 0, dead, diag, 32);
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
          typename OutT>
void launch_sk(const at::Tensor& ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab, int pmax,
               uint32_t base4, int M, int N, int K, int* diag) {
    using C = SkCfg<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL>;
    auto kern = sk_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT>;
    const int tab_bytes = COMP ? (pmax * 2 + C::R) * 4 : 0;
    const size_t smem = (size_t)C::FIXED + tab_bytes;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
        configured = true;
    }
    TORCH_CHECK(M >= 1 && M <= TOKR, "sk: M out of range for this config");
    TORCH_CHECK(N % C::R == 0 && K % (C::UPS * 64) == 0, "sk: shape does not tile");
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
        C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xmap, wsp, op, tp, pmax, base4, M, N, K, diag));
    } else {
        kern<<<N / C::R, C::THREADS, smem, stream>>>(xmap, wsp, op, tp, pmax, base4, M, N, K, diag);
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
#endif
"""

_SK_CPP = "void sk_fwd(int64_t cfg, at::Tensor ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t n, int64_t k, at::Tensor diag);"


def _sk_targs(c):
    name, mt, tokr, nt, nwr, wsk, upw, nstage, comp, hl, nacc, cl, outt = c
    ot = "float" if outt == "f32" else "__nv_bfloat16"
    return f"{mt}, {tokr}, {nt}, {nwr}, {wsk}, {upw}, {nstage}, {str(comp).lower()}, {str(hl).lower()}, {nacc}, {cl}, {ot}"


def sk_source(device_only=False):
    """The kernel source with the config table expanded; device_only adds explicit
    instantiations for a host-less compile (tools/compile_check.sh)."""
    dispatch = "\n".join(
        f"    case {i}: launch_sk<{_sk_targs(c)}>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg); break;"
        for i, c in enumerate(SK_CONFIGS))
    inst = "\n".join(
        f"template __global__ void sk_kernel<{targs}>(const __grid_constant__ CUtensorMap, const uint8_t*, "
        f"{'float' if c[12] == 'f32' else '__nv_bfloat16'}*, const int32_t*, int, uint32_t, int, int, int, int*);"
        for targs, c in {_sk_targs(c): c for c in SK_CONFIGS}.items())
    # Explicit instantiation must name the template at namespace scope, so the device-only
    # build reopens the anonymous namespace around it.
    src = _SK_SRC.replace("@@DISPATCH@@", dispatch).replace("@@INSTANCES@@", "namespace {\n" + inst + "\n}")
    return ("#define SK_DEVICE_ONLY 1\n" + src) if device_only else src


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
# Candidates, self-test, autotune, dispatch.
# --------------------------------------------------------------------------------------------
def bucket_of(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else None


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

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
        ext.sk_fwd(cid, stream, x, out, tab, pmax, base4, n, k, diag)
        return out
    return run


def _build_ext():
    src = sk_source()
    kwargs = dict(name="okc_sk_v1", cpp_sources="#include <torch/extension.h>\n" + _SK_CPP, cuda_sources=src,
                  functions=["sk_fwd"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                  verbose=False)
    try:
        return load_inline(no_implicit_headers=True, **kwargs)
    except TypeError:
        kwargs["cuda_sources"] = src.replace("#include <ATen/core/Tensor.h>", "#include <torch/types.h>")
        return load_inline(**kwargs)


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
            cands.setdefault((layer, bucket), []).append((name, fn))
        except Exception as exc:
            print(f"selftest {layer} {name}: disabled ({type(exc).__name__}: {str(exc)[-300:]})")

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
        for b in (8, 16, 32):
            nrows = rows_of(b)
            cands.setdefault((layer, b), []).append(("cublas", lambda x, w=w, nrows=nrows: _cublas(x, w[:nrows])))
    rnames = {b: names_of("router", b) for b in (8, 16, 32)}
    rgeoms = {nm: _sk_geom(nm) for lst in rnames.values() for nm in lst}
    _, _, rstreams = pack_router(router, rgeoms)
    for b, lst in rnames.items():
        for nm in lst:
            fn = _sk_runner(ext, nm, rstreams[nm], diag, 0, 0, router.shape[0], router.shape[1], torch.float32, diag)
            add("router", b, nm, fn, router, router.shape[0], ms_of[b], SELFTEST_TOL["router"])
        cands.setdefault(("router", b), []).append(("baseline", lambda x: _router_baseline(x, state)))
    print(f"selftests done {time.time() - t0:.1f} s")

    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    for (layer, b), lst in sorted(cands.items()):
        k = state[layer].shape[1]
        for m in ((1, 8) if b == 8 else (b,)):
            x, _ = _gen_x(m, k, gen, dev)
            if len(lst) > 1:
                us = _time_candidates([fn for _, fn in lst], x, flush)
                best = min(range(len(lst)), key=lambda i: us[i])
                print(f"tune {layer:9s} m={m:2d}: " + ", ".join(f"{nm} {t:.1f}" for (nm, _), t in zip(lst, us))
                      + f" -> {lst[best][0]}")
            else:
                best = 0
            state["pick"][(layer, m)] = lst[best][1]
        if b != 8:
            continue
        for m in (2, 3, 4, 5, 6, 7):
            state["pick"][(layer, m)] = state["pick"][(layer, 8)]
    for (layer, b), lst in cands.items():
        for m in range(9, 33):
            if (layer, m) not in state["pick"]:
                state["pick"][(layer, m)] = state["pick"][(layer, bucket_of(m))]
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
