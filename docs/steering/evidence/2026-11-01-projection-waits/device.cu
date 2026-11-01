#define SK_DEVICE_ONLY 1

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
namespace {
template __global__ void sk_kernel<1, 8, 3, 1, 4, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
template __global__ void sk_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>(const __grid_constant__ CUtensorMap, const uint8_t*, __nv_bfloat16*, const int32_t*, int, uint32_t, int, int, int, int*, const uint8_t*, float*, int*, float*);
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
    case 1: launch_sk<1, 16, 3, 1, 4, 2, 5, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 2: launch_sk<2, 32, 3, 1, 4, 2, 4, true, false, 1025, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 3: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 4: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 5: launch_sk<1, 16, 2, 2, 2, 4, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 6: launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 7: launch_sk<1, 8, 1, 1, 8, 2, 4, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 8: launch_sk<1, 16, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
    case 9: launch_sk<2, 32, 2, 2, 2, 2, 3, true, false, 1057, 1, __nv_bfloat16, false>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, (int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;
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

namespace {

template <int BN>
__device__ __forceinline__ void wgmma_ss(float* d, uint64_t a, uint64_t b);
template <> __device__ __forceinline__ void wgmma_ss<64>(float* d, uint64_t a, uint64_t b) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                 "%32, %33, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(a), "l"(b), "r"(1) : "memory");
}
template <> __device__ __forceinline__ void wgmma_ss<128>(float* d, uint64_t a, uint64_t b) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
                 "%64, %65, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "l"(a), "l"(b), "r"(1) : "memory");
}


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
namespace {
template __global__ void wd_kernel<1, 128, 3, 1, false, 8, __nv_bfloat16, 0, true, 0>(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, __nv_bfloat16*, int, int, int, int*);
template __global__ void wd_kernel<1, 64, 8, 1, true, 8, float, 0, false, 6190>(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, float*, int, int, int, int*);
template __global__ void wd_kernel<1, 128, 8, 1, true, 6, float, 0, false, 6190>(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap, float*, int, int, int, int*);
}
#else
void wd_fwd(int64_t cfg, at::Tensor w, at::Tensor x, at::Tensor out, int64_t n, int64_t k, at::Tensor diag) {
    int* dg = diag.numel() > 0 ? diag.data_ptr<int>() : nullptr;
    switch (cfg) {
    case 0: launch_wd<1, 128, 3, 1, false, 8, __nv_bfloat16, 0, true, 0>(w, x, out, (int)n, (int)k, dg); break;
    case 1: launch_wd<1, 64, 8, 1, true, 8, float, 0, false, 6190>(w, x, out, (int)n, (int)k, dg); break;
    case 2: launch_wd<1, 128, 8, 1, true, 6, float, 0, false, 6190>(w, x, out, (int)n, (int)k, dg); break;
    default:
        TORCH_CHECK(false, "wd: unknown config ", cfg);
    }
}
#endif

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
namespace {
template __global__ void rt_kernel<1, 8, 8, 4, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<2, 8, 8, 4, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<4, 8, 8, 4, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<1, 16, 8, 4, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<2, 16, 8, 4, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<4, 16, 8, 4, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<1, 16, 8, 8, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<2, 16, 8, 8, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
template __global__ void rt_kernel<4, 16, 8, 8, 3>(const uint4*, const __nv_bfloat16*, float*, int, int*);
}
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
    case 1: launch_rt<1, 8, 8, 4, 3>(wp, x, out, dg); break;
    case 2: launch_rt<2, 8, 8, 4, 3>(wp, x, out, dg); break;
    case 4: launch_rt<4, 8, 8, 4, 3>(wp, x, out, dg); break;
    case 9: launch_rt<1, 16, 8, 4, 3>(wp, x, out, dg); break;
    case 10: launch_rt<2, 16, 8, 4, 3>(wp, x, out, dg); break;
    case 12: launch_rt<4, 16, 8, 4, 3>(wp, x, out, dg); break;
    case 17: launch_rt<1, 16, 8, 8, 3>(wp, x, out, dg); break;
    case 18: launch_rt<2, 16, 8, 8, 3>(wp, x, out, dg); break;
    case 20: launch_rt<4, 16, 8, 8, 3>(wp, x, out, dg); break;
    default:
        TORCH_CHECK(false, "rt: unknown config ", cfg);
    }
}
#endif

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
    const uint32_t mine = esc[((size_t)rg * WK + wk) * 32 + lane];
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
            if (nxt >= 32) {
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
    TORCH_CHECK(esc.numel() >= (int64_t)(N / 16) * WK * 32, "rd: escape lists too short");
    rd_kernel<MT, WR, WK, KU, PF><<<N / (16 * WR), WR * WK * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(ws.data_ptr()), reinterpret_cast<const uint32_t*>(esc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N,
        K, base4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

}  // namespace

#ifdef SK_DEVICE_ONLY
namespace {
template __global__ void rd_kernel<1, 2, 16, 8, 1>(const uint8_t*, const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, uint32_t);
template __global__ void rd_kernel<1, 1, 8, 2, 1>(const uint8_t*, const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, uint32_t);
}
#else
void rd_fwd(int64_t cfg, at::Tensor ws, at::Tensor esc, at::Tensor x, at::Tensor out, int64_t n, int64_t k,
            int64_t base4) {
    switch (cfg) {
    case 0: launch_rd<1, 2, 16, 8, 1>(ws, esc, x, out, (int)n, (int)k, (uint32_t)base4); break;
    case 1: launch_rd<1, 1, 8, 2, 1>(ws, esc, x, out, (int)n, (int)k, (uint32_t)base4); break;
    default:
        TORCH_CHECK(false, "rd: unknown config ", cfg);
    }
}
#endif
