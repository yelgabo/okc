"""m = 1 cgemv PDL-ingredient bench on the exact public BF16 bytes (5610b8a6); never submit the output.

Question (CGPDL-01): why were the layer's m = 1 cgemv kernels 3 to 4.7 us slower in the layer's
standalone cold-L2 tuner replay once tools/dl_pdl8c_patch.py --cgpdl made them PDL participants
(g1b4_pdl8c2 65f0a3be76a1499b: 15.23 / 13.06 / 8.0 us against 12.35 / 8.74 / 7.01 in pdl8c
eba613c023414ca8)? The tuner ran inside prepare with _PDL8 False, so it timed the edited kernel
launched by cudaLaunchKernelEx without the attribute. Local SASS (CUDA 12.8, sm_90a): __ldcg is
LDG.E.128.STRONG.GPU, griddepcontrol.wait is ACQBULK and every x load is issued after it.

Arms per shape (qkv_gate cgemv40<256, 2>, o cgemv<64, 1>, unpooler slice cgemv40<128, 2>), one
ingredient at a time on C2's cgemv body, each bitwise against the plain kernel:
  ctrl      the board's own cgemv_fwd (production module)
  ctrl2     the same body compiled in this bench's module (null)
  ex0/exattr  plain kernel through cudaLaunchKernelEx, without / with the PDL attribute
  cg        x through __ldcg;  ord  compiler barrier after the weight loads (order only)
  wait      griddepcontrol.wait after the weight loads (x through __ldg)
  trig      griddepcontrol.launch_dependents after the weight loads
  cgord     __ldcg plus the order barrier, no griddepcontrol
  full      __ldcg + trigger + wait (pdl8c2), launched like the tuner did; fullattr with the attribute
  c2pdl     pdl8c2's own edited source (CG_PDL_EDITS), compiled verbatim, set_pdl 0 / 1
  cgrep     __ldcg with 16 replicas of x (CTA c reads replica c % 16): L2 same-line pressure only
  st1/st2/st4  candidate: all CTAs wait, then stage x into shared memory with cp.async.cg (no x
            registers), G = 1 / 2 / 4 row pairs per CTA share one x copy; st2np without griddepcontrol
  ch*       two-kernel chain: a producer (8 MB streaming read, writes x, triggers at entry) then
            the cgemv; chP plain, chF0 / chF1 full without / with PDL, chS<G>0 / chS<G>1 the same for st<G>

    python3 tools/proj_make_bench_cgpdl.py [--set 2] BASE OUT [DEVICE_CHECK.cu]
      BASE  .build/repeats/bf16_base.py (SHA256 5610b8a6...)
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4"

CG, WAIT, TRIG, ORD, REP, STG, TRIGEND = 1, 2, 4, 8, 16, 32, 64
FULL = CG | WAIT | TRIG
SHAPES = {"qkv_gate": (256, 2, 2304), "o": (64, 1, 8192), "unpooler": (128, 2, 1024)}
# name -> (VAR, G, launch mode); mode 0 <<<>>>, 1 cudaLaunchKernelEx without attribute, 2 with PDL
ARMS = {
    "ctrl2": (0, 1, 0), "ex0": (0, 1, 1), "exattr": (0, 1, 2),
    "cg": (CG, 1, 0), "ord": (ORD, 1, 0), "wait": (WAIT, 1, 0), "trig": (TRIG, 1, 0), "cgord": (CG | ORD, 1, 0),
    "full": (FULL, 1, 1), "fullattr": (FULL, 1, 2), "cgrep": (CG | REP, 1, 0),
    "st1": (STG | WAIT | TRIG, 1, 1), "st2": (STG | WAIT | TRIG, 2, 1), "st4": (STG | WAIT | TRIG, 4, 1),
    "st2np": (STG, 2, 0),
}
CHAINS = {"chP": ("ctrl2", 0), "chF0": ("full", 1), "chF1": ("full", 2), "chS10": ("st1", 1), "chS11": ("st1", 2),
          "chS20": ("st2", 1), "chS21": ("st2", 2), "chS40": ("st4", 1), "chS41": ("st4", 2)}
NREP = 16
# CGPDL-02 (--set 2): check 1 lost the qkv_gate rows to the 16 KB log tail and showed the trigger
# before the x loads costs +1.9 us on o; this set moves the trigger to the CTA's end (TRIGEND) and
# shares one staged x copy between more row pairs. A fourth field limits an arm to some shapes.
FE = CG | WAIT | TRIGEND
SE = STG | WAIT | TRIGEND
ARMS2 = {
    "ctrl2": (0, 1, 0), "ex0": (0, 1, 1), "exattr": (0, 1, 2), "cg": (CG, 1, 0), "ord": (ORD, 1, 0),
    "wait": (WAIT, 1, 0), "trig": (TRIG, 1, 0), "trige": (TRIGEND, 1, 0), "cgrep": (CG | REP, 1, 0),
    "full": (FULL, 1, 1), "fe": (FE, 1, 1),
    "s4": (STG | WAIT | TRIG, 4, 1, ("o", "unpooler")),
    "s2e": (SE, 2, 1, ("qkv_gate", "unpooler")), "s4e": (SE, 4, 1, ("o", "unpooler")), "s8e": (SE, 8, 1, ("o",)),
}
CHAINS2 = {"chP": ("ctrl2", 0), "chF1": ("full", 2), "chFe0": ("fe", 1), "chFe1": ("fe", 2),
           "chS2e1": ("s2e", 2), "chS4e0": ("s4e", 1), "chS4e1": ("s4e", 2), "chS8e1": ("s8e", 2)}
SET = "1"


def arms_for(layer):
    nth, cpt, n = SHAPES[layer]
    out = {}
    for nm, spec in (ARMS2 if SET == "2" else ARMS).items():
        var, g, mode = spec[:3]
        if g * (nth + 32) > 1024 or (n // 2) % g or (len(spec) > 3 and layer not in spec[3]):
            continue
        out[nm] = (var, g, mode)
    return out


DEVICE = r'''
constexpr int CB_CG = 1, CB_WAIT = 2, CB_TRIG = 4, CB_ORD = 8, CB_REP = 16, CB_STG = 32, CB_TRIGEND = 64;

__device__ __forceinline__ void cb_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void cb_trig() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <int VAR, typename T>
__device__ __forceinline__ T cb_ldx(const T* p) {
    if constexpr ((VAR & CB_CG) != 0) {
        return __ldcg(p);
    } else {
        return __ldg(p);
    }
}

// C2's cgemv_body with the pdl8c2 ingredients behind VAR bits, at pdl8c2's placements.
template <int NTH, int CPT, int VAR>
__device__ __forceinline__ void cb_body(const uint8_t* __restrict__ stream, int half_n,
                                        const int2* __restrict__ table, int pmax, unsigned base4,
                                        const __nv_bfloat16* __restrict__ x0, int nrep,
                                        __nv_bfloat16* __restrict__ out) {
    constexpr int NW = NTH / 32, K = 16 * NTH * CPT;
    __shared__ float red[2][NW + 1];
    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const __nv_bfloat16* x = x0;
    if constexpr ((VAR & CB_REP) != 0) {
        x = x0 + (size_t)(c % nrep) * K;
    }
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
        if constexpr ((VAR & CB_ORD) != 0) {
            asm volatile("" ::: "memory");
        }
        if constexpr ((VAR & CB_TRIG) != 0) {
            cb_trig();
        }
        if constexpr ((VAR & CB_WAIT) != 0) {
            cb_wait();
        }
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            xa[i] = cb_ldx<VAR>(xp + i * NTH * 2);
            xb[i] = cb_ldx<VAR>(xp + i * NTH * 2 + 1);
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
        const int2* tb = table + (size_t)c * pmax;
        int2 ent = __ldg(tb + min(lane, pmax - 1));
        const int count = __shfl_sync(0xFFFFFFFFu, ent.x, 0);
        const unsigned base = base4 & 0xFFu;
        float corr0 = 0.f, corr1 = 0.f;
        if constexpr ((VAR & CB_WAIT) != 0) {
            cb_wait();
        }
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
                    (unsigned)cb_ldx<VAR>(reinterpret_cast<const unsigned short*>(x) + kk) << 16);
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
    if constexpr ((VAR & CB_TRIGEND) != 0) {
        cb_trig();
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

__device__ __forceinline__ void cb_cp16(void* smem, const void* g) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
}

// Candidate: G row pairs per CTA (each pair is C2's NTH compute threads plus one escape warp, same
// arithmetic and summation order). Every thread issues its weight / first table loads, then the
// PDL trigger and wait, then x is staged once per CTA into shared memory by cp.async.cg (L2 only,
// no registers held), so no x load waits behind the 40-register weight set.
template <int NTH, int CPT, int VAR, int G>
__device__ __forceinline__ void cb_body_stg(const uint8_t* __restrict__ stream, int half_n,
                                            const int2* __restrict__ table, int pmax, unsigned base4,
                                            const __nv_bfloat16* __restrict__ x,
                                            __nv_bfloat16* __restrict__ out) {
    constexpr int NW = NTH / 32, K = 16 * NTH * CPT, NT = G * (NTH + 32);
    __shared__ float red[G][2][NW + 1];
    __shared__ __align__(16) uint4 xs[K / 8];
    const int ta = threadIdx.x, grp = ta / (NTH + 32), t = ta - grp * (NTH + 32);
    const int c = blockIdx.x * G + grp, lane = t & 31, warp = t >> 5;
    const bool comp = warp < NW;
    const int2* tb = table + (size_t)c * pmax;
    uint4 l0[CPT], l1[CPT];
    uint2 q0[CPT], q1[CPT];
    int2 ent = make_int2(0, 0);
    if (comp) {
        const uint8_t* lp = stream + (size_t)c * 2 * K;
        const uint8_t* cp = stream + (size_t)half_n * 2 * K + (size_t)c * K;
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            const int j = i * NTH + t;
            l0[i] = ld_stream(reinterpret_cast<const uint4*>(lp) + j);
            l1[i] = ld_stream(reinterpret_cast<const uint4*>(lp + K) + j);
            q0[i] = __ldg(reinterpret_cast<const uint2*>(cp) + j);
            q1[i] = __ldg(reinterpret_cast<const uint2*>(cp + K / 2) + j);
        }
    } else {
        ent = __ldg(tb + min(lane, pmax - 1));
    }
    if constexpr ((VAR & CB_TRIG) != 0) {
        cb_trig();
    }
    if constexpr ((VAR & CB_WAIT) != 0) {
        cb_wait();
    }
    constexpr int NIT = (K / 8 + NT - 1) / NT;
#pragma unroll
    for (int j = 0; j < NIT; ++j) {
        const int i = ta + j * NT;
        if ((K / 8) % NT == 0 || i < K / 8) {
            cb_cp16(&xs[i], reinterpret_cast<const uint4*>(x) + i);
        }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (comp) {
        float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            const uint4 xa = xs[2 * (i * NTH + t)], xb = xs[2 * (i * NTH + t) + 1];
            chunk_fma(l0[i], q0[i].x, q0[i].y, xa, xb, base4, acc0);
            chunk_fma(l1[i], q1[i].x, q1[i].y, xa, xb, base4, acc1);
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
        if (lane == 0) {
            red[grp][0][warp] = acc0;
            red[grp][1][warp] = acc1;
        }
    } else {
        const int count = __shfl_sync(0xFFFFFFFFu, ent.x, 0);
        const unsigned base = base4 & 0xFFu;
        const unsigned short* xh = reinterpret_cast<const unsigned short*>(xs);
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
                const float xv = __uint_as_float((unsigned)xh[kk] << 16);
                const float d = __fmul_rn(delta, xv);
                corr0 = __fadd_rn(corr0, row ? 0.f : d);
                corr1 = __fadd_rn(corr1, row ? d : 0.f);
            }
        }
        corr0 = warp_sum(corr0);
        corr1 = warp_sum(corr1);
        if (lane == 0) {
            red[grp][0][NW] = corr0;
            red[grp][1][NW] = corr1;
        }
    }
    if constexpr ((VAR & CB_TRIGEND) != 0) {
        cb_trig();
    }
    __syncthreads();
    if (t == 0) {
        float r0 = 0.f, r1 = 0.f;
#pragma unroll
        for (int w = 0; w <= NW; ++w) {
            r0 += red[grp][0][w];
            r1 += red[grp][1][w];
        }
        *reinterpret_cast<unsigned*>(out + 2 * c) = pack_bf16x2(r0, r1);
    }
}

template <int NTH, int CPT, int VAR, int G>
__device__ __forceinline__ void cb_any(const uint8_t* __restrict__ stream, int half_n,
                                       const int2* __restrict__ table, int pmax, unsigned base4,
                                       const __nv_bfloat16* __restrict__ x, int nrep,
                                       __nv_bfloat16* __restrict__ out) {
    if constexpr ((VAR & CB_STG) != 0) {
        cb_body_stg<NTH, CPT, VAR, G>(stream, half_n, table, pmax, base4, x, out);
    } else {
        static_assert(G == 1, "row-pair groups share x only through the staged path");
        cb_body<NTH, CPT, VAR>(stream, half_n, table, pmax, base4, x, nrep, out);
    }
}

template <int NTH, int CPT, int VAR, int G>
__global__ void __maxnreg__(32)
cb_k32(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,
       unsigned base4, const __nv_bfloat16* __restrict__ x, int nrep, __nv_bfloat16* __restrict__ out) {
    cb_any<NTH, CPT, VAR, G>(stream, half_n, table, pmax, base4, x, nrep, out);
}

template <int NTH, int CPT, int VAR, int G>
__global__ void __maxnreg__(40)
cb_k40(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,
       unsigned base4, const __nv_bfloat16* __restrict__ x, int nrep, __nv_bfloat16* __restrict__ out) {
    cb_any<NTH, CPT, VAR, G>(stream, half_n, table, pmax, base4, x, nrep, out);
}

// The staged path holds all 2 * CPT weight chunks across the barrier; at CPT = 2 that spills 8 bytes
// under 40 registers (local sm_90a build), and at G >= 2 a 48 cap leaves the CTAs per SM unchanged.
template <int NTH, int CPT, int VAR, int G>
__global__ void __maxnreg__(48)
cb_k48(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,
       unsigned base4, const __nv_bfloat16* __restrict__ x, int nrep, __nv_bfloat16* __restrict__ out) {
    cb_any<NTH, CPT, VAR, G>(stream, half_n, table, pmax, base4, x, nrep, out);
}

using cb_fn = void (*)(const uint8_t*, int, const int2*, int, unsigned, const __nv_bfloat16*, int, __nv_bfloat16*);

template <int NTH, int CPT, int VAR, int G>
cb_fn cb_ptr() {
    if constexpr (CPT == 1) {
        return &cb_k32<NTH, CPT, VAR, G>;
    } else if constexpr ((VAR & CB_STG) != 0) {
        return &cb_k48<NTH, CPT, VAR, G>;
    } else {
        return &cb_k40<NTH, CPT, VAR, G>;
    }
}

template <int NTH, int CPT, int VAR, int G>
cudaError_t cb_launch(int mode, const uint8_t* sp, int half_n, const int2* tp, int pmax, unsigned base4,
                      const __nv_bfloat16* xp, int nrep, __nv_bfloat16* op, cudaStream_t s) {
    cb_fn k = cb_ptr<NTH, CPT, VAR, G>();
    const dim3 grid((unsigned)(half_n / G)), block((unsigned)(G * (NTH + 32)));
    if (mode == 0) {
        k<<<grid, block, 0, s>>>(sp, half_n, tp, pmax, base4, xp, nrep, op);
        return cudaGetLastError();
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = mode == 2 ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, k, sp, half_n, tp, pmax, base4, xp, nrep, op);
}

template <int NTH, int CPT, int VAR, int G>
void cb_info_one(int64_t* r) {
    cb_fn k = cb_ptr<NTH, CPT, VAR, G>();
    cudaFuncAttributes a{};
    cudaFuncGetAttributes(&a, k);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, G * (NTH + 32), 0);
    r[0] = a.numRegs;
    r[1] = (int64_t)a.localSizeBytes;
    r[2] = (int64_t)a.sharedSizeBytes;
    r[3] = blocks;
    r[4] = G * (NTH + 32);
}

// Chain primary: triggers at entry, streams nbig4 float4 through L2 (a predecessor's own traffic),
// then writes x. Its dependents read x only after griddepcontrol.wait.
__global__ void __launch_bounds__(256) cb_prod_kernel(const float4* __restrict__ big, int nbig4,
                                                      float* __restrict__ sink, const uint4* __restrict__ src,
                                                      uint4* __restrict__ x, int nx4) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float a = 0.f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < nbig4; i += gridDim.x * 256) {
        const float4 v = __ldcs(big + i);
        a += v.x + v.y + v.z + v.w;
    }
    if (a == 1234.5f) {
        sink[threadIdx.x] = a;
    }
    for (int i = blockIdx.x * 256 + threadIdx.x; i < nx4; i += gridDim.x * 256) {
        x[i] = src[i];
    }
}
'''

HOST = r'''
#ifndef GEMV_DEVICE_ONLY
void cb_fwd(int64_t id, int64_t mode, at::Tensor stream, at::Tensor table, int64_t pmax, int64_t base4,
            at::Tensor x, at::Tensor out, int64_t nrep) {
    TORCH_CHECK(stream.is_cuda() && stream.scalar_type() == at::kByte && stream.is_contiguous()
                && stream.dim() == 3 && stream.size(2) == 48, "cb_fwd: bad stream");
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kInt && table.is_contiguous(), "cb_fwd: bad table");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous()
                && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0
                && x.numel() >= stream.size(1) * 16 * nrep, "cb_fwd: bad x");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.numel() == 2 * stream.size(0),
                "cb_fwd: bad out");
    const int half_n = (int)stream.size(0);
    const int64_t chunks = stream.size(1);
    const uint8_t* sp = stream.data_ptr<uint8_t>();
    const int2* tp = reinterpret_cast<const int2*>(table.data_ptr<int32_t>());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t e = cudaErrorInvalidValue;
    int nth = 0, cpt = 0;
    switch (id) {
@@LAUNCH@@
    default:
        TORCH_CHECK(false, "cb_fwd: unknown arm ", id);
    }
    TORCH_CHECK(chunks == (int64_t)nth * cpt, "cb_fwd: arm ", id, " expects K/16 = ", nth * cpt, ", got ", chunks);
    C10_CUDA_CHECK(e);
}

std::vector<int64_t> cb_info(int64_t id) {
    std::vector<int64_t> r(5, 0);
    switch (id) {
@@INFO@@
    default:
        TORCH_CHECK(false, "cb_info: unknown arm ", id);
    }
    return r;
}

void cb_prod(at::Tensor big, at::Tensor sink, at::Tensor src, at::Tensor x, int64_t nblk) {
    TORCH_CHECK(big.is_cuda() && big.scalar_type() == at::kFloat && big.is_contiguous() && big.numel() % 4 == 0,
                "cb_prod: bad big");
    TORCH_CHECK(src.numel() == x.numel() && src.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16
                && src.is_contiguous() && x.is_contiguous() && x.numel() % 8 == 0, "cb_prod: bad x");
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    cb_prod_kernel<<<(unsigned)nblk, 256, 0, s>>>(
        reinterpret_cast<const float4*>(big.data_ptr<float>()), (int)(big.numel() / 4), sink.data_ptr<float>(),
        reinterpret_cast<const uint4*>(src.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint4*>(x.data_ptr<at::BFloat16>()), (int)(x.numel() / 8));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
'''

CPP = """
void cb_fwd(int64_t id, int64_t mode, at::Tensor stream, at::Tensor table, int64_t pmax, int64_t base4, at::Tensor x, at::Tensor out, int64_t nrep);
std::vector<int64_t> cb_info(int64_t id);
void cb_prod(at::Tensor big, at::Tensor sink, at::Tensor src, at::Tensor x, int64_t nblk);
"""


def arm_table():
    """[(id, layer, arm, NTH, CPT, VAR, G, mode)] for every compiled (layer, arm)."""
    rows = []
    for layer, (nth, cpt, n) in SHAPES.items():
        for nm, (var, g, mode) in arms_for(layer).items():
            rows.append((len(rows), layer, nm, nth, cpt, var, g, mode))
    return rows


def cuda_source(gemv_src):
    head, sep, tail = gemv_src.rpartition("}  // namespace")
    assert sep and "}  // namespace" not in head, "GEMV_CUDA_SRC namespace anchor"
    rows = arm_table()
    launch = "\n".join(f"    case {i}: nth = {nth}; cpt = {cpt}; e = cb_launch<{nth}, {cpt}, {var}, {g}>((int)mode, sp, "
                       f"half_n, tp, (int)pmax, (unsigned)base4, xp, (int)nrep, op, s); break;"
                       for i, _, _, nth, cpt, var, g, _ in rows)
    info = "\n".join(f"    case {i}: cb_info_one<{nth}, {cpt}, {var}, {g}>(r.data()); break;"
                     for i, _, _, nth, cpt, var, g, _ in rows)
    return head + DEVICE + sep + tail + HOST.replace("@@LAUNCH@@", launch).replace("@@INFO@@", info)


RUN = r'''
import random as _cbrandom
import statistics as _cbstats
_CB_ROWS = @@ROWS@@
_CB_CHAINS = @@CHAINS@@
_CB_NREP = @@NREP@@
_CB_SRC = @@SRC@@
_CB_CPP = @@CPP@@
_CB_C2PDL_EDITS = @@C2EDITS@@
_CB_COMPACT = @@COMPACT@@
_CBLOG = []


def _cbprint(s):
    _CBLOG.append(s)


def _cbload(name, cpp, src, funcs):
    kw = dict(name=name, cpp_sources="#include <torch/extension.h>\n" + cpp, cuda_sources=src, functions=funcs,
              extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    try:
        return load_inline(no_implicit_headers=True, **kw)
    except TypeError:
        kw["cuda_sources"] = src.replace("#include <ATen/core/Tensor.h>", "#include <torch/types.h>")
        return load_inline(**kw)


def _cbc2pdl_source():
    src = GEMV_CUDA_SRC
    for old, new, n in _CB_C2PDL_EDITS:
        if src.count(old) != n:
            raise RuntimeError(f"c2pdl anchor {old[:50]!r} found {src.count(old)}")
        src = src.replace(old, new)
    return src


def _cbbench(weights, state, started):
    t1 = time.time()
    mod = _cbload("okc_cgpdl_bench", _CB_CPP, _CB_SRC, ["cb_fwd", "cb_info", "cb_prod"])
    c2 = _cbload("okc_cgpdl_c2pdl", GEMV_CPP_DECLS + "void cg_set_pdl(int64_t on);\n", _cbc2pdl_source(),
                 ["cgemv_fwd", "cg_set_pdl"])
    _cbprint(f"CB build {time.time() - t1:.1f} s")
    ext = state.get("wd_ext")
    dev = state["o"].device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    res = {}
    for i, layer, nm, nth, cpt, var, g, mode in _CB_ROWS:
        r = mod.cb_info(i)
        if _CB_COMPACT:
            res.setdefault(layer, []).append(f"{nm}:{r[0]}r/{r[1]}l/{r[3]}b")
        else:
            _cbprint(f"CBRES {layer} {nm} var={var} G={g} mode={mode} regs={r[0]} local={r[1]} static={r[2]} "
                     f"blocks/SM={r[3]} block={r[4]}")
    for layer, v in res.items():
        _cbprint(f"CBRES {layer} regs/local/blocksSM " + " ".join(v))
    big = torch.ones(2 * 1024 * 1024, dtype=torch.float32, device=dev)
    sink = torch.zeros(256, dtype=torch.float32, device=dev)
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261103)
    for layer in ("qkv_gate", "o", "unpooler"):
        if time.time() - started > 250:
            _cbprint(f"CBSTOP time guard before {layer}")
            break
        w = state[layer][:UNPOOL_SLICE] if layer == "unpooler" else state[layer]
        w = w.contiguous()
        n, k = w.shape
        p = pack_cgemv(w)
        if p is None:
            _cbprint(f"CBSTOP {layer} pack_cgemv: too many escapes")
            continue
        nth = cgemv_threads(n, k)
        rows = {nm: (i, var, g, mode) for i, ly, nm, _, _, var, g, mode in _CB_ROWS if ly == layer}
        xbuf = torch.zeros(1, k, dtype=torch.bfloat16, device=dev)
        rows_mode = {i: mode for i, _, _, mode in rows.values()}

        def mk(i, nrep=1):
            def run(x):
                out = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
                mod.cb_fwd(i, rows_mode[i], p["stream"], p["table"], p["pmax"], p["base4"], x, out, nrep)
                return out
            return run

        def ctrl(x):
            out = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
            ext.cgemv_fwd(p["stream"], p["table"], p["pmax"], p["base4"], x, out, nth)
            return out

        def c2run(on):
            def run(x):
                out = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
                c2.cg_set_pdl(on)
                try:
                    c2.cgemv_fwd(p["stream"], p["table"], p["pmax"], p["base4"], x, out, nth)
                finally:
                    c2.cg_set_pdl(0)
                return out
            return run
        # tag -> (fn, feed): feed maps the round's x to what fn takes
        plainfeed = lambda x, xr: x
        repfeed = lambda x, xr: xr
        arms = {"ctrl": (ctrl, plainfeed), "c2pdl": (c2run(0), plainfeed)}
        if not _CB_COMPACT:
            arms["c2pdlattr"] = (c2run(1), plainfeed)
        for nm, (i, var, g, mode) in rows.items():
            rep = var & 16
            arms[nm] = (mk(i, _CB_NREP if rep else 1), repfeed if rep else plainfeed)
        for cnm, (sec, smode) in _CB_CHAINS.items():
            if sec not in rows:
                continue
            i = rows[sec][0]

            def chain(x, i=i, smode=smode):
                mod.cb_prod(big, sink, x, xbuf, nsm)
                out = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
                mod.cb_fwd(i, smode, p["stream"], p["table"], p["pmax"], p["base4"], xbuf, out, 1)
                return out
            arms[cnm] = (chain, plainfeed)
        failed = set()
        for trial in range(4):
            x, probes = _gen_x(1, k, gen, dev)
            xr = x.repeat(1, _CB_NREP).contiguous()
            base = arms["ctrl2"][0](x)
            torch.cuda.synchronize()
            ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
            ok, err, perr = _check(base, ref, probes, SELFTEST_TOL["bf16"])
            if not ok:
                _cbprint(f"CBSTOP {layer} ctrl2 FP64 err={err:.3e} probe={perr:.3e}")
                failed.add("ctrl2")
                break
            for tag, (fn, feed) in arms.items():
                if tag in failed:
                    continue
                try:
                    out = fn(feed(x, xr))
                    torch.cuda.synchronize()
                    if not torch.equal(out, base):
                        failed.add(tag)
                        _cbprint(f"CBFAIL {layer} {tag} trial {trial} not bitwise equal to ctrl2 "
                                 f"({int((out != base).sum())} of {n} differ)")
                except Exception as exc:
                    torch.cuda.synchronize()
                    failed.add(tag)
                    _cbprint(f"CBFAIL {layer} {tag} {type(exc).__name__}: {str(exc)[-300:]}")
        if "ctrl2" in failed or "ctrl" in failed:
            _cbprint(f"CBSTOP {layer} a control failed")
            continue
        for tag in failed:
            arms.pop(tag, None)
        _cbprint(f"CBCHECK {layer} FP64+probe ctrl2, 4 inputs bitwise to ctrl2: "
                 + (f"{len(arms)} PASS, failed {sorted(failed)}" if _CB_COMPACT else f"PASS {sorted(arms)}"))
        for tag, (fn, feed) in list(arms.items()):
            x, _ = _gen_x(1, k, gen, dev)
            xr = x.repeat(1, _CB_NREP).contiguous()
            arg = feed(x, xr)
            fn(arg)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                out = fn(arg)
            for _ in range(3):
                x2, _ = _gen_x(1, k, gen, dev)
                arg.copy_(feed(x2, x2.repeat(1, _CB_NREP)))
                gr.replay()
                torch.cuda.synchronize()
                if not torch.equal(out, arms["ctrl2"][0](x2)):
                    arms.pop(tag)
                    _cbprint(f"CBFAIL {layer} {tag} captured replay not bitwise equal")
                    break
        _cbprint(f"CBREPLAY {layer} 3 fresh captured inputs bitwise: "
                 + (f"{len(arms)} PASS" if _CB_COMPACT else f"PASS {sorted(arms)}"))
        rounds = 100
        graphs = {}
        acc = {}
        for inp in range(3):
            if time.time() - started > 250:
                _cbprint(f"CBSTOP time guard {layer} input {inp}")
                break
            x, _ = _gen_x(1, k, gen, dev)
            xr = x.repeat(1, _CB_NREP).contiguous()
            graphs = {}
            for tag, (fn, feed) in arms.items():
                arg = feed(x, xr)
                fn(arg)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn(arg)
                graphs[tag] = gr
            times = {tag: [] for tag in arms}
            rng = _cbrandom.Random(20261103 + inp)
            for rd in range(rounds):
                order = list(arms)
                rng.shuffle(order)
                for tag in order:
                    flush.zero_()
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    graphs[tag].replay()
                    b.record()
                    b.synchronize()
                    times[tag].append(1000 * a.elapsed_time(b))
            null = [p_ - q_ for p_, q_ in zip(times["ctrl2"], times["ctrl"])]
            nmed = _cbstats.median(null)
            nmad = _cbstats.median(abs(z - nmed) for z in null)
            _cbprint(f"CBNULL {layer} in{inp} ctrl2-ctrl={nmed:+.3f} MAD={nmad:.3f} ctrl={_cbstats.median(times['ctrl']):.3f}")
            for tag, ts in times.items():
                refs = ("chP", "chF0") if tag.startswith("ch") else ("ctrl2", "full")
                if _CB_COMPACT:
                    refs = ("chP",) if tag.startswith("ch") else ("ctrl2",)
                parts = [f"raw={_cbstats.median(ts):.3f}"]
                a = acc.setdefault(tag, {"raw": []})
                a["raw"].append(f"{_cbstats.median(ts):.2f}")
                for rf in refs:
                    if rf in times and rf != tag:
                        d = [p_ - q_ for p_, q_ in zip(ts, times[rf])]
                        parts.append(f"d_{rf}={_cbstats.median(d):+.3f} lt={sum(z < 0 for z in d)}/{rounds}")
                        a.setdefault(rf, []).append(f"{_cbstats.median(d):+.2f}({sum(z < 0 for z in d)})")
                if not _CB_COMPACT:
                    _cbprint(f"CBTIME {layer} in{inp} {tag} " + " ".join(parts))
        if _CB_COMPACT:
            for tag, a in acc.items():
                extra = " ".join(f"d_{rf} " + "/".join(v) for rf, v in a.items() if rf != "raw")
                _cbprint(f"CBT {layer} {tag} raw " + "/".join(a["raw"]) + " " + extra)
        del graphs
    _cbprint(f"CBDONE elapsed={time.time() - started:.1f} s")


_cb_original_prepare = prepare


def prepare(weights):
    started = time.time()
    state = _cb_original_prepare(weights)
    try:
        _cbbench(weights, state, started)
    except Exception as exc:
        torch.cuda.synchronize()
        _cbprint(f"CBERROR {type(exc).__name__}: {str(exc)[-1500:]}")
    for line in _CBLOG:
        print(line, flush=True)
    return state
'''


def c2pdl_edits():
    spec = importlib.util.spec_from_file_location("dl_pdl8c_patch", str(ROOT / "tools/dl_pdl8c_patch.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CG_PDL_EDITS


def gemv_src(base):
    return next(ast.literal_eval(nd.value) for nd in ast.parse(base).body if isinstance(nd, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "GEMV_CUDA_SRC" for t in nd.targets))


def make(base):
    src = cuda_source(gemv_src(base))
    out = (base.rstrip("\n") + "\n" + RUN.replace("@@ROWS@@", repr(arm_table())).replace("@@CHAINS@@", repr(CHAINS2 if SET == "2" else CHAINS))
           .replace("@@NREP@@", repr(NREP)).replace("@@SRC@@", repr(src)).replace("@@CPP@@", repr(CPP))
           .replace("@@C2EDITS@@", repr(c2pdl_edits())).replace("@@COMPACT@@", repr(SET == "2")))
    ast.parse(out)
    return out, src


def device_check_source(base):
    """The bench's device code and launchers with stub host macros, for a local sm_90a compile."""
    src = cuda_source(gemv_src(base))
    dev = src.split('\n#ifndef GEMV_DEVICE_ONLY\nvoid cb_fwd(')[0]
    calls = "\n".join(f"    cb_launch<{nth}, {cpt}, {var}, {g}>(0, 0, 0, 0, 0, 0u, 0, 1, 0, 0);"
                      for _, _, _, nth, cpt, var, g, _ in arm_table())
    return "#define GEMV_DEVICE_ONLY\n" + dev + "\nvoid cb_all() {\n" + calls + "\n}\n"


def _stub_torch():
    class Any:
        def __getattr__(self, name):
            return Any()

        def __call__(self, *a, **k):
            return Any()

    def mod(name):
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: Any()
        return m
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch.utils", "torch.utils.cpp_extension"):
        sys.modules.setdefault(name, mod(name))


def host_gate(text):
    """Execute the bench file on CPU with stub torch: the c2pdl edits must apply to its GEMV source,
    and every arm must have a launch and an info case."""
    _stub_torch()
    ns = {"__name__": "cg_pdl_gate"}
    exec(compile(text, "cg_pdl.py", "exec"), ns)
    c2 = ns["_cbc2pdl_source"]()
    assert "griddepcontrol.wait" in c2 and "cg_set_pdl" in c2
    src = ns["_CB_SRC"]
    for i, layer, nm, nth, cpt, var, g, mode in ns["_CB_ROWS"]:
        assert f"case {i}: nth = {nth}; cpt = {cpt}; e = cb_launch<{nth}, {cpt}, {var}, {g}>" in src, nm
        assert f"case {i}: cb_info_one<{nth}, {cpt}, {var}, {g}>" in src, nm
    for cnm, (sec, smode) in ns["_CB_CHAINS"].items():
        assert sec in (ARMS2 if SET == "2" else ARMS) and smode in (0, 1, 2)
    return len(ns["_CB_ROWS"])


if __name__ == "__main__":
    if sys.argv[1] == "--set":
        SET = sys.argv[2]
        del sys.argv[1:3]
    base_path, out_path = sys.argv[1:3]
    base = Path(base_path).read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE_SHA:
        sys.exit("base bytes are not the public BF16 file 5610b8a6")
    text, src = make(base.decode("utf8"))
    narms = host_gate(text)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(device_check_source(base.decode("utf8")))
    Path(out_path).write_text(text, encoding="utf-8")
    data = Path(out_path).read_bytes()
    print(out_path, hashlib.sha256(data).hexdigest(), len(data), f"{narms} compiled arms")
