#!/usr/bin/env python3
"""Instruction-level micro-changes to the fused MoE of rv_k_combo2n.py (3ac83fb4), one per --variant.

    python3 tools/dl_ly_micro_patch.py --variant scl [--variant ftz ...] [--out muse_decode_layer/cand/ly_mo_<n>.py]

Variants (every one keeps the output bits; docs/runs-decode-layer.md "Micro-optimizations on the box"):
  scl     helper0 skinny_ring_ws (TINY / TINY2 / MID unit loop): both rows' block scales in one bf16x2 (one PRMT and
          one IMAD per 64-K chunk instead of two each), the multiplies broadcast its halves, as the WGS stage does.
  scl_notiny  scl on the TINY2 and MID builds only: the TINY build (decode 1) gets -DOKC_NOSCL=1 and preprocesses
          to the base loop (decode 1 sliding measured +0.36 to +0.55 us with scl, verify 1 / decode 16 -0.8 to -0.9).
  ftz     every MoE SwiGLU epilogue: exp through ex2.approx.ftz. The non-ftz __expf adds a compare and two predicated
          multiplies for results below 2^-126; there 1 + e^-g is 1.0f either way, so the SiLU bits do not change.
  wspack  helper0 ws_helper phase A: the two tokens' BF16 stores become one 32-bit store; lanes 4 apart (adjacent
          columns) swap one value with a shuffle, even groups store token 2t, odd groups token 2t + 1.
  wsmask  helper0 ws_helper phase A: the SwiGLU math runs for both tokens unconditionally, only the store is guarded.
  wgspack MOE_CUDA_SRC wgs_epi_tile phase A (B4 / MAIN): the same shuffle pairing into one 32-bit store into xB.
  cgpf    T = 1 unpooler cgemv (the PDL8 edit of cgemv_body): thread 0 of each CTA prefetches the CTA's weight bytes
          (2K codes, K scales) into L2 before griddepcontrol.wait. ptxas otherwise sinks the weight loads below the
          wait, so the 6.3 MB stream starts only when the MoE has finished. The weights are prepare-time constants.
No fence, barrier, mbarrier, griddepcontrol or launch attribute changes; no job sizing or range change.
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/rv_k_combo2n.py"
PIN = "3ac83fb437d05308c2871301a5a10c537d9b2ec504d55305403641beec2f5d27"
H0_START = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
MAIN_START = "MOE_CUDA_SRC = r'''\n"


def region(s, start, end):
    i = s.index(start)
    return i, s.index(end, i + len(start))


def once(t, old, new, what):
    n = t.count(old)
    if n != 1:
        sys.exit(f"{what}: anchor found {n} times")
    return t.replace(old, new)


def in_region(s, start, end, edits, what, sub=None):
    i, j = region(s, start, end)
    if sub:
        a = s.index(sub[0], i)
        i, j = a, s.index(sub[1], a)
    t = s[i:j]
    for old, new in edits:
        t = once(t, old, new, what)
    return s[:i] + t + s[j:]


SCL_OLD = """      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else"""
SCL_NEW = """      const uint32_t sbp = (uint32_t)(c & 1) * 2u;
      const uint32_t sgp = __byte_perm(swd, 0u, 0x4040u | sbp | ((sbp + 1u) << 8)) * k16 + 0x5C005C00u;
#else"""
SCL_MMA_OLD = """        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    // WSB:"""
SCL_MMA_NEW = """#if OKC_PRMTSC
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2l(r0, sgp), bmul2h(r1, sgp), bmul2l(r2, sgp), bmul2h(r3, sgp), b[2 * st], b[2 * st + 1]);
#else
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
#endif
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    // WSB:"""

FTZ_HELPER = """__device__ __forceinline__ float fork_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
"""
FTZ_HELPER_NEW = FTZ_HELPER + """// e^-g for the SiLU: ftz differs from __expf only below 2^-126, where 1 + e^-g rounds to 1.0f either way
__device__ __forceinline__ float exp_neg_ftz(float g) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(g * -1.4426950408889634f));
  return r;
}
"""

WS_A_OLD = """#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {"""
WS_PACK_NEW = """      float sv[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
        sv[h] = probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up);
      }
      {  // lanes 4 apart hold adjacent columns: even gg keeps token 2tt for (gg, gg + 1), odd gg token 2tt + 1
        const int od = gg & 1;
        const float give = __shfl_xor_sync(0xffffffffu, od ? sv[0] : sv[1], 4);
        const int tk = 2 * tt + od;
        const __nv_bfloat162 pr = od ? __floats2bfloat162_rn(give, sv[1]) : __floats2bfloat162_rn(sv[0], give);
        if (tk < cnt) st_keep_bf16x2(p.inter + ((size_t)(st + tk) * ID + rb * 8 + (gg & ~1)), pr);
      }
      if (q == nA - 1) {"""
WS_MASK_NEW = """#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
        const __nv_bfloat16 o = __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up));
        if (tk < cnt) st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), o);
      }
      if (q == nA - 1) {"""

ST2_ANCHOR = """__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
"""
ST2_NEW = ST2_ANCHOR + """__device__ __forceinline__ void st_keep_bf16x2(__nv_bfloat16* a, __nv_bfloat162 v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "r"(*reinterpret_cast<unsigned*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
"""
ST2_PLAIN = """__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
"""
ST2_PLAIN_NEW = ST2_PLAIN + """__device__ __forceinline__ void st_keep_bf16x2(__nv_bfloat16* a, __nv_bfloat162 v) { *reinterpret_cast<__nv_bfloat162*>(a) = v; }
"""

WGS_A_OLD = """#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = fork_bf16(d4[h] * gsc), up = fork_bf16(d4[2 + h] * gsc);
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }"""
WGS_A_NEW = """    float sv[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float gt = fork_bf16(d4[h] * gsc), up = fork_bf16(d4[2 + h] * gsc);
      sv[h] = probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up);
    }
    {  // lanes 4 apart hold adjacent columns: even g keeps token 2t for (col, col + 1), odd g token 2t + 1
      const int od = g & 1;
      const float give = __shfl_xor_sync(0xffffffffu, od ? sv[0] : sv[1], 4);
      const int tk = nt * 8 + 2 * t + od;
      const __nv_bfloat162 pr = od ? __floats2bfloat162_rn(give, sv[1]) : __floats2bfloat162_rn(sv[0], give);
      if (tk < cnt)
        st_keep_bf16x2(xb - od + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8), pr);
    }"""


CG_OLD = """'        asm volatile("griddepcontrol.wait;" ::: "memory");\\n#pragma unroll\\n        for (int i = 0; i < CPT; ++i) {\\n            xa[i] = __ldcg(xp"""
CG_NEW = ("""'        if (t == 0) {\\n"""
          """            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(lp), "r"(2 * K) : "memory");\\n"""
          """            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(cp), "r"(K) : "memory");\\n"""
          """        }\\n""" + CG_OLD[1:])


SCLN_NEW = ("#if !OKC_NOSCL\n" + SCL_NEW[:-len("#else")] + "#else\n" + SCL_OLD[:-len("#else")] + "#endif\n#else")
SCLN_MMA_NEW = SCL_MMA_NEW.replace("#if OKC_PRMTSC\n", "#if OKC_PRMTSC && !OKC_NOSCL\n", 1)
TINY_DEF = 'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1")'


def apply(s, v):
    if v == "scl_notiny":
        s = once(s, TINY_DEF, TINY_DEF[:-1] + ', "-DOKC_NOSCL=1")', v)
        return in_region(s, H0_START, '\n"""', [(SCL_OLD, SCLN_NEW), (SCL_MMA_OLD, SCLN_MMA_NEW)], v,
                         sub=("__device__ void skinny_ring_ws(", "__device__ void ws_helper("))
    if v == "cgpf":
        return once(s, CG_OLD, CG_NEW, v)
    if v == "scl":
        return in_region(s, H0_START, '\n"""', [(SCL_OLD, SCL_NEW), (SCL_MMA_OLD, SCL_MMA_NEW)], v,
                         sub=("__device__ void skinny_ring_ws(", "__device__ void ws_helper("))
    if v == "ftz":
        for start, end in ((H0_START, '\n"""'), (MAIN_START, "\n'''")):
            i, j = region(s, start, end)
            t = once(s[i:j], FTZ_HELPER, FTZ_HELPER_NEW, v)
            n = t.count("1.f + __expf(-gt)")
            if n < 2:
                sys.exit(f"{v}: only {n} SiLU sites")
            t = t.replace("1.f + __expf(-gt)", "1.f + exp_neg_ftz(gt)")
            s = s[:i] + t + s[j:]
        return s
    if v == "wspack":
        s = in_region(s, H0_START, '\n"""', [(ST2_ANCHOR, ST2_NEW), (ST2_PLAIN, ST2_PLAIN_NEW)], v)
        return in_region(s, H0_START, '\n"""', [(WS_A_OLD, WS_PACK_NEW)], v)
    if v == "wsmask":
        return in_region(s, H0_START, '\n"""', [(WS_A_OLD, WS_MASK_NEW)], v)
    if v == "wgspack":
        s = in_region(s, MAIN_START, "\n'''", [(ST2_ANCHOR, ST2_NEW), (ST2_PLAIN, ST2_PLAIN_NEW)], v)
        return in_region(s, MAIN_START, "\n'''", [(WGS_A_OLD, WGS_A_NEW)], v)
    sys.exit(f"unknown variant {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("base SHA mismatch")
    s = raw.decode()
    # ftz rewrites every SiLU site, including the ones the pack variants insert, so it goes last
    for v in sorted(a.variant, key=lambda v: v == "ftz"):
        s = apply(s, v)
    out = ROOT / a.out
    out.write_text(s)
    py_compile.compile(str(out), doraise=True)
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
