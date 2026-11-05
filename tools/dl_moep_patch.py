"""Port the layer's shared MoE changes (ly_stk7c) into the standalone muse_moe_w4a16 base.

Usage: python3 tools/dl_moep_patch.py BASE OUT

Every edit is scoped to one region (MOE_CUDA_SRC, the moe_helper0 source, or the Python
selection code) and must match exactly the stated number of times, or the tool aborts.
Layer-only context (PDL launch, unpooler fusion, route kernels) is not ported.
"""
import hashlib
import py_compile
import sys

BASE_SHA = "5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e"

src_path, out_path = sys.argv[1], sys.argv[2]
raw = open(src_path, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
if sha != BASE_SHA:
    sys.exit("base sha %s != pinned %s" % (sha, BASE_SHA))
text = raw.decode()


def span(start_marker):
    a = text.index(start_marker) + len(start_marker)
    b = text.index("'''", a)
    return a, b


m_a, m_b = span("MOE_CUDA_SRC = r'''")
h_a, h_b = span("_MA_ALT_SRCS = {'moe_helper0': r'''")
assert m_b < h_a
regions = {"head": text[:m_a], "main": text[m_a:m_b], "helper": text[h_a:h_b], "py": text[h_b:]}
log = []


def rep(reg, old, new, count=1, tag=""):
    s = regions[reg]
    n = s.count(old)
    if n != count:
        sys.exit("[%s] %s: expected %d matches, found %d" % (reg, tag, count, n))
    regions[reg] = s.replace(old, new)
    log.append("%-6s %-22s x%d" % (reg, tag, count))


def rep_near(reg, anchor, old, new, window, tag=""):
    """Replace the first `old` after the unique `anchor`, which must start within `window` chars of it."""
    s = regions[reg]
    if s.count(anchor) != 1:
        sys.exit("[%s] %s: anchor matches %d" % (reg, tag, s.count(anchor)))
    i = s.index(anchor)
    j = s.find(old, i)
    if j < 0 or j - i > window:
        sys.exit("[%s] %s: target not within %d chars of anchor" % (reg, tag, window))
    regions[reg] = s[:j] + new + s[j + len(old):]
    log.append("%-6s %-22s near" % (reg, tag))


ST_KEEP_X2_L2H = (
    '__device__ __forceinline__ void st_keep_bf16x2(__nv_bfloat16* a, __nv_bfloat162 v) {\n'
    '  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "r"(*reinterpret_cast<unsigned*>(&v)), '
    '"l"(0x14F0000000000000ull) : "memory");\n'
    '}\n')
ST_KEEP_X2_PLAIN = ('__device__ __forceinline__ void st_keep_bf16x2(__nv_bfloat16* a, __nv_bfloat162 v) '
                    '{ *reinterpret_cast<__nv_bfloat162*>(a) = v; }\n')
FORK = "__device__ __forceinline__ float fork_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }\n"
EXP_NEG_FTZ = (
    "// exp(-g) as one ex2.approx.ftz: the layer's ly_micro2 'ftz' change, output bits unchanged on the box\n"
    "__device__ __forceinline__ float exp_neg_ftz(float g) {\n"
    "  float r;\n"
    '  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(g * -1.4426950408889634f));\n'
    "  return r;\n"
    "}\n")
CREDIT_OLD = ("        __threadfence();\n"
              "        __syncwarp();\n"
              "        if (lane == 0) {\n"
              "          const int jf")
CREDIT_NEW = ("        __syncwarp();\n"
              "        if (lane == 0) {\n"
              "          // one fence.sc by the crediting lane after the warp barrier orders every lane's stores\n"
              "          // before the atomics (docs/lanes/moe-wsrel2.md); a per-lane fence before __syncwarp did not\n"
              '          asm volatile("fence.sc.gpu;" ::: "memory");\n'
              "          const int jf")

for reg in ("main", "helper"):
    rep(reg, '#else\n#define L2H_SUFFIX ""\n', ST_KEEP_X2_L2H + '#else\n#define L2H_SUFFIX ""\n', tag="st_keep_bf16x2 L2H")
    rep(reg, "__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }\n",
        "__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }\n" + ST_KEEP_X2_PLAIN,
        tag="st_keep_bf16x2 plain")
    rep(reg, FORK, FORK + EXP_NEG_FTZ, tag="exp_neg_ftz def")
    rep(reg, CREDIT_OLD, CREDIT_NEW, tag="doneA fence fix")

# MOE_CUDA_SRC: MFIN on B4 (ly_gb4) and on the RFIN main build (ly_gbm)
rep("main", "#if OKC_WGS\n#undef OKC_MFIN\n#define OKC_MFIN 0\n#endif\n",
    "#if OKC_WGS && !OKC_B4FIN && !OKC_RFIN\n#undef OKC_MFIN\n#define OKC_MFIN 0\n#endif\n", tag="MFIN gate")
rep("main", "for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }",
    "while (ld_acquire(cp) < target) { if (OKC_MFS) __nanosleep(OKC_MFS); }", count=2, tag="MFIN polls unbounded")

# MOE_CUDA_SRC: RFIN (ly_rfin): cut units finished in registers by the share holding stage 0
rep("main", "#define OKC_WGSREORD 1  //",
    "#ifndef OKC_RFIN\n#define OKC_RFIN 0\n#endif\n#ifndef OKC_RFINQ\n#define OKC_RFINQ 2\n#endif\n"
    "#define OKC_WGSREORD (!OKC_RFIN)  //", tag="RFIN defines")
rep("main", "#define OKC_WGSDFIN 2  //", "#define OKC_WGSDFIN (OKC_RFIN ? 3 : 2)  //", tag="WGSDFIN")
rep("main", "  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }\n",
    "#if OKC_RFIN\n  pd.kind = 0;\n  return;\n#endif\n"
    "  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }\n", tag="RFIN flush return")
RFIN_SLOT = r"""#if OKC_RFIN
  if (slot >= 0) {  // RFIN: shares not holding stage 0 publish their partials; the stage-0 share sums them in registers
    if (s0 != 0) {
      float* fb = wgs_fix(p, PB) + (size_t)(blockIdx.x * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt)
        *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
      pd = WgsPend{2, u, slot, s1 - s0};
      return;
    }
    int* fc = wgs_cnt(p, PB) + slot * 4 + wg;
    if (wt == 0) { while (ld_acquire(fc) < 16 - (s1 - s0)) __nanosleep(32); }
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    const int c1 = wgs_cta_of(S, PB, (u << 4) | 15, gridDim.x);
#pragma unroll 1
    for (int cc = (int)blockIdx.x + 1; cc <= c1; ++cc) {
      const float* fb = wgs_fix(p, PB) + (size_t)(cc * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
      for (int nt0 = 0; nt0 < NTT; nt0 += OKC_RFINQ) {
        float4 v[OKC_RFINQ];
#pragma unroll
        for (int k = 0; k < OKC_RFINQ; ++k)
          if (nt0 + k < NTT) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
        for (int k = 0; k < OKC_RFINQ; ++k)
          if (nt0 + k < NTT) {
            d[nt0 + k][0] += v[k].x; d[nt0 + k][1] += v[k].y; d[nt0 + k][2] += v[k].z; d[nt0 + k][3] += v[k].w;
            *reinterpret_cast<float4*>(const_cast<float*>(fb) + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
      }
    }
    if (wt == 0) *fc = 0;
  }
#else
"""
SLOT_OLD = "  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)\n"
rep("main", SLOT_OLD, RFIN_SLOT + SLOT_OLD, tag="RFIN slot block")
rep("main", "    return;\n  }\n  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n",
    "    return;\n  }\n#endif\n  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n",
    tag="RFIN slot endif")

# MOE_CUDA_SRC: wgspack (ly_micro2), the xB SwiGLU pair as one bf16x2 store
rep("main", """#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = fork_bf16(d4[h] * gsc), up = fork_bf16(d4[2 + h] * gsc);
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
""", """    float sv[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float gt = fork_bf16(d4[h] * gsc), up = fork_bf16(d4[2 + h] * gsc);
      sv[h] = probe_q(SILU_DIV(gt, 1.f + exp_neg_ftz(gt)) * up);
    }
    {  // rows g and g^1 hold adjacent xB columns: swap one token so each lane stores one bf16x2
      const int od = g & 1;
      const float give = __shfl_xor_sync(0xffffffffu, od ? sv[0] : sv[1], 4);
      const int tk = nt * 8 + 2 * t + od;
      const __nv_bfloat162 pr = od ? __floats2bfloat162_rn(give, sv[1]) : __floats2bfloat162_rn(sv[0], give);
      if (tk < cnt)
        st_keep_bf16x2(xb - od + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8), pr);
    }
""", tag="wgspack")

# helper0: wspack (ly_micro2), the phase A SwiGLU pair as one bf16x2 store
rep("helper", """#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
""", """      float sv[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float gt = fork_bf16((h ? sm.y : sm.x) * gsc), up = fork_bf16((h ? sm.w : sm.z) * gsc);
        sv[h] = probe_q(SILU_DIV(gt, 1.f + exp_neg_ftz(gt)) * up);
      }
      {  // rows gg and gg^1 are adjacent inter columns: swap one token so each lane stores one bf16x2
        const int od = gg & 1;
        const float give = __shfl_xor_sync(0xffffffffu, od ? sv[0] : sv[1], 4);
        const int tk = 2 * tt + od;
        const __nv_bfloat162 pr = od ? __floats2bfloat162_rn(give, sv[1]) : __floats2bfloat162_rn(sv[0], give);
        if (tk < cnt) st_keep_bf16x2(p.inter + ((size_t)(st + tk) * ID + rb * 8 + (gg & ~1)), pr);
      }
""", tag="wspack")

# helper0: scl (ly_micro2), one PRMT builds both rows' scales; TINY opts out (decode 1 lost with it)
SCL_ANCHOR = ("    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n#pragma unroll\n"
              "    for (int c = 0; c < NCH; ++c) {\n"
              "      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n"
              "#if OKC_PRMTSC\n"
              "      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n")
SG_OLD = "      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n"
rep_near("helper", SCL_ANCHOR, SG_OLD,
         "#if !OKC_NOSCL\n"
         "      const uint32_t sbp = (uint32_t)(c & 1) * 2u;\n"
         "      const uint32_t sgp = __byte_perm(swd, 0u, 0x4040u | sbp | ((sbp + 1u) << 8)) * k16 + 0x5C005C00u;\n"
         "#else\n" + SG_OLD + "#endif\n", window=len(SCL_ANCHOR) + 4, tag="scl scales")
MMA_OLD = ("        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), "
           "bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n")
rep_near("helper", SCL_ANCHOR, MMA_OLD,
         "#if OKC_PRMTSC && !OKC_NOSCL\n"
         "        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2l(r0, sgp), bmul2h(r1, sgp), "
         "bmul2l(r2, sgp), bmul2h(r3, sgp), b[2 * st], b[2 * st + 1]);\n"
         "#else\n" + MMA_OLD + "#endif\n", window=1400, tag="scl mma")

# ftz at the remaining SwiGLU sites
rep("main", "__expf(-gt)", "exp_neg_ftz(gt)", count=4, tag="ftz sites")
rep("helper", "__expf(-gt)", "exp_neg_ftz(gt)", count=4, tag="ftz sites")

# Python selection: TINY keeps the old scale path; MAIN builds RFIN (MFIN follows from the gate)
rep("head", "'@moe_helper0', '-DOKC_FINPF=1')\n", "'@moe_helper0', '-DOKC_FINPF=1', '-DOKC_NOSCL=1')\n", tag="TINY NOSCL")
rep("py", "\ndef _main_defines(sfx):\n",
    '\nMAIN_EXTRA_DEFINES = ("-DOKC_RFIN=1", "-DOKC_RFINQ=2")\n\n\ndef _main_defines(sfx):\n', tag="MAIN_EXTRA")
rep("py", '        key, defines = "main", _main_defines(sfx)\n',
    '        key, defines = "main", tuple(_main_defines(sfx)) + MAIN_EXTRA_DEFINES\n', tag="main key")

out = regions["head"] + regions["main"] + text[m_b:h_a] + regions["helper"] + regions["py"]
open(out_path, "w").write(out)
py_compile.compile(out_path, doraise=True)
for line in log:
    print(line)
ob = out.encode()
print("out", out_path, len(ob), "bytes sha256", hashlib.sha256(ob).hexdigest())
assert len(ob) < 1000000
