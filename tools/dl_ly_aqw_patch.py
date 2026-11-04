#!/usr/bin/env python3
"""ly_aqw: cold-code warming of the q_len 8 one-tile attention's q pass, on ly_stk1.

The verify 1 sliding attention (attn_kernel<8, false, 15>) spends about 0.96 us of thread 0's time in a q pass of two
iterations of register math (round 1 stamps: rope 1.25 -> qpass 2.21 us after qkv's end), while the CTA idles about
1.1 us in griddepcontrol.wait before it. Hypothesis: the pass runs from a cold instruction cache. One change in
ATT7_CUDA_SRC, behind constexpr QWARM = OKC_QWARM8 && QDIR && OKC_QP16 && ONE && L == 8 (only the one-tile q_len 8
per-request builds, MODE 13 / 15; every other instance compiles the base text): the OKC_QP16 q pass moves, unchanged,
into the out-of-line function qpass8_ni, and each CTA calls it twice: once before griddepcontrol.wait on whatever its
shared memory holds (reads only S.kvq / S.cs / S.sn, writes only S.q8s, no global access), and once after the wait at
the base's place on the real rows. The real call overwrites every byte the dry call wrote (same thread, same address);
fence.proxy.async after the pass, the wait, the trigger and the merge are unchanged.

    python3 tools/dl_ly_aqw_patch.py [--src muse_decode_layer/cand/ly_stk1.py] [--out muse_decode_layer/cand/ly_aqw.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_aqw.py"
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

FN = r'''// (OKC_QWARM8) the q_len 8 one-tile q pass out of line: a dry call before griddepcontrol.wait fetches the very
// instructions the real pass runs after it, which otherwise come from a cold instruction cache
template <int NT>
__device__ __noinline__ void qpass8_ni(const __nv_bfloat16* qst, const float* cs, const float* sn, uint8_t* q8s,
                                       const int tidL, const int rope) {
  constexpr int RP4 = NT / 4;
  constexpr int QP4 = 8 * 16 / RP4;
  const int q4 = tidL & 3;
#pragma unroll
  for (int ps = 0; ps < QP4; ++ps) {
    const int rr = ps * RP4 + (tidL >> 2);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qa = *reinterpret_cast<const uint4*>(&qst[rr * 64 + q4 * 16]);
    const uint4 qb = *reinterpret_cast<const uint4*>(&qst[rr * 64 + q4 * 16 + 8]);
    const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
    float x[16];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 fa = __bfloat1622float2(ha[e]);
      const float2 fb = __bfloat1622float2(hb[e]);
      x[2 * e] = fa.x; x[2 * e + 1] = fa.y;
      x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 16; ++e) x[e] *= r;
    if (rope) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = q4 * 8 + e;
        const float c = cs[j * 32 + i], s = sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * s;
        x[2 * e + 1] = a0 * s + a1 * c;
      }
    }
    const int hk4 = hh >> 3, gg4 = hh & 7;
    const int r4 = j * 8 + gg4, kb = q4 * 16;
    *reinterpret_cast<uint4*>(q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
        make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                   e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
  }
}

'''

SEG = "template <int L, bool FLAT, int ROLE, int MODE = 0>\n__device__ __forceinline__ bool segment("
ATT_EDITS = [
    ("#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n",
     "#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n"
     "#define OKC_QWARM8 1    // q_len 8 one-tile builds: the q pass out of line, called dry before griddepcontrol.wait\n"),
    (SEG, FN + SEG),
    ("  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n",
     "  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n"
     "  constexpr bool QWARM = OKC_QWARM8 && QDIR && OKC_QP16 && ONE && L == 8;\n"),
    ("\n\n  pdl_wait();   // qkv (q, new k / v, gate)",
     "\n  // (QWARM: shared memory only -- garbage in, S.q8s out, rewritten by the real call below)\n"
     "  if constexpr (QWARM) qpass8_ni<NT>(S.qst(), S.cs, S.sn, S.q8s, tidL, p.rope);\n"
     "\n  pdl_wait();   // qkv (q, new k / v, gate)"),
    ("  if constexpr (QDIR && OKC_QP16) {\n    // OKC_QP16",
     "  if constexpr (QWARM) {\n    qpass8_ni<NT>(S.qst(), S.cs, S.sn, S.q8s, tidL, p.rope);\n"
     "  } else if constexpr (QDIR && OKC_QP16) {\n    // OKC_QP16"),
]
HEADER = ("# ly_aqw: ly_stk1 + the q_len 8 one-tile attention's q pass out of line and called dry before "
          "griddepcontrol.wait (instruction cache warm-up; tools/dl_ly_aqw_patch.py).\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    for old, new in ATT_EDITS:
        blk = sub(blk, old, new, 1, "att7")
    text = HEADER + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
