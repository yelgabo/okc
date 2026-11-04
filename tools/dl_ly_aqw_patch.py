#!/usr/bin/env python3
"""ly_aqw: cold-code warming of the q_len 8 one-tile attention's q pass, on ly_stk1.

The verify 1 sliding attention (attn_kernel<8, false, 15>) spends about 1.0 us of thread 0's time in a q pass of four
iterations of register math (round 1 stamps: rope 1.25 -> qpass 2.21 us after qkv's end), while the CTA idles about
1.1 us in griddepcontrol.wait before it. Hypothesis: the pass runs from a cold instruction cache. One change in
ATT7_CUDA_SRC, behind constexpr QWARM (only the one-tile q_len 8 per-request builds, MODE 13 / 15, as the layer builds
them: -DOKC_QDIRECT=0, so the f16 staging pass; every other instance compiles the base text): the q pass moves,
unchanged, into the out-of-line function qpass8_ni, and each CTA calls it twice: once before griddepcontrol.wait on
whatever its shared memory holds (reads only S.kvq / S.cs / S.sn, writes only the last pipeline stage's q staging, no
global access, no copy in flight to that stage), and once after the wait at the base's place on the real rows. The
real call overwrites every byte the dry call wrote (same thread, same address); the wait, the trigger, every fence and
the merge are unchanged. (The first build of this tool gated on QDIR, which the layer compiles out, so it built the
base text.)

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
__device__ __noinline__ void qpass8_ni(const __nv_bfloat16* qst, const float* cs, const float* sn, uint8_t* qk,
                                       uint8_t* qv8, const int tidL, const int partL, const int rope, const float qsc) {
  constexpr int RPP = NT / 8;
  constexpr int QPASS = 8 * 16 / RPP;
#pragma unroll
  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&qst[rr * 64 + partL * 8]);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    float x[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(h2[e]);
      x[2 * e] = f.x;
      x[2 * e + 1] = f.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    ss = oct_sum(ss);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] *= r;
    if (rope) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = partL * 4 + e;
        const float c = cs[j * 32 + i], s = sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * s;
        x[2 * e + 1] = a0 * s + a1 * c;
      }
    }
    uint4 qw;
    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    const int hk = hh >> 3, gg = hh & 7;
    *reinterpret_cast<uint4*>((hk ? qv8 : qk) + (j * 8 + gg) * 128 + partL * 16) = qw;
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
     "  constexpr bool QWARM = OKC_QWARM8 && ONE && L == 8 && !QDIR && !WS && !Cfg<L, FLAT>::S2 && OKC_SM90 && OKC_Q8RAW;\n"),
    ("\n\n  pdl_wait();   // qkv (q, new k / v, gate)",
     "\n  // (QWARM: shared memory only -- garbage in, the last stage's q staging out, rewritten by the real call below)\n"
     "  if constexpr (QWARM)\n"
     "    qpass8_ni<NT>(S.qst(), S.cs, S.sn, S.k + (STAGES - 1) * STAGE_BYTES, S.v + (STAGES - 1) * STAGE_BYTES, tidL, partL,\n"
     "                  p.rope, 1.f);\n"
     "\n  pdl_wait();   // qkv (q, new k / v, gate)"),
    ("  } else\n#pragma unroll\n  for (int ps = 0; ps < QPASS; ++ps) {\n",
     "  } else if constexpr (QWARM) {\n"
     "    qpass8_ni<NT>(S.qst(), S.cs, S.sn, S.k + (STAGES - 1) * STAGE_BYTES, S.v + (STAGES - 1) * STAGE_BYTES, tidL, partL,\n"
     "                  p.rope, 1.f);\n"
     "  } else\n#pragma unroll\n  for (int ps = 0; ps < QPASS; ++ps) {\n"),
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
