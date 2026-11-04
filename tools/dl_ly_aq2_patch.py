#!/usr/bin/env python3
"""ly_aq2: ly_aqd + the one-tile q_len 8 QDIRECT q pass with two threads per query row (32 dims each, one pass per
thread) instead of four (16 dims each, two passes per thread).

Round 2 stamps on ly_aqd: the q pass (two QP16 passes per thread, about 280 SASS instructions) takes 0.85 us of
thread 0's time; the two passes run one after the other in the SASS and loading both passes' operands first (ly_aqi)
does not change the time, so each pass is a latency-bound dependent chain of about 0.4 us. With 128 query rows and 256
threads, two threads per row make it one pass whose two 16-dim halves are independent chains. One change, a new branch
ahead of the QP16 one behind OKC_QP32 (compiled for QDIR builds with L * 32 == NT, which on ly_aqd are only the
one-tile q_len 8 builds): the same per-element arithmetic, the sum of squares formed as (s0 + s1) + (s2 + s3) over the
same 16-dim sequential partial sums as QP16 (so the RMS factor is bitwise the same), and two 16-byte e4m3 stores per
thread at the QP16 addresses. Bitwise identical output.

    python3 tools/dl_ly_aq2_patch.py [--src muse_decode_layer/cand/ly_aqd.py] [--out muse_decode_layer/cand/ly_aq2.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_aqd.py"
DST = ROOT / "muse_decode_layer/cand/ly_aq2.py"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

QP32 = r'''  if constexpr (QDIR && OKC_QP32 && L * 32 == NT) {
    // OKC_QP32: two threads per query row, 32 dims each, one pass; the RMS sum keeps QP16's association
    // ((s0 + s1) + (s2 + s3) over sequential 16-dim partial sums), so the output is bitwise QP16's
    const int rr = tidL >> 1, q2 = tidL & 1;
    const int j = rr >> 4, hh = rr & 15;
    float x[32];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q2 * 32 + k * 8]);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[8 * k + 2 * e] = f.x;
        x[8 * k + 2 * e + 1] = f.y;
      }
    }
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) s0 += x[e] * x[e];
#pragma unroll
    for (int e = 0; e < 16; ++e) s1 += x[16 + e] * x[16 + e];
    float ss = s0 + s1;
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 32; ++e) x[e] *= r;
    if (p.rope) {
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        const int i = q2 * 16 + e;
        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * sn;
        x[2 * e + 1] = a0 * sn + a1 * c;
      }
    }
    const int hk4 = hh >> 3, gg4 = hh & 7;
    const int r4 = j * 8 + gg4;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int kb = q2 * 32 + h * 16;
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[16 * h + 0], x[16 * h + 1], x[16 * h + 2], x[16 * h + 3]),
                     e4m3x4f(x[16 * h + 4], x[16 * h + 5], x[16 * h + 6], x[16 * h + 7]),
                     e4m3x4f(x[16 * h + 8], x[16 * h + 9], x[16 * h + 10], x[16 * h + 11]),
                     e4m3x4f(x[16 * h + 12], x[16 * h + 13], x[16 * h + 14], x[16 * h + 15]));
    }
  } else if constexpr (QDIR && OKC_QP16) {
'''
EDITS = [
    ("#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n",
     "#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n"
     "#define OKC_QP32 1      // q_len 8 QDIRECT q pass with NT == 32 * L: two threads per row (32 dims each), one pass\n"),
    ("  if constexpr (QDIR && OKC_QP16) {\n", QP32),
]


def main():
    text = SRC.read_text()
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    for old, new in EDITS:
        if blk.count(old) != 1:
            sys.exit(f"anchor found {blk.count(old)} times: {old[:80]!r}")
        blk = blk.replace(old, new)
    text = (f"# ly_aq2: {SRC.stem} + the one-tile q_len 8 QDIRECT q pass with two threads per query row, one pass "
            "(tools/dl_ly_aq2_patch.py).\n") + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
