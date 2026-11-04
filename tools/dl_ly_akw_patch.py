#!/usr/bin/env python3
"""ly_akw: ly_aqw + the same cold-code warming for the new k,v pass of the q_len 8 one-tile attention.

Only the split that holds the block's new rows runs the new k,v pass (RMS norm, RoPE, e4m3 for 8 tokens x 2 heads of
k and v, the cache-row writes), so its code is cold in that CTA, and that split's ticket comes last in every input
(round 2 stamps on ly_stk1: new k,v pass +0.32 us on the critical CTA against +0.03 elsewhere). One change on top of
ly_aqw, behind the same constexpr QWARM: the pass moves, unchanged, into the out-of-line function newkv8_ni, called
dry before griddepcontrol.wait by the CTAs with new rows (has_new; shared memory only: S.kvst / S.cs / S.sn / S.cloc
read, S.kn / S.vn written, the cache-row stores off) and for real after the q pass as in the base. The real call
rewrites every shared byte the dry call wrote (same thread, same address).

    python3 tools/dl_ly_akw_patch.py [--src muse_decode_layer/cand/ly_aqw.py] [--out muse_decode_layer/cand/ly_akw.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_aqw.py"
DST = ROOT / "muse_decode_layer/cand/ly_akw.py"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

FN = r'''// (OKC_QWARM8) the new k,v pass out of line for the same reason; dry: the cache-row stores are off
template <int NT>
__device__ __noinline__ void newkv8_ni(const __nv_bfloat16* kvst, const float* cs, const float* sn, uint8_t* kn,
                                       uint8_t* vn, const long long* cloc, uint8_t* kc, uint8_t* vc, const int off32,
                                       const int lrb, const int64_t row_bytes, const int tidL, const int partL,
                                       const int pair, const int seq, const int start, const int end, const int rope,
                                       const float kscale, const float vscale, const bool dry) {
  constexpr int L = 8, RPP = NT / 8;
  constexpr int KPASS = (L * 4 + RPP - 1) / RPP;
#pragma unroll
  for (int ps = 0; ps < KPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const bool live = rr < L * 4;
    const int kind = rr / (2 * L);
    const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
    float x[8];
    const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
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
    float sc;
    if (kind == 0) {
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] *= r;
      if (rope && live) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = partL * 4 + e;
          const float c = cs[jj * 32 + i], s = sn[jj * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * s;
          x[2 * e + 1] = a0 * s + a1 * c;
        }
      }
      sc = __fdividef(1.f, kscale);
    } else {
      sc = __fdividef(1.f, vscale);
    }
    if (live) {
      uint32_t w0 = 0, w1 = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
        w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
      }
      uint8_t* dst = (kind == 0 ? kn : vn) + jj * ROWB + hh * 64 + partL * 8;
      *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
      const int pos = seq + jj;
      if (!dry && pos >= start && pos < end) {
        const long long row = cloc[jj];
        uint8_t* base = kind == 0 ? kc : vc;
        const int col = (2 * pair + hh) * 64 + partL * 8;
        uint8_t* g8 = off32 ? base + (((uint32_t)row << lrb) + (uint32_t)col) : base + row * row_bytes + col;
        *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
      }
    }
  }
}

'''

SEG = "template <int L, bool FLAT, int ROLE, int MODE = 0>\n__device__ __forceinline__ bool segment("
CALL = ("newkv8_ni<NT>(S.kvst, S.cs, S.sn, S.kn, S.vn, S.cloc, const_cast<uint8_t*>(p.k_cache), "
        "const_cast<uint8_t*>(p.v_cache), p.off32, p.lrb, p.row_bytes, tidL, partL, pair, seq, start, end, p.rope, "
        "{ks}, {vs}, {dry})")
ATT_EDITS = [
    (SEG, FN + SEG),
    ("  if constexpr (QWARM) qpass8_ni<NT>(S.qst(), S.cs, S.sn, S.q8s, tidL, p.rope);\n",
     "  if constexpr (QWARM) qpass8_ni<NT>(S.qst(), S.cs, S.sn, S.q8s, tidL, p.rope);\n"
     "  if constexpr (QWARM) if (end > seq) " + CALL.format(ks="1.f", vs="1.f", dry="true") + ";\n"),
    ("  if (has_new) {\n#pragma unroll\n    for (int ps = 0; ps < KPASS; ++ps) {\n      const int rr = ps * RPP + (tidL >> 3);\n"
     "      const bool live = rr < L * 4;\n",
     "  if constexpr (QWARM) {\n    if (has_new) " + CALL.format(ks="kscale", vs="vscale", dry="false") + ";\n"
     "  } else if (has_new) {\n#pragma unroll\n    for (int ps = 0; ps < KPASS; ++ps) {\n      const int rr = ps * RPP + (tidL >> 3);\n"
     "      const bool live = rr < L * 4;\n"),
]
HEADER = ("# ly_akw: ly_aqw + the new k,v pass of the q_len 8 one-tile attention out of line and called dry before "
          "griddepcontrol.wait in the CTAs with new rows (tools/dl_ly_akw_patch.py).\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
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
