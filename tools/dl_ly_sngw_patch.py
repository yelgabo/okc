#!/usr/bin/env python3
"""ly_sngw: the layer attention's verify 1 sliding merge with one workspace copy per split, written by plain stores,
summed in split order by the cooperative finalize.

Base ly_v1un (pinned). One change. Host: _plan7's one-request q_len 8 sliding plan takes nc = n_splits (each split
owns its copy) instead of nc = 1 (all 33 splits red.add into one copy). Kernel, behind constexpr SNGW = OKC_SNGW8 &&
L == 8 && ONE && !FLAT && cooperative mode, and only when p.nc == p.n_splits at run time:
- merge: the same rescaled partials and addresses, st.global.cg instead of red.global.add (one writer per address);
- the grid barrier, its acq_rel arrive after the CTA barrier and every fence are unchanged;
- finalize: each CTA sums its rows over the nact copies in split order (one load per copy in flight together),
  zeroes what it read (the workspace stays zero between calls, as every other plan assumes), then the same
  rmsnorm * sigmoid(gate) and BF16 store, one dimension per thread.
Every other instance compiles to the base text (sngw is a constant false there).

    python3 tools/dl_ly_sngw_patch.py [--src F --out G]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_sngw.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

FIN = r"""// sngw (one workspace copy per split): rows [r_lo, r_hi) summed over the nact copies in split order (deterministic,
// unlike the red.add merge), the copies zeroed after the read, then rmsnorm * sigmoid(gate); one dimension per thread
#define SNGW_MAX 40
template <int L, int NT>
__device__ __forceinline__ void fin_sngw(const Params& p, const int b, const int pair, const int bp, const int r_lo,
                                         const int r_hi, const int nact, const float vscale) {
  __shared__ float ssx[NT / 32];
  const int tid = threadIdx.x, d = tid & 63, warp = tid >> 5, lane = tid & 31;
  const int nc = p.nc;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 64) {   // the same trip count in every thread (barriers below)
    const int rr = rbase + (tid >> 6);
    const bool act = rr < r_hi;
    const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
    const float gt = act ? __bfloat162float(__ldcg(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 +
                                                   (16 * pair + hh) * 64 + d)) : 0.f;
    float y = 0.f, ls = 0.f;
    for (int c0 = 0; c0 < nact; c0 += SNGW_MAX) {
      float v[SNGW_MAX], lv[SNGW_MAX];
#pragma unroll
      for (int u = 0; u < SNGW_MAX; ++u) {
        const bool on = act && c0 + u < nact;
        const int row = ((bp * nc + c0 + u) * L + j) * 16 + hh;
        v[u] = on ? __ldcg(p.ws_acc + (ix_t)row * 64 + d) : 0.f;
        lv[u] = on ? __ldcg(p.ws_l + row) : 0.f;
      }
#pragma unroll
      for (int u = 0; u < SNGW_MAX; ++u) { y += v[u]; ls += lv[u]; }
#pragma unroll
      for (int u = 0; u < SNGW_MAX; ++u) {
        if (act && c0 + u < nact) {
          const int row = ((bp * nc + c0 + u) * L + j) * 16 + hh;
          p.ws_acc[(ix_t)row * 64 + d] = 0.f;
          if (d == 0) p.ws_l[row] = 0.f;
        }
      }
    }
    const float inv = ls > 0.f ? __fdividef(vscale, ls) : 0.f;
    const float yv = pr8(y * inv);
    float ss = yv * yv;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) ssx[warp] = ss;
    __syncthreads();
    const float tot = ssx[warp & ~1] + ssx[warp | 1];
    __syncthreads();
    const float rn = rsqrtf(tot * (1.f / 64.f) + 1e-5f);
    if (act) p.out[(ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + d] = __float2bfloat16_rn(yv * rn * sigm(gt));
  }
}
"""

ATT_EDITS = [
    ("#define OKC_PDLKV 0", "#define OKC_SNGW8 1     // q_len 8 one-request sliding: one workspace copy per split (see fin_sngw)\n"
     "#define OKC_PDLKV 0"),
    ("template <int L>\n__device__ __forceinline__ void sum_copies(", FIN + "template <int L>\n__device__ __forceinline__ void sum_copies("),
    ("  bool do_red = true;\n",
     "  bool do_red = true;\n"
     "  constexpr bool SNGW = OKC_SNGW8 && L == 8 && ONE && !FLAT && (MODE & 3) == 3;\n"
     "  const bool sngw = SNGW && p.nc == p.n_splits;   // this split owns copy `copy`: plain stores, no red\n"),
    ("        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);\n",
     "        float* d4 = acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1);\n"
     "        if (sngw) __stcg(reinterpret_cast<float4*>(d4), make_float4(odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1));\n"
     "        else red_v4(d4, odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);\n"),
    ("    if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);\n",
     "    if (tig == 0) {\n"
     "      float* dl = p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh;\n"
     "      if (sngw) __stcg(dl, lsum * f);\n"
     "      else red_f32(dl, lsum * f);\n"
     "    }\n"),
    ("    DBG(4);\n    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);\n    const int part = tid & 7;\n",
     "    DBG(4);\n    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);\n"
     "    if (sngw) {\n      if constexpr (SNGW) fin_sngw<L, NT>(p, b, pair, bp, r_lo, r_hi, nact, vscale);\n    } else {\n"
     "    const int part = tid & 7;\n"),
    ("    if (!OKC_SENSE && !OKC_SENSE2 && !OKC_PARITY) {\n", "    }\n    if (!OKC_SENSE && !OKC_SENSE2 && !OKC_PARITY) {\n"),
]
FILE_EDITS = [
    ("        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)   # (one copy since v101: b1 9.8 -> 9.6 us)\n",
     "        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=n if reqs == 1 else 1, coop=1)   # (one request: a copy per split, fin_sngw)\n"),
]


def sub(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    start = 'ATT7_CUDA_SRC = r"""\n'
    i = text.index(start) + len(start)
    j = text.index('\n"""\n', i)
    a = text[i:j]
    for old, new in ATT_EDITS:
        a = sub(a, old, new, "att7")
    text = text[:i] + a + text[j:]
    for old, new in FILE_EDITS:
        text = sub(text, old, new, "file")
    text = ("# ly_sngw: ly_v1un + verify 1 sliding attention merge with one workspace copy per split, plain stores, "
            "ordered sum (tools/dl_ly_sngw_patch.py).\n") + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
