#!/usr/bin/env python3
"""ly_a16nc2: the decode 16 / 32 sliding attention (attn_kernel<1, false, 1>) reds into two workspace copies per
request (even splits copy 0, odd splits copy 1: 8 / 4 CTAs per address instead of 16 / 8), and the last CTA's
finalize (fin_nc1, warmed by its dry call as in the base) loads both copies in its one L2 round trip, adds them in
copy order and zeroes both.

Base ly_stk3 (a1c48810). Stamps (round 2, tools/ly_box_a16.py): reds issued -> ticket return 1.46 us at decode 16
(16 CTAs per request on the same 4 KB), 0.93 at decode 32 (8 per request), 0.80 at decode 16 with 8 splits
(ly_a16sp): the same-address fan-in, not the red volume, tracks the ticket. Standalone nc 2 (e468255b, ruled out)
used the generic nc > 1 finalize, which is cold and rolled (ly_anc4 at decode 1: 1.10 vs 0.56 us); this build keeps
the warmed one-round-trip finalize.

Device: fin_nc1 gains `bool NC2` (false everywhere except the MODE 1, L == 1 instance under OKC_A16NC2, so every
other instance compiles the base code); the ticket, its acq_rel atomic, the CTA barrier before it and every fence
are unchanged. Host: _TUNED7 entries set nc = 2 (n_splits unchanged) at decode 16 and, unless --d16only, decode 32.

    python3 tools/dl_ly_a16nc_patch.py [--d16only] [--base F] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
D16ONLY = "--d16only" in sys.argv
DST = ROOT / f"muse_decode_layer/cand/{'ly_a16nc2s' if D16ONLY else 'ly_a16nc2'}.py"
if "--base" in sys.argv:
    SRC, PIN = ROOT / sys.argv[sys.argv.index("--base") + 1], None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

FIN_OLD = '''template <int L, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
'''
FIN_NEW = '''template <int L, int NT, bool NC2 = false>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
  // NC2 (ly_a16nc2): two workspace copies (p.nc == 2), both loaded in this round trip and added in copy order
  const int bq = NC2 ? bp * p.nc : bp;
  const bool two = NC2 && p.nc == 2;
'''
LOAD_OLD = '''      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
'''
LOAD_NEW = '''      const float* acc = p.ws_acc + ((ix_t)(bq * L + j) * 16 + hh) * 64 + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(p.ws_l + (bq * L + j) * 16 + hh);
      if constexpr (NC2) {
        if (two) {
          const float* acc2 = p.ws_acc + ((ix_t)((bq + 1) * L + j) * 16 + hh) * 64 + part * 8;
          const float4 ya = __ldcg(reinterpret_cast<const float4*>(acc2));
          const float4 yb = __ldcg(reinterpret_cast<const float4*>(acc2) + 1);
          const float l2 = __ldcg(p.ws_l + ((bq + 1) * L + j) * 16 + hh);
          xa[u].x += ya.x; xa[u].y += ya.y; xa[u].z += ya.z; xa[u].w += ya.w;
          xb[u].x += yb.x; xb[u].y += yb.y; xb[u].z += yb.z; xb[u].w += yb.w;
          lc[u] += l2;
        }
      }
'''
ZERO_OLD = '''      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
      }
'''
ZERO_NEW = '''      float* acc = p.ws_acc + ((ix_t)(bq * L + j) * 16 + hh) * 64 + part * 8;
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[(bq * L + j) * 16 + hh] = 0.f;
        if constexpr (NC2) {
          if (two) {
            float* acc2 = p.ws_acc + ((ix_t)((bq + 1) * L + j) * 16 + hh) * 64 + part * 8;
            *reinterpret_cast<float4*>(acc2) = make_float4(0.f, 0.f, 0.f, 0.f);
            *(reinterpret_cast<float4*>(acc2) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
            if (part == 0) p.ws_l[((bq + 1) * L + j) * 16 + hh] = 0.f;
          }
        }
      }
'''
GATE = "(OKC_A16NC2 && L == 1 && !FLAT && MODE == 1)"
ATT_EDITS = [
    ("#define OKC_IDXFIRST 1 ",
     "#define OKC_A16NC2 1   // ly_a16nc2: decode 16 / 32 sliding (MODE 1, q_len 1) finalize sums two workspace copies\n"
     "#define OKC_IDXFIRST 1 "),
    (FIN_OLD, FIN_NEW),
    (LOAD_OLD, LOAD_NEW),
    (ZERO_OLD, ZERO_NEW),
    ("    if (p.cs <= 1 && !p.coop && p.nc == 1) fin_nc1<L, NT>(p, b, pair, bp, 0.f, 1);\n",
     f"    if (p.cs <= 1 && !p.coop && (p.nc == 1 || ({GATE} && p.nc == 2))) fin_nc1<L, NT, {GATE}>(p, b, pair, bp, 0.f, 1);\n"),
    ("  if (p.nc == 1) {\n    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);\n",
     f"  if (p.nc == 1 || ({GATE} && p.nc == 2)) {{\n    fin_nc1<L, NT, {GATE}>(p, b, pair, bp, vscale, 0);\n"),
]
PY_EDITS = [
    ("_TUNED7 = {\n",
     "_TUNED7 = {\n    (16, 1, True, 1): dict(n_splits=16, nc=2),   # ly_a16nc2: two workspace copies\n"
     + ("" if D16ONLY else "    (32, 1, True, 1): dict(n_splits=8, nc=2),    # ly_a16nc2\n")),
]


def sub(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    if PIN and hashlib.sha256(text.encode()).hexdigest() != PIN:
        sys.exit("base SHA256 mismatch")
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    at = text[i:j]
    for n, (old, new) in enumerate(ATT_EDITS):
        at = sub(at, old, new, f"att edit {n}")
    text = text[:i] + at + text[j:]
    for n, (old, new) in enumerate(PY_EDITS):
        text = sub(text, old, new, f"py edit {n}")
    text = (f"# {DST.stem}: ly_stk3 with two workspace copies per request in the decode 16"
            f"{'' if D16ONLY else ' / 32'} sliding attention merge (tools/dl_ly_a16nc_patch.py).\n") + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
