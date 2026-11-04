#!/usr/bin/env python3
"""ly_anf4: decode 1 sliding attention with 4 FP32 workspace copies and a one-round-trip last-CTA finalize for them.

Base ly_v1un (88fbae86). Two edits that make one mechanism (fewer reds per workspace address without a slower
finalize):
- host (_plan7): the one-request q_len 1 window plan takes nc = 4 (B == 1 only); split s adds into copy s % 4 (the
  kernel's existing copy index), so 8 CTAs share each address instead of 32. Stamps (ly_ast --mrg): the ticket
  returns 1.49 us after the reds with one copy, 1.14 us with four (ly_anc4), but the generic nc > 1 finalize took
  1.07 us against fin_nc1's 0.56, so ly_anc4 lost 0.2 us at the o kernel's wait.
- kernel: the last CTA of a q_len 1 grid with nc == 4 calls fin_ncn<L, NT, 4>, fin_nc1 with every copy's loads (and
  the gate row's) issued before any use, the copies summed in copy order (as sum_copies does), then the same
  normalize / rmsnorm / gate / store and workspace zeroing. Every other plan keeps its finalize.
Ordering is unchanged: the same reds, the same acq_rel ticket after the CTA barrier, the same __ldcg reads.

    python3 tools/dl_ly_anf_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_anf4.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_anf4.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

FIN_NCN = r"""// ly_anf4: fin_nc1 for NCP workspace copies: every copy's loads and the gate row's in flight together, summed in copy
// order (sum_copies' order), then fin_nc1's arithmetic
template <int L, int NT, int NCP>
__device__ __forceinline__ void fin_ncn(const Params& p, const int b, const int pair, const int bp, const float vscale) {
  const int tid = threadIdx.x;
  constexpr int PASSES = L * 16 / (NT / 8);
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; ++p0) {
    const int rr = p0 * (NT / 8) + (tid >> 3);
    const int j = rr >> 4, hh = rr & 15;
    float4 xa[NCP], xb[NCP];
    float lc[NCP];
    const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8;
    const uint4 graw = __ldcg(reinterpret_cast<const uint4*>(gp));
#pragma unroll
    for (int c = 0; c < NCP; ++c) {
      const float* acc = p.ws_acc + ((ix_t)((bp * NCP + c) * L + j) * 16 + hh) * 64 + part * 8;
      xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[c] = __ldcg(p.ws_l + ((bp * NCP + c) * L + j) * 16 + hh);
    }
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
#pragma unroll
    for (int c = 0; c < NCP; ++c) {
      y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
      y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
      lsum += lc[c];
      float* acc = p.ws_acc + ((ix_t)((bp * NCP + c) * L + j) * 16 + hh) * 64 + part * 8;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) p.ws_l[((bp * NCP + c) * L + j) * 16 + hh] = 0.f;
    }
    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
    ss = oct_sum(ss);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
}

// OKC_PATCHNI: the q_len 8 flat loop's rare new-row patch as a call"""

ATT_EDITS = [
    ("// OKC_PATCHNI: the q_len 8 flat loop's rare new-row patch as a call", FIN_NCN, 1),
    ("  if (p.nc == 1) {\n    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);\n  } else\n",
     "  if (p.nc == 1) {\n    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);\n  } else if (L == 1 && p.nc == 4) {\n"
     "    fin_ncn<L, NT, 4>(p, b, pair, bp, vscale);\n  } else\n", 1),
]
PLAN_OLD = ("    if L == 1 and reqs < 24:\n        n = max(1, min(tiles, target // reqs))\n"
            "        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)\n")
PLAN_NEW = ("    if L == 1 and reqs < 24:\n        n = max(1, min(tiles, target // reqs))\n"
            "        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=4 if B == 1 else 1)   # (ly_anf4: fewer reds per address)\n")
HEADER = ("# ly_anf4: ly_v1un with 4 attention workspace copies at decode 1 sliding and a one-round-trip finalize for them "
          "(tools/dl_ly_anf_patch.py).\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    start = 'ATT7_CUDA_SRC = r"""\n'
    i = text.index(start) + len(start)
    j = text.index('\n"""\n', i)
    at = text[i:j]
    for old, new, n in ATT_EDITS:
        at = sub(at, old, new, n, "att")
    text = text[:i] + at + text[j:]
    text = HEADER + sub(text, PLAN_OLD, PLAN_NEW, 1, "plan")
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
