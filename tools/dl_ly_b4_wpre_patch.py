#!/usr/bin/env python3
"""ly_b4_wpre: the WGS weights producer issues its first two stages once the stage ranges exist, before WS_JOIN.

Base: muse_decode_layer/cand/ly_micro2.py (SHA256 0bc2bd8b..., checked). Target: the B4 ramp at decode 32 (entry to
first weight issue 4.3 us of a 78.6 us span; docs/runs-decode-layer.md "Decode 32: B4 MoE timeline"). The weights
producer waits at WS_JOIN, which also waits for every consumer's phase A x gather (done at 3.4 us), though the weight
stream needs only the routing tables and this CTA's stage ranges. The x producer computes the ranges right after
routing (bar 7), so it now arrives on a new named barrier 8 (64 threads: the two producer warps) after wgs_ranges, and
the weights producer, after bar.sync 8, issues the first min(2, phase A stages) stages of its sequence, then joins
WS_JOIN as before and continues from stage 2. Ordering: the consumers' routing tables and mbarrier inits reach the x
producer through bar 7 (as today) and the weights producer through bar 8; the weights ring is written by nothing else
before the producers start. Every thread still passes WS_JOIN; no fence, barrier, wait or release is removed, the
ring protocol and the M <= 64 pacing (stage n issued once n - 2 landed) are unchanged, and PDL is untouched. B4 and MAIN
(the same source) both change.

    python3 tools/dl_ly_b4_wpre_patch.py [--out muse_decode_layer/cand/ly_b4_wpre.py]
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_micro2.py"
PIN = "0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd"
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"

OLD_W = '''__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
'''
NEW_W = '''__device__ __forceinline__ void wgs_wissue(const Params& p, Smem& S, unsigned char* ring, int pb, int gs, int ws) {
  const int ubl = pb ? 4 : 5;
  const int u = gs >> 4, st = gs & 15;
  const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
  uint64_t* bar = S.bars + BW_F + ws;
  const unsigned char* src = (pb ? p.wBs : p.wAs) + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
}
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  // wpre: routing tables and this CTA's stage ranges are ready (the x producer arrives after wgs_ranges); the first
  // stages need nothing else, so they go out before WS_JOIN, which also waits for the consumers' x gather
  asm volatile("bar.sync 8, 64;\\n" ::: "memory");
  const int npre = min(2, S.wgr[1] - S.wgr[0]);
  if ((threadIdx.x & 31) == 0) {
    const int g0 = S.wgr[0], g1 = S.wgr[1], sp = wgs_split(0, g0, g1);
    for (int i = 0; i < npre; ++i) wgs_wissue(p, S, ring, 0, sp + i < g1 ? sp + i : g0 + (sp + i - g1), i);
  }
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  int ws = npre, n = npre;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
    const int i0 = pb ? 0 : npre;
#pragma unroll 1
    for (int i = i0, gs = sp + i0 < g1 ? sp + i0 : g0 + (sp + i0 - g1); i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
'''
OLD_X = '''  wgs_ranges(p, S);  // all 32 lanes (warp-parallel)
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
'''
NEW_X = '''  wgs_ranges(p, S);  // all 32 lanes (warp-parallel)
  __syncwarp();
  asm volatile("bar.arrive 8, 64;\\n" ::: "memory");  // wpre: the weights producer may start
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="muse_decode_layer/cand/ly_b4_wpre.py")
    a = ap.parse_args()
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PIN:
        sys.exit(f"ly_micro2.py is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    head, rest = raw.decode().split(START)
    cu, tail = rest.split(END)
    for old, new in ((OLD_W, NEW_W), (OLD_X, NEW_X)):
        if cu.count(old) != 1:
            sys.exit(f"anchor: {old[:80]!r}")
        cu = cu.replace(old, new)
    if "#define OKC_WGSPR 1" not in cu or "bar.sync 8" in cu.replace(NEW_W, ""):
        sys.exit("wpre needs OKC_WGSPR 1 and a free named barrier 8")
    out = head + START + cu + END + tail
    dst = ROOT / a.out
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
