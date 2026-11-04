#!/usr/bin/env python3
"""ly_rfin: a cut unit of the MAIN WGS build is finished in registers by the share holding stage 0 (verify 16 / 32).

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 3765470032..., checked). Target: docs/runs-decode-layer.md,
"2026-11-04 - Verify segment switches (box)". Per CTA and phase about 1.7 of the 5 to 7 segments are shares of a unit
cut between two CTAs (stream-K). In phase A (OKC_WGSDFIN bit 0 off) both shares red.add their partial sums into the
unit's fixup slot and, deferred into their next segment, each takes an atom.acq_rel.gpu round trip on the slot's
count; the last arriver reads the slot back, zeroes it and runs the epilogue. Measured on the base's stamps: the
switch after a cut share costs 2.0 to 3.9 us against 0.9 to 1.2 us after a whole unit.

Change (OKC_RFIN=1, MAIN build only; every other build compiles the base's text):
- ranges run in order in both phases (OKC_WGSREORD off), so the share holding stage 0 of a cut unit is its CTA's
  last share of that unit and the other shares, on the CTAs after it, were run first and deposited long before;
- a share that does not hold stage 0 stores its partial sums (plain float4 stores, no atomics) into its own CTA's
  fixup slot (a range has at most one such share per phase) and releases its stage count to the finisher's counter,
  deferred as before (kind 2, bar.sync + red.release.gpu);
- the share holding stage 0 (the finisher, both phases: OKC_WGSDFIN 3) waits for the count at its segment end, while
  its accumulators are live: ld.acquire loop, bar.sync, fence.acq_rel.gpu (the base's kind-3 sequence), reads the
  other shares' slots with ld.global.cg (OKC_RFINQ tiles in flight), zeroes each slot it read (the B4 build of
  decode 32 shares the scratch and adds onto zeros: without this its cut units pick up the stale deposits, startup
  comparison 0.83 at M = 32), adds them into the accumulators and falls through to the whole-unit epilogue and its
  deferred release (kind 1 / kind 4: the same fence.proxy.async.global, bar.sync and red.release.gpu as a whole unit).
No fence, barrier or release is removed; the kind-3 deferred read-back is replaced by the same acquire chain run at
the segment end. For two shares the sum is a + b as before (the base formed 0 + a + b by atomics).

    python3 tools/dl_ly_rfin_patch.py [--base F --pin SHA] [--q N] [--out muse_decode_layer/cand/ly_rfin.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
DST = ROOT / "muse_decode_layer/cand/ly_rfin.py"
Q = 2
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
if "--q" in sys.argv:
    Q = int(sys.argv[sys.argv.index("--q") + 1])
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"
MAIN = 'key, defines = "main", _main_defines(sfx)'

DEFS_OLD = "#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)\n"
DEFS_NEW = """#ifndef OKC_RFIN
#define OKC_RFIN 0  // rfin: cut units finished in registers by the share holding stage 0 (in-order ranges, stored partials)
#endif
#ifndef OKC_RFINQ
#define OKC_RFINQ 2  // rfin: fixup tiles read in flight by the finisher
#endif
#define OKC_WGSREORD (!OKC_RFIN)  // WGS: phase A starts with the trailing partial unit (see wgs_split)
"""
DFIN_OLD = "#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n"
DFIN_NEW = "#define OKC_WGSDFIN (OKC_RFIN ? 3 : 2)  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n"

FLUSH_OLD = """  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
"""
FLUSH_NEW = """  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
#if OKC_RFIN
  pd.kind = 0;  // (kind 3 never pends: the finisher completes at its segment's end, wgs_seg)
  return;
#endif
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
"""

SEG_OLD = """  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
"""
SEG_NEW = """#if OKC_RFIN
  if (slot >= 0) {  // a share of a unit split between CTAs (slot: the CTA holding stage 0, which finishes the unit)
    if (s0 != 0) {
      // not the finisher: the partial sums go to this CTA's own fixup slot (a range holds at most one such share per
      // phase), the stage count to the finisher's counter, deferred with a release as before
      float* fb = wgs_fix(p, PB) + (size_t)(blockIdx.x * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt)
        *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
      pd = WgsPend{2, u, slot, s1 - s0};
      return;
    }
    // the finisher: in range order this is the CTA's last share of the unit and the other shares, on the CTAs after
    // it, were run first; their counts are acquired and their partials added into the live accumulators, then the
    // whole-unit epilogue below publishes the unit
    int* fc = wgs_cnt(p, PB) + slot * 4 + wg;
    if (wt == 0) { while (ld_acquire(fc) < 16 - (s1 - s0)) __nanosleep(32); }
    asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
    asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
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
            // the slot is zero again for the next launch: the B4 build shares the scratch and adds onto zeros
            *reinterpret_cast<float4*>(const_cast<float*>(fb) + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
      }
    }
    if (wt == 0) *fc = 0;
  }
#else
""" + SEG_OLD + "#endif\n"


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if not sha.startswith(PIN[:16]):
        sys.exit(f"base {SRC.name} is {sha[:16]}, expected {PIN[:16]}")
    text = raw.decode("utf-8")
    a, b = text.index(START) + len(START), text.index(END)
    cu, head, tail = text[a:b], text[:a], text[b:]
    for old, new in ((DEFS_OLD, DEFS_NEW), (DFIN_OLD, DFIN_NEW), (FLUSH_OLD, FLUSH_NEW), (SEG_OLD, SEG_NEW)):
        if cu.count(old) != 1:
            sys.exit(f"anchor count {cu.count(old)}: {old[:60]!r}")
        cu = cu.replace(old, new)
    for name in ("OKC_WGSREORD", "OKC_WGSDFIN"):
        if f"#if {name}" in cu or f"#if !{name}" in cu:
            sys.exit(f"{name} used in a preprocessor condition")
    if tail.count(MAIN) != 1:
        sys.exit("MAIN define anchor")
    extra = f'MAIN_EXTRA_DEFINES = ("-DOKC_RFIN=1", "-DOKC_RFINQ={Q}")  # rfin: the MAIN build only (tools/ly_micro_sass.py reads it)\n\n\n'
    tail = tail.replace("\n\ndef _fork_native_moe(", "\n\n" + extra + "def _fork_native_moe(", 1)
    if tail.count("MAIN_EXTRA_DEFINES = (") != 1:
        sys.exit("_fork_native_moe anchor")
    tail = tail.replace(MAIN, 'key, defines = "main", tuple(_main_defines(sfx)) + MAIN_EXTRA_DEFINES')
    out = head + cu + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
