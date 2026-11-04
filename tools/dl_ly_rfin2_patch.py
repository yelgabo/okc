#!/usr/bin/env python3
"""ly_rfin2: the cut unit's finisher share runs before its CTA's last whole unit (MAIN WGS build, verify 16 / 32).

Base: muse_decode_layer/cand/ly_rfin.py (SHA256 c42c1239..., checked), chained on ly_fabq. Target:
docs/runs-decode-layer.md, "2026-11-04 - Verify cut unit order (box)". In ly_rfin the ranges run in stage order, so
the share holding stage 0 of a cut unit (the finisher) is the CTA's last segment: its count acquire, the read-back of
the partner's partial (one L2 round trip per n-tile pair) and the unit's epilogue sit at the phase end, and in phase A
the unit's doneAs release (deferred into the next segment) becomes the phase's end flush, which the phase B x producer
waited 1.1 to 1.8 us longer for.

Change (OKC_RFIN2, a bit mask: 1 phase A, 2 phase B; MAIN build only, every other build compiles the base's text): a
range [g0, g1) whose last stage is inside a unit (g1 & 15 != 0) and that holds the whole unit before it, [uw, ue) with
ue = first stage of the unit holding g1 - 1 and uw = ue - 16 >= g0, is walked as [g0, uw), [ue, g1), [uw, ue) by the
weights producer, the x producer and the consumers alike (the ring is matched by issue count, so all three use the
same function). The first segment, a depositing share when g0 is inside a unit, stays first; a range without a whole
unit before its cut share keeps the stage order. Ordering constraints: the finisher still acquires the partner's count
before reading its slot (ld.acquire loop, bar.sync, fence.acq_rel.gpu, the base's sequence); the partner on the next
CTA deposits in its first segment and releases its count at commit group OKC_WGSFCI of its second segment, so the
finisher, which now reaches the cut share after (range - 16 - share) stages, waits only when the next CTA starts the
phase more than about (range - 33) stages later (tools/ly_rfin2_order_check.py prints the slack distribution). No
fence, barrier or release is removed; the deferred release order (kind 1 / 2 / 4 at OKC_WGSFCI of the next segment,
the phase's end flush) and the slot zeroing are the base's.

    python3 tools/dl_ly_rfin2_patch.py [--base F --pin SHA] [--mask 1|3] [--out muse_decode_layer/cand/ly_rfin2.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_rfin.py"
PIN = "c42c12390ce22538260ebea22820002a2e8acaa6f7526920396b86a4b94cc029"
DST = ROOT / "muse_decode_layer/cand/ly_rfin2.py"
MASK = 1
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
if "--mask" in sys.argv:
    MASK = int(sys.argv[sys.argv.index("--mask") + 1])
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"
EXTRA_OLD = 'MAIN_EXTRA_DEFINES = ("-DOKC_RFIN=1", "-DOKC_RFINQ=2")'

DEFS_OLD = "#define OKC_WGSDFIN (OKC_RFIN ? 3 : 2)  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n"
DEFS_NEW = DEFS_OLD + """#ifndef OKC_RFIN2
#define OKC_RFIN2 0  // rfin2: the cut unit's finisher share runs before the CTA's last whole unit (bit0 phase A, bit1 phase B)
#endif
"""

SPLIT_OLD = """__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
"""
SPLIT_NEW = SPLIT_OLD + """// rfin2: a range whose last stage is inside a unit and that holds the whole unit [uw, ue) before that cut share runs
// [g0, uw), [ue, g1), [uw, ue): the finisher's count wait, read-back and the unit's doneAs release leave the phase end;
// the first segment (a depositing share when g0 is inside a unit) stays first. Same walk for both producers and the
// consumers (the ring is matched by issue count).
__device__ __forceinline__ bool wgs_rf2(int pb, int g0, int g1, int& uw, int& ue) {
  ue = ((g1 - 1) >> 4) << 4;
  uw = ue - 16;
  return ((OKC_RFIN2 >> pb) & 1) && (g1 & 15) != 0 && uw >= g0;
}
// the stage at position i of the walk, given the stage the in-order walk would be at
__device__ __forceinline__ int wgs_rf2_at(int pb, int g0, int g1, int i, int gs) {
  int uw, ue;
  if (!wgs_rf2(pb, g0, g1, uw, ue)) return gs;
  const int n0 = uw - g0, nc = g1 - ue;
  return i == n0 ? ue : i == n0 + nc ? uw : gs;
}
__device__ __forceinline__ int wgs_rf2_ge(int pb, int g0, int g1) {  // end of the walk's first sub-range
  int uw, ue;
  return wgs_rf2(pb, g0, g1, uw, ue) ? uw : g1;
}
"""

JUMP_OLD = "      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)\n"
JUMP_NEW = """#if OKC_RFIN2
      gs = wgs_rf2_at(pb, g0, g1, i, gs);
#else
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
#endif
"""

PHASE_OLD = """  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
"""
PHASE_NEW = """#if OKC_RFIN2
  for (int gs = g0, ge = wgs_rf2_ge(PB, g0, g1);;) {
    if (gs >= ge) {  // the walk's next sub-range (wgs_rf2); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1];
      int uw, ue;
      if (!wgs_rf2(PB, a0, a1, uw, ue)) break;
      if (ge == uw) { gs = ue; ge = a1; } else if (ge == a1) { gs = uw; ge = ue; } else break;
    }
#else
""" + PHASE_OLD + "#endif\n"


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if not sha.startswith(PIN[:16]):
        sys.exit(f"base {SRC.name} is {sha[:16]}, expected {PIN[:16]}")
    text = raw.decode("utf-8")
    a, b = text.index(START) + len(START), text.index(END)
    cu, head, tail = text[a:b], text[:a], text[b:]
    for old, new, n in ((DEFS_OLD, DEFS_NEW, 1), (SPLIT_OLD, SPLIT_NEW, 1), (JUMP_OLD, JUMP_NEW, 2), (PHASE_OLD, PHASE_NEW, 1)):
        if cu.count(old) != n:
            sys.exit(f"anchor count {cu.count(old)} (want {n}): {old[:70]!r}")
        cu = cu.replace(old, new)
    if tail.count(EXTRA_OLD) != 1:
        sys.exit("MAIN_EXTRA_DEFINES anchor")
    tail = tail.replace(EXTRA_OLD, f'MAIN_EXTRA_DEFINES = ("-DOKC_RFIN=1", "-DOKC_RFINQ=2", "-DOKC_RFIN2={MASK}")')
    out = head + cu + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
