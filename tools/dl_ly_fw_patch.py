#!/usr/bin/env python3
"""ly_fw: in the skinny MoE builds (moe_helper0: TINY T = 1, TINY2 T = 8, MID T = 16) only the CTAs that own a final
item wait at the grid barrier; the others release their arrival and exit.

Base: muse_decode_layer/cand/ly_stk4.py (SHA256 a054c8d5..., checked). Target: the decode 1 barrier release latency
(docs/runs-decode-layer.md: 0.9 to 1.1 us from the last arrival to the release, 1.26 to 1.31 us from the phase B end
to the barrier). In the base all 132 CTAs spin on one counter line with ld.acquire while the arrivals' atomics queue at
the same L2 slice, and only M x 4 CTAs (4 at T = 1, 32 at T = 8, 64 at T = 16) have final items. Hypothesis: with 4
pollers the last arrival lands and is seen sooner; the A/B decides.

Change (helper0 source only; MOE_CUDA_SRC's barrier is untouched):
- every CTA arrives with __threadfence + red.release.gpu add on the launch's counter (was atom.acq_rel; the arriving
  CTA never needed the old value except to find the last arriver);
- CTAs with blockIdx.x < nfw = min(G, M x FSEG) poll the counter to G (nanosleep only when nfw > 8, so T = 8 / 16
  poll as before); the others return;
- the generation bump moves from the last arriver to CTA 0 after its wait (no CTA of this launch reads ctr[3] after
  its prologue, and the next launch starts after this grid ends: the MoE launch has no PDL edge);
- the per-launch flag zeroing (ctr[0..1], doneA) runs on the nfw waiting CTAs, after their wait.

Happens-before: each CTA's phase A / B stores and ss atomics -> WS_JOIN / CTA_SYNC (bar.sync, CTA scope) ->
tid 0 __threadfence (fence.sc.gpu, cumulative) -> red.release.gpu on bc. A waiting CTA's tid 0 ld.acquire.gpu reads
G, the last value of bc's modification order made only of these RMWs, so it synchronizes with every arrival's release
(each later RMW continues the release sequence); CTA_SYNC then orders its threads' ybuf / ss reads and doneA zeroing
after it. Every doneA read is before its CTA's release, so zeroing after the acquire has no WAR race. The spin is safe
because the launch is cooperative (cudaLaunchCooperativeKernel after a residency check: all CTAs co-resident). The poll
has no cap and cannot fall through.

    python3 tools/dl_ly_fw_patch.py [--base muse_decode_layer/cand/ly_stk4.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk4.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_fw.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

START = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\""

OLD = """  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
"""

NEW = """  CTA_SYNC();
#if OKC_FINCOL && !OKC_WGS
  // only CTAs that own a final item wait; for the others the release below is their last access to launch state
  const int nfw = min((int)gridDim.x, p.M * (HD / (NT * 4)));
#else
  const int nfw = gridDim.x;
#endif
  const bool fwait = (int)blockIdx.x < nfw;
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(bc), "r"(1) : "memory");
    if (fwait) {
      while (ld_acquire(bc) < (int)gridDim.x) {
        if (nfw > 8) __nanosleep(64);
      }
      if (blockIdx.x == 0) atomicAdd(p.ctr + 3, 1);
    }
  }
  if (!fwait) return;
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (the waiting CTAs' share)
  {
    const int i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += nfw * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += nfw * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += nfw * NT) p.doneAs[i] = 0;
#endif
  }
"""

HEADER = ("# ly_fw: ly_stk4 with the skinny MoE builds' grid barrier waited on only by the CTAs that own final items "
          "(tools/dl_ly_fw_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    a = text.index(START)
    b = text.index('"""', a + len(START))
    region = text[a:b]
    if region.count(OLD) != 1:
        sys.exit(f"barrier anchor found {region.count(OLD)} times in helper0, expected 1")
    text = text[:a] + region.replace(OLD, NEW) + text[b:]
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
