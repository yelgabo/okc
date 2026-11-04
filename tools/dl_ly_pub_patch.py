#!/usr/bin/env python3
"""ly_pub: the MAIN WGS build's whole-unit publishes go through a publisher warp (verify 16 / verify 32).

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 3765470032..., checked). Target: docs/runs-decode-layer.md,
"2026-11-04 - Verify 16/32 MoE limit (box)": each segment switch of a consumer warpgroup costs 1.1 us + 0.25 us per
n-tile of stalled stream, and the n-tile-independent part is the deferred flush's gpu-scope release (bar.sync of the
warpgroup, then red.release.gpu by one thread: 0.25 + 0.66 to 0.69 us measured by the B4 lane), run by the four
consumer warpgroups while the 3-deep ring fills.

Change (OKC_PUB=1, passed to the MAIN build only; every other build compiles the same text as the base). A fifth
non-consumer warp (NTL 544 -> 576) is the publisher. For the flush kinds that only release (kind 1: phase A whole unit
-> doneAs; kind 4: phase B whole unit -> the expert's MFIN count; kind 2: phase B cut share -> its fixup count; the
finisher's trailing doneAs / MFIN release) every consumer thread keeps its own fence.proxy.async.global and stores,
then arrives (release.cta) on its warpgroup's request mbarrier (count 128) instead of bar.sync; thread 0 of the
warpgroup first writes the (address, value) mailbox, after waiting on the acknowledge mbarrier (count 1) of the
previous publish. The publisher polls the four request barriers (mbarrier.test_wait, acquire.cta), performs the
red.release.gpu and arrives on the acknowledge barrier. Causality order through the cta-scope arrive / test_wait pair
and then the gpu-scope release is the chain the base already relies on through bar.sync + one thread's release (PTX
ISA memory consistency model: release patterns are cumulative over operations that precede them in causality order,
each synchronizes-with pair morally strong at its scope). Nothing is removed: the proxy fences stay on the writers,
kind 3 (the finisher's acquire, bar.sync and read-back) and the phase A cut path (atom.acq_rel) are unchanged, and
the consumers' stage loop is the base's text. After phase B each warpgroup sends a null-address sentinel and the
publisher exits.

    python3 tools/dl_ly_pub_patch.py [--base F --pin SHA] [--out muse_decode_layer/cand/ly_pub.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
DST = ROOT / "muse_decode_layer/cand/ly_pub.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"
MAIN = 'key, defines = "main", _main_defines(sfx)'

PUBLISH = r"""
#if OKC_PUB
constexpr int PUB_RQ = BX_E + WGS_NX, PUB_AK = PUB_RQ + 4;  // publisher relay barriers: request (128), acknowledge (1)
static_assert(PUB_AK + 4 <= 24, "WGS publisher barriers");
// hands a gpu-scope release to the publisher warp: every thread of the warpgroup arrives (release.cta) behind its own
// stores and proxy fence; the publisher's test_wait (acquire.cta) and red.release.gpu complete the same causality
// chain as bar.sync + one thread's red.release.gpu
__device__ __forceinline__ void wgs_publish(Smem& S, int wg, int wt, int* addr, int val) {
  if (wt == 0) {
    const int n = S.pubn[wg];
    mbar_wait(S.bars + PUB_AK + wg, (uint32_t)(n & 1) ^ 1u);  // the previous mailbox has been consumed
    S.pubaddr[wg] = (unsigned long long)addr;
    S.pubval[wg] = val;
    S.pubn[wg] = n + 1;
  }
  mbar_arrive1(S.bars + PUB_RQ + wg);
}
__device__ void wgs_publisher(const Params& p, Smem& S) {
  WS_JOIN();  // (the publisher only joins once the barriers are initialized; nothing else of it is ordered here)
  if ((threadIdx.x & 31) != 0) return;
  uint32_t ph = 0;  // bit wg: parity of warpgroup wg's next request phase
  int open = 15;
  while (open) {
    bool any = false;
#pragma unroll
    for (int wg = 0; wg < 4; ++wg) {
      if (!((open >> wg) & 1)) continue;
      uint32_t ok;
      asm volatile("{\n .reg .pred q;\n mbarrier.test_wait.parity.shared::cta.b64 q, [%1], %2;\n selp.u32 %0, 1, 0, q;\n}\n"
                   : "=r"(ok) : "r"(smem_u32(S.bars + PUB_RQ + wg)), "r"((ph >> wg) & 1u) : "memory");
      if (!ok) continue;
      any = true;
      ph ^= 1u << wg;
      int* addr = reinterpret_cast<int*>(S.pubaddr[wg]);
      const int val = S.pubval[wg];
      if (addr == nullptr) { open &= ~(1 << wg); continue; }
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(addr), "r"(val) : "memory");
      mbar_arrive1(S.bars + PUB_AK + wg);
    }
    if (!any) __nanosleep(64);
  }
}
#endif
"""

EDITS = [
    # the define, off everywhere but the MAIN build
    ("#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n",
     "#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n"
     "#ifndef OKC_PUB\n#define OKC_PUB 0      // WGS: whole-unit releases by a publisher warp (ly_pub)\n#endif\n"),
    # the fifth warp
    ("constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);",
     "constexpr int NTL = NT + (OKC_WGS ? (OKC_PUB ? 96 : 64) : OKC_WS ? 32 : 0);"),
    # the mailbox
    ("  int wglast[4];      // WGS: this warpgroup completed a split unit\n",
     "  int wglast[4];      // WGS: this warpgroup completed a split unit\n"
     "  unsigned long long pubaddr[4];  // WGS publisher: the release's address per warpgroup (0: the warpgroup is done)\n"
     "  int pubval[4];      // WGS publisher: the release's value\n"
     "  int pubn[4];        // WGS publisher: publishes handed over so far (thread 0 of the warpgroup only)\n"),
    # the helpers, after wgs_cnt / mf_da are defined and before wgs_flush
    ("template <bool PB>\n__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {\n",
     PUBLISH + "template <bool PB>\n__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {\n"),
    # kinds 4 and 1: relay instead of bar.sync + red
    ("  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);\n"
     "  asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");\n"
     "#if OKC_MFIN && (OKC_MFP & 1)\n"
     "  if (pd.kind == 4) {\n",
     "  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);\n"
     "#if OKC_PUB\n"
     "#if OKC_MFIN && (OKC_MFP & 1)\n"
     "  if (pd.kind == 4) {\n"
     "    wgs_publish(S, wg, wt, p.mfl + MF_CE + ((OKC_MFP & 8) ? pd.slot : (int)(mf_gen(S) * 32 + S.jE[j])), 1);\n"
     "    pd.kind = 0;\n"
     "    return;\n"
     "  }\n"
     "#endif\n"
     "  if (pd.kind == 1) {\n"
     "    wgs_publish(S, wg, wt, mf_da(p, S) + j * 16 + (ub >> 1), 1);\n"
     "    pd.kind = 0;\n"
     "    return;\n"
     "  }\n"
     "#endif\n"
     "  asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");\n"
     "#if OKC_MFIN && (OKC_MFP & 1)\n"
     "  if (pd.kind == 4) {\n"),
    # kind 2 (designated-finisher phases): relay
    ("  if (pd.kind == 2) {\n"
     "    if (wt == 0) asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(fc), \"r\"(pd.nst) : \"memory\");\n",
     "  if (pd.kind == 2) {\n"
     "#if OKC_PUB\n"
     "    wgs_publish(S, wg, wt, fc, pd.nst);\n"
     "#else\n"
     "    if (wt == 0) asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(fc), \"r\"(pd.nst) : \"memory\");\n"
     "#endif\n"),
    # the finisher's trailing releases: relay
    ("  if (PB) {\n"
     "    asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");\n"
     "    if (wt == 0) asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), \"r\"(1) : \"memory\");\n"
     "  }\n",
     "  if (PB) {\n"
     "#if OKC_PUB\n"
     "    wgs_publish(S, wg, wt, p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j], 1);\n"
     "#else\n"
     "    asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");\n"
     "    if (wt == 0) asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), \"r\"(1) : \"memory\");\n"
     "#endif\n"
     "  }\n"),
    ("  if (!PB) {\n"
     "    asm volatile(\"fence.proxy.async.global;\\n\" ::: \"memory\");\n"
     "    asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");\n"
     "    if (wt == 0) asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(mf_da(p, S) + j * 16 + (ub >> 1)), \"r\"(1) : \"memory\");\n"
     "  }\n",
     "  if (!PB) {\n"
     "    asm volatile(\"fence.proxy.async.global;\\n\" ::: \"memory\");\n"
     "#if OKC_PUB\n"
     "    wgs_publish(S, wg, wt, mf_da(p, S) + j * 16 + (ub >> 1), 1);\n"
     "#else\n"
     "    asm volatile(\"bar.sync %0, 128;\\n\" ::\"r\"(3 + wg) : \"memory\");\n"
     "    if (wt == 0) asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(mf_da(p, S) + j * 16 + (ub >> 1)), \"r\"(1) : \"memory\");\n"
     "#endif\n"
     "  }\n"),
    # role dispatch
    ("    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n"
     "    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n",
     "    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n"
     "#if OKC_PUB\n"
     "    else if (tid < NT + 64) wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n"
     "    else wgs_publisher(p, S);\n"
     "#else\n"
     "    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n"
     "#endif\n"),
    # barrier init
    ("    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT);\n",
     "    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT);\n"
     "#if OKC_PUB\n"
     "    for (int k = 0; k < 4; ++k) { mbar_init(S.bars + PUB_RQ + k, 128); mbar_init(S.bars + PUB_AK + k, 1); S.pubn[k] = 0; }\n"
     "#endif\n"),
    # the sentinel after phase B
    ("    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n",
     "    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n"
     "#if OKC_PUB\n"
     "    wgs_publish(S, WARP_IDX(tid) >> 2, tid & 127, nullptr, 0);  // the publisher may exit\n"
     "#endif\n"
     "    DBG(3);\n"),
]


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    whole = raw.decode()
    assert whole.count(START) == 1 and whole.count(END) == 1
    head, rest = whole.split(START)
    cu, tail = rest.split(END)
    for old, new in EDITS:
        n = cu.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:120]!r}")
        cu = cu.replace(old, new)
    if tail.count(MAIN) != 1:
        sys.exit("MAIN define anchor")
    tail = tail.replace(MAIN, 'key, defines = "main", tuple(_main_defines(sfx)) + ("-DOKC_PUB=1",)')
    out = head + START + cu + END + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
