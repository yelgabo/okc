#!/usr/bin/env python3
"""Unit-loop restructures of the layer's fused-MoE skinny_ring_ws (helper0: TINY / TINY2 / MID) on ly_fabq.py.

    python3 tools/dl_ly_ul_patch.py --variant nb [--variant pf ...] --out muse_decode_layer/cand/ly_ul_<v>.py

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 pinned below; never edited). One mechanism per variant; variants
stack. Every variant keeps the math and its order (bitwise output), every fence and the ring discipline.
  nb   the per-unit compute-warp CTA barrier becomes a producer/consumer named barrier with the helper warp: each
       compute warp stores its partials, runs the same fence.proxy.async and arrives (bar.arrive) on barrier
       8 + q % NS without waiting; the helper warp waits on it (bar.sync, 288 threads), refills unit q's slot with
       unit q + NS and reads the partials. It replaces thread 0's refill and the wsrf mbarrier hand-off. The
       WSBL-kept last units keep the CTA barrier among the compute warps (their epilogues read every warp's
       partials). A warp can arrive for unit q + NS only after that unit's slot was refilled, which happens after
       the helper's wait for unit q returned, so NS barrier ids never see two generations at once.
  na   the per-unit compute-warp barrier split into arrive and wait: warps 1..NW-1 store their partials, fence and
       arrive (bar.arrive) on barrier 8 + q % NS without waiting; warp 0 waits (bar.sync, 256 threads), then thread 0
       refills and hands the partials to the helper as before. The kept last units keep the full CTA barrier. A
       warp can arrive for unit q + NS only after thread 0 refilled that slot, after its wait for unit q returned.
  pf   cross-unit software pipelining: after its last mma of unit i, each warp waits for unit i + 1's slot and
       loads that unit's scale word and first PFC code chunks into registers, so the next unit's decode starts
       right after the hand-off instead of behind the slot wait and the shared loads.
  tl   the MOE-TL-01 per-CTA timeline (tools/dl_moe_tl.py edits) on TINY2, for tools/box_moe_tl8.py.
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
H0_START = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
H0_END = '\n"""'
LOOP = ("__device__ void skinny_ring_ws(const Params& p", "__device__ __forceinline__ void ws_issue(")
HELPER = ("__device__ void ws_helper(", "template <bool BIG, bool PB>")


def once(t, old, new, what):
    n = t.count(old)
    if n != 1:
        sys.exit(f"{what}: anchor found {n} times: {old[:80]!r}")
    return t.replace(old, new)


def edit(s, edits, what, sub=None):
    i = s.index(H0_START) + len(H0_START)
    j = s.index(H0_END, i)
    if sub:
        a = s.index(sub[0], i)
        i, j = a, s.index(sub[1], a)
    t = s[i:j]
    for old, new in edits:
        t = once(t, old, new, what)
    return s[:i] + t + s[j:]


DEF_ANCHOR = "#define OKC_NOLOAD 0\n"


def default(name, val, why):
    return f"#ifndef {name}\n#define {name} {val}  // {why}\n#endif\n"


NB_LOOP_OLD = """    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
"""
NB_LOOP_NEW = """#if OKC_UNB
    if (OKC_WSBL && PB && i >= nU - OKC_WSBLK) {
      CTA_SYNC();  // kept units: every warp's partials visible to warps 0..K-1; no refill (OKC_WSBLK <= OKC_NS)
    } else {
      asm volatile("bar.arrive %0, %1;\\n" ::"r"(UNB_BAR0 + sl), "n"(NTL) : "memory");
    }
#else
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
#endif
"""
NB_HELPER_OLD = """    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));"""
NB_HELPER_NEW = """    const int bb = q % NRB;
#if OKC_UNB
    // every compute warp has read unit q's slot (and fenced) and stored its partials
    asm volatile("bar.sync %0, %1;\\n" ::"r"(UNB_BAR0 + q % OKC_NS), "n"(NTL) : "memory");
    if (lane == 0 && q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
#else
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
#endif"""
NB_DEFS = (default("OKC_UNB", 0, "unit loop: compute warps arrive on a named barrier the helper waits on (refill + partials)")
           + "constexpr int UNB_BAR0 = 8;  // named barriers 8 .. 8 + OKC_NS - 1 (1, 2, 3, 4, 7 are taken)\n")
NB_ASSERT_OLD = "#define OKC_WSRF 0\n"
NB_ASSERT_NEW = ("#define OKC_WSRF 0\n"
                 "static_assert(!OKC_UNB || (OKC_WSB && OKC_WSBLK <= OKC_NS && OKC_NS <= 4), \"OKC_UNB: kept units must not need a refill\");\n")

NA_LOOP_NEW = """#if OKC_UNA
    if ((OKC_WSBL && PB && i >= nU - OKC_WSBLK) || warp == 0) {
      if (OKC_WSBL && PB && i >= nU - OKC_WSBLK) CTA_SYNC();  // kept units: every warp's partials visible to warps 0..K-1
      else asm volatile("bar.sync %0, %1;\\n" ::"r"(UNB_BAR0 + sl), "n"(NT) : "memory");
      if (tid == 0) {
        const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
        if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
        if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);
      }
    } else {
      asm volatile("bar.arrive %0, %1;\\n" ::"r"(UNB_BAR0 + sl), "n"(NT) : "memory");
    }
#else
""" + NB_LOOP_OLD + "#endif\n"
NA_DEFS = (default("OKC_UNA", 0, "unit loop: warps 1..NW-1 arrive on the per-unit barrier, warp 0 waits and refills")
           + "constexpr int UNB_BAR0 = 8;  // named barriers 8 .. 8 + OKC_NS - 1 (1, 2, 3, 4, 7 are taken)\n")
NA_ASSERT_NEW = ("#define OKC_WSRF 0\n"
                 "static_assert(!OKC_UNA || (OKC_WSB && OKC_NS <= 4), \"OKC_UNA: barrier ids 8 .. 11\");\n")


PF_WAIT_OLD = """    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
"""
PF_WAIT_NEW = """#if OKC_UPF
    if (i == 0) {
      mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
      const unsigned char* s0 = ring + sl * UNIT_BYTES;
      psc = *reinterpret_cast<const uint4*>(s0 + UNIT_CODES + warp * 512 + lane * 16);
#pragma unroll
      for (int c = 0; c < OKC_PFC; ++c) pw[c] = *reinterpret_cast<const uint4*>(s0 + (warp * NCH + c) * 512 + lane * 16);
    }
#else
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
#endif
"""
PF_SC_OLD = """    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
"""
PF_SC_NEW = """#if OKC_UPF
    const uint4 sc = psc;
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
"""
PF_W_OLD = """      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
#if !OKC_NOSCL"""
PF_W_NEW = """#if OKC_UPF
      const uint4 w = c < OKC_PFC ? pw[c < OKC_PFC ? c : 0] : *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#else
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#endif
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
#if !OKC_NOSCL"""
PF_TAIL_OLD = """#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
"""
PF_TAIL_NEW = """#if OKC_UPF
    if (i + 1 < nU) {  // unit i + 1's slot was issued before this unit's hand-off: waiting here cannot deadlock
      const int q1 = q + 1, sl1 = q1 % OKC_NS;
      mbar_wait(bars + sl1, (uint32_t)((q1 / OKC_NS) & 1));
      const unsigned char* s1 = ring + sl1 * UNIT_BYTES;
      psc = *reinterpret_cast<const uint4*>(s1 + UNIT_CODES + warp * 512 + lane * 16);
#pragma unroll
      for (int c = 0; c < OKC_PFC; ++c) pw[c] = *reinterpret_cast<const uint4*>(s1 + (warp * NCH + c) * 512 + lane * 16);
    }
#endif
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
"""
PF_DECL_OLD = """  uint4 x0[2 * NCH];
  int curj = -1;
"""
PF_DECL_NEW = """  uint4 x0[2 * NCH];
#if OKC_UPF
  uint4 psc, pw[OKC_PFC];  // the next unit's scale word and first code chunks
#endif
  int curj = -1;
"""
PF_DEFS = (default("OKC_UPF", 0, "unit loop: the next unit's slot wait and first shared loads before this unit's hand-off")
           + default("OKC_PFC", 2, "code chunks of the next unit loaded early (OKC_UPF)"))


def apply(s, v):
    if v == "nb":
        s = edit(s, [(DEF_ANCHOR, DEF_ANCHOR + NB_DEFS.replace("OKC_UNB 0", "OKC_UNB 1"))], v)
        s = edit(s, [(NB_ASSERT_OLD, NB_ASSERT_NEW)], v)
        s = edit(s, [(NB_LOOP_OLD, NB_LOOP_NEW)], v, sub=LOOP)
        return edit(s, [(NB_HELPER_OLD, NB_HELPER_NEW)], v, sub=HELPER)
    if v == "na":
        s = edit(s, [(DEF_ANCHOR, DEF_ANCHOR + NA_DEFS.replace("OKC_UNA 0", "OKC_UNA 1"))], v)
        s = edit(s, [(NB_ASSERT_OLD, NA_ASSERT_NEW)], v)
        return edit(s, [(NB_LOOP_OLD, NA_LOOP_NEW)], v, sub=LOOP)
    if v == "pf":
        s = edit(s, [(DEF_ANCHOR, DEF_ANCHOR + PF_DEFS.replace("OKC_UPF 0", "OKC_UPF 1"))], v)
        return edit(s, [(PF_DECL_OLD, PF_DECL_NEW), (PF_WAIT_OLD, PF_WAIT_NEW), (PF_SC_OLD, PF_SC_NEW), (PF_W_OLD, PF_W_NEW),
                        (PF_TAIL_OLD, PF_TAIL_NEW)], v, sub=LOOP)
    if v.startswith("pf") and v[2:].isdigit():
        s = apply(s, "pf")
        return edit(s, [("#define OKC_PFC 2 ", f"#define OKC_PFC {int(v[2:])} ")], v)
    if v == "tl":
        return timeline(s)
    sys.exit(f"unknown variant {v}")


def timeline(s):
    tool = (ROOT / "tools/moe_tiny_timeline_cta.py").read_text()
    cuda = tool[tool.index("# ---------------------------------------------------------------- CUDA"):
                tool.index("# ---------------------------------------------------------------- host")]
    i = s.index(H0_START) + len(H0_START)
    j = s.index(H0_END, i)
    box = {"src": s[i:j]}

    def rep(old, new, count=1):
        n = box["src"].count(old)
        if n != count:
            sys.exit(f"tl: helper0 anchor found {n} times, {count} expected: {old[:100]!r}")
        box["src"] = box["src"].replace(old, new)
    exec(cuda, {"rep": rep})
    s = s[:i] + box["src"] + s[j:]
    add = 'MOE_TINY2_DEFINES += ("@moe_helper0",)\n'
    return once(s, add, add + 'MOE_TINY2_DEFINES += ("-DOKC_TLX=1",)\n', "tl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("ly_fabq.py SHA256 mismatch")
    s = raw.decode()
    for v in sorted(a.variant, key=lambda v: v != "tl"):  # the timeline edits match the base text: first
        s = apply(s, v)
    out = ROOT / a.out
    out.write_text(s)
    py_compile.compile(str(out), doraise=True)
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
