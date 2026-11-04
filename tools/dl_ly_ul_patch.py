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
  dq   the DQ7 unpack's r3 merge as an IMAD (x * one + y with an opaque 1) instead of a LOP3: one ALU-pipe op per
       code word moves to the FMA pipe; bitwise identical (disjoint operands). Taken from task 1.3a's cl_unit_loop.cuh.
  pr   a second helper warp (refill warp, NTL = NT + 64) takes the refill off warp 0: compute warps store partials,
       fence and arrive on barrier 8 + q % NS (NT + 32 threads); the refill warp waits on it, refills the slot with
       unit q + NS and arrives on wsrf for the epilogue helper, which is unchanged. Same generation argument as nb.
  er   early slot release: each warp loads its whole 4.5 KB segment of the unit into registers right after the
       slot wait, fences, and arrives on barrier 8 + q % NS; warp 0 waits on it and thread 0 refills the slot with
       unit q + NS at the start of unit q instead of at its end (one more unit of lead for the stream, no extra
       shared memory). The end-of-unit CTA barrier stays for the partials hand-off.
  pw   per-warp ring: every unit copy is split into 8 per-warp segments (4 KB codes + 512 B scales) on per-warp
       barriers; each warp waits only for its own segment and its lane 0 refills it and arrives on the partials
       barrier (count NW) after a __syncwarp. No CTA barrier per unit except for the kept last units.
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


def edit_all(s, old, new, what):
    i = s.index(H0_START) + len(H0_START)
    j = s.index(H0_END, i)
    t = s[i:j]
    if t.count(old) < 1:
        sys.exit(f"{what}: anchor missing: {old[:80]!r}")
    return s[:i] + t.replace(old, new) + s[j:]


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


PW_SMEM_OLD = "  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free\n"
PW_SMEM_NEW = PW_SMEM_OLD + "#if OKC_UPW\n  uint64_t pwb[8 * 8];  // per-warp ring: slot s, warp w's segment full (s * NW + w)\n#endif\n"
PW_INIT_OLD = "for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);"
PW_INIT_NEW = ("for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n"
               "#if OKC_UPW\n      for (int k = 0; k < OKC_NS * NW; ++k) mbar_init(S.pwb + k, 1);\n#endif\n")
PW_WSRF_OLD = "mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW)"
PW_WSRF_NEW = "mbar_init(S.wsrf + k, (OKC_WSB && !OKC_UPW) ? 1 : NW)"
PW_ISSUE_OLD = """  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}
"""
PW_ISSUE_NEW = """  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
#if OKC_UPW
  const int sl = (int)(bar - S.bars);
#pragma unroll 1
  for (int w = 0; w < NW; ++w) pw_seg(src, slot, const_cast<uint64_t*>(S.pwb) + sl * NW + w, w);
#else
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
#endif
}
"""
PW_SEG = """#if OKC_UPW
// warp w's part of a unit: its 4 KB of codes (chunks 8w .. 8w + 7) and its 512 B of scales, on its own barrier
__device__ __forceinline__ void pw_seg(const unsigned char* src, unsigned char* slot, uint64_t* bar, int w) {
  constexpr uint32_t SEGC = UNIT_CODES / NW, SEGS = UNIT_SC / NW;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(SEGC + SEGS) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot + w * SEGC)), "l"(src + w * SEGC), "r"(SEGC), "r"(smem_u32(bar)) L2H_ARG : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot + UNIT_CODES + w * SEGS)), "l"(src + UNIT_CODES + w * SEGS), "r"(SEGS), "r"(smem_u32(bar)) L2H_ARG : "memory");
}
#endif
"""
PW_SEG_ANCHOR = "template <bool PB>\n__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n"
PW_FAST1_OLD = """        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
"""
PW_FAST1_NEW = """#if OKC_UPW
#pragma unroll 1
        for (int w = 0; w < NW; ++w) pw_seg(src, slot, S.pwb + i * NW + w, w);
#else
""" + PW_FAST1_OLD + "#endif\n"
PW_WAIT_OLD = "    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n"
PW_WAIT_NEW = ("#if OKC_UPW\n    mbar_wait(S.pwb + sl * NW + warp, (uint32_t)((q / OKC_NS) & 1));\n#else\n"
               + PW_WAIT_OLD + "#endif\n")
PW_LOOP_NEW = """#if OKC_UPW
    if (OKC_WSBL && PB && i >= nU - OKC_WSBLK) {
      CTA_SYNC();  // kept units: every warp's partials visible to warps 0..K-1; no refill (OKC_WSBLK <= OKC_NS)
    } else {
      __syncwarp();  // every lane's reads of its segment and its partials precede lane 0's refill and arrival
      if (lane == 0) {
        const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
        if (q + OKC_NS < N) ws_issue_seg(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring, warp);
        mbar_arrive1(S.wsrf + bb);
      }
    }
#else
""" + NB_LOOP_OLD + "#endif\n"
PW_WSISSUE_OLD = "__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {\n"
PW_WSISSUE_NEW = """#if OKC_UPW
__device__ __forceinline__ void ws_issue_seg(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring, int w) {
  const int sl = q % OKC_NS;
  const bool PB = q >= nA;
  const int v = PB ? b0 + (q - nA) : a0 + q;
  const int lg = PB ? RB_B_LOG : RB_A_LOG;  // v >= 0 and the block counts are powers of two: no divide
  const int j = S.njF + (v >> lg), rb = v & ((1 << lg) - 1);
  const size_t blk = ((size_t)S.jE[j] << lg) + rb;
  pw_seg((PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES, ring + sl * UNIT_BYTES, S.pwb + sl * NW + w, w);
}
#endif
""" + PW_WSISSUE_OLD
PW_DECL_OLD = "__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\n"
PW_DECL_NEW = PW_DECL_OLD + ("#if OKC_UPW\n__device__ __forceinline__ void ws_issue_seg(const Params& p, Smem& S, int q, int nA, int a0, int b0, "
                             "unsigned char* ring, int w);\n#endif\n")
PW_DEFS = default("OKC_UPW", 0, "unit loop: per-warp ring segments, each warp refills and releases its own part")
PW_ASSERT_NEW = ("#define OKC_WSRF 0\n"
                 "static_assert(!OKC_UPW || (OKC_WSB && OKC_WSBLK <= OKC_NS && OKC_NS * NW <= 64 && UNIT_CODES % NW == 0 && UNIT_SC % (16 * NW) == 0), \"OKC_UPW layout\");\n")


ER_SC_OLD = "    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n"
ER_SC_NEW = ER_SC_OLD + """#if OKC_UER
    uint4 wv[NCH];  // the warp's whole segment in registers: the slot is released before the unit's math
#pragma unroll
    for (int c = 0; c < NCH; ++c) wv[c] = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    if (warp == 0) {
      asm volatile("bar.sync %0, %1;\\n" ::"r"(UNB_BAR0 + sl), "n"(NT) : "memory");  // every warp has read slot sl
      if (tid == 0) {
        const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
        if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      }
    } else {
      asm volatile("bar.arrive %0, %1;\\n" ::"r"(UNB_BAR0 + sl), "n"(NT) : "memory");
    }
#endif
"""
ER_W_OLD = "      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n"
ER_W_NEW = ("#if OKC_UER\n      const uint4 w = wv[c];\n#else\n" + ER_W_OLD + "#endif\n")
ER_LOOP_NEW = """#if OKC_UER
    CTA_SYNC();  // partials visible; the slot was refilled at the unit's start
    if (tid == 0 && !(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);
#else
""" + NB_LOOP_OLD + "#endif\n"
ER_DEFS = (default("OKC_UER", 0, "unit loop: each warp's segment to registers first, slot released and refilled at unit start")
           + "constexpr int UNB_BAR0 = 8;  // named barriers 8 .. 8 + OKC_NS - 1 (1, 2, 3, 4, 7 are taken)\n")
ER_ASSERT_NEW = ("#define OKC_WSRF 0\n"
                 "static_assert(!OKC_UER || (OKC_WSB && OKC_NS <= 4), \"OKC_UER: barrier ids 8 .. 11\");\n")


PR_NTL_OLD = "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);"
PR_NTL_NEW = "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_UPR ? 64 : 32) : 0);"
PR_DISPATCH_OLD = "  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n"
PR_DISPATCH_NEW = """  if (tid >= NT) {
#if OKC_UPR
    if (tid >= NT + 32) { ws_producer(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
    ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
"""
PR_LOOP_NEW = """#if OKC_UPR
    if (OKC_WSBL && PB && i >= nU - OKC_WSBLK) {
      CTA_SYNC();  // kept units: every warp's partials visible to warps 0..K-1; no refill (OKC_WSBLK <= OKC_NS)
    } else {
      asm volatile("bar.arrive %0, %1;\\n" ::"r"(UNB_BAR0 + sl), "n"(NT + 32) : "memory");
    }
#else
""" + NB_LOOP_OLD + "#endif\n"
PR_PRODUCER = """#if OKC_UPR
// refill warp: waits for every compute warp to be done with unit q (slot read and fenced, partials stored), refills
// the slot with unit q + NS and hands the partials to the epilogue helper, so no compute warp carries the refill
__device__ void ws_producer(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    asm volatile("bar.sync %0, %1;\\n" ::"r"(UNB_BAR0 + q % OKC_NS), "n"(NT + 32) : "memory");
    if (lane == 0) {
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
      mbar_arrive1(S.wsrf + q % NRB);
    }
  }
  WS_JOIN();
}
#endif

"""
PR_HELPER_ANCHOR = "__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n"
PR_DEFS = (default("OKC_UPR", 0, "unit loop: a refill warp waits on the per-unit named barrier; compute warps only arrive")
           + "constexpr int UNB_BAR0 = 8;  // named barriers 8 .. 8 + OKC_NS - 1 (1, 2, 3, 4, 7 are taken)\n")
PR_ASSERT_NEW = ("#define OKC_WSRF 0\n"
                 "static_assert(!OKC_UPR || (OKC_WSB && OKC_WSBLK <= OKC_NS && OKC_NS <= 4), \"OKC_UPR: kept units must not need a refill\");\n")


DQ_FN_OLD = "__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {\n"
DQ_FN_NEW = """// dq4 with the r3 merge as an IMAD (x * one + y, `one` an opaque runtime 1): the two masked operands are disjoint,
// so the add is the OR, and it runs on the FMA pipe instead of the ALU pipe that the decode saturates (from task 1.3a)
__device__ __forceinline__ void dq4o(uint32_t q, uint32_t one, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) * one + (__funnelshift_l(q, q, 14) & 0x01000100u);
}
""" + DQ_FN_OLD
DQ_K16_OLD = "  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n"
DQ_K16_NEW = DQ_K16_OLD + "  const uint32_t one = 1u + p.zero;  // opaque 1 (dq4o's r3 merge IMAD)\n"
DQ_CALL_OLD = "        dq4(ww[st], r0, r1, r2, r3);\n"
DQ_CALL_NEW = "        dq4o(ww[st], one, r0, r1, r2, r3);\n"


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
    if v == "er":
        s = edit(s, [(DEF_ANCHOR, DEF_ANCHOR + ER_DEFS.replace("OKC_UER 0", "OKC_UER 1")), (NB_ASSERT_OLD, ER_ASSERT_NEW)], v)
        return edit(s, [(ER_SC_OLD, ER_SC_NEW), (ER_W_OLD, ER_W_NEW), (NB_LOOP_OLD, ER_LOOP_NEW)], v, sub=LOOP)
    if v == "dq":
        s = edit(s, [(DQ_FN_OLD, DQ_FN_NEW)], v)
        return edit(s, [(DQ_K16_OLD, DQ_K16_NEW), (DQ_CALL_OLD, DQ_CALL_NEW)], v, sub=LOOP)
    if v == "pr":
        s = edit(s, [(DEF_ANCHOR, DEF_ANCHOR + PR_DEFS.replace("OKC_UPR 0", "OKC_UPR 1")), (NB_ASSERT_OLD, PR_ASSERT_NEW),
                     (PR_NTL_OLD, PR_NTL_NEW), (PR_DISPATCH_OLD, PR_DISPATCH_NEW),
                     (PR_HELPER_ANCHOR, PR_PRODUCER + PR_HELPER_ANCHOR)], v)
        return edit(s, [(NB_LOOP_OLD, PR_LOOP_NEW)], v, sub=LOOP)
    if v == "pw":
        s = edit(s, [(DEF_ANCHOR, DEF_ANCHOR + PW_DEFS.replace("OKC_UPW 0", "OKC_UPW 1"))], v)
        s = edit(s, [(PW_SMEM_OLD, PW_SMEM_NEW), (PW_SEG_ANCHOR, PW_SEG + PW_SEG_ANCHOR), (PW_ISSUE_OLD, PW_ISSUE_NEW),
                     (PW_FAST1_OLD, PW_FAST1_NEW), (NB_ASSERT_OLD, PW_ASSERT_NEW), (PW_DECL_OLD, PW_DECL_NEW),
                     (PW_WSISSUE_OLD, PW_WSISSUE_NEW)], v)
        s = edit_all(s, PW_INIT_OLD, PW_INIT_NEW, v)
        s = edit_all(s, PW_WSRF_OLD, PW_WSRF_NEW, v)
        return edit(s, [(PW_WAIT_OLD, PW_WAIT_NEW), (NB_LOOP_OLD, PW_LOOP_NEW)], v, sub=LOOP)
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
