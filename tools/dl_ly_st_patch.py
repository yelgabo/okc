#!/usr/bin/env python3
"""Weight-stream variants of the layer's fused-MoE helper0 (TINY / TINY2 / MID) on ly_fabq.py.

    python3 tools/dl_ly_st_patch.py --variant rec --out muse_decode_layer/cand/ly_st_rec.py

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 pinned in tools/dl_ly_ul_patch.py; never edited). Variants:
  rec  the MOE-TL-01 timeline (dl_ly_ul_patch.py's tl) on TINY2 and MID, plus a record of every weight copy the CTA
       issues (globaltimer, phase, expert block) and the time warp 0 saw each ring slot full, for
       tools/box_st_rec.py. Records are written only when the launch passes the fourth scratch tensor.
       The same file carries a diagnostic switch, off by default: built with -DST_NOMATH=1 -DST_SPIN=N (box_st_rec.py
       --extra), each unit's decode and mma become a globaltimer spin of N ns (output wrong), keeping the ring, the
       barrier, the refill and the epilogue, so helper0's own stream machinery can be timed with a consumer of known
       speed.
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dl_ly_ul_patch as ul  # noqa: E402

# p.dbg layout past the timeline's 136 x 64 stamps: 136 unused words, then per CTA REC_N records of two words
# (issue time, PB << 31 | expert block) indexed by the CTA's sequence number q (no atomic on the issuing thread), then per CTA REC_N ready times indexed by the CTA's sequence number q, then per CTA REC_N
# ring-slot wait durations of warp 0 (same index).
REC_DEFS = """#ifndef ST_BARE
#define ST_BARE 0  // diagnostic (output wrong): no partials hand-off; the helper credits doneA at once and runs no epilogue
#endif
#ifndef ST_NOEPI
#define ST_NOEPI 0  // diagnostic (output wrong): the helper warp skips its phase A epilogue except the CTA's last unit
#endif
#ifndef ST_PLAIN
#define ST_PLAIN 0  // diagnostic (output wrong): both phases as a plain stream loop over the CTA's ranges (st_plain), helper as ST_BARE
#endif
#if ST_PLAIN
#undef ST_BARE
#define ST_BARE 1
#endif
#ifndef ST_NOROUTE
#define ST_NOROUTE 0  // diagnostic (output wrong, needs ST_PLAIN, TINY2): job table from the FAST1 warp's expert mask, no route()
#endif
#ifndef ST_NOX
#define ST_NOX 0  // diagnostic: no x prefetch, no LUT fill
#endif
#ifndef ST_NOZERO
#define ST_NOZERO 0  // diagnostic (breaks the next launch's ss): no ss / barrier-counter zeroing before the phases
#endif
#if ST_NOROUTE && (!ST_PLAIN || OKC_SPLIT)
#undef ST_NOROUTE
#define ST_NOROUTE 0  // (the SPLIT build has no expert mask: it keeps route())
#endif
#define ST_REC_N 128
#define ST_CNT (136 * 64)
#define ST_ISS (ST_CNT + 136)
#define ST_RDY (ST_ISS + 136 * ST_REC_N * 2)
#define ST_WT (ST_RDY + 136 * ST_REC_N)
#if OKC_TLX
#define ST_ISSUE(kv, blkv, pbv) do { if (TLX_ON) { \\
    const int k_ = (kv); \\
    if (k_ == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); p.dbg[ST_CNT + blockIdx.x] = sm_; } \\
    if (k_ >= 0 && k_ < ST_REC_N) { p.dbg[ST_ISS + (blockIdx.x * ST_REC_N + k_) * 2] = gtimer(); \\
      p.dbg[ST_ISS + (blockIdx.x * ST_REC_N + k_) * 2 + 1] = ((unsigned long long)(pbv) << 31) | (unsigned long long)(blkv); } } } while (0)
#else
#define ST_ISSUE(kv, blkv, pbv) do { } while (0)
#endif
"""

TLS_ANCHOR = "#define TLS(k, v) do { if (TLX_ON) p.dbg[blockIdx.x * 64 + (k)] = (v); } while (0)\n"

RING_OLD = """  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
"""
RING_NEW = """  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
  ST_ISSUE(PB ? S.wsr[1] - S.wsr[0] + v - S.wsr[2] : v - S.wsr[0], blk, PB ? 1 : 0);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
"""
F1_OLD = """        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
"""
F1_NEW = """        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        ST_ISSUE(i, (size_t)e * RB_A + (v & (RB_A - 1)), 0);
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
"""
RDY_OLD = """    if (tl) {
      const unsigned long long t1 = gtimer();
      tl_wait += t1 - tl_t0;
      if (!PB && i == 0) TLW(2, t1);
    }
"""
RDY_NEW = """    if (tl) {
      const unsigned long long t1 = gtimer();
      tl_wait += t1 - tl_t0;
      if (!PB && i == 0) TLW(2, t1);
      if (lane == 0 && q < ST_REC_N) { p.dbg[ST_RDY + blockIdx.x * ST_REC_N + q] = t1; p.dbg[ST_WT + blockIdx.x * ST_REC_N + q] = t1 - tl_t0; }
      if (lane == 0 && !PB && i == 0) { p.dbg[ST_WT + blockIdx.x * ST_REC_N + 120] = clock64(); p.dbg[ST_RDY + blockIdx.x * ST_REC_N + 120] = t1; }
    }
"""
AEND_OLD = "      if (i == nU - 1) TLW(PB ? 6 : 3, t);\n"
AEND_NEW = AEND_OLD + "      if (i == nU - 1 && !PB && lane == 0) { p.dbg[ST_WT + blockIdx.x * ST_REC_N + 121] = clock64(); p.dbg[ST_RDY + blockIdx.x * ST_REC_N + 121] = t; }\n"


NM_START = "#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n"
NM_END = "#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n"
NM_PRE = """#ifndef ST_NOMATH
#define ST_NOMATH 0
#endif
#ifndef ST_SPIN
#define ST_SPIN 0
#endif
#if ST_NOMATH
    {  // diagnostic only (output is wrong): the unit's decode and mma replaced by a globaltimer spin of ST_SPIN ns
      acc[0] = __uint_as_float(sc.x);
      if (ST_SPIN) { const unsigned long long t_ = gtimer(); while (gtimer() - t_ < (unsigned long long)ST_SPIN) {} }
    }
#else
"""


EPI_OLD = "    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n"
EPI_ADD = "    if (ST_NOEPI && !PB && q != nA - 1) continue;\n"


BARE_WAIT = "    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit\n"
BARE_RED = "    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n"
BARE_ARR = "      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n"
BARE_HELPER = "  for (int q = 0; q < Nh; ++q) {\n    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {"
BARE_HELPER_NEW = """  if (ST_BARE) {
    if (lane == 0 && nA > 0) {
      const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
      for (int jj = jf; jj <= jl; ++jj) atomicAdd(p.doneA + jj, min(a1, (jj - njF + 1) * RB_A) - max(a0, (jj - njF) * RB_A));
    }
  } else
"""

HELPER_END = "  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n#endif\n"
PLAIN_FN = HELPER_END + """#if ST_PLAIN
// diagnostic (output wrong): phases A and B as one plain stream over this CTA's ranges on the same ring, with
// skinny_ring_ws's wait, fence, CTA barrier and thread-0 refill and nothing else (st_replay.cu's loop in place)
__device__ void st_plain(const Params& p, Smem& S, unsigned char* ring) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
  const bool tl = TLX_ON && warp == 0;
  unsigned long long tl_wait = 0ull, tl_t0 = 0ull;
  uint32_t keep = 0u;
  for (int q = 0; q < N; ++q) {
    const int sl = q % OKC_NS;
    if (tl) tl_t0 = gtimer();
    mbar_wait(S.bars + sl, (uint32_t)((q / OKC_NS) & 1));
    if (tl) {
      const unsigned long long t1 = gtimer();
      tl_wait += t1 - tl_t0;
      if (lane == 0) {
        if (q < ST_REC_N) { p.dbg[ST_RDY + blockIdx.x * ST_REC_N + q] = t1; p.dbg[ST_WT + blockIdx.x * ST_REC_N + q] = t1 - tl_t0; }
        if (q == 0) { TLS(2, t1); p.dbg[ST_WT + blockIdx.x * ST_REC_N + 120] = clock64(); p.dbg[ST_RDY + blockIdx.x * ST_REC_N + 120] = t1; }
        if (q == nA) TLS(5, t1);
      }
    }
    keep ^= *reinterpret_cast<const uint32_t*>(ring + sl * UNIT_BYTES + UNIT_CODES + warp * 512 + lane * 16);
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0 && q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
    if (tl && lane == 0) {
      const unsigned long long t = gtimer();
      if (q < 46) TLS(18 + q, t);
      if (q == nA - 1) { TLS(3, t); p.dbg[ST_WT + blockIdx.x * ST_REC_N + 121] = clock64(); p.dbg[ST_RDY + blockIdx.x * ST_REC_N + 121] = t; }
      if (q == N - 1) TLS(6, t);
    }
  }
  if (tl && lane == 0) {
    TLS(12, tl_wait); TLS(13, 0ull); TLS(14, 0ull); TLS(15, 0ull); TLS(16, 0ull);
    TLS(17, (unsigned long long)nA | ((unsigned long long)(N - nA) << 16) | ((unsigned long long)S.preissued << 32)
                | ((unsigned long long)S.njS << 40));
  }
  if (keep == 0x9E3779B9u) S.fseq = -1;  // keeps the shared read
  if (tid == 0) S.seq += N;
}
#endif
"""
RP_OLD = """  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
"""
RP_NEW = """#if ST_PLAIN
  st_plain(p, S, reinterpret_cast<unsigned char*>(xbuf));
#else
""" + RP_OLD + "#endif\n"
MASK_OLD = "    m = __reduce_or_sync(0xffffffffu, m);\n"
MASK_NEW = MASK_OLD + """#if ST_NOROUTE
    if (ln == 0) {  // the skinny job table route() would build from this mask: touched experts ascending, no fat jobs
      S.njF = 0; S.njS = __popc(m);
      for (int k = 0; k < __popc(m); ++k) S.jE[k] = (int)__fns(m, 0, k + 1);
    }
#endif
"""
ROUTE_OLD = "  route(p, S);\n#if OKC_WGS && OKC_WGSPR\n"
ROUTE_NEW = """#if ST_NOROUTE
  for (int i = tid; i < EMAX; i += NT) S.pos[i] = -1;  // (the final pass reads pos)
  CTA_SYNC();
#else
  route(p, S);
#endif
#if OKC_WGS && OKC_WGSPR
"""
X_OLD = "  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later\n"
X_NEW = "#if !ST_NOX\n" + X_OLD
X_END_OLD = "    S.lut[i] = b | (b << 16);\n  }\n"
X_END_NEW = X_END_OLD + "#endif\n"
Z_OLD = "    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;\n"
Z_NEW = "#if !ST_NOZERO\n" + Z_OLD + "#endif\n"  # (the barrier counter's zeroing stays: the next launch's grid barrier needs it)


def apply(s, v):
    if v == "rec":
        s = ul.timeline(s)
        s = ul.edit(s, [(TLS_ANCHOR, TLS_ANCHOR + REC_DEFS), (RING_OLD, RING_NEW), (F1_OLD, F1_NEW)], v)
        s = ul.edit(s, [(EPI_OLD, EPI_OLD + EPI_ADD), (BARE_HELPER, BARE_HELPER_NEW + BARE_HELPER)], v, sub=ul.HELPER)
        s = ul.edit(s, [(BARE_WAIT, "    if (!ST_BARE)\n  " + BARE_WAIT), (BARE_RED, "    if (!ST_BARE)\n  " + BARE_RED),
                        (BARE_ARR, "      if (!ST_BARE)\n  " + BARE_ARR)], v, sub=ul.LOOP)
        s = ul.edit(s, [(RDY_OLD, RDY_NEW), (AEND_OLD, AEND_NEW), (NM_START, NM_PRE + NM_START), (NM_END, "#endif\n" + NM_END)], v, sub=ul.LOOP)
        s = ul.edit(s, [(HELPER_END, PLAIN_FN), (RP_OLD, RP_NEW), (MASK_OLD, MASK_NEW), (ROUTE_OLD, ROUTE_NEW), (X_OLD, X_NEW),
                        (X_END_OLD, X_END_NEW), (Z_OLD, Z_NEW)], v)
        add = 'MOE_MID_DEFINES += ("@moe_helper0",)\n'
        return ul.once(s, add, add + 'MOE_MID_DEFINES += ("-DOKC_TLX=1",)\n', v)
    sys.exit(f"unknown variant {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = ul.SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ul.PIN:
        sys.exit("ly_fabq.py SHA256 mismatch")
    s = raw.decode()
    for v in a.variant:
        s = apply(s, v)
    out = ROOT / a.out
    out.write_text(s)
    py_compile.compile(str(out), doraise=True)
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
