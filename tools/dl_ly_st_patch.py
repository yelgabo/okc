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
REC_DEFS = """#define ST_REC_N 128
#define ST_CNT (136 * 64)
#define ST_ISS (ST_CNT + 136)
#define ST_RDY (ST_ISS + 136 * ST_REC_N * 2)
#define ST_WT (ST_RDY + 136 * ST_REC_N)
#if OKC_TLX
#define ST_ISSUE(kv, blkv, pbv) do { if (TLX_ON) { \\
    const int k_ = (kv); \\
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
    }
"""


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


def apply(s, v):
    if v == "rec":
        s = ul.timeline(s)
        s = ul.edit(s, [(TLS_ANCHOR, TLS_ANCHOR + REC_DEFS), (RING_OLD, RING_NEW), (F1_OLD, F1_NEW)], v)
        s = ul.edit(s, [(RDY_OLD, RDY_NEW), (NM_START, NM_PRE + NM_START), (NM_END, "#endif\n" + NM_END)], v, sub=ul.LOOP)
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
