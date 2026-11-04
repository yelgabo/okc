#!/usr/bin/env python3
"""ly_mtl3: the decode 1 MoE prologue (entry to the first unit's compute) stamped, on a ly_pmt copy (dl_ly_pmt_patch.py).

Instrument only, never a candidate. The TINY build's eight DBG slots per CTA are reassigned (the pmt runner keeps
eight; its own summary of slots 3 to 5 is meaningless for this copy, the raw rows are what counts):
    0 entry (thread 0, as before)
    3 FAST1 warp lane 0: its two first-unit copies issued
    4 thread 0: the launch generation (p.ctr[3], volatile) loaded and S.gen / S.ssp stored
    1 thread 0: route() returned (as before)
    5 thread 0: past the WS_JOIN that publishes the unit ranges (the prologue's last barrier)
    7 thread 0: the first phase A job's x rows in registers (every x0 register consumed by an empty asm)
    6 thread 0: the first ring slot's mbar_wait returned
    2 thread 0: phase A end (as before)
DBG(3), DBG(4) and DBG(5) at the phase B end, the grid barrier and the exit are removed from this copy.

    python3 tools/dl_ly_mtl3_patch.py --base muse_decode_layer/cand/ly_pmt_rfx.py --out muse_decode_layer/cand/ly_mtl3_rfx.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

STAMP = "if (OKC_DEBUG && p.dbg) p.dbg[blockIdx.x * 8 + {k}] = gtimer();"
MAIN_EDITS = [
    ("      S.preissued = n > 0 ? n : 0;\n    }\n  }\n#endif\n",
     "      S.preissued = n > 0 ? n : 0;\n      " + STAMP.format(k=3) + "\n    }\n  }\n#endif\n"),
    ("    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n", "    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n    " + STAMP.format(k=4) + "\n"),
    ("    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n  WS_JOIN();\n",
     "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n  WS_JOIN();\n"
     "  if (tid == 0) " + STAMP.format(k=5) + "\n"),
    ("  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n", "  run_phase<true, BIG>(p, S, xbuf);\n"),
    ("  DBG(4);\n  // phases A and B are over everywhere", "  // phases A and B are over everywhere"),
    ("  DBG(5);  // nothing to reset at exit", "  // nothing to reset at exit"),
]
FN = "__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {"
RING_EDITS = [
    ("      load_x_regs<PB>(p, S, j, lane, warp, x0);\n",
     "      load_x_regs<PB>(p, S, j, lane, warp, x0);\n"
     "      if (!PB && i == 0 && OKC_DEBUG && p.dbg) {\n"
     "#pragma unroll\n        for (int k = 0; k < 2 * NCH; ++k) asm volatile(\"\" ::\"r\"(x0[k].x), \"r\"(x0[k].w));\n"
     "        DBG(7);\n      }\n"),
    ("    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n",
     "    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n    if (!PB && i == 0) DBG(6);\n"),
]


def sub(seg, old, new):
    if seg.count(old) != 1:
        sys.exit(f"anchor found {seg.count(old)} times: {old!r}")
    return seg.replace(old, new)


def main():
    text = SRC.read_text()
    a = text.index("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"")
    b = text.index('\n"""\n', a)
    seg = text[a:b]
    for old, new in MAIN_EDITS:
        seg = sub(seg, old, new)
    f = seg.index(FN)
    e = seg.index("\n}\n", f)
    fn = seg[f:e]
    for old, new in RING_EDITS:
        fn = sub(fn, old, new)
    seg = seg[:f] + fn + seg[e:]
    text = text[:a] + seg + text[b:]
    text = "# ly_mtl3: instrument (MoE prologue stamps on a pmt copy, tools/dl_ly_mtl3_patch.py), never a candidate.\n" + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
