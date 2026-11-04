#!/usr/bin/env python3
"""ly_mtf1: the decode 1 FAST1 path (entry to the first weight copies issued) stamped, on a ly_pmt copy
(dl_ly_pmt_patch.py). Instrument only, never a candidate. The TINY build's eight DBG slots per CTA:
    0 entry (thread 0, as before)
    4 FAST1 warp lane 0: top of the FAST1 block (past griddepcontrol.wait)
    5 FAST1 warp lane 0: its mbarrier inits and fence.mbarrier_init done
    7 FAST1 warp lane 0: the touched-expert mask reduced (the topk_ids loads landed)
    3 FAST1 warp lane 0: its two first-unit copies issued
    1 thread 0: route() returned (as before)
    6 thread 0: the first ring slot's mbar_wait returned
    2 thread 0: phase A end (as before)
DBG(3), DBG(4) and DBG(5) at the phase B end, the grid barrier and the exit are removed from this copy.

    python3 tools/dl_ly_mtf1_patch.py --base muse_decode_layer/cand/ly_pmt_stk3.py --out muse_decode_layer/cand/ly_mtf1_stk3.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

STAMP = "if (OKC_DEBUG && p.dbg) p.dbg[blockIdx.x * 8 + {k}] = gtimer();"
FENCE = '      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n'
MAIN_EDITS = [
    ("    const int ln = tid & 31;\n#if OKC_SPLIT\n",
     "    const int ln = tid & 31;\n    if (ln == 0) " + STAMP.format(k=4) + "\n#if OKC_SPLIT\n"),
    (FENCE + "    }\n    if (ln < E8)", FENCE + "      " + STAMP.format(k=5) + "\n    }\n    if (ln < E8)"),
    ("    m = __reduce_or_sync(0xffffffffu, m);\n#endif\n",
     "    m = __reduce_or_sync(0xffffffffu, m);\n    if (ln == 0) " + STAMP.format(k=7) + "\n#endif\n"),
    ("      S.preissued = n > 0 ? n : 0;\n    }\n  }\n#endif\n",
     "      S.preissued = n > 0 ? n : 0;\n      " + STAMP.format(k=3) + "\n    }\n  }\n#endif\n"),
    ("  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n", "  run_phase<true, BIG>(p, S, xbuf);\n"),
    ("  DBG(4);\n  // phases A and B are over everywhere", "  // phases A and B are over everywhere"),
    ("  DBG(5);  // nothing to reset at exit", "  // nothing to reset at exit"),
]
FN = "__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {"
RING_EDITS = [
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
    text = "# ly_mtf1: instrument (FAST1 path stamps on a pmt copy, tools/dl_ly_mtf1_patch.py), never a candidate.\n" + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
