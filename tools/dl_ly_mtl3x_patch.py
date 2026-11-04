#!/usr/bin/env python3
"""ly_mtl3x: the decode 1 MoE prologue stamped (ly_mtl3's slots) plus a stamp that does not wait for x: the helper
warp, right after its own WS_JOIN and first ring issues, waits for ring slot 0 to fill. Applied to a ly_pmt copy
(tools/dl_ly_pmt_patch.py) of ly_stk3 or of ly_x1e. Instrument only, never a candidate.

Slots (8 per CTA, the source's DBG buffer; thread 0 unless stated):
    0 entry   4 launch generation stored   1 route() returned   5 past the WS_JOIN that publishes the unit ranges
    3 helper lane 0: ring slot 0 full (its mbar_wait issued after that WS_JOIN; max(join, slot 0 full))
    7 the first phase A job's x registers consumed   6 the first ring slot's mbar_wait returned   2 phase A end
DBG(3), DBG(4) and DBG(5) at the phase B end, the grid barrier and the exit are removed. The helper's added wait
precedes its unit loop, whose first step waits for unit 0's partials, which need slot 0 full anyway.

    python3 tools/dl_ly_mtl3x_patch.py --base muse_decode_layer/cand/ly_pmt_stk3.py --out muse_decode_layer/cand/ly_mtl3x_stk3.py
"""
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

STAMP = "if (OKC_DEBUG && p.dbg) p.dbg[blockIdx.x * 8 + {k}] = gtimer();"
MAIN_EDITS = [
    ("    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n", "    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n    " + STAMP.format(k=4) + "\n"),
    ("    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n  WS_JOIN();\n",
     "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n  WS_JOIN();\n"
     "  if (tid == 0) " + STAMP.format(k=5) + "\n"),
    ("  DBG(4);\n  // phases A and B are over everywhere", "  // phases A and B are over everywhere"),
    ("  DBG(5);  // nothing to reset at exit", "  // nothing to reset at exit"),
    ("  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n",
     "  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n"
     "  if (OKC_DEBUG && p.dbg && lane == 0 && N > 0) { mbar_wait(S.bars, 0u); p.dbg[blockIdx.x * 8 + 3] = gtimer(); }\n"),
]
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
    seg, n = re.subn(r"(  run_phase<true, BIG>\(p, S, xbuf[^;]*\);\n)  DBG\(3\);\n", r"\1", seg)
    if n < 1:
        sys.exit("phase B DBG(3) anchor missing")
    f = seg.index("void skinny_ring_ws(const Params& p")
    e = seg.index("\n}\n", f)
    fn = seg[f:e]
    for old, new in RING_EDITS:
        fn = sub(fn, old, new)
    seg = seg[:f] + fn + seg[e:]
    text = text[:a] + seg + text[b:]
    text = "# ly_mtl3x: instrument (MoE prologue stamps on a pmt copy, tools/dl_ly_mtl3x_patch.py), never a candidate.\n" + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
