#!/usr/bin/env python3
"""ly_mtf2: the decode 1 FAST1 issue segment (mask reduced to copies issued) stamped finer, on a ly_pmt copy
(dl_ly_pmt_patch.py). Instrument only, never a candidate. The TINY build's eight DBG slots per CTA:
    0 entry (thread 0, as before)
    5 FAST1 warp lane 0: its mbarrier inits and fence.mbarrier_init done
    7 FAST1 warp lane 0: the touched-expert mask reduced (the topk_ids loads landed)
    1 FAST1 warp lane 0: v0 and n computed (consumed by an empty asm), before fence.proxy.async
    4 FAST1 warp lane 0: past fence.proxy.async
    6 FAST1 warp lane 0: the first copy's expect_tx and cp.async.bulk issued
    3 FAST1 warp lane 0: both first-unit copies issued
    2 thread 0: phase A end (as before)
route()'s DBG(1) is removed. DBG(3), DBG(4) and DBG(5) at the phase B end, the grid barrier and the exit are removed from this copy.

    python3 tools/dl_ly_mtf2_patch.py --base muse_decode_layer/cand/ly_pmt_stk3.py --out muse_decode_layer/cand/ly_mtf2_stk3.py
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
    (FENCE + "    }\n    if (ln < E8)", FENCE + "      " + STAMP.format(k=5) + "\n    }\n    if (ln < E8)"),
    ("    m = __reduce_or_sync(0xffffffffu, m);\n",
     "    m = __reduce_or_sync(0xffffffffu, m);\n    if (ln == 0) " + STAMP.format(k=7) + "\n"),
    ("      const int n = min(OKC_FAST1, v1 - v0);\n#endif\n      asm volatile(\"fence.proxy.async.shared::cta;\\n\" ::: \"memory\");\n",
     "      const int n = min(OKC_FAST1, v1 - v0);\n#endif\n      asm volatile(\"\" ::\"r\"(v0), \"r\"(n));\n      " + STAMP.format(k=1) + "\n"
     "      asm volatile(\"fence.proxy.async.shared::cta;\\n\" ::: \"memory\");\n      " + STAMP.format(k=4) + "\n"),
    ("::\"r\"(smem_u32(slot)), \"l\"(src), \"r\"(UNIT_BYTES), \"r\"(smem_u32(S.bars + i)) L2H_ARG : \"memory\");\n      }\n      S.preissued",
     "::\"r\"(smem_u32(slot)), \"l\"(src), \"r\"(UNIT_BYTES), \"r\"(smem_u32(S.bars + i)) L2H_ARG : \"memory\");\n"
     "        if (i == 0) " + STAMP.format(k=6) + "\n      }\n      S.preissued"),
    ("      S.preissued = n > 0 ? n : 0;\n    }\n  }\n#endif\n",
     "      S.preissued = n > 0 ? n : 0;\n      " + STAMP.format(k=3) + "\n    }\n  }\n#endif\n"),
    ("  DBG(1);\n", ""),
    ("  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n", "  run_phase<true, BIG>(p, S, xbuf);\n"),
    ("  DBG(4);\n  // phases A and B are over everywhere", "  // phases A and B are over everywhere"),
    ("  DBG(5);  // nothing to reset at exit", "  // nothing to reset at exit"),
]
RING_EDITS = []


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
        if seg.count(old) == 0 and (old.startswith("      const int n = min(") or old.startswith(FENCE)):
            print("fence-early build: anchor absent, stamp skipped:", old.splitlines()[0][:60])
            continue
        seg = sub(seg, old, new)
    text = text[:a] + seg + text[b:]
    text = "# ly_mtf2: instrument (FAST1 issue segment stamps on a pmt copy, tools/dl_ly_mtf2_patch.py), never a candidate.\n" + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
