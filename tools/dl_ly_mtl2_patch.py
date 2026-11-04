#!/usr/bin/env python3
"""ly_mtl2: the decode 1 handoff's credit path stamped, on a ly_mtl instrument copy (tools/dl_ly_mtl_patch.py).

Instrument only, never a candidate. Repurposes two of the TINY build's eight DBG slots (the pmt runner keeps eight per
CTA): slot 1 (after route()) and slot 4 (past the grid barrier) are no longer written by thread 0; instead the helper
warp's lane 0 writes, at the CTA's last phase A unit (q == nA - 1):
    slot 1  after mbar_wait on the unit's partials, before the epilogue (the compute warps' last A partials are in)
    slot 4  after the doneA credit (after __threadfence, __syncwarp and the atomicAdd issue)
Slots 2 (thread 0's phase A end), 6 (first ring slot full) and 7 (first phase B unit past its doneA wait) stay. The
helper's epilogue and credit are timed as a whole; the fence is the only blocking instruction in it besides the
stores and the partial-buffer loads.

    python3 tools/dl_ly_mtl2_patch.py --base muse_decode_layer/cand/ly_mtl_rfx.py --out muse_decode_layer/cand/ly_mtl2_rfx.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

STAMP = "if (OKC_DEBUG && p.dbg) p.dbg[blockIdx.x * 8 + {k}] = gtimer();"
EDITS = [
    ("  DBG(1);\n  {  // the previous launch's ss copy", "  {  // the previous launch's ss copy"),
    ("  DBG(4);\n  // phases A and B are over everywhere", "  // phases A and B are over everywhere"),
    ("    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n",
     "    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n    if (q == nA - 1 && lane == 0) " + STAMP.format(k=1) + "\n"),
    ("            atomicAdd(p.doneA + jj, b - a);\n          }\n        }\n      }\n    } else {",
     "            atomicAdd(p.doneA + jj, b - a);\n          }\n          " + STAMP.format(k=4) + "\n        }\n      }\n    } else {"),
]


def main():
    text = SRC.read_text()
    a = text.index("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"")
    b = text.index('\n"""\n', a)
    seg = text[a:b]
    for old, new in EDITS:
        if seg.count(old) != 1:
            sys.exit(f"anchor found {seg.count(old)} times in helper0: {old!r}")
        seg = seg.replace(old, new)
    text = text[:a] + seg + text[b:]
    text = "# ly_mtl2: instrument (credit-path stamps on a ly_mtl copy, tools/dl_ly_mtl2_patch.py), never a candidate.\n" + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
