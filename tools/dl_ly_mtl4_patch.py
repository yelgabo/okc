#!/usr/bin/env python3
"""ly_mtl4: the decode 1 doneA credit split around its fence, on a ly_mtl instrument copy (tools/dl_ly_mtl_patch.py).

Instrument only, never a candidate. Slot use (eight per CTA, the pmt runner's count; its own summary of slots 1 and
3 to 5 is meaningless for this copy):
    0 entry   2 thread 0 phase A end   6 first ring slot full   7 first phase B unit past its doneA wait (as ly_mtl)
    1 helper lane 0: the CTA's last phase A unit's partials are in (after mbar_wait on wsrf)
    3 helper lane 0: the last unit's epilogue issued (inter stores), just before the credit fence
    4 helper lane 0: after the credit fence (__threadfence, or fence.acq_rel.gpu in a cfar copy)
    5 helper lane 0: after __syncwarp and the atomicAdd issue
DBG(1), DBG(3), DBG(4) and DBG(5) of thread 0 (after route, phase B end, past the barrier, exit) are removed.

    python3 tools/dl_ly_mtl4_patch.py --base muse_decode_layer/cand/ly_mtl_rfx.py --out muse_decode_layer/cand/ly_mtl4_rfx.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

STAMP = "if (OKC_DEBUG && p.dbg && lane == 0) p.dbg[blockIdx.x * 8 + {k}] = gtimer();"
FENCES = ("        __threadfence();\n", "        asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");  // release pattern with lane 0's atomicAdd\n")
EDITS = [
    ("  DBG(1);\n  {  // the previous launch's ss copy", "  {  // the previous launch's ss copy"),
    ("  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n", "  run_phase<true, BIG>(p, S, xbuf);\n"),
    ("  DBG(4);\n  // phases A and B are over everywhere", "  // phases A and B are over everywhere"),
    ("  DBG(5);  // nothing to reset at exit", "  // nothing to reset at exit"),
    ("    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n",
     "    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n    if (q == nA - 1) " + STAMP.format(k=1) + "\n"),
    ("            atomicAdd(p.doneA + jj, b - a);\n          }\n        }\n      }\n    } else {",
     "            atomicAdd(p.doneA + jj, b - a);\n          }\n          " + STAMP.format(k=5) + "\n        }\n      }\n    } else {"),
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
    head = "      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters\n"
    i = seg.index(head) + len(head)
    j = seg.index("        __syncwarp();\n", i)
    block = seg[i:j]
    s3, s4 = "        " + STAMP.format(k=3) + "\n", "        " + STAMP.format(k=4) + "\n"
    if block.startswith("#if OKC_CFAR\n") and block.endswith("#endif\n"):
        block = s3 + block + s4
    elif block == FENCES[0]:
        block = s3 + block + s4
    else:
        sys.exit(f"unexpected credit fence block: {block!r}")
    seg = seg[:i] + block + seg[j:]
    text = text[:a] + seg + text[b:]
    text = "# ly_mtl4: instrument (credit fence split on a ly_mtl copy, tools/dl_ly_mtl4_patch.py), never a candidate.\n" + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
