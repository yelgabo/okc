#!/usr/bin/env python3
"""ly_mtl: two more MoE stamps on a ly_pmt instrument copy (tools/dl_ly_pmt_patch.py), decode 1 ramp and handoff.

Instrument only, never a candidate. The pmt copy compiles helper0's DBG(k) stamps into the TINY build (thread 0,
p.dbg[cta * 8 + k], taken only when the launch passes the fourth scratch tensor); slots 6 and 7 are WGS-only, so the
TINY build leaves them free. This adds, in skinny_ring_ws:
    DBG(6)  phase A, first unit: thread 0's mbar_wait on the first ring slot returned (first weights in smem)
    DBG(7)  phase B, first unit: thread 0 (warp 0's poller) past its doneA wait and __syncwarp
tools/ly_box_pmt.py --raw N keeps every slot of the MoE rows for the first N inputs.

    python3 tools/dl_ly_mtl_patch.py --base muse_decode_layer/cand/ly_pmt_rfx.py --out muse_decode_layer/cand/ly_mtl_rfx.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

FN = "__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {"
EDITS = [
    ("        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n",
     "        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n"
     "        if (i == 0) DBG(7);\n"),
    ("    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n",
     "    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n    if (!PB && i == 0) DBG(6);\n"),
]


def main():
    text = SRC.read_text()
    a = text.index("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"")
    f = text.index(FN, a)
    e = text.index("\n}\n", f)
    seg = text[f:e]
    for old, new in EDITS:
        if seg.count(old) != 1:
            sys.exit(f"anchor found {seg.count(old)} times in skinny_ring_ws: {old!r}")
        seg = seg.replace(old, new)
    text = text[:f] + seg + text[e:]
    text = "# ly_mtl: instrument (pmt copy + MoE first-slot and first-B stamps, tools/dl_ly_mtl_patch.py), never a candidate.\n" + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
