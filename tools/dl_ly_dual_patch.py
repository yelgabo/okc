#!/usr/bin/env python3
"""ly_dual: the layer's fused-MoE helper0 at two CTAs per SM (264 CTAs, a 2-slot weight ring each).

    python3 tools/dl_ly_dual_patch.py [--base muse_decode_layer/cand/ly_fabq.py] [--pin SHA256]
        [--builds tiny,tiny2,mid] [--out muse_decode_layer/cand/ly_dual.py]

Why (docs/runs-decode-layer.md "2026-11-04 - Phase A producer cap (box)"): with the math removed helper0 streams
phase A at the replay's 4.7 to 4.95 TB/s in place, so the stream is not the limit; with the math the unit loop runs
1.22 to 1.26 us per unit at the 1740 to 1780 MHz the math's power sets, and the consumer waits only 1.6 to 3 us per
CTA. About half of each unit's time is the per-unit barrier, hand-off and refill rather than math, so a second CTA
on the SM can run its math under the first one's stalls. The ring drops from 3 to 2 slots so two CTAs fit the SM's
shared memory (2 x (36.7 KB tables + 73.7 KB ring + 1 KB)); the launch bound allows 112 registers per thread
(65,536 / (2 x 288)). The unit ranges, grid barrier, dynamic pool and final pass already index by gridDim.x, so the
larger grid changes only how many units each CTA streams. No fence, barrier or arithmetic order changes: bitwise
output.
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dl_ly_ul_patch as ul  # noqa: E402

SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_dual.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = ul.PIN
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
BUILDS = "tiny,tiny2,mid"
if "--builds" in sys.argv:
    BUILDS = sys.argv[sys.argv.index("--builds") + 1]

LB_OLD = "__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {\n"
LB_NEW = """#ifndef OKC_MINB
#define OKC_MINB 1  // CTAs per SM; the launcher multiplies the grid by it
#endif
__global__ void __launch_bounds__(NTL, OKC_MINB) moe_fused(Params p) {
"""
GRID_OLD = "  int device=0; cudaDeviceProp prop{}; int resident=0;\n"
GRID_NEW = "  grid *= OKC_MINB;\n" + GRID_OLD
CHECK_OLD = "  TORCH_CHECK(grid>0 && grid<=prop.multiProcessorCount && grid<=136 && grid<160,\n"
CHECK_NEW = "  TORCH_CHECK(grid>0 && grid<=OKC_MINB*prop.multiProcessorCount && grid<=OKC_MINB*136 && grid<OKC_MINB*160,\n"
DEF_ANCHOR = 'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n'
NAMES = {"tiny": "MOE_TINY_DEFINES", "tiny2": "MOE_TINY2_DEFINES", "mid": "MOE_MID_DEFINES"}


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit(f"{SRC.name} SHA256 mismatch")
    s = raw.decode()
    s = ul.edit(s, [(LB_OLD, LB_NEW), (GRID_OLD, GRID_NEW), (CHECK_OLD, CHECK_NEW)], "dual")
    add = "".join(f'{NAMES[b]} += ("-DOKC_NS=2", "-DOKC_MINB=2")\n' for b in BUILDS.split(","))
    s = ul.once(s, DEF_ANCHOR, DEF_ANCHOR + add, "dual")
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(DST.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
