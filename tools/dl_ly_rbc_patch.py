#!/usr/bin/env python3
"""ly_rbc: the T = 16 / 32 row boundaries' and route's launch geometry as module tables (sweepable by tools/ly_arms.py).

Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a1..., checked). At decode 16 / 32 the Gluon row boundary
`_graft_row_boundary_pre_cg` (b2 at T = 16; b1 and b2 at T = 32; one 32-warp program per token row, 4 elements per
vector, 8 per thread) and the route `_sl_tile_route_cg` / its prq form (8 warps, one pooled part, top-8 by tl.topk)
each hold 1.6 to 2.9 us of critical path for floors of 0 to 0.7 us; their geometry was set at other T and never
swept here. This tool makes three literal tuples tables with the base's values, so the file as written is the
base's launch for launch:

  RB_CFG[(T, post)] = (V, NW)  for the T = 16 b2 call and the 16 < T <= 32 call in `_pdl8_graft_boundary`
                               (num_warps = NW), base (4, 32);
  RT_CFG[T] = (num_warps, PARTS, MODE)  for the T in (16, 32) route calls, base (8, 1, 0).

--rb "16T:V,NW;32F:V,NW;32T:V,NW" and --rt "16:nw,parts,mode;32:..." write other values into the file. Every
variant keeps the kernels' bodies: the wait, the trigger, the loads after the wait, the stores and the math are the
same statements; only the reduction tree over the row (the layout) and the route's split of the pooled row change.

    python3 tools/dl_ly_rbc_patch.py [--rb ...] [--rt ...] [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_rbc.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
RB = {(16, True): (4, 32), (32, False): (4, 32), (32, True): (4, 32)}
RT = {16: (8, 1, 0), 32: (8, 1, 0)}
if "--rb" in sys.argv:
    for item in sys.argv[sys.argv.index("--rb") + 1].split(";"):
        k, v = item.split(":")
        RB[(int(k[:-1]), k[-1] == "T")] = tuple(int(x) for x in v.split(","))
if "--rt" in sys.argv:
    for item in sys.argv[sys.argv.index("--rt") + 1].split(";"):
        k, v = item.split(":")
        RT[int(k)] = tuple(int(x) for x in v.split(","))

EDITS = [
    ('                                                 HIDDEN, width, True, 4, 32, num_warps=32, launch_pdl=True)',
     '                                                 HIDDEN, width, True, *RB_CFG[(T, True)], num_warps=RB_CFG[(T, True)][1], launch_pdl=True)', 1),
    ('                                         g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,\n'
     '                                         num_warps=32, launch_pdl=True)',
     '                                         g["beta" + pfx], h_out, s_out, HIDDEN, width, post, *RB_CFG[(T, post)],\n'
     '                                         num_warps=RB_CFG[(T, post)][1], launch_pdl=True)', 1),
    ('{16: (8, 1), 32: (8, 1)}[T]', 'RT_CFG[T][:2]', 4),
    ('EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,', 'EARLY=False, MODE=RT_CFG[T][2], PARTS=parts, PRELOAD=True,', 4),
]


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new, n in EDITS:
        if text.count(old) != n:
            sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old[:80]!r}")
        text = text.replace(old, new)
    text = (f"# {DST.stem}: ly_stk1 with the T = 16 / 32 row boundary and route geometry as tables RB_CFG {RB} and "
            f"RT_CFG {RT} (tools/dl_ly_rbc_patch.py).\n" + text +
            f"\n\n# rbc (tools/dl_ly_rbc_patch.py): (V, NW) of the T = 16 / 32 Gluon row boundary by (T, post); "
            f"(num_warps, PARTS, MODE) of the T = 16 / 32 route\nRB_CFG = {RB!r}\nRT_CFG = {RT!r}\n")
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
