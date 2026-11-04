#!/usr/bin/env python3
"""ly_anc<N>: the decode 1 sliding attention plan (one request, q_len 1, window) with N FP32 workspace copies instead
of one; host-only (_plan7), no kernel source change.

Stamps (ly_ast) put 1.8 us per CTA between the tile loop end and the ticket's return at decode 1 sliding, and moving
the ticket to a grid boundary (ly_afk) did not shorten it, so the time is the 32 CTAs' red.global.add into the same
4 KB (32 reds per address). With N copies, split s adds into copy s % N (the kernel's existing copy index), so 32 / N
CTAs share each address; the last CTA's finalize sums the N copies (sum_copies: the nc 8 path loads all eight in one
round trip, the general path in groups of 4). Only B == 1 changes (decode 16 sliding also takes this plan branch).

    python3 tools/dl_ly_anc_patch.py N [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_ancN.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
N = int(sys.argv[1])
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / f"muse_decode_layer/cand/ly_anc{N}.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
OLD = ("    if L == 1 and reqs < 24:\n        n = max(1, min(tiles, target // reqs))\n"
       "        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)\n")
NEW = ("    if L == 1 and reqs < 24:\n        n = max(1, min(tiles, target // reqs))\n"
       f"        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc={N} if B == 1 else 1)   # (ly_anc{N}: fewer reds per address)\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    if text.count(OLD) != 1:
        sys.exit(f"anchor found {text.count(OLD)} times")
    text = (f"# ly_anc{N}: ly_v1un with {N} attention workspace copies at decode 1 sliding (tools/dl_ly_anc_patch.py).\n"
            + text.replace(OLD, NEW))
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
