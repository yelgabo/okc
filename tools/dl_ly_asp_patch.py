#!/usr/bin/env python3
"""ly_asp<N>: the decode 1 sliding attention plan (one request, q_len 1, window) with N splits instead of one per
64-key tile (32); host-only (_plan7), no kernel source change. With N < 32 a split holds several tiles, so the
launch takes the speculated-length build without the one-tile fold (attn_kernel<1, false, 9>).

Stamps (ly_ast --mrg) show every CTA's ticket returning at about the same time (4.1 us after qkv's end) whatever its
loop end (2.1 to 2.7 us): the 32 CTAs' red.global.add into one 4 KB region drain at the L2's rate for those lines.
N splits issue N x 1024 floats of reds instead of 32 x 1024, against a longer tile loop per CTA.

    python3 tools/dl_ly_asp_patch.py N [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_aspN.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
N = int(sys.argv[1])
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / f"muse_decode_layer/cand/ly_asp{N}.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
OLD = ("    if L == 1 and reqs < 24:\n        n = max(1, min(tiles, target // reqs))\n"
       "        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)\n")
NEW = ("    if L == 1 and reqs < 24:\n        n = max(1, min(tiles, target // reqs))\n"
       f"        if B == 1:\n            n = min(n, {N})   # (ly_asp{N}: fewer splits, fewer reds into the workspace)\n"
       "        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    if text.count(OLD) != 1:
        sys.exit(f"anchor found {text.count(OLD)} times")
    text = (f"# ly_asp{N}: ly_v1un with {N} attention splits at decode 1 sliding (tools/dl_ly_asp_patch.py).\n"
            + text.replace(OLD, NEW))
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
