#!/usr/bin/env python3
"""ly_v1r16: ly_v1un with the T = 8 unpooler slice sk2 at 16 rows per CTA (64 CTAs) instead of 8 (128 CTAs).

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). One line: the `slice8_r192` row of
bfp_SK2_CONFIGS goes from R 8, NT 1 to R 16, NT 2 (each consumer warp holds two n8 tiles; NWR, WSK, UPW, NACC, CL
and the ring cap unchanged). Every CTA reads all 64 KB of x by TMA after its wait, so halving the CTA count halves
the x bytes the L2 serves (8 MB -> 4 MB); the v1st stamps put the slice's post-wait time in four stage landings
0.37 us apart at about 5.4 TB/s of L2 output. Per output row the k split (WSK, UPW), the accumulator chains, the
escape-correction copies and the reduction order are those of the base, so the output is bitwise the base's.
The config is picked only for (unpooler, tokr 8), which the layer runs only at T = 8 (T = 1 runs the cgemv slice).

    python3 tools/dl_ly_v1r16_patch.py [--out muse_decode_layer/cand/ly_v1r16.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_v1r16.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
EDITS = [("    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),\n",
          "    ('slice8_r192', 'unpooler', 1024, 16, 2, 1, 8, 2, 2, 1),\n")]
HEADER = ("# ly_v1r16: ly_v1un with the T = 8 unpooler slice sk2 at 16 rows per CTA (64 CTAs, half the x bytes from L2) "
          "(tools/dl_ly_v1r16_patch.py)\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"anchor found {text.count(old)} times: {old!r}")
        text = text.replace(old, new)
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
