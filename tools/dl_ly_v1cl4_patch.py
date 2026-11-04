#!/usr/bin/env python3
"""ly_v1cl4: ly_v1un with the T = 8 unpooler slice sk2 split over K across a cluster of 4 CTAs (32 rows per cluster).

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). One line: the `slice8_r192` row of
bfp_SK2_CONFIGS goes from (R 8, NT 1, CL 1) to (R 32, NT 4, CL 4): 32 clusters of 4 CTAs (128 CTAs, as before), each
CTA one 1024-k stage of 32 rows; ranks 1 to 3 hand their per-warp sums to rank 0 by st.async (the file's existing
CL > 1 path, used by the router's CL 3 build). The v1st stamps put the base slice's post-wait time in a per-SM
ingest of about 55 GB/s over 64 KB of x and 24 KB of weights; here each CTA's whole 48 KB of weights is its one
pre-wait stage (P = 1 for CL > 1) and after the wait each CTA reads 16 KB of x. WSK, UPW and NACC are unchanged, so
every warp's k units are the base's, but each row's sum is now four rank partials added at rank 0, not one chain:
the output is within FP32 rounding of the base, not bitwise. With one stage per rank all escape corrections sit in
escape copy 0, the one copy the CL > 1 rank > 0 path folds. Picked only for (unpooler, tokr 8): the layer runs it
only at T = 8 (T = 1 runs the cgemv slice).

    python3 tools/dl_ly_v1cl4_patch.py [--out muse_decode_layer/cand/ly_v1cl4.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_v1cl4.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
EDITS = [("    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),\n",
          "    ('slice8_r192', 'unpooler', 1024, 32, 4, 1, 8, 2, 2, 4),\n")]
HEADER = ("# ly_v1cl4: ly_v1un with the T = 8 unpooler slice sk2 split over K across clusters of 4 CTAs, 32 rows per "
          "cluster (tools/dl_ly_v1cl4_patch.py)\n")


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
