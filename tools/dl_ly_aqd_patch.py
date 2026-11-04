#!/usr/bin/env python3
"""ly_aqd: the q_len 8 one-tile attention (verify 1 sliding, attn_kernel<8, false, 15>) builds its e4m3 query tile
directly in the q pass (the kernel's own OKC_QDIRECT + OKC_QP16 path) instead of the f16 staging pass plus the
consumers' f16 -> e4m3 conversion, on ly_stk1.

Round 2 stamps: the f16 q pass costs 0.85 to 1.0 us of thread 0's time after the wait whether or not its code is warm
(ly_aqw: the pass called dry before the wait, then twice after it: every call 0.83 to 0.86 us), and the consumers then
re-read the staging and convert it (part of the 0.37 us consumer setup). The layer builds ATT7 with -DOKC_QDIRECT=0,
which turns the direct path off for every q_len 8 instance. One change: QDIR = (OKC_QDIRECT || (OKC_QDIRECT1T && ONE))
&& (the base's conditions), so only the one-tile q_len 8 builds (MODE 13 / 15) take the direct path: two passes of four
threads per row, one fp32 -> e4m3 rounding (the base rounds fp32 -> f16 -> e4m3), the e4m3 tile written with st.shared
and published to the wgmma with fence.proxy.async (the kernel's own text). The wait, the trigger, the merge and every
other instance are unchanged.

    python3 tools/dl_ly_aqd_patch.py [--src muse_decode_layer/cand/ly_stk1.py] [--out muse_decode_layer/cand/ly_aqd.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_aqd.py"
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

EDITS = [
    ("#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n",
     "#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n"
     "#define OKC_QDIRECT1T 1 // the one-tile q_len 8 builds take the QDIRECT q pass even when OKC_QDIRECT is 0\n"),
    ("  constexpr bool QDIR = OKC_QDIRECT && L == 8 &&",
     "  constexpr bool QDIR = (OKC_QDIRECT || (OKC_QDIRECT1T && ONE)) && L == 8 &&"),
]
HEADER = ("# ly_aqd: ly_stk1 + the q_len 8 one-tile attention builds its e4m3 query tile directly (QDIRECT q pass) "
          "(tools/dl_ly_aqd_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    for old, new in EDITS:
        if blk.count(old) != 1:
            sys.exit(f"anchor found {blk.count(old)} times: {old[:80]!r}")
        blk = blk.replace(old, new)
    text = HEADER + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
