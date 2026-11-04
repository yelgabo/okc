#!/usr/bin/env python3
"""ly_atre: the q_len 8 one-tile attention (verify 1 sliding) triggers the o projection at kernel entry, before its own
griddepcontrol.wait, instead of right after it.

Round 2 stamps: o's CTAs enter 1.13 us after qkv's end, two per SM on the attention's 33 SMs too, and their pre-wait
weight stages overlap the attention's q pass; moving the trigger to the merge (ly_atrg) took 0.23 us off the q pass but
put o's arrival into the merge tail (+0.75 us at the attention's end). This variant moves it the other way, so o's
launch and pre-wait loads fall into the attention's own pre-wait idle time. One change in ATT7_CUDA_SRC: at kernel entry
the one-tile q_len 8 builds (MODE & 4, per-request) execute griddepcontrol.launch_dependents; the post-wait trigger
stays (a second launch_dependents has no effect). o still waits with griddepcontrol.wait for the whole attention grid
before reading its output, so no ordering changes.

    python3 tools/dl_ly_atre_patch.py [--src muse_decode_layer/cand/ly_stk1.py] [--out muse_decode_layer/cand/ly_atre.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_atre.py"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

OLD = "  const int pair = blockIdx.z;\n  DBG(0);\n  if (p.pf_chunk && threadIdx.x == 0) {\n"
NEW = ("  const int pair = blockIdx.z;\n  DBG(0);\n"
       "  if constexpr (OKC_TRIGE8 && L == 8 && !FLAT && (MODE & 4) != 0) pdl_launch_dependents();\n"
       "  if (p.pf_chunk && threadIdx.x == 0) {\n")
DEF_OLD = "#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n"
DEF_NEW = DEF_OLD + "#define OKC_TRIGE8 1    // one-tile q_len 8 builds: trigger the dependent grid at kernel entry\n"


def main():
    text = SRC.read_text()
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    for old, new in ((OLD, NEW), (DEF_OLD, DEF_NEW)):
        if blk.count(old) != 1:
            sys.exit(f"anchor found {blk.count(old)} times: {old[:80]!r}")
        blk = blk.replace(old, new)
    text = (f"# ly_atre: {SRC.stem} + the one-tile q_len 8 attention triggers its dependents at kernel entry "
            "(tools/dl_ly_atre_patch.py).\n") + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
