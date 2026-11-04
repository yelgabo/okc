#!/usr/bin/env python3
"""ly_aqk: ly_aqd + the new k,v pass of the q_len 8 one-tile attention in the same straight-line block as the q pass.

Round 2 stamps on ly_aqd: the split holding the block's new rows reaches its ticket 0.58 us after the median split and
the grid barrier waits for it; its extra work is the new k,v pass (+0.38 us), the patch (+0.18) and the masked tile.
The pass sits after the q pass's fence.proxy.async (a memory clobber) and behind `if (has_new)`, so it cannot overlap
the q pass. One change, behind constexpr QK = OKC_QKV1T && ONE && QDIR && L == 8: the q pass's fence moves to after the
new k,v pass (still before the CTA barrier that precedes the first wgmma), and every CTA runs the pass's arithmetic,
the shared and global stores staying behind `live && has_new` (a CTA without new rows reads its unused staging and
stores nothing). The cache-row writes, the patch, the wait, the trigger and the merge are unchanged.

    python3 tools/dl_ly_aqk_patch.py [--src muse_decode_layer/cand/ly_aqd.py] [--out muse_decode_layer/cand/ly_aqk.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_aqd.py"
DST = ROOT / "muse_decode_layer/cand/ly_aqk.py"
PIN = "df16832ff07578e2b40131a9975113d9720b16254d9db35a2531cb5762e7e5c9"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

EDITS = [
    ("#define OKC_QDIRECT1T 1 //",
     "#define OKC_QKV1T 1     // one-tile q_len 8 QDIRECT builds: the new k,v pass straight after the q pass, one fence after both\n"
     "#define OKC_QDIRECT1T 1 //"),
    ("  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n",
     "  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n"
     "  constexpr bool QK = OKC_QKV1T && ONE && QDIR && L == 8;\n"),
    ("  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's "
     "first barrier\n  DBG(10);\n\n  if (has_new) {\n",
     "  if constexpr (QDIR && !QK) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the "
     "loop's first barrier\n  DBG(10);\n\n  if (QK || has_new) {\n"),
    ("      if (live) {\n        uint32_t w0 = 0, w1 = 0;\n",
     "      if (live && (!QK || has_new)) {\n        uint32_t w0 = 0, w1 = 0;\n"),
    ("  csync<WS, NT>();\n  DBG(11);\n",
     "  if constexpr (QK) fence_proxy_async();   // (QK) the e4m3 q tile, after the new k,v pass\n"
     "  csync<WS, NT>();\n  DBG(11);\n"),
]
HEADER = ("# ly_aqk: ly_aqd + the one-tile q_len 8 attention's new k,v pass in the q pass's straight-line block, one "
          "proxy fence after both (tools/dl_ly_aqk_patch.py).\n")


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
