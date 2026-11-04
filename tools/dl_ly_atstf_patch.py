#!/usr/bin/env python3
"""ly_atst + two more attention stamps around the q_len 8 prologue's fence.proxy.async (instrument only):
slot 21 right before the fence after the q pass (e4m3 q tile), slot 22 right before the fence after the new-row patch.

    python3 tools/dl_ly_atstf_patch.py --src muse_decode_layer/cand/ly_atst_s1.py --out muse_decode_layer/cand/ly_atstf_s1.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
EDITS = [
    ("  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile",
     "  DBG(21);\n  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile"),
    ("    if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma\n",
     "    DBG(22);\n    if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma\n"),
]


def main():
    text = SRC.read_text()
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    for old, new in EDITS:
        if blk.count(old) != 1:
            sys.exit(f"anchor found {blk.count(old)} times: {old[:80]!r}")
        blk = blk.replace(old, new)
    text = "# ly_atstf: + stamps 21 / 22 before the q_len 8 prologue fences (tools/dl_ly_atstf_patch.py)\n" + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
