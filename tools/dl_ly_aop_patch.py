#!/usr/bin/env python3
"""ly_aop: the T = 8 sliding attention prefetches the o projection's packed weights into L2 instead of the pooler's
first 4 MB.

Round 2 stamps on ly_stk1 at verify 1 sliding: o's griddepcontrol.wait returns 0.43 us after the attention's end and o
then needs 3.3 to 3.4 us to its last store, which for its 16.8 MB weight stream is DRAM rate (about 4.9 TB/s), while
DRAM carries little else between qkv's end and o's wait (the attention's 0.5 MB of K/V, the pooler prefetch, o's own
pre-wait stages). One change in the layer's attention launcher (_pdl8_decode_attention7): at one request, q_len 8 and
a sliding window, the attention kernel's existing L2 prefetch (cp.async.bulk.prefetch.L2 by thread 0 of each CTA at
kernel entry, before griddepcontrol.wait) covers o's whole packed weight stream instead of the pooler's first 4 MB.
The weights are prepare-time constants that no layer kernel writes. No CUDA source changes.

    python3 tools/dl_ly_aop_patch.py [--src muse_decode_layer/cand/ly_stk1.py] [--out muse_decode_layer/cand/ly_aop.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_aop.py"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

OLD = '''        _ps = bst["sk2"]["pooler"][1]
        prefetch = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))
    out = _launch7('''
NEW = '''        _ps = bst["sk2"]["pooler"][1]
        prefetch = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))
        if cfg["batch"] * cfg["q_len"] == 8 and cfg["batch"] == 1 and cfg["window"]:   # aop: o's weights instead
            _os = bst["sk2"][bfp__sk2_name("o", 8)][1]
            prefetch = (_os.data_ptr(), _os.numel() * _os.element_size())
    out = _launch7('''
HEADER = ("# ly_aop: {src} + the verify 1 sliding attention prefetches the o projection's packed weights into L2 "
          "instead of the pooler's first 4 MB (tools/dl_ly_aop_patch.py).\n")


def main():
    text = SRC.read_text()
    k = text.index("def _pdl8_decode_attention7(")
    m = text.index(OLD, k)
    if text.count(OLD) < 1:
        sys.exit("anchor missing")
    text = text[:m] + NEW + text[m + len(OLD):]
    text = HEADER.format(src=SRC.stem) + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
