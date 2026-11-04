#!/usr/bin/env python3
"""ly_v1trg: the q_len 8 one-tile attention (verify 1 sliding, attn_kernel<8, false, 15>) triggers its dependents (the
o projection) at its merge instead of right after its own griddepcontrol.wait, on ly_stk1.

The pdl8 edit list already carries both trigger points behind OKC_PDLTRIG (0: after the wait, 1: at each segment's
merge, after its K/V tiles are consumed). One change: the macro becomes (L == 8 && ONE), so only the one-tile q_len 8
builds move the trigger; every other instance keeps the trigger after the wait. No wait, fence or memory operation
moves; the o kernel still waits with griddepcontrol.wait for the whole attention grid before reading its output.

    python3 tools/dl_ly_v1trg_patch.py [--src muse_decode_layer/cand/ly_stk1.py] [--out muse_decode_layer/cand/ly_v1trg.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
DST = ROOT / "muse_decode_layer/cand/ly_v1trg.py"
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

OLD = "#define OKC_PDLTRIG 0   // pdl8: trigger at each segment's merge"
NEW = "#define OKC_PDLTRIG (L == 8 && ONE)   // pdl8: trigger at each segment's merge"
HEADER = ("# ly_v1trg: ly_stk1 + the q_len 8 one-tile attention triggers the o projection at its merge instead of after "
          "its own griddepcontrol.wait (tools/dl_ly_v1trg_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    if text.count(OLD) != 1:
        sys.exit(f"anchor found {text.count(OLD)} times")
    text = HEADER + text.replace(OLD, NEW)
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
