#!/usr/bin/env python3
"""ly_atrg: the q_len 1 one-tile attention build triggers its dependents (the o projection) at kernel entry.

Base ly_v1un (88fbae86). Stamps (ly_ast, do0 diagnostic) show the o projection's pre-wait weight stages, issued at o's
entry (about 1.2 us after qkv's end, since attention triggers right after its own griddepcontrol.wait), in flight
during attention's merge: with one o stage pre-wait instead of two the attention's tickets return 0.5 us earlier,
but o then streams more after its wait and ends 0.15 us later. Triggering at attention's entry lets o launch about
1.4 us earlier, so its pre-wait stages land before attention's merge instead of during it.

Change: attn_kernel<1, false, MODE with the ONE bit> executes griddepcontrol.launch_dependents at entry (the existing
trigger after the wait stays; a second trigger is a no-op). Legal: the trigger only lets o's grid launch; o still
reads the attention output only after its own griddepcontrol.wait, which returns after this grid has completed.
Nothing is read or written earlier than before. q_len 8 and the other builds are unchanged.

    python3 tools/dl_ly_atrg_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_atrg.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_atrg.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
OLD = "  const int pair = blockIdx.z;\n  DBG(0);\n  if (p.pf_chunk && threadIdx.x == 0) {\n"
NEW = ("  const int pair = blockIdx.z;\n  DBG(0);\n"
       "  if constexpr (!FLAT && L == 1 && (MODE & 4) != 0) pdl_launch_dependents();   // (ly_atrg: o launches now; it reads\n"
       "                                                                             //  this grid's output after its own wait)\n"
       "  if (p.pf_chunk && threadIdx.x == 0) {\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    start = 'ATT7_CUDA_SRC = r"""\n'
    i = text.index(start) + len(start)
    j = text.index('\n"""\n', i)
    at = text[i:j]
    if at.count(OLD) != 1:
        sys.exit(f"anchor found {at.count(OLD)} times")
    text = ("# ly_atrg: ly_v1un with the q_len 1 one-tile attention build triggering o at its entry "
            "(tools/dl_ly_atrg_patch.py).\n" + text[:i] + at.replace(OLD, NEW) + text[j:])
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
