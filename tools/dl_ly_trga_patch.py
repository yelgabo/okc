#!/usr/bin/env python3
"""ly_trga: at decode 1 the TINY MoE build triggers its dependents at each CTA's phase A end instead of its phase B end.

Base: muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d..., checked). Target: the slice launch at decode 1
(docs/runs-decode-layer.md, "Decode 1 post-MoE tail" and "Decode 1 finish and slice launch"): the slice cgemv enters
1.3 to 1.6 us after the last MoE CTA's launch_dependents, which sits at the phase B end, and at decode 1 full that is
0.03 us before the last MoE CTA exits, so its 1.6 us code stream (6.3 MB of prepare-time constants) runs after the
MoE has ended and the wait returns 1.6 us after the exit. Triggering at the phase A end lets the resident slice CTAs
(3 per SM beside the MoE CTA) issue their code loads during the MoE's handoff and phase B, so the codes are in
registers when the MoE ends (hypothesis; the A/B decides).

Change, in the helper0 MoE source only (the TINY / TINY2 / MID builds; the MAIN source is not touched): the
compute threads' griddepcontrol.launch_dependents moves from after the phase B join to right after
run_phase<false> (DBG(2)) when OKC_TRIGA is set, and OKC_TRIGA=1 is added to MOE_TINY_DEFINES only (T <= 4), so the
TINY2 and MID builds preprocess to the base's code. launch_dependents orders nothing: the slice's
griddepcontrol.wait still waits until the whole MoE grid has completed and its writes are visible (PTX ISA,
griddepcontrol), and the slice reads nothing the MoE writes before that wait (its pre-wait loads are the codes, the
escape table and its count, all written at prepare). No fence, barrier, wait or launch attribute changes.

    python3 tools/dl_ly_trga_patch.py [--base muse_decode_layer/cand/ly_rfx.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_rfx.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_trga.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

HELPER_START = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\""
HELPER_EDITS = [
    ('  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n',
     '  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n'
     '#if OKC_TRIGA\n'
     '  // trga: the dependents may launch now; their griddepcontrol.wait still waits for this whole grid\n'
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n'
     '#endif\n'
     '  run_phase<true, BIG>(p, S, xbuf);\n', 1),
    ('  // waits for this whole grid to finish and its writes to be visible)\n'
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n',
     '  // waits for this whole grid to finish and its writes to be visible)\n'
     '#if !OKC_TRIGA\n'
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n'
     '#endif\n', 1),
]
FILE_EDITS = [
    ('MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n',
     'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n'
     'MOE_TINY_DEFINES += ("-DOKC_TRIGA=1",)  # trga: T <= 4 triggers the slice at the phase A end\n', 1),
]
HEADER = ("# ly_trga: ly_rfx with the TINY MoE build (T <= 4) triggering its dependents at each CTA's phase A end "
          "instead of its phase B end (tools/dl_ly_trga_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    a = text.index(HELPER_START)
    b = text.index('"""', a + len(HELPER_START))
    blk = text[a:b]
    for old, new, n in HELPER_EDITS:
        if blk.count(old) != n:
            sys.exit(f"helper0 anchor found {blk.count(old)} times, expected {n}: {old!r}")
        blk = blk.replace(old, new)
    text = text[:a] + blk + text[b:]
    for old, new, n in FILE_EDITS:
        if text.count(old) != n:
            sys.exit(f"file anchor found {text.count(old)} times, expected {n}: {old!r}")
        text = text.replace(old, new)
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
