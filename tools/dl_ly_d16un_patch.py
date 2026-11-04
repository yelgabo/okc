#!/usr/bin/env python3
"""ly_d16un: at T = 16 and 32, the unpooler sk2 launches as a PDL secondary of the cooperative MoE.

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). Target: decode 16 / 32 (docs/runs-decode-layer.md,
"Decode 16 / 32 sliding timeline (box)"). At T = 16 / 32 the unpooler is sk2 over all 8192 rows (50.4 MB of 12-bit
codes, CL = 1, P = 2 / 1 pre-wait weight stages from PCX_CFG); `_pdl8_bfp_pdl` sets bit 29 for role "un" at every
T but 8, so it launches without the programmatic attribute and starts 0.06 / 0.58 us after the MoE ends. The MID
(T = 16) and WGS B4 (T = 32) builds both run griddepcontrol.launch_dependents unconditionally at each CTA's phase B
end (helper0 source, after WS_JOIN, before the grid barrier and the final pass), so a programmatic unpooler can take
each SM as its MoE CTA retires and issue its first weight stages and escape table before its griddepcontrol.wait.

Change (one line): `_pdl8_bfp_pdl` leaves bit 29 off for role "un" at T in (8, 16, 32); T = 1 keeps it (the T = 1
slice is the cgemv path, unchanged). ly_v1un's T = 8 behaviour is the same expression's T = 8 case.
The sk2 kernel's PDL path is unchanged (the same code ly_v1un runs at T = 8): weights and the escape table
(prepare-time constants) before the wait, every thread's griddepcontrol.wait, then fence.proxy.async.global
(upstream generic stores of the MoE output -> TMA reads), then the x tiles; no global write before the wait.
griddepcontrol.wait returns only after the whole MoE grid has completed and its memory operations are visible
(PTX ISA 8.7, 9.7.13.13 griddepcontrol). The MoE stays the primary with its cooperative launch, trigger, fences and
grid barrier; all its CTAs are resident before any secondary CTA can launch. The in-prepare PDL8H guard at T = 16
(sliding) and T = 32 (full) runs with every edge on (`pdl8_edges` is None there), so it stresses the new edge.

    python3 tools/dl_ly_d16un_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_d16un.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

EDITS = [
    ('(role == "un" and T != 8)', '(role == "un" and T not in (8, 16, 32))', 1),
]
HEADER = ("# ly_d16un: ly_v1un with the T = 16 / 32 unpooler sk2 as a PDL secondary of the MoE "
          "(tools/dl_ly_d16un_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new, n in EDITS:
        if text.count(old) != n:
            sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old!r}")
        text = text.replace(old, new)
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
