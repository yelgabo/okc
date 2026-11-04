#!/usr/bin/env python3
"""ly_v1un: at T = 8 only, the unpooler-slice sk2 launches as a PDL secondary of the cooperative MoE.

Base: muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d..., checked). Target: verify 1 (docs/runs-decode-layer.md,
"Verify 1 timeline (box)"). At T = 8 the slice is sk2_kernel<1, 8, 1, 1, 8, 2, 4, ...> (128 CTAs, 4 stages of
12 KB weights each, the whole K); `_pdl8_bfp_pdl` sets bit 29 for role "un" at every T, so it launches without the
programmatic attribute and starts 0.2 us after the MoE ends, then streams its weights from DRAM. Every other edge
of the T = 8 chain is already programmatic. TINY2 (helper0) runs griddepcontrol.launch_dependents unconditionally at
each CTA's phase B end, before its grid barrier and final pass, so a programmatic slice can take each SM as its MoE
CTA retires (the two CTAs cannot share an SM: >= 115,712 + 118,240 B of shared memory) and issue its first P = 2
weight stages and the escape table before its griddepcontrol.wait.

Change (two lines, T = 8 only):
  1. PDL8D_EDGES gains "un", so the T = 8 in-prepare guard (_pdl8h_t8: paired A/B and the 2000-replay stress
     against the PDL-off output, a FAIL turns T = 8 off) covers the new edge with the other six.
  2. `_pdl8_bfp_pdl` leaves bit 29 off for role "un" when T == 8 and the edge is chosen; T = 1, 16, 32 keep it.
The sk2 kernel's own PDL path is unchanged: weights and the escape table (prepare-time constants) before the wait,
then every thread's griddepcontrol.wait, then fence.proxy.async.global (upstream generic stores of x -> TMA reads),
then the x tiles; no global write before the wait. griddepcontrol.wait returns only after the whole MoE grid has
completed and its memory operations are performed and visible to this grid (PTX ISA 8.7, 9.7.13.13 griddepcontrol).
The MoE keeps its launch, trigger, fences and grid barrier; it is the primary, never a secondary, so its
co-residency guarantee is untouched (the secondary's CTAs only take SMs its CTAs have left). The finish stays a PDL
secondary of the slice, as in the base.

    python3 tools/dl_ly_v1un_patch.py [--base muse_decode_layer/cand/ly_rfx.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_rfx.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_v1un.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

EDITS = [
    ('PDL8D_EDGES = ("attn", "o", "b1", "pool", "route", "fin")\n',
     'PDL8D_EDGES = ("attn", "o", "b1", "pool", "route", "fin", "un")\n', 1),
    ('        bfp_SK2_PDL = _pcx_val(role, T) | ((1 << 29) if role in ("qkv", "un") or not _pdl8_edge(role) else 0)\n',
     '        bfp_SK2_PDL = _pcx_val(role, T) | ((1 << 29) if role == "qkv" or (role == "un" and T != 8) '
     'or not _pdl8_edge(role) else 0)\n', 1),
]
HEADER = ("# ly_v1un: ly_rfx with the T = 8 unpooler-slice sk2 as a PDL secondary of the MoE, edge in the T = 8 guard "
          "(tools/dl_ly_v1un_patch.py).\n")


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
