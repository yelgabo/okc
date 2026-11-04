#!/usr/bin/env python3
"""ly_xca: the decode 1 unpooler-slice cgemv reads x (the MoE output) through L1 (ld.global.ca) after its wait.

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 37654700..., checked). Target: the post-MoE tail at decode 1
(docs/runs-decode-layer.md, "Decode 1 post-MoE tail"): after the slice's griddepcontrol.wait returns, 512 CTAs x 128
threads read the same 8 KB row of x with ld.global.cg (L2 only), 32k line requests on 64 lines in the same
microsecond; the stamps put the x tail (every lane of a warp landed) and the math at about 2 us of the 3.7 us from
the MoE's exit to the slice's end. With ld.global.ca the SM's L1 merges the requests of its 3 to 4 resident CTAs
(16 warps) into one fill per line, so L2 sees about 4x fewer requests (hypothesis; the A/B decides).

Change: in the pdl8 cgemv edits (applied to CG_CUDA_SRC at import) the two x loads of the compute threads and the
escape warp's x element load become __ldca; they stay after griddepcontrol.wait, which makes the MoE grid's writes
visible to this grid (PTX ISA, griddepcontrol: "all the memory operations from the prerequisite grids are performed
and made visible to the current grid"), and no thread of this kernel touches x before the wait, so no line of x can
sit in L1 before the first read. Trigger, launch attribute, fences, barriers and the PDL set per T are unchanged; the
weights keep their L1::no_allocate stream loads. The CPT == 1 cgemv kernels (not launched at T = 1 in the layer)
get the same code.

    python3 tools/dl_ly_xca_patch.py [--base muse_decode_layer/cand/ly_fabq.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_xca.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

EDITS = [
    ("xa[i] = __ldcg(xp + i * NTH * 2);\\n            xb[i] = __ldcg(xp + i * NTH * 2 + 1);",
     "xa[i] = __ldca(xp + i * NTH * 2);\\n            xb[i] = __ldca(xp + i * NTH * 2 + 1);", 1),
    ("(unsigned)__ldcg(reinterpret_cast<const unsigned short*>(x) + kk) << 16);",
     "(unsigned)__ldca(reinterpret_cast<const unsigned short*>(x) + kk) << 16);", 1),
]
HEADER = ("# ly_xca: ly_fabq with the unpooler-slice cgemv's x loads (after its griddepcontrol.wait) through L1, "
          "ld.global.ca instead of .cg (tools/dl_ly_xca_patch.py).\n")


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
