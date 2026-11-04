#!/usr/bin/env python3
"""ly_cg32: the decode 1 unpooler-slice cgemv launched as the 32-register kernel so all 512 CTAs fit beside the MoE.

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 37654700..., checked). Target: the MoE to unpooler-slice edge at
decode 1 (docs/runs-decode-layer.md, "Decode 1 timeline and edges"). The slice is cgemv40_kernel<128, 2>: 512 CTAs
of 160 threads at 40 registers (6,400 registers per CTA). The TINY MoE CTA (288 threads at 145 registers, about
43,800 registers) leaves 21,760 registers per SM, room for 3 slice CTAs: 396 of the 512 launch when the MoE triggers
(each CTA's phase B end) and the other 116 only as the MoE CTAs exit, so their weight loads start after the MoE ends.
At 32 registers (5,120 per CTA) 4 CTAs fit beside the MoE, 528 >= 512, and every slice CTA has its weights in
registers before the MoE ends (hypothesis; the A/B decides).

Change. The pdl8 launch edit's cgemv40_kernel launch (cg_go) becomes the file's existing cgemv_kernel (__maxnreg__ 32,
the same body) for the CPT == 2 case. No wait, trigger, fence or barrier moves; the cgemv PDL form (wait before the x
loads, trigger at the end) is unchanged. The only cgemv launch in the timed graph at T = 1 is the slice; the CG_GATE
compares its output to sk2 in prepare as before.

    python3 tools/dl_ly_cg32_patch.py [--base muse_decode_layer/cand/ly_fabq.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_cg32.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

EDITS = [
    ("cg_go(cgemv40_kernel<NTH, CPT>, half_n", "cg_go(cgemv_kernel<NTH, CPT>, half_n", 1),
]
HEADER = ("# ly_cg32: ly_fabq with the unpooler-slice cgemv (CPT 2) launched as the 32-register cgemv_kernel so all 512 "
          "CTAs are resident beside the MoE (tools/dl_ly_cg32_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new, n in EDITS:
        if text.count(old) != n:
            sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old!r}")
        text = text.replace(old, new)
    if text.count("cg_go(cgemv_kernel<NTH, CPT>, half_n") != 2:
        sys.exit("expected both cg_go branches to launch cgemv_kernel after the edit")
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
