#!/usr/bin/env python3
"""ly_b4_dfa: phase A split units finished by their designated share (OKC_WGSDFIN 2 -> 3), as phase B already is.

Base: muse_decode_layer/cand/ly_micro2.py (SHA256 0bc2bd8b..., checked). A phase A unit cut between two CTAs is
finished by the share that arrives last: each share's deferred wgs_flush takes an atom.acq_rel.gpu on the unit's fixup
counter (MEMBAR.ALL.GPU, then an ATOMG whose returned value the warpgroup waits for), then a bar.sync, and the last one
reads the partials back. The B4 timeline (docs/runs-decode-layer.md "Decode 32: B4 MoE timeline") shows every
segment's first stage, where that flush runs, holding the weights ring 1.5 us longer than any other stage. With bit 0
of OKC_WGSDFIN set, the share holding the unit's stage 15 is the designated finisher (under OKC_WGSREORD each CTA runs
its trailing cut share first, so that share is the later one by construction): the other share counts with a
fire-and-forget red.release.gpu (no returned value to wait for), and the finisher spins with ld.acquire until the
count is complete, then reads back as before. This is wgs_flush's existing phase B path (kind 2 / kind 3), used here
for phase A as well; no fence, barrier, acquire or release is removed, and the ring, MFIN counters, PDL and the final
pass are untouched.

  --scope all (default): MOE_CUDA_SRC's define, so B4 (decode 32) and MAIN (verify 16 / 32) both change.
  --scope b4: the define becomes #ifndef and only the B4 build gets -DOKC_WGSDFIN=3.

    python3 tools/dl_ly_b4_dfa_patch.py [--scope all|b4] [--out muse_decode_layer/cand/ly_b4_dfa.py]
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_micro2.py"
PIN = "0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd"
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"
DEF = ("#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): "
       "bit0 phase A, bit1 phase B\n")
B4 = 'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("all", "b4"), default="all")
    ap.add_argument("--out", default="muse_decode_layer/cand/ly_b4_dfa.py")
    a = ap.parse_args()
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PIN:
        sys.exit(f"ly_micro2.py is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    head, rest = raw.decode().split(START)
    cu, tail = rest.split(END)
    if cu.count(DEF) != 1:
        sys.exit("OKC_WGSDFIN anchor")
    if a.scope == "all":
        cu = cu.replace(DEF, DEF.replace("OKC_WGSDFIN 2 ", "OKC_WGSDFIN 3 "))
    else:
        cu = cu.replace(DEF, "#ifndef OKC_WGSDFIN\n" + DEF + "#endif\n")
        if tail.count(B4) != 1:
            sys.exit("B4 define anchor")
        tail = tail.replace(B4, B4[:-2] + ', "-DOKC_WGSDFIN=3")')
    out = head + START + cu + END + tail
    dst = ROOT / a.out
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
