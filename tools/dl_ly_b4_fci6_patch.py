#!/usr/bin/env python3
"""ly_b4_fci6: the WGS consumers' deferred segment flush moves behind the stage's weight-slot release.

Base: muse_decode_layer/cand/ly_micro2.py (SHA256 0bc2bd8b..., checked). In wgs_seg the previous segment's deferred
work (wgs_flush: the phase A doneA publish, a split unit's count or designated-finisher fixup, the phase B MFIN expert
count) runs behind commit group OKC_WGSFCI of the next segment's first stage. With OKC_WGSFCI 4 that is load group 2,
before load group 3 arrives on the stage's weights-empty barrier, so while the flush waits (proxy fence, warpgroup
bar.sync, release atomics, finisher spin) the weights producer cannot reuse that slot, and at M <= 64 it keeps only two
stages in flight. The B4 timeline (docs/runs-decode-layer.md "Decode 32: B4 MoE timeline") shows the issue gap after
every segment's first stage at 2.2 to 2.7 us against 1.0 elsewhere. Commit groups 6 and 7 are load group 3, after the
release; OKC_WGSFCI 6 runs the same flush, in the same order relative to the stage's MMAs and to every other memory
operation of the warpgroup, one load group later and after the slot release. Nothing else changes: every fence,
bar.sync, release, acquire and spin of wgs_flush is kept; the ring protocol, PDL and the final pass are untouched.

  --scope all (default): MOE_CUDA_SRC's define, so B4 (decode 32) and MAIN (verify 16 / 32) both change.
  --scope b4: the define becomes #ifndef and only the B4 build gets -DOKC_WGSFCI=<fci>.

    python3 tools/dl_ly_b4_fci6_patch.py [--fci 6] [--scope all|b4] [--out muse_decode_layer/cand/ly_b4_fci6.py]
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
DEF = "#define OKC_WGSFCI 4  // st2 (E374): the deferred flush at commit group 4 (FCI 3..7 sweep best)\n"
B4 = 'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fci", type=int, default=6)
    ap.add_argument("--scope", choices=("all", "b4"), default="all")
    ap.add_argument("--out", default="muse_decode_layer/cand/ly_b4_fci6.py")
    a = ap.parse_args()
    if not 0 <= a.fci <= 7:
        sys.exit("commit groups per stage are 0..7 (OKC_WGSGS 2)")
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PIN:
        sys.exit(f"ly_micro2.py is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    whole = raw.decode()
    head, rest = whole.split(START)
    cu, tail = rest.split(END)
    if cu.count(DEF) != 1 or "#define OKC_WGSGS 2" not in cu:
        sys.exit("OKC_WGSFCI / OKC_WGSGS anchors")
    why = (f"// fci6: behind the weight-slot release of load group 3 (groups 6, 7); at 4 the flush held the slot and the\n"
           f"// producer's next issue (B4 timeline: 2.2-2.7 us issue gaps after every segment's first stage)\n")
    if a.scope == "all":
        cu = cu.replace(DEF, why + f"#define OKC_WGSFCI {a.fci}\n")
    else:
        cu = cu.replace(DEF, "#ifndef OKC_WGSFCI\n" + DEF + "#endif\n")
        if tail.count(B4) != 1:
            sys.exit("B4 define anchor")
        tail = tail.replace(B4, B4[:-2] + f', "-DOKC_WGSFCI={a.fci}")')
    out = head + START + cu + END + tail
    dst = ROOT / a.out
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
