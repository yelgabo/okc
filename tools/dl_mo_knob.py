#!/usr/bin/env python3
"""One fused-MoE knob at a time: writes muse_decode_layer/cand/mo_<knob>.py from g1b4_pdl8h.py (33c2353b).

Each knob changes one setting of the layer's own MoE builds and nothing else: the token-count thresholds that pick
the TINY / TINY2 / MID / B4 / MAIN build (_fork_native_moe), a build's defines, or a compile-time constant of the
glue_moe sources (ring slots, WGS x ring, gather batch, flush commit group, per-unit cost of phase B). Builds the
layer uses at the swept cells (pdl8h): verify 1 (M = 8) TINY2, decode 16 (M = 16) MID, decode 32 (M = 32) B4,
verify 16 (M = 128) MAIN. Several knobs combine into one file with `--combo name1,name2,...` (written as mo_combo.py).

    python3 tools/dl_mo_knob.py <knob> [<knob> ...]
    python3 tools/dl_mo_knob.py --combo k1,k2 [--name combo]
    python3 tools/dl_mo_knob.py --list
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/g1b4_pdl8h.py"
PIN = "33c2353b52de29c5af341609de08a7e6f021e5395fc4ef633706978fecedf7cd"

B4_COND = "16 < M <= 32 and tuple(_main_defines(sfx)) == tuple(MOE_DEFINES)"
MID_TAIL = "'-DOKC_WSBLK=2', '-DOKC_FINPF=1')"
TINY2_TAIL = "'-DOKC_WSBLK=1', '-DOKC_FINPF=1')"
NSF = '"-DOKC_NSF={3 if major >= 9 else 1}"'
# prepare insists on exactly the five builds; a pick knob can leave one unused, so the sweep file accepts a subset
KEYS = ("    if set(st['ma_ext']) != {'tiny','tiny2','mid','b4','main'}:",
        "    if not set(st['ma_ext']) <= {'tiny','tiny2','mid','b4','main'}:", 1)

# name: (cells it can move, [(old, new, count)])
KNOBS = {
    # build picks
    "tiny2m16": ("decode 16 MID -> TINY2", [("MOE_TINY2_MAX_M = 8\n", "MOE_TINY2_MAX_M = 16\n", 1), KEYS]),
    "tiny2m4": ("verify 1 TINY2 -> MID", [("MOE_TINY2_MAX_M = 8\n", "MOE_TINY2_MAX_M = 4\n", 1), KEYS]),
    "tiny8": ("verify 1 TINY2 -> TINY", [("MOE_TINY_MAX_M = 4\n", "MOE_TINY_MAX_M = 8\n", 1), KEYS]),
    "mid8": ("decode 16 MID -> MAIN", [("MOE_MID_MAX_M = 16\n", "MOE_MID_MAX_M = 8\n", 1), KEYS]),
    "mid32": ("decode 32 B4 -> MID", [("MOE_MID_MAX_M = 16\n", "MOE_MID_MAX_M = 32\n", 1), KEYS]),
    "nob4": ("decode 32 B4 -> MAIN", [(B4_COND, B4_COND.replace("M <= 32", "M <= 16"), 1), KEYS]),
    # build defines
    "midskt16": ("decode 16 MID skinny jobs up to 16 tokens", [("'-DOKC_SPLIT=1', ", "'-DOKC_SPLIT=1', '-DOKC_SKT=16', ", 1)]),
    "midwsblk1": ("decode 16 MID WSBLK 2 -> 1", [(MID_TAIL, MID_TAIL.replace("WSBLK=2", "WSBLK=1"), 1)]),
    "midfinpf0": ("decode 16 MID FINPF 1 -> 0", [(MID_TAIL, MID_TAIL.replace("FINPF=1", "FINPF=0"), 1)]),
    "tiny2wsblk2": ("verify 1 TINY2 WSBLK 1 -> 2", [(TINY2_TAIL, TINY2_TAIL.replace("WSBLK=1", "WSBLK=2"), 1)]),
    # compile-time constants of every build
    "ns2": ("all builds: unit ring slots 3 -> 2", [("_MA_MAIN_NS = 3\n", "_MA_MAIN_NS = 2\n", 1)]),
    "ns4": ("all builds: unit ring slots 3 -> 4", [("_MA_MAIN_NS = 3\n", "_MA_MAIN_NS = 4\n", 1)]),
    "nsf4": ("all builds: fat-unit slots 3 -> 4", [(NSF, NSF.replace("{3 if", "{4 if"), 1)]),
    # WGS (B4 / MAIN) constants
    "wgsnx4": ("B4 / MAIN x ring slots 3 -> 4", [("#define OKC_WGSNX 3 ", "#define OKC_WGSNX 4 ", 2)]),
    "gathb16": ("B4 / MAIN x rows per gather batch 8 -> 16", [("#define OKC_GATHB 8 ", "#define OKC_GATHB 16 ", 2)]),
    "wgsfci3": ("B4 / MAIN deferred flush at commit group 4 -> 3", [("#define OKC_WGSFCI 4", "#define OKC_WGSFCI 3", 1)]),
    "wgsfci5": ("B4 / MAIN deferred flush at commit group 4 -> 5", [("#define OKC_WGSFCI 4", "#define OKC_WGSFCI 5", 1)]),
    "wgsub120": ("B4 / MAIN phase B unit cost 140 -> 120 (A/B balance)", [("#define OKC_WGSUB 140", "#define OKC_WGSUB 120", 1)]),
    "wgsub160": ("B4 / MAIN phase B unit cost 140 -> 160 (A/B balance)", [("#define OKC_WGSUB 140", "#define OKC_WGSUB 160", 1)]),
    # old-line MoE mode 51 (final pass forms each route's sum of squares, the phase-B epilogue writes y only) is
    # pdl8h's WGS default (OKC_WGSFINSS 1); this switches it off (the mode-47 path) to measure what it is worth
    "wgsfinss0": ("B4 / MAIN: phase-B epilogue sum-of-squares atomics instead of the final pass (old mode 47)",
                  [("#define OKC_WGSFINSS 1\n#ifndef OKC_MFIN", "#define OKC_WGSFINSS 0\n#ifndef OKC_MFIN", 1)]),
    "wgsu140": ("B4 / MAIN phase A unit cost 160 -> 140 (A/B balance)", [("#define OKC_WGSU 160", "#define OKC_WGSU 140", 1)]),
}


def build(names, out):
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("g1b4_pdl8h.py is not 33c2353b")
    s = raw.decode()
    for n in names:
        for old, new, cnt in KNOBS[n][1]:
            if s.count(old) != cnt:
                sys.exit(f"{n}: anchor {old[:60]!r} found {s.count(old)} times, {cnt} expected")
            s = s.replace(old, new)
    out.write_text(s)
    py_compile.compile(str(out), doraise=True)
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest(), "+".join(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("knobs", nargs="*")
    ap.add_argument("--combo", default="")
    ap.add_argument("--name", default="combo")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (what, _) in KNOBS.items():
            print(f"{k:12s} {what}")
        return
    cand = ROOT / "muse_decode_layer/cand"
    for k in a.knobs:
        build([k], cand / f"mo_{k}.py")
    if a.combo:
        build(a.combo.split(","), cand / f"mo_{a.name}.py")


if __name__ == "__main__":
    main()
