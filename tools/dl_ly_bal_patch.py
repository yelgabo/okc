#!/usr/bin/env python3
"""ly_bal: the MAIN WGS build's stream-K cost split charges each unit's switch by its n-tile count.

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 3765470032..., checked; --base / --pin for a ly_fin1 base, whose MAIN
define tuple is extended). Target: the phase B end spread at verify 16 / 32 (docs/runs-decode-layer.md, "Verify tail
(box)"): the CTAs that end last hold the widest jobs (64-token, 8 n-tiles) and the fewest stages. The split charges a
constant per unit (OKC_WGSU 160 phase A, OKC_WGSUB 140 phase B, in units of the per-stage cost 112 / 100 of an
8-token stage) while the measured first-of-segment stall grows with the n-tiles: phase A about 1.0 us + 0.3 us per
n-tile, phase B 0.8 us + 0.3 us per n-tile (the weights producer's issue gap after a segment's first stage, verify 16 /
32 timeline), against 0.96 us per 8-token stage.

Change (MAIN build only, -DOKC_BAL=1; every other build compiles the base's text): wgs_unitc takes the job's token
count and returns OKC_WGSU0 + OKC_WGSUN x n-tiles (phase A: 90 + 40 x ntt; phase B: 55 + 31 x ntt), used wherever
the prefix, the stage-at search and the CTA-of search add the unit cost; the per-stage tables, the job list, the
segment walk, the fixup path and every fence and barrier are unchanged (the ranges move, the code that runs them does
not). The ranges are computed by the x-producer warp before the stream starts; the consumers' stage loops are not
touched.

    python3 tools/dl_ly_bal_patch.py [--base F --pin SHA] [--out muse_decode_layer/cand/ly_bal.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
DST = ROOT / "muse_decode_layer/cand/ly_bal.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"
MAINS = [('key, defines = "main", _main_defines(sfx)',
          'key, defines = "main", tuple(_main_defines(sfx)) + ("-DOKC_BAL=1",)'),
         ('key, defines = "main", tuple(_main_defines(sfx)) + ("-DOKC_FIN1=1",)',
          'key, defines = "main", tuple(_main_defines(sfx)) + ("-DOKC_FIN1=1", "-DOKC_BAL=1",)')]

UC = "__device__ __forceinline__ int wgs_unitc(int pb) { return pb ? OKC_WGSUB : OKC_WGSU; }\n"
UC_F = ("#if OKC_BAL\n"
        "// bal: the segment switch stalls the consumers longer on wider units (measured issue gaps, verify 16 / 32)\n"
        "#define OKC_WGSU0 90\n"
        "#define OKC_WGSUN 40\n"
        "#define OKC_WGSUB0 55\n"
        "#define OKC_WGSUNB 31\n"
        "__device__ __forceinline__ int wgs_unitc(int pb, int cnt) {\n"
        "  const int ntt = (cnt + 7) >> 3;\n"
        "  return pb ? OKC_WGSUB0 + OKC_WGSUNB * ntt : OKC_WGSU0 + OKC_WGSUN * ntt;\n"
        "}\n"
        "#else\n"
        "__device__ __forceinline__ int wgs_unitc(int pb, int) { return pb ? OKC_WGSUB : OKC_WGSU; }\n"
        "#endif\n")
EDITS = [
    (UC, UC_F),
    ("    acc += ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;  // cepi: 32 units x (16 stages + unit cost)\n"
     "    accB += ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;\n",
     "    acc += ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + (uint32_t)wgs_unitc(0, S.jC[j])) << 5;  // cepi: 32 units x (16 stages + unit cost)\n"
     "    accB += ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + (uint32_t)wgs_unitc(1, S.jC[j])) << 4;\n"),
    ("  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = (uint32_t)wgs_unitc(pb), cu = 16u * cs + U;\n",
     "  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = (uint32_t)wgs_unitc(pb, S.jC[lo]), cu = 16u * cs + U;\n"),
    ("      ia = ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;\n"
     "      ib = ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;\n",
     "      ia = ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + (uint32_t)wgs_unitc(0, S.jC[j])) << 5;\n"
     "      ib = ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + (uint32_t)wgs_unitc(1, S.jC[j])) << 4;\n"),
    ("  const uint32_t sj = (uint32_t)(s & ((1 << (9 - pb)) - 1)), cs = (uint32_t)wgs_cost(pb, S.jC[j]), U = (uint32_t)wgs_unitc(pb);\n",
     "  const uint32_t sj = (uint32_t)(s & ((1 << (9 - pb)) - 1)), cs = (uint32_t)wgs_cost(pb, S.jC[j]), U = (uint32_t)wgs_unitc(pb, S.jC[j]);\n"),
]


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    whole = raw.decode()
    assert whole.count(START) == 1 and whole.count(END) == 1
    head, rest = whole.split(START)
    cu, tail = rest.split(END)
    for old, new in EDITS:
        n = cu.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:100]!r}")
        cu = cu.replace(old, new)
    hit = [(o, n) for o, n in MAINS if tail.count(o) == 1]
    if len(hit) != 1:
        sys.exit("MAIN define anchor")
    tail = tail.replace(*hit[0])
    out = head + START + cu + END + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
