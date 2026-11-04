#!/usr/bin/env python3
"""ly_fin1: the MAIN WGS build's final pass reads each y row once (pass 2 from a shared-memory copy of pass 1's loads).

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 3765470032..., checked; --base / --pin for another). Target: the
tail of okc::moe_fused<true> at verify 16 / 32 (docs/runs-decode-layer.md, "Verify tail (box)"). The WGSFINSS final
pass (one token per item, 512 threads x 8 columns) loads every local route's y row twice: pass 1 for the row's sum of
squares, then, after two CTA barriers, pass 2 for the weighted sum. The second load is a dependent global round trip
per token on every CTA at once.

Change (MAIN build only, -DOKC_FIN1=1; every other build compiles the base's text): pass 1 also stores the two float4
it loaded per route into a shared-memory row cache behind the `red` partials (8 routes x 16 KB in the free weight
ring), and pass 2 reads them back from shared memory. Same values, same FP32 adds in the same order, so the output is
bitwise the base's. The MFIN wait, both CTA barriers, every fence and the consumers' stage loops are unchanged.

    python3 tools/dl_ly_fin1_patch.py [--base F --pin SHA] [--out muse_decode_layer/cand/ly_fin1.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
DST = ROOT / "muse_decode_layer/cand/ly_fin1.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
START = "MOE_CUDA_SRC = r'''\n"
END = "\n'''\nMOE_CPP_SRC"
MAIN = 'key, defines = "main", _main_defines(sfx)'
MAIN_F = 'key, defines = "main", tuple(_main_defines(sfx)) + ("-DOKC_FIN1=1",)'

RED = ("    static_assert(HD == NT * 8, \"8 columns per thread\");\n"
       "    float* red = reinterpret_cast<float*>(xbuf);\n")
RED_F = ("    static_assert(HD == NT * 8, \"8 columns per thread\");\n"
         "    float* red = reinterpret_cast<float*>(xbuf);\n"
         "#if OKC_FIN1\n"
         "    // fin1: pass 1's row loads kept for pass 2 (8 routes x 16 KB behind red, inside the free weight ring)\n"
         "    float4* rc4 = reinterpret_cast<float4*>(red + 512);\n"
         "    static_assert(2048 + 8 * HD * 4 <= WGS_NS * UNIT_BYTES + WGS_NX * WGS_XS, \"fin1 row cache fits the ring\");\n"
         "#endif\n")
P1 = ("          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;\n"
      "        }\n"
      "      }\n")
P1_F = ("          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;\n"
        "#if OKC_FIN1\n"
        "          rc4[sl * (HD / 4) + tid * 2] = a; rc4[sl * (HD / 4) + tid * 2 + 1] = b;\n"
        "#endif\n"
        "        }\n"
        "      }\n")
P2 = ("          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n"
      "#if OKC_YBF\n")
P2_F = ("          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n"
        "#if OKC_FIN1\n"
        "          const float4 a = rc4[sl * (HD / 4) + tid * 2], b = rc4[sl * (HD / 4) + tid * 2 + 1];\n"
        "#elif OKC_YBF\n")
EDITS = [(RED, RED_F), (P1, P1_F), (P2, P2_F)]


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
    if tail.count(MAIN) != 1:
        sys.exit("MAIN define anchor")
    tail = tail.replace(MAIN, MAIN_F)
    out = head + START + cu + END + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
