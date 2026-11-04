#!/usr/bin/env python3
"""ly_aqi: ly_aqd + the QDIRECT q pass (OKC_QP16, two passes per thread) loads every shared operand of both passes
before its first store, so the two passes' arithmetic can interleave.

Round 2 stamps and SASS on ly_aqd: the q pass is about 280 instructions per thread and takes 0.85 us of thread 0's
time between the RoPE table's barrier and the fence (a stamp just before fence.proxy.async reads the same time as one
after it, so the fence waits for nothing). In the SASS the two passes run one after the other (MUFU.RSQ at 0x2ad0 and
0x3380, each pass's e4m3 store before the next pass's loads): the compiler cannot move the second pass's loads of the
raw q rows and the RoPE table above the first pass's st.shared to the e4m3 tile, which it cannot prove disjoint. One
change, inside the QP16 branch (compiled only for the one-tile q_len 8 builds on ly_aqd): the raw q chunks and the
cos / sin values of both passes are loaded into registers first (cos 1 / sin 0 when the layer has no RoPE, so the
rotation needs no branch), then the unchanged per-pass arithmetic and stores run from registers. Same values, same
floating-point operations per element: bitwise identical output with RoPE.

    python3 tools/dl_ly_aqi_patch.py [--src muse_decode_layer/cand/ly_aqd.py] [--out muse_decode_layer/cand/ly_aqi.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_aqd.py"
DST = ROOT / "muse_decode_layer/cand/ly_aqi.py"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

EDITS = [
    ("    const int q4 = tidL & 3;\n#pragma unroll\n    for (int ps = 0; ps < QP4; ++ps) {\n"
     "      const int rr = ps * RP4 + (tidL >> 2);\n      const int j = rr >> 4, hh = rr & 15;\n"
     "      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);\n"
     "      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);\n",
     "    const int q4 = tidL & 3;\n"
     "    // (OKC_QIL) every shared load of both passes first: the stores to S.q8s cannot be proven disjoint from them,\n"
     "    // so loads left inside the loop serialize the passes\n"
     "    uint4 qa_[QP4], qb_[QP4];\n    float cs_[QP4][8], sn_[QP4][8];\n"
     "#pragma unroll\n    for (int ps = 0; ps < QP4; ++ps) {\n"
     "      const int rr = ps * RP4 + (tidL >> 2);\n      const int j = rr >> 4;\n"
     "      qa_[ps] = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);\n"
     "      qb_[ps] = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);\n"
     "#pragma unroll\n      for (int e = 0; e < 8; ++e) {\n"
     "        cs_[ps][e] = p.rope ? S.cs[j * 32 + q4 * 8 + e] : 1.f;   // (identity rotation without RoPE: no branch below)\n"
     "        sn_[ps][e] = p.rope ? S.sn[j * 32 + q4 * 8 + e] : 0.f;\n      }\n    }\n"
     "#pragma unroll\n    for (int ps = 0; ps < QP4; ++ps) {\n"
     "      const int rr = ps * RP4 + (tidL >> 2);\n      const int j = rr >> 4, hh = rr & 15;\n"
     "      const uint4 qa = qa_[ps];\n      const uint4 qb = qb_[ps];\n"),
    ("      if (p.rope) {\n#pragma unroll\n        for (int e = 0; e < 8; ++e) {\n          const int i = q4 * 8 + e;\n"
     "          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];\n",
     "      {\n#pragma unroll\n        for (int e = 0; e < 8; ++e) {\n          const float c = cs_[ps][e], sn = sn_[ps][e];\n"),
]


def main():
    text = SRC.read_text()
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    for old, new in EDITS:
        if blk.count(old) != 1:
            sys.exit(f"anchor found {blk.count(old)} times: {old[:80]!r}")
        blk = blk.replace(old, new)
    text = (f"# ly_aqi: {SRC.stem} + the one-tile q_len 8 QDIRECT q pass loads both passes' shared operands before its "
            "first store (tools/dl_ly_aqi_patch.py).\n") + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
