#!/usr/bin/env python3
"""ly_aqp: ly_aqd + the one-tile q_len 8 attention writes the block's new e4m3 key / value rows straight into its K / V
stage in the new k,v pass, instead of staging them in S.kn / S.vn and copying them in a separate patch pass.

Round 2 stamps on ly_aqd: the split holding the new rows reaches its ticket about 0.5 us after the median split and the
grid barrier waits for it; 0.18 us of that is the patch pass (shared-memory copy of the staged rows into the swizzled
stage, then fence.proxy.async). One change, behind constexpr PD = OKC_NEWDIR8 && ONE && L == 8 && !S2 && !WS: in the
new k,v pass, a thread whose row lies in this CTA's tile ([start, end), tile 0, stage 0) also stores its 8 bytes at
the row's swizzled position in the stage (the same position the patch would write: chunk hh * 4 + partL / 2, xor row
& 7); the patch pass is skipped and one fence.proxy.async follows the pass in CTAs with new rows. The one-tile copies
never touch these rows (their cp.async is predicated off for positions in [seq, end)), so the stores race with no
copy. The cache-row writes, the wait, the trigger and the merge are unchanged.

    python3 tools/dl_ly_apd_patch.py [--src muse_decode_layer/cand/ly_aqd.py] [--out muse_decode_layer/cand/ly_aqp.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_aqd.py"
DST = ROOT / "muse_decode_layer/cand/ly_aqp.py"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

EDITS = [
    ("#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n",
     "#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)\n"
     "#define OKC_NEWDIR8 1   // one-tile q_len 8 builds: new k / v rows stored straight into the stage, no patch pass\n"),
    ("  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n",
     "  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;\n"
     "  constexpr bool PD = OKC_NEWDIR8 && ONE && L == 8 && !WS && !Cfg<L, FLAT>::S2;\n"),
    ("        const int pos = seq + jj;\n        if (pos >= start && pos < end) {\n",
     "        const int pos = seq + jj;\n        if (pos >= start && pos < end) {\n"
     "          if constexpr (PD) {   // tile 0 is stage 0; the patch pass's swizzled position, 8 of its 16 bytes\n"
     "            const int r = pos - start, c = hh * 4 + (partL >> 1);\n"
     "            *reinterpret_cast<uint2*>((kind == 0 ? S.k : S.v) + r * ROWB + ((c ^ (r & 7)) << 4) + (partL & 1) * 8) =\n"
     "                make_uint2(w0, w1);\n"
     "          }\n"),
    ("  csync<WS, NT>();\n  DBG(11);\n  if (!WS && has_new) {\n",
     "  if constexpr (PD) if (has_new) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma\n"
     "  csync<WS, NT>();\n  DBG(11);\n  if (!PD && !WS && has_new) {\n"),
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
    text = (f"# ly_aqp: {SRC.stem} + the one-tile q_len 8 attention stores its new k / v rows straight into the stage, "
            "no patch pass (tools/dl_ly_apd_patch.py).\n") + text[:i] + blk + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
