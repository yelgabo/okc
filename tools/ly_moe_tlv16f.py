#!/usr/bin/env python3
"""Verify 16 / 32 tail timeline: tools/ly_moe_tlv16.py's stamps plus final-pass stamps, on the MAIN WGS build.

Writes muse_decode_layer/cand/ly_fabq_tlv16f.py (or --out) from the pinned base (--base / --pin as ly_moe_tlv16.py).
Added slots per CTA (336 per CTA as before), all outside the consumers' stage loop, in the WGSFINSS final pass:
  324 + s  first token: thread s's expert wait ended (its count reached target), s < 8, local routes only
  332      first token: (expert id + 1) per route slot, one byte each (0 = not local)
  333      first token: pass 1 done (the sum of squares of every local route is in smem)
  334      first token: pass 2 done (output row stored)
  335      first token: min(255, tokens of the expert) per route slot, one byte each
  22 / 23  second token (verify 32): wait ended (CTA barrier passed) / pass 2 done

    python3 tools/ly_moe_tlv16f.py [--base F --pin SHA] [--out muse_decode_layer/cand/ly_fabq_tlv16f.py]
"""
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ly_moe_tlb4 as tb  # noqa: E402
import ly_moe_tlv16 as t16  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DST = ROOT / "muse_decode_layer/cand/ly_fabq_tlv16f.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

WAIT = ("          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;\n"
        "          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }\n"
        "        }\n"
        "      }\n"
        "      CTA_SYNC();\n"
        "      if (tk == blockIdx.x) DBG(4);\n"
        "#endif\n")
WAIT_F = ("          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;\n"
          "          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }\n"
          "#if OKC_TLB\n"
          "          if (TLB_ON && tk == (int)blockIdx.x) p.dbg[blockIdx.x * 336 + 324 + tid] = gtimer();\n"
          "#endif\n"
          "        }\n"
          "      }\n"
          "      CTA_SYNC();\n"
          "      if (tk == blockIdx.x) DBG(4);\n"
          "#if OKC_TLB\n"
          "      if (TLB_ON && tk == (int)blockIdx.x + (int)gridDim.x && tid == 0) p.dbg[blockIdx.x * 336 + 22] = gtimer();\n"
          "      if (TLB_ON && tk == (int)blockIdx.x && tid < 8 && S.pos[tk * 8 + tid] >= 0) {\n"
          "        const int e = S.eid[tk * 8 + tid];\n"
          "        atomicOr(p.dbg + blockIdx.x * 336 + 332, (unsigned long long)(e + 1) << (8 * tid));\n"
          "        atomicOr(p.dbg + blockIdx.x * 336 + 335, (unsigned long long)min(255, S.cnt[e]) << (8 * tid));\n"
          "      }\n"
          "#endif\n"
          "#endif\n")
P1 = ("        red[NW * 8 + tid] = tsum;\n"
      "      }\n"
      "      CTA_SYNC();\n"
      "      float acc[8];\n")
P1_F = ("        red[NW * 8 + tid] = tsum;\n"
        "      }\n"
        "      CTA_SYNC();\n"
        "#if OKC_TLB\n"
        "      if (TLB_ON && tk == (int)blockIdx.x && tid == 0) p.dbg[blockIdx.x * 336 + 333] = gtimer();\n"
        "#endif\n"
        "      float acc[8];\n")
P2 = ("      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\n"
      "      CTA_SYNC();  // red is reused by the next token\n")
P2_F = ("      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\n"
        "      CTA_SYNC();  // red is reused by the next token\n"
        "#if OKC_TLB\n"
        "      if (TLB_ON && tid == 0) {\n"
        "        if (tk == (int)blockIdx.x) p.dbg[blockIdx.x * 336 + 334] = gtimer();\n"
        "        else if (tk == (int)blockIdx.x + (int)gridDim.x) p.dbg[blockIdx.x * 336 + 23] = gtimer();\n"
        "      }\n"
        "#endif\n")
F_EDITS = [(WAIT, WAIT_F), (P1, P1_F), (P2, P2_F)]


def main():
    t16.DST = DST
    t16.main()
    whole = DST.read_text()
    head, rest = whole.split(tb.START)
    cu, tail = rest.split(tb.END)
    for old, new in F_EDITS:
        n = cu.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, 1 expected: {old[:100]!r}")
        cu = cu.replace(old, new)
    out = head + tb.START + cu + tb.END + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
