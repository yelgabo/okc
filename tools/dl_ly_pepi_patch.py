#!/usr/bin/env python3
"""ly_pepi: a shorter tail for the fused pooler + FP32 router kernel at T <= 8 (sk2_kernel_gle, FUSE, GLE_ILV 1).

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 37654700..., checked). Target: the 0.80 us every CTA spends between
its stage-loop exit and its final store (docs/runs-decode-layer.md, "Pooler + router remaining gap"): 0.23 us of warp
butterflies that reduce all 8 token slots of the router partial whatever M is, and three CTA-wide barrier rounds (the
ilv block's cbar wait + named barrier, the epilogue's cbar wait + named barrier before the K-warp reduction buffer is
written over the ring, the named barrier after it) for 0.22 us.

Change, gle fused build only (SK2_REDX 1, defined by the gle source transform; the bf16_proj build and the o kernel keep
SK2_REDX 0 and compile the same code as before):
  1. The K-warp reduction buffer `red` has its own shared-memory region after the zero row (WSK x TOKS x RP floats,
     20 KB; the launch adds it to the dynamic size), so the consumers write it right after the loop without first
     proving the ring idle: the epilogue's cbar wait + named barrier before the write go, the cbar wait moves after
     the write (the final sum reads the escape corrections), and the one named barrier after the write stays.
  2. The ilv block's cbar wait + named barrier after the router partial store run only when outr != nullptr (the
     counter path, never launched by this file; the barrier orders the partial stores before its release atom).
  3. At M == 1 the router warp butterflies reduce token 0 only (5 shuffles instead of 40); at M > 1 the original 8
     interleaved chains stay. The per-token shuffle order is unchanged, so the partials the route kernel sums are
     bitwise the same. (ly_pepi, dc425a65, guarded each token's chain with `if (t < M)`, which serialized the chains:
     0.96 us at T = 8.)
Kept: every fence, every mbarrier wait, the ring protocol, the escape warps' cbar, the named barrier that publishes
red, the summation orders of the pooler output and the router partials.

    python3 tools/dl_ly_pepi_patch.py [--base F] [--pin SHA] [--out muse_decode_layer/cand/ly_pepi2.py]
"""
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dl_ly_prtl_patch as tlp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_pepi2.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

K_EDITS = [
    ("#define SK2_DIAG 0\n",
     "#define SK2_DIAG 0\n"
     "#ifndef SK2_REDX\n#define SK2_REDX 0   // pepi: 1 in the gle fused build (red outside the ring, shorter tail)\n#endif\n"),
    ("  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed\n",
     "  constexpr bool REDX = SK2_REDX && FUSE;\n"
     "  // REDX: red has its own region after the zero row (sized by the launch), else it reuses the consumed ring\n"
     "  float* red = REDX ? reinterpret_cast<float*>(gbase + ((((zrow + 128u) + 15u) & ~15u) - sbase))\n"
     "                    : reinterpret_cast<float*>(gbase);\n"),
    ("                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;\n",
     "                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128 +\n"
     "                      ((SK2_REDX && FUSE) ? (size_t)WSK * TOKS * RP * 4 + 16 : 0);\n"),
    # the epilogue: without REDX as before; with REDX the ring-idle proof before the red write is not needed and the
    # cbar wait (escape corrections complete) moves after the write
    ("  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done\n"
     "  if (CL > 1 && crank > 0) {\n",
     "  if constexpr (COMP) if (!REDX) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done\n"
     "  if (CL > 1 && crank > 0) {\n"),
    ("    if (trace && tid == 0) tr[15] = gtime();\n    return;\n  }\n  named_bar(1, NC);\n  if (trace && tid == 0) tr[9] = gtime();\n",
     "    if (trace && tid == 0) tr[15] = gtime();\n    return;\n  }\n  if (!REDX) named_bar(1, NC);\n  if (trace && tid == 0) tr[9] = gtime();\n"),
    ("  if (trace && tid == 0) tr[11] = gtime();\n  named_bar(1, NC);\n  if (trace && tid == 0) tr[10] = gtime();\n",
     "  if (trace && tid == 0) tr[11] = gtime();\n"
     "  if constexpr (COMP) if (REDX) mbar_wait(cbar, 0);   // the escape corrections are complete before the final sum\n"
     "  named_bar(1, NC);\n  if (trace && tid == 0) tr[10] = gtime();\n"),
]
F_EDITS = [
    ('    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\\n"\n',
     '    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\\n#define SK2_REDX 1\\n"\n', 1),
    ('         "#pragma unroll\\n"\n'
     '         "      for (int t = 0; t < 8; ++t)\\n"\n'
     '         "#pragma unroll\\n"\n'
     '         "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);\\n"\n',
     # M == 1: one token's chain; else all 8 chains as before, interleaved by the compiler (a per-token `if (t < M)`
     # serialized the 8 chains: 0.96 us at T = 8 against 0.23, run-log "Pooler + router remaining gap")
     '         "      if (M == 1) {   // pepi: one live token\\n"\n'
     '         "#pragma unroll\\n"\n'
     '         "        for (int d = 16; d; d >>= 1) afi[0] += __shfl_xor_sync(0xffffffffu, afi[0], d);\\n"\n'
     '         "      } else {\\n"\n'
     '         "#pragma unroll\\n"\n'
     '         "      for (int t = 0; t < 8; ++t)\\n"\n'
     '         "#pragma unroll\\n"\n'
     '         "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);\\n"\n'
     '         "      }\\n"\n', 1),
    ('         "      if constexpr (COMP) mbar_wait(cbar, 0);\\n"\n'
     '         "      named_bar(1, NC);\\n"\n'
     '         "    } else if (ep.fpr) {\\n"))\n',
     '         "      if (!REDX || outr != nullptr) {   // pepi: the counter path orders the partial stores before its atom\\n"\n'
     '         "      if constexpr (COMP) mbar_wait(cbar, 0);\\n"\n'
     '         "      named_bar(1, NC);\\n"\n'
     '         "      }\\n"\n'
     '         "    } else if (ep.fpr) {\\n"))\n', 1),
]


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    out = tlp.patch(raw.decode(), K_EDITS, F_EDITS)
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
