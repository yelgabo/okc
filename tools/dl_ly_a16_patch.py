#!/usr/bin/env python3
"""ly_a16: per-CTA globaltimer stamps of the decode 16 / 32 sliding attention (attn_kernel<1, false, 1>) inside the
layer, on ly_stk2. Instrument only, never a candidate.

Everything tools/dl_ly_ast_patch.py --mrg does (attention slots 0..21, sk2 entry / wait / end stamps), plus:

- the sk2 stamps fire for M in {1, 16, 32} (qkv and o at decode 16 / 32), not only M == 1;
- per-tile stamps in the per-request cp.async loop: 22..24 tile 1..3's stage visible (after its cp_wait and CTA
  barrier), 25..27 tile 1..3's math done (slots 12 / 13 keep tile 0).

No fence, barrier, wait, trigger or launch attribute moves.

    python3 tools/dl_ly_a16_patch.py [--base muse_decode_layer/cand/ly_stk2.py] [--out muse_decode_layer/cand/ly_a16.py] [--kvw]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
base = "muse_decode_layer/cand/ly_stk2.py"
out = "muse_decode_layer/cand/ly_a16.py"
if "--base" in sys.argv:
    base = sys.argv[sys.argv.index("--base") + 1]
if "--out" in sys.argv:
    out = sys.argv[sys.argv.index("--out") + 1]
PIN = "4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846"
kvw = "--kvw" in sys.argv
sys.argv = [sys.argv[0], "--base", base, "--out", out, "--mrg"] + (["--kvw"] if kvw else [])
import dl_ly_ast_patch as ast  # noqa: E402

TILE_EDITS = [
    ("      __syncthreads();\n      PROF_MARK(0);\n      if (i == 0) DBG(12);\n      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);\n",
     "      __syncthreads();\n      PROF_MARK(0);\n      if (i == 0) DBG(12);\n      if (L == 1 && i > 0 && i < 4) { DBG(21 + i); }\n"
     "      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);\n"),
    ("    PROF_TILE;\n    if (i == 0) DBG(13);\n  }\n  PROF_DUMP;\n  } else {\n",
     "    PROF_TILE;\n    if (i == 0) DBG(13);\n    if (i > 0 && i < 4) { DBG(24 + i); }\n  }\n  PROF_DUMP;\n  } else {\n"),
]
SK2_M = ("if (g_ast_p[i_] != nullptr && M == 1 && N == g_ast_n[i_]",
         "if (g_ast_p[i_] != nullptr && (M == 1 || M == 16 || M == 32) && N == g_ast_n[i_]")


def main():
    src = (ROOT / base).read_text()
    if base.endswith("ly_stk2.py") and hashlib.sha256(src.encode()).hexdigest() != PIN:
        sys.exit("ly_stk2 SHA256 mismatch")
    ast.PIN = None
    ast.SK2_EDITS = [(o, n.replace(*SK2_M) if SK2_M[0] in n else n) for o, n in ast.SK2_EDITS]
    ast.ATT_EDITS = ast.ATT_EDITS + TILE_EDITS
    ast.HEADER = ("# ly_a16: ly_stk2 with per-CTA globaltimer stamps in the q_len 1 attention kernel (per-tile slots) and "
                  "the T = 1 / 16 / 32 qkv / o sk2 launches, off in the plain graph (tools/dl_ly_a16_patch.py). "
                  "Instrument, not a candidate.\n")
    ast.main()
    dst = ROOT / out
    py_compile.compile(str(dst), doraise=True)


if __name__ == "__main__":
    main()
