#!/usr/bin/env python3
"""Layer-b2 lane candidates on ly_stk3 (decode 16 / 32 row boundaries), one change each.

--mode g32: b1 at T = 32 runs the PDL8 graft glue b1 (8-CTA clusters of 128 threads per row, stream and constants
    loaded before griddepcontrol.wait, as b1 at T = 16 already runs) instead of the one-program-per-row Gluon
    kernel. Python dispatch in `_pdl8_graft_boundary` plus the glue host's row limit (16 -> 32) in b1().
--mode sp: the Gluon row boundary with POST (b2 at T = 16 / 32) loads its Stream row (s1, written by b1, which
    completed before the MoE's non-programmatic cooperative launch; not the unpooler's output) before gdc_wait.
    POST = False (b1 at T = 32) keeps the load after the wait.

    python3 tools/dl_ly_b2_patch.py --mode g32|sp [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
MODE = sys.argv[sys.argv.index("--mode") + 1]
DST = ROOT / f"muse_decode_layer/cand/ly_b2{MODE}.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

EDITS = {
    "g32": [
        ("  if(a.rows<1 || a.rows>16) throw std::runtime_error(\"graft b1: rows must be 1..16\");\n",
         "  if(a.rows<1 || a.rows>32) throw std::runtime_error(\"graft b1: rows must be 1..32\");\n", 2, 1),
        ("    if 16 < T <= 32 and width == HIDDEN:\n        _graft_al(br, stream, s_out, h_out)\n",
         "    if T == 32 and not post and width == HIDDEN and br.dtype == torch.bfloat16:   # b2g32: glue b1, as at T = 16\n"
         "        _graft_al(br, stream, s_out, h_out)\n"
         "        _pdl8_glue_call('b1', br, stream, g[\"pre1\"], g[\"alpha1\"], g[\"beta1\"], h_out, s_out,\n"
         "                        router_out if router_out is not None else s_out, router_out is not None)\n"
         "        return True\n"
         "    if 16 < T <= 32 and width == HIDDEN:\n        _graft_al(br, stream, s_out, h_out)\n", 1, None,
         "def _pdl8_graft_boundary("),
    ],
    "sp": [
        ("    if POST:\n        post_w = _gl.load(PostW + c).to(_gl.float32)\n    _gdc_wait()\n    _gdc_launch()\n"
         "    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)\n",
         "    if POST:\n        post_w = _gl.load(PostW + c).to(_gl.float32)\n"
         "        s = _gl.load(Stream + row * H + c, cache_modifier='.cg')   # b2sp: s1 is b1's output; b1 completed before the MoE\n"
         "    _gdc_wait()\n    _gdc_launch()\n"
         "    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)\n", 1),
        ("    b = (b * r).to(_gl.bfloat16).to(_gl.float32)\n    s = _gl.load(Stream + row * H + c, cache_modifier='.cg')\n"
         "    ns = a * s + beta * b\n",
         "    b = (b * r).to(_gl.bfloat16).to(_gl.float32)\n    if not POST:\n"
         "        s = _gl.load(Stream + row * H + c, cache_modifier='.cg')\n        ns = a * s + beta * b\n"
         "    else:   # the base's contraction, spelled out (moving the load changed what LLVM fused)\n"
         "        ns = _gl.where(c < 4096, _gl.fma(a, s, beta * b), _gl.fma(beta, b, a * s))\n", 1),
    ],
}
HEADER = {
    "g32": "# ly_b2g32: ly_stk3 with b1 at T = 32 on the PDL8 graft glue b1 (tools/dl_ly_b2_patch.py --mode g32).\n",
    "sp": "# ly_b2sp: ly_stk3 with b2's Stream row (s1) loaded before gdc_wait (tools/dl_ly_b2_patch.py --mode sp).\n",
}


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for old, new, n, *opt in EDITS[MODE]:
        # opt: [first] replaces only the first copy (_GRAFT_GLUE_CUDA, the PDL8 graft glue's source, precedes
        # _SL_GLUE_CUDA); [None, scope] edits only the text from `scope` on (_graft_boundary shares the anchor)
        first = opt[0] if opt and opt[0] else -1
        k = text.index(opt[1]) if len(opt) > 1 else 0
        head, tail = text[:k], text[k:]
        if tail.count(old) != n:
            sys.exit(f"anchor found {tail.count(old)} times, expected {n}: {old[:90]!r}")
        text = head + tail.replace(old, new, first)
    text = HEADER[MODE] + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
