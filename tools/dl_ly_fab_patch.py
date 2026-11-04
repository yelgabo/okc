#!/usr/bin/env python3
"""Blueprint micro-changes on ly_micro2.py (0bc2bd8b), one per --variant; the output SHA256 is printed.

    python3 tools/dl_ly_fab_patch.py --variant b2pre [--variant ...] --out muse_decode_layer/cand/ly_fab_<v>.py

Variants (docs/runs-decode-layer.md "Blueprint builds (fable lane)"):
  b2pre   T = 16 boundary 2 (glue_cluster_boundary POST, the PDL8 graft glue): the entry griddepcontrol.wait no longer
          precedes the weight preloads (alpha, beta, pre, post are prepare-time constants) and each thread prefetches its
          s1 chunk into L2; the wait comes right after the preloads and the s1 (.cg) loads stay behind it. The second
          wait before the branch (unpooler output) loads is unchanged. SLICED keeps its entry wait.
  b1d32   T = 32 boundary 1 (_graft_row_boundary_pre_cg, POST = False): the stream load moves before gdc_wait. For
          boundary 1 the stream is data['stream'], the harness input, which nothing in the call writes; boundary 2
          (POST) keeps its stream (s1) load behind the wait with .cg.
No fence, barrier, mbarrier or griddepcontrol.wait removed; no new PDL edge; no launch attribute change.
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_micro2.py"
PIN = "0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd"


def once(t, old, new, what):
    n = t.count(old)
    if n != 1:
        sys.exit(f"{what}: anchor found {n} times")
    return t.replace(old, new)


B2PRE_EDITS = [
    ('void glue_cluster_boundary(GlueArgs a) {\n  if constexpr (SLICED || POST) asm volatile("griddepcontrol.wait;" ::: "memory");',
     'void glue_cluster_boundary(GlueArgs a) {\n  if constexpr (SLICED) asm volatile("griddepcontrol.wait;" ::: "memory");', 1),
    ('    glfcg<V>(a.stream+row*8192+col,s+r*V);\n    glf<V>(a.alpha+col,alpha+r*V);\n',
     '    if constexpr(POST) asm volatile("prefetch.global.L2 [%0];" :: "l"(a.stream+row*8192+col));\n'
     '    else glfcg<V>(a.stream+row*8192+col,s+r*V);\n'
     '    glf<V>(a.alpha+col,alpha+r*V);\n', 1),
    ('  // DSMEM destinations must exist before the first remote store.',
     '  if constexpr(POST) {  // b2: the stream is s1 from boundary 1 of this call, so it waits; the weights above are prepare-time constants\n'
     '    asm volatile("griddepcontrol.wait;" ::: "memory");\n'
     '    #pragma unroll\n'
     '    for(int r=0;r<repeats;r++) {\n'
     '      int col=logical*V+r*logical_threads*V;\n'
     '      glfcg<V>(a.stream+row*8192+col,s+r*V);\n'
     '    }\n'
     '  }\n'
     '  // DSMEM destinations must exist before the first remote store.', 1),
]

EDITS_LINE = "_PDL8_GRAFT_GLUE_EDITS = ["

B1D32_OLD_1 = """    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)"""
B1D32_NEW_1 = """    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    else:
        s = _gl.load(Stream + row * H + c)  # boundary 1: the stream is the harness input, nothing in the call writes it
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)"""
B1D32_OLD_2 = """    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c, cache_modifier='.cg')
    ns = a * s + beta * b"""
B1D32_NEW_2 = """    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    if POST:
        s = _gl.load(Stream + row * H + c, cache_modifier='.cg')
    ns = a * s + beta * b"""


def apply(s, v):
    if v == "b2pre":
        i = s.index(EDITS_LINE)
        j = s.index("\n", i) + 1
        stmt = "_PDL8_GRAFT_GLUE_EDITS += " + repr(B2PRE_EDITS) + "   # fab b2pre\n"
        return s[:j] + stmt + s[j:]
    if v == "b1d32":
        i = s.index("def _graft_row_boundary_pre_cg(")
        j = s.index("\ndef ", i + 1)
        t = s[i:j]
        t = once(t, B1D32_OLD_1, B1D32_NEW_1, v)
        t = once(t, B1D32_OLD_2, B1D32_NEW_2, v)
        return s[:i] + t + s[j:]
    sys.exit(f"unknown variant {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("base SHA mismatch")
    s = raw.decode()
    for v in a.variant:
        s = apply(s, v)
    out = ROOT / a.out
    out.write_text(s)
    py_compile.compile(str(out), doraise=True)
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
