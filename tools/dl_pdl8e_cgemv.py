#!/usr/bin/env python3
"""pdl8e: pdl8c2 (tools/dl_pdl8c_patch.py --cgpdl) with the m = 1 cgemv PDL edits that keep the plain
kernel's standalone time. Built, not dispatched; writes muse_decode_layer/cand/g1b4_pdl8e.py.

    python3 tools/dl_pdl8e_cgemv.py CG_TOOL C2 [--variant fe]
      CG_TOOL, C2  as for tools/dl_pdl8c_patch.py

The BF16-board bench (CGPDL-01 aa5e4ae178d244de, CGPDL-02 990758fa4f634261, docs/runs.md
"2026-11-03 m = 1 cgemv as a PDL participant") found pdl8c2's cost in two placements, not in the
launch attribute or griddepcontrol.wait: launch_dependents before the x loads (+1.9 us on o) and x
through L2 for every CTA (+2.8 us on o). Variant fe keeps the wait and the __ldcg x reads (the rule
for anything a PDL secondary reads from its primary) and moves the trigger to the CTA's end, after the
last FMA and before the final barrier, in every thread.
"""
import ast
import hashlib
import importlib.util
import pathlib
import py_compile
import sys

HERE = pathlib.Path(__file__).resolve().parent

FE_EDITS = [
    ("#pragma unroll\n        for (int i = 0; i < CPT; ++i) {\n            xa[i] = __ldg(xp + i * NTH * 2);\n"
     "            xb[i] = __ldg(xp + i * NTH * 2 + 1);\n        }",
     "        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
     "#pragma unroll\n        for (int i = 0; i < CPT; ++i) {\n            xa[i] = __ldcg(xp + i * NTH * 2);\n"
     "            xb[i] = __ldcg(xp + i * NTH * 2 + 1);\n        }", 1),
    ("    __syncthreads();\n    if (t == 0) {\n        float r0 = 0.f, r1 = 0.f;",
     "    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
     "    __syncthreads();\n    if (t == 0) {\n        float r0 = 0.f, r1 = 0.f;", 1),
]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "fe"
    args = [a for a in args if a != variant]
    if variant != "fe":
        sys.exit(f"unknown variant {variant}")
    cg_tool, c2 = args
    P = load("dl_pdl8c_patch", HERE / "dl_pdl8c_patch.py")
    CG = load("dl_cg_patch", cg_tool)
    B = load("dl_pdl8b_patch", HERE / "dl_pdl8b_patch.py")
    release, block = B.make_block()
    c2_bytes = pathlib.Path(c2).read_bytes()
    plain = CG.build(release.encode(), c2_bytes, front=False)
    if hashlib.sha256(plain).hexdigest() != P.CG_BYTES_SHA:
        sys.exit("the cgemv tool no longer reproduces the parity lane's check-2 bytes 440397ab")
    composed = plain.decode()
    head = "# okc: problem=muse_decode_layer\n"
    composed = composed.replace(head, head + "# pdl8e: release + cgemv m = 1 port (tools/dl_cg_patch.py) + pdl8b"
                                " + cgemv PDL edits variant " + variant + ", composed by tools/dl_pdl8e_cgemv.py.\n", 1)
    lit = next(ast.literal_eval(n.value) for n in ast.parse(composed).body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "CG_CUDA_SRC")
    edits = FE_EDITS + P.CG_PDL_EDITS[1:]
    for old, new, n in edits:
        if lit.count(old) != n:
            sys.exit(f"cgemv anchor {old[:60]!r}: {lit.count(old)} found, {n} expected")
        lit = lit.replace(old, new)
    out = composed.rstrip("\n") + "\n" + block
    out = out.rstrip("\n") + "\n" + P.CG_PDL_BLOCK.replace("@@CGEDITS@@", repr(edits)).replace(
        "# pdl8c2: the three m = 1 cgemv kernels as PDL participants at T = 1 (wait before x, x through __ldcg, trigger\n"
        "# after the weight issue; programmatic launch only while T = 1 runs with PDL on).",
        "# pdl8e: the three m = 1 cgemv kernels as PDL participants at T = 1 (wait before x, x through __ldcg, trigger\n"
        "# at the CTA's end; programmatic launch only while T = 1 runs with PDL on).")
    if "trigger\n# at the CTA's end" not in out:
        sys.exit("pdl8c2 block comment anchor changed")
    dst = B.A.ROOT / "muse_decode_layer/cand/g1b4_pdl8e.py"
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst, hashlib.sha256(dst.read_bytes()).hexdigest(), len(dst.read_bytes()))


if __name__ == "__main__":
    main()
