"""Copy the fused MoE kernel from muse_moe_w4a16/submission.py into muse_decode_layer/submission.py.

The source keeps its device code as templates (_CUDA_SRC with //@WGMMA@, //@SKINNY@ and
//@DISPATCH@ markers, one _SKINNY_BLOCK per skinny variant, the wgmma wrappers from
_wgmma_code()) and assembles them in _cuda_source(). This tool runs the same assembly for the
computing variants only (skinny mode 0, the fat modes in MAIN_CANDIDATES; the loads-only
diagnostics stay out), drops the torch host functions after the anonymous namespace, wraps the
device code in a nested `moe1` namespace as _MOE1_SRC, writes the variant table the layer's
host wrapper dispatches on (_MOE1_VARIANTS, _MOE1_MAIN_MIN_M), and copies the repack functions
(unit_rows, repack_expert, repack_expert_fat and their byte helpers) under a moe1_ prefix
between the @MOE1_REPACK@ markers.

One device patch: the layer's local_start is an input tensor it never reads on the host (its
route_align takes it from the device too), so Params gains `const int* lo_p` and both kernels
take the local range's start from it. Each anchor must match exactly once per template; a
changed kernel fails here instead of compiling with a stale patch.

    python3 tools/dl_moe_inline.py [path/to/muse_moe_w4a16/submission.py]
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_decode_layer/submission.py"
SRC = ROOT / "muse_moe_w4a16/submission.py"

NAMES = {"_CUDA_SRC", "_SKINNY_BLOCK", "SKINNY_VARIANTS", "MAIN_MODES", "MAIN_CANDIDATES",
         "MAIN_MIN_M", "W4_BITPOS"}
FUNCS = {"_wgmma_code", "_skinny_tag"}
REPACK = ("unit_rows", "repack_expert", "repack_expert_fat", "_to_u8_words", "_cat_last")
RENAME = (("repack_expert_fat", "moe1_repack_expert_fat"), ("repack_expert", "moe1_repack_expert"),
          ("unit_rows", "moe1_unit_rows"))
MAIN_NAMES = {0: "main-d2", 5: "main-d3", 16: "main-m2-ar2-ub", 20: "main-own",
              23: "main-own-dynb"}
# Each fat mode is another kernel instance in the layer's build, so the layer builds only the
# fat modes that won a standalone bench: 0 (its fallback) and 16, which took 128 and 256
# tokens from legacy128 and mode 12 in checks 9a9fb08c134c4a16 and c65e91500215418d, and 20,
# 2-4 us under 16 at 128 and 256 tokens in three same-process A/Bs, and 23, 16-23 us under 20
# at 128 and 256 tokens on 24 of 24 paired seeds in checks a9dbec1d98334943 and
# 36812632dae242ba (docs/runs-moe.md).
LAYER_MODES = (0, 16, 20, 23)

# The load is a per-use ld.global.nc in inline asm, consumed where the host value was: a value
# loaded once and kept in a register doubled the fat d2 kernel's spills (1268 B against 604),
# whichever block held it, while this form leaves every kernel's allocation as in the source.
LO_HELPER = (
    "__device__ __forceinline__ int ld_lo(const int* p) {\n"
    "    int v;\n"
    "    asm volatile(\"ld.global.nc.s32 %0, [%1];\" : \"=r\"(v) : \"l\"(p));\n"
    "    return v;\n"
    "}\n\n"
)
CUDA_PATCHES = [
    ("    int cost[2][9];\n};\n",
     "    int cost[2][9];\n"
     "    // The layer's local_start is an input tensor, read here instead of the host value.\n"
     "    const int* lo_p;\n};\n"),
    ("__device__ __forceinline__ int ld_acquire(const int* p) {\n",
     LO_HELPER + "__device__ __forceinline__ int ld_acquire(const int* p) {\n"),
    ("                const int ev[4] = {v[k].x - p.lo, v[k].y - p.lo, v[k].z - p.lo, v[k].w - p.lo};\n",
     "                const int lo = ld_lo(p.lo_p);\n"
     "                const int ev[4] = {v[k].x - lo, v[k].y - lo, v[k].z - lo, v[k].w - lo};\n"),
]
SKINNY_PATCHES = [
    ("                const int e = ev[k] - p.lo;\n",
     "                const int e = ev[k] - ld_lo(p.lo_p);\n"),
]


def patch(text, patches):
    for old, new in patches:
        if text.count(old) != 1:
            raise SystemExit(f"anchor found {text.count(old)} times, need 1: {old!r}")
        text = text.replace(old, new)
    return text


def load(text):
    tree = ast.parse(text)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in FUNCS)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in NAMES for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "moe_submission", "exec"), ns)
    funcs = {n.name: ast.get_source_segment(text, n) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in REPACK}
    missing = set(REPACK) - set(funcs)
    if missing:
        raise SystemExit(f"repack functions missing from the source: {sorted(missing)}")
    return ns, funcs


def device_code(ns):
    cuda = patch(ns["_CUDA_SRC"], CUDA_PATCHES)
    block = patch(ns["_SKINNY_BLOCK"], SKINNY_PATCHES)
    tag = ns["_skinny_tag"]
    skinny_variants = [v for v in ns["SKINNY_VARIANTS"] if v[2] == 0]
    main_modes = [m for m in ns["MAIN_MODES"] if m in ns["MAIN_CANDIDATES"] and m in LAYER_MODES]
    if not main_modes:
        raise SystemExit(f"none of LAYER_MODES {LAYER_MODES} is a fat candidate in the source")
    skinny = "\n".join(block.replace("@TAG@", tag(nw, ns_, mode)).replace("@NW@", str(nw))
                       .replace("@NS@", str(ns_)).replace("@MODE@", str(mode))
                       for nw, ns_, mode in skinny_variants)
    cuda = cuda.replace("//@WGMMA@", ns["_wgmma_code"]()).replace("//@SKINNY@", skinny)
    end = "\n}  // namespace\n"
    if cuda.count(end) != 1:
        raise SystemExit(f"expected one anonymous-namespace close, found {cuda.count(end)}")
    head = cuda[:cuda.index(end)]
    if "p.lo" in head.replace("p.lo_p", ""):
        raise SystemExit("an unpatched use of Params::lo remains in the device code")
    head, n = re.subn(r"^namespace \{$", "namespace {\nnamespace moe1 {", head, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("no 'namespace {' line in the kernel source")
    code = head.rstrip() + "\n\n}  // namespace moe1\n}  // namespace\n"
    variants = [(f"sk{nw}w{ns_}s", "skinny", f"sk_{tag(nw, ns_, mode)}::launch<1>")
                for nw, ns_, mode in skinny_variants]
    variants += [(MAIN_NAMES.get(m, f"main-m{m}"), "main", f"launch_main<{m}>") for m in main_modes]
    return code, variants, int(ns["MAIN_MIN_M"])


def repack_code(funcs):
    out = []
    for name in REPACK:
        text = funcs[name]
        for old, new in RENAME:
            text = re.sub(rf"\b{old}\b", new, text)
        out.append(text)
    return "\n\n\n".join(out)


def replace_once(src, pattern, repl, what):
    new, n = re.subn(pattern, repl, src, count=1, flags=re.S | re.M)
    if n != 1:
        raise SystemExit(f"no {what} in submission.py")
    return new


def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    text = src_path.read_text()
    ns, funcs = load(text)
    sub = SUB.read_text()
    layer_bitpos = ast.literal_eval(re.search(r"^W4_BITPOS = (\(.*?\n\))", sub, re.S | re.M).group(1))
    if tuple(layer_bitpos) != tuple(ns["W4_BITPOS"]):
        raise SystemExit("the layer's W4_BITPOS differs from the fused kernel's; copy it too")
    for helper in ("_perm", "_stack_last", "_to_i64"):
        if f"\ndef {helper}(" not in sub:
            raise SystemExit(f"the layer lacks the repack helper {helper}")
    code, variants, main_min = device_code(ns)
    sub = replace_once(sub, r'_MOE1_SRC = r""".*?"""', lambda _: f'_MOE1_SRC = r"""\n{code}"""',
                       "_MOE1_SRC block")
    sub = replace_once(sub, r"^_MOE1_VARIANTS = \(.*?\n\)$",
                       lambda _: "_MOE1_VARIANTS = (\n" + "".join(f"    {v!r},\n" for v in variants) + ")",
                       "_MOE1_VARIANTS table")
    sub = replace_once(sub, r"^_MOE1_MAIN_MIN_M = \d+$", f"_MOE1_MAIN_MIN_M = {main_min}",
                       "_MOE1_MAIN_MIN_M line")
    sub = replace_once(sub, r"# @MOE1_REPACK_BEGIN@\n.*?# @MOE1_REPACK_END@\n",
                       lambda _: "# @MOE1_REPACK_BEGIN@\n" + repack_code(funcs) + "\n# @MOE1_REPACK_END@\n",
                       "@MOE1_REPACK@ region")
    SUB.write_text(sub)
    print(f"inlined {len(code.splitlines())} device lines and {len(REPACK)} repack functions from "
          f"{src_path}; variants {[v[0] for v in variants]}, fat path from {main_min} tokens")


if __name__ == "__main__":
    main()
