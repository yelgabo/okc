"""Copy the projection kernels from muse_bf16_proj/submission.py into muse_decode_layer/submission.py.

The source assembles its device code in sk_source(): the kernel template string _SK_SRC with
the config dispatch expanded from SK_CONFIGS, plus GEMV_CUDA_SRC. This tool runs that assembly
for the configs the layer uses (every SK_GEOM geometry, compressed and raw, and the two fused
pooler + router configs; the standalone router configs stay out), writes the result as
_SKP_SRC and _SKP_GEMV_SRC, the declarations as _SKP_CPP_SRC, the config table the dispatch was
generated from as _SKP_CONFIGS, and copies the packing code (the 12-bit code, tile units,
stage-major streams, escape tables, the balancing permutation, cgemv's row stream) and the
geometry helpers verbatim between the @SKP_PY@ markers, with the constants they read. The
kernel source itself is not patched. Its griddepcontrol sits behind SK_PDL (default 0, checked
here); the layer builds with -DSK_PDL=1 only when PORT_SK_PDL is set, and then launches an sk
config as a PDL secondary only when bit 16 of its cfg is set.

The wide kernel (wd_source()) goes in as _SKP_WD_SRC, built for the configs in LAYER_WD only,
with its geometry table as _SKP_WD_GEOM; the layer compiles it into the same extension after
_SKP_SRC, whose helpers (mbarrier waits, named barriers, the tensor-map encoder) it uses.

    python3 tools/dl_proj_inline.py [path/to/muse_bf16_proj/submission.py]
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_decode_layer/submission.py"
SRC = ROOT / "muse_bf16_proj/submission.py"

BUILD_NAMES = {"_SK_SRC", "_SK_CPP", "GEMV_CPP_DECLS", "GEMV_CUDA_SRC", "SK_GEOM", "ROUTER_GEOM",
               "FUSED_GEOM", "SK_USE", "SK_EXTRA", "SMEM_BUDGET", "SMEM_PER_SM", "UNPOOL_SLICE",
               "WD_GEOM", "WD_POL", "WD_ABLATE", "WD_ABL", "WD_NAMES", "WD_INDEX", "_WD_SRC",
               "_WD_CPP"}
BUILD_FUNCS = {"_sk_configs", "_sk_smem", "_fit_stages", "_sk_targs", "wd_smem", "wd_geom",
               "wd_fits", "_wgmma_spec", "_wd_targs", "wd_source"}
# Copied verbatim, in this order (constants first).
COPY_CONSTS = ("EXP_WINDOW", "MAX_ESCAPE_DENOM", "BALANCE_GROUP", "BALANCED_LAYERS", "SK_USE",
               "FUSED_GEOM", "_LAYER_THREADS")
COPY_FUNCS = ("_sk_geom", "_fused_router_geom", "bf16_bits", "tile_values", "window_base",
              "code_bits", "raw_units", "hilo_units", "comp_units", "stage_stream", "balance_perm",
              "escape_table", "pack_layer", "slices_balanced", "pack_raw_layer", "pack_router",
              "pack_cgemv", "cgemv_threads", "bucket_of")

LAYER_BUCKET_OF = """def bucket_of(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else None"""

# The standalone fuses the router into the pooler at 8 and 16 tokens; the layer also does at 32
# (the standalone problem's pooler has no router beside it to fuse).
LAYER_FUSED_GEOM = ("pool8", "pool16", "pool32")
LAYER_FUSED_GEOM_SRC = """# At 32 tokens the side-stream router stretched the pooler from 17 to 26 us (timeline in
# check 181ab39f1bd84912); one launch for both took the front chain from 37.6 to 32.4 us
# (5869cc624a394904).
FUSED_GEOM = ("pool8", "pool16", "pool32")"""

# The standalone's wide picks at 128 tokens (pooler and qkv_gate, check 39d6bfa9834f4a96). Each
# is another cluster kernel instance in the layer's build, so the other wide configs stay out.
LAYER_WD = ("b1n128c2s8h", "b1n128c3s8h")


def load(text):
    tree = ast.parse(text)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in BUILD_FUNCS)
            or (isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") in BUILD_NAMES for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "proj_submission", "exec"), ns)
    if tuple(ns["FUSED_GEOM"]) != LAYER_FUSED_GEOM[:len(ns["FUSED_GEOM"])]:
        raise SystemExit(f"the source's FUSED_GEOM {ns['FUSED_GEOM']} is not a prefix of the layer's")
    ns["FUSED_GEOM"] = LAYER_FUSED_GEOM
    missing = [nm for nm in LAYER_WD if nm not in ns["WD_GEOM"]]
    if missing:
        raise SystemExit(f"wide configs missing from the source: {missing}")
    # wd_source() and _wd_targs() read these as globals of ns, so the trimmed tables apply.
    ns["WD_POL"] = {nm for nm in LAYER_WD if nm in ns["WD_POL"]}
    ns["WD_GEOM"] = {nm: ns["WD_GEOM"][nm] for nm in LAYER_WD}
    ns["WD_ABL"] = {}
    ns["WD_NAMES"] = list(LAYER_WD)
    segments = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in COPY_FUNCS:
            segments[n.name] = ast.get_source_segment(text, n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if getattr(t, "id", "") in COPY_CONSTS:
                    segments[t.id] = ast.get_source_segment(text, n)
    # The standalone bucket_of also maps m > 32 to its wide-router buckets (WD_BUCKETS), which
    # the layer does not have: the layer sends m > 32 to cuBLAS and needs None there.
    segments["bucket_of"] = LAYER_BUCKET_OF
    segments["FUSED_GEOM"] = LAYER_FUSED_GEOM_SRC
    missing = [n for n in COPY_CONSTS + COPY_FUNCS if n not in segments]
    if missing:
        raise SystemExit(f"missing from the source: {missing}")
    return ns, segments


def device_code(ns):
    cfgs = [c for c in ns["_sk_configs"]() if c[0] not in ns["ROUTER_GEOM"]]
    targs = ns["_sk_targs"]
    dispatch = "\n".join(
        f"    case {i}: launch_sk<{targs(c)}>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, "
        f"(int)n, (int)k, dg, nullptr, nullptr, nullptr, nullptr); break;"
        for i, c in enumerate(cfgs) if not c[13])
    fdispatch = "\n".join(
        f"    case {i}: launch_sk<{targs(c)}>(ws, x, out, ptab, (int)pmax, (uint32_t)base4, M, "
        f"(int)n, (int)k, dg, rw, rpart.data_ptr<float>(), rcount.data_ptr<int>(), "
        f"rout.data_ptr<float>()); break;"
        for i, c in enumerate(cfgs) if c[13])
    src = ns["_SK_SRC"]
    for marker in ("@@DISPATCH@@", "@@FDISPATCH@@", "@@INSTANCES@@"):
        if src.count(marker) != 1:
            raise SystemExit(f"marker {marker} found {src.count(marker)} times in _SK_SRC")
    src = (src.replace("@@DISPATCH@@", dispatch).replace("@@FDISPATCH@@", fdispatch)
           .replace("@@INSTANCES@@", ""))
    if "griddepcontrol" in ns["GEMV_CUDA_SRC"]:
        raise SystemExit("the GEMV source now carries griddepcontrol; revisit the layer's plain launch")
    check_sk_pdl(src)
    wd = ns["wd_source"]()
    if "griddepcontrol" in wd:
        raise SystemExit("the wide kernel now carries griddepcontrol; revisit the layer's plain launch")
    cpp = ns["_SK_CPP"] + ns["GEMV_CPP_DECLS"] + ns["_WD_CPP"]
    return src, ns["GEMV_CUDA_SRC"], wd, cpp, cfgs, ns["WD_GEOM"]


def check_sk_pdl(src):
    """The sk source may carry griddepcontrol only behind SK_PDL (default 0, so the layer's
    default build is unchanged): every non-comment use must sit inside an `if constexpr (PDL)`
    block opened on one of the three lines before it, with PDL false unless SK_PDL is set."""
    if "#ifndef SK_PDL\n#define SK_PDL 0\n#endif" not in src:
        if "griddepcontrol" in src:
            raise SystemExit("griddepcontrol in the sk source without the SK_PDL switch")
        return
    lines = src.splitlines()
    for i, line in enumerate(lines):
        code = line.split("//")[0]
        if "griddepcontrol" in code and not any("if constexpr (PDL)" in prev
                                                 for prev in lines[max(0, i - 3):i]):
            raise SystemExit(f"griddepcontrol outside an `if constexpr (PDL)` block: {line.strip()}")


def replace_once(src, pattern, repl, what):
    new, n = re.subn(pattern, repl, src, count=1, flags=re.S | re.M)
    if n != 1:
        raise SystemExit(f"no {what} in submission.py")
    return new


def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    text = src_path.read_text()
    ns, segments = load(text)
    sub = SUB.read_text()
    layer_slice = int(re.search(r"^UNPOOL_SLICE = (.*)$", sub, re.M).group(1).replace("HIDDEN // 8", "1024"))
    if layer_slice != ns["UNPOOL_SLICE"]:
        raise SystemExit("UNPOOL_SLICE differs between the layer and the source")
    code, gemv, wd, cpp, cfgs, wd_geom = device_code(ns)
    sub = replace_once(sub, r'_SKP_SRC = r""".*?"""', lambda _: f'_SKP_SRC = r"""\n{code}"""',
                       "_SKP_SRC block")
    sub = replace_once(sub, r'_SKP_GEMV_SRC = r""".*?"""',
                       lambda _: f'_SKP_GEMV_SRC = r"""\n{gemv}"""', "_SKP_GEMV_SRC block")
    sub = replace_once(sub, r'_SKP_CPP_SRC = r""".*?"""',
                       lambda _: f'_SKP_CPP_SRC = r"""\n#include <torch/extension.h>\n{cpp}"""',
                       "_SKP_CPP_SRC block")
    sub = replace_once(sub, r'_SKP_WD_SRC = r""".*?"""', lambda _: f'_SKP_WD_SRC = r"""\n{wd}"""',
                       "_SKP_WD_SRC block")
    wd_table = ("_SKP_WD_GEOM = {\n" + "".join(f"    {nm!r}: {g!r},\n" for nm, g in wd_geom.items())
                + "}")
    sub = replace_once(sub, r"^_SKP_WD_GEOM = \{.*?\n\}$", lambda _: wd_table, "_SKP_WD_GEOM table")
    table = "_SKP_CONFIGS = (\n" + "".join(f"    {c!r},\n" for c in cfgs) + ")"
    sub = replace_once(sub, r"^_SKP_CONFIGS = \(.*?\n\)$", lambda _: table, "_SKP_CONFIGS table")
    body = "\n\n\n".join(segments[n] for n in COPY_CONSTS + COPY_FUNCS)
    sub = replace_once(sub, r"# @SKP_PY_BEGIN@\n.*?# @SKP_PY_END@\n",
                       lambda _: "# @SKP_PY_BEGIN@\n" + body + "\n# @SKP_PY_END@\n", "@SKP_PY@ region")
    SUB.write_text(sub)
    lines = len(code.splitlines()) + len(gemv.splitlines()) + len(wd.splitlines())
    print(f"inlined {lines} device lines, {len(cfgs)} sk configs ({sum(1 for c in cfgs if c[13])} "
          f"fused), wide configs {list(wd_geom)} and {len(COPY_FUNCS)} functions from {src_path}")


if __name__ == "__main__":
    main()
