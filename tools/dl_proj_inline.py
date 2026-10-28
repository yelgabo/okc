"""Copy the projection kernels from muse_bf16_proj/submission.py into muse_decode_layer/submission.py.

The source assembles its device code in sk_source(): the kernel template string _SK_SRC with
the config dispatch expanded from SK_CONFIGS, plus GEMV_CUDA_SRC. This tool runs that assembly
for the configs the layer uses (every SK_GEOM geometry, compressed and raw, and the two fused
pooler + router configs; the standalone router configs stay out), writes the result as
_SKP_SRC and _SKP_GEMV_SRC, the declarations as _SKP_CPP_SRC, the config table the dispatch was
generated from as _SKP_CONFIGS, and copies the packing code (the 12-bit code, tile units,
stage-major streams, escape tables, the balancing permutation, cgemv's row stream) and the
geometry helpers verbatim between the @SKP_PY@ markers, with the constants they read. The
kernel source itself is not patched: it has no griddepcontrol and the layer launches it plain.

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
               "FUSED_GEOM", "SK_USE", "SK_EXTRA", "SMEM_BUDGET", "SMEM_PER_SM", "UNPOOL_SLICE"}
BUILD_FUNCS = {"_sk_configs", "_sk_smem", "_fit_stages", "_sk_targs"}
# Copied verbatim, in this order (constants first).
COPY_CONSTS = ("EXP_WINDOW", "MAX_ESCAPE_DENOM", "BALANCE_GROUP", "BALANCED_LAYERS", "SK_USE",
               "FUSED_GEOM", "_LAYER_THREADS")
COPY_FUNCS = ("_sk_geom", "_fused_router_geom", "bf16_bits", "tile_values", "window_base",
              "code_bits", "raw_units", "hilo_units", "comp_units", "stage_stream", "balance_perm",
              "escape_table", "pack_layer", "slices_balanced", "pack_raw_layer", "pack_router",
              "pack_cgemv", "cgemv_threads", "bucket_of")


def load(text):
    tree = ast.parse(text)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in BUILD_FUNCS)
            or (isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") in BUILD_NAMES for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "proj_submission", "exec"), ns)
    segments = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in COPY_FUNCS:
            segments[n.name] = ast.get_source_segment(text, n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if getattr(t, "id", "") in COPY_CONSTS:
                    segments[t.id] = ast.get_source_segment(text, n)
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
    if "griddepcontrol" in src or "griddepcontrol" in ns["GEMV_CUDA_SRC"]:
        raise SystemExit("the source now carries griddepcontrol; revisit the layer's plain launch")
    return src, ns["GEMV_CUDA_SRC"], ns["_SK_CPP"] + ns["GEMV_CPP_DECLS"], cfgs


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
    code, gemv, cpp, cfgs = device_code(ns)
    sub = replace_once(sub, r'_SKP_SRC = r""".*?"""', lambda _: f'_SKP_SRC = r"""\n{code}"""',
                       "_SKP_SRC block")
    sub = replace_once(sub, r'_SKP_GEMV_SRC = r""".*?"""',
                       lambda _: f'_SKP_GEMV_SRC = r"""\n{gemv}"""', "_SKP_GEMV_SRC block")
    sub = replace_once(sub, r'_SKP_CPP_SRC = r""".*?"""',
                       lambda _: f'_SKP_CPP_SRC = r"""\n#include <torch/extension.h>\n{cpp}"""',
                       "_SKP_CPP_SRC block")
    table = "_SKP_CONFIGS = (\n" + "".join(f"    {c!r},\n" for c in cfgs) + ")"
    sub = replace_once(sub, r"^_SKP_CONFIGS = \(.*?\n\)$", lambda _: table, "_SKP_CONFIGS table")
    body = "\n\n\n".join(segments[n] for n in COPY_CONSTS + COPY_FUNCS)
    sub = replace_once(sub, r"# @SKP_PY_BEGIN@\n.*?# @SKP_PY_END@\n",
                       lambda _: "# @SKP_PY_BEGIN@\n" + body + "\n# @SKP_PY_END@\n", "@SKP_PY@ region")
    SUB.write_text(sub)
    print(f"inlined {len(code.splitlines()) + len(gemv.splitlines())} device lines, {len(cfgs)} sk "
          f"configs ({sum(1 for c in cfgs if c[13])} fused) and {len(COPY_FUNCS)} functions from {src_path}")


if __name__ == "__main__":
    main()
