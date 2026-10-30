#!/bin/sh
# Ordering-relevant SASS opcode counts per kernel of one of muse_decode_layer/submission.py's
# extensions: the fused MoE (default, as tools/dl_moe_compile_check.sh builds it) or, with
# second argument one, the one-launch attention (_ONE_SRC + _ONE_HOST_SRC), for comparing a
# layer copy with another or with its standalone source after an inline or a layer patch: fences, membars, mbarrier ops (SYNCS), barriers, bulk copies and prefetches
# (UBLKCP, UBLKPF), ACQBULK (griddepcontrol.wait), local memory and the instruction count.
#   [SASS_OUT=file] sh tools/dl_moe_sass_counts.sh muse_decode_layer/submission.py [moe|w4|one|one1|skp|skp1]
set -eu
src=$1
which=${2:-moe}
mkdir -p .build && work=$(mktemp -d "$PWD/.build/sass.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" "$which" <<'PY'
import ast, sys
text = open(sys.argv[1]).read()
tree = ast.parse(text)
want = {"_W4_SRC", "W4_CONFIGS", "_MOE1_SRC", "_MOE1_HOST_SRC", "_MOE1_VARIANTS", "_ONE_SRC", "_ONE_HOST_SRC", "_SKP_SRC",
        "_SKP_GEMV_SRC", "_SKP_WD_SRC"}
keep = [n for n in tree.body
        if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in want for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name in ("_moe1_source", "_w4_source"))]
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), "layer", "exec"), ns)
head = "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
if sys.argv[3] == "w4":
    body = ns["_w4_source"]()
elif sys.argv[3] in ("one", "one1"):
    # one1: the attention as PORT_SK_PDL builds it (-DONE_TRIG=1).
    body = (("#define ONE_TRIG 1\n" if sys.argv[3] == "one1" else "") + ns["_ONE_SRC"] + "\n"
            + ns["_ONE_HOST_SRC"])
elif sys.argv[3].startswith("skp"):
    # skp1: the projections as PORT_SK_PDL builds them (-DSK_PDL=1).
    body = (("#define SK_PDL 1\n" if sys.argv[3] == "skp1" else "") + ns["_SKP_SRC"] + "\n"
            + ns["_SKP_GEMV_SRC"] + "\n" + ns["_SKP_WD_SRC"])
else:
    body = ns["_MOE1_SRC"] + "\n" + ns["_moe1_source"]()
open(f"{sys.argv[2]}/ext.cu", "w").write(head + body)
PY
mkdir -p "$work/stub/c10/cuda/impl"
echo "#pragma once" > "$work/stub/c10/cuda/impl/cuda_cmake_macros.h"

docker run --rm --platform linux/arm64 -v "$work:/w" okc-torch-headers sh -c '
inc=$(python3 -c "import torch.utils.cpp_extension as c; print(\" \".join(\"-I\" + p for p in c.include_paths()))" 2>/dev/null)
inc="$inc -I/w/stub -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")"
nvcc -O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
    -DTORCH_EXTENSION_NAME=dl_moe_check -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    -Xcompiler -fPIC $inc -c /w/ext.cu -o /w/ext.o > /w/nvcc.txt 2>&1 || { cat /w/nvcc.txt; exit 1; }
cuobjdump -sass /w/ext.o > /w/ext.sass' >/dev/null 2>&1
if [ -n "${SASS_OUT:-}" ]; then cp "$work/ext.sass" "$SASS_OUT"; fi
awk '/Function :/ { f = $3; fs[f] = 1 }
     /^ *\/\*[0-9a-f]+\*\/ / && !/ NOP/ { n[f]++ }
     /FENCE.VIEW.ASYNC.S/ { fs_[f]++ }
     /FENCE.VIEW.ASYNC.G/ { fg[f]++ }
     /MEMBAR/ { mb[f]++ }
     /SYNCS/ { sy[f]++ }
     /BAR.SYNC|BAR.ARV|BAR.RED/ { ba[f]++ }
     /ERRBAR/ { eb[f]++ }
     /CCTL/ { cc[f]++ }
     /UBLKCP/ { bc[f]++ }
     /UBLKPF/ { bp[f]++ }
     /ACQBULK/ { aq[f]++ }
     /ATOMG/ { ag[f]++ }
     /ATOMS/ { as[f]++ }
     /REDG/ { rg[f]++ }
     /LDGDEPBAR/ { ld[f]++ }
     /LDL/ { ll[f]++ }
     /STL/ { sl[f]++ }
     END { for (k in fs) printf "%s instr %d FENCE.S %d FENCE.G %d MEMBAR %d SYNCS %d BAR %d ERRBAR %d CCTL %d UBLKCP %d UBLKPF %d ACQBULK %d ATOMG %d ATOMS %d REDG %d LDGDEPBAR %d LDL %d STL %d\n",
                                 k, n[k], fs_[k], fg[k], mb[k], sy[k], ba[k], eb[k], cc[k], bc[k], bp[k], aq[k], ag[k], as[k], rg[k], ld[k], ll[k], sl[k] }' \
    "$work/ext.sass" | sed -E 's/_ZN[0-9]+_GLOBAL__N__[a-z0-9_]*_[0-9]+//' | sort
