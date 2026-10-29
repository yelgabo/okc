#!/bin/sh
# Ordering-relevant SASS opcode counts per kernel of muse_decode_layer/submission.py's fused MoE
# extension (as tools/dl_moe_compile_check.sh builds it), for comparing two layer copies after
# a layer patch: fences, membars, mbarrier ops (SYNCS), barriers, bulk copies and prefetches
# (UBLKCP, UBLKPF), ACQBULK (griddepcontrol.wait), local memory and the instruction count.
#   sh tools/dl_moe_sass_counts.sh muse_decode_layer/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/sass.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import ast, sys
text = open(sys.argv[1]).read()
tree = ast.parse(text)
want = {"_MOE1_SRC", "_MOE1_HOST_SRC", "_MOE1_VARIANTS"}
keep = [n for n in tree.body
        if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in want for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == "_moe1_source")]
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), "layer", "exec"), ns)
head = "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
open(f"{sys.argv[2]}/ext.cu", "w").write(head + ns["_MOE1_SRC"] + "\n" + ns["_moe1_source"]())
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
     /LDL/ { ll[f]++ }
     /STL/ { sl[f]++ }
     END { for (k in fs) printf "%s instr %d FENCE.S %d FENCE.G %d MEMBAR %d SYNCS %d BAR %d ERRBAR %d CCTL %d UBLKCP %d UBLKPF %d ACQBULK %d LDL %d STL %d\n",
                                 k, n[k], fs_[k], fg[k], mb[k], sy[k], ba[k], eb[k], cc[k], bc[k], bp[k], aq[k], ll[k], sl[k] }' \
    "$work/ext.sass" | sed -E 's/_ZN[0-9]+_GLOBAL__N__[a-z0-9_]*_[0-9]+//' | sort
