#!/bin/sh
# Compile muse_decode_layer/submission.py's fused MoE extension (_MOE1_SRC plus the host source
# _moe1_source() assembles, as load_inline gets them) for H200 (sm_90a) with torch's headers,
# and print ptxas register and spill counts. tools/dl_compile_check.sh covers only the main
# extension. Needs Docker and the okc-torch-headers image (see tools/dl_compile_check.sh).
#   tools/dl_moe_compile_check.sh muse_decode_layer/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
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
inc=$(python3 -c "import torch.utils.cpp_extension as c; print(\" \".join(\"-I\" + p for p in c.include_paths()))")
inc="$inc -I/w/stub -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")"
nvcc -O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
    -DTORCH_EXTENSION_NAME=dl_moe_check -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    -Xcompiler -fPIC $inc -c /w/ext.cu -o /w/ext.o > /w/nvcc.txt 2>&1
echo "nvcc exit $?"
grep -E "error|Compiling entry|registers|spill" /w/nvcc.txt | grep -v "#3189-D" \
    | sed -E "s/ptxas info *: //"'
