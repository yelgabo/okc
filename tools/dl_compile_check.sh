#!/bin/sh
# Compile muse_decode_layer/submission.py's whole extension (every block in its load_inline
# cuda_sources list, host wrappers included, as one translation unit like load_inline) for H200
# (sm_90a) with torch's headers and torch's default nvcc flags, and print ptxas register and
# spill counts and the wall time. Needs Docker and the okc-torch-headers image (CUDA 12.8 devel
# plus a CPU torch wheel, which ships the ATen CUDA headers):
#   docker build --platform linux/arm64 -t okc-torch-headers .build/torchimg
#   tools/dl_compile_check.sh muse_decode_layer/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
blocks = dict(re.findall(r'^(_\w+_SRC) = r"""(.*?)"""', text, re.S | re.M))
names = re.search(r"cuda_sources=\[([^\]]*)\]", text).group(1)
parts = [blocks[n.strip()] for n in names.split(",") if n.strip()]
head = "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
open(f"{sys.argv[2]}/ext.cu", "w").write(head + "\n".join(parts))
PY
# A CPU wheel lacks this CMake-generated header; CUDA builds of torch ship it with only comments.
mkdir -p "$work/stub/c10/cuda/impl"
echo "#pragma once" > "$work/stub/c10/cuda/impl/cuda_cmake_macros.h"

docker run --rm --platform linux/arm64 -v "$work:/w" okc-torch-headers sh -c '
inc=$(python3 -c "import torch.utils.cpp_extension as c; print(\" \".join(\"-I\" + p for p in c.include_paths()))")
inc="$inc -I/w/stub -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")"
start=$(date +%s)
nvcc -O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
    -DTORCH_EXTENSION_NAME=dl_check -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    -Xcompiler -fPIC $inc -c /w/ext.cu -o /w/ext.o 2>&1 \
    | grep -E "error|warning|Compiling entry|registers|spill|bytes smem" | grep -v "#3189-D" \
    | sed -E "s/ptxas info *: //"
echo "nvcc wall: $(( $(date +%s) - start )) s"'
