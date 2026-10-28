#!/bin/sh
# Compile the router kernel in a submission's _CUDA_SRC for H200 (sm_90a) and print ptxas register
# and spill counts. Needs Docker; the host wrapper (torch headers) is left out, so only the
# kernels are checked.
#   tools/compile_check.sh muse_bf16_proj/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work/kernel.cu" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
cuda = re.search(r'_CUDA_SRC = r"""(.*?)"""', text, re.S).group(1)
cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', cuda)
# Drop the torch host wrapper, keep the kernels, and instantiate every template size.
cuda = cuda.split('torch::Tensor ')[0]
sizes = (1, 2, 4, 8, 16, 32)
inst = ''.join(
    f'template __global__ void router_kernel<{s}>(const float*, const __nv_bfloat16*, float*, int);\n'
    for s in sizes)
open(sys.argv[2], 'w').write(cuda + inst)
EOF

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c /w/kernel.cu -o /w/kernel.o 2>&1 \
    | grep -E 'error|warning|Compiling entry|registers|spill'
