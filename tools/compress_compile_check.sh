#!/bin/sh
# tools/compile_check.sh for _PROJ_SRC, plus instances of the compressed kernel (cproj_launch,
# both the multiply and the DUMP decode) that compile_check.sh does not instantiate. Prints
# ptxas register, shared memory and spill counts for sm_90a. Needs Docker.
#   tools/compress_compile_check.sh muse_bf16_proj/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
cuda = re.search(r'_PROJ_SRC = r"""(.*?)"""', text, re.S).group(1)
cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', cuda)
cuda = cuda.split('// Grids: up to 8 tokens')[0]
extra = "".join(
    f"template void cproj_launch<{s}, {d}>(const uint4*, const unsigned*, const uint4*, unsigned, "
    f"const __nv_bfloat16*, __nv_bfloat16*, uint4*, int, int, int, cudaStream_t);\n"
    for s in (2, 4, 8) for d in ("false", "true"))
# The bench variants (accs, exc_select, nodecode) of tools/compress_make_bench.py.
extra += "".join(
    f"template void cproj_launch<4, false, {a}, {e}, {nd}>(const uint4*, const unsigned*, "
    f"const uint4*, unsigned, const __nv_bfloat16*, __nv_bfloat16*, uint4*, int, int, int, "
    f"cudaStream_t);\n"
    for a, e, nd in ((1, "false", "false"), (1, "true", "false"), (4, "true", "false"),
                     (8, "true", "false"), (2, "true", "true")))
extra += "".join(
    f"template void proj_small<{s}>(const uint4*, const __nv_bfloat16*, __nv_bfloat16*, int, int, "
    f"int, cudaStream_t);\n" for s in (2, 4, 8))
cuda = cuda.replace("}  // namespace", extra + "}  // namespace", 1)
open(f"{sys.argv[2]}/proj_src.cu", "w").write(cuda)
PY

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c /w/proj_src.cu \
    -o /dev/null 2>&1 | grep -E 'error|warning|Compiling entry|registers|spill' \
    | sed -E 's/ptxas info *: //'
