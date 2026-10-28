#!/bin/sh
# Compile the kernels in muse_bf16_proj/submission.py (_CUDA_SRC and _PROJ_SRC) for H200
# (sm_90a) and print ptxas register and spill counts. Needs Docker. The torch host wrappers are
# left out, so this checks device code only.
#   tools/compile_check.sh muse_bf16_proj/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
inst = {
    "_CUDA_SRC": "".join(
        f"template void launch<{s}>(const float*, const __nv_bfloat16*, float*, int, cudaStream_t);\n"
        for s in (1, 2, 4, 8, 16, 32)),
    "_PROJ_SRC": "".join(
        [f"template void proj_small<{s}>(const uint4*, const __nv_bfloat16*, "
         f"__nv_bfloat16*, int, int, int, cudaStream_t);\n" for s in (2, 4, 8)]
        + [f"template void proj_large<{w}, {r}, {s}>(const uint4*, const __nv_bfloat16*, "
           f"__nv_bfloat16*, int, int, int, cudaStream_t);\n"
           for w, r, s in ((3, 3, 8), (4, 4, 8), (4, 4, 4))]
        + [f"template void router_mma<{mt}>(const uint4*, const __nv_bfloat16*, float*, int, "
           f"cudaStream_t);\n" for mt in (2, 4)]),
}
for name, extra in inst.items():
    m = re.search(name + r' = r"""(.*?)"""', text, re.S)
    if not m:
        continue
    cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', m.group(1))
    cuda = cuda.split('torch::Tensor ')[0]
    # Instantiate inside the anonymous namespace when the source has one.
    if "}  // namespace" in cuda:
        cuda = cuda.replace("}  // namespace", extra + "}  // namespace", 1)
    else:
        cuda += extra
    open(f"{sys.argv[2]}/{name.strip('_').lower()}.cu", "w").write(cuda)
PY

for f in "$work"/*.cu; do
    echo "== $(basename "$f")"
    docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
        nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c "/w/$(basename "$f")" \
        -o /dev/null 2>&1 | grep -E 'error|warning|Compiling entry|registers|spill' \
        | sed -E 's/ptxas info *: //'
done
