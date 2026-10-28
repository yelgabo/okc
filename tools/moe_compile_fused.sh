#!/bin/sh
# Compile the fused MoE kernel in muse_moe_w4a16/submission.py (_CUDA_SRC up to the end of its
# anonymous namespace) for H200 (sm_90a) and print ptxas register, shared memory and spill
# counts plus any C75xx wgmma warnings. With a second argument, also dump SASS to that file.
# Needs Docker.
#   tools/moe_compile_fused.sh muse_moe_w4a16/submission.py [sass_out]
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import ast, re, sys
text = open(sys.argv[1]).read()
tree = ast.parse(text)
ns = {}
for n in tree.body:
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_CUDA_SRC" for t in n.targets):
        exec(compile(ast.Module(body=[n], type_ignores=[]), "submission", "exec"), ns)
cuda = ns["_CUDA_SRC"]
cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', cuda)
cuda = cuda.split("}  // namespace")[0]
prelude = ("#include <cstdio>\n#include <algorithm>\n#define C10_CUDA_CHECK(x) (void)(x)\n"
           "#define C10_CUDA_KERNEL_LAUNCH_CHECK() (void)0\n")
cuda = (prelude + cuda + "void compile_check_uses() {\n  Params p = {};\n"
        "  launch<0>(p, 1, 0);\n  launch<1>(p, 1, 0);\n}\n}  // namespace\n")
open(f"{sys.argv[2]}/moe.cu", "w").write(cuda)
PY

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c 'nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c /w/moe.cu \
        -o /w/moe.o 2>&1; cuobjdump -sass /w/moe.o > /w/moe.sass 2>&1 || true' \
    | grep -E 'error|warning|Compiling entry|registers|spill|C75' | sed -E 's/ptxas info *: //'
if [ $# -ge 2 ]; then
    cp "$work/moe.sass" "$2"
fi
