#!/bin/sh
# Compile the W4A16 GEMM extension in muse_moe_w4a16/submission.py (_w4_source(): _W4_SRC with
# one launcher per W4_CONFIGS entry, up to the end of its anonymous namespace) for H200
# (sm_90a) and print ptxas register, shared memory and spill counts per kernel. With a second
# argument, also dump SASS to that file. Needs Docker.
#   tools/moe_compile_w4.sh muse_moe_w4a16/submission.py [sass_out]
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import ast, re, sys
text = open(sys.argv[1]).read()
tree = ast.parse(text)
names = {"W4_CONFIGS", "_W4_SRC", "_w4_source"}
keep = [n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))]
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), "submission", "exec"), ns)
cuda = ns["_w4_source"]()
cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', cuda)
cuda = cuda.split("}  // namespace")[0]
prelude = ("#include <cstdio>\n#define C10_CUDA_CHECK(x) (void)(x)\n"
           "#define C10_CUDA_KERNEL_LAUNCH_CHECK() (void)0\n")
uses = re.findall(r"// w4_compile_check: (.*)", cuda)
cuda = prelude + cuda + "void compile_check_uses() {\n" + "\n".join(uses) + "\n}\n}  // namespace\n"
open(f"{sys.argv[2]}/w4.cu", "w").write(cuda)
PY

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c 'nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c /w/w4.cu \
        -o /w/w4.o 2>&1; cuobjdump -sass /w/w4.o > /w/w4.sass 2>&1 || true' \
    | grep -E 'error|warning|Compiling entry|registers|spill' | sed -E 's/ptxas info *: //'
if [ $# -ge 2 ]; then
    cp "$work/w4.sass" "$2"
fi
