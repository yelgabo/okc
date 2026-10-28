#!/bin/sh
# Compile the previous kernel's W4 GEMM path as ported into muse_moe_w4a16/submission.py
# (_LEGACY_W4_SRC with one launcher per LEGACY_CONFIGS entry, and the route_align source) for
# H200 (sm_90a) and print ptxas register, shared memory and spill counts. Needs Docker.
#   tools/moe_compile_legacy.sh muse_moe_w4a16/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import ast, re, sys
text = open(sys.argv[1]).read()
tree = ast.parse(text)
names = {"LEGACY_CONFIGS", "_LEGACY_W4_SRC", "_LEGACY_ROUTE_SRC"}
keep = [n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_legacy_w4_source")
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))]
ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), "submission", "exec"), ns)
prelude = ("#include <cstdio>\n#include <vector>\n#define C10_CUDA_CHECK(x) (void)(x)\n"
           "#define C10_CUDA_KERNEL_LAUNCH_CHECK() (void)0\n")
w4 = ns["_legacy_w4_source"]()
w4 = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', w4)
w4 = w4.split("}  // namespace")[0]
uses = re.findall(r"// w4_compile_check: (.*)", w4)
w4 = prelude + w4 + "void compile_check_uses() {\n" + "\n".join(uses) + "\n}\n}  // namespace\n"
open(f"{sys.argv[2]}/w4.cu", "w").write(w4)
route = ns["_LEGACY_ROUTE_SRC"]
route = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', route)
route = route.split("}  // namespace")[0]
uses = re.findall(r"// compile_check: (.*)", route)
route = prelude + route + "void compile_check_uses() {\n" + "\n".join(uses) + "\n}\n}  // namespace\n"
open(f"{sys.argv[2]}/route.cu", "w").write(route)
PY

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c 'cd /w && for f in w4 route; do nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c $f.cu -o $f.o 2>&1; done' \
    | grep -E 'error|warning|Compiling entry|registers|spill' | sed -E 's/ptxas info *: //'
