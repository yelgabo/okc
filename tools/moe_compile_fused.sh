#!/bin/sh
# Compile the fused MoE kernel in muse_moe_w4a16/submission.py (_CUDA_SRC up to the end of its
# anonymous namespace) for H200 (sm_90a) and print ptxas register, shared memory and spill
# counts plus any C75xx wgmma warnings. With a second argument, also dump SASS to that file.
# Needs Docker. NVCC_EXTRA adds nvcc flags (for example --maxrregcount=112).
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
names = {"_CUDA_SRC", "_SKINNY_BLOCK", "SKINNY_VARIANTS", "MAIN_MODES", "VARIANT_MAIN0"}
funcs = {"_wgmma_code", "_cuda_source", "_skinny_tag"}
keep = [n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in funcs)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))]
exec(compile(ast.Module(body=keep, type_ignores=[]), "submission", "exec"), ns)
cuda = ns["_cuda_source"]()
uses = [f"  sk_{ns['_skinny_tag'](*v)}::launch<1>(p, 1, 0);" for v in ns["SKINNY_VARIANTS"]]
uses += [f"  launch_main<{m}>(p, 1, 0);" for m in ns["MAIN_MODES"]]
cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', cuda)
cuda = cuda[:cuda.rindex("}  // namespace\n")]
prelude = ("#include <cstdio>\n#include <algorithm>\n#define C10_CUDA_CHECK(x) (void)(x)\n"
           "#define C10_CUDA_KERNEL_LAUNCH_CHECK() (void)0\n")
cuda = (prelude + cuda + "void compile_check_uses() {\n  Params p = {};\n"
        + "\n".join(uses) + "\n}\n}  // namespace\n")
open(f"{sys.argv[2]}/moe.cu", "w").write(cuda)
PY

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c "nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v ${NVCC_EXTRA:-} -c /w/moe.cu \
        -o /w/moe.o 2>&1; cuobjdump -sass /w/moe.o > /w/moe.sass 2>&1 || true" \
    | grep -E 'error|warning|Compiling entry|registers|spill|C75' | sed -E 's/ptxas info *: //'
if [ $# -ge 2 ]; then
    cp "$work/moe.sass" "$2"
fi
