#!/bin/sh
# Compile the device code in muse_moe_w4a16/submission.py (_CUDA_SRC up to the end of its
# anonymous namespace) for H200 (sm_90a) and print ptxas register and spill counts. Needs
# Docker. The torch host wrappers are left out, so this checks device code only.
#   tools/moe_compile_check.sh muse_moe_w4a16/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
cuda = re.search(r'_CUDA_SRC = r"""(.*?)"""', text, re.S).group(1)
cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', cuda)
cuda = cuda.split("}  // namespace")[0]
prelude = ("#include <cstdio>\n#define C10_CUDA_CHECK(x) (void)(x)\n"
           "#define C10_CUDA_KERNEL_LAUNCH_CHECK() (void)0\n")
# Internal-linkage templates compile only when used, so reference every instance.
uses = re.findall(r"// compile_check: (.*)", text)
cuda = prelude + cuda + "void compile_check_uses() {\n" + "\n".join(uses) + "\n}\n}  // namespace\n"
open(f"{sys.argv[2]}/moe.cu", "w").write(cuda)
PY

keep=${KEEP:-.build/moe_compile_check.log}
mkdir -p "$(dirname "$keep")"
status=0
docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c /w/moe.cu \
    -o /dev/null > "$keep" 2>&1 || status=$?
if [ "$status" -ne 0 ]; then
    cat "$keep"
    exit "$status"
fi
grep -E 'error|warning|Compiling entry|registers|spill|C75' "$keep" \
    | sed -E 's/ptxas info *: //' || true
printf 'Full compiler log: %s\n' "$keep"
