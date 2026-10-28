#!/bin/sh
# Compile the device code of every _*_SRC block in a submission for H200 (sm_90a) and print ptxas
# register and spill counts. Needs Docker. Each block is cut at its "}  // namespace" line, so
# the torch host wrappers are left out; templates are instantiated by an optional
# "// INSTANTIATE: ..." line inside the block.
#   tools/attn_compile_check.sh muse_attention_decode/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for name, body in re.findall(r'(_\w+_SRC) = r"""(.*?)"""', text, re.S):
    cuda = re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', body)
    head, sep, _ = cuda.rpartition("}  // namespace")
    if not sep:
        continue
    extra = "".join(line.split("// INSTANTIATE:", 1)[1].strip() + "\n"
                    for line in head.splitlines() if "// INSTANTIATE:" in line)
    open(f"{sys.argv[2]}/{name.strip('_').lower()}.cu", "w").write(head + extra + sep + "\n")
PY

for f in "$work"/*.cu; do
    echo "== $(basename "$f")"
    docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
        nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c \
        "/w/$(basename "$f")" -o /dev/null 2>&1 \
        | grep -E 'error|warning|Compiling entry|registers|spill|bytes smem' \
        | sed -E 's/ptxas info *: //'
done
