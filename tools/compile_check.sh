#!/bin/sh
# Compile the skinny kernel in muse_bf16_proj/submission.py for H200 (sm_90a) without the torch
# host wrappers and print ptxas register and spill counts per instantiation. Needs Docker.
#   tools/compile_check.sh [muse_bf16_proj/submission.py]
set -eu
src=${1:-muse_bf16_proj/submission.py}
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

uv run --with torch python - "$src" "$work" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
open(f"{sys.argv[2]}/sk.cu", "w").write(mod.sk_source(device_only=True))
PY

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c 'cd /w && time nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c sk.cu -o /dev/null' 2>&1 \
    | grep -E 'error|warning|Compiling entry|registers|spill|real' | sed -E 's/ptxas info *: //'
