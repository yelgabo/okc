#!/bin/bash
# Device-only compile of muse_bf16_proj/submission.py (nvcc 12.8, sm_90a, Docker) to a SASS dump
# plus the ptxas log, for per-kernel SASS identity checks (tools/proj_sass_diff.py).
#   tools/proj_sass.sh SUBMISSION.py OUT_PREFIX     -> OUT_PREFIX.sass, OUT_PREFIX.ptxas
set -euo pipefail
src=$1
out=$2
mkdir -p .build && work=$(mktemp -d "$PWD/.build/sass.XXXXXX")
trap 'rm -rf "$work"' EXIT

uv run --with torch python - "$src" "$work" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
open(f"{sys.argv[2]}/sk.cu", "w").write(mod.sk_source(device_only=True))
PY

docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c 'cd /w && nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -diag-suppress 177 -Xptxas -v -cubin sk.cu \
           -o sk.cubin > ptxas.log 2>&1; rc=$?; [ $rc -eq 0 ] && cuobjdump -sass sk.cubin > sk.sass; exit $rc' \
    || { tail -n 60 "$work/ptxas.log"; exit 1; }
cp "$work/sk.sass" "$out.sass"
cp "$work/ptxas.log" "$out.ptxas"
echo "wrote $out.sass ($(grep -c 'Function :' "$out.sass") functions), $out.ptxas"
