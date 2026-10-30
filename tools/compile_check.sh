#!/bin/bash
# Compile every device kernel in muse_bf16_proj/submission.py (sk, cgemv/rgemv/sgemv, wd) for
# H200 (sm_90a) without the torch host wrappers and print ptxas register and spill counts per
# instantiation. Needs Docker. Exits nonzero when nvcc fails and then prints the log tail; the
# full ptxas log stays in $KEEP (default .build/compile_check.log).
#   tools/compile_check.sh [muse_bf16_proj/submission.py]
set -euo pipefail
src=${1:-muse_bf16_proj/submission.py}
keep=${KEEP:-.build/compile_check.log}
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

uv run --with torch python - "$src" "$work" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
open(f"{sys.argv[2]}/sk.cu", "w").write(mod.sk_source(device_only=True))
PY

rc=0
docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c 'cd /w && nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -diag-suppress 177 -Xptxas -v -c sk.cu -o /dev/null' \
    > "$work/ptxas.log" 2>&1 || rc=$?
cp "$work/ptxas.log" "$keep"
if [ "$rc" -ne 0 ]; then
    echo "nvcc failed (exit $rc); last 60 lines of $keep:"
    tail -n 60 "$keep"
    exit "$rc"
fi
grep -E 'error|warning|Compiling entry|registers|spill' "$keep" | sed -E 's/ptxas info *: //' || true
echo "nvcc ok; $(grep -c 'Compiling entry' "$keep") entries, full log in $keep"
