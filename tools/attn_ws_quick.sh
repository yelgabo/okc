#!/bin/sh
# Compile muse_attention_decode/attn_ws.cu (or the file given) alone for sm_90a, device code
# only, and print ptxas diagnostics; with -s also dump the SASS of attn_ws_kernel<8, 0>.
#   tools/attn_ws_quick.sh [-s] [file.cu]
set -eu
sass=0
if [ "${1:-}" = "-s" ]; then
    sass=1
    shift
fi
src=${1:-muse_attention_decode/attn_ws.cu}
mkdir -p .build/ws
python3 - "$src" .build/ws/k.cu <<'PY'
import re, sys
body = open(sys.argv[1]).read()
cuda = "#define TORCH_CHECK(cond, ...) (void)(cond)\n" + re.sub(
    r'#include <(torch|ATen)/[^>]*>\n', '', body)
head, sep, _ = cuda.rpartition("}  // namespace")
extra = "".join(line.split("// INSTANTIATE:", 1)[1].strip() + "\n"
                for line in head.splitlines() if "// INSTANTIATE:" in line)
open(sys.argv[2], "w").write(head + extra + sep + "\n")
PY
docker run --rm --platform linux/arm64 -v "$PWD/.build/ws:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c "nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -cubin \
        /w/k.cu -o /w/k.cubin 2>&1 | grep -E 'error|C7514|Compiling entry|registers|spill' \
        | sed -E 's/ptxas info *: //'; if [ $sass = 1 ]; then cuobjdump -sass /w/k.cubin \
        > /w/k.sass; fi"
