#!/bin/sh
# Compile the public entry's attention kernel (Apache-2.0 muse_attention_decode.py, CUDA_SRC) for
# sm_90a with the flags of its load_inline call plus the half/bf16 defines torch adds, print
# ptxas registers and spills per instance, and report the highest register each flat q_len 8
# region uses (the consumer region sits between USETMAXREG.TRY_ALLOC and DEALLOC). Needs Docker.
#   tools/attn_pub_compile.sh path/to/muse_attention_decode.py
set -eu
src=$1
work=.build/pub
mkdir -p "$work"
python3 - "$src" "$work/pub.cu" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
cuda = re.search(r'CUDA_SRC = r"""(.*?)"""', text, re.S).group(1)
head = cuda.split("template <int L>\nvoid launch(")[0]
head = re.sub(r'#include <(torch|ATen|c10)/[^>]*>\n', '', head)
# The kernels live in an anonymous namespace: instantiate them inside it.
body, sep, tail = head.rpartition("}  // namespace")
inst = "".join(f"template __global__ void attn_kernel<{l}, {f}, {m}>(const __grid_constant__ Params);\n"
               for l, f, m in [(8, "true", 0), (1, "true", 0), (8, "false", 1), (8, "false", 3),
                               (8, "false", 7), (8, "false", 15), (1, "false", 1),
                               (1, "false", 13)])
open(sys.argv[2], "w").write(body + inst + sep + tail)
PY
docker run --rm --platform linux/arm64 -v "$PWD/$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c "nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
        -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
        -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -Xptxas -v -cubin \
        /w/pub.cu -o /w/pub.cubin > /w/ptxas.txt 2>&1; cuobjdump -sass /w/pub.cubin > /w/pub.sass" \
    > /dev/null 2>&1
grep -E 'error|Compiling entry|registers|spill' "$work/ptxas.txt" \
    | sed -E 's/.*attn_kernelILi([0-9]+)ELb([01])ELi([0-9]+)E.*/attn_kernel<\1, \2, \3>/; s/ptxas info *: //'
python3 - "$work/pub.sass" <<'PY'
import re, sys
for f in re.split(r'\n\s+Function : ', open(sys.argv[1]).read())[1:]:
    name = f.split("\n", 1)[0]
    if "ILi8ELb1" not in name:
        continue
    inside, top_in, top_out = False, 0, 0
    for line in f.split("\n"):
        if "TRY_ALLOC" in line:
            inside = True
            continue
        if "DEALLOC" in line:
            inside = False
        regs = [int(r) for r in re.findall(r"\bR(\d+)\b", line)]
        if regs:
            if inside:
                top_in = max(top_in, max(regs))
            else:
                top_out = max(top_out, max(regs))
    print(f"attn_kernel<8, 1, 0>: highest register in the consumer region R{top_in}, elsewhere R{top_out}")
PY
