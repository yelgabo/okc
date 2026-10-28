#!/bin/sh
# Compile the kernels in muse_bf16_proj/submission.py (GEMV_CUDA_SRC) for H200 (sm_90a), print ptxas
# register and spill counts, and count the global loads each kernel issues after its first FFMA
# (the kernels are written to issue every load first; ptxas can undo that). Needs Docker. The
# image has no ATen, so GEMV_DEVICE_ONLY leaves out the torch entry points and the cgemv variants
# each layer uses are instantiated here; rgemv and sgemv are then unreferenced, hence the
# suppressed warning 177.
#   tools/gemv_compile_check.sh [muse_bf16_proj/submission.py]
set -eu
src=${1:-muse_bf16_proj/submission.py}
mkdir -p .build && work=$(mktemp -d "$PWD/.build/compile.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
cuda = re.search(r'GEMV_CUDA_SRC = r"""(.*?)"""', text, re.S).group(1)
extra = "".join(
    f"template void cgemv_launch<{nth}, {cpt}>(const uint4*, const int2*, int, unsigned, "
    f"const __nv_bfloat16*, __nv_bfloat16*, int, cudaStream_t);\n"
    for nth, cpt in ((256, 2), (64, 1), (512, 1), (128, 2)))
cuda = cuda.replace("}  // namespace", extra + "}  // namespace", 1)
open(f"{sys.argv[2]}/gemv_m1.cu", "w").write(cuda)
PY

docker run --rm --platform linux/arm64 --entrypoint sh -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 -c \
    'nvcc -O3 -std=c++17 -DGEMV_DEVICE_ONLY -diag-suppress 177 -gencode=arch=compute_90a,code=sm_90a \
     -Xptxas -v -cubin /w/gemv_m1.cu -o /w/gemv_m1.cubin > /w/ptxas.log 2>&1; \
     cuobjdump -sass /w/gemv_m1.cubin > /w/gemv_m1.sass 2>&1' || true
grep -E 'error|warning|Compiling entry|registers|spill' "$work/ptxas.log" | sed -E 's/ptxas info *: //' \
    | c++filt || true

python3 - "$work/gemv_m1.sass" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for body in re.split(r"\n\s*Function : ", text)[1:]:
    name = body.split("\n", 1)[0].strip()
    short = re.search(r"(cgemv_kernelILi\d+ELi\d+E|rgemv_kernel|sgemv_kernel)", name)
    ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_.]*)", body)
    first = next((i for i, o in enumerate(ops) if o.startswith("FFMA")), len(ops))
    loads = [i for i, o in enumerate(ops) if o.startswith("LDG")]
    late = sum(i > first for i in loads)
    print(f"{short.group(1) if short else name}: {len(loads)} global loads, {late} after the first FFMA"
          + ("" if late == 0 else "  <-- loads sunk below FMAs"))
PY
