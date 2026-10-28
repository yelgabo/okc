#!/bin/sh
# Compile the whole fused MoE extension source from muse_moe_w4a16/submission.py (device code
# and the moe_run host wrapper) against stub torch headers, so a host-side mistake shows up
# before a judge run. Needs Docker. Device-side register counts: tools/moe_compile_fused.sh.
#   tools/moe_host_check.sh muse_moe_w4a16/submission.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/host.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/torch" "$work/ATen/cuda"
cat > "$work/torch/extension.h" <<'EOF'
#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
namespace at {
struct BFloat16 { uint16_t x; };
}
namespace torch {
struct Tensor {
    template <typename T> T* data_ptr() const { return nullptr; }
    void* data_ptr() const { return nullptr; }
    int64_t size(int) const { return 0; }
    int64_t numel() const { return 0; }
    int get_device() const { return 0; }
};
}
#define TORCH_CHECK(cond, ...) (void)(cond)
EOF
cat > "$work/ATen/cuda/CUDAContext.h" <<'EOF'
#pragma once
#include <cuda_runtime.h>
namespace at { namespace cuda { inline cudaStream_t getCurrentCUDAStream() { return 0; } } }
#define C10_CUDA_CHECK(x) (void)(x)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() (void)0
EOF
python3 - "$src" "$work" <<'PY'
import ast, sys
text = open(sys.argv[1]).read()
tree = ast.parse(text)
ns = {}
names = {"_CUDA_SRC", "_SKINNY_BLOCK", "SKINNY_VARIANTS", "MAIN_MODES", "VARIANT_MAIN0", "_CPP_SRC"}
funcs = {"_wgmma_code", "_cuda_source", "_skinny_tag"}
keep = [n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in funcs)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets))]
exec(compile(ast.Module(body=keep, type_ignores=[]), "submission", "exec"), ns)
# The declarations first, as load_inline's main.cpp sees them, then the CUDA source that
# must define them with the same signatures.
open(f"{sys.argv[2]}/moe.cu", "w").write(ns["_CPP_SRC"] + ns["_cuda_source"]())
PY
docker run --rm --platform linux/arm64 -v "$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c "cd /w && nvcc -O1 -std=c++17 -I/w -gencode=arch=compute_90a,code=sm_90a -c moe.cu \
        -o /dev/null 2>&1" | grep -iE "error|undefined" || echo "no errors"
