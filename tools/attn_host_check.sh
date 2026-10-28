#!/bin/sh
# Compile a .cu file whole (device and host code) against a stub torch header, to catch host
# wrapper errors that tools/attn_compile_check.sh cuts off. Needs Docker.
#   tools/attn_host_check.sh file.cu [file.cu ...]
set -eu
stub=.build/ws/stub
mkdir -p "$stub/torch" "$stub/ATen/cuda"
cat > "$stub/torch/extension.h" <<'EOF'
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <initializer_list>
namespace torch {
enum Dtype { kFloat32, kFloat8_e4m3fn };
struct Options { Options dtype(Dtype) const { return *this; } };
struct Tensor {
    template <typename T> T* data_ptr() { return nullptr; }
    void* data_ptr() { return nullptr; }
    int64_t size(int) const { return 0; }
    int64_t numel() const { return 0; }
    Options options() const { return {}; }
};
inline Tensor empty(std::initializer_list<int64_t>, Options) { return {}; }
}
#define TORCH_CHECK(cond, ...) (void)(cond)
EOF
cat > "$stub/ATen/cuda/CUDAContext.h" <<'EOF'
#pragma once
namespace at { namespace cuda { inline cudaStream_t getCurrentCUDAStream() { return 0; } } }
EOF
for f in "$@"; do
    cp "$f" "$stub/$(basename "$f")"
    echo "== $f"
    docker run --rm --platform linux/arm64 -v "$PWD/$stub:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
        sh -c "cd /w && nvcc -O3 -std=c++17 -I/w -gencode=arch=compute_90a,code=sm_90a -c \
            $(basename "$f") -o /dev/null 2>&1" | grep -iE "error" || echo "no errors"
done
