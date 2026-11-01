# Local CUDA12.8 sm90a cross-compilation gate

November1. Parent-requested CPU-only setup; evidence branch parent `ec56698`.
**Actual NVRTC/PTXAS compilation succeeds for both byte and SC6 WGS kernels.**
No CUDA driver call, GPU execution, H200 quota, correctness result or speedup claim.

## Compiler and source pins

Installed under `/root/.cache/okc-cuda128` in WSL Ubuntu, with no global/system CUDA
installation or modification of the existing Torch environment:

- Official NVIDIA PyPI `nvidia-cuda-nvrtc-cu12==12.8.93`.
- `nvidia-cuda-nvcc-cu12==12.8.93`, which supplies actual PTXAS12.8.93 and NVVM
  headers/libdevice; this wheel does **not** supply the nvcc executable.
- `nvidia-cuda-runtime-cu12==12.8.90` and `nvidia-cuda-cccl-cu12==12.8.90`.

All four wheel filenames and SHA256 pins are archived in
`muse_moe_w4a16/profiles/local-cuda128.2026-11-01/wheels.sha256`.
The nvrtc/runtime/nvcc packages live under `packages/`; CCCL under `cccl/`, avoiding
pip namespace-directory overwrite. Downloaded wheels remain under `wheels/`.

Input is the exact parent-emitted `.build/today/g1-wgs.cu`, raw CRLF SHA256
`443c62ddcae2a0e1d5235b0aeabc9a3ce1531669d3593f9d4c1131806e8b2338`;
LF-normalized SHA256`c5ececf50cb8893ca6189cda42a051228b4a8576faf71142cccfa0b5917c3fa4`.
It is exactly the tracked G1b2 `MOE_CUDA_SRC` literal1540fe13... plus the reviewed
80-byte `nj==0` guard; independently checked a unified diff with no other change.
The source derives from the attributed keell0renz fork, upstream8e30e1be; this
tool does not change production source or the fork's attribution/license policy.

## Explicit compilation adapters

`tools/local_cuda128_compile.py` preserves the whole device prefix and drops only
the host tail beginning at `#include <torch/extension.h>`. It instantiates both
actual `okc::moe_fused<false>` and `<true>` through NVRTC name expressions. It does
not replace them with toy wrappers or skip the WGMMA algorithm.

Two virtual standard headers replace the unavailable host standard-library
environment: LP64 integer typedefs in `stdint.h`, and `std::integral_constant` /
`true_type` / `false_type` in `type_traits`. Source inspection asserts these are
the only `std::` names in the device prefix. CUDA BF16/runtime/CCCL headers remain
the genuine pinned NVIDIA headers. All virtual header text is in each manifest.
`--device-as-default-execution-space` handles NVRTC's rejection of unannotated
constexpr helper/lambda declarations; it changes no arithmetic or barriers.

Without `--forward-declare-wgs`, NVRTC12.8 rejects the late declaration of
`wgs_cta_of` at the dependent template call. The explicit opt-in adapter inserts
only its matching declaration before `wgs_phase`; its existing definition is
unchanged. This is required for this NVRTC frontend, **not** evidence that the
judge's nvcc13 build fails. No instruction, fence, loop, cost, layout or math body
is altered. Original and compiled device-prefix hashes are separately recorded.

Flags are compute90a/C++17/lineinfo, MAIN NW16/FAST1=0/EPI4=0/WGS1/NS3/NSF3,
PRMTSC1 and SC6W0/1. No fast math or external register cap was added. The original
`__launch_bounds__(NTL,1)` lowers to `.maxntid576,1,1` and `.minnctapersm1` in both
actual entries. PTXAS is run with `-arch=sm_90a -O3 -v --warn-on-spills`.

## Actual lowered resources

|Layout|Actual entry template|Registers|Stack bytes|Spill stores/loads bytes|Barriers|
|---|---|---:|---:|---:|---:|
|byte SC6W0|moe_fused false|96|0|0/0|16|
|byte SC6W0|moe_fused true|96|0|0/0|16|
|SC6 SC6W1|moe_fused false|96|0|0/0|16|
|SC6 SC6W1|moe_fused true|96|0|0/0|16|

Both builds retain PTXAS **C7513**, warning that non-WGMMA instructions defining
input registers serialize WGMMA pipeline stages. This is compiler evidence, not
a measured latency attribution. Dynamic shared memory is externally supplied;
zero static shared reporting does not mean zero launch shared memory. Runtime
occupancy, cooperative admission, graph capture, math and stream ordering still
require the actual H200 checks. These compiler12.8 numbers need not equal nvcc13.

Manifests include actual lowered names, source/PTX/cubin hashes, flags, adapter
text, status and parsed resources. Logs and manifests are committed under
`muse_moe_w4a16/profiles/local-cuda128.2026-11-01/`. PTX and cubins remain local in
`.build/cuda128/{byte,sc6}/`; no binary is inserted into a submitted candidate.

## Reuse and stop criteria

From the owned worktree in WSL:

```sh
/root/.cache/okc-host-validate/bin/python tools/local_cuda128_compile.py \
  ../okc/.build/today/g1-wgs.cu .build/cuda128/sc6 \
  --sc6 1 --forward-declare-wgs
# Repeat in a separate output directory with --sc6 0 for byte control.
```

Extra reviewed `-D` flags may be supplied as `--define=-DNAME=VALUE`; the full list
is recorded. This script is currently WGS-specific, not a general attention
extension compiler. To recreate dependencies, use `pip download --only-binary=:all:`
with the four exact versions above, verify every archived wheel SHA256, then
`pip install --no-index --find-links ... --target ...` with the package directories
described above. No system CUDA or local GPU capability is a prerequisite.

Use this gate on the next exact emitted control/diagnostic bodies before spending
H200 checks. Stop on compiler failure, absent lowered entry/resources or unexpected
spill/resource increase. A pass never substitutes for production correctness,
residency or timing gates. Future adaptations beyond headers, host-tail removal
and matching declaration require a new disclosure/review; do not silently simplify
the algorithm until compilation succeeds.
