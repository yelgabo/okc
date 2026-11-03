# Hardware and numeric rules

Source sections: architecture, numeric formats and compiler flags. Use [catalog.json](catalog.json) for exact line ranges and [references.md](references.md) for the recovered primary sources.

## H200 is Hopper

Target SM90a for the architecture-specific instructions used here. H200 adds memory capacity and bandwidth relative to H100, rather than a native FP4 instruction path. The uploaded 141 GB and 4.8 TB/s figures describe H200 SXM. Record the actual device, clocks and power settings before applying those peaks. Peaks and the task's stated floors are sizing assumptions, not measured performance limits.

The Hopper tuning guide lists 64 resident warps, 65,536 32-bit registers and 228 KB shared memory per SM, with up to 255 registers per thread. Available shared memory per block has its own limit and opt-in requirements. Optimize useful issued work and latency. One CTA per SM can be the right result. Distinguish static shared memory, launch-time dynamic shared memory, stack bytes and actual spills.

Use Hopper TMA, WGMMA, clusters and distributed shared memory terminology. A research mention of TMA2, tensor memory or Blackwell FP4 instructions does not identify a usable H200 mechanism. See [NVIDIA's Hopper guide](https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html).

## Our exact format contract

Read [the MoE task](../../../muse_moe_w4a16/TASK.md), [the layer task](../../../muse_decode_layer/TASK.md), and the selected source before changing math. The workload uses top-8 of 256 experts, 32 local experts, hidden width 4096, E2M1 codes with E4M3 group16 scales, and separate FP32 global scales. GEMM1 produces gate and up, GEMM2 produces the expert output. Global scales apply after the FP32 dot product and before the required BF16 boundary.

Preserve independently rounded scaled BF16 gate and up, FP32 SiLU/product followed by BF16 inter, scaled BF16 down output before gamma, and the layer's BF16-rounded effective gamma. Compute each expert's full-width RMS with epsilon `1e-8` before route-weighted FP32 combination and final BF16. The standalone and layer gamma interfaces differ. [The existing repair inventory](../../../tools/g_native_patch_ladder.py) identifies required boundaries and conservative implementation choices separately.

Hopper executes this NVFP4 workload by decoding to BF16 operands during the kernel. The selected WGS implementation decodes weights into register A for RS-WGMMA. It need not materialize a decoded shared-memory weight tile. See [NVIDIA's architecture compatibility table](https://docs.nvidia.com/nim/large-language-models/latest/reward-model.html) and [our selected-path source study](../../steering/2026-11-02-research-mainloop-overlap-decision.md).

Persistent BF16-expanded expert weights violate the challenge's storage rule. FP8 hardware support does not authorize replacing these BF16 MoE activation boundaries. A predecoded-BF16 diagnostic is only a separately disclosed laboratory comparison, never an eligible prepared-state implementation. Packing and lossless SC6 remain subject to scale eligibility and fallback checks.

## Compiler choices

Use the exact build's SM target, language standard and math flags. Our local device checks use CUDA 12.8.1; recent target receipts use a different toolkit. PyTorch host headers may require C++20 even when extracted device code accepts C++17. Report that distinction.

Use `-lineinfo` for a disclosed profiling build and `-Xptxas=-v` for resource diagnostics. Do not silently change matched build flags. Treat register caps and fast math as experiments with spill and numerical gates. A shorter PTX listing or higher occupancy does not establish lower latency. [The NVCC reference](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-compiler-driver-nvcc/index.html) documents those options.
