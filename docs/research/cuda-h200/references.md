# Primary references and provenance

Recovered November 2, 2026. These links were opened separately from the uploaded text. The upload supplies citation tokens without their original URL mapping; none of those tokens has been silently assigned a recovered URL. These references verify mechanisms, not a claimed Muse speedup. No external code was adopted.

| Reference | What to use it for | Version or access scope |
|---|---|---|
| [Hopper tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html) | Occupancy, TMA, clusters, shared-memory limits | CUDA 12.8.1 archive, matches local compiler family |
| [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html) | Asynchronous copies, swizzles, streams, events and graphs | CUDA 12.8.1 archive |
| [PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html) | WGMMA constraints, proxy fences and completion | CUDA 12.8.2 archive; inspect the exact applicable instruction/protocol |
| [NVCC manual](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-compiler-driver-nvcc/index.html) | SM90a targets, resource reporting, register caps and math flags | CUDA 12.8.1 archive |
| [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/) | SKU capacity and bandwidth | Live product page; actual job device remains authoritative |
| [NVIDIA NIM architecture table](https://docs.nvidia.com/nim/large-language-models/latest/reward-model.html) | Distinguishes Blackwell NVFP4 from Hopper W4A16/BF16 | Live support matrix, model-specific deployment guidance |
| [CUTLASS grouped Hopper example](https://github.com/NVIDIA/cutlass/blob/v3.8.0/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu) | Grouped TMA warp-specialized cooperative/ping-pong implementation | v3.8.0 tag, not an immutable commit pin or matching Muse format; re-pin before code adoption |
| [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | Timeline, graph and stream capture options | Live documentation; confirm installed version |
| [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Roofline, sections, issue activity and stall interpretation | Live documentation; query available metrics/sections |
| [Compute Sanitizer guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | Memory, shared-race, initialization and synchronization tools | Live documentation; confirm instrumentation support |
| [PyTorch CUDA extensions](https://docs.pytorch.org/docs/2.10/cpp_extension.html) | `load_inline`, CUDA flags and host build mechanics | PyTorch 2.10 documentation; confirm target's actual Torch/standard |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608v2) | Hopper async overlap and warp specialization as an algorithm example | Shah et al., July 2024 v2; not evidence for this MoE workload |
| [Stream-K](https://arxiv.org/abs/2301.03598) | Work-centric decomposition and its load-balance rationale | Osama et al., January 2023; reductions still need full pricing |

The web search covered Hopper limits, NVFP4 compatibility, CUTLASS grouped GEMM and CUDA 12.8 PTX proxy ordering. Primary pages above were opened to recover usable references. This was source verification for the pasted document, not an independent leaderboard survey or a refresh of every competitor implementation. Exact public comparators remain pinned in their linked repository reports.
