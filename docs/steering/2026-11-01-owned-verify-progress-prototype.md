# Owned verify51 cooperative admission and fatal-timeout prototype

A **separate, unqualified source epoch** now repairs the two issues identified in [the launch-progress addendum](2026-11-01-owned-verify-progress-addendum.md). Frozen public/owned/G files and the pure-comment candidates remain unchanged. No GPU, official check, public action or queue update occurred. Independent review is required before root adopts these pins.

## Bounded change and provenance

`tools/verify_owned_progress.py::patch(source,root)` accepts only three exact Python source pins (private owned384933, packed owned32 761227, packed owned16/32 c1c606), two exact native literal pins, and exact reused launcher SHA `38026abe7a6f3c70dc36980317c17d62e82e3823f9a78de16246c9e7502b2d28`. It rejects changed/previously patched input. It reuses `tools/dl_fork_owned_cooperative.py::LAUNCH` rather than inventing a new launcher. That prior optional helper is independently documented in the owned fallback audit; the standalone835 common-control work used the related checked cooperative/PDL-off pattern.

The new source changes only:

1. Replace shared native `moe1_launch` with the exact existing checked cooperative launcher. It rejects PDL, requires SM90/cooperative capability, positive bounded grid/block/shared shape, chunk/local bounds, successful shared-memory attribute and occupancy queries, at least one resident CTA and sufficient whole-grid capacity; it calls `cudaLaunchCooperativeKernel` with a Params copy on the supplied stream. CUDA API failures are fatal.
2. In common `spin_failed`, keep the original timer and diagnostic atomic, then execute `asm volatile("trap;" ::: "memory")` before its old return. This is an intentional **device** change, not merely host admission. Timeout may no longer return a success-like value to code that then reads unpublished memory or cleans up counters.
3. The single selected `kernel_fn` call changes `_moe1_run(...,10,True)` to `...,10,False`; the native module gets a distinct `okc_decode_layer_moe1_coop_trap_v1` name. Original MAIN51 ABI ordinal10 is unchanged; `_moe1_source` still exposes only this case and rejects the rest.

Native reversal proves every other byte of the device literal, including BF16/scale packing, computations, scratch/Params, fences, barriers, predicates and routing, unchanged. Python AST reversal proves only this literal, the one call's PDL constant and module name change. There is no new fallback, schedule, tuning, scale representation or baseline mutation.

## Progress and timeout scope

Cooperative admission addresses the missing documented whole-grid scheduling contract; PDL-off requires the prior stream task to complete normally before this launch. It does not remove the software publication protocol. The original release/acquire counters and all CTA/WGMMA/proxy barriers remain necessary and retained. [CUDA12.8 §8.7](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#grid-synchronization-cg) is the primary co-residency reference.

Source polling paths using the shared watchdog include full/empty shared-mbarrier waits (`mbar_wait_to` / diagnostic `mbar_wait_cyc`), global acquire thresholds (`spin_ge`, including xready and stage done counts), dynamic B-queue handoff (`bq_get`, active under mode51's dyn_b), and the final generation/grid rendezvous. They all inherit fatal timeout behavior. Diagnostic-only paths remain compiled out of MAIN51 where their traits say so. Ordinary finite metadata-search/work loops are not publication polls. Hardware WGMMA waits, named barriers and `__syncthreads` retain their original collective participation/progress obligations; this patch is not a new watchdog on hardware collectives and does not excuse a broken protocol.

PTX8.7 §9.7.19.4 defines `trap` as aborting execution with a host interrupt. A reached trap is a CUDA execution failure, not a normal tensor result; downstream qualification must preserve fatal launch/synchronize/graph errors. The diagnostic atomic is best effort, not a requirement that the CPU can inspect a poisoned device context afterward. [PTX8.7 trap](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#miscellaneous-instructions-trap).

This is a source-backed repair proposal. Passing CPU mocks or using cooperative launch alone is not a complete correctness certificate: independent path review and actual H200 math/empty/fresh replay/capture/resource gates remain required.

## Exact composable pins

`tools/verify_owned_progress_compose.py::compose(source,root)` replaces only the embedded owned backend and, for the private package, its owned source pin. AST reversal confirms the outer dispatcher, all timing/scoring gates, G backend, other source pins, startup tests and metadata/fallback policy remain unchanged. Every package adds770 UTF-8 bytes.

| Source | New SHA256 | Bytes |
|---|---|---:|
| Packed owned16/32 backend | 2b4e0740de26a92fa9ef0f260c28e55f48f9fb2ae8fa20de863e4d8e8718635d | 326791 |
| Private owned backend | a841443465c47656f3614885c41d851bb796cd917f6ef2505b8c77e6f8cabc08 | 512681 |
| Production mask4 | d82f3c9d105a4e66ea338767bb67e6116cb79bda1726db54b12cdf651e6f8ce6 | 908412 |
| Production mask15 | 81f056f5e495f5cab0a66183a5b56cab783e7fa47732ed58e0ffa4feed5127a1 | 952309 |
| Private four-cell package | 094841bd26822392d9bcabbe80297dbd84df10a8f29a9749edc0a9dd143b8f3a | 970042 |

Package manifests and original→new backend pins are in [evidence](evidence/2026-11-01-owned-verify-progress). These are candidate pins, not authorization to substitute a queued payload. First two G diagnostic payloads remain untouched.

## CPU and local compiler evidence

Three actual tests in `verify_owned_progress_test.py` pass: complete AST delta reversal for backend and all three packages (private input regenerated through tracked `dl_verify_pair.build`), unsupported/already-patched epoch rejection, and compiled C++17 execution of the **exact reused** launcher. The C++ mock exercises M128/M256 success,16 distinct prelaunch rejection/fatal API cases, a fatal launch return, and the timeout branch. Only PTX `trap` is mapped to a C++ exception for that CPU branch test; this does not execute a CUDA trap. The existing12 production dispatcher facades also pass on the actual newly composed mask4/15 function bodies.

The selected `moe1::moe_main_kernel<51>` control and trap repair both compile through local CUDA12.8 NVRTC/PTXAS for sm90a: **96 registers,0 stack/spill,16 barriers**. The host launcher is removed by this existing device-only compiler adapter; the actual cooperative CUDA host ABI/link has **not** been newly compiled against Torch here. NVRTC emits retained unused-symbol warnings, not a new pipeline warning.

Archived selected raw SASS gzip, logs and manifests permit independent recount with `tools/verify_owned_progress_static.py`:

| Static selected entry | Control | Repair |
|---|---:|---:|
| Instructions | 17528 | 17744 |
| HGMMA | 192 | 192 |
| MMA substring count (includes HGMMA) | 288 | 288 |
| WARPGROUP.ARRIVE | 96 | 96 |
| wait1 / wait0 | 96 /12 | 96 /12 |
| BAR /MEMBAR /FENCE | 189 /43 /33 | 189 /43 /33 |
| LDL /STL | 0 /0 | 0 /0 |
| BPT.TRAP | 0 | 73 |

The73 static trap sites are compiler inlining/unrolling of the common cold watchdog, not73 timed faults. Unchanged counts and resources do not forecast latency or prove dynamic admission. The216 extra static instructions are not a measured performance cost.

Reproduce with `verify_owned_progress.py <root> <owned-source> <new-owned>`, `verify_owned_progress_compose.py <root> <package> <new-package>`, CPU tests with explicit `OKC_REPO_ROOT`, `OKC_PACKED_OWNED`, `OKC_PROGRESS_PACKAGES`; then the existing `attention_owned_comment_compile.py <new-owned> <out> moe --root <root>` device-only recipe. Full driver/options/source pins are archived. Never submit this tooling or a private diagnostic as a production file.

## Next actions and stopping criteria

Root must obtain independent wait/ordering/admission review, regenerate exact pins from integrated tracked sources and preserve attribution. Then a bounded matched private comparison must use this legal common owned epoch, retain all prior strict math/cache/fresh/empty checks and original score gates, check actual main51 kernel attributes/cooperative admission and graph replay, and reject all CUDA errors. A single-arm production measurement additionally needs complete all13 official correctness/all12 benchmark coverage and whole preparation within295s. Any unsupported occupancy, capture, timeout, compile error or incomplete record fails closed; no fallback into ordinary launch and no automatic retry. Historical ordinary/PDL timings are different launch policy and cannot be called paired evidence.

At commit time the independent proof review is pending; no H200 runtime or performance claim is made.
