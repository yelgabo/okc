# Owned-only expert core for the attributed layer fork (CPU handoff)

2026-11-01. Parent requested G1a, the repaired non-expert layer with our proven expert
core, before G1b with repaired native experts. No H200 dispatch by this owner.

`tools/dl_fork_owned_moe.py:patch_owned_moe(fork_source,owned_source)` now fully wires
owned-only experts. Owned source is pinned to `layer_attn31s.py`
SHA585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b.
The optional `module_source(owned_source)` remains definitions-only for G1b's native
selector to use as fallback. Do not apply the owned-only transform to G1b: it deliberately
removes native selection. Root composes attention/front/projection repairs and notices.

## Independent G1b review, subsequent same-day addendum

Reviewed `okc-today-moe/tools/dl_fork_moe.py` live SHA256
`7a695ad5591c8bcf71b0cb6e11b26981f410b0f174e827a4933eabd38a62f1c8`.
Its three source/scale tests and six actual Torch CPU host tests pass independently.
The native repair uses ordinary same-stream cooperative launch, actual occupancy/grid
and fixup bounds, per-reader proxy fences with NT arrival counts, BF16 GEMM1/GEMM2
results, BF16 effective gamma, FP32 post-gamma products and disabled MFIN/DF/YBF/PDL.
Source review found no further device ordering/math blocker in those changes. Global
release/acquire handoffs retain writer and reader proxy bridges. Marlin fallback uses
baseline BF16 GEMMs and official fused post-expert normalization, direct local IDs and
effective BF16 gamma. This is source/CPU review, not GPU correctness certification.

Composition blocker reported to both owners: patch_moe appends HOST_BRIDGE at EOF,
but projection repair truncates at `_make_chen`. In that order the bridge disappears
after renaming original `_ma_prepare`, leaving its caller undefined. Root must move
or reappend HOST_BRIDGE after the cut and confirm those final AST definitions before
dispatch, then append owned adapter definitions. Native agent acknowledges this and
root owns the integration fix. Do not interpret this review as approving the broken
composition order.

The reviewed live source includes synchronization outside optional catches after
owned/Marlin packing, after native repack and before each resource probe; post-probe
synchronization is also outside. The actual CPU injected synchronization error stays
fatal. This addresses the suggested risk of an outstanding execution error masquerading
as optional resource fallback. Final composed CUDA compilation, numerical validation
and complete shape coverage remain root-owned; no extra judge dispatch here.

Final integration addendum: root draft SHA
`7935ff411d5e7ec1af41b25c7fecbff3524c4311efa65b6635b05e1d5bd41025`
and worker patcher SHA119c5c1aa1580033b9fcf82061984a6eb70da5391f19cb6db605967a2576a846
reviewed. Four source tests and six Torch CPU tests independently pass. Final bridge
survives the cut: last `_ma_prepare`16793, `_moe`16846, installer17008. Executed final
AST flags/selector at1/8/16/32/128/256 for both SC6 states: small paths select repaired
helper0/helper0df alias with WS1; large select repaired WGS main (B4FIN at32). SC6 is
per-state and enabled only for corresponding WGS packs. Source-level integration
cleared for root's off-board check; no GPU certification from this review.

## Optional owned cooperative launch, not enabled in G1a/G1b

Parent identified a launch-contract gap: ordinary owned launches avoid same-stream
PDL overlap but do not request guaranteed co-residency for the software global barrier.
`tools/dl_fork_owned_cooperative.py:cooperative_module_source(owned)` changes only the
shared host `moe1_launch` helper and extension build name. Computing kernels, packing,
scratch, BF16 gamma, ABI and barriers stay identical. Current G1a/G1b sources unchanged.

The optional helper rejects PDL, requires SM90/cooperativeLaunch, queries actual active
blocks per SM for the exact kernel/block/shared tuple, bounds grid by SM count/resident
capacity, then uses `cudaLaunchCooperativeKernel` with a by-value Params copy. CUDA
query/attribute/launch errors are fatal. Generated module169743B,
SHA81790204ede654ef03ecb92988653ee98a16c5dd5106fc5ca006f61547f049c4.
AST comparisons verify only helper/build-name differences; Python parse/compile pass.
Exact-helper C++ mock test emitter included, but execution is **not verified**:
neither WSL nor Windows has g++/clang++/cl/nvcc available.

CUDA supports cooperative kernel capture since CUDA11. This does not establish our
exact resource-query/capture/runner combination. Host graphs differ from device-launched
graph restrictions; the latter's MPS restriction is not a blanket prohibition here.
[NVIDIA CUDA11 graph support](https://developer.nvidia.com/blog/cuda-11-features-revealed/),
[CUDA graph/runtime constraints](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html).
Hardware support, actual residency, capture/replay, mixed-shape correctness and performance
need a future permitted GPU check. No speedup estimate; cooperative admission may change
overlap/performance. Stop if resource/capture validation fails. No new check dispatched.

## Interface and reachability

Only sk8w3s and main51 dispatch are compiled. The computing templates and original
owned C++ host wrapper are copied unchanged; three unused skinny namespaces are removed.
All unit/fat packing and scratch helpers come from the same pinned layer. Device local_start
is a persistent zero tensor because donor routes already provide local IDs -1/0..31.
Effective gamma is `(1+raw_BF16_gamma.float()).bfloat16().float()`, matching the layer
ABI rather than the standalone raw-weight API. Epsilon is1e-8.

The adapter accepts `okc_run(w,scratch,x,lids,wts,out,0,32,nsm)` and writes `out` with
one extra `copy_` after the owned kernel. It stores no result or prior activation.
All owned launches are ordinary ordered launches (`pdl=False`), avoiding a resident
global barrier overlapping a prerequisite kernel. Original layer chunk loops remain:
global T can exceed384, but each adapter call must contain1..384 tokens. Each chunk
uses skinny for M<=16 and main51 otherwise. Deepcopy shares only the compiled module;
packed tensors, gamma, zero-local_start and mutable scratch get independent copies.

The transform replaces both `_moe` definitions with per-state owned selection, replaces
`_ma_prepare` with owned installation before `_tune`, and removes donor expert background
builds, SC6 specialization, legacy expert repack, raw post-expert gamma blob assignment
and stateful prewarm. Empty ma_w/ma_scratch placeholders satisfy existing call signatures;
the adapter ignores them. Native builder entries are failclosed stubs, not silent fallbacks.
Inert native CUDA strings/CPU string patches remain to avoid new import dependencies.
Marlin is not selected in G1a; unsupported owned scales/preparation fail closed.

## Executed CPU evidence and limits

Two tests in `tools/dl_fork_owned_moe_test.py` pass under real Torch2.10 CPU:

- Actual transformed `_ma_prepare`/`_moe` dispatch for1/7/8/16/17/32/128/256/384/385/512;
  installation precedes tuner, no native builder in the background list, native entries
  raise, unsupported/unprepared state rejects, owned source hash drift rejects.
- Actual adapter/install with fake compiled extension and real CPU tensors; local IDs
  from global_start128 pass through unchanged (including nonlocals), gamma rounding
  distinguishes BF16-effective from FP32-effective, correct two variants/epsilon,
  pdlFalse, current output copy and independent scratch on deepcopy. No GPU computation
  is claimed. Python compile/AST gates pass for tools and generated source.

Measured source size on projection-repaired public fork: **939156B before NOTICE**.
Adapter definitions themselves are168957B. Existing repository whole-line comment stripper
provides size reduction; no runtime I/O, decompression or compressed source is introduced.
Root must append full provenance/LICENSE/NOTICE **after** stripping, and check final size.

CUDA compilation, resource allocation, final composed source reachability and GPU math
remain unverified. No local nvcc/H200 is available. Final root check must exercise all
scored routes and fresh inputs; this is not a certification inherited from standalone
tests. Stop on unsupported scale, unresolved compilation/ABI issue, missing coverage or
new precision/order mismatch. Runtime pack/compile overhead and extra copy latency are
unmeasured; no speedup forecast. This is an implementation option between experimental
native experts and the slow library fallback, subject to the coordinator's final check.
