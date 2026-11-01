# Owned-only expert core for the attributed layer fork (CPU handoff)

2026-11-01. Parent requested G1a, the repaired non-expert layer with our proven expert
core, before G1b with repaired native experts. No H200 dispatch by this owner.

`tools/dl_fork_owned_moe.py:patch_owned_moe(fork_source,owned_source)` now fully wires
owned-only experts. Owned source is pinned to `layer_attn31s.py`
SHA585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b.
The optional `module_source(owned_source)` remains definitions-only for G1b's native
selector to use as fallback. Do not apply the owned-only transform to G1b: it deliberately
removes native selection. Root composes attention/front/projection repairs and notices.

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
