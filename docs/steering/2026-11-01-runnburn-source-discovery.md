# runNburn: real Muse runtime, not a verified OKC layer competitor

2026-11-01,22:48-22:51UTC. Newly inventoried source lead from the parallel
survey `65b93a0`; its old timestamp did not establish prior body coverage.
This is a bounded primary-source classification, not a performance or full
correctness audit. Root owns README/index and the queued RMW measurement.
No GPU execution, build, judge/service call, kernel edit or source adoption.

**No credible stronger complete G replacement is established.** The repository
does contain substantive Muse-model loader, engine and CUDA layer code; it is
not an unrelated-name-only search hit. Its inspected execution contract,
formats and tests differ materially from the challenge. No source-associated
OKC/H200 result was identified in the inspected paths. Stop before a full
kernel audit or port, rather than count this snapshot as another candidate.

## Verified pin and inventory

Primary GitHub connector commit/main and an independent shallow Git tree both
resolve [coderredlab/runNburn](https://github.com/coderredlab/runNburn/tree/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a)
to **`f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a`**, committed2026-08-27
08:19:13UTC. Commit subject concerns Metal batched-verify initialization.
The current root inventory contains Cargo workspace files, README, LICENSE,
NOTICE, assets, crates, scripts and hidden.github entries.

Full recursive tracked inventory:737 files, including596 Rust,4 CUDA.cu,
27.cuh,73 Metal,25 Cargo manifests, one CSV and no Python or JSON files.
This is a file-type observation, not proof that an alternative-language
challenge implementation could not exist. All tracked paths, exact inspected
file/blob hashes and line anchors are preserved in
`evidence/2026-11-01-runnburn-source-discovery/inventory.json`.

## Actual relevant paths and content

All links below use the immutable pin above.

|Path|Verified content and relevance|
|---|---|
|[README.md](https://github.com/coderredlab/runNburn/blob/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a/README.md)|General Rust GGUF inference runtime with memory-budget/offloading policies and multiple backends; CPU default, CUDA model-specific. This is product/runtime positioning, not a challenge score.|
|[crates/rnb-loader/src/arch/muse_glimmer.rs](https://github.com/coderredlab/runNburn/blob/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a/crates/rnb-loader/src/arch/muse_glimmer.rs)|Actual `build_muse_glimmer_graph`; constructs F16 graph edges, separate Q/K/V projections and model-specific normalization/rotary/attention/FFN graph. Confirms real Muse support rather than just a README mention.|
|[crates/rnb-backend/cuda/src/runtime/decode_device.rs](https://github.com/coderredlab/runNburn/blob/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a/crates/rnb-backend/cuda/src/runtime/decode_device.rs)|Actual `decode_full_layer_device_resident`, weight pinning, norm/QKV/attention/FFN launch chain and KV state. Inspected weights are GGUF Q4_K/Q6_K, with f16 KV and FP32 staging APIs; this is not the challenge BF16 projection/NVFP4 expert/FP8 paged-cache interface.|
|[crates/rnb-backend/cuda/src/runtime/state/residency_budget.rs](https://github.com/coderredlab/runNburn/blob/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a/crates/rnb-backend/cuda/src/runtime/state/residency_budget.rs)|`prepare_muse_decode_layer_streaming_admission` adjusts a runtime resident-Q4K budget using free device memory. The matching identifier is a real runtime memory policy, not an OKC receipt or H200 score.|
|[crates/rnb-backend/cuda/cuda/kernels/persistent_decode.cuh](https://github.com/coderredlab/runNburn/blob/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a/crates/rnb-backend/cuda/cuda/kernels/persistent_decode.cuh)|Another complete-pipeline mechanism exists: cooperative persistent decode across35 layers. Its explicit assumptions target Gemma4 E2B Q4_K_M/RTX3080 sm86, not Muse/H200. Shared vocabulary is insufficient to transfer its precision/layout/progress proof.|
|[crates/rnb-backend/cuda/build.rs](https://github.com/coderredlab/runNburn/blob/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a/crates/rnb-backend/cuda/build.rs)|CUDA architecture is configurable and defaults sm86; builds multiple PTX/CUBIN runtime units. This does not establish inability to target Hopper, but no inspected sm90a/H200 challenge qualification is supplied.|
|[crates/rnb-llm/tests/muse_full_device_continuation_test.rs](https://github.com/coderredlab/runNburn/blob/f71c7eddac628d5f9b2bc08b6bf3c0fc9e74fc7a/crates/rnb-llm/tests/muse_full_device_continuation_test.rs)|Real fixture-gated, ignored-by-default test comparing carrier/eager token continuations on a product-size Muse GGUF. It exercises KV continuation/fallback. A test definition is not evidence it ran, a latency profile or an official source-associated receipt.|

The single tracked CSV is
`crates/rnb-backend/metal/src/qwen36_prefill128_layer0_routes.csv`: inspected
rows are Qwen routing data, not an OKC layer result table. Benchmark/profiling
paths in the inventory are runtime utilities/source; no benchmark was run.

## License, ancestry and evidence limits

The actual root LICENSE is Apache2.0; Cargo workspace metadata agrees. The
NOTICE identifies adapted llama.cpp/ggml portions under MIT and KVarN method/
reference provenance under Apache2.0, with per-file attribution obligations.
It does not establish G0 ancestry, an OKC team identity or entry eligibility.
No adoption was performed, and no complete per-file license/correctness audit
is claimed. These licenses remove an obvious metadata-only obstacle, not the
contract/performance gaps above.

Root/path inventory was complete and included hidden tracked entries. Targeted
primary bodies were read. A broader content grep of README/NOTICE/crates/
scripts/.github was stopped during lazy Git blob retrieval after it located
the relevant Muse chain; its partial results are not used as an exhaustive
negative search. Private/untracked/deleted branches, external result pages,
issue discussions and release binaries were not exhaustively surveyed. No
source-associated H200 challenge measurement is inferred from search ranking,
tests, commit age or general CUDA support.

## Decision and next gate

Do not replace the measured G base, import kernels or request a comparator
check from this evidence. Reopen this particular lead only with a complete
Muse challenge-contract artifact or concrete compatible mechanism plus
source-associated H200 affected-shape measurements and clear precision/
ownership/progress obligations. Generic memory-offloading or persistent-kernel
ideas alone do not supply exposed-time or full-score headroom for current G.
The next current decision remains root's owning RMW result; this discovery
does not alter its controls or authorize another dispatch.
