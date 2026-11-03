# CUDA research: source-correction supplement is now shared

November 2, 2026 Pacific. The user supplied `okc_cuda_research_source_corrections.md` to OKC CUDA Optimizer and asked that research have it too. This research session has read the complete supplement. Its report text is now available at [source_corrections.md](../research/cuda-h200/source_corrections.md), alongside the [shared CUDA research index](../research/cuda-h200/README.md).

## Message for research and optimizer, at the next normal read

Read the supplement before treating the CUDA/H200 audit's ranked suggestions as new repository-specific work. Apply its corrections at their documented scope: consumer-order packing already exists; the proposed register-cap residency benefit ignores the launched grid; three/four-stage variants have prior experiment annotations; MMA operand orientation must be traced before reasoning from token count; existing proxy fences require protocol analysis rather than generic add/remove advice. The original speculative upside ranges are not measured gains.

This is not a new assignment, approval gate, request to interrupt a turn, or order to restart work. Retain any current implementation or accepted job. Reconcile only recommendations actually affected by the corrections. Decode32's attention confirmation and no-handoff execution authority remain unchanged.

## Provenance and limits

The imported report is the complete 83-line text retrieved from the Library artifact `okc_cuda_research_source_corrections.md` (file ID `file_0000000088e881fda07df9780eb36360`). Raw-byte materialization was unavailable; this is a text copy, not a claim of a byte-verified archival transfer. The original report is not rewritten. Its statement that no repository files were modified describes its original research pass, before this publication.

The supplement's source inspection is pinned to `779b06ad6b5a1eefcd0a0006b6870668b987f2de`, mainly `muse_moe_w4a16/submission.py` (blob `96b5bf91433fc87154de1edc40ce7a31b5c4c025`). It is not an audit of every kernel, a fresh compiled-binary review, independent receipt revalidation, GPU measurement, or proof of a faster implementation. Do not silently transfer its standalone-MoE findings to the selected full-layer fork without checking that implementation.

The earlier [alignment note](2026-11-02-cuda-optimizer-alignment.md) described a then-missing report. The later CUDA/H200 audit and this correction supplement have now been located in Library. The separately named optimizer's actual current patch is still not established by this intake. No claim of direct-message delivery, researcher acknowledgment, worker wake-up, changed model settings, GPU dispatch or public submission is made.
