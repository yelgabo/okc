# H200 CUDA research reference

Read this index before using the November 2, 2026 playbook to choose CUDA work. Open only the relevant topic and its linked evidence. Run `python3 tools/check_cuda_research.py` from the repository root first. A stale pin means the affected notes need another source read, not that the new code is wrong.

## Source corrections — shared with research and optimizer

**Read [source_corrections.md](source_corrections.md) alongside the CUDA/H200 audit before selecting work.** The user pasted this supplement to OKC CUDA Optimizer and asked that research have it too. This research session has read it; repository availability does not establish that another session has read it. [Publication and scope note](../../steering/2026-11-02-cuda-source-corrections-intake.md).

The supplement corrects the audit's repository-specific assumptions: consumer-order NVFP4 packing already exists; the one-SM-count grid does not acquire a second block per SM merely from a register-cap change; three/four-stage variants have prior experiment annotations; MMA operand orientation must be traced before identifying token count with instruction M; and proxy fences are already present, without that alone proving the protocol correct. The audit's numeric upside ranges are not measured gains.

These are findings about the supplement's pinned standalone MoE source at `779b06ad`, not a completed four-kernel audit or automatic conclusions about the selected layer fork. The original report's evidence qualifications and source links are preserved. This copy was imported from the Library report's retrieved text, not verified against raw backing bytes. Its original statement that no repository files were modified refers to the research pass before this publication. The existing `source.txt` and its catalog remain unchanged; this supplement is separate from that original attachment.

At the next normal checkpoint, reconcile affected recommendations with these corrections. Do not interrupt an active turn, restart a candidate, duplicate a job, or add another review/handoff gate. Decode32's existing attention confirmation and end-to-end execution authority remain unchanged.

## Find a topic

| Task | Reference |
|---|---|
| Check source-specific corrections to the CUDA/H200 audit | [Source corrections](source_corrections.md) |
| Choose instructions, dtype or a weight representation | [Hardware and numeric rules](hardware-and-numerics.md) |
| Change staging, decoding, WGMMA or warp roles | [Pipelines and memory ordering](pipelines.md) |
| Change expert order, work queues, Split-K or clusters | [Scheduling and locality](scheduling.md) |
| Identify a bottleneck or design a profiler capture | [Profiling and experiments](profiling.md) |
| Compile, inspect SASS, test correctness or promote a change | [Validation and integration](validation.md) |
| See what each recommendation means for our code | [Application ledger](application.md) |
| Resolve an external claim | [Primary references](references.md) |

## Current application decision

The scored layer's source is [g1b4_stage_release.py](../../../muse_decode_layer/cand/g1b4_stage_release.py), SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. Its selected native MoE body is `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`. The file named [submission.py](../../../muse_decode_layer/submission.py) is a separate implementation. Inspect final dispatch and prepared state before transferring a claim between them.

The selected expert engine already implements much of the playbook. This intake repairs the MoE compiler evidence gate and preserves the remaining advice with its applicability conditions. It makes no H200 speedup claim and changes no runtime kernel or default selection. D32 implementation stays with Decode32 Worker under [D32-02](../../steering/decode32/2026-11-02-02-end-to-end-ownership.md). The [steering report](../../steering/2026-11-02-cuda-research-application.md) records verification and remaining work.

## Preserve and refresh the research

[source.txt](source.txt) preserves the complete attachment byte for byte. Its original citation tokens are unresolved source data. They are not verified citations. [catalog.json](catalog.json) records the attachment hash, every Markdown section outside code fences, its line range and topic note, plus hashes of the code and evidence behind the digest. The primary reference list was recovered separately on November 2, 2026.

When evidence changes, read the affected code and old experiment's stopping condition. Revise the topic, application decision and steering conclusion together. Update the reviewed hashes in the catalog only after that reconciliation. Do not turn a source-only hypothesis into a measured result. The checker verifies coverage, local links and file identity. It cannot establish that a claim follows from the evidence.
