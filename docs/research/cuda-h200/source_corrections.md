# CUDA/H200 research: source-checked corrections

**Prepared:** November 2, 2026 (Pacific time)

**Status:** This is a correction supplement, not an edited copy of the original Deep Research report and not a completed audit of all four kernels. No GPU profiling or performance experiment was executed for this supplement. No repository files were modified.

## Evidence boundary

The authenticated GitHub connection successfully read the pushed `yelgabo/okc` repository. Reads were pinned to commit `779b06ad6b5a1eefcd0a0006b6870668b987f2de`, rather than repeatedly following a moving `main` branch. The inspected MoE file has blob SHA `96b5bf91433fc87154de1edc40ce7a31b5c4c025`.

This verifies the pushed snapshot only. It does not establish the contents of the user's local working tree, uncommitted changes, the binary actually used in a judge run, or which variant won preparation-time selection.

The source inspection concentrated on the MoE packing, launch configuration, synchronization sites and experiment annotations, plus the November 1 steering document. It did not complete a fresh inspection of the projection, attention and full decode-layer implementations or independently reprocess the referenced raw benchmark receipts.

## 1. Consumer-order NVFP4 repacking already exists

**Original recommendation:** Treat conversion from model-order packing to MMA-consumer-order packing as a major, potentially overlooked P0 opportunity.

**Verified source:** `repack_expert()` explicitly constructs words in `[warp][chunk][lane][k16 step]` order. Each word represents one lane's `m16n8k16` A fragment. `W4_BITPOS` rearranges the individual E2M1 bits for the register-side unpack operations. The implementation performs the corresponding reshapes, permutations and bit assembly; this is not merely a design comment.

`repack_expert_fat()` constructs a different representation organized by stage, warpgroup, warp, chunk, lane and step. `_repack()` builds both representations, including their corresponding scale layouts.

**Correction:** The generic proposal to introduce consumer-order packing is already implemented. Its purported novelty and associated upside estimate must be withdrawn. This does not prove the existing layout is optimal. A useful new proposal must identify a particular remaining load, permutation, scale decode, dependency or bank-conflict mechanism in this existing layout and compare against it—not against hypothetical native packing.

**Source:** [MoE packing implementation, inspected source range 4700–4840](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L4700-L4840).

## 2. The two-CTA register-cliff proposal ignores the launched grid

**Original recommendation:** Sweep register caps around roughly 114 registers per thread to turn the nine-warp kernel from one resident CTA into two per SM.

**Verified source:** `moe_run()` obtains `sms` from `cudaDevAttrMultiProcessorCount`. Both the skinny launcher and `launch_main()` launch exactly `sms` blocks. The skinny kernel also declares `__launch_bounds__(kNTL, 1)`.

**Correction:** With one block per SM in total, reducing a block's register footprint cannot create enough additional blocks for two per SM across the GPU. Hardware resource capacity and the amount of launched work are different constraints. Changing the grid to exploit higher residency would be a separate scheduling and synchronization experiment, not an automatic benefit of a register-cap sweep.

Register pressure, spills and instruction scheduling still deserve inspection. The corrected question is whether the actual compiled variant wastes instructions or memory traffic, or constrains a demonstrated execution opportunity. It is not whether an idealized register arithmetic table admits a second block that was never launched. Shared-memory requirements also belong in the resource analysis.

The source contains software grid-barrier and cross-CTA readiness protocols. Any grid or work-distribution change must preserve their progress and memory-ordering requirements; this supplement does not certify such a change.

**Sources:** [SM-count lookup and fat launcher, 3620–3810](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L3620-L3810); [skinny kernel and barrier, 4300–4500](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L4300-L4500); [skinny launch, 4500–4530](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L4500-L4530). NVIDIA's [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) documents the hardware resource limits; those limits do not establish the launch geometry of this implementation.

## 3. Three-versus-four-stage tuning is not an untouched experiment

**Original recommendation:** Sweep two-, three- and four-stage rings as a fresh optimization avenue.

**Verified source:** `SKINNY_VARIANTS` already includes three- and four-slot variants. The adjacent annotations identify a prior paired comparison, run `75bca9775f6f4b0b`, favoring the three-slot version at 1, 8 and 16 tokens. Other annotations describe prior fat-path pipeline-depth and readiness-wait experiments.

**Correction:** Read the owning receipts before allocating another sweep. The source records that an important portion of the proposed experiment has already been conducted. A changed compiler, packing, tile geometry or execution path could justify reopening it, but that changed premise must be stated.

**Evidence qualification:** The recorded performance results were read from source annotations, not independently reproduced or revalidated against the raw receipts in this supplement.

**Source:** [Variant definitions and experiment annotations, 4580–4670](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L4580-L4670).

## 4. Global token count is not enough to explain the MMA choice

**Original reasoning:** WGMMA has an instruction M dimension of 64, so a tiny token count necessarily produces large M-axis padding in this implementation.

**Verified source:** The small-path description and packing put the weights in the A operand. `_wgmma_code()` generates register-A `m64nNk16` instructions with N ranging from 8 to 64. The variant annotations also describe hybrid modes using warp-level MMA for small jobs and WGMMA for larger ones.

**Correction:** Do not identify the workload's token-count variable with an instruction's M dimension without tracing the operand mapping. The report did not establish that mapping. A valid crossover analysis must follow actual per-expert job size, operand orientation, generated specialization and selected dispatch. “WGMMA is always better” and “a token count below 64 necessarily wastes 64/token_count of the M tile” are both inadequate rules for this code.

This finding challenges the report's justification; it does not establish a better crossover or a performance improvement.

**Source:** [Generated WGMMA instructions and hybrid-mode annotations, 4525–4640](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L4525-L4640), together with the packing implementation above.

## 5. Proxy fences are present; presence is not a completed correctness proof

The inspected activation producer calls `fence_proxy_async_global()` after readiness acquisition and before issuing its bulk copy. The gathered-activation writer also calls that fence before the subsequent publication sequence. The source describes the cross-proxy rationale and uses release publication after the participating-thread barrier.

The revised audit should trace these concrete writer-to-consumer paths, including scratch reuse and graph replay. It should not simply recommend “add proxy fences,” nor infer that the whole protocol is correct from their presence. No fence removal is justified by this supplement.

**Sources:** [Activation producer, 3000–3155](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L3000-L3155); [gather publication, 3620–3660](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/muse_moe_w4a16/submission.py#L3620-L3660).

## Revised disposition

The original report is useful as a literature checklist, but its ranked experiment list is not a reliable repository-specific roadmap. Its numeric upside ranges are not measurements of this code and should not be used for scheduling work or forecasting leaderboard movement.

The next audit step is to map the exact selected variants and compiled binaries to the incumbent receipts, then compare each proposed change with the existing experiment ledger. Resource/SASS inspection remains useful, but should answer a concrete question about a real executed path. A fresh “optimize everything” sweep is not justified by the evidence above.

The November 1 steering document already distinguishes microbenchmark wins from captured whole-layer improvements, identifies closed experiments and specifies a particular short-path diagnostic with acceptance gates. This supplement does not establish that its planned diagnostic is still pending or should outrank more recent work; it establishes that a new report must account for that history before replacing the work queue.

**Source:** [November 1 steering document at the pinned commit](https://github.com/yelgabo/okc/blob/779b06ad6b5a1eefcd0a0006b6870668b987f2de/docs/steering/2026-11-01-post-xor-ranking-pivot.md).

**Bottom line:** Actual source inspection has already overturned or substantially qualified several of the report's highest-priority recommendations. That is a meaningful correction, but it is not evidence that the entire codebase is optimal, a complete memory-model certification, or a finished four-kernel performance audit.
