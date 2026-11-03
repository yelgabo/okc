# Public scout: a compact MoE implementation and a routing guard

November 3, 2026, 20:22 UTC sweep. Instructions read through `7812d75c`. New public source, without an independently checked performance receipt.

[Aravind-11/muse-moe-w4a16-kernels](https://github.com/Aravind-11/muse-moe-w4a16-kernels/tree/dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40) publishes a 91,057-byte submission and version notes at **`dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40`**, dated **19:25:32 UTC**. All 31 files in its untruncated tree were fetched and matched against public Git blobs and SHA256 hashes. Fourteen branch heads, forks and releases were checked.

The [final-version note](https://github.com/Aravind-11/muse-moe-w4a16-kernels/blob/dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40/docs/v35_final_board.md) reports **1.8696x** for job **`366c4a635e5f4b01`**. It explicitly puts the small increase over v31 inside judge noise and says the new WGMMA variants were not selected. The tree contains no raw run/profile receipt or recorded final prepared plan. Treat this as an author-reported score, not a reproduced result or proof that WGMMA improved it.

## What the final source establishes

The [submission](https://github.com/Aravind-11/muse-moe-w4a16-kernels/blob/dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40/submission.py) builds native SM90a code, prepares weight/scale layouts and chooses per-shape implementations. Its custom path splits the work into gate/up GEMM with activation, down GEMM and post-expert normalization, with PDL and optional per-block completion machinery. The ordinary custom route is a separate launch; an optional mode derives routing inside the GEMM CTAs.

The final `kernel_fn` uses the custom callable only for shapes enabled by `prepare`; otherwise it calls the Marlin fallback. The gate requires the selected custom timing to beat Marlin by 3%. Unsupported dimensions, local-expert ranges and unmeasured batch sizes retain the fallback. Static selection logic is visible, but the actual variant chosen by a fresh preparation remains a runtime outcome.

The [v31-to-current diff](https://github.com/Aravind-11/muse-moe-w4a16-kernels/compare/50e4fbc9c4bac54c7dd6e7e15f3b37b53502a406...dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40) changes only `submission.py` and the v35 note. One material change is the [guard at line 1389](https://github.com/Aravind-11/muse-moe-w4a16-kernels/blob/dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40/submission.py#L1389): in-GEMM routing now also requires the custom normalization path. This branch passes raw global routing IDs; the SGLang normalization interface expects remapped local IDs. The guard keeps that ID representation paired with the matching custom consumer. This is a source-level compatibility correction, not an independently reproduced numerical failure of SGLang.

Earlier notes expose instruction-level dequantization and scale-layout changes. For example, the [v27 note](https://github.com/Aravind-11/muse-moe-w4a16-kernels/blob/dcf3ee7c7c0b42e0f534648b88a06c138b7dfc40/docs/v27_alu_to_fma_pipe.md) attributes a gain to moving work away from the integer ALU pipeline. The final source contains the described multiply/mask forms. No SASS, pipeline utilization or performance effect was checked here.

## Decision and limits

This adds a small, attributable source reference with reviewable version diffs. It does not establish a gain over our incumbent, a universal WGMMA loss or numerical/publication/progress correctness. No foreign code, compilation, scorer or GPU workload was executed. Independent review was unavailable under the repository's no-new-agents rule.

Keep the current owner queue. Any later comparison needs the final prepared variants, exact input/routing and normalization contract, counter lifetime/publication proof, observer cost and unchanged complete-call controls. This scout allocates no experiment and does not reopen a closed reserve review.

## Coverage and publication

This incremental refresh checked first and prior terminal code pages for all five identifiers: 606 unique file/blob hits, all previously seen. Interior pages were not re-read. Thirty-two timestamp-bounded repository/commit/issue/PR queries cover activity from 18:28:21 UTC, with an overlap before the prior full sweep. Older newly indexed or backdated activity may be missed. Known default/nondefault heads were checked separately. Exact queries, totals and limits are in the [evidence directory](evidence/2026-11-03-scout-aravind-moe-2022/).

The previously watched default and nondefault heads, fork/release samples and public review snapshots are unchanged. Eighteen Guild pages still yield the same 66 public body fragments, including thread 147. The B200 quality gist remains at `d3236cf8`. Known blog bytes are unchanged; cached social leads add no newly bound kernel result. Original X profiles remain inaccessible. No authenticated Slack/Discord, restricted Guild bodies or encrypted messages were read.

The preceding [Guild timing and B200 control report](2026-11-03-scout-guild-timing-caveat-1823.md), commit `4e1a0edc`, has now passed remote-byte verification. Its publication and this later refresh are separate captures. Current report publication closes after its own evidence checks, remote verification and state update.
