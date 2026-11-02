# SiLU ROI gate before any implementation or GPU allocation

November1, evidence at5fdbd56. User asks how much SiLU can recover from the historical5.77% G gap and directs us to avoid another low-impact check. Current [worker source screen](2026-11-01-universal-bf16-silu-table-screen.md) has already stopped the universal table at unresolved rule eligibility before any treatment implementation, compilation or GPU action. No private/scored check is allocated to it.

## What is known

There is no isolated SiLU latency or exposed critical-path measurement. Its recoverable fraction of the gap is currently unknown and its demonstrated gain is zero. Finite BF16 indexing proves a representation property, not useful performance.

The proposed replacement is a65536-entry FP32 lookup,262144bytes. It replaces exponent/division arithmetic with a dependent memory access. Cold-L2 cost, residency, dependent-load latency, instruction scheduling and final BF16 product fidelity are unmeasured. Required source/owner/resource and organizer-eligibility gates remain. G0 already performs SiLU; a faster inherited activation function would be a possible additional improvement, not automatically the explanation for upstream-to-fork regression.

## Conditional whole-score sizing

I re-read all12 official adjusted medians and the single displayed expert-kernel row per cell from incumbent receipt607eb52bdaf24160. Hold the baseline and all other work fixed. Hypothetically subtract the same fraction of that expert duration in every cell, form46/16 weighted times and take the geometric mean of six old/new ratios.

| Hypothetical reduction in entire expert duration, all cells | Whole-layer score increase |
|---|---:|
|1%|0.509%|
|3%|1.542%|
|5%|2.597%|
|10%|5.335%|
|10.773%|5.772%, matching the historical G0 score ratio|

These are arithmetic scenarios, not a SiLU ceiling or forecast. Profiles and adjusted benchmark medians came from different replays, and each saved expert microsecond is assumed to leave the whole graph's critical path with zero replacement cost. The old G0 score2.248790532 and incumbent2.126082676 also used different inputs/baselines. Exact inputs, formula, source hash and solved fraction are preserved in [conditional-budget.json](evidence/2026-11-01-silu-roi/conditional-budget.json).

## Allocation decision and next gate

Maintain the no-GPU/no-implementation HOLD. Resolving rule eligibility alone does not justify a check. Before allocating one, require a selected-backend cost/exposure argument for how much complete expert time the existing activation path adds, what replacement work remains and which official cells benefit. Map that estimate to the actual six-load score, compare it with the next G-integration alternative, and state uncertainty explicitly. At least1% repeatable whole-layer opportunity plus room over duplicate noise is the existing minimum gate; an explanation of the whole5.77% would require the much larger10.8%-of-entire-expert scenario above if all cells benefited uniformly. Partial-load proposals require their own weighting.

A source-only time-bounded feasibility screen may be useful, but no full benchmark, repeated seeds, thresholds or quota should follow from table indexing or static operation counts alone. If no credible exposed budget exists, keep the idea parked and choose another integration dependency with evidence. Preserve the current user's worker ownership, no-subagent rule and separate GPU/public authorization. The worker/captain already accepted this SiLU HOLD. This ROI clarification adds the sizing requirement; it launches no experiment or kernel change.
