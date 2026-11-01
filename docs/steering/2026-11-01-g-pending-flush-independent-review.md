# Independent pending-WGS flush movement review, November 1, 2026

**Source safety clearance** for the exact pending-flush movement below. It may be composed into the same single-arm attributed, self-testing public layer under the coordinator's existing user authorization. Clearance covers source ordering/lifetime/progress, not an unmeasured speedup, the final composed wrapper's byte identity, or H200 correctness. No jobs or candidate edits by this reviewer.

Pins: input pure G-stage SHA256`295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491`; original MOE_CUDA_SRC`c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`; original wgs_seg`c67020c073a65b7042104903c1f679fbd7230c23d8bcd0e76e69306631340be3`. Reviewed live MoE worktree `tools/dl_g_next_expert_flush.py` and `.build/today-moe/g-next-flush.cu`. Exact generated pure file **861728bytes**, SHA256`8195660dec57d9ceb6d6ab71d048fd5c144e657d8ec44022c75b0f34ca88bca1` independently verified. Worker evidence is `docs/steering/evidence/2026-11-01-g-next-expert/` in its worktree; not a new H200 receipt.

## Actual delta

Remove the previous pending record's `wgs_flush<PB>` from FIRST &&ci==OKC_WGSFCI inside the first stage. Execute it immediately after `stage(std::true_type{})` returns, before any following stage, current-segment epilogue/fixup additions, or pending-record overwrite. The existing full first-stage wg_wait0, every reader's shared async-proxy fence, X-slot empty arrival and seq advance now all precede the old pending flush. No fence, counter, barrier, WGMMA drain, weight release, data operation, host dispatch or precision operation is removed. The patch inverse-AST gate confines the entire Python change to this one CUDA literal.

## Publication and lifetime proof

- `WgsPend` describes the **previous** segment. Its epilogue stores or red.global.add partials occur before entering the current segment. The move delays its publication; it does not move publication before those stores. The entry phase-A global async-proxy fence and128-thread named barrier remain inside unchanged wgs_flush (2017–2092), before the leader's release/count operation. Delaying a release with additional independent computation does not weaken its ordering.
- Kind1 publishes doneA after the previous phase-A xB stores. Kind2 releases the previous split share's stage count. Kind3 retains the acquire wait for all other shares, named barrier, global acquire/release fence, fixup reads/zeroing and epilogue. The nondesignated-finisher acq_rel last-arriver path is unchanged too. Phase-A split-finisher xB writes still receive their **post-epilogue** global proxy fence/barrier before doneA publication. MFIN is off, so kind4 is inactive.
- The previous record is consumed/cleared before this current segment can write its own partials/epilogue and replace pd. Current first-stage accumulators are fully retired and remain live compiler-managed values; the moved flush does not access their registers. Its shared metadata/wglast accesses do not alias the weight/X rings, which begin after Smem. Job metadata remains immutable throughout these phases.
- Fixup ownership is unchanged: slot is the CTA containing the unit's first stage. A contiguous per-phase CTA range has at most one **partial unit it owns at its trailing boundary**. Its initial partial, if another unit, belongs to an earlier CTA; interior complete units have slot=-1. The trailing-partial reorder changes visit order, not owner. A range wholly within one unit is not reordered into duplicate visits. Thus a later distinct unit cannot newly reuse the previous unit's phase/slot buffer while the delayed flush is outstanding. Phase-A/B fixup and count regions are separate; later-launch reuse still follows the original whole-grid completion protocol.

## Progress and edge cases

The added work before publication is the rest of the **already entered first stage** and its existing full retirement. Both current X and weight full-barrier acquires happened before the old FCI4 flush as well. Phase A uses gathered original xA; phase B uses published phase-A xB. Neither stage's math consumes the previous pending record from the **same phase**. Therefore the added work does not introduce a future same-phase publication dependency or a new wait cycle.

A designated split finisher may wait longer before consuming other shares, but its wait target/owner/publication relation is unchanged. The current X slot is now released before that wait, allowing producer reuse; the flush itself never reads that slot. Cooperative whole-grid admission and original ordered range/finisher progress assumptions remain mandatory. Per-warpgroup publication order is preserved; no global total order across CTAs is claimed.

Empty phases never call wgs_seg and retain the terminal no-op flush. Every nonempty segment executes exactly one first stage, including one-stage or partial segments, then consumes prior pd once before generating a new record. Reordered trailing/head segments preserve the same per-warpgroup pending order. The final `wgs_phase` flush remains unchanged, including phase-A completion before that CTA enters phase B. No sequence, ring parity or arrival count changes.

The ordering argument relies on the same PTX8.7 WGMMA completion, proxy-fence, barrier and release/acquire rules as the [stage-X review](2026-11-01-wgs-stage-x-retire-review.md) and [native expert audit](2026-11-01-fork-moe-safety-audit.md), not absence of a compiler warning or passing numerical tests.

## CPU/compiler evidence and release conditions

Independently reran all **3 worker CPU tests PASS**: exact scope/inverse AST and primitive counts,3264 pending/stage/ring edge-model cases, per-phase publication-order model. These models do not simulate CUDA execution or prove runtime scheduling.

Identical sm90a/CUDA12.8 options within each control/treatment pair. Actual two-entry SASS census remains **1024HGMMA /512wait1 /96wait0** in main and B4. Both entries remain96registers/16barriers:

|Build|Stack bytes|Spill stores/loads bytes|SASS LDL/STL across both entries|
|---|---:|---:|---:|
|main control|8|4/32|16/2|
|main moved flush|0|0/0|0/0|
|B4 control|16|120/216|100/60|
|B4 moved flush|8|8/36|16/4|

**B4 is not spill-free.** Compiler-only resources are not observed H200 runtime admission, occupancy or latency. Preserve existing resource guards; no cap sweep or guard relaxation follows from these numbers.

Before a public measurement, freeze/review the composed source, preserve attribution/notices and original single timed `kernel_fn`, and retain the private Marlin real-routing/fresh/empty replay, scratch-isolation/guard and fatal CUDA/math/spin/budget tests used for607eb52bdaf24160. Require actual native backend/module coverage and complete official correctness. A failure blocks the release; a passing scored point remains single-arm measurement and does not establish a causal performance gain versus its control or the owned incumbent.
