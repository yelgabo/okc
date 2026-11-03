# Decode32 reconnaissance after the 460-experiment note

Date: November 2, 2026 Pacific (repository entries through main `93ed0192d3a60f199c87b9b4771c54fdefc6179f`). Requested as “recon” while viewing the Decode32 task inbox. This is a read-only evidence reconciliation. No kernel edits, private GPU jobs, public submission, new ownership assignment, or interruption of an active worker.

## Immediate correction: D32-MOE-01 was a duplicate

Research's [D32-12](decode32/2026-11-02-12-h200-460-next-leads.md) misidentified a serial prefix/range loop in a historical helper literal as the selected B4 code. Decode32 Worker **picked it up**, captured the final factory and selected compiler inputs, and conclusively closed the proposed serial-to-warp-parallel B4 change in [D32-13](../../../docs/steering/decode32/2026-11-02-13-b4-prefix-selected-source.md), committing source audit `810d8264` and subsequent reconciliation. The public607 B4/T32 and MAIN select native `c7c201ae`, which **already performs the 32-lane warp scan and 4-lane boundary lookup**, including the per-unit cost model. Tiny/Tiny2/MID use a different native helper with the serial loop. Three focused CPU tests passed. No new B4 kernel or judge job was spent. The 460-note's reported ~0.9% improvement is not additional headroom available from applying the same feature again.

**Research correction:** Do not assign or implement D32-MOE-01 as written in the historical main body of D32-12. Its dated addendum supersedes the premise. Future proposed improvements must capture final selected source plus flags, not rely on a textual match in the 869 KB layer file.

## Live batch32 execution state

The previously submitted one-tile full-context attention lookahead, candidate physical SHA256 `5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5`, private job `17383751c6a24010`, passed 13 tests and 12 benchmark/profiles. Its standalone layer D32 adjusted `C32 = 46*149.764us + 16*312.070us ≈ 11.882275 ms`; own D32 SGLang baseline ≈21.860480ms. The earlier unchanged `94ef698e91964dfa` control gave 12.324230ms **on different inputs and node**: that difference is not a matched gain.

Two successive attempts at a matched attention check produced **setup failures, no paired timing**: first unavailable target PyTorch FP8 `isfinite`, then `EXPERT_FRESH_ROUTING_MISMATCH` for corrected job `62a4f6b27ed74b50`. Strict natural-routing equality was retained. [D32-11](decode32/2026-11-02-11-matched-routing-mismatch.md) documents a new payload `9e5e5d64` that logs the load, context, seed, replay, placements and differing routing rows without relaxing the gate. **Owner job `5909739e8a4445b2` is accepted** and is pending at this GitHub snapshot (commit `1d654b5f`). Only Decode32 Worker owns the job waiter. Do not submit or poll it from a second actor; wait for the resulting own report before deciding whether attention is correct or beneficial. This observed activity refutes any claim that the worker is simply idle based on unchanged earlier commits; it cannot prove the desktop session's present runtime status.

Existing MID17–32 (job `a6fe2769779e4299`) and direct-weight (job `0b684a4065354fc5`) paths remain parked after negative D32 screens. Decode32 Worker remains owner of all T32/q_len1 implementation, private testing, analysis and integration, with research supplying architectural choices and no Driver handoff. The other five loads must not regress, and public/scored permission is unchanged.

## Parallel tracks and what they rule out

- **General Optimizer, GOPT-PIPE-01:** selected current Tiny/Tiny2/MID weight-ring NS3→NS4 treatment, private MoE job `f0b875484960438b`, passed 5 tests/6 benchmarks/6 profiles and complete matched replay. All 24 affected aggregate comparisons were below parity (0.966414–0.990167) and all six group gates failed. **Retain NS3; park this exact NS4 change**. It touched T1/T8/T16 helper paths, not T32 B4, and does not demonstrate B4 pipeline depth could improve.
- **CUDA Optimizer:** scale-only job `3aa3885f6b734f8c` removed local compiled B4 spills but matched factors 0.992411–1.004932 failed the promotion gate; combined scale/group job `3c45788127e945c4` passed correctness, but factors 0.995560–1.008180 also failed. No runtime promotion; the two-step WGMMA grouping stays. These results do not prove all dequantization or schedule redesigns impossible.

Source: [current General Optimizer terminal](../../../docs/steering/2026-11-02-general-optimizer-pipeline-depth.md), [CUDA optimizer terminal](../../../docs/steering/2026-11-02-cuda-part2-experiments.md), [current Decode32 ledger](../../../docs/steering/decode32/STATUS.md). Original unverified external source: [H200 ~460 experiments](../../../docs/research/h200-460-experiments.md). Its reported bandwidth, overhead and gains were not independently reproduced.

## Remaining research direction, without launching a speculative experiment

**First dependency:** finish the strict attention routing diagnostic; capture precise mismatched rows and distinguish whether duplicate controls disagree versus treatment alters the route or the checker is invalid. This is already assigned and in flight—no new task.

**Next technical hypothesis if a new B4 design is needed:** isolate **exposed B4 phase A→B handoff or terminal tail** rather than reapplying warp prefix or increasing steady-state weight bandwidth. The actual selected WGS implementation already has 32-lane prefix, SC6, cost-weighted per-unit scheduling, phase-B `doneAs` acquire checks, async-proxy publication and two-phase per-CTA overlap. Reuse a current non-distorting phase/CTA trace before devising a kernel patch. Previous detailed observer introduced 4.91–9.54% kernel overhead and invalidated attribution, so it cannot justify another optimization by itself. Any concrete treatment must retain producer-consumer lifetime, resident-grid progress, BF16 rounding, exact routing and replay. **No implementation-level B4 mechanism is qualified yet**; research must prove its particular new dependence or exposed cost instead of directing the medium-effort worker into another blind knob sweep.

**Alternative lower-priority check:** verify the actual T32 pooler/router preparation choice. The selected layer already offers conditional fused `bfpf` when pooler/router both select `sk2`; no new fusion work is warranted without proving the fallback was selected with exposed latency.

## Disposition

- **STOP:** D32-MOE-01 prefix task, as already implemented; no GPU check for it.
- **CONTINUE:** only the existing Decode32-owned routing diagnostic `5909739e8a4445b2`, with its own waiter.
- **HOLD:** further B4 scheduling/phase-boundary work pending selected-source and non-distorting exposed-time evidence. Do not reopen NS4 or CUDA scale/group tests as new D32 leads.
- **KEEP:** current public incumbent `607eb52bdaf24160` unchanged. Do not infer promotion or rank movement from an exploratory job.

Publication of this recon report is not new worker pickup, a direct message, a judge dispatch or a measured speedup.
