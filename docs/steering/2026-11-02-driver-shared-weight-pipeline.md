# Next implementation: producer-decoded shared-operand expert pipeline

November 2, 2026,22:39UTC evidence snapshot. Captain/sole judge dispatcher: **OK Driver**, `01a1963f-0c8e-7d41-be02-0e691dcfcaf7`. Implementation owner: existing **OKC Developer**, `01a19976-9e99-79c0-bab1-644bd8c0293a`. Evidence checkpoint **7cdb9a1a**; immutable control public607/physicalba9e327c, selected nativec7c201ae. Prior T3 captain remains user-stopped/paused.

The developer is idle after acknowledging the fixed K128 stop. Leaving the next decision unassigned would repeat the execution gap. **Allocate ONE concrete implementation: move weight dequantization out of the compute warpgroups into a dedicated producer group and feed BF16 shared-memory A operands to SS-WGMMA.** This changes dominant computation ownership/dataflow, rather than another small epilogue or an adjacent two-CTA geometry. Fund one90-active-minute implementation window and, once runnable and conforming, **one ordinary private full-layer qualification/measurement**. Captain alone dispatches; no public/scored/default action, sweep, automatic retry or confirmation allocation.

## Exact hypothesis and fixed design

Keep **256 output rows/unit, four compute warpgroups,64-token weight reuse, actual count-dependent NTT1..8 and one resident CTA/SM**. Change K stage to128, with exactly **two compact raw-weight stages, two decoded BF16 A stages and two activation stages**. The four compute groups execute SS BF16 WGMMA; a128-thread decode group absorbs the old32-thread weight-TMA role, and the independent32-thread activation producer remains. Total **672threads =512compute+128decode/weight-load+32activation**. Do not append128threads to both old helpers and accidentally request704threads at96registers.

Target actual B4/T32 and MAIN/T128,T256, both contexts. T1/T8/MID and unrelated layer routes stay at the immutable control. Do not inherit the losing K128 candidate's128-row/two-WG unit map, doubled grid, or assume its packing is directly reusable. Reuse relevant proven address/permutation helpers only after deriving the new full geometry.

The selected incumbent decodes register weights inside each compute group's issue path: `dq4`, scale reconstruction and four BF16x2 multiplies feed register A. It already overlaps some decode with an outstanding WGMMA group. The new design does **not** eliminate arithmetic or claim that overlap is absent. It separates that work from compute issue, removes live register-A ping-pong fragments from consumers, and permits producer decode to advance independently into the next shared A stage. Whether this repays producer throughput, new shared stores/reads and stage handshakes is the experiment.

This is a distinct mechanism from the parked two-front K128 resultbf8671b4: retain the original output ownership, grid/residency and64-token reuse; change WGMMA operand residence and dequantization ownership. No staged-ring parameter rescue of15688256 is authorized.

## Concrete resource and traffic price

Use the actual incumbent aligned header **22272B**, not the losing two-WG header21248B. Before any new header fields:

| Storage | Bytes |
|---|---:|
| Two BF16 A stages, each256x128x2 |131072|
| Two raw SC6 stages, each4x4480 |35840|
| Two activation stages, each64x128x2 |32768|
| Existing aligned header |22272|
| Total dynamic, SC6 |221952|
| Total dynamic, raw8-scale path |222976|

Including1024B reserved/block gives222976/224000B against233472B SM capacity. This is a source feasibility calculation, not target admission. Keep distinct raw/A storage; **no aliasing or unsafe early overwrite is needed**. Account for every actual new field, alignment, static shared memory and target reserved memory. Nominal672x96=64512registers is under65536, but real allocation granularity, compilation, static/dynamic storage and cooperative launch must be checked. Do not impose zero-spill proof on a control that already spills, or hide a treatment regression.

Compact weight bytes and logical X coverage remain unchanged for equivalent256-row units: twice as many K stages with half the payload. Unlike the losing128-row/two-CTA design, this does not double activation requests merely by duplicating row ownership. **It does add timed BF16 shared writes and async shared reads for every decoded weight, extra stage arrivals/descriptors and a producer throughput bottleneck.** No persistent expanded BF16 weight bank, new global dequant buffer, changed precision or untimed dequantization is allowed. Preparation may losslessly repack codes/scales; actual decode remains timed.

## Correctness and progress requirements

- Preserve the full logical raw/SC6 code-scale pairing and native gate/up mapping. Derive the SS A descriptor and exact m64/k16 lane-to-element mapping from the PTX ISA; test all producer addresses against the selected register-A reference. Do not treat a valid descriptor encoding as proof that the matrix is correct. Preserve all native scales, global FP32 postscales, BF16 boundaries, SiLU/intermediate, full per-expert RMS and slot-ordered weighted combination.
- **32 K stages/unit; preserve original256-row unit IDs and UBL values.** PhaseA emits128 SiLU-intermediate columns per unit. Therefore one phaseB K128 stage requires **one phaseA unit/four compute-WG publications**: publish doneAs index`ub`, waitcount4. The old K256 index`ub>>1`/count8 and losing128-row K128 index`ub>>1`/count4 are both wrong for this geometry. Keep JMAX88,32doneAs slots/job/epoch and sufficient MF storage.
- Raw-full readiness must precede every generic read; raw slots are released only after all decode readers finish. All generic BF16 A writers execute the documented writing-side proxy fence before publishing A-full. Consumers wait for the matching ready phase, issue correct SS operands and retain both A and X slots until **all** associated async readers retire. Separate raw/A/X phase tracking and tail drains must cover split segments, partial NTTs, phase transitions and empty work.
- Preserve inter-CTA publication/acquire, finalizer ownership, replay epochs/reset, occupied-to-empty-to-occupied behavior, independent mutable state, terminal progress and lifetime. A producer may not wait for a slot whose release depends on a consumer that it has prevented from running. Do not remove a fence/barrier based on tests. Cite the actual PTX ordering rule for any changed protocol.
- Compile the actual selected B4/MAIN bool variants with effective target flags. Reuse the established ordinary release/startup wrapper rather than writing a general paired measurement framework. Retain target resident>=1/cooperative admission and selected-path/source attestation. Respect source1MB limit and upstream notices.

## Independent implementation review

Independent read-only review found no fixed-design shared-memory fit contradiction and returned GO for implementation, not performance qualification. Preserve the existing `dq4` plus `bmul2l/h` fragment bits instead of recomputing FP32/global-scaled weights. The proposed no-swizzle K-major A layout follows the [PTX BF16 shared-memory layout](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout); writing-side publication must satisfy the [WGMMA async-proxy rule](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-async-proxy). Validate every address/bit independently; this review does not replace actual source tests.

For virtual old consumer thread t within targetWG: `wp=t>>5`, `lane=t&31`, `row0=64*targetWG+16*wp+(lane>>2)`, `k0=16*k16step+2*(lane&3)`. Existing af0/af1/af2/af3 pairs map respectively to `(row0,k0)`, `(row0+8,k0)`, `(row0,k0+8)`, `(row0+8,k0+8)`. Each pair stores both BF16 halves. Proposed BF16 shared index `F(row,k)=(row>>3)*1024+(k>>3)*64+(row&7)*8+(k&7)`. A descriptor base is `Astage+targetWG*16384`, LBO128B/SBO2048B, no swizzle/transA0; K16 step advances the encoded address by16. X uses the equivalent K128 descriptor. Exhaustively cover positions, group16 scales, KPERM and both contractions before accepting it.

The128 decoder threads perform approximately4x the decode work/thread of the512 original consumers. This throughput risk is explicit even if registers and shared storage fit. The producer must issue enough loads/decode independently to repay both that redistribution and the new A traffic. Nothing in this source review predicts a win.

## ROI and why this is the next allocation

Fresh cc474 affected weighted endpoints D32/V16/V32 are191.897812/261.713045/351.080222us; expert activities75.390323/146.141935/197.338710us. These activities are **not exposed removable time**. Under a deliberately fully-exposed sizing model,3% complete-layer improvement needs about11.37% of those activities with zero replacement cost,15.28% with5us net replacement cost per load. The current top100 layer gap is about6.67%, requiring a larger contribution. The new pipeline has no measured speedup or supported point forecast; the substantial dominant-path coverage, concrete different ownership and legal resource budget justify one bounded exploratory measurement. Do not require proof of a win before the first check or present these sensitivities as a prediction.

The strongest ready-adjacent alternatives were reconsidered in7cdb9a1a: hardware PR compression halving activity would size to only~1.61% layer, and the old owned-four-verify full implementation was slower on its targeted historical cells without a new performance mechanism in its repair. Those artifacts remain unmeasured/held in their exact newer forms, not universally rejected. The new shared-operand pipeline targets the expert compute body rather than those narrower unresolved paths. This is an engineering bet with explicit costs, not a guaranteed top100 result or a10x whole-layer claim.

## Delivery, measurement and stopping gate

Existing developer owns `tools/wgs_shared_a_pipeline.py`, focused tests and `2026-11-02-worker-shared-a-pipeline` report/evidence. Begin from immutableba9, preserve production/default and every losing artifact. Return the pinned runnable candidate, exact reversal/selection proof, both-phase address/math/protocol tests, actual selected compiler/resources and remaining target gates within90active minutes. A concrete hard source/resource/progress contradiction triggers an immediate STOP report; no silent extension into generic source surveys or another full observer harness. Independent review is read-only, not another implementer.

Captain dispatches one private ordinary layer check when this fixed source is ready and live quota/no-live/source checks repeat. Require actual changed-path selection, target admission, inherited56 startup replays, official13passing tests/12passing benchmarks/12complete profiles, complete own receipt/log and six46:16 endpoints. A setup/numerical/progress/admission failure closes this fixed allocation without an automatic rescue. A valid subthreshold/negative result parks it without a ring/tile/register sweep. A substantial favorable first signal (prefer~3%+) earns a separately recorded matched full-layer confirmation decision; it is not causal proof or public permission. Archive every outcome, including failures, under repository rules.

Fresh22:39UTC snapshot: **9private layer/10scored left,18private MoE, no live jobs; overall/layer374**, public incumbent607 unchanged; layer1002.2680x. The official served schedule still closes kernels **November5,2026 00:00 America/Los_Angeles**, asset SHAa40d07775d2797cba5e81933f905fdc858a07025a89ba78b58a27d7444c7c207. [Preconditions](evidence/2026-11-02-driver-shared-weight-pipeline/preconditions.json). Implementation delivery and acknowledgment will be recorded separately; this initial allocation claims neither pickup nor a new job.


## Direct delivery and implementation pickup

After13f5e41f was pushed and all8 changed remote blobs verified, the app directly delivered the fixed implementation assignment to existing OKC Developer. Its active turn began22:43:57UTC and explicitly states: “I’m picking up the shared-A pipeline implementation. I’ll start with the fixed storage and ownership map, then build the selected B4/MAIN paths. The parked K128 candidate stays untouched.” This is actual implementation pickup; no compiled candidate, GPU qualification, check or speed result is claimed. The hourly monitor was updated through the app and read back with its cadence/destination preserved; it now follows this allocation instead of waiting for another candidate-selection report.


## Parallel reserve decision while the implementation runs

A separate bounded read-only reviewer examined a genuinely different body route: SIMT/GEMV for per-expert count1..2 inside larger-global-M calls, retaining larger-count WGS. That route is absent from selected B4/MAIN, but **no next implementation/check is allocated**. WGMMA arithmetic padding does not establish endpoint savings; scalar FP32 issue/reduction work, unchanged packed-weight reads/decode and full normalization/layout remain costs. Exact selectedc7 disables MFIN, so crediting removal of its dormant per-expert completion would also be wrong.

The retained [high-load capsule census](2026-11-01-g-highload-weight-census.md), owning standalone1c6ebc76b91d4b7f/M256/seed4517, has21jobs/337routes/408padded rows. Count<=2 comprises3jobs/4actual routes/24padded rows,5.88% of that bank's padded matrix work. This is not fresh layer coverage or exposed headroom. The separate T1 three-launch loss does not close this body design, but novelty and8/count math do not fund it either. [Exact scoped review](evidence/2026-11-02-driver-shared-weight-pipeline/sparse-body-followup.json) records required new coverage/instruction/traffic evidence. Current shared-A implementation/measurement priority is unchanged; no generic source census or competing kernel worker was started.
