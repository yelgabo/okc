# alygg77 public kernels: measured constraints and a narrower next round

Date: 2026-10-30.

Our evidence snapshot: `269c110182286bbd89150b0f83f4a750f64b77ad`.
External repository: `alygg77/okc-muse-spark-kernels`, pinned to `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`.

## Provenance and scope

The user supplied https://x.com/alygg77/status/2115914318126735362. The exact post text could not be retrieved through X, public embed endpoints or the searched mirrors. A public OKC repository under the matching GitHub handle was located independently. This report analyzes that repository; it does not claim to quote the post, establish what it says, or establish that the post itself links to this repository.

This is a source/document review, not a fresh H200 measurement or a complete synchronization audit of the donor binaries. The donor's measured-facts and negative-results documents are first-party reports. Their numerical results are valuable leads, not independently reproduced facts about our kernels. No live leaderboard or quota query, GPU check, submission, kernel change or validation-policy change was performed here. Our current scored projection receipt remains `d85d0081fa8a4a43`, 1.4975856x, as recorded in the existing reports.

## Decision

Add this pinned repository as a reference, but do not replace our implementation wholesale. Its most useful contributions are specific candidate routes and negative measurements that constrain our next experiments.

The immediate projection queue should be:

1. Compare the donor's three successful small-token routes and its 128/256-token router against our exact current selections, after checking for mechanisms already present.
2. Query actual residency of our failed four-CTA QKV clusters; do not rebuild that family merely because its nominal CTA count is 128.
3. Separate dense QKV256 data movement from O256 mainloop/epilogue work. Do not use a single aggregate TFLOP/s target as the diagnosis of both.
4. Keep denser lossless compression as an offline feasibility study, with explicit escape/decode costs. It is not an established route to the missing 4.1%.

Attention and MoE receive scoped source-comparison tasks, not a new broad rewrite. Standalone and layer promotion remain separate.

## 1. What this release actually adds

The donor README explicitly credits vjeux commit `f35a41e` as the base for all five files. Attention and prefill are unchanged vjeux files apart from attribution headers. Projection retains the 12-bit BF16 format and adds custom WGMMA O-projection work, pooler tuning, selected Nathen Fernandes small-M routes, and robustness fixes. MoE adds KPERM3, a lossless K-layout change that shares scale decoding. These attributions prevent mistaking a new repository for five independent new architectures.

Archived leaderboard/API snapshot, 2026-10-29 20:54 UTC:

| Board | Reported score | Archived job |
|---|---:|---|
| Projection | 1.5502x | `f07235a6ca8a46e2` |
| Attention | 2.6530x | `1c8d81a0dbd94ee0` |
| MoE | 2.4015x | `e1bcad5a55cd4a29` |
| Decode layer | 2.2058x | `a61afb9687f14eff` |
| Prefill | 1.8167x | `7f87198d13a34650` |

Their projection rank 75 is historical, not today's rank. Their best 1.5502x is below our last recorded CLI top-100 cutoff of 1.5560x. They also report four runs of an unchanged projection file at 1.5259, 1.5410, 1.5443 and 1.5502. Do not budget the full difference between their highest run and our multi-receipt median as recoverable code performance.

Sources: donor README, RESULTS.md and H200_MEASURED_FACTS.md section 3, all at the pin above. No fresh official verification of those archived scores was performed.

## 2. Projection comparison: the useful discrepancies are not a giant O256 gain

The following is a cross-run prioritization table, NOT a paired improvement estimate. Our values are the six-receipt frontier medians, which mix the pre/post split-exit-barrier variants. Donor values are its published best-run table.

| Case | Our frontier, us | Donor published, us | Readout |
|---|---:|---:|---|
| O, m=8 | 5.88 | 5.52 | Candidate small-token route; measure together |
| Pooler, m=8 | 14.15 | 13.55 | Candidate small-token route; measure together |
| Pooler, m=16 | 14.72 | 13.54 | Candidate small-token route; measure together |
| Router, m=128 | 7.17 | 5.77 | Priority comparison, not a promised 1.40-us gain |
| Router, m=256 | 9.67 | 7.35 | Priority comparison, not a promised 2.32-us gain |
| O, m=128 | 7.10 | 6.96 | Near parity across these runs |
| O, m=256 | 9.01 | 9.26 | No demonstrated donor lead |
| Pooler, m=256 | 24.20 | 24.36 | No demonstrated donor lead |
| Unpooler, m=256 | 23.41 | 23.20 | Near parity across these runs |

More useful than the cross-run table is their controlled small-M comparison: they report O8 faster by 0.379 us, pooler8 by 0.601 us, and pooler16 by 0.468 us in eight runs each (four allocation layouts times two seeds), worth about +0.60% on THEIR board. Their unpooler8, router1 and O16 donor routes lost. A selective graft beat a wholesale graft. These savings must be re-established against us; the same mechanisms may already be incorporated in our portfolio.

Assignment: inspect the donor dispatch and generated kernels for these five target cases. Identify the exact differing geometry, packing, synchronization and algorithm choices. Check against our P1/round-2 work in runs.md and ruled-out.md before coding. Benchmark only genuinely distinct survivors, keeping our current production pick, independent duplicate preparations, fixed-policy comparisons and a complete picker comparison. Preserve third-party notices for any copied code.

Acceptance: pass the task's existing numerical and replay requirements, show a paired target-case improvement larger than the independent null variation across held-out inputs and allocation layouts, and recompute the six-load standalone score. Reject losing cases individually; do not require a universally winning donor file. No scored submission is authorized by this report.

## 3. A concrete clue to the failed QKV cluster design

Our check `ad6c673d9a444c1e` rejected the 72-row, four-CTA-cluster QKV design with a 9-12 us tuner regression. Its independent-null timing loop failed, so no precise paired delta or causal attribution was established. The existing report already names residency as a hypothesis.

New external evidence: the donor reports an occupancy query at approximately 200 KiB shared memory per CTA returning only **30 active clusters of four CTAs**. Their capacities for other cluster sizes were 66 clusters of two, 17 of six, and 15 of seven/eight. These are configuration/device-specific results, not universal H200 constants.

Our design needs 32 four-CTA clusters. If our exact launch has the reported capacity of 30, at least two clusters cannot join the initial resident set. Thus 128 nominal CTAs on 132 SMs is not a one-wave proof. NVIDIA's Hopper guide recommends `cudaOccupancyMaxActiveClusters` for this reason.

Cheapest test: on the next already-planned H200 diagnostic, query the exact compiled kernel and launch configuration, including dynamic shared memory and cluster dimensions. Record active-cluster capacity and sampled CTA/cluster start and finish times. Do not spend an optimization run on a presumed residency fix before this is known.

Also new: the donor reports that 1,536 QKV CTA-stages finished in 10.8-11.4 us on 96 CTAs and 10.7-10.8 us on 128 CTAs at m32, leaving at most 0.6 us for its tested fill change, less than its 1.5-4 us fixup tail. This is not our code, but it challenges the premise that idle SM count by itself implies unused aggregate throughput.

Stop rule: confirming a second wave explains a failure; it does not establish that a repaired variant wins. Reopen only with measured aggregate stage-rate headroom AND a full reduction/exit budget that can repay the change. Preserve the current parked status otherwise.

## 4. Dense m256: replace a generic compute-headroom argument with two mechanisms

The donor reports:

- QKV256, cuBLASLt nvjet 72x64: 66 bytes/clock/SM from L2 and 53% tensor-pipe utilization; characterized as transfer-bound.
- O256: the main K loop is already 87% tensor-busy; its 32-KiB-per-CTA epilogue store costs about 0.8 us.
- A QKV256 128x256 split-K-7 implementation on 126 CTAs reached 94% of clock-corrected peak in the K loop but lost 3.3-4.6 us end to end. The reported limitation was partial writes/read-back: about 40-50 GB/s per SM for L2 writes, with 128 KiB writes costing 2.6-4.1 us and peer reads another 1.3-2.8 us. DSMEM reads were approximately 30 GB/s per SM in that experiment.

These are scoped measurements on their kernels. They are not proof that ours has identical limitations, but they are better experiment-design inputs than assuming every gap to 989 TFLOP/s is unused compute.

### QKV256 action

Profile our selected kernel's operand movement, actual traffic, tensor issue and pipeline waits. Establish whether the bottleneck is HBM, L2-to-SM/shared-memory delivery, dependency stalls or a different mapping. Prefer a raw-BF16 schedule/layout intervention with full-K output ownership when it addresses the measured bottleneck; do not introduce large FP32 partials solely to make the tensor loop look better.

The proposed batched/larger-GEMM test remains a diagnostic, but its interpretation must be narrowed. Batching can alter algorithm selection, weight reuse, grid size and cache behavior. A faster per-instance batched result does NOT by itself prove that a single-instance one-wave kernel can recover the difference. Record the actual algorithms, layouts, bytes, batch strides and weight-sharing conditions; use fixed-algorithm controls where supported. Larger-M performance is an upper reference, not a causal isolation.

### O256 action

Separate mainloop time from startup, drain and output stores. A direct coalesced-store or reduced-scratch epilogue is justified only by measured exposed overhead and a correct FP32-to-BF16 final cast. At our frontier C256=83.88 us, even removing an entire 0.8-us epilogue would be only about 0.16% of aggregate projection score to first order; the real removable fraction may be smaller. This is a budget illustration, not an established bound on our implementation.

Acceptance for either: complete-call improvement, not a mainloop TFLOP/s record. Use current Lt picks and independent nulls; preserve asynchronous operand/output lifetimes. A new tile tuple alone is not enough to reopen the already-tested CUTLASS/wd families.

## 5. Denser lossless compression: keep the feasibility study, lower confidence in its ceiling

The donor's facts document reports BF16 entropy around 10.46 bits, roughly 1.3% escapes below its 12-bit format, and consumer issue pressure consuming byte savings. It estimates a free-decode ceiling of about +2.4% and a practical +0.5-1% for its setting. Those estimates differ from our frontier's 3.7% ceiling and must not be averaged or promoted into a universal limit; their exact codec, populations and score model need reconciliation.

The CPU study should report physical packed bytes including headers/escapes/alignment, per-CTA imbalance, bit-exact reconstruction on diverse weights, and extraction/correction instruction budgets. Compare only the affected BF16 weight streams; the FP32/hi-lo router is not automatically another compressed BF16 case. Give an explicit cost estimate for the new decoder rather than assuming decode is hidden.

The donor also reports in-register 12-bit reconstruction losing at m>=128, independent of our shared-tile P3 loss. Therefore moving P3 decoding into registers alone is not a sufficient reopening argument. Its same-SM ALU/HGMMA microbenchmark showed incomplete overlap; do not generalize that to an architectural impossibility of all integer/tensor overlap.

Keep GPU codec work gated by an actual net-byte and instruction budget. Do not call compression the only possible route to top 100: our listed portfolio is incomplete, and a measured streaming line is an empirical reference rather than a theorem.

## 6. Attention and MoE handoffs

**Attention:** the donor explicitly says its shipped attention file is unchanged vjeux. A 2.6530x archived run is not a newly discovered attention architecture. Pin/check its base against the reference already used. Its useful new evidence is negative: one-CTA intra-warpgroup pipelining measured 116.4 us versus 100.5 us, and balancing an SM's two CTAs made their completion times equal without improving the SM's completion time. Compare these mechanisms with our completed E-family work before assigning another overlap/balancing experiment. Do not infer that all cross-tile schedules are ruled out.

**MoE:** KPERM3 permutes K layout and corresponding operand mapping so one NVFP4 scale pair is reused across four k16 steps, reducing redundant decoding without changing codes/scales or represented values. The donor reports 20.12 to 18.44 instructions per weight word, no spill in its revised version, and about +1.25% same-cluster standalone gain. Ask the owner for a static overlap audit against our current grouped-load/scale-hoist/layout work before any check. A port must preserve the activation permutation, required BF16 boundaries and our proven proxy-ordering chain. No additive gain is credited yet.

These are separate owners; projection should not wait on MoE or layer M1 work.

## 7. Revised next-run order and reporting

| Order | Deliverable | Gate |
|---|---|---|
| 1, no GPU | Pinned source-difference inventory for O8/pool8/pool16/router128/router256, plus static KPERM overlap check | Distinguish already present, genuinely different and correctness-blocked mechanisms |
| 2, bounded projection bench | Matched current-vs-donor target cases, independent duplicate states, four allocation layouts where feasible, two or more held-out seeds | Credit only changes exceeding null variation; retain unchanged-shape controls |
| Same diagnostic | Exact failed-QKV occupancy and timestamps; current QKV256 transfer/mainloop and O256 epilogue attribution | No assumed one-wave residency or generic compute-bound label |
| Parallel, CPU only | Denser-code physical-byte/escape/instruction model | Explicit feasibility before GPU budget |
| After evidence | One mechanism-specific dense candidate or selective route port | Complete-call correctness and reproducible standalone gain; layer port separate |

Reserve confirmation capacity within the live shared quota. No new quota allocation is implied here. Preserve the valid release. Do not change production tolerances, delete fences on test evidence, manipulate allocation layout for a favorable judged run, or resubmit solely to select a high noise sample. Allocation padding belongs in robustness tests, not a way to game scoring.

Record source hashes, selected policies, compiler/resource reports, raw timings and independent nulls. Refresh the official CLI before a rank claim. This report adds references and refines the queue; it does not fill the missing top-100 portfolio budget with unmeasured gains.

## Sources and limitations

Pinned external sources:

- [README and provenance](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/README.md).
- [Published per-shape results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/RESULTS.md).
- [Measured facts](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md), especially sections 1, 3, 7 and 8.
- [Negative results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/NEGATIVE_RESULTS.md), decode section 8, projection sections 1-4, attention section 2.
- [Projection source](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_bf16_proj.py). Header/architecture inspected; not a full source/SASS or ordering audit.
- [NVIDIA Hopper tuning guide, CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html), occupancy/cluster sections.

Own records: [frontier and failed QKV test](2026-10-30-projection-top100.md), [preceding portfolio](2026-10-30-projection-top100-portfolio.md), [projection run log](../../../docs/runs.md), [rejection register](../../../docs/ruled-out.md), and the scored receipt named above.

Gaps: exact X post unread; live donor ranks and current quota unqueried; external microbenchmarks not rerun; donor algorithm selection may differ on our inputs; exact portability and synchronization of imported kernels not yet audited; no current score increase claimed.
