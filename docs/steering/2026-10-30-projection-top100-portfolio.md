# Projection top-100 portfolio: corrected evidence and the next structural tests

Date: 2026-10-30. Evidence snapshot: `c98018f61e9c3715bbc9443d4c41a3207e578bb8`.

This report responds to the user's review of the projection research. It supersedes the projection execution queue in [the earlier standalone roadmap](2026-10-30-attention-projection-standalone-roadmap.md), not the separate attention/layer work. It records an audit and proposed experiments. No GPU work, kernel change, validation-policy change or scored submission was performed or authorized here.

## 1. Corrections accepted; provenance kept explicit

The user's official-CLI snapshot gives the current planning cutoff as **1.5560x**, rank 78 at 1.5616x and rank 103 at 1.5554x. The Guild snapshot is stale. The user reports our rank as 314 of 1491. This audit did not independently query the live CLI; retain the timestamped CLI capture alongside the frontier map and refresh it before claiming a ranking milestone. The two bracketing ranks alone do not reproduce an exact rank-100 row.

The scored receipt is independently verified: [d85d0081fa8a4a43](../../muse_bf16_proj/profiles/scored.d85d0081fa8a4a43.json), **1.4975856117x**, 95% interval **1.4751218275–1.5200493959**, all nine correctness tests passed, no rejected/flagged checks. Its submitted content hash is `751bd89fa5bb314e8f57ade1e816e7b4aa09058720b00407de9a52d370c3acc0`; the run log identifies source `2265fa2`. The previous scored run `d59bc7af342a465d` was 1.4879x at `ce25f67`.

The user's six-receipt frontier uses 1.4947x as a more stable planning baseline. Its full table was not retrieved in this audit, so the aggregate is used as user-supplied evidence, not presented as an independently reconstructed dataset. Likewise, the reported residual donor-router leads are 0.4 us at 128 and 1.3 us at 256. The user identifies vjeux v141's same-process results as 1.516x and 1.506x; those must not be confused with a live leaderboard score. Nathen v18's 1.5186x remains an archived judged result.

Arithmetic:

- From the latest receipt: 1.556 / 1.4975856117 - 1 = **3.9006%** more score.
- From the six-receipt planning baseline: 1.556 / 1.4947 - 1 = **4.1012%**.
- The first ratio corresponds to a 3.7541% uniform reduction of all load times, not a 3.9006% latency reduction.

Public references remain useful controls and sources of mechanisms, but neither their aggregate scores nor their per-shape bests define an optimum. Equally, a gap to a roofline does not reveal what higher-ranked entrants implemented. The suggestion that leaders improved m=256 is a hypothesis, not an established competitor finding.

## 2. Remove completed work from the queue

Confirmed in [runs.md](../runs.md), especially the P1, P3 and split-exit-barrier entries:

| Work | Current disposition |
|---|---|
| Direct cuBLASLt enumeration, layouts, tile/cluster candidates, FP64 checks and stable picks | Shipped. Do not restart a generic library survey. Reopen only a specifically missing supported configuration or changed mechanism. |
| Small-token rt and large-token KLUE wd router | Shipped, including conforming cluster lifetime handling. Remaining donor comparison is incremental work. |
| Lossless compressed skinny projections at 1–32 tokens | Existing core, not a new optimization. |
| Shared-tile compressed WGMMA P3 at pooler 128 | Parked after measured loss; no second generic compression attempt. |
| Split exit cluster barrier | Shipped in the 1.4976x receipt. Do not count its gain again. |
| QKV one-wave cluster split-K | In progress per the user; preserve ownership and complete its decisive test. |

P3's negative evidence is stronger than just expensive decoding. Check `4922f366a0c142ca` measured the raw twin at about 35.6 us against Lt at 20.45 us; the compressed twin took about 48.5 us. Compression added about 12.8 us while the memory-only contrast saved about 1.0 us. Even the raw twin's memory-only pipeline exceeded the complete Lt GEMM. A new dense family must first compete in raw BF16, without inheriting that losing geometry or reduction.

The earlier CUTLASS sweep `8bea43d605c74546` already tried pingpong and cooperative/Stream-K configurations, including 128x64 and 128x128 classes. [ruled-out.md](../ruled-out.md) scopes those losses. A familiar tile tuple with no changed scheduling, layout or epilogue mechanism is not a new experiment.

## 3. A portfolio budget that actually reaches the target

Projection has six equally weighted loads; each load sums five operators. Their common multiplicity 62 cancels. For combined candidate load times C'_T:

    score_new / score_old = product_T(C_T / C'_T) ** (1/6)

Do not add separate projected score gains for changes to the same case. Combine their operator times, then recompute each load. Distinguish measured savings, predictions and unassigned budget.

The following is a **required-outcome envelope, not a performance forecast**. It allocates a little more than the required gain, without assuming a single breakthrough:

| Tokens T | Required reduction of the complete five-operator load | Intended contributors |
|---|---:|---|
| 1 | 0% | Preserve incumbent; do not require a decode-1 win to make the plan work. |
| 8 | 1.0% | QKV candidate if useful here, otherwise a measured short-shape startup improvement. |
| 16 | 2.9% | Primarily the in-progress QKV split-K. |
| 32 | 4.3% | Primarily the in-progress QKV split-K. |
| 128 | 4.0% | Remaining router gap plus a demonstrated dense scheduling/epilogue improvement. This budget is not yet earned. |
| 256 | 11.5% | Dense GEMM family, with remaining router gain counted only once. |

These reductions give a multiplier **1.04192515**, or **1.55737x from 1.4947x**. It is only a mathematical path across the current planning cutoff, with little margin for cutoff drift or noise. The coordinator must replace each row with actual microseconds from the frontier, then with paired measured deltas. If a row fails, expose the remaining deficit rather than claiming the others already sum to the target.

The user's -1.4 us QKV16 and -2.3 us QKV32 predictions account for approximately +1.2% score in their frontier model. They are hypotheses until tested. The 128-token budget cannot be supplied by the 0.4-us router gap and the o128 pin alone. The run log places the latter at at most about 0.1% overall score, with no single stable winning configuration across six preparations.

## 4. Finish QKV split-K, with a real residency gate

The proposed 72 rows per CTA and K split across four CTAs yields `(2304 / 72) * 4 = 128 CTAs`, arranged as 32 clusters. This is a sensible scoped reopening of the earlier excess-wave designs.

But **128 CTAs on 132 SMs does not prove one resident wave for a clustered kernel**. Check `cudaOccupancyMaxActiveClusters` for the exact compiled kernel, block shape, dynamic shared memory and cluster dimensions on the judge node. The intended all-resident launch needs capacity for at least 32 such clusters. Also measure CTA starts/ends in a diagnostic build; verify instrumentation does not change allocation. This is a performance condition, and would also be a progress requirement for any added global synchronization.

Before timing, independently verify escape ownership: every K partition contains exactly its corrections, including partition boundaries, unbalanced escape locations and one-hot input columns. Combine FP32 partials before the required BF16 output cast. Document remote shared-memory completion and cluster exit, and preserve the already reviewed fence/barrier obligations. Exercise adjacent sizes, outliers and queued graph replay.

Use independent duplicate modules/preparations, randomized order and held-out weights. Keep dispatch per shape; a win at 16/32 need not be applied at 8. Stop this prototype if reduction/setup erases its parallelism benefit, recording the actual residency rather than a nominal CTA count.

## 5. What 800 TFLOP/s can and cannot buy at m=256

For dense BF16 operators, useful work is 2*T*N*K. From the task shapes:

| Operator | N | K | GFLOP at T=256 | Compute-only us at 700 TF/s | At 800 TF/s | At 989 TF/s |
|---|---:|---:|---:|---:|---:|---:|
| qkv_gate | 2304 | 8192 | 9.664 | 13.81 | 12.08 | 9.77 |
| o | 8192 | 1024 | 4.295 | 6.14 | 5.37 | 4.34 |
| pooler | 4096 | 8192 | 17.180 | 24.54 | 21.47 | 17.37 |
| unpooler | 8192 | 4096 | 17.180 | 24.54 | 21.47 | 17.37 |

This is a counterfactual compute-equivalent table, **not new measured kernel times**. Across all four, 700 to 989 recovers about **20.17 us**; 700 to 800 recovers only **8.63 us**. Starting uniformly at 650 instead gives **13.94 us** to 800. Actual operators have different effective rates. Startup, pipeline drain, output conversion/store, repeated L2 traffic and memory limits are not free; model and measure them explicitly.

Keep two throughput metrics: steady-state useful arithmetic per loop time, and useful arithmetic per complete-call time. A loop exceeding 800 TF/s can still lose the GEMM. Conversely, o has only K=1024 and is a particularly poor target for demanding an arbitrary complete-call TF/s number; shortening startup/drain/epilogue is the better hypothesis there.

The latest receipt's profiler identifies a QKV256 symbol containing `72x64_64x12_1x1`, pooler256 `128x64_64x8_1x1`, and o/unpooler256 `128x128_64x6_2x1...coopA`. These names warrant an incumbent audit. They do not alone establish exact operand orientation or scheduler behavior. In particular, do not claim that a 72-by-64 QKV tile is novel just because a proposed kernel launches 128 output tiles.

## 6. New dense workstream: owner-completes raw BF16, schedule and epilogue co-design

### Candidate shape map, not a promise of 800 TF/s

Coordinates below are **token rows x output channels x K chunk**; this differs from swap-AB libraries that name weight rows first. Initially use cluster size 1 and no K split: each CTA owns complete output tiles. This avoids importing the old DSMEM reduction cost. Start with pooler and unpooler, which have larger absolute budgets; inspect QKV and short-K o second.

| Operator at T=256 | Candidate logical output tile / K chunk | Output tiles | Initial scheduling hypothesis |
|---|---|---:|---|
| qkv_gate | 64 x 72 / 64 | 128 | One consumer warpgroup; no padded output-channel work. Existing library symbol may already implement this geometry. |
| pooler | 64 x 128 / 64 | 128 | Direct full-K owner, one consumer warpgroup initially; remove only demonstrably exposed scheduler/epilogue overhead. |
| unpooler | 128 x 128 / 64 | 128 | Two consumer warpgroups cooperate on different 64-token halves of the same tile, with independent accumulators and shared input staging. |
| o | 128 x 128 / 64 | 128 | Same output ownership, but specialize short K for a smaller startup/drain budget. |

BF16 `wgmma.m64n72k16` is supported in the CUDA-12.8.1 PTX ISA; the orientation must put 64 on hardware M and 72 on N. That does not prove a legal efficient TMA layout: account for descriptor alignment, swizzle padding, leading strides and output coalescing before compiling.

These tuples are feasibility sketches. Some correspond to already tested CUTLASS geometries. **Before implementing, name the specific difference from the old losing build and the current Lt winner.** Acceptable differences are a documented shared-memory/mainloop layout, retirement policy, work assignment, or epilogue pipeline with measurable removable cost. If the proposed code is just the same old configuration, do not spend a check on it.

### Pipeline and epilogue choices

For BF16 A/B tiles, unpadded shared-memory staging is `2 * BK * (BM + BN) * stages` bytes. A 128x128 tile with BK=64 uses 32 KiB per stage: four/five stages use 128/160 KiB; six use 192 KiB before epilogue storage. A BF16 128x128 output tile adds 32 KiB. Barriers, alignment and swizzle padding still count. Start with four versus five stages; do not select a six-stage ring solely because the arithmetic sum almost fits the 227-KiB block limit. Shared-memory reuse must follow retirement of every outstanding reader/writer.

For 64x128, the unpadded stage is 24 KiB. Four versus six stages is a reasonable bounded comparison. A second factor, only if allocation allows it, compares equal lookahead in K, such as BK=64 with six stages versus BK=128 with three. It separates per-stage synchronization from total buffered K; it is not a broad depth sweep.

Use a real producer/consumer TMA pipeline and retain FP32 accumulation. Inspect WGMMA issue/retirement in SASS: an unnecessarily immediate wait after every issue can prevent overlap, but removing waits without buffer-lifetime proof is prohibited.

Specialize the actual operation `D = A*B`: alpha=1, beta=0, no unnecessary C fetch, final required BF16 conversion. Compare coalesced vector stores with the existing TMA/shared-memory epilogue only where measurement exposes a drain. Retire TMA stores and remote accesses before buffer reuse or CTA exit. No extra numerical fusion, no reduced-precision weights, no sparse shortcuts.

Persistent pingpong schedules overlap one consumer warpgroup's epilogue with another's next tile. With very few tiles per scheduled CTA, that benefit may be limited. Measure actual tiles per CTA; 128 logical output tiles does not imply 128 persistent CTAs. Test a static direct-output schedule only if it removes real overhead without sacrificing useful reuse/overlap. Do not assert that persistence or pingpong is universally inferior.

### Cheapest decisive test

1. **No GPU quota initially:** audit the selected Lt configuration and the previous CUTLASS build; produce work/byte/resource counts and a SASS-supported mechanism delta. Compile one pooler and one unpooler prototype against the installed toolchain, not a newer unsupported API. Reject spills, serialization, illegal TMA layouts and resource overcommit before benchmarking.
2. **One multi-arm feasibility check:** current selected Lt (not default cuBLAS), the prior best relevant custom/CUTLASS kernel, and at most a few factored new schedules. Independent A1/A2 and B1/B2 states; cold-L2 graph replay; fixed weights/activations across arms, then held-out matrices; correctness assertions. Print actual policies. Distinguish kernel timing, replay overhead and judge-derived times.
3. **One confirmation if promising:** complete-call savings beyond nulls, with all tests and adjacent shapes. An engineering expansion gate is at least approximately 5% on one targeted large operator or a modeled 0.5% standalone score gain, followed by a reproducible combined-load result. These are budget gates, not correctness tolerances or promised outcomes.

Measure prologue, first math, steady-state, drain and stores in a separate diagnostic binary. Check its register/shared-memory allocation and perturbation against production. Knockouts identify possible budgets, not additive causal savings. Stop a schedule that improves loop TF/s but loses complete-call time. Add cluster-2 multicast only after measured redundant traffic justifies its residency/synchronization cost; do not reopen the old multicast matrix blindly.

## 7. Reserve work: small residuals and a genuinely different compression question

**Router residual:** use the reported 0.4/1.3 us gaps as limited budgets. Compare the remaining mechanism with the conforming split-exit implementation. The shipped tail-barrier improvement is not another available saving. Keep this lane subordinate to QKV and dense work.

**o128 picker:** the run log's 0.13–0.45 us finalist advantage varies by layout/custom/cluster across preparations. Retain robust selection; do not pin the last winning draw. This is a reserve item worth at most about 0.1% overall score according to the log.

**Offline-only proposal, not yet evidence:** ask whether a narrower *lossless* code improves genuinely bandwidth-limited small-T cases. The present 12-bit representation uses a low byte plus sign and a three-bit exponent-window offset. An 11-bit candidate could use sign plus a two-bit offset and exact escapes. The ideal code-stream saving is only 1/12, or 8.33%, before metadata, padding and additional escapes. Count actual physical bytes and worst CTA escape imbalance for multiple weight sets, verify every reconstructed BF16 bit, and retain a raw/current fallback. A proposed screen is at least 5% net byte saving plus a credible decoder instruction budget before any GPU check. Bit-packed extraction and extra escape work may erase everything.

This is a reserve mechanism on the current small-T consumer path, not permission to repeat failed shared-tile P3 at m=128. It remains untested. Do not assign a GPU sweep until QKV and the dense feasibility test have returned; never lower weight precision to manufacture a compression win.

## 8. Execution, promotion and reporting

Run QKV and the raw-dense feasibility design as separate owner tasks; preserve both standalone and layer releases. Keep router/picker cleanup and offline compression in a small reserve queue. Do not reassign the projection owner to layer M1 support merely because a standalone mechanism does not port directly.

Reserve confirmation capacity before exploratory checks. This report does not claim remaining quota; the coordinator must read it through the working official CLI. Each experiment records the exact incumbent, candidate hash, changed mechanism, previously tested nearest neighbor, per-case delta/null, complete-load gain, proof obligations and stop rule.

The user's milestone gate remains: check at or above 1.556x, repeat at or above approximately 1.550x, all tests passing, and no load more than approximately 1% slower without a measured compensating overall gain. Apply it to a frozen candidate, not a succession of different bundles. Repeated sampling until one lucky score crosses the cutoff is not validation. Add matched-incumbent evidence beyond independent nulls, and inspect raw per-case timings for baseline drift.

This gate is an engineering milestone, not statistical proof that true performance or current rank is above the cutoff: judge intervals have a minimum width and the cutoff moves. Refresh the live board on an authorized scored submission. Smaller reproducible gains may still be promoted under the user's existing submission approvals; do not make top-100 success a prerequisite for banking every incremental win.

Required next handoff: the QKV ownership/residency verdict; a raw-dense mechanism and feasibility result; and an updated portfolio table in microseconds with measured, predicted and still-unassigned contributions. Include the six-receipt frontier and timestamped CLI snapshot so other sessions can reproduce the ranking arithmetic. Do not report an achieved 4.1% portfolio until the combined implementation is measured.

## Sources and coverage

Repository evidence pinned to the snapshot above: `AGENTS.md`, `docs/steering/README.md`, the earlier standalone roadmap, `docs/runs.md` (P1, P3 and split-exit entries), `docs/ruled-out.md`, projection task shapes and receipt `scored.d85d0081fa8a4a43.json`. The receipt's profiler rows were used only to identify selected symbols, not to synthesize a scored frontier.

External primary sources consulted for mechanisms:

- [NVIDIA Hopper tuning guide, CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html): cluster occupancy query, register/shared-memory resources and TMA. Cluster occupancy must be evaluated for the actual H200 launch.
- [PTX ISA 8.7, CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html): BF16 WGMMA supported shapes, asynchronous lifetime and synchronization requirements. The shape table includes m64n72k16.
- [NVIDIA CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html): cooperative versus pingpong scheduling and the relationship between tile work and epilogue overlap. Design reference only; compile against installed supported CUTLASS.
- [CUTLASS 4.5 GEMM API](https://docs.nvidia.com/cutlass/4.5.0/media/docs/cpp/gemm_api_3x.html): separate mainloop/epilogue construction and scheduling parameters.
- [NVIDIA matrix-multiplication performance background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html): 2MNK work model and memory/compute limits. The 700/800/989 calculations here are explicitly hypothetical.

Not verified here: a new live leaderboard query, full six-receipt frontier, current quota, the in-progress QKV source/benchmark, any claimed leader architecture, or an 800-TF/s custom kernel. Those omissions do not undo the committed receipt or measured P3 loss; they limit what this report can claim about future gains.
