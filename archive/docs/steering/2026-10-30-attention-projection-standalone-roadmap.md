# Standalone attention and projection steering: widen the search, keep separate scoreboards

Date: 2026-10-30. Evidence inspected through pushed main `a48edef4286a2168ccc3cffaec3a942358843204`; the preceding snapshot was `a2bdd7e65a05e5726ec062c999d2eb95425ccd1d`. The last head change concerns the layer's negative-control build, not a new attention/projection performance result.

This is a repository/source audit and an experiment plan. No GPU check, submission, compile, tolerance change, live CLI quota query, or live leaderboard query was performed. The public leaderboard page supplied no readable standings. Some requested individual profile paths were not retrievable through the connector; numerical evidence below is explicitly from the available run logs, task files, and named public receipts, not an invented complete current scorecard. No current top-50 threshold is claimed.

## Decision and objective change

**Attention and projection are not out of work. Their present implementations have reached diminishing returns in several local tuning directions. Give each owner a standalone research objective and one bounded structural challenger, rather than keeping both owners on layer-integration support.**

The user explicitly wants better standings on BOTH `muse_attention_decode` and `muse_bf16_proj`. For these two agents, this supersedes the older coordinator framing that standalone problems matter only when their kernels reach the combined layer. Keep the combined-layer campaign, but use two independent promotion gates:

1. Standalone promotion: correct, reproducible improvement in that problem's actual weighted score, with authorized submission by the coordinator.
2. Layer promotion: separate complete-layer evidence before importing the change into the layer.

A standalone pooler or router win is not invalidated because the layer fuses those operations into a different front. Conversely, a faster layer-only overlap path is not automatically a standalone projection improvement. Do not weaken correctness or reuse input-dependent outputs to pursue either score.

Do not call any family exhausted merely because another tile/warp-count tweak lost. Every closed family needs a scoped result and a concrete reopening trigger. Conversely, do not rename already-tested variants as new architecture work.

## 1. Use each problem's score, not the layer's hotspot table

Sources: [projection task](../../../muse_bf16_proj/TASK.md), [attention task](../../../muse_attention_decode/TASK.md).

Projection scores five operators at each of T = 1, 8, 16, 32, 128, 256: qkv_gate, o, pooler, unpooler, router. There are 30 cases and six equally weighted loads. Each operation has weight 62, so that common factor cancels within a load ratio. Define C_T = sum of the five operator times; the problem score is the geometric mean of the six baseline/C_T ratios. The unpooler output is only the 1,024-channel slice at T <= 8; do not compare it with the full-width operator.

Attention scores twelve cases: six loads, each sliding/full. C_l = 46*t_sliding + 16*t_full, then the geometric mean of six baseline/C_l ratios. Its own time is much smaller than the full decode layer's time: a saving dismissed as tiny by the layer map can be significant on this board.

For either problem, a proposed change gives:

    score_new / score_old = product_l(C_old,l / C_new,l) ** (1/6)

For one changed attention case, the local first-order value is w*delta_t/(6*C_l); for one projection case it is delta_t/(6*C_T), after cancelling 62. This is arithmetic, not a promised speedup.

### First deliverable from each owner

Produce a current standalone scorecard, not another whole-layer gap map. Include the exact submitted file/hash, best scored receipt, current candidate receipt separately, per-case raw/event and judge-derived time, selected kernel/plan, byte traffic, practical measured floor, and weighted score sensitivity. Query the official CLI for the current rank-50 score and remaining quota when execution starts. If inaccessible, mark it unavailable; do not turn a historical rank or score into today's target.

For comparisons, pin the public code blob, use the same weights/inputs, and validate all arms. Use independent duplicate controls (A1/A2 and B1/B2 where module/prepare effects matter), seeded arm ordering and held-out inputs. Print selected policies side by side. Fixed-policy comparisons isolate kernels; end-to-end picker comparisons test the complete product. A shared-state duplicate alone does not test independently prepared arm bias; see [the layer arm-bias report](2026-10-30-layer-bench-arm-bias.md).

## 2. What the existing evidence really closes

### Attention

The current flat verify kernel is much closer to the pinned vjeux reference than it used to be. In the paired cold benchmark recorded under `e9085c86d89449c2`, full verify-16/32 are 111.14/192.42 us against public 108.99/188.90; full decode-16/32 are 97.38/174.14 against 96.48/172.29. These include that bench's replay overhead and are NOT official scored medians. The remaining reference gap is only about 2.2/3.5 us on full verification. That supports ending some instruction-polishing work; it does not make that public implementation a performance ceiling.

The rejection register makes an especially useful distinction: neither the public flat kernel nor our mode 51 has tried true cross-tile pipelining. The public kernel ends each tile with `wgmma_wait<0>`. Earlier analysis closed cross-tile overlap as an EXPLANATION of the public lead, not as a NEW design.

Already done or closed in their measured forms: DCARRY descriptor carry, PK16/VHOIST, once-path diet, the producer-return/spill repair, simple extra loader warps, the old QKF/EARLY variants, and block-index/arrival-order balancing of two CTAs per SM. Do not repeat the pair-balancing campaign. Current mode 51 already uses native FP8 WGMMA P.V with ones rows for the denominator; proposing to introduce that operand/instruction choice repeats a previously corrected false premise. Keep its existing validated math and rounding.

The mode-49 history diagnostic `2b93c6ecf4f641c6` fixes merge order in a diagnostic path, obtains bitwise repeatability across the tested histories, and scans clean workspace. That is valuable completed diagnostic work, not a new production speedup and not a universal proof for different pipeline protocols.

Sources: [attention log](../../../docs/runs-attention.md), [rejection register](../../../docs/ruled-out.md).

### Projection

The implementation is already a portfolio: compressed skinny kernels, specialized single-token GEMVs, wide WGMMA kernels, library fallbacks, and a CUTLASS pooler path. In particular, the CUTLASS pooler at T=256 is ALREADY implemented: `8bea43d605c74546` identified it; `d6ff8305da474848` and `76051953a96a4168` validated the corrected build. Do not assign 'integrate CUTLASS pooler256' as new work.

The 8-stage wide qkv/pooler choices at T=128 are also already implemented; generic wide BF16 is not uniformly stuck on cuBLAS. Small-token wins include qkv8's reduced stage count, cgemv's planar stream/register policy, o16/unp32 geometry, and LATE_ARRIVE for the applicable 32-token templates.

Repeated local sweeps have rejected many more consumer warps, NACC=2, and simply extending qkv16/32 stages. Compressed cluster split-K was tested on pooler/unpooler in `a9ff49372ffa4eaf` and did not deliver the proposed benefit. This does not establish the result for a narrowly designed qkv split, but it must be included as negative evidence.

Sources: [projection log](../../../docs/runs.md), [current implementation](../../../muse_bf16_proj/submission.py), [rejection register](../../../docs/ruled-out.md).

## 3. Broaden the references, without copying leaderboard conclusions

The existing public reference has been useful, but matching one implementation is not the finish line. Compare mechanisms and per-shape times, not just aggregate ranks.

A concrete independent projection reference is [NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj/tree/solution). Its README records v18, job `66d8b007f6c641ee`, 1.5186x, CI 1.4958-1.5413, 9/9 passed. It explicitly labels its former first place as historical. Its source description includes selectively pinned cuBLASLt algorithms and shape-specific router paths. The [CSV](https://github.com/NathenFernandes/muse-bf16-proj/blob/solution/results/v18_shapes.csv), blob `f3ab30a0eef07c3e3501d0658257efd60dbe1820`, reports:

| T | router us | pooler us |
|---|---:|---:|
| 1 | 3.94 | 13.29 |
| 8 | 4.34 | 13.61 |
| 16 | 4.23 | 13.98 |
| 32 | 5.04 | 16.13 |
| 128 | 6.19 | 20.49 |
| 256 | 8.44 | 24.19 |

These are external recorded numbers, not measured wins over today's code. Use them to choose matched-process comparisons, not to promise the difference. Some of our shapes may already win. Inspect/expand public source offline without executing it during source extraction, retain license/attribution for any reused code, and do not submit a benchmark embedding another entrant's implementation as our production entry.

For attention, the independent [zhebrak archived receipt](https://github.com/zhebrak/muse-spark-kernels/blob/main/winners/muse_attention_decode/result.json), blob `24799957e49087c091f28f858b520142bd0a286e`, records H200 job `68bf03ae195c468f`, 2.24177x, October 28. That is NOT evidence it beats our current aggregate. Examine its per-case/architecture choices only where it has a different mechanism; do not spend checks blindly porting a slower aggregate implementation.

External mechanism references:

- [FlashAttention-3 author explanation](https://tridao.me/blog/2024/flash3/): overlap matrix operations and softmax within/across warpgroups; extra register pressure is the trade-off. Its published H100 workloads are not Muse's H200/hd64/paged-FP8 decode workload, so its speedup magnitude does not transfer.
- [cuBLASLt, CUDA 12.8 documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html): explicit matmul algorithm/configuration selection, layouts and workspace. Use the judge's installed version, not algorithm IDs blindly copied from another toolkit.
- [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html): warp-specialized schedules and resource trade-offs. Restrict implementation to installed SM90-compatible features.

## 4. Attention owner: three substantial directions

### A0 — One comparative scorecard and feasibility pass, not another survey

Run the current public-versus-ours harness across all twelve cases with a healthy independent null and exact plan selection. Reuse recent matching evidence rather than spending a run solely to recreate it. Rank gaps by standalone score value. For the two leading loads, attribute setup, first useful QK issue, steady-state tile work, reduction/finalization, and tail. Compare current code, not a pre-DCARRY or pre-VHOIST snapshot.

This determines which of A1-A3 starts first. A1 is the first substantive throughput challenger; latency-dominated small shapes can instead elevate A3. Do not funnel all effort into the largest raw-microsecond row without score sensitivity.

### A1 — True cross-tile overlap in the current producer/consumer kernel

Target full q_len=8 first, B=16/32. Keep current exact key coverage, QK/P.V operand choices, fixed-frame math and output rounding. Pipeline QK for tile t+1 with softmax/P.V work for tile t, using separately owned accumulator/operand buffers and explicit retirement before reuse.

Why new: QKF/EARLY moved work within the old consumer-copy organization; DCARRY/PK16/VHOIST simplify current tile work. Neither is a two-iteration dependency graph with matrix work in flight across the tile-loop edge. The rejection register explicitly leaves this open.

First artifact: a dependency/lifetime diagram, register/shared-memory budget, and generated SASS for one shape. Show actual cross-tile issue overlap, not just reordered CUDA statements. A static instruction position is evidence of the intended schedule, not a performance measurement.

Reject before judge use if the compiler serializes the overlap, buffers can be overwritten while WGMMA still reads them, or the requested residency cannot support the synchronization protocol. No fence deletion. Then validate fresh inputs and queued replay, benchmark the complete attention call with nulls, and confirm on held-out full contexts. Initial budget: up to two attention checks with several controlled arms. Target at least a reproducible 2% improvement on the targeted full shapes or a modeled >=0.5% standalone score gain. These are proposed engineering gates, not predicted results.

If overlap wins only at one full-context regime, dispatch there. A small-context regression is not grounds to throw away a useful specialized family; it is grounds not to apply it globally.

### A2 — Resource co-design, contingent on A1's measured limitation

If A1 is constrained by registers/shared memory at two CTAs per SM, compare a deliberately different tile/consumer organization that fits one resident CTA per SM against the current two-CTA design. Change work per CTA and the overlap schedule together for a stated resource reason, then factor the changes. Do not merely resubmit old mode 115 or move unequal work between the same two CTAs.

Measure useful work per SM per time, issued operations, register lifetimes, stage occupancy and end-of-grid tail. Larger tiles reduce repeated setup but can increase live state and reduce parallelism; model both. Keep any cooperative grid within actual guaranteed residency. A nonresident software grid barrier is not a candidate. Initial gate: compile/resource model first; at most one paired feasibility check before deciding whether to develop the family.

### A3 — Reduction ownership challenger using the current fast tile core

For a score-important regime whose measured reduction/finalization or residency constraint is material, compare the current one-launch atomic/cooperative scheme with per-split FP32 outputs plus a compact second merge/norm/gate kernel. Include every launch, scratch write/read, reset and final operation in timing.

Why this is a different test from old slow split-attention: retain the CURRENT fast producer/math core and change only reduction ownership and the scheduling freedom it permits. The potential benefit is removing atomic contention/cooperative-residency constraints or enabling a better core schedule, not 'two launches are faster.' If the measured removable cost is below the added launch/traffic budget, do not build it. Never infer a win from the partial kernel alone. Start on long full contexts; keep the one-launch path for short shapes unless complete-call measurements overturn it.

This is a hypothesis and lower priority than A1; a second launch may simply lose. A deterministic merge can also be a diagnostic control, but diagnostic usefulness does not establish a performance candidate.

## 5. Projection owner: four work packages

### P1 — Standalone router and high-token algorithm portfolio (first)

Run matched-process comparisons of our actual selected candidates against the independent projection reference, especially router T=8/16/32/128/256 and dense T=128/256. Audit current direct-cuBLASLt coverage before implementation. A previous CUTLASS geometry sweep is not the same as benchmarking valid cuBLASLt algorithms, layouts and workspace choices for the installed library.

For library candidates, create descriptors and persistent workspace in prepare, enumerate a bounded supported set, validate, benchmark graph-replay cold-L2 performance, then retain choices by shape. Keep fresh activations and outputs; never cache results. Record algorithm attributes, library version, workspace bytes, layouts and selected policy; unsupported configurations fall back safely. Include current wd/CUTLASS picks as controls, not just the old cuBLAS default.

For the router, inspect the donor's exact precision and reduction method before porting. Compare lower-overhead split-K/partial reduction or SIMD/tensor-core paths only where the matched benchmark shows a gap. Preserve FP32 weights or the task-permitted BF16 hi+lo representation with FP32 accumulation; BF16-only router weights are prohibited. Validate outliers and one-hot columns, adjacent sizes and replay-safe scratch resets. A self-resetting atomic output requires a documented publication/reuse proof; use explicit slots/ordered reduction as a diagnostic alternative.

Do not reject a standalone router winner because the layer selects concat. Initial budget: one multi-arm benchmark and one confirmation/shape-dispatch check. Success is a per-load standalone improvement beyond null variation; layer import is a separate decision.

### P2 — Compressed qkv split-K with correct escape ownership

Target qkv at 16/32 tokens ONLY if P1/current profiling confirms its underfilled grid or serial K work is still limiting. Current qkv geometry has been reported as 96 CTAs on 132 SMs; verify the selected geometry before using that premise. More CTAs do not automatically improve completion time: include residency, cluster placement, extra waves and reduction cost in a work model.

The implementation barrier is the compressed path's escape correction across K partitions. Define exactly which partition owns each escape and how FP32 partials are combined before the required output cast. Prove no omission/double correction and exercise one-hot columns at partition boundaries. Bound scratch/reduction overhead before coding a large family.

Prior negative evidence: `a9ff49372ffa4eaf` already tested compressed cluster split-K on pooler/unpooler. Reopening is limited to qkv's measured parallelism, not a claim that cluster split-K is generally untried or beneficial. If the parallelism model cannot pay for the extra reduction, close the proposal without a judge run. Otherwise one narrow prototype and at most two checks.

### P3 — Lossless compressed WGMMA, not another raw-BF16 tile sweep

Target one high-token projection selected by P1's score map. Combine exact 12-bit weight storage with a producer that reconstructs exact BF16 tiles, including escapes, into shared memory for WGMMA consumers. Reuse each reconstruction across an appropriate token tile. Retain a raw-BF16 control with the same compute layout to isolate compression/decode cost from geometry.

Why different: the current small-token compressed path decodes for mma.sync, while large-token wd/CUTLASS paths use different raw-weight pipelines. Repeating raw wd geometry sweeps is closed; changing where lossless decode occurs and how many consumers reuse it is a new mechanism.

It is not a promised 25% speedup. Weight-code bytes shrink, but escape tables, activation rereads, shared-memory writes, decode instructions, barriers and reduced residency can erase the saving. Produce a byte/instruction/resource budget first. Reject if reconstruction costs more than the measured traffic saving; then compare COMPLETE GEMM time. Do not lower weight precision, change router semantics, or move required rounding boundaries. Initial prototype: one shape, two controls, up to two checks. Expand only after a win.

### P4 — Short-shape startup/escape work, only after measuring a remaining budget

Use the independent reference to find a residual startup/escape/reduction gap on small T, rather than repeat NACC/warp sweeps. Existing cgemv stream/cap improvements and qkv8 stage reduction stay as the control. Make a new proposal only with a particular still-exposed dependency and a predicted microsecond budget. An isolated 0.2-us change with the same-sized module null is not a win. This is a reserve queue, not permission for another unbounded tuning matrix.

## 6. Execution order and budgets

One owner per problem directory. They should work independently of the layer's M1 gate and MoE investigations; retain collaboration for ports, not a global dependency that leaves either owner idle.

| Round | Attention | Projection | Coordinator |
|---|---|---|---|
| First | A0 standalone scorecard and exact current-core comparison; A1 compile/lifetime prototype | P1 current winner versus independent reference; bounded Lt/router candidate set | Refresh both live boards/quotas; verify submitted artifacts and reserve confirmation capacity |
| Second | A1 complete-call test; if resource-limited, A2 feasibility; A3 only with measured budget | Confirm P1; start only the higher-value of P2/P3 | Promote standalone winners under existing authorization; layer experiments are separate |

Suggested initial allocation, subject to LIVE shared quota: 4-6 checks per problem for the first round including reference/control/confirmation runs. Batch controlled variants into a benchmark rather than spend one check on each tiny setting. Reserve at least two available checks per problem for correctness/confirmation; do not assume the nominal daily limit is still available. No new scored submission is authorized by this report.

Before each run, write: exact baseline and candidate hashes; target loads; bottleneck evidence; changed mechanism; why prior losses do not already reject it; numerical/ordering obligations; expected score sensitivity; and pass/stop rules. Check a named job's actual receipt. Never treat a diagnostic bench's headline score as the result of each embedded arm.

## 7. Deliverables and exit criteria

Commit two live scorecards and family ledgers, one per standalone problem. Each case needs selected kernel, reference arm, paired delta/null, expected standalone contribution and layer relevance separately. A proposal gets one of: untested, compile-infeasible, correctness-blocked, measured loss, measured standalone win, layer-validated win. Keep raw receipts and code hashes, including losses.

A family is parked only after a decisive budgeted test or a resource/correctness proof, with a reopening condition. An agent saying 'nothing left' must show this ledger and a measured floor or representative family tests. 'We tied vjeux' is not a floor; 'not top 50' is not proof of easy gains either. The next round should establish which different mechanisms improve these exact tasks.

No production test tolerance, proxy fence, input coverage or existing release is changed by this report. Current layer M1/tuner experiments remain separate workstreams. This report changes their relative priority for these two owners, not their correctness requirements.

## Source map

- Own evidence: `docs/runs-attention.md`, `docs/runs.md`, `docs/ruled-out.md`, both problem TASK.md files, current projection implementation, steering index and layer arm-bias report.
- Attention named jobs: `e9085c86d89449c2` (paired current-core comparison), `2b93c6ecf4f641c6` (history/DETM diagnostic), `bf6b8ca9e37341fb` (older QKF/EARLY), `4ff339332fef4aff` (failed pair split). Do not generalize those failures to the different designs above.
- Projection named jobs: `a9ff49372ffa4eaf` (old compressed cluster split), `4098aa2361554229` (8-stage wide improvement), `8bea43d605c74546`, `d6ff8305da474848`, `76051953a96a4168` (CUTLASS pooler already adopted), `1093e22ca2534d7f` (LATE_ARRIVE repeat, as recorded in runs.md; its individual profile path was not retrieved for this audit).
- Public donor receipt/data: Nathen v18 `66d8b007f6c641ee`, README blob `ab8515b4132cabaa592cad9415f08309f81f0c8d`, CSV blob above; zhebrak attention receipt/blob above. Archived results, not current ranks or causal measurements against us.
- New experiments A1-A3/P1-P4 are proposals, not completed work. Bench numbers and predicted score contributions must be re-established against the selected current standalone baselines.
