# After both sparse-repair losses: ranking and one next investigation

November1,2026, official board reads completed around18:21UTC/10:21PST. Read-only addendum; no kernel change, GPU dispatch or public action. Root continues its separately owned stage-local decision and reset sequence. This report incorporates its terminal result rather than treating it as pending.

## Actual feasibility

Official CLI snapshots and source hashes are preserved in [identity](evidence/2026-11-01-post-xor-pivot/identity.json). Overall was queried directly: rank318, ranks-only API. Its first500 now differ from the layer list at95 ranks, starting121; our rank and the target rows do not establish a universal numerical overall formula. The following cutoffs are **problem-board** values, not invented overall scores. Ratios use rounded CLI values and are approximate required score increases, not latency forecasts.

|Problem|Our rank / score|Rank50 score / required increase|Rank10 score / required increase|
|---|---|---|---|
|Layer|318 /2.1261×|2.2907× /7.74%|2.4035× /13.05%|
|Projection|319 /1.5278×|1.5858× /3.80%|1.6090× /5.31%|
|Attention|228 /2.6043×|2.7208× /4.47%|2.9331× /12.63%|
|MoE|396 /2.2157×|2.4584× /10.95%|2.6077× /17.69%|

The earlier attention4.13% rank50 gap is stale. No measured portfolio establishes top10 or top50 before the November4 deadline. This does not prove impossibility; it says the pending work cannot be credited with unmeasured gains. Preserve layer607eb52bdaf24160/sourceba9e327c… and projectionC2/9eba96894ee1402e as known-good incumbents.

## What the two new negatives close

Global-record sparse repair `2b537518091248f1` passed9/30 and all10 active overrides but scored1.394120071×. Exact stage-local `dfcfdd01d6f04525` also passed9/30, clean checks/no flags, all10 active/0fallback,160startup math checks,10captures/30fresh replays/110installed keys; prepare129.385967s and score1.419146839×,CI[1.397859637,1.440434042]. Both are below C2's1.527806483×. Source/CPU correctness and register/occupancy admission were not performance evidence.

I independently verified the stage job ID/board/public mode/result and all thirty benchmark records against the owning raw receipt. Every one of its ten target cells has higher adjusted **and raw** latency than archived C2; adjusted increases span6.27–40.74% ([rows and receipt hash](evidence/2026-11-01-post-xor-pivot/stage-terminal.json)). These are cross-receipt comparisons, not a paired attribution. The higher stage-local headline than the failed global-record run is not evidence of a causal recovery mechanism. Close both exact representations, no post-hoc subset promotion, cap/tile/after-A sweep, private duplicate retry or layer port. No new inference that HBM, stage credits, decoding or MMA is the bottleneck follows.

## Strongest remaining comparator does not equal a ready candidate

[Receipt-owner ledger](evidence/2026-11-01-post-xor-pivot/comparator-owners.json) re-verifies all four existing comparator IDs and scores. Layer G0 `3a5f879df0d44825` at2.248790532× remains the largest measured layer discrepancy versus607e, but it is bundled with disabled deferred-GLE, different precision reductions and unsafe/uncertified completion paths. The [contract audit](2026-11-01-g-fork-precision-contract.md) prevents mislabelling all internal BF16 partials categorically forbidden; it does not make G0 an approved replacement or attribute the score gap. Even that entire external headline is below today's layer rank50 cutoff. Its isolated dense gaps are already sized at only+.48455% current-G six-load arithmetic, not a critical-path saving.

Projection comparator `3c6a2baf7b4b4bf3`1.573015516× has its largest gaps in BF16 split-reduction routes and the already-investigated router; the four positive largest cells supply only2.257% cross-receipt sizing. It is below today's projection rank50. Attention `cd6805e7ebce45ee`2.639606424× is below the new rank50 cutoff; the genuinely BF16-matched VP2 target signal `c79ddddb840a4fb7` wins one cell5.71% but fails the registered1% whole-score gate. MoE `b65c4c878014416c`2.395702934× is below its current rank50 and is not equivalent to the repaired kernel. Earlier qualified fixed-owner finalizer frontier `1c6ebc76b91d4b7f` had most-favorable whole-score sensitivities only1.003445–1.004271; the later detailed observer was distorted. Neither permits reopening MFIN or claiming local barrier waits are SM-idle budget.

This is a scoped comparison of artifacts measured by us, not a new exhaustive public-source survey. Unowned receipts/README ranks are not measurements to bank. The existing changed-source discovery ledger remains the reopening trigger.

## One concrete next investigation

**If choosing the next engineering investigation after stage-local loss, use the already prepared, fixed-natural-routing serial FRT-versus-serial-split short-path diagnostic.** Preserve the registered reset order: (1) repaired full-six-load stage comparison, (2) this short-path question. Do not replace either with another projection codec or the closed concurrent producer.

The precise question is whether the existing fused BF16 pooler/FP32 router producer's interleaved responsibilities extend the **whole captured short-load critical path** enough that ordinary serial separation improves it. This compares complete legal interfaces with exactly identical natural expert IDs and private mutable owners. It is distinct from942ebdec400d4e68's two-stream fork, whose decode1 regressed and whose verify1-only substitution gave about+.355% sizing. A single18us profiler row is not18us removable work; kernel-duration sums, unqualified device timestamps or independent-context subtraction cannot answer this question.

The reviewed tools are `dl_short_path_trace.py`, body, analyzer and tests. Before any future root dispatch, regenerate from exact607e and recheck current dependencies/byte pin,96math checks/48rows, three held-outs, busiest local expert group, native backend, private state/cache/guards, no input-derived caching, unchanged production return state,295s guard and full report retention. The normal path is retained. External profiler traces supply dependency ordering; unprofiled full-graph randomized timing and duplicate nulls decide performance. All fences and BF16/FP32 stage boundaries remain.

From owning607e, weighted decode1/verify1 times are78.549045/115.990823us. [Exact score sensitivity](evidence/2026-11-01-post-xor-pivot/short-score-sensitivity.json) says equal savings on both short loads must reach2.753184us each for1% six-load gain; if confined to sliding cells,3.710813us each. Decode1 alone needs4.552291us, verify1 alone6.722221us. These numbers hold the other loads fixed and are thresholds, not observed/exposed savings.

Accept further candidate engineering only if all three held-outs complete and every four-cross full-score factor exceeds1.01 and2×duplicate log-null with the frozen per-cell regression limit; routing/math/cache/backend/resource/ownership/report gates must pass. A positive producer microbenchmark without that full-graph gate is insufficient. Pure single-arm official qualification and whole-score measurement remain separate. If the serial comparison also misses the gate or is invalid, stop that FRT/separation scope and retain incumbents; do not automatically allocate the next slot to another fusion/stream/warp knob. The four-cell verify diagnostic remains later in order, with source-transparent production packaging still NOFIT, so it is not a shippable fallback plan.

The actionable pivot is therefore a specific already-reviewable whole-layer producer question with a quantified score gate, not a broad new backlog or a promise that another1% will close a7.74% layer gap.
