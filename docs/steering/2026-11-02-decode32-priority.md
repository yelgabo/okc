# User priority: improve decode, batch 32

November 2, 2026. Evidence snapshot: `b325b96546e6fc2d17515866fecc80fb48364afd`. The user explicitly asks to concentrate on `Decode, batch 32 | 32 | 22.149 ms | 11.888 ms | 1.8632x`, rather than continue a diffuse search across unrelated loads. This is the new prioritization for subsequent work. It does not change a running experiment, waive correctness, authorize another dispatcher or grant public submission permission.

## Decision

**Make decode batch 32 (D32, T=32, q_len=1) the primary performance objective. First aim for a reproducible 2.0x on that load, then 2.1-2.2x, while retaining the other five load paths unless independently improved.** These are engineering targets, not forecasts or achieved scores. Verify batch 32 has T=256 and is a different workload.

Put T32 expert computation first because it executes in both sliding and full contexts. Full-context q_len=1 attention is the second major target. T32 projections are secondary, preferably existing independently validated alternatives rather than another generic sweep. Short T1/T8 experiments and PR-only T128/256 allocation compression do not directly address this request and should not take the next dedicated D32 allocation.

Keep current execution ownership: OK Driver is captain/sole judge dispatcher; the existing OKC Developer owns implementation. No new agent, kernel edit, GPU call, quota query, benchmark or scored submission was made for this report. The accepted direct-weight job remains with its current owner.

## 1. Identify the row correctly

The [layer task](../../muse_decode_layer/TASK.md) weights 46 sliding and 16 full layers in each load. The user's row matches the retained public `607eb52bdaf24160` rounded per-cell data in [stage-shapes.md](../../archive/docs/steering/evidence/2026-11-01-g-public/stage-shapes.md):

| D32 cell | SGLang us/call | Our us/call | Count | Our aggregate contribution |
|---|---:|---:|---:|---:|
| Sliding-window decode | 264.304 | 154.820 | 46 | 7.121720 ms, 59.91% |
| Full-context decode | 624.448 | 297.883 | 16 | 4.766128 ms, 40.09% |
| Total | | | 62 | 11.887848 ms |

Baseline aggregate is 22.149152 ms. The rounded cells reconstruct about 1.863176x, consistent with the displayed 1.8632x. This is the weighted decoder load, not an 11.888-ms single CUDA kernel or a per-token latency. The corresponding mean layer time is about 191.740 us. Sliding's individual ratio is approximately 1.7072x; full's is 2.0963x. The longer full-context call is therefore not the only issue: the more frequent sliding calls account for most of the total.

D32 is below the recorded 2.1261x layer geometric mean, so improving it helps that score. All six loads have equal log weight: the lowest ratio does not, by itself, prove the greatest recoverable headroom. We select D32 because of the user's priority and its concrete expensive components, not because the scoring formula disproportionately penalizes its rank. The overall leaderboard is separate; do not invent its numerical aggregation or guarantee a rank movement.

## 2. Numeric targets and contribution to the complete layer score

Using the user's rounded 22.149/11.888-ms row, with baseline and the other five loads held fixed:

| D32 speedup target | D32 time target | Saving versus 11.888 ms | D32 latency reduction | Complete six-load layer score increase |
|---|---:|---:|---:|---:|
| 2.0x | 11.0745 ms | 0.8135 ms | 6.84% | about 1.19% |
| 2.1x | 10.5471 ms | 1.3409 ms | 11.28% | about 2.01% |
| 2.2x | 10.0677 ms | 1.8203 ms | 15.31% | about 2.81% |

The last column is `(target / 1.8632) ** (1/6) - 1`, not a new official measurement. More generally, `C32 = 46*t_sliding + 16*t_full`; a measured change contributes `(C32_control/C32_candidate) ** (1/6)` if the other five loads are unchanged.

A microsecond saved in sliding alone is worth 46 us in the aggregate; in full alone, 16 us; in both, 62 us. Thus a 1-us sliding improvement has 2.875 times the weight of a 1-us full-only improvement. Reaching 2x solely through full attention would require about 50.84 us off each full call. Reaching it through work shared by both contexts requires about 13.12 us per call.

A concrete, non-overlapping **target budget**, not a prediction: reduce expert time by a net 8 us in each context and full attention by another net 20 us. This saves `62*8 + 16*20 = 816 us`, yielding 11.072 ms and approximately 2.00045x. Only whole-layer measurements can establish that component savings reach the endpoint. Do not add profile activities as though they were independent critical-path costs.

## 3. What the actual profile supports

The [fresh unchanged-incumbent profile](evidence/2026-11-02-captain-fresh-incumbent-profile/hotspots.md), owning check `cc474883117947c0`, reports:

| Cell | Major activity | Reported duration |
|---|---|---:|
| D32 sliding | Expert kernel | 73.3 us |
| D32 sliding | Two large SK2 projection rows | 21.4 and 15.6 us |
| D32 full | Full-context attention | 171.4 us |
| D32 full | Expert kernel | 81.4 us |
| D32 full | Large SK2 projection row | 19.8 us |

These are profiler activities from a different run from public 607, not an exact additive decomposition of the user's 11.888 ms. The fresh layer endpoints were about 150.9/309.8 us. They support prioritizing expert work in both contexts and full attention; they do not demonstrate removable stalls, bandwidth saturation or a recoverable percentage.

The historical G0 comparator was already close on this load: weighted 190.719 us versus 607's 191.740 us, using different jobs/inputs. Merely reproducing G0's D32 row is not a route to 2.2x. Existing source and donor comparisons remain useful controls, but new D32 gains must be measured against the actual current incumbent.

## 4. Immediate execution order

### A. Finish the already accepted expert test; read its D32 result explicitly

At the evidence pin, direct-weight job **`0b684a4065354fc5`** is accepted/pending, not a measured win. It uses unchanged K256 packing and register-A matrix arithmetic, replaces the shared raw-weight ring/weight producer with direct global code/scale loads, and preserves the activation producer and existing publication/retirement obligations. It changes B4 as well as MAIN, so T32 is inside the test. See [accepted check and frozen source](../../archive/docs/steering/2026-11-02-driver-direct-weight-first-check.md).

Do not duplicate dispatch, poll another owner's job, interrupt the existing measurement or change its frozen gate. On its terminal, require exact source/backend identity and report D32 sliding/full endpoint times, weighted C32, selected B4 resources and routing workload, alongside the original all-six decision. Preserve its registered outcome even if the new user objective differs.

A strong D32 signal with a large-load regression would justify considering a **separately registered, prospective T32-family dispatch**, retaining the original path elsewhere. It would not retroactively make the broad test pass, nor justify post-hoc promotion from the same receipt. Confirm the actual general shape dispatch and fresh held-out performance before adopting such a combination. If D32 itself gives no useful signal, stop this exact weight-delivery lane; do not rename it or start an automatic cache/prefetch/register sweep.

The K128 and shared-A variants are already parked according to [the latest terminal decision](2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md). Their old geometry/resource screens are not new D32 candidates. This report does not restart them.

### B. Make the next implementation earn a T32-specific case

The expert owner should compare the actual B4/T32 execution, not assume MAIN128/256 costs describe it. Hold input activations, routing IDs/weights, local expert work, BF16 boundaries and output normalization fixed in component attribution. Report the selected code and any admission/fallback changes. The complete layer remains authoritative.

Prefer an actual reduction in weight-delivery overhead, redundant scheduling/partial work, or exposed publication/retirement over a parameter sweep. One runnable, conforming implementation plus a real result is the deliverable; do not require proof of a speedup before an ordinary exploratory measurement. Reuse the existing instrumentation with proven ownership; no new general harness or distorted timestamp campaign.

Do not require this T32 implementation to outperform the incumbent at T128/256. Workload-specific dispatch is valid engineering when based on ordinary shape/config metadata and tested on legitimate neighboring cases. It must not inspect seeds, expected outputs, evaluator internals or select cheaper routes to game work. Preserve valid nearby shapes, including the official batch-33 case, empty/local/skewed routing and replay behavior.

### C. Full-context decode32 attention is the second opportunity, not a generic attention restart

Scope is **batch=32, q_len=1, full context, page=1**, with all keys retained. It is not q_len=8 verify optimization, and the earlier global attention freeze should not be misread as forbidding a separately justified high-value D32 experiment under this new human priority.

First reconcile the exact current full-decode implementation, owned comparator and existing paired/closure evidence. Do not spend another check comparing an unchanged file whose result is already available. Choose a complete changed dataflow with an explicit expected net cost reduction, or a previously verified better complete route with matching numerical/cache semantics. A small carried-address tweak whose total D32 saving is only a few microseconds is secondary to a credible tens-of-microseconds change.

For scale, 20 us off full attention alone saves 0.320 ms from this row: about 2.69% D32 latency or 0.46% total layer score with other loads fixed. This is a useful contribution, not a complete 2x plan. Seven us full-only saves about 0.16% total layer score. Do not import a standalone attention percentage without checking its actual layer interface, cache update, BF16 output boundary and whole-call time.

If the expert experiment is promising, complete its validation first rather than divide the current developer among multiple implementations. The attention design can be queued with a concrete scope, but this report does not create another worker or dispatch owner.

### D. Pause work unrelated to this load

Do not make PR-only T128/256 hardware allocation, T1/T8 pooler/router, or large-verification-only paths the next primary objective. They cannot directly improve D32. Keep their evidence and any independently justified component releases; this is priority reassignment, not a claim those ideas are universally worthless. Preserve all other production paths and existing submission approvals.

## 5. Measurement and decision contract

Use the existing captain-owned facilities. An initial ordinary complete-layer check can establish correctness, actual dispatch and a useful signal. Claim a reproducible improvement only after matched held-out comparisons with independent mutable states/duplicate controls, cold-L2 replay and stable source/policy. Reuse receipt/score readers rather than make a new framework.

The D32 scorecard must retain: `t_sliding`, `t_full`, `46*t_sliding+16*t_full`, raw and adjusted medians, own baseline times, affected kernel/backend, actual expert counts/work and all-six timings. Match natural inputs and routes for causal component comparisons; do not change the judged computation or use altered routing as an optimization.

The first engineering milestone is a reproducible **at least 6.84% reduction in this load** (about 2x against the supplied baseline). This is a milestone, not a mandatory minimum for retaining every useful intermediate gain. A smaller cheap win can be banked; a complex change with only a tiny effect should lose priority. Do not require each independent component to add 1% to the entire layer before it can contribute to this combined target.

Keep all official mathematical/cache/routing/replay gates and ordering/progress proofs. Test all twelve standard cells and required neighbors. If another shape regresses, use a justified general dispatch/fallback or reject the integration; do not silently exchange an improvement at D32 for a worse six-load score. No public promotion follows from a lower isolated profile row, a synthetic occupancy result or passing CPU tests.

No current rank, quota, direct-weight terminal result, new kernel speedup or achieved 2x/2.2x is claimed in this report. No web research was needed to size this source-based reprioritization. The data are the user's row, pinned task, owning run summaries and current committed execution handoff.

## Next-session brief

> Prioritize decode32 (T32, q_len1), not verify32 (T256). Preserve 607 elsewhere. Reconcile the existing direct-weight job 0b684a4065354fc5 first and expose its two D32 cells. Target weighted C32 below 11.0745 ms for the first milestone, then 10.0677 ms for 2.2x using the supplied baseline. Concentrate on B4 expert work shared by both contexts, then full-decode attention. A net 8 us expert saving in both contexts plus 20 us full-attention saving is a target budget that reaches about 2x, not a forecast. Use prospective shape-specific variants and fresh confirmation rather than forcing one large-token design everywhere. Do not allocate the next D32 slot to PR128/256 or T1 work. Report runnable code and measured results; keep existing captain, correctness and public-submission ownership.
