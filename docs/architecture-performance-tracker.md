# Architecture performance tracker

Last review: 2026-10-29 14:25 PDT. Evidence snapshot: `efddc84`.
First review reconciles progress since `aab7d75`, the snapshot in the
[architecture-family guide](architecture-families-2026-10-29.md).
This is a review of recorded experiments, not a new benchmark run.

## Current family status

Negative timing deltas mean faster. Percentages below are elapsed-time changes against the
named control, not changes in aggregate score. Families overlap; their gains cannot be added.

| Family | Current result | What has actually been measured | Visibility gap |
|---|---|---|---|
| A. Hybrid persistent/legacy MoE | Improving | Skinny 3-slot ring beats 4-slot control on 36/36 paired seeds at 1/8/16 tokens. Large-token mode 37 also beats mode 28. | New skinny and mode 37 results need layer integration evidence. |
| B. Staged grouped MoE | Tested loss | Two-launch pair is +27.9 us at 128 and +24.2 us at 256 versus mode 16. | No newer winning staged design in reviewed evidence. |
| C. Owner-completes tiles | Retained in the persistent descendants | Mode 20's owned phase A saves about 2-3 us at 128 and 4 us at 256 over mode 16. Later modes combine ownership with other changes. | Later gains cannot be credited to ownership alone. |
| D. Dependency-aware scheduling | Dynamic claiming wins; full ready-only scheduler remains untested | Mode 23 claims whole phase B units at run time, beating mode 20 on every normal paired seed across two checks. | It still waits on done_as. This is dynamic work allocation, not a demonstrated queue of exclusively ready jobs. |
| E. Attention producer-consumer pipeline | Component gains; scheduling losses also recorded | DCARRY and the newer PK16/VHOIST package improve full verify at 128/256; run-time CTA pair weighting loses. | DCARRY/PK16/VHOIST awaits layer integration; FIN8 selected-mode replay coverage also needs repair. |

## Latest review: family E improves again

PK16 changes probability packing and VHOIST carries transpose addresses/descriptors per
segment. The combined attention candidate improves full verification beyond DCARRY.
Job `025ccaf9c57a4ba2`, switches introduced in `a50fc0c`, records cold same-process control
116.26 / 204.22 us versus candidate 110.91 / 194.59 at 128/256 tokens. That is **-5.35 / -9.63
us**, or **-4.6% / -4.7%**. Warm deltas are -7.52 / -13.88 us. ABCHECKSUM is 19 passed and
zero failed for every arm. These full-attention cases use batch 16/32, q_len 8, context
80,000 and TP8; MoE routing does not apply. The run-log table supplies no per-delta
confidence interval or independent repeat count. [Raw bench](../muse_attention_decode/profiles/bench_pk.025ccaf9c57a4ba2.log).

Candidate-only run `569b233dfe08414f`, source `c37d381`, passes 16/16 at 2.449x,
95% interval 2.413-2.502. Full verify times 103.65 / 187.67 us are 7.1 / 10.7 us below the
mean of the two previous DCARRY checks. Those are cross-run observations. Repeat
`e9085c86d89449c2`, source `61e3fd4`, uses the same candidate as its judged path and passes
16/16 at 2.504x, interval 2.467-2.542, with 100.83 / 182.24 us. Its machine is faster on
unchanged rows too; the extra headline-score increase is not another code improvement.
[Attention report](runs-attention.md).

Family E's pending package is now **DCARRY + PK16 + VHOIST**. No new layer result exists in
the reviewed records; layer score stays 1.964x. A-D standings and the mode 37/sk8w3s layer
gaps remain unchanged. The no-fence timing probe in the repeat is diagnostic only and is not
an eligible winning candidate; the production fence stays.

The [community audit](steering/2026-10-29-community-audit.md) identifies two validation gaps:
the intended q_len8 FIN8 stress selected mode 15 instead of mode 11 with eight copies, and
the whole-layer comparison lacks enforced per-seed acceptance, reliable arm-order balancing
and independent mutable cache state. These are open coverage/measurement gaps, not recorded
correctness failures of the incumbent. They should be resolved before relying on the next
combined layer comparison. Existing 13/13 layer judge passes remain recorded.

Decode-32 full attention's apparent +8.7 us regression in the older DCARRY comparison is
now attributed to SGLang's bimodal baseline statistic and machine variation on unchanged mode
49. Preserve the observed cross-run statistic but do not label it a demonstrated candidate
regression. [Rejection register](ruled-out.md), [attention analysis](runs-attention.md).

## Previous review: projection context, family standings unchanged

No new A-E or whole-layer result appeared since the 12:24 review. The three family candidates
awaiting layer tests remain mode 37, sk8w3s and ONE_DCARRY. New projection results belong
outside A-E; they do not change the winning-family badges.

Job `f6c80b4ba11048f3`, source `a35709c`, compares the new single-token cgemv with the
pre-change file `b79a627` in one process. Two passes use different inputs, each with 300
randomly ordered rounds and cold L2. New minus control, pass 0 / pass 1:

| Single-token projection | Paired change |
|---|---:|
| qkv_gate | -0.54 / -0.45 us |
| pooler | -0.86 / -0.96 us |
| unpooler slice | -0.35 / -0.22 us |
| o | +0.08 / -0.11 us, inconclusive |

ABCHECK passes on each reported shape/pass. The same-code null stays within 0.13 us.
The implementation combines row-major planes, a single-load escape start and a larger
register allowance for two chunks per thread. It is a combined projection change, not a
MoE ownership or scheduling experiment. Absolute paired control means and intervals are
not quoted here, so no elapsed-time percentage is assigned.

The candidate-only submission `81254e8aed794eab` records **1.444x, 95% interval 1.422-1.466**.
This is the projection problem's score, not the layer's. Layer integration remains pending;
current layer score stays 1.964x. The earlier qkv8 projection port is also still pending in
the port instructions. [Projection report](runs.md),
[paired raw log](../muse_bf16_proj/profiles/bench.f6c80b4ba11048f3.log),
[candidate result](../muse_bf16_proj/profiles/submission.81254e8aed794eab.json).

The proposed 128-token wide concat front was declined without a new run. Its estimated
24.4 us best-case time exceeds cuBLAS's recorded 24.0 us, before accounting for a second CTA
wave. This is an analytical rejection of that geometry, not a measured candidate loss and
not evidence against staged MoE B. [Rejection register](ruled-out.md).

## Previous review: corrected MoE wait measurements

Job `ca43908f02874886`, code `3f67a45`, corrects a diagnostic defect. The timer previously
started after the first potentially blocking try-wait, which could hide waiting entirely.
The earlier claim that fat-path consumers never wait is invalidated. Production instructions
are unchanged, so this is corrected attribution, not a new production speedup or regression.

| Diagnostic observation | 128 tokens | 256 tokens |
|---|---:|---:|
| Phase A weights + activation waits, per consumer warp | 15.73 us | 23.26 us |
| Phase A waits / stage-loop time | 19.6% | 20.2% |
| Phase B waits / stage-loop time | 20.9% | 20.8% |
| Instrumented mode 41 minus production mode 37 | +7.0 us, +4.3% | +6.1 us, +2.7% |

Six paired layer-routing seeds; production controls 164.6 / 227.2 us. The run passes 5/5
judge tests. Counter overhead is material, and per-warp waits cannot be summed into layer
savings. The producers still spend about 70% of phase A waiting for free ring slots.
[Raw log](../muse_moe_w4a16/profiles/bench_fw1.ca43908f02874886.log),
[measurement report](runs-moe.md#fat-path-waits-with-the-first-try-timed-2026-10-29).

A/C/dynamic-claiming production wins remain supported. B remains a tested loss; no new C,
ready-only D, E or whole-layer candidate result appeared in the reviewed changes. The same
three candidates await layer evidence. A proposed per-group load/hoist experiment remains
unbuilt in the [steering report](steering/2026-10-29-fat-moe-waits.md); its predicted savings
are hypotheses. This correction motivates a test, not a new winning-family badge.

## Large-token comparison ledger

| Family / candidate | Control | 128 tokens | 256 tokens | Evidence and scope |
|---|---|---:|---:|---|
| B: staged 19 then 21 | Mode 16 | +27.9 us, +11.2% | +24.2 us, +8.4% | `e221858b020c4dee`, six same-process seeds, complete standalone MoE. Controls 249.4 / 287.3 us. Correctness passed. [Staged gate](runs-moe.md#staged-gemm1-gate-2026-10-29). |
| C inside A: mode 20 | Mode 16 | -2.1 us, -0.8% | -3.8 us, -1.3% | Same job and controls. Other paired checks support the sign. This isolates owned phase A within a fused launch. [MoE log](runs-moe.md). |
| D-related inside A+C: mode 23 | Mode 20 | -16.2 us, -6.6% | -16.1 us, -5.7% | `a9dbec1d98334943`, source `10375cb`, six paired normal seeds; controls 246.2 / 283.7 us. Self-tests pass, spin flags zero. [Dynamic phase B](runs-moe.md#phase-b-claimed-at-run-time-mode-23-2026-10-29). |
| Mode 23 held-out repeat | Mode 20 | -22.9 us, -8.7% | -18.7 us, -6.1% | `36812632dae242ba`, source `ec219f0`, six held-out normal seeds; controls 263.3 / 304.6 us. Skewed routing also improves, -4.8 / -22.9 us. [MoE log](runs-moe.md). |
| A+C+dynamic claiming: mode 26 routing/gather changes | Mode 23 | -5.8 us | -9.5 us | `817eab6401d54a84`, source `ba9a457`, six held-out seeds; self-tests pass. Skewed -4.2 / -7.7 us. [MoE log](runs-moe.md). |
| Same combined design: mode 28 consumer/ring and sorting changes | Mode 26 | -10.6 / -11.6 us | -11.6 / -12.5 us | Two six-seed checks `f928f8a802014e4f` and `13fe68e7ca2e423f`, source `ec62b5a`; correctness passes. [Consumer and ring investigation](runs-moe.md#consumer-stage-and-the-ring-slot-releases-2026-10-29). |
| Same combined design: mode 37 phase A epilogue | Mode 28 | -3.2 / -4.3 us | -4.9 / -5.3 us | `987d024620e44eb7` and `6307d0c91c8d4ada`, source `96f2c43`, held-out seeds 17-22. Correctness passes. Reciprocal and publication-order changes, not a new ownership experiment. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29). |
| E: ONE_DCARRY, cold paired bench | Same attention without DCARRY | -3.65 us, -3.0% | -7.33 us, -3.5% | `4ff339332fef4aff`, full verify only, controls 120.03 / 210.59 us; all arms pass checksum checks and judge 16/16. [Attention bench](runs-attention.md). |
| E: ONE_DCARRY, judged repeats | ONE_PFAST pair without DCARRY | -3.6 us, -3.2% | -7.9 us, -3.8% | Cross-run mean versus mean, `f64e8680b3e441be` / `bf49eceb15a64f81`, source `0d76acc` / `503c0f4`; 16/16 both. Component evidence, not layer savings. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, cold paired bench | DCARRY control | -5.35 us, -4.6% | -9.63 us, -4.7% | `025ccaf9c57a4ba2`; controls 116.26 / 204.22 us, candidate 110.91 / 194.59. ABCHECKSUM 19/0 for every arm. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, candidate-only judged | Mean of previous DCARRY pair | -7.1 us, -6.4% | -10.7 us, -5.4% | `569b233dfe08414f`, source `c37d381`; 16/16 passed, cross-run component observation. Same-code repeat `e9085c86d89449c2` also passes but has machine-speed variation. [Attention log](runs-attention.md). |

Percentages use the logged paired delta divided by the logged control mean. Rounded means can
differ from the mean paired delta by 0.1 us. Rows without a comparable control mean report
microseconds only. Seed counts are reported instead of inventing confidence intervals for the
paired differences.

## Small-token results and rejected changes

For A's new `sk8w3s`, job `75bca9775f6f4b0b`, source `03887be`, uses six seeds per size and
routing distribution. This measures the MoE component using layer-like or standalone routing;
"layer-like" does not mean a whole-layer benchmark.

| Tokens | Layer-like control sk8w4s | Change with sk8w3s | Standalone-routing change |
|---|---:|---:|---:|
| 1 | 35.6 us | -0.9 us, -2.5% | -0.6 us |
| 8 | 66.2 us | -2.4 us, -3.6% | -1.8 us |
| 16 | 84.8 us | -3.1 us, -3.7% | -2.9 us |

All 36 paired seeds favor three slots. Every arm matches the control within 2.1e-5 and spin
flags are zero. The candidate entered the standalone variant list at `d0c062b`.
[Evidence and port status](runs-moe.md#skinny-weight-stream-against-the-public-kernel-2026-10-29).

Losses matter as much as wins:

- A's unhinted skinny weight copies lose 1.2-3.3 us across the six size/routing rows in that
  job. Early refill with three slots also loses in every row. Scale-decode and helper-handoff
  changes in `e937013292b24906` were inconclusive and not adopted.
- Phase A mode 35 reduces spills but loses +3.0 / +4.7 us at 128/256 versus mode 28.
  Mode 39's paired stores lose +1.0 / +0.2 us versus mode 37 in `473d87d03de94e96`.
  A shorter epilogue alone does not establish a faster kernel. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29).
- E's run-time pair weighting loses +9.0 / +12.0 us at full verify 256 for weights 1.40 / 1.50
  in `4ff339332fef4aff`. The arrival counter fails to predict scheduler priority. This is an
  attention scheduling loss, not evidence against MoE family D. [Attention log](runs-attention.md).

## Whole-layer outcome and attribution

Latest recorded layer submission in the reviewed log is `9b921febc7ad4140`, source `a923f11`:
**1.964x, 95% interval 1.935-1.994, 13/13 tests passed**. The original fenced control was
1.782x. Their ratio is about 10.2% higher score across different runs and multiple changes;
it is not a controlled estimate for one family. [Layer run log](runs-decode-layer.md).

That latest run combines fused route/front scheduling with a separate single-token projection
change. Its same-process front-plus-MoE comparison saves 2.9 us at 128 and 6.1 us at 256.
These are layer-boundary changes and must not be credited to the ready-job scheduler.

Mode 28 has reached the layer: `0fae646889864106` records its selection at 32/128/256 and
1.930x. The later producer fast path ONE_PFAST reaches the layer in `a80dc97048a940d7`, 1.934x.
Those aggregate scores alone do not isolate either change's gain. The earlier DCARRY component
submission is `4c12b0199efd4b8c`, 2.412x, versus the earlier ONE_PFAST entry's 2.416x. The slight
headline decline coexists with repeated improvements on the targeted verify shapes; aggregate
component scores and targeted timings answer different questions. [Attention log](runs-attention.md).

The current [layer inliner](../tools/dl_moe_inline.py) lists fat modes 0, 16, 26 and 28.
Mode 37 and the new skinny ring therefore need fresh layer evidence; adding a standalone
candidate does not update the embedded layer kernel automatically. Projection improvements,
including qkv8 staging, are tracked separately in [projection runs](runs.md).

## Decision and next evidence needed

Retain A with ownership and dynamic phase B claiming. B's measured loss still requires a new
mechanism before another staged test is useful. D's useful dynamic claiming result narrows the
next question: whether dispatching only ready jobs can improve on claiming followed by waiting.
The recorded mean dependency wait did not disappear with mode 23.

The next visibility checkpoints are layer results for mode 37, sk8w3s and the combined
DCARRY/PK16/VHOIST attention package, with the validation gaps above resolved. Keep
those changes attributable when possible. Preserve the whole-layer control when combining them.
Read the existing [rejection register](ruled-out.md) before treating any old candidate as new.

## Review history and checkpoint

- 2026-10-29 14:25 PDT: reviewed `615539c..efddc84`, including attention jobs
  `025ccaf9c57a4ba2`, `569b233dfe08414f`, `e9085c86d89449c2` and community-audit validation
  findings. Fetched origin/main; no remote-only commits. New E gains, no new layer result.
  Newest incorporated attention run is `e9085c86d89449c2`; other stream checkpoints remain.
  Existing untracked profiles preserved. Local commit only under the no-push rule.

- 2026-10-29 13:27 PDT: reviewed through `615539c`; fetched origin/main, no remote-only commits.
  Incorporated projection runs `5ff76a62b0044378`, `f6c80b4ba11048f3` and
  `81254e8aed794eab`, sources `803178f` / `a35709c`, reported in `754549d`, plus
  `f88e278`'s analytical concat rejection. A-E logs and layer outcomes unchanged. Untracked
  profiles remain untouched. Updated dashboard context without assigning projection gains
  to a family. Commit is local under this automation's no-push rule.

- 2026-10-29 12:24 PDT: no new experiment evidence since the 11:20 review. Fetched origin/main;
  no remote-only commits. Run logs and tracked profiles unchanged; all 18 untracked result
  JSONs are readable and their run IDs already appear in existing reports. Family outcomes,
  pending layer tests and latest incorporated run IDs remain unchanged. Refreshed review
  metadata only; no new evidence commit under the automation's commit policy.

- 2026-10-29 11:20 PDT: reviewed `d0c062b..14d6da1`, fetched origin/main and found no remote-only
  commits. Incorporated diagnostic run `ca43908f02874886`, source `3f67a45`, reported by
  `eced63d`. No new layer result. Prior candidate timing ledger retained. Dashboard summaries
  and review metadata updated; no kernel, benchmark or submission action. Commits remain local
  because this automation explicitly forbids pushing.

- 2026-10-29 10:24 PDT: initial reconciliation through `d0c062b`. Read current MoE, attention
  and layer run records, architecture baseline and layer build selection. Recorded wins and
  losses above without running GPU work or modifying implementation files.
- Latest incorporated streams: MoE skinny `75bca9775f6f4b0b`; MoE fat `6307d0c91c8d4ada`;
  attention `4c12b0199efd4b8c`; layer `9b921febc7ad4140`. These streams have independent
  histories; use run IDs and source revisions rather than treating this list as time order.
- Untracked profile JSONs were inventoried and readable. Their corresponding historical runs
  already appear in committed logs; they were not staged. This review uses the committed
  experiment reports, not an independent recomputation of every raw sample or memory-ordering
  proof. Later reviews should inspect changed reports and new run IDs, including uncommitted
  results, and explicitly flag any disagreement.
- Superseded "pending" and "not submitted" paragraphs remain in historical logs. Later named
  result rows control the current status. The old MoE footer still names mode 23's board entry;
  the later mode 37 submission row supersedes it. No live leaderboard was queried.

## Visual dashboard

Open the [local dashboard](http://127.0.0.1:4387/session/3de14ef757fbea5b) while Lavish is running,
or open the [standalone HTML](../.lavish/architecture-performance/index.html). It uses the
existing gap-analysis palette and typography stack. Filters separate measured wins, tested
losses, partial designs and candidates awaiting layer evidence.

Hourly reviews update this tracker and `docs/architecture-dashboard.json`, then run
`python3 tools/build_architecture_dashboard.py`. The builder derives the comparison table
from this tracker and embeds its fingerprint. Keep JSON summaries, pending candidates and
review timestamp aligned with the evidence. Layout changes belong in
`tools/architecture-dashboard-template.html`; rebuild and check them in the browser.
