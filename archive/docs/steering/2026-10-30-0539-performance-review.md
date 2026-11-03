# Hourly review: stream-K phase-A retest loses

Evidence snapshot `15cc6b9`. Reviewed 2026-10-30 05:39 PDT.

Job `098949b28fe84bd1`, source `86a633e`, tests mode 72 against mode 51. It replaces
whole-owned phase-A units with mode 16's cost-weighted stream-K partition and last-arrival
fixup while retaining 51's current stage loop, epilogue and dynamic phase-B claims.
This is the split-K alternative to family C inside A, with D-related phase-B claiming retained.
It is not staged family B or a full dependency-ready scheduler.
[Experiment and ordering review](../../../docs/runs-moe.md#stream-k-phase-a-on-51s-components-modes-72-and-73-2026-10-30).

Same-process cold measurements use six seeds 83-88 for each size and routing distribution,
20 graph replays per arm per seed after L2 flush. Layer-like routing is a component workload,
not a whole-layer test. Percentages divide the logged paired mean delta by the control mean.

| Tokens / routing | Mode 51 control | Mode 72 minus control | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 88.7 us | +13.5 us | +15.22% | 0/6 |
| 32 / standalone | 151.0 us | +9.4 us | +6.23% | 0/6 |
| 128 / layer-like | 166.1 us | +12.6 us | +7.59% | 0/6 |
| 128 / standalone | 195.3 us | +7.1 us | +3.64% | 0/6 |
| 256 / layer-like | 220.3 us | +6.6 us | +3.00% | 0/6 |
| 256 / standalone | 227.9 us | +9.2 us | +4.04% | 0/6 |

No paired confidence interval is recorded. Individual seed deltas remain in the
[raw log](../../../muse_moe_w4a16/profiles/bench_sk72.098949b28fe84bd1.log).
The [receipt](../../../muse_moe_w4a16/profiles/bench_sk72.098949b28fe84bd1.json) passes 5/5,
2.193x (2.160-2.226); that judge score is not an isolated mode 72 gain. First-seed output
checks in all six cells differ from 51 by 1.5e-4 to 2.4e-4 relative L2, consistent with the
changed FP32 summation order. Spin flags are zero. The diagnostic condition fails at 256 on
both routings, so the held-out/skewed gate was not run and nothing ships.

Stamped mode 73 versus 55 shows phase-A end spread widening to 15-19 us at 128/256 from 8-15 us.
Fixups are estimated at 9.4-10.9 us per CTA, with about 1.85 cut segments per CTA. The per-share
count path is about 3.3 us and finishing read-back 6-9 us. ABFIX's total subtracts a mean
pre-count time and is an estimate; the per-share and per-finish columns are measured directly.
Unlike the previous claiming builds, this kernel compiles no duplicate loop. Global fixup
round trips and an outdated cost table are hypotheses for the loss, not separately measured
causes. A shorter gather in the stamped twin does not help: the last first-x-ready is later.

C's ownership choice remains favored for this comparison. Preserve the loss at its tested
scope: mode 72's partition plus last-arrival fixup on current components at 32-256 tokens.
Revisit with a fixup well below 1 us per share on the critical path, or a stage-cost table
refitted to the current loop, then compare in-process with ABFIX. A designated finisher or
cuts where the finisher is idle are untested alternatives. [Scoped rejection](../../../docs/ruled-out.md).

A-versus-B is unchanged: the staged pair still needs its phase-B or phase-A-tail trigger.
D's full ready-only queue remains untested. PRQ's E win and previous XPIPE losses stand.
`layer_hp` still has no GPU check; the whole-layer release remains 2.014x. No new attention,
projection or layer measurement appears in this checkpoint. [Layer manifest](../../../docs/release-manifest.md).


The [tracker](../../../docs/architecture-performance-tracker.md) retains all prior results.

Validation: independent source review checked all six timing cells, percentages, receipt,
correctness scope, attribution and links. Dashboard filters, 24 comparison rows and four
pending cards pass executable checks; source fingerprint matches. No layout changes.
No benchmarks, submissions or pushes were performed by this review.
