# Hourly review: attention merge win and conforming projection results

Evidence snapshot `16573a6`. Reviewed 2026-10-30 03:37 PDT.

A-E outcomes and the A-versus-B decision at 128/256 remain unchanged. `ONE_HMRG` is an
attention key-half merge before global reductions, not the family E cross-tile pipeline or
MoE family C. It is a standalone component win awaiting a separate layer port/test.
The XPIPE cold losses remain valid. No new MoE or whole-layer result is recorded.
[Attention round 2](../runs-attention.md#standalone-round-2-10-30-a3-merge--finalize-and-the-batch-1-full-prologue).

### Separate attention context: HMRG

`ce92d32e26024bd2` at `d8d43f3` first reports HMRG cold ABSCORE 1.0180 against the
board-code control (duplicate null 0.9993). Per-input rows were truncated, so the sign gate
could not be checked from that log alone. Confirmation `d0819bb26907426b` at `9eafafc`
reports 1.0214 with the control's workspace, 1.0232 with an independent workspace, versus
null 1.0037. Both benches pass 16/16 judge tests and ABCHECKSUM 19/0 for real arms.
The bench uses 120 randomized rounds, two held-out inputs, seeds 1000/1001; routing distribution
is not applicable. Logged pooled cold deltas and control medians from the confirmation:

| Decode shape | Control | HMRG delta | Delta / control | Per-input deltas |
|---|---:|---:|---:|---|
| 1 token sliding | 9.89 us | -0.64 us | -6.47% | -0.54 / -0.70 us |
| 1 token full | 18.69 us | -1.54 us | -8.24% | -1.50 / -1.50 us |
| 16 tokens sliding | 14.34 us | -1.44 us | -10.04% | -1.47 / -1.41 us |
| 32 tokens sliding | 16.61 us | -0.83 us | -5.00% | -0.86 / -0.86 us |

Pooled deltas differ from per-input summaries. No paired confidence intervals are reported.
The independent-state result supports the gain despite the known decode-1 full placement
spread. HMRG changes only q_len=1 modes 1/3/11/13; 128/256-token verify paths have identical
code, so their timing movement is not an HMRG gain. It halves the q_len=1 global reductions
by merging key halves in shared memory first. The recorded independent ordering review finds
the added waits/barriers sound within its source scope.
[First bench](../../muse_attention_decode/profiles/bench_e.ce92d32e26024bd2.log),
[confirmation](../../muse_attention_decode/profiles/bench_f.d0819bb26907426b.log).

Candidate-only checks `5d309f9d13924dca` (`dc6b409`, SHA256 prefix `93297101`) and
`82b1bf0fad3244b9` (same code plus a comment) pass 16/16, scoring 2.512
(2.475-2.550) and 2.582 (2.543-2.620). These are cross-run observations; unchanged verify
paths also accelerate in the repeat. The coordinator's scored run `57b0bed35f694a23`, same
source, is logged as passed at 2.479x; its receipt, interval and test count are absent locally.
The attention board remains `e9085c86d89449c2` at 2.504x, a bench-file entry. The scored HMRG
result is the best clean candidate-only scored run recorded, not a board replacement.
[Check receipt](../../muse_attention_decode/profiles/hmrg.5d309f9d13924dca.json),
[repeat receipt](../../muse_attention_decode/profiles/hmrg_r.82b1bf0fad3244b9.json),
[scored log](../runs-attention.md#scored-57b0bed35f694a23-2479x-2026-10-30).

Preserve rejected alternatives: HMRG plus last-CTA ticket (`ONE_T9`, `hmtk`) adds 8.67 us
at verify-1 full and 1.79/1.70/1.41 us at verify-1/16/32 sliding versus control in
`ce92d32e26024bd2`. Its additional 0.22 us saving at decode-1 full is inconclusive under
the placement null. Removing cooperative launch gains only 0.38 us at decode-1 full and
retains no promotion claim. Phase comparisons close the batch-1 full prologue as the
public kernel's advantage: both pay nearly the same cost. Cluster/DSMEM merge and overlapped
finalize remain unbuilt. The next meaningful HMRG gate is a complete-layer comparison after
porting, not another isolated pipeline attribution. [Rejection register](../ruled-out.md).

### Separate projection context: conforming push paths

The current `ce25f67` candidate uses exit cluster barriers on every push path and local
stores for rt's own-rank share. It no longer relies on the two ambiguous behaviors discussed
in the previous review. This is a correction to the current candidate status, not a claim
that the old behavior was proven sound. [Port-guide decision](../proj-port-guide.md).

`3abb1cd974e2477d` and `d7c662296eaa4413` at `ce25f67` (header-only difference) pass
9/9 and score 1.495 (1.473-1.517) / 1.496 (1.473-1.518). Lt picks agree; router32
switches between tied rt variants. The following are cross-run means of the two conforming
checks versus the two earlier confirmations, not paired isolated barrier costs:

| Router tokens | Earlier mean | Conforming mean | Change | Change vs scored d0dc80f |
|---|---:|---:|---|---|
| 8 | 4.26 us | 4.43 us | +0.17 us (+3.99%) | -1.25 us (-22.01%) |
| 16 | 4.54 us | 4.80 us | +0.26 us (+5.73%) | -0.44 us (-8.40%) |
| 32 | 5.48 us | 5.49 us | +0.01 us (+0.18%) | +0.04 us (+0.73%) |
| 128 | 6.61 us | 7.19 us | +0.58 us (+8.77%) | -0.30 us (-4.01%) |
| 256 | 9.60 us | 9.81 us | +0.21 us (+2.19%) | 0.00 us (0.00%) |

Percentages use the rounded logged means; d0dc80f controls are reconstructed from the logged
deltas (5.68/5.24/5.45/7.49/9.81 us). Two runs per build, no interval for these per-shape
changes. The previously measured non-conforming KLUE/rt wins remain historical; current
cross-run data shows no retained gain at 32/256. Do not transfer the old paired percentage
to the corrected candidate. [Run log](../runs.md#conforming-push-paths-coordinator-decision-2026-10-30),
[receipt 1](../../muse_bf16_proj/profiles/submission_r2t1.3abb1cd974e2477d.json),
[receipt 2](../../muse_bf16_proj/profiles/submission_r2t2.d7c662296eaa4413.json).

Coordinator submission `d59bc7af342a465d` is logged at 1.4879x, source `ce25f67`, versus
previous projection board entry `1d277e7fe9c44712` at 1.4818x. Receipt/interval/correctness
count for this submission are absent locally; do not interpret the score difference as a
controlled gain. Projection remains unported to the layer. P3 remains unbuilt.

Next evidence: HMRG layer port and controlled layer test, receipts for the two new scored
runs, and a paired conforming projection comparison if isolated retained gains matter.
Standalone and layer promotion are separate decisions; do not add these component scores.


The [tracker](../architecture-performance-tracker.md) preserves all historical results.

Validation: independent source review checked the six receipts, timings, percentages and
local links. Dashboard filters, 22 comparison rows and four pending cards pass executable
checks. The generated tracker fingerprint matches. Browser screenshots of the header and
pending component cards showed readable text and no layout collisions at the observed width.
