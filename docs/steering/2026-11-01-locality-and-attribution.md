# Locality report intake and H200 attribution decisions

November 1, 05:51 PST. Evidence checkpoint `9904ded` includes the verified
primary-source ledger (`5631967`) and final independent MoE probe clearance.
This report follows the [selected-cell package](2026-11-01-selected-layer-pair.md),
whose H200 execution remains pending. Prefill remains outside the active scope.

## New pasted report: read and reconciled

The complete new upload is *Accelerating MoE Inference with Locality-Aware Kernel
Design: Technical Analysis and Reproduction Guide*. Its complete text is preserved
with normalized LF line endings in [uploaded.txt](evidence/2026-11-01-locality-report/uploaded.txt).
[identity.json](evidence/2026-11-01-locality-report/identity.json) records both the
normalized-text hash and the original 40965-byte attachment hash. Original SHA256:
`e4b4b031f4856a6dcfe0a90d202fd7b66c8baae2f0aa60ad1b1417ef55ca2ca2`.
This is distinct from the earlier *Reaching the Top 10* upload.

The [primary-source ledger](2026-11-01-moe-locality-primary-ledger.md) verifies
the blog, archived kernel, proposed correction, vLLM integration attempts and
later grouped-GEMM work. The archived 2024 kernel has a rectangular-grid coverage
defect; its fix remains unmerged and reports lower corrected performance. The
vLLM integration already used the corrected modulus, so its separate failures
cannot be assigned to that bug. Historical headline speedups and the unlinked
Twitter rank170→80 story are not measurements of our implementation. Four CPU
coverage tests pass. No foreign standalone kernel was adopted.

The transferable hypothesis is narrower: shorten the reuse distance of the same
expert's packed weight tile when that expert owns multiple token-row chunks.
Exact G already places an expert's jobs together, but processes
`job → output unit → K stages`. A bounded transpose of job and output unit could
bring repeated weight tiles closer together. It must consistently transform the
weight producer, activation producer, consumer and inverse fixup-owner mapping;
the ragged final job changes cost prefixes. A one-line index substitution is
unsafe. This is a hypothesis, not a measured gain.

The prior [G request census](2026-11-01-g-highload-weight-census.md) remains valid:
T32 cannot duplicate 64-row jobs, and the local routing sample's proportional
whole-layer scenario is about0.63%, not a physical bound or prediction. The new
article alone does not reopen a weight-layout or Split-K sweep. Independently
inspect the owned standalone's smaller job chunks for a larger, distinct
opportunity, while preserving the layer-only scope of the attributed fork.

## Completed attention measurement: stop this probe

H200 check **3f44f891d0ef47db** passed all16 official tests and all scoped
correctness checks, with exact runtime resource parity. However, all12 timed
rows exceed the preregistered instrumentation limit. Detail perturbation is
3.776–23.136µs; permitted values are0.990–1.959µs. Zero spills and matching
occupancy did not establish timing transparency.
[Terminal table and raw evidence](2026-11-01-attention-full-wait-trace.md#november-1-0548-pst-terminal-result).

Close this exact trace as causal evidence. Preserve the records and the common
BF16 control score2.490692705× (95% CI2.453332314–2.528053095), without calling
that diagnostic score an optimization. No production attention change, bottleneck
claim, automatic stamp sweep or public submission follows.

## MoE measurement running

Owned private job **f147d06a64504c1e** was dispatched for `muse_moe_w4a16` after
independent host/device review. Exact artifact392222bytes, SHA256
`bf0ca4ae44281d30efd80b988566dd8c431f3095a1d38e4310029c39667f8170`.
It measures the incumbent G MAIN backend at128/256 tokens on fixed independent
natural input banks, with duplicate controls, endpoint controls and separate
observed-readiness/retirement probes. B4FIN32 remains outside this device scope.
[Review and qualifications](2026-11-01-g-stage-probe-independent-review.md).

Five actual TorchCPU host tests pass, including negative finite-reference,
deadline and timing-distortion gates. Native H200 host compilation, preparation
budget, correctness, real resources and measurement distortion remain open.
READY has larger local-compiler spill counts than control; passing CPU tests
cannot qualify it. Readiness markers report observed sequential waits, not
physical copy completion or first/last MMA attribution. No stage intervals may
be added into an invented whole-kernel savings estimate.

## Priority and stopping rules

1. Finish and archive the owned MoE job, including losing or unusable rows.
   Require duplicate spread≤1.5%, endpoint overhead≤1.5%, stamped whole-kernel
   overhead≤3% and phase-span overhead≤3%, per load/input, with the registered
   resource/admission checks. Only qualified rows can steer a code change.
2. Use exposed phase evidence plus actual repeated expert chunks to decide one
   bounded work-order hypothesis. Prove full tile coverage and inverse ownership
   before compiling. Reject an equivalent prior loser without new evidence.
3. Preserve the16:00 PST layer reset plan: repaired six-load incumbent-versus-stage
   paired check, then the independent short-path fixed-routing trace. The smaller
   selected cross-family package is available for a subsequent scoped check;
   its source size result does not replace H200 validation.

The best public layer candidate remains `g1b4_stage_release.py`, owned job
**607eb52bdaf24160**,2.126082676×. The last measured board snapshot at04:48 PST
placed it305th, with a12.60% ratio gap to tenth. No newly measured portfolio
closes that gap. There is no basis to promise top10 before the deadline; continue
only through concrete opportunity and whole-score gates, not historical blog
speedups. This report introduces no new public attempt.

## November 1, 05:55 PST terminal decisions

MoE **f147d06a64504c1e** completed: PASS5, all six benchmark cells and profiles,
policy checks clean, uninstrumented diagnostic control **2.340277798×**, 95% CI
[2.305173631,2.375381965]. All216 registered numerical cases passed, plus the
separate decoded snapshots. Maximum global/row errors were0.0060922/0.0071880;
preparation finished in171.325seconds. This private control contains the
attributed layer backend and is not a standalone submission candidate.

All12 READY/RETIRE mode-by-input rows fail the frozen attribution gate. Whole
kernel displacement is4.91–9.54%; phase displacement10.26–14.90%. The endpoint
whole kernels are close to control, but endpoint phase duplicate spread reaches
5.33%. Close the exact probe without a rerun or a causal bottleneck claim.
Actual resource attributes show96registers,231168dynamic shared bytes,
576threads and a one-CTA/SM admission limit. RETIRE reports8local bytes;
control/endpoint/READY report0. These are resource attributes, not achieved
occupancy, bandwidth or measured utilization. The complete result is recorded
in the [native G probe report](2026-11-01-g-stage-observed-readiness.md).

The newly retained natural H200 route counts add useful population evidence:
all three128-token banks have one extra64-row job, and one of three256-token
banks has two extra jobs. Applying the prior explicitly hypothetical proportional
model to the current607e expert/layer envelopes gives per-seed six-load factors
1.012567/1.004403/1.004607, geometric mean about1.00719. This is not measured
HBM traffic, an upper bound or a projected gain. Only one pair exceeds1%, and
the probe supplies no qualified exposed-stall evidence. The bounded transpose
therefore remains deferred. [Mapping and inverse proof obligations](2026-11-01-g-expert-run-order-proof.md).

The [owned standalone audit](2026-11-01-owned-moe-locality-scope.md) separately
confirms the8-row transpose is distinct from earlier experiments. It also finds
zero repeated payloads in all four retained natural standalone16/32 count cases;
M1/8 cannot repeat an8-row chunk. Positive layer-routing samples cannot stand in
for standalone results. The large owned path has a different64-row scheduler.
There is no justified standalone transpose implementation or quota request yet.

### Refreshed actual target gaps

The05:51 PST [four-board snapshot](evidence/2026-11-01-0551-boards/identity.json)
uses official CLI results. Ratios below use its rounded display speedups, so gap
percentages are approximate; they are required score increases, not predicted
latency reductions. Overall remains represented by the layer board.

| Board | Our rank | Our speedup | Rank50 | Rank10 | Gap to50 /10 |
|---|---:|---:|---:|---:|---:|
| Layer |308|2.1261×|2.2894×|2.3983×|7.68% /12.80%|
| Attention |219|2.6043×|2.7119×|2.9331×|4.13% /12.63%|
| MoE |387|2.2157×|2.4547×|2.6077×|10.79% /17.69%|
| Projection |310|1.5278×|1.5848×|1.6087×|3.73% /5.30%|

### Ranked next actions and acceptance

| Priority | Action and evidence | Acceptance / stop |
|---|---|---|
|1|At16:00 PST, run the already repaired six-load incumbent-versus-stage comparison. Existing public607e improved the earlier owned public score, but matched six-load repeatability remains open.|All official math/order/resource gates; independent controls and routing; registered whole-score/null/regression gate. Keep607e if the comparison fails.|
|2|Run the prepared independent short-path fixed-natural-routing trace, rather than repeating the losing concurrent producer.|Require all recorded math/cache/ownership and low-distortion gates. A useful exposed interval must support the whole-score budget before another fusion.|
|3|Consider the850769-byte selected verify32/sliding cross-family package after the first reset checks.|Real H200 host linking, setup<300s, all24 private oracle checks, independent state and identical natural expert IDs. Report one-cell timing only; require broader evidence before promotion.|
|Deferred|Same-expert bounded unit transpose, for G or owned standalone.|Reopen with actual matching-route multiplicities plus physical traffic or qualified exposed stalls and repeatable score-covered opportunity. New article/census alone is insufficient.|

No speculative candidate should consume the remaining two layer public slots.
Remaining free checks after these jobs: layer0, projection0, attention4, MoE2.
The reset wakeup remains registered. Best submission remains byte-pinned
**g1b4_stage_release.py /607eb52bdaf24160**, SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
Both measurement failures are closed at their tested scope. No current evidence
supports a promised top10 outcome, and no new public submission was made.
