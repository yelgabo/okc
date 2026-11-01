# Locality report intake and H200 attribution decisions

November 1, 05:54 PST. Evidence checkpoint `9904ded` includes the verified
primary-source ledger (`5631967`) and final independent MoE probe clearance.
This report follows the [selected-cell package](2026-11-01-selected-layer-pair.md),
whose H200 execution remains pending. Prefill remains outside the active scope.

## New pasted report: read and reconciled

The complete new upload is *Accelerating MoE Inference with Locality-Aware Kernel
Design: Technical Analysis and Reproduction Guide*. Its exact original bytes and
identity are preserved in [uploaded.txt](evidence/2026-11-01-locality-report/uploaded.txt)
and [identity.json](evidence/2026-11-01-locality-report/identity.json). SHA256:
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
