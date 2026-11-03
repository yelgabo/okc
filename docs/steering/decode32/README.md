# Decode32 task inbox

## Current instruction: one owner, end to end, batch32 only

**[D32-02: Decode32 Worker owns batch-32 optimization end to end](2026-11-02-02-end-to-end-ownership.md)** is the latest direct user instruction. Decode32 Worker owns D32 research/selection, implementation, compilation, correctness tests, private GPU checks, result analysis, iteration, integration and commits. **No routine Driver/Developer handoffs or per-step captain approval.** This does not transfer the rest of the project or unrelated benchmarks.

Scope: decode batch32 / T32 / q_len1, both contexts; `C32 = 46*sliding_us + 16*full_us`. Prioritize actual improvements to this load and preserve the other five loads. Existing correctness and public-submission permissions remain.

## Current work

Continue **[D32-ATTN-01](2026-11-02-01-full-decode-attention.md)** from the [acknowledged 23:55 UTC pickup](2026-11-02-01-full-decode-attention-status.md), but take it through measurement and integration yourself. Its historical source-only/no-check/Driver-handoff wording is superseded by D32-02. A candidate/tests handoff is no longer completion.

Reuse existing [MID-at-T32 work and calibration](../2026-11-02-driver-d32-control-terminal-and-mid-gate.md). Concurrent publication confirmed [MID check a6fe2769779e4299 was already accepted](../2026-11-02-driver-d32-mid-first-check.md): its original owner finishes that exact frozen job/archive, without cancellation, duplicate submission or a second poller. After inspecting the latest shared ledger and quota, Decode32 Worker owns subsequent D32 submissions. This one-time existing-job exception is not a recurring handoff gate. OK Driver/OKC Developer retain unrelated tasks and must not open new overlapping D32 assignments.

Record acknowledgment of D32-02 and the current candidate/job in a short status update here. Do not wait for another research task to begin the authorized end-to-end cycle. This publication is not itself evidence that the expanded role has been read or executed.

## Useful concurrent source work, preserved

[Driver's L1 score-carry/source/lifetime notes](2026-11-02-02-driver-attention-source-feedback.md), added in7afce310, distinguish existing load overlap from next-tile score carry and record lifetime/softmax obligations and prior scoped receipts. Keep this technical feedback; only its old root-only dispatch/handoff restriction is superseded. It is not a second implementation task.

[Existing MID confirmation-tool reuse and blockers](../2026-11-02-worker-mid-matched-reuse.md), published65b94dcc, records the completed source investigation. It is not a ready paired payload. Reuse valid pieces rather than rebuild a framework: arm-local16/32policy, route/source witnesses, independent controls and prospectiveD32/all-five-regression gates still matter. Decode32 Worker owns the subsequent work under D32-02, without another Driver allocation requirement. Concurrent source reports and commits are preserved unchanged.

## Inbox convention and preserved evidence

Add dated Markdown tasks, status and result records here. Preserve earlier tasks/results; newer explicit user instructions supersede conflicting historical ownership. Record source and job identities, what actually ran, its result and the next decision. No general framework or repeated ownership review is needed.

Background:
- [D32 weighted targets](../2026-11-02-decode32-priority.md).
- [Earlier owner directive](../2026-11-02-decode32-owner-directive.md), historical ownership split superseded.
- [MID comparison design](../2026-11-02-driver-d32-mid-comparison.md).
- [Inbox setup and monitoring contract](../2026-11-02-decode32-inbox.md).

The recorded monitor checks this folder before work/between bounded work units and periodically through the desktop scheduler, including committed `origin/main` updates and the two documented local checkouts. That is periodic checking, not proof of an instantaneous notification or pickup. Preserve local work when fetching and push completed code/results under the normal repository workflow.
