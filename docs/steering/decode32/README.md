# Decode32 task inbox

Supporting feedback for the active owner: [Driver's bounded L1 score-carry/source/lifetime notes](2026-11-02-02-driver-attention-source-feedback.md). This is not a second assignment; D32-ATTN-01 and its existing owner remain active.

## Current actionable assignment

**[D32-ATTN-01: implement one D32 full-context attention candidate](2026-11-02-01-full-decode-attention.md)** is assigned to **Decode32 Worker** (the "Review decode32 research tasks" chat). Begin source/code work and record actual pickup in `2026-11-02-01-full-decode-attention-status.md`. Publication does not establish that the worker has read or begun it. No additional research assignment is required to start this bounded implementation.

**Ownership:** Decode32 Worker owns this isolated attention track; OKC Developer keeps the existing MID-at-T32 MoE comparison; OK Driver remains sole judge dispatcher/integration owner. Preserve other workers' files and do not duplicate their jobs. The deliverable is one reproducible candidate plus tests and a measurement handoff, or a concrete blocker—not another general review.

## Inbox convention

Use this folder to give Decode32 Worker assignments and direction from "okc research". Creating this inbox did not take over the existing MID-at-T32 implementation; the current attention assignment is separate.

Read these owning briefs before assigning or starting work:

- [D32 priority and weighted targets](../2026-11-02-decode32-priority.md).
- [Owner directive and acknowledgment ledger](../2026-11-02-decode32-owner-directive.md).
- [Captain's fixed MID comparison](../2026-11-02-driver-d32-mid-comparison.md).

Add dated Markdown files here. State the objective, owner, files they may change,
constraints, required evidence and stop condition. A short instruction is enough
when the existing brief already supplies those details. Preserve previous tasks
and add a dated update when direction changes. Record pickup separately from
completion and link the resulting code, checks or report.

The workload is T32, q_len=1. Its weighted time in microseconds is
`46 * sliding_us + 16 * full_us`. OK Driver remains sole GPU dispatcher. The
existing OKC Developer owns the allocated MID comparison. Reconcile new work
with that allocation before editing overlapping files or requesting measurement.
Existing correctness and public-submission permissions still apply.

The waiting chat checks this folder before work and between bounded work units.
A five-minute heartbeat also checks committed changes on `origin/main` and local
files in both the chat checkout and `/Users/gretchengabo/GitHub/okc`. The heartbeat
uses a private file manifest to distinguish new updates from previously read
ones. Local edits remain local; they are not automatically committed or pushed.
The scheduler checks while the desktop app can run local automations, so this is
periodic checking rather than an instant filesystem notification.

See the [setup handoff](../2026-11-02-decode32-inbox.md) for the checkout, baseline
and verification scope. The historical waiting state in that setup is superseded
by the assignment above; actual pickup still requires the worker's acknowledgment.
