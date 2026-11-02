# Decode32 task inbox

Use this folder to give the "Review decode32 research tasks" chat assignments and
direction from "okc research". The chat is waiting for an assignment. Creating this
inbox does not take over the existing MID-at-T32 implementation.

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
and verification scope.
