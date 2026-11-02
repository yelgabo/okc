# Shared project steering

This is the stable cross-session entry point. Read the latest human priority, its evidence snapshot and subsequent commits before continuing. Existing captain/worker ownership, correctness requirements and submission permissions remain in effect. A report is not an additional GPU dispatch.

## Latest human priority: concentrate on decode batch 32

**[November 2: Decode32 priority and weighted improvement plan](2026-11-02-decode32-priority.md)** — evidence `b325b965`. The user explicitly selects the `22.149 ms / 11.888 ms / 1.8632x` decode-batch32 row as the next objective. It is T32/q_len1, not verify32/T256. The row is `46*sliding + 16*full`, approximately 60% sliding and 40% full. First milestone: 2x, requiring 0.8135 ms saved from this load; 2.2x requires 1.8203 ms. These would contribute about 1.19%/2.81% to the six-load layer score if everything else stayed fixed, not guaranteed overall rank movements.

Prioritize expert/B4 work in both contexts, then batch32 full-decode attention. Use the existing accepted direct-weight result before another implementation. Preserve other load winners and test a prospective shape-specific candidate if warranted; do not rewrite the original test's outcome. PR-only T128/256 and T1/T8 work are not the next primary D32 assignments. The proposed 8-us expert saving in both contexts plus 20-us full-attention saving is a target budget, not measured performance.

## Current execution handoff at the evidence snapshot

**[Direct-weight private check accepted: 0b684a4065354fc5](2026-11-02-driver-direct-weight-first-check.md).** OK Driver remains captain/sole dispatcher, with the existing OKC Developer owning implementation. The accepted 875672-byte/a27a1baf file changes weight delivery in B4/MAIN and is relevant to T32. Target outcome remains pending in this reviewed source snapshot; reconcile its owning terminal before acting. No duplicate POST/poll, no mutation of frozen source, no second check or public/default authority is created here. Its conditional matched-instrument preparation does not establish a performance winner.

**[K128/shared-A terminal history and direct-weight allocation](2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md).** Preserve the completed losing implementations and their scoped stops. The latest human D32 focus is a prioritization change for subsequent work, not evidence that those old variants won.

## Full prior index and report history

**[Complete preceding steering index, preserved byte-for-byte](index-history-through-2026-11-02-2325.md)** contains the prior live handoffs, allocation history, source studies, all older latest pointers and report-history links. Its blob is exactly `3d09977417e4d5fe0c18f99ac82f91a2b2454ff6`, the README at `b325b965`. It remains in this directory so relative links keep their meaning. No preceding index content or report was discarded.

| Report | Evidence | Scope |
|---|---|---|
| [Decode32 priority](2026-11-02-decode32-priority.md) | b325b965; public607; fresh cc474 profile; user row | D32-first target, weighted budgets, B4/full-decode attention order, running direct-weight reconciliation, shape-specific confirmation and all-load regression protection |
| [Previous full index](index-history-through-2026-11-02-2325.md) | exact former README blob3d099774 | All earlier execution, research and report history |

## Publishing future steering

Check substantive steering into `docs/steering/YYYY-MM-DD-<topic>.md`, update this latest pointer/history, and commit/push/verify in the same task. Pin evidence and owning jobs; distinguish measured effects, hypotheses and unknowns. Preserve concurrent work and earlier reports; never force-push. Documentation is not a new public-submission permission. The captain retains live execution/index coordination; this user-requested handoff does not create another worker.
