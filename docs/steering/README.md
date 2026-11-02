# Shared project steering

This is the stable cross-session entry point. Read the latest human priority, its evidence snapshot and subsequent commits before continuing. Existing captain/worker ownership, correctness requirements and submission permissions remain in effect. A report is not an additional GPU dispatch.

## Latest assignment: Decode32 Worker — full-context attention

**[D32-ATTN-01: implement one isolated full-decode attention candidate](decode32/2026-11-02-01-full-decode-attention.md)** is now posted for the user-created **Decode32 Worker**. The user clarified that it was waiting for tasks, not already implementing one. Start source/code work on T32/q_len1 full-context attention, independently of the existing MID MoE comparison. Deliver a reproducible full-layer candidate plus focused tests and a Driver measurement handoff, or a concrete blocker. Preferred cross-tile-overlap hypothesis must first be reconciled with actual selected code and prior scoped failures; no gain is assumed.

**Ownership split:** Decode32 Worker owns the isolated attention track; OKC Developer retains MID-at-T32 MoE; OK Driver remains sole judge dispatcher/integration owner. No duplicate implementation or new public/default permission. The assignment is published in the watched folder; actual pickup is still to be recorded in `decode32/2026-11-02-01-full-decode-attention-status.md`. A commit is not an acknowledgment.

## Decode32 task inbox

[Add research assignments and direction here](decode32/README.md). The "Review decode32 research tasks" chat / Decode32 Worker reads this folder during work and through its recorded periodic monitor. Its former waiting state is superseded by the assignment above, not by an assumed pickup. Existing captain and MoE developer ownership remains. [Setup and monitoring contract](2026-11-02-decode32-inbox.md).

## Existing captain and MoE owner assignment

**[November 2: Explicit decode32 owner directive and pickup ledger](2026-11-02-decode32-owner-directive.md)** — the user reiterated "make it priority for them please". D32 is the primary next performance assignment, ahead of unrelated short-load or large-verify-only work. The directive is also at the top of both root instruction files, `AGENTS.md` and `CLAUDE.md`.

**Captain pickup is now confirmed:** concurrent commit `5b4a729a` explicitly adopted D32, deferred the unstarted broad dense-backing proposal, and allocated the fixed MID-at-T32 comparison below. Actual delivery of that new task and developer acknowledgment were not yet established in the read report; record them separately. This documentation update preserves that live allocation and creates no duplicate task, dispatcher, GPU run or public permission.

## Latest human priority: concentrate on decode batch 32

**[November 2: Decode32 priority and weighted improvement plan](2026-11-02-decode32-priority.md)** — evidence `b325b965`. The user explicitly selects the `22.149 ms / 11.888 ms / 1.8632x` decode-batch32 row as the next objective. It is T32/q_len1, not verify32/T256. The row is `46*sliding + 16*full`, approximately 60% sliding and 40% full. First milestone: 2x, requiring 0.8135 ms saved from this load; 2.2x requires 1.8203 ms. These would contribute about 1.19%/2.81% to the six-load layer score if everything else stayed fixed, not guaranteed overall rank movements.

Prioritize expert/B4 work in both contexts, then batch32 full-decode attention. Use the existing accepted direct-weight result before another implementation. Preserve other load winners and test a prospective shape-specific candidate if warranted; do not rewrite the original test's outcome. PR-only T128/256 and T1/T8 work are not the next primary D32 assignments. The proposed 8-us expert saving in both contexts plus 20-us full-attention saving is a target budget, not measured performance.

## Current captain reconciliation after the new priority

**[D32 result and fixed MID-family comparison](2026-11-02-driver-d32-mid-comparison.md)**: direct-weight own0b684a40 is terminal VALID but D32 sliding180.848us/full337.856us, weighted13.724704ms versus own baseline22.145344ms (1.61354x). Exact direct lane and held pair stopped; no D32 rescue signal. Overall1.926423591x does not pass its original gate. Current prior-control comparison remains descriptive across inputs/nodes/drivers.

Captain retains one current-service unchanged607 calibration, then one cheap fixed17..32 MID-versus-B4 dispatcher implementation/ordinary private measurement, existing developer only. M33/other five loads stay control. Broad dense-bank GENERIC work proposed concurrently was **not delivered or started** and is now an unallocated secondary reserve under the newer D32 instruction. [Full direct terminal and priority addendum](2026-11-02-driver-direct-weight-terminal-and-dense-backing.md). Fresh23:33UTC:overall/layer376,7private layer/no-live; no new public authority. Accepted control ID and direct assignment acknowledgment will be added when established.

## Current execution handoff at the evidence snapshot

**[Direct-weight private check accepted: 0b684a4065354fc5](2026-11-02-driver-direct-weight-first-check.md).** OK Driver remains captain/sole dispatcher, with the existing OKC Developer owning implementation. The accepted 875672-byte/a27a1baf file changes weight delivery in B4/MAIN and is relevant to T32. Target outcome remains pending in this reviewed source snapshot; reconcile its owning terminal before acting. No duplicate POST/poll, no mutation of frozen source, no second check or public/default authority is created here. Its conditional matched-instrument preparation does not establish a performance winner.

**[K128/shared-A terminal history and direct-weight allocation](2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md).** Preserve the completed losing implementations and their scoped stops. The latest human D32 focus is a prioritization change for subsequent work, not evidence that those old variants won.

## Full prior index and report history

**[Complete preceding steering index, preserved byte-for-byte](index-history-through-2026-11-02-2325.md)** contains the prior live handoffs, allocation history, source studies, all older latest pointers and report-history links. Its blob is exactly `3d09977417e4d5fe0c18f99ac82f91a2b2454ff6`, the README at `b325b965`. It remains in this directory so relative links keep their meaning. No preceding index content or report was discarded.

| Report | Evidence | Scope |
|---|---|---|
| [D32-ATTN-01 implementation task](decode32/2026-11-02-01-full-decode-attention.md) | a4192dec inbox; user-created worker; existing D32 plan and continuity source | Separate Decode32 Worker attention track, candidate/tests handoff, sole Driver dispatch; pickup pending |
| [Decode32 inbox setup](2026-11-02-decode32-inbox.md) | bdbf35bf; user request | Shared task folder, waiting chat, periodic monitoring and preserved execution ownership |
| [Decode32 owner directive](2026-11-02-decode32-owner-directive.md) | b9ca9dff and captain5b4a729a; explicit user follow-up | P0 in root instructions; captain pickup confirmed; new developer delivery/acknowledgment not yet verified |
| [Captain D32 MID comparison](2026-11-02-driver-d32-mid-comparison.md) | b9ca9dff human priority; own0b terminal; source auditba9/b8 | DirectD32 negative closure, current-service control and fixed existing MID17..32 exploration; no new shader/math/packing/sweep |
| [Direct terminal and superseded broad proposal](2026-11-02-driver-direct-weight-terminal-and-dense-backing.md) | own0b684a40;368b85c9; fullraw/profiles;23:33UTC | ExactPARK; broad P1 superseded before delivery; P0 calibration retained |
| [Decode32 priority](2026-11-02-decode32-priority.md) | b325b965; public607; fresh cc474 profile; user row | D32-first target, weighted budgets, B4/full-decode attention order, running direct-weight reconciliation, shape-specific confirmation and all-load regression protection |
| [Previous full index](index-history-through-2026-11-02-2325.md) | exact former README blob3d099774 | All earlier execution, research and report history |

## Publishing future steering

Check substantive steering into `docs/steering/YYYY-MM-DD-<topic>.md` or the user-designated `docs/steering/decode32/` inbox for D32 tasks, update this latest pointer/history, and commit/push/verify in the same task. Pin evidence and owning jobs; distinguish measured effects, hypotheses and unknowns. Preserve concurrent work and earlier reports; never force-push. Documentation is not a new public-submission permission. The captain retains live execution/index coordination; the new worker was created by the user, not by this research session.
