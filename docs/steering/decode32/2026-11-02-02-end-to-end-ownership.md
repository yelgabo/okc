# D32-02: Decode32 Worker owns batch-32 optimization end to end

November 2, 2026. Evidence checkpoint `28346bf563789879bcec28e52edeb05753a6b3f0`.
Latest user instruction: "have the decode32 worker do everything, no handoffs. its too slow if it keeps handing off stuff." The user immediately clarified: "i mean in terms of optimizsing batch-32 stuff" and "not EVERYTHING lol".

## Effective assignment — batch32 only, not project captain

**Decode32 Worker owns the complete decode-batch32 optimization cycle: choose the next D32 experiment, implement, compile, validate, run private GPU checks, read the results, iterate, integrate validated changes, and commit/report them. Do not stop at a candidate handoff or wait for OK Driver/OKC Developer to perform those steps.** This is a direct user ownership change, not a research proposal needing captain adoption.

Scope is decode B32 / q_len=1 / T32, both sliding and full contexts, and any attention, MoE, projection or boundary work that actually improves that load. The objective remains `C32 = 46*sliding_us + 16*full_us`, with the recorded 11.888 ms / 1.8632x reference and a first reproducible 2x milestone. Preserve the other five load paths and validate the complete layer; general metadata-based neighboring-shape support is still required. Verify32/T256 is not this objective.

This does **not** make Decode32 Worker captain of the repository, owner of unrelated benchmarks, manager of other sessions, or responsible for every project task. OK Driver and OKC Developer keep unrelated work. No new agents or delegation chain are requested.

## Authority and supersession

For this D32 lane only, this instruction supersedes D32-ATTN-01's source-only/measurement-handoff restriction and all older "OK Driver sole dispatcher/integration owner" or "await captain allocation" wording. Decode32 Worker may directly perform the needed ordinary private `okc check` runs and follow its own job IDs after checking current shared quota and live jobs. No per-experiment Driver approval or separate worker handoff is needed. Existing credentials/tools must be used; report a concrete access blocker if unavailable rather than call another ownership review.

Correctness, numerical boundaries, memory ordering/progress, graph lifetime, source attribution and regression checks remain required. Removing process handoffs does not remove technical validation. Reuse existing tools; do not build another general framework merely to change ownership. Existing user permissions for public/scored submissions remain unchanged: this is not blanket new public-posting authorization. Where an existing approval applies, the D32 owner executes it directly instead of requiring a Driver relay; otherwise obtain only the missing user approval.

## Continue now without duplicating the existing work

The worker already acknowledged the attention task at 23:55 UTC in [D32-ATTN-01 status](2026-11-02-01-full-decode-attention-status.md), published at `69cf82b3`. Continue that implementation from its current state; its endpoint is now a measured and validated candidate, or a specific failed experiment/blocker, not a handoff packet. The preferred mechanism remains a hypothesis, not an obligation to implement a known loser.

Reuse the existing MID-at-T32 implementation and current-service control rather than rebuilding them. At the evidence checkpoint the control `94ef698e91964dfa` is complete and the fixed MID candidate is source-ready; no MID target ID is established by this report. Read the newest [MID gate](../2026-11-02-driver-d32-control-terminal-and-mid-gate.md) and job ledger once before submitting. If Driver has already accepted a MID job, its current owner finishes that exact frozen job and archives it; do not cancel, mutate, resubmit or create a second poller. That one-time retirement of pre-existing work is not a recurring handoff requirement. If not accepted, the next D32 submission belongs to Decode32 Worker.

OK Driver/OKC Developer: stop opening new overlapping D32 assignments or launching new D32 jobs under the old ownership split. Preserve/publish already-created work and finish any already-accepted own-ID job; continue unrelated work normally. Shared quota coordination is bookkeeping, not an approval chain. Do not overwrite active worktrees or change another live source pin.

## Working loop and reporting

Maintain one compact D32 status/job ledger in this watched folder. Record the candidate/source pin and accepted job ID before starting another potentially overlapping measurement. Lead results with sliding/full raw and adjusted times, the weighted C32 total, own baseline, selected path and all-load regression summary. An ordinary first check supplies execution/correctness and a direction; use matched held-outs and independent controls before claiming a reproducible causal gain. Keep smaller cheap wins when justified; do not require every D32 contribution to win at T128/256 or alone add 1% to the whole-layer score.

Integrate confirmed D32 improvements yourself with rollback/source identity and full correctness coverage. Preserve raw losing receipts and their scoped stop reasons. Continue the next justified D32 task without routing routine implementation, testing, analysis or integration through another session. Escalate only a genuine technical/access/permission blocker, not a routine review checkpoint.

## Publication status

This report changes ownership instructions only; it launches no GPU job and changes no kernel. Prior attention pickup is verified; pickup of this expanded end-to-end role must be recorded by the worker and is not implied by publication. This session has not directly interrupted another running chat. The watched folder and root instructions carry the new directive.
