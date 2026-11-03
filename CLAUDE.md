# okc: Open Kernel Challenge work

## P0 user directive: Decode32 Worker owns the D32 lane end to end (2026-11-02)

The user explicitly removed handoffs for **batch-32 optimization only**, not the
whole project. **Decode32 Worker owns D32 implementation, compilation, validation,
private GPU checks, result analysis, iteration, integration and commits.** It does
not need OK Driver/OKC Developer to approve or execute each step. Read the latest
[scoped owner directive](docs/steering/decode32/2026-11-02-02-end-to-end-ownership.md)
and [watched task inbox](docs/steering/decode32/README.md).

Scope is decode T32/q_len1, sliding and full, optimizing
`46 * sliding_us + 16 * full_us`, not verify32/T256. Preserve other load winners.
First milestone is reproducible 2x from the recorded1.8632x, then2.1-2.2x;
these are targets, not promised gains. Continue the already-picked-up attention
work and reuse the existing MID implementation rather than rebuilding it.

This supersedes older Driver-only dispatch, attention-only/source-only and
measurement-handoff restrictions **for D32 only**. OK Driver/OKC Developer keep
unrelated work; do not start new overlapping D32 assignments. Existing accepted
jobs finish under their original owner without cancellation, duplicate submission
or a second poller. Decode32 Worker checks the live shared job/quota ledger before
new private runs and records its exact source/job. A one-time existing-job overlap
is not a recurring handoff gate. No new agents or project-wide takeover.

Technical correctness, source attribution, graph lifetime and regression gates
remain. Existing public/scored-submission permissions are unchanged; no blanket
new public authorization is implied. The same D32 owner performs approved release
work directly. Record actual pickup separately from publication; escalate only
real access/technical/permission blockers, not routine implementation checkpoints.

Background and decisions: `docs/context.md`. Coordinator working method (agents, quota,
submits, porting, traps): `.claude/agents/okc-coordinator.md`. Every run: `docs/runs.md` (bf16_proj),
`docs/runs-moe.md`, `docs/runs-attention.md`.

## Steering reports and cross-session handoffs

User preference, 2026-10-29: every substantive project-steering or audit response must be
checked into this repository so the other sessions can read it. Start steering work by
reading `docs/steering/README.md` and its latest report, then reconcile with newer commits
and run logs rather than repeating completed experiments.

Write the full report to `docs/steering/YYYY-MM-DD-<topic>.md` (add a time or sequence suffix
when needed), pin its evidence commit and run identifiers, distinguish measured results from
hypotheses, and include priorities, next actions, validation/stopping criteria and gaps.
Update `docs/steering/README.md` with the latest link and report history. Preserve old reports;
use explicit dated addenda or a new report for corrections and subsequent progress.

Commit and push the report and index during the same task using the normal repository
workflow. Preserve concurrent work; do not force-push or overwrite another session's changes.
Verify the remote files before reporting completion, and return the report path and commit.
A chat-only answer or sandbox attachment alone does not complete the handoff. If publishing
is blocked, state the blocker rather than claiming it is checked in. An explicit user request
for no repository writes takes precedence. This rule does not authorize kernel changes or
scored submissions beyond the user's existing permissions.

## CUDA research reference

Before selecting CUDA work from the November 2 H200 playbook, read
[the reference index](docs/research/cuda-h200/README.md), then the relevant topic
and application-ledger rows. Run `python3 tools/check_cuda_research.py` before
reusing its source claims. Reconcile changed evidence before refreshing hashes.
The complete attachment and section coverage are preserved there. These notes
do not reopen parked experiments or change D32 ownership or submission permission.
Compiler checks must preserve the compiler's failure status and full diagnostics;
successful output filtering is not build evidence.
When changing WGMMA grouping, read the active constants from the selected native
literal. Nearby helper literals and old comments can differ. Test that deferred
flush conditions remain reachable for every selected group size, and inspect
compiled publication operations before any GPU run. Record rounded flush timing
explicitly rather than claiming an unchanged boundary.

## Research and execution principles

Apply these principles when choosing work, reviewing results and handing off. Use the latest steering report for current priorities and reuse current dated evidence instead of repeating completed investigations.

1. **Own discovery.** Search for stronger methods independently of user-supplied names. Search challenge/problem identifiers, inspect later result pages, and follow leaderboard handles, forks, upstream credits and primary research. Record queries, date, scope, access limits and useful pins. A search miss establishes only that a search found no matches. Distinguish independent discoveries from user-supplied leads.
2. **Choose by the user's objective.** Compare plausible end-to-end benefit, implementation cost, uncertainty and quota with the strongest alternative before another experiment round. Reconsider the roadmap when evidence changes. A backlog entry or interesting local improvement does not establish priority. State what observation would change the choice.
3. **Let evidence choose the starting point.** Our implementation must earn its place as the base. Compare a complete alternative before deciding to graft isolated ideas onto ours; choose a different base or a measured hybrid when evidence and the user's adoption policy support it. Identify necessary computation, bytes moved, producer/consumer dependencies and actual selected routes. Derive a materially different design before renewing local tuning and size its contribution to the real weighted score. Profiler totals, peak hardware rates and logical duplicate bytes alone do not establish exposed time that can be saved.
4. **Compare complete implementations and choose per workload.** Pin and reproduce the full file, including preparation, packing, dispatch, scheduling and epilogues, before selecting individual ideas. A route may win on one workload and lose on another. Compare the actual choices per shape, preserve standalone and layer decisions, then measure the complete combined dispatcher rather than adding isolated gains. Shared ancestry does not establish performance parity. Separate a team's best leaderboard row, a published file's own result and a fresh matched comparison. Read active build flags and selected code paths. Record attribution, precision and ordering obligations before adoption.
5. **Preserve performance during integration.** Begin with the smallest justified changes to a working reference. Inventory each change, distinguish required properties from conservative implementation choices, and measure the cost of changed groups and relevant interactions. If our port loses while the complete comparator wins, record the losing variant and keep the discrepancy open with receipts, affected loads, competing explanations, an owner and a distinguishing test. Investigate it or document why another task has greater expected impact.
6. **Make experiments decide a question.** Before dispatch, write the hypothesis, controls, affected workloads, acceptance threshold and stopping condition. Hold unrelated choices fixed, use matched inputs and independently prepared duplicate controls, and inspect actual compiled code and resources where relevant. Separate measured effects from causal explanations, estimates and hypotheses. Report inconclusive results explicitly.
7. **Validate the measuring instrument.** Verify selected paths, source/job identity, private mutable state, fresh inputs, graph replay and observer overhead. Inspect captured outputs after replay. Compiler success, numerical agreement, ordering/progress proofs and measured speed answer different questions. Test the harness's real entry points and negative cases when they could change the decision.
8. **Preserve correctness while questioning its implementation.** Keep required numerical boundaries, publication, lifetime and progress guarantees explicit. Seek faster implementations of those properties. Passing tests alone cannot establish memory ordering, and a conservative repair's full cost cannot be declared necessary without evidence. Keep private diagnostic and public release eligibility distinct.
9. **Scope failures to exact implementations.** A failed version of a technique does not disprove the whole design family or a different working public implementation. Record the conditions and mechanism each loss rules out, reuse relevant public negative evidence, and state what new evidence would justify reopening. Read those scoped closures before proposing another variant. Develop independent relevant hypotheses and source proofs in parallel. Track ideas, implemented treatments, controls, qualified results and shippable candidates separately; source snapshots, audits and factorial combinations do not create independent candidates. Parallel work must respect ownership and current user priorities.
10. **Turn learning into execution.** While a judge job runs or service is unavailable, advance relevant analysis, implementation, compilation or correctness work. Use authorized scored measurements for eligible candidates when private checks are exhausted; recheck live jobs/quota and keep one dispatch owner. Existing submission permission and release rules still apply. At handoff, record the evidence that changed the next decision, actual outcome, remaining unknowns, owner and next gate in the repository. A written lesson is validated by better subsequent decisions and verified outcomes.

Use the coordinator's Research method for detailed execution. These principles preserve the existing external-code, correctness and submission requirements. Background and rationale are in docs/steering/2026-11-01-working-principles.md.

## Evidence rules

- **Profile SGLang before optimizing a problem.** A normal run profiles only our submission.
  Copy `baseline.py` to `<problem>/baseline_profile/baseline.py` with the
  `# okc: problem=<problem>` header, run it as a check, and keep its result JSON there. That is
  the hotspot map: which SGLang kernels take the time, per shape.
  Discover its receipt with `rg --files <problem>/baseline_profile`; baseline job IDs
  belong to their problem and must not be copied from another problem's path.
- **Find the hotspot, then attack it.** Before writing a kernel, run `tools/hotspots.py` on the
  latest run (with `--baseline`) and pick the target from the gap between each kernel's time and
  the floor. Append the table to `docs/hotspots.md` whenever a new best lands.
- **Keep every run's breadcrumbs.** Commit `profiles/*` for every check, and log each run in the
  problem's runs file (date, job id, commit, change, score with 95% CI, notes), including the
  ones that lose.
- **Wait on your own job id.** With several checks in flight from this machine, `okc check` can
  return another problem's job. Use `okc check <file> --no-wait --json`, then
  `okc status <job> --wait --json`, and confirm the problem and job id before using a number.
- **Never `okc submit` without the user's go-ahead.** It posts publicly under their GitHub name.
  `okc check` is free of that.
- **Never drop a memory-ordering fence or barrier for speed on test results alone.** Passing
  tests do not prove an ordering safe. Keep the documented pattern (e.g. `fence.proxy.async` on
  the writing side before an mbarrier arrive, when generic stores feed wgmma or TMA reads)
  unless the PTX memory model shows the alternative is correct, and cite that section. The
  attention verify variants with bit 2 (var 2, 18) do this; submissions use the fenced ones.
- **Keep audit attribution exact.** Verify every cited run ID against its owning receipt or
  run-log row. Distinguish what an uploaded report says from what its cited sources say;
  do not attribute a source recommendation to the upload without checking the upload text.
- **Record losers in `docs/ruled-out.md`.** Every experiment that loses or is closed gets a row
  there (evidence with job id, and what it rules out), in addition to its runs-log entry. Read
  it before proposing an experiment; reopening one needs new evidence, stated explicitly.
- **Parked is not pruned.** A design family that lost to the incumbent stays parked, with a
  revisit trigger in `docs/ruled-out.md`. After each integration round (and before declaring a
  ceiling), check whether any trigger has landed and re-test that family against the new
  incumbent with current components. Several component changes can flip a family comparison.
