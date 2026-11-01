# okc: Open Kernel Challenge work

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

## Choosing the next performance experiment

At campaign restart, before spending another round on a design family, and before declaring a ceiling, the person choosing the experiment owns the following checks. Reuse a dated steering entry if the board, public sources and incumbent are still current.

1. Search independently of user-supplied names. Search the challenge and problem identifiers, page through results, then follow leaderboard handles, forks and upstream credits. Record queries, date, results screened, access limits and promising source pins in the steering report. Inspect per-shape strengths as well as team rank. Distinguish team-best scores from measurements of the published file.
2. Challenge our design even when no public source turns up. For the costly path, name the bytes that must move, which consumers need which producers, and what prepare/dispatch selects at the scored shapes. Derive one materially different design from those constraints. Compare its plausible whole-score impact with the target and the best queued local tweak. State which observation would change the choice.
3. Carry an unexplained comparator win forward. If our port loses and the complete comparator wins on matched inputs, log the losing variant in ruled-out and keep the discrepancy open in steering. Record owning receipts, affected loads, competing explanations, one distinguishing test, an owner and a decision gate. Before another tuning round, either pursue that test or document why another use of quota has greater expected impact.
4. At handoff, distinguish leads discovered independently from leads supplied by the user. Record which evidence changed the next experiment and whether it ultimately improved the validated whole score. A recovered known repo checks retrieval; new independent leads and better decisions test the process prospectively.

Use the coordinator's Research method for execution. Existing rules govern external-code use, correctness and scored submissions.

## Evidence rules

- **Profile SGLang before optimizing a problem.** A normal run profiles only our submission.
  Copy `baseline.py` to `<problem>/baseline_profile/baseline.py` with the
  `# okc: problem=<problem>` header, run it as a check, and keep its result JSON there. That is
  the hotspot map: which SGLang kernels take the time, per shape.
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
