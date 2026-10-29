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
