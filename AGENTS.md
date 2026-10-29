# okc: Open Kernel Challenge work

Background and decisions: `docs/context.md`. Every run: `docs/runs.md` (bf16_proj),
`docs/runs-moe.md`, `docs/runs-attention.md`.

## Repository file discovery

For filename-unknown document requests, follow `docs/repository-file-discovery.md`.
Confirm the checkout and remote, enumerate repository paths with `rg --files` and
`git ls-files`, then inspect candidate contents. Include HTML when the user describes
diagrams or a visual document. Search misses establish only that a search found no
matches. Before reporting a scoped miss, widen the path inventory, account for hidden
and ignored files, and distinguish errors from empty results. Stay in the requested
repository unless evidence warrants a different source. Verify the title and relevant
content before recommending a candidate. User corrections require a different retrieval
method. Record the searched checkout or ref and return the verified path or link.

The 2026-10-28 incident and reproducible before/after evaluation are recorded in
`docs/file-discovery-eval/report.md`. This rule must remain in the root instruction file
so future sessions load the correction without needing the original conversation.

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
- **Record losers in `docs/ruled-out.md`.** Every experiment that loses or is closed gets a row
  there (evidence with job id, and what it rules out), in addition to its runs-log entry. Read
  it before proposing an experiment; reopening one needs new evidence, stated explicitly.
