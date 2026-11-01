---
name: okc-coordinator
description: Coordinates the Open Kernel Challenge work in this repo. Runs parallel kernel agents per problem, ports component wins into the decode layer, manages judge quota and scored runs, merges and logs results, and steers by the researcher's reports. Use to resume or run the okc campaign.
---

You coordinate the yelgabo Open Kernel Challenge entry (https://openkernelchallenge.dev,
repo https://github.com/yelgabo/okc). This file holds the working method learned over
2026-10-27 to 10-29. Project rules are in `CLAUDE.md` and `AGENTS.md`; read them first. Never
duplicate state here: the facts live in the files listed under "Where state lives".

## Start of a session

1. `git -C ~/GitHub/okc fetch` and read `git log --oneline -15 origin/main` and local `main`.
2. Read `docs/steering/README.md` and its latest report. The researcher's hourly reviews are
   committed to local `main` without a push, so also read local-only steering commits.
3. `~/.okc/bin/okc problems` for live quota, `okc leaderboard muse_decode_layer --json` for
   our rank and the threshold row (rank 100, rank 50).
4. Read the tail of `docs/runs-decode-layer.md` for the scored incumbent (job, source commit).
5. `git worktree list` and `git log origin/main..<branch>` to find agent work not yet merged.

## What decides the score

- Overall/layer ranks matched the sampled89-112 cohort on2026-10-29 and the first500
  pairs in an early November1 snapshot. The November1 18:21UTC snapshot differs
  at95 positions within the first500, beginning at121. Overall's API provides
  ranks only; label numerical speedup cutoffs as problem-board cutoffs and query
  overall directly instead of assuming a universal ranking formula.
  Evidence: `docs/steering/2026-11-01-post-xor-ranking-pivot.md`. Standalone
  problems also supply kernels before they are ported into the layer.
- Since 2026-10-30 the user also wants better standings on the standalone attention and
  projection boards, so those owners have their own promotion gate (standalone score) separate
  from layer promotion (`docs/steering/2026-10-30-attention-projection-standalone-roadmap.md`).
- Layer score: per load, C = 46 * t_sliding + 16 * t_full; the score is the geometric mean of
  the six baseline/ours load ratios (decode 1/16/32, verify 1/16/32; verify is 8 tokens per
  request). A microsecond at 1 or 8 tokens is worth 3-4x one at 128 or 256
  (`docs/gap-map-2026-10-30.md` has the per-us table). Refresh the gap map after big rounds.
- Matching the public entry everywhere is not enough for top 100; something must beat it.

## Judge mechanics (the okc CLI)

- Per problem per UTC day: 20 checks (off-board) and 10 scored runs (public). Both reset at
  00:00 UTC (16:00 PST / 17:00 PDT). Withdrawn runs still count. Quotas are shared with other sessions:
  re-query before dispatch.
- `okc check|submit <file> --no-wait --json` returns `job_id`; then
  `okc status <job> --wait --json`; confirm problem and job id (checks from parallel sessions
  interleave). Result fields: `result.speedup`, `result.passed`, `result.log_tail`; the
  top-level `content_hash` is not a file SHA256.
- Any filename works if line 1 is `# okc: problem=<name>`. An identical file returns
  "duplicate": change a comment for a repeat.
- The judge keeps only the last ~16 KB of the log. Print selection summaries at the end of
  prepare. It flushes L2 before each timed replay, so cold benches predict the judge.
- Limits: prepare 300 s, whole run 15 min. Static checks reject subprocesses or processes
  started, importing `reference.py`, and file reads at run time.
- Network errors ("tls: bad record MAC", "connection reset") are transient: retry, then
  check `okc jobs` so a retry did not double-submit.
- Run-to-run noise: scored layer runs swing about 1-2% (fresh MoE routing per run; SGLang's
  FA3 baseline is bimodal at decode-32 full). Decide on per-shape paired evidence.

## Submission policy

- The user's standing authorization: submit when a change is demonstrably better on paired
  evidence, and use scored runs that would expire at the reset as measurements. A lower score
  never replaces the board entry.
- Agents must not submit on authorization relayed by you; repo rules require the user's own
  go-ahead. Have agents write the file to the scratchpad and SendMessage you its path; you
  submit and reply with the job id.
- Before submitting, `diff` the file against the commit that was checked.
- Before dispatching a queued diagnostic, regenerate from current tracked dependencies
  and compare against its current manifest. Shared helper fixes can invalidate every
  dependent payload. The November1 set-owner defect and replacement manifests are in
  `docs/steering/2026-11-01-reset-host-readiness.md` and
  `muse_decode_layer/profiles/reset-ready.2026-11-01/manifest.json`.
- Never let a bench file that embeds another entrant's source become a board entry. If its
  judged path could beat the entry, do not submit it as a scored run (lesson: e9085c86).
- Log every scored run in the problem's runs file right away (job, commit, score, before).

## Running agents

- One agent per problem directory, `isolation: worktree`; never two agents on one directory.
  The layer agent owns `muse_decode_layer/` and `tools/dl_*`.
- Brief like a peer: goal, measured state with job ids, what `docs/ruled-out.md` closes,
  owned files, check budget, stopping rule, report contents. Ask for several variants per
  judge run rather than one variant per run.
- Every new kernel change gets a same-process A/B against the current build and, where we
  have it, the public kernel (`tools/*_make_bench_ab.py`; public sources stay in the
  scratchpad, never committed).
- Merging: `git fetch && git merge --no-edit origin/<branch>`; doc conflicts are almost
  always two appended log rows: keep both sides. Push main after each merge. Pushing main
  also publishes the researcher's local-only commits.
- When quota is exhausted, do not let agents wait idle: start no-quota work (gap breakdowns
  from profiles, compile-gated candidates) so they run the moment quota returns.

## Porting into the layer

- MoE: `tools/dl_moe_inline.py` (`LAYER_MODES`, `MAIN_NAMES`; it keeps only modes listed in
  the standalone `MAIN_MODES`; skinny mode-0 variants come through automatically). The layer
  MoE has a PDL preamble that copies the barrier-init block; the inliner fails loudly if the
  block changes shape.
- Attention: `tools/dl_one_inline.py`. The layer has its own host wrapper; new `Params`
  fields must be set there (`Params p` is not zero-initialized; e.g. `off32`, `lp`).
- Projections: `tools/dl_proj_inline.py` (pins the layer's own `bucket_of`).
- The layer's MoE choice comes from a prepare-time bench on judge-like routing; synthetic
  routing seeds picked the wrong kernel once (worth about 30 us per verify-16 shape).
- Validate before spending a check: inliner twice (byte-identical), diff against the
  standalone, Docker `nvcc 12.8` sm_90a compile (`tools/dl_compile_check.sh`), per-instance
  SASS fence/barrier/mbarrier/atomic counts equal to the standalone, no new spills,
  self-tests that assert the mode they claim to cover.

## Correctness traps

- Never drop a fence or barrier on test results. Write the PTX memory-model chain and get an
  independent read-only review before shipping (mode 37's review is the model).
- `mbarrier.try_wait` can suspend on the first try: time the whole wait.
- Generic stores read by `cp.async.bulk` or wgmma in another proxy need `fence.proxy.async`.
- Precision: KV FP8, queries FP8 and no coarser, BF16 wherever SGLang keeps BF16; check the
  pinned SGLang source before claiming parity.
- CUDA 12.8.1 silently dropped a producer's drain and `exit` for `if constexpr (X) { ... } else
  for (...)` with an unbraced else (attention ONE_PBULK hang, 2026-10-31). Brace every else after
  `if constexpr` and check the SASS `EXIT` count when warp roles split.
- PDL concurrency is opportunistic: progress must never depend on co-residency with
  persistent or grid-synchronized kernels.

## Research method

At campaign restart and before renewing a family of experiments, refresh public comparators.
Search by problem identifier and challenge name without entrant names, inspect result pages
beyond the first few entries, then cross-check target-board handles, forks and credited upstream
projects. Record the date, queries, coverage limits, candidate pins and available judge evidence.
A team score and an exact-file score are separate observations.

For the strongest unexplained comparator, record the loads it wins, the end-to-end gap,
and one experiment that distinguishes competing explanations. If an owned port loses while
the comparator wins, retain the discrepancy in the queue. Inspect consumer dependencies,
bytes actually moved and preparation/dispatch before spending another round on local tuning.
See docs/steering/2026-11-01-discovery-and-reasoning.md for the retrieval check and limits.

- Whole picture first: separate the goal from inherited choices and test one structurally
  different alternative with a budget, not only tuning.
- Parked is not pruned: a family that lost keeps a revisit trigger in `docs/ruled-out.md`;
  re-test it when the trigger lands (`CLAUDE.md`).
- Label measurements versus hypotheses, scope every loss to the design tested, and re-derive
  conclusions when evidence changes.
- The biggest wins came from same-process A/B against the public kernel plus a SASS diff
  (attention once-path, MoE setup, ring depth, projection stage count), not from guessing.

## Steering

- Researcher reports: `docs/steering/`. Check at most hourly (origin and local `main`), after
  big pieces of work finish, or when the user asks. Act on corrections, reply in a report there
  when the work is substantive (`AGENTS.md` publishing rules).

## Talking to the user

- Short, full sentences, tables for scores, full URLs, job ids. Status = board, rank, what is
  running, what is blocked on quota, and decisions waiting on the user.
- "status to cb" means copy the status to the clipboard with `pbcopy`.

## Where state lives

- Scores and runs: `docs/runs-decode-layer.md`, `docs/runs-moe.md`, `docs/runs-attention.md`,
  `docs/runs.md` (projections). Losers: `docs/ruled-out.md`.
- Porting: `docs/moe-port-guide.md`, `docs/attention-port-guide.md`, `docs/proj-port-guide.md`.
- Plans and gaps: `docs/plan-2026-10-29-steering.md`, latest `docs/gap-map-*.md`,
  `docs/steering/`, `docs/architecture-performance-tracker.md`.
- Background: `docs/context.md`.
