# Immediate backlog while the researcher reviews

November 1, 2026. Evidence checkout: `C:/Users/gretc/Documents/GitHub/okc`,
local main and fetched origin/main at
`97a91dfdfd254ae8b438e1b0bc67c4bef333bbbc`;
remote `https://github.com/yelgabo/okc.git`.
User request: find backlog work we can do now while they talk to the researcher.
This is a bounded backlog reconciliation, not a restart of the stopped GPU campaign.

## Decision and work completed

The best immediate engineering task is to finish the existing preparation
qualifier repair. It is a prerequisite to learning whether the surviving
short-counter RMW treatment helps. Its benefit is enabling a valid measurement;
there is no measured speedup or guaranteed setup-budget saving yet.

Completed here: reconciled the historical backlog entry points with the latest
run outcomes and owner worktrees, and added current-status pointers to the
architecture tracker and paper backlog. This prevents the old family table and
paper queue from being mistaken for dispatch instructions. No kernel, builder,
reader, owner worktree, automation or submission was changed.

## Ordered work queue

1. **Repair inference compatibility in the existing qualifier draft.**
   Resume from `codex/today-moe` at `b13ac0fd90dedc53a25edd5b1a739cd6103a1ea1`,
   where `tools/g_common_release_budget.py` and its CPU tests already exist.
   That worktree was clean at this inspection. The full-prepare reproduction
   already proves that draft `82234b75` reads unavailable inference-tensor
   version counters; repeating the same reproduction is not new progress.
   The existing owner should implement the narrow private-reference
   normal-tensor/no-grad scope, then test both arms under an outer inference
   context, ordinary context, and exceptions. Check context restoration,
   front/weight ownership, exact routing-ID mutation and cache misses, and
   rejection of stale/nonfinite candidate results even on memo hits.
   Preserve the complete coverage, native math, and 295-second guard.
   Acceptance: an exact new pin with passing complete-prepare tests and
   source review; stop on a missing version counter silently treated as
   immutability, weakened checks, or changed production preparation semantics.

2. **Finish the owning receipt reader against that new pin.**
   `codex/today-attn` remains at `3a08ce4` with uncommitted edits to
   `tools/g_common_rmw_receipt.py` and its test file. Preserve those edits;
   the stop checkpoint archives their patch. The canonical reader still
   explicitly accepts only the old `97935e7d` source. Its owner can finish
   CPU negative cases now; final accepted-source registration depends on
   item 1. Require exact source/job/log ownership and complete coverage,
   cache statistics and phase records. Retain old failed-receipt support;
   a nonzero CLI exit with valid terminal JSON must remain a judge failure,
   not become a missing transport response. Stop on unknown source pins,
   ambiguous capsules or incomplete evidence. This is measurement reliability
   work, not another performance candidate.

3. **Use phase evidence to select any further setup-cost work.**
   The existing source inventory identifies repeated front-reference work
   and immutable weight conversions, but no phase timings. After the narrow
   repair, use its bounded cell/phase breadcrumbs to distinguish production
   compile/tuning, front construction, clone/resource work and back-oracle work.
   Share a pure frontend prefix or cache immutable casts only if the observed
   cost warrants it, with separate oracle-equivalence and ownership tests.
   Do not combine these speculative rewrites with item 1 or remove validation
   to fit the budget. Owner: layer qualifier owner; next gate: actual phase
   evidence from an eligible, separately authorized measurement.

Items 1 and 2 are CPU/source work available without judge quota or a new
research direction. They do not require a new kernel design. Existing
implementation ownership is retained; this reconciliation does not overwrite
either owner's work.

## Why these precede other backlog entries

The strongest near-term alternative is another performance design or public
comparator intake. The latest source search already found the pinned G upstream
unchanged and no stronger inspected complete-file replacement. A new source
revision with useful matched evidence would change that choice. Repeating the
same search immediately has less decision value than restoring the current
measurement path.

Projection sparse-XOR variants already lost on H200; attention VP2 already
failed its frozen full-score acceptance gate; the two standalone MoE progress
repair representations already failed their compiler gate. The historical
family table is useful context, but is not a current queue. Reopening those
exact variants needs the new evidence specified in `docs/ruled-out.md`.
Likewise, the old MFIN/SC6 factorial queue is superseded by the later
integration-first and single-arm qualification decisions. More combinations
do not create independent candidates.

## Evidence, limits and handoff

Verified owning run-log rows: layer `607eb52bdaf24160` remains the recorded
incumbent; layer `3fd1317143304bd5` ended in `GQUAL_PREPARE_BUDGET`, with no
official correctness/benchmark/profile results and no score. Its successful
native expert startup marker does not establish complete-layer qualification.
The failing cell and phase are unknown. The inferred reduction from 144 to 48
back-oracle evaluations assumes repeated exact routing IDs; it is a source
count, not measured time saved.

Read the latest steering index and session/orientation reports, current run
logs and scoped closures; inventoried repository paths with `rg --files` and
`git ls-files`; fetched origin and inspected all three existing worktree heads.
The root's unrelated candidate-funnel edit and untracked evidence/profiles were
preserved. No fresh jobs/quota/leaderboard query was needed for this CPU-work
queue, and none was made. Prior balance/rank snapshots are not current claims.
No new experiment was performed and no loss row is warranted for this cleanup.

Supporting reports:

- [Terminal stop and preserved owner work](2026-11-01-rmw-terminal-stop-checkpoint.md)
- [Exact reference-work accounting](2026-11-01-g-reference-prepare-budget.md)
- [Inference-mode blocker and required repair scope](2026-11-01-g-qualifier-budget-independent.md)
- [Single surviving treatment and qualification gates](2026-11-01-g-candidate-funnel-qualification.md)
- [Latest bounded upstream check](2026-11-01-keell-upstream-refresh-2246.md)

The researcher handoff is therefore specific: finish the existing qualifier and
reader before allocating another optimization round. A new compatible pin,
complete checks and useful phase evidence are the next gates. Actual setup fit,
RMW performance and top-50 feasibility remain unknown. Reconcile live jobs,
quota and authorization before any later official action; reset automation
remains paused.
