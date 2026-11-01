# RMW H200 setup failure and user-cleared goal checkpoint

2026-11-01. Evidence checkpoint `0f090a10cad5ad6d4826c91456767b6fb701db0a`;
stop-state inventory at23:18:32UTC. The user cleared the goal during the
preparation-fix review. `get_goal` confirms no active goal. Optimization,
testing and new submissions have stopped. This report preserves completed
evidence and unfinished work; it does not authorize another run. Prefill
remains excluded. The reset automation remains separately paused.

## Authoritative H200 outcome

|Field|Observed result|
|---|---|
|Own public job|`3fd1317143304bd5`, muse_decode_layer/run|
|Source|`g1b4_common_short_rmw_release.py`,914572B, SHA256`97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7`|
|Terminal|done; result.status=ok, passed=false, returncode112|
|Actual failure|setup FAIL: `AssertionError: GQUAL_PREPARE_BUDGET`|
|Official records|1 failed setup;0 correctness tests,0 benchmarks,0 profiles|
|Score / CI / latencies|None|
|Static checks|ok=true, flagged/rejected/flags empty|
|Hardware|H200; server node`n-2bbs7xxbcpmk3emdvrn5jsqeic`, cluster sfo3, driver595.91.07|
|Worker timestamps|23:04:18.295910UTC start;23:10:10.610105UTC finish|

The CLI exited1 **with a valid owning terminal JSON on stdout**. Preserve
that JSON even when the CLI exit code is nonzero; this was a judge setup
failure, not a lost retrieval. The earlier queued intervals and local
observation timeouts did not cancel or restart the remote job.

Canonical [terminal receipt](../../muse_decode_layer/profiles/g-common-rmw.3fd1317143304bd5.terminal.json),
[archived log](../../muse_decode_layer/profiles/g-common-rmw.3fd1317143304bd5.log),
[transport envelope](evidence/2026-11-01-rmw-wait-2258/persistent-wait-result.json)
and [terminal review](evidence/2026-11-01-rmw-wait-2258/terminal-review.json)
retain source/job/owning-log provenance, evaluator image/file hashes and
machine snapshots. The original CLI log path is also retained for ownership
validation. GPU telemetry snapshots are not SM/HBM utilization profiles.

The log first prints `G_RELEASE_STARTUP_OK native` with56 verified expert
replays. It subsequently reaches `_gq_private` → `_gq_empty` → the total
295-second guard. The guard is at empty-function entry, but the log has no
cell ordinal or per-phase elapsed times. It cannot identify the first cell,
the dominant preparation stage, exact setup duration or reusable work already
performed. Earlier cells may have run. This is neither a numerical loss nor
an RMW performance result; full-layer qualification remains incomplete.

The old `1fad70bbe3854cbc` independently ended infra_error/refunded=true.
It produced no measurements and was not retried. At23:12:57UTC the fresh
layer ledger was9/10 scored used, **1 scored left /0 checks left**, with no
active jobs. This is a dated snapshot, not a future balance. The23:16:56UTC
board still retained607 at rank324/1169 and2.1261x; rank50 was2.2935x and
rank10 was2.4074x (board-rounded values). No top10/top50 result is established.

## Unfinished preparation fix: not ready

[Independent setup review](2026-11-01-g-qualifier-budget-independent.md),
rootda8c9c6/0f090a1, and [reference work inventory](2026-11-01-g-reference-prepare-budget.md),
root02876a3/ownerae0301c, separate source counts from absent stage measurements.
The original dequantization cache already exists. The bounded new hypothesis
was per-cell reference-back reuse for the same immutable front/weights owner
and **exact full actual routing IDs**. All72 validations, fresh/empty/cache/
capture/output checks and the295-second limit were to remain intact.
With stable repeated IDs this reduces144 reference-back evaluations to48;
that is an operation count, not a measured speedup or promised budget fit.

The draft is preserved on `codex/today-moe` at
[`b13ac0fd90dedc53a25edd5b1a739cd6103a1ea1`](https://github.com/yelgabo/okc/commit/b13ac0fd90dedc53a25edd5b1a739cd6103a1ea1),
including builder`tools/g_common_release_budget.py`,11 passing CPU-facade
tests, manifests and implementation report. Its draft RMW output is917896B,
SHA256`82234b757e64242fa23d704a909c6996c602e3f0dae327fb66a67260243544d8`.
**Do not submit this draft.** Review reproduced a full-prepare failure under
`torch.inference_mode()`: inference tensors do not provide `_version`, which
the draft cache reads. This is a new source-compatibility blocker, not an
H200 measurement. The existing11 passes do not cover that failing context.
No corrected inference-compatible source pin was produced before the stop.

The likely next review was a narrow normal-tensor/no-grad context for only
untimed private reference/clone work, leaving normal production preparation
and timed kernels unchanged. This remains an unimplemented proposal here.
Do not silently treat unavailable version counters as proof of immutability.
Any correction needs actual both-arm outer-inference-mode tests, context and
exception restoration, immutable ownership proof and unchanged full gates.

Reader changes are unfinished in
`C:/Users/gretc/Documents/GitHub/okc-today-attn`, base3a08ce4. A
[WIP patch](evidence/2026-11-01-rmw-stop-checkpoint/reader-wip.patch)
preserves the uncommitted reader/test edits; no new artifact was approved.
They were preparing explicit reviewed-source pins and cache-stat validation,
while preserving old979 receipt support. Root's active reader remains the
previous reviewed old-source reader. Owner worktree state and exact heads
are in the [stop inventory](evidence/2026-11-01-rmw-stop-checkpoint/state.json).
The failed979 artifact is unchanged. No draft builder or new release was
merged into the canonical candidate path.

## Pickup only after an explicit new instruction

1. Read this report and the preserved source/proof reports; do not repeat959
   blindly or infer a numerical/performance loss from setup failure.
2. If the user resumes, fix inference compatibility and complete peer review
   of the exact memo semantics, fallback, cleanup and phase breadcrumbs.
   Keep all native strings/timed kernels/reference math and coverage intact.
3. Generate a distinct artifact/pin; never overwrite the failed source. Add
   flushed bounded phase/cell timing so a future budget failure is attributable.
   No guarantee of fitting295 seconds exists from present evidence.
4. Reconcile live jobs/quota and existing authorization before a future
   official action. A private1% win or separate common GPU run is not a
   prerequisite merely to measure a locally eligible fixed arm; the user
   clearing the goal currently prevents further autonomous action.
5. Require actual setup success plus13 tests/12 benchmarks/12 profiles and
   full native startup/resource/dispatch evidence before interpreting score.
   Preserve any failed receipt and stop an exact losing implementation under
   its registered criteria. Stage evidence before reusable back work would
   falsify this proposed budget remedy; do not remove validation to fit.

The historical607 board entry remains retained. Neither the failed959 nor
the blocked822 draft is a qualified new submission recommendation. The
overall objective remains unachieved; it was cleared by the user, not marked
complete by the agent. No new GPU run, optimization test or submission was
performed after the goal was cleared.
