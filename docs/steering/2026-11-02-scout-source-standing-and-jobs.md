# Public source standing and live job snapshot

November 2, 2026 at 17:47 PST. Repository evidence base `810d82649db8ee64382db021b758c2d1d14d9cf9`. User asked whether the scouting leads are the best and which jobs are running. [Captured live responses](evidence/2026-11-02-scout-status-0147/snapshot.json).

## Which sources are strongest

The 36 repositories in the [scout baseline](2026-11-02-competition-watch.md) are watch targets, not a ranked recommendation list. Some contain starters, slower entries, orchestration or audit material. The strongest complete public implementation we have source-associated evidence for remains [keell0renz's fair archive](https://github.com/keell0renz/muse-open-kernel-challenge/tree/890809cee421500b00126d084e81b338ef9181ab), followed by the known alygg77/vjeux lineage. This is a conclusion within the recorded discovery coverage. It is not proof of the best available source globally.

The exact [README](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/README.md) distinguishes the public layer's reported check2.2641 and submit2.2532 from its historical team row. Those are author-published numbers, not fresh matched reproductions in this chat. Today's source head remains `890809ce`; the change over known parent `8e30e1be` is README only. Its author caveat about a #1 submission is historical attribution. The freshly queried board does not currently place this team first.

| Decode-layer entry | Live rank | Displayed speedup | Interpretation |
|---|---:|---:|---|
| dolanzhao | 1 | 2.4902x | Stronger competition row; this task has not established its exact public source |
| keell0renz | 99 | 2.2713x | Team row, not a fresh result for the pinned fair file |
| alygg77 | 299 | 2.2058x | Known public source lineage; row is historical best |
| yelgabo | 382 | 2.1261x | Our unchanged public job `607eb52bdaf24160` |
| vjeux | 405 | 2.1019x | Known public upstream; row alone does not compare current per-shape choices |

[Official leaderboard](https://openkernelchallenge.dev/leaderboard), [raw layer board](evidence/2026-11-02-scout-status-0147/layer-board.json). The separate [overall response](evidence/2026-11-02-scout-status-0147/overall-board.json) also has dolanzhao first and yelgabo382. Equality in this snapshot does not establish a universal ranking formula. Layer response header ranked1211 and returned1212 rows are preserved.

The [LESSONS.md page open beside the chat](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/LESSONS.md) was read at blob `36fa83cb4e913f05a1f7feb83e216cb651d05a28`. It is useful source-backed participant experience. Its numeric claims and scoped losses remain external reports. The current repository already has a dedicated research digest and a queued B4 prefix lead, so this status check creates no duplicate experiment or new kernel assignment.

## What is running

The authenticated [recent job response](evidence/2026-11-02-scout-status-0147/jobs.json), read at01:47:26UTC, returns50jobs. Exactly one of those is queued or running:

| Job | Owner and purpose | Observed state |
|---|---|---|
| `f0b875484960438b` | OKC General Optimizer, private `muse_moe_w4a16` comparison of current skinny helper NS3 versus NS4 at T1/T8/T16 | running; correctness and timing pending in this snapshot |

[Owner's assignment and accepted association](2026-11-02-general-optimizer-pipeline-depth.md). The source revision is `21db3bc31971ec9d7dd19da21e047adc069ad08c`, payload SHA256 `cf1dbda332e6a53fa581293d1ddf224c91447436e877c2ed005270c7dd1d04fe`. This is a component experiment; no layer gain follows before integration evidence.

Completed work should not be described as still running:

- D32 confirmation `62a4f6b27ed74b50` is done with passed=false. Its [owning failure record](decode32/2026-11-02-11-matched-routing-mismatch.md) identifies `EXPERT_FRESH_ROUTING_MISMATCH` during setup. It produced no matched timings. The worker's next registered step is the strict-gate diagnostic `9e5e5d64`, not a confirmed attention speedup. No accepted diagnostic ID was present in the inspected ledger or live list.
- CUDA Optimizer's `3aa3885f6b734f8c` and `3c45788127e945c4` are terminal. Their [owning report](2026-11-02-cuda-part2-experiments.md) and archived summaries show correctness passes but rejection by their matched speed gates. Neither is a promoted runtime win.
- [Serving jobs](evidence/2026-11-02-scout-status-0147/serving-jobs.json) returned an empty list.

A separate Codex chat listing reported "OKC General Optimizer" and "OKC Decode32 Worker" active, and "OKC CUDA Optimizer" idle. That is chat state, not proof of a GPU process. The job list is the evidence for the running check. No other owner's status waiter was started, and no job was dispatched, cancelled or changed.

## Decision and limits

Keep keell0renz, alygg77 and their upstream credits near the top of the public-source watch. Track stronger leaderboard handles as discovery leads, then require exact source and attributable receipts before comparing implementations. The strongest public archive is a study target, not an automatic base replacement. Our current implementation and worker assignments remain intact.

General Optimizer owns its one accepted waiter and terminal decision. Decode32 Worker owns the routing diagnostic and the already queued next lead. CUDA Optimizer's two measured candidates remain stopped under their owning report. This response adds no allocations or handoff gate.

The snapshot covers the account's50recent kernel jobs and returned serving jobs, not every process on every external GPU. Job state can change after the timestamp. No fresh numerical, compiler, replay or performance test was run here. Board rows may have null ranks; the first rendering stopped on one, while raw responses were preserved. The renderer was guarded and the remaining reads completed. Root instructions now retain that parsing rule.

