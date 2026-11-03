# Open Kernel Challenge

## Start and ownership

Read [shared steering](docs/steering/README.md), the latest applicable report and newer origin/main commits/run logs. Read only the records needed for the current decision. [Context](docs/context.md) is historical.

Decode32 Worker owns T32/q_len1 sliding/full implementation, compilation, correctness, private checks, own-job analysis, integration and commits. Optimize `46*sliding_us + 16*full_us`; preserve other loads. First target is reproducible2x, then2.1-2.2x. Research owns architectural choices and concrete experiments. No routine Driver approval or handoff. Read the [directive](docs/steering/decode32/2026-11-02-02-end-to-end-ownership.md), [inbox](docs/steering/decode32/README.md) and [status](docs/steering/decode32/STATUS.md). Continue selected work and reuse existing MID evidence.

General Optimizer reads the newest queue and active assignment on every user/scheduled wake. Implementation and measurement precede tracker maintenance. Record actual pickup, selected source/job and next action.

Write live collector output under ignored `.build/` and archive after exit. Rebase can replace a tracked redirect target's inode; recover ended output by its own job ID instead of resubmitting.

Other owners retain unrelated work. Preserve accepted jobs/files, shared quota and single pollers. Never duplicate, cancel or resubmit another owner's job. Check live jobs/quota before a private run. No new agents or project-wide takeover. Publication, pickup and execution are separate facts. Escalate concrete access, technical or permission blockers.

## Code and context

Trace dispatch, generators, imports and file reads before deleting code. Scored-shape inactivity alone does not establish dead code. Preserve fallback/neighbor behavior, frozen active sources, attribution and numerical/order/progress/lifetime guarantees.

Comments explain a non-obvious reason in at most two lines. Remove narration, banners and stale experiment histories. Preserve legal notices, shebangs, tool directives and generation markers. Longer explanations belong in maintained docs.

Generated benchmarks belong in ignored output paths; recover retired snapshots from exact Git pins. [.rgignore](.rgignore) omits historical copies from default searches. Use explicit paths or `rg --files --hidden --no-ignore -g '!.git'` for archived material. Keep current incumbent source and tests discoverable.

Preserve original encodings during bulk document edits. Some historical Markdown is CP1252; never decode with replacement.

## Research and experiments

Before substantial research, read [working principles](docs/steering/2026-11-01-working-principles.md) and the [coordinator method](.claude/agents/okc-coordinator.md). Follow the [standing hotspot goal](docs/steering/2026-11-02-incumbent-hotspot-goal.md) and current owner queue.

Search independently by problem/challenge identifiers, later pages, handles, forks and credits. Record date, queries, coverage and access limits. A miss establishes only that search's result. Choose by complete-call benefit, cost, uncertainty and quota; compare a complete alternative before fragment adoption. Our base must earn retention.

Pin full source, preparation/packing, dispatch, build flags, epilogues, attribution and precision. Separate team rows, exact-file receipts and matched comparisons. Preserve per-workload winners and measure the complete dispatcher instead of adding isolated gains.

Register hypothesis, controls, affected loads, acceptance threshold and stop before dispatch. Use matched fresh inputs/routing, independent mutable states, duplicate controls, graph replay, captured-output checks and priced observer cost. Compiler success, numerical agreement, ordering proof and measured speed establish different things.

Private observers must retain bounded partial evidence on budget/failure exits and preserve the failure status. Register preparation/observation cost and a finite stop; an empty budget-failure report is inconclusive. See [D32 observer closure](docs/steering/decode32/2026-11-02-17-observer-budget-closure-and-refill.md).

Read [ruled-out variants](docs/ruled-out.md) before proposing work; reopening requires stated new evidence. Record every loss in that file and its run log. If a port loses while its full comparator wins, retain receipts, competing explanations, owner and a distinguishing test, or document the stronger priority. Use waiting time for relevant implementation, compilation or correctness work; avoid repeated readiness reviews.

After integration and before ceiling claims, check whether a parked family's revisit trigger has landed and compare against the current incumbent.

## Evidence and release

Profile SGLang before optimizing: copy its baseline with the `# okc: problem=<problem>` header to `<problem>/baseline_profile/baseline.py`, run a private check and keep its JSON. Find receipts with `rg --files <problem>/baseline_profile`; job IDs belong to their problem.

Run `tools/hotspots.py` on the latest run with `--baseline` before choosing a kernel. Append a table to [hotspots](docs/hotspots.md) when a new best lands. Commit every check's profiles and log date, job, source commit, change, score/95%CI and losses in [projection](docs/runs.md), [MoE](docs/runs-moe.md), [attention](docs/runs-attention.md) or [layer](docs/runs-decode-layer.md) runs.

Use `okc check <file> --no-wait --json`, retain its ID and wait only on that ID. Confirm problem/job/source before using results. Never `okc submit` without the user's go-ahead. Cleanup grants no new scored/public permission. Source snapshots and audits are not independent performance candidates.

Never remove a fence/barrier for speed on tests alone. Preserve writing-side `fence.proxy.async` before publication when generic stores feed WGMMA/TMA unless the applicable PTX memory-model proof supports a cited alternative. Retain mathematical boundaries, graph lifetime and progress guarantees.

Cast FP8 values to FP32 for diagnostic finiteness checks, retain FP8 computation and cover finite/nonfinite inputs. Guard `rank: null` before numeric leaderboard comparisons; retain rows and distinguish rendering errors from empty results.

Use explicit source tags for private diagnostic call-site labels. Never inspect Python frames, including `sys._getframe`; the service rejects harness-frame access.

## CUDA source

Before using the November2 H200 playbook, read its [index](docs/research/cuda-h200/README.md), relevant topic/application rows and run `python3 tools/check_cuda_research.py`. Reconcile changed evidence before refreshing hashes. Reference notes do not reopen parked experiments or change ownership.

Resolve native code through the final factory and preprocessed flags. Public607 B4/MAIN already select the parallel prefix in `MOE_CUDA_SRC`. Tiny/Tiny2/MID use WS=1/WGS=0, excluding their helper's serial WGS prefix. Reproduce with `tools/d32_b4_prefix_audit.py`, `tools/gopt_short_prefix_audit.py` and the [source correction](docs/steering/decode32/2026-11-02-13-b4-prefix-selected-source.md).

For WGMMA grouping, read active constants, prove deferred flush conditions reachable for every selected group size, inspect compiled publication operations and disclose rounded flush timing. Preserve compiler failure status and complete diagnostics; output filtering is not build evidence.

## Document discovery and publication

Filename-unknown requests require checkout/remote confirmation and `rg --files` plus `git ls-files` before content search. Include HTML for visual documents. Widen to hidden/ignored files before a scoped miss; distinguish errors/truncation from empty results. Verify title/content, change retrieval method after user corrections and return searched ref plus verified path/link. Follow [the protocol](docs/repository-file-discovery.md). Keep this root rule for the [October28 correction/evaluation](docs/file-discovery-eval/report.md).

Every substantive steering/audit response belongs in `docs/steering/YYYY-MM-DD-topic.md`, with evidence commit/jobs, measurements versus hypotheses, priorities, next actions, validation/stop criteria and gaps. Update the index/history, preserve old reports and use dated corrections. Commit, push and verify remote files in the same task; return report path and commit. Chat-only answers do not complete handoff. An explicit no-write request overrides this rule.

Preserve concurrent work; never force-push or claim blocked publication succeeded. Verify cited runs against their owning receipts/logs and distinguish uploaded claims from checked sources. Plans/specs live under `docs/`; Superpowers specs use `docs/superpowers/specs/`.
