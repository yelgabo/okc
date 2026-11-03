# MoE worker pickup and whole-unit phase-B comparison

November 2, 2026 Pacific. OKC MoE Worker has picked up standalone `muse_moe_w4a16` P0. General Optimizer keeps Tiny/Tiny2/MID WS work. This experiment owns B4/MAIN scheduling only. CUDA Optimizer's former B4 study is background evidence, with no accepted phase job identified in current shared steering. Public submissions still require user approval. The worker's reminder remains paused.

## Decode32 accepted job discharged

Own private `4a2a79bdac04499a`, source `f7a5d1fb`, is terminal. Its sole waiter43239 ended with exit1. Strict ordered routing fails at T128/full/seed1102031 initial, placement2, row53. Experts175/68 swap slots5/6. Captured treatment selection is false. Twenty-seven of36 rows survive in22 source-pinned progress capsules, including all six T32 rows. Failure capsule elapsed150.244213s. The complete reader rejects the missing matched report and official success. These partial observations cannot qualify an improvement or establish a causal identity-treatment loss. Stop the exact candidate under its registered routing/coverage gate, with no confirmation or integration. No further Decode32 experiment is allocated.

[Complete receipt and log](decode32/evidence/D32-IDENTITY-01/status-wait.raw.json), [decoded progress and route mismatch](decode32/evidence/D32-IDENTITY-01/terminal-analysis.json), [ledger](decode32/STATUS.md).

## Target and baseline identity

A fresh board snapshot has our historical public job `abe7651dbaa44458` at6069.607719us, rank433, and rank100 at5559.830241us. Required reduction is8.39885%. This is the board's complete six-load geometric metric, not a sum of kernel profiles. [Snapshot](evidence/2026-11-02-moe-bqueue/leaderboard.json).

The tracked standalone `submission.py` is the older owned/autotuning architecture. The more recent privately validated standalone adapter executes the selected G1b4 native engine `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71` from full source `ba9e327c`. Its current baseline already has consumer-order weight packing, SC6, repaired activation retirement and B4 prefix. Current private versus-SGLang results around2.33–2.35x do not change the historical public row and are not matched improvements over each other. The new comparison uses this newer validated engine as control, rather than recovering its existing gains under a new name.

The baseline-aware latest profile maps the whole MoE to one persistent kernel. It cannot separate removable time inside that kernel. [Profile map](evidence/2026-11-02-moe-bqueue/hotspots.md). Historical dynamically claimed B saved16–23us in a different owned engine. Current weighted stage balancing changes its premise, so those savings are a hypothesis lead, not a forecast. [Historical comparison](../runs-moe.md#phase-b-claimed-at-run-time-mode-23-2026-10-29), [source transfer audit](2026-11-02-historical-wins-transfer-audit.md).

## Selected mechanism

MOE-BQUEUE-01 retains phase A's weighted stream-K scheduler, gather, packing, math, epilogues and publication. It replaces phase B's static split ranges with atomic claims of complete256-row output units. Each unit accumulates all16 K stages in one CTA. This removes phase-B partial accumulation, remote fixup and designated-finisher waits. A CTA claims new work after its previous unit completes. Weight and activation producers retain their independent rings inside each unit.

The alternative retained for comparison is the original weighted partition. Static whole-unit rounding would remove split fixups but keep a fixed B allocation despite differing A completion times. The claimed queue tests both costs coherently. More packing/compression work has no established material code-byte opportunity on uniform NVFP4 codes; scale-only and scale/group treatments already failed. The closed prefix, NS4, WS norms, ring and direct-weight designs are not reopened.

The extra per-unit CTA rendezvous and lost cross-unit producer lookahead are real costs. Complete-call timing includes them. The experiment does not claim an exposed16us tail or an8.4% expected gain on this engine.

## Ordering, lifetime and progress

Actual preprocessing disables MFIN in selected WGS. The first compiler assertion caught an incorrect parity-counter assumption. The delivered implementation instead uses `p.ctr[1]`, an unused WGS dynamic-pool counter, initialized by scratch creation and reset by the retained post-compute grid-barrier cleanup. A source test checks the forced MFIN0 definition and reset site. This correction lives in the generator and regression test.

Each atomic claim returns a unique unit. It carries an index only. Thread0 publishes the claim in `S.wgr[2]` through a full576-thread `WS_JOIN`; another full-CTA rendezvous follows the completed unit before reuse. Both producer warps remain live at these rendezvous, with only lane0 issuing their copies. Consumers use the unchanged complete-unit `wgs_seg<true,NTT>` specialization with slot=-1. The three roles carry matching stage sequence numbers from phase A through B, preserving slot and barrier parity. No shared allocation or new global workspace is required.

Phase-B activation reads still wait on all eight phase-A contributor releases for their K stage. Existing writer-side proxy fences, acquire loads, TMA completion barriers and activation-slot retirement remain. Phase-A split/fixup functions and the complete native host suffix are byte-identical. The cooperative grid/residency guards and ordinary same-stream launch remain. Every CTA finishes its A work before claiming B, so B cannot prevent another resident CTA from publishing its A work. The unchanged global completion barrier orders all claims before counter reset. The next graph replay follows prior completion. Empty-local runs claim no valid units and still reach cleanup.

B changes the FP32 reduction partition by replacing split partials with ascending full-K accumulation. Required gate/up BF16, SwiGLU BF16, down BF16 and norm boundaries remain. GPU numerical checks, fresh replay, empty transitions and scratch guards must pass; a compiler result does not prove them.

## Frozen source and local qualification

Paired payload SHA256 `4c8c470f408815616472746b99bcda4ae3bdcedb201bd0833284e15f29e685c5`,354612bytes. Candidate native `99a6888d9a39b290636de49b312432b4d0d1df1e8b6b66abe76c738648d8096f`. Runtime ID `dc626398d8b92bf7b89c5b5d204440367dc1fa1aa23239045d2fe5d810e7af30`. [Payload and seven actual compiler requests](evidence/2026-11-02-moe-bqueue/paired.manifest.json).

Both B4 and MAIN, both bool entries, compile under local CUDA12.8.1 SM90a with96 registers,16 barriers,zero stack andzero spills. B4 control has8B stack,4B spill stores,32B spill loads; MAIN control haszero. No C7510 function-boundary serialization warning remains. The direct complete-unit call lets the compiler remove unreachable B split handling. The earlier generic/outlined prototype did not satisfy the resource gate and receives no GPU check. Full final logs, native/device sources, objects and SASS are archived. This device-only build omits target host linking and actual H200 admission. [Resource evidence](evidence/2026-11-02-moe-bqueue/resource-audit.json).

Three CPU tests verify reversible edits, actual factory selection/unchanged skinny requests, untouched math/publication functions, unique complete-unit coverage, stage/parity agreement and replay counter ownership. Two real-Torch CPU facade tests exercise the complete generated host flow and injected guard failure. They observe36 rows,378 comparisons,216 captures and1152 timing calls in the facade, including failure cleanup and partial evidence. They do not execute CUDA math or measure GPU latency. Independent review is unavailable under the user's no-new-agent instruction. [Source tests](evidence/2026-11-02-moe-bqueue/cpu-source.txt), [host tests](evidence/2026-11-02-moe-bqueue/cpu-host.txt).

## Registered private comparison and finite stop

One initial private check of the frozen payload. Two independent controls and two independent treatments, separately owned packed weights/scratch/graphs/inputs, same compilation options and tensor representation. Three banks1102031/1102037/1102049, normal/skewed routes, all nine correctness sizes, all six scored loads.54 cases,378 output comparisons,216 captures,1152 randomized/reversed cold timed replays and54 input restorations. Keep the existing global.015 and row.075 error limits, exact empty output, input/pointer checks and scratch guards. Official five tests/six benchmarks/six profiles remain mandatory.

The unchanged strict six-load contract requires every bank/distribution and all four control/treatment crosses to improve the equal-six geometric metric by at least1% and exceed twice the duplicate-control noise. Any affected T32/T128/T256 cross regression of1% rejects promotion. Retain unchanged short-load rows and inspect their drift before attributing a win. Official elapsed score must be compared separately with5559.830241us; a matched1% improvement alone does not meet the user's8.4% target.

Preparation/measurement has the existing295s limit. On a Python failure, `MOE_BQUEUE_PARTIAL` retains source-bound accumulated rows, counters, stage and error before re-raising and resetting scratch. Blocking compiler/device failures can still prevent a capsule; such an outcome is inconclusive and fails qualification. No partial result can pass the reader. One source-policy, budget, correctness, guard, admission, regression or benefit failure stops this exact treatment. No seed/geometry/queue-granularity/cap sweep, larger guard or posthoc dispatch mask. A successful screen needs held-out confirmation before any integration decision.

Next action at registration: publish exact bytes, verify remote identity, check fresh shared jobs/quota and any accepted phase-job ownership, record dispatch-started, submit exactly one private check, persist the accepted own ID and follow one waiter. No public submission.


Dispatch checkpoint: exact bytes published in `b56bd8f1` and verified on origin/main. Fresh shared list has no live jobs and MoE quota has16 checks left. `dispatch-started.json` records the single submission intent before POST.


Own private `e044121f822d47b1` is accepted for the frozen source. [Accepted registration](evidence/2026-11-02-moe-bqueue/accepted.json), [compact ledger](moe-worker/STATUS.md). The shell wrapper failed after receiving the accepted response because zsh reserves `status`; the accepted JSON is complete and no POST was repeated. CLI exit was not recorded. Future collector wrappers use task-specific exit variables. Follow this own ID only.


The sole foreground collector is exec session56587, `okc status e044121f822d47b1 --wait --json`, writing `.build/moe-p0/status-wait.raw.json` with profiles under that ignored directory. Resume this session; establish no second poller.


While the accepted job waits, an additional strict-reader CPU test passes on a synthetic capsule using the archived official receipt schema. It validates complete source binding and rejects wrong-job, changed-runtime, missing timing-row and short-capture cases. This test is reader coverage, not a target result. [Reader check](evidence/2026-11-02-moe-bqueue/cpu-reader.txt).
