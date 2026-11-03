# GOPT-PIPE-01: pipeline-buffer depth — General Optimizer

November 2, 2026 Pacific. Evidence checkpoint: `d5f5918b2514a33eb6ad908037a17fbdbcb782af`.
User request: "give this to general optimzier, unless cuda optimizer already is on it". "This" is the preceding explanation and optimization question about three versus four pipeline-buffer slots, not four distinct mathematical operations.

Current decision: **retain three slots; park the exact current-helper four-slot treatment.** Own job `f0b875484960438b` passes correctness and fails the frozen performance gate. The terminal evidence is below. No production change or confirmation.

## Assignment and duplicate check

**Assigned in the shared queue to General Optimizer, unless OKC CUDA Optimizer already has this exact pipeline-depth experiment in flight.** Research inspected CUDA Optimizer's [current report](2026-11-02-cuda-part2-experiments.md) and [execution plan](../superpowers/plans/2026-11-02-cuda-issue-groups.md) at the evidence pin. They record an accepted constant-scale-multiplier MoE check `3aa3885f6b734f8c`, a larger WGMMA issue-group treatment stopped at new spills, and a compiler-only shared-pointer-hoisting interaction. The plan explicitly excludes a generic repeat ring sweep. No active stage-count experiment is documented there. This establishes the published scope, not the contents of another session's uncommitted work.

At the next normal checkpoint, General Optimizer should check the newest ownership/job records once. If CUDA Optimizer has since claimed this exact test, attach this brief to that result and do not duplicate it. Otherwise own this bounded task through its decision under existing private-measurement permissions, without a new review or handoff chain. Do not interrupt a running turn or poll, cancel, modify or resubmit another owner's job.

## The concept to carry forward

In the inspected skinny MoE path, `ring[kNS][kUnitBytes]` contains packed weight chunks and their scales. Stage count means the number of reusable shared-memory slots. Chunk q uses `q % kNS`; after all required readers release that slot, it can receive q+kNS. Chunks do not move through the slots as four different computations. Async copies can load later chunks while consumers unpack and compute earlier ones.

The question is **whether an additional staged/in-flight chunk reduces exposed waiting enough to pay for its shared-memory and synchronization footprint**. Four is not inherently better than three. This description is specific to the inspected weight ring; identify the actual operands and rings in any different selected implementation before transferring it.

Sources: [source corrections](../research/cuda-h200/source_corrections.md), [current pipeline reference](../research/cuda-h200/pipelines.md), and the standalone MoE `issue_unit` / `compute_loop` implementation at pinned source `779b06ad` (blob `96b5bf91433fc87154de1edc40ce7a31b5c4c025`). This assignment is not new timing evidence.

## Concrete task — reuse old results, then one missing comparison

1. Read the owning record for `75bca9775f6f4b0b` in `docs/runs-moe.md` and the relevant `docs/ruled-out.md` entries. The correction supplement says source annotations favor three slots over four at T1/T8/T16; research has not reprocessed that raw receipt here. Do not repeat that identical experiment or claim its scope covers the selected layer-native B4/MAIN paths automatically.
2. Pin the actual target source, prepared variant, ring being changed, current slot count, workload and existing comparison. For layer applicability use `muse_decode_layer/cand/g1b4_stage_release.py` / control `ba9e327c` and its final dispatcher, not an assumed equivalence with standalone `submission.py`. Identify one genuinely missing comparison or a concrete changed premise. If the exact selected case is already settled and unchanged, close this task with its result; do not manufacture a fresh sweep.
3. For an untested relevant case, compare the incumbent with one adjacent viable depth, holding weight packing, scale reconstruction, arithmetic order, tile geometry, grid, issue-group size and dispatch fixed. Reuse existing generation, compiler and matched timing tools. Do not bundle CUDA Optimizer's unconfirmed scale/pointer changes into this first treatment. Check phase wrap, producer progress, all-reader retirement, async operand lifetimes, graph replay and actual shared-memory admission. A source constant change alone does not establish a safe ring change.
4. Implement and measure under existing permissions after the normal shared-job/quota check. Report complete affected-call latency, independent-control spread, selected paths, correctness and resource differences. Resource counts alone are not the win. A promising result earns bounded confirmation; a valid loss closes that exact variant rather than triggering a 2/3/4/5-stage-by-tile-by-register sweep. No new generic framework or proof of a future speedup is required.

## Keep the independent efforts intact

CUDA Optimizer keeps its accepted scale-folding source and compiler work. Decode32 Worker keeps its running attention confirmation, files, execution authority and layer integration. This task is not a transfer of the D32 lane or permission to alter its frozen candidate. An isolated MoE result potentially relevant to T32 must be labeled as component evidence until measured in the complete layer; do not credit it directly to `46*sliding_us + 16*full_us`.

Public-submission permissions, numerical boundaries and ordering/progress requirements remain unchanged. Do not reopen a failed variant merely because this explanatory handoff exists.

## Deliverable and pickup

One short result: `selected path + prior receipt -> exact depth change or already-covered finding -> source/job if tested -> correctness and latency -> keep/park`.

Status at publication: **assigned in the repository; General Optimizer pickup not confirmed**. No General Optimizer session identifier or dedicated watched inbox was established in the inspected records, so this note uses the shared steering entry point and does not claim direct chat delivery. No kernel or GPU action was performed by research for this routing update.

## General Optimizer pickup, November 2 Pacific

OKC General Optimizer, chat `01a18696-ab6b-7cd1-a759-7e0366d672f6`, has read this assignment and takes GOPT-PIPE-01 through its bounded decision. The user asked this chat to wake, identified the assigned task, then steered the work toward measuring the current skinny implementation again if the comparison is missing. The one-time read-only CUDA Optimizer status check records scale folding and issue-group work, with no active ring-depth test. Its current source and jobs remain untouched.

The existing `75bca9775f6f4b0b` receipt and source annotations have been located. The next step is to map the exact current G1b4 tiny/tiny2/MID helper, prepared ring flags and prior controls before choosing one comparison. Earlier owned mode51 four-X-slot result39804be1a59c429c is also recorded as a loss, so it will not be repeated under a new name.

Work is isolated in `/Users/gretchengabo/GitHub/okc-gopt-pipeline-depth`, branch `codex/gopt-pipeline-depth`, based on e365d7b6. The managed-worktree tool rejected the required source checkout because it is outside the chat's attached roots, so Git created this new sibling worktree. The existing tracker checkout and untracked profiles were preserved.

## Current skinny comparison, November 2 at 17:26 PST

The selected missing comparison is the current helper at `OKC_NS=3` versus `OKC_NS=4`, for tiny/T1, tiny2/T8 and MID/T16. The [generator](../../tools/gopt_skinny_depth.py) pins the complete G1b4 control `ba9e327c`, helper `b8ed8543` and native MAIN `c7c201ae`. Its [payload manifest](../../muse_moe_w4a16/cand/gopt_skinny_depth.manifest.json) records all eight exact compiler requests and routing keys. Both arms use identical CUDA and C++ source bytes. Only the selected helper's depth define changes. B4/T32 and MAIN/T128/T256 requests remain identical. The immutable production layer file stays unchanged.

The old [75bca owning receipt](../../archive/docs/steering/evidence/2026-10-29-community-audit/status-75bca9775f6f4b0b.json) compares owned sk8w3s with owned sk8w4s, plus a public three-slot comparator. Its six-seed layer/std means favor owned three slots by 0.9/0.6 us at T1, 2.4/1.8 us at T8 and 3.1/2.9 us at T16. Those are rounded source values. Every paired seed favored three slots. The [extracted36 paired deltas](evidence/2026-11-02-gopt-skinny-depth/prior-skinny-pairs.json) verify that direction. This does not compare the current warp-specialized helper at both depths. The old mode51 two-weight/four-X tradeoff39804be1a59c429c remains closed and is outside this treatment. The [reprocessed old hotspot table](evidence/2026-11-02-gopt-skinny-depth/prior-hotspots.md) uses the archived SGLang baseline eae9a6b7; it describes the old run, not current latency.

The hypothesis is that one extra staged weight chunk may reduce exposed copy waiting in the current helper. It may instead lose through address calculation or register pressure. This bounded comparison preserves the current engine. A larger ring or a layout/issue-group combination would answer another question and is deferred unless this result supplies a reason.

### Source protocol and compiler gate

The current `OKC_WSB=1` path assigns chunk q to q modulo NS, waits for completion phase floor(q/NS) modulo2, reads its codes and scales, then retains the compute-CTA proxy fence and barrier before thread0 refills q+NS. The helper warp consumes the separate partial-sum buffers, not the released weight slot. The producer initially fills from the two preissued slots through min(NS,N); both tested NS values exceed FAST1=2. The A/B sequence uses the same q progression. Four slots stay within the five skinny barriers and eight unit-index entries. Partial-buffer counts, final epilogues, global publication, scratch ownership and cooperative-launch admission are unchanged. No synchronization instruction is removed. The target check must still establish replay correctness and actual admission.

The [local compile record](evidence/2026-11-02-gopt-skinny-depth/compile.json) and neighboring full logs/SASS cover tiny, tiny2 and MID at both depths, both bool entry instantiations. CUDA12.8.1 sm90a compilation passed all six builds with zero spills. Tiny/tiny2 registers change145 to148; MID152 to153. Both arms allocate214784 dynamic shared bytes with288 threads. The fat-buffer maximum already exceeds the four-slot weight ring, so this treatment adds no allocation bytes. Local compilation strips the host extension tail, adds the established declaration-only WGS adapter and compile-only kernel references. Its host metadata probe runs without GPU execution. These checks establish compiler/resource facts, not target latency. The first local attempt failed because Docker could not see the `/private/tmp` mount; its [full failure log](evidence/2026-11-02-gopt-skinny-depth/local-mount-failure.log) is preserved, and all successful builds used the repository mount.

The generated-entry CPU test in [test_gopt_skinny_depth.py](../../tools/test_gopt_skinny_depth.py) passed2 tests. It uses real CPU Torch storage and facaded CUDA/native leaves, checks all eight builds,216 captures,378 observations and1152 timed replay calls, then verifies a deliberately nonfinite candidate fails. It also checks that the decision uses small loads and rejects material movement in unchanged controls. No independent reviewer was used for this bounded task under the current ownership instruction. GPU and host-extension build evidence remain pending.

The research catalog checker already reported a changed `docs/ruled-out.md` hash before this work. The relevant current pipeline, hardware, source-correction and validation notes were reread. Their catalog hashes were left untouched; this task makes no new external PTX claim.

### Frozen private measurement and stopping rule

[Payload](../../muse_moe_w4a16/cand/gopt_skinny_depth.py) SHA256 `cf1dbda332e6a53fa581293d1ddf224c91447436e877c2ed005270c7dd1d04fe`,343611 bytes. Runtime identity `2c5896ff2afd5311cabcbd6e2aeafdf11dbdda9e0db6a865fe4e9a414e154793`. The existing four-state cold-graph harness uses independently prepared control1/control2/candidate1/candidate2, held-out banks1102031/1102037/1102049, normal/skewed EP8 routes, and8 repeats per placement/cell. It checks fresh inputs, empty/occupied transitions, guard tails and observed replay outputs at T1/4/8/16/32/33/128/256/257. Timed cells cover1/8/16/32/128/256, for36 rows. The official arm is candidate2.

Before dispatch, require fresh shared jobs/quota and record one owning private MoE job. No other owner's job is polled or resubmitted. The prospective gain gate is a T1/T8/T16 geometric mean of at least1% in all four control/candidate crosses, for each bank/distribution, exceeding twice both duplicate-control log nulls. Any affected-cell regression of at least1% fails promotion. Unchanged T32/T128/T256 movement beyond max1% or twice that row's duplicate null makes the result unresolved. A promising result earns at most one confirmation. A valid loss parks this exact current-helper NS4 choice. Inconclusive evidence remains inconclusive, without a depth sweep. Any component gain needs a separate whole-layer result before layer credit; this test does not change D32 ownership or production defaults.

### Accepted private job, November 2 at 17:27 PST

Own job `f0b875484960438b` was accepted for `muse_moe_w4a16`, source revision `21db3bc31971ec9d7dd19da21e047adc069ad08c`, SHA256 `cf1dbda3`,343611 bytes. [Accepted association](../../muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/accepted.json). Shared preflight showed19 MoE checks left and separately owned issue-group job3c45788127e945c4 plus D32 confirmation62a4f6b27ed74b50. The CUDA owner's compact status identified its new job as WGMMA issue grouping, not ring depth. General Optimizer owns one foreground waiter for f0b875 only. Target correctness and timing remain pending; no gain or layer attribution is established.

## Terminal result, November 02 at 18:48 MST in America/Vancouver

Private job `f0b875484960438b`, source `21db3bc31971ec9d7dd19da21e047adc069ad08c`, payload `cf1dbda332e6a53fa581293d1ddf224c91447436e877c2ed005270c7dd1d04fe`, completed with5 tests,6 benchmarks and6 profiles passing, no flags and return0. Private harness preparation took279.219903 seconds; official setup took279.220501 seconds. All54 private cases,378 observations,216 captures,1152 timed replays and54 restorations passed. Worst global/row relative errors were0.0000754371/0.0001989355. The [strict owning reader](../../tools/gopt_skinny_depth_read.py) passed source/job/log/record validation. [Raw receipt](../../muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/status-wait.raw.json), [analysis](../../muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/analysis.json), [all six profiles](../../muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/profiles).

Official2.293332488x,95% CI2.258932501–2.327732475, compares the candidate with SGLang, not NS4 with NS3. Every one of24 affected T1/T8/T16 aggregate crosses is below1: range0.966414–0.990167, descriptive geometric mean0.977832. All six bank/distribution groups fail the preregistered promotion gate;17 of18 affected bank/load cells have at least one>=1% regression. One unchanged T256/skewed cell, bank1102049, also exceeds the frozen drift threshold. Therefore the exact causal slowdown is unresolved. The decision is still clear: no measured gain, no confirmation, retain NS3 and park this exact NS4 treatment. This neither rejects four slots in every implementation nor credits any layer gain.

The table averages the two placement medians, then the three held-out banks, separately by routing distribution. Delta percentages are descriptive differences of those averages. The final column retains all12 paired cross factors for each token/distribution group. Each placement/cell has8 cold repeats; no paired confidence interval is recorded. Duplicate log nulls and all36 original rows remain in the linked analysis and [descriptive summary](../../muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/descriptive-summary.json).

| Tokens | EP8 routing | NS3 control us | NS4 candidate us | Delta us | Delta % | Paired factor range |
|---:|---|---:|---:|---:|---:|---|
| 1 | normal | 34.003 | 35.541 | +1.539 | +4.525% | 0.923102–0.986413 |
| 1 | skewed | 34.224 | 35.267 | +1.043 | +3.047% | 0.967369–1.002856 |
| 8 | normal | 79.264 | 80.248 | +0.984 | +1.241% | 0.959355–1.008782 |
| 8 | skewed | 40.792 | 41.467 | +0.675 | +1.654% | 0.955772–1.009459 |
| 16 | normal | 117.579 | 118.965 | +1.387 | +1.179% | 0.975043–1.002516 |
| 16 | skewed | 63.669 | 64.965 | +1.296 | +2.036% | 0.959137–0.994563 |
| 32 | normal | 126.064 | 126.363 | +0.299 | +0.237% | 0.989489–1.001918 |
| 32 | skewed | 57.349 | 57.531 | +0.181 | +0.316% | 0.970926–1.009390 |
| 128 | normal | 187.003 | 186.675 | -0.328 | -0.175% | 0.991055–1.005572 |
| 128 | skewed | 102.616 | 103.387 | +0.771 | +0.751% | 0.982400–1.002804 |
| 256 | normal | 233.579 | 233.936 | +0.357 | +0.153% | 0.990820–1.004343 |
| 256 | skewed | 158.691 | 159.765 | +1.075 | +0.677% | 0.982387–1.000636 |

T32/128/256 select identical compiler requests in both arms and are drift controls, not NS4 measurements. All affected mean cells are slower by0.675–1.539 us, but unchanged-path movement prevents assigning every microsecond to the depth change. Local compiler resources remain145→148 registers for tiny/tiny2,152→153 for MID, zero spills and214784 shared bytes in both arms. The target wrapper passed cooperative admission, but exported no target register/shared-memory census.

The first waiter finished successfully. A rebase had replaced its tracked empty redirected output file, so its terminal receipt was recovered with one owning-ID status read after the waiter ended. The owning downloaded log and all profiles were preserved and strictly validated; no GPU rerun occurred. The root wake-up instructions now require live collector output under an ignored directory and archival only after the process ends.
