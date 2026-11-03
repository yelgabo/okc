# GOPT-PIPE-01: pipeline-buffer depth — General Optimizer

November 2, 2026 Pacific. Evidence checkpoint: `d5f5918b2514a33eb6ad908037a17fbdbcb782af`.
User request: "give this to general optimzier, unless cuda optimizer already is on it". "This" is the preceding explanation and optimization question about three versus four pipeline-buffer slots, not four distinct mathematical operations.

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
