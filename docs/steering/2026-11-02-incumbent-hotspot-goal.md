# GOAL-OKC-HOTSPOT: keep execution flowing against incumbent bottlenecks

## Worker reassignment supersedes the earlier queue table

[New owner lanes](2026-11-02-worker-repurpose-moe-bf16.md): the user repurposed the former Decode32 Worker to **P0 standalone MoE**, and CUDA Optimizer to **P1 standalone BF16 projections**. General Optimizer retains separate MoE WS skinny work. Decode32 as a problem stays **P4**, but its former worker must not be re-tasked there by default. Its already accepted identity-rescale job, if unresolved, needs safe same-owner closure first. The user paused that worker's local self-reminder; **do not resume it or assume GitHub publication wakes the paused worker**. The older owner queue below is superseded; use the current `docs/steering/README.md` and this handoff. Existing hourly research steering is separate and must respect the new owner split.


## Latest user priority override — November 2, 2026 Pacific

**Decode32 is P4, not the default or exclusive optimization priority.** The user supplied top-100 cutoff gap measurements and explicitly ordered: **“stop treating decode32 ase the only important problem, its P4 right now.”** This overrides the earlier P0 wording in the original goal and any historical Decode32-first steering. It does **not** cancel an accepted Decode32 job, replace its worker, weaken correctness, or create new public-submission authority.

**Immediate resource allocation:** P0 `muse_moe_w4a16`, P1 `muse_bf16_proj`; pursue other standalone attention/whole-layer opportunities on measured ROI, but **reserve the P4 label for Decode32 specifically**. Relative ordering of other tracks is provisional pending source-qualified opportunity evidence. `muse_prefill_layer` is an independent large-gap track; do not silently assign it to the existing MoE/BF16 workers without a verified, separate plan. Keep full-layer integration as necessary to validate portable standalone wins, without treating all full-layer work as a Decode32-only project.

User-provided cutoff table (judge microseconds, historical as provided; NOT claim of a live CLI refresh):

| Problem | Your time (µs) | Top-100 threshold (µs) | Gap (µs) | Relative reduction |
|---|---:|---:|---:|---:|
| BF16 projection | 3,161.9 | 3,053.3 | 108.6 | 3.43% |
| MoE W4A16 | 6,069.6 | 5,559.8 | 509.8 | 8.40% |
| Attention decode | 1,446.2 | 1,405.8 | 40.3 | 2.79% |
| Decode layer | 10,639.4 | 9,958.4 | 681.0 | 6.40% |
| Prefill layer | 318,442.0 | 204,829.9 | 113,612.1 | 35.68% |

The MoE + BF16 absolute gaps sum to 618.4µs versus 681.0µs for decode layer. **This is a useful hypothesis, not an additive causal budget**: standalone and layer differ in workloads, selected variants and overlap. Any claimed transfer needs matched per-shape profiling and an end-to-end layer comparison.

**Steering change:** The hourly research/refill task must check the standalone MoE and BF16 queues **first**. When they are empty or stalled, request fresh source-qualified alternatives from “Monitor muse challenge disscussions” and “OKC CUDA research”, with primary artifacts and non-duplicate tests. Decode32 retains its existing ownership and any accepted work but must not consume the only new research task or monopolize steering attention. Its separate progress watch can still report real material outcomes at lower priority. Do not mislabel stale research from the original first-priority era as current P0.



**Owner directive, November 2, 2026 Pacific:** “/goal optimize hotspots in incumbent architecture. keep the tasks flowing to the workers pls. if running out, ask ‘Monitor muse challenge disscussions’ and OKC cuda research agents to widen the search space.”

This is a standing steering rule for the current project, not an instruction to optimize a different model/runtime or change safety/release permissions. **Optimize the selected current incumbent's critical path**, as measured across official scored shapes and especially the user's prioritized T32/q_len1 decode row. Avoid new architecture rewrites by default, while permitting genuinely different, source-backed bounded treatments when local tuning has saturated.

## Current working queue and independence

| Owner | NOW | NEXT / reserved action | Duplicate and boundary guard |
|---|---|---|---|
| **Decode32 Worker** | Finish own strict attention matched-routing diagnostic, accepted private layer job `5909739e8a4445b2` (frozen `9e5e5d64`), then resolve validity of lookahead from its own receipts. | If valid gain: confirm and integrate through D32-owned full-layer test. If parked/blocked: use a **specific research-approved B4 critical-path mechanism** from CUDA-MOE-PHASE-01; if none, perform only the bounded T32 pooler/router selected-choice check already identified in [recon](../../archive/docs/steering/2026-11-02-decode32-recon-after-460.md), then request fresh architecture work. | D32-MOE-01 warp-prefix is already implemented in selected B4, closed as duplicate. No second layer job/poller. No Driver approval handoff. |
| **OKC CUDA Optimizer** | Scale-only and scale+group treatments are terminal and not promoted. | **[CUDA-MOE-PHASE-01](2026-11-02-cuda-b4-phase-attribution-task.md)**: identify actual exposed B4/MAIN entry, A→B handoff or finish-tail cost on the selected WGS via valid low-perturbation evidence; deliver one specific mechanism or defensible no-op. | Do not edit Decode32’s layer experiment; no code-group/register sweep; prior observer `f147d06...` was too intrusive. |
| **OKC General Optimizer** | GOPT-PIPE-01 NS3→NS4 is terminal loss, three slots retained. | **[GOPT-SHORT-02](2026-11-02-general-short-prefix-task.md)**: selected Tiny/Tiny2/MID helper *still has a serial job-cost scan*, distinct from B4’s already parallel scan; qualify one tiny-helper-only treatment or close by source/codegen evidence. | No repeat NS4 or helper/B4 source confusion; no D32 layer integration authority. |
| **Research/steering (this chat + scheduled goal review)** | Maintain selected-source truth, experiment dependencies and measured ROI. | Post the next bounded worker-ready task when the owner's queue is empty or a frozen negative result closes it. | Reconcile all live GitHub commits, accepted job IDs, source hashes, prior closed variants and current separate owners before assigning. Never create routine permission gates. |

Public layer incumbent: source `muse_decode_layer/cand/g1b4_stage_release.py`, canonical LF SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`, scored job `607eb52bdaf24160`; prior six-load layer speedup `2.126082676x`. D32 public row ~11.888ms, SGLang ~22.149ms. The first attention experiment had C32~11.882ms, **not a verified improvement** because different-input/node independent comparison; current paired routing failed setup and strict diagnostic is pending. All numbers are dated, not live predictions. Update from newer measurements rather than freezing these as current.

## Task-flow protocol: no idle worker, no redundant experiments

1. **Pull facts before steering:** Read current `docs/steering/README.md`, owner's watched status/inbox, recent main commits and accepted job ledger. Distinguish source-selected paths from old inline literals; distinguish work in progress, terminal valid result, setup failure, unreviewed lead and stale report. Remote absence does not prove another agent idle or its local work nonexistent.
2. **One in flight + one ready:** Each active worker gets at most one independent **immediate** kernel experiment and one ranked **ready** candidate. The next task is prepared asynchronously in research while the owner runs the current job. Do not create a second experiment for an already accepted job, cancel a worker's own poller or interrupt an active turn to force a handoff.
3. **A task is concrete, not a literature review:** Source and final selected compiler inputs; observed real hotspot; exact performance mechanism; affected shapes/weighted upside *conditional on evidence*; validation of BF16/FP8 and memory ordering; matched nulls and full-endpoint rule; finite stop condition. Prefer small source-isolated tests; use private GPU quota only for runnable non-duplicate candidates. After a negative, park the exact implementation; do not sweep knobs.
4. **Release rules remain:** same correct outputs, fresh routing/replay and all five other layer paths, no seed/judge/data shortcuts, no unproved fence/barrier/progress changes, no unauthorized scored/public submits. A local microkernel or SASS gain is not automatically a layer gain.
5. **Refill from evidence:** When an experiment closes, inspect the next-ranked independent lead, verify it against current selected source and prior experiments, and post concise runnable instructions in the owner's existing watched inbox/shared steering. If new candidate would collide with another owner, reserve it rather than redirecting their work. Record pickup only on acknowledgment. Keep new steering reports in the repo with indexed pointers and verified commit.
6. **When the credible queue is depleted:** If an owner has no source-qualified next task after checking (a) live selected-code hotspots, (b) retained profiles/negative ledger, (c) complete comparable public incumbents, **request widened search from both named agents**:
   - **“Monitor muse challenge disscussions”**: inspect fresh public Muse/Open Kernel Challenge competitor code, exact problem names, organizers' updates and participant discussions (including the existing competition-watch baseline) for a materially new algorithm/schedule and exact source URLs/commit, relevant scored shape and legitimacy; no rumors, copied score claims or judging shortcuts.
   - **“OKC CUDA research”**: search H200 SM90a primary architecture/PTX/CUTLASS/FlashInfer and recent MoE/decode studies; reconcile each lead with actual current selected CUDA source/SASS, active experiments and scoped negative receipts; deliver a *ranked set of two to four falsifiable mechanisms* with expected critical-path effects and verification/stop gates.
   Publish a dated combined request under `docs/steering/` with both agents explicitly addressed, link it atop the steering index and relevant watcher inbox, and mark **shared-repo request only** unless the target agent acknowledges/direct messaging was independently verified. Do not pretend this tool can inject an asynchronous message into another running Codex chat. Keep other workers executing their qualified work rather than pausing for that research. If no new qualifying lead is discovered, report this limitation instead of fabricating a task.
7. **Alerts only for changed decisions:** completed job, verified useful new candidate, concrete blocker, queue exhausted/research request, or prolonged documented idle/waiting. No noisy repeated status updates, and no claim the worker is asleep just because recent commits are absent.

## Outcome measurement

For D32 use `C32_us=46*sliding_us+16*full_us` with raw and adjusted measurements and the same-input/control distinction. For whole-layer compare all six load ratios with the actual official score; do not add component speedups linearly. Record job ID, physical SHA, selected variants, target setup/correctness/bench profiles, null spread and complete-call effect. Pursue stable measurable gains even if small, but explicitly prioritize estimated `gain relative to noise × probability of correctness ÷ engineering/check cost` rather than busywork.

## Limitations and publication state

Only the repository can be used to publish cross-agent steering at this checkpoint. No direct cross-chat send or interruption of the named agents has been verified. New queue tasks are **posted, not picked up**. The existing hourly Decode32 status watch remains independent; the goal review should avoid duplicate notices or polling active GPU jobs. This report itself does not change kernels, run GPU jobs or authorize additional public submissions.
