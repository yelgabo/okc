# Human-directed captain takeover: build and measure one candidate

November2,2026. Latest direct human instructions in OK Driver:

> are we stuck in endless research now? tell the captain to turn this into a concrete high ROI candidate and measure it. we have 10 layer checks left—why are we still only doing source reviews?

> take over as captain

**OK Driver (`01a1963f-0c8e-7d41-be02-0e691dcfcaf7`) is now captain and sole judge-dispatch owner.** Existing OKC Developer (`01a19976-9e99-79c0-bab1-644bd8c0293a`) is implementation owner. No new chat or duplicate kernel worker is created. The prior T3 captain `d8754d09-1fc2-4819-8189-18d8dda5cd73`, “Open Kernel Challenge Top 100 Strategy”, was observed in the T3 app as **Run interrupted / Stopped by user / Goal paused**. Leave it paused; do not resume its research loop or allow competing dispatch/allocation. Existing researcher may finish the already active required-mainloop mechanism comparison and send evidence to this captain, not allocate further work.

This direct human instruction supersedes the earlier SOURCE-ONLY/no-implementation/no-target allocations for the new fixed candidate below. Correctness, memory ordering, stored-weight precision, external-code and public-submission rules remain. **Private measurement is authorized; no public submission is newly authorized.** Do not interpret10x design ambition as a requirement to prove a speedup before using a check.

## Decision: one native full-reuse two-front candidate

Build **two compute warpgroups /128 output rows per CTA, Kstage128, retained64-token WGS jobs**, against exact public607 sourceba9e327c. This changes the dominant native weight-processing/compute pipeline, rather than another boundary/helper or external-library wrapper. Target B4 atT32 and MAIN atT128/T256 in both contexts; keep T1/T8/MID and unrelated layer work unchanged.

Hypothesis: two independently progressing CTA producer/dequant/compute fronts per SM can hide stalls and improve effective utilization while preserving64-token weight reuse. Halving Kstage payload and CTA row ownership is intended to permit the second resident CTA. Additional K stages, barriers, activation requests, scheduling/fixup and larger grid are real costs included in the complete measurement. **No gain is established yet.** Equal arithmetic does not establish equal latency; we are now testing the execution hypothesis instead of requiring a foreign exact-Muse benchmark.

This is **not** the parked G224 repetition: that design reduced jobs to24tokens atK256 and its measured long-load losses remain recorded. The new fixed design keeps64-token jobs/reuse and changes Kstage to128. No seed, tile, stage, cap or compiler sweep. Read the exact old evidence to avoid inheriting its packing, reservation and gather-spill defects without scrutiny.

## Implementation owner and scope

Existing OKC Developer owns one implementation, generator, source/CPU/compile evidence and new report under `2026-11-02-worker-wgs-k128-full-reuse`. Work from immutableba9 control; preserve production/default files and all losing artifacts. First implementation window: **90active minutes**, report concrete blocking evidence promptly instead of silently extending research. This is an implementation allocation; another architecture-intake report alone does not complete it.

- Preserve NVFP4/E4M3/group16/global-scale semantics, all BF16 boundaries, SiLU, full per-expert norm before weighted combine, route identity/empty handling and final layer outputs. Repack only losslessly without persistent precision expansion. Prove actual K128 address/scale/permutation coverage and count-dependent NTT tails for both contractions.
- Preserve writing-side proxy fences, asynchronous reader retirement, split/fixup publication, epoch/reset and terminal drain. Explicitly reconstruct required progress/lifetime chains for the changed geometry. Never infer safety from tests alone.
- Account for complete static/dynamic shared storage, alignment, reserved block memory, registers/stack/spills and the doubled grid. Use real target occupancy/admission and cooperative launch requirements before launching a grid that requires simultaneous residence. Launch bounds alone do not prove admission.
- Compile the actual selected B4/MAIN bool entries with effective target flags, including applicable C++20 host requirements. Avoid imposing an absolute-zero resource gate stricter than the corresponding legal control, but do not hide spills or infeasible2CTA admission. A hard contradiction must include the exact source/resource pin and failed condition.
- Produce a deterministic plain eligible full-layer candidate under1MB with notices intact and selected-path attestation. Reuse existing proven host/replay checks and ordinary judge entry; do not build another general four-arm observer framework before obtaining a target result.

The existing researcher is advisory/reviewer, not a second implementer. An independent read-only review of this fixed proposal is running in the captain's current task. Correct a specific demonstrated defect within the fixed design; do not turn it into parameter fishing. If the design hits a genuine hard blocker, return exact evidence immediately so captain can select one different implementation, not start another open-ended research queue.

## Measurement allocation and immediate reporting

Fresh preconditions archived alongside this report show no live jobs; recheck at every dispatch. **Reserve up to TWO private layer checks for this fixed experiment:**

1. Captain dispatches the first ordinary private full-layer candidate check as soon as source, compile/admission and startup correctness gates are ready. It must exercise the actual treatment and return official13tests/12benchmarks/12profiles, selected kernel/source identity and owning job ID. Do not wait for reset, an already-published same-shape foreign benchmark, or certainty that it will win. Report the accepted job ID immediately.
2. If the valid result is promising, use one matched complete-layer treatment/control confirmation with independent controls, fresh replay outputs, fixed weighting and source identities. Prefer a substantial gain (roughly3%+ first piece, with a quantified path beyond the moving~6.6% top100 gap), above duplicate-control noise and with affected-load regression guards. A descriptive cross-job score is not a causal estimate. Captain chooses the smallest already-supported paired instrument; no broad new instrumentation project.

First invalid setup/numerical/admission result triggers an exact blocker decision; no blind retry. A valid subthreshold/regression result parks this fixed implementation without a tuning spiral. Any consumed check, including a loss, gets owning raw receipt/profiles, run-log and scoped closure. No public/scored substitution or release/default edit follows automatically. The remainder of the10private layer capacity stays unallocated until evidence changes the next decision.

**Next operational milestone is a runnable pinned candidate and accepted target job, not another generic source review.** Current legacy source gates do not block this human-authorized implementation/private experiment. No job is claimed submitted by this initial handoff.

## Ownership and evidence

Prior source checks and current live capacity/jobs are retained in [preconditions](evidence/2026-11-02-driver-captain-takeover/preconditions.json). The prior T3 captain's paused status was visually verified through the computer-use skill; no message was sent that would restart it. Shared README publishes this ownership transfer immediately. Direct implementation delivery and acknowledgment will be recorded separately. Preserve unrelated WIP and concurrent researcher publication.
