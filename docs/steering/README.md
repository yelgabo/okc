# Shared project steering

## Worker repurpose: former Decode32 → standalone MoE P0; CUDA Optimizer → BF16 P1

**[User-directed worker reassignment](2026-11-02-worker-repurpose-moe-bf16.md)** supersedes older owner rows below: **former Decode32 Worker takes P0 `muse_moe_w4a16` architecture/representation and whole-call performance**, **CUDA Optimizer takes P1 `muse_bf16_proj`**, and **General Optimizer retains separate MoE WS skinny work**. The former Decode32 worker must first safely archive/decide its already accepted D32-IDENTITY-01 job `4a2a79bdac04499a` with its sole waiter, if still unresolved; no abandonment or duplicate. CUDA Optimizer checks for its own active phase job before ceasing B4-phase ownership. Do not assign further D32 tasks automatically (D32 is P4); do not overlap the two MoE lanes. **The user has paused the former Decode32 Worker's self-reminder timer: leave it paused; do not re-enable or claim automatic pickup.** This is GitHub steering, not proof either running agent received the reassignment; no new public-submit authority.


## Current user priority: MoE/BF16 first; Decode32 P4

**Latest override** in [the standing hotspot goal](2026-11-02-incumbent-hotspot-goal.md#latest-user-priority-override--november-2-2026-pacific): **P0 standalone MoE W4A16**, **P1 standalone BF16 projections**, with **Decode32 explicitly P4**. Other workloads are prioritized by verified independent ROI; prefill is a separate large-gap research track. User-supplied historical top-100 thresholds imply 509.8µs (8.40%) MoE, 108.6µs (3.43%) BF16, 40.3µs (2.79%) attention, 681.0µs (6.40%) whole decode layer, and 113,612.1µs (35.68%) prefill. These standalone/layer gaps **cannot be summed as a causal forecast**. Research/optimizer task refill must prioritize MoE and BF16 before proposing fresh D32 work. **Existing D32-owned accepted jobs continue safely; no cancellation, duplicate runs, takeover, or public-submission change.** Older D32-P0 statements are superseded, not erased.


Read your owner's row, then its linked record. Check newer commits before acting on a dated job state. The scored whole-layer control is public607, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. [Incumbent source](../../muse_decode_layer/cand/g1b4_stage_release.py)

## Current work

| Owner | Assignment and next decision |
|---|---|
| **OKC MoE Worker → P0 whole-expert** | **MOE-BQUEUE-01 terminal stop.** Own e044121f822d47b1 passes5/6/6 and complete54-case replay checks;0/6 matched groups pass,15 affected regression rows. Official5856.634195us, target5559.830241us remains open. Keep incumbent, no queue confirmation/mask/sweep/integration. No active job. Distinct next treatment needs selected B4/MAIN phase/tail exposure with priced observer cost or a compatible stronger complete comparator. Research has read the terminal and retained this gap; no architecture-next treatment qualified. Reminder stays paused. [Outcome and evidence gap](2026-11-02-moe-worker-queue-experiment.md#terminal-decision-november-2-pacific) · [Ledger](moe-worker/STATUS.md) |
| **OKC CUDA Optimizer → P1 BF16** | **Pickup unconfirmed.** Same-refill C2 screen traces3 owned WGMMA/7 Lt/20 short cells. Register-A exact weight decoding is a structural hypothesis, not a ready replacement: raw complete-call competitiveness and decode economics are missing; M128 byte-only sizing does not reach3.43%. No O32/top100 claim or generic cluster sweep. Old B4 phase study is historical. [BF16 screen](2026-11-02-general-hotspot-search-refill.md#bf16-screen-register-decoded-exact-weights-not-another-clustertile-tweak) · [Handoff](2026-11-02-worker-repurpose-moe-bf16.md) |
| **OKC General Optimizer → P0 MoE skinny/WS** | GOPT-WS-03 b38a9ed6dc9743e9 is stopped; retain NS3/original norm. Same-refill terminal-finalizer split screen has active WS SASS but no measured recovered wait versus added launch/reset/reload cost. Prior three-launch T1 loss retained. **No immediate GPU-ready candidate.** Distinct T1/T8/T16 lane; no B4 queue, norm rescue or earlier-credit sweep. [WS screen](2026-11-02-general-hotspot-search-refill.md#general-ws-next-screen-split-finalization-only) · [Norm evidence](2026-11-02-general-ws-norm-aggregation.md#terminal-decision-retain-baseline-stop-exact-aggregation) |
| **Research/steering** | **P0/P1 scope pickup and findings appended to the SAME refill.** Reconciled through a0a60c53 including BQUEUE's matched loss. Two structural screens with selected baseline binaries, complete-call budgets and explicit gaps; zero new target-ready candidates or GPU allocations. Existing owners execute directly, no approval handoff. Do not assign CUDA back to B4 or revive D32/reminder. [Latest findings](2026-11-02-general-hotspot-search-refill.md#p0-ws-and-p1-bf16-research-pickup-and-source-screens--november-2-2026-pacific) · [Evidence](evidence/2026-11-02-hotspot-refill/p0-p1-source-screen.json) |
| Public scout | Check public changes every two hours. The new cluster-projection reference needs source binding and a matched comparison. [Scout discovery](2026-11-02-scout-utterwqlnut-projection-0328.md) [Competition watch](2026-11-02-competition-watch.md) · [Source standing](2026-11-02-scout-source-standing-and-jobs.md) |

## Execution rules

The Decode32 Worker owns its checks and job waits. Each other owner retains its own jobs. Check live jobs and quota before dispatch. Record pickup, accepted job and measured result separately. Public submissions require user approval. [D32 directive](decode32/2026-11-02-02-end-to-end-ownership.md) · [Research and worker responsibilities](decode32/2026-11-02-04-research-plan-worker-execute.md) · [Project instructions](../../AGENTS.md)

## Latest audit

[Same-refill P0 WS/P1 C2 source screens](2026-11-02-general-hotspot-search-refill.md#p0-ws-and-p1-bf16-research-pickup-and-source-screens--november-2-2026-pacific): terminal-join SASS and exact C2 reconstruction checked; structural alternatives remain cost/feasibility gaps, not measured gains. No kernels or worker jobs changed.

[Driver monitor04:14UTC](2026-11-02-driver-monitor-0414.md): MoE whole-unit queue accepted; norm performance and identity routing/coverage closures; latest MoE/BF16 priorities and monitor reconciled.

[November 2 steering compaction](2026-11-02-steering-compaction.md) records the moves, retained paths and verification.

## Report history

[Projection source comparison](2026-11-02-utterwqlnut-projection-comparison.md) checks the shared cluster/WGMMA design. Our C2 has lower times in 27/30 archived cells; O32 is an unpaired component lead.

Use the [shared report archive](../../archive/docs/steering/) for older decisions. The [index before compaction](https://github.com/yelgabo/okc/blob/977fd319/docs/steering/README.md) retains the previous audit and report history with links at that revision. See the [prior archive manifest](../../archive/manifest-2026-11-02.json) for earlier moves.

[Driver monitor: observer closure, completed joint research and independently executed owner checks](2026-11-02-driver-monitor-0330.md). Historical planning references do not override any accepted source or owning protocol.

[CUDA environment-fidelity clarification](2026-11-02-cuda-part2-experiments.md#environment-fidelity-clarification): local SM90a builds did not replicate the remote H200 environment one to one. Compiler resource observations are local; GPU correctness and timings came from remote H200 checks.

[CUDA local-gate correction](2026-11-02-cuda-part2-experiments.md#correction-local-results-cannot-decide-target-performance) withdraws local-only spill/code-identity rejection as target-performance evidence. Unsafe flush prototypes remain blocked by source correctness; the pointer treatment is target-untested; both measured H200 failures retain their scope.
