# Shared project steering

## Worker repurpose: former Decode32 → standalone MoE P0; CUDA Optimizer → BF16 P1

**[User-directed worker reassignment](2026-11-02-worker-repurpose-moe-bf16.md)** supersedes older owner rows below: **former Decode32 Worker takes P0 `muse_moe_w4a16` architecture/representation and whole-call performance**, **CUDA Optimizer takes P1 `muse_bf16_proj`**, and **General Optimizer retains separate MoE WS skinny work**. The former Decode32 worker must first safely archive/decide its already accepted D32-IDENTITY-01 job `4a2a79bdac04499a` with its sole waiter, if still unresolved; no abandonment or duplicate. CUDA Optimizer checks for its own active phase job before ceasing B4-phase ownership. Do not assign further D32 tasks automatically (D32 is P4); do not overlap the two MoE lanes. **The user has paused the former Decode32 Worker's self-reminder timer: leave it paused; do not re-enable or claim automatic pickup.** This is GitHub steering, not proof either running agent received the reassignment; no new public-submit authority.


## Current user priority: MoE/BF16 first; Decode32 P4

**Latest override** in [the standing hotspot goal](2026-11-02-incumbent-hotspot-goal.md#latest-user-priority-override--november-2-2026-pacific): **P0 standalone MoE W4A16**, **P1 standalone BF16 projections**, with **Decode32 explicitly P4**. Other workloads are prioritized by verified independent ROI; prefill is a separate large-gap research track. User-supplied historical top-100 thresholds imply 509.8µs (8.40%) MoE, 108.6µs (3.43%) BF16, 40.3µs (2.79%) attention, 681.0µs (6.40%) whole decode layer, and 113,612.1µs (35.68%) prefill. These standalone/layer gaps **cannot be summed as a causal forecast**. Research/optimizer task refill must prioritize MoE and BF16 before proposing fresh D32 work. **Existing D32-owned accepted jobs continue safely; no cancellation, duplicate runs, takeover, or public-submission change.** Older D32-P0 statements are superseded, not erased.


Read your owner's row, then its linked record. Check newer commits before acting on a dated job state. The scored whole-layer control is public607, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. [Incumbent source](../../muse_decode_layer/cand/g1b4_stage_release.py)

## Current work

| Owner | Assignment and next decision |
|---|---|
| **OKC MoE Worker → P0 whole-expert** | **Picked up.** D32 identity job4a2a79bdac04499a is terminal, strict routing/coverage failed, complete receipt archived. MOE-BQUEUE-01 implements complete-unit dynamic B claiming against selected c7c201ae, B4/MAIN only. Four CUDA entries pass96regs/zero stack/spills; three source and two CPU host tests pass. Frozen paired4c8c470f, native99a6888d; private e044121f822d47b1 accepted after no-live/16-check preflight. General retains WS skinny. Reminder stays paused. [Source, proof, gates and state](2026-11-02-moe-worker-queue-experiment.md) |
| **OKC CUDA Optimizer → P1 BF16** | **Queued, pickup unconfirmed.** Shift to standalone BF16 projection, first checking any still-accepted own B4 job; past B4 phase-attribution is **historical**, not its current assignment. Compare current selected C2 paths and 30 scoring cells against losses/competitors; prioritize a source-backed candidate capable of material full-score effect. [Handoff](2026-11-02-worker-repurpose-moe-bf16.md) · [Newer external code comparison](2026-11-02-utterwqlnut-projection-comparison.md) |
| **OKC General Optimizer → P0 MoE skinny/WS** | GOPT-WS-03 `b38a9ed6dc9743e9` **terminal loss** (all 6 matched groups fail). Retain NS3/original norm. **No fresh qualified immediate kernel candidate verified.** Distinct WS T1/T8/T16 lane; do not overlap former D32 worker's whole-expert/representation lane. Existing research-refill request remains pending, not a second task. [Evidence](2026-11-02-general-ws-norm-aggregation.md#terminal-decision-retain-baseline-stop-exact-aggregation) · [Refill](2026-11-02-general-hotspot-search-refill.md#general-norm-aggregation-terminal-and-moe-first-refill) |
| **Research/steering** | Prioritize P0 **MoE whole-expert and WS skinny as nonoverlapping lanes**, then P1 BF16. Existing joint widened-search request remains pending; do **not** reassign CUDA to B4 or repurpose the paused D32 timer. No source-qualified new experiments since the norm failure. [Current request](2026-11-02-general-hotspot-search-refill.md#general-norm-aggregation-terminal-and-moe-first-refill) · [Standing goal](2026-11-02-incumbent-hotspot-goal.md) |
| Public scout | Check public changes every two hours. The new cluster-projection reference needs source binding and a matched comparison. [Scout discovery](2026-11-02-scout-utterwqlnut-projection-0328.md) [Competition watch](2026-11-02-competition-watch.md) · [Source standing](2026-11-02-scout-source-standing-and-jobs.md) |

## Execution rules

The Decode32 Worker owns its checks and job waits. Each other owner retains its own jobs. Check live jobs and quota before dispatch. Record pickup, accepted job and measured result separately. Public submissions require user approval. [D32 directive](decode32/2026-11-02-02-end-to-end-ownership.md) · [Research and worker responsibilities](decode32/2026-11-02-04-research-plan-worker-execute.md) · [Project instructions](../../AGENTS.md)

## Latest audit

[November 2 steering compaction](2026-11-02-steering-compaction.md) records the moves, retained paths and verification.

## Report history

[Projection source comparison](2026-11-02-utterwqlnut-projection-comparison.md) checks the shared cluster/WGMMA design. Our C2 has lower times in 27/30 archived cells; O32 is an unpaired component lead.

Use the [shared report archive](../../archive/docs/steering/) for older decisions. The [index before compaction](https://github.com/yelgabo/okc/blob/977fd319/docs/steering/README.md) retains the previous audit and report history with links at that revision. See the [prior archive manifest](../../archive/manifest-2026-11-02.json) for earlier moves.

[Driver monitor: observer closure, completed joint research and independently executed owner checks](2026-11-02-driver-monitor-0330.md). Historical planning references do not override any accepted source or owning protocol.

[CUDA environment-fidelity clarification](2026-11-02-cuda-part2-experiments.md#environment-fidelity-clarification): local SM90a builds did not replicate the remote H200 environment one to one. Compiler resource observations are local; GPU correctness and timings came from remote H200 checks.
