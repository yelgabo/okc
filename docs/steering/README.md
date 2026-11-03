# Shared project steering

## Worker repurpose: former Decode32 → standalone MoE P0; CUDA Optimizer → BF16 P1

**[User-directed worker reassignment](2026-11-02-worker-repurpose-moe-bf16.md)** supersedes older owner rows below: **former Decode32 Worker takes P0 `muse_moe_w4a16` architecture/representation and whole-call performance**, **CUDA Optimizer takes P1 `muse_bf16_proj`**, and **General Optimizer retains separate MoE WS skinny work**. The former Decode32 worker must first safely archive/decide its already accepted D32-IDENTITY-01 job `4a2a79bdac04499a` with its sole waiter, if still unresolved; no abandonment or duplicate. CUDA Optimizer checks for its own active phase job before ceasing B4-phase ownership. Do not assign further D32 tasks automatically (D32 is P4); do not overlap the two MoE lanes. **The user has paused the former Decode32 Worker's self-reminder timer: leave it paused; do not re-enable or claim automatic pickup.** This is GitHub steering, not proof either running agent received the reassignment; no new public-submit authority.


## Current user priority: MoE/BF16 first; Decode32 P4

**Latest override** in [the standing hotspot goal](2026-11-02-incumbent-hotspot-goal.md#latest-user-priority-override--november-2-2026-pacific): **P0 standalone MoE W4A16**, **P1 standalone BF16 projections**, with **Decode32 explicitly P4**. Other workloads are prioritized by verified independent ROI; prefill is a separate large-gap research track. User-supplied historical top-100 thresholds imply 509.8µs (8.40%) MoE, 108.6µs (3.43%) BF16, 40.3µs (2.79%) attention, 681.0µs (6.40%) whole decode layer, and 113,612.1µs (35.68%) prefill. These standalone/layer gaps **cannot be summed as a causal forecast**. Research/optimizer task refill must prioritize MoE and BF16 before proposing fresh D32 work. **Existing D32-owned accepted jobs continue safely; no cancellation, duplicate runs, takeover, or public-submission change.** Older D32-P0 statements are superseded, not erased.


Read your owner's row, then its linked record. Check newer commits before acting on a dated job state. The scored whole-layer control is public607, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. [Incumbent source](../../muse_decode_layer/cand/g1b4_stage_release.py)

## Private results and public promotion

[All23 better private score receipts and release dispositions](2026-11-02-private-promotion-audit.md) are now indexed across all five problems. The recovered5,725.7us MoE file is a release-qualification lead; the5,613.6us MoE point is an external comparator. A treatment loss versus a newer control cannot erase that control's public-release gap. Before closing another report or choosing new tuning, run the required [promotion ledger check](../releases/promotion-queue.json) from AGENTS. CI checks new tracked receipts for a visible disposition. Public submission still requires the user's approval after a concrete release is prepared.

## Current work

| Owner | Assignment and next decision |
|---|---|
| **OKC MoE Worker → P0 whole-expert** | **MOE-RELEASE-01 private64f9998ef238476f accepted**, exact5df257af, published9edf1ef6. Current-control standalone release, target outcome pending; public approval remains required. [File and gates](2026-11-02-moe-control-release-qualification.md). Phase6028cae5 terminal failed serialization/global stability, complete partial archived; no retry or qualified attribution. BQUEUE remains stopped. [Ledger](moe-worker/STATUS.md) |
| **OKC CUDA Optimizer → P1 BF16** | **Bounded build brief available; pickup unconfirmed.** Four-hour engineering box for one fixed raw register-A kernel and exact packed twin versus actual C2, pooler/unpooler M128/M256. Fixed geometry, fragment-level raw fallback, FP32 partial reduction; no timing-win prerequisite to build and no ring/tile sweep. Four affected cells, all30 coverage; no measured gain or new GPU allocation. [Executable owner brief](2026-11-02-general-hotspot-search-refill.md#bf16-fixed-raw-register-a-skeleton-and-exact-packed-twin) · [Handoff](2026-11-02-worker-repurpose-moe-bf16.md) |
| **OKC General Optimizer → P0 MoE skinny/WS** | **GOPT-WS-SPLIT-04 picked up05:33UTC.** Two-hour bounded implementation of unchanged WS A+B plus one norm/combine/reset finalizer. NS3/original norm fixed; no accepted GPU job. Next: progress/reset/map proof, selected builds and host qualification, then one complete paired endpoint if qualified. [Owned report](2026-11-02-general-ws-split-finalizer.md). |
| **Research/steering** | **Same refill advanced from screens to concrete build decisions.** 8e276c89 appends only the next-decision brief; source-correct implementation precedes net-speed qualification. Fixed WS split and BF16 raw/packed endpoints have timeboxes, score coverage, competing costs and finite stops. No kernel/job/public/reminder changes or worker acknowledgment claimed. MoE observer and c7/recovered release qualification remain separately owned. [Next decision](2026-11-02-general-hotspot-search-refill.md#next-decision-build-the-complete-discriminators--november-2-2026-pacific) |
| Public scout | Check public changes every two hours. The new cluster-projection reference needs source binding and a matched comparison. [Scout discovery](2026-11-02-scout-utterwqlnut-projection-0328.md) [Competition watch](2026-11-02-competition-watch.md) · [Source standing](2026-11-02-scout-source-standing-and-jobs.md) |

## Execution rules

The Decode32 Worker owns its checks and job waits. Each other owner retains its own jobs. Check live jobs and quota before dispatch. Record pickup, accepted job and measured result separately. Public submissions require user approval. [D32 directive](decode32/2026-11-02-02-end-to-end-ownership.md) · [Research and worker responsibilities](decode32/2026-11-02-04-research-plan-worker-execute.md) · [Project instructions](../../AGENTS.md)

## Latest audit

[Bounded next decisions on the same P0/P1 refill](2026-11-02-general-hotspot-search-refill.md#next-decision-build-the-complete-discriminators--november-2-2026-pacific) replace the prior build HOLD with concrete WS two-node and BF16 raw/packed implementation briefs. Timing must decide complete endpoints; missing timing is not a prerequisite to implement. No new allocation or promotion claim.

[Driver monitor05:10UTC](2026-11-02-driver-monitor-0510.md): B queue terminal loss;23 faster private points have complete promotion dispositions; existing MoE profile/release tasks and next-stage General/BF16 research request reconciled.

[Internal MoE profile validation](2026-11-02-internal-moe-profile-validation.md) checks the existing sparse observer and unchanged controls. Five CPU tests pass, but a reproduced post-cleanup budget failure loses partial evidence. User-authorized request and finding delivered to the existing MoE worker. H200 attribution remains pending; no duplicate job or kernel change.

[Other-computer clarification](2026-11-02-internal-moe-profile-validation.md#other-computer-clarification) retains successful local RTX5070/SM120 GPU evidence after the user identifies Gretchie. The Mac probe was host-specific. Hopper sm90a compatibility still prevents treating that RTX as an unchanged-incumbent profiling host; live Gretchie access remains unverified here.

[Incumbent profile overlays](2026-11-02-incumbent-profile-overlays.md) add measured group costs and shares to the layer, WGS and phase diagrams. Internal phase costs remain unisolated; no new GPU measurement.

[Same-refill P0 WS/P1 C2 source screens](2026-11-02-general-hotspot-search-refill.md#p0-ws-and-p1-bf16-research-pickup-and-source-screens--november-2-2026-pacific): terminal-join SASS and exact C2 reconstruction checked; structural alternatives remain cost/feasibility gaps, not measured gains. No kernels or worker jobs changed.

[Driver monitor04:14UTC](2026-11-02-driver-monitor-0414.md): MoE whole-unit queue accepted; norm performance and identity routing/coverage closures; latest MoE/BF16 priorities and monitor reconciled.

[November 2 steering compaction](2026-11-02-steering-compaction.md) records the moves, retained paths and verification.

## Report history

[Projection source comparison](2026-11-02-utterwqlnut-projection-comparison.md) checks the shared cluster/WGMMA design. Our C2 has lower times in 27/30 archived cells; O32 is an unpaired component lead.

Use the [shared report archive](../../archive/docs/steering/) for older decisions. The [index before compaction](https://github.com/yelgabo/okc/blob/977fd319/docs/steering/README.md) retains the previous audit and report history with links at that revision. See the [prior archive manifest](../../archive/manifest-2026-11-02.json) for earlier moves.

[Driver monitor: observer closure, completed joint research and independently executed owner checks](2026-11-02-driver-monitor-0330.md). Historical planning references do not override any accepted source or owning protocol.

[CUDA environment-fidelity clarification](2026-11-02-cuda-part2-experiments.md#environment-fidelity-clarification): local SM90a builds did not replicate the remote H200 environment one to one. Compiler resource observations are local; GPU correctness and timings came from remote H200 checks.

[CUDA local-gate correction](2026-11-02-cuda-part2-experiments.md#correction-local-results-cannot-decide-target-performance) withdraws local-only spill/code-identity rejection as target-performance evidence. Unsafe flush prototypes remain blocked by source correctness; the pointer treatment is target-untested; both measured H200 failures retain their scope.

[Concrete current-control release candidate](2026-11-02-moe-control-release-qualification.md) is prepared as5df257af, preserving the five actual compiler calls. Exact-file private qualification follows the existing phase job closure; no public approval or timing claim yet.
