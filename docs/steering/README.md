# Shared project steering

## Worker repurpose: former Decode32 → standalone MoE P0; CUDA Optimizer → BF16 P1

**[User-directed worker reassignment](2026-11-02-worker-repurpose-moe-bf16.md)** supersedes older owner rows below: **former Decode32 Worker takes P0 `muse_moe_w4a16` architecture/representation and whole-call performance**, **CUDA Optimizer takes P1 `muse_bf16_proj`**, and **General Optimizer retains separate MoE WS skinny work**. The former Decode32 worker must first safely archive/decide its already accepted D32-IDENTITY-01 job `4a2a79bdac04499a` with its sole waiter, if still unresolved; no abandonment or duplicate. CUDA Optimizer checks for its own active phase job before ceasing B4-phase ownership. Do not assign further D32 tasks automatically (D32 is P4); do not overlap the two MoE lanes. **The user has paused the former Decode32 Worker's self-reminder timer: leave it paused; do not re-enable or claim automatic pickup.** This is GitHub steering, not proof either running agent received the reassignment; no new public-submit authority.


## Current user priority: MoE/BF16 first; Decode32 P4

**Latest override** in [the standing hotspot goal](2026-11-02-incumbent-hotspot-goal.md#latest-user-priority-override--november-2-2026-pacific): **P0 standalone MoE W4A16**, **P1 standalone BF16 projections**, with **Decode32 explicitly P4**. Other workloads are prioritized by verified independent ROI; prefill is a separate large-gap research track. User-supplied historical top-100 thresholds imply 509.8µs (8.40%) MoE, 108.6µs (3.43%) BF16, 40.3µs (2.79%) attention, 681.0µs (6.40%) whole decode layer, and 113,612.1µs (35.68%) prefill. These standalone/layer gaps **cannot be summed as a causal forecast**. Research/optimizer task refill must prioritize MoE and BF16 before proposing fresh D32 work. **Existing D32-owned accepted jobs continue safely; no cancellation, duplicate runs, takeover, or public-submission change.** Older D32-P0 statements are superseded, not erased.


Read your owner's row, then its linked record. Check newer commits before acting on a dated job state. The scored whole-layer control is public607, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. [Incumbent source](../../muse_decode_layer/cand/g1b4_stage_release.py)

## Private results and public promotion

[Original23 private leads and the newly qualified release](2026-11-02-private-promotion-audit.md) are now indexed across all five problems. The recovered5,725.7us MoE file is a release-qualification lead; the5,613.6us MoE point is an external comparator. A treatment loss versus a newer control cannot erase that control's public-release gap. Before closing another report or choosing new tuning, run the required [promotion ledger check](../releases/promotion-queue.json) from AGENTS. CI checks new tracked receipts for a visible disposition. Public submission still requires the user's approval after a concrete release is prepared.

## Current work

| Owner | Assignment and next decision |
|---|---|
| **OKC MoE Worker → P0 whole-expert** | Public4aa033ef23a142cd/5df257af remains5788.720720us, user target5559.8us. **MOE-LUT-01c985 terminal correctness pass/performance loss**,0/6groups,18affected regressions, official12337.764699us. Stop exact LUT. **MOE-DIRECT-X-01 private94c57451574c4cd3 accepted**, paired1191e44c/nativec705df6a publishedf1955498, eliminates phase-A global input gather. Eight CPU tests/four local builds pass; target result pending. [Next experiment](2026-11-02-moe-direct-input.md). [Lookup closure](2026-11-02-moe-pair-lookup.md#terminal-decision-stop-exact-pair-lookup) · [Public result](2026-11-02-moe-control-release-qualification.md#public-result-verified) · [Ledger](moe-worker/STATUS.md) |
| **OKC CUDA Optimizer → P1 BF16** | **Bounded build brief available; pickup unconfirmed.** Four-hour engineering box for one fixed raw register-A kernel and exact packed twin versus actual C2, pooler/unpooler M128/M256. Fixed geometry, fragment-level raw fallback, FP32 partial reduction; no timing-win prerequisite to build and no ring/tile sweep. Four affected cells, all30 coverage; no measured gain or new GPU allocation. [Executable owner brief](2026-11-02-general-hotspot-search-refill.md#bf16-fixed-raw-register-a-skeleton-and-exact-packed-twin) · [Handoff](2026-11-02-worker-repurpose-moe-bf16.md) |
| **OKC General Optimizer → P0 MoE skinny/WS** | **GOPT-WS-OWN-05 picked up 06:39UTC**, four-hour build box through10:39UTC. Implementing the fixed T8/T16 full-K row-owned fused design against standalone5df257af. No job or gain yet; original fused WS/NS3 stays production. [Owned progress](2026-11-02-general-ws-row-owned.md) · [Fixed brief](2026-11-02-general-hotspot-search-refill.md#post-split-research-decision-full-k-output-row-ownership--november-2-2026-pacific) |
| **Research/steering** | **Outcome-driven same-refill pass completed**,3ac45602. One General ownership reorganization, not boundary/helper tuning: explicit unchanged weight/MMA counts, added activation costs,2/6score coverage and finite complete-endpoint test. No pre-build timing-win prerequisite. BF16 brief retained without claiming pickup; newer whole-expert LUT/direct-input records preserved independently. No kernel, job, public, D32 or reminder action. [Decision](2026-11-02-general-hotspot-search-refill.md#post-split-research-decision-full-k-output-row-ownership--november-2-2026-pacific) · [CPU model and binary evidence](evidence/2026-11-02-hotspot-refill/ws-row-owned-evidence.json) |
| Public scout | Check public changes every two hours. The new cluster-projection reference needs source binding and a matched comparison. [Scout discovery](2026-11-02-scout-utterwqlnut-projection-0328.md) [Competition watch](2026-11-02-competition-watch.md) · [Source standing](2026-11-02-scout-source-standing-and-jobs.md) |

## Execution rules

The Decode32 Worker owns its checks and job waits. Each other owner retains its own jobs. Check live jobs and quota before dispatch. Record pickup, accepted job and measured result separately. Public submissions require user approval. [D32 directive](decode32/2026-11-02-02-end-to-end-ownership.md) · [Research and worker responsibilities](decode32/2026-11-02-04-research-plan-worker-execute.md) · [Project instructions](../../AGENTS.md)

## Latest audit

[Post-split General row-ownership decision](2026-11-02-general-hotspot-search-refill.md#post-split-research-decision-full-k-output-row-ownership--november-2-2026-pacific) advances the same refill after the valid two-launch loss. One fixed full-call build; CPU byte-permutation/partition and archived SASS checks are not GPU correctness or a speedup. Other worker files/jobs remain independent.

[Driver monitor06:11UTC](2026-11-02-driver-monitor-0611.md): MoE public release verified at5788.7us/rank374; lookupc985 accepted; phase/WS split stopped;19promotion leads covered and9stale base-release relationships reconciled.

[Faster standalone MoE diagram](2026-11-02-moe-release-diagram.md) compares the 5762.1 us private release and 5788.7 us public snapshot with the older 6069.6 us entry. The retained c7 engine matches the earlier layer diagram. T32 dispatch, large-batch work assignment and eligible scale packing differ from the older public source. Internal component times remain unqualified.

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

[Current-control public release](2026-11-02-moe-control-release-qualification.md#public-result-verified) is verified as4aa033ef23a142cd, exact5df257af,5788.720720us/rank373. The user approved this exact file. Continued B4/MAIN work is MOE-LUT-01; its target measurement is separate.
