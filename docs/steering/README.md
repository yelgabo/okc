# Shared project steering

## FINAL-DAY RULE OF LAW, November 2 late Pacific

[Final-day directive](2026-11-02-final-day-directive.md): short loads only (decode 1, verify 1, decode 16 where it falls out). Owner table and stop rules inside; supersedes the queues below for the remaining day.

## Top-100 gap reframe, November 2 late Pacific

[Gap reframe](2026-11-02-top100-gap-reframe.md): the geomean pays 4-8x more per microsecond at decode 1 / verify 1 than at verify 32, and those loads are nine-launch latency chains at 1.8x the byte floor. Proposes a persistent post-attention kernel for T <= 8 (hypothesis: the whole 6.8% layer gap), with a MoE T=1/T=8 timeline and a two-stage fusion microbench as the first decisive tests.

## Latest audit, November 2 23:08 Pacific

[Public scout: eight-token specialized decode](2026-11-02-scout-vllm-specialized-0724.md) pins vLLM PR64077 at31ebba98 and verifies42changed files. The author reports1.53x on GB300 against stock vLLM. SM100, BF16 KV, quantized expert activations and auxiliary streams prevent treating it as an H200 challenge gain. Source reference only; current short-load assignments remain intact.

[Repository audit](2026-11-02-2308-repo-audit.md): rank 389 of 1223 at 2.1261x, rank 50 at 2.3216x, top 50 by the deadline judged out of reach on measured deltas; main checkout merged with the G line; orphaned kernel work in one worktree; seven doc contradictions listed with line numbers.

## Worker repurpose: former Decode32 → standalone MoE P0; CUDA Optimizer → BF16 P1

## Latest component results and historical transfer audit

[18:54 Vancouver review](2026-11-02-1854-performance-review.md). Corrected scale/groups and current skinny NS4 pass correctness and fail promotion. NS3 and public G1b4 are retained. The historical source audit distinguishes retained wins from changed policies. D32 diagnostic5909739e8a4445b2 is accepted.

## Current performance evidence, November 2

[18:33 Vancouver review](2026-11-02-1833-performance-review.md). Both D32 matched attempts stopped in setup without timing. Constant WGS scales pass correctness and fail their matched performance gate. Corrected grouping and current skinny depth checks remain pending.

## Current architecture diagrams, November 2

[Incumbent layer and MoE diagrams with performance labels](../incumbent-architecture-2026-11-02.html). [In-layer activity labels and validation](2026-11-02-incumbent-inlayer-labels.md). The [initial diagram report](2026-11-02-incumbent-diagrams.md) is historical.

## Family F registration, historical policy

**[User-directed worker reassignment](2026-11-02-worker-repurpose-moe-bf16.md)** supersedes older owner rows below: **former Decode32 Worker takes P0 `muse_moe_w4a16` architecture/representation and whole-call performance**, **CUDA Optimizer takes P1 `muse_bf16_proj`**, and **General Optimizer retains separate MoE WS skinny work**. The former Decode32 worker must first safely archive/decide its already accepted D32-IDENTITY-01 job `4a2a79bdac04499a` with its sole waiter, if still unresolved; no abandonment or duplicate. CUDA Optimizer checks for its own active phase job before ceasing B4-phase ownership. Do not assign further D32 tasks automatically (D32 is P4); do not overlap the two MoE lanes. **The user has paused the former Decode32 Worker's self-reminder timer: leave it paused; do not re-enable or claim automatic pickup.** This is GitHub steering, not proof either running agent received the reassignment; no new public-submit authority.


## Current user priority: MoE/BF16 first; Decode32 P4

**Latest override** in [the standing hotspot goal](2026-11-02-incumbent-hotspot-goal.md#latest-user-priority-override--november-2-2026-pacific): **P0 standalone MoE W4A16**, **P1 standalone BF16 projections**, with **Decode32 explicitly P4**. Other workloads are prioritized by verified independent ROI; prefill is a separate large-gap research track. User-supplied historical top-100 thresholds imply 509.8µs (8.40%) MoE, 108.6µs (3.43%) BF16, 40.3µs (2.79%) attention, 681.0µs (6.40%) whole decode layer, and 113,612.1µs (35.68%) prefill. These standalone/layer gaps **cannot be summed as a causal forecast**. Research/optimizer task refill must prioritize MoE and BF16 before proposing fresh D32 work. **Existing D32-owned accepted jobs continue safely; no cancellation, duplicate runs, takeover, or public-submission change.** Older D32-P0 statements are superseded, not erased.


Read your owner's row, then its linked record. Check newer commits before acting on a dated job state. The scored whole-layer control is public607, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. [Incumbent source](../../muse_decode_layer/cand/g1b4_stage_release.py)

## Private results and public promotion

[Original23 private leads and the newly qualified release](2026-11-02-private-promotion-audit.md) are now indexed across all five problems. The recovered5,725.7us MoE file is a release-qualification lead; the5,613.6us MoE point is an external comparator. A treatment loss versus a newer control cannot erase that control's public-release gap. Before closing another report or choosing new tuning, run the required [promotion ledger check](../releases/promotion-queue.json) from AGENTS. CI checks new tracked receipts for a visible disposition. Public submission still requires the user's approval after a concrete release is prepared.

## Current work

| Owner | Assignment and next decision |
|---|---|
| **OKC MoE Worker → P0 whole-expert** | Public4aa/5df remains5788.720720us, target5559.8us. **Factorized7479372115c64702 terminal PASS5/6/6, lower private5698.508243us exposed**, but matched0/6groups/13regressions/two unchanged drifts against publicC7; stop exact storage-only experiment. The separately authorized single-arm release belongs to the final-day lead below. **MOE-PARAM-Y-01 parked, target-untested under final-day short-load directive. MOE-TINY-TIMELINE-01 picked up**, T1/T8 source instrumentation in progress, no new job. [Tiny timeline source and bounds](2026-11-02-moe-tiny-timeline.md) · [Latest measured decision](2026-11-02-moe-factorized-output.md#terminal-decision-lower-point-with-failed-matched-gate) · [Combined contract](2026-11-02-moe-prepared-gamma.md) · [Public result](2026-11-02-moe-control-release-qualification.md#public-result-verified) · [Ledger](moe-worker/STATUS.md) |
| **Final-day lead → factorY single-arm release** | Human **“yes, submit on confirmation”** at20a6cfa9 covers ONE factorized-output release derived from7479372115c64702 after the lead's own qualifying private check, without a second permission question. Retain the original matched0/6 loss alongside official5698.508243us. Not approval for combined MOE-PARAM-Y or large-load R&D; no duplicate source/check/public action by research or another owner. [Exact authorization](2026-11-02-final-day-directive.md#ledger) · [Owning evidence](2026-11-02-moe-factorized-output.md#terminal-decision-lower-point-with-failed-matched-gate) |
| **OKC CUDA Optimizer → P1 BF16** | **Final-day short-load assignment supersedes the M128/M256 register-A/raw-packed brief.** Target M1–32 O/router/unpooler-slice load-set-at-entry scheduling with unchanged codec; one complete candidate comparison covers all30cells. Directive gate: >=1.5us summed decode1 or verify1 saving, no other-cell loss beyond noise. Bind actual C2 dispatch, bytes and complete-call costs; no large-M build or inferred pickup/inactivity. [Binding owner table](2026-11-02-final-day-directive.md#owner-assignments-for-the-day) |
| **OKC General Optimizer → P0 MoE skinny/WS** | **Tiny timeline support picked up07:31UTC** under the final-day directive. General owns source-bound trace schema/coverage/budget tests only; MoE owner retains native integration and sole GPU dispatch. No new job. Row-owned7f10243c98e74746 is complete and stopped; no repeat. [Support record](2026-11-03-general-tiny-timeline-support.md) · [Row-owned closure](2026-11-02-general-ws-row-owned.md#terminal-decision-retain-original-fused-ws-stop-full-k-row-ownership) |
| **Research/steering** | **Superseding short-load scope actually read and picked up.** Broader General search stopped; no new brief outside the776f owner table. Support the existing Tiny/Tiny2 instrument; General's actual support pickup is separately recorded. Keep row-owned896b5350/7f102 terminal, phase6028 invalid and BF16 large-M brief superseded. FactorY's matched loss and the lead's narrow conditional release approval both stand. Same-refill reconciliation drafted; its append is NOT published because the connector blocked that file write. No kernel/job/public/D32/timer action. [Directive](2026-11-02-final-day-directive.md) · [Existing refill](2026-11-02-general-hotspot-search-refill.md) |
| Public scout | Check public changes every two hours. The new cluster-projection reference needs source binding and a matched comparison. [Scout discovery](2026-11-02-scout-utterwqlnut-projection-0328.md) [Competition watch](2026-11-02-competition-watch.md) · [Source standing](2026-11-02-scout-source-standing-and-jobs.md) |

## Execution rules

The Decode32 Worker owns its checks and job waits. Each other owner retains its own jobs. Check live jobs and quota before dispatch. Record pickup, accepted job and measured result separately. Public submissions require user approval. [D32 directive](decode32/2026-11-02-02-end-to-end-ownership.md) · [Research and worker responsibilities](decode32/2026-11-02-04-research-plan-worker-execute.md) · [Project instructions](../../AGENTS.md)

## Latest audit

[Post-split General row-ownership decision](2026-11-02-general-hotspot-search-refill.md#post-split-research-decision-full-k-output-row-ownership--november-2-2026-pacific) advances the same refill after the valid two-launch loss. One fixed full-call build; CPU byte-permutation/partition and archived SASS checks are not GPU correctness or a speedup. Other worker files/jobs remain independent.

[Driver monitor06:11UTC](2026-11-02-driver-monitor-0611.md): MoE public release verified at5788.7us/rank374; lookupc985 accepted; phase/WS split stopped;19promotion leads covered and9stale base-release relationships reconciled.

[Faster standalone MoE diagram](2026-11-02-moe-release-diagram.md) compares the 5762.1 us private release and 5788.7 us public snapshot with the older6069.6us entry. The retained c7 engine matches the earlier layer diagram. T32 dispatch, large-batch work assignment and eligible scale packing differ from the older public source. Internal component times remain unqualified.

[Bounded next decisions on the same P0/P1 refill](2026-11-02-general-hotspot-search-refill.md#next-decision-build-the-complete-discriminators--november-2-2026-pacific) replace the prior build HOLD with concrete WS two-node and BF16 raw/packed implementation briefs. Timing must decide complete endpoints; missing timing is not a prerequisite to implement. No new allocation or promotion claim.

[Driver monitor05:10UTC](2026-11-02-driver-monitor-0510.md): B queue terminal loss;23 faster private points have complete promotion dispositions; existing MoE profile/release tasks and next-stage General/BF16 research request reconciled.

[Internal MoE profile validation](2026-11-02-internal-moe-profile-validation.md) checks the existing sparse observer and unchanged controls. Five CPU tests pass, but a reproduced post-cleanup budget failure loses partial evidence. User-authorized request and finding delivered to the existing MoE worker. H200 attribution remains pending; no duplicate job or kernel change.

[Other-computer clarification](2026-11-02-internal-moe-profile-validation.md#other-computer-clarification) retains successful local RTX5070/SM120 GPU evidence after the user identifies Gretchie. The Mac probe was host-specific. Hopper sm90a compatibility still prevents treating that RTX as an unchanged-incumbent profiling host; live Gretchie access remains unverified here.

[Incumbent profile overlays](2026-11-02-incumbent-profile-overlays.md) add measured group costs and shares to the layer, WGS and phase diagrams. Internal phase costs remain unisolated; no new GPU measurement.

## Latest hourly report, November 2 17:53 Vancouver time

[Calibration, MID terminal and D32 attention confirmation](2026-11-02-1753-performance-review.md).

## Previous hourly report, November 2 16:50 Vancouver time

[Shared-A and direct-weight terminal losses; D32 priority](2026-11-02-1650-performance-review.md).

## Previous hourly report, November 2 15:43 Vancouver time

[Fixed K128 whole-layer results and next expert proposal](2026-11-02-1543-performance-review.md).

## Previous hourly report, November 2 12:48 Vancouver time

Two expert-engine proposals stopped at source review. The isolated attention continuity candidate is frozen before H200 testing. [Review and evidence](2026-11-02-1248-performance-review.md)

## Previous hourly report, November 2 11:53 Vancouver time

The one-token expert split passed correctness and lost its paired MoE gate. [Review and timings](2026-11-02-1153-performance-review.md)

## Previous hourly report, November 2 10:44 Vancouver time

G pre-gamma transport passed component correctness and missed the paired performance gate. [Review and timings](2026-11-02-1044-performance-review.md)

## Previous hourly report, November 2 09:46 Vancouver time

The expert component adapter passed CPU checks. Its paired harness and target measurements remain pending. [Review and evidence](2026-11-02-0946-performance-review.md)

## Previous hourly report, November 2 08:49 Vancouver time

The PR buffer check failed host compilation before timing. Two expert prototypes have offline evidence, and an attention continuity toggle stopped at source review. [Review and evidence](2026-11-02-0849-performance-review.md)

## Previous hourly report, November 2 07:49 Vancouver time

The graph-owned PR candidate has a conditional private allocation following host checks. [Review and qualification requirements](2026-11-02-0749-performance-review.md)

## Previous hourly report, November 2 06:47 Vancouver time

Short serial passed correctness but missed its confirmation thresholds. The last private check remains reserved. [Review and six-load timings](2026-11-02-0647-performance-review.md)

## Previous hourly report, November 2 05:28 Vancouver time

A fresh G1b4 control is queued. The serial short producer awaits its candidate run. The canonical timed SiLU receipt is available. [Review](2026-11-02-0528-performance-review.md)

## Previous hourly report, November 2 04:32 Vancouver time

The matched G short-counter treatment missed its gate. G224 and timed SiLU passed correctness and were parked. [Review and job ledger](2026-11-02-0432-performance-review.md)

## Previous hourly report, November 1 14:49 Vancouver time

G’s reset queue is held. The common control needs H200 qualification before the remaining short-counter hypothesis can be tested. [Review and evidence](2026-11-01-1449-performance-review.md)

## Previous hourly report, November 1 13:47 Vancouver time

The four-cell H200 job was accepted. Its result is unknown during service maintenance. Both standalone progress-repair forms failed the local spill gate. [Review and receipts](2026-11-01-1347-performance-review.md)

## Previous hourly report, November 1 12:45 Vancouver time

Stage-local XOR lost on all ten targets. The repaired verify hybrid fits the file limit and awaits H200 qualification. [November 1 review](2026-11-01-1245-performance-review.md)

## Previous hourly report, November 1 11:19 Vancouver time

The private VP2 comparator wins its target cell but misses the six-load gate. Global sparse XOR loses on all ten projection targets. The stage-local result is pending. [November 1 review](2026-11-01-1119-performance-review.md)

## Previous hourly report, November 1 08:04 MST

[Retained G layer and final measured outcomes](2026-11-01-0804-performance-review.md)
records remote snapshot `af3b550`. Family G retains2.126083x. Owned Lt-plus-boundary
has matched gains. New owned scheduling and projection variants fail gates, and detailed
wait observers fail causal-usability checks. The final handoff defines pending reset checks.

## Previous hourly report, November 1 01:16

**[2026-11-01 01:16 PDT: remote architecture evidence](2026-11-01-0116-performance-review.md)**

Remote snapshot `6e078b5`, checkout left at `d6337e1`. Ring87 and skinny wait gates fail;
masked grouped MoE loses. Late-tid2 attention is promoted standalone. Earlier PRQ layer
port has one win and an unresolved repeat. C3 router gains remain component-only.
Timeline routing counts are invalid and CUPTI access is denied. Prefill remains deferred.

## Previous hourly report, 23:32

**[2026-10-30 23:32 PDT: PBULK measured, mode 86 and fusion gates](2026-10-30-2332-performance-review.md)**

Snapshot `8f2c89b`. PBULK1 passes with a gain below its bar; PBULK2 loses. Mode86 misses
its performance gate. M-C C-1 loses locally before a bench fault. New layer score2.018705x
remains within prior uncertainty; changed validation and sensitivity findings are retained.
GTICKET review is complete, with corrected code awaiting benchmark validation.

## Previous hourly report, 22:35

[Same-refill P0 WS/P1 C2 source screens](2026-11-02-general-hotspot-search-refill.md#p0-ws-and-p1-bf16-research-pickup-and-source-screens--november-2-2026-pacific): terminal-join SASS and exact C2 reconstruction checked; structural alternatives remain cost/feasibility gaps, not measured gains. No kernels or worker jobs changed.

[Driver monitor04:14UTC](2026-11-02-driver-monitor-0414.md): MoE whole-unit queue accepted; norm performance and identity routing/coverage closures; latest MoE/BF16 priorities and monitor reconciled.

[November 2 steering compaction](2026-11-02-steering-compaction.md) records the moves, retained paths and verification.

## Report history

[Projection source comparison](2026-11-02-utterwqlnut-projection-comparison.md) checks the shared cluster/WGMMA design. Our C2 has lower times in27/30 archived cells; O32 is an unpaired component lead.

Use the [shared report archive](../../archive/docs/steering/) for older decisions. The [index before compaction](https://github.com/yelgabo/okc/blob/977fd319/docs/steering/README.md) retains the previous audit and report history with links at that revision. See the [prior archive manifest](../../archive/manifest-2026-11-02.json) for earlier moves.

[Driver monitor: observer closure, completed joint research and independently executed owner checks](2026-11-02-driver-monitor-0330.md). Historical planning references do not override any accepted source or owning protocol.

[CUDA environment-fidelity clarification](2026-11-02-cuda-part2-experiments.md#environment-fidelity-clarification): local SM90a builds did not replicate the remote H200 environment one to one. Compiler resource observations are local; GPU correctness and timings came from remote H200 checks.

[CUDA local-gate correction](2026-11-02-cuda-part2-experiments.md#correction-local-results-cannot-decide-target-performance) withdraws local-only spill/code-identity rejection as target-performance evidence. Unsafe flush prototypes remain blocked by source correctness; the pointer treatment is target-untested; both measured H200 failures retain their scope.

[Current-control public release](2026-11-02-moe-control-release-qualification.md#public-result-verified) is verified as4aa033ef23a142cd, exact5df257af,5788.720720us/rank373. The user approved this exact file. Continued B4/MAIN work is MOE-LUT-01; its target measurement is separate.
