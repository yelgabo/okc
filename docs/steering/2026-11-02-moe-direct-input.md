# Direct phase-A routed-input staging

November2,2026 Pacific. MOE-DIRECT-X-01 is the next owned B4/MAIN alternative after the complete pair-lookup loss. The public5df257af release remains5788.720720us; user target5559.8us remains open. General owns T1/T8/T16. No new GPU job yet.

## Work removed and competing cost

The selected c7c201ae engine first gathers routed input rows into globally materialized xA. Each activation producer waits on the job's cross-CTA gathered-row count and bulk-copies xA stages into its shared ring. This candidate eliminates global xA writes and their gathered-row publication. The activation producer's32lanes gather the current input directly into the existing shared stage in the exact KPERM3 descriptor layout. Phase B keeps its bulk copies and eight-contributor phase-A release waits. All weights, arithmetic, precision boundaries, stage ranges, fixups, WGMMA grouping and output finalization remain unchanged.

This is a whole work-elimination alternative to global gather plus TMA, rather than a decode adjustment. The prior queue loss, invalid sparse observer and pair-codebook loss justify testing a different organization. The original gx_pre/gx_st16 losses rearranged a retained global gather. They did not test eliminating xA or its cross-CTA readiness dependency. Their exact variants remain closed. The fixed direct-weight and shared-A losses change another operand and remain closed.

Direct staging adds repeated scattered input loads, producer-warp instructions and shared stores in place of a one-instruction bulk copy. It can lose even while removing a global intermediate. There is no current qualified exposed-gather-time estimate, so no numerical gain forecast is claimed. A complete matched endpoint is the distinguishing test. One fixed mapping groups8rows and4K groups across a warp, giving contiguous16byte shared stores. No mapping, cache or ring-depth sweep is registered.

## Ordering, layout and lifetime proof obligations

Each valid route r reads the current input token S.ent[S.jS[j]+r]>>3. Stage st selects256BF16 K values. K group G maps source32bit indices B(G)+32h for h=0..3, where B(G)=8*((G>>1)&3)+2*(G>>3)+(G&1). Destination uint4 index is((r>>3)*32+G)*8+(r&7), exactly the corresponding slice of the incumbent xA gather. Invalid padded rows store zero and read no input. The CPU witness must cover every K stage,1..64valid rows, all code-bit positions, repeated/reordered token IDs and zero-local routing.

Every shared writer executes fence.proxy.async.shared::cta. A dedicated32participant CTA barrier8 follows all writes/fences, then lane0 arrives once on the existing BX_F barrier. Selected named barriers1/2/7and consumer subgroup3..6do not use8. Every phase-A producer lane reaches8once per stage; phase B stays lane0-only and never enters it. Its initialization count remains1. The existing consumer acquire wait observes publication before WGMMA reads. This replaces the phase-A bulk transaction's expect_tx/completion with a synchronous shared-write publication, without an outstanding async copy. All512consumers still drain WGMMA, execute their proxy fences and arrive on BX_E before that ring slot can be overwritten. Slot/parity/sequence order and both producer roles remain unchanged.

The removed gather call's global stores and cross-CTA gathered-row release have no readers in the candidate. Its internal consumer barrier disappears with that work. The prior job-row prefix barrier and full producer/consumer WS_JOIN remain, publishing routing/ranges before stage issue. Phase-A-to-B inter stores, per-writer proxy fences, contribution counts/acquires and final completion barriers are unchanged. No consumer can prevent a resident producer from publishing its current slot. Empty work reaches the unchanged completion/reset path. Inputs remain owned and unchanged through the captured kernel; direct reads use the same live input tensor as the incumbent gather. Scratch allocation and graph state ownership remain unchanged.

This proof depends on per-writer cross-proxy ordering plus warp convergence before lane0 publication, not on tests alone. [PTX8.7 memory and mbarrier contract](https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html) and [CUDA12.8 shared-store publication example](https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUDA_C_Programming_Guide.pdf) supply the ordering rules. Source assertions and emitted operations must confirm the protocol. Independent review is unavailable under the user's no-new-agents instruction.

## Finite screen

Use the original54cases/three banks/normal and skewed routing, independent duplicate controls/candidates,378fresh observations,216captures,1152cold replays and54restores. Require exact selected source and7compiler requests, original numerical/zero/guard/ownership/lifetime checks, every-bank/allfour-cross complete-six gain at least1% and above twice duplicate-log noise, no affected1%regression and no unchanged-load drift. Preparation remains295s with bounded failed partial capsules and failure status retained. A successful screen needs held-out confirmation before integration or another public approval request.

Stop the exact direct-input implementation on source-policy, correctness, progress, lifetime, source attribution, coverage, budget, unchanged drift, regression or benefit failure. No blind retry, larger guard, favorable subgroup mask, mapping/cache/ring sweep or public action. Compiler spills alone are local resource evidence, not a target performance verdict. Verify exact remote/local source, fresh shared jobs/quota and record dispatch-started plus accepted own ID before measurement.

Local qualification retained the same protocol with an explicit named32participant producer barrier8. ptxas eliminated both intrinsic and constant-mask warp-sync instructions as redundant in preliminary local builds. That observation is not a compiler-correctness failure. The final named barrier makes this publication boundary explicit in emitted code. No GPU retry or mapping/cache/ring sweep was performed.

## Frozen implementation and local qualification

Native candidatec705df6aafdc6ddee86879675d4d2704f440e3a17599220d9224dd57a9b822ad versus controlc7c201ae. Private paired source1191e44cf87cacc36e89729a9e2f33d3c80f04361afe30062e022cd07009d537,350697bytes; runtimec89d5a8dcdf15f4911da75e27b416e38b13413c7ca6cfca468ff30e119e643e9. [Full source/flags/route manifest](evidence/2026-11-02-moe-direct-input/paired.manifest.json). Host preparation and packed weights are unchanged; original helperb8ed8543still owns T1/T8/T16.

Eight unit tests pass:2layout/source,4real Torch storage/orchestration/failure and2strict full/partial readers. The CPU native witness executes the actual helper's address/load/store statements across1040count/stage cases and4718592uint32values, including zero-local and padded rows. The injected row-swap executable returns1 as expected. CUDA synchronization is excluded from this CPU witness, and the Torch tests use an explicit CUDA facade with synthetic timing. They establish mapping and orchestration, not target correctness or performance.

Four local device entries compile under CUDA12.8.1 forSM90a, exit0 with full diagnostics retained. Both bool entries use96registers/16barriers. B4 control/candidate have8stack bytes/4spill stores/32spill loads; MAIN control/candidate havezero. All entries retain512static HGMMA and48full drains. In each candidate entry, emitted shared stores lead to MEMBAR.ALL.CTA, FENCE.VIEW.ASYNC.S, named BAR.SYNC.DEFER_BLOCKING8/count32 and then the lane0 readiness arrival. [Exact emitted sequences](evidence/2026-11-02-moe-direct-input/shared-publication.json), [resources](evidence/2026-11-02-moe-direct-input/resources.json), [full compile archive](evidence/2026-11-02-moe-direct-input/archive-manifest.json). Named barrier8 is separate from the memory-resident S.bars mbarrier objects. Local builds do not establish H200 codegen parity.

The frozen runtime verifies all7registered compiler requests, four independent states, original256word lookup prefixes, complete case/record coverage and unchanged-path drift. Bounded partial evidence cannot qualify even with a full favorable timing table attached to a failed setup. Preparation remains295s. The H200 research checker still reports the changed ruled-out pin; hashes were left unchanged and the affected current negatives were read. All claims are author-checked; independent review remains unavailable. No GPU outcome is known.

## Accepted private screen

Own94c57451574c4cd3 accepted for frozen1191e44c/350697bytes, publishedf1955498. Fresh shared list hadzero live jobs; MoE checks11left before this dispatch. Accepted source/job association is archived. Follow only this ID with one foreground waiter. No target result yet.

## Terminal decision, retain global gather

Own94c57451574c4cd3 passed5official tests/6benchmarks/6profiles, all54private cases/378observations/216captures/1152cold timings/54restores and cleanup. Exact7build/table-owner/source pins pass the strict reader. Prepare272.829260s; worst global/row error0.000135036/0.000206546 within unchanged0.015/0.075limits. Solewaiter38355ended0 before archive.

Performance fails0/6groups,24complete-six factors0.755195813–0.803710954 and all18affected regressionrows. T16/skewed/1102031also breaches unchanged-path drift. This drift limits precise attribution; it does not erase the large frozen rejection. Official candidate2 score7194.998534us/1.869145621x CI[1.841108436,1.897182805]. This is not a release lead or a reason to replace the public5788.720720us source.

| Tokens | Official adjusted us |
|---|---:|
|1|25.431312|
|16|96.453180|
|32|146.644775|
|8|64.959878|
|128|310.721120|
|256|407.752651|

[Owning receipt](evidence/2026-11-02-moe-direct-input/status-wait.raw.json), [strict36-row analysis](evidence/2026-11-02-moe-direct-input/analysis.json), [summary](evidence/2026-11-02-moe-direct-input/summary.json), [complete profiles](evidence/2026-11-02-moe-direct-input/profiles). Stop this exact implementation. No mapping/cache/ring sweep, retry, shape mask, integration or public action. The result does not isolate producer instruction, scattered traffic or synchronization cost, nor establish a family ceiling.

Next bounded source preparation is lossless factorized expert-output storage. Store the required BF16 down result and reconstruct the identical FP32 gamma product before the unchanged norm/combine operations. This is distinct from folding gamma into W2 or rounding the post-gamma product. Mathematical/store-layout/fallback proof and compiled-source qualification precede any new private check.
