# Distributed A-ready publication for Tiny and Tiny2

November3,2026 Pacific. MOE-SHORT-READY-01 e3e2495464f34b2c is terminal. Tested correctness and coverage pass; the registered performance gate fails. Independent source reviews also identify an unresolved cross-lane publication proof. Retain public control5df/C7. This candidate is stopped, unintegrated and ineligible for release; no own GPU job remains active.

## Evidence and mechanism

The completed source-bound observer106a05c5 passes all correctness, replay, ownership and bounded cost checks. At T1 its per-CTA activation-ready waits have median4.60us/max5.95us; retained CTA2 has last A compute-unit retirement16.22us, helper retirement20.51us, and first B activation wait16.76–21.18us. This is a measured wait across helper completion/publication. It does not identify an atomic-contention cause. T8 has median2.43us/max4.38us. The source already prefetches phase-B weights, so adding generic cross-phase prefetch duplicates current work.

Current helper publication uses one contended atomic sum per touched job after all of a CTA's A epilogues. B compute warps poll that job total. Replace the total with one completion flag per CTA. After the original writer fences and warp synchronization, helper lane0 publishes its private flag. For each B job, all32lanes acquire the flags of exactly its contributing A CTAs, then retain the existing warp synchronization before reading activations. This reorganizes the publication dependency without changing unit work, epilogues, arithmetic, ring depth or the final grid barrier.

A different option is to move late A epilogues to compute warps, similar to current late B epilogues. That changes partial-buffer ownership and adds publication joins or atomics. Choose flags first because it preserves all arithmetic and producer scheduling and directly tests whether the existing aggregation costs matter. Expected benefit is a hypothesis, bounded by part of the observed4.6us T1 wait. The main uncertainty is that helper math or store-drain latency, rather than aggregation, may dominate. One matched comparison resolves this design; no ingredient must win separately.

The earlier short-RMW c5e4f89672f743bb changes the final global-barrier counter's memory ordering and failed its complete gate. This candidate preserves that final counter exactly and changes the separate phase-A readiness representation. It is not a retry of that treatment, NS4, row-owned full-K, WGS prefix, queue or scale/group work. See the archived short-RMW source/terminal reports.

## Memory, coverage and progress obligations

Use existing reserved int32 words p.doneA[128+cta]. JMAX is below128 and128+136 is below512; the next xready region begins at512 and the second sum buffer begins at1024. Static assertions enforce these boundaries. No new scratch allocation or host argument is needed. Every flag has one writer and resets after the original global B-completion barrier. Initial scratch is zero, and same-stream completion orders reset before the next graph replay. Other paths keep their original source and storage behavior.

For job j let its A unit interval be[L,R), total A units T and grid G. CTA c owns[floor(T*c/G),floor(T*(c+1)/G)). Its intersecting CTAs are first=ceil(G*(L+1)/T)-1 through last=ceil(G*R/T)-1 inclusive. All products fit32bits under G<=136 and32jobs. Positive contributions partition exactly the job, so the old sum reaches RB_A if and only if all those CTA flags are ready. This preserves the existing coarse publication point, including a CTA spanning jobs. Empty routing executes no B units or division.

The selected producer retains each helper lane's original GPU fence, then the warp join, then lane0's aligned st.relaxed.gpu.global.u32. For lane0's own preceding stores, the fence and strong relaxed store form a release pattern. The join relays the other lanes' stores after lane0's fence. The argument does not establish that those stores precede the first operation of lane0's release pattern. Whole-warp publication therefore remains unproved. Consumer GPU acquire reads, votes and the retained __syncwarp before generic inter reads do not supply this missing producer edge. No writer or generic-to-async proxy fence was removed. See the [independent General audit](2026-11-03-general-ready-flag-audit.md#publication-proof-gap) and [research cross-check](2026-11-03-research-ready-proof-crosscheck.md#publication-result-the-written-proof-is-not-closed). Neither review demonstrates an H200 stale read or a complete allowed PTX execution; passing tests and the predecessor's related publication shape cannot close the proof.

A producers never wait for B. Each resident CTA publishes before its helper reaches B. Consumer waits depend only on prior A work, and original cooperative admission remains. Reset occurs only after every compute warp and helper passed the existing B join and full-grid barrier. Preserve the original final publication/acquire, generation advance and sum-buffer parity. CPU schedules, compiler lowering and GPU tests establish separate facts; tests alone do not prove ordering.

## Qualification and finite decision

Before dispatch, pin the actual selected Tiny/Tiny2 helper and unchanged flags; reverse the patch exactly; enumerate all job/grid interval mappings and randomized producer-completion schedules; inject a missing contributor; test empty/occupied graph transitions, guard/reset/ownership and partial-failure retention. Compile both selected variants and inspect the explicit strong GPU flag stores, acquire loads, vote, retained fences/joins and unchanged final counter. Local compilation is not target parity.

One private matched screen with the original three normal/skewed fresh banks, four independent owners, all54cases/36timed rows/378observations/216captures/1152cold replays/54restores. Candidate affects T1/T8 and same selected Tiny fallback sizes; other four scored sources remain original. Official candidate2. Use tools/moe_short_gate.py: one same short cell must save at least2us across every bank/distribution and all four owner crosses, beat twice duplicate noise, preserve other four loads within noise and have no affected regression. Whole-six benefit must also resolve above duplicate noise. Retain all numerical/cache/graph/ordering/source guards,295second preparation and failed partial evidence.

Stop exact design on any correctness, policy, source, coverage, budget, drift, regression or benefit failure. No flag-layout/lane/epoch/granularity sweep, larger guard, fence removal, selected-mask rescue or blind retry. A successful screen needs held-out confirmation, exact release qualification and the applicable public approval. No new public authorization follows from the previously approved5df file or the lead's separate factor-Y approval.

## Frozen candidate and local qualification

Pairedb6107a783c8ded6dc626f6cfa99e7e1b53201e5225bd2073f736c69125f16964/490593bytes, native helper974d7743bb8d049150fe46b62f84c2308158e382588903f185d8155c787faf20, runtime8e2dd47c28fb0f82c6261bef869185b3b93781b171eeb8e06938712200c588c8. Five CPU integration/source/mapping tests pass, plus two short-gate arithmetic tests. The final targeted two-test pass checks the delivered tracked receipt fixture, positive owning-reader association, wrong job, a missing reset observation and failed complete evidence with otherwise favorable performance rows. Full host orchestration covers54cases/378observations/216captures/1152synthetic timings/54restores and1512flag-zero checks. These tests use real CPU Torch storage and explicit CUDA/native facades; synthetic timings are not a GPU gain.

The actual native wait function also passes24CPU warp-model cases with32lanes, withheld publication, visibility after the modelled warp join and exact contributor sets. A last-contributor omission fails as expected. The CPU producer uses C++ release stores, a stronger publication operation than the selected relaxed PTX store. This exercises wait/address/convergence and contributor coverage under that stronger model. It neither executes PTX nor proves the native cross-lane publication edge.

Both actual Tiny/Tiny2 variants compile with CUDA12.8.1 SM90a locally,154registers/3barriers/zero stack or spills. Original controls use145registers and are unchanged from the timeline builds. Both entries lower the new publication to STG.E.STRONG.GPU, retain explicit acquire loads/votes, all8proxy-fence sites and both original final acq_rel RMW sites. Register counts, source coverage, memory-model reasoning and measured H200 speed remain separate facts. Full compiler diagnostics and source/adaptor pins are archived; full SASS stays ignored under .build per the repository audit.

These local checks preceded dispatch and were not an independent native GPU review. The terminal target result is below. Subsequent independent General and research source reviews validate contributor mapping/reset placement and identify the publication proof gap. Neither review certifies the candidate. The recorded dispatch preflight had3MoE private checks left; that historical quota is not a current dispatch authorization.

## Reconciled concurrent short-load evidence

Origin/main fa762950 includes another owner's completed Tiny timeline19128cda79ca4e25 and closed NW16+balanced ab54144f6c4d4727 and BAL-only b45cce36b47f4e81. No poll or retry of those jobs follows. This candidate retains NW8, the original proportional A/B ranges, original ring and code/scale packing. Its ready-publication source and owned files are distinct; the other lane is recorded closed. Preserve its losses rather than redispatching them.

The other timeline reports about1.7–1.8us ready waiting on its judge-input observation, less than this instrument's cold normal-bank4.60/2.43us medians. Inputs, replay/cache conditions and instrumentation differ; they are not a matched comparison. The measured wait is not a promised2us saving. This reinforces the need for the frozen multi-bank control comparison and separate official score. No causal fence-versus-aggregation conclusion is drawn from either timeline alone.

## Accepted private check

Own e3e2495464f34b2c accepted once for pairedb6107a78/490593bytes, published73ab76cf/reconciled8fad06be, dispatch intent9e778ae6. Fresh quota had3checks left; shared live jobs were recorded before the POST. Accepted ID persisted before collection or other measurement. The sole own-ID foreground waiter wrote ignored .build/moe-short-ready/status-wait.raw.json and has ended with exit0. The complete terminal archive is below.

## Conditional follow-through prepared during the screen

Tools/moe_short_ready_confirm.py fixes held-out banks2203011/2203017/2203029, disjoint from the screen and its fresh97offset banks. Its two CPU tests establish that only BANKS and runtime identity change, the exact helper/compiler requests remain, and the original screen/gate module stays immutable. No seed search or alternate held-out bank is allowed. No confirmation is dispatched unless the complete frozen screen passes.

Tools/moe_short_ready_release.py prepares a single-arm draft from exact approved5df257af, preserving the original MID/B4/MAIN, quantized weights, packing, scale fallbacks, operator checks and three-buffer ABI. Both changed short compiler requests equal those of the active paired candidate; the other three equal the original public file. Two CPU tests check contract identity and actual preparation/forwarding. This source is not GPU-qualified or public-approved. Conditional manifests/test output are archived; generated draft payloads stay ignored until their gates permit use.

## Terminal decision: stop exact flags and retain the incumbent

Own e3e2495464f34b2c is complete. PASS5tests/6benchmarks/6profiles, no flags, return0. Strict source/job/manifest reader accepts full54cases/36timed rows,378observations/216captures/1152cold replays/54restores, all seven compiler requests and1512zeroed flag observations. Prepare252.570135seconds; worst global/row paired errors0.000085418680/0.000300182204. Collector95504 ended0 before archival. This establishes the tested correctness and coverage. The cross-lane publication proof remains incomplete, independently of those passing tests and the failed speed gate.

The frozen performance gate fails. No short cell qualifies;0/6bank/distribution groups resolve a whole-six gain,11affected rows breach the regression gate and one unchanged T128/normal bank1102037 row breaches its drift envelope. Across all24owner-cross/bank/distribution comparisons, median T1 savings are negative1.079999us and T8 negative0.592000us. Thus the candidate is slower at both target medians. Unchanged-path drift limits precise attribution; it does not repair the failed benefit gate.

| Tokens | Savings min us | Savings median us | Savings max us | Official adjusted median us |
|---|---:|---:|---:|---:|
|1|-1.791999|-1.079999|0.304000|27.365950|
|8|-1.567999|-0.592000|0.656001|65.509585|
|16|-1.248002|0.015997|1.184000|99.167387|
|32|-1.744002|0.208000|1.104005|124.499396|
|128|-2.191998|0.504002|2.335995|182.574146|
|256|-1.872003|0.031997|1.711994|216.281217|

Positive savings mean faster. The matched table and official score measure different inputs/timing procedures; they are not pooled. Official candidate2 is5815.571639us/2.312498381x, CI[2.277810905,2.347185857]. It is above the verified terminal public-board point5737.988910us, job2821fdb184d740b1, rank330 at this snapshot. Physical source attribution of that other owner's public run remains with its owner. The user's fixed5559.8us target is still178.188910us below the board point.

Stop this exact readiness representation for both the failed performance gate and incomplete publication proof. Retain the original job-counter helper. Do not retry, change flag layouts/lanes/epochs/granularity, remove fences, select favorable masks or infer a general distributed-flag loss. The result does not identify whether scan/issue overhead, publication timing or another cost caused the loss. The original observed dependency wait is not a recovered speedup. No near-ceiling claim follows.

The prepared held-out70da64b4 and pure release00f13180 drafts remain unsubmitted, unqualified and unintegrated because their screen condition failed and the publication proof is incomplete. No public action follows and no public permission was requested for a losing file. All raw receipt/profiles, strict analysis and the conditional disposition are archived in evidence/2026-11-03-moe-short-ready.

## Independent publication review, November3

General25ad1c20 and research1fd98d1e reviewed the frozen helper974d7743 from pairedb6107a78. Both independently checked contributor intervals and reset placement. Both found the missing nonleader-store-to-leader-release edge. This is a limitation of the publication argument, not a demonstrated wrong GPU output or a proven baseline defect. No source repair or additional GPU run follows for this stopped implementation. The root AGENTS.md now requires an explicit cross-lane relay/release proof and disclosure of stronger CPU-model operations. The owner's resulting prose and standing-rule edits have not received another independent review.

## Qualified refill required

There is no qualified next kernel treatment in the current owner queue. The finite short-load endpoint allocation is complete; unused quota does not reopen it. The precise gap is a source-backed way to remove at least2us at T1/T8 while preserving the BF16 boundaries, writer/proxy fences, progress, graph lifetime and other four loads. The two timelines do not separate helper arithmetic, store-drain/publication cost and issue/partial-buffer delay well enough to select another correction. The flag comparison supplies no isolated causal split either.

Research refill should identify the actual selected helper/flags, a concrete removable or reorganized dependency or work item, its credible complete-call microsecond benefit and one decisive matched test. An instruction-count or local spill observation alone cannot supply H200 speed. Do not refill with the stopped NW16/BAL/row-owned/final-counter/flag variants, old WGS observers, a prefetch already present, large-load/PARAM-Y work or the deferred megakernel. This is a scoped missing-evidence request in the shared inbox, not a claim of delivery, a new approval gate or task completion at5559.8us.
