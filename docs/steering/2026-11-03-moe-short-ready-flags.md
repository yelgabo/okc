# Distributed A-ready publication for Tiny and Tiny2

November3,2026 Pacific. MOE-SHORT-READY-01 is the single next short-load candidate after own timeline f4012a0531af4013. No new candidate GPU job yet. MoE owner implements and measures; General retains independent validation support. Public control5df/C7 and every unrelated owner remain intact.

## Evidence and mechanism

The completed source-bound observer106a05c5 passes all correctness, replay, ownership and bounded cost checks. At T1 its per-CTA activation-ready waits have median4.60us/max5.95us; retained CTA2 has last A compute-unit retirement16.22us, helper retirement20.51us, and first B activation wait16.76–21.18us. This is a measured wait across helper completion/publication. It does not identify an atomic-contention cause. T8 has median2.43us/max4.38us. The source already prefetches phase-B weights, so adding generic cross-phase prefetch duplicates current work.

Current helper publication uses one contended atomic sum per touched job after all of a CTA's A epilogues. B compute warps poll that job total. Replace the total with one completion flag per CTA. After the original writer fences and warp synchronization, helper lane0 publishes its private flag. For each B job, all32lanes acquire the flags of exactly its contributing A CTAs, then retain the existing warp synchronization before reading activations. This reorganizes the publication dependency without changing unit work, epilogues, arithmetic, ring depth or the final grid barrier.

A different option is to move late A epilogues to compute warps, similar to current late B epilogues. That changes partial-buffer ownership and adds publication joins or atomics. Choose flags first because it preserves all arithmetic and producer scheduling and directly tests whether the existing aggregation costs matter. Expected benefit is a hypothesis, bounded by part of the observed4.6us T1 wait. The main uncertainty is that helper math or store-drain latency, rather than aggregation, may dominate. One matched comparison resolves this design; no ingredient must win separately.

The earlier short-RMW c5e4f89672f743bb changes the final global-barrier counter's memory ordering and failed its complete gate. This candidate preserves that final counter exactly and changes the separate phase-A readiness representation. It is not a retry of that treatment, NS4, row-owned full-K, WGS prefix, queue or scale/group work. See the archived short-RMW source/terminal reports.

## Memory, coverage and progress proof

Use existing reserved int32 words p.doneA[128+cta]. JMAX is below128 and128+136 is below512; the next xready region begins at512 and the second sum buffer begins at1024. Static assertions enforce these boundaries. No new scratch allocation or host argument is needed. Every flag has one writer and resets after the original global B-completion barrier. Initial scratch is zero, and same-stream completion orders reset before the next graph replay. Other paths keep their original source and storage behavior.

For job j let its A unit interval be[L,R), total A units T and grid G. CTA c owns[floor(T*c/G),floor(T*(c+1)/G)). Its intersecting CTAs are first=ceil(G*(L+1)/T)-1 through last=ceil(G*R/T)-1 inclusive. All products fit32bits under G<=136 and32jobs. Positive contributions partition exactly the job, so the old sum reaches RB_A if and only if all those CTA flags are ready. This preserves the existing coarse publication point, including a CTA spanning jobs. Empty routing executes no B units or division.

Keep each helper lane's original GPU fence and original warp join. The leader's aligned st.relaxed.gpu.global.u32 following that fenced relay is a release pattern; GPU acquire reads observe it. The retained __syncwarp redistributes the lane-wise acquired visibility before generic inter reads. This is an inference from [PTX8.7 release/acquire and synchronization rules](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model) and [CUDA12.8.1 warp synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions), applied to these exact operations. The vote alone is not a memory barrier. No writer fence or generic-to-async proxy fence is removed.

A producers never wait for B. Each resident CTA publishes before its helper reaches B. Consumer waits depend only on prior A work, and original cooperative admission remains. Reset occurs only after every compute warp and helper passed the existing B join and full-grid barrier. Preserve the original final publication/acquire, generation advance and sum-buffer parity. CPU schedules, compiler lowering and GPU tests establish separate facts; tests alone do not prove ordering.

## Qualification and finite decision

Before dispatch, pin the actual selected Tiny/Tiny2 helper and unchanged flags; reverse the patch exactly; enumerate all job/grid interval mappings and randomized producer-completion schedules; inject a missing contributor; test empty/occupied graph transitions, guard/reset/ownership and partial-failure retention. Compile both selected variants and inspect the explicit strong GPU flag stores, acquire loads, vote, retained fences/joins and unchanged final counter. Local compilation is not target parity.

One private matched screen with the original three normal/skewed fresh banks, four independent owners, all54cases/36timed rows/378observations/216captures/1152cold replays/54restores. Candidate affects T1/T8 and same selected Tiny fallback sizes; other four scored sources remain original. Official candidate2. Use tools/moe_short_gate.py: one same short cell must save at least2us across every bank/distribution and all four owner crosses, beat twice duplicate noise, preserve other four loads within noise and have no affected regression. Whole-six benefit must also resolve above duplicate noise. Retain all numerical/cache/graph/ordering/source guards,295second preparation and failed partial evidence.

Stop exact design on any correctness, policy, source, coverage, budget, drift, regression or benefit failure. No flag-layout/lane/epoch/granularity sweep, larger guard, fence removal, selected-mask rescue or blind retry. A successful screen needs held-out confirmation, exact release qualification and the applicable public approval. No new public authorization follows from the previously approved5df file or the lead's separate factor-Y approval.

## Frozen candidate and local qualification

Pairedb6107a783c8ded6dc626f6cfa99e7e1b53201e5225bd2073f736c69125f16964/490593bytes, native helper974d7743bb8d049150fe46b62f84c2308158e382588903f185d8155c787faf20, runtime8e2dd47c28fb0f82c6261bef869185b3b93781b171eeb8e06938712200c588c8. Five CPU integration/source/mapping tests pass, plus two short-gate arithmetic tests. The final targeted two-test pass checks the delivered tracked receipt fixture, positive owning-reader association, wrong job, a missing reset observation and failed complete evidence with otherwise favorable performance rows. Full host orchestration covers54cases/378observations/216captures/1152synthetic timings/54restores and1512flag-zero checks. These tests use real CPU Torch storage and explicit CUDA/native facades; synthetic timings are not a GPU gain.

The actual native wait function also passes24CPU warp-model cases with32lanes, withheld publication, visibility after the modelled warp join and exact contributor sets. A last-contributor omission fails as expected. This models the primitive contracts and exercises the actual wait/address code; it does not execute PTX on a GPU.

Both actual Tiny/Tiny2 variants compile with CUDA12.8.1 SM90a locally,154registers/3barriers/zero stack or spills. Original controls use145registers and are unchanged from the timeline builds. Both entries lower the new publication to STG.E.STRONG.GPU, retain explicit acquire loads/votes, all8proxy-fence sites and both original final acq_rel RMW sites. Register counts, source coverage, memory-model reasoning and measured H200 speed remain separate facts. Full compiler diagnostics and source/adaptor pins are archived; full SASS stays ignored under .build per the repository audit.

No independent native review or target result is claimed. The independent General support validates the preceding actual timeline export; it does not certify this candidate. Fresh shared preflight before dispatch currently shows3MoE private checks left and other accepted owner jobs. One new private source may be queued, with no duplicate, cancellation or other-ID waiter.

## Reconciled concurrent short-load evidence

Origin/main fa762950 includes another owner's completed Tiny timeline19128cda79ca4e25 and closed NW16+balanced ab54144f6c4d4727 and BAL-only b45cce36b47f4e81. No poll or retry of those jobs follows. This candidate retains NW8, the original proportional A/B ranges, original ring and code/scale packing. Its ready-publication source and owned files are distinct; the other lane is recorded closed. Preserve its losses rather than redispatching them.

The other timeline reports about1.7–1.8us ready waiting on its judge-input observation, less than this instrument's cold normal-bank4.60/2.43us medians. Inputs, replay/cache conditions and instrumentation differ; they are not a matched comparison. The measured wait is not a promised2us saving. This reinforces the need for the frozen multi-bank control comparison and separate official score. No causal fence-versus-aggregation conclusion is drawn from either timeline alone.

## Accepted private check

Own e3e2495464f34b2c accepted once for pairedb6107a78/490593bytes, published73ab76cf/reconciled8fad06be, dispatch intent9e778ae6. Fresh quota had3checks left; shared live jobs were recorded before the POST. Accepted ID persisted before collection or other measurement. One own-ID foreground waiter writes ignored .build/moe-short-ready/status-wait.raw.json. No target outcome yet.

## Conditional follow-through prepared while the screen runs

Tools/moe_short_ready_confirm.py fixes held-out banks2203011/2203017/2203029, disjoint from the screen and its fresh97offset banks. Its two CPU tests establish that only BANKS and runtime identity change, the exact helper/compiler requests remain, and the original screen/gate module stays immutable. No seed search or alternate held-out bank is allowed. No confirmation is dispatched unless the complete frozen screen passes.

Tools/moe_short_ready_release.py prepares a single-arm draft from exact approved5df257af, preserving the original MID/B4/MAIN, quantized weights, packing, scale fallbacks, operator checks and three-buffer ABI. Both changed short compiler requests equal those of the active paired candidate; the other three equal the original public file. Two CPU tests check contract identity and actual preparation/forwarding. This source is not GPU-qualified or public-approved. Conditional manifests/test output are archived; generated draft payloads stay ignored until their gates permit use.
