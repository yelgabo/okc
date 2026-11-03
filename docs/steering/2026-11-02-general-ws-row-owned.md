# GOPT-WS-OWN-05: full-K output-row ownership

Current decision: retain the original fused WS kernel. The fixed row-owned replacement passed correctness and failed all six matched performance groups. Stop this exact topology.

Actual pickup: 2026-11-03 06:39 UTC, November 2 23:39 America/Vancouver. General Optimizer read origin/main `55298ac7883795d23d272a6da20ab7dc529ca3db` and accepted the [fixed build brief](2026-11-02-general-hotspot-search-refill.md#post-split-research-decision-full-k-output-row-ownership--november-2-2026-pacific). Four-hour build deadline: 2026-11-03 10:39 UTC. No GPU job has been dispatched.

## Selected design and decision

Use exact standalone release `5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e`, helper `b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716`. One isolated T8/T16 native entry, four output-owning compute warps and one weight producer, 64 rows, K1024 stages, three 36 KiB weight slots, generic activation staging and fused finalization. T1/B4/MAIN and disabled compiler requests remain unchanged. This reorganizes reduction ownership after three helper negatives. It does not retry NS4, norm aggregation or the stopped two-launch split.

The benefit is a hypothesis: remove cross-warp K partial writes, reads and reduction. Weight bytes and MMA counts remain equal; added activation traffic, fewer compute warps and lost early overlap may outweigh it. CPU layout and archived SASS establish source feasibility only. The target result below validates the tested numerical/replay/admission path and preparation fit. Full-call timing rejects this implementation; no general short-path ceiling follows.

## Execution and gates

Next action is native implementation plus byte-permutation, fragment, publication, lifetime and admission checks. Preserve all finite stops in the brief. A runnable qualified candidate receives at most the authorized private same-process comparison against two independent release controls. Three fresh normal/skewed EP8 banks, all nine correctness sizes, all six scored loads, cold replay, exact source/job identity, input/scratch guards and cumulative bounded partial records are required. Freeze complete-six factor >=1.01, greater than twice duplicate noise for all four crosses in every bank/distribution, no affected 1% regression, and unchanged-load drift <=max(1%, twice null). No topology/shape/ring sweep or automatic retry.

Live outputs stay in ignored `.build/gopt-ws-owned/`; archive after collection ends. No public submission or production promotion is authorized. MoE Worker's direct-input job and CUDA Optimizer's BF16 work remain separate. Independent review is unavailable under the no-new-agents instruction; source, model, compiler and target checks will be identified separately.

## Progress

- Pickup recorded; implementation and qualification underway.

## Frozen build qualification, 06:59 UTC

The implementation finished inside the four-hour box. [Frozen endpoint](evidence/2026-11-02-gopt-ws-owned/paired.py), SHA256 `f31b652f2bf315465072f8c43771befd9a16101354d8bf131d34de8999b173a5`, 389590 bytes, selects new native `896b5350238b5d82880bcc9af0682051b7914e84b51164c68e9c249e3e3b5aac` only at T8/T16. Its [manifest](evidence/2026-11-02-gopt-ws-owned/paired.manifest.json) freezes six compiler requests and the gate before dispatch. All five normal release requests and all four raw-scale fallback requests are unchanged when the treatment is disabled. Candidate states use one new entry for both affected sizes. T1/T4, T32 and larger dispatch remain original.

[Ten CPU tests passed](evidence/2026-11-02-gopt-ws-owned/cpu-final.txt). They cover all 147456 permutation bytes,8192 code-fragment mappings and scale vectors, range/512-credit coverage for 0..88 jobs, ring phases, exact baseline build requests, the complete-six gate and unchanged guards, actual CPU Torch packing with an injected corruption control, FP32 decoder agreement with the problem's NVFP4 decoder, generated orchestration/private storage/fresh replay, cumulative completed rows on cleanup/serialization/budget failure, primary-error preservation and failed-receipt identity. The CUDA execution and timings in these host tests are synthetic. They prove host behavior, not GPU correctness. The final native source also passed the five source/model tests after the unused partial-sum allocation was removed.

Both [normal](evidence/2026-11-02-gopt-ws-owned/compile-final/compile.json) and [raw-scale](evidence/2026-11-02-gopt-ws-owned/fallback-final/compile.json) local SM90a device builds passed. [Full host/device translation and binding objects](evidence/2026-11-02-gopt-ws-owned/full-host-final/compile.json) passed against the local CPU Torch header image with its required C++20 adaptation. Local CUDA12.8.1 reports160 threads,146816 dynamic shared bytes,96 registers and zero spills for both normal bool entries. The raw-scale fallback has its own retained diagnostics. This is not target binary/toolchain parity or measured performance. Target linking, residency and numerical behavior remain gates inside the check.

The [SASS audit](evidence/2026-11-02-gopt-ws-owned/sass-audit.json) finds the new MMA/bulk-copy paths, consumer and all-thread barriers, generic coherent activation loads and emitted `FENCE.VIEW.ASYNC.S` proxy fences. There is no cross-warp partial buffer or helper epilogue queue. [Archive hashes](evidence/2026-11-02-gopt-ws-owned/archive-manifest.json) bind the artifacts. A [variable-entropy capsule test](evidence/2026-11-02-gopt-ws-owned/entropy-test.json) using actual prior target rows occupies5844 bytes within 12000, with all 36 rows recoverable.

During qualification, the host tests caught an incorrect expected-extension inventory. T8/T16 replace Tiny2/MID and T4 still uses Tiny. The emitted factory inventory and persistent test now agree. The Mac Python environment lacks Torch, so that attempted test run failed; the saved container run above passed. Original diagnostics remain archived. The research catalog checker still reports the independently changed ruled-out ledger pin. Its diagnostic is retained, the relevant scoped negatives were read, and no catalog hashes were refreshed to hide it.

## Publication, progress and lifetime review

The ring starts only after stable routing. Its immutable weight producer issues all A and B stages independently of activation readiness. Full-barrier completion precedes generic weight reads. Consumers fence the async proxy after reads and rendezvous before one release arrival permits reuse. Consumers alone publish and retire the shared activation tile. Empty/full parity is tied to one continuous stage sequence across A and B. No outstanding copy remains at the final all-thread join.

Every phase-A writer executes a device fence before the 128-consumer join. The publisher then adds four credits per completed64-row block, preserving 512 per job. Phase-B acquires that counter before generic intermediate loads. All 132 cooperative CTAs must fit the actual queried residency; a failed query/admission stops before launch. A producer cannot hold an unconsumed A slot while waiting for B readiness. The terminal all-CTA barrier, generation advance, alternate-sum clearing and slot0..7 final combination are retained with 128 compute threads. Scratch and packed tensors remain private per prepared state and live across captures. This source reasoning uses [PTX8.7 proxy fences, barriers and acquire/release](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-try-wait-mbarrier-test-wait); compiler output and ordinary tests cannot replace the ordering argument.

Before any timings, the target adapter must independently verify every GPU-packed code/scale slice against its original, run both affected sizes against a full FP32 one-local-expert MoE reference with TF32 disabled and restored, and observe actual launch admission. The existing three-bank normal/skewed tests then cover the full routed endpoint and replay transitions. No partial or selected-bank result can qualify a gain.

The fresh preflight public MoE snapshot remains4aa033ef23a142cd at5788.720720us. Its observed rank372 and rank1005558.603217us are timestamped board observations, not changed kernel timings. The saved promotion ledger check has19leads and zero gaps. No new result or public-release opportunity is claimed here. Next action is the one private own-ID dispatch, after fresh quota/job checks. No public action, production edit or other-owner job is included.

## Private check accepted

Private own-ID job `7f10243c98e74746` was accepted for `muse_moe_w4a16` at 2026-11-03 07:01:27 UTC. Frozen source revision `37964e414aaa36a52937c09418626ff4776c7406`, physical payload `f31b652f2bf315465072f8c43771befd9a16101354d8bf131d34de8999b173a5`,389590bytes, native896b5350. Official judged arm is candidate2, with only T8/T16 changed. No target result is recorded yet.

The sole waiter is exec86811, writing ignored `.build/gopt-ws-owned/status-wait.raw.json` and retaining its own exit status. Do not create another poller, redispatch or archive its live output. Collection will end before receipt/profile archival and strict identity/coverage/gate analysis. Fresh quota reports ten MoE private checks remaining before this dispatch. MoE Worker's independently accepted factorized-output job7479372115c64702 is separate and remains with that owner.

## Terminal decision: retain original fused WS, stop full-K row ownership

Finished 2026-11-03 07:09:31 UTC, November 3 00:09:31 MST America/Vancouver. [Own receipt](evidence/2026-11-02-gopt-ws-owned/status-wait.raw.json), [strict analysis](evidence/2026-11-02-gopt-ws-owned/analysis.json) and [summary](evidence/2026-11-02-gopt-ws-owned/summary.json) bind job7f10243c98e74746 to source37964e41, physicalf31b652f/389590bytes and native896b5350. Service payload identifierb944695a7043b2ddc9a2dfd981eb51987cd53728ebafe91d34d65c4c89382d84 is distinct from the physical SHA. The sole collector86811 ended0; outputs were archived after exit. Never poll or resubmit this completed job.

Correctness passed 5 official tests, 6 benchmarks and 6 profiles, return 0, no flags/static rejections. All 54 private cases, 378 observations, 216 captures, 1152 cold timed replays, 54 restorations and cleanup completed. Eight untimed FP32 reference cases passed. GPU preparation independently checked every packed slice. Worst global/row relative error was 0.004596856423/0.004795130342 against limits 0.015/0.075. Private preparation took 219.480960 s within 295 s. Profiles show one fused kernel at each affected size, with actual host admission exercised.

Performance passed 0/6 groups. All 24 complete-six factors are 0.845725665–0.901633789; their descriptive geomean is0.875757072. Every one of 12 affected bank/distribution rows crosses the 1% regression threshold. There are no unchanged-load drift-guard failures. That supports rejecting this exact complete replacement, rather than blaming a failed instrument. The official judged-candidate score is 6679.194366823 us/2.013491338x versus SGLang, 95% CI 1.983288968–2.043693708x. This CI is not a paired treatment-effect CI. The three-bank paired rows retain four placement medians, four cross ratios, two duplicate log-null magnitudes and routing census, with 8 cold replays per placement. Raw per-replay private samples and a paired treatment CI were not retained.

The following table averages the two independent controls and two treatments over the three banks. It is descriptive; the frozen decision uses each bank, distribution and all four crosses. T1/T32/T128/T256 are unchanged controls. The 128/256 rows are explicit regression coverage, not an architectural gain.

| Tokens | EP8 routing | Control us | Candidate us | Candidate minus control us | Delta % |
|---|---|---:|---:|---:|---:|
| 1 | normal | 34.200000 | 33.608000 | -0.592000 | -1.730996% |
| 8 | normal | 78.509333 | 104.352000 | +25.842667 | +32.916682% |
| 16 | normal | 116.640000 | 173.418663 | +56.778663 | +48.678552% |
| 32 | normal | 125.522664 | 125.887999 | +0.365334 | +0.291050% |
| 128 | normal | 186.421334 | 186.509331 | +0.087998 | +0.047204% |
| 256 | normal | 233.890668 | 234.485332 | +0.594664 | +0.254249% |
| 1 | skewed | 33.954667 | 33.928000 | -0.026667 | -0.078536% |
| 8 | skewed | 40.504000 | 64.493332 | +23.989332 | +59.227070% |
| 16 | skewed | 63.202667 | 97.522667 | +34.320000 | +54.301506% |
| 32 | skewed | 56.613333 | 56.490667 | -0.122666 | -0.216674% |
| 128 | skewed | 102.586666 | 102.170667 | -0.415999 | -0.405510% |
| 256 | skewed | 159.226668 | 160.336000 | +1.109332 | +0.696700% |

This is a combined A+C skinny treatment: hybrid/WS execution with full-K output ownership, changed weight layout and generic activation staging. It isolates none of those ingredients. Fewer compute warps, activation restaging, stage rendezvous, coarser work assignment and lost early issue remain possible costs; this job does not separate them. Removing cross-warp partials did not produce a complete-call gain here. Scope the loss to the 160-thread/64-row/K1024/NS3/132-CTA fused implementation at T8/T16 under these normal/skewed EP8 banks.

Retain original WS, NS3 and original norm updates. No row/warp/K/ring/shape sweep, split-finalizer rescue, retry, confirmation or integration follows. No layer measurement ran and no isolated family score can be inferred. The public standalone4aa033ef23a142cd/5df257af remains 5788.720720 us, 890.473646 us below this stopped candidate. A fresh all-five-problem public snapshot and promotion guard found 19 leads and zero gaps; this losing official point creates no public-release lead.

General has no qualified immediate task after this bounded structural alternative. Continue the same shared [research refill](2026-11-02-general-hotspot-search-refill.md), requiring a distinct selected-source complete-call mechanism or compatible stronger short-path comparator. Do not manufacture another helper knob experiment or infer a ceiling. Earlier-boundary credit still lacks exposed-wait evidence. Broader MoE factorized-output work and BF16 remain with their current owners. This closure requests shared-repository refill only; no message to another chat, GPU allocation or public authority is created.
