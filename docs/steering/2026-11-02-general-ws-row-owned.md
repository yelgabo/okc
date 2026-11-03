# GOPT-WS-OWN-05: full-K output-row ownership

Actual pickup: 2026-11-03 06:39 UTC, November 2 23:39 America/Vancouver. General Optimizer read origin/main `55298ac7883795d23d272a6da20ab7dc529ca3db` and accepted the [fixed build brief](2026-11-02-general-hotspot-search-refill.md#post-split-research-decision-full-k-output-row-ownership--november-2-2026-pacific). Four-hour build deadline: 2026-11-03 10:39 UTC. No GPU job has been dispatched.

## Selected design and decision

Use exact standalone release `5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e`, helper `b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716`. One isolated T8/T16 native entry, four output-owning compute warps and one weight producer, 64 rows, K1024 stages, three 36 KiB weight slots, generic activation staging and fused finalization. T1/B4/MAIN and disabled compiler requests remain unchanged. This reorganizes reduction ownership after three helper negatives. It does not retry NS4, norm aggregation or the stopped two-launch split.

The benefit is a hypothesis: remove cross-warp K partial writes, reads and reduction. Weight bytes and MMA counts remain equal; added activation traffic, fewer compute warps and lost early overlap may outweigh it. CPU layout and archived SASS establish source feasibility only. Numerical, device progress, admission, full-call timing and preparation cost remain unproved.

## Execution and gates

Next action is native implementation plus byte-permutation, fragment, publication, lifetime and admission checks. Preserve all finite stops in the brief. A runnable qualified candidate receives at most the authorized private same-process comparison against two independent release controls. Three fresh normal/skewed EP8 banks, all nine correctness sizes, all six scored loads, cold replay, exact source/job identity, input/scratch guards and cumulative bounded partial records are required. Freeze complete-six factor >=1.01, greater than twice duplicate noise for all four crosses in every bank/distribution, no affected 1% regression, and unchanged-load drift <=max(1%, twice null). No topology/shape/ring sweep or automatic retry.

Live outputs stay in ignored `.build/gopt-ws-owned/`; archive after collection ends. No public submission or production promotion is authorized. MoE Worker's direct-input job and CUDA Optimizer's BF16 work remain separate. Independent review is unavailable under the no-new-agents instruction; source, model, compiler and target checks will be identified separately.

## Progress

- Pickup recorded; implementation and qualification underway.

## Frozen build qualification, 06:59 UTC

The implementation finished inside the four-hour box. [Frozen endpoint](evidence/2026-11-02-gopt-ws-owned/paired.py), SHA256 `f31b652f2bf315465072f8c43771befd9a16101354d8bf131d34de8999b173a5`, 389590 bytes, selects new native `896b5350238b5d82880bcc9af0682051b7914e84b51164c68e9c249e3e3b5aac` only at T8/T16. Its [manifest](evidence/2026-11-02-gopt-ws-owned/paired.manifest.json) freezes six compiler requests and the gate before dispatch. All five normal release requests and all four raw-scale fallback requests are unchanged when the treatment is disabled. Candidate states use one new entry for both affected sizes. T1/T4, T32 and larger dispatch remain original.

[Ten CPU tests passed](evidence/2026-11-02-gopt-ws-owned/cpu-final.txt). They cover all147456 permutation bytes,8192 code-fragment mappings and scale vectors, range/512-credit coverage for0..88 jobs, ring phases, exact baseline build requests, the complete-six gate and unchanged guards, actual CPU Torch packing with an injected corruption control, FP32 decoder agreement with the problem's NVFP4 decoder, generated orchestration/private storage/fresh replay, cumulative completed rows on cleanup/serialization/budget failure, primary-error preservation and failed-receipt identity. The CUDA execution and timings in these host tests are synthetic. They prove host behavior, not GPU correctness. The final native source also passed the five source/model tests after the unused partial-sum allocation was removed.

Both [normal](evidence/2026-11-02-gopt-ws-owned/compile-final/compile.json) and [raw-scale](evidence/2026-11-02-gopt-ws-owned/fallback-final/compile.json) local SM90a device builds passed. [Full host/device translation and binding objects](evidence/2026-11-02-gopt-ws-owned/full-host-final/compile.json) passed against the local CPU Torch header image with its required C++20 adaptation. Local CUDA12.8.1 reports160threads,146816 dynamic shared bytes,96registers and zero spills for both normal bool entries. The raw-scale fallback has its own retained diagnostics. This is not target binary/toolchain parity or measured performance. Target linking, residency and numerical behavior remain gates inside the check.

The [SASS audit](evidence/2026-11-02-gopt-ws-owned/sass-audit.json) finds the new MMA/bulk-copy paths, consumer and all-thread barriers, generic coherent activation loads and emitted `FENCE.VIEW.ASYNC.S` proxy fences. There is no cross-warp partial buffer or helper epilogue queue. [Archive hashes](evidence/2026-11-02-gopt-ws-owned/archive-manifest.json) bind the artifacts. A [variable-entropy capsule test](evidence/2026-11-02-gopt-ws-owned/entropy-test.json) using actual prior target rows occupies5844bytes within12000, with all36rows recoverable.

During qualification, the host tests caught an incorrect expected-extension inventory. T8/T16 replace Tiny2/MID and T4 still uses Tiny. The emitted factory inventory and persistent test now agree. The Mac Python environment lacks Torch, so that attempted test run failed; the saved container run above passed. Original diagnostics remain archived. The research catalog checker still reports the independently changed ruled-out ledger pin. Its diagnostic is retained, the relevant scoped negatives were read, and no catalog hashes were refreshed to hide it.

## Publication, progress and lifetime review

The ring starts only after stable routing. Its immutable weight producer issues all A and B stages independently of activation readiness. Full-barrier completion precedes generic weight reads. Consumers fence the async proxy after reads and rendezvous before one release arrival permits reuse. Consumers alone publish and retire the shared activation tile. Empty/full parity is tied to one continuous stage sequence across A and B. No outstanding copy remains at the final all-thread join.

Every phase-A writer executes a device fence before the128-consumer join. The publisher then adds four credits per completed64-row block, preserving512 per job. Phase-B acquires that counter before generic intermediate loads. All132cooperative CTAs must fit the actual queried residency; a failed query/admission stops before launch. A producer cannot hold an unconsumed A slot while waiting for B readiness. The terminal all-CTA barrier, generation advance, alternate-sum clearing and slot0..7 final combination are retained with128 compute threads. Scratch and packed tensors remain private per prepared state and live across captures. This source reasoning uses [PTX8.7 proxy fences, barriers and acquire/release](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-try-wait-mbarrier-test-wait); compiler output and ordinary tests cannot replace the ordering argument.

Before any timings, the target adapter must independently verify every GPU-packed code/scale slice against its original, run both affected sizes against a full FP32 one-local-expert MoE reference with TF32 disabled and restored, and observe actual launch admission. The existing three-bank normal/skewed tests then cover the full routed endpoint and replay transitions. No partial or selected-bank result can qualify a gain.

The fresh preflight public MoE snapshot remains4aa033ef23a142cd at5788.720720us. Its observed rank372 and rank1005558.603217us are timestamped board observations, not changed kernel timings. The saved promotion ledger check has19leads and zero gaps. No new result or public-release opportunity is claimed here. Next action is the one private own-ID dispatch, after fresh quota/job checks. No public action, production edit or other-owner job is included.

## Private check accepted

Private own-ID job `7f10243c98e74746` was accepted for `muse_moe_w4a16` at 2026-11-03 07:02 UTC. Frozen source revision `37964e414aaa36a52937c09418626ff4776c7406`, physical payload `f31b652f2bf315465072f8c43771befd9a16101354d8bf131d34de8999b173a5`,389590bytes, native896b5350. Official judged arm is candidate2, with only T8/T16 changed. No target result is recorded yet.

The sole waiter is exec86811, writing ignored `.build/gopt-ws-owned/status-wait.raw.json` and retaining its own exit status. Do not create another poller, redispatch or archive its live output. Collection will end before receipt/profile archival and strict identity/coverage/gate analysis. Fresh quota reports ten MoE private checks remaining before this dispatch. MoE Worker's independently accepted factorized-output job7479372115c64702 is separate and remains with that owner.
