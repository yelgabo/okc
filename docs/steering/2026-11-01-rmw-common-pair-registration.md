# Repaired-common OFF/ON paired check registration

November 1, 2026 Pacific / November 2 UTC. Evidence baseline: captain allocation **fb2c28f / 8b223b3**, process checkpoint **94cc717**, terminal qualification **516a93a**, own prior private **c1931f0c9e834485**. This is the implementation owner's preregistration for **one private check, tranche #2/4**. No public/scored authorization exists. Source and CPU qualification are complete; this document initially precedes registration. Append the exact accepted ID and outcome below; never infer an ID from another check.

## Ready source and reader pins

- Generated upload: .build/rmw-pair-v3/check.py, **923915 bytes**, SHA256 **c4a5ee29bcfeca248c0fee638ed103e9f211f7a8cc32b86bd3bb642676bf45e3**.
- Frozen upload contract: tools/g_common_rmw_pair_manifest.json, SHA256 **09f2132b25cb1ec8456efcf2ab0f9197780d4612f31bac710eaee27c5db14cf4**.
- Owning reader: tools/g_common_rmw_pair_read.py, **10920 bytes**, SHA256 **745d7bb508d170dbc6bcd4b86e0f8a0b39c9a9ba08885aec54a45a33e68f004d**.
- Shared repaired device/instrumentation prefix SHA256 **aea9735e7d91a7d51a7afe63c8cf0774b5bbd15de6ab416bbc5acbd2469ddbd2**; harness-core SHA256 **d2e96c73998491eb20294b06b222b7f3742720a0dc5e9231287e11b4976b5ad0**.
- Raw repaired-common OFF production base **4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280**. Fixed ON raw base **2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd**.
- Existing single-arm artifacts remain unchanged: qualified-wrapper OFF **32196b8bc1c954e9b7f2c6667e6b3e0aeed14da0fed6fe53bc153c9d563e5047 / 918490B**; ON **978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a / 918978B**. No repeat qualification is allocated here.

Full file pins, factory capture, compiler resources/logs and CPU test receipt: [evidence directory](evidence/2026-11-01-rmw-common-pair/). The builder regenerates the upload deterministically; the historical f042 machinery and its old reader stay immutable and are not dispatched.

## Bounded source changes

The existing paired builder is reused, with the exact common-control AST deltas applied to its device prefix. Both private variants and the official return path retain the attention cooperative/admission repair, effective preexpert coefficient and all expert cooperative/progress fixes. Product functions are identical to repaired common except the explicit per-owner short compile policy and additional native export lists. The only ON device treatment is the already proved Tiny/Tiny2 returning final-counter RMW representation. No unsafe ordinary-launch arm or historical607 comparator is run. See the retained [RMW ordering proof](2026-11-01-g-ordering-restoration-proof.md); no fence/barrier has been removed by this rebase.

The narrow harness repair adds the previously qualified native attention/20-field resource observers, counts attention only after a successful native return, reuses the inference-safe four-entry exact-front/weight-owner/routing-ID/version memo, restores stream inputs before the first post-capture replay, and inspects occupied-to-empty transitions using the same captured native graph with poisoned outputs. Empty scratch/counters are never manually reset. Private references run under inference_mode(False)/no_grad; normal production prepare retains the caller's context.

The comparator returns the original **repaired-common OFF production state**. On success and failure, its finally block restores native builder, native dispatch, attention and reference bindings; disables native attention observation; clears private reference/dequant/build-observer state; restores TF32; and retires only private guard bookkeeping. The total guard starts before production preparation and is checked after cleanup/report construction. A printed capsule alone cannot establish completion; the reader requires the owning terminal official success.

## Frozen experiment and stopping rules

Four independently owned mutable placements, two code variants: OFF0/OFF1 and ON0/ON1. Shared immutable code is allowed; private mutable backing storage, inputs, caches, outputs, scratch and Lt workspaces must be disjoint. Original banks **993811 / 993817 / 993821**, all six token loads **1 / 8 / 16 / 32 / 128 / 256**, sliding/full contexts, **36 rows**, **24 cold randomized rounds**, **256 MiB flush**, all four OFF/ON crosses, and **46:16** weighting are unchanged.

The reader requires 144 initial math checks, 144 final fresh checks, eight fresh replays per placement/bank, 32 eager/empty-graph checks, 16 occupied-to-empty checks, all resource/ownership guards, and 432 successful native attention witnesses (two warmups plus capture for every placement/row). It recomputes the full-six raw timing gate independently. Every bank/cross must exceed log(1.01) and 2x duplicate log-null, with context regression no greater than max(0.5us, 1%). A complete null, small win, inconsistent bank/cross or regression failure ends RMW-only promotion this tranche. A setup failure stops for its specific diagnosis; no blind retry or seed/mask chase.

A successful official OFF single-arm score is descriptive. The private raw cold-graph comparison is the registered attribution measurement, not an official paired score. #3 remains reserved for fresh compilation/preparation with **202611021 / 202611023 / 202611029** only if #2 clears its unchanged gate. #4 remains reserved for an actually needed pure-file qualification; the already qualified exact ON is not repeated. Public/scored dispatch needs separate direct authorization.

## Validation and 295-second assessment

**16/16 CPU tests pass**: nine new paired/source/owner/reader gates plus seven preserved historical builder/reader regressions. Actual generated entry points run on the existing explicit CPU arithmetic/CUDA stand-ins. The tests cover inference scope, memo owner/result/ID mutation, front-version invalidation, aliased owner rejection, stale fresh outputs, wrong output after memo reuse, native attention failure, unconditional cleanup on prepare/private/resource/budget failure, source equality, wrong owning receipts, missing witnesses/resources/cleanup and recomputed decision mismatch. These are CPU/source results, not H200 execution or CUDA graph qualification.

The actual production source factories request **seven native builds**: five OFF and two extra Tiny/Tiny2 ON builds. Duplicate placements reuse the corresponding code modules. Exact current source/flags were device-compiled for Tiny/Tiny2 OFF/ON, both bool entries. All eight selected entries use **147 registers, zero stack/spill stores/spill loads and three barriers** under local NVRTC/PTXAS12.8. OFF/ON resources agree. Four compile durations were **1.061234 / 0.943311 / 0.959378 / 0.957335 seconds**; extra ON device compiles totaled **1.916713 seconds**. This omits Torch host compilation/linking and is not a forecast of judge CUDA13.0 resources. The source-only memory-order proof is retained; identical SASS or isolated mathematical instruction schedules are not asserted.

c193 measured normal production preparation **106.901449s**, total **109.269811s**. At that observed normal-prepare cost, the new 295s guard leaves **188.098551s** for the two extra builds, private clones, weight-bank hashing, the paired exercise, references and cleanup. Memoization bounds cached back-reference banks to four per row and at least144/at most1152 misses across1152 reference calls; it does not skip any candidate validation. The paired loop has **3456** timed whole-graph replays/flushes. Scaling c193's reference phases by the new front-bank/miss counts suggests roughly5.5s of reference work, but this is an estimate, not a measured paired upper bound. Host compilation, four-placement cloning/hash/graph capture/event overhead and queue hardware load remain unmeasured.

**Assessment: plausible for one authorized 295s attempt, not guaranteed fit.** The bounded extra compiler work and retained188s headroom support dispatch without another harness rewrite. The hard guard and cleanup stay in force. A timeout is setup evidence and stops this allocation; it is not a treatment loss and will not trigger an automatic retry.

## Registration and next gate

Before registration: send these exact pins/assessment to captain, commit/push this preregistration, and refresh authenticated layer quota and latest jobs. Require positive private quota and no live owned layer job. Sole dispatch remains with this implementation thread. Next allowed action is one private check of the pinned upload, then wait on its exact accepted ID and archive the owning terminal receipt/log/profiles. No second experiment is implied.
