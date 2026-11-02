# Shared-A measured negative; next fixed weight-delivery implementation

November 2, 2026, terminal23:03:31UTC, boards/quota23:05UTC. Evidence checkpoint **2d4548c2**; shared-A implementation **dcb24972**, accepted-job registration **26079e8a**. Captain/sole dispatcher: **OK Driver**, `01a1963f-0c8e-7d41-be02-0e691dcfcaf7`. Existing implementation owner: **OKC Developer**, `01a19976-9e99-79c0-bab1-644bd8c0293a`. Previous T3 captain stays user-stopped/paused.

**Own private layer job `5adf172986ff43dc` is a valid measured negative.** All13 correctness tests and12 benchmarks pass,12 complete profiles are present, actual GPU admission succeeds and56 startup replays are verified. Score **1.6127943994197347x**,95%CI **[1.5886024834284387,1.6369863154110305]**, versus public incumbent2.1260826764231093x. **PARK this exact shared-A implementation; no confirmation, producer/stage/register tuning, paired framework or rescue check.** Developer explicitly acknowledged the exact stop. No public/default change occurred.

## Owning target evidence

[Unmodified own terminal JSON](evidence/2026-11-02-driver-shared-a-first-check/status-wait.raw.json) verifies job/problem/check/team, done/ok, passedtrue, returncode0, no errors/flags. Preparation **99.364714113s**. Physical894471LF source **fbca3b75125dd45b3e2801358e95aa14a16750860482989a5fef444a576cd578**, nativec2958fc9. Service contentd697db44e259d52c65efcf11bd6ce68e1d6fba7d28e3f3f5c7a9bc086707acba is separately associated through the own POST/receipt. H200/node`n-5imphluruu3bb3ci5c2w24thxw`, driver580.159.03, clusternyc3b.

Actual B4 and both MAIN signatures report grid132,672threads,resident1,dynamic221952/static0 shared bytes,80registers; local bytes40for B4 and24for MAIN. Startup prints `G_RELEASE_STARTUP_OK ...56 verified_replays`. This is actual target qualification, not inference from CPU or secondary compiler success.

Normalized46:16 endpoints `(46*sliding+16*full)/62`, in microseconds:

| Load | Shared-A | Fresh unchanged607 `cc474883117947c0` | Descriptive change |
|---|---:|---:|---:|
| decode1 |79.0084|78.7645|+0.31%|
| decode16 |165.2180|152.0162|+8.68%|
| decode32 |302.1113|191.8978|+57.43%|
| verify1 |113.3161|112.6817|+0.56%|
| verify16 |467.0937|261.7130|+78.48%|
| verify32 |628.1421|351.0802|+78.92%|

Geometric fresh-control/candidate endpoint factor **0.752409236**; official score/public ratio **0.758575580**. The six baseline/candidate ratios exactly reconstruct the official score. Evaluator-file hashes match freshcc474, but all12 benchmark seeds, node, cluster and driver differ. These are **descriptive continuation screens, not matched causal effects**. Unchanged decode16 also drifts. Independent terminal review recomputed all six endpoints and confirms the exact PARK/no-confirmation decision.

[Owning hotspot map](evidence/2026-11-02-driver-shared-a-first-check/hotspots.md) shows the new expert activity dominating affected cells (for example355.2us V16sliding and492.1us V32sliding). This localizes observed cost to the complete selected expert call; it does **not** identify producer throughput, shared bandwidth, extra staging or spills as the cause. Original [profiles/log manifest](evidence/2026-11-02-driver-shared-a-first-check/profile-manifest.json), raw log/JSON/text, source association and [calculation](evidence/2026-11-02-driver-shared-a-first-check/terminal-analysis.json) are retained. Closure concerns this fixed source, not every shared-operand or ownership architecture.

## Current standing and capacity

Fresh23:05UTC: **overall/layer375**, public incumbent`607eb52bdaf24160` unchanged at2.126082676x; layer100cutoff2.2684x, approximately6.69% higher. The overall API supplies ranks only; the layer numeric gap is not an overall formula. **8private layer checks,10scored slots remain; no live jobs.** Other private capacity remains MoE18,attention20,projection20. No reset wait or public substitution is needed or authorized. [Live summary](evidence/2026-11-02-driver-shared-a-first-check/terminal-board-summary.json).

The two captain-directed architecture checks have produced valid unfavorable results, rather than another setup/correctness failure loop. They have not produced a new winner. Preserve that distinction and the actual costs; do not count source proofs, closed ideas or held harnesses as runnable alternatives.

## Next actual implementation allocation: direct weight loads into RS registers

Do not leave the developer awaiting an unspecified next decision. **Allocate ONE fixed implementation from immutableba9/c7**, owned by the same developer, **60active minutes**, then **one ordinary private layer qualification/measurement** when conforming/runnable. Captain alone dispatches after exact source publication and fresh quota/no-live checks. This is a new delivery mechanism, not tuning either losing K128 or SS implementation.

Keep original **256-row units, four compute WGs,64-token reuse,K256,16stages/unit,count-dependent NTT, prefix/range choices, RS-WGMMA math, original finalization and one-CTA grid**. Remove the compact-weight shared ring and its32-thread bulk-TMA producer. Consumers load the **same immutable prepared wAs/wBs code and scale vectors directly from global memory into the existing register-prefetch/decode loop**. Retain the independent32-thread X producer and original three X slots. Total **544threads=512compute+32X**; X warp IDs512..543, full join544, routing/range barrier544. No new packing, persistent expansion, global dequant pass, extra launch or ownership protocol.

Independent bounded source/closure review found no hard contradiction or exact prior closure in the searched ruled-out/run/steering/tool evidence. Related legacy TMA/register GEMM, projection RD and T1 three-launch results are different bodies; they do not establish a result for this candidate. That bounded miss is not a claim that no prior experiment exists anywhere.

**Critical physical address:** pass the actual stage`st=s0..s1-1` to the stage lambda. Never use`seq%16`: sequence continues through reordered/split unit ranges. With`j=u>>UBL`, `ub=u&((1<<UBL)-1)`:

`pc=(PB?p.wBs:p.wAs)+(((S.jE[j]<<UBL)+ub)*16+physical_st)*WGS_STG+wg*WGS_PIECE`.

Existing per-lane code and raw/SC6 offsets remain unchanged and aligned. Keep original **doneAs index`ub>>1`/waitcount8**,16stage counters and original UBL5A/4B; do not import the shared-A K128 counters or packing. Preserve exact `dq4`, BF16x2 scale operations, global postscales, BF16 intermediates, SiLU, full per-expert norm and weighted combination.

Place all three X slots at the former ring base; update X producer/consumer descriptors and allocation consistently. Unchanged22272B header plus98304B X gives **120576B dynamic** before any new fields. This still permits only one CTA/SM after reservation; do not claim doubled occupancy. Remove only the raw shared ring's now-dead producer/full/empty waits/releases. Preserve X ready/retirement/proxy fences, register-A fencing, inter-CTA publication/acquire, all split/terminal drains, empty/replay lifetime and cooperative admission. Prepared weight storage stays immutable and retained for every captured read.

The hypothesis is avoiding raw shared delivery/load/wait work while keeping the original compute dataflow. **Logical HBM payload does not decrease.** Global-LDG latency/cache/scoreboard pressure replaces LDS plus bulk TMA, and independent three-stage weight lookahead is lost. Preserve one-load-group register prefetch; issue the first code/scale loads before the X wait where legal, while retaining every activation dependency. Do not add a cache/prefetch/tile/stage/register sweep or assume the compiler implements overlap from source order alone. The SS loss is not causal evidence for this alternative.

Affected loads remain D32/V16/V32 in both contexts. The existing conditional fully-exposed sizing needs~11.37% affected expert-activity conversion for3% layer gain before replacement cost; this is an exploration threshold, not a forecast or removable-time ceiling. Compared with the narrowed source reserves below, this proposal changes dominant delivery with no new format or nonlinear pass, so one bounded build/measurement is justified without first proving a win.

Developer owns`tools/wgs_direct_weight_registers.py`, focused address/selection/protocol tests and worker-direct-weight-registers report/evidence. Explicitly test skewed/reordered/split ranges where`seq%16` is wrong, both contractions, raw/SC6 global addresses, original counter/scratch boundaries and unchanged selected short routes. Compile actual selected bool entries with effective target flags, disclose resources, retain ordinary56startup/295s wrapper and source1MB limit/notices. A concrete source/progress/resource contradiction stops promptly; no silent research extension. First target setup/math/progress/admission failure closes the fixed allocation without rescue; valid weak/negative parks without tuning. A substantial positive first signal earns a separately recorded matched confirmation decision, not public permission. Publish the runnable pin promptly rather than waiting out the60-minute window.

## Independent reserve rejected while the job ran

Packed-MoE hardware-compressible backing is not the next allocation. The [actual generator](../../muse_decode_layer/nvfp4.py) produces uniform random code bytes; canonical lossless code packing cannot be credited with systematic code entropy reduction. Selected WGS banks12/13 total840MiB, of which768MiB are codes and72MiB SC6 scales; unit banks10/11 add864MiB for short/MID routes. Changing original input banks0/3 would miss the actual readers.

Even a deliberately generous WGS scenario eliminating all8.57% scale bytes and turning the same fraction of the **entire** profiled expert activity into endpoint savings gives only2.236% layer, not a physical ceiling or forecast. There is no evidence of useful compression of the random-code bulk. Existing BF16 PR/Lt-specific owner hooks also do not cover native startup/private/eager/captured readers; new exact-byte allocation and state-local native proxies would be required. No implementation/check is allocated. This is an ROI/source STOP for this proposal, not a general VMM safety prohibition or a result on trained/redundant weights. The earlier sparse-count SIMT reserve likewise lacks actual layer coverage and an instruction/traffic advantage; its scoped review remains in the allocation report.

Delivery and explicit pickup of the new direct-weight implementation will be recorded separately. This terminal publication claims neither a third submitted job nor a speedup.
