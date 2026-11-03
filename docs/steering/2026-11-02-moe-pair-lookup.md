# Native group16 pair lookup

November 2, 2026 Pacific. The user approved public submission of exact5df257af and directed continued MoE execution toward5559.8us. Public4aa033ef23a142cd passed5/6/6 and is the verified public best at5788.720720us, rank373. The remaining target gap is228.920720us. This worker owns B4/MAIN and whole-expert representation. General owns WS split-finalizer work; no overlap with its source or allocation.

## One coherent alternative

MOE-LUT-01 replaces the selected WGS arithmetic decoder with a generic pair codebook and a lossless byte-pair code layout. Keep c7c201ae as two independent controls. Two independently prepared treatment owners use native8c2cfc7c. All existing consumer waits, proxy fences, arrival counts, producer stages, phase partition, WGMMA grouping, FP32 accumulation, epilogues, BF16 rounding, norm and output order remain unchanged. T32/T128/T256 change; T1/T8/T16 remain controls.

The incumbent reconstructs four BF16x2 operands through DQ7 bit extraction and four BF16x2 scale multiplies for every k16 fragment. The alternative stores two natural FP4 codes in each byte and reads four precomputed packed BF16 values from a generic64KiB table. Each pair uses its original group16 FP8 scale. There is no group64 assumption, post-dot scale, narrowed activation or decoded per-weight bank. Every actual weight code and scale remains in its lossless packed representation. The table enumerates all code-pair/scale combinations independently of supplied weights and occupies the tail of the existing per-owner LUT tensor; its original256entry prefix remains intact for other paths.

This changes packing and decoding together. Testing either ingredient alone inside the incompatible old layout would answer the wrong question. The previous scale/group, multiplier-only and ring negatives remain closed. The unmodified FLUTE drop-in was ineligible for group16 and operator semantics; this is an owned compatible native decoder, not its wrapper or a claim that the foreign kernel wins.

The possible benefit is moving decode work to the read-only lookup path while asynchronous WGMMA continues. Added address instructions, data-dependent reads, a64KiB working set and compiler spills can overwhelm that benefit. No lower-byte-traffic, cache-fit or exposed-stall claim is made. The failed sparse observer is not used to select a favorable subset or quantify savings. One complete endpoint comparison is the cheapest decisive test of this coherent representation.

## Exactness, ownership and order

The actual factory chooses C7 B4/MAIN with SC6 and PRMT when admitted. Candidate packing changes only the WGS copies under a treatment-bound state with SC6 enabled. Original helper packing and all other weight copies stay unchanged. Non-SC6/native fallback source is preserved. Lookup addresses are256+256*scale6+pair8, within16640uint32 entries. Low-row scales serve registers0/2; high-row scales serve1/3. The table's bit patterns include signed zeros.

An exhaustive CPU witness compares all63finite admitted scale bytes times256code pairs,16128entries, against the incumbent DQ7/BF16 multiplication path. A separate basis test covers all8code positions/all16values. Real Torch packing checks the actual K-permuted layout and verifies that scale bytes are unchanged. These are source/CPU proofs, not target execution. NVIDIA's [PTX8.7 BF16 multiplication contract](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul) defines the packed multiply and default subnormal support used by that equivalence. Generic table construction performs no operation on supplied weight values beyond lossless packing.

All four mutable owners retain private weights, scratch, outputs and captured inputs. Generic tables are separate immutable tensors per candidate owner, held through graph lifetime. Wrong-arm mutation stays rejected. No barrier or fence moves; existing af parity and WGMMA completion protect register reuse exactly as before. No new timed launch, scratch protocol, routing input cache, dynamic work claim or public permission is introduced.

## Prospective test and stop

One private screen under the original295s guard, after the accepted public job exits and a fresh shared job/quota check is clear. Three fixed normal/skewed banks, all54scored/neighbor cases,378fresh/transition observations,216captures,1152cold timed replays,54restores and independent duplicate controls/candidates. Every bank/distribution/allfour cross placements must improve the complete-six metric by at least1% and exceed twice duplicate-log noise; any affected1%cell regression, unchanged-path drift, numerical/zero/guard/ownership/source/coverage/budget failure rejects promotion. A successful screen requires held-out confirmation before integration or another public request. No subgroup mask, LUT-size/layout/cache-policy sweep or automatic retry.

Preparation and final cleanup/report/budget failures retain bounded cumulative partial rows and preserve the failed status. Source generation preserves the original36row success report size; the actual previous target timing rows and variable-value size fixtures pass the strict reader and remain below the original14000character bound. No full-evidence success claim can come from partial coverage.

Four local CUDA12.8.1 SM90a device entries compile with exit0 and retained complete diagnostics. Both arms use96registers/16barriers. B4 control has8stack bytes/4spill stores/32spill loads; candidate has zero. MAIN control has zero; candidate has8stack bytes/8spill stores/36spill loads. These are local compiler observations, not H200 parity or measured performance. The decoder keeps32bit indices and computes each final64bit table address inside one PTX lookup. A preliminary unpublished pointer expression increased local spills; the frozen implementation uses the same four lookups with compact indices. This is one implementation qualification before the first GPU screen, not a target experiment or a cache-policy sweep.

Twelve CPU tests pass:4codec/source/gate,7actual Torch packing/ownership/fallback/failure/repeated-prepare, and1strict reader with injected failures. The Torch tests use an explicit CPU facade; their synthetic timing values prove orchestration only. The reader size fixture uses retained variable target rows under an explicitly synthetic receipt association. It proves serialization and strict rejection, not a new GPU result. Local source checks preserve WGMMA issue/fence/full-drain counts; the archived full SASS and logs are available for inspection. H200 research hash validation still reports the changed ruled-out pin; it was left unchanged and actual current negatives were read.

## Frozen source and execution

| Item | Exact identity |
|---|---|
| Private payload |542cd6506c72fa4823abd79217d2817f5647ebd6675c0ab1c222e0b2ba427036,541745bytes|
| Runtime |4fd6296182b841b9a6f9adea41c4d8c08c044369320f0fcc478bc22f5118ec25|
| Native candidate |8c2cfc7cd87dd863bf32cd6389b9f056e954329201fee1d7011a39efe5ce31c8|
| Native control |c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71|
| Generic table |3c86f4d8a36e8441712e4b3245d519dc54a4c5a0316bff46c365f3e5e96e6f82,65536bytes|

[Payload and full compiler/packing pins](evidence/2026-11-02-moe-pairlut/paired.manifest.json), [source](evidence/2026-11-02-moe-pairlut/paired.py), [local resources](evidence/2026-11-02-moe-pairlut/resources.json), [archive manifest](evidence/2026-11-02-moe-pairlut/archive-manifest.json). Runtime requires all7registered compiler requests observed and owner table lengths256/256/16640/16640 with identical original prefixes. Persistent verified compiler caching is allowed; mutable input/state caching is not.

No target outcome is known. Public approval covered5df257af only. Author checked these claims against the frozen source and outputs; independent review was unavailable under the user's no-new-agents instruction. Publish the exact source, verify remote bytes, then perform fresh shared jobs/quota preflight and one private POST. Record the accepted own ID before any overlapping work. A successful matched screen needs independent held-out confirmation; an unsuccessful screen stops this exact treatment.

## Accepted private screen

Ownc985bbdb2980407a accepted for frozen542cd650/541745bytes, published5ff38524. Fresh shared list hadzero live jobs; MoE checks12left before dispatch. Exact accepted ID and source association are archived. No target result yet; collect only this ID with one foreground waiter.

During the accepted job, the offline reader gained failed-partial support. One additional CPU test covers source/job/runtime binding, coverage bounds, incomplete evidence and a complete favorable timing table attached to a failed setup. Both remain unqualified. The existing full-reader test also passes after this change. The frozen runtime, payload, manifests and gates are unchanged. This raises the author-checked CPU test count to13.

## Terminal decision, stop exact pair lookup

Ownc985bbdb2980407a passed5official tests/6benchmarks/6profiles, all54private cases,378observations/216captures/1152cold timings/54restores. Prepare276.790803s. All7compiler requests and256/256/16640/16640table owners match the frozen source. Worst global/row error is0.000089915/0.000183763, within unchanged0.015/0.075bounds. Cleanup/restoration passes. Solewaiter73200ended0 before archive.

Performance rejects the exact treatment:0/6bank/distribution groups, all18affected cells regress,24complete-six factors0.473011105–0.576767080, no unchanged-path drift under the registered duplicate-noise gate. Official candidate2 score12337.764699us/1.090027272x CI[1.073676863,1.106377681]. This is not a release lead. The current public remains5788.720720us.

| Tokens | Official adjusted us |
|---|---:|
|1|25.380326|
|16|93.750820|
|32|548.533231|
|8|68.863805|
|128|828.671003|
|256|1006.233880|

[Complete owning receipt](evidence/2026-11-02-moe-pairlut/status-wait.raw.json), [strict analysis and36rows](evidence/2026-11-02-moe-pairlut/analysis.json), [summary](evidence/2026-11-02-moe-pairlut/summary.json), [full profiles](evidence/2026-11-02-moe-pairlut/profiles). No retry, LUT-size/layout/cache-policy sweep, subgroup mask, integration or public action. This rejects the frozen full-pair-codebook implementation; the data do not isolate whether lookup latency, traffic, dependency or resource changes caused the loss.

Continue toward5559.8us with a different work-elimination mechanism. The next source check is direct phase-A routed-input staging, eliminating global xA materialization and its cross-CTA readiness exchange. Older gx_pre/gx_st16 only rearranged the global gather and did not remove it; those exact variants stay closed. The existing invalid observer provides no removable-time estimate for this new alternative. Preserve the current public release and General WS ownership.
