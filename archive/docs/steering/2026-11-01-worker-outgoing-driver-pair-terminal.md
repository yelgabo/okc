# Repaired outgoing check: routing mismatch, campaign seam lane closed

November1 Pacific / November2 UTC, 2026. Own private **6f83ee875dd847c1**, muse_decode_layer/check/yelgabo, is terminal **done / passed=false / returncode112**. Setup failed **AssertionError: SRM_ROUTING_ARM_MISMATCH**. This is an invalid/incomplete qualification, **not a measured PDL timing loss**. Per the explicit one-repaired-check allocation, the **outgoing seam lane is CLOSED for this campaign**: no rescue, retry, new payload/check, fallback, public/scored/default change, or weakening of the equality gate. Zero new performance-qualified candidates.

## Immutable identity and owning evidence

Preregistration **c40930c5e49efb05ff7ac476e34c8b50fdc1541a** was pushed/all10 remote blobs verified before fresh authenticated preflight, exact rebuild and sole dispatch. Acceptance handoff **120a069d51c0ee93aa5eddbfc2edfd2dcc13cceb** was pushed/all11 remote blobs verified. [Registration](2026-11-01-worker-outgoing-driver-pair-registration.md), [owning receipt](evidence/2026-11-01-worker-outgoing-driver-pair/own-terminal-receipt.json), [transport](evidence/2026-11-01-worker-outgoing-driver-pair/own-terminal-transport.json).

- Physical upload SHA256 **ad314060c98ccdde81a1a38b9f7454d7e4fdbb185d178094e3f197f9c8baa75b**, **943363 bytes**, unchanged at terminal.
- Exact owning reader SHA256 **a79a59c58d4035f7a247bdc247145827d9c5c000078602f2d5f7df699851f2b9**, unchanged. Old failed f25 upload and543 reader remain immutable too.
- Server content **da049704a1a40c266af60cc2568b7e12966207a3ef7d664ef3c120250242ffd2** matches the owning payload_ref; it is not the physical SHA256.
- H200 node **n-434neka4hbrvz7ee7sd46ord7g**, sfo3, driver595.91.07. Evaluator **1ba270d841be831b06a5862f4094cd0d5072edfbd0f6e84b95f79248296367dd**. Node differs from failed9ad; no cross-node causal timing comparison is made.

Accepted November2 **05:55:55.771UTC**; started **06:11:13.093UTC**; finished **06:14:16.954UTC**. Queue917.321654s; worker183.860948s. **prepare_s=null**: worker duration is not measured preparation time or a295s-budget result. Own wait **47783** exited1 and is terminal; do not restart it. Final quota5private/5scored remaining is not authorization for another dispatch.

## Exact failure and what is not localized

The retained traceback is prepare -> _srp_prepare -> _wl_prepare, at the frozen cross-arm check:

    assert all(torch.equal(outputs[0][2],out[2]) for out in outputs[1:]), 'SRM_ROUTING_ARM_MISMATCH'

The third returned output is routing IDs in the selected layer return path. Last logged query context is **row10 / arm0 / T8 / full(window0) / seed993817**. All four row10 capture/query calls had returned before this assertion. The failing comparison arm, differing indices/values and upstream cause are **not logged**. The last query being arm0 does not identify arm0 as the culprit. The exact mismatch is observed; attribution to adapterON, PDL, native expert arithmetic, input ownership, nondeterministic routing ties or harness/reference behavior is not established. Do not reopen diagnosis as an implicit rescue allocation.

There is **no OUTGOING_DRIVER_REPORT capsule**, no official tests/benchmarks/profiles, no score or CI, and no retained complete private timing table. The frozen exact reader rejects the owning failed receipt with **exit1**. Classification: **complete=false; private_raw_gate=null**, not a valid negative performance gate. [Reader rejection](evidence/2026-11-01-worker-outgoing-driver-pair/frozen-reader-rejection.json), [compact terminal summary](evidence/2026-11-01-worker-outgoing-driver-pair/terminal-summary.json).

## Actual target evidence, bounded interpretation

This run progressed beyond the earlier graph-query stopping site. The surviving log has **44 query-context records**, rows0..10 with all four arms in each row. Every record is printed before querying; that print alone is not success. **Frozen control flow plus the later routing assertion** imply those44 canonical identity/driver graph queries and existing edge checks returned. This is useful partial target integration evidence, **not all48 intended graph captures, complete row coverage, retained exact node/edge identities or full qualification**. No native SEAM_DRIVER_ERROR / SEAM_DRIVER_FAILURE appears in the retained tail. It does not establish the cause of9ad or prove every graph representation is supported.

The actual pre-query prefix survives as complete JSON: **11 build/load census entries**, five primary backends, common resource vectors and original/adapter secondary resource vectors. This is now real target build/load/resource evidence, not CPU mocks. Both original and adapter query exports have **-lcuda** recorded. Selected secondary flags are **-O3 / compute_90a,sm_90a** with actual zero delay (no SK2_DELAY define). No runtime architecture/flag substitution was used. Repeated loader records are not independent compiler invocations; do not sum their overlapping wall durations into preparation time. [Full pre-query prefix](evidence/2026-11-01-worker-outgoing-driver-pair/target-prequery-prefix.json), [parsed build observation](evidence/2026-11-01-worker-outgoing-driver-pair/target-build-observation.json).

Logged resources match across all four placements:

| Backend | Registers | Local bytes | Dynamic shared | Active blocks/SM | Threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny/tiny2 | 145 | 0 | 214784 | 1 | 288 |
| mid | 152 | 0 | 214784 | 1 | 288 |
| b4 | 96 | 8 | 231168 | 1 | 576 |
| main | 96 | 0 | 231168 | 1 | 576 |
| original/adapter unpool, T1/T8 | 66 | 0 | 118304 | 1 | 416 |

Static shared is0 throughout. These are actual resource/export observations, not a performance explanation. The prefix source location is after resource/empty/occupied-empty assertions; those had returned. Per-arm approximate numerical checks for the current row precede the stricter cross-arm routing equality; reaching the equality does not satisfy the failed identity requirement. Cleanup is in finally, but final cleanup-success capsule evidence is unavailable.

The owning CLI .log artifact is a **16000-byte retained result.log_tail**, not full process stdout. It retains the complete census prefix and44 contexts but only two standalone BUILD_DONE lines. The11-entry prefix is the retained build census; missing earlier standalone lines must not be invented. Raw receipt/log and explicitly synthesized failed-setup TXT are archived at **muse_decode_layer/profiles/g-outgoing-driver-pair.6f83ee875dd847c1.{json,log,txt}**. TXT is **not a GPU profile**.

## Campaign decision and handoff

**Close the seam lane now**, for this exact campaign allocation, on correctness/instrument validity failure. Keep the raw0/raw1 controls, arm3-only decision, >1%/>2xraw-null/regression thresholds and all numerical/protocol/ownership obligations unchanged; none can be evaluated as a complete surviving result here. CPU26/26 and captain's independent11/11 remain source/control-flow evidence, not a substitute for this target failure.

Captain owns README/shared runs-decode-layer/ruled-out indexing. Suggested scoped ledger wording: 6f83ee875dd847c1 repaired-driver outgoing fixed pair failed cross-arm routing equality atT8/full/993817; no score/full qualification; campaign seam lane closed with no rescue. This does not establish that every atomic/PDL design is slow or mathematically incorrect. A different future direction needs explicit allocation; no next execution is claimed by this archive. [Terminal archive manifest](evidence/2026-11-01-worker-outgoing-driver-pair/terminal-archive-manifest.json) pins the exact owned evidence.
