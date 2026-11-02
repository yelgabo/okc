# Direct-weight result, current-runner control and next broad backing candidate

November 2,2026, terminal23:26:37UTC; boards/quota23:33UTC. Evidence checkpoint **b325b965**, direct implementation **368b85c9**, registration **64cd5ba0**. **OK Driver remains captain/sole judge dispatcher**; existing OKC Developer is implementation owner. Old T3 captain stays user-stopped/paused.

**Own private layer `0b684a4065354fc5` is valid but unfavorable:1.9264235913174548x,95%CI[1.897527237447693,1.9553199451872163], versus public2.1260826764231093x. PARK this fixed direct-weight implementation.** All13 tests/12benchmarks pass,12complete profiles,56startup replays and actual cooperative admission succeed. Developer explicitly acknowledged no confirmation/tuning/poll/POST, and held native pair **b325b965** is stopped without hold removal or a second check. No public/default change.

## Terminal evidence and practical limits

Physical875672LF sourcea27a1baf, nativef95fbb11, servicecontentb2bf4281a26819c6a77c985cbf67b0ba70a2f6228dcf97dabd4f0f581be27fa7 are associated separately through the owning receipt. H200/node`n-td72fo4wauotceojbgm4smtst7`, driver580.159.03, clusternyc3b; preparation **109.370262382s**. All three DW admissions show544threads/grid132/resident1/dynamic120576/static0/register96, actual local bytes24B4/16MAIN. Those differ from secondary compiler stack reports; neither secondary spills nor these local bytes diagnose the loss.

Normalized46:16 endpoint microseconds:

| Load | Direct weight | Fresh unchanged607 `cc474883117947c0` | Descriptive change |
|---|---:|---:|---:|
| D1 |78.8599|78.7645|+0.12%|
| D16 |156.0925|152.0162|+2.68%|
| D32 |221.3662|191.8978|+15.36%|
| V1 |118.5978|112.6817|+5.25%|
| V16 |335.9725|261.7130|+28.37%|
| V32 |442.7975|351.0802|+26.12%|

Independent review recomputes the exact official score and all six endpoints. Fresh-control/candidate geometric endpoint factor **0.889351467**; public score factor0.906090630. Evaluator-file hashes match, but inputs, node, cluster and driver differ. **Descriptive continuation screen, not matched causal loss or global-memory/spill attribution.** Scope the stop to this fixed source, not all direct-register designs. Full [raw receipt](evidence/2026-11-02-driver-direct-weight-first-check/status-wait.raw.json), [calculation](evidence/2026-11-02-driver-direct-weight-first-check/terminal-analysis.json), profiles/log manifest and hotspot map are archived.

The three captain-directed body experiments now have valid correctness/admission and unfavorable speed results. We have no new winner. Repeating different pipeline arrangements without checking the current control would not resolve the remaining measurement uncertainty.

Fresh23:33UTC: **overall/layer376**, incumbent607unchanged, layer1002.2690x; **7private layer/10scored slots and no live jobs**. Layer numeric gap is~6.72%; overall API gives ranks only. [Live summary](evidence/2026-11-02-driver-direct-weight-first-check/terminal-board-summary.json).

## P0: one unchanged-incumbent calibration on the current service

Allocate **ONE ordinary private layer check** of the exact public607 implementation now. Retained freshcc474 used driver595.91.07/node`n-434...`; all three new body checks used580.159.03 on differentnyc3b nodes. We verified those identities and avoided causal claims, but do not yet have a fresh unchanged control from the current service pool. This check validates the instrument/working baseline before another architecture result; it is **not an optimization, causal comparison or reason to reopen parked candidates**.

Prepared [control.py](evidence/2026-11-02-driver-current-incumbent-control/control.py) is **869198LF bytes**, SHA256 **04314b16743988c5b172a945e0f10437d78a70bb847ddd865f4a5c1101d6b305**. Its original869076-byte ba9 prefix, every original ASTnode, nativec7, compiler choices, prepare and timed paths are unchanged. Only a comment and unused metadata assignment are appended to distinguish a fresh request; [manifest/proof](evidence/2026-11-02-driver-current-incumbent-control/source-manifest.json).

Captain alone dispatches after commit/remote/physical identity and fresh authenticated quota/no-live checks, then follows only its own ID. Require no duplicate/historical substitution, full13/12/12,56startup, actual source/backend/GPU/node/driver/evaluator and all six endpoints. Scheduler may choose a different node/pool: label the observed identity rather than pretending it is matched. Update current profile/uncertainty; no speedup/public-rank improvement is inferred from a control's score. No automatic retry. No source-only secondary recompilation is needed for the unchanged shader; the target run is the new environment qualification.

## P1: actual selected dense-bank GENERIC backing, one fixed implementation

Allocate existing OKC Developer **one60-active-minute implementation-or-STOP window**, while calibration runs. Own `tools/dense_generic_backing.py`, focused tests and worker-dense-generic-backing report/evidence. Start immutableba9; reuse the reviewed final graph-owner/owner-event/helper-local-RELAXED protocol, with **explicit C++20** and generalized exact tensor byte/dtype/shape handling. Do not merely retry PR alone. **Reserve one ordinary private full-layer check** when this fixed broad candidate is conforming/runnable and the calibration's outcome reconciled; captain alone decides/dispatches after recorded exact source/fresh quota/no-live. No public/default permission, new comparison framework, allocation variant or compression-layout tuning.

Change allocation backing only, preserving every packed byte, escape record, base4, matrix layout, selected native source, flags, algorithm and math. The actual layer's dense BF16 weights are Gaussian-scaled, unlike uniform NVFP4 codes. Short paths already use lossless12-bit streams: their hardware compressibility is unknown, not established by entropy or an allocation flag. Actual reader inventory:

| Actual banks | Scope/size |
|---|---|
| QKV shared qkv8/16/32 stream |27MiB, shared across names |
| O o8h_r88/o32h_r100 |2x12MiB |
| Fused pooler stream |48MiB |
| UN slice8_r192/unpooler16h_r100/unpooler32_r144 |6+48+48MiB |
| Fused router hi/lo stream |8MiB |
| Actual Lt QKV/O/UN/PR |36+16+64+72=188MiB |

Eight selected short streams total209MiB plus seven escape tables. Preparation additionally constructs two48MiB pooler streams:305MiB/ten prepared streams plus nine escape tables. **Fixed implementation maps all prepared SK2 streams/tables once at their producer, preserving aliases, and all four actual Lt banks**. That is493MiB logical copies before tables/granularity/temporary duplication (397MiB would cover only the selected209+188). Keep the separate FP32 GLE router ordinary, not reduced in precision. Empty tables need explicit safe handling rather than zero-byte VMM allocation. No physical footprint/compression-ratio reduction is claimed.

Install mapped streams/escape tables **inside `bfp_sk2_prepare` before first self-test/tuning reads**. Replace actual `_graft_lt_plan` operands including NN transposes and specialQKV256 construction; changing only original weights misses these copies. Guard actual readers `bfp__run_sk2` including saved`_DE_RUN`, `bfp_fused_pooler_router`, `_gle_frt`, `_graft_lt_run`; a final whole-entry wrapper misses preparation and saved aliases. Retain owners before capture and record eager completion in`finally` after the true enqueue, including partial-begin/error cleanup. Preserve graph clone/executable/inflight lifetime, owner-specific events, callback restrictions, balanced context leases and helper-only capture-mode restoration. Never restore context-wide synchronization or unsafe unmap. Reuse existing critical protocol code rather than inventing a new cleanup model.

Preserve immutable owner-bearing SK2 tuples/tagged Lt weights through deepcopy; separate mutable counters/workspaces/caches. Pointer/dtype/shape/stride ownership must work under inference mode without silently bypassing version checks. Deduplicate actual shared storage rather than allocating once per alias. Generalized helper computes checked exact contiguous byte lengths and preserves original dtype/strides; BF16 PR's hardcoded4608x8192/numel*2 is insufficient. Copy on the correct producer stream and verify bit identity before publication. New owner/identity failures must be fatal for this fixed treatment despite broad backend-fallback catch sites; do not silently benchmark untreated/fallback banks.

Require actual hardware capability and accepted GENERIC properties for intended mappings, actual reader selection/owner receipts, copy identity and full qualification. Unsupported/no-treatment is a capability STOP, not numerical or compression-performance evidence. Acceptance flags do not establish useful compression. Source/CPU tests must exercise producer-before-first-use order, all reader/saved-alias routes, dedup/deepcopy/inference, eager/capture/error/cleanup and failclosed fallback. Reuse existing full entry/facades where helpful; no general instrumentation project. Check effective target C++20 build contract/host syntax, preserve source1MB limit/notices and295s deadline; local headers are not target ABI proof.

Freshcc474 dense-containing activities aggregate38.706/49.171/54.239/39.842/70.790/87.697us over D1/D16/D32/V1/V16/V32. If10% became **net endpoint savings**, score would rise3.401%;3% requires~8.857% proportional conversion. A short5%/long20% scenario gives3.047%. These are conditional sizing, **not additive exposed-time budgets, compression forecasts or promised gains**. Useful hardware reduction on already12-bit short streams, raw BF16 long banks and added owner/preparation costs is the funded question. It covers all six loads and is broader than PR-only's~1.61% halving scenario.

Deliver one plain pinned full-layer candidate, actual selected-bank/source/flag/owner inventory, CPU and host-build evidence, or an exact hard blocker/held checkpoint within60active minutes. No silent extension into research. First target host/lifetime/admission/numerical failure or no-treatment closes this fixed allocation without rescue; valid weak/negative performance parks without variants. A substantial favorable first signal earns a separately recorded matched full-layer decision, never automatic release. No proof of a win is required before the funded first check.

## Parallel source conclusions

The incoming weight/routing-overlap proposal is STOP for current material ROI. Whole-layer top-k/local IDs come from current routing after earlier dense operations; unlike standalone MoE they are not independent immutable inputs. `S.jE`/ranges/useful weight addresses therefore cannot be computed before dependency completion. Global scratch/epochs are also reused. Only small CTA-local initialization is clearly independent, without a substantial budget. Smaller shared footprint alone does not prove concurrent/cooperative admission. CUDA13 documents cooperative and programmatic attributes but the bounded review found neither an explicit incompatibility nor target proof of the combination. This does not reopen outgoing seam failures or ban every PDL edge. See [repair inventory](2026-11-01-g-native-repair-inventory.md) and [CUDA13 PDL guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization).

Held direct pair b325b965 has five passing complete-entry CPU tests and explicit HOLD/dispatchfalse; target fit/math/timing remain unknown. Preserve it, do not continue it after the first result's stop. Delivery/pickup of the new broad-backing assignment and calibration's own accepted ID will be recorded separately. This report claims neither a fourth optimization result nor a new winner.
