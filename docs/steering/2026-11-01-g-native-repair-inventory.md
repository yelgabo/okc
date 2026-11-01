# G0 to 607e: native expert repair inventory

November 1, 2026. Source-only audit after branch evidence `88ba394`; no GPU,
service call, production change or submission. Dense/router/attention inventory
is independently owned by today_proj. This report audits expert scheduling and
its interfaces, rather than assigning the entire score difference to repairs.

## Finding and scope

The observed headline ratio is 2.248790532 / 2.126082676 = approximately
1.057716. It is **not a measured 5.77% native synchronization tax**. The jobs have
different inputs, routing, baseline timings and bundled precision changes.
G0 check `3a5f879df0d44825` is the comparison here; another author's rank/score
or standalone MoE job is not that layer receipt.

The largest credible *location* of unexplained cost is the short native expert
path, not main-only MFIN. Root's owning-receipt ledger gives weighted D1
68.203874→78.549045 us and V1 103.618625→115.990823 us. Substituting both old
whole-load times gives conditional six-load factor 1.043244918, not a forecast.
Peer's expert-row-only cross-profile scenario is 1.05696168; it is neither exposed
time nor additive with the whole-load scenario. Short dense is already faster
in that comparison. No retained matched experiment separates the short expert's
precision, reader-release, admission and PDL changes.

## Exact source and reproduction

Upstream: keell0renz/muse-open-kernel-challenge commit
`8e30e1be994db4310794b8d04e90d18d895deb84`, layer raw SHA
`1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a`,
LF SHA `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
Current `muse_decode_layer/cand/g1b4_stage_release.py`: LF 869076 bytes,
SHA `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`,
owning public job `607eb52bdaf24160`. Repaired G1b2 source
`258c848c662149f9c6ac178039f179de6f0b0ec4cb0b503451d2974c7a7a8dfc`
belongs to `736a5cbc977f4ad4`, not G0.

`tools/g_native_repair_inventory.py ROOT --out FILE` regenerates the attached
source-ledger.json without importing either submission. It checks both complete
source pins and records literal hashes, base flags and macro definitions. It
deliberately does not claim base assignments alone establish final dispatch.
The upstream private cache is a reproducibility input, not a runtime file read.
The tracked incumbent and upstream pin remain unchanged.

Active `_fork_native_moe` selects:

| actual token count | selected module | source/algorithm |
|---|---|---|
|1|tiny, max4|helper0, WS skinny, NW8/FAST1=2, FINPF1|
|8|tiny2, max8|helper0, WS skinny, different WSBLK1|
|16|mid, max16|helper0df alias **repaired helper0**, SPLIT1|
|32|b4|main WGS plus B4FIN1|
|128/256|main|main WGS NW16/FAST1=0|

The existing shape flags remain; no new geometry was introduced by this audit.
Tiny/Tiny2 do not select the standalone donor's dedicated M8 ring issuer.
SC6 is added only when state `ma_sc6` and the selected definitions have WGS1.
The active main CUDA literal defines **WGSFCI4**. A helper literal contains3
and a comment mentions a CI3 scout; neither establishes active MAIN's value.

## What changed, and what is actually required

The exact transforming source is `tools/dl_fork_moe.py` (LF SHA
`1350676ca93663b1379ed6520381c20b6c7b164ef2c45aabccafe39e62266eed`),
followed by attributed-layer composition and `dl_g1_stage_retire.py`. Full
baseline/official-rule reasoning is in
[the precision contract](2026-11-01-g-fork-precision-contract.md).

|change|classification and restoration constraint|
|---|---|
|GEMM1 gate/up BF16 before SiLU; BF16 SwiGLU; scaled GEMM2 BF16 **before** gamma|Required baseline stage-boundary parity. Restoring old unrounded intermediates changes values; not a safe performance restoration. Both short helper and main affected.|
|Effective gamma BF16(1+rawBF16), then FP32 norm arithmetic|Required by this baseline-parity release gate; not a universal claim that every internal accumulator must be BF16. No isolated H200 cost.|
|YBF1→0 in main|Conservative removal of an extra post-gamma quantization. Distinct from required pre-gamma round; no measured YBF-only cost and no certified restoration. Inactive on WS shorts.|
|All-reader shared proxy fences and NT-counted weight/X releases; short reader bridge|Required publication/lifetime repair of the chosen protocol. Lane0 cannot fence generic reads on behalf of other reader threads. Test success cannot authorize reverting these.|
|Global xA/xB generic-store proxy bridges, including split-finisher post-store bridge|Required before this in-kernel generic→async handoff. Flush-entry fence alone cannot publish stores made later by the finisher.|
|Acq_rel GPU atomic full-grid rendezvous|Explicit transitive publication chain retained with CTA/acquire joins. A weaker RMW is not a proved alternative merely because the old source passed.|
|Checked cooperative admission, fixed bounds, ordinary same-stream launch|Repairs unsupported software-grid progress assumption. Resource admission is not achieved occupancy. Earlier dependent primary must complete; cannot blindly combine old secondary-PDL launch with full-grid residency.|
|Global layer PDL off|Conservative whole-pipeline reference policy, broader than a proof that PDL is inherently invalid. A particular edge may be requalified only with exact producer publication, consumer wait, descriptor ownership and resident-grid progress proof. No such short paired edge proof exists in this ledger.|
|MFIN forced0; full join retained|Old flag poll may fall through after4M unsuccessful iterations. The *implementation* is unsafe. Distributed finalization is not inherently forbidden, but re-enabling the flag is not a repair. Main/B4 only.|
|helper0df sentinel alias→helper0|Conservative exclusion of a separate unproved publication protocol, at M16. Cannot explain D1/V1. Safe restoration would require independent sentinel generation/publication/retirement proof.|
|SC6|**Retained**, not a removed optimization. Lossless six-bit scales where admitted; byte fallback otherwise. Code payload unchanged, scale payload reduced. PRMT eligibility and state-local packing/defines must agree.|
|Deferred GLE two-row path off|Stage-round changes and carry read-before-wait defect; cannot restore unchanged. `GLE_T={}`/PDLFalse makes it dead; active FRT at actual T rows is a separate path.|
|MFAB_LAYER off and native state-local caches/scale format|Preparation/ownership correctness, not a measured timed-kernel cost. Removes contaminated A/B setup and incompatible layout selection.|

The publication arguments use [PTX8.7 memory consistency and proxy rules](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model)
and [CUDA cooperative launch admission](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__EXECUTION.html).
They establish obligations, not a measured price for each repair. Older G's
watchdog/progress and legal-input bounds are separate eligibility questions;
this audit does not upgrade old passed jobs into formal proofs.

## What existing measurements can attribute

|evidence|supported conclusion; limitation|
|---|---|
|`63f5a2a8ed2341d1`, four-arm matched stage retirement|Six-load factors1.018493–1.028699 on three heldouts. Restores real wait1 overlap with full current-stage drain/readers before X release. This recovery is **already in607e**.|
|Actual stage SASS|G0-grid/MFINoff has512HGMMA/256wait1/16wait0 per entry; repaired serialized control512/0/512; stage512/256/48. Thus compiler serialization was real and partly repaired, not evidence every extra fence costs a given latency.|
|`1c6ebc76b91d4b7f` fixed-owner frontier|Optimistic modeled six-load max1.003616/1.004271/1.003445 by seed on128/256. Not a broad MFIN ceiling, but refutes assigning the whole gap to that fixed-owner final wait. Does not measure B4 or shorts.|
|`05e51ddeb7014dd9` SC6/byte four-arm|Factors.999807/1.007119 fail noise/regression gate. No repeatable standalone SC6-first win; G already uses SC6.|
|`835253441a214a1c` owned multiply|Fewer static decoder instructions/registers did not clear full-score gate. Different owned implementation; no causal estimate of G repair cost.|
|`f147d06a64504c1e` observed-ready/retire probe|All12 attribution rows fail distortion gates (kernel4.91–9.54%, phase10.26–14.90%). No valid main-loop weight/X/retire bottleneck attribution.|
|`27b2ed49fc6344ec` pending-flush placement; `942ebdec400d4e68` split overlap|Both public losses vs607e. Exact placements closed; not causal disproof of every publication or overlap design.|
|`a5c7c7e3fe344919` direct band32; `903e5b…` cached band32|Owned locality/representation variants closed. Neither reopens from this source inventory.|

## Next action and stop criteria

There is **no proof-backed large restoration ready to enable**. Prioritize a
short Tiny/Tiny2 expert-specific causal gate over another main finalizer design.
It must preserve BF16 boundaries and all-reader release, use checked cooperative
admission, natural heldout routing and four private owners, and measure the full
six-load function. Do not time unsafe old math/order as a candidate arm.

One minimal exact alternative is to precompute immutable effective gamma in
preparation: BF16(1+raw.float()), store it in a separate BF16 weight tensor, and
replace each repeated `fork_bf16(1+raw)` expression by conversion of that stored
BF16 value. Keep all GEMM2 scaling/rounding and final arithmetic ordered exactly
as now. Pointer type/length stay unchanged; raw supplied weights and Marlin
fallback must retain their own ownership/parameter semantics. This is constant
hoisting, not permission to merge multiplication across BF16. Prove every BF16
raw bit (finite-domain qualification separately), all direct/ldg/uint32 load
sites, and all five module preparations before considering a source change.
No implementation or timing is claimed here. It saves neither weight payload
nor activation traffic; existing prefetch already amortizes some expressions.
It therefore **does not yet justify a >=1% check**, much less explain5.77%.

A meaningful next short-PDL proposal instead needs a concrete edge proof:
producer completes all published BF16 outputs; secondary waits before reads;
immutable descriptors survive replay; cooperative expert grid does not require
concurrent residency of its unfinished producer. If that cannot be proved, keep
same-stream completion. No blanket flag restoration or fence-removal A/B.
Require >=1% whole-six, >2x duplicate spread, each heldout seed and per-load
regression gates; stop before GPU if static/source upper opportunity cannot
clear them. A per-repair matched causal budget remains the decisive missing
evidence. Historical headline score and current607e remain unchanged.
