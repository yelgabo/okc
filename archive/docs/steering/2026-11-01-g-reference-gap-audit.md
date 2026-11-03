# Family G reference-gap audit: location established, mechanism only partly isolated

November 1, 2026 local / November 2 UTC. Captain evidence checkpoint `7456f0c`; this is an audit, not a new GPU allocation or performance candidate. Reconciles the earlier G0/607 inventory, current common OFF, the valid RMW null, and the new selected-short compiler audit. No kernel/default change or public submission.

## Executive answer

Our G implementation is not merely the reference with different packaging. We preserved its selected short geometry but changed its numerical dependency chains, read-retirement/publication protocol, admission and pipeline policy. The strongest observed gap remains T1/T8. New device-only compilation finds **the same short-path resources but more instructions and a longer gate-to-exp dependency**, not a missing tile choice or a new spill regression. We still cannot honestly price each change or call the full score difference an unavoidable compliance tax.

The optimization process also contributed: we spent the latest measurement tranche testing one narrow counter-ordering mechanism without first attributing the larger short-path loss. Its valid matched result is negative. That does not vindicate every other repair implementation, and it does not create a candidate. The audit must produce a specific cheaper implementation or a specific dependency-edge restoration, not another generic instrumentation campaign.

## Exact comparators and independently reproduced arithmetic

- Complete G0 donor: keell0renz/muse-open-kernel-challenge at `8e30e1be994db4310794b8d04e90d18d895deb84`. Fresh branch/source lookup November 2 03:27:39.822 UTC confirms this same revision. Served LF file 979654 bytes, SHA256 `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`; retained CRLF file 995999 bytes, SHA256 `1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a`. Complete LF-normalized files match. No new donor discovered.
- Our public stage release: `muse_decode_layer/cand/g1b4_stage_release.py`, 869076 bytes, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
- Current repaired-common OFF: `4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280`. The paired diagnostic upload is a different, instrumented file; do not substitute its helper literals for production factories.
- Owning raw G0 private receipt `3a5f879df0d44825`: PASS13, score **2.248790532234**, 95% CI [2.215058674250, 2.282522390217]. Our own exact-file measurement, not the author's team best.
- Owning raw public receipt `607eb52bdaf24160`: PASS13, score **2.126082676423**, 95% CI [2.094191436277, 2.157973916569]. Ratio G0/607 = **1.057715467593**.

New offline [reconcile.py](evidence/2026-11-01-captain-g-reference-audit/reconcile.py) reads the two original API receipts, verifies job/board/mode/pass/return code, all13 tests and12 benchmark/profile pairs, reconstructs both official scores, preserves exact64-bit seeds as decimal strings, and asserts every non-seed case field agrees. [Output](evidence/2026-11-01-captain-g-reference-audit/reconciled.json), [upstream pin](evidence/2026-11-01-captain-g-reference-audit/upstream-pin.json).

**Important confounders independently checked:** same evaluator code hash and H200 GPU model, but different physical node IDs and all12 benchmark seeds differ. Therefore different expert routing/weight banks and device/calibration conditions remain. This is not a matched causal comparison. No power or temperature snapshot is used as a timing correction.

| Weighted whole load, (46 sliding +16 full)/62 | G0 us | Public607 us | Public extra latency |
|---|---:|---:|---:|
| Decode1 / T1 |68.203874|78.549045|15.1680%|
| Decode16 / T16 |149.700710|158.241010|5.7049%|
| Decode32 / T32 |190.719032|191.739736|0.5352%|
| Verify1 / T8 |103.618625|115.990823|11.9401%|
| Verify16 / T128 |258.670706|269.347029|4.1274%|
| Verify32 / T256 |350.796973|359.297216|2.4231%|

Replacing just the two short times by G0's historic times, with607's other loads and baselines held fixed, gives **1.043244918072**. Replacing all six gives **1.065243721**, not the official1.057715468 ratio because baseline observations also differ. These are conditional arithmetic scenarios, not attainable speedups or proposed hybrid scores. They justify prioritizing short paths over another main-only finalizer study, but do not promise that recovering short paths is sufficient for top100.

## What the profiles locate, and what they cannot prove

The expert profile row is slower in both short contexts but very unevenly: D1 sliding23.6 ->29.8us, D1 full21.8 ->42.8us; T8 sliding44.2 ->71.6us, T8 full57.6 ->58.8us. Larger cells are mixed, including D32 sliding81.7 ->70.6us and T128 sliding161.1 ->155.3us. Exact rows are in the new output. This is consistent with short expert work being a priority, not a uniform measured fence tax.

D1 sliding also changes GLE O+deferred boundary into ordinary O plus an explicit boundary; the latter appears as3.4us in607's profile. G0 and607 total profiles are73.1/81.6us while adjusted whole-load medians are66.3525/76.7481us. **Do not sum profiler deltas into benchmark savings:** profiles are rounded activity samples from independent inputs and can include different overlap. A component duration is not its exposed critical-path time.

## Which explanations survive source reconciliation

| Explanation | Finding | Decision |
|---|---|---|
| Lost short tile/warp tuning, missing SC6 | Actual Tiny/Tiny2 geometry/flags retained; SC6 remains selected where eligible. | Not a supported explanation. Do not renew a tile/mask sweep. |
| New short spill/occupancy cliff | Selected donor/public secondary device compilation has equal147 registers,3 barriers,0 stack/spill and launch bounds288/min1 for both bool entries. | No resource regression in this screen. Not a target-NVCC or runtime-occupancy proof. |
| Changed expert numerical/protocol dependency | BF16 gate/up before SiLU, scaled-down output before gamma, effective gamma, and reader/publication implementation differ. New SASS contains extra conversions/fences and a gate conversion/widen before EX2. | Concrete remaining mechanism, dynamic price not isolated. Required values/order must survive optimization. |
| New SwiGLU output BF16 cast | The output cast already existed in G0. Added operand boundaries are different. | Earlier simplification corrected; no mythical newly inserted output-round cost. |
| Blanket PDL-off / admission policy | Donor contains ordinary/PDL launch alternatives; our expert uses checked cooperative admission, and pipeline-wide PDL is off. | Conservative policy may leave overlap behind; incoming and outgoing dependencies must be considered separately. No global flag restoration. |
| Final counter's acq_rel RMW alone | Corrected paired run has valid complete capsule and all whole-six crosses below promotion threshold/noise guard. | Exact RMW-only lane closed. It does not explain or eliminate other numerical/read-retirement costs. |
| Main-stage serialization | Matched stage retirement recovered1.018493–1.028699 in63f5a2a8ed2341d1. | Already in607; not an unclaimed new recovery. |
| Dense integration broadly ruined | Short QKV/unpool geometry unchanged; all8 saved Lt plans accepted. Large T128/256 RQ routes replaced by Lt, separately from short changes. | No missing-plan fallback found. Large-RQ restoration cannot explain T1/T8 gap and has separate precision/progress obligations. |

Selected-source equivalence has a limited scope: current common OFF Tiny/Tiny2 source, CPP and full factory flags match607; current whole-layer preparation, attention/progress and pre-expert effective-gamma changes are later deltas. Do not attribute those later repairs to the historical G0/607 difference, nor infer whole-file performance equivalence from equal short modules.

## Latest experiment reconciliation

Corrected private `c5e4f89672f743bb` / upload785a90ca, archived339523d: PASS13/12 benchmarks/12 profiles; frozen reader745d7bb5 complete=true, private_raw_gate=false. All12 bank/cross whole-six log gains lie from -0.001829079 to +0.002828144, versus duplicate-null spreads0.00283197–0.00355555; every bank fails the registered1% whole-six rule, and993821 also fails the context regression guard. No fourth check or seed rescue. Official repaired-common OFF2.112499365 is descriptive only, not an ON result or a matched estimate of607 drift.

The universal SiLU table is [eligibility HOLD](2026-11-01-universal-bf16-silu-table-screen.md), publication5fdbd56, before implementation/compiler/GPU. The [BF16 staging alias](2026-11-01-alias-exposure-and-bf16-staging.md), publication7b8d4b0, remains design-only with unmeasured exposed timing/helper capacity. Neither is a qualified candidate or an explanation measured against G0.

## Source evidence and corrections retained

Prior inventories remain supporting evidence, not new experiments: [native repair inventory](../../../docs/steering/2026-11-01-g-native-repair-inventory.md), [minimal native ladder](2026-11-01-g-native-minimal-delta-ladder.md), [dense delta](2026-11-01-g-dense-minimal-delta.md), [precision inventory](2026-11-01-g-upstream-precision-inventory.md), and [original integration gap](2026-11-01-g0-607e-integration-gap.md). Their older RMW-pending recommendations are superseded by the valid terminal result.

No isolated per-repair latency decomposition exists. No inference that every conservative choice is necessary, no claim that old donor test passes prove ordering/eligibility, and no claim that the unchanged147-register secondary build proves exact target compiler equivalence. The project still has zero qualified new winners from the latest tranche. This audit narrows the next experiment; it is not itself a speedup.

## New selected compiler finding and precise lifetime interpretation

[Worker compiled audit](2026-11-01-worker-g-short-compiled-audit.md), publication337cfe7bc6081d8f7570199e3535534f8177e81d, verifies the actual selected factories, not a dormant helper literal. Public607 and current common OFF short CUDA SHA b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716, CPP and complete compiler flags are byte-identical. Tiny uses WSBLK2, Tiny2 WSBLK1/CPF0. Donor resources matching these exact factories were not in the scoped archive, so four fixed device builds were performed, with both bool entries; actual T1/T8 entry is false.

| Secondary CUDA13 compiled selected entry | G0 |607/common OFF |
|---|---:|---:|
| Tiny non-NOP instructions |5194|5234|
| Tiny2 non-NOP instructions |5241|5283|
| Registers / barriers / stack / spill stores / spill loads |147 /3 /0 /0 /0|same|
| Launch bounds max threads / minimum CTA per SM |288 /1|same|

Both families add16 scalar BF16 conversions. The gate chain becomes scaled gate -> BF16 round -> widen -> log2 multiply -> EX2; independent up rounding remains separate. This confirms an added dependency in emitted code, **not16 conversions per dynamic unit or40/42 extra cycles**. Counts cover all paths in an emitted entry. Secondary NVRTC/PTXAS13 with retained declaration/header adapters is not actual target NVCC13/host-wrapper equivalence; no compiled artifact was executed.

Two extra emitted proxy-fence sites are phase-A/B copies of one skinny-ring reader-retirement bridge: generic ring read -> register compute/reduction -> partial-buffer wait/store -> per-reader proxy bridge/CTA join -> asynchronous refill. They are not merely unnecessary publication of S.red, and their lifetime role must survive any rescheduling. No cheaper proven bridge has been found. The counter's separate acq_rel instruction sequence must not be conflated with them.

## Chosen next action: one outgoing-only overlap seam

[Researcher selected-entry and dependency audit](2026-11-01-g-short-arithmetic-and-pdl-seam.md), publication478f85e123617b133775c0eb85cd6f40e9ade468, finds a narrower alternative than global PDL restoration:

- Keep route -> expert ordinary completion and the expert's checked cooperative admission unchanged. No unfinished predecessor is introduced into expert residency admission.
- Keep expert arithmetic, all reader/writer fences, counters, join, final output and existing outgoing trigger unchanged. The trigger occurs before final BF16 output readiness; it is **not** permission for early X reads.
- Only T1/T8 unpool, actual slice8_r192/1024 rows/CL1: enable the consumer's programmatic launch attribute and its existing weight-only preamble, default two stages/zero pacing. No global PDL setting or stage sweep.
- Consumer waits for producer completion and executes the global-to-async proxy bridge **before every X TMA/read**; downstream final boundary remains ordinary completion. Descriptors, prepared state and early weight metadata remain private/immutable across graph capture and replay.

Source DAG has no expert -> consumer -> expert back-edge: expert can finish without the consumer, and all primary CTAs trigger/complete before dependent launch becomes eligible. Nevertheless, source structure and CUDA documentation do not establish support or useful overlap for this exact cooperative-primary/captured-secondary combination. Runtime graph edge qualification and possible lack of free resources remain open.

Captain independently read CUDA13.0 guide6.2.8.6 and6.2.8.7.1.2: programmatic synchronization requires producer-block participation, consumer completion wait, and a programmatic graph dependency rather than assuming an ordinary captured edge overlaps. Primary sources and precise code excerpts are retained in the researcher's report/evidence. This is not a categorical cooperative+PDL eligibility verdict.

**Decision and owner:** replacement worker1c3bc6ca receives one bounded25-minute SOURCE/CPU/offline candidate allocation after its audit publication. Implement only a private scratch treatment/control and source/real-entry tests if no categorical source/API blocker appears. Native expert source must remain identical; launch changes must be scoped to this consumer and cannot leak through a mutable global setter. Reuse A/B machinery. Do not require a predicted1% gain to build this concrete treatment, but do not call it fast before timing. No GPU/CLI/public/default action is authorized by this allocation.

This takes priority over an arithmetic scheduling variant because it leaves the repaired expert instructions and BF16/protocol boundaries untouched, and targets an identified conservative policy. It is not proven to have greater exposed time. If source/API/state ownership fails, stop that exact seam; otherwise next decision is a separately authorized single runtime graph/correctness/replay/full-six comparison against independently prepared common-OFF controls. Runtime qualification may share that measurement rather than spawning another generic observer project. If graph edges remain ordinary, X is prefetched before wait, progress/lifetime fails, or resources regress, reject before performance interpretation. A valid measured loss/subthreshold effect closes the fixed seam without pacing/tile/seed rescue. Public promotion remains separately authorized and evidence-gated.

Fallback **not allocated now**: one scalar exact-operator interleaving of independent up/down work around the gate EX2/divide chain; retain every numerical boundary/fence. The old packed-RNE19cfb4ee introduced a gate-on-up dependency and is closed; neither it nor gamma12.8/13 is reopened. Unchanged emission, extra dependency or worse resources would stop a new scalar scheduling attempt before GPU. No work on both candidates in parallel is requested.

## Handoff / validation

Offline raw-receipt reconciler passes all assertions and reconstructs scores to1e-12. Supporting worker source/compiler publication and researcher source/DAG publication are indexed alongside this audit. No new performance candidate has qualified; no scored/public/default change occurred. CPU/compiler facts, source protocol argument and unmeasured runtime/score hypotheses remain separate.

