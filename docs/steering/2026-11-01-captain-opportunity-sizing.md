# Captain opportunity sizing after the paired setup failure

November 1,2026 Pacific / November 2 UTC. Evidence: owning measured run **c1931f0c9e834485**, archive **516a93a4fc5fc882565ff0d67661b03d798e2480**; paired failure archive **d756f298b7f52cba8c0b6baf46497b4ee0701d22**. This is scope arithmetic for allocating work, not a new optimization or forecast. Previous goal turn made progress by archiving the stop decision and sending bounded diagnosis; this turn adds independently reconstructed opportunity sizing and changes the research priority.

## Current target and failed experiment

Official CLI snapshots at **November2 02:06:57UTC** show overall **329/1833**, layer **329/1173**, our public speedup **2.1261**, layer rank100 **Cekaru2.2562**, job **d91101bb0e774cbd**. The rounded layer cutoff ratio is **1.06119185**. Overall is separately queried; no universal ranking formula is inferred. These are time-stamped standings, not a fixed deadline target. Private results cannot change our public entry.

The [owning paired terminal report](2026-11-01-rmw-common-pair-terminal.md), published in d756f29, supersedes queue snapshots: **16b4db27421d4714 setup FAIL, passed=false, return112/CUDA719**, no official tests/benchmarks/profiles/score.237.581422s worker wall is not prepare time; prepare_s is null. No cleanup/resource completion capsule exists. Frozen reader rejects before performance interpretation. Two accepted slots used, no automatic retry or #3 reallocation; worker owns bounded source/CPU fixture diagnosis. The duplicate-ID/trap explanation remains unlocalized.

## Scope comparison

I extracted official adjusted medians and displayed expert/attention kernel durations for all12cells from the successful c193 archive. For each load, C=(46*sliding+16*full)/62. The scenarios subtract a chosen fraction of selected displayed kernel durations from adjusted latencies and calculate exp(mean(log(C_original/C_scenario))) over the six loads. Exact input values, source hash, board snapshots and limitations are in [scope arithmetic evidence](evidence/2026-11-01-captain-opportunity-sizing.json).

| Hypothetical duration reduction | Six-load factor under this arithmetic |
| --- | ---: |
|10% of expert kernel duration, all12cells|1.047290|
|15% of expert kernel duration, all12cells|1.072692|
|25% of attention duration, all12cells|1.039375|
|25% of attention duration, full context across all six loads|1.028885|
|25% of attention duration, full-context verify only|1.011299|
|100% of attention duration, full-context verify only|1.048041|

**These are neither achievable speedups nor physical upper bounds.** They assume fully exposed savings, no new cost/interactions and no regression, while mixing separately observed profile durations and official medians. Profiler instrumentation/rounding can distort the mapping. The100% row means zero modeled attention time, not a realizable kernel. It remains useful for seeing why a narrow scope should not become the entire top100 roadmap.

Within this model, bridging the present6.12% layer ratio requires approximately12.77% reduction of all expert durations,37.99% of all attention durations, or51.08% of full-context attention durations. Full-verify attention alone does not span it even at zero modeled duration. Earlier short-path sensitivity gave about1.11% whole-six for3us removed from both short loads in both contexts; that remains a potential contribution, not the whole goal. None of these comparisons identifies removable critical-path work.

## Allocation change and next executable gates

Researcher was sent these numbers and asked to prioritize **broad legal expert/mainloop or all-context mechanisms**, comparing complete implementations before grafting. Their verify-exp idea can still earn a place as a low-cost contribution; ask whether it also applies to selected expert activation paths with required FP32/BF16 values intact. A large profiler share alone is insufficient: next candidate needs a distinct computation/dataflow mechanism, selected routes, correctness/progress obligations, and smallest distinguishing test.

The existing isolated CUDA13.0.88 cache is documented in [verify and attention](2026-11-01-verify-and-attention.md). Researcher independently identified it, correcting the earlier narrow system-nvcc search. An exact MAIN three-arm target-compiler screen could cheaply distinguish the old gamma spill closure, but this report does not authorize a compiler sweep or assume that fewer scalar instructions have a material score budget. Compare its cost and plausible contribution against the broader design first.

The subsequent completed fixture diagnosis supports the explicit bounded repair/reallocation below. Researcher returns a ranked decision-ready broader mechanism, not a prolonged literature survey. The accepted failing upload stays immutable. Index/report publication is not an extra readiness gate for an already-authorized experiment.

## Explicit narrow repair and private-slot reallocation

After the initial stop, worker completed **5/5 source/CPU diagnosis tests**. Exact Tiny/Tiny2 OFF/ON factories use TINY1/SPLIT0/WGS0/SKT8/FT_T64. Frozen all-zero IDs yield8 entries/expert atT1 (invalid distinct-topk domain even without a fat job),64 atT8, one fat job and32 A fat units; the tiny fat-unit trap predicate is true. Legal distinct IDs0..7 per token yield at mostT<=8 entries/expert and zero fat jobs. The CPU model would first flagT8OFF0; it does not localize the observed CUDA719. The permissive old mock accepted any nonnegative route and missed this domain defect. The in-memory proposal restores distinct IDs on every occupied transition and retains all16empty checks/40capture-replay observations. Initializing IDs once is insufficient after empty replay. Evidence: worker's `occupied-contract-cpu.json` and `occupied-contract-tests.txt` under the paired evidence directory; final diagnosis publication remains worker-owned.

**Captain explicitly reallocates tranche slot#3/4 to ONE corrected private paired check**, superseding its previous held-out-only reservation. This is not an automatic retry of identical bytes. The now-proven tiny host-fixture defect, existing whole-layer protocol and lack of a ready stronger treatment justify the bounded repair rather than abandoning the causal comparison or waiting for a new architecture. Research alternatives continue in parallel. RMW alone is still not our complete top100 roadmap.

Worker may implement only the distinct occupied-ID fixture and discriminating actual-entry CPU regression, update exact upload/manifest/reader identity bindings, and prove native/production/timed bodies unchanged. Preserve all prior failing source and receipts. Retain36rows,3banks,4placements,24cold rounds,256MiB flush,46:16,allcrosses,duplicate-null,1%and regression gates. No safety weakening, additional treatment, mask/cap/compiler sweep or generic harness work. Publish exact changed bytes and CPU/invariant results, refresh authenticated positive private quota/no-live, then register exactly one private job and own its ID. No additional conversational approval once these conditions clear; no scored/public action.

An initial15minute effort checkpoint was sent with the authorization; worker's subsequent15-25minute repair estimate is accepted as a **25minute ready-or-specific-blocker checkpoint from authorization receipt**, not permission for open-ended work or a GPU completion promise. Setup295s fit remains unproved;237s failed worker wall is not the prep budget. A second setup/instrument failure ends this RMW allocation without using#4 as another retry. Valid null/loss/subthreshold also stops RMW-only promotion. Slot#4 becomes fresh held-out confirmation only after#3 validly qualifies, using previously frozen held-out banks. Avoid a redundant pure-ON qualifier already covered by c193 unless genuinely new release eligibility requires it; escalate that conflict rather than silently spending another slot.
