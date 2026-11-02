# Captain decision: qualify once, then isolate short RMW

November 1, 2026 Pacific / November 2 UTC. Evidence commit **516a93a4fc5fc882565ff0d67661b03d798e2480**; implementation **a0b6af2ec00dfed54f37fd81de00c14098748711**. Continues [captain top-100 direction](2026-11-01-captain-top100.md) and reconciles the [owning terminal report](2026-11-01-rmw-v2-h200-terminal.md). Captain owns allocation; existing worker owns implementation, registration, retrieval and sole dispatch. This report authorizes no public submission, scored measurement, automation resume or kernel change by captain.

## Decision and immediate allocation

**Spend one next private check on matched repaired-common OFF versus repaired short-RMW ON. Do not run a separate OFF official score and compare it to c193. Do not repeat ON alone.** The successful single-arm run solved qualification, not causal attribution. It does not close the treatment or promote it.

The initial tranche remains at most four checks total; one is used. Allocate the remaining three as follows, rather than spending the entire live quota:

1. **Tranche #2: one matched attribution experiment**, conditional on the exact source/ownership/compile/runtime protocol below. Worker first reports exact generated source pins and the bounded preparation-cost assessment. With those gates clear and fresh no-live/quota reconciliation, worker may register this private check; no further COMMON-first or private-win prerequisite is added.
2. **Tranche #3: reserve for independent confirmation only if #2 passes its registered gate.** Use fresh compilation, preparation and mutable allocations; freeze fresh banks **202611021 / 202611023 / 202611029** before seeing #2. Preserve all three routing categories and four control/treatment crosses. Same thresholds; do not select the winning seed, route or context. A confirmation failure ends promotion of this treatment alone.
3. **Tranche #4: reserve, not automatic dispatch.** Consider exact pure-file qualification only if the paired/confirmation evidence qualifies a release candidate and the final pure source differs from the already qualified ON epoch or otherwise needs qualification. Existing c193 evidence should not be duplicated just to fill a slot. Otherwise return the slot for a newly justified integration experiment; captain chooses after receipts.

Live worker terminal ledger recorded nine private checks and five scored slots remaining, no live jobs. That is a dated observation, not a reservation or current dispatch preflight. No scored/public authority is inferred from it.

## Independently checked evidence

Captain read the owning terminal, strict-reader invocation, observation, measured summary, profile hotspot table, latest terminal report and existing short-path/closure ledgers. Local physical ON bytes match SHA256 **978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a**, **918978B**. OFF is registered **32196b8bc1c954e9b7f2c6667e6b3e0aeed14da0fed6fe53bc153c9d563e5047**, **918490B**; c193 does not independently qualify OFF.

Own job **c1931f0c9e834485** is mode check, board muse_decode_layer, done/passed, checks.ok and empty flags/rejections. Reader exit0 explicitly requested mode check. Log hash matches its observation. Terminal execution ran November2 **00:45:20.293–00:49:11.578 UTC**; submission/queue time is separate.

Complete native startup and actual RMW exercise passed: 12 cells, 72 checks, 12 captures, 36 fresh checks, 60 empty checks, 12 empty captures, 48 restores. Official evaluation has 13 passing tests, 12 benchmarks and 12 profiles. Actual Tiny/Tiny2 resources are 145 registers, zero local bytes, 214784B dynamic shared, 288 threads, one resident CTA/SM, grid132. Do not replace these actual resources with the old local CUDA12.8 compile vectors.

Official score **2.1212410493206884**, CI **[2.089422433580878,2.1530596650604985]**. Independently reconstructed from six weighted baseline/latency ratios without changing weights; exact double-precision result agrees. Historical607 and c193 have different seeds, baselines and overhead. Negative latency-change percentages mean lower latency, not score regression. Neither their point-score difference nor overlapping intervals estimates the isolated treatment effect.

Preparation **109.269811068s**, of which **106.90144866799892s** was normal production preparation. The approximately2.37s remainder demonstrates feasibility of this particular single-arm qualifier. It does not establish that two compiled variants, four private placements, three banks and timing fit the remaining185.73s under the295s guard. No runtime guarantee is inferred from this subtraction.

[Captain verification and budget arithmetic](evidence/2026-11-01-captain-rmw-attribution/verification.json) pins source/log/input hashes and the independent score reconstruction. Raw GPU evidence remains owned and archived by the worker; no new GPU run was made by captain.

## Why this experiment rather than another route

RMW is the only surviving implemented performance treatment in the current G funnel. Its native selection now runs successfully, and the old four-placement comparator already exists. One bounded matched experiment has higher immediate decision value than another unmatched official OFF/ON score or a new broad source survey. This is an allocation judgment, not a performance forecast.

The strongest plausible competing project is the larger G0-to-conforming integration discrepancy, especially the short native expert chain. That remains unattributed: historical G0 advantage is neither recoverable time nor an eligible unsafe control. Dense short paths were not the strongest cross-profile deficit. Current profiles still motivate native expert investigation, but activity totals are not exposed savings. Broad independent complete-file discovery was archived in the preceding captain report; no newly verified stronger conforming complete file has appeared in this handoff. Do not rerun that survey solely to delay measurement.

Read scoped closures before proposing follow-ons: full-family gamma failed MAIN resource gate; paired RNE failed dependency/codegen screen; scale movement violates exact required boundary behavior; the bounded O-boundary finalizer failed its source budget screen; the expert-to-unpool PDL edge has no measured retirement/prefix exposure. Neither c193 nor this report reopens those implementations. Correctness properties are not optional ablations.

### Remaining short-path budget

Using c193's observed adjusted weighted D1 **77.7200013us** and V1 **112.8152580us**, suppose the SAME exposed duration d is removed in both contexts of both shorts, holding the four other loads fixed. The six-load factor is:


a(d) = [(D1/(D1-d)) * (V1/(V1-d))]^(1/6).

| Hypothetical d per short context | Whole-six factor |
| ---: | ---: |
| 1us | 1.003648935 |
| 2us | 1.007353139 |
| 3us | 1.011114132 |
| 5us | 1.018812870 |

This is sensitivity arithmetic, NOT a measurement of removable time, a roofline, or an RMW prediction. Equivalently, a1% aggregate factor from only these two loads needs about2.941% latency reduction on each if fractional gains are equal. RMW must earn its place through the complete graph, not static atomic instruction count. A positive but sub-threshold result may be preserved for a later independently justified combination; it is not an excuse to relax the frozen gate or claim top100 is solved. The earlier rank/cutoff snapshot stays explicitly historical; no fresh leaderboard or top100 feasibility claim is made here.

## Exact matched protocol for tranche #2

Reuse **tools/g_short_rmw_check.py** and its existing owner/reference/timing machinery only after rebasing it on current common progress repairs and inference-compatible memo ownership. The old f042/906841B comparator is historical and **must not be dispatched unchanged**. Its official entry was historical607; the new official entry must return **current repaired-common OFF**, never an unrepaired progress arm. Both comparison arms retain all common cooperative-launch, proxy/publication, reader-retirement, activation-boundary and admission guarantees.

Hypothesis: the fixed Tiny/Tiny2 final-returning-RMW representation reduces matched cold whole-graph latency sufficiently to clear the existing whole-six promotion screen. Only that representation differs in timed control/treatment code. MID/main/B4, actual attention, dispatcher, packing, routing and epilogues stay common. No gamma mask, stage toggle, shape sweep or timing-selected variant.

- Two separately compiled Tiny/Tiny2 code variants; four independently writable placements (OFF0/OFF1/ON0/ON1). Read-only weights/code may share exactly as audited. Distinguish allocation duplicates from independent compilation; #3 supplies fresh compilation/preparation.
- Six loads1/8/16/32/128/256, sliding/full; original frozen banks993811/993817/993821 and busiest/second/least natural EP mapping. All36 rows required, exact IDs and oracle-compatible weights/outputs/cache behavior. Each placement sees the same semantic inputs/routing; comparing independently generated official job inputs is not matching.
- Preserve original24 cold randomized timing rounds per placement/row and256MiB flush, identical policy for every arm. Use whole captured graphs and46:16 context weighting followed by six equal log weights. Do not reuse official overhead corrections, infer official board score from GPU events, or insert internal timing observers into production replay.
- Preserve reference/fresh/restored checks and actual native selection/resource witnesses for BOTH arms. Reference memo reuse must retain exact tensor identity/mutation invalidation and inference-mode behavior. No writable alias between owners or production; no silent fallback credit. Original old harness inspected only the final of eight fresh replays: state the actual coverage, and reuse the repaired qualifier's individually observed replay coverage where integrated. Never label uninspected intermediate replays checked.
- Exercise empty-to-occupied and occupied-to-empty transitions, including replay and restored occupied validation, without manual counter reset that bypasses the lifecycle under test. Inspect replay outputs/cache state, not just capture success. Keep observer bindings and reference scratch out of the timed graph; restore bindings, TLS, caches and production state on failure and success.
- Strict own-ID reader checks board/mode/source/terminal, complete paired capsule, rows, finite timings, flags, identities, resource/coverage counts and actual treatment exercise. Candidate-coded official success is not a substitute for a complete paired result.

**Promotion screen remains unchanged:** every one of four control/treatment crosses in each bank must show whole-six factor>=1.01 and exceed twice the larger duplicate log-null; every context/load must obey regression<=max(0.5us,1%). Report affected-only effects descriptively, without swapping them into the full-six gate. Fixed banks are not a claim of exhaustive data coverage. Do not post-hoc adjust thresholds, discard inconvenient placements or combine favorable crosses.

Before dispatch, provide exact source/hash/byte count, one-file-size eligibility, code-delta proof, selected compile/resource evidence, inference-mode/negative-owner/reader tests and a bounded total-cost assessment. Keep the295s fail-closed guard from before production preparation through cleanup. Do not weaken coverage to make the budget fit. If preparation cannot credibly fit with existing bounded reuse, return that concrete finding to captain instead of starting a generic harness rewrite. Qualification and numerical success do not replace ordering/progress proof.

## Stopping, ownership and next decision

- Valid decisive loss, positive sub-threshold effect, inconsistent banks/crosses or regression breach: end RMW-only promotion in this tranche, preserve observations and add a scoped ruled-out/run entry. Do not call a noise-limited result a universal RMW failure. No new seed/cap/mask hunt.
- Incomplete setup, source identity, ownership, native selection, resource admission, budget or reader failure: instrument failure, not treatment loss. Preserve the exact phase/receipt; no blind retry. Captain reallocates before another slot.
- Full screen pass: request #3 fresh-bank confirmation under the reserved protocol. Report exact pure-source equivalence and retained c193 qualification before deciding whether #4 is necessary. Public dispatch remains a separate user decision.
- Worker owns implementation and sole dispatch, and has acknowledged no follow-on dispatch from c193. Captain owns this report/index after worker released index ownership at516a93a. Existing unrelated WIP and source snapshots remain untouched.
- If this bounded lane does not qualify, the next decision returns to a materially different legal G integration design with actual short-chain dependency/exposed-time evidence and complete-file comparison, not automatic generic attention/projection tuning. No second ready candidate is asserted today.

## Research scope and gaps

On this turn captain inspected the source of the existing comparator, its registration, common single-arm qualifier report, short-path evidence, minimal-delta ledger, current profile/baseline hotspot table, ruled-out ledger and owning terminal artifacts. Targeted web search for challenge/problem and CUDA graph timing returned no usable output in the general web tool. Direct official rules fetch returned no readable lines; it supplies no new rule interpretation. NVIDIA's current CUDA Graphs documentation was readable: graph definition/capture is separate from replay execution, reinforcing reuse of whole-graph replay measurement rather than eager-launch/profile arithmetic. Primary source: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html . This is protocol support, not novel optimization evidence or a fresh complete-file survey.

Unknowns remaining: paired runtime/fit; actual RMW causal benefit; complete common-OFF runtime qualification; fresh-compilation/heldout robustness; exposed short expert costs attributable to individually replaceable conservative choices; current top100 cutoff and final rank. No public submission, GPU call, new kernel or altered release threshold occurred in this captain task.

