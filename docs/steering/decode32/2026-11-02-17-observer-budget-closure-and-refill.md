# D32 observer budget failure and qualified-task refill

November 2, 2026 Pacific. Evidence checkpoint `02e8d7f4`. Own private job `8b1e364dafb54c71`, source `00193aa2` published in `a7e08ee4`, is terminal. It failed setup on `EXPERT_PREPARE_295S` with return code112. [Complete owning receipt](../../../archive/docs/steering/decode32/evidence/D32-ATTN-06/status-wait.raw.json), [complete log](../../../archive/docs/steering/decode32/evidence/D32-ATTN-06/profiles/observer.8b1e364dafb54c71.log), [receipt analysis and hashes](../../../archive/docs/steering/decode32/evidence/D32-ATTN-06/terminal-analysis.json).

## Decision

Close this observer as inconclusive under [its registered stop](../../../archive/docs/steering/decode32/2026-11-02-16-explicit-route-observer.md). No retry, larger time limit, seed sweep or further local instrumentation variant follows. The exact route gate stays intact. Attention lookahead remains unconfirmed and is not integrated. This closes a measurement instrument, not the attention design family.

The log contains the intended attention preparation and inherited 56 native replay marker. The failure location is the budget check inside the fresh-replay loop. No `D32_ROUTE_OBSERVER` capsule was emitted. The observer reader rejects the log with `expected one observer report`. No branch, boundary tensor, exact selected pool/router choice or same-graph comparison is available for analysis. No official tests, benchmark cells, timing profiles, score, confidence interval or matched C32 were produced.

The 295-second guard covers original preparation and the private work. The receipt does not separate compilation, references, cache copies or observations. Assigning the failure to one of those costs would be a hypothesis. The instrument buffered observations until mismatch/completion, so this failure also lost partial evidence. Future diagnostics must preserve bounded partial observations on exceptional exits while preserving the failure status. The closed source is not being repaired or resubmitted.

## Bounded pooler-choice source check

The standing queue's lower-cost pooler check was executed without another GPU allocation. The pinned preparation selector at public607 line15280 explicitly chooses `bfpf` when pooler and router both select SK2. Executing that actual selector with controlled CPU preparation fixtures gives:

| Fixture | T32 pool/router choice |
|---|---|
| Both SK2 banks available and enabled | `bfpf` |
| Pooler bank missing | `bfp` |
| Router bank missing | `bfp` |
| Router SK2 route disabled | `bfp` |
| BFP state absent | Parent choice retained |

All five assertions passed. [Reproducible source audit](../../../archive/docs/steering/decode32/evidence/D32-ATTN-06/pooler-choice-audit.py), [source/function hashes and results](../../../archive/docs/steering/decode32/evidence/D32-ATTN-06/pooler-choice-audit.json). Run from the repository with `PYTHONPATH=tools python3 docs/steering/decode32/evidence/D32-ATTN-06/pooler-choice-audit.py`.

These fixtures validate the existing conditional selection rule. They do not run device preparation, establish the actual target's bank/disabled state or measure fallback cost. There is no new fusion candidate without those observations. The failed observer cannot supply them.

## Reframe and next research request

No D32 job is active. The prepared matched instrument has produced setup-only failures rather than a qualified attention comparison. Continuing to add observers to that same prepare path has low value without a demonstrated cost budget and retained partial evidence. A different diagnostic organization must justify itself before another allocation. A detached attention-only comparison could answer numerical or execution questions, but cannot replace the required complete-layer C32 and other-load gates.

There is no qualified immediate D32 kernel treatment in the current inbox. B4 phase attribution remains with CUDA Optimizer; this worker will not duplicate it. Prefix, scale/group, NS4 and parked MID closures remain.

**Shared-repository request to "OKC CUDA research" and "Monitor muse challenge disscussions", through the existing research queue.** Please refill the D32 lane with one bounded source-qualified experiment. This adds the D32 gap to the [existing widened-search request](../2026-11-02-general-hotspot-search-refill.md); it does not create another owner or claim direct-chat delivery or pickup.

The missing evidence is one of:

- A selected incumbent B4 entry, A-to-B or tail cost shown to be exposed, with exact source/build flags, a concrete treatment and conditional savings against C32. Reuse CUDA Optimizer's separate attribution result when available.
- A complete, attributable alternative for T32 attention or another D32 hotspot, with selected preparation/dispatch and a bounded comparison that fits the setup budget. Preserve ordered routes, math, cache, graph lifetime and all-load gates.
- Actual target evidence of the T32 pool/router fallback, why the existing fusion is not selected and enough exposed latency to justify changing it. The conditional source rule alone is insufficient.

If attention lookahead remains the strongest lead, research needs a distinct measurement design that can identify unchanged-path route-order instability with demonstrated preparation cost and useful evidence on failure. The current source's ordinary result is a signal, not permission to promote it or weaken routing.

The next accepted experiment must identify the source, controls, affected workloads, acceptance and finite stop before dispatch. Decode32 Worker continues implementation, private checks, analysis and integration directly once that task is qualified. No Driver approval is required. Public permissions are unchanged.

## Checks and limits

Verified the owning job/board/mode and terminal setup record, return code, full log identity, missing capsules and reader rejection. Archived the raw receipt and log without decoding substitutions. All six loads remain unqualified by this check. The CPU pooler selector audit passed five controlled cases. No new compilation, numerical GPU check or performance measurement passed in this work unit.
