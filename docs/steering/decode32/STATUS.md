# D32 status and job ledger

## Current experiment

D32-IDENTITY-01 is implemented and locally qualified for one private full-context comparison. Frozen paired source `f7a5d1fb7fba3365f52285ef84f7d888e6f4b347280f257833469f85382bef9a`; native `b599e5b2`. Both CUDA twins compile with128 registers/zero spills; disabled SASS equals incumbent. Eleven CPU/source/host/reader tests pass. No accepted identity job or GPU result yet. Next publish/verify source and fresh shared preflight, then dispatch the registered comparison. [Source, tests, budgets and gates](2026-11-02-19-reduction-result-identity-next.md).

D32-REDUCE-01 job `bd5776bdfcab42a7` is terminal,13/12/12 passing. Source `93a05524`, published `0585bd21`. Adjusted sliding/full161.438408/304.957678us, C32=12.305490ms, own D32 baseline speedup1.814995x; whole-six2.119214x CI[2.087426,2.151002]. Stop exact eight-copy performance treatment: no useful affected sliding signal, no matched gain or routing correction established. Waiter47403 ended. [Owning receipt](evidence/D32-REDUCE-01/status-wait.raw.json), [all-six analysis and decision](2026-11-02-19-reduction-result-identity-next.md). [Watched inbox](README.md).

## Completed checks and closed instruments

| Experiment | Source | Job | Recorded outcome |
|---|---|---|---|
| Attention lookahead | `5a580c7a` | `17383751c6a24010` | Valid ordinary check, C32 11.882275 ms. Matched gain unresolved. [Record](../../../archive/docs/steering/decode32/2026-11-02-05-attention-first-result.md) |
| Control | ba9, upload `04314b16` | `94ef698e91964dfa` | Valid ordinary check, C32 12.324230 ms. Different inputs. [Record](../../../archive/docs/steering/decode32/2026-11-02-05-attention-first-result.md) |
| MID17..32 | `e607fb89` | `a6fe2769779e4299` | C32 12.752280 ms. Parked. [Record](2026-11-02-03-mid-result.md) |
| Matched confirmation | `109cdb07` | `4fece18b20a2447b` | Setup failed on FP8 finiteness check. [Record](../../../archive/docs/steering/decode32/2026-11-02-09-pair-setup-failure.md) |
| Corrected confirmation | `f397bca8` | `62a4f6b27ed74b50` | Setup failed on strict routing agreement. [Record](../../../archive/docs/steering/decode32/2026-11-02-11-matched-routing-mismatch.md) |
| Routing diagnostic | `9e5e5d64` | `5909739e8a4445b2` | Two experts swapped order in one row with treatment inactive. No timing. [Record](../../../archive/docs/steering/decode32/2026-11-02-14-routing-order-diagnostic.md) |
| Upstream observer | `a0998224` | No accepted ID | Service rejected frame access before dispatch. [Record](../../../archive/docs/steering/decode32/2026-11-02-16-explicit-route-observer.md) |
| Explicit-tag observer | `00193aa2` | `8b1e364dafb54c71` | Preparation budget expired. No capsule or timing. Instrument closed. [Record](2026-11-02-17-observer-budget-closure-and-refill.md) |

The routing gate still applies. D32-MOE-01 closed at source inspection because B4 already selects the parallel prefix. [Selected-source correction](2026-11-02-13-b4-prefix-selected-source.md). The full sequence of postings, preflights and handoffs is preserved in the [ledger at the compaction boundary](https://github.com/yelgabo/okc/blob/0585bd216cb980e8b47a66eef0ba4c3b0c5768a3/docs/steering/decode32/STATUS.md).
