# D32 status and job ledger

D32-02 picked up November 2, 2026. This chat owns D32 implementation through
private checks, analysis and integration directly. Other project work stays with
its existing owners. Public-submission permissions remain unchanged.

| Experiment | Exact source | Job | State and next action |
|---|---|---|---|
| D32-REDUCE-01, one copy per split | 93a05524b1c10b44ac5bd1ea307a25fdc906db4a617737a579cfae6332fd0d49 | bd5776bdfcab42a7 | Accepted private check; three CPU tests pass; private ordinary check next after publication/preflight; device source/build flags unchanged |
| D32-ATTN-04, route localization | 9e5e5d64 | 5909739e8a4445b2 | Terminal setup failure; inactive-treatment order swap, no timing; archived bc6c7a2f |
| D32-ATTN-05, upstream observer | a0998224 | No accepted ID | HTTP422 reads harness frames; exact source closed |
| D32-ATTN-06, explicit-tag observer | 00193aa2ef2f8cefd22e2949bfd98f90a47ebfef370639dbf7bebd776de07cb4 | 8b1e364dafb54c71 | Terminal preparation-budget failure; no observer capsule or timing; instrument closed, no retry |
| D32-ATTN-01, one-tile score lookahead | 5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5 | 17383751c6a24010 | Terminal valid; D32 C32 11.882275 ms; promising cross-run signal, matched confirmation required |
| Existing unchanged control | ba9 plus unused metadata, upload04314b16 | 94ef698e91964dfa | Terminal, valid; descriptive D32 C32 12.324230 ms |
| Existing MID17..32 | e607fb894b7ca9fb8c0245d986e1c1894024295efd5b1509ac5b8314329ca089 | a6fe2769779e4299 | Terminal, valid but no useful D32 signal; exact implementation parked; original owner finished its wait |

[Attention experiment and frozen gates](2026-11-02-01-full-decode-attention-experiment.md).
[Ownership instruction](2026-11-02-02-end-to-end-ownership.md).
[Existing MID terminal](2026-11-02-03-mid-result.md).

[First attention result and confirmation gate](2026-11-02-05-attention-first-result.md).

Matched confirmation preparation: `tools/d32_attn_worker_pair.py` now provides
state-local control/treatment selection for the pinned candidate. Two CPU tests
cover the wrapper and source identity. This is one part of the instrument, not
a runnable or qualified paired check. Next bind four independently owned states
to the existing full-layer capture, cold replay, numerical and receipt machinery.
Prospectively register D32 and other-load gates before dispatch.

D32-06 read. Confirmation remains in implementation: the state-local arm
selector and the prospective timing decision now have four passing CPU tests.
The complete capture/replay payload and target qualification are still open.
[Confirmation contract](2026-11-02-07-confirmation-contract.md).

D32-ATTN-02 matched payload: `109cdb07dbe04fe6a9723b816a8fa5942969ce0150310dcdc02fca58b4f62929`, 928942 bytes, generated for one private check. Actual generated-entry selection tests and a four-arm real-CPU-Torch orchestration test pass. Target GPU, preparation budget, and the matched timing result remain untested. [Payload and gates](2026-11-02-08-confirmation-payload.md).

D32-ATTN-02 private check `4fece18b20a2447b` accepted for source `109cdb07`, published at `c7e13600`. Terminal setup failure: target PyTorch lacks FP8 `isfinite`. No timing was collected. The diagnostic was corrected and covered by a CPU Torch regression; the new payload is being prepared for one private retry.

[Setup failure and corrected payload](2026-11-02-09-pair-setup-failure.md).

Corrected matched payload `f397bca8c869b775fe79a4bc7d2c2749fba0542702d740431cbd7028e78908b3` is published in `f90da774`. Retry has not been submitted: fresh preflight found another problem's check `3aa3885f6b734f8c` running. Recheck the shared job list and quota before dispatch; do not poll that other owner's job.

Corrected matched retry `62a4f6b27ed74b50` was accepted for source `f397bca8`; fresh preflight found no live jobs and 18 private checks available. Follow this own ID; do not submit another copy.

Corrected matched check `62a4f6b27ed74b50` ended in setup with `EXPERT_FRESH_ROUTING_MISMATCH`; no matched timings. A strict-gate diagnostic payload `9e5e5d64` records load, seed, replay and differing placements on a repeat. One bounded diagnostic is next after publication and fresh shared job/quota preflight. [Failure and decision](2026-11-02-11-matched-routing-mismatch.md).

D32-MOE-01 picked up at the user's request while the shared GPU queue was occupied. Actual final-factory capture shows that B4/T32 already selects the parallel prefix/range scan in native `c7c201ae`. The serial literal from D32-12 belongs to Tiny/Tiny2/MID. Three CPU selection/negative tests pass. This exact proposal is closed as already implemented, with no new kernel, compilation or GPU job. [Source evidence and decision](2026-11-02-13-b4-prefix-selected-source.md). The frozen attention diagnostic `9e5e5d64` remains pending shared queue clearance.

D32-ATTN-04 strict routing diagnostic accepted as `5909739e8a4445b2` for source `9e5e5d64`, published `77023d44`. Fresh preflight found no live jobs and 17 private layer checks available. Follow this own ID only; do not submit another copy.

D32-ATTN-04 `5909739e8a4445b2` is terminal with strict routing setup failure, no timing. Diagnostic localizes D32 sliding / seed1102031 / fresh-7 / row6: placement3 swaps experts93/153 in slots3/4, same expert set, captured attention selection false. No causal attention loss or gate relaxation follows. [Owning evidence and next bounded observer](2026-11-02-14-routing-order-diagnostic.md). Next prepare upstream branch/tensor/replay observability on CPU before a new source or GPU allocation; do not poll or repeat this terminal job.

D32-ATTN-05 observer implemented from frozen9e5e5d64: source `a09982243ff3bd5139f5ba87acc1a10905dda58614ad21246e1cefe99894a324`, 936829 bytes. Five real-CPU-Torch generated-entry/forwarding/replay/negative tests pass; native compiler inputs and device literals are unchanged. No target result or timing. [One-cell diagnostic and finite stop](2026-11-02-15-upstream-route-observer.md). Next publish, verify exact source and fresh shared jobs/quota, then execute one private observer check if the queue is clear. Strict routing remains; no performance report can qualify from this instrument.

D32-ATTN-05 POST was rejected before acceptance with HTTP422 `reads harness frames`; the subsequent shared list shows no new job or live run. The corrected D32-ATTN-06 uses fixed source call-site tags, SHA256 `00193aa2`, six passing CPU tests and unchanged device compiler inputs. [Rejection, correction and one-run contract](2026-11-02-16-explicit-route-observer.md). No qualified next D32 optimization exists yet; research needs exposed B4 phase/tail cost and a concrete treatment, or evidence of a selected T32 pool/router fallback with meaningful latency.

D32-ATTN-06 observer accepted as `8b1e364dafb54c71`, source `00193aa2`, published `a7e08ee4`. Fresh preflight found no live jobs and 16 private layer checks. Follow only this own ID. A completed diagnostic intentionally stops before timing; strict routing is preserved.

D32-ATTN-06 terminal: `8b1e364dafb54c71` failed `EXPERT_PREPARE_295S` inside the fresh loop, no observer capsule or timings. Inconclusive instrument closed under its stop rule; do not poll, repair or resubmit it. Routing remains unresolved and lookahead unconfirmed. The bounded CPU pooler check passes five fixtures and confirms existing conditional fusion, without actual target-choice evidence. [Closure, exact missing evidence and shared-repository research refill](2026-11-02-17-observer-budget-closure-and-refill.md). No active D32 job or qualified next kernel treatment. Await a distinct source-qualified task, preserving CUDA Optimizer's phase study.

User continuation: work toward verified2.3x instead of waiting for a research-only refill. D32-REDUCE-01 is implemented and CPU-checked, changing only D32 sliding's accumulator-copy policy from shared to one copy per split. [Exact source, rationale and one-check stop](2026-11-02-18-active-reduction-experiment.md). Closed observers remain closed; CUDA Optimizer's MoE phase study remains separate.

D32-REDUCE-01 accepted as `bd5776bdfcab42a7`, source `93a05524`, published `0585bd21`. Fresh preflight found no live jobs and 15 private layer checks. Follow this own ID. Next implement/compile the newly supplied full-context exact identity-rescale treatment while this check runs, without another GPU POST.
