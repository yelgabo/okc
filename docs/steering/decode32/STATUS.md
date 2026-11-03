# D32 status and job ledger

D32-02 picked up November 2, 2026. This chat owns D32 implementation through
private checks, analysis and integration directly. Other project work stays with
its existing owners. Public-submission permissions remain unchanged.

| Experiment | Exact source | Job | State and next action |
|---|---|---|---|
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
