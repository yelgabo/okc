# Decode32 report history

These reports preserve the decisions and results from November 2, 2026. Resume work from the [watched inbox](../../../../docs/steering/decode32/README.md) and [job ledger](../../../../docs/steering/decode32/STATUS.md). Their earlier versions remain available at the [compaction boundary](https://github.com/yelgabo/okc/tree/0585bd216cb980e8b47a66eef0ba4c3b0c5768a3/docs/steering/decode32).

## Attention design and pickup

- [D32-ATTN-01: Decode32 Worker. implement one full-decode attention candidate](2026-11-02-01-full-decode-attention.md)
- [D32-ATTN-01 status](../../../../docs/steering/decode32/2026-11-02-01-full-decode-attention-status.md)
- [D32-ATTN-01 implementation plan](../../../../docs/steering/decode32/2026-11-02-01-full-decode-attention-plan.md)
- [D32-ATTN-01: one-tile score lookahead candidate](../../../../docs/steering/decode32/2026-11-02-01-full-decode-attention-experiment.md)
- [D32-ATTN-01: nonbinding source feedback, ownership unchanged](2026-11-02-02-driver-attention-source-feedback.md)

## MID and first attention results

- [Completed pre-existing MID job. no handoff or approval needed](../../../../docs/steering/decode32/2026-11-02-03-mid-result.md)
- [D32 attention first private result](2026-11-02-05-attention-first-result.md)

## Matched confirmation

- [D32-06: continue the measured attention candidate; progress watch](2026-11-02-06-confirm-attention-and-progress-watch.md)
- [D32 attention matched confirmation contract](2026-11-02-07-confirmation-contract.md)
- [Message to Decode32 Worker. read at your next checkpoint](2026-11-02-07-next-checkpoint-message.md)
- [D32 attention matched payload](2026-11-02-08-confirmation-payload.md)
- [Matched attention setup failure and correction](2026-11-02-09-pair-setup-failure.md)
- [Corrected confirmation already accepted. follow its existing ID](2026-11-02-10-driver-live-queue.md)
- [Matched attention check stopped on routing mismatch](2026-11-02-11-matched-routing-mismatch.md)

## Source leads and routing observers

- [D32-12: Extract usable batch-32 leads from the H200 ~460-experiment note](2026-11-02-12-h200-460-next-leads.md)
- [D32 routing diagnostic found an inactive-treatment ordering difference](2026-11-02-14-routing-order-diagnostic.md)
- [D32-ATTN-05: bounded upstream routing observer](2026-11-02-15-upstream-route-observer.md)
- [D32-ATTN-06: explicit call-site routing observer](2026-11-02-16-explicit-route-observer.md)
