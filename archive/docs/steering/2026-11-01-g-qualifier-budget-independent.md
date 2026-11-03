# G common qualifier setup-budget failure: independent source review

Evidence: source97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7; owning H200 job3fd1317143304bd5. Read root `docs/steering/evidence/2026-11-01-rmw-wait-2258/persistent-wait-result.json` (outer stdout is the owning receipt). No GPU/API calls in this audit.

## What the failure establishes

The terminal record is setup FAIL/return112, GQUAL_PREPARE_BUDGET; no benchmark score exists. G_RELEASE_STARTUP_OK reports56 verified native replays. Traceback reaches `_gq_empty` ENTRY from `_gq_private`, where its first statement checks total elapsed<295. Source979 remains a failed qualification artifact. This is not a measured performance or kernel correctness loss.

Before that check, the source has completed normal production prepare (including G56), private-state inventory/clone/ownership checks, five native resource queries, and the current cell's base/fresh input creation and four reference-front computations (BF16/FP8 for each bank), plus input deepcopy. The traceback carries no cell ordinal or per-phase timing: it cannot establish first-cell failure, compilation cost, particular reference cost, or cache-warming benefit. Earlier cells may have run. Do not infer one bottleneck from the line alone.

## Bounded instrumentation

Add prepare-only flushed monotonic milestones with token/layer/seed/ordinal: enter/exit normalprepare, readonly/clone, resources, each input allocation and each reference-front dtype, eager/capture/replay validation, empty graph, final cleanup. Keep a last-phase label and print elapsed/phase in exception handling before reraising. Do not add device synchronization solely for logging: these are host wall intervals including synchronization already present, not pure GPU durations. Bound line count/output bytes, preserve failure visibility and unchanged295-second deadline. Counts remain72 model checks/12 captures/36 observed changed replays plus60 empty checks/12 empty captures/48 occupied restores. Original G56 remains intact unless a separately proved equivalent integration reuses its evidence.

## Concrete duplicate-work reduction with unchanged oracle

`_qual_oracle` invokes `_ab_ref_back` twice on every validation, using the two immutable reference fronts and the ACTUAL output expert IDs. That is144 expert-reference-back calls for72 model validations. Within each cell, the same base/fresh reference banks recur. A private per-cell memo may key each back by reference-front identity AND exact actual IDs content (not tensor address, top-k prediction, or assumed deterministic routes). On repeated IDs, reuse immutable reference outputs; on any ID change recompute. Clear per cell and finally; retain weights identity/immutability and all finite/reference/output/routing/cache/bounds checks on every observation. In the identical-route case,24 bank pairs require48 unique backs instead of144. This is a source operation-count reduction, not a measured runtime saving or guaranteed fix for this deadline.

No reduction to coverage, no math/order/resource change, and no prospective success claim is justified yet. Owner implements the fix; independently review final cache ownership, fallback and phase coverage, then require actual complete setup under295 with printed counts before attributing any RMW performance. If telemetry still cannot locate cost or full checks cannot fit, stop rather than blindly repeat the failed artifact.

## Cache-epoch inference-mode blocker, 2026-11-01 UTC

Independent review of proposed82234b757e64242fa23d704a909c6996c602e3f0dae327fb66a67260243544d8 found that `_gqb_signature` and saved ID/result guards access `Tensor._version`. Torch2.10CPU raises `RuntimeError: Inference tensors do not track version counter` for tensors constructed under `torch.inference_mode()`. Reproduced both on a direct tensor and the actual complete-prepare CPU facade under outer inference mode: prepare→_gq_private→_gqb_validate→_qual_oracle→_gq_ref_back→_gqb_signature. This source is not ready to seal until corrected; the judge context need not be assumed either way.

A narrow untimed qualification scope disabling inference mode (and retaining no_grad) can construct versioned private fronts/snapshots/results without changing production prepare or native math. It requires both-arm whole-prepare inference-context tests and context/finally restoration proof. Silently treating absent version counters as unchanged is not acceptable. No GPU/API call was made; the failed979 source remains preserved.
