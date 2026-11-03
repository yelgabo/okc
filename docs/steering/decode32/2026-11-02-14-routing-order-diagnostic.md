# D32 routing diagnostic found an inactive-treatment ordering difference

November 2, 2026 Pacific. Own private job `5909739e8a4445b2`, source `9e5e5d64` published in `77023d44`, is terminal with setup failure and return code 112. Registration was published in `1d654b5f`. [Complete owning receipt](evidence/D32-ATTN-04/status-wait.raw.json), [complete log](evidence/D32-ATTN-04/profiles/paired.5909739e8a4445b2.log) and [parsed routing evidence](evidence/D32-ATTN-04/routing-analysis.json) are preserved.

The bounded diagnostic answered which routes differed. It did not produce a matched performance result. Keep the exact route gate and stop this source; do not repeat it.

## Observed difference

At D32 sliding, seed `1102031`, fresh replay `7`, row `6`, placement `3` differed from placement `0`:

| Placement | Captured attention selection | Returned expert IDs |
|---|---|---|
| 0, control | Control by binding | `216, 156, 94, 93, 153, 49, 144, 221` |
| 3, treatment-bound state | `[false]` | `216, 156, 94, 153, 93, 49, 144, 221` |

The expert set is identical. Zero-based slots 3 and 4 swap experts 93 and 153. The other two placements have no reported difference from placement 0. A treatment-bound state does not imply that the treatment kernel ran. The captured selector reports it inactive on this sliding-context workload, as required by the full-context-only source scope.

The receipt reports the intended attention module prepared, inherited native startup completed with 56 verified replays, then the strict natural-routing assertion failed. Its records contain one failed setup and no official tests, benchmark cells, timing profiles, score or confidence interval. There is no `D32_ATTENTION_PAIR_REPORT`. The existing success reader rejects the receipt as an unsuccessful terminal job.

None of the six loads has a qualified matched result from this job. Earlier timing work inside preparation is neither exported evidence nor a completed regression gate. The first ordinary attention result remains exploratory and unconfirmed.

## What this establishes

The diagnostic localizes an exact order mismatch on a workload that uses the unchanged attention path. It does not identify a score-lookahead numerical failure. Equal expert sets also do not establish equal routing weights, upstream logits or complete outputs, and do not justify weakening slot equality.

Source inspection of the frozen generated payload shows that the fresh loop restores each owner's stream and attention input before replay, checks its complete outputs against the corresponding reference, then compares route IDs across placements. The graphs and mutable state are independently owned. The reported selector state comes from capture. This evidence alone does not prove repeatability of upstream arithmetic or exclude an ownership, cache-reset or measurement defect.

The T32 post-attention paths call `_pool_router_fp32`; that function exposes the normalized hidden input and the returned pool/router tensors before top-k selection. These boundaries provide a narrower next observation than another full timing attempt. Which T32 post-attention branch and projection choice actually ran still needs capture evidence.

## Reframe and next bounded action

Two matched attempts failed before timing, one on FP8 diagnostics and one on unexplained routing. This diagnostic now shows that exact slot equality can fail on the unchanged path. More full-matrix timing retries would spend quota without distinguishing upstream variation from a measurement defect.

Keep the current candidate and control frozen. Next prepare an observer-only diagnostic that records the actual selected T32 projection/routing branch and retains each graph's attention output, normalized router input and pool/router result tensors. On a mismatch, compare those boundaries across owners and against the preceding replay of the same graph. Record exact equality and finite maximum differences, plus the scores of the swapped experts when the selected layout exposes them. Compare caches and graph input snapshots too. Missing branch or layout coverage must fail closed.

This observer changes tensor retention and can affect allocation, so its timings cannot qualify a gain. It must preserve the original tensor values, native launch choices and strict route assertion. CPU generated-entry tests must exercise unchanged forwarding, graph replay updates, pointer ownership and injected same-set order swaps before one new private diagnostic is considered. Publish its exact source and discriminating hypothesis, then perform fresh queue/quota preflight. A new check requires the published source, discriminating hypothesis and existing preflight gates.

An alternative is to replace exact ordered IDs with set equality. That would change the registered gate and could hide weight or slot-association errors. It is not the selected correction. A component-only attention timing experiment could answer a narrower speed question, but would leave the complete-layer confirmation unresolved. The upstream observer has greater value for the current decision.

Stop after the observer identifies the first differing boundary, or after one qualified diagnostic remains inconclusive. Choose a correction from that evidence before any further run. Decode32 Worker owns this technical investigation under D32-02; no Driver handoff is required. Public permissions remain unchanged.

## Evidence checks

The local receipt check verified job, board, mode, terminal state, return code, single setup-failure record and the absence of a paired report. The complete profile log equals `result.log_tail`. The parsed event confirms one reported row, identical expert sets and exactly two different slots. SHA256 pins are in `routing-analysis.json`. These are source/receipt checks; no new kernel correctness or timing check passed in this step.
