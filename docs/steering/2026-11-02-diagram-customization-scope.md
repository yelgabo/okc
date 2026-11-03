# Architecture diagram customization scope

November 2, 2026. Evidence base `b93c4c7cd6d8347f0506a5bd5b44f9655f6ac9ec`, plus live official problem descriptions retrieved with `okc show`.

## Answer and diagram correction

Every computational box in the attached architecture screenshot is customizable in `muse_decode_layer`, subject to its mathematical, precision and execution requirements. This includes pooler/router, O and its norm/residual boundary, and unpooler and its final boundary. Implementations may fuse work across the pictured boxes while preserving those requirements.

The screenshot's green badges label performance groups. They do not define editable regions or map one-to-one onto the standalone problems. The badge labelled "All dense + router GEMMs" covers projections across the diagram, including QKV/gate, O, pooler, unpooler and router. Placing it beside QKV makes that coverage ambiguous.

The diagram needs a whole-layer scope label and a separate legend explaining performance groups. This report records the clarification; it does not claim that the original HTML has been edited.

## Problem boundaries

| Pictured work | Standalone dense projections | Whole decoder layer |
|---|---|---|
| QKV/gate and O matrix multiplies | Included | Included |
| Pooler, unpooler and router matrix multiplies | Included | Included |
| Router sigmoid, bias, top-8 selection and routing weights | Outside the standalone GEMM contract | Included |
| Residual updates, RMS norms and boundary fusion | Outside the standalone GEMM contract | Included |

The dense problem returns one requested matrix product. Its router result is logits, before expert selection. The whole-layer problem returns the next attention input, residual stream and expert IDs, and updates the KV cache. Cross-GPU communication is excluded, with the specified identity behavior and small-token unpooler slice.

## Evidence and validation

- [Live dense-projection specification receipt](evidence/2026-11-02-diagram-scope/muse_bf16_proj.json), from `okc show muse_bf16_proj --json`. Its task text matches [the local task](../../muse_bf16_proj/TASK.md) after stripping surrounding whitespace.
- [Live whole-layer specification receipt](evidence/2026-11-02-diagram-scope/muse_decode_layer.json), from `okc show muse_decode_layer --json`. Its task text matches [the local task](../../muse_decode_layer/TASK.md) by the same check.
- Canonical pages are [dense projections](https://openkernelchallenge.dev/problems/muse_bf16_proj) and [whole decoder layer](https://openkernelchallenge.dev/problems/muse_decode_layer). Browser text retrieval returned no readable lines, so the official CLI supplied the specification evidence.
- The screenshot is the evidence for badge placement and labels. Its original HTML was not identified in the current checkout, including hidden and ignored HTML paths. The previously listed `/Users/gretchengabo/Documents/GitHub/okc/docs/incumbent-architecture-2026-11-02.html` path was unavailable.

No benchmark was dispatched and no new timing or speedup was measured. This review checks customization scope, not the accuracy of the pictured speedup values. Independent review was unavailable because the project prohibits new agents.

## Priority and next action

Treat every pictured computational stage as eligible when evaluating complete-layer optimizations. Preserve current ownership and experimental gates. Eligibility establishes where we may optimize; measurements must establish whether a proposed change helps.

For the next architecture-diagram edit, add the whole-layer scope label and attach the dense/router badge to all five projection roles. Stop the scope review here. A new kernel experiment or performance claim needs its own source-qualified hypothesis, controls and acceptance criteria.
