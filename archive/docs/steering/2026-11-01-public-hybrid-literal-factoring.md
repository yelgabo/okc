# Public verify hybrid: exact transparent factoring and budget

November1, source-only packaging review, root starting main4c7233d and08:02 report. No GPU/check/public submission and no semantic kernel change. Full607e default remains authoritative. Independent work is isolated to new tools; no edits to projection/MoE owners' files.

## Measured source inventory

`tools/dl_hybrid_literal_audit.py` inventories actual AST string constants, including subscript assignments missed by a top-level-variable-only count. Evidence `evidence/2026-11-01-hybrid-literals/inventory.json` pins every source and large literal. Original fullG869076B; four-cell G406564B; owned four-cell511911B. The old1.829MB whole-AST closure is not reused as a lower bound.

The active G main CUDA155790B and `_MA_ALT_SRCS['moe_helper0']`139768B contain131792B of exact common line spans. The helper is NOT unused: Tiny/Tiny2/Mid selection references it and helper0df, including original prepare/warm-up. Deleting it changes the default pipeline. `_GRAFT_GLUE_CUDA`10146B and `_SL_GLUE_CUDA`10575B contain8085B exact common spans. Remaining fullG internal pairs have only sub-kilobyte useful common spans; an all-large-string G/owned comparison finds no exact>=256B common span with the registered line-matching procedure. Owned internal comparison finds only367B between WD and MoE. This is an algorithmic scoped result, not proof that no alternative factoring exists.

## Implemented proof-preserving factor

`tools/dl_g_literal_factor.py`: pure `factor(source)->(source,records)`, also CLI. It emits63 explicitly named, readable multiline raw C++ literals and concatenates them with visible unique fragments in the original G module namespace. No encoding, compression, runtime file access, dynamic reconstruction algorithm, cross-backend symbol sharing or code deletion. Exact fragments are held once; Python string concatenation reconstructs the original full strings.

The implementation evaluates only its string expressions and proves BOTH reconstructed strings identical after each pair; restores those two AST values and removes its new constants, then asserts the rest of the module AST unchanged. Four runtime CUDA hashes remain exactly: mainc7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71, helperb8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716, GRAFT39c020a1a59cf2f82e5b64f1ec72d873bdcb0714adb5b55d7ead6299f5546a62, SLa2ce67a78cbc3bf1bfaa1f3fe49a503b6fe6bf0fefec88ddb33dfb4c41220172. Aliased helper0df receives the same immutable string as before; later replacement operations cannot mutate the shared raw literal chunks.

Two CPU tests pass for exact raw-literal round trips, reconstructed values, untouched AST and rejected repeat factoring. These tests are source proofs, not host/CUDA runtime validation. No new compile is needed to establish string identity; H200 qualification of the final composed module remains required.

## Actual current budget

Projection owner's FIRST840235B reduction omitted interstatement license comments and is superseded. Use corrected854399B SHA2563bdbaed0ddbe8d8d1e4b63139d91292fea4b154baa47bcdd54363fdee4e63045, which retains notices/comments. Applying the factor produces720525B SHA256a6f97289c7abc8e49e7b7943e63f9b8138a8682988bb4c1b2603e3704fdc6464. Main/helper net saving126435B; glue7439B; total133874B. Full proof receipt is `factor-proof.json`.

Embedding this G backend through root's readable multiline transport costs721481B. Current metadata-guarded dispatcher adds3977B, leaving LESS THAN274542B for the embedded owned backend under the strict1,000,000B cap. Raw owned ONE151256+MoE121690 already totals272946B before wrappers/host/projection/boundary, so factoring alone does NOT establish fit. Owned owner must supply additional independently proven inactive-code pruning or another source-level reduction. Do not call fit achieved or dispatch an oversized combined file.

## Dispatcher and reduction review

Root `dl_verify_hybrid.py/body.py`: mask0 preserves byte-identical original607e source869076B; nonzero masks select only configured verify16/32 sliding/full cells, use G elsewhere, preserve separate module namespaces and separately prepared backend state. The revised selector checks expected shapes, BF16 activation/FP32 stream, page/seq/loc/local-start dtypes, FP8 cache layout, scales/common devices and correct RoPE using metadata only. Nonselected cells skip owned metadata checks. Unsupported metadata delegates to G. Seven actual CPU dispatcher tests rerun PASS, including all16masks,12cells, metadata negatives and state isolation stubs.

Projection owner's GLE_T fold is source-safe at the pinned file: the sole binding is an empty dict, the only two reads are `.get` calls in the short-circuit condition, and there is no alias/mutation. Removing its always-false branch leaves `_gle_prior_after` unchanged and retains independent ACTIVE_GLE_FRT code. Final fullG preparation/tuning/native bodies must otherwise remain intact; profile nonselection is insufficient to delete code that prepare still builds or tunes.

Remaining source/runtime gates: final owned reduction/compiler proof, actual final byte/hash manifest, real cross-backend mutable-storage/closure ownership review, full G startup plus owned setup within295s, official13 correctness and fresh replay validation, and measured cell override benefits from the private four-cell comparison. Stub state isolation does not prove real nested Lt/attention ownership. No supplied source reduction or packaging success authorizes a speculative override, fallback change or public submit. If owned cannot meet the exact274542B embedded budget, report the measured gap and stop; do not silently remove native fallbacks/startup tests or use opaque transport.
