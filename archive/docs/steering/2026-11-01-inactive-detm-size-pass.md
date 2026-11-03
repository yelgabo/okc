# One inactive diagnostic helper: byte reduction and compiler equivalence

November1,2026. Bounded pre-reset pass following `2026-11-01-owned-load-slice-checkpoint.md` and G slice checkpoint. No production edits, source obfuscation/compression, runtime source reads, cubin packaging, cross-job claim, GPU execution or public activity.

**Result: safe2772-byte reduction, still no feasible paired payload. Stop this pass.**

`tools/dl_slice_detm_reduce.py --root <main-checkout> --out .build/detm-reduce` starts from the pinned owned851c4695 layer, applies the already documented BF16 attention-boundary repair and existing verify32/sliding owned slice. It deletes only the named `finalize_det` helper's enclosing `#if ONE_DETM` block. The actual emitted `_one_load` has no overriding ONE_DETM compiler flag; its source default is0. This diagnostic deterministic flat-decode helper is not selected under those exact flags or L8MODE3. No other branch/family is specialized. The original block is retained as a readable evidence file. Every notice and the remaining CUDA/Python source survives unchanged.

The generator verifies the exact upstream owned pin through `dl_owned_slice.build`, the default and loader flags, balanced conditional/one-helper scope, inverse string deletion, and exact Python AST after restoring just the changed `_ONE_SRC` value. Registration builders and active launch policy are therefore byte-identical. Existing3owned-slice CPU tests (source, actual inert-registration execution, runtime dispatch/order) independently rerun PASS; this is not a numerical prepare/capture test.

|Budget|Bytes|
|---|---:|
|Original owned slice|504940|
|Reduced owned slice|502168|
|Saved inactive helper|2772|
|Four required unchanged G literal values only|520047|
|Existing paired timing/reference harness BODY|40191|
|G literal floor + reduced owned, **zero harness**|1022215|
|Same lower bound + existing BODY|1062406|
|Actual existing857866-byte G checkpoint + reduced owned + BODY|1400225|

The last two figures still exclude namespace glue, added declarations and parameters; the literal floor also excludes G's Python/host support and LICENSE/NOTICE. Thus they are conservative lower bounds for this named composition, not a successfully linked combined artifact or a universal minimum for all possible semantic slices. Even with no harness it exceeds999999 by22216bytes. With the existing BODY it exceeds the limit by62407bytes before omitted support. Do not reopen the unchanged-literal pair from a2772-byte saving.

## Actual local compile equivalence

Original repaired owned `_ONE_SRC` SHA256`beba98d82228815d58d4a5be3e7a3836ef58064ad06085542f198782fca66708`; reduced`05d7e70bdf9f2bec4418838e7011a3766e71997d00c6b9aea55e6e51abb57884`.

The first invocation of the standalone-attention compile helper failed identically for both sources because the layer wraps the CUDA in **both** anonymous and `one` namespaces. Its one-brace adapter left the outer namespace open and the unqualified expression was wrong. This was a local harness error, not a source/algorithm failure. The saved compile adapter adds exactly the second closing namespace after the complete device prefix and selects `&one::attn_one_kernel<8,3>`.

Both then compile with actual NVRTC12.8.93/PTXAS12.8 sm90a: **114registers,16barriers,0stack/0spill stores/0spill loads** each. All **2328 SASS instructions are exactly identical** after removing address/encoding comments and excluding ELF/debug-data directives; opcode/predicate/operands/branch targets remain compared. Normalized instruction SHA256`2cc23924ab13071875970c9d8a66157bf9e99bf3f8fd347a988788e63ecbedca`. Debug records and compiler-generated anonymous symbols differ; no claim of identical whole cubins is made.

Evidence is `muse_decode_layer/profiles/detm-size.2026-11-01/`: budget, equivalence, both compiler manifests/logs, exact readable deleted unit and offline compile adapter. That adapter excludes host `Instance`/launch wrappers, replaces Torch/ATen includes with CUDA runtime/cstdint and minimal type_traits declarations, maps NEG_INF to exact negative-infinity bits, and closes the two original namespaces. The manifests disclose all options/headers/prefix hashes. No active device math is stripped. The equivalence proves this selected native entry's lowered code, not host extension linkability, other entries, actual whole-layer state construction, H200 resource admission or timings. No compiled binary is embedded in any entrant.

Reproduce local compile for each original/reduced `.cu` emitted by the generator:

```text
/root/.cache/okc-host-validate/bin/python <evidence>/compile_owned.py INPUT.cu OUTDIR --expression='&one::attn_one_kernel<8,3>'
```

## Next necessary unit

A useful next pass must remove at least62407 additional bytes **plus currently omitted G support** for this BODY-based composition. The G checkpoint still uses ordinary prepare, which builds all5expert keys and tunes all token counts. A genuine selected-load G prepare/call-graph slice that can omit a mandatory native family, or a much larger owned active-family specialization, is necessary. That needs an explicit replacement preparation/policy equivalence contract; deleting more small diagnostic wrappers cannot establish it. This review does not perform that additional task or certify the incomplete owned prepare.

Stop here: no paired H200 input or readiness claim, no quota request. Reviewer is available for the separate short dual-stream code review.
