# Independent OFF pre-expert coefficient review

2026-11-01. Reviewed projection-owner commitad3bf7c and its pinned exact607 transform. Source-only scope; no GPU/service calls, candidate changes or performance claim. Root explicitly owns index integration; no README edit.

No source blocker found. The seven named Triton consumers contain eight GPRE loads; all15 state-field reads are direct launch arguments to those consumers. Independent full-source field scan finds one creation and15 uses, with only one additional docstring mention. The two _sl_tile_route branches are both covered; generic/split/graft/deferred fallback consumers retain the same BF16 effective-coefficient ABI. All eight original1+load expressions are removed exactly once, so no double addition remains. Source reversal and the independent AST-transform test exclude unrelated native, routing, normalization, output-rounding or fence changes.

The baseline _norm supplies a BF16 parameter to MuseSpark1xRMSNorm and precompute_effective_weight at pinned SGLanga55ab7d5; gain_center1 yields a stored BF16 effective coefficient. The new prepare expression FP32(rawBF16)+1→BF16 agrees with literal baseline BF16(raw+1) for all65,280 finite BF16 encodings in the actual CPU test. Raw weights and g_preexp remain unmodified; a distinct contiguous BF16 tensor supplies the consumers. This is inherited G0/607 baseline-interface parity, not an added integration cost or a measured optimization. No categorical rule judgment beyond the concrete baseline interface is inferred.

The4096-element/8192-byte effective tensor is retained by prepared state for graph lifetime. It is input-independent derived weight data, so legitimate immutable reuse is possible. Existing recursive state clone machinery can privately clone it safely; a newly inventoried shared-readonly scheme must explicitly include it rather than assume the old inventory covers a new field. No pointer escape, captured old-field alias, process-global replacement or raw fallback mutation was found. Independent states and immutable-weight policy must remain preserved when composing new source epochs.

Independently executed tools/test_proj_g_preexpert_gamma.py under the existing WSL Torch2.10CPU environment with explicit OKC_REPO_ROOT:3tests PASS3.399s, including actual generated prepare assignment and all finite BF16 bits. No Triton execution, full norm/layer mathematical equivalence, FP8 cache behavior, GPU resource/capture or fresh/empty progress validation was performed. Exact-pin transforms intentionally reject prior modifications; composition with ATT7 host repair requires a separately pinned coordinator transform and disjoint-delta proof. Existing607 and f042 stay unchanged/HOLD.

Reviewed source hashes:

- tools\proj_g_preexpert_gamma.py: `d0d53f5ff24165219e007d524210fac5e9b976bf7c31b98e3491bf087ea08d7f`
- tools\test_proj_g_preexpert_gamma.py: `7a87846eb888f76292bae5ac1ec35563061d6cc1e8674fc55f206e9cf51418ae`
- docs\steering\2026-11-01-g-preexpert-gamma-off.md: `9e95b0d484174639847b984b9f10afdbd70988feb3d0b8191084b217d426a424`


## Disjoint common composition review

Also reviewed root tools/g_integration_common.py and its two tests. Both leaves remain exact607-only. The coordinator obtains host and coefficient treatments independently from the same source, verifies pre-expert transformation leaves ATT7 unchanged, inserts only the reviewed host assignment, and proves deleting the exact REPAIR body restores the complete coefficient-only source. The independent AST composition test agrees;2tests independently PASS5.593s. No new selector, global ownership or flag beyond the leaves is introduced. Combined output871077B/SHA4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280 remains OFF/source-only, not H200-qualified. No new device-math, Torch-extension, capture/resource or speed claim follows from composition.
