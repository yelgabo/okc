# Owned low-token MoE: immutable decode and activation-read feasibility

November1,2026; continuation after coordinator main `ff05d88`. This is a source/CPU audit, not a kernel benchmark. **No new GPU candidate is justified by the audited mechanisms.** Existing owned production and Family G607e remain untouched. No check or public action was dispatched.

## Evidence and active path

Owned source LF SHA256 `c7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7` is the tracked299717-byte source, not historical235700-byte ec565a7. The qualified common control is job835253441a214a1c (BF16 effective norm parameter and cooperative admission), with skinny8/3 at1/8/16/32 and main47 at128/256. That job passed5official tests and264private gates. This audit does not alter its precision or launch policy.

I reran `tools/hotspots.py` on its normalized receipt with SGLang baseline-as-submission eae9a6b7d5dd4809; output is archived in the evidence directory. All six owned profiles are a single fused kernel. Benchmark medians and separately collected kernel-profile times are different observations, not additive stage times:

| tokens | benchmark us | profile kernel us | official floor us |
|---|---:|---:|---:|
|1|28.656|23.8|11.4|
|8|66.704|79.9|37.3|
|16|95.364|86.9|52.5|
|32|132.629|144.0|70.8|
|128|199.271|179.3|101.3|
|256|239.468|230.1|108.9|

SGLang Marlin accounts for62/64/77/92/96/96% of its corresponding profiles. The owned kernel already fuses routing, both expert projections, BF16 intermediate rounds, SwiGLU, post-expert RMS and weighted reduction. Its isolated floor gap does not establish a removable decode or activation stall.

`_repack` validates positive-normal scale bytes and retains BOTH `repack_expert` and `repack_expert_fat` immutable banks. These are permuted FP4 code bits plus original8-bit scales, not predecoded BF16. Skinny layout is32768 code bytes plus4096 scale bytes per16-output-row/4096-K unit. `issue_unit` streams each unit into the original three-slot ring; `compute_loop` decodes four BF16 fragment registers, then HMMA. It reloads `xr` only when the physical job or phase changes, not on every output unit. PhaseA uses `__ldg`; phaseB uses `__ldcg` after the existing acquire of done_a. Each job retains at most8 live rows in64registers/thread atNW8. No new producer ownership or metadata assumption is introduced here.

## Representation decision

| representation | streamed unit | local32-expert bank | same3-slot total shared |
|---|---:|---:|---:|
|current codes+byte scales|36864B|905969664B|155520B actual control|
|preexpanded16-bit scale, codes unchanged|40960B|1006632960B|167808B arithmetic only|
|fully predecoded BF16 fragments|131072B|3221225472B|438144B arithmetic only|

Full BF16 expansion is3.556x the per-unit payload and cannot retain the current ring under the H200227KiB per-block limit. A one-slot/direct-global replacement would be a new pipeline/admission design, not an isolated decode optimization. Keeping the existing fat bank adds905969664B beyond the BF16 bank. Present skinny+fat packs together are1811939328B; preparation has already paid for both layouts, so removing a redundant pack would save setup/storage, not timed decode work.

Scale-only expansion fits the arithmetic capacity screen but adds11.11% stream/ring traffic and96MiB to one local32-expert bank. The existing `scale_word` is a byte permutation plus shift/add of the exponent bias, with no general FP8 conversion. Expanded scales would remove that construction, retaining the four BF16 pair products and all HMMA work. It has no measured exposed budget that offsets the extra traffic. This does not establish that scale expansion can never win; it establishes insufficient evidence for spending the next check on it.

The actual prior835 multiplier experiment is stronger than a static instruction argument: skinny833→801 hot-loop instructions,32HMMAs unchanged, actual130→128registers/zero local, yet full-six-load crosses only1.001248–1.002952 versus duplicate log-null.008029. It failed all frozen gain/noise/regression gates. Earlier mode84 NOSCALE-like WGMMA ablation serialized and is NOT a valid decode ceiling; old loads-only controls are explicitly older kernels and not a current absolute bound. No multiplier, SC6, position-map, wait, retirement, or band experiment is reopened.

## Activation reload accounting

A new exact CPU model intersects each canonical512-unit A /256-unit B job interval with all132 CTA floor-partitions and charges only live rows. A brute-force independent oracle covers zero routes, one route, partial9,256routes, all experts, and grids1/7/132/136. Retained64 local RTX natural banks are the preregistered census65001–65032 atM16/32, selected busiest local32-expert group. They are not new H200 inputs or timings; pinned census byte hash and original scope are retained in `audit.json`.

| M | mean weight requested | mean A activation requested | mean B activation requested | source X capacity upper |
|---|---:|---:|---:|---:|
|16|348585984B|2569984B|2569984B|131072B|
|32|501645312B|3654912B|3654912B|262144B|

Requested activation duplication is real but is not repeated HBM traffic: input X is at most128/256KiB for these loads, the pair-intermediate is also small, and read-only/L2 reuse may already serve the repetitions. No retained counter or acceptable-distortion observation attributes runtime to these loads. A zero-cost removal of ALL activation requests with latency proportional to total requested bytes yields only1.004862x full-six-load in this two-load scenario; this is neither a speed forecast nor a timing upper bound. It cannot revive the failed reorder family.

For M1 with a single local route, logical activation requests can be132*8192B per phase versus28.3MB weights, so the M16/32 proportional scenario does not close all tiny-input activation architectures. However, the underlying source row is only8KiB; eliminating logical requests is not evidence of eliminating an exposed HBM stall. No credible lower-token duplication fix follows from the current data.

## Validation, stopping rule and next action

`tools/moe_lowtoken_representation_audit.py` pins source and retained receipts, derives byte/CTA counts and six-load sensitivity, and proves1904 code/positive-normal-scale products exactly reproduce the existing scaled BF16 representation, including signed zero. Three CPU tests pass; no CUDA compilation was warranted because no conforming candidate passed the traffic/admission/evidence screen. This numerical proof does not qualify a kernel or GPU subnormal behavior.

Reproduce with `python tools/moe_lowtoken_representation_audit.py --root CHECKOUT --output OUTPUT.json` and `python -m unittest discover -s tools -p test_moe_lowtoken_representation_audit.py -v`. The checkout must include the retained tracked census and835summary. No raw judge weights were available or invented.

**Stop:** do not allocate a reset check to full BF16 expansion, scale-only expansion, or a generic activation cache based on this audit. No prototype or performance claim is hidden behind the audit tool. A future low-token proposal must first identify a separately exposed activation/decoder critical interval on the actual skinny backend, with controls that preserve decode/math/reader fences, before choosing an ownership or representation change. The retained fat-stage observer and invalid NOSCALE ablation cannot supply that interval. If such source-pinned evidence is absent, pivot away rather than repeat a reduced-instruction hypothesis.

For all future experiments retain the same prospective release gate: four private computing arms,3held-out natural banks across all six loads, all FP32 global.008/row.075/fresh/skew/nonlocal/replay checks, actual cooperative admission/resources and no regression larger than max(.5us,1%). All four six-load crosses must exceed1% and2xduplicate log-null for every seed. One improved load alone requires5.795% latency reduction to reach1% aggregate; two equal improvements require2.941%, four require1.481%. These are algebraic thresholds, not projected gains. Only root can authorize a reset measurement, and production changes require separate qualification.
