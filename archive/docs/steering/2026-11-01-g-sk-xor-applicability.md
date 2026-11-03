# 607e layer applicability of owned sparse-XOR projections

November1,2026; coordinator09:02 request. **No layer port, GPU check or public proposal is established here.** This separates the active607e layer interface from the standalone C2 SK experiment. Source: `muse_decode_layer/cand/g1b4_stage_release.py`,869076B SHA256`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`; public receipt607eb52bdaf24160 PASS13/2.1260826764x. Candidate inventory uses emitted prototype SHA`d23aa17be4cf9e7c5869803b3b4603d8e72395953687adc0287fb1c75738d586`, with native CUDA SHA in the evidence JSON. This is a metadata/source-size read, not independent clearance of changes after reviewed4907helper/source.

`tools/dl_sk_xor_applicability.py` verifies the exact G pin, parses actual dispatch constants, associates12profile rows with the owning receipt's12adjusted medians, and produces `evidence/2026-11-01-g-sk-xor-applicability/map.json`. Three CPU tests check coverage, score weighting and byte arithmetic. No standalone speedup is imported into this model.

## Actual dispatch and direct replacement boundary

607e pins small qkv/o/un to bfp static choices, sets pool/router to fused `bfpf` when available, and pins large128/256 projection picks to the existing Lt/torch path. Its bfp implementation is **SK2**, not C2 SK. It prepares different geometry-specific streams/tables. No existing G packed pointer is compatible with the new SX metadata/header format; any replacement must repack original BF16 matrices into separately owned immutable SX storage.

| T | qkv G config | O G config | unpool G config / rows |
|---|---|---|---|
|1|qkv8_r176|o8h_r88|slice8_r192 /1024|
|8|qkv8_r176 plus narrow override|o8h_r88|slice8_r192 /1024|
|16|qkv16_r176|o8h_r88 (TOKR16)|unpooler16h_r100 /8192|
|32|qkv32_r208|o32h_r100|unpooler32_r144 /8192|

T8 qkv actually invokes `_de_ext`: a separate narrow/global-X-escape source with altered escape reader counts, not the nominal SK2 body. Profiles show `sk2_kernel_narrow_5ca70d56e6dc`9.1/9.3us. The dead GLE deferred-UV path must not be used to infer2T rows: `GLE_T={}` and `PDL=False`, so actual `_sl_tile_after_attention` passes T rows to the pool/router front.

The following projection-boundary replacements are API-compatible without altering surrounding stages, conditional on new packs/module, independent math and actual resource qualification:

- QKV: BF16 `[T,8192]` times raw BF16 `[2304,8192]`→BF16 `[T,2304]`. Candidate qkv8F_nf atT1, qkv8F atT8, qkv16F atT16, qkv32 atT32. Preserve the BF16 QKV store before existing normalization/RoPE/cache/gates.
- O: BF16 `[T,1024]` times `[8192,1024]`→BF16 `[T,8192]`. Existing norm/gate, including FA3 fallback handling, must remain upstream. Candidate o16F_nf can coverT1/8; o16F/o32F cover16/32. These small1/8 replacements change token reservation/geometry from G and have NO measured layer evidence yet.
- Unpool: BF16 expert result `[T,4096]`→BF16 `[T,N]`; use ONLY first1024original output rows atT1/8, all8192at16/32. Candidate slice8F_nf/slice8F/unp16F/unp32F match this ABI. Return original `(view(1,T,N),1)` behavior where intercepted by dense; retain sliced-boundary semantics rather than zero-pad or expand into a full unpooler.

Thus24SL/full projection cells across4short loads have a potential unchanged-stage ABI. They are not24qualified wins. Many geometry/ring/accumulator/FULL/NOVOTE choices differ between C2 and G, so replacing a projection compares complete consumers, not an isolated XOR effect on SK2. Native SX is restricted to COMP, !HL, CL1, !FUSE, !ESCG. T1 uses non-FULL variants; do not let a FULLM8/M16 kernel overwrite runtime M. Unsupported weights/configs must preserve original G fallback. State/streams/outputs/diagnostics must remain private where writable; packs may be shared read-only only after matching original weight identity.

## Pool/router requires a new extension

AtT1/8 actual `_pool_router_fp32` dispatches `_gle_frt(h,hr,state)`: one fused kernel, pooled BF16 `[T,4096]`, logits FP32 `[T,256]`. Router consumes FP32 `hr` from the boundary, while pooler consumes BF16 `h`. Profile identity is `sk2_kernel_gle4944bfa97e00`,17.9/17.9us(T1 SL/full),18.7/18.6us(T8). The previously measured split-overlap candidate lost publicly; its gains must not be assumed here.

AtT16/32 actual `bfp_fused_pooler_router` uses the fixed `state['sk2']['pooler']` geometry (NT4,NWR1,WSK8,UPW1), not the separate `bfp_SK2_PICK` pooler32 geometry. It appends the router continuation, with BF16 hi+lo router operands/FP32 accumulation and existing global partial/counter lifecycle. Profile times are18.9/17.3us(T16) and20.4/19.6us(T32). Current SX explicitly forbids FUSE/HL; replacing pool alone would split this pipeline and is not an immediate like-for-like layer port. Neither using BF16 h in place of FP32 hr at1/8 nor dropping the router continuation/join is numerically or structurally covered.

A future fused extension must separately prove pool-only repaired fragments, untouched router arithmetic/input precision, full transaction accounting for both phases, counter generations/reset and reader retirement. Merely removing `!FUSE` from the static guard is not a valid port. Router itself is not a target for this E7 sparse repair. Verify128/256 Lt projections remain unchanged throughout this applicability map.

## Actual12medians and conditional whole-layer sensitivity

The table uses607e adjusted per-cell medians; kernel-profile times elsewhere are separate instrumented observations, not additive causal stall measurements.

| load / T | sliding us | full us | qkv+O+un profile us SL/full |
|---|---:|---:|---:|
|decode1 /1|76.748101|83.726757|20.9 /21.0|
|decode16 /16|138.660714|214.534360|31.6 /31.1|
|decode32 /32|154.820445|297.882697|33.6 /33.5|
|verify1 /8|115.431860|117.597840|19.8 /19.9|
|verify16 /128|247.130995|333.218126|unchanged Lt|
|verify32 /256|316.389537|482.656794|unchanged Lt|

For each load j, Cj=46*tSL+16*tFull. A scenario's relative factor is `[product_j Cj/(Cj-46*dSL-16*dFull)]^(1/6)`. Baselines cancel in this sensitivity calculation; it is not the actual score of an unmeasured candidate.

For1%full-six-load improvement, assuming equal savings in both contexts across all4short loads, required whole-cell saving is **1.800217us** per call; if evenly distributed over the3direct projections, **0.600072us per projection**. Assuming savings proportional to their separately profiled qkv+O+un times, required fraction is **7.3021%**. Hypothetical1/2/5/10% removal of those profile times gives full-layer factors1.001355/1.002714/1.006820/1.013760. These are algebraic scenarios, not predicted gains or measured upper bounds. They exclude pool/router, MoE, attention and all128/256changes. Standalone30-cell score weights and measured candidate wins cannot be substituted for this46/16six-load function.

## Concrete slim byte plan, without the old full-owned backend

The earlier399KBfull-owned verify backend is NOT required. Retain full factored G720525B raw, append a separate owned SX extension and its pack/run closure:

| item | source bytes |
|---|---:|
|full G license-preserved factored source|720525|
|full current SX native literal, visible raw text|88712|
|SX C++ declaration literal value|474|
|14needed pack/build/run helper definitions (including control-pack dependencies)|10756|
|all10config metadata repr|3519|
|explicit adapter/import/glue allowance|4096|
|additional notice allowance|2048|
|source-plan sum|**830130**|

This leaves substantial arithmetic headroom under1MB even retaining all native40control/treatment/nearby entries. It is an inventory/allowance plan, NOT a built combined artifact or a proof that an eventual dispatcher fits. Actual namespace isolation, literal quoting, dependencies/constants, attribution, compile registration and source pins must be verified when building. G already imports Torch/load_inline; only the SX closure is needed, not full original C2/projection/Fa3/MoE. Candidate helper names should be isolated rather than shadow G functions. Fixed selections could later prune entries, but this estimate does not require doing so or special-casing active CUDA math.

## Priorities and gates

First finish the standalone matched experiment and preserve its scope. If it passes, compare the exact qualified SX consumers against actual G SK2 consumers at the projection boundary; do not transfer C2 gains. Prioritize direct qkv/O/un cells with enough measured coverage to exceed the1.800us aggregate scenario requirement. Keep PR fused and128/256Lt unchanged. A future same-SK2 repaired-fragment version or fused PR extension is a new implementation/review, not implied by this map.

Before any public layer proposal require: pinned source/selected builds and actual G dispatch logs, all supported fallback/weight coverage, unchanged BF16 boundaries and FP32 router, fresh native+FA3/normgate/sliced-unpool validation, per-owner graph/counter lifetime checks, actual runtime resources/admission, prepare<=300s, complete source<1MB, and a paired full-six46/16score gain>=1%+2xnull with per-seed/no-regression gates. No current candidate is eligible for such a proposal based solely on this audit. No port or GPU/public action was performed.
