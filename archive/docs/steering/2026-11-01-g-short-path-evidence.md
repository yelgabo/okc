# G short producer-to-expert path: bounded fixed-routing diagnostic

November1,2026. Evidence: public layer607eb52bdaf24160, source commit0ce0b23; current coordinator throughcd010c8. Exact scored file869076bytes, SHAba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. This round performs CPU-only source/transport validation. No GPU job or public submission was dispatched and no new release candidate qualifies yet.

## Measured opportunity and preregistration

607 passed13official cases at2.126082676x, CI[2.094191436,2.157973917]. Its D1 sliding/full medians76.75/83.73us give weighted78.5512903us. V1 medians115.43/117.60 give115.990us. Exact scoring uses six equal geometric loads, each46sliding+16full. Holding the other four loads fixed, subtracting the SAME hypothetical exposed time from both contexts of both short loads gives:

| Hypothetical time removed per short cell | Six-load factor |
|---|---:|
|1us|1.003584923|
|2us|1.007223308|
|3us|1.010916603|
|5us|1.018474017|

These are sensitivities, not projected gains. Before implementation the coordinator received the stop rule: require a credible approximately3us combined-short-path opportunity (~1%whole score), not a sub-.5%microknob alone; no quota or public slot spent here. Diagnostic deadline295s including normal preparation; actual completion cost remains unknown.

## What the profiles establish

| Owning receipt/cell | Fused pool/FP32router activity | Native expert activity |
|---|---:|---:|
|G0 3a5f879df0d44825 D1sliding/full|17.6/17.4us|23.6/21.8us|
|607 D1sliding/full|17.9/17.9us|29.8/42.8us|
|G0 V1sliding/full|19.9/18.6us|44.2/57.6us|
|607 V1sliding/full|18.7/18.6us|71.6/58.8us|

Source: respective archived profile text files under muse_decode_layer/profiles. These are separate inputs/preparations and profiler activities, not matched exposed times. Actual expert routing/job ownership was not retained in those profile rows. The near-equal producer activity does not explain the endpoint difference; larger expert differences motivate fixed legal routing before attribution. G0 also uses a GLE O-epilogue/_gle_route on D1;607 uses separate b1/_sl_tile_route and PDLoff. G0 is not a conforming replacement inferred from these numbers. Activity sums differ from whole-graph medians and receive no speedup credit.

Existing closures remain scoped: rank-zero short RMS1023d9b0dcaa4c75 failed/incomplete its whole-layer gate; old owned router wins and high-token split-pooler losses do not establish this T1/T8 mechanism. The failed whole-file comment-strip caad524754b54680 is not reused. Native source literals remain byte-exact in this diagnostic.

## Exact path, precision and dependencies

The selected `_sl_tile_after_attention` at15862 allocates h BF16 and hr FP32, calls b1, then `_pool_router_fp32`. Latest selector at16642 chooses `_gle_frt` for T1/T8, actual ppick bfp/bfpf, contiguous h/hr, FP32 router weights. `_gle_frt` at16622 returns pooled BF16[T,4096] and logits FP32[T,256]. Its128 pool tiles also compute the router's four quarter partials per eight-row group. Existing per-group counters certify only that group's router reduction; they do not certify all4096pooled columns or all256logits for a global norm/top8.

`_sl_tile_route` at15794 has T*(PARTS+1)programs: PARTS1atT1,4atT8. Norm programs read ALL4096pooled BF16 values, reduce squares in FP32, apply pre-expert gamma formed FP32, and store xm BF16. The router program consumes all256FP32logits, FP32sigmoid/bias/top8/weight normalization; IDS/LIDS int32 and WTS FP32. Expert launch consumes actual xm/lids/wts. Its weights are immutable, but selected expert identities and job assignment depend on the complete router result.

PDL=False; route launch_pdl=False; native repaired launch is ordinary same-stream cooperative. Independent GPRE/bias/LS loads already sit before route's gdc_wait under PRELOAD, but current serialization offers no producer overlap. Merely moving those small loads cannot justify3us. Launching selected expert weight work before IDs exist lacks a legal address proof; speculative all-expert reads lack measured benefit. A fused producer finalizer would need ALLpooler/router producer coverage, release/acquire publication and generation-safe reset. Existing quarter counters alone are insufficient. No synchronization change is made or proposed as cleared.

## Concrete prospective comparison

The only prepared alternative invokes the ALREADY PRESENT serial path `_gle_prior_pool_router_fp32`: BF16 h→pooler and FP32 hr/weights→FP32router, then the same route and native experts. It preserves activation boundaries, FP32 router math, natural IDs, fences and ordinary ordering. Its FMA reduction tree may change rounding; independent model/routing/cache gates remain mandatory. It has no measured win and is NOT an eligible public release.

A future pure switch must apply after the unchanged normal preparation, exactly as the diagnostic's per-state switch. Flipping global GLE_FRT before prepare would also change small-T tuner execution/RNG/choices and is a distinct experiment. Only a completed useful trace could justify implementing/qualifying the post-prepare single-arm variant. All12official scoring cells and official13tests remain required for its release.

## Reviewable diagnostic and ownership

New tools: dl_short_path_trace.py, dl_short_path_body.py, dl_short_path_analyze.py, dl_short_path_test.py. Generation uses exact607source plus observational Python capture of the real route tensors; ALLCUDA/CPPliteral values are unchanged. No native compile flag, barrier or fence changes. Source903137bytes, SHA8308dcf30f03ec5fec69160c6a4bb1939577b608890495a4bd018c46e926f99b.

One normal607prepare, then four private mutable owners (fused1/fused2/split1/split2) via the existing alias-preserving clone. Packed weights/immutable tables shared; nested Lt workspaces, attention ownership and mutable storage private, overlap/unknown owners fail. Actual natural IDS/LIDS must agree across all4arms AND every timed/profiled replay. No IDs are supplied, replaced, sorted into a different model, or fitted to timings. Reference top8 chooses the official busiest32expert range before data clones; actual occupied work is required and its counts are reported.

Three preregistered seeds401031/401032/401033, D1/V1, both contexts;20random cold replays per arm and3CUDA-only profiler replays.96FP32model/routing/cache validations, private guard tails, exact producer coverage, same selected expert binary and PDLoff are fail-closed. A trace must show one FRT kernel, or the serial pooler+FP32router pair, between b1 and route, then the actual native expert. Whole graph span, PR→route gaps, route→expert gap and PR-start→expert-end intervals are reconstructed from retained events; never sum activities into a gain. Fixed input repetitions are not advertised as every-replay fresh mathematical references.

The judged primary returns the ORIGINAL normal prepared state, preserving all-shape caches/attention lifetime. Private short-only owners are never returned. Bindings and mutable-storage inventory must match the pre-diagnostic production state. Diagnostic `_AB_DEQ` is cleared before/finally after prepare; TF32global restored even on failure. No synthetic activation/reference cache is retained in production.

## CPU gates and remaining limits

Seven CPU tests pass: exact native literal scope; BF16pool/FP32router interfaces; full producer/route/expert interval topology; six-load math/null/regression; emitted reference-helper recursive global resolution; cache cleanup/primary owner; full48row output transport. The coordinator caught shared PRELUDE's missing ROUTE_EPS; final generation uses root's fixed1e-15 binding. Helper root file LF SHAa3871836f4ca6b140fc7343038bd026947c38278654b7216db4f3c71840f1bc3.

The initial plausible codec fixture overflowed after all-work output. Final actual-code fixture48rows×20samples×10events encodes9520bytes under11000. Every row/event/sample remains; sample metadata prints at.001us, scoring medians remain UNROUNDED. Only redundant chain summaries are omitted, then reconstructed offline from full retained0.1us event coordinates. Actual payload size still fails closed. Budget checks run inside rounds, arms and profile loops. No H200 host/owner facade or performance result is claimed.

Potential cost: normal preparation/startup, four mutable copies,12case banks,960cold replays,144profile replays,96validators with dual FP32expert-back references.295s and memory preflight may fail; no partial coverage may qualify. Structural production-owner checks do not substitute for actual H200 graph correctness. No physical HBM/SM counters or internal expert wait cycles are collected.

Acceptance: all48rows/96checks and terminal official control13PASS; all three seeds' four-cross full-graph six-load counterfactual≥1% and greater than twice duplicate log variability; each affected context regression≤max(.5us,1%control). Other four loads held fixed. Failure closes this serial-path hypothesis; no automatic PDL, prefetch, graph or fusion sweep. A pass only authorizes proposing a distinct pure candidate gate, never submitting this comparator publicly.

Generate offline with `python tools/dl_short_path_trace.py muse_decode_layer/cand/g1b4_stage_release.py .build/today/short-path.py --root .`. Run tests with `python tools/dl_short_path_test.py`; worktree override is explicit OKC_REVIEW_ROOT. No check/submit command was executed. Coordinator owns integration/index/push and any future quota decision.
