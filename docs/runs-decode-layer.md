# muse_decode_layer runs

Latest captain terminal addendum (2026-11-02 09:53UTC): PRIVATE **e1271ad5e5e34b24**, finished09:42:17.129UTC; prereg5b997e07, acceptance73e4774, terminal217a98c. Notice-corrected0d84d0e/999492B, native53d09576 unchanged. H200 done/passedtrue/return0,13tests/12benchmarks/12profiles,56native startup replays; **1.9746646855150114x CI[1.9450447152322863,2.0042846557977363]**, measured prepare98.749577628s; queue834.882672s/worker203.656080s distinct. Raw profiles/log/receipt under steering/evidence/2026-11-02-worker-g224-notice/variant.e1271ad5e5e34b24.* and terminal-status-resumed.json;33manifest entries+manifest captain-verified. Ordinary target correctness PASS, performance descriptive only; changed long-load warnings against both historical controls do not establish a causal loss. Exact candidate PARKED/no repeat/rescue/public/default action. Independent analysis steering/2026-11-02-captain-g224-terminal-decision.md. Wait27522 lost during tool transition; same-ID status wait32878 completed, no second POST. Public607 unchanged.

Every `okc check` on `muse_decode_layer`, newest last. Scores are the judge's speedup over
SGLang with its 95% interval. Profiles and logs are in `muse_decode_layer/profiles/`. The
submission is baseline.py's call sequence with pieces ported from muse_moe_w4a16 and
muse_bf16_proj and muse_attention_decode, each behind a switch (`PORT_MOE`, `PORT_MOE_W4`,
`PORT_ROUTER`, `PORT_PROJ`, `PORT_ATTN_PREP`, `PORT_ATTN_FUSED`) and a self-test against
SGLang's path (or, for the W4 GEMMs, an FP32 reference) in prepare().

Per-shape columns, in TASK.md's order: d1 = decode 1, d16, d32, v1 = verify 1 (8 tokens), v16
(128 tokens), v32 (256 tokens); s = sliding layer, f = full layer.

## How the decode layer calls the ported pieces (checked against baseline.py)

- MoE: same Marlin bank, workspace, `is_ep=True` and eps 1e-8 as muse_moe_w4a16. Differences:
  `local_start` is a device tensor (the MoE problem had it in the config), so our align kernel
  reads it from device memory; the post-expert norm weight is SGLang's effective weight
  (`post_expert._get_effective_weight()`), passed to both paths.
- MoE GEMMs (`PORT_MOE_W4`): muse_moe_w4a16's W4A16 kernel source, repack and dequant copied
  unchanged into a second extension; its 7 configs that lost everywhere in that problem's
  tuner (check 84f3975334f048c3) and its timing-only modes are left out (31 GEMM configs). The
  decode layer's route_align (local_start read from the device) now also emits the W4 work
  items. The tuner and the FP32 self-test run in prepare() on synthetic routings with
  local_start 0 and 96; the finalizer gets the effective post-expert norm weight, as Marlin's
  path does. Policy key is T (1, 8, 16, 32, 128, 256; other T use the next one up).
- Fused MoE (`PORT_MOE_FUSED`): muse_moe_w4a16's one-launch kernel, assembled by
  tools/dl_moe_inline.py from source commit 8a8f6e4 for the computing variants only (skinny
  sk8w4s and sk16w4s; fat main-d2 from 17 tokens; the source's own tables, trimmed as its
  guide asks) inside a nested `moe1` namespace and built as its own
  extension (`okc_decode_layer_moe1`) on a thread from the top of prepare(). The inliner
  gives Params a `lo_p` pointer so both kernels take the local expert range from the
  layer's `local_start` tensor on the device (as route_align does) and copies the repack
  (16-row units and 256-row x 16-stage units, 906 MB each for the 32 experts) under a
  moe1_ prefix. Scratch in `state["moe1"]`, allocated once; the kernel resets its own
  flags and alternates the barrier counter and sums of squares by generation. The post-expert
  weight is the layer's effective 1 + w (`post_expert_w`) widened to FP32; the kernel rounds
  to BF16 where the Marlin path does (gate and up before SiLU, the SwiGLU output, GEMM2's
  output before the norm). Wiring: `_moe_opts` carries the chosen variant; `_route_meta`
  then returns only the routing tensors (no route_align on the side stream) and
  `_experts_run` makes the one launch on the main stream after the join, with no PDL
  attribute (both kernels begin with `griddepcontrol.wait` since 8a8f6e4, so the attribute
  would be safe) and no early trigger, since its grid barrier needs all 132 CTAs resident; the
  pre-expert norm stays W0's kernel. Policy: `_moe1_tune` times the W4 chain and every
  passing fused variant per T in (1, 8, 16, 32, 128, 256) as cold-L2 graph replays of the
  main-stream MoE call (metadata built outside the graph, as the side stream builds it) and
  keeps the fastest. Self-test (`moe fused`): every variant vs the FP32 reference at T 1-264
  including 257 (normal at local_start 0, skewed at 96; bound 8e-3) with the W4 chain's
  distance printed, the scratch invariant (spin flags, next barrier counter, next sums of
  squares, per-job flags) after each, then 2,000 graph replays at T 8 (skinny) and 128 (fat)
  with the generation count, the replayed output against an eager call and the invariant.
- Router: up to 8 tokens SGLang routes on the boundary's FP32 router input, not the BF16 FFN
  input, and the reference routes on full-precision h. The muse_bf16_proj router kernel took
  BF16 x, so the port stages FP32 x instead. Above 8 tokens SGLang uses BF16 hi + lo on the
  BF16 FFN input, as on muse_bf16_proj. The router runs on a side stream, overlapping the pooler.
- Projections: qkv_gate, pooler and the unpooler's 1024-row slice at T <= 8, o at T = 1, as
  `_proj_fused_wins` chose on muse_bf16_proj.
- Skinny projections (`PORT_PROJ_SK`): muse_bf16_proj's sk kernel and GEMVs, assembled by
  tools/dl_proj_inline.py for the layer's 26 configs (12 geometries compressed and raw, the
  two fused pooler + router configs; the standalone router configs stay out) as a fifth
  extension (`okc_decode_layer_skp`) on a thread from the top of prepare(); the packing
  code is copied verbatim. The kernel has no griddepcontrol, so every launch is plain (the
  PDL secondaries after it start at its completion). Prepare packs qkv_gate (rows balanced),
  o, pooler and the unpooler (full and the 1024-row slice) in 12 bits with the 1/200 escape
  rule and raw BF16 fallback streams, one stage-major stream per geometry, escape tables per
  (R, rows), cgemv's row stream and sgemv at m = 1, and the router's hi/lo stream for the
  fused entry. Self-test (`proj sk`): every candidate against FP64 at the bucket's token
  counts (1, 2, 5, 7, 8; 9, 12, 16; 17, 31, 32) with probe rows (relative L2 < 5e-3, probe
  rows bitwise, no timed-out wait), the fused entries at 1, 3, 8 and 9, 16 (router < 2e-5);
  failures are dropped with a printed reason. Autotune: per (layer, m) for m in 1, 8, 16, 32,
  the survivors against today's path (proj_fwd or cuBLAS) as cold-L2 graph replays; the pick
  (None keeps today's path) is used by `_proj` for every m <= 32, other m sharing the
  bucket's pick; the full unpooler above 8 tokens now also goes through `_proj`. The fused
  pooler + router is the `skf` front variant, tuned at T 1, 8 and 16 beside fp32 (T <= 8) or
  split / mma / concat; a variant that disagrees with the reference is dropped at that T
  instead of failing the tuner.
- Attention: the decode layer feeds attention exactly as muse_attention_decode does (qkv
  [T, 2304] BF16 from the qkv_gate projection, q 16x64 | k 2x64 | v 2x64 | gate 16x64; RoPE
  interleaved with theta 500000 on sliding layers only; FP8 caches [pages, page, 2, 64]; the
  norm-gate has no weight, eps 1e-5). FusedRMSNormGate gives the same bytes as our gate launch
  (self-test rel diff 0). The kernel source is muse_attention_decode/attn_kernel.cu at 3b84d10,
  copied in by tools/dl_attn_inline.py inside a nested `attn` namespace; only its q_len 1
  instances are built. Routing as on muse_attention_decode: fused kernel on q_len 1 except
  full layers at batch 1-2; idea 1 + FA3 elsewhere.
- Verify attention (`PORT_ATTN_WS`): muse_attention_decode/attn_ws.cu copied in by
  tools/dl_ws_inline.py inside a nested `verify` namespace of the same extension; the host
  wrapper launches var 16 only (fenced loaders, FP32 Q.K and softmax). It takes the same qkv,
  seq_lens, page table, cache_loc and FP8 caches as the q_len 1 kernel, writes the new cache
  rows from its last split, and its combine launch merges the splits (FP32 partials in a
  per-call buffer) and applies the norm-gate. Routed on sliding verify at every batch and full
  verify above batch 2.
- One-launch attention (`PORT_ATTN_ONE`): muse_attention_decode/attn_one.cu copied in by
  tools/dl_one_inline.py inside a nested `one` namespace, built as a third extension
  (`okc_decode_layer_one`) on a thread started at the top of prepare() so its compile
  overlaps the main and W4 builds. Eleven instances: q_len 1 modes 1, 3, 11, 13, 49 and
  q_len 8 modes 1, 3, 11, 15, 19, 51 (the plan's modes for the 12 scored shapes and the 13
  tests). Since 075525d full verify at batch > 1 takes the producer build 51, with 19 as the
  fallback when 51's residency is short: the producer arm now leaves by PTX `exit`, so 51
  compiles to 80 registers with no spills (the layer's copy, checked with
  `tools/dl_compile_check.sh muse_decode_layer/submission.py _ONE_SRC`) and timed 2.5-2.7%
  under 19 standalone (d90a73871d8e4481). 17 and 115 are left out. Since d60c478 the copy
  carries the once-path diet (ONE_DIET 1, ONE_DIAG 0), so the page size must be a power of
  two (`_one_wins` and the host wrapper both check). The inliner adds two
  `griddepcontrol.wait` lines (before the query group, whose rows are the projection's
  output and whose constants group reads the epoch; before the flat kernel's epoch read),
  so the launch is correct with or without the PDL attribute; `_ONE_PDL = False` keeps the
  plain stream semantics the standalone checks exercise, and the kernel sets no early
  trigger (the o projection, a PDL secondary, starts at completion). Persistent workspace
  in `state["one"]` (16 MB of FP32 rows, row sums, 32K counters), allocated and zeroed once
  in prepare; the cooperative modes launch through cudaLaunchKernelEx with
  cudaLaunchAttributeCooperative after a graph-capture probe (a plain launch then a
  cooperative one, as attention follows the projection). Self-test (`attn one`): the shared
  cases plus batch 1 full at page 1 and 64 for both q_len and the ctx-10 sliding case,
  the workspace invariant (rows, row sums, tickets, next-parity counters zero) after each,
  then 2,000 graph replays on decode 1 sliding, decode 16 full, verify 1 sliding, verify 16
  full, (since 075525d) verify 32 full and (since c306013) decode and verify 1 full with the replayed output against an eager call (< 1e-4) and SGLang's
  path (< 5e-2), and on every batch-1 shape the speculated length forced wrong (seq - 64,
  50 replays each).
  Routed on every tp8 shape at q_len 1 and 8 unless `_ONE_ROUTE` excludes a
  (layer, q_len, batch); the fused, ws and prep paths stay as fallbacks.
- Selection summary (since 075525d): the judge returns only the last 16,000 characters of the
  log, so the per-stage lines from early in prepare (MOE1BENCH, MOE1POLICY, TUNE) can fall
  off the front. The last thing prepare prints is a block of about 2.5 KB of `SUMMARY` lines:
  a digest of the module's own bytecode and upper-case string constants (comments do not
  change it; the result JSON's `content_hash` identifies the exact payload), a 12-hex hash of
  each extension's generated CUDA source (main, one, w4, moe1, skp), the on/off state of every
  stage, per MoE token bucket the chosen variant with every candidate's bench time (judge-like
  routing since 1b5bb52, with the busiest local expert's route count per seed), the W4
  configs and the MoE-front variant, the attention path and `_one_plan` (splits, copies,
  mode) for each of the 12 scored shapes (planned at ctx 80000), and every failed candidate
  or fallback (self-test failures, dropped tuner variants, stages that fell back to SGLang).
- Stream overlap (`ROUTE_ON_SIDE`): route_align runs on the router's side stream after top-8;
  the pre-expert norm runs on main before the join; a graph-capture self-test guards it.
- FFN boundary at T <= 8 (`PORT_FFN_BOUNDARY`): reads SGLang's module tensors
  (`_post_norm._frozen_weight`, `_residual_add._alpha` / `_beta`, `_pre_norm._frozen_weight`),
  self-tested against the padded FusedPostNormResidualPreNorm call.

## Runs

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-28 | c61c26cec41d425e | cd71cb3 | MoE port only: own align (local_start from device), one-kernel finalizer, block_size_m 8 below 12 tokens else 16 | 1.168x (1.150-1.185) | per shape d1 s/f 1.222/1.205, d16 1.204/1.117, d32 1.043/1.021, v1 1.168/1.161, v16 1.261/1.194, v32 1.219/1.148. 13/13 tests; prepare 42.6 s (extension build 41.0 s); MoE self-test worst rel diff 4.85e-04 vs SGLang's path |
| 2026-10-28 | 6ef2d969b6104e10 | e3e2c61 | + router port: FP32-input CUDA-core kernel at T <= 8, hi/lo mma at 9-32 | 1.182x (1.164-1.199) | per shape d1 1.221/1.211, d16 1.196/1.102, d32 1.042/1.019, v1 1.308/1.291, v16 1.252/1.184, v32 1.195/1.111. Gain is verify 1 only (+0.14 on both layers: SGLang's FP32 F.linear at 8 tokens). d1, d16, d32 within noise of the MoE-only run: the router runs on a side stream under the pooler, so a faster router does not shorten the step there |
| 2026-10-28 | e6944460b65d46ce | 3f35116 | + projection port: mma kernel for qkv_gate, pooler, unpooler slice at T <= 8, o at T = 1 | **1.207x** (1.188-1.225) | per shape d1 1.276/1.256, d16 1.210/1.111, d32 1.042/1.019, v1 1.384/1.328, v16 1.253/1.187, v32 1.228/1.130. Gain on d1 (+0.05) and v1 (+0.04 to +0.08), the only shapes that use the kernel; prepare 43.3 s |
| 2026-10-28 | 46dab8fb87274880 | daf59dc | as e6944460 but SGLang's split router at 9-32 tokens (isolates the mma router) | 1.201x (1.183-1.219) | per shape d1 1.269/1.253, d16 1.221/1.132, d32 1.045/1.018, v1 1.381/1.341, v16 1.209/1.148, v32 1.213/1.138. d16 +0.011/+0.021 and d32 flat against e6944460; adding the mma router (c61c -> 6ef2) had moved d16 by -0.008/-0.015. Both inside per-shape run-to-run noise (v16 moved -0.04 here with no change on its path), but in the same direction, so the final file keeps SGLang's split router at 9-32 |
| 2026-10-28 | 003409c1282043d6 | a45b08c | attention idea 1 (prep launch + FA3 + norm-gate launch, from muse_attention_decode) behind a self-test | 1.205x (1.187-1.223) | self-test failed (output rel diff 5.35e-02, new cache rows 1.72e-02, bound 2e-2), so attention fell back to SGLang: this is a repeat of the e6944460 configuration. Per shape d1 1.267/1.247, d16 1.227/1.129, d32 1.051/1.021, v1 1.363/1.356, v16 1.230/1.162, v32 1.231/1.138 (v16 full -0.04 and v32 sliding +0.03 against 2c49c79b with no code change on their path: per-shape run-to-run noise). The bound was wrong, not the kernel: SGLang's FusedQKNorm rounds k to BF16 before the FP8 cast, and with unit random keys (logit std ~5.5) one-ulp FP8 query differences move the output by several percent |
| 2026-10-28 | 6b5e79af2654442d | 212350a | attention idea 1 on (self-test conditioned: keys x0.3, values with a per-dim mean; new-cache bound 4e-2, output 5e-2), idea 2 compiled but off | 1.220x (1.202-1.238) | per shape d1 1.301/1.295, d16 1.258/1.143, d32 1.069/1.026, v1 1.413/1.374, v16 1.238/1.166, v32 1.195/1.124. Against the repeat run 003409c1 (same code otherwise): d1 -4.3/-7.1 µs, d16 -9.5/-9.8 µs, v1 -14.3/-1.7 µs; d32 and v16/v32 inside noise (v32 sliding -0.036 vs 003409c1, -0.005 vs 2c49c79b). Self-test: norm-gate identical to FusedRMSNormGate (rel diff 0); whole-path output 0.4-3.6% from SGLang's (one-ulp FP8 query differences), new cache rows 2.8%. The judge's 13 tests pass. Score gain sits inside the 95% interval of the previous runs |
| 2026-10-28 | 2919e4a616b24e90 | ee4dc77 | + attention idea 2: fused q_len 1 kernel on sliding decode (all batches) and full decode at batch > 2; full batch 1 and all verify stay on idea 1 + FA3 | **1.268x** (1.249-1.287) | per shape d1 1.330/1.287, d16 1.249/1.357, d32 1.089/1.304, v1 1.419/1.394, v16 1.261/1.186, v32 1.229/1.137. Against 6b5e79af (idea 1 only): full d16 378.5 -> 318.0 µs (attention kernel 130.2 µs + 4.3 µs merge vs FA3 191.5), full d32 591.9 -> 478.4 µs (250.2 vs 376.7); sliding d1/d16/d32 129.3/209.0/241.8 vs 131.3/206.3/237.8 µs, inside noise (fused kernel 14.0/15.3/20.9 µs vs FA3 main + combine 13.5/18.3/18.9 plus the prep and gate launches). Shapes not on the fused path moved inside noise. CI clear of the previous best's (1.188-1.225). prepare 47.0 s (extension build 44.6 s) |
| 2026-10-28 | 6c9eb0ed5d3e464e | d0e69df | + W4A16 expert GEMMs from muse_moe_w4a16 (GEMM1 with SwiGLU, GEMM2 with the FP32 finalizer, PDL), config per T tuned in prepare() with Marlin as a candidate | **1.384x** (1.364-1.405) | per shape d1 1.445/1.369, d16 1.383/1.418, d32 1.165/1.362, v1 1.580/1.560, v16 1.415/1.284, v32 1.364/1.229 (vs 2919e4a6: every shape up, +0.06 to +0.17). Our µs vs 2919e4a6: d1 129.3 -> 119.8 / 142.3 -> 134.4, d16 209.0 -> 189.2 / 318.0 -> 310.5, d32 241.8 -> 224.9 / 478.4 -> 455.7, v1 181.9 -> 169.5 / 195.6 -> 172.9, v16 414.6 -> 376.9 / 605.2 -> 537.6, v32 591.9 -> 527.9 / 937.7 -> 878.1. Marlin won no GEMM at any T. Tuner (synthetic routing, cold L2, graph replay incl. about 5 µs overhead, Marlin -> best W4): GEMM1 + activation T1 37.3 -> 31.0, T8 65.2 -> 51.9, T16 103.2 -> 77.8, T32 138.2 -> 102.5, T128 205.7 -> 165.9, T256 273.3 -> 243.5; GEMM2 + finalizer 29.3 -> 25.2, 42.4 -> 34.5, 61.3 -> 48.0, 79.0 -> 61.3, 112.9 -> 90.7, 149.9 -> 121.0. Policy {1: (27, 21, pdl, cl 8), 8: (26, 24, pdl, 1), 16: (18, 28, pdl, 8), 32: (26, 28, pdl, 4), 128: (4, 29, pdl, 1), 256: (30, 13, pdl, 1)}. W4 vs FP32 3.67e-3 to 3.87e-3 at T 1-264 (Marlin path 4.61e-3 to 4.92e-3). Profile per-kernel MoE times overlap under PDL (`tools/dl_moe_split.py`), so the scored µs are the measure. 13/13 tests; prepare 95.5 s (main extension 42.8 s, W4 extension 46.8 s, tuning 2.7 s) |
| 2026-10-28 | 04e07f16944f4107 | 26e17ec | + FP32 warp-specialized q_len 8 verify kernel from muse_attention_decode (var 16 only, fenced): sliding verify at every batch, full verify above batch 2; full verify b1-2 stays on prep + FA3 | 1.395x (1.375-1.416) | per shape d1 1.403/1.352, d16 1.406/1.419, d32 1.196/1.416, v1 1.621/1.554, v16 1.404/1.311, v32 1.371/1.263. Our µs vs 6c9eb0ed on the routed shapes: v1 sliding 169.5 -> 160.9, v16 376.9 -> 367.7 / full 537.6 -> 535.0, v32 527.9 -> 527.0 / full 878.1 -> 854.8. Profile: ws kernel full b16 188.6 µs (FA3 main kernel 191.4 in 6c9eb0ed), b32 348.4 (375.4); sliding b1/b16/b32 11.7/18.1/22.0 µs. Shapes off the ws path moved inside noise (d1 sliding -0.04, d32 full +0.05 with no code change there). Score gain inside the CI. Self-test vs SGLang's path: ws output 2.3e-2 to 3.7e-2 on 9 verify cases (prep + FA3 2.3e-2 to 3.6e-2 on the same cases), new cache rows 2.8e-2. 13/13 tests; prepare 96.6 s (main extension 43.8 s with the ws kernel, was 42.8; W4 extension 46.7 s). ptxas (tools/dl_compile_check.sh): attn_ws_kernel<7, 3, 0, 4, 16> 115 registers, no spills. The run also printed the FFN boundary's attributes (DESC lines in the log) for X5 |
| 2026-10-28 | bd28a84740ac4b84 | 9789d23 | + X1: route_align (and its work lists) on the router's side stream right after top-8, pre-expert norm on the main stream before the join; the streams join right before GEMM1. `_experts` split into `_route_meta` + `_experts_run` (one-stream `_experts` kept as the control and for the tuner and self-tests) | **1.433x** (1.412-1.455) | per shape d1 1.469/1.414, d16 1.500/1.469, d32 1.174/1.363, v1 1.675/1.617, v16 1.426/1.336, v32 1.430/1.288. Our µs vs 04e07f16: d1 121.4 -> 117.8 / 133.1 -> 129.1, d16 190.3 -> 171.8 / 316.0 -> 292.3, d32 220.3 -> 206.5 / 444.6 -> 440.4, v1 160.9 -> 156.9 / 178.4 -> 168.2, v16 367.7 -> 359.6 / 535.0 -> 538.3, v32 527.0 -> 522.2 / 854.8 -> 857.9. Down on 10 of 12 shapes; v16/v32 full +3 µs inside noise. d32 sliding's speedup fell (1.196 -> 1.174) while our time fell 13.8 µs: SGLang's time on it moved 263.4 -> 242.5 between runs. CI overlaps 04e07f16's (1.375-1.416) at its top. Self-test (MoE call with the metadata built on the side stream, eager and captured in a graph with the W4 GEMM1 PDL launch after the cross-stream join, T 1-256): bitwise equal to the one-stream call. Cross-stream tensors (top-k and every metadata tensor) get record_stream outside capture; no scratch buffer is shared between the streams (route_align allocates its outputs on the side stream, Marlin's workspace stays on main). 13/13 tests; prepare 97.6 s |
| 2026-10-28 | b57b66f36fb044c6 | ac08667 | + X5: at T <= 8 our FFN boundary reads the unpooler slice's 1024 live channels (no F.pad to 8192); branch RMS still over 8192; SGLang's own precomputed weights (post-norm and pre-norm `_frozen_weight` BF16, residual `_alpha`/`_beta` FP32, eps 1e-8 / 1e-5); one 1024-thread block per token | 1.425x (1.404-1.447) | per shape d1 1.493/1.435, d16 1.475/1.459, d32 1.177/1.371, v1 1.683/1.624, v16 1.447/1.298, v32 1.367/1.239. Only d1 and v1 use the change; our µs vs bd28a847: d1 117.8 -> 114.7 / 129.1 -> 127.3, v1 156.9 -> 160.9 / 168.2 -> 168.7, so -3.1/-1.8 on decode 1 and +4.0/+0.5 on verify 1: inconclusive. Profile: our boundary kernel 3.8 µs at T=1 and 4.4 µs at T=8 vs SGLang's `muse_spark_norm_residual_kernel` 3.9 / 3.7 µs in bd28a847; the removed pad launch is not among the profile's listed rows. The score drop vs bd28a847 comes from shapes this change does not touch (v32 sliding 1.430 -> 1.367, v16 full 1.336 -> 1.298): run-to-run noise. Self-test vs the padded SGLang call (T 1, 2, 3 with a zero row, 5, 8): next input 6.2e-4 to 8.7e-4, residual update 2.2e-3 to 2.4e-3 (SGLang's kernel appears to round inside the branch; ours stays FP32 between the BF16 input and the BF16/FP32 outputs). 13/13 tests; prepare 98.4 s |
| 2026-10-28 | b216cf4adb0d4033 | ad2994c | X5 boundary kernel as a cluster of 8 CTAs per token (1024 channels each, 256 threads; every CTA takes the branch RMS from the 1024 live channels; the stream's sum of squares is reduced over the cluster through distributed shared memory) | **1.449x** (1.427-1.471) | best. per shape d1 1.516/1.442, d16 1.483/1.476, d32 1.210/1.382, v1 1.729/1.645, v16 1.441/1.311, v32 1.414/1.286. Shapes on the change, our µs vs bd28a847 (X1 without X5): d1 117.8 -> 114.1 / 129.1 -> 126.1, v1 156.9 -> 157.3 / 168.2 -> 169.1, so X5 is -3.7/-3.0 µs on decode 1 and flat on verify 1 (inside noise). Profile: boundary kernel 3.2 µs at T=1 (one-block version 3.8; SGLang's 3.9-4.7), below the listed rows at T=8. Shapes off the change moved inside noise (v32 sliding 1.367 -> 1.414 with no code change there). CI overlaps bd28a847's (1.412-1.455). Self-test vs the padded SGLang call: next input 6.9e-4 to 8.3e-4, update 2.3e-3 to 2.4e-3. 13/13 tests; prepare 98.8 s |
| 2026-10-28 | 3b68cd76b4924f11 | 9157f47 | W0 item 1: X5 (cluster boundary kernel at T <= 8) in submission.py with SGLang's two BF16 roundings from norm_residual.cuh (the input-scale product f * w_post before the sum of squares, the normalized branch before the residual add); self-test bound tightened to 1e-4 | 1.431x (1.409-1.452) | per shape d1 1.491/1.427, d16 1.479/1.474, d32 1.190/1.380, v1 1.705/1.643, v16 1.391/1.291, v32 1.409/1.269. Self-test vs the padded SGLang call: next input, update and stream all 0.00e+00 at T 1, 2, 3, 5, 8, so the kernel is SGLang's math exactly (was 7e-4 / 2.3e-3 without the roundings): the X5 hold is lifted. Our us vs the submitted 0e28c26271c94570: d1 116.8 -> 114.1 / 128.3 -> 126.8, v1 155.3 -> 152.7 / 173.7 -> 165.7; vs b216cf4a (X5 unrounded) d1 114.1 -> 114.1 / 126.1 -> 126.8, v1 157.3 -> 152.7 / 169.1 -> 165.7, so the roundings cost nothing. The eight shapes off the change moved +1 to +21 us (v32 sliding 514.0 -> 534.9, v16 full 535.1 -> 544.5: MoE routing noise), which is what puts the geomean under b216cf4a's 1.449. Profile: boundary_kernel 3.6 us at T=1 and T=8 (SGLang's FFN boundary 4.4 / 4.1 in the same profiles). 13/13 tests; prepare 94.9 s |
| 2026-10-28 | 28e6c2fa656f413a | 77416dd | W0 item 2: own pre-expert norm kernel in place of torch's F.rms_norm (one 128-thread block per token, torch's vectorized_layer_norm summation order), launched with the PDL attribute and triggering at entry so GEMM1 is resident when it ends; also takes a strided FP32 input rounded to BF16 on load | 1.446x (1.424-1.468) | per shape d1 1.513/1.459, d16 1.480/1.479, d32 1.197/1.361, v1 1.720/1.629, v16 1.440/1.319, v32 1.433/1.276. Self-test vs the module at T 1-264, BF16 and strided FP32 input: 0 of 6,127,616 elements differ (bitwise torch); route-side graph-capture test with the norm as GEMM1's PDL primary: 0.00e+00 at every T. Our us vs 3b68cd76: d1 114.1 -> 113.1 / 126.8 -> 124.1, d16 175.5 -> 180.2 / 294.5 -> 290.6, d32 222.3 -> 218.5 / 447.5 -> 441.5, v1 152.7 -> 156.4 / 165.7 -> 170.7, v16 367.4 -> 367.2 / 544.5 -> 538.4, v32 534.9 -> 517.8 / 860.0 -> 848.7: down on 9 of 12, but d1 fell 1-3 us against the 5 expected and v1 rose 4-5. The W4 tuner picked different configs in the two runs (T=8 GEMM1 cfg with finalize<1> at 48 us in 3b68 vs finalize<8> at 35 us here), and the judge's routing differs per run, so the MoE-heavy shapes are noisy at the 5-15 us level. The norm's row is below the listed ones (torch's was 5.7-6.8 us on every shape). Kept: same math, one launch less exposed. 13/13 tests; prepare 96.2 s |
| 2026-10-28 | 179cecdac87b4872 | f825404 | W0 item 3a: PDL across our chain, projection and attention launches only. griddepcontrol.launch_dependents as the first instruction of proj_kernel, router_kernel, route_align, finalize_row, prep, gate, attn_kernel, attn_ws_kernel, both combine kernels and the FFN boundary; proj_fwd, the attention launches and the combines launched with cudaLaunchAttributeProgrammaticStreamSerialization (cudaLaunchKernelEx); proj_kernel streams its first three weight stages before griddepcontrol.wait and stages those stages' x rows with plain loads after it (cp.async groups unchanged); attn_kernel stages the page entries and the first K/V tiles before its wait; the ws kernel's loader warps never wait (inputs only), its consumers wait before the query prep. Glue launches (router, route_align, finalize_row, boundary) behind kPdlGlue = false | 1.463x (1.441-1.485) | per shape d1 1.496/1.468, d16 1.507/1.494, d32 1.246/1.396, v1 1.727/1.637, v16 1.487/1.352, v32 1.406/1.295. No gain in our own times: vs 28e6c2fa d1 113.1 -> 115.4 / 124.1 -> 123.6, v1 156.4 -> 156.7 / 170.7 -> 167.5, d16 180.2 -> 180.6 / 290.6 -> 297.9, d32 218.5 -> 219.0 / 441.5 -> 448.3, v16 367.2 -> 371.6 / 538.4 -> 534.8, v32 517.8 -> 527.3 / 848.7 -> 847.5; the geomean rose only because SGLang's times rose 1-5% this run. The profile shows the early launches happen (each proj row grew 0.8-1.6 us, the wait on the predecessor now inside the row) but the step did not shrink: within a graph the launch latency was already hidden, and what remains per boundary (about 0.7 us: the predecessor's completion and memory flush before griddepcontrol.wait returns) PDL cannot hide. Kept as the structure items 5 and 6 build on; not a win on its own. 13/13 tests; prepare 96.5 s |
| 2026-10-28 | e8b74a3519a94bc6 | 9525ba7 | W0 item 3b + 4: PDL on the glue launches too (router, route_align, finalize_row, FFN boundary: kPdlGlue = true), and a MoE-front tuner in prepare() that times the side-chain variants per token count (split router; our hi/lo mma router at 16-32; one cuBLAS GEMM for the pooler and both router halves with FP32 output, the pooled columns rounded to BF16 by the norm on load) as a graph with a cold L2 and keeps the fastest | 1.443x (1.421-1.465) | per shape d1 1.526/1.481, d16 1.462/1.480, d32 1.217/1.376, v1 1.720/1.645, v16 1.431/1.336, v32 1.376/1.251. The tuner fell back (its agreement test measured per-element BF16 ulps, which blow up on near-zero elements when the FP32 summation order differs: 3.0 ulp here, 25 and 626 in the two runs after), so every T stayed on the split router and this run measures the glue PDL alone. Our us vs 179cecda: d1 115.4 -> 112.8 / 123.6 -> 124.2, d16 180.6 -> 177.9 / 297.9 -> 296.6, d32 219.0 -> 215.9 / 448.3 -> 444.8, v1 156.7 -> 156.1 / 167.5 -> 166.8, v16 371.6 -> 366.1 / 534.8 -> 523.9, v32 527.3 -> 513.1 / 847.5 -> 843.6: equal or better on all 12, inside noise on each. Glue PDL kept. 13/13 tests; prepare 100.4 s |
| 2026-10-28 | 9cb961d383ab42be | 218e810 | W0 item 5: own attention boundary kernel (post-attention norm without weight, residual gate, pre-FFN norm, the FP32 router copy at T <= 8) with SGLang's one BF16 rounding of the normalized branch; a cluster of 8 CTAs per token with both sums of squares reduced through distributed shared memory, a PDL secondary that loads the stream, alpha, beta and the norm weight before its wait on the o projection and triggers at entry (so the pooler's weight stream starts during it) | 1.438x (1.416-1.460) | per shape d1 1.541/1.489, d16 1.475/1.472, d32 1.213/1.363, v1 1.718/1.636, v16 1.410/1.316, v32 1.359/1.246. Self-test vs FusedPostNormResidualPreNorm at T 1-9: next input, update, stream 0.00e+00 (router copy 4e-8); at T 16/33/256 next input 3e-6 to 1.5e-5 (SGLang's kernel changes its block shape there; the stream stays bitwise). Our us vs e8b74a35 (same code otherwise): d1 112.8 -> 111.0 / 124.2 -> 121.0, d16 177.9 -> 174.8 / 296.6 -> 296.8, d32 215.9 -> 216.6 / 444.8 -> 452.4, v1 156.1 -> 157.8 / 166.8 -> 177.6, v16 366.1 -> 365.7 / 523.9 -> 543.7, v32 513.1 -> 517.8 / 843.6 -> 841.1; decode 1 is the clean signal (-1.8 / -3.2 us; -4.4 / -2.6 vs 179cecda), the boundary row 3.9 us against SGLang's 4.4 (its wait on the o projection is inside the row now). The geomean fell because SGLang's own times fell back 2-7% on the verify shapes (v16 sliding 523.8 -> 515.6, v32 sliding 706.0 -> 703.6, v32 full 1055.6 -> 1048.3). Kept. 13/13 tests; prepare 81.1 s |
| 2026-10-28 | 985732949c07474f | 5f0dc0f | W0 item 6 (first form): at T <= 8 a kernel on the side stream, forked right after the qkv projection, issues cp.async.bulk.prefetch.L2 with an evict_last policy over the o weights (packed at T = 1, cuBLAS's above), the FP32 router and the unpooler slice's 8.4 MB of packed tiles while attention runs | 1.446x (1.424-1.468) | per shape d1 1.534/1.456, d16 1.477/1.481, d32 1.173/1.478, v1 1.731/1.638, v16 1.436/1.303, v32 1.369/1.262. The o GEMV row did not fall: 7.7 -> 7.3 us at decode 1 sliding (the bar was about 3.5), unpooler 6.0 -> 6.0, router 5.2 -> 4.9; the prefetch kernel itself shows 5.4-5.8 us. Our us vs 9cb961d3: d1 111.0 -> 111.2 / 121.0 -> 123.8, v1 157.8 -> 149.8 / 177.6 -> 169.7 (verify 1 moved 8 us with no row moving: the W4 configs and routing differ per run), the eight shapes off the change +/- 5-30 us of MoE noise. The bulk prefetch hint does not fill L2 in time or at all; replaced by real 16-byte loads with the evict_last cache hint in the next run. 13/13 tests; prepare 99.0 s |
| 2026-10-28 | 34568f495b024c2f | 40ecc07 | W0 items 4 and 6 again: the MoE-front tuner's agreement test on a row-relative bound (1e-3 relative L2, routes >= 99%), and the L2 prefetch as real 16-byte loads with the evict_last cache hint, eight in flight per thread | 1.447x (1.425-1.469) | per shape d1 1.504/1.409, d16 1.491/1.490, d32 1.212/1.372, v1 1.740/1.592, v16 1.442/1.327, v32 1.411/1.285. The tuner ran: T=16 concat 39.8 us, split 41.0, mma 41.5; T=32 concat 40.1, mma 42.5, split 45.2; T=128 concat 41.6, split 51.5; T=256 concat 48.2, split 62.4 (the front from the FFN input to the join, as a graph with a cold L2), so the one-GEMM pooler + router variant runs at every T above 8; it agrees with the split path to 8e-5 (T=16) or 0 and routes 100% equal. Our us vs 98573294: d1 111.2 -> 114.7 / 123.8 -> 127.9, d16 175.4 -> 177.4 / 293.3 -> 294.2, d32 222.1 -> 211.0 / 422.9 -> 444.1 (the 422.9 was the day's outlier; 441-452 in every other run), v1 149.8 -> 153.1 / 169.7 -> 171.6, v16 371.7 -> 372.5 / 546.1 -> 538.7, v32 539.9 -> 531.2 / 868.5 -> 860.9. At T <= 8 only the prefetch changed and all four shapes rose 2-4 us: the o GEMV row fell 7.3 -> 5.9 us (the bar was 3.5), the unpooler 6.0 -> 5.6, the router did not move, and the prefetch kernel itself streams 34 MB for 9.5 us beside the latency-bound attention (14.0 -> 14.8). Prefetch dropped (PORT_L2_PREFETCH = False). Above 8 tokens the concat variant's gain sits inside the MoE noise at the step level (d32 sliding the day's best, v32 sliding and full mid-pack); the tuner's A/B on identical inputs is the evidence, kept. 13/13 tests; prepare 98.9 s |
| 2026-10-28 | f23f1c8171374285 (submit) | 4721074 | three ports behind their own switches: fused MoE (muse_moe_w4a16 at 8a8f6e4, variants sk8w4s / sk16w4s / main-d2, one launch with device-side routing, no PDL attribute), skinny projections (muse_bf16_proj at dfbf010, 26 configs, 12-bit weights with escape warps, cgemv / sgemv at m = 1, the fused pooler + router as the skf front variant), attention one-launch compiled but off (PORT_ATTN_ONE = False pending its standalone verdict) | **1.643x** (1.618-1.667) | best. per shape d1 1.876/1.738, d16 1.751/1.650, d32 1.323/1.446, v1 2.056/1.925, v16 1.471/1.342, v32 1.670/1.425. Our us vs 34568f495b024c2f: d1 114.7 -> 91.8 / 127.9 -> 105.1, d16 177.4 -> 148.2 / 294.2 -> 264.4, d32 211.1 -> 203.8 / 444.1 -> 433.0, v1 153.1 -> 127.5 / 171.6 -> 141.8, v16 372.5 -> 361.6 / 538.7 -> 540.4, v32 531.2 -> 427.8 / 860.9 -> 757.6; vs the board entry 0e28c26271c94570 (1.403x): d1 116.8 -> 91.8 / 128.3 -> 105.1, d16 177.0 -> 148.2 / 293.4 -> 264.4, d32 217.5 -> 203.8 / 448.0 -> 433.0, v1 155.3 -> 127.5 / 173.7 -> 141.8, v16 360.0 -> 361.6 / 535.1 -> 540.4, v32 514.0 -> 427.8 / 854.4 -> 757.6. Decode 1, decode 16 and verify 1 fall 23-30 us each (fused MoE and the 12-bit projections), decode 32 7-11, verify 32 103 (main-d2 at 256 tokens, 275 us against the W4 chain's GEMMs); verify 16 inside noise, where the policy keeps the W4 chain. MOE1POLICY {1: sk8w4s, 8: sk8w4s, 16: sk8w4s, 32: sk8w4s, 128: w4, 256: main-d2} (MOE1BENCH T=1 sk8w4s 41.7 vs w4 49.9, T=8 73.8 vs 78.4, T=16 107.5 vs 116.1, T=32 150.4 vs 153.2, T=128 w4 247.2 vs main-d2 255.4, T=256 main-d2 288.7 vs w4 356.5 us). SKP tune: cgemv at m = 1 for qkv_gate (14.9 vs proj_fwd 19.8), o (10.0 vs 12.6) and the unpooler slice (8.4 vs 11.1), pool8 for the pooler (22.1 vs 28.4), the compressed sk config at every other (layer, m): qkv 16.1/16.4/17.9 us at m 8/16/32 against 20.7/21.1/21.6, pooler 22.2/22.8/24.7 vs 29.7/30.2/30.8, o 10.9/10.8/10.8 vs 11.9/12.0/12.1, unpooler 9.0/22.0/23.2 vs 11.9/27.4/27.6; raw BF16 lost to compressed everywhere. TUNE front: T=1 fp32 28.0 vs skf 29.8 -> fp32; T=8 skf 30.0 vs fp32 37.7 -> skf; T=16 skf 30.4 vs mma 35.4, concat 37.2, split 40.2 -> skf; T=32/128/256 concat; every variant routes 100% equal to its reference. Self-tests: MOE1TEST all three variants 3.89e-3 vs FP32 (within 7.8e-4 of the W4 chain), both 2,000-replay stresses clean (generations 2003, no spin flags, scratch invariant held); SKP 30 candidates pass (GEMVs 0 error, worst 2.5e-4), 0 inexact escape deltas, both fused configs ok. Profile decode 1 sliding: fused MoE 25.0 us (the W4 chain was GEMM1 21.3 + GEMM2 10.8 + swiglu + finalize 4.2), pooler sk 16.0, qkv cgemv 10.5, o cgemv 5.6, unpooler cgemv 4.0, attention 14.4 (the fused kernel, since the one-launch port is off). 13/13 tests; prepare 102.5 s (fused MoE build 47.5 s and sk build 15.1 s on their threads, both finished before they were needed; moe fused 1.8 s and proj sk 0.4 s on the main thread); the whole run 216 s |
| 2026-10-28 | cc7a07e2964a4d67 (submit) | fc1edbe | one-launch attention on (PORT_ATTN_ONE = True) on every tp8 shape: _ONE_SRC re-inlined by tools/dl_one_inline.py from attn_one.cu at 83e0112 (carries the 4352d4e fix: oct_sum's full-mask shuffle moved out of the K-only branch of the new-token pass), q_len 8 instance 51 swapped for 19 and the plan sends full verify at batch 16 / 32 to 19 as the standalone planner does (planners compared on 240 shape / width combinations: identical), host sets the new Params::phase_limit to 0, dbg buffer 4 -> 32 ints for the kernel's phase stamps | **1.761x** (1.734-1.787) | best. per shape d1 2.054/1.956, d16 1.816/1.934, d32 1.391/1.736, v1 2.162/2.092, v16 1.404/1.469, v32 1.704/1.680. Our us vs f23f1c8171374285: d1 91.8 -> 84.0 / 105.1 -> 93.3, d16 148.2 -> 142.4 / 264.4 -> 223.6, d32 203.8 -> 185.0 / 433.0 -> 349.3, v1 127.5 -> 124.7 / 141.8 -> 131.2, v16 361.6 -> 343.2 / 540.4 -> 457.1, v32 427.8 -> 423.3 / 757.6 -> 634.0: down on all 12; the full layers at batch 16 / 32 move most (-41 / -84 decode, -83 / -124 verify). v16 sliding's speedup fell (1.471 -> 1.404) while our time fell 18.4 us: SGLang's time on it moved between runs. Attention rows in the profiles (ours, main kernel only; the old combine rows include the wait on their kernel), old -> new: d1 sliding fused 14.4 -> attn_one<1,13> 6.1, d1 full FA3 19.1 + combine 7.1 -> <1,11> 15.2, d16 sliding 16.4 -> <1,1> 10.8, d16 full 127.8 -> <1,49> 93.7, d32 sliding 22.3 -> <1,1> 12.3, d32 full 247.5 -> <1,49> 181.9, v1 sliding ws 12.2 -> <8,15> 7.4, v1 full FA3 18.8 + combine 7.5 + prepare 3.2 -> <8,11> 18.0, v16 sliding ws 18.1 -> <8,3> 13.1, v16 full ws 187.7 -> <8,19> 125.6, v32 sliding ws 23.5 -> <8,3> 17.6, v32 full ws 356.8 -> <8,19> 236.2 (mode 19 matches the standalone's 127 / 236). Self-test: cooperative_launch=True pdl=False; residency L1 modes 1/3/11/13 396, 49 264, L8 all 264; cases worst output 4.84e-02 vs SGLang's path, new cache rows 2.76e-02, workspace invariant held after every case; stresses (2,000 replays each) plan (32,1,13), (264,1,49), (33,1,15), (264,1,19) replayed vs eager 0 to 1.8e-08, vs SGLang 3.4e-03 to 2.5e-02, max_polls 0 / 0 / 3 / 31, workspace zero after each, both speculated-length fallbacks clean. 13/13 tests; prepare 103.2 s (one-launch build 45.2 s on its thread) |
| 2026-10-28 (before the 00:00 UTC reset) | a41ace0b94ba4369 (submit by the coordinator, snapshot of 1099d9b) | 1099d9b | MoE ordering repair in the layer: the fused MoE re-inlined by tools/dl_moe_inline.py from muse_moe_w4a16 at 1099d9b (fence.proxy.async.global by every x-tile writer before its release and by the issuing thread after its acquire, the gather's xready published by red.release after its barrier; docs/runs-moe.md "Proxy ordering of the x tiles"), fat mode 0 only (LAYER_MODES), host zeroes Params::cost (the compiled stage-cost table). The re-inline also brings the standalone's changes since 8a8f6e4 (topk_ids read as int4 by every thread, weights producer started before the x gather, stage costs read from S.cost) | **1.782x** (1.756-1.811) | best (board entry). per shape d1 2.055/1.948, d16 1.809/1.922, d32 1.414/1.830, v1 2.021/2.002, v16 1.536/1.544, v32 1.771/1.719. Our us: d1 82.99/93.33, d16 149.14/231.59, d32 193.73/347.14, v1 132.24/135.07, v16 362.84/474.33, v32 431.84/638.15. Against cc7a07e2964a4d67 (1.761x) the CIs overlap, so the gain is not separable from noise; the point of the run is the repaired ordering in the board entry. 13/13 tests; prepare 100.0 s. Profiles and result JSON in muse_decode_layer/profiles/submission.a41ace0b94ba4369.* |
| 2026-10-29 | e87fcb1ed5c64a62 (check) | bdb4c7a | MoE re-inlined with fat modes 0 and 16 (mode 16: two mma.sync chains per n8 tile and unit-aligned phase B boundaries, which took 128 and 256 tokens standalone); every fat variant stress-tested at 128 tokens | 1.785x (1.759-1.812) | per shape d1 2.067/1.966, d16 1.824/1.949, d32 1.400/1.763, v1 2.162/2.089, v16 1.474/1.556, v32 1.720/1.714. MoE kernel in the profiles against a41ace0b94ba4369: v32 mode 16 279.0 / 266.0 us vs mode 0 276.2 / 292.9 (mean -12 us); v16 stays on the W4 chain (w4_gemm 173.9 + 83.1 us, as before); v1 ran sk8w4s at 52.9 / 52.7 us where a41ace0b ran sk16w4s at 82.6 / 92.1, a tuner pick that differs between runs (v1 119.9 / 135.3 us vs 132.2 / 135.1). The MoE prepare lines (MOE1BENCH, MOE1POLICY) are not in the judge's log, so the policy is read from the profiles. Score inside the noise of the 1.782x entry; not submitted |
| 2026-10-29 | 69ddb896965d4279 (check) | 075525d | attention mode 51 ported: _ONE_SRC re-inlined by tools/dl_one_inline.py from attn_one.cu at e7521d9 (the only device change against the layer's previous copy is the producer arm's `return` replaced by PTX `exit`; every fence, barrier and wait count is unchanged, the MoE copy re-inlines byte-identical from muse_moe_w4a16), instance 51 built and preferred for full verify at batch > 1 (19 the fallback); a verify 32 full stress added; the SUMMARY block at the end of prepare | 1.795x (1.768-1.822) | per shape d1 2.084/1.948, d16 1.815/1.928, d32 1.391/1.752, v1 2.206/2.090, v16 1.488/1.540, v32 1.776/1.737. Attention kernel row on the shapes 51 changes, against e87fcb1ed5c64a62: v16 full <8,19> 124.9 -> <8,51> 120.3 us (-3.7%), v32 full 238.2 -> 228.1 (-4.2%); every other attention row within 0.8 us. Step (ours, us) vs e87fcb1: v16 full 464.0 -> 464.5, v32 full 628.2 -> 617.0; the MoE rows move 10-20 us between runs on the same code and hide the attention gain on v16. SUMMARY printed and kept in the tail: MoE picks T 1-32 sk8w4s, 128 w4 (247-249 vs mode 16 at 250.7), 256 main-m2-ar2-ub (mode 16, 282.3 vs main-d2 290.7, w4 357.0); attention plans as the port guide's table with (264, 1, 51) on v16 / v32 full; failed 0. Stresses: 16x8 and 32x8 full on 51, 2,000 replays each, replayed vs eager 3.7e-05 / 2.9e-05, max_polls 28 / 49, workspace zero. 13/13 tests; prepare 85.8 s |
| 2026-10-29 | 6abe9acb0d984d82 (check) | 6819a5f | repeat of 69ddb896 (comment-only change so the judge does not return the cached result; SUMMARY source digest identical, ade14376fd244cf0) | 1.788x (1.761-1.814) | per shape d1 2.079/1.946, d16 1.825/1.930, d32 1.374/1.739, v1 2.206/2.107, v16 1.461/1.509, v32 1.765/1.745. Attention row v16 full 122.1 us, v32 full 227.4 (vs 124.9 / 238.2 on 19 in e87fcb1: -2.2% / -4.5%). Step vs e87fcb1: v16 full 464.0 -> 451.5, v32 full 628.2 -> 612.9. Both runs and both controls (1.782x, 1.785x) overlap, so the score gain is not separable from noise; the component gain repeats (51 under 19 by 2-4.5% in four paired rows) and matches the standalone 2.5-2.7%. The MoE tuner's W4 configs differed between the two runs (T=1 g1 27 vs 17, T=128 g2 13 vs 29), which the SUMMARY now shows. Kept |
| 2026-10-29 | 0cb15c11accf4190 (check, MoE agent) | bdb4c7a + probe (snapshot outside the repo; probe in muse_moe_w4a16/profiles/layer-flip/) | the layer unchanged plus a prepare-time probe of the 128-token MoE ranking: synthetic routing with and without the L2 flush, the W4 chain with route_align outside and inside the graph, and the layer's front plus experts on an isotropic FFN input | 1.773x (1.746-1.799) | measurement only; per seed mode 16 minus W4 (metadata outside) +1.6 +2.3 -35.8 -12.6 -9.4 +2.4 us on synthetic seeds 11-16 (the bench uses 11 and 12). docs/runs-moe.md, "Why the layer keeps the W4 chain at 128 tokens" |
| 2026-10-29 | 7d1205a6959c41a9 (check, MoE agent) | bdb4c7a + probe 2 | experts alone on the isotropic-input routing and on synthetic routing, and the front plus experts, paired per seed | 1.798x (1.771-1.825) | measurement only; isotropic routing (13-22 routes per expert at most) favours the W4 chain by 8-18 us at 128, synthetic repeats probe 1 within 0.5-4.2 us per seed; route_align is 2.0-3.1 us |
| 2026-10-29 | 9d046da0207b40b2 (check, MoE agent) | bdb4c7a + probe 3 | experts alone on judge-like routing (reference.py's residual stream through the layer's attention boundary and the judge's router, local_start on the busiest 32-expert range) | 1.789x (1.762-1.815) | measurement only; mode 16 minus W4 per seed -31.6 -27.6 -38.6 -31.8 -22.7 -25.0 us at 128 tokens and -82 to -107 at 256. Inference: the layer's bench keeps the W4 chain at verify 16 because its two synthetic seeds favour it by 1-3 us; on judge-like routing mode 16 is about 30 us faster per verify-16 shape |
| 2026-10-29 | b9bd3a382ba9449b (check) | 1b5bb52 | MoE selection on judge-like routing: `_moe1_tune` benches every candidate on `_moe_judge_routing` (reference.py's residual stream with its STREAM_SEED basis and the verify draft-row share, a random attention branch through the layer's attention boundary, the layer's router and SGLang's routing, local_start on the 32-expert range with the most pairs, as case_fields sets it), 4 seeds per bucket, q_len 1 for the decode buckets (1, 16, 32) and 8 for the verify ones (8, 128, 256), in prepare only (the policy is the only thing kept); fat mode 20 (owned phase A, muse_moe_w4a16 308ed5b) inlined by tools/dl_moe_inline.py with LAYER_MODES (0, 16, 20). Against the standalone assembly the layer copy differs only in the documented local_start patch (15 lines); every fence and barrier count matches (fence_proxy_async_global 5, red_release_add 3, named_sync 11, __threadfence 7, mbar_wait 7, spin_ge 3, __syncthreads 16, ld_acquire 13) | 1.814x (1.787-1.841) | per shape d1 2.053/1.950, d16 1.797/1.920, d32 1.393/1.755, v1 2.173/2.093, v16 1.645/1.628, v32 1.770/1.757. SUMMARY: T 1-32 sk8w4s (as before), T=128 main-m2-ar2-ub (mode 16) 218.5 vs main-d2 220.5, main-own 221.1, w4 251.3; T=256 main-own (mode 20) 280.8 vs mode 16 283.3, main-d2 290.4, w4 372.5 us; busiest local expert 59-68 routes at 128, 120-139 at 256; failed 0 (mode 20 passed the FP32 self-test and the 2,000-replay stress, or it would not be benched). Step (ours, us) vs 69ddb896 / 6abe9acb: v16 sliding 320.7 (-36.7 / -21.2), v16 full 419.4 (-45.2 / -32.1), v32 sliding 392.3 (-14.5 / -8.9), v32 full 609.8 (-7.2 / -3.1). MoE row v16: mode 16 234.1 / 204.0 us against the W4 chain's four kernels 274 / 239 (6abe) and 267 / 257 (69ddb). 13/13 tests; prepare 102.4 s |
| 2026-10-29 | 2fed900548ec4d5b (check) | 9307b9e | repeat of b9bd3a38 (comment-only change; SUMMARY source digest identical, 99b6bb3ca7941b0e) | 1.838x (1.811-1.866) | per shape d1 2.066/1.981, d16 1.824/1.922, d32 1.376/1.841, v1 2.199/2.115, v16 1.652/1.656, v32 1.842/1.790. SUMMARY: same picks (T=128 mode 16 230.4 vs mode 20 230.8, w4 257.3; T=256 mode 20 287.8 vs mode 16 291.4, w4 370.8). Step vs 69ddb896 / 6abe9acb: v16 sliding 327.7 (-29.7 / -14.2), v16 full 436.2 (-28.3 / -15.3), v32 sliding 413.2 (+6.5 / +12.0), v32 full 622.6 (+5.6 / +9.7). Over the four pairings verify 16 falls 25.5 us sliding and 30.2 full on average, matching probe 3's inferred 30 us; the verify-16 speedups rose from 1.461-1.488 / 1.509-1.540 to 1.645-1.652 / 1.628-1.656 in both runs (the ratio is within-run, so the judge's per-run router seed cancels). Verify 32 (mode 20 in place of mode 16) moves -7 to +12 us, inside the MoE's run-to-run spread; the 2-4 us standalone gain is not visible at the layer level. Score: both runs above both controls (1.795x, 1.788x), CIs overlap. Submit candidate |
| 2026-10-29 | 344ae513913e4525 (submit, authorized by the user) | aab7d75 | file of 2fed900548ec4d5b: judge-like MoE bench routing, modes 16/20 inlined, attention mode 51 | 1.833x | passed. SUMMARY: MoE 128 main-m2-ar2-ub 207.0, 256 main-own 260.4; failed 0. New board entry over a41ace0b94ba4369 (1.782x) |
| 2026-10-29 | a2310eddf6f248eb (check) | 8592ed8 | two changes bundled. (1) MoE fat mode 23 (phase B units claimed at run time through ctr[10] and S.bq, muse_moe_w4a16 beb1924) added to `LAYER_MODES` (0, 16, 20, 23) and re-inlined; the inliner output is byte-identical on a second run. Against the standalone assembly of the same variants the layer copy differs only in the 18 lines of the local_start patch; SmemMain is textually identical (228,736 B, as run standalone); counts match the standalone assembly (fence_proxy_async_global 6, red_release_add 3, named_sync 11, __threadfence 10, mbar_wait 10, spin_ge 4, __syncthreads 16, ld_acquire 16, st_release_cta 2, bq_get 3). ctr is int32[64], so index 10 is in range; the kernel zeroes ctr[10] from CTA 0 after the grid barrier, every claim precedes that CTA's barrier arrival, and the layer launches the MoE on the main stream after joining the side stream with no PDL attribute, so the next launch (next graph replay, or the next layer's call on the shared scratch) starts after the reset. The scratch check now includes ctr[10]; the 2,000-replay stress runs every fat mode with 128 and 256 tokens captured back to back in one graph (two launches per replay on the shared scratch); a fat mode that fails it is dropped from the bench alone. (2) pool32 at four units per stage (muse_bf16_proj f0374c6, `tools/dl_proj_inline.py`; the regenerated `bucket_of` would reference WD_BUCKETS, which the layer lacks, so the layer's own `bucket_of` is kept) | 1.851x (1.823-1.878) | per shape d1 2.048/1.935, d16 1.835/1.922, d32 1.400/1.759, v1 2.179/2.113, v16 1.735/1.708, v32 1.874/1.830. SUMMARY: moe stress ok for all five variants; T 1-16 sk8w4s; T=32 main-own-dynb 107.3 vs sk8w4s 112.8; T=128 main-own-dynb 196.2 vs mode 16 217.1, mode 20 219.1, w4 241.9; T=256 main-own-dynb 267.0 vs mode 20 286.6, mode 16 288.3, w4 380.0; failed 0. MoE kernel row: v16 183.5/192.7 us, v32 243.7/265.7. pool32 row d32 16.3/24.1 us (control 18.4/18.7). 13/13 tests; prepare 87.4 s |
| 2026-10-29 | ff254f51bb5f4bde (check) | 9acb4ed | repeat of a2310edd (comment-only change) | 1.852x (1.824-1.880) | per shape d1 2.074/1.980, d16 1.816/1.951, d32 1.424/1.775, v1 2.138/2.101, v16 1.714/1.720, v32 1.866/1.822. SUMMARY: stress ok for all five; T=32 sk8w4s 107.7 vs main-own-dynb 112.6 (the T=32 pick flips between the two from run to run, within 5 us); T=128 main-own-dynb 194.7 vs 215.2 (mode 16); T=256 main-own-dynb 257.5 vs 276.1 (mode 20); failed 0. MoE row v16 194.4/187.3, v32 245.5/243.9. pool32 row d32 16.2/16.1. prepare 105.0 s |
| 2026-10-29 | e9512628e4e64adb (check) | 9acb4ed, muse_decode_layer/experiments/pool32_old | attribution: the bundle with pool32 back at (2, 32, 2, 2, 2, 2, 5), so mode 23 alone against the controls | 1.843x (1.815-1.870) | per shape d1 2.036/1.949, d16 1.809/1.962, d32 1.365/1.728, v1 2.187/2.091, v16 1.741/1.731, v32 1.866/1.808. T=32 main-own-dynb 102.8 vs sk8w4s 107.3, T=128 and 256 main-own-dynb. pool32 row d32 18.7/26.0 us. Summary of the three runs against 344ae513 (1.833x; v16 1.648/1.659, v32 1.772/1.735) and the checks b9bd3a38 / 2fed9005: mode 23 raises the verify-16 speedup to 1.714-1.741 sliding and 1.708-1.731 full and verify 32 to 1.866-1.874 sliding and 1.808-1.830 full in all three runs, against 1.645-1.652 / 1.628-1.659 and 1.770-1.842 / 1.735-1.790 in the three controls; at the bench, 23 is 17-21 us under the next variant at 128 tokens and 17-20 us at 256. pool32 alone: the kernel row is 16.1-16.3 us in three of four samples against 17.8-18.8 in seven of eight old-geometry samples (24.1 and 26.0, both full-layer d32, are one outlier on each side); paired against this run, d32 is 1.400/1.759 with the new geometry and 1.365/1.728 without (both on mode 23 at T=32), a 2 us kernel gain at a step of 180 and 350 us, inside the run-to-run spread of d32 (1.365-1.424 / 1.728-1.841 over these three runs and the three controls). Kept both. Submit candidate: the file of ff254f51 |
| 2026-10-29 | dcf9fefab76f4c27 (submit, under the standing replace-when-demonstrably-better authorization) | 975d59a | file of ff254f51bb5f4bde: MoE mode 23 inlined, pool32 4 units per stage | 1.849x | passed. Board entry before: 344ae513913e4525 (1.833x) |
| 2026-10-29 | 9f3b71601eeb4ba4 (check) | d60c478 | attn_one.cu from 3335c00 re-inlined by tools/dl_one_inline.py (the once-path diet of muse_attention_decode's 03a21fbc99cf445f: no diagnostic hooks, page shift instead of division, straight-line query passes, predicated prologue copies; the q_len 8 producer builds keep the plain path; ONE_DIAG 0, ONE_FIN8 0). The host wrapper now sets `Params::lp` and refuses a page size that is not a power of two; `_one_wins` sends such sizes to the fallback paths. The layer copy differs from the standalone source only by the nested `one` namespace, the dropped INSTANTIATE notes and host functions, and the two `griddepcontrol.wait` insertions; the inliner output is byte-identical on a second run. SASS of all 11 layer instances against the previous layer copy: BAR.SYNC, FENCE.VIEW.ASYNC, MEMBAR, SYNCS, ATOMG, REDG and LDGDEPBAR counts identical; only DEPBAR drops (by the `wait_group 0` of the removed phase exits) and one EXIT goes in most instances; no spills (51 still 80 registers). The layer runs no phase exits (it passes phase_limit 0), so no bench needs -DONE_DIAG=1 | 1.868x (1.840-1.896) | per shape d1 2.055/1.975, d16 1.816/1.981, d32 1.430/1.799, v1 2.139/2.096, v16 1.775/1.758, v32 1.903/1.826. Attention self-test 22 cases passed with a clean workspace after each; the five 2,000-replay stresses (and both speculated-length fallbacks) passed, workspace zero. SUMMARY printed, failed 0, attention plans unchanged. 13/13 tests |
| 2026-10-29 | ec9ba59d82184a1e (check) | 08c9fe5 | repeat of 9f3b7160 (comment-only change) | 1.859x (1.831-1.887) | per shape d1 2.079/1.957, d16 1.851/1.989, d32 1.411/1.816, v1 2.174/2.102, v16 1.721/1.724, v32 1.845/1.796. Stresses passed, SUMMARY failed 0, 13/13 tests; prepare 105.7 s. Attention kernel row (us, one profiled sample per shape) in the two new runs against ff254f51 / a2310edd: d1 sliding 5.4, 5.5 vs 6.5, 6.1; d1 full 14.1, 15.1 vs 15.3, 15.9; d16 sliding 9.0, 9.2 vs 10.8, 10.0; d16 full 91.1, 88.4 vs 94.3, 93.7; d32 sliding 11.5, 11.5 vs 12.9, 12.2; d32 full 170.9, 174.1 vs 180.8, 175.9; v1 sliding 6.8, 7.0 vs 7.4, 7.3; v1 full 17.3, 17.5 vs 18.4, 18.0; v16 sliding 11.7, 12.2 vs 13.1, 13.0; v16 full 119.0, 117.2 vs 121.1, 119.7; v32 sliding 16.2, 16.3 vs 17.7, 17.6; v32 full 224.5, 224.6 vs 225.7, 227.7. Every row is lower in both new runs than in both controls; the sliding rows fall 0.4-1.8 us, as the standalone diet did (0.6-1.6). The headline (1.868x, 1.859x against 1.851x, 1.852x) moves less than the CIs, as expected from a 1-10 us change at steps of 180-620 us. Submit candidate: the file of ec9ba59d (08c9fe5) |
| 2026-10-29 | 5b95c24e318d46fa (submit, under the standing replace-when-demonstrably-better authorization) | 1f0012c | file of ec9ba59d82184a1e (08c9fe5): attention once-path diet inlined | 1.878x | passed. Board entry before: dcf9fefab76f4c27 (1.849x) |
| 2026-10-29 | 725d34e9511c4370 (check) | d192130 | attn_one.cu from ccd0757 re-inlined by tools/dl_one_inline.py: `ONE_FIN8` 1 (mode 11, the batch-1 full plan with eight workspace copies, loads all eight copies before zeroing any; standalone checks b0b146dfbb4841b2 2.347x and fe08878a7e0c46f4 2.316x, ordering argument in docs/runs-attention.md, 10-29), `ONE_KVFAST` 0. Inliner reproducible: 3335c00's source re-inlined gives the committed 08c9fe5 layer byte for byte, and the new inline is byte-identical on a second run. Layer diff against 08c9fe5: the `ONE_FIN8` default and its comment, and the `ONE_KVFAST` define with its in-loop copy block (compiled out at 0); nothing else in attn_one.cu changed since 3335c00. `Params::lp`, the host wrapper's page-shift setup and power-of-two refusal, and `_one_wins` are untouched. SASS of the layer's 11 instances (its one extension built with torch's nvcc flags, CUDA 12.8) against 08c9fe5: BAR.SYNC, FENCE.VIEW.ASYNC, MEMBAR, ATOMG, REDG, CCTL, SYNCS, DEPBAR and EXIT identical per instance; only <1,11> and <8,11> change, LDG 25 to 49 and STG 25 to 49 (<1,11>) / 28 to 52 (<8,11>), the unrolled eight-copy block. Registers <1,11> 108 to 115 (cap 128 at 128 threads, min 4 blocks; residency still 396, shared memory bound), <8,11> 109 to 108, no spills anywhere | 1.894x (1.865-1.922) | per shape d1 2.086/2.013, d16 1.846/2.003, d32 1.393/1.878, v1 2.186/2.176, v16 1.787/1.793, v32 1.937/1.854. Attention self-test 22 cases passed (cases 17 and 18 run mode 11 at q_len 1 and 8) with a clean workspace after each; the five 2,000-replay stresses and both speculated-length fallbacks passed, workspace zero. Plans unchanged (d1 / v1 full (252, 8, 11)). SUMMARY failed 0, 13/13 tests |
| 2026-10-29 | 80b98f567aa74dd0 (check) | c306013 | repeat of 725d34e9 with two more replay stresses in prepare (decode and verify 1 full); no timed code changes | 1.873x (1.845-1.901) | per shape d1 2.062/2.002, d16 1.837/1.957, d32 1.395/1.804, v1 2.185/2.138, v16 1.788/1.764, v32 1.890/1.826. All seven stresses passed, workspace zero; the new decode 1 full stress ran mode 11 (plan (44, 8, 11), 2,000 replays plus 50 on the speculated-length fallback), so it exercises the FIN8 finalize; the verify 1 full stress's shorter context planned (62, 8, 15), a one-tile mode FIN8 does not compile into, so the q_len 8 FIN8 build is covered only by self-test case 18 and the scored runs. SUMMARY failed 0, 13/13 tests; prepare 104.8 s. Attention kernel rows (us, one profiled sample per shape), the two new runs against 9f3b7160 / ec9ba59d: d1 full (<1,11>) 13.4, 12.9 vs 14.1, 15.1 (means -1.45); v1 full (<8,11>) 15.9, 15.7 vs 17.3, 17.5 (means -1.60). Standalone judged -1.44 / -1.68. Both new rows are below both controls on both changed shapes. Rows whose code is unchanged moved -11.0 to +6.6 us (d32 full 163.1, 172.1 vs 170.9, 174.1; v32 full 222.8, 231.1 vs 224.5, 224.6), so the headline (1.894x, 1.873x against 1.868x, 1.859x; board 1.878x for the 08c9fe5 file) moves inside the CIs, as expected from a 3 us change in a 12,000 us score. Layer per-shape totals: d1 full 90.45, 90.63 vs 91.71, 93.23; v1 full 125.01, 128.54 vs 131.12, 132.15. Submit candidate: c306013 (same timed code as d192130, plus the mode 11 stress) |
| 2026-10-29 | 22d5710c4e8540d6 (submit, under the standing replace-when-demonstrably-better authorization) | b3b561a | file of 80b98f567aa74dd0 (c306013): attention ONE_FIN8 inlined | 1.878x | passed. Board entry before: 5b95c24e318d46fa (1.878x) |
| 2026-10-29 | 5285e58d978f42c6 (check) | 3eec2bb | Gap-map E3, the attention boundary above 32 tokens. Five more launch shapes of our kernel (`attn_boundary_geo_kernel<CL, NT>`: 2-CTA x 512 and 4-CTA x 256 clusters, one CTA per token at 256, 512 and 1024 threads; same math, the one BF16 rounding, the stream loaded before the PDL wait; the 8-CTA kernel and its fences and cluster barriers unchanged) and `_abnd_tune` in prepare: per token count (1, 8, 16, 32, 128, 256) it times the o projection plus each boundary, SGLang's module included, as graphs after an L2 flush, six rounds with rotated order, median of the round medians; a challenger needs 0.3 us over the 8-CTA kernel. Every geometry passes the existing self-test (bound 1e-4; worst 9.8e-6 to 2.9e-5 against 4.2e-5 for the 8-CTA one) and an agreement check on the real o output. Occupancy query in prepare: the 8-CTA kernel fits 124 resident clusters, so 128 tokens (128 clusters) need two waves and 256 need three | 1.891x (1.863-1.919) | per shape d1 2.083/2.021, d16 1.847/1.980, d32 1.446/1.830, v1 2.185/2.164, v16 1.791/1.756, v32 1.915/1.840. Picks: T=1 and 8 the 8-CTA kernel, T=16 SGLang's module, T=32 one CTA x 1024, T=128 SGLang's module, T=256 one CTA x 512. Chain times (us) 8-CTA vs pick: T=128 22.88 vs 17.89, T=256 29.63 vs 22.53; at T <= 32 the best three candidates sit within 1.1 us of each other. Boundary rows: v16 10.0-10.6 in the controls -> SGLang's kernel, rows 5.6 / 4.6 (sliding) and 4.7 / 4.4 (full), the FFN boundary being the other; v32 15.3-16.3 -> 7.5 / 7.7; d32 4.5-4.6 -> 3.8 / 3.7. Our us against 725d34e9 / 80b98f56 / 22d5710c: v16 290.4 vs 311.3 / 303.9 / 302.4 and 397.0 vs 408.9 / 406.7 / 407.9; v32 361.5 vs 385.9 / 385.6 / 372.8 and 582.6 vs 588.5 / 596.7 / 586.8. The step fell more than the boundary row on v16 / v32 sliding (-15.4, -19.9 against the control mean); the rest is the MoE row's run-to-run spread. d32 full rose 3.5 us against the mean with its boundary row 0.8 us lower: MoE drift. SUMMARY failed 0, 13/13 tests; prepare 105.6 s (the tuner 0.6 s) |
| 2026-10-29 | 7631043129f74b61 (check) | 849d666 | comment-only repeat of 5285e58d (the measured residency in the kernel comment) | 1.887x (1.858-1.915) | per shape d1 2.091/2.002, d16 1.824/1.984, d32 1.413/1.814, v1 2.186/2.120, v16 1.796/1.788, v32 1.932/1.852. Same picks at every token count; chain times 8-CTA vs pick T=128 22.43 vs 17.82, T=256 29.47 vs 22.27. Boundary rows v16 5.1 / 4.6 and 5.0 / 4.4 (SGLang's two), v32 7.5 / 7.5, d32 3.7 / 3.8. Our us against the three controls: v16 290.6 (-15.3 vs their mean) and 390.7 (-17.1); v32 375.7 (-5.7) and 579.5 (-11.2). Over both runs the v16 / v32 steps are below the control mean on all four shapes, by 5.7-19.9 us, against the predicted 5 / 8; within-run ratios v16 1.791-1.796 / 1.756-1.788 and v32 1.915-1.932 / 1.840-1.852: v16 sliding above the controls' range (1.760-1.788) in both runs, the other three inside their ranges, none below. SUMMARY failed 0, 13/13 tests; prepare 105.3 s. Submit candidate: 849d666 (timed code of 3eec2bb) |
| 2026-10-29 | 8a4ebdf2ce474b24 (submit, under the standing replace-when-demonstrably-better authorization) | 5e71840 | file of 7631043129f74b61 (849d666): attention boundary geometry picked per bucket | 1.876x | passed. Board entry before: 22d5710c4e8540d6 (1.878x) |
| 2026-10-29 | 181ab39f1bd84912 (check) | 90a8d85 | Timeline instrumentation only (timed code of 3eec2bb): at the end of prepare, `_timeline` captures kernel_fn as a graph on synthetic inputs (ctx 80000 on every request, not the judge's agentic spread) for verify 16 and decode 32, sliding and full, replays it 7 times after an L2 flush and a host sync under torch.profiler (CUDA activity), and prints the median-span replay: each kernel's stream, start, end, and idle time before it on its stream and on the device (`TL` lines in the log). The printed span includes the next replay's 80 us flush and the 13 us host sync before it; the step ends at the FFN boundary's end | 1.867x (1.839-1.895) | per shape d1 2.048/1.980, d16 1.831/1.966, d32 1.396/1.858, v1 2.171/2.135, v16 1.739/1.739, v32 1.906/1.835. Timeline (measured, under the profiler, one synthetic input): device idle inside the step is 4.9 us at verify 16 (11 boundaries of 0.0-0.8 us) and 2.1-2.2 us at decode 32, so most of the gap map's "time between kernels" residual is not idle time. Largest specific items: (1) decode 32, mma front: the router mma (11.5 us) and routing run on the other stream beside the pooler sk kernel, which stretches from its 17.3 us profile row to 26.0 / 25.2 us; attention boundary end to MoE start is 28.8 / 27.9 us against about 19 for the pooler and norm alone. (2) verify 16, concat front: the second stream's hi + lo add (2.1-2.3 us) and routing (2.9) end 3.6-4.2 us after the main stream's pre-expert norm, and the MoE launch waits for them (MoE start 0.4 us after routing ends). (3) qkv at 128 tokens is cuBLAS split-K plus a 3.1 us reduce kernel. No memset, copy or host op is in the graph; the attention (cooperative) and MoE (persistent) launches start 0.4-0.7 us after their predecessor ends, like every other boundary. SUMMARY failed 0, 13/13 tests; prepare 103.5 s |
| 2026-10-29 | 5869cc624a394904 (check) | 2f4927b | Item (1): the fused pooler + router launch (`skf`, one sk launch that streams the 12-bit pooler and then one 8-row hi/lo router piece per CTA through the same ring, last-block reduction of the router pieces) built at 32 tokens as `pool32f` (`launch_sk<2, 32, 2, 2, 2, 4, 3, true, false, 1, 1, bf16, true>`, the pool32 geometry; skf_fwd case 26), self-tested at m 17 and 32 against FP64 like pool8f / pool16f, offered to the MoE-front tuner at T=32. Same template, so the same fences and barriers (`fence.proxy.async` before the empty-slot arrive, `__threadfence` around the piece counter); compiled for sm_90a in Docker: 111 registers, no spills. rpart [8, 16, 256] holds 4 pieces x 32 tokens | 1.898x (1.870-1.926) | per shape d1 2.073/2.018, d16 1.859/2.000, d32 1.447/1.886, v1 2.180/2.152, v16 1.770/1.751, v32 1.950/1.870. Tuner T=32: skf 32.4, concat 37.6, mma 37.8, split 43.2 us -> skf (routes 1.000, rel diff 0). Timeline: attention boundary end to MoE start 24.6 / 24.1 us (was 28.8 / 27.9); the fused launch is 20.3 / 19.6 us. d32 us 183.5 / 332.2. MoE pick at T=32 main-own-dynb, as in the controls. Unexplained: the timeline's mode 23 MoE kernel took 129.9 / 127.3 us on the same synthetic input where check 181ab39f had 83.2 / 84.3; the judged d32 steps fell, so it is not in the scored time, but it is not understood. SUMMARY failed 0, 13/13 tests; prepare 104.5 s |
| 2026-10-29 | cb88e4bad03c4157 (check) | 43459ff | comment-only repeat of 5869cc62 (the measured front times on pool32f) | 1.893x (1.865-1.922) | per shape d1 2.072/2.045, d16 1.845/1.972, d32 1.462/1.863, v1 2.195/2.171, v16 1.761/1.755, v32 1.923/1.847. Tuner T=32 skf 32.4 vs mma 37.3 -> skf; MoE pick at T=32 flipped to sk8w4s (94.6 vs 95.0), a confounder on d32 in this run only. Timeline front 25.3 / 24.3 us. d32 within-run ratios, new vs the three controls on the old front (181ab39f, 7631043129f74b61, 5285e58d978f42c6): sliding 1.447, 1.462 vs 1.396, 1.413, 1.446; full 1.886, 1.863 vs 1.858, 1.814, 1.830. Both new runs are above every control on both d32 shapes (sliding by 0.001 over the best control in 5869cc62). d32 us 183.5, 177.3 vs 192.8, 187.5, 182.8 (sliding) and 332.2, 326.8 vs 335.1, 340.2, 338.6 (full), means -7.3 / -8.5 us against the -4 us the timeline front predicts; the rest is run-to-run spread. Other loads use unchanged code and move inside their ranges (v16 1.761-1.770 / 1.751-1.755 vs 1.739-1.796 / 1.739-1.788). The headline (1.898x, 1.893x vs 1.867x, 1.887x, 1.891x) moves inside the CIs, as a 4 us change on two shapes predicts. Submit candidate: 43459ff (timed code of 2f4927b; prepare also prints the timeline, about 5 KB of log; prepare 103.5-106.9 s in these runs against 105.3-105.7 before) |
| 2026-10-29 | 4b61ce76968f45c0 (submit, under the standing replace-when-demonstrably-better authorization) | dbe0596 | file of cb88e4bad03c4157 (43459ff): fused pooler+router front at 32 tokens, prepare-time timeline | 1.865x | passed. Board entry before: 22d5710c4e8540d6 (1.878x) |
| 2026-10-29 | pending (no layer checks left today; the coordinator runs the scored check) | see the commit that adds this row | three ports, one bundle. (1) MoE fat mode 26 ("main-dynb-fs": 23 with one shared route count across warps, x gather loads before stores, no stage-range searches; muse_moe_w4a16 817eab6401d54a84) added to `LAYER_MODES` (0, 16, 20, 23, 26) and re-inlined; the layer's judge-like routing bench picks. The inliner output is byte-identical on a second run; against the standalone assembly of the same variants the layer copy differs only in the local_start patch (Params::lo_p, ld_lo, three uses) and the nested namespace; textual counts equal (fence_proxy_async_global 6, red_release_add 3, named_sync 11, __threadfence 10, mbar_wait 10, spin_ge 4, __syncthreads 19, ld_acquire 16, st_release_cta 2, bq_get 3, griddepcontrol 4); sm_90a in Docker: mode 26 96 registers, 64 B frame, 120 / 308 B spill stores / loads, the same as mode 23 here and as mode 26 compiled from the standalone (spills outside the stage loop, docs/runs-moe.md). (2) The wide projection kernel (`wd`, muse_bf16_proj c09dc54) for 65 to 128 tokens with the two configs of 39d6bfa9834f4a96 only, b1n128c2s8h and b1n128c3s8h (8-stage ring, L2 policy, uneven K split), as `_SKP_WD_SRC` in the skp extension via `tools/dl_proj_inline.py` (which also now generates pool32f through a layer FUSED_GEOM instead of the hand edit, and keeps the layer's `bucket_of`). Findings first: at T=128 `_proj` sent the qkv_gate to cuBLAS, and the pooler ran inside the "concat" front (one cuBLAS GEMM for the pooler and both router halves; TUNE front T=128 concat 39.8 vs split 50.0 in cb88e4bad03c4157), so the wide pooler can only reach the pooler through a front variant. Prepare self-tests every config that tiles qkv_gate and the pooler against FP64 at 65, 100 and 128 tokens (5e-3, probe rows bitwise, diag clear), times them at 128 tokens against cuBLAS as cold-L2 graph replays, and keeps one only under 0.97x cuBLAS (the standalone's WD_MARGIN); `_proj` runs the pick for 65 to 128 tokens. New front variant "wide" (offered at the T=128 front key when a pooler pick exists): the pooler on the wide pick on the main stream, the router as one cuBLAS GEMM over the stacked hi/lo rows with FP32 output on the side stream; the front tuner times it beside split (which now also gets the wide pooler through `_proj`) and concat, with its 1e-3 / 99% routes check. A wide failure leaves the sk picks on. sm_90a in Docker: both wd instances 90 registers, no spills, every one of the 43 skp kernels spill-free; SASS fence/barrier counts equal for the two configs (FENCE.VIEW.ASYNC.S 2, MEMBAR.ALL.CTA 3, MEMBAR.ALL.GPU 2, BAR.SYNC 4, UCGABAR arrive/wait 2/2, SYNCS try-wait 4); the device text is the standalone's wd_source() output. CPU dry run (fake wd_fwd with the launcher's argument checks): picks qkv_gate c3 and pooler c2 on the standalone's times, `_proj` routes 100 and 128 tokens to the pick and 64, 129, 32 elsewhere, "wide" offered at T=128 only. (3) attn_one.cu from main c26a32a re-inlined (`ONE_QKOV`, `ONE_EREL`, `ONE_PCHK`, `ONE_UNR2` on; mode 51 and 115 only; cb85d590c301410e / 6f75235c0d364a7a): the old source re-inlined gives the previous layer byte for byte and the new one is reproducible, so the layer change is exactly the attn_one.cu change; only `_ONE_SRC` moved (Params::lp, the host wrapper and `_one_wins` untouched). sm_90a: 12 kernels, no spills, mode 51 80 registers; SASS sync counts change only in mode 51, from 4 / 11 / 33 / 2 to 10 / 16 / 41 / 6 (FENCE.VIEW.ASYNC.S / BAR.SYNC / SYNCS / WARPGROUP.DEPBAR), the standalone's numbers before and with UNR2 (docs/runs-attention.md). No subprocess, reference import or file read in the file. Not run on a GPU: wd in the layer, the wide front, prepare time (mode 26 adds a fat kernel to the moe1 build and two cluster kernels to the skp build, both on background threads) | pending | expected from standalone: v16 MoE -4 to -6 us (26 vs 23), qkv_gate -2.9 us at T=128 if picked; the pooler only if "wide" or split beats concat (39.8 us), unverified in the layer; v32 full -8 us from mode 51 |
| 2026-10-29 | 5bb37c21c13245b8 (submit, under the standing replace-when-demonstrably-better authorization; no layer checks left, so the scored run was the test) | 745021d | bundle: MoE mode 26 inlined, wide 8-stage qkv_gate/pooler picks at 65-128 tokens and a "wide" front variant, attn_one.cu with the public tile order | 1.908x | passed, failed 0. MoE picks mode 26 (main-dynb-fs) at 32/128/256: 106.2 / 192.1 / 240.7 us. Front at 128 kept concat (39.3) over wide (41.4) and split (46.7). Prepare 90.0 s. Board entry before: 22d5710c4e8540d6 (1.878x) |
| 2026-10-29 | pending (no layer checks left today; the coordinator runs the scored check) | see the commit that adds this row | MoE fat mode 28 ("main-lean": 26 with the ring size and x slot base in registers, every consumer thread releasing the weights slot with its own fence.proxy.async.shared::cta and mbarrier.arrive behind the stage's first wgmma group, wempty and xempty initialised to kNC x 32 = 512 arrivals, one-pass S.ord sort; muse_moe_w4a16 f928f8a802014e4f 2.149x and repeat 13fe68e7ca2e423f 2.158x) re-inlined with `LAYER_MODES` (0, 16, 26, 28) and `28: "main-lean"` in `MAIN_NAMES`; 20 and 23 leave with the standalone's MAIN_MODES. Variants now sk8w4s, sk16w4s, main-d2, main-m2-ar2-ub, main-dynb-fs, main-lean (one fat kernel fewer to build). The inliner output is byte-identical on a second run. Against the standalone `_cuda_source()` with the same variants the device code differs only in the local_start patch (Params::lo_p, ld_lo, three uses) and the moe1 namespace; textual counts equal (fence_proxy_async_global 6, fence.proxy.async.shared::cta 2, fence_proxy_async 15, red_release_add 3, named_sync 11, __threadfence 10, mbar_wait 10, mbar_arrive 12, mbarrier.arrive 3, spin_ge 4, __syncthreads 19, ld_acquire 16, st_release_cta 2, bq_get 3, griddepcontrol 4, __syncwarp 22). sm_90a in Docker (CUDA 12.8, torch's flags), layer moe1 extension vs the standalone of the same variants: main<28> 96 registers, 80 B frame, 184 / 512 B spill stores / loads; main<26> 96, 64 B, 120 / 308; main<16> 96, 136 / 228; main<0> 96, 136 / 236; sk8w4s 140 and sk16w4s 96 with no spills; every number equal to the standalone. SASS per kernel: FENCE.VIEW.ASYNC.S, FENCE.VIEW.ASYNC.G, MEMBAR (ALL.CTA / SC.GPU / ALL.GPU), SYNCS.ARRIVE / PHASECHK / EXCH, BAR.SYNC, WARPSYNC, ERRBAR, CCTL, UBLKCP, WARPGROUP.ARRIVE / DEPBAR, LDL / STL counts identical to the standalone for all six kernels (main<28>: 21 / 11, 40 / 2 / 17, 48 / 74 / 16, 15, 103, 19, 16, 4, 96 / 108, 128 / 46); non-NOP instruction counts differ by 1 to 6 per kernel (the ld_lo loads). Self-test (every _MOE1_VARIANTS entry, 17 to 264 tokens on the fat path, normal and skewed, scratch invariant incl. ctr[10]), the 2,000-replay stress (every fat mode at 128 + 256 tokens in one graph, a failing fat mode dropped alone) and the judge-like routing bench at every TUNE_TOKENS count from 17 up iterate the variant table, so mode 28 gets the coverage 23 and 26 had; SUMMARY prints `moe stress main-lean=...` and the pick with every variant's time per T. Not run on a GPU | pending | expected from standalone layer-like routing: MoE -6.4 / -8.6 / -8.8 us at 32 / 128 / 256 tokens against mode 26 if 28 is picked (5bb37c21c13245b8 picked 26 at all three; at 32 tokens 28 was also 18 us under sk8w4s on layer routing, 6 of 6 seeds) |
| 2026-10-29 | 0fae646889864106 (submit, under the standing replace-when-demonstrably-better authorization; no layer checks left) | 7dde781 | MoE mode 28 (main-lean) inlined, LAYER_MODES (0, 16, 26, 28) (d39615b) | 1.930x | passed, failed 0. MoE picks main-lean at 32/128/256: 106.3 / 174.9 / 233.2 us (mode 26: 111.6 / 184.7 / 242.4). Board entry before: 5bb37c21c13245b8 (1.908x) |
| 2026-10-29 | pending (no layer checks left today; the coordinator runs the scored check) | see the commit that adds this row | attn_one.cu from main ce7e8fe re-inlined by tools/dl_one_inline.py: `ONE_PFAST` 1 (modes 49 and 51 here; a producer tile whose rows are all cached is gathered without per-row branches, with 32-bit offsets and the page index by shift; standalone checks ca0f13d324ef48e8 2.393x and 0d05f9b2df1e4834 2.450x, judged -5.8 / -12.3 us at verify 16 / 32 full and -2.6 / -10.4 at decode 16 / 32 full), plus `ONE_STAMP` 0 and `ONE_PROBE` 0 (bench-only, compiled out). Host: the layer wrapper now sets `Params::off32` by the standalone rule, `(1 << lp) == page_size && max(k_cache.numel(), v_cache.numel()) <= 2^32`. The layer passes the same e4m3 `[pages, page_size, 2, 64]` caches as the standalone (128-byte rows from data_ptr), so the same bound holds; larger caches keep the 64-bit path. Without this line the inlined kernel would read an uninitialized `off32` (`Params p` has no initializers). Inliner reproducible: c26a32a's attn_one.cu re-inlined gives the committed 7dde781 layer byte for byte, and the new inline is byte-identical on a second run. Layer `_ONE_SRC` against the standalone attn_one.cu: only the nested `one` namespace, the two `griddepcontrol.wait` insertions, the dropped INSTANTIATE notes and the dropped host functions. sm_90a in Docker (CUDA 12.8, torch's flags), the 11 layer instances against attn_one.cu compiled with the same instances and flags: BAR, FENCE, MEMBAR, SYNCS, ATOMG, ATOMS, REDG, CCTL, ERRBAR, DEPBAR, LDGDEPBAR and EXIT counts identical per instance (51: 16 / 10 / 11 / 41 / 1 / 2 / 11 / 4 / 1 / 2 / 1 / 4; 49: 10 / 1 / 2 / 28 / 1 / 2 / 5 / 2 / 1 / 2 / 1 / 5); the only opcode difference is ACQBULK (griddepcontrol.wait), 1 per per-request instance and 2 per flat one, the documented layer patch. Registers equal to the standalone everywhere (51 80, 49 128, 19 112), no spills (`tools/dl_compile_check.sh muse_decode_layer/submission.py _ONE_SRC`: 0 bytes spill on all 12 kernels). Against the previous layer copy, 9 of 11 instances are SASS-identical; only <1,49> (3144 to 3456 instructions) and <8,51> (6240 to 6544) change. Coverage with PFAST on: self-test cases (batch, q_len, layer, page) (2,1,full,1), (3,1,full,1), (16,1,full,1), (33,1,full,1), (4,1,full,64) plan 49 and (3,8,full,1), (5,8,full,1), (16,8,full,1), (3,8,full,64) plan 51 (page 1 and 64, so lp 0 and 6), and the 16x1, 16x8 and 32x8 full stresses run 49 / 51 for 2,000 replays; their caches are at most about 21 MB, so off32 is 1 and the fast path runs on every fully cached tile. Not run on a GPU here | | |
| 2026-10-29 | a80dc97048a940d7 (submit, under the standing replace-when-demonstrably-better authorization; no layer checks left) | f2d4eb5 | attn_one.cu re-inlined from ce7e8fe (ONE_PFAST on modes 49/51; wrapper sets off32) | 1.934x | passed. Board entry before: 0fae646889864106 (1.930x) |
| 2026-10-29 | 9b921febc7ad4140 (submit, by the coordinator with the last scored run of the day) | a923f11 | Gap map E2: front variants `concat-r` and `skf-r` (one main-stream route kernel: hi + lo logit add, SGLang's top-8 and renormalized weights, the pre-expert norm with pre_norm_kernel's arithmetic; no side stream, no join) with the fused MoE launched as its PDL secondary (layer patch 2 in tools/dl_moe_inline.py: mbarrier init and an L2 prefetch hint before griddepcontrol.wait); `_front_tune` picks by front + MoE time where a route variant is offered. Also the m = 1 cgemv re-inlined from main 01b553f (planar 16-byte stream, 32-register cap) | **1.964x** (1.935-1.994) | passed 13/13, route self-test on, failed 0. Per shape us (a80dc97048a940d7 before): d1 77.6 / 85.5 (82.5 / 89.1), d16 143.5 / 212.9 (142.0 / 222.7), d32 168.0 / 329.3 (176.5 / 319.2), v1 118.6 / 133.0 (123.1 / 133.4), v16 271.2 / 368.3 (282.3 / 376.7), v32 345.9 / 532.0 (356.8 / 539.8); cross-run. Front picks: T=1 fp32, 8 skf-r, 16 skf, 32 skf-r, 128 concat-r, 256 concat-r. Same-process front + MoE (graph medians of 30, prepare): T=128 concat-r 251.7 vs concat 254.6 (-2.9), T=256 311.3 vs 317.4 (-6.1), T=32 skf-r 191.3 vs skf 191.9 (-0.6; concat-r vs concat -6.6), T=8 skf-r 56.8 vs skf 57.0, T=16 skf 141.5 vs skf-r 141.6. TL: front GEMM end to MoE start 1.9 / 1.8 us at v16s / v16f against 5.9 / 6.5 in a80dc97048a940d7 (the MoE CTAs start 1.6-1.7 us after the route kernel; its row 4.5-5.5 us includes the wait on the GEMM), so the E2 gate (>= 2 us at 128 tokens) passes. The cgemv port is a separate change in the same run: decode 1 qkv row cgemv<256,2> 10.1 / 9.5 us against 10.6 / 10.9; the T = 1 pooler moved from sk_kernel 15.6 / 15.4 to cgemv<512,1> 14.6 / 14.9 and the unpooler slice to sgemv 3.7 / 3.8 (cgemv<128,2> 4.0 / 4.1); single-input rows. The SKP selftest and SKP tune m=1 lines fell outside the 16,000-character log tail. Profiles committed |
| 2026-10-30 | 2cd4e3b292904662 (check 1 of the 10-30 plan) | d006531 (file SHA256 c61ea2f8...) | Integrated candidate: route kernel, m = 1 cgemv (reworked), MoE mode 37, sk8w3s, sk8w4s-er, qkv8, attn_one DCARRY + PK16 + VHOIST, FIN8 stress at ctx 20000 with plan assertions and a fresh-qkv step, mixed-size MoE replay stress, summary manifest | 1.785x (1.758-1.811), **failed the plan** | 13/13 judge tests passed, but `attn one` fell back to SGLang's paths: the 1x1 sliding stress's speculated-length fallback, now run on the fresh qkv rows the new step left in the captured buffer, measured 5.20e-2 against SGLang's path (bound 5e-2; replayed equal to eager, workspace clean), so `attn_one_ours=0` and every shape ran attn_fused / attn_ws / attn_prep (d16 full 263 us, v16 full 448, v32 full 682). Not a kernel defect: a test-input change pushed a tolerance comparison over its bound. Stopped as planned; fix in the next row. Everything else selected as intended: MoE picks sk8w3s at 1 / 8 / 16 and main-lean-silu-bar at 32 / 128 / 256; fronts fp32 / skf / skf-r / skf-r / concat-r / concat-r at 1 / 8 / 16 / 32 / 128 / 256; SKP cgemv self-test ok on all four layers; m = 1 tune: qkv_gate cgemv 14.1, o cgemv 10.1, pooler cgemv 19.9, unpooler cgemv 8.3 (picks cgemv on all four); m = 8 qkv8 16.6; MoE PDL self-test every variant <= 4.4e-5; every MoE mixed-size stress ok. Prepare 119.5 s. Profiles committed |
| 2026-10-30 | 3e2b5771d7254fd2 (check 1 again) | d006531 + the stress fix (the fallback check restores the first qkv rows; file SHA256 71ff5152...) | as above | 1.960x (1.931-1.990) | passed 13/13, failed 0, every stage on (attn_one_ours=1). Plans as scored: d1 / v1 full (252, 8, 11), d16 / d32 full (264, 1, 49), v16 / v32 full (264, 1, 51). FIN8 stress now plans (252, 8, 11) at 1x1 and 1x8 full (asserted), 2000 replays each, replayed vs eager 0.0, vs SGLang 3.25e-3 / 2.41e-2, workspace clean. MoE picks: sk8w3s at 1 / 8 / 16, main-lean at 32 (101.6 vs main-lean-silu-bar 102.2 on this run's routing), main-lean-silu-bar at 128 / 256. Fronts: fp32 / skf-r / skf-r / ... / concat-r. SKP m = 1 picks cgemv on all four layers (qkv_gate 14.1, o 9.8, pooler 20.2, unpooler 8.1 us). Per shape us against 9b921febc7ad4140 (cross-run, SGLang times also moved): d1 77.9 / 84.2 (77.6 / 85.5), d16 140.6 / 218.3 (143.5 / 212.9), d32 174.6 / 331.7 (168.0 / 329.3), v1 118.6 / 123.7 (118.6 / 133.1), v16 267.5 / 355.2 (271.2 / 368.3), v32 350.8 / 522.0 (345.9 / 532.0). Neutral against 1.964x within both CIs: verify full -9 to -13 us, decode 32 sliding +6.6 (its SGLang time 252.1 vs 259.8, ratio 1.444 vs 1.546, the largest single loss). The decision waits for the same-process comparison (next rows). Profiles committed |
| 2026-10-30 | (no job: request rejected, 413) | bench, tools/dl_make_bench_pub.py at 1772ef6 | Four-arm bench (inc, cand, null, pub) in one file | none | The 2.12 MB file was refused before a job existed ("request too large"); a size probe on muse_prefill_layer (also rejected before a job: "submission is larger than 1000000 bytes", 422) gave the limit. No quota used. The bench was split into one other arm per check, sent comment-stripped and lzma-compressed (bf4257f) |
| 2026-10-30 | 135ab2a9f22b49a5 (check) | bench at bf4257f: cand = the 3e2b file plus the FP32 fallback diagnostic (SHA256 4ed09046...), other = inc (a923f11) | cand / null / inc, all 12 shapes | judge 2.012x (1.981-2.042) on cand; bench arm inc **invalid** | Every ABCHECK passed for all three arms, but the inc arm ran degraded (prepare 3 s, MoE policy None, no route variants): the comment stripper had dropped the `//@MOE1_DISPATCH@` and `//@W4_*@` marker lines that `_moe1_source()` and `_w4_source()` replace, so the incumbent's fused MoE and W4 dispatch had no cases. Its timings (cand - inc -23 to -240 us) are not a comparison and are not used. The fallback diagnostic did run (row below). Fixed in 0d380c5 (comment lines named in a string are kept); resubmitted as aa4276e91c924dad |
| 2026-10-30 | 121dff3ab17a4245 (check) | bench at bf4257f, other = pub | cand / null / pub | judge 1.965x (1.935-1.994) on cand; bench **did not run** | The public module failed at import: Triton's @jit needs the kernel source from a file ("@jit functions should be defined in a Python file"), and the arm was exec'd from a string. A first fix (a module in sys.modules with a source loader) was refused by the static check ("rejected: edits loaded modules", no job); the accepted fix runs the arm in a plain globals dict and registers its source with linecache.lazycache (6c6038b). Resubmitted as d9e07c554f614d83 |
| 2026-10-30 | (fallback diagnostic, from 135ab2a9f22b49a5 and 121dff3ab17a4245) | cand | Batch-1 speculated-length fallback on fresh qkv rows against an FP32 attention block | - | ours vs FP32 / SGLang vs FP32: 1x1 sliding 1.63e-3 / 5.21e-2, 1x8 sliding 3.63e-2 / 4.06e-2, 1x1 full 1.75e-3 / 6.75e-3, 1x8 full 1.76e-2 / 2.19e-2. Ours is closer to FP32 than SGLang's path on all four; the 5.20e-2 trip in 2cd4e3b292904662 is SGLang's own error on that input, so restoring the original rows for the fallback comparison (the 5e-2 bound unchanged) is sound |
| 2026-10-30 | aa4276e91c924dad (check, bench) | tools/dl_make_bench_pub.py at 6c6038b; cand = the 3e2b file + FP32 diagnostic (SHA256 4ed09046...), other = inc a923f11 (SHA256 3e9ba3df...) | Same-process whole-layer comparison, cand / null / inc, 12 shapes x 2 held-out inputs (seeds 50001+, busiest and second-busiest local range), 8 rounds each, cold-L2 graph replays, seeded random arm order | judge 1.970x (1.941-2.000) on cand; bench: **cand / inc 1.0158** (geometric mean of the six load ratios), null / inc 1.0164 | 72 of 72 ABCHECK PASS (judge rules against the restated FP32 reference, routing acceptance, cache entries). The inc arm ran healthy this time (its stages all 1, failed 0, plans as scored). Paired cand - inc us (null - inc): d1 +0.2 / -1.3 (-0.3 / -2.1), d16 -1.9 / -2.1 (-1.5 / -1.5), d32 +0.9 / -1.0 (+1.4 / -2.3), v1 -0.7 / -0.7 (-0.7 / -0.4), v16 -6.2 / -13.3 (-5.0 / -13.7), v32 -3.5 / -22.5 (-2.0 / -22.0). Loads (46 s + 16 f) / 62, inc -> cand: d1 90.7 -> 89.6, d16 166.4 -> 165.1, d32 215.9 -> 213.9, v1 136.2 -> 134.8, v16 304.9 -> 295.3, v32 402.2 -> 393.5. The null arm is the candidate file timed a second time, and with inc as the reference arm its printed 1.0164 is null / inc, not null / cand: cand load / null load per load is d1 1.0034, d16 1.0043, d32 0.9986, v1 1.0045, v16 0.9970, v32 0.9960, geometric mean 1.0006, so the bench's noise is about +-0.5% per load and the candidate's 1.016 over the incumbent stands at about 3x it (the harness now prints the null against cand explicitly, ABLOAD and ABSCORE). The gain sits in verify 16 / 32 (attention DCARRY + PK16 + VHOIST, MoE mode 37); every other load moves 1-2 us, the size of the null arm's spread. The cand MoE pick at 32 tokens was sk8w3s in this run and main-lean-silu-bar in the next (the bench picks per run on judge-like routing). Timelines lost to the log tail |
| 2026-10-30 | d9e07c554f614d83 (check, bench) | as above, other = pub (vjeux/okc-2026 blob 936fefff, SHA256 cf4a75ee...; sent comment-stripped, SHA256 e0ec5c48...) | cand / null / pub, seeds 60001+ | judge 1.995x (1.965-2.025) on cand; bench: **pub / cand 1.0464**, null / cand 0.9989 | 72 of 72 ABCHECK PASS, the public arm included. Paired pub - cand us (null - cand): d1 +2.6 / -0.4 (+0.7 / -0.7), d16 -6.0 / -8.2, d32 -12.9 / -25.2, v1 -9.6 / -10.2, v16 -16.6 / -11.8, v32 -17.0 / -19.1 (null within +-2.2 on every shape). Loads, cand -> pub: d1 86.9 -> 88.8 (we lead), d16 157.1 -> 151.1, d32 211.8 -> 195.3, v1 126.6 -> 116.7, v16 305.8 -> 290.8, v32 400.6 -> 384.5. Shares of the summed log gap (0.271): v1 30.0%, d32 29.9%, v16 18.5%, v32 15.1%, d16 14.4%, d1 -8.0%. This replaces the cross-run gap map: decode 32 (16.5 us weighted) and verify 1 (9.9 us) are the largest measured same-process gaps; the decode 1 gap of the cross-run maps does not exist (+1.9 us our way). Timelines lost to the log tail (ABTL sorts first; a next bench prints fewer) |
| 2026-10-30 | 261a76ac7b364091 (check 4, confirmation) | the bench candidate file (3e2b file + FP32 diagnostic) with one comment line added (SHA256 1355d39f...) | release candidate, repeated | **1.989x** (1.959-2.019) | passed, every stage on, failed 0. Per shape us: d1 77.5 / 83.9, d16 140.8 / 217.4, d32 176.2 / 330.0, v1 118.2 / 128.4, v16 280.5 / 365.2, v32 351.9 / 528.0. MoE at 32 tokens picked main-lean-silu-bar this run. The judge's six readings of this candidate's code today: 1.960 (3e2b), 2.012 (135a), 1.965 (121d), 1.970 (aa42), 1.995 (d9e0), 1.989 (261a), mean 1.982, against the incumbent's single 1.964 (9b921febc7ad4140); same process it leads the incumbent by 1.6% with the null at 1.0006 (aa4276e91c924dad). Profiles committed |
| 2026-10-30 | e2d1d81447cc48d9 (check, M1 follow-up) | 6377b5a (file SHA256 5d0580f7...) | Release candidate + M1 (SK_PDL build, ONE_TRIG, o / skf front as PDL secondaries via _m1_tune) + LATE_ARRIVE + o16 / unp32 geometry + MoE mode 47 | 1.963x (1.933-1.992) | passed 13/13, failed 1: **M1 blocked**. The bit-16 self-test passed (PDL launch bitwise equal to the plain one on 14 o / skf launches), the forced-on replay stress passed 1x1 sliding / full and 1x8 sliding, then failed 1x8 full: after 2000 replays hidden 2.0e-4 and stream 1.6e-4 from the M1-off eager outputs (bound 1e-4; ids equal). M1 was switched off (policy {}), so it never ran in the timed path; the chain timing and the per-bucket pick did not run. Everything else selected as intended: MoE picks sk8w3s at 1 / 8 / 16, main-lean-silu-bar at 32, main-lean-grp-nohoist (47) at 128 / 256 (183.3 / 225.9 against 185.5 / 228.4 for 37 on this run's judge-like routing); every MoE mixed-size stress ok including 47; MoE PDL self-test every variant <= 2.9e-5. Per shape us: d1 76.7 / 82.6, d16 151.2 / 220.5, d32 176.3 / 324.6, v1 123.3 / 134.6, v16 264.6 / 353.2, v32 346.5 / 506.0 (one judge reading; the SK_PDL build changes the sk kernels' first-ring issue order even with M1 off). Profiles committed |
| 2026-10-30 | edd71d3152a946fe (submit, by the coordinator under the standing replace-when-demonstrably-better rule) | the 261a76ac7b364091 file (SHA256 1355d39f...) | Release: route kernel, reworked m = 1 cgemv, MoE mode 37, sk8w3s, sk8w4s-er, qkv8, attention DCARRY + PK16 + VHOIST, the stress and diagnostic changes | **1.982x** (1.953-2.012) | passed 13/13. Per shape us: d1 76.8 / 84.6, d16 138.2 / 221.2, d32 171.3 / 327.9, v1 122.6 / 122.4, v16 270.0 / 356.4, v32 366.5 / 514.3. Board entry before: 9b921febc7ad4140 (1.964x). Basis: same-process +1.6% over the incumbent with a clean null (aa4276e91c924dad) and six check readings averaging 1.982. Profiles committed |
| 2026-10-30 | c2ba945496ab47b4 (check) | 2cee446 (file SHA256 b9c23bda...) | Clean candidate: the release file (1355d39f) through tools/dl_proj_inline.py (main: LATE_ARRIVE, o16 / unp32 geometry; SK_PDL left at its default 0, so no M1 code is built) and tools/dl_moe_inline.py (mode 47); nothing else. Against the release only _MOE1_SRC, _MOE1_VARIANTS, _SKP_SRC and _SKP_CONFIGS differ; inliners byte-identical on a second run. SASS (CUDA 12.8.1, sm_90a) against the release: the nine shared MoE kernels identical in every counted opcode, main<47> as recorded against the standalone at 6377b5a (MEMBAR 43, BAR 15, ERRBAR 22, CCTL 8, UBLKCP 4, SYNCS 202, LDL 22, STL 1); projections 43 -> 39 instances (o16 and unp32 now reuse the unp16 and o32 instances), 35 identical, the four LATE_ARRIVE instances differ in instruction totals only (FENCE, MEMBAR, SYNCS, BAR, ERRBAR, LDL, STL equal), so no fence or barrier dropped and no new spill | **2.006x** (1.976-2.037) | passed 13/13, failed 0, every stage on. MoE picks sk8w3s at 1 / 8 / 16, main-lean-grp-nohoist (47) at 32 / 128 / 256 (102.9 / 176.3 / 244.0 against 37's 104.9 / 178.1 / 248.1). Per shape us: d1 77.5 / 84.1, d16 136.6 / 216.9, d32 169.4 / 316.5, v1 123.6 / 128.8, v16 260.8 / 353.0, v32 350.7 / 525.1. Per load against the mean of the release's six readings (tools/dl_shape_table.py): d1 -0.7%, d16 +3.3%, d32 +3.2%, v1 -2.3%, v16 +4.2%, v32 -0.4%, geometric mean +1.2%. Profiles committed |
| 2026-10-30 | 86ced49f092a48d8 (check, bench) | tools/dl_make_bench_pub.py at this commit (ABK per-kernel tables, ABCHK counts); cand = the c2ba file, other = pub (vjeux/okc-2026 blob 936fefff, SHA256 cf4a75ee..., sent comment-stripped e0ec5c48...), seeds 70001+, order seed 11, 8 rounds | Same-process whole-layer comparison with per-kernel tables, cand / null / pub | judge 1.991x (1.961-2.021) on cand; bench: **pub / cand 1.0364**, null 1.0002 | 72 of 72 checks pass. Paired pub - cand us: d1 +2.5 / +0.5, d16 -9.5 / -1.7, d32 -15.7 / -10.8, v1 -8.7 / -10.5, v16 -10.7 / -7.3, v32 -15.8 / -9.5. Loads cand / pub: d1 0.974 (we lead), d16 1.040, d32 1.070, v1 1.074, v16 1.032, v32 1.032 (was 1.0464 overall against the release file in d9e07c554f614d83). Per kernel (median over five profiled replays of each kernel's summed lifetime per step, tools/dl_kernel_gap.py): at v1 full every one of our kernels is equal or faster (qkv 12.7 / 11.9, attention 15.2 / 15.0, o 7.0 / 7.8, pooler + router 19.7 + 4.6 / 19.9 + 6.2, MoE 68.0 / 70.6, boundaries 4.2 + 3.7 / 5.2 + 4.9) yet the span is 139.2 / 130.3; our summed lifetimes equal our span (139.7) while the public one's exceed its span by 15.8. Same at v1 sliding (153.9 / 154.0 against 160.8 / 145.1), d16 sliding and d32 sliding (180.4 / 179.8 against 179.2 / 166.1). Whether the public excess is real overlap or PDL waits counted inside its secondaries is not separable here (start times in ABTL fell out of the tail). Largest per-kernel lifetime gap: MoE at 128 / 256 tokens, mode 47 against okc::moe_fused<true>: v32 243.7 / 217.8 (s), 238.3 / 212.6 (f); v16 163.6 / 144.5, 169.0 / 154.2; d32 97.9 / 91.5 |
| 2026-10-30 | 50c4c1328ab742f4 (check, bench) | bench at 2cee446; cand = the c2ba file, other = inc = the release file (SHA256 1355d39f..., sent comment-stripped 86ec0223...), seeds 80001+, order seed 13, 8 rounds | Same-process clean candidate against the release, cand / null / inc | judge 2.003x (1.972-2.033) on cand; bench: **cand / release 1.0182**, null 0.9987 | 72 of 72 checks pass, both arms healthy (prepare 19 / 15 s; inc MoE policy mode 37 at 32 / 128 / 256, cand mode 47). Paired cand - inc us: d1 -5.2 / -5.0 (null -4.3 / -4.9), d16 +0.4 / -0.9, d32 -3.2 / -3.6, v1 +0.9 / +0.0, v16 -5.4 / -6.1, v32 -2.1 / -3.8. Loads inc / cand: d1 1.065, d16 1.006, d32 1.017, v1 0.995, v16 1.018, v32 1.010. The d1 lead is an arm artifact (no d1 code path differs and the null arm, the same file as cand, sits as far under inc); without d1 the geometric mean is about 1.009. The gain sits at 32 / 128 / 256 tokens (mode 47, LATE_ARRIVE) as intended |
| 2026-10-30 | 6372154fcd7a4fa9 (check, M1 diagnostic) | tools/dl_m1_diag.py on the 6377b5a file | M1 determinism: per stress shape, against one eager M1-off call: six eager M1-off calls, two M1-off and two M1-on graph stresses (2000 replays), two eager M1-on calls; attention alone six times. M1 left off for timing | judge 1.995x (1.966-2.025) (the 6377b5a timed path) | 1x8 full (the shape that failed in e2d1d81447cc48d9), 1x8 sliding, 1x1 full and 1x1 sliding: everything bitwise equal, M1 on and off, attention 6/6. 16x1 full: one M1-off eager call 8.5e-5 / 7.9e-5, M1-on graph 1.1e-4 / 8.5e-5 twice, one M1-on eager 1.3e-4. 32x1 full: M1-off eager calls 8.4e-5 to **6.7e-4** (two of six over 1e-4), M1-off graph 8.3e-5 and 1.1e-4, M1-on graph 1.0e-4 and 1.4e-4, attention 4.0e-5 (5/6 equal). The MoE's own mixed-size replay stress dropped sk8w4s at 1.4e-4 against its 1e-4 bound (M1 not involved; sk8w4s unused by the policy). So M1-off code exceeds the 1e-4 replay bound by itself at 32 tokens and in the MoE stress; the 1x8 full failure of e2d1 did not recur in four M1-on trials |
| 2026-10-30 | 3e2aa83358dd4f60 (check, M1 diagnostic 2) | `tools/dl_m1_diag.py --fresh` on the 6377b5a file | As 6372154fcd7a4fa9, plus a fresh-input stress: every replay copies one of five (stream, attn_input) pairs, picked by a device counter, into the captured inputs and is compared on the device with that pair's M1-off eager outputs; per run of 2000 replays the largest hidden / stream relative difference, the replays over 1e-4 and over 1e-2, and replays whose ids differ. Verify 1 full three runs each way, the other five shapes one | judge 1.975x (1.945-2.004) (the 6377b5a timed path) | Fresh inputs, M1 **off** against its own eager outputs: 1x8 full max 1.2e-3 / 1.2e-3 with 459, 217 and 230 of 2000 replays over 1e-4; 1x8 sliding 1.0e-3 (305); 1x1 full 0 (0); 1x1 sliding 3.7e-4 (19); 16x1 full 9.2e-4 (675); 32x1 full 6.6e-4 (1192). M1 **on**, same comparison: 1x8 full 1.2e-3 (307, 288, 295), 1x8 sliding 9.9e-4 (309), 1x1 full 0, 1x1 sliding 3.7e-4 (14), 16x1 full 9.2e-4 (556), 32x1 full 6.6e-4 (823). No replay over 1e-2 in either arm (a read of the previous replay's rows gives order 1), ids never differ. Repeated-input diag again bitwise at 1x1; 16x1 full M1-off eager calls up to 7.8e-4 (four of six). The MoE PDL self-test dropped sk8w4s-er at 1.2e-4 (unused). So the M1-off layer itself exceeds the M1 stress's 1e-4 bound on 11-60% of fresh-input replays at T >= 8, with the same maximum as M1 on; the e2d1d81447cc48d9 failure is explained by the layer's own run-to-run variation, not attributed to the PDL boundary |
| 2026-10-30 | 3fbb3ad027e84c6b (**scored**, submitted by the coordinator) | 2cee446 (file SHA256 b9c23bda..., judge content hash c4b9a3d3..., the c2ba file) | The clean candidate: release + mode 47 + LATE_ARRIVE + o16 / unp32, M1 off | **2.014x** (1.984-2.045) | passed 13/13, failed 0. Board entry before: edd71d3152a946fe (1.982x). MoE 47 at 32 / 128 / 256. Per shape us: d1 77.3 / 82.9, d16 140.8 / 215.7, d32 159.1 / 311.3, v1 121.6 / 130.3, v16 264.0 / 352.5, v32 358.9 / 518.7. Manifest: docs/release-manifest.md. Profiles committed |
| 2026-10-30 | f5ee8632e1f64bf2 (check, bench) | bench at this commit with `--tlend=v1s,v1f,d32s,d32f,d16s` (ABZ: per kernel start, lifetime and exposed time, printed last); cand = the scored file, other = pub, seeds 90001+, order seed 17 | Same-process timelines against the public layer | judge 1.974x (1.945-2.004) on cand; bench: **pub / cand 1.0463**, null 0.9994 | 72 of 72 checks pass. Paired pub - cand us: d1 +1.0 / -1.0, d16 -10.2 / -8.9, d32 -12.2 / -15.9, v1 -10.2 / -10.8, v16 -10.1 / -11.3, v32 -13.8 / -9.0. Exposed time per kernel (how far each kernel moves the latest end; sums to the span) for the profiled median replay, ours - public: v1 full (140.5 / 129.4): attn -> o +2.8 (public o starts 2.6 us before attention ends, exposed 4.7 of its 7.4), attention boundary +0.7, boundary -> pooler/front +3.3 incl. the pre-norm and routing (public pooler exposed 17.4 of 19.6), MoE +2.2, unpooler +1.7, qkv +0.3, attention +0.7, final boundary -0.7; our step has 4.2 us of gaps with no kernel running (0.1-1.2 at eight boundaries), the public one none. v1 sliding (113.3 / 103.3) the same pattern. d16 sliding (140.7 / 129.9): pooler +5.0, o +3.0, rest within 1.5. d32 sliding (184.4 / 166.0): MoE +9.8 (mode 47 at 32 tokens 98.0 exposed against the public 88.2), qkv +1.9, attention +1.7, o +1.7, front +1.5, unpooler +1.0. d32 full (315.9 / 303.3): front +4.8, o +2.4, MoE +3.9, unpooler +2.2, qkv +1.5, attention -2.3. Profiled timelines; the paired ABTIME rows are the decision metric |
| 2026-10-30 | 720079090ee44cc3 (check, bench) | bench at fa7200c; cand = the scored file, other = inc = the same file with `MOE1_HELD = ("main-lean-grp-nohoist",)` (SHA256 84d8507a..., one line differs: mode 47 built, self-tested and benched but never picked, so the arms share every other compiled kernel and the prepare-time policy elsewhere), seeds 110001+, order seed 19, `--tlend=d32s,v16s` | Mode 47's isolated contribution inside the release + P package (P = LATE_ARRIVE + o16 / unp32), actual prepare-time policy | judge 1.987x (1.957-2.016) on cand; bench: cand / held 1.0155, null 1.0009 | 72 of 72 checks pass. Policies: cand MoE mode 47 at 32 / 128 / 256, inc mode 37 there; fronts equal. Paired cand - inc us (null - inc): d1 -4.1 / -5.7 (-4.5 / -5.4), d16 +1.3 / +0.0 (+0.3 / -0.1), d32 -1.3 / -2.6 (-1.5 / -3.0), v1 -0.6 / +0.4 (-1.1 / -1.2), v16 -4.9 / -5.5 (-3.8 / -4.0), v32 -5.7 / -3.6 (-4.8 / -1.0). The inc arm runs about 5 us slow at d1 whatever it holds (here it differs from cand only in the MoE pick at 32+ tokens, which d1 never runs; 50c4c1328ab742f4 showed the same), so d1 and the 1.0155 overall are arm-position artifacts; at d16 and v1, where 47 is never picked, the arms agree within 1.3 us. Mode 47 is worth about 4-5 us at v16 (128 tokens; MoE exposed 162.8 against 170.9 in the d32s / v16s timelines), 2-5 us at v32, and 1.3-2.6 us at d32 (32 tokens; MoE exposed 101.9 against 102.5, inside the arm spread). P's share by difference from 50c4c1328ab742f4 (cross-bench, so only indicative): about 1-2 us at d32 and under 1 us elsewhere. At 32 tokens either MoE mode leaves about 14 us of exposed time against the public kernel's 88.2 (f5ee8632e1f64bf2) |
| 2026-10-30 | 6ce5be0ba29642ea (check) | 8f545ae: `tools/dl_gate_patch.py --m1` on the 6377b5a file put through `tools/dl_moe_inline.py` with LAYER_MODES + 51 (file SHA256 39635763...) | Release + M1 + MoE mode 51 + the fresh-input gate (first use, approved above) + the reader-side `fence.proxy.async.global` after the sk kernel's wait | 1.988x (1.959-2.018) | passed 13/13; **M1 blocked by its gate on the control**: at the first shape (1x1 sliding) the M1-off replays differed from their eager references by 2.7e-3 / 2.5e-3 (hidden / stream), while the M1-on replays were bitwise equal to them over 2000 replays; the failure reason was cut at 120 characters in SUMMARY. At one token M1 changes no launch (o is cgemv, the front fp32), so the two graphs are the same code: an arm-order or first-graph effect, or a real fresh-input divergence in the M1-off layer, not an M1 effect. M1 timing did not run (policy {}). MoE: no variant dropped (every variant in the bench); route PDL gate passed on all 11 variants, candidate / control relative error 2e-5 to 1e-4, element error 2e-3 to 6e-3, order-1 0; **mode 51 at 128 / 256 tokens bitwise in both arms (0 / 0)**, as the MoE agent predicted (no sum-of-squares atomics from 64 tokens); at 32 tokens 51 shows 5e-5 like 47. MoE picks sk8w3s 1 / 8 / 16, 47 at 32 (87.9 against 51's 88.5), 51 at 128 / 256 (171.5 / 217.7 against 47's 175.0 / 222.4). The MoE stress gate lines fell out of the 16 KB tail (the six-shape timeline came last). Per shape us: d1 77.5 / 83.5, d16 146.0 / 222.5, d32 168.0 / 326.3, v1 122.1 / 135.9, v16 257.3 / 345.8, v32 335.1 / 501.9. Profiles committed |
| 2026-10-30 | 60cf58b69b75491e (check, M1 stage diagnostic) | `tools/dl_m1_stage_diag.py` on the 6ce5 file | Per T <= 32 shape, kernel_fn's stages (qkv, attention, o, FFN input, MoE input, routing weights, MoE output, unpooler, next input, stream) on fresh inputs (five pairs), arms M1 off, M1 off again, M1 on (1000 graph replays each) and 200 eager M1-off calls, all against eager M1-off references; never raises; M1 timing then ran with the policy left empty | judge 1.965x (1.936-1.995) (M1 off) | **No arm differs from another**: per stage the largest relative errors are equal across off / off2 / on / eager (qkv 0 always; attention 1e-5 to 2e-4; MoE input up to 1e-3; MoE output up to 4e-3; next input up to 3e-3), and the counts of order-1 elements (>= 1e-2 of |ref| + row RMS) are alike: 1x1s 69 / 69 / 75 per 1000 and 20 per 200 eager; 1x8s 205 / 197 / 307 and 41 / 200; 1x8f 25 / 31 / 31 and 6 / 200; 16x1s 69 / 76 / 71 and 15 / 200 with ids flips 14 / 16 / 26 and 7 / 200 eager; 16x1f 157 / 167 / 153 and 25 / 200; 32x1s ~185 and 39 / 200; 32x1f ~440 and 80 / 200. 1x1 full: no order-1 (unpooler onward 6e-4). The order-1 elements first appear at the MoE output (or the routing weights at 16x1s); relative L2 never exceeds 5e-3 (a stale read gives 0.3-1.4). So gate clause (b) as implemented fails on the M1-off layer itself, eagerly too; no stale-read signature; M1 changes nothing measurable in the error. **M1 timing** (same build, M1 off vs on, interleaved medians, cold L2): chain attention -> o -> boundary sliding / full saves 2.69 / 0.70 us at T=8, 1.89 / 2.46 at 16, 2.43 / 3.33 at 32; boundary -> front 1.47 / 1.50 at 8, 0.03 / 0.38 at 16, 0.38 / 0.35 at 32; whole step off -> o+front: T=8 105.9 -> 100.5 sliding, 105.9 -> 102.6 full; T=16 153.4 -> 150.0, 229.1 -> 226.6; T=32 217.7 -> 214.7, 360.1 -> 357.1; o+front best in every bucket; stop rule not met (5.3 us at 8 tokens sliding). MoE gate: no order-1, element error <= 6e-3; sk8w4s and main-d2 dropped by clause (c) because the control made 400 calls against 2000 replays (fixed: equal counts); mode 51 bitwise at 128 / 256 in both arms. `_route_variant_check` (1e-4, unchanged) dropped skf-r at T=1 (1.3e-4). Per shape us: d1 78.1 / 84.9, d16 144.9 / 221.2, d32 177.8 / 329.1, v1 128.2 / 137.2, v16 258.1 / 349.0, v32 348.6 / 507.5. Profiles committed |
| 2026-10-30 | fdf75ba5a8124544 (check, bench) | `tools/dl_make_bench_pub.py --cand-strip=1`, cand = `tools/dl_m1_bench_build.py` on the gated M1 + 51 file (M1 measurement build, not a candidate: gate report-only, M1 on as its tuner picks, MoE prefetch toggle), other = inc = the release (SHA256 b9c23bda...), seeds 130001+, order seed 23, 8 rounds | Same process: M1 build + 51 against the release, null = cand again | judge 2.007x (1.977-2.038) on cand; bench: **cand / release 1.0040**, null 0.9984 | 72 of 72 checks pass. Policies (ABARMS variant indices): both arms mode 37 at 32 tokens; at 128 / 256 inc mode 47, cand mode 51; fronts equal except 16 tokens (inc skf, cand skf-r); M1 policy o+front at 8 / 16 / 32. Paired cand - inc us (null - inc): d1 +0.2 / -0.3 (+0.0 / -0.1), d16 +0.6 / +3.0 (+1.0 / +3.0), d32 -0.1 / -0.2 (+0.7 / +0.8), v1 -1.3 / +0.2 (-1.2 / +0.3), v16 -4.7 / -3.2 (-4.4 / -3.4), v32 +1.2 / -4.6 (+1.8 / -4.6). So **M1 nets about zero against the release at T <= 32** although inside the build it saves 5.1 us at 8 tokens sliding (whole step off 91.1 -> o+front 86.0; full 111.5 -> 108.4; 16: 151.6 -> 149.0, 231.3 -> 228.5; 32: 206.3 -> 200.9, 363.8 -> 359.1; chains attention -> o -> boundary 2.2 / 1.8 / 2.1 us sliding, boundary -> front 1.8 / 0.3 / 1.9): the SK_PDL build's issue order costs plain sk launches about as much. Mode 51 accounts for v16 -3 to -5 us and v32 full -4.6 (M1 and the SK_PDL build touch no T >= 128 launch). M1 gate (report-only, 1000 replays per arm): 1x1 s / f bitwise in both arms; 1x8 s cand rel 1.1e-3 / ctrl 2.8e-4, order-1 17 / 10; 1x8 f 1.2e-3 / 3.5e-4, order-1 2 / 0; 16 and 32 tokens equal between arms (order-1 85-639 per 1000 in both). MoE gate with equal controls: all 11 variants pass, no order-1, 51 bitwise at 128 / 256. MoE prefetch probe at d32 (prefetch off minus on, null in brackets): sliding +0.22 (-0.06) with 3 active local experts, +0.00 (+0.26) with 9; full -0.16 (-0.32) with 2, -0.48 (-0.67) with 8: no effect; the step itself moves 158.3 -> 190.6 us between the two sliding inputs. Prepare 78 s. Profiles committed |
| 2026-10-30 | a438b82ff2c141c2 (check) | `tools/dl_gate_patch.py` on the release through `tools/dl_moe_inline.py` (LAYER_MODES + 51); file `muse_decode_layer/cand/layer_g51.py`, SHA256 c8a843cf... | **Candidate**: release + MoE mode 51 + the fresh-input gate on the MoE replay self-tests; no M1, no SK_PDL | **2.022x** (1.991-2.052) | passed 13/13, failed 0; every MoE stress and route PDL gate ok, order-1 0 everywhere, element error <= 5e-3, mode 51 bitwise at 128 / 256. MoE picks sk8w3s 1 / 8 / 16, 51 at 32 / 128 / 256 (98.4 / 179.7 / 229.0 against 47's 98.7 / 184.8 / 232.4). Per shape us: d1 75.9 / 83.4, d16 143.5 / 215.3, d32 162.5 / 322.8, v1 118.8 / 128.7, v16 273.6 / 358.9, v32 348.3 / 503.8. Against the release only `_MOE1_SRC`, `_MOE1_VARIANTS` (51 added; SASS of 51 as the port guide: instr 17268, FENCE.S 22, FENCE.G 11, MEMBAR 43, SYNCS 202, BAR 16, ATOMG 148, REDG 9, LDL 0, STL 0, 96 registers, no spills; 47 unchanged) and the gate code (`_moe1_stress`, `_moe1_pdl_self_test`, SUMMARY gate lines) differ. Prepare 146 s. Profiles committed |
| 2026-10-30 | 29fd004fd58e40ef (check, bench) | `tools/dl_make_bench_pub.py --cand-strip=1` at d9b7d67; cand = `muse_decode_layer/cand/layer_g51.py` (SHA256 dacd4e0d..., sent comment-stripped 7e046232...), other = inc = the release (SHA256 b9c23bda...), seeds 140001+, order seed 29, 8 rounds | **Mode 51 against the clean mode-47 release in isolation** (the candidate also carries the decided gate, which changes only prepare-time checks: no timing effect expected), cand / null / inc | judge 2.007x (1.977-2.037) on cand; bench: **cand / release 1.0084**, null 0.9997 | 72 of 72 checks pass. Policies: MoE equal at 1 / 8 / 16 (sk8w3s) and 32 (mode 47 both), 128 / 256 inc 47, cand 51; fronts differ at 16 only (inc skf-r, cand skf; prepare-time picks within noise). Paired cand - inc us (null - inc): d1 -0.3 / -1.7 (-0.1 / -2.5), d16 +0.3 / +0.0 (-0.0 / +0.0), d32 -1.2 / -1.1 (-1.1 / -0.8), v1 -0.6 / -0.2 (+0.1 / -0.8), **v16 -5.1 / -5.3 (-3.7 / -5.2), v32 -6.5 / -1.1 (-4.4 / +0.3)**. Loads inc / cand: d1 1.0099, d16 0.9997, d32 1.0078, v1 1.0023, v16 1.0174, v32 1.0133. No shape regresses beyond the null spread; the gain sits at 128 / 256 tokens, where only the MoE mode differs. Gate in the candidate's prepare: 11 MoE variants pass, no order-1, element error <= 7e-3, 51 bitwise at 128 / 256; route gates 8 of 8 ok (skf-r at 1 / 8 / 16 / 32, concat-r at 16 / 32 / 128 / 256; 1, 128 and 256 bitwise); negative controls on real outputs: every injection rejected in all 19 gates (tile, 1%, 0.1%, row, one element), every uninjected observation passes; the single element shows relative L2 2e-4 to 7e-3 (under the 1e-2 order-1 bound, caught only by the element check, element error 0.05-0.6); manifest: rejected none, failures 0 in every class. Zero reference rows (tokens without a local route) make the element check reject any nonzero value there (element errors of 1e11 in the injected route controls). Prepare 78 s. Profiles committed |
| 2026-10-30 | 724aa0dd6ed140ca (check, bench) | four-arm `tools/dl_make_bench_pub.py` (a877401), bench4_aa.py (SHA256 9473029f...): inc = cand = the release (b9c23bda...), `--dup=own`, seeds 180001+, order seed 43, 8 rounds | **A/A calibration**: the release against itself, four arms (inc, inc2, cand, cand2), each copy with its own prepare | judge 2.009x (1.979-2.039); bench pooled inc / cand **1.0045**, nulls inc2 / inc 0.9965, cand2 / cand 1.0030; ABDECIDE not resolved | The harness keeps a 0.45% offset in favour of cand, and it sits at d1 alone: d1 pooled 1.0262 (inc 3-6 us slow at d1 sliding and full against all three other copies, null inc2 / inc 0.9554); every other load pooled 0.994-1.004. The same inc-at-d1 penalty shows in 2dc566298be1498f (d1 null 0.939) and 40b9b3f795f84073 (0.949), so it belongs to the inc arm (the exec'd, comment-stripped module), not to any candidate. ABTWIN within 0.3%. ABPOL DIFF between copies of the same file: w4 configs (1 / 16 / 256 tokens), front at 8 (skf against skf-r); MoE, abnd, skp, wide, m1 and stages same. ABEND 72 s. Profiles committed |
| 2026-10-30 | 2dc566298be1498f (check, bench) | bench4_g51.py (SHA256 e95f3218...): cand = `muse_decode_layer/cand/layer_g51.py` (dacd4e0d..., stripped), inc = the release, seeds 160001+, order seed 37 | Mode 51 against the release, four arms | judge 2.000x (1.970-2.030); pooled **1.0057**, nulls 0.9898 / 0.9972 (larger 1.03%); not resolved | Per load pooled: d1 1.0203 (inc-arm penalty again, null 0.939), d16 1.0039, d32 0.9990, v1 0.9924 (same kernels, front and abnd picks differ by copy; all four crosses 0.991-0.994), **v16 1.0080, v32 1.0107** (mode 51's shapes; crosses 1.003-1.013 and 1.006-1.015). Net of d1, the gain sits where mode 51 runs, but the run's nulls do not resolve it; not shipped on this evidence (user decision). Policy DIFF between copies: front 8 / 16, abnd 1 (c1x1024 in cand) and 16, w4. ABEND 155 s. Profiles committed |
| 2026-10-30 | 40b9b3f795f84073 (check, bench) | bench4_m1rt.py (SHA256 824e1e45...): cand = the M1 runtime measurement build (`dl_m1_bench_build.py --no-pf-probe` on the gated M1 + **mode 51** file, SHA256 d57b5e38..., gate report-only), inc = the release, seeds 170001+, order seed 41 | M1 (runtime issue order) + mode 51 against the release, four arms | judge 1.992x (1.962-2.022); pooled **1.0162**, crosses 1.0111-1.0214, nulls 0.9928 / 0.9970 (larger 0.72%); **ABDECIDE cand faster** | M1 picks {8: o+front, 16: o, 32: o} in both cand copies; M1 gate ok at all 8 shapes (1000 replays, order-1 0, ids flips 11 against 8 at 32x1 full). The candidate carries mode 51, so the pooled gain is M1 + 51 + the d1 inc-arm offset. Per load pooled (A/A, g51): d1 1.0320 (1.0262, 1.0203: arm offset), d16 1.0074 (1.0013, 1.0039), d32 1.0065 (1.0014, 0.9990), **v1 1.0166** (1.0044, 0.9924), v16 1.0257 (0.9995, 1.0080), v32 1.0096 (0.9942, 1.0107). M1 acts only at T <= 32 (v1, d16, d32; d1's o is cgemv and its front fp32, so no launch changes). Net of the A/A per load, M1's share is about (ln 1.0122 + ln 1.0061 + ln 1.0051) / 6 = **+0.4%** of the score, mode 51's (v16, v32) about +0.7%, d1 +0.1%: total +1.2%, matching pooled / A/A. v1 sliding paired -2.1 to -2.9 us, as M1's own tuner measured (whole step at 8 tokens sliding saves 3.2 us). Estimates from one run per file; the v16 load of the same mode 51 code read 1.0080 in 2dc5 and 1.0257 here, so a per-load reading moves by about 1.5% between runs. Policy DIFF between copies: MoE at 32 (inc 47, inc2 37: d32s +3.1 us), front 8 / 16, abnd 16 / 256, w4. ABEND 166 s. Profiles committed |
| 2026-10-30 | (build, no check) | `tools/dl_gate_patch.py --m1 --m1-runtime` on 6377b5a's file plus one header line: `muse_decode_layer/cand/layer_m1rt.py` | **Release candidate**: release + M1 (runtime issue order, reader-side fences, ONE_TRIG) with the fresh-input gate enforced; mode 47, no mode 51 | | See the release manifest's candidate section for the build, the diff against the release, the SASS and the offline negative-control test. Not yet checked or scored |
| 2026-10-30 | b68bdcd7a1ba464c (**scored**, submitted by the coordinator) | `muse_decode_layer/cand/layer_m1rt.py` (SHA256 d6958449..., cmp-identical to dd1777e's file) | Release candidate layer_m1rt | **2.013x** (1.983-2.043) | passed 13/13; prepare 143.1 s (setup record). Per load speedups (sliding / full): d1 2.243 / 2.159, d16 1.834 / 1.969, d32 1.571 / 1.967, v1 2.233 / 2.140, v16 1.966 / 1.976, v32 2.114 / 2.124. Against the board entry's 2.014x this is within one run's noise, as the +0.4% estimate predicted. **Whether M1 was on in this run is unknown**: the 16 KB log tail holds only timeline lines (M1 adds v1s and v1f to the timeline, which prints after the SUMMARY), so the M1 gate verdict, policy and every other SUMMARY line were lost. layer_m1rt51 prints the timeline first. Profiles committed |
| 2026-10-30 | (build, no check) | `tools/dl_moe_inline.py` (LAYER_MODES + 51) on layer_m1rt with the MoE standalone at 6bbb252: `muse_decode_layer/cand/layer_m1rt51.py` | Candidate layer_m1rt51: layer_m1rt + mode 51, the combination 40b9b3f795f84073 timed | | Release manifest, layer_m1rt51 section. Four-arm benches for after the reset prepared in the scratchpad (`dl6/bench5_m1rt.py`, `dl6/bench5_m1rt51.py`, inc = the release, seeds 200001+ / 190001+, order seeds 53 / 47, 8 rounds) |
| 2026-10-30 | d6c30f88c617438b (**scored**, submitted by the coordinator) | `muse_decode_layer/cand/layer_m1rt51.py` at 7cdb755 (SHA256 65264f31...) | Candidate layer_m1rt51: layer_m1rt + MoE mode 51, M1 gate enforced | **2.013x** (1.983-2.043) | passed 13/13; prepare 154.8 s. **M1 off for the whole run**: its gate failed at the first shape, `1x1s FAIL cand rel 3.2e-04/1.9e-04 el 2.2e-03/1.3e-03 >1e-4 1/2000 o1 0 \| ctrl rel 0.0e+00` (only the largest-error envelope failed: limit max(1e-4, 2 x 0)); M1's negative controls at 1x1s all rejected, uninjected passed; selftest bitwise on 14 launches. At one token M1 changes no launch (o is cgemv, the front fp32), so this is the mirror of 6ce5be0ba29642ea and not an M1 effect: analysis archive/docs/steering/2026-10-30-m1-1x1s-gate-failure.md. Two sensitivity failures: route concat-r@32 and skf-r@32 `row:pass(1e-06,8e-05)` (the injected row was a zero row in both inputs: fixed in `tools/dl_gate_patch.py`, below). MoE: 11 variants checked, none rejected, every MoE stress and PDL gate ok, order-1 0, mode 51 bitwise at 128 / 256; picks sk8w3s 1 / 8 / 16, 51 at 32 / 128 / 256 (96.0 / 171.5 / 221.5 against 47's 96.0 / 178.3 / 227.0); fronts fp32 / skf / skf-r / skf-r / concat-r / concat-r; abnd c8x256 at 1 and 8, sglang above. Per shape us: d1 76.7 / 83.1, d16 141.1 / 224.1, d32 173.4 / 327.0, v1 116.0 / 127.3, v16 268.6 / 351.6, v32 345.2 / 513.6. As a measurement this is release + mode 51 in the SK_PDL / ONE_TRIG build with M1 off, at the board entry's 2.014x within noise. Profiles committed |
| 2026-10-30 | (tool change, offline test) | `tools/dl_gate_patch.py`, `tools/dl_gate_negctl_test.py` at this commit | Negative controls always perturb: the row control injects the row nearest rows // 2 whose two references differ by at least 1/8 of the row maximum; the unrestricted element draw skips elements where the references are equal; a bank entry equal to the fresh one is replaced by one that differs (all equal: reported, no miss). Gate bounds unchanged | | `uv run --with torch python tools/dl_gate_negctl_test.py tools 200` (CPU, width 8192, noise 3e-3, 200 trials per setting), new sparse model with rows zero at probability 0.34 per entry (a token without a local route): old tool row misses 22 / 13 / 21 at 8 / 16 / 32 rows (unfloored element setting) and 29 / 19 / 20 (floored), and every kind 22-28 at one row; new tool row and uninjected never missed in any setting, the unrestricted element 2-8 per 200 from near-equal values as before, floored element 0. Dense model unchanged (row 0, element 2-8 unfloored, 0 floored) |
| 2026-10-30 | (build, no check) | the release through `tools/dl_gate_patch.py` at this commit, then layer_g51's MoE regions (`_MOE1_SRC`, `_MOE1_VARIANTS`, `_MOE1_MAIN_MIN_M`, repack region: the dl_moe_inline.py output with LAYER_MODES + 51 from 6bbb252), new header line: `muse_decode_layer/cand/layer_g51b.py`, SHA256 acd2385c... | Candidate layer_g51b: release + mode 51 + the current gate, no M1 | | Byte-identical on a second build. Every `_*SRC` region equals layer_g51's except `_W4_SRC`, which equals layer_m1rt51's (W4 GEMM2 wait; SASS checked for layer_m1rt, compiled in b68bdcd7a1ba464c and d6c30f88c617438b). Python differs from layer_g51 by the negative-control fix and the SUMMARY-after-timeline order. Four-arm bench against the release prepared for after the reset (steering note above, section 3) |
| 2026-10-30 | (build, no check) | `tools/dl_m1_1x1_repro.py` on `muse_decode_layer/cand/layer_m1rt51.py`: scratchpad `m1repro/m1_1x1_repro.py`, SHA256 16665ed7... | M1 1x1 reproduction diagnostic (not a candidate): census of M1 PDL launches and kernel sequences per arm, per-replay stage records (relative error, differing elements, matches with the previous input's reference) for off / on / off2 / on2 / off-kfn / on-kfn / off-same / on-same and eager, at 1x1 sliding (gate seeds) and 1x8 sliding; M1 off in the timed path | | Byte-identical on a second build; compiles; the record and summary code passed a CPU smoke test with injected stale and ULP events. First check after the reset |
| 2026-10-30 | (tool change, build, offline test) | `tools/dl_gate_patch.py` (census skip), `tools/dl_m1_census_patch.py` on `muse_decode_layer/cand/layer_m1rt51.py`: `muse_decode_layer/cand/layer_m1rt51c.py`, first SHA256 525b85b0..., rebuilt with `--gate-src` as 1eda8ec9... | Candidate layer_m1rt51c: layer_m1rt51 plus the user-decided census skip in the M1 gate and (rebuild, coordinator's request) the negative-control fix, nothing else | | Diff against layer_m1rt51: line 2, the launch counter before the two `ext.sk_fwd` / `ext.skf_fwd` calls, `_M1_PDL_LAUNCHES` and `_m1_census`, and the skip in `_m1_stress`; no device code changes, so its SASS is layer_m1rt51's. Byte-identical on a second build; `dl_gate_patch.py --m1 --m1-runtime` on 6377b5a's file also applies the census cleanly. `uv run --with torch python tools/dl_m1_gate_census_test.py tools` (CPU, stub layer with M1 launches at 8 tokens and above, 60 replays per arm): clean run skips 1x1s / 1x1f and judges the other six ok; a candidate-only one-replay difference at 1 token is skipped, at 8 tokens raises `1x8s FAIL`; with the census broken it is judged and raises at 1x1s, and the clean run is judged ok at all eight shapes with "census failed". Rebuild: the diff against the first build is the negative-control code only, the file contains `dl_gate_patch.GATE_SRC` verbatim, and the offline negctl test passes on it; a route negctl miss in the old code was logged only (no variant dropped). Four-arm bench against the release prepared for after the reset (scratchpad `m1repro/bench7_m1rt51c.py`, regenerated from the rebuild in place, seeds 220001+, order seed 61, SHA256 aef5fd56...) |
| 2026-10-30 | (build, offline test) | `tools/dl_stable_tuner_patch.py` on the release: `muse_decode_layer/cand/layer_stable.py`, SHA256 752123dd... | Release + stable prepare-time picks (archive/docs/plan-2026-10-30-tuner-stability.md, design 1, at the coordinator's request after the researcher's 23:35 review): MoE, front and attention-boundary tuners time in 9 interleaved rotated rounds and keep the receipt pick of 3fbb3ad027e84c6b unless another saves more than max(0.3 us, 2 x MAD of the preferred's rounds) in median paired difference and wins at least 8 of 9 rounds; MoE rounds run on the variants within max(3 us, 3%) of the first-pass fastest plus the preferred one (4 routings, 10 replays per graph per round); W4 unchanged; one SUMMARY `stable` line per tuner and token count. Not a candidate | | Byte-identical on a second build; every `_*SRC` region equals the release's (no device code change); py_compile and pyflakes clean. `python3 tools/dl_stable_pick_test.py tools`: a true tie at 0.4 us noise leaves the preferred pick in 1991 of 2000 draws (lowest median of the same rounds: 1016 of 2000 off it), a 2 us win at 0.3 us noise switches 500 of 500, a 3 us median win with 3 of 9 rounds against stays, a 1.5 us win under a 4 us (2 x MAD) margin stays, an absent preferred pick takes the lowest median. A/A four-arm bench of this file against itself prepared for after the reset (scratchpad `m1repro/bench8_stable_aa.py`, seeds 230001+, order seed 67): pass if ABPOL reads same everywhere but W4 and the nulls stay within about 0.4% outside d1 (724aa0dd6ed140ca: three tuners DIFF, null 0.35%); stop if ABPOL still shows DIFF in front, abnd or MoE, or prepare grows by more than 40 s |
| 2026-10-30 | (build, no check) | `tools/dl_one_inline.py muse_attention_decode/attn_one.cu OUT` (the inliner now takes the target file) and `tools/dl_one_port_patch.py OUT HEADER` on a copy of the release (b9c23bda...): `muse_decode_layer/cand/layer_hp.py`, SHA256 0efd553c60815db9c2f6650ddbc3a9e6762d05fa81ff56d802cd9cc12cb2ee6f | **Candidate layer_hp**: release + the standalone attention of rounds 2 and 3 (`ONE_HMRG` 1, `ONE_PRQ` 31, attn_one.cu at 1faadbc) + the workspace placement pick (`_ONE_WS_PICK` 3); no M1, no mode 51 | | Byte-identical on a second build. Against the release only `_ONE_SRC` differs among the `_*_SRC` regions; `_ONE_HOST_SRC` and `Params` unchanged (no new fields: PRQ's copy bit reads `off32`, which the wrapper already sets, and `lp`). With `-DONE_HMRG=0 -DONE_PRQ=0` the attention extension's SASS is text-identical to the release's (namespace hashes normalized), so the inert additions (XPIPE, DETM, T9, TRIG, all 0) change nothing. SASS (CUDA 12.8.1, `tools/dl_moe_sass_counts.sh FILE one`), release -> candidate: q_len 1 modes 1 / 3 / 11 / 13 BAR +2 and DEPBAR +1 each (HMRG's two CTA barriers and `cp_wait<0>`), REDG, ATOMG, MEMBAR equal; q_len 8 modes 1 / 3 / 11 / 15 WARPGROUP.ARRIVE 2 -> 3 (PRQ's split P.V), every fence, barrier, mbarrier, atomic and GMMA count equal; 49, 19 and 51 unchanged. Candidate against the standalone attn_one.cu (`tools/attn_one_sass.sh`, same 11 instances): every counted opcode equal except ACQBULK (the inliner's `griddepcontrol.wait`, 1 per per-request instance, 2 per flat) and instruction totals within 1-7. Registers as the standalone (1: 108, 3: 109, 11: 115, 13: 72, 49: 128; q_len 8 1: 111, 3: 111, 11: 112, 15: 82, 19: 112, 51: 80), no spills, no C75xx warning (`tools/dl_compile_check.sh FILE _ONE_SRC`, 44 s; the 11 #114-D warnings are the release's too). Python: the pick (`_one_ws_alloc` / `_one_ws_pick`, the standalone's, run in `_one_setup` before any attention self-test; an exception or a dirty workspace after it falls back to a fresh zeroed allocation, the release's state; SUMMARY line `attn ws pick`); four more stresses with plan assertions, so every instance HMRG or PRQ changes is stressed: 16x1 sliding (1, 1), 1x1 full page 64 (8, 3), 16x8 sliding (1, 3) and 16x8 sliding with mode 1 forced (no tested shape plans q_len 8 mode 1 on an H200; `_attn_one` takes a fixed plan for this); modes 11, 13 and 15 were already asserted. Plans checked offline with the candidate's `_one_plan` at H200 residency. pyflakes clean. No GPU run yet. Four-arm bench against the release prepared (`tools/dl_make_bench_pub.py ... --cand-strip=1 --rounds=8 --seed0=230001 --order-seed=67 --dup=own`, 885,566 bytes, SHA256 a6fbe13f..., ABPOL now prints each arm's workspace pick) |
| 2026-10-31 | 6001272c2b1c4852 | scratchpad `dlp/bench8_hp.py` (inc = the release, cand = `muse_decode_layer/cand/layer_hp.py` 0efd553c...) | Four-arm bench: layer_hp (attention rounds 2 and 3 + workspace pick) against the release, 8 rounds, seeds 230001+, order seed 67, `--dup=own` | 1.981x (1.951-2.010) as a check | **Not resolved, no layer gain**: pooled 1.0003, cross 0.9964-1.0042, nulls inc2/inc 0.9987 and cand2/cand 1.0064 (the larger 0.64%). Per load pooled: d1 0.9857 (cand2 alone slow: 95.3 against 91.9-92.4, cand2/cand 1.0379), d16 1.0101, d32 1.0055, v1 1.0028, v16 0.9964, v32 1.0015. The d16 reading (all four crosses 1.0074-1.0127, nulls under 0.4%) is the only per-load signal; the standalone's v1 / v16 / v32 savings do not show. Every arm 24 of 24 checks PASS; ABPOL: MoE, skp, wide, stages same; front, abnd, W4 DIFF (the release tuners); wspick candidate 1 of 3 in both cand copies. ABEND 72 s. Profiles committed. The user removed the workspace pick on 2026-10-31, so this file is not carried forward |
| 2026-10-31 | 9d00f194ed29471b | scratchpad `m1repro/m1_1x1_repro.py` (16665ed7..., from layer_m1rt51 by `tools/dl_m1_1x1_repro.py`) | M1 1x1 reproduction diagnostic | 2.005x (1.975-2.035) as a check; M1 off in the timed path | `SUMMARY m1 ok=0 policy {}`; selftest bitwise on 14 launches. Kernel sequences per replay: 1x1s off 11 / on 11 and 1x8s off 9 / on 9, **identical sequences** at both shapes (the policy is empty in this build, so "on" launched as "off" at 1x8 as well). The per-replay records then failed at both shapes with `AcceleratorError: CUDA error: operation failed due to a previous error during capture` (`cudaErrorStreamCaptureInvalidated`): something in the record path makes a call that is illegal while a stream is captured, so no stage records were produced. Not fixed (optional per the coordinator); the census skip already covers the 1x1 shapes. Profiles committed |
| 2026-10-31 | 64691c2e128247f4 | scratchpad `m1repro/bench6_g51b.py` (inc = the release, cand = `muse_decode_layer/cand/layer_g51b.py` acd2385c...) | Four-arm bench: release + MoE mode 51 + the current gate, no M1, against the release | 2.020x (1.990-2.051) as a check | **Not resolved**: pooled 0.9991, cross 0.9952-1.0030, nulls 0.9993 and 1.0071. Per load: v16 1.0164 (all four crosses 1.0132-1.0196, nulls under 0.6%) and v32 1.0041 on mode 51's shapes, d32 0.9920 (mode 51 at 32 tokens slower by 1-2 us on all four crosses), d1 0.9773 (cand2 alone slow: 91.5 against 87.2-87.6, cand2/cand 1.0439), d16 0.9997, v1 1.0052. ABPOL: MoE DIFF only between files (inc 47, cand 51 at 32 / 128 / 256, each file's copies equal); front, abnd, W4 DIFF within files. ABEND 151 s. Profiles committed |
| 2026-10-31 | 4d4ea5db46db4a10 | scratchpad `m1repro/bench7_m1rt51c.py` (aef5fd56...; inc = the release, cand = `muse_decode_layer/cand/layer_m1rt51c.py` 1eda8ec9...) | Four-arm bench: layer_m1rt51c (M1 runtime + mode 51 + census skip) against the release | 1.994x (1.964-2.024) as a check | **M1 on** in both cand copies: cand `{8: 'o+front', 16: 'o', 32: 'o+front'}`, cand2 `{8: 'o+front', 16: 'o', 32: 'o'}`; every M1 gate and negative control passed (24 of 24 checks per arm; manifest: 11 MoE variants, none rejected, route gates 8/8, two sensitivity reports for `main-d2 T=256` elem, report-only). Pooled **1.0137**, all four crosses 1.0072-1.0202, every load's pooled ratio above 1: v1 1.0155, d32 1.0154, d16 1.0111, v16 1.0089, v32 1.0071 (d1 1.0242 is the inc-arm effect: inc 95.5 against inc2 90.9 us). **Not resolved** only because the incumbent's own null was 1.16% (inc2/inc 0.9884 overall, 0.9525 at d1, 0.9788 at d32); cand2/cand 1.0012. Release tuners flipped within the incumbent (front 32 concat-r / skf-r, abnd 8 / 16 / 256). ABEND 173 s (cand prepares 67 s each). The M1 tuner's report lines fell out of the 16 KB tail. Profiles committed |
| 2026-10-31 | e582fde557114062 | scratchpad `m1repro/bench8_stable_aa.py` (both arms `muse_decode_layer/cand/layer_stable.py` 752123dd...) | Stable-tuner A/A: release + stable prepare-time picks against itself | 2.001x (1.971-2.031) as a check | **The stable tuners work**: ABPOL same for MoE, front, abnd, skp, wide, M1 and stages (only W4 DIFF, which is off the scored path); nulls inc2/inc 0.9988 and cand2/cand 0.9997 (0.12% and 0.03%, against 1.16% / 0.12% in 4d4ea5db46db4a10 and 0.35% in the old A/A 724aa0dd6ed140ca); pooled 0.9989, cross 0.9981-0.9996, not resolved as an A/A should be. Per-load nulls at most 0.56% (d1 inc2/inc 0.9944). Prepare per copy 18-21 s (ABEND 81 s for four prepares), so the extra rounds cost little. Pass by the plan's stop rule (no DIFF in front, abnd or MoE; prepare did not grow by 40 s). Profiles committed |
| 2026-10-31 | (tool change, build, offline tests) | `tools/dl_stable_tuner_patch.py` (`--pref`, `--m1`, keeps a gated file's fresh-input route check; with no option the output is byte-identical to the earlier tool's, checked by rebuilding layer_stable) on the release and on layer_m1rt51c: `muse_decode_layer/cand/release_s.py` a45fae28..., `muse_decode_layer/cand/layer_m1rt51s.py` f1235356... | **Candidates release_s and layer_m1rt51s**: release_s = the release + stable picks (release preference table; same code as layer_stable). layer_m1rt51s = layer_m1rt51c + stable picks with MoE mode 51 preferred at 32 / 128 / 256 and the M1 tuner on the stable rule (preferred o+front / o / o+front at 8 / 16 / 32) | | Both byte-identical on a second build; every `_*SRC` region equal to its input's (SASS unchanged: layer_m1rt51c's for the candidate, the release's for release_s); changed top-level functions only `_moe1_tune`, `_front_tune`, `_abnd_tune`, `_m1_tune`, `_selection_summary`, plus `STABLE_*`, `_stable_pick`, `_m1_rounds`. py_compile and pyflakes clean. `python3 tools/dl_stable_pick_test.py tools FILE` (now also checks the file's own `_stable_pick`, the M1 ties and `_m1_rounds`): all cases pass on both. `uv run --with torch python tools/dl_m1_gate_census_test.py tools muse_decode_layer/cand/layer_m1rt51s.py` (now takes the gate code from the file): all five cases as expected; the file's M1 gate region equals `dl_gate_patch.M1_SRC` except `_m1_tune` |
| 2026-10-31 | 3aee22ba4a404c8a | scratchpad `m1s/bench9_m1rt51s.py` (inc = release_s, cand = layer_m1rt51s; 903,475 bytes) | Four-arm bench, both arms with stable picks: layer_m1rt51s against release_s, 8 rounds, seeds 250001+, order seed 73, `--dup=own` | 2.008x (1.978-2.038) as a check | **cand faster**: pooled 1.0052 (+0.52%), cross 1.0036-1.0068, nulls inc2/inc 1.0008 and cand2/cand 1.0024 (larger 0.24%). Per load pooled: v16 1.0163 (paired -3.9 to -5.8 us on both layers, all four crosses), v1 1.0149 (v1s -1.9 to -2.6 us), d16 1.0069, d32 1.0052, v32 0.9993, d1 0.9887 (cand2 alone slower at d1s: 90.1 against 87.6-88.0). ABPOL: front, abnd, wide, stages same in all four copies; MoE same within each file (51 at 32 / 128 / 256 in cand); M1 same in both cand copies: `{8: 'o+front', 16: 'o', 32: 'o'}` (32 left the preferred o+front for o by the stable rule in both); W4 DIFF (off the scored path). 24 of 24 checks PASS per arm. Prepare: cand 62-64 s, inc 18-19 s; ABEND 167 s. Profiles committed |
| 2026-10-31 | (tool, build, offline tests) | `tools/dl_mb_unp_patch.py` on a stable build of layer_m1rt51c with M1 preferring o+front+unp at 8 / 16 / 32: `muse_decode_layer/cand/layer_mb1s.py`, SHA256 16feec526461b778e1d286b0b77b8d91cd945fe4515a8a9463455684d46fac6b | M-B slice 1 (archive/docs/plan-2026-10-31-m-b-pdl-chain.md): M1 mode o+front+unp launches the unpooler sk pick as a PDL secondary of the MoE; the gate replays that mode; the self-test adds the unpooler picks | | Byte-identical on a second build; every `_*SRC` region equal to layer_m1rt51s's (Python only); pyflakes clean; the pick test and the census test (gate code from the file) pass |
| 2026-10-31 | c8d1e57261dd4901 | scratchpad `m1s/bench10_mb1s.py` (inc = release_s, cand = layer_mb1s; 904,512 bytes) | Four-arm bench: M-B slice 1 against release_s, 8 rounds, seeds 260001+, order seed 79, `--dup=own` | 1.984x (1.954-2.013) as a check | **The slice was not used**: both cand copies left the preferred o+front+unp at every bucket and picked `{8: 'o+front', 16: 'o', 32: 'o'}`, the same policy as layer_m1rt51s in 3aee22ba4a404c8a. The stable rule leaves the preferred mode only when another mode saves more than max(0.3 us, 2 x MAD) in median paired whole-step load and wins at least 8 of 9 rounds, so at 8, 16 and 32 tokens the unpooler as a PDL secondary made the step **slower** beyond the noise in this prepare (the per-mode values fell out of the 16 KB tail). The timed comparison is therefore a second reading of layer_m1rt51s's policy against release_s: **cand faster**, pooled 1.0106, cross 1.0092-1.0120, nulls 0.15% / 0.13%; v16 1.0241, v1 1.0139, v32 1.0135, d32 1.0097, d16 1.0059, d1 0.9966. Against 3aee22ba4a404c8a (pooled 1.0052; v32 0.9993 there, 1.0135 here) the run-to-run spread of one comparison is larger than either bench's nulls. ABPOL front DIFF in inc only (16 skf / skf-r), abnd same, MoE same within files. One replay-class gate report in the cand: `moe1 stress main-d2` (an unpicked variant; manifest "rejected none"); not seen in 3aee22ba4a404c8a, which ran the same MoE code. ABEND 181 s (cand prepare 68-69 s). Profiles committed |
| 2026-10-31 | (tool, build, offline tests) | `tools/dl_mb_chain_patch.py` on layer_m1rt51s: `muse_decode_layer/cand/layer_mb2s.py`, SHA256 3aee22fd2b89de7072f5ff9362ef5fe1a120be3409e2377894869ea18bd2abcb | M-B slices 1-3 as M1 switches (unp, trig, attn) timed by the stable M1 tuner in `M1_MODES` against the incumbent's preferred modes; the fused MoE kernels trigger after their grid barrier; the gate replays every switch on (archive/docs/plan-2026-10-31-m-b-pdl-chain.md) | | Byte-identical on a second build. Device SASS (CUDA 12.8.1, `tools/dl_moe_sass_counts.sh`, hash prefixes stripped) against layer_m1rt51s: MoE (11 kernels) instruction count +1 each (the trigger), every fence, membar, mbarrier, barrier, bulk-copy, ACQBULK, atomic and local-memory count equal; projections with SK_PDL (`skp1`, 39 kernels): the 23 single-CTA sk instances gain 1-6 instructions (the conditional trigger), the other 16 are unchanged, every counted opcode equal. Attention extension with ONE_TRIG (`tools/dl_compile_check.sh FILE _ONE_SRC`, host change `pdl && !coop`): compiles, registers as before, no spills. pyflakes clean; `tools/dl_mb_modes_test.py` (183 cases: PDL, trigger and attention flags per mode and token count, census counting), `tools/dl_m1_gate_census_test.py` with the file's gate code, `tools/dl_stable_pick_test.py` all pass. Bench builder: ABPOL `m1stable` prints every M1 mode's paired whole-step saving per bucket and arm |
| 2026-10-31 | (tool, build) | `tools/dl_moe_inline.py` (now takes a target file; `LAYER_MAIN_EXTRA = ((78, 64),)` writes `_MOE1_MAIN_MAX_M` and the tuner filter) with muse_moe_w4a16/submission.py at d73c4fd on a copy of layer_m1rt51s: `muse_decode_layer/cand/layer_m1rt51s78.py`, SHA256 96f69ff097c2021f11bf8e168aca597866be463d35a46cb64df8abe2f51ecec3 | MoE mode 78 offered at 32 tokens only (coordinator's queue item; rule pre-registered in archive/docs/plan-2026-10-31-m-b-pdl-chain.md) |
| 2026-10-31 | a3760918c13c4e40 | scratchpad `m1s/bench11_mb2s.py` (inc = layer_m1rt51s, cand = layer_mb2s; 916,702 bytes) | Four-arm bench, `--dup=shared` (both files run M1, and four M1 prepares do not fit in 300 s), 8 rounds, seeds 270001+, order seed 83 | 1.968x (1.939-1.998) as a check | **Not resolved**: pooled 1.0015, cross 0.9981-1.0048, nulls 0.13% / 0.53% (cand2 slow at v1s, 129.1 against 124.8). The measurement that answers the slices is the cand's stable M1 tuner (ABPOL `m1stable`, whole-step load, median paired saving against the preferred mode / rounds of 9 faster): **8 tokens** (preferred o+front, 124.5 us): off -3.01/0, o -2.26/0, o+front+trig -1.00/3, o+front+unp+trig **+1.39/9**, o+front+unp+trig+attn +1.46/9 (attn is inert at 8) -> picked o+front+unp+trig+attn in both copies; **16 tokens** (preferred o, 180.1): o+front -0.51/0, o+front+trig **-6.06/0**, o+front+unp+trig -6.54/0, +attn -5.81/0 -> o; **32 tokens** (preferred o+front, 240.4): o +0.80/9, o+front+trig **-5.46/0**, o+front+unp+trig -2.30/0, +attn -4.55/0 -> o. So the sk entry trigger costs 1 us at 8 tokens and 5.5-6 us at 16 / 32, and the unpooler PDL behind the MoE's new post-barrier trigger saves about 2.4 us at 8 tokens on top of it (unp + trig against trig alone). Timed: v1s -1.5 to -1.9 us and v1f -1.1 to -2.5 on the cand copy, d32s -0.0 to -2.7; other shapes within 1 us. MoE picks: cand chose 47 at 32 tokens in this prepare (inc 51); front and abnd same in all arms. Every gate passed (manifest: 11 MoE variants, none rejected, no failures of any class), 24 of 24 checks per arm. Prepare cand 73 s, inc 70 s; ABEND 146 s. Profiles committed |
| 2026-10-31 | (build) | `tools/dl_mb_chain_patch.py --round=3` on layer_m1rt51s: `muse_decode_layer/cand/layer_mb3s.py`, SHA256 76da38c1c41e3bb71ae5a4c98399dc10bc530d8b28df4f0f9adecee0d51528aa | Round 3 modes from a3760918c13c4e40: off, o, o+front, o+unp, o+front+unp, o+front+unp+utrig (the trigger on the unpooler launch only), o+unp+attn; the gate replays every switch (o+front+unp+utrig+trig+attn) | | Byte-identical on a second build; every `_*SRC` region equal to layer_mb2s's (so its SASS); pyflakes, mode test, pick test and census test pass |
| 2026-10-31 | 36f93b13d4c14d5b | scratchpad `m1s/bench12_m78.py` (inc = layer_m1rt51s, cand = layer_m1rt51s78; 940,046 bytes) | Four-arm bench of MoE mode 78 at 32 tokens, `--dup=shared`, seeds 280001+, order seed 89 | **failed**: "the submission process exited" | **Illegal memory access** in the cand's prepare: the tail shows the attention self-tests and stresses passing, `TUNE front` at 1 / 8 / 16 tokens, then at 32 tokens `mma dropped` (the usual routes 0.973) and `moe front tuner: SGLang fallback (AcceleratorError: CUDA error: an illegal memory access was encountered ... cuCtxSynchronize)`, then the process aborted on the sticky error. The MoE self-tests, stress and tuner ran earlier (cut from the 16 KB tail), so which MoE variant the 32-token front chain ran is not in the log. Offline: every `Params` field the device code reads is set by the layer's host wrapper (none missing against the standalone's host), `Params` is unchanged, and 78's partial index `(q0 + tok) * 256 + rb` stays inside ybuf's last 32 rows because the pair rows q are an unpadded prefix sum over the local routes (q < M x 8 = 256 at 32 tokens). Not resolved; diagnostic below. Profiles committed |
| 2026-10-31 | 18d48c5676fa4461 | scratchpad `m1s/diag78.py` from `tools/dl_moe78_diag.py` on layer_m1rt51s78 (SHA256 a17a51df...; not a candidate) | Mode 78 crash diagnostic: one layer (no bench), 78 preferred at 32 tokens, a synchronized `D78` print before and after each step of the front tuner at 32 tokens | 2.016x (1.986-2.047) as a check | **No fault**: passed 13/13; MoE 32-token pick main-lean-grp-fin4ps (graph bench 91.8 us against 51's 93.2 and 47's 92.9; stable rounds: 51 -1.31/0, 47 -1.19/0, 37 -2.02/0 against 78), so 78 ran through the front tuner's front + MoE chains (route variants with the PDL launch included), the stress and gates (`failed 0: none`), the timeline and the judge's tests. The crash of 36f93b13d4c14d5b has so far appeared only in the two-layer bench process (cand comment-stripped and exec'd beside the incumbent's module). Profiles committed |
| 2026-10-31 | a03039679e304d95 | scratchpad `m1s/bench14_m78.py` (inc = layer_m1rt51s, cand = layer_m1rt51s78; `--dup=shared`, seeds 280101+, order seed 101; 940,048 bytes) | Rerun of the mode 78 bench on fresh seeds | 1.995x (1.965-2.025) as a check | **No crash** this time (manifest: 12 MoE variants checked, none rejected; one route sensitivity report at skf-r@16). **78 not picked**: ABPOL MoE same in all arms, 47 (main-lean-grp-nohoist) at 32 tokens in both files, so the cand's stable tuner left the preferred 51 for 47 and 78 did not save more than 47 did (its values fell out of the tail); in 18d48c5676fa4461 78 had beaten 47 by 1.19 us in 9 of 9 rounds. Pre-registered rule (1) fails, so 78 is not shipped. The timed comparison ran identical policies everywhere and still read **cand slower**, pooled 0.9987, cross 0.9982-0.9993, nulls 0.06% / 0.05%: with `--dup=shared` the copies share one prepare, so the nulls miss module and prepare effects and a 0.1% difference between near-identical files resolves. Readings of shared-state benches are therefore not decisions (a3760918c13c4e40 and 6e51f2c9dc774a79 were not resolved either way). Prepare inc 71 s, cand 74 s; ABEND 148 s. Profiles committed |
| 2026-10-31 | (tools, build, SASS) | `tools/dl_proj_c1_port.py muse_bf16_proj/submission.py layer_m1rt51s.py OUT HEADER` at this commit (muse_bf16_proj at f1e59f3, which carries C1 05286d9 and C2 3044ee7): `muse_decode_layer/cand/layer_m1rt51sc1.py`, SHA256 9373b976efa8de7cfde795a2a59b987b04885588f63fc9c6af9577ec8e0af80b | **Candidate layer_m1rt51sc1**: layer_m1rt51s + projection C1 (NOVOTE consumer waits on every compressed sk config, FULL twins at M == TOKR on all but qkv32, launched by the layer's sk runner when the call has TOKR tokens) | | `tools/dl_proj_inline.py` now takes a target file and reads `SK_NOVOTE`, `SK_NACC`, `SK_FULL`; the port re-applies the layer's M1 sk patches from `tools/dl_gate_patch.py` (reader-side `fence.proxy.async.global` in thread 0 and the producer lane, runtime PDL order in bit 30 of M, with M kept mutable for FULL's `M = TOKR`). Byte-identical on a second build; outside the projection regions only `_skp_runner` (FULL dispatch, `_SKP_FULL_SEEN`) and the SUMMARY line `skp full twins launched` change; the @SKP_PY@ packing code is unchanged. SASS (CUDA 12.8.1, `tools/dl_moe_sass_counts.sh FILE skp1`, matched by template arguments except NACC) against layer_m1rt51s: the 10 compressed NOVOTE configs and their 9 FULL twins (the 19 NOVOTE kernels) differ from their base only in instruction count (-4 to -483) and SYNCS +1; every FENCE, MEMBAR, BAR, ERRBAR, CCTL, bulk copy, ACQBULK, atomic and LDL / STL count is equal; the raw, fused (skf) and GEMV kernels are identical; the two wide kernels lose 2 instructions (a new template argument for the standalone's bench-only GRED path, off here) with every count equal. In all 19 NOVOTE kernels a VOTEU.ANY and then a WARPSYNC.ALL precede the first LDSM, as the independent review of C1 found in the standalone. pyflakes clean. C1's third part (pooler m = 128 cuBLASLt pin) is not ported: the layer's 128-token front is its concat GEMM. **C2 (rd kernel) not ported**: its two picks do not exist in the layer's 8-token path (o at 8 is M1's PDL sk launch, which the rd kernel cannot replace without losing M1's o PDL; the pooler at 8 is fused with the router in skf) |
| 2026-10-31 | (tool) | `tools/dl_bench_fix_m1.py` | Bench-only: fixed M1 policy (layer_m1rt51s's {8: o+front, 16: o, 32: o}) and 500 gate replays, applied to both files of a bench so four independent prepares fit in 300 s (the coordinator's suggestion after a03039679e304d95) | | Never part of a candidate; ABPIN records the patched files' hashes and the runs row the originals' |
| 2026-10-31 | b70cfa48c0264d20 | scratchpad `m1s/bench15_c1.py` (inc = layer_m1rt51s f1235356..., cand = layer_m1rt51sc1 9373b976..., both through `tools/dl_bench_fix_m1.py`; 938,857 bytes) | Four-arm bench of projection C1 in the layer, **independent prepares** (`--dup=own`), 8 rounds, seeds 300001+, order seed 103 | 2.009x (1.979-2.040) as a check | **Not resolved**: pooled 1.0007, cross 0.9991-1.0022, nulls 0.08% / 0.23%. Per load: v1 1.0034 (all four crosses 1.0022-1.0047), d16 1.0027 (all four 1.0009-1.0045), d32 1.0013, v32 1.0005, v16 1.0001, d1 0.9959 (crosses split, nulls 0.8% / 1.1%). Per shape the cand is 0.4-1.6 us faster at d16s, d32s (all four crosses -0.8 to -1.3), v1s / v1f and v16s, and 0.1-1.7 us slower at d1f and d32f: about the standalone's 0.2-0.45 us per projection launch, under 0.3% of a load. Policies equal in all four copies (MoE, front, skp, M1 fixed) except abnd at 8 tokens in cand2 (c8x256 instead of c1x1024). Gates: 24 of 24 checks per arm; one reference-class report in the cand's prepare: the route side-stream self-test (still on the 1e-4 bound) read 1.21e-4 and fell back, a path the skf-r front does not use at these picks. Four independent M1 prepares took 61-63 s each with the fixed policy; ABEND 249 s, so this layout has about 50 s of headroom. Profiles committed |
| 2026-10-31 | (follow-up, from b70cfa48c0264d20) | `_route_side_self_test` in every M1 file since the release | **Follow-up: the route side-stream self-test's 1e-4 bound trips on noise and costs about 1.7 us at d1 sliding when it does** | | In b70cfa48c0264d20 the test read 1.21e-4 against its 1e-4 bound in one of the two cand prepares (ABPOL stages: `route_side_ok` 0 in cand2 only; `_FAILED` is shared by the two copies, so ABCAND showed it): `route side: fallback (RuntimeError: route-side self-test: rel diff 1.21e-04)`. The test compares the MoE call with its routing metadata built on the side stream against the one-stream call, eagerly and as a graph, at 1 / 8 / 16 / 32 / 128 / 256 tokens; the MoE's FP32 atomics make the two differ by merge order, the same noise the fresh-input gate replaced for the MoE stresses (user decision 2026-10-30). The fallback is on the timed path: at one token the front is fp32, a non-route variant, so `route_side` decides whether the pre-expert norm and metadata run on the side stream. cand2's d1s ran 84.8 us against cand's 83.1 (null cand2 / cand 1.0109 at d1, the largest of the bench). Proposal for the user (not applied): move this test's comparison to the fresh-input gate, or bound it by the control envelope like the MoE stress; until then any run can lose about 2% at d1 by chance |
| 2026-10-31 | (build, review, no check yet) | `tools/dl_mc_c1_patch.py layer_m1rt51sc1.py OUT HEADER` at this commit: `muse_decode_layer/cand/layer_mc1s.py`, SHA256 2998259e7109cfaadfd21c33db79266ba73297e0dbbaf577088aff02f7d1c38f | **Candidate layer_mc1s (M-C C-1)**: at one token the o projection and the attention boundary as one cooperative kernel (`mc1_kernel`: cgemv o rows, grid barrier, residual and second norm partials, grid barrier, outputs), behind a self-test against the two-kernel path, the fresh-input gate at 1x1 sliding and full, and a stable pick against off (archive/docs/plan-2026-10-31-m-c-t1-fusion.md, "C-1 implementation") | | Built by an implementation agent (branch worktree-agent-ac562c4173873483c, 30c6cab), reviewed read-only by an independent agent: sound with caveats, nothing blocking; coordinator's condition checked explicitly (the launch carries only the cooperative attribute, no griddepcontrol in the kernel, no successor can launch before every C-1 CTA exits). Review should-fixes applied here: a `__threadfence()` before the generation bump (the arrival atomics are relaxed), and the SUMMARY reads the timeout flag after the timeline; PTX citations corrected (coherence order 8.9.6, causality axiom 8.10.6). Byte-identical on a second build; pyflakes clean; every other source block equals layer_m1rt51sc1's (the kernel is its own extension). SASS (CUDA 12.8.1, `tools/dl_mc_c1_sass.sh`): `mc1_kernel` 1305 instructions, 40 registers, no spills, 2016 B static shared memory, MEMBAR 3 (MEMBAR.SC.GPU: two barrier releases and the bump fence), ATOMG 5, BAR 7, NANOSLEEP 2, LDGSTS 4, ACQBULK 0, PREEXIT 0; the projection extension's 48 kernels identical to the base. 768-thread blocks, grid = occupancy x SMs (264 expected), within [64, 1024]. CPU emulation (`tools/dl_mc_c1_cpu_check.py`, 1000 inputs): the stream never flipped, the next input differed on at most 2 elements, the router copy within 4 ulps. Expected saving (hypothesis): 1.5-2.5 us at d1 sliding |
| 2026-10-31 | 5ae9ce318e014edf | scratchpad `m1s/bench16_mc1.py` (inc = layer_m1rt51sc1 9373b976..., cand = layer_mc1s 2998259e..., both through `tools/dl_bench_fix_m1.py --replays=250 --mc1-replays=250`; 975,167 bytes) | Four-arm bench of M-C C-1, independent prepares | **failed**: "the submission process exited" | **C-1 measured slower in-process**: the cand's prepare completed; `_mc1_enable` runs setup (the cooperative probe), the self-test against the two-kernel path and the fresh-input gate at 1x1 before the stable pick, and the pick ran: `stable mc1 T=1: pref off 66.0 margin 0.30 rounds 9 on -1.16/0 -> off`, so the fused kernel made the whole 1x1 step (46 x sliding + 16 x full) 1.16 us slower in all 9 rounds. The stopping rule (stop under 1 us of saving) closes C-1, and C-2 is not built. Then the bench process died in its third prepare with an illegal memory access (sticky; the faulting kernel is not in the tail), the second such fault seen only inside a two-layer bench process (36f93b13d4c14d5b was the first; both runs carried an extension the incumbent arm lacks). Side finding: with the bench-only gate cut to 250 replays the M1 gate failed at 1x8 sliding in the cand and inc2 prepares (`ids!=` flips 6-7 and the element envelope against a control measured on 250 calls), turning M1 off in those copies; `tools/dl_bench_fix_m1.py` now refuses fewer than 500. Profiles committed |
| 2026-10-31 | 2edb120747cb4611 | `muse_decode_layer/cand/layer_m1rt51sc1g.py`, SHA256 a00a0efa0a761d990d7298a64340751e6a3979369aa8247a1f3d958f975c7480, from `tools/dl_route_side_gate_patch.py` on layer_m1rt51sc1 at this commit | **Candidate layer_m1rt51sc1g**: layer_m1rt51sc1 (layer_m1rt51s + projection C1) with the route side-stream self-test on the fresh-input gate (user decision 2026-10-31); single-file check as the coordinator asked | **2.005x** (1.975-2.035) | passed 13/13, no fault. Route side gate ok at all six token counts (T=1: rel 6e-05 cand / 6e-05 ctrl, element 1e-03 / 1e-03, 0 of 200 over 1e-4, order-1 0; T=8 / 16 / 32: rel 4e-05 to 8e-05 against the same control; T=128 / 256: 0), negative controls reject every injection and pass the uninjected output; manifest: 11 MoE variants, none rejected, route gates ok 14 / 14 (8 variant gates and the 6 side gates), `failed 0: none`, so `route_side` did not fall back (a fallback is listed there; the stage-flag line fell out of the tail). M1 on with {8: o+front, 16: o, 32: o}, every M1 gate ok. Byte-identical on a second build; only `_route_side_self_test` and one SUMMARY line differ from layer_m1rt51sc1. Profiles committed |
| 2026-10-31 | 6e51f2c9dc774a79 | scratchpad `m1s/bench13_mb3s.py` (inc = layer_m1rt51s, cand = layer_mb3s; `--dup=shared`, seeds 290001+, order seed 97) | Four-arm bench of M-B round 3 | 2.010x (1.980-2.041) as a check | **Not resolved**: pooled 1.0002, cross 0.9986-1.0018, nulls 0.25% / 0.06%. Stable M1 tuner (paired whole-step saving against the preferred mode / rounds of 9 faster, identical in both copies since the state is shared): **8 tokens** (preferred o+front, 127.6 us): o+unp -1.38/0, o+front+unp +0.31/7, o+front+unp+utrig **+0.80/9** -> picked; **16 tokens** (preferred o, 194.2): o+unp -1.80/0, o+front+unp -2.54/0, +utrig -2.04/0, o+unp+attn -1.67/0 -> o; **32 tokens** (preferred o+front, 202.2): o +1.10/9, o+unp +0.95/9, o+front+unp -0.05/3, +utrig +0.27/7, o+unp+attn **+1.25/9** -> picked, 0.15 us better than o alone. Timed shapes all within about 1 us (v1 pooled 0.9996, d32 1.0027, v32 1.0033). Gates: MoE PDL self-test rejected the unpicked sk8w4s-er (replay class, share of calls over 1e-4 0.077 against 2 x the control's 0.038), one sensitivity report (unpicked main-m2-ar2-ub at 256); 24 of 24 checks per arm. Prepare cand 73 s, inc 70 s; ABEND 147 s. Profiles committed | | Byte-identical on a second build; repack region and `Params` unchanged; MoE SASS: the eleven existing kernels identical (instruction counts included), mode 78 instr 17485, BAR 17 (51: 16), ATOMG 136 (51: 148), no LDL / STL, every fence count as 51's. ybuf: (4096, 4096) FP32 in the layer, 78's partial rows 4064-4095 are unused by any pair row below 64 tokens (at most 504) |

## Bench arm bias (2026-10-30)

The three-arm bench (inc, cand, null) cannot measure a bias of the inc arm: its null was the
candidate's own kernel_fn and state. On identical code the inc arm ran 4-6 us slow at d1
(50c4c1328ab742f4, 720079090ee44cc3), and the identical-code loads of 29fd004fd58e40ef read
inc / cand 1.0066 against 1.0084 overall. The mode 51 (+0.84%, 29fd004fd58e40ef) and M1
runtime (+0.87%, f0ee07c4d4904511) readings are therefore not established. The harness now
runs four arms (inc, inc2, cand, cand2), each copy with its own prepare in random order, and
decides on the pooled cross ratio against both nulls (`tools/dl_make_bench_pub.py`). Analysis,
prepared A/A, mode 51 and M1 runtime files, and reading rules:
archive/docs/steering/2026-10-30-layer-bench-arm-bias.md. f0ee07c4d4904511's profiles are committed
with that change.

Addendum (four-arm runs 724aa0dd6ed140ca, 2dc566298be1498f, 40b9b3f795f84073): the A/A run
reads pooled 1.0045 and the whole offset sits at d1, where the inc copy runs 3-6 us slower than
the other three copies (d1 null inc2 / inc 0.955, 0.939 and 0.949 in the three runs). inc is the
exec'd, comment-stripped module, so item 4 (module identity) holds at d1 and nowhere else; the
mechanism is still unknown. Read candidates per load against the A/A per-load ratios, and leave
d1 out of any attribution. Prepare-time policy flips between copies of one file (ABPOL DIFF in
front, abnd, MoE at 32, W4) add to the nulls; design for stable picks:
archive/docs/plan-2026-10-30-tuner-stability.md.

## Validation policy change: fresh-input replay gate (user decision, 2026-10-30)

The user approved this change on 2026-10-30, in chat to the coordinator, who relayed it to the
layer agent as the explicit approval that the post-reset steering (section 5) and the
attribution report (section 3) require. It is recorded here before any run uses it.

What it replaces: the 1e-4 bound on one replay's output against one eager call, which the
M1-off layer itself fails on FP32-atomics merge order (checks 6372154fcd7a4fa9 and
3e2aa83358dd4f60), in two places only:

1. M1's replay stress (`_m1_stress`).
2. The shipped MoE replay self-tests: the mixed-size stress (`_moe1_stress`) and the
   route-kernel PDL self-test (`_moe1_pdl_self_test`), so atomics noise can no longer drop
   sk8w3s or another variant in a scored run.

The gate (`tools/dl_gate_patch.py`):

- (a) Fresh inputs on each replay: a device counter picks the next of a bank of input sets
  (5 for M1, `GATE_BANK` = 4 for the MoE), copied into the captured inputs; every replay's
  outputs are compared on the device with that input set's reference run (M1 off eagerly, the
  plain eager MoE launch). No host sync between replays.
- (b) Order-1 failure: any element error of 1e-2 or more fails (also a relative L2 error of
  1e-2 or more, a non-finite value, or differing ids). The element error is
  |out - ref| / (|ref| + the row's RMS), so a one-ULP BF16 flip counts at most 2^-7 and a read
  of another input's rows, a whole row or one tile of it, counts about 0.3-1.
- (c) Spread matches the control: M1 on (candidate) against M1 off (control), both replayed
  2000 times; for the MoE, the graph replays (and eager PDL launches) against the same step
  made eagerly (`GATE_CTRL_CALLS` = 400 for the stress; the plain launch as often as the PDL
  one for the PDL test). Per output, the candidate's largest relative L2 error must be under
  1e-4 or within 2x the control's, and its share of calls over 1e-4 within 2x the control's
  share (at least 1%). Both arms' envelopes (largest relative and element errors, calls over
  1e-4, order-1 calls) are printed in SUMMARY (`m1 gate`, `moe gate`, `moe pdl gate`) and
  recorded in the run rows.
- (d) Unchanged: the FP32-reference self-tests (`_moe1_self_test` against `W4_TEST_TOL`, the
  attention, SKP, route and norm tests), the M1 bit-16 self-test (PDL launch bitwise equal to
  the plain one), the generation count and scratch invariant of the MoE stress, and the
  judge's tolerances.

Addendum, user decisions of 2026-10-30 (given in chat to the coordinator, relayed to the layer
agent; recorded here as such):

1. Clause (b) becomes: order-1 = an output's relative L2 error >= 1e-2. Routing-id flips are
   judged against the control's envelope (the candidate's flips at most 2x the control's rate
   plus two calls), not an outright failure. Reason: the first element bound failed the M1-off
   layer itself, eager calls included (60cf58b69b75491e).
2. The same fresh-input gate applies to `_route_variant_check`'s replay part (the route
   variant's front + MoE graph, 200 replays against 200 eager calls of the same step on fresh
   FFN inputs); its comparisons with the base variant (top-8 sets, 1e-3 bounds) and the scratch
   check are unchanged.

Second check, added at the researcher's caution (archive/docs/steering/2026-10-29-course-check-validation-prefetch.md,
section 4) that relative L2 can miss a stale subset or one element: per call, the largest
element error in units of its row's largest reference magnitude must stay within 2x the
control's measured maximum (floor 2^-7, one BF16 ULP of the row maximum). Negative controls
(`_gate_negctl`, report-only, on a real output and the measured control envelope in every MoE
stress and route gate): inject entry j's reference into entry i's output as (a) one 64 x 128
tile, (b) 1% and 0.1% of elements and one row, (c) one random element, plus an uninjected
observation for false rejection. Offline (CPU, synthetic rows at the measured MoE and layer
noise levels, `tools/dl_gate_patch.py` at this commit): relative L2 alone rejects the tile
(0.1-0.3), 1% (0.1), 0.1% (0.04-0.05) and the row (0.09-1.0) but **misses the single element**
(relative 2e-4 to 2e-2); the element check rejects all five (element error 0.03-2 against
limits of 7.8e-3 to 1.4e-2), and the uninjected observation passes in all six cases. Failures
now carry a class tag (reference, replay, nonfinite, workspace, sensitivity) and SUMMARY ends
with a one-line manifest (MoE variants checked and rejected, route gates passed, failures per
class).

Addendum, user decision of 2026-10-30 (given in chat to the coordinator, relayed to the layer
agent; recorded here as such), option 1 of archive/docs/steering/2026-10-30-m1-1x1s-gate-failure.md:
the M1 gate judges only shapes whose captured candidate graph makes at least one M1 PDL launch.
The census counts launches with the PDL bit in the sk and fused runners (the only two places
that set it) while kernel_fn's graph is captured under the candidate policy; a shape with none
is skipped and logged "skipped: no M1 launch". The bounds at the shapes M1 changes stay
exactly as approved. Fail-closed: if the census raises, the shape is judged as before and its
line says "census failed". Reason: at 1x1 the M1-on and M1-off graphs are the same kernels,
and the gate failed there on one arm in 6ce5be0ba29642ea and the other in d6c30f88c617438b.
Implemented in `tools/dl_gate_patch.py` (M1_SRC, CENSUS_REPS); offline test
`tools/dl_m1_gate_census_test.py`.

Addendum, user decision of 2026-10-31 (asked by the coordinator through AskUserQuestion and
relayed to the layer agent; recorded here as such): the route side-stream self-test
(`_route_side_self_test`) moves to the same fresh-input gate as the MoE stresses and the route
variant check: order-1 = relative L2 >= 1e-2, everything else judged against the control's
envelope. Reason: its 1e-4 bound tripped on MoE atomics noise (1.21e-4 in b70cfa48c0264d20)
and the fallback it triggers costs about 1.7 us at d1 sliding. Implemented in
`tools/dl_route_side_gate_patch.py` (candidate `layer_m1rt51sc1g`): per token count GATE_BANK
fresh routings with one-stream eager references, the side-stream step as a replayed graph
(candidate) and eagerly (control), ROUTE_STRESS calls each, report-only negative controls.

Not changed by these decisions (every other check keeps its bound): `_one_stress` (attention,
1e-4 against eager and 5e-2 against SGLang) and the norm self-test; the route side-stream test
keeps 1e-4 only in files built before the 2026-10-31 addendum above.

## Closed lead (2026-10-30, unproven): input-history dependence in the attention output

**Closed 2026-10-30 (verdict A, benign).** Layer diagnostic d93286b5110a4410 and attention check 2b93c6ecf4f641c6: the history effect is FP32 merge-order rounding (1-3 elements by one bf16 ulp). Production repeats on one input are not bitwise (4-5 distinct outputs in 9 calls); a fixed-order debug build (ONE_DETM) is bitwise under every history; 283 whole-workspace scans after production calls found rows, sums, tickets and counters zero. Details and the PTX reset/reuse chain: `docs/runs-attention.md`.

In 60cf58b69b75491e the attention stage of an eager call on a fresh input differs from that
input's reference by up to 1e-4 at decode 1 sliding and 2e-4 at verify 1 sliding (every arm,
eager calls included), while the attention stress on repeated inputs is bitwise
(replayed_vs_eager 0.0 at 1x1 / 1x8 in fdf75ba5a8124544's inc prepare). Merge-order noise would
show on repeated inputs too, so the hypothesis is that the output depends on the previous call's
input (a low-amplitude stale read, for example of the cache rows the step writes). Not yet
tested. Prepared for the next quota: `tools/dl_history_diag.py` on the g51 candidate measures
repeated-input noise, previous-call dependence, and the same with the previous call on a
separate case (cache or workspace as the carrier), for qkv, attention, the whole step and the
MoE alone. The relative-L2 gate cannot see a 1e-4 effect; this needs the history test.

## Pre-check audits for the integrated candidate (2026-10-29, before the 10-30 checks)

Source: the layer at the commit that adds this section (bundle bb98b7b plus the FIN8 stress and
the summary manifest below). No GPU run; local compiles only.

### PDL provenance of every load before a griddepcontrol.wait

A load before the wait is safe when its writer has completed before the launch, whatever the
overlap. In a captured step every kernel completes only after its own wait (or launch) has seen
its predecessor complete, so anything written by a kernel two or more launches back, by the
previous replay, by prepare(), or by the harness before the step is complete.

| kernel (launch) | load before the wait | writer | writer live at the load? |
|---|---|---|---|
| route_kernel (PDL secondary of the concat cuBLAS GEMM or the skf sk_kernel) | `w` (pre-expert norm weight, 4096 BF16) | prepare (`_norm_self_test` sets `pre_expert_w`) | no |
| route_kernel | `bias` (router bias, 256 FP32) | prepare (`route_bias`) | no |
| route_kernel | nothing else: logits and the pooled rows (the predecessor's output) are read after the wait | the front GEMM | yes, hence after the wait |
| fused MoE, fat and skinny, under `pdl` (PDL secondary of route_kernel) | no global load. Shared memory: the ring's mbarrier inits and `fence.mbarrier_init`. `cp.async.bulk.prefetch.L2` of `w13f` (fat) or `w13` (skinny) units | prepare (`_moe1_prepare` repacks) | no; a prefetch hint is not a load into registers or shared memory, and nothing reads its result |
| fused MoE | `local_start` (`ld_lo`, `ld.global.nc`), `topk_ids`, `topk_w`, `x`, `ctr[3]`, `ss` | harness input; route_kernel; route_kernel; route_kernel; the previous MoE launch | all read after the wait (the preamble ends at the wait) |
| attention `attn_one_kernel` (launched without the PDL attribute: `_ONE_PDL = False`) | `page_table`, `seq_lens`, cached K/V rows at positions below `seq_lens` (cp.async), page indices | harness inputs written before the step | no |
| attention | query rows (`qkv`), the constants group, the flat kernel's epoch | qkv projection; previous attention launch | after the two inserted waits (tools/dl_one_inline.py) |

M1 candidate (not in the bundle; built after it, `PORT_SK_PDL`): with the bit set, the sk o
projection (PDL secondary of attention built with ONE_TRIG) and the skf front (PDL secondary of
the attention boundary) load before their wait only: the weight units of the first
min(NSTAGE, stages) stages and the fused router pieces (`bulk_g2s` from the repacked streams,
prepare), the escape table (`ptab`, prepare), the tensor map (a kernel parameter built on the
host; `prefetch.tensormap`); writer not live. x (attention's or the boundary's output) is
copied by TMA only after every thread's `griddepcontrol.wait` (`issue_x` in the
`if constexpr (PDL)` block after the wait), and each stage's full barrier expects the weight and
x bytes together, so no stage completes on the weights alone.

The attention waits return at once today because the launch carries no PDL attribute; the
table records what would hold if `_ONE_PDL` were turned on. The cache rows the kernel itself
writes (positions `seq_lens` onward) are written in-kernel, not by a predecessor. Every
inserted wait has the `"memory"` clobber. `ld_lo`'s `ld.global.nc` reads an input that no
kernel of the step writes, so the non-coherent load form is not the read-after-write hazard of
a predecessor's output. The other PDL secondaries of the layer (pre_norm_kernel, the boundary
kernels, router and route_align, the W4 GEMMs) date from the blanket PDL change
(docs/ruled-out.md, "PDL across our own chain"), are unchanged here, and were not re-audited.

### Compiler audit of the attention instances the layer builds

nvcc 12.8.93 (`nvidia/cuda:12.8.1-devel-ubuntu24.04`), `-O3 -std=c++17
-gencode=arch=compute_90a,code=sm_90a -Xptxas -v` (tools/attn_one_sass.sh) on
muse_attention_decode/attn_one.cu at this merge (ONE_DCARRY, ONE_PK16, ONE_VHOIST on), the 11
instances `_ONE_BUILT` names; the layer's copy differs from these only by the inserted
`griddepcontrol.wait` (ACQBULK 1-2 per instance; every other ordering opcode count equal, see
the bundle commits). Warnings: only #177 (unused declarations); no C7510-C7515 wgmma
serialization warning. Registers / spills: <8,51> 80, <8,19> 112, <8,15> 80, <8,11> 109,
<8,3> 110, <8,1> 110, <1,49> 128, <1,13> 76, <1,11> 115, <1,3> 109, <1,1> 108; 0 bytes of
spill everywhere.

Trap paths: a scratch build with every `__trap()` (bounded waits in `mbar_wait` and
`coop_barrier`) replaced by an empty statement, for comparison only, gives the same GMMA,
`WARPGROUP.ARRIVE` and `WARPGROUP.DEPBAR.LE gsb0, 0x0` counts on every instance (<8,51>: 12, 6,
6; the q_len 8 per-request instances 4, 2, 2; q_len 1 runs HMMA only). The differences are the
BPT.TRAP sites themselves (11 in <8,51>, 7 in <1,49>) and fewer BSSY convergence regions
without traps (<8,51> 84 to 74, <8,19> 48 to 39). A GMMA followed within two instructions by a
`gsb0, 0x0` wait occurs once in <8,51> with traps and four times without, so the trap-bearing
build is not the more serialized one by that measure. With nvcc 12.8.1 the participant's C7512
report (nvcc 13.0) does not reproduce; no 13.0 toolchain was available to test the claim
itself. The traps stay.

### FIN8 stress coverage

`_one_stress` now takes `ctx` and an expected `(nc, mode)` and raises when the plan differs.
The batch-1 full stresses run at ctx 20000 (313 tiles, more than the 252-split cap), which plans
mode 11 with eight workspace copies at q_len 1 and 8; the scored plans in run 9b921febc7ad4140
are (252, 8, 11) for both, while that run's 1x8 full stress planned (62, 8, 15). Per stress: 2000
graph replays back to back, then the replayed output against an eager call (1e-4) and against
SGLang's path (5e-2), a NaN check and the workspace invariant (rows, l, tickets, cooperative
parity, flat counters); then 50 replays on fresh qkv rows written into the captured buffer with
the same checks; then on batch 1 page 1 the speculated-length fallback (sequence length 64 lower
in place) for 50 replays. Every other stress asserts its scored plan too: 1x1 sliding (1, 13),
16x1 full (1, 49), 1x8 sliding (1, 15), 16x8 and 32x8 full (1, 51).

### Pins, mixed-size MoE replays and package controls

- Public layer arm: vjeux/okc-2026 `muse_decode_layer.py`, git blob
  936fefffb493c4a836e8c08fc58dff50ce338dd5, the blob at that repository's head commit
  f35a41ee1c13e316dabbf772830d47b23b453393 ("v2: dead-code removal", 2026-10-28T02:03:32Z, the
  last commit touching the file; read with `gh api` on 2026-10-29), SHA256
  cf4a75ee6e4a5cd9e33b82b5347c47abcdc7b81d80dc6a8b5262dea7388f849c. The judged public layer
  profile cited in the gap maps (job f662ee18) came from another account's memos; that this blob
  is the file behind that job is not established. Incumbent arm: the layer file at a923f11,
  SHA256 3e9ba3df33f4b8e6fc5d26e17a33ad1c9c0e3ebd669b87e89c8681ce1eae5b16 (the file hash the
  community audit records; the judge receipt's payload content hash 8de70993... is a different
  hash). tools/dl_make_bench_pub.py now hashes both files as read and prints them as ABPIN.
- MoE replay stress: every skinny variant (sk8w4s, sk16w4s, sk8w3s, sk8w4s-er) captures 8, 16,
  1, 32 and 8 tokens back to back in one graph on the shared scratch, and every fat mode 32, 128,
  17 and 256; 2000 replays with one host sync after them, then the last replay's outputs against
  an eager call (1e-4 each), the generation count and the scratch invariant. A failing variant is
  dropped alone unless it is the last of its kind. The bench still picks per token count; the
  sk8w3s evidence at 1 / 8 / 16 tokens decides nothing at 32. Attention stresses print the same
  retained-output and sync-interval description.
- Package controls, prepared and not run: `tools/dl_package_control.py OUT BASE --minus
  attn|moe|proj` takes one package's generated regions back from the incumbent source (attn:
  `_ONE_SRC`, `_ONE_HOST_SRC`; moe: `_MOE1_*`; proj: `_SKP_*` and the `@SKP_PY@` region). The
  three single-package controls of the current candidate compile (sm_90a, CUDA 12.8): attention
  12 kernels, no spills; MoE 6 kernels, spills only in the four fat modes as at a923f11; SKP 43
  kernels, spills only in the four-chunk cgemv instances the layer does not run. Used only if
  the combined candidate is neutral or mixed.

### M1 follow-up file (2245513): LATE_ARRIVE against the standalone

SASS of the layer's projection extension as PORT_SK_PDL builds it (`-DSK_PDL=1`,
`tools/dl_moe_sass_counts.sh ... skp1`) against `muse_bf16_proj/submission.py` at 31b853d built
device-only with `-DSK_PDL=1` (the steering evidence's `gen_sk.py`), CUDA 12.8.1, sm_90a: the 26
kernels present in both have identical counts of every FENCE, MEMBAR, SYNCS (mbarrier), BAR,
ERRBAR, CCTL, UBLKCP, UTMALDG, ACQBULK, WARPSYNC, VOTE, LDL and STL opcode and identical
instruction totals. That includes the four LATE_ARRIVE instances (MT 2, UPW 2): sk<2,32,3,1,4,2,4>
(qkv32) and sk<2,32,2,2,2,2,3> (o32, unp32), compressed and raw: FENCE 5 / 8, SYNCS 18 / 32 and
16 / 30, BAR 3, WARPSYNC 3 / 9, MEMBAR 3 / 5, ACQBULK 1. Layer-only kernels: the four cgemv
instances (the device-only standalone build leaves cgemv out) and the fused pooler + router at 32
tokens (UPW 4, so not LATE_ARRIVE). No spills (earlier compile check).

### Summary manifest in the log tail

SUMMARY now also prints the SKP self-test count with the layers whose cgemv passed, the three
fastest SKP candidates per layer at m = 1 and 8, and the MoE PDL self-test line, next to the
existing stage flags, MoE picks per token count, front picks, attention plans and failures.

## F-LAYER-0 (Family F donor comparison), 2026-10-31 UTC

Measurement only. User decision of 2026-10-31 (relayed by the coordinator): Family F is for
learning; donor files and donor code are never submitted, and a donor win is re-implemented in
our own code with attribution in comments. The bench files below carried the donor only inside
check runs; no donor source is in this repository.

### Arms and pins

| arm | file | SHA256 |
|---|---|---|
| R1 = cand, R2 = cand2 | our board entry 3fbb3ad027e84c6b, `muse_decode_layer/submission.py` at 2cee446 | b9c23bda16bb9d6311f025062e415841c67b16284737172badaa190fdc506a49 as read; sent comment-stripped (whole-line comments only, `strip_comments`), e88797b0e18e3646d6b98695493935ae968f3df5d70331aa2ea265358a848b64 |
| D1 = donor, D2 = donor2 | alygg77/okc-muse-spark-kernels at 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0, `kernels/muse_decode_layer.py` (Apache-2.0, NOTICE credits vjeux f35a41e and Matsvei) | 8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a as read and as sent (byte-identical: no patch, no strip); equals the donor results.json `kernel_sha256` of its receipt a61afb9687f14eff (2.2058x, rank 40 at 2026-10-29 20:54 UTC) |

Bench builder: `tools/dl_make_bench_pub.py` (at bc10cb1) passed through a scratchpad patch script
(`make_donor_tool.py`, 12 asserted string edits, not committed) that adds `--other=donor`:
the donor travels exact (lzma + base85, exec'd from a string as the pub arm is); its extensions
(`_att7`, the five `_moe*` builds, `bfp__get_ext`, `_graft_glue_module`, `_graft_b2_module`,
`_sl_cluster_module`, `_de_ext`, `_router_fp32_ext`, `_gle_ext`, `_gle_b1x_module`, and
`_moe(T, state)` for T in 1, 8, 16, 32, 128, 256) build on the bench's import threads with
`_MA_SFX` preset to the PRMT suffix, as the pub arm does; ABPOL prints each side's own picks;
`--pad-seed` allocates a seeded 0-32 MiB pad (512-byte steps) before every arm's prepare(),
imitating the judge's `prepare_pad`. Each arm has its own prepare() (`--dup=own`): D2's tuner
ran independently (its picks differ from D1's at T = 64 in run 1 and at T = 128 in run 2) but
shares the donor module's per-device scratch (`_WS7`, `_CL7`, lines 3721-3737, 3841-3856) and
compiled extensions, so the donor null is weaker than the incumbent's on allocation layout. D2's
prepare took 1 s against D1's 16 s (Triton JIT and module caches warm). Both benches: all six
loads (12 cases), 12 rounds x 2 inputs (busiest / second-busiest EP range), cold L2 (256 MB
memset before every replay), graph replays in a seeded random order per round, every arm on its
own copy of the inputs, ABCHECK against the restated judge checks for every arm and input.

| run | job | bench SHA256 (922,559 bytes) | seeds | order / prep / pad seed | prepare order | pads (MiB) | judge (cand path) |
|---|---|---|---|---|---|---|---|
| 1 | 20abe6291eb04ed6 (check) | 664acccb... (benchF0a) | 410001+ | 71 / 72 / 73 | cand cand2 donor donor2 | 21.9 / 2.3 / 16.7 / 16.7 | 1.995x, passed |
| 2 | 00607fc6f8c84c77 (check) | 304a4117... (benchF0b) | 420001+ | 81 / 84 / 89 | cand2 donor donor2 cand | 15.0 / 15.7 / 26.9 / 3.7 | 1.994x, passed |

ABCHK: every arm 24 of 24 PASS in both runs (48 judge-rule checks per arm). ABEND at 55 s in
both; no ABSTOP. Logs, JSON and profiles: `muse_decode_layer/profiles/benchF0a.20abe6291eb04ed6.*`,
`benchF0b.00607fc6f8c84c77.*`. The seed ranges were unused in this log before.

### Per case (us, medians, n = 24 per arm)

| case | run 1 R1 / R2 / D1 / D2 | run 1 paired D - R (four cross) | run 2 R1 / R2 / D1 / D2 | run 2 paired D - R (four cross) |
|---|---|---|---|---|
| d1s | 87.6 / 92.8 / 83.0 / 81.6 | -6.2 -6.3 -10.9 -11.2 | 84.3 / 79.0 / 73.6 / 74.4 | -10.6 -10.7 -6.2 -5.8 |
| d1f | 95.2 / 99.6 / 88.4 / 88.1 | -8.4 -8.3 -12.9 -13.2 | 100.2 / 96.5 / 87.0 / 86.7 | -13.6 -13.5 -9.0 -9.0 |
| v1s | 133.0 / 132.8 / 118.9 / 120.4 | -14.1 -13.2 -12.7 -11.6 | 136.7 / 135.5 / 123.0 / 122.8 | -13.2 -14.4 -12.6 -12.8 |
| v1f | 128.8 / 128.4 / 114.4 / 114.3 | -14.3 -14.4 -13.7 -14.1 | 125.9 / 126.3 / 110.7 / 110.6 | -15.2 -15.8 -14.9 -15.1 |
| d16s | 153.4 / 152.5 / 140.6 / 140.5 | -12.8 -12.3 -12.9 -12.1 | 146.1 / 145.6 / 133.3 / 133.0 | -12.1 -13.6 -11.9 -12.6 |
| d16f | 231.8 / 232.5 / 219.8 / 220.3 | -13.4 -13.2 -12.7 -12.3 | 233.3 / 231.6 / 217.9 / 218.3 | -14.2 -13.7 -13.3 -12.8 |
| d32s | 174.4 / 171.5 / 153.6 / 154.6 | -20.4 -20.2 -18.5 -18.0 | 174.6 / 175.3 / 154.8 / 155.3 | -19.3 -19.4 -19.6 -19.7 |
| d32f | 341.8 / 342.8 / 320.6 / 322.0 | -21.8 -21.0 -21.6 -21.6 | 343.5 / 344.0 / 321.8 / 322.5 | -22.4 -23.0 -22.4 -22.1 |
| v16s | 295.0 / 293.3 / 272.9 / 275.6 | -20.4 -20.9 -20.3 -20.8 | 286.2 / 279.1 / 262.5 / 264.0 | -20.9 -21.1 -16.0 -16.2 |
| v16f | 383.1 / 379.9 / 355.2 / 354.6 | -24.7 -26.8 -23.9 -25.0 | 394.7 / 388.9 / 372.7 / 372.4 | -22.5 -20.3 -17.0 -12.7 |
| v32s | 362.3 / 362.0 / 330.4 / 328.2 | -34.8 -33.5 -32.8 -32.6 | 367.0 / 362.2 / 328.8 / 328.8 | -36.2 -37.5 -32.5 -33.7 |
| v32f | 515.1 / 512.5 / 492.9 / 495.5 | -19.4 -20.3 -17.2 -16.6 | 521.6 / 514.1 / 494.4 / 490.2 | -27.7 -30.8 -22.4 -24.1 |

Cross order: D1-R1, D2-R1, D1-R2, D2-R2. Every one of the 96 paired medians is negative.

### Weighted layer score (load C = 46 t_sliding + 16 t_full; score ratio = geometric mean of the six load-time ratios R / D)

| load | run 1 pooled R/D | run 1 nulls R2/R1, D2/D1 | run 2 pooled R/D | run 2 nulls |
|---|---:|---|---:|---|
| d1 | 1.0976 | 1.0563, 0.9870 | 1.1116 | 0.9445, 1.0059 |
| v1 | 1.1144 | 0.9982, 1.0090 | 1.1154 | 0.9942, 0.9983 |
| d16 | 1.0763 | 0.9971, 1.0003 | 1.0849 | 0.9952, 0.9995 |
| d32 | 1.0981 | 0.9913, 1.0056 | 1.1027 | 1.0030, 1.0025 |
| v16 | 1.0734 | 0.9935, 1.0061 | 1.0664 | 0.9786, 1.0036 |
| v32 | 1.0792 | 0.9978, 0.9975 | 1.0893 | 0.9865, 0.9969 |
| **six loads** | **1.0897** (cross 1.0863-1.0932) | 1.0054, 1.0009 | **1.0949** (cross 1.0852-1.1047) | 0.9835, 1.0011 |

ABDECIDE "donor faster" in both runs: all four cross ratios above 1 and |log pooled| (0.0859,
0.0907) above twice the larger null (0.0054, 0.0167). The large R null at d1 in both runs (5.6%,
5.5%) is the known first-prepared/last-prepared d1 arm effect (archive/docs/steering/2026-10-30-layer-bench-arm-bias.md),
which the four-way pooling covers. Projected donor layer score if the ratio carried to the judge:
2.014 x 1.09 = about 2.19-2.21 (hypothesis; the donor's own receipt reads 2.2058).

### Where the donor's microseconds come from (ABK, run 2, per-kernel median sums per step)

The bench's 16 KB log tail kept the ABK tables but lost most ABKN code-to-name lines (only
K81 `_graft_row_boundary`, K82 an nvjet cuBLASLt tile, K83 `_graft_route` survived), so the
kernel identities below come from launch order in the donor source and ours and are inferred,
not printed. Treat the stage split as provisional; the totals are measured.

| load | ours span / sum | donor span / sum | MoE ours / donor | front (o, boundary 1, pooler, router, route) ours / donor | qkv ours / donor |
|---|---|---|---|---|---|
| d1s | 85.9 / 85.7 | 74.0 / 80.0 | 24.9 / 25.3 | 37.7 / 30.4 | 10.8 / 11.3 |
| v1s | 134.3 / 130.5 | 119.7 / 129.5 | 66.2 / 68.1 | 36.8 / 35.4 | 12.6 / 10.9 |
| d16s | 147.6 / 144.2 | 134.7 / 143.4 | 67.1 / 68.9 | similar | 12.8 / 11.5 |
| d32s | 181.5 / 179.0 | 158.8 / 167.9 | 96.1 / 84.8 | similar | 14.0 / 12.7 |
| v16s | 273.5 / 267.7 | 248.8 / 244.5 | 164.3 / 143.1 | 31.0 / 39.4 | 16.6 / 19.8 |
| v32s | 359.1 / 353.8 | 324.5 / 320.5 | 220.9 / 199.3 | 22.7 / 31.1 (glue only) | 25.8 / 23.1 |

Mechanisms, each to be re-implemented by its owner (decision column per the user's rule):

| # | mechanism | stage | where it pays (measured) | donor file:line | decision |
|---|---|---|---|---|---|
| M-A | Routed experts at 32 / 128 / 256 tokens: the donor's persistent WGS MoE build with the PRMT block-scale decode and the KPERM3 K-permuted weight copy (scale pairs decoded once per stage), and at 17-32 tokens a separate `-DOKC_B4FIN=1` finish build | MoE variant | MoE kernel -11.3 us at d32s, -21.2 at v16s, -21.6 at v32s (similar on full layers); ties or loses 1.8-1.9 us at T = 8 / 16 | `_moe(M, state)` dispatch 13097-13116 (b4 key for 16 < M <= 32), `_main_defines` 6905, `_wgs_on` 6911, `_kperm_src` / `_kperm_codes` / `_kperm_rows` 6851-6880, `_wgs_copy` 6882, `_ma_prepare` 9883 | re-implement mechanism M-A (MoE owner; the standalone F-MOE-0 comparison should attribute it) |
| M-B | PDL chaining at T <= 32 across every glue and projection launch, each kernel issuing its first weight stages before `griddepcontrol.wait` and triggering dependents after its main loop, so the step's kernels overlap | fusion / PDL chaining | donor sum - span = 6.0 (d1s), 9.8 (v1s), 8.7 (d16s), 9.1 (d32s) us; ours is 0 or negative (span above sum by up to 3.8 us at v1s) | `PDL = True`, `PDL_MAX_T = 32`, `_pdl` 87-96; `BFP_PDL_N = 2`, `_bfp_pdl` 9852-9858; sk2 pre-wait weight stream 7853-7858, wait 7885, trigger 8072; Triton glue `launch_pdl=True` with `gdc_launch` / `gdc_wait` (e.g. 9963-9965, 11841-11846) | re-implement mechanism M-B (our M1 / PORT_SK_PDL family is the same idea on fewer kernels; the donor result says the whole-chain version is worth about 6-10 us per small step) |
| M-C | T = 1 front by deferred normalization: the o projection's epilogue forms boundary 1 (alpha s + beta P) and writes the router input in FP32 plus the pooler input in BF16 and per-CTA partial sums; one fused pooler + FP32-router sk2 launch; a route kernel that also forms the residual carry | front / norm, fusion | front 37.7 -> 30.4 us at d1s (inferred mapping) | `GLE_T`, `GLE_FRT`, `GLE_RC` and design comment 13227-13241; `_gle_source` 13246; `_gle_route` 13577; `_gle_frt` 13651; `_gle_after_attention` 13690 | re-implement mechanism M-C only with the ordering repair below (finding E2) |
| M-D | 256-token QKV through a pinned cuBLASLt plan | projection route | qkv -2.7 us at v32s; +3.2 at v16s | `_GRAFT_QKV256` 11986, `_graft_lt_plan` 11992 | inconclusive (mixed sign) |
| - | v16 / v32 boundaries (`_graft_row_boundary` x2, 13.9 us each at v32s) | front / norm | donor slower by about 8 us of glue at v16s / v32s | 11786 | keep ours |
| - | attention | attention plan | q_len 8 sliding: donor 8.2 against ours 6.9 at v1s; other cases within 1.5 us | `_decode_attention7` 11051 | keep ours |

### Eligibility review (static, bounded; not a complete ordering audit of the 13,887-line file)

- E1, precision, q_len 8 attention, kv_scale other than 1 (open). The donor's own comment at
  lines 11136-11137 records that its q_len 8 attention path "rounds coarser than FA3" on the
  kv_scale = 1/64 test (1.41x SGLang's error), and says FA3 keeps every q_len 8 shape; the code
  at 11143 (`agent_attn = state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)`)
  sends every supported q_len 8 shape to that kernel, so the comment is stale against the code.
  TASK.md allows FP8 E4M3 queries "and no coarser". Scored cases have kv_scale 1, so the timing
  is unaffected; we did not reproduce the claim. Not a mechanism we would copy (M-A to M-C do
  not touch attention).
- E2, asynchronous ordering, T = 1 `_gle_route` (blocks M-C as written). Every program of
  `_gle_route` (13577-13650) reads `PART` (the o epilogue's partial sums) before `_gdc_wait()`,
  and the carry programs (`kind > PARTS`) read `PO` (the o output) and `PART` and never wait.
  The kernel is a PDL secondary of the fused pooler `gle_pr`, not of `gle_o`, which wrote those
  buffers. In practice `gle_pr` triggers dependents only after its own wait (sk2 source 7885
  before 8072), so `gle_o` has completed when `_gle_route` starts; but the PTX ISA makes a
  prerequisite grid's memory visible only to a grid that executes `griddepcontrol.wait`, and only
  for its own prerequisites. The comment's safety argument (13581-13582: "the FP32 router is a
  plain launch") describes the `GLE_FRT = False` path, not the shipped `GLE_FRT = True` one.
  Repair for a re-implementation: read `PART` and `PO` after the wait in every program,
  including the carry programs, and measure that version.
- E3, lifetimes (low). `_workspace7` (3725-3737) and `_cluster_bufs7` (3844-3856) replace their
  module-level buffers when a larger shape arrives, which frees memory an earlier captured graph
  may still reference. `_attn_selftest` (11090) sizes them for the scored shapes, so only an
  unscored larger shape after a capture could hit it. Do not copy the grow-on-demand pattern.
- E4, data-derived caching: none found. Module-level caches hold extensions, per-device scratch
  and the T = 1 norm-weight pointer `_GLE_W_CUR` (13866, set from state weights per call);
  `fused_part` (9458-9462, 13660-13662) is per-state scratch allocated on first use in prepare's
  tuner. The compose tuner (`_policy_parent_prepare` wrapper, ~12315-12347) times synthetic
  inputs only; `_tune` restores the router bias it perturbs (13075-13080).
- E5, router precision: conforming as read. T <= 8: FP32 GEMV on FP32 weights with the FP32
  router input (`_router_fp32_gemv`, `_gle_frt` with `x32` and `w32`); T = 16 / 32: BF16 hi + lo
  router pair with FP32 accumulation in the fused sk2 (9449-9455); T = 128 / 256: hi + lo rows of
  `st["pr"]` (10893) summed in FP32 by `_graft_route` (11861; also 12768, 12839). TASK.md allows hi + lo above 8
  tokens.
- E6, PDL completion: the Triton glue kernels (`_gemm_splitk`, `_qk_prep`, `_graft_route`) have
  no early return before their wait; sk2 waits in every thread (7885). The CUDA attention q8
  staging fences (`fence_proxy_async` after the generic q8s stores, 2002). Not audited: the
  attention and MoE CUDA sources in full (inherited from vjeux and the donor's MoE entry).

### Decision

Our release stays the layer entry, and nothing donor-derived is a submission candidate. The
donor layer is faster by 8.97% / 9.49% pooled in two independent same-process runs, beyond both
nulls, on every case. Re-implement M-A (MoE at 32-256 tokens) and M-B (whole-chain PDL with
pre-wait weight streaming) in our own code with attribution comments; M-C only with the E2
ordering repair; M-D inconclusive; keep our boundaries at v16 / v32 and our attention. The
per-stage attribution is inferred from launch order because the log tail lost the kernel names;
a bench that prints ABKN first (or names the kernels in ABK) would make it a measurement.

## Ruled out

The losers of the W0 checks (PDL across our own chain, the two L2 prefetch forms, the hi/lo mma router against the one-GEMM front) are in `docs/ruled-out.md` under muse_decode_layer.

### Scored: layer_m1rt51s, three runs (2026-10-31)

layer_m1rt51s.py from 0bc16c6 (SHA256 f1235356...; M1 on the stable rule, MoE mode 51 at
32/128/256, stable tuner; no workspace pick, no donor code), submitted by the coordinator after
four-arm benches 3aee22ba4a404c8a (pooled 1.0052) and c8d1e57261dd4901 (pooled 1.0106) both read
cand faster than release_s. Runs 2 and 3 differ only by a trailing `# repeat N` comment. All
passed. The best, 2.0148x, is a hair above the previous entry 2.0143x, inside run-to-run noise.

| job | score (95% CI) |
|---|---|
| ad42a066bd3a4e01 | 2.0069x (1.977-2.037) |
| 138ae1a402934c44 | 2.0016x (1.972-2.032) |
| 0f58d5d8223f45af | 2.0148x (1.985-2.045) |

Receipts: muse_decode_layer/profiles/scored.<job>.json.

### Scored: layer_m1rt51sc1g, two runs (2026-10-31)

layer_m1rt51sc1g.py from 2840157 (SHA256 a00a0efa...; layer_m1rt51s + projection C1 + the route
side-stream self-test on the fresh-input gate per the user's 2026-10-31 decision), submitted by the
coordinator after single-file check 2edb120747cb4611 (2.005x, route side gate passing at all six
token counts, no fallback). Run 2 differs only by a trailing `# repeat 2` comment. Both passed.

| job | score (95% CI) |
|---|---|
| dc25fef4d65642a9 | 2.0063x (1.976-2.036) |
| 7b84ed4346d5497b | 2.0187x (1.988-2.049) |

Board entry moves from 0f58d5d8223f45af (2.0148x) to 7b84ed4346d5497b, inside run-to-run noise.
Receipts: muse_decode_layer/profiles/scored.<job>.json.

### Recovered after Windows login: d1/v1 donor comparison `0778baa78efa4aad`

Historical check completed October31 UTC, recovered October31 PDT; not a new dispatch.
Verified board/job from its authenticated receipt, passed13 tests,2.013476x (95%CI
1.983274–2.043678) on the candidate judged path. Source build `layer_m1rt51sc1g.py`, source
SHA a00a0efa..., sent comment-stripped b7dd9976...; donor exact8f17613b... at pinned6cae49a.
Receipt/log saved as `muse_decode_layer/profiles/recovered.0778baa78efa4aad.{json,log}`.

Only d1/v1,24 observations per arm. Candidate independent copies differ: cand M1
`{8:o+front,16:o,32:o+front}`, cand2 rejects M1 on the fresh-input gate and falls back to{}.
MoE/front/attention-boundary policies agree; W4 picks differ. Donor picks also differ at64/128,
outside timed loads. Therefore this is not a fixed-M1 comparison or a six-load layer gain.

| Case | cand / cand2 us | donor / donor2 us | donor-cand paired us |
|---|---:|---:|---:|
| d1s |83.4 /83.8 |77.6 /77.6 |-5.6 /-5.9 |
| d1f |91.5 /92.0 |83.1 /83.6 |-8.4 /-8.0 |
| v1s |126.3 /129.2 |120.4 /120.5 |-6.4 /-6.0 |
| v1f |143.7 /146.0 |132.4 /132.3 |-12.3 /-11.7 |

Pooled two-load ratio1.0768, crosses1.0693–1.0843; nulls1.0132/1.0009. Bench declares
donor faster beyond its null rule, with selection asymmetry above. d1 policies do not use M1;
its donor advantage survives both independent candidate arms. Named kernel profile shows
o+boundary cand5.5+5.9 vs donor fused6.8 at d1s, but the secondary boundary includes a wait;
the apparent4.6us sum gap is not exposed-time savings. Current front already overlaps router.
This receipt satisfies the previously proposed d1/v1 attribution pass, with M1 qualification;
do not dispatch that exact measurement again as missing. Keep donor measurement-only.

### October31 PDT campaign: stable attention-only port, check pending

Check1c1fcf1d49924050, problem confirmed muse_decode_layer. Candidate
`cand/layer_attn31s.py`, SHA256585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b,
copies layer_m1rt51sc1g then replaces only _ONE_SRC through dl_one_inline.py using current
owned attn_one.cu (HMRG1,PRQ31). No workspace pick. Attention Params ABI matches explicitly,
two dependency waits preserved, host PDL flags unchanged. Revisit trigger: stable-pick build
and removal of workspace-placement pick from old layer_hp. Await official correctness, then
paired whole-layer timing; no score or gain inferred yet. Status snapshot is under
archive/docs/steering/evidence/2026-10-31-top10-campaign/1c1fcf1d49924050.json.

Result1c1fcf1d49924050: pass13tests,2.048010x CI2.017290–2.078730, candidate-only.
Paired four-arm confirmation dispatchedc0402e4c92bc4b20, all6loads, twoheldoutinputs,
8rounds, independentprepares, seed610001/order601, deadline280s. Builder dl_make_bench_pub.py
with other=inc, dup=own, cand-strip=1; exact sourcepins printed and candidate585d46e2...
unchanged. Do not credit standalone check gain until this paired gate completes.

Scored control repeat36e57280ff1f4c63: pass13tests,2.014780x CI1.984558–2.045002,
M1 open (8-token whole-step saves3.68us), mode51 at32/128/256, no reported preparation
failures. Below best2.018705x within ordinary run variance. Rawreceipt retained in profiles/.

Second authorized scored control repeat43026195f8c84ffb dispatched: same frozen
layer_best.py plus comment-only repeat2, no new kernel change. Pending at dispatch;
records fresh run variation while paired attention-port c0402e4c92bc4b20 is unresolved.

Completed four-armc0402e4c92bc4b20: all24checks per arm pass; all6loads covered,
fourprepares finish258s. Pooled inc/cand1.0042, crosses1.0036–1.0048, nulls0.9995/1.0007;
registered decision candfaster. Perload pooled d1/d16/d32/v1/v16/v32 =
1.0102/1.0022/1.0074/1.0020/1.0026/1.0008. M1 policy differs at32tokens: inc o+front,
otherthree o. Thus this measures the complete prepared candidate, not isolated attention
instruction latency. Candidate585d46e2... submitted as8f2780d7382047e6 after this gate.
Independent repeat confirmation9414182118414889 dispatched with seeds620001/order701;
same source,8rounds,6loads,dupown,deadline280s. First scored result pending at dispatch.

Control repeat43026195f8c84ffb completed13tests pass,2.009960x CI1.979811–2.040110.
No new speedup; keep control run variation visible. Receipts/profile records retained.

Scored attentionport8f2780d7382047e6 passed13tests at2.022870x (1.992527–2.053214).
Official board selects this job at currentread; release-manifest.md records source, SHA and
policy caveats, including3unselected/report-only diagnostic issues. Hotspot table appended
with SGLang baselineprofile7d42771c3384449d. Second paired9414182118414889 stillpending.

## October31 23:07 PDT repeatability qualification

9414182118414889, independent second four-arm attentionport:24checks perarm,6loads,
pooled1.0015,crosses1.0006–1.0024,maxlognull0.0011; registeredgate NOT RESOLVED.
All four M1policies{} after failclosed gate; original candidate16x1s gate reports rel0.14
with4order1events. Different prepared regime from c0402e4c92bc4b20; no repeated gain claim.
273ff292536e4429 matched-M1 diagnostic passes13official tests but early diagnostic lines
lost from logtail; observation failure,no math conclusion. Fixedtail03bb478f8b0b4629 passes
13tests/control2.035941422x,95%CI2.005402301–2.066480543. Matchedoff/off,on/off,on/on gates
at16x1sliding and1x8sliding allpass/order1zero; prior0.14 event not reproduced or explained.
Judged M1forcedoff; never promote diagnostic controlscore. Thresholds/CUDA unchanged.
Helper tools/dl_make_m1_matched_diag.py; raw receipt evidence retained in profiles/.

## November1 current-layer/SGLang all12 dependency timeline

Source097c38f, generated984090B SHA256
5affd5cc9dd480af2a49f89db678cc989dfcd593fdb011a459330e618151abe9.
Check41788b39fb824179 dispatched and confirmedrunning; diagnostic-only, never publicsubmit.
All12 scoredshapes, independent owned/SGLang inputs/state, actualsoft/agenticdistribution,
referenceargmax busiestEPgroup, fivecoldCUDAevent replays andthreeisolatedprofiledreplays/arm.
Productionkernels/CUDA/policiesunchanged. FourCPUtests pass plusindependentreview.
Complete24rows requiredwithin280s and11KBlogbudget; no score/improvementclaimedpendingresult.

41788b39fb824179 terminal prepare failure: all24rows collected but encoded12192B exceeds
11000Bguard; no timeline values printed/retained, no officialscore/correctnessresult. Raw
receipt/log in profiles/timeline.41788b39fb824179.*. Corrected hosttransport records trace
coordinates at explicit0.1us ratherthan0.001us; aggregate span/union remain rawresolution,
cold eventtimes0.001us, fullnames/streams/order retained. FourCPUtests pass; independent
review requiredbefore one correctedcheck. ProductionCUDA/kernel/input/policy unchanged.

Correctedcheck01183c2992eb4153 dispatched/source d327abd and confirmedrunning.
Payload984798B SHA256dae91104f666fbfc323145dda56dfad39593e1631c12624f39b056b9bc26c66d.
Varint delta/duration/name-stream transport, explicit0.1us eventcoordinates; unchanged
kernel/workload. Independent56events×24row fixture10264B<11000, exactroundtrip;
WSL4CPUtests pass. No fourth/further observation retry without scope review ifthisfails.

01183c2992eb4153 completedPASS13,checksOK/no flags, unchangedcontrol2.017452507x
CI1.987190720–2.047714295;prepare166.968s. All24tracesretained. Rawreceipt/log+normalized
profile JSON/text committed. Decoded evidence andofficial12shapecomparison in
docs/steering/evidence/2026-11-01-fusion-audit/. CandidateM1now{8:o+front,16:o,32:o+front};
freshscore/per-shape changes notisolatedimprovement. T1topkfinish34.8/42.4 beforepoolnorm
46.6/53.6us, noexposedroutertailinthesesamples; uncovered1.056–3.263us notSMutilization.
BUG: reportedactualrouting counts were readfromcapturedoutputBEFOREfirstreplay, invalid;
referencegroup/local_startvalid, actualtimedgraphs/officialtestsunaffected. Decoder marks
routing_metadata_validFalse. Hosthelperfixedreplay+syncbeforeinspection;5CPUtestswith
negativecapture-without-replaycontrolpass. NoadditionalGPUchecksolelymetadata. No submit.

2026-11-01 | 04f0339fa9f24807 | beedbca | CUPTI availability on official SGLang baseline | PASS 0.997335x (95% CI 0.982375–1.012295). Size 9184 bytes, image read denied with CUPTI_ERROR_INSUFFICIENT_PRIVILEGES. Baseline timing only. Full receipt and CLI profiles saved.
2026-11-01 | 1023d9b0dcaa4c75 | 40b7b01 | Four-arm short-boundary experiment, rank zero computes live branch RMS. Twelve rounds, d1/v1, independent duplicate prepares. Pending.
1023d9b0dcaa4c75 completed, official 13/13, 2.015410x CI 1.985179–2.045641. Four arms 6/6. D1 pooled 0.9956, null 1.0093. V1f missing at 280 s. V1s paired -0.4 to -0.9 us with different front paths. No promotion.
2026-11-01 | a0da77c5f3c647c3 | baseline 32f1970 plus tools/dl_make_matched_short_boundary.py | identical prepared state, boundary and whole layer, four graphs per case and two seeds. Pending.
a0da77c5f3c647c3 PASS13, control 2.017944x CI 1.987675–2.048213. Complete affected-load gate passes each seed on d1/v1. Fresh six-load confirm 35f2b5d66f51465b dispatched.
35f2b5d66f51465b PASS13 control 2.022121x CI 1.991789–2.052452. All 24 rows. Six-load 1.003428 vs threshold 1.001450 and 1.001829 vs 1.003343. Both affected loads faster, second seed misses aggregate gate. Preserve experiment for combination.
2026-11-01 | 59cc1343940f4bcb | 68c769d plus tools/dl_make_concat_layout.py | matched physical weight layout for concat-r at 128/256 tokens, FP64 and whole-layer comparisons. Pending.
59cc1343940f4bcb passed 13 tests, control 2.015711x CI 1.985476–2.045947. Contiguous K-major concatenated weight slower in every GEMM and whole-layer comparison, both seeds and sizes. Retain original.
2026-11-01 | 3a5f879df0d44825 | off-board external comparator keell0renz at 8e30e1be, source 1ec189dc... | exact file check for current per-shape attribution. Never a public submission.
3a5f879df0d44825 PASS13, exact external layer 2.248790532x CI 2.215059–2.282522. Independent from owned best, 12 lower shape medians. User authorized attributed fork after audits. Upstream comparison alone is not an accepted release.

### 2026-11-01: scoped standalone Lt pins in selected decode layer

Root authorizes one layer check; no public submission, standalone check or prefill work.
Baseline exact `layer_attn31s.py` SHA585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b.
Source audit: `_proj`20975-20991 has no direct Lt route; wide128 tuner can override _gemm,
256 falls through to `_bf16_gemm_dispatch_impl`. Prior M-D row is decoded donor attribution,
not an owned port trial. C1 port explicitly excluded pooler128 because its front is concat.

`tools/dl_proj_lt_port.py` ports frozen standalone LT_CPP (FP32 compute, BF16 input/output)
and three existing validated pins only: QKV256 L0 tile124/cluster2; unpooler128 L3
tile18/custom2/cluster5; unpooler256 L1 tile20/custom4/cluster3. O128 was near zero in
standalone trials, O256 marginal, so both excluded. No fused pooler/router route touched.
Candidate SHAda7f00fbb5f6f7e3cea296abce6f450ad864e99f5ddd2e55959ca6ae7ece638d.
Original layer executable/CUDA AST unchanged except appended wrapper; only Python
docstrings become pass for payload size. Actual CPU dispatch gates verify shape, full
output, contiguous/aligned BF16, fallback and baseline without the new state key.

Each route gets its own plan/32MiB workspace and exact-layout weight copy. Prepare requires
accepted pin, full FP64 random error<.005 and row<.01, exact onehot, three fresh graph inputs.
Original projection and every other route remain fallback. No heuristic search or math change.
`tools/dl_proj_lt_probe.py` wraps the established four-arm layer benchmark, independent
baseline/candidate prepares and shared-state duplicate controls (limitation: duplicates do
not sample independent tuner/build noise). v16/v32, sliding/full, busiest+second distributions,
fresh seeds1101433+,12 randomized cold rounds; all32 whole-layer reference checks required.
Busiest uses actual argmax tie semantics. Fail/skip evidence prints before failclosed gates.
Payload966038B; only two affected loads timed, so no full-score claim from their geometricmean.
Acceptance: whole-layer improvement beyond both nulls in all four cross-ratios on affected
loads, no consistent context regression, all FP64/fresh/reference gates and policy attribution
complete. Otherwise retain baseline and record loser; no automatic retry. Layer quota before
dispatch12used/8left, standalone19used/1left untouched. Job pending source breadcrumb.

Terminal **1e3349e377234e89**, source1ecf680: DONE/pass13, control-path score2.0260216001,
95%CI[1.9956312761,2.0588187683]. Both prepares150s, comparisoncomplete152s, all32
reference checks pass. Three exact pins accepted; FP64rel.001660–.001662,row.001714–.001751,
exactprobe and3freshgraphs pass. v16 pooled1.0049 (fourcross1.0043–1.0056,null1.0011/1.0002),
v32 pooled1.0129 (cross1.0118–1.0140,null1.0017/.9995); two-load pooled1.0089 beyond2×null.
Paired4cross latencies: v16s-1.2..-2.0us, v16f-1.1..-2.2,v32s-4.1..-4.4,v32f-4.2..-4.8.
Affected128/256 MoE/front/abnd/wide policies match. Unaffectedsmall-M front/M1/W4 differ
between independentprepares; no full-six-load extrapolation from this receipt. Root requests
common-prepared-policy all-six-load combined Lt+short-boundary follow-up; no public submit yet.
Raw receipt and normalizedprofiles archived.

Combined follow-up authorized by root: `tools/dl_proj_lt_matched.py` uses root's exact
`dl_short_rms_owner.candidate` for pure combined source and its matched-short duplication
helper for diagnostic source. Pure SHA851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a,
927545B; probe SHAa6411cbdc528ef0b39d009f008e85503d2caa603e0d273d6e19b829a029ec736,
957176B. One original prepare, common policies/compiled boundary extension/mutable scratch;
baseline shallow-state clone removes Lt entries, candidate routes owner boundary and Lt.
Private arm input/cache/output tensors, synchronized sequential graph replays. Shared-state
controls do not measure independent preparation/build variability; clean pure gate follows
only if this comparison wins. No production correctness/tuning gates reduced.

Before timing: all3 Lt fullFP64/exactprobe/3freshgraph gates; T1/T8 old/owner boundary
bitwise ABI on two fresh inputs. All6loads×2contexts×2fresh seeds,4arms,96 full judge-equivalent
FP32/FP8-relative reference gates including routing/newcache;40randomized cold rounds.
Reset stream/attn_input before each replay; busiestgroup actualargmax. Score independently
per seed with46sliding+16full per load, geometricmean over6loads. Both seeds must have
all4cross ratios>1 and pooledlog improvement>2×larger nullabslog. No aggregate forecast.
CPU actual score helper winner/loser and dual-route parse/count gates pass. Deadline275s
fails closed on missing coverage; no public submit. Root authorizes a clean exact pure
file check only after all these gates pass. Standalone quota untouched.

Terminal **828974e3bbcc4d9b**, sourcedd1a4b6: DONE/pass13; diagnostic control2.0211224939,
95%CI[1.9908056565,2.0514393313], not algorithmcredit. Commonpolicy comparison complete188.9s,
96/96 reference checks, three Lt gates and T1/T8 twofresh bitwise boundaryABI pass.
Seed0 sixload cross[1.0027005,1.0038941,1.0029371,1.0041309],pooled1.0034154,
null[1.0002359,.9988111]; seed1cross[1.0032824,1.0045025,1.0027918,1.0040113],
pooled1.0036468,null[.999511,.9987853]. Both preregistered2×null gates pass.
Raw per-shape/seed4arm medians retained in receipt; these are ~0.34–0.36% paired aggregate
gains, not a top50 or top10 promise. Mutable scratch/extension sharing qualification remains.
Consequently root's conditional cleancheck authorized: **47bec92021ef4bbe**, exact frozen pure
SHA851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a, pending.
No public submission. Original selected baseline unchanged; purecandidate in persistent scratch.
2026-11-01 | cb71acec29d141c6 | c80ab0e pure candidate muse_decode_layer/cand/layer_lt_boundary.py | scored after clean 47bec92021ef4bbe PASS13 (2.010247860×, 95% CI 1.980094142–2.040401578) and all-six paired 828974e3bbcc4d9b pass. Compared exact bytes with checked artifact, SHA256 851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a. Previous board best 8f2780d7382047e6 (2.022870479×). Pending.

Clean exact **47bec92021ef4bbe** DONE/pass13, pureSHA851c4695... above;
score2.01024785996,95%CI[1.98009414206,2.04040157786]. Production manifest failures0;
all3Lt fullFP64/probe/fresh gates pass (rel.001660–.001663,row.001696–.001735).
Raw receipt and normalizedprofiles retained. Root independently verified bytes and, under
user's standing authorization, publicly submitted exact pure as **cb71acec29d141c6**;
scored receipt root-owned. Paired fullsixload result remains the algorithm evidence,
not cross-run comparison of this noisy headline with other candidates.
cb71acec29d141c6 PASS13, 2.043438469x (95% CI 2.012786892–2.074090046), selected by public board at rank 373. Old best 2.022870479x. Owned incumbent line. Added SGLang paired hotspot table.
2026-11-01 | caad524754b54680 | Family G1a complete attributed file muse_decode_layer/cand/g1a_owned_experts.py SHA256 63aa3a7a8fd5e784011a60869050f920f0948514006ee513cdaf419b818001c4 | repaired layer and owned MoE, ordinary ordered launches. Pending.
caad524754b54680 | 0329226 | H200 setup failed in _gle_source before correctness/timing. Source substitution anchor lost during packaging. No score, CI or profile timings. JSON and log archived.
2026-11-01 | 606f47f251f34873 | 76a8d09, corrected G1a owned expert fork, SHA256 f564944e53db0ddf94d2af3453571d0c34878ddd329bcf647a5bc40134cfac28 | Exact payload CUDA generator and Linux/Windows reproduction pass. Off-board check pending.
606f47f251f34873 PASS13, 2.010975134× (95% CI 1.980810507–2.041139761), 12 shapes/profiles. Selected owned experts, eight Lt plans. Keep as control. Public incumbent stays cb71acec29d141c6.
2026-11-01 | ffe303a12a954304 | G1b repaired native fork, source patch 710f2b8, file SHA256 7935ff411d5e7ec1af41b25c7fecbff3524c4311efa65b6635b05e1d5bd41025 | Same layer paths as G1a, native precision/order/scale repairs, explicit fallback. Off-board check pending.
ffe303a12a954304 | 90cff63 | native SC6 selected, seven startup probes completed. Official setup FAIL, 353.520482 s > 300 s. No score, CI, tests, or shape/kernel profile. Raw receipt and log archived.
2026-11-01 | 736a5cbc977f4ad4 | 148ecb5, native G1b2 prep-only revision (SHA256 258c848c662149f9c6ac178039f179de6f0b0ec4cb0b503451d2974c7a7a8dfc) | Five parallel build keys. Owned/Marlin fallback prepared only if selected. Pending.
736a5cbc977f4ad4 PASS13, checks OK and flags empty, 2.065558557× (95% CI 2.034575178–2.096541935). Native SC6 selected, 119.270751 s prepare, 12 benchmarks and 12 profiles. Fastest measured repaired G. Hold public release for zero-local WGS metadata fix, fallback residency and independent matched validation.

2026-11-01 | 607eb52bdaf24160 | PUBLIC whole-layer measurement, source 0ce0b23f5342 muse_decode_layer/cand/g1b4_stage_release.py | 869076 bytes SHA256 ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. Shippable attributed G1b4 with stage-end activation retirement, one timed kernel_fn. Private, fail-closed startup self-tests compare seven sizes and fresh/empty routing with Marlin and check scratch guards. Clean prefix/payload and Windows+Linux hash verified. Baseline public cb71acec29d141c6 at 2.043438469x. User explicitly authorized a public measurement before the free quota reset. Pending.

607eb52bdaf24160 terminal PUBLIC PASS 13/13, 12 benchmarks and 12 profiles, no flags. Native G1b4 with stage retirement, 56 individually verified private replays and 127.759508348 s prepare. Scored 2.126082676x (95% CI 2.094191436-2.157973917), prior public best 2.043438469x. Six-load score independently reconstructed.

2026-11-01 | 74a29e0384324952 | PUBLIC G1b4 control qualification, source a2de34ea15a4 muse_decode_layer/cand/g1b4_control_release.py, SHA256 706ca85d5c048ae4d9c1e3919d2f0895cda31561f6ed2f39ad7351e07621daa1 (870877 bytes). Exact pure G1b4 prefix. Same startup checks plus 2000 queued replays each at 32/33/128/256 with private scratch, empty and restored occupied outputs verified. Single timed implementation. Separate scored points do not substitute for paired scheduling evidence. Pending.

74a29e0384324952 terminal PUBLIC G1b4 control PASS 13/13, all 12 benchmark/profile cells and no flags. Native, 56 verified replays plus 2000 queued each at 32/33/128/256 with remote and restored outputs checked, 128.576794633 s prepare. Score 2.052870208x, 95% CI 2.022077155-2.083663261; stage run 607eb52bdaf24160 remains public best at 2.126082676x, rank 303. Different fresh inputs and machines may contribute to the gap. Both official scores reconstructed.

2026-11-01 | 27b2ed49fc6344ec | PUBLIC scored G forward candidate 3191783d2cd5, muse_decode_layer/cand/g1b4_stage_flush_release.py, 22fc41e1a3213a198bf6984e9010c47d22fd2febef53f57414d2ef41d1a05645 (868880 bytes). Relative to 607eb52bdaf24160, only the WGS pending-unit flush moves from CI4 to after first-stage drain/X release. Same startup validator and timed layer, placement proof independently reviewed. Expected latency benefit unmeasured. Prior public best 2.126082676x at rank 303. Pending.

27b2ed49fc6344ec terminal PUBLIC PASS 13/13, 12 timings and 12 profiles, no flags. Native expert backend, eight Lt plans, 56 private replays, 124.675400 s preparation. Scored 2.102308440x with 95% CI 2.070773814–2.133843067. The incumbent 607eb52bdaf24160 at 2.126082676x remains the board entry, now rank 304 overall and on the layer. Targeted weighted decode-32 and verify-16 were 1.84% and 2.13% slower, verify-32 was 0.85% faster. Independent workloads limit attribution. Close this exact publication placement for public retries. [Twelve measured cells](../archive/docs/steering/evidence/2026-11-01-g-public/stage-shapes.md#post-stage-publication).

2026-11-01 | 942ebdec400d4e68 | PUBLIC short pooler and FP32 router overlap, source 0b63a197aaf2, muse_decode_layer/cand/g1b4_short_overlap_release.py | 878527 bytes, SHA256 5b5ec131f1d9d427d04817bf1489903a9fabfa4a35c21eda05a15e6c5ae9a7d4. The single timed implementation forks and rejoins the normal stream at T1/T8 for eligible existing SK2 poolers. Other paths retain the incumbent implementation. Independent final source review cleared stream ownership, scratch lifetimes and exact startup coverage. Local RTX 5070 ran the actual router and route kernel with a substitute BF16 pooler. The H200 job must pass the real pooler, all 32 startup replay checks, incumbent native checks and all judge cases. Incumbent at dispatch 2.126082676x. Pending.

942ebdec400d4e68 terminal PUBLIC PASS 13/13 at 2.095900316x (95% CI 2.064461811–2.127338821), all 12 timings and profiles, no flags, 127.822270 s preparation. Native G, eight Lt plans, 56 expert replays and 32 split-front replays across T1/T8 with two owners each. The two targeted weighted medians were 83.051/113.549 µs versus 78.549/115.991 in the incumbent. Full score was 1.42% lower. Close public retries and retain 607eb52bdaf24160 at 2.126082676x. Its current rank is 305 on both boards. [Twelve cells and component activity](../archive/docs/steering/evidence/2026-11-01-g-public/stage-shapes.md#concurrent-short-pooler-and-router).

2026-11-01 | CPU/source-only, no GPU job | Fixed-mask verify production dispatcher and exact-source packaging audit. Default869076B is byte-identical607e. Complete verify32 singlecell1072104B; allfour1125548B; every nonzero mask exceeds1000000B. No latency/score/CI/resource-utilization measurement. Nine host tests plus independent source proofs; native/runtime gates separately documented. Incumbent607eb52bdaf24160 remains2.126082676x. [Source-size matrix and stop criteria](../archive/docs/steering/2026-11-01-verify-hybrid-packaging.md).

2026-11-01 | 1fad70bbe3854cbc | PUBLIC single-arm fixed-four-cell measurement, source43c6c09b848b8ab1c28c57ebbed455b6b76821e9, muse_decode_layer/cand/g1b4_owned_verify_release.py. 980787B SHA2564ad326c3451d1b9a7ba22e6c89a2b06ebd8f565c20300abd9fc0d3b0a35001bc. Comment-preserving representation now fits; separate reviewed ownedMoE51 cooperative/PDLoff/fatal-watchdog repair retained. One unchanged timed dispatcher: ownedverify16/32 sliding/full, G elsewhere/fallback. Startuprequires48model/8captures/24changed,40empty/8emptycaptures/32restored and3actualresourcevectors; root7host+2buildertests pass. Standing authorization permits expiring scored slots for measurement; private comparison is not submitted. Physical bytes=committedblob, freshquota2scored/0checks and no livejobs beforedispatch. OwnID wait pending; no score, CI, runtime/resource value or promotion yet. [Registration](../archive/docs/steering/2026-11-01-owned-verify-public-measurement.md).

1fad70bbe3854cbc retrieval update: official endpoint first reported running; subsequent same-ID status and jobs calls returnAPI530 during reported service maintenance. Bounded local statuswait ended after1100s with TimeoutExpired/no terminalJSON. This is NOT an observed H200 preparation/kernel timeout, correctness loss or scored loss. Source/job remain accepted, outcomeUNKNOWN; no duplicate submission or quota-refund assumption. Same existingresetreminder requires exact-ID reconciliation and live service/ledger/quota beforenewGPUdispatch. Raw service/preflight/leaderboard evidence retained; no profiles or latency invented. [Maintenance handoff](../archive/docs/steering/2026-11-01-maintenance-and-progress-handoff.md).

2026-11-01 | No official job; source/CPU/local compiler only | G Tiny/Tiny2 finalbc relaxed-strong-GPU RMW restoration, defaultOFF. Pure opt-in d33915c7/869551B; private check f0426f7160768de01997b271744542ae3442d0851a1c27e586ced322a8fc6864/906841B. Retainsallwriter/proxyfences, joins/acquire/epoch/BF16/coop/PDLoff; source/actualPTX proof and localTiny/Tiny2 both147reg/3bar/0stack/spill. Root11source/score/host+5reader tests pass, doublegenexact. Normaljudgedprivatepath607control, two variants/key/fourplacements,36medianrows/144normal/144freshfinal/32empty, fullsixrawevent>1%2xnull/allseeds+regression gate. No H200latency/score/CI/utilization or gain claim. Conditionalthirdprivatecheck afterexistingstage/serial-front and1fad/service reconciliation; oldowned-four-cellHOLD. [Inventory and handoff](../archive/docs/steering/2026-11-01-g0-607e-integration-gap.md).


## 2026-11-01 - native gamma OFF/ON local compiler gate (no job ID)

CPU compiler only; no GPU/official call, score/CI or GPU failure. Exact607 ba9e327c versus OFF e17901d5 and ON d6756bb5, 15 fixed CUDA12.8 sm90a build requests /30 bool entry instances across Tiny/Tiny2/Mid/B4/MAIN. Baseline-to-OFF normalized PTX/SASS/resources/bounds identical all5. ON MAIN both entries spill stores4->44B and loads32->68B (96regs/16barriers/8Bstack unchanged); other4 resource vectors unchanged. Status CLOSED_RESOURCE; no retry, short-only mask, format/cap sweep or speed claim. Full report/evidence: archive/docs/steering/2026-11-01-g-native-gamma-compile.md.

2026-11-01 steering update: [G integration-first plan](../archive/docs/steering/2026-11-01-g-minimal-delta-plan.md) supersedes the earlier stage/serial-front reset order. Goal/reset paused; historicalf042 and other private files HOLD pending common attention/preexpert qualification and new source/reader registration. Exact commonOFF opt-in871077B/4cbb13a9 has source/CPU gates only; no new official job, latency, score or CI. Latest21:38UTC exact1fad/jobs/quota retrievalAPI530; accepted outcomeUNKNOWN, no duplicate. GammaON compiler loss closes that arm and holds its factorial; shortRMW remains next eligible integration question after the common control.

## 2026-11-01 - G short paired-RNE source/compiler screen (no job ID)

One OFF representation, exact607 control vs19cfb4ee/870233B, no GPU/API/official run or score. CPU2PASS/524288rounding-classpairs. CUDA12.8 sm90a selectedTiny/Tiny2 each147regs/3bars/0stack/spills unchanged;12scalarF2F->6PACK_AB but8PRMT added. Tiny5272->5280static instructions (nonNOP5230->5233);Tiny25328->5328 (nonNOP5279->5281). Gate now waits for scaled up plus PRMT before EX2. CLOSED_DEPENDENCY_BUDGET, no harness/alternateunpack/sweep; no credible >=1% sixload budget. Conditional3us shortcuts are sizing only. Report2026-11-01-g-short-pair-rne.md.

| 2026-11-01 | source/CPU only; no job | G short scale-to-BF16 feasibility after aafc32d | No score | CLOSED_NO_EXACT_DEPENDENCY_REMOVAL. Scale multiply preexists G0; moving across required BF16 changes finite values (1.015625 vs 1.0078125). No variant, GPU or second compiler screen. Evidence: archive/docs/steering/2026-11-01-g-short-scale-feasibility.md. |

## 2026-11-01 - G single-arm source readiness and parallel screens (no job ID)

Evidence603ed5b: frozen common+shortRMW914572B/SHA97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7, one performance treatment. Root7 CPU generated-prepare facade tests pass8.050s; independent source/header/protocol reviews pass. Native/CUDA model, capture and scheduling are stand-ins in those tests. Actual H200 compilation/admission/math/replay/resources/setup/performance unmeasured; no latency, score, CI or utilization claim. The same fixed arm may qualify shared repairs inside prepare in an authorized measurement; no prior private1% win or separate common GPU job required. Goal/reset remain PAUSED. Latest22:23UTC status/jobs/quota probes hit15s local retrieval timeouts; accepted1fad outcomeUNKNOWN, no new/duplicate GPU dispatch. [Candidate/manifest/graph and stopping gates](../archive/docs/steering/2026-11-01-g-candidate-funnel-qualification.md).

Parallel source-only closures: expert→unpool PDL edge parked at necessary shared bound/missing exposed retirement budget (b384ec4); O-boundary last-owner proposal closed at source/score-budget screen without implementation (1eaf919). Neither is a GPU performance failure or a second candidate. Reopen only with the specific dependency/parallelism/exposure evidence in their reports; no blanket family prohibition.

2026-11-01 22:32UTC | Recovered own public1fad70bbe3854cbc terminal | statusfailed/resultinfra_error/refunded=true; no records/log/score/CI. Not a kernel correctness or performance loss; quota refund now explicit. User resumed exact RMW H200 measurement, prefill excluded. Live ledger no active jobs, layer2scored/0checks. Exact97935e7d/914572B regeneration/blob/dependency gates pass; registration in steering/2026-11-01-rmw-h200-resume.md. No older hybrid retry; new job ID recorded only after acceptance.

2026-11-01 | PUBLIC3fd1317143304bd5 | Fixed common+shortRMW, source registrationfaa1befebdf465395a954769ea8889b76fd8eb5d/exact97935e7d/914572B. Explicit user resume authorized one H200 measurement; prefill excluded. Immediate jobs/quota/source preflight repeated, no active jobs/two scored slots. Accepted correct problem and own ID; own-ID wait underway. No score/CI/pass/resource claim yet. Startup/official/actual-exercise/full-six gates and stop rules in steering/2026-11-01-rmw-h200-resume.md.

2026-11-01 22:55UTC | Existing3fd1317143304bd5 statusQUEUED, worker/started_at/result null; live layer1scored/0checks. Local1050s own-ID observation timed out, remote job remained queued, same-ID retrieval reattached (not resubmitted). No H200 timeout/failure/pass/score/CI/resources measured. Parallel source-only intake: G upstream unchanged; newly read complete PostTrain38175B/48655c1b has published AUTHOR-OWNED beb82b1.2503460947x/13 correctness/12 benchmarks, all12 medians/all6 weighted loads slower than archived607. Root verifies rawsource/results hashes and both score reconstructions; missing component profiles remain gap. runNburn is real Muse GGUF runtime with different contract/no inspected OKC H200 receipt; two other leads are articles/evaluator audit. No new GPU job or adoption. Full pins, normalized mean correction, scoped closure/reopen criteria in steering/2026-11-01-rmw-h200-resume.md and linked source reports.

2026-11-01 23:02 UTC | Existing public job3fd1317143304bd5 remains queued, no start/cluster/result. Second600s local retrieval observation ended; same-ID API still live/queued. Persistent observer started23:02:40 UTC, no new submission or remote cancellation. No GPU setup/kernel timeout, correctness result, profiles, score, CI or resources observed. Scoped official public service-guidance inspection found no verified kernel-worker notice/action/ETA; serving-track guidance not transferred. See steering/2026-11-01-rmw-h200-resume.md and linked exact observation receipts. Goal active; prefill excluded.

2026-11-01 23:06:12 UTC | Existing public job3fd1317143304bd5 is RUNNING in sfo3; authoritative worker started_at converts to23:04:18.295910 UTC. No new submission. Exact97935e7d/914572B source unchanged; persistent own-ID wait live. result=null, so no startup/official pass, GPU identity, profiles, latency/score/CI or utilization observed yet. Evidence and fixed gates in steering/2026-11-01-rmw-h200-resume.md.

2026-11-01 23:10:10 UTC | PUBLIC3fd1317143304bd5 TERMINAL: done/passed=false/returncode112, H200 node n-2bbs7xxbcpmk3emdvrn5jsqeic, driver595.91.07, evaluator image/files in owning receipt. Setup FAIL AssertionError:GQUAL_PREPARE_BUDGET;0 correctness tests/0 benchmarks/0 profiles, no latency/score/CI. Original G56 native56 expert replay success printed. Trace reaches total295s guard at an empty-function entry; no cell ordinal/phase times, so no bottleneck attribution or numerical/RMW performance loss. CLI exit1 included valid terminal JSON and original log; both archived under profiles/g-common-rmw.3fd... and steering evidence. Last live quota23:12:57 was1scored/0checks/no live jobs. User subsequently cleared goal: no retry or new submission. Draft822 memo candidate blocked by full CPU inference-mode failure; branch/WIP preserved. Full handoff steering/2026-11-01-rmw-terminal-stop-checkpoint.md.

2026-11-02 00:10:29 UTC (Nov1 local) | PRIVATE c1931f0c9e834485 | Repaired COMMON+short-RMW ON978b29f9/918978B, implementation/registration a0b6af2ec00dfed54f37fd81de00c14098748711. Narrow inference-scope/exact-ID qualifier repair and owning private receipt support pass32 CPU tests; native/timed source unchanged. Fresh preflight layer5scored/10checks/no live, then first of capped4 private checks accepted; initial own receipt modecheck/statusqueued/resultnull. No GPU qualification, timings, score or CI yet; no public submission. Source/OFF pins, full startup/official/actual-exercise/stop gates and exact owning evidence in steering/2026-11-01-qualifier-inference-repair.md. No separate COMMON-first/private1% prerequisite for descriptive single-arm measurement; matched all-six/heldout/duplicate-null/regression promotion gates remain. Root owns retrieval; no blind retry.

2026-11-02 00:49:11 UTC (Nov1 local) | PRIVATE c1931f0c9e834485 TERMINAL PASS | ON978b29f9/918978B, implementation a0b6af2. Actual H200/132SM, complete_native + RMWexercised, full startup12cells/72model/12captures/36fresh/60empty/12emptycaptures/48restores;13 official tests/12 benchmarks/12 profiles, clean flags, return0. Prepare109.269811s (normal106.901449s, remaining2.368223s, cleanup0.000111838s); memo90hit/54miss/peak4. Score2.1212410493206884 CI[2.089422433580878,2.1530596650604985]. Unmatched607 point score-0.227725%, latency-only log gain-0.000546368; no causal RMW effect/promotion. All raw profiles/own receipt/reader/phase/resource evidence archived. Terminal live layer9checks/5scored/no live;1/4 initial tranche used, no public or follow-on dispatch. Report steering/2026-11-01-rmw-v2-h200-terminal.md.

2026-11-02 01:31:40 UTC (Nov1 local) | PRIVATE 16b4db27421d4714 ACCEPTED | Tranche#2/4, matched current repaired-common OFF/ON; uploadc4a5ee29/923915B, implementation2686a7653d6cf2325e944f419cdf0f81e4b6308a. Official entry returns current repaired-common OFF4cbb13a9; both private arms retain common cooperative/progress/preexpert fixes. Frozen36rows/3banks993811,993817,993821/4placements/24cold/256MiB/46:16/allcrosses/1%+2xnull/regression gates.16 CPU/source/owner/reader tests and Tiny/Tiny2 bothbool OFF/ON local selected compiles pass;295s total fit plausible, unmeasured. Own initial statusqueued/resultnull; server content09017717fcca125fa947d2e8e143da7409080668cd55c77d6229bf252d6ebed1. Preflight9private/5scored/no live; immediate postaccept APIstill9/5. ExactIDwait92724 running. No GPU setup/numerical/latency/score/CI/promotion result yet, no public/scored action or retry. Full preregistration and owning receipt: steering/2026-11-01-rmw-common-pair-registration.md.


2026-11-02 02:00:03 UTC (Nov1 local) | PRIVATE16b4db27421d4714 TERMINAL SETUP FAIL | Exact paired uploadc4a5ee29/923915B, implementation2686a765, tranche#2/4. H200; done/result.statusok but passed=false/return112, setup CUDA launch failure719. Log reports normal native startup56 verified replays, then trace _srp_occupied_empty graph.replay/synchronize; no WL_PREP/capsule,0 correctness tests/0benchmarks/0profiles/no score or CI. Frozen745d7bb5 reader correctly rejects official-success guard. Worker237.581422s is not prepare time; setup prepare_s null, phase/cleanup/runtime-fit unknown. Fixture duplicate-ID concern is a source hypothesis, not localized trap causality or RMW loss. Raw receipts/logs/attestation/rejection archived in profiles/g-common-rmw-pair.16b4db27421d4714.* and steering/evidence/2026-11-01-rmw-common-pair. Stop for bounded source/CPU diagnosis; no retry/#3/public authorization. Full report steering/2026-11-01-rmw-common-pair-terminal.md.


2026-11-02 02:22:46 UTC (Nov1 local) | PRIVATEc5e4f89672f743bb ACCEPTED | Explicit replacement tranche#3/4 after fixture-domain diagnosis52628b6; implementation/prereg d5f29167b34f78db864d22550b23f4a0516f6a21. Corrected upload785a90ca08ddde0c3b194fc219a0645cb8cd14bac02620d8714f0b66163a0b66/924005B uses distinct occupied IDs0..7 per token and copy_ restores samecaptured IDs storage. Native7factories/productionprefix/timedfunctions byte/AST invariant, all23 CPU/source/owner/reader/entrynegative gates PASS. Frozen36rows/3originalbanks/4placements/24cold/256MiB/46:16/allcrosses/1%+2xnull/regression/16empty/295s guards unchanged. Owning statusqueued/resultnull, servercontent28d360741906efb0ff5a90be98b92a053aff04480442d92992d47907ee557cab. Preflight8private/5scored/no live latest50layer; postaccept API7private/5scored. Exactwait11629 active. No result/promotion/publicaction; furtherinstrumentfailure stopsRMW, #4freshheldoutonlyconditionalvalid#3pass. Registration/receipts steering/2026-11-01-rmw-legal-fixture-registration.md.


2026-11-02 02:48:30 UTC (Nov1 local) | PRIVATE c5e4f89672f743bb TERMINAL PASS, MATCHED GATE FAIL | Corrected fixture upload785a90ca/924005B, preregd5f29167, acceptancecc80a2c, tranche#3/4. H200;13tests/12benchmarks/12profiles pass, return0, clean flags. Frozen745d7bb5reader exit0/complete=true/private_raw_gate=false. All12bank/cross whole-six logs -0.001829079..0.002828144 (<log1.01 and2xnull); all3banks fail gain gate,993821 additionally T16sliding OFF0->ON0 +1.151997596us versus1.020480022us cap.144initial+144fresh/32empty/16occupied-to-empty/432actual-attention/resource-parity/ownership/cleanup pass. Prepare178.253238s (capsule178.251212s); queue1232.758803s and worker310.826853s distinct. Official repaired-common OFF descriptive2.112499365 CI[2.080811875,2.144186856], not promotedpubliccandidate. Valid subthreshold/regression outcome CLOSES RMW-only promotion; DO NOTspend conditional#4. Fresh02:53:24UTC quota7private/5scored,no live layer latest50. Exact receipt/log/txt/capsule/all18weightedrows/all12crosses/phase/resource evidence archived; report steering/2026-11-01-rmw-legal-fixture-terminal.md. No retry/public/scored action.
| 2026-11-01 Pacific / Nov2 UTC | 9adfd4dbc98f4998 PRIVATE check H200 | f25d714e/934942B; prereg3d00a62; terminal26b3b88 | Fixed outgoing-only four placements | Setup FAIL112; no tests/bench/profile/score; prepare_s null | Graph query binding cudaErrorInvalidDeviceFunction; internal API/node/row/arm unlocalized.56 startup replays and WL_PREP common resource parity retained, no complete capsule. Instrument blocker NOT valid timing loss/PDL closure. STOP/no retry. Canonical steering/2026-11-01-worker-outgoing-pair-terminal.md; captain steering/2026-11-01-outgoing-query-stop-decision.md. |
| 2026-11-01 Pacific / Nov2 06:14:16.954UTC | 6f83ee875dd847c1 PRIVATE check H200 | ad314060/943363B; preregc40930c5; acceptance120a069d; terminaldba5a01 | Repaired driver-query outgoing four placements | Setup FAIL112/SRM_ROUTING_ARM_MISMATCH; no official tests/benchmarks/profiles/score/CI; prepare_s null | 44 query contexts through row10 T8/full/993817; returned queries/edge checks inferred from frozen control flow before routing equality assertion, not complete48 qualification. Mismatch arm/values/cause unlocalized; 11 retained build/load census entries, secondary66regs/0local parity. Invalid qualification, NOT measured PDL timing loss. Seam campaign CLOSED per1a20b31; no rescue/retry. Exact steering/2026-11-01-worker-outgoing-driver-pair-terminal.md; captain steering/2026-11-01-captain-seam-terminal-decision.md. |

| 2026-11-02 11:27:23UTC | dc62cff972934f54 PRIVATE H200/sfo3 | source9a1f6e4, preregc19d8b2; d70a1bcc/870087B, nativef5344c9d | Fully timed WGS SiLU table, ordinary official check | PASS13tests/12bench/12profiles,2.073288676 CI[2.042189346,2.104388007] | prepare112.535864s,56startup replays; queue615.643852s/worker238.836952s separate. No favorable>=1%signal; STOP/PARK exact treatment/no follow-on. Historical607 score ratio0.9751684 and affected d32/v16 regressions are unmatched/descriptive, not causal. Raw worker archive publishing; captain verified local owning receipt and all12aggregation rows. Captain steering/2026-11-02-captain-timed-silu-terminal-decision.md. No public/default change. |


| 2026-11-02 12:36:13.857UTC | cc474883117947c0 PRIVATE H200 | exact unchanged public607 ba9e327c/869076B; allocation1df0000 | Fresh incumbent profile/control | PASS13tests/12bench/12profiles,2.109201357 CI[2.077563337,2.140839378] | prepare107.986481s,56startup; queue595.461545s/worker208.530653s separate. Same source, NOT optimization win; old/fresh latencyfactor1.0197593 and baselinefactor0.9728373 explain scorefactor0.9920599. Fresh raw receipt/hotspots/strict comparison: steering/2026-11-02-captain-fresh-incumbent-profile.md. No public/default change. |
| 2026-11-02 12:37:24.360UTC | e4c8f00532b74d15 PRIVATE check ACCEPTED | candidate5d883c43/869931LF bytes; source8c0bfba, allocation1df0000 | State-local serial BF16pooler/FP32router atT1/T8 | Owning receipt queued/resultnull; NOT numerical/performance result | Server85a3334b/payload association retained; after valid exact-control terminal and freshpreflight2private/5scored/no live. Worker owns sameIDwait, one finalprivatecheck reserved, no public/scored/default. Captain checkpoint steering/2026-11-02-captain-fresh-incumbent-profile.md. |

| 2026-11-02 12:51:26UTC | e4c8f00532b74d15 PRIVATE H200/sfo3 TERMINAL | source8c0bfba, candidate5d883c43/869931LF bytes, archive87cfa4a | Exact state-local T1/T8 serial policy | PASS13tests/12bench/12profiles,2.093072852 CI[2.061676760,2.124468945] | prepare117.972810s,56original startup. Independent strict freshcontrol comparison ownlatencyfactor0.9698273; all6loads>1%slower descriptively, shortfactor0.9625515/long0.9734858; not causal. All4serial signatures observed,8long multisets equal. PARK/no retune/confirmation/public/default. Captain steering/2026-11-02-captain-serial-terminal-decision.md. Final1privatecheck reserved. |
| 2026-11-02 15:03:47.250UTC | 60b38a62a135442c PRIVATE H200/sfo3 TERMINAL | physical0f90a2a8/882844LF; server805748b6; prereg01b8e23/archive335887c | PR graph-owned generic-compressed backing | INVALID SETUP passedfalse/return112;0tests/bench/profiles; no score | Actual host flags C++20 then C++17 conflict with target Torch C++20 requirement. No receipt/allocation/runtime/cleanup/performance qualification; prepare_s null. Queue372.234307s/worker166.265381s separate. Exact allocation CLOSED/no retry/scored/default;0private left. Captain steering/2026-11-02-captain-pr-terminal-next-implementation.md. |


## 2026-11-02 - captain K128 terminal

PRIVATE own `bf8671b4bfa845a2`, implementation15688256020e80c2c2f72f97e76cbd13b034ac0a / physical440f41b118eeb3676867e15d109c2d5f7da0213873f84a17c9d68d1af868703f /885489LF. Fixed2WG/128row/K128/64tokenreuse B4T32+MAINT128/256 in both contexts; other routes unchanged. H200 PASS13correctness/12benchmarks,12completeprofiles; prepare108.018165902s, actualresident2 and56startupreplays. Official1.965158178216973x,95%CI[1.9356808055437182,1.9946355508902274]. Score ratio versus public6070.924309388; descriptive six-load freshcc474 endpoint factor0.910733407, not matched causal effects (different input/cluster/node/driver, evaluator file hashes match). PARK exactfixedimplementation, cancelconfirmation; no public/default change or tuning. Original profiles in steering/evidence/2026-11-02-worker-wgs-k128-full-reuse/generated/profiles/; fullraw, manifest, board/quota and calculation archive in steering/evidence/2026-11-02-driver-k128-first-check/. [Terminal decision](steering/2026-11-02-driver-k128-terminal.md). Fresh21:32UTC overall/layer371, incumbent607unchanged,9private/10scored layer left,no live jobs.


## 2026-11-02 - shared-A pipeline terminal

PRIVATE own5adf172986ff43dc; implementationdcb24972, physicalfbca3b75125dd45b3e2801358e95aa14a16750860482989a5fef444a576cd578/894471LF, nativec2958fc9. Fixed256row/fourcomputeWG/672thread K128 producer-decode SS-WGMMA, separate raw/A/X two-slot rings,64tokenreuse B4/MAIN. H200 PASS13tests/12benchmarks/12completeprofiles+56startup, actualresident1/grid132/dynamic221952/static0/register80/local40B4/24MAIN; prepare99.364714113s. Official1.6127943994197347x,95%CI[1.5886024834284387,1.6369863154110305]. Independent reconstruction exact; descriptive freshcc474 six-load endpointfactor0.752409236, public607scorefactor0.758575580. Different seeds/cluster/node/driver despite matching evaluatorfiles: not matchedcausal or spill/shared-bandwidth attribution. PARK exactsource/no confirmation/tuning/public/default. Originalgenerated/profiles and fullraw/manifest/calculation/hotspots archived under steering/evidence/2026-11-02-driver-shared-a-first-check and worker-shared-a-pipeline/generated/profiles. [Terminal and next fixed implementation](steering/2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md). Fresh23:05UTC overall/layer375,8private/10scored,no live jobs,incumbent607unchanged.


## 2026-11-02 - direct-weight register pipeline terminal

PRIVATE own0b684a4065354fc5, implementation368b85c9, physicala27a1baff54f5496e339044760d1a58927af8733fa6af6fadcc68c60dcf140bd/875672LF/nativef95fbb11. OriginalK256/256rows/fourWG/64tokenreuse, directglobal code/scales into RS registers, rawshared/TMAproducerremoved,544threads/threeXslots. H200 PASS13tests/12benchmarks/12completeprofiles/56startup, DWactualresident1/grid132/dynamic120576/static0/register96/local24B4/16MAIN; prepare109.370262382s. Score1.9264235913174548x,95%CI[1.897527237447693,1.9553199451872163]. Exactscore reconstruction, descriptivefreshcc474 endpointfactor0.889351467; all12seeds/cluster/node/driverdiffer despiteevaluatorhashmatch. Not matchedcausal/globalbandwidth/spill attribution. PARK exactsource/no tuning/confirmation/public/default; heldb325pair stopped. Raw/profiles/manifest/hotspots under steering/evidence/2026-11-02-driver-direct-weight-first-check and worker-direct-weight-registers/generated/profiles. [Terminal, calibration and next allocation](steering/2026-11-02-driver-direct-weight-terminal-and-dense-backing.md). Fresh23:33UTC overall/layer376,7private/10scored/no-live,public607unchanged.


## 2026-11-02 - unchanged current-service calibration terminal

PRIVATE94ef698e91964dfa TERMINAL VALID; source/control5b4a729a,04314b16/869198LF (exactba9 prefix +comment/unusedtag). PASS13tests/12benchmarks/12completeprofiles, native56replays; prepare114.752085144s. Score2.0917476853194494x,95%CI[2.060371470039658,2.123123900599241], reconstructed exactly. D32adjusted sliding157.620500us/full317.105412us,C32=12.324229608ms, ownbaseline22.328927428ms/1.811790930x. H200/nyc3b/n-inespekiexhkaykscufvmyulra/driver580.159.03. Evaluator matches retainedcc474/direct0b, all12seeds/node differ; descriptive only, no optimization gain or causal driver attribution. Fullraw/log/txt/profiles/manifest/reconstruction in steering/evidence/2026-11-02-driver-current-incumbent-control. [Terminal and next fixedMID gate](steering/2026-11-02-driver-d32-control-terminal-and-mid-gate.md). No public/default change.


## 2026-11-02 - fixed D32 MID ordinary exploration accepted

PRIVATE a6fe2769779e4299 ACCEPTED; source28346bf563789879bcec28e52edeb05753a6b3f0,e607fb89/870649LF, existinghelperb8 selectedfor17..32,otherfive/M33unchanged. No shader/math/packing/scratch changes.6rootCPUtests pass8.897s, independent conditionalSOURCEACCEPT. Remote49blobs verified and exactphysicalsource gate, fresh6private/10scored/no-live, calibration94ef reconciled; onePOST,initialownstatusqueued/teamyelgabo/boardmuse_decode_layer/check/duplicate_ofnull. Servercontent3170a56ac500b038ad1c10bbdceddd79f8e13d348d411b88416d2f77eb70481d differsphysicalSHA. Target13/12/12+MID_SELECTION+native56replay gates pending, no timings/score/CI orpublic/default action. ImmediateAPIstillreported6private,refreshaftercharge/reset. Overall/layer379,public607unchanged,layer100cutoff2.2698. [Registration/stop rules](steering/2026-11-02-driver-d32-mid-first-check.md).


## 2026-11-02 local / Nov3 UTC - fixed D32 MID terminal

PRIVATE a6fe2769779e4299 TERMINAL VALID; source28346bf5/e607fb89/870649LF, server3170a56a. ActualD32_MID_SELECTION helperb8/exactdefines/keymid,resource_queryNone/native56replays,13tests/12benchmarks/12completeprofiles/cleanflags/return0; prepare104.089840085s. Wholelayer2.0894234868005395x,95%CI[2.058082134498531,2.1207648391025473], independently reconstructed. D32adjusted sliding166.128808us/full319.397185us,C32=12.752280143ms,ownbaseline21.851072311ms/1.713503159x; rawC32=12.959999934ms. Versus unchanged94efC32=12.324229608ms,descriptive3.47324%slower,allseeds/node differ,sameH200nyc3b/580.159.03/evaluator. PARK exactMID17..32,no usefulsignal/no automaticconfirmation/tile-helper-thresholdrescue/public/default. FullrawCP1252 preserved,labeledcanonicalUTF8 derivative/integerseeds/analysis/manifest and originalgeneratedprofiles archived; bothreceiptreaders passWSLPython. Fresh20private/10scored/no-live,overall/layer379,public607unchanged. [Fullterminal](steering/2026-11-02-driver-d32-mid-terminal.md). Root accepted-job exception retired; Decode32Worker owns laterD32directly under humanD32-02/9dea190b, no recurringcaptainhandoffgate.


2026-11-02: private check `17383751c6a24010`, source `31c7f920` / `5a580c7a`, D32 score lookahead. VALID, 13 tests / 12 benchmarks / 12 profiles, whole-layer 2.128829x, 95% CI [2.096897, 2.160762]. D32 adjusted sliding/full 149.764/312.070 us, weighted 11.882275 ms. Different-input control comparison is descriptive; matched confirmation pending. [Receipt and result](../archive/docs/steering/decode32/2026-11-02-05-attention-first-result.md).


2026-11-02: private matched check `4fece18b20a2447b`, diagnostic source `109cdb07`, setup failed with `NotImplementedError` from `torch.isfinite` on FP8. No correctness test, timing or score was produced. [Owning receipt and correction](../archive/docs/steering/decode32/2026-11-02-09-pair-setup-failure.md).


2026-11-02: private matched check `62a4f6b27ed74b50`, diagnostic source `f397bca8`, failed setup with `EXPERT_FRESH_ROUTING_MISMATCH`; no official tests, benchmark cells, profiles or score. [Receipt and diagnostic plan](../archive/docs/steering/decode32/2026-11-02-11-matched-routing-mismatch.md).

2026-11-02: private routing diagnostic `5909739e8a4445b2` accepted, source `77023d44` / `9e5e5d64`. Fresh no-live/17-check preflight and published source identity verified. One own-ID waiter is active. No terminal result or timing yet. [Registration and strict gates](../archive/docs/steering/decode32/2026-11-02-11-matched-routing-mismatch.md#diagnostic-accepted-november-2-at-1749-pst).

2026-11-02: private routing diagnostic `5909739e8a4445b2`, source `77023d44` / `9e5e5d64`, terminal setup failure112. D32 sliding/seed1102031/fresh-7/row6 swaps experts93 and153 between slots3/4 in placement3, same set, captured attention selection false. Native startup56 replay marker present, no official tests/benchmarks/timing profiles/score/CI or paired report. Strict route gate retained. [Complete receipt and next distinguishing observation](../archive/docs/steering/decode32/2026-11-02-14-routing-order-diagnostic.md).

2026-11-02: D32-ATTN-05 source `0ddbc284` / `a0998224` rejected at POST with HTTP422 `reads harness frames`. No accepted job ID, target result, profile or timing. Shared jobs reconciled before source repair. Corrected explicit-call-site observer D32-ATTN-06 is source `00193aa2`; six CPU tests pass. [Evidence and one-run contract](../archive/docs/steering/decode32/2026-11-02-16-explicit-route-observer.md).

2026-11-02: D32-ATTN-06 private diagnostic `8b1e364dafb54c71` accepted, source `a7e08ee4` / `00193aa2`. Fresh no-live/16-check preflight and exact published bytes verified; one foreground own-ID waiter. Six CPU tests pass. No target correctness/timing result yet; this observer intentionally stops before timing. [Registration and decision contract](../archive/docs/steering/decode32/2026-11-02-16-explicit-route-observer.md#accepted-private-diagnostic).

2026-11-02: D32-ATTN-06 private diagnostic `8b1e364dafb54c71`, source `a7e08ee4` / `00193aa2`, terminal setup failure112 on `EXPERT_PREPARE_295S`. Native56 replay marker present; no observer capsule, official tests, benchmark cells, timing profiles, score/CI or matched C32. Instrument closed as inconclusive under its one-run stop, no retry or kernel-loss claim. [Receipt, source check and research gap](steering/decode32/2026-11-02-17-observer-budget-closure-and-refill.md).

2026-11-02: D32-REDUCE-01 private ordinary check `bd5776bdfcab42a7` accepted, source `0585bd21` / `93a05524`, one private accumulator copy per D32 sliding split. Fresh no-live/15-check preflight; exact decompressed remote source verified. Three CPU tests pass. Official correctness and D32/full-six times pending. One waiter writes ignored `.build/`. [Registration and stop](steering/decode32/2026-11-02-18-active-reduction-experiment.md#accepted-check-and-next-local-work).

2026-11-02: D32-REDUCE-01 own private `bd5776bdfcab42a7`, source `0585bd21` / `93a05524`, terminal VALID13tests/12benchmarks/12profiles/56startup replays, selected eight private copies. D32 adjusted sliding/full161.438408/304.957678us, C32=12.305490ms, own baseline22.334400ms/1.814995x. Whole-six2.119214x,95%CI[2.087426,2.151002]. No useful affected sliding signal against independent94ef; STOP exact treatment/no copy sweep, matched gain or routing-correction claim. Incumbent unchanged. [Full receipt, all-six results and distinct next candidate](steering/decode32/2026-11-02-19-reduction-result-identity-next.md).

2026-11-02: D32-IDENTITY-01 own private comparison `4a2a79bdac04499a` accepted, physical paired source `f7a5d1fb`/920106 bytes, published `59011284`, native `b599e5b2`. Full-context exact identity-rescale skip, two independent control/treatment pairs,36-row all-load gate and bounded partial evidence. Eleven local CPU/host/reader tests and enabled/disabled CUDA device compilation pass; no target result yet. Fresh14 private layer checks, separate General MoE check queued and untouched. [Registered source, budgets and acceptance](steering/decode32/2026-11-02-19-reduction-result-identity-next.md).


November2 D32-IDENTITY-01 terminal: own private4a2a79bdac04499a, pairedsourcef7a5d1fb/sourcecommit59011284. Setup fails exact ordered routing at T128/full/seed1102031 initial,row53,placement2,treatment selectedfalse, experts175/68 swapped slots5/6. Source-pinned partial evidence covers27/36rows at150.244213s. No complete matched gate, official timing or integration. Stop exact candidate; waiter43239 ended. Complete raw/log/progress capsules and terminal-analysis under docs/steering/decode32/evidence/D32-IDENTITY-01/. Worker now owns standalone MoE P0, no new D32 iteration.

## 2026-11-03 - UNP slice 1 (fused unpooler slice + finish tail), check 1: control-equivalent

PRIVATE `4c2d5e3a8e1147bf`, source `muse_decode_layer/cand/g1b4_unp_release.py` at worktree commit b7862c5d (SHA256 d0c537ec...), built by `tools/dl_unp_patch.py` from the public 607 file (ba9e327c). Change: at T = 1 and 8 the tiny / tiny2 MoE builds export `okc_run_unp`, which streams the 8.4 MB unpooler slice as 4-row units through the same bulk-copy ring behind the last phase-B unit, computes the slice GEMV after a per-launch counter on the final-pass items (no grid barrier) and runs `_sl_cluster_finish`'s boundary after a counter on the slice units; the separate launches stay as the fallback and a prepare-time gate (fresh-input replays against the separate path, then both tails timed as graphs) picks per T. Result: PASS 13/13, 2.117x, 95% CI [2.085, 2.149], 10.69 ms; prepare within budget (G_RELEASE_STARTUP_OK 56 replays). Gate lines: `UNP_GATE T=1 ok` fused rel m 1.7e-05, P 1.3e-04, s2 2.7e-05, out 1.9e-05, elem 2.0e-03 (control replays 0); `T=8 ok` m 5.8e-06, P 1.9e-04, s2 2.5e-05, out 9.8e-05, elem 4.9e-03 (control m 5.8e-06, P 5.3e-05, s2 2.4e-05, out 9.7e-05, elem 4.5e-03). **Timing in prepare (8 local routes per token, cold L2): tails sep 75.1 / fused 81.2 us at T = 1 and 126.7 / 132.6 at T = 8, so the gate chose the separate path at both T and the scored loads ran the unchanged kernels.** Per case us (this run / 607eb52 / cc474883): d1 77.6 / 76.75 / 77.04 sliding, 85.4 / 83.73 / 83.73 full; v1 113.7 / 115.43 / 110.76 sliding, 122.1 / 117.60 / 118.20 full; the other eight cases within the two references' spread. Mechanism of the loss as built (named from the code, not yet from a timeline): the slice GEMV used CUDA-core FMAs over all eight token slots regardless of M (about 1500 instructions per lane per unit, roughly 3.6 us per CTA at every M), and the pick timed a routing far from the judge's. Receipts: `muse_decode_layer/profiles/g1b4_unp_release.4c2d5e3a8e1147bf.{json,txt,log}`. Next: the tail on `mma.sync` (A = out rows, B = the unit's rows, one shared K permutation), a per-CTA globaltimer timeline printed by the gate, and the pick decided on a judge-like routing (check 2).

## 2026-11-03 - UNP slice 1, check 2: mma tail, timeline; gate kept the separate path again

PRIVATE `0c35e5f5cc614806`, source `g1b4_unp_release.py` at worktree commit 9a4619dd (SHA256 79cb50f6...). Change against check 1: the slice unit compute on `mma.sync m16n8k16` (A = the MoE output rows, B = the unit's four rows at a 32 B row pad, one shared K permutation, one accumulator), a per-CTA `globaltimer` timeline printed by the gate, and the pick decided on a judge-like routing (`_release_routes` "partial": about one local route per token). Result: PASS 13/13, 2.096x, 95% CI not better than the references; per case us d1 77.48 / 84.81, v1 115.09 / 121.92 (607eb52: 76.75 / 83.73, 115.43 / 117.60; cc474883: 77.04 / 83.73, 110.76 / 118.20); d16 132.77 / 210.94 and the four long loads inside the two references' spread (this machine read verify 16 and 32 about 2-3% above cc474883 on unchanged kernels). Gate: numerically ok at both T (fused rel P 4.3e-06 / 2.0e-04, out 0 / 4.9e-05, elem 1.3e-04 / 3.7e-03). **Timeline, judge-like routing, median/max us from kernel entry: T=1 entry 0.1, barrier 13.8, tail 14.0, final items published 15.4, x loaded 16.8, slice units done 18.7, boundary start 19.3, done 20.1; T=8 barrier 35.6, tail 35.8, fin 37.3, x 40.2, units 42.0, ucnt 42.5, done 43.4.** So the fused tail is 6.3 us (T=1) / 7.8 us (T=8) after the grid barrier: final pass + counter 1.4-1.5, the x round trip 1.4 / 2.9, two slice units 1.9 / 1.8 (32 dependent mma on one accumulator per unit), the boundary 0.8-0.9. In-process tails as graphs, cold L2: judge-like sep 27.8 / fused 29.4 at T=1, 47.3 / 48.8 at T=8 (hot routing 73.1 / 74.3 and 124.1 / 125.5), so the gate chose the separate path at both T and the scored loads again ran the unchanged kernels (control-equivalent). Receipts: `muse_decode_layer/profiles/g1b4_unp_release.0c35e5f5cc614806.{json,txt,log}`. Next (check 3, the last): four independent accumulator chains, x as 16-byte chunks per lane (chunk-interleaved K mapping, two smem wavefronts per B step), and the fused path forced on at T <= 8 (UNP_LOSS_US 3.0) so the judge's profile rows measure it.

## 2026-11-03 - UNP slice 1, check 3 (last): fused path forced on, measured by the judge, loses at decode 1

PRIVATE `c77b95102baf4232`, source `g1b4_unp_release.py` at worktree commit 79f5452e (SHA256 3d7f252f...). Change against check 2: four independent mma accumulation chains per unit, x as 16-byte chunks per lane (chunk-interleaved K mapping, two shared-memory wavefronts per B step), and the pick threshold raised to 3 us so the fused tail runs on the scored loads whenever the numeric gate passes. Result: PASS 13/13, 2.081x, 95% CI [2.050, 2.112], 10.87 ms; gate numerically ok at both T (fused rel P 0 / 2.0e-04, out 0 / 8.3e-05, elem 0 / 5.4e-03, the same as the control replays), tails in-process on the judge-like routing sep 28.2 / fused 29.4 us at T = 1 and 47.9 / 48.0 at T = 8, both picked fused. **Scored loads (this run / 607eb52 / cc474883): decode 1 sliding 80.67 / 76.75 / 77.04 us, full 85.35 / 83.73 / 83.73; verify 1 sliding 112.19 / 115.43 / 110.76, full 122.46 / 117.60 / 118.20.** The profiles show seven launches at T <= 8 (no `sk2_kernel<1, 8, 1, 1, 8, 2, ...>` slice row, no `_sl_cluster_finish`): decode 1 sliding `moe_fused` 39.1 us against 29.8 + 4.4 + 2.3 = 36.5 for the three kernels it replaces (+2.6, with the step +3.6 to +3.9); decode 1 full 31.4 against a routing-dependent 42.8 + 4.5 + 1.9 (single-sample rows); verify 1 sliding 65.4 against 71.6 + 4.7 + 1.7 (same caveat, the step -3.2 / +1.4). Timeline at T = 1 (median us from entry): barrier 14.0, tail 14.1, fin 15.5, x 16.4, units 18.2, ucnt 18.8, done 20.0; at T = 8: barrier 35.5, fin 37.1, x 38.7, units 40.6, ucnt 41.2, done 42.1. The accumulator and load changes moved nothing (units still 1.8 us, which is the x round trip's completion plus two units). The other four loads moved by -2.1% to +3.5% on unchanged kernels (machine noise of this run). Decision: slice 1 fails its acceptance (decode 1 slower on both references; verify 1 not resolved); recorded in docs/ruled-out.md with the mechanism; no promotion-queue entry (no receipt below the public 10,639.4 us). Receipts: `muse_decode_layer/profiles/g1b4_unp_release.c77b95102baf4232.{json,txt,log}`. Slice 2 (o + boundary + pooler/router fusion) not attempted: its precondition (slice 1 passing) failed and the three-check budget is spent.
2026-11-03: F3 (reframe section 5 item 2) private check `43aa191eb94948a4` accepted, source `muse_decode_layer/cand/g1b4_f3.py` (sha256 5951c822, commit e7ac9cbf on worktree-agent-a7ab0a887829a6341): the public release file plus a selectable fused launch at T = 1 / 8 (o projection as two row blocks per CTA, the attention boundary on 8 CTAs per token with the glue kernel's exact operation order, the pooler stream with the interleaved FP32 router; completion counters, pooler weights prefetched before the boundary wait, writer- and reader-side fence.proxy.async.global). prepare() self-tests the fused launch (boundary and pooler/router bitwise on the fused P/h/hr, eager and three fresh-input graph replays) and enables it per T only if the paired chain timing saves 0.5 us in 7 of 9 rounds. Local sm_90a compile (tools/dl_f3_sass.sh): 124 registers, 0 spills. Pre-registered stop: the fused row must sit 2 us or more under the three rows it replaces (27.3 us at decode 1 sliding in 607eb52bdaf24160) with decode 1 and verify 1 each 2 us better; otherwise record the loss and stop. Result pending.

2026-11-03: F3 check `43aa191eb94948a4` terminal: timed out at the run's limit before the first test (no tests, cells or profiles). Log (muse_decode_layer/profiles/g1b4_f3.43aa191eb94948a4.log): the T = 1 self-test passed (boundary and pooler/router bitwise on the fused P/h/hr, o max rel diff 5.98e-4 on 3 elements) and the in-process paired chain timing read off 62.56 / on 61.06 us, median saving 1.28 us, 9 of 9 rounds; the T = 8 self-test then hung. Cause (read from the source, not measured): the counter targets were (launch index + 1) x (8 M) and (launch index + 1) x 8 per token, but the per-launch increments depended on M, so after the T = 1 launches the T = 8 boundary wait could never be met and the producer's poll inside its ring wait was unbounded. Fix (commit after this entry, sha256 9231e381): every launch adds 128 to the o and boundary counters and 8 to each of the 8 per-token counters (CTAs without the work arrive at once), the producer's poll is bounded and fails closed through diag, and prepare() self-tests T = 1, 8, 1, 8 in that order. Check 2 of 3 is the fixed file.

2026-11-03: F3 check 2 of 3, private `c9240b8205904baf` accepted, source `muse_decode_layer/cand/g1b4_f3.py` sha256 9231e381 (commit b4aa2557): the counter fix above. Same pre-registered stop. Result pending.

2026-11-03: F3 check `c9240b8205904baf` terminal, source `muse_decode_layer/cand/g1b4_f3.py` sha256 9231e381 (commit b4aa2557): VALID, 13/13 tests, score 10,825.7 us (2.0895x, CI 2.058-2.121) against 10,639.4 (607eb52bdaf24160) and 10,724.6 (cc474883117947c0). prepare 109.7 s (the release: 108.0). In-process tuner (cold-L2 paired chain replays, 9 rounds): T = 1 off 62.34 / on 61.70 us, saving 0.45, 8 of 9 -> OFF (threshold 0.5 us); T = 8 off 104.51 / on 103.36 us, saving 1.09, 9 of 9 -> ON. So decode 1 ran the separate kernels and verify 1 the fused launch. Self-tests: boundary and pooler/router bitwise on the fused P/h/hr at T = 1, 8, 1, 8 (eager and three fresh-input graph replays each), o max rel diff 9.9e-7 (T = 1) and 1.2e-3 on 14 elements (T = 8; another K-split order of the same stream). Cells (us, this run / 607eb52 / cc474883): decode 1 sliding 77.67 / 76.75 / 77.04, decode 1 full 84.44 / 83.73 / 83.73, verify 1 sliding 118.90 / 115.43 / 110.76, verify 1 full 125.15 / 117.60 / 118.20, decode 16 132.24 / 138.66 / 133.04 and 208.07 / 214.53 / 206.58, decode 32 155.89 / 154.82 / 150.87 and 308.54 / 297.88 / 309.84, verify 16 247.09 / 247.13 / 237.04 and 331.85 / 333.22 / 332.65, verify 32 323.49 / 316.39 / 308.62 and 481.18 / 482.66 / 473.15. Profile rows: verify 1 sliding f3_kernel 27.2 us against o 6.0 + boundary 3.0 + pooler/router 18.7 = 27.7 (607) and 6.0 + 3.0 + 20.2 = 29.2 (cc); verify 1 full 28.0 against 27.9 (607) and 28.3 (cc); decode 1 (separate path in this run) o 5.8 + boundary 3.2 + pooler/router 18.0 = 27.0. The verify 1 endpoints moved with the MoE row (50.4 / 71.6 / 50.6 sliding, 72.4 / 58.8 / 85.7 full: input routing), not with the fused row. Decision: the pre-registered stop is met. The fused launch sits 0.5 us (607) to 2.0 us (cc) under the three rows it replaces at T = 8 and the in-process paired saving at T = 1 was 1.28 us (43aa191eb94948a4) and 0.45 us (this run), under the 2 us bar; neither short load improved. Recorded as a loss in docs/ruled-out.md; no third check. Receipts: muse_decode_layer/profiles/g1b4_f3.c9240b8205904baf.{json,txt,log,status.json}.

## 2026-11-03 - late PDL trigger placement: not run, the release has no PDL edge

No job. The public-sweep item (docs/steering/2026-11-03-0115-public-techniques-research.md, first row of "Techniques not in our logs") asks to move `griddepcontrol.launch_dependents` from kernel entry to right after each producer's last weight load. A source read of `muse_decode_layer/cand/g1b4_stage_release.py` (SHA256 ba9e327c, the file behind 607eb52bdaf24160) shows that no kernel in the layer launches as a PDL secondary:

- all five `cudaLaunchAttributeProgrammaticStreamSerialization` sites (attention launcher line 3593, sk2 launcher 8598, the two `glue_launch` copies 11784 and 15604, `b2s` 11991) set `programmaticStreamSerializationAllowed = 0`. The fork builder rewrites every `= 1` to `= 0` (`tools/dl_build_attributed_fork.py` line 138, "Establish a sequential reference for G1");
- `PDL = False` (line 93), so `_pdl(T)` is False, every Triton launch passes `launch_pdl=False`, `bfp_SK2_PDL` is 0 (the sk2 kernels skip their `griddepcontrol` instructions entirely) and the attention plan's PDL bit is 0;
- the MoE is a `cudaLaunchCooperativeKernel` launch with no attributes.

So in the nine kernels of the T <= 8 chain (qkv sk2, attention, o sk2, `glue_cluster_boundary`, the fused pooler / FP32 router `sk2_kernel_gle`, `_sl_tile_route`, `moe_fused`, unpooler sk2, `_sl_cluster_finish`; the 607eb52 profile rows), `griddepcontrol.wait` returns at once and the trigger has nothing to release. The sk2 kernels also already place their trigger after the last weight stage is issued (line 8181). A trigger-only treatment would run the control's schedule, so no check was spent (muse_decode_layer had 8 of 20 checks left). Recorded in docs/ruled-out.md with the reopening condition: the gain in the public receipts needs PDL itself on for named edges, which the repair inventory ("Global layer PDL off") permits only per edge with a full proof.

## 2026-11-03 - PDL consumer load audit (source only, no quota)

Scope: every kernel of the T <= 8 chain of `g1b4_stage_release.py` (ba9e327c) that contains a `griddepcontrol.wait`, and every buffer it reads that an earlier kernel of the same replay wrote. The rule being checked: under a live PDL edge such a read must bypass L1 (`__ldcg`, `ld.global.cg`, `cp.async.cg`, `cp.async.bulk` or TMA), because a dependent grid resident before its primary finishes can hit L1 lines that hold the previous replay's values (Kushagrabainsla L11). **Status in the release: latent, not live.** No edge in the release is programmatic (section above), so every kernel starts after its predecessor completes and the ordinary kernel-boundary visibility applies. The plain loads below are defects only for a build that turns PDL on for that edge.

| consumer kernel | buffer (writer in the same replay) | load used in the release | site (line) | verdict |
|---|---|---|---|---|
| `attn_kernel` (qkv as PDL secondary) | q, new k / v rows of `qkv` (qkv sk2) | `cp.async.cg` (16 B) | 1525, 1532 | bypasses L1 |
| | output-gate rows of `qkv` (qkv sk2) | plain `ld.global` (`*reinterpret_cast<const uint4*>`) | 1086 (`fin_nc1`), 2839 and 2900 (`segment`), 3015 (`flat_finalize`) | **plain: latent defect** |
| | gate rows staged by `cluster_stage_gate` | `cp.async.cg` | 978 | bypasses L1 |
| | `ws_acc`, `ws_l` (other CTAs of the same grid) | `__ldcg` | 1083-1085 | bypasses L1 (in-grid) |
| o sk2 (`sk2_kernel<1, 8, 4, 1, 4, ...>`) | x = attention output | TMA (`tma_load_3d`), `fence.proxy.async.global` after the wait when `pdl` | 8155-8158, 8173 | bypasses L1 |
| `glue_cluster_boundary<4, 32, false, false, true, *>` (b1) | `branch` = o projection output P | plain `ld.global.v2.b32` (`glb`) | 15546 (live copy), 11739 (older graft copy) | **plain: latent defect** |
| | `part` (SLICED builds only, not instantiated at T <= 8) | plain `ld.global.v4.f32` (`glf`) | 15538 | not live at T <= 8 |
| fused pooler + FP32 router `sk2_kernel_gle` (FUSE) | x (BF16 h) and x32 (FP32 router copy), both b1 outputs | TMA for x, `cp.async.bulk` (`bulk_g2s`) for x32 | 8173, 16349 (`_gle_source` fpr stage) | bypasses L1 |
| | `fused_part` (other CTAs of the same grid) | `__ldcg` | 8419 | bypasses L1 (in-grid) |
| `_sl_tile_route` (Triton) | `pooled`, `logits` (fused pooler / router) | `tl.load` with no cache modifier | 15805, 15807, 15814, 15816, 15829, 15831 | **plain: latent defect** |
| `moe_fused` (helper0 source, Tiny / Tiny2 / MID builds) | `topk_ids` = local ids (route) | `__ldg` (`ld.global.nc`) | 12785, 12804, 14712, 14736, 14746 | **`__ldg`: latent defect** |
| | `topk_w` (route) | `__ldg` at 14903 (FINPF), plain at 14972, 15026, 15046 | as listed | **latent defect** |
| | x rows = `xm` (route) | `__ldg` in `load_x_regs` / `load_x1_smem` (phase A), `cp.async.cg` in `fat_stream`, `__ldg` in `wgs_gather` (WGS builds) | 13002-13003, 13024; 13687-13718; 14334 | **`__ldg`: latent defect** (the `cp.async.cg` sites bypass L1) |
| | `inter`, `xA` / `xB`, `ss` (same grid) | `__ldcg`, bulk copies | 13002 (PB arm), 14283, 15026 | bypasses L1 (in-grid) |
| unpooler sk2 (`sk2_kernel<1, 8, 1, 1, 8, 2, 4, ...>`) | x = MoE output m | TMA | 8173 | bypasses L1 |
| `_sl_cluster_finish` (Triton) | P = unpooler output; `CARRY` (s1) and `TAIL` (b1) | `tl.load` with no cache modifier | 15645, 15646, 15648, 15650 | **plain: latent defect** |

The qkv sk2 kernel is the first launch of the step and reads only the layer input. Counts: eight consumer kernels; five of them hold plain or `__ldg` reads of a same-replay predecessor's output (attention gate rows, b1 branch, route pooled / logits, MoE ids / weights / x, finish P / s1 / tail), on 29 source lines; none is live in the release. The three projection kernels and the fused pooler / router read their predecessor's output only through TMA or bulk copies.

Fix, kept out of the release: `tools/dl_pdl_ldcg_patch.py` writes `muse_decode_layer/cand/g1b4_pdl_ldcg.py` from the pinned release with every site above switched to `__ldcg`, `ld.global.cg` (a new `glbcg` helper for the b1 branch) or `tl.load(..., cache_modifier='.cg')`; exact replacement counts are asserted, and the result passes `py_compile`. It was not compiled for sm_90 (this machine has no nvcc) and has no job. It changes SASS in five of the nine kernels with no correctness benefit while PDL stays off, so it belongs only in a build that turns PDL on for these edges, and it is not a release candidate by itself.

## 2026-11-03 - PDL8 check 1: PDL on for the non-cooperative T <= 8 edges

PRIVATE `8f6acddaca2e4c50` accepted, source `muse_decode_layer/cand/g1b4_pdl8.py` (SHA256 6c94513b0c614e89ce9a4c67594e84f4fe626901c5b3e092b97a293160747692, commit fbc8c46f), generated by `tools/dl_pdl8_patch.py` from the public file ba9e327c: the release bytes plus one appended block behind `PDL_CHAIN` (False: no source edit and no rebinding, so the release itself). Lead decision 2026-11-03, budget two layer checks. Edges made programmatic at T = 1 and 8: qkv sk2 -> attention, attention -> o sk2, o sk2 -> b1, b1 -> fused pooler / router sk2, pooler / router -> route, unpooler sk2 -> finish. Route -> MoE and MoE -> unpooler stay serialized (cooperative MoE). Consumers read predecessor outputs only after `griddepcontrol.wait`, through `__ldcg` (attention gate rows), `cp.async.cg` (q / k / v), TMA behind `fence.proxy.async.global` (sk2), `ld.global.cg` (b1 branch, new `glbcg`) or `cache_modifier='.cg'` (route, finish); b1 no longer waits at entry (its pre-wait reads are the layer input and weights). Triggers: sk2 after its last weight stage (unchanged), attention at each segment's merge (`OKC_PDLTRIG`), b1 after its branch reads; qkv and unpooler run the sk2 PDL prologue and trigger but launch without the attribute (pdl word bit 29). prepare() times PDL on against PDL off of the same build at decode 1 and verify 1, sliding and full (ctx 80000, page 1, cold-L2 graph replays, 9 paired rounds), then runs 2000 replays per arm of decode 1 sliding and verify 1 full on fixed inputs against the PDL-off output (bound 1e-4, also printed against the off arm's own drift) and turns PDL on per T only if the stress passes. Not compiled locally (no nvcc).

**Result: VALID 13/13, 10,554.9 us, 2.1431x (95% CI 2.111 to 2.175)** against the public 607eb52bdaf24160 at 10,639.4 us (2.1261x) and its four day-1 repeat draws 10,642 to 10,783 us. Prepare log: `PDL8_AB` (median paired saving, 9 of 9 rounds each) T=1 sliding 90.08 -> 82.88 us (7.17), T=1 full 89.54 -> 83.33 (6.94), T=8 sliding 133.54 -> 128.22 (5.06), T=8 full 133.66 -> 127.81 (5.95); weighted d1 7.11 us, v1 5.29 us, six-load estimate +2.27% if both T were on. `PDL8_STRESS` (2000 replays per arm, max |diff| / max |ref| against the PDL-off output for out / s2, ids mismatches): T=1 sliding off 1.01e-3 / 7.19e-4 / 0, on 1.01e-3 / 7.19e-4 / 0 (strict 1e-4 fails for both arms: the PDL-off layer itself drifts 1e-3 between replays of fixed inputs; on is inside the off arm's drift, so T=1 on); **T=8 full off 1.68e-3 / 1.22e-3 / 0, on 8.39e-3 / 8.77e-3 / 0: FAIL, so the gate left T=8 off** (`enabled {1: True, 8: False}`). Scored cells (this run / 607eb52, us):

| cell | 8f6acddaca2e4c50 | 607eb52 | delta |
|---|---:|---:|---:|
| decode 1 sliding | 70.28 | 76.75 | -6.46 |
| decode 1 full | 75.83 | 83.73 | -7.90 |
| decode 16 sliding | 131.12 | 138.66 | -7.54 |
| decode 16 full | 210.84 | 214.53 | -3.69 |
| decode 32 sliding | 152.50 | 154.82 | -2.32 |
| decode 32 full | 307.71 | 297.88 | +9.83 |
| verify 1 sliding | 117.12 | 115.43 | +1.69 |
| verify 1 full | 119.52 | 117.60 | +1.93 |
| verify 16 sliding | 249.82 | 247.13 | +2.69 |
| verify 16 full | 330.89 | 333.22 | -2.33 |
| verify 32 sliding | 306.18 | 316.39 | -10.21 |
| verify 32 full | 470.26 | 482.66 | -12.40 |

The decode 1 gain matches the in-prepare A/B (7.2 / 6.9 us); verify 1 ran PDL off (control-equivalent); decode 16 to verify 32 ran unchanged code paths and sit inside the repeat draws' spread. Profile rows at decode 1 sliding: moe 22.6, pooler / router 18.0, qkv 10.1, o 6.1, attention 5.3, unpooler 4.4, b1 3.3, route 2.2, finish 1.8 us (the rows shrink less than the step: the gain is the overlap between kernels). Keep rule against the brief (six-load over 0.8% with decode 1 and verify 1 both improving): the six-load gain is 0.79% on the public receipt (10,639.4 -> 10,554.9), but verify 1 did not improve because T=8 stayed off. Lead decision: T=1 kept, T=8 hard off after the audit below, a second check extends the edges to T=16 and 32. Receipts: `muse_decode_layer/profiles/g1b4_pdl8.8f6acddaca2e4c50.{json,txt,log,status.json}`.

**T=8 stress failure, source audit (about 30 minutes, nothing found).** I read every T=8 consumer for its first read of a predecessor's output. Attention `<8, false, 15 / 11>`: `pdl_wait` comes before the q / new k,v `cp.async.cg` copies and before every gate-row read (`__ldcg` at all four sites, `cluster_stage_gate` `cp.async.cg`); its pre-wait work is page-index copies of the page table (input); the paths of CTAs without segments call `pdl_wait` before `cluster_stage_gate` / finalize. o sk2 at M = 8: x by TMA only after `griddepcontrol.wait` and `fence.proxy.async.global` in every thread. `glue_cluster_boundary<4, 32, false, false, true, true>`: its pre-wait reads are `data['stream']` (nothing in the call writes it) and weights; the branch is read with `ld.global.cg` after the wait; PRELOADED_BASE changes only an operation order. Fused pooler / router: x by TMA and x32 by `cp.async.bulk`, both after the wait and the proxy fence. Route and finish: every predecessor read is after `gdc_wait` with `.cg`. No consumer writes global memory before its wait, so allocator reuse of a freed buffer stays ordered. The qkv kernel at T = 8 (`sk2_kernel_narrow`) is a producer here, not a consumer: it reads only the layer input. Two observations bound the cause without naming it: with fixed inputs a read of the previous replay's values returns identical numbers, so the drift is not a stale-replay read; and its size (8e-3 of the output's maximum, no routing change) is a few BF16 rounding steps, which fits either a read of partly written current-replay data or a wider spread of FP32-atomic merge orders in the T = 8 attention when its CTAs start staggered. The audit does not separate the two. T = 8 stays hard off; the open question is in docs/ruled-out.md.

## 2026-11-03 - PDL8 check 2: T = 1 on, T = 8 off, T = 16 and 32 added

Source `muse_decode_layer/cand/g1b4_pdl8b.py` (SHA256 7f5651e55f397158c7c5928013a11ba330bf3c50c8559b9dcf7b0666cdde79de), generated by `tools/dl_pdl8b_patch.py` from ba9e327c (release bytes plus one appended block). T = 1 is the check-1 build. T = 8 is hard off (`PDL8_T = (1, 16, 32)`). T = 16 (graft path): qkv sk2 (no attribute, trigger) -> attention -> o sk2 -> graft glue b1 -> fused pooler / router sk2f -> `_sl_tile_route` (.cg) -> MoE (plain, cooperative) -> unpooler sk2 (no attribute, trigger) -> graft glue b2c; the glue kernels come from a separate module built from `_GRAFT_GLUE_CUDA` with the attribute set at run time, the entry wait kept only for SLICED / POST builds (b2c reads s1 after it), the stream and branch reads through `ld.global.cg`, and the trigger after the branch reads. T = 32: the same edges with the Gluon `_graft_row_boundary_pre` as b1 and b2, its branch and stream reads through `cache_modifier='.cg'` (a copy used only while PDL is on). The T = 16 / 32 kernels with changed loads run only with PDL on, so the PDL-off arm runs the release's kernels there. In prepare, per enabled T: A/B on decode-shaped inputs (ctx 80000, agentic-like plus or minus 10% lengths at batch 16 / 32), and a 2000-replay stress at decode 1 sliding, decode 16 sliding and decode 32 full; a failure, or a compile / runtime error in the eager try, turns that T off. Keep rule as registered in the directive ledger. Not compiled locally. PRIVATE `0cb69c8b68bd4d89` accepted (7 of 20 layer checks left before it).

**Result: VALID 13/13, 10,531.3 us, 2.1479x (95% CI 2.116 to 2.180)** against 10,639.4 (607eb52bdaf24160) and 10,554.9 (check 1). Prepare log, `PDL8_AB` median paired saving (9 of 9 rounds each): T=1 sliding 77.44 -> 72.06 us (5.31), full 84.35 -> 78.05 (6.34); T=16 sliding 167.62 -> 162.72 (4.96), full 252.58 -> 247.62 (5.15); T=32 sliding 221.38 -> 216.67 (4.54), full 391.07 -> 387.26 (3.71). `PDL8_STRESS` (2000 replays per arm, out / s2 / ids): T=1 sliding off 0 / 1.25e-4 / 0, on 0 / 1.25e-4 / 0; T=16 sliding off 6.37e-3 / 5.75e-3 / 0, on the same; T=32 full off 6.10e-3 / 6.06e-3 / 0, on the same. Every stress: on drift equal to the off arm's own drift (strict 1e-4 fails on both arms at T=16 / 32 because the PDL-off layer itself drifts); enabled {1, 16, 32}, T=8 off. Six-load estimate from the A/B +2.08%.

| cell | 0cb69c8b68bd4d89 | 607eb52 | delta |
|---|---:|---:|---:|
| decode 1 sliding | 71.30 | 76.75 | -5.45 |
| decode 1 full | 78.39 | 83.73 | -5.33 |
| decode 16 sliding | 129.21 | 138.66 | -9.45 |
| decode 16 full | 207.13 | 214.53 | -7.40 |
| decode 32 sliding | 147.13 | 154.82 | -7.69 |
| decode 32 full | 302.10 | 297.88 | +4.22 |
| verify 1 sliding | 119.33 | 115.43 | +3.90 |
| verify 1 full | 125.98 | 117.60 | +8.39 |
| verify 16 sliding | 245.01 | 247.13 | -2.12 |
| verify 16 full | 329.89 | 333.22 | -3.33 |
| verify 32 sliding | 311.62 | 316.39 | -4.77 |
| verify 32 full | 484.44 | 482.66 | +1.78 |

Verify 1 ran the release's T=8 path (PDL off); its +3.9 / +8.4 us sit on unchanged kernels (the MoE row moves with routing between runs). Keep rule met (score below 10,554.9, every stress at the off arm's drift). The lead submitted the file publicly under the user's standing authorization: public job `fefd7f7b8e834bd7` (result pending at the time of writing). Receipts: `muse_decode_layer/profiles/g1b4_pdl8b.0cb69c8b68bd4d89.{json,txt,log,status.json}`.

## 2026-11-03 - PDL8c composed candidate (built, not dispatched)

`tools/dl_pdl8c_patch.py CG_TOOL C2 [--front]` writes `muse_decode_layer/cand/g1b4_pdl8c.py` (SHA256 6539d8d6a6ae34079b296e319c01c9b088123a7c049166e69210481886126eb8): the release ba9e327c, then the parity lane's m = 1 cgemv block (`tools/dl_cg_patch.py` at d26e1d5b on worktree-agent-a51254a802fef2de0, C2 = `git show 3044ee7:muse_bf16_proj/submission.py`; the tool first checks that it still reproduces that lane's check-2 bytes 440397ab), then the pdl8b block unchanged (the pdl8b tool still reproduces 7f5651e5). `--front` adds that lane's split T = 1 front (cgemv pooler plus side-stream FP32 router, its own prepare A/B) for the case its check 3 (ec6a1c7beb0f443e) wins: `g1b4_pdl8cf.py`, SHA256 ff3483da7c1fbb41d91338f52ae04aad67b80dcf68a427aa9ef5c6e850337b81. Composition findings from the source: (a) the cgemv kernels (`cgemv_kernel` / `cgemv40_kernel`) are plain `<<<>>>` launches with no `griddepcontrol` and read x with `__ldg`; since they are never launched with the programmatic attribute, each starts only after its predecessor completes, so the `__ldg` reads are ordered and the PDL map stays correct; the cost is that the qkv -> attention, attention -> o and MoE -> unpooler-slice edges lose the overlap at T = 1 (the attention and b1 / finish secondaries start at the cgemv's exit instead of at a trigger); (b) the o cgemv takes the BF16 attention output and returns BF16 rows, and `_sl_tile_after_attention` still requires a BF16 o result, so the attention boundary is the release's; (c) the cgemv block's prepare (re-applying the tuner's m = 1 cgemv picks after the pinned-pick prepare) runs inside the pdl8b prepare, which sets the PDL map afterwards from `PDL8_HARD` and the stress gates; the cgemv numerics gate (rel L2 < 2e-3 against sk2 per projection on a fixed input) is the parity lane's, unchanged. The PDL A/B and stress therefore measure the composed T = 1 chain. Not compiled locally; waiting for the lead's dispatch.

## 2026-11-03 - Component parity audit of the public layer (no quota)

Inputs: the public layer profile `g1-stage-release.607eb52bdaf24160.txt` (ba9e327c, 10,639.4 us), the PDL lane's
check `g1b4_pdl8.8f6acddaca2e4c50.txt` (PDL on at T = 1, copied into `muse_decode_layer/profiles/`), and the
best judged component profiles: BF16 `9eba96894ee1402e` (C2, source 3044ee7), attention `8914751e3b9f468b`,
MoE public draw `2821fdb184d740b1` (5,738.0 us, copied as `muse_moe_w4a16/profiles/scored-draw2.2821fdb184d740b1.txt`).
Every number is a single profiled replay (judge `GPU time`), so a row's gap carries the profile's run-to-run
spread (about 0.2-0.5 us on the small kernels, tens of us on the MoE row). The tool is `tools/dl_parity_audit.py`
(`table`, `rank`, `glue` modes). The value column is 46 or 16 divided by 372 (six loads of 62) divided by the
cell's endpoint, in percent of score per microsecond; the two decode 1 cells sum to the directive's 0.212%.

Kernel identity per cell comes from the sk2 template prefixes (`SK2` in the tool) and, at 128 / 256 tokens,
from the BF16 board's cuBLASLt picks for the same shapes; at verify 16 the two 128x64 calls are o and the
unpooler together. The pooler + router component is the board's pooler plus its router; at T <= 8 the board's
router takes BF16 x, while the layer routes on the FP32 router input with FP32 weights, so that row is an
upper bound on what a port can recover. The MoE rows compare different routings (the layer's own router
against the board's fixture) and are not a parity measurement: the layer runs the same `c7c201ae` engine.

| cell | kernel | layer us (607eb52) | layer us (pdl8 8f6acdd) | component us | gap us | % score per us | weighted gap, % score |
|---|---|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | moe | 29.8 | 22.6 | 29.6 | 0.2 | 0.161 | 0.032 |
| decode 1 sliding | pooler+router | 17.9 | 18.0 | 16.3 | 1.6 | 0.161 | 0.258 |
| decode 1 sliding | qkv_gate | 10.5 | 10.1 | 8.0 | 2.5 | 0.161 | 0.403 |
| decode 1 sliding | o | 6.0 | 6.1 | 5.0 | 1.0 | 0.161 | 0.161 |
| decode 1 sliding | attention | 5.4 | 5.3 | 4.5 | 0.9 | 0.161 | 0.145 |
| decode 1 sliding | unpooler | 4.4 | 4.4 | 2.4 | 2.0 | 0.161 | 0.322 |
| decode 1 full | moe | 42.8 | 23.1 | 29.6 | 13.2 | 0.051 | 0.678 |
| decode 1 full | pooler+router | 17.9 | 19.1 | 16.3 | 1.6 | 0.051 | 0.082 |
| decode 1 full | attention | 12.1 | 11.7 | 10.5 | 1.6 | 0.051 | 0.082 |
| decode 1 full | qkv_gate | 10.6 | 10.5 | 8.0 | 2.6 | 0.051 | 0.134 |
| decode 1 full | o | 5.9 | 5.8 | 5.0 | 0.9 | 0.051 | 0.046 |
| decode 1 full | unpooler | 4.5 | 4.5 | 2.4 | 2.1 | 0.051 | 0.108 |
| verify 1 sliding | moe | 71.6 | 65.2 | 84.0 | -12.4 | 0.107 | -1.328 |
| verify 1 sliding | pooler+router | 18.7 | 19.2 | 17.2 | 1.5 | 0.107 | 0.161 |
| verify 1 sliding | qkv_gate | 9.1 | 9.2 | 7.6 | 1.5 | 0.107 | 0.161 |
| verify 1 sliding | attention | 7.4 | 7.3 | 6.3 | 1.1 | 0.107 | 0.118 |
| verify 1 sliding | o | 6.0 | 5.9 | 4.9 | 1.1 | 0.107 | 0.118 |
| verify 1 sliding | unpooler | 4.7 | 4.6 | 3.6 | 1.1 | 0.107 | 0.118 |
| verify 1 full | moe | 58.8 | 51.1 | 84.0 | -25.2 | 0.037 | -0.922 |
| verify 1 full | pooler+router | 18.6 | 18.8 | 17.2 | 1.4 | 0.037 | 0.051 |
| verify 1 full | attention | 15.0 | 14.8 | 14.4 | 0.6 | 0.037 | 0.022 |
| verify 1 full | qkv_gate | 9.3 | 9.3 | 7.6 | 1.7 | 0.037 | 0.062 |
| verify 1 full | o | 6.1 | 5.9 | 4.9 | 1.2 | 0.037 | 0.044 |
| verify 1 full | unpooler | 4.5 | 4.5 | 3.6 | 0.9 | 0.037 | 0.033 |
| decode 16 sliding | moe | 77.0 | 60.2 | 93.1 | -16.1 | 0.089 | -1.436 |
| decode 16 sliding | pooler+router | 18.9 | 18.4 | 18.1 | 0.8 | 0.089 | 0.071 |
| decode 16 sliding | unpooler | 14.9 | 14.7 | 13.4 | 1.5 | 0.089 | 0.134 |
| decode 16 sliding | qkv_gate | 10.6 | 9.8 | 9.1 | 1.5 | 0.089 | 0.134 |
| decode 16 sliding | attention | 8.0 | 8.1 | 7.1 | 0.9 | 0.089 | 0.080 |
| decode 16 sliding | o | 6.1 | 5.9 | 4.4 | 1.7 | 0.089 | 0.152 |
| decode 16 full | attention | 87.5 | 89.5 | 84.7 | 2.8 | 0.020 | 0.056 |
| decode 16 full | moe | 65.2 | 65.6 | 93.1 | -27.9 | 0.020 | -0.559 |
| decode 16 full | pooler+router | 17.3 | 17.5 | 18.1 | -0.8 | 0.020 | -0.016 |
| decode 16 full | unpooler | 14.6 | 14.4 | 13.4 | 1.2 | 0.020 | 0.024 |
| decode 16 full | qkv_gate | 10.4 | 9.7 | 9.1 | 1.3 | 0.020 | 0.026 |
| decode 16 full | o | 6.1 | 6.0 | 4.4 | 1.7 | 0.020 | 0.034 |
| decode 32 sliding | moe | 70.6 | 74.1 | 126.2 | -55.6 | 0.080 | -4.441 |
| decode 32 sliding | pooler+router | 20.4 | 21.4 | 19.7 | 0.7 | 0.080 | 0.056 |
| decode 32 sliding | unpooler | 16.0 | 16.2 | 15.1 | 0.9 | 0.080 | 0.072 |
| decode 32 sliding | qkv_gate | 11.4 | 10.9 | 10.2 | 1.2 | 0.080 | 0.096 |
| decode 32 sliding | attention | 9.6 | 9.3 | 9.2 | 0.4 | 0.080 | 0.032 |
| decode 32 sliding | o | 6.2 | 6.2 | 5.1 | 1.1 | 0.080 | 0.088 |
| decode 32 full | attention | 170.0 | 170.6 | 160.3 | 9.7 | 0.014 | 0.140 |
| decode 32 full | moe | 87.6 | 63.7 | 126.2 | -38.6 | 0.014 | -0.557 |
| decode 32 full | pooler+router | 19.6 | 20.4 | 19.7 | -0.1 | 0.014 | -0.001 |
| decode 32 full | unpooler | 15.9 | 15.0 | 15.1 | 0.8 | 0.014 | 0.012 |
| decode 32 full | qkv_gate | 11.4 | 10.8 | 10.2 | 1.2 | 0.014 | 0.017 |
| decode 32 full | o | 6.2 | 6.3 | 5.1 | 1.1 | 0.014 | 0.016 |
| verify 16 sliding | moe | 155.3 | 136.4 | 179.3 | -24.0 | 0.050 | -1.201 |
| verify 16 sliding | o+unpooler | 28.5 | 29.7 | 24.2 | 4.3 | 0.050 | 0.215 |
| verify 16 sliding | pooler+router | 25.4 | 25.1 | 26.8 | -1.4 | 0.050 | -0.070 |
| verify 16 sliding | qkv_gate | 16.9 | 16.8 | 13.9 | 3.0 | 0.050 | 0.150 |
| verify 16 sliding | attention | 11.8 | 11.9 | 10.8 | 1.0 | 0.050 | 0.050 |
| verify 16 full | moe | 178.1 | 130.3 | 179.3 | -1.2 | 0.013 | -0.015 |
| verify 16 full | attention | 104.1 | 102.1 | 99.6 | 4.5 | 0.013 | 0.058 |
| verify 16 full | o+unpooler | 28.6 | 28.6 | 24.2 | 4.4 | 0.013 | 0.057 |
| verify 16 full | pooler+router | 24.9 | 26.3 | 26.8 | -1.9 | 0.013 | -0.025 |
| verify 16 full | qkv_gate | 16.8 | 17.6 | 13.9 | 2.9 | 0.013 | 0.037 |
| verify 32 sliding | moe | 204.8 | 201.4 | 227.8 | -23.0 | 0.039 | -0.899 |
| verify 32 sliding | pooler+router | 31.5 | 31.2 | 33.3 | -1.8 | 0.039 | -0.070 |
| verify 32 sliding | unpooler | 25.6 | 25.6 | 23.5 | 2.1 | 0.039 | 0.082 |
| verify 32 sliding | qkv_gate | 20.7 | 20.3 | 20.4 | 0.3 | 0.039 | 0.012 |
| verify 32 sliding | attention | 15.4 | 15.4 | 14.8 | 0.6 | 0.039 | 0.023 |
| verify 32 sliding | o | 9.1 | 8.9 | 8.1 | 1.0 | 0.039 | 0.039 |
| verify 32 full | moe | 203.3 | 195.5 | 227.8 | -24.5 | 0.009 | -0.218 |
| verify 32 full | attention | 186.2 | 185.6 | 178.6 | 7.6 | 0.009 | 0.068 |
| verify 32 full | pooler+router | 31.8 | 31.1 | 33.3 | -1.5 | 0.009 | -0.013 |
| verify 32 full | unpooler | 25.4 | 25.2 | 23.5 | 1.9 | 0.009 | 0.017 |
| verify 32 full | qkv_gate | 20.5 | 19.9 | 20.4 | 0.1 | 0.009 | 0.001 |
| verify 32 full | o | 9.0 | 8.9 | 8.1 | 0.9 | 0.009 | 0.008 |

Largest gaps, weighted (both cells of a load summed, MoE excluded):

| rank | kernel at load | gap us (sliding + full) | % of score | portable? | what blocks it |
|---|---|---:|---:|---|---|
| 1 | qkv_gate at decode 1 | 5.1 | 0.54 | yes | Same 12-bit lossless code and the same BF16 [1, 8192] input and BF16 [1, 2304] output at the boundary; only the weight layout differs (C2's cgemv reads two row-major planes, the layer's sk2 reads tile units), so a second packed copy (28 MB) at prepare is the whole cost. The layer has no cgemv at all (the G fork never carried it). |
| 2 | unpooler slice at decode 1 | 4.1 | 0.43 | yes | As above: C2 `cgemv40<128, 2>` on the 1024-row slice, BF16 MoE output in, BF16 out; 6.3 MB second copy. |
| 3 | pooler + router at decode 1 | 3.2 | 0.34 (upper bound) | partly | The pooler half is portable (C2 `rd_kernel<1, 2, 16, 8, 1>`, 13.5 us, BF16 x), but C2's router reads BF16 x and the layer must route on FP32 x with FP32 weights at T <= 8, and the layer fuses the two into one launch (`sk2_kernel_gle`, 17.9 us). A port means the rd source (larger than cgemv) plus a separate FP32 router launch beside it; the fused-versus-split choice already exists in the layer's tuner (`bfpf` against `bfp`). |

Next: o at decode 1 (1.9 us, 0.21%, same cgemv port, C2 `cgemv_kernel<64, 1>`); qkv_gate and pooler + router at
verify 1 (0.22% and 0.21%, C2's sk geometries differ from sk2's: `sk_kernel<1, 8, 3, 1, 4, 4, 4>` at 7.6 us
against the narrow sk2 at 9.1); attention at decode 1 (2.5 us, 0.23%, our `attn_one_kernel` against the G fork's
`attn_kernel`; blocked by a different kernel family and ABI in the G layer and by the BF16 output boundary
noted in ruled-out row "G #303 full-context attention").

Glue per cell (judge GPU time of the glue launches only; the launch boundaries themselves are not in these rows):

| cell | boundary 1 | route | finish | row boundaries | glue sum (607eb52) | glue sum (pdl8) | % score per us | glue, % score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 3.4 | 2.0 | 2.3 | - | 7.7 | 7.3 | 0.161 | 1.241 |
| decode 1 full | 3.3 | 2.0 | 1.9 | - | 7.2 | 7.2 | 0.051 | 0.370 |
| verify 1 sliding | 3.0 | 1.9 | 1.7 | - | 6.6 | 6.7 | 0.107 | 0.707 |
| verify 1 full | 3.2 | 2.0 | 1.9 | - | 7.1 | 7.0 | 0.037 | 0.260 |
| decode 16 sliding | 3.1 | 2.0 | 3.4 | - | 8.5 | 8.8 | 0.089 | 0.758 |
| decode 16 full | 3.5 | 2.1 | 3.7 | - | 9.3 | 9.3 | 0.020 | 0.186 |
| decode 32 sliding | - | 2.1 | - | 6.9 | 9.0 | 8.8 | 0.080 | 0.719 |
| decode 32 full | - | 2.1 | - | 7.5 | 9.6 | 9.4 | 0.014 | 0.139 |
| verify 16 sliding | - | 2.8 | - | 9.7 | 12.5 | 12.0 | 0.050 | 0.625 |
| verify 16 full | - | 2.8 | - | 9.4 | 12.2 | 12.2 | 0.013 | 0.157 |
| verify 32 sliding | - | 3.6 | - | 13.6 | 17.2 | 17.5 | 0.039 | 0.672 |
| verify 32 full | - | 3.6 | - | 13.4 | 17.0 | 18.3 | 0.009 | 0.151 |

At decode 1 the glue is 7.2-7.7 us per cell (1.6% of score over both cells). The donor's T = 1 front
(`muse_decode_layer.3a5f879df0d44825.txt`) spends `_gle_route` 2.6 us where we spend boundary 3.4 plus route
2.0, but it gets there by moving boundary 1 into the o epilogue with deferred normalization, which moves the
BF16 rounding of the pre-FFN activation and reads producer partials before the PDL wait
(`tools/dl_build_attributed_fork.py` sets `GLE_T = {}` for exactly that reason). Boundary 1 and the route
cannot share a launch without that deferral, because the pooler GEMM sits between them. The bitwise-identical
alternatives (last-arriver counters in the o, pooler or unpooler kernels) are the F3, C-1 and expert-to-unpooler
slices in docs/ruled-out.md, which measured 0.45-1.3 us per removed launch at best and lost or tied as built.
Decision for today's checks: the cgemv port (gaps 1, 2 and the o row, one hypothesis) goes first; no glue fusion
check is spent unless that port lands and a concrete single-round-trip design remains.


## 2026-11-03 - C2 m = 1 cgemv port into the public layer (checks 6e8939c91ce648dd, 8cfa2056a6f6461b)

Tool `tools/dl_cg_patch.py` appends one block to the release bytes (ba9e327c) and copies C2's GEMV source and
packing (3044ee7, public 9eba96894ee1402e) verbatim under `cg_` names. The block packs qkv_gate, o and the
unpooler's 1024-row slice into C2's planar 12-bit row stream, builds the extension on a thread at the top of
prepare, gates each projection against the release's own sk2 output on a fixed input, and offers cgemv to the
layer's bf16 tuner as an m = 1 candidate.

| date | job | source | change | score (CI) | cells and decision |
|---|---|---|---|---|---|
| 2026-11-03 | 6e8939c91ce648dd (check) | `g1b4_cg_release.py` 579e23ee | cgemv as an m = 1 bf16 tuner candidate | 10,786.6 us, 2.0971x (2.0656-2.1285) | The tuner picked cgemv for all three (cold-L2 graph medians cg / sk2: qkv_gate 12.06 / 13.31 us, o 8.35 / 9.06, slice 6.82 / 7.68), but the release's pinned-pick prepare (`_stack_unpinned_prepare`, line 15279) resets every T <= 32 choice to the static sk2 pick afterwards, so the timed path ran sk2 (profile rows unchanged). Control-equivalent run: d1 75.81 / 83.24, v1 120.24 / 124.36. Not a test of the hypothesis. |
| 2026-11-03 | 8cfa2056a6f6461b (check) | `g1b4_cg_release.py` 440397ab | as above, with the tuner's m = 1 cgemv picks re-applied after the pinned-pick prepare | 10,726.7 us, 2.109x (2.077-2.140) | Timed path ran cgemv at T = 1: profile qkv_gate 9.0 / 9.1 us (sk2 10.5-10.7), o 5.3 / 5.3 (5.9-6.4), slice 3.5 / 3.4 (4.4-4.5). d1 74.64 / 79.89 us, below all ten release-byte draws (75.32-77.62 / 83.24-84.41: eight repeats, 607eb52, check 1); v1 and higher loads inside the control range (T = 1 only changes). Numerics gate: 2303/2304, 8191/8192, 1023/1024 elements bitwise equal to sk2. Keep rule fails on the six-load score (bar 10,642; d32 sliding drew 160.0). Not kept; ruled-out row. |

Reading: the port recovers about 3.2 us of judge GPU time per decode 1 cell (2.2 / 4.1 us of endpoint against the
control medians, about 0.56% of score), which is the predicted direction but about 60% of the profile-to-profile
gap in the audit (5.5 us): in the layer cgemv runs 0.9-1.1 us slower than on the BF16 board for the same shape
(qkv_gate 9.0 against 8.0, slice 3.5 against 2.4). A hypothesis for the remainder, not measured: on the board the
kernel's x is the only activation in L2 and the weights stream from a quiet HBM, while in the layer the previous
kernel's tail and write-back share the first microseconds. The score bar is set by the release's own
draw-to-draw spread (1.3%), which is larger than the effect, so a single check of this size cannot clear it.
| 2026-11-03 | ec6a1c7beb0f443e (check) | `g1b4_cgf_release.py` 11ccb8e9 (`tools/dl_cg_patch.py --front`) | check 2 plus, at T = 1, C2's `cgemv<512, 1>` pooler with the release's FP32 router GEMV on a side stream instead of the fused `sk2_kernel_gle`, picked by a prepare A/B (`CG_FRONT`) | 10,715.4 us, 2.111x (2.079-2.143) | Prepare A/B fused 22.40 / split 20.42 us, 15/15 rounds; gate pooled rel 7.95e-6, logits rel 2.02e-7. Profile: pooler cgemv 14.6 / 14.1 us, router 6.8 us on the side stream. d1 74.43 / 79.09 (check 2: 74.64 / 79.89), v1 115.31 / 120.37. Keep rule fails on the score; not kept; lane closed. |

Combined reading of checks 2 and 3: decode 1 sliding / full sit at 74.4-74.6 / 79.1-79.9 us with the port against
75.3-77.6 / 83.2-84.4 for ten release-byte draws, about 2.5 / 4.6 us lower (about 0.6% of score), and no other load
moves. The six-load score of a single draw spreads 10,642-10,787 us on identical bytes, so the bar in the brief
(below the release's draw range) needs about 1.4% of real gain, which this T = 1 change cannot supply on its own.
PDL8c check 3: PRIVATE `eba613c023414ca8` accepted, source `g1b4_pdl8c.py` (6539d8d6, commit c3081dd8), keep rule as registered in the directive ledger.

**Result (measured composition result): VALID 13/13, 10,460.9 us, 2.1624x (95% CI 2.130 to 2.195).** cgemv gates ok (rel L2 against sk2: qkv_gate 4.30e-5, o 4.16e-5, slice 4.96e-4) and picked by the tuner (cold-L2 us, cg / sk2: qkv_gate 12.35 / 13.47, o 8.74 / 9.38, slice 7.01 / 8.00). `PDL8_AB` T=1 saving fell to 2.24 / 2.78 us (sliding / full) from 5.31 / 6.34 in pdl8b; T=16 5.34 / 5.95, T=32 3.49 / 6.08. Stress, PDL on against PDL off: T=1 sliding 7.46e-3 / 6.60e-3 / 0 on both arms, T=16 sliding 2.84e-3 / 2.48e-3 / 0 on both, T=32 full 5.52e-3 / 4.50e-3 / 0 on both; enabled {1, 16, 32}. Cells (this run / pdl8b 0cb69c8b68bd4d89, us): decode 1 74.05 / 71.30 sliding, 77.77 / 78.39 full; decode 16 129.31 / 129.21, 206.85 / 207.13; decode 32 150.70 / 147.13, 298.92 / 302.10; verify 1 111.64 / 119.33, 122.60 / 125.98 (PDL off at T=8 in both); verify 16 249.03 / 245.01, 336.45 / 329.89; verify 32 321.32 / 311.62, 485.50 / 484.44. Profile rows at decode 1 sliding: qkv cgemv40<256,2> 9.3, o cgemv<64,1> 5.2, slice cgemv40<128,2> 3.4 us (sk2 rows in pdl8b: 10.1, 6.1, 4.4). Decision: the decode 1 clause of the keep rule fails (74.05 / 77.77 against 71.30 / 78.39); the six-load score is below 10,531.3 mainly through verify 1 sliding, a PDL-off cell. Mechanism identified from the A/B: the cgemv kernels are plain launches with no `griddepcontrol`, so the three T = 1 edges into them (qkv, o, slice) lost the overlap that the sk2 versions had, about 3 us, which is about what the faster kernels gained. This is a composition result, not a closed idea; pdl8c2 (the cgemv kernels as PDL participants) is the test. Receipts: `muse_decode_layer/profiles/g1b4_pdl8c.eba613c023414ca8.{json,txt,log,status.json}`; the public pdl8b job's receipts are beside them as `g1b4_pdl8b.fefd7f7b8e834bd7.{json,txt,log}`.

## 2026-11-03 - PDL8c2 (built, not dispatched): the cgemv kernels as PDL participants at T = 1

`tools/dl_pdl8c_patch.py CG_TOOL C2 --cgpdl` writes `muse_decode_layer/cand/g1b4_pdl8c2.py` (SHA256 d07b512bd1b15a675794bf9f3271107b63a0ed2334396ae340b3c87bb449c8b0): g1b4_pdl8c.py (the tool still reproduces 6539d8d6) plus one block that edits the cgemv source (`CG_CUDA_SRC`) under `PDL_CHAIN`. In `cgemv_body` the compute warps issue their weight loads (unchanged `ld.global.nc.L1::no_allocate` / `__ldg`), then `griddepcontrol.launch_dependents`, then `griddepcontrol.wait`, then read x through `__ldcg`; the escape warp waits after its first table load and reads its x elements through `__ldcg`; the output store follows the CTA barrier, after every warp's wait. `cgemv_launch` goes through `cudaLaunchKernelEx`, adding `programmaticStreamSerializationAllowed = 1` only while a host flag (`cg_set_pdl`, exported) is set, and the Python `_cg_call` sets it only while `_PDL8` is on, which happens only at T = 1 (cgemv runs only at m = 1, and `PDL8_T` has no other T with m = 1). Edges that regain overlap at T = 1: qkv cgemv -> attention (cgemv triggers), attention -> o cgemv (o is a secondary), o cgemv -> b1, MoE -> unpooler-slice cgemv (the slice is a secondary of the cooperative MoE: its CTAs launch only after every MoE CTA has triggered or exited, so they take only resources left after the whole MoE grid is resident, and they read the MoE output only after the wait), slice -> finish. The pdl8b prepare's T = 1 A/B and 2000-replay stress at decode 1 sliding cover these consumers, with abort-to-off. Not compiled locally; waiting for the lead.

PDL8c2 check 4: PRIVATE `65f0a3be76a1499b` accepted, source `g1b4_pdl8c2.py` (d07b512b, commit 7d18a50f), keep rule as registered in the directive ledger. Result pending.

## 2026-11-03 - PDL8d check 5: per-edge PDL at T = 8

PRIVATE `4a737387f88b4d7d`, source `muse_decode_layer/cand/g1b4_pdl8d.py` (SHA256 7cb74240d6ab5edd1c72e869777637157d17c67b615d508565a6084496028904, commit 1514839f), generated by `tools/dl_pdl8d_patch.py` from the pdl8b block (the tool checks that pdl8b still reproduces 7f5651e5 and that no CUDA edit and no T = 16 / 32 launch changed). T = 1 / 16 / 32 run pdl8b's launches. At T = 8 each of pdl8's six edges is named by its consumer launch (attn, o, b1, pool, route, fin). An edge that is off drops only that consumer's programmatic attribute: the griddepcontrol code, the .cg / TMA reads and the producer triggers all stay. In prepare, the tool runs 2000-replay stresses at verify 1 full (all edges, each half, each edge alone, then the union of the edges that were clean alone), followed by a fresh stress of the chosen set at full and at sliding and the paired A/B. The edge set is fixed in prepare. No CUDA source changed, so the local nvcc step does not apply (`py_compile` and a mocked run of the bisection logic were the local gates).

**Result: VALID 13/13, 10,443.9 us, 2.166x (95% CI 2.133 to 2.198)**, prepare 128.3 s (each stress 0.4-0.5 s).

| T = 8 stress, verify 1 full (out / s2 max rel diff to the PDL-off output; ids mismatches 0 everywhere) | on drift | verdict |
|---|---|---|
| off arm (own drift) | 0 / 2.61e-8 | reference |
| all six edges | 6.67e-3 / 5.93e-3 | drift |
| half 1: attn + o + b1 | 1.67e-3 / 1.16e-3 | drift |
| half 2: pool + route + fin | 0 / 2.61e-8 | clean |
| attn alone (qkv -> attention) | 6.67e-3 / 5.93e-3 | **drift** |
| o alone (attention -> o sk2) | 6.67e-3 / 5.93e-3 | **drift** |
| b1 alone | 0 / 2.61e-8 | clean |
| pool alone | 0 / 2.61e-8 | clean |
| route alone | 0 / 2.61e-8 | clean |
| fin alone | 0 / 2.61e-8 | clean |
| union b1 + pool + route + fin | 0 / 2.61e-8 | clean |
| final b1 + pool + route + fin, full | 0 / 2.61e-8 | clean |
| final b1 + pool + route + fin, sliding (off arm 6.37e-3 / 5.37e-3) | 6.37e-3 / 5.37e-3 | clean |

The off arm at verify 1 full repeated bit-exactly here (0 / 2.61e-8; check 1 saw 1.68e-3 / 1.22e-3 on other data), so this stress separates the edges cleanly. The two drifting edges are the ones on either side of the T = 8 attention. That half 1 drifts less than either of its members alone suggests the drift is a discrete outcome (which ordering the attention merge takes), not a magnitude that adds up; this is a hypothesis, untested. A/B for the shipped set: sliding 134.43 -> 132.22 us (2.27), full 142.08 -> 138.85 (2.91), 9 of 9 each. Other T: saving T = 1 5.98 / 5.73, T = 16 5.34 / 4.80, T = 32 4.38 / 4.93 us; every stress on drift equal to off drift.

| cell | 4a737387f88b4d7d | pdl8b 0cb69c8b68bd4d89 | delta |
|---|---:|---:|---:|
| decode 1 sliding | 70.8 | 71.30 | -0.5 |
| decode 1 full | 77.0 | 78.39 | -1.4 |
| decode 16 sliding | 127.7 | 129.21 | -1.5 |
| decode 16 full | 205.2 | 207.13 | -1.9 |
| decode 32 sliding | 149.4 | 147.13 | +2.3 |
| decode 32 full | 306.8 | 302.10 | +4.7 |
| verify 1 sliding | 112.9 | 119.33 | -6.4 |
| verify 1 full | 118.7 | 125.98 | -7.3 |
| verify 16 sliding | 245.1 | 245.01 | +0.1 |
| verify 16 full | 336.9 | 329.89 | +7.0 |
| verify 32 sliding | 321.0 | 311.62 | +9.4 |
| verify 32 full | 483.9 | 484.44 | -0.5 |

Keep rule (registered in the directive ledger) met: score below 10,531.3, every stress clean, verify 1 below 119.33 / 125.98. The verify 1 drop (6.4 / 7.3 us) is larger than the in-prepare A/B (2.3 / 2.9 us), so part of it is likely draw noise (the MoE row moves with routing); the public file's own draw at verify 1 was 118.00 / 120.63. The six-load score is 0.23% above the public pdl8b run (10,419.9, fefd7f7b8e834bd7), inside the 1.3% draw spread. Not submitted by this owner; the decision is the lead's. Receipts: `muse_decode_layer/profiles/g1b4_pdl8d.4a737387f88b4d7d.{json,txt,log,status.json}`.

## 2026-11-03 - PDL8e (built; box dry run, layer check after the 16:00 PT reset)

Source `muse_decode_layer/cand/g1b4_pdl8e.py` (SHA256 27b5f25236ceda79934798dac5595cc37395e071f1c8f3cd65f93f5c064b130f), generated by
`tools/dl_pdl8e_patch.py` from the pdl8d block (the tool checks that pdl8d still reproduces 7cb74240 and that no CUDA edit and no
T = 16 / 32 launch changed). It changes only the T = 8 gate. Attention check 2c84c321b3bb4dbb (docs/runs-attention.md) showed that
the drift on the two attention edges is the attention's FP32 merge order: the order already varies without PDL, and with an
order-free merge (n_splits = nc = 8) both edges are bitwise clean. The new gate runs four stages at verify 1 full and at sliding,
all with the six edges on. `det` and `det_fresh` run the attention with the order-free plan in both arms, the second with two
inputs alternating per replay inside one graph, and PDL-on must stay within the off arm's own drift. `fresh` and `fixed` run the
production plan, within 1e-2 of the PDL-off output of the same input and with ids equal. All six edges turn on only if every stage
passes and the A/B saving is positive; otherwise the gate falls back to pdl8d's four edges under pdl8d's rule, else T = 8 is off.

Box dry run (okc-h200, H200, /root/venv per docs/box.md: torch 2.14.1+cu130, CUDA 13.0; prepare on reference weights seed 1;
`.build/pdl8e/box_prepare.py`; logs `muse_decode_layer/profiles/box/g1b4_pdl8{e,d}.box_prepare.log`). The box is an A/B tool,
not a judge replica.

| stage (pdl8e, six edges) | layer | off (out / s2) | on (out / s2) | verdict |
|---|---|---|---|---|
| det | full | 3.68e-3 / 2.96e-3 | 3.68e-3 / 2.96e-3 | clean |
| det_fresh | full | 3.68e-3 / 2.96e-3 | 3.68e-3 / 2.96e-3 | clean |
| fresh | full | 7.35e-3 / 6.89e-3 | 7.35e-3 / 6.89e-3 | clean (bound 1e-2) |
| fixed | full | 7.30e-3 / 6.89e-3 | 7.30e-3 / 6.89e-3 | clean (bound 1e-2) |
| det | sliding | 0 / 1.51e-6 | 0 / 1.51e-6 | clean |
| det_fresh | sliding | 1.84e-3 / 1.64e-3 | 1.84e-3 / 1.64e-3 | clean |
| fresh | sliding | 7.30e-3 / 5.55e-3 | 7.87e-3 / 6.25e-3 | clean (bound 1e-2) |
| fixed | sliding | 7.30e-3 / 5.55e-3 | 7.30e-3 / 5.55e-3 | clean (bound 1e-2) |

ids mismatches were 0 everywhere. Gate decision: all six edges on at T = 8. A/B saving: sliding 119.84 -> 114.72 us (4.93), full
134.69 -> 128.45 (5.86), 9 of 9 each, weighted 5.17 us (pdl8d's four edges saved 2.27 / 2.91 at the judge). The other T are as in
pdl8d (T = 1 / 16 / 32 stress clean). Prepare took 23.3 s on the box. The det off arm is not bitwise at full here (3.68e-3), so on
the box the layer has order noise downstream of the attention too, and the det stage bounds a race by that noise rather than by 0.

Control, pdl8d on the same box: its off arm at T = 8 full drifted 7.35e-3 by itself, so all six edges passed at full, but the
sliding final stress drifted (s2 6.39e-3 against off 5.09e-3) and the gate turned T = 8 off. The same order noise thus flips
pdl8d's rule either way, depending on whether the off arm happens to repeat; pdl8e's stages do not depend on that.

### PDL8e check `61c009066a5d40a6` (g1b4_pdl8e.py 27b5f252)

**Result: VALID 13/13, 10,441.7 us, 2.166x (95% CI 2.134 to 2.199)** against pdl8d's check 10,443.9 (4a737387f88b4d7d) and the
public pdl8d run f931abf568544018. At T = 8 the new gate turned all six edges on, both attention edges included. Stages at the
judge (out / s2 against the PDL-off output; ids mismatches 0 everywhere):

| stage | layer | off | on | verdict |
|---|---|---|---|---|
| det | full | 7.25e-3 / 6.15e-3 | 7.25e-3 / 6.15e-3 | clean |
| det_fresh | full | 7.25e-3 / 6.15e-3 | 7.25e-3 / 6.15e-3 | clean |
| fresh | full | 7.25e-3 / 6.66e-3 | 7.25e-3 / 6.66e-3 | clean (bound 1e-2) |
| fixed | full | 3.40e-3 / 2.83e-3 | 6.80e-3 / 4.67e-3 | clean (bound 1e-2) |
| det | sliding | 9.26e-4 / 1.15e-3 | 9.26e-4 / 1.15e-3 | clean |
| det_fresh | sliding | 9.26e-4 / 1.15e-3 | 9.26e-4 / 1.15e-3 | clean |
| fresh | sliding | 7.30e-3 / 7.02e-3 | 7.30e-3 / 7.02e-3 | clean (bound 1e-2) |
| fixed | sliding | 7.30e-3 / 7.02e-3 | 7.30e-3 / 7.02e-3 | clean (bound 1e-2) |

The paired A/B saving at T = 8 was 4.70 us at sliding and 5.79 at full (9 of 9 each), weighted 4.98 us; pdl8d's four edges saved
2.27 / 2.91 at the judge. **But the unchanged pdl8b stress at T = 16 drifted: s2 was 4.40e-3 on against 3.70e-3 off (out 5.78e-3
on both), so T = 16 ran PDL off in this check.** T = 1 and T = 32 passed. Prepare log `PDL8 six-load estimate +2.53%`, enabled
{1, 8, 32}.

| cell | 61c009066a5d40a6 | pdl8d 4a737387f88b4d7d | delta |
|---|---:|---:|---:|
| decode 1 sliding | 71.03 | 70.8 | +0.2 |
| decode 1 full | 78.22 | 77.0 | +1.2 |
| decode 16 sliding | 133.05 | 127.7 | +5.4 (PDL off at T = 16) |
| decode 16 full | 210.28 | 205.2 | +5.1 (PDL off at T = 16) |
| decode 32 sliding | 142.54 | 149.4 | -6.9 |
| decode 32 full | 299.32 | 306.8 | -7.5 |
| verify 1 sliding | 112.21 | 112.9 | -0.7 |
| verify 1 full | 115.51 | 118.7 | -3.2 |
| verify 16 sliding | 250.00 | 245.1 | +4.9 |
| verify 16 full | 332.46 | 336.9 | -4.4 |
| verify 32 sliding | 308.05 | 321.0 | -13.0 |
| verify 32 full | 483.94 | 483.9 | +0.0 |

Keep rule (six-load below 10,443.9, every stage clean at every T, both T = 8 attention edges on): **not met.** The score is 2.2 us
lower, inside the draw spread, and the T = 16 stress was not clean. Both attention edges were on and every T = 8 stage was clean.
Verify 1 dropped by 0.7 / 3.2 us against pdl8d's check, less than the 2.4 / 2.9 us difference between the two arms' in-prepare
A/B savings; the MoE row moves with routing between draws.

The T = 16 failure is the same kind of order noise: the T = 16 path in pdl8e is pdl8d's code, which passed this stress in
4a737387f88b4d7d. pdl8b's rule (PDL-on within the off arm's own drift, on one fixed input) fails whenever the on arm happens to
pick a merge order that the off arm did not reach in its 2000 replays. The next step is to give T = 1 / 16 / 32 the same
det / fresh gate as T = 8, so that the abort-to-off decision depends on a race test rather than on that draw.
Receipts: `muse_decode_layer/cand/profiles/g1b4_pdl8e.61c009066a5d40a6.{json,log,txt}`.

## 2026-11-03 - PDL8f (built; box dry run; layer check pending go)

Source `muse_decode_layer/cand/g1b4_pdl8f.py` (SHA256 c4164566f0315cfc9059558b1531bf0b9e1ce844fd137d437df355b7ccb509cc), generated
by `tools/dl_pdl8f_patch.py` from the pdl8e block (the tool checks that pdl8e still reproduces 27b5f252 and that no CUDA edit and
no T = 16 / 32 launch changed). It replaces pdl8b's single fixed-input stress at T = 1 / 16 / 32 with pdl8e's four stages (det,
det_fresh, fresh, fixed) on the same stressed layer per T (T = 1 sliding, T = 16 sliding, T = 32 full). For B > 1 the order-free
plan is per-request with n_splits = nc = 8 and no flat kernel. It also changes the pass rule at every T, T = 8 included: a stage
passes when ids are equal and every PDL-on value is within max(2e-2, the off arm's own drift).

Why the rule changed. The first pdl8f build (rules as in pdl8e; box log `g1b4_pdl8f_v1.box_prepare.log`) turned T = 8 and T = 32
off on the box. T = 8 det_fresh at full drifted (on 1.69e-3 / 3.33e-3 against off 9.62e-4 / 1.18e-3) with the order-free
attention, so the layer has order noise downstream of the attention too, and "within the off arm's own drift" is a draw there as
well. At T = 32 the fresh stage failed on the PDL-off arm alone (s2 1.00e-2 against the 1e-2 bound). The same coin flip appears in
the box control: pdl8d turned T = 16 and T = 32 off in both of its paired runs below (for example T = 16 on 6.37e-3 against off
3.58e-3) and T = 8 on.

Sensitivity of the new bound, measured on the box (`tools/dl_pdl8f_box_fault.py`, log `g1b4_pdl8f.fault.log`): the attention fed
the other input's qkv rows under alternating inputs, worst of 5 replays, out / s2:

| shape | all rows stale | one token stale | one q head stale | one new K row stale |
|---|---|---|---|---|
| T8 full | 1.58 / 1.48, ids 300 | 0.78 / 0.76, ids 39 | 0.27 / 0.26, ids 26 | 8.2e-3 / 6.1e-3, ids 0 |
| T8 sliding | 1.16 / 1.46, ids 285 | 1.04 / 1.20, ids 33 | 0.24 / 0.22, ids 22 | 7.8e-3 / 9.2e-3, ids 3 |
| T1 sliding | 1.13 / 1.08, ids 35 | 1.13 / 1.08, ids 35 | 0.31 / 0.38, ids 19 | 7.8e-3 / 7.8e-3, ids 4 |
| T16 sliding | 1.01 / 1.19, ids 570 | 0.94 / 0.99, ids 30 | 0.18 / 0.22, ids 23 | 1.4e-2 / 1.4e-2, ids 0 |
| T32 full | 1.25 / 1.12, ids 1165 | 1.05 / 1.01, ids 35 | 0.45 / 0.51, ids 35 | 6.1e-3 / 6.3e-3, ids 0 |

A stale q head or a whole stale token exceeds 2e-2 by 9x or more and also flips ids. A single stale new K row of one token stays
under the bound wherever ids do not flip; the old own-drift rule cannot see it either, since off arms drift up to about 7e-3. That
case is covered by the attention-board bench 2c84c321b3bb4dbb, whose alternating-input order-free arms are bitwise.

Box dry run of the final build (log `g1b4_pdl8f.box_prepare.log`, prepare 29.8 s): every stage clean at every T and PDL on at
T = 1, 8 (all six edges), 16 and 32. The largest PDL-on value was 1.00e-2 (T = 32 fresh s2, off 5.57e-3), and ids were equal
everywhere. In-prepare A/B savings: T = 1 5.73 / 6.50, T = 8 5.09 / 5.28, T = 16 4.96 / 4.61, T = 32 3.62 / 4.03 us
(sliding / full).

Box paired timing (box rule): pdl8d (7cb74240) against pdl8f in one session, in the order d, f, f, d, with the same router seed
(106218864389449120) and the f931abf568544018 case seeds, 60 runs, net us, mean of the two runs per file (logs
`pdl8f_pair_*.log`). In both of its runs the pdl8d control had T = 8 on with all six edges and T = 16 / 32 off, so the verify 1
row compares the same T = 8 configuration.

| cell | pdl8d (box) | pdl8f (box) | pdl8f - pdl8d |
|---|---:|---:|---:|
| decode 1 sliding | 68.59 | 68.44 | -0.15 |
| decode 1 full | 74.51 | 74.27 | -0.24 |
| decode 16 sliding | 137.77 | 131.45 | -6.32 |
| decode 16 full | 209.72 | 202.59 | -7.13 |
| verify 1 sliding | 104.78 | 104.80 | +0.02 |
| verify 1 full | 112.61 | 111.75 | -0.86 |

The decode 16 gain is the T = 16 PDL saving that pdl8d's rule dropped on this box draw. Caveat: in pdl8d's own T = 32 stress on
the box, the PDL-off arm's ids changed between replays (2524 and 2668 mismatches). If the judge's T = 32 data does the same,
pdl8f's ids-equal clause turns T = 32 off; that is the safe direction.

### PDL8f check `9ae1d02f0c2e4244` (g1b4_pdl8f.py c4164566)

**Result: VALID 13/13, 10,549.1 us, 2.144x (95% CI 2.112 to 2.176)** against pdl8e's check 10,441.7 (61c009066a5d40a6). PDL was
on at T = 1, 8 (all six edges) and 32, but **T = 16 turned off on the ids clause**. In its fresh stage the PDL-off arm itself
changed routing ids between replays (52 mismatches against the reference of the same input) and the PDL-on arm changed 184; the
float values were equal (5.59e-3 / 4.29e-3 in both arms). Every other stage was clean (out / s2; ids 0 unless noted):

| T | layer | det | det_fresh | fresh | fixed |
|---|---|---|---|---|---|
| 1 | sliding | 2.40e-3 / 2.41e-3 both arms | same as det | 9.30e-3 / 7.96e-3 both | same as fresh |
| 8 | full | 2.02e-3 / 1.52e-3 both | 3.82e-3 / 3.87e-3 both | 8.00e-3 / 7.29e-3 both | 5.73e-3 / 6.18e-3 both |
| 8 | sliding | 3.33e-3 / 2.79e-3 both | same as det | off 1.66e-3 / 2.59e-3, on 6.62e-3 / 5.59e-3 | off 6.82e-4 / 7.30e-4, on 3.97e-3 / 5.57e-3 |
| 16 | sliding | 2.79e-3 / 2.70e-3 both | 3.01e-3 / 2.80e-3 both | 5.59e-3 / 4.29e-3 both, **ids off 52, on 184: T off** | not run |
| 32 | full | 0 / 0 both | 0 / 0 both | 6.10e-3 / 6.78e-3 both | 6.06e-3 / 6.78e-3 both |

In-prepare A/B savings: T = 1 5.66 / 7.14, T = 8 4.61 / 5.76, T = 32 4.29 / 5.54 us (sliding / full).

| cell | 9ae1d02f0c2e4244 | pdl8e 61c009066a5d40a6 | pdl8d 4a737387f88b4d7d |
|---|---:|---:|---:|
| decode 1 sliding | 70.05 | 71.03 | 70.8 |
| decode 1 full | 77.43 | 78.22 | 77.0 |
| decode 16 sliding | 136.69 | 133.05 | 127.7 |
| decode 16 full | 211.01 | 210.28 | 205.2 |
| decode 32 sliding | 152.66 | 142.54 | 149.4 |
| decode 32 full | 302.00 | 299.32 | 306.8 |
| verify 1 sliding | 107.47 | 112.21 | 112.9 |
| verify 1 full | 112.33 | 115.51 | 118.7 |
| verify 16 sliding | 243.62 | 250.00 | 245.1 |
| verify 16 full | 322.85 | 332.46 | 336.9 |
| verify 32 sliding | 310.15 | 308.05 | 321.0 |
| verify 32 full | 473.31 | 483.94 | 483.9 |

Keep rule (six-load below 10,441.7, every stage clean with PDL on at T = 1 / 8 / 16 / 32, decode 16 at or below 127.7 / 205.2):
**not met** on all three clauses. T = 16 was off and decode 16 is 136.69 / 211.01. The six-load of 10,549.1 is higher, even though
every cell except decode 16 / 32 sliding is at or below pdl8e's check, because the SGLang column of this draw differs (for example
verify 16 sliding SGLang 503.5 against 540.3 in pdl8e's draw) and the score is the paired ratio. Verify 1 is the lowest of the
three checks (107.47 / 112.33), with both attention edges on. Per the lead's stop rule, no further check.

Finding: routing ids also move with merge-order noise. At T = 16 under alternating inputs, the PDL-off arm flipped ids on 52
entries by itself, so "ids equal" is the same kind of draw as "within the off arm's own drift". A gate that does not flip on
noise needs an ids criterion relative to the off arm (for example on mismatches no larger than a multiple of the off arm's,
with the float bound applied only to rows whose ids match), or a stress input whose routing has no near ties.
Receipts: `muse_decode_layer/cand/profiles/g1b4_pdl8f.9ae1d02f0c2e4244.{json,log,txt}`.

## 2026-11-03 - PDL8g (built; box dry run turned T = 8 off; not dispatched)

Source `muse_decode_layer/cand/g1b4_pdl8g.py` (SHA256 58fe7f009a83b1851caa5ea45d8c1b9843d421de7833dd5aab19e264cc775051), generated
by `tools/dl_pdl8g_patch.py` from the pdl8f block (pin c4164566; no CUDA edit and no T = 16 / 32 launch changed). It changes only
how the stress inputs are built. Every stress input (x0 and the alternating x1, at every T) is resampled row by row until each
token's gap between its 8th and 9th router selection score (sigmoid(logit / sqrt(32)) + bias, read from the route kernel's
launch arguments in eager calls only) is at least 1e-3. Before use, the input must keep identical ids over 50 PDL-off graph
replays. Stages, bound and fallbacks are pdl8f's, and stage lines also print the ids-mismatch counts on / off.

Box dry run (`muse_decode_layer/profiles/box/g1b4_pdl8g.box_prepare.log`, prepare 22.2 s):

| T | layer | input build (min gap, resample rounds) | det | det_fresh | fresh | fixed | ids on / off |
|---|---|---|---|---|---|---|---|
| 1 | sliding | x0 2.79e-3 (0), x1 1.21e-3 (0) | clean | clean | clean | clean | 0 / 0 |
| 16 | sliding | x0 1.39e-3 (1), x1 1.08e-3 (1) | clean | clean | clean | clean | 0 / 0 |
| 32 | full | x0 1.08e-3 (2), x1 1.16e-3 (1) | clean | clean | clean | clean | 0 / 0 |
| 8 | full | x0 1.69e-3 (0); **x1 failed: ids moved on 52 entries over 50 PDL-off replays** | not run | | | | |

The input-stability check raised, so T = 8 went off through the existing exception path; T = 1 / 16 / 32 were on. Per the lead's
rule (stop if the box run turns any T off), no layer check was dispatched.

What this shows and what it does not: at T = 1 / 16 / 32 the tie-free inputs removed the ids flips (0 / 0 everywhere, against 52
off-arm flips at T = 16 in 9ae1d02f0c2e4244). At T = 8 full, an x1 whose score gaps were at least 1e-3 in the eager call still
moved 52 id entries over 50 PDL-off replays, about one per replay. Two explanations are not separated: (1) order noise at T = 8
full moves the selection scores by more than 1e-3; (2) the graph-replayed path routes from scores that differ from the eager
call the gaps were read from (for example a different route kernel or router layout under capture). A per-replay readout of the
graph's own route scores would separate them; it was not run.

Lane status per the lead's last-gate rule: closed with pdl8d public. The T = 8 attention edges stay a measured saving (judge
in-prepare A/B 4.6-5.8 us per verify 1 cell across 61c009066a5d40a6 and 9ae1d02f0c2e4244) waiting for a gate that does not flip on
order noise.
## 2026-11-03 - Routing slack at T = 1 (g1b4_rs, check `f8a55f1043804158`)

Source `muse_decode_layer/cand/g1b4_rs.py` (SHA256 01426b891ea12af13e8f713c9ca3a9f47900a07798212e53f583ed0005ff7f07),
written by `tools/dl_rs_patch.py` from g1b4_pdl8d.py (the tool checks 7cb74240 first). One launch differs: at T = 1 the
route program of `_sl_tile_route_cg` becomes `_rs_tile_route1`. It reads the same FP32 router logits (.cg, after
griddepcontrol.wait), forms the same biased scores and sort key, takes a top 16 instead of a top 8, then applies TASK.md's
near-tie slack. With one token every local pick is its own expert and a replacement cannot be one the call already uses,
so the selection is closed-form: drop the k lowest-scoring local picks whose lead over the 9th best is at most 0.02
(KEEP_TOL 0.03 minus a 0.010 margin) and take the k best non-local experts ranked 9th or lower within 0.035 of the 8th
best (ROUTE_TOL 0.05 minus 0.015). k is capped so distinct experts and routes stay 0.25 above their bounds, with the
reference count raised by any local expert within 0.012 of the 8th best (one the judge's own top 8 may hold through a
near tie). Weights are the picked unbiased scores over their sum. The pooled-normalize programs, every PDL edge and gate,
and every other T are pdl8d's code.

Why T = 1 only. On the box the layer's scores at T = 8 sit up to 0.019 from the judge's FP64 ones (the layer's verify
attention uses FP8 queries; the scores are closer to the E4M3-query reference than to the BF16 one). The quantities the
slack thresholds are off by up to 0.021 (lead over the 9th best) and 0.026 (distance below the 8th best), against
tolerances of 0.03 and 0.05. A first build that also ran at T = 8 failed check_implementation on 1 of 20 inputs per
verify 1 cell for exactly that reason (a dropped pick 3.05e-2 above its 9th best). At T = 1 the same errors are at most
5.4e-3 and 1.16e-2 over 800 inputs (`tools/box_rs_check.py`).

Two earlier T = 1 builds lost on the box and are not dispatched. A multi-token greedy route program (whole-expert drops,
local or non-local replacements, 8 sequential argmax passes; commit f312f87f) raised decode 1 by 2.4 to 2.9 us net in two
paired rounds although moe_fused shrank; the same greedy on one 256-element sort took 7.0 / 7.3 us against the base
program's 4.2 / 4.1 us. The closed-form program takes 4.9 / 4.6 us (`--kernels`, logs
`muse_decode_layer/profiles/box/rs/rs_k_*.log`). Its first version allowed at most one drop per input (count margin 1.0
on the reference's own count) and measured -0.7 / -0.5 us net, -2.0 / -1.3 us shown in one run; the dispatched version
allows a second drop where the bound permits it and measures the same within noise (table below). The judge's
statistic is a median over inputs, and most inputs keep at least one of their two experts either way.

Correctness on the box (okc-h200, router seed 106218864389449120, the f931 public case seeds, local_start per input from
case_fields; `tools/box_rs_check.py`, logs `muse_decode_layer/profiles/box/rs/rs2_{d1,rest,exact}.log`). Every input runs
kernel_fn eagerly and is judged by reference.check_implementation; margins are measured under the reference's FP64 scores.

| set | inputs | pass | distinct ref -> picked | min margin keep (0.03) | window (0.05) | routes | distinct |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 400 | 400 | 2.46 -> 1.76 | 0.0098 | 0.0180 | 2.16 | 0.30 |
| decode 1 full | 400 | 400 | 2.51 -> 1.88 | 0.0090 | 0.0176 | 2.16 | 0.30 |
| T = 1 tests (ctx 10 sliding, 3000 full, 80000 full) | 60 | 60 | 2.55 / 2.40 / 2.55 -> 1.65 / 1.75 / 1.70 | 0.0118 | 0.0255 | 2.16 | 0.30 |
| other 10 test shapes | 200 | 200 | unchanged (pdl8d path) | | | | |
| verify 1 sliding / full | 60 / 60 | 120 | unchanged (pdl8d path) | | | | |
| decode 16 / 32, verify 16 / 32 | 10 per cell | 80 | unchanged (pdl8d path) | | | | |
| decode 1, all margins 0 | 400 | 398 | 2.46 / 2.52 -> 1.50 / 1.57 | -0.0005 | 0.0170 | 2.16 | 0.30 |

The zero-margin run lands on the keep bound (two inputs fail with a dropped pick 3.03e-2 / 3.05e-2 above its 9th best):
the margins are what keeps the file inside it. A distinct margin of 0.30 is an input whose two local experts were both
dropped (0 against a bound of -0.3).

Paired box A/B, candidate against pdl8d bytes (`.build/repeats/layer_base.py`), same session, same seeds, 60 fresh inputs
per cell, run order control, candidate, control, candidate (`tools/box_pair.py`, logs
`muse_decode_layer/profiles/box/rs/rs2_ab_*.log`). "shown" is the judge's statistic (SGLang median over the median paired
ratio), averaged over the two runs per side.

| cell | control net | candidate net | per-input ratio | control shown | candidate shown | distinct ref -> picked |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 68.54 | 67.68 | 0.937 | 69.01 | 67.24 | 2.48 -> 1.82 |
| decode 1 full | 74.50 | 73.81 | 0.950 | 74.92 | 73.44 | 2.43 -> 1.78 |
| verify 1 sliding | 109.90 | 110.40 | 0.997 | 109.34 | 109.74 | unchanged |
| verify 1 full | 118.96 | 118.67 | 1.006 | 118.36 | 117.26 | unchanged |
| decode 16 sliding / full | 132.11 / 208.35 | 132.27 / 208.53 | 1.000 / 1.000 | 133.01 / 209.88 | 132.94 / 208.75 | unchanged |
| decode 32 sliding / full | 150.26 / 301.58 | 153.25 / 305.49 | 1.020 / 1.008 | 149.86 / 301.20 | 153.03 / 305.23 | unchanged |
| verify 16 sliding / full | 244.80 / 326.82 | 244.35 / 327.79 | 0.999 / 1.001 | 242.17 / 325.43 | 242.66 / 326.29 | unchanged |
| verify 32 sliding / full | 298.16 / 462.00 | 298.02 / 462.10 | 1.002 / 1.001 | 296.21 / 463.04 | 296.43 / 464.07 | unchanged |

Only decode 1 runs different code. The other rows follow pdl8d's own per-process PDL gates, which differed between the
four processes (T = 8 on in control 1 and candidate 2, T = 16 on in control 2 and candidate 1, T = 32 on in control 2
only: the decode 32 row). The T = 1 gate was on in all four.

### Judge check `f8a55f1043804158`

VALID 13/13, 10,297.4 us (2.197x, CI 2.164-2.230), prepare log `muse_decode_layer/cand/profiles/g1b4_rs.f8a55f1043804158.log`.

| cell | pdl8d check 4a737387 | this check | delta |
|---|---:|---:|---:|
| decode 1 sliding / full | 70.8 / 77.0 | 65.5 / 74.1 | -5.3 / -2.9 |
| decode 16 sliding / full | 127.7 / 205.2 | 127.1 / 211.4 | -0.6 / +6.2 |
| decode 32 sliding / full | 149.4 / 306.8 | 153.1 / 300.7 | +3.7 / -6.1 |
| verify 1 sliding / full | 112.9 / 118.7 | 109.7 / 122.6 | -3.2 / +3.9 |
| verify 16 sliding / full | 245.1 / 336.9 | 248.8 / 331.2 | +3.7 / -5.7 |
| verify 32 sliding / full | 321.0 / 483.9 | 322.4 / 485.7 | +1.4 / +1.8 |

PDL gates (pdl8d code): T = 1 on, T = 16 on, T = 32 on, T = 8 off (six-edge final stress 5.14e-3 against the off arm's
1.71e-3). Only decode 1 runs different code; its two cells moved -5.3 / -2.9 us against pdl8d's check, more than the box's
-1.8 / -1.5 (the judge's router draw and per-input routing differ from the box's). The other cells are a pdl8d draw with
T = 8 off. Keep rule: score and decode 1 met; verify 1 full (122.6 against 118.7) and the T = 8 stress clause missed because
pdl8d's own T = 8 gate aborted on this draw. Not submitted by the lane.

## 2026-11-03 - Routing slack at T = 16 / 32 (g1b4_rs2, box only, not built for dispatch)

Question: can the TASK.md near-tie slack that g1b4_rs uses at T = 1 also cut distinct local experts at decode 16 and
decode 32, by at least 2 us per cell more than the extra route-program cost? Answer from the box: no. No judge check
was spent. Logs are in `muse_decode_layer/profiles/box/rs2/`, all on okc-h200 with router seed 106218864389449120,
the f931 case seeds, and local_start set per input from case_fields.

Per-expert slope at T = 16 / 32 (`tools/box_route_map.py` on g1b4_rs, 60 inputs per cell, least squares; `map.log`).
The fused MoE costs 5.6 / 5.5 us per distinct local expert at decode 16 (sliding / full) and 5.3 / 5.5 us at decode 32.
With routes as a second regressor the fit is 5.0-5.4 us per expert plus 0.1-0.5 us per route. The net layer time rises
by 5.0 to 5.9 us per expert. So the slope is a little lower than at T = 1 (6.3 us), not higher.

Score error at T = 16 / 32 (`tools/box_rs_check.py` on g1b4_rs, 200 inputs per cell, 800 in all; `err.log`). These
loads take their router logits from the bfp router projection, not the T = 1 FP32 GEMV. The error is larger than at
T = 1:

| quantity the slack thresholds | T = 1 (round 1, 800 inputs) | T = 16 / 32 (800 inputs) |
|---|---:|---:|
| lead over the 9th best, max error | 5.4e-3 | 9.97e-3 (decode 32 full) |
| distance below the 8th best, max error | 1.16e-2 | 1.43e-2 (decode 32 sliding) |

The margins used were keep 0.016, window 0.021, rank eps 0.016 and count 0.25, with the same headroom ratio as round 1.

Slack model (`tools/box_route_slack.py`, 200 inputs per cell; it now takes `--margins keep,window,count`; logs
`slack.log` and `slack2.log`). It counts local experts that could be skipped, with non-local replacements only:

| cell | distinct (ref) | skippable, no margins | margins 0.010 / 0.015 / 0.25 | margins 0.016 / 0.021 / 0.25 |
|---|---:|---:|---:|---:|
| decode 16 sliding | 7.49 | 1.94 | 1.35 | 0.95 |
| decode 16 full | 7.63 | 1.99 | 1.43 | 1.14 |
| decode 32 sliding | 9.05 | 2.15 | 1.52 | 1.08 |
| decode 32 full | 9.21 | 2.15 | 1.59 | 1.23 |

The model does not include the rank-eps correction. That correction raises the reference counts by local experts
that might sit in the judge's top 8 through a near tie, and with 16 or 32 tokens it binds. The prototype
`tools/dl_rs2_patch.py` builds g1b4_rs2.py (SHA256 84e47ce3...; it was a measurement build and is not committed).
It routes the T = 16 / 32 launches of both after-attention paths through `_rs_tile_routeN`, one program for all tokens.
It handles both paths because prepare's compose policy picks the graft path or the `_route_split` path per process.
Checked on 50 inputs per cell (`v_*.log`; `tools/box_rs_check.py` now takes `--set NAME=VALUE`):

| rank eps | decode 16 sliding | decode 16 full | decode 32 sliding | decode 32 full | pass |
|---|---:|---:|---:|---:|---:|
| 0.016 (margins as built) | 7.68 -> 7.32 | 7.88 -> 7.30 | 9.04 -> 8.90 | 9.18 -> 9.04 | 200/200 |
| 0.010 (no room over the 9.97e-3 error) | 7.68 -> 7.06 | 7.88 -> 7.06 | 9.04 -> 8.56 | 9.18 -> 8.40 | 200/200 |
| 0 | 7.68 -> 6.70 | 7.88 -> 6.64 | 9.04 -> 7.92 | 9.18 -> 7.92 | 199/200 |

At rank eps 0 one decode 16 sliding input fails ("6 of this GPU's experts touched where the reference touches 10"),
so the correction is load-bearing. At the safe setting the slack removes 0.36 / 0.58 / 0.14 / 0.14 experts per input.
At 5.5 us per expert that is 2.0 / 3.2 / 0.8 / 0.8 us per cell before any route-program cost, and it misses the
2 us bar at three of four cells even if the route program were free. At rank eps 0.010 the saving would be
3.4 / 4.5 / 2.6 / 4.3 us, but that setting has no room over the measured error.

Route-program cost (`--kernels`, 10 cold replays, `k1_ctl.log` / `k1_cand.log`). The base route kernel runs 3.95 /
3.94 us at decode 16 and 4.80 / 5.20 us at decode 32. The single-program prototype runs 42.4 / 42.5 and
104.3 / 104.1 us, and the layer net rises from 131.6 / 203.2 / 148.0 / 298.7 to 172.6 / 250.5 / 257.6 / 408.2 us.
moe_fused moved by -1.8 / -5.6 / -0.4 / +0.0 us. A per-token top 16 with an atomic-ticket last-CTA decision would
probably cost about 2 us over the base. That is an estimate; I did not build it, and it cannot rescue a saving of
0.8 to 3.2 us.

Decision: stop. Nothing was registered or dispatched. Ruled-out row in docs/ruled-out.md.


### November 3 23:43 UTC monitor clarification: RS2 scope

The retained complete-call box comparison also changed prepared state: control enabled T16/T32 PDL, while the candidate disabled both. Its full-call differences do not isolate routing cost. Independent route-kernel profiles (about 42/104 us versus 4-5 us) support stopping this exact single-program prototype. The safe-margin expert-count model gives 2.0/3.2/0.8/0.8 us before route cost; the earlier blanket free-route-below-2-us wording is too broad, especially T16 full. The alternative approximately 2-us route-overhead idea is unbuilt, not a measured rescue or family ceiling. No judge check or promotion follows. See docs/steering/2026-11-03-driver-monitor-2343.md for pins, prepared-mask confound and the existing box-revival/efficiency allocations.
## 2026-11-03 - PDL8h: PDL on at every T at build time behind a coarse guard (check `b6c0efcd06324246`)

Source `muse_decode_layer/cand/g1b4_pdl8h.py` (SHA256 33c2353b52de29c5af341609de08a7e6f021e5395fc4ef633706978fecedf7cd),
written by `tools/dl_pdl8h_patch.py` from g1b4_rs.py (the tool checks 01426b89 first). User decision: enable PDL at build
time at T = 1, 8 (all six edges), 16 and 32, on the source audit, the bitwise attention bench 2c84c321b3bb4dbb and the clean
T = 8 stages of 61c00906 / 9ae1d02f, and keep the in-prepare stress only as a guard against gross errors. Changes against
g1b4_rs: `PDL8_HARD` has T = 8 on; the T = 1 / 16 / 32 stress decision and pdl8d's T = 8 edge search are replaced by one
guard. Every CUDA source, launch attribute, edge wiring and the T = 1 routing slack are byte-identical (83 changed diff lines, all in
the gate code and the header comment).

Guard, per T on pdl8d's existing stress inputs (T = 1 sliding, T = 8 full and sliding, T = 16 sliding, T = 32 full; 2000
graph replays per arm, PDL-off and PDL-on, against the PDL-off arm's first replay): the T stays on when the PDL-on routing ids
mismatch on at most 2 x (off-arm mismatches) + 16 entries and every PDL-on output is within max(2e-2, the off arm's own drift).
Otherwise that T aborts to off; any exception turns it off, as before. Sensitivity: the stale-read faults in the PDL8f table
move outputs by 0.18 to 1.6 and flip 19 to 1165 ids, so they fail both clauses; a single stale new K row stays under the
bound, as it did for every earlier gate, and is covered by the attention bench.

Box (okc-h200, `flock /root/okc/.build/box/gpu.lock`, router seed 106218864389449120, f931abf568544018 case seeds; logs
`muse_decode_layer/profiles/box/pdl8h/`). The two dry runs (pdl8h, then g1b4_rs) overlapped an unlocked box_rs_check job of
another lane, so their in-prepare timings are not used; their gate outcomes are. Guard, identical in all four pdl8h prepares
(dry run plus three A/B runs) except where noted (out / s2):

| T | layer | ids off / on (limit) | off drift | on drift | decision |
|---|---|---|---|---|---|
| 1 | sliding | 0 / 0 (16) | 1.99e-3 / 3.16e-3 | 1.99e-3 / 3.16e-3 | on |
| 8 | full | 0 / 0 (16) | 7.25e-3 / 6.89e-3 | 7.25e-3 / 6.89e-3 | on |
| 8 | sliding | 0 / 0 (16) | 3.16e-3 / 3.98e-3 | 6.33e-3 / 6.09e-3 | on (pdl8d's own-drift rule fails this row) |
| 16 | sliding | 0 / 0 (16) | 3.58e-3 / 4.08e-3 (dry 3.98e-3 / 4.19e-3) | up to 3.98e-3 / 4.19e-3 | on |
| 32 | full | 0 / 0 (16) | 5.52e-3 / up to 6.41e-3 | 5.52e-3 / 6.41e-3 | on |

Paired box A/B (box rule: box-candidate against box-control, same session and seeds, 60 fresh inputs per cell, run order
ctl1, cand1, ctl2, cand2, ctl3, cand3; ctl3 / cand3 added after the coordinator's note that ctl1 overlapped unlocked jobs of
another lane). Control PDL enable sets from their logs: ctl1 {1, 16, 32 on; 8 off}, ctl2 {1, 16, 32 on; 8 off}, ctl3
{1, 8 on; 16, 32 off}. Every candidate run had {1, 8, 16, 32} on. Net us, candidate pooled over its three runs:

| cell | ctl1 | ctl2 | ctl3 | median control | candidate | cand - median ctl | cand - ctl2 (most T on, unoverlapped) |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 67.28 | 67.60 | 67.62 | 67.60 | 67.95 | +0.35 | +0.35 |
| decode 1 full | 73.98 | 74.16 | 74.03 | 74.03 | 73.82 | -0.21 | -0.34 |
| verify 1 sliding | 112.82 | 112.94 | 105.15 | 112.82 | 105.02 | -7.80 | -7.92 |
| verify 1 full | 120.51 | 121.07 | 111.86 | 120.51 | 112.02 | -8.49 | -9.06 |
| decode 16 sliding | 131.68 | 131.90 | 138.06 | 131.90 | 131.57 | -0.33 | -0.34 |
| decode 16 full | 203.33 | 202.93 | 210.45 | 203.33 | 202.89 | -0.44 | -0.04 |
| decode 32 sliding | 147.63 | 147.87 | 153.39 | 147.87 | 147.50 | -0.37 | -0.38 |
| decode 32 full | 298.91 | 299.39 | 305.02 | 299.39 | 298.32 | -1.07 | -1.07 |

The deltas decompose by enable set: where a control run had a T on, the candidate matches it within 1.1 us (verify 1 against
ctl3: -0.13 / +0.16; decode 16 / 32 against ctl1 and ctl2: within 1.07); where the control's gate had turned a T off, the
candidate is 5.9 to 9.1 us faster (verify 1 against ctl1 / ctl2, decode 16 / 32 against ctl3). pdl8h does not change the
speed of any configuration; it removes the draw that turns a T off.

### Judge check `b6c0efcd06324246` (g1b4_pdl8h.py 33c2353b)

**Result: VALID 13/13, 10,377.1 us, 2.180x (95% CI 2.147 to 2.213)** against g1b4_rs's check 10,297.4 (f8a55f1043804158) and
the public run 10,340 (9d80ae2494334136). Prepare log `muse_decode_layer/cand/profiles/g1b4_pdl8h.b6c0efcd06324246.log`.

Guard (out / s2, 2000 replays per arm):

| T | layer | ids off / on (limit) | off drift | on drift | decision |
|---|---|---|---|---|---|
| 1 | sliding | 0 / 0 (16) | 7.01e-3 / 7.19e-3 | 7.01e-3 / 7.19e-3 | on |
| 8 | full | 0 / 0 (16) | 6.37e-3 / 6.21e-3 | 6.37e-3 / 6.21e-3 | on |
| 8 | sliding | 0 / 0 (16) | 7.02e-4 / 2.53e-4 | 4.21e-3 / 5.13e-3 | on (pdl8d's own-drift rule would have turned T = 8 off here) |
| 16 | sliding | 0 / 0 (16) | 4.60e-3 / 5.29e-3 | 4.60e-3 / 5.29e-3 | on |
| 32 | full | 0 / 0 (16) | 5.56e-3 / 6.08e-3 | 5.56e-3 / 6.08e-3 | on |

In-prepare A/B savings: T = 1 7.01 / 6.75, T = 8 3.97 / 6.24, T = 16 5.57 / 4.93, T = 32 3.74 / 5.09 us (sliding / full).
Enabled {1, 8, 16, 32}.

| cell | g1b4_rs check f8a55f10 (yours / SGLang) | this check (yours / SGLang) | yours delta |
|---|---:|---:|---:|
| decode 1 sliding | 65.5 / 169.9 | 65.9 / 169.9 | +0.4 |
| decode 1 full | 74.1 / 180.2 | 71.7 / 179.9 | -2.4 |
| decode 16 sliding | 127.1 / 262.2 | 129.3 / 259.3 | +2.2 |
| decode 16 full | 211.4 / 440.2 | 207.3 / 433.2 | -4.1 |
| decode 32 sliding | 153.1 / 265.5 | 153.2 / 256.8 | +0.1 |
| decode 32 full | 300.7 / 632.1 | 303.9 / 609.4 | +3.2 |
| verify 1 sliding | 109.7 / 259.1 | 107.1 / 258.5 | -2.6 |
| verify 1 full | 122.6 / 276.4 | 117.7 / 281.1 | -4.9 |
| verify 16 sliding | 248.8 / 530.1 | 240.3 / 503.5 | -8.5 |
| verify 16 full | 331.2 / 697.8 | 331.0 / 684.7 | -0.2 |
| verify 32 sliding | 322.4 / 738.6 | 316.7 / 722.4 | -5.7 |
| verify 32 full | 485.7 / 1,103.5 | 475.8 / 1,064.0 | -9.9 |

Keep rule: VALID 13/13 met; every T on with the guard clean met; verify 1 below 109.7 / 122.6 met (107.1 / 117.7); routing
margins unchanged (the T = 1 route program is byte-identical). **Not met:** six-load 10,377.1 is above 10,297.4, and decode 16
sliding 129.3 is above 127.1 (full 207.3 is below 211.4). Not kept, not for submission.

Reading (hypothesis, not separated by this one draw): our times are at or below the g1b4_rs check in 8 of 12 cells, and the
gate now does what it was built for (every T on, including a T = 8 sliding row that pdl8d's rule fails). The score moved the
other way because this draw's SGLang column is lower in most long cells (verify 16 sliding 503.5 against 530.1, verify 32
full 1,064.0 against 1,103.5, decode 32 full 609.4 against 632.1), and the score is the paired ratio. The box A/B predicted
about -8 us per verify 1 cell against a T = 8-off control; f8a55f10 had T = 8 off, and verify 1 moved -2.6 / -4.9 at the
judge. A second draw of the same bytes would separate the draw from the file; that is the lead's quota call.

## 2026-11-03 - Slack-free rebuild: pdl8hn and combo2n on pdl8d (box only, no check)

The organizers disqualified the routing-slack file g1b4_rs (final-day ledger, 17:45 PT), so pdl8h and combo2, which
were built on it, are rebuilt on g1b4_pdl8d (7cb74240d6ab5edd1c72e869777637157d17c67b615d508565a6084496028904).

| file | SHA256 | built by | diff |
|---|---|---|---|
| muse_decode_layer/cand/g1b4_pdl8hn.py | df5cded96b8dc18ab98e022c8a6fde9c4ea18d0d7f85a81a9adcce2b63959968 | `python3 tools/dl_pdl8h_patch.py --base pdl8d` (checks the pdl8d SHA) | against pdl8d: the PDL block header line, PDL8_HARD (T = 8 True), the guard functions and the two gate call sites; against pdl8h: exactly tools/dl_rs_patch.py's two hunks reversed (applying them to pdl8hn gives pdl8h byte for byte) |
| muse_decode_layer/cand/rv_k_combo2n.py | 3ac83fb437d05308c2871301a5a10c537d9b2ec504d55305403641beec2f5d27 | `python3 tools/rv_combo_patch.py --base pdl8hn` (runs `tools/rv_cgfe_rs_patch.py --base pdl8hn`, which writes rv_cgfe_hn.py 0b315b61) | against pdl8hn: the same 582 changed lines as combo2 against pdl8h; the only differing line is the generated header (`rv_cgfe_hn ... g1b4_pdl8hn's PDL tail` for `rv_cgfe_h ... g1b4_pdl8h's PDL tail`). The rs hunks applied to combo2n, plus that header name, give combo2 byte for byte |

Checks (Mac): `grep -c _rs_tile_route1` is 0 in both files, and no RS_* margin or `_rs_route_launch` remains; a second build
gives the same two SHAs; the release prefix up to the PDL block (with `_release_expert_self_test` / `_release_routes`) is
byte-identical to pdl8d in pdl8hn, and the T = 1 route call is pdl8d's `_sl_tile_route_cg` line in both. pyflakes: pdl8hn
reports pdl8d's 17 messages plus two `undefined name 'data'` in `_pdl8h_t8` (the lambda-after-`del` pattern pdl8d's own
stress has, and pdl8h reports the same 19); combo2n reports the same 20 messages as combo2. The default builds still give
pdl8h 33c2353b and combo2 7f0a3e36.

Box (okc-h200, 2026-11-04 01:44-01:51 UTC, `tools/bx_box_run.py` with `--lock`, `--ls-cache`, `--ext-cache`, router seed
106218864389449120, f931 case seeds, `--runs 60`, all six loads, no PDL pin, every reference check PASS, G_RELEASE_STARTUP_OK
in all 13 prepares). One process per run, interleaved by `tools/ly_slackfree_ab.sh`: dry run of pdl8hn, then hn, c2n, d,
hn, c2n, d, then hn, c2n three more times. `tools/ly_ab_table.py` prints the tables.

Guard: in all 11 pdl8hn / combo2n prepares every T stayed on (55 PDL8H_GUARD lines, 55 pass, ids mismatches 0 on every
line, largest on drift 7.25e-3); CG_GATE passed and the unpooler pick was cg in all five combo2n prepares. pdl8d's own gate
turned T off in both of its runs: {1, 8} on (16 and 32 off) in d_a, {1, 16, 32} on (T = 8 off, final stress DRIFT) in d_b.

Box net us, mean per cell (pdl8hn n = 6, combo2n n = 5, pdl8d n = 2):

| cell | pdl8hn (range) | combo2n | combo2n - pdl8hn, mean | paired per round a..e | pdl8d | pdl8d - pdl8hn |
|---|---:|---:|---:|---|---:|---:|
| decode 1 sliding | 68.30 (67.50-68.82) | 67.33 | -0.97 | -1.66 -0.19 -1.68 -0.77 -0.02 | 67.98 | -0.31 |
| decode 1 full | 74.22 (74.00-74.32) | 72.80 | -1.42 | -1.73 -1.58 -1.22 -1.09 -1.46 | 74.29 | +0.07 |
| decode 16 sliding | 131.53 (131.18-131.90) | 131.03 | -0.49 | -0.82 +0.14 -0.58 -0.66 -0.29 | 134.50 | +2.97 |
| decode 16 full | 203.40 (202.69-204.16) | 202.53 | -0.87 | +0.02 -1.82 -0.10 -1.02 -1.86 | 206.62 | +3.21 |
| decode 32 sliding | 147.54 (146.70-148.70) | 145.91 | -1.63 | -1.86 -1.47 -0.94 -1.17 -3.12 | 150.64 | +3.10 |
| decode 32 full | 298.22 (297.54-299.09) | 297.21 | -1.01 | -1.84 -0.38 -1.95 -1.02 -0.22 | 301.46 | +3.25 |
| verify 1 sliding | 104.73 (103.92-105.20) | 103.87 | -0.86 | -1.14 -0.06 -1.23 -0.77 -0.62 | 107.31 | +2.58 |
| verify 1 full | 112.05 (111.65-112.51) | 111.96 | -0.09 | -0.99 -0.24 +0.03 +0.27 +0.06 | 116.28 | +4.23 |
| verify 16 sliding | 244.89 (243.84-245.63) | 244.38 | -0.52 | -1.65 -0.64 -0.93 -1.02 +1.71 | 244.76 | -0.13 |
| verify 16 full | 326.74 (325.92-327.36) | 327.55 | +0.81 | +0.90 -0.34 +0.51 +2.24 -0.10 | 327.89 | +1.15 |
| verify 32 sliding | 297.84 (297.34-298.56) | 297.45 | -0.39 | -0.38 -0.27 -0.51 -0.05 -0.78 | 298.08 | +0.24 |
| verify 32 full | 462.18 (461.68-463.06) | 461.55 | -0.63 | +0.75 -0.40 +0.05 -1.39 -1.94 | 462.06 | -0.12 |

Weighted six-load sum (46 x sliding + 16 x full per load, summed over six loads):

- pdl8hn: 69403.1 (dry), 69400.2, 69274.9, 69441.9, 69463.4, 69363.1; mean 69391.1, control-vs-control spread (max - min)
  188.5.
- combo2n: 69008.7, 69083.8, 69129.1, 69227.3, 69131.5; mean 69116.1, spread 218.6. Against pdl8hn: mean -275.0; per
  interleaved pair -391.5, -191.1, -312.8, -236.1, -231.6 (all five below zero).
- pdl8d: 70102.3 ({1, 8} on), 69833.7 ({1, 16, 32} on); +711.2 / +442.6 against the pdl8hn mean. Where pdl8d's gate left a T
  on it matches pdl8hn within the pdl8hn range (d_a verify 1 103.90 / 112.21, d_b decode 16 130.88 / 203.23); every larger
  difference sits on a T its gate turned off (decode 16 / 32 +6.0 to +7.1 in d_a, verify 1 +5.7 / +7.8 in d_b).
- combo2n against pdl8d: -1093.6 / -749.9 per interleaved pair (rounds a, b), -851.9 mean against mean; most of it is the gate, as above.

Reading. The slack-free pair reproduces the slack-carrying one: combo2 beat pdl8h by -182 (in-process) and -322 (separate
processes); combo2n beats pdl8hn by -275 on the mean of five pairs, with the same cell pattern (decode 1 -1.0 / -1.4,
decode 32 -1.6 / -1.0, verify 1 sliding -0.9; verify 1 full -0.1). The mean is below one control spread (188.5) but not
below two (377). The one cell above the control range is verify 16 full (+0.81 mean, +2.24 in round d), which the revival
lane also saw on combo2 (+1.42, with pdl8h's own repeat at +1.44 there). The judge check and submission are the lead's call;
nothing was registered or dispatched by this lane.

## 2026-11-03 - Decode 16 / 32 sliding: floor and gap (box, no check)

The two weakest cells of the slack-carrying twin's check a88203c2400e4d56 are decode 32 sliding (155.8 us, 1.752x) and
decode 16 sliding (130.1 us, 2.009x). The question: where do those cells spend their time above the byte floor, and what is
the largest piece that is not floor? Base: muse_decode_layer/cand/rv_k_combo2n.py (3ac83fb4, slack-free; at T = 16 / 32 it
runs the same code as the checked combo2, since the slack was a T = 1 program). Box only, no judge quota. Logs, traces and
JSON: `muse_decode_layer/profiles/box/ly/`.

Runs (okc-h200, 2026-11-04 01:58-02:10 UTC, router seed 106218864389449120, f931 case seeds, local_start per input, PDL on
at T = 1, 8, 16, 32 in every run):

- `tools/ly_tl_run.py` (bx_box_run.py with `--kernels`, plus each cell's CUDA trace written out; `--pin-pdl8 1,8,16,32
  --skip-pdl8-gate`, all four reference checks PASS) and `tools/ly_tl_parse.py`: per-kernel start, duration and the gap to
  the previous kernel over 10 cold replays (`tl_c2n.timeline.txt`). Net cell times in that run: decode 16 sliding 129.84,
  decode 32 sliding 147.74 us.
- `tools/box_route_map.py` (under flock on gpu.lock; the file's own PDL8H guard passed and kept every T on): 60 inputs per
  cell, the in-graph `moe_fused` time against the distinct local experts of each input (`rmap_c2n.log` / `.json`).
- `tools/ly_moe_tl16.py` writes `ly_tl16.py` (combo2n plus MOE-TL-01's globaltimer stamps on the MID build only, the
  same edits `tools/dl_moe_tl.py` applies to TINY2; the file is rebuilt by the tool, not committed) and
  `tools/ly_box_moe_tl16.py` replays the decode 16 MID launch of 40 inputs per cell plain and instrumented (`tl16.log` /
  `.json`; instrumented output within 1.95e-3 of plain).

### Floors

Box read bandwidth 4.435 TB/s (`sum` over 4 GiB, measured in the tl16 run). Bytes per kernel:

| kernel | bytes | floor us |
|---|---|---:|
| qkv_gate (12-bit code, 0.751 of BF16) | 2304 x 8192 x 2 x 0.751 = 28.35 MB | 6.39 |
| attention, sliding window | K + V FP8, 2048 tokens x 256 B per sequence: 8.39 MB (16) / 16.78 MB (32) | 1.89 / 3.78 |
| o (12-bit) | 8192 x 1024: 12.60 MB | 2.84 |
| b1, b2 boundaries | T x 8192 x 12 B in and out: 1.57 MB (16) / 3.15 MB (32) | 0.35 / 0.71 each |
| pooler + router (12-bit pooler, router counted at FP32 size) | 50.40 + 8.39 MB | 13.26 |
| route | ids and logits only | 0 |
| moe_fused | 28.31 MB per distinct local expert (MID, FP8 scales); 27.53 MB (B4, 6-bit SC6 scales) | 6.38 / 6.21 per expert |
| unpooler (12-bit, all 8192 rows at T > 8) | 50.40 MB | 11.36 |

At the judge's distinct-expert targets (7.5 at decode 16, 9.0 at decode 32) the MoE floor is 47.9 / 55.9 us, and the cell
floors are 84.3 us (decode 16 sliding) and 95.0 us (decode 32 sliding). Box net is 129.8 / 147.7 us, 54% / 55% above.
Neither cell is within 3% of its floor.

### Measured breakdown

The critical-path share of a kernel is its end minus the previous kernel's end in the replay timeline (with PDL a kernel
starts before its predecessor ends, so its own duration includes waiting). The MoE row uses the route map's in-graph fit at
the judge's distinct target instead of the 10 profiled inputs, whose routing differs.

| kernel | d16 critical path | d16 floor | d16 gap | d32 critical path | d32 floor | d32 gap | judge a88203c2 duration d16 / d32 |
|---|---:|---:|---:|---:|---:|---:|---|
| qkv_gate sk2 | 10.24 | 6.39 | 3.85 | 10.91 | 6.39 | 4.52 | 10.4 / 11.0 |
| attention | 7.84 | 1.89 | 5.95 | 9.59 | 3.78 | 5.81 | 7.6 / 9.3 |
| o sk2 | 3.58 | 2.84 | 0.74 | 4.59 | 2.84 | 1.75 | 6.0 / 6.4 |
| b1 (glue / Gluon row boundary) | 2.31 | 0.35 | 1.96 | 2.93 | 0.71 | 2.22 | 3.1 / 8.4 (both row calls) |
| pooler + router sk2 | 17.21 | 13.26 | 3.95 | 18.80 | 13.26 | 5.54 | 19.1 / 20.3 |
| route | 1.54 | 0 | 1.54 | 2.20 | 0 | 2.20 | 2.0 / 2.0 |
| moe_fused at 7.5 / 9.0 experts | 66.9 | 47.9 | 19.0 | 79.8 | 55.9 | 23.9 | 68.4 / 81.0 |
| unpooler sk2 | 14.63 | 11.36 | 3.27 | 15.24 | 11.36 | 3.88 | 15.0 / 15.5 |
| b2 | 3.66 | 0.35 | 3.31 | 2.76 | 0.71 | 2.05 | 6.3 / (in the b1 row) |

MoE fits over 60 inputs (in-graph kernel time, us): decode 16 sliding 24.49 + 5.66 x distinct (residual 3.60), decode 32
sliding 34.25 + 5.06 x distinct (residual 2.52). The slopes sit below the per-expert floor (6.38 / 6.21), so the intercept is
not a clean fixed cost; part of the time does not scale with the distinct count in a straight line (hypothesis: the split
jobs below add units that follow the busiest expert's token count, not the distinct count).

Inter-kernel gaps: every edge but two overlaps under PDL (gap -1.7 to -8.6 us: the next kernel starts early and waits).
The two serial edges are route -> MoE (+0.06 / +0.45 us) and MoE -> unpooler (+0.18 / +0.58 us): the MoE launch is
cooperative with PDL off ("MFIN/DF/YBF/PDL off"), and the unpooler sk2 carries bit 29 (no programmatic attribute), as qkv
does. Launch gaps therefore cost under 1 us per cell; the time is inside the kernels.

### Inside the decode 16 MoE (MID build, 40 inputs per cell)

Median input (8 distinct sliding / 7 full), us: span 74.06 / 73.30, ramp 4.72 / 4.61 (routing and ranges done at 2.82 /
2.78), phase A 40.25 / 39.94, A-to-B handoff 3.02 / 3.04, phase B 19.82 / 19.72, tail 4.42 / 4.34 (B end to barrier 0.82,
barrier idle 3.13 / 3.05, final pass 0.47; B-end spread across CTAs 4.05 / 3.79), unit retire interval 1.18 (A) / 1.15 (B),
weights wait 3.90 / 3.75 over phase A and 1.02 / 1.08 over phase B. The eager instrumented span runs about 7 us longer than
the in-graph kernel (77.66 against 70.19 at 8 distinct, cause not separated); the shares below are of the span.

The units each launch processes give the largest single piece. MID splits every expert with more than SK_T = 8 tokens into
jobs of at most 8 tokens (OKC_SPLIT; the source comment reads "a split expert re-streams its weights"). On 34 / 33 of 40
inputs (sliding / full) the launch processed 768 x (distinct + 1) units or more: one whole expert more than the distinct
count on 32 / 31 inputs, two more on 2 / 2, none on 6 / 7. At decode 16 the busiest local expert holds 10 to 12 of the 16
tokens (route map max_tok). The unit loop is issue-bound (the T = 8 L2 probe in docs/runs-moe.md retires units at the same
interval with the weights in L2), so the repeat costs its loop time whether or not its bytes come from L2: 512 / 132 phase A
units x 1.18 + 256 / 132 phase B units x 1.15 = 6.8 us per affected input.

Span excess over the floor at 8 distinct (77.66 - 51.07 = 26.6 us), largest first:

| piece | us | of which |
|---|---:|---|
| phase A above its floor (44.19 against 34.05) | 10.1 | split re-stream 4.6; unit loop 7.5% above the byte rate plus weights wait at job switches 5.5 |
| ramp | 4.7 | routing prologue 2.8, first fill 1.9 |
| tail | 4.4 | barrier idle 3.1 (B-end spread 3.9 across CTAs) |
| phase B above its floor (21.05 against 17.02) | 4.0 | split re-stream 2.2 |
| A-to-B handoff | 3.2 | |

The decode 32 MoE is the B4 build (WGS warpgroup kernel, fat jobs up to FT_T = 64 tokens, so no split at these token
counts). It has no timeline instrument, so its 23.9 us has no breakdown beyond the fit.

### Largest non-floor piece

At both cells it is the fused MoE's excess over its byte floor: 19.0 us at decode 16 sliding (15% of the cell) and 23.9 us
at decode 32 sliding (16%), three to four times the next piece (attention, 5.8 to 6.0 us, latency-bound on 8 to 17 MB of
KV). Inside it, at decode 16, the largest piece with a known cause is the split expert's repeated stream, 6.8 us per
affected input. The MoE lane's knob sweep already measured the plain alternative: MID with OKC_SKT = 16 (two n-tiles per
job, no split) is noise at decode 16 (+0.61 / -0.10 us). The source asserts `!OKC_WS || OKC_SKT == 8`, so that build also
gives up the warp-specialized ring, which is what makes MID fast. A build that keeps the warp-specialized ring and runs the
hot expert's 9 to 16 tokens as one two-n-tile job would remove the repeat without that trade (hypothesis; estimated -4 to
-6 us at decode 16 and nothing at decode 32, which runs B4).

## 2026-11-03 - ly_ws2t: decode 16 MID with two-n-tile skinny jobs (box, loss, no check)

Hypothesis (from the floor-and-gap section above): the decode 16 MID MoE re-streams and re-decodes the busiest local
expert's 768 units because OKC_SPLIT cuts every expert with more than 8 tokens into 8-token jobs, about 6.8 us per affected
input. A job of up to 16 tokens that runs both n-tiles on one decode of each unit removes the repeat. Stop rule (the lead's
bar): beat combo2n at decode 16 and decode 32 sliding by more than max(2 x control spread, 1% of the cell) and lose nowhere
beyond it. Decode 32 runs the B4 build, which this change does not touch, so this candidate could not meet the decode 32
half of the bar by construction; I built it because the split re-stream was the largest identified piece with a known
cause, and the decode 32 MoE (B4) has no instrument yet.

Also found on the way: the MoE lane's knob "midskt16" (docs/runs-moe.md, knob sweep) measured the control. helper0 has an
unconditional `#define OKC_SKT 8`, so `-DOKC_SKT=16` was redefined back to 8; a correction line is under that table.

Build: `python3 tools/dl_ly_ws2t_patch.py` writes `muse_decode_layer/cand/ly_ws2t.py`
(dd413963012bee07ecb4db1494c0a36bb80cb6ddd08a8bb6da5dfe116058338b) from rv_k_combo2n.py (3ac83fb4, SHA checked). MID
only gets `-DOKC_SKT=16`; every source edit is under `#if OKC_SKT > 8` with the original text in the `#else` arm, so
TINY / TINY2 preprocess to the original code. The edits:

- skinny_ring_ws: per job `two = cnt > 8`; tokens 8..15 staged per warp in shared memory (after the ring, the region the
  smem size formula already reserves for OKC_SKT > 8) by a loader that issues a warp's 16 loads before its stores; per
  unit, the second n-tile's MMA on the same dequantized fragments; tile-1 partial sums in a separate `red1` array, written
  before the same `fence.proxy.async`, CTA barrier and mbarrier hand-over as tile 0; the WSBLK last-unit epilogue loops
  over both tiles.
- ws_helper: reads the tile-1 partials before it frees the buffer, then runs the epilogue for tokens 8..15.
- Unit ranges split by cost (`ws_ranges`, one warp scan over the local experts; a two-tile unit costs 20, a one-tile unit
  16), used by the helper's ranges, run_phase and the FAST1 preissue, so all three still agree.

No fence, barrier, mbarrier protocol, griddepcontrol or launch attribute is changed. Correctness: G_RELEASE_STARTUP_OK
(56 verified replays) in every prepare, and the reference check passed at all 12 scored cells (50 of 50 checks in the A/B
runs plus verify 32 sliding and full in a separate run). No new PDL edge.

Iterations (decode 16 only, box, in-graph `--kernels` moe_fused or the MID timeline `tools/ly_box_moe_tl16.py` on the
instrumented copy; logs in `muse_decode_layer/profiles/box/ly/`):

| version | change | units at 8 distinct | B-end spread / barrier idle | moe_fused in graph | cell net d16 sliding |
|---|---|---:|---|---:|---:|
| combo2n | | 6912 | 4.05 / 3.13 | 70.94 | 128.64 (same session) |
| v1 | two-tile jobs, count split | 6144 | 16.9 / 14.0 | | 143.10 |
| v2 | x1 loads batched | 6144 | 15.1 / 13.3 | | |
| v3 | cost-weighted split (64-bit divide) | 6144 | 5.0 / 5.9 | | 134.54 |
| v4 | one scan for all four range ends | 6144 | 3.8 / 4.1 | 74.67 | 132.48 |
| v5 | tile 1 in `red1`; no run-time divides in the split | 6144 | 3.8 / 3.9 | 73.18 | 130.40 |

The repeat is gone (units = 768 x distinct) and, with the cost split, the CTAs finish together again. What eats the
saving: every unit retires slower, one-tile units included (median phase A interval 1.18 -> 1.25 us), and the routing
prologue ends later (route stamp 2.8 -> 4.1 us, the cost scan and the FAST1 preissue's own range computation). Register
count went from 152-159 to 164 with no spills (cuobjdump), so the slower loop is code generation in the shared unit loop
(hypothesis, not separated). The eager instrumented span is shorter than the control's (72.29 against 74.06 at the median
input), but the in-graph kernel is longer (73.18 against 70.94); I did not resolve that difference.

Paired A/B, `tools/ly_ws2t_ab.sh` (okc-h200, 2026-11-04 02:34-02:36 UTC, `tools/bx_box_run.py --lock --ls-cache
--ext-cache --pin-pdl8 1,8,16,32 --skip-pdl8-gate`, router seed 106218864389449120, f931 case seeds, `--runs 60`, one
process per run: control, candidate, control, candidate, control; every run timed with PDL on at every T). Table by
`tools/ly_ws2t_table.py muse_decode_layer/profiles/box/ly/ab`; net us:

| cell | c_a | w_a | c_b | w_b | c_c | control spread | candidate - control (means) | threshold | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.82 | 66.56 | 65.55 | 65.89 | 65.92 | 0.37 | +0.46 | 0.74 | inside |
| decode 1 full | 72.29 | 72.75 | 72.46 | 72.51 | 72.37 | 0.18 | +0.26 | 0.72 | inside |
| decode 16 sliding | 130.11 | 131.42 | 130.05 | 131.71 | 129.54 | 0.58 | +1.67 | 1.30 | loss |
| decode 16 full | 202.13 | 204.22 | 201.23 | 204.16 | 202.90 | 1.66 | +2.11 | 3.33 | inside (all four pairs +1.3 to +3.0) |
| decode 32 sliding | 147.09 | 146.43 | 145.95 | 147.09 | 146.37 | 1.14 | +0.29 | 2.27 | inside |
| decode 32 full | 297.23 | 297.28 | 296.90 | 297.34 | 297.09 | 0.34 | +0.24 | 2.97 | inside |
| verify 1 sliding | 103.36 | 104.06 | 103.50 | 103.46 | 102.85 | 0.66 | +0.52 | 1.31 | inside |
| verify 1 full | 111.94 | 111.65 | 112.08 | 111.17 | 111.87 | 0.21 | -0.55 | 1.12 | inside |
| verify 16 sliding | 244.72 | 244.70 | 244.58 | 244.66 | 243.81 | 0.91 | +0.31 | 2.44 | inside |
| verify 16 full | 327.52 | 328.75 | 327.89 | 328.26 | 327.10 | 0.78 | +1.00 | 3.28 | inside |

Decision: loss at decode 16 sliding (+1.67 us against a 1.30 threshold), slower at decode 16 full in every pair, flat at
decode 32 as expected. Not for a judge check; ruled-out row added. What it establishes: the split re-stream is real (768
units per affected input) and a two-tile WS job removes it, but as built the two-tile path makes the shared unit loop and
the prologue slower by more than the repeat costs. Reopen with a two-tile build whose one-tile loop compiles to the
control's code (the two-tile loop in a separate function body that one-tile units never enter; the host cannot pick a
build per launch, because it does not see the routing inside the graph) and a range split that costs the prologue
nothing, or measure the loop's issue slots once Nsight counters are available on some host.

## 2026-11-03 - Micro-optimizations on the box (layer lane, box only, no check)

Question: do instruction-level and memory-layout changes to the layer's hot kernels (store coalescing, store
vectorization, math precision, branch removal around guarded stores, register pressure) save measurable time on
rv_k_combo2n.py (3ac83fb437d05308c2871301a5a10c537d9b2ec504d55305403641beec2f5d27)? No routing slack, no job sizing or
range change, no fence, barrier or griddepcontrol.wait removed, no new PDL edge, every change keeps the output bits.
Box only (okc-h200, 2026-11-04 02:55-03:25 UTC), no judge quota.

Tools:

- `tools/ly_micro_sass.py FILE OUTDIR`: imports the layer file with CUDA hidden and a recording load_inline, so
  `_ma_ext` hands over the exact source and flags of the five fused-MoE builds (TINY decode 1, TINY2 verify 1, MID
  decode 16, B4 decode 32, MAIN verify 16 / 32; production suffix `-DOKC_PRMTSC=1`, `-DOKC_SC6W=1` on WGS, DQ7), compiles
  them on the box CPU with nvcc 13.0 and reports registers, local bytes and every loop that holds an HMMA / HGMMA. The
  captured sources are byte-identical to the production builds' `cuda.cu` in `/root/.cache/bx_ext` (md5 match).
  `tools/ly_micro_sass_diff.py` compares two reports.
- `tools/dl_ly_micro_patch.py --variant V [...] --out F`: one change per variant on combo2n (SHA checked, output SHA
  printed). `tools/ly_micro_ab.sh` runs control and variants as separate `bx_box_run.py --lock --ls-cache --ext-cache
  --pin-pdl8 1,8,16,32 --skip-pdl8-gate` processes, `--runs 60`, router seed 106218864389449120, f931 case seeds;
  `tools/ly_micro_table.py` makes the tables. Threshold: max(2 x control spread, 0.5 us at decode 1 / verify 1, 1% of
  the control mean elsewhere).
- Non-MoE kernels: a read-only audit (same capture method extended to the tvm_ffi glue builds, CPU compiles only;
  sources, cubins and SASS in `/root/okc/.build/ly_micro/audit/` on the box).

Logs, JSONs and tables: `muse_decode_layer/profiles/box/lym/` (`lym/` first round, `lym_r2/` five more rounds of scl and
the stack, `lym_r3/` six loads); SASS diffs in `muse_decode_layer/profiles/box/lym/sass/`.

### Four-pattern findings before any change

Patterns: (1) coalescing, adjacent lanes more than one element apart in a global access; (2) narrow back-to-back stores
to adjacent offsets; (3) 16-bit inputs through 32-bit math or transcendentals back to 16-bit; (4) boundary branch
around a short math block plus store. Line numbers are in rv_k_combo2n.py.

| kernel (cells) | pattern | line | what the source and SASS show |
|---|---|---|---|
| moe_fused helper0 WS builds (decode 1, verify 1, decode 16) | 1 | 14030 (ws_helper phase A), 14059-14060 (phase B) | lane & 3 picks the token row (8 KB apart), lane >> 2 the column: a phase A store covers 4 rows x 16 B (4 sectors, half used), phase B 4 rows x 32 B (full sectors). This is not the 32-transaction case. Both run on the helper warp, off the compute warps' unit loop |
| same | 2 | 14030 | two STG.E.U16 per unit (tokens 2t and 2t + 1, different rows); the adjacent column sits in the lane 4 away. Change wspack |
| same | 3 | 14029-14030 | gt and up are BF16-rounded (fork_bf16), then FP32 `__expf` (FMUL, FSETP, two predicated FMUL, MUFU.EX2) and `__fdividef` (MUFU.RCP, FMUL), rounded once to BF16. No `--use_fast_math`; the intrinsics are already the MUFU path. A BF16 hexp or bf16x2 SiLU changes bits, so it was not built; the ftz form of ex2 keeps them (change ftz) |
| same | 4 | 14028 | `tk < cnt` per token compiles to BSSY / BRA around the whole SiLU and store (SASS 0x116c0, 0x11870 in TINY2). Change wsmask |
| same, compute warps' unit loop | (loop) | 13920-13945 | 706 / 757 instructions per warp per unit (phase A / B, TINY2), 32 HMMA, LOP3 169, HMUL2 128, IMAD 114, SHF 82; each 64-K chunk builds its two rows' scales with two PRMT + two IMAD. Change scl. 145 / 145 / 152 registers (TINY / TINY2 / MID), 0 local |
| helper0 epi_A / epi_B (the snippet in the brief) | - | 12901-12926 | called only from fat_stream (13808-13823); no scored build runs a fat job (TINY / TINY2 jobs hold at most 8 tokens, MID splits at 8), so this epilogue is compiled but not executed (source reading) |
| moe_fused WGS B4 / MAIN (decode 32, verify 16, verify 32) | 1 | 5819 (wgs_epi_tile phase A into xB), 5836-5837 (phase B) | phase A: per instruction 4 x 16 B inside one 128 B line; phase B 4 rows x 32 B |
| same | 2, 4 | 5812-5821 | 108 STG.E.U16 and per-token branches in B4 SASS (the epilogue is inlined per n-tile count). Change wgspack |
| same | 3 | 5817-5818 | 50 MUFU.EX2 + 50 MUFU.RCP, 108 FSETP of the non-ftz exp range fix. Change ftz |
| same | (registers) | | 96 registers at 544 threads (launch bound), 0 local, B4 stack 8 B |
| sk2 / gle / narrow projections (qkv, o, pooler + router, unpooler at T > 1) | 1, 2 | 8536, 16286 | weights and x by TMA / cp.async.bulk; every output store is STG.E.64 (4 BF16 per thread), 0 STG.E.U16; 56-130 registers, 0 local |
| sk2 FUSE router logits | 1 | 8422 | 8-lane groups cover 32 B, full sectors; once per tile |
| sk2_kernel_narrow escape loop (T = 8 qkv) | 1 | 15953 | lane = token, 16 KB apart (8 sectors for 16 B); the layout requires it, off the MMA warps, once per escape |
| sk2 escape correction | 4 | 8223-8230 | `lane < M` guard around a shared read-modify-write, once per escape |
| gle FP32 router | 4 | 16441 | already predicated (36 @LDS.128); its MUFU.RCP are integer divides |
| cgemv40<128,2> unpooler (T = 1) | 1 | 17396, 17409-17410 | x as two LDG.E.128 per thread at +0 / +16 B: 512 B per instruction over 32 half-used sectors, L1-bypassing `.cg` (PDL8 edit at 18522). A register-shuffle fix spills 72 B under maxnreg 40 (compiled, not run) |
| cgemv (outside the patterns) | loads | 17402-17405 | ptxas sinks all 8 weight loads below griddepcontrol.wait (ACQBULK 0x1040, first weight LDG 0x10e0), so the 6.3 MB weight stream starts after the MoE. Change cgpf |
| glue boundary kernels (SL glue, graft glue pdl8) | 1, 2, 4 | 15491-15593, 11685-11773 | float4 / bf16 x 4 loads, STG.E.128 / .64 stores, predicated; 54-64 registers, 0 spill. Math is inline `.rn` PTX and `rsqrt.approx.ftz` matching the reference |
| attention (ATT7) | 1-4 | 590, 1915-1952, 2721 | hot loop clean; workspace RED.128 at 64 B lane stride (estimated under 0.1 us, not measured); sigmoid, RMSNorm and exp2 would change bits if demoted; masks are FSEL already; 96-128 registers, 0 spill |
| router_fp32_kernel, b2_single, plain graft glue | - | 16122, 11819, 11563 | compiled, not launched at a scored shape |

### Changes, SASS effect and box A/B

Files (from `tools/dl_ly_micro_patch.py`, SHA256 prefixes): ly_mo_scl 6341505232ec, ly_mo_ftz 8b1b356dde1b, ly_mo_wspack
e14d103308bf, ly_mo_wsmask 5a3cc4a24f53, ly_mo_wgspack 5144c07af7bb, ly_mo_cgpf 743cde948f65; stack ly_micro.py
(scl + wspack + wgspack + ftz) dc3e0d00ab5a06c42566b45b4ba48a99b061da4bfef140708f249d0b7292ef79. Every run of every arm passed
G_RELEASE_STARTUP_OK (56 verified replays) and the reference check at every cell it timed (all 12 cells for the stack
and cgpf in lym_r3).

| change | what | SASS effect | runs (ctl / var) | cells: control mean / variant mean / delta, us | decision |
|---|---|---|---|---|---|
| scl | skinny_ring_ws: both rows' scales in one bf16x2 (one PRMT + one IMAD per chunk), HMUL2 with .H0_H0 / .H1_H1 broadcast, as the WGS stage already does | unit loops 702 -> 686, 754 -> 738 (TINY), 706 -> 690, 757 -> 741 (TINY2), 708 -> 692, 763 -> 747 (MID): 8 PRMT and 8 IMAD fewer per warp per unit; registers 145 -> 141 (TINY, TINY2), 152 -> 144 (MID); B4 / MAIN unchanged | 6 / 5 (lym_r2) | d1 s 66.05 / 66.41 / +0.36; d1 f 72.20 / 72.29 / +0.09; d16 s 129.89 / 129.10 / -0.79 (5 of 5 runs -0.64 to -0.94); d16 f 202.39 / 201.54 / -0.85; d32 s -0.42; d32 f -0.54; v1 s 103.94 / 103.07 / -0.87 (5 of 5 negative); v1 f 112.17 / 111.24 / -0.93 (5 of 5 negative) | inside the threshold at every cell (d16 s threshold 1.30 by the 1% rule, v1 3.20 / 2.27 by the spread); consistent sign at decode 16 and verify 1 |
| ftz | `ex2.approx.ftz` for e^-g in every MoE SiLU; differs from `__expf` only for results below 2^-126, where 1 + e^-g is 1.0f either way | FSETP -2 (helper0 builds); B4 -108 FSETP, -324 instructions; MAIN -323; unit loops unchanged | 3 / 2 (lym) | all cells -1.18 to +0.40; d32 s per run +0.20 / -2.55 | inside |
| wspack | ws_helper phase A: shuffle pairing, one 32-bit store for (col, col + 1) per lane | 2 STG.E.U16 -> 1 STG.E + 1 SHFL.BFLY, one BSSY fewer (helper0 builds) | 3 / 2 | -0.91 (v1 s) to +0.14 (d1 f) | inside |
| wsmask | ws_helper phase A: SiLU unconditional, only the store guarded | BSSY -2, 6-8 instructions fewer | 3 / 2 | -0.26 to +0.33 | inside |
| wgspack | wgs_epi_tile phase A: the same pairing into xB | B4: STG.E.U16 108 -> 0, +54 STG.E, +54 SHFL, BSSY -82, 26796 -> 25850 instructions; stage loops trade 1-3 IMAD for SHF; registers 96 unchanged | 3 / 2 | d32 s -0.55, d32 f -0.51, others -0.24 to +0.39 | inside |
| cgpf | T = 1 unpooler cgemv: thread 0 prefetches the CTA's weight bytes into L2 (cp.async.bulk.prefetch.L2) before griddepcontrol.wait | UBLKPF.L2 x 2 before the compute warps' ACQBULK in every cgemv instance; 40 registers, stack 0 (audit compile of the same text) | 4 / 3 (lym_r3) | d1 s 65.87 / 66.26 / +0.40; d1 f 72.14 / 72.39 / +0.25; others -0.68 to +0.56 | inside; no saving at decode 1 (the hypothesis was 0.3 to 1.4 us) |

### Stacked file ly_micro.py, six loads (lym_r3: control 4 runs, stack 3 runs, net us)

| cell | control mean | control spread | ly_micro mean | delta | per run | threshold | verdict |
|---|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 65.87 | 0.59 | 66.42 | +0.55 | +0.46 +1.02 +0.18 | 1.18 | inside |
| decode 1 full | 72.14 | 0.56 | 72.12 | -0.02 | +0.53 -0.24 -0.36 | 1.12 | inside |
| decode 16 sliding | 129.93 | 0.48 | 128.81 | -1.12 | -1.48 -0.70 -1.18 | 1.30 | inside |
| decode 16 full | 202.30 | 1.60 | 201.92 | -0.38 | -0.67 -0.45 -0.03 | 3.20 | inside |
| decode 32 sliding | 145.86 | 2.13 | 145.28 | -0.58 | -0.34 -0.56 -0.85 | 4.26 | inside |
| decode 32 full | 297.28 | 2.72 | 296.42 | -0.86 | -0.78 -0.64 -1.16 | 5.44 | inside |
| verify 1 sliding | 103.88 | 1.62 | 103.24 | -0.64 | -1.10 -0.12 -0.70 | 3.23 | inside |
| verify 1 full | 111.97 | 0.98 | 111.89 | -0.08 | -0.24 -0.05 +0.04 | 1.95 | inside |
| verify 16 sliding | 244.12 | 2.30 | 244.01 | -0.11 | +0.14 +0.44 -0.92 | 4.61 | inside |
| verify 16 full | 327.24 | 1.89 | 327.89 | +0.65 | +1.03 -0.08 +1.00 | 3.78 | inside |
| verify 32 sliding | 297.92 | 0.98 | 296.37 | -1.55 | -1.31 -1.68 -1.66 | 2.98 | inside |
| verify 32 full | 462.45 | 1.02 | 460.99 | -1.46 | -2.29 -1.17 -0.92 | 4.62 | inside |

The five-round run of the same stack at four loads (lym_r2, 6 control / 5 stack runs) agrees in sign: decode 16 sliding
-1.12 (5 of 5 runs negative), decode 16 full -1.04 (5 of 5), decode 32 sliding -1.32 (5 of 5, threshold 2.14), decode 32
full -1.31 (5 of 5), verify 1 -0.43 / -0.25, decode 1 sliding +0.55 (5 of 5 positive, +0.40 to +0.67).

### Reading

- No single change, and not the stack, clears the threshold at any cell, so by the stated rule there is no winner. The
  stack's sum of mean deltas over the 12 cells is -5.6 us (lym_r3). Across its eight runs in two sessions it was faster
  at decode 16 sliding, decode 32 sliding and decode 32 full in every run, and faster at verify 32 in every lym_r3 run.
  These are box measurements; whether they carry to the judge is untested.
- decode 1 sliding is slower with the stack in all 8 stack runs (+0.18 to +1.02, mean +0.55) and with scl alone in 3
  of 5 lym_r2 runs (mean +0.36), inside the 0.5 us floor and 2 x spread. Hypothesis, not separated: scl's register
  allocation change in the TINY build (145 -> 141) moves the decode 1 MoE (2 to 3 experts, ramp-dominated) the wrong
  way. A stack without scl on TINY would test it.
- The time is in the unit loop: scl removes 2.3% of the WS unit loop's instructions (16 of 706) and decode 16 moved
  -0.8 us of a 67 us MoE (1.2%), so the loop is partly issue-bound, as the T = 8 L2 probe found (docs/runs-moe.md). The
  helper-warp epilogue changes (wspack, wsmask) moved nothing measurable; the WGS epilogue changes (wgspack, ftz)
  remove about 1,000 instructions per B4 image, mostly outside the stage loop, and gave -0.5 us at decode 32 in two runs.
- The brief's epilogue snippet (epi_A with `p.inter + (pstart + tk) * ID + col`) is not executed at any scored shape.
- Fast math: the MoE already uses `__expf` / `__fdividef` (MUFU.EX2 / MUFU.RCP); the non-ftz range fix was the only
  removable part. BF16 or bf16x2 SiLU math would change bits against combo2n and was not built.
- Registers and spills: no MoE build spills (local 0 in all five); B4 has an 8-byte stack frame. No register-pressure
  change was needed.
- Judge check candidate, if the lead wants one: ly_micro.py (dc3e0d00), or ly_micro without scl on TINY if the decode 1
  sign matters.

### ly_micro2: the stack without scl on the decode 1 build (box, 2026-11-04 03:31-03:34 UTC)

`python3 tools/dl_ly_micro_patch.py --variant scl_notiny --variant wspack --variant wgspack --variant ftz --out
muse_decode_layer/cand/ly_micro2.py` gives ly_micro2.py, SHA256
0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd. `scl_notiny` wraps the scl edits in `#if !OKC_NOSCL`
and adds `-DOKC_NOSCL=1` to MOE_TINY_DEFINES only. SASS (`muse_decode_layer/profiles/box/lym/sass/stack2.diff.txt`): the
TINY unit loops are back to the base's 702 / 754 instructions and 145 registers; the TINY2, MID, B4 and MAIN SASS is
identical to ly_micro's (instruction text md5 equal for all four).

A/B (`lym_r4/`, `tools/ly_micro_ab.sh`, same flags and pin, `--runs 60`, order c, ly_micro, ly_micro2 three times, then
c; all 10 runs passed the startup expert comparison and every reference check). Net us:

| cell | control mean (4) | control spread | ly_micro delta (per run) | ly_micro2 delta (per run) | threshold |
|---|---:|---:|---|---|---:|
| decode 1 sliding | 66.19 | 0.70 | +0.35 (+0.61 +0.36 +0.07) | +0.16 (+0.60 -0.04 -0.06) | 1.41 |
| decode 1 full | 72.47 | 0.22 | +0.12 (-0.34 +0.83 -0.12) | -0.29 (-0.45 -0.12 -0.31) | 0.50 |
| verify 1 sliding | 102.61 | 0.86 | -0.05 (+0.20 -0.70 +0.34) | +0.30 (+0.34 +0.23 +0.32) | 1.73 |
| verify 1 full | 112.38 | 0.98 | -0.24 (-0.07 -0.55 -0.10) | -0.19 (+0.09 -0.50 -0.17) | 1.95 |

Reading: ly_micro2's decode 1 sliding delta (+0.16) is inside the control spread (0.70), and decode 1 full moved to
-0.29 in all three runs. Verify 1 runs the TINY2 build, whose SASS is the same in both files, so the +0.30 against
-0.05 at verify 1 sliding is session variation between arms of identical verify 1 code, inside the 0.86 spread. In
this session ly_micro's own decode 1 sliding excess was +0.35 (earlier sessions +0.55), and neither file reproduced the
earlier verify 1 sliding gain (-0.64 for ly_micro in lym_r3). All cells are inside the threshold.
## 2026-11-03 - ly_ws2u: ly_ws2t's 16-token MID jobs with the control's one-tile loop (box, loss, no check)

The reopen test for the ly_ws2t row: keep the 16-token MID jobs that remove the decode 16 split re-stream, but (1) give
the two-tile path its own function body so the one-tile unit loop is the control's code, and (2) take the cost split out
of the routing prologue. Base: rv_k_combo2n.py (3ac83fb4, SHA checked by the tool). Box only, no judge check, nothing
registered. Box runs from 02:56 to 03:56 UTC (18:56 to 19:56 PST), router seed 106218864389449120, f931 case seeds,
local_start per input, PDL on at T = 1, 8, 16, 32 in every run. Logs and JSON: `muse_decode_layer/profiles/box/ly2u/`.

Build: `python3 tools/dl_ly_ws2u_patch.py` (default variant b; `--variant a`; `--tl` applies the same edits to
ly_tl16.py from tools/ly_moe_tl16.py). MID only gets `-DOKC_SKT=16`; every edit is under `#if OKC_SKT > 8`, so TINY and
TINY2 keep the original text.

- skinny_ring_ws and ws_helper are the control's text. ly_ws2t's two-tile versions sit beside them as skinny_ring_ws2 and
  ws_helper2. A CTA whose unit range holds no job of more than 8 tokens runs the control's functions; the choice is
  CTA-uniform, read from S.jC after the WS_JOIN that publishes the job table. ws_helper's leading WS_JOIN moves into the
  selecting wrapper at the same program point. red1 sits at the end of Smem (alignas 16), so every other member keeps
  its offset.
- Variant a: ly_ws2t's cost split as built (ws_ranges on warp 0 after route(), again in the FAST1 warp).
- Variant b: a closed-form cost split computed only in the FAST1 warp, which already holds the per-expert counts and runs
  beside warp 0's route(). It needs one job per local expert, which MID guarantees at M <= 16 with 16-token jobs, so it
  is ballots and popcounts with no scan and no double divide. The phase A ranges are computed before the preissue, the
  phase B ranges after it, and lane 0 writes S.wsr before the WS_JOIN; warp 0 computes nothing after route().
- The two-tile x rows (tokens 8 to 15) load by cp.async (zero-filled past the job's tokens) and are waited for after the
  unit's ring-slot wait (`cp.async.wait_all`, `__syncwarp`), in place of ly_ws2t's blocking loader.

No fence, barrier, mbarrier protocol, griddepcontrol or launch attribute is removed or changed. No new PDL edge.

### Step 1: the two costs, separated

SASS (`tools/ly_ws2u_sass.py` compiles the MID build with nvcc only, `tools/ly_ws2u_sass_cmp.py` compares). Control:
152 registers, the one-tile unit loop 534 instructions with 32 HMMA. ly_ws2u.py (c77d594f, the A/B file): 151 registers,
no stack, no local memory; its one-tile loop is 530 instructions (BSSY and BSYNC -2 each, IADD3 -3, LOP3 -3, IMAD.SHL
-2, IMAD.IADD +2, IMAD +2, LDS +4). The loop is neither byte-identical to the control's nor identical up to register
names, so "compiles to the control's code" holds for the instruction count and the per-unit retire interval (below), not
for the bytes. A `__noinline__` two-tile path (`-DOKC_WS2INL=0`) gives 160 registers, a 272-byte stack and 240 local
loads and stores, so the inlined form is the one measured.

Eager MID timeline (`tools/ly_box_moe_tl16.py`, 40 inputs per cell; medians sliding / full, us):

| build | route stamp | ramp | unit A | unit B | units (median) | B-end spread | span |
|---|---:|---:|---:|---:|---:|---:|---:|
| control ly_tl16.py (2d3b0dd6) | 2.83 / 2.78 | 4.72 / 4.62 | 1.18 / 1.18 | 1.15 / 1.15 | 6144 / 6144 | 4.13 / 4.40 | 72.22 / 72.13 |
| variant a (ly_ws2ua_tl.py f6fdd3e0, blocking x1 loader) | 4.15 / 4.15 | 5.46 / 5.45 | 1.18 / 1.18 | 1.18 / 1.20 | 6144 / 5376 | 9.79 / 9.70 | 75.28 / 73.41 |
| variant b, all four ranges before the preissue (4b29a74d) | 3.60 / 3.58 | 5.57 / 5.57 | 1.18 / 1.18 | 1.17 / 1.15 | 6144 / 5376 | 10.53 / 10.70 | 76.74 / 75.33 |
| variant b, phase B ranges after the preissue, weights 5:4 | 2.99 / 2.97 | 4.83 / 4.80 | 1.18 / 1.18 | 1.18 / 1.18 | 6144 / 5376 | 10.91 / 10.45 | 75.42 / 74.40 |
| variant b, cp.async x1, weights 4:3 (ly_ws2ub_tl.py 3d4733f9: the A/B build plus stamps) | 3.01 / 2.98 | 4.90 / 4.77 | 1.18 / 1.18 | 1.18 / 1.18 | 6144 / 5376 | 11.54 / 7.79 | 71.73 / 70.62 |

At 8 distinct experts the control processes 6912 units and every variant 6144: the re-stream is gone. The separate body
brings the unit interval back to the control's (ly_ws2t: 1.25), and the FAST1-warp split brings the route stamp to within
0.2 us of the control's (ly_ws2t: +1.3). Variant a's prologue matches ly_ws2t's. The f6fdd3e0 and 4b29a74d bytes came from
uncommitted tool states between 199f6418 and a30ff1d7 and were overwritten later; their logs are kept.

The probe files named below (ly_ws2ub_tl_pr3.py, ly_ws2ub_tl_all.py, ly_ws2ub_tl_all2.py, ly_c2n_smpad.py) are ad hoc
copies of the timeline build with a few stamp or predicate edits each; they are not committed, their logs are.

What is left is the finish: the B-end spread is 8 to 12 us against 4.1 to 4.4. Weight sweeps of the split did not move it
(both phases 5:4, 4:3, 7:5, 3:2, 5:3: spread 9.2 to 10.9 sliding; phase B alone 7:4, 2:1, 5:2: 8.5, 9.4, 12.2). Per CTA
(`--raw`, eager), the 20 to 29 CTAs whose ranges hold the two-tile job end phase A with the others (40.5 against 39.7 us),
but their first unit of each phase is late:

- Phase A: first ring slot plus x1 at 7.2 us against 4.3 to 4.5 for the other CTAs. With ly_ws2t's blocking loader the x1
  load itself ran from 3.81 to 7.49 us; with cp.async the x1 wait is short, but the first unit still completes at the same
  time.
- Phase B (`ly_ws2ub_tl_pr3.py` probe): the job's doneA wait ends at 45.1 us, x0 arrives at 46.6 to 46.9, x1 at 46.6, the
  ring slot is full at 46.5, and the first two-tile B unit's MMA ends at 52.8, at the same time on every one of those CTAs.
  The unit body takes about 6 us the first time and 1.47 after.
- Running every CTA through ws2 (`ly_ws2ub_tl_all.py`, one-tile CTAs on ws2's one-tile branch) leaves the stall on the
  two-tile CTAs (first B unit 7.2 / 9.1 / 3.8 us after the dependency on three inputs, against 2.0 to 2.4 for the rest)
  and slows the one-tile units to 1.22 to 1.25. Running every job through the two-tile body (`ly_ws2ub_tl_all2.py`, x1
  zero-filled for jobs of 8 or fewer tokens) removes it: first B unit 2.2 to 2.4 us after the dependency (max 3.7 to 4.0)
  on every CTA.

So the stall follows the code, not the data: it appears when a code region runs on 20 to 29 SMs and disappears when all
132 run it. Hypothesis: cold instruction fetch for a code region that few SMs execute (the cold-L2 flush evicts code as
well). Not confirmed with counters (Nsight counters are blocked on this pod).

### In-graph

The eager span understates the loss. `tools/ly_box_moe_tl16g.py` takes the same stamps inside the layer's CUDA graph
(fresh input per replay, cold L2; 20 inputs per cell), sliding / full:

| build | graph us | span | ramp | A | handoff | B | B-end spread | barrier idle | final | units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control ly_tl16.py | 137.73 / 213.39 | 72.13 / 72.98 | 3.68 / 3.69 | 41.34 / 41.58 | 2.34 / 2.40 | 20.57 / 20.63 | 3.07 / 3.81 | 2.55 / 2.90 | 0.46 / 0.49 | 6912 / 6912 |
| ly_ws2ub_tl.py | 145.17 / 218.66 | 79.34 / 79.36 | 4.06 / 4.01 | 37.45 / 37.42 | 3.33 / 3.31 | 21.78 / 21.76 | 10.77 / 9.71 | 9.83 / 8.37 | 1.46 / 1.38 | 6144 / 6144 |

Phase A gains 3.9 us (the re-stream). The finish loses that and more: B-end spread +7.7 / +5.9, handoff +1.0, final pass
+1.0. In-graph per-kernel (`--kernels`, 10 cold replays) moe_fused: control 71.29 / 70.09, ly_ws2u.py 79.45 / 75.97. A
control with ly_ws2u's 16 KB of red1 added to Smem and left unused (`ly_c2n_smpad.py`) gives 70.72 / 69.82, so the shared
memory size is not the cause.

### Step 2: paired A/B against combo2n

`tools/ly_ws2u_ab.sh` (`tools/bx_box_run.py --lock --ls-cache --ext-cache --pin-pdl8 1,8,16,32 --skip-pdl8-gate`,
`--runs 60`, five processes: control, candidate, control, candidate, control, 03:28:14 to 03:29:14 UTC). Before it, the
candidate ran once at all six loads (`chk`, `--runs 5`): reference check PASS at all 12 scored cells.
G_RELEASE_STARTUP_OK (56 verified replays) in every prepare; every run timed with PDL on at T = 1, 8, 16, 32. Candidate
ly_ws2u.py c77d594f1c8713a520a4b03271ece2e206d7327d263eb1f31fb8d73f2983d946 (the tool at a7cfdce4; the current tool's
default output, 3b22dd4a, adds the phase B weight defines and was not measured). Table by
`tools/ly_ws2t_table.py muse_decode_layer/profiles/box/ly2u/ab`; net us:

| cell | c_a | w_a | c_b | w_b | c_c | control spread | candidate - control (means) | threshold | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.81 | 66.18 | 65.73 | 66.06 | 65.50 | 0.30 | +0.44 | 0.66 | inside |
| decode 1 full | 72.14 | 72.18 | 72.11 | 72.26 | 72.54 | 0.43 | -0.05 | 0.86 | inside |
| decode 16 sliding | 129.42 | 134.00 | 129.62 | 133.38 | 129.63 | 0.21 | +4.13 | 1.30 | loss |
| decode 16 full | 201.65 | 206.08 | 201.55 | 206.90 | 202.10 | 0.54 | +4.72 | 2.02 | loss |
| decode 32 sliding | 146.99 | 147.15 | 146.43 | 146.08 | 146.18 | 0.82 | +0.08 | 1.63 | inside |
| decode 32 full | 297.26 | 296.62 | 297.06 | 297.95 | 296.61 | 0.66 | +0.31 | 2.97 | inside |
| verify 1 sliding | 102.72 | 103.15 | 103.62 | 103.33 | 103.31 | 0.90 | +0.02 | 1.79 | inside |
| verify 1 full | 111.82 | 112.10 | 111.54 | 111.82 | 111.86 | 0.32 | +0.22 | 1.12 | inside |

A second build with phase B weights 7:4 (`--spb 4:7 --out ly_ws2u_b47.py`, ly_ws2u_b47_b47.py c1555731, decode 16 only,
03:33 to 03:35 UTC, same protocol; `muse_decode_layer/profiles/box/ly2u/b47`): decode 16 sliding +3.80 (threshold 1.30,
every pair +3.6 to +4.2), full +1.87 (threshold 2.18).

Decision: loss at decode 16, sliding and full, larger than ly_ws2t's (+1.67 / +2.11). Not for a judge check; the
ruled-out row is extended. Step 3 (the same job-size question for the B4 build at decode 32) was not run because decode
16 did not clear; B4 builds fat jobs of up to FT_T = 64 tokens, so it has no 8-token split at decode 32 by construction
(source reading, not measured).

What this establishes: with the one-tile loop at the control's instruction count and the split out of the prologue, the
two costs ly_ws2t paid are gone, and the re-stream saving shows up in phase A (-3.9 us in graph). The loss comes from the
CTAs that run the two-tile code: their first unit in each phase is late (about 2.8 us in A and 5 to 6 us in B), which a
per-unit cost split cannot balance because the delay is per CTA. That delay disappears when every SM runs the two-tile
body. A next design, if any: one shared unit body for one-tile and two-tile units (so every SM fetches the same code)
whose one-tile issue rate stays at 1.18 us, which ly_ws2t's shared body did not reach (1.25).

## 2026-11-03 - Layer attention against the standalone board's kernel at verify 1 (attention lane, box only, no check)

Question: is the layer's attention (ATT7, `attn_kernel` in muse_decode_layer/cand/ly_micro2.py, SHA256 0bc2bd8b) slower
than the standalone board's public kernel (`.build/repeats/attention_base.py`, aa3535d0, the 8914751e3b9f468b bytes,
attn_one with HMRG 1, PRQ 31, FIN8 1, pick off) at the verify 1 shapes (batch 1, q_len 8, ctx 80000, page 1), and by
more than 1 us? Answer: no. The standalone kernel is faster by 0.66 us at sliding and by 0.02 to 0.13 us at full, so
nothing was ported.

Method: `tools/att_v1_cmp.py` (box, 03:58-04:02 UTC, under gpu.lock with the ext cache) imports both files in one
process, feeds both the same fresh attention-board inputs (60 per cell, one per run, seeds 880001 + 100000 i + r),
captures one CUDA graph per arm, copies each input into every arm's buffers, flushes L2 with a 1 GiB read, and replays
the arms in rotated order. Net = raw minus the fixed replay cost (3.90 us). Arms: `std` (the standalone file's
kernel_fn), `lay` (the layer's `_launch7` with PDL off and no prefetch), `lay_pf` (as `lay` plus the 4 MB pooler L2
prefetch the layer issues at T = 8, read from a dummy buffer). Every arm passed the reference check on a held-out
input and on the first timed input in processes 2 to 5; process 1's replay check read the wrong cache (harness bug,
fixed before process 2), and its times agree with the others. Full runs on both the layer's input distribution (soft)
and the attention board's (coherent).

Plans at these shapes are the same in both files: sliding 33 splits x 1, cooperative, one tile per CTA, one workspace
copy (standalone mode 15 = cooperative | one tile | speculated length); full 252 splits x 1, cooperative, 8 workspace
copies (standalone mode 11). ATT7's `_TUNED7` has no entry for (1, 8).

Median of the five process medians (net us; spread = max - min across the five process medians or paired deltas):

| cell | std | lay | lay_pf | lay - std (paired) | lay_pf - std (paired) |
|---|---:|---:|---:|---:|---:|
| verify 1 sliding (soft) | 6.88 (0.11) | 7.52 (0.10) | 7.78 (0.06) | +0.66 (0.03) | +0.90 (0.11) |
| verify 1 full (soft) | 14.98 (0.10) | 15.09 (0.13) | 15.39 (0.19) | +0.13 (0.08) | +0.38 (0.08) |
| verify 1 full (coherent) | 15.12 (0.11) | 15.15 (0.06) | 15.39 (0.06) | +0.02 (0.16) | +0.30 (0.19) |

Per process, lay - std at sliding: +0.656, +0.656, +0.640, +0.672, +0.672; at full soft: +0.080, +0.160, +0.080,
+0.128, +0.144.

In the layer (bx_box_run.py --kernels, ly_micro2, PDL pinned 1,8,16,32, 60 runs, f931 seeds): `attn_kernel<8, false,
15>` 9.04 us at verify 1 sliding and `attn_kernel<8, false, 11>` 16.53 us at verify 1 full (10-replay profiler means;
the attention is PDL-launched and starts during the qkv projection, so its lifetime includes the wait and is not a
kernel-alone time). Cell nets in that run: 102.48 / 110.67 us.

Reading: the kernel-alone gap the brief asked about is 0.66 us at sliding and 0.13 us or less at full, under the 1 us
port threshold at both cells. The layer's own pooler prefetch adds 0.24 us (sliding) and 0.25 to 0.28 us (full) to the
attention; it buys the pooler's weight stream, so it is a layer trade, not an attention-port question. The 16.5 us
o-projection reading at verify 1 full is not explained by a slower attention kernel: kernel-alone, ATT7 and attn_one
are within 0.13 us there. A sliding port could at best save 0.66 us of attention time at one cell (0.66 x 46 weight),
less any part already hidden under PDL. Logs and JSONs: `muse_decode_layer/profiles/box/attv1/`.

## 2026-11-03 - Compute-store fusion audit (layer lane, box timeline only, no check)

Question (docs/micro-optimization-rule.md, pattern 0): at which kernel boundaries of muse_decode_layer/cand/ly_micro2.py
(SHA256 0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd) can the consumer's math move into the
producer's epilogue (or the producer's output be recomputed by the consumer), and what would that save per cell? Audit
only; nothing was built.

Measurements (okc-h200, 2026-11-03 20:13 PST): `LY_TRACE_DIR=.build/box/fu/tl_m2 tools/ly_tl_run.py` on ly_micro2,
all 12 cells, `--runs 20 --kernels --lock --ls-cache --ext-cache --pin-pdl8 1,8,16,32 --skip-pdl8-gate`, router seed
106218864389449120, f931 case seeds, every reference check PASS. Net cell times in that run: decode 1 65.74 / 72.59,
decode 16 130.14 / 205.65, decode 32 144.77 / 297.42, verify 1 104.93 / 116.64, verify 16 241.86 / 324.48, verify 32
297.30 / 462.26 us (sliding / full). Per-kernel timelines over 10 cold replays: `muse_decode_layer/profiles/box/fu/`
(`tl_m2.timeline.txt`, `tl_m2.log`, `tl_m2.json`). Cluster occupancy: `tools/fu_occ.cu` (cudaOccupancyMaxActiveClusters
for a 416-thread, one-CTA-per-SM kernel, run under gpu.lock, 20:27 PST).

### Chains per T (kernel names from the timeline)

| T (cells) | qkv | attention | o | boundary 1 | pooler + router | route | MoE | unpooler | boundary 2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 (decode 1) | sk2 | attn_kernel | sk2 o8h | glue_cluster_boundary (8-CTA cluster, router copy) | sk2_kernel_gle (BF16 pooler + FP32 router, one launch) | _sl_tile_route_cg | moe_fused TINY | cgemv40 slice (1024 rows), PDL secondary | _sl_cluster_finish_cg |
| 8 (verify 1) | sk2_kernel_narrow | attn_kernel | sk2 o8h | glue_cluster_boundary (preloaded base) | sk2_kernel_gle | _sl_tile_route_cg | moe_fused TINY2 | sk2 slice (bit 29: no PDL attribute) | _sl_cluster_finish_cg |
| 16 (decode 16) | sk2 | attn_kernel | sk2 o8h | glue_cluster_boundary | sk2 FUSE (BF16 hi / lo router) | _sl_tile_route_cg | moe_fused MID | sk2 unpooler (8192 rows) | glue_cluster_boundary POST |
| 32 (decode 32) | sk2 | attn_kernel | sk2 o32h | _graft_row_boundary_pre_cg (Gluon, one program per row) | sk2 FUSE | _sl_tile_route_cg | moe_fused B4 | sk2 unpooler | _graft_row_boundary_pre_cg |
| 128 / 256 (verify 16 / 32) | nvjet | attn_kernel | nvjet | _graft_row_boundary | nvjet (pooler + router) | _graft_route | moe_fused MAIN | nvjet | _graft_row_boundary |

### Critical-path share per kernel (end minus the previous kernel's end, us)

| cell | qkv | attn | o | b1 | pool + rt | route | MoE | unpool | b2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 9.94 | 5.29 | 3.59 | 2.57 | 15.68 | 1.27 | 23.10 | 3.73 | 1.61 |
| decode 1 full | 9.94 | 11.32 | 3.64 | 2.51 | 15.66 | 1.30 | 24.14 | 3.36 | 1.62 |
| decode 16 sliding | 10.40 | 7.63 | 3.89 | 2.21 | 17.10 | 1.68 | 69.19 | 14.25 | 3.81 |
| decode 16 full | 10.59 | 82.64 | 4.16 | 2.13 | 17.02 | 1.56 | 69.76 | 15.26 | 3.78 |
| decode 32 sliding | 10.94 | 9.75 | 4.70 | 2.88 | 18.53 | 2.10 | 77.13 | 15.23 | 2.85 |
| decode 32 full | 10.75 | 159.91 | 4.75 | 2.72 | 18.69 | 2.13 | 80.24 | 15.05 | 3.06 |
| verify 1 sliding | 9.33 | 6.83 | 3.74 | 2.61 | 17.11 | 1.64 | 58.37 | 5.28 | 2.10 |
| verify 1 full | 9.66 | 14.00 | 3.92 | 2.69 | 16.98 | 1.61 | 64.60 | 5.00 | 2.07 |
| verify 16 sliding | 16.19 | 12.07 | 8.30 | 4.86 | 25.54 | 2.83 | 141.09 | 21.81 | 5.42 |
| verify 16 full | 16.22 | 98.24 | 7.92 | 4.80 | 26.00 | 2.98 | 139.22 | 22.73 | 5.99 |
| verify 32 sliding | 18.19 | 15.39 | 9.67 | 7.01 | 31.76 | 3.60 | 177.43 | 26.74 | 7.23 |
| verify 32 full | 18.40 | 177.98 | 9.48 | 6.96 | 32.09 | 3.67 | 179.40 | 26.77 | 7.43 |

Edges at T <= 32 overlap under PDL (start gaps -1.1 to -12.9 us) except route -> MoE (+0.06 to +0.42, cooperative MoE,
PDL off) and MoE -> unpooler at T >= 8 (+0.13 to +0.80). At T = 128 / 256 every edge is serial (+0.22 to +0.96 us; nvjet
has no PDL), about 3.5 to 4.5 us of gaps per cell.

Cell weights, from d(ln load time) = w x dt with load time = 46 x sliding + 16 x full (the cells measured above),
relative to decode 1 sliding (0.0110 per us): decode 1 1.00 / 0.35, verify 1 0.63 / 0.22, decode 16 0.45 / 0.16,
decode 32 0.37 / 0.13, verify 16 0.26 / 0.09, verify 32 0.20 / 0.07 (sliding / full). One us at decode 1 sliding is worth
five at verify 32 sliding.

### Boundary table

Bytes at T = 1 / 8 / 16 / 32. Every intermediate here is small (a [T, 8192] FP32 row is 32 KB per token): the consumer's
own byte floor is 0.01 to 0.7 us at T <= 32, so what a fusion can recover is launch and dependency latency, not
bandwidth.

| # | boundary | producer stores, consumer reads back | consumer floor / critical path (us) | reduction the consumer needs | tried before and what it showed | expected saving, us per cell (hypothesis) |
|---|---|---|---|---|---|---|
| 1 | qkv -> attention | qkv BF16 [T, 2304] (q 1024, k 128, v 128, gate 1024): 4.6 / 37 / 74 / 147 KB | ~0 / attention is a large kernel either way (5.3 to 9.8 us at sliding) | per head, not row-wide: q / k RMSNorm over each 64-column head (reference line 364) and RoPE; the new k / v go to the paged cache. The qkv sk2 CTA owns 24 rows, so a 64-column head spans three CTAs | no | under 0.3: no launch is removed (attention stays); only the per-head norm / RoPE prologue could move, and only after re-tiling qkv into head-aligned CTAs |
| 2 | attention -> o | attention output BF16 [T, 1024]: 2 / 16 / 32 / 64 KB; the gate rows come from qkv | ~0 / 3.6 to 4.8 | element-wise: o's input is attn x sigmoid(gate). Already fused: `dense_ng` applies the gate in o's x path | the only element-wise boundary in the chain, already done | 0 |
| 3 | o -> boundary 1 | P BF16 [T, 8192] 16 KB / token (16 / 128 / 256 / 512 KB); b1 also reads the stream FP32 32 KB / token and writes s1 FP32 32 KB, h BF16 16 KB, the FP32 router copy 32 KB (T <= 8) and tail | 0.05 / 0.25 / 0.35 / 0.71 floor; 2.57, 2.51 (d1), 2.61, 2.69 (v1), 2.21, 2.13 (d16), 2.88, 2.72 (d32) | two dependent row-wide reductions over all 8192 columns: the sum of bf16(P)^2 for the post norm, then the sum of ns^2 of the new stream ns = alpha s + beta bf16(P r1) for the pre-FFN norm. o's 8192 output columns are spread over 256 CTAs, so either reduction needs a cross-SM exchange | yes, three forms. M-C C-1 (o + boundary, cooperative, two grid barriers; check 5ae9ce318e014edf): 1.16 us slower in 9 of 9 rounds. F3 (o + boundary + pooler, completion counters; c9240b8205904baf): 0.45 to 1.28 us in process, no short load better at the judge. GLE (deferred-norm algebra in o's epilogue, `_gle_after_attention`, still in the file but gated off): ineligible, it moves the BF16 rounding of bf16(P r1) (ruled-out.md, 2026-11-01 G boundary row) | at most 0.5 producer-side: each exchange through global memory or a grid barrier costs about what a kernel boundary costs (~0.7 us, ruled-out.md PDL row). Today b1 runs both reductions on DSMEM inside one 8-CTA cluster, so a fused form has more global round trips, not fewer |
| 4 | boundary 1 -> pooler + router | h BF16 16 KB / token and (T <= 8) the FP32 router copy; every pooler CTA reads the whole h row (it is the K operand) | pooler floor 13.26 / 15.66 to 18.69 after b1 | the same two reductions, seen from the consumer: recompute b1 in each pooler CTA | consumer-side recompute: not tried. Ruled out here by arithmetic: each pooler CTA would read P, stream, alpha, beta and pre for all 8192 columns, 128 KB per CTA, 16 MB of L2 reads over 128 CTAs at T = 1 (about 3 us at an L2 rate near 5.5 TB/s, assumed, not measured), against b1's 2.57 us; the pooler (about 206 KB of shared memory) cannot share an SM with o's CTAs, so none of that work can start before o ends. Cluster form (F3's open condition: b1 run per 8-CTA pooler cluster on DSMEM): measured infeasible, `tools/fu_occ.cu` gives at most 15 active clusters of 8 (120 CTAs) and 30 of 4 (120 CTAs) at one CTA per SM, for 128 pooler CTAs | none at T = 1; worse at T = 8 (8x the per-CTA reads) |
| 5 | pooler + router -> route | pooled BF16 [T, 4096] 8 KB / token, logits FP32 [T, 256] 1 KB / token (9 / 72 / 144 / 288 KB) | ~0 / 1.27, 1.30 (d1), 1.64, 1.61 (v1), 1.68, 1.56 (d16), 2.10, 2.13 (d32) | row-wide: the pre-expert RMSNorm over all 4096 pooled values (128 pooler CTAs own 32 rows each) and top-8 over all 256 logits (32 router tiles, each finished by its own last-arriving quarter) | the pooler + router fusion exists (`sk2_kernel_gle` at T <= 8, sk2 FUSE at 16 / 32). Route in a last-arriver tail of the pooler: not tried; F3's counters recovered about 0.3 us per removed launch | 0.2 to 0.4 at T = 1 / 8 (one counter round trip replaces one kernel boundary); negative at 16 / 32, where one CTA would read 144 to 288 KB that route now spreads over 2T programs |
| 6 | route -> MoE | xm BF16 [T, 4096] 8 KB / token, ids / lids / weights | ~0 / route above; MoE start gap +0.06 to +0.42 (cooperative launch, PDL off) | consumer-side: each MoE CTA could redo top-8 (256 logits) and the pooled norm for its tokens: 9 KB per CTA at T = 1 (1.2 MB over 132 CTAs), 72 KB at T = 8 (9.5 MB) | no. It needs the MoE prologue to stop reading `topk_ids` / `p.x` (FAST1 pre-issue, `load_x_regs`) and read logits / pooled instead, which is the MoE's x staging, owned by the MoE lanes; bitwise-equal ids need Triton's sigmoid and reduction order replicated | 0.5 to 1.0 at decode 1 (route's 1.27 / 1.30 less the added prologue), ~0 at T = 8 (L2 reads), none above |
| 7 | MoE -> unpooler | m BF16 [T, 4096] 8 KB / token | 1.42 (1024-row slice, T <= 8) / 3.73, 3.36 (d1), 5.28, 5.00 (v1); 11.36 / 14.25 to 15.26 (T = 16 / 32) | a contraction over all 4096 inputs of every routed expert, after the MoE's final pass: not element-wise | yes: the expert-to-unpooler slice (`g1b4_unp_release.py`; checks 4c2d5e3a8e1147bf, 0c35e5f5cc614806, c77b95102baf4232): the counter-chained tail costs five dependent global round trips, d1 80.67 / 85.35 against 76.75 / 83.73. Its reopen condition (at most one round trip after the barrier) is inside the MoE | ruled out as built; the reopen is MoE-internal |
| 8 | unpooler -> boundary 2 | T <= 8: slice BF16 [T, 1024] 2 KB / token; T >= 16: [T, 8192] 16 KB / token. b2 also reads s1 FP32 32 KB / token and writes s2 32 KB and the next input BF16 16 KB per token | ~0 to 0.7 / 1.61, 1.62 (d1), 2.10, 2.07 (v1), 3.81, 3.78 (d16), 2.85, 3.06 (d32) | two dependent row-wide reductions: over the 1024 slice values plus the carried tail (T <= 8), over 8192 at T >= 16; every one of the 8192 next-input values needs the second norm | the slice tail of row 7 included this boundary (lost). Producer-side last-arriver: one CTA would write 48 KB per token after the counter, about as long as b2 takes now | under 0.2 at T = 1, negative at T = 8 |
| 9 | store-side only: s2 columns >= 1024 at T <= 8 | b2 writes s2[:, 1024:] = s1[:, 1024:] (28 KB / token), a copy that needs no norm | - | element-wise copy; b1 could write it into s2 directly | no | under 0.1: b2 is latency-bound, not store-bound |
| 10 | T = 128 / 256 boundaries | nvjet GEMMs feed _graft_row_boundary and _graft_route | 4.8 to 7.4 per boundary, +0.2 to +1.0 us start gaps per edge | the same row-wide reductions as rows 3, 5, 8 | cuBLAS has no epilogue hook for a row norm | not a fusion target; the serial gaps are a PDL / kernel-choice question at weights 0.07 to 0.26 |

### A boundary cost the audit found that is not a fusion

At decode 16, b2 (glue_cluster_boundary POST) costs 3.81 / 3.78 us of critical path against 2.21 / 2.13 for b1, the same
template. The PDL8 edit `if constexpr (SLICED || POST) griddepcontrol.wait` puts the POST build's whole preload loop
(stream s1 and the alpha, beta, alpha2, pre and post weights) behind the entry wait, and the following `cluster.sync`
waits for those loads, so their latency is on the critical path (the weights are cold after the 1 GiB flush; inferred
from source, not separated). b1 issues the same loads before its wait. The decode 32 Gluon boundaries load their weights
before the wait and the stream after it, and there b1 and b2 cost the same (2.88 / 2.85). Hypothesis: issuing b2's weight
loads before the entry wait (no proof needed: they are weights) and an L2 prefetch of the s1 rows (a prefetch returns no
data, so it cannot break ordering) saves about 1 us at decode 16 sliding and full. The same applies to the stream load
after the wait in the decode 32 Gluon b1 (data['stream'] is the harness input; nothing in the call writes it) and to the
`NEXT` weights in `_sl_cluster_finish_cg`. Loading s1 itself before the wait would rest on a transitive argument
(route -> MoE and MoE -> unpooler are serialized launches, so b1 completed and was flushed before b2 can launch); that
moves a read across a griddepcontrol.wait and is the lead's call.

### Ranking (expected saving x cell weight, decode 1 sliding = 1.00)

| rank | candidate | cells | expected us per cell (hypothesis) | weighted | status |
|---|---|---|---|---:|---|
| 1 | route into the MoE prologue (consumer-side recompute, row 6) | d1 s / f | 0.5 to 1.0 | 0.7 to 1.35 | not ruled out by evidence; needs the MoE lanes (x staging) and a bitwise top-8 replica |
| 2 | b2 weight loads before the entry wait + s1 L2 prefetch (row 8; not a fusion) | d16 s / f; d32 b1 stream load | ~1.0 at d16; 0.3 to 0.6 at d32 | 0.75 to 0.9 | not tried; no fence, barrier or wait removed |
| 3 | route into a last-arriver tail of the pooler + router kernel (row 5) | d1 and v1, four cells | 0.2 to 0.4 | 0.45 to 0.9 | not tried; F3's counter result predicts the low end, below the 0.5 us short-cell bar |
| 4 | per-head q / k norm and RoPE in a head-aligned qkv epilogue (row 1) | T <= 32 | under 0.3 | under 1.0 (upper bound) | not tried; needs a qkv re-tile |
| 5 | b2 producer-side (row 8), s2 copy in b1 (row 9) | d1, v1 | under 0.2 | under 0.3 | not worth a build |
| - | b1 into o (row 3), b1 into the pooler (row 4), MoE -> unpooler (row 7) | | | | ruled out: M-C, F3, GLE (BF16 rounding), the unpooler slice checks; the consumer-side form by L2 arithmetic; the cluster form by the 15-cluster occupancy limit |

Reading: at T <= 32 exactly one boundary is element-wise (the output gate between attention and o), and it is already in
o's x path. Every other boundary needs a row-wide reduction (an RMSNorm over 4096 or 8192 values, or top-8 over 256
logits) over a producer whose output columns are spread across 128 to 256 CTAs, so moving the consumer into the
producer's epilogue needs a cross-SM exchange, and three earlier builds measured that exchange at about the cost of the
launch it removes. The fusion with the best expected value that no evidence rules out is route into the MoE prologue at
decode 1 (rank 1), and it sits inside the MoE lanes' kernel. Rank 2 is a load-ordering fix rather than a fusion. Every
expected saving here is a hypothesis from the timeline and the source; none was measured as a change.
## 2026-11-03 - Decode 32: B4 MoE timeline (box, no check)

The decode 32 fused MoE runs the B4 build (WGS warpgroup kernel, jobs of up to 64 tokens, the B4FIN final pass). Its
excess over the byte floor was 23.9 us at decode 32 sliding, the largest non-floor piece of the two weakest cells,
with no timeline behind it ("Decode 16 / 32 sliding: floor and gap"). Base: muse_decode_layer/cand/ly_micro2.py
(SHA256 0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd). Box only, no judge check, nothing
registered. Runs 2026-11-04 04:07 to 04:33 UTC (2026-11-03 20:07 to 20:33 PST), router seed 106218864389449120, f931
case seeds, local_start per input, PDL on at T = 1, 8, 16, 32. Logs, JSON (raw per-CTA stamps of 3 inputs per cell)
and SASS reports: `muse_decode_layer/profiles/box/lyb4/`.

### Instrument

- `tools/ly_moe_tlb4.py` writes `ly_tlb4.py` (rebuilt by the tool, not committed): per-CTA globaltimer stamps in
  MOE_CUDA_SRC under `-DOKC_TLB=<level>`, added to the B4 build only (MAIN compiles the same text with OKC_TLB 0). The
  existing DBG points give entry, routing done, x gather done, producers released (WS_JOIN), phase A end, phase B end
  and exit; the weights producer stamps its first issue and, for its first 40 stages, the time it issues stage n,
  which it does once the consumers released stage n - 3's slot and stage n - 2 has landed (the M <= 64 pacing); the x
  producer stamps and sums its waits (phase A xready, phase B doneAs); the B4FIN loop stamps the first final item
  after its MFIN wait. Stamps run only when the launch passes a fourth scratch tensor.
- Level 1 leaves the consumers' stage loop untouched: `tools/ly_tlb4_sass.py` (tools/ly_micro_sass.py plus a TLB build
  and nvcc warnings) gives the control's 304-instruction stage loop with the same opcode mix, 96 registers, 0 local, 8 B
  stack (`sass/sass_tl5.report.json`). Two first attempts did not: a per-thread branch anywhere in wgs_phase's segment
  loop, or a stamp inside the stage body even as a warp-uniform branch with predicated stores, moves the stage's
  descriptor arithmetic from the uniform datapath into regular registers (32 R2UR per stage, IMAD 36 -> 80 or more,
  304 -> 386 to 436 instructions; `sass/sass_tl3.report.json`). So level 1 has no consumer-side stage stamps; the
  producer's issue times are the stage clock, and segment counts come from each CTA's stage ranges on the host.
- `tools/ly_box_moe_tlb4g.py` captures two graphs of the whole layer per cell (plain, and one whose 32-token B4 launch
  passes the stamp buffer) and replays each fresh input through both with a cold L2. The paired graph time difference
  (instrumented minus plain, median per input) is +1.89 / +3.65 us (sliding / full); the stage loop is the control's,
  so the cost sits in the stamp stores or in replay order, not separated. `tools/ly_tlb4_cta.py` reads the raw stamps.

### Floor

One stage is 256 rows x 256 K of 4-bit codes plus SC6 scales, 35,840 B; one local expert is 512 phase A plus 256 phase
B stages, 27.53 MB. Box read bandwidth 4.443 TB/s (`sum` over 4 GiB, this run). The minimum stage count is 768 x
distinct, and the floor is 6.20 us per distinct expert: 55.76 us at the judge's 9.0 experts (6,912 stages, 52.4 per
CTA). One stage on every CTA at once is 132 x 35,840 B / 4.443 TB/s = 1.065 us, the per-stage floor interval.

### Measured, 40 inputs per cell (6 to 12 distinct, median 9)

Span fit: 32.13 + 5.18 x distinct (residual 2.00) sliding, 32.55 + 5.21 x distinct (1.92) full. Every input of both
cells processed exactly 768 x distinct stages: B4 never re-streams an expert at decode 32 (52 / 51 routes on 9
experts, so every job holds at most 64 tokens). Medians over the 12 / 14 inputs with 9 distinct experts, mean CTA,
sliding / full, us from the first CTA's entry:

| piece | sliding | full | floor | excess (sliding) | what it is |
|---|---:|---:|---:|---:|---|
| ramp: entry to first weight issue | 4.34 | 4.25 | 0 | 4.34 | routing done 1.75 / 1.70, x gather done 3.39 / 3.33, producers released 3.42 / 3.36, then 0.9 us to the first TMA issue |
| phase A: first issue to A end | 44.22 | 44.32 | 37.18 (34.9 stages) | 7.04 | A-end spread across CTAs 8.58 / 8.72 |
| phase B: A end to B end | 22.21 | 22.32 | 18.59 (17.5 stages) | 3.62 | handoff inside it: the x producer's first doneAs wait 3.23 / 3.19, all its doneAs waits 9.20 / 9.26 per CTA |
| MFIN wait: B end to first final item ready | 5.78 | 6.91 | 0 | 5.78 | B-end spread 7.70 / 8.83 |
| final pass | 1.97 | 2.00 | 0 | 1.97 | exit spread 2.10 / 2.13 |
| span | 78.59 | 79.87 | 55.76 | 22.83 | |

Stages per CTA 44 to 55 (cost-weighted stream-K ranges); per cell 404 phase A and 260 phase B segments (one CTA's
contiguous stages of one unit), 116 / 116 units cut between two CTAs. The weights producer's median issue interval is
0.99 us in phase A and 1.20 / 1.23 in phase B against the 1.065 floor, but the mean per-stage time over phase A is 1.24
(median CTA, raw inputs): the stream runs at the floor rate between segment switches and stalls at them.
Registers: 96 at 544 threads (the launch bound), 0 spill, 8 B stack (control and level 1 alike, `okc_attrs` on the box
96 / 8 / 0).

Issue gaps by where the released stage sits in its segment (`tools/ly_tlb4_cta.py`, 6 raw inputs, issue n waits for the
release of stage n - 3):

| released stage | gaps, median | excess over the median gap, per CTA, phase A |
|---|---:|---:|
| first stage of a segment | 2.21 to 2.69 us | 4.87 to 6.28 us |
| second stage | 1.02 to 1.06 | 0.13 to 0.82 |
| any later stage | 0.96 to 0.99 | -0.79 to +0.19 |

### Largest non-floor piece

Phase A's 7.0 us, and inside it the segment switch: the issue after each segment's first stage waits 2.2 to 2.7 us
against 1.0 for every other stage, 4.9 to 6.3 us per CTA over its three phase A segments. Cause, from the source and
SASS: the consumers run the previous segment's deferred `wgs_flush` behind commit group OKC_WGSFCI = 4 of the first
stage, before load group 3 arrives on that stage's weights-empty barrier. The flush is a proxy fence, a warpgroup
bar.sync and a release (`red.release.gpu` is MEMBAR.ALL.GPU + REDG in SASS; a cut unit's share takes
`atom.acq_rel.gpu`, MEMBAR + ATOMG with a returned value, and the last one reads the partials back). A diagnostic build
that times the flush (`--flush`, `ly_tlb4f.py`; its stage loop is perturbed to 386 instructions, so only the flush
times count) gives, per whole-unit doneA publish: proxy fence 0.004 us, bar.sync 0.25, release 0.66 to 0.69. The next
pieces are the MFIN wait (5.8 us, set by the 7.7 us B-end spread) and the ramp (4.3 us).

### Mechanisms on the segment switch (both losses)

Paired A/B against ly_micro2 (`tools/ly_b4_ab.sh`: five bx_box_run.py processes c, w, c, w, c, `--runs 60`, `--lock
--ls-cache --ext-cache --pin-pdl8 1,8,16,32 --skip-pdl8-gate`, every run timed with PDL on at every T; table
`tools/ly_ws2t_table.py`, threshold max(2 x control spread, 1% of the control mean)). Each candidate first ran all six
loads once (`chk`, `--runs 5`): reference check PASS at all 12 scored cells, G_RELEASE_STARTUP_OK (56 verified replays)
in every prepare.

1. ly_b4_fci6.py (`tools/dl_ly_b4_fci6_patch.py`, SHA256 12a0c52bb32ec0cef6a5ac5063b87f9d3614121839cdf7f00c30366f4706717a):
   OKC_WGSFCI 4 -> 6, the flush behind the weight-slot release (B4 and MAIN). SASS (`sass/fci6.diff.txt`): stage loops
   unchanged in count (304 / 307), B4 25,528 -> 25,390 instructions, stack 8 -> 0, and in the first stage the
   SYNCS.ARRIVE on the weights-empty barrier now precedes the FENCE.VIEW.ASYNC.G / BAR.SYNC / MEMBAR.ALL.GPU group. The
   timeline moves the stall instead of removing it: first-stage gaps fall to 1.50 to 1.60 us, second-stage gaps rise to
   2.46 to 2.96 (the consumers are still held about 1.5 us, now while the next stage's slot is theirs), phase A 45.14 ->
   46.45 us. A/B (`fci6/`, 04:16-04:18 UTC), candidate minus control mean, us: decode 32 +1.66 / +0.65, verify 16 +3.37 /
   +2.15, verify 32 +2.37 / +2.99, decode 1 +0.02 / -0.20, verify 1 -0.19 / -0.56. All inside the thresholds (decode 32
   2.59 / 4.67, verify 16 3.49 / 3.28, verify 32 3.20 / 4.60), but slower in all four pairs at decode 32 sliding and at
   all four verify 16 / 32 cells. Not kept.
2. ly_b4_dfa.py (`tools/dl_ly_b4_dfa_patch.py`, SHA256 47ac5034965b0c69d915f4894c5f83ffcc0d613099dc2413c7d320a0e638a621):
   OKC_WGSDFIN 2 -> 3, phase A cut units finished by the share holding stage 15 (phase B's existing path), so the other
   share counts with a fire-and-forget release instead of an atom.acq_rel round trip. SASS (`sass/dfa.diff.txt`): ATOMG
   20 -> 2, REDG 56 -> 74, MEMBAR.ALL.GPU 112 unchanged, stage loops unchanged in count. Timeline: A-end spread 8.58 ->
   17.87 us, B-end spread 7.70 -> 13.71, MFIN wait 5.78 -> 10.64, span 78.6 -> 83.9 (sliding medians); the changed code
   is the finisher's spin on its partner's count, but which partners arrive late was not traced CTA by CTA. A/B (`dfa/`, 04:28-04:30 UTC): decode 32 +5.27 / +3.28 (thresholds
   1.44 / 2.96), verify 16 +7.59 / +7.10, verify 32 +6.74 / +5.03, every pair positive; decode 1 and verify 1 inside.
   Loss.

Reading: the switch costs the consumers about 1.5 us per segment wherever the flush sits, and at M <= 64 the producer
keeps two stages in flight, so a held consumer stops its CTA's stream. What remains for this piece is to take the
release (bar.sync plus MEMBAR, 0.9 us of each publish) off the consumer warpgroups: the consumers would arrive on an
mbarrier after their proxy fence and a spare lane would wait on it and issue the `red.release.gpu`. That keeps every
fence and release but needs a mailbox of pending publishes with back-pressure, and it was not built in this round.
Estimated at most 1 to 3 us per CTA in phase A before the other CTAs absorb the freed bandwidth (hypothesis).

### The ramp (ly_b4_wpre, no gain)

ly_b4_wpre.py (`tools/dl_ly_b4_wpre_patch.py`, SHA256 f5bbb64f6065e904a5fed63723d713e2bdb6958c4a6b4e3425a583e7716993b5)
targets the third piece: the weights producer waits at WS_JOIN, which also waits for the consumers' x gather, though
the weight stream needs only the stage ranges. The x producer now arrives on a new named barrier 8 (the two producer
warps, 64 threads) after wgs_ranges; the weights producer, after bar.sync 8, issues the first min(2, phase A stages)
stages and then joins WS_JOIN as before (every thread still passes it; no fence, barrier or wait removed). SASS
(`sass/wpre.diff.txt`): BAR.ARV / BAR.SYNC 0x8, 0x40 once each per kernel, B4 25,528 -> 25,585 instructions, 96
registers, 0 local, 8 B stack, stage loops unchanged. Reference check PASS at all 12 cells, G_RELEASE_STARTUP_OK in
every run. Timeline (`tlb4g_wpre.log`; the tool also stamps the early issues): first weight issue 4.34 -> 4.02 us
(sliding) and 4.25 -> 4.02 (full), so the ranges finish close to the gather (routing done at 1.75 us, ranges about
2 us later); phases A, B and the tail are unchanged (span 78.59 -> 79.02 sliding). A/B (`wpre/`, 04:40-04:42 UTC,
same protocol): decode 32 +0.61 / +1.04 us (thresholds 1.45 / 2.97), verify 16 +0.72 / +0.30, verify 32 +0.36 /
+0.70, decode 1 +0.16 / -0.31, verify 1 -0.00 / +0.77 (TINY2, unchanged code, so this is the session's null). No
gain; not kept. The ramp's lever, if any, is the 2 us wgs_ranges takes after routing, not the barrier.

## 2026-11-03 - Blueprint builds (fable lane, box only, no check)

Question: on muse_decode_layer/cand/ly_micro2.py (SHA256 0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd),
which instruction-level, memory-layout or kernel-boundary changes allowed by the rules save measurable time, one change
at a time, with a SASS (or PTX) diff before each paired box A/B? No routing slack, no job sizing change, no fence, barrier
or griddepcontrol.wait removed, no new PDL edge, every change keeps the output bits. The decode 32 B4 MoE build's unit
loop and phases belong to another lane and were not touched. Box only (okc-h200, 2026-11-04 04:40-05:00 UTC), no judge
quota.

Tools: `tools/dl_ly_fab_patch.py --variant V --out F` (base SHA checked, output SHA printed), `tools/ly_fab_ab.sh`
(one `bx_box_run.py --lock --ls-cache --ext-cache --pin-pdl8 1,8,16,32 --skip-pdl8-gate --runs 60` process per run,
router seed 106218864389449120, f931 case seeds, order control, variant(s), repeated, then a final control; a process
takes 15 to 20 s with warm caches, so a round of five is two minutes), `tools/ly_micro_table.py` for the tables
(threshold max(2 x control spread, 0.5 us at decode 1 / verify 1, 1% of the control mean elsewhere)), and
`tools/ly_nm_capture.py` + `tools/ly_nm_compile.py` (the box audit's non-MoE capture, now in the repo: import with CUDA
hidden, compile the tvm_ffi glue sources with nvcc 13.0 on the CPU, SASS and resource summary). Logs, JSONs, summaries:
`muse_decode_layer/profiles/box/lyf/` (`lyf_b2pre/`, `lyf_b2pre6/`, `lyf_glu/`, `lyf_six/`, `sass/`,
`lyf_tl_*.timeline.txt`). Every run of every arm passed the startup expert comparison (G_RELEASE_STARTUP_OK, 56
verified replays) and the reference check at every cell it timed.

### Analysis first

The four-pattern table of "Micro-optimizations on the box" above was re-read against ly_micro2 and stands: the MoE
helper-warp epilogues are packed (wspack, wgspack) and predicated-store-only (wsmask measured null), the MoE SiLU is
MUFU with the ftz range fix, no MoE build spills, the sk2 projections and glue kernels store STG.E.64 / .128 and load
through TMA or float4, attention's hot loop is clean. Pattern 0 (compute-store fusion) is covered by the audit section
above: the only element-wise boundary is already fused; the rest need a row-wide exchange that three earlier builds
measured at the cost of the launch they remove. What the audit left was its rank 2 (load ordering around boundary 2's
entry wait, not a fusion), the same ordering question at the other boundary kernels, and its rank 1 (route recomputed
in the MoE prologue at decode 1), which is weighed below.

Load-ordering survey (what each T <= 32 chain kernel issues before its griddepcontrol.wait, from the source and the
PTX / SASS): qkv, o, pooler and unpooler sk2 stage weights before the wait (one stage at T = 32 since combo2); attention
waits at entry (its K / V reads are the attention lane's); glue boundary 1 at T <= 16 loads the stream (the harness
input) and its weights before the wait; the PDL8 glue boundary 2 at T = 16 waited at entry for everything (POST);
`_sl_tile_route_cg` preloads bias, local_start and the pre-expert norm weight (PRELOAD = True); the Gluon boundaries
at T = 32 load weights before the wait and the stream after it (boundary 1's stream is the harness input);
`_sl_cluster_finish_cg` (boundary 2 at T <= 8) loads beta and post before the wait and the NEXT norm weight after it;
the MoE is a cooperative launch with PDL off, so its entry wait returns at once.

### Changes, SASS / PTX effect and box A/B

Files (SHA256): ly_fab_b2pre.py 14afa879da4ddefe4deb5e97a528e51a22efa9c7af4707f736890dae21bb8f59, ly_fab_b1d32.py
be297008c330053f9ce6331ba9b1afeea87b085f2bdfef6fe5ffd4b453446521, ly_fab_finpre.py
5dc2827e702a32c7cb572ece6cd3b4729a6652f2830c9edf4141ef3513f9c03a. Cells are sliding / full, net us.

| change | what | SASS / PTX effect | runs (ctl / var) | cells: control mean / variant mean / delta | decision |
|---|---|---|---|---|---|
| b2pre | T = 16 boundary 2 (PDL8 graft glue, `glue_cluster_boundary<4,32,true,...>`): the entry griddepcontrol.wait no longer precedes the alpha / beta / pre / post weight preloads (prepare-time constants); each thread prefetches its two s1 chunks into L2 (`prefetch.global.L2`); the wait follows the preloads and the s1 loads (`.cg`) stay behind it; the second wait before the branch (unpooler output) loads is unchanged; SLICED keeps its entry wait | POST kernels: 8 LDG (weights) and 2 CCTL.E.PF2 now precede the first ACQBULK (base: ACQBULK at 0x60 before every load); the two LDG.E.128.STRONG.GPU s1 loads follow it; second ACQBULK before the LDG.E.64.STRONG.GPU branch loads unchanged; 408 -> 416 and 416 -> 432 instructions, 54 / 55 registers, no local. b1 instantiations byte-identical (`sass/glue_*.summary`) | round 1: 3 / 2; round 2: 7 / 6; six loads: 4 / 3 | d16 s: round 1 129.71 / 129.07 / -0.64 (per run -1.04, -0.24); round 2 129.14 / 128.89 / -0.25 (-0.46 -0.11 -0.05 +0.58 -0.88 -0.58); six-load 129.34 / 128.59 / -0.75 (-0.99 -0.67 -0.60). d16 f: 202.39 / 201.10 / -1.29 (-1.47 -1.10); 202.26 / 201.38 / -0.59 (-0.07 -1.17 -0.98 -0.60 -0.90 +0.19); 201.17 / 200.81 / -0.36 (+0.03 -0.43 -0.67). Over all 11 variant runs: d16 s -0.46 (10 of 11 negative), d16 f -0.65 (9 of 11). Guards: d1 s +0.03 / -0.45 / -0.06, d1 f -0.18 / +0.10 / +0.18, v1 s +0.05 / -0.29 / +0.27, v1 f +0.06 / -0.25 / +0.04 (unchanged code), d32 and v16 / v32 within 0.75 of the control (unchanged code) | inside the threshold (1.29 to 1.30 at d16 s by the 1% rule, 2.0 to 3.0 at d16 f); consistent sign at both decode 16 cells |
| b1d32 | T = 32 boundary 1 (`_graft_row_boundary_pre_cg`, POST = False): the stream load (data['stream'], the harness input; nothing in the call writes it) moves before gdc_wait; boundary 2 (POST) keeps its s1 load behind the wait with .cg | Triton PTX (`/root/.triton/cache/.../_graft_row_boundary_pre.ptx`, 04:48:47 UTC): the non-POST instance issues 8 ld.global before griddepcontrol.wait (base 6: the two stream v2.b32 loads moved up); the POST instance unchanged | 4 / 4 (lyf_glu) | d32 s 145.11 / 145.51 / +0.40 (+0.04 +0.39 +1.14 +0.01), d32 f 296.56 / 295.91 / -0.65 (-1.80 +0.18 -0.46 -0.52); other cells (unchanged code) -0.43 to +0.07 | inside; no consistent sign at the changed cells (d32 s 3 of 4 positive, d32 f 3 of 4 negative), not kept |
| finpre | T <= 8 boundary 2 (`_sl_cluster_finish_cg`): the NEXT norm weight load (prepare-time constant) moves before gdc_wait; carry, tail and the slice stay behind it with .cg | Triton PTX: `ld.global.b16` (NEXT) at line 94, before `griddepcontrol.wait` at 100 (base: wait at 86, the b16 load at 147 after the four .cg loads); the T = 1 and T = 8 instances both move | 4 / 4 (lyf_glu) | d1 s 66.09 / 66.33 / +0.24 (+0.08 +0.50 +0.34 +0.04), d1 f 72.96 / 72.69 / -0.28 (-0.70 -0.06 -0.18 -0.16), v1 s 103.54 / 103.73 / +0.19, v1 f 111.64 / 111.66 / +0.02; d16 / d32 (unchanged code) -0.11 to +0.58 | inside; signs mixed (d1 sliding positive in all 4, d1 full negative in all 4), not kept |
| b2gl16 | T = 16 boundary 2 dispatched to the one-program-per-row Gluon kernel `_graft_row_boundary_pre_cg` (the file's T = 32 boundary: 16 programs of 32 warps, weights before gdc_wait, branch and s1 after it with .cg, PDL launch) instead of the 8-CTA-cluster glue `b2c` (128 CTAs of 128 threads, cluster dim 8). Python dispatch only (`_pdl8_graft_boundary`); both kernels, their waits and launch attributes are the file's own | no new SASS: the Gluon POST instance is the one in the b1d32 row (its PTX: pre / alpha / beta / post loads, griddepcontrol.wait, launch_dependents, branch and stream .cg loads); the glue POST kernels are no longer launched at T = 16 (timeline below) | 5 / 4 (lyf_gl, d1 + d16); 7 / 6 (lyf_gl6, d1 + d16 + v1); 4 / 3 (lyf_fabsix, six loads) | d16 s: 129.29 / 127.88 / -1.41 (-1.61 -1.24 -1.67 -1.11); 129.23 / 128.05 / -1.18 (-1.34 -1.21 -1.10 -1.07 -0.81 -1.58); 129.12 / 128.05 / -1.07 (-0.85 -1.25 -1.12). d16 f: 201.66 / 200.67 / -0.99 (-1.10 -0.67 -1.48 -0.70); 201.82 / 200.93 / -0.89 (-1.76 -1.07 -1.93 -0.64 -0.97 +1.01); 201.78 / 200.70 / -1.08 (-1.00 -0.97 -1.27). All 13 variant runs negative at d16 s (mean -1.23), 12 of 13 at d16 f (mean -0.97). Guards (unchanged code): d1 s -0.08 / -0.00 / -0.11, d1 f +0.01 / -0.17 / -0.02, v1 s -0.09 / -0.46, v1 f -0.26 / -0.77, d32 +0.15 / +0.08, v16 +0.13 / -0.02, v32 +0.63 / +0.06 | inside by the 2 x spread clause (thresholds 1.73 / 1.92 / 1.44 at d16 s, 2.02 / 3.39 / 2.02 at d16 f), beyond the 1% clause (1.29) in two of three sessions; the one consistent effect of this lane, and the kernel timeline puts all of it on boundary 2. Kept as the stack |
| b1gl16 | the same dispatch for T = 16 boundary 1 (POST = False, no router copy at T = 16) | no new SASS (the b1d32 row's non-POST instance, base load order) | 5 / 4 (lyf_gl) | d16 s 129.29 / 128.83 / -0.46 (-0.67 -0.20 -0.95 -0.01), d16 f 201.66 / 202.07 / +0.41 (+0.68 +0.12 -0.38 +1.24); d1 +0.04 / +0.10 | inside, mixed signs: the cluster glue boundary 1 (2.2 us after the o projection) has no penalty to recover; not kept |

Kernel timeline at decode 16 for ly_fab (= b2gl16; `lyf_tl_ly_fab.timeline.txt`): boundary 2 is now
`_graft_row_boundary_pre_cg`, duration 4.78 / 4.93 us against the glue's 5.73 / 5.84, critical-path share 2.71 / 2.73 us
against 3.70 / 3.73 (sliding / full), the other eight kernels within 0.3 us of the control's; so the whole cell delta is
boundary 2's. Reading of the mechanism (hypothesis, consistent with b1gl16's null): the glue's 8-CTA clusters need eight
free SMs in one GPC to be scheduled, and the unpooler's 8192-row sk2 grid holds the SMs until it retires, so the
PDL-launched boundary 2 clusters start late; after the o projection (boundary 1) the same clusters schedule on time, and
the per-row Gluon programs need one SM each. b2pre's preloads recovered 0.35 to 0.45 us of the same kernel; the dispatch
swap recovers 1.0 us and makes b2pre moot at T = 16 (the glue POST kernel is no longer launched there).

### Stack ly_fab.py: b2gl16 alone (SHA256 916a8020deb7588920ee0dbe71aaef5a9ffcd55b466bc163b4e299fc16efd38a, the same bytes as ly_fab_b2gl16.py), six loads (lyf_fabsix: 4 controls, 3 stack runs)

| cell | control mean | control spread | ly_fab mean | delta | per run | threshold | verdict |
|---|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 66.24 | 0.69 | 66.12 | -0.11 | -0.12 +0.20 -0.41 | 1.38 | inside |
| decode 1 full | 72.46 | 0.32 | 72.44 | -0.02 | -0.32 +0.40 -0.14 | 0.64 | inside |
| decode 16 sliding | 129.12 | 0.72 | 128.05 | -1.07 | -0.85 -1.25 -1.12 | 1.44 | inside |
| decode 16 full | 201.78 | 0.90 | 200.70 | -1.08 | -1.00 -0.97 -1.27 | 2.02 | inside |
| decode 32 sliding | 145.43 | 0.53 | 145.58 | +0.15 | -0.06 +0.56 -0.03 | 1.45 | inside |
| decode 32 full | 296.51 | 1.39 | 296.59 | +0.08 | +0.13 +0.54 -0.43 | 2.97 | inside |
| verify 1 sliding | 103.83 | 1.17 | 103.37 | -0.46 | -1.45 -0.01 +0.07 | 2.34 | inside |
| verify 1 full | 112.09 | 0.69 | 111.32 | -0.77 | -1.69 +0.12 -0.74 | 1.38 | inside |
| verify 16 sliding | 243.41 | 0.77 | 243.53 | +0.13 | +0.99 -1.02 +0.42 | 2.43 | inside |
| verify 16 full | 326.70 | 1.78 | 326.67 | -0.02 | +1.26 -1.14 -0.18 | 3.55 | inside |
| verify 32 sliding | 296.29 | 3.01 | 296.92 | +0.63 | +0.45 +0.35 +1.10 | 6.02 | inside |
| verify 32 full | 460.85 | 2.99 | 460.91 | +0.06 | -0.19 -0.08 +0.45 | 5.98 | inside |

Every stack run passed the startup expert comparison and the reference check at all 12 cells (33 of 33 runs across
lyf_gl, lyf_gl6 and lyf_fabsix print G_RELEASE_STARTUP_OK; 12 of 12 cells PASS in each six-load run). b2pre is not in
the stack: with b2gl16 the glue POST kernel it edits is not launched at any scored shape, and the two were not measured
together.

Kernel timeline at decode 16 (`tools/ly_tl_run.py`, 20 runs, 10 profiled replays, `lyf_tl_*.timeline.txt`): boundary
2's critical-path share (its end minus the unpooler's end) is 3.70 / 3.73 us on ly_micro2 and 3.37 / 3.29 us on
ly_fab_b2pre (sliding / full), its duration 5.73 / 5.84 -> 5.42 / 5.54 us, every other kernel within 0.3 us of the
control's; cell net 130.03 / 204.70 -> 129.09 / 205.17 in those single profiled runs. So the preloads moved about 0.35
to 0.45 us off boundary 2, not the audit's hypothesised 1 us: the remaining 1.1 to 1.2 us that boundary 2 costs above
boundary 1 (2.2 us, same template) is not weight latency. Hypothesis, not separated: boundary 2 starts later relative to
its producer's end (start gap -1.95 against -2.45 for boundary 1) because the unpooler's 8192-row sk2 grid leaves fewer
free SMs for the 128 PDL-launched glue CTAs than the o projection does.

### Six-load run of b2pre (ly_fab_b2pre against ly_micro2; lyf_six: 4 controls, 3 variants; measured before b2gl16 superseded it)

| cell | control mean | control spread | ly_fab_b2pre mean | delta | per run | threshold | verdict |
|---|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 66.26 | 0.21 | 66.19 | -0.06 | -0.10 -0.26 +0.16 | 0.50 | inside |
| decode 1 full | 72.64 | 0.69 | 72.81 | +0.18 | +0.22 +0.01 +0.31 | 1.38 | inside |
| decode 16 sliding | 129.34 | 0.59 | 128.59 | -0.75 | -0.99 -0.67 -0.60 | 1.29 | inside |
| decode 16 full | 201.17 | 0.64 | 200.81 | -0.36 | +0.03 -0.43 -0.67 | 2.01 | inside |
| decode 32 sliding | 145.09 | 0.46 | 145.08 | -0.01 | +0.41 -0.12 -0.32 | 1.45 | inside |
| decode 32 full | 295.90 | 0.29 | 295.88 | -0.02 | +0.66 -0.46 -0.27 | 2.96 | inside |
| verify 1 sliding | 103.22 | 0.34 | 103.49 | +0.27 | +0.46 +0.09 +0.25 | 0.67 | inside |
| verify 1 full | 111.56 | 0.86 | 111.59 | +0.04 | +0.08 -0.63 +0.67 | 1.73 | inside |
| verify 16 sliding | 243.42 | 1.46 | 243.11 | -0.31 | -0.35 -0.40 -0.19 | 2.91 | inside |
| verify 16 full | 327.53 | 1.30 | 327.34 | -0.19 | -0.48 -0.04 -0.04 | 3.28 | inside |
| verify 32 sliding | 297.52 | 1.84 | 296.79 | -0.73 | -0.52 -1.21 -0.44 | 3.68 | inside |
| verify 32 full | 460.14 | 1.58 | 459.99 | -0.15 | +0.00 -0.60 +0.16 | 4.60 | inside |

The change touches only the T = 16 PDL8 glue, so every cell but decode 16 is a null here; the nulls span -0.73 to +0.27.

### Route recomputed in the MoE prologue (the audit's rank 1): bounded on the box, not built

The decode 1 chain runs `_sl_tile_route_cg` as a (1, 2) grid: one program normalizes the pooled row into xm, one does
sigmoid, bias, top-8 (packed uint64 `tl.topk`), the weight normalization and the ids / lids / wts stores; the MoE
(cooperative launch, PDL off) consumes lids with lo = 0, wts in its final pass and xm as x (`load_x_regs`, and the
FAST1 pre-issue warp reads the ids at entry to start the first weight units before the routing tables exist). The
fusion would move all of that into every TINY CTA (logits and bias reads, top-8 by one warp, a 4096-wide norm by the
rest, xm into shared memory, CTA 0 writing ids / lids / wts for the PDL8H guard, which compares them) and skip the
route launch at T = 1 in `_pdl8_sl_tile_after_attention`, for both pooler output forms (split pooled + logits, and the
4608-wide FP32 row).

Bound on the saving: route's whole critical-path share at decode 1 is 1.27 / 1.30 us (timeline above), and it includes
the wait-return latency every PDL secondary pays. The MoE's FAST1 warp would then have to load the logits (L2) and run
the top-8 before it can issue the first units, which delays the first fill by the same work route does now minus the
launch: on the box the route kernel's own compute is about 0.7 to 0.9 us of its 1.27, so the net is 0.4 to 0.6 us at
decode 1 if the added prologue costs 0.3 to 0.5 us, at or under the 0.5 us threshold and inside this session's decode 1
control spread (0.21 to 0.61 us). The one load-ordering change that did measure (b2pre) realised 0.35 to 0.45 us of a
hypothesised 1 us. The build is two to three hours (a Params and launcher change in the TINY build, a bitwise top-8
replica of the Triton kernel's packed sort and `div.full` sigmoid, the pooled norm with Triton's reduction order or an
accepted last-bit difference in xm, and the Python path for both output forms).

Measured upper bound (diagnostic pair, timing only, `--no-check`; `tools/dl_ly_fab_patch.py --variant rtdiag_ctl /
rtdiag_var`, files ly_rtdiag_ctl.py 5601d00d and ly_rtdiag_var.py cc022635; logs `lyf/lyf_rtd/`, timeline
`lyf/lyf_tl_rtvar.timeline.txt`): both arms route the T = 1 token from the previous call's pooled row and logits
(state buffers copied after the finish kernel, two memcpy nodes that both arms carry), so the MoE runs the same stale
routing in both; the control keeps the route kernel after the pooler, the variant launches it before the pooler, so the
MoE follows the pooler directly. The variant's timeline confirms the order (b1, route, pooler + router, MoE with a
+0.42 us launch gap after the pooler against +0.06 after route in the control). Four controls, three variants, 60
runs each: decode 1 sliding 62.77 control (spread 0.40) against 62.52, -0.25 us (per run -0.10 -0.32 -0.32); decode 1
full 70.90 (spread 0.10) against 71.33, +0.42 us (+0.50 +0.50 +0.26). So removing the route kernel from the chain
entirely, with no prologue work added to the MoE, is worth -0.25 us at decode 1 sliding and costs +0.42 at decode 1
full (the cooperative MoE launch waits longer behind the 128-CTA pooler than behind the 2-CTA route). A fusion that
also adds the top-8 and the pooled norm to the MoE prologue cannot clear the 0.5 us threshold at either cell. Closed on
that measurement; not built.

### Reading

- Nothing cleared the letter of the threshold (2 x control spread, with five to seven controls). One change is a
  measured effect by every other reading: b2gl16, the T = 16 boundary 2 dispatched to the file's own per-row Gluon
  kernel, is faster in all 13 of its runs at decode 16 sliding (mean -1.23 us) and 12 of 13 at decode 16 full (-0.97),
  the kernel timeline puts -1.0 us on that kernel alone, and every other cell is a null. It is the stack, ly_fab.py
  916a8020, rule-clean (an existing kernel of the file at an existing PDL edge; no wait, fence or barrier touched),
  and the judge check is the lead's call (cell weights put 1 us at decode 16 sliding at 0.45 of a decode 1 us).
- b2pre (weights and an s1 L2 prefetch before boundary 2's entry wait) recovered 0.35 to 0.45 us of the same kernel
  and is superseded by b2gl16; b1d32 and finpre were nulls. The load-ordering family is exhausted at T <= 32: every
  chain kernel now issues its constants before its wait, and the two remaining producer-written reads (s1 in
  boundary 2, the stream in the T = 32 boundary 2) must stay behind it.
- The route-in-MoE fusion is closed by the diagnostic bound (-0.25 / +0.42 us at decode 1 with the route kernel
  removed outright).
- The time at decode 16 / 32 remains where the gap tables put it: the fused MoE's issue rate (the other lane's B4 build
  at decode 32; TINY2 / MID at 1.18 us per unit against 1.10 of bytes), the sk2 projections' 2.4 to 3.6 us per kernel
  above their byte floors (wave quantization and ramp, the bf16 lane's closed item), and attention's latency.
## 2026-11-03 - Pooler + router timeline (box, no check)

bf16 lane. The question: where do the 15.7 to 18.5 us of the fused pooler + router kernel go against its byte floor?
It is the second-largest kernel on the decode 1 critical path ("Compute-store fusion audit"). And does one mechanism
against the largest non-floor piece shorten the layer? Base: muse_decode_layer/cand/ly_micro2.py (SHA256
0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd), never edited. Box only (okc-h200), every run under
gpu.lock, no judge check, nothing registered. Runs 2026-11-04 05:10 to 05:40 UTC (2026-11-03 21:10 to 21:40 PST),
router seed 106218864389449120, f931 case seeds, local_start per input, PDL on at T = 1, 8, 16, 32 (`--pin-pdl8
1,8,16,32 --skip-pdl8-gate`). Logs, JSONs (raw per-CTA stamps of 2 inputs per cell), kernel timelines and SASS
reports are in `muse_decode_layer/profiles/box/prtl/`.

### Kernel

At T = 1 and 8 the kernel is `sk2_kernel_gle` (gle extension): the BF16 pooler on the 12-bit lossless stream plus the
FP32 router, whose 4 router stages are interleaved into the pooler ring (GLE_ILV 1: stages 4, 9, 14, 19 of 20). At
T = 16 and 32 it is sk2 FUSE (bf16_proj extension), with the BF16 hi / lo router as 4 stages after the 16 pooler
stages. There are 128 CTAs of 416 threads (8 consumer warps, 1 producer warp, 4 escape warps), one per SM (about
206 KB of shared memory), 32 pooler rows per CTA, a ring of 6 / 6 / 5 / 3 stages at T = 1 / 8 / 16 / 32, and 2 weight
stages issued before griddepcontrol.wait (1 at T = 32). Each CTA also computes one router piece: 8 router rows (tile
rt = cta / 4) over one K quarter (rq = cta % 4). It stores that partial, then tid 0 does `atom.add.release.gpu` on the
tile's counter. The CTA that brings the count to 4 issues `fence.acq_rel.gpu`, reads the four partials back and writes
the logits. The pooler epilogue (K-warp reduction through the ring, BF16 store) runs after that.

### Instrument

- `tools/dl_ly_prtl_patch.py` writes `ly_prtl.py` (1f2f47c9): per-CTA globaltimer stamps in bfp_SK2_SRC under
  `if (FUSE && dbg != nullptr)`. The fused launches pass a host pointer (new export `sk2_set_tlb`, null by default) as
  the kernel's dbg argument, which they passed as nullptr before. Stamps: entry, first pre-wait weight issue, pre-wait
  issues done, after griddepcontrol.wait, x tiles issued, the producer's issue time of each stage, escape warp 0's
  full-barrier return for each stage (the stage-landing clock, never earlier than the landing), each consumer warp's
  stage-loop exit, router partial stored, counter done, last-quarter output stored, epilogue start, reduction written,
  final store, SM id. Nothing is stamped inside the consumers' stage loop.
- SASS (`tools/ly_prtl_sass.py`, `tools/ly_prtl_loopdiff.py`, `sass_bfp.log`): the consumer stage loop keeps its
  instruction count at every T (gle T <= 8: 386 -> 386; T = 16: 177 -> 177 with the same opcode mix; T = 32: 203 -> 203
  with the same mix, and the T = 32 router-stage loop 67 -> 65). At T <= 8 the 386 instructions differ in 5 integer
  opcode choices (IADD3 / IMAD.MOV / VIADD / LEA substitutions and one MOV for an IMAD.MOV.U32) with the same
  registers. Registers 128 -> 128 (gle), 84 -> 88 (T = 16), 126 -> 127 (T = 32), 0 local, 0 stack; this does not
  change occupancy at one 416-thread CTA per SM.
- `tools/ly_box_prtl.py` captures two graphs of the whole layer per cell (stamps off, stamps on) and replays 5 fresh
  inputs through both with a cold L2. Stamped minus plain graph time, median per cell: -0.9 to +4.1 us (median +0.4;
  single replays, so this is mostly replay noise). `tools/ly_prtl_summary.py` and `tools/ly_prtl_cta.py` read the
  JSONs.

### Floor

The kernel reads the pooler stream, 50,331,648 B (4096 x 8192 at 12 bits); the escape tables, 59,392 B (pmax 58 x
8 B x 128 CTAs); the router weights, 8,388,608 B (256 x 8192 FP32 at T <= 8, BF16 hi + lo at T >= 16, the same bytes);
and the activations, T x 8192 x 2 B plus the FP32 copy T x 8192 x 4 B at T <= 8 (49,152 / 393,216 / 262,144 /
524,288 B at T = 1 / 8 / 16 / 32). The totals are 58.83 / 59.17 / 59.04 / 59.30 MB, so at 4.443 TB/s the floor is
13.24 / 13.32 / 13.29 / 13.35 us. One pooler stage on all 128 CTAs (24,576 B each) takes 0.708 us at that rate, one
router stage (16,384 B) 0.472. The activations are counted although b1 wrote h a moment earlier (likely L2 hits), so
the floor is slightly high.

### Stamps, ly_micro2 (us; medians over 5 inputs per cell)

"wait" is the first CTA's griddepcontrol.wait return, in us after the first CTA's entry; the other columns are in us
after it. land0 is the first stage landing (escape warp 0). Loop end is the mean CTA's stage-loop exit (T <= 8:
pooler and router stages; T >= 16: pooler stages only). Router is the partial stores, the counter and the last-quarter
read-back (at T >= 16 also the 4 router stages). Tail is the last CTA's final store minus the latest loop end.
Post-wait span is the last final store minus the first wait return.

| cell | wait | land0 | loop end | loop-end spread | router | epilogue | tail | post-wait span | floor | excess | issue interval pool / router |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 1.54 | 0.83 | 14.41 | 1.18 | 1.79 | 0.48 | 2.02 | 17.50 | 13.24 | +4.26 | 0.58 / 0.51 |
| decode 1 full | 1.02 | 1.20 | 13.37 | 0.77 | 1.79 | 0.47 | 1.98 | 15.94 | 13.24 | +2.70 | 0.58 / 0.54 |
| verify 1 sliding | 1.38 | 0.84 | 15.39 | 1.09 | 1.87 | 0.47 | 1.82 | 17.86 | 13.32 | +4.54 | 0.58 / 0.54 |
| verify 1 full | 1.38 | 0.73 | 15.33 | 1.02 | 1.75 | 0.46 | 1.92 | 17.86 | 13.32 | +4.54 | 0.58 / 0.54 |
| decode 16 sliding | 1.18 | 1.29 | 11.51 | 4.13 | 3.64 | 0.51 | 2.88 | 16.83 | 13.29 | +3.54 | 0.48 / 0.51 |
| decode 16 full | 0.77 | 1.49 | 11.55 | 4.06 | 3.61 | 0.51 | 2.85 | 16.45 | 13.29 | +3.16 | 0.45 / 0.51 |
| decode 32 sliding | 1.89 | 0.65 | 12.66 | 2.88 | 3.90 | 0.72 | 4.13 | 18.05 | 13.35 | +4.70 | 0.64 / 0.54 |
| decode 32 full | 1.54 | 0.64 | 12.78 | 3.20 | 4.38 | 0.72 | 3.90 | 18.46 | 13.35 | +5.12 | 0.67 / 0.54 |

Ramp. The CTAs enter within 0.7 to 1.6 us of each other. At decode 1 sliding 4 CTAs enter about 1 us late, and 26 to
28 at decode 32 (hypothesis: they wait for the SMs the preceding boundary kernel still holds). The two pre-wait stages
go out 0.5 to 1.0 us after entry. The first stage lands 0.6 to 1.5 us after the wait, because its x tile is issued
only after the wait.

Loop. From the first landing the stream runs at or above the 4.443 TB/s rate. The producer's issue interval is 0.58 us
per pooler stage at T <= 8 and 0.45 to 0.67 at T >= 16, against 0.708 at the floor rate (TMA streaming beats the `sum`
read used for the floor), and the median CTA's 20 stages land within 11.9 and 13.6 us in the two raw decode 1 sliding
inputs. The ring's first refills are spaced about 0.5 us apart although 4 slots are free after the wait (stages 2 to 5
issued at 1.8, 2.4, 2.9, 3.9 us in the median CTA at decode 1), so the copies queue behind each other rather than
behind a free slot (an observation; the queue was not traced further).

Tail. At T <= 8 the last CTA spends 1.8 to 2.0 us between its stage-loop exit and its final store: 0.4 us for the
router partial (warp butterflies, stores, the escape warps' cbar, a barrier), 0.66 to 0.93 us for the counter atom
(the release waits for the partial stores) and its barrier, 0.25 to 0.73 us for the last quarter's acquire fence,
read-back and logit store, then 0.47 us of pooler epilogue. In all 16 raw inputs the 6 latest-finishing CTAs are the
last quarters of their router tile, and on average the last quarters' final store comes 0.57 to 1.32 us after the
other CTAs'. At T = 16 / 32 the 4 router stages (1.5 / 2.2 us) come before the same counter and read-back.

### Largest non-floor piece

The largest non-floor piece is the tail after the stage loop at T <= 8 (1.8 to 2.0 us on the last CTA), and inside it
the router's counter round trip plus the last quarter's read-back (1.2 to 1.5 us). At T = 16 / 32 the same counter
and read-back sit behind the router stages, and the largest piece there is the loop-end spread across CTAs (2.9 to
4.1 us). The ramp after the wait (0.6 to 1.5 us) comes next at every T.

### Mechanism: the router's quarter sum moves into the route kernel (ly_prq, ly_prq2)

`tools/dl_ly_prq_patch.py` builds both. The fused kernel skips the counter block when it is launched with
outr == nullptr (the launchers pass nullptr for an empty outr tensor). The partial stores, the barrier after them and
the pooler epilogue are unchanged, and the counter path stays in the code for a non-empty outr. The fused-launch
wrappers (`_gle_frt` at T <= 8, `bfp_fused_pooler_router` at T = 16 / 32) pass an empty outr and return the partial
buffer, tagged, as the logits. Each route kernel that reads split logits gets a `_prq_` copy whose logit load is
((p0 + p1) + p2) + p3 of the four quarter partials, the FP32 adds in the order the last quarter used. That covers
`_sl_tile_route_cg`, the one the layer launches at T = 1, 8, 16, 32, and `_route_split`, `_graft_route`,
`_sl_graft_route` and `_sl_tile_route`, which prepare's composition timing reaches. A wrapper launches the copy when it
is given the tagged buffer, and `_route` / `_gle_route` raise if they ever get it. The route's loads stay after its
griddepcontrol.wait, and no fence, barrier or wait is removed from a kernel that still runs.

BF16 and the logits stay bitwise. `tools/ly_prq_bitwise.py` (random rows, T = 1, 8, 16, 32) finds 0 differing logits
between the counter path and the torch quarter sum, identical partials and pooled rows with and without outr, and
bitwise-equal ids, local ids, weights and expert inputs between the old route kernel on the logits and the copy on the
partials (`prq_bit.log`, `prq2_bit.log`).

- ly_prq (5ce323a2872e318a8529f800f829b74070d511863f8347d808d4b8745f37c11d) keeps the partial layout
  [32 tiles][4 quarters][32 tokens][8 rows], so the route reads each quarter as 32 separate 32-byte pieces.
- ly_prq2 (80b6ab9c4982aebc7f630156648723279062244c55231ddd6b6392570170d36b, `--contig`) stores the partials as
  [4 quarters][32 tokens][256 rows] (the last-quarter read-back follows), so each quarter of a token is one contiguous
  1 KB row for the route.

Kernel timeline (`tools/ly_tl_run.py --kernels`, 10 profiled replays, `ktl_both.txt`, `ktl_prq2.txt`). The fused
kernel's duration falls from 17.76 to 16.91 / 16.80 us at decode 1 sliding (control to ly_prq / ly_prq2) and by 0.9 to
1.5 us at the other short cells. With ly_prq the route ends 1.89 us after the fused kernel at decode 1 sliding
(control 1.25) and 2.21 at verify 1 sliding (control 1.63); with ly_prq2 it ends 1.42 and 1.66 us after it. Route end
minus boundary 1 end, control to ly_prq2 (sliding / full): decode 1 16.87 -> 16.25 / 18.08 -> 16.03, verify 1 18.70
-> 17.44 / 18.19 -> 17.28, decode 16 18.70 -> 17.44 / 18.60 -> 17.54, decode 32 21.09 -> 19.25 / 20.51 -> 19.30. In the
stamped ly_prq2 build (`tl_prq2.json`) the tail falls from 2.02 to 0.80 us and the post-wait span from 17.50 to 14.75
us at decode 1 sliding; the verify 1 tail is 0.77 to 0.80 us and the decode 16 / 32 tail 1.70 to 2.94 us.

Paired A/B: `tools/ly_prtl_ab.sh` runs five bx_box_run.py processes (control, candidate, control, candidate, control)
at all six loads with `--runs 60`; `tools/ly_prtl_table.py` makes the table with threshold max(2 x control spread,
0.5 us) at decode 1 / verify 1 and max(2 x control spread, 1% of the control mean) elsewhere. Every candidate run had
reference check PASS at all 12 cells, G_RELEASE_STARTUP_OK (56 verified replays) in prepare, and PDL on at T = 1, 8,
16, 32 in every timed run. Candidate minus control mean, us (sliding / full). Verify 16 / 32 (T = 128 / 256) do not
run the changed code and are the session's null:

| cell | ly_prq round 1 (`prq/`) | ly_prq round 2 (`prq2/`) | ly_prq2 round 1 (`prq2c/`) | ly_prq2 round 2 (`prq2d/`) | ly_prq2 pooled (6 controls, 4 runs) |
|---|---|---|---|---|---|
| decode 1 | -0.68 (win) / -0.47 | -0.43 / -0.41 | -0.94 / -0.51 | -1.07 / -1.15 (win) | -1.01 / -0.83 (win) |
| verify 1 | -0.82 / -0.40 | -0.18 / -0.70 | -0.81 / -1.33 (win) | -0.85 (win) / -1.49 (win) | -0.83 / -1.41 |
| decode 16 | -1.04 / -0.67 | -0.90 / +0.14 | -0.93 / -1.37 | -0.88 / -0.54 | -0.91 / -0.95 |
| decode 32 | -0.95 / -0.84 | -1.15 / -0.22 | -1.66 (win) / -1.35 | -1.15 / -1.47 | -1.41 / -1.41 |
| verify 16 | +0.40 / -0.45 | +1.01 / +0.26 | -0.64 / +0.26 | -0.35 / +0.27 | -0.50 / +0.27 |
| verify 32 | -0.21 / -0.63 | -0.40 / +0.64 | -0.60 / +0.04 | -0.65 / +0.35 | -0.62 / +0.19 |

The ly_prq2 pooled thresholds (sliding / full) are decode 1 1.47 / 0.83, verify 1 2.50 / 1.54, decode 16 1.29 / 3.46
and decode 32 2.05 / 3.10, set by the spread of six controls across two sessions. Over both ly_prq2 rounds the
candidate run is below the control run in 24 of 24 pairs at decode 1 sliding and full, decode 16 sliding, decode 32
sliding and verify 1 full, in 23 of 24 at decode 32 full, in 20 of 24 at decode 16 full and verify 1 sliding, and in
7 to 18 of 24 at the long cells.

Reading. Removing the counter round trip shortens the fused kernel by about 1 us at every short cell, and with the
contiguous layout the route gives none of it back. ly_prq2 is faster than ly_micro2 on all eight short cells in both
rounds (0.5 to 1.7 us per cell, 0.8 to 1.4 pooled) and slower nowhere, and the unchanged long cells stay within -0.65
/ +0.35. It passes the threshold on 2 of 12 cells in round 1, 3 of 12 in round 2 and 1 of 12 pooled. So the strict
keep rule (every short cell beyond max(2 x spread, 0.5 us)) is not met on all cells: the shortfall is the threshold
(two-session control spreads of 0.4 to 1.7 us), not a cell that moved the wrong way. ly_prq (strided partials) is
superseded, because its route ends 0.6 us later at decode 1 / verify 1 sliding. What is left of the tail after ly_prq2
at T <= 8 is the router partial's 0.4 us (escape-warp cbar wait and barrier) and the 0.4 us epilogue. The next pieces
are the loop-end spread at T = 16 / 32 (2.9 to 4.3 us) and the ramp after the wait (0.6 to 1.5 us).

## 2026-11-04 - Unit loop at issue rate (box)

Task 1.3b of docs/plans/2026-11-04-moe-cluster-rewrite.md (section 9): restructure the fused MoE's skinny_ring_ws unit
loop (helper0, the TINY / TINY2 / MID builds: decode 1, verify 1, decode 16) for latency hiding, on the public layer
base `muse_decode_layer/cand/ly_fabq.py` (SHA256 3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555,
never edited). Box only (okc-h200, 2026-11-04 07:14-08:14 UTC), no judge job, no routing slack.

Tools: `tools/dl_ly_ul_patch.py --variant V [...] --out muse_decode_layer/cand/ly_ul_<v>.py` (SHA-pinned base, one
mechanism per variant, variants stack, output SHA printed); `tools/ly_ul_ab.sh` (ly_micro_ab.sh with ly_fabq as the
control: one `bx_box_run.py --lock --ls-cache --ext-cache --pin-pdl8 1,8,16,32 --skip-pdl8-gate --runs 60` process per
run, router seed 106218864389449120, f931 case seeds, rounds of control then variants, a final control, a kill
timeout); `tools/ly_micro_table.py` for the tables (threshold max(2 x control spread, 0.5 us at decode 1 / verify 1,
1% elsewhere)); `tools/ly_micro_sass.py` for SASS (the five builds' exact sources and flags, compiled on the box CPU
with nvcc 13.0); the variant `tl` adds the MOE-TL-01 timeline (`tools/moe_tiny_timeline_cta.py` edits, as
`tools/dl_moe_tl.py` applies them) to TINY2 for `tools/box_moe_tl8.py` (20 verify 1 inputs per cell, local_start per
input). Every A/B run printed `G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1, 8, 16) 56 verified_replays` with PDL
pinned on at T = 1, 8, 16, 32, and passed the reference check at every cell it ran. Every variant keeps the unit math
and its order (four chains combined c0 + ((c1 + c2) + c3), warp-order cross-warp sums in the helper), so the BF16
points are bitwise the base's; no fence and no griddepcontrol.wait is removed. Logs, JSONs, timelines and SASS
reports: `muse_decode_layer/profiles/box/ul/`.

### What the base loop is

Per unit (36,864 bytes: 32 KB of codes, 4 KB of scales) each of the 8 compute warps decodes its 512-wide K range: 32
code words, each 9 integer ops (5 LOP3, 2 SHF, 2 IMAD.SHL: the DQ7 unpack) and 4 HMUL2 (scale), then one HMMA, in
four accumulator chains. The token operand stays in 64 registers for the whole job (16 LDG.128 per lane at a job
switch), so the loop has 9 LDS.128 per warp per unit and no shared-memory B loads. At the end of every unit each warp
waits for its partial buffer, stores its partials, runs `fence.proxy.async`, and the 8 warps meet at a CTA barrier;
then thread 0 refills the slot with unit q + 3 and hands the partials to the helper warp. SASS per warp per unit
(phase A / phase B loop bodies, including the job-switch branch and thread 0's refill branch): TINY 702 / 754, TINY2
690 / 741, MID 692 / 747; 141 to 145 registers, no local memory.

The pipe count per SM sub-partition (2 compute warps each) is about 1,060 ALU-pipe cycles (LOP3 / SHF / PRMT at 16
lanes) and about 1,200 issue slots per unit, which is 0.6 to 0.65 us at 1.8 to 1.9 GHz; the loop retires a unit every
1.15 us (timeline below), against 1.098 us for the unit's bytes on 132 SMs at 4.44 TB/s. So the loop is about 5%
over the byte rate, not 2x: it runs at about half its issue rate, but the stream is the next limit.

### Variants (SASS: TINY / TINY2 / MID loop bodies, phase A / phase B; registers; local 0 everywhere)

| variant | mechanism | loop instructions | registers |
|---|---|---|---|
| base | ly_fabq | 702/754, 690/741, 692/747 | 145, 141, 144 |
| nb (d75f37ca) | compute warps arrive on a named barrier (8 + q % 3, 288 threads) instead of the CTA barrier; the helper warp waits on it, refills, reads the partials | 612/673, 596/656, 590/654 | 143, 140, 144 |
| na (46c6512b) | the CTA barrier split: warps 1-7 arrive, warp 0 waits (named barrier, 256 threads) and thread 0 refills as before | 715/769, 703/756, 699/757 | 151, 142, 146 |
| pf (c02b7841) | software pipelining across units: after its last mma each warp waits for unit i + 1's slot and loads its scale word and first 2 code chunks | 725/776, 713/763, 717/771 | 154, 149, 157 |
| napf (8d34a394) | na + pf | 743/-, 731/-, 727/- | 152, 152, 161 |
| nbpf (be1847c8) | nb + pf | 644/693, 630/679, 622/674 | 151, 153, 157 |
| er (e7d3ec55) | early slot release: each warp loads its whole 4.5 KB segment into registers after the slot wait, fences and arrives; thread 0 refills at the unit's start (one more unit of stream lead with no extra shared memory); the end barrier stays for the partials | 715/764, 703/751, 705/757 | 159, 162, 163 |
| pw (fb9392c6) | per-warp ring: each unit copy split into 8 per-warp segments (4 KB codes + 512 B scales) on per-warp barriers; each warp waits for, refills and releases only its own segment (fence, `__syncwarp`, lane 0 issues and arrives on the partials barrier, count 8) | 712/761, 700/749, 701/749 | 145, 142, 146 |
| pr (217cc386) | a refill warp (NTL 288 -> 320): compute warps store partials, fence and arrive on a named barrier (8 + q % 3, 288 threads); the refill warp waits on it, refills the slot and arrives on the helper's partials barrier; the epilogue helper is unchanged | 612/673, 596/656, 590/654 | 143, 140, 144 |
| ns4 (3a008642) | four ring slots on TINY / TINY2 / MID (`-DOKC_NS=4`) | 699/751, 687/738, 689/744 | 148, 142, 148 |
| prns4 (cfe2758e) | pr + ns4 | 608/669, 592/652, 586/650 | 146, 142, 146 |
| dq (f3960a7d) | task 1.3a's r3 merge as an IMAD (x * one + y) instead of a LOP3 | 735/786, 723/773, 725/779 | 137, 140, 146 |

Ordering for the barrier variants. nb, na and pr keep every compute thread's `fence.proxy.async` after its last read of
the slot and before its arrival, and the refilling thread issues only after the barrier wait returns, the same
pattern as the base's barrier followed by thread 0's issue; the partials reach the helper through the barrier wait
followed by an mbarrier arrive (release) and the helper's wait (acquire), as in the base. A warp can arrive for unit
q + 3 only after the slot for q + 3 was refilled, which happens after the wait for unit q returned, so three barrier
ids never hold two generations. The WSBL-kept last units keep the CTA barrier (their epilogues read every warp's
partials). pw replaces the CTA barrier by `__syncwarp` before lane 0's refill and arrival; each warp reads only its own
segment (codes [4096 w, 4096 w + 4096), scales [32768 + 512 w, + 512)).

Not built, with the reason:

- ldmatrix for the token operand: there is no shared-memory token load in this loop (x stays in registers per job);
  CL-1.2's 31 LDS per unit came from its smem token slab.
- More accumulator chains: the four chains run 8 dependent HMMA per unit, about 260 cycles of chain latency against
  about 2,100 cycles per unit, and more chains change the FP32 order.
- A bit-permuted repack that cuts the DQ7 unpack: no layout gives four single-rotation groups. A search over every
  triple of rotations of the 8-bit field mask 0x81C081C0 (bits 6, 7, 8, 15 and their +16 copies) finds no partition
  of the 32 bits, so one group needs two pieces and 7 ALU ops plus 2 IMAD per 8 weights is the minimum for a
  rotate-and-mask decode with this BF16 field placement (`tools/dq7_rotation_search.py`, an exhaustive search, not a proof for
  other decodes). A BF16-scale repack would add bytes.
- dq was compiled but not run: ptxas already merges r3 with a three-input LOP3, so the IMAD form keeps 169 LOP3 and
  adds 28 to 33 IMAD per unit.
- Scheduling hints on the rolled loop: CL-1.2 measured an unrolled 120 KB hot loop losing; nothing else to hint.

### Paired box A/B against ly_fabq (net us; delta = variant mean minus control mean; per-run deltas in the logs)

| variant (round, runs) | decode 1 s / f | decode 16 s / f | verify 1 s / f | control spread (d1 s / f, d16 s / f, v1 s / f) | verdict |
|---|---|---|---|---|---|
| nb (r1, 2) | +1.39 / +1.51 | +0.81 / +0.19 | +2.30 / +1.49 | 0.67 / 0.37, 0.38 / 0.32, 0.66 / 0.35 | loss at decode 1 and verify 1 |
| nbpf (r1, 2) | +1.78 / +1.99 | +2.12 / +0.36 | +3.15 / +2.89 | same controls | loss |
| na (r3, 4) | +0.29 / -0.68 | +0.51 / +0.31 | +0.21 / +0.69 | 1.42 / 1.23, 0.56 / 0.56, 0.77 / 0.64 | inside everywhere |
| pf (r3, 4) | +1.75 / +1.16 | +2.29 / +2.29 | +0.51 / +0.39 | same controls | loss at decode 16 |
| napf (r3, 4) | +0.13 / +0.45 | +2.31 / +2.19 | +2.06 / +2.28 | same controls | loss at decode 16 and verify 1 |
| er (r4, 3) | +2.67 / +3.03 | +6.17 / +5.47 | +4.45 / +4.84 | 0.40 / 0.40, 0.67 / 0.51, 0.43 / 0.69 | loss everywhere |
| pr (r5, 3) | +0.31 / +0.25 | -0.21 / -0.14 | +0.41 / +1.31 | 0.24 / 0.37, 0.21 / 0.78, 0.24 / 0.51 | loss at verify 1 full, inside elsewhere |
| pw (smoke, 20 runs, no paired control) | 66.38 / 72.64 | 135.36 / 210.19 | 106.72 / 121.42 | (same-day controls 65.1-65.5 / 71.6-72.2, 126.6-127.2 / 197.6-198.6, 100.9-101.7 / 109.0-109.7) | loss, +8 to +12 us at decode 16 and verify 1 |
| ns4 (r6, 3) | +2.02 / +2.24 | +0.16 / +0.06 | +2.72 / +2.53 | 0.48 / 0.34, 0.54 / 1.30, 0.67 / 0.59 | loss at decode 1 and verify 1 (the knob sweep's ns4 result, again on ly_fabq) |
| prns4 (r6, 3) | +0.54 / +0.89 | +0.47 / +0.69 | +1.41 / +0.93 | same controls | loss at decode 1 full and verify 1 sliding, positive in every run of every cell |

Round 2 (na, three rounds) ran while another lane's jobs shared the box and its control spread reached 2.3 us; it is
in the logs but not used. Every run of every round passed the reference check at all six cells.

### Timeline: where the loop's time goes (TINY2, verify 1, 20 inputs, median of the per-input mean CTA, us)

| build | span s / f | phase A s / f | phase B s / f | unit interval (warp 0, A / B) | warp 0 weights wait A / B (s) | B-end spread s |
|---|---|---|---|---|---|---|
| base (run 1 / run 2) | 62.00 / 70.19; 61.65 / 69.17 | 34.49 / 40.16; 34.66 / 38.50 | 15.92 / 18.22; 16.12 / 19.11 | 1.15 / 1.12-1.17 | 3.87 / 0.60; 3.67 / 0.60 | 2.96; 2.75 |
| nb | 65.01 / 70.90 | 36.50 / 40.14 | 16.13 / 18.66 | 1.12 / 1.06 | 9.96 / 2.21 | 3.54 |
| na | 62.06 / 67.87 | 33.37 / 38.43 | 16.29 / 18.75 | 1.22 / 1.18 | 4.13 / 0.63 | 3.04 |
| pf | 61.87 / 68.48 | 34.60 / 38.68 | 16.52 / 19.06 | 1.18 / 1.18 | 1.76 / 0.39 (first unit only) | 3.74 |
| er | 63.65 / 71.15 | 35.56 / 39.98 | 17.91 / 20.71 | 1.25 / 1.28 | 2.40 / 0.45 | 2.80 |
| pw | 63.62 / 71.17 | 33.42 / 38.04 | 17.48 / 20.29 | 1.25 / 1.20 | 2.10 / 0.68 | 3.20 |
| pr | 62.83 / 70.67 | 33.47 / 39.18 | 15.86 / 17.84 | 1.09 / 1.02 | 6.98 / 1.54 | 4.21 |
| prns4 (pr + 4 ring slots) | 64.10 / 70.42 | 34.37 / 38.80 | 14.96 / 17.40 | 1.06 / 0.99 | 7.21 / 1.04 | 4.62 |

prns4's A-to-B handoff is 4.2 / 4.3 us (base 3.1 to 3.2) and its barrier idle 3.5 (base 2.4 to 2.5).

Two base runs differ by up to 1.7 us in phase A and 1 us in span, so a single timeline run resolves only changes
larger than that.

### Reading

1. The loop can run below the byte rate. With the refill taken off warp 0 and no barrier wait in the compute warps
   (pr), warp 0 retires a unit every 1.09 us in phase A and 1.02 to 1.04 us in phase B, under the 1.098 us the bytes
   need on 132 SMs; in the lockstep base it is 1.15. The difference (0.06 to 0.1 us per unit) is the refill on warp 0
   plus the per-unit barrier, not the dequant chain: the decode itself has room.
2. The time does not come back, because the stream is the limit once the loop is faster. pr's phase A is the base's
   within the run-to-run band (33.5 / 39.2 against 34.5-34.7 / 38.5-40.2) and warp 0's weights wait grows by the time
   the loop saved (3.7 to 7.0 us sliding, 3.9 to 8.7 full). Phase A streams 3,584 x 36,864 bytes in about 34.5 us at
   7 experts, 3.83 TB/s, 87% of the box's 4.44 TB/s read rate, in every variant. Phase B gains at most 0.3 to 0.8 us,
   and the faster loop leaves a larger B-end spread (2.9 to 4.2 us) and barrier idle, which the grid barrier pays.
3. Moving the refill later costs the stream: nb's helper warp refills only after its own epilogue of the previous
   unit, its weights wait rises to 10 us and phase A grows 2 us. Splitting the copy into per-warp segments (pw) costs
   2 us of span (16 bulk copies per unit and per-warp refill code: unit interval 1.25) and much more in the layer.
   Releasing the slot a unit early through registers (er) cuts the weights wait by 1.3 us but slows the loop to 1.25
   us per unit (162 registers, a second barrier per unit).
4. Shared loads are not the latency either: pf moves the next unit's slot wait and first loads ahead of the hand-off
   and the unit interval stays at 1.18; at decode 16 it costs 2.3 us.
5. With the loop's slack (pr) a fourth ring slot changes sign in phase B only. ns4 alone still loses 2 to 2.7 us at
   decode 1 and verify 1, as in the knob sweep. prns4's phase B runs 0.99 us per unit and finishes in 14.96 / 17.40 us
   (base 15.9-16.1 / 18.2-19.1), which is 66 MB in 15 us, 4.4 TB/s, the box's read rate; but phase A stays at
   34.4 / 38.8 and the A-to-B handoff (+1.1 us), the B-end spread (+1.7) and the barrier idle (+1.0) take the gain
   back: a faster loop on uneven per-CTA weight arrival widens the finish spread that the doneA waits and the grid
   barrier pay for.
6. So the binding limit at verify 1 is the weight stream in phase A (3.83 TB/s, 87% of the read rate, the same in
   every variant, including the job-switch waits inside it) and the per-CTA spread of its delivery, not the unit
   loop's instruction latency. The plan's premise (loop latency-bound at 2x its issue rate) holds for the loop in
   isolation, but in the layer the lockstep loop runs about 5% over the bytes, so a loop at issue rate could save at
   most about 0.05 us x 27 units = 1.4 us of phase A and 0.5 us of phase B per verify 1 cell, and only if the stream
   kept up; pr and prns4 show that it does not.

Decision: no winner, and no judge check recommended. The nearest arms are pr (inside the threshold at five of six
cells, verify 1 full +1.31 against 1.02) and prns4 (+0.5 to +1.4 us, positive in every run of every cell). Rows added
to docs/ruled-out.md. What would move phase A is the stream itself: the 13% of read bandwidth phase A does not reach
with 132 x 3 bulk copies of 36 KB in flight, and the job-switch waits (the L2 probe in "T = 8 MoE floor and gap"
drops them from about 1 us to 0.3 us when the weights are in L2). That is a memory-system question (request size,
DRAM page locality across 7 to 8 expert regions, the evict-first policy), with pr as the loop to pair it with: once
phase A streams above about 4.1 TB/s, pr's slack is worth up to about 1.4 us per verify 1 cell (hypothesis).
Lane 1.3a's cluster loop (muse_moe_w4a16/cl/cl_unit_loop.cuh on worktree-agent-a1596f680f085e146, read 2026-11-04
08:00 UTC, no committed result yet) is a different loop: its token operand comes from shared memory and its 16
consumer warps take setmaxnreg registers, so its limits need not match helper0's. One idea came from it (the IMAD
r3 merge, dq above), which does not help helper0's SASS.

## 2026-11-04 - Phase A stream efficiency (box)

Question from the lead (stream lane): why does helper0's phase A weight stream in the public layer file run at about
3.83 TB/s while the Family H skeleton reads the same bytes at 93 to 99%, and which change to helper0's streaming
brings phase A to 95% or more. Base `muse_decode_layer/cand/ly_fabq.py` (SHA256 3765470032deb505...dde555, never
edited). Box only (okc-h200, 2026-11-04 08:20-10:10 UTC, every GPU run under /root/okc/.build/box/gpu.lock), no
judge job, no routing slack. Logs: `muse_decode_layer/profiles/box/st/`.

Tools:

- `muse_moe_w4a16/cl/st_replay.cu` and `st_replay_bench.py`: a weights-only kernel. Each CTA streams its own list
  of 36,864-byte units through a cp.async.bulk ring of depth D with the L2 evict-first hint, one CTA per SM, no
  math. Options: consumer delay per unit (a globaltimer spin), 1 or 8 consumer warps polling the full barrier,
  helper0's refill form (consumer warp 1 lane 0 issues unit i + D after a named barrier), helper0's constant
  policy word, helper0-shaped evict_last bf16 stores per unit, extra mma.sync + integer burn warps, the ring's
  offset in shared memory, cooperative or eager launch, read or write flush, flush size. Patterns: h0 (helper0's
  muldiv ranges, expert-major), rr (unit i to CTA i mod G), rrb3 (blocks of 3), em (CTAs split into k groups,
  group g owns expert g, adjacent units to adjacent CTAs), emc (the same groups, contiguous chunks), and any
  pattern at G = 56 or 96 CTAs (em@56 is the H skeleton's 7 x 8 geometry with a static assignment).
- `tools/dl_ly_st_patch.py --variant rec` writes `muse_decode_layer/cand/ly_st_rec.py`: the MOE-TL-01 timeline on
  TINY2 and MID plus, per CTA and sequence number q, the issue time and expert block of every weight copy, the time
  warp 0 saw the slot full, warp 0's wait, and the SM id. Off-by-default diagnostic switches (wrong output, timing
  only): `ST_NOMATH` / `ST_SPIN` (each unit's decode and mma replaced by a spin of N ns), `ST_NOEPI` (no phase A
  epilogue in the helper warp), `ST_BARE` (no partials hand-off at all: wait, fence, CTA barrier, refill).
- `tools/box_st_rec.py`: runs the layer once per input (f931 case seeds, router seed 106218864389449120,
  local_start per input), records the helper0 launch, and replays it after an L2 flush: 3 plain launches (CUDA
  events), one recorded launch, and optionally the recorded phase A sequence streamed by st_replay.cu from the
  layer's own w13 tensor (`--replay`, same bytes, same order, same CTA lists). `--extra` builds the recorded
  launch with extra defines while prepare() and its checks use the file's own builds.

### Experiment 1: helper0's address pattern

helper0 (TINY2 at verify 1, MID at decode 16) gives CTA c the phase A units [floor(N c / 132), floor(N (c + 1) /
132)) of the expert-major list (N = 512 per distinct local expert), so each CTA streams one contiguous run of
27 to 29 units (about 1 MB), crossing at most one expert boundary, and phase B the same with 13 to 14 units. A
unit is one contiguous 36,864-byte copy: 8 gate rows and 8 up rows of a 16-row block with their scales, so gate
and up are not separate streams. The record confirms it on every input. At 30% and 60% of phase A, all 132 CTAs
have copies in flight (270 to 380 units), spread over all k experts (6 to 12) and 54 to 108 distinct 2 MB regions
(every 2 MB page of every active expert), with neighbouring CTAs 27 to 50 units apart (the muldiv stride, 31 at
7 experts). The H skeleton (CL-0.1) reads 7 experts with 8 CTAs per expert claiming adjacent units, so at any
moment about 7 windows of 32 units each.

### Experiment 2: replay table (TB/s, phase A, 7 distinct experts, consumer delay 0, read flush, 20 replays)

"span" is units x 36,864 B over first CTA entry to last CTA done (ramp and end spread included); "steady" is the
middle 80% of unit arrivals over their time window. The box's 4 GiB torch sum reads at 4.40 to 4.43 TB/s.

| pattern | CTAs | D = 3 span / steady | D = 4 span / steady |
|---|---:|---|---|
| h0 (helper0's ranges) | 132 | 4.44 / 4.82 | 4.42 / 4.84 |
| rr (round robin) | 132 | 4.44 / 4.81 | 4.42 / 4.82 |
| rrb3 | 132 | 4.41 / 4.79 | 4.40 / 4.84 |
| em (expert groups, adjacent units) | 132 | 4.41 / 4.80 | 4.43 / 4.83 |
| emc (expert groups, contiguous) | 132 | 4.40 / 4.82 | 4.41 / 4.85 |
| em@56 (H skeleton geometry) | 56 | 4.26 / 4.61 | 4.22 / 4.72 |
| h0@56 | 56 | 4.12 / 4.55 | 4.18 / 4.71 |
| h0@96 | 96 | 4.34 / 4.80 | 4.24 / 4.82 |

h0 at D = 2: 4.41 / 4.80. Phase B (h0): 4.15 / 4.91; A then B in one ring: 4.53 / 4.80. At 14 distinct experts
(264 MB, five times the L2) h0 steady is still 4.80, so the 4.8 TB/s is HBM, not L2 hits; with an 8 GiB flush it
is the same. The pattern makes no difference: every 132-CTA pattern streams within 1% of the others, and fewer
CTAs (the H geometry) is slower, not faster. With a write flush (dirty L2, as `tools/box_moe_tl8.py` uses) the
same replay drops to 3.63 span / 4.01 steady.

Replay with helper0's other traits, h0 at D = 3 (steady TB/s): 8 polling consumer warps 4.82; constant policy word
4.82; helper0-shaped evict_last stores 4.81; no L2 hint 4.75; refill by consumer warp 1 after a named barrier
4.84; ring at shared offsets 128 to 896 or behind 110 KB 4.81 to 4.85; cooperative launch 4.81; eager launch
4.82; 8 or 14 burn warps (mma.sync + integer) 4.82. A consumer that holds each slot 0.3 us (helper0's measured
hold) still reads 4.82 at a 0.99 us unit interval.

Same bytes: the recorded helper0 phase A sequence of each input, streamed by st_replay.cu from the layer's own
w13 tensor in the same process right after the recorded launch, reads 4.27 to 4.51 span and 4.80 to 4.85 steady
(8 inputs, verify 1), per-unit latency 2.78 us at D = 3.

### helper0 in place (verify 1, read flush, record build)

| build | unit interval A | warp 0 weights wait A per CTA | phase A steady TB/s | issue-to-ready latency |
|---|---:|---:|---:|---:|
| base (with math) | 1.184 | 3.2 to 3.4 us | 3.90 | 2.72 us |
| no math (`ST_NOMATH`, spin 0) | 1.15 | 20 to 25 us | 3.72 to 3.99 | 3.4 us |
| no math, no helper epilogue (`ST_NOEPI`) | 1.15 to 1.17 | 21 to 25 us | 4.0 (tbsA) | 3.4 us |
| bare loop (`ST_BARE`: wait, fence, barrier, refill) | 1.18 to 1.22 | 22 to 28 us | 3.82 | 3.5 us |
| no math, 4 ring slots | 1.15 to 1.17 | 20 to 24 us | 3.79 to 3.97 | 4.6 us |
| no math, 2 ring slots | 1.22 | 19 to 24 us | 4.05 (tbsA) | - |
| no math, spin 900 / 1000 ns | 1.376 / 1.472 | 1.7 to 1.9 us | consumer-bound | 2.9 / 3.1 us |

Base with a read flush against the write flush the 1.3b timeline used: phase A 36.6 / 37.3 us against 37.5 /
36.9 (verify 1 sliding / full medians), 3.85 against 3.76-3.84 TB/s, so the flush type explains at most 1 us.
Plain launch times (CUDA events, eager, no timeline) differ by only 3.1 to 5.1 us between the base and the no-math
build on the same inputs (for example 53.5 against 49.95 us), so the math costs little of the kernel.

### Reading

1. The address pattern is not the cause. helper0's exact per-CTA sequence, on the same bytes, at the same ring
   depth and unit size, streams at 4.8 TB/s steady (and 4.4 TB/s over the whole span, 100% of the torch read
   rate) in a weights-only kernel, and no alternative assignment does better. No prepare-time or route-time unit
   order can bring phase A to 95%, so experiment 3 as planned (port a winning pattern) has nothing to port.
2. The limit is inside helper0's kernel and is not its compute loop. With the decode and mma removed, and even
   with the partials hand-off and epilogue removed, helper0 still streams phase A at 3.8 to 4.0 TB/s, the same as
   with math (3.90). Ring depth 2, 3 or 4 gives the same rate with latency growing in proportion (2.72 -> 3.4 ->
   4.6 us), so it is a throughput ceiling of about 3.9 TB/s, not a latency or depth problem. The replay kernel,
   given every helper0 trait we could isolate (polling warps, refill thread and barrier, hold time, policy word,
   stores, shared offset, cooperative launch, compute load), keeps 4.8.
3. This corrects lane 1.3b's reading in one point: the loop is not 5% over the bytes; the bytes in this kernel
   are capped at about 3.9 TB/s and the loop (1.18 us per unit, 4.1 TB/s equivalent) sits just above that cap.
   It agrees with its conclusion that a faster loop alone does not help (pr, prns4).
4. Size of the lever (hypothesis): if helper0's phase A and B streamed at the replay's 4.8 TB/s, phase A at 7.5
   experts would drop from about 36.5 to about 31 us with the current loop (loop-bound at 1.18 us per unit), and
   to about 29 us with pr's 1.09 us loop; phase B by about 1 to 2 us. That is 5 to 8 us per verify 1 cell, more
   than the 2.7 us the brief assumed, and decode 16 (MID, 1.18 to 1.38 us per unit) likewise.

What is not yet isolated is which part of helper0's kernel caps the stream. Excluded so far by measurement:
pattern, bytes, flush type (up to 1 us), ring depth, the compute loop, the helper epilogue, the partials hand-off,
the policy word, the evict_last stores, 8 polling warps, the refill-after-barrier form, the ring's shared offset,
cooperative launch, eager launch, power load from mma warps. The timeline instrument is unlikely: removing it
(`-DOKC_TLX=0`) moves the bare build's plain launch time by -1.9 to +0.4 us per input (46.9 -> 45.7, 71.1 -> 69.2,
47.5 -> 45.9, 51.8 -> 52.2, 58.5 -> 58.3, 64.6 -> 64.5 us) and the base build's by -1.6 to -5.5 us, which cannot
hide a 20% stream gap in a kernel whose phase A alone is 27 to 37 us (inference from plain times; no counters:
ncu returns ERR_NVGPUCTRPERM on this pod). The per-launch zeroing is 3,072 floats (EMAX), negligible. FAST1's two pre-routing copies are not it either: the bare build
with `-DOKC_FAST1=0` streams 3.93 / 3.98 TB/s steady (2 inputs, latency 3.46 us). Not yet tested inside helper0:
the x prefetch, the route itself, and anything else in the kernel that runs while phase A streams.

Decision: no candidate, nothing to A/B, no judge check. The pattern lever is closed (rows in docs/ruled-out.md).
The lever that exists is helper0's stream ceiling of about 3.9 TB/s against 4.8 for the same copies in a plain
streaming kernel; the next step is to keep stripping the BARE build (the x prefetch, then the route itself with ranges
passed in) until it reaches the replay's 4.8, which names the part
to change. Expected value if found and removable: 5 to 8 us per verify 1 cell (hypothesis above).

## 2026-11-04 - MID single weight read at decode 16 (box)

Question from the lead (moe lane): at decode 16 the MID build splits any local expert with more than 8 tokens into two
8-token jobs and streams that expert's 28 MB of NVFP4 weights twice (stream lane's side finding). Build a candidate on
`muse_decode_layer/cand/ly_fabq.py` (SHA256 3765470032deb505...dde555, never edited) in which an expert's weights are
read once at decode 16 and decode 32 whatever its token count, output identical to ly_fabq, paired A/B on the box.
Box only (okc-h200, 2026-11-04 09:55 to 10:35 UTC, every GPU run under /root/okc/.build/box/gpu.lock; nvidia-smi showed
0% utilization and no other process before the first run), no judge job, no routing slack. Logs and JSON:
`muse_decode_layer/profiles/box/mid1/`.

Prior rows on the same lever, read first: ly_ws2t (two-tile jobs in one shared loop body, +1.67 / +2.11 us at
decode 16) and ly_ws2u (two-tile body only on the CTAs that hold a two-tile job, +4.13 / +4.72 us) on combo2n, both
losses; the MoE knob sweep's mid8 row (decode 16 on the MAIN WGS build, +2.72 / +3.84 us). Decode 32 runs the B4 build,
whose jobs hold up to 64 tokens, so it already streams every expert once (B4 timeline: 768 x distinct stages on every
input); the decode 32 half of the goal holds by construction and was only re-verified by the A/B cells.

### Ceiling: what the second read costs (ly_fabq, timeline in the layer graph)

`tools/ly_moe_tl16.py --base muse_decode_layer/cand/ly_fabq.py` (the tool now takes `--base / --pin / --out`) writes
`ly_fabq_tl16.py` (13368de9), the MID per-CTA globaltimer stamps on ly_fabq's own helper0 text; `tools/ly_box_moe_tl16g.py`
replays the whole layer graph with a cold L2, 40 fresh inputs per cell (f931 case seeds, router seed 106218864389449120,
local_start per input), units counted per input. Medians, sliding / full, us:

| | span | ramp | A | handoff | B | B-end spread | bar idle | final | unit A | unit B | warp 0 weight wait A | units | distinct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ly_fabq | 66.34 / 66.75 | 3.71 / 3.68 | 36.64 / 36.91 | 2.31 / 2.33 | 18.49 / 18.43 | 3.50 / 3.36 | 2.95 / 2.78 | 0.46 / 0.48 | 1.15 / 1.15 | 1.12 / 1.15 | 2.74 / 2.65 | 6144 / 6144 | 8 / 7 |

Excess units over 768 x distinct per input: 34 of 40 sliding inputs and 33 of 40 full inputs carry one split expert
(768 extra units), 2 per cell carry two, 6 / 7 carry none. Least squares over the 40 inputs, span = a + b x distinct +
c x split experts: sliding 11.02 + 6.77 x distinct + 7.29 x split (residual sd 0.80 us), full 12.05 + 6.63 x distinct +
7.31 x split (0.90); on the graph time 7.63 / 7.06 us per split expert. Phase A carries 4.42 / 4.45 of it and phase B
2.25 / 2.22. So the second read of one expert costs the same as one more distinct expert, 7.3 us of MID span, which
is the loop-bound cost of its 768 units (768 / 132 x 1.152 us = 6.7) plus its share of the finish. Averaged over the
inputs (0.9 split experts per input) the lever is 6.0 / 5.9 us per decode 16 cell; that is its ceiling, reached only
if the non-split units kept their 1.15 us interval.

Phase A is loop-bound, not stream-bound: warp 0 waits for weights 2.7 us per CTA over a 36.6 us phase (the stream
lane's finding on verify 1 again). That rules out the brief's second option (both jobs of a split expert on one CTA,
decoding the shared-memory unit twice): it keeps 6,912 loop iterations per 8-expert input and can save nothing
while the loop is the bound. The third option (decode 16 on the WGS build) is the knob sweep's mid8 loss. What is
left is the first option: one decode per unit, two n-tiles.

### Mechanism: ly_mid1 (every unit two n-tiles, every CTA the same loop)

`python3 tools/dl_ly_mid1_patch.py` (base ly_fabq, SHA checked; `--base / --pin / --out` like tools/dl_ly_prq_patch.py)
writes `muse_decode_layer/cand/ly_mid1.py`, SHA256 f1c9b4e31fa6afb999fa4433156413e10bc6bc586ac3bbf166a20cfb3466dd01
(the A/B ran on 3a2320ad, the same file with the two x1 declarations two lines later; the SASS report and the bitwise
check are of f1c9b4e3). MID gets `-DOKC_SKT=16`: one job per local expert (up to 16 tokens), and every unit of every
job runs two n-tiles on one decode of the weights. Tokens 0..7 come from the register fragments as in the control;
tokens 8..15 are staged per warp in shared memory (the region the smem size formula already reserves for OKC_SKT > 8)
by cp.async, zero-filled past the job's tokens, issued with the job's x0 loads and waited for after the unit's
ring-slot wait. A job of 8 or fewer tokens runs the same code against a zero second tile, so every SM runs one loop
body (the ly_ws2u reading: its two-tile CTAs stalled 3 to 6 us on their first unit of each phase while the one-tile
CTAs ran the control's loop). Tile 0 keeps the control's four accumulator chains and combine order, so tokens 0..7 of
every job are bitwise the control's; tile 1 has its own four chains combined the same way, so tokens 8..15 are bitwise
what the control computes for them as a separate job. Tile-1 partials go to `red1` at the end of Smem, written before
the same fence.proxy.async, CTA barrier and mbarrier hand-over as tile 0; the helper warp reads them before it frees
the buffer and runs the epilogue for tokens 8..15 when the job has more than 8 (the WSBLK last-unit epilogue likewise).
Unit ranges stay the control's count split, since every unit costs the same. No fence, barrier, mbarrier protocol,
griddepcontrol or launch attribute changes; every edit is under `#if OKC_SKT > 8` with the control text in the
`#else` arm, so TINY and TINY2 (decode 1, verify 1) preprocess to the base.

Correctness (`chk`, all six loads, `--runs 5`): reference check PASS at all 12 scored cells, G_RELEASE_STARTUP_OK in
prepare; in the A/B every run's check passed. Bitwise comparison against ly_fabq (`tools/ly_bitcmp.py`: both files
prepared on the same weights in one process, 20 fresh inputs per cell run eagerly through both, the `reference.collect`
bytes compared with torch.equal): not bitwise, and not measurable this way. ly_mid1 against ly_fabq differs on
19 / 18 of 20 decode 16 inputs (hidden and stream, max abs diff 4.7e-2 / 3.1e-2), but also on 7 / 2 decode 1 inputs,
13 / 14 decode 32 inputs and 12 / 12 verify 1 inputs, whose builds are byte-identical in both files; ly_fabq against
itself (`bitcmp_self.log`) differs the same way, 16 / 19 of 20 at decode 16 (3.1e-2 / 3.1e-2), 8 / 2 at decode 1,
14 / 15 at decode 32 (one decode 32 full input even flips a routing id), 15 / 12 at verify 1. The layer's output
is not reproducible bit for bit from one call to the next (FP32 atomics in the MoE's sum of squares and the
router, order-dependent), so the candidate's differences are inside the control's own run-to-run noise; the
reference check passed 20 / 20 at every cell on both sides. By construction the candidate's partial sums are
the control's chains in the control's order (source reading, not a measured identity).

SASS (`tools/ly_micro_sass.py`, MID build, nvcc 13.0 on the box CPU): control 144 registers, phase A unit loop 692
instructions with 32 HMMA and 17 LDS, phase B loop 747; ly_mid1 162 registers, 0 local, 0 stack, phase A loop 1006
instructions with 64 HMMA and 102 LDS (16 LDS.128 of x1 fragments per unit; the x1 loader's cp.async and S.ent loads
sit inside the loop region), phase B loop 912 with 64 HMMA. The second tile costs 32 HMMA + 16 LDS + the address
arithmetic per unit per warp on every unit, split expert or not.

### Timeline, ly_mid1_tl16.py (the same instrument on the candidate, 40 inputs per cell)

| | span | ramp | A | handoff | B | B-end spread | bar idle | final | unit A | unit B | units | distinct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ly_mid1 | 80.10 / 76.27 | 4.53 / 4.53 | 43.38 / 38.29 | 2.64 / 2.68 | 24.47 / 21.87 | 5.95 / 6.11 | 4.90 / 4.94 | 1.23 / 1.24 | 1.41 / 1.41 | 1.54 / 1.54 | 6144 / 5376 | 8 / 7 |

The re-stream is gone (0 of 80 inputs with excess units; span = 17.59 + 8.17 x distinct, residual sd 1.18, no split
term left). What eats it: every unit retires slower, 1.15 -> 1.41 us in phase A (+23%) and 1.12 to 1.15 -> 1.54 in
phase B (+35%), so the per-expert slope rises from 6.3 to 6.8 us to 8.0 to 8.2 us, more than the 7.3 us one split
expert cost; the ramp (+0.8 us, the x1 cp.async and the longer first unit), B-end spread (+2.5) and final pass (+0.8)
grow too. The loop is issue-bound: 1006 instructions per warp per unit against 692 at about the same IPC.

### Paired A/B against ly_fabq (`tools/ly_mid1_ab.sh`, 10:12 to 10:13 UTC)

Five `bx_box_run.py --lock --ls-cache --ext-cache --pin-pdl8 1,8,16,32 --skip-pdl8-gate` processes (control,
candidate, control, candidate, control), `--runs 60`, router seed 106218864389449120, f931 case seeds, every run timed
with PDL on at T = 1, 8, 16, 32; table by `tools/ly_ws2t_table.py muse_decode_layer/profiles/box/mid1/ab`; net us,
threshold max(2 x control spread, 1% of the control mean):

| cell | c_a | w_a | c_b | w_b | c_c | control spread | candidate - control (means) | threshold | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 64.94 | 65.26 | 65.26 | 64.93 | 65.46 | 0.51 | -0.13 | 1.02 | inside (TINY, unchanged code) |
| decode 1 full | 72.24 | 71.36 | 71.52 | 71.54 | 71.82 | 0.72 | -0.41 | 1.44 | inside |
| decode 16 sliding | 126.99 | 137.14 | 126.58 | 137.71 | 126.82 | 0.42 | +10.63 | 1.27 | loss (every pair +10.1 to +11.1) |
| decode 16 full | 200.00 | 208.66 | 198.66 | 208.59 | 199.42 | 1.34 | +9.26 | 2.69 | loss (every pair +8.7 to +10.0) |
| decode 32 sliding | 144.16 | 143.76 | 142.96 | 143.90 | 143.10 | 1.20 | +0.42 | 2.40 | inside (B4, unchanged code) |
| decode 32 full | 295.42 | 294.48 | 294.99 | 294.03 | 294.98 | 0.45 | -0.87 | 2.95 | inside |
| verify 1 sliding | 102.35 | 101.78 | 101.79 | 101.92 | 101.90 | 0.56 | -0.17 | 1.12 | inside (TINY2, unchanged code) |
| verify 1 full | 110.62 | 109.95 | 110.32 | 109.41 | 109.78 | 0.85 | -0.56 | 1.70 | inside |

Decision: loss at decode 16 (+10.6 / +9.3 us against a 6 us ceiling), the largest of the three two-tile attempts.
Not for a judge check; ruled-out row added. What it establishes, with ly_ws2t and ly_ws2u: on helper0's
register-resident mma.sync loop a second n-tile costs 23 to 35% per unit, and no split of that cost between CTAs has
beaten the 12.5% of units it saves. Running it on every unit (this candidate) loses the most; running it only where
needed (ly_ws2u) loses 4 us to a per-CTA first-unit stall that follows the code, not the data. The ceiling itself is
modest: 6 us per decode 16 cell at 0.105% of the score per us, about 0.6% at best.

What would be needed (not built): a two-tile loop no slower per unit than the one-tile loop, which means the second
tile's HMMAs and fragment loads must hide in the loop's latency slack (IPC about 0.6 per SMSP) rather than add issue
slots; or the ly_ws2u form with the two-tile code made warm before its first unit (its stall disappeared when every
SM ran the body, consistent with cold instruction fetch after the L2 flush; unproven without counters). Neither fits
the day's rule at 0.6% of the score.
## 2026-11-04 - Phase A producer cap (box)

Question from the lead (stream lane, second leg): which part of helper0's producer path caps its phase A stream at
about 3.9 TB/s when a weights-only replay of the same bytes streams at 4.8. Base `muse_decode_layer/cand/ly_fabq.py`
(SHA256 3765470032deb505...dde555, never edited). Box only (okc-h200, 2026-11-04 10:00-11:30 UTC, every GPU run under
/root/okc/.build/box/gpu.lock, no foreign GPU process in `nvidia-smi` before or during the runs, ncu unavailable),
no judge job, no routing slack. Logs: `muse_decode_layer/profiles/box/st/v_*.log`, `w_*.log`, `b_*.log`.

Tools added (commit 68b58532): `tools/dl_ly_st_patch.py --variant rec` gains off-by-default switches `ST_PLAIN`
(both phases as st_replay.cu's loop on helper0's own ring and ranges: wait, one shared read, `fence.proxy.async`,
CTA barrier, thread-0 refill, nothing else; helper as `ST_BARE`), `ST_NOROUTE` (job table from the FAST1 warp's
expert mask, no `route()`), `ST_NOX` (no x prefetch, no LUT), `ST_NOZERO` (no ss zeroing), and records `clock64`
with `globaltimer` at the first full slot and at the phase A end of every CTA, so each launch reports its SM clock.
`st_replay.cu` records the same clock and has a prologue stand-in (cold loads, 6 CTA barriers, a staggered spin).
`tools/box_st_rec.py` prints `smclk_mhz` and `cta_unit_a` (median per-CTA phase A unit interval from the ready
stamps) and replays the A+B sequence from the lower of the two unit tensors (w2 sits 288 MB below w13), so
the whole-kernel CUDA-event time of helper0 can be set against the replay of its exact bytes.

### helper0 in place, verify 1, read flush, 6 inputs each (3 sliding, 3 full)

`A` is tl8's phase A window (mean first-full to mean phase-A-end over CTAs), tbsA the phase A bytes over it,
`cta_unit` the median per-CTA interval between consecutive ready stamps, `wwait` warp 0's weights wait per CTA
summed over phase A, `lat` the median issue-to-ready time, `plain` the CUDA-event time of the whole launch.

| build (extra defines) | SM MHz in phase A | cta_unit us | tbsA | wwait per CTA us | lat us | plain us (seed 078 / 931) |
|---|---:|---:|---:|---:|---:|---:|
| base (math), run 1 | 1754 to 1774 | 1.218 to 1.235 | 3.97 to 4.02 | 1.6 to 3.0 | 2.72 | 57.1 / 62.8 |
| base (math), run 2 | 1739 to 1777 | 1.221 to 1.263 | 3.88 to 4.00 | 1.6 to 2.8 | 2.72 to 2.78 | 55.9 / 63.4 |
| no math (`ST_NOMATH=1 ST_SPIN=0`) | 1966 to 1979 | 1.018 to 1.086 | 4.66 to 4.95 | 13 to 23 | 2.72 to 2.78 | 51.0 / 56.5 |
| bare (`+ ST_BARE=1`) | 1967 to 1980 | 1.024 to 1.030 | 4.90 to 4.96 | 15 to 25 | 2.78 to 2.82 | 47.1 / 53.2 |
| plain loop (`ST_PLAIN=1`), run 1 | 1969 to 1979 | 1.033 to 1.097 | 4.64 to 4.93 | 22 to 37 | 2.82 to 2.91 | 46.8 / 52.1 |
| plain loop, run 2 | 1977 to 1979 | 1.031 to 1.039 | 4.86 to 4.93 | 22 to 37 | 2.7 to 2.8 | 46.9 / 52.2 |
| plain, no route, no x, no zeroing (`+ ST_NOROUTE=1 ST_NOX=1 ST_NOZERO=1`) | 1968 to 1979 | 1.040 to 1.111 | 4.62 to 4.91 | 23 to 39 | 2.7 to 2.8 | 46.1 / 52.4 |
| same-bytes replay (st_replay.cu, phase A list, same process) | 1978 to 1980 | 0.977 to 0.992 | 4.32 to 4.47 span / 4.79 to 4.85 steady | - | 2.69 to 2.75 | - |
| same-bytes replay, A then B in one ring, eager event time | 1979 to 1980 | 0.985 to 1.02 | 4.41 to 4.52 span | - | - | 43.0 / 49.1 |

### Replay with helper0's remaining traits (st_replay_bench, h0 pattern, 7 experts, D = 3, 10 replays)

| replay options | SM MHz | span / steady TB/s | cta_unit us | lat us |
|---|---:|---|---:|---:|
| plain (1 consumer warp, graph launch) | 1980 | 4.42 / 4.82 | 0.988 | 2.74 |
| every trait at once: 8 polling warps, refill by warp 1 after the 256-thread barrier, constant policy word, evict_last stores, cooperative eager launch, ring behind 40 KB | 1979 | 4.37 / 4.84 | 0.982 | 2.72 |
| prologue stand-in (64 cold ints, 6 barriers, 6 us + up to 2.8 us stagger before the first issue) | 1979 | 3.64 / 4.79 | 0.916 | 2.61 |
| both | 1979 | 3.60 / 4.82 | 0.926 | 2.62 |
| 8 polling consumer warps, no refill thread | 1979 | 4.26 / 4.54 | 1.032 | 2.72 |
| 8 burn warps (mma.sync + integer, power load) | 1766 | 4.25 / 4.81 | 1.002 | 2.54 |

### Reading

1. There is no producer-path cap. With the math removed and everything else of helper0 kept (its launch, prologue,
   routing, ring, barriers, refill thread, helper warp, grid barrier, final pass), phase A streams in place at 4.66
   to 4.96 TB/s (tbsA) and 1.02 to 1.09 us per unit per CTA, the same as the replay of the same bytes in the same
   process (0.98 to 0.99 us, 4.8 steady). Removing the route, the x prefetch, the LUT and the ss zeroing changes
   nothing. Adding every remaining helper0 trait and a prologue stand-in to the replay changes nothing. The
   issue-to-ready latency is 2.7 to 2.8 us in every build, with or without math.
2. The previous leg's no-math and bare numbers (1.15 to 1.22 us per unit, 3.72 to 4.05 TB/s, latency 3.4 to 3.5 us;
   `nm0.log`, `s_bare.log`, `t_bare.log`, 08:20-10:10 UTC) are not reproduced with the same defines, the same tool
   and the same inputs: today the same builds give 1.02 to 1.09 us and 4.6 to 4.96, in two runs each. Today's runs
   were made with no other process on the GPU. The only code difference between the two legs' bare builds is two
   clock64 stores per CTA. Hypothesis for the earlier numbers: their issue-to-ready latency was 3.4 to 3.5 us
   against 2.7 for the replay in the same session and 2.7 to 2.8 in every build today, which is what a memory
   system carrying load beyond the kernel's own looks like; the h-loop census was running outside the lock in that
   window. Not verifiable after the fact; the earlier in-place rows should not be cited.
3. What limits phase A in the base build is the unit loop: 1.22 to 1.26 us per unit with warp 0 waiting for weights
   only 1.6 to 3 us per CTA over a 28 to 47 us phase (the ring is never empty after the ramp), at an SM clock of
   1740 to 1780 MHz, against 1966 to 1980 MHz in every build without the math and in the replay. The math's power
   sets the clock (a power load in the replay, 8 burn warps, drops it to 1766 MHz too), and at 1765 MHz the base
   loop's 1.23 us is 2170 cycles per unit. The stream itself is not clock-bound: the burn replay at 1766 MHz still
   reads 4.81 TB/s steady, so a faster loop would be fed at up to 4.8 (1.02 us per unit at 132 CTAs).
4. Size of the lever (hypothesis, corrected from the previous leg's 5 to 8 us): phase A at the stream's 1.02 us per
   unit instead of the loop's 1.23 is 17% of phase A, 4 to 8 us per verify 1 cell, and phase B in proportion; it is
   a loop lever (h-loop and moe lanes), not a stream lever. Each unit's loop has about half of its 2170 cycles in
   the per-unit CTA barrier, hand-off and refill rather than in the decode and mma (lane 1.3b's timeline), which is
   what a second CTA on the SM can fill: the one stream-side change that acts on it is occupancy.

### Candidate: ly_dual (two CTAs per SM, 2-slot ring)

`tools/dl_ly_dual_patch.py` (base ly_fabq, pinned) -> `muse_decode_layer/cand/ly_dual.py`: `__launch_bounds__(NTL,
OKC_MINB)`, the launcher multiplies the grid by OKC_MINB and its residency checks by the same factor, and the
TINY / TINY2 / MID builds get `-DOKC_NS=2 -DOKC_MINB=2` (two CTAs of 36.7 KB tables + 73.7 KB ring fit the SM's
228 KB; 112 registers per thread at 2 x 288 threads against the single-CTA builds' 141 (TINY2) and 162 (MID)).
Every unit range, the dynamic pool, the grid barrier and the final pass already index by gridDim.x; no fence,
barrier or arithmetic order changes, so the output is bitwise the base's.

First A/B (`.build/prtl/dual`, 10:31 UTC) was void: the launcher's residency check failed and prepare() fell back
to the Marlin adapter in both candidate arms (+50 to +90 us; `FORK_MOE_ROUTE Marlin: FORK_MOE_RESOURCE cooperative
residency` in chk.log), because `okc_smem_bytes()` sizes by the larger of the skinny and fat rings and the fat ring
(3 x 36 KB + 66 KB of x stages) kept the footprint at 178 KB. Commit 237d7a97 sizes the tiny builds by the skinny
ring alone (their fat path traps); the rebuilt candidate takes the native path (`FORK_MOE_ROUTE native repaired`)
and passes the reference check at all 12 cells (chk.log).

Paired A/B (`tools/ly_prtl_ab.sh`, CTL ly_fabq, CAND ly_dual, five bx_box_run.py processes c/w/c/w/c under the
lock, --runs 60, PDL pinned at T = 1, 8, 16, 32, 11:12-11:13 UTC; logs `muse_decode_layer/profiles/box/st/dual/`,
table `tools/ly_prtl_table.py`). Threshold max(2 x control spread, 0.5 us) on decode 1 / verify 1, max(2 x spread,
1%) on decode 16. The other three loads use the big build, which the patch does not touch.

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | threshold | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.25 | 73.68 | 65.31 | 73.94 | 65.44 | 0.19 | +8.47 | 0.50 | loss |
| decode 1 full | 72.02 | 79.68 | 71.81 | 79.65 | 71.42 | 0.59 | +7.91 | 1.18 | loss |
| decode 16 sliding | 126.56 | 137.01 | 127.26 | 136.62 | 127.06 | 0.70 | +9.86 | 1.41 | loss |
| decode 16 full | 199.09 | 206.82 | 198.58 | 207.12 | 199.47 | 0.90 | +7.92 | 1.99 | loss |
| verify 1 sliding | 103.10 | 108.45 | 102.80 | 108.24 | 102.77 | 0.34 | +5.45 | 0.67 | loss |
| verify 1 full | 110.61 | 116.83 | 110.11 | 117.47 | 110.11 | 0.50 | +6.87 | 0.99 | loss |

Why it loses (record build of ly_dual, `muse_decode_layer/cand/ly_st_dualrec.py` from `dl_ly_st_patch.py --base
ly_dual.py --nc 264`, `box_st_rec.py --minb 2 --nc 264`, `x_dual2.log`, against the base's `x_base16.log` and
`v_base.log`): the overlap did not show. Per CTA the unit interval doubles (2.30 to 2.58 us at two CTAs per SM
against 1.22 to 1.26 at one), so per SM the phase A throughput gains 4 to 6% (tbsA 3.85 to 4.02 at decode 16 and
4.14 to 4.25 at verify 1 against 3.96 to 4.02), phase A is the same length (decode 16 full 34.1 to 34.3 us against
33.3 to 33.4; sliding 37.6 against 37.9), phase B is 1 to 2 us longer (21.0 against 19.1 us at decode 16 sliding;
latency p90 5.4 us, the 2-slot ring exposes it), the SM clock is the same 1739 to 1772 MHz, and the whole launch is
7 to 10 us longer (79.8 against 72.0 us, 74.1 against 64.0 at decode 16; 64.6 against 55.9 at verify 1) because
every per-CTA cost (FAST1 pre-issue, routing, x prefetch, LUT, ranges, the grid barrier, the final pass's
per-CTA items) is paid 264 times and the 96-register builds spill (STACK 40 / 64 B). This agrees with the h-loop
lane's finding that the unit loop sits at an SMSP issue-throughput ceiling, the same at 4 and 6 consumer warps per
SMSP: a second CTA's warps compete for the same issue slots, so the per-unit barrier and refill stalls are not free
to fill.

Decision: ly_dual is a loss and is not a release candidate; no judge check. Nothing on the stream side is left to
change: the stream delivers at 4.8 TB/s whenever the loop consumes faster, the loop is issue-bound at the clock the
math sets, and more CTAs per SM do not add issue slots.

## 2026-11-04 - Full-context attention limit (box)

Attention lane, worktree branch `worktree-agent-af6a22d2b2de070a8`. Target: `attn_kernel<1, true, 0>` (the flat
q_len 1 producer-warpgroup build, 264 CTAs of 128 consumer + 128 producer threads, two per SM, three 16 KB stages)
at decode 32 full (3.9% of the score) and decode 16 full (2.5%) in ly_fabq (3765470032deb505). Box okc-h200, every
run under `/root/okc/.build/box/gpu.lock`; nvidia-smi showed no foreign process before the first run (0 MiB used).
Box numbers are box-candidate against box-control only (Box rule).

### Floor table (measurement)

Bytes: B x seq x 2 KV heads x 64 x 1 byte x (K + V) = 256 B per token, plus the page table (4 B per token at page
1). Lengths are uniform in ctx +/- 7.5%, so the expected total is B x 80,000 tokens (seed 1000 at batch 32 draws
2,539,033 rows = 650.0 MB of K + V). Achieved time is the kernel's duration under torch.profiler in the layer
(`tools/ly_tl_run.py`, 10 cold replays, PDL pinned on; `.build/box/attn/tl_fabq.log`, traces
`.build/box/attn/tl_fabq/cell{1,3}.json`).

| cell | K + V MB | + page table MB | floor at 4.45 TB/s | kernel in the layer (box) | achieved TB/s (K + V + table) | gap |
|---|---:|---:|---:|---:|---:|---:|
| decode 16 full | 327.7 | 5.1 | 74.8 us | 84.41 us | 3.94 | 9.6 us (11%) |
| decode 32 full | 655.4 | 10.2 | 149.6 us | 160.93 us | 4.14 | 11.3 us (7%) |

Layer timeline around the kernel (`tools/ly_tl_parse.py`, medians of 10 replays): decode 32 full qkv 0.00 to 10.83,
attention 9.38 to 170.34 (starts 1.5 us before qkv ends, PDL), the o projection's CTAs start at 160.51, that is
10.1 us before the attention ends (its PDL secondary occupies SMs the attention has already left); decode 16 full the
same with 1.9 and 6.7 us. Launch shape: grid (264, 1, 1), `REG:128 STACK:0 LOCAL:0` (cuobjdump of the box build,
`/root/.cache/bx_ext/muse_att_v298w1pf_90a-*`), 56 producer / 200 consumer registers after setmaxnreg, two CTAs per
SM, all 264 resident on 132 SMs (the stamps below saw every SM with exactly two).

### What any gather of this pattern reaches (measurement, `tools/attn_gather_bench.py`)

A kernel that only reads the same K and V rows (16-byte `ld.global.cg` per thread, DEPTH rows per thread in flight,
nothing else; cold L2 per run, 15 runs, batch 32 seed 1000, `.build/box/attn/gather32.log`):

| rows | CTAs x threads x depth | median us | TB/s |
|---|---|---:|---:|
| page table order (as the kernel reads them) | 264 x 256 x 16 | 184.5 | 3.52 |
| page table order | 528 x 256 x 8 | 165.9 | 3.92 |
| page table order | 1056 x 256 x 8 | 165.7 | 3.92 |
| sorted within each CTA's range (DRAM locality) | 528 x 256 x 8 | 158.5 | 4.10 |
| sequential rows 0..N | 528 x 256 x 8 | 158.6 | 4.10 |
| K only, page table order | 1056 x 256 x 16 | 91.4 | 3.56 |

Plain loads of random 128-byte rows top out at 3.92 TB/s whatever the depth or CTA count (the limit is not bytes in
flight: 64 KB to 1 MB per SM give the same rate at 528 CTAs or more); 128-byte rows in order reach 4.10. The
kernel's cp.async producer in the layer already reads the random rows at 4.04 to 4.14 TB/s over the whole launch, so
the gather mechanism is not the headroom.

### Per-CTA stamps in the layer (measurement, `tools/attn_stamps_make.py` + `tools/attn_stamps_run.py`)

Diagnostic build of ly_fabq with the kernel's OKC_DBGON / OKC_SEGDBG stamps compiled in (globaltimer stores, one per
CTA and event; the stock FPRO stamps made unconditional because their first-launch-only form costs a global load),
run through bx_box_run.py (5 replays, PDL pinned on, decode 16 and decode 32; the last replay's stamps,
`.build/box/attn/stamps{,2,3,4}_d{16,32}.log`). Times in us from the earliest CTA entry, medians over the 264 CTAs
(run 4; runs 1 to 3 agree within 0.5 us):

| event | decode 16 full | decode 32 full |
|---|---:|---:|
| CTA entry | 0.26 (max 1.06) | 1.18 (max 1.82) |
| partition barrier passed (seq_lens read, prefix sums) | 1.15 | 1.95 |
| segment list done (64-bit divisions, vmap, thread 0's scan) | 2.37 | 3.49 |
| setmaxnreg done, segment 0 starts | 2.46 / 2.75 | 3.62 / 3.84 |
| producer: page-index ring set up, 3 TMA fetches issued | 3.17 | 4.32 |
| producer: tile 0 copies issued | 3.55 | 4.86 |
| consumer: first tile consumed | 4.99 | 6.00 |
| loop end (last segment) | 77.38 | 154.74 |
| merge done (reds, release) | 78.46 (+1.09 after the loop) | 155.78 (+1.06) |
| per-SM end: p10 / median / max | 78.18 / 78.91 / 81.73 | 153.79 / 156.82 / 159.71 |
| pair end gap on one SM: median / max | 0.96 / 3.74 | 1.84 / 6.02 |
| finalize after the last merge | 0.7 | 0.7 |

With PDL off (`--pin-pdl8 ""`, decode 16) the prologue has the same shape: partition 1.02, segment list 2.21,
tile 0 issued 3.39, first tile 4.88 us. So the 2.3 us between the partition barrier and the first copy is code, not
SM sharing with the qkv kernel's tail: the dependent latencies in it are one seq_lens read and one page-index
fetch (the TMA page fetch returns in 0.27 us, an L2 hit), the rest is first-execution instruction fetch of a long
once-path (every step of the prologue costs 0.3 to 0.5 us: barrier setup, three TMA issues, the index unpack).

Steady state: first tile to loop end is 72.4 us at decode 16 (327.7 MB, 4.53 TB/s with the page table) and
148.7 us at decode 32 (4.47 TB/s), that is at the 4.45 TB/s floor. The whole 9.6 / 11.3 us gap is the intercept:

| piece | decode 16 full | decode 32 full |
|---|---:|---:|
| entry spread + prologue to the first consumed tile | 5.0 us | 6.0 us |
| loop end to merge (per CTA, only the last ones count) | 1.1 us | 1.1 us |
| end spread, median SM to the last SM | 2.8 us | 2.9 us |
| finalize of the last request | 0.7 us | 0.7 us |

Verdict: the cell is neither bandwidth-bound in steady state nor bound by the combine step; it is bound by the
startup latency (code-bound prologue, 3.4 to 4.9 us from entry to the first copy, then a 1.4 us DRAM round trip)
and the end spread. Compute is hidden (the 2026-11-03 standalone probes put it at 2.6 / 3.5 us; the per-CTA rate
here is at the floor). The largest single measured piece is the startup, 5 to 6 us per cell, of which about 1 us is
unavoidable dependent latency.

### Candidate: ly_apre (first STAGES tiles issued right after the partition barrier)

`tools/dl_ly_apre_patch.py` (base ly_fabq, pinned) -> `muse_decode_layer/cand/ly_apre.py`
(SHA256 b8bd56f714bae9958698972df291f2e70455bc1a4279dfb4654dcd159f55bde2). In `attn_kernel<1, true, 0>` only: once
the partition barrier has passed, the producer threads find their first segment (the scan thread 0 does for the
list), load its first min(3, tiles) tiles' page indices with plain loads and issue exactly the loop's copies (same
cp.async instructions, zero-fill and swizzled stage addresses) with the same arrival on the full barriers; the
producer loop of segment 0 skips the empty wait, copies and arrival for those tiles and keeps the page-ring
accounting. The stage barriers come from the stock EARLYBAR init before the partition barrier; no fence, barrier or
arrival count changes; consumers untouched. Reference check PASS at all 12 cells (`.build/prtl/apre/chk.log`,
`FORK_MOE_ROUTE native repaired`, `G_RELEASE_STARTUP_OK`).

Paired A/B (`tools/ly_prtl_ab.sh`, CTL ly_fabq, CAND ly_apre, five bx_box_run.py processes c/w/c/w/c under the lock,
`--runs 60`, PDL pinned at T = 1, 8, 16, 32, 12:16-12:17 UTC; logs `/root/okc/.build/prtl/apre/`, copied to
`muse_decode_layer/profiles/box/attn/apre/`; table `tools/ly_prtl_table.py`). Threshold max(2 x control spread,
0.5 us) on decode 1 / verify 1, max(2 x spread, 1%) elsewhere.

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | threshold | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.42 | 65.30 | 65.09 | 65.18 | 65.23 | 0.34 | -0.01 | 0.67 | inside |
| decode 1 full | 71.89 | 71.86 | 71.81 | 71.97 | 71.90 | 0.10 | +0.05 | 0.50 | inside |
| decode 16 sliding | 126.43 | 127.06 | 126.56 | 127.30 | 126.93 | 0.50 | +0.54 | 1.27 | inside |
| decode 16 full | 199.31 | 200.48 | 198.51 | 200.02 | 198.80 | 0.80 | +1.37 | 1.99 | inside, every pair slower (+1.2 to +2.0) |
| decode 32 sliding | 144.24 | 143.79 | 143.68 | 143.84 | 143.87 | 0.56 | -0.11 | 1.44 | inside |
| decode 32 full | 295.18 | 296.35 | 294.56 | 295.47 | 293.97 | 1.22 | +1.34 | 2.95 | inside, every pair slower (+0.9 to +1.8) |
| verify 1 sliding | 102.30 | 101.44 | 101.63 | 101.87 | 101.44 | 0.86 | -0.14 | 1.73 | inside |
| verify 1 full | 110.53 | 110.75 | 109.70 | 110.18 | 110.21 | 0.83 | +0.32 | 1.66 | inside |
| verify 16 sliding | 243.28 | 243.79 | 244.05 | 242.22 | 242.83 | 1.22 | -0.38 | 2.43 | inside |
| verify 16 full | 327.39 | 326.30 | 327.20 | 326.83 | 326.29 | 1.10 | -0.39 | 3.27 | inside |
| verify 32 sliding | 297.63 | 295.65 | 296.86 | 296.78 | 296.88 | 0.77 | -0.91 | 2.97 | inside |
| verify 32 full | 460.78 | 460.85 | 459.82 | 459.70 | 460.32 | 0.96 | -0.04 | 4.60 | inside |

The two target cells are slower in all four candidate-against-neighbour pairs, by about 1.3 us; the other ten cells
(untouched kernels) sit inside the control spread.

Why it loses (stamps build of ly_apre, `.build/attn/ly_apre_stamps.py` from `tools/attn_stamps_make.py --base
ly_apre.py`, `muse_decode_layer/profiles/box/attn/stamps/stamps_apre_d{16,32}.log`; medians, us from the earliest
entry, decode 16 / decode 32 full, base in brackets): partition barrier 1.42 / 2.05 [1.15 / 1.95], segment list done
3.95 / 4.86 [2.37 / 3.49], setmaxnreg done 5.49 / 6.21 [2.46 / 3.62], segment 0 start 5.66 / 6.42 [2.75 / 3.84],
first tile consumed 6.85 / 7.55 [4.99 / 6.00]. The pre-issue block itself takes about 4 us on the producer warps
(a seq_lens read and the page-index loads as two dependent round trips, three tiles of copies, all first-execution
code), and since the producers reach `setmaxnreg.dec` only after it, the consumers' `setmaxnreg.inc` holds the whole
consumer side for those 3 us; the copies are in flight earlier, but nobody can consume them until 5.5 us, so the
first tile lands 1.6 to 1.9 us later than in the base. Steady state and the end spread are unchanged
(first tile to merge 72.7 / 146.8 us [73.2 / 147.9]; per-SM end median to max 3.5 / 5.0 us [2.8 / 2.9]).

### Candidate: ly_apre2 (producers release their registers first, no seq_lens read)

`tools/dl_ly_apre_patch.py` without `--v1` -> `muse_decode_layer/cand/ly_apre2.py`
(SHA256 52fed0e84e0d790a82481156df8034c7a5512794d10ef0a3052f92810c531b74). Same mechanism, two changes from the
stamps above: the producer warpgroup executes its `setmaxnreg.dec` at the start of the pre-issue block (the one in
the role split is skipped under `apre`), so the consumers' `setmaxnreg.inc` no longer waits for the pre-issue; and
the first request's length comes from the prefix sums (`S.cum[b0 + 1] - S.cum[b0] - L`, exact on the full layer,
the flat kernel's only use; a window launch gets `npre = 0`), which removes one dependent round trip.

Reference check PASS at all 12 cells (`muse_decode_layer/profiles/box/attn/apre2/chk.log`). Build: `REG:128 STACK:0
LOCAL:0`, two CTAs per SM, as the base (cuobjdump of `/root/.cache/bx_ext/muse_att_v298w1pf_90a-536017ad614e3377`;
the ly_apre build is `STACK:8`).

Paired A/B (same recipe, CTL ly_fabq, CAND ly_apre2, 12:25-12:26 UTC; logs
`muse_decode_layer/profiles/box/attn/apre2/`):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | threshold | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.41 | 65.26 | 65.20 | 65.20 | 65.23 | 0.21 | -0.05 | 0.50 | inside |
| decode 1 full | 71.84 | 71.58 | 71.90 | 71.87 | 71.97 | 0.13 | -0.18 | 0.50 | inside |
| decode 16 sliding | 127.10 | 126.99 | 126.72 | 126.35 | 127.06 | 0.38 | -0.29 | 1.27 | inside |
| decode 16 full | 200.21 | 201.25 | 199.06 | 201.49 | 199.41 | 1.15 | +1.81 | 2.30 | inside, every pair slower (+1.0 to +2.4) |
| decode 32 sliding | 143.63 | 143.74 | 143.57 | 143.58 | 143.89 | 0.32 | -0.03 | 1.44 | inside |
| decode 32 full | 295.55 | 295.44 | 295.15 | 295.33 | 294.85 | 0.70 | +0.20 | 2.95 | inside (pairs -0.1 to +0.5) |
| verify 1 sliding | 102.00 | 101.79 | 102.29 | 101.84 | 102.46 | 0.46 | -0.43 | 0.93 | inside |
| verify 1 full | 110.37 | 109.81 | 110.16 | 110.38 | 110.98 | 0.82 | -0.41 | 1.63 | inside |
| verify 16 sliding | 241.78 | 243.22 | 242.96 | 242.02 | 243.30 | 1.52 | -0.06 | 3.04 | inside |
| verify 16 full | 327.18 | 326.54 | 327.07 | 326.94 | 327.06 | 0.13 | -0.36 | 3.27 | inside |
| verify 32 sliding | 296.13 | 295.57 | 295.86 | 296.74 | 296.86 | 1.01 | -0.13 | 2.96 | inside |
| verify 32 full | 461.15 | 461.49 | 460.94 | 460.40 | 459.92 | 1.23 | +0.27 | 4.61 | inside |

Decode 32 full is neutral; decode 16 full is slower in all four pairs. Neither candidate reaches a win at the two
target cells, and both lose at decode 16 full.

ly_apre2 stamps (same diagnostic build, 5 runs, logs `muse_decode_layer/profiles/box/attn/stamps/stamps_apre2_d{16,32}.log`),
medians in us from the earliest entry, decode 16 / decode 32: producer after partition sync 1.39 / 1.63 (base 1.15
/ 1.95), segment list done 4.02 / 4.72 (base 2.37 / 3.49), consumer after setmaxnreg.inc 4.13 / 4.94 (base 2.46 /
3.62), first tile consumed 6.16 / 6.59 (base 4.99 / 6.00), loop end 80.74 / 155.57 (base 77.38 / 154.74). The
pre-issue block runs on the producer warps before the segment list and costs about 1.6 us at decode 16 (0.9 us at
decode 32) of first-execution time, so the first copies are issued earlier but the first tile is consumed 1.2 /
0.6 us later than in the base. That is the measured mechanism of the decode 16 full loss.

### Decision

The full-context q_len 1 attention cells are at the DRAM floor in steady state (4.47 to 4.53 TB/s per CTA from the
first tile to the loop end, against a 4.45 TB/s floor and a 3.92 TB/s plain-load ceiling for the random rows). The
11% / 7% distance to the byte floor is the launch intercept: 5 to 6 us from CTA entry to the first consumed tile
(a code-bound once-path of about 2.5 us before the first copy, then a 1.4 us DRAM round trip; the same with PDL
off, so not SM sharing) and about 3 us of end spread across SMs plus the last merge and finalize. Pre-issuing the
first tiles (ly_apre, ly_apre2) does not convert the earlier copies into an earlier start: the block's own
first-execution cost on the producer warps is as long as what it skips, and at decode 16 full both forms lose
about 1.4 to 1.8 us. No judge check is recommended for either file. The remaining levers are (a) a shorter once-path
before the first copy (the instruction stream, not the data dependencies, sets the 2.5 us) and (b) the end
spread, where every static or arrival-order pair split has lost before (docs/ruled-out.md). Each is worth at most
about 3 us per cell on these measurements, which is the 1% keep threshold itself.
## 2026-11-04 - Verify 16/32 MoE limit (box)

Question from the lead (moe lane): what limits `okc::moe_fused<true>` at verify 16 and verify 32 (sliding and full) in
the layer, and which single mechanism has the largest measured headroom there. Base `muse_decode_layer/cand/ly_fabq.py`
(SHA256 3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555, never edited). Box only (okc-h200, 2026-11-04
11:40 to 12:30 UTC, every GPU run under /root/okc/.build/box/gpu.lock, `nvidia-smi` showed no foreign process before
the runs), no judge job, no routing slack. Logs: `/root/okc/.build/box/v16/` (`tl/` the per-kernel timeline,
`tlv16g.log` / `.json` the in-kernel timeline, `sass/` the SASS report, `wgmma/` the microbenchmark). Tools:
`tools/ly_moe_tlv16.py` (writes the stamped copy `muse_decode_layer/cand/ly_fabq_tlv16.py` from the pinned base) and
`tools/ly_box_moe_tlv16g.py` (graph runner and the per-input statistics).

### Which form runs at these T (source reading, confirmed by the box SASS)

At M = 128 and 256 `_fork_native_moe` selects the MAIN build (`MOE_DEFINES`: `-DOKC_NW=16 -DOKC_WGS=1`, plus SC6W and
PRMTSC from the weights): 16 consumer warps in 4 warpgroups plus a weights-producer warp and an x-producer warp (544
threads, 96 registers, 0 spill), register-A `wgmma.mma_async.m64nNk16.bf16` with N = 8 x NTT, NTT = ceil(tokens of the
job / 8) up to 8 (jobs of up to 64 tokens; an expert with more tokens gets several jobs and its weights are streamed
once per job). One stage is 256 rows x 256 K (35,840 B with SC6 scales): per warpgroup 16 k16 steps, 2 per commit group
with `wgmma.wait_group 1`, the DQ7 unpack and scale multiply in registers before each step. So the wgmma form the brief
asked to try first is what the layer already runs at T >= 16; the mma.sync loop (helper0) runs only at T <= 16 (TINY /
TINY2 / MID), and decode 32 runs this same WGS source as B4.

SASS of the MAIN build (`tools/ly_micro_sass.py`, nvcc 13.0 on the box CPU; `.build/box/v16/sass/report.json`): the 16
stage loops (8 NTT variants x 2 phases) are 304 to 308 instructions per warp per stage with 16 HGMMA each, 19.0
instructions per wgmma at every N: LOP3 85, HMUL2 64, IMAD 39 to 44, SHF 36, UIADD3 17, WARPGROUP (arrive / depbar) 17,
HGMMA 16, LDS 6, SYNCS 4, PRMT 4, BRA 3, MEMBAR 2 + FENCE 2 (the two `fence.proxy.async.shared::cta`), the rest
uniform-datapath address arithmetic. Per SMSP (one warp of each warpgroup) that is 1,216 issue slots per stage.

### Floors per stage and per cell (measurements unless marked)

Box read bandwidth this session 4.424 TB/s (`sum` over 4 GiB): one stage on every CTA at once is 132 x 35,840 B /
4.424 TB/s = 1.069 us. Tensor floor from the RS-wgmma microbenchmark (`.build/box/v16/wgmma/wgmma_rs.cu`, grid 132,
the kernel's commit-group pattern, 4 warpgroups per SM, no decode, no C7510/C7512/C7513, 46 to 80 registers; median of 3
runs): per stage (16 wgmma per warpgroup, 4 warpgroups) N <= 24: 832 cycles per warpgroup, a 13-cycle per-SM issue
floor whatever N (31% of peak at N = 8); N >= 32: N x 32 cycles, the full 4,096 BF16 FLOP per cycle per SM. At the
measured clock that is 0.46 us (N <= 24), 0.57 (32), 0.73 (40), 0.89 (48), 1.03 (56), 1.17 us (64). Removing the
per-group `wait_group 1` or using 4-step groups changes nothing at 4 warpgroups. Issue floor (hypothesis from the SASS
count): 1,216 slots per SMSP per stage, 0.69 us at 1 IPC and 1,760 MHz, 0.98 us at 0.7 IPC.

Achieved per-stage time in the layer, from the in-kernel timeline (below): least squares of each CTA's phase time on its
stage counts per NTT class over 20 inputs x 132 CTAs (residual sd 1.9 to 2.7 us per CTA), and the steady issue gap
(median interval between the weights producer's issues when the released stage is the third or later of its segment):

| NTT (N) | fit A us per stage, v16 s / v32 s | fit B | steady gap A v16 s / v32 s | steady gap B | byte floor | tensor floor |
|---|---|---|---|---|---:|---:|
| 1 (8) | 0.995 / 0.981 | 0.931 / 0.844 | 0.96 / 0.96 | 0.99 / 0.96 | 1.069 | 0.46 |
| 2 (16) | 1.117 / 1.093 | 1.052 / 0.961 | 1.02 / 1.06 | 1.06 / 1.06 | 1.069 | 0.46 |
| 3 (24) | 1.156 / 1.142 | 1.143 / 1.045 | 1.09 / 1.09 | 1.12 / 1.12 | 1.069 | 0.46 |
| 4 (32) | 1.302 / 1.264 | 1.245 / 1.122 | 1.15 / 1.15 | 1.22 / 1.22 | 1.069 | 0.57 |
| 5 (40) | 1.441 / 1.370 | 1.353 / 1.204 | 1.22 / 1.25 | 1.28 / 1.28 | 1.069 | 0.73 |
| 6 (48) | - / 1.470 | - / 1.320 | - / 1.31 | - / 1.34 | 1.069 | 0.89 |
| 7 (56) | 1.661 / 1.589 | 1.540 / 1.441 | 1.41 / 1.41 | 1.44 / 1.44 | 1.069 | 1.03 |
| 8 (64) | 1.767 / 1.697 | 1.706 / 1.563 | 1.47 / 1.50 | 1.54 / 1.54 | 1.069 | 1.17 |

Stages by NTT class, phase A, summed over the 20 inputs: verify 16 sliding 132,096 / 33,280 / 12,288 / 8,192 / 512 / 0
/ 2,048 / 7,168 (NTT 1..8: 67% of the stages are 8-token jobs, 5% are wider than 32 tokens); verify 32 sliding 115,712 /
32,256 / 14,848 / 17,408 / 10,240 / 6,144 / 7,680 / 16,384 (52% NTT 1, 18% wider than 32 tokens).

Per cell (medians over 20 fresh inputs per cell, f931 case seeds, router seed 106218864389449120; the instrumented graph
is within 1.5 us of the plain graph at verify 16 and verify 32 sliding, +8.6 us at verify 32 full):

| cell | moe_fused in the graph (ly_tl_run) | span (stamps) | distinct / routes / jobs | re-streamed experts | byte floor | tensor floor (hyp.) | ramp | phase A (floor) | phase B (floor) | MFIN wait + final | A-end / B-end spread | stream A (agg.) | SM MHz |
|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---:|---:|
| verify 16 sliding | 141.9 | 145.6 | 18.5 / 182 / 18.5 | 0 | 115.1 | 36 | 5.5 | 86.0 (76.7) | 43.1 (38.4) | 6.6 + 4.3 | 12.8 / 10.4 | 3.98 TB/s | 1781 |
| verify 16 full | 140.3 | 141.2 | 18 / 175 / 18 | 0 | 112.0 | 35 | 5.4 | 83.3 (74.7) | 41.3 (37.3) | 6.2 + 4.1 | 12.4 / 9.4 | 3.97 | 1798 |
| verify 32 sliding | 177.7 | 178.7 | 20 / 354 / 21 | 1 (118 tokens) | 130.7 | 62 | 7.4 | 103.1 (87.1) | 52.8 (43.6) | 7.2 + 6.1 | 15.8 / 12.3 | 3.76 | 1749 |
| verify 32 full | 177.8 | 177.7 | 20 / 358 / 21.5 | 1 (116 tokens) | 133.8 | 63 | 7.5 | 103.4 (89.2) | 53.2 (44.6) | 6.3 + 6.0 | 16.3 / 11.9 | 3.81 | 1772 |

The byte floor counts the stages actually streamed (768 per job, so the re-streamed expert at verify 32 is inside it;
without it 124.4 / 127.5 us) at 4.424 TB/s. The tensor floor column is the microbenchmark's per-stage cost summed over
the stage mix and divided by 132 (a hypothesis for the kernel, whose stages also decode). The ramp is routing 2.1 / 2.7
us, the x gather 2.9 / 4.5 us (the weights producer waits for it at WS_JOIN) and 0.3 us to the first TMA issue. The
judge's profile of the same file shows 143.2 / 169.7 / 203.5 / 191.6 us for one input per cell; the box medians above
sit inside the per-input range that the distinct-expert count explains (span = 11 + 7.3 x distinct at verify 16, 12 +
8.3 x distinct at verify 32, from the per-distinct rows in `tlv16g.log`).

### The limit, named

1. The kernel is byte-bound in its narrow stages and tensor-bound in its wide ones, and neither is the main excess.
   NTT = 1 to 3 stages run at 0.96 to 1.09 us steady, the byte rate (1.069); NTT = 8 stages run at 1.47 to 1.54 us,
   1.26x the microbenchmark's 1.17 us tensor floor at N = 64 (the decode's 1,216 issue slots and the smem B-operand
   reads compete with the wgmma issue: hypothesis, no counters on the pod). Wide stages are 5% of verify 16 and 18% of
   verify 32; their excess over the byte floor is about 1.5 us per verify 16 cell and 7 us per verify 32 cell.
2. Segment switches are the largest excess (measured). Every CTA's range is 5 to 7 segments in phase A and 3 to 4 in
   phase B (a segment = one 256-row unit's stages on this CTA). The weights producer's issue gap after a segment's first
   stage is released is 2.1 to 2.3 us at NTT = 1 against 0.96 steady, rising to 4.1 to 5.5 us at NTT = 8 (about 1.1 us
   + 0.25 us per n-tile), and the second stage's gap is 0.03 to 0.2 us over steady. Summed per CTA
   (`.build/box/v16/switch_cost.py` on the raw stamps of 3 inputs per cell): phase A 10.1 to 11.0 us at verify 16 and
   11.9 to 12.5 us at verify 32, phase B 4.1 to 5.1 and 5.0 to 6.1 us, so 15 to 18 us per cell, which is the whole 9 to
   16 us phase A excess over the byte floor plus most of phase B's. While the four warpgroups run a unit's epilogue
   (SwiGLU + bf16 stores in phase A, y stores in phase B; its issue count grows with NTT) and then the deferred
   `wgs_flush` (fence.proxy.async.global, `bar.sync` of the warpgroup, `red.release.gpu`: 0.25 + 0.66 to 0.69 us
   measured by the B4 lane on the same code, and for a cut unit an `atom.acq_rel.gpu` round trip and a read-back), no
   stage is consumed, the 3-deep ring fills, and this CTA's stream stops; the loop then catches up only at its small
   margin over the byte rate (0.96 against 1.069 us at NTT = 1), so the stall is exposed. The aggregate phase A stream of
   3.76 to 3.98 TB/s (85 to 90% of the box rate) is this stall's mirror image.
3. The tail is the second piece: after its own phase B end the mean CTA waits 6.2 to 7.2 us before its first final item
   (a token's experts complete when the slowest CTA holding their units finishes; B-end spread 9.4 to 12.3 us, sd 1.5
   to 2.8 us per CTA), then the final pass takes 4.1 to 6.1 us (1 to 2 tokens per CTA, each a dependent L2 round trip
   over the routes' y rows, two CTA barriers, a re-read). 10 to 13 us per cell.
4. The ramp is 5.4 to 7.5 us (routing, then the x gather on the weights producer's critical path); ly_b4_wpre
   (2026-11-03) already measured that issuing the weights before the gather gains nothing at verify 16 / 32.
5. Ruled out as the limit: the stream (4.42 TB/s delivered whenever stages are consumed; the narrow steady stretches run
   at 0.96 us per stage), the clock (1,750 to 1,800 MHz under the math, as at T <= 8), registers (96, no spill), the
   decode instruction count per wgmma (19, against 15.7 per mma.sync in the T <= 8 loop; amortized over N, it binds
   only through the wide stages).

Candidate chosen (the single mechanism with the largest measured headroom): take the GPU-scope release of each
whole-unit publish (`bar.sync` + `red.release.gpu`, 0.9 us of consumer time per segment, 6 to 7 whole-unit publishes
per CTA and cell) off the consumer warpgroups and onto a publisher warp through an mbarrier relay, keeping every fence
(the writers' `fence.proxy.async.global` stays on the writers), so the consumers go straight from the epilogue into the
next segment's stages. Expected 4 to 6 us per cell (hypothesis: 0.9 us x the whole-unit publishes per CTA, if the stall
is fully exposed as the fci6 experiment indicated); the epilogue's own issue time (the NTT-dependent part) stays. Build
and A/B below.

### Candidate ly_pub: whole-unit releases through a publisher warp (box, loss)

`tools/dl_ly_pub_patch.py` writes `muse_decode_layer/cand/ly_pub.py` (SHA256
368bec058526ec18c978ba1cb7094bdafdd0245f30773a557cd5e822a408aa0e) from the pinned base. The MAIN build alone gets
`-DOKC_PUB=1` (every other build compiles the base's text): a fifth non-consumer warp (NTL 544 to 576) is the
publisher. For the flush kinds that only release (kind 1, phase A whole unit to doneAs; kind 4, phase B whole unit to
the expert's MFIN count; kind 2, phase B cut share to its fixup count; the finisher's trailing doneAs / MFIN release)
every consumer thread keeps its own `fence.proxy.async.global` and stores, waits (warp-uniform `mbarrier.try_wait`) on
the acknowledge mbarrier of its warpgroup's previous publish, and arrives (release.cta) on the warpgroup's request
mbarrier (count 128) instead of `bar.sync`; thread 0 of the warpgroup writes the (address, value) mailbox before its
arrive. The publisher polls the four request barriers (`mbarrier.test_wait`, acquire.cta, `__nanosleep(64)` between
empty polls), performs the `red.release.gpu` and arrives on the acknowledge barrier; after phase B each warpgroup sends
a null sentinel and the publisher exits. Causality order through the cta-scope arrive / test_wait pair and then the
gpu-scope release is the chain the base already relies on through `bar.sync` + one thread's release (PTX ISA memory
model: release patterns are cumulative over what precedes them in causality order). Kind 3 (the finisher's count wait,
`bar.sync`, read-back) and the phase A cut path (`atom.acq_rel`) are unchanged; no fence or barrier removed; the
consumers' stage loop is the base's text. The publish count is a register threaded through wgs_phase / wgs_seg /
wgs_flush like `seq`.

Build facts (box, `tools/ly_micro_sass.py` main build, `.build/box/v16/pub_sass2/`): 96 registers, 0 spill, 8 B stack
(base 0), stage loops 304 / 307 / 308 instructions with 16 HGMMA (one 310), the same opcode mix as the base; the phase A
outer loop drops from 24 to 16 BAR, 96 to 64 MEMBAR and 52 to 36 REDG, and gains 32 SYNCS (the mbarrier ops) and 32
STS (the mailbox). A first build whose acknowledge wait sat inside `if (wt == 0)` (divergent spin, count in shared
memory) compiled to 388-instruction stage loops (the uniform-datapath address arithmetic fell back to IMAD / R2UR, 104 B
stack, an LDL in the loop) and crashed the prepare probe with `unspecified launch failure`; it was replaced by the
uniform wait before any measurement and its cause was not isolated further. Reference check (`.build/prtl/pub/chk.log`,
`muse_decode_layer/profiles/box/pub/chk.log`): `FORK_MOE_ROUTE native repaired`, `G_RELEASE_STARTUP_OK native (128, 32,
33, 256, 1, 8, 16) 56 verified_replays`, check PASS at all 12 cells. The bisect variant `ly_pubx.py`
(ec01371016798e63d4d51b31b1d45a36ad0a9315bb70928b25014d801cb9a9c6, publisher warp present, base release path, relay
for the sentinel only) also passes 12 / 12.

Paired A/B (`tools/ly_prtl_ab.sh`, c/w/c/w/c, `--runs 60`, PDL pinned at 1, 8, 16, 32 in both arms, all six loads,
12:46 to 12:48 UTC, no foreign GPU process; logs `/root/okc/.build/prtl/pub/`, copied to
`muse_decode_layer/profiles/box/pub/`; table `tools/ly_prtl_table.py`, threshold max(2 x control spread, 0.5 us short
cells, 1% long cells)):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | w_a-c_a | w_a-c_b | w_b-c_b | w_b-c_c | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.25 | 64.80 | 64.70 | 65.01 | 65.22 | 0.54 | -0.15 | -0.45 | +0.10 | +0.30 | -0.21 | 1.09 | inside |
| decode 1 full | 71.71 | 71.57 | 71.92 | 71.46 | 72.14 | 0.43 | -0.41 | -0.14 | -0.35 | -0.46 | -0.69 | 0.86 | inside |
| decode 16 sliding | 127.20 | 126.83 | 126.64 | 126.64 | 126.98 | 0.56 | -0.20 | -0.37 | +0.19 | +0.00 | -0.34 | 1.27 | inside |
| decode 16 full | 199.73 | 199.87 | 200.91 | 199.76 | 199.78 | 1.18 | -0.32 | +0.14 | -1.04 | -1.15 | -0.02 | 2.37 | inside |
| decode 32 sliding | 143.57 | 144.26 | 143.52 | 144.67 | 144.05 | 0.53 | +0.75 | +0.69 | +0.74 | +1.15 | +0.62 | 1.44 | inside |
| decode 32 full | 295.44 | 295.15 | 294.14 | 294.80 | 295.42 | 1.30 | -0.03 | -0.29 | +1.01 | +0.66 | -0.62 | 2.95 | inside |
| verify 1 sliding | 102.14 | 102.27 | 102.35 | 103.06 | 101.87 | 0.48 | +0.54 | +0.13 | -0.08 | +0.70 | +1.18 | 0.96 | inside |
| verify 1 full | 110.70 | 110.74 | 110.88 | 110.66 | 110.80 | 0.18 | -0.10 | +0.03 | -0.14 | -0.22 | -0.14 | 0.50 | inside |
| verify 16 sliding | 243.31 | 245.28 | 244.03 | 245.20 | 244.29 | 0.98 | +1.36 | +1.97 | +1.25 | +1.17 | +0.91 | 2.44 | inside |
| verify 16 full | 327.38 | 329.06 | 327.02 | 328.77 | 327.41 | 0.38 | +1.64 | +1.68 | +2.03 | +1.74 | +1.36 | 3.27 | inside |
| verify 32 sliding | 297.57 | 299.65 | 297.52 | 299.50 | 297.22 | 0.35 | +2.14 | +2.08 | +2.13 | +1.98 | +2.29 | 2.97 | inside |
| verify 32 full | 459.95 | 463.04 | 459.74 | 462.16 | 460.54 | 0.80 | +2.52 | +3.09 | +3.30 | +2.42 | +1.62 | 4.60 | inside |

The eight cells whose builds did not change sit inside the control spread. At the four target cells every one of the
four pair differences is positive: +1.4 / +1.6 / +2.1 / +2.5 us mean, a loss (inside the 1% long-cell threshold, but
with the same sign in all sixteen pair differences and the only cells whose build changed).

Where the time went (candidate timeline `ly_moe_tlv16.py --base ly_pub.py`, 20 inputs per cell, same inputs as the base's
run; `.build/box/v16/pub_tlv16g.log`, both logs in `muse_decode_layer/profiles/box/pub/tl/`; base -> candidate medians):

| cell | span | phase A | phase B | MFIN wait | final | x producer wait on doneAs (b_dep_wait) | phase A fit intercept | first-of-segment issue gap A at NTT 1 / 2 / 4 | first gap B at NTT 1 / 4 |
|---|---|---|---|---|---|---|---|---|---|
| verify 16 sliding | 145.6 -> 147.1 | 86.0 -> 84.2 | 43.1 -> 44.7 | 6.6 -> 7.5 | 4.3 -> 4.4 | 15.2 -> 16.1 | 7.7 -> 4.0 | 2.18 / 2.56 / 3.33 -> 1.70 / 2.43 / 3.17 | 1.82 / 2.82 -> 1.76 / 2.78 |
| verify 16 full | 141.2 -> 141.8 | 83.3 -> 81.1 | 41.3 -> 42.3 | 6.2 -> 7.8 | 4.1 -> 4.4 | 15.0 -> 15.2 | 7.2 -> 2.5 | 2.30 / 2.69 / 3.52 -> 1.82 / 2.56 / 3.49 | 1.86 / 2.91 -> 1.76 / 2.82 |
| verify 32 sliding | 178.7 -> 179.1 | 103.1 -> 102.0 | 52.8 -> 54.2 | 7.2 -> 7.1 | 6.1 -> 6.4 | 16.7 -> 18.0 | 10.0 -> 9.3 | 2.08 / 2.40 / 2.94 -> 1.63 / 1.95 / 2.82 | 1.79 / 2.72 -> 1.73 / 2.78 |
| verify 32 full | 177.7 -> 179.1 | 103.4 -> 102.4 | 53.2 -> 54.0 | 6.3 -> 6.9 | 6.0 -> 6.4 | 17.1 -> 18.0 | 4.4 -> 3.9 | 1.95 / 2.37 / 2.85 -> 1.60 / 1.95 / 2.69 | 1.76 / 2.69 -> 1.73 / 2.78 |

The steady issue gaps are unchanged at every NTT (0.96 at NTT 1, 1.47 to 1.50 at NTT 8, phase A; the publisher's polling
does not slow the stage loop), and the per-stage fit moves by at most 0.04 us. The mechanism did what it was built for in
phase A: the first-of-segment gap at NTT 1 fell by 0.35 to 0.48 us per switch (2.1 to 2.3 -> 1.6 to 1.8 us), the phase A
intercept fell, and phase A shortened by 1.0 to 2.2 us per cell. Phase B gave it back and more (+0.8 to +1.6 us): its
segments end in kind 3 finishers that wait for the other shares' kind 2 counts, and the x producer waits for the doneAs
counts of phase A (b_dep_wait +0.2 to +1.3 us); both now arrive through a polling warp instead of directly from the
warpgroup, so the release latency seen by the waiters grew by about the poll period, and phase B's first-of-segment gaps
did not move (1.8 -> 1.7 us at NTT 1). The MFIN wait before the final pass grew by 0.6 to 1.6 us at verify 16 for the same
reason. Net: the consumer-side cost of a release was real (about 0.45 us per phase A switch), but the kernel is also
latency-bound on those releases at its waiters, and the relay trades 0.45 us of issue time for more than that in waiting.

Decision: ly_pub is a loss at verify 16 / 32 and unchanged elsewhere; no judge check. Ruled-out row added. What remains
of the switch excess after this result: the epilogue's own issue time (NTT-dependent, 0.25 us per n-tile per segment),
which the relay does not touch, and the finisher / producer waits on cross-CTA counts, which any added latency makes
worse; a next step would have to shorten the release latency (for example `red.release.gpu` by thread 0 without the
`bar.sync`, which needs a cheaper intra-warpgroup ordering than a named barrier and is not available without the
mbarrier round trip measured here), or reduce the number of segments per CTA (fewer, longer ranges), which the
stream-K balance forbids at 132 CTAs.

## 2026-11-04 - Verify tail (box)

Question from the lead (moe lane, after "Verify 16/32 MoE limit"): the tail of `okc::moe_fused<true>` at verify 16 / 32
(the 6.2 to 7.2 us wait after each CTA's phase B end plus the 4.1 to 6.1 us final pass). Decompose the wait, pick one
mechanism, A/B it. Base `muse_decode_layer/cand/ly_fabq.py` (SHA256 3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555,
never edited). Box only (okc-h200, 14:20 to 14:50 UTC, every GPU run under `/root/okc/.build/box/gpu.lock`,
`nvidia-smi` showed no foreign process before the runs), no judge job, no routing slack. Logs `/root/okc/.build/box/v16/tail/`
(timelines) and `/root/okc/.build/prtl/fin1/`, `/root/okc/.build/prtl/bal/` (A/B), copied to
`muse_decode_layer/profiles/box/tail/`, `.../fin1/`, `.../bal/` (the 45 MB raw-stamp JSONs stay on the box). Tools:
`tools/ly_moe_tlv16f.py` (the previous lane's level-1 timeline plus final-pass stamps: pass 1 done, pass 2 done, the
second token's pass done; writes `ly_fabq_tlv16f.py`, 16e7f055), `tools/ly_box_moe_tlv16g.py --raw 20` (raw stamps of
every input), `muse_decode_layer/profiles/box/tail/tail_pass.py`, `tail_decomp.py`, `spread_fit.py` (the statistics below).

### The wait is a grid barrier, not a per-expert count

The WGS builds force the per-expert completion counters off: `#if OKC_WGS / #undef OKC_MFIN / #define OKC_MFIN 0` (the
MoE source, after the MFIN defaults), and prepare prints `FORK_MOE_ROUTE native repaired: cooperative+BF16, MFIN/DF/YBF/PDL
off`. The reason is on record (`docs/steering/2026-11-01-g-native-repair-inventory.md`, "MFIN forced 0"): the readiness
poll in the final pass is bounded (`n < 4000000`) and may fall through, so re-enabling the flag is not a repair. So at
verify 16 / 32 every CTA, after its phase B, runs the grid barrier (`atom.acq_rel.gpu` by thread 0, `ld.acquire` poll with
`__nanosleep(64)`, CTA barrier), zeroes its share of the per-launch flags, then runs the WGSFINSS final pass with no
per-token wait. The previous section's "MFIN wait" column is that barrier wait.

Measured (20 fresh inputs per cell, raw stamps of all 132 CTAs in every input, medians over inputs, us):

| cell | wait = fin - own B end | max(B end) - mean(B end) | fin - max(B end) | zeroing + pass 1 | pass 2 | first token total | second token (v32) | B-end spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| verify 16 sliding | 6.49 | 5.12 | 1.16 | 2.92 (per-CTA 2.72 to 3.14) | 1.69 (1.62 to 2.02) | 4.65 | - | 10.1 |
| verify 16 full | 7.06 | 5.97 | 1.10 | 2.88 | 1.71 | 4.67 | - | 10.4 |
| verify 32 sliding | 6.61 | 5.50 | 1.14 | 2.95 | 1.75 | 4.73 | 3.2 | 11.3 |
| verify 32 full | 6.48 | 5.23 | 1.09 | 2.98 | 1.68 | 4.64 | 3.2 | 12.0 |

Every CTA's `fin` (barrier passed) lies within 0.2 us of every other's in every input, and 0.96 to 1.44 us after the last
CTA's phase B end: the wait is end spread (5.1 to 6.0 us for the mean CTA) plus one barrier release latency (1.1 us), with
no count-wait component. The CTAs that end last (B end within 2 us of the last: 6 to 10 of 132, blockIdx 10 to 102) hold 24
to 30 phase B stages against the all-CTA median of 36 to 46: they carry the widest jobs (64-token, 8 n-tiles), which sit
first in the job list.

How much of the spread a static rebalance could remove (`spread_fit.py` on the previous lane's 20-input JSON: each CTA's
phase A + B time regressed on its stage counts per n-tile class with a per-input intercept):

| cell | per-CTA end sd | part explained by the stage mix (sd) | residual sd | max - mean, total | max - mean, residual alone | residual correlation between inputs by blockIdx |
|---|---:|---:|---:|---:|---:|---:|
| verify 16 sliding | 1.95 | 1.10 | 1.66 | 5.73 | 4.50 | 0.05 |
| verify 16 full | 1.87 | 1.10 | 1.52 | 5.21 | 4.30 | 0.06 |
| verify 32 sliding | 2.39 | 1.26 | 2.04 | 6.05 | 4.87 | -0.01 |
| verify 32 full | 2.37 | 1.10 | 2.18 | 5.64 | 4.97 | -0.01 |

Within an input a CTA's phase A and phase B times are anti-correlated (-0.75 to -0.79): a CTA that ends phase A late
finds its phase B inputs ready and runs phase B shorter. The part of the end spread the stage mix explains is about half
the total variance, it is not tied to a blockIdx (no SM-position effect), and a perfect static rebalance would cut the
mean wait by about 0.7 to 1.2 us (5.2 to 6.0 -> 4.3 to 5.0). The rest is per-CTA variation the schedule cannot predict.

The final pass itself: one token per CTA at verify 16 (128 tokens on 132 CTAs), two at verify 32. The first token costs
4.6 to 4.7 us from the barrier: 2.9 us to the end of pass 1 (the flag zeroing, the y-row loads of the token's local routes
(1.4 of 8 on average), the shuffles and two CTA barriers) and 1.7 us for pass 2 (the same rows again, the weighted sum, the
output store, a CTA barrier); the second token costs 3.2 us. The kernel exits 0.05 us after the last pass.

### Candidate ly_fin1: pass 2 from a shared-memory copy (box, no gain)

`tools/dl_ly_fin1_patch.py` writes `muse_decode_layer/cand/ly_fin1.py` (SHA256
bb918ff32a40b8db05bd486713c59e74af47bf38b3af11afca832ddaf5da478a). MAIN build only (`-DOKC_FIN1=1` on its define tuple;
every other build compiles the base's text): pass 1 also stores the two float4 it loaded per route into a row cache
behind the `red` partials (8 routes x 16 KB inside the free weight ring; static_assert on the ring size), and pass 2 reads
them back from shared memory instead of global. Same values, same FP32 adds in the same order (bitwise the base's); the
barrier, both CTA barriers, every fence and the stage loops are unchanged. Build (`tools/ly_micro_sass.py` with the new
`LY_SASS_MAIN_EXTRA` hook so the report compiles the MAIN build with the candidate's define; `.build/box/v16/tail/sass_fin1/`):
96 registers, 0 local, 22,642 instructions (base 22,647), the 16 stage loops 304 / 307 / 308 instructions with 16 HGMMA,
identical to the base's; the only loop that changed is the 2,449-instruction outer body that holds the final pass
(2,444). Reference check (`muse_decode_layer/profiles/box/fin1/chk.log`): `G_RELEASE_STARTUP_OK native (128, 32, 33, 256,
1, 8, 16) 56 verified_replays`, check PASS at all 12 cells.

Paired A/B (`tools/ly_prtl_ab.sh`, CTL=ly_fabq.py, c/w/c/w/c in one session, `--runs 60`, PDL pinned at 1, 8, 16, 32 in
both arms, all six loads, 14:35 to 14:37 UTC, no foreign GPU process; `tools/ly_prtl_table.py`, threshold max(2 x
control spread, 0.5 us short cells, 1% long cells)):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | w_a-c_a | w_a-c_b | w_b-c_b | w_b-c_c | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.33 | 65.10 | 64.86 | 65.10 | 65.26 | 0.46 | -0.05 | -0.22 | +0.24 | +0.24 | -0.16 | 0.93 | inside |
| decode 1 full | 72.11 | 71.73 | 71.86 | 71.78 | 71.23 | 0.88 | +0.02 | -0.38 | -0.13 | -0.08 | +0.54 | 1.76 | inside |
| decode 16 sliding | 126.74 | 127.06 | 126.61 | 126.67 | 127.04 | 0.43 | +0.07 | +0.32 | +0.45 | +0.06 | -0.37 | 1.27 | inside |
| decode 16 full | 200.03 | 199.66 | 200.46 | 199.41 | 200.48 | 0.45 | -0.79 | -0.37 | -0.80 | -1.06 | -1.07 | 2.00 | inside |
| decode 32 sliding | 144.29 | 143.78 | 144.66 | 143.60 | 142.93 | 1.73 | -0.27 | -0.51 | -0.88 | -1.06 | +0.67 | 3.46 | inside |
| decode 32 full | 295.50 | 295.76 | 295.76 | 295.07 | 294.85 | 0.91 | +0.05 | +0.26 | -0.00 | -0.69 | +0.22 | 2.95 | inside |
| verify 1 sliding | 102.53 | 102.14 | 102.82 | 101.76 | 102.86 | 0.34 | -0.78 | -0.38 | -0.67 | -1.06 | -1.10 | 0.67 | "win" (unchanged build) |
| verify 1 full | 111.01 | 110.93 | 111.10 | 110.03 | 110.03 | 1.07 | -0.23 | -0.08 | -0.18 | -1.07 | -0.00 | 2.14 | inside |
| verify 16 sliding | 243.57 | 242.80 | 243.09 | 241.82 | 243.30 | 0.48 | -1.01 | -0.77 | -0.29 | -1.26 | -1.47 | 2.43 | inside |
| verify 16 full | 326.99 | 326.88 | 327.12 | 326.48 | 326.86 | 0.26 | -0.31 | -0.11 | -0.24 | -0.64 | -0.38 | 3.27 | inside |
| verify 32 sliding | 296.75 | 297.60 | 297.98 | 297.22 | 297.10 | 1.23 | +0.13 | +0.85 | -0.38 | -0.77 | +0.11 | 2.97 | inside |
| verify 32 full | 459.76 | 460.40 | 460.88 | 460.91 | 460.80 | 1.12 | +0.18 | +0.64 | -0.48 | +0.03 | +0.11 | 4.60 | inside |

The verify 1 sliding "win" is on the MID build, whose text the patch does not touch (the 0.67 us threshold is below the
run-to-run movement of that cell). At the four target cells: -1.0 / -0.3 / +0.1 / +0.2 us, inside. The candidate's timeline
(`ly_fin1_tlv16f.py`, 20 inputs, `muse_decode_layer/profiles/box/tail/fin1_tlv16f.log`) shows why: pass 2 fell 1.69 /
1.71 / 1.75 / 1.68 -> 1.48 / 1.49 / 1.49 / 1.52 us, pass 1 rose 2.92 / 2.88 / 2.95 / 2.98 -> 3.15 / 3.11 / 3.03 / 3.06
(the shared-memory stores), first token total 4.65 / 4.67 / 4.73 / 4.64 -> 4.62 / 4.66 / 4.57 / 4.59; the second token at
verify 32 unchanged (span - max(B end) 9.31 / 9.42 vs 9.41 / 9.30). The second global round trip was worth 0.2 us, not
the 1.2 us the pass-2 duration suggested: the pass is bound by the first load after the barrier (every CTA's loads land on
the L2 at the same instant) and by its barriers, not by the re-read.

### Candidate ly_bal: the switch cost charged by n-tiles in the cost split (box, loss)

`tools/dl_ly_bal_patch.py` writes `muse_decode_layer/cand/ly_bal.py` (SHA256
fe5477fe0c68da1b59e023e60d3c6983b906205089748a946824fc7f0f51301c). MAIN build only (`-DOKC_BAL=1`): `wgs_unitc` takes
the job's token count and returns 90 + 40 x n-tiles (phase A) / 55 + 31 x n-tiles (phase B) instead of the constant 160 /
140, in the units of the per-stage cost tables (8-token stage 112 / 100), from the measured first-of-segment issue gaps
(phase A about 1.0 us + 0.3 us per n-tile, phase B 0.8 + 0.3, against 0.96 us per 8-token stage); used in the prefix, the
stage-at search and the CTA-of search (x-producer warp, before the stream). Per-stage tables, job list, segment walk,
fixup path, fences and barriers unchanged; stage loops identical (SASS 22,671 instructions, 96 registers, 0 local, the
16 stage loops 304 / 307 / 308 with 16 HGMMA equal to the base's). Reference check PASS at all 12 cells
(`muse_decode_layer/profiles/box/bal/chk.log`, `G_RELEASE_STARTUP_OK native ... 56 verified_replays`).

Paired A/B (same protocol, 14:41 to 14:42 UTC, `/root/okc/.build/prtl/bal/`):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | w_a-c_a | w_a-c_b | w_b-c_b | w_b-c_c | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.41 | 65.52 | 65.04 | 65.14 | 65.18 | 0.37 | +0.12 | +0.11 | +0.48 | +0.10 | -0.05 | 0.74 | inside |
| decode 1 full | 72.03 | 72.00 | 71.98 | 71.74 | 71.79 | 0.24 | -0.06 | -0.03 | +0.02 | -0.24 | -0.05 | 0.50 | inside |
| decode 16 sliding | 127.04 | 127.06 | 127.34 | 126.67 | 127.41 | 0.37 | -0.40 | +0.02 | -0.29 | -0.67 | -0.74 | 1.27 | inside |
| decode 16 full | 198.93 | 200.00 | 199.71 | 199.09 | 200.21 | 1.28 | -0.07 | +1.07 | +0.29 | -0.62 | -1.12 | 2.56 | inside |
| decode 32 sliding | 144.11 | 144.59 | 144.30 | 143.89 | 144.46 | 0.35 | -0.05 | +0.48 | +0.29 | -0.42 | -0.58 | 1.44 | inside |
| decode 32 full | 294.91 | 295.44 | 295.17 | 294.83 | 295.82 | 0.91 | -0.17 | +0.53 | +0.27 | -0.34 | -0.99 | 2.95 | inside |
| verify 1 sliding | 102.16 | 102.32 | 102.30 | 102.67 | 101.92 | 0.38 | +0.37 | +0.16 | +0.02 | +0.37 | +0.75 | 0.77 | inside |
| verify 1 full | 110.74 | 110.51 | 110.00 | 110.98 | 109.86 | 0.88 | +0.55 | -0.22 | +0.51 | +0.98 | +1.12 | 1.76 | inside |
| verify 16 sliding | 241.74 | 243.89 | 243.81 | 243.68 | 242.14 | 2.06 | +1.22 | +2.14 | +0.08 | -0.13 | +1.54 | 4.13 | inside |
| verify 16 full | 326.24 | 327.01 | 327.47 | 327.49 | 326.82 | 1.23 | +0.41 | +0.77 | -0.46 | +0.02 | +0.67 | 3.27 | inside |
| verify 32 sliding | 297.70 | 300.24 | 296.58 | 299.26 | 297.82 | 1.25 | +2.39 | +2.54 | +3.66 | +2.69 | +1.44 | 2.97 | inside |
| verify 32 full | 459.79 | 463.94 | 459.94 | 464.83 | 459.17 | 0.77 | +4.75 | +4.14 | +4.00 | +4.90 | +5.66 | 4.60 | loss |

A loss at verify 32 (+2.4 / +4.8 us, all eight pairs positive) and +1.2 / +0.4 at verify 16. Charging wide units more
hands their stages to more CTAs, so the wide jobs are cut more often and their fixups (n-tile serialized L2 round trips
on the finisher) and extra segments cost more than the end spread they were meant to remove; the deterministic part of
the spread was worth at most about 1 us (table above).

### Decision

No judge check: ly_fin1 is inside (no gain at the target cells), ly_bal is a loss; both get ruled-out rows. What the tail
is made of, measured: a 1.1 us grid-barrier release, 5.1 to 6.0 us of end spread of which about 1 us is predictable from
the stage mix, and a 4.6 us first-token final pass (plus 3.2 us for the second token at verify 32) that is bound by the
first load after the barrier and its CTA barriers, not by the second read. Levers not taken, with their estimated worth:
per-expert gating of the final pass (MFIN on the MAIN build) would let early CTAs start their tokens before the last CTA
ends, but it was forced off because its poll is bounded, it adds a `bar.sync` + `red.release.gpu` per phase B unit
(about 0.4 us per segment, the switch cost the previous section measured in phase A and not in phase B), and nearly every
token routes to one of the four widest experts, which end last, so the net is at most about 1 to 1.5 us per cell
(hypothesis) and needs an unbounded wait plus a review of the launch-parity scheme; BF16 y rows (OKC_YBF) would halve
the final pass's bytes but round y before the weighted sum (numerics); fewer segments per CTA belongs to the
segment-switch lane. The ramp (option c) was not attempted: the weights producer waits for the x gather, and
ly_b4_wpre already measured no gain from issuing weights before it.
## 2026-11-04 - Verify segment switches (box)

Task from the lead (moe lane): cut the number of segment switches each CTA of `okc::moe_fused<true>` pays at verify 16 /
32, or the cost of each switch at its source, without changing the release path's latency (ly_pub's loss). Base
`muse_decode_layer/cand/ly_fabq.py` (SHA256 3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555, never
edited). Box only (okc-h200, 2026-11-04 14:20 to 15:10 UTC, every GPU run under /root/okc/.build/box/gpu.lock, `nvidia-smi`
showed no foreign process before each launch; the verify-tail lane's runs shared the lock and interleaved with the A/B
steps), no judge job, no routing slack. Logs: `/root/okc/.build/prtl/rfin3/`, `rfin4/`, `rfin5/` (the three A/B rounds,
copied to `muse_decode_layer/profiles/box/rfin/round1..3/`), `/root/okc/.build/box/v16/rfin_tl/` (the candidate timeline,
log copied to `muse_decode_layer/profiles/box/rfin/tl/`), `/root/okc/.build/box/v16/rfin_sass2/` (SASS). Tools:
`tools/ly_seg_kind.py` (segments per CTA by kind and the switch cost after each kind, from the raw stamps of
`tools/ly_box_moe_tlv16g.py`, which now takes `--no-reorder`), `tools/dl_ly_rfin_patch.py` (the candidate).

### Segment count and switch cost by kind, before (measurement, base stamps)

`tools/ly_seg_kind.py` on the previous lane's raw stamps (`.build/box/v16/tlv16g.json`, 3 inputs x 132 CTAs per cell).
A segment is one unit's stages on one CTA; a cut unit has two shares (stream-K). The switch cost is the excess of the
first two issue gaps of a segment over the steady gap of its n-tile class, attributed to the previous segment's kind
(its epilogue and deferred flush run inside the next segment's first stage). Reading the base again: `OKC_WGSDFIN` is
2, so phase A does not use a designated finisher: both shares of a cut unit `red.add` their partials into the unit's
fixup slot and, deferred, each takes an `atom.acq_rel.gpu` round trip on its count; the last arriver reads the slot
back, zeroes it and runs the epilogue. Phase B uses the designated finisher (the share holding stage 0, at the CTA's
end), whose partner only counts with a fire-and-forget release.

| cell | phase A segments per CTA (whole / cut shares) | after a whole unit | after the trailing share (first in order) | after the leading share | phase B segments (whole / cut) | after whole | after the leading share |
|---|---|---:|---:|---:|---|---:|---:|
| verify 16 sliding | 5.55 (3.83 / 1.72) | 1.01 us | 2.64 | 3.16 | 3.19 (1.49 / 1.70) | 0.91 | 0.92 |
| verify 16 full | 5.33 (3.56 / 1.78) | 1.05 | 2.31 | 3.35 | 3.08 (1.37 / 1.71) | 0.92 | 0.99 |
| verify 32 sliding | 5.81 (4.21 / 1.60) | 1.21 | 2.78 | 3.90 | 3.38 (1.63 / 1.75) | 1.06 | 1.11 |
| verify 32 full | 6.28 (4.55 / 1.72) | 1.18 | 2.97 | 3.61 | 3.48 (1.93 / 1.55) | 1.07 | 1.25 |

At NTT 1 the three phase A kinds cost 0.90 to 1.03 / 1.96 to 2.53 / 2.50 to 2.88 us; at NTT 8 2.1 to 2.6 / 3.0 to 5.1 /
4.8 to 7.0. The cut cost does not depend on a job change (same job 2.53 vs job change 2.39 at verify 16 sliding) and is
therefore the shares' own work, not the ramp. Per CTA and phase A the two cut shares cost about 3 us more than two whole
switches (NTT 1), 6 to 9 us more at NTT 8; phase B's leading share costs the same as a whole switch (its finisher's cost
sits in the trailing flush at the phase end, outside the gaps). The segment count itself is set by the unit size (256
rows x 16 stages, 4.5 to 6.3 units per CTA) and the two range ends; a balance-aware snap of the range ends to unit
boundaries was modelled and rejected: with the cut cost at 1.3 to 1.6 us per share, a snap window of that size catches
about 11% more boundaries (the cepi window `OKC_WGSU` already snaps 14%) and leaves the slowest CTAs untouched, so the
span would not move. The lever chosen is the cost of a cut share at its source.

### Candidate ly_rfin: cut units finished in registers by the share holding stage 0

`tools/dl_ly_rfin_patch.py` writes `muse_decode_layer/cand/ly_rfin.py` (SHA256
c42c12390ce22538260ebea22820002a2e8acaa6f7526920396b86a4b94cc029) from the pinned base. The MAIN build alone gets
`-DOKC_RFIN=1 -DOKC_RFINQ=2` (`MAIN_EXTRA_DEFINES`; every other build, B4 of decode 32 included, compiles the base's text).
Change: ranges run in order in both phases (`OKC_WGSREORD` off), so the share holding stage 0 of a cut unit is its CTA's
last share of that unit and the other shares, on the CTAs after it, were run first and deposited long before. A share
that does not hold stage 0 stores its partial sums (plain float4 stores, no atomics) into its own CTA's fixup slot (a
range holds at most one such share per phase) and releases its stage count to the finisher's counter, deferred as before
(kind 2: `fence.proxy.async.global`, `bar.sync`, `red.release.gpu`). The share holding stage 0 (both phases,
`OKC_WGSDFIN` 3) waits for the count at its segment end while its accumulators are live (the base's kind-3 sequence:
`ld.acquire` loop, `bar.sync`, `fence.acq_rel.gpu`), reads the other shares' slots with `ld.global.cg` (two tiles in
flight), zeroes each slot it read, adds the values into the accumulators and falls through to the whole-unit epilogue
and its deferred release (kind 1 / kind 4: the same proxy fence, `bar.sync` and `red.release.gpu` as a whole unit).
Removed per cut unit: two `atom.acq_rel.gpu` round trips and their deferred waits, one share's `red.add.v4` traffic
and the atomics' own acknowledgement before the release membar. No fence, barrier or release is removed; the finisher
never waits by construction (its partner ran first on the next CTA); the sum is a + b as before (the base formed
0 + a + b by atomics). The inverse range mapping the finisher relies on (`wgs_cta_of` of stage 15 names the last
depositor) was checked in a 32-bit Python replica of `wgs_prefix` / `wgs_stage_at` / `wgs_cta_of`: 88,884 cut units
over 800 random job lists, 0 mismatches.

Bug found on the way, in the first build (9b12447d): the finisher did not zero the slots it read, so the B4 build of
decode 32, which shares the scratch and `red.add`s onto zeros, picked up stale deposits in its own cut units: the
startup expert comparison failed at M = 32 (global error 0.83, 0.88, 0.69 on the three comparisons; M = 33 / 128 / 256
bitwise equal to the base on all three route sets). The zeroing store after each read restores the invariant.

Build facts (box, `tools/ly_micro_sass.py` main build, `.build/box/v16/rfin_sass2/`) [correction 2026-11-04 15:30 UTC, cut-unit-order
lane: the box copy of the tool lacked the `MAIN_EXTRA_DEFINES` hook, so this report and the ly_rfinq4 one compiled the base's text
(main.sass md5 equal to the base's); rebuilt with the defines in `.build/box/v16/rfin_sassx/`: 96 registers, 0 spill, 22,867
instructions, stage loops 305 (x 7) / 307 (x 8) / 311 against the base's 304 / 307 / 308, so one instruction more in nine of the
sixteen loops (LEA / R2UR replacing ULEA / UIMAD address arithmetic, see "Verify cut unit order (box)")]: 96 registers, 0 spill, 0 stack,
stage loops 304 / 307 / 308 instructions with 16 HGMMA (the base's counts). Correctness (`.build/prtl/rfin3/chk.log`,
`muse_decode_layer/profiles/box/rfin/round1/chk.log`): `FORK_MOE_ROUTE native repaired`, `G_RELEASE_STARTUP_OK native
(128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, check PASS at all 12 cells. Direct comparison against the base's MAIN
build on the self-test's inputs (`.build/box/v16/rfin_diff2.log`: M = 33 / 128 / 256, route sets regular / hot (64-token
jobs, NTT 8) / partial, two runs each): max |diff| 0 between runs, 0 bad tokens against the base in all nine cases.

### Segment count and switch cost, after (measurement, candidate stamps)

`tools/ly_moe_tlv16.py --base ly_rfin.py` writes `muse_decode_layer/cand/ly_rfin_tlv16.py` (b304d051); run with
`tools/ly_box_moe_tlv16g.py ... --inputs 20 --raw 20 --no-reorder` (same f931 case seeds and router seed as the base's
run; `.build/box/v16/rfin_tl/`). Segments per CTA are unchanged (phase A 5.49 / 5.20 / 6.08 / 6.12, phase B 3.16 / 3.03
/ 3.49 / 3.48, cut shares 1.7 per CTA and phase); the switch after the depositing share now costs 0.92 / 0.95 / 1.04 /
1.01 us and after a whole unit 0.98 / 1.01 / 1.13 / 1.10 (the four cells in the table's order), so every switch inside a
phase costs the whole-unit price. The finisher's work moved to the phase end. Base -> candidate medians over the 20
inputs (us):

| cell | span | phase A | phase B | MFIN wait | final | A-end spread | B-end spread | x producer wait on doneAs | first gap A / B at NTT 1 |
|---|---|---|---|---|---|---|---|---|---|
| verify 16 sliding | 145.6 -> 143.2 | 86.0 -> 85.9 | 43.1 -> 42.4 | 6.6 -> 5.3 | 4.3 -> 4.4 | 12.8 -> 10.9 | 10.4 -> 8.3 | 15.2 -> 16.8 | 2.18 / 1.82 -> 1.76 / 1.79 |
| verify 16 full | 141.2 -> 139.4 | 83.3 -> 82.7 | 41.3 -> 40.9 | 6.2 -> 5.4 | 4.1 -> 4.5 | 12.4 -> 9.9 | 9.4 -> 8.2 | 15.0 -> 16.1 | 2.30 / 1.86 -> 1.79 / 1.79 |
| verify 32 sliding | 178.7 -> 175.0 | 103.1 -> 103.5 | 52.8 -> 51.6 | 7.2 -> 5.1 | 6.1 -> 6.2 | 15.8 -> 13.7 | 12.3 -> 9.2 | 16.7 -> 18.5 | 2.08 / 1.79 -> 1.79 / 1.76 |
| verify 32 full | 177.7 -> 177.3 | 103.4 -> 104.8 | 53.2 -> 51.7 | 6.3 -> 5.1 | 6.0 -> 6.1 | 16.3 -> 12.4 | 11.9 -> 9.5 | 17.1 -> 18.7 | 1.95 / 1.76 -> 1.76 / 1.73 |

Reading: phase B shortens by 0.4 to 1.5 us and the MFIN wait by 0.8 to 2.1 us (the phase B finisher at each CTA's end
no longer adds its own partial through atomics before reading back; the B-end spread falls 1.2 to 3.1 us), and both end
spreads shrink. Phase A is flat: the 3 us per CTA that left the switch gaps is paid again at the phase end, where the
finisher's count acquire, read-back round trip and epilogue now sit, and the phase B x producer waits 1.1 to 1.8 us longer
on doneAs because in range order each CTA's last phase A unit is its cut unit. Steady gaps are unchanged at every NTT
(0.96 at NTT 1, 1.50 to 1.54 at NTT 8). What remains of the cut cost is one L2 round trip per cut unit per phase at the
phase end plus the depositor's stores.

### Paired A/B against ly_fabq (box, three rounds)

`tools/ly_prtl_ab.sh` (CTL=ly_fabq.py, c/w/c/w/c in one session per round, `--runs 60`, PDL pinned at 1, 8, 16, 32 in
both arms, all six loads, five `bx_box_run.py` processes under the lock), 14:52 to 15:01 UTC, no foreign GPU process; the
verify-tail lane's steps interleaved on the lock. Threshold max(2 x control spread, 0.5 us short cells, 1% long cells);
table `tools/ly_prtl_table.py`. Round 1 (`.build/prtl/rfin3/`):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | w_a-c_a | w_a-c_b | w_b-c_b | w_b-c_c | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 64.96 | 65.07 | 65.33 | 65.66 | 65.22 | 0.37 | +0.20 | +0.11 | -0.26 | +0.34 | +0.45 | 0.74 | inside |
| decode 1 full | 72.19 | 71.39 | 70.75 | 71.55 | 71.09 | 1.44 | +0.13 | -0.80 | +0.64 | +0.80 | +0.46 | 2.88 | inside |
| decode 16 sliding | 126.62 | 127.15 | 127.23 | 127.49 | 127.76 | 1.14 | +0.11 | +0.53 | -0.08 | +0.26 | -0.27 | 2.27 | inside |
| decode 16 full | 199.04 | 198.70 | 200.88 | 200.67 | 200.56 | 1.84 | -0.47 | -0.34 | -2.18 | -0.21 | +0.11 | 3.68 | inside |
| decode 32 sliding | 143.92 | 143.87 | 143.76 | 143.57 | 144.08 | 0.32 | -0.20 | -0.05 | +0.11 | -0.19 | -0.51 | 1.44 | inside |
| decode 32 full | 295.18 | 295.17 | 294.59 | 294.64 | 295.57 | 0.98 | -0.21 | -0.02 | +0.58 | +0.05 | -0.93 | 2.95 | inside |
| verify 1 sliding | 102.58 | 101.97 | 102.45 | 102.42 | 102.62 | 0.18 | -0.36 | -0.61 | -0.48 | -0.03 | -0.21 | 0.50 | inside |
| verify 1 full | 110.16 | 109.95 | 110.22 | 111.17 | 110.64 | 0.48 | +0.22 | -0.21 | -0.27 | +0.94 | +0.53 | 0.96 | inside |
| verify 16 sliding | 243.90 | 243.30 | 244.06 | 243.14 | 243.12 | 0.94 | -0.48 | -0.61 | -0.77 | -0.93 | +0.02 | 2.44 | inside |
| verify 16 full | 328.13 | 326.74 | 326.85 | 325.78 | 327.10 | 1.28 | -1.10 | -1.39 | -0.11 | -1.07 | -1.33 | 3.27 | inside |
| verify 32 sliding | 298.02 | 293.42 | 295.78 | 293.74 | 297.42 | 2.24 | -3.49 | -4.59 | -2.35 | -2.03 | -3.68 | 4.48 | inside |
| verify 32 full | 461.10 | 459.23 | 460.69 | 459.49 | 460.26 | 0.85 | -1.32 | -1.87 | -1.46 | -1.20 | -0.77 | 4.61 | inside |

Rounds 2 and 3 (`rfin4/`, `rfin5/`), the four target cells, w - c mean and the four pair differences: verify 16 sliding
-1.12 (+0.94, -0.80, -2.35, -2.74) and +0.29 (-0.35, -1.07, +0.30, +1.60); verify 16 full -0.04 (-0.46, +0.90, -0.03,
-0.10) and -0.99 (-2.38, -2.53, -0.29, +0.83); verify 32 sliding -1.95 (-3.65, -3.18, -1.23, -0.00) and -2.81 (-3.33,
-2.94, -2.74, -2.26; threshold 2.97); verify 32 full -2.17 (-2.93, -2.51, -1.63, -1.50) and -0.95 (-0.74, -1.62, -1.01,
-0.82). Pooled over the three rounds (9 controls, 6 candidate runs, 54 pairs per cell):

| cell | control mean | spread | candidate mean | w - c | thr | pairs < 0 |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 65.08 | 0.61 | 65.26 | +0.18 | 1.22 | 14/54 |
| decode 1 full | 71.67 | 1.44 | 71.42 | -0.25 | 2.88 | 42/54 |
| decode 16 sliding | 126.94 | 1.20 | 127.09 | +0.15 | 2.40 | 17/54 |
| decode 16 full | 199.59 | 2.35 | 199.31 | -0.28 | 4.70 | 33/54 |
| decode 32 sliding | 143.98 | 0.59 | 144.17 | +0.18 | 1.44 | 25/54 |
| decode 32 full | 295.03 | 1.20 | 294.75 | -0.28 | 2.95 | 37/54 |
| verify 1 sliding | 102.43 | 1.31 | 102.17 | -0.26 | 2.62 | 45/54 |
| verify 1 full | 110.42 | 0.94 | 110.22 | -0.20 | 1.89 | 40/54 |
| verify 16 sliding | 243.37 | 2.27 | 242.93 | -0.44 | 4.54 | 35/54 |
| verify 16 full | 327.09 | 1.98 | 326.38 | -0.71 | 3.97 | 42/54 |
| verify 32 sliding | 297.01 | 2.26 | 294.26 | -2.75 | 4.51 | 54/54 |
| verify 32 full | 460.49 | 1.23 | 459.01 | -1.48 | 4.60 | 54/54 |

The eight cells whose builds are the base's text sit within 0.3 us of the control with mixed signs. At verify 32 the
candidate is faster in every one of the 54 pairs of both cells (-2.75 and -1.48 us mean); at verify 16 full 42 of 54
(-0.71 us), at verify 16 sliding 35 of 54 (-0.44 us). By the table's rule every cell is inside the threshold, because
the control spread (2.0 to 2.3 us at three of the four cells, with other lanes' runs between the steps) sets a 4 to 4.6
us bar that a 1 to 3 us mechanism cannot clear; the sign counts and the timeline (span -2.4 / -1.8 / -3.7 / -0.4 us on
20 paired inputs) agree on a gain of about 1 to 3 us at verify 32 and about 0.5 to 1 us at verify 16.

Knob variant `ly_rfinq4.py` (f6cc316c, `OKC_RFINQ=4`, four fixup tiles in flight): SASS identical in instruction count [same correction: measured on the base's text]
(22,647) and loops to Q = 2, ptxas already hoists the constant-offset loads. Its own round (`.build/prtl/rfinq4/`,
`muse_decode_layer/profiles/box/rfin/q4/`, check PASS 12 / 12, startup OK): verify 16 sliding -0.11 (+0.29, +0.43, -0.37,
-0.64), verify 16 full -1.01 (-1.42, -0.99, -0.98, -0.62), verify 32 sliding -2.41 (-0.88, -1.94, -3.30, -3.74; threshold
3.01), verify 32 full -1.95 (-0.58, -2.29, -2.51, -2.88): the same class as Q = 2, as the identical SASS predicts; the
knob is a no-op and ly_rfin (Q = 2) stays the candidate.

Decision: ly_rfin is correct (12 / 12, startup conformance, bitwise equal to the base on nine direct comparisons),
changes no fence, barrier or release path, costs nothing at the eight untouched cells and gains 1.5 to 2.8 us at
verify 32 and 0.4 to 0.7 us at verify 16 on the box (pooled means), below the 2 x spread bar of an interleaved session.
Value at the judge if the box deltas carry: about 0.15% of the layer score (0.046% per us at verify 32, 0.062% at
verify 16). Recommended to the lead as a judge-check candidate behind anything with a larger measured gain; not a
ruled-out row. Next lever on the same evidence: the finisher's read-back round trip at the phase end (one per cut unit
and phase) and the longer doneAs wait of the phase B x producer in range order (1.1 to 1.8 us), which a finisher that
runs the cut unit before the CTA's last whole unit would move off the phase A end without reintroducing a wait.

## 2026-11-04 - Verify cut unit order (box)

Task from the lead (moe lane, continuing "Verify segment switches"): run each CTA's cut unit before its last whole unit, so
that the finisher's L2 round trip and the phase B x producer's longer doneAs wait leave the phase A end. Base
`muse_decode_layer/cand/ly_rfin.py` (SHA256 c42c12390ce22538260ebea22820002a2e8acaa6f7526920396b86a4b94cc029, under a
judge check, never edited); the control of every A/B is ly_rfin, not ly_fabq. Box only (okc-h200, 2026-11-04 15:14 to
15:35 UTC, every GPU run under `/root/okc/.build/box/gpu.lock`, `nvidia-smi` showed no foreign process before each launch;
this lane was alone on the box), no judge job, no routing slack. Logs: `/root/okc/.build/prtl/rfin2/` (check and round 1),
`rfin2r2/` (round 2), `rfin2b/` (the phase B variant), `/root/okc/.build/box/v16/rfin2_tl/` (candidate timeline, the 45 MB
raw stamps stay on the box), `rfin2_sassx/`, `rfin_sassx/`, `rfin2b_sassx/` (SASS), `rfin2_diff.log` (direct comparison);
the logs are copied to `muse_decode_layer/profiles/box/rfin2/`. Tools: `tools/dl_ly_rfin2_patch.py` (the candidate),
`tools/ly_rfin2_order_check.py` (the ordering replica), `tools/ly_box_moe_tlv16g.py --rfin2 MASK` and
`tools/ly_seg_kind.py --rfin2 MASK` (the timeline decoders now know the new walk).

### Candidate ly_rfin2: the cut share before the CTA's last whole unit

`tools/dl_ly_rfin2_patch.py` writes `muse_decode_layer/cand/ly_rfin2.py` (SHA256
6c960ac6c06b4fdc771c064a6aee7d76d5662b9e196833a6cbe1b156dba13083) from the pinned ly_rfin; `--mask 3` writes
`ly_rfin2b.py` (e1a0f65d026e062cf51b49741d32b3911d253cde043249b6001d6fe6bb49aada), the same walk in phase B too. The MAIN
build alone gets `-DOKC_RFIN2=1` (`MAIN_EXTRA_DEFINES`; every other build compiles the base's text). A range [g0, g1) whose
last stage is inside a unit (g1 & 15 != 0) and that holds the whole unit before that cut share ([uw, ue), ue = first stage
of the unit holding g1 - 1, uw = ue - 16 >= g0) is walked as [g0, uw), [ue, g1), [uw, ue) by the weights producer, the x
producer and the consumers (one function, `wgs_rf2`, for all three: the ring is matched by issue count). A range without a
whole unit before its cut share keeps the stage order. The first segment, a depositing share when g0 is inside a unit,
stays first, so a range still holds at most one depositing share per phase and it still deposits before anything else.
Nothing else changes: the finisher's count acquire (`ld.acquire` loop, `bar.sync`, `fence.acq_rel.gpu`), the slot reads
and zeroing, the whole-unit epilogue, the deferred releases (kind 1 / 2 at commit group OKC_WGSFCI of the next segment's
first stage, the phase's end flush) and every fence and barrier are ly_rfin's text.

Ordering constraints (`tools/ly_rfin2_order_check.py`: a 32-bit replica of `wgs_prefix` / `wgs_stage_at` / `wgs_range` /
`wgs_cta_of` on 800 random job lists of 14 to 26 jobs, 1 to 64 tokens each, G = 132; per phase about 91,000 cut units,
none with three shares). Every CTA walks exactly its range once; `wgs_cta_of` of stage 0 names the finisher and of stage
15 the last depositor in every case. The finisher reaches its cut share after (range - 16 - share) stages, the depositor
on the next CTA releases its count (16 - share) + 1 stages after its phase start, so the slack is range - 33 stages when
both start together. Phase A: slack min -1 (14 of 91,154 units, two-unit ranges in 64-token regions), p1 4, p10 17,
median 43 stages, against min 15 / median 59 for ly_rfin's in-order walk; phase A starts are grid-synchronous (gather,
xready), so the finisher's wait is a wait by construction in 0.015% of cut units and the acquire loop covers it. Phase B:
slack min -16, p10 -4, median 6, 23% of cut units negative before the phase A end spread (10 to 14 us) is added: a wait
by construction, so the phase B walk (mask 3) was built and checked for correctness but predicted to lose.

Correctness (`muse_decode_layer/profiles/box/rfin2/round1/chk.log`): `FORK_MOE_ROUTE native repaired`,
`G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, check PASS at all 12 cells. Direct
comparison of the MAIN build against ly_fabq's on the self-test's inputs (`profiles/box/rfin2/diff/rfin2_diff.log`:
M = 33 / 128 / 256, route sets regular / hot (64-token jobs, NTT 8) / partial, two runs each, ly_rfin2 and ly_rfin2b): max
|diff| 0 between runs in all 18 cases, 0 bad tokens against ly_fabq in all 18.

SASS (`tools/ly_micro_sass.py` main build, box CPU). Finding first: the box copy of the tool lacked the `MAIN_EXTRA_DEFINES`
hook (the merge kept it in the worktree only), so the ly_rfin and ly_rfinq4 reports in the previous section compiled the
base's text (main.sass md5 equal to the base's `sass/`); the previous section carries a dated correction. Rebuilt with the
hook (`rfin_sassx/`, `rfin2_sassx/`, `rfin2b_sassx/`), moe_fused<true>:

| build | instructions | registers | local | stage loops (16, instructions each, 16 HGMMA) |
|---|---:|---:|---:|---|
| ly_fabq (base) | 22,647 | 96 | 0 | 304 x 7, 307 x 8, 308 |
| ly_rfin | 22,867 | 96 | 0 | 305 x 7, 307 x 8, 311 |
| ly_rfin2 | 22,893 | 96 | 0 | 307 x 8, 308 x 7, 314 |
| ly_rfin2b | 22,924 | 96 | 0 | 307 x 8, 308 x 7, 314 |

So 96 registers and 0 spill hold, but the stage loops are not the base's text: ly_rfin added one instruction to nine
loops (LEA x 3 and R2UR x 2 replacing ULEA x 3 and UIMAD x 2, UIADD3 17 -> 15: address arithmetic moved off the uniform
datapath) and ly_rfin2 adds two to three more (IMAD 39 -> 42, SHF + 1), the compiler's allocation, not code of the patch
(the walk runs outside the stage loop). The steady issue gaps below show no per-stage cost from it.

### Paired A/B against ly_rfin (box, two rounds)

`tools/ly_prtl_ab.sh` (CTL=ly_rfin.py, c/w/c/w/c in one session per round, `--runs 60`, PDL pinned at 1, 8, 16, 32 in
both arms, all six loads, 15:15 to 15:17 and 15:19 to 15:20 UTC, nothing else on the lock; table `tools/ly_prtl_table.py`,
threshold max(2 x control spread, 0.5 us short cells, 1% long cells)). Round 1 (`.build/prtl/rfin2/`):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c mean | w_a-c_a | w_a-c_b | w_b-c_b | w_b-c_c | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.18 | 65.26 | 64.42 | 64.82 | 64.77 | 0.77 | +0.25 | +0.08 | +0.85 | +0.40 | +0.05 | 1.54 | inside |
| decode 1 full | 70.99 | 71.62 | 71.31 | 71.06 | 71.39 | 0.40 | +0.10 | +0.62 | +0.30 | -0.26 | -0.34 | 0.80 | inside |
| decode 16 sliding | 126.43 | 126.43 | 127.28 | 126.74 | 126.88 | 0.85 | -0.28 | +0.00 | -0.85 | -0.54 | -0.14 | 1.70 | inside |
| decode 16 full | 199.22 | 197.92 | 199.14 | 198.59 | 199.74 | 0.61 | -1.11 | -1.30 | -1.22 | -0.54 | -1.15 | 1.99 | inside |
| decode 32 sliding | 143.60 | 143.58 | 144.46 | 143.12 | 143.44 | 1.02 | -0.48 | -0.02 | -0.88 | -1.34 | -0.32 | 2.05 | inside |
| decode 32 full | 294.82 | 295.62 | 294.72 | 295.39 | 294.27 | 0.54 | +0.90 | +0.80 | +0.90 | +0.67 | +1.12 | 2.95 | inside |
| verify 1 sliding | 101.55 | 101.71 | 101.31 | 102.14 | 102.40 | 1.09 | +0.17 | +0.16 | +0.40 | +0.83 | -0.26 | 2.18 | inside |
| verify 1 full | 109.44 | 109.95 | 109.94 | 109.79 | 109.92 | 0.50 | +0.11 | +0.51 | +0.02 | -0.14 | -0.13 | 0.99 | inside |
| verify 16 sliding | 242.18 | 243.15 | 242.43 | 241.65 | 242.91 | 0.74 | -0.11 | +0.98 | +0.72 | -0.78 | -1.26 | 2.43 | inside |
| verify 16 full | 326.21 | 325.86 | 325.65 | 325.94 | 326.66 | 1.01 | -0.27 | -0.35 | +0.21 | +0.29 | -0.72 | 3.26 | inside |
| verify 32 sliding | 294.77 | 294.86 | 294.53 | 293.42 | 293.81 | 0.96 | -0.22 | +0.10 | +0.34 | -1.10 | -0.38 | 2.94 | inside |
| verify 32 full | 457.98 | 457.89 | 458.74 | 459.28 | 458.30 | 0.75 | +0.24 | -0.10 | -0.85 | +0.54 | +0.98 | 4.58 | inside |

Round 2 (`rfin2r2/`), the four target cells, w - c mean and the four pair differences: verify 16 sliding +0.93 (+0.90,
+2.11, +1.41, +0.14), verify 16 full +0.80 (-0.05, +0.46, +1.44, +1.49), verify 32 sliding +1.22 (+1.09, +0.91, +1.34,
+1.46; spread 0.18), verify 32 full +0.45 (+0.62, +0.75, +0.19, +0.24; spread 0.18). Pooled (6 controls, 4 candidate runs,
24 pairs per cell):

| cell | control mean | spread | candidate mean | w - c | thr | pairs < 0 |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 64.90 | 0.85 | 65.08 | +0.17 | 1.70 | 7/24 |
| decode 1 full | 71.26 | 0.45 | 71.34 | +0.08 | 0.90 | 10/24 |
| decode 16 sliding | 126.89 | 0.85 | 126.71 | -0.18 | 1.70 | 17/24 |
| decode 16 full | 199.31 | 0.86 | 198.76 | -0.55 | 1.99 | 18/24 |
| decode 32 sliding | 144.01 | 1.09 | 143.86 | -0.15 | 2.18 | 16/24 |
| decode 32 full | 294.91 | 1.14 | 294.82 | -0.08 | 2.95 | 12/24 |
| verify 1 sliding | 102.10 | 1.26 | 102.35 | +0.25 | 2.53 | 8/24 |
| verify 1 full | 110.10 | 1.15 | 110.15 | +0.05 | 2.30 | 12/24 |
| verify 16 sliding | 242.63 | 1.26 | 243.05 | +0.41 | 2.53 | 7/24 |
| verify 16 full | 326.23 | 1.01 | 326.49 | +0.26 | 3.26 | 12/24 |
| verify 32 sliding | 294.38 | 0.96 | 294.88 | +0.50 | 2.94 | 6/24 |
| verify 32 full | 458.30 | 0.75 | 458.65 | +0.34 | 4.58 | 7/24 |

The eight untouched cells sit within 0.6 us of the control with mixed signs. At the four target cells the candidate is
slower by 0.3 to 0.5 us pooled, inside the threshold but with only 6 to 12 of 24 pairs negative and a clean +0.8 to +1.2
us second round: a small loss, not a win.

### Timeline: where the phase A end moved (candidate stamps, the same 20 inputs per cell as ly_rfin's run)

`tools/ly_moe_tlv16.py --base ly_rfin2.py` writes `ly_rfin2_tlv16.py` (bcb7a288); `tools/ly_box_moe_tlv16g.py ... --inputs
20 --raw 20 --rfin2 1` (`.build/box/v16/rfin2_tl/`, log in `profiles/box/rfin2/tl/`). ly_rfin -> ly_rfin2 medians (us):

| cell | span | phase A | phase B | MFIN wait | A-end spread | B-end spread | x producer wait on doneAs | first gap A at NTT 1 | steady gap A at NTT 1 / 8 |
|---|---|---|---|---|---|---|---|---|---|
| verify 16 sliding | 143.2 -> 144.0 | 85.9 -> 86.8 | 42.4 -> 42.0 | 5.3 -> 4.9 | 10.9 -> 11.7 | 8.3 -> 8.4 | 16.8 -> 14.0 | 1.76 -> 1.89 | 0.96 / 1.50 -> 0.96 / 1.50 |
| verify 16 full | 139.4 -> 140.0 | 82.7 -> 84.0 | 40.9 -> 40.3 | 5.4 -> 5.2 | 9.9 -> 10.9 | 8.2 -> 8.1 | 16.1 -> 12.9 | 1.79 -> 1.89 | 0.96 / 1.50 -> 0.96 / 1.50 |
| verify 32 sliding | 175.0 -> 176.9 | 103.5 -> 104.6 | 51.6 -> 51.2 | 5.1 -> 5.5 | 13.7 -> 14.4 | 9.2 -> 10.5 | 18.5 -> 15.5 | 1.79 -> 1.89 | 0.96 / 1.54 -> 0.99 / 1.54 |
| verify 32 full | 177.3 -> 177.2 | 104.8 -> 105.3 | 51.7 -> 51.2 | 5.1 -> 5.1 | 12.4 -> 14.4 | 9.5 -> 9.7 | 18.7 -> 15.4 | 1.76 -> 1.82 | 0.96 / 1.54 -> 0.96 / 1.54 |

Switch cost by the kind of the previous segment (`tools/ly_seg_kind.py --rfin2 1`, the excess of a segment's first two
issue gaps over the steady gap of its n-tile class, `profiles/box/rfin2/tl/seg_kind.txt`; in phase A the tool's k2 is the
share holding stage 0, the finisher, and k3 the depositor): phase A after a whole unit 1.02 / 1.02 / 1.12 / 1.04 us mean,
after the depositor 0.90 / 0.88 / 1.06 / 1.00, after the finisher 4.38 / 4.41 / 4.27 / 4.40 us (median 4.3; by the
finisher's job 4.2 at NTT 1 to 5.4 at NTT 8; n = 2,260 to 2,313 per cell). In ly_rfin the finisher was the last segment,
so its cost sat in the phase end and in no gap. Phase B is unchanged (after whole 0.89 to 1.12, after the depositor 0.83
to 0.96 in both files). Segments per CTA and cut shares per CTA are identical to ly_rfin's (the walk only permutes).

Reading: the mechanism did what it was built for on the dependency side. The cut unit's doneAs release now comes one unit
before the phase end and the phase B x producer's doneAs wait fell 2.8 to 3.3 us per CTA (below ly_fabq's 15.2 / 15.0 /
16.7 / 17.1); phase B shortened 0.4 to 0.6 us and the MFIN wait held or fell at verify 16. But the finisher's segment end,
no longer at the phase end, is a 4.3 us stall of the stream in the middle of the phase: the count acquire (no wait by
construction), `bar.sync`, `fence.acq_rel.gpu` under full HBM load, the slot read-back round trip, the zeroing stores and
the whole-unit epilogue, during which the 3-deep ring fills and the weights producer stops, and the loop then catches up
only at its 0.1 us per stage margin. Phase A grew 0.5 to 1.3 us and its end spread 0.8 to 2.0 us, more than phase B gave
back; the spans moved +0.8 / +0.6 / +1.9 / -0.1 us, the same picture as the A/B. At the phase end (ly_rfin) the same work
was hidden behind the phase B dependency wait that every CTA pays anyway; moving it earlier exposed it. The steady gaps
are the same at every n-tile count, so the extra loop instructions did not cost per-stage time.

### Decision

ly_rfin2 is a loss (+0.3 to +0.5 us pooled at the four target cells, +0.8 to +1.2 in a clean round) and ly_rfin2b has a
finisher wait by construction in phase B (its one round is recorded below); no judge check for either, ruled-out rows for
both. What the lane establishes: the finisher's cost (about 4.3 us at NTT 1, mostly the GPU-scope acquire fence and one L2
round trip under load) is paid wherever the cut share sits, and the phase end is the one place where it overlaps with a
wait the CTA takes anyway, so the in-order walk of ly_rfin is the right order for a finisher that reads back. The 3 us of
doneAs wait this lane removed from the phase B x producer is real but is not on the critical path once the finisher's
stall sits upstream of it; a version that keeps the cut share last and still releases the unit's doneAs earlier would need
the epilogue before the read-back, which the sum a + b forbids. The correction of the SASS claims (ly_rfin's stage loops
carry one more instruction, not zero) does not change ly_rfin's A/B reading: its steady gaps were measured unchanged on the
stamps.

ly_rfin2b, one round (`rfin2b/`, `profiles/box/rfin2/b_round1/`, 15:29 to 15:30 UTC, same protocol): verify 16 sliding
+6.63 us (+6.11, +7.38, +7.07, +6.54; threshold 2.53), verify 16 full +8.48 (+8.78, +8.88, +8.91, +7.76; 3.26), verify 32
sliding +3.27 (+3.23, +2.35, +3.39, +3.71; 2.94), verify 32 full +2.31 (+2.11, +1.31, +2.22, +3.06; 4.58), the eight other
cells within 0.6 us: a loss in every one of the sixteen target pairs, the phase B finisher's wait by construction that the
order check predicted (23% of cut units with negative slack before the phase A end spread), larger at verify 16 where the
phase B ranges are shortest.

## 2026-11-04 - Decode 1 timeline and edges (box)

Layer lane. Question: at decode 1 sliding and full (the two cells with the highest score weight), where does the time
between kernels and at the kernel edges go, and which single edge item can one candidate take? Base:
muse_decode_layer/cand/ly_fabq.py (SHA256 3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555, the
public file, never edited). Box only (okc-h200), every run under gpu.lock, no judge check. The bf16 lane shared the
box (its ly_box_prtl.py process held the lock before this run; nvidia-smi showed no other process).

Timeline run (15:36 UTC): `LY_TRACE_DIR=.build/box/ltl/tl_fabq tools/ly_tl_run.py muse_decode_layer
muse_decode_layer/cand/ly_fabq.py --loads "decode 1" --runs 20 --kernels --lock --ls-cache --ext-cache --pin-pdl8
1,8,16,32 --skip-pdl8-gate --router-seed 106218864389449120 --judge g1b4_pdl8d.f931abf568544018.json`, reference
check PASS at both cells, net 65.10 / 71.44 us (sliding / full, 20 runs). Per-kernel medians over the 10 profiled cold
replays (`tools/ly_tl_parse.py`; traces and log in `muse_decode_layer/profiles/box/ltl/`). All numbers are
measurements unless marked. Floors at the box read rate 4.44 TB/s (from the floor table of "Decode 16 / 32 sliding:
floor and gap"): qkv 6.39, attention sliding 0.12 (2048 tokens of FP8 K/V) and full 4.61 (80,000 tokens), o 2.84, b1
0.05, pooler + router 13.26, route 0, slice 1.42 (6.3 MB of 12-bit codes), finish 0.01; the MoE floor is 6.38 us per
distinct local expert, and the route-map fit (9.4 + 6.42 x distinct) puts the 10 profiled inputs at about 2.1
experts (13.5 us) against the judge's 2.5 target (16.0 us). Critical path = this kernel's end minus the previous
kernel's end (with PDL a secondary starts before its producer ends, so its duration includes its wait); gap = start
minus the previous end (negative: PDL overlap).

Decode 1 sliding (span 65.76 us, sum of durations 82.37):

| # | kernel | start | dur | end | gap | critical path | floor | over floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | qkv sk2<1,8,3,...> (no PDL producer, plain launch) | 0.00 | 9.86 | 9.86 | - | 9.86 | 6.39 | 3.47 |
| 1 | attn_kernel<1,false,13> | 8.14 | 6.83 | 14.88 | -1.70 | 5.02 | 0.12 | 4.90 |
| 2 | o sk2<1,8,4,...> | 10.27 | 8.24 | 18.48 | -4.56 | 3.60 | 2.84 | 0.76 |
| 3 | b1 glue_cluster_boundary<4,32,...,true,false> | 16.19 | 4.98 | 21.14 | -2.30 | 2.66 | 0.05 | 2.61 |
| 4 | pooler + router sk2_kernel_gle | 18.96 | 16.86 | 35.90 | -2.18 | 14.76 | 13.26 | 1.50 |
| 5 | _prq_sl_tile_route_cg | 33.63 | 3.71 | 37.36 | -2.27 | 1.46 | 0 | 1.46 |
| 6 | moe_fused<false> (TINY, cooperative, PDL off) | 37.46 | 22.96 | 60.69 | +0.10 | 23.33 | 13.5 (2.1 experts) | 9.8 |
| 7 | slice cgemv40_kernel<128,2> | 58.59 | 5.66 | 64.29 | -2.26 | 3.60 | 1.42 | 2.18 |
| 8 | _sl_cluster_finish_cg | 63.25 | 2.67 | 65.76 | -1.06 | 1.47 | 0.01 | 1.46 |

Decode 1 full (span 71.68 us, sum of durations 94.42):

| # | kernel | start | dur | end | gap | critical path | floor | over floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | qkv sk2 | 0.00 | 9.95 | 9.95 | - | 9.95 | 6.39 | 3.56 |
| 1 | attn_kernel<1,false,11> | 7.97 | 13.12 | 21.12 | -1.97 | 11.17 | 4.61 | 6.56 |
| 2 | o sk2 | 10.50 | 14.24 | 24.67 | -10.74 | 3.55 | 2.84 | 0.71 |
| 3 | b1 glue | 22.32 | 4.85 | 27.28 | -2.27 | 2.61 | 0.05 | 2.56 |
| 4 | pooler + router gle | 25.10 | 16.77 | 41.90 | -2.18 | 14.62 | 13.26 | 1.36 |
| 5 | route | 39.60 | 3.68 | 43.33 | -2.27 | 1.43 | 0 | 1.43 |
| 6 | moe_fused<false> | 43.44 | 22.45 | 66.30 | +0.13 | 22.97 | 13.5 | 9.5 |
| 7 | slice cgemv40 | 64.38 | 5.41 | 69.74 | -2.03 | 3.44 | 1.42 | 2.02 |
| 8 | finish | 68.72 | 2.69 | 71.68 | -1.06 | 1.94 | 0.01 | 1.93 |

Gaps and edge waits. The only positive launch gap is route -> MoE (+0.10 / +0.13 us: the cooperative launch has no
programmatic attribute, so the MoE's entry wait returns at once and its CTAs start after the route completes). Every
other edge is a PDL edge whose secondary starts 1.06 to 4.56 us (sliding) before its producer ends and waits inside
its own duration; the sum of those overlaps is 16.5 / 22.7 us (sum of durations minus span), which is prefetch, not
loss. What an edge still costs after the producer's end is inside the critical-path column: the ruled-out PDL row
puts the producer's completion and flush before `griddepcontrol.wait` returns at about 0.7 us per edge (its
estimate, not re-measured here), so the eight PDL edges plus the serial one hold about 6 us of the 65.8 us span
(hypothesis). The PDL guard of the base is clean at every T (judged log: PDL8H_GUARD pass, T stays on, at T = 1 / 8 /
16 / 32); this run pinned the same set.

Three largest non-floor items (critical path minus floor, sliding / full):

1. The MoE's non-stream time, 9.8 / 9.5 us at the profiled inputs (MOE-TL-01 at T = 1 on the same TINY engine: ramp
   3.1, A-to-B handoff 2.2, tail 2.1 of which barrier latency 0.9; the rest is the unit loop's 1.15 us per unit
   against 1.10 of bytes). The loop and the handoff are the MoE lane's; the ramp's first fill and the barrier ticket
   are declined in docs/ruled-out.md (final-day brief mechanisms 1 and 2).
2. Attention, 4.90 / 6.56 us: a latency-bound kernel at sliding (0.52 MB) and 5 to 6 us of startup at full; the
   attention lane's (its decode 32 / 16 full startup is recorded in "Full-context attention limit").
3. qkv, 3.47 / 3.56 us: the first kernel of the step, a plain launch with a cold start after the L2 flush (the bf16
   lane's ramp item).

In this lane's scope (between kernels and at the edges): the tail after the MoE, slice + finish, is 5.07 / 5.38 us
for a 1.43 us floor (3.64 / 3.95 over); b1 2.61 / 2.56; route 1.46 / 1.43. The load-ordering family at these edges
is closed by the blueprint section (finpre null, b1d32 null, pre-wait stages 3 / 4 lose at decode 1, sk2 and glue
trigger placements lose both ways, route-in-MoE bounded at -0.25 / +0.42), and the fusions by the audit (M-C, F3, the
slice tail, the last-arriver forms).

### The slice edge, read from the source and the resource usage

The slice is `cgemv40_kernel<128, 2>`: 512 CTAs of 160 threads (4 K-split warps plus the escape warp), 40 registers
(`cuobjdump -res-usage` of the box's cgemv extension: REG 40, LOCAL 0, SHARED 1064), 12 KB of codes per CTA loaded
into registers before `griddepcontrol.wait`, then the x row (`__ldcg`), the FMAs, a warp reduction and one store; the
trigger is at the end. The MoE triggers `launch_dependents` at each CTA's phase B end, before the grid barrier, so the
slice launches when the last CTA leaves phase B (2.26 / 2.03 us before the MoE ends: the barrier, the final pass and
the exit). A TINY MoE CTA is 288 threads at 145 registers (the unit-loop section's census), about 43,800 of the SM's
65,536 registers at the 256-per-warp granularity; a slice CTA takes 6,400, so 3 slice CTAs fit beside the MoE CTA
on every SM, 396 of 512. The other 116 can only launch as MoE CTAs exit, and then load their 12 KB of codes from DRAM
after the MoE's end; the slice's end is the last of those (hypothesis: it explains 3.6 us of critical path for a
kernel whose post-wait work is one L2 row read, 8 FMAs per thread, a reduction and a store, about 1.5 us). At 32
registers (5,120 per CTA) 4 slice CTAs fit, 528 >= 512, and every CTA holds its codes before the MoE ends.

### Candidate ly_cg32: the slice launched as the 32-register cgemv_kernel

`tools/dl_ly_cg32_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_cg32.py`: the pdl8 launch
edit's `cg_go(cgemv40_kernel<NTH, CPT>, ...)` becomes `cg_go(cgemv_kernel<NTH, CPT>, ...)`, the file's own
`__maxnreg__(32)` kernel with the same body (the CPT == 1 branch already launches it). Nothing else changes: the
wait before the x loads, the end trigger, the launch attribute, every fence and barrier of every kernel, and the
PDL set per T. At T = 8 the slice is the sk2 kernel (bit 29, no cgemv), so verify 1 and the longer cells run the
base's code and are the session's null. Expected (hypothesis): decode 1 sliding and full -1.0 to -1.5 us if the
second wave is the cost, less the 0.2 us the standalone bench charged 32 registers against 40 (docs/runs.md,
5ff76a62b0044378).

Candidate file: muse_decode_layer/cand/ly_cg32.py (SHA256
8bbb1b94dfb6a45709e855066888224b72f4e711a7f59c8d2f91ee7288bbcc3e); `diff` against the base is the first-line header
and the one `cg_go(...)` string. Correctness (box, `bx_box_run.py --runs 5`, six loads, 16:0x UTC, log
`.build/box/ltl/cg32/chk.log`): reference check PASS at all 12 cells, G_RELEASE_STARTUP_OK (56 verified replays),
`CG_GATE unpooler cg_vs_sk2 1.21e-04 cg_vs_fp64 0.00e+00 sk2_vs_fp64 1.21e-04 equal 1023/1024 ok` (the 32-register
cgemv matches FP64 on the gate input; the sk2 arm is the one 1 ulp off there, which the judged base log also shows
as input-dependent: 0 / 0 / 0 on its input). Resource and SASS comparison (`cuobjdump -res-usage` and `-sass` on the
box's two cgemv extensions): cgemv_kernel<128, 2> REG 32, STACK 0, LOCAL 0, SHARED 1064, 688 instructions;
cgemv40_kernel<128, 2> REG 40, STACK 0, LOCAL 0, SHARED 1064, 696 instructions. The floating-point and load mix is
identical (FADD 54, FFMA 64, FMUL 5, FSEL 10, LDG 23, SHFL 31, PRMT 32, LOP3 105 in both); the eight-instruction
difference is integer bookkeeping (IMAD 114 vs 110, IADD3 21 vs 16, VIADD 21 vs 24, SHF 32 vs 33, NOP 9 vs 13, no
UMOV). No spill at 32 registers, so the 40-register choice of the standalone bench bought nothing in code here.

### Paired A/B against ly_fabq (box, 15:52 to 15:54 UTC)

`tools/ly_prtl_ab.sh` with CTL=ly_fabq.py, CAND=ly_cg32.py: five bx_box_run.py processes c/w/c/w/c under gpu.lock,
all six loads, `--runs 60`, `--pin-pdl8 1,8,16,32 --skip-pdl8-gate`, router seed 106218864389449120, f931 case
seeds, local_start per input; every arm's timed PDL set {1, 8, 16, 32} on, reference check PASS in every run.
Logs and JSONs: `muse_decode_layer/profiles/box/cg32/` (box `.build/prtl/cg32/`). Table by
`tools/ly_prtl_table.py` (threshold max(2 x control spread, 0.5 us) at decode 1 / verify 1, max(2 x spread, 1%)
elsewhere); net us, candidate mean minus control mean, pairs w_a-c_a, w_a-c_b, w_b-c_b, w_b-c_c.

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 65.74 | 65.58 | 65.47 | 65.52 | 65.07 | 0.67 | +0.12 | -0.16 +0.11 +0.05 +0.45 | 1.34 | inside |
| decode 1 full | 72.02 | 72.03 | 71.79 | 71.68 | 72.19 | 0.40 | -0.14 | +0.02 +0.24 -0.11 -0.51 | 0.80 | inside |
| decode 16 sliding | 126.88 | 126.69 | 127.28 | 126.19 | 127.25 | 0.40 | -0.70 | -0.19 -0.59 -1.09 -1.06 | 1.27 | inside |
| decode 16 full | 198.88 | 198.98 | 199.12 | 198.51 | 199.63 | 0.75 | -0.47 | +0.10 -0.14 -0.61 -1.12 | 1.99 | inside |
| decode 32 sliding | 144.22 | 144.10 | 143.87 | 143.04 | 143.65 | 0.58 | -0.35 | -0.13 +0.22 -0.83 -0.61 | 1.44 | inside |
| decode 32 full | 295.97 | 294.93 | 294.46 | 295.20 | 294.77 | 1.50 | -0.00 | -1.04 +0.46 +0.74 +0.43 | 3.01 | inside |
| verify 1 sliding | 102.51 | 102.75 | 101.94 | 102.13 | 102.19 | 0.58 | +0.23 | +0.24 +0.82 +0.19 -0.06 | 1.15 | inside |
| verify 1 full | 110.83 | 110.78 | 109.76 | 110.32 | 110.54 | 1.07 | +0.17 | -0.05 +1.02 +0.56 -0.22 | 2.14 | inside |
| verify 16 sliding | 242.45 | 242.74 | 243.26 | 243.01 | 242.64 | 0.82 | +0.09 | +0.29 -0.53 -0.26 +0.37 | 2.43 | inside |
| verify 16 full | 327.49 | 327.02 | 327.22 | 327.18 | 327.12 | 0.37 | -0.17 | -0.46 -0.19 -0.03 +0.06 | 3.27 | inside |
| verify 32 sliding | 297.76 | 297.66 | 296.93 | 296.64 | 297.14 | 0.83 | -0.12 | -0.10 +0.74 -0.29 -0.50 | 2.97 | inside |
| verify 32 full | 459.95 | 460.06 | 461.46 | 459.20 | 459.57 | 1.89 | -0.69 | +0.11 -1.39 -2.26 -0.37 | 4.60 | inside |

Sign counts at the changed cells: decode 1 sliding 3 of 4 pairs positive, decode 1 full 2 of 4; the ten unchanged
cells (the base's code) move by -0.70 to +0.23 and are the session's null. A null, not a loss: the candidate is
indistinguishable from the base at decode 1.

### Candidate timeline (same instrument, `muse_decode_layer/profiles/box/ltl/tl_cg32/`, net 65.02 / 71.58 us)

Decode 1 sliding, base -> candidate: slice start 58.59 -> 58.26 (2.26 -> 2.29 us before the MoE's end), duration
5.66 -> 5.62, critical path 3.60 -> 3.33; finish start 63.25 -> 61.97 (1.06 -> 1.90 us before the slice's end),
duration 2.67 -> 3.44, critical path 1.47 -> 1.63; span 65.76 -> 65.55. Full: slice 3.44 -> 3.48, finish 1.94 ->
1.46, span 71.68 -> 71.92. The other seven kernels are within 0.25 us of the base.

Reading. The slice's post-MoE critical path moved by 0.27 us at sliding and not at all at full, and whatever it gave
back the finish absorbed (it launches earlier and waits longer), so the second-wave residency explains at most
about 0.3 us of the slice's 2.2 us excess, and the step is unchanged. The 3.3 to 3.5 us from the MoE's end to the
slice's end is therefore not the late CTAs' code loads. What remains unseparated (hypotheses): the wait-return
latency behind a 132-CTA cooperative kernel's completion (its CTAs exit after the final pass, and the flush covers
the MoE's scratch and output writes), the x row read after the wait, and the escape warp's table walk; a per-CTA
entry / wait-return / store stamp in the slice would separate them (not built; one kernel edit plus the launch
plumbing). ly_cg32 is not kept; the ruled-out row records the measurement. No judge check.

### Decision

No candidate for a judge check from this lane tonight. The decode 1 edges are measured: the eight PDL edges hold
about 6 us (hypothesis from the 0.7 us per-edge estimate), the route -> MoE serial edge 0.1 us, and the post-MoE
tail (slice + finish) 3.6 to 4.0 us over its floor, of which residency was 0.3. Every load-ordering, trigger and
fusion lever at these edges is closed by earlier sections; what is left needs an instrument inside the slice and
finish kernels or a change inside the MoE's tail (the MoE lane's).
## 2026-11-04 - Pooler + router remaining gap (box)

bf16 lane. Question: inside the fused pooler + FP32 router kernel (`sk2_kernel_gle<1, 8, 4, 1, 8, 1, 6, true...>`,
decode 1 sliding 2.8% of the score) of the public file, where does the time above the byte floor go now that the
route-side quarter sum (ly_prq2) is in, and does one candidate against the largest piece shorten the layer? Base
`muse_decode_layer/cand/ly_fabq.py` (SHA256 3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555, never
edited). Box only (okc-h200, no foreign GPU process at any run, every run under gpu.lock), no judge check, nothing
registered. Timeline runs 2026-11-04 15:30 to 15:47 UTC (07:30 to 07:47 PT), router seed 106218864389449120, f931 case
seeds, local_start per input, PDL on at T = 1, 8, 16, 32. Logs and JSONs: box `/root/okc/.build/prtl/fabq/`
(`tl_fabq.{log,json}`, `tl_fabqc.{log,json}`, `tl_fabqc2.{log,json}`).

### Instruments

- `tools/dl_ly_prtl_patch.py --src muse_decode_layer/cand/ly_fabq.py --out muse_decode_layer/cand/ly_fabq_prtl.py`
  (47966e43): the 2026-11-03 stamps (entry, pre-wait issues, wait, x issue, producer issue per stage, escape warp 0's
  landing per stage, consumer loop exits, router partial, epilogue, final store) on ly_fabq's text; the patch applies
  unchanged.
- `tools/dl_ly_prtlc_patch.py` (new) writes `ly_fabq_prtlc.py` (35cc918f): the per-stage clocks taken by consumer warp 0
  instead (slot 36 + s: its full-barrier return for stage s; slot 16 + s: done with stage s, pooler stages at the end of
  the body and the interleaved FP32 router stages before their `continue`). Two globaltimer reads and stores per stage in
  warp 0 sit inside the loop, so this build's loop SASS is not the base's: a diagnostic only. Back-to-back stamps
  (done19 to the loop exit) measure 0.07 us, the stamp overhead bound. `tools/ly_prtlc_summary.py` reads it.
- `ly_fabq_prtlc2.py` (056ce493, built inline from the same tool with three extra tail stamps: after
  `griddepcontrol.launch_dependents`, after the router warp butterflies and partial store, after the `cbar` wait).
- `tools/ly_box_prtl.py` as before (two graphs per cell, 5 fresh inputs, cold L2, `--raw 2 / 3`).

### Post-wait span and its decomposition (measurements, us)

Floor as in the 2026-11-03 section: 58.83 / 59.17 MB at 4.443 TB/s = 13.24 / 13.32 us (T = 1 / 8). Medians over 5 inputs
per cell, ly_fabq_prtl build (escape-warp landing clocks; same columns as the 2026-11-03 table):

| cell | wait | land0 | loop end | loop-end spread | router | epilogue | tail | post-wait span | floor | excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 1.50 | 0.77 | 13.44 | 0.77 | 0.41 | 0.40 | 0.77 | 14.75 | 13.24 | +1.51 |
| decode 1 full | 0.99 | 1.16 | 13.37 | 0.96 | 0.41 | 0.41 | 0.80 | 14.82 | 13.24 | +1.58 |
| verify 1 sliding | 1.28 | 0.85 | 15.28 | 0.96 | 0.40 | 0.38 | 0.74 | 16.61 | 13.32 | +3.29 |
| verify 1 full | 1.31 | 0.76 | 14.97 | 0.93 | 0.39 | 0.39 | 0.74 | 16.58 | 13.32 | +3.26 |
| decode 16 sliding | 1.02 | 1.46 | 11.54 | 3.81 | 1.55 | 0.42 | 2.05 | 15.71 | 13.29 | +2.42 |
| decode 16 full | 0.83 | 1.37 | 11.55 | 3.81 | 1.56 | 0.42 | 2.08 | 15.46 | 13.29 | +2.17 |
| decode 32 sliding | 1.79 | 0.62 | 12.68 | 2.91 | 2.20 | 0.63 | 2.75 | 17.18 | 13.35 | +3.84 |
| decode 32 full | 1.63 | 0.63 | 12.48 | 2.69 | 2.18 | 0.63 | 2.88 | 16.74 | 13.35 | +3.39 |

Against the 2026-11-03 ly_micro2 table the tail at T <= 8 fell from 1.8-2.0 to 0.74-0.80 us and the post-wait span by
2.7 to 3.1 us at decode 1 / verify 1 (ly_prq2's counter removal, now measured in the public file). The ly_fabq_prtlc
build (consumer clocks) gives post-wait spans of 14.14 to 14.24 (decode 1 sliding), 14.56 to 16.54 (decode 1 full, one
input with a 0.9 us stall at stage 3), 15.5 to 16.1 (verify 1): the two instruments differ by about 0.3 to 0.5 us at
decode 1 and the plain graph times of the two builds are inside replay noise of each other, so the spans below carry
that uncertainty.

Consumer warp 0's clocks (ly_fabq_prtlc, means over 3 inputs x 128 CTAs; wait = full-barrier return minus the previous
stage's done, process = done minus return):

| cell | first return after wait | pooler stage process (median) | router stage process | wait per stage (median / mean) | last stage process | loop exit to final store |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 0.40 | 0.512 | 0.416 | 0.128 / 0.149 | 0.36 | 0.80 |
| decode 1 full | 0.64 | 0.512 | 0.416 | 0.128 / 0.190 | 0.34 | 0.79 |
| verify 1 sliding | 0.72 | 0.512 | 0.608 | 0.128 / 0.149 | 0.64 | 0.80 |
| verify 1 full | 0.75 | 0.512 | 0.608 | 0.128 / 0.148 | 0.66 | 0.80 |

Decomposition of the decode 1 sliding post-wait span (about 14.2 to 14.7 us):

1. Ramp, wait to the first stage consumed: 0.40 us (sliding) / 0.64 (full). The first stage's weights were issued 0.5
   to 1.0 us before the wait (first issue 0.32 us after entry, wait 1.0 to 1.5 us after entry) and land 1.75 us after
   their issue in every cell; the x tile is issued 0.15 us after the wait and the stage is consumed 0.65 us after the x
   issue when the weights are already in (sliding). Measurement: the ramp is bounded below by the x tile's TMA round
   trip (about 0.65 us from issue to consumption) and above by the pre-wait weight latency when the predecessor ends
   early (full). The prologue before the first weight load (0.32 us after entry) is hidden under the predecessor at
   every short cell.
2. Loop, 20 stages (16 pooler, 4 interleaved FP32 router): 12.9 us. Pooler stage 0.51 us of processing plus 0.13 us of
   wait = 0.64 us per stage, which is 3.146 MB per stage on 128 CTAs at 4.9 TB/s; router stage 0.42 + 0.13 at T = 1,
   0.61 + 0.13 at T = 8. The stream is at the DRAM rate (4.9 TB/s is above the 4.443 used for the floor, so the loop
   beats the written floor by about 0.6 us) and the consumers process a pooler stage in 80% of its DRAM time: no
   lever in the loop's rate at T = 1. Hypothesis for the T = 8 router stage (0.61 us): shared-memory bandwidth, since
   every warp re-reads all 8 x32 rows (36 LDS.128 per lane per stage, 147 KB per CTA per stage at 128 B per cycle is
   0.58 us); it is hidden under the DRAM time except on the last stage.
3. Last stage, landing to loop exit: 0.36 us at T = 1 (a router stage), 0.65 at T = 8. Exposed, since nothing streams
   behind it.
4. Tail per CTA, loop exit to final store, 0.80 us at every short cell (ly_fabq_prtlc2, stamp cost 0.07 included once):
   `launch_dependents` 0.07, router warp butterflies and partial store 0.23 (40 `shfl.xor` plus adds for all 8 token
   slots whatever M is), `cbar` wait 0.06, named barrier 0.06, epilogue `cbar` wait and named barrier 0.10, reduction
   buffer write and named barrier 0.10, final sum (4 escape copies + 8 K-warps, 8 threads at M = 1) and BF16 store
   0.18. Three CTA-wide barrier rounds for 0.22 us, and 0.2 us of shuffles that reduce 7 unused tokens at T = 1.
5. CTA spread: the last CTA's loop end is 0.4 to 0.5 us after the mean CTA's (loop-end spread 0.77 to 1.25 max minus
   min). Weak correlation with entry, wait and first landing (r = 0.1 to 0.3); not explained.

Sum: 12.0 us of bytes at the measured 4.9 TB/s + 0.4 ramp + 0.36 last stage + 0.8 tail + 0.45 spread = 14.0, against
14.2 to 14.7 measured. The remaining gap over the written floor is 1.2 to 1.6 us at decode 1 and 2.3 to 3.3 us at
verify 1 (where the four T = 8 router stages add 4 x 0.1 us in the loop and 0.3 more on the last stage). The x gather
(x tiles by TMA, x32 rows by bulk copies inside the router stages) costs nothing visible beyond the ramp's 0.65 us
round trip. The largest non-floor item inside the kernel is the tail (0.8 us on every CTA, so on the critical path in
full), then the ramp, the last stage and the CTA spread (0.35 to 0.65 each).

### Candidate: ly_pepi, a shorter tail at T <= 8

`tools/dl_ly_pepi_patch.py` (base ly_fabq, pinned; `--base / --pin / --out` as tools/dl_ly_prq_patch.py) ->
`muse_decode_layer/cand/ly_pepi.py` (SHA256 dc425a65d5f912af601ea8bd2ae0f64701392bc6f2500fe5a30d7a9f622885fc). Only
the gle fused build (sk2_kernel_gle, FUSE, T <= 8) changes, through a define the gle source transform sets
(`SK2_REDX 1`); the bf16_proj build (T = 16 / 32) and the o kernel compile the same code as before:

1. The K-warp reduction buffer `red` gets its own shared-memory region after the zero row (WSK x TOKS x RP floats,
   20,480 B; the launch adds it to the dynamic size: 206,800 -> 227,296 B at pmax 58, under the 227,836 B the 4.6 KB of
   static gle smem leaves). The consumers write it right after the loop without first proving the ring idle: the
   epilogue's `cbar` wait and named barrier before the write go; the `cbar` wait (escape corrections complete) moves
   after the write, ahead of the final sum; the named barrier after the write stays.
2. The ilv block's `cbar` wait and named barrier after the router partial store run only when outr != nullptr (the
   counter path, compiled but never launched by this file; there the barrier orders the partial stores before the
   release atom).
3. The router warp butterflies reduce tokens t < M only (5 shuffles at T = 1 instead of 40); the per-token order is
   unchanged.

Kept: every fence, every mbarrier wait and arrival, the ring protocol, the escape warps' `cbar`, the named barrier that
publishes red, and the summation orders of the pooler output and the router partials. Expected (hypothesis from the
tail decomposition): 0.25 to 0.3 us off every CTA's tail at decode 1, 0.1 to 0.15 at verify 1.

Correctness. Box chk run (`.build/prtl/pepi/chk.log`): reference check PASS at all 12 cells, G_RELEASE_STARTUP_OK (56
verified replays), PDL on at T = 1, 8, 16, 32; the CG_GATE line is `cg_vs_sk2 1.21e-04 cg_vs_fp64 0.00e+00 sk2_vs_fp64
1.21e-04 equal 1023/1024 ok`, which is what every box run of ly_fabq and of every earlier candidate prints
(`.build/prtl/*/ab_c_*.log`; the judge prints 0.00e+00 and 1024/1024 for the same file), so the box line is a box
property and the judged-log criterion can only be checked by a judge run. `tools/ly_pepi_bitwise.py` (fused kernel
launched by each file's own gle_pr on the same seeded inputs at T = 1, 2, 3, 5, 8, with the empty outr the layer passes
and with a real outr): pooled rows, the four quarter partials and the counter-path logits are bitwise equal between
ly_fabq and ly_pepi in every case (`bit_cmp.log`: ALL BITWISE EQUAL). SASS (`tools/ly_prtl_sass.py`, gle build,
`.build/prtl/sass_pepi/`): fused kernel 128 -> 128 registers, 0 stack, 0 local; the consumer stage loop 377 -> 377
instructions with the same HMMA / LDS / FFMA / LOP3 counts and five integer opcode substitutions (IMAD 29 -> 30, SHF
16 -> 18, VIADD 16 -> 15, IADD3 9 -> 8, LEA 6 -> 4, + 1 MOV), whole kernel 2,949 -> 3,032; the o kernels are SAME.

### Paired A/B against ly_fabq (box)

`tools/ly_prtl_ab.sh` (CTL=ly_fabq.py, CAND=ly_pepi.py, c/w/c/w/c in one session, five bx_box_run.py processes under
gpu.lock, `--runs 60`, all six loads, PDL pinned at 1, 8, 16, 32, `--skip-pdl8-gate`, router seed 106218864389449120,
f931 judge JSON), 2026-11-04 15:56 to 15:58 UTC, logs and JSONs `/root/okc/.build/prtl/pepi/` (`ab_c_a` ... `ab_c_c`),
table by `tools/ly_prtl_table.py`. No foreign GPU process before or during the session. Net us; delta = candidate mean
minus control mean; threshold max(2 x control spread, 0.5 us) at decode 1 / verify 1 and max(2 x spread, 1%) elsewhere;
pairs = sign of w_a - c_a, w_a - c_b, w_b - c_b, w_b - c_c.

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs > 0 | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.07 | 65.14 | 64.93 | 65.33 | 65.33 | 0.40 | +0.12 | 3/4 | 0.80 | inside |
| decode 1 full | 71.58 | 72.29 | 72.06 | 71.87 | 71.73 | 0.48 | +0.29 | 3/4 | 0.96 | inside |
| decode 16 sliding | 126.53 | 126.99 | 127.09 | 126.77 | 126.98 | 0.56 | +0.02 | 1/4 | 1.27 | inside |
| decode 16 full | 200.59 | 199.79 | 199.54 | 199.01 | 199.25 | 1.34 | -0.39 | 1/4 | 2.69 | inside |
| decode 32 sliding | 143.62 | 144.45 | 143.82 | 144.32 | 144.70 | 1.09 | +0.34 | 3/4 | 2.18 | inside |
| decode 32 full | 294.91 | 295.49 | 294.99 | 294.02 | 294.74 | 0.26 | -0.13 | 2/4 | 2.95 | inside |
| verify 1 sliding | 102.22 | 103.78 | 102.35 | 102.40 | 102.66 | 0.43 | +0.68 | 3/4 | 0.86 | inside |
| verify 1 full | 110.94 | 111.38 | 110.72 | 111.70 | 111.09 | 0.37 | +0.62 | 4/4 | 0.74 | inside |
| verify 16 sliding | 242.35 | 242.98 | 244.38 | 243.22 | 243.79 | 2.03 | -0.41 | 1/4 | 4.06 | inside |
| verify 16 full | 326.69 | 327.14 | 327.54 | 327.57 | 327.76 | 1.07 | +0.02 | 2/4 | 3.27 | inside |
| verify 32 sliding | 296.99 | 297.89 | 297.86 | 297.82 | 298.91 | 1.92 | -0.06 | 2/4 | 3.84 | inside |
| verify 32 full | 460.18 | 461.47 | 460.58 | 461.12 | 460.72 | 0.54 | +0.81 | 4/4 | 4.60 | inside |

Every cell inside its threshold; the changed cells (decode 1, verify 1) lean the wrong way (+0.1 to +0.7 us, 13 of 16
pairs positive) and the unchanged cells (T >= 16, the session's null) scatter -0.4 to +0.8. Not a win; by the sign
counts at verify 1 a small loss.

Why: the stamped candidate (`ly_pepi_prtl.py` 90595351, `tl_pepi.{log,json}`) shows the tail did not move at decode 1
(loop exit to final store 0.80 -> 0.80 us per CTA: 0.39 + 0.05 + 0.16 + 0.20 against 0.41 + 0.11 + 0.11 + 0.19, the
removed barrier rounds reappear as a longer red write) and grew at verify 1 (0.78 -> 1.31): the per-token `if (t < M)`
around each butterfly chain made the compiler serialize the 8 chains (40 dependent shuffles at about 30 cycles each,
0.96 us against 0.23 for the interleaved chains). Kernel timeline (`ly_tl_run.py --kernels`, 10 replays,
`ktl_ly_fabq` / `ktl_ly_pepi`): fused kernel 16.56 -> 16.62 (decode 1 sliding), 16.80 -> 16.78 (full), 17.79 -> 18.11
(verify 1 sliding), 17.49 -> 18.11 (full); the route kernel after it 3.76 -> 3.70 / 3.89 -> 3.90 at decode 1 but 3.73 ->
4.90 / 3.44 -> 4.59 at verify 1 (it starts under PDL at the same trigger and ends 0.8 us later behind the longer tail).

### ly_pepi2: the same with the butterflies fixed

`tools/dl_ly_pepi_patch.py` now writes `muse_decode_layer/cand/ly_pepi2.py` (SHA256
f760505cc666eab3547c141360928a5415105d84f17dbe226da7f00211402890): changes 1 and 2 as above; change 3 becomes a single
uniform branch, `if (M == 1)` one token's chain, else the original 8 interleaved chains. Box chk: PASS at all 12 cells,
G_RELEASE_STARTUP_OK, PDL on at every T, the same box CG_GATE line. Stamped (`ly_pepi2_prtlc2.py` b34fa74f, consumer
clocks plus the tail stamps, `.build/prtl/pepi2/tl_pepi2c.{log,json}`): tail per CTA 0.69 us at decode 1 and 0.75 at
verify 1 (ly_fabq 0.80 in the same instrument class): launch_dependents 0.07, butterflies + partial store 0.20 (T = 1,
5 shuffles) / 0.27 (T = 8), 0.05, 0.04, red write + cbar wait + named barrier 0.12, final 0.20. So the two barrier
rounds and 35 shuffles were worth 0.1 us at decode 1 and 0.05 at verify 1, not the 0.25 to 0.3 expected: the 0.2 us
between launch_dependents and the partial store does not depend on the shuffle count (measurement; what it is, not
traced).

Paired A/B (same recipe, CTL=ly_fabq.py, CAND=ly_pepi2.py, 16:04 to 16:06 UTC, `/root/okc/.build/prtl/pepi2/`):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs > 0 | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 65.02 | 65.36 | 65.30 | 65.31 | 65.07 | 0.27 | +0.21 | 4/4 | 0.54 | inside |
| decode 1 full | 71.95 | 72.45 | 71.74 | 72.00 | 71.68 | 0.27 | +0.43 | 4/4 | 0.54 | inside |
| decode 16 sliding | 126.98 | 126.21 | 126.38 | 126.88 | 127.07 | 0.69 | -0.27 | 1/4 | 1.38 | inside |
| decode 16 full | 199.12 | 198.59 | 199.86 | 199.71 | 199.34 | 0.74 | -0.29 | 1/4 | 1.99 | inside |
| decode 32 sliding | 144.82 | 144.42 | 143.57 | 143.26 | 144.05 | 1.25 | -0.30 | 1/4 | 2.50 | inside |
| decode 32 full | 295.30 | 295.31 | 294.29 | 294.21 | 295.49 | 1.20 | -0.26 | 2/4 | 2.95 | inside |
| verify 1 sliding | 102.37 | 102.67 | 102.19 | 102.64 | 102.61 | 0.42 | +0.27 | 4/4 | 0.83 | inside |
| verify 1 full | 110.42 | 111.39 | 110.14 | 110.50 | 110.70 | 0.56 | +0.52 | 3/4 | 1.12 | inside |
| verify 16 sliding | 243.31 | 243.28 | 242.54 | 241.74 | 242.19 | 1.12 | -0.17 | 1/4 | 2.43 | inside |
| verify 16 full | 328.11 | 327.92 | 327.22 | 327.95 | 326.85 | 1.26 | +0.54 | 3/4 | 3.27 | inside |
| verify 32 sliding | 297.33 | 297.66 | 296.83 | 296.59 | 297.07 | 0.50 | +0.05 | 2/4 | 2.97 | inside |
| verify 32 full | 462.06 | 461.10 | 460.22 | 460.64 | 460.03 | 2.03 | +0.10 | 2/4 | 4.61 | inside |

Inside everywhere again, and the four changed cells are positive in 15 of 16 pairs (+0.2 to +0.5 us) while the
unchanged cells scatter around zero. Two candidates with a measured 0.1 us shorter kernel tail and a consistent +0.3
us at the layer: the loss has another cause than the tail code.

### ly_pepi3: the extra shared memory alone

`muse_decode_layer/cand/ly_pepi3.py` (SHA256 60d76809b1908f85a994f5bafbafe1b6756b2b3e2e0e7e17fe929947dbc54583, built
inline from the first and third kernel edits and the first file edit of tools/dl_ly_pepi_patch.py): ly_fabq with only
the `SK2_REDX` define and the launch's extra 20,496 B of dynamic shared memory for the gle fused kernel (206,800 ->
227,296 B); the kernel code is the base's (red stays in the ring, every barrier as before). Box chk PASS at 12 cells.
Paired A/B (same recipe, 16:07 to 16:09 UTC, `/root/okc/.build/prtl/pepi3/`):

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs > 0 | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 64.99 | 65.47 | 64.78 | 64.38 | 64.27 | 0.72 | +0.25 | 3/4 | 1.44 | inside |
| decode 1 full | 72.24 | 71.79 | 70.70 | 71.07 | 71.12 | 1.54 | +0.08 | 2/4 | 3.07 | inside |
| decode 16 sliding | 126.99 | 127.01 | 126.93 | 126.66 | 127.17 | 0.24 | -0.20 | 2/4 | 1.27 | inside |
| decode 16 full | 198.83 | 199.20 | 198.58 | 199.15 | 199.41 | 0.83 | +0.24 | 3/4 | 1.99 | inside |
| decode 32 sliding | 144.05 | 143.94 | 144.35 | 143.84 | 144.34 | 0.30 | -0.36 | 0/4 | 1.44 | inside |
| decode 32 full | 294.78 | 295.71 | 295.86 | 295.12 | 294.19 | 1.66 | +0.47 | 2/4 | 3.33 | inside |
| verify 1 sliding | 102.02 | 103.73 | 102.29 | 102.35 | 102.38 | 0.37 | +0.81 | 3/4 | 0.74 | loss |
| verify 1 full | 111.10 | 111.87 | 110.50 | 111.07 | 110.69 | 0.61 | +0.71 | 4/4 | 1.22 | inside |
| verify 16 sliding | 245.38 | 244.13 | 242.72 | 242.74 | 242.40 | 2.98 | -0.07 | 3/4 | 5.95 | inside |
| verify 16 full | 328.29 | 326.45 | 325.70 | 326.66 | 326.35 | 2.59 | -0.23 | 3/4 | 5.18 | inside |
| verify 32 sliding | 297.14 | 297.12 | 298.11 | 297.07 | 297.87 | 0.98 | -0.61 | 0/4 | 2.98 | inside |
| verify 32 full | 460.90 | 460.69 | 461.55 | 460.34 | 460.26 | 1.30 | -0.39 | 1/4 | 4.61 | inside |

The shared-memory growth by itself reproduces the candidates' short-cell drift (+0.25 / +0.08 at decode 1, +0.81 /
+0.71 at verify 1, 12 of 16 pairs positive, one cell past its threshold), with the kernel code untouched. Hypothesis
(not traced): the fused kernel at 227 KB asks for the SM's full shared-memory carveout while its PDL neighbours (the
boundary kernel before it, the route kernel after it) run at smaller carveouts, and the reconfiguration or the lost
co-residency costs a few tenths of a microsecond per launch; the gle fused kernel at 207 KB already sits in the
largest carveout class on paper, so the step may be the 4.6 KB static + 227 KB dynamic sum crossing a boundary. Either
way the dedicated red region cannot be paid for by the 0.1 us it saves in the tail.

### Reading

- The pooler + router kernel in the public file is 1.2 to 1.6 us over its written byte floor at decode 1 (2.3 to 3.3 at
  verify 1), and the loop streams at 4.9 TB/s, above the 4.443 the floor assumes. The remaining excess is four pieces of
  0.35 to 0.8 us each (ramp, last stage, tail, CTA spread), none of them a barrier or a wait that a one-line change
  removes: the tail's three barrier rounds and 35 idle shuffles were worth 0.1 us when removed (ly_pepi2 stamps), and
  the mechanism's shared-memory cost outweighed it.
- No judge check recommended for ly_pepi, ly_pepi2 or ly_pepi3. Ruled-out rows added.
- Left for a later day (hypotheses with their sizes): the T = 8 FP32 router stage is shared-memory-bandwidth bound
  (0.61 us per stage against 0.47 of DRAM time; 0.1 us per stage hidden in the loop, 0.3 exposed on the last stage), a
  2-row x 4-token warp mapping would cut its smem reads by a third; a red region that fits without growing the launch
  (16 tokens x 8 warps x 40 floats could be 8 x 8 x 40 at M <= 8 if the write were guarded) would test the barrier
  removal without the carveout cost, for at most 0.1 us; the ramp (0.4 to 0.65 us, the x tile's TMA round trip after
  the wait) and the CTA spread (0.45 us) have no in-kernel lever identified.

## 2026-11-04 - Decode 1 post-MoE tail (box)

Layer lane. Question: of the 3.3 to 3.7 us from the MoE's last CTA exit to the slice's end at decode 1 (sliding /
full, one token), which part is the largest: the wait-return latency behind the cooperative MoE, the x read after the
wait, the escape warp's table walk, or something else; then one candidate on that part. Base:
muse_decode_layer/cand/ly_fabq.py (SHA256 3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555, the
public file, never edited). Box only (okc-h200), every run under gpu.lock, no judge check. The bf16 lane shared the
box (its ly_box_prtl.py process held the lock when this lane's first run queued; nvidia-smi showed no process outside
the lock at any run start).

### Instrument: ly_pmt (stamped copy of ly_fabq, never a candidate)

`tools/dl_ly_pmt_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_pmt.py`: per-CTA globaltimer
stamps in the three kernels of the tail, every one behind a switch the plain graph leaves off, so the plain graph runs
the base's kernels:

- slice cgemv (`CG_CUDA_SRC` after the pdl8 edits): `cgemv_body` / `cgemv_kernel` / `cgemv40_kernel` get a `bool TL`
  template parameter (default false, the launched kernel unchanged); `cgemv_launch` picks the TL instantiation only
  while `cg_set_tlb(ptr)` holds a stamp buffer. 16 int64 per CTA: entry, weight loads issued, after
  `griddepcontrol.wait`, lane 0's x landed, every lane of warp 0's x landed (consumed, `__syncwarp`), every lane of
  warp 1's weights landed, warp 0's FMAs and warp sums done, escape warp count landed / after its wait / first x
  element landed / after its loop / after its warp sums, after the CTA barrier, output stored, SM id, escape count.
- MoE TINY build (helper0): `-DOKC_DEBUG=1` compiles the source's own `DBG(k)` stamps (8 per CTA: entry, prologue,
  gather, join, phase A end, phase B end = `launch_dependents`, after the grid barrier, exit after the final pass),
  taken only when a launch passes a fourth scratch tensor (`_PMT_MOE_DBG`), which the plain graph does not.
- finish (`_sl_cluster_finish_cg`, Triton): a stamp pointer argument and a `TLON` constexpr (entry, after
  `gdc_wait`, the slice output P consumed, carry and tail consumed, stores issued); off, a dummy pointer and no code.

No fence, barrier, wait, trigger or launch attribute moves. `tools/ly_box_pmt.py` captures a plain and an
instrumented graph of the whole layer per cell (PDL pinned on at every T, the gate skipped, router seed
106218864389449120, f931 case seeds, local_start per input), replays each fresh input through both with a cold L2 and
reads the stamps; it prints per-input lines and medians and writes a JSON. Observer cost: the instrumented graph is
1.3 to 2.9 us slower than the plain one (first round: the stamp that consumed thread 0's weights before its wait
delayed warp 0; removed in the second round), so the absolute stamp times carry that bias, the plain graph does not.

### Round 1 (16:0x to 16:11 UTC, `muse_decode_layer/profiles/box/pmt/pmt_fabq.{log,json}`): 20 inputs per cell

The slice's escape table has pmax 6 (at most 5 escapes per CTA, median 1), so the escape warp's walk is one
iteration per lane; the escape-table hypothesis is small before any timing. Medians over the 20 inputs, us from the
first MoE CTA's entry; reference points are the last MoE CTA's `launch_dependents` (trig), the grid barrier's
release (bar), the last MoE CTA's exit (exit), the last slice CTA's store (slice end) and the last finish program's
stores (fin end). All numbers are measurements unless marked.

| item | sliding | full |
|---|---:|---:|
| MoE trig (last CTA's launch_dependents) | 19.36 | 19.46 |
| MoE barrier release (last CTA) | 21.20 | 20.80 |
| MoE exit, median CTA / last 8 (final pass) / last | 20.74 / 22.20 / 23.01 | 20.80 / 21.20 / 21.49 |
| final pass after the barrier (last 8 CTAs) | 1.34 | 0.49 |
| slice entry min / max (all 512 CTAs) | 20.59 / 20.69 | 21.46 / 21.55 |
| slice entry after trig | 1.26 | 1.63 |
| slice CTAs entering after the MoE exit | 0 of 512 (one input 107) | 0 (five inputs 512) |
| slice weights landed (warp 0 lane 0), median / max | 22.30 / 23.04 | 23.18 / 24.16 |
| slice CTAs with weights landed before the MoE exit | 498 of 512 | 0 of 512 |
| slice wait return, min / median / max | 23.47 / 23.49 / 23.50 | 22.72 / 23.40 / 24.48 |
| wait return after the MoE exit, median / max | 0.48 / 0.51 | 1.73 / 2.70 (gated by the weights, see below) |
| lane 0's x after the wait, median / p90 / max | 0.38 / 0.61 / 0.80 | 0.19 / 0.40 / 0.69 |
| lane 0's x landed to warp 0's sums done, median / max | 1.98 / 2.75 | 1.76 / 3.17 |
| escape warp: wait to walk done (one iteration), median / p90 | 1.38 / 1.92 | 1.38 / 1.68 |
| escape walk done minus warp 0's sums done | -0.96 | -0.92 |
| CTA barrier wait / store | 0.14 / 0.06 | 0.40 / 0.06 |
| CTA wait return to its store, median / max | 2.62 / 3.20 | 2.54 / 3.78 |
| slice end median / p90 / last | 26.13 / 26.34 / 26.74 | 26.03 / 26.51 / 26.93 |
| slice end after the MoE exit | 3.66 | 5.22 |
| finish entry before the slice end | 0.38 | 0.30 |
| finish wait return after the slice end / P landed / carry landed / stores | 0.39 / 0.42 / 0.48 / 0.06 | 0.42 / 0.38 / 0.48 / 0.06 |
| finish end after the slice end / after the MoE exit | 1.63 / 5.34 | 1.57 / 6.74 |
| plain graph / instrumented graph | 68.54 / 71.39 | 75.76 / 77.02 |

Readings (sliding). The wait-return latency is 0.48 us and tight (all 512 CTAs released within 0.03 us), so the
cooperative kernel's completion and flush is a small part, not the 2 us the earlier section guessed. All 512 slice
CTAs enter together 1.26 us after the trigger, at the moment the first MoE CTAs exit, and 498 of them hold their
weights before the last MoE CTA exits (the final pass takes 1.34 us after the barrier, which is the slice's prefetch
window). The largest part is between lane 0's x landing and warp 0's sums being done: 1.98 us median for about 250
instructions, and the escape warp's single iteration (one x element load, one table load that hits in L1) takes
1.38 us in the same window, so the x row read by 512 CTAs at the same instant is the suspect (hypothesis: 32k line
requests on the 64 lines of one 8 KB row; the second round separates every lane's x landing from the math). The
finish is 1.6 us: 0.4 wait, 0.4 P, 0.5 carry, serial loads after its wait.

Readings (full). The MoE's final pass is short (0.49 us after the barrier) and the slice enters 1.63 us after the
trigger, 0.03 us before the last MoE CTA exits, so its weights land 1.7 us after the exit; the wait stamp of round 1
was gated by thread 0 consuming its weights before the wait (an instrument artefact fixed in round 2), so the "wait
return" column at full is the weights landing, and the real wait-return latency at full is not separated in this
round. The post-exit tail is therefore: launch latency remainder plus the 1.7 us weight stream (which the MoE's
longer final pass hides at sliding), then the same 2 us x-plus-math block, then a 0.9 us end spread. Nothing in the
slice can start its weights earlier than its launch, and the launch follows the trigger by 1.3 to 1.6 us in both
cells (the trigger sits at the MoE's phase B end, before the barrier; trigger placements are the MoE lane's and are
covered by ruled-out rows).

### Round 2 (16:16 to 16:21 UTC, `profiles/box/pmt/pmt2.{log,json}` base, `pmtx2.{log,json}` candidate): the parts

Second stamp set: the stamp before the wait no longer consumes anything (thread 0's weight loads are only issued),
warp 1 reports when every lane's weights landed, warp 0 reports when every lane's x landed (consumed, `__syncwarp`),
and escape lane 1 reports when its x element landed. 20 inputs per cell, medians, us; "base" is ly_pmt (ly_fabq
stamped), "xca" is ly_pmtx (the candidate below, stamped the same way). Per-CTA items are median over the 512 CTAs
then median over inputs; "crit" is the CTA with the last store.

| item | base sliding | xca sliding | base full | xca full |
|---|---:|---:|---:|---:|
| MoE last exit | 22.86 | 23.06 | 21.82 | 21.49 |
| final pass after the barrier (last 8 CTAs) | 1.29 | 1.33 | 0.45 | 0.53 |
| slice entry after the trigger | 1.26 | 1.28 | 1.63 | 1.73 |
| weight loads issued = landed, after entry (the issue blocks on the LSU queue) | 1.63 | 1.63 | 1.63 | 1.63 |
| CTAs with weights landed before the MoE exit | 479 | 488 | 0 | 0 |
| wait return after the MoE exit, median / max | 0.58 / 0.61 | 0.50 / 0.51 | 1.57 / 2.50 (wait reached after the weights) | 1.60 / 2.30 |
| lane 0's x after the wait | 1.22 | 0.55 | 0.67 | 0.42 |
| every lane of warp 0's x after the wait, median / p90 / max | 1.82 / 2.18 / 2.50 | 1.28 / 1.47 / 1.76 | 1.47 / 1.76 / 2.67 | 1.14 / 1.41 / 1.76 |
| math after the last x (FMAs, warp sums), median / max | 0.74 / 1.18 | 0.77 / 1.06 | 0.86 / 1.47 | 0.80 / 1.20 |
| escape warp: x element after its wait / loop / sums | 1.17 / 1.09 / 0.10 | 0.38 / 0.38 / 0.13 | 0.83 / 0.86 / 0.18 | 0.35 / 0.42 / 0.14 |
| escape done minus warp 0 done (negative: the compute warps are the laggards) | -1.34 | -1.47 | -1.34 | -1.42 |
| CTA barrier / store | 0.06 / 0.06 | 0.06 / 0.06 | 0.06 / 0.06 | 0.06 / 0.06 |
| CTA wait return to store, median / max | 2.74 / 3.36 | 2.18 / 2.69 | 2.51 / 3.60 | 2.06 / 2.69 |
| slice end (last store) after the MoE exit | 4.00 | 3.17 | 5.12 | 4.29 |
| crit CTA: wait latency / every lane's x / math | 0.59 / 2.46 / 0.74 | 0.50 / 1.65 / 0.86 | 1.55 / 2.53 / 0.83 | 1.70 / 1.60 / 0.86 |
| finish: wait after the slice end / P / carry / end after the slice end | 0.42 / 0.40 / 0.48 / 1.60 | 0.38 / 0.35 / 0.48 / 1.60 | 0.42 / 0.42 / 0.54 / 1.60 | 0.42 / 0.42 / 0.51 / 1.63 |
| finish end after the MoE exit | 5.62 | 4.75 | 6.77 | 5.90 |
| plain graph / instrumented graph (us, 20 cold replays) | 68.75 / 70.62 | 68.58 / 70.05 | 75.98 / 77.68 | 75.15 / 76.94 |
| max output difference, instrumented vs plain | 0 | 0 | 0 | 0 |

Decomposition of the base's MoE-exit-to-slice-end at decode 1 sliding (4.00 us on the instrumented graph, 3.66 on
round 1's; the plain graph's tail is about 0.3 shorter than the instrumented one): wait return 0.58; x landing for
every lane of a warp 1.82 (the largest part; lane 0 alone 1.22, the escape warp's single element 1.17, so it is the
L2 serving 512 CTAs x 64 lines of one 8 KB row at the same instant, not a lane's luck: hypothesis named by the
candidate's effect below); math 0.74; barrier and store 0.13; the rest (about 0.7) is the spread of the last CTAs,
whose x came last (crit CTA: x 2.46). At full the order is: launch 1.63 after the trigger (0.03 before the last MoE
exit, since the final pass is short), weights 1.63, then the same x 1.47 and math 0.86 block, end spread 0.9. The
escape table walk is one iteration and off the critical path (the compute warps finish 1.3 us after it), the finish
is 1.6 us in both cells (wait 0.4, P 0.4, carry 0.5, store 0.1), and the wait-return latency behind the cooperative
kernel is 0.5 to 0.6 us (a measurement; the earlier 0.7 us per-edge estimate stands).

### Candidate ly_xca: the slice reads x through L1 after its wait

`tools/dl_ly_xca_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_xca.py` (SHA256
1d45bee2ef4c8cd8fa7dfd52fdac385a79579d606709192aa3426151eae66373): in the pdl8 cgemv edits, the compute threads' two
x loads and the escape warp's x element load become `__ldca` (ld.global.ca, L1 allocating) instead of `__ldcg` (L2
only). They stay after `griddepcontrol.wait`; nothing in the kernel touches x before the wait, and the wait makes the
MoE grid's writes visible to this grid (PTX ISA 8.x, griddepcontrol: "all the memory operations from the
prerequisite grids are performed and made visible to the current grid"), so no line of x can be in L1 before the
first read and the loads see the MoE's output. Trigger, launch attribute, fences, barriers, the PDL set per T and the
weights' L1::no_allocate stream loads are unchanged; the `diff` against the base is the header line and the one
`_PDL8_CG_EDITS` line (two `__ldcg` tokens). The mechanism (hypothesis): the SM's L1 merges the x requests of its 3
to 4 resident slice CTAs (16 warps) into one fill per line, so L2 sees about 4x fewer requests on the hot row.

SASS (`tools/ly_cg_sass.py`, device-only nvcc 13.0 build of the post-edit cgemv source on the box, sm_90a):
cgemv40_kernel<128, 2> base and candidate both 688 instructions, REG 40, STACK 0, LOCAL 0, SHARED 1064; the only
difference is 9 lines, the x loads `LDG.E.128.STRONG.GPU` -> `LDG.E.128.STRONG.SM` (4) and the escape warp's
`LDG.E.U16.STRONG.GPU` -> `LDG.E.U16.STRONG.SM` (5, unrolled copies). Opcode histogram identical (FFMA 64, FADD 54,
LOP3 105, PRMT 32, LDG 23, SHFL 31).

Stamped candidate (ly_pmtx, table above): every lane's x lands 1.28 instead of 1.82 us after the wait (sliding) and
1.14 instead of 1.47 (full), the escape warp's x element 0.38 instead of 1.17, the math unchanged, the slice's end
after the MoE exit 3.17 instead of 4.00 (sliding) and 4.29 instead of 5.12 (full), the finish unchanged: -0.83 us on
the instrumented graph in both cells, all 20 inputs; output identical (max difference 0).

Correctness (box, `ly_prtl_ab.sh` chk step, `bx_box_run.py --runs 5`, six loads, 16:22 UTC, log
`profiles/box/xca/chk.log`): reference check PASS at all 12 cells, G_RELEASE_STARTUP_OK (56 verified replays),
`CG_GATE unpooler cg_vs_sk2 1.21e-04 cg_vs_fp64 0.00e+00 sk2_vs_fp64 1.21e-04 equal 1023/1024 ok`, the same line as
the control's prepare in the same session (ab_c_a.log) and as the base's earlier box logs (the one sk2 ulp is
input-dependent, the cgemv arm matches FP64); timed PDL set {1, 8, 16, 32} on in both arms.

### Paired A/B against ly_fabq (box, 16:22 to 16:26 UTC, two rounds)

`tools/ly_prtl_ab.sh` with CTL=ly_fabq.py, CAND=ly_xca.py: five bx_box_run.py processes c/w/c/w/c per round under
gpu.lock, all six loads, `--runs 60`, `--pin-pdl8 1,8,16,32 --skip-pdl8-gate`, router seed 106218864389449120, f931
case seeds, local_start per input; every arm's timed PDL set {1, 8, 16, 32} on, reference check PASS in every run;
nvidia-smi showed no process outside the lock at either round's start. Round 1 (PFX xca, chk + c/w/c/w/c) was inside
the bar, so round 2 (PFX xca_r2, c/w/c/w/c) followed at once. Logs and JSONs: `muse_decode_layer/profiles/box/xca/`
(round1/, round2/, `table_both_rounds.txt`; box `.build/prtl/xca/` and `.build/prtl/xca_r2/`). Tables by
`tools/ly_prtl_table.py` (threshold max(2 x control spread, 0.5 us) at decode 1 / verify 1, max(2 x spread, 1%)
elsewhere); net us, candidate mean minus control mean, pairs w_a-c_a, w_a-c_b, w_b-c_b, w_b-c_c.

Round 1:

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 65.49 | 64.58 | 65.15 | 64.74 | 65.20 | 0.34 | -0.62 | -0.91 -0.58 -0.42 -0.46 | 0.67 | inside |
| decode 1 full | 72.30 | 71.26 | 71.58 | 71.57 | 72.06 | 0.72 | -0.57 | -1.04 -0.32 -0.02 -0.50 | 1.44 | inside |
| decode 16 sliding | 126.70 | 126.54 | 126.77 | 127.25 | 126.90 | 0.19 | +0.11 | -0.16 -0.22 +0.48 +0.35 | 1.27 | inside |
| decode 16 full | 198.96 | 199.06 | 199.86 | 199.15 | 199.42 | 0.90 | -0.31 | +0.10 -0.80 -0.70 -0.27 | 1.99 | inside |
| decode 32 sliding | 143.36 | 143.52 | 143.46 | 143.81 | 143.55 | 0.19 | +0.21 | +0.16 +0.06 +0.35 +0.26 | 1.43 | inside |
| decode 32 full | 295.98 | 294.99 | 294.83 | 294.66 | 294.96 | 1.15 | -0.43 | -0.99 +0.16 -0.18 -0.30 | 2.95 | inside |
| verify 1 sliding | 102.85 | 101.82 | 101.98 | 102.48 | 102.83 | 0.86 | -0.40 | -1.02 -0.16 +0.50 -0.35 | 1.73 | inside |
| verify 1 full | 110.05 | 109.82 | 111.49 | 110.51 | 110.88 | 1.44 | -0.64 | -0.22 -1.66 -0.98 -0.37 | 2.88 | inside |
| verify 16 sliding | 242.96 | 241.90 | 243.26 | 244.54 | 244.45 | 1.49 | -0.33 | -1.06 -1.36 +1.28 +0.10 | 2.98 | inside |
| verify 16 full | 327.34 | 326.00 | 326.85 | 326.27 | 326.85 | 0.50 | -0.88 | -1.34 -0.85 -0.58 -0.58 | 3.27 | inside |
| verify 32 sliding | 296.48 | 296.06 | 296.34 | 296.53 | 296.16 | 0.32 | -0.03 | -0.42 -0.27 +0.19 +0.37 | 2.96 | inside |
| verify 32 full | 460.93 | 460.51 | 460.22 | 460.40 | 460.16 | 0.77 | +0.02 | -0.42 +0.29 +0.18 +0.24 | 4.60 | inside |

Round 2:

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 65.52 | 64.34 | 65.02 | 64.64 | 65.06 | 0.50 | -0.71 | -1.18 -0.69 -0.38 -0.42 | 0.99 | inside |
| decode 1 full | 71.94 | 71.04 | 71.68 | 71.54 | 72.42 | 0.74 | -0.72 | -0.90 -0.64 -0.14 -0.88 | 1.47 | inside |
| decode 16 sliding | 126.91 | 126.80 | 126.42 | 127.02 | 126.77 | 0.50 | +0.21 | -0.11 +0.38 +0.61 +0.26 | 1.27 | inside |
| decode 16 full | 200.03 | 199.23 | 200.10 | 198.99 | 199.01 | 1.09 | -0.60 | -0.80 -0.86 -1.10 -0.02 | 2.18 | inside |
| decode 32 sliding | 144.67 | 144.27 | 144.32 | 144.02 | 143.95 | 0.72 | -0.17 | -0.40 -0.05 -0.30 +0.06 | 1.44 | inside |
| decode 32 full | 294.88 | 295.15 | 295.23 | 295.04 | 295.46 | 0.58 | -0.09 | +0.27 -0.08 -0.19 -0.42 | 2.95 | inside |
| verify 1 sliding | 102.56 | 102.35 | 102.74 | 102.46 | 102.35 | 0.38 | -0.14 | -0.21 -0.38 -0.27 +0.11 | 0.77 | inside |
| verify 1 full | 110.67 | 110.43 | 110.82 | 110.34 | 110.30 | 0.51 | -0.21 | -0.24 -0.38 -0.48 +0.03 | 1.02 | inside |
| verify 16 sliding | 243.15 | 242.62 | 244.26 | 243.49 | 243.52 | 1.10 | -0.59 | -0.53 -1.63 -0.77 -0.03 | 2.44 | inside |
| verify 16 full | 327.38 | 327.60 | 327.73 | 327.39 | 327.84 | 0.46 | -0.15 | +0.22 -0.13 -0.34 -0.45 | 3.28 | inside |
| verify 32 sliding | 296.67 | 297.02 | 297.95 | 297.41 | 297.17 | 1.28 | -0.05 | +0.35 -0.93 -0.54 +0.24 | 2.97 | inside |
| verify 32 full | 460.26 | 460.00 | 461.18 | 459.70 | 460.45 | 0.93 | -0.78 | -0.26 -1.18 -1.49 -0.75 | 4.61 | inside |

Pooled over both rounds (6 control runs, 4 candidate runs, 24 pairs per cell):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 65.24 | 0.50 | 64.57 | -0.67 | 0.99 | 24/24 | inside |
| decode 1 full | 72.00 | 0.83 | 71.35 | -0.65 | 1.66 | 24/24 | inside |
| decode 16 sliding | 126.74 | 0.50 | 126.90 | +0.16 | 1.27 | 7/24 | inside |
| decode 16 full | 199.56 | 1.14 | 199.11 | -0.45 | 2.27 | 17/24 | inside |
| decode 32 sliding | 143.89 | 1.31 | 143.90 | +0.02 | 2.62 | 11/24 | inside |
| decode 32 full | 295.22 | 1.15 | 294.96 | -0.26 | 2.95 | 15/24 | inside |
| verify 1 sliding | 102.55 | 0.86 | 102.28 | -0.27 | 1.73 | 18/24 | inside |
| verify 1 full | 110.70 | 1.44 | 110.28 | -0.43 | 2.88 | 18/24 | inside |
| verify 16 sliding | 243.60 | 1.49 | 243.14 | -0.46 | 2.98 | 15/24 | inside |
| verify 16 full | 327.33 | 0.99 | 326.82 | -0.51 | 3.27 | 16/24 | inside |
| verify 32 sliding | 296.79 | 1.79 | 296.76 | -0.04 | 3.58 | 12/24 | inside |
| verify 32 full | 460.53 | 1.02 | 460.15 | -0.38 | 4.61 | 17/24 | inside |

Reading. The two cells that run the changed kernel (the cgemv slice is launched only at T = 1; T = 8, 16 and 32 use
the sk2 slice) gain 0.67 / 0.65 us with every one of the 24 pairs in the candidate's favour in both, the same
magnitude the stamps measured on the instrumented graph (0.83) and the same in both rounds (0.62 / 0.57 then 0.71 /
0.72); the ten unchanged cells are the session's null (7 to 18 of 24 pairs, -0.51 to +0.16 us). By the table's
2x-spread rule the cells are "inside"; by sign count a 24/24 result in two independent cells is not noise. The gain
is 1.0 / 0.9% of the cell, about 0.14% of the board score at the directive's 0.212% per us.

### Decision

The post-MoE tail at decode 1 is separated: wait-return latency 0.5 to 0.6 us (sliding; at full the wait is reached
only after the 1.63 us weight stream, which the MoE's short final pass no longer hides), x landing for every lane 1.5
to 1.8 us (the largest part), math 0.7 to 0.9, barrier and store 0.13, end spread 0.7 to 0.9, finish 1.6 (wait 0.4, two
serial loads 0.9). ly_xca takes the largest part: -0.67 / -0.65 us at decode 1 sliding / full, 24/24 pairs each,
bitwise identical output, 9 SASS lines changed (cache operator of the x loads), no register or spill change, every
fence, barrier, wait and trigger as in the base, PDL guard clean at every T, CG_GATE line equal to the control's.
Judge check: recommended only as a stacking ingredient or as the repeat base switch's content, not for a standalone
keep-rule decision: 0.14% of score is below one judge draw's spread (ly_rfin, 0.15%, was indistinguishable over two
draws, ledger 07:18 PT), so one check can establish validity (13/13, guards) but cannot rank it against the row.
Left open, each at most 1 us: the slice's launch 1.3 to 1.7 us after the MoE's trigger (PDL launch latency, the
trigger is the MoE lane's); the remaining x tail (1.1 to 1.3 us with L1, a cluster multicast of x would cut L2
requests further and is untested, ruled-out.md row on x staging notes it); the finish's two serial loads after its
wait (0.9 us, hypothesis: issue carry and P together). No ruled-out row: the candidate did not lose.

## 2026-11-04 - ly_rfx stacked A/B (box)

Revival lane, one bounded measurement: the stacked candidate `muse_decode_layer/cand/ly_rfx.py` (SHA256
60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda) against the public file
`muse_decode_layer/cand/ly_fabq.py` (3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555) on the box,
with the same driver, flags and rounds the two component lanes used ("Verify segment switches (box)" for ly_rfin,
"Decode 1 post-MoE tail (box)" for ly_xca). Box only (okc-h200, 16:30 to 16:34 UTC), every run under
`/root/okc/.build/box/gpu.lock`, no judge job, no candidate building. `nvidia-smi` before the launch (16:30:06 UTC):
0% utilization, 531 MiB used, no compute process, no box_run / ly_prtl_ab / timeline process on the host; after the
last run (16:33:51 UTC) again no compute process, so no other lane's run shared the GPU during either round. Logs:
`/root/okc/.build/prtl/rfx/` (round 1 with the chk step) and `/root/okc/.build/prtl/rfx_r2/` (round 2), copied with
their JSONs, session logs and the pooled table to `muse_decode_layer/profiles/box/rfx/` (`round1/`, `round2/`,
`table_both_rounds.txt`).

### Diff confirmation

`ly_rfx.py` is `ly_rfin.py` (c42c12390ce22538260ebea22820002a2e8acaa6f7526920396b86a4b94cc029, hash verified in the
worktree and on the box) plus two lines: the header comment on line 1 and the `_PDL8_CG_EDITS` line (19031 lines in
both files after the header is dropped; one line differs, 18925 of ly_rfin). `ly_xca.py` against `ly_fabq.py` is the
same shape (18976 lines, one line differs, 18870 of ly_fabq). Checked in Python: the old `_PDL8_CG_EDITS` line is
byte-identical between the two pairs, the new line is byte-identical between the two pairs, and the new line equals
the old line with its three `__ldcg` tokens replaced by `__ldca` (the two compute-thread x loads and the escape
warp's x element load, all after the slice cgemv's `griddepcontrol.wait`). Nothing else differs: the `OKC_RFIN=1
OKC_RFINQ=2` MAIN build defines, the fences, barriers, releases, PDL set and launch attributes are ly_rfin's.

### Correctness (chk step, `bx_box_run.py --runs 5`, six loads, 16:30 UTC, `profiles/box/rfx/round1/chk.log`)

`FORK_MOE_ROUTE native repaired: cooperative+BF16, MFIN/DF/YBF/PDL off`; `CG_GATE unpooler cg_vs_sk2 1.21e-04
cg_vs_fp64 0.00e+00 sk2_vs_fp64 1.21e-04 equal 1023/1024 ok` (the same line as the control's prepare in every
control arm of both rounds and as ly_xca's and the base's earlier logs); `G_RELEASE_STARTUP_OK native (128, 32, 33,
256, 1, 8, 16) 56 verified_replays` (the startup comparison of the MAIN build against the base's path at M = 128, 32,
33, 256 and T = 1, 8, 16); reference check PASS at all 12 cells; timed PDL set {1, 8, 16, 32} on. Every one of the
ten A/B arms also printed the same CG_GATE and G_RELEASE_STARTUP_OK lines and passed the check at all 12 cells
(12 PASS / 0 FAIL per log, timed PDL {1, 8, 16, 32} on in each).

### Paired A/B against ly_fabq (box, two rounds)

`tools/ly_prtl_ab.sh` with CTL=ly_fabq.py, CAND=ly_rfx.py: five `bx_box_run.py` processes c/w/c/w/c per round under
gpu.lock, all six loads, `--runs 60`, `--lock /root/okc/.build/box/gpu.lock --ls-cache
/root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext --pin-pdl8 1,8,16,32 --skip-pdl8-gate
--router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json`, f931 case seeds,
local_start per input (60 cache hits, 0 computed). Round 1 16:30:46 to 16:32:17 UTC (`.build/prtl/rfx/`), round 2
16:32:35 to 16:33:48 UTC (`.build/prtl/rfx_r2/`), back to back with nothing else on the lock. Tables by
`tools/ly_prtl_table.py` (threshold max(2 x control spread, 0.5 us) at decode 1 / verify 1, max(2 x spread, 1% of the
control mean) elsewhere); net us, candidate mean minus control mean, pairs w_a-c_a, w_a-c_b, w_b-c_b, w_b-c_c.

Round 1:

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 65.12 | 64.64 | 65.23 | 64.37 | 65.20 | 0.11 | -0.68 | -0.48 -0.59 -0.86 -0.83 | 0.50 | win |
| decode 1 full | 71.95 | 70.99 | 71.76 | 70.82 | 71.66 | 0.29 | -0.89 | -0.96 -0.77 -0.94 -0.85 | 0.58 | win |
| decode 16 sliding | 127.18 | 127.04 | 126.58 | 127.15 | 126.85 | 0.61 | +0.23 | -0.14 +0.46 +0.58 +0.30 | 1.27 | inside |
| decode 16 full | 198.64 | 199.39 | 199.25 | 198.70 | 200.54 | 1.90 | -0.43 | +0.75 +0.14 -0.54 -1.84 | 3.81 | inside |
| decode 32 sliding | 144.10 | 144.00 | 143.79 | 144.06 | 144.45 | 0.66 | -0.08 | -0.10 +0.21 +0.27 -0.38 | 1.44 | inside |
| decode 32 full | 295.87 | 295.28 | 294.61 | 294.98 | 295.94 | 1.33 | -0.34 | -0.59 +0.67 +0.37 -0.96 | 2.95 | inside |
| verify 1 sliding | 102.46 | 102.77 | 101.97 | 102.30 | 102.21 | 0.50 | +0.32 | +0.30 +0.80 +0.34 +0.10 | 0.99 | inside |
| verify 1 full | 111.12 | 110.51 | 110.32 | 110.32 | 110.19 | 0.93 | -0.13 | -0.61 +0.19 +0.00 +0.13 | 1.86 | inside |
| verify 16 sliding | 243.14 | 243.42 | 243.74 | 242.75 | 243.25 | 0.61 | -0.29 | +0.29 -0.32 -0.99 -0.50 | 2.43 | inside |
| verify 16 full | 327.30 | 326.34 | 327.62 | 327.26 | 327.95 | 0.66 | -0.82 | -0.96 -1.28 -0.35 -0.69 | 3.28 | inside |
| verify 32 sliding | 298.30 | 294.45 | 296.77 | 294.03 | 296.24 | 2.06 | -2.86 | -3.86 -2.32 -2.74 -2.21 | 4.13 | inside |
| verify 32 full | 459.44 | 457.97 | 459.68 | 457.95 | 460.35 | 0.91 | -1.86 | -1.47 -1.71 -1.73 -2.40 | 4.60 | inside |

Round 2:

| cell | c_a | w_a | c_b | w_b | c_c | spread | w - c | pairs | thr | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| decode 1 sliding | 65.28 | 64.51 | 65.65 | 64.06 | 65.63 | 0.37 | -1.23 | -0.77 -1.14 -1.58 -1.57 | 0.74 | win |
| decode 1 full | 71.66 | 70.64 | 71.68 | 71.10 | 71.87 | 0.21 | -0.87 | -1.02 -1.04 -0.58 -0.77 | 0.50 | win |
| decode 16 sliding | 126.43 | 126.90 | 126.99 | 126.40 | 126.69 | 0.56 | -0.06 | +0.46 -0.10 -0.59 -0.29 | 1.27 | inside |
| decode 16 full | 199.58 | 199.28 | 199.71 | 199.62 | 199.34 | 0.37 | -0.10 | -0.30 -0.43 -0.10 +0.27 | 2.00 | inside |
| decode 32 sliding | 144.48 | 143.33 | 143.78 | 144.58 | 143.79 | 0.70 | -0.06 | -1.15 -0.45 +0.80 +0.78 | 1.44 | inside |
| decode 32 full | 294.99 | 295.28 | 295.87 | 295.73 | 295.49 | 0.88 | +0.05 | +0.29 -0.59 -0.14 +0.24 | 2.95 | inside |
| verify 1 sliding | 102.16 | 101.86 | 101.90 | 102.00 | 102.32 | 0.42 | -0.20 | -0.30 -0.05 +0.10 -0.32 | 0.83 | inside |
| verify 1 full | 110.64 | 110.82 | 110.98 | 110.46 | 111.12 | 0.48 | -0.27 | +0.18 -0.16 -0.51 -0.66 | 0.96 | inside |
| verify 16 sliding | 242.90 | 243.22 | 244.19 | 243.92 | 242.90 | 1.30 | +0.24 | +0.32 -0.98 -0.27 +1.02 | 2.59 | inside |
| verify 16 full | 327.71 | 326.45 | 327.55 | 327.76 | 327.90 | 0.35 | -0.62 | -1.26 -1.10 +0.21 -0.14 | 3.28 | inside |
| verify 32 sliding | 297.10 | 295.04 | 297.33 | 295.06 | 297.06 | 0.27 | -2.11 | -2.06 -2.29 -2.27 -2.00 | 2.97 | inside |
| verify 32 full | 460.51 | 457.52 | 460.50 | 458.06 | 460.58 | 0.08 | -2.74 | -2.99 -2.98 -2.43 -2.51 | 4.61 | inside |

Pooled over both rounds (6 control runs, 4 candidate runs, 24 pairs per cell; threshold from the spread of the six
controls), with the two component lanes' pooled deltas for the stacking comparison (ly_xca 24 pairs, ly_rfin 54
pairs, from their own sections):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict | ly_xca alone | ly_rfin alone |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| decode 1 sliding | 65.35 | 0.53 | 64.40 | -0.96 | 1.06 | 24/24 | inside | -0.67 (24/24) | +0.18 (14/54) |
| decode 1 full | 71.77 | 0.29 | 70.89 | -0.88 | 0.58 | 24/24 | win | -0.65 (24/24) | -0.25 (42/54) |
| decode 16 sliding | 126.79 | 0.75 | 126.87 | +0.09 | 1.50 | 10/24 | inside | +0.16 | +0.15 |
| decode 16 full | 199.51 | 1.90 | 199.25 | -0.26 | 3.81 | 14/24 | inside | -0.45 | -0.28 |
| decode 32 sliding | 144.06 | 0.70 | 143.99 | -0.07 | 1.44 | 12/24 | inside | +0.02 | +0.18 |
| decode 32 full | 295.46 | 1.33 | 295.32 | -0.15 | 2.95 | 16/24 | inside | -0.26 | -0.28 |
| verify 1 sliding | 102.17 | 0.56 | 102.23 | +0.06 | 1.12 | 12/24 | inside | -0.27 | -0.26 |
| verify 1 full | 110.73 | 0.93 | 110.53 | -0.20 | 1.86 | 15/24 | inside | -0.43 | -0.20 |
| verify 16 sliding | 243.35 | 1.30 | 243.33 | -0.02 | 2.59 | 12/24 | inside | -0.46 | -0.44 (35/54) |
| verify 16 full | 327.67 | 0.66 | 326.95 | -0.72 | 3.28 | 20/24 | inside | -0.51 | -0.71 (42/54) |
| verify 32 sliding | 297.13 | 2.06 | 294.64 | -2.49 | 4.13 | 24/24 | inside | -0.04 | -2.75 (54/54) |
| verify 32 full | 460.18 | 1.14 | 457.88 | -2.30 | 4.60 | 24/24 | inside | -0.38 | -1.48 (54/54) |

Reading (measurements on the box, paired, no judge number involved). The components stack: the four cells each
component owns show the candidate faster in every one of the 24 pairs. Decode 1 sliding / full -0.96 / -0.88 us
(ly_xca alone -0.67 / -0.65; round 2's sliding -1.23 pulled the pooled mean past the component's), verify 32
sliding / full -2.49 / -2.30 (ly_rfin alone -2.75 / -1.48, both inside ly_rfin's three-round range of -1.95 to
-3.49 and -0.95 to -2.17), verify 16 full -0.72 with 20 of 24 pairs (ly_rfin -0.71), verify 16 sliding -0.02 with
12 of 24 (ly_rfin's weakest cell, -0.44 at 35 of 54, so this round reads it as null). The six cells neither component
touches (decode 16 / 32, verify 1) sit within 0.26 us of the control with 10 to 16 of 24 pairs below, the session's
null. By the 2x-spread rule decode 1 full is a win in both rounds and pooled, decode 1 sliding a win in both rounds
and inside pooled (the six-control spread 0.53 sets a 1.06 bar), and verify 32 is inside because its control spread
(2.06 us sliding, from one 298.30 control in round 1) sets a 4.1 to 4.6 us bar; the sign counts (24/24 at four cells,
across two rounds and both cells of each load) are the evidence of the gain, as in both component sections.

Value if the box deltas carry to the judge (hypothesis, the Box rule forbids reading them as a judge prediction):
decode 1 about 0.9 us at 0.212% per us (0.19%), verify 32 about 2.4 us at 0.046% (0.11%), verify 16 full 0.7 us at
0.062% (0.04%): about 0.3% of the layer score, twice either component alone and still below the 1.3% spread of a
single judge draw (ly_rfin's two draws, ledger 07:18 PT). The judge check of ly_rfx (job 92543dcfd8164b3a, ledger
08:28 PT) was already registered by the lead before this run; this section does not dispatch one. Decision: ly_rfx is
correct on the box (12 / 12, startup comparison OK, CG_GATE equal to the control's), stacks the two measured gains
without interference, and is the finalist from these two lanes; no ruled-out row.

## 2026-11-04 - Verify 1 timeline (box)

Verify 1 lane (T = 8, one request of 8 tokens). Question: where does the step go at verify 1, sliding and full, kernel
by kernel against its byte floor and edge by edge, and which exposed excess can one change take? Base:
muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda, the
public layer row, job caed0290397b44a7), never edited. Box only (okc-h200, 17:18 to 17:35 UTC), every run under
gpu.lock, no judge check. Logs, traces and tables: `muse_decode_layer/profiles/box/v1tl/` (box `.build/v1tl/`,
`.build/prtl/v1un*`). Driver for every candidate: `tools/ly_v1_ab.sh` (a gate run with the file's own in-prepare
PDL8 guard, then `tools/ly_prtl_ab.sh` twice against ly_rfx at all six loads, then a candidate timeline).

### Timeline at T = 8 (measurement)

`LY_TRACE_DIR=.build/v1tl/tl_rfx tools/ly_tl_run.py muse_decode_layer muse_decode_layer/cand/ly_rfx.py --loads
"verify 1" --runs 20 --kernels --pin-pdl8 1,8,16,32 --skip-pdl8-gate --router-seed 106218864389449120 --judge
g1b4_pdl8d.f931abf568544018.json` (lock, ls-cache and ext-cache as in the decode 1 section); reference check PASS at
both cells, net 101.47 / 115.07 us (sliding / full, 20 runs). Medians over the 10 profiled cold replays
(`tools/ly_tl_parse.py`). Critical path = this kernel's end minus the previous kernel's end; gap = start minus the
previous end (negative: PDL overlap). Floors at 4.44 TB/s as in the decode 1 section (the T = 8 x rows add under
0.05 us anywhere; b1 at T = 8 moves about 0.8 MB, 0.2 us); the MoE floor is 6.38 us per distinct local expert at the
box's per-input mean of 7.38 / 7.92 distinct (docs/box.md, "Routing"), so it is approximate for these 10 inputs.

Verify 1 sliding (span 105.50 us, sum of durations 121.89):

| # | kernel | start | dur | end | gap | critical path | floor | over floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | qkv sk2_kernel_narrow<1,8,3,...> (plain launch) | 0.00 | 9.52 | 9.52 | - | 9.52 | 6.39 | 3.13 |
| 1 | attn_kernel<8,false,15> | 7.42 | 8.62 | 16.27 | -1.95 | 6.75 | 0.12 | 6.63 |
| 2 | o sk2<1,8,4,1,4,...> | 9.94 | 10.06 | 20.02 | -6.24 | 3.75 | 2.84 | 0.91 |
| 3 | b1 glue_cluster_boundary<4,32,...> | 17.87 | 4.86 | 22.64 | -2.24 | 2.62 | 0.2 | 2.4 |
| 4 | pooler + router sk2_kernel_gle | 20.54 | 17.79 | 38.30 | -2.16 | 15.66 | 13.26 | 2.40 |
| 5 | _prq_sl_tile_route_cg | 36.40 | 3.65 | 40.14 | -1.98 | 1.84 | 0 | 1.84 |
| 6 | moe_fused<false> (TINY2, cooperative, no programmatic attribute) | 40.19 | 58.99 | 98.82 | +0.05 | 58.68 | about 47 | about 11.6 |
| 7 | slice sk2<1,8,1,1,8,2,4> (bit 29: plain launch) | 99.04 | 4.42 | 103.42 | **+0.22** | 4.60 | 1.42 | 3.18 |
| 8 | _sl_cluster_finish_cg | 101.01 | 4.50 | 105.50 | -2.43 | 2.08 | 0.05 | 2.03 |

Verify 1 full (span 119.03 us, sum of durations 144.18):

| # | kernel | start | dur | end | gap | critical path | floor | over floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | qkv | 0.00 | 9.73 | 9.73 | - | 9.73 | 6.39 | 3.34 |
| 1 | attn_kernel<8,false,11> | 7.38 | 16.19 | 23.66 | -2.32 | 13.93 | 4.61 | 9.32 |
| 2 | o sk2 | 10.46 | 17.15 | 27.62 | -13.23 | 3.96 | 2.84 | 1.12 |
| 3 | b1 glue | 24.59 | 5.58 | 30.26 | -2.98 | 2.64 | 0.2 | 2.4 |
| 4 | pooler + router gle | 28.10 | 17.76 | 45.92 | -2.11 | 15.66 | 13.26 | 2.40 |
| 5 | route | 44.02 | 3.66 | 47.52 | -2.05 | 1.60 | 0 | 1.60 |
| 6 | moe_fused<false> | 47.60 | 64.24 | 112.34 | +0.10 | 64.82 | about 50.5 | about 14.3 |
| 7 | slice sk2 | 112.55 | 4.43 | 116.96 | **+0.26** | 4.62 | 1.42 | 3.20 |
| 8 | finish | 114.58 | 4.45 | 119.03 | -2.40 | 2.07 | 0.05 | 2.02 |

Edges. Six of the eight edges are PDL edges whose secondary starts 1.95 to 13.2 us before its producer ends (the
T = 8 set attn, o, b1, pool, route, fin of PDL8D_EDGES). Two are serial: route -> MoE (+0.05 / +0.10 us; the
cooperative MoE has no programmatic attribute, the MoE lane's) and MoE -> slice (+0.22 / +0.26 us). The second is
the difference from decode 1: at T = 1 the slice is the cgemv, a PDL secondary of the same MoE (cg_set_pdl) that
enters 1.3 us after the MoE's trigger and holds its weights before the MoE ends ("Decode 1 post-MoE tail"); at
T = 8 the slice is sk2 and `_pdl8_bfp_pdl` sets bit 29 (no programmatic attribute) for role "un" at every T, so it
starts after the MoE has completed and then streams all 6.3 MB of its weights. Slice + finish after the MoE's end
is 6.68 / 6.69 us for a 1.47 us floor.

Largest excesses over floor (critical path minus floor, sliding / full) and their owners: MoE 11.6 / 14.3 (ramp,
handoff and tail, MOE-TL-01; MoE lane), attention 6.6 / 9.3 (attention lane), qkv 3.1 / 3.3 (bf16 lane; first
kernel after the flush), slice 3.2 / 3.2 and finish 2.0 / 2.0 (this lane's scope: the edge and the tail), b1 2.4,
pooler + router 2.4 (bf16 lane, "Pooler + router remaining gap"), route 1.8 / 1.6.

### Why ly_xca and the finish loads do not move verify 1 (step 2)

- ly_xca changes the cgemv source only (three `__ldcg` -> `__ldca` tokens in `_PDL8_CG_EDITS`). The cgemv runs only
  at m = 1 (the restored `cg_pick` is keyed (layer, 1); PIN_PICK keeps (8, "un") on bfp), so at T = 8 the edited
  kernel is never launched; ly_xca's verify 1 deltas (-0.27 / -0.43, 18/24) and ly_rfx's (+0.06 / -0.20) are the
  session null. The T = 8 slice reads x by TMA into shared memory, every one of its 128 CTAs taking all 64 KB of x
  across its 4 stages (8 MB of L2 traffic), after its wait; that cost is inside the slice's 4.4 us duration and was
  not separated here (hypothesis: under 1 us, L2 delivering 8 MB).
- The finish at T = 8 is 64 programs of 16 warps with BC = 1024 (SL_CLUSTER_CONFIG[8]; T = 1: 32 programs of 8 warps,
  BC = 256). Its critical path after the slice is 2.08 / 2.07 us (T = 1: 1.47 / 1.94), the same serial shape after
  the wait (carry row and slice row loads, two dependent 1024-wide reductions, the stores) with 4x the outputs per
  program. Its one pre-wait move (finpre: the NEXT weight before gdc_wait) was already null at verify 1 (+0.19 /
  +0.02, "finpre" row of the blueprint table). Not stamped at T = 8 in this lane.
- So at T = 8 the slice's excess is not the x load the decode 1 lane fixed but the serial edge in front of it:
  launch after the MoE's completion, then the full weight stream. That is the item taken below.

### Candidate ly_v1un: the T = 8 slice as a PDL secondary of the MoE

Evidence that reopens the edge (docs/ruled-out.md rows "M-B slice 1" and "G Tiny/Tiny2 expert->single sliced-unpool
PDL edge"): the M-B row's revisit trigger has landed (helper0, the TINY2 build at T = 8, runs
`griddepcontrol.launch_dependents` unconditionally at each CTA's phase B end, before its grid barrier and final
pass), and the G row's missing budget is now measured: the slice starts 0.22 / 0.26 us after the MoE's end, and the
MoE's CTAs retire over 52.6 to 53.5 us after a last trigger at 50.8 (MOE-TL-01 at T = 8), so a programmatic slice can
take each SM as its MoE CTA leaves (the two CTAs cannot share an SM: >= 115,712 + 118,240 B of shared memory) and
issue its first stages before its wait.

`tools/dl_ly_v1un_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_v1un.py` (SHA256
88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145). Two lines change, T = 8 only:

- `PDL8D_EDGES = ("attn", "o", "b1", "pool", "route", "fin", "un")`: the T = 8 in-prepare guard (`_pdl8h_t8`, the
  paired A/B and the 2000-replay stress against the PDL-off output; a FAIL turns T = 8 off) covers the new edge with
  the other six.
- `_pdl8_bfp_pdl`: `(1 << 29) if role == "qkv" or (role == "un" and T != 8) or not _pdl8_edge(role)`; T = 1, 16 and
  32 keep bit 29 on "un".

No kernel source changes, so there is no SASS diff: the sk2 binary is the base's and `pdl` is a kernel argument. The
kernel's own PDL path does the ordering: weights (prepare-time constants) and the escape table issued before the
wait, every thread's `griddepcontrol.wait`, then `fence.proxy.async.global` (upstream generic stores of x -> TMA
reads), then the x tiles; no global write before the wait. `griddepcontrol.wait` returns only when the whole MoE grid
has completed and its memory operations are performed and visible to the waiting grid (PTX ISA 8.7, 9.7.13.13
griddepcontrol). The MoE stays the primary with its cooperative launch, trigger, fences and grid barrier; its CTAs
are all resident before any secondary CTA can launch, so its progress does not depend on the slice. The finish stays
a PDL secondary of the slice. P = BFP_PDL_N = 2 pre-wait stages (of the CTA's 4).

Gate run (the file's own prepare, no pin, `gate_v1un.log`): `PDL8_AB T=8 full edges=attn+o+b1+pool+route+fin+un:
off 132.13 on 125.76 us, saving 6.21, wins 9/9`; sliding off 124.99 on 117.89, saving 7.17, 9/9; `PDL8H_GUARD T=8`
full and sliding: ids mismatches 0 / 0, on drift equal to the off arm's (7.25e-03 / 6.89e-03 full, 6.33e-03 /
6.09e-03 sliding): pass, T = 8 on; the T = 1 / 16 / 32 guards pass; reference check PASS at both verify 1 cells.
Correctness (chk step, six loads, `--runs 5`): PASS 12/12, `G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1, 8, 16)
56 verified_replays`, `CG_GATE ... equal 1023/1024 ok` (the same line as the control's).

Paired A/B against ly_rfx (`tools/ly_prtl_ab.sh`, CTL=ly_rfx.py, c/w/c/w/c per round, 60 runs, PDL pinned on at
every T, router seed 106218864389449120, f931 case seeds, local_start per input; three rounds 17:23 to 17:35 UTC).
Per round, verify 1 sliding / full w - c: -0.99 / -1.78 us (4/4 and 4/4 pairs), -1.56 / -1.29 (3/4, 4/4), -1.74 /
-1.13 (4/4, 4/4). Pooled over the three rounds (9 control runs, 6 candidate runs, 54 pairs per cell;
`profiles/box/v1tl/v1un/table_three_rounds.txt`):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.60 | 0.94 | 64.59 | -0.01 | 1.89 | 26/54 | inside |
| decode 1 full | 70.87 | 0.59 | 70.78 | -0.10 | 1.18 | 33/54 | inside |
| decode 16 sliding | 126.86 | 1.15 | 126.67 | -0.19 | 2.30 | 36/54 | inside |
| decode 16 full | 199.73 | 1.66 | 199.14 | -0.59 | 3.33 | 42/54 | inside |
| decode 32 sliding | 143.98 | 1.26 | 144.01 | +0.03 | 2.53 | 32/54 | inside |
| decode 32 full | 295.34 | 2.19 | 295.40 | +0.06 | 4.38 | 25/54 | inside |
| verify 1 sliding | 102.31 | 2.16 | 100.89 | **-1.43** | 4.32 | **54/54** | inside |
| verify 1 full | 110.55 | 1.38 | 109.15 | **-1.40** | 2.75 | **54/54** | inside |
| verify 16 sliding | 242.67 | 1.79 | 242.79 | +0.12 | 3.58 | 26/54 | inside |
| verify 16 full | 326.57 | 2.35 | 326.08 | -0.49 | 4.70 | 40/54 | inside |
| verify 32 sliding | 294.58 | 2.08 | 294.60 | +0.01 | 4.16 | 28/54 | inside |
| verify 32 full | 458.56 | 2.14 | 458.45 | -0.11 | 4.59 | 33/54 | inside |

The first two rounds pooled (the 24-pair form) read -1.27 / -1.54 us at 24/24 and 24/24. The ten unchanged cells
(the code differs only at T = 8) sit within 0.6 us of the control, the session null. By the 2x-spread rule verify 1
full is a win in round 1 and verify 1 sliding a win in round 3, and the pooled cells are inside (one 103.62 control in
round 2 sets the sliding bar); by sign count, 54 of 54 pairs in both cells across three rounds is the evidence of
the gain.

Candidate timeline (same instrument, `profiles/box/v1tl/tl_v1un/`, net 101.36 / 114.42 us): the slice now starts
1.06 / 1.30 us before the MoE's end (base: 0.22 / 0.26 after), its end follows the MoE's by 3.36 / 3.28 us (base
4.60 / 4.62), and the finish's critical path grows 2.08 -> 2.40 / 2.07 -> 2.48 (it waits on a slice that ends sooner
and gives part back). Slice + finish after the MoE: 5.76 / 5.76 us against 6.68 / 6.69, -0.92 / -0.93 us on the
kernel timeline; the rest of the A/B delta is inside the MoE's input-to-input spread.

### Candidate ly_v1un4: all four slice stages before the wait (stacked on ly_v1un, no gain)

`tools/dl_ly_v1un4_patch.py` (base ly_v1un, pin checked) writes `muse_decode_layer/cand/ly_v1un4.py` (SHA256
da9e30b5275b2aca0938464245221dc53b420a6b47d4f7a7d4d0111034090aba): one line, `PCX_CFG[("un", 8)] = (4, 0, 0, 0)`, so
the T = 8 slice issues its whole K of weights (4 x 12 KB per CTA) before the wait; it is read only on the PDL8 path
at T = 8. Gate run: PDL8_AB T = 8 savings 6.88 / 6.72 us (9/9), PDL8H_GUARD pass at both cells, ids mismatches 0;
the sliding on-drift 6.33e-03 / 6.09e-03 sat above that run's off-drift 3.16e-03 / 3.98e-03 and inside the guard
bound (max(2e-2, off)); the same 6.33e-03 / 6.09e-03 was ly_v1un's off-drift, so the off arm's own drift varies by
that much between prepares. Check PASS 12/12, startup OK, CG_GATE equal to the control's. A/B against ly_rfx, two
rounds (`profiles/box/v1tl/v1un4/table_both_rounds.txt`): verify 1 sliding / full -0.95 (23/24) / -0.82 (24/24),
per round -1.00 / -0.78 and -0.89 / -0.87; other cells -0.50 to +0.22. Timeline (`tl_v1un4/`): the slice starts
1.22 / 1.51 us before the MoE's end but runs 5.15 us instead of 4.42 and ends 4.03 / 3.61 us after it (ly_v1un
3.36 / 3.28); its producer loop is empty, so its `launch_dependents` fires at once and the finish launches earlier
(finish critical path 1.92 / 2.13). Hypothesis for the longer slice: all 128 CTAs' 48 KB requests at once behind the
MoE's exit, and the early finish programs sharing the SMs. Smaller than ly_v1un's gain in every round; not kept,
ruled-out row.

### Decision

ly_v1un is the lane's candidate: verify 1 sliding / full -1.43 / -1.40 us on the box with 54/54 pairs in each cell
over three rounds (24/24 and 24/24 over the first two), nothing else beyond noise, guard clean at T = 8 with the new
edge, check PASS 12/12. Value if the box delta carries to the judge (hypothesis; the Box rule forbids reading it as a
prediction): about 1.4 us x 0.144% per us = 0.2% of the layer score, below one draw's 1.3% spread, so one check
establishes validity (13/13 and the PDL8H guard on the judge's prepare with seven T = 8 edges) rather than rank. It
touches only the T = 8 path and two Python lines that the moe and layer lanes' T = 1 edits do not touch, so it stacks
with them by construction. A ledger registration line proposing the check is written; this lane dispatched no judge
job. Left open at T = 8: the slice's TMA x traffic (8 MB of L2 reads for 64 KB of x) and the finish's 2.1 to 2.5 us
after the slice, both unstamped.

## 2026-11-04 - Decode 1 finish and slice launch (box)

Layer lane (relaunched 09:17 PT; the 08:38 launch died before writing anything). Question: two leads left by the
post-MoE tail section at decode 1. (a) The finish kernel (1.5 to 1.6 us) shows two serial steps after its wait (P,
then carry): can the second load be issued earlier, or b1's operands be loaded before the wait? (b) The slice cgemv
enters 1.3 to 1.6 us after the MoE's trigger: does the trigger position or the slice's pre-wait prologue leave work
that can move ahead of the wait? Base: `muse_decode_layer/cand/ly_rfx.py` (SHA256
60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda, the public row caed0290397b44a7, never edited).
Box only (okc-h200, 17:24 to 17:40 UTC), every GPU process under `/root/okc/.build/box/gpu.lock` (bx_box_run.py
`--lock`); the verify1 lane's ly_v1un A/B shared the lock. No judge check. The stamped instrument
`muse_decode_layer/cand/ly_pmt_rfx.py` (a1b60fdb...) is `tools/dl_ly_pmt_patch.py --base ly_rfx --xld __ldca`; the
untracked copy the dead launch left in worktree agent-aab9c6edc6da3f995 is byte-identical to it. `tools/ly_box_pmt.py`
now also prints the last MoE CTA's phase A end (`moe_pa_end`) and the slice's first entry after it.

### (a) The finish: what the SASS does (measurement)

The PTX of `_sl_cluster_finish_cg` (T = 1, 8 warps) issues all five loads after `griddepcontrol.wait` back to back
(carry v4, tail, outcarry, NEXT, P v2). ptxas does not keep that order: in the cubin the P load
(`LDG.E.64.STRONG.GPU`) follows ACQBULK, the carry load (`LDG.E.128.STRONG.GPU`) is issued only after the second
`SHFL.BFLY` of the first reduction (which consumes P, so after P has landed), and tail / outcarry / NEXT after that
reduction's shared-memory store. The two loads are therefore serial in the binary even though the source issues them
together; the 0.48 us that the stamps put between "P consumed" and "invnext formed" holds whatever of carry's L2
latency reduction 1 does not cover, plus reduction 2. Forcing the carry load ahead of P inside the post-wait block
would need inline assembly in Triton; loading b1's operands before the wait does not.

Per-operand proof for loading b1's outputs before the finish's wait (the predecessor's output P stays behind it):
`s1` (CARRY) and `tail` are written only by `_pdl8_b1` in `_pdl8_sl_tile_after_attention`; both are allocated at the
call's top and live to its return, and no later kernel of the call writes them (the pooler writes pooled / logits,
route writes xm / ids / lids / wts, the MoE writes m, the slice writes P). At T <= 8 the MoE is launched by the
helper0 launcher with `cudaLaunchCooperativeKernel` and no programmatic attribute, so it starts only after all
earlier work in the stream has completed; route completes only after its `gdc_wait` (every route program waits),
so after the pooler has completed, and the pooler only after b1. The slice can launch only after the MoE's CTAs
have triggered (after the MoE started), and the finish only after the slice's CTAs have triggered, so b1 has
completed before any finish CTA exists; the loads stay `ld.global.cg` (L2, no L1 line). This holds at T = 8 too,
including on ly_v1un (T = 8 slice as a PDL secondary of the MoE), since the MoE launch is unchanged there.

### Candidate ly_finb1: b1's operands before the finish's wait

`tools/dl_ly_finb1_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_finb1.py` (SHA256
dc6e2af57bb43d43d97a3d3a7e4f93ddb43d67530fbcee8a34afa0ead59e3c11): in `_sl_cluster_finish_cg` only (ly_rfx lines
18286-18292, the T <= 8 PDL8 finish), the carry, tail and outcarry loads move above the `if USE_PDL:` wait block; P
and NEXT stay after it. No wait, trigger, fence, launch attribute or other kernel changes; the T = 1 and T = 8
instances both change (not T-gated; the proof above covers both). SASS (box Triton cache, cuobjdump): the T = 1
instance has 160 instructions, 32 registers and no local memory in both builds; the carry `LDG.E.128.STRONG.GPU` and
the two `LDG.E.STRONG.GPU` now precede ACQBULK and only P and NEXT follow it; the T = 8 instance (16 warps) moves the
same three loads. Correctness (chk, `bx_box_run.py --runs 5`, six loads): PASS 12 / 12, `G_RELEASE_STARTUP_OK native
(128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, `CG_GATE ... equal 1023/1024 ok` (the control's line), timed PDL
{1, 8, 16, 32} on.

Paired A/B against ly_rfx (`tools/ly_prtl_ab.sh`, CTL=ly_rfx.py, c/w/c/w/c, 60 runs, six loads, PDL pinned, router
seed 106218864389449120, f931 case seeds, local_start per input), two rounds, logs
`muse_decode_layer/profiles/box/finb1/` (round1/, round2/, `table_both_rounds.txt`). Round 1 decode 1 sliding / full
-0.03 / -0.07 us, round 2 -0.27 / -0.22. Pooled (6 controls, 4 candidate runs, 24 pairs):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.51 | 0.78 | 64.36 | -0.15 | 1.57 | 14/24 | inside |
| decode 1 full | 70.78 | 0.85 | 70.64 | -0.15 | 1.70 | 17/24 | inside |
| decode 16 sliding | 127.12 | 0.74 | 126.86 | -0.27 | 1.47 | 18/24 | inside |
| decode 16 full | 199.22 | 1.33 | 199.08 | -0.14 | 2.66 | 17/24 | inside |
| decode 32 sliding | 144.14 | 1.01 | 143.91 | -0.23 | 2.02 | 16/24 | inside |
| decode 32 full | 295.24 | 1.71 | 294.86 | -0.37 | 3.42 | 16/24 | inside |
| verify 1 sliding | 102.09 | 1.20 | 102.04 | -0.04 | 2.40 | 14/24 | inside |
| verify 1 full | 110.50 | 0.85 | 110.14 | -0.35 | 1.70 | 19/24 | inside |
| verify 16 sliding | 242.06 | 1.58 | 241.50 | -0.56 | 3.17 | 19/24 | inside |
| verify 16 full | 326.21 | 3.50 | 326.07 | -0.14 | 7.01 | 13/24 | inside |
| verify 32 sliding | 294.44 | 1.39 | 294.02 | -0.42 | 2.94 | 18/24 | inside |
| verify 32 full | 458.26 | 2.06 | 458.64 | +0.38 | 4.58 | 8/24 | inside |

Reading: a null. The changed cells (decode 1, verify 1) move by -0.04 to -0.35 us with 14 to 19 of 24 pairs, the same
size and sign spread as the cells whose code did not change (decode 16 / 32, verify 16 / 32: -0.56 to +0.38).
Hypothesis for why: carry's latency was already mostly hidden under reduction 1 (its load is issued two shuffles into
that reduction), so what the finish spends after its wait is the P load and its two block reductions (6 `BAR.SYNC`
in both builds), which no load placement removes. Not kept; ruled-out row added.

### (b) The slice launch: what the stamps show (measurement, ly_pmt_rfx, 20 inputs per cell, us from the first MoE CTA's entry)

| item | sliding | full |
|---|---:|---:|
| last MoE CTA's phase A end / trigger (phase B end) / barrier / first exit / last exit | 12.78 / 20.10 / 21.58 / 21.36 / 23.71 | 12.66 / 19.47 / 20.82 / 20.69 / 21.46 |
| final pass after the barrier (last 8 CTAs) | 1.29 | 0.47 |
| slice entry min / max (512 CTAs), first entry after the trigger | 21.39 / 21.49, 1.31 | 21.18 / 21.26, 1.63 |
| slice CTAs with codes landed before the MoE's last exit | 468 | 0 |
| slice wait return after the MoE exit, p50 / max | 0.53 / 0.56 | 1.54 / 2.50 |
| slice end after the MoE exit; finish end after the slice end | 3.20; 1.57 | 4.30; 1.54 |
| plain / instrumented graph | 68.53 / 70.93 | 75.46 / 76.10 |

In both cells the slice's first CTA enters within 0.03 us of the first MoE CTA's exit, so this run alone cannot tell
PDL launch latency from waiting for a free slot; the candidate below separates them. The slice's pre-wait prologue is
its 12 KB of codes per CTA, the escape count and the escape table, all prepare-time constants; x, the only operand
the MoE writes, is already behind the wait. Nothing else can move ahead of the wait, so the lever left is the trigger
position.

### Candidate ly_trga: the TINY MoE build triggers at its phase A end

`tools/dl_ly_trga_patch.py` writes `muse_decode_layer/cand/ly_trga.py` (SHA256
c4576e5536862bf7a84d73ff6ee1fdee29e4da1390dea6b5dd083604fd943645): in the helper0 MoE source only (ly_rfx lines
15077-15087), the compute threads' `griddepcontrol.launch_dependents` is issued right after `run_phase<false>`
(`DBG(2)`) under `#if OKC_TRIGA` and skipped at its old place under `#if !OKC_TRIGA`; one new line after ly_rfx line
16262 adds `-DOKC_TRIGA=1` to `MOE_TINY_DEFINES` (T <= 4). TINY2 and MID preprocess to the base's code. The trigger
orders no memory: the slice's `griddepcontrol.wait` still waits for the whole MoE grid. SASS (`tools/ly_micro_sass.py`,
nvcc 13.0, sm_90a): TINY `moe_fused<false / true>` 5197 -> 5200 instructions, 145 registers, no spill, `PREEXIT`
moved from 0xd2d0 (after phase B) to 0x8bc0 (between the phases), every other difference a shifted branch target;
TINY2 SASS identical. Correctness (chk): PASS 12 / 12, startup OK, CG_GATE line equal to the control's, PDL {1, 8,
16, 32} on.

Paired A/B against ly_rfx, same driver and settings, logs `muse_decode_layer/profiles/box/trga/`. Round 1 decode 1
sliding +1.78 us (pairs +1.62 +1.33 +1.54 +2.29), full +1.29 (+1.22 +1.36 +1.86 +1.04); round 2 sliding +2.13 (+2.06
+2.13 +2.21 +2.16), full +1.44 (+1.46 +1.26 +1.47 +1.49). Pooled:

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.47 | 0.93 | 66.42 | +1.95 | 1.86 | 0/24 | loss |
| decode 1 full | 70.85 | 0.82 | 72.21 | +1.36 | 1.63 | 0/24 | inside (every pair a loss) |
| decode 16 sliding | 126.99 | 0.80 | 126.91 | -0.08 | 1.60 | 12/24 | inside |
| decode 16 full | 199.71 | 1.20 | 199.44 | -0.27 | 2.40 | 16/24 | inside |
| decode 32 sliding | 144.35 | 1.07 | 144.35 | +0.00 | 2.14 | 13/24 | inside |
| decode 32 full | 295.39 | 2.13 | 295.19 | -0.20 | 4.26 | 13/24 | inside |
| verify 1 sliding | 102.35 | 1.02 | 102.52 | +0.16 | 2.05 | 9/24 | inside |
| verify 1 full | 110.40 | 2.18 | 110.37 | -0.03 | 4.35 | 13/24 | inside |
| verify 16 sliding | 242.76 | 1.31 | 242.80 | +0.04 | 2.62 | 9/24 | inside |
| verify 16 full | 326.14 | 2.42 | 326.12 | -0.03 | 4.83 | 13/24 | inside |
| verify 32 sliding | 294.89 | 2.05 | 294.60 | -0.28 | 4.10 | 15/24 | inside |
| verify 32 full | 458.72 | 1.90 | 458.55 | -0.17 | 4.59 | 13/24 | inside |

What the candidate's stamps (`ly_pmt_trga.py` 9814ef6c, the same 20 inputs) and kernel timeline (`tools/ly_tl_run.py`,
10 profiled replays, `profiles/box/trga/tl/`) show:

- The first slice CTA enters 1.34 / 1.57 us after the last MoE CTA's phase A end (sliding / full), with no MoE CTA
  exiting nearby, so the PDL launch latency from the last trigger is 1.3 to 1.6 us by itself.
- Only 132 slice CTAs, one per SM, enter beside the running MoE CTA (132 with codes landed before the exit, in both
  cells); the other 380 enter when MoE CTAs exit (entry p50 21.26 / 21.20 us, as in the base). The register census of
  the cg32 section allowed 3 per SM (the 145 x 288 MoE registers leave 21,760), so another resource binds
  (hypothesis: the TINY build's dynamic shared memory leaves room for one 1 KB slice CTA). This would also explain
  why ly_cg32 (4 per SM by registers) was null.
- The 132 early CTAs' codes no longer stream under the MoE's final pass, which got short at sliding (0.39 us after
  the barrier against 1.29 in the base), so the last MoE exit came earlier relative to the 380 late CTAs, whose
  1.45 us code stream now runs after the exit: slice wait return p50 / max 1.33 / 3.33 us after the exit (base 0.53 /
  0.56), slice end 5.22 / 5.25 us after the exit (base 3.20 / 4.30). Timeline: the slice's critical path after the
  MoE's end is 4.19 / 4.22 us against 2.96 / 2.87 in the base. Which part of the base's final pass the slice's code
  stream lengthens is not separated.

ly_trga is a loss; ruled-out row added.

### Decision

No candidate for a judge check from this lane: ly_finb1 is a null (-0.15 / -0.15 us at decode 1, 14 and 17 of 24
pairs), ly_trga a loss (+1.95 / +1.36 us, 0 of 24 pairs below the control). Two measurements carry forward: after
its wait the finish spends its time on P and two block reductions, not on serialized loads; and only one slice CTA
fits beside a TINY MoE CTA, so while the slice is 512 CTAs of 12 KB each no trigger move can preload more than a
quarter of its codes. Next idea (hypothesis, not built): a 132-CTA slice (one CTA per SM, about 4 row pairs and 48 KB
of codes per CTA held in registers, about 40 registers at 544 threads, inside the 21,760 the MoE CTA leaves, and the
same 1 KB of shared memory) launched from an early MoE trigger, so every code byte is resident before the MoE's exit
and only the x read and the math remain after the wait. Bound: the base's post-exit code stream at full (about 1.5
us) and the late-CTA spread at sliding (about 0.5 us). Its first test is the residency: a stamped run showing all
132 slice CTAs holding their codes before the MoE's exit.

## 2026-11-04 - Decode 1 MoE ramp, handoff, tail (box)

MoE lane, relaunched 09:17 PT. Question: of the decode 1 MoE's non-stream time (TINY build, helper0, T = 1; the
timeline lane's ramp 3.1, handoff 2.2 and tail 2.1 us), which piece has a mechanism one edit can take? Base:
muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda, the public
row caed0290397b44a7, never edited). Box only (okc-h200), every GPU process under gpu.lock, no judge check. The layer
lane shared the box (ly_trga, ly_finb1, ly_v1un); its runs and mine queued on the lock. At decode 1 the MoE is the
`tiny` build (M <= MOE_TINY_MAX_M = 4: OKC_NW 8, WS 1, FAST1 2, three 36,864-byte ring slots, helper warp).

### Instruments (stamped copies, never candidates)

All four are `tools/dl_ly_pmt_patch.py` copies (helper0's own DBG slots, written only when the launch passes the
fourth scratch tensor, so the plain graph runs the base's code) with more stamps, run by `tools/ly_box_pmt.py --raw
20` (plain and instrumented graph of the whole layer per fresh input, PDL pinned, router seed 106218864389449120,
f931 case seeds, local_start per input, 20 inputs per cell), summarized by `tools/ly_mtl_summary.py` and
`tools/ly_mtl23_summary.py`. JSONs and logs: `muse_decode_layer/profiles/box/moe/`.

- ly_mtl (`tools/dl_ly_mtl_patch.py`): adds thread 0's first ring slot full (slot 6) and first phase B unit past
  its doneA wait (slot 7).
- ly_mtl3 (`tools/dl_ly_mtl3_patch.py`): the prologue: FAST1 copies issued, launch generation stored, route()
  done, WS_JOIN passed, first job's x in registers (an empty asm consumes every x0 register), first slot.
- ly_mtl2 / ly_mtl4 (`tools/dl_ly_mtl2_patch.py`, `dl_ly_mtl4_patch.py`): the helper's doneA credit for the CTA's
  last phase A unit: partials in, epilogue issued, before and after the credit fence, atomicAdd issued.

### Where the time goes (measurements, base, medians over 20 inputs, us from the first MoE CTA's entry; mean CTA, latest CTA in brackets)

| stamp | sliding | full |
|---|---:|---:|
| launch generation stored (thread 0) | 0.60 | 0.58 |
| FAST1 warp's two first-unit copies issued | 1.13 (1.25) | 1.10 (1.23) |
| route() done | 2.20 (2.34) | 2.15 (2.29) |
| WS_JOIN passed (unit ranges published) | 2.39 (2.53) | 2.35 (2.50) |
| first job's x rows in registers | 2.99 (3.14) | 2.94 (3.09) |
| first ring slot full (thread 0's mbar_wait returns) | 3.06 (3.65) | 3.01 (3.65) |
| phase A end (thread 0) | 11.93 (12.51) | 11.91 (12.69) |
| first phase B unit past its doneA wait | 14.21 (14.40) | 14.23 (14.45) |
| phase B end | 19.15 (19.46) | 19.17 (19.50) |
| past the grid barrier | 20.70 | 20.63 |
| last CTA's exit | 23.20 | 21.39 |

(The prologue rows come from ly_mtl3, the rest from ly_mtl; the first-slot row agrees between the two within 0.03.)

Ramp. The compute warps reach their first weights at 3.0 us, and on the mean CTA the weights are not what they wait
for: the x rows of the first job land 0.60 us after WS_JOIN, the slot is full 0.07 to 0.10 us later. The chain is
the prologue: route() ends 1.6 us after the generation is stored, then the ranges, WS_JOIN, and an L2 round trip for
x. Which warp route()'s CTA barriers wait on is not stamped; the FAST1 warp, which issues its copies only at 1.1 us
and then runs the x prefetch and LUT loops before it enters route(), is the likely one (hypothesis). The weights gate
only the latest CTAs (3.65 against a 3.06 mean).

Handoff (ly_mtl4, two runs of the base; the instrumented graph is about 1.7 us slower than the plain one):

| piece of the CTA's last phase A unit | sliding | full |
|---|---:|---:|
| thread 0's phase A end to the helper holding all eight warps' partials | 0.57 / 0.87 | 0.55 / 0.56 |
| the helper's epilogue (sum, SwiGLU, inter stores) | 0.31 / 0.35 | 0.31 / 0.32 |
| the credit's `__threadfence` (MEMBAR.SC.GPU), mean CTA (latest) | 0.54 (1.06) / 0.46 (0.86) | 0.66 (1.23) / 0.65 (1.10) |
| the same fence on the CTA that credits last | 0.58 / 0.53 | 0.53 / 0.50 |
| __syncwarp and the atomicAdd issue | 0.14 / 0.13 | 0.14 / 0.15 |
| last credit to the first phase B start (poll detection) | 0.19 / 0.21 | 0.19 / 0.19 |

So the 2.2 us is a chain of five pieces, none over 0.9 us: the warps' last unit and partial hand-off, the epilogue,
the fence's drain of the inter stores, the atomic, the poll. Every CTA starts phase B within 0.2 us of the last
credit.

Tail. Measured by the layer lane's ly_pmt_rfx / ly_pmt_trga runs ("Decode 1 finish and slice launch"): the last 8
CTAs' final pass takes 1.29 us after the barrier at sliding and 0.47 at full; at sliding it overlaps the slice's 6.3
MB code stream (468 of 512 slice CTAs hold their codes before the MoE's exit), at full the code stream runs after the
exit (0 of 512; the wait returns up to 2.5 us after the exit). With the slice's codes moved off that window (ly_trga)
the final pass takes 0.39 us in both cells.

### Candidate ly_rpf: the route kernel prefetches each MoE CTA's first unit (ramp; loss)

`tools/dl_ly_rpf_patch.py` writes `muse_decode_layer/cand/ly_rpf.py` (SHA256
6c270dcbbf9819351e17f5ed0c637501df7727fce7b07cfb53dcc47f132f489f): at T = 1 the route launch
(`_prq_sl_tile_route_cg`) gets NSM = 132 extra programs; each runs the same sigmoid / bias / top-8 as the storing
program, stores nothing, and issues one `cp.async.bulk.prefetch.L2.global` (36,864 B, thread 0) of the unit that MoE
CTA c requests first under TINY's FAST1 split (no dynamic pool in TINY): v0 = floor(popc(m) x 512 x c / 132),
expert = the (v0 >> 9)-th set bit of the local mask, address wAu + (expert x 512 + (v0 & 511)) x 36,864. The PTX of
the compiled route kernel holds the predicated prefetch once. No value changes; ids / lids / wts stores, every wait,
trigger and the MoE source are the base's. Built before the prologue stamps existed, on the hypothesis that the
first wave's 9.7 MB gated the ramp.

Correctness (`ly_prtl_ab.sh` chk step, `bx_box_run.py --runs 5`, six loads): reference check PASS at 12 cells,
G_RELEASE_STARTUP_OK (56 verified replays), CG_GATE line equal to the control's. (The first build missed the T = 16
graft path's call of the same kernel and failed there with a missing-argument TypeError; the parameters got
defaults and the run was restarted; that file is not the one measured.)

Paired A/B against ly_rfx (c/w/c/w/c, 60 runs, six loads, PDL pinned, two rounds back to back,
`muse_decode_layer/profiles/box/rpf/`, `table_both_rounds.txt`), net us, pooled over 6 control and 4 candidate runs:

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.64 | 0.80 | 66.49 | +1.85 | 1.60 | 0/24 | loss |
| decode 1 full | 70.95 | 0.70 | 73.06 | +2.11 | 1.41 | 0/24 | loss |
| decode 16 sliding | 126.80 | 1.18 | 127.03 | +0.23 | 2.37 | 9/24 | inside |
| decode 16 full | 199.32 | 2.42 | 199.08 | -0.24 | 4.83 | 12/24 | inside |
| decode 32 sliding | 143.47 | 1.28 | 144.06 | +0.58 | 2.56 | 3/24 | inside |
| decode 32 full | 294.88 | 0.86 | 294.64 | -0.24 | 2.95 | 18/24 | inside |
| verify 1 sliding | 102.14 | 0.67 | 101.95 | -0.19 | 1.34 | 13/24 | inside |
| verify 1 full | 110.38 | 0.98 | 110.34 | -0.04 | 1.95 | 12/24 | inside |
| verify 16 sliding | 242.71 | 0.93 | 242.31 | -0.40 | 2.43 | 16/24 | inside |
| verify 16 full | 326.47 | 1.44 | 326.70 | +0.23 | 3.26 | 5/24 | inside |
| verify 32 sliding | 294.51 | 1.20 | 294.41 | -0.10 | 2.95 | 12/24 | inside |
| verify 32 full | 458.23 | 1.23 | 458.21 | -0.02 | 4.58 | 12/24 | inside |

Round by round: decode 1 +1.79 / +2.06 (round 1), +1.90 / +2.17 (round 2), every pair positive. Why (kernel
timeline, `tools/ly_tl_run.py --kernels`, 10 profiled cold replays, `profiles/box/rpf/kernel_timeline.txt`): the
route kernel's critical path (its end minus the pooler's end) grows from 1.44 to 3.90 us (sliding) and 1.43 to 3.86
(full); the MoE is 0.21 / 0.28 us shorter; span 65.57 -> 67.54, 70.93 -> 73.42. Stamped candidate (ly_mtl_rpf): the
mean first slot moves 3.06 -> 3.02 / 3.01 -> 2.98, the latest CTA's 3.65 -> 3.22 / 3.65 -> 3.20; phase A end
unchanged within 0.1. The prefetch reaches L2 (the slowest CTAs' first slot is 0.4 us earlier), but the mean CTA
was gated by the prologue (table above), so it bought 0.2 to 0.3 us and the 133-program route launch cost 2.4.

### Candidate ly_cfar: the credit fence as fence.acq_rel.gpu (handoff; null)

`tools/dl_ly_cfar_patch.py` writes `muse_decode_layer/cand/ly_cfar.py` (SHA256
c7586fadec52f9d5bacce2de9cdc944c139d6a8fb75cb99bd941d651b8064a3d): in ws_helper's credit block (q == nA - 1) every
helper lane executes `fence.acq_rel.gpu` where it executed `__threadfence()` (fence.sc.gpu), TINY only
(`-DOKC_CFAR=1`; TINY2 and MID preprocess to the base). Ordering (PTX ISA, Memory Consistency Model, "Release and
acquire patterns"): a release pattern is a release operation, or a strong write preceded in program order by
fence.acq_rel or fence.sc, so lane 0's [fence; atomicAdd] stays a release pattern; the consumers' ld.acquire.gpu poll
is unchanged; the other lanes' inter stores reach lane 0's pattern through __syncwarp exactly as in the base, whose
argument for them never used the fence-SC total order (no reader of doneA executes fence.sc). No fence removed, no
proxy fence, barrier, acquire, atomic, wait or trigger moved. SASS (`tools/ly_micro_sass.py` on the box, nvcc 13.0,
tiny build, `profiles/box/moe/sass_tiny_{rfx,cfar}_report.json`): the two files' SASS differ in exactly two lines,
MEMBAR.SC.GPU -> MEMBAR.ALL.GPU in each moe_fused instantiation; 145 registers, 0 local, same instruction count.

Correctness: PASS at 12 cells, startup OK, CG_GATE equal to the control's. Paired A/B against ly_rfx, same recipe,
two rounds (`muse_decode_layer/profiles/box/cfar/`):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.58 | 0.54 | 64.66 | +0.07 | 1.09 | 9/24 | inside |
| decode 1 full | 70.75 | 0.38 | 70.64 | -0.11 | 0.77 | 15/24 | inside |
| decode 16 sliding | 126.78 | 1.09 | 126.88 | +0.10 | 2.18 | 8/24 | inside |
| decode 16 full | 199.24 | 1.31 | 199.65 | +0.41 | 2.62 | 9/24 | inside |
| decode 32 sliding | 143.86 | 1.33 | 143.92 | +0.06 | 2.66 | 12/24 | inside |
| decode 32 full | 295.12 | 1.23 | 295.06 | -0.06 | 2.95 | 14/24 | inside |
| verify 1 sliding | 102.06 | 0.90 | 101.93 | -0.12 | 1.79 | 14/24 | inside |
| verify 1 full | 110.29 | 1.12 | 110.65 | +0.36 | 2.24 | 7/24 | inside |
| verify 16 sliding | 242.20 | 3.28 | 242.98 | +0.78 | 6.56 | 9/24 | inside |
| verify 16 full | 326.46 | 2.16 | 325.91 | -0.54 | 4.32 | 18/24 | inside |
| verify 32 sliding | 294.01 | 1.97 | 294.35 | +0.34 | 3.94 | 9/24 | inside |
| verify 32 full | 458.14 | 0.83 | 458.32 | +0.18 | 4.58 | 8/24 | inside |

Stamped candidate (ly_mtl4_cfar, two runs): the fence takes 0.46 to 0.48 us on the mean CTA (latest 0.93 to 0.99,
the last-crediting CTA 0.42 to 0.56) against the base's 0.46 to 0.66 (0.86 to 1.23; 0.50 to 0.58); the other pieces
are unchanged. MEMBAR.ALL.GPU waits for the same store drain as MEMBAR.SC.GPU, so the fence kind is not the cost.

### Not run: ly_etrig

`tools/dl_ly_etrig_patch.py` / `muse_decode_layer/cand/ly_etrig.py` (b01c57e7..., helper lane 0 fires
`griddepcontrol.launch_dependents` after the CTA's doneA credit) was built at 09:25 PT, before I found the layer
lane's ly_trga (the same mechanism, triggered at the compute warps' phase A end) running on the box. Not measured:
ly_trga's stamps show only one slice CTA per SM fits beside the TINY CTA, so the same late-wave loss applies to any
earlier trigger while the slice is 512 CTAs.

### Decision

No candidate for a judge check: ly_rpf loses (+1.85 / +2.11 us at decode 1, 0 of 24 pairs below the control in
either cell), ly_cfar is a null (+0.07 / -0.11, 9 and 15 of 24). Measurements that carry forward: the decode 1 ramp
is the prologue chain (route() done at 2.2 us, x for the first job 0.6 us after WS_JOIN), not the first wave of
weights; the handoff is five pieces of 0.13 to 0.9 us each, and the credit fence costs the same in either kind.

Next (hypotheses, not built):

1. The MoE prefetches the slice's codes. At phase B start (past the doneA wait, when the B units are already in the
   3-slot ring and DRAM carries only the fourth unit), each of the 132 MoE CTAs issues `cp.async.bulk.prefetch.L2`
   for its 48 KB share of the slice's 6.3 MB of prepare-time codes. The slice keeps its launch at the phase B end
   trigger and its 512 CTAs; their code loads then hit L2 instead of DRAM. Targets: at full the 1.5 to 1.6 us code
   stream after the MoE's exit, at sliding the 0.8 to 0.9 us the code stream adds to the final pass. Bound about 1
   us per decode 1 cell. Cheapest test: the stamped pair (ly_pmt) on the edited file, then the A/B. It needs no
   slice reshaping and no trigger move, so it is an alternative to the layer lane's 132-CTA slice idea. ly_rpf
   showed a 132-SM bulk prefetch does reach L2 (its slowest CTAs' first slot moved 0.4 us earlier).
2. At M = 1 every job's x is token 0: the compute warps can load their x slice right after the entry wait,
   which removes the 0.6 us x round trip after WS_JOIN from the ramp; the first slot then gates (about 0.1 to 0.6
   us left on the mean CTA, unmeasured).
3. The handoff's largest piece is the 0.55 to 0.87 us from warp 0's last phase A unit to the helper holding all
   eight warps' partials (warp skew or the partial hand-off; not separated, needs per-warp stamps).

## 2026-11-04 - Verify 1 slice and finish (box)

Verify 1 lane, round 2 (T = 8). Question: the two T = 8 items round 1 left unmeasured per CTA, the unpooler slice sk2
(4.4 us, all of x read into each of its 128 CTAs after the wait) and the finish (2.1 to 2.5 us after the slice ends):
which part of each is largest, and can one T = 8-gated change take it? Base: `muse_decode_layer/cand/ly_v1un.py`
(SHA256 88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145, judge check b734f44afbf24acd VALID), never
edited. Box only (okc-h200, 17:50 to 18:09 UTC), every GPU process under `/root/okc/.build/box/gpu.lock` (the layer
lane's ly_rsl runs shared the lock). No judge check. Logs: `muse_decode_layer/profiles/box/v1sf/`.

### Instruments (never candidates)

- `muse_decode_layer/cand/ly_v1st.py` (26ee95cb...): `tools/dl_ly_pmt_patch.py --base ly_v1un.py --pin 88fbae86...
  --xld __ldca --out ly_pmt_v1un.py` (1041126b..., the MoE DBG and finish TLON stamps of the decode 1 tail), then
  `tools/dl_ly_v1st_patch.py`: the TINY2 MoE build with `-DOKC_DEBUG=1`; per-CTA globaltimer stamps in bfp_SK2_SRC's
  sk2_kernel (non-fused instantiations, written only when dbg is non-null; the host `sk2` passes the buffer set by the
  new export `sk2_set_tlu` as dbg for the slice's shape M = 8, N = 1024 only); the finish program's SM id. Slots: entry,
  prologue done (pre-wait weight stages, escape table, barriers), after the wait, x of the pre-issued stages issued,
  producer refills issued, escape table landed, consumer warps 0 and 7 seeing each of the 4 stages full, leaving the
  stage loop, the 4 escape warps done, epilogue cbar / barriers / tid 0's store, SM id, escape count. No fence,
  barrier, wait, trigger or launch attribute moves.
- `muse_decode_layer/cand/ly_v1sx.py` (3d2c61ff..., `--diag`): the same plus `sk2_set_tdg(n)`, which hands sk2's
  diagnostic bits to the stamped launch only; `--dg 4` drops x from the slice (stages carry weights only; output wrong
  by design), so the stage landings time the weight stream alone.
- `tools/ly_box_v1st.py`: plain and instrumented graph per cell (PDL pinned on at every T, gate skipped, router seed
  106218864389449120, f931 case seeds, local_start per input), 20 fresh inputs per cell, cold L2. `tools/ly_v1st_fin.py`
  reads the finish placement from its `--raw` JSON (`st2.json.gz`). Observer cost: plain / instrumented graph 103.86 /
  105.31 us (sliding), 118.02 / 117.55 (full).

### Slice and finish parts at T = 8 (measurement, ly_v1un, us, medians over 20 inputs, sliding / full)

Reference: the last MoE CTA's exit. Per-CTA items are the median over the 128 CTAs, then over inputs (`st1.log`,
`st2.log`: two runs, they agree within 0.1 us).

| part | sliding | full |
|---|---:|---:|
| slice CTAs entering before the last MoE exit (they take SMs as MoE CTAs leave) | 100 to 103 of 128 | 103 to 106 |
| first slice entry after the last MoE trigger | 1.71 | 1.70 |
| prologue: barriers, 2 weight stages (24 KB), escape table issued | 0.61 | 0.61 |
| wait return after the MoE exit, p50 / max (late-entering CTAs) | 0.47 / 0.94 | 0.51 / 0.90 |
| x of stages 0 and 1 issued / producer issues stages 2 and 3 (weights + x), after the wait | 0.16 / 0.48 | 0.16 / 0.51 |
| stage 0 / 1 / 2 / 3 full after the wait (no stage lands before the wait) | 0.77 / 1.15 / 1.54 / 1.86 | 0.67 / 1.06 / 1.44 / 1.79 |
| consumers done after the last stage | 0.26 | 0.26 |
| escape warps after the consumers, p50 / max | 0.00 / 0.38 | 0.00 / 0.40 |
| epilogue: cbar 0.10, k-warp reduction 0.06, store 0.16 | 0.32 | 0.32 |
| CTA wait return to its store, p50 / max | 2.46 / 2.77 | 2.40 / 2.70 |
| slice end (last store) after the MoE exit; end spread (last minus median CTA) | 3.57; 0.61 | 3.42; 0.54 |
| finish: entry before the slice end | 1.0 to 1.5 | 0.8 to 1.4 |
| finish: wait return / P stage (P load and first reduction) / carry stage / stores, p50 | 0.45 / 0.51 / 0.42 / 0.10 | 0.45 / 0.51 / 0.42 / 0.10 |
| finish: P stage, max over the 64 programs | 1.20 | 0.91 |
| finish end after the slice end; after the MoE exit | 2.30; 5.87 | 1.92 to 2.27; 5.34 to 5.74 |

Two diagnostics separate the two largest parts.

1. **The slice's post-wait time is its weight stream, not x** (`sx4.log`, ly_v1sx with dg 4, the same inputs): with
   no x at all the stages land 0.41 / 0.97 / 1.46 / 1.95 us (sliding) and 0.38 / 0.93 / 1.44 / 1.92 (full) after the
   wait, against 0.75 / 1.15 / 1.50 / 1.86 with x. Even stages 0 and 1, whose 24 KB of weights go out in the
   prologue, have not landed when the wait returns. x costs about 0.35 us at stage 0 and nothing after it. The weights
   (48 KB per CTA, 6.3 MB in all, from DRAM) can only start when a MoE CTA leaves its SM (the two cannot share an SM:
   >= 115,712 + 118,240 B of shared memory), and the MoE CTAs leave over the last 0.9 us before the wait returns.
   Round 1's hypothesis that the 8 MB of x reads from L2 were a separate cost does not hold at this size: they overlap
   the weight stream.
2. **The finish's tail is four SMs holding three or four finish programs each** (`fin_placement.txt` from
   `st2.json.gz`, 20 inputs per cell): the slice occupies 128 SMs, and the PDL launch of the finish's 64 programs puts
   14 of them on the 4 SMs without a slice CTA (3 or 4 per SM, 16 warps each). Those programs take 0.78 / 0.80 us in
   the P stage against 0.47 / 0.48 for the 50 programs that sit alone beside a slice CTA; they end 1.90 / 1.86 us
   (p50) and 2.30 / 2.27 (max) after the slice end, the single ones 1.45 / 1.41 and 1.58 / 1.57. The finish's end is
   set by the stacked programs: 0.7 us of its 2.3 us tail is the stacking (measurement; the ly_v1r16 stamps below show
   it from the other side). The tile 0 / 1 pattern seen first (P stage 0.6 to 0.95 us for tiles 0 and 1) is the same
   thing: blocks b and b + 8 (tiles 0 and 1 of token b) are dispatched first and go to the free SMs.

### Candidate ly_v1r16: 16 rows per slice CTA (64 CTAs)

`tools/dl_ly_v1r16_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_v1r16.py` (SHA256
0b5a22a084a2771c74806c0a72f3cbe6aa2ea0e00977ab6759a7937fbeddbfa1): one line, the `slice8_r192` row of
bfp_SK2_CONFIGS from (R 8, NT 1) to (R 16, NT 2); picked only for (unpooler, tokr 8), which the layer runs only at
T = 8 (T = 1 runs the cgemv slice). Built on the step 1 reading, before diagnostic 1: half as many CTAs read x, so the
L2 serves 4 MB of x instead of 8. Per row the k split (WSK, UPW), the accumulator chains, the escape copies and the
reduction order are the base's, so the output is bitwise the base's. SASS (`tools/ly_prtl_sass.py`, bfp extension,
nvcc 13.0, sm_90a, `sass.log`): the slice instantiation (PK) 807 -> 895 instructions, 66 -> 74 registers, no stack or
local memory, stage loop 485 -> 587 instructions with 8 -> 16 HMMA; every other kernel identical. Gate run: PDL8_AB
T = 8 savings 7.33 (full) / 7.30 (sliding) us, 9/9; PDL8H_GUARD T = 8 pass, ids mismatches 0. Check (chk, six loads):
PASS 12/12, `G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, CG_GATE line equal to the
control's.

Paired A/B against ly_v1un (`tools/ly_v1_ab.sh` with CTL=ly_v1un.py: c/w/c/w/c per round, 60 runs, six loads, PDL
pinned, two rounds; `r16_table_both_rounds.txt`):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.57 | 0.78 | 64.31 | -0.27 | 1.57 | 19/24 | inside |
| decode 1 full | 70.78 | 0.83 | 70.60 | -0.18 | 1.66 | 17/24 | inside |
| decode 16 sliding | 127.04 | 0.70 | 127.00 | -0.04 | 1.41 | 14/24 | inside |
| decode 16 full | 199.24 | 1.95 | 199.39 | +0.15 | 3.90 | 10/24 | inside |
| decode 32 sliding | 144.13 | 0.56 | 144.22 | +0.10 | 1.44 | 11/24 | inside |
| decode 32 full | 295.15 | 1.39 | 294.89 | -0.26 | 2.95 | 16/24 | inside |
| verify 1 sliding | 100.52 | 0.91 | 100.92 | +0.40 | 1.82 | 6/24 | inside |
| verify 1 full | 108.85 | 0.80 | 108.82 | -0.03 | 1.60 | 14/24 | inside |
| verify 16 sliding | 242.73 | 1.34 | 242.81 | +0.09 | 2.69 | 11/24 | inside |
| verify 16 full | 326.60 | 1.78 | 326.37 | -0.23 | 3.55 | 15/24 | inside |
| verify 32 sliding | 294.22 | 1.58 | 293.84 | -0.38 | 3.17 | 18/24 | inside |
| verify 32 full | 458.22 | 1.36 | 458.65 | +0.43 | 4.58 | 6/24 | inside |

Stamps (`st_r16.log`, the same instrument on ly_v1r16): the stages land 0.71 / 1.23 / 1.71 / 2.16 us after the wait
(sliding; base 0.77 / 1.15 / 1.54 / 1.86), the CTA's wait-to-store grows 2.46 -> 2.94 us and the slice ends 4.08 us
after the MoE exit (base 3.57); but with 68 SMs free the finish's programs no longer stack (end spread 0.10 against
0.83) and the finish ends 1.57 us after the slice (base 2.30). The two moves cancel: finish end after the MoE exit
5.65 / 5.49 against 5.87 / 5.34. Reading: twice the weights per CTA on half the SMs streams slower, as diagnostic 1
predicts, and the finish gain is diagnostic 2's 0.7 us. Null (+0.40 / -0.03 us, 6 and 14 of 24 pairs); not kept.

### Candidate ly_v1cl4: the slice split over K across clusters of 4 CTAs

`tools/dl_ly_v1cl4_patch.py` writes `muse_decode_layer/cand/ly_v1cl4.py` (SHA256
e79745cc1002f4334d0acafc20b2116ccff60cf3dbbb0a7c64ed2976eb14401a): one line, `slice8_r192` to (R 32, NT 4, CL 4):
32 clusters of 4 CTAs, each CTA one 1024-k stage of 32 rows, ranks 1 to 3 handing their sums to rank 0 by st.async
(the file's existing CL > 1 path, used by the router's CL 3 build). Intent: each CTA's whole 48 KB of weights goes
out before the wait (P = 1 is its whole K share) and only 16 KB of x follows it. With one stage per rank every escape
correction sits in escape copy 0, the one copy the CL > 1 rank > 0 path folds. Not bitwise: each row is four rank
partials added at rank 0 (CG_GATE at T = 1 reads sk2_vs_fp64 0.00e+00, equal 1024/1024, against the base's 1.21e-04,
1023/1024). Gate: PDL8_AB T = 8 savings 7.23 / 7.58 us, 9/9; guard pass. Check PASS 12/12, startup OK.

A/B against ly_v1un, two rounds (`cl4_table_both_rounds.txt`): verify 1 sliding +4.65 us (0/24 pairs, threshold
3.55), full +4.50 (0/24, threshold 2.18), both losses; the other ten cells -0.67 to +0.60, inside. Stamps
(`st_cl4.log`): two of the 32 clusters (8 CTAs) enter about 4 us after the MoE exit in every input. With one 198 KB
CTA per SM the GPU holds 30 four-CTA clusters at once (hypothesis from the two late clusters: GPC SM counts that are
not multiples of 4), so the grid runs in two waves and the slice ends 8.7 us after the MoE exit. The first wave's CTAs
are no faster either (wait-to-store 2.64 against 2.46): their 48 KB of weights still land 1.2 us after the wait, and
the one escape warp that owns the single stage walks 12 escapes (4x the rows) and finishes 0.48 us after the
consumers. Loss; not kept.

### Decision

No candidate for a judge check. ly_v1r16 null (+0.40 / -0.03 us at verify 1, 6 and 14 of 24 pairs), ly_v1cl4 loss
(+4.65 / +4.50, 0 of 24). Two ruled-out rows. What carries forward, both measured with a stated mechanism:

1. The finish's tail: 14 of its 64 programs stack 3 or 4 per SM on the 4 SMs the 128-CTA slice leaves free and end
   0.7 us after the others (2.30 against 1.58 us after the slice end, both cells). A placement change that keeps the
   finish's programs one per SM, without moving the slice to fewer SMs and without changing the finish's reduction
   layout (num_warps, hence its numerics), is worth up to about 0.7 us per verify 1 cell on the kernel timeline
   (hypothesis for the A/B size). Untested: the lane's two candidates were spent before diagnostic 2.
2. The slice's post-wait time is the 6.3 MB weight stream from DRAM, which cannot start before the MoE's CTAs leave
   their SMs; x costs about 0.35 us at stage 0 only. A lever has to bring the slice's weights closer before the wait:
   into L2 during a window where DRAM is idle (an L2 prefetch with an evict-last hint, for example from the MoE's tail
   or the route kernel; the moe lane's current T = 1 slice prefetch is the same mechanism at T = 1). Bound: the 1.86 us
   of stage landings after the wait, minus what an L2-resident stream would still need (unmeasured).
## 2026-11-04 - Decode 1 register slice (box)

Layer lane, round 2 (09:44 to 10:35 PT). Question: can the T = 1 unpooler slice hold all of its codes on chip
before the MoE ends, launched from a phase A end trigger, so that only x and the math remain after the MoE exits
(the next step proposed in "Decode 1 finish and slice launch")? Base: `muse_decode_layer/cand/ly_v1un.py` (SHA256
88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145; at T = 1 it runs the same code as ly_rfx). Box
only (okc-h200), every GPU process under `/root/okc/.build/box/gpu.lock`. No judge check. Logs and tables in
`muse_decode_layer/profiles/box/rsl/`.

### Residency: what binds beside a TINY MoE CTA (measurement)

`tools/ly_rslc_occ.cu` runs an occupier kernel with the TINY moe_fused footprint measured on ly_v1un
(`tools/ly_micro_sass.py`, tiny build: 288 threads, REG 145, allocated as 152; SHARED 1024 static; okc_smem_bytes()
= 214,784 dynamic), one CTA per SM, and launches a probe kernel on a second stream; a probe CTA that enters before
the occupier's first exit was resident beside it (`occ1.log`, `occ2.log`, two repetitions each, all 132 SMs alike):

| probe CTA shape | probe CTAs per SM beside the occupier |
|---|---:|
| 5 warps, 40 registers (the base slice, cgemv40_kernel<128, 2>) | 1 |
| 5 warps, 32 registers (ly_cg32) | 1 |
| 5 warps, 24 registers | 2 |
| 3 warps, 56 registers | 2 |
| 4, 6 or 7 warps at 56 registers; 7 warps at 40 or 48; 11 warps at 24 | 1 |
| 4 warps at 64; 8 warps at 40 or 48; 12, 13 or 16 warps at 40; 17 warps (544 threads) at 40 or 32 | 0 |
| 5 warps at 24 with 8 or 12 KB static shared memory | 1 |
| 5 warps at 24 with 16 KB | 0 |
| 7 warps at 56 with 14 or 15 KB | 1 |

The occupier's nine warps sit 3 / 2 / 2 / 2 on the four `%warpid % 4` classes (taken here as the SM
sub-partitions). With 152 allocated registers per thread, the sub-partition holding three of them keeps 16,384 -
3 x 4,864 = 1,792 registers, one warp at up to 56 registers; the other three keep 6,656 each. A probe CTA's warps
are spread over the four sub-partitions, with the short side placed on the full one (the 5-, 6-, 7- and 11-warp
probes put their fewer warps there), so the binding resource is the register file of that one sub-partition, not
the SM's 21,760 free registers in total. That explains the base's one slice CTA per SM beside the MoE (ly_trga)
and why ly_cg32 was null. Shared memory leaves 16,640 bytes (233,472 - 214,784 - 1,024 - 1,024); after the slice's
own 1 KB reserve, between 12 and 16 KB fit, as measured. The proposed 544-thread / 40-register shape fits 0 per
SM. The largest shape that fits is 7 warps at 56 registers (12,544 registers) with up to 15 KB of shared memory.

### Candidate ly_rsl: 128 slice CTAs of 7 warps holding their codes

`tools/dl_ly_rsl_patch.py` writes `muse_decode_layer/cand/ly_rsl.py` (SHA256
2e69f45cd80cadf12912cfc4dd367704de19a9c5621c2a1204fc4fdf74d95e82). Two parts, both reached only at T = 1:

- MoE trigger, the ly_trga edit: in the helper0 source, after ly_v1un line 15079 (`DBG(2)` after
  `run_phase<false, BIG>`) an `#if OKC_TRIGA` block issues `griddepcontrol.launch_dependents`, and the existing one
  after line 15087 goes under `#if !OKC_TRIGA`; one new line after line 16263 adds `-DOKC_TRIGA=1` to
  `MOE_TINY_DEFINES` (T <= 4, so only T = 1 in the layer). Nothing else in the MoE changes.
- slice: a new kernel `cgrs_kernel` in CG_CUDA_SRC in front of `cgemv_launch` (line 17817), and in the pdl8 edit
  that turns the cgemv40 launch into `cg_go` (line 18927) a branch that launches it for `NTH == 128 && CPT == 2`
  with `half_n == 512` (cgemv_fwd runs only for the unpooler slice at m = 1). 128 CTAs of 7 warps at
  `__maxnreg__(56)`, 8,352 bytes of shared memory (x and the partial sums), one per SM. CTA b owns row pairs
  4b..4b+3. A task (row, quarter q) is one cgemv_body compute warp's work for one row: lane l takes chunks 32q + l
  and 32q + l + 128. Warp w owns tasks w, w + 7, ...; before `griddepcontrol.wait` it loads its first three tasks'
  codes into 36 registers (21 of 32 tasks, 66% of the 6 MB) and issues `prefetch.global.L2` for the other tasks'
  lines and for the escape table. After the wait: x (8 KB, `__ldca`) to shared memory, one barrier, then every task
  runs `chunk_fma` in cgemv_body's order and the same butterfly; the L2 tasks load after the first held tasks free
  their registers; warps 3..6 walk the four escape tables with cgemv_body's loop; the four quarters and the escape
  sum are added in cgemv_body's order. Pre-wait reads are the prepare-time stream and table only; trigger at the
  CTA's end as before. ptxas: 56 registers, 0 spill, 8,352 bytes shared; SASS has the 12 code loads
  (`LDG.E.NA.128.CONSTANT` / `LDG.E.64.CONSTANT`) and 6 `CCTL.E.PF2` before `ACQBULK`, and the x loads after it.
- `--pair` (ly_rsl2, 471ecf40): the same with two tasks' FMA chains interleaved per call. `--no-trig` (ly_rsl_nt,
  0a1114e8): the 128-CTA slice with the MoE trigger left at its phase B end, to separate the two parts.

Correctness (chk, `tools/ly_rsl_box.sh`, `bx_box_run.py --runs 5`, six loads, PDL pinned): ly_rsl and ly_rsl2 PASS
12 / 12, `G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, and
`CG_GATE unpooler cg_vs_sk2 1.21e-04 ... equal 1023/1024 ok`, the control's line character for character.

### Step 1, residency (stamps, `ly_pmt_rsl.py` 7499c7c2 = `tools/dl_ly_pmt_patch.py` on the `--tl` variant)

20 inputs per cell, per-input local_start, `tools/ly_rsl_resid.py` (`residency.txt`): in every one of the 40 inputs
all 128 slice CTAs were resident on 128 distinct SMs before the last MoE CTA's exit, and every CTA's held codes had
landed before that exit (min 128 of 128). The slice CTAs enter 1.31 / 1.71 us (sliding / full, medians) after the
last phase A end, and their codes land 6.66 / 5.63 us before the MoE's exit. After the exit: wait return 0.38 us
(control 0.61 / 0.45), x in shared memory 0.32 us later (control: every lane's x 1.25 / 1.28 us), then the tasks
1.60 us (control 0.80 / 0.77), slice end 2.82 / 2.85 us after the exit. The interleaved form (ly_rsl2) did not
shorten the task phase (1.60 / 1.59 us). The residency design works as built.

### A/B and where the time goes

Paired A/B against ly_v1un at decode 1 (`tools/ly_prtl_ab.sh`, c/w/c/w/c, 60 runs, PDL pinned, router seed
106218864389449120, `ab_tables.txt`):

| candidate | round | decode 1 sliding w - c | decode 1 full w - c |
|---|---|---:|---:|
| ly_rsl | 1 | +1.46 (pairs +1.46 +1.46 +1.20 +1.58) | +1.34 (+1.34 +1.18 +1.34 +1.42) |
| ly_rsl | 2 | +0.73 (+0.77 +0.83 +0.35 +0.82) | +1.58 (+1.10 +1.34 +1.95 +1.98) |
| ly_rsl, pooled (6 controls, 24 pairs) | | +1.09, 0 / 24 pairs below the control | +1.46, 0 / 24 |
| ly_rsl_nt | 1 | +1.10 (+1.18 +1.01 +1.28 +0.96) | +0.99 (+0.80 +0.50 +1.17 +1.34) |

The other four loads were not A/B'd: the code changes reach only T = 1, and the candidate loses its target cells.

The slice-instrumented stamps above cannot rank the arms: the control's stamped slice (512 CTAs forced to land their
codes before the wait) costs the control 1.7 us of graph time (plain 68.91 against instrumented 70.62 at sliding),
the candidate's nothing. `--moe-only` (new in `tools/ly_box_pmt.py`) stamps the MoE and the finish only and runs
every slice plain, so both arms carry the same instrument (`moe_only_stamps.txt`; medians over 20 inputs, us from
the first MoE CTA's entry; two control runs, two of each candidate):

| item | ly_v1un | ly_rsl_nt | ly_rsl |
|---|---|---|---|
| sliding: phase B end / grid barrier / MoE exit | 20.51, 19.92 / 21.82, 21.18 / 22.48, 21.79 | 20.80, 20.53 / 23.28, 23.14 / 23.68, 23.34 | 20.88, 21.57 / 22.19, 22.91 / 22.66, 23.46 |
| sliding: finish end after the MoE exit | 4.48, 4.43 | 4.43, 4.61 | 4.54, 4.58 |
| full: phase B end / grid barrier / MoE exit | 19.28, 19.39 / 20.64, 20.77 / 21.28, 21.20 | 19.73, 19.38 / 22.16, 20.91 / 22.43, 22.27 | 20.53, 20.72 / 21.78, 22.00 / 22.32, 22.46 |
| full: finish end after the MoE exit | 4.45, 4.48 | 4.51, 4.86 | 4.80, 4.77 |
| plain graph, candidate - control per input (median), sliding / full | | +1.39, +0.61 / +0.16, +1.49 | +0.93, +0.91 / +0.86, +1.02 |

Reading (measurements, then hypotheses):

- With the trigger at the phase A end, the MoE's phase B ends 0.4 to 1.7 us later (full: 1.1 to 1.4) and the MoE
  exits 0.2 to 1.7 us later (full: 1.0 to 1.3): the 6 MB the slice loads during phase B (4 MB into registers, 2 MB
  into L2) take DRAM time phase B needs. 6 MB at 4.5 TB/s is 1.3 us (hypothesis: phase B at decode 1 runs at the
  DRAM rate in the layer, so moving bytes into it is zero-sum). ly_trga moved 1.6 MB and showed no phase B change.
- With the trigger left at the phase B end (ly_rsl_nt), the 128 resident CTAs load their 6 MB while the MoE runs its
  grid barrier, and the barrier comes 1.1 to 1.35 us later after the phase B end (2.48 / 2.61 us against 1.31 /
  1.26 at sliding; 2.43 against 1.36 in one of the two full runs, 1.53 in the other). The
  base's 512 small CTAs fit one per SM there and load most of their codes only as MoE CTAs exit.
- After the MoE exits, the 128-CTA slice plus the finish take as long as the base's (4.4 to 4.9 us against 4.4 to
  4.5): the shorter wait and x landing are spent on 1.6 us of tasks in 7 warps per SM, against 0.8 us spread over
  the base's 512 CTAs of 5 warps (about 19 warps per SM once the MoE has left). With seven warps per SM beside the MoE, the post-exit math cannot have the base's
  latency hiding.

### Decision

ly_rsl is a loss at decode 1 (+1.09 / +1.46 us, 0 of 24 pairs below the control in either cell), ly_rsl_nt as well
(+1.10 / +0.99, one round, every pair a loss); ly_rsl2 was stamped only (same task phase as ly_rsl). No ledger
registration, no judge check. What carries forward: beside a TINY MoE CTA only a 7-warp, 56-register,
<= 15 KB slice CTA fits per SM (the sub-partition with three MoE warps binds); every code byte can be on chip
before the MoE ends, but at decode 1 the slice's 6 MB of DRAM time then lands inside the MoE (phase B or its grid
barrier) instead of after its exit, and the base's ordering, where most code loads follow the MoE CTAs' exits and
overlap only the final pass, is already close to the best one. This bears on the moe lane's current step (the MoE's
CTAs prefetching the slice's 6.3 MB into L2 at phase B start): the same bytes would share phase B's DRAM time.

## 2026-11-04 - Decode 1 slice L2 prefetch from the MoE (box)

MoE lane, round 2 (relaunched 10:00 PT). Question: at decode 1, can the MoE's 132 CTAs pull the unpooler slice's
6.3 MB of prepare-time codes into L2 before the slice reads them, so that the slice's code read after the MoE's exit
(full) and under the MoE's final pass (sliding) hits L2? The slice kernel, its launch point and its 512 CTAs are not
touched. Base: `muse_decode_layer/cand/ly_v1un.py` (SHA256 88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145,
judge-valid b734f44afbf24acd). Box only (okc-h200, 18:07 to 18:26 UTC), every GPU process under
`/root/okc/.build/box/gpu.lock`; no judge check. At T = 1 the MoE is the TINY build (M <= MOE_TINY_MAX_M = 4; TINY2
serves M 5 to 8, that is verify 1), so the change is gated on MOE_TINY_DEFINES. Artifacts:
`muse_decode_layer/profiles/box/slpf/`.

### L2 budget (numbers before building)

- L2: 62,914,560 B (torch `L2_cache_size` on the box H200).
- Slice codes: e["stream"] of the m = 1 unpooler cg entry, 512 CTAs x 12,288 B = 6,291,456 B; per MoE CTA
  ceil(6,291,456 / 132) rounded to 16 B = 47,664 B, issued as two bulk prefetches (32,768 + 14,896 B).
- MoE DRAM traffic after a prefetch issued at the CTA's doneA credit: phase B is 256 units x 36,864 B = 9,437,184 B
  per distinct local expert, and thread 0 has already issued each CTA's first 3 phase B units (396 units,
  14,598,144 B) during phase A's last units (ws_issue q + OKC_NS). Left: 4.28 MB at 2 distinct experts, 13.71 MB at
  3, 23.15 MB at 4 (the box's decode 1 inputs: 34 / 23 / 3 of 60 by the routing map).
- MoE data re-read after the prefetch (inter, ybuf, ss, x): under 0.3 MB, touched later than the prefetch.
- Newer-than-prefetch bytes before the slice reads: 6.3 + 4.3 to 23.2 + 0.3 = 10.9 to 29.8 MB, 17% to 47% of L2;
  under 30 MiB even if the two L2 partitions behaved as separate halves. No MoE line that is read again can be
  evicted by the prefetch; the weight units are read once.
- Issue path: the TINY build's helper warp (warp 8, `ws_helper`) runs the epilogues only. After its doneA credit for
  the CTA's last phase A unit it has nothing to do until the first phase B unit's partials arrive (every CTA's
  credit, then one unit of compute, at least about 1 us), so lane 0 issues the prefetch there, after the atomicAdd.
  The compute warps' code is unchanged.

The budget leaves out one cost: DRAM time. It shows the bytes fit in L2, not that phase B has DRAM time to spare,
and the stamps below show it does not at the phase A end.

### Candidates

`tools/dl_ly_slpf_patch.py` (base and pin checked) writes both; `--at a` = ly_slpf, `--at b` = ly_slpfb. Line numbers
are in the candidate file (ly_v1un's line before the insertion in brackets):

- line 1: header comment.
- 6921 (after 6919): `void okc_set_slpf(torch::Tensor t);` in MOE_CPP_SRC.
- 6984-6985 (6982): `_ma_ext` binds `okc_set_slpf` when the source defines it (as it does for okc_attrs).
- 12705-12707 (12701): `#ifndef OKC_SLPF / #define OKC_SLPF 0` in the helper0 source.
- 12838-12840 (12831): Params gains `slpf`, `slpf_bytes`, `slpf_share` under `#if OKC_SLPF`.
- 14193-14203 (14183): `slpf_issue(p)`: `cp.async.bulk.prefetch.L2.global` of [blockIdx.x x share, min(bytes,
  (blockIdx.x + 1) x share)) in 32 KB pieces; the share comes from the host (no integer divide in the kernel).
- ly_slpf 14259-14261 (14238): in ws_helper's credit block, lane 0 calls it right after the doneA atomicAdds.
  ly_slpfb 14281-14283 (14260): lane 0 calls it after the helper's loop, before its `__threadfence(); WS_JOIN()`, so
  after the helper's last epilogue (the CTA's last OKC_WSBLK = 2 units' epilogues run in the compute warps, so this
  is about two units before the CTA's phase B end; every weight copy of the CTA has been issued by then).
- 15310-15319 (15286): host `okc_set_slpf(tensor)` keeps the address and byte count (16-B multiple and alignment
  checked; empty tensor = off).
- 15365-15369 (15331): okc_run fills the three fields under `#if OKC_SLPF`, share = ceil(bytes / grid) rounded up
  to 16 B.
- 19071-19089 (end of file): `MOE_TINY_DEFINES += ("-DOKC_SLPF=1",)` and a `_moe` wrapper that hands the TINY
  extension `state["bfp"]["cg"]["unpooler"]["stream"]` on every M <= 4 call (an empty tensor when no cg entry), so
  every T = 1 launch, captured or eager (in-prepare guards included), carries the pointer.

ly_slpf SHA256 b7609b3c7f7b7853605ec535b5686e8abe6f343f51344e04cf93db5e8d0fdf3d; ly_slpfb SHA256
a850fe2928340fd78295b8a404b66535f7dfd90419db19559c874304e832f1b5. No fence, barrier, acquire, release, atomic, wait,
trigger or launch attribute is added, removed or moved; the prefetch writes nothing, so no value changes.

SASS (box ext cache, cuobjdump, nvcc 13.0, sm_90a): the TINY build's `moe_fused<false / true>` hold 145 registers,
no stack and no local memory in the control (glue_moe_f380dc69...) and both candidates; total SASS lines 10,464 ->
10,688 (ly_slpf) / 10,704 (ly_slpfb), the difference being the share arithmetic and the `UBLKPF.L2` prefetches.
The TINY2 and MID builds of the candidates contain no UBLKPF (their device code preprocesses to the base's).

Correctness (chk, `bx_box_run.py --runs 5`, six loads): both candidates PASS at 12 cells, `G_RELEASE_STARTUP_OK
native (128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, `CG_GATE unpooler cg_vs_sk2 1.21e-04 ... equal 1023/1024
ok` (the box control's line), `CG_CHAIN ... {'unpooler': ('cg',)}`.

### Stamps (prefetch on against off, the lead's requested first measurement)

`tools/dl_ly_pmt_patch.py --xld __ldca` on each file (ly_pmt_v1un 1041126b, identical to the copy on main;
ly_pmt_slpf 69e215fa; ly_pmt_slpfb d2de131f), `tools/ly_box_pmt.py --raw 20`, 20 inputs per cell, f931 case seeds,
local_start per input. These ran beside the first A/B round, not before it. Each entry is the median of that
statistic over the 20 inputs (medians do not add), us from the first MoE CTA's entry unless stated:

| statistic | v1un sliding | slpf | slpfb | v1un full | slpf | slpfb |
|---|---:|---:|---:|---:|---:|---:|
| last CTA's phase A end | 12.45 | 13.01 | 12.58 | 12.72 | 12.58 | 12.59 |
| last CTA's phase B end (trigger) | 19.38 | 20.78 | 19.78 | 19.42 | 19.84 | 19.58 |
| last CTA past the grid barrier | 20.83 | 22.13 | 21.41 | 20.83 | 21.22 | 21.18 |
| last MoE exit | 23.04 | 23.10 | 23.55 | 21.62 | 21.82 | 21.70 |
| final pass (last 8 CTAs, barrier to exit) | 1.27 | 0.46 | 1.16 | 0.48 | 0.43 | 0.58 |
| slice codes landed, after its CTA's entry (p50) | 1.63 | 1.29 | 1.31 | 1.60 | 1.31 | 1.31 |
| slice CTAs with codes landed before the MoE exit | 477 | 12 | 501 | 0 | 0 | 0 |
| slice wait return after the MoE exit (p50) | 0.50 | 1.06 | 0.51 | 1.58 | 1.28 | 1.26 |
| slice end after the MoE exit | 3.26 | 3.79 | 3.18 | 4.37 | 4.14 | 4.08 |
| finish end after the MoE exit | 4.86 | 5.33 | 4.78 | 5.94 | 5.68 | 5.63 |
| plain graph, raw median | 68.72 | 68.56 | 67.98 | 75.10 | 75.60 | 75.23 |

Every instrumented output matched its plain graph (maxdiff 0 in the summaries).

### A/B against ly_v1un (`tools/ly_prtl_ab.sh`, CTL=ly_v1un.py, c/w/c/w/c, 60 runs, six loads, PDL pinned at 1, 8, 16, 32 in every run, router seed 106218864389449120)

ly_slpf, two rounds (`slpf/round1`, `round2`, `slpf_table_both_rounds.txt`). Round 1 decode 1 -0.17 / -0.47, round
2 -0.12 / -0.05:

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.39 | 1.14 | 64.24 | -0.15 | 2.27 | 14/24 | inside |
| decode 1 full | 70.86 | 0.56 | 70.60 | -0.26 | 1.12 | 17/24 | inside |
| decode 16 sliding | 126.94 | 0.64 | 127.14 | +0.20 | 1.28 | 5/24 | inside |
| decode 16 full | 199.34 | 1.12 | 199.65 | +0.30 | 2.24 | 7/24 | inside |
| decode 32 sliding | 143.88 | 1.58 | 143.64 | -0.25 | 3.17 | 14/24 | inside |
| decode 32 full | 295.17 | 1.46 | 295.12 | -0.05 | 2.95 | 16/24 | inside |
| verify 1 sliding | 100.51 | 0.90 | 101.16 | +0.65 | 1.79 | 3/24 | inside |
| verify 1 full | 108.99 | 0.93 | 109.19 | +0.20 | 1.86 | 11/24 | inside |
| verify 16 sliding | 241.50 | 4.58 | 242.88 | +1.39 | 9.15 | 3/24 | inside |
| verify 16 full | 326.10 | 1.70 | 327.11 | +1.01 | 3.39 | 6/24 | inside |
| verify 32 sliding | 294.05 | 2.05 | 294.39 | +0.34 | 4.10 | 9/24 | inside |
| verify 32 full | 458.51 | 2.26 | 458.54 | +0.02 | 4.59 | 11/24 | inside |

ly_slpfb, two rounds, then two more (`slpfb/round1` to `round4`, `slpfb_table_four_rounds.txt`). Rounds 1+2:
decode 1 sliding -0.38 (21/24), full -0.57 (24/24); rounds 3+4: sliding -0.34 (21/24), full -0.59 (24/24). Per
round: -0.16 / -0.56, -0.59 / -0.58, -0.59 / -0.57, -0.09 / -0.61. All four rounds pooled (12 controls, 8 candidate
runs):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.45 | 0.75 | 64.09 | -0.36 | 1.50 | 84/96 | inside |
| decode 1 full | 70.76 | 1.18 | 70.18 | -0.58 | 2.37 | 89/96 | inside |
| decode 16 sliding | 127.07 | 0.88 | 127.10 | +0.04 | 1.76 | 44/96 | inside |
| decode 16 full | 199.35 | 1.22 | 199.44 | +0.09 | 2.43 | 45/96 | inside |
| decode 32 sliding | 143.98 | 1.28 | 144.09 | +0.11 | 2.56 | 45/96 | inside |
| decode 32 full | 295.07 | 1.71 | 295.26 | +0.19 | 3.42 | 30/96 | inside |
| verify 1 sliding | 100.72 | 1.44 | 101.19 | +0.47 | 2.88 | 18/96 | inside |
| verify 1 full | 108.89 | 1.20 | 108.99 | +0.10 | 2.40 | 35/96 | inside |
| verify 16 sliding | 242.61 | 2.82 | 242.63 | +0.02 | 5.63 | 46/96 | inside |
| verify 16 full | 326.35 | 3.14 | 326.21 | -0.14 | 6.27 | 55/96 | inside |
| verify 32 sliding | 294.45 | 2.13 | 294.34 | -0.12 | 4.26 | 51/96 | inside |
| verify 32 full | 458.30 | 2.34 | 458.45 | +0.15 | 4.67 | 39/96 | inside |

### Reading

Measurements:

- An L2 hit shortens the slice CTA's code fetch by only 0.3 us: codes land 1.29 to 1.31 us after the CTA's entry
  with either prefetch, 1.60 to 1.63 us from DRAM. The 1 us estimate assumed the post-exit code read was DRAM time;
  most of it is not.
- Prefetching at the doneA credit (ly_slpf) lengthens phase B: the last CTA's phase B end moves +1.41 us at sliding
  and +0.42 at full, and phase A's end +0.56 at sliding (early CTAs prefetch while late CTAs still stream phase A).
  6.3 MB at 4.5 to 4.8 TB/s is 1.3 to 1.4 us, so at sliding phase B grew by about the prefetched bytes' DRAM time,
  as the layer lane's register slice found ("Decode 1 register slice (box)"). The same prefetch removed the slice's
  DRAM stream from under the sliding final pass (1.27 -> 0.46 us), so the exit barely moved and the A/B is a null.
- Prefetching after the helper's last epilogue (ly_slpfb), when the CTA has issued every weight copy, costs phase B
  +0.40 / +0.16 us and the A/B gains -0.36 / -0.58 us, reproduced in two independent two-round sets (21/24 and 24/24
  pairs in each).

Hypotheses (not separated): the remaining 1.3 us of the slice's code fetch is the 512 CTAs' simultaneous 6.3 MB
burst through L2 plus the new kernel's cold start, not DRAM. At sliding the slpfb stamps do not show the gain the
plain graph shows (exit +0.51, finish end after exit -0.08 against the control's stamp medians, plain-graph medians
-0.74); stamps of separate processes move by that much, so the paired A/B is the measurement and the sliding
attribution is open. verify 1 sliding reads +0.47 (18 of 96 pairs) for slpfb and +0.65 (3 of 24) for slpf although
the T = 8 device code is identical in both arms (OKC_SLPF is a TINY define; TINY2 has no UBLKPF); the candidates'
MoE modules load under different names and allocate in a different order, which may move page placement (the box
doc's untested hypothesis for MoE variance). It is inside the threshold (2.88 / 1.79 us).

### Decision

ly_slpf is a null (-0.15 / -0.26 us, 14 and 17 of 24 pairs); ruled-out row added. ly_slpfb wins decode 1 full by
0.58 us (24 of 24 pairs in each two-round set) and decode 1 sliding by 0.36 us (21 of 24 in each), 0.42 us on the
46:16 load weighting; every other cell is inside its threshold. Against the lead's bar (0.4 us with 20 or more of
24 pairs, nothing else losing beyond noise) it passes at full and on the weighted load, and misses at sliding by
0.02 to 0.06 us. Ledger line written as a proposal; no judge check dispatched by this lane. Expected judge value:
about 0.5 us x 0.212% per us = 0.1% of score, below one draw's spread, so a check would establish validity, not
rank.
## 2026-11-04 - Verify 1 finish spread (box)

Verify 1 lane, round 3 (T = 8). Question: round 2 measured that 14 of the finish kernel's 64 programs land 3 or 4 per
SM on the 4 SMs the 128-CTA slice sk2 leaves free and end about 0.7 us after the programs that sit alone. Can a
T = 8-gated launch change put the programs one per SM without touching the finish's warps or reduction order? Base:
`muse_decode_layer/cand/ly_v1un.py` (SHA256 88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145, the
public file), never edited. Box only (okc-h200, 18:20 to 18:30 UTC), every GPU process under
`/root/okc/.build/box/gpu.lock`. No judge check. Logs: `muse_decode_layer/profiles/box/v1fs/`.

### Resource arithmetic (measurement on the box, Triton 3.8)

The T = 8 finish `_sl_cluster_finish_cg` (BC 1024, 16 warps) compiles to 27 registers, 0 spills and 4,096 B of
dynamic shared memory (`CompiledKernel.metadata`, probe run under the lock). The slice CTA (`slice8_r192`: 13 warps,
66 registers) takes 1024 + 4 x 28,672 + 16 x 4 + 32 + 4 x 576 + 128 = 118,240 B + 8 x pmax of dynamic shared memory
(`launch_sk2`). An H200 SM has 233,472 B, and each CTA reserves 1,024 B more (the 512 B overshoot in docs/ruled-out.md
row "Fixed G2/24"). Warps (16 per program) and registers (16,384 per program) let 4 finish programs share an idle SM,
which is what round 2 saw. Shared memory is the one resource a launch can change without touching code:

- one finish per SM needs 2 x (S + 1,024) > 233,472, so S > 115,712;
- one finish beside a slice CTA needs S + 1,024 + 119,264 + 8 x pmax <= 233,472, so S <= 113,184 - 8 x pmax.

Both cannot hold, so the choice is between S >= 115,713 (one per SM, none beside a running slice CTA: 4 programs
start on the free SMs and 60 wait for slice CTAs to exit) and 76,801 <= S <= 113,184 - 8 x pmax (at most 2 per free
SM, one beside a slice CTA).

### Change: ly_v1fs

`tools/dl_ly_v1fs_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_v1fs.py` (SHA256
7415c5a2ed5291a9060f4739c07a1f7e1bc6a76106f01e65d48cbe6485378b69). Lines changed against ly_v1un:

- line 1: the header comment;
- lines 18331 to 18350 (new, before `_pdl8_sl_tile_after_attention`): `_V1FS_SMEM = 118784` and `_v1fs_reserve()`,
  which, for every compiled `_sl_cluster_finish_cg` in Triton's cache with 16 warps (the T = 8 config; T = 1 runs 8),
  sets the function's MAX_DYNAMIC_SHARED_SIZE_BYTES to 118,784 (libcuda `cuFuncSetAttribute`) and the launcher's
  packed shared-memory size (`packed_metadata[2]`) from 4,096 to 118,784;
- lines 18415 to 18416 and 18418 to 18419: `if T == 8: _v1fs_reserve()` before and after the unchanged finish launch
  (line 18417). Before covers kernels compiled by earlier calls; after covers the one the current call compiled.

The kernel source, its Triton cache key, grid, warps and every load, wait, trigger and store are the base's; the
launch only reserves shared memory that the code never addresses. So the SASS is the base's and the output is bitwise
the base's by construction (the reduction order cannot change). No fence, wait or trigger moves; the finish still
reads the slice's output only after `gdc_wait`.

### Where the programs land (stamps)

Instrument `muse_decode_layer/cand/ly_v1fsst.py` (46005dfd...): `dl_ly_v1fs_patch.py --env` (the reservation read from
V1FS_SMEM; 0 leaves Triton's 4,096, the base's launch), then `tools/dl_ly_pmt_patch.py --xld __ldca` and
`tools/dl_ly_v1st_patch.py`, the round 2 stamps unchanged. `tools/ly_v1fs_sweep.sh` runs `tools/ly_box_v1st.py` (20
fresh inputs per cell, cold L2, PDL pinned) at three reservations; `tools/ly_v1fs_fin.py` reads the placement
(`fin_placement.txt`). Medians over 20 inputs, sliding / full, us from the slice's last store unless stated:

| finish reservation | 4,096 (base) | 118,784 (ly_v1fs) | 98,304 |
|---|---:|---:|---:|
| programs on SMs that hold a slice CTA / on the 4 free SMs | 50 / 14, 52 / 12 | 60 / 4, 60 / 4 | 58 / 6, 58 / 6 |
| programs sharing an SM; most per SM | 14, 12; 4, 3 | 0, 0; 1, 1 | 4, 4; 2, 2 |
| program entry, p50 | -1.30 / -0.99 | -0.34 / -0.35 | -1.37 / -0.94 |
| wait return, p50 / max | 0.45 / 0.61, 0.45 / 0.58 | 0.45 / 0.45, 0.45 / 0.45 | 0.45 / 0.48, 0.43 / 0.45 |
| P stage, p50 / max | 0.48 / 1.18, 0.49 / 0.93 | 0.46 / 0.54, 0.46 / 0.54 | 0.48 / 0.62, 0.48 / 0.64 |
| finish end after the slice end | 2.29 / 1.95 | 1.57 / 1.57 | 1.63 / 1.63 |
| slice end after the MoE exit (summary line) | 3.57 / 3.44 | 3.50 / 3.52 | 3.58 / 3.52 |
| finish end after the MoE exit (summary line) | 5.87 / 5.50 | 5.07 / 5.06 | 5.20 / 5.12 |

At 118,784 B every input puts exactly one program on each of 64 SMs: 4 on the free SMs at once, the other 60 on SMs
whose slice CTA has exited. Those 60 enter 0.34 us before the slice's last store instead of 1.3 us, and the wait still
returns 0.45 us after it, so entering late costs nothing: the slice's CTAs exit with a 0.6 us spread (median store
-0.56) and the first 60 exits come early enough. The finish tail drops to the single-program time round 2 measured
(1.58), and the slice is unchanged. 98,304 B behaves as the arithmetic predicts (at most 2 per free SM) and ends 0.06
us later than 118,784; it was not built as a candidate.

### Checks and A/B

Gate (`.build/v1tl/gate_v1fs.log`, the file's own in-prepare benches): PDL8_AB T = 8 savings 7.74 (full) / 7.78
(sliding) us, 9/9; PDL8H_GUARD T = 8 full and sliding pass with all seven edges, ids mismatches 0; CG_GATE line equal to
the control's. Reference check (`chk.log`, six loads): PASS 12/12, `G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1,
8, 16) 56 verified_replays`.

Paired A/B against ly_v1un (`tools/ly_v1_ab.sh` with CTL=ly_v1un.py: c/w/c/w/c per round, 60 runs, six loads, PDL
pinned; two rounds `v1fs`, `v1fs_r2`, `table_two_rounds.txt`):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.72 | 0.54 | 64.48 | -0.23 | 1.09 | 21/24 | inside |
| decode 1 full | 70.92 | 0.38 | 70.97 | +0.05 | 0.77 | 7/24 | inside |
| decode 16 sliding | 127.12 | 0.86 | 127.09 | -0.03 | 1.73 | 12/24 | inside |
| decode 16 full | 199.13 | 0.78 | 199.40 | +0.27 | 1.99 | 7/24 | inside |
| decode 32 sliding | 144.10 | 0.46 | 143.74 | -0.36 | 1.44 | 21/24 | inside |
| decode 32 full | 295.09 | 0.77 | 294.88 | -0.21 | 2.95 | 15/24 | inside |
| verify 1 sliding | 100.69 | 0.69 | 100.28 | -0.41 | 1.38 | 24/24 | inside |
| verify 1 full | 108.82 | 1.47 | 108.50 | -0.32 | 2.94 | 13/24 | inside |
| verify 16 sliding | 243.09 | 2.69 | 243.07 | -0.03 | 5.38 | 12/24 | inside |
| verify 16 full | 326.35 | 1.34 | 326.56 | +0.21 | 3.26 | 10/24 | inside |
| verify 32 sliding | 294.88 | 2.32 | 294.80 | -0.09 | 4.64 | 16/24 | inside |
| verify 32 full | 458.48 | 1.57 | 458.29 | -0.19 | 4.58 | 15/24 | inside |

A third round (`v1fs_r3`, run after the two-round table was read, so supplementary evidence only;
`table_three_rounds.txt`) gave verify 1 -0.58 / -0.90 us; pooled over three rounds verify 1 sliding -0.46 us (50/54
pairs), full -0.51 (36/54), every other cell inside its threshold (largest +0.42 at decode 16 full, 12/54). The
timeline run (`tl_v1fs.log`, `--kernels`) shows `_sl_cluster_finish_cg` at 3.76 / 3.80 us of kernel span.

### Decision

ly_v1fs does what was asked: the 64 finish programs land one per SM in every input (stamps), the finish ends 1.57 us
after the slice instead of 2.29 / 1.95, and its output is the base's by construction. On the box it is 0.41 us faster
at verify 1 sliding with 24/24 pairs and 0.32 us faster at verify 1 full with 13/24; nothing else moves beyond noise.
Against the lane's registration rule (at least 0.4 us at verify 1 with 20+ of 24 pairs, nothing else losing) sliding
qualifies and full does not, so no ledger registration line from this lane. Size: about 0.4 to 0.5 us per verify 1
cell, 0.06 to 0.07% of the layer score at 0.144% per us, well below the judge's 1.3% draw spread, so one judge check
could not resolve it alone. Recommendation for the lead: stack it onto the next layer file rather than spend a check
on it alone (numerics identical, T = 8 only, three lines of dispatch). No loss, so no ruled-out row.
## 2026-11-04 - Decode 16 / 32 sliding timeline (box)

Layer-d16 lane. Question: at decode 16 and decode 32 (the sliding cells hold 5.0% and 5.2% of the score above speed of
light, docs/steering/2026-11-04-top100-push.md), where does the step go kernel by kernel and edge by edge in the
current public file, and which excess can one change take? Base: muse_decode_layer/cand/ly_v1un.py (SHA256
88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145), never edited. Box only (okc-h200, from 18:30 UTC),
every run under gpu.lock, no judge job. Logs, traces and tables: `muse_decode_layer/profiles/box/d16tl/` (box
`.build/d16tl/`, `.build/prtl/d16*`). Driver for each candidate: `tools/ly_d16_ab.sh` (a gate run with the file's own
in-prepare PDL8 guard at decode 16 / 32, then `tools/ly_prtl_ab.sh` twice against ly_v1un at all six loads with the chk
step first, then a candidate timeline). Tables: `tools/ly_d16_tltab.py DIR`.

### Timeline at T = 16 and T = 32 (measurement)

`LY_TRACE_DIR=.build/d16tl/tl_v1un tools/ly_tl_run.py muse_decode_layer muse_decode_layer/cand/ly_v1un.py --loads
"decode 16,decode 32" --runs 20 --kernels --pin-pdl8 1,8,16,32 --skip-pdl8-gate --router-seed 106218864389449120
--judge g1b4_pdl8d.f931abf568544018.json` (lock, ls-cache and ext-cache as in the decode 1 section); reference check
PASS at all four cells, net 127.20 / 203.55 (decode 16 sliding / full) and 144.13 / 296.24 us (decode 32). Medians
over the 10 profiled cold replays. Critical path = this kernel's end minus the previous kernel's end; gap = start minus
the previous end (negative: PDL overlap). Floors at 4.44 TB/s from "Decode 16 / 32 sliding: floor and gap"; the MoE
floor is taken at the judge's distinct-expert target (7.5 / 9.0 experts: 47.9 / 55.9 us), not the 10 profiled inputs'
own count, so the MoE row's excess is approximate (the route-map fits put about 5 to 6 us per distinct expert on it).

Decode 16 sliding (span 128.77 us, sum of durations 145.57):

| # | kernel | start | dur | end | gap | critical path | floor | over floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | qkv sk2<1,16,3,...> (plain launch, first kernel) | 0.00 | 10.27 | 10.27 | - | 10.27 | 6.39 | 3.88 |
| 1 | attn_kernel<1,false,1> | 8.42 | 9.55 | 18.02 | -1.82 | 7.71 | 1.89 | 5.82 |
| 2 | o sk2 | 10.86 | 10.85 | 21.71 | -7.15 | 3.68 | 2.84 | 0.84 |
| 3 | b1 glue_cluster_boundary | 19.22 | 4.82 | 23.98 | -2.48 | 2.30 | 0.35 | 1.95 |
| 4 | pooler + router sk2 (fused) | 22.24 | 17.68 | 40.05 | -1.76 | 15.94 | 13.26 | 2.68 |
| 5 | _prq_sl_tile_route_cg | 38.32 | 3.33 | 41.62 | -1.74 | 1.57 | 0 | 1.57 |
| 6 | moe_fused<false> (MID, cooperative) | 41.70 | 69.49 | 111.76 | +0.06 | 69.55 | 47.9 | 21.65 |
| 7 | unpooler sk2<1,16,4,1,4,1,5,...> (bit 29: plain launch) | 111.84 | 14.24 | 126.05 | **+0.06** | 14.32 | 11.36 | 2.96 |
| 8 | b2 _graft_row_boundary_pre_cg (16 programs) | 124.03 | 4.80 | 128.77 | -2.06 | 2.72 | 0.35 | 2.37 |

Decode 32 sliding (span 143.07 us, sum of durations 160.61):

| # | kernel | start | dur | end | gap | critical path | floor | over floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | qkv sk2<2,32,3,...> (plain launch) | 0.00 | 10.96 | 10.96 | - | 10.96 | 6.39 | 4.57 |
| 1 | attn_kernel<1,false,1> | 9.58 | 11.26 | 20.85 | -1.52 | 9.71 | 3.78 | 5.93 |
| 2 | o sk2 | 12.19 | 13.31 | 25.41 | -8.74 | 4.58 | 2.84 | 1.74 |
| 3 | b1 _graft_row_boundary_pre_cg (32 programs) | 23.54 | 4.82 | 28.32 | -1.92 | 2.88 | 0.71 | 2.17 |
| 4 | pooler + router sk2 (fused) | 26.00 | 19.82 | 46.06 | -2.29 | 17.52 | 13.26 | 4.26 |
| 5 | route | 44.03 | 3.87 | 47.74 | -2.19 | 1.63 | 0 | 1.63 |
| 6 | moe_fused<false> (B4 WGS, cooperative) | 48.24 | 76.30 | 125.09 | **+0.45** | 76.74 | 55.9 | 20.84 |
| 7 | unpooler sk2<2,32,4,2,2,2,3,...> (bit 29) | 125.66 | 14.58 | 140.26 | **+0.58** | 15.22 | 11.36 | 3.86 |
| 8 | b2 _graft_row_boundary_pre_cg (32 programs) | 138.34 | 4.93 | 143.07 | -2.13 | 2.80 | 0.71 | 2.09 |

The full cells (`tl_v1un.tables.txt`) have the same shape after attention: decode 16 full MoE 69.39, unpooler 14.30
(+0.13 gap), b2 2.74; decode 32 full MoE 80.77 (+0.42 gap), unpooler 15.34 (+0.58), b2 2.83; attention 83.01 /
159.63 us critical against about 73.9 / 147.7 of KV bytes.

Edges. Six of the eight edges are PDL edges whose secondary starts 1.5 to 8.7 us before its producer ends. Two are
serial at both loads: route -> MoE (+0.06 / +0.45 us; the cooperative MoE has no programmatic attribute; the MoE lane's,
and its route-in-MoE form is bounded at -0.25 / +0.42 in the blueprint section) and MoE -> unpooler (+0.06 / +0.58 us):
`_pdl8_bfp_pdl` sets bit 29 (no programmatic attribute) on role "un" at every T except 8 (ly_v1un opened T = 8 only),
so the 50.4 MB unpooler starts after the MoE grid has completed and only then issues its first weight stages.

Excesses over floor ranked by microseconds (critical path minus floor, decode 16 / decode 32 sliding) and owners:

| rank | item | d16 | d32 | owner / status |
|---:|---|---:|---:|---|
| 1 | MoE (MID at 16, B4 at 32) | about 21.7 | about 20.8 | moe lane; split re-stream, loop rate, ring, ranges and tail all worked (rows ly_ws2t, ly_ws2u, ly_mid1, ly_ul_*, ly_b4_*, ly_dual) |
| 2 | attention (q_len 1, 8.4 / 16.8 MB of KV) | 5.82 | 5.93 | attention-d1 lane (the q_len 1 kernel) |
| 3 | qkv sk2, first kernel after the flush | 3.88 | 4.57 | bf16 projections; config space swept (P2 split-K declined, qkv16n2 lost) |
| 4 | pooler + router | 2.68 | 4.26 | bf16 lane ("Pooler + router remaining gap" at T <= 8) |
| 5 | unpooler sk2 behind the serial MoE edge | 2.96 | 3.86 | this lane: candidate ly_d16un |
| 6 | b2 row boundary (and b1 at T = 32: 2.17) | 2.37 | 2.09 | this lane: candidate ly_d16rb |
| 7 | b1 glue (T = 16) | 1.95 | - | blueprint rows (b1d32 null) |
| 8 | route | 1.57 | 1.63 | ruled out in the decode 1 lanes (ly_rpf, route-in-MoE) |
| 9 | o sk2 | 0.84 | 1.74 | bf16 lane |

The largest excess with a concrete mechanism outside docs/ruled-out.md is the unpooler's serial edge (rank 5): the same
change ly_v1un made at T = 8 (where it gave -1.43 / -1.40 us), whose two ruled-out rows ("M-B slice 1", "G Tiny/Tiny2
expert->single sliced-unpool PDL edge") were reopened for T = 8 only. The MID and WGS B4 builds both run
`griddepcontrol.launch_dependents` unconditionally at each CTA's phase B end (helper0 source, after WS_JOIN and outside
the `OKC_WGS` branch), so a programmatic unpooler can launch before the MoE's grid barrier and final pass. The second
(rank 6) is the row boundary's serial second load, below.

### Candidate ly_d16un: the T = 16 / 32 unpooler as a PDL secondary of the MoE

`tools/dl_ly_d16un_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_d16un.py` (SHA256
ff1a2cd1873d6cf818ed413236f0efcbcb1f2e6a5bd3a4a15d537563f888e4aa). One line changes (ly_v1un line 18511, the
candidate's 18512, in `_pdl8_bfp_pdl`): `(role == "un" and T != 8)` -> `(role == "un" and T not in (8, 16, 32))`, so bit
29 stays off for the unpooler at T = 16 and 32 and its sk2 launch gets the programmatic attribute; T = 1 is unchanged
(its slice is the cgemv path). No kernel source changes and no SASS diff: the sk2 binary is the base's, and its `pdl`
argument already carried bit 30 and P (2 at T = 16, 1 at T = 32 from PCX_CFG), so the pre-wait weight stages, the
escape table, every thread's `griddepcontrol.wait`, `fence.proxy.async.global` (upstream generic stores -> TMA reads)
and the x tiles after it are the code the base already ran, with a wait that returned at once. `griddepcontrol.wait`
returns only after the whole MoE grid has completed and its memory operations are visible (PTX ISA 8.7, 9.7.13.13).
The MoE stays the primary, cooperative, with its trigger, fences and grid barrier; all its CTAs are resident before any
secondary CTA can launch. At T = 16 / 32 `pdl8_edges` is None, so the in-prepare PDL8H guard (T = 16 sliding, T = 32
full: paired A/B and the 2000-replay stress against the PDL-off output) runs with the new edge on.

Gate run (the file's own prepare, no pin, `gate_d16un.log`): `PDL8_AB T=16 sliding: off 157.92 on 150.72 us, saving
7.20, wins 9/9`; full 8.26 (9/9); T = 32 sliding 4.58 (9/9), full 6.62 (9/9); `PDL8H_GUARD T=16 sliding`: ids
mismatches 0 / 0, on drift 3.98e-03 / 4.19e-03 against off 3.58e-03 / 4.08e-03, bound max(2e-2, off): pass, T stays on;
`PDL8H_GUARD T=32 full`: on drift equal to off (5.52e-03 / 6.41e-03), pass; T = 1 and T = 8 guards pass; enabled at
every T. Correctness (chk step, six loads, `--runs 5`): reference check PASS 12/12, `G_RELEASE_STARTUP_OK native
(128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, `CG_GATE ... equal 1023/1024 ok` (the control's line).

Paired A/B against ly_v1un (`tools/ly_prtl_ab.sh`, CTL=ly_v1un.py, c/w/c/w/c per round, 60 runs, PDL pinned on at every
T, router seed 106218864389449120, f931 case seeds, local_start per input; two rounds 18:33 to 18:37 UTC; table
`profiles/box/d16tl/prtl/d16un_table_both_rounds.txt`). Per round, decode 16 sliding / full w - c: -1.32 / -2.13 us
(4/4, 4/4 pairs), -2.20 / -2.00 (4/4, 4/4); decode 32 sliding / full -0.93 / -0.88, -1.02 / -1.28. Pooled (6 control
runs, 4 candidate runs, 24 pairs per cell):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 64.65 | 1.23 | 64.56 | -0.08 | 2.46 | 17/24 | inside |
| decode 1 full | 70.73 | 0.53 | 70.58 | -0.15 | 1.06 | 10/24 | inside |
| decode 16 sliding | 127.29 | 0.88 | 125.53 | **-1.76** | 1.76 | **24/24** | inside by 0.002 us (win in round 2) |
| decode 16 full | 199.83 | 0.99 | 197.77 | **-2.06** | 2.00 | **24/24** | win |
| decode 32 sliding | 144.23 | 2.11 | 143.25 | **-0.97** | 4.22 | **21/24** | inside |
| decode 32 full | 295.28 | 1.23 | 294.20 | **-1.08** | 2.95 | **20/24** | inside |
| verify 1 sliding | 101.35 | 1.47 | 101.06 | -0.29 | 2.94 | 17/24 | inside |
| verify 1 full | 108.94 | 1.26 | 109.02 | +0.08 | 2.53 | 14/24 | inside |
| verify 16 sliding | 242.38 | 4.53 | 242.30 | -0.07 | 9.06 | 14/24 | inside |
| verify 16 full | 325.99 | 4.56 | 325.89 | -0.10 | 9.12 | 15/24 | inside |
| verify 32 sliding | 293.97 | 1.07 | 294.02 | +0.05 | 2.94 | 13/24 | inside |
| verify 32 full | 459.15 | 1.81 | 459.00 | -0.15 | 4.59 | 14/24 | inside |

The eight unchanged cells (the code differs only at T = 16 / 32) sit within 0.3 us of the control, the session null.

Candidate timeline (same instrument, `profiles/box/d16tl/tl_d16un/`, `tl_d16un.tables.txt`, net 125.41 / 201.33 /
141.66 / 294.29 us): the unpooler now starts 1.41 / 1.36 us (decode 16 sliding / full) and 2.37 / 2.38 us (decode 32)
before the MoE's end instead of 0.06 to 0.58 after it; its critical path after the MoE drops 14.32 -> 13.25 and
14.30 -> 13.23 at decode 16, 15.22 -> 14.37 and 15.34 -> 14.45 at decode 32 (-1.07 / -1.07 / -0.85 / -0.89 us); the
MoE and b2 rows stay within their input spread.

### Candidate ly_d16rb: the row boundary's Stream row loaded with its Branch row (null)

The b2 row boundary (`_graft_row_boundary_pre_cg`, Gluon, one 32-warp program per token row; b2 at T = 16, b1 and b2 at
16 < T <= 32, launched nowhere else) has a 2.7 to 2.8 us critical path after its producer for a 0.35 / 0.71 us floor.
After its `_gdc_wait()` it loads the Branch row, reduces it over the CTA, and only then loads the Stream row. SASS of
the base's cubin on the box (Triton cache, POST build, 44 registers): the Branch loads `LDG.E.64.STRONG.GPU` at
0x01b0 / 0x01c0, `BAR.SYNC` at 0x04a0, `MUFU.RSQ` at 0x0580, the Stream loads `LDG.E.128.STRONG.GPU` at 0x06d0 / 0x0710:
the `.cg` loads compile to strong GPU-scope loads that ptxas keeps behind the barrier, so the Stream row's latency is
serial after the first reduction.

`tools/dl_ly_d16rb_patch.py` (base ly_v1un, pin checked) writes `muse_decode_layer/cand/ly_d16rb.py` (SHA256
dbf7ecb882dabc205926ad76110ca7bc46413b9699f58f7a4ae975895a4f39da). One line moves (ly_v1un line 18322, `s =
_gl.load(Stream + row * H + c, cache_modifier='.cg')`, to directly after the Branch load at line 18317; candidate
line 18319), both after the wait; the math, its order and the stores are unchanged. SASS diff (POST build): the
Stream loads now at 0x04a0 / 0x04b0, before the first `BAR.SYNC` (0x04f0); 51 registers (base 44), same instruction
count to EXIT (0x0d50), 0 local; the POST = False build (b1 at T = 32) moves the same way (Stream at 0x0410, BAR.SYNC
0x0460). Gate: PDL8H guards pass at every T; chk PASS 12/12, startup OK, CG_GATE as the control.

Paired A/B against ly_v1un, two rounds, 24 pairs per cell (18:38 to 18:42 UTC,
`profiles/box/d16tl/prtl/d16rb_table_both_rounds.txt`): decode 16 sliding +0.22 us (7/24 pairs below), full +0.33
(10/24), decode 32 sliding -0.29 (16/24), full -0.18 (13/24), the other eight cells -0.37 to +0.44, all inside.
Candidate timeline (`tl_d16rb.tables.txt`): b2's critical path 2.75 / 2.75 at decode 16 (base 2.72 / 2.74) and 2.72 /
2.72 at decode 32 (base 2.80 / 2.83), b1 at T = 32 2.70 / 2.66 (base 2.88 / 2.72). The second load's latency was not
what holds the row boundary at 2.7 us; what does (the producer's completion and flush before the wait returns, the
first-program launch, the two CTA-wide reductions of 1024 threads) is not separated here. Null, ruled-out row.

### Decision

ly_d16un is the lane's candidate: decode 16 sliding / full -1.76 / -2.06 us at 24/24 pairs each, decode 32 sliding /
full -0.97 / -1.08 at 21/24 and 20/24, the eight other cells within 0.3 us (session null), guards clean at T = 16 and 32
with the new edge, check PASS 12/12. It meets the lane's registration bar (decode 16 sliding at least 1 us with 20+ of
24 pairs, nothing else losing beyond noise). Value if the box deltas carried to the judge (hypothesis; the box rule
forbids reading them as a prediction): 1.76 x 0.096 + 2.06 x 0.021 + 0.97 x 0.085 + 1.08 x 0.014 = about 0.31% of the
score, below one draw's spread, so one check establishes validity (13/13 and the PDL8H guard on the judge's prepare
with the T = 16 / 32 edge) rather than rank. It changes one Python expression on the T = 16 / 32 path only, so it
stacks with the T = 1 and T = 8 lanes by construction (it is ly_v1un's own T = 8 expression widened). A ledger
registration line proposing the check is written; this lane dispatched no judge job. ly_d16rb is null.

Left open: the unpooler still runs 13.2 / 14.4 us after the MoE for an 11.36 us stream (it now starts 1.4 / 2.4 us
early, but its first stages land only as MoE CTAs exit, as at T = 8); the pre-wait stage count at T = 16 / 32 (P = 2 /
1) was set when the edge had no attribute and is untuned for it (at T = 8, P = 4 lost to P = 2); the row boundaries'
2.7 us are unstamped.
## 2026-11-04 - Verify 1 sliding attention in the layer (box)

Attention-v1 lane. Question: the "Verify 1 timeline (box)" section puts `attn_kernel<8, false, 15>` (the T = 8
instance at verify 1 sliding: per-request grid of 33 splits, one 64-key tile each, OKC_SPEC speculated length,
cooperative finalize, one workspace copy) at 6.75 us on the critical path against a 0.12 us byte floor. Where do the
6.75 us go, which K/V bytes are read before `griddepcontrol.wait`, and can one change take the excess? Base:
`muse_decode_layer/cand/ly_v1un.py` (SHA256 88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145), never
edited. Box only (okc-h200, 18:31 to 19:01 UTC), every run under gpu.lock, no judge check. Logs, JSONs and tables:
`muse_decode_layer/profiles/box/atst/`.

### Instrument (never a candidate)

`tools/dl_ly_atst_patch.py` writes `muse_decode_layer/cand/ly_atst.py` (771c7f9b) from ly_v1un: the attention's own
DBG stamps switched on (OKC_DBGON 1, OKC_MRGDBG 1, 32 slots per CTA, thread 0, written only when `p.dbg` is set) plus
five slots (pre-wait work done, wait returned, q group landed, ticket atomic returned, SM id), and entry / wait /
output-store stamps in the sk2 kernel for the T = 8 qkv (narrow build) and o launches. `tools/ly_box_atst.py`
captures the whole layer twice per cell (plain, instrumented; PDL pinned on at every T as bx_box_run does), replays
20 fresh inputs (local_start per input, router seed 106218864389449120, f931 case seeds) through both with a cold L2
and reports times from `qkv_end`, the last qkv CTA's output store. The base's stamps: `atst1.{log,json}`.

### The 6.75 us as a sum of parts (measurement, verify 1 sliding, medians over 20 inputs, us after qkv_end)

| part | from | to | us |
|---|---:|---:|---:|
| qkv's last store to the attention's `griddepcontrol.wait` return (all 33 CTAs within 0.03 us) | 0.00 | 0.42 | 0.42 |
| q / new k,v / constants group issued, page-index group waited, CTA barrier, K/V copies issued, q group landed | 0.42 | 1.12 | 0.70 |
| RoPE table, q pass (RMS norm, RoPE, e4m3 into the wgmma A tile) | 1.12 | 2.21 | 1.09 |
| new k,v pass and cache-row writes, new-row patch | 2.21 | 2.27 | 0.06 |
| consumer setup, tile 0 wait + CTA barrier | 2.27 | 2.64 | 0.37 |
| tile math (QK^T wgmma, softmax, P.V) | 2.64 | 3.46 | 0.82 |
| merge: 8 `red.global.add.v4` + 1 `red.add` per thread issued (32 KB per CTA into one copy) | 3.49 | 4.34 | 0.85 |
| ticket: thread 0's `atom.acq_rel.gpu` returns (the CTA's reds performed) | 4.35 | 5.22 | 0.87 |
| grid barrier: last ticket (5.79, the split holding the new rows) seen by every CTA | 5.22 | 5.92 | 0.70 |
| finalize: 4 rows per CTA, sum the copy, rmsnorm, gate, BF16 store (last CTA) | 5.92 | 6.72 | 0.80 |
| attention end | | 6.72 | **6.72** |

The o kernel's `griddepcontrol.wait` returns 0.42 us after the attention's last store (7.14). The o kernel's first CTAs
enter at 1.18 us (the attention triggers right after its own wait), so o's pre-wait weight stream runs on the same SMs
and memory system through the whole merge. The critical CTA is split 32, the one holding the new rows: it does the new
k,v pass (+0.3 us), the patch (+0.15) and a masked tile, and its ticket comes last in every input (5.70 to 5.79 against
5.06 to 5.38 for the others). Verify 1 full (`<8, false, 11>`, 252 splits, 8 copies): wait return 0.48, q landed 1.50,
tile 0 landed 3.23, merge start 9.22 (median; several tiles per split), last ticket 12.62, end 14.13 (the timeline's
13.93).

Pre-wait work today (entry -1.25 us, pre-wait work done -0.69): the pooler L2 prefetch (`cp.async.bulk.prefetch`,
4 MB over the 33 CTAs), the `seq_lens` load and the page-index `cp.async` of the CTA's tile (64 entries from the
speculated length), then the CTA idles in `griddepcontrol.wait` for 1.1 us. **No K/V byte is read before the wait:**
the K/V copies are issued at 0.99 us, after the wait, after the q group's issue and after the page-index group's
wait (0 of 33 CTAs before the wait in all 20 inputs).

### Which operand may be read before `griddepcontrol.wait` (source proof)

The qkv kernel is a plain launch (bit 29), so every kernel before it has completed when it starts, and the attention
can overlap only qkv. qkv writes only the qkv buffer. So, per operand of the attention:

- q rows, new k / v rows, gate rows: columns of the qkv buffer, the predecessor's output: after the wait only.
- `seq_lens`, `page_table`, `cache_loc`, `k_scale`, `v_scale`, `inv_freq`: layer inputs or prepare-time constants
  that no kernel of the layer writes: legal before the wait (seq_lens and page_table already are).
- `k_cache` / `v_cache` rows at positions < seq (the cached window): layer inputs. The only kernel of the layer that
  writes either cache is this attention kernel itself, after its wait, at `cache_loc[j]` for the 8 new positions
  (`S.cloc`, the new-k,v pass); `kernel_fn`'s agent path has no other cache writer (the `_qk_prep` writer belongs to
  the FA3 fallback). The ONE straight-line issue copies only rows with position < min(end, seq), zero-fills rows past
  the range without a read, and skips the new rows (patched from qkv after the wait). Legal before the wait.
- the workspace epoch (`ws_cnt + 24576 + bp`) and the zeroed workspace: written by the previous launch of this kernel,
  which completed before qkv started: legal, unused here.
- `out` and the workspace are written only after the wait.

### Candidate ly_apw: the window's K/V before the wait (loss/null)

`tools/dl_ly_apw_patch.py` writes `muse_decode_layer/cand/ly_apw.py` (SHA256
73e26604def283ef9e93ef4caa5ee26202e4fc939221ccb10d9f6d05a2f4026f). One change in ATT7_CUDA_SRC's `segment()`, behind
`constexpr bool PREKV = OKC_PREKV8 && L == 8 && ONE && !WS && !S2` (only the one-tile q_len 8 per-request builds;
every q_len 1 and multi-tile instance compiles the base text): before `pdl_wait` the CTA waits for its page-index
group, runs the CTA barrier and the OKC_SPEC length check, and issues its tile's K/V (the unchanged ONE issue text);
after the wait the q group is issued as before and one `cp.async.wait_group 0` + barrier replaces the base's two
partial waits. Wait placement, trigger, fences and the merge are unchanged; nothing global is written before the wait.

Stamps (`ly_atst_apw.py` cd2b6e34, `atst_apw.{log,json}`): the page indices land at -0.08 us (1.2 us after their
issue, with qkv streaming at full DRAM rate), the K/V copies go out at 0.19 (32 of 33 CTAs before the wait returns
at 0.45), but q lands at 1.10 (base 1.12), tile 0 is waited at 2.69 (base 2.64) and the attention ends at 6.69 (base
6.72). The K/V tile was never what the post-wait chain waited on: the 0.37 us between the patch and the tile-0 stamp
is consumer setup and the CTA barrier, and it stays 0.43 with the tile in shared memory long before.

Gate run (the file's own prepare): PDL8_AB T = 8 savings 6.30 / 7.62 us (9/9), PDL8H_GUARD pass at both T = 8 cells
(ids mismatches 0), reference check PASS at both verify 1 cells. Check: PASS 12/12, `G_RELEASE_STARTUP_OK`, CG_GATE
line equal to the control's. Paired A/B against ly_v1un (`tools/ly_prtl_ab.sh` via `tools/ly_atw_run.sh`, c/w/c/w/c,
60 runs, six loads, PDL pinned on, two rounds; `atst/apw/table_both_rounds.txt`):

| cell | ctl mean | spread | cand mean | w - c | pairs < 0 |
|---|---:|---:|---:|---:|---|
| decode 1 sliding | 64.57 | 0.90 | 64.73 | +0.16 | 10/24 |
| decode 1 full | 70.76 | 0.69 | 70.83 | +0.07 | 11/24 |
| decode 16 sliding | 127.23 | 0.88 | 126.93 | -0.30 | 18/24 |
| decode 16 full | 199.55 | 2.06 | 199.39 | -0.16 | 12/24 |
| decode 32 sliding | 144.46 | 1.23 | 144.32 | -0.15 | 14/24 |
| decode 32 full | 295.38 | 1.70 | 295.45 | +0.07 | 10/24 |
| verify 1 sliding | 100.93 | 1.12 | 101.25 | **+0.31** | **8/24** |
| verify 1 full | 109.31 | 1.60 | 109.12 | -0.20 | 13/24 |
| verify 16 sliding | 242.01 | 1.94 | 243.04 | +1.03 | 2/24 |
| verify 16 full | 325.79 | 1.01 | 325.76 | -0.04 | 13/24 |
| verify 32 sliding | 294.46 | 2.75 | 293.89 | -0.57 | 19/24 |
| verify 32 full | 458.91 | 1.66 | 459.23 | +0.32 | 11/24 |

Every cell inside its threshold; verify 1 sliding +0.31 us with 8 of 24 pairs below the control (round 1 -0.18,
round 2 +0.8); the other cells run unchanged code (verify 16 sliding's +1.03 is the session's, its kernel is the
multi-tile instance). Not kept; ruled-out row.

### Candidate ly_sngw2: one workspace copy per split, plain stores, ordered sum (loss)

The merge tail (reds issued to the finalize's last store, 3.49 to 6.72 us) is the largest block. Hypothesis: 33 CTAs
`red.add` into the same 8,192 floats, so the reds' issue (0.85 us) and their drain before the ticket (0.87) are
same-address atomic serialization. `tools/dl_ly_sngw_patch.py` writes `muse_decode_layer/cand/ly_sngw2.py` (SHA256
4a7f2497bd246626c1596630803e8966f7f56d2781a957a57f2a34b5de5f059c). One change: `_plan7`'s one-request q_len 8 sliding
plan takes nc = n_splits (each split owns a workspace copy); in the kernel, behind `constexpr SNGW = OKC_SNGW8 && L == 8
&& ONE && !FLAT && cooperative` and `p.nc == p.n_splits`, the merge writes the same rescaled partials with
`st.global.cg` instead of `red.global.add`, the ticket / grid barrier / every fence stay as they are, and the finalize
(`fin_sngw`) sums each row over the 33 copies in split order (thread = row, 4 dims, copy phase; quad butterfly, so the
merge is deterministic), zeroes what it read (the workspace stays zero between calls, as the red-based plans assume)
and applies the same rmsnorm * sigmoid(gate). A first build (ly_sngw 0302ec93, one dimension per thread, 40 loads per
thread) had the same finalize time, so the cost is not its register footprint.

Stamps (`ly_atst_sngw2.py` 52834b3e, `atst_sngw2.{log,json}`): plain stores issue in 0.61 us (reds 0.85), tickets
return at 4.86 median / 5.44 last (base 5.22 / 5.79), the grid barrier passes at 5.58 (base 5.92), but the finalize
takes 2.4 us (5.58 to 7.95, every CTA alike; base 0.80) and the attention ends at 8.10 (base 6.72). Reading 33 copies
moves 33 KB per CTA through its SM's L2 port and writes 33 KB of zeros back, against 1 KB read and 1 KB zeroed in the
base, while o's 256 CTAs stream their weights through the same SMs; the red merge lets L2 do the reduction, so each CTA
sends its 32 KB once and reads back only its rows.

Gate run: PDL8_AB T = 8 savings 6.40 / 7.36 us (9/9), PDL8H_GUARD pass at both T = 8 cells, reference check PASS at
both verify 1 cells. Check PASS 12/12, `G_RELEASE_STARTUP_OK`, CG_GATE line equal to the control's. Paired A/B against
ly_v1un, two rounds (`atst/sngw2/table_both_rounds.txt`): verify 1 sliding **+1.18 us, 1 of 24 pairs** below the
control (rounds +0.51 / +1.84), verify 1 full -0.39 (19/24, unchanged code), other cells -1.12 to +0.68, all inside.
Loss; ruled-out row.

### Decision

No candidate. The verify 1 sliding attention's 6.72 us after qkv's end is a dependent post-wait chain (measurement):
q landing 0.70, q pass 1.09, consumer setup 0.37, tile math 0.82, merge reds 0.85 + drain 0.87, wait for the
new-rows split 0.70, finalize 0.80. K/V DRAM time is not on it: moving the window before the wait (ly_apw, legal
by the operand proof above) changes nothing measurable, and replacing the red merge by per-split copies trades 0.3 us of
merge for 1.6 us of finalize (ly_sngw2). Nothing registered for the judge. Leads not tested (hypotheses): (1) the q pass
(1.1 us for 128 rows of RMS norm, RoPE and e4m3 per CTA) and the consumer setup (0.37) look like once-per-launch code
from a cold instruction cache rather than arithmetic; a stamp inside the q pass would separate the two; (2) the split
that holds the new rows ends its merge 0.4 to 0.7 us after the others in every input (new k,v pass, patch, masked
tile), so the grid barrier waits for it; giving it fewer cached keys, or computing the new rows' e4m3 form in a
split that has none, would shorten the barrier wait by up to that much; (3) the merge's 32 KB per CTA competes with
o's pre-wait weight stream, which starts 1.18 us after qkv's end because the attention triggers right after its own
wait (OKC_PDLTRIG 0); a later trigger trades that against o's own start (PDL lane's knob).

## 2026-11-04 - Decode 1 qkv_gate in the layer (box)

bf16 lane, 10:25 to 11:00 PT (Mac and box clock). Question: the qkv_gate projection is the first kernel of the decode 1
graph (plain launch, no PDL producer) and sat at 9.9 us on the critical path against a 6.4 us byte floor in "Decode 1
timeline and edges (box)". Where do the 3.5 us go, and does one change shorten the layer? Base
`muse_decode_layer/cand/ly_v1un.py` (SHA256 88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145, the
public file, never edited). Box only (okc-h200), every GPU process under `/root/okc/.build/box/gpu.lock`, router
seed 106218864389449120, f931 case seeds, local_start per input. No judge check. Evidence in
`muse_decode_layer/profiles/box/q1/`.

### What runs at T = 1, and what was tried before (step 0)

The kernel is not the C2 cgemv. `CG_LAYERS = ("unpooler",)` in the base, so qkv_gate at m = 1 runs the static sk2
pick for tokr 8, `qkv8_r176`: `sk2_kernel<1, 8, 3, 1, 4, 2, 6, ...>`, 96 CTAs of 24 rows (one per SM, 36 SMs idle),
4 consumer warps x 2 units x 3 n8 tiles per 18,432-byte stage, 16 stages over K, a 6-slot ring, 4 escape warps,
NACC 1, P = 2 weight stages before the (no-op) griddepcontrol.wait, trigger after the last weight issue. The qkv
cgemv (`cgemv40<256, 2>`, 1152 CTAs) left the layer after the box revival lane's rv_cgfe_six (docs/box-revival.md):
when the isolated tuner picked it, decode 1 lost 1.0 / 3.4 us, and its kernel took 10.0 us against sk2's 9.9.

Earlier m = 1 qkv_gate attempts and why each closed:

| attempt | evidence | why it closed |
|---|---|---|
| cgemv (12-bit CUDA-core GEMV, 2 rows per CTA) on the BF16 board | ruled-out rows "cgemv (12-bit ...)", planar stream, 40 registers; judged 8.66 to 9.40 us | won on the board; in the layer see the next rows |
| cgemv with 2 or 4 row pairs per CTA (`cgemv_r2`, fewer CTAs) | 793602020fd64ce9: tuner +1.2 us | the ramp-geometry family closed for o, the slice and the router at m = 1 ("2026-11-02 ramp candidates") |
| TLB warm-up at kernel start (one `prefetch.global.L2` per 2 MB page) | 00005928fed6437b: no arm saves more than 0.15 us | page walks are not on the critical path |
| denser codes (pair7, LUT) on cgemv qkv | +2.5 to +9.95 us | decode cost exceeds the 1-3% byte saving |
| escape handled inside consumer warp 0 instead of an extra warp | bac995b3: qkv +0.96 us | the schedule loses despite occupancy |
| C2 cgemv40<256, 2> in the layer at T = 1 (checks 8cfa2056, eba613c0) | judged profile qkv 9.0 vs sk2 10.5; pdl8c lost the PDL overlap on the three plain-launch cgemv edges | composition result, then rv_cgfe_six above |
| cgemv as a PDL participant (pdl8c2) | early trigger +1.6 to +2.0 us on qkv; end trigger 0 | trigger placement only |
| sk2 qkv at 16 / 32 tokens: 8 stages, 8 / 12 consumer warps, 16 rows at 2 CTAs per SM, one-wave 4-CTA K-split clusters | 01703396, ad6c673d: every variant lost | measured at m >= 8 only; m = 1 runs the tokr-8 config untuned |
| sk2 producer and escape waits without the suspend-time hint | ba85985b: qkv16 +0.13 | wait form only |

What stayed open: the sk2 kernel at m = 1 itself (its escape-warp protocol, accumulator count and consumer count were
never measured at m = 1 in the layer).

### Instrument (diagnostic only, never a candidate)

`tools/dl_ly_qtl_patch.py` writes `ly_qtl.py` from ly_v1un: per-CTA `globaltimer` stamps compiled only into the qkv
geometry (NT 3, WSK 4, UPW 2, not FUSE) and written only when the host `sk2()` sees N = 2304, M = 1 and the new
`sk2_set_qtl` buffer is set. Slots: entry, weight issue per stage, x issue, escape warp 0's full-barrier return per
stage, consumer warp 0's full-barrier return and slot hand-back per stage (these two sit inside the consumer loop and
cost about 0.4 us of loop time), consumer loop exits, escape `cbar` arrivals, epilogue, final store, SM id.
`tools/ly_box_qtl.py` captures a plain and a stamped graph per cell and replays 10 fresh inputs with a cold L2.
Runs: `stamps_qtl_v1un.txt` (no in-loop stamps), `stamps_qtl2_v1un.txt` and `stamps_st_base.txt` (with them).

### Stamps, base (medians over 10 inputs, us from the first CTA's entry)

| item | sliding | full |
|---|---:|---:|
| CTAs, SMs, waves | 96 on 96, 1 | 96 on 96, 1 |
| last CTA start | 0.26 | 0.27 |
| first weight issue (mean CTA) | 0.41 | 0.44 |
| x tiles issued (after the barrier inits, escape table copy and `__syncthreads`) | 1.00 | 1.04 |
| first stage landed, earliest CTA / mean CTA | 1.63 / 1.88 | 1.66 / 1.90 |
| issue to landing of an 18 KB stage (first ring / later stages) | 2.06 / 1.98 | 2.08 / 2.02 |
| stream rate, 10% to 90% of stages landed | 4.20 TB/s | 4.18 TB/s |
| consumer loop end, mean / last CTA | 7.87 / 8.59 | 7.91 / 8.64 |
| loop end to epilogue start / store | 0.64 / 0.31 | 0.64 / 0.35 |
| kernel end, mean CTA / last CTA | 8.81 / 9.62 | 8.91 / 9.70 |
| byte floor (28.3 MB at 4.443 TB/s) | 6.38 | 6.38 |

Consumer-side stamps (`stamps_st_base.txt`): consumer warp 0 spends 0.32 / 0.35 us processing a stage and waits 0.064
us for the next one (about its own stamp cost), so data sits in shared memory 0.1 to 0.8 us before use and the
consumers set the pace. Yet the producer refills a slot a median 0.26 us after consumer warp 0 hands it back: the
slot also needs all 4 escape warps' hand-backs, and escape warp 0 observes stages a median 0.13 us after the
consumers (p90 0.61 us). In the build without in-loop stamps the same lag shows at the end: the consumers wait 0.64
us at `cbar` for the escape warps.

Split of the 3.44 us between the timeline's 9.81 us (sliding) and the floor:

| part | us | mechanism |
|---|---:|---|
| launch to first CTA entry | about 0.2 | timeline duration minus the last CTA's stamped end |
| ramp: entry to the mean CTA's first stage in shared memory | 1.88 | 0.41 to the first issue (prologue), then 1.3 to 1.5 us loaded DRAM latency while every CTA's 6-stage ring (10.6 MB) is in flight |
| stream: 16 stages, consumer-paced | -0.39 | 5.99 us against the 6.38 floor; the pace is set by consumers plus escape-gated refills |
| wave quantization | 0 directly | one wave, no second wave; 96 CTAs leave 36 SMs idle, which shows only through the consumer pace above |
| tail: loop end to final store | 0.94 | 0.64 waiting at `cbar` for the lagging escape warps, 0.31 store |
| tail: CTA spread (last CTA end minus mean) | 0.81 | uneven per-CTA progress, largest where refills waited on escape warps |

### Candidates (one change each, both reached only at M = 1 on qkv_gate)

`tools/dl_ly_q1_patch.py --mode n|w8` (base pinned to 88fbae86):

- `ly_q1n.py` (SHA256 71fd1780700b8fac86cc513b5750e15ea0251870a46877f99bab8585a27d6a86): base line 16302 in the
  narrow `bfp__run_sk2`, `if layer!='qkv_gate' or not 2<=m<=8:` becomes `not 1<=m<=8`. T = 1 qkv_gate then runs
  the file's own narrow build, which the public file already runs at T = 2..8 (`_de_source`, ext
  `proj_dl_escape_*`): NACC 2, escape warps off the ring (each walks every fourth table entry and reads its x element
  from global memory after `griddepcontrol.wait`; the empty barriers count the 4 consumer warps only), 144 KB ring =
  5 stages, same packed `qkv8_r176` stream and table. The wait, `fence.proxy.async.global`, the trigger and the
  launch attributes are the same source lines as in the base kernel.
- `ly_q1w8.py` (SHA256 466dd289afb9674309d077ae24820ca5b0a21a49cd170b2e6f5680aae5593834): config row
  `('qkv1w8_r176', 'qkv_gate', 2304, 24, 3, 1, 8, 1, 1, 1)` after base line 9575, `bfp__sk2_name` returns it for
  (qkv_gate, m == 1) after base line 9599, and base line 9637 lets its tokr-8 dispatch line through: 8 consumer warps
  of 1 unit instead of 4 of 2, same stream order (from `bfp_sk2_stream`: the unit index wk * UPW + uu is the same
  for NWR 1), same ring and escape warps, NACC 1.

Correctness: reference check PASS 12 / 12 for both (`chk_q1n.log`, `chk_q1w8.log`), `G_RELEASE_STARTUP_OK native ...
56 verified_replays`, CG_GATE line equal to the control's. Numerics (`tools/ly_q1_numerics.py`, 8 seeded inputs at
four scales, the candidate's path against the base `qkv8_r176` kernel in the same process): q1n worst relative L2
1.65e-4, 18,421 / 18,432 elements bitwise equal; q1w8 2.05e-4, 18,421 / 18,432. Both are inside CG_GATE_REL (2e-3);
the differing elements are single BF16 rounding flips, and on every input the candidate is at least as close to FP64
as the base.

### Paired A/B against ly_v1un (`tools/ly_prtl_ab.sh`, two rounds of c/w/c/w/c, 60 runs, six loads, PDL pinned at 1, 8, 16, 32)

Pooled over both rounds (6 controls, 4 candidate runs, 24 pairs; `ab_q1n_tables.txt`, `ab_q1w8_tables.txt`), w - c
in us and pairs with the candidate below the control:

| cell | ly_q1n | ly_q1w8 |
|---|---|---|
| decode 1 sliding | **-1.32, 24/24** | -0.31, 17/24 |
| decode 1 full | **-0.37, 23/24** | -0.29, 19/24 |
| decode 16 sliding / full | -0.37 (18/24) / -0.65 (18/24) | -0.22 (20/24) / -0.19 (14/24) |
| decode 32 sliding / full | -0.10 (14/24) / -0.87 (20/24) | -0.51 (20/24) / -0.04 (15/24) |
| verify 1 sliding / full | -0.15 (14/24) / -0.12 (14/24) | -0.58 (20/24) / -0.84 (24/24) |
| verify 16 sliding / full | -0.67 (16/24) / -0.40 (16/24) | -0.68 (21/24) / +0.33 (10/24) |
| verify 32 sliding / full | -0.04 (12/24) / +0.35 (9/24) | +0.69 (5/24) / +0.03 (12/24) |

Per round, q1n decode 1 sliding read -1.28 and -1.37 us (8 of 8 neighbour pairs negative). Every non-target cell of
both candidates is inside its threshold (max(2 x control spread, 0.5 us or 1%)). The unchanged cells move up to
0.84 us in q1w8 (verify 1 full, 24/24), which runs identical T = 8 code: q1w8 packs a second 28 MB stream, so the
device memory layout differs; q1n allocates nothing new.

### Where q1n's time went (kernel timeline and stamps)

`tools/ly_tl_run.py --kernels`, decode 1, 20 runs, same session (`tl_ly_v1un.txt`, `tl_ly_q1n.txt`): the qkv kernel
9.81 -> 8.85 us (sliding), 10.05 -> 9.06 (full); attention ends 14.69 -> 13.86 and 21.39 -> 20.67. At full, the
attention kernel's own pre-wait startup absorbs part of the earlier qkv end, which matches the smaller full-cell gain
(hypothesis).

Stamps, same instrument on both (in-loop stamps in both; `stamps_st_base.txt`, `stamps_st_q1n.txt`; sliding / full):

| item | base | q1n |
|---|---:|---:|
| consumer warp 0 first stage | 1.77 / 1.68 | 1.81 / 1.83 |
| consumer processing per stage | 0.320 / 0.352 | 0.288 / 0.288 |
| consumer wait per stage | 0.064 / 0.064 | 0.096 / 0.096 |
| refill after consumer warp 0's hand-back | 0.256 / 0.264 | 0.064 / 0.064 |
| consumer loop end, mean / last CTA | 8.28 / 8.66, 8.30 / 8.72 | 7.58 / 7.84, 7.60 / 7.90 |
| `cbar` passed after the loop | 0.21 / 0.22 | 0.08 / 0.08 |
| kernel end, mean / last CTA | 8.87 / 9.57, 8.90 / 9.71 | 8.01 / 8.19, 8.09 / 8.35 |

The escape warps now finish 2.5 us before the loop ends, refills follow the consumers, NACC 2 takes about 10% off each
stage, the consumers now wait slightly longer per stage (closer to DRAM-paced), and the CTA spread falls from 0.70 to
0.18 us. ly_q1w8's consumer parallelism alone moved decode 1 by about 0.3 us: the consumer count was not the binding
piece; the escape-gated refills and the tail were.

What is left in q1n's 8.85 us against the 6.38 floor (hypothesis from the stamps): the ramp (first stage at 1.8 us:
0.4 us prologue plus the loaded first-ring latency), about 0.4 us of epilogue and store, and a stream about 0.3 us
over the DRAM rate. The ramp is the next piece; nothing upstream of qkv in the graph can hide it.

### Decision

ly_q1n meets the lane's registration rule (decode 1 by at least 0.4 us with 20 or more of 24 pairs, nothing else
losing beyond noise): sliding -1.32 us (24/24), full -0.37 (23/24). Registered in the directive ledger for a judge
check; not dispatched by this lane. Weighting sliding over full 46:16 as the six-load sum does gives 1.07 us per
decode 1 cell, about 0.23% of the layer score at 0.212% per us, if the box delta carries to the judge. The edit is
one condition, and the same anchor appears once in the lead's stack ly_stk1 (d7db12a1), whose T = 1 code is
ly_v1un's: `python3 tools/dl_ly_q1_patch.py --mode n --base muse_decode_layer/cand/ly_stk1.py --pin
d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878 --out muse_decode_layer/cand/ly_stk1_q1n.py`
builds the stacked file (SHA256 e010ef869d29387dc85a30d4acbbbcb8248d735c6f6e1ebb49dc105ebc1a31fc; diff against
ly_stk1 is the header and that condition; reference check PASS 12 / 12 on the box with `G_RELEASE_STARTUP_OK` and the
control's CG_GATE line, `.build/prtl/stk1q1n/chk.log`; not A/B'd, every A/B here is against ly_v1un). ly_q1w8 is
below the bar (ruled-out row added).

Gate runs with the file's own in-prepare guard (no pin, no `--skip-pdl8-gate`; decode 1 and verify 1, 20 runs;
`gate_ly_q1n.log`, `gate_ly_stk1_q1n.log`): both files pass PDL8H_GUARD at T = 1 sliding, 8 sliding and full, 16
sliding and 32 full (ids mismatches 0 on both arms, T = 1 max relative difference to the PDL-off output 3.74e-04 /
4.81e-04 on both arms), prepare and timed sets {1, 8, 16, 32} all on, check PASS at all four cells.

## 2026-11-04 - Verify 16 / 32 edges and unpooler stages (box)

Layer-d16 lane, round 2. Questions, in the lead's order: (1) do verify 16 / 32 (T = 128 / 256, 16 / 32 requests x 8
tokens) have the serial edges ly_d16un removed at decode 16 / 32, and does opening them pay; (2) with the MoE ->
unpooler edge programmatic at T = 16 / 32, what is the best number of unpooler weight stages issued before the wait
(P); (3) any mechanism outside docs/ruled-out.md in the remaining decode 16 / 32 excess (pooler + router, row
boundaries). Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878,
judge-valid cf84eae4905f4745), never edited. Box only (okc-h200, 18:45 to 19:10 box UTC), every GPU process under
gpu.lock, no judge job. Evidence: `muse_decode_layer/profiles/box/d16r2/`.

Instrument for the sweeps: `tools/ly_arms.py` (new). One process, one prepare; per cell it captures one CUDA graph
per arm (an arm is a state override and / or module globals set while its graph is captured, so a PCX_CFG entry or a
table read at launch is baked into that graph), runs the reference check of every arm on a held-out input, then
replays all arms on the same fresh input (per-input local_start, cold L2, the judge's replay overhead subtracted) in
rotated and alternately reversed order. Reported: each arm's median net and the median of its paired difference to
the first arm, with the count of negative pairs. Control arms with the first arm's settings ("base2", "off2", "m0b")
measure the instrument's null: -0.66 to +0.70 us at 48 runs, -0.22 to +0.62 at 96 runs. Router seed
106218864389449120, f931 case seeds, PDL pinned at T = 1, 8, 16, 32. Every arm of every sweep passed the reference
check.

### 1. Verify 16 / 32 edges (measurement, and candidate ly_v16pdl: null)

At T = 128 / 256 `_pdl(T)` is False (PDL = False) and T is not in PDL8_T, so `_PDL8` is False and every launch in the
chain is a plain launch. The chain (kernel names from the trace, `tl_stk1_v.tables.txt`): nvjet qkv -> attn_kernel<8,
false, 3> -> nvjet o -> `_graft_row_boundary` (b1) -> nvjet pooler + router (concat, 4608 rows) -> `_graft_route` ->
moe_fused<true> (MAIN, cooperative) -> nvjet unpooler -> `_graft_row_boundary` (b2). Start gaps on ly_stk1, us (verify
16 sliding / full / verify 32 sliding / full): qkv -> attn +0.45 / +0.29 / +0.38 / +0.38, attn -> o +0.77 / +0.67 /
+0.93 / +0.66, o -> b1 +0.43 / +0.45 / +0.42 / +0.35, b1 -> pooler +0.43 / +0.45 / +0.45 / +0.43, pooler -> route
+0.26 / +0.32 / +0.26 / +0.29, route -> MoE +0.42 / +0.43 / +0.42 / +0.42, MoE -> unpooler +0.80 / +0.80 / +0.82 /
+0.88, unpooler -> b2 +0.29 / +0.29 / +0.27 / +0.26; 3.9 to 4.3 us of gaps per cell. Box nets 240.05 / 321.38 /
295.60 / 459.17 us, check PASS.

So every edge is serial, MoE -> unpooler included, but four of the nine kernels are cuBLASLt nvjet GEMMs: the box's
cublasLt.h (13.1, both copies) has no programmatic-launch attribute and the kernels execute no griddepcontrol, so an
nvjet can be neither a secondary (it would read its input without a wait) nor an early-triggering primary. The MoE ->
unpooler edge at T = 128 / 256 is therefore not the T = 16 / 32 edge: its secondary is nvjet. What can be opened are
the four edges whose secondary is our kernel and waits at entry: qkv -> attention (`pdl_wait()` before the first qkv
read; the same binary PDL8D opens at T = 8), o -> b1 and unpooler -> b2 (`_gdc_wait()` is `_graft_row_boundary`'s
first statement), pooler -> route (`_gdc_wait()` before any load, EARLY = False).

`tools/dl_ly_v16pdl_patch.py` (base and pin checked) writes `muse_decode_layer/cand/ly_v16pdl.py` (SHA256
920ed61a420244919c846fc3b693576f177f1ca8b7fe83142d1b6c7a7a343b80): the T > 32 `_graft_row_boundary` call takes
`launch_pdl=_v16e("b2" if post else "b1")` (ly_stk1 lines 12512-12514), the four `_graft_route` calls
`launch_pdl=_v16e("route") and T > 32` (lines 12557, 12568, 18462, 18473; the live ones are the last two), the
attention launcher `... or _v16e("attn")` (line 18504), and an appended block: `kernel_fn` sets `_V16PDL` and the edge
subset from `state["v16pdl_on"]` / `state["v16pdl_edges"]`, and a prepare step `_v16_setup` runs the PDL8H test at
16 x 8 sliding and 32 x 8 full (2000 replays per arm against the PDL-off output; a failure turns that T off) and
prints a paired V16_AB at all four cells. Gate (the file's own prepare, `gate_v16pdl.log`, `arms_v16.log`):
PDL8H_GUARD T = 128 sliding and T = 256 full pass with ids mismatches 0 / 0 and on drift equal to off (5.71e-03 /
6.05e-03, 5.03e-03 / 4.82e-03); the five existing guards pass; check PASS at verify 16 / 32. In-prepare V16_AB
(synthetic inputs, 9 rounds): +0.22 / +0.58 us saved at T = 128, -0.32 / -0.38 at T = 256 (a first build gave -1.18 /
-1.86 and +1.22 / +1.12: the in-prepare A/B swings by more than the effect).

Arms (`arms_v16.log`, 48 runs), paired difference to PDL off, us (sliding / full):

| arm | verify 16 | verify 32 |
|---|---|---|
| all four edges | +0.24 / +0.21 | +0.94 / -0.98 |
| attention only | -0.03 / +0.32 | -0.21 / -1.39 |
| b1 only | +0.22 / -0.16 | +0.03 / -1.18 |
| route only | +0.93 / +0.02 | +0.24 / -1.36 |
| b2 only | +0.10 / -0.24 | +0.26 / -0.85 |
| b1 + route + b2 | -0.40 / +1.18 | +0.35 / -0.90 |

No arm moves a cell beyond the instrument's null with a consistent sign; verify 32 full leans negative in every arm,
including the attention-only one, which reads as the arms' position noise rather than any one edge. The candidate's
timeline (`tl_v16pdl_v.tables.txt`) shows why: the gaps into the opened kernels stay +0.11 to +0.80 us (verify 16 full:
attn +0.80, b1 +0.67, route +0.64). An nvjet primary never triggers early, so a programmatic secondary becomes
launchable only when every primary CTA has exited, about when the plain launch would start it.

Variant ly_v16pre (`--pre`, SHA256 54ea208f08b25f07297d946dede1e7ecf49c6e47b15916d41c854e57832ce78f): with the edge
on, b1 / b2 at T > 32 take the `_graft_row_boundary_pre` body (norm weights, alpha and beta loaded before the wait).
A loss: in-prepare V16_AB -1.25 / -0.48 / -3.01 / -2.43 us; arms (`arms_v16pre.log`, 48 runs) all edges +1.26 / +2.82
/ +3.25 / +3.63, b1 + b2 only +1.52 / +2.13 / +3.01 / +3.65 (0 to 10 of 48 pairs faster), null -0.59 / +0.70 / -0.38 /
-0.66.

### 2. Unpooler pre-wait stages P at T = 16 / 32 (null: the base's P is best)

The unpooler sk2 at T = 16 is `sk2_kernel<1, 16, 4, 1, 4, 1, 5, ...>` (unpooler16h_r100, NSTAGE 5, 256 CTAs, 16
stages of K), at T = 32 `sk2_kernel<2, 32, 4, 2, 2, 2, 3, ...>` (unpooler32_r144, NSTAGE 3, 128 CTAs, 16 stages);
CL = 1 in both, so P = min(PCX P, NSTAGE, 16). Base P: 2 at T = 16 (BFP_PDL_N), 1 at T = 32 (PCX_CFG[("un", 32)]).
Arms on ly_stk1 (`arms_unp.log`, 48 runs; each arm overrides PCX_CFG[("un", T)] for one T, so it is a null at the
other T), paired difference to the base, us (sliding / full):

| arm | decode 16 | decode 32 |
|---|---|---|
| T = 16 P = 1 | +0.48 / +0.18 | +0.32 / +0.74 (null) |
| T = 16 P = 3 | +0.21 / +0.08 | +0.70 / -0.37 (null) |
| T = 16 P = 4 | +1.23 / +0.29 (6/48 faster at sliding) | +0.75 / +0.43 (null) |
| T = 16 P = 5 | +1.44 / +1.06 (10/48, 11/48) | +0.37 / +0.56 (null) |
| T = 32 P = 2 | +0.29 / -0.21 (null) | +1.06 / +0.40 |
| T = 32 P = 3 | +0.42 / -0.14 (null) | +1.34 / +0.70 (5/48 faster at sliding) |

Every other P loses or ties at its own T; P = 4 / 5 at T = 16 and P = 3 at T = 32 lose beyond the null at sliding. As
at T = 8 (ly_v1un4), more of the unpooler's weight stages before the wait does not help: they go out while the MoE's
last CTAs still stream phase B. Nothing to change.

### 3. The remaining decode 16 / 32 excess

Timeline of ly_stk1 at decode 16 / 32 (`tl_stk1_d.tables.txt`, sliding / full): MoE -> unpooler start gap -1.30 /
-1.26 us at decode 16 and -2.42 / -2.30 at decode 32 (the round 1 edge); unpooler critical path 13.12 / 13.04 and
14.10 / 14.32 us against an 11.36 us stream; b2 2.75 / 2.78 and 2.78 / 2.78. Three mechanisms tested.

ly_unpf: the MoE pulls the unpooler's first stages into L2. `tools/dl_ly_unpf_patch.py` writes
`muse_decode_layer/cand/ly_unpf.py` (SHA256 02e12ada96f82db7cd5aa4ce98dde30e8850a1c3fbfeac6dd1ff717aa4fc8d3e). In both
MoE sources (MOE_CUDA_SRC, which the B4 build at T = 32 compiles, and the helper0 source, which MID at T = 16
compiles), right after the CTA's `griddepcontrol.launch_dependents` at its phase B end, thread 0 issues
`cp.async.bulk.prefetch.L2.global` over its 1/grid share of the first UNPF_STAGES[T] of the unpooler's 16
stage-major stages (the stream is 50,331,648 B at both T, so one stage is 3,145,728 B; the share is computed on the
host). Only the MID and B4 builds define OKC_UNPF (the fork's b4 key gains -DOKC_UNPF=1, MOE_MID_DEFINES gains it);
SASS on the box: 16 UBLKPF in the MID and B4 builds, 0 in TINY, TINY2 and MAIN. The prefetch writes nothing and orders
nothing. Arms (`arms_unpf.log`, 48 runs, k = stages prefetched, paired difference to k = 0, sliding / full): decode 16
k = 2 -0.26 / +0.02, k = 4 -0.26 / +0.22, k = 6 +0.34 / -0.30, k = 8 -0.21 / +0.21, k = 12 +0.56 / +0.26; decode 32
k = 2 -0.21 / +0.70, k = 4 +0.50 / +0.11, k = 6 -0.24 / +0.08, k = 8 +0.99 / +0.66, k = 12 +1.09 / +1.28. Timeline at
k = 4 (`tl_unpf_d.tables.txt` against `tl_stk1_d`): unpooler critical path 13.12 -> 12.93, 13.04 -> 12.75, 14.10 ->
14.11, 14.32 -> 14.05 us, MoE within its run-to-run spread. As at decode 1 (moe lane, "Decode 1 slice L2 prefetch from
the MoE"), an L2 hit shortens the unpooler's start by 0.3 us at most: its excess after the MoE is not the DRAM latency
of its first stages.

ly_rbc: the row boundary's and route's geometry at T = 16 / 32. `tools/dl_ly_rbc_patch.py` writes
`muse_decode_layer/cand/ly_rbc.py` (SHA256 d31ec064dada68e26c7d9d8a408e040867131bbdf6928d438ca225b719ab1cbc), ly_stk1
launch for launch with two tables: RB_CFG, the (V, NW) of `_graft_row_boundary_pre_cg` (ly_stk1 lines 18898 and
18906-18907), and RT_CFG, the (num_warps, PARTS, MODE) of the T = 16 / 32 route (the `{16: (8, 1), 32: (8, 1)}[T]`
tuples and `MODE=0`). Arms (`arms_rbc.log`, 48 runs; decode 16 sliding / full, decode 32 sliding / full): row
boundary 16 warps x 4 +0.24 / -0.10 / +0.48 / +0.50, 16 x 8 +0.32 / +0.40 / +1.22 / +1.82, 8 x 8 +0.96 / +0.48 /
+2.21 / +2.78, 4 x 8 +1.10 / +1.20 / +4.99 / +4.93; route 4 warps -0.30 / -0.40 / -0.96 / -0.05, 16 warps +0.34 /
+0.27 / +0.51 / +0.72, 2 parts -0.46 / -0.74 / +0.19 / -0.13, MODE 1 (sort) +0.46 / +0.03 / +0.75 / +1.15. The
route's small arms rerun at 96 runs (`arms_rt2.log`): 4 warps -0.08 / +0.22 / -0.37 / +0.51, 2 warps +0.46 / -0.26 /
-0.02 / +0.08, 2 parts -0.13 / +0.26 / +0.53 / +0.90, 4 warps x 2 parts +0.08 / +0.13 / -0.40 / +0.29, null +0.16 /
-0.22 / +0.10 / +0.21. The row boundary gets slower with every halving of its warps (about 0.3, 1.1 and 2.5 us per
boundary at 16, 32 and 64 elements per thread, from the decode 32 arms where b1 and b2 both use it), so at 8 elements
per thread it is near what one 1024-thread program per row gives; the route is at its optimum within the null.

ly_apf: the attention's pooler L2 prefetch at T = 16 / 32. At T = 1 / 8 the attention launcher passes the first 4 MB
of the pooler stream and each attention CTA prefetches its share into L2 at entry; the gate `in (1, 8)` (ly_stk1 lines
11423 and 18507) leaves it off at T = 16 / 32, although the FUSE pooler there reads the same `sk2["pooler"]` stream
(FUSE is static_asserted to the pooler config's geometry). `tools/dl_ly_apf_patch.py` writes
`muse_decode_layer/cand/ly_apf.py` (SHA256 b8ed6c7c025f4a94d5a0abac6c85b09c9c3fda2d655ae8ce78e0f76438eace81) with the
gate and the cap as a table APF_BYTES. Arms (`arms_apf.log`, 48 runs, MB at T = 16 / 32): 4 MB -0.38 / -0.08 / -0.67 /
-0.29, 8 MB +0.37 / +0.56 / +0.30 / +0.22, 12 MB +0.26 / +1.02 / +0.75 / +0.66, 16 MB +1.26 / +0.82 / +1.86 / +1.49,
24 MB +1.26 / +0.96 / +2.00 / +0.70, null -0.16 / +0.24 / -0.43 / +0.32; at 96 runs (`arms_apf2.log`) 2 MB -0.11 /
+0.14 / -0.21 / +0.50, 4 MB -0.14 / -0.05 / -0.29 / +0.24, 6 MB -0.34 / -0.24 / +0.19 / +0.80, null +0.27 / -0.10 /
+0.24 / +0.62. Null up to 6 MB and a loss from 16 MB (hypothesis: the prefetch then competes with the attention's
own KV reads).

### Decision

No candidate. Task 1: verify 16 / 32 do have serial edges, eight of them, but every one touches an nvjet GEMM or the
cooperative MoE, and nvjet cannot take part in PDL; opening the four edges into our own kernels is a null
(ly_v16pdl) and the pre-wait boundary body at T > 32 a loss of 1.3 to 3.6 us (ly_v16pre). Task 2: the base's P (2 at
T = 16, 1 at T = 32) is the best of P = 1..5 / 1..3 with the programmatic edge. Task 3: the unpooler's 1.7 / 2.7 us
excess after the MoE is not the DRAM latency of its first stages (ly_unpf null at 2 to 12 stages), the row boundary
and route geometries are at their optimum (ly_rbc), and the attention's pooler prefetch does not pay at T = 16 / 32
(ly_apf). The pooler + router at decode 32 (17.52 us against 13.26) was not built against: its geometry space is in
docs/ruled-out.md (pool32w8, pool32n1b, rd pooler 16), and its ring depth is set by the 32-token x tile (32 KB of each
56 KB stage, 3 stages), which only a change to the bf16 lane's sk2 kernel could move. Ruled-out rows added; no ledger
registration line, no judge job.
## 2026-11-04 - qkv_gate at 16 / 32 tokens in the layer (box)

bf16 lane, round 2, 11:10 to 11:30 PT (Mac and box clock). Question: at decode 16 / 32 sliding, qkv sits at 10.3 /
10.9 us on the critical path against a 6.39 us byte floor ("Decode 16 / 32 sliding timeline (box)"). Does the T = 1
defect (escape warps on the ring refilling slots late, fixed by the narrow build in ly_q1n) exist at T = 16 / 32, and
does the narrow build's mechanism shorten the layer there? Base `muse_decode_layer/cand/ly_stk2.py` (SHA256
4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846, never edited). Box only (okc-h200), every GPU
process under `/root/okc/.build/box/gpu.lock`, router seed 106218864389449120, f931 case seeds, local_start per
input. No judge job. Evidence in `muse_decode_layer/profiles/box/q32/`; driver `tools/ly_q32_run.sh`, then
`tools/ly_q32_next.sh`.

### Build map at T = 16 / 32 (source, step 0)

Every dense projection at T = 16 / 32 runs the wide sk2 build (`bfp__get_ext()`, the `bfp_SK2_PICK` row for the
load's tokr). The narrow build (`_de_source`, NACC 2, escape warps off the ring) existed only for qkv_gate at tokr 8,
reached at T = 1..8. In every wide pick the escape warps sit on the ring: each walks every stage and hands every slot
back, so the empty barriers count consumer warps plus NESC (4, or the stage count when the ring has fewer than 4
slots).

| role | T = 16 | T = 32 |
|---|---|---|
| qkv_gate | `qkv16_r176`: sk2_kernel<1, 16, 3, 1, 4, 2, 5, ..., NACC 1>, 96 CTAs of 24 rows, stage 34,816 B (x 16 KB + weights 18 KB), 16 stages, 5 slots, NESC 4; plain launch (bit 29), P 2 | `qkv32_r208`: sk2_kernel<2, 32, 3, 1, 4, 2, 4, ..., NACC 1>, 96 CTAs, stage 51,200 B (x 32 KB + weights 18 KB), 4 slots (the most that fit), NESC 4; P 1 |
| o | `o8h_r88`: sk2_kernel<1, 16, 4, 1, 4, 1, 4, ..., NACC 2>, 256 CTAs of 32 rows, 4 stages = all of K, NESC 4 | `o32h_r100`: sk2_kernel<2, 32, 2, 2, 2, 2, 3, ..., NACC 1>, 256 CTAs, 4 stages through 3 slots, NESC 3 |
| unpooler | `unpooler16h_r100`: sk2_kernel<1, 16, 4, 1, 4, 1, 5, ..., NACC 2>, 256 CTAs of 32 rows, 16 stages, 5 slots, NESC 4; PDL secondary of the MoE (ly_d16un) | `unpooler32_r144`: sk2_kernel<2, 32, 4, 2, 2, 2, 3, ..., NACC 1>, 128 CTAs of 64 rows, 16 stages, 3 slots, NESC 3; PDL secondary |

### Instrument (diagnostic only, never a candidate)

The round 1 stamp instrument with one host change: `tools/dl_ly_qtl_patch.py --m-any` stamps the qkv_gate launch
(N 2304) at every M, not only M = 1. `ly_qtl_stk2.py` (2f4fb010) and `ly_qtl_q32n.py` (ad3691ed) are the instrumented
twins. `tools/ly_box_qtl.py --tokr 16|32` picks the config and ring depth; `tools/ly_q32_sttab.py` makes the table.
Ten fresh inputs per cell, cold L2, in-loop stamps on in every run (they cost about 0.4 us of loop time in both arms).

### Stamps (medians over 10 inputs, us from the first CTA's entry; `stamps_table.txt`)

| item | base T = 16 (sliding / full) | q32n T = 16 | base T = 32 | q32n T = 32 |
|---|---:|---:|---:|---:|
| last CTA entry | 0.22 / 0.22 | 0.29 / 0.24 | 0.26 / 0.27 | 0.32 / 0.32 |
| x tiles of the pre-issued stages out | 0.97 / 0.95 | 0.99 / 0.96 | 0.80 / 0.82 | 0.83 / 0.83 |
| consumer warp 0's first stage | 2.06 / 2.05 | 2.04 / 2.03 | 1.67 / 1.66 | 1.66 / 1.64 |
| issue to landing, first ring / later stages | 2.14 / 1.52 | - | 1.54 / 1.09 | - |
| stream rate, 10% to 90% of stages landed | 4.30 / 4.28 TB/s | - | 3.84 / 3.81 TB/s | - |
| stage landing interval (median) | 0.352 | - | 0.448 | - |
| slot refill after its stage landed | 0.48 | - | 0.77 | - |
| refill after consumer warp 0's hand-back | **0.416** | 0.000 | 0.064 | 0.032 |
| consumer processing per stage | 0.320 | 0.288 | 0.384 | 0.384 |
| consumer wait per stage / sum over 15 | 0.064 / 1.31 | 0.096 / 1.33 | 0.096 / 1.57 | 0.096 / 1.68 |
| consumer loop end, mean / last CTA | 8.39 / 9.12 | 7.86 / 8.29 | 9.57 / 10.10 | 9.38 / 10.06 |
| escape warps done, relative to loop end | +0.27 | -0.38 / -0.42 | -0.43 | -2.50 / -2.29 |
| `cbar` passed after the loop | 0.30 | 0.12 | 0.06 | 0.08 |
| kernel end, mean / last CTA | 9.08 / 9.81, 9.07 / 9.90 | 8.39 / 9.30, 8.41 / 8.94 | 10.18 / 10.62, 10.20 / 10.59 | 10.03 / 10.62, 10.01 / 10.45 |
| CTA spread (last minus first end) | 1.36 / 1.42 | 1.18 / 0.78 | 0.72 / 0.59 | 0.88 / 0.72 |

Reading, per load:

- T = 16 has the T = 1 defect, larger. A slot is refilled 0.42 us after consumer warp 0 hands it back (T = 1: 0.26),
  because the refill also needs all four escape warps' hand-backs, and the escape warps finish 0.27 us after the
  consumer loop. Stage landings step by about 0.55 us once every 5 stages (2.08, 2.73 ... 4.45 ... 6.09 ... 7.73),
  one bubble per ring turn, against 0.35 to 0.39 us elsewhere. In the narrow build the refill follows the hand-back
  (0.000 at 32 ns resolution), consumer processing drops 0.320 -> 0.288 us per stage (NACC 2), the loop ends 0.53 us
  earlier on the mean CTA and 0.83 us earlier on the last, and the kernel ends 0.51 / 0.96 us earlier on the last CTA.
- T = 32 does not have it. The escape warps finish 0.43 us before the loop, refills follow the hand-back within
  0.064 us, and the consumers set the pace: 0.384 us of processing per stage at MT 2 (twice the MMAs of T = 16 on the
  same 4 consumer warps), landing interval 0.448 us, stream rate 3.8 TB/s, below the 4.44 TB/s floor rate. The 4-slot
  ring holds 3 weight stages in flight per CTA (about 5.3 MB over 96 CTAs) against the about 4.9 MB that 4.44 TB/s x
  1.1 us of loaded latency needs. The narrow build changes the T = 32 kernel end by -0.15 us on the mean CTA and 0 to
  -0.14 us on the last.

### Candidates (one change each; `tools/dl_ly_q32_patch.py`, base pinned to 4bbf4010)

- `ly_q32n.py` (SHA256 e490ccae8d019c1f796b80070dd33af935cfe688caf8fc9a51af3af3a2d7310d): qkv_gate at T = 9..32 through
  the narrow build. ly_stk2 line 16276 (`_de_source`'s single `launch_sk2<1, 8, ...>` dispatch line) becomes the same
  line behind two new dispatch lines, `if (tokr == 16) launch_sk2<1, 16, 3, 1, 4, 2, 5, true, false, 2, 1, bf16>` and
  `if (tokr == 32) launch_sk2<2, 32, 3, 1, 4, 2, 4, true, false, 2, 1, bf16>` (ring depths from `bfp__sk2_geom` with the
  wide picks' caps r176 / r208, so the same 5 / 4 slots); line 16305 `not 1<=m<=8` becomes `not 1<=m<=32`. Same packed
  streams and escape tables (`state['sk2']['qkv16_r176' / 'qkv32_r208']`). The wait, `fence.proxy.async.global`, the
  trigger and the launch attributes are the same source lines.
- `ly_q32u.py` (SHA256 e5a4d5c99e0717c07d47d53ae991ce535221d516a2aea09112ca9361e0a24cf8): **the unpooler** at T =
  9..32 through the narrow build, with the wide picks' templates, rings and NACC (`launch_sk2<1, 16, 4, 1, 4, 1, 5, ...,
  2, 1>` and `launch_sk2<2, 32, 4, 2, 2, 2, 3, ..., 1, 1>`, keyed on N 8192, K 4096); line 16305 becomes `if not
  ((layer=='qkv_gate' and 1<=m<=8) or (layer=='unpooler' and 9<=m<=32))`. This touches the unpooler kernel build (the
  escape protocol only); its PDL attribute, P and the MoE edge are unchanged.

Numerics (`tools/ly_q32_numerics.py`, the candidate's path against the wide kernel launched directly from the same
packed stream, 8 seeded inputs at four scales per M): q32n qkv_gate at M = 16 / 32 worst relative L2 1.13e-4 (CG_GATE_REL
2e-3), 884,036 / 884,736 elements bitwise equal, the candidate closer to FP64 than the base on all 16 inputs
(`num_q32n.log`); q32u unpooler at M = 16 / 32 bitwise equal, 3,145,728 / 3,145,728 (`num_q32u.log`).

Gate runs (the file's own prepare, no pin; `gate_q32n.log`, `gate_q32u.log`): both files pass PDL8H_GUARD at T = 1
sliding, 16 sliding, 32 full, 8 full and 8 sliding (ids mismatches 0 on both arms), enabled {1, 8, 16, 32}. Reference
check (six loads, `--runs 5`): PASS 12 / 12, `G_RELEASE_STARTUP_OK native ... 56 verified_replays`, `CG_GATE unpooler ...
equal 1023/1024 ok` (the control's line) for both.

### Paired A/B against ly_stk2 (`tools/ly_d16_ab.sh` with CTL=ly_stk2.py: two rounds of c/w/c/w/c, 60 runs, six loads, PDL pinned at 1, 8, 16, 32)

Pooled, 6 control runs and 4 candidate runs, 24 pairs per cell (`ab_q32n_tables.txt`, `ab_q32u_tables.txt`); w - c in
us and pairs with the candidate below the control:

| cell | ly_q32n | ly_q32u |
|---|---|---|
| decode 1 sliding / full | -0.32 (22/24) / -0.23 (17/24) | +0.07 (8/24) / -0.10 (14/24) |
| decode 16 sliding / full | **+0.46 (4/24)** / -0.75 (19/24) | +0.03 (12/24) / +0.02 (13/24) |
| decode 32 sliding / full | **+0.46 (3/24)** / +0.18 (12/24) | +0.14 (9/24) / +0.24 (8/24) |
| verify 1 sliding / full | -0.20 (16/24) / +0.04 (10/24) | -0.06 (15/24) / +0.20 (10/24) |
| verify 16 sliding / full | -0.44 (20/24) / +0.20 (13/24) | +0.56 (8/24) / -0.13 (13/24) |
| verify 32 sliding / full | +0.50 (5/24) / -0.62 (19/24) | -0.81 (20/24) / +0.34 (8/24) |

Every cell of both candidates is inside its threshold (max(2 x control spread, 0.5 us or 1%)). Per round, q32n decode 16
sliding read +0.47 and +0.45 us, decode 32 sliding +0.48 and +0.45: a consistent small loss at both sliding cells.
The cells whose code is unchanged (decode 1, verify 1 / 16 / 32) move by up to 0.8 us in both candidates, the session
null.

Kernel timelines (`tools/ly_tl_run.py --kernels`, 10 profiled cold replays per cell, decode 16 / 32, same session;
`tl_stk2.tables.txt`, `tl_q32n.tables.txt`, `tl_q32u.tables.txt`; control first): decode 16 sliding qkv end 10.30 ->
10.06, attention start 8.58 -> 7.30, attention end 18.62 -> 17.97, o end 22.24 -> 21.68; decode 16 full qkv 10.35 ->
9.97; decode 32 sliding qkv 10.88 -> 10.85, attention end 20.83 -> 20.91; decode 32 full qkv 10.90 -> 11.81. For q32u the
unpooler's critical path after the MoE reads 13.34 -> 13.06, 13.12 -> 13.07, 14.50 -> 14.59, 14.46 -> 14.96 us
(decode 16 sliding / full, decode 32 sliding / full), inside the MoE-end spread.

Where q32n's kernel gain went (hypotheses, not separated today): (1) the narrow escape warps read their x values from
global memory, one 2-byte element per token per escape (up to 41 escapes per CTA, 16 or 32 tokens each); after the L2
flush that is a scattered gather of up to 2.0 / 4.0 MB of 32-byte sectors (96 x 41 x 16 or 32) at T = 16 / 32 beside the 28 MB weight stream,
where the wide build reads the same values from the x tile already in shared memory. At T = 1 the gather is one element
per escape. (2) The escape warps now exit early, so at decode 16 the attention kernel launches 1.3 us earlier as a PDL
secondary and its CTAs share the SMs and DRAM with the qkv tail. Neither explains why the profiled replays show a
0.56 us earlier o end at decode 16 sliding while the paired cold replays show +0.46 us; the paired A/B decides.

### What differs from the earlier T = 16 / 32 qkv rows (docs/ruled-out.md "P2" and "stream-K in one wave")

Those rows changed the geometry (144 to 192 CTAs, 8 / 12 consumer warps, 16-row CTAs at 2 per SM, 2- and 4-CTA K-split
clusters, 128-CTA stream-K with a red.add join) and were measured in the BF16 tuner with an in-process cold L2. Both
candidates here keep the 96-CTA geometry, stream, ring depth and stage order, and change only the escape-warp protocol
(plus NACC 2 for qkv), measured in the layer. The stamps show that this protocol mattered at T = 16 (0.42 us refill gate)
but not at T = 32 (consumer-paced), and the layer A/B shows the kernel-level gain at T = 16 does not reach the step.

### Decision

Neither candidate meets the registration bar (decode 16 or 32 sliding by at least 0.8 us with 20+ of 24 pairs): ly_q32n
loses 0.46 us at both sliding cells (4/24, 3/24) while winning decode 16 full by 0.75 (19/24); ly_q32u is null at every
cell. Weighted by the judge's cell values, q32n is about +0.07% of score (a loss). No ledger registration, no judge job.
Two ruled-out rows. The T = 1 narrow route (ly_stk2's q1n line) is unaffected: its stamps and A/B stand. What is left
at T = 32 qkv is consumer pace (0.384 us per stage at MT 2 on 4 consumer warps) with a 4-slot ring that x tiles fill
to 64%; a next step would move x out of the ring or give the consumers more warps without more CTAs, both outside the
escape protocol and not tested here.
## 2026-11-04 - Decode 1 sliding attention in the layer (box)

Attention-d1 lane. Question: of the 5.02 us that `attn_kernel<1,false,13>` holds on the decode 1 sliding critical
path (it starts 1.70 us before qkv ends and ends 5.02 us after; byte floor 0.12 us; "Decode 1 timeline and edges
(box)"), what are the parts, which K/V bytes may be loaded before `griddepcontrol.wait`, and does a change confined
to the q_len 1 build shorten it? Base: muse_decode_layer/cand/ly_v1un.py (SHA256
88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145, the public layer file, never edited). Box only
(okc-h200), every run under gpu.lock, router seed 106218864389449120, f931 case seeds, local_start per input, PDL
pinned on at every T. No judge check.

### Instrument

`tools/dl_ly_ast_patch.py` writes `ly_ast.py` (instrument, never a candidate): the embedded attention source with
`OKC_DBGON 1` (its own `DBG(k)` slots, written only when the launch's dbg tensor is non-null), DBGSTRIDE 32, new
slots after `griddepcontrol.wait` (15), the q / new k,v / constants group visible (16), SM id (18), and with `--mrg`
the merge steps (19 key halves exchanged, 20 thread 0's reds issued, 21 the CTA barrier before the ticket); the
T = 1 qkv and o sk2 launches stamp entry, wait return and end. `tools/ly_box_ast.py` captures two graphs per cell
(plain, every stamp off; instrumented) and replays each of 20 fresh inputs through both with a cold L2. Times are
us after `qkv_end`, the last qkv CTA's end stamp (its output stores issued). Whole-layer time, instrumented minus
plain replay of the same input: median -1.33 us (ast) and -0.46 us (ast --mrg), per input -4.9 to +2.5 us, the
layer's replay-to-replay spread (MoE), so the stamps do not slow the layer measurably; logs and JSONs in
`muse_decode_layer/profiles/box/ast/`.

### The 5.02 us as a sum of parts (measurement; base, decode 1 sliding, median over 20 inputs; 32 CTAs, one 64-key tile each)

| step | time after qkv_end (p50 / latest CTA) | part on the critical path |
|---|---:|---:|
| attention CTAs enter (launched by qkv's trigger) | -1.07 / -1.02 | (pre-wait: seq_lens, page indices, 4 MB pooler L2 prefetch) |
| page indices landed (stamp after the post-wait barrier) | 0.64 / 0.67 | off the path (landed before the wait) |
| `griddepcontrol.wait` returns | 0.42 / 0.42 | 0.42 qkv's completion and flush |
| q, new k / v, constants visible (issued after the wait, L2 hits) | 0.93 / 0.93 | 0.51 |
| RoPE table | 0.99 / 1.02 | 0.06 |
| q pass (norm, RoPE, f16 pack) | 1.41 / 1.52 | 0.42 |
| new k / v pass (the new token's FP8 cache row written here) | 1.48 / 1.84 | 0.07 (0.43 on the new-token CTA) |
| new rows patched into the stage | 1.60 / 2.02 | 0.10 |
| K/V stage visible (copies issued at 0.83 after the wait, DRAM) | 1.63 / 2.03 | 0.03 / 0.01: off the path |
| tile math done | 2.24 / 2.59 | 0.61 |
| key halves exchanged in smem | 2.37 / 2.72 | 0.13 |
| reds into the workspace issued (thread 0) | 2.50 / 2.83 | 0.13 |
| ticket (`atom.acq_rel.gpu` after the CTA barrier) returns | 4.00 / 4.19 | 1.49 |
| last CTA: finalize start / stores issued | 4.22 / 4.78 | 0.56 |
| o projection's `griddepcontrol.wait` returns | 5.20 | 0.42 attention's completion and flush |

The profiler's 5.02 us end is the attention grid's exit (last CTA's stores at 4.78 plus the exit). Two readings:

- K/V is not on the critical path. The copies go out 0.4 us after the wait, behind the q copies, and land 0.03 us
  (median) after the q side finishes; the q side (L2 round trip for q, then 0.6 to 1.1 us of norm / RoPE / pack /
  new-row work) sets when the tile math can start.
- Every CTA's ticket returns at about the same time (raw stamps of two inputs: loop ends 2.14 to 2.69 us, tickets
  3.90 to 4.22) whatever its own loop end: the merge drains collectively. Diagnostics split the 1.49 us:
  without the 32 x 1,024 output reds (wrong output, `--noacc`) the ticket returns 0.98 us after the barrier (p50
  3.46); without the row-sum reds (`--nol`) it is unchanged (3.98); with the o projection issuing one weight stage
  before its wait instead of two (do0) it returns 0.45 us earlier (3.55). So about 1.0 us is the release plus the
  atomic round trip itself, about 0.5 us the red volume into one 4 KB region, and the o projection's pre-wait weight
  stream, in flight during the merge, slows both.

### Pre-wait legality per operand (source audit of ly_v1un)

qkv is the step's first kernel and a plain launch: `_pdl8_bfp_pdl` gives role "qkv" bit 29 (no programmatic
attribute, ly_v1un.py line 18511), so every writer of a layer input has completed before qkv starts, and the
attention grid launches only after qkv has started. The PDL8 guard and every other edge are unchanged by any of the
operands below.

| operand | written by | read before the wait today | may be read before the wait |
|---|---|---|---|
| seq_lens | layer input | yes (`__ldg` at kernel entry) | yes |
| page_table | layer input | yes (`issue_idx` cp.async, group 0) | yes |
| k_cache / v_cache rows at positions lo .. seq - 1 (the cached window, 2,048 x 128 B each) | layer input (earlier steps / the harness copy) | no (issued after the wait) | yes |
| k_cache / v_cache row at cache_loc (position seq, the new token) | this attention kernel, after the wait (new k / v pass, `cache_row` store by the CTA whose range holds seq; not the qkv epilogue, which writes only the BF16 qkv row) | never read: the ONE copy skips rows in [seq, end) and the patch pass writes them from smem | no (and not needed) |
| cache_loc, k_scale, v_scale, inv_freq | layer input / prepare-time constant | no (group with q) | yes |
| q rows, new k / v rows, gate rows (qkv) | qkv, this step | no | no |
| workspace ws_acc / ws_l / ws_cnt | this kernel (zeroed by the previous launch's finalize) | no | no global access before the wait (writes stay after it) |
| pooler weights (L2 prefetch) | prepare | yes | yes |

Answer: today no K/V byte is loaded before the wait; only seq_lens, the page indices and the pooler prefetch are.

### Variants (stamps, 20 inputs each; o_end = the o projection's last CTA end, the metric an attention-only change moves; base 8.45 / 8.48 in two runs)

| file (SHA256) | change | ticket p50 / max | att end | o wait | o_end | reading |
|---|---|---:|---:|---:|---:|---|
| ly_akv (c82d1faf...) | `tools/dl_ly_akv_patch.py`: the ONE build's cached K/V copies issued before `griddepcontrol.wait` (after the page-index group and the speculated-length check; new-token rows still skipped), q group waited with wait_group 0 | 4.16 / 4.27 | 4.85 | 5.28 | 8.58 | K/V issued about 0.4 us before the wait; stage visible 1.61 vs 1.63, patch 1.58 vs 1.60: nothing moves, as the parts predicted |
| ly_afk (84deab80...) | `tools/dl_ly_afk_patch.py`: no ticket; reds then exit; a one-CTA `attn_fin_kernel<1>` (PDL secondary, trigger at entry, wait, `fin_nc1`) finalizes; o becomes its secondary | - | - | 5.54 | 8.85 | the grid boundary costs more than the ticket: loss |
| ly_anc8 / ly_anc4 (6aaf4977... / 256c234d...) | `tools/dl_ly_anc_patch.py`: 8 / 4 workspace copies at B = 1 (host only) | 3.78 / 3.98, 3.65 / 3.86 | 5.66, 4.99 | 6.06, 5.41 | 9.44, 8.72 | the ticket gains 0.3 / 0.4 but the generic nc > 1 finalize takes 1.66 / 1.10 us: loss |
| ly_anf4 (ddcea269...) | `tools/dl_ly_anf_patch.py`: 4 copies plus `fin_ncn<1,128,4>` (every copy's loads and the gate in one round trip) | 3.56 / 3.84 | 4.64 | 5.23 | 8.50 | finalize 0.77 us; attention ends 0.14 us earlier, o's wait returns 0.6 us after it (0.42 in the base: 16 KB of zeroing stores to drain): null |
| ly_asp16 / ly_asp11 (e6c21972... / a6b48637...) | `tools/dl_ly_asp_patch.py`: 16 / 11 splits at B = 1 (host only; build 9) | 4.91 / 5.10, 5.78 (last) | 5.73, 6.37 | 6.14, 6.84 | - | merge 0.76 / 1.0 us after the barrier, but the multi-tile build's loop ends at 4.08 / 4.75: loss |
| ly_atrg (1abac1d6...) | `tools/dl_ly_atrg_patch.py`: the ONE build triggers its dependents at entry | 4.22 / 4.35 | 4.96 | 5.38 | 8.82 | o launches at -0.11 instead of +1.15; its pre-wait stream now lands on attention's q loads (q visible 1.87 vs 0.93): loss |
| diagnostics | --nol, --noacc (wrong outputs), dpf0 (no pooler prefetch), do0 (o one pre-wait stage), warm replay | see above | | 5.33, 4.90, 5.18, 4.67, 5.47 | 8.62, 8.13, 8.42, 8.62, 8.93 | do0: attention 0.53 us earlier, o 0.15 us later (more stream after its wait); warm (no L2 flush before the stamped replay): q side unchanged (q pass 1.41), so the q side is not cold code fetched from DRAM |

### Paired A/B against ly_v1un (box, 19:22 to 19:31 UTC)

`tools/ly_prtl_ab.sh` with CTL=ly_v1un.py: a check run of the candidate at all six loads (`--runs 5`), then two
rounds of c/w/c/w/c (`--runs 60`, all six loads, PDL pinned on at T = 1, 8, 16, 32 in every arm, `--skip-pdl8-gate`),
pooled by `tools/ly_prtl_table.py` (threshold max(2 x control spread, 0.5 us) at decode 1 / verify 1; pairs = 6
controls x 4 candidate runs). Logs and JSONs in `muse_decode_layer/profiles/box/ad1/`. Net us, candidate minus
control:

| cell | ly_akv w - c (pairs < 0 of 24) | ly_anf4 w - c (pairs < 0 of 24) |
|---|---:|---:|
| decode 1 sliding | -0.00 (12) | -0.11 (13) |
| decode 1 full | -0.22 (19) | -0.32 (20) |
| decode 16 sliding | -0.00 (10) | -0.01 (11) |
| decode 16 full | -0.20 (12) | +0.40 (6) |
| decode 32 sliding | -0.30 (17) | -0.40 (17) |
| decode 32 full | -0.15 (14) | -0.33 (17) |
| verify 1 sliding | +0.12 (11) | -0.08 (13) |
| verify 1 full | -0.09 (13) | -0.66 (24) |
| verify 16 sliding | +0.29 (13) | -0.45 (17) |
| verify 16 full | -0.34 (14) | -1.42 (20) |
| verify 32 sliding | +0.17 (14) | -0.44 (16) |
| verify 32 full | +0.87 (5) | -0.23 (16) |

Reference check PASS at all 12 cells for both candidates, G_RELEASE_STARTUP_OK in both check runs. Every cell is
inside its threshold; the target cell, decode 1 sliding, moves by 0.00 and -0.11 us. ly_anf4's verify 1 full
(-0.66, 24 of 24 pairs) is a cell whose plan and launch it does not change (q_len 8); the kernel module differs
only by the dead `fin_ncn<8, ...>` instantiation in the q_len 8 builds (the `L == 1 && p.nc == 4` test is a run-time
`if`), so this is a code-layout or session effect, not the mechanism; it is recorded, not claimed.

### Candidate ly_acl: the ONE_CL cluster kernel inside the layer

`tools/dl_ly_acl_patch.py` writes `muse_decode_layer/cand/ly_acl.py` (SHA256
5c9bc573a8e5cbee5828e6e3d386913157a6c106dbe554602de9c944e1121ee9): the file embeds
`muse_attention_decode/attn_one.cu` built as its own extension with `-DONE_CL=1` and only the `X(1, 136)` instance,
and decode 1 sliding (batch 1, q_len 1, page size 1, window <= 2048 keys, contiguous 2304-wide qkv rows) launches
`attn_cl_kernel<1>` (16 CTAs, two portable clusters of 8, one KV head and 4 tiles per CTA, DSMEM sum after one
cluster barrier, rank r finalizes head r; no workspace, red, ticket or counter) as a PDL secondary of qkv. Two edits
to the standalone kernel: the page-index group is waited, the speculated length checked and every tile's K/V issued
before `griddepcontrol.wait` (layer inputs only), then the wait, the trigger, and the q / new k,v / constants group;
the cluster launch adds the programmatic attribute. The cluster merge lost +1.31 us standalone (c1becf5a66274d37),
where its launch and K/V pull were on the critical path; inside the layer both can sit before the wait.

Check run at all six loads: reference check PASS at all 12 cells, G_RELEASE_STARTUP_OK. Decode 1 screen
(c/w/c/w/c, `--runs 60`, decode 1 only): decode 1 sliding 64.50 / 64.58 / 64.74 control against 66.30 / 66.30,
+1.70 us, every pair a loss (+1.57 to +1.81); decode 1 full (base code) -0.03. Kernel timeline (`ly_tl_run.py`,
10 profiled replays, `profiles/box/ad1/tl_*.log`): attention ends 6.56 us after qkv's end against 5.08 in the base
session. Stamps (`--stamp`, `ly_ast_acl.py`, 20 inputs, us after qkv's end, p50 / latest CTA):

| step | cluster kernel | base kernel (same quantity) |
|---|---:|---:|
| entry | -0.74 | -1.07 |
| page indices visible | -0.03 | (before the wait) |
| K/V of 4 tiles issued (before the wait) | 1.06 / 1.25 | (after the wait) |
| `griddepcontrol.wait` returned | 1.09 / 1.26 | 0.42 |
| q group visible | 1.63 / 1.81 | 0.93 |
| tiles and new rows ready | 2.15 / 2.74 | 1.63 / 2.03 |
| tile math done | 3.72 / 4.34 | 2.24 / 2.59 |
| CTA partial summed (8 warp slots) | 4.52 / 5.07 | (key-half exchange 2.37 / 2.72) |
| after cluster.sync 1 | 5.13 / 5.42 | |
| peers' rows gathered (DSMEM) | 5.55 / 5.84 | |
| finalize stored | 5.98 / 6.29 | 4.78 (last CTA) |
| o's wait returns | 6.98 | 5.20 |

Reading: no single phase holds the loss. Issuing 32 KB of K/V per CTA (8 copies per thread) takes 1.1 us because the
copies stall issue once the SM's outstanding requests fill, and the thread reaches `griddepcontrol.wait` only after
that, so the wait returns 0.67 us later than in the base; the tile math takes 1.6 us (four tiles' halves over 8
warps, against one tile over 4 warps in 0.6 us), the 8-slot partial sum 0.8 us, the first cluster barrier 0.6 us, the
gather and finalize 0.85 us. The merge without global atomics (partial to stored: 1.46 us) is no cheaper than the
base's exchange, reds, ticket and finalize (2.37 to 4.78: 2.41 us) by enough to pay for the slower math on half as
many SMs. Loss; no second round.

### Decision

No candidate. Decode 1 sliding moves by -0.00 (ly_akv), -0.11 (ly_anf4) and +1.70 us (ly_acl) at the box; every
other variant was stopped at the stamp level because the o projection's end did not move earlier (table above). No
ledger registration, no judge check. What the measurements leave:

- The cached window's K/V is off the critical path today, so loading it before the wait buys nothing; the q side
  after the wait (0.5 us L2 round trip for q, 0.6 to 1.1 us of norm / RoPE / pack and the new token's row) sets the
  tile start.
- The merge is 1.5 us of ticket (about 1.0 us release plus atomic round trip with 32 contenders, about 0.5 us red
  volume into one 4 KB region) plus 0.56 us of finalize plus 0.42 us of completion flush. Copies, fewer splits, a
  finalize kernel and a cluster merge each moved cost between these items without removing it.
- The o projection's pre-wait weight stages compete with the merge (one stage instead of two: tickets 0.45 us
  earlier, o 0.15 us later; an entry trigger: o's stages land on the q loads instead). Attention and o share the
  memory system in the 3 us after qkv; a gain there needs fewer bytes or fewer dependent round trips across both
  kernels (for example q-norm / RoPE in qkv's epilogue, or the finalize inside o's prologue), which is cross-lane
  work (bf16 owns qkv and o) and not tested here.

## 2026-11-04 - Decode 16 / 32 sliding MoE (box)

MoE lane, round 3 (11:15 PT start). Question: of the fused MoE's excess over its byte floor at decode 16 (MID build,
helper0's warp-specialized mma.sync ring) and decode 32 (B4 build, WGS warpgroup kernel), which part has a mechanism not
ruled out, and does one change take at least 1 us at decode 16 or 32 sliding? Base:
muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878), never
edited. Box only (okc-h200), every GPU process under gpu.lock (bx_box_run.py `--lock`), no judge job, no routing slack,
no change to which experts run. Evidence: `muse_decode_layer/profiles/box/moe3/` (logs, A/B tables, SASS reports, raw
per-CTA stamps of 6 inputs per cell as `.json.gz`). Driver: `.build/moe3/ab.sh` on the box (copied there as `ab.sh`):
the file's own prepare at decode 16 / 32 (the five PDL8H guards), the chk step at six loads, two `tools/ly_prtl_ab.sh`
rounds against ly_stk1 (c/w/c/w/c, 60 runs, PDL pinned at 1, 8, 16, 32), then the candidate's stamps.

### Step 0: what was tried on these builds

MID (decode 16): the split expert's re-stream (an expert with 9 to 16 tokens becomes two 8-token jobs, each streaming
its 768 units; 7.3 us of span per split expert, 6.0 us per cell averaged) was attacked three times with two n-tiles per
unit (ly_ws2t +1.7, ly_ws2u +4.1, ly_mid1 +10.6 us), plus the knob sweep's mid8 (decode 16 on MAIN, +2.7). The unit
loop: named-barrier variants (nb, na, pr, pw), software pipelining (pf), early release (er), four ring slots (ns4,
prns4), the DQ7 IMAD merge, two CTAs per SM (ly_dual), phase A unit order and ring depth for the 3.9 TB/s ceiling
(replay and in-place no-math builds), the run-time claim of phase B units (mo_claim), prefetch at job switches (mo_xpf).
At decode 1 (TINY, the same helper0 text): the route kernel prefetching each CTA's first unit (ly_rpf, loss), the credit
fence kind (ly_cfar, null), the MoE trigger at the phase A end (ly_trga, loss), the slice prefetch from the MoE (ly_slpf /
ly_slpfb). B4 (decode 32): the deferred flush behind commit group 6 (ly_b4_fci6, loss), the designated finisher in phase
A (ly_b4_dfa, loss), weights issued before WS_JOIN (ly_b4_wpre, null); on MAIN at verify 16 / 32 the publisher warp
(ly_pub, loss), the cut-unit finish in registers (ly_rfin, a gain at verify 32, MAIN only), its walk orders (ly_rfin2 /
ly_rfin2b, losses), the final-pass row cache (ly_fin1, null), the n-tile switch charge (ly_bal, loss); knobs WGSFCI,
WGSUB / WGSU, nob4, GATHB, NS (noise or loss). Layer edges: the MoE -> unpooler PDL edge (ly_d16un, in ly_stk1), the
unpooler L2 prefetch from the MoE (ly_unpf, null), the route geometry (ly_rbc, null). Left at the start of this round:
the B4 port of ly_rfin, a phase B K-stage order by readiness, the MID prologue, and the two-tile loop with x in
registers.

### Step 1: decode 16 (MID) on ly_stk1, per CTA (measurement)

`tools/ly_moe_tl16.py --base ly_stk1.py` writes the stamped copy (ly_stk1_tl16.py, 6006d80a, rebuilt by the tool, not
committed); `tools/ly_box_moe_tl16g.py --inputs 40 --raw 6` replays the whole layer graph per fresh input with a cold L2
(`tl16_stk1.log`). Medians over 40 inputs, sliding / full, us from the first CTA's entry (the mean CTA enters 0.08 us after the first, the latest 0.19):

| piece | sliding | full | reading |
|---|---:|---:|---|
| span (entry to last exit) | 66.43 | 67.20 | 6144 units at 8 distinct: 51.07 us of bytes at 4.435 TB/s; 47.9 at the judge's 7.5 experts |
| ramp: first ring slot full (mean CTA; latest 5.0 to 5.3) | 3.71 | 3.70 | route() done at 2.73 / 2.72, 2.65 after each CTA's own entry |
| first phase A unit (from slot full) | 1.60 | 1.54 | steady unit 1.15, so +0.45 |
| phase A | 36.74 | 36.66 | unit 1.15 against 1.098 us of bytes (+4.5%, about 2.4 us per CTA); warp 0 waits 2.70 / 2.58 for weights |
| A to B handoff (A end to the first B unit past doneA) | 2.32 | 2.31 | the first B unit retires 4.06 / 4.10 after the CTA's A end: doneA wait 2.2, then x and the unit |
| phase B | 18.62 | 18.35 | unit 1.12 to 1.15 |
| B end to barrier, barrier idle, final pass | 0.80, 2.75, 0.46 | 0.80, 2.76, 0.46 | B-end spread 3.41 / 3.49; the last exit comes 2.6 to 3.1 us after the latest B end (raw stamps) |
| split re-stream | 6.0 averaged | | 768 extra units on 34 of 40 inputs (prior fit; still the largest single part) |

Per CTA (`tools/ly_m3_midcta.py`, `tools/ly_m3_midgaps.py` on the raw stamps): the latest B ends belong to CTAs with the
most units (corr(B end, nA + nB) 0.38 to 0.74) and the latest doneA (corr(B end, first B unit) 0.29 to 0.78). A combined
A + B count split would leave the per-CTA maximum unchanged (6912 units over 132 CTAs: 52 or 53 either way), so the
count is not a lever. Job switches are rare (37 in phase A and 20 in phase B over 6 inputs x 132 CTAs) and cost +0.55 (A)
and +1.1 (B) each.

### Step 1: decode 32 (B4) on ly_stk1, per CTA (measurement)

`tools/ly_moe_tlb4.py --src ly_stk1.py` (level 1: the consumer stage loop untouched) and `tools/ly_box_moe_tlb4g.py
--inputs 40 --raw 6` (`tlb4_stk1.log`). Medians over 40 inputs (the median input has 9 distinct experts, the judge's
target), mean CTA, sliding / full:

| piece | sliding | full | reading |
|---|---:|---:|---|
| span | 78.22 | 80.11 | floor 55.8 at 9 experts |
| ramp: route, gather, WS_JOIN, first weight issue | 1.75, 3.40, 3.43, 4.32 | 1.71, 3.38, 3.41, 4.30 | 0.9 us from WS_JOIN to the first TMA issue |
| phase A | 45.00 | 45.66 | floor 37.2; steady issue interval 0.992 against the 1.064 floor; A-end spread 9.26 / 8.91 |
| phase B | 22.14 | 22.51 | floor 18.6; first doneAs wait 3.19 / 3.12, all doneAs waits 9.35 / 9.44 per CTA |
| MFIN wait (B end to the first final item) | 5.03 | 5.67 | B-end spread 7.23 / 7.41 |
| final pass | 1.96 | 1.88 | exit spread 2.03 / 1.98 |

Phase A per CTA against the split's cost model (`tools/ly_m3_cost.py`, 1584 CTA-phases): least squares gives 1.135 /
1.379 / 1.356 / 1.66 us per stage at 1 / 2 / 3 / 4 n-tiles (the OKC_WGSA table's ratios 1 : 1.21 : 1.25 : 1.41 hold)
and +2.15 us per cut (stream-K) share, which the split does not charge; residual sd 1.08 us against 1.83 for the split's
own cost. 85 to 113 of 132 CTAs hold two cut shares in phase A, and the latest A ends sit in that group
(`tools/ly_m3_cuts.py`), so the cut cost is mostly a constant; what spreads the A ends inside the group is not in the
model.

### Step 2, candidate 1: ly_b4rf, ly_rfin's cut-unit finish on B4 (null)

`tools/dl_ly_b4rf_patch.py` writes `muse_decode_layer/cand/ly_b4rf.py` (SHA256
41c9a389c7d81a87e179b1abdafe14ae5aa2568c012c2840efc55a2c170f6a8a). One line (ly_stk1 line 16428, `_fork_native_moe`):
the "b4" key's defines gain `MAIN_EXTRA_DEFINES` (`-DOKC_RFIN=1 -DOKC_RFINQ=2`). The RFIN path is MOE_CUDA_SRC's
generic WGS text, with no OKC_B4FIN branch; the finisher zeroes the slots and the counter it reads, so the zero-slot
invariant B4 and MAIN share holds. Every fence, barrier and release is ly_rfin's (see "Verify segment switches (box)").
Gate: the five PDL8H guards pass (T = 1 sliding, 16 sliding, 32 full, 8 full, 8 sliding); chk PASS 12/12,
G_RELEASE_STARTUP_OK (its M = 32 case compares B4 against the reference build).

Pooled A/B against ly_stk1, two rounds (19:28 to 19:32 UTC, `table_b4rf_both_rounds.txt`), us:

| cell | ctl mean | spread | cand mean | w - c | pairs < 0 |
|---|---:|---:|---:|---:|---|
| decode 16 sliding / full | 125.58 / 197.88 | 1.09 / 1.34 | 125.38 / 198.12 | -0.19 / +0.24 | 14 / 9 of 24 |
| decode 32 sliding / full | 143.44 / 294.58 | 1.36 / 1.09 | 143.87 / 294.37 | +0.43 / -0.22 | 5 / 17 of 24 |
| the other eight cells | | | | -0.15 to +1.38 | verify 16 sliding +1.38 with 0/24 on unchanged MAIN code: session drift |

Stamps of the candidate (`tlb4_b4rf.log`, the same instrument, medians over 40 inputs, sliding): phase A 45.00 -> 44.16 and
the A-end spread 9.26 -> 7.68, but the x producer's first doneAs wait 3.19 -> 5.61 and all its doneAs waits 9.35 ->
11.51 (in range order each CTA's last phase A unit is its cut unit, finished at the phase end), MFIN wait 5.03 -> 5.80,
span 78.22 -> 77.70. As at verify 16 / 32, the cut cost leaves phase A and returns as phase B's dependency wait; at
decode 32 nothing is left over. Null.

A phase B K-stage order by readiness (each B unit's 16 stages sorted by when the two phase A units they read complete,
both producers walking the same permutation) was modelled before building: `tools/ly_m3_bsim.py` replays every CTA's
phase B on the raw stamps (stage n starts at max(previous end, its A units' completion proxy + lag)). With the stage
time and lag set so the forward order reproduces the measured B ends (1.2 us, 3.0 us), sorting moves the latest B end
by 0.69 / 0.39 us (base walk) and 0.12 / 0.09 us (ly_b4rf's walk). Not built.

### Step 2, compile-only: ly_mid2, the two-tile MID loop with tokens 8..15 in registers (rejected on SASS)

`tools/dl_ly_mid2_patch.py` writes ly_mid2.py (ede3cc19, not committed): ly_mid1's single weight read at decode 16 with
the second n-tile's B fragments loaded at the job switch like tile 0's, so the loop gains only the second mma per k16
step. MID SASS (`sass_mid2_mid_report.json`, `sass_mid2_ptxas_v.txt`): 168 registers, a 136-byte stack frame, 336 bytes
of spill stores and 216 of spill loads, a phase A loop of 944 instructions with 32 STL and 20 LDL (control: 144
registers, loop 692, no stack). 168 is the cap: helper0 launches 9 warps (8 compute and the helper), so one SM
sub-partition holds 3 warps, and 3 x 32 x R <= 16384 registers gives R <= 168. Tile 0's x fragments alone are 64
registers per thread, so a register-resident second tile cannot fit beside the 144 the loop needs. Not run.

### Step 2, candidate 2: ly_m16r, the MID prologue's global loads before their uses (null)

The MID SASS of ly_stk1 shows two serialized round trips in the prologue: the FAST1 warp's OKC_SPLIT count loop issues
one topk_ids load per 32 entries and consumes it (MATCH.ANY, ATOMS) before the next (four dependent loads at M = 16),
and thread 0 stores S.gen / S.ssp from its volatile p.ctr[3] load before its own route() load. `tools/dl_ly_m16r_patch.py`
writes `muse_decode_layer/cand/ly_m16r.py` (SHA256 042d104b08afaf399056fe993b262da546ad452db755919e139ee4cdb858d97b),
MID only (`-DOKC_F1HOIST=1`): the FAST1 warp loads all four words first, then runs the same match and count steps;
thread 0 issues the generation load at the same point, and route() stores S.gen / S.ssp right after its first CTA
barrier, so route()'s three later CTA barriers order the stores before every reader. Counts, jobs, ranges, copies,
fences and barriers are unchanged. SASS (`sass_m16r_mid_report.json`): the four LDG at 0x260 to 0x2c0 come before the
first MATCH.ANY, the S.gen store after the first BAR.SYNC; 144 registers, loops 692 / 747 as in the control. Gate: the
five guards pass; chk PASS 12/12, startup OK.

Pooled A/B against ly_stk1, two rounds (19:47 to 19:50 UTC, `table_m16r_both_rounds.txt`): decode 16 sliding +0.42 us
(3 of 24 pairs below), full +0.34 (5/24); decode 32 -0.30 / +0.20; the other cells -0.07 to +0.33, except verify 32
sliding -0.62 (unchanged code). Stamps (`tl16_m16r.log`): route() done 2.73 -> 2.69, first slot 3.71 -> 3.67, warp 0's
phase A weights wait 2.70 -> 2.36, span 66.43 -> 66.48. The copies land earlier, but the first unit still starts when
route() and the first job's x are done, and route() is not shorter: per CTA it still ends 2.64 us after the CTA's own
entry (2.67 in the base) for about one L2 round trip of loads, four CTA barriers and a few hundred instructions. What
holds it there is not identified (the attention-d1 lane's warm-replay test found no cold instruction fetch cost in the
attention kernel; TLB misses after the 1 GiB flush are untested). Null.

### Decision

No candidate for a judge check and no ledger registration: ly_b4rf +0.43 / -0.22 us at decode 32 (5 and 17 of 24 pairs
below the control), ly_m16r +0.42 / +0.34 at decode 16 (3 and 5 of 24), ly_mid2 rejected on its SASS. Where the 21 us
per cell sit after this round: at decode 16 the re-stream (6 us; every way to remove it adds per-unit work to an
mma.sync loop capped at 168 registers), the handoff (2.3 us, a chain of five pieces each under 0.9 us, measured at
decode 1), the tail after the latest B end (2.6 to 3.1 us) and a 2.65 us prologue that load order does not move; at decode 32
the cut shares (+2.15 us each in phase A, moved but not removed by RFIN), the MFIN wait set by a 7 to 9 us end spread,
and the 4.3 us ramp. Open items: the cause of the 2.65 us MID prologue (TLB or instruction fetch, measurable with a
cold-versus-warm microbenchmark of route() alone), and a B4 cost split that charges both shares of a cut unit (the
regression's +2.15 us), which needs range boundaries every CTA can compute without its neighbours' adjustments.
## 2026-11-04 - Decode 16 / 32 sliding attention in the layer (box)

Attention-d16 lane. Question: the layer-d16 timeline puts attention at 7.71 / 9.71 us on the decode 16 / 32 sliding
critical path against 1.89 / 3.78 us byte floors; what are the parts, do the decode 1 findings (K/V off the path, q side
and merge ticket on it) hold at batch 16 / 32, and does one change gated to these two cells shorten it? Base:
muse_decode_layer/cand/ly_stk2.py (SHA256 4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846, the draw
base), never edited. Box only (okc-h200, 19:52 to 20:08 UTC), every run under gpu.lock, router seed
106218864389449120, f931 case seeds, local_start per input, PDL pinned on at every T. No judge check. Logs, JSONs and
tables: `muse_decode_layer/profiles/box/a16/`.

### Plan map (source, `_plan7` and `launch<1>`)

Both cells run `attn_kernel<1, false, 1>` (kn[1], the plain per-request split-KV build, MODE 1), the only q_len 1 launch
of that build in the layer (decode 1 sliding and full take the speculative builds 13 / 9, decode 16 / 32 full the flat
kernel). `_plan7`: decode 16 has 16 requests (reqs < 24) and gets 16 splits, decode 32 has 32 (reqs >= 24) and gets 8;
grid (n_splits, B) = 256 CTAs of 128 threads at both loads, 2 / 4 whole 64-key tiles per CTA, 3-stage cp.async ring
(two tiles in flight), nc = 1, not cooperative. Merge: every CTA's key halves exchanged in shared memory, warps 0 / 1
`red.global.add.v4.f32` into its request's 4 KB workspace region, then thread 0's `atom.acq_rel.gpu` ticket on the
request's counter after the CTA barrier; the request's last CTA (16 / 8 CTAs per request contend) runs `fin_nc1` and
zeroes the region. PDL secondary of qkv (trigger right after its `griddepcontrol.wait`), primary of the o projection.

### Instrument (never a candidate)

`tools/dl_ly_a16_patch.py` writes `ly_a16.py` (SHA256 33d316dd...) and with `--kvw` `ly_a16k.py`: everything
`tools/dl_ly_ast_patch.py --mrg` adds (slots 0 to 21), the sk2 stamps also at M = 16 / 32, and per-tile stamps (22 to 24:
tile 1 to 3's stage visible; 25 to 27: their math done). `tools/ly_box_a16.py` replays 20 fresh inputs per cell through a
plain and an instrumented graph (cold L2) and summarizes per request: the critical request is the one whose finalize
stores issue last, each of its slots the latest of its CTAs. Times are us after qkv_end (the last qkv CTA's stores
issued; the profiler's qkv end is about 0.5 us later). Plain against instrumented graph: 130.69 / 131.34 and 147.18 /
148.14 us (medians), maxdiff at the reds' order-of-addition level.

### The excess as parts (measurement; base, medians over 20 inputs, us after qkv_end, critical request / p50 over CTAs)

| step | decode 16 | decode 32 | part on the path (d16 / d32) |
|---|---:|---:|---:|
| CTAs enter (qkv's trigger) | -0.70 / -0.80 | -0.16 / -0.34 | |
| `griddepcontrol.wait` returns | 0.46 / 0.45 | 0.62 / 0.48 | 0.46 / 0.62 (qkv completion and flush) |
| page indices landed (post-wait barrier) | 0.88 / 0.83 | 1.06 / 0.90 | |
| K/V of tiles 0, 1 issued (16 cp.async per thread) | 1.98 / 1.79 | 2.62 / 2.14 | 1.52 / 2.00 (index wait plus issue) |
| q group visible | 2.06 / 1.86 | 2.69 / 2.21 | 0.08 / 0.07 (q landed during the issue) |
| RoPE, q pass, new k / v pass, new rows patched | 3.34 / 2.42 | 4.30 / 2.83 | 1.28 / 1.61 |
| tile 0 visible | 3.55 / 2.78 | 4.46 / 3.04 | 0.21 / 0.16 (K/V DRAM after the q side) |
| tile loop done (2 / 4 tiles) | 5.26 / 4.81 | 8.61 / 7.23 | 1.71 / 4.15 |
| key halves exchanged, reds issued | 5.92 / 5.30 | 9.01 / 7.71 | 0.66 / 0.40 |
| ticket returns | 7.22 / 6.58 | 10.18 / 8.70 | 1.30 / 1.17 |
| last CTA's finalize stores issued (attention end) | 8.27 / 7.90 | 10.96 / 10.25 | 1.05 / 0.78 |
| o projection's `griddepcontrol.wait` returns | 8.78 | 11.49 | 0.51 / 0.53 |

Placement. Every input put the 256 CTAs on 85 to 87 SMs, three per SM (`sm_used` 87; 82 to 84 SMs with three CTAs, the
rest one or two). The grid launches 0.5 to 1 us before qkv's last CTA ends; qkv's 96 CTAs (large shared memory) still
hold about 45 SMs then, three attention CTAs fit on each free SM, so all 256 land at once on the free SMs and the SMs
qkv frees last get none (the o projection's first 88 CTAs land there at +1.3 us; its other 168 only as attention CTAs
exit, +6.4 to +7.8 us). CTAs sharing an SM with two others issue their K/V slower (0.99 vs 0.83 us at decode 16, 1.34
vs 1.07 at decode 32) and run the loop slower (1.97 vs 1.87, 4.22 vs 3.68 us). The window's 8.4 / 16.8 MB arrive at
about 2.3 / 2.4 TB/s (index landing to last tile visible), against 3.8 TB/s standalone at two CTAs per SM on 128 SMs
("batch 16 / 32 sliding timeline and split arms", runs-attention.md).

The batch 1 findings, checked at batch 16 / 32:

- K/V is still mostly off the path in the base (tile 0 visible 0.2 us after the q side), but the K/V issue is on it:
  16 cp.async per thread take 1.0 to 1.5 us at three CTAs per SM, and the q side cannot start until the thread has
  issued them. At decode 32 the in-loop issue of tiles 2 / 3 sits inside the first two tiles' math (1.1 / 1.3 us
  against 0.6 / 0.5 for tiles 2 / 3, which issue nothing).
- The q side is 1.3 / 1.6 us (batch 1: 0.67).
- The merge tail holds: ticket 1.3 / 1.2 us, finalize 1.05 / 0.78, flush 0.5; 3.0 / 2.5 us in all (batch 1: 2.5).
- New at these loads: the tile loop (1.7 / 4.2 us) on 87 SMs.

`ly_a16k` (thread 0 waits for its own K/V right after the q group; diagnostic, it delays the q side): thread 0's two
tiles land at 3.63 / 3.52 us (p50), 1.8 / 1.4 us after their issue ends.

### Candidate ly_a16kv: the first tiles' K/V before the wait (loss at the stamp level)

`tools/dl_ly_a16kv_patch.py` (base pinned) writes `ly_a16kv.py` (SHA256
6012a503917f9381a1e856e66d5581bc3e720f679e09c31e506805a84d39e400): in `attn_kernel<1, false, 1>` only (KVPRE = L == 1
&& !FLAT && MODE == 1), the page-index group plus the layer-input constants (inv_freq, k_scale, v_scale, cache_loc) are
waited before `griddepcontrol.wait`, tiles 0 / 1 issued (new-token rows skipped as in the base issue), then the wait and
trigger as before, and q / the new k, v rows as `ld.global.cg` loads stored to shared memory (no cp.async group, so the
q pass does not wait behind the K/V groups). Gate (file's own prepare): five PDL8H guards pass, drift equal to the
base's. Stamps (`st_a16kv.log`): the indices land only 0.50 us (p50) after qkv_end, so the K/V issue still ends after
the wait (1.39) and the wait stamp moves to 1.43; the q loads queue behind 16 outstanding copies per thread (q visible
3.10 vs 1.86), and the attention ends at 9.17 / 11.87 (base 8.27 / 10.96), o's wait at 9.66 / 12.40 (base 8.78 /
11.49): +0.9 us at both loads. Stopped after one A/B pair; ruled-out row.

### Candidate ly_a16o2: two CTAs per SM

`tools/dl_ly_a16o2_patch.py` (base pinned) writes `muse_decode_layer/cand/ly_a16o2.py` (SHA256
28855d72bd3d120db19edd621481c92fec4aa1294cec9ac478c199417c9db678). Host-only launch change in `launch<L>`: the kn[1]
launch at L == 1 (decode 16 / 32 sliding only, see the plan map) asks for 81,920 B of dynamic shared memory instead of
sm_n, so three CTAs no longer fit on an SM (3 x 81 KB > 228 KB) and two do. Kernel source, the wait, the trigger, the
fences, the ticket and the PDL attribute are the base's; every other launch keeps sm_n. Diff against ly_stk2 (line
numbers of ly_stk2): a header line; `#define OKC_A16SM 81920` after line 169; after line 3529 `if (L == 1 && OKC_A16SM >
sm_n) cudaFuncSetAttribute(kn[1], cudaFuncAttributeMaxDynamicSharedMemorySize, OKC_A16SM);`; after line 3580 `const int
smk = (L == 1 && kern == kn[1] && OKC_A16SM > sm) ? OKC_A16SM : sm;`; lines 3587 and 3607 launch with `smk` instead of
`sm`.

Gate (file's own prepare, `gate_a16o2.log`): PDL8_AB T = 16 sliding saving 7.90 us (9/9), T = 32 sliding 6.02 (9/9);
PDL8H_GUARD T = 1 sliding, T = 16 sliding, T = 32 full, T = 8 full and T = 8 sliding all pass (ids mismatches 0, drift
equal to the off arm's). Correctness (chk step, six loads): reference check PASS 12/12, `G_RELEASE_STARTUP_OK native
(128, 32, 33, 256, 1, 8, 16) 56 verified_replays`, `CG_GATE ... equal 1023/1024 ok`.

Stamps (`st_a16o2.log`, `ly_a16o2_st.py`): 128 SMs, two CTAs each, at every input. Decode 16 / 32 (base in brackets):
K/V issued 1.70 / 2.10 (1.98 / 2.62), q side done 2.86 / 3.28 (3.34 / 4.30), tile 0 visible 3.23 / 3.52 (3.55 / 4.46),
loop done 4.77 / 7.22 (5.26 / 8.61; decode 32 tiles visible 3.52 / 4.59 / 5.70 / 6.48), ticket 6.64 / 8.56 (7.22 /
10.18), attention end 7.86 / 9.57 (8.27 / 10.96), o's wait 8.42 / 10.11 (8.78 / 11.49), o's end 11.70 / 14.29 (12.10 /
15.71). At decode 16 the ticket grows (reds to ticket 1.44 vs 1.30 us) and takes back part of the front's gain.

Paired A/B against ly_stk2 (`tools/ly_a16_ab.sh` -> `tools/ly_prtl_ab.sh` with CTL=ly_stk2.py, two rounds of
c/w/c/w/c, 60 runs, six loads, PDL pinned on at T = 1, 8, 16, 32; 20:03 to 20:08 UTC; `prtl/a16o2_table_both_rounds.txt`).
Per round, decode 32 sliding w - c -1.15 / -1.23 us (4/4 pairs each), decode 16 sliding -0.03 / -0.38. Pooled (6
control runs, 4 candidate runs, 24 pairs per cell):

| cell | ctl mean | spread | cand mean | w - c | thr | pairs < 0 | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| decode 1 sliding | 63.59 | 0.54 | 63.68 | +0.09 | 1.09 | 13/24 | inside |
| decode 1 full | 70.19 | 1.47 | 69.88 | -0.31 | 2.94 | 14/24 | inside |
| decode 16 sliding | 125.71 | 0.93 | 125.51 | -0.21 | 1.86 | 18/24 | inside |
| decode 16 full | 197.56 | 2.24 | 197.45 | -0.11 | 4.48 | 15/24 | inside |
| decode 32 sliding | 143.67 | 1.12 | 142.48 | **-1.19** | 2.24 | **24/24** | inside |
| decode 32 full | 294.43 | 1.65 | 294.02 | -0.42 | 3.30 | 16/24 | inside |
| verify 1 sliding | 100.21 | 0.53 | 100.68 | +0.48 | 1.06 | 7/24 | inside |
| verify 1 full | 108.73 | 0.70 | 108.40 | -0.33 | 1.41 | 18/24 | inside |
| verify 16 sliding | 242.40 | 2.27 | 241.92 | -0.48 | 4.54 | 15/24 | inside |
| verify 16 full | 326.09 | 2.59 | 325.25 | -0.84 | 5.18 | 19/24 | inside |
| verify 32 sliding | 294.28 | 3.01 | 293.82 | -0.45 | 6.02 | 15/24 | inside |
| verify 32 full | 458.66 | 1.38 | 459.32 | +0.66 | 4.59 | 4/24 | inside |

The ten cells whose launches the change cannot reach (`launch<8>` and the flat / speculative builds take the same host
code with the test false) sit within -0.84 to +0.66 us of the control, inside every threshold; verify 1 sliding (+0.48,
7/24) and verify 32 full (+0.66, 4/24) are the session's drift on unchanged code.

### Decision

ly_a16o2 is the lane's candidate: decode 32 sliding -1.19 us with 24/24 pairs, decode 16 sliding -0.21 (18/24), every
other cell inside its threshold, guards and check clean. It meets the registration bar (decode 32 sliding at least 1 us
with 20+ of 24 pairs, nothing else beyond noise); a ledger registration line is written, and this lane dispatched no
judge job. Value if the box deltas carried to the judge (hypothesis; the box rule forbids reading them as a
prediction): 1.19 x 0.085 + 0.21 x 0.096 = about 0.12% of the score. It changes the launch's shared memory request
only, so it stacks with the other lanes' changes (none touches `launch<1>`'s kn[1] path). ly_a16kv is a loss.

Left open, measured above: the merge tail (ticket 1.3 to 1.4 us, finalize 0.8 to 1.1, flush 0.5) is now the largest
part at decode 16; the K/V issue (16 cp.async per thread, 0.8 to 1.0 us even at two CTAs per SM) still precedes the q
side; with ly_a16o2 the CTAs that land as qkv retires enter at about 0 us and set the slowest request at decode 16.

## 2026-11-04 - Verify 1 sliding attention merge (box)

Attention-v1 lane, round 2. Question: round 1 ("Verify 1 sliding attention in the layer (box)") split the 6.72 us that
`attn_kernel<8, false, 15>` holds after qkv's end into a q side (wait 0.42, q group 0.70, RoPE and q pass 1.09,
consumer setup 0.37), the tile (0.82) and a merge tail (reds 0.85, ticket 0.87, grid barrier 0.70, finalize 0.80).
Which of (a) fewer splits, (b) operands before the wait, (c) relief for the split that holds the new rows takes at
least 0.5 us off verify 1 sliding? Base: `muse_decode_layer/cand/ly_stk1.py` (SHA256
d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878), never edited. Box only (okc-h200, 19:19 to 20:19
UTC), every run under gpu.lock (bx_box_run `--lock`), router seed 106218864389449120, f931 case seeds, local_start per
input, PDL pinned on at every T. No judge check. Logs, JSONs and tables: `muse_decode_layer/profiles/box/av1r2/`.

### Instruments

`tools/dl_ly_atst_patch.py --src` (round 1's stamps) on each file; `tools/ly_av1_stamps.sh` runs
`tools/ly_box_atst.py` (20 fresh inputs, verify 1 sliding, one plain and one stamped capture of the whole layer, times
in us after qkv's last output store). `tools/dl_ly_atstf_patch.py` adds slot 21 before the q pass's
fence.proxy.async and slot 22 before the new-row patch's fence; slot 23 marks an extra q-pass call
(`ly_atst_aqw2.py`). The per-input seeds are the same in every run, so each file is also compared input by input with
the first ly_stk1 run (`stamp_summary.md`). Two runs of ly_stk1 differ by +0.06 us at the attention's end (6 of 20
inputs faster) and +0.14 at o's end: that is the noise of this table.

### Stamps (verify 1 sliding, medians over 20 inputs, us after qkv's end)

| file | q landed | q pass end | tile landed | tile done | critical ticket | barrier passed | attention end | o end | attention end vs base (inputs faster) | o end vs base (inputs faster) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ly_stk1 | 1.06 | 2.18 | 2.62 | 3.46 | 5.71 | 5.84 | 6.66 | 10.38 |  |  |
| ly_stk1 (repeat) | 1.12 | 2.18 | 2.62 | 3.44 | 5.74 | 5.90 | 6.72 | 10.45 | +0.06 (6/20) | +0.14 (8/20) |
| 17 splits | 1.57 | 2.48 | 3.06 | 3.95 | 7.01 | 7.17 | 7.98 | 11.66 | +1.23 (0/20) | +1.23 (1/20) |
| 11 splits | 1.57 | 2.50 | 3.02 | 4.03 | 8.27 | 8.43 | 9.28 | 12.91 | +2.59 (0/20) | +2.56 (0/20) |
| 9 splits | 1.68 | 2.53 | 3.14 | 4.16 | 9.09 | 9.26 | 10.06 | 13.92 | +3.39 (0/20) | +3.57 (0/20) |
| 5 splits | 3.92 | 4.77 | 5.34 | 6.38 | 13.65 | 13.79 | 14.66 | 18.48 | +8.03 (0/20) | +8.11 (0/20) |
| ly_aqw | 1.12 | 2.08 | 2.54 | 3.38 | 5.63 | 5.84 | 6.66 | 10.30 | -0.06 (13/20) | -0.03 (10/20) |
| ly_akw | 1.09 | 2.06 | 2.50 | 3.31 | 5.97 | 6.22 | 6.96 | 10.74 | +0.27 (5/20) | +0.40 (5/20) |
| ly_anpf | 1.12 | 2.21 | 2.62 | 3.46 | 5.73 | 5.89 | 6.70 | 10.43 | +0.02 (7/20) | +0.10 (7/20) |
| ly_apg | 1.10 | 2.19 | 2.62 | 3.46 | 5.79 | 5.95 | 6.72 | 10.48 | +0.03 (7/20) | +0.10 (5/20) |
| ly_v1trg | 1.12 | 1.98 | 2.40 | 3.20 | 5.87 | 6.34 | 7.46 | 11.30 | +0.75 (0/20) | +0.82 (0/20) |
| ly_aqf | 1.10 | 2.18 | 3.55 | 4.42 | 6.29 | 6.40 | 7.28 | 11.02 | +0.59 (0/20) | +0.64 (1/20) |
| ly_aop | 1.95 | 3.07 | 3.52 | 4.34 | 6.62 | 6.88 | 7.62 | 10.99 | +0.88 (0/20) | +0.56 (1/20) |
| **ly_aqd** | 1.09 | 2.02 | **2.18** | 2.98 | 5.15 | 5.31 | **6.14** | **10.02** | **-0.54 (19/20)** | **-0.40 (18/20)** |
| ly_aqk (on aqd) | 1.06 | 1.98 | 2.37 | 3.20 | 5.18 | 5.33 | 6.16 | 10.08 | -0.53 (19/20) | -0.29 (18/20) |
| ly_aqp (on aqd) | 1.09 | 2.02 | 2.18 | 2.98 | 5.14 | 5.31 | 6.08 | 10.03 | -0.58 (20/20) | -0.29 (19/20) |
| ly_aqi (on aqd) | 1.07 | 2.05 | 2.18 | 3.01 | 5.31 | 5.44 | 6.22 | 9.95 | -0.45 (17/20) | -0.45 (18/20) |
| ly_aq2 (on aqd) | 1.28 | 2.11 | 2.30 | 3.14 | 5.44 | 5.60 | 6.45 | 10.32 | -0.16 (19/20) | -0.05 (11/20) |
| ly_aqd_atre (on aqd) | 1.73 | 2.35 | 2.46 | 3.26 | 5.46 | 5.62 | 6.43 | 10.46 | -0.30 (17/20) | +0.13 (9/20) |

"Critical ticket" is the last CTA's ticket (always the split holding the new rows); "attention end" is the last
finalize store; "o end" is the o projection's last output store.

### (a) Fewer splits: ruled out by stamps

`tools/dl_ly_v1sp_patch.py --splits N` adds one `_TUNED7` entry `(1, 8, True, 1): dict(n_splits=N, nc=1, coop=1)`; with
N < 33 every split holds 2 to 7 tiles, so the host launches the multi-tile speculative build `<8, false, 11>` (verify
1 full's instance). Every step down loses about 1 us per extra tile (17: +1.23, 11: +2.59, 9: +3.39, 5: +8.03 us at the
attention's end, 0/20 inputs faster each), and the merge tail does not shrink with the split count: reds issued to
ticket returned stays 0.85 us at 33, 17 and 5 CTAs, and the grid barrier still waits for the new-rows split. The
ticket is a release atomic's round trip after the CTA's own reds drain, not contention between CTAs. Not built
further.

### (b) and the q pass: what the 1.0 us is

- Instruction cache (ly_aqw, ly_akw): the q pass (and, ly_akw, the new k,v pass) moved out of line and called dry
  before `griddepcontrol.wait` on shared-memory garbage, then for real after it. The dry call costs about 1 us before
  the wait (the pre-wait-done stamp moves from -0.74 to +0.21 us) and the real call after the wait still costs
  0.86 us; a third call (`ly_atst_aqw2.py`, slot 23) costs 0.85 us too. Warm code is not faster: not cold code. (The
  first builds of both tools gated on `QDIR`, which the layer compiles out with `-DOKC_QDIRECT=0`, so they built the
  base text; those stamps are kept in `qdir_noop/` and not used.) Attention-d1 found the same at decode 1 (warm replay:
  q pass unchanged).
- Pooler L2 prefetch (ly_anpf: off at T = 8; ly_apg: as per-thread `prefetch.global.L2` lines): unchanged.
- K/V copies in flight (ly_aqf: the one-tile K/V issue moved after the q pass): the q pass is unchanged and the tile
  lands 0.93 us later (K/V latency about 1.3 us from issue): loss.
- The fence after the q pass (slot 21 before it, on ly_aqd): 2.016 against 2.032 us after it: free.
- o's co-residency (o has 256 CTAs, two per SM on 124 SMs including all 33 attention SMs): with o's trigger at the
  merge (ly_v1trg) the q pass is 0.23 us shorter but o's arrival lands on the merge (+0.75 us at the end); with the
  trigger at kernel entry (ly_aqd_atre) o's pre-wait stages land on the q loads (q landed 1.73 against 1.09): both lose.
- SASS of the q pass (ly_aqd, `QP16`): about 280 instructions per thread, no local memory (113 registers, LOCAL 0),
  cos / sin read with LDS.128; the two passes run one after the other. Loading both passes' operands first (ly_aqi,
  bitwise equal) or one 32-dim pass per thread (ly_aq2, bitwise equal) does not shorten it. Reading: a latency-bound
  chain of about 0.4 us per pass with o's warps on the same SMs; not a memory wait.
- The RoPE table (the part (b) could move before the wait) is 0.11 to 0.13 us of the 1.09; not built.
- o's weights into L2 instead of the pooler's (ly_aop: o's whole packed stream, about 16.8 MB per the
  2026-10-28 ruled-out row, prefetched from the attention's entry): thread 0's bulk-prefetch
  issue and the traffic push q's arrival to 1.95 us; o after its wait shrinks only 3.28 to 2.94 us: loss.

### Candidate ly_aqd: the direct e4m3 q pass for the one-tile build

`tools/dl_ly_aqd_patch.py` writes `muse_decode_layer/cand/ly_aqd.py` (SHA256
df16832ff07578e2b40131a9975113d9720b16254d9db35a2531cb5762e7e5c9). The kernel already has a q pass that writes the
e4m3 A tile of the QK^T wgmma directly (`OKC_QDIRECT` + `OKC_QP16`: four threads per row, two passes, one fp32 ->
e4m3 rounding) instead of the f16 staging pass (eight threads per row, four passes) plus the consumers' f16 -> e4m3
conversion; the layer turns it off for every instance with `-DOKC_QDIRECT=0`. The change, against ly_stk1:

```
+#define OKC_QDIRECT1T 1 // the one-tile q_len 8 builds take the QDIRECT q pass even when OKC_QDIRECT is 0
-  constexpr bool QDIR = OKC_QDIRECT && L == 8 && ...
+  constexpr bool QDIR = (OKC_QDIRECT || (OKC_QDIRECT1T && ONE)) && L == 8 && ...
```

(plus the header comment line). Only the one-tile q_len 8 builds (MODE 13 / 15; verify 1 sliding in this layer)
change; every other instance, the wait, the trigger, the fences (the direct path's own `fence.proxy.async` after the
st.shared e4m3 tile) and the merge are the base's. Numerics: the query is rounded once (fp32 -> e4m3) instead of twice
(fp32 -> f16 -> e4m3); the merge order is unchanged.

Stamps: the q pass is 0.16 us shorter and the consumer setup (patch to tile landed) drops from 0.35 to 0.06 us
because the consumers no longer convert the staging; the attention ends 0.54 us earlier (19/20 inputs) and o's last
store 0.40 us earlier (18/20); the new-rows split is still the last ticket (5.15 against a median 4.67).

Gate run (the file's own prepare, no pin, `gate_aqd.log`): PDL8_AB T = 8 savings 7.71 / 7.94 us (9/9), all five
PDL8H_GUARD lines pass with every T on (T = 1 sliding, T = 16 sliding, T = 32 full, T = 8 full, T = 8 sliding; ids
mismatches 0 everywhere; T = 8 sliding on drift 6.37e-3 / 4.72e-3 against off 3.98e-3 / 4.31e-3, bound 2e-2),
`G_RELEASE_STARTUP_OK`, CG_GATE equal to the control's. Reference check (`aqd/chk.log`): PASS 12/12.

Paired A/B against ly_stk1 (`tools/ly_av1_ab.sh` -> `tools/ly_prtl_ab.sh`, c/w/c/w/c, 60 runs, six loads, per round
12 pairs per cell). Rounds 1 and 2 (`aqd/table_both_rounds.txt`), the two-round form the lane's rule asks for:

| cell | ctl mean | spread | cand mean | w - c | pairs < 0 |
|---|---:|---:|---:|---:|---|
| decode 1 sliding | 64.46 | 1.04 | 64.38 | -0.08 | 14/24 |
| decode 1 full | 70.53 | 0.91 | 70.57 | +0.04 | 11/24 |
| decode 16 sliding | 125.45 | 1.15 | 125.76 | +0.31 | 6/24 |
| decode 16 full | 197.81 | 1.17 | 197.85 | +0.04 | 10/24 |
| decode 32 sliding | 143.71 | 1.74 | 143.17 | -0.53 | 20/24 |
| decode 32 full | 294.29 | 2.19 | 294.30 | +0.01 | 10/24 |
| verify 1 sliding | 100.46 | 2.08 | 100.20 | **-0.25** | **14/24** |
| verify 1 full | 109.01 | 1.20 | 108.33 | -0.68 | 22/24 |
| verify 16 sliding | 242.45 | 3.01 | 242.55 | +0.10 | 10/24 |
| verify 16 full | 326.49 | 1.23 | 325.75 | -0.74 | 18/24 |
| verify 32 sliding | 294.76 | 1.92 | 294.90 | +0.14 | 10/24 |
| verify 32 full | 459.03 | 1.58 | 458.97 | -0.06 | 12/24 |

Rounds 3 and 4 (`aqd/table_rounds34_and_all.txt`): verify 1 sliding -0.58 us (22/24), verify 1 full -0.02 (14/24),
decode 1 sliding -0.31 (20/24), verify 32 sliding +1.03 (3/24), every other cell within 0.6 us. All four rounds pooled
(96 pairs per cell): verify 1 sliding **-0.42 us, 67/96**; verify 1 full -0.35 (70/96); decode 1 sliding -0.20
(66/96); decode 32 sliding -0.30 (62/96); verify 32 sliding +0.58 (25/96); the other seven cells -0.27 to +0.15; every
cell inside its threshold. Only verify 1 sliding runs changed device code; the verify 32 sliding skew (+0.58, 25/96)
is on byte-identical device code for that instance (the module is rebuilt, so code addresses move), the same kind of
skew as ly_slpfb's verify 1 sliding. By the judge's cell values the measured verify 1 sliding part is about -0.05% of
score.

Not registered: the two-round A/B gives -0.25 us with 14/24 pairs, below the lane's bar (0.5 us, 20/24); four rounds
give -0.42 us with 67/96. It is a small, consistent gain on the edited cell (stamps 19/20 inputs at the attention's
end) of the size of ly_v1fs, available for a stacked file: the patch applies unchanged to ly_stk3
(`--src muse_decode_layer/cand/ly_stk3.py`).

### (c) The new-rows split, on ly_aqd (stamps only)

The split that holds the block's new rows passes its q pass with the others (2.05 us) and then runs the new k,v pass
(+0.38 us, one pass, the cache-row writes included), the patch (+0.15 to 0.18) and a masked tile (1.0 against 0.8
us); its ticket returns 0.48 us after the median and the grid barrier waits for it in every input.
- ly_akw (the pass warmed before the wait): no change in that split; loss overall (+0.27 / +0.40).
- ly_aqk (the pass in the q pass's straight-line block, one fence after both, stores behind `has_new`): every split
  now pays the pass's arithmetic (median patch 2.30 against 2.11 us) and the critical split is unchanged: null.
- ly_aqp (new rows stored straight into the stage, no patch pass): the critical split's patch is 0.08 us shorter, its
  ticket unchanged (5.14 against 5.15): null.
No mechanism found that shortens the new-rows split's chain by more than 0.1 us without lengthening the others.

### Decision

No registration. ly_aqd (verify 1 sliding -0.25 us in the two-round A/B, -0.42 over four rounds, -0.54 us at the
attention's end in the stamps) goes to the lead as a stack item, not as a lone check. Ruled out: fewer splits (17 / 11
/ 9 / 5), warming the q or new k,v pass, the pooler prefetch's form, K/V after the q pass, moving o's trigger to the
merge or to entry, o's weights into L2 from the attention, and three q-pass / new-row reorderings on ly_aqd. Reading
for the next owner: after ly_aqd the verify 1 sliding attention is a chain of dependent latencies (q landing 0.65 us after the wait,
q pass 0.8, tile 0.8, reds 0.85, release ticket 0.8, barrier wait for the new-rows split 0.15 to 0.5, finalize 0.67)
with o's 58 co-resident CTAs on the same SMs; no single step is bandwidth or instruction-fetch bound. Untested
follow-ups (hypotheses): the same direct q pass for the other q_len 8 builds (verify 1 full's `<8, false, 11>` and
the PLAIN8 builds at verify 16 / 32, a one-token change of the gate to `MODE & 8` or to every L == 8 build; the base
of the `-DOKC_QDIRECT=0` flag is not recorded in the repo); the RoPE table before the wait (0.11 us).

File-name note: attention-d1 committed `muse_decode_layer/cand/ly_atrg.py`, `tools/dl_ly_asp_patch.py` and
`tools/dl_ly_atrg_patch.py` while this lane used the same names for different files; this lane's are renamed ly_v1trg,
`tools/dl_ly_v1sp_patch.py` and `tools/dl_ly_v1trg_patch.py`, and the box copies of attention-d1's three files (which
this lane had overwritten at 19:19 / 19:37 UTC) were restored from origin/main (sha 1abac1d6, 5e4a3d8c, a3c7eb67).
Attention-d1's recorded ly_atrg result (decode 1, trigger at entry) matches its own file's behaviour.
## 2026-11-04 - Decode 16 / 32 sliding attention merge (box)

Attention-d16 lane, round 2. Question: after ly_a16o2, can one change gated to the decode 16 / 32 sliding attention
(`attn_kernel<1, false, 1>`) shorten its merge tail (reds -> acq_rel ticket -> last-CTA finalize -> o's wait), its tile
loop, or o's start? Base: muse_decode_layer/cand/ly_stk3.py (SHA256
a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6; its attention source is ly_a16o2's), never edited.
Box only (okc-h200, 20:24 to 20:33 UTC), every run under gpu.lock (`tools/ly_a16r2_st.sh` -> `tools/ly_box_a16.py`,
20 fresh inputs per cell, cold L2, router seed 106218864389449120). No judge job, no A/B run (no arm passed the stamp
level). Logs: `muse_decode_layer/profiles/box/a16r2/`.

### Stamp sweep (medians over 20 inputs, us after qkv's last stores; decode 16 / decode 32)

| arm (change) | loop end | reds issued | ticket / barrier | attention end | o's wait | o end |
|---|---:|---:|---:|---:|---:|---:|
| ly_stk3 (base) | 4.69 / 7.26 | 5.17 / 7.66 | 6.72 / 8.59 | 7.95 / 9.57 | 8.51 / 10.10 | 11.87 / 14.21 |
| ly_aopf (o stream L2 prefetch after non-last tickets) | 4.70 / 6.91 | 5.10 / 7.22 | 6.78 / 8.51 | 9.18 / 10.56 | 9.76 / 11.10 | 12.74 / 14.86 |
| ly_aopfl (same prefetch after each tile loop) | 4.67 / 6.98 | 5.86 / 7.89 | 8.45 / 10.03 | 9.94 / 11.17 | 10.45 / 11.68 | 13.46 / 15.41 |
| ly_a16sp (8 / 4 splits, one CTA per SM) | 6.18 / 10.16 | 6.50 / 10.48 | 7.30 / 11.36 | 8.93 / 12.14 | 9.47 / 12.66 | 12.85 / 16.78 |
| ly_a16cl (cluster 16 / 8, DSMEM merge, sm_n) | 5.89 / 8.74 | - | 6.78 / 9.86 | 9.47 / 11.89 | 10.06 / 12.32 | 14.14 / 16.50 |
| ly_a16cl2 (same, 81,920 B) | 12.70 / 16.21 | - | 13.54 / 17.02 | 15.65 / 19.14 | 16.06 / 19.55 | 20.16 / 23.74 |

Findings (measurements):
- o's weight stream (12,582,912 B at both loads) fully prefetched into L2 by the attention shortens o after its wait
  only 3.36 -> 2.98 us (decode 16) and 4.11 -> 3.76 (decode 32); the fill traffic slows the last CTA's finalize from
  1.14 to 2.34 us (aopf) or the ticket by 1.7 us (aopfl). o after its wait is mostly not DRAM-bound.
- The ticket tracks same-address fan-in: reds -> ticket 1.46 us at 16 CTAs per request (decode 16), 0.93 at 8
  (decode 32), 0.80 at 8 with 128 CTAs (ly_a16sp). Fewer splits pay it back in the loop (+1.5 / +2.9 us).
- The MODE 2 cluster merge places worse (about three CTAs per SM on 96 to 100 SMs; at two per SM clusters of 16 do
  not all land at launch) and its DSMEM reduce takes 1.67 us at decode 16.

### Built, not yet run: ly_a16nc2

`tools/dl_ly_a16nc_patch.py` writes `muse_decode_layer/cand/ly_a16nc2.py` (SHA256
dbadd5302873cea3385f497b6d24837604bc67e2118e4e1dae1f87bf456c24bb): two workspace copies per request at decode 16 / 32
(split parity picks the copy, 8 / 4 CTAs per address), fin_nc1 under a `NC2` template flag (MODE 1, q_len 1 only)
loads both copies in its one round trip, adds them in copy order and zeroes both; ticket, CTA barrier, acq_rel atomic
and fences unchanged. `--d16only` gates decode 16 alone. Hypothesis from the fan-in row above: ticket 1.46 -> about
0.95 us at decode 16, finalize +0.1, so about -0.3 to -0.4 us there; below the 0.8 us registration bar even if it
holds. Standalone nc 2 (ruled out, e468255b) used the cold generic finalize; this build keeps the warmed one.

### Decision

No candidate this round. Five arms lose at the stamp level (three ruled-out rows); ly_a16nc2 is built and gated but
untested. No ledger registration.

## 2026-11-04 - Decode 1 TINY x at entry (box)

Question: at T = 1, does the TINY MoE build (helper0 source, OKC_NW=8, FAST1=2) wait on the x load at the top of its first unit, and does loading x into registers at kernel entry shorten the ramp? Base ly_stk3.py (a1c48810...b135f6, judge 10,109.9 us); every A/B is paired against it on the RunPod H200 through the GPU lock.

Legality: the MoE runs griddepcontrol.wait at entry, and its stream predecessor is the route kernel that writes xm, so a load after that wait reads finished data. The base already reads x at the same point with prefetch.global.L2.

Candidate: muse_decode_layer/cand/ly_x1e.py, SHA256 c385fe76677295c38ea21337fe8466963e00676441ccf98a824fece579edf17a, written by tools/dl_ly_x1e_patch.py. Under a new OKC_X1E define (set only in MOE_TINY_DEFINES, default 0, so TINY2 and MID build the base device code): a new load_x_tok0 loads token 0's x slice (lanes with g == 0, 16 uint4) at moe_fused entry when p.M == 1; skinny_ring_ws takes those registers and skips load_x_regs for phase A jobs with jC == 1 while M == 1. Numerics unchanged (same x values, same mma order). SASS: TINY moe_fused 145 registers, 0 local, 0 stack in both files (5197 to 5296 instructions).

Stamps (tools/dl_ly_mtl3x_patch.py instrument, 20 inputs, mean CTA, us, sliding / full; muse_decode_layer/profiles/box/x1e/st_summary_x1e.txt):

| stamp | ly_stk3 | ly_x1e |
|---|---|---|
| gen stored | 0.60 / 0.58 | 0.75 / 0.71 |
| route done | 2.11 / 2.09 | 2.21 / 2.21 |
| WS_JOIN passed | 2.30 / 2.29 | 2.41 / 2.41 |
| x consumed (unit 0) | 2.90 / 2.89 | 2.90 / 2.91 |
| first ring slot ready | 3.03 / 3.01 | 3.03 / 3.02 |
| phase A end | 12.70 / 12.50 | 13.07 / 12.46 |

The first ring slot lands 0.13 us after x is consumed in both files, so the mean CTA's first unit waits on the weight copy (FAST1's warp 7 issues at about 1.1 us), not on x. The entry loads push the prologue back by 0.10 to 0.15 us and buy nothing at the slot wait. The latest CTA's slot moves 3.62 to 3.55 (sliding) and 3.58 to 3.52 (full).

Gate: five PDL8H_GUARD lines pass in the file's own prepare; chk 12/12 check PASS, CG_GATE ok, G_RELEASE_STARTUP_OK.

A/B (tools/ly_prtl_ab.sh, two rounds c/w/c/w/c, 24 pairs per cell, pooled; muse_decode_layer/profiles/box/x1e/x1e_table_both_rounds.txt):

| cell | ly_stk3 | ly_x1e | w - c | thr | pairs < 0 |
|---|---|---|---|---|---|
| decode 1 sliding | 63.61 | 63.80 | +0.19 | 2.18 | 8/24 |
| decode 1 full | 70.41 | 70.16 | -0.25 | 2.37 | 14/24 |
| decode 16 sliding | 125.29 | 125.06 | -0.23 | 2.46 | 16/24 |
| decode 16 full | 197.85 | 198.27 | +0.42 | 3.78 | 10/24 |
| decode 32 sliding | 142.72 | 142.98 | +0.26 | 3.10 | 10/24 |
| decode 32 full | 294.50 | 294.52 | +0.01 | 2.95 | 12/24 |
| verify 1 sliding | 100.66 | 100.52 | -0.14 | 4.06 | 16/24 |
| verify 1 full | 108.87 | 108.74 | -0.14 | 2.46 | 15/24 |
| verify 16 sliding | 242.41 | 242.73 | +0.32 | 4.26 | 10/24 |
| verify 16 full | 325.74 | 326.49 | +0.75 | 3.94 | 6/24 |
| verify 32 sliding | 294.54 | 294.23 | -0.31 | 4.96 | 16/24 |
| verify 32 full | 458.96 | 459.24 | +0.27 | 4.59 | 8/24 |

Round 1 alone read decode 1 full -0.89 us (win verdict); round 2 read +0.39, so the pooled cell is noise. Decision: no registration, no judge check. The decode 1 MoE ramp is gated by the first weight unit's copy, so an x-side prologue change cannot move it; the ramp lever is the FAST1 copy issue time (about 1.1 us after entry), which sits behind route.

## 2026-11-04 - Decode 16 / 32 MoE prologue (box)

MoE lane, round 4 (11:57 PT start). Question: what does each microsecond of the MID prologue (decode 16) and the B4
prologue (decode 32) wait on, and does one change take at least 0.8 us at decode 16 or 32 sliding? Base:
muse_decode_layer/cand/ly_stk2.py (SHA256 4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846), never
edited. Box only, every GPU process under gpu.lock, no judge job, no routing slack. Evidence:
`muse_decode_layer/profiles/box/moe4/` (box `.build/moe4/`, A/B runs `.build/prtl/m4_*`).

### Instrument

`tools/ly_moe_pro16.py` (MID) and `tools/ly_moe_prob4.py` (B4) extend the round 3 stamps to 128 slots per CTA; the
new stamps read clock64 (the box's globaltimer advances in 256 ns steps) and are converted per CTA with its own
clock64 / globaltimer rate (1.81 GHz). A clock read placed after `bar.sync` was hoisted above it (a helper stamp after
WS_JOIN read 0.45 us while the barrier could not release before 3.4), so every post-barrier stamp first does a volatile
shared load. Stamps after a load branch on the loaded value. Runner `tools/ly_box_moe_pro.py` (20 fresh inputs, cold L2,
the layer graph; `--warm` adds a replay without the flush). The instrument adds about 0.6 us to route() against the
round 3 stamps, so read the parts as relative sizes. Values: us after each CTA's own entry, CTA mean, median over 20
inputs, decode 16 / 32 sliding (full within 0.1).

### MID prologue (`pro_mid2.log`, `pro_midr2.log`, `pro_mid1p.log`)

| part | ends at | length | what it waits on |
|---|---:|---:|---|
| x L2 prefetch issue, lut (thread 0) | 0.25 | 0.25 | issue only |
| launch generation load (p.ctr[3]) | 0.64 | 0.39 | one global round trip (0.55 in the unflushed replay) |
| route() start, its topk load back | 0.83 | 0.19 | L2 hit (the route kernel's output) |
| barrier 1 | 2.09 | 1.26 | the FAST1 warp: four dependent topk loads (back at 0.47 / 0.70 / 0.89 / 1.08), job scan 0.45, barrier inits 0.09, two TMA issues 0.19, x prefetch and lut 0.13 |
| count loop, barrier 2 | 2.38 | 0.29 | shared atomics |
| warp 0 job scan, barrier 3 | 3.11 | 0.73 | warp 0 alone: three shuffle scans and the job table |
| position loop, barrier 4 (route end) | 3.45 | 0.34 | shared stores |
| ss zeroing, unit ranges, WS_JOIN | 3.69 | 0.24 | |
| skinny_ring_ws setup to the first x load issue | 4.10 | 0.41 | |
| first job's x rows back (64 KB per CTA, L2) | 4.90 | 0.80 | per-SM L2 bandwidth (first slot full at 4.64: weights are not the gate) |
| helper: third ring unit issued | 5.12 | 1.36 after WS_JOIN | |

Tests: the unflushed replay gives the same shape (route end 3.66 at a lower clock, x back 5.52), so cold DRAM data,
code evicted from L2 and TLB refills are not the cost. A second route() call right after the first (`--r2`) takes
1.57 us (topk 0.14, barrier 1 0.07, count 0.21, barrier 2 0.08, scan 0.63, barrier 3 0.08, positions 0.26, barrier 4
0.07), the same as the first call's body, so route() is not waiting on instruction fetch. Probe (helper warp, before
WS_JOIN, a phase B weight page nothing else touches): first touch 414 ns, a new line 64 KB away in the same 2 MiB 400 ns.

### B4 prologue (`pro_b42.log`, `pro_b4r2.log`)

| part | ends at | length | what it waits on |
|---|---:|---:|---|
| lut, barrier inits (thread 0) | 0.35 | 0.35 | |
| launch generation atomic | 0.67 | 0.32 | one global atomic round trip |
| route() (topk loads back 1.14, scan 0.39) | 2.41 | 1.56 | its own body (1.52 when repeated) |
| parity zeroing, jrow scan and barrier | 2.95 | 0.54 | |
| gather: x row loads and stores, proxy fence, barrier | 4.11 | 1.16 | x rows from L2 (0.98) |
| xready releases (thread 0) | 4.56 | 0.45 | |
| x producer wgs_ranges, from barrier 7 (2.53) | 5.07 | 2.54 | an immediate second call takes 0.90 |
| WS_JOIN, first weight issue | 5.2 to 5.4 | | gated by wgs_ranges, not by the gather |

Round 3's "WS_JOIN 3.43, first weight issue 4.32" came from a hoisted stamp; the weights producer cannot leave
WS_JOIN before the x producer finishes wgs_ranges.

### Candidates (one change each, against ly_stk2)

ly_m4rb, MID only (`tools/dl_ly_m4_patch.py rb7`, `muse_decode_layer/cand/ly_m4rb.py`, SHA256
b12e1b47688ed569d2e8202326f475f54166788495e59debbf4b2de378b740a8, `-DOKC_RB7=1` on MOE_MID_DEFINES): at E = 8 M <= 224
the FAST1 warp holds no routing entry, so route()'s first three CTA barriers become named barrier 3 over warps 0..6,
warp 0 zeroes the FAST1 warp's count row, and the FAST1 warp joins only route()'s last CTA barrier (bar 1, all 256
threads), which orders every routing table before any use as before. Gate: five PDL8H guards pass; chk PASS 12/12,
G_RELEASE_STARTUP_OK. Pooled A/B, two rounds (`table_rb_both_rounds.txt`): decode 16 sliding +0.19 us (5/24 pairs
faster), full +0.45 (5/24), decode 32 +0.03 / -0.26, every cell inside. Stamps (`pro_rb.log`): route end 3.45 ->
2.84, but the FAST1 warp, now running beside routing instead of alone, issues its two TMA copies at 2.59 instead of
1.81 (its barrier-init step 0.09 -> 0.79), the x load takes 1.79 instead of 0.80 as the weight copies land at the same
time, and the first slot fills at 5.04 instead of 4.64. Null.

ly_m4rw, B4 only (`tools/dl_ly_m4_patch.py rwarm`, `muse_decode_layer/cand/ly_m4rw.py`, SHA256
f66973f1aba39ae4b144fc6f56528ad2f3661dd9c6c5c750d6474fd95f25a6f8, `-DOKC_RWARM=1` on the b4 defines): the x producer
runs wgs_ranges once on a fixed two-job table (no routing table read; its shared writes are rewritten by the real call
before WS_JOIN) before it waits on barrier 7. MAIN compiles the unchanged text. Gate: five guards pass; chk PASS 12/12,
startup OK. Pooled A/B (`table_rw_both_rounds.txt`): decode 32 sliding +0.22 us (6/24), full -0.57 (23/24, inside the
2.94 threshold), decode 16 -0.14 / -0.32, every cell inside. Stamps (`pro_rw.log`): the real call takes 1.28 us
(2.54 before) but the dry run itself ends at 3.21, so ranges finish at 4.49 instead of 5.07 and the weights producer
leaves WS_JOIN at 4.96 instead of about 5.25. Null.

### Decision

No candidate for a judge check and no ledger registration. What the prologue waits on at both loads: route()'s own
body (1.5 us, warm or cold), MID's barrier 1 on the FAST1 warp's serial topk loads (1.26), the first job's x rows over
the SM's L2 bandwidth (0.8 to 1.0), and at B4 a 2.5 us cold first wgs_ranges. Moving one of these alone shifts the
wait onto the next: in ly_m4rb onto the FAST1 copies and the x load sharing the SM's input path, in ly_m4rw onto the
dry run. Not built: ly_m16r's hoisted FAST1 loads together with rb7 (copies out before routing starts), and the B4
dry run placed so it ends before route() does.

## 2026-11-04 - Decode 16 / 32 row boundaries (box)

Layer-b2 lane, 12:24 to 12:55 PT (Mac clock; box 20:27 to 20:52 UTC). Question: where do the row boundaries at decode
16 / 32 spend their 2.3 to 2.9 us after their producer (b1 after o: glue cluster at T = 16, Gluon
`_graft_row_boundary_pre_cg` at T = 32; b2 after the unpooler: the Gluon kernel at T = 16 and 32), and can one change
take part of it? Base: muse_decode_layer/cand/ly_stk3.py (SHA256
a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6), never edited. Box only, every GPU process under
gpu.lock, no judge job. Evidence: `muse_decode_layer/profiles/box/rbst/`. Checkpoint: docs/lanes/layer-b2.md.

### Instrument (never a candidate)

`tools/dl_ly_rbst_patch.py` writes `muse_decode_layer/cand/ly_rbst.py` (801adafc): thread-0 globaltimer stamps in a copy
of the Gluon row boundary (entry, pre-wait constants landed, after gdc_wait, branch landed, first reduction, stream
landed, NewStream store, second reduction, Out store, SM id; data-dependent stamps take the value as an asm input),
in the PDL8 graft glue `glue_cluster_boundary` (tid 0 per CTA: entry, after cluster.sync, after the wait, branch,
r1, r2, end, SM id), and in the o / unpooler sk2 at M = 16 / 32 (entry, after the wait, tid 0 after the output
stores). All switches are host-side at capture, so the plain graph takes no stamp. `tools/ly_box_rbst.py` captures a
plain and a stamped graph per cell and replays 20 fresh inputs (local_start per input, router seed
106218864389449120, f931 case seeds, cold L2). Log `st1.log` / `st1_summary.txt`.

### Stamps (medians over 20 inputs; us after the producer's last CTA end stamp; steps per program from its own wait)

| item | b1 d16 (glue, 128 CTAs) | b2 d16 (Gluon, 16) | b1 d32 (Gluon, 32) | b2 d32 (Gluon, 32) |
|---|---:|---:|---:|---:|
| first / last program entry | -1.15 / -0.74 | -1.07 / -0.99 | -1.31 / -0.14 | -1.39 / -0.96 |
| pre-wait loads landed (last) | 0.19 (cluster.sync) | -0.32 | **+0.38** | -0.16 |
| wait return (last) | 0.45 | 0.48 | 0.51 | 0.48 |
| branch landed (thread 0) | 0.03 | 0.08 | **0.42** | 0.06 |
| first reduction (all loads landed + reduce) | 0.90 (DSMEM cluster) | 0.54 | 0.26 | 0.58 |
| stream landed | (before the wait) | 0.35 | 0.29 | 0.32 |
| NewStream store issued | - | 0.22 | 0.51 | 0.26 |
| second reduction | 0.74 (DSMEM cluster) | 0.48 | 0.50 | 0.45 |
| Out store issued | 0.06 | 0.13 | 0.13 | 0.13 |
| wait return to end, p50 / max | 1.73 / 1.87 | 1.78 / 1.86 | 2.15 / 2.51 | 1.78 / 1.92 |
| last program end | 2.30 | 2.30 | 3.04 | 2.37 |

The timeline's critical path (kernel end minus producer end: 2.30 / 2.72 / 2.88 / 2.80) adds about 0.4 us of store
drain and grid completion after the last stamp. What each part waits on: (1) launch spread is not on the path: every
boundary's programs are resident 0.7 to 1.4 us before their wait returns, except b1 at T = 32, whose 1024-thread
programs land only as o's CTAs leave (last at -0.14 to -0.27), so its 80 KB of pre-wait constants per SM land 0.38 us
after the wait and its branch load queues behind them (0.42 against 0.06 in b2); (2) every wait returns 0.45 to 0.51
us after the producer's last store; (3) the two row-wide reductions cost 0.45 to 0.58 us each in the Gluon kernel
(each includes the tail of every thread's loads) and 0.74 to 0.90 us each as the glue's 8-CTA DSMEM cluster reduction;
(4) b2's stream load (s1, 32 KB per row) sits between the reductions although b2's programs idle about 0.6 us at the
wait with their constants landed. So the lead's suggested "more programs per row with a cluster / DSMEM reduction"
is contradicted by the glue's own numbers (cluster reductions slower than the CTA's), and halving warps already lost
(ly_rbc).

### Candidates (one change each; `tools/dl_ly_b2_patch.py --mode sp|g32`, base pinned to a1c48810)

ly_b2sp (`muse_decode_layer/cand/ly_b2sp.py`, SHA256 c97fe37063504a951ad304ce273392302e273308d1070b15899f1cd23944f4b5):
in `_graft_row_boundary_pre_cg`, POST (b2 only) loads the Stream row before `_gdc_wait()`. Legality: the stream of b2
is s1, b1's output (call site `_pdl8_graft_boundary(P, S, T, s1, state, True, s2, out, nu=nu)`); b1 completed before
the cooperative MoE's non-programmatic launch, and nothing after b1 writes s1 (the ly_finb1 / b2pre argument); the
branch (the unpooler's output) stays behind the wait; no fence, wait, trigger or barrier moves. Moving the load
changed LLVM's FMA contraction (first build: 15% of NewStream elements and 1 to 5 Out elements differ,
`bitwise_first_sp_contraction.log`), so the POST path spells out the base's contraction, `ns = where(c < 4096,
fma(a, s, beta * b), fma(beta, b, a * s))`; then NewStream and Out are bitwise equal to the base at T = 16 / 32 (8
seeds x 4 scales, `tools/ly_b2_bitwise.py`, `bitwise.log`). SASS (POST, `sass_J6E7.txt` base, `sass_6IZQ.txt`
candidate): the stream LDG.E.128.STRONG.GPU pair moves from 0x6d0 / 0x710 (after the first BAR.SYNC) to 0x190 / 0x1a0
(before ACQBULK at 0x1b0); 51 registers (base 44), EXIT 0xd60 (0xd50), no local memory.

ly_b2g32 (`muse_decode_layer/cand/ly_b2g32.py`, SHA256 c5d276b13adcdd1d2b649efeaf6ece0a725d5104c5fddd3279f34c324567fd86):
b1 at T = 32 dispatched to the PDL8 graft glue b1 (8-CTA clusters of 128 threads per row, stream and constants before
the wait), the form b1 already runs at T = 16, to remove b1's late entry; the glue host's row limit in `_GRAFT_GLUE_CUDA`
b1() 16 -> 32. Output bitwise equal to the Gluon b1 at T = 32 (`bitwise.log`).

Gates (own prepare, `gate_b2sp.txt`, `gate_b2g32.txt`): five PDL8H guards pass, every T on; check PASS 12/12 with
`G_RELEASE_STARTUP_OK` and the control's CG_GATE line (`chk_*.txt`).

Paired A/B against ly_stk3 (`tools/ly_d16_ab.sh` CTL=ly_stk3.py: two rounds of c/w/c/w/c, 60 runs, six loads, PDL
pinned; pooled 6 controls, 4 candidate runs, 24 pairs; `b2sp_table_both_rounds.txt`, `b2g32_table_both_rounds.txt`):

| cell | ly_b2sp w - c (pairs < 0) | ly_b2g32 w - c (pairs < 0) |
|---|---|---|
| decode 1 sliding / full | +0.16 (9/24) / -0.11 (14/24) | +0.04 (11/24) / -0.00 (11/24) |
| decode 16 sliding / full | +0.09 (10/24) / +0.01 (11/24) | -0.30 (17/24) / -0.26 (14/24) |
| decode 32 sliding / full | +0.25 (10/24) / -0.03 (13/24) | **+1.04 (0/24) / +1.04 (2/24)** |
| verify 1 sliding / full | -0.27 (15/24) / -0.11 (13/24) | +0.04 (14/24) / +0.09 (10/24) |
| verify 16 sliding / full | -0.04 (7/24) / +0.29 (8/24) | +0.16 (11/24) / -0.42 (16/24) |
| verify 32 sliding / full | +0.31 (10/24) / +0.75 (5/24) | -0.16 (14/24) / -0.06 (12/24) |

Kernel timelines (`tl_b2sp.tables.txt`, `tl_b2g32.tables.txt`, against ly_stk2's `tl_stk2.tables.txt`, whose
boundary code is ly_stk3's): ly_b2sp's b2 critical path 2.59 / 2.62 / 2.59 / 2.59 us (decode 16 sliding / full,
decode 32 sliding / full) against 2.78 / 2.78 / 2.75 / 2.82, so the hoist takes 0.16 to 0.23 us off the kernel, below
the cell's run-to-run noise; ly_b2g32's b1 at T = 32 runs 3.54 / 2.69 against 3.07 / 2.72 (the 256 cluster CTAs land
behind o's grid at sliding).

### Decision

No candidate; nothing registered, no judge job. ly_b2sp is null at the cell (b2 -0.2 us at the kernel); ly_b2g32
loses decode 32 by 1.0 us. What remains of the boundaries' 2.3 to 2.9 us is structural (measurement): 0.45 to 0.51
us from the producer's last store to the wait return, two serial row-wide reductions of about 0.5 us each, and about
0.4 us of store drain and grid completion. Untested leads (hypotheses): one reduction instead of two by expanding
sum(ns^2) into three partial sums of the first pass (not bitwise: the bf16 rounding of b * r1 sits between them;
would need the CG_GATE and a cancellation argument); b1 at T = 32 with smaller programs that can co-reside with o's
CTAs (512 threads, two programs per row, cross-CTA reduction), whose cost the glue's cluster reductions put at 0.7
to 0.9 us.
