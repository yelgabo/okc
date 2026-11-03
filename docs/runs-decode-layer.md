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
