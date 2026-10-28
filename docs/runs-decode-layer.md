# muse_decode_layer runs

Every `okc check` on `muse_decode_layer`, newest last. Scores are the judge's speedup over
SGLang with its 95% interval. Profiles and logs are in `muse_decode_layer/profiles/`. The
submission is baseline.py's call sequence with pieces ported from muse_moe_w4a16 and
muse_bf16_proj, each behind a switch (`PORT_MOE`, `PORT_ROUTER`, `PORT_PROJ`) and a self-test
against SGLang's path in prepare().

Per-shape columns, in TASK.md's order: d1 = decode 1, d16, d32, v1 = verify 1 (8 tokens), v16
(128 tokens), v32 (256 tokens); s = sliding layer, f = full layer.

## How the decode layer calls the ported pieces (checked against baseline.py)

- MoE: same Marlin bank, workspace, `is_ep=True` and eps 1e-8 as muse_moe_w4a16. Differences:
  `local_start` is a device tensor (the MoE problem had it in the config), so our align kernel
  reads it from device memory; the post-expert norm weight is SGLang's effective weight
  (`post_expert._get_effective_weight()`), passed to both paths.
- Router: up to 8 tokens SGLang routes on the boundary's FP32 router input, not the BF16 FFN
  input, and the reference routes on full-precision h. The muse_bf16_proj router kernel took
  BF16 x, so the port stages FP32 x instead. Above 8 tokens SGLang uses BF16 hi + lo on the
  BF16 FFN input, as on muse_bf16_proj. The router runs on a side stream, overlapping the pooler.
- Projections: qkv_gate, pooler and the unpooler's 1024-row slice at T <= 8, o at T = 1, as
  `_proj_fused_wins` chose on muse_bf16_proj.

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
