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
