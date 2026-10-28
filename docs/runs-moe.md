# muse_moe_w4a16 runs

Every `okc check` on `muse_moe_w4a16`, newest last. Scores are the judge's speedup over
SGLang with its 95% interval. Profiles and logs are in `muse_moe_w4a16/profiles/`.

## Source facts (SGLang a55ab7d5, read from GitHub and hashed against the image)

- `fused_post_expert_norm_weighted_sum` falls back to PyTorch ops below
  `_MIN_TOKENS_FOR_FUSED = 32` tokens: a `where` to zero filtered slots, then float, scale,
  pow, mean, rsqrt, multiply, multiply by the routing weights, sum and a cast, about ten
  launches. At 32 tokens and above it is one Triton kernel, one program per token.
- With an expert map, `moe_align_block_size` aligns over 257 buckets (256 global experts
  plus the filtered -1 bucket). At 64 routes or fewer (bs <= 8) that is the single-CTA
  Triton `moe_align_small_numel`; above, SGLang's two-kernel CUDA align. The filtered routes
  get blocks with expert id -1, which every Marlin CTA scans past.

## Runs

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-28 | a61a16e1f1e84206 | 79edc53 | own align (all bs) + one-block-per-token finalizer (bs 1, 8, 16); bench in prepare | 1.193x (1.175-1.211) | per shape: bs1 1.542, bs16 1.271, bs32 1.046, bs8 1.335, bs128 1.027, bs256 1.025. Bench (graph replay, cold L2, own routing seed): finalizer alone is most of the gain (bs1 87.1 -> 54.0 us; align alone 87.1 -> 83.3); finalizer variants 0/2/4/8 within 0.5 us of each other; block_size_m 16 beats SGLang's pick at bs128 (361 vs 482 us) and bs256 (471 vs 632 us), bs16 16 vs 8: 92 vs 97 us |
