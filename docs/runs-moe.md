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
| 2026-10-28 | 1c4d11ad0cab4943 | 7593542 | block_size_m 8 below 12 tokens else 16; our finalizer at every size; bench sweeps in prepare | 1.327x (1.307-1.347) | per shape: bs1 1.543, bs16 1.252, bs32 1.046, bs8 1.339, bs128 1.446, bs256 1.396. Bench, 3 routing seeds each, block_size_m 8/16/32 (us): 16 wins from 48 tokens up (bs128 319-350 vs 383-430 at 8 and 376-404 at 32; bs256 398-501 vs 535-749 and 455-504); 8 and 16 tie within seed noise at 12-32 tokens. is_ep=False Marlin (our align emits no filtered blocks) saved 2-7 us but prepare grew from 36 s (check a61a16e1) to 244 s of the 300 s cap with it, the bench also having grown, so a second Marlin JIT build costs on the order of 200 s; not used |
| 2026-10-28 | 947c28a7dd3d4de9 | accd987 | same choices as 7593542, benches removed from prepare (self-test kept) | 1.323x (1.303-1.343) | per shape: bs1 1.527, bs16 1.242, bs32 1.046, bs8 1.341, bs128 1.443, bs256 1.395; prepare 43 s. Gate/up paired layout with SwiGLU in the GEMM1 epilogue not built: silu_and_mul is 0.9-3.4% of GPU time per shape in the profiles (geomean ceiling about 1.8%, near the CI width), and a patched Marlin needs its own JIT build, inferred at on the order of 200 s of prepare from the is_ep=False build in 1c4d11ad |
