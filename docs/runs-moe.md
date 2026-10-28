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
| 2026-10-28 | 16fd966f071340eb | 7137cde (+bench) | measurement only: `tools/moe_make_bench.py` appends a cold-L2 streaming-read kernel and a per-shape Marlin GEMM1 / silu / GEMM2 / finalizer split to prepare(); kernel_fn unchanged | 1.327x (1.307-1.347) | Streaming reads (best of 5 grids, graph replay minus a 5.0 us empty replay): 2.68 TB/s at 9 MB, 3.06 at 18, 3.28 at 28, 3.65 at 57, 3.80 at 113, 3.95 at 283, 4.06 at 640 MB, so at these sizes the measured ceiling is 3.3-3.9 TB/s, not 4.8. Marlin over unique weight bytes (active experts x 18.9 MB for GEMM1, 9.4 MB for GEMM2), 3 routing seeds: bs1 (2-4 experts) g1 21-34 us at 1.8-2.2 TB/s, g2 18-22 us at 1.1-1.7; bs8 (7-10) g1 55-74 us at 2.4-2.6, g2 31-41 at 2.1-2.3; bs16 (11-13) g1 85-99 at 2.5, g2 46-53 at 2.3; bs32 (14-20) g1 107-149 at 2.5, g2 57-78 at 2.3-2.4; bs128 (23-25, hottest expert 30-124 routes) g1 185-244 at 1.9-2.4, g2 96-128 at 1.8-2.3; bs256 (21-25, hottest 48-256 routes) g1 239-289 at 1.4-1.9, g2 124-149 at 1.3-1.8. silu 0.9-5.1 us, finalizer 2.9-7.8 us. GEMM1 holds about two thirds of the recoverable time on every shape, so it is replaced first |
| 2026-10-28 | 5dea75d7c668473b | a31223f | W4A16 GEMM1 (own kernel, SwiGLU fused) for bs <= 32, bench of 16 configs in prepare | 1.320x (1.301-1.340) | W4 path disabled itself: the self-test compared it to the Marlin path with a 5e-3 bound and got 6.0e-3. Against an FP32 reference the W4 path was the more accurate one (3.7e-3 to 3.8e-3 vs Marlin's 5.2e-3 to 5.5e-3 at bs 1, 8, 32): Marlin rounds gate and up to bf16 before SiLU, ours keeps FP32. So this score is the Marlin path again. Extension build 39.5 s for 16 kernels. Gate changed to FP32 error |
