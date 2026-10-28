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
| 2026-10-28 | 42924d2c6e314921 | 1f9417c | W4 GEMM1 for bs <= 32 with a fixed config table (Marlin GEMM2 kept), self-test gated on FP32 error; bench of 16 GEMM1 and 12 GEMM2 configs | 1.333x (1.313-1.353) | per shape: bs1 1.398, bs16 1.380, bs32 0.991, bs8 1.437, bs128 1.444, bs256 1.417 (bs128/256 still all Marlin). The fixed table was a poor pick: bs32 used nt=4 rows=32 (0.90x of Marlin in the bench). Bench (cold L2, graph replay minus 4.8 us; the log tail lost bs1 and bs16): GEMM1 + SwiGLU best configs were 1 n8 tile with K split 2 (cfg (1,1,8,2,4,4)) or 2 tiles with split 4, 1.15-1.27x faster than Marlin GEMM1 + silu at bs 8/32/128, 1.17-1.22x at bs256; without a K split 0.55-0.75x. W4 GEMM2 + FP32 finalizer best (split 4) 1.19-1.36x faster than Marlin GEMM2 + finalizer. Full forward with both W4 GEMMs: bs8 105 -> 86 and 96 -> 80 us, bs32 237 -> 172 and 186 -> 148, bs128 334 -> 265 and 292 -> 234, bs256 454 -> 368 and 378 -> 319, error vs FP32 2.9e-3 (Marlin path 5.3e-3). More K splits won almost everywhere: the limit is CTA count and per-block prologue, not ALU |
| 2026-10-28 | cfad171137bf45f8 | 186b046 | W4 GEMM1 and GEMM2 at every size, config per batch size tuned in prepare() (34 configs; Marlin also a candidate), timed on synthetic routing only | 1.672x (1.647-1.697) | per shape: bs1 1.695, bs16 1.674, bs32 1.422, bs8 1.731, bs128 1.819, bs256 1.721. Tuned policy {1: g1 (1,1,8,2,4,8) g2 (2,1,1,4,4,8); 8, 16, 32: g1 nt1 rows8 split 2-4, g2 (2,1,8,4,4,4); 128: g1 (1,1,8,2,8,4) g2 (2,2,16,4,4,4); 256: g1 (1,2,16,4,4,4) g2 (2,2,16,4,4,4)}. Marlin never won. Profile GEMM times (judge routing): bs1 g1 20.5 g2 12.3 us; bs8 57.3/33.1; bs16 77.5/39.4; bs32 85.1/44.6; bs128 179.9/85.1; bs256 195.5/95.9. Self-test error vs FP32 2.9e-3 to 3.0e-3 at bs 1-257 (Marlin path 5.2e-3 to 5.3e-3). Build 47 s for 36 kernels, tuning 1.7 s. Registers 76 (nt1 d4) to 163 (nt2 d8), no spills (ptxas) |
| 2026-10-28 | 05924173c44046e9 | ad87094 | programmatic dependent launch on the W4 kernels (tuned on/off per size); timing-only kernel modes: loads without dequant or mma (mode 1), mma on raw words without dequant (mode 2) | 1.677x (1.652-1.702) | per shape: bs1 1.730, bs16 1.689, bs32 1.421, bs8 1.739, bs128 1.819, bs256 1.695; same as cfad171137bf45f8 within the interval. PDL: whole forward 57.0 -> 54.4 us (bs1), 86.9 -> 84.0 (bs8), 124.5 -> 122.7 (bs16), 162.8 -> 160.2 (bs32), 250.6 -> 248.7 (bs128), 358.4 -> 356.9 (bs256), kept on everywhere. Diagnostics (tuner timings incl. ~5 us replay overhead, 2 seeds): GEMM1 nt1 rows8 split 4 is 52.4 us at bs8 with dequant, 50.4 without (mode 2), 60.4 with loads only (mode 1); bs32 103.8 / 101.5 / 121.6; nt2 rows16 split 4 at bs128 166.8 / 149.6 / 181.9. Dequant costs 2-4% (10% on the nt2 bs128 config); removing all compute makes the kernel no faster, so the GEMMs are bound by the weight access pattern (about 3000 warps each streaming its own 1 KB per stage), not by ALU |
| 2026-10-28 | 0f478ae5371a4b75 | 9971904 | second weight layout (8 neighbouring units interleaved per k stage, so a block's warps read 4-8 KB contiguous per stage), tuned against the first; configs pruned to 34 | 1.683x (1.658-1.708) | per shape: bs1 1.731, bs16 1.685, bs32 1.414, bs8 1.748, bs128 1.817, bs256 1.734; same as 05924173c44046e9 within the interval. Layout made no difference: matching configs within 1 us (bs8 (1,1,8,4,4,4) 52.2 layout 0 vs 52.5 layout 1; bs32 103.2 vs 104.2); the tuner took layout 1 only for bs256 GEMM1. Loads-only diagnostics: 60.3 (layout 0, 4 warps) and 61.1 (layout 1, 4 warps) vs 50.2 us (layout 1, 8 warps) at bs8, where the real GEMM takes 52.2. DRAM locality across warps is not what holds the GEMMs at about 3 TB/s; the remaining suspects are per-block start (dependent loads of items, sorted ids, x staging) and many short blocks, which a persistent kernel would remove |
| 2026-10-28 | e5ddb50d82234690 | 8da4bf0 | persistent GEMM mode (blocks walk (item, chunk, split) units, activations double-buffered, the next unit's first weight stages loaded during this unit's last), GEMM2 streams weights before its PDL wait, SwiGLU kernel triggers its dependent first; tuner checks every candidate against Marlin | 1.651x (1.626-1.676) | per shape: bs1 1.750, bs16 1.627, bs32 1.429, bs8 1.622, bs128 1.825, bs256 1.681; overlaps the previous intervals. Persistent configs landed within about 1 us of their one-unit-per-block twins (bs8 GEMM1 52.3 vs 52.4 us; bs32 108.6 vs 103.0), so block start-up and short blocks are not the limit either. Tuner chose persistent GEMM1 at bs 1-16 and persistent GEMM2 at bs 1 and 8. No config failed the Marlin comparison; self-test vs FP32 3.0e-3 |
| 2026-10-28 | c4b1b5366bc54857 | 8811b31 | diagnostics: loads-only kernels with no activation staging, with streaming-style addresses (every warp reads consecutive 1 KB chunks at each stage), with and without the L2 256-byte prefetch hint; GEMM configs without the hint as candidates | 1.690x (1.664-1.715) | per shape: bs1 1.722, bs16 1.749, bs32 1.461, bs8 1.710, bs128 1.823, bs256 1.698. Loads-only at bs8 (tuner timings incl. ~5 us replay overhead and the ~3 us SwiGLU-reduce kernel that split-K adds): real addresses 60.4 us, no activation staging 59.7, streaming addresses 59.6, streaming without hint 59.3, streaming with 8 warps per block 50.8. So the access pattern, the activation staging and the hint are not the limit; resident warps are. Net of overhead and the reduce kernel, GEMM1 at bs8 moves 141 MB in about 43.6 us, 3.25 TB/s, against about 3.85 TB/s for the plain streaming kernel at that size (85%). The no-hint GEMM variants won narrowly at most sizes (policy {1: (34, 26), 8: (33, 30), 16: (33, 11), 32: (33, 35), 128: (4, 36), 256: (37, 36)}, PDL on). Per-forward small kernels at bs1 (profile cfad171137bf45f8): finalizer 4.7 us on one block, SwiGLU reduce 2.2, route_align 2.4, against 32.8 us of GEMMs |
| 2026-10-28 | c1025175b73548fd | b911811 | finalizer over FP32 partials as 1, 4 or 8 blocks per token (a cluster sharing the per-slot sums of squares through distributed shared memory), tuned per size with PDL on/off; SwiGLU reduce over 8 blocks per route | 1.682x (1.656-1.707) | per shape: bs1 1.809, bs16 1.645, bs32 1.461, bs8 1.770, bs128 1.794, bs256 1.638; same as c4b1b5366bc54857 within the interval (per-shape swings of +-0.06 between runs follow the judge's per-run routing seeds). Whole forward, tuner: cluster finalizer saves at most 0.5 us (bs1 52.3 -> 52.0), PDL 2-4 us. Stop rule: four controlled iterations (layout, persistent blocks, access-pattern diagnostics, finalizer) moved nothing outside the interval. Superseded by 84f3975334f048c3 (BF16 boundaries) |
| 2026-10-28 | 84f3975334f048c3 | 965e415 | BF16 at SGLang's boundaries (TASK.md: activations stay BF16 wherever SGLang keeps BF16): gate and up rounded to bf16 before SwiGLU (Marlin's cache1), SwiGLU output bf16 (GEMM2 input), GEMM2 output rounded to bf16 after its FP32 split-K sum and before the norm (Marlin's cache3); rounding in registers | 1.683x (1.658-1.708) | per shape (vs c1025175b73548fd): bs1 1.780 (1.809), bs16 1.753 (1.645), bs32 1.430 (1.461), bs8 1.728 (1.770), bs128 1.795 (1.794), bs256 1.645 (1.638); same within the interval. Error vs FP32 in the self-test rose from 3.0e-3 to 4.2e-3 to 4.4e-3, still below the Marlin path's 5.2e-3 to 5.5e-3. This is the final file |

## W4A16 GEMM (own kernel), 2026-10-28

Per shape, GEMM1 and GEMM2 over the unique weight bytes of the experts that received tokens
(18.9 MB for W13, 9.4 MB for W2 per expert, NVFP4 values plus E4M3 scales; global scales
negligible), routing seeds 11 and 12 (`977 * seed + bs`, the same in check 16fd966f071340eb's
split and in the tuner). "Stream" is the cold-L2 streaming-read kernel's bandwidth at that
size, interpolated between the sizes measured in check 16fd966f071340eb. Marlin columns are
that check's kernel-only times
(replay overhead subtracted). Tuner columns are check c1025175b73548fd's medians, Marlin
GEMM1 + silu_and_mul vs the best W4 GEMM1 + its split-K SwiGLU reduce, and Marlin GEMM2 +
finalizer vs W4 GEMM2 + finalizer, same routings, each including about 5 us of graph replay
overhead.

| bs | experts | GEMM1 MB | floor @4.8 / @stream us | Marlin g1 us | GEMM2 MB | floor @4.8 / @stream us | Marlin g2 us | tuner g1+act: Marlin -> W4 | tuner g2+fin: Marlin -> W4 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.5 | 66 | 13.8 / 17.9 | 30.3 | 33 | 6.9 / 9.9 | 19.9 | 37.2 -> 30.8 (1.21x) | 29.3 -> 24.6 (1.19x) |
| 8 | 7.5 | 142 | 29.5 / 36.8 | 57.6 | 71 | 14.7 / 19.1 | 32.7 | 65.0 -> 51.4 (1.26x) | 42.5 -> 33.8 (1.26x) |
| 16 | 12.5 | 236 | 49.1 / 60.2 | 95.2 | 118 | 24.6 / 31.1 | 51.4 | 102.3 -> 77.6 (1.32x) | 60.9 -> 47.6 (1.28x) |
| 32 | 17.5 | 330 | 68.8 / 83.3 | 131.7 | 165 | 34.4 / 42.6 | 69.6 | 139.1 -> 101.4 (1.37x) | 79.2 -> 60.3 (1.31x) |
| 128 | 24 | 453 | 94.4 / 113.5 | 198.6 | 226 | 47.2 / 57.8 | 103.3 | 207.2 -> 168.1 (1.23x) | 113.7 -> 91.5 (1.24x) |
| 256 | 22.5 | 425 | 88.5 / 106.8 | 263.8 | 212 | 44.2 / 54.4 | 136.7 | 275.3 -> 244.6 (1.13x) | 150.9 -> 121.8 (1.24x) |

Marlin ran GEMM1 at 1.6-2.5x and GEMM2 at 1.6-2.5x the streaming floor (the 2.5x at bs256).

Lessons:
- Kernel: swap-AB mma.sync m16n8k16, weights as A (16 rows x 16 k, so one E4M3 scale per row
  per k block), tokens as B from shared memory via ldmatrix, FP32 accumulation, global scale
  times 2^7 in the epilogue. Weights are repacked losslessly in prepare() so each lane's
  16-byte loads are its own fragments (no shared memory for weights), and each 32-bit word's
  bits are permuted so that 4 bf16x2 registers come out of 1 AND, 2 shift+AND, and a 3-op
  merge; the scale byte becomes bf16 scale * 2^119 with one PRMT and one IMAD (valid for
  positive normal E4M3, checked in prepare()). About 17 instructions per 8 weights, 9 on the
  integer pipe (SASS). `tools/moe_emulate_w4.py` checks the bit permutation, fragment order,
  scale order, row mapping and the layout-1 addressing against plain NVFP4 dequantization.
- Numerics: the W4 path rounds to bf16 exactly where SGLang materializes bf16 (gate and up
  before SwiGLU, SwiGLU's output, GEMM2's output before the norm), and reduces in FP32 inside
  each GEMM. Its error against FP32 is 4.2e-3 to 4.4e-3 relative L2 (Marlin path 5.2e-3 to
  5.5e-3). Before check 84f3975334f048c3 it kept those tensors in FP32 (3.0e-3), which
  TASK.md's activation rule does not allow.
- K splits (2 to 4) won at every size: without them there are too few warps in flight. The
  split-K partials cost a reduce kernel (SwiGLU for GEMM1, the finalizer for GEMM2), still a
  net win.
- Dequant costs 2-4% (mode 2), and removing all compute does not speed the kernel up (mode 1).
  Streaming-style addresses, a block-contiguous weight layout, persistent blocks with
  cross-unit prefetch, and dropping the L2 256-byte hint each moved GEMM time by about 1 us or
  less. At bs8 GEMM1 runs at about 3.25 TB/s net of overhead, 85% of the streaming kernel.
- Programmatic dependent launch saves 2-4 us per forward. It also makes the profiler's
  per-kernel times include time spent waiting, see docs/hotspots.md.
- bs256's hottest expert gets up to all 256 tokens; items of 8-16 tokens re-read its weights
  (through L2), which is why W4 GEMM1 gains least there (1.13x). A tile with more tokens per
  dequantized weight would help that shape.
- Decision 2026-10-28: TMA weight loads and hot-expert large tiles stay out of submission.py.
  TMA won only bs256 GEMM1 (about 4%, about 0.5% of the score); large tiles lost on normal
  routing. The final file scored 1.691x (a49578e5c30348bb), overlapping the 1.699x entry. Code
  kept at muse_moe_w4a16/experiments/tma_and_large_tiles.py.
