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
| 2026-10-28 | 4a8852d513874318 | fbbff53 | bench.py: TMA (bulk-copy) weight kernel and large-tile item lists built in prepare(), benchmarked after the tuner; kernel_fn and tuner unchanged | 1.676x (1.651-1.701) | per shape: bs1 1.771, bs16 1.644, bs32 1.450, bs8 1.722, bs128 1.808, bs256 1.688; a control run of the 84f3975334f048c3 policy. The bench itself failed at its first call (it asked the W4 extension for route_align, which lives in the routing extension), after printing registers, shared memory and occupancy per new config (see the TMA and large-tile section below). Prepare 104 s |
| 2026-10-28 | e1619f311d3747e3 | 238317a | the same bench, fixed | 1.671x (1.646-1.696) | per shape: bs1 1.771, bs16 1.649, bs32 1.438, bs8 1.738, bs128 1.779, bs256 1.675; control again (same policy). Bench results in the TMA and large-tile section below: TMA wins only at bs256 GEMM1 (NT2 W8 D4, 1.04x on normal routings), large tiles lose on normal routings at every size. Prepare 108 s, the bench itself about 7 s |
| 2026-10-28 | a49578e5c30348bb | 600e323 | TMA configs are tuner candidates (large tiles benchmark-only) | 1.691x (1.665-1.716) | per shape: bs1 1.764, bs16 1.724, bs32 1.445, bs8 1.734, bs128 1.826, bs256 1.679. The tuner chose TMA only for bs256 GEMM1 (config 49, (1, 2, 16, 4, 8, 4, 7, 1, 0): 233.7 us vs 243.6 for config 37, the previous pick); every other size kept a register-path config. Same as 84f3975334f048c3 (1.683x) and the 1.699x submission within the interval: a 4% gain on one GEMM of one shape is about 0.5% of the geomean. Error vs FP32 4.40e-3. Prepare 101 s, build 52.8 s for 64 kernels |
| 2026-10-28 | 5bc2b26850514b44 | bdbac38 | rewrite: one persistent launch (grid = SM count): routing prologue in every CTA, phase A (GEMM1 + SwiGLU to bf16 inter), phase B (GEMM2, FP32 y and sums of squares), software grid barrier, final norm and sum; 16-row x 4096-K units of 36,864 B by one bulk copy each into a 3-slot ring, 8 compute warps split K with mma.sync (x in registers), a helper warp sums partials and runs the epilogue; every M on the SPLIT path (jobs of <= 8 tokens, split experts re-stream) | 1.429x (1.408-1.451) | per shape: bs1 34.2 us (2.266x, best so far), bs16 125.7 (1.484), bs32 173.9 (1.180), bs8 92.6 (1.614), bs128 368.0 (1.260), bs256 581.2 (1.057). Every shape runs at 1.9-2.0 us per unit, 2.5 TB/s, against 1.2 us at 4 TB/s: the helper warp fenced (__threadfence, MEMBAR.SC.GPU) and credited done_a after every phase A unit, and under a saturated memory system that fence costs about a microsecond, so with 4 partial buffers of slack the compute warps waited on the helper. Error vs FP32 3.9e-3 to 4.2e-3; prepare 38 s (build 37 s); 140 regs, no spills |
| 2026-10-28 | 9f1b1b8a9f0d425e | d0dd9ca | done_a credited once per (CTA, job) instead of per unit; weight ring of 4 slots | failed (deadlock, no score) | the helper warp flushed a job's phase A credits only when it reached the next job or its first phase B unit, but it waited for that B unit's partials first, and the compute warps produce none until done_a is complete, which needs this CTA's own credits; the spin timeout fired (flag 1, about 1.09 s per launch) so the run ended instead of hanging. Self-test error 3.94e-3 (results were right once the timeout released the wait). Fix: flush before waiting at the A-to-B transition (commit after this one) |

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


## TMA weight loads and large token tiles, 2026-10-28

Bench in check e1619f311d3747e3 (`tools/moe_bench_w4x.py` appended to bench.py by `tools/moe_make_bench.py`): per batch size, GEMM and routing seed (normal 11, 12, 13; skewed 21, 22, seed `977 * seed + bs`), the median of 30 CUDA graph replays, each after a 256 MB write that evicts L2. Times include the 5.4 us replay overhead and the SwiGLU reduce (GEMM1 with K splits) or the finalizer (GEMM2), as the tuner times them; GEMM2 reads the current policy's GEMM1 output. "cur" is the policy the tuner picked in that run. Ratios are cur over variant, summed over the three normal seeds (n) and the two skewed ones (s). Every variant matched cur to within 1.0e-4 relative L2 (0 where the K splits match). The best TMA and the best large-tile variant per row are picked by the sum over normal seeds.

| bs | GEMM | cur us (n11/n12/n13/s21/s22) | best TMA | best large tile |
|---|---|---|---|---|
| 1 | g1 | cfg 20: 29.8/32.3/28.5/30/29.9 | tE 30.5/42.4/27.1/30.7/30.7 (0.91x n, 0.98x s) | - |
| 1 | g2 | cfg 26: 23.9/26.8/20.8/23.7/23.6 | tA 26.8/30.3/23.5/27.2/27.4 (0.89x n, 0.87x s) | - |
| 8 | g1 | cfg 33: 53.3/50/61.4/36.7/31 | tG 59.8/56/68.3/43.3/33.4 (0.89x n, 0.88x s) | - |
| 8 | g2 | cfg 30: 35/34.1/46.3/31.7/31.3 | tA 42.1/38.7/47.3/30.6/28 (0.90x n, 1.08x s) | - |
| 16 | g1 | cfg 21: 79.9/76/74/43.5/52.4 | tC 84.2/79.3/73.5/43.9/50.5 (0.97x n, 1.02x s) | - |
| 16 | g2 | cfg 35: 49.9/45.6/43.8/27.7/32.3 | tE 55/50.8/49/30.1/31.6 (0.90x n, 0.97x s) | - |
| 32 | g1 | cfg 33: 110/93.7/91.7/79.6/62 | tG 123/98.1/92.4/56.8/49.8 (0.94x n, 1.33x s) | h8s2+32 120/102/100/78/60.5 (0.92x n, 1.02x s) |
| 32 | g2 | cfg 35: 65.4/56.2/53.7/47.9/38.4 | tE 70.4/60.5/59.2/37.8/32.4 (0.92x n, 1.23x s) | h8+32 70.9/61.5/59.3/55/43.2 (0.91x n, 0.88x s) |
| 128 | g1 | cfg 4: 176/157/213/182/142 | tG 180/160/206/133/109 (1.00x n, 1.34x s) | h8s2+32 181/166/198/137/108 (1.00x n, 1.32x s) |
| 128 | g2 | cfg 36: 96.1/86.1/108/73.2/59.5 | tE 98.3/88.3/111/71.7/59.5 (0.98x n, 1.01x s) | h16+64 103/92.4/115/73.5/58.5 (0.93x n, 1.01x s) |
| 256 | g1 | cfg 37: 258/229/254/285/224 | tG 248/219/246/272/207 (1.04x n, 1.06x s) | h16+64 255/245/269/253/219 (0.96x n, 1.08x s) |
| 256 | g2 | cfg 13: 129/114/129/143/106 | tE 131/113/129/143/108 (1.00x n, 0.99x s) | h16+64 131/122/139/136/100 (0.95x n, 1.06x s) |

Labels: TMA tA (1,1,8,4,4,4), tB (1,1,8,4,4,8), tC (1,1,8,4,8,4), tD (1,1,8,2,8,4), tE (1,1,8,2,4,8), tF (1,2,16,4,4,8), tG (1,2,16,4,8,4) for GEMM1 and tA (2,1,8,4,4,8), tB (2,1,8,4,8,4), tC (2,1,8,2,4,8), tD (2,2,16,4,4,8), tE (2,2,16,4,8,4) for GEMM2, all mode 7 layout 1, d = shared-memory stages. Large tiles: u32/u64 every item 32/64 routes; hA+B items of A routes for experts below the threshold and B routes (a second launch, run first) for experts at or above it (32 for +32, 64 for +64); h8s2+32 uses 2 K splits.

Registers, shared memory and occupancy (cudaFuncGetAttributes and the occupancy API in check 4a8852d513874318; warps per SM count the TMA kernel's producer warp; no spills in ptxas for any of them):

| config | regs | smem B | blocks/SM | warps/SM |
|---|---|---|---|---|
| register (1,1,8,4,4,4) | 76 | 16576 | 6 | 24 |
| register (1,2,16,4,4,4) | 92 | 33152 | 5 | 20 |
| register (1,2,16,4,8,4) layout 1 | 98 | 33152 | 2 | 16 |
| TMA (1,1,8,4,4,4) | 40 | 35040 | 6 | 30 (24 consumer) |
| TMA (1,1,8,4,4,8) | 40 | 53536 | 4 | 20 (16) |
| TMA (1,1,8,4,8,4) | 40 | 53472 | 4 | 36 (32) |
| TMA (1,2,16,4,8,4) (tG) | 56 | 70016 | 3 | 27 (24) |
| TMA (2,2,16,4,8,4) (GEMM2 tE) | 56 | 70016 | 3 | 27 (24) |
| large (1,4,32,4,4,4) | 124 | 66304 | 3 | 12 |
| large (1,8,64,4,8,4) | 190 | 132608 | 1 | 8 |
| large (1,8,64,4,4,2) | 146 | 132608 | 1 | 4 |
| large (1,4,32,2,8,4) | 128 | 131840 | 1 | 8 |

Lessons:
- TMA weight loads (one producer thread, two bulk copies per stage of a block's W units, full and empty mbarriers) cut registers from 76-98 to 40-56 and raised resident warps (36 per SM at W8 D4 against 16-24), yet lost 3-11% at bs 1-32 and tied at bs128 on normal routings. It won only at bs256 GEMM1 (tG, 1.04x, all five seeds) and against its own register twin (1,2,16,4,8,4) layout 1 by 6-10% at bs 8-256. So more resident warps, by themselves, do not close the gap to the streaming kernel; the loads-only result (50.8 us with 8 warps per block vs 59.6 with 4) did not carry over to the real GEMM. One likely reason, not measured: each stage is now a serial wait on a full barrier, then shared-memory loads, then the arrive, and the register path overlaps D stages of global loads per warp without that round trip.
- Large token tiles cut weight traffic where routing is concentrated: uniform 64-route items were 1.5x faster at bs128 skewed (120/91 us vs 182/142) but 2x slower on normal routings, where most experts get far fewer than 64 routes. The device-side split (large items for experts at or above a threshold, a second launch run first) kept part of the skewed gain (h8s2+32 1.32x at bs128 skewed) but lost or tied on normal routings at every size, including n11 at bs256 whose hottest expert has all 256 tokens (h16+64 255 vs 258 us). The large-tile kernel runs at 4-12 warps per SM (124-190 registers, 66-132 KB of activations) and the two launches serialize, which together cost about what the fewer weight re-reads save. The scored shapes use the normal routing (TASK.md: skewed appears only as a correctness test), so this is not wired in.
- The tuner now includes the TMA configs; it picked one (bs256 GEMM1) in check a49578e5c30348bb, which scored 1.691x (1.665-1.716), the same as before within the interval.
- Decision 2026-10-28: TMA weight loads and hot-expert large tiles stay out of submission.py.
  TMA won only bs256 GEMM1 (about 4%, about 0.5% of the score); large tiles lost on normal
  routing. The final file scored 1.691x (a49578e5c30348bb), overlapping the 1.699x entry. Code
  kept at muse_moe_w4a16/experiments/tma_and_large_tiles.py.
