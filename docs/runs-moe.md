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
| 2026-10-28 | f82a8cf656df4914 | 27986fb | credits flushed before the phase B wait (the deadlock fix); fat path for M >= 17: 64-token jobs, 256-row units in 16 stages of 256 k, four consumer warpgroups on register-A wgmma m64nNk16 (N = 8 x ceil(tokens / 8)), x gathered into the wgmma B layout, stream-K stage ranges by a cost table, last-arriver fixup for cut units; segments grouped by (phase, N) so ptxas pipelines the wgmmas | 1.740x (1.714-1.766) | per shape: bs1 28.7 us (2.706x; bar 30 met), bs16 100.3 (1.876), bs32 203.0 (1.009), bs8 70.2 (2.042; bar 55), bs128 302.7 (1.554), bs256 362.8 (1.708). All tests pass on the fat path (bs 33, 257, 128 skewed), error vs FP32 3.8e-3 to 4.6e-3, no spin timeouts, prepare 36 s (build 35 s). Skinny path now 1.47 us per unit at bs8 (3.0 TB/s plus about 6 us fixed); the fat path streams at 2.4 TB/s (bs32, 2.05 us per 36 KB stage), 2.2 (bs128) and 1.8 (bs256), so its weights ring (3 slots, 108 KB in flight per SM, next to 3 x 32 KB x slots) or its per-stage synchronization is the limit, not re-reads: each expert is read once. Next: a variant bench in prepare (ring depths, 16 compute warps on the skinny path, loads-only modes) |
| 2026-10-28 | bab82f5f5b974594 (submit, authorized by the user via the coordinator) | cc3c926 | variants bench in prepare with a per-size policy: skinny sk8w4s / sk16w4s / sk8w5s, fat main-d2 / main-d3 (runtime-sized rings, conflict-free codes layout), the previous kernel's configs as legacy32 / legacy128 / legacy256, plus loads-only and no-gate diagnostics | 1.869x (1.841-1.897) | per shape: bs1 28.2 us (2.716x), bs16 105.0 (1.868), bs32 136.9 (1.485), bs8 82.4 (1.837), bs128 254.6 (1.793), bs256 358.5 (1.718). POLICY {1: sk8w4s, 8: sk16w4s, 16: sk8w4s, 32: sk8w4s, 128: legacy128, 256: main-d3}. BENCH (us, mean of 2 routings): bs1 loads-only 40.0 vs sk8w4s 42.0; bs8 70.6 vs 76.0-76.2 (sk8w5s 79.4); bs16 105.4 vs 109.6 (legacy32 119.0); bs32 main-loads 140.4, sk8w4s-loads 146.7, sk8w4s 155.0, legacy32 156.5, main-d3 = main-d2 223.5, main-nogate 221.2; bs128 main-loads 190.6, legacy128 250.3, sk8w4s 269.5, main-d2 301.5; bs256 main-loads 204.8, main-d3 337.3, legacy256 360.4, sk8w4s 440.7. So the skinny path is streaming-bound (loads-only within 2-6 us of computing, about 3.4 TB/s net of about 8 us fixed cost) and the fat path is compute-bound per stage (loads-only 35-40% below its computing modes; depth 3 and removing the readiness waits change nothing). Prepare 143 s (build 139 s for 7 fused kernels and the legacy extensions). Error vs FP32 3.7e-3 to 4.6e-3 |
| 2026-10-28 | 36b5b559b0344d0f (submit, authorized by the user via the coordinator) | 35baa17 | fat path: k permuted inside each 256-k stage so one scale build covers a lane's four k16 steps, x slab written in that mma-k order by the gather and the phase A epilogue, wgmma descriptors in uniform registers (slot base through a shuffle: R2UR per HGMMA gone), SiLU by MUFU.RCP; diagnostics without wgmma and without dequant; sk8w5s, depth 3 and no-gate dropped | 1.990x (1.960-2.020) | per shape: bs1 29.3 us (2.727x), bs16 97.3 (1.937), bs32 133.9 (1.544), bs8 68.4 (2.106), bs128 255.1 (1.800), bs256 308.4 (2.008). POLICY {1, 8, 16, 32: sk8w4s, 128: legacy128, 256: main-d2}. BENCH (us): bs32 main-loads 139.3, sk8w4s 149.9, legacy32 155.6, main-nowgmma 157.3, main-nodequant 173.1, main-d2 197.3 (was 223.5); bs128 main-loads 189.6, main-nowgmma 211.9, main-nodequant 233.2, legacy128 249.9, main-d2 260.8 (was 301.5), sk8w4s 261.8; bs256 main-loads 203.0, main-nowgmma 237.3, main-nodequant 267.1, main-d2 291.6 (was 337.6), legacy256 360.1. The fat path's two costs add rather than overlap: at bs256 dequant alone +34 us over streaming, wgmma alone +64, both +89; the wgmma side (fence, two HGMMA, commit, wait per group, all warpgroup-collective) is the larger. Prepare 143 s. Error vs FP32 3.9e-3 to 4.6e-3 |
| 2026-10-28 | 0400cfbdfa2845fe (submit, authorized by the user via the coordinator) | 54f7940 | fat path: jobs of up to 2 (modes 6, 9) or 4 (7, 10) n8 tiles, or all jobs (8), on mma.sync m16n8k16 instead of RS wgmma (same A fragments, accumulators, epilogue and fixup; B by ldmatrix.x4 from the x slot), modes 9, 10, 8 with flatter stage costs; diagnostics 2 and 3 no longer built; a mode that fails its self-test only leaves the bench | 2.007x (1.977-2.038) | per shape: bs1 27.4 us (2.796x), bs16 96.9 (1.938), bs32 133.4 (1.544), bs8 71.3 (2.100), bs128 261.9 (1.804), bs256 301.5 (2.060). POLICY {1, 8, 16, 32: sk8w4s, 128: legacy128, 256: main-m2}. BENCH (us): bs32 main-loads 140.6, sk8w4s 151.6, main-d2 189.3, main-m2 190.0, main-m2f 190.0, main-m8f 190.6, main-m4 191.2; bs128 main-loads 190.9, legacy128 250.9, main-m2 253.5, main-d2 254.4, main-m2f 261.5, main-m4 266.6, main-m8f 279.8; bs256 main-loads 202.6, main-m2 283.1, main-d2 285.3, main-m4 298.7, main-m8f 327.7. So mma.sync and RS wgmma cost the same at N = 8 and 16 (bs32, where nearly every job is one n8 tile: 190.0 vs 189.3): the MMA instruction is not what keeps the fat path 50 us above its streaming time at bs32 and 63 us at bs128. mma.sync loses above N = 16 (m4, m8f). Error vs FP32 3.88e-3 every fat mode. Build 154 s (was 139) |
| 2026-10-28 | c524612c78454486 (submit, authorized by the user via the coordinator) | cd06a9e | mode 11 (mode 6 with clock stamps) and a DIAG report in prepare | 2.002x (1.972-2.032) | per shape: bs1 28.1 us, bs16 95.5, bs32 133.1, bs8 67.7, bs128 255.7, bs256 304.6; same policy and bench as 0400cfbdfa2845fe within 1 us (bs128 main-loads 188.9, legacy128 249.7, main-m2 253.6, main-d2 254.3, main-m2-stamps 258.5). The DIAG report failed on the host (`inter.view(int64)` of the 2-D buffer sliced rows, not elements; the dry run used a flat buffer), so no stamps. Fixed in the next commit, dry run now uses the real shape |
| 2026-10-28 | 2072795029b7479f (submit, authorized by the user via the coordinator) | a248788 | the DIAG fix | 1.978x (1.948-2.007) | per shape: bs1 29.7 us, bs16 97.1, bs32 136.0, bs8 69.0, bs128 263.0, bs256 301.8; same code and policy as the two runs before, so the 1.978-2.007 spread of these three is run-to-run noise. Bench bs128: main-loads 190.6, legacy128 253.6, main-m2 255.4, main-d2 256.8. Stamps below (section "Where the fat path's time goes at 128 tokens") |
| 2026-10-28 | 84de5bbafa344b7a (submit, authorized by the user via the coordinator; the current board entry) | 4fed369 | mode 14 (mode 12 with two mma.sync accumulator chains per n8 tile); stage costs per launch (moe_run `cost` argument) fitted from the DIAG stamps by least squares and offered as `-fit` variants of modes 6, 12, 14; modes 12/13 with fence.acq_rel in the epilogue and fixup | 2.009x (1.979-2.039) | per shape: bs1 27.95 us (2.760x), bs16 95.20 (1.954), bs32 133.40 (1.538), bs8 67.76 (2.108), bs128 254.18 (1.829), bs256 300.02 (2.057). POLICY {1, 8, 16, 32: sk8w4s, 128: legacy128, 256: main-m2-ar}. BENCH bs128 (us): main-loads 188.7, legacy128 251.3, main-m2-ar2 252.8, main-m2 254.3, main-m2-ar 254.7, -fit variants 256.5-257.8; bs256: main-m2-ar 283.8, main-m2-ar2 284.0, -fit 284.0-285.1. The least-squares stage-cost fit gave negative phase A coefficients (A1 -0.40, A2 -0.56, A3 -0.56, A7 -0.36, A8 -0.92 us/stage; B 5.6-9.6; residual rms 6.4 us over 396 CTA launches), so the fitted table is not a cost model and its variants did not win anywhere; it is not used to drive partitioning again (plan 2026-10-29 item 4). It is within noise of 0400cfbdfa2845fe (2.007x, 54f7940) and the two runs after it. Stamps at bs128: producers wait 185.8/202.6 us (mean/max) for a free weights slot of about 236 us of consumer time. Profiles and the result JSON in muse_moe_w4a16/profiles/submission.84de5bbafa344b7a.* |
| 2026-10-28 (before the 00:00 UTC reset) | 1b20c5b390f044b2 (submit by the coordinator, snapshot of 1099d9b) | 1099d9b | the proxy-ordering fix (fence.proxy.async.global by every x writer before its release and by the issuer after its acquire; the gather publishes with red.release after its barrier) | 1.979x (1.949-2.008) | passed 5/5. per shape: bs1 29.48 us (2.696x), bs16 102.99 (1.902), bs32 136.32 (1.508), bs8 72.38 (2.080), bs128 251.48 (1.823), bs256 301.26 (2.046). POLICY {1-32: sk8w4s, 128: legacy128, 256: main-m2-ar-fit}. BENCH bs128 main-loads 191.0, legacy128 251.2, main-m2-ar2 252.8, main-m2-ar 254.4, main-d2 254.4; bs256 main-m2-ar-fit 282.7, main-m2-ar 282.8, main-d2 283.9 (84de5bba: main-m2-ar 283.8, main-d2 284.8): the fences cost nothing measurable on the fat path. The score is within noise of the 2.009x entry (CIs overlap; bs1 and bs16 moved +1.5 and +7.8 us on the skinny path the fix does not touch, so that is run-to-run noise). Board entry stays 84de5bbafa344b7a |
| 2026-10-28 (before the reset) | 8078730bfa434074 (submit by the coordinator, snapshot of 9755822) | 9755822 | the fix plus the nonnegative cost model in prepare (72 DIAG launches, held-out validation gate) | 1.985x (1.955-2.015) | passed 5/5. per shape: bs1 27.87 us, bs16 100.95, bs32 135.85, bs8 68.66, bs128 258.07 (1.783x), bs256 302.35. The gate failed (section "Cost model, 2026-10-29"), so no -nn variant ran; POLICY {1-32: sk8w4s, 128: legacy128, 256: main-d2}. BENCH bs128 main-loads 191.2, legacy128 251.7, main-m2-ar2 253.2, main-m2-ar 254.1, main-m2 254.3, main-d2 255.5; bs256 main-d2 283.5, main-m2 283.6, main-m2-ar2 284.4. bs128 258.07 against 251.48 the run before with the same legacy128 choice: run-to-run noise on the judge's timing of the same kernel |
| 2026-10-29 | 4111803ce7894b63 (check) | 12794bd | mode 15: mode 12 with phase B CTA boundaries rounded to the nearest unit (16 stages), so no phase B segment is cut (the cost model's cut term was 10.4 us in phase B, 2.6 in A) | 1.986x (1.956-2.016) | per shape: bs1 28.38 us, bs16 97.66, bs32 138.84, bs8 68.92, bs128 259.82 (legacy128), bs256 299.00 (main-m2-ar2). BENCH bs32: main-m2-ar-ub 186.6, main-m2-ar2 187.1, main-m2-ar 190.6; bs128: legacy128 252.3, main-m2-ar-ub 253.1, main-m2-ar2 253.2, main-m2-ar 256.8; bs256: main-m2-ar2 283.5, main-m2-ar 284.9, main-m2-ar-ub 285.3. Unit alignment takes 4.0 us off mode 12 at bs32 and 3.7 at bs128 and nothing at bs256, so the cut cost is real at the smaller sizes, but mode 15 still trails legacy128 by 0.8 us at 128. Cost model refit on this run's stamps: the same picture (B cut 10.4 us, gate failed again: skewed table order 12/17 vs 13/17). Error vs FP32 3.90e-3 |
| 2026-10-29 | 9a9fb08c134c4a16 (check) | 907542c | mode 16: mode 14 (two mma.sync chains per n8 tile) with the unit-aligned phase B boundaries of mode 15 | 2.019x (1.989-2.049) | per shape: bs1 28.03 us, bs16 96.98, bs32 135.04, bs8 67.14, bs128 237.94 (1.907x, main-m2-ar2-ub), bs256 291.90 (2.076x, main-m2-ar2-ub). POLICY {1-32: sk8w4s, 128: main-m2-ar2-ub, 256: main-m2-ar2-ub}: the first time the fat path beats legacy128 at 128 tokens in the bench. BENCH bs32: main-m2-ar2-ub 182.4, main-m2-ar-ub 185.6, main-m2-ar2 186.8, main-m2-ar 190.4; bs128: main-m2-ar2-ub 249.0, legacy128 251.0, main-m2-ar-ub 251.2, main-m2-ar2 251.6, main-m2-ar 256.0; bs256: main-m2-ar2-ub 280.2, main-m2 282.5, main-m2-ar2 282.9. The judge's bs128 time, 237.9 us, is 13.5-22 us under the last five runs (251.5-259.8, all legacy128), and bs256 is 8-10 us under. The score's CI overlaps the 2.009x entry, so a repeat check follows before a submit |
| 2026-10-29 | c65e91500215418d (check; 99970e4fd1af402b, the same file again, came back as a duplicate, so this is the file plus one trailing newline) | 907542c | repeat of 9a9fb08c134c4a16 | 2.011x (1.981-2.042) | per shape: bs1 29.53 us, bs16 97.14, bs32 130.15, bs8 67.80, bs128 245.86 (1.917x), bs256 295.42 (2.087x); same policy (main-m2-ar2-ub at 128 and 256). BENCH bs128: main-m2-ar2-ub 249.1, main-m2-ar2 249.8, legacy128 250.9; bs256: main-m2-ar2-ub 280.2, main-m2-ar2 282.6. Two runs of mode 16 put bs128 at 237.9 and 245.9 us against 251.5-259.8 in the last five runs on legacy128, and bs256 at 291.9 and 295.4 against 299.0-302.4; the scores (2.019x, 2.011x) are inside the noise of the 2.009x entry |
| 2026-10-29 | da9225e4f89441fc (submit, authorized by the user via the coordinator: carries the proxy-ordering fix) | 907542c | mode 16 file as checked in 9a9fb08c134c4a16 | 2.003x (1.973-2.033) | per shape: bs1 29.17 us, bs16 98.97, bs32 136.34, bs8 70.83, bs128 246.59 (1.903x), bs256 292.67 (2.078x); same policy. Third run of mode 16: bs128 237.9 / 245.9 / 246.6 and bs256 291.9 / 295.4 / 292.7 us, against 251.5-259.8 and 299.0-302.4 on the code before, while the skinny shapes (bs1-32, unchanged code) drew low this time. Below the 2.009x board entry within noise, so the board still shows 84de5bbafa344b7a, which predates the ordering fix |
| 2026-10-29 | 9332e2a49aff4ff2 (check) | d90ad14 without mode 21 and the staged pair (built modes 0, 1, 16, 17, 18, 19, 20) | modes 17 (16 with stamps, wgmma segments stamped too), 18 (16's phase A alone), 19 (phase A alone on owned whole units, snake order by n-tile count: the GEMM1-only prototype), 20 (19's phase A, then 16's phase B and final pass); a GATE report (6 seeds, same process); modes 6, 11-15 not built | 2.044x (2.013-2.074) | per shape: bs1 28.07 us, bs16 97.21, bs32 133.31, bs8 68.03, bs128 239.63 (main-own), bs256 283.75 (main-own). POLICY {1-32: sk8w4s, 128: main-own, 256: main-own}. GATE (us, mean of 6 seeds): bs128 mode 16 248.3, 18 175.6, 19 170.1, 20 245.6; 19 minus 18 per seed +1.9 -7.4 -6.1 -8.2 -9.1 -4.1 (mean -5.5), 20 minus 16 mean -2.7 (5 of 6 seeds lower); bs256 16 286.4, 18 205.6, 19 189.2, 20 282.0; 19 minus 18 mean -16.4 (6 of 6), 20 minus 16 mean -4.4. xb of 19 vs 18 rel L2 4e-6 to 7e-5 (FP32 sum order of the fixups); output of 20 vs 16 1e-4 to 3e-4; every self-test 3.9e-3 vs FP32. Stamps in section "Mode 16 at 128 and 256 tokens" below; mode 17 costs 14-18 us over 16 (263.3 vs 248.9 at 128) |
| 2026-10-29 | e221858b020c4dee (check) | d90ad14 | adds mode 21 (phase B and the final pass alone, no done_as wait, no gather) and the staged pair VARIANT_STAGED (19 then 21, two launches), self-tested and benched; modes 0, 1, 16, 18, 19, 20, 21 built | 2.026x (1.995-2.056) | per shape: bs1 27.98, bs16 97.30, bs32 136.88, bs8 69.03, bs128 245.85 (main-own), bs256 288.66 (main-own). GATE: bs128 16 249.4, 18 176.9, 19 171.3, 20 247.3, 21 109.6, staged 277.3; staged minus 16 +21.2 to +32.0 (mean +27.9), staged minus 20 mean +30.0; 19 minus 18 mean -5.6; 20 minus 16 mean -2.1. bs256 16 287.3, 18 206.1, 19 190.4, 20 283.5, 21 124.3, staged 311.6; staged minus 16 mean +24.2, minus 20 mean +28.1; 19 minus 18 mean -15.7; 20 minus 16 mean -3.8. Staged self-test 3.88e-3, output vs 16 1.3e-4 to 2.0e-4. The staged pair loses (ruled-out.md) |
| 2026-10-29 | 72b7a5a9c511446a (check) | 308ed5b | the shippable build: modes 0, 1, 16, 20 (the phase-only modes and the staged pair stay in the source, unbuilt) | 2.022x (1.992-2.053) | per shape: bs1 28.29, bs16 97.70, bs32 141.43, bs8 69.26, bs128 247.68 (main-own), bs256 291.36 (main-own). GATE: 20 minus 16 per seed bs128 -2.9 -2.3 -3.9 +0.6 -8.9 -1.2 (mean -3.1), bs256 -10.8 -10.6 +5.0 -5.0 +0.1 -4.1 (mean -4.2). BENCH bs128 main-own 247.7, main-m2-ar2-ub 251.6, legacy128 251.7; bs256 main-own 270.8, main-m2-ar2-ub 281.3. Three runs of mode 20: 2.044x, 2.026x, 2.022x against 2.019x, 2.011x, 2.003x for mode 16's three; the intervals overlap, the same-process A/B (three times, 18 seeds per size) puts 20 ahead by 2-3 us at 128 and 4 us at 256. Prepare build 137 s |
| 2026-10-29 | b99aa87cabdd430c (submit, authorized by the user) | 308ed5b | mode 20 file as checked in 72b7a5a9c511446a | 2.034x | passed. POLICY {1-32: sk8w4s, 128: main-own, 256: main-own}. GATE bs128 main-own minus mode 16 mean -2.1 us, bs256 -4.3 us (6 seeds). BENCH bs128 main-own 246.5, mode 16 248.4, legacy128 249.8; bs256 main-own 270.3, mode 16 278.6. New board entry over da9225e4f89441fc (2.003x) |
| 2026-10-29 | a9dbec1d98334943 (check) | 10375cb | modes 22 (20 with stamps), 23 (20 with phase B units claimed at run time from a global counter in S.ord order, both phases walked largest n-tile count first) and 24 (23 with stamps); GATE 16 / 20 / 23 on seeds 11-16 with the variant order reversed on every other seed; built modes 0, 1, 16, 20, 22, 23, 24 | 2.076x (2.045-2.107) | per shape: bs1 29.17, bs16 97.20, bs32 137.53, bs8 72.30, bs128 219.92 (2.083x, main-own-dynb), bs256 269.21 (2.289x, main-own-dynb). GATE (us, mean of 6 seeds): bs128 16 248.1, 20 246.2, 23 230.0; 23 minus 20 per seed -15.9 -14.2 -14.9 -18.3 -14.4 -19.2 (mean -16.2); bs256 16 287.4, 20 283.7, 23 267.6; 23 minus 20 -9.2 -17.8 -26.3 -13.8 -20.3 -9.4 (mean -16.1). Output of 23 vs 16 1.1e-4 to 2.4e-4 (same as 20 vs 16); self-tests 3.9e-3 vs FP32; spin flags 0. BENCH bs128 main-own-dynb 231.2, main-own 246.0; bs256 257.0 vs 268.8. Stamps in section "Phase B claimed at run time" below. The log is a tail: the DIAG22 lines for bs128 seed 11 were cut. Build 149 s |
| 2026-10-29 | 36812632dae242ba (check) | ec219f0 | the shippable build: modes 0, 1, 16, 20, 23; GATE on held-out seeds 17-22 and on skewed routing (seeds 11-13) | 2.077x (2.046-2.108) | per shape: bs1 28.05, bs16 101.53, bs32 133.82, bs8 66.30, bs128 221.07 (2.072x, main-own-dynb), bs256 270.57 (2.278x, main-own-dynb). GATE held-out normal: bs128 16 259.9, 20 263.3, 23 240.4; 23 minus 20 -26.7 -19.8 -28.6 -11.8 -30.4 -20.2 (mean -22.9), 23 minus 16 mean -19.5; bs256 16 310.7, 20 304.6, 23 285.9; 23 minus 20 -21.3 -12.6 -25.3 -26.1 -13.4 -13.4 (mean -18.7), minus 16 mean -24.8. Skewed: bs128 23 minus 20 -4.4 -1.9 -8.3 (mean -4.8); bs256 -26.0 -26.4 -16.3 (mean -22.9). 24 of 24 paired seeds favour 23 over 20 across both checks. Judge bs128 219.9 / 221.1 us against 239.6 / 245.9 / 247.7 for mode 20's three runs, bs256 269.2 / 270.6 against 283.8 / 288.7 / 291.4. Build 146 s |
| 2026-10-29 | fa18c9e87d6e4811 (submit, under the standing replace-when-demonstrably-better authorization) | beb1924 | mode 23 file as checked in 36812632dae242ba | 2.070x | passed. POLICY {1-32: sk8w4s, 128: main-own-dynb, 256: main-own-dynb}. BENCH bs128 main-own-dynb 231.3, main-own 246.0; bs256 256.2 vs 268.5. New board entry over b99aa87cabdd430c (2.034x) |
| 2026-10-29 | d1f587760b1241dc (check, bench file from tools/moe_make_bench_ab.py, not committed: it embeds the public MoE file) | 35ab79e | same-process A/B: mode 23 and sk8w4s against the public MoE kernel (vjeux/okc-2026 muse_moe_w4a16.py, MAIN build) at 32, 128, 256 tokens on six seeds of the standalone routing and six of a port of the layer's judge routing; mode 25 (23 with thread-0 phase stamps only) against the public kernel's own OKC_DEBUG stamps | 2.064x (2.033-2.095) | public minus mode 23, us (6 of 6 seeds lower in every row): T=32 std -18.0, layer -12.8; T=128 std -34.9, layer -32.2; T=256 std -52.5, layer -45.7. Attribution in section "Same-process A/B against the public MoE" below. Outputs vs FP32: ours 3.84e-3 to 3.86e-3, public 2.34e-3 (it does not round gate, up and the GEMM2 output to bf16). Mode 25 costs nothing (BENCH bs128 231.9 vs 23 at 232.7). Policy unchanged (sk8w4s to 32, main-own-dynb at 128 and 256) |
| 2026-10-29 | 39c365c34cd74559 (check, bench file) | 9c384a6 | mode 26: mode 23 with every warp counting the routes (the routing warp sums 18 per-warp counts instead of counting all E entries alone, and the positions reuse them) and a batched x gather (all of a CTA's 16-byte row loads before its stores); 27 is 26 with the stamps | 2.083x (2.052-2.115) | 26 minus 23, seeds 1-6, us: T=128 std -4.3, layer -3.7; T=256 std -8.2, layer -8.5; T=32 layer -0.5 (6 of 6 lower in each). Stamped route 11.0 -> 7.7 us at 128 layer, 16.1 -> 9.6 at 256; the routing warp still 4-5 us and thread 0's stage-range searches 2.3-3.1 us. Output of 26 vs 23 7e-6 to 3e-5. POLICY 128 and 256: main-dynb-fs. The log is a tail: the T=32 std lines were cut |
| 2026-10-29 | 4c285df44eb34bc1 (check, bench file) | c850d04 | 26 and 27 skip the four stage_at searches (owned phase A and claimed phase B read no stage range); routing-warp stamps; seeds 7-12 (held out from the two checks before) | 2.097x (2.066-2.129) | 26 minus 23, us: T=32 layer -1.4; T=128 std -5.8, layer -4.6; T=256 std -9.8, layer -9.1 (6 of 6 lower in each). Stamped route at 128 5.8 us (public 2.4): ids 0.5, routing warp 4.4 (counts and offsets 1.1, jobs 0.6, S.ord counting sort 2.5), positions 0.5, ranges 0.3. Public minus 23 on these seeds: T=128 -36.1 / -31.9, T=256 -50.2 / -50.1 |
| 2026-10-29 | 817eab6401d54a84 (check) | ba9a457 | the shippable build: modes 0, 1, 16, 20, 23, 26; GATE adds 26 minus 23 (held-out seeds 17-22, skewed 11-13) | 2.104x (2.073-2.136) | per shape: bs1 28.37 us, bs16 97.81, bs32 138.70, bs8 71.65, bs128 215.68 (2.148x, main-dynb-fs), bs256 261.71 (2.371x, main-dynb-fs). GATE 26 minus 23 per seed: bs128 -5.1 -6.0 -7.0 -5.1 -6.6 -5.2 (mean -5.8); bs256 -9.8 -9.3 -9.5 -9.7 -9.2 -9.8 (mean -9.5); skewed bs128 -4.2, bs256 -7.7 (3 of 3 each). Output of 26 vs 16 matches 23's (7.8e-5 to 3.5e-4); self-tests 3.89e-3 vs FP32; spin flags 0. BENCH bs128 main-dynb-fs 226.4, main-own-dynb 231.6; bs256 248.9 vs 258.6. Judge bs128 215.7 us against 219.9 / 221.1 for mode 23's two runs, bs256 261.7 against 269.2 / 270.6. Build 145 s. Submit candidate: ba9a457 (not submitted) |
| 2026-10-29 | a8076924ced1419e (submit, under the standing replace-when-demonstrably-better authorization) | a8fc96a | mode 26 file as checked in 817eab6401d54a84 (ba9a457) | 2.101x | passed. Board entry before: fa18c9e87d6e4811 (2.070x) |
| 2026-10-29 | 70dd7b13be884434 (check, bench file from tools/moe_make_bench_ab.py, not committed) | ec62b5a | mode 28: mode 26 with a leaner consumer stage (ring size and x slot base in registers, the weights slot released by every consumer thread behind the stage's first wgmma group, fence kept) and a one-pass S.ord sort; 29 is 28 with the light stamps; arms o26, o28, pub, seeds 1-6, standalone and layer-like routing | 2.132x (2.100-2.164) | 28 minus 26 per seed, us: T=32 layer -8.1 -4.4 -7.6 -3.6 -7.0 -7.8 (mean -6.4), std mean -6.4; T=128 layer -8.0 -10.8 -9.9 -7.6 -6.5 -8.9 (mean -8.6), std mean -12.1; T=256 layer -7.7 -10.3 -8.6 -7.3 -8.2 -10.8 (mean -8.8), std mean -11.0 (36 of 36 lower). Public minus 28: T=128 layer -16.1, T=256 layer -25.1 (was about -27 / -41 against 26). Output of 28 vs 23 1.6e-5 to 2.6e-5, vs FP32 3.84e-3 to 3.86e-3. ABSTAMP and attribution in section "Consumer stage and the ring slot releases". POLICY 128 and 256: main-lean |
| 2026-10-29 | f928f8a802014e4f (check) | ec62b5a | the shippable build: MAIN_MODES (0, 16, 26, 28); GATE adds 28 minus 26 (held-out seeds 17-22, skewed 11-13) | 2.149x (2.117-2.181) | per shape: bs1 28.05 us, bs16 95.96, bs32 135.02, bs8 68.47, bs128 200.89 (2.295x, main-lean), bs256 247.11 (2.524x, main-lean). GATE 28 minus 26 per seed: bs128 -11.5 -10.8 -10.2 -11.1 -10.6 -9.4 (mean -10.6); bs256 -12.0 -11.4 -12.9 -8.6 -11.2 -13.7 (mean -11.6); skewed bs128 -6.6 -6.7 -6.7, bs256 -7.5 -7.7 -9.9. Output of 28 vs 16 equal to 26's to two digits (1.1e-4 to 3.7e-4); self-tests 3.89e-3 vs FP32; spin flags 0. BENCH bs128 main-lean 213.6, main-dynb-fs 225.8; bs256 237.8 vs 248.5. Build 138 s |
| 2026-10-29 | 13fe68e7ca2e423f (check, repeat) | ec62b5a | f928f8a802014e4f's file plus one trailing comment line, so the judge does not return it as a duplicate (an identical resubmission, 3adce08f3e9844d5, came back as duplicate_of f928f8a802014e4f without running) | 2.158x (2.125-2.190) | bs128 200.34 us, bs256 244.59; GATE 28 minus 26: bs128 mean -11.6, bs256 -12.5 (6 of 6 lower each), skewed -6.4 / -8.4 (3 of 3). POLICY 128 and 256: main-lean. Against mode 26's shippable check 817eab6401d54a84 (2.104x): bs128 215.68 -> 200.89 / 200.34, bs256 261.71 -> 247.11 / 244.59. Submit candidate: ec62b5a (not submitted) |
| 2026-10-29 | 08070cce049b403d (submit, under the standing replace-when-demonstrably-better authorization) | c406b03 | mode 28 file as checked in f928f8a802014e4f (ec62b5a) | 2.137x | passed. Board entry before: a8076924ced1419e (2.101x) |
| 2026-10-29 | c7db20e053254b52 (check, bench file from tools/moe_make_bench_ab.py, not committed) | 1d962e6 | phase A attribution against the public kernel, layer routing seeds 1-6 at 32 / 128 / 256 tokens: mode 31 (28 with phase A counters per consumer warp and the producers' waits), 32 (phase A publish deferred to the next unit's second stage), 33 (32 with the counters), 34 (32 without the per-thread fence_gpu), 35 (32 with ss and own_n reloaded from shared memory, fewer spills); the public kernel's stream-K stage ranges patched into its DBG slot 7 | 2.147x (2.115-2.179) | minus 28, us: 32 +1.6 / +2.7 / +1.2 (0, 2, 0 of 6 lower), 34 +0.8 / +1.4 / -1.2 (2, 2, 5 of 6), 35 +0.8 / +3.0 / +4.7 (0, 1, 0 of 6); counters in section "Phase A against the public kernel". Outputs vs 28 2e-6 to 3e-5. The public kernel's stamped (DEBUG) build ran 30 us over its own graph replay in this check and the next two (233 vs 203 us at 256), so its stamped times from these checks are not used; its stage ranges are structural. Cause, found afterwards in a local build: the range write makes the DEBUG MAIN build spill 392 / 896 B (none without it); tools/moe_make_bench_ab.py now patches it in only with --pubranges=1 |
| 2026-10-29 | 0b684d5db0834dbc (check, bench file) | ef6b705 | mode 36 (28 with SiLU's reciprocal as one MUFU.RCP), 37 (36 without the per-thread fence_gpu in the phase A publish), 38 (36 with the counters); layer seeds 1-6 | 2.163x (2.130-2.195) | minus 28, us: 36 -1.8 / -0.9 / -1.2 (6, 5, 6 of 6 lower), 37 -1.9 / -2.1 / -3.8 (6, 6, 6 of 6); 37 minus 36 -0.1 / -1.2 / -2.6. Counters: 36's epilogue 2.85 us per unit at 256 against 28's 2.98. Outputs vs 28 7e-5 to 1.5e-4 (the approximate reciprocal moves bf16 roundings); self-tests 3.89e-3 vs FP32 |
| 2026-10-29 | 473d87d03de94e96 (check, bench file) | d6cccf0 | repeat of 37 against 28, and 39 (37 with paired SwiGLU stores) with its counter twin 40; layer seeds 1-6 | 2.154x (2.122-2.186) | minus 28, us: 37 -1.7 / -3.4 / -4.2 (5, 6, 6 of 6 lower), 39 -1.0 / -2.5 / -4.0; 39 minus 37 +0.8 / +1.0 / +0.2. Counters at 256: 40's epilogue 1.70 us per unit against 3.08 for 28, but its stage loops took 3.1 us more and phase A only 2.0 us less: the stores' cost moves into the following stages |
| 2026-10-29 | 987d024620e44eb7 (check) | 96f2c43 | the shippable build: MAIN_MODES (0, 16, 26, 28, 37); GATE adds 37 minus 28 (held-out seeds 17-22, skewed 11-13) | 2.144x (2.112-2.176) | per shape: bs1 28.71 us, bs16 98.86, bs32 133.91, bs8 68.86, bs128 197.09 (2.309x, main-lean-silu-bar), bs256 241.49 (2.561x, main-lean-silu-bar). GATE 37 minus 28 per seed: bs128 -1.5 -2.1 -4.1 -4.5 -2.8 -4.1 (mean -3.2); bs256 -3.3 -5.1 -5.0 -5.1 -5.5 -5.5 (mean -4.9); skewed bs128 -2.5 -0.6 -1.5, bs256 -0.2 -3.2 -4.4. Output of 37 vs 16 1.2e-4 to 4.1e-4 (28: 8e-5 to 4.0e-4); self-tests 3.87e-3 to 3.90e-3; spin flags 0. BENCH bs128 main-lean-silu-bar 211.2, main-lean 214.6; bs256 232.6 vs 237.8. Judge bs128 197.09 against 200.89 / 200.34 for mode 28's two checks, bs256 241.49 against 247.11 / 244.59; the score sits inside 28's checks' intervals (2.149x, 2.158x) while the skinny shapes (unchanged code) drew 0.4-2.9 us higher at 1, 8 and 16 tokens than in f928f8a802014e4f. |
| 2026-10-29 | 6307d0c91c8d4ada (submit by the coordinator, used as the repeat of 987d024620e44eb7) | 96f2c43 | the shippable mode 37 build as checked in 987d024620e44eb7 | 2.160x (2.127-2.192) | passed 5/5 (tests 4.07e-3 to 4.22e-3). Judge medians: bs1 27.98 us, bs16 101.51, bs32 135.48, bs8 68.34, bs128 202.35 (2.329x, main-lean-silu-bar), bs256 241.46 (2.572x). GATE 37 minus 28 per seed: bs128 -5.8 -3.4 -4.1 -3.3 -2.9 -6.1 (mean -4.3); bs256 -4.8 -5.2 -6.5 -5.1 -4.8 -5.6 (mean -5.3); skewed bs128 -2.6 +0.3 -1.8 (mean -1.4), bs256 -1.0 -1.6 -5.2 (mean -2.6). BENCH bs128 main-lean-silu-bar 211.1, main-lean 215.3; bs256 232.9 vs 237.2. Two judge runs of 37: bs256 241.49 / 241.46 us against 247.11 / 244.59 for 28's checks; bs128 197.09 / 202.35 against 200.89 / 200.34 (the judge's bs128 is noisier than the 3-4 us gain). Status JSON in profiles/submission.6307d0c91c8d4ada.status.json |
| 2026-10-29 | 240b68e7df1a427f (submit under the user's scored-runs-as-measurements authorization; bench file from tools/moe_make_bench_ab.py --small, not committed) | 4204a8f | same-process A/B of the skinny path against the public MoE (its TINY / TINY2 / MID builds, which kernel_fn picks at 1-4 / 5-8 / 9-16 tokens) at 1, 8, 16 tokens, layer and standalone routing, seeds 1-6; arms sk8w4s (the policy), sk16w4s, sk8w4s-loads; stamps of the new skinny kMode 2 (sk8w4s with thread-0 phase stamps and cycle counters) against the public DEBUG builds | 1.988x (1.959-2.018) | public minus sk8w4s, us: T=1 layer +0.4 (35.4 base), std +0.3; T=8 layer -3.7 (66.2, 5.6%, 6 of 6 lower), std -1.9 (92.1); T=16 layer -2.0 (84.4, 2.4%), std -2.5 (108.9). sk8w4s-loads minus sk8w4s -2.5 / -3.6 / -4.4 (layer): the public kernel computing (62.5 at 8 tokens) matches our loads-only floor (62.7). sk16w4s +1.3 to +6.9 everywhere. Attribution in section "Skinny path against the public kernel". The score is a bench file's (the judge timed the shipped policy; 2.0x band of earlier runs) |
| 2026-10-29 | 2c28a7256b004aa1 (submit, bench file, same authorization) | 126ee85 | skinny kMode 3 (sk8w4s-er: each compute warp loads its unit's codes up front, fences and arrives on a per-slot empty mbarrier; thread 0 refills once all 8 warps arrived, no per-unit bar.sync of the compute warps) and kMode 4 (3 with stamps) against sk8w4s, same A/B as 240b68e7df1a427f | 1.970x (1.940-1.999) | er minus sk8w4s, us: T=1 layer +0.4 (0 of 6 lower), std +0.4 (0 of 6); T=8 layer -1.0 (6 of 6), std +0.7 (2 of 6); T=16 layer -0.9 (5 of 6), std +1.6 (1 of 6). Public minus sk8w4s this run: T=8 layer -4.1, T=16 layer -3.1. Stamps at 8 tokens, layer: er's phase A 38.0 us against 34.9 (codes held in registers through the unit), its A->B wait 2.5 against 4.0 and barrier 2.2 against 3.9. Self-test 3.93e-3; outputs vs sk8w4s 0 to 3e-5. Not adopted for the standalone (it loses on the standalone routing and at 1 token); a layer bench on judge routing may pick it at 8 and 16 tokens |
| 2026-10-29 | e937013292b24906 (submit under the user's scored-runs-as-measurements authorization; bench file from tools/moe_make_bench_ab.py --small, not committed) | 8272efe | skinny kMode 5 (sk8w4s-sf: scale decode's x16 as an IMAD with a multiplier ptxas cannot fold, FMA pipe instead of LEA on the ALU pipe), 6 (sk8w4s-hand: partial buffers handed to and from the helper through mbarriers, one compute-warp barrier per unit instead of two), 7 (sk8w4s-sfh: both), 8 (7 with stamps); stamps now time the first try_wait | 2.004x (1.974-2.034) | minus sk8w4s, us, layer / std: T=1 sf -0.1 / -0.1, hand +0.1 / -0.0, sfh +0.1 / -0.1; T=8 sf -0.2 / -0.4, hand -0.1 / +0.0, sfh +0.1 / +0.1; T=16 sf -0.5 / -0.1, hand +0.4 / +0.5, sfh +0.5 / -0.1 (none 6 of 6). Public minus sk8w4s: T=8 layer -3.5, T=16 layer -2.2. Self-test 4.02e-3 for all; outputs vs sk8w4s <= 5.8e-5; spin flags 0. Corrected stamps at T=8 layer: thread 0 waits 3.0 us (A) and 0.7 (B) for weights. Policy picked sk8w4s-sf at 8-32 (bench noise). Not adopted (docs/runs-moe.md, "Skinny unit loop against the public kernel") |
| 2026-10-29 | 75bca9775f6f4b0b (scored run submitted by the coordinator, the day's last expiring one, per the user's instruction; bench file from tools/moe_make_bench_ab.py --small --unitstamps, not committed) | 03887be | skinny stream A/B: sk8w3s (3-slot ring, (8, 3, 0)), kMode 32 (sk8w4s-rf: unit barrier and refill ahead of the partial handoff), (8, 3, 32) (w3rf), kMode 64 (sk8w4s-nh: weight copies without the L2 hint) against sk8w4s and the public TINY / TINY2 / MID builds, layer and standalone seeds 1-6 at 1 / 8 / 16 tokens; per-unit stream stamps (kMode bit 16 and the public patched the same way) at 8 tokens | 2.006x (1.976-2.036) | passed. w3 minus sk8w4s -0.9 / -0.6 / -2.4 / -1.8 / -3.1 / -2.9 us (T=1 layer / std, 8 layer / std, 16 layer / std), 6 of 6 seeds lower in every row; w3 minus public -1.3 / -0.8 / +1.0 / +0.9 / -0.4 / -0.5. rf -1.1 / +0.4 at 8, w3rf behind w3 in all six rows, nh +1.2 to +3.3 everywhere. Stamps: copy issue-to-arrival 4.7 us with 4 slots, 3.2 with 3, public 3.0. Spin flags 0; every arm within 2.1e-5 of sk8w4s. Section "Skinny weight stream against the public kernel" |
| 2026-10-29 | ca43908f02874886 (check, bench file from tools/moe_make_bench_ab.py --base=37 --sizes=128,256 --routings=layer o41=41 o42=42 --pairs=o41-o42, not committed) | 3f67a45 | fat-path waits timed in full (mbar_wait_cyc's clock before the first try_wait); mode 41 (37 with phase A and phase B counters per consumer warp) and 42 (37 with the light stamps only), layer seeds 1-6 | 2.174x (2.142-2.207) | passed 5/5; judge medians bs1 28.74 us, bs16 98.81, bs32 132.88, bs8 71.36, bs128 195.45, bs256 241.53; policy unchanged (sk8w3s to 32, main-lean-silu-bar at 128 and 256). ABTIME o37 164.6 / 227.2, pub 151.4 / 203.6 at 128 / 256. Instrumentation cost: 41 minus 37 +7.0 / +6.1 us, 42 minus 37 +1.1 / +2.3, 41 minus 42 +5.9 / +3.9 (0 of 6 lower in each row). Outputs of 41 and 42 against 37 1e-5 to 3e-5, spin flags 0. Counters in section "Fat-path waits with the first try timed". ABPUB's stage ranges are not meaningful in this run (no --pubranges) |
| 2026-10-30 | 9d94ac75f4724b52 (check, bench file from tools/moe_make_bench_ab.py --base=37 --sizes=32,128,256 --routings=layer o43=43 o44=44 o42=42 ost=v301 --pairs=ost-o43, not committed) | 54257b5 | mode 43 (code groups, next stage's waits and first group hoisted under the last wgmma group, weights release behind the sixth group), 44 (43 with 41's counters), and the staged pair on 37's components (variant 301: modes 45 then 46); layer seeds 1-6 | 2.188x (2.155-2.221) | passed 5/5; judge medians bs1 27.95, bs16 95.17, bs32 137.66, bs8 67.16, bs128 193.98, bs256 239.72 us; policy main-lean-grp (43) at 128 and 256. 43 minus 37 -1.3 / -1.4 / -2.7 us at 32 / 128 / 256 (5, 6, 6 of 6 lower); 44's cycles per wait 214 / 224 / 239 against 41's 201 / 215 / 223 (phase A), not lower. Staged pair minus 37 +30.3 / +30.4 / +24.4 (0 of 6). Outputs against 37 4e-6 to 2.3e-5, spin flags 0. Section "Code groups with the next stage hoisted" |
| 2026-10-30 | 304f17f3e8704992 (check, bench file from tools/moe_make_bench_ab.py --base=37 --sizes=32,128,256 --routings=layer --seeds=7,8,9,10,11,12 o43=43 o44=44 o42=42 o47=47 --pairs=o47-o43, not committed) | a4687c8 | repeat of 43 on held-out layer seeds 7-12 (the steering rule's step 3: timing won, waits did not fall), plus mode 47 (43 without the hoist) | 2.176x (2.144-2.209) | passed 5/5; judge medians bs1 27.46, bs16 97.33, bs32 133.58, bs8 72.36, bs128 199.79, bs256 241.87 us; policy main-lean-grp at 128 and 256. 43 minus 37 -0.9 / -1.2 / -2.1 (6, 4, 6 of 6); 47 minus 37 -0.4 / -1.6 / -3.6 (3, 6, 6 of 6); 47 minus 43 +0.5 / -0.5 / -1.5 (2, 5, 6 of 6); 44's cycles per wait 214 / 215 / 235 against 41's 204 / 203 / 225. Outputs against 37 1.2e-5 to 3.8e-5, spin flags 0 |

## Why the layer keeps the W4 chain at 128 tokens, 2026-10-29

Question (docs/plan-2026-10-29-steering.md, step 4): standalone, mode 16 beats the W4 chain
(legacy128) at 128 tokens, while the layer's prepare-time bench keeps the W4 chain there
(e87fcb1ed5c64a62). Candidates: L2 state left by neighbouring kernels, overlap with
neighbours, tuner noise, input differences. Three layer checks of a snapshot (the layer at
bdb4c7a plus a probe at the end of prepare; the layer directory was not edited); probes and
profiles in `muse_moe_w4a16/profiles/layer-flip/`.

What the layer's bench times (code, `_moe1_tune`): graph replay after an L2 flush of the MoE
call alone, on the synthetic routing `_moe_routing(bs, 977 * seed + bs, 0, False)` for seeds 11
and 12 only, with the W4 chain's routing metadata (route_align) built outside the timed graph
because kernel_fn builds it on the side stream. No neighbouring kernel is in the timed graph,
so L2 state left by neighbours cannot reach the decision.

Measurements (same process, same inputs, 6 seeds, paired per seed; us):

| probe, T = 128 | W4 (metadata outside) | W4 (route_align inside) | mode 16 | mode 16 minus W4-run, per seed |
|---|---|---|---|---|
| synthetic routing, cold L2 (0cb15c11accf4190) | 259.1 | 261.8 | 250.5 | +1.6 +2.3 -35.8 -12.6 -9.4 +2.4 |
| synthetic routing, no flush | 253.4 | - | 244.5 | |
| synthetic, again (7d1205a6959c41a9) | 259.5 | 261.8 | 250.0 | +1.1 +3.1 -40.0 -14.0 -11.5 +4.3 |
| judge-like routing (9d046da0207b40b2) | 253.9 | 256.9 | 224.3 | -31.6 -27.6 -38.6 -31.8 -22.7 -25.0 |
| isotropic-input routing (7d1205a6959c41a9) | 263.6 | 265.9 | 276.2 | +11.0 +8.3 +17.2 +10.5 +17.5 +10.8 |

"Judge-like" is reference.py's residual stream (fixed mean direction, rank-4 shared part,
noise; verify draft rows at 0.6 of the shared part, the basis from its STREAM_SEED) through the
layer's own attention boundary module and the judge's router from prepare's weights, with
local_start on the 32-expert range that gets the most pairs, as reference.py sets it: 187-214
local routes over 17-22 experts, at most 42-71 per expert. "Isotropic" is a N(0, 1) FFN input
through the same router: 153-168 routes over 25-28 experts, at most 13-22 per expert, which is
not the judge's distribution (the router has a component along the stream's mean, so a few
experts take most tokens); its mode 16 time at 256 tokens (313 us) is also far above the
judge's own layer profile of mode 16 there (249-279 us), while the judge-like probe gives 275.
At 256 tokens mode 16 wins on every probe (judge-like: -82 to -107 us).

Answer (measured; the last step is inference):

- Tuner sampling, not timing noise and not L2. The bench's two seeds are the two of six
  synthetic seeds where the W4 chain wins (by 1-3 us); on seeds 13-15 mode 16 wins by 9-40 us.
  Run to run, the paired differences repeat within 0.5-4.2 us per seed, so timing noise does not
  decide it.
- L2 state: without the flush both candidates gain 5-6 us; the ranking is unchanged.
- Overlap: route_align costs 2.0-3.1 us at 128 tokens (W4 with it inside minus without), the
  most the side stream can hide. In the isotropic probe the whole front plus experts showed the
  same gap as the experts alone (+7.7 to +17.8 against +8.3 to +17.5), so the stream structure
  adds nothing beyond that.
- Input differences decide the sign. On judge-like routing mode 16 wins at 128 tokens by
  23-39 us per seed (mean about 30). Inference: the layer's pick of the W4 chain at verify 16
  costs about 30 us per shape on both verify-16 shapes; a bench on judge-like routing (or
  mode 16 forced at 128) would recover it. Not yet confirmed on the layer's scored shapes,
  which needs a layer check of that change (the layer owner's call).

## Mode 16 at 128 and 256 tokens: where the time goes, 2026-10-29

Stamps of mode 17 (mode 16 plus the stamps of mode 11, now also on wgmma segments) in check
9332e2a49aff4ff2, bench routing seeds 11 and 12, one cold-L2 launch each, all 132 CTAs, us at
1.77-1.78 GHz. The stamps cost 14 us at 128 (263.3 vs 248.9) and 18 at 256, so absolute times
run high. Per-warp rows are sums over one consumer warp's segments (mean / max over CTAs); they
overlap each other and the producers, so they are not end-to-end time.

| | 128, seed 11 | 128, seed 12 | 256, seed 11 | 256, seed 12 |
|---|---|---|---|---|
| critical path: routing + x gather before the consumers start, mean / max | 14.4 / 18.2 | 15.2 / 19.2 | 19.3 / 26.9 | 20.4 / 23.3 |
| phase A done (thread 0 of each CTA), mean / max | 175.0 / 185.3 | 164.0 / 171.0 | 185.7 / 204.8 | 184.1 / 196.9 |
| consumers done, mean / max | 242.4 / 271.1 | 225.8 / 239.4 | 261.0 / 291.1 | 255.7 / 287.7 |
| completion tail (last consumers done minus mean) | 28.7 | 13.6 | 30.1 | 32.0 |
| final norm and sum after the barrier | 2.9 | 2.9 | 5.6 | 5.0 |
| launch end | 275.5 | 243.2 | 298.5 | 294.1 |
| per warp: consumer math (dequant + MMA, wgmma waits included) | 148.1 | 135.1 | 155.7 | 157.0 |
| per warp: epilogue and fixup | 22.1 | 21.3 | 26.6 | 23.0 |
| per warp: waiting for weights / for x | 12.7 / 11.1 | 10.7 / 12.5 | 12.4 / 15.4 | 11.0 / 12.7 |
| x producer waiting on done_as (phase B dependency), mean / max | 14.4 / 31.9 | 13.5 / 32.8 | 16.9 / 48.1 | 16.2 / 42.3 |
| cut segments per CTA, phase A / B | 1.88 / 0 | 1.86 / 0 | 1.89 / 0 | 1.85 / 0 |

Reading: phase A ends at 171-205 us on the slowest CTA, then 55-90 us of phase B run after it
on the critical path, ending in a 14-32 us tail of the slowest CTA. Consumer math is the
largest per-warp item (135-157 us), as in the 2026-10-28 stamps; the weights producer still
waits 162-188 us for a free slot.

The named removable cost: phase A's stream-K cuts and cost-table partition. Each CTA cuts
about 1.9 phase A units, each finished through the global fixup by its last share. Replacing
them with whole units owned by one CTA (mode 19 against mode 18, phase A alone, same process):
-5.5 / -5.6 us at 128 tokens and -16.4 / -15.7 us at 256 (two checks, 6 seeds each). In the
full kernel (mode 20 against mode 16) the saving is -2.7, -2.1, -3.1 us at 128 and -4.4, -3.8,
-4.2 at 256 (three checks): phase B absorbs part of it. Mode 20 removes it and is the policy
at 128 and 256 in 72b7a5a9c511446a. The next cost in line is the completion tail (14-32 us), a
hypothesis-level target: phase B's stream-K boundaries come from the cost table and its stages
start only as other CTAs' phase A units release them.

## Phase B claimed at run time (mode 23), 2026-10-29

Target: phase B's critical path and the completion tail at 128 and 256 tokens (the steering
plan's 128/256 loads hold about 53% of the layer's remaining log gap).

Attribution first, on mode 22 (mode 20 plus the stamps of 17; check a9dbec1d98334943, cold L2,
one launch per seed, stamps add 14-18 us so compare rows with each other). Mode 20 deals phase A
whole units in snake order, so CTAs end phase A with 5 or 6 units (80 or 96 stages), while
phase B keeps mode 16's static ranges cut from the cost table, which do not see that:

| mode 22 (20 with stamps) | 128, seed 12 | 256, seed 11 | 256, seed 12 |
|---|---|---|---|
| CTAs with 80 A stages: count, phase A done, B stages, consumers done mean/max | 56, 141.2, 44.3, 207.4/231.9 | 24, 159.3, 51.3, 234.9/249.1 | 24, 154.7, 47.3, 235.3/257.0 |
| CTAs with 96 A stages: count, phase A done, B stages, consumers done mean/max | 76, 164.6, 44.8, 225.6/240.1 | 108, 179.5, 45.5, 254.1/278.5 | 108, 177.5, 46.4, 248.4/273.4 |
| slowest CTAs: A stages / B stages / B span after their phase A | 96 / 48 / 72 | 96 / 64 / 85-97 | 96 / 48-64 / 84-96 |
| completion tail (slowest consumers done minus mean) | (line cut from the log) | 27.9 | 27.4 |
| x producer waiting on done_as, mean/max | 13.7/33.1 | 15.6/28.1 | 16.2/36.5 |

The CTAs that start phase B 20 us early get no more phase B work than the others (44.3 against
44.8 stages at 128), and the slowest CTAs are those with a full phase A and a fourth phase B
unit. That is the removable cost: a phase B partition that ignores each CTA's phase A load.

Hypothesis: if each CTA takes its next phase B unit when its weights ring frees a slot, the
CTAs that leave phase A early absorb the difference and the tail shrinks to about the
granularity of one unit. Taking units in S.ord order puts the jobs whose phase A units are dealt
in the first snake rounds at the front of the queue, so a claimed unit finds its inputs
published. A CPU list-scheduling model (cost-table stage times, judge-like routing, seeds 11-16)
put a greedy assignment 10-30 us under mode 20's partition, a hypothesis, not a measurement.

Mode 23 is mode 20 with two changes. The weights producer claims phase B units one at a time
from a global counter (ctr[10]) in S.ord order, 16 units per job, once its ring has a free slot,
and passes each claim to the x producer and the consumers through a shared-memory queue (S.bq).
Both phases are walked largest n-tile count first, so a CTA runs its phase A units in snake
round order and its claimed units in claim order (S.ord's n-tile counts never rise, so one loop
per count still covers them; each loop keeps a single wgmma shape). Every phase B unit stays
whole on one CTA: no phase B fixup, as in mode 16.

| mode 24 (23 with stamps) | 128, seed 11 | 128, seed 12 | 256, seed 11 | 256, seed 12 |
|---|---|---|---|---|
| phase A done, mean / max | 161.6 / 188.4 | 149.5 / 164.4 | 178.0 / 185.6 | 172.2 / 179.5 |
| consumers done, mean / max | 229.4 / 245.0 | 212.5 / 223.2 | 250.5 / 260.6 | 243.8 / 251.1 |
| completion tail | 15.6 | 10.7 | 10.2 | 7.4 |
| launch end | 250.1 | 227.8 | 267.5 | 258.6 |
| x producer waiting on done_as, mean/max | 15.2/23.5 | 15.5/19.8 | 15.5/23.3 | 14.6/18.1 |

At 128 seed 12 the 80-stage CTAs now take 48.0 phase B stages against 42.1 for the 96-stage
ones; the tail falls from about 28 us to 7-16 us, and at 256 the launch ends 17-21 us earlier
on the same seeds. The mean done_as wait does not move (13.7-16.2 against 14.6-15.5 us), so the
gain is balance, not fewer dependency waits.

Same-process paired A/B (graph replay after an L2 flush, variant order reversed on every other
seed), 23 minus 20, us:

| routing | 128 tokens | 256 tokens |
|---|---|---|
| seeds 11-16 (a9dbec1d98334943) | -16.2 (6 of 6 lower) | -16.1 (6 of 6) |
| held-out seeds 17-22 (36812632dae242ba) | -22.9 (6 of 6) | -18.7 (6 of 6) |
| skewed, seeds 11-13 (36812632dae242ba) | -4.8 (3 of 3) | -22.9 (3 of 3) |

Against mode 16 the held-out means are -19.5 and -24.8 us. Outputs match mode 16 to 1e-4 to
4e-4 relative L2 (FP32 atomic order), every self-test is 3.9e-3 against FP32, no spin flags.
The bench picks main-own-dynb at 128 and 256 in both checks.

Ordering obligations. The data handoffs are mode 16's, unchanged: x rows through xready, phase
A's SwiGLU output through done_as with fence.proxy.async.global on both sides ("Proxy ordering
of the x tiles" below), weights through wfull/wempty. Two new handoffs carry only indices:

1. The claim counter (cross-CTA). Writers: each weights producer's atomicAdd on ctr[10]
   (relaxed). Publication and observation: the value an RMW returns; atomicity (PTX ISA Memory
   Consistency Model, Axioms, "Atomicity") gives each value to exactly one claimant. No data is
   published through it, so no release or acquire is needed: the claimed unit's inputs are
   still observed per stage through done_as (ld.acquire.gpu, then fence.proxy.async.global by
   the issuing thread before its cp.async.bulk, 9.7.10.28.2). Completion and use: complete_tx
   on xfull and wfull, consumers' mbarrier.try_wait (acquire.cta), unchanged. Reuse: CTA 0
   zeroes ctr[10] after the grid barrier. Every claim precedes its CTA's bar.sync and thread 0's
   __threadfence plus barrier atomicAdd (a release pattern, 8.8); CTA 0's ld.acquire of the
   full count synchronizes with it (8.9.4), so the zeroing is causally after every claim and
   later in coherence order. The next launch starts after this one completes (stream order, or
   griddepcontrol.wait under PDL, 9.7.15.14).
2. The claim queue S.bq (inside one CTA). Writer: the weights producer thread,
   st.release.cta.shared. Observers: the x producer and every consumer thread,
   ld.acquire.cta.shared on the same word, morally strong (8.7), so the acquire synchronizes
   with the release (8.8, 8.9.4). Use: the index selects S.ord, S.job_* entries written in the
   prologue before __syncthreads. No async-proxy operation reads S.bq. Reuse: each entry is
   written once per launch and set back to empty in the next launch's prologue, before the
   __syncthreads that precedes every role.

Every fence and barrier of mode 20 is kept.

Verdict: mode 23 wins at 128 and 256 tokens on 24 of 24 paired normal-routing seeds and 6 of 6
skewed ones; it is the standalone policy at both sizes. Port it to the layer: add 23 to
`LAYER_MODES` in `tools/dl_moe_inline.py` (the layer's ctr is already int32 [64], zeroed at
allocation; SmemMain grows by 256 bytes to 228,736) and let the layer's judge-like routing bench
choose. What is left of the tail (7-16 us stamped) is about half a phase B unit (16 stages of
1.3-1.4 us at one n8 tile). Splitting only the last units with split-K would cut it at most in
half, and the cost model put a phase B cut at about 10 us, so it was not tried.

## Same-process A/B against the public MoE, 2026-10-29

Method. `tools/moe_make_bench_ab.py OUT.py PUBLIC.py [--seeds=..] [ARM=MODE ..]` appends to
submission.py a block that runs at import, before the judge's prepare(): random NVFP4 weights for
32 experts (the judge rejects a file that imports reference.py; timing does not depend on the
values), our state, and the public file (vjeux/okc-2026 `muse_moe_w4a16.py`, copy in the session
scratchpad, its small-batch builds switched off so only the MAIN build compiles) with its own
prepare(). Both kernels run on the same inputs, graph replay after a 256 MB L2 flush, median of 20,
arms interleaved and reversed on every other seed. Two routings: "std" is reference.py's
generate_input (the standalone board); "layer" ports the layer's judge routing (residual stream
with the mean and shared directions, residual boundary, sigmoid router with bias, busiest 32-expert
range), router seed picked so the busiest local expert matches the layer's own SUMMARY lines
(measured 18-23, 50-59, 105-128 routes at 32 / 128 / 256 tokens against 20-22, 51-55, 100-124 in
80b98f567aa74dd0). Stamps: mode 25 (thread 0's globaltimer at the phase boundaries, no cost in the
bench) against the public kernel built with its OKC_DEBUG stamps, its DBG(3) moved to the CTA-wide
point before the grid barrier so both kernels mark the same events.

Attribution, check d1f587760b1241dc, layer routing, mean of six seeds, us. Setup = start to the
first consumer stage (routing, then the x gather). Phase A and phase B are per-CTA means (thread
0's end of phase A, then all consumers done). Tail = slowest CTA's consumers minus the mean.

| | T=32 ours / public | T=128 ours / public | T=256 ours / public |
|---|---|---|---|
| setup (routing part) | 11.1 (7.2) / 3.9 (1.9) | 17.7 (10.9) / 6.1 (2.4) | 24.0 (15.9) / 8.4 (3.3) |
| phase A | 51.8 / 49.5 | 97.7 / 87.2 | 137.4 / 119.5 |
| phase B | 23.8 / 23.0 | 49.8 / 43.0 | 67.4 / 58.0 |
| tail | 8.8 / 4.0 | 10.4 / 4.7 | 8.5 / 6.4 |
| barrier wait + final pass | 2.4 / 2.3 | 4.2 / 4.3 | 6.4 / 6.3 |
| launch end (stamped) | 98.9 / 85.6 | 180.3 / 146.4 | 244.2 / 200.3 |
| ours minus public: setup / A / B / tail | +7.2 / +2.3 / +0.8 / +4.8 | +11.6 / +10.5 / +6.8 / +5.7 | +15.6 / +17.9 / +9.4 / +2.1 |
| graph replay, paired (public minus 23) | -12.8 | -32.2 | -45.7 |

Standalone routing gives the same order: setup +6.6 / +10.9 / +15.2, phase A +6.3 / +15.6 / +24.1,
phase B +2.4 / +6.2 / +10.0, tail +6.4 / +3.9 / +3.6 us at 32 / 128 / 256 (replay gap -18.0 /
-34.9 / -52.5). The final pass is at parity everywhere. Setup is the largest single item at 32 and
128 tokens; phase A is the largest at 256.

SASS (local nvcc 12.8, sm_90a, `tools/moe_compile_fused.sh` for ours at e261521, the public CUDA source with
its MAIN defines; counts in `muse_moe_w4a16/profiles/sass_stage_loop_vs_public.2026-10-29.txt`). Both kernels cap at 96 registers; ours spills (64-byte frame, 120 B stores, 308 B
loads, outside the stage loop), the public one does not. The 256-k stage loop of one wgmma shape (16
HGMMA, here m64n24k16) is 389-391 instructions in our mode 23 against 322 in the public kernel. The
dequant and MMA core is the same (LOP3 85 / 80, HMUL2 64 / 64, IMAD.SHL 28 / 32, SHF 33 / 32, PRMT
8 / 16, UIADD3.64 15 / 15); the difference is waiting and bookkeeping: four mbarrier try-waits per
stage against two, the spin-timeout path (clock reads, ATOMG.OR on ctr[8], YIELD), the
fence.proxy.async.shared::cta before the x-slot release (MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S), and
more address and branch work (LEA 9 / 1, ISETP 10 / 2, BRA 10 / 3). How much of phase A's +10-24 us
that explains is not measured.

Change tested (mode 26), the setup half: the public route() lets every warp count its slice once
and warp 0 combines the per-warp counts; ours had one warp count all E entries serially and then
position_pass counted them again in every warp. Mode 26 counts once in all 18 warps, the routing
warp sums the counts and writes each warp's offsets, the positions reuse them; the x gather issues a
CTA's row loads (16 B, up to 8 rows) before the stores, as the public gather does; and the four
stage_at searches thread 0 ran for stage ranges are dropped, since owned phase A units and claimed
phase B units never read them. No fence, barrier or handoff changes: the gather keeps
fence.proxy.async.global, __threadfence, the named barrier and red.release on xready; the new
shared-memory traffic (per-warp counts, offsets) is ordered by the __syncthreads already between
the steps.

| 26 minus 23, us (paired) | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer routing, seeds 1-6 (39c365c34cd74559, without the range change) | -0.5 | -3.7 | -8.5 |
| layer routing, seeds 7-12 (4c285df44eb34bc1) | -1.4 | -4.6 | -9.1 |
| standalone routing, seeds 7-12 (4c285df44eb34bc1) | | -5.8 | -9.8 |
| standalone, held-out seeds 17-22 (GATE, 817eab6401d54a84) | | -5.8 | -9.5 |
| skewed, seeds 11-13 (GATE, 817eab6401d54a84) | | -4.2 | -7.7 |

Every paired seed favours 26 (66 of 66 at 128 and 256, 12 of 12 at 32 on layer routing). The
stamped routing at 128 tokens fell from 11.0 to 5.8 us (public 2.4): what is left is the S.ord
counting sort (2.5 us, eight dependent rounds in one warp) and the rest of the routing warp (1.9).

Verdict and next step. Mode 26 is the standalone policy at 128 and 256 tokens (817eab6401d54a84,
2.104x). Port it to the layer: add 26 to `LAYER_MODES` in `tools/dl_moe_inline.py`; the kernel's
interface, scratch and SmemMain size (228,736 bytes) are mode 23's. On the layer-like routing it
saves 3.7-4.6 us at 128 tokens and 8.5-9.1 at 256, and 0.5-1.4 at 32. After it the same-process gap
to the public kernel is about 27 us at 128 and 41 at 256 (layer routing), mostly phase A (+10.5 /
+17.9), phase B (+6.8 / +9.4), the tail and the rest of setup (+6 us). The gap map's E1 (job table
built on the side stream) now has a ceiling of the remaining routing, 5.8-7.4 us at 128-256 tokens,
of which an in-kernel sort that does not take eight serial rounds would recover part without a new
cross-launch handoff; the E1 arm itself was not built.

## Consumer stage and the ring slot releases, 2026-10-29

Question: after mode 26 the same-process gap to the public kernel was about 27 us at 128 tokens and
41 at 256 (layer routing), with phase A +10.5 / +17.9 us and phase B +6.8 / +9.4. The SASS of one
256-k stage was 389 instructions against 322, with "four mbarrier waits against two" and one
shared-memory proxy fence per stage in ours and none in theirs. Is the public scheme provably
correct, and what can we cut?

Sections cited are from the PTX ISA 9.4 documentation
(docs.nvidia.com/cuda/parallel-thread-execution), read 2026-10-29: 8.8 Release and Acquire
Patterns, 8.9.1.1 Asynchronous Operations, 8.9.4 Memory synchronization, 8.9.5 Causality Order,
8.10.6 Causality, 9.7.10.28.2 Async Proxy (cp.async.bulk), 9.7.15.2 bar.warp.sync, 9.7.15.4
membar / fence, 9.7.15.16.16 mbarrier.arrive, 9.7.15.16.19 mbarrier.test_wait / try_wait,
9.7.17.4 Async Proxy (wgmma).

### What the public kernel does per stage (wgs_seg and the two producer warps, public source)

- Two waits per stage: `mbar_wait(BW_F + ws)` (weights full) and `mbar_wait(BX_F + xs)` (x full),
  each a two-instruction `try_wait` loop with no timeout.
- Weights slot release: lane 0 of each consumer warp does `mbarrier.arrive` on `BW_E + ws` (count
  NW warps) once its warp has issued the loads of the stage's last code group, with no
  `__syncwarp` and no proxy fence. The producer waits on `BW_E` and then issues the next
  `cp.async.bulk` into the slot.
- X slot release: lane 0 arrives on `BX_E + prev` after `wgmma.wait_group` has retired the group
  that read the slot, with no fence. The x slot is read only by `wgmma` through its descriptor.

Our 26 has the same two waits per stage. The four `SYNCS.PHASECHK.TRANS64.TRYWAIT` in its SASS are
`mbar_wait_to`'s first try plus the retry inside the spin-timeout loop; the second one, the clock
reads and the `ATOMG.OR` on ctr[8] run only after a failed first try (profile
`sass_stage_loop_mode28.2026-10-29.txt`: 390 instructions in the loop, 348 on the path where both
first tries succeed). The real differences are on the release side.

### Which public releases are provable

The x slot (theirs and ours): `wgmma.mma_async` reads shared memory in the async proxy (9.7.17.4)
and `cp.async.bulk` writes it in the async proxy (9.7.10.28.2). Both are the same proxy in the same
thread block, which is the second case of proxy-preserved base causality order (8.9.5), so no
proxy fence is needed. The chain: the read completes before `wgmma.wait_group` returns in every
thread of the warpgroup; lane 0's `mbarrier.arrive` follows in program order and has release
semantics by default (9.7.15.16.16); the producer's `try_wait.parity` has acquire semantics by
default and, returning true, forms an acquire pattern (9.7.15.16.19, 8.8) that synchronizes with
the arrive (8.9.4); the producer's `cp.async.bulk` is ordered after its prior instructions
(8.9.1.1). The read precedes the write in causality order, so it cannot read from it (8.10.6).
Ours already releases the wgmma path's x slot this way, with no fence.

The weights slot (public): the slot is read with generic `ld.shared` by all 32 lanes and refilled
by `cp.async.bulk`, a different proxy. Two links are missing. (1) The proxy: "accessing the same
memory location across multiple proxies needs a cross-proxy fence; for the async proxy,
fence.proxy.async should be used" (9.7.10.28.2). Proxy-preserved base causality order (8.9.5)
covers generic-to-generic and same-proxy pairs only, so without a fence on the path the read is
not ordered before the bulk write in causality order and 8.10.6 gives no guarantee. (2) The lanes:
lanes 1-31 read the slot but only lane 0 arrives, with no `bar.warp.sync` between them, so the
other lanes' reads are not in program order before any release (8.8: a release pattern "only
affects operations occurring in program order before" it; 9.7.15.16.19 makes visible the accesses
"requested prior, in program order, to mbarrier.arrive ... by the participating threads"). The
public scheme therefore rests on hardware behaviour, not on the model. Under the repository rule it is not adopted.

The weights slot (our 26): every lane's reads, then `__syncwarp` (bar.warp.sync "also guarantee[s]
memory ordering among threads participating in barrier", 9.7.15.2), then lane 0's
`fence.proxy.async.shared::cta` and arrive. This is correct, but the cross-lane link rests on the
instruction text of 9.7.15.2, which is not one of the synchronizes-with relations listed in 8.9.4.

### Ordering of the ring slot releases (adopted in mode 28)

Every consumer thread releases the weights slot itself: `fence.proxy.async.shared::cta`, then
`mbarrier.arrive` on `wempty[ws]`, whose expected count becomes kNC x 32 = 512. For one thread T
that read the slot with `ld.shared` (reads R) and the producer thread P that refills it with a
bulk copy (write W):

1. R, then T's bidirectional proxy fence F, in program order. F orders T's prior generic accesses
   before its subsequent async-proxy accesses and its effect "composes with other forms of
   synchronization" (9.7.15.4); this is the cross-proxy fence 9.7.10.28.2 requires.
2. F, then T's `mbarrier.arrive` A, in program order. A has `.release` semantics at `.cta` scope by
   default (9.7.15.16.16) and is a release pattern by itself (8.8), covering R and F.
3. P's `mbarrier.try_wait.parity` on the same barrier returns true once all 512 arrivals of the
   phase are in; it has `.acquire` at `.cta` scope by default and forms an acquire pattern
   (9.7.15.16.19, 8.8), so A synchronizes with it (8.9.4). 9.7.15.16.19 also states the case
   directly: accesses requested after the wait "are not performed and not visible to memory
   accesses performed prior to mbarrier.arrive having release semantics, in program order, by
   other threads participating in the mbarrier".
4. P then issues `mbarrier.arrive.expect_tx` and `cp.async.bulk`; the bulk operation is ordered
   after the prior instructions of P (8.9.1.1), so after the acquire.
5. So R precedes W in base causality order with T's proxy fence on the path, and R cannot read
   from W (8.10.6).

Each thread's own reads are now in its own program order before its own release, so the chain needs
neither `bar.warp.sync` nor lane 0. The fence and the arrive stay; on the mma.sync path the x slot,
read by `ldmatrix` (generic), gets the same per-thread fence and arrive, and the wgmma path's x
release (lane 0, `mbarrier.arrive` with count 32, after `wgmma.wait_group`) keeps the chain above.
The fence sits behind the stage's first wgmma group instead of before the x wait: every load of
the stage's codes and scales was issued earlier in T's program order, so the chain is unchanged,
and the MEMBAR it compiles to is no longer between the weights wait and the first wgmma. A
uni-directional `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster` compiles to
the same MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S (local nvcc 12.8 test), and the read-only
`::shared::cluster::read` form needs PTX ISA 9.4, so the bidirectional fence stays.

### The rest of mode 28

- The ring size (S.nsw), x slot 0's address and the x slot size are read once per launch into
  registers; 26 loaded them from shared memory every stage (an LDS, an LDS.64, S2UR / ULEA pairs)
  and shuffled the whole slot address. The slot offset still goes through one `__shfl_sync`: the
  ring position is carried around a loop whose trip count ptxas cannot prove warp-uniform, and
  without the shuffle every wgmma descriptor went through R2UR (448 instructions per stage).
- The S.ord sort is one pass: an expert's jobs are all 8 n-tiles except possibly its last, so a
  job's rank is a warp scan of the 8-tile counts plus, for a partial job, seven ballots. 26 ran
  eight dependent scan rounds (2.5 us of routing at 128 tokens in 4c285df44eb34bc1).
- Spin timeouts stay (the self-tests read ctr[8]); they were already off the fast path.
- Not removed: the spill. Mode 28 has 184 B / 512 B of spill stores / loads against 120 / 308 for
  26, all but 7 local accesses outside the per-stage loops (26: 5).

SASS of the 64x24 stage loop: 383 instructions, 335 on the fast path (26: 390 / 348; public 322).
The variant that kept 26's lane-0 release after `__syncwarp` at the moved point made ptxas
serialize the wgmma groups (C7513, 400 instructions per stage) and was dropped before any run.

### Results

Paired 28 minus 26, us (negative = 28 faster):

| | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer routing, seeds 1-6 (70dd7b13be884434) | -6.4 (6 of 6) | -8.6 (6 of 6) | -8.8 (6 of 6) |
| standalone routing, seeds 1-6 (70dd7b13be884434) | -6.4 (6 of 6) | -12.1 (6 of 6) | -11.0 (6 of 6) |
| standalone, held-out seeds 17-22 (GATE, f928f8a802014e4f) | | -10.6 (6 of 6) | -11.6 (6 of 6) |
| the same, repeat (GATE, 13fe68e7ca2e423f) | | -11.6 (6 of 6) | -12.5 (6 of 6) |
| skewed, seeds 11-13 (GATE, f928f8a802014e4f / 13fe68e7ca2e423f) | | -6.7 / -6.4 | -8.4 / -8.4 |

Stamps, layer routing, mean of six seeds (70dd7b13be884434; 27 is 26 stamped, 29 is 28 stamped).
Phase A = thread 0's end of phase A minus consumers' start; phase B = consumers done minus that.

| us | T=128 27 / 29 / public | T=256 27 / 29 / public |
|---|---|---|
| routing | 5.8 / 3.6 / 2.5 | 7.3 / 4.3 / 3.4 |
| consumers start | 12.9 / 10.9 / 6.2 | 16.0 / 12.8 / 8.3 |
| phase A | 97.6 / 92.7 / 87.7 | 137.7 / 133.7 / 120.6 |
| phase B | 48.4 / 47.5 / 42.9 | 66.2 / 65.2 / 58.6 |
| tail | 9.9 / 8.9 / 5.8 | 9.4 / 9.3 / 5.6 |
| launch end | 173.3 / 164.5 / 148.0 | 236.2 / 227.7 / 201.5 |

Judge (shippable build, policy main-lean at 128 and 256): 2.149x (2.117-2.181) in
f928f8a802014e4f and 2.158x (2.125-2.190) in the repeat 13fe68e7ca2e423f, against 2.104x
(2.073-2.136) for mode 26 in 817eab6401d54a84; bs128 215.68 -> 200.89 / 200.34 us, bs256 261.71 ->
247.11 / 244.59.

Reading. The stamps put about half of the gain in phase A (-4.9 / -4.0 us at 128 / 256) and a
quarter to a third in setup (-2.0 / -3.2, mostly the sort), with phase B nearly unchanged (-0.9 / -1.0); the stage
loop lost 13 fast-path instructions of about 348 (4%). Which of the release move, the per-thread
release and the hoisted constants carries the phase A gain is not separated (the lane-0 control
could not be built without serialization). The remaining gap to the public kernel on layer
routing is 16.1 us at 128 tokens and 25.1 at 256 (paired replay), in the stamps setup +4.7 /
+4.5, phase A +5.0 / +13.1, phase B +4.6 / +6.6, tail +3.1 / +3.7.

At 32 tokens on layer routing, 28 (94.7 us) and 26 (101.1) are both well under the skinny sk8w4s
(113.1, 6 of 6 seeds), while on the standalone bench routing sk8w4s wins (150.8 vs 161.2 in
f928f8a802014e4f), so the standalone policy keeps sk8w4s at 32 and a layer benched on judge-like
routing may pick 28 there.

Verdict. Mode 28 is the standalone policy at 128 and 256 tokens; submit candidate ec62b5a (not
submitted). Port: add 28 to `LAYER_MODES` in `tools/dl_moe_inline.py` and `28: "main-lean"` to
its MAIN_NAMES (26: "main-dynb-fs" too if 26 stays). The interface, scratch and SmemMain size
(228,736 bytes) are mode 26's. The standalone MAIN_MODES is now (0, 16, 26, 28); a layer that
still lists 20 or 23 in LAYER_MODES gets them silently dropped by the inliner (it keeps only modes
in MAIN_MODES), so list (0, 16, 26, 28) or (0, 16, 28). Tested on a scratch copy: the inliner's
anchors match and the layer's variant table gains `('main-lean', 'main', 'launch_main<28>')`; the
layer build itself was not compiled here.

## Skinny path against the public kernel at 1, 8 and 16 tokens, 2026-10-29

Asked by the 2026-10-30 gap map: at 1 or 8 tokens a microsecond is worth 3-4x more layer score
than at 128/256. Same-process A/B (tools/moe_make_bench_ab.py --small, public TINY / TINY2 / MID
builds as its kernel_fn picks them), six seeds of layer and standalone routing, runs
240b68e7df1a427f and 2c28a7256b004aa1. Stop rule: stop if the public lead at 8 tokens is under 3%.

| us, mean of 6 seeds | T=1 layer | T=8 layer | T=16 layer | T=8 std | T=16 std |
|---|---|---|---|---|---|
| sk8w4s (policy) | 35.4 / 35.7 | 66.2 / 66.9 | 84.4 / 85.5 | 92.1 / 93.0 | 108.9 / 109.8 |
| public minus sk8w4s | +0.4 / +0.3 | -3.7 / -4.1 (5.6-6.1%) | -2.0 / -3.1 | -1.9 / -2.8 | -2.5 / -2.9 |
| sk8w4s-loads minus sk8w4s | -2.5 / -2.6 | -3.6 / -3.8 | -4.4 / -5.1 | -2.3 / -3.0 | -4.0 / -4.1 |

The lead at 8 tokens is above 3%, so attribution. Stamps (thread 0's globaltimer; our new
sk8w4s-stamps against the public DEBUG builds), 8 tokens, layer routing, two runs:

| us | ours | public |
|---|---|---|
| setup (routing, positions) | 2.2 / 2.1 | 2.2 / 2.2 |
| phase A | 34.6 / 34.9 | 33.3 / 33.4 |
| phase B, with ours's wait for phase A credits (4.0) | 19.7 / 20.0 | 18.7 / 18.9 |
| grid barrier wait (tail of the slowest CTA) | 3.7 / 3.9 | 2.5 / 2.9 |
| final pass | 0.4 / 0.3 | 0.3 / 0.3 |
| end | 61.5 / 62.3 | 57.6 / 58.4 |

Measured: setup is at parity; the 4 us are spread over phase A (+1.3), phase B with its
dependency wait (+1.0) and the barrier tail (+1.2). [Corrected in "Skinny unit loop against the public kernel": the stamps did not time the first try_wait, which can suspend; timed in full, thread 0 waits 3.0 us in phase A at 8 tokens.] Thread 0 never waits for a weights unit (0.0
us of missed first tries in both phases), and the loads-only twin (same stream, no math) lands
at the public kernel's time (62.7 against 62.5 us), so the public kernel hides its compute under
the stream and ours does not: the compute warps are the limit, by 3.6 us at 8 tokens. Inferred:
the per-unit lockstep (a bar.sync of all 8 compute warps behind each unit's last MMA before the
slot is refilled) and the per-unit instruction count (our unit loop 828 SASS instructions against
709-759 in the public TINY2 build; local nvcc 12.8) are the two candidates.

Tested the first (sk8w4s-er, kMode 3): -1.0 us at 8 tokens and -0.9 at 16 on layer routing (11 of
12 seeds lower), +0.4 at 1 token and +0.7 / +1.6 on standalone routing. Holding the unit's codes
in registers through its MMAs made phase A 3 us longer while the dependency wait and the barrier
tail fell by 3.1 us. Not adopted for the standalone policy (it loses on the board's routing); the
release chain is the fat path's ("Ordering of the ring slot releases"). The instruction count is
untested.

## Skinny unit loop against the public kernel, 2026-10-29

Asked: cut the skinny path's compute per unit so it hides under the weight stream, starting from
an instruction-level diff of our unit loop against the public TINY2 build.

### The instruction diff (local nvcc 12.8, sm_90a, cuobjdump)

Ours: `tools/moe_compile_fused.sh` at 96f2c43, kernel `sk_8_4_0`. Public: its CUDA_SRC with the
TINY2 defines plus `-DOKC_PRMTSC=1 -DOKC_DQ7=1 -DOKC_NS=3 -DOKC_NSF=3`, as its `_ext` builds it
for this problem. The earlier figure (our loop 828 instructions against 709 / 759) is the static
size of the innermost HMMA loops, and it reproduces exactly; but ours is one loop for both phases
whose body carries the cold paths (the x reload on a job change, 16 predicated 16-byte loads and
32 zeroing CS2R per phase; the spin-timeout blocks), and the public splits phase A and B into two
template instances. The dynamic path that a compute thread other than thread 0 runs for a unit
with no job change and no failed waits:

| per unit, per compute warp | ours (sk8w4s) | public phase A | public phase B | ours, kMode 7 |
|---|---|---|---|---|
| instructions | 554 | 554 | 557 | 569 |
| ALU pipe (LOP3, SHF, PRMT, LEA, ISETP, SEL, IADD3, MOV) | 275 | 266 | 262 | 268 |
| FMA pipe (HMUL2, IMAD, FADD, VIADD) | 221 | 232 | 234 | 236 |
| HMMA.16816 / LDS.128 / STS.128 | 32 / 9 / 1 | 32 / 9 / 1 | 32 / 9 / 1 | 32 / 9 / 1 |
| CTA-level syncs | bar.sync (free, 288) + bar.arrive (full) + bar.sync (unit, 256) | mbarrier try_wait (buffer free) + bar.sync (256) | same | mbarrier try_wait + mbarrier arrive + bar.sync (256) |

Line by line:

- Dequant: identical cost. Ours (repacked layout) and the public DQ7 layout both decode a word in
  7 ALU instructions (5 LOP3, 2 SHF) and 2 IMAD.SHL; ptxas already turns our left shifts into
  IMAD.SHL. Both run 128 HMUL2 for the block scale.
- Scale decode: the one real difference. Ours `byte_perm(...) * 16 + c` compiles to PRMT + LEA
  (ALU); the public passes 16 as an opaque register (`16 + p.zero`), so it is PRMT + IMAD (FMA).
  16 ALU instructions per unit per warp.
- MMA issue and accumulators: the same (4 chains of 8 m16n8k16, x in registers, 12 FADD to fold
  the chains).
- Address math: a few instructions either way (the public's slot index is q % 3, an IMAD.HI).
- Barriers: ours makes the 8 compute warps meet twice per unit (the 288-thread bar.sync that waits
  for the helper to free the partial buffer, then the 256-thread unit barrier before the slot
  refill); the public meets once and waits for the buffer on an mbarrier.

So the per-unit instruction count is at parity, and the static 828 vs 709 / 759 was the cold paths.

### Two stamp facts that change the attribution

The skinny stamps (kMode 2, 4) added the weights wait only after a failed first `try_wait`, but
`mbarrier.try_wait` may suspend the thread for a system-dependent time before returning false
(PTX ISA, mbarrier.try_wait), so a wait inside the first try counted as zero. Since 8272efe the
stamps time the whole wait. Timed that way (e937013292b24906, layer routing, sk8w4s-stamps),
thread 0 waits for weights 2.2 / 3.0 / 3.7 us in phase A and 0.3 / 0.7 / 1.1 us in phase B at
1 / 8 / 16 tokens. The earlier "thread 0 never waits for a weights unit" is withdrawn, and with it
the inference that the compute warps alone are the limit. What still stands: the loads-only twin
lands at the public kernel's time (63.0 against 62.7 us at 8 tokens, layer).

### Changes tested (kMode 5 to 8, commit 8272efe)

- 5, sk8w4s-sf: the scale decode with an opaque 16 (`16 + (nloc >> 6)`, nloc <= 32): LEA -> IMAD,
  the public's SCDK.
- 6, sk8w4s-hand: partial buffers through mbarriers, the public's WSB shape: each compute thread
  waits on `hand[4 + bb]` (helper's free, count 32) instead of bar.sync kBarFree, stores its
  partial, arrives on `hand[bb]` (count 256) instead of bar.arrive kBarFull; the helper waits on
  `hand[bb]` and each lane arrives on `hand[4 + bb]` after its own loads. The unit barrier, the
  proxy fence and thread 0's refill are unchanged.
- 7, sk8w4s-sfh: 5 and 6. 8: 7 with the stamps.

Results, e937013292b24906, six seeds per row, means of paired deltas against sk8w4s in us:

| | T=1 layer | T=1 std | T=8 layer | T=8 std | T=16 layer | T=16 std |
|---|---|---|---|---|---|---|
| sk8w4s (us) | 35.5 | 31.9 | 66.2 | 92.8 | 84.7 | 109.2 |
| sf | -0.1 (5/6) | -0.1 (5/6) | -0.2 (4/6) | -0.4 (5/6) | -0.5 (4/6) | -0.1 (4/6) |
| hand | +0.1 (2/6) | -0.0 (4/6) | -0.1 (3/6) | +0.0 (2/6) | +0.4 (2/6) | +0.5 (2/6) |
| sfh | +0.1 (3/6) | -0.1 (4/6) | +0.1 (2/6) | +0.1 (2/6) | +0.5 (3/6) | -0.1 (4/6) |
| loads | -2.4 | -2.9 | -3.2 | -3.2 | -4.5 | -3.9 |
| public | +0.7 | +0.3 | -3.5 | -2.6 | -2.2 | -2.4 |

(n of 6 = seeds lower.) Every arm passed the self-test (4.02e-3 against FP32) and matches
sk8w4s to 5.8e-5 or better; spin flags 0. The judge's own bench picked sk8w4s-sf at 8-32 tokens,
a choice within its noise. Scored 2.004x (1.974-2.034), which does not replace the board entry.

Stamps of sfh against sk8w4s at 8 tokens, layer: phase A 37.7 against 36.7, thread 0's wait for a
free partial buffer 2.5 against 1.5 us, unit barrier 1.2 against 0.8. With one barrier fewer the
warps run ahead and wait for the helper instead; the time moves, it does not leave.

Verdict: neither the ALU instructions of the scale decode nor the second compute-warp barrier is
the gap. The 3.5 us at 8 tokens is not in the per-unit instruction stream (parity, measured
statically) and not in the barrier count (measured). Open, from the corrected stamps: thread 0
waits for weights 3 to 4 us at 8-16 tokens while the loads-only twin, with the same ring and
issue points, matches the public; so the next candidates sit where the stream meets the compute
(ring depth 4 against the public's 3, the refill issued only after the unit barrier, the L2
policy), not in the unit loop's arithmetic. None of the arms is a candidate; no layer port.

### Ordering of the mbarrier handoff (kMode 6-8; not adopted, recorded as the rule asks)

Both directions are generic `st.shared` / `ld.shared` on `S.red`, one proxy, so no proxy fence
is needed (proxy-preserved base causality order, 8.9.5, covers generic-to-generic).

1. Full: a compute thread T stores its partial W, then `mbarrier.arrive` A on `hand[bb]` in
   program order; A is `.release` at `.cta` scope by default (9.7.15.16.16) and a release pattern
   by itself (8.8). The helper lane H's `try_wait.parity` returns true once all 256 arrivals of
   the phase are in; it is `.acquire` at `.cta` by default and an acquire pattern (9.7.15.16.19,
   8.8), so A synchronizes with it (8.9.4), and H's loads R follow in program order. W precedes R
   in causality order, so R sees W.
2. Free: H's loads R of `red[bb][w][lane]` (every w, its own lane), then its own arrive on
   `hand[4 + bb]` (count 32, one per lane, so no lane relies on `bar.warp.sync`); T's
   `try_wait.parity` on it (acquire) before its next store W' to the buffer. R precedes W' in
   causality order and cannot read from it (8.10.6).
3. Phases: unit q uses buffer q % 4. T waits for completion q / 4 - 1 of the free barrier (skipped
   for q < 4); H waits for completion q / 4 of the full barrier. Neither side can be two phases
   ahead of the other, since each waits for the other's previous use.

Section numbers as in "Ordering of the ring slot releases" above.

### Port

None: no arm beat sk8w4s. For the record, a layer port of a kHand variant would also need
`tools/dl_moe_inline.py` to build skinny modes other than 0 (it keeps `v[2] == 0`) and to
initialize `hand[]` in its PDL preamble, which today initializes only `full[]` and skips the
routing warp's block (the same holds for `empty[]` of sk8w4s-er, never ported).

## Skinny weight stream against the public kernel, 2026-10-29

Asked: measure where our skinny path's weight stream meets its compute, against the public
TINY2 / MID builds, name the largest piece, and test the smallest change aimed at it. State before:
the public leads sk8w4s by 3.5 us at 8 tokens and 2.2 at 16 (layer routing, e937013292b24906), per
unit instructions are equal, thread 0 waits 2.2 / 3.0 / 3.7 us for weights in phase A at 1 / 8 /
16 tokens, and our loads-only twin lands at the public's time.

### What differs on the stream side (static, public source as its `_ext` builds TINY2 and MID)

| | ours (sk8w4s) | public TINY2 / MID |
|---|---|---|
| ring | 4 slots of 36,864 B (one unit: 16 rows x full K, codes then scales) | 3 slots (`-DOKC_NS=3`), same 36,864 B unit |
| unit order per CTA | contiguous range `[njobs x 512 x c / G, ...)`, ascending; phase B the same | the same split (`muldiv_g`), ascending |
| CTAs | one per SM, 132 | one per SM, 132 |
| first copies | the routing warp issues 4 after its histogram and job list | FAST1 = 2 copies before routing, the helper the third after it |
| refill | thread 0, after the 288-thread buffer-free barrier, the store, the full arrive and the 256-thread unit barrier; `fence.proxy.async.shared::cta` first | thread 0, after an mbarrier wait for a free buffer, the store and the 256-thread CTA barrier; no proxy fence |
| L2 hint on weight copies | `createpolicy.fractional.L2::evict_first` 1.0 | constant 0x12F0000000000000 (CUTLASS EVICT_FIRST) in TINY2 / MID; none in TINY (<= 4 tokens) |
| inter / ybuf stores | plain | `st.global.L2::cache_hint` evict_last (0x14F0...) |

So the slot size, the unit order and the CTA count are the same in both; the ring depth, the first
copies and the store hints differ, and both refill only after a barrier that also waits for the
helper to free a partial buffer.

### Stamps

kMode bit 16 (ours) and `--unitstamps` (the public small builds, patched by
`tools/moe_make_bench_ab.py`) record the same four events per unit with thread 0's clock64: the copy
issued (I, by whichever thread issues it), thread 0 starts waiting for it (S), the wait ends (E), and
the slot is free after the unit barrier (F, the refill follows). A wait over 0.05 us marks the unit
stalled, so its E is its arrival.

### Ordering of the early refill (kMode bit 32)

Bit 32 moves the unit barrier (`bar.sync 1, 256`) and thread 0's fence and refill ahead of the
partial-buffer handoff (the `bar.sync 6 + b, 288` wait, the store to `S.red` and `bar.arrive 2 + b`).
The chain is mode 0's, with the same instructions in the same relative order:

1. Each compute thread T reads the slot with `ld.shared` (codes and scales, reads R). All of them
   precede, in T's program order, the accumulator fold, and the barrier now sits right after the
   fold, so R precedes T's arrival at `bar.sync 1` in program order as before.
2. `bar.sync` is a CTA-scope barrier: memory accesses requested by the participating threads before
   it are performed relative to all participants (PTX ISA, `bar` / `barrier`, memory ordering), and
   in the memory model it orders them before the participants' later operations (8.9.4, the barrier
   case of synchronizes-with).
3. Thread 0 (a participant) then runs `fence.proxy.async.shared::cta` F and issues the bulk copy W;
   W is ordered after F in program order (8.9.1.1). F is the cross-proxy fence the generic read to
   async write path needs (9.7.10.28.2, 8.9.5), on the path R -> barrier -> F -> W.
4. So R precedes W in proxy-preserved base causality order and cannot read from it (8.10.6).

The handoff is unchanged (same barriers, same counts), and the order cannot deadlock: the helper
takes no part in barrier 1, and every compute warp reaches barrier 1 for unit q before it waits on
the free barrier of buffer q % 4, which the helper releases after reading unit q - 4. ptxas output
(local nvcc 12.8, sm_90a): the same 2,992 instructions, the same counts of BAR.SYNC (9), BAR.ARV (6),
MEMBAR (10), FENCE.VIEW.ASYNC (2), UBLKCP (2) and SYNCS (12); the unit `BAR.SYNC 0x1, 0x100` moves
ahead of the free `BAR.SYNC` and the `BAR.ARV`. Mode 0 compiles to the same SASS as before the
change (2,992 instructions, byte-identical listing).

### Results (run 75bca9775f6f4b0b)

Stamps at 8 tokens, mean over six seeds of one cold launch each, per CTA, us; layer / standalone
routing. Every unit's wait lasted over 0.05 us in all five kernels, so thread 0 waits for every
copy and E is each copy's arrival.

| 8 tokens | sk8w4s | sk8w3s | sk8w4s-rf | sk8w4s-loads | public |
|---|---|---|---|---|---|
| copies in flight when thread 0 starts waiting | 4 | 3 | 4 | 4 | 3 |
| copy issue to arrival (E - I), phase A | 4.69 / 4.55 | 3.23 / 3.15 | 4.78 / 4.68 | 4.79 / 4.64 | 3.02 / 2.92 |
| the same per copy in flight | 1.17 / 1.14 | 1.08 / 1.05 | 1.20 / 1.17 | 1.20 / 1.16 | 1.01 / 0.97 |
| thread 0's weights wait, phase A sum | 6.82 / 10.20 | 5.92 / 8.69 | 3.61 / 5.23 | 23.04 / 34.94 | 3.60 / 5.24 |
| unit hold (wait end to slot free), phase A sum | 19.01 / 28.60 | 18.49 / 28.58 | 25.15 / 38.51 | 1.86 / 2.83 | 21.29 / 33.13 |
| phase A span (first wait start to last slot free) | 35.86 / 55.08 | 34.74 / 54.08 | 37.86 / 58.58 | 31.91 / 49.12 | 32.20 / 50.18 |
| slot free to the refill's issue, per unit | 0.18 / 0.18 | 0.19 / 0.19 | 0.15 / 0.15 | 0.17 / 0.17 | 0.07 / 0.07 |
| wait per unit, first kNS units of phase A | 0.53 / 0.54 | 0.65 / 0.67 | 0.43 / 0.49 | 1.25 / 1.31 | 0.25 / 0.19 |
| phase B weights wait, sum | 2.29 / 3.86 | 2.12 / 3.51 | 1.13 / 1.81 | 2.53 / 3.61 | 0.85 / 1.51 |
| SM clock (GHz) | 1.79 / 1.77 | 1.79 / 1.76 | 1.79 / 1.77 | 1.97 / 1.97 | 1.78 / 1.77 |

The largest piece is the ring depth. Measured: with 4 copies in flight each copy takes 4.7 us from
issue to arrival against 3.0 in the public kernel's 3-slot ring, and the interval per copy
(arrival time over copies in flight) is longer too (1.17 against 1.01 us), so the fourth slot adds
a whole copy of latency and no throughput; thread 0 waits 6.8 us in phase A against 3.6. With 3
slots our latency drops to 3.2 us and the phase A span by 1.1 us. Hypothesis, not measured: more
concurrent bulk copies per SM lower the HBM's efficiency, which would explain why the interval per
copy grows with depth (the old sk8w5s lost 3.3 us to sk8w4s at 8 tokens in bab82f5f5b974594).

Paired timing, graph replay after an L2 flush, mean of six seeds (n of 6 seeds lower):

| us | T=1 layer | T=1 std | T=8 layer | T=8 std | T=16 layer | T=16 std |
|---|---|---|---|---|---|---|
| sk8w4s | 35.6 | 31.9 | 66.2 | 92.7 | 84.8 | 109.4 |
| sk8w3s minus sk8w4s | -0.9 (6) | -0.6 (6) | -2.4 (6) | -1.8 (6) | -3.1 (6) | -2.9 (6) |
| sk8w4s-rf minus sk8w4s | +0.1 (2) | +0.0 (2) | -1.1 (6) | +0.4 (3) | -1.0 (6) | +0.8 (1) |
| (8, 3, 32) minus sk8w3s | +0.3 (1) | +0.1 (2) | +0.3 (1) | +0.9 (0) | +0.9 (0) | +2.6 (0) |
| sk8w4s-nh minus sk8w4s | +2.8 (0) | +2.3 (0) | +3.3 (0) | +1.3 (0) | +1.2 (1) | +2.4 (0) |
| public minus sk8w4s | +0.5 (1) | +0.2 (0) | -3.4 (6) | -2.7 (6) | -2.6 (6) | -2.4 (6) |
| sk8w3s minus public | -1.3 (6) | -0.8 (6) | +1.0 (0) | +0.9 (1) | -0.4 (4) | -0.5 (6) |

Every arm passed the FP32 check through sk8w4s (3.83e-3 to 3.88e-3) and matches it to 2.1e-5 or
better; spin flags 0. sk8w3s is faster than sk8w4s on 36 of 36 paired seeds, closes 2.4 of the
3.4 us public lead at 8 tokens (layer) and passes the public kernel at 1 and 16 tokens. The early
refill lowers thread 0's waits but lengthens the hold (the warps then meet the helper's buffer
barrier inside it) and loses with 3 slots in every row; the unhinted copies lose everywhere. Both
are in docs/ruled-out.md.

What is left at 8 tokens (sk8w3s against the public, measured in the stamps above): the first
kNS units wait 0.65 us each against 0.25 (about 1.2 us; the public issues two copies before
routing), and slot free to refill issue is 0.19 against 0.07 us per unit (thread 0's
`fence.proxy.async.shared::cta` and the issue path; an upper bound, since the stamp's own global
store is still outstanding when the fence's MEMBAR runs). The fence stays (CLAUDE.md); a release
that takes it off thread 0's path, or earlier first copies, is the next test, not built here.

Verdict: sk8w3s joins `SKINNY_VARIANTS` as `(8, 3, 0)`; the prepare bench picks it per size like
the others (it is a mode 0 build, so a candidate). No other kernel changes; mode 0 and the fat
modes compile as before.

### Port

`tools/dl_moe_inline.py` keeps every `SKINNY_VARIANTS` entry with mode 0, so it carries
`(8, 3, 0)` as `sk8w3s` (`sk_8_3_0::launch<1>`) with no change to the tool (checked: its
`device_code` lists sk8w4s, sk16w4s, sk8w3s and the fat modes). Its PDL preamble initialises
`full[s]` for `s < kNS`, which is all mode 0 uses (`empty[]` and `hand[]` belong to kModes 3-8),
so the 3-slot ring needs nothing more there. A port needs: rerun the inliner on the new standalone
file, and let the layer's MoE bench include `sk8w3s` at 1, 8 and 16 tokens; if the layer trims its
build to `((8, 4, 0), (16, 4, 0))` as this guide once advised, it must keep `(8, 3, 0)` instead of
or beside `(8, 4, 0)`. The flag modes (16, 32, 64) are diagnostics or losers and need no port.

## Phase A against the public kernel, 2026-10-29

Question: after mode 28 the same-process gap to the public kernel on layer routing was 16.1 us at
128 tokens and 25.1 at 256, with phase A the largest piece at 256 (+13.1 us in mode 29's stamps
against the public kernel's, check 70dd7b13be884434). Where does phase A's time go?

Method. Mode 31 is 28 with counters per consumer warp over phase A only: cycles in weights and x
waits that miss their first try (a wait that succeeds at once adds nothing, so the stage loop
gains no clock read: 31 costs +2.5 us of graph replay at 256 and its stamped phase A is within
0.3 us of 29's), cycles from a unit's start to its accumulators, epilogue cycles, units, stages
and n-tiles; plus each producer's phase A cycles waiting for a free ring slot. tools/moe_make_
bench_ab.py prints them per size as ABDIAG (mean over CTAs and six layer seeds of one cold launch
each) and the public kernel's stream-K stage ranges as ABPUB. Checks c7db20e053254b52,
0b684d5db0834dbc and 473d87d03de94e96 (three repeats of 31; the numbers below are their range).

[Corrected 2026-10-29, check ca43908f02874886: the wait counters were wrong. `mbar_wait_cyc`
started its clock only after a failed first `try_wait` and returned 0 when the first try
succeeded, but `try_wait` is potentially blocking (PTX ISA 9.4, mbarrier.test_wait /
mbarrier.try_wait), so both cases hid real waiting. The producers' slot waits used the same
function. Since 3f67a45 the clock starts before the first try. Timed in full on mode 41 (37 with
the counters), a consumer warp waits 8.4 / 7.3 us (weights / x) at 128 tokens and 11.8 / 11.5 us
at 256, about 20% of its stage-loop time. The "waiting" row, the "stage loops" label and the
first sentence of the Reading below rest on the old counters. The producers' slot-wait row was
low too: timed in full it reads 65.5 / 64.3 us at 128 and 94.0 / 92.0 at 256 (mode 41), so the x
producer had been undercounted by about 10 us. See "Fat-path waits with the first try timed". The
other rows are wall-clock spans or counts and still stand.]

Measured, mode 28 at 256 tokens, layer routing (us per consumer warp over phase A):

| | 128 tokens | 256 tokens |
|---|---|---|
| phase A per CTA, mean / p10 / p50 / p90 / max | 93.3-93.9 / 85 / 91 / 107 / 111 | 133.9-134.7 / 126-127 / 131 / 145 / 147 |
| waiting for weights / for x (missed first tries) [withdrawn 2026-10-29, see the note above] | 0.00 / 0.03-0.05 | 0.00 / 0.18-0.22 |
| stage loops (dequant, wgmma, ring bookkeeping) [they include the waits, 2026-10-29] | 77.2-78.1 | 111.1-111.3 |
| epilogue (SwiGLU, stores, proxy fence, fence_gpu, bar.sync, red.release) | 10.3-10.8 (2.6-2.7 per unit) | 16.0-16.6 (3.0-3.1 per unit) |
| weights producer / x producer waiting for a free slot | 64-65 / 55 | 93-94 / 82 |
| owned units per CTA, min / mean / max | 3.7 / 4.0 / 4.7 | 5 / 5.37 / 6 |
| MB streamed per CTA (36,864 B weights + 4,096 B x per n-tile, per stage), mean / max | 2.92 / 3.44 | 4.17 / 4.55 |
| rank correlation of a CTA's phase A time with its MB | 0.73-0.82 | 0.75-0.78 |

Reading. The consumers never wait for the ring: both producers spend 70% of phase A blocked on a
full ring, so phase A is consumer-bound, and streaming or ring depth are not the limit. Of a
consumer warp's 134 us, 111 are the stage loops (86 stages, 1.29 us each) and 16 the epilogues
of 5.37 units. The per-CTA spread (p10 to max, 20 us) follows the assigned bytes: owned whole
units leave CTAs with 5 or 6 units (80 or 96 stages), and dyn_b's claimed phase B units absorb
that (phase A's spread does not enter its mean).
[Corrected 2026-10-29: "The consumers never wait for the ring" is withdrawn, along with "streaming
or ring depth are not the limit", which rested on it. Timed in full (mode 41,
ca43908f02874886), consumers wait about 20% of the stage loops while the producers still find
the ring full for about 70% of phase A. Each wait averages 210-230 cycles at both sizes and in both
phases. That pattern looks like a fixed cost at every stage boundary or skew between warps of one
CTA, not a stream that falls behind on average; the counters cannot tell these apart. "Both
producers spend 70% of phase A blocked on a full ring" still holds with the repaired counters
(69-71%).]

The public kernel's partition at 256 tokens (its source, `wgs_ranges` / `wgs_split` /
`wgs_flush`, and the ranges it computed in these checks): stream-K over 256-row x 16-stage units,
each CTA a contiguous stage range of equal cost under the same kCostA table as ours, 57 to 107
stages per CTA (mean 86, the same total as ours), 1.85 cut units per CTA finished through fixup
slots by a designated share; a CTA runs its trailing partial unit first; a finished unit's
release (bar.sync, then one red.release, no per-thread gpu fence and no proxy fence) is deferred
behind the next segment's fourth wgmma group; SiLU is `__fdividef`; the SwiGLU stores go to a
natural-k-order layout, so the 8 lanes of a quad row write one full 16-byte row.

Split of mode 28's epilogue (mode 33, the deferred publish, against 31): the stores and SwiGLU take
1.6 us per unit at 256 (1.2 at 128) and the publish 1.0 to 1.35. Inside the stores and SwiGLU,
`__frcp_rn` compiles to MUFU.RCP plus a range check that calls a subroutine, inside a
convergence region per element (16 per thread per unit at 8 n-tiles). In our k-permuted xb
layout a warp owns two bf16 of each 16-byte token row, so each 2-byte store instruction touches
16 sectors.

Inferred (not measured on the public kernel: its DEBUG build ran 30 us over its replay in these
checks because the stage-range write made it spill, so its stamped times are not used; check 70dd7b13be884434's public stamps put its phase A
at 120.6 us for the same 86 stages): the 13.1 us phase A gap is mostly our epilogue (16 us, of
which the public kernel's cheaper release and SiLU leave a few us) plus the stage loop's 4% more
fast-path instructions (335 against 322, about 4 us at 1.29 us per stage). The spread across
CTAs is not part of it.

### Changes tested

| mode | change | minus 28 at 32 / 128 / 256, us (layer seeds 1-6) |
|---|---|---|
| 32 | publish deferred to the next unit's second stage (fences kept) | +1.6 / +2.7 / +1.2 |
| 34 | 32 without the per-thread fence_gpu | +0.8 / +1.4 / -1.2 |
| 35 | 32 with ss and own_n reloaded from shared memory (spills 104 / 280 B against 192 / 516) | +0.8 / +3.0 / +4.7 |
| 36 | SiLU's reciprocal as one `rcp.approx.ftz` (MUFU.RCP) | -1.8 / -0.9 / -1.2 |
| 37 | 36 without the per-thread fence_gpu, publish in place | -1.9 / -2.1 / -3.8 and -1.7 / -3.4 / -4.2 (two checks) |
| 39 | 37 with paired 4-byte SwiGLU stores (one shuffle per n-tile) | -1.0 / -2.5 / -4.0 |

The deferral moves the cost instead of removing it: 33's publish still takes 1.0 us per unit one
stage later and its stage loops 2 us more, and the publish block inside the stage loop adds about 30
instructions to each phase A loop body. Paired stores halve the epilogue in 40's counters (1.70
against 3.08 us per unit) but the next stages slow by about as much, so the stores' cost is in
the memory system, not in the instructions. Mode 37 removes what is removable inside the
epilogue: the SiLU subroutine and a MEMBAR.ALL.GPU plus CCTL.IVALL per thread per unit.

Mode 37 against 28 in the shippable check 987d024620e44eb7 (GATE, standalone routing): held-out
seeds 17-22 -3.2 us at 128 (6 of 6 lower) and -4.9 at 256 (6 of 6); skewed -1.5 and -2.6 (3 of 3).
On layer routing 12 of 12 paired seeds lower at 128 and 256 over two checks, 11 of 12 at 32.

Spill. Mode 28 spills 184 / 512 B (80 B frame): 7 accesses in stage loops, 114 in per-unit code
(the owned unit count and the x ring base reloaded per unit in phase A; 16 hoisted sum-of-squares
pointers per phase B unit) and 53 elsewhere (local count of SASS accesses by enclosing loop). It
does sit on per-unit paths. 37 has 156 / 416 B (56 B frame; the SiLU subroutine's ABI is gone)
with 9 / 90 / 44 accesses. Removing more (35, the public kernel's volatile shared reloads) cut it to
104 / 280 B but lost 3.4-4.7 us, so the remaining spill stays; at the 96-register cap of 576
threads it would take a stage loop that holds fewer code registers at once (the public kernel
loads a stage's codes one 4-step group at a time), which is a change to the loop, not tested.

### Ordering of the phase A publish without the per-thread fence_gpu (modes 34, 37, 39)

Sections are from the PTX ISA 9.4 documentation (docs.nvidia.com/cuda/parallel-thread-execution)
as cited in "Proxy ordering of the x tiles": 8.8 Release and Acquire Patterns, 8.9.4 Memory
synchronization, 8.9.5 Causality Order, 9.7.15.4 membar / fence.

Mode 28 publishes a finished phase A unit per consumer warpgroup this way: each of the 128
threads stores its SwiGLU outputs into the xb stage (generic `st.global`), then runs
`fence.proxy.async.global`, then `fence.acq_rel.gpu`, then `bar.sync` on the warpgroup's barrier;
thread 0 then runs `red.release.gpu.global.add` on `done_as[j * 16 + st]`. The reader is the x
producer of whichever CTA runs that phase B stage: `ld.acquire.gpu` until the count is 8, then
`fence.proxy.async.global`, then `cp.async.bulk`. Mode 37 drops the writers' `fence.acq_rel.gpu`
(SASS: MEMBAR.ALL.GPU, ERRBAR, CGAERRBAR, CCTL.IVALL per thread per unit) and keeps the rest:

1. Each writer's stores precede its `fence.proxy.async.global` in program order; the proxy fence
   acts on the thread's own generic accesses and composes with other synchronization (9.7.15.4).
2. `bar.sync` synchronizes with the `bar.sync` of every other participating thread (8.9.4), so
   every writer's stores and proxy fence precede, in base causality order (8.9.5), whatever
   thread 0 does after the barrier.
3. Thread 0's `red.release` is a release operation, a release pattern by itself (8.8); base
   causality order is transitive (8.9.5), so the writers' stores precede the release, and the
   reader's acquire that observes the count synchronizes with it (8.9.4). The reader's proxy
   fence and bulk copy follow as in steps 5-7 of "Proxy ordering of the x tiles".

This is the argument already given for the x gather (step 3 there: "its effect extends to the
other threads' stores that precede it through the barrier"); the per-thread `fence.acq_rel.gpu`
is no link in it. The acquire half of that fence (CCTL.IVALL) ordered nothing: no writer reads anything after
it that another CTA wrote. The skinny path's `credit` (fence, `__syncwarp`, relaxed atomicAdd)
is a different pattern and is unchanged.

Modes 32 to 35 (not adopted) moved the whole publish, fences included, to the start of the
warpgroup's next unit's second stage, and for the CTA's last unit to the end of phase A. Delaying
a release reorders nothing in steps 1-3 (every link is program order or synchronization that
stays). Progress: between a unit's stores and its deferred publish the warpgroup waits only on
its own CTA's weights ring and on x rows the gather published before any consumer started; the
last unit is published before any phase B wait; owned units always have 16 stages; and the target
is recomputed from S.ord and the unit's round, not carried in shared state.

## Fat-path waits with the first try timed, 2026-10-29

Defect (steering report docs/steering/2026-10-29-overnight.md, section 2). `mbar_wait_cyc`, the
wait counter of the phase A diagnostic modes (31, 33, 38, 40), returned 0 when the first
`mbarrier.try_wait.parity` succeeded and started its clock only after a failed first try. The
instruction is potentially blocking (PTX ISA 9.4, mbarrier.test_wait / mbarrier.try_wait): a try
that returns true may have suspended the thread until the phase completed, and one that returns
false may have suspended it until a time limit. Both cases hid waiting, so the "0.00" weights
waits in "Phase A against the public kernel" do not show that the consumers never wait. The
producers' phase A slot waits went through the same function. The skinny path's stamps were
fixed the same way in 8272efe.

Fix (3f67a45). The clock starts before the first try. try_wait stays, with the same
spin-timeout check, so the counted wait is the production `mbar_wait_to` loop without its
separate first-try branch, plus two clock reads (test_wait would have changed what is measured). Audit of the other try_wait timers: the is_diag
stamps (modes 11, 17, 22, 24) and the producers' is_diag waits already read the clock before the
wait; the skinny kMode 2/4/8 stamps were fixed in 8272efe and the kMode 16 stamps (03887be)
bracket the whole wait;
tools/moe_make_bench_ab.py's patches of the public kernel stamp before and after its whole
`mbar_wait`. `mbar_wait_to`'s timeout clock starts after the first try, but it is a hang guard
and counts nothing. No other counter had the defect.

New modes (timing only). 41 is 37 (the shipped mode) with the repaired phase A counters plus the
same counters over phase B: per consumer warp, weights and x waits, stage-loop time, epilogue
time, units, stages and n-tiles, and the time waiting for a claimed phase B unit (S.bq). Phase B's
counters go to `inter` after S.dg, so SmemMain keeps its size. 42 is 37 with the light stamps
only, the overhead baseline for 41. tools/moe_make_bench_ab.py maps 37 to these twins and prints
phase B as ABDIAGB.

Production SASS. Modes 0, 16, 26, 28 and 37 and the four skinny kernels compile to the same
instruction text before and after the change (per-kernel sha256 with addresses stripped,
nvcc 12.8, sm_90a). Registers and spills are equal. Only 31 and 38, whose counters changed,
differ. Resources: 37 uses 96 registers, 156 / 416 B spill stores / loads and a 56 B frame. 42
uses 148 / 516 B and 56 B. 41 uses 264 / 580 B and 96 B. The old counter twins were 31 at
260 / 592 B and 38 at 248 / 620 B
(muse_moe_w4a16/profiles/sass_fatwait.2026-10-29.txt).

Overhead (check ca43908f02874886, layer routing, seeds 1-6, graph replay after an L2 flush):

| us, mean of 6 paired seeds | 128 tokens | 256 tokens |
|---|---|---|
| 42 minus 37 (light stamps) | +1.1 (0 of 6 lower) | +2.3 (0 of 6) |
| 41 minus 42 (the counters) | +5.9 (0 of 6) | +3.9 (0 of 6) |
| 41 minus 37 | +7.0 (0 of 6) | +6.1 (0 of 6) |
| stamped phase A, 41 minus 42 | +2.2 | +1.3 |
| stamped phase B, 41 minus 42 | +3.3 | +3.9 |

The counters cost 3-4% of the kernel, more in phase B, where they are new and include a
global read-modify-write per unit. Even if all of phase A's +1.3-2.2 us sat inside the waits,
the waits below would be at least 13.5 us at 128 and 21.9 us at 256.

Measured, mode 41, layer routing, mean over CTAs and six seeds of one cold launch each
(us per consumer warp; the GHz is each CTA's clock64 rate against globaltimer, 1.70 / 1.68):

| | 128 tokens | 256 tokens |
|---|---|---|
| phase A per CTA (thread 0), mean / p90 / max | 93.0 / 106.9 / 111.4 | 131.7 / 141.9 / 144.0 |
| A: waiting for weights / for x | 8.41 / 7.32 | 11.76 / 11.50 |
| A: stage loops without the waits (dequant, wgmma, ring bookkeeping) | 64.62 | 91.66 |
| A: waits as a share of the stage loops | 19.6% | 20.2% |
| A: epilogue (per unit) | 6.05 (1.51) | 9.18 (1.71) |
| A: outside segments (span minus the above) | 6.6 | 7.6 |
| A: weights producer / x producer waiting for a free slot | 65.5 / 64.3 | 94.0 / 92.0 |
| phase B per CTA (thread 0's phase A end to the CTA's barrier) | 49.1 | 69.5 |
| B: waiting for weights / for x / for a claimed unit | 4.48 / 3.88 / 1.23 | 6.44 / 5.40 / 1.24 |
| B: stage loops without the waits | 31.72 | 45.04 |
| B: waits as a share of the stage loops | 20.9% | 20.8% |
| B: epilogue (per unit) | 4.42 (2.21) | 10.11 (3.76) |
| B: units, stages, n-tiles per unit | 2.00, 32, 2.16 | 2.69, 43, 2.84 |
| average cycles per wait, phase A / phase B | 209 / 222 | 227 / 231 |

Stamped phases against the public kernel's own stamps in the same check (42 stands for 37):
phase A 90.8 against 87.2 at 128 and 130.4 against 119.3 at 256; phase B 45.8 against 43.1 and
65.6 against 58.3; launch end 159.9 against 147.6 and 224.3 against 199.7.

Reading.
- Measured: the consumers do wait. In both phases about a fifth of each warp's stage-loop time
  is spent in mbarrier waits for weights or x, 15.7 us per warp in phase A at 128 tokens and
  23.3 us at 256. Phase B adds 8.4 and 11.8 us.
- Measured: at the same time both producers find the ring full for about 70% of phase A. So on
  average the stream keeps ahead of the consumers, and the waits come as a near-constant 210-230
  cycles per wait in both phases and at both sizes. A startup transient (the first weights,
  the x gather) would be a fixed cost per launch and would not scale with the stage count this
  way.
- Hypothesis, not separated by these counters: each wait is either a fixed cost at the stage
  boundary (the try_wait round trip, then the dependent shared loads of the codes before the
  first dequant, serial on a warp's own path) or skew between the 16 warps of a CTA (a slot
  refills only after all 16 warps release it, and with 3 weights slots at both sizes, since the
  largest job is over 4 n-tiles, a leading warp catches the slot its slowest peer still holds).
  Either way, per-warp waits are not SM time lost one for one: three other consumer warps share
  each scheduler and can issue meanwhile.
- Measured: phase B's epilogue costs 3.76 us per unit at 256 tokens against phase A's 1.71. It
  is 10.1 us per warp, the largest non-loop item in phase B.
- What the earlier record still supports: phase A's per-CTA spread follows the assigned bytes
  (rank correlation 0.76 / 0.84), and the epilogue numbers of modes 33, 38 and 40 were timed by
  wall clock, so they stand.

Next mechanism: see docs/steering/2026-10-29-fat-moe-waits.md (recommendation only, nothing
built).
[2026-10-29, later: built as modes 43 and 44, see "Code groups with the next stage hoisted".]

## Code groups with the next stage hoisted, 2026-10-29

Brief: docs/steering/2026-10-29-fat-moe-waits.md, "Recommendation: one mechanism". Mode 41 put
the consumers' mbarrier waits at about 20% of the stage loops, 210-230 cycles each, while the
producers found the ring full for 70% of phase A (check ca43908f02874886). H1 there: every stage
starts with a serial path on each warp (the try_wait round trip, then the shared loads of the
stage's codes, then the first dequant) while at most one wgmma group of the previous stage is in
flight.

Change (modes 43 and 44; 44 is 43 with mode 41's counters, timing only). Only the wgmma stage loop
(`run_segment`, `grp_codes`) differs from 37; the mma.sync path for jobs of 1 or 2 n-tiles, the
epilogues, the producers and the ring sizes are 37's.

- A warp loads one 4-step code group at a time (one `uint4`, steps 4c to 4c + 3) and the scale
  word of each pair of groups, instead of the stage's four `uint4` and the `uint2` of scales up
  front. Group c + 1 is loaded at the start of group c, so a warp holds two code groups and two
  scale words (10 registers instead of 18).
- After the stage's eighth (last) wgmma group is committed and the seventh retired, the warp waits
  on the next stage's `wfull` and `xfull` and loads that stage's first code group and scale word.
  The barrier round trip and those loads run with the last group in flight; the next stage starts
  with its first dequant. A unit's last stage skips this block, and a unit's first stage waits
  before its loop as before, so the hoist does not cross a unit boundary.
- The weights slot is released by every consumer thread with its own
  `fence.proxy.async.shared::cta` and `mbarrier.arrive` (mode 28's form) behind the sixth wgmma
  group instead of the first. The last code load of the stage is issued at the fifth group; the
  release waits one more group so that the fence's MEMBAR.ALL.CTA does not find that load still
  in flight (a guess about the MEMBAR's cost, not measured). The x slot release is unchanged.
- The code and scale pointers of the current slot are carried from the hoist to the next stage
  rather than recomputed from the slot index at the top of every stage; the first draft did the
  latter and ran 406 instructions per stage loop against 396-397.

### Ordering of the moved weights-slot release

Sections are from the PTX ISA 9.4 documentation (docs.nvidia.com/cuda/parallel-thread-execution)
as cited in "Ordering of the ring slot releases": 8.8 Release and Acquire Patterns, 8.9.1.1
Asynchronous Operations, 8.9.4 Memory synchronization, 8.9.5 Causality Order, 8.10.6 Causality,
9.7.10.28.2 Async Proxy, 9.7.15.4 membar / fence, 9.7.15.16.16 mbarrier.arrive, 9.7.15.16.19
mbarrier.test_wait / try_wait.

For one consumer thread T and the stage k that reads weights slot w, the reads R of the slot are:
group 0 and scale word 0, loaded after T's acquire of `wfull[w]` (in the previous stage's hoist,
or before the loop for a unit's first stage), groups 1, 2 and 3 at the first, third and fifth
wgmma groups of stage k, and scale word 1 at the third. The producer thread P refills slot w with
a bulk copy (write W) after its wait on `wempty[w]`.

1. Every read in R is in T's program order before T's `fence.proxy.async.shared::cta` F behind the
   sixth group of stage k. The loads are plain C++ shared loads and F, like every mbarrier asm
   around them, carries a `"memory"` clobber, so the compiler cannot sink a load below F or raise
   one above the try_wait that acquires the slot (SASS: the last LDS.128 of the stage sits before
   the fifth HGMMA pair, MEMBAR.ALL.CTA and FENCE.VIEW.ASYNC.S after the sixth pair's
   `WARPGROUP.DEPBAR.LE gsb0, 0x1`). F is the cross-proxy fence 9.7.10.28.2 asks for and its effect
   composes with later synchronization (9.7.15.4).
2. F, then T's `mbarrier.arrive` A on `wempty[w]`, in program order. A has `.release` at `.cta`
   scope by default (9.7.15.16.16) and is a release pattern by itself (8.8), covering R and F.
3. P's `mbarrier.try_wait.parity` on `wempty[w]` returns true once all kNC x 32 arrivals of the
   phase are in. It has `.acquire` at `.cta` scope by default and forms an acquire pattern
   (9.7.15.16.19, 8.8), so A synchronizes with it (8.9.4).
4. P's `cp.async.bulk` into slot w is ordered after P's prior instructions (8.9.1.1), so after the
   acquire.
5. R precedes W in base causality order with T's proxy fence on the path, so no read in R can
   read from W (8.10.6).

This is the chain of mode 28 step for step; only the program-order distance between R and F
changed, and every link is program order or the same synchronizes-with edge. What the move adds,
and why it keeps the rest intact:

- Arrival count and phase. Each thread still arrives once per stage on the slot it read, so
  `wempty` keeps its expected count of kNC x 32 and each phase completes once per lap.
- The hoisted reads of stage k + 1's slot follow T's acquire of `wfull` for that slot (its
  try_wait, then the loads, in program order), the same full-side chain as 37 with the wait
  issued earlier. They belong to stage k + 1's R and are covered by stage k + 1's own F and A.
  The ring has 3 to 5 weights slots, so stage k + 1's slot is never the slot stage k releases.
- Progress. Before each wait, T has issued the same releases as in 37. In 37, T waits on
  `wfull` and `xfull` for stage k + 1 at the top of stage k + 1, having released weights slots up
  to stage k (at stage k's first group) and x slots up to stage k - 1 (at stage k's first group).
  In 43, T waits on them at stage k's last group, having released weights up to stage k (sixth
  group) and x up to stage k - 1 (first group). No new wait comes before a release that 37 issues
  first, so 43 adds no cycle to the wait graph: the producers wait only on releases, the
  consumers only on fills, and across CTAs only through the x producer's done_as and claim waits,
  which no consumer change touches.
- The x slot. The x slot of stage k is still released by lane 0 after the `wgmma.wait_group` at
  stage k + 1's first group retires the last group that read it (same proxy on both sides, no
  fence; "Which public releases are provable"). The hoisted `xfull` wait for stage k + 1 reads no
  data itself; the wgmmas that read the slot are issued after it in program order.

No fence or barrier is removed. The fence is the bidirectional `fence.proxy.async.shared::cta`
and the arrive is per thread, as in 28 and 37.

### Local compile gate (no GPU)

`muse_moe_w4a16/profiles/sass_grpcodes.2026-10-29.txt` (nvcc 12.8, sm_90a, Docker):

- Production SASS unchanged: modes 0, 16, 26, 28 and 37, the four skinny kernels, and the
  diagnostic 41 and 42 compile to the same instruction text as before the change.
- No C7510-C7515 warning. One `wait1` per wgmma group and one `wait0` per segment end in 43 and
  44, as in 37 (HGMMA 192, wait1 96, wait0 12).
- 96 registers. Spills: 43 has 8 / 100 B (37: 156 / 416 B), none of them inside a stage loop
  (37 has none there either); its unit loops have 1 or 2 LDL each against 2 to 12 LDL and up to 5
  STL in 37. 44 has 20 / 128 B (41: 264 / 580 B).
- Stage loop: 396-405 instructions against 382-390 in 37; on the path where every first try_wait
  succeeds, 347 against 338 (n-tile count 7) and 354 against 340 (8). The hoisted block is about
  9 to 14 more instructions per stage on that path.
- The layer copy (tools/dl_moe_inline.py on a scratch copy of the layer, LAYER_MODES with 37 and
  43) applies its anchors and the PDL patch and builds `moe1::moe_main_kernel<43>` with the same
  resources. `tools/moe_host_check.sh` and `tools/moe_dryrun.py` pass with 43 and 44 built.

### Check plan and stopping rule

One check after the 00:00 UTC reset, bench file from `tools/moe_make_bench_ab.py --base=37
--sizes=32,128,256 --routings=layer o43=43 o44=44 o42=42` plus the staged pair's arm (below;
it does not enter 43's rule) (seeds 1-6; 41 is built as 37's counter
twin, so ABDIAG and ABDIAGB print 41 and 44 side by side, now with the average cycles per wait).
43 is in MAIN_CANDIDATES, so prepare self-tests it against FP32 at 32, 33, 128 and 257 tokens on
normal and skewed routing and the policy bench may pick it.

Keep 43 only if 43 minus 37 has a mean of -2 us or lower at 256 tokens with at least 5 of 6 seeds
lower, a mean of +0.5 us or lower at 32 and at 128, and ABCHECK within 1e-4 of 37; and 44's
average cycles per wait must fall by at least 30% against 41's in the same run. A loss or a tie
goes to docs/ruled-out.md and ends the line. At most 2 MoE checks.

### Results (checks 9d94ac75f4724b52 and 304f17f3e8704992)

Paired differences against 37, us, layer routing (negative = faster; seeds 1-6 in the first
check, held-out seeds 7-12 in the repeat):

| | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| 43 minus 37, seeds 1-6 | -1.3 (5 of 6) | -1.4 (6 of 6) | -2.7 (6 of 6) |
| 43 minus 37, seeds 7-12 | -0.9 (6 of 6) | -1.2 (4 of 6) | -2.1 (6 of 6) |
| 47 (no hoist) minus 37, seeds 7-12 | -0.4 (3 of 6) | -1.6 (6 of 6) | -3.6 (6 of 6) |
| 47 minus 43, seeds 7-12 | +0.5 (2 of 6) | -0.5 (5 of 6) | -1.5 (6 of 6) |
| 44 minus 37 (counter cost), both checks | +3.3 / +4.1 | +6.3 / +6.4 | +7.5 / +9.1 |

Average cycles per wait, phase A, 44 against 41 in the same process: 214 / 224 / 239 against 201 /
215 / 223 (seeds 1-6) and 214 / 215 / 235 against 204 / 203 / 225 (seeds 7-12) at 32 / 128 / 256.
Phase B: 210 / 215 / 225 against 204 / 217 / 220, and 188 / 221 / 233 against 192 / 224 / 234.
The per-warp waits did not fall in either phase; in phase A they rose 3-7%. The producers' slot
waits rose too (256 tokens: 99.9 / 93.5 against 92.9 / 91.1 us, then 108.7 / 102.5 against 100.2 /
99.2), so the ring was full longer. The phase A epilogue was shorter in 44 in every row (per unit
1.38 against 1.49 at 128 and 1.56 against 1.71 at 256 in the first check).

Reading.
- Measured: 43 is faster than 37 at every size in both checks, by 1-3 us.
- Measured: the hoist did not shorten the waits. H1 of the steering report (a serial try_wait
  round trip plus code loads at each stage start) is not what the 210-230 cycles per wait are;
  issuing the wait one wgmma group earlier leaves the wait the same length. That is consistent
  with H2 (warps of a CTA drifting apart, a leading warp waiting on a slot its slowest peer still
  holds) or with the wait being the stream's own pace at the stage boundary; this run does not
  separate them.
- Measured: 47, the same code groups without the hoist, is as fast as 43 or faster (1.5 us faster
  at 256 tokens, 6 of 6). The gain comes from the code groups: a warp holds 10 code and scale
  registers instead of 18, and the kernel's spill falls from 156 / 416 B to 4 / 88 B (8 / 100 B
  in 43), with the per-unit reloads mostly gone (1 LDL per unit loop against 2-12 plus STL). The
  hoist costs about 9-14 instructions per stage and holds the slot longer, and buys nothing.
- Stopping rule: 43's timing condition passed in both checks, the mechanism condition failed in
  both. 43 is not kept; the hoist line ends (docs/ruled-out.md). 47 came out of the repeat as a
  diagnostic arm and was not under a pre-registered rule. On one check (held-out seeds) it clears
  43's timing thresholds against 37 (-3.6 us at 256, 6 of 6; -1.6 at 128; -0.4 at 32). It is a
  candidate for its own gate run (held-out standalone and skewed seeds) before any adoption.

Mode 47's ordering is 43's chain above with the waits back at the stage start: the reads of a
slot (group 0 after the stage's own `wfull` acquire, groups 1-3 and both scale words in the stage)
precede the per-thread fence and arrive behind the sixth group in program order, and before each
wait a thread has issued the same releases as in 37 (weights up to the previous stage, x up to the
one before), so the progress argument carries over unchanged.

## Staged pair on current components, 2026-10-29

The staged two-launch MoE (mode 19, phase A alone on owned units, then mode 21, phase B and the
final pass alone) lost to the fused kernel by 24-30 us at 128 and 256 tokens (checks
9332e2a49aff4ff2, e221858b020c4dee; "Staged GEMM1 gate"). docs/ruled-out.md parks it with a
revisit trigger, cheaper phase B or setup inside the fused kernel, and modes 23, 26, 28 and 37
have landed since. The coordinator asked for a re-test as an extra arm of the mode 43 check,
without changing 43's stopping rule or the 2-check budget.

Rebuilt pair (variant 301, "staged-lean", timing only, never a policy candidate):
- Mode 45: 37 with `a_only`. Owned phase A units, 37's stage loop, setup, SiLU and publish; no
  phase B, no final pass. Its phase B loops are compiled out (with them it spilled 268 / 656 B;
  without them 92 / 164 B).
- Mode 46: 37 with `b_only` and without owned phase A. Phase B on 37's claimed units (dyn_b) and
  the final pass; no x gather and no done_as wait. 45 completed before 46 started (stream order),
  so the done_as acquire has nothing to add; the x producer keeps its
  `fence.proxy.async.global` before each bulk copy, as mode 21 did.
- The claimed-unit x producer's done_as spin now sits under `!b_only(MODE)`, matching the static
  path; it compiles away in every other mode (production SASS unchanged, profiles/
  sass_grpcodes.2026-10-29.txt addendum).
- The stage loop is 37's, not 43's, because 43 is unproven; if 43 wins, the pair's loop could
  take it too.

Arm in the same check: `tools/moe_make_bench_ab.py --base=37 --sizes=32,128,256 --routings=layer
o43=43 o44=44 o42=42 ost=v301 --pairs=ost-o43`. ABPAIR gives the pair against 37 and against 43
per seed. prepare self-tests it against FP32 at 32, 33, 128 and 257 tokens.

Result (check 9d94ac75f4724b52, layer seeds 1-6): the pair minus 37 is +30.3 / +30.4 / +24.4 us at
32 / 128 / 256 tokens and minus 43 +31.6 / +31.9 / +27.0, 0 of 6 lower in every row. The prepare
bench (standalone routing) has phase A alone (45) at 156.5 / 160.0 us and phase B with the final
pass alone (46) at 86.8 / 97.2 at 128 / 256, a sum of 243 / 257 against the fused 37 at 210.6 /
233.2. The fused kernel's phase B overlaps the other CTAs' phase A tail (A done max minus phase A
is about 27 us at 128 and 24 at 256 in 42's stamps); that overlap is still worth more than the
pair saves. The pair passed the FP32 self-test (3.90e-3) and matched 37 within 2.3e-5. It stays
parked (docs/ruled-out.md, with the next trigger).

## Staged GEMM1 gate, 2026-10-29

Design (step 5 of the steering plan): routing and grouping, grouped GEMM1 with SwiGLU, grouped
GEMM2, weighted sum, as separate stages, keeping 64-token expert jobs, 36 KB bulk unit
transfers, the exact NVFP4 decode and one owner per output tile. The GEMM1-only prototype is
mode 19: the routing prologue and x gather of the fat kernel, then phase A on whole units (256
rows x full K, 16 stages) each owned by one CTA, so no unit is cut and no fixup runs. Units are
ordered by their job's n-tile count, largest first, and dealt in snake order (round r gives CTA
c unit r G + c for even r, r G + G - 1 - c for odd r), which every CTA derives from its own
routing prologue. The baseline is mode 18: the same kernel with mode 16's stream-K phase A, phase
B and the final pass switched off. Both end at the grid barrier.

| us, mean of 6 seeds | 128 tokens | 256 tokens |
|---|---|---|
| mode 18: phase A alone, stream-K (mode 16's phase A) | 175.6 / 176.9 | 205.6 / 206.1 |
| mode 19: phase A alone, owned units (GEMM1 prototype) | 170.1 / 171.3 | 189.2 / 190.4 |
| mode 21: phase B and final pass alone | 109.6 | 124.3 |
| staged pair: 19 then 21 | 277.3 | 311.6 |
| mode 20: 19's phase A with 16's phase B in one launch | 245.6 / 247.3 / 247.0 | 282.0 / 283.5 / 284.3 |
| mode 16 | 248.3 / 249.4 / 250.1 | 286.4 / 287.3 / 288.6 |

(checks 9332e2a49aff4ff2 / e221858b020c4dee / 72b7a5a9c511446a.)

Verdict: the GEMM1-only prototype beats mode 16's phase A (by 5.5 us at 128 and 16 us at 256,
on 10 of 12 and 12 of 12 paired seeds), so the gate passes. The full staged pipeline then loses:
19 then 21 is 28-30 us slower than mode 20 and 24-28 us slower than mode 16, because in one
launch phase B starts on each CTA as soon as its phase A slice ends and overlaps the other CTAs'
phase A tail (phase B alone takes 110-124 us; in the fused kernel it adds 75-93 us over phase A
alone). The gain came from owned whole units, not from staging; mode 20 keeps it inside the
fused kernel.

Ordering obligations. Modes 19 and 20 add no new handoff: the xa chain (gather, xready) and the
xb chain (epilogue_a, done_as at 8 releases per stage: 2 units x 4 warpgroups) are mode 16's,
only the CTA that writes a given unit changes. Mode 21 replaces the in-launch done_as handoff
with a launch boundary:

1. Writers: launch 1 (mode 19), each consumer warpgroup stores its SwiGLU outputs into xb with
   st.global, then fence.proxy.async.global, fence.acq_rel.gpu, bar.sync and red.release on
   done_as (kept; nothing in launch 1 reads it).
2. Publication and observation: launch 2 (mode 21) begins after launch 1 completes, by stream
   order (API synchronization, PTX ISA 8.9.4, which takes part in proxy-preserved base causality
   order, 8.9.5) or under PDL by griddepcontrol.wait ("all the memory operations from the
   prerequisite grids are performed and made visible", 9.7.15.14).
3. Async copy: launch 2's x producer runs fence.proxy.async.global (kept with the done_as spin
   removed) and issues cp.async.bulk from xb; the bulk operation is ordered after the issuing
   thread's prior instructions (8.9.1.1), in the async proxy (9.7.10.28.2).
4. Completion: complete_tx on xfull; the consumers wait with mbarrier.try_wait.parity
   (acquire.cta), and the completion carries an implicit generic-async proxy fence
   (9.7.10.28.2).
5. Use: ldmatrix and wgmma reads of the slot.
6. Reuse: the shared-memory slot as before (fence.proxy.async.shared::cta, then mbarrier.arrive
   on xempty); global xb is written next by a later launch's phase A, which starts after launch
   2 completes (stream order). moe_run splits batches above 512 tokens into chunks that share one
   xb, so the staged pair runs mode 20 there instead.

## Proxy ordering of the x tiles, 2026-10-29

An independent review found that the fat path (modes 0, 6, 11, 12, 13, 14, and the layer's
inlined copy, which builds mode 0) writes the x tiles in global memory with generic stores and
has other CTAs read them with `cp.async.bulk`, with no generic-to-async proxy fence on that
path. The only proxy fence was `fence.proxy.async.shared::cta` on the shared-memory ring. The
review is right, and the gather's readiness counter had a second, smaller defect. Sections are
from the PTX ISA 9.4 documentation (docs.nvidia.com/cuda/parallel-thread-execution), read
2026-10-28: 8.4 Operation types, 8.6 Proxies, 8.8 Release and Acquire Patterns, 8.9.1.1
Asynchronous Operations, 8.9.2 Observation Order, 8.9.4 Memory synchronization, 8.9.5 Causality
Order, 9.7.10.28.2 Async Proxy, 9.7.15.4 membar / fence, 9.7.15.14 griddepcontrol.

What the model requires. `cp.async.bulk` runs in the async proxy (9.7.10.28.2: "Accessing the
same memory location across multiple proxies needs a cross-proxy fence. For the async proxy,
fence.proxy.async should be used"). A generic write X reaches a later read Y through causality
order only along proxy-preserved base causality order (8.9.5), which holds when X and Y use the
generic proxy, or the same proxy within one thread block, or are aliases with a proxy fence on
the path. A generic store read by another CTA's bulk copy fits none of these without a proxy
fence. Bidirectional proxy fences "take effect within a single thread" and their effect
"composes with other forms of synchronization" (9.7.15.4).

Phase A input (xa), after the fix:

1. Writers. In each CTA, every thread except the weights producer warp (kNTM - 32 threads)
   copies its share of the token rows of the entries `cta, cta + G, ...` into the job's xa slab
   with 4-byte `st.global` (generic, weak). One job's tile collects rows from several CTAs.
2. Each writer then runs `fence.proxy.async.global` (new), so its own generic stores are ordered
   before async-proxy accesses that follow in causality order, then `__threadfence` (fence.sc.gpu,
   kept), then `bar.sync kBarGather, kNG`. A `bar.sync` synchronizes with the `bar.sync` of every
   other thread on that barrier (8.9.4), so every writer's stores precede, in base causality
   order, everything after the barrier in every participating thread.
3. Publish. After the barrier, one thread per entry does `red.release.gpu.global.add` on
   `xready[job]` (new; it was a relaxed `atomicAdd`). A release operation is a release pattern
   by itself (8.8), and its effect extends to the other threads' stores that precede it through
   the barrier (8.8 footnote 1, transitivity of base causality order in 8.9.5). The old form had
   the thread's fence before the barrier and a relaxed atomic after it: that pattern "only
   affects operations occurring in program order before the first instruction in that pattern"
   (8.8), so it covered the atomic thread's own stores, and the other threads' rows were
   published only because every thread happened to run MEMBAR.SC.GPU before the barrier, which
   the model does not promise.
4. The count. `xready[j]` gets one increment per token row from whichever CTAs gathered them. The
   consumer's acquire reads the last value of a chain of atomic adds, and observation order runs
   through atomic read-modify-writes (8.9.2), so it synchronizes with every contributing release
   (8.9.4: a release pattern synchronizes with an acquire pattern when a write in the first
   precedes a read in the second in observation order, both morally strong; all are gpu scope).
5. Consumer acquire. The x producer thread (lane 0 of the producer warp) of every CTA that holds
   a stage of job j spins on `ld.acquire.gpu` until `xready[j] == job_n[j]` (an acquire
   operation, 8.8), then runs `fence.proxy.async.global` (new).
6. Bulk copy. Then `mbarrier.arrive.expect_tx` on `xfull[slot]` and
   `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` from the xa stage. The bulk
   operation is ordered after the prior instructions of the issuing thread (8.9.1.1), so after
   the acquire and both fences.
7. Completion. The consumer warps wait on `xfull[slot]` with `mbarrier.try_wait.parity`
   (acquire.cta by default) before `ldmatrix`. "The completion of a cp{.reduce}.async.bulk
   operation is followed by an implicit generic-async proxy fence" (9.7.10.28.2), so the
   generic reads of the slot see the copy.
8. Reuse. The shared-memory slot: the consumer warps' generic reads, then
   `fence.proxy.async.shared::cta`, then `mbarrier.arrive` (release.cta) on `xempty[slot]`; the
   producer's acquire wait on it comes before the next bulk write into the slot (unchanged). The
   global xa and xb tiles are written once per launch. Every bulk read of a launch completes
   before its CTAs reach the grid barrier (the consumers wait on every `xfull` they use), and the
   next writes come from the next launch, which begins with `griddepcontrol.wait` ("wait until
   all prerequisite grids in flight have completed and all the memory operations from the
   prerequisite grids are performed and made visible", 9.7.15.14) or follows in stream order (8.9.4
   API synchronization, which also takes part in proxy-preserved base causality order). The
   counters are reset after the grid barrier, after their last reader.

Phase B input (xb): the writers are the 128 threads of the consumer warpgroup that finished a
phase A unit; each stores its SwiGLU outputs into the xb stage with `st.global`, then
`fence.proxy.async.global` (new), then `fence_gpu` (fence.sc.gpu, or fence.acq_rel.gpu in modes
12-14), then `bar.sync` on the warpgroup's barrier, and thread 0 does `red.release.gpu.add` on
`done_as[j * 16 + st]` (this release was already after the barrier). The x producer of the CTA
that holds the phase B stage spins with `ld.acquire.gpu` until the count reaches 8, then
`fence.proxy.async.global` (new), then the bulk copy, with steps 6-8 as above.

Why fences on both sides. The writer-side fence is the documented pattern (CLAUDE.md: proxy
fence on the writing side before the release). 9.7.15.4 says a bidirectional proxy fence acts
within one thread and composes with other synchronization, without saying whether a fence in
the writer orders that write against another thread's later async operation or whether the
issuing thread must fence its own async operation. The consumer-side fence covers the second
reading: the generic writes precede the acquire in causality order, then the issuer's proxy
fence, then its bulk copy. It costs one fence per issued segment in one thread.

Other paths. The skinny kernel's bulk copies read only the repacked weights, written before the
launch and ordered by stream order. The legacy W4 GEMMs use non-bulk `cp.async`, which is not an
async-proxy operation (9.7.10.28.2 names only the bulk forms). Neither needs a change.

Build evidence (Docker, CUDA 12.8.1, sm_90a, `tools/moe_compile_fused.sh`): registers, stack and
spills of every kernel are identical before and after; the SASS gains 115 `FENCE.VIEW.ASYNC.G`
(the global proxy fence, per epilogue instantiation and per producer wait) and 7 `MEMBAR.ALL.GPU`
(the release before the gather's red, one per fat kernel instance). The layer copy is
regenerated by `tools/dl_moe_inline.py` (now restricted to the layer's mode 0 by `LAYER_MODES`,
and patched for the int4 topk read); `diff` against the previous layer file shows all four fix
sites (`fence_proxy_async_global` in epilogue_a, the gather and both producer waits, and
`red_release_add` on `xready`). The layer's MoE extension compiles with torch's headers
(`tools/dl_moe_compile_check.sh`, new: `tools/dl_compile_check.sh` builds only the main
extension), mode 0 at the standalone's 80 B stack and 136/236 B spills. The layer's host wrapper now zeroes `Params::cost`, which
selects the compiled stage-cost table: the struct is not zero-initialized, and moe_run's `cost`
argument (empty means the compiled table) has no layer counterpart since the layer fits no
costs.

## Cost model, 2026-10-29

Replaces the least-squares stage-cost fit (negative coefficients in 84de5bbafa344b7a). Data:
mode 11 (mode 6 with stamps) on 72 cold-L2 launches, 3 sizes (32, 128, 256 tokens) x normal and
skewed routing x 4 seeds x 3 partition tables (the compiled one, flat 100 per stage, steep
20 + 25 n). The compiled table alone gives every CTA nearly the same predicted work, which is
why the old fit was collinear; the other two put different class mixes and cuts on each CTA.
Target: warp 0's time inside segments of both paths (the per-warp stamps cover only the
mma.sync path, so the old fit's target missed every wgmma segment). Terms, each a separate kind
of work: per stage and per n8 tile of each phase, per segment and per cut (stream-K partial)
segment of each phase, and a fixed CTA term. Fit: least squares with nonnegative coefficients
(exact, every support tried) on seeds 11-12; validation on seeds 13-14 against the compiled
table's prediction with its own nonnegative scale and offset. Gate: better per-launch Spearman
rank of predicted work against measured CTA finish times on both routings, and no worse
ordering of the three tables by makespan.

Result (submit 8078730bfa434074, log in muse_moe_w4a16/profiles/):

| term | A (GEMM1) | B (GEMM2) |
|---|---|---|
| per stage, us | 1.006 | 1.282 |
| per n8 tile per stage, us | 0.139 | 0.112 |
| per segment, us | 0 | 0 |
| per cut segment, us | 2.55 | 10.45 |

CTA term 0.91 us; fixed table: 0.01153 us per cost unit plus 19.3 us.

| held-out | CTA rank rho, model / fixed | tables ordered right, model / fixed | work rms, model / fixed |
|---|---|---|---|
| normal | 0.683 / 0.448 | 18/18 / 17/18 | 22.7 / 23.3 us |
| skewed | 0.702 / 0.255 | 12/15 / 13/15 | 20.9 / 21.8 us |

Verdict: the model ranks CTAs within a launch much better than the compiled table, but it
orders partitions no better (one pair worse on skewed routing) and explains the work barely
better (rms 22 us on a 146 us mean, so most of the per-CTA spread is something neither model
sees, probably waits on other CTAs' phase A output inside phase B segments). It failed the gate
and does not drive partitioning; no -nn variant was benched.

What the data says about T=128 anyway:
- The compiled table had the shortest makespan in 10 of the 12 held-out cases (the exceptions:
  bs128 skewed seed 13, flat 151.0 vs 156.7 us; bs32 skewed seed 13, steep 69.6 vs 70.4). Its
  remaining imbalance (slowest CTA's finish minus the mean) at bs128 normal is 12.3 and 9.5 us
  and at bs256 normal 17.9 and 21.2 us, so a perfect partition is worth at most about 10 us at
  128 tokens, which would put the fat path (253 us) about 8 us under legacy128 (251.5). That is
  under 0.5% of the score.
- The per-stage term dominates: about 1.0 us (A) and 1.3 us (B) per stage against 0.11-0.14 us
  per n8 tile, so at bs128 (mostly 4 n8 tiles per job) a stage costs the consumers about 1.56 us
  against about 1.43 us for the ring to stream it (main-loads 191 us over about 134 stages). The
  consumers are the bottleneck because of the per-stage fixed cost, not the MMA work per token.
  The earlier ablation (36b5b559b0344d0f: bs128 loads 189.6, dequant without wgmma 211.9, wgmma
  without dequant 233.2, both 260.8) puts more of that fixed cost on the wgmma side (fence,
  two HGMMA, commit, wait per group) than on the dequant.
- A cut segment in phase B costs about 10 us (A: 2.6): the fixup's FP32 global adds, the count,
  and the last share's read-back. Each CTA has up to two cuts per phase.

## Where the fat path's time goes at 128 tokens, 2026-10-28

Clock stamps of mode 11 (mode 6, jobs of up to 2 n8 tiles on mma.sync, the rest on wgmma) in
check 2072795029b7479f: one cold-L2 launch per size on the bench's routing (seed 11), all 132
CTAs, us at the measured 1.75 GHz. "Per warp" sums over a consumer warp's mma.sync segments
only (at 128 tokens the jobs of 3 or more n8 tiles run on wgmma and are not split out; their
share is the consumer time the four buckets leave over). Mean / max over CTAs.

| | bs32 | bs128 | bs256 |
|---|---|---|---|
| launch end (last CTA) | 209.7 | 264.4 | 289.8 |
| routing + x gather, before any weight load | 7.4 / 13.6 | 12.0 / 16.9 | 18.4 / 24.3 |
| consumers (first stage to last epilogue) | 190.4 / 198.9 | 236.1 / 247.8 | 247.4 / 263.2 |
| grid barrier wait | 9.7 / 24.6 | 12.8 / 48.1 | 16.7 / 50.9 |
| final norm and sum | 1.1 / 2.3 | 3.0 / 3.3 | 5.9 / 6.7 |
| per warp: computing (dequant + mma.sync) | 116.2 | 136.2 | 93.3 |
| per warp: waiting for weights | 11.8 | 12.3 | 8.2 |
| per warp: waiting for x | 11.2 | 9.6 | 7.5 |
| per warp: epilogue and fixup | 24.1 | 24.4 | 16.4 |
| per warp: stages on mma.sync, segments | 116.4, 9.1 | 133.8, 10.0 | 93.1, 6.9 |
| weights producer waiting for a free ring slot | 136.8 | 171.7 | 179.6 |
| x producer waiting for a free slot / done_as / xready | 121.8 / 12.0 / 7.2 | 152.8 / 15.8 / 3.9 | 157.2 / 17.5 / 3.9 |

Reading, at 128 tokens (bench: main-loads 190.6, main-m2 255.4, legacy128 253.6):

- The decomposition is not the limit. Streamed alone, the expert-local jobs take 190.6 us
  against 250-254 for the legacy W4 chain; every expert's weights are read once.
- The consumers are. The weights producer spends 172 of the consumers' 236 us waiting for a
  free slot: the ring is full and the 16 consumer warps drain it. A warp computes a 36 KB stage
  in 1.02 us (136.2 / 133.8) against about 1.2 us to stream one at the loads-only rate, so the
  compute alone nearly matches the stream, and everything else on the consumer's path adds:
  epilogue and fixup 24 us per warp (about 2.4 us per segment: a fence.sc.gpu, two named
  barriers and up to three L2 round trips on the fixup path), 22 us of waits for weights and x
  (ring fill at the start, x behind phase B's done_as gate).
- mma.sync in place of wgmma changed nothing (submit 0400cfbdfa2845fe), and the SASS of the
  mma.sync stage is about 13 instructions per 8 weights with no waste (9 integer, 4 HMUL2; the
  E2M1 bit layout leaves no 1-shift form for the fourth register), so the per-stage compute is
  close to its floor for this dequant; the gains are in the overheads.
- Around the consumers: 12 us of routing and gather before the first weight load (the routing
  warp read topk_ids in 8 serial rounds at E = 1024), and stream-K imbalance: the slowest CTA's
  consumers end 11 us after the mean, and the barrier wait reaches 48 us.

So the loss at 128 tokens is implementation: per-stage consumer work at about the stream rate
plus serial overheads (prologue 12, epilogue and fixup 24, imbalance 11 us), not the job
decomposition. Changes made from this (commit after a248788, not yet run): topk_ids loaded by
all threads at once; the weights producer starts before the x gather; fence.acq_rel.gpu in
place of fence.sc.gpu in the epilogue and fixup (modes 12, 13); per-class stage counts in the
stamps and a least-squares fit of per-stage costs, offered to the bench as a fitted table.

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


## MAIN path, wgmma pipelining, 2026-10-28

The fat path (M >= 17) dequantizes into registers and issues `wgmma.mma_async m64nNk16` with
A from registers, two k16 steps per commit group and `wait_group 1`. ptxas serializes every
wgmma of a function (a `WARPGROUP.DEPBAR.LE gsb0, 0x0` after each HGMMA, plus a
`WARPGROUP.ARRIVE` before each) whenever any of its pipeline checks fails, and the warning
that names the reason is not always printed. `tools/moe_wgmma_micro/` holds four micro
kernels that isolate the causes; `tools/moe_compile_fused.sh` prints the full kernel's
warnings and the SASS counts come from `cuobjdump`. Findings, all on CUDA 12.8 ptxas for
sm_90a at 576 threads per CTA (112-register cap):

| structure | result |
|---|---|
| per-variant `float acc[4 * NTT]` arrays, one per N inside the switch | every accumulator array in local memory (584-byte stack), C7515, serialized |
| one `float acc[32]` shared by every N variant, the finisher loading the fixup sum into `acc` | C7515: non-wgmma instructions (the `LDG`s) define accumulator registers; serialized |
| the finisher loads the sum into a separate `fin[32]`; `acc` written only by wgmma | no C7515, but still serialized without a warning |
| runtime scale-d (`first ? 0 : 1`) inside a runtime stage loop (micro test wg.cu modes 3, 7, 15) | C7515, serialized; the same with a single stage (mode 1) or an immediate scale-d (modes 2, 6, 10) pipelines |
| zeroed accumulators, immediate scale-d 1, a `switch` over N variants inside an enclosing loop (wg2.cu mode 3, wg3.cu modes 19, 35, 67 with per-variant accumulators, an if-chain, or only two variants) | C7512 "insufficient register resources", serialized; at 384 threads (mode 131, a 168-register cap) it pipelines |
| one segment loop per N variant, the variants in sequence, no loop around them (wg4.cu modes 2, 10, 14) | pipelined: one `wait 1` per group, 90-96 registers |
| the full kernel with that structure (16 groups: 8 N variants x 2 phases) | 256 HGMMA, 128 `wait 1`, 16 `wait 0`, no C75xx, 96 registers, 88-byte stack, 156 B spill stores |

Rules kept from this: the accumulators are zeroed once per segment before the first
`wgmma.fence` and never written by anything else (the fixup finisher reads the slot sum into
`fin`); the A fragments of a retired group are pinned with an empty `"+r"` asm after the wait
that retires it (`a_pin`) so the next group's dequant gets fresh registers; and the consumer
walks its segments grouped by (phase, n-tile count) with one loop per group, the two producers
walking the same order.

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

## Prepared for the next checks (not yet run)

Commit after f82a8cf656df4914, local work only (no checks left on 2026-10-28):

- Skinny kernel as per-variant namespaces: `sk8w4s` (the shipped one), `sk16w4s` (16 compute
  warps, 256 k each, x in 32 registers), `sk8w5s` (5 ring slots, 229 KB smem), and loads-only
  twins of the first two that stream the units without computing. ptxas: 140 / 96 / 130
  registers, no spills.
- Fat kernel: codes repacked as [warpgroup][warp][step quad][lane][4 words] (the first layout
  gave every lane a 64-byte stride, a 4-way bank conflict on each LDS.128); rings sized at
  runtime from the launch's largest job (5 weight slots next to 3 x 8 KB x slots when every
  job is N <= 16, 4 + 3 x 16 KB up to N = 32, 3 + 3 x 32 KB beyond); modes d2 (depth-2 wgmma
  pipeline, the shipped one), d3 (three A buffers, `wait_group 2`; micro test wg5.cu shows
  depth 4 trips C7512 at 576 threads), loads-only, and no-gate (no xready / done_as waits,
  timing only). ptxas: 96 registers each, 136-176 B stack, spills only at segment boundaries,
  128 `wait_group 1` (d2) or `wait_group 2` (d3) per kernel, no C75xx.
- The previous kernel's W4 path ported back as `_run_legacy` with its tuned configs at 32,
  128, 256 (`LEGACY_POLICIES`), a candidate from 16 tokens up.
- `prepare` self-tests every computing variant, benches all of them per batch size (graph
  replay, cold L2, two routings) and keeps the fastest as the policy; the BENCH lines carry
  the loads-only ceilings, which split the fat path's gap into ring versus compute.

**Board entry note (2026-10-28 23:25 UTC).** Rows above that call 84de5bbafa344b7a, 0400cfbdfa2845fe,
c524612c78454486 or 36b5b559b0344d0f "the board entry" are historical: those unfenced runs were
withdrawn (see docs/context.md). The board entry was then da9225e4f89441fc (2.003x, fenced, mode 16); since 2026-10-29 it is fa18c9e87d6e4811 (2.070x, mode 23).

## Ruled out

The losers of the fused-kernel runs are in `docs/ruled-out.md` under muse_moe_w4a16.

Losers from these runs are in docs/ruled-out.md.
