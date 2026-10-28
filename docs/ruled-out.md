# Ruled out

Every experiment that lost or was closed, with the evidence and what it rules out. Check here
before proposing an experiment; reopen an entry only with new evidence that changes its
measured ceiling, and say what that evidence is. Full details are in the runs logs named per
entry.

## muse_bf16_proj (`docs/runs.md`)

| Experiment | Evidence | Rules out |
|---|---|---|
| CUDA-core router v2 (x in smem, 8-block cluster reduce) above 8 tokens | job 3d746c76cadc46fa, 1.031x: 15.4 / 26.1 us at m=16 / 32 vs v1's 10.3 / 19.3 | CUDA-core FP32 math at m >= 16: smem/L1-bound, one FMA per weight per token |
| mma projection kernel, serial 4-byte x staging (proj v1) | job 04a9c46b48164048, 0.802x | scattered dependent staging loads; use wide coalesced loads |
| Larger row groups per block above 8 tokens (proj v4, R tiles per warp) | job 17ebe3d9f831439d, 0.952x | L2 re-reads of x as the m >= 16 bottleneck |
| Cluster split-K reduction above 8 tokens | bench 4afc0b73f2a74a69: +2 to +17 us, grows with S | cluster reductions at high split counts |
| Scratch-buffer split-K, last block reduces (proj v5) | job e378eace83174734, 0.945x, slower even at m=1 | global scratch reduction: fixed L2-latency tail |
| Our mma kernel on BF16 layers above 8 tokens, any config | sweeps 4afc0b73f2a74a69, e378eace83174734 | beating cuBLAS there with mma.sync + cp.async; ties at best |
| Router hi/lo mma with token groups at 128 / 256 tokens | job 457c09b7f2a04ec3: 28.9 / 48.1 us vs SGLang 16.4 / 17.7 | per-group re-reads of the 8 MB router weights from L2 |
| Lossless 12-bit BF16 weight compression, decoded in-kernel | job 135b2743d1194e41, 1.100x; only the pooler gained | software weight compression across projections; cost is escape-tile lookup, not decode |

## muse_moe_w4a16 (`docs/runs-moe.md`)

| Experiment | Evidence | Rules out |
|---|---|---|
| Gate/up paired layout + fused SwiGLU | profiles: silu_and_mul 0.9-3.4% of MoE time | SwiGLU fusion as a material win |
| Per-expert tile size (8 vs 16 per expert) | 8 never beat 16 above 12 tokens | per-expert tile selection |
| Skipping Marlin's expert-parallel mode | saved 2-7 us but prepare() 36 -> 244 s | second Marlin build within the 300 s cap |
| W4 GEMM with no K split | 0.55-0.75x Marlin | no-split at decode sizes: too few warps in flight |
| Block-contiguous weight layout, streaming addresses, no L2 prefetch hint, persistent blocks with cross-unit prefetch, cluster finalizer | each moved times about 1 us or less | layout/persistence tweaks to the W4 mainloop |
| TMA weight loads (M2) | job a49578e5c30348bb, 1.691x; lost 3-11% at bs 1-32, won only bs256 GEMM1 (~4%) | more resident warps as the cure for the gap to streaming bandwidth |
| Hot-expert large token tiles (M1) | bench e1619f311d3747e3: 1.5x faster on skewed routing, 2x slower on normal | large tiles on the judge's (normal) routing |

## muse_attention_decode (`docs/runs-attention.md`)

| Experiment | Evidence | Rules out |
|---|---|---|
| FP8 mma.sync Q.K / P.V | SASS: emulated (unpack + f16 HMMA) on H200 | FP8 mma.sync; use f16 or wgmma |
| Fused q_len 8 kernel without warp specialization (six variants) | ~630 us vs FA3 383 at b32 full | per-tile block-barrier designs for verify |
| Per-tile page-table load without prefetch | job 418d8d492c074b88, 0.642x | dependent page-entry loads; prefetch ahead |
| Integer E4M3-to-f16 conversion | 3-5% slower than F2FP | software FP8 conversion |
| Paired tiles in the ws kernel | job 1f4f986279834925, 1.307x; tied in bench, lost 6-13% on sliding in judge | paired-tile loops; route by judge numbers, not bench |
| Eight loader warps | slower than four once math runs | more loader warps |
| Dropping the loader-side fence.proxy.async (bit 2: var 2, 18) | ~3% faster on full verify, but ordering not proven safe | removing fences for speed (CLAUDE.md rule) |
| A2: smem compaction for two blocks per SM | static audit: 106-128 registers x 384 threads exceed 65,536 for two blocks | occupancy via smem alone; registers block first |
| A1: bounded-shift softmax, tau 8 (var 272) | submit 06966cba66924747, 1.315x vs 1.321x; full verify +1.2% / +2.0% | the O-rescale as the FP32 math's cost |
| DS3: cheaper loader index math, page size 1 (var 528) | submit 971e1ef49fc647aa, 1.292x; sliding verify -5%, full verify +6.7% / +11.1% | loader instruction issue as the full-context limit (cause of the full-context slowdown still open) |
| Fused q_len 1 kernel at full decode batch 1 | judge: 29.4 vs 27.5 us | that shape on the fused kernel; stays on FA3 |

## muse_decode_layer (`docs/runs-decode-layer.md`)

| Experiment | Evidence | Rules out |
|---|---|---|
| Tensor-core hi/lo router at 9-32 tokens in the layer | 6ef2d969b6104e10 vs 46dab8fb87274880: decode 16 -0.01 to -0.02 both ways | speeding up the router in the layer: it runs hidden under the pooler |
| X5 boundary, one block per token | job b57b66f36fb044c6, inconclusive; 0.7 us slower than SGLang at T=8 | (superseded by the cluster version; that one is held for BF16 rounding, not ruled out) |

## Proposals declined without a run

| Proposal | Reason |
|---|---|
| Moving routing weights before the per-expert RMSNorm | not equivalent: g * RMSNorm(y) != RMSNorm(g * y) |
| Sparse or compressed attention, stale routing or coefficients, token-rounding routers | change the model's math, not its speed |
| Moving normalization across a GEMM | BF16 rounding between them makes it non-equivalent |
| Blackwell-only instructions (tcgen05, TMEM, TMA gather4) | not on H200 |
