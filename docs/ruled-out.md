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
| PDL across our own chain (trigger at entry in every kernel, the projection, attention, combine, router, route_align and boundary launches with the programmatic attribute, weight prologues before `griddepcontrol.wait`) | 179cecdac87b4872 (proj + attention) vs 28e6c2fa656f413a: d1 113.1 -> 115.4 / 124.1 -> 123.6, v1 156.4 -> 156.7 / 170.7 -> 167.5 us; e8b74a3519a94bc6 (glue too) vs 179cecda: every shape 0-3 us better at T <= 32, inside noise. The profile shows the early launches (each proj row grew 0.8-1.6 us with the wait inside it) but the step did not shrink | launch latency as a cost in the captured graph: the graph already hides it; what a boundary still costs (about 0.7 us: the predecessor's completion and memory flush before the wait returns) PDL cannot remove, only fusion can. The code stays as structure (the new boundary and norm kernels are PDL secondaries), not as a win |
| L2 prefetch of the o, router and unpooler-slice weights (34 MB) on the side stream during attention at T <= 8, as `cp.async.bulk.prefetch.L2` with an evict_last policy | 985732949c07474f vs 9cb961d383ab42be: o GEMV row 7.7 -> 7.3 us at decode 1 (bar 3.5), unpooler and router rows unchanged, d1 +0.2 / +2.8 us | the bulk prefetch hint as a way to warm L2 ahead of a GEMV (dropped or too late) |
| The same prefetch as real 16-byte loads with the `L2::cache_hint` evict_last policy | 34568f495b024c2f vs 98573294: o GEMV 7.3 -> 5.9 us (bar 3.5), unpooler 6.0 -> 5.6, router 4.9 -> 5.0; the step rose on all four T <= 8 shapes: d1 +3.5 / +4.1, v1 +3.3 / +1.9 us; attention 14.0 -> 14.8 beside the 9.5 us, 34 MB stream | warming L2 by streaming weights beside the latency-bound attention: the stream costs more than the L2 hits save. A prefetch that pays would have to touch less (the o weights alone, 16.8 MB) or run where HBM is idle; not tried |
| Our hi/lo mma router at 16-32 tokens (`PORT_ROUTER_MMA`) against the one-GEMM pooler + router front | tuner in 34568f495b024c2f: T=16 mma 41.5 vs concat 39.8 us, T=32 42.5 vs 40.1 (both beat the split router: 41.0 / 45.2) | the mma router as the side-chain fix once the norm is off the main chain; the concatenated GEMM wins, and the tuner keeps choosing per run |

## Proposals declined without a run

| Proposal | Reason |
|---|---|
| Moving routing weights before the per-expert RMSNorm | not equivalent: g * RMSNorm(y) != RMSNorm(g * y) |
| Sparse or compressed attention, stale routing or coefficients, token-rounding routers | change the model's math, not its speed |
| Moving normalization across a GEMM | BF16 rounding between them makes it non-equivalent |
| Blackwell-only instructions (tcgen05, TMEM, TMA gather4) | not on H200 |

## Reopened 2026-10-28

Entries above whose mechanism was wrong rather than the idea. Each reopening names the new
mechanism and the evidence, per the rule at the top. Sources: the five component audits and
the public entry vjeux/okc-2026 with its design memos (see `docs/gap-analysis-2026-10-28.md`).

| Entry | New mechanism | New evidence |
|---|---|---|
| Our mma kernel on BF16 layers above 8 tokens; cluster split-K above 8 tokens; larger row groups | swap-AB with x as the mma A operand and 8-row weight tiles as B, K split only across the warps of one CTA and reduced through the idle smem ring, weights by one bulk copy per stage, x by a 3-D TMA box; no cross-CTA reduction at m <= 32. Separately, a TMA + wgmma skinny GEMM with S <= 4 cluster split-K (projection audit item 1) | the ruled-out sweeps' own no-reduce rows capped at cuBLAS level, so the mma.sync + cp.async mainloop was the limit; the public entry's dense block is 51 / 60 us at decode 16 / 32 against our 74 / 77 on cuBLAS |
| Lossless 12-bit BF16 weight compression | escapes applied by 3-4 dedicated warps that take turns on stages, with private accumulators and an uncapped per-CTA table arriving by one bulk copy; rows permuted to balance escapes | the public entry's pooler GEMV at m = 1 takes 13.7 us on 12-bit weights against our 18.2 on BF16, which sits on the measured 4.05 TB/s streaming line; our own compressed kernel only lost on the escape-tile lookup |
| Hot-expert large token tiles (M1) | register-A `wgmma m64nNk16` with the dequantized weights in registers and x through a descriptor, N up to 64 (public kernel); or dequant once per CTA into swizzled smem with cluster multicast of the token slab (MoE audit P1) | the MoE audit's shared-memory-port argument (mma.sync with tokens as B through `ldmatrix` saturates at NT = 8), the skewed-vs-normal rows showing time follows L2-to-SM bytes, and the public MoE at 163 / 198 us on verify 16 / 32 against our 246 / 381 |
| TMA weight loads (M2) | bulk copies of whole 36 KB units into a 3-slot ring consumed through registers by mma.sync warps, in one persistent launch, the first units issued before routing | the public kernel's design; our TMA variant fed an smem pipeline with a barrier round trip per stage |
| Speeding up the router in the layer (hidden under the pooler) | once the pre-expert norm leaves the main chain the side chain is co-critical at T >= 8; at T > 32 run pooler + router hi + router lo as one concatenated cuBLAS GEMM | the layer audit's margin analysis (14 / 7 / 5 us at T = 1 / 8 / >= 16); our router rows at verify 32 are 43 us on the side stream |
| Scratch-buffer split-K, last block reduces (proj v5) | not reopened for projections; the MoE audit's P2 (per-unit last-split SwiGLU) is a different shape: the reduction is per unit and spread over the kernel, with `__ldcg` reads after a fence and an atomic | one launch hop (2-3 us) per MoE call in every profile |
| Fused-shift / bounded-shift softmax (A1) | fixed reference frame for the whole merge: splits add o x 2^m and l x 2^m with FP32 atomics and no running max is exchanged | QK-norm bounds every logit (unit-RMS q and k, scale 5.473 / 8), so 2^s stays inside FP32 range; the public kernel runs every scored shape this way in one launch |
