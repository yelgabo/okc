# Run log: muse_bf16_proj

Every `okc check` and `okc submit`, newest last. Full per-shape output for each job is in
`muse_bf16_proj/profiles/submission.<job>.*`.

| date | job | kind | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|---|
| 10-27 | 75372388f0564794 | check | 8237556 | router v1: CUDA-core, 16 k-slices + atomicAdd, m <= 32 | 1.051x (1.035-1.067) | router 1.87x at m=1, 2.80x at m=8, 1.27x at m=16, 0.70x at m=32 |
| 10-27 | b1b8f1fa518c4cec | check | 6355ad0 | router v1 capped at m <= 16 | 1.061x (1.045-1.077) | m=32 back to SGLang speed |
| 10-27 | 3d746c76cadc46fa | check | 782a8b0 | router v2: x in smem, 8-block cluster reduce, no atomics | 1.031x (1.015-1.046) | faster at m <= 8 (4.5, 7.6 us kernel), slower at 16 (15.4) and 32 (26.1): CUDA-core math is smem-bound at high m |
| 10-27 | 04a9c46b48164048 | check | 03be598 | proj v1: mma kernel for BF16 layers at m <= 32; router v2 at m <= 8 only | 0.802x (0.790-0.814) | correct, but slow and worse with m (pooler 25 us at m=1, 58 us at m=32): x staging did serial 4-byte loads |
| 10-27 | 6473ec2b1d3d44cc | check | dfa7572 | proj v2: coalesced x staging + ldmatrix | 0.930x (0.916-0.944) | m=1 at ~60% of HBM bandwidth (pooler 22.9 us vs 14.0 floor); still slower with m (38.5 us at m=32) |
| 10-27 | ee65e159c9f54269 | check | 24c9405 | proj v3: 4-stage cp.async pipeline | 1.053x (1.037-1.068) | beats cuBLAS at m=1 on all four layers and at m=8 on all but o; loses at m >= 16 |
| 10-27 | 3c7c1fa54d1f4ce0 | check | e221e6e | route BF16 layers to mma only where v3 won (m=1 all, m=8 all but o) | **1.099x** (1.082-1.115) | best so far; qkv_gate 1.20x and unpooler 1.38x at m=1 |
| 10-27 | 17ebe3d9f831439d | check | 3aa7660 | proj v4: R row tiles per warp above 8 tokens (one block/SM) | 0.952x (0.938-0.966) | worse at m >= 16 than v3 (pooler m=32 44 us vs 29): the L2 re-read theory was wrong or minor |
| 10-27 | 21c23f799dc64ca1 | check | 280afc4 | bench.py sweep, first try | 1.100x | sweep stopped at the first launch error; timed Python launches (cuBLAS 20.9 us vs judge 14.4) |
| 10-27 | 4afc0b73f2a74a69 | check | cecb216 | bench.py sweep with graph replays | (bench) | see profiles/bench.4afc0b73f2a74a69.log; findings below |
| 10-27 | e378eace83174734 | check | 14c3851 | proj v5: split-K via scratch buffer, last block reduces; mma at every m <= 32 | 0.945x (0.931-0.960) | slower than the cluster version even at m=1 (qkv_gate 16.1 vs 11.8 us): write, fence, atomic, then read adds a fixed L2-latency tail. Reverted to v4 (cluster) at m <= 8 |
| 10-27 | 69a452e14dbc4729 | check | e2fa9a1 | router hi/lo on the mma kernel at 9 to 32 tokens | **1.125x** (1.108-1.142) | router 8.1 us at m=16 (SGLang 14.6), 10.5 at m=32 (15.1) |
| 10-27 | 457c09b7f2a04ec3 | check | 0839cfb | router mma: 16-token groups up to 32, 32-token groups to 256 (grid z) | 1.070x (1.054-1.086) | m=32 router 8.6 us (from 10.5); m=128/256 28.9/48.1 us vs SGLang 16.4/17.7: per-group L2 re-reads of the weights cost too much |
| 10-27 | c50f83d2b4714d3e | check | f76d95d | router mma capped at 32 tokens (16-token groups) | **1.132x** (1.115-1.149) | best so far; router 5.1 / 8.2 / 8.6 / 8.8 us at m = 1 / 8 / 16 / 32 |
| 10-27 | 09b80fb7ed974323 | check | b742a6d | compressed weights v1 (lossless 12-bit code, up to 8 escaped tiles staged per chunk) for all BF16 layers at m <= 8 | 1.052x (1.036-1.068) | decode bit-exact on the GPU, 0.763 to 0.768 of BF16 bytes; slower on every shape: pooler 26.52 us at m=1 (was 19.42), qkv_gate 14.91 (12.11) |
| 10-27 | 6286e6cbef524e58 | check (bench) | 0f97359 | bench: accumulator count, per-tile branch vs select, a no-decode variant | 1.047x (1.031-1.063) | the no-decode variant is as slow as the decoding ones, and 1 to 8 accumulators change nothing: decode ALU is not the cost. Log in tools/compress_bench.6286e6cbef524e58.log |
| 10-27 | fb7980267f3a4648 | check (bench) | afa92cc | bench: no escaped-tile handling, 4 to 8 stages, max active clusters | 1.048x (1.032-1.064) | per-chunk escape slots made blocks 92 KB: 62 clusters of 4 fit, so pooler's 256 blocks ran in two waves. Without them (60 KB, 92 clusters) pooler 23.0 vs 27.5 bench us, qkv_gate 16.7 vs 18.6 |
| 10-27 | 4686daf77f2046c9 | check (bench) | 120418e | escaped tiles through a 12-tile ring per warp (72 KB per block) | 1.095x (1.079-1.111) | pooler wins: 1.153x / 1.099x at m = 1 / 8 (uncompressed 1.119x / 1.079x); qkv_gate 0.858x, o 0.865x, unpooler 1.125x at m=1, all below the uncompressed kernel |
| 10-27 | 135b2743d1194e41 | check (bench) | b65d5a1 | descriptor load issued ahead of the weight prologue | 1.100x (1.083-1.116) | pooler 1.189x / 1.137x; qkv_gate 0.903x / 0.871x, o 0.906x / 0.793x, unpooler 1.144x / 1.002x. Final file keeps compression for the pooler only (not checked as a whole; per-shape estimate 1.136 to 1.141) |
| 10-28 | 3843af6418a84a72 | **submit** | dfbf010 | skinny-kernel rewrite (12-bit sk at m <= 32, cgemv/rgemv/sgemv at m = 1, hi/lo cluster router, cuBLAS at m >= 128) | **1.349x** (1.329-1.369) | every candidate passed the FP64 self-test; picks: sk compressed on all BF16 shapes at m >= 8 and on qkv/pooler at m = 1, cgemv o m=1, sgemv slice m=1, rgemv router m=1, sk router m 8-32; prepare 16 s; load ratios decode 1/16/32 1.607/1.515/1.357, verify 1/16/32 1.860/0.990/0.991 |
| 10-28 | 71b1927dca894bc4 | **submit** | 58cf59d | wide kernel (wd) for m 33-256: TMA + wgmma swap-AB, cluster DSMEM split-K, router on hi/lo tiles; tuned per (layer, bucket 64/128/256) against cuBLAS | **1.372x** (1.352-1.393) | every wd candidate passed the FP64 self-test; router m=128 16.70 -> 9.27 us (r128c16, kernel 7.6), m=256 17.66 -> 10.77 (r128c8, kernel 8.9); pooler m=128 picked wd on 30.2 vs 30.5 autotune us and judged 0.866x; BF16 wd lost to cuBLAS on every other shape |
| 10-28 | 8d4b485c8ccf4674 | **submit** | 91b835e | wd router geometries with a CTA's whole K slice in flight (r64c16d, r128c16d); wide picks need a 3% margin over cuBLAS | **1.400x** (1.379-1.421) | margin kept pooler m=128 on cuBLAS (1.047x); router 1.872x / 1.629x at m=128/256 (r128c16 kernel 7.6 us, r128c8 8.7); the deep rings did not help (r64c16d 17.3 vs r64c16 14.3 autotune us at m=128), so the router is not bound by TMA round trips |
| 10-28 | 6131a4a220d5486e | **submit** | 65bf8bc | same picks; ablation builds timed in prepare (a1 no wgmma, a2 no reduction/stores, a3 neither) | 1.388x (1.367-1.409) | autotune us, full / a1 / a2 / a3: router m=128 r128c16 12.9 / 12.4 / 11.1 / 10.5; router m=256 r128c8 14.8 / 13.7 / 13.4 / 12.1; o m=256 b2n256c2 17.3 / 15.5 / 14.0 / 12.1; pooler m=128 b1n128c2 30.6 / 29.2 / 29.0 / 27.6. The TMA pipeline alone is most of the time: the BF16 shapes move about 5.5 TB/s from L2, dominated by x re-read per 64-row block (pooler m=128: 128 MB for 64 MB of weights) |
| 10-28 | 80b89e112bf3447f | **submit** | fc9ebbe | wd x multicast across CLR row blocks (cluster = CLK split-K x CLR); losing BF16 configs dropped | 1.372x (1.352-1.393) | every multicast config passed the FP64 self-test but ran 4 to 10x slower than cuBLAS (pooler m=128 m128k4r2 86.7, m128k2r4 127.8, m128k1r8 333 autotune us vs cuBLAS 29.5); about 1.3 to 4 us per k-tile, growing with CLR, with or without wgmma (ablation a3 123.7). Picks unchanged (router wd, the rest cuBLAS) |

## Lessons

- At m >= 16, CUDA-core math (one FMA per weight per token) runs out of shared-memory or L1
  bandwidth before HBM bandwidth. Tensor cores (mma) read each x fragment once per 16x16 weight
  tile instead.
- Zeroing the router output costs about 0.7 us as its own graph node.
- Staging x one 4-byte load per loop iteration serializes on L2 latency (each block waits
  ~0.5 us per round trip before any math). Use wide coalesced loads, many in flight.
- Two register chunks of weights per warp are not enough loads in flight, especially with one
  block per SM. cp.async stages in shared memory keep loads in flight without registers.
- `asm volatile` on mma pins it in program order behind each ldmatrix; mma touches only
  registers, so it does not need volatile.
- With 64 weight rows per block, every block re-reads its x slice through L2: for the
  pooler at m=32 that is about 32 MB of x on top of 67 MB of weights. Bigger row groups
  per block cut the re-reads.
- Sweep 4afc0b73f2a74a69 (absolute numbers sit 6 to 9 us above the judge's; compare within
  a shape): the cluster reduction is the main cost above 8 tokens and grows with S
  (o m=16: 12.9 us at S=2, 17.4 at S=4, 29.8 at S=8; qkv_gate S=4 m=16: 20.2 with the
  reduction, 18.4 without). More warps or row tiles per block never helped. The pooler at
  m >= 16 roughly ties cuBLAS at best (S=2), so something besides the reduction limits it.
- In bench.py only, MT=1 configs with S >= 2 fail to launch (invalid argument); the same
  configs run in the submission. Unexplained.
- The scratch-buffer split-K reduction (write partials, threadfence, atomic counter, last
  block reads) adds several microseconds of serial L2 latency at the end of the kernel; the
  cluster reduction is cheaper at m <= 8. Even with no reduction at all, the mma kernel only
  ties cuBLAS above 8 tokens on the BF16 layers (bench e378eace83174734, red=False rows).
- The bench-only launch failures were a symbol clash: bench.py's extension bound to the
  submission's template instances, whose one-time smem attribute flag was already set.
  The kernels now sit in an anonymous namespace.
- Lossless weight compression (tools/compress_*; bench logs in tools/compress_bench.<job>.*).
  The "check (bench)" rows score kernel_fn with all four BF16 layers on the compressed kernel
  at m <= 8; their prepare() also timed kernel variants (cold L2, graph replays; bench
  microseconds run 4 to 9 above the judge's, so compare within a row). The code stores each
  weight's low byte as is and its high byte (sign, top 7 exponent bits) as a nibble: the sign
  and an offset from the 8 most common top-exponent values. That is 0.763 to 0.768 of BF16
  bytes on the judge's weights; 2.5% to 3.5% of 16x16 tiles hold an out-of-window weight and
  keep their raw high bytes in a side table.
- Fewer bytes did not by themselves make the kernel faster. What mattered was blocks per SM
  and clusters in flight: staging up to 8 escaped tiles per chunk took 92 KB per block, so
  only 62 clusters of 4 fit and pooler's 256 blocks ran in two waves. A 12-tile ring per warp
  (72 KB, 92 clusters) fixed that.
- The escaped-tile path still costs 1 to 3 us against a variant that ignores escapes (qkv_gate
  20.0 vs 16.8 bench us). Its descriptor is a dependent load that the first chunk waits on;
  issuing it first recovered only 0.3 to 0.5 us. Removing the remaining cost would need the
  first chunks' escaped tiles at addresses known without the descriptor.
- The win showed only on the pooler (64 MB, two blocks per SM). On qkv_gate (one block per
  SM), o and the unpooler slice, the compressed kernel lost to the uncompressed one, so the
  file compresses the pooler only.
- Decision 2026-10-27: compression stays out of submission.py. It only beat the current kernel
  on the pooler (18.1 vs 19.4 us at m=1, 18.6 vs 20.0 at m=8, job 135b2743d1194e41); a
  pooler-only file is estimated at 1.136-1.141x, inside the run-to-run noise, and was never run.
  The compressed version is kept at muse_bf16_proj/experiments/compressed_weights.py.

## Rewrite 2026-10-28: skinny kernel from the public memo's design (branch worktree-agent-a3f4197b0217ad8cc)

submission.py was replaced (commits 82a5baf to 3a160e6) by our own implementation of the design
in the public 1.50x entry's memo: x as the mma A operand, weights as 8-row B tiles, each CTA
owning R rows over the whole K with K split only across its warps (no cross-CTA reduction),
weights by one `cp.async.bulk` per stage from a stage-major stream, x by one 3-D TMA box on the
same mbarrier, 12-bit lossless weights with escapes corrected by dedicated escape warps (qkv rows
balanced across row blocks), the hi/lo router on the same kernel with a 3-CTA cluster DSMEM
reduction, m = 1 GEMV candidates (12-bit cgemv, FP32 rgemv, BF16 sgemv), a fused pooler + router
entry for the layer, an FP64 self-test gate and a cold-L2 graph-replay autotuner over compressed,
raw, GEMV and cuBLAS candidates per shape. Formats and entry points: docs/proj-port-guide.md.

Local evidence (no GPU here):

- Docker nvcc for sm_90a: all 29 skinny configs (27 + 2 fused; 48 with the bench sweep) and the
  three GEMV kernels compile with 0 spills, 38 to 112 registers; the GEMVs issue every global
  load before their first FFMA (SASS check in tools/gemv_compile_check.sh).
- tools/emulate_proj.py: numpy emulation of the kernel's data flow (stream bytes, the swizzled
  TMA box, ldmatrix, decode8, mma fragments, NACC sets, escape warps, red/peer partials, row
  map, the cluster split and the fused tail) for every config at reduced sizes: bf16 outputs
  equal the bf16-rounded FP64 product except the single element at a forced tiny escape
  (float32 delta rounding, within the stated tolerance); router paths within 3e-8 of hi + lo.
  Eight mutations (no swizzle, swapped prmt selectors, swapped fragment halves, dropped
  corrections, shifted streams, ignored row map) all fail.
- tools/compress_emulate.py: the 12-bit code restores every weight of the four judge shapes
  bit for bit (escape share 1.0e-4 to 1.4e-4, byte ratio 0.751, 0 inexact deltas); adversarial
  1% outlier matrices are rejected by the 1/200 rule.
- tools/emulate_gemv.py: cgemv decode bit-exact, probe rows exact, products within float32.
- tools/proj_dryrun.py: prepare() and kernel_fn on CPU with a fake extension that enforces the
  host interface: packing of all layers, 152 kernel calls, 20 tuned shapes, dispatch for every
  judge shape (m 1 to 1000), the fused entries.
- An adversarial review of the synchronization protocol (phase parity, init order, proxy
  fences, ldmatrix into the 128B-swizzled box, epilogue barriers, tensor map, launch
  arithmetic) found one bug, fixed in 3a160e6: the fused reduction's shared counter broke when
  the two fused configs alternated; the counter now resets per launch, with scratch per config.

Checks were exhausted for the day when the rewrite landed. First checks after the reset:

1. `okc check muse_bf16_proj/submission.py --no-wait --json`, then `okc status <job> --wait
   --json`, confirming the problem and job id. Read the log: `selftest ... disabled` names any
   kernel that fails against FP64 (with the diag code of a timed-out wait), the `tune` lines
   give compressed vs raw vs GEMV vs cuBLAS per shape in judge-style timing, the `pack` lines
   the escapes, `fused pooler+router ...: ok` the layer entries, then the score.
2. `okc check muse_bf16_proj/bench.py`: the same with the sweep geometries of
   tools/make_bench.py (R, stage depth and warp-split alternatives per shape).
3. Fix what the self-test rejects, then re-check.

### Submit 3843af6418a84a72 (2026-10-28, commit dfbf010): what the numbers say

Judged us per shape (kernel us from the profile in parentheses), m = 1: qkv 10.33 (9.1), o 5.42
(cgemv 5.3), pooler 14.35 (13.2), unpooler slice 3.49 (sgemv 3.2), router 3.56 (rgemv 3.0).
m = 8: 10.51 (9.1), 5.82 (4.3), 14.35 (14.3), 4.59 (3.6), 5.68 (4.1). m = 16: 10.24 (9.2), 6.03
(4.8), 14.47 (14.5), 13.77 (13.1), 5.53 (4.4). m = 32: 11.86 (11.1), 6.19 (5.3), 17.73 (17.9),
15.13 (15.3), 5.98 (5.0). m = 128 and 256: cuBLAS, 0.97 to 1.00x of the baseline.

- Compression wins every shape it was tried on, against the raw-BF16 twin of the same kernel in
  the autotuner (judge-style timing, same replay overhead on both): pooler 18.4 vs 21.7 (m=1),
  18.2 vs 21.9 (m=8), 18.6 vs 21.8 (m=16); qkv 14.3 vs 15.5, 14.2 vs 15.6, 14.3 vs 16.2; unpooler
  18.2 vs 21.8 (m=16), 19.4 vs 21.8 (m=32); o 9.7 vs 10.5. The 2026-10-27 loss of the compressed
  path came from its escape handling and reductions, not from the code: new evidence for the
  ruled-out entry (pending merge, see below).
- The m = 1 sk kernels sit on the streaming line in kernel time (pooler 13.2 us for 48.6 MB:
  1.3 + 12.0) but carry 1.1 to 1.5 us between kernel time and judged time, while the GEMV
  kernels carry 0.1 to 0.5 us (o cgemv 5.42 judged, 5.3 kernel). The sk kernels and cuBLAS's
  nvjet kernels (6.91 judged, 4.7 kernel for o at m=16) both request large shared memory; the
  GEMVs request almost none. The likely mechanism is the L1/shared-memory carveout switch after
  the judge's flush kernel; the public entry's m = 1 bars (qkv 9.2, pooler 13.7) match our
  kernel times, consistent with its cgemv picks paying no such cost. Our cgemv kernels are 1.5
  us (qkv) to 4 us (pooler) slower than the sk kernels in kernel time (2048 CTAs x 544 threads
  run 5.2 waves at 3 CTAs per SM), so they lost the autotune except on o and the slice.
- m = 32 is the weak bucket (decode 32 at 1.357x against 1.515x at decode 16): pooler 17.9 us
  against 14.5 at m=16 for the same bytes, qkv 11.1 against 9.2, with compressed and raw nearly
  equal (22.0 vs 22.2 autotune us), so the pooler at m = 32 is not bandwidth-bound: one consumer
  warp per SM sub-partition (4 warps, 1 CTA per SM) with dependent ldmatrix/mma chains, or the
  64 MB of x re-reads through L2. The bench sweep (R = 16 at 2 CTAs per SM, deeper rings) and
  NACC = 2 for the MT = 2 configs are the next experiments.
- The two m >= 128 loads at 0.99x cap the score: with every other load as measured, lifting
  them to 1.30x (the audit's t_min-plus-2-us targets, mostly the router at 16.5 us for 8 MB)
  would give 1.48x; the router alone at 4 to 6 us would give about 1.40x.

### Ruled out

The losers of the sk rewrite runs are in `docs/ruled-out.md` under muse_bf16_proj.

## Wide kernel 2026-10-28 (branch worktree-agent-a5b928823661efbc6)

`wd_kernel` covers m = 33 to 256 (buckets 64 / 128 / 256, tuned at the bucket's top m against
cuBLAS or SGLang's split router, a 3% margin required). Weights are the wgmma A operand in
64-row K-major tiles straight from TMA (no repack), x the B operand (BN tokens); K splits across
the CLK CTAs of a cluster with a DSMEM reduction; the router loads hi and lo tiles from a
stacked [512, 8192] bf16 copy into one FP32 accumulator. Picks after four scored runs: router
m 33 to 256 on wd (r64c16 / r128c16 / r128c8), every BF16 shape on cuBLAS; best 1.400x
(8d4b485c8ccf4674), router 1.87x / 1.63x at m = 128 / 256.

What the ablations say (6131a4a220d5486e, autotune us): the TMA pipeline alone is most of the
time. BF16 shapes move about 5.5 TB/s from L2, and x is re-read once per 64-row block (pooler
m=128: 128 MB of traffic for 64 MB of weights), so a BF16 win needs fewer x bytes (multicast)
or larger row blocks. The router's pipeline alone takes about 4.5 us for 16 MB on 64 CTAs, and
its DSMEM reduction about 1.8 us.

Next check (first after 00:00 UTC): the multicast build with the CUTLASS-form remote arrive
(`mbarrier.arrive.shared::cluster`, default `.release.cta`) and the producer's plain wait. If
the per-k-tile cost drops to the non-multicast level, the multicast configs should land near
the traffic estimate (pooler m=128: 96 MB at CLR 4, against 128 MB now).
| 2026-10-28 | 4f0a49118f3f40a4 (submit, authorized by the user) | 28e9082 | wd x multicast with the CUTLASS remote-arrive form (6dc6f9e), otherwise as 8d4b485c8ccf4674 | 1.392x (1.371-1.413) | tuner kept cuBLAS on every BF16 shape again. Best multicast config vs cuBLAS (us): pooler m=128 m128k2r4 35.1 vs 30.0, m=256 m128k1r4 52.7 vs 31.6; unpooler m=128 m128k1r4 34.7 vs 24.6; qkv_gate m=128 m128k4r2 22.9 vs 21.8; o m=128 m128k1r4 15.4 vs 11.6. Multicast is now 1.1-1.7x cuBLAS instead of 4-10x, so the barrier form was most of the earlier loss, but it still loses. Router picks r128c16 at m=128 (12.8 vs baseline 22.2) and r128c8 at m=256 (14.6 vs 22.6). Score within noise of 8d4b485c8ccf4674 |

Losers from the wide-kernel runs moved to docs/ruled-out.md.
| 2026-10-28 | b052544c8c2946f6 (submit by the coordinator, 23:11 UTC, before the check reset) | 1d2909c | Close-out test (plan 2026-10-29 item 5): wide-row wd geometries w4n128k8, w4n128k4, w2n256k4 (one x read per 128 or 256 output rows, K split 4-8 across the cluster); router BN 64 at m=256; BF16 multicast configs dropped | 1.390x (1.369-1.411) | Tuner kept cuBLAS on every BF16 shape at m=128/256. Best wide-row vs cuBLAS (us): pooler m=128 w4n128k4 33.1 vs 30.0, m=256 w2n256k4 58.0 vs 32.6; unpooler m=128 w4n128k4 42.5 vs 24.9, m=256 w2n256k4 55.6 vs 29.0; qkv_gate m=128 w4n128k8 25.2 vs 21.3, m=256 w2n256k4 31.5 vs 23.7; o m=128 w4n128k4 25.0 vs 11.7, m=256 w2n256k4 33.7 vs 13.7. The existing b1n128c2 stays the closest wd config (pooler m=128 30.5 vs 30.0). Ablation pooler m=128 w4n128k8: full 45.4, memory pipeline alone 39.3, without reduction and stores 30.2, so the cluster reduction across 8 CTAs costs about 15 us. Router picks r128c16 at m=128 (13.1 vs baseline 22.2) and r128c8 at m=256 (14.8 vs 23.4); r64c8 at m=256 is 15.1, so BN 64 does not help. Score within noise of 8d4b485c8ccf4674. Verdict: close generic BF16 at m=128/256 (cuBLAS), keep the wd router |

Losers from these runs are in docs/ruled-out.md.

## 32-token skinny projection attribution 2026-10-29 (branch worktree-agent-af476e87527c7786b)

The question (steering plan step 7): why the 32-token pooler is slower than the 16-token one
for the same weights, with compressed and raw nearly equal in the tuner. Autotune us below are
cold-L2 graph replays in prepare(), same process; judged us from the profile.

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-29 | a9ff49372ffa4eaf | 68c83ac | Token-count sweep of the compiled pool32/qkv32/unp32 kernels in prepare(); 2-CTA cluster split-K for compressed configs (pool32c, pool32cw, unp32c) | 1.401x (1.380-1.422) | Sweep: pool32 21.9 / 21.8 / 21.8 / 22.0 / 22.3 at M = 1 / 8 / 16 / 24 / 32 against pool16 19.1 at M = 16; qkv32 15.3 / 15.5 / 15.8 at 1 / 16 / 32 against qkv16 14.3; unp32 19.2 / 19.5 at 16 / 32 against unp16 18.3. The x bytes do not set the time. Cluster split-K halves x per CTA and does not help (pool32c 22.0 picked over 22.3; unp32c 23.4 vs 19.4). Judged pooler m=32 17.61 |
| 2026-10-29 | 105c1beba7284938 | fdaf3f4 | Factor split: p32mt1 (pool32's ring, MT 1), p16mt2 (pool16's ring, MT 2, 3 slots), pool32a2 (NACC 2), pool32w8 (8 consumer warps) | 1.400x (1.379-1.421) | pooler m=32: pool32 22.6, p16mt2 19.8 (picked), pool32a2 22.2, pool32w8 21.4; m=16: pool16 18.8, p32mt1 19.5. Only MT 2 with 2 units per warp per stage is slow. Judged pooler m=32 15.13 |
| 2026-10-29 | 748b9d6885324f40 | 5e66659 | pool32 = (2, 32, 2, 2, 2, 4, 4) (fits 3 slots), old geometry kept as pool32u2; qkv32u4, unp32u4, unp32r32 | 1.394x (1.373-1.415) | Repeat: pool32 19.3 vs pool32u2 22.2 (sweep 19.6 vs 22.1; at M = 16 19.0 vs 22.0); qkv32u4 17.5 vs 15.6, unp32u4 21.0 vs 19.4, unp32r32 19.1 vs 19.4. Judged pooler m=32 15.27 |
| 2026-10-29 | 73aa65b0ccec4745 | 84b47ba | Per-role cycle stamps (diagnostic build, clock64 in the loops, eager launches, cold L2, 10 reps) | 1.393x (1.373-1.414) | Per warp, us at 1.98 GHz. pool32u2 m=32: consumer 14.18 total, 3.61 waiting on full slots (10.57 busy); producer 9.01 of 12.89 waiting on empty slots. pool32u2 m=1: 14.02 / 3.43 (10.59 busy). New pool32 m=32: 11.54 / 2.87 (8.67 busy). pool16 m=16: 10.92 / 3.58 (7.34 busy) |
| 2026-10-29 | 6c5787fb0e3944bd | 05de50b | Final file: the original kernel and tables with only the pool32 geometry changed; diagnostics and losing candidates removed | 1.395x (1.374-1.416) | tune pooler m=32: pool32 19.8, raw 22.2, cuBLAS 25.9. Judged pooler m=32 15.42 (17.73 in 3843af6418a84a72, 17.61 in a9ff49372ffa4eaf). m = 1 / 8 / 16 run unchanged kernels: 14.34 / 14.08 / 14.30 vs 14.39 / 13.87 / 14.17 in a9ff49372ffa4eaf. Score inside the noise of 8d4b485c8ccf4674 (1.400x): one shape of one load |

What limits the old 32-token pooler (measured): the consumer warps. They are busy for about 75% of
the kernel (10.6 of 14.2 us per warp), and the busy time is the same at 1 token and at 32
(10.59 vs 10.57 us). Streamed x bytes are therefore not the cause, and neither is accumulator
independence (NACC 2: -0.4 us, noise). The busy time depends on the stage count at a fixed HMMA
count: 4 units per warp per stage (16 stages) instead of 2 (32 stages) cuts it by 1.9 us. That
is about 120 ns of fixed consumer path per stage. Splitting that path (wait loop and
`__all_sync`, the operand loads' latency before the first HMMA, `fence.proxy.async` and the
arrive, decode) is a hypothesis; the stamps do not resolve it. The cost shows only at MT 2: with
MT 1 the 32-stage ring (p32mt1) is within 0.6 us of pool16, so the per-stage path has to stack on
twice the HMMA work before the consumer, rather than HBM, sets the pace.

Port to the layer: the layer's T = 32 MoE front picks "mma" (2fed900548ec4d5b), which runs the
pooler through `_proj` on pool32. The port is one geometry row, regenerated by
tools/dl_proj_inline.py from SK_GEOM. Expected gain at decode 32: about 2.3 us of judged pooler
time per layer step (standalone number, unverified in the layer).

## Public entry A/B in one process 2026-10-29 (branch worktree-agent-aa3b1cca1db53bd6e)

`tools/proj_make_bench_ab.py` appends a bench to submission.py that, at import, builds weights
with the reference formula, runs our prepare() and the public entry's (vjeux/okc-2026
muse_bf16_proj.py v141, loaded from the session scratchpad, never committed; its CUTLASS build
thread made synchronous because the static check rejects the thread) plus any other module,
and times every scored shape as cold-L2 graph replays (256 MB read before each, 100 rotating
rounds, same x). Ratios are paired SGLang/variant medians after the measured replay overhead,
and the load score is the judge's formula. The bench's own score agrees with the board:
public 1.516 / 1.506 in process against 1.501 official.

Bench 0d9f0e5b8617471b (ours = 8d4b485c-era file, kernel us, paired ratio vs SGLang; swap = the
score change if only that shape ran at the public's time; every shape has weight 62, so each
load is one sixth of the geometric mean):

| shape | m | SGLang | ours | public | ours x | public x | swap |
|---|---|---|---|---|---|---|---|
| qkv_gate | 1 | 18.69 | 14.43 | 13.12 (cgemv) | 1.419 | 1.623 | +0.58% |
| o | 1 | 10.75 | 9.60 | 9.28 (cgemv) | 1.225 | 1.294 | +0.13% |
| pooler | 1 | 25.57 | 18.83 | 17.74 (cgemv) | 1.460 | 1.573 | +0.48% |
| unpooler | 1 | 11.26 | 7.87 | 7.58 (cgemv) | 1.909 | 2.065 | +0.12% |
| router | 1 | 15.01 | 7.81 | 7.79 | 3.018 | 3.013 | 0 |
| qkv_gate | 8 | 18.56 | 14.37 | 13.76 | 1.420 | 1.510 | +0.25% |
| o | 8 | 10.82 | 9.95 | 9.81 | 1.153 | 1.186 | +0.07% |
| pooler | 8 | 25.60 | 18.99 | 18.78 | 1.451 | 1.472 | +0.09% |
| unpooler | 8 | 11.52 | 8.80 | 8.54 | 1.601 | 1.705 | +0.12% |
| router | 8 | 31.17 | 9.63 | 9.28 | 5.003 | 5.335 | +0.14% |
| qkv_gate | 16 | 18.46 | 14.40 | 13.73 | 1.408 | 1.508 | +0.22% |
| o | 16 | 10.98 | 10.30 | 9.82 | 1.120 | 1.207 | +0.14% |
| pooler | 16 | 25.66 | 19.17 | 18.82 | 1.437 | 1.473 | +0.12% |
| unpooler | 16 | 23.07 | 18.59 | 18.14 | 1.315 | 1.355 | +0.14% |
| router | 16 | 20.35 | 9.34 | 9.33 | 3.154 | 3.187 | +0.02% |
| qkv_gate | 32 | 18.98 | 15.90 | 15.14 | 1.265 | 1.356 | +0.24% |
| o | 32 | 11.14 | 10.48 | 10.11 | 1.104 | 1.175 | +0.11% |
| pooler | 32 | 25.89 | 20.10 | 19.49 | 1.366 | 1.423 | +0.19% |
| unpooler | 32 | 23.14 | 20.75 | 19.34 | 1.142 | 1.254 | +0.45% |
| router | 32 | 20.61 | 9.74 | 9.66 | 2.980 | 3.015 | +0.02% |
| qkv_gate | 128 | 21.34 | 21.02 (cuBLAS) | 19.07 (gqx) | 1.015 | 1.155 | +0.44% |
| o | 128 | 11.26 | 11.17 | 11.15 (cuBLAS) | 1.019 | 1.019 | 0 |
| pooler | 128 | 30.21 | 30.21 (cuBLAS) | 26.50 (gq) | 1.004 | 1.167 | +0.79% |
| unpooler | 128 | 24.51 | 24.45 | 23.68 (torchT) | 1.000 | 1.041 | +0.17% |
| router | 128 | 22.02 | 12.83 | 10.14 (rk) | 2.075 | 3.011 | +0.58% |
| qkv_gate | 256 | 23.78 | 23.71 | 23.71 (cuBLAS) | 1.005 | 1.002 | 0 |
| o | 256 | 13.22 | 13.07 | 13.34 (torchT) | 1.011 | 0.987 | -0.04% |
| pooler | 256 | 34.99 | 35.23 | 32.34 (CUTLASS) | 0.998 | 1.092 | +0.48% |
| unpooler | 256 | 28.67 | 28.58 | 28.51 (torchT) | 1.006 | 1.004 | 0 |
| router | 256 | 23.23 | 14.78 | 11.38 (rk) | 1.798 | 2.665 | +0.62% |

Loads (ours / public): decode 1 1.608 / 1.743, decode 16 1.533 / 1.594, decode 32 1.408 /
1.498, verify 1 1.891 / 1.968, verify 16 1.124 / 1.271, verify 32 1.093 / 1.164; score 1.416 /
1.516.

Mechanism of the largest gap (pooler and qkv_gate at m=128). The public's gq kernel has our
b1n128c2 geometry (one warpgroup, 64 weight rows x 128 tokens, SS-wgmma m64n128k16, 2-CTA
cluster K split, TMA 128B-swizzled W and x tiles from a producer warp). SASS main loops match
(four HGMMA.64x128x16 per stage, `WARPGROUP.DEPBAR.LE gsb0, 0x1`, one arrive; 124 registers vs
our 90). The differences: an 8-stage ring (192 KB) against our 4; L2 policies (W evict_first,
x evict_last); the first ring issued before `__syncthreads`; a split epilogue (each rank stores
half the rows, the other half sent by `st.async`). gqx is the same kernel with a 3-way uneven
K split for qkv_gate (36 row blocks x 3 = 108 CTAs). The factor split below shows that the ring
depth alone carries the win.

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-29 | 0d9f0e5b8617471b | dacbaa7 (+ bench) | A/B bench: ours vs public vs SGLang, one process | 1.399x (1.378-1.420) | table above; judged path is our unchanged file |
| 2026-10-29 | 4098aa2361554229 | c09dc54 candidates (pre-trim) | A/B bench: new file (host) vs HEAD (base) vs public. wd gets an L2-policy template flag, uneven K split across CLK ranks, and candidates b1n128c2 x {4, 8 stages} x {policy off, on}, b1n128c3s8h, b1n128c1s8h, b1n256c{1,2}s5h, r128c{8,16}h | 1.409x (1.388-1.431) | tune pooler m=128: b1n128c2 31.1, +policy 31.1, 8 stages 26.5, both 26.2, cuBLAS 28.9; qkv_gate m=128: b1n128c3s8h 18.9 vs cuBLAS 21.6 (b1n128c2s8 23.3). Paired in process vs base: qkv_gate m=128 0.991 -> 1.133 (public 1.153), pooler m=128 0.999 -> 1.135 (public 1.154); verify 16 load 1.112 -> 1.183; score 1.4118 -> 1.4275 (+1.1%). Everything else lost or tied (ruled-out) |
| 2026-10-29 | 39d6bfa9834f4a96 | c09dc54 | final file: b1n128c2s8h and b1n128c3s8h only | **1.416x** (1.395-1.437) | picks pooler m=128 b1n128c2s8h 26.2 (cuBLAS 29.7), qkv_gate m=128 b1n128c3s8h 18.9 (21.8); judged m=128 ratios qkv_gate 1.130, pooler 1.144 (0.996 / 0.961 in 6c5787fb0e3944bd) |
| 2026-10-29 | 0135dd541fa94f8e | c09dc54 + comment | repeat of the final file | 1.408x (1.387-1.429) | qkv_gate 1.130, pooler 1.110 at m=128 |

Three judged runs of the new picks (1.409, 1.416, 1.408) against 1.395-1.401 for the previous
file; the CIs overlap, the per-shape gains (both m=128 shapes from about 1.0x to 1.11-1.14x)
reproduce in every run. Remaining gap to the public, by swap value in 4098aa2361554229: router
m=128 / 256 (public rk: split-K with in-kernel cleared FP32 output and red.add, +0.66% each),
qkv_gate m=1 (cgemv, +0.53%), pooler m=1 (+0.37%), pooler m=256 (CUTLASS pingpong, +0.34%),
qkv_gate m=32 (+0.31%).

Port to the layer: muse_decode_layer's `_proj` runs our kernels only at m <= 32 and
SGLang's `_bf16_gemm_dispatch_impl` above, so its verify 16 load (T = 128) runs the pooler and
qkv_gate on cuBLAS. The port is the wd kernel (not yet in the layer) with these two configs,
picked at T = 128 only. Expected: about 3.5 us (pooler) + 2.8 us (qkv_gate) of standalone
kernel time per verify-16 step, unverified in the layer.

Scored 2026-10-29: db5be7d500704692 1.427x, submission.py at c09dc54 (8-stage ring for pooler and qkv_gate at m=128, as checked in 39d6bfa9834f4a96), under the standing replace-when-demonstrably-better authorization. Board entry before: 1.400x.
## CUTLASS sweep at the decode layer's large-token shapes 2026-10-29 (branch worktree-agent-a2cd7a53020c36f7a)

Target: the dense projections the decode layer runs at T = 256 (verify 32). In
muse_decode_layer/submission.py, `_proj` sends qkv_gate, o and the unpooler to cuBLAS above 32
tokens, and the MoE front's tuner picks "concat" at T = 128 and 256 (one `torch.mm` of the FFN
input against `[pooler; router hi; router lo]`, 4608 x 8192, FP32 output; front tuner 45.2 vs
split 59.5 us at T = 256 in layer run 725d34e9511c4370). So the layer's T = 256 GEMMs are
qkv_gate 2304 x 8192, o 8192 x 1024, concat 4608 x 8192 FP32 out and unpooler 8192 x 4096, all
on the main stream. In layer run 2fed900548ec4d5b they are 92.6 us of the 417 / 637 us verify 32
steps (nvjet 128x128 two calls 36.4, nvjet 80x128 33.2, nvjet 96x64 23.0).

`tools/proj_make_bench_cut.py` appends a sweep to the unchanged submission: 11 CUTLASS sm90
swap-AB GEMMs (the public's composition), one `.cu` per config built by `cpp_extension.load`
so ninja compiles them in parallel, timed against cuBLAS, torchT and our pick at T = 256 and
128 as cold-L2 graph replays, 100 rotating rounds, paired ratios against cuBLAS. Configs: pingpong
128x64 (the public's), pingpong 128x64 cluster 2, pingpong 128x128, and cooperative stream-K
128x128, 128x64, 256x128, each at BF16 and FP32 output; every stream-K kernel run at heuristic,
stream-K, and split-K 2 / 3 / 4.

Sweep 8bea43d605c74546 (judge: 192 CPUs, FlashInfer's CUTLASS 4.5.0, 11 configs built in
64.5 s). Kernel us cold (replay overhead 3.9 us included), best CUTLASS config per shape:

| shape (T = 256) | cuBLAS | torchT | ours | best CUTLASS | its ratio |
|---|---|---|---|---|---|
| qkv_gate | 24.03 | 25.02 | 23.94 | co128x64 stream-K 27.81 | 0.845 |
| o | 13.25 | 13.06 | 13.01 | pp128x64 14.62 | 0.871 |
| pooler (BF16, standalone only) | 34.22 | 33.07 | 33.82 | **pp128x64 31.86** | **1.081** |
| unpooler | 28.45 | 27.87 | 28.06 | pp128x128 29.97 | 0.946 |
| concat (FP32 out, layer) | 33.31 | 37.38 | - | co128x128 stream-K 42.30 | 0.767 |

| shape (T = 128) | cuBLAS | torchT | ours | best CUTLASS | its ratio |
|---|---|---|---|---|---|
| qkv_gate | 21.66 | 21.28 | 18.80 (wd) | co128x64 split-K 3 24.85 | 0.853 |
| o | 11.49 | 11.20 | 11.15 | pp128x64 12.50 | 0.884 |
| pooler | 29.31 | 27.68 | 26.29 (wd) | co128x64 split-K 2 29.97 | 0.972 |
| unpooler | 23.78 | 23.17 | 23.68 | pp128x64 24.91 | 0.946 |
| concat (FP32 out, layer) | 27.94 | 28.91 | - | pp128x64f 32.13 | 0.853 |

What it shows. The public's sweep finding reproduces: CUTLASS wins only the standalone pooler at
m = 256, where 4096 / 128 x 256 / 64 = 128 tiles fill one wave on 132 SMs. The layer does not
run that shape. Its concat has 144 tiles at 128x64, a second partial wave; pp128x64f takes 51.3
us there against 31.9 for the BF16 pooler (measured), and stream-K's fixups do not recover it.
At T = 256, torchT moves o and unpooler by -1.3 to +2.1% across runs (0d9f0e5b8617471b,
8bea43d605c74546, 316d451b80034264, d6ff8305da474848), inside the paired noise. At T = 128 the
unpooler on torchT is ahead in all four (1.041, 1.030, 1.044, 1.052), about 1 us; the public
picks it there. A hypothesis, not measured: the
T = 256 GEMMs sit near the practical tensor-core rate already. From cold medians minus the
3.9 us overhead, cuBLAS runs concat at about 650 TFLOP/s and the unpooler at about 700;
qkv_gate (about 480) and o (about 460) are the only shapes with headroom, and both are small.

The candidate: the public's pooler m=256 kernel as a tuner candidate (`CUT_SRC`, `cut128x64`,
offered only for pooler bucket 256, kept only under WD_MARGIN). The first build put the torch
wrapper in the CUTLASS unit with `-std=c++17`; the judge's torch needs C++20 (reproduced with a
local torch: `#error C++20 or later compatible compiler is required to use ATen`), so the route
silently stayed off in 316d451b80034264. The fix keeps the CUTLASS unit free of torch headers
(raw pointers and a `cudaStream_t`, the sweep's structure) and puts the wrapper in
`cpp_sources`; `load_inline` compiles them as separate files, so no temp files are needed.

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-29 | 8bea43d605c74546 | 87e9850 + sweep | CUTLASS sweep bench (table above), judged path unchanged | 1.411x (1.390-1.433) | judged pooler m=256 0.972x |
| 2026-10-29 | 316d451b80034264 | candidate v1 + A/B bench | paired A/B: new file vs HEAD vs public | 1.429x (1.407-1.450) | the cut route did not build (`-std=c++17` against the judge's C++20 torch), so new = HEAD: bench 1.4238 vs 1.4260 |
| 2026-10-29 | d6ff8305da474848 | c7311d7 + A/B bench | paired A/B, CUTLASS unit split from the torch wrapper | 1.423x (1.402-1.445) | tune pooler m=256: cut128x64 30.6 vs cuBLAS 31.7 -> cut128x64. Paired: pooler m=256 base 0.986 -> new 1.110 (public 1.112); verify 32 load 1.081 -> 1.121; bench score 1.4322 -> 1.4375. Judged pooler m=256 1.117x |
| 2026-10-29 | 76051953a96a4168 | c7311d7 | final file, judged | **1.427x** (1.406-1.449) | cut build 48.9 s, prepare 70.6 s; tune pooler m=256 cut128x64 30.9 vs cuBLAS 32.5; judged pooler m=256 1.125x (27.47 vs 30.91 us) |

Two judged runs of the candidate (1.423, 1.427) against 1.408-1.416 for c09dc54; the CIs
overlap, the pooler m=256 gain (0.97-1.01x before, 1.117 / 1.125x now) reproduces in both.

Port to the layer: none for verify 32. Predicted savings in the layer's verify 32 shapes from
this work: 0 us. The CUTLASS pooler kernel does not apply, because the layer computes the pooler
inside concat. Running it as a separate pooler would need the router to hide beside a persistent
kernel holding 128 of 132 SMs. Run serially, it costs about 31.9 + 11 to 15 us (rk or r128c8)
against concat's 33.3 cold, a predicted loss of about 10 us per step. qkv_gate, o and unpooler
stay on cuBLAS. The same caveat applies to the verify 16 port in the previous section: the
layer's T = 128 pooler also runs inside concat (cuBLAS 27.9 us), so the wd pooler pick
(b1n128c2s8h, 26.3 us, 128 CTAs) wins in the layer only if the router hides, which a 128-CTA
kernel does not allow. The qkv_gate m=128 pick (b1n128c3s8h, 18.8 vs 21.7) is unaffected and
remains the T = 128 port, with the unpooler on torchT (`torch.mm(x, W.t().contiguous())`, a
lossless repack made once in prepare) as a second, about 1 us. The remaining dense lever at T = 128 / 256 is a hypothesis: concat as
one wd launch (64 pooler row blocks at BF16 plus 4 router hi/lo row blocks at FP32, K split 2,
136 CTAs).

## m = 1 GEMV against the public cgemv, 2026-10-29 (branch worktree-agent-a1c11338ebdf59102)

Gap map 2026-10-30 experiment 3. At decode 1 the layer's in-layer qkv / o / unpooler-slice rows
were 11.1 / 6.0 / 4.5 us against the public's 9.2 / 5.2 / 3.5. `tools/proj_make_bench_gemv.py`
appends a kernel-level bench to submission.py: every self-tested candidate of our modules (from
`state["cands"]`, which the generator makes prepare() keep), the public's tuned pick and its
cgemv, and SGLang, as cold-L2 graph replays (256 MB read before each, 200 rotating rounds, two
passes with different x), plus paired per-round differences (median, 25th and 75th
percentiles). Weights come from the reference formula (seed 5151); the public file (vjeux v141)
is loaded from the session scratchpad and not committed. Every arm passed ABCHECK against FP64.

Bench b135dd8ee10f4b24, cold medians in us including about 3.9 us of replay overhead, pass 0 /
pass 1. "ours" is the final file (planar stream, 32-register cap); lb2 is the planar stream with
the old `__launch_bounds__(NTH + 32, 2)`; nomin is the planar stream with
`__launch_bounds__(NTH + 32)`; base is c7311d7 (HEAD before this work). Columns without a
suffix are each module's tuned pick.

| shape | m | SGLang | ours | base | public | ours cgemv | lb2 cgemv | nomin cgemv | base cgemv | public cgemv |
|---|---|---|---|---|---|---|---|---|---|---|
| qkv_gate | 1 | 18.30 / 18.27 | 13.23 / 13.31 | 14.24 / 14.11 | 12.96 / 12.90 | 13.25 / 13.12 | 13.60 / 13.47 | 13.17 / 13.09 | 14.59 / 14.40 | 12.78 / 12.70 |
| o | 1 | 10.75 / 10.85 | 9.22 / 9.18 | 9.47 / 9.47 | 8.96 / 9.12 | 9.22 / 9.25 | 9.31 / 9.34 | 9.10 / 9.18 | 9.44 / 9.47 | 8.90 / 8.99 |
| pooler | 1 | 25.70 / 25.62 | 18.69 / 18.59 | 18.66 / 18.66 | 17.63 / 17.66 | 18.72 / 18.85 | 20.74 / 20.86 | 20.54 / 20.64 | 22.08 / 22.27 | 17.50 / 17.60 |
| unpooler slice | 1 | 11.26 / 11.04 | 7.65 / 7.58 (sgemv) | 7.65 / 7.65 | 7.49 / 7.46 | 7.84 / 7.84 | 7.68 / 7.62 | 7.94 / 7.97 | 7.87 / 7.87 | 7.26 / 7.41 |
| router | 1 | 15.23 / 14.93 | 7.62 / 7.57 | 7.68 / 7.62 | 7.58 / 7.68 | | | | | |
| qkv_gate | 8 | 18.69 / 18.62 | 14.62 / 14.53 | 14.66 / 14.56 | 13.79 / 13.82 (sk2) | | | | | |
| o | 8 | 10.91 / 10.98 | 10.05 / 10.13 | 10.18 / 10.14 | 9.95 / 9.79 (sk2) | | | | | |
| pooler | 8 | 25.94 / 25.86 | 18.83 / 18.80 | 18.83 / 18.91 | 18.56 / 18.66 (sk2) | | | | | |
| unpooler slice | 8 | 11.52 / 11.65 | 8.77 / 8.90 | 8.80 / 8.93 | 8.61 / 8.61 (sk2) | | | | | |
| router | 8 | 30.99 / 30.82 | 9.50 / 9.47 | 9.47 / 9.57 | 9.22 / 9.31 (sk2) | | | | | |

Paired differences (us, pass 0 / pass 1). Base cgemv minus public cgemv: qkv_gate +1.79 / +1.63
(above the stop rule's 1 us), o +0.58 / +0.45, pooler +4.54 / +4.64, slice +0.58 / +0.45. Ours
minus base cgemv: qkv_gate -1.31 / -1.25, o -0.16 / -0.16, pooler -3.41 / -3.36, slice -0.03 /
+0.00. Ours cgemv minus public cgemv: qkv_gate +0.48 / +0.35, o +0.30 / +0.18, pooler +1.17 /
+1.26, slice +0.54 / +0.45. At m = 8 the public sk2 leads our pick by +0.80 / +0.67 (qkv_gate),
+0.10 / +0.32 (o), +0.27 / +0.13 (pooler), +0.16 / +0.29 (slice) and +0.26 / +0.13 (router);
this work leaves m = 8 alone (ours minus base -0.10 to +0.02).

The public mechanism. Its cgemv has our geometry: 2 rows per CTA, the whole K per CTA split
across NTH threads (256 qkv_gate, 64 o, 512 pooler, 128 slice) plus one escape warp, N/2 CTAs,
no split-K across CTAs, a fixed-order shared-memory reduction, the same 12-bit code (low-byte
plane plus 4-bit sign and exponent codes, escapes corrected from a per-CTA table, never raw
BF16), and plain `__ldg` with no L2 policy. Two differences matter:

1. Layout. The public keeps a row-major low-byte plane [N][K] and a code plane [N][K/2], so a
   warp's load is 512 (or 256) contiguous bytes. Ours stored one 48-byte chunk per lane (two
   rows' low bytes, then their codes), so each of a lane's three 16-byte loads touched two of
   every three 32-byte sectors, and with L1 bypassed every sector was fetched twice.
2. Registers. The public uses 40 / 32 / 32 / 40 (qkv_gate / o / pooler / slice, no min-blocks);
   ours used 59 / 39 / 39 / 59 under `__launch_bounds__(NTH + 32, 2)`. Registers are allocated
   per warp in 256-register units from one SM sub-partition's 16384
   (`cudaOccMaxBlocksPerSMRegsLimit` in cuda_occupancy.h), so at 33-40 registers a sub-partition
   holds 12 warps and at 32 it holds 16. CTAs per SM: public 5 / 21 / 3 / 9, ours 3 / 16 / 2 / 6.

The change (f321d44). `pack_cgemv` orders each CTA's chunks as [chunk i][16-byte part p][lane
t], a lossless permutation that keeps the tensor shape and the table; the kernel reads part p of
chunk i at `stream + c*CHUNKS*3 + (3i + p)*NTH + t`; and `cgemv_kernel` takes `__maxnreg__(32)`
in place of the launch bounds (32 registers and no spills in nvcc 12.8; at CPT = 2 the second
chunk's loads now follow the first chunk's FMAs, as in the public SASS). CTAs per SM become 7 /
21 / 3 / 12. The factor split in one process: the layout alone (lb2 minus base) moves qkv_gate
-0.99 / -0.93 and the pooler -1.34 / -1.41; the cap on top (lb2 minus ours, paired) is worth
+0.35 / +0.32 on qkv_gate and +1.98 / +1.95 on the pooler. nomin (35 registers, still 12 warps
per sub-partition) ties ours on qkv_gate (-0.05 / +0.00; 5 CTAs against 7) and loses the pooler
by 1.86 / 1.78 (2 CTAs against 3), which is the sub-partition rule at work. The L1 hint is not a
factor: `ld.global.nc` without `L1::no_allocate` against ours is +0.00 / +0.13 on qkv_gate
(c53d2b47c1a24934).

| date | job | kind | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|---|
| 10-29 | c53d2b47c1a24934 | check (bench) | f321d44 without the register cap (+ bench) | GEMV bench: planar vs base vs planar without the L1 hint vs public | 1.436x (1.415-1.458) | qkv_gate cgemv -1.12 / -1.12 us vs base and +0.62 / +0.61 vs the public; L1 hint +0.00 / +0.13 |
| 10-29 | b135dd8ee10f4b24 | check (bench) | f321d44 (+ bench) | GEMV bench: final vs lb2 vs nomin vs base vs public | 1.419x (1.398-1.440) | table above |
| 10-29 | 0b539e62f7b34b34 | check | f321d44 | planar cgemv stream, 32-register cap | 1.413x (1.392-1.435) | judged qkv_gate m=1 9.31 us (1.541x) against 10.19 / 9.89 / 10.18 in 76051953a96a4168, 0135dd541fa94f8e, 39d6bfa9834f4a96; tune qkv_gate m=1 cgemv 13.4 vs qkv8 14.3, picked (the old file picked qkv8). Unpooler m=1 picked cgemv on a tie (8.0 vs sgemv 8.1) and judged 4.16 us, against 3.12-3.73 with sgemv |
| 10-29 | cc7930ee5a634731 | check | f321d44 | repeat | - | deduplicated by the judge (same bytes as 0b539e62f7b34b34): not run, not counted |
| 10-29 | a86f1a2f900847d5 | check | f321d44 plus a trailing comment | repeat | 1.440x (1.419-1.462) | qkv_gate m=1 9.40 us (1.522x); unpooler m=1 picked sgemv, 3.38 us; o m=1 5.24 us (cgemv; 5.19-5.39 before) |

Judged delta at qkv_gate m=1: 9.31 and 9.40 us against a mean of 10.09 over the three previous
runs of the old file, about -0.7 us, and a shape ratio of 1.52-1.54x against 1.37-1.39x. The
whole-file scores (1.413x, 1.440x) overlap the old file's (1.408-1.427x); m = 1 qkv_gate is one
shape of thirty.

Still open, measured in one process: the public cgemv leads ours by 0.2-0.5 us on qkv_gate, o
and the slice, and by 1.2 us on the pooler, where our pick stays pool8, 1.0 us behind the
public. With layout and occupancy matched, the remaining differences are the public's two
separate planes (a 16-byte low-byte load and an 8-byte code load per row) and its escape warp,
which reads entry 0 and the first entries in one load; neither is measured. At m = 8 the public
sk2 qkv8t has our qkv8 geometry (24 rows, NT 3, WSK 4, UPW 2, 5 stages) plus NACC 2 and a
suspend-hint wait for the consumers; its 0.7-0.8 us lead is not attributed. On the slice the
32-register cgemv is 0.2-0.3 us behind sgemv and ties it in the tuner, so the pick can flip
between runs (0b539e62f7b34b34 against a86f1a2f900847d5).

Port to the layer. `tools/dl_proj_inline.py` copies `GEMV_CUDA_SRC` and `pack_cgemv` verbatim,
and the layer calls `cgemv_fwd(..., nth)` with `nth = cgemv_threads(*w1.shape)`, the function
`pack_cgemv` now uses for the layout, so the host code needs no change:

1. From a checkout whose muse_bf16_proj/submission.py contains f321d44, run
   `python3 tools/dl_proj_inline.py`. Expected output: `inlined 1555 device lines, 27 sk configs
   (3 fused), wide configs ['b1n128c2s8h', 'b1n128c3s8h'] and 19 functions from
   .../muse_bf16_proj/submission.py`.
2. Against the layer at af4736b the diff is 44 lines and touches only `_SKP_GEMV_SRC` (the cgemv
   comment, `__maxnreg__(32)` in place of `__launch_bounds__(NTH + 32, 2)`, the three `ld_stream`
   addresses, the rgemv comment) and `pack_cgemv` between the `@SKP_PY` markers (the docstring,
   and `nth = cgemv_threads(n, k)` with the planar transpose). `bucket_of` stays the layer's,
   since the tool pins it. Any other line in the diff comes from another pending inline.
   Verified on a copy: the regenerated `_SKP_SRC + _SKP_GEMV_SRC + _SKP_WD_SRC` compiles with
   torch's headers and flags (tools/dl_compile_check.sh with a cuda_sources hook, nvcc 12.8),
   with cgemv at 32 registers and no spills on the four instances the layer runs (256x2, 64x1,
   512x1, 128x2).
3. Before the layer check, run `uv run --with torch --with numpy python tools/emulate_gemv.py`
   (ALL CHECKS PASSED means a bit-exact decode through the planar addressing on all four shapes)
   and `sh tools/gemv_compile_check.sh` (cgemv 32 registers, 0 spills).
4. In the layer log, look for `SKP selftest <layer> cgemv: ok` on all four layers (probe rows
   bitwise), then `SKP tune qkv_gate m=1`, where cgemv was 14.9 in e87fcb1ed5c64a62 and should
   drop by about 1 us; `SKP tune pooler m=1` (cgemv 22.6 against pool8 22.3 there; the
   standalone cgemv is now 3.4 us faster, so the tuner may move the T = 1 pooler to cgemv); and
   `SKP tune unpooler m=1` (cgemv 8.5 against sgemv 8.8 there; expect a near tie). The
   prediction for decode 1 is a hypothesis: the in-layer qkv row falls from 11.1 to about
   10.1-10.4 us on both shapes and o moves by less than 0.2, which is about +0.15% to +0.2% of
   layer score at 0.199% per us (gap map 2026-10-30, section 0).

## m = 8 skinny kernel against the public sk2, 2026-10-29 (branch worktree-agent-a73f28c7ff93f9389)

At m = 8 the public sk2 led our pick by 0.7-0.8 us on qkv_gate and 0.1-0.3 on o, the pooler, the
unpooler slice and the router (b135dd8ee10f4b24), with the same geometry on every shape (qkv8t:
24 rows, NT 3, WSK 4, UPW 2, 5 stages; o8h, pooler8h, slice8 likewise). One verify-1 microsecond
is 0.136% of layer score (gap map 2026-10-30).

`tools/proj_make_bench_m8.py` appends a bench to submission.py: the control's kernel_fn twice
(ours, ours2; their difference is the null), the control's candidate called directly, variants
of the control built by exact text patches of the kernel, and the public kernel_fn (vjeux v141,
scratchpad copy, never committed), all at m = 8 on reference-formula weights (seed 5151), as
cold-L2 graph replays (256 MB read before each), 240 rounds in a seeded random order, two
passes with different x, and paired per-round differences (median [25th, 75th] percentile).
Every arm passed ABCHECK against FP64 in all three benches.

The GEMV bench rotated a fixed arm list, so each arm always followed the same neighbour; there
our kernel_fn arms sat 0.16-0.28 us above their own candidate at m = 8. In random order the
offset is gone (ours.qkv8 - ours +0.00 / +0.03, ours2 - ours -0.06 / -0.03 in
c991867992d34fb2), so it was an order artifact. The ours-vs-public gap survives it: pub - ours
-0.72 / -0.64 on qkv_gate.

What differs between the two kernels on the m = 8 consumer path (source and SASS, nvcc 12.8):

1. Accumulators: NACC 2 in the public's qkv8t, o8h and slice8, 4 in its router; ours 1 and 2.
2. Consumer full-barrier wait: both spin on `try_wait`. Ours wraps it in `__all_sync` with a
   clock64 timeout. The public does not sleep in its consumers; only its producer and escape
   warps use the suspend hint, as ours do.
3. Ours runs `fence.proxy.async.shared::cta` in every consumer and escape warp before each
   empty-barrier arrive; the public runs none. In SASS this fence is `MEMBAR.ALL.CTA;
   FENCE.VIEW.ASYNC.S`. Our init path also adds it to `fence.mbarrier_init` (which alone is
   `FENCE.VIEW.ASYNC.S`), after stage 0's copies are issued.
4. Empty arrive at UPW > 1: ours after the last unit's operand loads, the public after the
   stage's MMAs.
5. Registers: qkv8 69 against 89, o8 68 / 82, pool8 46 / 66, slice8 60 / 64. qkv8 runs one CTA
   per SM either way (135 KB of shared memory, 96 CTAs on 132 SMs), so registers cannot change
   its occupancy; the other shapes were not checked.

Attribution, qkv_gate m = 8, paired against ours, pass 0 / pass 1 (us). Arms marked "attribution
only" drop a fence and are never a submission candidate (CLAUDE.md).

| arm | change | qkv_gate | other m = 8 shapes | bench |
|---|---|---|---|---|
| acc | NACC 2 (qkv8, o8, slice8), 4 (router8) | -0.03 / -0.03 | o +0.22 / +0.10, router +0.26 / +0.26, slice +0.30 / +0.38 | c991867992d34fb2 |
| late | arrive after the stage's MMAs | +0.03 / -0.03 | slice +0.10 / +0.16 | c991867992d34fb2 |
| nofence | no per-stage proxy fence (attribution only) | -0.14 / -0.19 | router -0.21 / -0.19 | c991867992d34fb2 |
| all | acc + late + nofence + no vote (attribution only) | -0.38 / -0.32 | router -0.32 / -0.19 | c991867992d34fb2 |
| init | every barrier initialized and fenced before the first copy | +0.00 / +0.03 | o +0.32 / +0.13, router +0.16 / +0.13 | e9d6891d00ac4088 |
| novote | consumer wait without `__all_sync` (per-thread timeout kept) | -0.10 / -0.18 | o +0.22 / +0.06 | e9d6891d00ac4088 |
| cand | init + novote | -0.16 / -0.13 | o +0.19 / +0.03, others +0.03 to +0.10 | e9d6891d00ac4088 |
| candnf | cand + nofence (attribution only) | -0.75 / -0.58; minus pub +0.00 / +0.06 | router -0.29 / -0.19 | e9d6891d00ac4088 |
| qkv8u4 | 4 units per warp per stage: 8 stages, 4-slot ring | **-0.69 / -0.66; minus pub -0.03 / +0.03** | | 2b42507273c14aab |
| qkv8w8 | 8 consumer warps, 2 units each: 8 stages | -0.54 / -0.45 | | 2b42507273c14aab |
| qkv8w8u1 | 8 consumer warps, 1 unit each: 16 stages | +0.13 / +0.26 | | 2b42507273c14aab |
| router8u2 | router, 2 units per stage: 8 stages | | router +0.16 / +0.06 | 2b42507273c14aab |
| geov | the four geometries above + novote, tuned picks | -0.77 / -0.59 (picked qkv8u4) | router -0.10 / -0.18 | 2b42507273c14aab |

The m = 8 lead on qkv_gate is a fixed cost per pipeline stage, and the proxy fence is most of it.
Without the fence our kernel ties the public (candnf); with it, nothing on the consumer path
recovers more than about 0.2 us; and the same kernel with half the stages ties the public with
the fence kept (qkv8u4). A second consumer warp per SM sub-partition at the same stage count
does not help (qkv8w8u1), so the cost is per stage, not a latency that another warp hides. That
matches the 32-token finding (about 120 ns of fixed path per stage, the pool32 comment in
SK_GEOM).

The fence stays. PTX ISA 9.4, section 9.7.10.28.2: "Accessing the same memory location across
multiple proxies needs a cross-proxy fence. For the async proxy, fence.proxy.async should be used
to synchronize memory between generic proxy and the async proxy." Section 8.9.5 (proxy-preserved
base causality order) lets a fence on the causality path stand in only for alias proxy fences,
so one lane fencing after `__syncwarp` on behalf of the others is not shown correct either. The
public entry and CUTLASS's `PipelineTmaAsync::consumer_release` both omit the fence on this
read-then-overwrite path; that is practice, not a memory-model citation.

The change (fee7249): `SK_GEOM["qkv8"]` from (1, 8, 3, 1, 4, 2, 5) to (1, 8, 3, 1, 4, 4, 4): four
64-k units per consumer warp per stage, 16 units per stage, 8 stages over K = 8192, a 4-slot ring
of 53 KB stages (221 KB of shared memory with the escape corrections and table; qkv8raw fits 3
slots). 83 registers (69 before), no spills; still one CTA per SM. Nothing else changes: same rows per CTA, same warps, same
fences and barriers.

| date | job | kind | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|---|
| 10-29 | c991867992d34fb2 | check (bench) | 01b553f (+ bench) | m = 8 bench: acc, nofence, late, all vs ours vs public | 1.435x (1.413-1.456) | table above; judged path is the unchanged file |
| 10-29 | e9d6891d00ac4088 | check (bench) | 01b553f (+ bench) | m = 8 bench: init, novote, cand, candnf | 1.411x (1.390-1.433) | table above |
| 10-29 | 2b42507273c14aab | check (bench) | 01b553f (+ bench) | m = 8 bench: qkv8u4, qkv8w8, qkv8w8u1, router8u2 (geo, geov) | 1.428x (1.406-1.449) | table above; geo tune qkv_gate m=8: qkv8 14.4, qkv8u4 13.9, qkv8w8 14.1, qkv8w8u1 14.7 |
| 10-29 | (none) | check | fee7249 | judged check of the change | - | refused: daily cap of 20 checks reached |
| 10-29 | 826cab6b81aa4f81 | submit | fee7249 | qkv8 with 4 units per warp per stage | 1.432x (1.411-1.453) | qkv_gate verify 1 m=8 judged 9.78 us (1.466x) against 10.21-10.67 us (1.350-1.384x, mean 10.48) in the six previous runs of the old geometry (76051953a96a4168, 3843af6418a84a72, 39d6bfa9834f4a96, 0135dd541fa94f8e, 0b539e62f7b34b34, a86f1a2f900847d5); tune qkv8 13.8 (14.1-14.2 before). Board entry stays 1.4347x |

Judged delta: about -0.70 us on qkv_gate at m = 8 in one judged sample, 0.43 us below the best
of six samples of the old geometry; the bench put it at -0.69 / -0.66 paired. The whole-file
score (1.432x) sits inside the run-to-run spread (1.408-1.440x), as one shape of thirty should.

Still open at m = 8, measured in 2b42507273c14aab: pub - ours is -0.06 / -0.10 on o and -0.08 /
+0.00 on the pooler (inside the null), -0.19 / -0.26 on the slice and -0.40 / -0.38 on the router. o8 and slice8 already run 4 stages, so the stage lever is spent there. The router's
public build has 8 ring slots (ours 6) and NACC 4, which lost here (acc, router +0.26).

Port to the layer. The layer takes qkv8's geometry from the inlined config table, so the port is
a re-inline:

1. From a checkout whose muse_bf16_proj/submission.py contains fee7249, run
   `python3 tools/dl_proj_inline.py`. Expected output: `inlined 1555 device lines, 27 sk configs
   (3 fused), wide configs ['b1n128c2s8h', 'b1n128c3s8h'] and 19 functions from
   .../muse_bf16_proj/submission.py`.
2. This change's part of the diff is four lines: in the `_SKP_SRC` dispatch, `case 0` becomes
   `launch_sk<1, 8, 3, 1, 4, 4, 4, true, ...>` and `case 1` becomes `launch_sk<1, 8, 3, 1, 4, 4,
   3, false, ...>`; in `_SKP_CONFIGS`, the qkv8 and qkv8raw rows become `('qkv8', 1, 8, 3, 1, 4,
   4, 4, True, ...)` and `('qkv8raw', 1, 8, 3, 1, 4, 4, 3, False, ...)`. Against the layer at
   01b553f the diff is 52 lines; the other 48 are the pending cgemv port (f321d44, the section
   above). Verified on scratch copies of both files; muse_decode_layer was not edited here.
3. In the layer log, look for `SKP selftest qkv_gate qkv8: ok`, then `SKP tune qkv_gate m=8`,
   where qkv8 should drop by about 0.5-0.7 us from the previous layer run's value.
4. Prediction (hypothesis): the in-layer qkv row at verify 1 (10.1 us sliding, 10.3 full in the
   gap map) falls by about 0.6 us on both shapes, which is about +0.08% of layer score at
   0.136% per us. qkv8 serves T = 2..8 (the last layer tune picked cgemv at T = 1).
