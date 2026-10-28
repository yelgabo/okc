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
