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
