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
