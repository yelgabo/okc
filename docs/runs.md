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
