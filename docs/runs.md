# Run log: muse_bf16_proj

Every `okc check` and `okc submit`, newest last. Full per-shape output for each job is in
`muse_bf16_proj/profiles/submission.<job>.*`.

| date | job | kind | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|---|
| 10-27 | 75372388f0564794 | check | 8237556 | router v1: CUDA-core, 16 k-slices + atomicAdd, m <= 32 | 1.051x (1.035-1.067) | router 1.87x at m=1, 2.80x at m=8, 1.27x at m=16, 0.70x at m=32 |
| 10-27 | b1b8f1fa518c4cec | check | 6355ad0 | router v1 capped at m <= 16 | 1.061x (1.045-1.077) | m=32 back to SGLang speed |
| 10-27 | 3d746c76cadc46fa | check | 782a8b0 | router v2: x in smem, 8-block cluster reduce, no atomics | 1.031x (1.015-1.046) | faster at m <= 8 (4.5, 7.6 us kernel), slower at 16 (15.4) and 32 (26.1): CUDA-core math is smem-bound at high m |
| 10-27 | 04a9c46b48164048 | check | 03be598 | proj v1: mma kernel for BF16 layers at m <= 32; router v2 at m <= 8 only | 0.802x (0.790-0.814) | correct, but slow and worse with m (pooler 25 us at m=1, 58 us at m=32): x staging did serial 4-byte loads |

## Lessons

- At m >= 16, CUDA-core math (one FMA per weight per token) runs out of shared-memory or L1
  bandwidth before HBM bandwidth. Tensor cores (mma) read each x fragment once per 16x16 weight
  tile instead.
- Zeroing the router output costs about 0.7 us as its own graph node.
- Staging x one 4-byte load per loop iteration serializes on L2 latency (each block waits
  ~0.5 us per round trip before any math). Use wide coalesced loads, many in flight.
