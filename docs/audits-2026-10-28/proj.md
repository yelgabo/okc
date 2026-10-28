# Audit: dense BF16 projections in muse_decode_layer (2026-10-28)

Read-only audit of the projection path (qkv_gate, o, pooler, unpooler, router) as it runs
inside `muse_decode_layer/submission.py`, with the design questions from the brief answered
against the repo's own measurements and the Hopper documentation. Nothing in the repo was
edited.

## 0. The calibration that changes the brief's premises

The repo already measured the judge's streaming-read ceiling. `docs/runs-moe.md:24` (check
16fd966f071340eb, kernel in `tools/moe_make_bench.py:27-60`: plain `ld.global.nc.L1::no_allocate`
16-byte loads, 4 in flight per thread, best of five grids up to 2112 x 256 threads, cold L2,
graph replay minus a 5.0 us empty replay) gives:

| buffer | 9 MB | 18 | 28 | 57 | 113 | 283 | 640 |
|---|---|---|---|---|---|---|---|
| TB/s | 2.68 | 3.06 | 3.28 | 3.65 | 3.80 | 3.95 | 4.06 |
| implied us | 3.5 | 6.2 | 9.0 | 16.4 | 31.2 | 75.1 | 165.3 |

Every pairwise fit in the 18 to 113 MB range gives an asymptote of 3.97 to 4.03 TB/s and an
intercept of 1.3 to 1.5 us. I use `t_min(B) = 1.3 us + B / 4.05 TB/s` as the practical floor
for one kernel on this judge. The physical 4.8 TB/s floors in TASK.md are not reachable by any
single kernel here; "85-90% of 4.8 TB/s" is above anything this machine has shown.

| weight | bytes | TASK floor @4.8 | t_min (kernel us) | our m=1 kernel (bf16_proj profile c50f83d2) |
|---|---|---|---|---|
| qkv_gate 2304x8192 | 36.0 MiB | 7.9 | 10.6 | 10.2 |
| o 8192x1024 | 16.0 MiB | 3.5 | 5.4 | 4.7 |
| pooler 4096x8192 | 64.0 MiB | 14.0 | 17.9 | 18.2 |
| unpooler 8192x4096 | 64.0 MiB | 14.0 | 17.9 | (cuBLAS 18.5 at m=16) |
| unpooler slice 1024x4096 | 8.0 MiB | 1.75 | 3.4 | 4.1 |
| router FP32 256x8192 | 8.0 MiB | 1.75 | 3.4 | 4.4 (m=1), 7.8 (m=8) |
| router hi+lo BF16 | 8.0 MiB | 1.75 | 3.4 | cuBLAS 9.4 to 12.5 + add + reduce |

Consequences, each used below:

1. The existing `proj_kernel` at m <= 8 is already at the measured ceiling on qkv_gate, o and
   the pooler (within 0.4 us, in both directions). The brief's "65-72% of peak" is the judge's
   per-shape time, which adds 1.2 to 2.2 us of launch overhead on top of the kernel
   (`docs/hotspots.md:13-16`: 12.1 vs 10.2, 6.9 vs 4.7, 19.4 vs 18.2, 5.3 vs 4.1). A new GEMV
   load path cannot recover more than about 0.7 us (unpooler slice) unless it beats the
   streaming kernel's asymptote, which experiment E1 decides first.
2. Above 8 tokens cuBLAS is not uniformly "1.5-1.9x the floor" in a way a kernel can fix:
   at m=16/32 its unpooler kernel (18.5 us for 64 MiB, `docs/hotspots.md:21`) is 0.6 us above
   t_min, and its o kernel (4.6 us) is below it. The recoverable slack at m=16/32 sits in the
   pooler (19.3 + 1.8 us reduce) and qkv_gate (8.1 + 1.8 reduce, two launches), about 5 to 7
   us per step. At m=128/256 cuBLAS's tile choices leave 10 to 14 us per step.
3. The bf16_proj leaderboard (1.637x) cannot be reached with full-size BF16 weights streamed
   at this ceiling: putting every shape at t_min plus 1.2 us launch overhead gives about
   1.35x on that problem (worked in section 7). The leaders either stream fewer bytes or beat
   4.05 TB/s. That is new evidence relevant to the ruled-out compression entry; it is noted
   in section 7, not proposed here.
4. The layer's profile inflates short kernels: in `muse_decode_layer/profiles/submission.0e28c26271c94570.txt`
   the kernel rows sum to 224.5 us at decode 16 against a judge time of 177 us (the file's own
   "GPU busy 197.8"), and the same kernels run 1 to 4 us longer than in the standalone
   problem (o 6.9 vs 4.7 us, qkv nvjet 12.6 vs 8.1, unpooler slice 5.2 vs 4.1). Savings read
   off the layer profile overstate what the judge will show. Every estimate below is given as
   a range: the low end from standalone kernel times, the high end from the layer profile.

## 1. Evidence read

- `muse_bf16_proj/submission.py` (all 665 lines): `router_kernel` 74-160, `proj_kernel`
  276-419 (4-stage cp.async ring, `mma.m16n8k16`, cluster split-K over DSMEM 402-418),
  routing 465-507 (S = 8 for the unpooler slice, 4 for K = 8192, 2 for o), packing 520-531,
  self-tests 534-597 (probe-row exactness 567, 596).
- `muse_decode_layer/submission.py`: FP32-input router 417-519 (x staged as FP32, smem
  `M * 1024 * 4`), the same proj kernel 534-738, PDL helper `launch_pdl` 3719-3731, early
  triggers in the W4 kernels 3307-3308, 3552-3553, 3677-3678, mbarrier/wgmma helpers
  2070-2145 (attention ws kernel), `_proj_wins`/`_proj` 4928-4940, `kernel_fn` 4943-4997
  (router on the side stream 4955-4982, sliced unpooler + `F.pad` 4989-4993).
- `docs/runs.md` (bench sweeps 4afc0b73f2a74a69, e378eace83174734 and lessons 28-79),
  `docs/ruled-out.md:8-19` and 49-54, `docs/hotspots.md:7-42` and 98-115,
  `muse_bf16_proj/TASK.md` (rules 95-101, tests 64-76, correctness 78-85),
  `muse_decode_layer/TASK.md` (tests 56-72), `docs/runs-decode-layer.md`, `docs/runs-moe.md`
  (streaming bench 24, W4 lessons 60-90, TMA section 91+),
  `muse_moe_w4a16/experiments/tma_and_large_tiles.py:946-975` (in-repo `cp.async.bulk` +
  `mbarrier.arrive.expect_tx` precedent), the layer profile 0e28c26271c94570 (all 12 shapes),
  bench logs `muse_bf16_proj/profiles/bench.{4afc0b73f2a74a69,e378eace83174734}.log`.
- Hardware and ISA facts: section 9.

## 2. Where the projection time goes now (kernel us)

Layer profile 0e28c26271c94570 unless marked "standalone" (bf16_proj profile c50f83d2b4714d3e,
SGLang column = baseline-as-submission run bde8b4c49c4a46bd).

| T | qkv_gate | o | pooler | unpooler | router | notes |
|---|---|---|---|---|---|---|
| 1 | ours ~11.5 (two-call row 30.7 with pooler) | ours 6.9 (standalone 4.7) | ours ~19.2 | slice, ours 5.2 (4.1) | FP32 ours 5.2, side stream | all ours |
| 8 | ours ~12.5 (row 33.8) | cuBLAS 7.2 (4.2) | ours ~21.3 | slice, ours 6.0 (5.0) | FP32 ours 13.5 (7.8 with BF16 x standalone), side stream | |
| 16 | cuBLAS 12.6 + splitK share (standalone 8.1 + 1.8) | cuBLAS, in the 27.8 two-call row (4.6) | cuBLAS 20.2 (19.3 + 1.8) | cuBLAS, same row (18.5) | cuBLAS hi/lo 9.4 (2) + add + reduce, side stream | |
| 32 | 12.9 + share (8.6 + 1.9) | in 29.1 (4.8) | 20.5 (20.0 + 1.9) | in 29.1 (18.9) | 10.1 (2) + add + reduce | |
| 128 | 15.1 + reduce (12.3 + 3.0) | in 31.9 (5.4) | 25.7 (24.8) | in 31.9 (20.5) | 10.6 (2) + add + reduce | |
| 256 | 22.5 (19.4) | in 37.5 (8.1) | 32.1 (28.7) | in 37.5 (24.9) | 12.5 (2) + reduce 19.4 (2) | |

Standalone efficiencies against t_min: cuBLAS pooler 1.18x (m=16) to 1.60x (m=256); cuBLAS
qkv 0.93x kernel-only but two launches; cuBLAS unpooler 1.03x (m=16) to 1.39x (m=256); cuBLAS
o 0.85x (m=16) to 1.5x (m=256). The router above 8 tokens is 2.8x to 3.7x t_min, but it runs
on the side stream under the pooler (`docs/ruled-out.md:53`).

## 3. Q1: TMA + wgmma skinny GEMM for m in {16, 32, 128, 256}

### 3.1 Shape of the kernel

Swap-AB: `D[64 rows of W, N = tokens] = W_tile[64, K] * x^T[K, N]`. W is the wgmma A operand
from shared memory (64 x 16 per instruction, K-major, which is W's natural row-major layout),
x^T is the B operand from shared memory (N x 16 per instruction, K-major, which is x's
natural layout). No transposes, no repack of x. Both operands are BF16 with FP32 accumulation
(`wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16`). The output tile is [rows, tokens], the
transpose of `out[t, n]`, so the epilogue goes through shared memory once (section 3.6).

Token padding: N must be a multiple of 8 and at most 256. Instantiate N in
{8, 16, 32, 64, 128, 256} only (all of these exist as CUTLASS atoms, see section 9) and pad
T up to the next one; padded rows of x are zero (TMA out-of-bounds fill, or `cp.async` with
src-size 0 as `proj_kernel` does today at `submission.py:240-244`). Padding costs tensor
throughput only, which is free at T <= 128 (section 3.5) and at most 2x waste in a rare test
shape (T = 33 -> 64, T = 72 -> 128). T > 256 (test shape 264, and bf16_proj's 512 and 1000)
stays on cuBLAS.

### 3.2 Tiles, stages, shared memory

BK = 64 (one 128-byte swizzle row per operand row). Per stage: W 64 x 64 x 2 = 8 KB, x
N x 128 B.

| config | tokens N | CTA rows | consumer WGs | acc regs/thread | stage KB (W + x) | stages | smem KB | CTAs/SM |
|---|---|---|---|---|---|---|---|---|
| A | 8, 16, 32, 64 | 64 | 1 | N/2 (4 to 32) | 8 + 1..8 | 8 | 72 to 128 | 2 (1 at N=64) |
| B | 128 | 64 | 1 | 64 | 8 + 16 = 24 | 6 | 144 + epilogue | 1 |
| C | 256 | 128 | 2 (64 rows each) | 128 | 16 + 32 = 48 | 4 | 192 + epilogue | 1 |

Accumulators: a 64 x N FP32 tile over the 128 threads of a warpgroup is N/2 registers per
thread (CUTLASS `CLayout_64xN` has thread mode 4 x 8 x 4 = 128 and value mode 2 x 2 x N/8 =
N/2, section 9). At N = 256 that is 128 registers; with one producer warp plus two consumer
warpgroups (288 or 384 threads) `__launch_bounds__(384, 1)` leaves 168 registers per thread,
enough for 128 accumulators plus descriptors without `setmaxnreg`. Config C cannot be two
CTAs per SM (smem) and does not need `setmaxnreg`; keep it out of v1.

Bytes in flight per SM: config A with 2 CTAs x 7 stages x 9 to 16 KB = 126 to 224 KB; B: 5 x
24 = 120 KB; C: 3 x 48 = 144 KB. The streaming bench reached its ceiling with 128 KB in
flight per SM (528 x 512 threads x 64 B), so these depths are sufficient. Little's law at
4.05 TB/s / 132 SMs = 31 GB/s per SM and ~2 us loaded latency needs about 62 KB per SM.

x in shared memory across the whole K loop: m = 16 needs 16 x 8192 x 2 = 256 KB, more than
the 227 KB a block may hold, and the per-CTA K-slice of x (m x K/S x 2 = 64 KB at m = 16,
S = 4) would only fit at the cost of the W ring. x streams per stage, as the brief expects.

### 3.3 Weight path: bulk copies of pre-swizzled contiguous tiles, no tensor map

`prepare()` repacks each weight once (as `_pack_mma_a` does now, `submission.py:520-524`)
into [row tile of 64][k tile of 64] chunks of 8 KB, each already in the 128B-swizzle
canonical layout (8-row x 128-byte atoms, 16-byte chunk index XOR row mod 8). The producer
warp then issues one `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`
per stage (8 or 16 KB, contiguous), the exact instruction already used in
`muse_moe_w4a16/experiments/tma_and_large_tiles.py:970-975`. Benefits: no
`cuTensorMapEncodeTiled` for the 38 to 67 MB operands, one instruction per stage instead of
512 `cp.async` per stage, and each CTA streams one contiguous region of DRAM (its row tile's
K-slice), which is the access pattern the streaming bench's best grids had.

The wgmma A descriptor for a tile: start address >> 4 in bits 0-13, LBO field ignored for
swizzled K-major layouts (set 1), SBO = 1024 B (>> 4 = 64) between 8-row groups, base offset
0 (tile bases 1024-byte aligned; round the dynamic smem pointer up by hand because the
runtime guarantees only 16 bytes), layout type 1 = 128B swizzle (section 9). Advancing one
k16 step inside a BK = 64 tile adds 32 bytes, i.e. +2 in the start-address field, the same
arithmetic the attention kernel's `desc(ks + 256, ...)` calls use at `submission.py:2636-2637`.

### 3.4 x path: v1 cp.async, v2 TMA with multicast

v1 (configs A and B): the consumer threads copy the x tile (N x 128 B per stage) with 16-byte
`cp.async.cg` into the swizzled layout (chunk ^= row & 7), then `cp.async.wait_group`,
`fence.proxy.async.shared::cta`, and an mbarrier arrive, so the wgmma (async proxy) sees the
generic-proxy writes. This is the fenced pattern the ws attention kernel uses
(`submission.py:2119`, CLAUDE.md rule). At N = 16 it is one `cp.async` per thread per stage.

v2 (config B at m = 128 and config C): x through `cp.async.bulk.tensor.2d` with a tensor map
(box N x 64, SWIZZLE_128B, out-of-bounds rows zero-filled, created per call on the host with
`cuTensorMapEncodeTiled` obtained via `cudaGetDriverEntryPoint`, passed as a
`__grid_constant__` parameter). The reason for v2 is L2 traffic, not DRAM: every CTA
re-reads x's K-slice, so x traffic = (N_rows / CTA rows) x m x K x 2 B. For the pooler that
is 16 MB at m = 16, 32 MB at m = 32, 128 MB at m = 128, 256 MB at m = 256 with 64-row CTAs.
L2-to-SM bandwidth on this part is uncertain: Chips and Cheese measured 5.5 TB/s near-partition
and 3.8 TB/s aggregate on a throttled H100 PCIe; cuBLAS's own m = 128 pooler kernel (64x64
tiles, 24.8 us) moves at least 195 MB through that path, so H200 sustains >= 7.7 TB/s there.
Multicast (`.multicast::cluster` with a 16-bit CTA mask, section 9) across the row tiles of a
cluster divides x traffic by the cluster's row extent. Targets: keep x traffic under ~40 MB
per GEMM.

### 3.5 Grid, split-K and clusters per layer

Rows per 64-row tile: qkv 36, o 128, pooler 64, unpooler 128 (slice: 16). Split-K S is a
cluster along K with a DSMEM reduce of the [64 x N] FP32 partials (at most 64 KB per CTA at
N = 256; 4 KB at N = 16), done exactly as `proj_kernel` does now (`submission.py:402-418`) but
with S <= 4.

| layer (K) | m = 16/32 (config A) | m = 128 (B) | m = 256 (C, 128-row CTAs) |
|---|---|---|---|
| qkv_gate (8192) | 36 tiles x S=4 = 144 CTAs, cluster (1,4) | 36 x S=4 = 144, cluster (2 rows mcast, 4 k) = 8; x traffic 36 MB | 18 x S=8 = 144, cluster (2, 4); x 36 MB |
| o (1024) | 128 x S=1 = 128, no cluster | 128 x 1, cluster (2,1) mcast; x 16 MB | 64 x S=2 = 128, cluster (2,2); x 16 MB |
| pooler (8192) | 64 x S=2 = 128 (or S=4 = 256 at 2 CTAs/SM), cluster (1,2) | 64 x S=2 = 128, cluster (4 rows, 2 k) = 8; x 32 MB | 32 x S=4 = 128, cluster (2,4); x 64 MB |
| unpooler (4096) | 128 x S=1 = 128 (S=2 = 256 at 2/SM) | 128 x 1, cluster (4,1); x 32 MB | 64 x S=2 = 128, cluster (2,2); x 64 MB |
| unpooler slice (4096), T <= 8 | 16 x S=8 = 128, cluster (1,8) | | |
| router hi+lo (2 x 8192) | 4 x S=8 = 32 CTAs (cluster (1,8)), or S=32 with TMA reduce-add | same | same |

All clusters are portable (<= 8). Cluster size 16 (non-portable) would cut the m = 256 x
traffic to 32 MB but needs a whole GPC free, which the side-stream router kernel can deny;
not recommended.

Tensor-core budget (section 9: 989 TFLOPS dense, 1892 FMA/clk/SM at 1.98 GHz):

- m <= 32: a 64 x 16 x 16 wgmma needs 2.5 KB of smem operands per 16k FMA; at 128 B/clk that
  caps the tensor pipe at ~43% of peak, which is still 3.5x more than the K loop needs at the
  memory rate. Memory-bound by a wide margin.
- m = 128: pooler 8.6 GFLOP = 8.7 us at peak, 11.1 us at 78%; under the 17.9 us memory time.
- m = 256: pooler/unpooler 17.2 GFLOP = 17.4 us at peak, 22.3 us at 78%; compute-bound.
  qkv 9.7 GFLOP = 12.5 us at 78%; o 4.3 GFLOP = 5.6 us. Two consumer warpgroups per CTA are
  needed here so one issues wgmma while the other waits on its barrier; smem operand traffic
  is 74 B/clk for the two 64x256 tiles plus 44 B/clk of TMA writes, under 128 B/clk.

Expected kernel times if the kernel reaches t_min (and 78% of tensor peak where that binds):

| m | qkv | o | pooler | unpooler | router hi/lo |
|---|---|---|---|---|---|
| 16, 32 | 10.6 | 5.4 | 17.9 | 17.9 | 3.4 |
| 128 | 10.8 | 5.6 | 18.5 | 18.5 | 3.6 |
| 256 | 12.5 to 13 | 6 | 22 to 23 | 22 to 23 | 4 |

Against cuBLAS (standalone kernel us, `docs/hotspots.md`): m=16: 9.9 (two kernels), 4.6,
21.1 (two), 18.5; m=32: 10.5, 4.8, 21.9, 18.9; m=128: 15.3 (two), 5.4, 24.8, 20.5; m=256:
19.4, 8.1, 28.7, 24.9. Per-step kernel savings: m=16 about 4 us plus two fewer launches;
m=32 about 5 plus two launches; m=128 about 10.5 plus one launch; m=256 about 17. The layer
profile's deltas are larger (12.5, 12.7, 21, 27 us) because of the inflation in section 0.

### 3.6 Epilogue and numerics

Each consumer warpgroup writes its 64 x N FP32 tile to shared memory (after the DSMEM
split-K sum when S > 1), converts to BF16 once, and stores `out[t, n0..n0+63]` as 128-byte
rows (16-byte stores, or a TMA 2D store). This is FP32 accumulation across all of K followed
by one rounding, the same contract cuBLAS and today's `store_out` keep. Probe rows (one
power-of-two entry) stay exact in any summation order because every product is exact and the
other terms are zero. The router variant keeps FP32 output.

### 3.7 Pipeline mechanics

Producer warp (one elected lane): for stage s, wait `empty[s]` (parity), `mbarrier.arrive.expect_tx`
with the stage's bytes, issue the W bulk copy (and the x TMA in v2). Consumers: wait
`full[s]`, `wgmma.fence`, four `wgmma` per BK = 64 (one per k16), `wgmma.commit_group`,
`wgmma.wait_group 0` before arriving on `empty[s]`. The attention kernel's comment at
`submission.py:2784-2785` records that leaving a wgmma in flight across the loop back-edge
made ptxas serialize every wgmma (C7514); waiting to zero per stage is the safe default and
costs nothing while memory-bound. In config C (compute-bound) let each warpgroup keep one
stage in flight (`wait_group 1`) and check the ptxas output for that warning.

## 4. Q2: the m = 1 and m = 8 GEMV

### 4.1 What limits proj_kernel today

Nothing inside the kernel that a load path can fix. Its standalone kernel times (10.2, 4.7,
18.2 us) sit on the judge's streaming line `1.3 us + B / 4.05 TB/s` (10.6, 5.4, 17.9). The
runs.md lessons that the brief lists (cluster reduction, cp.async issue, bytes in flight,
pipeline depth) were all measured above 8 tokens; at m = 1 the bench showed the reduction
costing about 1 us (qkv `(1,4,1,8,4,False)` 17.6 vs `(1,8,1,8,4,True)` 18.6 bench us,
`bench.4afc0b73f2a74a69.log`) and nothing else moved it. The two visible inefficiencies:

- The unpooler slice: 4.1 us for 8 MiB against t_min 3.4. Its grid is 16 row tiles x S = 8 =
  128 blocks of 4 warps whose whole K-slice (64 KB) fits in the prologue, so the kernel is
  pure latency: ramp, one burst, an 8-way cluster reduce. About 0.7 us is recoverable with
  256 CTAs (S = 16 is not a portable cluster; use 2 CTAs per SM with 32 KB each and S = 8).
- The per-launch overhead and ramp (judge adds 1.2 to 2.2 us per kernel; the kernel's own
  intercept is ~1.3 us). This is the only lever of size at T <= 8: programmatic dependent
  launch (PDL). Launch each projection with `cudaLaunchAttributeProgrammaticStreamSerialization`
  (the layer already has `launch_pdl`, `submission.py:3719-3731`, captured into graphs as
  `cudaGraphDependencyTypeProgrammatic` edges), issue the first STAGES-1 weight stages before
  `griddepcontrol.wait`, and load x after it. Weights do not depend on the previous kernel, so
  the ring fills during the predecessor's tail. The predecessor must execute
  `griddepcontrol.launch_dependents` early for the overlap to be more than launch latency; our
  attention, finalize and (if X5 is adopted) boundary kernels can, SGLang's norm kernels and
  cuBLAS cannot (then the implicit trigger fires when all primary blocks exit, which still
  removes the launch gap). The W4 port measured "PDL saves 2-4 us per forward"
  (`docs/runs-moe.md`, lessons). Estimate 0.5 to 1.0 us per projection launch, 2 to 4 us per
  step at every T.

So the "85-90%" GEMV design is not available on this judge; the design that is available is
the same TMA + wgmma kernel at N = 8 (configs A with S = 4/8), whose only expected gains over
today are the unpooler slice (~0.7 us) and whatever E1 shows above 4.05 TB/s, plus PDL. If E1
shows the bulk-copy streamer no faster than the plain-load streamer, keep `proj_kernel` at
m <= 8 and add PDL to it (3 to 4 hours, item 3 below).

Register-only GEMV (the brief's "16-byte loads straight to registers, no smem"): workable at
m = 1 (x K-slice of 2048 in registers is 32 regs per lane), but at m = 8 x no longer fits and
the FMAs go back to shared memory at 128 B of x per 16 B of weights, which is the smem-bound
regime runs.md lesson 1 already hit. Tensor cores (N = 8 wgmma or today's mma) are the right
tool at m = 8; there is no bandwidth to gain from the register design at m = 1.

### 4.2 The FP32 router at T <= 8

Why T = 8 is slow (13.5 us in the layer, 7.8 us standalone with BF16 x, 4.4 at T = 1):

1. The kernel is phase-serial: all 8 MB of weights are loaded into registers first
   (`submission.py:93-98`, 32 regs per lane), then x is staged, then the FMA loop runs, then
   the cluster reduce. Nothing overlaps.
2. The FMA loop is shared-memory bound: one 16-byte `LDS` per 4 FMAs per token
   (`submission.py:118-131`), 512 KB of smem reads per block at T = 8 with FP32 x (the layer
   stages FP32, `muse_decode_layer/submission.py:434-470`), about 4096 clocks = 2.1 us at
   128 B/clk, twice the BF16-x version's traffic.
3. In the layer it runs on the side stream concurrently with the pooler
   (`submission.py:4955-4967`), sharing HBM with a 64 MiB stream; its wall time stretches to the
   pooler's pace. The same kernel is 7.8 us standalone.

Fix, if it mattered: register-block x (each lane keeps its 4 k-columns x 8 tokens = 32
FP32 registers) and stream weights row by row from global with 8 x 16-byte loads in flight
per lane, accumulating `acc[rows][8]` and reducing across lanes once per warp; no shared
memory in the loop, loads overlap FMAs, FP32 FMA count unchanged (exact FP32 semantics
kept). Expected ~3.5 to 4 us standalone at T = 8. In the layer it is worth about nothing:
the side chain (router + top-8 + route_align) is already hidden under the pooler at T = 8
(the router port's whole gain at verify 1 came from replacing SGLang's 26 us F.linear,
`docs/runs-decode-layer.md:58`), and the 8 MB it reads cost the pooler the same 2 us either
way. A hi+lo tensor-core split at T <= 8 would need x rounded to BF16 (or a three-term
x_hi/x_lo split); SGLang keeps FP32 activations here, so I do not propose it.

## 5. Q3: the router above 8 tokens

Same kernel, HILO mode as today (`submission.py:272-274`, 317-318): the packed stream
alternates an 8 KB hi tile and an 8 KB lo tile per 64-k step; the producer loads one x tile
per pair; the consumer runs the four wgmma twice (hi then lo) against the same B descriptor
into one FP32 accumulator; FP32 epilogue, no rounding. 256 rows = 4 row tiles, K = 8192, 16
MiB of hi+lo bytes... correction: hi and lo are 256 x 8192 x 2 B each, 8 MiB together.
Parallelism: 4 row tiles x S = 8 = 32 CTAs with a DSMEM reduce (each CTA 256 KB, about
1.3 + 256 KB at ~100 GB/s per SM = 4 us), or S = 32 (128 CTAs, 64 KB each) with
`cp.reduce.async.bulk.tensor ... .add` into a zeroed FP32 [m x 256] buffer (zeroing is a
0.7 us graph node, `docs/runs.md:33`; confirm `.add` with f32 in the PTX ISA table before
relying on it, I could not re-fetch that table). Expected 3.5 to 4.5 us at any m <= 256
against cuBLAS's 9.4 to 12.5 us plus the add and the splitK reduce.

Its value in the layer is indirect: `docs/ruled-out.md:53` shows the router hidden under the
pooler at 9-32 tokens. Once the pooler drops from 20 to 18 us and PDL trims the main chain,
the side chain (router 9.4 + add 1 + reduce 2 + routing 3 + route_align 2, about 17 us at
T = 16, and 12.5 + 19.4 (splitK reduce, 2 calls) at T = 256) becomes the critical path.
Treat it as part of item 1, not a separate gain. The earlier failure at 128/256 tokens
(`docs/ruled-out.md:18`) was per-token-group re-reads of the 8 MB; this kernel reads it once
with N = 128 or 256, so that entry's mechanism does not apply.

## 6. Q4: porting into muse_decode_layer

- T is arbitrary in tests (9, 33, 72, 264 appear in `muse_decode_layer/TASK.md:59-71`). Pad
  N to {8, 16, 32, 64, 128, 256}; T > 256 falls back to cuBLAS (`_gemm`); keep the
  `_proj_wins`-style table keyed on T and layer, and route the unpooler at T <= 8 to the
  1024-row slice with S = 8 (the kernel takes `n` as a runtime row count; the packed weight
  holds all 8192 rows and the slice is its first 16 row tiles).
- Inputs must be contiguous [T, K] BF16 with row stride K: `_proj` already calls
  `.contiguous()`; the attention output `o` and `expert_output` come from our kernels. A
  tensor map (v2) bakes x's address at capture; graph replays reuse the captured buffers, as
  every captured kernel argument already assumes.
- Numerics: FP32 accumulate, one BF16 rounding (section 3.6). The split-K sum order differs
  from cuBLAS's; the judge compares against FP64 at 1% / 0.01% (router), and probe rows stay
  exact. Keep the self-tests at `submission.py:551-568` (relative L2 plus `torch.equal` on the
  probe row) and add T in {9, 16, 33, 72, 128, 256, 264}.
- Shared memory: dynamic smem base must be rounded up to 1024 bytes for the 128B swizzle;
  configs B/C need `cudaFuncAttributeMaxDynamicSharedMemorySize` above 144 KB; mbarriers in
  static smem.
- Clusters and the side stream: cluster launches need co-resident SMs in one GPC; the router
  kernel on the side stream occupies up to 32 SMs. Keep clusters <= 8 and 128 to 144 CTAs per
  GEMM; if E3 shows cluster waits, use S = 1 with 2 CTAs per SM where the row count allows.
- PDL ordering: `griddepcontrol.wait` must precede the first read of x and the first write of
  `out`; weights may be prefetched before it. On the side stream the router kernel's
  predecessor is the boundary kernel (SGLang's at T > 8), so only the implicit trigger
  applies there.
- Compile budget: prepare() is at 98.8 s of 300 (`docs/runs-decode-layer.md:68`); the W4
  extension alone took 46.8 s. Keep the new kernel to about 12 instantiations (6 N values x
  {bf16 out, f32 out} with ROWS and S as runtime or few-valued template parameters) and build
  it as its own extension so a failure falls back cleanly, as `_try` does today.
- Environmental slowdown: the layer profile shows our o kernel at 6.9 us (4.7 standalone) and
  cuBLAS's o at 7.2 (4.2). Part is profiler inflation (section 0), part may be real. A new
  kernel inherits whatever is real; only overlap (PDL) attacks it.

## 7. Ranked list of changes

Score sensitivities from the brief (per 10 us saved on both layers of a load, sliding + full):
d1 0.020, v1 0.0148, d16 0.0115, d32 0.0086, v16 0.0058, v32 0.0039. Ranges: low end from
standalone deltas, high end from layer-profile deltas.

### 1. Skinny TMA + wgmma GEMM for 9 <= T <= 256, with the hi/lo router on the same kernel

- Mechanism: section 3 (configs A, B, C), section 5 for the router. Replaces cuBLAS for
  qkv_gate, o, pooler, unpooler and the two-GEMM router above 8 tokens.
- us saved per step (kernel): T=16: 4 to 12.5 (+1 to 2 from two fewer launches); T=32: 5 to
  12.7 (+2); T=128: 10.5 to 21 (+1.5); T=256: 12 to 27; T=1, T=8: 0 (not routed here).
  Per shape: qkv 0 to 4 (m <= 32), 4.5 to 6 (128), 6.5 to 9.5 (256); pooler 2.2 to 3.2
  (m <= 32), 6 to 7 (128), 6 to 9 (256); unpooler 0 to 1 at m <= 128, 0 to 2 at 256; o 0 to
  1.5 (m <= 128), 2 to 6 (256); router 6 to 20 but on the side stream.
- Score: d16 +0.006 to +0.014, d32 +0.005 to +0.011, v16 +0.007 to +0.012, v32 +0.005 to
  +0.011; total +0.025 to +0.048.
- Confidence: medium. The ceiling is measured, cuBLAS's standalone distance from it is
  measured per layer, and the tensor-core and smem budgets have 3x margin below m = 128. The
  m = 256 case rests on reaching ~78% of wgmma peak with two warpgroups, which is cuBLAS-level
  tuning; medium-low there. The L2 re-read budget at m >= 128 is inferred from cuBLAS's own
  traffic, not measured.
- Effort: v1 (configs A and B with cp.async x, bulk-copy W, cluster split-K, epilogue,
  packing, CPU emulation of the swizzle and descriptors, self-tests, bench harness): 16 to 24
  h. v2 (tensor map + multicast, config C, router mode): 14 to 18 h. Layer port and routing
  table: 6 to 8 h. Total 36 to 50 h.
- Risk: cluster co-scheduling against the side-stream router; ptxas serializing wgmma
  (C7514); compile time; x re-read traffic at m = 256 (64 MB per GEMM) if L2 bandwidth is
  nearer the 3.8 TB/s aggregate figure than cuBLAS's implied 7.7+.
- Cheapest confirming experiment: E3, config A on the pooler and qkv_gate at m = 16 and 32,
  with and without the cluster reduce, in a check (bench) run beside cuBLAS (as
  `tools/make_bench.py` did). Go/no-go: pooler kernel <= 18.5 us and qkv <= 11.5 us.
- Reopens: "Our mma kernel on BF16 layers above 8 tokens, any config" (`docs/ruled-out.md:17`).
  New evidence: that entry's own no-reduction rows show the mma.sync + cp.async mainloop capped
  at cuBLAS level (bench e378 red=False: qkv m=16 18.9 vs 20.6, pooler 27.3 vs 29.9 bench us),
  so the mechanism ruled out was that mainloop; this kernel has a different load path (one
  bulk copy per stage, operands read by the tensor core from swizzled smem, no ldmatrix/lds)
  and the measured streaming line shows 2 to 7 us per layer of headroom at m >= 32. Also
  reopens "Cluster split-K reduction above 8 tokens" (`ruled-out.md:15`) in a narrowed form:
  S <= 4, partials of 4 to 64 KB, with the no-reduce control measured in E3; the old evidence
  grew with S up to 8 on 128-thread blocks. "Larger row groups per block above 8 tokens"
  (`ruled-out.md:14`): config C's 128-row CTAs are justified by x traffic at m = 256 (4 MB per
  CTA pass), a cost that did not exist at the m = 16/32 shapes that run measured.

### 2. PDL weight prefetch on every projection and router launch

- Mechanism: section 4.1. Applies to today's `proj_kernel`/`router_kernel` immediately and
  to item 1's kernel later. Prefetch the W ring before `griddepcontrol.wait`; add
  `griddepcontrol.launch_dependents` to our kernels that precede a projection (attention q_len
  1 kernel, ws combine, finalize, boundary).
- us saved per step: 2 to 4 at every T (0.5 to 1 per launch, 3 to 4 launches).
- Score: d1 +0.004 to +0.008, v1 +0.003 to +0.006, d16 +0.002 to +0.005, d32 +0.002 to
  +0.003, v16 +0.001 to +0.002, v32 +0.001 to +0.002; total +0.013 to +0.026.
- Confidence: medium-high that it is positive (the W4 port measured 2 to 4 us per forward
  from the same mechanism); low on the exact size, since cuBLAS/SGLang predecessors only
  give the implicit trigger.
- Effort: 3 to 4 h for the existing kernels (launch through `launch_pdl`, two asm lines,
  a self-test under graph capture as `_route_side_self_test` does). Risk: low; a kernel that
  reads x before the wait is a correctness bug the self-test catches.
- Cheapest experiment: E4, one check run of the layer with PDL on the four `proj_fwd` sites
  and `router_fwd`; compare d1 and v1 sliding against 0e28c26271c94570 (116.8 / 155.4 us).
- Reopens nothing.

### 3. Same kernel at N = 8 for T <= 8 (replace proj_kernel), gated by E1

- Mechanism: config A with S = 4 (qkv, pooler), S = 2 (o), S = 8 (unpooler slice), N = 8.
- us saved per step: T=1: 1 to 5.5; T=8: 3.6 to 9.7 (the high ends assume the layer-profile
  slowdowns are real kernel time that a better streamer removes; the low ends are the
  standalone differences, mostly the unpooler slice).
- Score: d1 +0.002 to +0.011, v1 +0.005 to +0.014.
- Confidence: low-medium. Standalone, today's kernel is already on the streaming line.
- Effort: 4 to 6 h on top of item 1 (routing, S choices, slice handling). Risk: low once
  item 1 exists.
- Cheapest experiment: E1 (2 h): a bulk-copy streaming kernel (8 KB stages, 8-deep ring,
  2 CTAs per SM, contiguous region per CTA) over 8, 16, 36 and 64 MiB in the moe bench
  harness, against the plain-load numbers in `docs/runs-moe.md:24`. If it does not exceed
  4.05 TB/s asymptotically or cut the 1.3 us intercept, skip this item and keep item 2.
- Reopens nothing.

### 4. FP32 router rewrite at T <= 8

- Mechanism: section 4.2 (register-blocked x, streamed weights, no smem loop).
- us saved: standalone T=8 about 4 us, T=1 about 1; in the layer about 0 (hidden under the
  pooler; the bytes still flow). Score: ~0 on the layer; on bf16_proj verify 1 it is worth
  about +0.03 of that problem's score.
- Confidence: high on the standalone number, high that the layer gains nothing.
- Effort: 4 to 6 h. Risk: low. Cheapest experiment: a bf16_proj check at m = 8.
- Reopens nothing ("CUDA-core router above 8 tokens" stays closed; this is T <= 8).

### Not proposed, with the evidence

- Lossless compression stays closed per `docs/ruled-out.md:19`. One observation for the
  record: with every shape at t_min plus 1.2 us launch overhead, bf16_proj scores about 1.35x
  (load ratios 1.34, 1.22, 1.25, 1.67, 1.34, 1.32), while the board's leader is at 1.637x and
  rank 50 at 1.545x. Either they stream fewer bytes or they beat 4.05 TB/s; E1 separates the
  two. That is the evidence a reopening would need to state.
- Fusing the unpooler slice GEMV into the X5 boundary cluster kernel (one launch fewer at
  T <= 8) depends on X5, which is held for the BF16 rounding question; not costed here.

### Sum

Items 1 to 3 together: +0.04 to +0.07 on the layer score (1.403 -> about 1.44 to 1.47). This
path does not approach rank 50 (2.135x); at T >= 16 the step is dominated by the W4 expert
GEMMs (85 to 350 us per step against streaming floors of 37 to 107 us, `docs/runs-moe.md`
table) and attention, which this audit did not cover.

## 8. Experiments in order

1. E1 (2 h, no kernel change to the submission): bulk-copy streaming ceiling at 8/16/36/64
   MiB vs `docs/runs-moe.md:24`. Decides item 3 and calibrates item 1's t_min.
2. E4 (3 to 4 h): PDL on the existing projection and router launches in the layer; one check.
3. E3 (after v1, ~20 h in): config A at m = 16/32 on pooler and qkv_gate, with a no-reduce
   control, in a check (bench) run; then o and unpooler; then the router HILO mode.
4. E2 (1 h, alongside E3): each CTA re-reads a shared 2 to 4 MB buffer S times; measures
   L2-to-SM bandwidth and sets the x-traffic budget for configs B and C.
5. E5 (after v2): m = 128/256 with multicast; compare to cuBLAS 24.8 / 28.7 (pooler) and
   12.3 + 3.0 / 19.4 (qkv).

## 9. Hardware and ISA facts relied on

- H200 SXM: 4.8 TB/s HBM3e, 141 GB, 1,979 BF16 TFLOPS with sparsity (989 dense), 67 FP32
  TFLOPS (nvidia.com/en-us/data-center/h200). 132 SMs and ~1.98 GHz boost are the GH100 SXM
  figures; 989e12 / (132 x 1.98e9) = 3784 FLOP/clk/SM = 1892 FMA/clk/SM.
- Hopper SM (CUDA Hopper tuning guide): 228 KB shared memory per SM, 227 KB per block,
  64K 32-bit registers per SM, 255 per thread, 50 MB L2 in two partitions, TMA moves 1D to
  5D tensors global<->shared and between cluster CTAs' shared memory, portable cluster size
  8 (16 non-portable with `cudaFuncAttributeNonPortableClusterSizeAllowed`), DSMEM reads,
  writes and atomics across a cluster.
- wgmma (CUTLASS `cute/atom/mma_traits_sm90_gmma.hpp` and `cute/arch/mma_sm90_gmma.hpp`,
  fetched): `CLayout_64xN = Layout<Shape<Shape<_4,_8,_4>, Shape<_2,_2,Int<N/8>>>, ...>`, i.e.
  128 threads x N/2 values; BF16 x BF16 -> F32 atoms instantiated for N = 8, 16, 32, 64, 96,
  128, 192, 256 (`MMA_64xNx16_F32BF16BF16_SS`, A and B from smem descriptors, and `_RS` with A
  in registers; B is always smem). The PTX ISA's shape table allows every multiple of 8 for
  16-bit inputs; I could not re-fetch that page (the fetch tool truncates it and the PDF is
  over its size limit), and the design uses only the instantiated set. Operand constraints
  in the atom: accumulators `"+f"`, descriptors `"l"`, scale/transpose flags `"n"`, scale_D
  via a predicate. `wgmma.fence.sync.aligned`, `wgmma.commit_group.sync.aligned`,
  `wgmma.wait_group.sync.aligned N` are already wrapped at
  `muse_decode_layer/submission.py:2133-2142`.
- Shared-memory matrix descriptor (`cute/arch/mma_sm90_desc.hpp`, fetched): start address
  bits [0,14) (>> 4), leading byte offset [16,30) (>> 4, ignored for swizzled layouts),
  stride byte offset [32,46) (>> 4), base offset [49,52), layout type [62,64) with
  SWIZZLE_NONE 0, 128B 1, 64B 2, 32B 3. K-major SW128 atom: 8 rows x 1024 bits = 8 x 64 BF16,
  `Swizzle<3,4,3>`.
- TMA / bulk copy instruction text (`cute/arch/copy_sm90_tma.hpp`, fetched):
  `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [dst], [map, {c0, c1}], [mbar], hint;`
  multicast form adds `.multicast::cluster` and a 16-bit `"h"` CTA mask;
  `cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [map, {c0, c1}], [src];`
  `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [src], bytes, [mbar];`
  fences `fence.proxy.async.shared::cta`, `cp.async.bulk.commit_group`,
  `cp.async.bulk.wait_group.read N`. In-repo use: `muse_moe_w4a16/experiments/tma_and_large_tiles.py:948-975`.
- `setmaxnreg.inc/dec.sync.aligned.u32 N` (`cutlass/arch/reg_reconfig.h`, fetched; compiled
  only for sm_90a). Not needed by the proposed configs.
- PDL (CUDA programming guide 4.5, fetched): `cudaGridDependencySynchronize` blocks until the
  primaries "have completed and flushed results to global memory"; if the primary never calls
  the trigger "it implicitly occurs after all thread blocks in the primary kernel exit";
  stream capture with `programmaticStreamSerializationAllowed = 1` records a
  `cudaGraphDependencyTypeProgrammatic` edge; concurrency "is opportunistic and not
  guaranteed". In-repo: `muse_decode_layer/submission.py:3307-3308, 3719-3731`.
- L2 bandwidth: Chips and Cheese, H100 PCIe (throttled to ~1.4 GHz), over 5.5 TB/s near
  partition, 3.8 TB/s across the full 50 MB; no SXM/H200 measurement found. cuBLAS's m = 128
  pooler kernel implies >= 7.7 TB/s on this judge (section 3.4).
- Streaming ceiling on this judge: `docs/runs-moe.md:24`, kernel `tools/moe_make_bench.py:27-60`.
