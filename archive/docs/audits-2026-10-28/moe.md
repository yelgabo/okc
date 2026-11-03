# Audit: NVFP4 W4A16 MoE kernels in muse_decode_layer

Date 2026-10-28. Read-only audit of the MoE path as it runs inside `muse_decode_layer/submission.py`
(kernel source copied from `muse_moe_w4a16/submission.py`). No GPU was available; every number
below is from the repo's profiles, run logs and docs, from reading the code, or from a pure-Python
port of the tuner's synthetic routing (`routing_model.py` beside this file). Where I model, I say so.

Sources read in full: `muse_moe_w4a16/submission.py` (1791 lines), `docs/runs-moe.md`,
`docs/ruled-out.md`, `muse_moe_w4a16/experiments/tma_and_large_tiles.py`, `tools/moe_emulate_w4.py`,
the MoE glue in `muse_decode_layer/submission.py` (lines 140-260, 740-800, 3140-3170, 4058-4225,
4456-4665, 4837-4997), `muse_decode_layer/profiles/submission.0e28c26271c94570.{txt,log,json}`,
`muse_moe_w4a16/profiles/submission.a49578e5c30348bb.{txt,log}`, both baseline profiles, the two
TASK.md files, `docs/hotspots.md`, `docs/runs-decode-layer.md`.

Score sensitivity used throughout (from the brief, both layers of a load summed, per 10 us saved on
that T): T=1 0.020, T=8 0.0148, T=16 0.0115, T=32 0.0086, T=128 0.0058, T=256 0.0039.

---

## 1. Config map and the tuned policy

`W4_CONFIGS` in the layer (`muse_decode_layer/submission.py:3151-3165`, 31 entries, tuple
`(kind, nt, rows, s, w, d, mode, layout, p)`): 0 (1,1,1,4,4,4,0,0,0), 1 (1,1,8,2,4,4), 2 (1,1,8,4,4,4),
3 (1,1,8,2,4,8), 4 (1,1,8,2,8,4), 5 (1,2,16,4,4,4), 6 (1,2,16,2,4,4), 7 (1,2,16,4,8,4),
8 (1,2,16,4,8,4, layout 1), 9 (2,1,1,4,4,8), 10 (2,1,1,4,4,4), 11 (2,1,8,4,4,4), 12 (2,1,8,2,4,4),
13 (2,2,16,4,4,4), 14 (2,2,16,2,4,4), 15-19 persistent GEMM1 twins, 20-25 persistent GEMM2 twins
(21 = (2,1,1,4,4,4,P), 24 = (2,1,8,2,8,4,P)), 26 (1,1,8,4,4,4, mode 5 no hint), 27 (1,1,8,2,8,4, mode 5,
persistent), 28 (2,1,8,4,4,4, mode 5), 29 (2,2,16,4,4,4, mode 5), 30 (1,2,16,4,4,4, mode 5).

Policy from `profiles/submission.0e28c26271c94570.log` (`TUNE policy`):

| T | GEMM1 | GEMM2 | PDL | finalize cluster |
|---|---|---|---|---|
| 1 | 27: NT1 ROWS8 S2 W8 D4, no hint, persistent | 21: NT1 ROWS1 S4 W4 D4, persistent | on | 8 |
| 8 | 2: NT1 ROWS8 S4 W4 D4 | 24: NT1 ROWS8 S2 W8 D4, persistent | on | 4 |
| 16, 32 | 26: NT1 ROWS8 S4 W4 D4, no hint | 28: NT1 ROWS8 S4 W4 D4, no hint | on | 8 |
| 128 | 4: NT1 ROWS8 S2 W8 D4 | 29: NT2 ROWS16 S4 W4 D4, no hint | on | 1 |
| 256 | 30: NT2 ROWS16 S4 W4 D4, no hint | 13: NT2 ROWS16 S4 W4 D4 | on | 1 |

The profile's kernel names confirm the map (`w4_gemm_kernel<1, 2, 16, 4, 4...` at verify 32 and so
on). Registers, shared memory and residency (docs/runs-moe.md:114-127, measured in check
4a8852d513874318): (1,1,8,4,4,4) 76 regs, 16.6 KB, 6 blocks/SM, 24 warps/SM; (1,2,16,4,4,4) 92 regs,
33.2 KB, 5 blocks, 20 warps; cfg 4 (1,1,8,2,8,4) is not in that table; its per-thread state matches
cfg 2 and its smem is 8 x 2056 x 2 = 32.9 KB, so 3 blocks of 8 warps per SM, 24 warps/SM.

---

## 2. Q1: work decomposition at T=128 and T=256

### 2.1 Units of work in the kernel

From `w4_gemm_kernel` (`muse_moe_w4a16/submission.py:635-899`):

- An item is up to `8*nt` routes of one expert (`route_align_kernel`, lines 142-165). A unit is 32
  weight rows x one K split; a warp owns one unit at a time (`decode`, lines 675-690: `u = chunk*W +
  warp`). U = 256 units per expert for GEMM1 (8192 rows), 128 for GEMM2. A block is W consecutive
  units of one (item, split). Grid = `dim3(U/W, max_items, S)` (line 1109).
- Per stage of 64 k a warp loads 2 x 16 B per lane of values (1024 B) plus one 16 B scale word per
  4-lane group (128 B unique per warp): 1152 B per warp-stage, 16 stages per unit at S=4
  (KC=1024), so 18.4 KB per warp-unit and 73.7 KB per block (lines 738-747, 834-843).
- Each block stages the item's activation rows for its K slice through `cp.async` into shared memory
  before its first mma (`stage_x`, lines 710-727, `cp.async.wait_all; __syncthreads()` at 761-762):
  `nrows x KC x 2` bytes, 32 KB for a full 16-row item at S=4.
- Dequant: `dequant<TILE>` (lines 597-609) is 1 LOP3 (r0), SHL+LOP3 (r1), SHF+LOP3 (r2), 2 shifts +
  3 LOP3 (r3), PRMT + IMAD for the scale, 4 HMUL2: 14 SASS per 32-bit word (8 weights), 9 on the
  integer pipe, matching the doc's "about 17 per 8 weights" once the loop, ldmatrix, loads and mma
  are shared out. Per warp-stage: 8 dequants (112 instr), 8*NT `mma.sync m16n8k16`, 2*NT
  `ldmatrix.x4`, 3 loads, about 10 for addressing and the loop: about 136 (NT1) to 150 (NT2)
  warp-instructions per 1152 B.
- Epilogue (lines 848-885): FP32 partials, 8*NT store instructions of 4 rows x 32 B per unit.

### 2.2 Item counts and bytes (routing model, 30 seeds of the tuner's "normal" routing)

`routing_model.py` ports `_make_inputs` (lines 1658-1680). Medians with min-max; the brief's expert
counts (bs128 about 24, bs256 about 22.5) sit inside these ranges.

| | T=256 GEMM1 cfg 30 | T=256 GEMM2 cfg 13 | T=128 GEMM1 cfg 4 | T=128 GEMM2 cfg 29 |
|---|---|---|---|---|
| local routes | 380 (293-615) | same | 188 (145-316) | same |
| unique experts | 25 (20-30) | same | 23 (17-29) | same |
| hottest expert routes | 72 (27-256) | same | 35 (14-128) | same |
| items (rows per item) | 38 (30-54) at 16 | 38 at 16 | 36 (30-53) at 8 | 28 (22-37) at 16 |
| blocks (grid real part) | 9,600 (7,680-13,824) | 4,864 | 2,304 (1,920-3,392) | 3,584 |
| threads per block / warps per SM | 128 / 20 | 128 / 20 | 256 / 24 | 128 / 20 |
| hot expert's items (re-reads of its weights through L2) | 5 (2-16) | 5 | 5 (2-16) | 3 (1-8) |
| weight bytes L2->SM, items x expert | 709 MB (567-1021) | 352 MB | 680 MB (567-1002) | 258 MB |
| unique weight bytes (HBM) | 472 MB (378-567) | 235 MB | 435 MB (321-548) | 216 MB |
| activation staging L2->SM | 199 MB (154-322) | 100 MB | 49 MB (38-83) | 49 MB |
| FP32 partial writes | 50 MB | 25 MB | 12 MB | 12 MB |
| total SM-side traffic | about 960 MB | about 480 MB | about 740 MB | about 320 MB |
| warp-stages | 622 K | 311 K | 590 K | 230 K |
| dequant warp-instructions | 70 M | 35 M | 66 M | 26 M |
| mma.sync count | 10.0 M (16 per warp-stage) | 5.0 M | 4.7 M (8 per warp-stage) | 3.7 M |
| measured time (layer profile; tuner) | 234 us; 243 incl. SwiGLU reduce and about 5 us replay | 112 us; 121.5 incl. finalize | 150 us; 166 | 69 us; 91 |
| effective SM-side rate | 4.1-4.5 TB/s | 4.3 | 4.9 | 4.6 |
| effective HBM rate (unique + writes) | 2.2-2.4 TB/s | 2.3 | 3.0 | 3.3 |

The grid is launched with `max_items = count + numel/item_rows + 1` (host bound, line 401;
layer 771-772), so at T=256 the launch is 64 x 161 x 4 = 41,216 blocks of which about 9,600 do
work; the rest exit after `num_items[0]` (lines 668-672). Blocks are launched in (x, y, z) order,
so the empties are interleaved: after every split plane. Section 5 covers this.

### 2.3 Budget per SM at the HBM rate

At 4.8 TB/s every SM must consume 18.4 B/cycle (1.98 GHz, 132 SMs); at the measured 3.95 TB/s
streaming ceiling, 15.1 B/cycle; the kernels actually push 16-17 B/cycle of SM-side traffic. From the
instruction mix above (computed in `routing_model.py`'s companion run):

| SM-side rate | issue slots used (of 4/cycle) | INT pipe (16 lanes/sub-partition) | tensor pipe (mma.sync, about 8 cycles per m16n8k16) |
|---|---|---|---|
| 4.8 TB/s, NT1 / NT2 | 54% / 58% | 57% | 26% / 51% |
| 4.3 TB/s (what the big shapes run at), NT1 / NT2 | 48% / 52% | 51% | 23% / 46% |

So at the rate the big shapes achieve, every scheduler already spends about half its issue slots and
the INT pipe is half busy, with 5 warps per scheduler (20 warps/SM) to hide a 1.5-2.5 us load
latency. The bytes-in-flight bound is fine on paper (20 warps x 4 stages x 1152 B x 132 SMs = 12 MB,
against the 6-10 MB that Little's law needs at 3.95 TB/s), but it only holds while the warps keep
issuing their next loads, and a warp issues the loads for stage s+4 only after it has finished the
dequant and mma of stage s (lines 834-838).

### 2.4 Why GEMM1 at bs256 is at 1.75-2.0 TB/s over unique bytes

Three facts together:

1. Each unique weight byte crosses L2->SM 1.5x (items per expert, 38/25) and the activation staging
   and partial writes add another 0.5x, so SM-side traffic is about 2.05x the unique bytes: about
   960 MB for 472 MB of weights. Reaching the HBM ceiling of 3.95 TB/s would need 8.1 TB/s of
   SM-side traffic.
2. This access pattern (about 10,000 short blocks, each warp pulling its own 1 KB per stage, half of
   the slices in the far L2 partition) tops out at 4.1-4.9 TB/s of SM-side traffic. The direct
   evidence is already in the repo: docs/runs-moe.md:107, bench e1619f311d3747e3, cfg 37 at bs256:
   normal routings n11/n12/n13 take 258/229/254 us, the skewed routings s21/s22 take 285/224 us.
   The skewed routing has the same item count (my model: 37 vs 38), the same SM-side traffic
   (about 1.0 GB) and 6x fewer unique bytes (76 MB vs 472 MB, 4 experts vs 25), and it is not
   faster. The kernel's time follows SM-side bytes, not HBM bytes. The same holds at bs128 (cfg 4:
   s21 182 us against n12 157 us with 76 MB vs 435 MB unique).
3. The model `t = max(SM-side bytes / 4.3 TB/s, unique bytes / 3.9 TB/s)` reproduces all four big
   kernels within 10%: 960/4.3 = 223 (measured 215-234), 480/4.3 = 112 (105-112), 740/4.3 = 172
   against a measured 141-150 for cfg 4 (24 warps/SM reach about 5 TB/s), 320/4.3 = 75 (69-75).
   At bs8-32 the two bounds coincide (one item per expert, 1.0-1.1x) and the kernels sit at 85-97%
   of the streaming ceiling, which is why every mainloop tweak moved about 1 us there.

Smaller contributors, all measurable but secondary:

- Activation staging: 32 KB per block for a full 16-row item, loaded and waited on before the first
  mma (about 1 us of a 12 us block, partly overlapped by the 4-stage weight prologue), and 20-28% of
  the weight bytes in SM-side traffic at NT2.
- Tail and imbalance: blocks are uniform in bytes, so the hot expert makes no long pole; its 5
  (worst 16) items are consecutive in y and share L2 fetches within a wave of about 660 resident
  blocks. A 256-route expert (n11) spans 4,096 blocks, more than one wave, so its weights come from
  HBM about twice: about 19 MB extra, under 5%. The last partial wave (9,600 / 660 = 14.5 waves)
  costs about 4 us of reduced memory parallelism.
- Split-K partials: 50 MB written and 50 MB read back by the SwiGLU reduce at bs256 (12 + 12 MB at
  bs128).
- Persistence hurts at these sizes (cfg 19 at T=256: 272.8 us vs cfg 5's 244.1; at T=128 195.1 vs
  169.6): the persistent walk is item-major with the split fastest (lines 679-684), so only about
  2.6 items are in flight at once and consecutive items of a hot expert no longer share L2 fetches.

Cheapest confirming experiments (the first one decides the design question):

- E1 (2 hours, one check run through `_graph_time`): time GEMM1 cfg 30 at bs256 on two hand-built
  item lists with identical item counts: 16 items on one expert vs 16 experts with one 16-row item
  each. If the one-expert case is not at least 2x faster, the limit is SM-side traffic and the
  dequant-once design (section 3) is right. The existing skewed rows already say this; E1 removes
  the routing difference.
- E2 (1 hour): the mode-4 "streamlike" loads-only kernel (cfg index of `(1,1,8,4,8,4,4,0,0)` in the
  MoE table) over a 16 MB window re-read 32x. That measures the L2-resident SM-side ceiling for this
  pattern; the repo only has the cold-L2 (HBM) curve.

---

## 3. Q2: a design for bs128/bs256

### 3.1 Constraints any hot-expert design must satisfy

- Reuse per weight byte. Today each item re-reads and re-dequantizes the expert. The cure is one
  read and one dequant per CTA, reused across all tokens the CTA owns.
- Shared-memory operand bandwidth rules out mma.sync for large N. With weights as A fragments in
  registers and tokens as B through `ldmatrix`, a warp reads NT KB of B per 1152 B stage (lines
  806-811). At 18 B/cycle/SM that is 64 x NT B/cycle of shared-memory reads: NT=8 (64 tokens)
  saturates the 128 B/cycle shared-memory port on its own. That, plus 124-190 registers and one
  block per SM (docs/runs-moe.md:124-126), is why the large-tile experiment lost 2x on normal
  routing. wgmma reads A and B through the tensor core's own path at 10 KB per 262,144 MACs for
  m64n256k16, 74 B/cycle at the full 989 TFLOP/s rate, so it is the only way to get N above 32.
- Activation traffic. With weights as A (M_tile rows per CTA) and N tokens as B, every CTA reads
  N x K x 2 B of activations against M_tile x K x 0.5 B of weights: the ratio is 4N/M_tile. For
  N=256 and M_tile=128 that is 8x the weight bytes, worse than today. Either M_tile >= 4N (impossible:
  accumulators of 1024 rows x 256 tokens) or the activation slab must be multicast across the CTAs
  of a cluster that own neighbouring row tiles. Cluster of 4 x 128 rows = 512 rows per slab load:
  activation traffic 2x weights at N=256, 0.5x at N=64.
- Tensor time of the hot experts. A 256-route expert's GEMM1 is 17.2 GFLOP, 17.4 us at the dense
  BF16 peak on all 132 SMs; a CTA of 128 rows x 256 tokens x K/2 is tensor-bound 4.4x over its byte
  share (1108 cycles of wgmma per 64-k step against 250 cycles of HBM time at 18 B/cycle). So hot
  experts need >= 128 CTAs each (K split 2 with 128-row tiles) and must be scheduled first so that
  cold-expert streaming fills the other SMs meanwhile. Total MoE math at bs256 is about 38 GFLOP,
  39 us at peak, so the device as a whole stays memory-bound.
- Occupancy. 128 accumulator registers per thread at n256 means one CTA per SM for the hot kernel;
  the cold register path keeps running beside it (its 92 regs x 128 threads and 33 KB fit next to a
  53K-register, 160 KB CTA).

### 3.2 The options in the brief

(a) Dequantize once into smem as BF16, wgmma with weights as A from smem and tokens as B from
smem, N = 64..256: right mechanism, but on its own it moves the problem to activation traffic
(ratio 4N/M_tile above). It needs the cluster multicast of the activation slab, so (a) and (c) merge.

(b) Keep the register path for cold experts and use (a) above a threshold, with the device-side
class split from `experiments/tma_and_large_tiles.py` (`ItemLists`, lines 99-104; `route_align`
lists with lo/hi, lines 152-175; `_w4_launches`, 1783-1790): yes, with the threshold at 16 routes,
not 32 or 64. At bs128 the median hot expert has only 35 routes and the traffic amplification comes
from the many 17-40-route experts split into 8-row items (36 items for 23 experts). With the
threshold at 16, every cold expert is exactly one 16-row item (1.0x) and every hot expert is read
once. Instantiate the hot kernel for n32, n64, n128, n256 (padding a 17-route expert to n256 would
make its CTAs 2x tensor-bound; padding to n32 keeps them balanced).

(c) A 2-CTA cluster multicasting a dequantized weight tile: multicasting A between two CTAs that own
different token tiles of one expert is the same as one CTA with twice the N, which wgmma already
gives up to n256; it only halves accumulator registers per CTA. It does not touch the activation
traffic, which is the binding term once weights are read once. Use the cluster for the activation
slab (B) instead, across CTAs that own neighbouring weight-row tiles.

(d) Raw-weight multicast inside the existing TMA kernel across 2-4 consecutive items of the same
expert (proposal P3 in section 6): cheaper, keeps mma.sync and the per-item dequant, cuts only the
weight re-reads. A good two-day intermediate if the full design slips.

### 3.3 Chosen design: hot-expert wgmma kernel with multicast activation slabs

Item classes (route_align, as in the experiments file): cold = experts with <= 16 routes, one
NT2/ROWS16 item each on the register kernel (cfg 5/30 class); hot = experts with > 16 routes, one
item of up to 256 routes per expert (two at the 257/264 tests). Hot items first.

Hot kernel, per CTA:

- Tile: 128 weight rows (two consumer warpgroups, m64 each) x N tokens (n32/n64/n128/n256 by
  item size) x K/2 (split-K 2: 32 k-steps of 64). GEMM1 takes 128 consecutive rows of the 8192
  (gate and up rows need not pair: the reduce pairs them by feature index as the SwiGLU kernel does
  today, lines 1049-1072). GEMM2 the same over W2's 4096 rows.
- Warps: 1 producer warp (TMA issue), 1 dequant warpgroup (128 threads: per k-step each thread
  converts 8 words = 64 weights with the existing bit trick and writes 8 x 16 B into the 128 B
  swizzled K-major A tile), 2 consumer warpgroups (4 x `wgmma.mma_async m64nNk16 bf16` per
  k-step, `wgmma.wait_group 1` so the next stage's barrier wait overlaps). 13 warps per CTA.
- Shared memory per stage: raw values 128 x 64 x 0.5 = 4 KB + scales 512 B (one bulk copy each
  from a new row-major repack: per expert, per 128-row tile, per 64-k step, contiguous); A bf16
  16 KB; B bf16 N x 128 B (n64 8 KB, n256 32 KB), TMA 2D box from a tensor map over the item's
  tokens, multicast to the 4 CTAs of the cluster (cluster dims (4,1,1) along the row-tile axis).
  n256: 52.5 KB per stage, 3 stages = 158 KB, 1 CTA/SM. n64: 28.5 KB per stage, 3 stages = 86 KB,
  2 CTAs/SM.
- Registers: consumers 128 accumulators (n256) + about 40 = about 168 (`setmaxnreg.inc`), producer
  and dequant warps about 56-64 (`setmaxnreg.dec`): 2 x 128 x 168 + 160 x 64 = 53K per CTA.
  n64: consumers about 72, 2 CTAs/SM.
- CTA counts: GEMM1 128 per hot expert (64 row tiles x 2 splits), GEMM2 64. Normal routing, model
  medians: bs128 about 8 hot experts -> 1,024 + 512 hot CTAs beside 15 cold experts x 256 = 3,840
  (GEMM1) and 1,920 (GEMM2) cold blocks; bs256 about 13 hot -> 1,664 + 832 hot CTAs beside about
  12 x 256 = 3,072 and 1,536 cold blocks.
- Launch: the hot kernel on a forked capture stream, the cold register kernel on the main stream,
  no PDL edge between them (they are independent: same x, disjoint partial rows); join before the
  SwiGLU reduce. With the KIND1 `griddepcontrol.wait` at block start (line 665) a PDL edge would
  serialize them, which is what the earlier two-launch split paid for.
- B operand sourcing: x rows of an item are scattered token rows, so either gather with
  consumer-side `cp.async` into the swizzled tile (2,048 x 16 B per k-step at n256, 16 per thread)
  or have route_align's order define a packed copy. For GEMM2, write `inter` in sorted-row order
  (row = position in `sorted_ids`) from the SwiGLU reduce so the hot GEMM2's B tile is a plain
  contiguous 2D box and the multicast works without a gather; the cold path's `stage_x` then reads
  by position (one-line change, line 721).
- Epilogue: FP32 partials in the existing `[z][route][8192]` layout so the SwiGLU reduce and the
  finalizer stay as they are (or in P2's compact layout).

Expected traffic and time (model medians, SM-side at 4.0-4.3 TB/s, HBM at 3.9):

| | bs256 GEMM1 | bs256 GEMM2 | bs128 GEMM1 | bs128 GEMM2 |
|---|---|---|---|---|
| cold weights (once each) | 227 MB | 113 | 284 | 141 |
| cold staging | 25 | 12 | 30 | 15 |
| hot weights (once each) | 246 | 122 | 151 | 75 |
| hot activation slabs (cluster 4) | 40 | 20 | 32 | 16 |
| partial writes | 50 | 25 | 12 | 12 |
| SM-side total | about 590 MB | 290 | 510 | 260 |
| HBM (unique + writes) | 522 MB | 260 | 447 | 228 |
| expected time | 140-165 us (now 220-234) | 70-80 (now 105-112) | 115-130 (now 150) | 58-65 (now 69) |

GEMM1+GEMM2: bs256 about 215-245 us (now 325-346), bs128 about 175-195 (now 219). That is
2.0-2.2x the judge floor at bs256 and 1.7-1.9x at bs128, short of the 1.4x in the brief: the cold
experts' unique bytes are irreducible and the cold path keeps its 4.3 TB/s SM-side pattern. Getting
to 1.4x needs the cold path on the TMA/bulk pattern too (tG won 4% at bs256) and the measured L2
ceiling from E2 to be above 5 TB/s.

BF16 rounding points, unchanged from today and to be kept: gate and up are rounded to bf16 from
the FP32 split sums before SiLU (SwiGLU reduce, lines 1067-1070; S=1 epilogue, lines 865-871);
SwiGLU's output is written bf16 (GEMM2's B operand is bf16 by construction); GEMM2's FP32 split sum
is rounded to bf16 before the norm (finalizer, line 959). wgmma accumulates FP32 like mma.sync; the
dequantized bf16 values are identical (value x scale x 2^-7 exact, global scale x 2^7 in the
epilogue); only the FP32 summation order inside a k-step changes.

Ruled-out register: this reopens "Hot-expert large token tiles (M1)" (docs/ruled-out.md:31). New
mechanism: wgmma with both operands in shared memory and the weight tile dequantized once per CTA,
instead of mma.sync with per-warp B fragments (which the shared-memory bandwidth argument in 3.1
shows cannot scale past NT=4) and 124-190-register accumulators. New evidence: the skewed-vs-normal
bench rows (same items, 6x fewer unique bytes, slower) and the traffic accounting in 2.2-2.4.

---

## 4. Q3: the small-T path

### 4.1 Where the time goes at T=1 and T=8

Standalone MoE profile (`muse_moe_w4a16/profiles/submission.a49578e5c30348bb.txt`, bs1: 3.5
experts, 66 + 33 MB): GEMM1 22.7 us, GEMM2 13.7, finalize 4.5, SwiGLU reduce 3.0, route_align 2.4;
kernel sum 46.3, GPU busy 45.2, judged 43.5 (about 4 us of replay cost subtracted). Layer profile at
decode 1: GEMM1 21.3, GEMM2 10.7, finalize 4.3, SwiGLU about 2-5 (listed at 5.0 on the full layer),
route_align hidden on the side stream (`kernel_fn`, layer lines 4957-4978).

Streaming time for the same bytes with the cold-L2 curve (docs/runs-moe.md:24): GEMM1 66 MB at
3.65 TB/s = 18 us, GEMM2 33 MB at 3.3 TB/s = 10 us. So at T=1:

- GEMM1 runs at 2.9 TB/s, GEMM2 at 2.4-3.1: about 5-6 us above streaming for the pair. The
  tuner shows doubling the bytes in flight does not help (cfg 18, S4 W8 D4 persistent: 32.6 us vs
  cfg 27's 31.0), so this is not memory-level parallelism. It is the fixed start and end of each
  grid: all 128-224 blocks start together and spend about 1 us on `items -> sorted_ids -> first
  loads` before the HBM pipe fills, and the last 4 stages of every warp drain with nothing queued
  behind them (another 1-2 us). About 2-3 us per GEMM kernel, 4-5 us per MoE call.
- Three dependent-launch hops: GEMM1 -> SwiGLU reduce -> GEMM2 -> finalize. With PDL each hop costs
  the primary's completion flush plus the dependent's first dependent loads, about 2-3 us, and the
  reduce and the finalizer are latency-bound kernels (224 KB and 512 KB of reads at T=1). GEMM2
  prefetches only its first D=4 stages (4.6 KB per warp) before `griddepcontrol.wait` (lines
  738-751); that covers about 25% of its 33 MB at T=1 when all its 448 blocks are resident, which
  is why PDL was worth 2-4 us and no more.
- route_align: 2.4 us, hidden by the side stream unless the side stream is the longer branch. At
  T=1 the side branch is router 5.2 + routing 2.9 + route_align 2.4 = 10.5 us against pooler 5.2 +
  pre-expert norm about 3.5 on main, so about 2 us of it is on the critical path at T=1.

Total recoverable at T=1: about 10-13 us of the 38-43 us MoE call (+0.02 to +0.026 on the score).
At T=8 (tuner: 81.3 us including about 5 us of replay cost, for 151 + 75 MB): GEMM1 about 44 us at
3.4 TB/s is at the ceiling (97%), GEMM2 about 24 us at 3.1 TB/s (87%); the rest, about 8-10 us, is
the three hops plus the two ramps. At T=16/32 the layer's GEMM times match streaming for its (smaller
than the MoE problem's) expert counts, so the chain overhead of 8-10 us is again the recoverable part.

The profile's 12.0 us SwiGLU at T=8 and 17.0 us finalizer at T=16 are not costs: with PDL a
dependent kernel is launched when the last wave of the primary starts (every primary block runs
`griddepcontrol.launch_dependents` at its start, lines 666 and 750), so its measured duration
includes waiting for that last wave: about one block lifetime, 12-16 us, exactly what is shown. The
true cost of each is 2-3 us of dependent latency after the primary completes.

### 4.2 Fixes and estimates

- P2 (section 6): SwiGLU applied by the last-arriving split of each (item, unit) inside GEMM1's
  epilogue (threadfence + atomic counter per unit, the CUDA threadFenceReduction pattern, loads with
  `__ldcg`), and GEMM2's last split writing bf16 y so the finalizer reads 8 KB instead of 4 x 16 KB
  per slot. Removes one hop everywhere: 2-3 us at T <= 32, about 5-8 us at T=128/256 (the reduce's
  25-50 MB of reads move into GEMM1's stream, so the net there is small). Not the ruled-out proj v5
  pattern (one block reducing everything at the end): here the reduction is per unit, spread over
  the kernel, and only the last units' reductions sit on the tail. BF16 rounding stays at the same
  points.
- P5: one persistent kernel for GEMM1 + SwiGLU + GEMM2 at T <= 32: a queue of GEMM1 units in item
  order followed by GEMM2 units in item order, per-item completion counters, GEMM2 units spin on
  their item's counter. Deadlock-free because every unit's dependencies are earlier in the queue and
  are always held by running CTAs; in practice no spinning happens because GEMM2 item i depends only
  on GEMM1 item i, which finished long before. Removes the second hop and overlaps GEMM1's drain with
  GEMM2's fill: 5-7 us at T=1 and T=8, 4-6 at T=16, 3-5 at T=32. The finalizer stays a PDL kernel
  (fusing it needs a per-token two-level reduction; a later step).
- P4: at T=1 the untested axis is W8 with D8 (more bytes in flight per block when there are only
  1-2 blocks per SM): add `(1,1,8,2,8,8,5,0,1)`, `(1,1,1,1,8,8,0,0,1)` (S=1: fused SwiGLU epilogue,
  no reduce kernel) and `(2,1,1,4,8,8,0,0,1)`; the tuner picks. 2-4 us at T=1 if the ramp model is
  right.
- Different K-split counts, 2 blocks per SM, deeper D at W4: already covered by the tuner's table
  (D=8 configs 3 and 9, 6 blocks/SM for W4 D4); no gain expected.

---

## 5. Q4: pure overhead in the layer's MoE path

- Empty blocks inside the grid. `max_items` is the host bound `count + numel/rows + 1` (line 401;
  layer 771-772), so the launch is 41K blocks at T=256 for 9.6K real, 10.3K for 2.3K at T=128, 16.6K
  for 4.5K at T=32, 12.5K for 3.1K at T=16, 10.5K for 2.0K at T=8. Because blocks are dispatched in
  (x, y, z) order, the empties sit between the split planes and each one occupies a block slot
  (registers and 33 KB of smem are allocated at dispatch) for a `griddepcontrol.wait` plus one
  dependent load before exiting. The persistent twins (no empties) tie the non-persistent configs
  within 1 us at bs8, so the cost is probably small, but it has never been isolated. Fix: index
  work by linear block id over the real `num_items[0]` so all real blocks come first and the empties
  drain at the end (about 15 lines in `decode`, lines 675-690). P6.
- Dead blocks in the reduce and finalizer grids: `w4_swiglu` launches `dim3(numel, 8)` (line 1209),
  16,384 blocks at T=256 of which about 85% exit on `local_ids[p] < 0`; the finalizer launches per
  token (line 1133) and skips dead slots. About 1 us at T=256; index the reduce by sorted position
  (P7).
- Partial buffers sized by all routes: `part` is `(splits, numel, 8192)` FP32 (line 1494), 268 MB at
  T=256 and 134 MB for GEMM2, of which 50 and 25 MB are written. In a graph these are fixed
  allocations, so the cost is footprint, not time; the compact `[S][item][unit][rows][32]` layout
  in P2 removes it.
- route_align's work-list build (lines 142-165) is one warp with serial per-expert loops of at most
  16 iterations; 2.4-4.2 us total, hidden on the side stream except the about 2 us at T=1 noted in
  4.1. Fusing it into GEMM1's prologue at T <= 8 (each block recounts the 8-64 routes itself) buys
  1-2 us at decode 1 only (P7).
- `local_ids` handling costs nothing beyond the dead blocks above; the `.to(int32)` and
  `.to(float32)` in `_route_meta` (layer 4129, 4150) are no-ops when the routing kernel already
  emits those dtypes, which the profile's kernel list suggests.
- The finalizer at T=16 (17 us) and the SwiGLU at T=8 (12 us): PDL wait inflation, see 4.1; the real
  cost is 2-3 us each, addressed by P2.

---

## 6. Ranked list

Savings are per MoE call at that T (both layers of the load see it), in us; score deltas use the
sensitivities in the header.

### P1. Hot-expert wgmma kernel (dequant once per CTA, multicast activation slabs), T >= 128

- Mechanism: section 3.3. Experts with > 16 routes go to a warp-specialized kernel: TMA bulk copies
  of raw NVFP4 + scales, a dequant warpgroup writing bf16 tiles to swizzled smem once per CTA, two
  consumer warpgroups on `wgmma m64nNk16`, cluster-of-4 multicast of the token slab. Cold experts
  stay on the register kernel as one 16-row item each.
- Files: new kernel and repack beside `w4_gemm_kernel` in `_W4_SRC` (both submissions); the class
  split and item lists from `experiments/tma_and_large_tiles.py` (`ItemLists`, `route_align`,
  `_w4_launches`); `_w4_gemm1/_w4_gemm2` launch on a forked capture stream; tuner candidates.
- Savings: T=1/8/16/32 0; T=128 30-45; T=256 90-130. Score +0.055 to +0.075.
- Confidence: medium. The traffic model fits the four big kernels within 10% and the skewed-routing
  rows show time follows SM-side bytes; the uncertainty is whether the hot CTAs' 1-per-SM occupancy
  and the cold path co-running reach 4 TB/s together, and the L2 ceiling (E2).
- Effort: 5-6 days for an expert (the attention kernel already has wgmma descriptors, mbarriers and
  warp specialization to copy from; tensor maps and multicast are new in this repo).
- Risk: medium-high (swizzle and descriptor bugs, bank conflicts in the dequant stores, hot CTAs
  scheduled late creating a tensor-bound tail).
- Cheapest experiment: E1 (2 hours). If one hot expert with 16 items is not at least 2x faster than
  16 one-item experts on the current kernel, build P1; if it is, the limit is HBM reuse and the
  cheaper P3 suffices.
- Reopens M1 with the new mechanism and evidence stated in 3.3.

### P2. SwiGLU in GEMM1's last-split epilogue, bf16 y from GEMM2's last split, compact partials

- Mechanism: section 4.2. Per (item, unit) counter, `__threadfence` + `atomicAdd`, the last split
  sums the others' partials for its 16 gate + 16 up features x rows, applies bf16 rounding and SiLU,
  writes `inter`; GEMM2 likewise writes bf16 y (rounded as Marlin's cache3) so the finalizer reads
  8 KB per live slot. Counters zeroed by route_align. Removes `w4_swiglu_kernel` at every T.
- Files: `w4_gemm_kernel` epilogue (lines 848-885), `route_align_kernel`, `w4_finalize_kernel`
  (lines 946-953), `_w4_gemm1/_w4_gemm2` buffers, both submissions.
- Savings: T=1 2.5; T=8 2.5; T=16 3; T=32 3; T=128 5; T=256 8. Score about +0.02.
- Confidence: medium-high (one hop per call is 2-3 us in every profile; the atomic pattern is
  standard). Effort 1.5 days. Risk low (ordering: fence before the atomic on the writer, `__ldcg`
  loads on the reader; the proj v5 ruled-out entry is a different shape, see 4.2).
- Cheapest experiment: `_graph_time` of the MoE call with `w4_swiglu` replaced by a no-op launch
  (numerics ignored): the upper bound of the saving per T. 1 hour.

### P3. Raw-weight multicast across same-expert items in the TMA kernel, T >= 128

- Mechanism: `w4_tma_kernel` (experiments file, lines 982-1163) launched with cluster dims (1,4,1)
  along the item axis; when the 4 items belong to one expert, rank 0's producer issues the two bulk
  copies per stage with `.multicast::cluster` and the empty barriers are cluster-scoped; otherwise
  each CTA loads its own. route_align orders multi-item experts first and pads to cluster boundaries.
- Files: experiments kernel moved into `_W4_SRC`, `route_align` item order, tuner candidates.
- Savings: T=128 25-40; T=256 50-80 (weight re-reads only; staging and dequant stay). Score +0.035
  to +0.055. Superseded by P1; worth doing first only if P1 cannot start within the week.
- Confidence: medium (the TMA kernel tied or won at T >= 128; the gain scales with items per
  expert, 1.5x at these sizes). Effort 2-2.5 days. Risk medium (cluster lockstep, the TMA
  consumer's per-stage barrier round trip that lost 3-11% at small T).
- Cheapest experiment: E1, then a loads-only multicast variant (mode 7 with the XOR consumer) to
  measure the SM-side traffic reduction before wiring the GEMM. 1 day.

### P4. T=1 configs with W8 D8, and S=1 D8 with the fused SwiGLU epilogue

- Mechanism: section 4.2. Three table entries; the tuner decides.
- Files: `W4_CONFIGS` in both submissions.
- Savings: T=1 2-4 (others 0). Score +0.004 to +0.008. Confidence low-medium (the ramp model; the
  tuner's T=1 spread is only 2 us across configs). Effort 1 hour plus one check (build +3 kernels).
  Risk none (tuner-gated).
- Cheapest experiment: the check run itself.

### P5. One persistent kernel for GEMM1 + SwiGLU + GEMM2 at T <= 32

- Mechanism: section 4.2, on top of P2 (item-ordered unit queue, per-item counters).
- Files: `w4_gemm_kernel` P=1 path (lines 668-690, 760-772, 886-898) generalized to two kinds;
  `_experts_run`.
- Savings beyond P2: T=1 5-7; T=8 5-7; T=16 4-6; T=32 3-5. Score +0.025 to +0.04.
- Confidence: medium (removes one hop and overlaps the two ramps; the persistent twins tied today,
  which says the queue machinery itself is free). Effort 3-4 days. Risk medium (deadlock reasoning,
  cross-kind register pressure in one kernel, the GEMM2 units' smem for 1 row per item at T=1).
- Cheapest experiment: a PDL-off profile (opts `pdl=False`) to read the true hop costs, plus
  `_graph_time` of GEMM1 and GEMM2 captured on two streams with no dependency (garbage numerics)
  to measure the overlap ceiling. 2 hours.

### P6. Dense block indexing (real blocks first, empties at the end)

- Mechanism: section 5, `decode` lines 675-690 and the early exit at 668-672.
- Savings: 0-4 us per GEMM at T=8-32, 0-3 at T=128/256; score 0 to +0.02. Confidence low (the
  persistent twins, which have no empties, tie). Effort 1 hour plus one check. Risk none.
- Cheapest experiment: the check run; compare GEMM times against 0e28c26271c94570's tuner lines.

### P7. route_align in GEMM1's prologue at T <= 8; reduce/finalizer grids over local routes only

- Savings: T=1 1-2; T=256 about 1. Score about +0.004. Effort 0.5-1 day. Risk low.
- Cheapest experiment: time the side branch and the main branch at T=1 separately (events in the
  tuner) to confirm route_align is on the critical path by about 2 us.

Not recommended: a last-arriving-CTA whole-token finalizer (the proj v5 mechanism in
docs/ruled-out.md:16: one CTA reading 512 KB at the tail, slower than the 8-block cluster); larger
token tiles on mma.sync (ruled out, and the shared-memory bandwidth bound in 3.1 says why); the
TMA kernel at T <= 32 (ruled out, lost 3-11%); more K splits or blocks per SM at small T (tuner
covers them).

If P1, P2, P4, P5 and P6 all land at their midpoints the MoE path gives about +0.13 to +0.16 on the
layer score (1.40 -> about 1.53-1.56). The remaining distance to rank 50 (2.135) cannot come from
the MoE: at T=1 the MoE call is about 38 us of a 117 us step, and at bs256 the design above is
still 2x the judge floor.

---

## 7. Caveats

- The layer's routing ("dist soft") differs from the MoE problem's: SGLang's Marlin pair takes
  41/45 us at decode 1 in the layer against 50 us for 3.5 experts in the MoE problem, 136/93 us at
  decode 32 against 201, and 513-530 us at verify 32 against 400. Per-shape unique-expert counts in
  the layer are unknown; the big-shape estimates use the tuner's synthetic distribution and the
  small-shape estimates use the standalone profiles.
- Every per-kernel time from the layer profile includes PDL wait (docs/runs-moe.md:85 and tools/dl_moe_split.py:8); the tuner's
  TUNE lines (graph replay, about 5 us of fixed cost) are the cleaner per-GEMM numbers and were used
  wherever both exist.
- The SM-side ceiling of 4.1-4.9 TB/s is inferred from the kernels, not measured; E2 measures it.
- mma.sync tensor cycles (about 8 per m16n8k16) and the 14-instruction dequant count are from the
  source and public Hopper rates, not from a SASS dump (Docker was not running on this machine; the
  repo's `tools/moe_compile_w4.sh` produces the dump).
