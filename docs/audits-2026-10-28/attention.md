# Attention kernel audit (muse_attention_decode kernels as used by muse_decode_layer)

Date: 2026-10-28. Repo at commit b0aa793, read only. Evidence: the sources named below, the
stamps and bench lines in `muse_attention_decode/profiles/*.log`, `docs/runs-attention.md`,
`docs/ruled-out.md`, `docs/paper-backlog-2026-10-27.md`, the layer profile
`muse_decode_layer/profiles/submission.0e28c26271c94570.txt`, and two things measured here:

- An sm_90a compile (CUDA 12.8.1 in Docker, same flags as `tools/attn_ws_quick.sh`) of
  `attn_kernel.cu` and `attn_ws.cu`, with the SASS split at the clock reads. Outputs:
  `/private/tmp/claude-501/-Users-gretchengabo-GitHub-okc/ec960068-b6fb-4760-a2de-cd2a92f6dafb/scratchpad/sass/`
  (`fused.sass`, `fused.ptxas`, `ws.sass`, `ws.ptxas`, `analyze.py`).
- The PTX ISA / CUDA programming guide text on `cp.async.bulk`, mbarrier tx accounting, proxy
  fences, `wgmma` operand layouts and FP8 conversions, fetched today from
  https://docs.nvidia.com/cuda/parallel-thread-execution/ and
  https://docs.nvidia.com/cuda/cuda-programming-guide/ (quotes below are from those pages).

Score frame: the layer is at 1.403x; the sensitivities given (score per 10 us saved) are used
throughout. Every attention item in this report together is worth about +0.10 to +0.12 on
the layer score. That is real, but it does not reach 2.135x on its own; the other kernels of
the layer must move too.

## 0. Verdict in one paragraph

The sliding shapes (weight 46) are bound by fixed per-launch cost, not by bandwidth or math:
the q_len 1 cluster kernel spends about 9 of its 11.6 stamped microseconds in phases whose
cost tracks dependent cold-memory round trips and the size of cold code, and only 1.7 us in
its load loop. The kernel is 98.6 KB of SASS with 17 inlined `sincosf` copies, and the merge
phase executes 10.8 KB of straight-line code for about 1k cycles of arithmetic while the
stamps charge it 5.9k cycles. Shrinking the code, hoisting every cold load to the top,
attending the new token under the first tile's latency instead of after the loop, and
removing the second launch on the q_len 8 path are the cheap, high-confidence wins. On full
context the ws kernel is coupled at about 1.73k cycles per tile on both the loader and math
sides; the conversion unit (MUFU/F2FP at 16 lanes per clock per SM) is 89% subscribed at the
gather ceiling, so the loader-side wins (row TMA, FP8 P.V with a transposed V) and the
math-side wins (part of the exponentials on the FMA pipe) are both needed. Plain 1-D
`cp.async.bulk` copies cannot feed `wgmma`'s K operand (layout proof in 2a); the viable form
is the tensor-map TMA with a one-row box, which the backlog already lists as A4. For q_len 1
on full layers the fused kernel's loop runs at 4.7k cycles per 64-key tile against the ws
loaders' 1.1 to 1.7k, so the right move is a q_len 1 math path behind the ws loader design,
with one consumer warpgroup so two CTAs fit per SM.

## 1. Sliding window, q_len 1, batch 1 (14.5 us in the layer; 13.85 us scored standalone; 10.3 us kernel in the standalone profile)

### 1.1 What the kernel does, in order

Routing: `muse_decode_layer/submission.py:4288-4295` (`_fused_splits`) gives sliding layers
(16, 8, 4) splits for batch (1, 16, 32) with `cluster=True`; `_attn_fused` (4298-4307) calls
`attn_fused` (1950-1996), which launches `launch<1, 1, 4, 1, true>`. `Cfg<1,1,4,1>`
(`attn_kernel.cu:172-184`) is 8 warps: `TP=4` warps per KV head, each one 16-token chunk of
every 64-token tile. Grid = splits x batch, cluster dims = splits
(`attn_kernel.cu:946-959`; `NonPortableClusterSizeAllowed` set at 941, so the 16-CTA cluster
at batch 1 is already non-portable). Per CTA at batch 1: `per = ceil(2047/16) = 128` keys,
rounded to 64 (`attn_kernel.cu:279-283`), so 2 tiles, 32 KB of K+V.

Critical path of one CTA (`attn_kernel.cu` line numbers):

1. 267-285: `stamp(0)`; gate row prefetch (`load_gate`, 272-274); `*p.k_scale`, `*p.v_scale`
   (275-276); `seq = p.seq_lens[b]` (277). All cold (L2 flushed before every replay).
2. 322-325: `stage_pt(0..5)`: cp.async of 6 tiles of page entries. Addresses depend on
   `seq`, so this is the second dependent trip.
3. 331-385: Q prep. 8 `bfloat162` loads per thread from `qkv` (cold), RMS norm, then
   `sincosf(pos * p.inv_freq[...])` (373) 8 times per thread, where `p.inv_freq` is a cold
   load inside the `if (p.rope)` branch (369) that cannot be hoisted above the branch by the
   compiler. For `L == 1` the `half` loop (337) runs twice for the same row: the second copy
   is dead work and dead code (8 more `sincosf`, 8 more loads).
4. 391-456 (last split only): new K/V rows loaded (cold), normalized, rotated (one more
   `sincosf`), rounded, written to the cache at `p.cache_loc[tok]` (446, another cold
   dependent load before the store), and staged in `newb` (448-453).
5. 458-460: `cp_wait<0>; __syncthreads(); stamp(1)`: page entries landed.
6. 461-468: prologue `issue(0..2)` (only 2 tiles exist), `stamp(2)`.
7. 750-776: loop: `cp_wait<STAGES-2>; __syncthreads();` issue tile `it+3`, stage page
   entries `it+6`, `chunk()` per warp; `stamp(3)` after `cp_wait<0>; __syncthreads()`.
8. 778-780: new-token tile: `chunk(newb, 0, seq, seq + L - 1)` on the `tp == 0` warps only
   (2 of 8 warps), last split only; `stamp(4)` (812).
9. 813-876: partials to smem, `__syncthreads`, TP merge (4 partials per row, 2 rows per
   warp), `__syncthreads`; `stamp(5)`.
10. 884-900: push merge: each warp stores its rows' partials into the owner CTA's `recv`
    via `cluster.map_shared_rank` (893), then `cluster.sync()` (900); `stamp(6)`.
11. 902-911: `merge_row` (206-240) for `ROWS/S = 1` row per CTA, 16 partials, by warp 0;
    norm-gate; bf16 store; `stamp(7)`.

### 1.2 Where the cycles go (stamps of the current kernel build)

The stamps in `profiles/bench.fa01e4d1a9f54b0f.log` are from commit 9a15913, whose kernel
body is the current one (3b84d10 changed only the host routing of the new-token tile and
line lengths). Stamp placement in 9a15913: lines 265, 454, 462, 770, 806, 872, 895, 907,
which are the current file's 267, 460, 468, 776, 812, 878, 901, 913. Cycles at 1.98 GHz;
"last" is split 15 (does the new-token work), "first" is split 0.

| phase | last CTA | first CTA | SASS bytes on the path (this compile) | what sets the cost |
|---|---|---|---|---|
| 0-1 prologue | 7,039 | 6,025 | 39.2 KB (2,508 instr, 56 LDG, 2 calls) | seq -> page entries (2 cold trips); q loads -> inv_freq loads -> sincosf chain; 17 inlined sincosf |
| 1-2 prologue issue | 1,038 | 1,341 | 10.5 KB | issue of 2 tiles |
| 2-3 loop (2 tiles) | 3,305 | 3,986 | 16.7 KB | first-tile latency plus two chunk computes |
| 3-4 new-token tile | 3,343 | 824 | 6.5 KB | one chunk on 2 warps; the first CTA pays 824 cycles to skip it |
| 4-5 partials + TP merge | 716 | 1,773 | 1.9 KB | two block barriers |
| 5-6 push + cluster.sync | 1,701 | 3,233 | 3.4 KB | everyone waits for the slowest CTA (the last split) |
| 6-7 merge + gate + store | 5,922 | 5,907 | 10.8 KB (691 instr, 6 calls, 72 LDS, 43 SHFL) | about 1k cycles of work in 5.9k |
| total | 23,064 (11.6 us) | 23,089 | 98.6 KB for the kernel | |

Older builds confirm the attribution: in 5d1187f9 (Q prep after the page-table wait) the
same Q prep sat in phase 1-2 and cost 6.5k cycles; moving it under the page-table wait
(c35deb23) saved only 1.6k, which says it is a dependent-load chain plus code, not overlap
that was missing. The push merge (8499f13) cut phase 6-7 from 8.6k to 5.9k; the remaining
5.9k is the same in both CTAs and the same at b16 and b32 within 1k cycles, with one warp
doing 16 iterations of a shuffle, an LDS.64 and two FMAs.

Reading: three things dominate, and none is bandwidth.

- Dependent cold round trips: seq -> page entries -> K/V rows is three trips (about 670
  cycles each by the gather chase: 1,343 for entry+row, 741 for a row alone), and the Q path
  adds q rows -> inv_freq -> sincosf, the new-token path adds cache_loc -> store. The
  measured prologue (6-7k cycles) is about 3x the irreducible two trips before the first
  K/V request.
- Cold code. The judge flushes L2 before every replay, so instruction lines come from HBM
  on first execution. The per-phase cycles track the per-phase code bytes when the phase
  has little memory work: the merge executes 10.8 KB for 5.9k cycles, the new-token chunk
  6.5 KB for 3.3k, and the first CTA pays 824 cycles for a taken branch over the new-token
  code. This is the one claim in this report that rests on correlation; the confirming
  experiment is cheap (1.5 below).
- Serialization at the end: the last split's extra tile (3.3k) is paid by every CTA at the
  cluster barrier, then the merge runs on one warp per CTA.

The layer profile shows 14.5 us for this kernel against 10.3 us in the standalone profile
and 11.6 us by stamps; the extra is launch and drain inside the graph (the standalone judge
scores 13.85 us for the same 10.3 us kernel), so about 3 to 3.5 us per launch is the price
of each kernel on these shapes. That is also why the q_len 8 path's second launch matters.

### 1.3 Design that reaches about 5-7 us at b1 and 8-11 us at b16/b32

Keep the one-launch cluster structure (it already exists and already uses a 16-CTA
non-portable cluster at b1), and remove the fixed costs:

1. Issue every independent cold load at the top, before `seq` is used: `k_scale`,
   `v_scale`, the gate row, the q row (once, not twice), `inv_freq` (unconditionally, not
   inside `if (p.rope)`), and on the last split the new k/v rows and `cache_loc[tok]`. Then
   the prologue is one cold trip for all of them in parallel, the page-entry trip, and the
   K/V trip. Expected: 6-7k -> about 3k cycles.
2. Shrink the code to roughly 20 KB: `#pragma unroll 1` around the RoPE loops (one
   `sincosf` body instead of 17), drop the dead `half == 1` work for `L == 1`, one `chunk()`
   instantiation instead of two inlined copies, `#pragma unroll 2` in `merge_row`. Static
   check: SASS size and `sincosf` copy count from `cuobjdump` (the `analyze.py` in the sass
   folder does this).
3. Attend the block's new token first, under the first tile's latency, not after the loop:
   `newb` is ready at the end of the prologue, and the math warps idle until tile 0 lands.
   The online softmax does not care about order. Removes phase 3-4 (3.3k cycles) from the
   critical path of every CTA (through the barrier).
4. Merge: keep the push, but complete it on the owner's mbarrier with `st.async` and
   `expect_tx` (PTX 9.7.10.12: "upon completion of the asynchronous operation, a
   complete-tx operation will be performed on the mbarrier object specified by the operand
   mbar"; the barrier must be in the destination CTA and `%cluster_nctarank > 1`) so an owner
   waits only for its own bytes instead of the whole cluster at `cluster.sync`. One cluster
   barrier after `mbarrier.init` is still needed (arrive early, wait right before the push).
   With the code diet the merge itself is about 1k cycles.
5. Optional, b32 only: 6-8 stages or 8 splits once the fixed costs are gone (`_fused@s=8`
   lost at b32 in bench c2a8b716 only because the fixed costs grow with S).

Estimate (kernel time in the layer profile's terms, launch included): b1 14.5 -> 7-8 us,
b16 15.8 -> 9-10 us, b32 21.4 -> 13-15 us. Reasoning: prologue 3k + issue 1k + loop (b1
2.5k, b16 5k, b32 10k) + merge 2.5k + launch 3 us. Confidence medium: the code-size
mechanism is inferred, the dependent-trip arithmetic is measured.

The alternative the question raises, 32-64 CTAs per request with a global-counter merge:
`cudaLaunchKernelEx` cluster size tops out at 16, so this needs a last-arriving-CTA merge
through global memory (partials, `__threadfence`, atomic counter, the last CTA merges). At
S=16 the last CTA reads 67 KB of partials from L2 (about 0.6 us) after one atomic round
trip (about 0.4 us): comparable to a fixed cluster merge, and it lifts the cluster-size
cap. Do not let CTAs spin on the counter: the sliding test with batch 9 and 16 splits is
144 CTAs, and with 2 CTAs per SM resident a spinning design can deadlock. This reopens the
ruled-out "scratch-buffer split-K, last block reduces" pattern from the projection problem;
the new evidence is that the cluster merge here costs 7.6k cycles (push + barrier + merge)
against the 2.5k an L2-tail merge would cost, which was not the ratio on the projection.

### 1.4 Sliding, q_len 8 (ws kernel + ws_combine)

Layer profile: b1 11.9 + 3.3 us, b16 13.9 + 2.7, b32 18.4 + 3.3, plus a launch gap between
the two kernels. Routing `_ws_splits` (`muse_decode_layer/submission.py:4317-4322`): 16, 8,
4 splits; 2, 4, 8 tiles per CTA. The combine is a separate kernel
(`attn_ws.cu:1164-1209`, launched at 1334-1336 and in the layer at 3135-3137) of
`batch * 8` blocks of 512 threads.

Stamps (bench a8a4666a91d54d68, var 0; var 16 is not stamped yet), cycles per tile per warp:

| shape | math: wait-data / qk / softmax / pv / tail / prologue | loader: wait-free / issue / entries / wait-own / convert |
|---|---|---|---|
| b1 sliding (2 tiles) | 3,881 / 185 / 1,036 / 330 / 1,591 / 3,695 | 597 / 3,630 / 1,294 / 18 / 821 |
| b16 sliding (4 tiles) | 2,848 / 156 / 1,102 / 330 / 795 / 2,792 | 343 / 3,171 / 813 / 31 / 1,041 |
| b32 sliding (8 tiles) | 1,450 / 186 / 1,077 / 362 / 268 / 1,859 | 269 / 2,179 / 501 / 30 / 950 |

At b1 the math warps wait 3.9k cycles per tile for the loaders, whose per-tile cost (6.4k)
is the first tile's cold chain (entries, rows) amortized over two tiles plus the issue of
32 `LDGSTS` per warp. The prologue (3.7k cycles) is the Q prep (`attn_ws.cu:613-675`): 16
`sincosf` per thread in a fully unrolled loop (646-659; the SASS has 20 inlined copies and
85 KB of code), the new-token staging (681-728), and cold loads. The tail (1.6k) writes the
partials to global (1126-1151). Then the combine launch reads them back (3.3 us plus the
gap).

Design (backlog A3, with the smem problem solved by a sliding-specialized instantiation):

- Sliding launches use `NS=4, DA=2` instead of `7, 3` (2-8 tiles per CTA never fill 7
  stages; at b32 the loaders spend 2.2k cycles per tile issuing, so the ring is not the
  limit). `Smem<4,2>` is 120 KB, which leaves room for a 33 KB receive buffer
  (`ROWS/S` rows x S partials x 264 B = 128 x 264 B for any S), so no stage reuse and no
  second barrier.
- Cluster launch with cluster dims = splits (16 at b1 non-portable, 8, 4). Push partials
  with DSMEM stores (or `st.async` with `complete_tx` on the owner's mbarrier), one barrier,
  owners merge `ROWS/S` rows with one warp per row from local smem, apply the norm-gate
  (gate rows prefetched at kernel start), store bf16. The combine kernel disappears.
- Prologue diet as in 1.3: hoisted cold loads, one `sincosf` body, no dead code.
- Attend the new-token tile first (under tile 0's latency) instead of last (1116-1122).

Estimate (layer terms): b1 16 (11.9 + 3.3 + gap) -> 8-9 us; b16 17.5 -> 10-11; b32 22.5 ->
13.5-15. Reasoning: prologue 2k + loop (b1 5k, b16 9k, b32 16k; the first tile's chain is
about 2.6k and each further tile about 1.7k) + merge 2.5k + launch 3 us. Confidence medium.
Risk: 16-CTA clusters of 384-thread, 150 KB CTAs must co-schedule in one GPC; check
`cudaOccupancyMaxActiveClusters` in the self-test and fall back to the two-launch path.

## 2. Full context

Measured: q_len 1 fused b16 133.1 + 4.7 combine, b32 249.6 + combine; q_len 8 ws b16 185.3,
b32 353.7; gather ceiling 181 us for 655 MB with 256 CTAs (2 per SM) and 219 us with 128
CTAs (1 per SM, `GATHER b=32 rowb=128 pref=1 splits=4 u=8 219.4us` in
`profiles/bench.8cb557afaeab4e08.log`). The 1-CTA-per-SM ceiling matters: the ws kernel
cannot run two CTAs per SM (115 registers x 384 threads), so its floor is nearer 200-220 us
than 181.

### 2.1 Where the ws kernel's time goes at b32 full (stamps, bench a8a4666a91d54d68, var 0)

Math warp per tile: wait-data 276, qk 186, softmax 919, pv 349 = 1,730 cycles. Loader warp
per tile: wait-free 175, issue 660, entries 84, wait-own 29, convert+release 789 = 1,737.
Both sides sit at 1.73k per tile, 312 tiles per CTA: 540k cycles = 273 us of the 333 us
(the rest is prologue, tail, uneven lengths, launch). The sides are coupled through the
ring (small wait-free, small wait-data), so either one alone sets the pace only after the
other is fixed. Var 16 (FP32 softmax, now routed) measured 8% slower than var 0 on the
judge, so its math side is at about 1.8-1.9k.

Budget for the conversion/transcendental pipe (MUFU and F2FP, 16 lanes per clock per SM
on sm_90, "All other type conversions: 16" in the CUDA guide's throughput table), per
64-key tile per CTA:

| consumer | lane-ops per tile | SM cycles |
|---|---|---|
| softmax `ex2` (256 threads x 32) | 8,192 | 512 |
| P pack to f16 (`F2FP.F16.F32.PACK_AB`, 256 x 16) | 4,096 | 256 |
| loader V convert (`F2FP.F16.E4M3.UNPACK_B`, 4,096 per tile) | 4,096 | 256 |
| total | | 1,024 |

At the 181 us ceiling a tile has 1,150 cycles, so this pipe is 89% subscribed before any
stall. This is the quantitative reason the kernel sits at 1.73k and not at 1.15k, and why
"integer conversion" (ALU instead of F2FP, var 1) did not lose on pipe grounds but on
loader issue slots (it was 3 ALU ops per pair in warps that were the pace-setters). The two
levers that cut this budget are FP8 P.V with V transposed instead of converted (removes
256) and part of the exponentials on the FMA pipe (var 80: a quarter, removes 128;
`ex2_poly`, `attn_ws.cu:227-237`). Both are already written (var 80/112) or specified (N2).

### 2.2 (a) 1-D bulk async copies for the row gather

PTX facts (9.7.10.28.4.1, fetched): `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [src], size, [mbar]`;
"size must be a multiple of 16", "The addresses dstMem and srcMem must be aligned to 16
bytes"; no minimum beyond 16, so 128-byte rows are fine and so are 64-byte half rows. It is
a per-thread instruction without `.sync/.aligned`, so 32 lanes can each issue their own
copy, and "If multiple threads update the transaction count, the expected transaction
count will be the sum of the updates." The tx-count range is +-(2^20 - 1) per phase, so a
16 KB tile is far inside it, and `complete_tx` may land before the `expect_tx` of the same
phase as long as the pending-arrival count keeps the phase open. Completion: "The
cp{.reduce}.async.bulk operations are performed in the asynchronous proxy", "The completion
of a cp{.reduce}.async.bulk operation is followed by an implicit generic-async proxy fence",
and after `mbarrier.try_wait` (acquire by default) the bytes are visible to the waiting
thread's generic loads with no `fence.proxy.async`. A later `wgmma` read of the same smem is
async-proxy to async-proxy in the same CTA, ordered by complete_tx (release, cluster scope)
-> try_wait (acquire) -> program order, so no proxy fence is needed there either; CUTLASS's
sm90 TMA mainloops do exactly that. Generic writes that `wgmma` must see (the converted V, or
a transposed V) still need the writer-side `fence.proxy.async` before the arrive, which is
the fence the repo rule protects, and it stays. Destination in the issuing CTA's own smem is
allowed with a non-cluster launch ("implicit cluster launch with default dimension 1x1x1";
`.shared::cta` as destination needs PTX 8.6, CUDA 12.8's 8.7 has it). Throughput: no NVIDIA
document states a small-copy issue rate; the programming guide only says "issuing as few
bulk copies with as big a size as possible results in the best performance".

The layout problem, which decides the design: `wgmma` reads e4m3 operands from smem only
in the K-major canonical layouts (9.7.17.5.1.2; "The transpose operation is only supported
for the wgmma.mma_async variants with .f16/.bf16 types"). Those layouts interleave 16-byte
chunks across 8 rows (no-swizzle core matrices: 8 tokens x 16 B contiguous, which is what
`kmaj()` at `attn_ws.cu:285-289` builds, LBO 128, SBO 512) or permute the chunks inside a
row by an XOR of the row index (32/64/128-byte swizzle). A 1-D bulk copy writes its bytes
contiguously, so one 128-byte (or 64-byte) copy per row cannot land in either family: the
no-swizzle layout needs a token's four chunks 128 B apart, and the swizzled ones need the
chunks inside the row reordered per row. Permuting tokens across rows does not help,
because every row 2-7 of a swizzle atom needs a nontrivial chunk permutation, and permuting
the K index per row breaks the dot product. Two ways out:

- The tensor-map TMA with a one-row box (`cp.async.bulk.tensor.2d` over the cache viewed as
  [rows x 128 B], box = 1 row, `CU_TENSOR_MAP_SWIZZLE_128B`): the TMA unit applies the
  swizzle from the destination address bits, so each row lands as row `t` of an 8-row
  SW128 atom and both heads' Q.K descriptors address it (head 1 at +64 B inside the atom,
  the same start-address stepping CUTLASS uses for k-slices). This is the backlog's A4, and
  it is the only bulk form that feeds `wgmma` directly. One copy per K row and one per V
  row: 128 TMA requests per tile instead of 32 `LDGSTS` warp instructions. The unknown is
  the TMA unit's rate on 128-byte boxes; the loads-only build (`mode 1`) answers it in one
  bench. The page entry is still a dependent `__ldg` per row (prefetched 4 tiles ahead as
  now), fed as the row coordinate.
- Plain 1-D copies into a staging ring plus a loader-side re-layout with LDS/STS into
  `kmaj` order: this keeps the generic-write path and its fence and adds 16 KB of smem
  traffic per tile; it only pays if LDGSTS issue, not smem traffic, is the loader's limit.
  Prefer the first.

For the q_len 1 path (2.4 below) 1-D copies are fine: `mma.sync` fragments are loaded by
the math warps from any layout; use a 144-byte row pitch so eight consecutive rows' chunk
0 fall in different banks (a linear copy cannot apply the XOR swizzle `k_chunk`/`v_chunk`
use today).

Expected effect on the ws loaders: issue 660 -> about 150 cycles per tile (4 instructions
per lane per tile for one warp), conversion unchanged at about 790, so loads-only drops
from 301 us toward about 230 us; the full kernel moves only if the math side also moves
(it is at 1.73-1.9k per tile). With item 2.3 as well, loaders at about 400-500 per tile.

### 2.3 (b) FP8 P.V on wgmma with V transposed in smem (backlog N2)

Requirements, from the fetched PTX: `wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3`
takes A from registers ("four .b32 registers, with each register containing four .e4m3
elements") or smem, B from smem, both K-major only. For P.V the K dimension is the token
index, so B = V must be stored as [dim][64 tokens contiguous]: a 64 x 64 byte transpose per
head per tile (8 KB per tile for both heads), in a K-major layout, which here means 8-dim x
16-token core matrices or a swizzled 64-byte-row atom. A-fragment layout for e4m3
(Figure 155): register 0 holds row g, k 4c..4c+3; register 1 row g+8, same k; registers
2-3 the same at k+16. The f32 accumulator of Q.K gives each thread columns {2c, 2c+1, 8+2c,
9+2c, ...}, so P's packing needs a permutation; the cheap way (FA3's) is to write V's rows
in the matching permuted token order during the transpose, which is free.

Transpose cost versus conversion cost: Hopper has no 8-bit `ldmatrix`/`stmatrix`/`movmatrix`
(`.b8` and `.m16n16` are sm_100a; on sm_90a `ldmatrix` is `.m8n8 .b16` only), so the
transpose is `PRMT` on 4x4 byte blocks: per tile 512 blocks x (4 LDS.32, about 6 PRMT, 4
STS.32), about 60 instructions per loader thread per tile against the current 44 (4 LDS.128,
8 STS.128, 32 F2FP). The difference that matters is the pipe: PRMT runs on the integer ALU
at 64 lanes per clock per SM, F2FP on the 16-per-clock conversion pipe, so the loaders'
share of the saturated pipe (256 of 1,024 cycles per tile) goes to about 50 ALU cycles. P's
pack becomes `cvt.rn.satfinite.e4m3x2.f32` (one F2FP per pair, same count as the f16 pack),
and P.V becomes 2 wgmma k32 instead of 4 k16. The V slot shrinks from 16 KB to 8 KB per
stage, freeing 32 KB of smem.

Precision and rules: this is the baseline's own arithmetic. SGLang's FA3 FP8 path rounds P
to E4M3 with the `Max_offset = 8` trick (`hopper/softmax.h`, as recorded in
`docs/runs-attention.md:97-112`), so "no coarser than the baseline" holds if the kernel
keeps the 2^8 offset (P x 256 before rounding, l carries the same factor and cancels in the
merge). The runs log says the f16 P is finer than FA3's, so this gives that margin back;
the judge's bound is 2.5x the FP8-query reference error, and FA3 passes under it with FP8
P, so the risk is margin, not pass/fail. Keep it behind a flag and measure e= against
`_ref32` on the coherent full shapes, where the low-attention values share a direction.

Expected: removes 256 of the 1,024 conversion-pipe cycles per tile and the f16 V traffic;
alone, about 354 -> 310-320 us at b32 full and 185 -> 165 at b16; with 2.2 and the ex2
split (var 80), the math side reaches about 1.3-1.4k per tile: b32 about 220-240 us, b16
about 112-125 us. Confidence medium-low on the exact numbers, medium that the direction is
right (the budget table is arithmetic on counted instructions).

This does not reopen "FP8 mma.sync" (ruled out): that path unpacked to f16 on every call;
wgmma takes e4m3 natively, which is the backlog's own distinction.

### 2.4 (c) q_len 1 on full layers

The fused kernel's loop: stamps at b32 full (fa01e4d1, last CTA) 375,890 - 10,020 = 365.9k
cycles for 78 tiles = 4.7k cycles per 64-key tile per CTA, 3.5 B per cycle per CTA, 7 B per
cycle per SM with two CTAs resident. The ws loaders at 1 CTA per SM sustain 16 KB per
1.73k cycles = 9.5 B per cycle, and the gather bench 13.8 B per cycle per SM at two CTAs per
SM. The fused loop's structure is the cause: `cp_wait<STAGES-2>; __syncthreads()` before
every tile (`attn_kernel.cu:751-752`), so tile `it+3` is issued only after tile `it` landed
and all 8 warps synchronized, the pipeline depth is 3 tiles at best, and the per-tile chunk
compute (16 F2FP + 4 HMMA + 8 MUFU + 16 F2FP + 16 HMMA per warp, with one chunk per warp)
runs in series with the issue. It is latency-bound, not pipe-bound: per tile per CTA the
conversion pipe needs about 192 warp-instructions x 8 cycles / 4 SMSPs = 384 cycles, a
third of the 1.2k a memory-bound tile would allow.

The fix is the ws structure with a q_len 1 math path, not more tuning of the fused loop:
4 loader warps (cp.async now, row TMA later) feeding an mbarrier ring with no block barrier,
and one consumer warpgroup for both heads. Two options for the math: (i) keep `mma.sync`
f16 with per-fragment conversion (the fused kernel's `chunk`, 384 conversion-pipe cycles
per tile, fine), or (ii) wgmma `m64n64k32` e4m3 with the 8 query rows of a head replicated
to fill M=64 (the tensor pipe is 15% busy at q_len 8, so the waste is free) and 4 scores per
thread, which cuts the softmax to about 1/8 of the q_len 8 cost. Either way the math is
far below the loader rate, so the kernel runs at the loader rate.

Occupancy is the point of the one-warpgroup design and is what reopens the ruled-out A2
("registers block two blocks per SM"): that audit was for 384 threads at 106-128
registers. With 128 math threads + 128 loader threads = 256 threads, two CTAs need <= 128
registers per thread (65,536 / 512), and the FP32 ws math uses 115. Smem: 6 stages x 16 KB
+ Q + barriers = about 105 KB per CTA, two per SM fits in 227 KB. Two CTAs per SM is the
regime where the gather bench reached 181 us at b32 and 97 us at b16 (`splits=16 u=8`).

Estimate: b32 full 250 + 4 -> 190-200 us; b16 133 + 5 -> 100-105 us; and the same kernel
takes b1 full (section 3). Confidence medium-high for b16/b32: it reuses measured parts
(the loaders' rate, the gather regime) and removes a measured 4.7k-per-tile loop.

## 3. Full context, batch 1 (d1 full 18.3 + 6.9 + prep; v1 full 19.2 + 7.2 + prep)

Floor 4.3 us; the gather bench at b1 is 13.8 us with 264 CTAs (`GATHER b=1 ... splits=264
u=8 13.8us`, 1.57 TB/s; 15.1 us at 132 CTAs, 19.2 us at 64). Those are eager event
timings that include one launch (the bench printed an empty-launch figure to subtract, but
the retained log tail does not keep it), so the raw gather at b1 is under 13.8 us, likely
8-10 us. The earlier fused attempt
lost (29.4 vs 27.5 us, job a15d258c) because its loop ran 10 tiles per CTA at 4.7k cycles
each (24 us) with 128 splits. With the loader design of 2.4 at about 1.3-1.7k per tile: 128
splits, 10-11 tiles per CTA, 14-19k cycles = 7-9.5 us, plus a 2.5k prologue, plus a
combine. The combine must be rewritten for this shape: `combine_kernel`
(`attn_kernel.cu:917-929`) launches `batch * q_len` blocks, so at b1 one block of 512
threads reads all 128 x 16 x 264 B = 540 KB (q_len 1) or 4.3 MB (q_len 8, one block per
token) from L2 at roughly 126 GB/s per SM; use one block per (token, head) instead (16 or
128 blocks reading 34 KB each), which is under 1 us plus its launch. The ws kernel's own
tail at b1 full (`_ws@s=16=99`, `s=8=180` in bench 34929136) shows the same per-CTA rate
problem: one CTA streams at 3.6 B per cycle when it has 10 tiles and a cold start, which is
why 128 splits are needed there and why the loader design, not the split count, is the fix.

Estimate: d1 full 28 -> 14-16 us, v1 full 30 -> 15-17 us (kernel 9-11 us + combine 2 +
launch gap). Confidence medium. This reopens "fused q_len 1 kernel at full decode batch 1"
with a different kernel; the evidence is the per-tile loop cost above.

## 4. Layer level: norm-gate/combine and the o projection

`kernel_fn` (`muse_decode_layer/submission.py:4943-4949`) runs attention, then
`_proj(o, "o", ...)`, which is our `proj_kernel<1,4,1,8,2>` at T=1 (`_proj_wins`,
4928-4933: o only at one token) and cuBLAS otherwise. The o weight is 8192 x 1024 BF16 = 16
MB; the T=1 kernel takes 6.9 us (floor 3.5 us) and is pure weight streaming; the attention
output it reads is 2 KB.

Fusing the norm-gate arithmetic into the GEMV's x-load is possible and exact (round to BF16
at the same point SGLang does), but it saves nothing: our paths already apply the norm-gate
inside the merge (cluster kernel) or the combine, so there is no separate gate launch to
remove, and pulling the split merge into the GEMV would make each of the 256 GEMV CTAs
re-read S partials of 8 heads (138 MB of L2 reads at b16 full). Not worth doing.

What is worth doing is overlap, with programmatic dependent launch (PDL), which the layer
already uses for the W4 GEMMs:

- o projection after attention: launch the o GEMV with
  `cudaLaunchAttributeProgrammaticStreamSerialization`; its CTAs start while the attention
  kernel runs, prefetch their weight tiles (the 4-stage cp.async pipeline already stages
  weights in smem; 256 CTAs x 96 KB covers the whole 16 MB), then `griddepcontrol.wait`
  before touching the attention output. At b1 sliding the attention kernel occupies 16 SMs
  for 14 us while 116 SMs idle, so the GEMV's 6.9 us collapses to its dot products plus
  drain, about 1-2 us. The attention kernel should call `griddepcontrol.launch_dependents`
  at its start (its output is only needed at the wait, and the wait semantics guarantee
  completion and visibility of the primary's writes).
- attention after the qkv_gate projection: everything in the attention prologue except Q
  prep (seq, page entries, K/V rows, scales, inv_freq, even the whole K/V gather and Q.K
  cannot start, but the gather can) is independent of the qkv projection's output. With
  PDL on the attention launch, the attention CTAs can run the gather under the qkv GEMV's
  tail and wait only before reading `qkv`. The qkv `proj_small` at T<=8 is 144 CTAs of about
  70 KB smem, so an attention CTA (66-90 KB, 256 threads) fits beside one on most SMs.

Estimate: d1 sliding -6 to -9 us, d1 full -3 to -5, v1 sliding -5 to -8, v1 full -3,
b16/b32 sliding -2 to -4 each (less idle room). This overlaps with items 1.3/1.4 (what
those remove, this no longer hides); counted together with them the layer gain is about
+0.02. Effort 6-10 hours: launch attributes on two launches, the two `griddepcontrol`
calls, a graph-capture self-test (programmatic edges capture in CUDA 12.3+), and checking
that cuBLAS launches at T>1 are left alone.

## 5. Ranked list

Estimated savings are kernel-time deltas in the layer profile's terms; score deltas use the
given sensitivities. Effort is for an expert CUDA/PTX engineer. Local checks first: today's
20 checks on this problem are used up; `tools/attn_ws_quick.sh -s`, `tools/attn_compile_check.sh`
and the SASS analysis in the scratchpad `sass/` folder run without a GPU.

### R1. q_len 1 sliding: fixed-cost diet of the cluster kernel (section 1.3, items 1-3)

- Mechanism: hoist all cold loads to the kernel top; one `sincosf` body (unroll 1) and no
  dead `half == 1` work for L=1; one `chunk()` instantiation; attend the new token under
  tile 0's latency instead of after the loop. Code 98.6 KB -> about 20 KB.
- Files: `muse_attention_decode/attn_kernel.cu` (prologue 267-468, new-token tile 778-780,
  `merge_row` 206-240); re-inline into the layer with `tools/dl_attn_inline.py`.
- Savings: d1s 14.5 -> 8-9 (5.5-6.5 us), d16s 15.8 -> 10-11 (5), d32s 21.4 -> 15-16 (5.5);
  full q_len 1 shapes gain the prologue part only (about 2 us). Score about +0.017.
- Confidence: medium-high for the load hoisting and new-token fold (measured phases of
  3.3k and 3k cycles); medium for the code-size part (correlation plus the 824-cycle
  skipped branch; experiment below).
- Effort: 6-10 h. Risk: low; same math, same fences (none involved), same tests.
- Cheapest experiment: a STAMP build that calls `merge_row` twice on the same row (second
  pass warm): if the second pass is about 1k cycles, the code-fetch reading is confirmed.
  Then one bench with the diet applied, reading phase 0-1, 3-4 and 6-7.

### R2. q_len 1 full: ws-style loaders with a q_len 1 math path, two CTAs per SM (2.4, 3)

- Mechanism: 4 loader warps + 1 consumer warpgroup, mbarrier ring (6 stages, about 105 KB),
  `__launch_bounds__(256, 2)`; math by `mma.sync` f16 (the fused `chunk`) or replicated-row
  wgmma; a combine with one block per (token, head); route b1 full as well.
- Files: new kernel in `attn_ws.cu` (loader code 375-596 reused, a new consumer branch) or
  a new file; host routing `_fused_ok`/`_ws_ok` in `muse_attention_decode/submission.py`
  and `_attn_path` in the layer (4343-4353); `combine_kernel`.
- Savings: d32f 254 -> 190-200 (55-60 us), d16f 138 -> 100-105 (35), d1f 28 -> 14-16 (13).
  Score about +0.030.
- Confidence: medium-high (removes a measured 4.7k-per-tile loop; reuses the measured
  loader rate and the 2-CTA gather regime). Reopens A2 with the register arithmetic above.
- Effort: 16-24 h. Risk: register fit for two CTAs (ptxas check first); the loader-side
  `fence.proxy.async` is not needed on this path if the math reads smem with generic loads
  (no async-proxy consumer), which the review should state explicitly rather than assume.
- Cheapest experiment: ptxas register count of the new instantiation; then loads-only mode
  at b32 full against `_ws@x=16,m=1` (301 us) and the gather's 181.

### R3. q_len 8 full: row TMA loads (A4) plus FP8 P.V with transposed V (N2), then the ex2 split (sections 2.2, 2.3)

- Mechanism: tensor-map TMA, one-row box, SW128, one request per K row and per V row;
  loaders transpose V with `PRMT` into a K-major e4m3 B tile in FA3's permuted token order;
  P packed with `cvt.rn.satfinite.e4m3x2.f32` with the 2^8 offset; `wgmma m64n64k32 e4m3`
  for P.V; keep the writer-side `fence.proxy.async` before the full-barrier arrive (generic
  writes of the transposed V are read by wgmma); var 80's `ex2_poly` on a quarter of the
  exponentials once the loaders are off the critical path.
- Files: `attn_ws.cu` loader branch (375-596), `pv_issue`/`wgmma_pv` (207-213, 904-910),
  `softmax32` pack (873-887), `Smem` (30-38); a tensor map built in the host wrapper
  (`cuTensorMapEncodeTiled` via the driver API) per call, which must be graph-safe (encode
  once per cache tensor address; the caches are fixed per shape).
- Savings: v32f 354 -> 230-260 (95-120 us), v16f 185 -> 115-130 (55-70). Score about
  +0.019. The ex2 split alone (already built, pending bench) is worth 5-15 us at b32.
- Confidence: medium-low on the TMA rate (undocumented), medium on the pipe budget.
- Effort: TMA 8-12 h; N2 12-16 h; var 80 0 h (built, in the pending bench).
- Risk: N2's precision interpretation (matches the baseline's own FP8 P.V; keep behind a
  flag and record e= on the coherent shapes); TMA descriptor lifetime under graph replay;
  the tail mask (rows past `hi`) becomes an `expect_tx` of `valid_rows x 256` per tile
  instead of src-size 0.
- Cheapest experiment: loads-only build with TMA rows (mode 1) at b32 full; then N2 in
  the stamped build reading `v-convert+release` and the math `softmax`/`pv` columns.
- Not reopened: eight loader warps, paired tiles, DS3 index math (TMA supersedes it), the
  loader fence (kept).

### R4. q_len 8 sliding: in-kernel cluster merge and norm-gate, sliding-specialized ring (1.4)

- Mechanism: `NS=4, DA=2` instantiation for sliding launches, 33 KB receive buffer, cluster
  launch with dims = splits, DSMEM push (or `st.async` with `complete_tx`), local merge +
  norm-gate, no `ws_combine_kernel`; prologue diet (one `sincosf` body, hoisted loads); new
  tokens attended first.
- Files: `attn_ws.cu` (prologue 613-728, new-token tile 1116-1122, tail 1126-1151, launch
  1211-1221 needs `cudaLaunchKernelEx` with cluster attributes), host `attn_ws`/`attn_verify`.
- Savings: v1s 16 -> 8-9 (7), v16s 17.5 -> 10-11 (6.5), v32s 22.5 -> 14-15 (8). Score about
  +0.013.
- Confidence: medium. Effort: 10-16 h. Risk: cluster co-scheduling of 150 KB, 384-thread
  CTAs at cluster size 16 (query `cudaOccupancyMaxActiveClusters`, keep the two-launch
  fallback); barrier placement for the receive buffer; fences unchanged.
- Cheapest experiment: ptxas and smem of the `<4,2>` instantiation; a bench variant with
  the merge in-kernel on sliding b1/b16/b32 against `_ws@x=16`.

### R5. Layer: PDL overlap around attention (section 4)

- Mechanism: PDL attribute on the o GEMV launch and on the attention launch;
  `griddepcontrol.launch_dependents` early in attention; `griddepcontrol.wait` in the GEMV
  before reading `o` and in attention before reading `qkv`.
- Files: `muse_decode_layer/submission.py` `proj_launch` (717-731), `attn_fused` (1950-1996),
  `attn_verify` (3098-3140), `launch` in the inlined attention code; kernels get the two
  PTX calls.
- Savings: d1s 6-9, d1f 3-5, v1s 5-8, v1f 3, b16/b32 sliding 2-4 each. Score about +0.02
  net of overlap with R1/R4.
- Confidence: medium. Effort: 6-10 h. Risk: graph capture of programmatic edges; the qkv
  projection at T=1 is a cluster launch (`__cluster_dims__(1, S, 1)`), check PDL works with
  it; cuBLAS paths at T>1 untouched.
- Cheapest experiment: one layer check with PDL on the o GEMV only.

### R6. Sliding b32 q_len 1: deeper ring or 8 splits after R1 (1.3 item 5)

- Savings: d32s 1-3 us. Effort 2 h. Try only after R1; `_fused@s=8` lost at b32 before
  because of the fixed costs R1 removes.

### R7. Global-counter last-CTA merge as the alternative to the cluster merge (1.3 item 4)

- Keep as the experiment to run if R1's merge phase stays above 2k cycles. Reopens the
  ruled-out scratch-buffer reduction with the evidence in 1.3. No spinning on the counter.

## 6. What this report does not establish

- The cold-code reading of phases 3-4 and 6-7 is inferred from code bytes, cycle counts and
  the 824-cycle skipped branch; the two-pass stamp experiment settles it.
- The TMA unit's rate on 128-byte boxes is unmeasured anywhere in NVIDIA's documents; the
  loads-only bench is the only way to learn it.
- Var 16's stamps (FP32 softmax) have not been recorded; the budget table uses counted
  instructions of var 16's SASS (68 `MUFU.EX2` and 184 `F2FP` in the kernel) and var 0's
  stamps for the loader side.
- All estimates are kernel-time deltas; the judge's per-shape noise is +-1.5% and the
  bench and judge have disagreed twice on sliding shapes (paired tiles, b1 full), so route
  on judge numbers as the runs log says.
