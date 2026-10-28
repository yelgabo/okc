# Gap analysis: our decode layer (1.403x) against the public 2.1x entry

Written 2026-10-28. The public entry is https://github.com/vjeux/okc-2026 (Apache-2.0, five
single-file CUDA submissions; official rows 1.501 / 2.610 / 2.327 / 2.102 / 1.798 on proj /
attention / MoE / layer / prefill). A second account, Madhavan113, ships the same kernels
(byte-identical CUDA per its own diff) and published line-referenced design memos plus its
judged layer A/Bs at https://github.com/Madhavan113/kerneloptimizer (`notes/`). Its best judged
layer run is 2.111x (job b73aff83). "Their" numbers below come from those memos, mainly the
judged profile of job f662ee18 (2.092x); ours from job 0e28c26271c94570 (1.403x). Both profiles
run under programmatic dependent launch (PDL), so per-kernel rows include waiting on the kernel
before them and overstate the later kernels on both sides by similar amounts. Where I infer
rather than read, I say so. Copies of the public sources and memos are in this session's
scratchpad (`audit/raw/vjeux/`, `audit/raw/gh4/`); nothing from them is in this repo.

## 1. The four answers

**What was missing.** Not tuning, structure. On every shape our time is the same kernels in the
same arrangement as SGLang's call sequence, with each piece replaced by a faster piece. Their
layer is a different arrangement: 9 launches instead of our 17-23, every launch chained with
PDL instead of only our four MoE kernels, attention and MoE as one launch each with the merge
and the reduction inside the kernel, and the dense projections on a kernel that never reduces
across CTAs. The per-shape table in section 2 puts about 40 us of the 40 us gap at decode 1 on
attention (10 us), launches and glue (about 20 us) and the dense GEMVs (4-8 us); on the verify
shapes the MoE kernel accounts for 85-180 us of the gap and the q_len 8 full-context attention
for 80-170 us.

**What we could have done.** Six decisions, in the order they would have paid (section 4):
(1) build the attention merge in a fixed frame so splits add into an FP32 workspace with
atomics and the last CTA finalizes, which deletes the combine launch and the cross-split max
exchange; (2) launch every kernel of the layer with PDL and issue weight loads before
`griddepcontrol.wait`; (3) replace torch's and SGLang's glue with three one-program-per-token
kernels and drop the side stream; (4) make the MoE one persistent launch with routing done
redundantly by every CTA, TMA bulk copies of whole 36 KB weight units, and a register-A wgmma
path above 16 tokens; (5) put the dense projections on a swap-AB kernel with x as the mma A
operand, weights as B, split-K only across warps, and 12-bit lossless weights whose escapes are
applied by dedicated warps; (6) treat cold code as a cost on the sliding shapes.

**Would our path have arrived?** No. Our last two days of experiments were tuning inside fixed
structures (variant bits, loader arithmetic, K-split counts, layouts, a boundary kernel) and
the score moved +0.02 per day at the end, with nine consecutive MoE and attention checks inside
the judge's confidence interval. The four component audits of today put everything still on
that path at about 1.55-1.65x: the orchestration items +0.10-0.12, the cheap MoE items +0.03,
the attention diet +0.05. The remaining 0.5x is the three structural rewrites above, which were
not on our roadmap except as backlog entries (A3 in-kernel merge, M1 large tiles) that we had
either deferred or ruled out on evidence from the old structure.

**Was the public entry doing something we completely missed?** Yes, several things (section 5):
the fixed-frame softmax merge justified by the QK-norm bound; speculating the sequence length
from the page table's width; a "flat" full-context kernel that gives every CTA an equal share
of all requests' keys; FP8 wgmma for P.V with eight rows of ones in V^T that produce the row
sums for free; register-A wgmma (`wgmma` with A in registers) so a MoE job can have up to 64
tokens without the shared-memory operand traffic that killed our large tiles; weight streaming
by one bulk copy per unit consumed through registers, not through an smem pipeline feeding
mma.sync; escape warps for lossless compression; and a prepare-time tuner that times whole
candidate chains in the layer and pins picks per token count.

## 2. Per-shape comparison

Judged microseconds per shape (ours job 0e28c26271c94570, theirs job f662ee18), and the profile
rows grouped by component. Their component split is the memo's table; ours sums the profile's
kernel rows by name (attention: the fused, ws, prep, gate, FA3 and combine kernels; MoE: W4
GEMMs, SwiGLU reduce, finalize, route_align, SGLang's routing kernel; dense: proj kernels,
cuBLAS, split-K reduces, router; glue: torch's layer norm, SGLang's two boundaries, the pad).
Ratios are ours over theirs.

| shape | judged ours / theirs | attention | MoE | dense | glue | launches ours / theirs |
|---|---|---|---|---|---|---|
| decode 1 sliding | 116.8 / 77.1 (1.52) | 14.5 / 4.9 | 39.2 / 36.7 | 48.0 / 39.3 | 8.2 / 13.2 | 17 / 9 |
| decode 1 full | 128.3 / 83.1 (1.54) | 25.2 / 11.0 | 45.5 / 29.0 | 48.0 / 38.9 | 4.8 / 13.0 | 20 / 9 |
| decode 16 sliding | 177.0 / 136.0 (1.30) | 15.8 / 7.4 | 112.2 / 60.4 | 73.8 / 51.1 | 3.9 / 12.8 | 22 / 9 |
| decode 16 full | 293.4 / 214.8 (1.37) | 137.8 / 88.1 | 112.6 / 77.2 | 70.6 / 50.6 | 3.9 / 13.8 | 23 / 9 |
| decode 32 sliding | 217.5 / 161.7 (1.34) | 21.4 / 9.3 | 120.0 / 86.3 | 76.8 / 59.9 | 4.3 / 13.3 | 22 / 9 |
| decode 32 full | 448.0 / 291.8 (1.54) | 249.6 / 169.8 | 120.0 / 85.6 | 75.4 / 61.1 | 3.9 / 14.1 | 23 / 9 |
| verify 1 sliding | 155.3 / 109.1 (1.42) | 11.9 / 7.0 | 74.8 / 37.9 | 60.5 / 40.8 | 8.9 / 12.6 | 18 / 9 |
| verify 1 full | 173.7 / 122.2 (1.42) | 26.4 / 14.7 | 85.9 / 72.8 | 60.1 / 40.6 | 4.5 / 12.7 | 21 / 9 |
| verify 16 sliding | 360.0 / 259.9 (1.39) | 17.8 / 11.9 | 246.1 / 162.6 | 83.3 / 76.5 | 9.4 / 16.3 | 22 / 10 |
| verify 16 full | 535.1 / 345.4 (1.55) | 185.3 / 105.3 | 255.4 / 160.2 | 83.4 / 76.6 | 9.2 / 16.1 | 23 / 10 |
| verify 32 sliding | 514.0 / 331.2 (1.55) | 22.2 / 15.4 | 381.1 / 198.3 | 104.6 / 96.6 | 15.9 / 25.6 | 22 / 10 |
| verify 32 full | 854.4 / 497.6 (1.72) | 353.7 / 186.3 | 382.5 / 207.6 | 141.7 / 94.8 | 16.1 / 25.0 | 23 / 10 |

Reading the table:

- Their glue rows are larger than ours because their "route" row carries routing and the
  pre-expert norm, which our table files under MoE and glue. Like for like, glue work is the
  same size on both sides. The difference is that ours sits in 8-14 more launches without PDL.
- Our dense rows at T = 128 and 256 include the router as two cuBLAS GEMMs plus two split-K
  reduces (24.0 + 19.4 us at verify 32 full), on a side stream that is no longer hidden at
  that size. Theirs runs pooler, router hi and router lo as one cuBLAS GEMM on a concatenated
  [4608 x 8192] weight. That alone is most of the 47 us dense gap on verify 32 full.
- Their MoE is faster at every T above 1 even though both kernels stream the same unique
  bytes; see 3.2. Their own notes say the MoE row moves 10-20 us between runs of identical
  code because the judge's fresh routing differs per run, so single-run MoE differences below
  about 15 us are noise on both sides.
- Their attention at decode 1 sliding is 4.9 us for the same 0.5 MB we take 14.5 us over.

## 3. Component by component

### 3.1 Attention

| | ours | theirs |
|---|---|---|
| launches per call | q_len 1 sliding: 1 (cluster merge inside); q_len 1 full: 2 (split kernel + combine); q_len 8: 2 (ws kernel + ws_combine); full batch 1: prep + FA3 + FA3 combine + gate (4) | 1 everywhere |
| merge across splits | running max exchanged through partial buffers, merged by a cluster (sliding) or a second kernel (full, verify) | fixed reference frame: each split adds o x 2^m and l x 2^m into an FP32 workspace with `red.global.add.v4.f32`; a key-count ticket makes the last CTA finalize, or every CTA finalizes a slice (cooperative); the finalizer zeroes what it read, so no memset |
| why that is legal | | QK-norm bounds every logit (unit-RMS q and k, scale 5.473/8), so 2^s stays inside FP32 range without a running max |
| batch 1 sliding plan | 16-CTA cluster, 2 tiles per CTA, cluster barrier, push merge; 7.0k cycles of prologue, 5.9k of merge, 3.3k of data tiles (our stamps) | 32 CTAs with one 64-key tile each, last-CTA finalize, the sequence length speculated from the page table's width and verified when the index group lands; 4.9 us judged |
| full context batch 16/32 | one CTA per (request, split), 8-32 splits, 64-token tiles, 4-7 stage rings | a "flat" kernel: 264 CTAs (2 per SM) each take an equal share of all requests' keys with a per-boundary penalty, a producer warpgroup feeds a 3-4 stage mbarrier ring; 88 / 170 us at decode 16 / 32 full against our 133 / 250 |
| q_len 8 math | FP32 Q.K on wgmma e4m3, FP32 softmax, P rounded to f16, P.V on f16 wgmma with V converted to f16 by the loader warps (800 cycles per tile) | FP8 wgmma for Q.K and for P.V; V transposed in smem by `ldmatrix.trans` + `prmt` + `stmatrix`; eight rows of e4m3 ones appended to V^T so the same wgmma produces the row sums; lazy softmax offset with a byte test on packed P |
| q_len 1 math | FP8 mma.sync lowered to F2FP + HMMA (found in SASS, then f16 mma.sync with K/V converted once) | f16 mma.sync with in-register e4m3 to f16 conversion, "unpadded quads" layout, no ldmatrix |
| cold-code cost | 98.6 KB of SASS for the fused kernel with 17 inlined `sincosf` copies, a 10.8 KB straight-line merge, 824 cycles to branch over the new-token path (today's attention audit, local sm_90a compile) | a "scout census" of instructions and taken branches on the once-per-launch path, magic-number division, lane-0 shuffles for uniform registers, one build per launch mode so each carries only its own epilogue |

What we had and did not use: the backlog listed the fixed-shift softmax (FlashDecoding++) and
rejected it because that paper needs a recomputation fallback; the QK-norm bound removes the
need for one here, which neither chat nor the backlog noticed. The backlog's A3 (in-kernel
merge for sliding verify) is their cooperative finalize; it was deferred behind A1 and A2, both
of which lost. Our A1 test showed the rescale was not the FP32 path's cost, and their design
has no rescale at all.

### 3.2 MoE

| | ours | theirs |
|---|---|---|
| launches per call | route_align (side stream), GEMM1, SwiGLU reduce, GEMM2, finalize: 4 on the main stream, PDL between them | 1: routing prologue run redundantly by every CTA, GEMM1 + SwiGLU, GEMM2 + norm statistics, a software grid barrier, the final norm and weighted sum; counters reset themselves |
| grid | thousands of short blocks (41K launched at T = 256 of which 9.6K work), 4-6 blocks per SM | 132 persistent CTAs, one per SM, static equal slices of the unit list per phase, stream-K with cost tables above 16 tokens |
| weight loads | per-lane 16 B loads straight to registers, a 1 KB stage per warp, 4 stages ahead | one `cp.async.bulk` of a whole 36,864 B unit (16 rows x 4096 K with scales) into a 3-slot ring; the first two units issued before routing starts |
| tokens per weight dequant | 8 or 16 (NT1 / NT2 items); above that the shared-memory port saturates on B fragments, which is why our 64-token tiles lost 2x | up to 8 on mma.sync below 17 tokens; up to 64 on `wgmma m64nNk16` with A (the dequantized weights) in registers and x in smem through a descriptor, above 16 tokens |
| K split | 2-4 across blocks, FP32 partials through HBM (50 MB written and read back at T = 256), reduced by the SwiGLU kernel and the finalizer | across the 8 warps of a CTA with an smem reduction (small T); stream-K across CTAs with a global fixup only for the units cut between CTAs (large T) |
| GEMM1 to GEMM2 | bf16 `inter` through global memory, two launches apart | bf16 `inter` with an evict_last hint and per-job readiness flags; above 16 tokens the GEMM1 epilogue writes straight into GEMM2's wgmma B layout |
| dequant | 14 SASS per 32-bit word (8 weights), 9 on the integer pipe, scale by PRMT + IMAD (the same 2^119 trick) | 7 ALU + 2 IMAD on the FMA pipe per word, one scale PRMT + IMAD per four words thanks to a K permutation that puts one 16-K scale block on one lane's four k-steps |
| x operand | staged in smem per block (32 KB per 16-row item at S = 4), read by ldmatrix | kept in registers for the whole job (16 uint4 per lane), pulled into L2 by `prefetch.global.L2` at kernel start |
| result | 1.699x standalone; at bs 8-32 within 85-97% of the streaming ceiling, at bs 128/256 bound by L2-to-SM traffic at 4.1-4.5 TB/s (today's MoE audit) | 2.327x standalone; MoE rows 60 / 86 us at decode 16 / 32 sliding against our 112 / 120 |

Where our experiments touched theirs and stopped short: our TMA weight kernel (M2) fed a
smem pipeline consumed by mma.sync and lost 3-11% at small T; theirs issues the same bulk
copies but each warp pulls its fragments from the ring into registers and the only per-unit
synchronization is one CTA barrier. Our large-tile kernel (M1) kept mma.sync and put 64
tokens through ldmatrix, which the MoE audit shows saturates the shared-memory port; theirs
uses register-A wgmma, whose B operand goes through the tensor core's own path. Our persistent
kernel walked (item, chunk, split) units and tied the non-persistent one because it kept the
four-launch structure; theirs is persistent to remove launches and to overlap the two GEMMs.

### 3.3 Dense projections

| | ours | theirs |
|---|---|---|
| m = 1 | mma.sync kernel with a 4-stage cp.async ring and a cluster split-K reduction; 65-72% of HBM peak on the big matrices, 32-51% on the small ones | CUDA-core GEMV on 12-bit weights (`cgemv`), `__ldg` straight to registers with every load issued before the first FMA; autotuned against an sk2 config and cuBLAS; 9.2 / 5.2 / 13.7 / 3.5 us for qkv / o / pooler / unpooler slice in the layer |
| m = 8-32 | our kernel only where it won at m = 8 (all but o), cuBLAS above | `sk2`: x is the mma A operand (16 tokens), weights the B operand (8 rows per tile); each CTA owns R rows over the whole K; K split only across the warps of the CTA and reduced through the dead ring; weights by one bulk copy per stage, x by a 3-D TMA box on the same mbarrier; 96-256 CTAs, 1-2 per SM; no cross-CTA reduction at all |
| compression | 12-bit lossless code built, won only on the pooler because the escape-tile lookup sat on the critical path; left out | 12-bit lossless with the same high-byte window idea; escapes applied by 3-4 dedicated warps that take turns on stages, with private accumulators and an uncapped per-CTA table arriving by one bulk copy; the qkv rows are permuted to balance escapes across row blocks |
| router | FP32 CUDA-core kernel at m <= 8 (5.2 / 13.5 us in the layer at T = 1 / 8); cuBLAS hi/lo above | FP32 GEMV `rgemv` at m = 1; sk2 with hi and lo tiles into alternating accumulators and a 3-CTA cluster DSMEM reduction at m = 8-32; fused into the pooler kernel's ring at T = 1-16 (`sk2f`, one launch for pooler + router) |
| m = 128 / 256 | cuBLAS | cuBLAS, plus a 2-3 CTA cluster split-K kernel for qkv at m = 128 (16.4 to 14.0 us) and CUTLASS pingpong for the pooler at m = 256; their notes say no custom path beat cuBLAS by more than about 1 us and a persistent stream-K kernel lost four judged checks |

Their proj memo lists, almost item for item, why ours costs 10 us more at small m: a
cross-CTA split-K reduction in the compressed path, escapes on the critical path, a capped
escape table, and global partials with a ticket. Those are exactly the mechanisms our ruled-out
register records as losers; we concluded "split-K reductions above 8 tokens" and "software
compression" were dead ends, they concluded the reductions and the escape handling were the
problem and removed them.

### 3.4 Layer glue and launches

| | ours | theirs |
|---|---|---|
| launches at T <= 32 | 17-23 (14-16 main, 3-7 side) | 9, single stream |
| PDL | the four W4 kernels only; none of our proj, attention, router or boundary kernels triggers or is launched as a secondary, so SGLang's own PDL kernels gain nothing after ours | every launch: Triton glue with `launch_pdl`, sk2 with the first weight stages issued before `griddepcontrol.wait`, attention and MoE with the attribute; measured as a launch-latency win at T <= 32 and a 1% loss at T = 256, so off above 32 |
| glue kernels | SGLang's two boundary modules (4.7 + 3.5 us), torch `F.rms_norm` for the pre-expert norm (6.3 us on a [1, 4096] tensor), SGLang's routing kernel, our route_align, F.pad (two launches) | three Triton kernels, one program per token with 16 warps: boundary 1, route (sigmoid, bias, top-8, local ids, pre-expert norm), boundary 2; about 4.4 us each |
| overlap | router, routing and route_align on a side stream under the pooler (our X1) | none; PDL overlaps tails and heads on one stream |
| unpooler at T <= 8 | the 1024-row slice, then F.pad (fixed by X5, held back on a rounding question that today's read of SGLang's kernel settles) | the slice; the boundary reads the 1024 live columns |
| what lost for them, judged | | fusing glue into a last-CTA tail of the GEMVs (o launch 9 to 49 us: one CTA's serial instruction stream), a boundary prologue that recomputes the norm in every CTA (+13 us at T = 1), a grid barrier at the end of a PDL-chained kernel (3-5 us), PDL above T = 32, routing fused into the pooler kernel |

### 3.5 Method

- Candidate chains timed in the layer at prepare time: per token count, the tuner captures
  and replays each dense backend, the MoE kernel against Marlin, and the glue variants with a
  cold-L2 sweep, then pins picks (`PIN_PICK`) where the isolated timing disagreed with the
  chain. Ours tunes the W4 configs the same way, and routes everything else by tables taken
  from the component problems' checks.
- A composition tool embeds the standalone attention and MoE files into the layer
  byte-identically, so a component win lands in the layer without a port. Ours carries hand
  copies made by `tools/dl_attn_inline.py` and `tools/dl_ws_inline.py`.
- Every new path is self-tested in prepare and gated off unless it wins on the chain, so a
  losing idea costs one check and never ships. Ours is the same in spirit (`_try`, self-tests),
  without the chain timing.
- Iteration depth: their attention source is tagged v298, the MoE v216, the proj v141; the
  memos record switch-level measurements ("ST1F 3 measured best: b16 full 93 to 91.5 us vs
  6"). Our two days produced 75 checks across four problems. They also started earlier: the
  memos are dated from Oct 25 and the repo was created Oct 27; the version numbers point at
  work from the opening day.

## 4. What we could have done, in the order it would have paid

1. Day one, attention: the fixed-frame atomic merge with a key-count ticket. It removes the
   combine launch on every verify shape and the cluster merge on sliding ones, and it is the
   precondition for one-tile-per-CTA plans at batch 1. Our stamps already showed the merge at
   5.9k cycles for 1k cycles of arithmetic; we read that as a DSMEM cost and pushed partials
   instead of asking whether a merge was needed.
2. Day one, layer: PDL on every kernel we own, with weight prologues before the wait, and our
   own three glue kernels. Today's layer audit puts this at +0.03 for PDL and +0.033 for the
   norm alone; the public entry's 9-launch chain is the existence proof.
3. Day two, dense: the swap-AB orientation with no cross-CTA reduction (`sk2`) for m = 8-32,
   and 12-bit weights with escape warps. We built the compression and the mma kernel; the
   missing pieces were the orientation (weights as B so R can be 8-64 rows per CTA with 96-256
   CTAs streaming full K) and moving escapes off the compute warps.
4. Days two to four, MoE: the single persistent launch with redundant routing and bulk-copied
   units, then register-A wgmma above 16 tokens. The MoE audit's hot-expert design reaches the
   same place by a different route (dequant once into smem, multicast token slabs); theirs is
   simpler because A stays in registers and N is capped at 64.
5. Throughout: a chain tuner in prepare, so one check compares many variants in the layer
   where PDL overlap and the judge's routing decide the winner.
6. Sliding shapes: treat code size and cold loads as the cost. Hoist every cold load into
   one group, one `sincosf` body, no dead paths in the once-per-launch code.

## 5. What they did that we missed entirely

- The fixed-frame merge and the QK-norm bound that makes it exact. We discussed fixed shifts
  (backlog, A1) and never connected the bound to deleting the merge.
- Speculating the sequence length from `page_table.shape[1]` at batch 1, a shape and therefore
  legal, verified on the device after the index group lands; it removes one dependent cold
  round trip from the batch-1 chain.
- The flat full-context kernel: equal key shares across all requests with a boundary penalty,
  so the 264 CTAs finish together regardless of per-request lengths. We split per request and
  per power-of-two split count.
- FP8 P.V on wgmma with eight ones-rows in V^T producing the row sums, and keys permuted into
  P's register order so P needs no shuffle; V transposed by `ldmatrix.trans`, `prmt`,
  `stmatrix`. We converted V to f16 in the loader warps (800 cycles per tile) and ruled FP8
  P.V out on mma.sync evidence.
- Register-A wgmma for the MoE, which is what makes 64-token jobs cheap; our large-tile test
  used mma.sync and ran into the shared-memory port.
- One persistent MoE launch with a software grid barrier, per-job readiness flags, and
  counters that reset themselves by parity and generation; the first weight units issued
  before routing.
- Bulk-copied 36 KB weight units consumed through registers, with the first-of-ring issue by
  one thread and a helper warp for epilogues. Our TMA variant fed mma.sync through smem and
  paid a barrier round trip per stage.
- Escape warps for lossless compression, and row permutation to balance escapes.
- The weights-as-B swap in the skinny GEMM so no cross-CTA reduction is needed at m <= 32.
- A fused pooler + router launch at T <= 16, and one concatenated cuBLAS GEMM for pooler +
  router hi + router lo above 32 tokens.
- Cold-code engineering as a first-class concern on the latency-bound shapes.
- A chain tuner with pinned picks, and a composition tool.

## 6. What we did that they did not, and what both of us found

- Both found the same losers: last-CTA glue tails (their o launch 9 to 49 us; our cluster
  finalizer and scratch-buffer split-K), warp-specialized producer rings for the MoE at small
  T (their note "small builds lose 1-2%" with RLOAD; our TMA kernel), PDL above T = 32 as a
  loss, and cluster launches that fail on GPC packing at size 8-16 (their clusters are unused
  for scored shapes; our 16-CTA cluster is non-portable and costs a cluster barrier).
- Our FP32-accumulator attention path keeps the error below FA3's (3.8e-5 to 6.4e-5 against
  FP64 on the same queries); their q_len 8 path rounds P to E4M3 as FA3's FP8 path does. Both
  are inside the rules; ours is the more conservative reading and costs about 8% on full
  verify. Their q_len 1 path pre-scales queries into f16 without E4M3 rounding, which is finer
  than ours.
- Our W4 numerics are closer to FP32 than Marlin's (4.2e-3 vs 5.3e-3) with BF16 rounding at
  SGLang's three points; theirs match the same points (gate/up, inter, y).
- Our route-on-side (X1) and X5 boundary are real wins that they got by other means (one
  stream with PDL; a boundary that reads 1024 columns).
- Our evidence discipline (every losing experiment with a job id and what it rules out) is
  the same discipline their notes show; the difference is which hypotheses were on the list.

## 7. Would our path have reached 2.1x by Nov 4

The honest projection from today's four audits, each on the existing kernels:

| step | score | time |
|---|---|---|
| today | 1.403 | |
| layer orchestration (own norm kernel, X5 with SGLang's rounding, PDL across our chain, L2 prefetch, own boundaries, router at T >= 16) | 1.50-1.52 | 2 days |
| MoE small items (SwiGLU in GEMM1's epilogue, dense block indexing, T = 1 configs) | 1.53-1.55 | 1-2 days |
| attention diet on sliding (code size, hoisted loads, new token under tile 0, in-kernel q_len 8 merge) | 1.58-1.62 | 3-4 days |

That is the ceiling of the path we were on, and it lands below rank 100 on today's board
(2.07x) and well below rank 50 on the board as it will stand on Nov 4 (2.14x today, rising
about 0.1x per day over the last three days). The rest of the distance is the three structural
rewrites: a one-launch attention family (their 2.61x against our 1.32x on that board), a
persistent register-A-wgmma MoE (2.33x against 1.70x), and the no-reduction skinny GEMM with
escape warps (1.50x against 1.14x). Each is three to six days of expert work by the audits'
estimates; the public entry's version counters say the same. Two days of our schedule went
into the FP32 attention precision question, A1, DS3, TMA weights and large tiles, all of which
were measurements inside the old structures and all of which came back inside the interval.

The decision that cost the most was not any single experiment; it was porting SGLang's
arrangement and optimizing each kernel in place, when the per-shape budget says the
arrangement itself (launch count, merges, reductions, side streams) is a third of the time at
decode 1 and the kernels' structures are the rest.
