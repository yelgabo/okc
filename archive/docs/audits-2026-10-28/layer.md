# muse_decode_layer: whole-layer orchestration audit (2026-10-28)

Submitted run 0e28c26271c94570 (1.403x). Sources read in full: `muse_decode_layer/submission.py`
(kernel_fn 4943-4997, `_proj` 4936-4941, attention wrappers 4223-4355, `_route_meta` /
`_experts_run` 4121-4215, W4 wrappers 4096-4119, prepare 4837-4925, the CUDA host code of every
launch), `baseline.py`, `TASK.md`, `docs/runs-decode-layer.md`, `docs/ruled-out.md`,
`docs/context.md`, `experiments/x5_unpadded_boundary.py` (as a diff against submission.py),
`profiles/submission.0e28c26271c94570.{txt,json,log}`, `baseline_profile/profiles/
baseline.7d42771c3384449d.txt`, and SGLang's kernel sources at commit a55ab7d5 fetched from
GitHub: `python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh` (saved at
`scratchpad/sglang_src/norm_residual.cuh`), `kernels/ops/muse_spark_v12/norm_residual.py`,
`fused_norm_residual.py`, `fused_muse_spark_routing.py`, `srt/layers/muse_spark_layers.py`.

Sensitivity used throughout (score per 10 us saved on one shape): d1s .015, d1f .005, d16s .0086,
d16f .0029, d32s .0064, d32f .0022, v1s .011, v1f .0038, v16s .0043, v16f .0015, v32s .0029,
v32f .0010. The sum over all 12 shapes is 0.0666 per 10 us.

## 0. Two facts that change the picture

1. **SGLang's boundary kernel rounds exactly twice, and both points are reproducible.**
   `norm_residual.cuh` (single-CTA kernel `muse_spark_norm_residual_kernel<kHidden, kNumThreads,
   kUsePDL, kPrefetchWeights, kPostNormIsInputScale, kHasRouter, kPostWeight, kGate>`):
   - line 150-157: for the input-scale post norm (the FFN boundary; `post_ffn` is
     `MuseSpark1xRMSNormWithInputScale`), `v = branch * post_w; v = round_bf16(v)` before the sum
     of squares;
   - line 201-208: `normed_b = vals * post_rstd (* post_w if not input-scale); normed_b =
     round_bf16(normed_b)` before `new_s = alpha*s + beta*normed_b` (fp32, line 214-215);
   - line 221 and 257: the pre-norm input and output round only when `pre_norm_bf16` is set;
     baseline.py:317 and :345 pass no `pre_norm_dtype`, so the stream stays FP32 and `out =
     bf16(new_s * pre_rstd * pre_w)` is rounded once at the store; the router copy (line 260)
     is the unrounded FP32 value.
   The X5 kernel (`x5_unpadded_boundary.py`, `boundary_kernel`) skips the first two roundings:
   `br = fv*pv` unrounded and `b = br*rb` unrounded. Adding `bf16r()` at those two lines makes it
   SGLang's math. That settles the hold recorded in `docs/context.md:166-168`.
2. **SGLang's boundary and routing kernels are PDL secondaries.** `norm_residual.cuh:121`
   `PDLWaitPrimary<kUsePDL>()` runs before any load, `:271` `PDLTriggerSecondary` at the end, and
   the launch passes `.config({.use_pdl = kUsePDL})` (`:698`), with `kUsePDL =
   is_arch_support_pdl()` (`norm_residual.py`). The routing JIT op is launched with
   `launch_pdl` too (`fused_muse_spark_routing.py`). They can only overlap a predecessor that
   calls `griddepcontrol.launch_dependents`; none of our main-extension kernels do (the only
   `griddepcontrol` lines in submission.py are 3307-3308, 3391-3392, 3552-3553, 3677-3678: the
   W4 GEMMs, finalize and swiglu), and torch's kernels never do. So today SGLang's PDL buys
   nothing in our graph, and our own chain (proj, attention, combine, prep/gate, route_align,
   finalize_row, router) is launched with no PDL attribute at all (`proj_launch` 729,
   `router_launch` 515, `attn::launch` 1932-1942 sets only the cluster attribute, `launch_ws`
   3086, the `<<<>>>` launches at 778, 795, 954, 970, 1991, 3136).

## 1. Launch tables

Stream: M = main (capturing stream), S = the router side stream (`state["side_stream"]`,
fork at submission.py:4957 `side_stream.wait_stream(main_stream)`, join at :4978
`main_stream.wait_stream(side_stream)`). PDL column: how the launch is made. Times are the
profile's rows (us, under the profiler); "n/l" = not among the listed rows. Host-only work
(`record_stream` at :4982, graph-captured `torch.empty`) launches nothing.

Launch counts were checked against the profile: each profile lists 12 rows plus "N more" rows;
rows are kernel names, and the two SGLang boundaries are separate rows (different template
instantiations). d1s: 13 listed calls + 4 rows (route_align, swiglu, pad fill, pad copy) = 17.
d1f: 13 + 7 = 20. v1s: 13 + 5 = 18. v1f: 13 + 8 = 21 (one more than d1f: FA3 builds its own
tile schedule when `scheduler_metadata` is None, which TASK.md says it is on verify blocks).
d16s: 15 calls + 5 rows (second boundary, bf16 split-K reduce x2, `logits +=` add, routing,
route_align) = 22. The dtype casts at :4129 and :4150 (`topk_weights.to(float32)`,
`topk_ids.to(int32)`) and every `.contiguous()` (:4939, :4966, :4282, :4330) are no-ops; the
routing op returns float32 weights and int32 ids (`fused_muse_spark_routing.py`), and the
launch counts leave no room for a cast kernel.

### T=1 sliding (d1s): profile 0, 116.8 us measured, 17 launches (14 M, 3 S)

| # | kernel | stream | PDL | us | note |
|---|---|---|---|---|---|
| 1 | `proj_kernel<1,4,1,8,4>` qkv_gate (37.7 MB) | M | none | ~11 (row 30.7 is 2 calls with the pooler) | `_proj` :4939 |
| 2 | `attn::attn_kernel<1,1,4,1,true>` fused, cluster 16 | M | cluster attr only | 14.5 | `_attn_fused` :4298 |
| 3 | `proj_kernel<1,4,1,8,2>` o (16.8 MB) | M | none | 6.9 | 2.4 TB/s |
| 4 | SGLang `muse_spark_norm_residual_kernel<8192,256,..>` attention boundary, writes FP32 router input | M | SGLang PDL secondary; our #3 never triggers | 4.7 | :4951 |
| fork | event edge M -> S | | | | :4957 |
| 5 | `proj_kernel<1,4,1,8,4>` pooler (67 MB) | M | none | ~19.5 | |
| 6 | torch `vectorized_layer_norm` = `F.rms_norm` on [1,4096] BF16 (pre_expert_norm) | M | none | 6.3 | :4962; same 6.3 in SGLang's profile |
| 7 | `router_kernel<1>` (8.4 MB FP32) | S | none | 5.2 | :4966 |
| 8 | SGLang `routing_kernel` | S | SGLang PDL secondary; #7 never triggers | 2.9 | :4974 |
| 9 | `route_align_kernel` (1 block) | S | none | n/l (~2.4) | :4150 |
| join | event edge S -> M | | | | :4978 |
| 10 | `w4_gemm_kernel<1,1,8,2,8,4,5,0,1>` GEMM1 (cfg 27, 2 K-splits) | M | PDL launch; waits at entry, triggers right after (:3307) | 21.3 | time includes its wait |
| 11 | `w4_swiglu_kernel` | M | PDL; triggers then waits (:3677) | n/l (~2-3) | reads 2 x 8 x 8192 fp32 |
| 12 | `w4_gemm_kernel<2,1,1,4,4,4,0,0,1>` GEMM2 (cfg 21, 4 splits) | M | PDL; prefetches weights, waits before reading #11 (:3391) | 10.7 | |
| 13 | `w4_finalize_kernel<8>` cluster 8 | M | PDL (:3552) | 4.3 | |
| 14 | `proj_kernel<1,4,1,8,8>` unpooler slice (8.4 MB) | M | none | 5.2 | 1.6 TB/s |
| 15 | torch `fill_` (F.pad) | M | none | n/l (~1) | :4991 |
| 16 | torch `copy_` (F.pad) | M | none | n/l (~1) | :4991 |
| 17 | SGLang boundary (FFN) | M | PDL secondary; torch copy never triggers | 3.5 | :4995 |

Side chain (7-9 plus two gaps) is about 12-13 us against 27 us of main-stream work between
fork and join (pooler + norm + gap): hidden with ~14 us to spare.

### T=1 full (d1f): profile 1, 128.3 us, 20 launches (17 M, 3 S)

As d1s with #2 replaced by the `_attn_prep` path (:4272-4282): `prep_kernel` (n/l, ~2) ->
FA3 main `flash::FlashAttnFwdSm90` 18.3 -> FA3 `FlashAttnFwdCombine` 6.9 -> `gate_kernel`
(n/l, ~2). None of the four carries PDL; FA3's combine alone costs as much as our whole
sliding attention merge. Everything after #3 is identical to d1s (W4 GEMM1 22.8, GEMM2 13.1,
swiglu 5.0, finalize 4.6).

### T=8 sliding (v1s): profile 6, 155.3 us, 18 launches (15 M, 3 S)

| # | kernel | stream | PDL | us |
|---|---|---|---|---|
| 1 | `proj_kernel<1,4,1,8,4>` qkv | M | none | ~12 (33.8 for 2) |
| 2 | `verify::attn_ws_kernel<7,3,0,4,16>` (16 splits) | M | none | 11.9 |
| 3 | `ws_combine_kernel` (merge + norm-gate) | M | none | n/l (3.3 in bd28a847) |
| 4 | cuBLAS `nvjet_sm90_tst_64x8_64x16_2x1` o | M | none | 7.2 (2.3 TB/s) |
| 5 | SGLang attention boundary (router out FP32) | M | PDL secondary, no trigger before it | 4.9 |
| 6 | `proj_kernel<1,4,1,8,4>` pooler | M | none | ~21.5 |
| 7 | torch rms_norm | M | none | 6.0 |
| 8-10 | `router_kernel<8>` 13.5, routing 2.9, route_align ~2.5 | S | none/PDL-sec/none | 19 + gaps |
| 11 | GEMM1 cfg 2 `<1,1,8,4,4,4,0,0,0>` 4 splits | M | PDL | 33.3 |
| 12 | `w4_swiglu_kernel` (reads 4 x 64 x 8192 fp32 = 8 MB) | M | PDL | 12.0 (mostly its wait) |
| 13 | GEMM2 cfg 24 `<2,1,8,2,8,4,0,0,1>` 2 splits | M | PDL | 22.5 |
| 14 | `w4_finalize_kernel<4>` | M | PDL | 7.0 |
| 15 | `proj_kernel<1,4,1,8,8>` unpooler slice | M | none | 6.0 |
| 16-17 | F.pad fill + copy | M | none | n/l |
| 18 | SGLang FFN boundary | M | PDL secondary | 4.0 |

Side chain at T=8: 13.5 + 2.9 + 2.5 + 2 gaps = ~21 us against pooler 21.5 + norm 6 + gap = ~28.5:
hidden by about 7 us. Remove the norm and the margin is ~1 us (see item 6).

### T=8 full (v1f): profile 7, 173.7 us, 21 launches (18 M, 3 S)

As v1s with #2-3 replaced by prep -> FA3 schedule kernel (scheduler_metadata is None on verify)
-> FA3 main 19.2 -> FA3 combine 7.2 -> gate. GEMM1 40.1, GEMM2 26.0, finalize 12.6, swiglu 7.2.

### T=16 and T=32 (d16, d32): profiles 2-5, 22 launches sliding (15 M, 7 S), 23 full

| # | kernel | stream | PDL | us d16s / d32s |
|---|---|---|---|---|
| 1 | cuBLAS `nvjet 64x16(32)_2x1` qkv | M | none | ~14 (row 27.8 is qkv + unpooler) |
| 2 | fused `attn_kernel` cluster (8 or 4 splits) | M | none | 15.8 / 21.4 (full: 133.1 / 249.6 + `combine_kernel` 4.7) |
| 3 | cuBLAS `nvjet 64x16_4x1 splitK` o | M | none | 12.6 / 12.9 (1.3 TB/s) |
| 4 | cuBLAS `splitKreduce` (bf16) for o | M | none | n/l (~2) |
| 5 | SGLang attention boundary (no router out: router_dtype None) | M | PDL secondary | 3.9 / 4.3 |
| 6 | cuBLAS `nvjet 128x16(32)_4x1 splitK` pooler | M | none | 20.2 / 20.5 |
| 7 | cuBLAS `splitKreduce` (bf16) for pooler | M | none | ~2 (row `<..,__nv_bfloat16>` 2 calls 4.2 at d32) |
| 8 | torch rms_norm | M | none | 5.8 / 6.4 |
| 9-15 | `torch.mm` hi (`nvjet tss 64x16 splitK NNT`), `splitKreduce` fp32, `torch.mm` lo, reduce, `logits +=` add (:4973), routing, route_align | S | none (routing PDL-sec) | 9.4 (2 mm) + 3.8 (2 red) + ~1.5 + 2.9 + 2.5 |
| 16 | GEMM1 cfg 26 (4 splits) | M | PDL | 45.4 / 65.9 |
| 17 | swiglu | M | PDL | 10.3 / 6.1 |
| 18 | GEMM2 cfg 28 (4 splits) | M | PDL | 39.5 / 38.3 |
| 19 | `w4_finalize_kernel<8>` | M | PDL | 17.0 / 9.7 |
| 20 | cuBLAS `nvjet 64x16_2x1` unpooler (67 MB) | M | none | ~14 |
| 21 | SGLang FFN boundary | M | PDL secondary | n/l (~4) |

Side chain ~26 us with its six gaps, against pooler 20.2 + reduce 2 + norm 5.8 + gaps 3 = ~31:
hidden by ~5 us. Remove the norm and the side chain is co-critical.

### T=128 and T=256 (v16, v32): profiles 8-11, ~22 launches sliding, ~23 full

Same structure as T=16/32 with: ws attention (17.8 / 22.2 sliding; 185.3 / 353.7 full) +
`ws_combine_kernel`; o = `nvjet 96x64 splitK TNN` 15.1 / 22.5 (+ reduce); pooler = `nvjet
64x64_2x1` 25.7 at T=128 (no split-K), `128x64 1x2` at T=256; GEMM1 cfg 4 (2 splits, partials
67 MB) 150.5 at T=128, cfg 30 (4 splits, partials 268 MB) 234.0 at T=256; swiglu 16.1 / 21.9
(bandwidth-bound on those FP32 partials); GEMM2 cfg 29 / 13 (4 splits) 69.1 / 111.6; finalize<1>
10.4 / 13.6; the SGLang boundaries move to 512 threads at T=256 (8.6 / 7.3 us).

### Idle estimate per shape

Within a CUDA graph the pure dependent-launch gap is close to 1 us, not 3: the one calibrated
point we have is X5 (job b216cf4a vs bd28a847), which removed two torch launches of ~1 us work
each and took 3.7 / 3.0 us off d1. The profiler cannot resolve it: in every profile with a
"GPU busy" line, busy exceeds the measured raw replay (d1f 139.0 vs 132.8 raw; d16s 197.8 vs
181.5; v1s 161.7 vs 160.0; d32f 456.0 vs 453.3), so the per-kernel rows already absorb the gaps
(about 0.5-1 us of CUPTI inflation per kernel). The table therefore uses the launch-count model:
exposed boundaries on the main stream x 1-2 us (the 1.5-3 us rule of thumb as the upper band).
PDL hides three boundaries everywhere (GEMM1 -> swiglu -> GEMM2 -> finalize); the SGLang PDL
secondaries hide none (no trigger before them).

| shape | launches (M/S) | exposed main boundaries | idle 1-2 us | idle 1.5-3 us | measured | kernel rows sum |
|---|---|---|---|---|---|---|
| d1s | 17 (14/3) | 10 | 10-20 | 15-30 | 116.8 | 122.8 (incl. ~10.5 side) |
| d1f | 20 (17/3) | 13 | 13-26 | 20-39 | 128.3 | 143.2, busy 139.0 |
| v1s | 18 (15/3) | 11 | 11-22 | 17-33 | 155.3 | 173.8, busy 161.7 |
| v1f | 21 (18/3) | 14 | 14-28 | 21-42 | 173.7 | 200.2, busy 185.0 |
| d16s / d32s | 22 (15/7) | 11 | 11-22 | 17-33 | 177.0 / 217.5 | 224.5 / 242.7 |
| d16f / d32f | 23 (16/7) | 12 | 12-24 | 18-36 | 293.4 / 448.0 | 348.6 / 472.0 |
| v16s / v32s | ~22 (15/7) | 11 | 11-22 | 17-33 | 360.0 / 514.0 | 380.1 / 545.3 |
| v16f / v32f | ~23 (16/7) | 12 | 12-24 | 18-36 | 535.1 / 854.4 | 557.7 / 913.5 |

So launch idle is 10-15% of the T<=8 shapes and 3-7% of the rest. The bigger T<=8 losses are
inside kernels (W4 GEMMs at 1.7 TB/s, the proj GEMVs at 1.6-3.4 TB/s), which is the MoE and
bf16_proj problems' domain, not this audit's.

## 2. Pure-overhead and tiny kernels

(a) **torch `F.rms_norm` for `pre_expert_norm`** (submission.py:4962 and :4987; module
`MuseSpark1xRMSNorm.forward` = `F.rms_norm(x, (4096,), effective_w, eps).to(dtype)`,
muse_spark_layers.py). 5.8-6.8 us on all 12 shapes, always on the main stream between the pooler
and the join, so always exposed. The BF16 point SGLang materializes is the normalized activation
`bf16(x_fp32 * rstd * w)`, which GEMM1 reads; nothing else is rounded. Two ways to remove it:
- (a1) our own one-block-per-token norm kernel launched with the PDL attribute and
  `griddepcontrol.launch_dependents` at its start, so GEMM1 (which already waits at entry,
  :3307) is resident when the norm ends. Expected ~2 us instead of 6 plus the hidden gap:
  about 5 us on every shape. Works at every T including the cuBLAS pooler.
- (a2) at T<=8 fold it into GEMM1's activation staging: `stage_x` (:3352-3369) stages the rows
  with `cp_async16`; add an in-smem pass after the stage's `cp.async.wait` that rescales each
  staged row by `rsqrt(ss[t]/4096 + eps) * w[k]` and rounds to BF16, with `ss[t]` produced by
  our pooler `proj_kernel` epilogue (:699-710): each cluster rank already holds its final fp32
  sums for its quarter of the 64 rows, so write `sum(bf16r(out)^2)` per (token, row-block, rank)
  to a `[T, 256]` fp32 buffer (no atomics, no zeroing, deterministic), and let staging sum 256
  floats per row (1 KB, L2-resident; at T<=8 under ~10 MB of L2 reads in total). Removes the
  launch entirely: 6.3 + ~1 us on d1/v1. The "moving normalization across a GEMM" decline does
  not apply: the rounding happens on the activation before the mma, exactly where SGLang does
  it. The W4 self-test (`_w4_self_test`, 2e-2 bound) and the MoE self-test cover it.

(b) **F.pad at T<=8** (:4991): `constant_pad_nd` = `fill_` + `copy_`, two launches of about 1 us
each plus two gaps. X5 removes them and replaces the SGLang FFN boundary with our cluster
kernel (3.2 us vs 3.5-4.7): measured -3.7 / -3.0 us on d1, flat on v1 (inside noise). The
rounding concern is settled by section 0: add `bf16r(fv*pv)` and `bf16r(br*rb)` to
`boundary_kernel` (x5 file, the `br[PER]` initializer and the `b = rank == 0 ? br[k]*rb` line).
With those two roundings the self-test against the padded `FusedPostNormResidualPreNorm` call
should fall from 2.3e-3 (update) / 7e-4 (next input) to the fp32-summation-order level
(~1e-6), which is the evidence that lifts the hold. Everything else in X5 already matches
SGLang: FP32 stream, alpha/beta FP32 from `_residual_add._alpha/_beta`, `_frozen_weight` BF16
for both norms, eps 1e-8 / 1e-5, pre-norm not rounded (pre_norm_bf16 is False in baseline.py).

(c) **routing + route_align + casts.** The casts are no-ops (section 1). Routing (SGLang JIT,
2.9 us) and route_align (one block, ~2.4 us) plus their gap are ~7 us on the side stream, which
is hidden at every T today. Fusing them into one kernel (sigmoid, bias, top-8, renormalize,
local ids, sorted ids, W4 work items) saves nothing until the side chain is critical, which (a)
makes true at T>=8 (see section 3). Then it is worth ~3-4 us at T=8 and T>=16. One kernel of
ours, ~4 h; numerics: the top-8 tie handling must match SGLang's `routing_kernel` (routing
tolerance in TASK.md is loose, the reference's clear winners must be kept).

(d) **the two SGLang boundaries** (4.7 + 3.5 us at T=1; 4.9 + 4.0 at T=8; 8.6 + 7.3 at T=256).
Both are latency-bound single-CTA-per-row kernels that wait for PDL before loading anything.
Our versions gain in three ways: a prologue that loads stream/alpha/beta/weights before
`griddepcontrol.wait` (they do not depend on the branch), a `launch_dependents` at entry so the
next proj kernel streams weights during the boundary, and for the FFN side the 1024-live-channel
read (X5). The attention boundary also writes the FP32 router input at T<=8
(`router_dtype=float32`, :4951-4953; `kHasRouter` variant), which must stay the unrounded FP32
value (norm_residual.cuh:260). Expected 2-4 us per boundary at T<=8 with PDL on both sides;
~1 us at T>=16 where cuBLAS sits on either side and cannot trigger.

(e) **cuBLAS split-K reduce kernels at T>=16**: o (`64x16_4x1 splitK` + reduce), pooler
(`128x16_4x1 splitK` + reduce), and the router hi/lo `torch.mm` pair (two `tss 64x16 splitK`
+ two fp32 reduces + the `logits +=` add, :4972-4973). The router ones are on the side stream
(hidden today). The o GEMM is the weak one: 12.6-13.6 us for 16.8 MB (1.3 TB/s) + ~2 us reduce +
gap. Our proj kernel above 8 tokens is in `docs/ruled-out.md` ("ties at best", sweeps
4afc0b73f2a74a69 / e378eace83174734); those sweeps compared kernel time against cuBLAS's GEMM,
not GEMM + reduce + gap inside this graph. Reopening needs that layer-level number: it requires
MT=2/4 instances of `proj_small` (proj_fwd checks `m <= 8`, :830) and an extended
`_proj_self_test`, ~3 h, then one check on d16/d32 (expect -3 to -7 us per shape if the kernel
ties cuBLAS's GEMM time). Low confidence.

(f) **ws combine** (3.3 us at T=8, 4.5-5 at T=128/256) feeding cuBLAS o at T>=8: cannot be
folded into cuBLAS, and folding it into our proj kernel's staging would redo the 16-split merge
in all 256 blocks (hundreds of MB of L2 reads). The real fusion is a last-CTA merge inside
`attn_ws_kernel` (atomic counter per request, the last split CTA merges its request's rows and
applies the norm-gate), which removes the launch and gap (~4 us at v1 s/f, ~5 at v16/v32).
Kernel surgery in the ws kernel, 6-8 h, medium risk; lower priority than PDL on the pair.

(g) **router_kernel<8> at 13.5 us** (:426-513): 128 blocks x 512 threads, one per SM, 64 KB of
weights per block straight to registers, x staged in 32 KB of shared memory, 8 x 8 FMA steps
per lane, cluster-of-8 DSMEM reduction. Standalone it is a ~6-7 us kernel; in the layer it runs
concurrently with the pooler `proj_kernel` (256 blocks, 2 per SM, ~75 KB smem each) that
saturates HBM at ~3.4 TB/s, so its 8.4 MB read gets a fraction of the bandwidth and its profile
time stretches to 13.5 (5.2 at T=1 for the same reason). It is not slow; it is starved, and
hidden. It becomes the critical path only after (a); the cheap fix then is to have its weights
in L2 before it starts (section 3, prefetch), which takes it to ~3-4 us.

## 3. Streams, PDL, prefetch

**Today's structure.** One fork (:4957) before the pooler and one join (:4978) before GEMM1.
The side stream runs router -> routing -> route_align. PDL exists only inside the W4 chain:
GEMM1 waits at entry then triggers (:3307-3308), swiglu and finalize trigger then wait
(:3677-3678, :3552-3553), GEMM2 prefetches its weights before waiting (:3391-3392). GEMM1's
graph node has a programmatic edge from the norm and a full edge from route_align (the join),
which the route-side self-test proved capture-safe (bitwise, log lines 97-102). Because torch's
norm never triggers, GEMM1's early launch today only pre-stages the launch; the real overlap is
swiglu/GEMM2/finalize.

**Where PDL can apply (all same-stream, our kernel -> our kernel):**
1. qkv proj -> attention (fused at decode, ws at verify): attention loads page table, seq_lens
   and the first K/V tiles before `griddepcontrol.wait`, q/k/v/gate of the new token after.
   Hides ~2 us of the attention ramp on 10 of 12 shapes (not the two FA3 shapes).
2. attention -> o proj (T=1 only, 2 shapes): proj issues its weight `cp.async` prologue before
   the wait, x after. Its 6.9 us at 2.4 TB/s is mostly ramp; expect -2 to -3 us.
3. o proj / gate -> attention boundary: our proj triggers at entry; SGLang's secondary launches
   early (section 0). ~1 us at T=1. With our own boundary (item d) the prologue loads add ~1-2.
4. boundary -> pooler proj (T<=8): SGLang's kernel triggers only at its end; with our boundary
   triggering at entry the pooler's 67 MB stream starts ~3 us earlier. The pooler is the single
   largest kernel on the main chain, so this is worth the boundary rewrite on its own.
5. norm -> GEMM1: item (a1)/(a2).
6. finalize -> unpooler proj (T<=8): finalize already triggers at entry; launch the unpooler with
   the attribute and prefetch its weights before the wait: -2 us (its 5.2 us for 8.4 MB is ramp).
7. unpooler -> FFN boundary: X5's kernel as a PDL secondary with its stream/weight loads before
   the wait: -1 to -2 us.
8. ws attention -> ws combine: -1 us.
Implementation: add `griddepcontrol.launch_dependents` as the first instruction of
`proj_kernel`, `attn_kernel`, `attn_ws_kernel`, `combine_kernel`, `ws_combine_kernel`,
`prep_kernel`, `gate_kernel` (harmless when the successor is not a PDL launch), add the
`cudaLaunchAttributeProgrammaticStreamSerialization` attribute to `proj_launch` (it already uses
`<<<>>>` with a cluster via `__cluster_dims__`; switch to `cudaLaunchKernelEx` as `attn::launch`
does), `attn::launch`, `launch_ws`, and place `griddepcontrol.wait` before the first read of the
predecessor's output. The one trap: a load of the predecessor's output that slips above the wait
is a race the self-tests will not catch reliably; review each kernel's prologue by hand. Where
the proj kernel is a secondary, keep each stage's weight and x copies in one `cp.async` group
(issue the stage-0 weights, wait, then the stage-0 x, commit) so `wait_group` accounting is
unchanged.

**Where the join forces a bubble.** At T=1 the side chain (~12 us) ends 14 us before the main
chain reaches the join: no bubble. At T=8 the margin is ~7 us and at T>=16 ~5 us; after item (a)
both margins are ~0-1 us, so the next us saved on the main chain is lost to the join unless the
side chain shrinks too. Options in order of cost: at T>=16 flip `PORT_ROUTER_MMA = True`
(:141, one launch replaces mm + reduce + mm + reduce + add, kernel and self-test exist); at T=8
prefetch the router weights into L2 (below); at both, fuse routing + route_align (item c).
This reopens the ruled-out entry "speeding up the router in the layer: it runs hidden under
the pooler" (jobs 6ef2d969 / 46dab8fb). The new evidence is the margin computed above, and the
confirming observation is the (a) check: if d16/d32 drop by less than the norm's ~5 us while
d1 drops the full amount, the join binds.

**A second side stream.** The only non-dependent work left is prefetch; the unpooler GEMV, the
finalize and the boundaries are all dependent. Prefetch is where a side stream pays.

**L2 prefetch.** Rules: TASK.md:100-105 forbid host syncs, allocation or branching on input
values, caching anything derived from `data`, lower precision, skipped keys, and harness
tampering. A kernel that touches weight addresses derives nothing from the inputs and keeps no
state across calls, so it is inside the rules as written (the full rules page should be read
once before shipping; TASK.md:107). Mechanism: a kernel on the side stream, forked right after
the qkv proj (a second `side_stream.wait_stream(main)` at the top of kernel_fn; the side
stream is in-order so the existing fork before the router still holds), issuing
`prefetch.global.L2::evict_last [addr]` per 128 B line (PTX 7.4+, no register result, no
dependency; `cp.async.bulk.prefetch.L2.global` is the sm_90 bulk form). The kernel completes
when its instructions issue; the lines land asynchronously, which is harmless. What fits in the
50 MB L2 while attention runs:
- T<=8 (attention 11.9-14.5 us sliding, ~30 us full; HBM nearly idle): o (16.8 MB) + router FP32
  (8.4 MB) + unpooler slice (8.4 MB) + the boundary and norm vectors (<0.3 MB) = 34 MB. Consumers
  then read from L2 at >2x the HBM rate: o proj 6.9 -> ~3.5 (T=1) and nvjet o 7.2 -> ~4.5 (T=8),
  unpooler 5.2-6.0 -> ~3 (if the lines survive the 54 MB W4 stream; the W4 loads use
  `L2::256B` hint with normal eviction, :3196, so evict_last lines should outlast them, to be
  measured), router 5.2 / 13.5 -> ~3 (side chain; matters at T=8 after item a). Estimate -5 to
  -6 us on d1s, d1f, v1s, v1f: +0.019. Do not prefetch the pooler: its 67 MB does not fit, and a
  partial prefix needs a layout-aware address pattern for little gain.
- T=16/32 full (attention 133 / 250 us at ~2.5 TB/s of KV, half the bandwidth free) and
  sliding (16-21 us): o (16.8) + router hi/lo (8.4 MB): nvjet o 12.6-13.6 -> ~7, router hidden:
  -6 us on four shapes: +0.012. The fused attention kernel's KV loads should carry
  `L2::evict_first` (our kernel) so the 330-655 MB KV stream does not push the prefetched lines
  out; check its load path first.
- verify 16/32: o (`96x64 splitK` 15.1 / 22.5) -> ~8 / 12: -7 to -10 us on four shapes: +0.008.
- Not for expert weights (data-dependent, too large), but GEMM1 could prefetch GEMM2's W2
  (8 MB per local expert) while it runs: at T=1 that is ~16 MB for the two experts and GEMM2
  (10.7 us for 17 MB) would run from L2. MoE-kernel territory; noted, not ranked.
Risks: thrash if the evict_last fraction the hardware allows is below ~34 MB (gain degrades
toward zero, never negative beyond ~2 us of side-stream issue); the prefetch kernel's blocks
compete for SM slots with FA3 at full b1 (FA3 fills all SMs there; keep the prefetch kernel to
small blocks with no shared memory).

## 4. Allocations, copies, conversions, FA3 prep

- Every `torch.empty` in the call (proj/router/routing outputs, route_align's six tensors,
  `cache2` :4183, the split-K partials :4105 and :4115, finalize's out, F.pad's out, the
  boundaries' outputs, the ws `part` buffer, FA3's workspace) is captured into the graph's
  private pool: no replay cost. At T=256 the GEMM1 partials are 268 MB and GEMM2's 134 MB; the
  pool holds them, and the swiglu/finalize read them at HBM rate (swiglu 21.9 us at v32 is
  exactly 268 MB at 4.2 TB/s). That traffic is the W4 tuner's choice (it timed GEMM + activation
  together, log lines 5-23), not an orchestration bug.
- No redundant copies: all `.contiguous()` calls are on already-contiguous tensors; the only
  dtype work is `out.to(bfloat16)` at :4281, which FA3 does not trigger (its output is BF16).
  `logits += torch.mm(...)` at :4973 adds one elementwise kernel on the side stream.
- `record_stream` (:4982, :4807) is host-only and skipped under capture. Nothing branches on
  input values; `local_start` is read on the device (route_align :230).
- FA3 path at full b1 (d1f, v1f): prep (~2) + FA3 main (18.3 / 19.2) + FA3 combine (6.9 / 7.2) +
  gate (~2) + 3-4 gaps, plus FA3's own schedule kernel on verify (scheduler_metadata None), about
  33-36 us against 14.5 us for the fused sliding kernel. The fused kernel is ruled out there
  (29.4 vs 27.5 us). Two cheap trims: fuse `prep_kernel` into the qkv proj epilogue (each 64-row
  block is exactly one head; the cluster already reduces across its 4 k-slices, so the per-head
  RMS, RoPE, FP8 q and the cache row can be written there: -3 us on d1f/v1f, +0.0026), and at
  T=1 fold `gate_kernel` into the o proj's x staging (per-head RMS over 64 contiguous columns,
  -3 us on d1f, +0.0015). FA3's 6.9 us combine is FA3's own and cannot be skipped through
  `flash_attn_with_kvcache`.

## 5. Ranked changes

Score deltas use the sensitivity table; "us" are per shape.

1. **Pre-expert norm off the critical path.** (a1) own PDL norm kernel, all T: -5 us x 12 shapes,
   +0.033; then (a2) fold into GEMM1 staging at T<=8 with pooler-epilogue sums of squares:
   another -2.5 us on d1/v1, +0.009. Files: `_CUDA_SRC` (new kernel; proj epilogue for a2),
   `_W4_SRC` `stage_x` (a2), kernel_fn :4962. Confidence high for a1 (a 6 us fixed kernel
   replaced by a 2 us one plus a hidden gap, same math), medium for a2. Effort 2-3 h / 6-8 h.
   Risk: numerics limited to fp32 summation order before one BF16 rounding; the W4 and MoE
   self-tests cover it. Experiment: one check of a1; expect d1s/v1s -4 to -5 us and, as the
   side-chain probe, whether d16/d32 drop by the same amount.
2. **X5 with SGLang's rounding** (settles the hold): -3.7 / -3.0 us on d1, ~-3 on v1 once the
   boundary is also a PDL secondary; +0.006 to +0.011. Files: x5 `boundary_kernel` (two
   `bf16r` calls), `_bnd_enable` bound tightened to 1e-4. Confidence high (measured once;
   rounding points read from the source). Effort 1 h. Risk: none new. Experiment: the
   self-test numbers in the log, then one check.
3. **PDL across our own chain** (section 3, items 1-8): -8 to -10 us d1s, -6 to -8 d1f, -6 to -8
   v1s, -4 to -6 v1f, -2 to -3 on the eight larger shapes: about +0.03. Files: the seven kernel
   prologues, `proj_launch`, `attn::launch`, `launch_ws`, the `<<<>>>` sites. Confidence medium
   (mechanism proven by the W4 chain; per-boundary gain is an estimate of 1-2 us). Effort 4-6 h.
   Risk: a read above the wait is an intermittent race; graph capture of programmatic edges is
   already exercised. Experiment: one check with proj + attention only (expect d1s -5 or more),
   then the rest.
4. **L2 weight prefetch on the side stream** (section 3): -5 to -6 us on the four T<=8 shapes,
   -6 on d16f/d32f/d16s/d32s, -7 to -10 on verify 16/32: +0.019 (T<=8) up to +0.039 (all).
   Files: new prefetch kernel in `_CUDA_SRC`, kernel_fn fork after the qkv proj, evict_first
   hints on the fused attention kernel's KV loads for the large shapes. Confidence medium-low
   (cache-policy behaviour must be measured). Effort 3-4 h. Risk: rules (none in TASK.md; read
   the full page), thrash. Experiment: T<=8 only first; the profile's o-proj row should fall
   from 6.9 to ~3.5 at d1s.
5. **Own attention boundary kernel** with SGLang's math (section 2d, 3.3-3.4): -3 to -4 us at
   T<=8, ~-1 above: +0.013. Files: `_CUDA_SRC`, kernel_fn :4951. Confidence medium. Effort 3-4 h.
   Risk: one rounding point and the FP32 router copy to reproduce; self-test against the module.
   Experiment: one check together with item 3's pooler-side PDL.
6. **Shorten the side chain once item 1 lands**: `PORT_ROUTER_MMA = True` at T>=16 (0 h), router
   prefetch (item 4) at T=8, routing + route_align fusion (4 h) at both. Worth whatever item 1
   leaves on the table at T>=8 (estimated 3-5 us on 8 shapes, +0.012). Reopens the router
   ruled-out entry with the margin analysis above as the new evidence; confirm with item 1's
   check before spending a run on it.
7. **FA3-path trims and ws combine fusion** (section 4, 2f): prep into the qkv proj epilogue
   (+0.0026), gate into the o proj staging at T=1 (+0.0015), last-CTA merge in the ws kernel
   (+0.006). Effort 4-8 h each, medium risk. After items 1-6.
8. **Own o GEMM at T=16/32** to delete cuBLAS's split-K reduce and gain PDL (2e): +0.004 to
   +0.014, low confidence, reopens "our mma kernel above 8 tokens" with the layer-level launch
   accounting as the new evidence; 3 h plus one check.

Budget: 7 checks today, 20 tomorrow. Order of runs: item 2 (self-test evidence + check), item 1
a1, item 3 (proj + attention), item 4 (T<=8), item 5 + item 3 remainder, item 1 a2, item 6.

## 6. Where this lands

Items 1-6 together take d1s from 116.8 to roughly 88-92 us and the score from 1.403x to about
1.50-1.52x by the sensitivity table (+0.10 to +0.12). Rank 50 is 2.135x, so orchestration
closes a quarter of that gap; the rest is kernel bandwidth: the W4 GEMMs move 54 MB in 32 us at
T=1 (floor 11 us), the proj GEMVs run at 1.6-3.4 TB/s, and FA3's full-b1 path costs 33 us
against 14.5 for the fused sliding kernel.
