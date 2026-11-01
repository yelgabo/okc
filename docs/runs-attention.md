# Run log: muse_attention_decode

Every `okc check`, newest last. Full output for each job is in
`muse_attention_decode/profiles/<file>.<job>.*`. Bench numbers come from
`tools/attn_make_bench.py`: CUDA graph replays after a 256 MB flush, median of 40, timed at
import on the judge's H200. They include about 5 µs of graph-replay cost that the judge
subtracts, so compare them within a shape, not against the judge's column.

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 10-27 | 2b3c25820fee4b2d | 99f2d18 | baseline copy + component bench | 1.005x (0.990-1.020) | profile: FA3 main kernel is 94-97% of full-layer time at b16/b32 (195 / 374 µs vs floors 63 / 126); sliding is 7 launches of 1-16 µs each |
| 10-27 | 43ef868a181e49b2 | 6416ab1 | idea 1: one prep launch (norm, RoPE, FP8 q, cache write) + FA3 + one norm-gate launch; fused kernel failed to build (-std=c++17) so this timed idea 1 alone | 1.128x (1.111-1.145) | sliding 1.24-1.34x, full b1 1.12-1.17x, full b16/b32 1.02-1.03x |
| 10-27 | 418d8d492c074b88 | 3341f56 | idea 2 v1: one-launch fused kernel (FP8 QK mma, f16 PV mma, cluster merge on sliding, split + merge launch on full) | 0.642x (0.633-0.652) | 16/16 tests pass. Decode full b16/b32 1.14x / 1.19x; everything else slower, verify 0.45-0.53x. Per-CTA rate ~10 GB/s: each tile waited on a dependent page-table load; cluster merge did up to 128 serial DSMEM reads per row |
| 10-27 | 4b52fbd2aaf24247 | b0e02e3 | fused v2: page-table entries staged by cp.async 6 tiles ahead, per-CTA merge of warp partials, lane-parallel merges, softmax per 2-4 chunks | 0.858x (0.845-0.872) | decode: sliding 1.20-1.47x, full b16 1.47x, b32 1.52x, b1 0.73x; verify still 0.49-0.63x. Full b32 flat at ~253 µs across 8-32 splits: about 50% of HBM peak, likely the random 128-byte gather ceiling |
| 10-27 | 5b5af5ddf6b74171 | 8393e76 | route q_len 1 (except b1 full) to fused, rest to idea 1; bench q_len 8 variants (values converted to f16 once per stage + ldmatrix.trans; interior mask skip; conditional rescale) | **1.233x** (1.215-1.257) | f16-once values did not help q_len 8 (b32 full 763 vs 621 µs bench); verify full stays 2.4x slower than decode full |
| 10-27 | 5d1187f90830484b | 4135442 | same routing; bench FP8 P.V variants (P x 2^8 as FA3) and per-phase clock64 stamps | 1.243x (1.225-1.265) | FP8 P.V slower (b32 full 763 vs 662 µs). SASS shows why: FP8 mma.sync on sm_90 is F2FP unpack x12 + 2 HMMA per call, so the FP8 Q.K re-converted Q every call. Stamps (b1 sliding, cycles): Q prep ~6k, new-token tail ~6k, cluster merge ~8.6k, loop ~3k |
| 10-27 | c35deb23dada45d4 | 37e8cc3 | f16 Q.K (Q converted once), Q and new tokens prepared while page entries load, gate prefetch for the merge | **1.254x** (1.235-1.275) | decode sliding 1.28-1.53x, full 1.13-1.47x; fused b1 full now 34.0 µs bench vs 35.5 for idea 1. q_len 8 still ~630 µs at b32 full: variants with more warps per KV head get slower, which points at each warp re-converting the same FP8 K/V |
| 10-27 | a15d258c37f44232 | 062b027 | q_len 8 variants with K and V converted to f16 once per stage (ldmatrix for both mmas); fused path for all q_len 1 | **1.263x** (1.244-1.282) | conversion was not the q_len 8 limit (b32 full 631-655 µs, unchanged). Judge put fused b1 full at 29.4 µs vs 27.5 for idea 1, though the bench had fused ahead |
| 10-27 | c2a8b716effb4512 | 8499f13 | push-based cluster merge (each CTA stores partials into the owning CTA's smem, one barrier, local merge); b1 full decode back on idea 1 | **1.285x** (1.265-1.308) | sliding decode b1 15.08 µs (1.40x), b16 1.61x, b32 1.47x |
| 10-27 | fa01e4d1a9f54b0f | 9a15913 | new tokens staged in their own tile during the prologue; scales read at kernel start | 1.276x (1.257-1.295) | sliding decode faster (b1 14.26 µs, 1.44x; b16 1.64x; b32 1.51x) but full decode b16/b32 2-3% slower on judge and bench; the extra tile now applies to cluster launches only |
| 10-27 | 944b519698894c2a | 3b84d10 | submission.py itself (no bench): new-token tile on cluster launches only | **1.289x** (1.269-1.308) | best. Decode: sliding 1.45 / 1.66 / 1.47x, full 1.13 (idea 1) / 1.48 / 1.51x. Verify (idea 1 + FA3): sliding 1.36 / 1.35 / 1.25x, full 1.19 / 1.03 / 1.02x |
| 10-28 | 24daba2202e547d6 | 9d7e3bf | baseline.py as the submission (`baseline_profile/`), SGLang's kernels per shape | 0.997x (0.982-1.011) | FA3 main kernel 376 µs = 95% of verify b32 full; the profiler put decode b32 full's at 499 µs (timed total 385), so profiler numbers are rough |
| 10-28 | e2816b2cfe2e409e | 9d7e3bf | bench: warp-specialized q_len 8 kernel (attn_ws.cu: 2 consumer warpgroups, FP8 Q.K and f16 P.V on wgmma, 1 loader warp, 8-stage mbarrier ring, no block barrier in the loop); gather microbenchmark failed to build | 1.291x (1.272-1.310), routing unchanged | correct (e=1.3e-2 vs prep+FA3) but slow: b32 full 594-628 µs (FA3 392), b16 305-320 (209), b1 52-56 (38). Loads-only mode 523 µs at b32: the one loader warp spent 2735 cycles per tile issuing cp.async. Math warps ~2600 cycles per tile, 1341 of them in softmax (MUFU-bound: 8192 f32 ex2 per tile per SM) |
| 10-28 | 8cb557afaeab4e08 | dbe1fdd | bench: ws v2 (2-8 loader warps; Q.K accumulates in f16, half2 softmax with f16x2 ex2; scale folded into Q before E4M3) + gather microbenchmark | 1.280x (1.261-1.302), routing unchanged | ws b32 full 364 µs vs FA3 path 392 (load-only 231); b16 full 190 vs 210; b1 full 39 vs 39; sliding b32 29-30 vs 32, b16 24 vs 25.5, b1 21 vs 21. Error vs prep+FA3 rose to 8.4e-2 (v1 1.3e-2): v2 rounds a scaled Q, so it no longer shares FA3's FP8 rounding; needs an FP32 reference to judge. Gather (cold L2, page table, b32): 128-byte rows 3.6 TB/s (181 µs for 655 MB), 64-byte rows (FA3's per-head reads) 1.8-2.2 TB/s; b16 3.4 TB/s; b1 1.6 TB/s (13.8 µs). Dependent page entry + row: 1343 cycles; row alone 741. Stamps: loader waits 1143 cycles per tile for free stages, math warps ~1930 per tile (softmax 925, V convert + barrier 418, P.V 310): math-bound |
| 10-28 | 34929136fa004b17 | f0251de | bench: ws v3 (loader warps convert values to f16 into a 4-slot ring, DA = 3 tiles behind the loads; math warps have no barrier at all; both rows' softmax in shared half2 ops) + FP32 reference | 1.290x (1.271-1.309), routing unchanged | error vs FP32: ws 7.0e-2 at b32 full vs 7.1e-2 for prep+FA3 (6.2e-2 vs 6.5e-2 sliding), so v2's gap to FA3 was FA3's own error. ws b32 full 339-343 µs (FA3 391), b16 183-186 (209), b1 40-41 (38), sliding b32 27.8 (32.3-32.9), b16 23.5 (25.2-26.0), b1 20.3 (20.2-21.0). Loader warps now spend 965 cycles per tile on F2FP conversion; load-only mode rose to 282-302 µs |
| 10-28 | 13b7b297c1f34b29 | 6374644 | ws routed for q_len 8 except full b1 (integer E4M3-to-f16 conversion as default) + bench | **1.297x** (1.278-1.321) | 16/16 tests. Judge, verify: full b16 193.3 µs (1.061x, was 1.032), b32 363.8 (1.074x, was 1.017); sliding b1 1.354x, b16 1.359x, b32 1.244x (was 1.277). Bench: integer conversion lost to F2FP (b32 full 361 vs 344 µs, b16 192.5 vs 186.7). Loader conversion time stayed ~950 cycles per tile with 8 loader warps as with 4, which points at the loader's fence.proxy.async waiting for its in-flight cp.async, not at the conversion work |
| 10-28 | a8a4666a91d54d68 | a5016ee | ws var 0 (F2FP conversion) routed; bench: paired tiles (var 4), loader release without its own proxy fence (var 2), both (var 6), stages 6/8 | **1.327x** (1.307-1.346) | 16/16 tests. Judge, verify: full b16 185.9 µs (1.101x), b32 347.4 (1.122x); sliding b1 1.354x, b16 1.421x, b32 1.340x. Bench b32/b16 full: var 0 344.6/186.8, var 2 333.8/178.8, var 4 347.3/186.0, var 6 339.4/181.8 (8 stages 333.2/178.6). Pairs cut the Q.K wait (296 to 186 cycles per tile) but not the total. Load-only mode 301 µs: the loaders (cp.async issue ~660 + conversion ~790 cycles per tile) now set the pace |
| 10-28 | 1f4f986279834925 | 203eb4e | ws var 2 routed on full layers, var 6 (paired tiles, no loader fence) on sliding; bench var 3 (integer conversion, no fence), var 10 (8 loader warps), 8 stages | 1.307x (1.288-1.332) | 16/16 tests. Full verify gained: b16 176.3 µs (1.167x), b32 331.6 (1.177x). Sliding verify lost on the judge although the bench had var 6 level with var 0: b1 19.30 µs (1.174x, was 1.354x), b16 1.247x (was 1.421x), b32 1.264x (was 1.340x). Bench b32 full: var 2 333.9, 8 stages 331.7, var 3 349.3, var 10 342.5 |
| 10-28 | dc37819eef384acb | c326dcc | submission.py: ws var 0 on sliding verify, var 2 on full verify (b16, b32); everything else as before | **1.347x** (1.327-1.368) | best. 16/16 tests. Verify: sliding b1 16.64 µs (1.388x), b16 1.420x, b32 1.358x; full b1 1.191x (prep+FA3), b16 176.99 µs (1.162x), b32 333.42 (1.174x). Decode unchanged (1.48 / 1.14 / 1.67 / 1.48 / 1.49 / 1.51x) |
| 10-28 | dcea57a9e84f441a | 6f6179f | same routing as dc37819eef384acb; docstring and line wrapping only | **1.348x** (1.328-1.368) | best, final. 16/16 tests. Verify: sliding 1.368 / 1.426 / 1.380x, full 1.181x (prep+FA3) / 1.173x (175.80 µs) / 1.180x (331.68 µs) |

## Baseline profile (job 2b3c25820fee4b2d)

| shape | SGLang µs | FA3 main | FA3 combine | other launches |
|---|---|---|---|---|
| decode 1 sliding | 21.5 | 7.9 | 2.4 | 5 (prepare_varlen, qk norm, 3 casts, store, norm-gate) |
| decode 1 full | 31.0 | 14.7 | 6.9 | 5 |
| decode 16 full | 202.8 | 194.6 | 3.6 | 5 |
| decode 32 full | 387.0 | 373.9 | 3.5 | 5 |
| verify 32 full | 392.2 | 378.5 | 4.1 | 6 |

FA3 takes the same time at q_len 1 and 8 on full layers, so it is bound by its page-size-1
gather, not by math.

## Lessons

- FP8 `mma.sync` (m16n8k32 e4m3) on sm_90 compiles to 12 `F2FP.F16.E4M3.UNPACK_B` plus 2
  `HMMA.16816` per call (checked in SASS). Hopper has native FP8 math only through wgmma. With
  mma.sync, convert reused operands to f16 once; FP8 P.V buys nothing.
- With page size 1, every token's K and V row is a random 128-byte read. Reading both KV heads
  per row (one CTA per request covers both) gets decode full layers to ~50% of HBM peak
  (b32: 252 µs vs a 126 µs floor; FA3 reads 64 bytes per head and sits near 33%). Time stayed
  flat from 8 to 32 splits, which points at the gather itself as the ceiling.
- A dependent page-table load before each tile's cp.async serialized the pipeline (~10 GB/s
  per CTA). Staging page entries by cp.async 6 tiles ahead fixed it.
- Remote DSMEM loads in a cluster merge cost microseconds per row on b1 sliding. Pushing
  partials into the owner CTA's smem with remote stores, then one barrier, then a local
  merge, is cheaper, and no second barrier is needed.
- Cold global scalar loads (scales, gate) sit on the critical path of a ~10 µs kernel; read
  them at kernel start.
- q_len 8 in the fused kernel stayed at ~630 µs for b32 full (FA3: 383) across six variants:
  more or fewer warps per KV head, FP8 vs f16 P.V, K/V converted once per stage. The loop runs
  at ~3.6k SM-cycles per 64-token tile, with tensor, MUFU and conversion pipes each 15-30%
  busy and IPC near 1. It looks latency-bound by the per-tile barrier structure; a
  warp-specialized pipeline (producer warps, mbarriers, no block-wide barrier per tile) is the
  untested next step. Verify stays on idea 1 + FA3.
- The bench (import-time graph replays) and the judge disagreed on b1 full decode (bench:
  fused 34.0 vs idea 1 35.5 µs; judge: 29.4 vs 27.5). Route on judge numbers.

- Scattered-gather ceiling on H200 (cold L2, rows through a shuffled page table, b32 at 80K):
  128-byte rows reach 3.6 TB/s (181 µs for the 655 MB of K and V), 64-byte rows 1.8-2.2 TB/s.
  FA3 reads 64 bytes per KV head, which is why it sits near 375 µs at b32 whatever q_len is.
  One dependent page entry + row costs 1343 cycles cold; a row alone 741 (job 8cb557afaeab4e08).
- K/V reuse: every 128-byte row is read once per split and serves 2 x 64 query rows (8 tokens x
  8 heads per KV head). FA3 also reuses across the block, so bytes are equal; the gain is the
  row size of the gather, not fewer bytes.
- q_len 8 on wgmma (attn_ws.cu): Q.K in FP8 with f16 accumulators, softmax in half2 with
  f16x2 ex2, P.V in f16 with the values converted by the loader warps. Error vs an FP32
  reference matches FA3's (7.0e-2 vs 7.1e-2 on the bench inputs).
- A wgmma left in flight across a loop back-edge made ptxas serialize every wgmma in the kernel
  (C7514). Retiring each async group inside the iteration avoids it; two tiles per iteration
  can still overlap (Q.K of the second under the first softmax).
- One loader warp could not issue a tile's 1024 cp.async in time (2735 cycles per tile); four
  can. A loader-side fence.proxy.async after its own cp.async.wait_group cost ~3% at b32 full;
  the math warps' fence after acquiring the full barrier is enough (var 2, errors unchanged).
- E4M3-to-f16 by integer ops (PRMT, SHF, LOP3, exact, scaled by 2^-8) was slower than F2FP.
- The bench and the judge disagreed again: paired tiles tied var 0 in the bench on sliding
  verify but lost 6-13% on the judge (1f4f986279834925). Route on judge numbers.
- Where the time still goes (b32 full verify, 333 µs vs 181 µs gather ceiling): the loader
  warps' cp.async issue (~600 cycles per tile) and value conversion (~800) now set the pace
  (load-only mode alone is ~300 µs); the math warps spend ~1500 cycles per tile (softmax ~900).
  Untested next steps: TMA row loads (a 1-row box per K/V row with 128-byte swizzle lands rows
  in the SW128 K-major layout and removes the per-lane cp.async issue), FP8 P.V with values
  transposed instead of converted, and merging sliding-layer splits inside the kernel (the
  merge launch is 16-24% of sliding verify time).

## FP32-accumulator ws kernel (10-28, not yet checked)

FA3 accumulates Q.K in FP32 and runs its softmax in FP32; attn_ws v5 used f16 accumulators and a
half2 softmax. VAR bit 4 (var 16, 18 without the loader fence) matches FA3's precision: Q
rounded to E4M3 without the softmax scale (as SGLang hands FA3), Q.K on
`wgmma.m64n64k32.f32.e4m3.e4m3`, running max, sum, rescale factors and exp2 in FP32 (one FFMA and
one `ex2.approx.ftz.f32` per score, scale_log2 = sm_scale x 5.473 x log2 e x k_scale, as FA3's
`scale_apply_exp2`), P rounded to f16 for P.V with FP32 accumulators. FA3's FP8 path converts P
to E4M3 (`exp2(x + 8)`, `Max_offset = 8`) before P.V, so f16 P is finer than FA3's. Source read:
Dao-AILab/flash-attention `hopper/softmax.h`, `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
(`convert_type_out(tOrP_acc, tOrP)` into `Element`), `hopper/flash_fwd_kernel_sm90.h`
(`Softmax<..., Max_offset = !Is_FP8 ? 0 : 8>`) at main 98c1cee; SGLang's sgl-kernel builds its
FA3 from this `hopper/` code, but its pinned commit was not checked. Variants: bit 5 (48, 50)
adds FA3's intra-warpgroup overlap (P.V of tile i - 1 issued behind Q.K of tile i, retired
before the back-edge, P in ping-pong arrays); bit 6 (80, 82, 112, 114) computes a quarter of the
exponentials on the FMA pipe (`ex2_poly`, degree 5, max relative error 1.7e-7 over [-126, 0]).

Local checks (commit 72acd7c + routing table):
- ptxas (`tools/attn_ws_quick.sh -s`, CUDA 12.8, sm_90a): f16 vars 0/2 106 registers; FP32
  serial 16/18/80/82 115; overlapped 48/50/112/114 128; no spills in any ws kernel, no C7513 or
  C7514. A first overlapped version that copied P between iterations hit C7513 (non-wgmma
  instructions defining wgmma inputs); ping-pong P arrays removed it.
- SASS, var 18: each score is `FFMA` then `MUFU.EX2`, no range fixups; P packs with
  `F2FP.F16.F32.PACK_AB`. Var 50: `QGMMA.64x64x32.F32.E4M3.E4M3` x2 then `HGMMA.64x64x16.F32` x4
  issued back to back, `WARPGROUP.DEPBAR.LE gsb0, 0x1` before the softmax, `0x0` after it.
- Numerics (`uv run --with numpy python tools/attn_emulate_fp32.py --seeds 4 --keys 16384`),
  rel L2 vs FP64 with unrounded queries, mean: sliding f16 5.90e-2, fp32 5.60e-2, fa3 5.66e-2,
  qfp8 reference 5.60e-2; full f16 3.22e-2, fp32 3.13e-2, fa3 3.20e-2, qfp8 3.13e-2. Against FP64
  over each path's own rounded queries (accumulator + softmax + P error only): f16 1.1-1.8e-3,
  fp32 3.8-6.4e-5, poly same as fp32, fa3 5.2-7.5e-3. The FP32 path's error is below FA3's.
- Host wrapper (`tools/attn_host_check.sh`): no errors. `tools/attn_compile_check.sh` on
  submission.py and bench.py: no errors (the two spilling entries are the unchanged q_len 1
  fused kernel in `attn_src`).

### Next checks (after the 00:00 UTC reset on 10-29), in this order

1. Bench: `muse_attention_decode/bench.py`, already generated by
   `python3 tools/attn_make_bench_ws.py _ref32 _fused_prep _ws@x=0 _ws@x=2 _ws@x=16 _ws@x=18
   _ws@x=48 _ws@x=50 _ws@x=80 _ws@x=82 _ws@x=112 _ws@x=114`. From `muse_attention_decode/`:
   `okc check bench.py --no-wait --json`, then `okc status <job> --wait --json`; confirm problem
   and job id. Its BENCH lines time f16 (x=0 sliding default, x=2 full default) against every
   FP32 variant per verify shape, with e= against the FP32 reference (_ref32 first). Its STAMPS
   lines are var 18. Its judge score is the FP32 serial routing (16 sliding, 18 full).
2. Final: set `_WS_VAR` in `submission.py` per verify shape to the fastest FP32 variant from
   the bench, and to `None` where `_fused_prep` (prep + FA3) is faster; keep only the routed
   variants' `INSTANTIATE` lines and dispatch branches if compile time matters; edit
   `attn_ws.cu`, run `python3 tools/attn_inline_ws.py`, `tools/attn_ws_quick.sh`,
   `tools/attn_compile_check.sh muse_attention_decode/submission.py`. Then
   `okc check submission.py --no-wait --json` and `okc status <job> --wait --json`. Compare
   with 1.291x (leaderboard, fd376b11ca9b49ef) and 1.348x (f16 ws, dcea57a9e84f441a).

## Pending bench: backlog #4 plus A1 bounded-shift softmax (10-28, not yet checked)

A1 (backlog section 5) is in `attn_ws.cu` as VAR bits 7-8 on top of var 16 (FP32 path, loader
fence kept): var 144 is tau 4, var 272 is tau 8. Each math thread keeps its two rows' retained
shift c in `mrow` (log2 units, FP32). Per tile it takes the max over unmasked scores; if that
max exceeds c + tau (or the row had no valid key yet), c moves to it and l and O are rescaled
together by 2^(c_old - c_new), else c stays and P = 2^(S * sl2 - c) can reach 2^tau. The O
rescale stays behind var 16's `__any_sync` vote, so a warp skips it only when none of its 16
rows moved c. Partials store c instead of the row max; `ws_combine_kernel` already weights
each partial by 2^(c_s - max c), and l_s and O_s are both relative to c_s, so the merge needs
no change. Bits 7-8 clear compile to var 16 unchanged (SASS of var 16 is instruction-identical
to commit 1f49f32's).

Range: P <= 2^tau (plus one FP32 rounding of c), so with tau 8 P rounds to at most 256 in f16
(max 65504). Per-thread l <= keys x 2^8 and |O| <= keys x 2^8 x 448 (E4M3 max, before
v_scale): over 1,048,584 keys that is 2.7e8 and 1.2e11, against FP32's 3.4e38. The merge
weights are at most 1, so the merged sums stay within 128 times those.

Local checks (CUDA 12.8, sm_90a):
- ptxas: var 144 and 272 use 113 registers in mode 0 (var 16: 115), 128 in mode 2; mode 1
  (loads only) 61. No spills in any ws kernel, no C7513 or C7514.
- SASS of var 272: the A1 decision is `FADD c, 8` / `FSETP.GT` / `FSEL` per row, then var 16's
  `FSETP.NEU alpha, 1` x2, `VOTE.ANY` and `@!P0 BRA` over the 32 `FMUL`s of the O rescale, in
  both tile32 call sites (loop and new-token tile).
- Numerics (`uv run --with numpy python tools/attn_emulate_a1.py --seeds 3`): the tiled
  recurrence with FP32/f16 rounding, splits and merge, rel L2 against FP64. Total error is the
  same to three digits for tau 0, 4 and 8 in every case (worst 8.41e-2, full outlier, judge
  bound there ~0.15). Error against FP64 with the same E4M3 queries (accumulators, softmax and
  P rounding only), worst over cases: var 16 9.8e-5, tau 4 1.5e-4, tau 8 2.0e-4; the rise is
  the dominant key's P no longer being exactly 1 in f16. Share of (warp, tile) steps that run
  the O rescale, var 16 / tau 4 / tau 8: sliding soft 0.96 / 0.68 / 0.24, full coherent 80K
  0.18 / 0.06 / 0.03, full edges 6K 1.00 / 0.95 / 0.71, full ramp 0.19 / 0.05 / 0.02. Max P
  seen at tau 8: 255.2.

`muse_attention_decode/submission_a1.py` (written by `python3 tools/attn_build_a1.py`) is
submission.py with this attn_ws.cu and var 272 wherever var 16 was routed (sliding verify at
every batch, full verify above batch 2). submission.py is unchanged (var 16).

`muse_attention_decode/bench.py` (same tool) is submission.py with this attn_ws.cu, routing
unchanged, and an import-time sweep on each scored shape (b32/b16/b1 full and sliding verify,
plus b32 full decode), each a CUDA-graph replay after a 256 MB flush, median of 40:
`_ref32` (FP32 torch reference, the e= baseline), `_fused_prep` (prep + FA3), `_ws@x=0` (f16,
fenced), `_ws@x=16` (submitted), `_ws@x=16,m=1` (var 16's loaders with math warps that only
wait and release), `_ws@x=144` and `_ws@x=272` (A1). Then, per q_len 8 shape, STAMPS lines for
x=16, 144 and 272: math-warp cycles per tile (wait-data, qk, softmax including the rescale,
pv, tail, prologue) and loader cycles per tile. No GATHER part.
Since DS3 (below), bench.py also times var 528 (page-size-1 loaders) and stamps it; written by
`tools/attn_build_loader.py` instead.

How to read it:
- Loader-branch vs math-branch: on full verify b16/b32, if x=16 is close to x=16,m=1, the
  loaders set the pace (loader-branch: N2, A4). If x=16 is clearly slower, the math warps do
  (math-branch: A1, A5). Var 16's STAMPS agree if wait-data is small and softmax large
  (math-bound) or wait-data dominates (starved by the loaders).
- A1: compare x=144 and x=272 with x=16 per shape; their e= should match x=16's to about
  1e-4. If their STAMPS softmax cycles drop but the BENCH time does not, the loaders still set
  the pace and A1 stops there (backlog gate).

## DS3: loader index arithmetic for page size 1, var 528 (10-28, static audit, not yet checked)

Question: can var 16's loader warps (four warps, about 600 cycles per tile issuing cp.async and
800 converting V in the clock64 stamps) do the same loads with cheaper address and index
arithmetic? Static audit only; nothing here is measured on a GPU.

What var 16's loader loop does per tile, page size 1 (`attn_ws.cu`, `if (warp >= 8)`):
- `i % NS`, `i / NS` and `f % NS` (NS = 7) as magic-number divisions (`IMAD.WIDE.U32 ...,
  0x24924925` then `LEA.HI`, `SHF`, `IMAD`), recomputed for the empty-barrier wait, the stage
  address and the full-barrier arrive.
- Per row a lane issues (two rows, four cp.async each): a runtime `psize != 1` branch around a
  32-bit division sequence (`I2F.RP`, `MUFU.RCP`, `F2I`, `IMAD.HI`; skipped at run time with page
  size 1, but its setup and branch are paid), the row and `row * 128 + ck * 16` in 64 bits
  (`SHF.R.S32.HI`, `SHF.L.U64.HI`, two `LOP3`), four 64-bit adds to the K and V bases (`IADD3`
  + `IMAD.X` each) and eight `SEL`s for `valid ? ptr : base`, since each of the four pointers
  keeps its own 64-bit value.
- The page-entry prefetch (4 tiles ahead) checks `tile >= ntiles || pos >= hi`, branches on
  `psize == 1` around another division, then sign-extends and scales `pos` into the address.

Var 528 (VAR bit 9 on var 16; bit 2 clear, so the loaders' `fence.proxy.async` stays) keeps every
load, the 128-byte combined row, the src-size-0 tail mask, the -1 padding treatment (entries past
`hi` are never read; the prefetch returns 0 there, as before), the cp.async commit/wait sequence,
the mbarriers and all fences, and changes only the index arithmetic:
- ring indices as counters (`sl = i % NS`, `par = ((i / NS) + 1) & 1`, `sc = (i - DA) % NS`,
  `f % NV` as a mask);
- no page-size division or branch: the cache row is the page entry;
- one range compare per prefetched entry (`tile * BN < hi - lo - pt0` implies `tile < ntiles`)
  and a per-lane page-table pointer;
- per-lane base pointers `k_cache + cq * 16`, `v_cache + cq * 16` and shared offsets hoisted out
  of the loop; the second 16-byte chunk of a row (+64 bytes) and the second row's shared
  offsets (+512 K, +1,024 V) become instruction immediates;
- out-of-range lanes use row 0 instead of the bare base pointer. They still copy zero bytes
  (src-size 0), and row 0 keeps the unread address inside the cache.

No 32-bit offset was needed: the row offset is `uint64_t(uint32_t(entry)) * 128`, formed in 64
bits (`IMAD.WIDE.U32 r, e, 0x80`), so it cannot overflow and no bound on the cache size enters.
A `mad.wide.u32` written in inline PTX saved only 8 instructions per 4 tiles (ptxas splits it
anyway and shares `row * 128` between K and V), so it was dropped.

The host runs 528 only when `config["page_size"] == 1` (`attn_ws` resets var 528 to 16
otherwise, for mode 0, 1 and 2), so the page-size-64 tests keep var 16's generic path.

SASS (CUDA 12.8.1, sm_90a, `tools/attn_ws_quick.sh -s`), counted by
`python3 tools/attn_loader_sass.py .build/ws/k.sass ILi7ELi3ELi0ELi4ELi16E ILi7ELi3ELi0ELi4ELi528E`.
The loader loop is the backward branch holding the LDGSTS (one trip is PF = 4 tiles). Static is
every instruction in it; path is one steady-state trip with page size 1 and full tiles (division
skips and successful try-waits taken, slow paths not):

| class (per 4 tiles) | var 16 static | var 528 static | var 16 path | var 528 path |
|---|---|---|---|---|
| integer/address (IMAD, IADD3, LEA, SHF, LOP3, SEL, ISETP, LDC, ...) | 870 | 310 | 519 | 254 |
| I2F/F2I/MUFU (division) | 36 | 0 | 0 | 0 |
| LDGSTS + LDGDEPBAR + DEPBAR | 40 | 40 | 40 | 40 |
| LDG (page entries) | 4 | 4 | 4 | 4 |
| SHFL.IDX | 8 | 8 | 8 | 8 |
| LDS/STS (incl. `@!PT LDS` scheduling no-ops) | 88 | 80 | 88 | 80 |
| F2FP (V to f16) | 128 | 128 | 128 | 128 |
| SYNCS/MEMBAR/FENCE | 20 | 20 | 16 | 16 |
| branch/BSSY/BSYNC | 72 | 44 | 56 | 28 |
| other | 4 | 4 | 0 | 0 |
| total | 1270 | 638 | 859 | 558 |

Steady-state path: 859 to 558 instructions per 4 tiles (-35%), integer/address 519 to 254
(-51%). The cp.async issue section of one tile (tile check to LDGDEPBAR) runs 113 instructions
in var 16 and 53 in var 528. The loads-only builds (mode 1) give the same path counts, 855 to
558; mode 2 (stamps) 923 to 632. Unchanged in both: 32 LDGSTS.E.BYPASS.128, 16 LDS.128 + 32
STS.128 and 128 F2FP.F16.E4M3.UNPACK_B (the conversion), 4 x (MEMBAR.ALL.CTA, FENCE.VIEW.ASYNC.S,
SYNCS.ARRIVE) per trip.

One row (tb = 0) of tile u = 0, executed instructions. Var 16 (division block skipped):

```
IMAD.WIDE.U32 R32, R17, 0x24924925, RZ ; IMAD.IADD ; LEA.HI ; SHF.R.U32.HI
IMAD R37, R32, -0x7, R17          (i % 7)
ISETP.NE.AND P1, PT, R36, 0x1, PT ;  SHFL.IDX ; SHF.R.S32.HI ; IMAD.MOV.U32 ; IMAD.IADD
LDC.64 (k_cache) ; LDC.64 (v_cache) ; @!P1 BRA (over 30-instruction division)
IMAD.SHL.U32 R44, R43, 0x80 ; SHF.L.U64.HI R46, R43, 0x7, R46 ; IMAD x3 (stage) ; ISETP (valid)
LOP3 x2 (| ck*16) ; IADD3 x4 + IMAD.X x4 (64-bit adds) ; SEL x8 (valid ? ptr : base)
IADD3 x2, VIADD, IMAD.IADD (shared addresses) ; LDGSTS x4
```

Var 528:

```
IMAD R29, R17, -0x40, R32        (rem)
IMAD R33, R28, 0x4000, R13 ; IMAD R35, R28, 0x4000, R21 ; IMAD.IADD R37  (stage offsets)
SHFL.IDX ; LDC.64 (k base) ; ISETP.GE (valid) ; LDC.64 (v base) ; SEL R26, R26, RZ, !P0
IMAD.WIDE.U32 R26, R26, 0x80, RZ ; LOP3 (| cq*16) ; IADD3 x2 + IMAD.X x2
LDGSTS [R33], [R30.64] ; LDGSTS [R37], [R26.64]
LDGSTS [R33+0x1000], [R30.64+0x40] ; LDGSTS [R37+0x40], [R26.64+0x40]
```

The second row reuses the stage offsets (`[R33+0x200]`, `[R37+0x400]`, `+0x1200`, `+0x440`).

Registers and spills (ptxas): var 528 uses 115 registers in mode 0 (var 16: 115), 62 in mode 1
(61), 128 in mode 2 (128); no spills, no C7513/C7514. Var 16's SASS is instruction-identical
before and after the change (modes 0 and 1). Var 528's math-warp code has the same instructions
as var 16's apart from register allocation (16 more IMAD.MOV.U32, one more IMAD.IADD and
VIADD, one IADD3 and one CS2R fewer; the same 8 HGMMA and 4 QGMMA). Host wrapper
(`tools/attn_host_check.sh`): no errors.
`tools/attn_compile_check.sh muse_attention_decode/bench.py`: no errors; the two spilling
entries are the unchanged q_len 1 fused kernel in `attn_src`.

What this does not show: fewer issued instructions need not mean fewer cycles. The loader's
issue time may be bound by LDGSTS throughput or the memory system rather than by integer issue,
and the V conversion (128 F2FP + 48 LDS/STS per 4 tiles, about 800 cycles per tile in the
stamps) is untouched. Decision (drop under 10% would have closed this): the path count fell by
35%, so var 528 goes into the pending bench.

`muse_attention_decode/bench.py` is now written by `python3 tools/attn_build_loader.py`, which is
`tools/attn_build_a1.py`'s bench with two more variants and one more stamp variable, and leaves
`submission_a1.py` and `submission.py` unchanged. It times `_ref32 _fused_prep _ws@x=0 _ws@x=16
_ws@x=16,m=1 _ws@x=144 _ws@x=272 _ws@x=528 _ws@x=528,m=1` per shape and prints STAMPS for x=16,
144, 272 and 528 on each q_len 8 shape (24 STAMPS lines of about 360 characters plus 7 BENCH
lines: about 11.5K characters, under the 16,000-character tail; the last bench with 12 STAMPS
lines logged 5.2K). Read it with the backlog #4 notes above, plus:
- x=528 against x=16 and x=528,m=1 against x=16,m=1 per shape: e= of x=528 should equal x=16's
  (same loads, same math). If loads-only time drops but the full kernel's does not, the math
  warps set the pace on that shape.
- STAMPS x=528 against x=16: the `cp.async-issue` and `page-entries` loader columns are the ones
  this change can move; `v-convert+release` should not change.

- 10-28 submit 971e1ef49fc647aa: `submission_528.py` (var 16 with DS3's page-size-1 loaders,
  var 528), 1.292x (1.273-1.317), 16/16, vs var 16's 1.321x (cb42811004a9440c). Verify sliding
  b16 20.1 -> 19.1 us (-4.7%), b32 25.7 -> 24.2 (-5.6%); verify full b16 191.3 -> 204.2 (+6.7%),
  b32 357.6 -> 397.4 (+11.1%). Decode shapes (not on this kernel) moved +-1-3%. Fewer loader
  instructions (859 -> 558 per 4 tiles) helped short windows and hurt 80K contexts, so issue
  cost alone is not the full-context limit; untested guess: the changed LDGSTS order (second
  64-byte chunk via immediate offsets) hurts how scattered reads are served. The pending bench
  times var 528 and var 16 loads-only side by side to separate loader speed from interaction.
  Entry stays var 16.

## One-launch kernel, attn_one.cu (10-28, built and checked locally; judge checks from 10-29)

Re-implemented from the public 2.58x entry's design memo (one launch per call, fixed-frame
merge): `muse_attention_decode/attn_one.cu`, inlined as `_ONE_SRC`, routed for every tp8
shape by `kernel_fn` (`_one_ok`, `_one_plan`); dp heads stay on prep + FA3. Interface,
buffers and plans: `docs/attention-port-guide.md`.

What is in the kernel: per-request split-KV with whole-tile chunks by magic-number
division; a 3-stage cp.async ring with page indices two tiles further ahead; the query
group (q rows, new k/v rows, constants, gate prefetch) in flight while the first K/V tiles
are issued; one `sincosf` body (Cody-Waite + `__sinf`/`__cosf`); q_len 1 on f16
`mma.sync` with K and V converted from E4M3 in registers (one 16-byte LDS per key per
thread, 4 HMMA per 8 keys, no ldmatrix; values by 8-byte loads + `prmt`); q_len 8 on
`wgmma.m64n64k32.f32.e4m3.e4m3` reading K in place from the 128B-swizzled stage, P rounded to
E4M3 under a lazy offset (moved only when a tile's max exceeds it by 8 log2 units),
`wgmma.m64n72k32` against V transposed per tile by `ldmatrix.x4.trans` + `prmt` +
`stmatrix.x4` into a 64B-swizzled K-major buffer in P's key order with rows 64..71 of ones
(the row sums come out of the same wgmma); `fence.proxy.async` before every wgmma read of
generic-written smem (loader side, kept per the repo rule); merge by `red.global.add.v4.f32`
of o 2^m and `red.global.add.f32` of l 2^m into a persistent workspace; last-CTA finalize
through a key-count ticket (`atom.acq_rel.gpu` after a CTA barrier) or cooperative finalize
(parity counters, epoch, `ld.acquire` spin with `__nanosleep`, trap after 2^24 polls); the
finalizer zeroes what it read. Modes per instance (MODE bits 1 plain / 3 coop, |4 one tile,
|8 speculated length from `page_table.shape[1] - L`, |16 flat). Flat mode (full context,
batch > 1): 264 CTAs share a virtual key axis with 400 / 600 keys per request boundary,
whole-tile segments per request, per-request tickets at q_len 1 and a grid-wide cooperative
finalize at q_len 8 (every CTA finalizes B x 128 / 264 rows). The flat loader is the same
cp.async ring with a loop-top barrier, not yet the memo's producer warpgroup.

Local evidence (CUDA 12.8.1 in Docker, sm_90a):
- ptxas, no spills, no C7513/C7514: q_len 1 modes 1 / 3 / 11 / 13 / 17: 104 / 105 / 107 / 79 /
  119 registers (58.9 KB smem: three CTAs per SM with the 1 KB reservation, 396 resident);
  q_len 8 modes 1 / 3 / 11 / 15 / 19: 104 / 104 / 104 / 80 / 110 (90.6 KB, two per SM).
  `tools/attn_host_check.sh`: no errors;
  `tools/attn_compile_check.sh muse_attention_decode/submission.py`: all blocks build.
- SASS: 29-45 KB per instance (the old fused kernel: 98.6 KB), 2 `MUFU.SIN/COS` per
  instance (one body; the old kernels inlined 17-20), the v4 reds are single `REDG`
  instructions. Mode 13 (decode 1 sliding) once-path: the `seq_lens` load is instruction 7,
  the index copies land at 68 with the page-size division branched around, 65 `BRA` in 1872
  instructions.
- `tools/attn_emulate_one.py`: the mma.sync / wgmma fragment maps, the SW128 K read, the V^T
  transpose (key permutation, dim order, ones rows), the P register order and the merge's
  4-dim groups all agree with Q K^T and P V on random tiles. Numerics vs FP64 (soft keys,
  33 / 16 splits merged in the fixed frame): sliding q8 5.54e-2 (E4M3-query reference
  5.53e-2, bound 0.150), q1 4.7e-4; full 16K keys q8 3.20e-2 (reference 3.12e-2, bound
  0.093), q1 2.4e-4; about one lazy-offset move per (row, split).
- Plans (`_one_plan`, checked statically for all 16 tests and 12 scored shapes): decode 1
  sliding 32 CTAs mode 13; verify 1 sliding 33 x mode 15; decode 16 / 32 sliding 16 x 16 /
  8 x 32 mode 1; verify 16 / 32 sliding 8 x 16 / 4 x 32 mode 3; batch 1 full 252 x nc 8
  mode 11; batch 16 / 32 full 264 flat (17 / 19); page-64 and short tests on the plain
  multi-tile instances.

### First checks after the 00:00 UTC reset (10-29), in this order

1. `cd muse_attention_decode && okc check bench.py --no-wait --json`, then
   `okc status <job> --wait --json`; confirm problem and job id. Read the SELFTEST lines
   first: every test shape and scored shape with its plan, error, bound, cache-row errors,
   and a second call on the same inputs (workspace left zero). Then the BENCH lines:
   `_one` against `_fused`, `_ws`, `_fused_prep` per scored shape with e= columns.
2. If SELFTEST is clean: `okc check submission.py --no-wait --json` for the judge's
   per-shape times; log them against the bars (decode 1 sliding 4.9 us, 16: 7.4, 32: 9.3;
   verify 1 sliding 7.0, 16: 11.9, 32: 15.4; decode 1 full 11.0, verify 1 full 14.7; decode
   16 / 32 full 88 / 170; verify 16 / 32 full 105 / 186).
3. Route per shape on judge numbers (`_ONE_ROUTE`: a `(layer, q_len, batch)` key set to
   False sends that shape back to the previous path), then the next check.

### Producer-warpgroup loader for the flat kernel (10-28, compiled, not yet checked)

Mode |32 (instances 49 for q_len 1, 51 for q_len 8): a 128-thread producer warpgroup gathers
tiles into an mbarrier ring (3 stages at q_len 1, 4 at q_len 8; full barriers count the 128
producer arrivals by `cp.async.mbarrier.arrive.noinc`, empty barriers the consumer threads'
`mbarrier.arrive`), page entries three tiles ahead in registers, the block's own rows patched
by the consumers after the full wait with a `fence.proxy.async` and a consumer barrier,
counters continuous across a CTA's segments, producers leave only after every stage was
released. Register split `setmaxnreg.dec 56 / .inc 200` (q_len 1) and `32 / 104` (q_len 8)
at two CTAs per SM. ptxas (CUDA 12.8.1): mode 49 128 registers, no spills; mode 51 80
registers with 168 bytes of spills, and its SASS shows the consumer region never using a
register above R78 after `USETMAXREG.TRY_ALLOC 0x68`: ptxas caps every region at the
launch-bounds count and pads the kernel's count to it. So mode 115 (|64: one CTA per SM, no
split) is built as the comparison: 113 registers, no spills. `_one_plan` prefers 49 / 51 with
`_ONE_PROD = True` and falls back to 17 / 19; `bench.py` times `_one` and the explicit modes
17 / 49 / 19 / 51 / 115 per full-context shape, and its STRESS lines replay six shapes 2,000
times (decode and verify 1 sliding: speculated one-tile launches; verify 16 sliding:
cooperative; decode 16 full: flat tickets; verify 16 full: grid-wide finalize; decode 1 full:
252 splits with 8 copies) and read the workspace back.

### Protocol review applied (10-28)

The independent review (scratchpad `audit/attn_one_review.md`; verdict: release/acquire chain,
key-count targets, cp.async group counts and proxy fences sound under stream ordering) asked
for six changes, all in: (1) cooperative modes launch through `cudaLaunchKernelEx` with
`cudaLaunchAttributeCooperative`, probed once under graph capture (`attn_one_probe_coop`),
with a one-SM residency headroom when the probe fails, and the longest cooperative spin
reported through `p.dbg` (`_ONE_DBG`, printed per self-test shape as max_polls); (2) the
self-test asserts the workspace invariant (rows, row sums, tickets, next-parity counters) after
every shape and after the replay stress, compares replays with a tolerance, and forces the
SPEC fallback on the batch-1 shapes by shrinking `seq_lens` in place (positions and cache_loc
updated to match) for 50 replays with the judge-style check; (3) the flat epoch is read with
`__ldcg`; (4) one lane reads and zeroes a row's sum and broadcasts it by `__shfl_sync` in a
uniform loop; (5) the port guide states the PDL ordering rule (epoch reads, tickets and reds
after `griddepcontrol.wait`) and the one-stream rule; (6) q_len 1 residency corrected to three
CTAs per SM (58.9 KB + 1 KB reserve). Also from the review: the ticket test is `>=` and a
zero-key CTA never acts as the cooperative last arriver.

### Scored submits of 10-28 (authorized in place of the exhausted checks)

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 10-28 | e9138b431fc3461f | 2f0a1b9 | bench.py (submit) | timeout at the 15-minute cap, returncode -9, no score | All four extensions compiled, `cooperative_launch=True` (the probe captured a cooperative launch into a graph), residency as computed (q_len 1: 396 = 3/SM, q_len 8: 264 = 2/SM, mode 115: 132). The first self-test shape (batch 1, ctx 10, sliding, page 64; plan 32 x mode 1) never printed, so its kernel hung; earlier runs of the whole suite took 422-447 s, so compile time is not the explanation. The SASS of that instance shows no barrier reachable by part of a CTA and no unbounded loop. Log: `profiles/bench.e9138b431fc3461f.log`. |
| 10-28 | a9c92e9ff8d048b0 | 766cab6 | bench.py (submit) | the first kernel call hung; the watchdog ended the run after 226 s (returncode 112), no score | Compile of the four extensions took about 3 minutes (the bench clock read t=0 at the residency line, 226 s into the run). First call: `MODES b1 sliding L1 mode 13` (32 CTAs, one tile each, speculated length; every CTA non-empty) was still running after 20 s with no fault raised, so the hang is a spin or deadlock on the common per-request path, not an empty-CTA or page-64 effect. Static checks of that instance's SASS: the two cp.async commits, `wait 1`, barrier; the K/V commits, `wait 2`, barrier; RoPE barrier; new-token barrier; loop-top `wait 1` + barrier; ticket barriers, all reached by every thread; every guarded exit reconverges (the speculative empty return jumps to the exit with its predicate forced true, the fall-through forces it false); no loop with a data-dependent bound. Unexplained by inspection. Log: `profiles/bench.a9c92e9ff8d048b0.log`. |
| 10-28 | 9c4d3c649f944742 | 9221bf9 | bench.py (submit, `--no-probe`) | timeout at the 15-minute cap, returncode -9, no score | `cooperative_launch=False`; the workspace read zero before the first launch. The phase sweep on batch 1 sliding q_len 1 mode 13 printed `phase 1` through `phase 7 returned` at t=0s, then nothing: the phase-8 call (the new-token pass and the barrier after it) never returned and the watchdog never printed its HANG line. Every stamp column printed `-` even for phases that returned. Log: `profiles/bench.9c4d3c649f944742.log`. |

Diagnostic for the first check of 10-29 (`bench.py` at commit after 766cab6, generated with
`--no-probe`): `Params.phase_limit` makes every thread drain its copies and return after phase
1..10 of the per-request path (index group, query group, first wait + barrier, speculation
check, K/V prologue + wait + barrier, RoPE, query pass, new tokens + barrier, patch, tile loop)
without writing anything; the self-test sweeps the phases on batch 1 sliding (modes 13 and 1)
and on the ctx-10 page-64 shape under the 20-second watchdog, so the last `PHASE ... returned`
line names the phase that hangs. The same run has the cooperative-launch probe disabled
(`_ONE_COOP_LAUNCH = False`), the only device activity before the first launch that the earlier
kernels never had; if the whole self-test then passes, the probe was the cause and cooperative
grids keep one SM of headroom instead. The third authorized submit of 10-28 was not used: the
fix is not certain.

### Cause of the hang: a full-mask shuffle under a lane-divergent branch (10-28)

Phase 8 of the per-request path is the new-token pass (`hi > seq && tid < L * 32`: the block's
own K and V rows are normalized, rotated, rounded and written to the cache) followed by the
CTA barrier. In that pass each warp holds K rows in lanes 0..15 and V rows in lanes 16..31
(`is_k = (rr & 3) < 2`, `rr = tid >> 3`), and the K branch called `oct_sum`, three
`__shfl_xor_sync(0xffffffff, ...)`. The full mask names the V lanes, which never execute the
shuffle; that is undefined behaviour, and the SASS of `attn_one<1,13>` shows how it hangs:
`@P4 BRA` sends the V lanes past the K branch toward the `BAR.SYNC`, so the warp is diverged at
the shuffle, `BRA.DIV` takes the K lanes to the `WARPSYNC.COLLECTIVE` fallback with mask
0xffffffff, and that waits for V lanes that are parked at the barrier waiting for the K lanes.
The pass runs only in the CTA whose range holds the new token, and every per-request instance
has it (q_len 8 splits each warp the same way), which is why every shape hung on its first
call. The earlier static reviews checked that every thread reaches each barrier, but not lane
convergence at warp collectives. The other collectives in `attn_one.cu` sit in warp-uniform
code (checked: the finalize loop, the flat prefix scan in warp 0, the tile-loop `__any_sync`
and `quad_*` reductions).

Fix: the sum of squares and `oct_sum` run on every lane of the warp, and only the
normalization, RoPE and scale choice stay under `is_k` (the 8-lane groups are aligned to
`is_k`, so the K sums are unchanged). In the rebuilt SASS of every instance the shuffle group
now precedes the `is_k` branch, with only the warp-uniform `hi > seq && tid < 32` branch before
it.

Why every stamp printed `-`: `stamp(k)` writes `dbg[1 + k]`, but `_ob_stamps` took `dbg[1]`
(never written, zeroed before each call) as the base and so printed `-` for everything. Fixed
in `tools/attn_make_bench_one.py` (base `dbg[2]`, stamp k at `dbg[1 + k]`). Why no HANG line
appeared: the watchdog's snapshot allocated its pinned host buffer after the hang, and a
page-locked allocation implicitly synchronizes with the device (CUDA programming guide,
implicit synchronization), so the process blocked there until the judge killed it. The side
stream, pinned buffer and event are now made before the first launch.

Local evidence (commit after 8d1789d): `tools/attn_ws_quick.sh -s` builds all 13 instances
for sm_90a (mode 13: 78 registers, no spills; only mode 51 spills, 292 bytes of stores at 80
registers; the producer-loader entry above recorded 168 bytes at an earlier commit); `tools/attn_compile_check.sh` compiles all four
inlined blocks of `submission.py` without errors; `tools/attn_harness_dry.py` passes (exact
PASS, perturbed FAIL on all five shapes); a CPU check of `_ob_stamps` on a simulated dbg buffer
prints deltas for stamps 1..10 and `-` after.

Next check after the reset: `okc check muse_attention_decode/bench.py --no-wait --json`, then
`okc status <job> --wait --json`. Every `PHASE ... returned` line through phase 10 for both
mode 13 and mode 1, and `MODES ... PASS`, confirm the cause; the run then goes on to the
correctness suite, the replay stress and the timings. A `HANG` line names a second hang, with
the stamps showing the last point thread 0 of CTA 0 reached. A timeout with no `HANG` line
would mean the snapshot copy itself cannot run beside the hung kernel.

### Scored run 16fcddf80a714e0d (bench.py at 29bc072): 1.594x, CI 1.570-1.618

The first run of the one-launch kernel to finish: every judge test passed, SELFTEST had 0
failing shapes, the phase sweep returned through phase 10, and the result is on the board
(submitted with the user's authorization). Profiles: `profiles/bench.16fcddf80a714e0d.*`.
Per shape (judge medians, us; weight from the task):

| shape | SGLang | ours | speedup | weight |
|---|---|---|---|---|
| decode 1 sliding | 21.1 | 6.0 | 3.5 | 46 |
| decode 1 full | 31.6 | 15.4 | 2.06 | 16 |
| decode 16 sliding | 26.9 | 9.7 | 2.79 | 46 |
| decode 16 full | 202.7 | 91.8 | 2.21 | 16 |
| decode 32 sliding | 31.5 | 12.4 | 2.55 | 46 |
| decode 32 full | 388.0 | 176.7 | 2.2 | 16 |
| verify 1 sliding | 23.2 | 7.8 | 2.97 | 46 |
| verify 1 full | 36.8 | 18.1 | 2.03 | 16 |
| verify 16 sliding | 29.4 | 13.2 | 2.23 | 46 |
| verify 16 full | 206.6 | 374.8 | 0.55 | 16 |
| verify 32 sliding | 34.8 | 17.6 | 1.98 | 46 |
| verify 32 full | 392.8 | 756.8 | 0.52 | 16 |

Instances on the scored path (profiled): decode 1 sliding mode 13 (6.0 us), decode 1 full 11
(14.6), decode 16 / 32 sliding 1 (10.0 / 11.4), decode 16 / 32 full 49 (89.4 / 173.5), verify 1
sliding 15 (7.3), verify 1 full 11 (17.7), verify 16 / 32 sliding 3 (12.7 / 16.9), verify 16 / 32
full 51 (374 / 759). The two verify full shapes at batch 16 / 32 are the only ones below 1x.

Mode choice for the flat (full context, batch 16 / 32) shapes, from the run's BENCH lines (us):

| shape | mode 17 / 19 (consumers copy) | mode 49 / 51 (producer warpgroup) | mode 115 (producer, 1 CTA/SM) | ws kernel |
|---|---|---|---|---|
| decode 16 full | 103.5 | 100.9 | - | 137.1 (fused) |
| decode 32 full | 189.3 | 183.4 | - | 252.1 (fused) |
| verify 16 full | 127.4 | 389.8 | 150.1 | 189.2 |
| verify 32 full | 236.3 | 773.8 | 282.7 | 352.7 |

Routing change: q_len 8 flat shapes go to mode 19 (`_one_plan` no longer tries 51); q_len 1
stays on 49, which is 3% ahead of 17. Expected on the judge: verify 16 full about 127 us
(1.62x against SGLang's 206.6, from 0.55x), verify 32 full about 236 us (1.66x against 392.8,
from 0.52x); the profiled times ran 4% below the bench times for mode 51, so these may land
slightly lower. The other ten shapes are unchanged.

Why mode 51 is three times slower than 19: its SASS (sm_90a, CUDA 12.8.1) has 105 local-memory
instructions (LDL/STL), 92 of them inside the per-tile loop that holds the four wgmmas; mode 19
has none, and mode 115 (the same producer design at 113 registers, one CTA per SM) has none
and runs 18-20% behind 19. So most of the gap is the spill: ptxas caps every region of the
two-CTA-per-SM producer build at 80 registers (the launch-bounds count), and the consumers'
score and accumulator fragments (32 + 36 floats per thread) do not fit, so each tile goes
through local memory. The rest is the producer design itself at q_len 8 (115 vs 19). Not fixed:
it needs the consumer register budget raised above what two CTAs per SM allow, which is the
same trade 115 already measures and loses.

STRESS FAIL on b1 sliding L1 mode 13 (`replayed_vs_eager=2.3e-04` against the 1e-4 bound,
finite, workspace clean): judged to be FP32 merge-order nondeterminism, not an ordering bug,
on this evidence. (1) Size: the output is 1024 BF16 elements (16 heads x 64), the smallest of
any stress shape, so one element moving by one BF16 ulp (2^-8 relative) already gives a
relative L2 of order 1e-4; the larger stress shapes (8,192 to 131,072 elements) dilute the same
event to 2e-5 or less. (2) Model: `/private/tmp/claude-501/-Users-gretchengabo-GitHub/4e9778d1-55dc-4e88-9d29-4844bd573620/scratchpad/w2/order_nondet.py` (session scratchpad) takes the emulator's q_len 1
arithmetic (`tools/attn_emulate_one.py`) for 32 one-tile partials on 2,048-key sliding cases,
sums them into the FP32 workspace in random arrival orders, finalizes (1/l, per-head RMS norm,
sigmoid gate, BF16) and compares order against order with the stress metric: over 4,000 pairs,
11% differ at all, never in more than 2 of 1,024 elements, and the largest relative L2 was
2.2e-4 from a single element (P(rel >= 1e-4) = 0.1% in the model). The model's partials are less
peaked than the judge-style soft inputs, so it likely understates how often a large element
sits near a rounding boundary. (3) What a real ordering bug would look like: a finalizer that
read the workspace before another CTA's red landed would lose a whole tile (about 1/32 of the
mass, relative error of order 1e-2 across all 64 elements of a head), and the late red would
land after the zeroing and leave the workspace nonzero; after 2,000 replays and an eager run
the workspace check reads zero rows, zero row sums and zero tickets. The same 2.3e-4 also
appears in a passing SELFTEST rerun (b4 ctx80000 full L1 mode 49, 4096 elements), where the
merge is also atomic. The bound is not loosened: the stress and SELFTEST lines now also print
`diff_elems=n/total max_ulp=... max_abs=...` (elements that differ, largest gap in BF16 ulps of
the larger magnitude), so the next run shows directly whether the difference is a few elements
at about one ulp (merge order) or many elements (a lost or stale contribution).

Local checks of the routing change (commit after 29bc072): `submission.py` and `bench.py`
regenerated and parse; pyflakes clean apart from unused imports; `_one_plan` on a CPU replica
of the residency table returns mode 19 for full context q_len 8 at batch 16 / 32 and leaves
every other shape's plan as profiled; `tools/attn_compile_check.sh` compiles all four inlined
blocks with no errors; `tools/attn_harness_dry.py` passes; `_ob_diffstat` reads 0 differences
on identical tensors and `1/1024 max_ulp=1.0` for a one-ulp change.

### Scored run b458555d3e3e4e9a (bench.py at 83e0112): 2.162x, CI 2.129-2.202

Mode 19 on the full-context q_len 8 shapes at batch 16 / 32; everything else as in
16fcddf80a714e0d. Every judge test passed, SELFTEST 0 failing shapes; now the attention board
entry (submitted with the user's authorization). Profiles: `profiles/bench.b458555d3e3e4e9a.*`.

| shape | SGLang | ours | speedup | weight |
|---|---|---|---|---|
| decode 1 sliding | 20.7 | 6.0 | 3.45 | 46 |
| decode 1 full | 31.4 | 15.3 | 2.05 | 16 |
| decode 16 sliding | 26.9 | 10.1 | 2.65 | 46 |
| decode 16 full | 200.8 | 93.6 | 2.15 | 16 |
| decode 32 sliding | 31.3 | 12.5 | 2.51 | 46 |
| decode 32 full | 385.2 | 180.8 | 2.13 | 16 |
| verify 1 sliding | 23.0 | 8.0 | 2.87 | 46 |
| verify 1 full | 36.4 | 18.2 | 2.00 | 16 |
| verify 16 sliding | 29.1 | 13.3 | 2.18 | 46 |
| verify 16 full | 205.9 | 127.1 | 1.62 | 16 |
| verify 32 sliding | 34.3 | 17.5 | 1.96 | 46 |
| verify 32 full | 390.0 | 239.1 | 1.63 | 16 |

The routing change landed as the bench predicted (127.4 / 236.3 us): verify 16 / 32 full went
from 374.8 / 756.8 us to 127.1 / 239.1. The other shapes are within the judge's run-to-run drift
of 16fcddf80a714e0d.

Stress verdict confirmed. STRESS b1 sliding L1 mode 13 again reads `replayed_vs_eager=2.3e-04`
(FAIL against 1e-4), and the new per-element line shows `diff_elems=1/1024 max_ulp=1.0
max_abs=3.9e-03`, workspace clean: one element one BF16 ulp apart, the merge-order signature the
model predicted. Mode 3 (b16 sliding L8) shows 7/131072 and mode 19 (b16 full L8) 6/131072, both
at one ulp and under the bound only because their outputs are larger. Across the log, 14 lines
report a one-ulp maximum and one SELFTEST rerun (b2 ctx100000 full L8 mode 19) reports 2/16384
elements at 10.5 ulps of the larger magnitude but 4.9e-04 absolute: a near-zero element, where
cancellation in the FP32 sum turns a last-bit difference into several ulps of a small value.
The 1e-4 relative bound stays as it is; on a 1024-element output it trips on a single one-ulp
element, so read the mode 13 STRESS line by its per-element numbers (a real lost or stale
contribution would change most of a head's 64 elements, by far more than one ulp).

### Sliding plan parity with the public entry (10-28, static)

Public plans read from `_plan`, `_TUNED` and `launch<L>` in the Apache-2.0
`muse_attention_decode.py` (132 SMs, target 264, tp8: one KV pair). Ours from `_one_plan`.

| shape | ours: splits x batch, mode, finalize | public: splits x batch, instance, finalize | stages | threads | regs ours / public |
|---|---|---|---|---|---|
| decode 1 sliding | 32 x 1, mode 13 (one tile, speculated length), last CTA | 32 x 1, MODE 13 (one tile, speculated), last CTA | 3 / 3 | 128 / 128 | 78 / 96 |
| decode 16 sliding | 16 x 16, mode 1, last CTA | 16 x 16, MODE 1, last CTA (no clusters: q_len 1 with 16 requests) | 3 / 3 | 128 / 128 | 109 / 123 |
| decode 32 sliding | 8 x 32, mode 1, last CTA | 8 x 32, MODE 1, last CTA | 3 / 3 | 128 / 128 | 109 / 123 |
| verify 1 sliding | 33 x 1, mode 15 (one tile, speculated, cooperative) | 33 x 1, MODE 15, cooperative | 3 / 3 | 256 / 256 | 70 / 96 |
| verify 16 sliding | 8 x 16, mode 3, cooperative | 8 x 16 (`sms // reqs`), MODE 3, cooperative | 3 / 3 | 256 / 256 | 110 / 119 |
| verify 32 sliding | 4 x 32, mode 3, cooperative | 4 x 32 (`_TUNED`), MODE 3, cooperative | 3 / 3 | 256 / 256 | 110 / 119 |

The launch plans are the same on all six shapes (split counts, one-tile and speculated
instances, cooperative or last-CTA finalize, stages, threads; neither uses clusters on a scored
shape). So there is no plan variant to test, and the 1.1-2.1 us per shape that separates us from
the public times sits inside the kernel body. Register counts are ours at CUDA 12.8.1 and theirs
from `tools/attn_pub_compile.sh`; none spill.

### Per-request q_len 8 tile loop: ours against the public one

| step, per tile i | ours (modes 3 / 15 / 19) | public per-request (3 / 15) | public flat (producer warpgroup) |
|---|---|---|---|
| tile i ready | `cp.async.wait_group 1`, `fence.proxy.async`, `bar.sync` | `cp.async.wait_group 1`, `bar.sync` (no proxy fence) | full mbarrier wait |
| copies of tile i + 2 | right after the barrier, before Q.K | after the Q.K wgmma is issued | producer warpgroup |
| V transpose (raw V to V^T) | after the Q.K wait, softmax and P packing | while Q.K is in flight | while Q.K is in flight |
| V^T sync | `fence.proxy.async` + warpgroup barrier, then P.V | the same, before the Q.K wait | the same |
| raw stage released | next tile's loop-top barrier, after P.V | the same | empty-mbarrier arrive after the Q.K wait and the transpose, before softmax |
| P.V | both k-halves after all exponentials, waited in the tile | each half issued as soon as it is packed, waited at the tile's end | in flight across the loop edge, waited by the next tile's `wgmma.wait_group 1` |

At q_len 1 the question does not arise: the mma.sync loop reads K and V straight from the raw
stage (V in the P.V step itself), so P.V is the stage's last raw reader, and both kernels issue
the next copies right after the loop-top barrier.

### Public flat q_len 8 compile (10-28): the consumer region goes above R79

`tools/attn_pub_compile.sh <public .py>`: CUDA_SRC up to the host `launch`, torch includes
dropped, kernels instantiated inside their anonymous namespace, `nvcc -O3 -std=c++17
-gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr` plus torch's
`-D__CUDA_NO_HALF*` / `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` defines, CUDA 12.8.1 in Docker.

| instance | registers | spills |
|---|---|---|
| flat q_len 8 (`attn_kernel<8, true, 0>`: 256 consumers + 128 producers, 4 stages, 2 CTAs per SM, `setmaxnreg` 32 / 104) | 80 | 0 |
| flat q_len 1 (`<1, true, 0>`) | 128 | 0 |
| per-request q_len 8 MODE 1 / 3 / 7 / 15 | 125 / 119 / 96 / 96 | 0 |
| per-request q_len 1 MODE 1 / 13 | 123 / 96 | 0 |

In the flat q_len 8 SASS the consumer region (after `USETMAXREG.TRY_ALLOC.CTAPOOL 0x68`, up to
`USETMAXREG.DEALLOC`) uses registers up to R100; outside it the highest is R76. So ptxas does
give a `setmaxnreg.inc` region more than the launch-bounds count. That contradicts the 10-28
reading of our mode 51 ("ptxas caps every region at the launch-bounds count"): our mode 51,
with the same thread split, register split and stage count, stays at R78 or below in its
consumer region and spills 304 bytes. The cap comes from how our kernel is built, not from
ptxas. One structural difference: the public kernel runs each role as its own instantiation
of the segment loop, called from the two arms of an if/else around `setmaxnreg`, while ours
returns from the producer arm and continues the consumers inline after `setmaxnreg.inc`. Not
yet tested as the cause.

### q_len 8 schedule variants (10-28, compiled, not yet checked)

Two new schedule bits on the consumer-copy q_len 8 modes (3, 11, 15, 19), chosen in
`_one` by `_ONE_Q8_SCHED` (0 in `submission.py`):

- |128 (QKF): the public per-request order. Q.K is issued, then the copies of tile i + 2,
  then the V transpose runs under the in-flight Q.K, then `fence.proxy.async` + warpgroup
  barrier + Q.K wait. Stage release unchanged.
- |384 (QKF + EARLY): Q.K issued, V transpose under it, Q.K wait, `fence.proxy.async`, one
  CTA barrier, then the raw stage just read is refilled with tile i + 3. With the same three
  stages, three tiles are in flight instead of two (prologue, index ring and waits moved to a
  depth of 3: `PRE` in `attn_one.cu`). The CTA barrier replaces the warpgroup barrier because a
  cache row holds both KV heads, so both warpgroups must be done with the stage.

Fences and barriers: none removed. The loop-top `cp.async.wait_group`, `fence.proxy.async`
and barrier before the Q.K wgmma stay; the V^T writes keep their `fence.proxy.async` before
the barrier that precedes P.V. The early refill is ordered the same way the control's
loop-top refill is (the stage's async-proxy reads by Q.K are complete at `wgmma.wait_group`,
PTX ISA "Asynchronous Warpgroup Level Matrix Instructions: wgmma.wait_group"; generic reads by
the transpose; then a CTA barrier, then the new `cp.async` writes), and in EARLY the
`fence.proxy.async` (PTX ISA "membar / fence", proxy fences) also sits between the Q.K wait and
that barrier. ptxas 12.8.1 segfaults when `fence.proxy.async` is followed directly by
`wgmma.wait_group` in this loop, so EARLY waits first and fences after; the order of the two
does not change what the fence orders (the thread's earlier generic V^T writes).

Local evidence: all 21 instances of `attn_one.cu` build for sm_90a with no spills except mode
51 (unchanged: 304 bytes): 131 / 387: 115 / 111 registers, 139 / 395: 113 / 112, 143 / 399: 71 /
71, 147 / 403: 118 / 117 (control 19: 115). The control instances' SASS differs from the
previous build only by a few integer ALU instructions (mode 19: 3624 vs 3632). No C7514/C7515.
`tools/attn_compile_check.sh` compiles every block of `bench.py`; `tools/attn_host_check.sh`
reports no errors; `tools/attn_harness_dry.py` passes on all five shapes. `bench.py` is
generated with `--sched=384`, so SELFTEST, STRESS and the judge's own tests and timings run the
EARLY schedule on every q_len 8 shape, and BENCH times `_one@x=0` (control), `x=128`,
`x=384` and `_one` (= 384 again, a noise pair) on the six q_len 8 scored shapes.

### Scored run bf6b8ca9e37341fb (bench.py at c901a3e, `--sched=384`): 2.083x, CI 2.051-2.114

Submitted on 10-28 at 23:07 UTC with the day's last scored run, which would have expired at
the 00:00 reset (the user asked that it not go unused); checks were at 20 of 20. Every judge
test passed, SELFTEST 0 failing shapes, every STRESS line PASS with a clean workspace. The board
entry stays 2.162x (b458555d3e3e4e9a). Profiles: `profiles/bench.bf6b8ca9e37341fb.*`.

Judge medians (us), EARLY (|384) on every q_len 8 shape, against b458555d3e3e4e9a:

| shape | b458555d | bf6b8ca9 | instance |
|---|---|---|---|
| decode 1 / 16 / 32 sliding | 6.0 / 10.1 / 12.5 | 6.4 / 9.8 / 12.4 | 13 / 1 / 1 (unchanged) |
| decode 1 / 16 / 32 full | 15.3 / 93.6 / 180.8 | 16.3 / 93.5 / 179.6 | 11 / 49 / 49 (unchanged) |
| verify 1 sliding | 8.0 | 7.5 | 399 |
| verify 16 / 32 sliding | 13.3 / 17.5 | 14.1 / 18.6 | 387 |
| verify 1 full | 18.2 | 19.1 | 395 |
| verify 16 / 32 full | 127.1 / 239.1 | 137.4 / 261.2 | 403 |

BENCH lines (graph replays, us; control `x=0`, QKF `x=128`, EARLY `x=384`, `_one` = EARLY again):

| shape | x=0 | x=128 | x=384 | _one |
|---|---|---|---|---|
| verify 1 sliding | 11.0 | 11.1 | 11.1 | 11.2 |
| verify 1 full | 21.9 | 22.2 | 22.5 | 22.3 |
| verify 16 sliding | 16.8 | 17.3 | 17.6 | 17.6 |
| verify 16 full | 128.1 | 137.2 | 139.5 | 139.6 |
| verify 32 sliding | 21.7 | 22.1 | 22.7 | 22.7 |
| verify 32 full | 237.9 | 257.0 | 260.9 | 261.1 |

Both schedules lose to the control on every q_len 8 shape. Most of the loss is already in QKF
(Q.K first, copies after it, V transpose under the in-flight Q.K): +7% / +8% on the flat shapes,
+2-3% on sliding b16 / b32. EARLY adds another 1.5-2.5%. The repeated EARLY measurement agrees
to 0.1-0.2 us, so these gaps are well above the bench's noise. The judge's verify 1 sliding
7.5 us (from 8.0) is not supported by the bench (11.0 vs 11.1) and sits beside a +0.4 us drift
on decode 1 sliding, which runs the unchanged instance 13. Why it loses is not measured. One
candidate is that the transpose's shared-memory traffic and the copy issue compete with the
Q.K wgmma for the same shared-memory and issue slots, so overlapping them buys nothing. Another
is that EARLY's CTA-wide barrier ties the two warpgroups together each tile. `submission.py`
keeps `_ONE_Q8_SCHED = 0`.

Losers from these runs are in docs/ruled-out.md.

### Mode 51 register cap: the producer's `return` (10-29)

Cause, from the SASS of `attn_one_kernel<8, 51>` (CUDA 12.8.1, `tools/attn_one_sass.sh`):
the producer arm ended in `return`, which in PTX is a branch to the kernel's single `ret`
block at the end of the consumer code. ptxas therefore opens one convergence region around
both roles, `BSSY B0, 0x11990` at 0x15a0 just before the role branch (`ISETP.GT tid, 0xff`,
`@P0 BRA` to the producer at 0x9d30), with both roles meeting at `BSYNC B0` + one `EXIT` at
0x11980. Inside that shared region the consumer code after `USETMAXREG.TRY_ALLOC.CTAPOOL 0x68`
(0x1650-0x9d30) never goes above R77, and ptxas spills 292 / 304 bytes (107 LDL/STL in the
kernel, most of them in the tile loop at 0x5000-0x6000); even the value carried across the
branch is spilled (`STL [R1+0x50]` before the branch, `LDL` after the `TRY_ALLOC`). The public
flat q_len 8 kernel has no such region: its role branch at 0x4240 is a plain `@P0 BRA`, the
consumers end in their own `EXIT` (0x150c0) and the producer, after `USETMAXREG.DEALLOC 0x20`,
in another (0x16da0). Its producer and consumer paths share no block after the split, and its
segment list lives in shared memory, so only one uniform register crosses the branch.

Change: the producer arm ends in `asm volatile("exit;")` instead of `return`. Nothing else;
no numerics, fence or barrier changes. SASS after: producer first (`DEALLOC` at 0x15e0, its own
`EXIT` at 0x91a0), consumers from `TRY_ALLOC` at 0x9240 to their own `EXIT` at 0x11080, no
convergence region across the split.

| build (mode 51) | ptxas regs | spill stores / loads | LDL/STL | consumer region top | instrs |
|---|---|---|---|---|---|
| `return` (main at 1185ff1) | 80 | 292 / 304 B | 107 | R77 | 5040 |
| `exit` | 80 | 0 / 0 | 0 | R95 | 4904 |
| `exit` + `__builtin_unreachable()` | 80 | 292 / 304 B | 107 | R77 | 5040 |

The third row is a control: with the unreachable hint the PTX still has `exit` before
`setmaxnreg.inc`, but ptxas again joins both roles at one `BSYNC B0` + `EXIT` and the cap
returns, so the shared exit block (not the PTX instruction) is what moves the cap. Modes 49 /
115 / 17 / 19 compile as before (128 / 115 / 120 / 113 registers, no spills; mode 49's
consumer region top R114, was R113).

`submission.py` gains `_ONE_Q8_PROD` (False: q_len 8 flat stays on mode 19);
`tools/attn_make_bench_one.py --q8-prod` sets it in `bench.py` only, so SELFTEST, STRESS and the
judge's own timing run mode 51 on verify 16 / 32 full, and BENCH times `_one@m=19`,
`_one@m=51`, `_one` (= 51 again, a noise pair) and `_one@m=115`. Bench generated with
`--no-probe --q8-prod`, control schedule (no |128 / |384 bits).

### Check d90a73871d8e4481 (bench.py at the `exit` change, `--q8-prod`): 2.187x, CI 2.154-2.220

10-29 00:01 UTC, first attention check after the reset. 16/16 judge tests passed, SELFTEST 0
failing shapes (mode 51 on b16 / b32 full L8 scored, b2 ctx100000 and b4 ctx6000 page-64
tests), every STRESS line PASS with a clean workspace (b16 full L8 mode 51: 2,000 replays,
0/131072 differing elements, max_polls 122). Profiles: `profiles/bench.d90a73871d8e4481.*`.

Judge medians (us) against the board entry b458555d3e3e4e9a (mode 19 on q_len 8 flat):

| shape | b458555d | d90a7387 | instance now |
|---|---|---|---|
| decode 1 / 16 / 32 sliding | 6.0 / 10.1 / 12.5 | 6.4 / 10.3 / 12.3 | 13 / 1 / 1 |
| decode 1 / 16 / 32 full | 15.3 / 93.6 / 180.8 | 14.9 / 93.1 / 178.8 | 11 / 49 / 49 |
| verify 1 sliding | 8.0 | 7.5 | 15 |
| verify 16 / 32 sliding | 13.3 / 17.5 | 13.4 / 18.1 | 3 |
| verify 1 full | 18.2 | 18.3 | 11 |
| verify 16 / 32 full | 127.1 / 239.1 | 121.8 / 230.3 | 51 (was 19) |

BENCH (graph replays, us), paired in one process:

| shape | `_one@m=19` | `_one@m=51` | `_one` (= 51) | `_one@m=115` |
|---|---|---|---|---|
| verify 16 full | 128.2 | 125.0 | 125.1 | 150.6 |
| verify 32 full | 238.8 | 232.4 | 231.9 | 277.3 |

Mode 51 without spills beats mode 19 by 2.5% / 2.7% at batch 16 / 32 (the 51 noise pair agrees
to 0.1-0.5 us), where the spilling build timed 389 / 774 us (16fcddf80a714e0d). It is still
behind the public 105 / 186 us. The score gain this predicts from the two shapes alone is about
+0.35% (16 / 372 of the weight each); 2.187x against 2.162x is inside both runs' CIs, so it is
not a gain beyond noise by itself. The planner now prefers 51 (`_ONE_Q8_PROD = True` in
`submission.py`, 19 as the fallback).

### Check 384334430cfe4d37 (submission.py, `_ONE_Q8_PROD = True`): 2.182x, CI 2.149-2.215

16/16 tests passed. Judge medians (us): decode 1 / 16 / 32 sliding 6.4 / 9.6 / 12.2; decode 1 /
16 / 32 full 15.6 / 93.7 / 179.7; verify 1 sliding 7.8; verify 16 / 32 sliding 13.2 / 17.8;
verify 1 full 18.1; verify 16 / 32 full 121.7 / 230.2 (mode 51; profiler 122.9 / 231.7). Profiles:
`profiles/submission.384334430cfe4d37.*`. The verify full times repeat d90a73871d8e4481 to
0.1 us; the score again sits inside the board entry's CI (2.129-2.202), so no scored run was
used: the authorization required a gain beyond noise, and the expected gain (+0.35%) is
smaller than the judge's CI half-width (about 1.5%). The board stays b458555d3e3e4e9a (2.162x).

## Sliding body against the public kernel: the once-path diet (10-29)

Question: the sliding launch plans already match the public entry's (table above), so what does
its kernel body do differently? Control: the board build, mode 51 with the producer's `exit`
(96d568f61c69412f, 2.178x, CI 2.145-2.210).

### Static comparison, same toolkit and flags

`tools/attn_pub_compile.sh` (public instances) and `tools/attn_one_sass.sh` (ours), CUDA 12.8.1,
`nvcc -O3 -gencode=arch=compute_90a,code=sm_90a`. Instructions per instance, with the classes
that differ most (the rest of the mix is close; the public kernel has more FP32 instructions
from its packed-P checks):

| instance | build | instrs | BRA | BSSY | IMAD | IABS | regs |
|---|---|---|---|---|---|---|---|
| decode 1 sliding (q1 mode 13) | ours, board | 2368 | 121 | 53 | 494 | 14 | 78 |
| | public | 2040 | 68 | 15 | 201 | 0 | 96 |
| | ours, diet | 1552 | 53 | 24 | 252 | 2 | 76 |
| decode 16 / 32 sliding (q1 mode 1) | ours, board | 3088 | 167 | 64 | 693 | 36 | 109 |
| | public | 2672 | 113 | 29 | 344 | 0 | 123 |
| | ours, diet | 1944 | 74 | 30 | 301 | 2 | 108 |
| verify 1 sliding (q8 mode 15) | ours, board | 2296 | 116 | 49 | 440 | 13 | 70 |
| | public | 2176 | 63 | 16 | 273 | 0 | 96 |
| | ours, diet | 1968 | 62 | 25 | 277 | 5 | 80 |
| verify 16 / 32 sliding (q8 mode 3) | ours, board | 2840 | 142 | 55 | 570 | 30 | 110 |
| | public | 2680 | 99 | 25 | 361 | 0 | 119 |
| | ours, diet | 2280 | 75 | 29 | 320 | 8 | 110 |

None of these spill. The per-tile math is the same size on both sides (32 HMMA per q1 instance,
80 vs 88 F2FP), so the static difference is in the once-per-launch code. Read from the SASS of
mode 13, this is where it comes from (IABS + I2F + F2I + MUFU.RCP is the software signed
division):

1. Diagnostic hooks in the judged build: 20 `stamp()` sites (a test of `p.dbg`, and a body
   with a globaltimer read and a 64-bit divide by 1000) and 10 `PHASE_EXIT` tests of
   `p.phase_limit`, each a branch on the cold once-path. The public kernel has none.
2. Division by the page size: `pos / psize` and `pos % psize` in `issue_idx`, `issue_kv` and
   the producer, unrolled once per row, are inline software divisions branched around at page
   size 1 (four 70-instruction blocks between the first barrier and the first HMMA of mode
   13). The public kernel shifts and masks by `log2(page_size)`.
3. The query copies and the query pass are runtime-bounded loops (`i = tid; i < ROWS * 8;
   i += NC`), which ptxas compiles to an unrolled loop plus a remainder; the trip count is a
   compile-time 1 (q1) or 4 (q8). The public kernel runs them straight-line (`tfor`).
4. Prologue row copies take a three-way branch per row (cached, own new row, past the
   range); the public one-tile path copies with predicated straight-line `cp.async`.
5. Zero-byte page-index copies for tiles past the CTA's range (tiles 2 and 3 in the one-tile
   builds); the public kernel skips them.

### Change: ONE_DIAG / ONE_DIET build switches (`attn_one.cu`)

- `ONE_DIAG` (default 0): the stamps and phase exits compile only into a diagnostic build;
  the host refuses `phase_limit > 0` without it.
- `ONE_DIET` (default 1): `Params::lp = log2(page_size)` with shift and mask in place of
  division (the host checks a power of two; `_one_ok` routes any other page size to the
  fallback path), straight-line query copies and query pass, predicated prologue row copies
  (`cp_async16_if`), no index copies past `ntiles`. The q_len 8 producer builds (51, 115)
  keep the plain path (see below).

No fence, barrier or ordering change. Per instance, the SASS counts of `BAR.SYNC`,
`FENCE.VIEW.ASYNC.S` (fence.proxy.async), `MEMBAR`, `SYNCS` (mbarrier), `ATOMG`, `REDG` and
`LDGDEPBAR` are identical between the board build and the diet build; only `DEPBAR`
(`cp.async.wait_group`) drops, by the `wait_group 0` inside the removed phase exits. A
predicated-off `cp.async` is not issued, so the commit-group counts the waits rely on are
unchanged. With the old defaults (`-DONE_DIET=0 -DONE_DIAG=1`) all 13 instances are
SASS-identical to the board build; with the new defaults every instance is SASS-identical to
an arm timed in d57d1fe064164611 (51 and 115 to `nodiag`, the rest to `diet`).

### Bench d57d1fe064164611 (check, `tools/attn_make_bench_ab.py`, paired in one process)

Arms: `ctrl` (the board build), `nodiag` (`-DONE_DIAG=0`), `diet` (`-DONE_DIAG=0 -DONE_DIET=1`),
`pub` (the public entry's own file, loaded from the session scratchpad; not in the repo). Same
inputs per shape; ABCOLD = median of 120 graph replays after a 256 MB flush, arms interleaved
in rotating order; ABWARM = a graph of 16 back-to-back calls, median of 25, divided by 16.
ABWARM tracks the judge's medians (ctrl verify 1 sliding 7.53 us both here and in
96d568f61c69412f). ABCHECK: 0 failing (the control against the FP32 reference, the arms against
the control: at most 7.2e-5 relative, one-ulp merge-order differences, workspace clean). The
judge's tests passed 16/16. Profiles: `profiles/bench_ab.d57d1fe064164611.*`.

| | shape | ctrl | nodiag | diet | pub | diet - ctrl | diet - pub |
|---|---|---|---|---|---|---|---|
| warm | decode 1 sliding | 6.28 | 5.75 | 5.50 | 4.42 | -0.78 | +1.08 |
| warm | decode 16 sliding | 10.40 | 9.62 | 9.34 | 7.37 | -1.06 | +1.97 |
| warm | decode 32 sliding | 12.33 | 11.45 | 11.09 | 9.26 | -1.24 | +1.83 |
| warm | verify 1 sliding | 7.53 | 7.12 | 6.93 | 6.68 | -0.60 | +0.25 |
| warm | verify 16 sliding | 13.04 | 12.35 | 11.68 | 11.17 | -1.36 | +0.51 |
| warm | verify 32 sliding | 17.64 | 16.92 | 16.05 | 14.64 | -1.59 | +1.41 |
| warm | decode 1 full | 15.95 | 15.13 | 14.58 | 9.96 | -1.37 | +4.62 |
| warm | decode 16 full | 93.54 | 93.14 | 88.79 | 86.52 | -4.75 | +2.27 |
| warm | decode 32 full | 184.90 | 184.40 | 173.12 | 166.66 | -11.78 | +6.46 |
| warm | verify 1 full | 17.99 | 17.42 | 17.34 | 13.73 | -0.65 | +3.61 |
| warm | verify 16 full | 124.78 | 122.36 | 123.70 | 100.07 | -1.08 | +23.63 |
| warm | verify 32 full | 244.98 | 240.62 | 242.52 | 189.10 | -2.46 | +53.42 |
| cold | decode 1 sliding | 10.56 | 10.14 | 9.92 | 9.12 | -0.64 | +0.80 |
| cold | decode 16 sliding | 15.30 | 14.98 | 14.37 | 13.18 | -0.93 | +1.19 |
| cold | decode 32 sliding | 17.38 | 16.96 | 16.70 | 15.78 | -0.68 | +0.92 |
| cold | verify 1 sliding | 11.90 | 11.55 | 11.23 | 11.04 | -0.67 | +0.19 |
| cold | verify 16 sliding | 16.90 | 16.45 | 16.06 | 15.58 | -0.84 | +0.48 |
| cold | verify 32 sliding | 21.82 | 21.44 | 20.86 | 19.42 | -0.96 | +1.44 |

On the sliding shapes the hooks alone (ctrl to nodiag) are 0.4-0.9 us and the rest of the
diet (nodiag to diet) another 0.2-0.9 us. Mode 51 (verify 16 / 32 full) is the one place the
diet ran slower than nodiag, by 0.9-1.9 us in all four samples, so the q_len 8 producer builds
keep the plain path (`DIET` in the kernel). Applying the warm deltas to 96d568f61c69412f's
per-shape times predicted 2.315x for the diet (nodiag alone 2.252x).

### Judged: check 03a21fbc99cf445f 2.299x (CI 2.264-2.333), repeat d0adceb5f83b4e1e 2.291x (CI 2.257-2.325)

`submission.py` at fd068c4 (the repeat differs by one comment line, 6471dfa). 16/16 tests
passed in both. Profiles: `profiles/submission.03a21fbc99cf445f.*`,
`profiles/submission.d0adceb5f83b4e1e.*`. Judge medians (us):

| shape | 96d568f (control) | 03a21fbc | d0adceb5 | change (mean of the two) |
|---|---|---|---|---|
| decode 1 sliding | 6.64 | 5.78 | 5.91 | -0.80 (-12.0%) |
| decode 16 sliding | 9.72 | 9.24 | 9.08 | -0.56 (-5.8%) |
| decode 32 sliding | 12.20 | 11.08 | 11.52 | -0.90 (-7.4%) |
| verify 1 sliding | 7.53 | 6.83 | 7.03 | -0.60 (-8.0%) |
| verify 16 sliding | 13.45 | 12.07 | 11.85 | -1.49 (-11.1%) |
| verify 32 sliding | 17.56 | 15.94 | 15.96 | -1.61 (-9.2%) |
| decode 1 full | 16.38 | 14.72 | 15.34 | -1.35 (-8.2%) |
| decode 16 full | 92.67 | 89.95 | 89.41 | -2.99 (-3.2%) |
| decode 32 full | 178.79 | 170.21 | 171.55 | -7.91 (-4.4%) |
| verify 1 full | 18.42 | 17.47 | 17.61 | -0.88 (-4.8%) |
| verify 16 full | 120.83 | 120.18 | 119.93 | -0.78 (-0.6%) |
| verify 32 full | 229.75 | 226.68 | 226.57 | -3.13 (-1.4%) |

Both runs sit above the control's CI and above the control family's other checks (2.182x,
2.187x), and every shape is faster in both. Win: +5.2% to +5.6% on the attention score. No
scored run was used.

### Bench 0c4b4923d4b44b79: the diet again, and the batch-1 full finalize (ONE_FIN8)

Arms: `ctrl` (the new default: diet), `board` (`-DONE_DIET=0 -DONE_DIAG=1`, all 13 instances
SASS-identical to 96d568f's build), `fin8` (`-DONE_FIN8=1`), `pub`. ABCHECK 0 failing, tests
16/16. Profiles: `profiles/bench_fin8.0c4b4923d4b44b79.*`. Warm (us):

| shape | board | ctrl (diet) | fin8 | pub |
|---|---|---|---|---|
| decode 1 sliding | 5.91 | 5.08 | 5.08 | 4.66 |
| decode 16 sliding | 9.92 | 8.89 | 8.85 | 7.59 |
| decode 32 sliding | 12.48 | 11.34 | 11.29 | 9.30 |
| verify 1 sliding | 7.51 | 6.94 | 6.97 | 6.60 |
| verify 16 sliding | 13.30 | 11.96 | 11.97 | 11.36 |
| verify 32 sliding | 17.46 | 15.88 | 15.90 | 14.65 |
| decode 1 full | 15.97 | 14.65 | 13.11 | 10.00 |
| verify 1 full | 18.22 | 17.55 | 15.55 | 13.93 |

The diet's sliding deltas (board minus ctrl: 0.83 / 1.03 / 1.14 / 0.57 / 1.34 / 1.58 us)
repeat d57d1fe0's to within 0.1 us. `ONE_FIN8`: the finalize of
the batch-1 full plan (cooperative, 8 workspace copies) read each copy, shuffled its row sum
and zeroed it before the next copy's loads, so the 8 copies cost 8 dependent L2 round trips
(SASS of mode 11: two `LDG.128`, `SHFL.IDX`, two `STG.128` per copy, in sequence). The public
kernel loads all 8 first (its `sum_copies` notes the same problem). Issuing all loads first
cut decode 1 full by 1.5 us and verify 1 full by 2.0 us (cold: 1.6 / 2.2) and left every
other shape unchanged; it compiles only into mode 11 (`FIN8 = ONE_FIN8 && COOP && SPEC &&
!ONE`), and every barrier, fence and atomic count is unchanged. Predicted +1.7% on the score.
Not judged: it is a second change outside this sliding brief, so `ONE_FIN8` stays 0 until a
judged check and a repeat.

### What still separates us from the public kernel on sliding (diet vs pub, warm, two samples)

decode 1 0.4-1.1 us, decode 16 1.3-2.0, decode 32 1.8-2.0, verify 1 0.25-0.35, verify 16
0.5-0.6, verify 32 1.2-1.4. What remains sits in the builds with several tiles per CTA (mode 1
at q_len 1: 2 and 4 tiles; verify 32: 8 tiles), so it grows with tiles, which points at the
tile loop or the in-loop copy issue rather than the once-path. Hypothesis, not measured: the
in-loop `issue_kv` (`patch == true`) still takes a three-way branch and 64-bit offsets per row,
where the public in-loop issue has a branch-free shift-add path for fully cached tiles. The
public Q.K-first order at q_len 8 is already ruled out for our kernel (bf6b8ca9e37341fb). The
full-context q_len 8 flat gap (23 / 53 us at batch 16 / 32) is unchanged by this work.

### Scored: da872fcdb70445c3 2.281x (2026-10-29)

Submit of the diet file exactly as checked in d0adceb5f83b4e1e (submission.py at 6471dfa, before the ONE_FIN8 switch was added), under the standing replace-when-demonstrably-better authorization. Passed. Board entry before: 96d568f61c69412f (2.178x).
## Batch-1 full finalize (ONE_FIN8) judged, and a branch-free in-loop copy (ONE_KVFAST) (10-29)

Control: the diet build, checks 03a21fbc99cf445f (2.299x) and d0adceb5f83b4e1e (2.291x).

### Ordering argument for ONE_FIN8

`ONE_FIN8` compiles only into mode 11 (`FIN8 = ONE_FIN8 && COOP && SPEC && !ONE`), the
cooperative batch-1 full plan with eight workspace copies. In that mode no single last CTA
finalizes: every active split waits at the cooperative barrier, then finalizes its own slice
of rows. The last-CTA ticket path (`ticket()`, modes 1 and 13) calls `finalize_rows` without
`NC8` and does not change. Sections are from the PTX ISA 9.4 documentation
(docs.nvidia.com/cuda/parallel-thread-execution, read 10-29): 8.7.1 Conflict and Data-races,
8.8 Release and Acquire Patterns, 8.9.2 Observation Order, 8.9.4 Memory synchronization, 8.9.5
Causality Order, 8.9.6 Coherence Order, 8.10.3 Atomicity, 8.10.6 Causality, and the `atom` /
`red` instruction pages (with no `.sem` or `.scope`, `.relaxed` and `.gpu` are assumed).

The chain, the same in both orderings up to step 5:

1. Writers. Each active split adds its partial `o * 2^m` and `l * 2^m` into copy
   `split % 8` of the request's rows with `red.global.add.v4.f32` and `red.global.add.f32`
   (relaxed, gpu scope). Different splits add into the same addresses; the reds are strong
   operations at gpu scope, so they are morally strong with each other, and Atomicity (8.10.3)
   keeps every addition.
2. Publication. `__syncthreads()` (`bar.sync 0`), then thread 0's `atom.acq_rel.gpu.add` on the
   parity counter. A `bar.sync` synchronizes with every other `bar.sync` on that barrier
   (8.9.4), so every thread's reds precede thread 0's atom in base causality order (8.9.5:
   program order, then synchronizes-with, then program order). The atom is a release operation,
   a release pattern by itself (8.8), and base causality is transitive (8.9.5 footnote 1,
   cumulativity), so the release covers the other threads' reds too. A split whose range is
   empty returns before step 1 (`lo >= end` after the length check), so no CTA adds after the
   count is complete.
3. Observation. Thread 0 of every split spins on `ld.acquire.gpu` until the counter reaches
   the request's key count. That value ends a chain of atomic adds, and observation order runs
   through atomic read-modify-writes (8.9.2), so each split's atom precedes the poller's acquire
   in observation order and its release synchronizes with that acquire (8.9.4: a write in the
   release pattern precedes a read in the acquire pattern in observation order, both morally
   strong, both gpu scope).
4. Hand-off inside the CTA. Thread 0 then runs `bar.sync 1, NC` (`named_sync`), which
   synchronizes with the other consumer threads' `bar.sync 1` (8.9.4). Every red of every split
   now precedes, in causality order, every finalizer thread's later operations. Reds and loads
   use the generic proxy on the same addresses, so the order is proxy-preserved (8.9.5).
5. Loads. Each finalizer thread reads its eight floats of a row (and lane `t8 == 0` the row
   sum) with `ld.global.cg` (weak). A weak load and a red to the same address that are ordered
   by causality are not a data race (8.7.1), and by Causality (8.10.6) the load reads the last
   red or a later write in coherence order. Nothing else writes these addresses in this launch:
   rows are split between CTAs by slice, the eight lanes of a row own disjoint 32-byte pieces,
   and only lane 0 touches `ws_l`. Each address is read and zeroed by exactly one thread.
6. Zeroing. That thread stores zeros to exactly the addresses it read, with `st.global.cg`
   (weak). Default ordering: load copy c, shuffle, store copy c, then copy c + 1. FIN8: load
   all eight copies, shuffle once, store all eight. In both, each store follows the load of the
   same address in program order, hence in causality order, and Causality (8.10.6: a read that
   precedes an overlapping write in causality order cannot read from it) forbids the load from
   seeing its own zero. Each store also follows every red to that address in causality order
   (steps 2-4), so it comes after them in coherence order (8.9.6) and the row ends the launch at
   zero.
7. Reuse. The next launch is the next task in the stream. "Completion of a task enqueued in a
   CUDA stream synchronizes with the start of the following task in the same stream" (8.9.4,
   API synchronization), so its reds follow this launch's zero stores and loads in causality
   order: they add onto zero, and this launch's loads cannot see them. The counter reset (the
   last arriver clears the other parity and bumps the epoch) is outside the changed code.

What FIN8 changes: in one thread, the loads of copies 1..7 now come before the stores of copies
0..6. Those touch different addresses that no other thread accesses after step 4, so no rule
relates them, and steps 1-5 and 7 do not change. The obligation that matters, each load before
the store to the same address, still holds by program order. FIN8 needs no new fence, and
every existing fence and barrier stays. The SASS agrees: building all 13 instances with
`-DONE_FIN8=0` and `=1` (`tools/attn_one_sass.sh`, CUDA 12.8.1), only mode 11 (q_len 1 and 8)
changes, and its `BAR.SYNC`, `FENCE.VIEW.ASYNC`, `MEMBAR`, `ATOMG`, `REDG` and `CCTL` counts are
identical; only `LDG` and `STG` grow (the unrolled eight-copy block beside the rolled loop that
`nc != 8` still uses).

### Judged: check b0b146dfbb4841b2 2.347x (CI 2.312-2.382), repeat fe08878a7e0c46f4 2.316x (CI 2.281-2.351)

`submission.py` at aedad22 (`ONE_FIN8` default 1); the repeat, 8860211, adds one comment line.
16/16 tests in both. Profiles: `profiles/submission.b0b146dfbb4841b2.*`,
`profiles/submission.fe08878a7e0c46f4.*`. Judge medians (us), with the kvfast check described
below:

| shape | diet 03a21fbc | diet d0adceb5 | fin8 b0b146df | fin8 fe08878a | fin8 - diet (means) | kvfast 422f81a1 | kvfast - fin8 mean |
|---|---|---|---|---|---|---|---|
| decode 1 sliding | 5.78 | 5.91 | 5.92 | 6.01 | +0.12 (+2.1%) | 5.69 | -0.27 |
| decode 1 full | 14.72 | 15.34 | 13.41 | 13.78 | -1.44 (-9.5%) | 13.21 | -0.38 |
| decode 16 sliding | 9.24 | 9.08 | 9.27 | 9.18 | +0.06 (+0.7%) | 10.09 | +0.87 |
| decode 16 full | 89.95 | 89.41 | 89.30 | 89.53 | -0.27 (-0.3%) | 89.80 | +0.39 |
| decode 32 sliding | 11.08 | 11.52 | 11.19 | 11.50 | +0.04 (+0.4%) | 11.45 | +0.11 |
| decode 32 full | 170.21 | 171.55 | 171.44 | 170.54 | +0.11 (+0.1%) | 170.46 | -0.53 |
| verify 1 sliding | 6.83 | 7.03 | 6.65 | 7.24 | +0.02 (+0.2%) | 6.87 | -0.08 |
| verify 1 full | 17.47 | 17.61 | 15.71 | 16.00 | -1.68 (-9.6%) | 15.59 | -0.27 |
| verify 16 sliding | 12.07 | 11.85 | 11.84 | 12.15 | +0.04 (+0.3%) | 12.22 | +0.22 |
| verify 16 full | 120.18 | 119.93 | 120.49 | 119.99 | +0.18 (+0.2%) | 120.85 | +0.61 |
| verify 32 sliding | 15.94 | 15.96 | 16.40 | 16.26 | +0.38 (+2.4%) | 15.67 | -0.66 |
| verify 32 full | 226.68 | 226.57 | 226.90 | 226.68 | +0.17 (+0.1%) | 227.47 | +0.68 |

Both FIN8 runs beat both diet runs on decode 1 full and verify 1 full, and both scores are
above both diet scores (+0.7% to +2.4%). The other ten shapes run SASS-identical code; their
spread here (up to 0.4 us) is run-to-run noise. Promoted: `ONE_FIN8` is the default. Judged
gain on the score: +1.6% (mean 2.331x vs 2.295x), close to the +1.7% predicted from bench
0c4b4923d4b44b79.

### ONE_KVFAST: the in-loop copy without the per-row three-way branch

The gap analysis above named the in-loop `issue_kv` (`patch == true`) as a suspect for the
remaining sliding gap. `ONE_KVFAST` gives a tile whose rows all lie below `seq` (every tile but
the one that reaches the block's own positions) an unrolled path with unconditional `cp.async`;
the tile that holds the new rows keeps the three-way path. It applies to modes 1, 3 and 11 (the
per-request builds with several tiles); 13 and 15 (one tile) and the producer builds do not use
it. No fence or barrier change: the copies are the same `cp.async` into the same commit groups,
and the SASS sync counts are identical with it on and off.

Bench 82e52f7b23ec49da (`tools/attn_make_bench_ab.py`, arm `kvfast` against the FIN8 default,
paired in one process; ABCHECK 39 lines, 0 failing; tests 16/16). Profiles:
`profiles/bench_kvfast.82e52f7b23ec49da.*`. Warm / cold (us):

| shape (mode) | ctrl warm | kvfast warm | ctrl cold | kvfast cold |
|---|---|---|---|---|
| decode 1 full (11) | 13.07 | 12.84 | 18.82 | 18.78 |
| decode 16 sliding (1) | 9.48 | 9.44 | 14.30 | 14.18 |
| decode 32 sliding (1) | 11.05 | 10.78 | 16.74 | 16.67 |
| verify 1 full (11) | 15.63 | 15.44 | 20.03 | 19.90 |
| verify 16 sliding (3) | 11.60 | 11.49 | 16.03 | 15.84 |
| verify 32 sliding (3) | 15.99 | 15.63 | 21.18 | 20.74 |

Shapes whose code does not change moved by -0.30 to +0.25 us between the arms, so only the
32-batch sliding deltas (-0.27 / -0.36 warm) stand clear of that, and decode 16 sliding
(-0.04) does not move. The branch is not the 1.3-2.0 us decode 16 gap to the public kernel:
that hypothesis is rejected for this build.

Judged: check 422f81a11f384deb (a18937b, `ONE_KVFAST` default 1) 2.322x (CI 2.287-2.370),
16/16 tests, profiles `profiles/submission.422f81a11f384deb.*`, per shape in the table above.
It is below b0b146dfbb4841b2 on the score and slower on decode 32 sliding (11.45 vs 11.19);
decode 16 sliding read 10.09 us (all four earlier runs 9.08-9.27); verify 32 sliding was the
fastest yet (15.67). The judge's spread on shapes whose code is identical in this check (up to
0.68 us) is larger than the effect the bench measured, so one check cannot confirm the bench,
and a repeat could not have met the promotion rule (both checks ahead of the FIN8 pair on the
changed shapes, score not lower), so the fifth check was not spent. `ONE_KVFAST` stays in the
source at 0. The branch head (`ONE_FIN8` 1, `ONE_KVFAST` 0) compiles to SASS identical to
aedad22's on all 13 instances (only the anonymous-namespace hash in the mangled names differs).

Submit candidate: `muse_attention_decode/submission.py` at aedad22 (judged b0b146dfbb4841b2;
repeat fe08878a7e0c46f4 at 8860211). Attention checks used this round: 4 of 5.

### Scored: 9104344cb757454e 2.334x (2026-10-29)

Submit of submission.py at aedad22 (the ONE_FIN8 build judged in b0b146dfbb4841b2), under the standing replace-when-demonstrably-better authorization. Passed. Board entry before: da872fcdb70445c3 (2.281x).

## Gap map E2, FP8 P.V on wgmma in mode 51: already in the board build (10-29, static, no check)

Question from [gap-map-2026-10-29.md](gap-map-2026-10-29.md) E2: does the public flat q_len 8
kernel win its 23.6 / 53.4 us at verify 16 / 32 full (d57d1fe064164611) by running P.V in e4m3
on wgmma with ones rows in V^T, where ours rounds P to f16? Answer: no. Both kernels already
run the same P.V path, so E2 has nothing to build.

Public source (the Apache-2.0 `muse_attention_decode.py` timed as `pub` in d57d1fe0, sha1
0e9d4c78), L == 8 path: P is packed from the FP32 exponentials by `e4m3x4f`
(`cvt.rn.satfinite.e4m3x2.f32`) into wgmma A registers; P.V is
`wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3` (`wgmma_m64n72k32_e4m3`, issued per
32-key half, and in the PCHK build both halves together) against V^T in shared memory, K-major,
64B swizzle, keys permuted to P's register order, written per tile by `ldmatrix.x4.trans` +
`prmt` + `stmatrix.x4`; rows 64..71 of each V^T buffer are e4m3 ones (0x38), so accumulator
columns 64..71 (`lacc`) are the row sums of the same rounded P. The comment above that block
("PV with f16 mma.sync from a per-tile f16 copy of V") is stale: the only `mma_f16` calls are
on the q_len 1 path.

Ours (`attn_one.cu`, since 464ef7e, 10-28 03:32): the same design (`wgmma_pv`,
m64n72k32 e4m3, ones rows, `ldmatrix.trans` + `prmt` + `stmatrix`). The f16 P.V that the
gap map's mechanism describes (gap-analysis-2026-10-28.md 3.1, written at 02:26 on 10-28) is
the older `attn_ws.cu` (`wgmma ... m64n64k16.f32.f16.f16`), which no board shape runs.

SASS, CUDA 12.8.1, sm_90a (`tools/attn_pub_compile.sh`, `tools/attn_one_sass.sh ... 8,51`):

| | public flat q_len 8 (`attn_kernel<8, true, 0>`) | ours, mode 51 |
|---|---|---|
| registers / spills | 80 / 0, consumer region to R100 | 80 / 0, consumer region to R91 |
| instructions | 6432 | 4688 |
| Q.K | 6 x `QGMMA.64x64x32.F32.E4M3.E4M3` | 2 x the same |
| P.V | 6 x `QGMMA.64x72x32.F32.E4M3.E4M3` (P in registers) | 2 x the same |
| P to e4m3 (`F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C`) | 104 | 24 |
| `MUFU.EX2` | 240 | 44 |
| `LDSM` / `STSM` | 6 / 0 (flat build writes V^T with 4-byte `st.shared`: `STSM_ON = !FLAT`, its comment says stmatrix cost the flat kernel 0-1.5%) | 2 / 2 |
| `FENCE.VIEW.ASYNC.S` / `BAR.SYNC` / `SYNCS` | 6 / 21 / 19 | 4 / 11 / 33 |

Precision and instruction match. The public kernel carries three copies of the tile body
(the PCHK pack-both-halves path, the per-half path and the rare offset-move repack), which is
where its larger counts come from; whether that code layout matters to time is not measured.
One real difference in the V^T path: the public flat build stores V^T with per-thread 4-byte
stores and keeps `stmatrix` for its per-request kernels only; ours uses `stmatrix.x4` in mode
51. By the public author's own comment that is worth 0-1.5% on their flat kernel, far below
the gap; a lead, not tested.

Rules check, for the record: TASK.md allows FP8 queries "as FA3 does, and no coarser" and BF16
wherever SGLang's kernel keeps BF16; the reference's SGLang path is FA3 with `qkv_dtype` FP8,
whose FP8 forward also feeds P.V with e4m3 P. Our board build (judged 16/16 in
b0b146dfbb4841b2 and scored 9104344cb757454e) and the public entry both already do this.

The 23.6 / 53.4 us flat q_len 8 gap stays open and is not in the P.V operand type. No check
used.

## The public flat q_len 8 tile loop, and its order in mode 51 (10-29)

Question: what does the public flat kernel's tile loop do that mode 51 does not? The SASS lead
was three copies of the tile body (6432 vs 4688 instructions, 240 vs 44 `MUFU.EX2`).
Hypotheses going in: software pipelining across tiles (Q.K of tile t+1 issued during the
softmax of tile t, FA3-style), or plain unrolling.

### What the public loop does (source read, SASS measured, timing measured)

Source (the Apache-2.0 public file timed as `pub` in d57d1fe064164611, sha1 0e9d4c78, the
`WS` / `FLAT` branch of `segment`, about lines 1776-2097):

- `#pragma unroll (UNR8)` with `UNR8 = WS ? 2 : 1`: the producer loop is unrolled twice.
- No pipelining across tiles. Every tile ends in `wgmma_wait<0>()` ("done before the next
  tile (no wgmma in flight across the loop edge)"), so the `wgmma_wait<1>()` after the next
  Q.K finds nothing pending. The P.V of tile t has finished before the Q.K of tile t+1 is
  issued.
- Per tile, in order: wait on the full barrier, patch, Q.K issued and committed; **while it
  runs**, this warp's V tile is transposed into V^T (ldmatrix.trans, byte permutes, 4-byte
  `st.shared`; `STSM_ON = !FLAT`); `wgmma_wait<0>`; **the stage is released** (`mbar_arrive`
  on the empty barrier) before the softmax; mask; on every tile after a segment's first, **no
  row max**: P is packed straight from the scores with the current offset, and an overflow
  test on the packed bytes (`+0x07070707`, bit 7 set means P > 256) moves the offset and
  repacks only on a hit; `fence.proxy.async` + named barrier; P.V issued; `wgmma_wait<0>`.

SASS (CUDA 12.8.1, `tools/attn_pub_compile.sh`, flat q_len 8 instance): the public file as is
has 6432 instructions, 6 + 6 QGMMA, 240 `MUFU.EX2`; the same file with `UNR8 = 1` has 5376,
2 + 2 QGMMA, 108 `MUFU.EX2`, no spills either way. So the three bodies are the two unrolled
copies plus the remainder. They are not a pipeline prologue, steady state and epilogue, and
not per-case specializations. The 10-29 E2 entry's guess (PCHK path, per-half path and repack
path) was wrong: the per-half loop runs `kk < (PK2 ? 0 : 2)`, zero times in this build. The
108 `MUFU.EX2` per body are the pack (32), the repack on a hit (32), and the offset moves.

Timing (bench e1d8a55eaa2f455c, below): the public kernel with the unroll removed (`pubu1`)
is 1.3 / 2.9 us slower warm at verify 16 / 32 full than with it, against a 23 / 56 us lead
over mode 51 in the same process. The unroll is a small part of the public lead.

### The change: `ONE_QKOV`, `ONE_EREL`, `ONE_PCHK`, `ONE_UNR2` (modes 51 and 115 only)

`attn_one.cu`, a separate branch of the q_len 8 tile loop, taken only when a switch is on and
the build is a q_len 8 producer build (`PROD && L == 8`):

- `ONE_QKOV`: V^T is written (same ldmatrix.trans + stmatrix code) between the Q.K
  `wgmma.commit_group` and its `wgmma.wait_group 0`, instead of after the softmax.
- `ONE_EREL`: the empty-barrier arrive moves from after the P.V wait to right after the Q.K
  wait (the transpose is done by then).
- `ONE_PCHK`: the public packed-P offset check after a segment's first tile; on a hit the
  same per-row move as before (`move = tmax > off + 8`), then a repack.
- `ONE_UNR2`: `#pragma unroll (2)` on the tile loop.

With all four off, all 13 instances are SASS-identical to the board build
(`tools/attn_one_sass.sh`, normalized diff). With all four on, only 51 and 115 change. Mode 51
with all four: 80 registers, no spills, consumer region up to R101 (setmaxnreg grants 104),
6232 instructions; the three partial combinations benched below also compile with no spills.
The SASS of the QKOV build shows both LDSM/STSM pairs between the two Q.K QGMMAs and the
`WARPGROUP.DEPBAR`, and with EREL the `SYNCS.ARRIVE` on the empty barrier directly after the
DEPBAR; in the board build both follow the DEPBAR and the arrive follows the P.V wait.

### Ordering argument (no fence or barrier removed, none added)

Sections are from the PTX ISA 9.4 documentation, located by number and title on 10-29: 8.6
Proxies, 8.8 Release and Acquire Patterns, 8.9.4 Memory synchronization, 9.7.15.16.16
`mbarrier.arrive`, 9.7.17.4 Async Proxy (wgmma), 9.7.17.7.1-3 `wgmma.fence`,
`wgmma.commit_group`, `wgmma.wait_group`. The fetched page cut off before the bodies of 9.7.17,
so the wgmma rules below are the ones the board build already relies on, not re-read this
session.

1. QKOV, the stores under the in-flight Q.K. The Q.K wgmma reads the Q buffer and the K half of
   stage `st` (async proxy, 9.7.17.4) and writes the accumulator registers `s`. The moved
   stmatrix writes go to this tile's V^T buffer, `VT + (2 (it & 1) + hk) * VT_STRIDE`, which
   overlaps neither Q nor the stage, and nothing reads or writes `s` until after
   `wgmma.wait_group 0` (9.7.17.7.3 forbids touching the accumulators of an unfinished
   wgmma; `reg_fence(s)` still follows the wait). The `wgmma.fence` before the Q.K (9.7.17.7.1)
   is unchanged. The V^T buffer of tile `it` was last read by the P.V of tile `it - 2`; every
   warp of the warpgroup finished that P.V's `wait_group 0` before the named barrier of tile
   `it - 1`, which every warp passes before writing at tile `it`, as in the board build. The
   writer-side `fence.proxy.async` (8.6, generic writes read by the async proxy) and the named
   barrier still sit between the stmatrix writes and the P.V wgmma that reads them; only the
   distance between the writes and the fence grows.
2. EREL, the earlier release. The stage holds K (read by the Q.K wgmma) and V (read by this
   thread's ldmatrix). After `wgmma.wait_group 0` the Q.K wgmma is complete, so its async-proxy
   reads of the stage are done (9.7.17.7.3); the ldmatrix loads precede the arrive in program
   order. `mbarrier.arrive` has release semantics at CTA scope by default (9.7.15.16.16), so
   both are ordered before the producer's acquiring wait on the empty barrier (8.8, 8.9.4) and
   so before its next cp.async into the stage. The board build relies on the same chain for the
   same reads; it only arrived later, after the P.V, which reads V^T and never the stage. The
   patch path (generic writes into the stage before the Q.K, with `fence.proxy.async` and the
   CTA barrier) is unchanged and also precedes the arrive.
3. PCHK and UNR2 change arithmetic and code layout, not memory ordering.

SASS sync counts of mode 51 per tile body are unchanged by QKOV, EREL and PCHK (4 / 11 / 33 /
2 for `FENCE.VIEW.ASYNC.S` / `BAR.SYNC` / `SYNCS` / `WARPGROUP.DEPBAR` in all three); with UNR2
they grow with the extra bodies (10 / 16 / 41 / 6).

Numerics: QKOV and EREL do not change the arithmetic (5e-6 relative to the board build, merge
order). PCHK tests the overflow on the rounded e4m3 byte instead of the FP32 max: an unrounded P
up to about 272 rounds to 256 and does not move the offset, where the board build moves above
256 exactly. P stays far below 448, and the row sums come from the same rounded P through the
ones rows. Against the board build: 1.1e-3 / 1.3e-3 relative at verify 16 / 32 full, 7.3e-4 on
the page-64 test shape; the public kernel differs from the board build by 6.0e-3 / 6.2e-3 /
1.6e-2 on the same shapes, and the board build is 2.1e-2 from the FP32 reference at verify 32.

### Bench e1d8a55eaa2f455c (check, `tools/attn_make_bench_ab.py --shapes full`, paired)

Arms: `ctrl` (board build), `ov` (QKOV), `ovr` (+EREL), `ovrp` (+PCHK), `ovrpu` (+UNR2), `pub`,
`pubu1` (public with `UNR8 = 1`). ABCHECK 59 lines, 0 failing; tests 16/16. Decode full runs
mode 49, whose code no arm changes (ours within 0.2 us of ctrl there: the noise floor of this
run). Profiles: `profiles/bench_tile.e1d8a55eaa2f455c.*`. Microseconds:

| | shape | ctrl | ov | ovr | ovrp | ovrpu | pub | pubu1 |
|---|---|---|---|---|---|---|---|---|
| warm | decode 16 full | 89.47 | 89.53 | 89.52 | 89.51 | 89.49 | 86.70 | 86.94 |
| warm | decode 32 full | 173.69 | 173.78 | 173.82 | 173.63 | 173.75 | 167.04 | 166.69 |
| warm | verify 16 full | 123.04 | 125.92 | 122.87 | 123.68 | 123.27 | 99.84 | 101.09 |
| warm | verify 32 full | 250.69 | 253.14 | 245.30 | 244.16 | 241.54 | 195.08 | 198.00 |
| cold | verify 16 full | 124.29 | 126.46 | 124.51 | 125.22 | 125.79 | 110.75 | 110.98 |
| cold | verify 32 full | 227.39 | 231.49 | 226.88 | 221.98 | 222.24 | 192.67 | 193.09 |

At verify 32 full warm: QKOV alone +2.5 us (slower), EREL -5.4 cumulative, PCHK -6.5, UNR2
-9.2. Verify 16 full does not move (-0.2 to +2.9). The all-four arm cleared the 5 us stop
line at verify 32 warm, so it went to the judge.

### Judged: check cb85d590c301410e 2.311x (CI 2.277-2.346), repeat 6f75235c0d364a7a 2.361x (CI 2.326-2.406)

`submission.py` at a6868f9 (all four switches on); the repeat, bc59f08, adds one comment
line. 16/16 tests in both. Profiles: `profiles/submission.cb85d590c301410e.*`,
`profiles/submission.6f75235c0d364a7a.*`. Judge medians (us) of the changed shapes, against
the five earlier checks of the same mode 51 code (03a21fbc, d0adceb5, b0b146df, fe08878a,
422f81a1):

| shape | earlier five (range) | cb85d590 | 6f75235c | change (mean vs mean of five) |
|---|---|---|---|---|
| verify 16 full | 119.93-120.85 | 119.95 | 120.49 | -0.1 |
| verify 32 full | 226.57-227.47 | 217.72 | 219.50 | -8.3 (-3.6%) |

Verify 32 full is below the earlier range in both checks by 7-9 us, the size the bench
measured; verify 16 full stays inside it. The other ten shapes run SASS-identical code. Scores
2.311x and 2.361x against the FIN8 pair's 2.347x / 2.316x: one below, one above, the highest
attention score in this log. The expected effect on the score (one shape of twelve, -3.6%) is about
+0.3%, below the run-to-run spread. Attention checks used: 3 of 4.

Submit candidate: `muse_attention_decode/submission.py` at bc59f08 (judged 6f75235c0d364a7a;
a6868f9, judged cb85d590c301410e, differs only by the comment line). Not submitted.

What is left of the gap at verify 32 full: about 46 us warm in the bench (241.5 vs 195.1),
and 23 us at verify 16 full, where this change does nothing. The public loop order, its packed
check and its unroll are now in mode 51, so the rest is elsewhere. Candidates from the source,
untested: the public flat partition and producer (gather code, stage size, `r_lo` / `r_span`
common-tile fast path, descriptors carried across tiles instead of rebuilt per tile), and the
EREL release placed after the transpose without QKOV (QKOV alone was slower here).

### Scored: df988353208245f7 2.354x (2026-10-29)

Submit of submission.py at bc59f08 (public tile order in mode 51, as checked in 6f75235c0d364a7a), under the standing replace-when-demonstrably-better authorization. Passed. Board entry before: 9104344cb757454e (2.334x).
## Attributing the flat q_len 8 gap, and the producer fast path (ONE_PFAST) (10-29)

Question: where do the remaining verify 16 / 32 full microseconds between mode 51 and the public
flat kernel go, now that both run the same tile order?

### Which bench number predicts the judge

The judge flushes L2 before every timed replay (TASK.md, "How a run is timed"), so the bench's
ABCOLD row is the judged quantity and ABWARM is not. The board build measured 222.24 / 217.86 us
cold at verify 32 full in e1d8a55eaa2f455c / 1e6626a427454f9a but 241.54 / 213.27 warm: the warm
row moved 28 us between two processes, the cold row 4.4. The "46 us" gap of the previous entry was
a warm figure. Cold, in the same process (1e6626a427454f9a), mode 51 trails the public kernel by
31.5 us at verify 32 full and 16.5 at verify 16 full, close to the judged gap (218-220 vs 186,
120 vs 105).

### Static comparison (SASS, CUDA 12.8.1, `tools/attn_one_sass.sh`, `tools/attn_pub_compile.sh`)

- Producer, per tile per thread on the common path: mode 51 about 230 instructions (per cached
  row about 28: the page-size branch, BSSY / BSYNC, a 64-bit offset, `LDC` reloads of the cache
  bases; then about 90 to refill the page entries three tiles ahead, each with its own
  page-size branch). The public producer about 115 (one predicated page load per row one tile
  ahead, then a branch-free fast path of 32-bit offsets for tiles whose rows are all cached).
  With 4 producer warps and 8 consumer warps per CTA, the producer's issue slots come out of
  the consumers' SMSPs.
- Consumer, full-barrier wait to the Q.K issue: 94 instructions in mode 51 (the per-tile
  `fence.proxy.async`, the patch test, the descriptors rebuilt per tile), 37 in the public kernel
  (no fence on unpatched tiles, descriptors carried in uniform registers). The fence stays
  (CLAUDE.md); the descriptors are untested.

### Bench 1e6626a427454f9a (check, `tools/attn_make_bench_ab.py --shapes full --nocoop`)

Arms, one process, same inputs: `ctrl` (board build 6f75235c0d364a7a), `nocoop` (same module,
launched without `cudaLaunchAttributeCooperative`, same 264-CTA grid), `stamp` (`ONE_STAMP=1`:
per-CTA records, five cold calls per verify full shape; 4 bytes of spill on 51),
`nocopy_probe` (`ONE_PROBE=1`: the producer's cached-row copies zero-fill, no cache reads),
`nofence_probe` (`ONE_PROBE=2`: no per-tile consumer `fence.proxy.async`; timing only),
`pfast` (`ONE_PFAST=1`), `pub`, `pubnc_probe` (the public file with its producer's cached-row
copies at src-size 0, the same probe). ABCHECK 0 failing (the log keeps its last 16,000
characters, so only the test-shape rows survive; `pfast` matches `ctrl` with 0 differing
elements on the three surviving full-context rows (modes 49, 51, 49, page size 64), and differs
by one element of 1 ulp on a sliding row whose code it does not change, the order of the float
reds). Profiles: `profiles/bench_attr.1e6626a427454f9a.*`. Microseconds:

| | shape | ctrl | nocoop | stamp | nocopy | nofence | pfast | pub | pubnc |
|---|---|---|---|---|---|---|---|---|---|
| cold | decode 16 full | 98.02 | 97.70 | 98.53 | 79.58 | 98.02 | 97.38 | 96.29 | 65.73 |
| cold | decode 32 full | 175.52 | 175.26 | 176.29 | 148.06 | 175.58 | 173.76 | 172.00 | 121.63 |
| cold | verify 16 full | 124.38 | 122.59 | 126.46 | 107.68 | 122.56 | 118.82 | 107.87 | 89.28 |
| cold | verify 32 full | 217.86 | 216.48 | 221.28 | 195.84 | 216.00 | 206.37 | 186.40 | 157.54 |
| warm | verify 16 full | 114.33 | 114.14 | 116.13 | 101.64 | 113.26 | 106.09 | 93.10 | 83.24 |
| warm | verify 32 full | 213.27 | 213.06 | 217.03 | 197.11 | 211.06 | 197.21 | 175.35 | 158.99 |

Stamps (`stamp` build, verify 32 full, five cold calls; first / median / last CTA; times from
the earliest CTA start): first tile ready 4.0-4.1 / 4.9-5.0 / 6.0-6.8 us; tile loops done
167.5-173.9 / 205.2-205.8 / 215.9-218.8 us; wait at the grid barrier median 27.4-30.0, max
42.9-52.1 us; finalize after the barrier median 1.1-1.9 us; tiles per CTA 143 / 152 / 153, one or
two segments. Consumer thread 0 waits on the full barrier 13.6-17.2 us (median) of a 194 us tile
loop; producer thread 0 waits on the empty barrier 24.1-27.1 us of its 202 us. Verify 16 full:
loops done 86.5-89.9 / 106.5-108.8 / 116.0-117.8 us, tiles 67 / 76 / 78. The two CTAs of one SM
finish apart: in the first verify 32 call CTA 22 (SM 62, 153 tiles) finished its loop at 192.1 us
and CTA 146 (SM 62, 152 tiles) at 218.8.

### Attribution at verify 32 full, cold (gap 31.5 us)

| piece | evidence | size | status |
|---|---|---|---|
| Gather rate or memory | without cache reads both kernels keep the gap: 195.84 vs 157.54 (38.3 us) | none of the gap | measured: the gap is in the instruction streams, not in DRAM |
| Producer instruction issue | `pfast` 206.37 vs 217.86 | 11.5 cold, 16.1 warm | measured, the largest named piece |
| Consumer per-tile proxy fence | `nofence_probe` 216.00 | 1.9 cold, 2.2 warm | measured; the fence stays |
| Cooperative launch attribute | `nocoop` 216.48 | 1.4 cold, 0.2 warm | measured, at the noise floor |
| Prologue to the first tile, finalize | stamps: about 5 us and 1-2 us | under 7 us, and the public kernel runs the same steps | measured for mode 51; the public share not measured |
| Imbalance between co-resident CTAs | stamps: loops end 167-219 us with equal tiles; same-SM pairs 27 us apart | upper bound about 13 us (last CTA vs median) | inferred; the public kernel's spread not measured |
| Remaining consumer instruction path | 38.3 us no-copy gap minus the producer's share | the rest | inferred, not split further (descriptors, the pre-Q.K path) |

The verify 16 full gap (16.5 us cold) splits the same way: pfast -5.6 cold, -8.2 warm; no-copy
gap 18.4 us.

### The change: `ONE_PFAST` (modes 49, 51, 115)

`attn_one.cu`, producer warpgroup: a tile whose rows are all cached (`pos0 + BN <= seq`, which
also puts every row below `hi`) is gathered without the per-row branches, with the byte offset
`(((pe << lp) + ((pos0 + r) & (psize - 1))) << 7) + 16 ch` in 32 bits; its page entries three
tiles ahead are refilled by shift when that tile is also fully cached. Every other tile takes
the old code. The host sets `Params::off32` only when the page size is a power of two and
both caches hold at most 2^32 bytes, so the offsets fit. Producer path about 125 instructions
per tile (from about 230). With the switch on, only 49, 51 and 115 change
(`tools/attn_one_sass.sh` over all 13 instances, per-function diff); no spills, mode 51's
consumer region still tops out at R101, its producer at R29. With `ONE_PFAST=0`, `ONE_STAMP=0`,
`ONE_PROBE=0` all 13 instances are SASS-identical to the board build.

Ordering: the fast path issues the same `cp.async` set into the same stage addresses as the
slow path would for such a tile (every row cached, no zero-fill, no patch), followed by the same
`cp.async.mbarrier.arrive.noinc`. No fence, barrier or arrive moves; only address arithmetic
and branch structure change.

### Judged: check ca0f13d324ef48e8 2.393x (CI 2.357-2.429), repeat 0d05f9b2df1e4834 2.450x (CI 2.413-2.487)

`submission.py` at 70323d2 (`ONE_PFAST=1`); the repeat, 46b1708, adds one comment line. 16/16
tests in both. Profiles: `profiles/submission.ca0f13d324ef48e8.*`,
`profiles/submission.0d05f9b2df1e4834.*`. Judge medians (us), against the previous pair
(cb85d590c301410e / 6f75235c0d364a7a, the same code without the switch):

| shape | cb85d590 | 6f75235c | ca0f13d3 | 0d05f9b2 | change (mean vs mean) |
|---|---|---|---|---|---|
| decode 16 full | 88.91 | 89.60 | 88.29 | 85.08 | -2.6 |
| decode 32 full | 170.68 | 170.52 | 157.49 | 162.84 | -10.4 |
| verify 16 full | 119.95 | 120.49 | 114.49 | 114.33 | -5.8 (-4.8%) |
| verify 32 full | 217.72 | 219.50 | 206.86 | 205.69 | -12.3 (-5.6%) |

Verify 16 / 32 full moved by the sizes the bench measured cold (-5.6 / -11.5). Decode 32 full
dropped by more than the bench predicted (-1.8 cold): five earlier checks of mode 49 sat at
170.5-171.4, both of these are 7.7-13 us below. That gain is measured twice; why the judge sees
more of it than the bench is not measured. Scores 2.393x and 2.450x, the highest attention
checks in this log (previous pair 2.311x / 2.361x). Attention checks used today: 3 of 4
(1e6626a427454f9a, ca0f13d324ef48e8, 0d05f9b2df1e4834).

Submit candidate: `muse_attention_decode/submission.py` at 46b1708 (judged 0d05f9b2df1e4834;
70323d2, judged ca0f13d324ef48e8, differs only by the comment line). Not submitted.

What is left at verify 32 full, cold in the bench: 206.4 vs 186.4 (20 us); verify 16 full 118.8
vs 107.9. Leads in order of the evidence: the consumer path before the Q.K issue (94 vs 37
instructions; descriptors carried in uniform registers, the patch test by one compare as the
public `r_lo` / `r_span` does), and the spread between the two CTAs of an SM (stamps above; the
public q_len 8 flat path cuts equal shares too, `cpart = c`, so whether it shows the same spread
is not known; a partition weighted per SM slot is untested).

### Scored: 51d4d137b9d14254 2.416x (2026-10-29)

Submit of submission.py at 46b1708 (ONE_PFAST, as checked in 0d05f9b2df1e4834), under the standing replace-when-demonstrably-better authorization. Passed. Board entry before: df988353208245f7 (2.354x).
## Per-CTA finish spread in both flat kernels, and two switches for mode 51 (10-29)

Question: is the spread between the two CTAs of an SM a piece of the verify full gap to the
public kernel, and is it larger than the consumer path before the Q.K issue?

### Bench 7631033a0c93485d (check, `tools/attn_make_bench_ab.py --shapes full`)

Arms in one process: `ctrl` (board build 0d05f9b2df1e4834 code, `ONE_PFAST=1`), `stamp`
(`ONE_STAMP=1`), `w115` / `w130` (`ONE_WAVE` as first built: CTAs 0 .. 131 take 1.15 / 1.30 of a
CTA 132 .. 263 share), `stampw130`, `pub`, and `pubstamp`: a scratchpad copy of the public file
with `%globaltimer` stamps per CTA at kernel entry, at the consumers' loop end and after the
grid-wide finalize, plus `%smid` and the tile count (into its own `dbg` argument; output
unchanged). New bench lines: `STAMPWAVE` (loop ends by dispatch half and by SM: the gap between
the two CTAs of an SM, and each SM's later end) and `PUBSTAMP`. Judge tests 16/16, run score
2.373x (CI 2.337-2.409; the bench's own arms do not enter it). The log keeps its last 16,000
characters: the ABCHECK rows and the `stamp` arm's first four repetitions are cut, so the wave
arms' output check is not on record (they are losers either way). Profiles:
`profiles/bench_wave.7631033a0c93485d.*`. Microseconds:

| | shape | ctrl | stamp | w115 | w130 | stampw130 | pub | pubstamp |
|---|---|---|---|---|---|---|---|---|
| cold | decode 16 full | 100.06 | 100.32 | 100.03 | 100.03 | 100.26 | 98.94 | 99.01 |
| cold | decode 32 full | 179.90 | 180.26 | 180.13 | 179.78 | 180.29 | 178.14 | 178.14 |
| cold | verify 16 full | 121.76 | 123.17 | 126.62 | 132.06 | 133.66 | 111.20 | 111.78 |
| cold | verify 32 full | 212.48 | 214.30 | 222.30 | 232.83 | 234.62 | 193.25 | 192.64 |
| warm | verify 16 full | 116.36 | 118.06 | 121.69 | 127.38 | 129.21 | 101.47 | 100.88 |
| warm | verify 32 full | 221.96 | 223.28 | 232.94 | 249.83 | 253.13 | 193.61 | 192.78 |

Loop ends at verify 32 full, cold, microseconds from the grid's first start (min / median /
max over CTAs or SMs; ours one repetition, the public kernel five):

| | mode 51 (`stamp`) | public (`pubstamp`, 5 reps) | mode 51, `w130` (5 reps) |
|---|---|---|---|
| all CTAs | 158.2 / 195.4 / 205.6 | 156.3-158.8 / 173.9-174.9 / 182.8-184.6 | 141.0-144.9 / 186.8-188.2 / 223.3-228.4 |
| gap within an SM (later minus earlier CTA) | 28.8 / 35.5 / 41.1 | 1.0-2.6 / 11.2-12.2 / 19.9-23.8 | 2.3-5.5 / 10.8-11.7 / 42.9-45.8 |
| lower block index ends first | 132 of 132 SMs | 130-132 of 132 | 132 of 132 |
| each SM's later end | 195.4 / 200.5 / 205.6 | 171.9-174.4 / 177.7-179.1 / 182.8-184.6 | 174.6-176.4 / 192.9-195.8 / 223.3-228.4 |

Verify 16 full, public: within-SM gap median 5.4-6.3 us, SM ends 91.8-94.1 / 96.4-98.4 /
102.6-104.9.

What this measures:

- The public kernel has the same pattern (the lower block index of an SM ends first on 130-132
  of 132 SMs) with a third of the gap (12 vs 35 us). Its SMs end within 11-12 us of each other,
  ours within 10. The two kernels differ by 22 us in the median SM's end (178 vs 200.5), the
  same size as the cold gap (193.3 vs 212.5 here): the gap is per-SM throughput, not spread
  across SMs.
- Weighting by block index lost: +9.8 / +20.4 us cold at verify 32 full (w115 / w130), +4.9 /
  +10.3 at verify 16. Block dispatch does not put CTAs 0 .. 131 on distinct SMs: under w130 the
  last SMs to finish (124-131) held two of the heavy CTAs (173 tiles each, ending 224-228 us),
  while SMs holding two light ones ended at 144. On SMs that got one of each, the pair gap fell
  from 35 to 11 us and the SM's end from 200.5 to about 193 (median over all SMs 193.4).
- Bound for a correct pair split, inferred from those SMs: the earlier CTA's end rises about
  0.9 us per tile moved to it and the later one's falls about 0.26, so equal ends come near 192
  us, about 8 us before today's median SM end, at a weight near 1.5.

### The two switches (not judged yet)

`ONE_WAVE=w` (rebuilt after this bench): the pair is found at run time. One thread per CTA
takes a rank from a per-SM counter (`ws_cnt[CNT_SMR + smid]`, RMW, the CTA that sees an odd
value resets it to 0, so the counter is zero between launches whatever it held); slot
`2 smid + rank` sets the CTA's share, rank 0 weighted w / 1024 against its partner. The host
enables it only for cooperative q_len 8 flat launches (co-residency of 2 x SMs CTAs at two per
SM puts exactly two on every SM) and only when a one-time probe saw every `%smid` below the SM
count. Ordering argument: coherence and atomicity of one location (PTX ISA 8.10.1, 8.10.3); the
counter feeds only the partition, which both roles compute before they split, and no fence or
barrier moves.

`ONE_DCARRY`: the tiles whose keys all lie below the block's own positions (`it < (seq - lo) /
64`, window 0) skip the patch test and the mask test with one uniform compare; the Q descriptor
low words and the K descriptor base are broadcast from lane 0 (uniform registers), the K
descriptor of a stage is base + 1024 x stage. SASS, mode 51 unrolled body: full-barrier wait to
the Q.K issue about 25 instructions on a common tile (from about 50), function 6512
instructions (from 6632), no spills, consumer region R101. The fence after the full barrier and
the patch path of the other tiles are unchanged.

With both switches at 0 all 13 instances are SASS-identical to the board build
(`tools/attn_one_sass.sh`, per-function compare). Next: one bench of `dc`, `wv140`, `wv150`,
`stampwv150`, `dcwv150` against `ctrl` and `pub` after the 00:00 UTC reset, then judge the best
if it saves at least 4 us cold at verify 32 full.

### Bench 4ff339332fef4aff (check, `--shapes full`): ONE_DCARRY and the run-time pair split

Arms: `ctrl` (board code), `dc` (`ONE_DCARRY=1`), `wv140` / `wv150` (`ONE_WAVE` with the per-SM rank
counter, weight 1.40 / 1.50), `stampwv150`, `dcwv150`, `pub`. ABCHECKSUM 11/0 pass/fail for every
arm (`dc`, `wv*` compared with `ctrl` on the scored and seven tp8 test shapes). Judge tests 16/16,
run score 2.400x (CI 2.364-2.436; the arms do not enter it). Profiles:
`profiles/bench_dc.4ff339332fef4aff.*`. Microseconds:

| | shape | ctrl | dc | wv140 | wv150 | stampwv150 | dcwv150 | pub |
|---|---|---|---|---|---|---|---|---|
| cold | decode 16 full | 100.03 | 100.13 | 100.03 | 100.03 | 100.26 | 100.13 | 99.01 |
| cold | decode 32 full | 180.03 | 180.06 | 180.29 | 180.22 | 180.51 | 180.10 | 178.78 |
| cold | verify 16 full | 120.03 | 116.38 | 125.50 | 125.15 | 127.39 | 122.24 | 110.40 |
| cold | verify 32 full | 210.59 | 203.26 | 219.58 | 222.59 | 223.97 | 215.55 | 191.78 |
| warm | verify 16 full | 113.63 | 108.54 | 119.62 | 120.79 | 122.54 | 114.92 | 99.07 |
| warm | verify 32 full | 221.05 | 207.37 | 230.33 | 232.11 | 235.17 | 221.05 | 192.14 |

- `dc`: -7.3 / -3.7 us cold, -13.7 / -5.1 warm at verify 32 / 16 full; decode full (mode 49)
  unchanged, as the SASS says. It closes about a third of the cold gap to the public kernel at
  verify 32 full (19.0 -> 11.5 us). Promoted for judging.
- Run-time pair split: loses (+9.0 / +12.0 cold at verify 32 full for 1.40 / 1.50). Stamps
  (`stampwv150`, five reps): the rank counter did not pick the favoured CTA; the lower block
  index now ends first on only 21-63 of 132 SMs, the median pair gap is -1.5 to -3.9 us but the
  worst pair is 72-83 us apart, and the latest SM ends at 212-215 us against 205.6 unweighted.
  Arrival order at the counter does not follow the warp schedulers' priority.

### Judged: check f64e8680b3e441be 2.395x (CI 2.359-2.431), repeat bf49eceb15a64f81 2.413x (CI 2.377-2.461)

`submission.py` at 0d76acc (`ONE_DCARRY=1`); the repeat, 503c0f4, adds one comment line. 16/16
tests in both. Profiles: `profiles/submission.f64e8680b3e441be.*`,
`profiles/submission.bf49eceb15a64f81.*`. Judge medians (us) against the ONE_PFAST pair
(ca0f13d324ef48e8 / 0d05f9b2df1e4834, the same code without the switch):

| shape | ca0f13d3 | 0d05f9b2 | f64e8680 | bf49eceb | change (mean vs mean) |
|---|---|---|---|---|---|
| decode 16 full | 88.29 | 85.08 | 88.00 | 88.37 | +1.5 (mode 49, SASS unchanged) |
| decode 32 full | 157.49 | 162.84 | 168.87 | 168.78 | +8.7 (mode 49, SASS unchanged) |
| verify 16 full | 114.49 | 114.33 | 111.05 | 110.49 | -3.6 (-3.2%) |
| verify 32 full | 206.86 | 205.69 | 197.83 | 198.91 | -7.9 (-3.8%) |

Verify 16 / 32 full moved by the sizes the bench measured cold (-3.7 / -7.3). Decode 32 full rose
8.7 us on a kernel whose SASS is identical to the previous pair's (all instances but 51 and 115
compare equal); five checks before ONE_PFAST sat at 170.5-171.4, so the 157-163 of the previous
pair, not this one, is the outlier there. That row is why the scores (2.395x / 2.413x) do not
rise over the previous pair (2.393x / 2.450x). Checks run in this
session: 7631033a0c93485d, 4ff339332fef4aff, f64e8680b3e441be, bf49eceb15a64f81.

Submit candidate for the verify full rows: `muse_attention_decode/submission.py` at 503c0f4
(judged bf49eceb15a64f81; 0d76acc, judged f64e8680b3e441be, differs only by the comment line).
Not submitted. Left at verify 32 full, cold in bench 4ff339332fef4aff: 203.3 vs 191.8 (11.5 us).

### Scored: 4c12b0199efd4b8c 2.412x (2026-10-29)

Submit of submission.py at 503c0f4 (ONE_DCARRY, as checked in bf49eceb15a64f81), submitted by the coordinator under the user's instruction to use expiring scored runs. Passed. Board entry before: 51d4d137b9d14254 (2.416x).

## The decode 32 full row, and P packing plus V^T hoisting in mode 51 (10-29)

Questions: why does the judge put mode 49 at 157-171 us at decode 32 full on identical SASS, and
what is left of the consumer's per-tile instruction stream in mode 51 against the public kernel?

### Decode 32 full: the spread is in SGLang's times, not ours (no run, profile JSONs)

Read from `benchmarks[*]` in the committed profiles (`runs_ns`, `baseline_runs_ns`, `ratios`,
`paired_ratio`). Mode 49's raw median at decode 32 full, microseconds: 174.9 (6f75235c), 172.7
(ca0f13d3), 166.9 (0d05f9b2), 172.8 (f64e8680), 172.6 (bf49eceb). SGLang's FA3 baseline on the
same shape is bimodal: most repetitions take 377-395 us and 5-15 of the 60 take 513-528 us
(15 of 60 in ca0f13d3, 5 in 0d05f9b2, 10-13 in the others). The shape's speedup is the median of
paired ratios, and the reported time is SGLang's median over it, so the slow FA3 repetitions move
our reported time while our own replays do not move:

| check | our raw median | slow FA3 reps | paired ratio | median of per-rep ratios | reported |
|---|---|---|---|---|---|
| 6f75235c0d364a7a | 174.9 | 13 / 60 | 2.263 | 2.262 | 170.5 |
| ca0f13d324ef48e8 | 172.7 | 15 / 60 | 2.447 | 2.283 | 157.5 |
| 0d05f9b2df1e4834 | 166.9 | 5 / 60 | 2.375 | 2.375 | 162.8 |
| f64e8680b3e441be | 172.8 | 11 / 60 | 2.280 | 2.280 | 168.9 |
| bf49eceb15a64f81 | 172.6 | 11 / 60 | 2.274 | 2.272 | 168.8 |

- ca0f13d3's 157.5 is the judge's statistic, not our kernel: its paired ratio (2.447) sits above
  the median of its own per-repetition ratios (2.283), with a quarter of the FA3 repetitions slow.
  Our replays there are 169.5-175.5 us, the same as every other check.
- 0d05f9b2 is about 6 us faster in our raw times, and its decode 16 full (89.2 raw vs 91.9-93.6)
  and verify rows were faster too: a faster machine or run, same SASS. Not reproducible on demand.
- So the "+8.7 us" of the DCARRY pair at decode 32 full is the board's earlier lucky draw
  returning to the usual 172-175, as the five checks before PFAST (170.5-171.4) said. Nothing
  in mode 49 to fix from this row; it trails the public kernel by about 1-2 us in the bench.

### Mode 51's consumer stream against the public one (SASS, CUDA 12.8.1, `tools/attn_one_sass.sh`)

With DCARRY in, the common tile's path from the full-barrier wait to the P.V wait is about 223
instructions per consumer thread in mode 51. Two pieces of it have no counterpart in the public
loop (its `attn_kernel<8, true, 0>`):

- P packing: our `e4m3x4` joined its halves with `__byte_perm(lo, hi, 0x5410)`, which compiles
  to `F2FP ... RZ`, `LOP3 & 0xffff` per half and a `PRMT`: 40 instructions per tile for the
  16 words of P. The public `e4m3x4f` writes `lo | hi << 16`, which ptxas folds into the second
  `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C`: 16. Same bits.
- V^T addressing: the first body of the twice-unrolled loop rebuilt the ldmatrix and stmatrix
  addresses from `tid` every tile (about 25 instructions; ptxas rematerialized them) and every
  body rebuilt the P.V descriptor from `vt_of()` (about 14). The public loop carries both.

`ONE_PK16` and `ONE_VHOIST` (`attn_one.cu`, commit a50fc0c, both default 0 there) remove them:
the unrolled pair of tiles in mode 51 drops from 1265 to 1122 SASS instructions, the first body
from Q.K issue to the stage release from 45 to 16, no spills, consumer region R99. With both off
all 13 instances are SASS-identical to the board build. VHOIST's offsets were checked against the
original formulas for every (head, warp, lane, half) in Python (0 mismatches); its loads and
stores move the same bytes to the same addresses, so no fence, barrier or arrive moves.

### Bench 025ccaf9c57a4ba2 (scored run, `tools/attn_make_bench_ab.py --shapes all`)

Arms: `ctrl` (board code), `pk` (`ONE_PK16=1`), `vh` (`ONE_VHOIST=1`), `pkvh` (both), `pub`.
ABCHECKSUM 19/0 pass/fail for every arm; against `ctrl` the arms differ in 0-6 elements per shape
by 1 ulp of bf16 (21 ulp on one near-zero element), the order of the float reds. Run score 2.394x
(CI 2.358-2.442; the board build's own score, the arms do not enter it). Profiles:
`profiles/bench_pk.025ccaf9c57a4ba2.*`. Microseconds:

| | shape | ctrl | pk | vh | pkvh | pub |
|---|---|---|---|---|---|---|
| cold | verify 16 full | 116.26 | 112.70 | 114.24 | 110.91 | 110.34 |
| cold | verify 32 full | 204.22 | 197.89 | 200.29 | 194.59 | 192.67 |
| warm | verify 16 full | 114.15 | 109.00 | 111.15 | 106.63 | 103.29 |
| warm | verify 32 full | 214.13 | 203.91 | 208.77 | 200.25 | 197.19 |
| cold | verify 1 full | 20.22 | 19.90 | 20.22 | 19.97 | 19.14 |
| cold | verify 16 / 32 sliding | 16.13 / 21.12 | 16.13 / 21.22 | 16.06 / 21.25 | 16.13 / 21.34 | 15.71 / 19.78 |
| cold | decode 16 / 32 full | 100.45 / 180.22 | 100.38 / 180.16 | 100.42 / 180.22 | 100.32 / 180.16 | 99.46 / 178.59 |

- `pkvh`: -9.6 / -5.4 us cold, -13.9 / -7.5 warm at verify 32 / 16 full. The cold gap to the
  public kernel falls from 11.5 to 1.9 us at verify 32 full and to 0.6 at verify 16 full. The two
  switches add (6.3 + 3.9 = 10.2 vs 9.6 cold at verify 32).
- Every other row moves by less than 0.35 us either way (q_len 1 builds change only where they
  call `e4m3x4` once per launch). Verify 32 sliding +0.2 us is inside the bench's run-to-run
  spread for that row.
- Promoted for judging: `submission.py` with both switches on.

### Scored: 569b233dfe08414f 2.449x (CI 2.413-2.502) (2026-10-29)

Scored run of `submission.py` at c37d381 (`ONE_PK16=1`, `ONE_VHOIST=1`), submitted by the
coordinator. 16/16 tests. Board entry before: 51d4d137b9d14254 (2.416x); this is above it.
Profiles: `profiles/cand_pkvh.569b233dfe08414f.*`. Judge medians (us) against the DCARRY pair
(the same code without the switches):

| shape | f64e8680 | bf49eceb | 569b233d | change (vs mean) |
|---|---|---|---|---|
| verify 16 full | 111.05 | 110.49 | 103.65 | -7.1 (-6.4%) |
| verify 32 full | 197.83 | 198.91 | 187.67 | -10.7 (-5.4%) |
| decode 16 / 32 full | 88.00 / 168.87 | 88.37 / 168.78 | 87.83 / 168.12 | -0.4 / -0.7 |
| verify 16 / 32 sliding | 11.78 / 16.26 | 11.98 / 16.41 | 11.89 / 16.42 | +0.0 / +0.1 |

Verify 16 / 32 full moved by more than the bench's cold deltas (-5.4 / -9.6); the public
kernel's judged verify full times were 105 / 186 (docs/runs-attention.md, 10-28 / 10-29), so
mode 51 is now level with it at verify 16 full and within 2 us at verify 32 full.

### Bench e9085c86d89449c2 (scored run, `--shapes full`): repeat of the candidate, stamps, fence probe

The judged path is `submission.py` at 61e3fd4 (the 569b233d code; docs only changed). Arms:
`ctrl` (that build), `stamp` (`ONE_STAMP=1` on it), `nofence_probe` (`ONE_PROBE=2`), `pub`,
`pubstamp`. ABCHECKSUM 11/0 pass/fail for every real arm (the probe's 11 marked PROBE). Profiles:
`profiles/bench_spread.e9085c86d89449c2.*`.

Judged (the repeat): **2.504x (CI 2.467-2.542)**, 16/16 tests. Verify 16 / 32 full 100.83 /
182.24 us (569b233d: 103.65 / 187.67). This run's machine was faster on every full row, ours and
SGLang's raw medians alike (decode 16 full raw 89.0 vs 92.1, decode 32 full raw 166.3 vs 172.8 on
unchanged mode 49 SASS), so part of the step from 2.449x is the draw, as with 0d05f9b2df1e4834.
Its verify full ratios (2.043 / 2.153) against 569b233d's (1.979 / 2.079) are the cleaner
comparison: both are this code.

Bench, microseconds:

| | shape | ctrl | stamp | nofence_probe | pub |
|---|---|---|---|---|---|
| cold | decode 16 / 32 full | 97.38 / 174.14 | 97.82 / 174.40 | 97.47 / 173.98 | 96.48 / 172.29 |
| cold | verify 16 full | 111.14 | 111.94 | 109.98 | 108.99 |
| cold | verify 32 full | 192.42 | 193.86 | 190.98 | 188.90 |
| warm | verify 16 full | 103.00 | 104.51 | 102.25 | 99.73 |
| warm | verify 32 full | 195.55 | 198.52 | 194.55 | 192.05 |

Loop ends at verify 32 full, cold, microseconds from the grid's first start (median over five
repetitions of min / median / max), against bench 7631033a0c93485d (before DCARRY, PK16, VHOIST):

| | mode 51 now | mode 51 in 7631033a | public now |
|---|---|---|---|
| gap within an SM (later minus earlier CTA) | 14.1 / 21.6 / 27.6 | 28.8 / 35.5 / 41.1 | 6.6 / 14.4 / 22.8 |
| each SM's later end | 169.0 / 175.7 / 181.3 | 195.4 / 200.5 / 205.6 | 168.5 / 173.7 / 179.9 |
| earlier CTA of an SM (wave 1 median) | 154.5 | about 165 | 159.7 |

- The pair gap fell from 35.5 to 21.6 us as the consumer's per-tile instructions fell, and the
  median SM end from 200.5 to 175.7 (public 173.7). This fits the reading of the 7631033a entry:
  the scheduler's favoured CTA of each pair takes the issue slots it needs and its partner gets
  the rest, so fewer instructions per tile shorten the partner most. The favoured CTA now ends
  before the public one's (154.5 vs 159.7) and the partner still later (pair gap 21.6 vs 14.4).
- Left at verify 32 / 16 full, cold: 3.5 / 2.2 us. The consumers' per-tile proxy fence is 1.4 /
  1.2 of it (`nofence_probe`; the fence stays, as before). Decode 16 / 32 full (mode 49, not
  touched): 0.9 / 1.9 us behind the public kernel.

Attention scored runs today: 025ccaf9c57a4ba2, 569b233dfe08414f, e9085c86d89449c2 (3 of 4).

## Input-history dependence at decode 16 / 32 full: merge order, not a carry (10-30)

Question (from the layer's history test, d93286b5110a4410 on the 2.014x layer 3fbb3ad027e84c6b):
attention at 16x1f / 32x1f showed rep 0, hist 3e-5 / 8e-5, sep 3e-5 / 8e-5. Either (A) the
previous call changes timing and so the order in which the FP32 reds reach L2, or (B) state
carried through the workspace or counters, or read before it is published.

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 2b93c6ecf4f641c6 (check) | 389da5c | `muse_attention_decode/hist_diag.py` (`tools/attn_hist_diag.py`) | **A: merge order.** 16/16 tests, 2.462x (2.425-2.499), production path unchanged | diagnostic, not a candidate; profiles committed |

Build: `attn_one.cu` gained `ONE_DETM` (default 0, diagnostic only). With 1, modes 17 / 49 store
each warp's fixed-frame partial (o * 2^m, l * 2^m) into its own slot in `dbg` instead of reds,
and the request's last CTA (same ticket) sums the slots of every split that holds keys of the
request in split order, warp 0 before warp 1. The slots are never zeroed, so a read before
publication would read the previous call's partials. Compile gate: the production build's SASS
(all 13 instances, CUDA 12.8.1, `tools/attn_ws_quick.sh -s`) is identical to the build before
the change (77,581 lines, kernel names normalized); host code compiles in both builds
(`tools/attn_host_check.sh`). Mode 49 in the DETM build: no REDG, 4 STG.128 + 1 STG per thread
for the slots, the same ATOMG ticket and MEMBAR.ALL.GPU, no spills.

Protocol (decode, ctx 80000 on every request, page size 1, plan (264, 1, 49), four random qkv
inputs, the layer's `_hist_probe` definitions; `other` is a second case with the same seed and
its own caches):

| arm | 16x1f rep / hist / sep | 32x1f rep / hist / sep | distinct outputs, 9 calls on one input | worst diff |
|---|---|---|---|---|
| prod | 5e-5 / 6e-6 / 5e-5 | 9e-6 / 7e-5 / 7e-5 | 4 / 5 | 1-3 elements, 1 BF16 ulp |
| prod, previous call on a copy of input k (same values, other address) | 5e-5 | 7e-5 | | 1-2 elements |
| prod, 256 MB L2 flush before every call | 2e-4 / 2e-4 / 2e-4 | 7e-5 / 7e-5 / 7e-5 | 4 / 4 | 1 BF16 ulp |
| det | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 | 0 |
| det, flush before every call | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 | 0 |

- det against prod: 5.2e-5 / 7.3e-5 (one element of 16,384 / 32,768, one BF16 ulp). det on
  input 0 after interleaved prod calls and det calls on the other case: 0.
- Workspace scan after every production call (283 scans): every row, row sum and ticket zero,
  the next cooperative parity counter of every request and the flat one zero, every int from
  `ws_cnt[16387]` on zero. Dirty: 0. Including after a call on each other plan: modes 11
  (1x1f, 1x8f, nc 8), 13 (1x1s), 1 (32x1s), 3 (16x8s), 51 (16x8f) and 49 at batch 32 page 64;
  after each, 16x1f on a fixed input differed from its reference by 0 or one element at one
  ulp (the same noise as rep).

Reading: in production, repeated inputs are not bitwise either (4-5 distinct outputs in nine
calls); the layer's rep 0 was a draw, not a property. A previous call with identical values at
another address moves the output as much as a call on a different input, which a carry of
values cannot do. With the merge order fixed, every arm is bitwise under every history
(previous input, the separate case, flushes, interleaved production calls), and the slots
the DETM build reads are never cleared, so a read before publication or a stale reuse would
have shown there. The entry state is all-zero after every plan. The magnitude, one element
moved by one BF16 ulp, is FP32 sum-order rounding. Verdict A; no change to the shipped kernel.

### Ordering chain: mode 49 workspace reset and reuse

Sections are from the PTX ISA 9.4 documentation as cited in the 10-29 entries above (8.8
Release and Acquire Patterns, 8.9.4 Memory synchronization, 8.10.1 / 8.10.3 coherence order
of a location's writes and RMWs, 9.7.15.14 `griddepcontrol`). The barrier rules are the
`bar.sync` semantics the ticket already relied on in review; their subsection number was not
re-read this session.

Global state a mode 49 launch reads that an earlier launch wrote: the rows `ws_acc` / `ws_l`
`[b * 16 + head]` (nc 1, by `red.global.add` and the finalizer's `ld.global.cg`) and the ticket
`ws_cnt[b]` (by `atom`). Nothing else: the epoch copy is `cp.async` with source size 0 when
not cooperative (zero fill, no read), `ws_cnt[CNT_FLAT + 2]` is read only by cooperative
modes, and `seq_lens`, `page_table`, `qkv`, `cache_loc` and the caches are inputs.

1. Every consumer thread of every CTA that holds keys of request b adds its partial with
   `red.global.add.f32` / `.v4.f32` (relaxed, .gpu scope by default) into the rows of b.
2. `bar.sync 1, 128` across the consumers, then thread 0 issues `atom.acq_rel.gpu.add` on
   `ws_cnt[b]` with its key count. The barrier puts every consumer's reds before thread 0's
   release in causality order (program order plus the barrier's synchronization, 8.9.4); the
   atom is a release pattern (8.8).
3. All RMWs of `ws_cnt[b]` are totally ordered in its coherence order (8.10.1, 8.10.3), and the
   key ranges tile the request exactly, so exactly one CTA's RMW reads `old + mine >= n_keys`.
   That RMW reads the value written through the chain of every other CTA's release RMW, so each
   of those releases synchronizes with its acquire (observation order through an RMW chain,
   8.8 / 8.9.4). Every red of every other CTA to b's rows therefore happens before it.
4. Thread 0 writes the shared flag and `bar.sync` again; the other consumers' `ld.global.cg`
   of b's rows follow that barrier, so the reds of step 1 are visible to them. No red to b's
   rows can follow in this launch: every contributing CTA has already arrived, after its reds.
5. Each finalizer thread zeroes, with `st.global.cg`, exactly the words it loaded, after its
   own load (same thread, same address: coherence order, 8.10.1). Thread 0 then stores
   `ws_cnt[b] = 0`; no other CTA touches the ticket after the completing RMW.
6. The next launch on the stream starts after this grid completes and its writes are performed
   (stream order; under PDL in the layer, `griddepcontrol.wait` waits for the prerequisite
   grid's completion and memory flush, 9.7.15.14, and the layer places the wait before the
   first workspace access). So it sees zero rows and a zero ticket.

Other plans keep the same invariant with their own finalizers (mode 11 reads and zeroes all
eight copies; cooperative modes clear the next parity counter before bumping the epoch); the
283 scans above measured it after every plan the judge's shapes route to. The cache row the
launch writes (the block's own key and value) is read by no other CTA in the launch: the one
CTA whose range holds the position patches its tile from shared memory after the full-barrier
wait, over whatever the producer gathered.

## Standalone round 10-30: A0 scorecard and A1 cross-tile overlap in mode 51

Brief: docs/steering/2026-10-30-attention-projection-standalone-roadmap.md, sections 1, 2, 4, 6, 7.
Start: main 7703f0a, board entry e9085c86d89449c2 (2.504x, a bench file), clean candidate-only best
569b233dfe08414f (2.449x). Quota at the start (`okc problems`, 07:49 UTC): 19 of 20 checks.

### A1 design: what crosses the tile-loop edge

The board build ends every tile with `wgmma.wait_group 0` twice (after Q.K and after P.V), as the
public flat kernel does. `ONE_XPIPE` (`attn_one.cu`, default 0, modes 51 and 115 only) keeps the
operands, the masks, the lazy offset, the P rounding and the merge, and changes only where the
waits sit. Per tile t of a segment (tile 0's Q.K and V^T transpose run in a prologue, the last
tile's P.V is drained before the merge):

```
                        XPIPE=2 (PV first)                    XPIPE=1 (QK first)
softmax(t): mask, P(t)  [s(t) -> pa]                          [s(t) -> pa'; PV(t-1) still reads pa, o]
                                                              rare offset move: wait 0, then rescale o
                        PV(t) issue  (bar.sync V^T(t))        QK(t+1) issue (stage t+1 full wait)
in flight now:          PV(t)                                 PV(t-1), QK(t+1)
                        QK(t+1) issue (stage t+1 full wait)   wait 1  -> PV(t-1) retired
                                                              PV(t) issue (bar.sync V^T(t))
in flight now:          PV(t), QK(t+1)                        QK(t+1), PV(t)
                        transpose V(t+1) -> V^T buf (t+1)&1   transpose V(t+1) -> V^T buf (t+1)&1
                        wait 0 -> PV(t), QK(t+1) retired      wait 1 -> QK(t+1) retired
                        release stage t+1                     release stage t+1
carried to tile t+1:    nothing in flight                     PV(t) (under softmax(t+1))
```

Lifetimes (who owns each buffer, and the event that ends the previous owner's use):

| buffer | written by | read by | reuse allowed after |
|---|---|---|---|
| stage of tile t+1 (K, V) | producer cp.async | Q.K(t+1) wgmma, this thread's ldmatrix of V(t+1) | the wait that retires Q.K(t+1) (wait 0 / wait 1) and the ldmatrix before it in program order; then `mbarrier.arrive` on the empty barrier (unchanged from EREL) |
| V^T buffer (t & 1) | stmatrix of the 4 warps of the warpgroup (tile t, then t+2) | P.V(t) wgmma | P.V(t) retired in every warp: each warp retires it before it arrives at the bar.sync of tile t+1, and the transpose of tile t+2 follows that bar.sync in every warp |
| s (32 f32) | Q.K(t+1) wgmma | softmax(t+1) | P(t) packed from s(t) before Q.K(t+1) is issued |
| pa (8 b32) | pack of tile t | P.V(t) wgmma (A from registers) | XPIPE=2: retired in the same tile. XPIPE=1: pa(t+1) is a new value; ptxas must keep pa(t)'s registers until the wait (SASS: alternating R104/R100 and R96/R92 in the two unrolled bodies of mode 115) |
| o (36 f32) | P.V wgmmas (accumulate chain) | merge | in flight across the edge only in XPIPE=1; the offset rescale waits 0 before writing it; the merge follows a wait 0 |

Resources (CUDA 12.8.1, `tools/attn_one_sass.sh`, sm_90a; no shared memory added, the V^T buffers
were already two per KV head):

| build | mode 51 (two CTAs per SM, setmaxnreg 32 / 104) | mode 115 (one CTA per SM) |
|---|---|---|
| board (`ONE_XPIPE=0`) | 6328 SASS instr, consumer top R99, no spills | 6560 instr, R142 |
| `ONE_XPIPE=2` | 6920 instr, consumer top R100, no spills, no C75xx | 7264 instr, R139 |
| `ONE_XPIPE=1` | **C7512 "wgmma serialized due to insufficient register resources"** with setmaxnreg 112 (the most two CTAs per SM can give: 2 x (128 x 32 + 256 x 112) = 65,536); 16 WARPGROUP.DEPBAR (5 in XPIPE=2) | 7304 instr, 157 registers, no spills, no C75xx |

So XPIPE=1 is compile-infeasible in mode 51 (rejected before any judge run, per the brief) and
feasible only at one CTA per SM, which is A2's territory (mode 115). XPIPE=2 is the A1 candidate.

SASS evidence of real cross-tile issue overlap, mode 51 with `ONE_XPIPE=2` (steady-state body):
`QGMMA.64x72x32` x2 (P.V of tile t, last one `gsb0`) at f980 / f9a0, then without any
`WARPGROUP.DEPBAR` the stage-(t+1) `SYNCS.PHASECHK.TRANS64.TRYWAIT` loop (f9f0 - faa0), the
`FENCE.VIEW.ASYNC.S` and patch path, `BAR.SYNC 0x1` at it 0, `QGMMA.64x64x32` x2 (Q.K of tile t+1)
at fe00 / fe60, two `LDSM.16.MT88.4` + `STSM.16.M88.4` pairs (V^T of tile t+1), and only then
`WARPGROUP.DEPBAR.LE gsb0, 0x0` (fef0). Mode 115 with `ONE_XPIPE=1`: P.V issue (11e90 / 11ef0),
`DEPBAR.LE 0x1` (11f80, retires Q.K only), then the next tile's mask, max and 32 `MUFU.EX2` run
while the P.V is pending; the next `DEPBAR.LE 0x1` (145a0) follows the next Q.K issue.

With `ONE_XPIPE=0` all 13 instances are SASS-identical to 7703f0a (per-function compare,
namespace hashes normalized).

### Ordering argument (no fence or barrier removed; one wait moved)

Per tile, every path still executes: the full-barrier wait, `fence.proxy.async` after it, the
patch path with its own fence and the consumer bar.sync where the tile holds the block's rows,
`fence.proxy.async` + `bar.sync(2 + hk, 128)` between the V^T stores and the P.V issue, and one
`mbarrier.arrive` on the empty barrier per stage. Static counts rise (FENCE.VIEW.ASYNC 10 -> 13,
SYNCS.ARRIVE 3 -> 4) because the prologue and last tile are peeled copies.

1. Stage reuse (WAR between the Q.K wgmma's async-proxy reads, the generic ldmatrix of V, and the
   producer's next cp.async). `wgmma.wait_group N` returns when at most N of the most recent
   groups are pending and every earlier group of the executing thread is complete (PTX ISA,
   `wgmma.wait_group`; section number not re-read this session). XPIPE=2: wait 0 after Q.K(t+1). XPIPE=1: groups pending at
   the second wait are Q.K(t+1) then P.V(t) in commit order, so wait 1 completes Q.K(t+1). The
   ldmatrix of V(t+1) precedes the arrive in program order and its destination registers are
   consumed by the stmatrix before it. The release is the same arrive EREL used (judged in
   cb85d590c301410e); its producer side is unchanged.
2. V^T reuse (WAR between P.V(t)'s async-proxy reads of buffer t & 1 and the stmatrix of tile
   t+2 into it by any warp of the warpgroup). In XPIPE=2 each warp retires P.V(t) with the wait 0
   of tile t; in XPIPE=1 with the wait 1 that follows the Q.K(t+2) issue in tile t+1. Both come
   before that warp's arrival at the bar.sync of tile t+1, and the transpose of tile t+2 follows
   that bar.sync in every warp. This is the chain the board build already relies on (its
   transpose of tile t+2 under Q.K(t+2) follows the bar.sync of tile t+1, after the wait 0 that
   retires P.V(t)).
3. V^T publication (RAW, generic stmatrix to async-proxy wgmma read): unchanged,
   `fence.proxy.async.shared::cta` by every writing thread, then the bar.sync, then the P.V issue.
4. Registers: `wgmma.fence` precedes every Q.K and P.V issue; s is rewritten only by Q.K(t+1)
   after P(t) is packed; o is written by other instructions only after a wait 0 (rare offset move
   in XPIPE=1, the merge in both). ptxas reports no serialization for the builds kept.
5. Segment boundaries: the last tile is drained with wait 0 before the merge reds; the next
   segment's prologue transposes into the buffer the tile before last read, retired as in 2.

Independent read-only review (fresh agent, before the first judge run, 10-30): **XPIPE=2 sound**:
no stage, V^T or Q buffer can be overwritten while a wgmma reads it, including segment
boundaries and ntiles == 1 (ntiles == 0 cannot reach the loop: PROD implies FLAT and
`seg_bounds` requires lo < hi); release after the retiring wait; mbarrier order and phases as in
the EREL loop; no fence or barrier dropped; per-CTA o expected bitwise equal to the shipped loop.
XPIPE=1 (mode 115 only): sound with two caveats. (a) `pa` reuse while P.V reads it relies on
ptxas keeping the registers apart (a clean C75xx report is evidence for that build only; with
`ONE_UNR2=0` it would not hold). (b) The rare offset-move branch executes `wgmma.wait_group 0`
under a warp-uniform but not warpgroup-uniform condition; the `.aligned` wording, as the
reviewer reads it, makes that undefined behaviour, though `WARPGROUP.DEPBAR` is a per-warp wait
in SASS. If A2 adopts XPIPE=1, it needs a warpgroup-uniform condition or a recorded deviation.

Follow-up to caveat (b), XPIPE=1 only (XPIPE=0 and XPIPE=2 SASS unchanged, per-function compare):
the PCHK overflow vote is taken with `bar.red.or.pred` on the warpgroup's V^T barrier id
(2 + hk, 128 threads) instead of `__any_sync`, so `offset_to_max` runs under warpgroup-uniform
conditions only (tile 0, or that vote), and its `wgmma.wait_group 0` now sits at its start,
outside any divergent branch. This adds one barrier per tile after the first; none removed.
Two other placements were tried and rejected by ptxas: deferring the rescale to after the
retiring wait 1 (C7514, "non wgmma instructions reading accumulator registers", modes 51 and
115), and the same with the vote folded into the V^T bar.sync (C7514 again). Current build:
mode 115 6944 instructions, 154 registers, no spills, no C75xx; mode 51 still C7512 (rejected).

### A1 gate (written before the bench that tests it)

Score sensitivity from the 569b233dfe08414f judge medians: verify 16 C = 46 x 11.89 + 16 x 103.65 =
2205 us, verify 32 C = 46 x 16.42 + 16 x 187.67 = 3758 us. A 2% cut of t_full is 1.50% / 1.60% of
those loads' C, so +0.25% / +0.27% of the standalone score each, +0.5% for both.

Bench (one check): `tools/attn_make_bench_ab.py --null --shapes all --inputs 2` with arms `ctrl`,
`ctrl2` (independent build and state), `xp2@own` (`-DONE_XPIPE=2`), `pub`, `pub2`; seeded order
20261030; timing on held-out inputs (seeds 1000, 1001; ABCHECK on seed 1).

Promote `ONE_XPIPE=2` to a judged check only if all hold:
1. ABCHECK passes for every arm; `xp2` against `ctrl` differs only at the red-order level seen
   before (a few elements at one BF16 ulp per shape).
2. Cold, at verify 16 full and verify 32 full: `xp2 - ctrl` <= -2% of `ctrl` on the pooled row and
   on each input, and its magnitude exceeds `|ctrl2 - ctrl|` on that row; or ABSCORE for `xp2`
   >= 1.005 while `ctrl2`'s ABSCORE lies within half that gain of 1.
3. No other shape slower than `ctrl` by more than max(the shape's `|ctrl2 - ctrl|`, 1%).

Then a confirmation bench with another seed and a judged check of `submission.py` with the
switch on (dispatch: modes 51 and 115 only, the only builds it changes). If 2 fails while 1 and 3
hold: measured loss for the design as built, row in docs/ruled-out.md, and A2 (mode 115 with
`ONE_XPIPE=1`, the variant two CTAs per SM cannot hold) gets the next bench.

### Check 5575e7d15d3e49a8 (bench `bench_a0`, generator at 8a0a4a0, kernel at 86f2bec): A0 scorecard and A1 judged by its gate

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 5575e7d15d3e49a8 (check) | 86f2bec | bench (`--null --shapes all --inputs 2`, `xp2@own=-DONE_XPIPE=2`, `--pub` vjeux) | 2.454x (judged path = board code), 16/16 tests, ABCHECKSUM 19/0 every arm | A0 scorecard; A1 XPIPE=2 fails its gate (cold loss at verify 16 full) |

Profiles: `muse_attention_decode/profiles/bench_a0.5575e7d15d3e49a8.*`. Plans were identical across
`ctrl`, `ctrl2`, `xp2` on every shape (ABPLAN), and the public arms used their own plans (flat
264 at batch > 1 full, 252 x nc 8 at batch 1 full, per-request elsewhere, same split counts as
ours). Every arm passed ABCHECK; `xp2` against `ctrl` differed in 0 elements on the shapes shown
in the tail.

**Standalone scorecard (cold medians, pooled over the two held-out inputs, us; bench times
include about 4-5 us of replay cost the judge subtracts).** Sensitivity is the standalone score
change per us saved, from this run's judge medians (C_l: d1 485, d16 1816, d32 3214, v1 574,
v16 2189, v32 3722 us). "Value of the gap" = (ctrl2 - pub) x sensitivity: what matching the
public kernel on that row alone would add.

| shape | ctrl | ctrl2 | pub | pub2 | null ctrl2-ctrl / pub2-pub | ctrl2 - pub | sens %/us | value of the gap |
|---|---|---|---|---|---|---|---|---|
| d1 sliding | 9.76 | 9.60 | 9.06 | 8.99 | -0.16 / -0.06 | +0.54 | 1.580 | **+0.85%** |
| d1 full | 18.78 | 18.11 | 16.48 | 16.64 | -0.67 / +0.16 | +1.63 | 0.550 | **+0.90%** |
| d16 sliding | 14.34 | 14.53 | 12.77 | 13.12 | +0.19 / +0.35 | +1.76 | 0.422 | **+0.74%** |
| d16 full | 100.77 | 100.70 | 99.94 | 99.97 | -0.06 / +0.03 | +0.76 | 0.147 | +0.11% |
| d32 sliding | 16.70 | 16.80 | 15.81 | 15.97 | +0.10 / +0.16 | +0.99 | 0.239 | +0.24% |
| d32 full | 179.58 | 179.65 | 178.08 | 178.18 | +0.06 / +0.10 | +1.57 | 0.083 | +0.13% |
| v1 sliding | 11.14 | 11.14 | 10.88 | 10.98 | +0.00 / +0.10 | +0.26 | 1.336 | +0.35% |
| v1 full | 19.71 | 19.78 | 18.75 | 18.88 | +0.06 / +0.13 | +1.03 | 0.465 | **+0.48%** |
| v16 sliding | 15.84 | 15.84 | 15.46 | 15.62 | +0.00 / +0.16 | +0.38 | 0.350 | +0.13% |
| v16 full | 113.41 | 113.02 | 111.23 | 111.10 | -0.38 / -0.13 | +1.79 | 0.122 | +0.22% |
| v32 sliding | 20.93 | 20.99 | 19.49 | 19.74 | +0.06 / +0.26 | +1.50 | 0.206 | +0.31% |
| v32 full | 192.45 | 192.38 | 192.19 | 192.35 | -0.06 / +0.16 | +0.19 | 0.072 | +0.01% |

ABSCORE (cold, every arm against `ctrl`): `ctrl2` 1.0034 (the null), `pub` 1.0382, `pub2` 1.0342;
per load for `pub`: d1 1.102, d16 1.039, d32 1.018, v1 1.034, v16 1.021, v32 1.018.

What this measures:

- The public kernel's standalone lead over the board build is about 3.5% in this process, and it
  sits in the short and per-request shapes, not where A1 aims: by value of the gap, d1 full,
  d1 sliding and d16 sliding lead (+0.7-0.9% each), then v1 full (+0.5%). Verify 16 / 32 full,
  A1's target, are worth +0.22% / +0.01% to close. The two leading loads by value are decode 1
  and decode 16 (their per-load public ratios 1.10 and 1.04).
- The null has structure: `ctrl` (the judged path's own state, allocated first) is slower than
  `ctrl2` and `xp2` (two later, independently allocated states that agree with each other to
  0.06 us) on d1 full by 0.67-0.70 us on both inputs, and on d1 sliding by 0.03-0.16 us; at
  batch > 1 the three agree within 0.4 us. Hypothesis, not tested: the batch-1 plans' few hot
  workspace rows (mode 11: 8 copies x 16 rows; mode 13: 16 rows) land on an address-dependent
  set of L2 slices, so the first allocation's placement costs about 0.7 us at d1 full. Worth
  +0.37% of the score if real and controllable; the next bench tests placement directly.

**A1 (`ONE_XPIPE=2`) against its gate:** clause 1 holds (19/0, 0 differing elements in the tail).
Clause 2 fails: cold, `xp2 - ctrl` is +3.32 / +4.55 us at verify 16 full (input 0 / 1, +3.4%) and
+0.54 / +0.99 us at verify 32 full; ABSCORE v16 0.9755, v32 0.9968. Warm, the same arm is faster
(-1.8 / -1.9 us at verify 16 full, -3.5 / -3.3 at verify 32 full): the overlap works when L2 is
warm and loses when the tiles come from DRAM. Clause 3 holds (its other rows equal `ctrl2`,
which shares no state with `ctrl`; its d1 and d16 gains are the null's structure, not XPIPE,
whose SASS changes only modes 51 and 115). Verdict: **measured loss** for XPIPE=2 in mode 51 on the
judged (cold) quantity; not promoted, row in docs/ruled-out.md. Mechanism not measured: issuing
P.V(t) before the stage-(t+1) wait delays that wait and the Q.K behind it, which costs most when
the producer, not the consumer, sets the pace (cold).

### Next bench (check 2): A2 feasibility, workspace placement, batch-1 phase attribution

Arms (generator after this commit; `--null --shapes all --inputs 2 --seed 20261031`): `ctrl`,
`ctrl2`; A2: `s115` (board code forced to mode 115, one CTA per SM) and `s115xp1`
(`-DONE_XPIPE=1`, mode 115); placement: `ws64k`, `ws5m` (control module, own state, workspace at
+64 KB / +5 MB + 32 KB); attribution: one `-DONE_DIAG=1` build timed to the early exit after
phase 5 (prologue loads landed), 7 (query norm), 8 (new-token pass), 9 (V^T ones / patch), 10
(tile loop and reds), and whole (`diag0_probe`), on the per-request plans (every sliding shape and
batch-1 full).

Gates, written before the run:
- A2: develop the mode-115 family only if `s115xp1` beats `ctrl` cold by >= 2% on verify 16 or
  verify 32 full with both inputs agreeing and beyond `|ctrl2 - ctrl|`; `s115xp1 - s115` alone
  isolates the overlap at one CTA per SM. Otherwise measured loss, A2 parked.
- Placement: a real effect needs `ws64k` or `ws5m` to differ from `ctrl2` at d1 full by more than
  0.3 us on both inputs. If so, the next step is a production change that spreads the batch-1
  workspace copies (a layout change in `_one_state` / the row map), judged on its own.
- Attribution: no gate; it names the phase that holds the d1 / d16 gap to the public kernel.

### Check 318ae6fc3c3746ce (bench `bench_b`, generator and kernel at 682aeaa): A2, placement, phase attribution

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 318ae6fc3c3746ce (check) | 682aeaa | bench (`--null --shapes all --inputs 2 --seed 20261031`, 12 arms) | 2.479x (judged path = board code), 16/16 tests, ABCHECKSUM 19/0 every real arm | A2 measured loss; placement offset no effect; finalize is the largest batch-1 phase |

Profiles: `muse_attention_decode/profiles/bench_b.318ae6fc3c3746ce.*` (the log tail holds the
cold summaries; per-input rows were cut by the 16,000-character limit). Cold medians pooled over
two held-out inputs, us.

**A2 (mode 115, one CTA per SM) against its gate.** Verify 16 / 32 full: `ctrl` 110.27 / 187.26,
`s115` 137.79 / 244.93 (+25% / +31%), `s115xp1` 138.37 / 239.87; null `ctrl2 - ctrl` +0.06 /
+0.42. The Q.K-first overlap is worth 0 / -5.1 us inside mode 115 (2% at verify 32), far from
the 27-58 us the one-CTA-per-SM organization costs. Gate fails: **measured loss**, A2 parked
(row in docs/ruled-out.md). ABSCORE `s115` 0.945, `s115xp1` 0.944.

**Placement.** At d1 full: `ctrl` 18.62, `ctrl2` 18.18, `ws64k` 17.89, `ws5m` 18.08, and `s115`
(own state, normal plan there) 17.95. Every independently allocated state is 0.45-0.74 us faster
than the control's own (first-allocated) state, for the third time in two runs; the offset of
the workspace inside a fresh allocation changes nothing beyond 0.3 us. So the effect is tied to
the control's allocation, not to the low address bits. At batch > 1 all arms agree within 0.4 us.

**Phase attribution (one `-DONE_DIAG=1` build, own state; time to the early exit after phase k,
cold, us; `diag0` is the whole DIAG call, 0.1-0.6 us slower than `ctrl` at batch 1).**

| shape (plan) | p5 prologue loads landed | p7 query norm | p8 new-token pass | p9 patch / ones | p10 tile loop + reds | whole (`diag0`) | loop (p10 - p9) | finalize (whole - p10) | public total (5575e7d1) |
|---|---|---|---|---|---|---|---|---|---|
| d1 sliding (13, 32 CTAs) | 6.88 | 6.82 | 6.82 | 6.88 | 7.55 | 9.92 | 0.67 | **2.37** | 9.06 |
| d1 full (11, 252 x nc 8) | 10.08 | 10.05 | 10.08 | 10.08 | 14.78 | 18.82 | 4.70 | **4.04** | 16.48 |
| d16 sliding (1, 256 CTAs) | 10.11 | 10.11 | 10.14 | 10.18 | 11.20 | 14.82 | 1.02 | **3.62** | 12.77 |
| d32 sliding (1, 256 CTAs) | 10.34 | 10.34 | 10.30 | 10.37 | 14.05 | 16.80 | 3.68 | 2.75 | 15.81 |
| v1 sliding (15, 33 coop) | 6.98 | 7.07 | 7.39 | 7.39 | 8.64 | 11.17 | 1.25 | 2.53 | 10.88 |
| v1 full (11, 252 x nc 8) | 9.86 | 9.82 | 9.86 | 9.86 | 16.42 | 20.06 | 6.56 | **3.64** | 18.75 |
| v16 sliding (3, 128 coop) | 8.51 | 8.51 | 8.54 | 8.61 | 13.47 | 16.22 | 4.86 | 2.75 | 15.46 |
| v32 sliding (3, 128 coop) | 8.90 | 8.86 | 8.90 | 8.90 | 17.92 | 21.09 | 9.02 | 3.17 | 19.49 |

Reading (measured, with the replay cost of about 4.5 us inside every column): the query norm,
RoPE, new-token pass and patch cost nothing visible (p5 to p9 within 0.1 us, 0.4 at v1 sliding).
After the fixed launch and prologue, the **merge and finalize** (the reds' release, the ticket or
cooperative barrier, the last CTA's or every CTA's workspace reads, norm, gate and store) is
the largest phase on the four shapes that hold most of the gap's value: 2.4 us at d1 sliding,
4.0 at d1 full, 3.6 at d16 sliding, 3.6 at v1 full. The prologue at batch-1 full (252 CTAs,
cooperative) sits 3.2 us above batch-1 sliding (32 CTAs). The public kernel's whole call is 0.5-
1.6 us below ours on those shapes; whether its finalize or its prologue is the shorter is not
measured (its source shows the same last-CTA scheme with `__ldcg` reads; it has no phase exits).
This is the measured budget A3 asked for: reduction ownership is worth attacking on d1 / d16
sliding and batch-1 full, not on verify full.

Checks used this round: 5575e7d15d3e49a8, 318ae6fc3c3746ce.

### Next bench (check 3): does carving the workspace from one allocation remove the first-allocation cost?

`--set _ONE_WS_CARVE=1048576` (submission.py knob, default 0 so the judged file is unchanged):
every state's ws_acc, ws_l and ws_cnt come from one allocation, 1 MB in. Arms `ctrl` (first
allocated, carved), `ctrl2`, `ws64k`; ABADDR prints each state's addresses. Gate, written before
the run: the carve is a production candidate only if `ctrl2 - ctrl` at d1 full is within +-0.2 us
on both inputs (it was -0.67 and -0.45 in the two runs before) and no shape regresses beyond the
null; then a judged pair of submission.py with the carve on.

### Check a92aa9c43f064c49 (bench `bench_c`, at df4060d): the workspace carve does not remove the control's lag

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | a92aa9c43f064c49 (check) | df4060d | bench (`--set _ONE_WS_CARVE=1048576`, arms ctrl, ctrl2, ws64k; the judged path also ran carved) | 2.429x, 16/16 tests, ABCHECKSUM 19/0 | carve gate fails |

Profiles: `muse_attention_decode/profiles/bench_c.a92aa9c43f064c49.*`. d1 full cold: `ctrl` 18.62 /
18.56 (inputs 0 / 1), `ctrl2` 18.08 / 18.02, `ws64k` 17.98 / 17.98: the control is still 0.54 us
behind with every state carved, so the gate (within +-0.2 us) fails; the carve stays a bench knob
at 0. ABADDR: `ctrl` acc 0x82478100000, `ctrl2` 0x82479260000, `ws64k` 0x82474010000: the
fast `ws64k` sits below the control's address and `ctrl2` right above it, so neither address
order nor allocation shape explains it. The judged score of this run (2.429x, d1 full 14.16 us
judged) is inside the earlier runs' spread. What is left: in all three benches the control is
also the arm whose cold graph is captured first for every shape. Next bench tests that
directly.

### Next bench (check 4): capture order

Generator change: per (shape, input) the cold graphs are captured in a seeded random order and
ABCOLD prints the first-captured arm (`cap0`). `--shapes b1 --inputs 6`, arms `ctrl`, `ctrl2`,
`ws64k`, production allocation. Reading, written before the run: if the slow arm at d1 full
follows `cap0` rather than `ctrl`, the lag is a bench artifact of capture order (the earlier
nulls `ctrl2 - ctrl` at d1 full are then capture-order bias, and the A0 ranking should be read
from `ctrl2`); if it stays with `ctrl`, it belongs to the control's state and a production
placement fix stays worth pursuing.

| 2026-10-30 | 7144310358564fa3 (check) | a0a78ce (generator after the --shapes b1 edit, uncommitted then) | bench `bench_d` (capture order, `--shapes b1 --inputs 6`) | 2.469x judged path, 16/16 tests, ABCHECKSUM 11/0; **no bench data** | generator bug: `_AB_CAP` never defined (a scripted edit did not apply), every ABBENCH row NameError. Caught afterwards by `pyflakes` on the generated file, which now runs before every bench check. Profiles `bench_d.7144310358564fa3.*` |

Repeat as `bench_d2` (seed 20261034, generator fixed, pyflakes clean).

### Check b7762d764d474eae (bench `bench_d2`, at 46df563): the d1 full null is workspace placement, not capture order

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | b7762d764d474eae (check) | 46df563 | bench `bench_d2` (`--null --shapes b1 --inputs 6 --seed 20261034`, ctrl, ctrl2, ws64k) | 2.439x judged path, 16/16 tests, ABCHECKSUM 11/0 | lag follows the state, not the capture order |

Profiles: `muse_attention_decode/profiles/bench_d2.b7762d764d474eae.*`. d1 full cold over six
held-out inputs: `ctrl` 18.62-18.78, `ctrl2` 18.66-18.85, `ws64k` 17.95-18.14, on every input,
whichever arm was captured first (`cap0` was `ctrl` on five inputs, `ctrl2` on one). Other
batch-1 shapes: all three within 0.2 us. ABADDR: `ctrl` acc 0x...108000000, `ctrl2`
0x...105010000 (slow), `ws64k` 0x...104010000 (fast). With bench c (`ctrl` 0x...478100000 slow,
`ctrl2` 0x...479260000 fast, `ws64k` 0x...474010000 fast):

- The 0.5-0.7 us at d1 full belongs to where a state's workspace lands: two states with the same
  address bits below 2 MB (`ctrl2` and `ws64k` here, both ...010000) differ, so it depends on
  physical placement above the 2 MB page offset, which a process cannot choose by offsetting.
  Measured in 4 of 4 benches; mechanism (hypothesis): the batch-1 full plan's 252 CTAs send
  their reds and the cooperative counters to 128 workspace rows (32 KB, nc 8) plus their row
  sums, and the L2 slice / partition those lines hash to sets the merge latency.
- Capture order and allocation order are ruled out as the cause; the earlier "ctrl slower"
  pattern was placement luck in the control's allocation three times.
- Consequence for every bench: at d1 full a duplicate null is +-0.7 us by placement alone, so a
  d1 full difference under 0.7 us between arms with different states is not evidence. Fixed-
  state comparisons (variants sharing `ctrl`'s state, the old default) remove this term.
- A possible production use, not built: at the first call (outside capture), time the mode 11
  batch-1 plan on two or three candidate workspace allocations and keep the fastest. Expected
  value about +0.18% of the score (0.67 us x 0.55%/us when the first allocation is slow, about
  half the time in these four runs); placement chosen by a timing at prepare is not
  input-dependent. Lower priority than A3.

## Standalone round 10-30: family ledger and where the score is

Checks used by this round: 5575e7d15d3e49a8, 318ae6fc3c3746ce, a92aa9c43f064c49,
7144310358564fa3 (no data, generator bug), b7762d764d474eae. Quota after them (`okc problems`):
14 of 20 attention checks, 10 of 10 runs. No file was submitted; no file is offered for
submission (nothing beat the board build).

| family | state | evidence | reopening trigger |
|---|---|---|---|
| A1 cross-tile overlap, P.V first, mode 51 (`ONE_XPIPE=2`) | measured loss (cold) | 5575e7d15d3e49a8: +3.4% at verify 16 full cold, warm -1.8% | the producer stops setting the cold pace, or a P.V-first order that keeps the stage wait ahead of the P.V issue |
| A1 cross-tile overlap, Q.K first, mode 51 (`ONE_XPIPE=1`) | compile-infeasible | C7512 at setmaxnreg 112 (the two-CTA ceiling) | consumer live state cut by about 8 registers |
| A2 one CTA per SM (mode 115), with and without XPIPE=1 | measured loss | 318ae6fc3c3746ce: +25% / +31% at verify 16 / 32 full; overlap worth 0 / -2% inside it | a resource model that pays back the halved residency |
| Workspace carve from one allocation (`_ONE_WS_CARVE`) | measured loss (no effect) | a92aa9c43f064c49 | none; the cause is physical placement |
| Prepare-time workspace placement choice at batch-1 full | untested | b7762d764d474eae: 0.5-0.7 us placement spread, 4 of 4 benches | build when A3 is done or as a cheap side bet |
| A3 reduction ownership on short shapes | untested, budget measured | 318ae6fc3c3746ce phases: merge + finalize 2.4 us (d1 sliding), 4.0 (d1 full), 3.6 (d16 sliding), 3.6 (v1 full) | next round's first target |
| Standalone wins / layer-validated wins | none this round | | |

Where the standalone score is (A0, bench 5575e7d15d3e49a8, value of matching the public kernel
per row): d1 full +0.9%, d1 sliding +0.85%, d16 sliding +0.74%, v1 full +0.48%, v1 sliding
+0.35%, v32 sliding +0.31%, d32 sliding +0.24%, v16 full +0.22%, everything else under +0.15%.
The d1 full row's 1.6 us gap includes up to 0.7 us of placement noise on either side. The short
and sliding shapes, where the phase exits put 2.4-4 us in the merge and finalize after a
6.9-10 us launch-plus-prologue (both including the bench's replay cost), carry about 3% of the
3.5% total; verify 16 / 32 full, A1's target, carry 0.23%.

Next round, in order:
1. A3, scoped by these numbers: on d1 / d16 sliding and batch-1 full, measure what the public
   kernel spends after its loop (a stamp copy of its per-request path, as `pubstamp` did for the
   flat kernel), then compare our ticket + last-CTA finalize against (a) a cluster / DSMEM merge
   of the splits that share a request (no L2 round trip for the partials), (b) a finalize that
   overlaps the reds' release with the gate and norm-weight loads it already prefetches. Gate:
   0.5 us at d1 sliding (+0.8% score) or 1 us at d16 sliding / d1 full, null-checked with fixed
   state.
2. Prologue at batch-1 full: 10.1 us to p5 against 6.9 at batch-1 sliding (252 cooperative CTAs
   against 32); compare the launch with fewer CTAs per request (the public plan uses the same
   252 x nc 8, so this needs its own measurement first).
3. The placement side bet above, if quota is spare.

## Standalone round 2 (10-30): A3, merge / finalize and the batch-1 full prologue

Brief: A3 on the per-request paths that hold most of the measured gap (d1 sliding mode 13, d16
sliding mode 1, d1 full mode 11 at q_len 1, v1 full mode 11 at q_len 8), compared phase by phase
with a timed copy of the public per-request path. Start: main b200eda. Quota at the start
(`okc problems`): 14 of 20 attention checks; this round uses at most 5, two kept for
confirmation.

### What the two per-request paths do after the tile loop (source read, 10-30)

Public file sha1 0e9d4c78 (the `pub` arm of every bench since d57d1fe064164611), plans read from
its `_plan` / `_TUNED`: d1 sliding 32 splits nc 1 last-CTA (our mode 13, same grid), d16 sliding
16 splits x 16 requests last-CTA (our mode 1, same grid), d1 full 252 x nc 8 cooperative finalize
(`_TUNED`), v1 full 252 x nc 8 cooperative finalize. Same splits and copies as ours on all four.
Differences in the merge and finalize:

| step | ours (attn_one.cu, b200eda) | public per-request path |
|---|---|---|
| q_len 1 reds | each of the 4 warps (2 KV heads x 2 key halves) reds its own partial: 64 red.v4 + 16 red.f32 per half per CTA, so every workspace address takes 2 reds per split | the second key half hands (m, l, o) to the first through shared memory, which merges them and alone issues the reds: half the reds per CTA and per address |
| q_len 8 reds | 8 red.v4 + 1 red.f32 per thread pair, 2048 red.v4 per CTA | 16-byte reds after a lane-pair swap, 2048 per CTA (same count) |
| ticket (last-CTA plans) | CTA barrier, thread 0 `atom.acq_rel.gpu`, barrier, last CTA finalizes | the same |
| cooperative wait (batch-1 full) | `cudaLaunchAttributeCooperative`; CTA barrier, thread 0 `atom.acq_rel`, then every CTA, the last arriver included, polls `ld.acquire` with 64 ns sleeps | plain `<<<>>>` launch after an occupancy check (no cooperative attribute); the last arriver skips the poll |
| finalize | gate load, copies loaded (FIN8: all 8 before any store), norm, gate, store, zero | the same structure (`sum_copies` nc 8 path), `__ldcg` reads |

Hypotheses, in order of their measured budget: (H1) the doubled q_len 1 reds load the L2 atomic
units at d1 sliding, d16 / d32 sliding and d1 full; (H2) at batch-1 full the cooperative launch
attribute and the grid poll cost time the public plain launch does not pay (bench 1e6626a427454f9a
measured -1.8 / -1.4 us cold for mode 51 launched without the attribute); (H3) a last-CTA ticket
in place of the cooperative finalize at batch-1 full and the q_len 8 sliding plans keeps the
progress guarantee without any co-residency requirement and removes the poll.

### Switches (all default 0; with all off the 13 instances are SASS-identical to b200eda)

- `ONE_HMRG` (modes 1, 3, 11, 13 at q_len 1): the key-half handoff of the public kernel, keeping
  our fixed frame. Each half still scales its partial by its own 2^m (m floored and capped as
  before); the second half stores o * 2^m and l * 2^m to shared memory (stage 0, after every
  thread's `cp_wait<0>` and a CTA barrier), a second barrier, and the first half adds them to
  its own scaled values and issues the reds. Rounding: one FP32 add of the two halves before the
  red instead of in L2, an order the reds never fixed; the output rounding is unchanged. SASS
  (CUDA 12.8.1, `tools/attn_one_sass.sh`): mode 13 1544 -> 1608 instructions, 2 more BAR, the
  same 4 REDG.E.ADD.F32 (v4) + 1 scalar REDG + 1 ATOMG + 1 MEMBAR.ALL.GPU per thread statically
  (executed by half the warps); no spills (mode 1 108 registers, 11 115, 13 72).
- `ONE_T9` (bench builds): adds instances q_len 1 mode 9 and q_len 8 modes 9 and 13 (the last-CTA
  versions of the cooperative plans), and the FIN8 finalize applies to every speculated-length
  multi-tile instance (it was cooperative-only; no existing instance changes, per-function SASS
  compare). Bench arm option `:tkt` runs 3 -> 1, 11 -> 9, 15 -> 13 where the control's plan is
  cooperative and per-request.
- `ONE_DIAG` exits 11 (after the merge reds) and 12 (after the ticket or grid barrier and its CTA
  barrier, before the finalize), next to the existing 9 (before the tile loop) and 10 (after it).

Ordering (no fence or barrier removed): HMRG adds a `cp_wait<0>` and two CTA barriers. The
shared-memory handoff is generic stores, `bar.sync`, generic loads in one CTA (bar.sync orders
shared memory among the participating threads). Stage 0 is reused only after the barrier that
follows every warp's last read of the stages and every thread's `cp.async.wait_group 0`; no
async-proxy reader touches the stages at q_len 1 (mma.sync). The reds are still issued before
the CTA barrier that precedes thread 0's `atom.acq_rel` (ticket) or the cooperative arrive, so
the release chain the judged modes use is unchanged. T9's mode 9 / 13 use the existing ticket
(judged in modes 1 and 13) and the FIN8 finalize (judged in mode 11, b0b146dfbb4841b2): each
finalizing thread loads an address before it zeroes it and no other thread touches it after the
ticket's acquire. It removes the cooperative barrier instead of reordering it.

Public phase copy (scratchpad `a3/make_pubph.py`, never committed or submitted): the public
source with a `Params::phx` field and exits on the per-request path only: 1 before its consumer
state (prologue copies drained), 2 before the merge, 3 after the reds, 4 after the ticket or
cooperative wait and its CTA barrier. One build shared by arms `pubph0_probe` (no exit, the
instrumentation's own cost) to `pubph4_probe`.

### Gate for check 1 (written before the bench that tests it)

Bench `bench_e` (`tools/attn_make_bench_ab.py`, scratchpad `a3/gen_e.sh`; `--null --shapes all
--inputs 2 --seed 20261035 --nocoop`): arms `ctrl`, `ctrl2`, `pub`, `pub2`, `hmrg`
(`-DONE_HMRG=1,-DONE_T9=1`, shares ctrl's state), `hmtk` (same build with `:tkt`, shares ctrl's
state), `nocoop` (ctrl's module without the cooperative attribute; a diagnostic, it drops the
co-residency guarantee and is not a candidate), our DIAG exits `p9_probe` .. `p12_probe` and
`diag0_probe`, the public exits `pubph0_probe` .. `pubph4_probe`.

A candidate (`hmrg`, or `hmtk` as the coherent pair) goes to a confirmation bench only if all hold:
1. ABCHECK passes for every real arm; against ctrl it differs only at the red-order level (a few
   elements at one BF16 ulp) and leaves the workspace zero.
2. Cold ABSCORE against ctrl >= 1.008 (0.5 us at d1 sliding is +0.79%), and its excess over 1 is
   larger than |ABSCORE(ctrl2) - 1| + 0.004 (ctrl2 carries the 0.5-0.7 us d1 full placement
   term; the candidates share ctrl's state and do not).
3. On every targeted row (d1 sliding, d1 full, d16 sliding, v1 full) the change has the same sign
   on both held-out inputs.
4. No row slower than ctrl by more than max(that row's |ctrl2 - ctrl|, 1%).

Attribution (no gate): per phase, ours (p9 / p10 / p11 / p12 / whole) against the public
(pubph1 / 2 / 3 / 4 / pub), cold and warm, on the four targeted rows; the largest difference
names the mechanism. `nocoop - ctrl` sizes H2 at batch-1 full and the q_len 8 sliding plans.

### Check ce92d32e26024bd2 (bench `bench_e`, at d8d43f3): the q_len 1 reds are the mechanism; HMRG passes its gate

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | ce92d32e26024bd2 (check) | d8d43f3 | bench `bench_e` (`a3/gen_e.sh`, 17 arms) | 2.432x (CI 2.395-2.468) judged path = board code, 16/16 tests, ABCHECKSUM 19/0 every real arm | HMRG ABSCORE 1.0180 (null 0.9993); q_len 8 ticket a measured loss; cooperative attribute worth <= 0.4 us |

Profiles: `muse_attention_decode/profiles/bench_e.ce92d32e26024bd2.*`. The log tail kept the
pooled summaries only (per-input rows and ABCHECK details of the q_len 1 shapes were cut by the
16,000-character limit; the generator now prints compact ABIN lines at the end).

**Phase by phase, cold medians pooled over two held-out inputs (us; every column includes the
bench's replay cost; the exits time grid completion, so a red's drain lands in the phase that
issued it).** Ours = `p9` / `p10` / `p11` / `p12` / `diag0` (own state, DIAG build); public =
`pubph1` / `pubph2` / `pubph3` / `pubph4` / `pubph0` (one instrumented build).

| shape | arm | prologue (to loop) | loop | reds | ticket or barrier | finalize | whole |
|---|---|---|---|---|---|---|---|
| d1 sliding (13) | ours | 6.66 | 0.57 | **0.87** | 0.76 | 0.74 | 9.60 |
| | public | 6.46 | 0.64 | **0.45** | 0.58 | 0.64 | 8.77 |
| d1 full (11, nc 8) | ours | 9.92 | 4.80 | **1.95** | 0.64 | 0.61 | 17.92 |
| | public | 9.50 | 4.52 | **0.38** | 1.38 | 0.64 | 16.42 |
| d16 sliding (1) | ours | 10.37 | 0.86 | **1.76** | 0.55 | 1.40 | 14.94 |
| | public | 10.08 | 0.77 | **0.45** | 1.12 | 0.38 | 12.80 |
| d32 sliding (1) | ours | 10.43 | 3.68 | 1.06 | 0.93 | 0.86 | 16.96 |
| | public | 10.24 | 3.55 | 0.45 | 0.61 | 0.86 | 15.71 |
| v1 full (11, nc 8) | ours | 9.82 | **6.72** | 1.51 | 1.12 | 0.89 | 20.06 |
| | public | 9.79 | **5.79** | 1.35 | 1.09 | 0.89 | 18.91 |
| v1 sliding (15) | ours | 7.49 | 1.21 | 0.61 | 1.25 | 0.77 | 11.33 |
| | public | 7.01 | 1.12 | 0.96 | 0.93 | 0.89 | 10.91 |

What it names:

- **q_len 1 (d1 sliding, d1 full, d16 / d32 sliding): the reds.** Up to the end of the loop the
  two kernels agree within 0.2-0.7 us; from the loop's end to the reds' completion ours takes
  0.87 / 1.95 / 1.76 / 1.06 us against the public 0.45 / 0.38 / 0.45 / 0.45. The public kernel
  issues half as many reds (key-half handoff). The ticket and finalize phases move in both
  directions by a few tenths and sum to less (where our reds drain late, our ticket waits less).
- **Batch-1 full prologue is not a gap.** To the loop: 9.92 vs 9.50 at d1 full, 9.82 vs 9.79 at
  v1 full. Both kernels pay about 3 us more than batch-1 sliding: the first tiles of 252 CTAs
  from DRAM. Roadmap item 2 (the batch-1 full prologue) is closed by this measurement.
- **v1 full: the tile loop** (6.72 vs 5.79 us); merge, barrier and finalize agree within 0.2 us.
  Not an A3 target.
- v1 sliding: prologue 0.5 and barrier 0.3 us behind, reds 0.35 ahead; gap 0.26 us in total.

**Candidates against the check-1 gate** (cold pooled, us, arm - ctrl; null ctrl2 - ctrl in
brackets; pub - ctrl for scale):

| row | hmrg | hmtk | nocoop | null | pub |
|---|---|---|---|---|---|
| d1 sliding | **-0.51** | -0.45 | -0.03 | +0.10 | -0.67 |
| d1 full | **-1.38** | -1.60 | -0.38 | -0.64 | -2.11 |
| d16 sliding | **-1.44** | -1.38 | -0.13 | -0.26 | -1.47 |
| d32 sliding | -0.80 | -0.83 | -0.03 | +0.13 | -0.74 |
| v1 sliding | -0.10 | **+1.79** | -0.13 | +0.19 | -0.26 |
| v1 full | +0.10 | **+8.67** | -0.10 | +0.13 | -0.77 |
| v16 sliding | -0.06 | +1.70 | +0.03 | +0.03 | -0.45 |
| v32 sliding | -0.06 | +1.41 | +0.00 | -0.06 | -1.41 |
| v32 full | +0.51 | +0.26 | +0.35 | +0.80 | -0.51 |
| ABSCORE | **1.0180** | 0.9702 | 1.0033 | 0.9993 | 1.0359 |

`hmrg` per load: d1 1.0664, d16 1.0307, d32 1.0105, v1 1.0035, v16 0.9999, v32 0.9987 (q_len 8
SASS is identical to ctrl, so the v rows are the null's spread). Gate: 1 holds (19/0, workspace
check inside PASS; the per-element detail was cut from the tail); 2 holds (1.0180 >= 1.008, excess
0.018 > |0.9993 - 1| + 0.004); 3 not verifiable from this log (per-input rows cut; the v1 full row
has no code change); 4 holds (largest regression +0.51 at v32 full, under its 1% bound, code
identical). **HMRG goes to the confirmation bench.** It closes the public lead on d16 sliding
(12.99 vs 12.96) and d32 sliding, and three quarters of it at d1 sliding; at d1 full, hmrg
shares ctrl's state, whose placement has been the slow one (ctrl2 -0.64).

`hmtk` (HMRG plus the last-CTA finalize on the cooperative plans): at q_len 1 batch-1 full it adds
-0.22 us over hmrg; at q_len 8 the one-CTA finalize of 128 rows costs +8.7 us at v1 full and +1.4
to +1.8 us on the sliding plans. **Measured loss at q_len 8**; at d1 full its 0.22 us is under the
null. `nocoop`: -0.38 us at d1 full and within 0.13 us elsewhere; the cooperative attribute is
not where the batch-1 gap is, and it stays (it carries the co-residency guarantee).

### Gate for check 2 (confirmation, written before the bench)

Bench `bench_f` (`--pub <0e9d4c78> --null --shapes all --inputs 2 --seed 20261036`, arms `ctrl`,
`ctrl2`, `hmrg` (`-DONE_HMRG=1`, ctrl's state), `hmrgown` (same build, own state), `pub`,
`pub2`; ABIN lines carry per-input differences). HMRG is confirmed if clauses 1, 2 and 4 of the
check-1 gate hold for `hmrg` again, clause 3 holds on d1 sliding, d1 full and d16 sliding from the
ABIN lines (v1 full has no code change), and `hmrgown`'s ABSCORE is above 1.008 as well, so the
gain does not depend on the control's workspace. Then a judged check of submission.py with
`ONE_HMRG` on.

### Check d0819bb26907426b (bench `bench_f`, at 9eafafc): HMRG confirmed

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | d0819bb26907426b (check) | 9eafafc | bench `bench_f` (`--null --shapes all --inputs 2 --seed 20261036`, ctrl, ctrl2, hmrg, hmrgown, pub, pub2) | 2.452x (CI 2.415-2.504) judged path = board code, 16/16 tests, ABCHECKSUM 19/0 every arm | HMRG ABSCORE 1.0214 (own state 1.0232), null 1.0037 |

Profiles: `muse_attention_decode/profiles/bench_f.d0819bb26907426b.*`. Cold, arm - ctrl (us),
per held-out input from ABIN:

| row | hmrg in 0 / 1 | hmrgown in 0 / 1 | ctrl2 in 0 / 1 | pub in 0 / 1 |
|---|---|---|---|---|
| d1 sliding | -0.54 / -0.70 | -0.67 / -0.64 | -0.03 / -0.03 | -0.67 / -0.70 |
| d1 full | -1.50 / -1.50 | -1.82 / -1.79 | -0.74 / -0.70 | -2.14 / -2.14 |
| d16 sliding | -1.47 / -1.41 | -1.44 / -1.47 | +0.22 / +0.26 | -1.44 / -1.57 |
| d32 sliding | -0.86 / -0.86 | -1.09 / -0.90 | +0.19 / +0.03 | -0.80 / -0.77 |
| v1 full (no code change) | -0.03 / +0.13 | -0.06 / +0.13 | -0.03 / +0.16 | -0.86 / -0.74 |

ABSCORE per load, hmrg: d1 1.0772, d16 1.0306, d32 1.0115, v1 1.0059, v16 1.0031, v32 1.0020.
Gate: 1 holds (19/0); 2 holds (1.0214, excess 0.0214 > 0.0037 + 0.004); 3 holds (same sign on both
inputs on d1 sliding, d1 full, d16 sliding); 4 holds (no pooled row slower than +0.06 us;
per-input worst +0.26 at v16 full, q_len 8 code identical); `hmrgown` 1.0232 >= 1.008.
**Confirmed.** On the q_len 1 per-request rows HMRG matches the public kernel (d1 sliding 9.25
vs 9.18, d16 sliding 12.90 vs 12.86, d32 sliding 15.78 vs 15.84); at d1 full 0.6 us remain,
inside the placement spread (hmrgown 16.90 vs pub 16.51).

Independent read-only review of the HMRG ordering (fresh agent, 10-30, before the judged check):
**sound.** Stage 0 holds no pending copy (the q_len 1 per-request loop issues copies only for
tiles below ntiles, and every thread's `cp.async.wait_group 0` plus the barrier precede the
stores), both barriers sit outside the `wsub` branches and every earlier exit is CTA-uniform, the
reds still precede the CTA barrier ahead of thread 0's `atom.acq_rel` (ticket) or the
cooperative arrive, the handoff's 4,352 bytes lie inside the stage-0 K tile and nothing after
the merge reads it, and the red layout and per-half 2^m frame match the old path. Not verified
by the reviewer: SASS and runtime behaviour (covered by the benches and the SASS compares here).

Production: `ONE_HMRG` defaults to 1 in attn_one.cu and submission.py. SASS (CUDA 12.8.1, all 13
instances): the default build equals the benched `hmrg` build; `-DONE_HMRG=0` equals b200eda;
only q_len 1 modes 1, 3, 11, 13 change; no spills. Next: a judged check of submission.py.

### Judged: check 5d309f9d13924dca 2.512x (CI 2.475-2.550), repeat 82b1bf0fad3244b9 2.582x (CI 2.543-2.620)

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 5d309f9d13924dca (check) | dc6b409 | submission.py (`ONE_HMRG` on; sha256 93297101) | 2.512x (CI 2.475-2.550), 16/16 tests | judged d1 sliding 5.3 us, d1 full 12.7, d16 sliding 8.7, d32 sliding 10.0 |
| 2026-10-30 | 82b1bf0fad3244b9 (check) | dc6b409 + one comment line | repeat of the same file | 2.582x (CI 2.543-2.620), 16/16 tests | 5.1 / 12.2 / 8.2 / 10.3 us; full q_len 8 rows also faster with no code change there (machine speed) |

Profiles: `muse_attention_decode/profiles/hmrg.5d309f9d13924dca.*`, `hmrg_r.82b1bf0fad3244b9.*`.
For scale, the judged path of the same day's benches (board code, no HMRG): ce92d32e26024bd2
2.432x with d1 sliding 6.0, d1 full 14.2, d16 sliding 9.6, d32 sliding 11.3 us; d0819bb26907426b
2.452x. Board entry e9085c86d89449c2 2.504x (a bench file); clean candidate-only best
569b233dfe08414f 2.449x. Both HMRG checks are above both. Not submitted: handed to the
coordinator (`main`) with the file path and these job ids.

## Standalone round 2 (10-30): family ledger update

Checks used by this round: ce92d32e26024bd2, d0819bb26907426b, 5d309f9d13924dca,
82b1bf0fad3244b9. Quota after them (`okc problems`): 10 of 20 attention checks, 10 of 10 runs.

| family | state | evidence | reopening trigger |
|---|---|---|---|
| A3 q_len 1 key-half merge before the reds (`ONE_HMRG`, modes 1, 3, 11, 13) | **measured standalone win**, on by default | benches ce92d32e26024bd2 (1.0180) and d0819bb26907426b (1.0214, own state 1.0232, null 1.0037); judged 2.512x / 2.582x; review sound | layer port is a separate gate (`tools/dl_one_inline.py`) |
| A3 last-CTA ticket on the q_len 8 cooperative plans (`ONE_T9` + `:tkt`) | measured loss | ce92d32e26024bd2: +8.7 us at v1 full, +1.4 to +1.8 on q_len 8 sliding | see docs/ruled-out.md |
| A3 ticket at q_len 1 batch-1 full | untested beyond 0.22 us under the null | ce92d32e26024bd2 | a fixed-state repeat above 0.3 us |
| Cooperative launch attribute (H2) | measured, not a gap | ce92d32e26024bd2: -0.38 us at d1 full, ~0 elsewhere | none; it keeps the co-residency guarantee |
| Batch-1 full prologue | closed as A3 work | phase exits: equal to the public kernel within 0.4 us | a plan that changes the CTA count for both |
| A3 cluster / DSMEM merge, overlapped finalize (round-1 ideas a, b) | not built | after HMRG the q_len 1 rows sit within 0.1 us of the public kernel except d1 full (0.6, inside placement) | a phase exit showing merge + finalize above the public's again |
| Prepare-time workspace placement choice at batch-1 full | untested | b7762d764d474eae; with HMRG d1 full is 0.4-0.6 us behind the public, the same size as the placement spread | build when quota allows; modeled about +0.2% |

Where the remaining gap is (bench d0819bb26907426b, pub - hmrg, cold): v32 sliding -1.38 us, v16
full -2.0, d32 full -1.4, d16 full -0.9, v1 full -0.86 (phase exits put it in the tile loop),
d1 full -0.64, v16 sliding -0.32, v1 sliding -0.10, d1 sliding -0.07, d16 sliding -0.04.

### Scored: 57b0bed35f694a23 2.479x (2026-10-30)

Submit of submission.py at dc6b409 (ONE_HMRG), submitted by the coordinator under the standing authorization. Passed. Below the board entry e9085c86d89449c2 (2.504x, a bench file whose judged path is the PK16/VHOIST candidate), which stays. Best clean candidate-only scored run so far (previous 569b233dfe08414f at 2.449x).
## Standalone round 3 (10-30): the per-request q_len 8 tile loop (`ONE_PRQ`)

Brief: re-rank what is left with HMRG on (compact scorecard, independent duplicates for both
arms), then attribute the top remaining gap inside the tile loop and build one coherent change
as a switch. Start: main 5c11a11 (ONE_HMRG on). Quota at the start (`okc problems`): 10 of 20
attention checks; this round uses at most 5, two kept for confirmation.

### Where verify 1 full's tile loop differs from the public one (source and SASS, 10-30)

Verify 1 full runs our mode 11 (q_len 8, per-request, cooperative, speculated length; 252 x nc 8;
1251 tiles, so 5 tiles per CTA) and the public `attn_kernel<8, false, 11>` (file sha1 0e9d4c78,
same grid and copies). Both have 3 stages, 256 consumer threads, two CTAs per SM, the same
wgmma shapes (`QGMMA.64x64x32` x2 for Q.K, `QGMMA.64x72x32` x2 for P.V with ones rows) and the
same lazy offset. SASS (CUDA 12.8.1, `tools/attn_one_sass.sh`, the public build with its own
flags; public copy only in .build/ and the scratchpad): ours 109 registers, 2504 instructions,
tile body 0x46c0-0x6800 (about 530 instructions); public 119 registers, 2752 instructions,
tile body 0x4e40-0x7690 (about 645). The public tile is longer, so the gap is the order of the
work on the per-tile critical path, not its amount:

| step | ours (judged) | public per-request |
|---|---|---|
| after `cp.async.wait_group 1` + CTA barrier | copies of tile i + 2 first: the in-loop `issue_kv` with a three-way branch per row and 64-bit offsets (about 85 SASS instructions before the Q.K issue) | Q.K issued first; then the copies, branch-free with 32-bit offsets when the tile is fully cached (4 `LDGSTS` in a straight line) |
| V^T transpose (2 `LDSM.MT88.4` + 2 `STSM.M88.4`) | after the Q.K wait and the softmax, on the critical path | before the Q.K wait, under the in-flight Q.K |
| offset test | quad max (2 x 2 shuffles) every tile, then the vote | each thread's own partial max against the offset, vote; quad reduction only inside a move |
| P.V | both halves packed, then one group of two wgmmas | half 0 packed and issued, half 1's exponentials under it, then half 1 |

The old QKF experiment (mode bit 128, scored run bf6b8ca9e37341fb, 10-28) tested Q.K-first plus
the transpose under Q.K in the kernel generation of that day (before PK16, DIET, DCARRY and HMRG)
and lost on verify 16 / 32 full (flat mode 19) and by 0.4-0.5 us on verify 16 / 32 sliding; verify 1
full was not reported. New evidence for reopening it on the per-request plans: the phase exits of
ce92d32e26024bd2 put verify 1 full's remaining gap inside the tile loop (6.72 vs 5.79 us, merge
and finalize equal), and the public kernel runs this order there. The switch below separates the
steps so a loss of one step is attributed to that step.

### The switch: `ONE_PRQ` (bit mask, default 0)

Applies to the per-request q_len 8 instances (1, 3, 11, 15; 9 and 13 under ONE_T9); the flat and
producer builds and every q_len 1 instance are untouched. Bits: 1 TOV (V^T under the in-flight
Q.K), 2 QKF (Q.K issued before the next tile's copies), 4 FAST (in-loop copy of a fully cached
tile branch-free with 32-bit offsets when the host's `off32` holds, as PFAST in the producer),
8 SPV (P.V per 32-key half, two commit groups), 16 TCHK (offset test on each thread's partial
max; the quad reduction and the move run only when the vote fires). It is a separate branch
ahead of the judged loop, so with 0 the judged loop's source is unchanged.

SASS (CUDA 12.8.1, all 13 instances, per-function compare with namespace hashes normalized):
`ONE_PRQ=0` is identical to main 5c11a11 on every instance. Each of 1, 2, 4, 8, 16 and 31 changes
only q_len 8 modes 1, 3, 11, 15; no spills, no C75xx warnings; mode 11 2504 -> 2512 / 2520 /
2544 / 2504 / 2504 / 2568 instructions, highest register R106-R110. With 31 the mode 11 tile issues Q.K
(0x4950), then the fast copies, `LDGDEPBAR`, both `LDSM`/`STSM` pairs (0x55b0-0x5750), and only
then `WARPGROUP.DEPBAR` (0x5760); the vote (`VOTE.ANY`) replaces the per-tile shuffles; P.V is
two `WARPGROUP.ARRIVE` + `QGMMA ... gsb0` pairs.

Arithmetic: no step changes an operation or its order on any value. TCHK: with sl2k > 0,
fl(max_i x_i * sl2k) = max_i fl(x_i * sl2k) (rounding is monotone), so a row's scaled quad max
exceeds off + 8 exactly when one of its threads' scaled partials does; the vote fires on the same
tiles, and inside it the move uses the judged code's `quad_max(mx) * sl2k`. SPV accumulates
half 0 then half 1 into o, as the single group did. So each CTA's partial is expected bitwise
equal to the judged loop's; only the reds' arrival order can differ.

### Ordering argument (no fence or barrier removed or moved across a barrier)

Every tile still executes, in each thread: `cp.async.wait_group 1`, `fence.proxy.async`, the CTA
barrier, the Q.K group and its `wgmma.wait_group 0`, the V^T stores, `fence.proxy.async` and
`bar.sync(2 + hk, 128)` before any P.V issue, and `wgmma.wait_group 0` after the last P.V. One
copy group is committed per tile, as before, so `wait_group 1` covers the same copies.

1. TOV, V^T buffer (it & 1), WAR against P.V(it - 2), which read it: each warp retired P.V(it - 2)
   with its `wait_group 0` in tile it - 2, before it reached the CTA barriers of tiles it - 1 and
   it; the transpose of tile it now runs after the barrier of tile it and before the Q.K wait,
   still after that barrier. RAW to P.V(it): unchanged (stores, `fence.proxy.async`, `bar.sync`,
   wgmma). The ldmatrix of V(it) from stage st: the copies of tile it landed at this thread's
   `wait_group 1` and every thread's at the CTA barrier, as for the judged transpose; the
   ldmatrix reads only the V half while the Q.K wgmma reads the K half (two readers).
2. QKF, stage (it + 2) % 3 (tile it - 1's stage), WAR against Q.K(it - 1) (async proxy) and the
   ldmatrix of V(it - 1): both retired before the CTA barrier of tile it, and the copies are still
   issued after that barrier (now after the Q.K(it) issue, which reads stage st, a different
   stage). Index ring: slot (it + 4) & 7 written, slot (it + 2) & 7 read, as before.
3. FAST: the same rows, the same destination addresses and byte counts as the generic path takes
   for a tile with pos0 + 64 <= seq (every row cached, no patch, no zero fill); 32-bit offsets
   only when the host set `off32` (both caches below 4 GB, page size a power of two), the rule
   PFAST was judged under.
4. SPV: `wgmma.fence` before each P.V issue, so the generic writes of pa[1] between the two
   groups are ordered before the second wgmma reads them; o is written by no other instruction
   between the issues and is read only after `wait_group 0` + the register fence; ptxas reports
   no serialization.
5. TCHK: registers and a warp vote only.

Independent read-only review (fresh agent, 10-30, during check 1, source only): **sound** for
every bit and combination; no fence, barrier, cp.async wait or commit, or wgmma fence / commit /
wait dropped or moved across a barrier; per-CTA output expected bitwise equal. Caveats: (a) TCHK's
equivalence needs sl2k = sl2 x k_scale > 0, and k_scale is a device input (the task's scales are
1 or 1/64, so it holds on every scored and tested case; a negative scale would make the vote and
the judged test disagree); (b) SPV relies on ptxas leaving pa[0] and o untouched while the first
group is in flight, which only the absence of C75xx warnings shows (none in the PRQ 8 and 31
builds above). Also noted: the V^T WAR in point 1 rests on the (it & 1) double buffer, and FAST,
like PFAST, assumes cache row indices below 2^25 (`off32`).

### Gate for check 1 (written before the bench that tests it)

Bench `bench_g` (`tools/attn_make_bench_ab.py`, scratchpad `prq/gen_g.sh`; `--pub <0e9d4c78>
--null --shapes all --inputs 2 --seed 20261037`): arms `ctrl`, `ctrl2` (own state), `pub`,
`pub2` (the compact scorecard with a duplicate of each), and the cumulative ladder `p1`, `p3`,
`p7`, `p15`, `p31` (`-DONE_PRQ=<n>`, sharing ctrl's state, so the d1 full placement term is
absent from their differences).

Modeled gain of an arm X: G(X) = sum over the rows its SASS changes (v1 sliding, v1 full, v16
sliding, v32 sliding) of (ctrl - X) x sensitivity, cold pooled medians, sensitivities from the
judged run 5d309f9d13924dca (v1 sliding 1.420, v1 full 0.494, v16 sliding 0.349, v32 sliding 0.206
%/us). Matching the public kernel on those four rows was worth about 0.9% in d0819bb26907426b.

A ladder arm goes to the confirmation bench only if all hold:
1. ABCHECK passes for every real arm, and X against ctrl differs only at the red-order level (a
   few elements at one BF16 ulp per shape), with the workspace left zero.
2. G(X) >= 0.30% and G(X) > |G(ctrl2)| + 0.10%.
3. On v1 full and on each sliding row that adds more than 0.05% to G(X), the change has the
   same sign on both held-out inputs (ABIN).
4. No row slower than ctrl by more than max(that row's |ctrl2 - ctrl|, 1%).

Choice: the passing arm with the largest G; a smaller bit set within 0.05% of it is preferred.
If a ladder step (for example p1 -> p3, the QKF bit) costs more than 0.05%, the confirmation
candidate drops that bit and is measured fresh. If no arm passes: measured loss for the steps
as built, a row in docs/ruled-out.md, and the round turns to the placement side bet.

Scorecard (no gate): pub - ctrl per row against both nulls, re-ranked by value (difference x
sensitivity), to name what is left after this change.

### Check a7a4c23e98ca484a (bench `bench_g`, generator and kernel at 1baad66): the ladder passes; p31 to confirmation

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | a7a4c23e98ca484a (check) | 1baad66 | bench `bench_g` (`prq/gen_g.sh`, 9 arms) | 2.511x (CI 2.473-2.561) judged path = board code, 16/16 tests, ABCHECKSUM 19/0 every arm | p31 G = +0.86% (null ctrl2 -0.11%); public lead on those rows 0.91% |

Profiles: `muse_attention_decode/profiles/bench_g.a7a4c23e98ca484a.*` (the per-element ABCHECK
lines were cut by the 16,000-character tail; ABCHECKSUM, ABSUM, ABIN and ABSCORE survived).
Cold medians pooled over two held-out inputs, arm - ctrl in us (the ladder shares ctrl's state;
its q_len 1 and flat rows run code SASS-identical to ctrl, so they show this run's noise):

| row | ctrl | ctrl2 - ctrl | pub - ctrl | pub2 - ctrl | p1 | p3 | p7 | p15 | p31 |
|---|---|---|---|---|---|---|---|---|---|
| v1 sliding | 11.23 | +0.07 | -0.06 | -0.13 | -0.03 | -0.13 | +0.00 | +0.03 | -0.13 |
| v1 full | 19.62 | +0.00 | -0.80 | -0.84 | -0.13 | -0.26 | -0.42 | -0.55 | **-0.64** |
| v16 sliding | 16.00 | +0.00 | -0.45 | -0.48 | +0.03 | -0.10 | -0.32 | -0.35 | **-0.45** |
| v32 sliding | 20.86 | +0.04 | -1.34 | -1.50 | +0.10 | -0.16 | -0.57 | -0.80 | **-0.96** |
| G (four rows) | | -0.11% | +0.91% | +1.08% | +0.08% | +0.38% | +0.44% | +0.52% | **+0.86%** |
| ABSCORE (all rows) | | 1.0006 | 1.0168 | 1.0163 | 0.9998 | 1.0019 | 1.0033 | 1.0039 | 1.0059 |

Per-step value (G differences along the ladder): TOV +0.08%, QKF +0.30%, FAST +0.06%, SPV
+0.08%, TCHK +0.34%. Every step is positive. Q.K-first, the step that lost in bf6b8ca9e37341fb's
kernel generation, is now the second largest. TCHK's size is measured, not explained: it removes
the four quad-max shuffles per tile from the critical path between the Q.K wait and the exponentials.

Gate: 1 holds (19/0 for every arm, workspace check inside PASS; per-element detail cut, as in
the two rounds before); 2 holds for p31 (0.86% >= 0.30% and > 0.11% + 0.10%), also for p3, p7 and
p15; 3 holds for p31 (same sign on both inputs: v1 full -0.61 / -0.64, v16 sliding -0.38 / -0.54,
v32 sliding -1.02 / -0.93, v1 sliding -0.03 / -0.03); 4 holds (no row beyond its bound; the largest
is v32 full +0.32 us on unchanged flat code, under ctrl2's +0.99). Choice: p31 (largest G, every
step above 0.05%). On these four rows p31 closes 0.86 of the public kernel's 0.91% lead.

Compact scorecard (pub - ctrl, cold, value = difference x sensitivity; d1 full carries the
placement term): d1 full -0.61 (0.36%), d1 sliding -0.20 (0.34%), v1 full -0.80 (0.40%), v16 full
-2.44 (0.30%), v32 sliding -1.34 (0.28%), v16 sliding -0.45 (0.16%), d16 full -0.83 (0.12%), d32 full
-1.38 (0.12%), v1 sliding -0.06 (0.09%), d16 sliding -0.19 (0.08%), d32 sliding +0.09, v32 full +0.16.
ABSCORE pub 1.0168 / pub2 1.0163 against ctrl: the public lead is down to about 1.7% (3.6% in
5575e7d15d3e49a8 before HMRG). After p31 the per-request q_len 8 rows leave about 0.06%; what
remains sits in d1 (full and sliding, 0.7% together, partly placement) and the flat full rows
(v16 full 0.30%, d16 / d32 full 0.12% each).

### Gate for check 2 (confirmation, written before the bench)

Bench `bench_h` (`prq/gen_h.sh`; `--pub <0e9d4c78> --null --shapes all --inputs 2 --seed
20261038`): arms `ctrl`, `ctrl2`, `pub`, `pub2`, `p31` (`-DONE_PRQ=31`, ctrl's state), `p31own`
(same build, own state), and the placement side bet on the control module (no compile): `n3`
(own state, a third null allocation), `k1` and `k2` (own state, `:pick=3`: `_ONE_WS_PICK = 3`,
three candidate workspaces timed on synthetic batch-1 full inputs at state creation, the fastest
kept; ABADDR prints each pick and its candidates' medians).

PRQ=31 is confirmed if clauses 1, 2 and 4 of the check-1 gate hold for `p31` again, clause 3 holds
on v1 full, v16 sliding and v32 sliding from ABIN, and `p31own`'s G is above 0.30% as well (the
gain does not depend on ctrl's workspace). Then a judged check of submission.py with `ONE_PRQ`
defaulting to 31.

Placement is a production candidate only if: the non-pick states (ctrl, ctrl2, n3) spread by
more than 0.3 us at d1 full (a slow placement occurred), both pick arms sit within 0.15 us of the
fastest non-pick state at d1 full on both inputs, and no row of k1 / k2 is slower than ctrl beyond
the check-1 clause-4 bound. If no slow state occurs among the nulls, the result is inconclusive,
not a win. The pick's cost at state creation (allocation plus 2 x 3 x 9 cold launches) is logged.

### Check 3c6e1a7f1cc644b6 (bench `bench_h`, at 300b915): p31 misses clause 2 by 0.04%; placement misses its row clause

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 3c6e1a7f1cc644b6 (check) | 300b915 | bench `bench_h` (`prq/gen_h.sh`, 9 arms) | 2.476x (CI 2.439-2.514) judged path = board code, 16/16 tests, ABCHECKSUM 19/0 every arm | p31 G +0.55% vs written bar 0.59%; own-state null unusually slow |

Profiles: `muse_attention_decode/profiles/bench_h.3c6e1a7f1cc644b6.*`. Cold pooled, arm - ctrl (us):

| row | ctrl | p31 | p31own | ctrl2 | n3 | k1 | k2 | pub |
|---|---|---|---|---|---|---|---|---|
| d1 full | 17.22 | +0.03 | -0.23 | +0.06 | -0.32 | -0.32 | -0.23 | -0.71 |
| v1 sliding | 10.94 | +0.07 | +0.07 | +0.23 | +0.10 | +0.13 | +0.13 | +0.07 |
| v1 full | 19.90 | **-0.70** | **-0.64** | +0.04 | -0.06 | -0.12 | -0.16 | -0.86 |
| v16 sliding | 15.90 | **-0.35** | -0.19 | +0.26 | +0.13 | +0.23 | +0.29 | -0.22 |
| v32 sliding | 20.77 | **-0.87** | **-0.90** | +0.25 | -0.03 | +0.06 | +0.19 | -1.28 |
| G (four q_len 8 rows) | | +0.55% | +0.47% | -0.49% | -0.15% | -0.22% | -0.25% | +0.67% |

PRQ=31 against its check-2 gate: 1 holds (19/0); 2 **fails as written**: G(p31) = 0.548% >= 0.30%,
but not above |G(ctrl2)| + 0.10% = 0.589%; 3 holds (v1 full -0.77 / -0.54, v16 sliding -0.38 /
-0.35, v32 sliding -0.93 / -0.86); 4 holds (largest +0.87 at v16 full on unchanged flat code,
under ctrl2's +1.35); p31own G = 0.47% > 0.30% holds. Reading: every own-state arm on the control's
code (ctrl2, n3, k1, k2) is slower than ctrl on the q_len 8 sliding rows (G -0.15 to -0.49%), so
ctrl's own state was the fast one in this process; p31 shares that state and ctrl2 does not, so
ctrl2 is not a matched null for p31. The bar is not relaxed after the fact: the confirmation is
repeated with a same-state null (gate below). Across the two benches p31 has the same sign on all
three targeted rows and both inputs every time (v1 full -0.54 to -0.77 us).

Placement against its gate: a slow state occurred (ctrl 17.22 and ctrl2 17.28 against n3 16.90 at
d1 full, spread 0.38 us > 0.3); both pick arms sat within 0.13 us of the fastest non-pick state on
both inputs (k1 -0.29 / -0.32, k2 -0.26 / -0.19 against n3 -0.32 / -0.32); the picks' prepare-time
medians (d1 + v1 full, cold, us) were 36.29 / 35.90 / **35.78** (k1, candidate 2 kept) and
**36.38** / 36.48 / 36.93 (k2, candidate 0). The row clause **fails as written**: k2 is +0.29 us at
v16 sliding against a bound of 0.26 (ctrl2's +0.26). That row's code and plan do not depend on the
pick and the same-sized excess shows on every own-state arm, so the clause compared own-state arms
against ctrl's state. Repeated in check 3 with own-state nulls.

### Gate for check 3 (repeat confirmation, written before the bench)

Bench `bench_i` (`prq/gen_i.sh`; `--null --shapes all --inputs 2 --seed 20261039`, no public
arms): `ctrl`; `ctrlb` (`-DONE_PRQ=0`: the control's code in a separate build that shares ctrl's
state, the matched null for p31); `p31` (ctrl's state); `ctrl2` and `n3` (own state, nulls for the
own-state arms); `p31own` (own state); `k1`, `k2` (own state, `:pick=3`).

PRQ=31 is confirmed if all hold: (1) ABCHECKSUM 19/0 for every arm; (2) G(p31) - G(ctrlb) >= 0.30%
and G(p31) > |G(ctrlb)| + 0.10%; (3) G(p31own) - max(G(ctrl2), G(n3)) >= 0.30%; (4) p31 has the same
sign on both inputs on v1 full, v16 sliding and v32 sliding; (5) no row of p31 slower than ctrl by
more than max(|ctrlb - ctrl|, 1%). Then a judged check of submission.py with `ONE_PRQ` 31.
Otherwise PRQ=31 stays off, recorded as unconfirmed (three benches, consistent sign), not as a loss.

Placement is a production candidate if: (a) the own-state arms on q_len 1 code identical to ctrl
(ctrl2, n3, p31own) spread by more than 0.3 us at d1 full; (b) both pick arms sit within 0.15 us
of the fastest of those at d1 full on both inputs; (c) no row of k1 / k2 slower than the slower of
ctrl2 and n3 on that row by more than max(0.1 us, 1%). If (a) fails the result is inconclusive.

### Check 409996a7214f411a (bench `bench_i`, at 8a6c8bb): PRQ=31 confirmed, placement pick passes

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 409996a7214f411a (check) | 8a6c8bb | bench `bench_i` (`prq/gen_i.sh`, 8 arms, no public arms) | 2.496x (CI 2.459-2.543) judged path = board code, 16/16 tests, ABCHECKSUM 19/0 every arm | p31 G +0.72% (same-state null -0.00%); picks at the fastest state |

Profiles: `muse_attention_decode/profiles/bench_i.409996a7214f411a.*`. Cold pooled, arm - ctrl (us):

| row | ctrl | ctrlb | p31 | p31own | ctrl2 | n3 | k1 | k2 |
|---|---|---|---|---|---|---|---|---|
| d1 full | 17.22 | +0.06 | +0.09 | -0.26 | +0.09 | -0.29 | -0.32 | -0.32 |
| v1 sliding | 10.75 | -0.03 | -0.03 | -0.09 | +0.03 | -0.13 | -0.03 | +0.03 |
| v1 full | 19.65 | +0.03 | **-0.64** | **-0.58** | +0.03 | -0.13 | -0.07 | -0.07 |
| v16 sliding | 15.87 | +0.03 | **-0.48** | **-0.51** | +0.03 | -0.13 | -0.06 | +0.00 |
| v32 sliding | 20.93 | +0.09 | **-0.96** | **-0.99** | +0.00 | -0.07 | -0.10 | +0.03 |
| G (four q_len 8 rows) | | -0.00% | +0.72% | +0.80% | -0.07% | +0.31% | +0.12% | -0.01% |

PRQ=31 against the check-3 gate: (1) 19/0 every arm; (2) G(p31) - G(ctrlb) = 0.73% >= 0.30% and
0.72% > 0.00% + 0.10%; (3) G(p31own) - max(G(ctrl2), G(n3)) = 0.80 - 0.31 = 0.49% >= 0.30%; (4) same
sign on both inputs: v1 full -0.51 / -0.74, v16 sliding -0.42 / -0.51, v32 sliding -0.93 / -0.99;
(5) largest p31 regression +1.38 us at v32 full (flat code unchanged; ctrlb +0.87, ctrl2 +1.25),
under its 1% bound of 1.92. **Confirmed.**

Placement against the check-3 gate: (a) own-state arms on q_len 1 code identical to ctrl at d1
full: ctrl2 +0.09, n3 -0.29, p31own -0.26: spread 0.38 us > 0.3; (b) per input the picks are at
or below the fastest of those (in 0: k1 -0.29, k2 -0.26 against n3 -0.22; in 1: k1 -0.35, k2 -0.42
against n3 -0.38); (c) largest excess over the slower of ctrl2 and n3: k1 at d16 sliding, +0.12
against +0.03, 0.09 us, under max(0.1 us, 1%). **Passes.** Prepare-time medians (d1 + v1 full, cold,
us): k1 36.32 / 36.10 / **35.84** (candidate 2), k2 **36.29** / 36.61 / 36.64 (candidate 0); the same
choices as in 3c6e1a7f1cc644b6. In two benches the judged path's first allocation (ctrl) and ctrl2
were the slow ones at d1 full, 0.3-0.4 us behind; the pick is worth about that times 0.60 %/us
(+0.18-0.23%) whenever the first allocation lands slow, and nothing when it does not.

Three-bench summary for PRQ=31 (cold, us, p31 - ctrl): v1 full -0.64 / -0.70 / -0.64, v16 sliding
-0.45 / -0.35 / -0.48, v32 sliding -0.96 / -0.87 / -0.96; G +0.86% / +0.55% / +0.72%.

Production: `ONE_PRQ` defaults to 31 in attn_one.cu and submission.py; `_ONE_WS_PICK = 3` in
submission.py. SASS (CUDA 12.8.1, all 13 instances): the default build equals the benched p31
build; `-DONE_PRQ=0` equals 5c11a11; no spills. pyflakes clean; `tools/attn_compile_check.sh`
no errors. Next: a judged check of submission.py and a repeat (checks 4 and 5 of this round).

### Judged: check 640ee49b691b45ac 2.588x (CI 2.549-2.627), repeat 8b00100bf85d4ef5 2.545x (CI 2.506-2.583)

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 640ee49b691b45ac (check) | 57f57dd | submission.py (`ONE_PRQ` 31, `_ONE_WS_PICK` 3, HMRG on; sha256 9e627a90) | 2.588x (CI 2.549-2.627), 16/16 tests | pick: candidate 2 of 3, medians 35.46 / 35.74 / 35.30 us |
| 2026-10-30 | 8b00100bf85d4ef5 (check) | 57f57dd + one comment line | repeat of the same file | 2.545x (CI 2.506-2.583), 16/16 tests | pick: candidate 1 of 3, 35.62 / 35.39 / 35.62 us; full rows d16 / d32 / v16 / v32 slower with no code change there (machine speed) |

Profiles: `muse_attention_decode/profiles/sub_prq.640ee49b691b45ac.*`, `sub_prq_r.8b00100bf85d4ef5.*`.
Judged per-shape times (us, baseline median / paired speedup) on the rows PRQ changes, against
the HMRG pair (same code elsewhere) and this round's three bench checks' judged paths (board code):

| row | PRQ 640ee49b / 8b00100b | HMRG 5d309f9d / 82b1bf0f | board paths a7a4c23e / 3c6e1a7f / 409996a7 |
|---|---|---|---|
| v1 sliding | 7.13 / 6.80 | 6.39 / 6.77 | 6.92 / 6.95 / 6.82 |
| v1 full | **14.50 / 14.89** | 15.36 / 15.61 | 15.26 / 15.72 / 15.14 |
| v16 sliding | **11.45 / 11.45** | 11.77 / 12.16 | 11.58 / 11.93 / 11.77 |
| v32 sliding | **15.03 / 15.02** | 15.93 / 15.89 | 16.23 / 16.38 / 15.98 |

The judged scores pair 2.588 / 2.545 against HMRG's 2.512 / 2.582: the run-to-run spread (1-3%,
machine speed: the full rows moved by up to 6 us with no code change) is larger than the modeled
+0.7% gain, so the judged pair confirms that the file passes and that the targeted rows moved,
and the bench checks carry the size. v1 sliding's 7.13 in the first check sits 0.2 us above its
range on board code (6.39-6.95 in five runs) and came back to 6.80 in the repeat; no bench saw an
effect there (-0.13 / +0.07 / -0.03). Not submitted: handed to the coordinator (`main`) with the
file path and these job ids.

## Standalone round 3 (10-30): family ledger update

Checks used by this round: a7a4c23e98ca484a, 3c6e1a7f1cc644b6, 409996a7214f411a, 640ee49b691b45ac,
8b00100bf85d4ef5 (5 of the 5 budgeted). Quota after them (`okc problems`): 5 of 20 attention checks, 9 of 10 runs.

| family | state | evidence | reopening trigger |
|---|---|---|---|
| Per-request q_len 8 tile in the public order (`ONE_PRQ` = 31: TOV, QKF, FAST, SPV, TCHK; modes 1, 3, 11, 15) | **measured standalone win**, on by default | benches a7a4c23e98ca484a (G +0.86%), 3c6e1a7f1cc644b6 (+0.55%, missed its written null clause by 0.04%), 409996a7214f411a (+0.72%, same-state null -0.00%); judged 2.588x / 2.545x; review sound with two caveats | layer port is a separate gate (`tools/dl_one_inline.py`) |
| Its steps one by one (ladder in a7a4c23e98ca484a) | measured, each positive | TOV +0.08%, QKF +0.30%, FAST +0.06%, SPV +0.08%, TCHK +0.34% | none needed while all ship together |
| Prepare-time workspace placement pick (`_ONE_WS_PICK` = 3) | **measured standalone win** where the first allocation is slow, on by default | 3c6e1a7f1cc644b6 (row clause missed against ctrl's state), 409996a7214f411a (passes with own-state nulls): picks at the fastest null state at d1 full on both inputs, 0.3-0.4 us ahead of the slow states | a judged d1 full above 13 us with the pick on |
| Old QKF (mode bit 128, bf6b8ca9e37341fb, 10-28 kernel) | superseded on the per-request plans | QKF as a step of PRQ is worth +0.30% here | the flat consumer-copy mode 19, where it lost, is not affected |

Where the remaining gap is (bench 3c6e1a7f1cc644b6, pub - p31, cold, us; value at the
sensitivities above): v16 full -1.95 (0.24%), d1 full -0.74 (0.44%, of which up to 0.3-0.4 is
placement the pick now takes), d32 full -1.50 (0.13%), d16 full -0.73 (0.11%), v1 full -0.16
(0.08%), v32 sliding -0.41 (0.08%), d1 sliding -0.06, v16 sliding +0.13. Next targets, by value:
batch-1 full at q_len 1 (d1 full, after the pick) and the flat full-context rows (v16 / d16 / d32
full, the producer kernels 51 and 49), which no round-2 or round-3 change touched.

### Scored: 1896e72c11ae4b0f 2.521x (2026-10-30)

Submit of submission.py at a70192d (ONE_PRQ=31, _ONE_WS_PICK=3, ONE_HMRG on), submitted by the coordinator under the standing authorization. Passed. Above the bench-file entry e9085c86d89449c2 (2.504x), so a clean file of our own code becomes the attention board entry. Receipt: muse_attention_decode/profiles/scored.1896e72c11ae4b0f.json.

## Standalone round 4 (10-30): the flat full-context rows (modes 49 and 51)

Brief: attribute the flat full-context gap to the public flat kernels in one process (stamps,
SASS of the per-tile paths), name the mechanism, build one switch (off = SASS-identical), gate
written here before the check that tests it (modeled >= 0.2% standalone score gain beyond the
independent null). Start: main 15cc6b9 (HMRG, PRQ=31, `_ONE_WS_PICK` 3; board entry
1896e72c11ae4b0f, 2.521x). Quota at the start (`okc problems`, 19:09 UTC): 5 of 20 attention
checks, 8 of 10 runs, all expiring at 00:00 UTC. Gaps to the public kernel this round starts from
(bench 3c6e1a7f1cc644b6, cold, pub - p31): v16 full -1.95 us (0.24%), d32 full -1.50 (0.13%), d16
full -0.73 (0.11%); d1 full -0.74 runs the per-request mode 11 and is not in this round's scope.

### Static comparison of the q_len 1 flat kernels (SASS, CUDA 12.8.1)

Ours mode 49 (`tools/attn_one_sass.sh`, 3448 instructions, consumer region R115, no spills)
against the public `attn_kernel<1, true, 0>` (`tools/attn_pub_compile.sh` on the public file sha1
0e9d4c78, 3416 instructions, 128 registers; the public copy stays in the scratchpad):

| piece | ours (mode 49) | public flat q_len 1 |
|---|---|---|
| consumer tile body (mbarrier wait to empty arrive) | 381 instructions: 32 HMMA, 13 LDS, 68 F2FP, 9 MUFU, 2 SHFL, one VOTE (alpha) | 403: 32 HMMA, 26 LDS, 68 F2FP, 9 MUFU, 2 SHFL |
| producer, page entries | `__ldg` per row, three tiles ahead in registers, 64-bit pointer math (about 31 instructions per tile for 4 entries) | one bulk copy (`cp.async.bulk`) of the tile's 64 entries per tile into a shared-memory slot, three tiles ahead, an mbarrier per slot and a producer `bar.sync` per tile |
| producer, cached-row copies | PFAST: 8 `LDGSTS` per tile per thread, 32-bit offsets | the same shape (8 `LDGSTS`, 32-bit offsets) |
| ring across segments | continuous (the producer runs ahead into the next segment) | drained and re-initialized per segment |
| merge after each segment | each of the 4 warps (2 KV heads x 2 key halves) reds its own partial: every workspace address takes 2 reds per CTA | the second key half hands (m, l, o) to the first through shared memory; only the first issues reds (the same handoff as its per-request path) |
| partition, grid | 264 CTAs, equal shares of a virtual key axis, 400 keys per request boundary | the same (264, `SEGKF` 400) |

The tile loops are within 6% of each other in instructions, ours shorter. The one structural
difference after the loop is the merge: the per-request q_len 1 modes got the key-half handoff in
round 2 (`ONE_HMRG`, judged; -0.80 / -1.44 us at d32 / d16 sliding, where the reds were the
measured mechanism in ce92d32e26024bd2), but it was scoped to `!FLAT`, so mode 49 still issues
twice the public kernel's reds. In the flat kernel the grid finishes together by construction
(equal shares), so the last segments' reds of all 264 CTAs land on L2 in one burst at the end
of the call, about 8 CTAs per request on the same 1,040 addresses, and the last CTA of each
request waits for them before its finalize. Hypothesis (H1): the doubled reds at the flat
kernel's tail are the d16 / d32 full gap. The stamps below test where the time goes; the switch
tests H1 directly.

### The switch: `ONE_HMRGF` (default 0)

`attn_one.cu`: the HMRG handoff for the flat q_len 1 instances (17, 49). In producer mode the
stage ring belongs to the producer across segments (it may already be filling the next
segment's tiles while the consumers merge), so the handoff cannot use stage 0 as HMRG does; it
uses the 4,352 bytes from `S::Q` (query fragments, bf16 query staging, raw new rows: 4,608 bytes
up to `S::KN`, checked by a static_assert), which nothing reads between a segment's tile loop and
the next segment's query pass. Arithmetic as in HMRG: each half scales by its own 2^m, one FP32
add before the red.

SASS (CUDA 12.8.1, all 13 instances, per-function compare with namespace hashes normalized,
scratchpad `flat/sass_all.sh`): with `ONE_HMRGF=0` (and `ONE_STAMP=0`) every instance is
identical to main 15cc6b9; with 1 only modes 17 and 49 change (3448 -> 3520 and 2648 -> 2712
instructions), mode 49 consumer region R109, no spills, no warnings beyond the usual unused ones.

`ONE_STAMP=1` now also records the q_len 1 flat kernel (bench-only): 3 the last segment's tile
loop end, 4 its reds issued (before its ticket), 5 the kernel end. The bench prints them per CTA
and a compact `STSUM` line per shape and arm (medians over five cold calls) at the end of the log.
The public stamp copy `pubstamp1` (scratchpad `flat/make_pubstamp1.py`, never committed) records
start, SM, tiles and, at q_len 1, the last tile-loop end and the kernel end.

### Ordering argument (no fence or barrier removed or moved)

The handoff adds the judged HMRG sequence to mode 49's merge: `cp_wait<0>`, consumer barrier
(`bar.sync 1, 128`; mode 17 `__syncthreads`), generic `st.shared` by the second-half warps,
consumer barrier, generic `ld.shared` by the first-half warps, their reds. The reds are still
issued before the ticket's consumer barrier that precedes thread 0's `atom.acq_rel.gpu`, so the
release chain to the finalizing CTA is the judged one. The buffer's previous contents are read
only before or inside the tile loop: the query fragments (`ld.shared` into registers before the
loop), the bf16 query rows and raw new rows (written by the consumers' own `cp.async`, waited by
`cp.async.wait_group` and a barrier, read in the query and new-token passes). Every consumer
thread finishes those reads in program order before it arrives at the merge's first barrier, and
`cp_wait<0>` leaves no copy of its own in flight. The next writes into those bytes are the next
segment's query copies, issued after that segment's `cp_wait<0>` and consumer barrier, which
every first-half thread reaches only after its loads (and after the ticket's two barriers). The
producer warpgroup writes only the stages and arrives on the stage mbarriers; it never touches
`[S::Q, S::KN)`. At q_len 1 no async-proxy instruction (wgmma, bulk copy) reads shared memory, so
no proxy fence is involved. A segment the CTA does not own (`seg_bounds` false) runs no merge.

### Gate for check 1 (written before the bench that tests it)

Bench `bench_j` (scratchpad `flat/gen_j.sh`; `tools/attn_make_bench_ab.py --pub <0e9d4c78> --pub
pubstamp1 --pub pubnc_probe --null --shapes full --inputs 2 --rounds 200 --seed 20261040`):
`ctrl`, `ctrl2` (own state, the independent null), `ctrlb` (`-DONE_HMRGF=0`, the control's code in
a separate build on ctrl's state: the matched null), `hf` (`-DONE_HMRGF=1`, ctrl's state), `hfown`
(the same build, own state), `stamp` (`-DONE_STAMP=1`), `nocopy_probe` (`-DONE_PROBE=1`: cached
rows zero-filled, no cache reads), `pub`, `pub2`, `pubstamp1`, `pubnc_probe` (the public file
with its producer's cached-row copies at src-size 0).

Modeled gain G(X) = sum over d16 full and d32 full (the rows whose code the switch changes) of
(ctrl - X) x sensitivity, cold pooled medians, sensitivities from the board run
1896e72c11ae4b0f (d16 full 0.147, d32 full 0.084 %/us; v16 / v32 full 0.121 / 0.072).

`ONE_HMRGF=1` goes to a confirmation bench only if all hold:
1. ABCHECKSUM passes for every real arm; `hf` against ctrl differs only at the red-order level
   and leaves the workspace zero.
2. G(hfown) - G(ctrl2) >= 0.20% (the independent null) and G(hf) - G(ctrlb) >= 0.20% (the matched
   null).
3. `hf` has the same sign on both held-out inputs on d16 full and d32 full (ABIN).
4. No row of `hf` slower than ctrl by more than max(|ctrlb - ctrl|, 1%) (v16 / v32 full run
   unchanged code).

If `hf` is faster on both rows but misses clause 2, it is recorded as a measured standalone gain
below the bar (not promoted, a row in docs/ruled-out.md with its size). If it is not faster, H1 is
a measured loss and the stamps name what is left.

Attribution (no gate): per flat row, ours against the public: start and first-tile time, loop
end (min / median / max), tail from the loop end to the kernel end, per-SM end, pair gap
(`STSUM`), and the no-copy probes (`nocopy_probe` against `pubnc_probe`: the instruction streams
without DRAM).

Independent read-only review of the ordering argument (fresh agent, 10-30, during check 1,
source only): **sound**, with caveats: (a) `ONE_DETM=1` takes the merge branch first, so HMRGF
has no effect in a DETM build; (b) the static_assert bounds the buffer by `S::KN` only, and the
overlap with the raw new rows (`KVST`, first 256 bytes) is safe because those are read before the
loop and drained by `cp.async.wait_group` and a barrier; a future change that reads `KVST` after
the loop would not be caught; (c) the buffer in use is [S::Q, S::Q + 4352), narrower than the
comment's [S::Q, S::KN).

### Check bce37d22a0ec4045 (bench `bench_j`, at d12b20e): HMRGF misses its gate; the gap is the per-SM pair spread

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | bce37d22a0ec4045 (check) | d12b20e | bench `bench_j` (`flat/gen_j.sh`, 11 arms, `--shapes full`) | 2.571x (CI 2.533-2.610) judged path = board code, 16/16 tests, ABCHECKSUM 11/0 every real arm | HMRGF G +0.054% (bar 0.20%); stamps name the pair spread |

Profiles: `muse_attention_decode/profiles/bench_j.bce37d22a0ec4045.*`. Cold medians pooled over two
held-out inputs, arm - ctrl (us):

| row | ctrl | ctrlb | ctrl2 | hf | hfown | pub | pub2 | nocopy_probe | pubnc_probe |
|---|---|---|---|---|---|---|---|---|---|
| d16 full (49) | 97.89 | +0.06 | +0.00 | -0.29 | -0.22 | -0.90 | -1.06 | -22.14 | -31.42 |
| d32 full (49) | 173.25 | +0.03 | +0.00 | -0.13 | -0.10 | -1.66 | -1.76 | -34.62 | -52.22 |
| v16 full (51) | 111.78 | -0.58 | -0.35 | -0.96 | -0.77 | -3.26 | -3.14 | -26.56 | -22.88 |
| v32 full (51) | 189.12 | -0.29 | -0.19 | -0.06 | -0.26 | -0.93 | -1.02 | -37.47 | -30.30 |

HMRGF against its gate: (1) holds (11/0); (2) **fails**: G(hfown) - G(ctrl2) = 0.042%, G(hf) -
G(ctrlb) = 0.065%, against 0.20%; (3) holds (d16 -0.29 / -0.22, d32 -0.32 / -0.06); (4) holds.
Recorded as a measured standalone gain below the bar (about 0.05%), not promoted.

Stamps (five cold calls per shape, medians of the per-call statistics, `STSUM`; times from the
grid's first start, us; ours `stamp`, public `pubstamp1`; the stamp builds cost +0.5 / +0.9 us at
d16 / d32 full and +1.3 / +2.7 at v16 / v32 full over their unstamped builds):

| row | arm | loop end min / median / max | tail after the loop, median / max | per-SM end median / max (loop) | pair gap (median) | span (kernel end) |
|---|---|---|---|---|---|---|
| d16 full | ours | 80.9 / 86.9 / 90.8 | 1.3 / 2.4 | 88.2 / 90.8 | 3.1 | 92.8 |
| | public | 82.0 / 86.7 / 89.4 | 1.2 / 2.4 | 87.3 / 89.4 | 1.0 | 91.4 |
| d32 full | ours | 152.1 / 161.3 / 168.1 | 1.3 / 2.6 | 164.6 / 168.1 | 6.2 | 170.0 |
| | public | 154.1 / 161.5 / 166.2 | 1.2 / 2.8 | 162.2 / 166.2 | 2.1 | 167.9 |
| v16 full | ours | 80.6 / 92.1 / 101.5 | 12.3 / 23.1 (barrier + finalize) | 97.0 / 101.5 | 11.5 | 104.1 |
| | public | 82.5 / 92.5 / 100.4 | 10.0 / 19.6 | 95.2 / 100.4 | 7.0 | 102.9 |
| v32 full | ours | 148.6 / 171.1 / 183.7 | 23.0 / 37.1 | 177.5 / 183.7 | 22.3 | 186.4 |
| | public | 151.6 / 169.0 / 180.2 | 14.6 / 31.3 | 174.5 / 180.2 | 16.0 | 183.3 |

What this names (measured):

- **Not the merge.** At q_len 1 the tail from the last loop end to the kernel end is the same in
  both kernels (1.3 vs 1.2 us median, 2.4-2.6 vs 2.4-2.8 max). H1 (the doubled reds) is at most
  the 0.1-0.3 us HMRGF measured. The first tile is ready 4.4 us after the start in ours (the
  public copy has no first-tile stamp); starts spread 0.2 us in both.
- **Not the median throughput.** The median CTA's loop ends at the same time in both kernels at
  every row (86.9 vs 86.7, 161.3 vs 161.5, 92.1 vs 92.5, 171.1 vs 169.0).
- **The pair spread.** The two CTAs of an SM end 3.1 / 6.2 us apart in mode 49 against 1.0 / 2.1 in
  the public kernel (d16 / d32 full), 11.5 / 22.3 against 7.0 / 16.0 in mode 51. The favoured CTA
  ends earlier than the public one (loop min 152.1 vs 154.1 at d32), the partner later, and the
  latest SM ends 1.4-3.5 us later than the public kernel's: the size of each row's cold gap.
  Mode 51 showed the same pattern before DCARRY / PK16 / VHOIST (bench e9085c86d89449c2), and its
  pair gap fell as its per-tile instructions fell.
- **The instruction streams differ without DRAM.** With the cache reads removed, mode 49 trails
  the public q_len 1 kernel by 9.3 / 17.6 us (75.7 vs 66.5, 138.6 vs 121.0), while mode 51 is ahead
  of the public q_len 8 kernel by 3.7 / 7.2 us. So at q_len 1 the partner CTA's pace is set by a
  heavier or longer per-tile stream; at q_len 8 the remaining spread is not an instruction-count
  deficit (mode 51 keeps the per-tile proxy fence the public kernel omits, 1.2-1.4 us in
  e9085c86d89449c2).
- v16 full's gap depends on the input: -5.47 / -1.50 us (input 0 / 1), the largest per-input
  difference in this bench; not pursued this round.

### The switch for check 2: `ONE_F1` (bit mask, default 0; modes 17 and 49 only)

Two pieces of mode 49's per-tile stream that differ from the public kernel, measured separately:
bit 1 `QK2`: each key block's Q.K was one chain of four dependent `mma.sync` (k-steps 0..3 into one
accumulator); it becomes two independent chains of two (even and odd k-steps, as the public
kernel's se / so) summed with one FP32 add per element (a different FP32 summation order, the
same products); bit 2 `PE16` (producer): with page size 1 the page entries three tiles ahead of a
fully cached tile load from one base with immediate offsets (32-bit index) instead of a 64-bit
pointer per row (the refill drops from about 31 to about 10 SASS instructions per tile). No
fence, barrier, mbarrier, wait or commit is added, removed or moved: QK2 changes register
arithmetic only, PE16 loads the same addresses (pos >> 0 = pos) into the same registers.

SASS (all 13 instances, `flat/sass_all.sh`): `ONE_F1=0` identical to main 15cc6b9; 1 changes only
17 and 49 (3448 -> 3472, consumer region R121), 2 only 49 (3496), 3 only 17 and 49 (3520); no spills.

### Gate for check 2 (written before the bench that tests it)

Bench `bench_k` (`flat/gen_k.sh`, `--pub <0e9d4c78> --null --shapes full --inputs 2 --rounds 200
--seed 20261041`): `ctrl`, `ctrl2` (independent null), `ctrlb` (`-DONE_F1=0`, matched null on
ctrl's state), `f1`, `f2`, `f3` (`-DONE_F1=1/2/3`, ctrl's state), `f3h` (`-DONE_F1=3
-DONE_HMRGF=1`), `f3own` (`-DONE_F1=3`, own state), `stampf3` (`-DONE_F1=3 -DONE_STAMP=1`), `pub`,
`pub2`. G as in check 1 (d16 / d32 full, 0.147 / 0.084 %/us).

An arm X goes to confirmation only if: (1) ABCHECKSUM passes for every real arm; (2) G(X) -
G(ctrlb) >= 0.20%, and for f3 or f3h also G(f3own) - G(ctrl2) >= 0.20% (another winner needs an
own-state twin in the confirmation); (3) same sign on both inputs at d16 and d32 full; (4) no
row slower than ctrl by more than max(|ctrlb - ctrl|, 1%). Choice: largest G; a smaller bit set
within 0.05% of it is preferred. An arm faster on both rows but below the bar is a measured gain
below the bar (not promoted). The stamps of `stampf3` against check 1's `stamp` say whether the
pair gap moved.

### Check c8e99e038d9f4575 (bench `bench_k`, at 2bf3967): ONE_F1 is a measured loss

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | c8e99e038d9f4575 (check) | 2bf3967 | bench `bench_k` (`flat/gen_k.sh`, 11 arms, `--shapes full`) | 2.504x (CI 2.466-2.541) judged path = board code, 16/16 tests, ABCHECKSUM 11/0 every arm | no arm passes; QK2 slower, PE16 null |

Profiles: `muse_attention_decode/profiles/bench_k.c8e99e038d9f4575.*`. A slower machine than check 1
(ctrl 100.58 / 179.36 at d16 / d32 full against 97.89 / 173.25). Cold pooled, arm - ctrl (us):

| row | ctrl | ctrlb | ctrl2 | f1 (QK2) | f2 (PE16) | f3 | f3h (+HMRGF) | f3own | pub |
|---|---|---|---|---|---|---|---|---|---|
| d16 full | 100.58 | +0.06 | +0.00 | +0.12 | -0.04 | -0.04 | -0.16 | -0.10 | -0.80 |
| d32 full | 179.36 | +0.13 | +0.16 | +0.32 | +0.03 | +0.10 | +0.03 | +0.13 | -1.31 |
| G (d16 + d32 full) | | -0.02% | -0.01% | -0.05% | +0.00% | -0.00% | +0.02% | +0.00% | +0.23% |

Against the gate: (1) holds (11/0 every arm); (2) fails for every arm (the best, f3h, is +0.04%
over the matched null). QK2 is slower on both rows and both inputs (+0.10 / +0.26 at d16, +0.13 /
+0.22 at d32): the four-step Q.K chain is not on mode 49's critical path, and the extra adds and
six more registers cost a little. PE16 moves nothing (-0.04 / +0.03). Both are measured losses for
the switches as built (rows in docs/ruled-out.md). `stampf3`: pair gap 2.8 / 5.2 us at d16 / d32
full (check 1's `stamp`: 3.1 / 6.2 on a faster machine), tails 1.3 us as before. The v16 / v32 full
rows (code unchanged in every arm) moved by -1.1 to +0.6 us across arms, mostly on input 0: the
q_len 8 flat rows carry about 1 us of run noise in this bench.

### Check 3 (diagnostic, no gate): the producer-bound pace of both q_len 1 kernels

What is left unnamed: without DRAM traffic mode 49 trails the public q_len 1 kernel by 9-18 us,
and neither the consumer's Q.K chain nor the producer's page-entry refill is it. Bench `bench_l`
(`flat/gen_l.sh`, `--null --shapes full --inputs 2 --rounds 200 --seed 20261042`): `nocopy_probe`
(`ONE_PROBE=1`) and `pubnc_probe` as before; `prodonly_probe` (`ONE_PROBE=3`, new: no cache reads,
and the consumers release each stage without computing on it; mode 49 only, SASS of every other
instance unchanged by it apart from the zero-fill it shares with 1) and `pubncc_probe` (the public
no-copy copy whose q_len 1 consumers do the same: one `mbarrier.arrive` per thread per stage,
scratchpad `flat/pubncc.py`); `hf` / `hfown` (HMRGF repeat for the ledger) with `ctrl2`, `pub`,
`pub2`. Reading: producer-only time, ours against the public, says whether the producer stream or
the consumer stream holds the no-DRAM deficit. No candidate is judged by this check.

### Check 4b60264a276c4805 (bench `bench_l`, at 0b326a4): both q_len 1 streams carry the no-DRAM deficit; HMRGF repeats small

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 4b60264a276c4805 (check) | 0b326a4 | bench `bench_l` (`flat/gen_l.sh`, 10 arms, `--shapes full`) | 2.498x (CI 2.461-2.535) judged path = board code, 16/16 tests, ABCHECKSUM 11/0 every real arm | diagnostic, no gate |

Profiles: `muse_attention_decode/profiles/bench_l.4b60264a276c4805.*`. Cold pooled (us):

| row | ctrl | pub | nocopy (ours / public) | producer only (ours / public) | hf - ctrl | hfown - ctrl | ctrl2 - ctrl |
|---|---|---|---|---|---|---|---|
| d16 full | 100.45 | 99.49 | 75.55 / 66.05 | 37.63 / 32.93 | -0.22 | -0.29 | -0.06 |
| d32 full | 179.39 | 177.60 | 138.43 / 120.93 | 63.23 / 55.20 | -0.19 | -0.35 | -0.22 |

- Producer only (no cache reads, consumers release each stage untouched): mode 49's producer is
  4.7 / 8.0 us (14%) slower than the public producer at d16 / d32 full, about half of the no-DRAM
  deficit (9.5 / 17.5 us). The rest is in the consumers' stream or in how the two interleave. Both
  kernels take about twice as long with the consumers computing as without them.
- With DRAM both deficits shrink to the 1.0 / 1.8 us gap: DRAM sets the median pace and the
  streams decide how unevenly the two CTAs of an SM share it.
- HMRGF again: -0.22 / -0.19 (hf), -0.29 / -0.35 (hfown, null -0.06 / -0.22), same sign on all
  four per-input rows: about +0.05-0.07%, as in check 1. Still below the 0.20% bar.
- The v16 full row varies with the input again (pub - ctrl -4.6 / -0.5 us on inputs 0 / 1).

### The switch for check 4: `ONE_NS1` (mode 49 ring depth, default 3)

Check 1's stamps at d32 full: the consumers wait on the full barrier about 50 us (median thread 0)
and the producer on the empty barrier about 95 us of a 165 us call, at the same time; the ring
(3 stages of 16 KB) is full from the producer's side while the consumer of the partner CTA starves.
`ONE_NS1` sets mode 49's stage count (`one_ns`); shared memory per CTA is NS x 16 KB + 9.9 KB (4:
75.4 KB, 6: 108.2 KB), so two CTAs per SM still fit up to 6 (2 x (108.2 + 1) KB of 228 KB). The
ring code is generic in NS (mbarrier init, phase flip, the producer's final drain); no fence,
barrier, wait or arrive changes, only the ring length. SASS: `ONE_NS1=3` identical to main on all
13 instances; 4 and 6 change only mode 49 (3448 -> 3472 / 3520 instructions, consumer region R115,
no spills).

### Gate for check 4 (written before the bench that tests it)

Bench `bench_m` (`flat/gen_m.sh`, `--pub <0e9d4c78> --null --shapes full --inputs 2 --rounds 200
--seed 20261043`): `ctrl`, `ctrl2`, `ctrlb` (`-DONE_NS1=3`), `ns4`, `ns5`, `ns6` (`-DONE_NS1=4/5/6`,
ctrl's state), `ns6h` (`-DONE_NS1=6 -DONE_HMRGF=1`), `ns4own`, `ns6own` (own state), `stampns6`
(`-DONE_NS1=6 -DONE_STAMP=1`), `pub`, `pub2`. ABPLAN must show 264 CTAs in mode 49 for every ns arm.

An arm goes to confirmation (check 5) only if: (1) ABCHECKSUM passes for every real arm; (2) G(X) -
G(ctrlb) >= 0.20%, and for ns4 / ns6 / ns6h also G(own twin) - G(ctrl2) >= 0.20%; (3) same sign on
both inputs at d16 and d32 full; (4) no row slower than ctrl by more than max(|ctrlb - ctrl|, 1%).
Choice: largest G; a shallower ring within 0.05% is preferred (less shared memory). The quota
leaves no judged check today: a passing arm is confirmed in check 5 and judged after the reset.

### Check 42c4739cf6b247c4 (bench `bench_m`, at 5064e2b): a deeper mode 49 ring is a measured loss, monotonic in depth

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | 42c4739cf6b247c4 (check) | 5064e2b | bench `bench_m` (`flat/gen_m.sh`, 12 arms, `--shapes full`) | 2.492x (CI 2.455-2.530) judged path = board code, 16/16 tests, ABCHECKSUM 11/0 every arm | every ns arm slower; 264 CTAs in mode 49 for all (ABPLAN) |

Profiles: `muse_attention_decode/profiles/bench_m.42c4739cf6b247c4.*`. Cold pooled, arm - ctrl (us):

| row | ctrl | ctrlb | ctrl2 | ns4 | ns5 | ns6 | ns6h | ns4own | ns6own | pub |
|---|---|---|---|---|---|---|---|---|---|---|
| d16 full | 100.58 | -0.07 | +0.00 | +0.80 | +1.24 | +1.72 | +1.85 | +0.67 | +1.69 | -0.71 |
| d32 full | 179.49 | -0.07 | -0.13 | +0.29 | +0.99 | +2.24 | +2.27 | +0.19 | +1.95 | -1.63 |

Against the gate: (1) holds; (2) fails for every arm (G from -0.11% to -0.46%); slower on both
inputs on both rows for every depth. `stampns6` at d32 full against check 1's `stamp`: the producer's
empty-barrier wait falls from about 95 to 31 us, the consumers' full-barrier wait stays about 54, and
the loop ends spread wider (148.0 / 167.1 / 176.9 against 152.1 / 161.3 / 168.1; pair gap 4.8, latest
SM 176.9). A deeper ring lets each SM's favoured CTA run further ahead of its partner on the shared
DRAM bandwidth, and the partner, which sets the end, finishes later. So the ring is not too shallow;
if anything the reverse.

### Gate for check 5 (written before the bench that tests it)

Bench `bench_n` (`flat/gen_n.sh`, `--pub <0e9d4c78> --null --shapes full --inputs 2 --rounds 200
--seed 20261044`): `ctrl`, `ctrl2`, `ctrlb` (`-DONE_NS1=3`), `ns2` (`-DONE_NS1=2`, ctrl's state),
`ns2h` (`-DONE_NS1=2 -DONE_HMRGF=1`), `ns2own`, `ns2hown` (own state), `stampns2` (`-DONE_NS1=2
-DONE_STAMP=1`), `pub`, `pub2`. SASS: `ONE_NS1=2` changes only mode 49 (3416 instructions, R115,
no spills). Same clauses as check 4: (1) ABCHECKSUM; (2) G(X) - G(ctrlb) >= 0.20% and G(own twin) -
G(ctrl2) >= 0.20%; (3) same sign on both inputs at d16 and d32 full; (4) no row beyond max(|ctrlb -
ctrl|, 1%). This is the last check of the day: a passing arm still needs a confirmation bench and a
judged check after the reset before it can be offered.

### Check a7e5331d89ba4f77 (bench `bench_n`, at 5cb2be0): a two-stage ring narrows the pair spread and loses the median pace

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-30 | a7e5331d89ba4f77 (check) | 5cb2be0 | bench `bench_n` (`flat/gen_n.sh`, 10 arms, `--shapes full`) | 2.506x (CI 2.469-2.544) judged path = board code, 16/16 tests, ABCHECKSUM 11/0 every arm | ns2 +3.7 / +8.9 us at d16 / d32 full |

Profiles: `muse_attention_decode/profiles/bench_n.a7e5331d89ba4f77.*`. Cold pooled, arm - ctrl (us):
d16 full (ctrl 100.61) ns2 +3.65, ns2h +3.39, ns2own +3.68, ns2hown +3.42, ctrlb +0.00, ctrl2 +0.00,
pub -0.87; d32 full (ctrl 179.26) ns2 +8.87, ns2h +8.80, ns2own +8.77, ns2hown +8.64, ctrlb +0.20,
ctrl2 +0.07, pub -0.96. Gate: fails clause 2 for every arm (G -1.2 to -1.3%). Stamps (`stampns2`, d32
full): pair gap 3.2 us (6.2 at three stages in check 1, 4.8 at six in check 4), loop ends 169.9 /
178.2 / 183.0, consumer full-barrier wait 75.7 us: the spread narrows, and the median CTA slows by
more because two stages do not cover the DRAM latency.

Ring depth at d32 full over checks 1, 4 and 5 (median loop end; pair gap; cold time against the
three-stage control): two stages 178.2, 3.2 us, +8.9; three 161.3, 6.2, 0; six 167.1, 4.8 (latest SM
176.9), +2.2. Three stages, the depth the public kernel also uses, is the best of the five tried.

## Standalone round 4 (10-30): family ledger update

Checks used by this round: bce37d22a0ec4045, c8e99e038d9f4575, 4b60264a276c4805, 42c4739cf6b247c4,
a7e5331d89ba4f77 (5 of 5). Quota after them (`okc problems`): 0 of 20 attention checks, 8 of 10
runs. No file was offered for submission: nothing passed its gate.

| family | state | evidence | reopening trigger |
|---|---|---|---|
| Key-half merge in the flat q_len 1 kernel (`ONE_HMRGF`) | measured gain below the bar, off | bce37d22a0ec4045 (+0.04-0.07%), 4b60264a276c4805 (+0.05-0.07%), same sign on every input; review sound | promote with any flat q_len 1 change that passes (it adds about 0.05%); or a lower bar for riders |
| Mode 49 Q.K as two chains (`ONE_F1` bit 1) | measured loss | c8e99e038d9f4575: +0.12 / +0.32 us | a stamp showing the consumers' per-tile latency on the critical path |
| Mode 49 page-entry refill from one base (`ONE_F1` bit 2) | measured null | c8e99e038d9f4575: -0.04 / +0.03 us | a producer-bound probe showing its issue as the limit |
| Mode 49 ring depth (`ONE_NS1` 2, 4, 5, 6) | measured loss, 3 is best | 42c4739cf6b247c4, a7e5331d89ba4f77 | a change that equalizes the two CTAs' DRAM share |
| The public kernel's page-index bulk copies in mode 49 | untested, motivated | 4b60264a276c4805: mode 49's producer alone is 4.7 / 8.0 us (14%) slower than the public producer at d16 / d32 full without DRAM; the refill's instruction count is not it (PE16 null) | next round's first q_len 1 candidate |
| v16 full (mode 51) | attributed, no change | bce37d22a0ec4045 stamps: same median loop end as the public kernel, pair gap 11.5 vs 7.0 us, tail 12.3 vs 10.0; the gap depends on the input (-5.5 / -1.5 us in two benches); mode 51 keeps the per-tile proxy fence (1.2-1.4 us) | a per-input attribution of the input-0 case |

Where the flat gap is (measured): not in the merge, the start or the median CTA's pace. In both flat
kernels the two CTAs of an SM progress unequally and the later one sets each SM's end; ours more
unequal than the public kernel's (q_len 1: 3.1 / 6.2 us vs 1.0 / 2.1 at d16 / d32 full; q_len 8: 11.5 /
22.3 vs 7.0 / 16.0). Without DRAM, mode 49's producer and consumer streams each trail the public ones
by about half of a 9.5 / 17.5 us deficit. Ring depth trades the median pace against the spread and
three stages is the best point. Next round, in order: (1) bulk-copied page indices in mode 49's
producer (the public design; its producer-only pace is the measured target, 4.7 / 8.0 us), judged
first by the producer-only probe and then by cold time; (2) the consumer half of the no-DRAM deficit
(SASS of the mode 49 tile body against the public one, instruction by instruction on the path from the
full-barrier wait to the empty arrive); (3) the input dependence of v16 full.

## F-ATTN-0 (10-31): the complete donor attention file against our release

Steering: docs/steering/2026-10-30-family-f-donor-baseline.md (F-ATTN, priority 0) and its parent
donor-first report. Donor `alygg77/okc-muse-spark-kernels` at 6cae49a, `kernels/muse_attention_decode.py`
(sha256 e6d929d4..., 3182 lines; its RESULTS.md: judge job 1c8d81a0dbd94ee0, 2.6530x, an archived
donor result, not ours). The donor's attention file is vjeux's public file submitted unchanged:
`diff` against our `pub` arm (sha1 0e9d4c78, the public kernel of every bench since d57d1fe064164611)
shows only the two added attribution comment lines. So F0 for attention is the public kernel, and
F0 vs our release is the comparison our scorecards have tracked (latest all-shape public lead about
1%: rows listed in the round 3 ledger), now run under the F protocol. Our frozen release: board
entry 1896e72c11ae4b0f, `submission.py` at a70192d (identical to main 15cc6b9's).

Eligibility finding (for the coordinator): the F protocol says not to select favorable allocation
layouts to improve a judged score. Our release's `_ONE_WS_PICK = 3` times three workspace
allocations at state creation and keeps the fastest (measured 0.3-0.4 us at d1 full when the first
allocation lands slow). That is an allocation-layout selection. This comparison therefore carries a
release arm with the pick off (`_ONE_WS_PICK = 1`) so its weight in the result is visible; whether
the release may keep it is a coordinator / user decision.

### Decision rule (written before the check)

Bench `bench_o` (`flat/gen_o.sh`; generated from the a70192d `submission.py`; `--null --shapes all
--inputs 3 --rounds 150 --seed 20261101`): R1 `ctrl` and R2 `ctrl2` (the release, independent states),
`rnp` (the release with `_ONE_WS_PICK = 1`, own state), D1 `pubf0` and D2 `pubf02` (the exact donor
file, independent modules and workspaces). Cold medians pooled over three held-out inputs; the
weighted standalone ratio is ABSCORE (prod over loads of C_ctrl / C_arm, ^(1/6)); warm (16 queued
replays) reported alongside.

- D wins overall if ABSCORE(pubf0) - 1 and ABSCORE(pubf02) - 1 both exceed |ABSCORE(ctrl2) - 1| +
  |ABSCORE(pubf02) / ABSCORE(pubf0) - 1| and 0.3%, and ABCHECK passes for both D arms.
- Per row, a side wins a shape if both D arms differ from both R arms in the same direction by more
  than that row's larger null (|ctrl2 - ctrl|, |pubf02 - pubf0|). Rows won by each side define the F2
  hybrid proposal (not built in this round).
- Otherwise inconclusive. The judge checks of the exact files (two of F0, one or two of the
  release) give the judged ratios; they are compared by the mean of each pair against the judge's
  1-3% run-to-run spread, and decide nothing alone.

### Check 0519d0b15e9c4583 (bench `bench_o`, generated from release a70192d): the donor file leads beyond both nulls

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-31 | 0519d0b15e9c4583 (check) | 589fb00 (bench from a70192d) | bench `bench_o` (`flat/gen_o.sh`, 5 arms, all 12 shapes, 3 inputs) | 2.512x (CI 2.474-2.550) judged path = release, 16/16 tests, ABCHECKSUM 19/0 every arm | ABSCORE donor 1.0097 / 1.0083 against the release |
| 2026-10-31 | 45adaeddef9d464e (check) | donor 6cae49a | the exact donor file (sha256 e6d929d4) | 2.574x (CI 2.535-2.628) | judged F0 |
| 2026-10-31 | cb99c110c9cd4c01 (check) | a70192d | the exact release file | duplicate of 640ee49b691b45ac (2.588x on 10-30), no run | |
| 2026-10-31 | b90af2a967b946b1 (check) | a70192d + one comment line | release, same-day control | 2.542x (CI 2.504-2.580) | judged R |
| 2026-10-31 | ab38543d2c8642fb (check) | donor + one comment line | donor repeat | 2.605x (CI 2.566-2.644) | judged F0 |

Profiles: `muse_attention_decode/profiles/bench_o.0519d0b15e9c4583.*`, `release_r.b90af2a967b946b1.*`,
`f0_donor.45adaeddef9d464e.json`, `f0_donor_r.ab38543d2c8642fb.json` (result receipts only; no donor
source is committed). Cold medians pooled over three held-out inputs (us):

| row | R1 ctrl | R2 ctrl2 | rnp (pick off) | D1 pubf0 | D2 pubf02 | D - R (means) | side |
|---|---|---|---|---|---|---|---|
| d1 sliding | 9.06 | 9.28 | 9.15 | 9.06 | 9.06 | -0.11 | null |
| d1 full | 16.99 | 17.02 | 17.18 | 16.48 | 16.58 | -0.47 | donor |
| d16 sliding | 12.83 | 12.93 | 12.90 | 12.80 | 12.96 | +0.00 | null |
| d16 full | 100.38 | 100.45 | 100.38 | 99.52 | 99.52 | -0.89 | donor |
| d32 sliding | 15.62 | 15.62 | 15.55 | 15.58 | 15.55 | -0.05 | donor (0.05 us) |
| d32 full | 179.42 | 179.65 | 179.42 | 177.95 | 178.08 | -1.52 | donor |
| v1 sliding | 11.01 | 10.91 | 10.91 | 10.85 | 10.88 | -0.10 | null |
| v1 full | 19.04 | 19.14 | 19.01 | 19.04 | 18.98 | -0.08 | null |
| v16 sliding | 15.68 | 15.58 | 15.52 | 15.49 | 15.52 | -0.12 | null |
| v16 full | 112.61 | 113.73 | 112.61 | 111.04 | 111.04 | -2.13 | donor |
| v32 sliding | 19.97 | 19.90 | 19.78 | 19.30 | 19.36 | -0.61 | donor |
| v32 full | 193.70 | 194.02 | 193.28 | 193.25 | 193.44 | -0.52 | null |

ABSCORE against ctrl: ctrl2 0.9964 (R null 0.36%), rnp 1.0003, pubf0 1.0097, pubf02 1.0083 (D null
0.14%). Warm (16 queued replays) the donor is faster on all 12 rows. By the rule written before the
check, the donor wins overall: both D arms exceed 1 by more than 0.50% (the sum of the nulls) and
0.3%. It wins six rows and the release wins none, so an F2 hybrid would take nothing from the
release. Judged, same day: release 2.542x, donor 2.574x / 2.605x (means 2.542 vs 2.590, +1.9%, inside
the judge's 1-3% run spread but in the direction and about the size the bench measured). The pick
(`rnp` against ctrl) moved d1 full by +0.19 us, inside the R null; ABSCORE 1.0003.

What the donor's lead is made of: on the per-request q_len 1 / 8 rows it is gone after HMRG and PRQ
(every per-request row null except d1 full, which is placement-sensitive); it sits in the flat full
rows (d16 / d32 / v16 full) that round 4 attributed to the per-SM pair spread, plus v32 sliding and
d1 full.

### User decisions recorded 2026-10-31 (relayed by the coordinator)

1. Remove the workspace placement pick: `_ONE_WS_PICK = 0` (default allocation), everything else
   unchanged (HMRG, PRQ=31), so the board entry does not depend on a selected placement.
2. Family F is measurement and learning only: never submit donor code. Where the donor wins,
   re-implement the mechanism in our own code with attribution comments. For attention the donor
   file is vjeux's public file, so F-ATTN-0 is the public-kernel comparison this log has tracked
   since d57d1fe064164611; the rows above are its current state.

The pick-off release is the control from now on.

### Check 3250ab04126b4865: the pick-off release

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-31 | 3250ab04126b4865 (check) | dcda02e | `submission.py` with `_ONE_WS_PICK = 0` | 2.537x (CI 2.499-2.575), passed | handed to the coordinator for a scored run |

The host code differs from release a70192d only in the pick line (plus its comment); the inlined
kernel equals `attn_one.cu`, whose default build is SASS-identical to main 15cc6b9 on all 13
instances (`flat/sass_all.sh`, last run with the round 4 switches in). Same-day release with the pick:
2.542x (b90af2a967b946b1), so turning the pick off costs nothing visible in the judge.
Profiles: `muse_attention_decode/profiles/sub_nopick.3250ab04126b4865.*`.

Next for attention, by the evidence above: the flat full rows (d16 / d32 / v16 full, about 0.4% of
the donor's 0.9%), re-implementing in our kernel what makes the public kernel's two co-resident CTAs
share an SM more evenly (round 4: bulk-copied page indices in mode 49's producer first), then v32
sliding (-0.6 us) and d1 full.

### Scored: 68d3b88fbc8b40da 2.537x (2026-10-31)

Submit of submission.py at dcda02e (`_ONE_WS_PICK = 0` per the user's decision, ONE_PRQ=31,
ONE_HMRG on; kernel SASS identical to a70192d on all 13 instances; no public code), submitted by
the coordinator under the standing authorization. Passed, 2.537x (CI 2.499-2.575), above
1896e72c11ae4b0f (2.521x), so the board entry no longer uses the workspace pick. Receipt:
muse_attention_decode/profiles/scored.68d3b88fbc8b40da.json.

## Attention round 5 (10-31): page indices by bulk copy in mode 49 (`ONE_PBULK`)

Brief (coordinator, 10-31): the pick-off file scored 68d3b88fbc8b40da (2.537x) and is the board entry
and the control. Rank 100 is 2.6289, about 3.6% above it. Start with page indices fetched by bulk
copy in mode 49's producer, in our own code with attribution, aimed at the flat full rows where the
public / donor kernel leads (d16 / d32 / v16 full); then v32 sliding and d1 full. Budget: up to 10
attention checks today. Rules unchanged: paired same-process evidence, every fence kept, no submits.

### The switch: `ONE_PBULK` (default 0, mode 49 only)

Mechanism (the public flat q_len 1 kernel's `fetch_pages` / `load_pages_t`, vjeux okc-2026,
Apache-2.0; re-implemented for our ring, attribution in the source): producer thread 0 issues one
`cp.async.bulk` per tile of the 16-byte-aligned window of page entries that covers the tile's cached
rows (at most 68 entries) into one of 4 shared windows (`Smem::PSL`, 320 bytes each, an mbarrier
each at `Smem::PBAR`), three tiles ahead, armed by `mbarrier.arrive.expect_tx`. Every producer thread
waits on the window's phase, reads its 4 entries from shared memory, then issues its K / V copies
exactly as before; after its `cp.async.mbarrier.arrive.noinc` the producers `bar.sync 4, 128`, which
frees the window the tile used. Tiles with no cached rows, a window past the table's end, or a
table base not 16-byte aligned take the old `__ldg` entries; the predicate is the same for the
issuing thread and the readers, so fills and waits pair one to one per window. This replaces 16
warp-level `LDG` per tile per CTA (4 per producer thread, three tiles ahead in 12 registers) with one
bulk copy, four `LDS` per thread and one producer barrier.

Ordering: nothing is removed or moved. Added: the window's RAW (bulk write, `complete_tx`, phase
observed by `mbarrier.try_wait.parity`, then generic loads) and WAR (every producer's generic loads,
then `bar.sync 4`, then thread 0's `fence.proxy.async.shared::cta` before the async-proxy refill,
three tiles later). The K / V ring, its full / empty mbarriers and the consumers are unchanged. The
`fence.proxy.async` before the refill is one more fence than the public kernel uses.

SASS (CUDA 12.8.1, `flat/sass_all.sh`, all 13 instances): `ONE_PBULK=0` identical to main 15cc6b9;
1 changes only mode 49 (3448 -> 2912 instructions: the three-way unrolled register lookahead is
gone; 1 `UBLKCP`), consumer region R113, no spills.

### Gate for check 1 (written before the bench that tests it)

Bench `bench_p` (`flat/gen_p.sh`; from the pick-off `submission.py`; `--pub <0e9d4c78> --null
--shapes full --inputs 3 --rounds 200 --seed 20261102`): `ctrl` (the board entry's code), `ctrl2`
(independent null), `ctrlb` (`-DONE_PBULK=0`, matched null), `pb` (ctrl's state), `pbown` (own state),
`pbh` (`ONE_PBULK=1 ONE_HMRGF=1`), the producer-pace probes `prodonly_probe` (`ONE_PROBE=3`) and
`pbprod_probe` (`ONE_PBULK=1 ONE_PROBE=3`), `stamppb` (stamps), `pub`, `pub2`. G over d16 / d32 full
at 0.147 / 0.084 %/us.

`pb` (or `pbh`) goes to a confirmation bench if all hold: (1) ABCHECKSUM passes for every real arm
(the seven tp8 test shapes include page size 64 and the batch-4 full shape, so `p.lp` > 0 and the
mode 49 test path are covered); (2) G(X) - G(ctrlb) >= 0.10% and G(Xown or pbown) - G(ctrl2) >=
0.10%; (3) same sign on all three inputs at d16 and d32 full; (4) no row slower than ctrl by more
than max(|ctrlb - ctrl|, 1%). The bar is lower than round 4's 0.20%: the coordinator's criterion is
beating the board entry on paired evidence, and each confirmed 0.1% counts toward the 3.6%; the
confirmation bench and two judged checks still follow before any file is offered.
Diagnostic: `pbprod_probe` against `prodonly_probe` says whether the producer-only pace (4.7 / 8.0
us behind the public producer in 4b60264a276c4805) moved.

Independent read-only review of `ONE_PBULK` (fresh agent, 10-31, source and SASS layout): **sound**, with
caveats: the local tile counter `g` shadows the outer `g = lane >> 2` (unused there); `pt_total`
assumes a contiguous table of at least `batch` rows, as the old path does; the per-tile producer
`bar.sync` is a performance question.

### Check fcf904b32d9a427b (bench `bench_p`, at 03675e8): `pb` hung in ABCHECK

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-31 | fcf904b32d9a427b (check) | 03675e8 | bench `bench_p` | failed: the bench's HANG guard ended the process | `HANG pb b16 ctx80000 tp8 full L1 ... t=366s: device busy for 20s`; ctrl and ctrlb had passed ABCHECK on that shape |

Profiles: `muse_attention_decode/profiles/bench_p.fcf904b32d9a427b.*`. No timing; nothing judged; the
board is unaffected. The first launch of the `ONE_PBULK=1` build at decode 16 full did not finish.
A static read of the source and of the mode 49 SASS (slot index, parity bits in a uniform register,
`SYNCS.ARRIVE.TRANS64` with the same byte count as the `UBLKCP`, the elect loop around it, the warp
reconverged at `BAR.SYNC 0x4, 0x80`) did not find the cause, and neither did the review.

### Check 2 (diagnostic, no gate): which piece hangs

Bench `bench_q` (`flat/gen_q.sh`, `--shapes full --inputs 1 --rounds 60`): ABCHECK runs the arms in
order and the HANG guard stops at the first hang. `dbg2` (`ONE_PBULK=1 ONE_PBDBG=2`: the new loop
and its per-tile producer `bar.sync`, no fills, no window waits, entries by the old loads), `dbg1`
(`ONE_PBDBG=1`: fills and window waits run, entries still by the old loads), `pb` (entries from the
windows). A hang in `dbg2` puts it in the loop structure or the producer barrier; in `dbg1`, in the
bulk copy / mbarrier protocol; only in `pb`, in the entries read from the windows (wrong entries
feeding the K / V copies). SASS: both debug builds change only mode 49, no spills.

### Check 61069efbf90d4c44 (bench `bench_q`, at c811908): the hang is in the loop structure; root cause found offline

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-31 | 61069efbf90d4c44 (check) | c811908 | bench `bench_q` (diagnostic) | failed by design at the first hang: `HANG dbg2 b16 ... full L1` | `dbg2` has no bulk copy and no window wait |

Profile: `muse_attention_decode/profiles/bench_q.61069efbf90d4c44.log`. Since `dbg2` hung, the cause lay
outside the bulk-copy protocol. Offline bisection (Docker, CUDA 12.8.1, mode 49 only):
- In every `ONE_PBULK=1` build, mode 49's SASS had no producer `EXIT` and no producer drain; the
  producer loop's exit edge fell straight into the consumers' `USETMAXREG.TRY_ALLOC 0xc8`. The
  PTX had no `exit;` at all, so the front end, not ptxas, had treated the code after the loop as
  unreachable. With the producer warps running into consumer code, the kernel hung.
- Removing the producer `bar.sync`, the fills, the window reads, or the lambdas: still no `EXIT`.
  The PBULK branch holding a verbatim copy of the old loop: still no `EXIT`. `if constexpr
  (false)` in place of `if constexpr (PBULK)` with the old loop in the `else`: `EXIT`s present.
- So the trigger was the shape `if constexpr (PBULK) { ... } else for (...) { ... }`: with the
  condition true, the statements after the discarded, unbraced `else` statement (the drain and
  the inline-asm `exit`) were dropped. With the `else` braced (`} else { for (...) {...} }`), mode
  49 has its five `EXIT`s again, in the default build's pattern. The other `} else` sites in the file
  sit inside `#if ONE_DETM` (off) and are followed by braced blocks.

The fix carries a comment at the `else`. SASS after it (`flat/sass_all.sh`): `ONE_PBULK=0`
identical to main 15cc6b9 on all 13 instances; `ONE_PBULK=1` changes only mode 49 (2832
instructions, one `UBLKCP` in the prologue fill and one in the loop, `BAR.SYNC 0x4, 0x80`, producer
`EXIT`s back), consumer region R115, no spills. The ordering argument and the review above are
unchanged (the brace changes no statement). Check 3 repeats `bench_p` unchanged on the fixed source,
under the check-1 gate as written.

### Check bb06f8e57bd84636 (bench `bench_p` repeated, at 0671cc7): `ONE_PBULK=1` is a small gain below the bar

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-31 | bb06f8e57bd84636 (check) | 0671cc7 | bench `bench_p` (11 arms, `--shapes full`, 3 inputs) | 2.496x (CI 2.459-2.534) judged path = board code, 16/16 tests, ABCHECKSUM 11/0 every real arm | G +0.063% over each null, bar 0.10% |

Profiles: `muse_attention_decode/profiles/bench_p.bb06f8e57bd84636.*`. Cold pooled over three inputs,
arm - ctrl (us):

| row | ctrl | ctrlb | ctrl2 | pb | pbown | pbh | pub | prodonly_probe / pbprod_probe (abs.) |
|---|---|---|---|---|---|---|---|---|
| d16 full | 100.45 | +0.03 | +0.06 | -0.07 | -0.13 | -0.19 | -0.80 | 37.60 / 47.33 |
| d32 full | 179.42 | +0.07 | -0.09 | -0.51 | -0.51 | -0.38 | -1.34 | 63.42 / 82.75 |

Against the gate: (1) holds; (2) **fails**: G(pb) - G(ctrlb) = +0.063%, G(pbown) - G(ctrl2) = +0.063%,
under 0.10%; (3) d32 full has the same sign on all three inputs (-0.61 / -0.67 / -0.32), d16 full does
not (+0.03 / -0.13 / -0.16); (4) holds. A measured gain below the bar, not promoted.

The producer-only probe got slower with the switch (37.6 -> 47.3 and 63.4 -> 82.8 us): the per-tile
producer `bar.sync` locks the four producer warps together, and each waits for its window and for the
empty stage in lockstep. The stamps (`stamppb`, d32 full) show the producer's empty-barrier wait down
from about 95 to 57 us and the pair gap unchanged (6.2 us). So the real-time gain comes while
DRAM sets the pace, and the lockstep costs the producer pace without DRAM.

### The variant for check 4: `ONE_PBULK=2` (free mbarriers in place of the producer barrier)

Each window gets a second mbarrier (`Smem::PFREE`, count 128). Every producer thread arrives on it
after reading its 4 entries; thread 0 waits on it before refilling the window (fill k waits for phase
k - 1; fill 0 passes at once) and then executes `fence.proxy.async.shared::cta` and the bulk copy. The
per-tile `bar.sync 4` is gone, so the producer warps no longer run in lockstep. Ordering of the WAR:
each reader's generic loads, then its `mbarrier.arrive` (release, CTA scope), then thread 0's
`mbarrier.try_wait.parity` (acquire), then the proxy fence and the async-proxy write. The RAW is
unchanged. A refill cannot overtake: fill k + 1 is issued only after the 128 arrivals for fill k, and
readers of fill k + 1 arrive only after it lands. No fence or barrier of the K / V ring changes.
SASS: `ONE_PBULK=2` changes only mode 49 (2864 instructions, consumer region R119, no spills,
producer drain and `EXIT` present); 0 is still identical to main on all 13 instances.

### Gate for check 4 (written before the bench that tests it)

Bench `bench_r` (`flat/gen_r.sh`, `--pub <0e9d4c78> --null --shapes full --inputs 3 --rounds 200
--seed 20261104`): `ctrl`, `ctrl2`, `ctrlb` (`-DONE_PBULK=0`), `pb1`, `pb2`, `pb2own`, `pb2h` (with
HMRGF), `prodonly_probe`, `pb2prod_probe`, `pub`, `pub2`. The check-1 clauses apply to `pb2` (with
`pb2own` as its own-state twin) and to `pb2h`: (1) ABCHECKSUM; (2) G(X) - G(ctrlb) >= 0.10% and
G(pb2own) - G(ctrl2) >= 0.10%; (3) same sign on all three inputs at d16 and d32 full; (4) no row beyond
max(|ctrlb - ctrl|, 1%). `pb1` repeats check 3's arm for the ledger.

Independent read-only review of `ONE_PBULK=2` (fresh agent, 10-31, source at 8bdf607): **sound**;
the caveat (two comments that still named the producer `bar.sync` as the only ordering) is fixed in
the source (comments only, no SASS change).

### Check f3f1e4a5249a4228 (bench `bench_r`, at 8bdf607): `ONE_PBULK=2` loses; `ONE_PBULK=1` repeats its small gain

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-31 | f3f1e4a5249a4228 (check) | 8bdf607 | bench `bench_r` (11 arms, `--shapes full`, 3 inputs) | 2.519x (CI 2.481-2.570) judged path = board code, 16/16 tests, ABCHECKSUM 11/0 every real arm | pb2 +1.4 / +2.9 us; pb1 G +0.074% |

Profiles: `muse_attention_decode/profiles/bench_r.f3f1e4a5249a4228.*`. Cold pooled, arm - ctrl (us):

| row | ctrl | ctrlb | ctrl2 | pb1 | pb2 | pb2own | pb2h | pub | prodonly / pb2prod (abs.) |
|---|---|---|---|---|---|---|---|---|---|
| d16 full | 100.42 | +0.03 | +0.00 | -0.16 | +1.37 | +1.34 | +0.22 | -0.96 | 37.57 / 53.73 |
| d32 full | 179.58 | +0.04 | -0.12 | -0.60 | +2.92 | +2.98 | +0.71 | -1.24 | 63.49 / 95.39 |

Gate: `pb2` and `pb2h` fail clause 2 (G -0.45% / -0.09%) and `pb2` clause 4 (beyond 1% at d32).
`pb1` again: G +0.074%, d32 -0.48 / -0.67 / -0.48 on the three inputs, d16 +0.00 / -0.26 / -0.13.
Two benches put `ONE_PBULK=1` at +0.06-0.07% (rows in docs/ruled-out.md: a real gain below the bar;
`ONE_PBULK=2` a loss). The free-barrier variant makes thread 0 wait each tile for the slowest
producer's read of the previous fill before refilling, which slowed the producer more than the
lockstep did (producer-only 53.7 / 95.4 us against 47.3 / 82.8).

### Check 5 (attribution, no gate): the per-request rows by phase, with HMRG and PRQ in

Outside the flat rows the donor / public lead is d1 full (0.47 us, about 0.28%), v32 sliding (0.61
us, 0.13%) and d1 sliding. The last phase comparison (ce92d32e26024bd2) predates HMRG's promotion
and PRQ. Bench `bench_s` (`flat/gen_s.sh`, `--null --shapes all --inputs 2 --rounds 120 --seed
20261105`): our DIAG phase exits `p9_probe` .. `p12_probe` and `diag0_probe` (own state) against the
public phase copies `pubph1_probe` .. `pubph4_probe` (scratchpad `a3/pubph0.py`, as in round 2),
`ctrl2`, `pub`, `pub2`, and `pb1` a third time.

### Check 1c861d3944f149a0 (bench `bench_s`, at a191c34): where the per-request gap is now

| date | job | commit | file | result | notes |
|---|---|---|---|---|---|
| 2026-10-31 | 1c861d3944f149a0 (check) | a191c34 (bench generated one comment edit earlier; same code) | bench `bench_s` (15 arms, all 12 shapes, 2 inputs) | 2.584x (CI 2.546-2.635) judged path = board code, 16/16 tests, ABCHECKSUM 19/0 every real arm | attribution, no gate |

Profiles: `muse_attention_decode/profiles/bench_s.1c861d3944f149a0.*`. Phase times, cold pooled (us),
ours (`p9` / `p10` / `p11` / `p12` / `diag0`, own state, DIAG build of the board code) against the
public phase copies (`pubph1` .. `pubph4`, `pub`). Each column is the difference of consecutive
exits, so a red's drain lands in the phase that issued it:

| row | prologue | tile loop | reds | ticket / barrier | finalize | whole |
|---|---|---|---|---|---|---|
| d1 sliding | 7.10 / 6.98 | 0.64 / 0.54 | 0.32 / 0.38 | 0.80 / 0.64 | 0.77 / 0.74 | 9.63 / 9.28 |
| d1 full | **10.27 / 9.73** | 4.58 / 4.48 | 0.41 / 0.35 | 1.19 / 1.47 | 0.57 / 0.61 | 17.02 / 16.64 |
| d16 sliding | 10.34 / 10.27 | 0.99 / 0.67 | 0.38 / 0.52 | 0.83 / 0.96 | 0.96 / 0.54 | 13.50 / 12.96 |
| d32 sliding | 10.40 / 10.24 | 3.58 / 3.49 | 0.36 / 0.38 | 0.70 / 0.67 | 0.90 / 0.93 | 15.94 / 15.71 |
| v1 sliding | 7.55 / 7.17 | 1.19 / 1.09 | 0.80 / 0.99 | 0.99 / 0.80 | 0.73 / 0.83 | 11.26 / 10.88 |
| v1 full | 9.82 / 9.76 | **6.21 / 5.79** | 1.35 / 1.35 | 1.15 / 1.05 | 1.02 / 1.06 | 19.55 / 19.01 |
| v16 sliding | 8.93 / 8.77 | **4.35 / 3.93** | 0.74 / 0.84 | 1.21 / 0.83 | 0.77 / 1.28 | 16.00 / 15.65 |
| v32 sliding | 9.18 / 9.06 | **8.13 / 7.26** | 0.74 / 0.90 | 1.28 / 0.92 | 1.09 / 1.28 | 20.42 / 19.42 |

What it names (measured):

- **q_len 8 per-request tile loop**: 0.42 / 0.42 / 0.87 us behind at v1 full / v16 sliding / v32
  sliding after PRQ. At this run's sensitivities that is about 0.20% + 0.15% + 0.18%, the largest
  pool left (the flat full rows hold about 0.25%). Round 3 found the public per-request tile body
  longer than ours in instructions (645 vs about 530 before PRQ), so the remaining difference is
  in latency or order, not count; its per-tile descriptor and V^T address carries (the per-request
  counterpart of DCARRY / VHOIST, which ours applies only to the flat mode 51) are one untested
  candidate.
- **d1 full prologue**: 10.27 vs 9.73 us (+0.54, about 0.33%). The q_len 8 batch-1 full prologue
  (same mode 11 plan, same grid) is level (9.82 vs 9.76), and the sliding prologues trail by 0.07-0.38.
  So the deficit is specific to the q_len 1 batch-1 full prologue; the public phase copy has no
  sub-phase exits, so which load in the chain it is has not been measured.
- Ticket / barrier and finalize trade a few tenths both ways (ours faster at d1 full's barrier,
  slower at v16 / v32 sliding's barrier and faster in their finalize).
- With the pick off, the judged path's own state sat 0.3-0.5 us slow at d1 full in this process
  (ctrl 17.47, ctrl2 17.15, `pub` 16.67).
- `pb1` (`ONE_PBULK=1`) a third time: d16 / d32 full -0.10 / -0.80 us; ABSCORE 0.9974, which is inside
  the run's null spread on rows it does not change (ctrl2 0.9968).

## Attention round 5 (10-31): ledger

Checks used: fcf904b32d9a427b (hang), 61069efbf90d4c44 (diagnostic), bb06f8e57bd84636,
f3f1e4a5249a4228, 1c861d3944f149a0 (5 of the 10 budgeted). No file beat the board entry
(68d3b88fbc8b40da, the pick-off release), so none was offered.

| family | state | evidence | reopening trigger |
|---|---|---|---|
| Mode 49 page indices by bulk copy (`ONE_PBULK=1`) | measured gain below the bar, off | bb06f8e57bd84636 (+0.063%), f3f1e4a5249a4228 (+0.074%), 1c861d3944f149a0 (d32 -0.80 us); review sound | a rider for any promoted mode 49 change, or a producer that keeps it without the per-tile lockstep |
| Same with per-window free barriers (`ONE_PBULK=2`) | measured loss | f3f1e4a5249a4228: +1.4 / +2.9 us | a refill that does not wait for the slowest reader of the previous fill |
| `if constexpr (...) { } else for (...)` (unbraced `else` statement) | compiler trap, fixed by bracing | 61069efbf90d4c44 plus offline bisection: CUDA 12.8.1 dropped the producer drain and `exit` | any new `if constexpr` with an unbraced `else`: check the producer `EXIT` count in SASS |

Next, in order of measured value: (1) the q_len 8 per-request tile loop (0.4-0.9 us on v1 full, v16
and v32 sliding, about 0.5% together); first test the per-tile descriptor / V^T address carries in
the PRQ loop against the public order, with a stamp copy of the public per-request loop to split the
latency; (2) the d1 full prologue (0.54 us): sub-phase exits on both sides (page indices, query
group, first tiles) before any change; (3) the flat full rows by the pair spread (round 4).

## October 31 PDT, post-restart: PRQ address carry hypothesis

Authenticated Ubuntu okc 0.7.2; account reports 20/20 checks per problem before dispatch
(shared quota, not a reservation). Control remains pick-off entry `68d3b88fbc8b40da`, PRQ31.

One mechanism: reduce per-tile address reconstruction in the per-request verify loop. New
bench-only PRQ bits32/64 reuse the flat loop's address identities:32 carries the Q.K descriptor
from stage0;64 toggles V^T addresses/descriptor words by precomputed XOR and hoists transpose
offsets. Default31, submission.py and production candidates are unchanged. Precision, memory
operations, waits, fences and synchronization retain their order. V^T toggles only after the
existing `wg_wait<0>()` and register fence.

Check `c77ce71c81af4fb1`, pending at dispatch, problem confirmed `muse_attention_decode`.
Bench SHA256 `2fa39d4bae82fdfe1d1e1793fc0207d2a2a97e512d72409524797b809863f1be`,350281bytes,
generated outside the repository at `/tmp/okc-top10/attn_prqcarry.py`:

```
python3 tools/attn_make_bench_ab.py /tmp/okc-top10/attn_prqcarry.py --null --shapes all --seed 311031 --inputs 2 --rounds 80 kcarry@own=-DONE_PRQ=63 vcarry@own=-DONE_PRQ=95 both@own=-DONE_PRQ=127
python3 tools/attn_inline_one.py /tmp/okc-top10/attn_prqcarry.py
```

Independent compiled/prepared control duplicate, independent state per candidate, all12 shapes,
two held-out timing inputs, randomized paired cold-L2 graph replays, judge-style/relative-output
checks. Official tests run the unchanged control; a winning experimental arm still needs
candidate-only official tests and queued replay stress. Acceptance: repeatable score gain
beyond independent null variation, localized to changed verify shapes, then layer confirmation.
Stop/revert if the mechanism fails its gate; headline control score is not candidate score.

### Result: `c77ce71c81af4fb1`, no promotion

H200, completed666.7s after start. Official control tests16/16 pass, control score2.524717x,
95%CI2.486846–2.562588. Every arm's harness checks19/19 pass. All arms select the same launch
plans; no new SM-utilization, register or spill counters were collected. Receipts/log/patch:
`profiles/prqcarry.c77ce71c81af4fb1.{json,log,patch}`. The raw status JSON retains all benchmark
and profile records. No candidate-only judge score or queued stress is claimed for the arms.

Cold medians (microseconds; raw graph times include replay overhead), candidate minus control:

| shape | control | QK carry | V carry | both | identical ctrl2 |
|---|---:|---:|---:|---:|---:|
| v1 sliding |11.17 |+0.06 |+0.00 |+0.13 |-0.03 |
| v1 full |19.20 |-0.22 |-0.22 |-0.38 |+0.03 |
| v16 sliding |15.58 |+0.03 |+0.06 |+0.00 |+0.10 |
| v32 sliding |20.10 |+0.06 |+0.03 |-0.19 |+0.03 |

Raw ABSCORE ratios versus control: QK1.0020, V1.0021, both1.0004, identical ctrl2 1.0008.
The single-arm headline gains largely occur on unchanged decode instances (e.g. d1 full
QK/V -0.26us, exactly ctrl2's -0.26us), so cannot be attributed to address carry. Both's aggregate
is smaller than the identical-control variation. v16 sliding flips sign by held-out input
(both -0.19/+0.19us); v1 full both -0.38us on each input, v32 sliding -0.16/-0.13us, a small
localized signal that does not establish a beneficial general change.

Scoring qualification: ABSCORE uses cold raw medians. Recomputing from rounded ABSUM rows with
each official scored shape's replay overhead subtracted gives approximate QK1.002914,
V1.003433, both1.000519, ctrl2 1.001257. These are sensitivity estimates, not candidate scores:
the AB graphs did not separately measure their replay overhead. If unchanged instances are
held fixed to control and only the four affected per-request verify rows vary, the analogous
approximation is QK0.999968, V1.000749, both1.000438. No result supports a repeatable meaningful
score gain; no selective dispatch is promoted on these two-input microsecond readings.

Decision: revert experimental edits from attn_one.cu, retain exact patch and source commit
`fdaaa11` for reproducibility, keep PRQ31/submission.py/board entry unchanged. Park this
mechanism as a possible rider only after another per-request loop change makes its saved
instructions material, or a bounded shape-specialized test is supported by a larger gap.
Do not spend a second check repeating the same general toggle.

## Matched batch-1 prologue attribution, 2026-10-31 PDT

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-31 | 191e57a0f45a4717 (check) | 1f31181 | Matched owned/donor startup exits 1/2/3/5/7/9; generated by tools/attn_make_prologue_probe.py; judged path unchanged | pending | Batch-1 decode/verify, sliding/full; 3 held-out timing inputs, 100 cold rounds; independent control and donor null builds. Measurement only; never submit this donor-carrying bench. |

Baseline profile reused: `baseline_profile/status.24daba2202e547d6.json`; latest hotspot map
reconstructed from `profiles/prqcarry.c77ce71c81af4fb1.json` with `tools/hotspots.py --baseline`.
Pinned external donor SHA256 `e6d929d40e6cff8aa82270a423fca24dfccad64b6a82428db9c3377b661bf124`
(revision `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`), measurement-only copy outside git.
Probe phase exits drain outstanding copies and return before normal merge/finalize; output is
intentionally unguaranteed and excluded from correctness promotion. All phase arms of one side
share a CUDA build; control duplicate and donor duplicate have independent builds/states.
Donor runtime phase uses new Params field carried through unused launch-bitfield bits 20+;
normal donor/control phase 0 executes the complete call. No production source is changed.

Boundaries: 1 page-index copies committed; 2 query/new-row/constants copies committed;
3 initial page-index wait and barrier; 5 first tiles issued and query group waited/barrier;
7 query normalization/packing complete; 9 new-row handling/patching complete before tile loop.
These are drained early-exit kernel timings, not additive live-path timings: early drains
remove overlap and diagnostic branching can perturb compilation. Require a real A/B change
before claiming an optimization. Main question is whether the residual d1-full difference is
already present before query/new-row staging, or develops during it.

## Authorized scored control repeat, 2026-10-31 PDT

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-31 | cccc131c55874949 (submit by root coordinator) | d3eeeda + repeat comment | Owned best attention source; PRQ31, HMRG1, workspace pick off; authorized daily scored control repeat | 2.534850x (2.496827-2.572873) | Passed all 16/16 official tests. Twelve benchmark and twelve profile records saved in profiles/scored.cccc131c55874949.json. No performance promotion: overlaps prior 2.537x entry and its interval. |

Exact root file `.build/today/attention_best.py` SHA256
`4d36ddbc94ed81bfe95e3395555cbf0cf117bbe505d73528a18564ea76c454ca`;
source body is submission.py at d3eeeda after experimental address changes were reverted,
plus the trailing comment `# Authorized daily scored control 2026-10-31, repeat 1.`.
Selected kernel instances from receipt profiles: decode sliding batch1/16/32 modes13/1/1;
decode full11/49/49; verify sliding15/3/3; verify full11/51/51. Receipt identifies
board muse_attention_decode and id cccc131c55874949; no new resource counters were collected.

### Result of 191e57a0f45a4717

Correct owning receipt: board `muse_attention_decode`, id `191e57a0f45a4717`; passed16/16
official tests. Unchanged owned control scored2.514672x (95%CI2.476952-2.552392).
All four complete real arms passed11/11 harness checks (four batch-1 shapes plus seven
nearby tests); the twelve early-exit arms are explicitly probes, not correctness candidates.
No resource-utilization or SASS counters were collected. Receipt and retained log are
`profiles/prologue.191e57a0f45a4717.{json,log}`. All batch-1 full plans use252splits/nc8;
owned mode11 and donor per-request cooperative plans agree (donor verify full coop derived).

Cold drained-exit medians over three inputs, microseconds including graph replay overhead:

| Boundary | d1 full owned | d1 full donor | v1 full owned | v1 full donor |
|---|---:|---:|---:|---:|
|1 pages committed/drained|7.04|6.02|5.98|6.18|
|2 query/new-row/constants committed/drained|7.20|6.08|6.18|6.34|
|3 page wait and barrier|7.23|6.14|6.18|6.34|
|5 initial K/V tiles issued, query waited/barrier|9.79|9.41|9.57|9.50|
|7 query packed|9.82|9.38|9.57|9.50|
|9 new rows/patching before loop|9.79|9.41|9.57|9.54|
|complete diagnostic build|17.18|16.22|19.17|18.59|
|independent complete duplicate|16.96|16.48|19.14|18.78|

Measured: d1-full has an approximately1us difference already at the first drained page group,
shrinking to0.38us after initial tile issue. The corresponding v1-full startup has no comparable
deficit. Complete-call null spreads are0.22/0.26us at d1 full,0.03/0.19us at v1 full.
Do not subtract these drained phase medians as live pipeline costs: moving a drain destroys overlap.

Hypothesis selected: d1 SPEC query-address preparation may be hoisted before page issue;
the pinned donor explicitly re-reads tid after the page group to prevent such hoisting.
Test an owned opaque late-tid scheduling toggle while preserving address formulas and ordering.
New-KV-only staging is deferred: its omission cannot explain the deficit already present atphase1.
No optimization is promoted by this diagnostic.

## Late thread-index scheduling experiment, 2026-10-31 PDT

|date|job|commit|change|score (95%CI)|notes|
|---|---|---|---|---|---|
|2026-10-31|b4d75c3185744d0a (check)|fac5188|ONE_LATE_TID=1 for L1 SPEC staging, defaultoff|pending|Own control/candidate/independent duplicate over all12shapes,3heldoutinputs,100coldrounds; seed311102. No donor code in bench.|

Generator command: `python3 tools/attn_make_bench_ab.py /tmp/okc-attn-campaign/latetid.py
--null --shapes all --inputs 3 --rounds 100 --seed 311102 late@own=-DONE_LATE_TID=1`.
Acceptance: d1-full cold improvement on all3inputs beyond independent control variation,
weighted standalone gain beyond nulls; then candidate-only official correctness and explicit
SPEC-miss/fresh-input replay stress before any promotion. Unaffected shapes must not supply the
gain through placement noise. No precision change; no fence, copy, commit, wait or barrier moved.

### Late-tid first result, b4d75c3185744d0a

Correct receipt board/id confirmed. Passed16/16 official tests; unchanged default-off control
scored2.561845x (95%CI2.523418-2.613075). Control, late, ctrl2 each passed19/19 harnesschecks.
Receipt/log `profiles/latetid.b4d75c3185744d0a.{json,log}`. No new resource counters.

|Cold raw graph median, us|control|late|independent control2|
|---|---:|---:|---:|
|decode1 sliding|9.28|9.28|9.31|
|decode1 full|17.28|16.80|17.15|
|decode16 sliding (unchanged code)|13.06|12.86|13.12|
|verify32 sliding (unchanged code)|19.97|20.16|20.06|

D1-full candidate deltas on three held-out inputs: -0.51/-0.42/-0.45us;
control2 deltas -0.13/-0.06/-0.10us. This is a localized positive signal, not a promotion.
Raw ABSCORE late1.0020 versus null0.9985. Using rounded ABSUM medians and each official
shape's replay-overhead calibration gives approximate all-shape ratios1.003026/0.998234.
If unchanged instances are held equal to control, changed decode1-only ratios are
1.002760/1.000249. These are scoring sensitivity estimates, not official candidate scores:
the A/B graphs do not independently calibrate each arm's replay overhead. Unchanged decode16
and verify32 readings show why the complete raw ratio cannot be credited to this mechanism.

Four-arm confirmation dispatched: `ab53721b09854717`, sourcefac5188, generated with
`--null --shapes all --inputs 3 --rounds 150 --seed 311103
late1@own=-DONE_LATE_TID=1 late2@own=-DONE_LATE_TID=1,-DOKC_LATE_NULL=1`.
The unused second flag forces a separate extension build; both candidates have independent
workspace/preparation state, as do controlsctrl/ctrl2. Gates: all4 d1-full cross comparisons
must improve consistently on all3inputs beyond control/candidate null differences; corrected
changed-only aggregate gain must exceed twice the largest corrected changed-only null in log
space; no material regression on affected d1sliding. If confirmed, run candidate-only official
correctness and 2000fresh queued replays, including speculative-length misses, before promotion.

## Scored control repeat2: new board score without algorithm change

|date|job|commit|change|score (95%CI)|notes|
|---|---|---|---|---|---|
|2026-10-31|8914751e3b9f468b (root submit)|d3eeeda + repeatcomment2|Same owned PRQ31/HMRG1/pick-off control body|2.604290x (2.565226-2.643355)|Passed16/16 official tests; new best score caused by a repeat, not a new algorithm.|

Receipt `profiles/scored.8914751e3b9f468b.json`; exact root file
`.build/today/attention_repeat2.py` SHA256
`b296c13ad2745336e97488e297fcde751a5fc10c818c92b6f2250656369efe75`.
Default production flags remain ONE_LATE_TID0, PRQ31, HMRG1, workspacepick0.
Do not attribute the score increase over2.534850 to late-tid: this file does not enable it.
Latest best's full per-shape hotspot table appended to docs/hotspots.md with saved baseline
receipt24daba2202e547d6. Kernel-profile plans remain decode sliding13/1/1, full11/49/49;
verify sliding15/3/3, full11/51/51. Rank must be read live; no cutoff inferred here.

### Four-arm late-tid confirmation: ab53721b09854717

Receipt confirms correct board/id,16/16 official tests passed. Default-off control score
2.582317x (95%CI2.543582-2.630284); all4arms19/19 harnesschecks.
Files `profiles/lateconfirm.ab53721b09854717.{json,log}`.

|Cold rawus|ctrl|ctrl2|late1|late2|
|---|---:|---:|---:|---:|
|decode1 full|17.25|17.22|16.83|16.83|
|decode1 sliding|9.54|9.34|9.28|9.34|

Full-context candidate deltas by held-out input: late1 -0.38/-0.42/-0.38us;
late2 -0.38/-0.38/-0.42us; controlnull +0.03/-0.03/-0.03us. All four full-context
cross comparisons improve. Sliding differences are largely reproduced by the controlnull:
late1 -0.19/-0.26/-0.22, late2 -0.19/-0.26/-0.19, null -0.19/-0.22/-0.22us.

Raw whole ABSCORE late1/late2=1.0036, controlnull1.0010. Rounded-median official-overhead
sensitivity: all-shape ratios late1=1.005989, late2=1.005881, null=1.001871;
changed decode1-only ratios1.007110/1.006037/null1.003640. The preregistered
changed-only gain >2maxnulllog gate FAILS (2nulllog requires approximately1.007294).
This gate includes the sliding instance because ONE_LATE_TID1 changes its compiled code too.
No candidate promotion or stress dispatch follows this result. These corrected ratios remain
sensitivity estimates, not official candidate scores.

Measured conclusion: the full-context scheduling improvement repeats across builds, states
and inputs; sliding has no independently established gain. A new proposal may restrict the
mechanism to SPEC/non-one-tile L1 instances and preregister a full-only causal scoring gate.
This restriction would be based on new evidence above, not a repeat of the general knob.

## Fresh restricted late-tid registration, 2026-10-31 PDT

This is a NEW variant, not retroactive promotion of ONE_LATE_TID1, which failed its original
changed-instance gate in ab53721b09854717. New evidence is the repeated full-context gain and
sliding null shift in that check. Value2 applies only to L1 SPEC non-one-tile instances;
value1 retains the original broad toggle for reproducibility. Default0 remains unchanged.
Restriction uses launch geometry (`SPEC && !ONE`), never tensor values or benchmark identity.

Fresh four-arm generation: `--null --shapes all --inputs 3 --rounds 150 --seed 311106
late1@own=-DONE_LATE_TID=2 late2@own=-DONE_LATE_TID=2,-DOKC_LATE_NULL=1`.
Both candidates have separate builds/states, and controls ctrl/ctrl2 are independent.

Gate preregistered before dispatch:
1. All real-arm harness checks pass; all official unchanged-control tests pass.
2. At d1full, all4 candidate/control cross comparisons improve on every held-out input.
3. Full-only causal standalone score: hold all unchanged shapes, including d1sliding, exactly
at ctrl's measured time. Subtract each official shape's saved replay overhead from matching
raw medians, then compute the actual weighted load formula46sliding+16full and six-load
geometric mean. Pooled cross loggain must exceed twice the larger absolute lognull from
ctrl2/ctrl and late2/late1 under this SAME full-only calculation. These remain calibration
sensitivity estimates until candidate-only official timing, not separately measured scores.
4. No real regression at changed d1full. Unchanged rows' improvements receive zero credit;
any anomalous unmodified row must be documented and not used as the winning mechanism.
5. After confirmation, candidate-only official correctness plus2000fresh queued inputs per
changed/sentinel batch1 shape, every output compared on device with independent owned control,
three additional FP32-reference samples, explicit speculative-length misses, clean workspace,
and prepare-time resource attributes. No public submission before these gates pass.

|date|job|commit|change|score (95%CI)|notes|
|---|---|---|---|---|---|
|2026-10-31|a9b416a0c7324ecc (check)|554445d|ONE_LATE_TID2 full/non-one-tile restriction, fresh four-arm gate above|pending|Three inputs,150coldrounds,independent candidate builds+states; judged source defaultoff.|

Live standalone attention snapshot during restricted experiment (2026-10-31 late PDT):
`profiles/leaderboard.2026-10-31-late-campaign.json` reports ours rank206,2.6043x;
rank10 is jt914,2.9098x, screened job5cb3655bf3c9499a. The ratio2.9098/2.6043
requires approximately11.73% more score at this snapshot. This is a live official-row
comparison, not an invented cutoff or forecast. The bounded late-tid mechanism alone does
not establish that gap; the final review/rerun ranking remains provisional.

### Restricted full-only result: a9b416a0c7324ecc

Owning attention receipt verified,16/16 official control tests passed; control2.498407x
(95%CI2.460931-2.535883), not a candidate-only score. Every arm19/19 harnesschecks.
`profiles/latefull.a9b416a0c7324ecc.{json,log,gate.json}` retains receipt, log and evaluator
output. Registered value2 source is554445d, generated exactly as its predispatch command.

|Cold rawus|ctrl|ctrl2|late1|late2|
|---|---:|---:|---:|---:|
|decode1 full|17.28|17.25|16.90|16.80|
|decode1 sliding (unchanged)|9.31|9.28|9.34|9.31|

Full cross differences by input, candidate-minus-control microseconds:
input0 late1/R1 -0.32,L1/R2 -0.29,L2/R1 -0.42,L2/R2 -0.39;
input1 -0.42/-0.39/-0.48/-0.45; input2 -0.42/-0.39/-0.51/-0.48.
All12 cross comparisons improve. Pooled full-only corrected sensitivity ratio1.0023928;
controlnull1.0001715,candidatenull1.0005802. Loggain exceeds2largerlognull: gatePASS.
Tool `tools/attn_causal_score.py` gives the reproducible calculation; all unchanged rows,
including sliding, receive zero credit. Raw wholeABSCORE1.0014/1.0028 versusnull0.9998
is also recorded but is not the causal gate. The preceding value1 failure remains unchanged.

Candidate-only official/stress check dispatched3588375a529c4fe8, source194b92b+stress-tool
b470ab1/5c546fc: generated owned file enables ONE_LATE_TID2 in actual judged path;
repository productiondefault stays0 pending correctness/stress. Generator
`tools/attn_make_candidate_stress.py` uses independent defaultoff owned control and two
batch1 shapes, each with2000unique queued qkv inputs and a device error comparator for EVERY
output; additionally3FP32-reference samples,50replays after shortening sequence64tokens,
workspace scans and host prepare resource attributes. Comparator threshold is relativeL2
1e-4, stricter than official math thresholds; classify any BF16/atomic-order null before
attributing a failure to the scheduling change. No public submission yet.

|date|job|commit|change|score (95%CI)|notes|
|---|---|---|---|---|---|
|2026-10-31|3588375a529c4fe8 (check)|554445d + owned stress generator5c546fc|Actual full-only late-tid2 candidate, fresh queued stress/resource gate|pending|Productiondefault0; official judgedpath2;2000uniqueinputs peraffected/sentinelshape,alloutputs device-compared.|

### Candidate stress3588375a529c4fe8 FAILED, no promotion

Owning receipt board/id verified; prepare/import failed before official tests or timing.
No candidate score or CI exists. Owned selftest reports0failing shapes, but the strict queued
sliding comparison reports17/2000 fresh outputs above relativeL2=1e-4;
maxrelative_squared=2.0038797288179921e-7 (relativeL2 approximately4.48e-4).
Full-context queued stress was NOT reached. These are all-output comparator results,
not3reference samples. No source activation is promoted; productiondefault0 stays.
Files `profiles/latestress.3588375a529c4fe8.{json,log}` preserve the failed check.
Prepare host resource print records did not survive in the receipt tail, so no resource
configuration change is claimed from this failed run.

Sliding is compiled unchanged by value2, making an independent-control null diagnostic
necessary before attributing the failure. This is a hypothesis, not dismissal as noise.
The original strict failure remains a failure. No threshold or gate was weakened.

Root-authorized diagnostic26c8a3ea28554454: four independent compiled modules/states
0a/0b/2a/2b, identical2000freshqkv inputs on EACHsliding/full, sixcomparisons (bothnulls,
all4crosses), original1e-4 global relativeL2 criterion, plus maximumrowerror androwcounts.
Allshape statistics print before the final strict assertion. Three FP32reference samples
perarm/pershape and owned selftests provide independent math coverage. Actual judged source
remains0; failed diagnostic cannot promote performance. Generatorc2197bd+0d5901e.
An initial >1MB payload was rejected422without a job; shared owned-source text reduced the
accepted payload to624471bytes. No donor source, compression or extra package is used.

|date|job|commit|change|score (95%CI)|notes|
|---|---|---|---|---|---|
|2026-10-31|26c8a3ea28554454 (check)|0d5901e|Strict fresh-output null/cross diagnostic,4independent modules/states|pending|Judgeddefault0; bothshapes beforefinalassert; unchanged1e-4 threshold.|

### Strict matched null diagnostic26c8a3ea28554454

Correct owning receipt verified. Prepare failed at final strict assertion AFTER bothshape
statistics and all24FP32samples; no official test/timing score exists. Owned selftest0fails.
Files `profiles/latenull.26c8a3ea28554454.{json,log}`.

|Shape/pair|Strictbad/2000|maxglobalrel|maxrowrel|rowbad|
|---|---:|---:|---:|---:|
|sliding0b/0a|19|0.000457718|0.00179499|32|
|sliding2b/2a|16|0.000458139|0.00187698|32|
|slidingcandidate/controlcrosses|11-18|0.000442782-0.000458118|0.00180121-0.00187558|28-30|
|full0b/0a|10|0.000231625|0.000989572|23|
|full2b/2a|12|0.000444784|0.00173686|24|
|fullcandidate/controlcrosses|10-14|0.000452304|0.00174637|19-32|

Measured: strict1e-4 fails incumbent-null on BOTHshapes. Candidatefull maxerror is also larger
than this singlecontrol-null maximum, so calling everydifference benign is unsupported.
All24 independentFP32samples pass: sliding globalerr0.003708-0.004668 (bounds0.136-0.145),
full0.001468-0.002188 (bounds0.048-0.050); cacheerrors0. These24samples doNOT establish
FP32correctness for everyqueued input. Original strict failures remain failures; no threshold
was changed, no candidate markedcorrect and no production/publicpromotion follows.

Root's next authorized investigation is a NEWprospective official-TASK FP32-reference fresh
gate for everyqueued input, with independentcontrol and stale/zero/order1 corruption controls.
A different gate must be preregistered and measured independently, never used to relabel either
failed strict check as passing. The causal~0.24% performancegain remains farbelow live11.73%
top10score gap; afterfreshmathclassification, close thismicroknob and reassess larger measured
opportunities rather than continuinglate-tid sweeps.

## Prospective official-math fresh gate, registered before dispatch

This NEWgate answers official TASK math compliance for everyqueued input. It doesNOT turn
3588375a529c4fe8 or26c8a3ea28554454 strict1e-4 failures into passes, and doesnotchange the
native value2 candidate. Tool `tools/attn_make_fp32_fresh_gate.py` generates actualcandidate2
plus independentcontrol0; pure candidate SHA remainscbee36a852fbc62f5491e75f957eeb4298dce2db50947984cdbbcd08aa69b356.

Coverage: BOTHbatch1 sliding/full,2000uniqueqkv inputs each, everycandidate ANDcontrol output
checked ondevice. For eachinput, ownFP32 attention references are computed twice (BF16queries
and E4M3queries) from the fixedcachedpast and thatinput's actualstorednewKV. Control-graph
prepass collectsactualnewKV for everyinput; candidatecache must exactlymatchthose rows, in
addition to official8%cacheerror checks against own normalized/rotated/quantized raw rows.
Thus the sharedFP32reference uses exactly the newKV that eachcheckedoutput attends; no stale
new-row approximation. No reference.pyimport, cachedoutput reuse in kernel_fn or timed shortcut.

Official criteria from reference.py/checks.py: globalrelativeL2 <=clamp(2.5*E4M3_reference_error
+0.015,0.03,0.15); worst64-element row <=5*thatbound with rownormfloor0.1*meanrownorm clamped
1e-12; finiteoutputs; newK/VrelativeL2 <=0.08. Bound derivesonly from ownFP32/E4M3references,
never fitted to candidate/null errors. Threeinputs per shape crosscheck BOTHbatched references
against canonical `_ob_reference` beforequeue (relativeL2<1e-3, allowing reference summation
rounding); eachshape's stalecompleteoutput, allzerooutput and +10oneheadcorruption must all
be rejected. Independentcontrol must pass everyofficial-math check too. Workspace mustclean.
Anyincompletecoverage orfailure aborts; prepare280s budget startsbeforeANYmodule compilation.
No publicpromotion beforethisgate ANDactualcandidate officialtests/timing pass.

CPUwork/shape model: atI2000,Hq16,T80001,D64, twoQK+PV references require
8*I*Hq*T*D=1,310,736,384,000 multiply/add operations (not a measured runtime). Chunk32 limits
simultaneouslylive score/concatenation/probability matrices to
3*32*16*80001*4=491,526,144bytes; prior loop matrices are explicitlydeleted. TwoBF16reference
banks require8,192,000bytes; qkvbank9,216,000bytes; decodedpastK/V~81.92MB; actualnewKVbanks
~2.048MB. Additionalgraph/output/tensorallocator storage is not included in these analytic
figures. Bothshape references and actualduration are printed duringrun; no projectedGPUtime.
CPUalgebra tests compare flattenedbatched past+perinputnew attention with directcanonical
nestedattention for empty/short/longerpast and native2KVheads/8groups/64dims, andcheckwork/byte
arithmetic:2/2pass. Pythonandgeneratedsource compile. Runtimealso crosschecksRoPE/quantization
math on canonicalreferences; CPUalgebra alone doesnotvalidateGPUexecution.

Read-only CUDAresourceattributes are exposed through a new hostmethod in the generatedfile
and printed byPython duringprepare; nativekernel bodies are unchanged. This avoids the
bufferedC++prints missing from the failedstressreceipt. Everyindexedcheckerargument isruntime,
not constexpr perinput, avoiding2000JITspecializations. Final280s guard failsclosed ifcoverage
is not complete. Source/state controlseparate; ordinarysame-stream graph/copy dependencies.
