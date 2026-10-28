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

## Ruled out (pending merge)

| experiment | evidence | what it rules out |
|---|---|---|
| q_len 8 tile loop in the public per-request order (QKF, mode bit 128: Q.K issued before the next copies, V transpose under the in-flight Q.K) | bf6b8ca9e37341fb BENCH: verify 16 / 32 full 137.2 / 257.0 us vs control 128.1 / 237.9; sliding b16 / b32 17.3 / 22.1 vs 16.8 / 21.7 | that copying the public per-request ordering gains anything in our kernel; the sliding gap to the public times is not in this ordering |
| early raw-stage release (EARLY, mode bit 384: refill a stage once Q.K and the V transpose are done, three tiles in flight on three stages) on flat and per-request q_len 8 | bf6b8ca9e37341fb judge: verify 16 / 32 full 137.4 / 261.2 us vs 127.1 / 239.1 (b458555d3e3e4e9a); sliding verify 16 / 32 14.1 / 18.6 vs 13.3 / 17.5; scored 2.083x vs 2.162x | a deeper prefetch through early release on the consumer-copy loop (mode 19 and the per-request modes); the flat q_len 8 gap to the public 105 / 186 us is not prefetch depth in this design |
