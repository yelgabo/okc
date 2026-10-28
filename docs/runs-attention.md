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
