# Run log: muse_attention_decode

Every `okc check`, newest last. Full output for each job is in
`muse_attention_decode/profiles/<file>.<job>.*`. Bench numbers come from
`tools/attn_make_bench.py`: CUDA graph replays after a 256 MB flush, median of 40, timed at
import on the judge's H200. They include about 5 µs of graph-replay cost that the judge
subtracts, so compare them within a shape, not against the judge's column.

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 10-27 | 2b3c25820fee4b2d | 99f2d18 | baseline copy + component bench | 1.005x (0.990-1.020) | profile: FA3 main kernel is 94-97% of full-layer time at b16/b32 (195 / 374 µs vs floors 63 / 126); sliding is 7 launches of 1-16 µs each |

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
