# G1b2 H200 layer comparisons

Adjusted medians in µs. Each column is from an independent judge run with its own routing and SGLang measurements. G0 is the external comparator. G1a and G1b2 share the repaired sequential layer. G1b2 uses audited native experts on the measured run.

| Load | Context | Owned public | G0 | G1a | G1b2 |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 76.170 | 66.352 | 80.454 | 76.923 |
| decode 1 | full | 82.231 | 73.527 | 87.405 | 83.615 |
| decode 16 | sliding | 139.969 | 130.187 | 146.839 | 132.798 |
| decode 16 | full | 221.464 | 205.802 | 215.133 | 212.540 |
| decode 32 | sliding | 171.442 | 152.172 | 172.811 | 159.381 |
| decode 32 | full | 323.233 | 301.541 | 323.291 | 310.008 |
| verify 1 | sliding | 118.260 | 100.606 | 120.094 | 110.857 |
| verify 1 | full | 127.790 | 112.279 | 127.612 | 114.062 |
| verify 16 | sliding | 263.258 | 237.373 | 270.582 | 266.664 |
| verify 16 | full | 348.986 | 319.902 | 356.593 | 350.368 |
| verify 32 | sliding | 342.446 | 308.691 | 346.562 | 336.668 |
| verify 32 | full | 512.243 | 471.851 | 519.244 | 499.824 |

| Board receipt | Speedup | 95% CI |
|---|---:|---|
| Owned public `cb71acec29d141c6` | 2.043438× | 2.012787–2.074090 |
| G0 `3a5f879df0d44825` | 2.248791× | 2.215059–2.282522 |
| G1a `606f47f251f34873` | 2.010975× | 1.980811–2.041140 |
| G1b2 `736a5cbc977f4ad4` | 2.065559× | 2.034575–2.096542 |

| Load | Owned public | G0 | G1a | G1b2 |
|---|---:|---:|---:|---:|
| decode 1 | 77.734 | 68.204 | 82.248 | 78.650 |
| decode 16 | 161.000 | 149.701 | 164.463 | 153.376 |
| decode 32 | 210.614 | 190.719 | 211.645 | 198.252 |
| verify 1 | 120.719 | 103.619 | 122.034 | 111.684 |
| verify 16 | 285.381 | 258.671 | 292.778 | 288.265 |
| verify 32 | 386.265 | 350.797 | 391.125 | 378.773 |

Load times above are (46 × sliding + 16 × full) / 62. The six-load geometric mean of the weighted SGLang-to-G1b2 ratios is 2.065558556753×. The official score is 2.065558556753×.

The log confirms native experts with SC6 and eight eligible cuBLASLt plans. The setup record reports one prepared copy. Twelve CUPTI kernel-time tables are available in the raw receipt. Register counts, exact occupancy, achieved bandwidth and stall reasons for this run were not reported. Runtime residency checks passed for seven native startup probes. The compiled kernels and observed shape times do not constitute an empty-routing test.
