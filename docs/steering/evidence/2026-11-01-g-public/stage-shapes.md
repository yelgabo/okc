# Family G scored layer shape comparison

November 1. All values are adjusted judge medians in microseconds. Each file ran independently with its own routing and SGLang baseline. The upstream comparator is G0. The new public score was reconstructed from the six workload ratios to machine precision.

| Load | Context | Owned public cb71acec29d141c6 | G0 check 3a5f879df0d44825 | G stage public 607eb52bdaf24160 | SGLang in G stage run |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 76.170 | 66.352 | 76.748 | 170.672 |
| decode 1 | full | 82.231 | 73.527 | 83.727 | 180.816 |
| decode 16 | sliding | 139.969 | 130.187 | 138.661 | 267.056 |
| decode 16 | full | 221.464 | 205.802 | 214.534 | 435.552 |
| decode 32 | sliding | 171.442 | 152.172 | 154.820 | 264.304 |
| decode 32 | full | 323.233 | 301.541 | 297.883 | 624.448 |
| verify 1 | sliding | 118.260 | 100.606 | 115.432 | 265.312 |
| verify 1 | full | 127.790 | 112.279 | 117.598 | 270.256 |
| verify 16 | sliding | 263.258 | 237.373 | 247.131 | 539.568 |
| verify 16 | full | 348.986 | 319.902 | 333.218 | 709.984 |
| verify 32 | sliding | 342.446 | 308.691 | 316.390 | 733.632 |
| verify 32 | full | 512.243 | 471.851 | 482.657 | 1092.240 |

| Load | Owned weighted µs | G0 weighted µs | G stage weighted µs | G stage vs owned latency |
|---|---:|---:|---:|---:|
| decode 1 | 77.734 | 68.204 | 78.549 | +1.05% |
| decode 16 | 161.000 | 149.701 | 158.241 | -1.71% |
| decode 32 | 210.614 | 190.719 | 191.740 | -8.96% |
| verify 1 | 120.719 | 103.619 | 115.991 | -3.92% |
| verify 16 | 285.381 | 258.671 | 269.347 | -5.62% |
| verify 32 | 386.265 | 350.797 | 359.297 | -6.98% |

Official G stage score 2.126082676423×; reconstruction 2.126082676423×. The 95% score interval was 2.094191–2.157974×. The scored incumbent improved by 4.04% in the independent public point estimates. Stage retirement alone has no same-process whole-layer attribution yet.

All six official load times use (46 × sliding + 16 × full) / 62. The comparison excludes prefill.

## G1b4 control qualification and independent comparison

Control job 74a29e0384324952 passed 13 official cases, 56 individually checked private replays, and 2000 queued private replays on each of 32/33/128/256 tokens. It scored 2.052870208× with a 95% CI of 2.022077155–2.083663261. Stage and control were measured on separate judge runs.

| Load | G1b4 control weighted µs | G stage weighted µs | Control sliding/full µs | Stage sliding/full µs |
|---|---:|---:|---:|---:|
| decode 1 | 79.741 | 78.549 | 77.917/84.984 | 76.748/83.727 |
| decode 16 | 160.073 | 158.241 | 141.156/214.461 | 138.661/214.534 |
| decode 32 | 201.433 | 191.740 | 163.434/310.680 | 154.820/297.883 |
| verify 1 | 115.903 | 115.991 | 112.684/125.157 | 115.432/117.598 |
| verify 16 | 291.310 | 269.347 | 269.541/353.895 | 247.131/333.218 |
| verify 32 | 375.677 | 359.297 | 332.181/500.726 | 316.390/482.657 |
