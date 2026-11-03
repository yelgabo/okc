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

## Post-stage publication

Job `27b2ed49fc6344ec` used the same startup checks as `607eb52bdaf24160`. All 13 official cases and 56 private replays passed. The new weighted score reconstructed from its own 12 SGLang and submission medians was 2.102308440336×. Stage retirement with the original publication point retains the higher public score. [Judge receipt](../../../../../muse_decode_layer/profiles/scored.27b2ed49fc6344ec.json).

| Load | Incumbent weighted µs | New sliding µs | New full µs | New weighted µs | Latency change |
|---|---:|---:|---:|---:|---:|
| decode 1 | 78.549 | 78.094 | 85.098 | 79.901 | +1.72% |
| decode 16 | 158.241 | 136.177 | 214.112 | 156.289 | −1.23% |
| decode 32 | 191.740 | 156.613 | 306.434 | 195.277 | +1.84% |
| verify 1 | 115.991 | 115.241 | 122.983 | 117.239 | +1.08% |
| verify 16 | 269.347 | 254.424 | 334.463 | 275.079 | +2.13% |
| verify 32 | 359.297 | 313.884 | 478.023 | 356.243 | −0.85% |

Decode-32 and verify-16/32 dispatch the changed expert kernel. Both runs had fresh route inputs and their own baseline measurements. The overlapping score intervals and the movement in unaffected loads preclude assigning these differences to publication timing alone.

## Concurrent short pooler and router

Public job `942ebdec400d4e68` passed all 13 tests. The judge confirmed the real SK2 pooler and separate FP32 router at one and eight tokens, 56 expert replays and 32 producer replays. The six weighted ratios reconstruct the official 2.095900315804×. The two targeted loads are marked in bold. [Raw judge receipt](../../../../../muse_decode_layer/profiles/scored.942ebdec400d4e68.json).

| Load | Fused incumbent weighted µs | New sliding µs | New full µs | New weighted µs | Latency change |
|---|---:|---:|---:|---:|---:|
| **decode 1** | 78.549 | 81.382 | 87.849 | 83.051 | +5.73% |
| decode 16 | 158.241 | 132.505 | 209.809 | 152.455 | −3.66% |
| decode 32 | 191.740 | 156.184 | 304.622 | 194.491 | +1.43% |
| **verify 1** | 115.991 | 110.126 | 123.391 | 113.549 | −2.10% |
| verify 16 | 269.347 | 248.842 | 332.329 | 270.387 | +0.39% |
| verify 32 | 359.297 | 315.877 | 474.045 | 356.694 | −0.72% |

| Targeted cell | Fused producer activity in incumbent µs | SK2 pooler activity µs | FP32 router activity µs |
|---|---:|---:|---:|
| decode 1, sliding | 17.9 | 15.3 | 6.9 |
| decode 1, full | 17.9 | 14.3 | 6.7 |
| verify 1, sliding | 18.7 | 14.9 | 6.0 |
| verify 1, full | 18.6 | 15.3 | 6.5 |

The score runs used fresh inputs. The scored profiles give per-kernel durations and activity shares. They do not contain issue timestamps, a measured stream-overlap interval, SM utilization or HBM bandwidth. Adding the pooler and router durations would double-count any concurrent execution. [Profile tables](../../../../../muse_decode_layer/profiles/g1-short-overlap.942ebdec400d4e68.txt).
