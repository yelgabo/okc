# G1a official shape medians

All times are adjusted judge medians in µs. Each column comes from a different run, with fresh routing and independent baseline measurements.

| Load | Context | Owned public cb71acec29d141c6 | G0 3a5f879df0d44825 | G1a 606f47f251f34873 |
|---|---|---:|---:|---:|
| decode 1 | sliding | 76.170 | 66.352 | 80.454 |
| decode 1 | full | 82.231 | 73.527 | 87.405 |
| decode 16 | sliding | 139.969 | 130.187 | 146.839 |
| decode 16 | full | 221.464 | 205.802 | 215.133 |
| decode 32 | sliding | 171.442 | 152.172 | 172.811 |
| decode 32 | full | 323.233 | 301.541 | 323.291 |
| verify 1 | sliding | 118.260 | 100.606 | 120.094 |
| verify 1 | full | 127.790 | 112.279 | 127.612 |
| verify 16 | sliding | 263.258 | 237.373 | 270.582 |
| verify 16 | full | 348.986 | 319.902 | 356.593 |
| verify 32 | sliding | 342.446 | 308.691 | 346.562 |
| verify 32 | full | 512.243 | 471.851 | 519.244 |

The official G1a result is 2.010975134× (95% CI 1.980810507–2.041139761). The setup record reports four copies and a prepare time of 136.83 s. The backend diagnostic reported owned experts and eight eligible cuBLASLt plans. CUPTI kernel-time reports cover twelve cells. They do not report achieved bandwidth, occupancy or SM stall counters.

| Load | G1a time divided by owned time | G1a time divided by G0 time |
|---|---:|---:|
| decode 1 | 1.0581 | 1.2059 |
| decode 16 | 1.0215 | 1.0986 |
| decode 32 | 1.0049 | 1.1097 |
| verify 1 | 1.0109 | 1.1777 |
| verify 16 | 1.0259 | 1.1319 |
| verify 32 | 1.0126 | 1.1150 |
