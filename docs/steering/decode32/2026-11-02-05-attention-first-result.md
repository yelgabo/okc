# D32 attention first private result

November 2, 2026 Pacific. Candidate `5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5`, source commit `31c7f920`, private job `17383751c6a24010`. The job completed with clean checks, 13 passing tests, 12 benchmark cells and 12 profiles. The preparation log recorded treatment eligibility, the exact CUDA source hash, capability 9.0 and 132 SMs; the inherited expert startup recorded 56 replays. Preparation took 141.737 seconds.

| D32 cell | Adjusted us | Raw us | Baseline adjusted us |
|---|---:|---:|---:|
| Sliding | 149.764 | 153.744 | 259.520 |
| Full | 312.070 | 315.056 | 620.160 |
| Weighted total | 11.882275 ms | 12.113120 ms | 21.860480 ms |

The run's D32 ratio is 1.839755x. Its whole-layer score is 2.128829x with reported 95% interval [2.096897, 2.160762]. These are this private run's own measurements.

The earlier unchanged control `94ef698e91964dfa` recorded D32 C32 12.324230 ms, compared with 11.882275 ms here. The two runs used different inputs and nodes. The difference is a promising exploratory signal, not a matched causal estimate or a reproducible gain. The [receipt comparison](evidence/D32-ATTN-01/comparison.json) records all six loads; its generic short-load threshold is not this task's gate. The original [terminal receipt](evidence/D32-ATTN-01/status-wait.raw.json) contains all cells and profiles.

Next: prepare a bounded matched held-out comparison against ba9 with independent mutable state, duplicate controls and cold replay. Confirm actual treatment selection, fresh output/cache correctness and all-six regressions before integrating or claiming a gain. No public submission follows from this result.
