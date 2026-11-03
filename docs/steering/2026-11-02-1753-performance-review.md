# Architecture performance review, November 2 at 17:53 Vancouver time

Evidence snapshot `2ffc98c1`.

**The fixed MID17–32 expert route is parked. The new D32 attention lookahead has an exploratory whole-layer signal, with matched confirmation queued.** Public G1b4 remains the retained 2.126083× layer. The new attention run is an E scheduling descendant inside G, with the existing producer ring, page TMA, half merge and finalizer preserved. It changes full-context, single-query batches 17–32. MID is a separate G expert-route comparison. [MID terminal](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/2026-11-02-driver-d32-mid-terminal.md) · [Attention first result](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/2026-11-02-05-attention-first-result.md)

## D32 result and drift

| Run | Sliding µs | Full µs | C32 ms | Own D32 ratio against SGLang |
|---|---:|---:|---:|---:|
| Unchanged control94ef | 157.621 | 317.105 | 12.324230 | 1.811791× |
| Fixed MID17–32 | 166.129 | 319.397 | 12.752280 | 1.713503× |
| One-tile score lookahead | 149.764 | 312.070 | 11.882275 | 1.839755× |

`C32_ms = (46 * sliding_us + 16 * full_us) / 1000`. MID is 3.473% slower than the new unchanged control. Attention C32 is 0.441955 ms lower, or 3.586%, in a separate run. The changed full-context D32 cell falls by 5.035 µs (1.588%). Unchanged sliding D32 falls by 7.856 µs (4.984%). Its weighted difference accounts for about 81.8% of the observed C32 reduction. This is an arithmetic decomposition of cross-run observations. It cannot isolate the lookahead contribution. [Attention comparison](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/evidence/D32-ATTN-01/comparison.json)

## Correctness and measurement scope

All three new receipts pass 13 official tests, 12 benchmarks and 12 profiles, with clean checks and return code0. The task checks every timed call against the reference, including outputs and new cache entries. The attention D32-full benchmark reports pass, supporting ordinary timed-cell reference acceptance. The inherited 56 private replays exercise experts. They do not establish the new attention variant’s matched cold-replay, neighboring-shape or independent-state coverage. Those gates belong to confirmation. [Task reference-check contract](https://github.com/yelgabo/okc/blob/2ffc98c1/muse_decode_layer/TASK.md) · [Attention owning receipt](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/evidence/D32-ATTN-01/status-wait.raw.json)

## Unchanged current-service calibration

Private H200 job `94ef698e91964dfa`, source `5b4a729a / physical04314b16`, completed 2026-11-02 23:53:59 UTC. Official score **2.091748×**, within-run 95% CI 2.060371–2.123124×. Preparation took 114.752 seconds. [Owning receipt](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/evidence/2026-11-02-driver-current-incumbent-control/terminal-canonical.utf8.json)

The following normalized endpoints use `(46 * sliding + 16 * full) / 62` against `cc474883117947c0`. This is an unchanged-source calibration, so differences measure cross-run drift.

| Load | Tokens | Candidate µs | Control µs | Delta µs | Delta % |
|---|---:|---:|---:|---:|---:|
| decode 1 | 1 | 79.074 | 78.765 | +0.310 | +0.39% |
| decode 16 | 16 | 157.437 | 152.016 | +5.421 | +3.57% |
| decode 32 | 32 | 198.778 | 191.898 | +6.880 | +3.59% |
| verify 1 | 8 | 120.705 | 112.682 | +8.023 | +7.12% |
| verify 16 | 128 | 274.180 | 261.713 | +12.467 | +4.76% |
| verify 32 | 256 | 356.045 | 351.080 | +4.965 | +1.41% |

Per-cell repeat standard deviations are in candidate/control order. Decode1 has 60 repeats per arm; other cells have20. No paired treatment interval is recorded.

| Tokens / context | Candidate µs | Control µs | Repeat SD µs, candidate / control |
|---|---:|---:|---:|
| 1 / sliding | 77.208 | 77.036 | 4.203 / 3.912 |
| 1 / full | 84.440 | 83.733 | 4.380 / 4.155 |
| 16 / sliding | 138.048 | 133.037 | 8.270 / 8.072 |
| 16 / full | 213.179 | 206.582 | 10.322 / 11.385 |
| 32 / sliding | 157.621 | 150.875 | 11.126 / 7.812 |
| 32 / full | 317.105 | 309.839 | 10.270 / 8.874 |
| 8 / sliding | 118.645 | 110.761 | 12.335 / 10.201 |
| 8 / full | 126.628 | 118.204 | 10.662 / 10.824 |
| 128 / sliding | 251.282 | 237.040 | 11.363 / 9.045 |
| 128 / full | 340.013 | 332.647 | 10.052 / 10.321 |
| 256 / sliding | 314.988 | 308.622 | 7.015 / 8.556 |
| 256 / full | 474.084 | 473.149 | 9.117 / 8.212 |

## Fixed MID17–32, parked

Private H200 job `a6fe2769779e4299`, source `28346bf5 / physicale607fb89`, completed 2026-11-03 00:07:00 UTC. Official score **2.089423×**, within-run 95% CI 2.058082–2.120765×. Preparation took 104.090 seconds. [Owning receipt](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/evidence/2026-11-02-driver-d32-mid-first-check/terminal-canonical.utf8.json)

The following normalized endpoints use `(46 * sliding + 16 * full) / 62` against `94ef698e91964dfa`. These jobs have different seeds and nodes, so the deltas are descriptive.

| Load | Tokens | Candidate µs | Control µs | Delta µs | Delta % |
|---|---:|---:|---:|---:|---:|
| decode 1 | 1 | 78.547 | 79.074 | -0.528 | -0.67% |
| decode 16 | 16 | 151.610 | 157.437 | -5.827 | -3.70% |
| decode 32 | 32 | 205.682 | 198.778 | +6.904 | +3.47% |
| verify 1 | 8 | 113.603 | 120.705 | -7.102 | -5.88% |
| verify 16 | 128 | 258.150 | 274.180 | -16.031 | -5.85% |
| verify 32 | 256 | 352.253 | 356.045 | -3.792 | -1.06% |

Per-cell repeat standard deviations are in candidate/control order. Decode1 has 60 repeats per arm; other cells have20. No paired treatment interval is recorded.

| Tokens / context | Candidate µs | Control µs | Repeat SD µs, candidate / control |
|---|---:|---:|---:|
| 1 / sliding | 77.009 | 77.208 | 3.925 / 4.203 |
| 1 / full | 82.969 | 84.440 | 3.970 / 4.380 |
| 16 / sliding | 132.218 | 138.048 | 8.810 / 8.270 |
| 16 / full | 207.360 | 213.179 | 9.820 / 10.322 |
| 32 / sliding | 166.129 | 157.621 | 11.729 / 11.126 |
| 32 / full | 319.397 | 317.105 | 7.447 / 10.270 |
| 8 / sliding | 111.186 | 118.645 | 10.955 / 12.335 |
| 8 / full | 120.552 | 126.628 | 13.658 / 10.662 |
| 128 / sliding | 235.869 | 251.282 | 8.375 / 11.363 |
| 128 / full | 322.207 | 340.013 | 11.137 / 10.052 |
| 256 / sliding | 313.012 | 314.988 | 8.621 / 7.015 |
| 256 / full | 465.071 | 474.084 | 8.164 / 9.117 |

## D32 attention one-tile lookahead, inconclusive

Private H200 job `17383751c6a24010`, source `31c7f920 / physical5a580c7a`, completed 2026-11-03 00:29:58 UTC. Official score **2.128829×**, within-run 95% CI 2.096897–2.160762×. Preparation took 141.737 seconds. [Owning receipt](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/evidence/D32-ATTN-01/status-wait.raw.json)

The following normalized endpoints use `(46 * sliding + 16 * full) / 62` against `94ef698e91964dfa`. These jobs have different seeds and nodes, so the deltas are descriptive.

| Load | Tokens | Candidate µs | Control µs | Delta µs | Delta % |
|---|---:|---:|---:|---:|---:|
| decode 1 | 1 | 78.610 | 79.074 | -0.464 | -0.59% |
| decode 16 | 16 | 155.106 | 157.437 | -2.330 | -1.48% |
| decode 32 | 32 | 191.650 | 198.778 | -7.128 | -3.59% |
| verify 1 | 8 | 114.094 | 120.705 | -6.611 | -5.48% |
| verify 16 | 128 | 266.789 | 274.180 | -7.391 | -2.70% |
| verify 32 | 256 | 352.146 | 356.045 | -3.898 | -1.09% |

Per-cell repeat standard deviations are in candidate/control order. Decode1 has 60 repeats per arm; other cells have20. No paired treatment interval is recorded.

| Tokens / context | Candidate µs | Control µs | Repeat SD µs, candidate / control |
|---|---:|---:|---:|
| 1 / sliding | 76.576 | 77.208 | 3.555 / 4.203 |
| 1 / full | 84.459 | 84.440 | 3.659 / 4.380 |
| 16 / sliding | 134.568 | 138.048 | 9.925 / 8.270 |
| 16 / full | 214.154 | 213.179 | 8.850 / 10.322 |
| 32 / sliding | 149.764 | 157.621 | 8.207 / 11.126 |
| 32 / full | 312.070 | 317.105 | 11.426 / 10.270 |
| 8 / sliding | 111.784 | 118.645 | 8.168 / 12.335 |
| 8 / full | 120.733 | 126.628 | 9.616 / 10.662 |
| 128 / sliding | 245.223 | 251.282 | 8.707 / 11.363 |
| 128 / full | 328.791 | 340.013 | 8.926 / 10.052 |
| 256 / sliding | 310.867 | 314.988 | 7.179 / 7.015 |
| 256 / full | 470.824 | 474.084 | 10.055 / 9.117 |

## Attribution, controls and pending confirmation

The new control and MID use H200/nyc3b/driver580.159.03 with matching evaluator files, but different input seeds and nodes. The attention comparison likewise matches evaluator/driver identity but has different seeds and nodes. Workloads use soft routing, 80k context and agentic spread on multibatch loads. Exact routing histograms are absent. MID changes the T17–32 expert selection; attention changes only full-context q_len1/page1 attention at batches17–32. The standard128/256 verify paths retain the original implementation in both candidates. Their faster cross-run rows supply no isolated family gains. [MID source and scope](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/2026-11-02-worker-d32-mid-family.md) · [Attention source and scope](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/2026-11-02-01-full-decode-attention-experiment.md)

The first attention result remains inconclusive for an isolated E gain. Matched private job `4fece18b20a2447b` is accepted for payload `109cdb07`, 928942 bytes, source commit `c7e13600`. It embeds treatment5a580c7a and ba9 control. The published contract uses three seed banks, all12 cells, two control and two treatment placements,24 cold replays and a256MiB flush, with numerical/cache/state and duplicate-control/regression gates. There is no matched terminal result or performance verdict at this pin. [Accepted job](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/evidence/D32-ATTN-02/accepted.json) · [Payload and scope](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/2026-11-02-08-confirmation-payload.md) · [Prospective contract](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/2026-11-02-07-confirmation-contract.md)

Decode32 Worker owns D32 implementation, checks, analysis and integration directly under the scoped user instruction. Driver’s pre-existing MID wait is complete, and its one-time exception is retired. Routine Driver handoffs are superseded for D32. The read-only local chat snapshot reported the matched check still queued under the owner’s progress watch. No messages were sent to that chat. [Current D32 ledger](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/decode32/STATUS.md)

A remains ahead of the staged B implementations tested at128/256. The E lookahead changes an existing attention schedule and supplies no new A-versus-B evidence. Broad dense backing remains unallocated reserve. The new CUDA source-correction supplement is source-only context, with no new GPU measurement or performance gain. [Source-correction scope](https://github.com/yelgabo/okc/blob/2ffc98c1/docs/steering/2026-11-02-cuda-source-corrections-intake.md)

Checkpoint `2ffc98c1`. New terminal jobs94ef698e91964dfa, a6fe2769779e4299 and17383751c6a24010 are counted once. Matched4fece18b20a2447b is pending, with dispatch beginning2026-11-03 00:49:30 UTC. Read-only owner snapshot cursor `1cba8e71-65d3-4a48-b3c2-f00e5feaafa6:3` reports queued. Parsed 18 untracked historical profile JSONs without staging or modifying them. Integer seeds in the MID CP1252 raw receipt match its labeled UTF-8 canonical derivative. Independent review verified raw identities, source hashes, endpoint calculations and correctness scope.
