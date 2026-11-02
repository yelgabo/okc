# Architecture performance review, November 2 at 16:50 Vancouver time

Evidence snapshot `bdbf35bf`.

**Both new G expert implementations passed correctness and lost their whole-layer continuation screens.** The shared-operand producer and direct weight delivery are parked, with confirmation and tuning canceled. Public G1b4 remains at 2.126083×. The comparison concerns these fixed implementations. Their pipelines execute expert work, so they remain in Family G. They supply no new attention-E or staged-B result. [Shared-A terminal](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md) · [Direct-weight terminal](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/2026-11-02-driver-direct-weight-terminal-and-dense-backing.md)

## Decode batch 32 first

D32 is 32 tokens with q_len=1. The new user priority targets a reproducible 2.0× on this load, followed by 2.1–2.2×. These are targets. Captain pickup is recorded, while developer delivery/pickup and accepted calibration/MID job IDs remain unestablished at this evidence pin. The next allocated comparison uses the existing MID helper for metadata 17–32, preserving MAIN at 33 and other standard-load routes. Broad dense backing is an unallocated reserve. [Owner directive](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/2026-11-02-decode32-owner-directive.md) · [Fixed comparison allocation](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/2026-11-02-driver-d32-mid-comparison.md)

| Candidate | D32 sliding µs | D32 full µs | Weighted C32 ms | D32 ratio against own SGLang |
|---|---:|---:|---:|---:|
| Shared BF16 A producer | 264.996 | 408.819 | 18.730900 | 1.184293× |
| Direct global weight-to-register delivery | 180.848 | 337.856 | 13.724704 | 1.613539× |

C32 is `46 * sliding + 16 * full`, expressed in milliseconds. Unchanged G1b4 control `cc474883117947c0` measured 11.897664 ms. The shared-A and direct-weight normalized D32 endpoints were respectively 110.213 µs slower (+57.43%) and 29.468 µs slower (+15.36%) than that control in separate runs. [Direct D32 scorecard](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/2026-11-02-driver-d32-mid-comparison.md)

## Shared BF16 A producer

Private H200 job `5adf172986ff43dc`, revision `dcb24972`, physical source `fbca3b75`, completed at 23:03:31 UTC. All 13 correctness tests and 12 benchmarks passed, with 12 complete profiles and 56 startup replays. The score is **1.612794×**, with within-run 95% CI 1.588602–1.636986×. [Owning terminal receipt](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/evidence/2026-11-02-driver-shared-a-first-check/status-wait.raw.json) · [Endpoint calculations](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/evidence/2026-11-02-driver-shared-a-first-check/terminal-analysis.json)

Each endpoint below is `(46 * sliding + 16 * full) / 62`. The control is the unchanged G1b4 job `cc474883117947c0`.

| Load | Tokens | Candidate µs | Control µs | Delta µs | Delta % |
|---|---:|---:|---:|---:|---:|
| decode 1 | 1 | 79.008 | 78.765 | +0.244 | +0.31% |
| decode 16 | 16 | 165.218 | 152.016 | +13.202 | +8.68% |
| decode 32 | 32 | 302.111 | 191.898 | +110.213 | +57.43% |
| verify 1 | 8 | 113.316 | 112.682 | +0.634 | +0.56% |
| verify 16 | 128 | 467.094 | 261.713 | +205.381 | +78.48% |
| verify 32 | 256 | 628.142 | 351.080 | +277.062 | +78.92% |

Per-cell uncertainty below is repeat standard deviation from the raw receipt, in candidate/control order. It describes the within-run samples. There is no recorded paired treatment interval. Both arms have 60 repeats for decode 1 and 20 for other cells.

| Tokens / context | Candidate µs | Control µs | Repeat SD µs, candidate / control |
|---|---:|---:|---:|
| 1 / sliding | 77.288 | 77.036 | 4.803 / 3.912 |
| 1 / full | 83.955 | 83.733 | 3.770 / 4.155 |
| 16 / sliding | 145.873 | 133.037 | 8.209 / 8.072 |
| 16 / full | 220.834 | 206.582 | 8.932 / 11.385 |
| 32 / sliding | 264.996 | 150.875 | 16.823 / 7.812 |
| 32 / full | 408.819 | 309.839 | 21.421 / 8.874 |
| 8 / sliding | 112.006 | 110.761 | 9.737 / 10.201 |
| 8 / full | 117.082 | 118.204 | 9.423 / 10.824 |
| 128 / sliding | 440.654 | 237.040 | 36.795 / 9.045 |
| 128 / full | 543.107 | 332.647 | 26.027 / 10.321 |
| 256 / sliding | 582.258 | 308.622 | 41.963 / 8.556 |
| 256 / full | 760.059 | 473.149 | 36.233 / 8.212 |

## Direct global weight-to-register delivery

Private H200 job `0b684a4065354fc5`, revision `368b85c9`, physical source `a27a1baf`, completed at 23:26:37 UTC. All 13 correctness tests and 12 benchmarks passed, with 12 complete profiles and 56 startup replays. The score is **1.926424×**, with within-run 95% CI 1.897527–1.955320×. [Owning terminal receipt](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/evidence/2026-11-02-driver-direct-weight-first-check/status-wait.raw.json) · [Endpoint calculations](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/evidence/2026-11-02-driver-direct-weight-first-check/terminal-analysis.json)

Each endpoint below is `(46 * sliding + 16 * full) / 62`. The control is the unchanged G1b4 job `cc474883117947c0`.

| Load | Tokens | Candidate µs | Control µs | Delta µs | Delta % |
|---|---:|---:|---:|---:|---:|
| decode 1 | 1 | 78.860 | 78.765 | +0.095 | +0.12% |
| decode 16 | 16 | 156.092 | 152.016 | +4.076 | +2.68% |
| decode 32 | 32 | 221.366 | 191.898 | +29.468 | +15.36% |
| verify 1 | 8 | 118.598 | 112.682 | +5.916 | +5.25% |
| verify 16 | 128 | 335.972 | 261.713 | +74.259 | +28.37% |
| verify 32 | 256 | 442.798 | 351.080 | +91.717 | +26.12% |

Per-cell uncertainty below is repeat standard deviation from the raw receipt, in candidate/control order. It describes the within-run samples. There is no recorded paired treatment interval. Both arms have 60 repeats for decode 1 and 20 for other cells.

| Tokens / context | Candidate µs | Control µs | Repeat SD µs, candidate / control |
|---|---:|---:|---:|
| 1 / sliding | 76.992 | 77.036 | 4.531 / 3.912 |
| 1 / full | 84.231 | 83.733 | 3.865 / 4.155 |
| 16 / sliding | 137.038 | 133.037 | 8.684 / 8.072 |
| 16 / full | 210.873 | 206.582 | 5.757 / 11.385 |
| 32 / sliding | 180.848 | 150.875 | 11.273 / 7.812 |
| 32 / full | 337.856 | 309.839 | 14.145 / 8.874 |
| 8 / sliding | 115.464 | 110.761 | 7.940 / 10.201 |
| 8 / full | 127.608 | 118.204 | 8.348 / 10.824 |
| 128 / sliding | 312.066 | 237.040 | 16.445 / 9.045 |
| 128 / full | 404.703 | 332.647 | 10.639 / 10.321 |
| 256 / sliding | 396.071 | 308.622 | 17.466 / 8.556 |
| 256 / full | 577.137 | 473.149 | 16.135 / 8.212 |

## Interpretation and next evidence

Both candidates change B4/T32 and MAIN/T128/256, using soft routing at 80k context with agentic spread on multibatch loads. Exact routing histograms are absent. All twelve benchmark seeds, GPU node, cluster and driver differ from cc474 despite matching evaluator-file hashes. Unchanged smaller-token paths also drift. These are descriptive cross-run whole-layer losses. The records do not isolate spills, decoding throughput, shared bandwidth or global memory as the cause. [Control receipt](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json)

Three fixed expert-body experiments have now passed validity gates and missed their continuation screens, counting the earlier two-front K128 test. The captain has changed the next allocation to an unchanged-incumbent calibration followed by the fixed MID-at-T32 comparison. A remains ahead of the staged B implementations tested at 128/256. These G outcomes leave that A-versus-B decision intact. [Current allocation and rationale](https://github.com/yelgabo/okc/blob/bdbf35bf/docs/steering/2026-11-02-driver-d32-mid-comparison.md)

The new incumbent diagram labels use integrated-layer profiles. Their component activity ratios compare historical SGLang and G1b4 runs, with smaller SGLang rows omitted. They are approximate profile bounds and do not add up to causal shares of the whole-layer gain. [Diagram evidence correction](2026-11-02-incumbent-inlayer-labels.md).

Reviewed through `bdbf35bf`. New run IDs `5adf172986ff43dc` and `0b684a4065354fc5` are terminal and counted once. Parsed 18 untracked profile JSON files without modifying or staging them. Their job IDs are historical entries in the fetched run logs. Read-only live chat snapshots failed because the durable host was unavailable, so published delivery/acceptance gaps remain explicit. Independent review recomputed both source hashes, score/endpoints and repeat metadata.
