# Architecture performance review, November 2 at 04:32 Vancouver time

Evidence snapshot `57930d0`.

**G1b4 retains the verified layer score of 2.126083×.** Four-cell verify job `1fad70bbe3854cbc` ended in a refunded infrastructure error with no correctness or timing records. The legal-fixture short-counter comparison completed on the repaired common control. Its twelve matched six-load comparisons ranged from −0.183% to +0.283%, each below the 1% threshold. [Four-cell terminal record](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-owned-verify.1fad70bbe3854cbc.terminal.json) · [Matched counter decision](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-rmw-legal-fixture-terminal.md)

Two subsequent private layer candidates passed all 13 official tests. G224 scored 1.974665× and the fully timed SiLU table scored 2.073289×. G224 was parked after its long-load comparisons. Timed SiLU missed its 1% follow-up threshold. Differences versus public G1b4 span separate runs with different seeds and machines. The run records below distinguish those observations from the matched counter experiment. [G224 decision](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-02-captain-g224-terminal-decision.md) · [Timed SiLU decision](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-02-captain-timed-silu-terminal-decision.md)


### G layer qualification and results

**G1b4 stage release remains the public incumbent**, job `607eb52bdaf24160`, 2.126082676× (95% CI 2.094191436–2.157973917). The previously accepted four-cell verify hybrid `1fad70bbe3854cbc` has a terminal infrastructure error and an explicit refund. The receipt contains no tests, timing or score. Its 128/256-token hybrid benefit remains unmeasured. [Public receipt](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/scored.607eb52bdaf24160.json) · [Refunded hybrid](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-owned-verify.1fad70bbe3854cbc.terminal.json)

| H200 job | Candidate and control | Recorded outcome |
|---|---|---|
| [3fd1317143304bd5](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-common-rmw.3fd1317143304bd5.terminal.json) | Public common repairs + short-counter RMW, faa1befe, upload 97935e7d | 56 native startup replays completed before the 295 s preparation guard failed. Zero official tests or timings. The log does not identify the slow phase. |
| [c1931f0c9e834485](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-common-rmw-v2.c1931f0c9e834485.json) | Private revised RMW ON, a0b6af2, upload 978b29f9 | Actual path exercised. 13 tests, 12 timing/profile cells, 109.270 s preparation. 2.121241049× (2.089422434–2.153059665). Point score 0.228% below historical 607e in an independent run. |
| [16b4db27421d4714](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-common-rmw-pair.16b4db27421d4714.json) | Matched repaired-common OFF/ON, 2686a765, upload c4a5ee29 | H200 setup failed with CUDA 719 during occupied/empty graph replay or synchronization. No complete comparison, score or CI. The fixture diagnosis did not localize the fault. |
| [c5e4f89672f743bb](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-common-rmw-pair-v4.c5e4f89672f743bb.json) | Corrected distinct-ID fixture, d5f29167, upload 785a90ca. Common OFF 4cbb13a9 versus RMW ON 2884adb6 | Valid comparison. All three banks failed the six-load gain gate. Official judged OFF passed 13 tests at 2.112499365× (2.080811875–2.144186856). RMW-only promotion closed. |
| [9adfd4dbc98f4998](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-outgoing-pair.9adfd4dbc98f4998.json) | Outgoing dependency-edge comparison, 3d00a62, upload f25d714e | Graph-query binding failed during setup. Zero tests/timings. This run measures no PDL effect. |
| [6f83ee875dd847c1](https://github.com/yelgabo/okc/blob/57930d0/muse_decode_layer/profiles/g-outgoing-driver-pair.6f83ee875dd847c1.json) | Repaired driver-query comparison, c40930c5, upload ad314060 | Routing equality assertion failed after 44 query contexts. Mismatch cause unresolved. Zero official tests/timings. Exact seam campaign closed on qualification failure. |
| [e1271ad5e5e34b24](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/evidence/2026-11-02-worker-g224-notice/terminal-status-resumed.json) | Private G224, preregistration 5b997e07, upload 0d84d0e3, native 53d09576 | 13 tests, 12 timing/profile cells, 56 startup replays. 1.974664686× (1.945044715–2.004284656), 98.750 s preparation. Exact candidate parked. |
| [dc62cff972934f54](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-02-captain-timed-silu-terminal-decision.md) | Private fully timed WGS BF16 SiLU table, 9a1f6e4, upload d70a1bcc | 13 tests, 12 timing/profile cells, 56 startup replays. 2.073288676× (2.042189346–2.104388007), 112.536 s preparation. No favorable 1% screening signal. Exact candidate parked. At this snapshot the captain published an independent recomputation after reading the worker receipt. Worker raw archive publication was pending. |

RMW's corrected paired test used four independent mutable arm placements, 24 randomized cold rounds per cell, six loads and three routing banks (993811/993817/993821). All 12 bank/cross six-load log gains lay between −0.001829079 and +0.002828144. None cleared the 1% and duplicate-noise gate. At 16-token sliding, bank 993821 also exceeded its regression allowance (+1.152 µs against a 1.020 µs cap). At 128 tokens the four-cross gain range across banks was −0.1705% to +0.3583%, and at 256 it was −0.3888% to +0.2371%. For OFF0 versus ON0, weighted latency deltas by bank were −0.755/−0.889/−0.253 µs at 128 and +0.613/−0.645/+1.335 µs at 256. The 128/256 paths do not use the short-counter treatment. The exact ON/OFF comparison closed with no conditional fourth run. [Method, checks and all arm medians](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-rmw-legal-fixture-terminal.md)

G224 and the timed SiLU table each ran once on fresh inputs. The following are weighted sliding/full latency differences against the earlier public 607e job. They describe separate runs with different seeds and machines. Neither is a paired treatment estimate.

| Tokens | Historical 607e | G224 difference | Timed SiLU difference |
|---|---:|---:|---:|
| 128 | 269.347 µs | +39.372 µs (+14.617%) | +6.102 µs (+2.2655%) |
| 256 | 359.297 µs | +73.887 µs (+20.564%) | −0.802 µs (−0.2231%) |

Against the separate repaired-common OFF history, G224 was 15.790% slower at 128 and 21.698% slower at 256. Timed SiLU's independent full score was 2.4832% below public 607e. These observations led to parking both candidates. No results isolate a spill, occupancy, pipeline or table effect. The public four-cell infrastructure failure, the failed outgoing-edge instruments and the valid matched RMW timing decision represent three different outcomes. [G224 calculation](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/evidence/2026-11-02-captain-g224-terminal/verification.json) · [SiLU calculation](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/evidence/2026-11-02-captain-timed-silu-terminal/recompute.json)

Two short-path ideas stopped before H200 timing. The paired-rounding build preserved register and spill counts but added eight permutations and extended the dependency chain into the exponential. Moving the scale across the required BF16 rounding changed finite outputs in the CPU check. These are compiler and numerical feasibility decisions. [Paired-rounding screen](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-g-short-pair-rne.md) · [Scale feasibility](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-g-short-scale-feasibility.md)

#### Timed SiLU and G224 workload detail

| Load | Tokens | Table eligible | Historical control µs | Candidate µs | Cross-run difference µs (%) |
|---|---:|---|---:|---:|---:|
| decode 1 | 1 | No | 78.549 | 78.757 | +0.208 (+0.26%) |
| decode 16 | 16 | No | 158.241 | 160.430 | +2.189 (+1.38%) |
| decode 32 | 32 | Yes | 191.740 | 198.814 | +7.074 (+3.69%) |
| verify 1 | 8 | No | 115.991 | 114.822 | -1.169 (-1.01%) |
| verify 16 | 128 | Yes | 269.347 | 275.449 | +6.102 (+2.27%) |
| verify 32 | 256 | Yes | 359.297 | 358.496 | -0.802 (-0.22%) |

Each value weights sliding and full context 46:16. All six affected profile cells contain one table-build call, displayed as 1.0 µs in five cells and 1.2 µs in one. Those rounded profiler times cannot determine the isolated cost of lookup or the prior SiLU path. [Per-cell times and reconstruction](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/evidence/2026-11-02-captain-timed-silu-terminal/recompute.json)

**G224 cross-run detail.** Private whole-layer job `e1271ad5e5e34b24`, frozen native pin `53d09576`, publication `217a98c`, passed 13 official tests, 12 benchmark cells and 56 startup replays. Its score was 1.974665× (95% CI 1.945045–2.004285). Both historical controls used different seeds from all 12 current cells. The team parked G224. At 128/256 tokens the weighted per-token-load comparisons were:

| Tokens | G224 µs | Public 607 µs | Difference µs (%) | Repaired OFF c5e4 µs | Difference µs (%) |
|---|---:|---:|---:|---:|---:|
| 128 | 308.719 | 269.347 | +39.372 (+14.62%) | 266.619 | +42.099 (+15.79%) |
| 256 | 433.185 | 359.297 | +73.887 (+20.56%) | 355.950 | +77.235 (+21.70%) |

[Weighted G224 evidence](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/evidence/2026-11-02-captain-g224-terminal/verification.json) [G224 gate and remaining uncertainties](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-02-captain-g224-terminal-decision.md)

The standalone MoE and attention run ledgers and profile trees have no new entries in this interval. SiLU and G224 are Family G whole-layer results. The comparison between A and B at 128/256 tokens still rests on prior paired MoE evidence. The expert-to-unpool PDL source screen stopped before a GPU test. Its candidate requires first-to-last CTA retirement spread and an independent-weight prefix large enough to improve the short path. Actual overlap and critical-path gain remain unmeasured. [PDL scope and resource screen](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-g-expert-unpool-pdl-edge.md)

### Projection, prefill and scoreboard context

The interval `d79089b..57930d0` adds no standalone projection or prefill receipts, candidate changes or ledger entries. C2 job `9eba96894ee1402e` holds the projection board entry at **1.527806483×**. The last stage-local sparse-XOR test (`dfcfdd01d6f04525`) and global test (`2b537518091248f1`) both lost across their ten targeted 8/16/32-token cells. The 128/256-token paths ran C2 code. There is no new timing, paired control or layer integration result for either representation in this interval. Prefill remains deferred. [Projection ledger](https://github.com/yelgabo/okc/blob/57930d0/docs/runs.md) · [Prefill ledger](https://github.com/yelgabo/okc/blob/57930d0/docs/runs-prefill.md)

For the 128/256-token layer workloads, the archive review recovered the known fixed cuBLASLt policy from existing artifacts. The profile rows combine multiple calls, so the review could not establish operation-level exposed time or a gain from replacing the selected plans. The team closed this specific long-dense comparison. [Long-dense archive review](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-long-dense-archive-join.md)

The November 2 07:14 UTC snapshot recorded projection rank **339** with C2, standalone MoE **411** at 2.215711562×, and attention **237** at 2.604290205×. Subsequent 10:51:40 UTC data put both overall and layer at **345**, with G job `607eb52bdaf24160` still at **2.126082676×**. The layer rank-100 score was 2.262745075×, about 6.43% above the incumbent. The overall endpoint reports rank only. [Component board snapshot](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-captain-rank0714-next-gate.md) · [10:51 board capture](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/evidence/2026-11-02-captain-timed-silu-decision/boards.json)
