# Architecture performance review, November 1 at 12:45 Vancouver time

Evidence snapshot `08d336a`.

**The stage-local sparse-XOR projection lost to archived C2 on all ten target cells.** Public H200 job `dfcfdd01d6f04525` used sources `e27c86d` and `fd21fdf`, registered at `a08f501`. It scored **1.419147×** (95% CI 1.397860–1.440434) versus **1.527806×** (1.504889–1.550724) for C2 job `9eba96894ee1402e`. One official run per candidate and different input seeds, tuner state and baseline measurements limit attribution to a cross-run comparison. The ten targets covered 8, 16 and 32 tokens. All ten overrides executed, all nine tests passed, and the run produced 30 timings and profiles. Startup completed 160 math checks, ten captures and 30 fresh replays. All 40 resource checks passed. Both global and stage-local sparse repair before activation loading are now closed for these implementations. [Terminal result](https://github.com/yelgabo/okc/blob/08d336a/docs/steering/2026-11-01-projection-stage-xor-terminal.md)

| Projection | Tokens | Archived C2 µs | Stage-local µs | Adjusted delta µs | Adjusted change |
|---|---:|---:|---:|---:|---:|
| qkv_gate | 8 | 9.771 | 12.789 | +3.019 | +30.90% |
| unpooler | 8 | 4.535 | 4.819 | +0.285 | +6.27% |
| qkv_gate | 16 | 9.741 | 12.946 | +3.205 | +32.90% |
| o | 16 | 5.366 | 6.701 | +1.334 | +24.87% |
| pooler | 16 | 13.800 | 16.956 | +3.156 | +22.87% |
| unpooler | 16 | 13.557 | 15.568 | +2.011 | +14.83% |
| qkv_gate | 32 | 10.557 | 14.858 | +4.301 | +40.74% |
| o | 32 | 5.869 | 6.819 | +0.950 | +16.18% |
| pooler | 32 | 15.084 | 18.664 | +3.581 | +23.74% |
| unpooler | 32 | 14.721 | 16.520 | +1.800 | +12.23% |

Raw target latencies also rose, by 5.86–30.69%. At 128 and 256 tokens the run used existing C2 paths. The sums of the five adjusted operator medians moved by +1.140 µs (+1.70%) and +1.117 µs (+1.35%) respectively across runs. These cells provide workload coverage only. [All 30 timing rows](https://github.com/yelgabo/okc/blob/08d336a/docs/steering/evidence/2026-11-01-projection-stage-xor-terminal/rows30.csv)

**The four-cell verify hybrid now fits the source limit.** The repaired production core is 952,309 bytes against a limit below 1,000,000. A source audit found that the owned mode-51 software grid join relied on whole-grid progress without a documented guarantee under the ordinary launch. The separate repair uses cooperative launch with PDL disabled and traps if an explicit readiness poll expires. Source review, host tests and local compilation support the change. H200 admission, startup, correctness and timing still require qualification. The regenerated 970,042-byte private diagnostic remains on hold. The first two reset studies retain priority with repaired stage and short-path payloads of 907,822 and 904,020 bytes. [Fit and reset handoff](https://github.com/yelgabo/okc/blob/08d336a/docs/steering/2026-11-01-reset-and-production-fit.md) · [Progress review](https://github.com/yelgabo/okc/blob/08d336a/docs/steering/2026-11-01-owned-progress-repair-review.md)

No new layer measurement changes the retained G score of 2.126083×. A’s tested advantage over B at 128/256 and the earlier C, D and E conclusions still rest on the previous receipts. The new projection timing and the hybrid’s local build results supply no isolated family gain.
