# Architecture performance review, November 2 at 15:43 Vancouver time

Evidence snapshot `13f5e41`.

The G fork’s fixed K128 two-front expert passed 13 correctness tests and all 12 whole-layer benchmarks in private H200 check `bf8671b4bfa845a2` (source `15688256`). It scored **1.965158×** (within-run 95% CI 1.935681–1.994636×). The team parked this implementation and canceled confirmation. Public G1b4 `607eb52bdaf24160` retains 2.126083×. [Terminal decision](https://github.com/yelgabo/okc/blob/13f5e41/docs/steering/2026-11-02-driver-k128-terminal.md)

The two-CTA candidate changed expert execution at 32, 128 and 256 tokens. The following complete-layer endpoints weight sliding and full context 46:16. Control `cc474883117947c0` ran the unchanged G1b4 in a separate job. Both jobs used the same evaluator files but different inputs, GPU nodes and drivers. The candidate used soft routing at 80k context with agentic spread on multibatch loads. Each timing cell had 20 repeats, except decode 1 with 60. No paired treatment interval was recorded. [Owning receipt](https://github.com/yelgabo/okc/blob/13f5e41/docs/steering/evidence/2026-11-02-driver-k128-first-check/status-wait.raw.json) · [Endpoint calculations](https://github.com/yelgabo/okc/blob/13f5e41/docs/steering/evidence/2026-11-02-driver-k128-first-check/terminal-analysis.json)

| Load | Tokens | Candidate µs | Control µs | Change µs | Change % |
|---|---:|---:|---:|---:|---:|
| decode 1 | 1 | 79.325 | 78.765 | +0.560 | +0.71% |
| decode 16 | 16 | 157.215 | 152.016 | +5.199 | +3.42% |
| decode 32 | 32 | 208.955 | 191.898 | +17.057 | +8.89% |
| verify 1 | 8 | 119.568 | 112.682 | +6.886 | +6.11% |
| verify 16 | 128 | 319.297 | 261.713 | +57.584 | +22.00% |
| verify 32 | 256 | 419.045 | 351.080 | +67.965 | +19.36% |

The observed regressions cover the complete layer. This run cannot separate the effects of the grid, tile geometry or scheduling. The next funded G proposal keeps 256 output rows per unit and four compute warpgroups. A separate producer would decode weights into shared BF16 operands. It targets 32/128/256 tokens and has no GPU timing. This is an expert pipeline, distinct from Family E attention. [Next expert implementation](https://github.com/yelgabo/okc/blob/13f5e41/docs/steering/2026-11-02-driver-shared-weight-pipeline.md)

A still leads the staged B implementations tested at 128/256. The grouped-contiguous work census found that the native path already skips unoccupied token tiles. The proposed scheduler has no demonstrated reduction in padded matrix work at equal geometry. The FLUTE screen also found no qualified routed two-GEMM configuration with the required scaling. Neither source screen adds a timing result. [Scheduled work census](https://github.com/yelgabo/okc/blob/13f5e41/docs/steering/2026-11-02-worker-grouped-engine-work-decision.md) · [FLUTE source screen](https://github.com/yelgabo/okc/blob/13f5e41/docs/steering/2026-11-02-research-flute-engine-screen.md)
