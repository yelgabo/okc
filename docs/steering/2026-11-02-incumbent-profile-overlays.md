# Profiling overlays on the incumbent diagrams

Added a [profiled architecture page](../incumbent-architecture-profiled.html) and standalone [WGS](../figures/incumbent-wgs-profiled.svg) and [phase](../figures/incumbent-phases-profiled.svg) figures. The page preserves the original charcoal/gold design and links the full-layer, WGS and phase views to the same selected workload. The [local preview](http://127.0.0.1:4387/artifact/c771edd2bc0f7a01/incumbent-architecture-profiled.html#inside-moe) serves the revised page.

## What is measured

The requested screenshot uses public607 `607eb52bdaf24160`, source SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. The overlay deliberately retains that historical profile snapshot and the existing separate SGLang baseline `7d42771c3384449d`. It is not a new profile run or a claim that later identical-source timings match.

At the default 128-token sliding workload, recorded candidate GPU activity is:

| Group | Activity | Share of summed activity |
|---|---:|---:|
| Fused expert kernel | 155.3 us | 62.0% |
| All dense/router GEMMs | 70.8 us | 28.3% |
| Attention group | 11.8 us | 4.7% |
| Listed routing/boundary rows | 12.5 us | 5.0% |
| Unlisted/rounding remainder | 0.1 us | About 0.04% |

The denominator is 250.5 us of recorded activity. The same workload's benchmark endpoint median is 247.131 us. Kernel activities can overlap, include waits and come from a separate profiling pass. These shares describe the profile, not additive latency or removable time.

The complete 155.3 us envelope appears above both expert diagrams. It covers all loading, waiting, Phase A/B work and combination. The repeated total is one kernel, not two costs. Internal producer, consumer, readiness and combine labels say “not isolated.” Shared-memory rings are storage, not separately timed launches. A valid internal trace is still missing; the perturbed observer discussed in the [phase-attribution brief](2026-11-02-cuda-b4-phase-attribution-task.md) is not used as timing evidence.

The page adds a group-activity bar and gold cost labels on the layer diagram. Dense rows remain pooled across QKV, O, pooler/router and unpooler because their printed names/call aggregation do not uniquely attribute every box. Existing performance ratios remain separate from the new cost labels.

## Verification and scope

[Verification evidence](evidence/2026-11-02-incumbent-profile-overlays/verification.json) pins the input artifact hash and repository evidence base. The original diagram was located at `/Users/gretchengabo/GitHub/okc/docs/incumbent-architecture-2026-11-02.html` after checking this checkout, its tracked/hidden paths, the canonical checkout and the existing preview's recorded source path. The revised page is a new file; the original artifact was preserved.

All 244 embedded visible profiler rows matched their owning receipt's printed name, rounded time and call count. Duplicate truncated kernel names were checked as a multiset of name/time/call tuples. All 24 candidate/baseline profile totals matched. The embedded measurement data is unchanged, and the current incumbent file matches the recorded source SHA256.

All 12 token/context combinations were exercised through real browser controls. The whole-kernel time and phase-envelope time agree. At 1, 8 and 16 tokens, the page mutes the B4/MAIN diagrams and names the selected Tiny/Tiny2/MID route. Both enlarged diagrams and the standalone exports were rendered in Chrome and inspected; no label collisions were observed. Saved [WGS preview](evidence/2026-11-02-incumbent-profile-overlays/wgs-profiled.png) and [phase preview](evidence/2026-11-02-incumbent-profile-overlays/phases-profiled.png) show the default workload. Independent review was unavailable under the current no-new-agent instruction.

No kernel source, job, quota, timer or worker assignment changed. Current MoE/BF16 priorities remain with their existing owners. The next evidence gate for internal numbers is a selected-source phase trace with a priced observer control and bounded stop. Until that exists, retain the unknown labels rather than deriving stage times from thread counts, FLOPs or source size.
