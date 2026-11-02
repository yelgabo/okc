# Architecture performance review, November 2 at 10:44 Vancouver time

Evidence snapshot `db7800a`.

**G’s pre-gamma expert transport passed H200 correctness and missed its paired component speed target.** Private standalone MoE job `56e0c8b5634545e6` finished November 2 at 17:33:23 UTC. Registration `adcfa8e` pinned payload `8f12932a`, with G incumbent builds in placements 0/1 and pre-gamma builds in 2/3. All five official tests passed. The official candidate scored 2.332483× against the judge baseline (95% CI 2.297496–2.367471). The decision on changing G comes from the separate within-process comparison below. [Owning receipt](https://github.com/yelgabo/okc/blob/db7800a/docs/steering/evidence/2026-11-02-worker-moe-component/status-wait.raw.json) · [Terminal decision](https://github.com/yelgabo/okc/blob/db7800a/docs/steering/2026-11-02-worker-moe-component-terminal.md)

Three held-out routing banks (1102031, 1102037, 1102049) each covered normal and skewed inputs. The harness recorded 1,152 timed replays, 216 captures and 54 restorations. All 24 equal-six-load control/candidate ratios fell below 1.01 (range 0.978693–1.002284). Sixteen of 18 affected size/bank/routing combinations had at least one regression of 1% or more. Duplicate placements provide noise estimates. The archive supplies no confidence interval for the paired effect. The exact treatment is parked at the component gate. [Frozen reader and original comparisons](https://github.com/yelgabo/okc/blob/db7800a/docs/steering/evidence/2026-11-02-worker-moe-component/frozen-reader-result.json)

| Tokens | Mean control µs | Mean candidate µs | Change µs | Change |
|---:|---:|---:|---:|---:|
| 1 | 33.959 | 34.181 | +0.223 | +0.66% |
| 8 | 60.173 | 60.045 | -0.128 | -0.21% |
| 16 | 90.504 | 90.656 | +0.152 | +0.17% |
| 32 | 93.296 | 94.217 | +0.921 | +0.99% |
| 128 | 144.656 | 147.441 | +2.785 | +1.93% |
| 256 | 196.451 | 199.229 | +2.779 | +1.41% |

These descriptive differences average the two published placement means per side across the six routing groups. Only the 32/128/256-token paths contain the treatment. The gate uses each bank and cross individually. No whole-layer test followed. Family G retains its public 2.126083× layer entry. The one-token three-launch prototype remains a separate offline design. The A-versus-B conclusion at 128/256 still rests on the existing paired MoE evidence. [Source means](https://github.com/yelgabo/okc/blob/db7800a/docs/steering/evidence/2026-11-02-worker-moe-component/descriptive-load-summary.json)
