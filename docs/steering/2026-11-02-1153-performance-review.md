# Architecture performance review, November 2 at 11:53 Vancouver time

Evidence snapshot `1786dbf`.

**The one-token, three-launch expert path lost in all six targeted routing cases and was parked before layer testing.** Private H200 MoE check `28248eb5e5854041` tested native source `dbdd3381`, wrapper `ce14d11f`, registered at `9d7937d2`. The one-token profile confirmed gate/up, down and norm/combine all executed. The official candidate passed five tests and produced six benchmark cells and six profiles. It scored 2.165858× versus SGLang (95% CI 2.133370–2.198346). The within-process comparison against the fused G control determined the decision. [Terminal receipt and profile analysis](https://github.com/yelgabo/okc/blob/1786dbf/docs/steering/2026-11-02-worker-t1-moe-terminal.md)

Four independent placements (two controls and two candidates) covered held-out banks 1102031, 1102037 and 1102049 with normal and skewed routing. The instrument recorded 1,152 timed replays, 216 captures, 54 restorations and clean final state. None of the 24 six-load comparisons reached the required 1.01 factor. All six targeted cells tripped the 1% regression rule. The one-token control/candidate geometric factor was 0.707998. The paired evidence supplies duplicate noise estimates and no formal paired confidence interval. [Frozen decision and individual medians](https://github.com/yelgabo/okc/blob/1786dbf/docs/steering/evidence/2026-11-02-worker-t1-moe-check/frozen-reader-result.json)

| Tokens | Mean fused control µs | Mean candidate µs | Latency change |
|---:|---:|---:|---:|
| 1 | 34.143 | 48.353 | +14.211 µs (+41.62%) |
| 8 | 59.811 | 60.269 | +0.459 µs (+0.77%) |
| 16 | 90.409 | 90.412 | +0.003 µs (+0.00%) |
| 32 | 91.455 | 91.732 | +0.277 µs (+0.30%) |
| 128 | 144.997 | 145.336 | +0.339 µs (+0.23%) |
| 256 | 197.500 | 197.548 | +0.048 µs (+0.02%) |

These descriptive means average the published placement medians equally over both copies and six routing cases. Only the one-token row exercised the three-launch treatment. The 128/256 rows measure variation on fused paths shared by both arms. The exact implementation is closed with no layer follow-up. Its standalone result gives no new A-versus-B decision at 128/256. Family G retains 2.126083× on the layer board. [Scoped decision](https://github.com/yelgabo/okc/blob/1786dbf/docs/steering/2026-11-02-worker-t1-moe-terminal.md)

The earlier reference-repository access failure was traced to a wrong repository URL. The correct keell0renz source is accessible and matches the archived reference. No new comparator run follows from that correction. [Source identity correction](https://github.com/yelgabo/okc/blob/1786dbf/docs/steering/2026-11-02-captain-source-access-correction.md)
