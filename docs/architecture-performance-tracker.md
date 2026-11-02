# Architecture performance tracker

Last review: 2026-11-02 12:48 MST. Evidence snapshot: `ba0e1ff`.
First review reconciles progress since `aab7d75`, the snapshot in the
[architecture-family guide](architecture-families-2026-10-29.md).
This is a review of recorded experiments, not a new benchmark run.

## Current family status

Negative timing deltas mean faster. Percentages below are elapsed-time changes against the
named control, not changes in aggregate score. Families overlap; their gains cannot be added.

| Family | Current result | What has actually been measured | Visibility gap |
|---|---|---|---|
| A. Hybrid persistent/legacy MoE | Retained owned baseline; new variants fail gates | Position map, token locality, multiply decoder, direct band32 and cached band32 fail their aggregate/regression gates. | Cached schedule recovers direct-band overhead but fails against original. Physical traffic opportunity remains open. Historical standalone skinny/main47 progress remains unqualified. Both disabled repairs failed the compiler spill gate. |
| B. Staged grouped MoE | New masked library schedule loses | GROUPED_MASKED versus51 loses about 145-235 us across 128/256 layer/std cases, with math checks passing. Earlier45+46 losses remain. | Failure is scoped to tested library schedule; two GEMMs already exceed owned whole-call time in the profiled 256 case. New grouped-contiguous Humming proposal stopped before measurement. |
| C. Owner-completes tiles | Earlier ownership gains retained | The complete private VP2 design wins its eight-token sliding cell. Both owned cluster implementations previously lost. | The comparison leaves the contribution of output ownership unresolved. |
| D. Dependency-aware scheduling | Dynamic claiming retained; new phase-A claim layouts lose | Phase-A modes 62/63 and K=0 controls 66/69 lose; lean claimed mode 68 is not timed. Earlier split-tail 52/53 losses stand. | Full ready-only queue remains untested. Reopen phase-A claims only after the no-work control ties mode 51 within 1 us at 128/256. |
| E. Attention producer-consumer pipeline | Private VP2 cell win, six-load gate failed | VP2 took 10.168 µs versus 10.784 µs for the owned control on eight-token sliding attention. All three held-outs missed the full-score gate. | VP2 only changes the eight-token target. A later half-merge-preserving continuity source passed CPU checks and is frozen before H200 testing. |
| F. Donor-based baseline/hybrid | Historical alygg77 measurement track | Prior F comparisons and owned C3 router gains remain recorded. Two new QKV128 variants lose despite correctness/resource passes. | G has a separate attributed-fork policy. Projection C2 remains best scored, with C3 paired component wins retained. |
| G. Attributed keell0renz layer fork | Retained layer 2.126083× | Matched short-counter test missed its gate. G224 and timed SiLU passed correctness and missed performance triage. | Four-cell verify ended in a refunded infrastructure error. Outgoing dependency tests failed setup. Both fresh checks completed. The short serial policy missed confirmation gates. PR check failed target host compilation. Pre-gamma and one-token three-launch expert paths each passed standalone correctness and missed all 24 six-load comparisons. Both are parked before layer testing. |

## November 2 review at 12:48 Vancouver time

**Two complete expert-engine proposals stopped at source review.** The grouped-contiguous Humming design is a new Family B proposal. Reviewers found compatible formats in principle, but lacked an exact Muse configuration, numerical qualification and a whole-layer timing budget covering the added operations. Its outcome is untested. The earlier measured loss belongs to the masked grouped implementation. The second proposal would store a prepared BF16 weight bank. The task requires stored NVFP4 weights to remain NVFP4. The review also modeled 3.66× the logical weight bytes of eligible SC6 packing, or 3.56× those of ordinary packing. That estimate measures neither HBM traffic nor execution time. The proposal letters in the design report refer to those two alternatives. [Engine comparison](https://github.com/yelgabo/okc/blob/ba0e1ff/docs/steering/2026-11-02-worker-broad-expert-architecture.md) · [Independent feasibility review](https://github.com/yelgabo/okc/blob/ba0e1ff/docs/steering/2026-11-02-research-broad-expert-feasibility.md)

**Family E has an isolated continuity source, currently frozen.** Revisions `19ab26c` and `6e18116` preserve the decode half-merge that confounded the older toggle. Seven CPU tests passed. A separate audit exercised 16 final-entry tuples. The normal selector falls back at batch 33, while directly overriding the module at the lower-level launch entry can bypass that guard. That finding scopes the source claim. A simultaneous duplicate draft was retired. Under the existing workload weights, even a hypothetical 7 µs whole-call saving in each of four affected full-context cells would yield only a 0.560% layer-score improvement. The team froze this candidate on that opportunity screen. [Source reconciliation](https://github.com/yelgabo/okc/blob/ba0e1ff/docs/steering/2026-11-02-worker-attention-continuity-reconciliation.md) · [Layer sensitivity calculation](https://github.com/yelgabo/okc/blob/ba0e1ff/docs/steering/2026-11-02-attention-roi-and-step-change-priorities.md)

There are no new target job IDs, controls, timings or confidence intervals in this interval. A remains ahead of the tested staged B variants at 128/256 tokens. The newer Humming configuration and isolated attention source have no layer measurements at either size. G1b4 retains its verified 2.126083× result.

## November 2 review at 11:53 Vancouver time

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

## November 2 review at 10:44 Vancouver time

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

## November 2 review at 09:46 Vancouver time

The standalone MoE adapter for G’s pre-gamma expert candidate passed its CPU source and storage checks. Source `ece009b` reproduces the pinned control and candidate builds in a 323,514-byte payload. Ten adapter tests passed. The adapter accepts the current official 32-expert shard starting at zero and rejects other ranges. Its H200 build, FP32 numerical checks, replay and timings remain pending. [Adapter and source checks](https://github.com/yelgabo/okc/blob/ca5cf8f/docs/steering/2026-11-02-top100-moe-adapter.md)

The new allocation covers construction of a paired component harness and receipt reader. The proposed timing gate requires a six-load geometric gain of at least 1% on every held-out routing bank and every cross, beyond twice the duplicate noise. The candidate targets 32/128/256 tokens. There is no dispatch ID or measured delta at any size. A component pass would justify a later layer test. [Registered component gates](https://github.com/yelgabo/okc/blob/ca5cf8f/docs/steering/2026-11-02-top100-moe-adapter.md)

The existing CPU Torch runtime is available for host validation. The separate 918,254-byte layer harness still needs full generated-entry storage and ownership checks. Its preparation stops before native builds. The independent review narrowed the earlier CPU coverage claims for capacity, gamma and finalization. These are qualification updates only. G retains 2.126083×, and the A-versus-B decision still rests on existing paired MoE evidence. [Layer harness](https://github.com/yelgabo/okc/blob/ca5cf8f/docs/steering/2026-11-02-top100-expert-harness.md) · [Coverage review](https://github.com/yelgabo/okc/blob/ca5cf8f/docs/steering/2026-11-02-top100-expert-review.md)

## November 2 review at 08:49 Vancouver time

**The graph-owned PR buffer check failed during host compilation.** Private layer job `60b38a62a135442c` used the candidate registered at `01b8e23` (882,844 bytes, SHA256 prefix `0f90a2a8`). An explicit C++17 flag followed C++20 on the actual compiler command, leaving a language standard below the target Torch headers’ requirement. The job exited setup with code 112 before any official tests, benchmark cells or profiles. Compression selection, buffer lifetime on H200 and latency at 128 and 256 tokens remain unmeasured. There is no score, timing delta, repeat count or confidence interval for this treatment. The separate queue and worker durations measure service activity. [Terminal receipt](https://github.com/yelgabo/okc/blob/049b223/docs/steering/evidence/2026-11-02-worker-pr-target/target/terminal-status.json) · [Build failure analysis](https://github.com/yelgabo/okc/blob/049b223/docs/steering/2026-11-02-worker-pr-target-terminal.md)

The language-standard issue now has an offline regression check. At the 15:30 UTC status checkpoint the private quota was zero, five scored slots remained, and the job list was empty. A private retry requires available quota. Public G1b4 job `607eb52bdaf24160` retains 2.126083×. This PR backing experiment belongs to layer context. The A-versus-B choice at 128/256 still rests on the paired MoE measurements. [Quota and status](https://github.com/yelgabo/okc/blob/049b223/docs/steering/2026-11-02-pr-build-failure-monitor.md)

**Family E source screen.** The standalone attention path already keeps the ring across segments. In the layer, the proposed continuity switch also changes decode merging and the synchronization point. CPU inspection covered 12 actual layer launch paths and four standalone plans. The team closed this switch as an isolated test. No GPU result was recorded for it. [Attention screen](https://github.com/yelgabo/okc/blob/049b223/docs/steering/2026-11-02-top100-attention-prototype.md)

**Expert prototypes awaiting target evidence.** The one-token, three-launch expert design has local source, CPU and secondary compiler evidence. A separate B4/MAIN candidate changes the stored intermediate before gamma, targeting decode at 32 and verify at 128/256 tokens. Its package reproduces the original five control loader calls and binds the candidate to prepared state before capture. Three CPU tests passed. Runtime ownership, fresh replay correctness and whole-layer timing still require qualification. Neither prototype supplies a new family score. [One-token prototype](https://github.com/yelgabo/okc/blob/049b223/docs/steering/2026-11-02-worker-t1-three-launch.md) · [Pre-gamma source](https://github.com/yelgabo/okc/blob/049b223/docs/steering/2026-11-02-top100-expert-prototype.md) · [Comparison package](https://github.com/yelgabo/okc/blob/049b223/docs/steering/2026-11-02-top100-pair-package.md)

## November 2 review at 07:49 Vancouver time

The graph-owned PR buffer candidate passed host syntax and CPU checks at worker source `178c772` (881,883 bytes, SHA256 `4c6d3f13e0b9c72c4d4ab67f440a35f6ae39a954c818f78d35f3ff3d9090423b`). The earlier cleanup protocol stopped at a source-level capture conflict. Its replacement waits for the buffer owner’s completion events and restores the helper’s prior capture mode even on errors. [Cleanup finding](https://github.com/yelgabo/okc/blob/218a048/docs/steering/2026-11-02-captain-pr-cleanup-capture.md) · [Host qualification and allocation](https://github.com/yelgabo/okc/blob/218a048/docs/steering/2026-11-02-captain-pr-host-gate.md)

One private layer check is conditionally allocated. The worker must first add a preparation receipt that identifies actual owner-tagged PR plans at 128 and 256 tokens. At this snapshot there is no dispatch ID, H200 correctness result or timing for the candidate. A run without the intended plans would establish no treatment. The fresh G1b4 control is `cc474883117947c0`. Any future comparison against that run would be cross-run. [Host qualification and allocation](https://github.com/yelgabo/okc/blob/218a048/docs/steering/2026-11-02-captain-pr-host-gate.md)

This buffer proposal is separate layer context. The A-versus-B decision still rests on the existing MoE comparisons. Public G1b4 retains 2.126083×.

## November 2 review at 06:47 Vancouver time

**The short serial pooler/router candidate passed correctness and was parked.** Private job `e4c8f00532b74d15` tested source `8c0bfba`, file hash `5d883c43`, against fresh G1b4 control `cc474883117947c0`, file hash `ba9e327c`. The serial candidate scored 2.093073× (95% CI 2.061677–2.124469) and the control scored 2.109201× (2.077563–2.140839). Each ran once on the same H200 node, passed 13 official tests, and produced 12 timing cells and profiles. The 56 startup replays in each job exercised the original startup path. [Terminal decision](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-serial-terminal-decision.md) · [Worker archive](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-worker-short-serial-v2-terminal.md)

| Load | Tokens | Fresh control µs | Serial µs | Difference µs | Change |
|---|---:|---:|---:|---:|---:|
| d1 | 1 | 78.765 | 81.741 | +2.977 | +3.78% |
| d16 | 16 | 152.016 | 157.256 | +5.240 | +3.45% |
| d32 | 32 | 191.898 | 194.559 | +2.661 | +1.39% |
| v1 | 8 | 112.682 | 117.191 | +4.509 | +4.00% |
| v16 | 128 | 261.713 | 269.831 | +8.118 | +3.10% |
| v32 | 256 | 351.080 | 361.511 | +10.431 | +2.97% |

Four short-path signatures confirm that the one- and eight-token cells ran serially. All eight longer-cell kernel multisets matched. The table uses 46:16 sliding/full weights. The control-to-candidate latency factor across six loads was 0.969827×, below the 1.01 confirmation threshold, and every load exceeded the 1% regression limit. All 12 seeds differed across sessions. The longer cells also slowed despite using the same dispatch, so the cross-run timings cannot isolate the cost of serialization. G1b4 retains the public score of 2.126083×. [Comparison and selected paths](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-serial-terminal-decision.md)

**B1 is a separate 128/256-token buffer-lifetime study.** The plain-owner prototype stopped before candidate generation because synchronization alone could not guarantee the allocation would outlive a caller-owned graph. The CUDA graph user-object alternative has a conditional source-level ownership argument. H200 support, correctness, compression and timing are unmeasured. The last private check remains reserved. The B1 name refers to a buffer proposal and adds no staged grouped-MoE evidence to Family B. [Ownership decision](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-b1-graph-owner-discriminator.md) · [Independent lifetime review](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-research-b1-user-object-proof.md)

## November 2 review at 05:28 Vancouver time

The renewed G evaluation has two distinct entries. Exact G1b4 control job `cc474883117947c0` was accepted and queued with an empty result at this snapshot. Its purpose is to establish contemporary timings across all 12 cells. The serial candidate is registered at source `8c0bfba`, SHA256 `5d883c43db730990246d07a0e52800e01b3161973705b8cb785c26e0d3f08933`. It passed five CPU tests with mocked device boundaries and awaits the control’s terminal receipt before dispatch. [Control receipt](https://github.com/yelgabo/okc/blob/8c0bfba/docs/steering/evidence/2026-11-02-worker-short-serial-v2/control/accepted-status.json) · [Serial candidate registration](https://github.com/yelgabo/okc/blob/8c0bfba/docs/steering/2026-11-02-worker-short-serial-v2-registration.md)

On eligible one- and eight-token inputs, the candidate runs the BF16 pooler followed by the FP32 router. The 128- and 256-token cells use incumbent dispatch and will help identify between-run drift. The plan permits a confirmation request if the six-load latency factor reaches 1.01, every load stays within the 1% regression limit, and the short workloads support the result. The earlier deadline stop produced no GPU timing. [Evaluation plan](https://github.com/yelgabo/okc/blob/8c0bfba/docs/steering/2026-11-02-captain-profile-refresh.md)

The canonical archive for timed SiLU job `dc62cff972934f54` is now published at `d20670d`. This supplies the previously missing raw evidence for the 2.073289× result already counted. Family G’s retained public score is 2.126083×. This interval adds no measured family gain or loss. [Archived worker receipt and checks](https://github.com/yelgabo/okc/blob/8c0bfba/docs/steering/2026-11-02-worker-timed-silu-check.md)

## November 2 review at 04:32 Vancouver time

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

## November 1 review at 14:49 Vancouver time

Historical checkpoint. The November 2 review records the terminal receipts and replaces the pending common-control and short-counter statuses below.

**G’s previous reset queue is on hold.** The new plan starts with recovery of accepted job `1fad70bbe3854cbc`. At 21:38 UTC the service still returned HTTP 530, so correctness, timing and score for all four 128/256-token verify cells remain unknown. The retained layer is still `607eb52bdaf24160` at 2.126083×. The previous stage comparison, short-path trace and private four-cell artifacts are held. [Integration plan and service status](https://github.com/yelgabo/okc/blob/d79089b/docs/steering/2026-11-01-g-minimal-delta-plan.md)

The proposed common control `4cbb13a9` (871,077 bytes) combines attention progress and pre-expert coefficient repairs. Source and CPU checks passed. H200 qualification is pending. Of the two implemented performance hypotheses for the G0 integration gap, only the short final-counter read-modify-write (RMW) treatment survived its local checks. The old `f0426f71` harness requires rebasing and a new registration against the qualified common control. A proposed PDL dependency edge and a separate short-expert dataflow investigation remain at the source-analysis stage. [Candidate inventory](https://github.com/yelgabo/okc/blob/d79089b/docs/steering/2026-11-01-g-candidate-funnel.md)

The post-expert gamma prototype `d6756bb5` stopped at the local CUDA 12.8 compiler gate. Across 15 builds and 30 entry instances, enabling it raised MAIN spill stores from 4 to 44 bytes and spill loads from 32 to 68 bytes in both MAIN entries. The other four resource vectors were unchanged. Its combination with the RMW treatment is on hold. These results have no GPU job ID, latency delta or score interval. [Compiler evidence](https://github.com/yelgabo/okc/blob/d79089b/docs/steering/2026-11-01-g-native-gamma-compile.md)

The source inventory also corrects the precision explanation. G0 already retained the BF16 SwiGLU output boundary, so that rounding step cannot account for the fork’s cost as a newly added operation. The 5.77% G0-to-retained score gap remains a comparison across earlier jobs. This review adds no new timing at 128 or 256 tokens and leaves the measured A-versus-B decision intact. [Dated correction](https://github.com/yelgabo/okc/blob/d79089b/docs/steering/2026-11-01-g0-607e-integration-gap.md)

## November 1 review at 13:47 Vancouver time

Historical checkpoint. The 14:49 review supersedes the reset order recorded below.

**The four-cell verify candidate reached H200 dispatch. Its outcome is unknown.** The service accepted public job `1fad70bbe3854cbc` from source `43c6c09` and reported it running. The 980,787-byte single-arm file selects the owned backend for 128- and 256-token verify, in both sliding and full contexts. The remaining cells use G. A local status wait expired after 1,100 seconds. Subsequent requests for the same job and the job list returned HTTP 530 during maintenance. The archive contains no terminal receipt, timings, correctness verdict, confidence interval or runtime resource counts. The private four-cell comparison is a separate held job plan. [Dispatch](https://github.com/yelgabo/okc/blob/e1adc0e/muse_decode_layer/profiles/g1-owned-verify.1fad70bbe3854cbc.dispatch.json) · [Last status response](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/evidence/2026-11-01-reset-operational-root/maintenance-final-status.json) · [Handoff](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/2026-11-01-maintenance-and-progress-handoff.md)

**Family A has a new qualification gap.** Historical standalone MoE job `abe7651dbaa44458` scored 2.215712×. Its selected skinny and main47 kernels use software grid joins under ordinary launches, with a watchdog that can return before readiness. The separate disabled cooperative-launch repair passed five host tests. On local CUDA 12.8, main47 spill stores/loads rose from 12/112 bytes to 16/128 bytes. Skinny stayed at zero. Both termination variants failed the no-new-spill rule before GPU qualification. The measured A-versus-B comparisons at 128/256 still stand, while reuse of the historical A file requires separate progress qualification. The C2 projection audit found CTA-local waits or checked cluster launches on its selected custom paths. [Standalone audit](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/2026-11-01-historical-moe-progress-gap.md) · [Compiler comparison](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/2026-11-01-standalone-progress-off-prototype.md) · [C2 audit](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/2026-11-01-c2-projection-progress.md)

**The G0 integration gap remains open.** Earlier independent jobs `3a5f879df0d44825` (upstream G0, 2.248791×) and `607eb52bdaf24160` (retained G, 2.126083×) differ by 5.77% in score. The largest relative latency differences occur at one-token decode and eight-token verify. Different routing, inputs, baseline times and bundled repairs limit attribution. The next distinguishing evidence is a matched short-path comparison. [Receipt reconstruction and repair inventory](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/2026-11-01-g0-607e-integration-gap.md)

| Tokens | G0 weighted latency | Retained G weighted latency | Retained minus G0 |
|---|---:|---:|---:|
| 128 | 258.671 µs | 269.347 µs | +10.676 µs (+4.13%) |
| 256 | 350.797 µs | 359.297 µs | +8.500 µs (+2.42%) |

Each row uses sliding and full adjusted medians with weights 46 and 16, one official run per file. The score CIs are 2.215059–2.282522 for G0 and 2.094191–2.157974 for retained G. No matched-effect CI is available. [Underlying records](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/evidence/2026-11-01-g-integration-gap/cross-job-gap.json)

The coordinator’s reset plan requires recovery of the accepted job’s terminal receipt and a live quota check before another dispatch. The registered six-load G stage comparison and the fixed-routing short-path trace retain priority. [Reset conditions](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/2026-11-01-maintenance-and-progress-handoff.md)

## November 1 review at 12:45 Vancouver time

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

## November 1 review at 11:19 Vancouver time

Historical checkpoint. The 12:45 review records the stage-local result and the repaired hybrid’s source fit.

Snapshot `fa8173a` covers 53 remote commits since `af3b550`. The private VP2 comparator
saved 0.616 µs at eight-token sliding attention. Its six-load factors fell below the
registered 1% threshold. The owned global sparse-XOR projection ran slower on all ten
target cells than archived C2. The stage-local version has a dispatch receipt and awaits
its own H200 result. Family G retains the 2.1261× layer entry.

**E. The private VP2 comparator saved 0.616 µs (5.71%) at sliding verify batch 1, eight tokens. It missed the six-load acceptance gate.** Job `c79ddddb840a4fb7` used source `e86b0d5`, frozen at `23fb760`, with the external VP2 bundle from `8e30e1b`. Four independently prepared arms shared the same BF16 output boundary. Three held-out inputs per cell and 32 rotated cold rounds produced 36 timing rows. All 640 private math and cache checks passed. The private judged path passed 16 official tests and scored 2.594320× (95% CI 2.555405–2.633234). The interval applies to the official standalone score. [Official receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.receipt.json)

| Held-out seed | Owned mode 8/15 µs | VP2 µs | Delta µs | Delta % |
|---|---:|---:|---:|---:|
| 824731 | 10.856 | 10.136 | -0.720 | -6.63% |
| 824737 | 10.744 | 10.144 | -0.600 | -5.58% |
| 824743 | 10.752 | 10.224 | -0.528 | -4.91% |

The four contemporaneously calibrated six-load comparisons ranged from 1.006309× to 1.009060×. The duplicate log-null was 0.002371. Every seed failed the combined gate, which required all four comparisons to clear both 1% and twice the duplicate noise. VP2 combines 128-key tiles, dedicated ownership and an asynchronous mailbox. The evidence isolates the bundle only. The earlier owned 64-key complete-head cluster loss stands. This round closed the automatic rewrite and promotion route. A new owned design would need independent evidence. [Terminal decision](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-attention-vp2-bf16-terminal.md)

At 128 and 256 tokens all four arms ran owned code. The following raw averages across the three seeds and two placements per arm record coverage. Differences measure variation among those unchanged-code placements.

| Tokens | Context | Control µs | Comparator arm µs | Difference µs (%) |
|---|---|---:|---:|---:|
| 128 | sliding | 15.771 | 15.717 | -0.053 (-0.34%) |
| 128 | full | 109.637 | 109.413 | -0.224 (-0.20%) |
| 256 | sliding | 19.869 | 19.888 | +0.019 (+0.09%) |
| 256 | full | 190.675 | 190.971 | +0.296 (+0.16%) |

[Paired timing rows](https://github.com/yelgabo/okc/blob/fa8173a/muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.analysis.json)

**MoE.** This interval adds a CPU audit of low-token weight representations. It parks full BF16 expansion, scale expansion and generic activation caching pending evidence that decoding or activation handling limits the skinny kernel. The audit provides no H200 candidate timing, so A through D and the 128/256 A-vs-B decision keep their prior measured status. [MoE audit](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-owned-lowtoken-representation.md)

### Projection: global sparse XOR lost on all ten target cells. Stage-local XOR awaits its result.

Job `2b537518091248f1` tested the owned global-record E7-XOR repair before activation loading (source `c138969`, registration `1bd55da`, SHA256 `685ffab5d1614343d3c46d505967ae6817e3784736297018c78984354333f765`). This H200 public run passed nine official tests with 30 timed cells and 30 profiles. All ten overrides executed with zero fallbacks. Startup passed 160 math checks, ten captures and 30 fresh-input checks in 123.33 seconds. All 40 resource vectors passed the local-memory, layout and CTA-admission gates. The candidate scored 1.394120071× (95% CI 1.373208270–1.415031872) against archived C2 job `9eba96894ee1402e` at 1.527806483× (1.504889386–1.550723580), an 8.75% lower point score. Each receipt contains one scored submission. Different inputs, machines, driver versions and calibration preclude a paired estimate or a mechanism-specific confidence interval. [Owning receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_bf16_proj/profiles/sk-xor-release.2b537518091248f1.receipt.json), [terminal analysis](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-sk-xor-release-terminal.md).

| Projection / tokens | Adjusted change vs C2 | Raw event change vs C2 |
|---|---:|---:|
| qkv_gate / 8 | +3.562 µs (+36.46%) | +4.176 µs (+30.53%) |
| unpooler / 8 | +0.234 µs (+5.16%) | +0.624 µs (+7.37%) |
| qkv_gate / 16 | +4.246 µs (+43.59%) | +4.128 µs (+29.66%) |
| o / 16 | +1.544 µs (+28.77%) | +1.536 µs (+16.16%) |
| pooler / 16 | +3.876 µs (+28.08%) | +3.648 µs (+20.00%) |
| unpooler / 16 | +1.893 µs (+13.96%) | +1.456 µs (+8.15%) |
| qkv_gate / 32 | +5.533 µs (+52.41%) | +5.184 µs (+35.14%) |
| o / 32 | +1.457 µs (+24.83%) | +1.488 µs (+15.00%) |
| pooler / 32 | +4.853 µs (+32.17%) | +5.008 µs (+26.37%) |
| unpooler / 32 | +1.211 µs (+8.23%) | +1.488 µs (+8.08%) |


At 128 and 256 tokens all five operators used existing C2 paths. The table gives across-receipt changes in the sum of five medians as coverage context. Attribution is limited to the ten target rows above. [All 30 cell medians, timing offsets and seeds](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/evidence/2026-11-01-sk-xor-release-terminal/rows30.csv).

| Tokens | Adjusted C2 → candidate | Adjusted change | Raw C2 → candidate | Raw change |
|---:|---:|---:|---:|---:|
| 128 | 67.096 → 68.591 µs | +1.496 µs (+2.23%) | 87.168 → 89.568 µs | +2.400 µs (+2.75%) |
| 256 | 82.416 → 84.070 µs | +1.654 µs (+2.01%) | 103.408 → 103.920 µs | +0.512 µs (+0.50%) |

The recorded decision closes the global-record schedule and its layer port. The review corrects an earlier dependency claim. Global record loading can delay stage-credit publication and back up the producer, even with valid lifetime ordering. The public timing leaves the cause of the slowdown unresolved. C2 remains the projection entry. [Corrected explanation and decision](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-sk-xor-release-terminal.md).

Public job `dfcfdd01d6f04525` tests a distinct stage-local representation (source and preregistration `a08f501`, 450,781 bytes, SHA256 `de1111b98072f5251ff9b2edd5d453cc22eb8fe1710a943c60922909cc796229`). It derives capacity from supplied weights. It retains the same ten targets and 100 nearby keys with C2 as comparison, subject to unchanged C2 resource admission, 160 math checks, ten captures and 30 fresh replays before enabling overrides. Local CPU and source checks passed. H200 correctness, coverage, timings and scored outcome are pending at `fa8173a`. The 128/256 cells use the original C2 paths. [Dispatch receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_bf16_proj/profiles/stage-xor-release.2026-11-01/dispatch.json), [preregistration](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-projection-stage-xor-release.md).

Two offline projection audits recorded no new GPU timing. The local compilation found terminal waits needed for partials and ring reuse on the selected C2 WGMMA paths. The mailbox audit found that the selected SK/RD reducers are CTA-local and RT already uses async completion. Both score sensitivity scenarios are hypothetical. No layer or A–E performance attribution follows. [Lifetime audit](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-projection-wgmma-lifetime-audit.md), [mailbox audit](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-projection-mailbox-scope.md).

### Family G precision and layer packaging

The latest retained layer receipt is still `607eb52bdaf24160`, Family G stage retirement, 2.126082676× (95% CI 2.094191436–2.157973917). The board snapshot at 17:27 UTC on November 1 records layer rank 316. This review interval adds no layer H200 timing, paired result or submission. [Board snapshot](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/evidence/2026-11-01-projection-sk-xor-root/live-muse_decode_layer.json), [retained receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_decode_layer/profiles/scored.607eb52bdaf24160.json).

The source audit at `bf75af6` narrows the earlier precision claim. The rules preserve BF16 activation boundaries where SGLang uses BF16, including the expert GEMM outputs before the next operator. The pinned JIT short router receives the unrounded FP32 normalized vector while the pooler receives BF16. The audit found no categorical rule requiring every internal split partial to be FP32. Keeping rs128 BF16 partial reductions, YBF and opaque Lt split-K paths disabled is a conservative implementation choice. MFIN can leave its poll before readiness, an independent reason to keep that path disabled. The audit measured no per-repair speedup and did not qualify a new release. [Precision contract and source identities](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-g-fork-precision-contract.md).

At 128 and 256 tokens, the private verify comparison now covers sliding and full contexts in one 968,389-byte package. A corrected owned-128 selector restores the recorded projection and boundary choices. Seven CPU tests and independent source review support that correction. H200 linking, capture, matching natural expert IDs, 96 reference checks and the 295-second local preparation guard remain open. These four cells cover two of six scored loads. The six-load G control-versus-stage comparison and short-path trace retain priority. [Private comparison review and runtime gates](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-verify-four-cell-slice-review.md).

The corresponding **production hybrid fails the size gate** for every nonzero override mask in the tested representation. The default is byte-identical to the 869,076-byte incumbent. One verify32 cell requires 1,072,104 bytes, both verify16 cells require 1,115,832, and all four require 1,125,548, against a strict limit below 1,000,000. The 16-mask matrix reproduced and 19 host tests passed. Local device-only builds preserve the selected WD, boundary and router instructions. This closes the current packaging approach without a runtime or latency result. [Production matrix and stop rule](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-verify-hybrid-packaging.md).

The separate sparse-XOR applicability map identifies 24 potential projection cells on four short loads. G uses SK2 and distinct packed weights. Direct transfer would require a separately qualified consumer and repack. The 128/256 Lt paths are outside this map. Its 830,130-byte figure is a source inventory with allowances, not a generated layer payload. No layer gain follows from the standalone projection experiments. [Layer applicability map](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-g-sk-xor-applicability.md).

## November 1 review at 08:04 Vancouver time

Evidence snapshot `af3b550` contains187 remote commits since `6e078b5`. The local checkout
remains on the documentation branch at `e79359f`. This review parsed135 added remote
profile JSON files and18 historical untracked local receipts. Source links pin remote files.

The repository now registers Family G for the attributed keell0renz layer fork. Its retained
public entry is2.126082676x. The preceding owned Lt-plus-boundary combination has matched
six-load gains. New owned band schedules and QKV experiments fail their performance gates.
Detailed attention and G wait probes fail their timing-distortion gates.

### Layer: owned combination wins; attributed G stage candidate retained

At remote snapshot `af3b550`, the retained public layer is **Family G `g1b4_stage_release.py`**, source `0ce0b23f5342`, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab` (869076 bytes). Job **607eb52bdaf24160** passes 13 official tests, 12 timing/profile cells, 56 verified private replays, no flags; score **2.126082676x**, 95% CI **2.094191436-2.157973917**. The final 06:30 PST report records rank308 and retains this entry after the later experiments. Family G is the repository's attributed-fork track, separate from measurement-only donor F and from isolated A-E mechanisms. [Retained receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/scored.607eb52bdaf24160.json), [final handoff](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-0630-measurement-outcome.md).

The preceding owned Lt-plus-short-boundary combination has a real same-process six-load win, but G's higher public score is an independent-run observation. Do not attribute the whole difference to stage retirement or add it to component gains. The retained G score is 4.04% above the owned `cb71acec29d141c6` point estimate; routing, preparation and baseline measurements differ. [Shape comparisons and attribution limits](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/evidence/2026-11-01-g-public/stage-shapes.md).

#### Complete new layer job ledger

All successful jobs below pass 13 official tests. Diagnostic control scores are checks, not candidate-effect estimates. Intervals are official score CIs, not paired-effect CIs. Raw receipts are linked per row.

| Job / source | Outcome / score and 95% CI | Decision |
|---|---|---|
| [1023d9b0dcaa4c75](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/short_rms.1023d9b0dcaa4c75.json),40b7b01 |2.015410 [1.985179,2.045641]|Independent-prepare short boundary,12 rounds,d1/v1. D1 ratio0.9956 versus null1.0093; v1full missing at280s. No promotion.|
| [a0da77c5f3c647c3](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/matched-short.a0da77c5f3c647c3.json),32f1970+matched helper |Control2.017944 [1.987675,2.048213]|Common prepared state,4graphs/case,2seeds; affected d1/v1 gates pass.|
| [35f2b5d66f51465b](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/matched-six.35f2b5d66f51465b.json),matched six-load helper |Control2.022121 [1.991789,2.052452]|24rows. Seed factors1.003428 and1.001829 against1.001450/1.003343 thresholds; second misses aggregate gate. Boundary alone unpromoted.|
| [59cc1343940f4bcb](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/concat-layout.59cc1343940f4bcb.json),68c769d |Control2.015711 [1.985476,2.045947]|Contiguous K-major concat layout loses every GEMM and layer comparison,2seeds,128/256. Original retained.|
| [3a5f879df0d44825](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/kel.3a5f879df0d44825.json),external8e30e1be |2.248791 [2.215059,2.282522]|Exact external comparator,12 lower independent medians; not an accepted release or paired improvement.|
| [1e3349e377234e89](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/dlplt.1e3349e377234e89.json),1ecf680 |Control2.026022 [1.995631,2.058819]|Three Lt pins,128/256 only; pooled1.0089 passes,32reference checks. Shared-state duplicate controls.|
| [828974e3bbcc4d9b](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/combo_ab.828974e3bbcc4d9b.json),dd1a4b6 |Control2.021122 [1.990806,2.051439]|Lt+boundary common-policy six-load comparison wins both seeds,96/96 checks,40cold randomized rounds.|
| [47bec92021ef4bbe](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/combo_pure.47bec92021ef4bbe.json),pure851c4695 |2.010248 [1.980094,2.040402]|Exact pure combined candidate passes; all3Lt fullFP64/exact/fresh gates, production manifest failures0.|
| [cb71acec29d141c6](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/scored.cb71acec29d141c6.json),c80ab0e |2.043438 [2.012787,2.074090]|Exact851c4695 pure file scored; owned incumbent, subsequently superseded by G.|
| [caad524754b54680](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/g1a.caad524754b54680.json),0329226 |Setup failure; no score/CI|G1a packaging loses source-substitution anchor before correctness/timing.|
| [606f47f251f34873](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/g1a.606f47f251f34873.json),76a8d09 |2.010975 [1.980811,2.041140]|Corrected G1a owned expert fork passes, retained as control, not public best.|
| [ffe303a12a954304](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/g1b.ffe303a12a954304.json),90cff63 |Setup353.520482s exceeds300s; no score/CI|Native selected and7probes complete, but official tests/timing never run.|
| [736a5cbc977f4ad4](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/g1b2.736a5cbc977f4ad4.json),148ecb5 |2.065559 [2.034575,2.096542]|Native G1b2,119.270751s preparation; held for zero-local metadata/fallback-residency and matched validation.|
| [607eb52bdaf24160](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/scored.607eb52bdaf24160.json),0ce0b23f5342 |2.126083 [2.094191,2.157974]|Native stage-retirement G1b4; retained public best.|
| [74a29e0384324952](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/scored.74a29e0384324952.json),a2de34ea15a4 |2.052870 [2.022077,2.083663]|G1b4 control qualification,56 private replays plus2000queued at32/33/128/256; independent run, not paired scheduling evidence.|
| [27b2ed49fc6344ec](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/scored.27b2ed49fc6344ec.json),3191783d2cd5 |2.102308 [2.070774,2.133843]|Post-stage WGS publication;13official/56private pass, lower score; exact placement closed for public retries.|
| [942ebdec400d4e68](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/scored.942ebdec400d4e68.json),0b63a197aaf2 |2.095900 [2.064462,2.127339]|Short pooler/router overlap;56expert+32split-front replays pass. Lower score, public retries closed.|

#### Owned combination: genuine paired evidence, limited attribution

Lt-only `1e3349e377234e89` compares frozen attention-port585d46e2 with candidate da7f00fb: QKV256 pin L0, unpooler128 L3 and unpooler256 L1. Twelve cold randomized rounds, seeds1101433+, busiest/second distributions. Exact pins pass fullFP64, onehot and3fresh graph checks. 128 pooled1.0049 (cross1.0043-1.0056,null1.0011/1.0002),256 pooled1.0129 (cross1.0118-1.0140,null1.0017/.9995). Four-cross latency deltas:128sliding -1.2..-2.0us,128full -1.1..-2.2,256sliding -4.1..-4.4,256full -4.2..-4.8. Affected policies match; small-token policies differ. No six-load extrapolation from this two-load receipt.

Combined `828974e3bbcc4d9b` uses pure SHA851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a, one original prepare and common policies/extensions/mutable scratch with private arm tensors and sequential synchronized replays. Both independent held-out seeds pass all four cross-ratios and twice-null gates: pooled1.0034154/1.0036468; cross ranges1.0027005-1.0041309/1.0027918-1.0045025; nulls1.0002359/.9988111 and.999511/.9987853. Shared preparation does not measure independent build/tuner variability. These are a combined projection/boundary gain, not isolated A-E credit.

The following representative first candidate minus first control **arm-median differences** are calculated from the combined receipt; they are not paired-round median deltas. Both seeds had40randomized cold rounds and all96reference checks passed. No paired CI.

| Tokens/context | Seed0 control→candidate us; change | Seed1 control→candidate us; change |
|---|---|---|
|128 sliding|262.944→262.272; -0.672(-0.26%)|267.904→266.256; -1.648(-0.62%)|
|128 full|353.552→352.704; -0.848(-0.24%)|380.496→380.672; +0.176(+0.05%)|
|256 sliding|346.960→344.688; -2.272(-0.65%)|343.952→341.936; -2.016(-0.59%)|
|256 full|503.072→502.544; -0.528(-0.10%)|498.944→496.256; -2.688(-0.54%)|

[Combined raw receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/combo_ab.828974e3bbcc4d9b.json), [methods and pins](https://github.com/yelgabo/okc/blob/af3b550/docs/runs-decode-layer.md).

#### Public G outcomes at128/256 and the final retained state

These are **independent scored-run weighted medians**, not paired causal effects. Each load weights sliding/full46/16. The named unchanged workload also moved between runs, reinforcing the attribution limit.

| Load | Owned cb71 weighted us | Retained607e us | G control74a us | Postflush27b us | Short-overlap942 us |
|---|---:|---:|---:|---:|---:|
|128 verify|285.381|269.347|291.310|275.079|270.387|
|256 verify|386.265|359.297|375.677|356.243|356.694|

Retained G versus owned is -16.034us/-5.62% at128 and -26.968us/-6.98% at256, observed across different runs. Postflush versus retained is +5.732us/+2.13% at128 and -3.054us/-0.85% at256. Short-overlap versus retained is +1.040us/+0.39% at128 and -2.603us/-0.72% at256, but it changes only T1/T8; these large-token movements are not evidence of its effect. Its targeted weighted T1 regresses +4.502us/+5.73%, T8 improves -2.442us/-2.10%, and full score falls1.42%. Keep the original publication point and fused short front. Profiles supply durations, not actual overlap intervals, SM utilization or HBM bandwidth. [All12cells](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/evidence/2026-11-01-g-public/stage-shapes.md).

The final round does not replace607e. Attention observer3f44f891d0ef47db and native-G observerf147d06a64504c1e pass official controls but reject every detailed attribution row for timing distortion. Owned band32a5c7c7e3fe344919 and cached-band32903e5b03411c4e2d fail full-score gates; their control scores are not layer candidates. The final report explicitly stops that round with no promotion. [Final decisions and next checks](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-0630-measurement-outcome.md).

Next layer evidence remains the registered matched six-load G control-versus-stage comparison and a short-path trace with fixed natural routing. The old whole-literal G-versus-owned legacy harness fails interface/size gates. A new850769-byte selected verify32/sliding package solves that cell's offline packaging problem and preserves selected native SASS, but H200 linking, preparation, naturally identical expert IDs and24oracle checks are untested. It cannot establish a six-load gain. The owned comparator includes an explicit BF16 attention-output repair; comparisons must preserve actual precision and ordering. No matched G-stage improvement should be inferred from the independent public scores. [Legacy closure](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-layer-legacy-pair-closure.md), [selected-cell package and outstanding gates](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-selected-layer-pair.md).


### November 1 remote review, 6e078b5 through af3b550

A's new owned variants fail their registered gates; B has no new winning staged result. C has new negative ownership/finalizer evidence, with scope split between external WGS and owned attention clusters. D's complete ready-job design remains untested. E has no new accepted owned optimization; the detailed wait probe fails its causal-perturbation gate. The positive component result belongs to external-derived Family G on a repaired SC6 control, not isolated A or C. Later Family G layer integration is retained in G1b4_stage_release; its whole-layer evidence must remain separate from isolated component attribution.

Retained-file qualification matters: new QK2 and attention wait studies use a common BF16 attention-output boundary before RMS/gating. Historical owned native attention and the public layer lack that boundary and require repair before reuse. Their historical official passes/scores remain recorded but do not qualify a repaired file. The earlier pending G1b4 qualification text is superseded by the final06:30 report. Retained G1b4_stage_release public job607eb52bdaf24160 passes13official tests at2.126082676x, CI2.094191436-2.157973917, exactSHAba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab. Stage retirement is already included and gets no future gain credit. [current conformance and release status](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-0630-measurement-outcome.md)

#### Family G positive component evidence, separate from A-E

External comparator `b65c4c878014416c` executes exact keell0renz source8e30e1be994db4310794b8d04e90d18d895deb84, SHA2d931f09…83185c, passes5/5 at2.395703x CI2.359767-2.431638.128/256 medians176.505/211.115us are independent-run observations, not causal gains over owned51. SC6W is active; no adoption follows from this comparator. [exact comparator receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/kel_exact.b65c4c878014416c.json)

`63f5a2a8ed2341d1` at a1031ff compares activation-slot current-stage retirement against repaired SC6 control in four private placements, all six loads, held-outs291031/291032/291033. Official control5/5;72each initial/fresh/all-remote/restored checks plus3negative controls pass. Equal runtime96registers/0local/231168dynamic shared/1CTA per SM. Every seed's four six-load crosses passes,1.018493-1.028699; duplicate lognulls0.001021/0.001281/0.001931. Ratios excluding nontransferable B4FIN32 remain1.016389-1.023732. These are matched component outcomes on this repaired external-derived backend, not the owned main47/51 or a layer score. Table uses duplicate geometric-mean controls/treatments and calculated differences; no paired CI recorded.

| Tokens/seed | Control us | Treatment us | Delta us | Delta % |
|---|---:|---:|---:|---:|
| 128/291031 | 200.784 | 188.976 | -11.808 | -5.88% |
| 128/291032 | 201.912 | 194.448 | -7.464 | -3.70% |
| 128/291033 | 194.328 | 186.000 | -8.328 | -4.29% |
| 256/291031 | 241.952 | 223.672 | -18.280 | -7.56% |
| 256/291032 | 217.871 | 204.440 | -13.432 | -6.16% |
| 256/291033 | 233.903 | 220.072 | -13.831 | -5.91% |
[raw timing and gates](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/stage-retire.63f5a2a8ed2341d1.log) [all18cases and validation](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/stage-retire.63f5a2a8ed2341d1.analysis.json)

Candidate-only `46aafddc195c4c57`, exactSHAfde6b1ee…00891b, passes5/5 at2.348799x CI2.313567-2.384031; medians1/8/16/32/128/256 are25.601/69.097/99.162/123.898/178.264/212.460us. This exact job is private standalone qualification. Later layer integration is separately retained in G1b4_stage_release; do not treat its whole-layer result as this component score. [candidate receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/stage-retire-candidate.46aafddc195c4c57.json)

#### A owned variants and external-backend ablations

| Job/source | Coverage, checks | Measured outcome and decision |
|---|---|---|
|85a113a3f11d47a5 /a776f32, single-warp position map|1/8/16/32 layer/std,seeds441/442, four arms,64eager+64postgraph,official5/5|Modified paired deltas-0.320 to+0.704us; six-load fixed-sk8 contributions0.999496/0.999798 vsnull0.998990, >=1%/2xnull and regression gates fail.128/256 assumedunchanged, not candidate measurements.|
|bfbedf7d48e248a0 /6e078b5+token-locality probe|All6loads,seeds781/782,four graphs,12cases,official5/5|Sorted-row factors0.998626/1.000309 and1.000687/1.000423 fail1% gate; close token sorting.|
|05e51ddeb7014dd9 /7886294, SC6 vs byte|All6loads,seeds901/902,four private states;48each normal/fresh/empty,official5/5|Repaired external backend. Six-load pairs0.999807/1.007119 vsduplicate lognull0.006403 fail1%+2xnull;256/901 regresses2.576us. No SC6 port.|
|835253441a214a1c /574cc1c, owned multiply decoder|All6loads,seeds991/997/1009,18cases;72normal+72fresh+24skew+24empty+72post,official5/5|Four factors1.001248-1.002952 vsnull0.008029 fail1%+2xnull and regression gates. Real skinny registers130->128, not predicted140->128. No selective large-only promotion.|
|a5c7c7e3fe344919 /1bf61d2+fc28a77, owned fixed-band32|18cases,threeheldouts993701/993703/993709,264privatechecks,official5/5|All6target rows lose: M16+18.368/+15.416/+20.872us,M32+31.048/+22.000/+30.304. Six-load0.942199-0.945964; all gates fail.128/256 unchanged control.|
|903e5b03411c4e2d /6460509+e99d848+fc28a77, cached-band32|6arms,18rows,seeds993727/993733/993739,396privatechecks,official5/5|M16control102.555,cached103.808,direct117.456us; M32control137.653,cached139.448,direct160.368. Cached loses1.253us/1.22%,1.795us/1.30% vscontrol; six-load0.994470-0.998904 vsnull0.009646 fails. Recovery vs failed direct variant is not baseline gain.|

[position raw evidence](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/position.85a113a3f11d47a5.log) [token-order evidence](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-expert-token-locality.md) [SC6 rows](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/sc6_compare.05e51ddeb7014dd9.summary.json) [multiply rows](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/multiply_decode.835253441a214a1c.summary.json) [fixed-band rows](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/band32.a5c7c7e3fe344919.independent-decoded.json) [cached-band rows](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/cached_band32.903e5b03411c4e2d.independent-decoded.json)

Explicit128/256 coverage for SC6 and multiply follows. Control/treatment are means of the two displayed placements; deltas are computed from those means, not a separate score estimate. Preserve full-six-load failed gates despite favorable multiply rows.

| Experiment/tokens/seed | Mean control us | Mean treatment us | Delta us (%) |
|---|---:|---:|---:|
| SC6/128/901 | 193.272 | 192.784 | -0.488 (-0.25%) |
| SC6/128/902 | 174.280 | 173.632 | -0.648 (-0.37%) |
| SC6/256/901 | 211.680 | 213.136 | +1.456 (+0.69%) |
| SC6/256/902 | 235.872 | 237.104 | +1.232 (+0.52%) |
| multiply/128/991 | 173.184 | 171.496 | -1.688 (-0.97%) |
| multiply/128/997 | 208.376 | 205.272 | -3.104 (-1.49%) |
| multiply/128/1009 | 170.584 | 168.912 | -1.672 (-0.98%) |
| multiply/256/991 | 252.376 | 249.712 | -2.664 (-1.06%) |
| multiply/256/997 | 261.192 | 258.016 | -3.176 (-1.22%) |
| multiply/256/1009 | 252.296 | 249.688 | -2.608 (-1.03%) |

C-related external WGS frontier job `1c6ebc76b91d4b7f`, SHA76c5b0e7…ff9628b, passes5/5 and6attribution cases. Favorable fixed-owner model factors1.003616/1.004271/1.003445 fail the registered stop threshold. These are model outcomes, not measured owner-completes implementation gains. [frontier scope and stop](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-native-finalizer-frontier.md)

G diagnostic `f147d06a64504c1e` at5c1a1f8, MAIN128/256 seeds76111/76117/76123, six computing arms,216math checks/official5pass. All12detail attribution gates fail: whole-kernel displacement4.91-9.54%, phase10.26-14.90%. Keep descriptive traces only; no causal weights/X/retirement bottleneck, no win. B4FIN32 is unmeasured. [invalidated causal attribution](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-g-stage-observed-readiness.md)

#### C/E owned attention changes and private external comparators

- `4862f8e0dc8a4b5a` external-native WST+VP2 matrix passes16official and4arms21/21, but changed-native-only score0.997060 fails. WST v16full regresses12.10us; VP2 v1sliding saves0.77us. VP2-only~1.086% is a prospective sensitivity, not a passing restricted gate. No adoption. Source/artifact details in linked report.
- `f74cbe27819147c1` at36622ef, owned hierarchical cluster merge,3heldout inputs: v1sliding control11.17us vs32.22/32.03, +21.05/+20.86us, +188.45/+186.75%; all inputs lose20.7-21.5us.16official and4arms19/19 pass;64freshQKVx4graphs and3negative controls pass by fail-closed terminal logic, maxima truncated. Corrected sensitivity0.846483 fails1%/2xnull. This combines different mode3/32CTA specialization with global-finalize-preserving DSM; it does not isolate DSM latency or disprove VP2.
- `eac485ea0d554e50` at eee3b5b, owned complete-head64key/genericDSM,3inputs: v1sliding control11.10 vs14.98/15.01us, +3.88/+3.91us, +34.95/+35.23%.16official and21/21eager each pass, but inherited fresh loop shares data/cache across arms; freshness is qualified screen-only. No retry/promotion. External VP2's128key asynchronous implementation remains distinct.

[WST/VP2 and owned merge evidence](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-attention-alternate-engine-decision.md) [complete-head screen](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-attention-complete-output-clusters.md)

`cd6805e7ebce45ee` tests exact rohanarun attention018b7a10/SHA394f2517…17b4b,16pass,2.639606x CI2.600012-2.679201. Independent official128-token v16sliding/full medians11.105/100.722us and256-token v32sliding/full14.634/184.146us are external cross-run observations. Owned historical891 values11.454/101.555 and15.603/183.564 are not matched controls. Published synthetic banks repeat ratios, not absolute times; use official receipts. Comparator stays private, not an accepted owned optimization. [all12official cells and attribution correction](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/evidence/2026-11-01-component-frontier/attention-shapes.md)

`fe75e566c1ae4d51` at77fd87c tests owned per-request two-chain QK, common BF16 controls,3seeds713301-713303. Official15/16 fails cold-prepare280s, so no official score/CI exists. Later cached preparation retains complete48eager and80fresh/final dual-oracle checks and6timing rows. Sliding control8.864-8.992 vsQK2 8.992-9.024us; full16.768-16.992 vs16.736-16.864. All seed score/context gates fail, cross factors0.998247-1.000335 below1.002+2xnull. Record both host failure and complete negative scoped timing, never call it an official pass.128/256not changed. [QK2 complete retained evidence](https://github.com/yelgabo/okc/blob/af3b550/muse_attention_decode/profiles/pr_qk2.fe75e566c1ae4d51.analysis.json)

`3f44f891d0ef47db`, source9dc5788/artifacta882bd2a…51d9c0,2controls+2detail builds,3heldouts at full16/32 decode/verify, passes16official and scoped4/0/0 correctness perarm. Runtime parity passes mode49R128/0local/58960shared/2CTA, mode51R80/0local/107104shared/2CTA. ALL12causal rows fail: perturbation3.776-23.136us exceeds permitted0.990-1.959us.24traces are descriptive only; do not convert waits to bottleneck, SMidle or realizable speedup. This is new invalidation of this exact detailed probe, not invalidation of prior clean candidate timings. [resource and perturbation evidence](https://github.com/yelgabo/okc/blob/af3b550/muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.decoded.json)

No new accepted owned A-E design or A+B/C/D/E combination emerges. Keep positive G stage-retirement component evidence and the later retained G1b4_stage_release layer result distinct, retain all failed owned variants, and separate external comparator opportunities from measured owned gains. Detailed per-case rounds and nulls not repeated above remain in linked raw/decoded records; no paired CI is invented.


### Projection: two owned QKV128 tests lose; C2 remains best scored

Remote evidence `6e078b5..af3b550` adds four projection jobs. Prefill has no new run-ledger entries and remains deferred. The multiply-decoder job `835253441a214a1c` belongs to standalone MoE, despite its projection reviewer and decoder topic; it must not be counted as a projection experiment. [Projection ledger](https://github.com/yelgabo/okc/blob/af3b550/docs/runs.md), [multiply ownership and result](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-owned-multiply-decode.md).

QKV128 depth-two `4542b2b04cb344ec`, source `e67b7ec`, frozen bench SHA `919a2a38e62f303486ec9650d9e11652c5144f321a95ee29653196492d103a2d`, changes outstanding WGMMA groups and corresponding stage release from i-1 to i-2. Geometry, cache hints, fences, watchdogs and epilogue stay fixed. Two independent builds, four full arms plus C3, two held-out inputs and100 randomized cold rounds per input. FP64 tests at100/127/128/129, exact onehot probes and three fresh captured inputs per arm pass; watchdogs zero. Actual resources are identical:90registers,0local bytes,197760B dynamic shared,1CTA/SM,39active clusters against36 required. The resource gate passes, the performance gate fails.

| Held-out | Production raw us | Depth2 A/B minus production | Depth2 A/B minus own depth1 |
|---|---:|---:|---:|
|0|20.192|+0.576/+0.640 us (+2.85%/+3.17%)|+0.512/+0.640 us|
|1|20.224|+0.640/+0.512 us (+3.16%/+2.53%)|+0.608/+0.544 us|

Depth2 wins only9–19/100 against its own control. Depth1 B-A null medians -0.112/-0.064us, MAD0.304/0.256; frozen gain thresholds0.608/0.512us with at least90/100 wins. No paired CI. Official unchanged-C3 control passes9/9 at1.517509533x (95%CI1.494746890–1.540272176), not a depth2 score. Close this depth change, retaining the earlier donor whole-path advantage as unattributed. [Raw receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_bf16_proj/profiles/qkv_depth.4542b2b04cb344ec.json).

QKV128 owned cyclic-push/register-own/shared-transpose epilogue `4edaa20c98904404`, source `aded9cf`, retains the original depth1 mainloop and160-thread CTA. Full computing/fresh-input gates above pass; zero watchdogs; both builds retain90registers,0local,1CTA/SM and39clusters, while dynamic shared increases197760→222352B. Same two-held-out/100-round protocol.

| Held-out | Production raw us | Push A/B minus production | Push A/B minus own pull |
|---|---:|---:|---:|
|0|20.432|+0.320/+0.288 us (+1.57%/+1.41%)|+0.384/+0.384 us|
|1|20.448|+0.416/+0.272 us (+2.03%/+1.33%)|+0.480/+0.352 us|

Push wins18–26/100 vs production and17–28/100 vs own pull. Null B-A -0.032/-0.096us, MAD0.320/0.272, frozen thresholds0.640/0.544us. No paired CI. Official unchanged C3 passes9/9 at1.504909063x (95%CI1.482335427–1.527482698). No promotion, copy/shuffle/barrier costs are not individually attributed. Neither128 test provides a256-token treatment result. Percentages above divide paired deltas by same-run production medians; they are descriptive component changes. [Raw receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_bf16_proj/profiles/qkv_push.4edaa20c98904404.json).

External projection comparator `3c6a2baf7b4b4bf3` uses exact keell0renz source at `8e30e1be994db4310794b8d04e90d18d895deb84`, raw SHA `922da8e54073b2836610d2832e06cf694dba691362dfc9969fc2e50c108c591b`. Off-board check passes9tests and30timed shapes at1.573015516x (95%CI1.549420283–1.596610748), with no public submission. Separate-run comparisons to C2 are opportunity sizing, not causal deltas. Largest C2-minus-comparator gaps: qkv1282.547us, pooler1282.807, unpooler1281.529 and router2562.091. Three128 cells round each split partial to BF16 before BF16 tensor reduce-add, unlike owned FP32 reduction. The router uses global FP32 atomic reduction after in-kernel clear. These are not newly discovered equivalent Lt pins. The audit declines to treat precision-changing mechanisms as unchanged-precision wins without alleging a blanket rules violation. [Comparator receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_bf16_proj/profiles/kel.3c6a2baf7b4b4bf3.json), [dispatch and weighted audit](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-projection-weighted-dispatch-audit.md).

The four-cell substitution gives calculated optimistic +2.2572% score headroom, not a measured gain or ranking forecast. Router256's entire observed gap is only +0.4293% score; prior C3 F/V/order gains already address part. The audit closes another reset sweep without a distinct precision-preserving mechanism and sufficient score-covered opportunity. Existing GTICKET fixes have no new standalone measurement in this range.

Public C3 repeat `40b9f54be6fe4d4e` is the validated pure source SHA `9fd8e794...` plus one trailing label (submitted SHA `8e45ef09...`), with byte-prefix/AST verification recorded. It passes9/9,30timed cells/profiles, no flags, at1.516021653x (95%CI1.493281329–1.538761978). This is the third C3 score below C2, with overlapping uncertainty and no algorithm change. Retain C2's1.527806483x public best; confirmed C3 router component gains remain distinct. No further C3 repeats are planned without a new candidate or decision. [Public receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_bf16_proj/profiles/scored.40b9f54be6fe4d4e.json).

### Final snapshot and constraints

The November1 06:30PST final report supersedes earlier ranks: overall308 (API gives ranks only), layer308/2.1261x, attention220/2.6043x, MoE387/2.2157x, projection311/1.5278x. Projection rank50/10 cutoffs1.5850/1.6087 imply approximately3.74%/5.30% score increases, using rounded CLI scores. Do not turn these into measured latency gains or an overall numerical score. The objective is now top10–50 overall and each non-prefill board, remains active and unmet; deadline November4 23:59Pacific is recorded from official endpoint. [Final outcome and snapshots](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-0630-measurement-outcome.md).

A new **G** is explicitly registered as the attributed keell0renz whole-layer lineage, separate from F (alygg77). User approval for an audited attributed layer fork supersedes that layer base's earlier measurement-only restriction. This does not authorize importing standalone projection code or convert G whole-layer results into isolated A–E wins. Final retained layer is `g1b4_stage_release.py`, job `607eb52bdaf24160`,2.126082676x (95%CI2.094191436–2.157973917),13official tests. The incumbent includes stage retirement. The matched gain applies to standalone MoE. [Family registration and policy change](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-top50-reset.md), [retained layer receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_decode_layer/profiles/scored.607eb52bdaf24160.json).

The final round's attention/G observers fail timing-distortion gates, and direct/cached owned band32 schedules fail full-score gates. Their official scores belong to diagnostic controls, not candidate gains. Six-arm cached improvement over the losing direct schedule is not a win over original. No public submission happened in that final locality round, although earlier public C3/G jobs in this broader range did occur. At final snapshot free checks remaining: layer0/projection0/MoE0/attention4. Reset16:00PST work is already specified: frozen six-load G comparison, low-distortion short-path trace, then possibly selected verify32/sliding pair. New promotion requires all four crosses≥1.01 and>2×duplicate noise, every held-out seed and frozen per-cell regression gates. This review should report those recorded plans, not dispatch them. Prefill remains deferred; no new prefill evidence appears in this range. [Final measured gates and handoff](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-0630-measurement-outcome.md).


## Review at November 1 01:16 PDT

Historical checkpoint. The08:04MST review records the new G lineage, retained layer entry, precision qualifications and completed experiment outcomes.

Reviewed 2026-11-01 01:16 PDT against fetched `origin/main` at `6e078b5`.
The checkout remains `d6337e1`; 148 remote commits were inspected without merging or
changing kernels. New source links pin the remote evidence, which is absent from the local
checkout. Parsed 93 added remote profile JSON files; 18 local untracked receipts remain
already logged historical evidence.

A's deeper ring and skinny wait policy fail their gates. B's masked grouped library
implementation loses at 128/256. E's earlier HMRG/PRQ layer port has one paired win and an
unresolved repeat; the newer late-tid2 scheduling change is promoted standalone with its
layer port deferred. Projection C3 router gains are component results, separate from family C.

### Layer: attention port wins once; repeat leaves gain unresolved

The attention-only port `layer_attn31s.py` (SHA256 `585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b`, campaign recorded at `aea73a1`) replaces only `_ONE_SRC` in `layer_m1rt51sc1g` with owned HMRG1/PRQ31 code, without the withdrawn workspace pick. Candidate check `1c1fcf1d49924050` passes 13 tests, 2.048009646x, 95% CI 2.017289501-2.078729790. Its raw log retains route-side T1 and front skf-r T1 replay fallbacks; this score alone does not establish a gain. [Check receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31.1c1fcf1d49924050.json).

First same-process four-arm job `c0402e4c92bc4b20` uses two held-out inputs, eight rounds, 16 observations per arm, independent preparations, input seeds 610001+ and order 601. All 24 checks per arm pass. Pooled control/candidate is **1.0042**, four crosses 1.0036-1.0048, nulls 0.9995/1.0007: the registered rule selects candidate faster. The complete prepared candidate differs in M1 selection at 32 tokens (inc o+front, other three arms o), so this is not an isolated attention-instruction measurement. Per-load pooled ratios d1/d16/d32/v1/v16/v32 are 1.0102/1.0022/1.0074/1.0020/1.0026/1.0008. [Raw first comparison](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31_ab.c0402e4c92bc4b20.log).

Independent repeat `9414182118414889`, same source and method with seeds 620001+ and order 701, passes all 24 arm checks but fails the timing decision: **1.0015**, crosses 1.0006-1.0024, larger absolute log-null 0.0011. All four M1 policies are empty after fail-closed gates. The original candidate's 16x1 sliding gate reports relative error 0.14 and four order-1 events. This is a different prepared regime, and does not replicate the first gain. No paired confidence intervals are recorded. [Repeat receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31_ab2.9414182118414889.json), [raw repeat](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31_ab2.9414182118414889.log).

At 128/256 tokens, representative cand-inc paired deltas are below. Percentages divide those deltas by the corresponding inc median, not a difference of marginal medians. Other crosses and nulls remain in the linked logs; these cells do not override the aggregate decisions.

| Comparison | Tokens / context | Inc median us | Paired delta us | Elapsed change |
|---|---|---:|---:|---:|
| c0402 first |128 / sliding|278.9|-0.0|0.00%|
| c0402 first |128 / full|362.8|-1.2|-0.33%|
| c0402 first |256 / sliding|355.0|-0.8|-0.23%|
| c0402 first |256 / full|499.4|+0.3|+0.06%|
| 9414 repeat |128 / sliding|278.5|-1.4|-0.50%|
| 9414 repeat |128 / full|356.5|-1.8|-0.50%|
| 9414 repeat |256 / sliding|361.0|-0.7|-0.19%|
| 9414 repeat |256 / full|538.0|+1.5|+0.28%|

The source was scored after the first paired win. Recorded board job `8f2780d7382047e6` passes 13 tests at **2.022870479x**, 95% CI **1.992527422-2.053213536**. Raw diagnostics retain element-injection sensitivity misses in MoE main-m2-ar2-ub T256 and route side T128, plus the unselected skf-r T1 replay failure. The scored M1 policy is {8:o+front,16:o,32:o}. Frozen old-control repeats `36e57280ff1f4c63` and `43026195f8c84ffb` pass at 2.014780x (1.984558-2.045002) and 2.009960x (1.979811-2.040110); these cross-run scores establish no isolated improvement. [Scored port](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/scored.8f2780d7382047e6.json), [control 1](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/scored.36e57280ff1f4c63.json), [control 2](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/scored.43026195f8c84ffb.json).

Matched-M1 diagnostic `273ff292536e4429` passes official tests but loses its early observation lines, so supplies no gate explanation. Fixed-tail `03bb478f8b0b4629` passes 13 tests; matched off/off, on/off and on/on gates at 16x1 sliding and 1x8 sliding all pass with zero order-1 events. The 0.14 event is neither reproduced nor explained. Its judged M1-forced-off control score, 2.035941422x (2.005402301-2.066480543), is diagnostic and must not replace the board score. Thresholds and CUDA are unchanged. [First diagnostic](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/m1matched.273ff292536e4429.json), [fixed-tail diagnostic](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/m1matched.03bb478f8b0b4629.json), [qualification](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-decode-layer.md).

### Donor comparison recovered; timeline retained with invalid routing metadata

Historical job `0778baa78efa4aad` is recovered evidence, not a new dispatch. Candidate `layer_m1rt51sc1g` a00a0efa (sent b7dd9976), donor 8f17613b at pinned 6cae49a, d1/v1 only, 24 observations per arm. Donor pooled advantage is 1.0768, crosses 1.0693-1.0843, nulls 1.0132/1.0009. Candidate copy 2 rejects M1 while copy 1 enables it, so v1 is qualified; d1 uses no M1 and donor advantage survives both candidate copies. Paired donor-minus-candidate deltas: d1s -5.6/-5.9 us, d1f -8.4/-8.0, v1s -6.4/-6.0, v1f -12.3/-11.7. Official candidate check passes 13 tests, 2.013476x (1.983274-2.043678). No 128/256 coverage. This fulfills the previous d1/v1 measurement request; donor remains measurement-only. Profile sums that include PDL waits cannot identify exposed savings. [Recovered receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/recovered.0778baa78efa4aad.json), [raw comparison](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/recovered.0778baa78efa4aad.log).

All-12-shape timeline `41788b39fb824179`, source 097c38f, collects 24 rows but fails its 11,000-byte output guard at 12,192 bytes; no usable timeline or official correctness result survives. Corrected `01183c2992eb4153`, source d327abd, payload SHA256 `dae91104f666fbfc323145dda56dfad39593e1631c12624f39b056b9bc26c66d`, passes 13 tests and retains all 24 owned/SGLang traces. It uses one input per shape, five cold-event and three isolated-profile replays per arm, with explicit 0.1 us trace coordinates and raw-resolution span/union. Control score 2.017452507x (1.987190720-2.047714295) is not an optimization result. [Failed receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/timeline.41788b39fb824179.json), [retained receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/timeline.01183c2992eb4153.json), [decoded timeline](https://github.com/yelgabo/okc/blob/6e078b5/docs/steering/evidence/2026-11-01-fusion-audit/layer-timeline.md).

| Timeline workload | Owned cold median us | SGLang cold median us | Owned uncovered us |
|---|---:|---:|---:|
|128 / sliding|292.416|572.608|2.947|
|128 / full|415.968|825.568|2.944|
|256 / sliding|343.264|691.680|2.817|
|256 / full|543.936|1205.600|3.263|

These are diagnostic workload observations, not gains over the prior candidate. Across all shapes, uncovered time is 1.056-3.263 us; CUDA activity includes waits and neither union nor uncovered time measures SM utilization. At T1, top-k finishes at 34.8/42.4 us before pooler/norm at 46.6/53.6, so these samples show no exposed router tail. **Actual routing counts are invalid** because they were read from captured output before its first replay. Reference group/local_start, timed graphs and official tests are unaffected; the decoder marks routing_metadata_valid=false. The host helper now replays and synchronizes before inspection and passes five CPU checks, including a negative capture-without-replay case. No new GPU result validates corrected metadata. [Evidence and qualification](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-decode-layer.md), [decoded data](https://github.com/yelgabo/okc/blob/6e078b5/docs/steering/evidence/2026-11-01-fusion-audit/layer-timeline.json).

CUPTI probe `04f0339fa9f24807`, source beedbca, initializes and obtains a 9,184-byte image size, but reading the availability image returns **CUPTI_ERROR_INSUFFICIENT_PRIVILEGES**. Official SGLang baseline check passes at 0.997335405x (0.982375373-1.012295436). This is counter-access failure and baseline timing, not traffic, utilization or kernel performance evidence. [Counter receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/counter.04f0339fa9f24807.json).

Current conclusion: E's port is layer-tested and scored, with one paired win and one unresolved repeat under changed M1 gate outcomes. Keep that mixed status visible. No new isolated A/B/C/D gain follows from these layer jobs, and the full ready-job scheduler remains untested. The next evidence gap is the unexplained gate event and reliable traffic attribution, not another claim from summed overlapping kernel durations.


### Remote evidence review through 6e078b5

Reviewed committed run ledgers and retrieved raw logs/receipts with git show; checkout unchanged. Negative timing deltas mean faster. Paired microbenchmark confidence intervals are absent unless stated. Percentages below divide the displayed paired delta by the displayed control median or mean; they are approximate, not official score changes.

### A: new ring and skinny wait-policy gates close without promotion

`e06255199f6d4de1`, baseline d3eeeda, source SHA256 65439aaa…7622e3a, is a producer-only diagnostic on mode51, with layer/std seeds211/212, 128/256 tokens. Official5/5 pass, xp equals51. Controls162.5/214.5us at128 layer/std and235.2/236.2 at256; instrumentation costs1.7/2.0/1.0/1.3us. Empty-region sums exceed readiness-region sums but overlap across CTAs and include overhead, so they cannot become recoverable latency. The old small-job-minority premise is invalidated: at256 seed211, <=4-n-tile jobs supply78.3% of layer stages and88.9% of std stages;128std is100%. [xprod.e06255199f6d4de1.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/xprod.e06255199f6d4de1.log)

Mode87 tests four x slots plus fewer weight slots against51, job `39804be1a59c429c`, dispatch41bf882, generated SHA2565e8eaa8f…156de31. Four seeds221-224 per routing,20 cold replays/arm. Official5/5 pass, outputs equal51 at128/256, maximum32-token relative difference3.5e-5. Required savings >=1us at128 and1.5us at256 fail. This closes the fixed capacity tradeoff, not every deeper/per-job ring. Loss of weight lead is an unisolated hypothesis.

| Tokens/routing | 51 control us | 87 paired delta us | Delta % | Faster seeds/4 |
|---|---:|---:|---:|---:|
| 32 layer | 94.1 | +0.1 | +0.11% | 1 |
| 32 std | 136.3 | -0.4 | -0.29% | 2 |
| 32 skewed | 68.8 | -0.5 | -0.73% | 3 |
| 128 layer | 158.3 | +2.2 | +1.39% | 0 |
| 128 std | 212.6 | +1.5 | +0.71% | 1 |
| 128 skewed | 97.9 | +1.2 | +1.23% | 0 |
| 256 layer | 211.0 | +2.9 | +1.37% | 0 |
| 256 std | 252.9 | +3.9 | +1.54% | 0 |
| 256 skewed | 150.3 | +1.9 | +1.26% | 0 |
[ring4.39804be1a59c429c.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/ring4.39804be1a59c429c.log)

The final skinny FULL-slot test_wait check `9be599cf333a47c4` at036b581 compares sk8w3s base, separately compiled neutral null, test, and test duplicate. Seeds321/322 on1/8/16/32 layer/std,20 cold randomized rounds;64 eager and64 postgraph FP32 checks pass, spin0, identical resources130registers/0local/155520dynamic bytes/1block per SM. Both the >=1% contribution beyond2xnull gate and per-case regression gate fail, KEEP=False. Six-load contribution estimates are null1.000295776, test1.000956258, duplicate0.998638858, assuming128/256 ratio1; these are not official policy scores. No128/256 candidate test was performed.

Stamped wait-region reductions are thread0 CTA averages, not critical-path gains. Prior `87f22aa71e614460` failed a host round-name collision; `252df4a54d764ca5` failed stamp decoding before printing its core summary. Neither preserves a performance conclusion and neither is a kernel loss. The final retry changes host handling, not CUDA. [testwait.9be599cf333a47c4.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/testwait.9be599cf333a47c4.log) [testwait.9be599cf333a47c4.receipt.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/testwait.9be599cf333a47c4.receipt.json)

Count-only job `53c07507b9fd4f8c`, dispatch1bfeec9, tests the proposed paired-job multicast opportunity on16/32 layer/std seeds311/312. Official5/5 pass. The >=10% pairable logical-request savings screen fails6/8 cases:16layer0/0%,16std0/0%,32layer18.18/18.18%,32std0/0%. No copy/multicast kernel ran; HBM/L2 traffic and benefit remain unmeasured. Park this registered scope rather than narrow to32layer after failure. [weight_count.53c07507b9fd4f8c.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/weight_count.53c07507b9fd4f8c.log)

### B: structurally different grouped library schedules now tested at128/256

Standalone INDEXED adapter `8d64807ea7504495` passes5/5 official tests but scores0.911513x. Candidate/SGLang us at1/8/16/32/128/256 are91.01/80.00,167.23/142.91,228.86/187.47,253.86/206.05,448.17/459.39,546.62/612.37. These controls are SGLang, not paired51. The adapter inherited a BF16gamma cast from prefill, while standalone51 uses FP32gamma; preserve that comparison limitation. It remains a losing, unpromoted implementation, not evidence against all grouped/TMA designs.

GROUPED_MASKED job `ae8993f4e3e1435c`, dispatch7785d25, corrects FP32gamma and compares four arms, owned51/duplicate and masked/duplicate, independent scratch.20 randomized cold rounds per128/256 layer/std seeds291/292, skew291 correctness. Official5/5 plus40eager and32postgraph FP32 checks pass. Registered savings >=3us at128 and4us at256, beyond2xnull on both routings, fail by large margins.

| Tokens/routing/seed | Owned51 us | Masked paired delta us (%) | Masked duplicate delta us | Owned-null delta us |
|---|---:|---:|---:|---:|
| 128/layer/291 | 172.86 | +180.16 (+104.22%) | +180.22 | -0.48 |
| 128/layer/292 | 172.90 | +180.00 (+104.11%) | +179.87 | -2.05 |
| 128/std/291 | 204.10 | +184.19 (+90.25%) | +183.71 | +0.86 |
| 128/std/292 | 227.30 | +144.64 (+63.64%) | +143.71 | -2.69 |
| 256/layer/291 | 212.26 | +234.88 (+110.66%) | +235.23 | +0.10 |
| 256/layer/292 | 214.11 | +231.78 (+108.25%) | +232.00 | -0.19 |
| 256/std/291 | 225.86 | +228.70 (+101.26%) | +228.83 | -0.48 |
| 256/std/292 | 217.15 | +202.75 (+93.37%) | +202.24 | -0.86 |
An untimed256/std292 profile reports G1/G2 244.447/136.064us of421.599us total,90.25%; GEMMs alone exceed the same-case owned217.15us. Glue fusion alone cannot close this measured loss. Exact config/CAP/count values fell outside the receipt tail and must remain unknown. No confirmation, fresh promotion gate, local-tile retune or layer integration followed. A remains the working baseline; B's default masked schedule is a tested loss, not an untested family. [humming_masked_moe.ae8993f4e3e1435c.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/humming_masked_moe.ae8993f4e3e1435c.log)

### E: late-tid2 promoted locally after separate correctness gates; no layer result

PRQ address-carry job `c77ce71c81af4fb1`, sourcefdaaa11, all12shapes/two inputs/80rounds, official16/16 and all arms19/19 pass. QK/V/both ratios1.0020/1.0021/1.0004 versus null1.0008 receive substantial credit from unchanged instances. Changed-row causal sensitivity estimates0.999968/1.000749/1.000438 do not establish a useful general gain. Source reverted; possible rider only. At v16sliding control15.58us, QK/V/both +0.03/+0.06/+0.00, null+0.10; v32sliding20.10, +0.06/+0.03/-0.19, null+0.03. These are128/256-token changed rows. v1full19.20 gives-0.22/-0.22/-0.38, null+0.03; v1sliding11.17 gives+0.06/0/+0.13, null-0.03. Full v16/v32 are unchanged and cannot supply this mechanism's gain. [prqcarry.c77ce71c81af4fb1.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/prqcarry.c77ce71c81af4fb1.log)

Matched prologue diagnostic `191e57a0f45a4717` at1f31181,3inputs/100rounds, passes16/16 and real arms11/11. D1full first drained page boundary owned7.04/donor6.02us, narrowing after first tile issue to9.79/9.41; complete owned/control-null17.18/16.96 versus donor16.22/16.48. V1full first boundary5.98/6.18, complete19.17/19.14 versus18.59/18.78. Drained exits destroy overlap and are not additive live-phase costs. This motivated late thread-index scheduling, not a measured pipeline optimization. [prologue.191e57a0f45a4717.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/prologue.191e57a0f45a4717.log)

| Job/source | Repeats | d1full controls and candidates us | Gate/status |
|---|---|---|---|
| b4d75c3185744d0a / fac5188, late-tid1 |3inputs,100rounds|ctrl17.28,ctrl2 17.15,late16.80; -0.48us/-2.78%|-0.51/-0.42/-0.45us perinput, localized signal only;16/16 and19/19 pass|
| ab53721b09854717 / fac5188, broad late-tid1 |3inputs,150rounds|ctrl17.25,ctrl2 17.22,late1/2 16.83; -0.42/-2.43%|All full crosses win, but changed-instance score >2xnull gate FAILS because sliding code also changes and its null shifts;16/16 and19/19 pass|
| a9b416a0c7324ecc /554445d, restricted late-tid2 |3inputs,150rounds|ctrl17.28,ctrl2 17.25,late1 16.90,late2 16.80; -0.38/-2.20%, -0.48/-2.78%|All12full crosses improve; full-only causal sensitivity1.0023928 vs nulls1.0001715/1.0005802 passes2xlognull;16/16 and19/19 pass|

[latetid.b4d75c3185744d0a.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latetid.b4d75c3185744d0a.log) [lateconfirm.ab53721b09854717.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/lateconfirm.ab53721b09854717.log) [latefull.a9b416a0c7324ecc.gate.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latefull.a9b416a0c7324ecc.gate.json)

These scheduling changes affect batch1 full decode, not128/256tokens. Keep broad-value1's failed gate. Restricted-value2 is a separately registered scope, not retroactive rescue. Full-only~0.24% is a calibrated score sensitivity from paired timing, not an official scored delta or isolated producer-consumer pipeline gain.

Fresh gate history must remain visible:

- `3588375a529c4fe8`, candidate554445d plus stress generator5c546fc, failed prepare:17/2000 sliding outputs exceed strict1e-4, maxglobal~4.48e-4. Full was not reached; no score.
- `26c8a3ea28554454` at0d5901e, four independently compiled arms and2000inputs each sliding/full, also failed strict1e-4 after both shapes. Incumbent nulls fail19/2000sliding and10/2000full; candidate nulls16/12; crossfailures11-18/10-14.24FP32samples pass but do not establish every queued output. No score. Both strict failures stay failed.
- NEWprospective TASK-math gate `80415fe736bb4890` atc8e709a passes all4000candidate and4000control outputs against FP32/global/row/cache criteria,6negative controls rejected, both workspace scans clean, canonical references crosschecked.16/16 official pass. The2.623065x score is qualified by validation harness allow_tf32=False; it is not the clean candidate score.
- Pure SHA256cbee36a852fbc62f5491e75f957eeb4298dce2db50947984cdbbcd08aa69b356 check `d331aca81c554d70` passes16/16, checks.ok true/flags empty,2.569431x CI2.530890-2.619575. Exact pure scored `c9bc5fbd85b440e2` passes16/16 at2.515774x CI2.478037-2.553510. Production defaults now late-tid2 after those gates; public best remains older8914751e3b9f468b/2.604290x.

[latestress.3588375a529c4fe8.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latestress.3588375a529c4fe8.log) [latenull.26c8a3ea28554454.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latenull.26c8a3ea28554454.log) [fp32fresh.80415fe736bb4890.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/fp32fresh.80415fe736bb4890.log) [latepure.d331aca81c554d70.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latepure.d331aca81c554d70.json) [scored.c9bc5fbd85b440e2.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/scored.c9bc5fbd85b440e2.json)

No layer port ran. The source estimates0.021980% layer-score sensitivity for0.4us atd1full using actual load weights, below paired-layer noise. This is a forecast from measured component timing, not layer evidence. No new128/256late-tid claim is supported.

### E: mode51 Q8 attribution closes an arithmetic-only direction

`8ac6563343f74bcd` at4f99802,2inputs/120coldrounds, official16/16 control tests pass; two real arms pass targeted correctness. Four wrong-output probes stay probes. All six arms pass the same107104dynamic bytes/CTA and2CTAs/SM capacity gate before timing. Source uses a pinned default0 diagnostic control even though production late-tid2 is now promoted.

| Workload | Control us | Control2 | Read producer, math removed | Zero producer, math removed | No-cache-copy, math retained | Stamped |
|---|---:|---:|---:|---:|---:|---:|
|128tokens,v16full|110.88|110.82|110.69 (-0.19,-0.17%)|74.24 (-36.64,-33.04%)|85.15 (-25.73,-23.20%)|112.70|
|256tokens,v32full|188.38|188.10|197.06 (+8.68,+4.61%)|132.51 (-55.87,-29.66%)|151.17 (-37.21,-19.75%)|190.85|

Consumer-math removal is mixed on128 and slower on both256inputs. Cache traffic/producer pacing merit investigation, but these probes change instruction pacing/compilation and cannot be added or interpreted as realizable savings. Five-repetition stamp medians show full waits13.2/17.5us, producer-empty waits35.1/75.0, partner completion spread12.1/22.7. None is SM idle time. The proposed VT compaction saving was corrected from10240 to4096bytes because query staging aliases it; NS4 requires119392bytes/CTA, exceeding two-CTA capacity. No NS4 build or GPU experiment resulted. Existing mode51 pair-reweight losses remain closed; no new mechanism is justified by aggregate stamps alone. [q8attrib.8ac6563343f74bcd.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/q8attrib.8ac6563343f74bcd.log)

### Context: unchanged controls and prefill evidence

New standalone MoE control checks/submissions7631974774944a2a/336f3a63bd6a44f3 atd3eeeda score2.193073/2.199082x,5/5each; historical47 repeat116c5185b14b487d fromec565a7 scores2.213803x,5/5. None beats existing2.215712. Attention unchanged PRQ31/HMRG1/pick-off repeatscccc131c55874949 and8914751e3b9f468b atd3eeeda+repeatcomments score2.534850/2.604290x,16/16each. The higher latter score is a repeat, not a new architecture gain. [MoE ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md) [attention ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-attention.md)

Prefill Humming results are separate workload context. They must not turn B green for standalone128/256 or for the decode layer. The user later deferred prefill; completed in-flight evidence remains recorded, no ongoing prefill work implied:

The prefill sequence includes initial ABI failure be3cbc7f5a234c64; passing corrected INDEXED0de84000097946c9 and fresh gateece54526780a4f28; local-M component signal7598f81b9b4645f9 followed by fresh failures ec3618d1604e44ba/d26a87ca2f484145; deterministic passes74eb65dad3fa4e54/da9700816b1641ea and paired confirmation21e98162b7114898; masked16384 failurec1f6023f03d34c72, then I64-address correctionf92ffc2953284a9a and fresh passdefbb042449d44e7. These are documented context, not independently raw-audited here. Final masked fresh score1.300971x remains a prefill result.

[full prefill source/hash and evidence ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md)

No new full ready-job D design, owner-completes C result or combined-family layer score is established by this review. Keep experimental failure scope specific: A ring87 tradeoff; A skinny two-full-slot test_wait; B indexed adapter and default masked backend; E broad late-tid1 gate and strict fresh comparator studies. Late-tid2 is a small standalone scheduling descendant promoted after a different, prospectively registered TASK-math gate, not proof that E's complete pipeline is solved.


### Projection: confirmed C3 router gains; later structural experiments lose

Review covers remote `8f2c89b..6e078b5` without changing the checkout. These are projection results, separate from A–E and whole-layer outcomes. [Projection run ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs.md).

C3 adds owned `r64c8s8fv` (prefetch F + uniform descriptor V) and `r64c8s8ordfv` (memset-zeroed atomic reduction + block order + F/V). It contains no GTICKET, donor code or removed fences. Clean source SHA256 `9fd8e794f7eb989943718348da2bf21ece934be3b4527b349eed34ccbceea4a4`, clean check `61330d87d32c4190`, passes 9/9, score 1.523256530x (95% CI 1.500407682–1.546105378). Initial rk-c source `d3eeeda`; C3 confirmation is recorded by `88860db`. All component FP64/probe/adjacent-size and fresh-input replay gates pass. The following percentages divide paired median deltas by the same-run control median, so they are descriptive component estimates, not aggregate gains. No paired CI is recorded.

| Job / workload | Candidate and control | Paired change | Repeats and nulls |
|---|---|---|---|
| `3170985420f74869`, router128 | FV vs r64c8s8 KLUEYZ, control 7.20 us | -0.42 us (-5.83%); independent FV duplicate -0.46 us (-6.39%) | 100 randomized cold rounds; 88/92 faster; control null +0.03 us |
| Same, router256 | r64c8s8 KLE+GRED+GZERO+RKORD+FV vs r128c8s6 KLUEYZ, control 9.44 us | -0.62 us (-6.57%); order-only -0.46 us (-4.87%) | 91/85 of 100 faster; null -0.03 us |
| `7d2fe593fdfa4720`, independent router128 confirmation | Same FV vs same owned base, control 7.10 us | -0.45/-0.46 us (-6.34%/-6.48%) | Fresh generator/order seeds 5353/4000; 88/87 of 100 faster; null -0.06 us |
| Same, router256 confirmation | Same order+FV vs owned r128c8s6, control 9.31 us | -0.69 us (-7.41%); order-only -0.45 us (-4.83%) | 85/90 of 100 faster; null -0.06 us |

Both benchmark receipts pass 9/9 on their judged paths. F alone at128 is inconsistent (-0.03 us initially, +0.10 us confirmation); FV on production r128c8s6 at256 is near noise (-0.06 then -0.13 us), and the r64c8s8 order path at128 loses +1.18/+1.09 us. Retain this shape specificity. [Initial raw log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/router_rkc.3170985420f74869.log), [confirmation raw log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/router_confirm.7d2fe593fdfa4720.log), [clean receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/cand_c3.61330d87d32c4190.json).

Scored repeats do not establish a new aggregate win: C2 `9eba96894ee1402e`, source `3044ee7` plus repeat comment, passes 9/9 at **1.527806x**, CI 1.504889–1.550724, and remains the recorded best projection entry. Duplicate dispatch `68248e326be14626` is not another measurement. C3 `7a15f46e673b43d5` passes 9/9 at 1.513805x, CI 1.491098–1.536512; exact frozen C3 repeat `2383f5173a934c63` passes at 1.519919441x, CI 1.497120649–1.542718233. Their intervals overlap C2. The paired component gains remain evidence for C3; the lower headlines do not isolate a C3 regression. [C2 receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/scored.9eba96894ee1402e.json), [C3 first score](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/scored.7a15f46e673b43d5.json), [C3 repeat](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/scored.2383f5173a934c63.json).

The previously pending cgemv/rd codec probes are complete. `075dc89d74d144d6`, source `d3eeeda`, passes 9/9 on unchanged control. Pair7 timing model loses qkv1/o1/pooler1/slice1 by +2.50/+2.02/+6.34/+1.34 us and rd pooler8/o8 by +1.23/+0.42 us. Even bytes-only perturbations lose, so this model cannot establish an ideal traffic ceiling. Real codecs were subsequently built and tested, superseding any “unbuilt codec” label. [Model log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/codec_cx.075dc89d74d144d6.log).

| Actual experiment / job | Correctness, control and measured result | Disposition |
|---|---|---|
| Scalar actual pair7 `962d2f9c1d9644f7`; builder introduced after `ed506b0`, C++20 fix `1755795` | Exact stored-byte/escape and exhaustive BF16 roundtrips plus FP64 random/probe pass; 100 randomized cold rounds vs original cgemv. qkv/o/pooler/slice m1 +9.952/+5.344/+24.944/+4.384 us, duplicate null -0.064 to 0. Physical bytes drop only 1–2.9%. | Loses; benchmark-only. Earlier `0ab8f0510515473b` failed compile (C++17), no GPU inference. |
| Vector-scaffold pair7 `4aef50bcd265448b`, builder `9f8e7fb` | Roundtrip/LUT/FP64 checks pass, same 100-round protocol. qkv/o/pool512/pool256/slice +4.960/+3.392/+13.024/+7.568/+1.312 us; null 0 to -0.064. No local-memory allocation. | Loses; no production adoption. |
| TOP7 LUT32 `ee39ec933a8542d8`, experiment `acc2f5a` | Stored-bit/LUT/FP64 gates pass. Same paired control/null protocol. qkv/o/pool512/pool256/slice +4.480/+2.160/+11.344/+8.528/+1.632 us, null 0–0.032. Actual byte saving about 3–3.4%; no spills. | Codec series closed as built, broader codec ideas remain untested. |
| Integrated escape consumer `bac995b3b94541c4`, source `0680370` | Original 12-bit codec; consumer warp0 takes dedicated escape warp's work. Two held-out inputs, 100 cold rounds each, independent old/inline builds. FP64 checks pass; actual CTA residency rises without spills. A/B losses qkv +0.960/+0.960, o +1.504/+1.504, pool +2.272/+2.240, slice +0.448/+0.480 us. Rebuilt-old null -0.128 to +0.032. | Ownership variant closed. The control headline 1.550070x is not the candidate's score. |
| rd16 bounded async ring `b103af15ba564ef6`, exact worker sources `c291a9d`/`dbfb451`, receipt owner `08d8150` | M1/7/15/16 random/probe/fresh-graph gates pass. Removes 24B local allocation but stays 1 CTA/SM; adds 98,304B dynamic shared. Ring loses +0.480/+0.528 us on input0 and +0.528/+0.608 on input1; only 12–17/100 faster. Rebuilt-old gains are noisy (null -0.192 us, MAD .272–.304). | Closed; no retrospective promotion, no128/256 gain. |

All successful projection jobs in this table pass 9 official tests on their unchanged control/C3 path, separately from the named experimental-arm gates. Their official aggregate scores must not be assigned to losing arms. The ring memory probes have different residency, so costs are non-additive. [Scalar log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/pair7_actual20.962d2f9c1d9644f7.log), [vector log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/pair7_vector.4aef50bcd265448b.log), [TOP7 log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/pair_top7.ee39ec933a8542d8.log), [escape receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/cgemv_ie.bac995b3b94541c4.json), [ring receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/rd16_ring.b103af15ba564ef6.json).

Dense qkv256 series uses owned raw128-channel x128-token tiles, four-way K split, three ring stages, 144 CTAs. The residency prerequisite was met: 62 active clusters, two CTAs/SM, 90 registers, zero local bytes, 99,376 dynamic shared bytes. Four stages only supported30clusters and were skipped. Full arms pass FP64/probe at129/200/256 and fresh replay; 128 is not a measured timing workload. These experiments retain C3 judged paths and pass9/9. [Dense result ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs.md).

| Job / evidence revision | Same-run measured result | Conclusion |
|---|---|---|
| `4ca25ee4eccc495c`, `0a75361` | Lt raw26.624 us; full A/B39.472/39.408; paired losses +12.656/+12.816 us (+47.54%/+48.14% of control median),0/100 faster | Actual residency and spill gates pass, performance fails. |
| `e6e5bd6d0a104faf`, `9e10edc` | Lt26.912; full39.536/39.648; no-reduction36.384/36.336; memory-only no-reduction32.224/32.240 us | Ablations are incorrect-output diagnostic probes, not candidate gains. Even removing math and reduction leaves about5.3 us. |
| `d271218896df4a85`, tokenfast builder fixed in `9f8e7fb` | Lt26.592; old full39.328/39.440, tokenfast39.504/39.376; memory probe saves only0.1–0.3 us | Work-order repair fails. Prior `303ba7fc3bb44b9d` failed host harness before CUDA build, not a kernel result. |
| `7803be789c624f8e`, `46b1e45` | One128-row weight TMA replaces two64-row transfers with same bytes/barriers. Lt26.160; old full38.832/38.816, new38.320/38.336 us | About0.5 us saved versus losing experimental control still leaves about12 us to Lt. Local dense series closed. |

No bandwidth counters were captured. H200's4.8TB/s specification corrects an older3.35TB/s H100 premise but establishes no recoverable measured gain. GTICKET rk-d has no new result in this range; C3 deliberately succeeds without it. [Dense raw log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_dsm.4ca25ee4eccc495c.log), [ablation](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_ablate.e6e5bd6d0a104faf.log), [tokenfast](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_tokenfast_fixed.d271218896df4a85.log), [wide TMA](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_widew.7803be789c624f8e.log).

### Prefill context: measured improvements, now explicitly deferred

Prefill is a different problem with753/3012/16384-token workloads, not128/256 evidence for architecture A–E. User steering now defers all further prefill work. Preserve completed results and failures without listing them as active follow-ups. [Prefill ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-prefill.md), [detailed Humming validation ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md).

- Baseline `09ba15bd18fd4000`, source `e0ce491`, passes5/5 at1.000953x. Scored baseline `181880202c5742db` passes at0.999115x but did not establish a ranked entry in the later board snapshot. Fixed Marlin M16 `e6c12149c99a4a96` and M32 `334149df33f64c54` pass5/5 but lose at0.715181x/0.884745x. These are cross-run whole-prefill results, no same-process fixed-tile gain.
- Capability check `e30ec1bf8e8142df` passes5/5, unchanged baseline0.999768x; installed Humming0.1.12. First indexed attempt `be3cbc7f5a234c64` fails prepare on scale ABI [32,1] vs [32], before GEMM. Corrected `0de84000097946c9`, SHA `bf6d95ae...`, passes5/5 at1.176448889x (CI1.158802–1.194096). Fresh gate `ece54526780a4f28` passes2000 queued/64 eager M32 checks, four synthetic routing banks. Pure-v3 SHA `f3024d1b...` scored `92408f1b65c643c4`:5/5,1.146465578x (CI1.129268595–1.163662562).
- Component `7598f81b9b4645f9`, six arms/two seeds/12cold rounds, passes all arm math checks and5official tests. Estimated-local-M vs routed-M heuristic saves109/162 us at753 and228/111 us at3012, but16384 is unresolved against duplicate noise. Global heuristic `ec3618d1604e44ba` fails unchanged fresh envelope at32 before official tests. Bounded heuristic `d26a87ca2f484145` also fails at unchanged32 control, before changed753/3012; this is sparse replay nondeterminism, not evidence those larger-shape timings lost.
- Disable-StreamK `74eb65dad3fa4e54`:5/5,1.092871039x (CI1.076477973–1.109264104); all ten fresh statistics exactlyzero at32/753/3012,64eager and2000queued each. Deterministic bounded heuristic `da9700816b1641ea`, SHA `29770518...`, passes same gates at1.161679904x. Paired component confirmation `21e98162b7114898`, two seeds/12cold rounds: bounded minus routed -197.9/-191.6 us at753, -399.8/-260.7 at3012; duplicate nulls -0.90/+0.74 and -0.61/-0.83. Component compensation is not an aggregate score estimate.
- Nozero plus filtered SiLU `2bd4f06042864771`, pure SHA `561e4180...`:5official and24arm math checks pass including NaN poison. Four arms/two seeds/12cold rounds, nozero-minus-control -75.808/-80.512 us at753, -314.400/-309.920 at3012, -1705.791/-1752.192 at16384; duplicate nulls +0.672/+0.064,-1.728/-2.208,-3.200/+4.000. Both clearing removal and filtered SiLU change, so credit neither alone. Judged candidate1.262738192x is not the benchmark file's authorization for submission.
- Nozero+noStreamK `36d2347642e14e37`, SHA `a0fe61e7...`:5official and24arm checks pass, fresh stats zero at32/753/3012;1.197430766x. Pure scored `1530ec79d50a4f69`:1.216560352x (CI1.198311946–1.234808757), selected by recorded23:26board. Combined bounded/nozero/noStreamK `e9bd4269a6584df8`, SHA `abce3925...`, completes5official/fresh checks at1.273593087x (CI1.254489190–1.292696983), archived unsubmitted.
- GROUPED_MASKED `c1f6023f03d34c72` passes initial5tests then fails16384 benchmark recheck (residual relativeL2 .6951 vs .0422345 bound), no score. Four widened-before-stride int64 casts correct own Triton overflow: `f92ffc2953284a9a`, SHA `63eb4171...`, passes5/5 plus all8 benchmark rechecks at1.294149997x (CI1.274737747–1.313562247). Poisoned fresh `defbb042449d44e7` subsequently passes5/5 and32/753/3012 unchanged envelope at1.300970887x (CI1.281456324–1.320485450), completed after scope deferral; no promotion. This is envelope-valid, not deterministic arithmetic.

Prefill component timings have no paired CI recorded; do not compute percentage improvements without corresponding exact same-run control medians. Do not import successful prefill Humming results into standalone MoE: interfaces, shapes and precision boundaries differ, and the standalone adapter tests are separate evidence.


## Review at 2026-10-30 23:32 PDT: PBULK measured; mode 86 and one-token fusion miss their gates

Historical checkpoint. The November 1 review supersedes pending layer-port, router, codec and donor-comparison labels below.

Evidence through `8f2c89b`, reviewed 2026-10-30 23:32 PDT. PBULK now passes GPU checks;
variant 1 has a small gain below its promotion bar, while variant 2 loses. Mode 86 fails
its performance rule. M-C C-1 loses its local whole-step test before the bench faults.
The recorded layer entry is now 2.018705x, within prior uncertainty, with a changed
route-side validation rule and report-only sensitivity findings. No isolated family gain
or new A-versus-B decision follows from that scored result.

### Layer: one-token fusion loses its local test; gate update passes and is scored

M-C C-1 is now tested. `layer_mc1s`, SHA256 `2998259e7109cfaadfd21c33db79266ba73297e0dbbaf577088aff02f7d1c38f`, built at `b5ebf08`, fuses the one-token o projection and attention boundary into one cooperative kernel with two grid barriers. Job `5ae9ce318e014edf` compares it with `layer_m1rt51sc1` (`9373b976`), using independent prepares and bench-only fixed M1 policy with 250 gate replays. Its completed candidate prepare passes the cooperative probe, self-test and C-1 fresh-input gate before timing. The stable pick records a **+1.16 us loss against a 66.0 us control, +1.76%**, for the weighted one-token step (46 sliding + 16 full over 62), with C-1 faster in **0 of 9 rounds**. No paired CI is recorded. This is one prepare's local whole-step comparison, not a completed four-arm result. The predeclared stop rule closes this implementation; C-2 remains unbuilt. M-C C-1 is a layer fusion name, not architecture family C. [Raw log](../muse_decode_layer/profiles/bench16_mc1.5ae9ce318e014edf.log), [source and method](runs-decode-layer.md), [M-C plan](plan-2026-10-31-m-c-t1-fusion.md).

The bench later exits with an illegal memory access and has no judged score, completed cross ratios or 128/256-token results. The faulting kernel is absent from the tail. This resembles the unresolved two-layer-process fault in `36f93b13d4c14d5b`; shared module or device state is a hypothesis, not an established cause. The new follow-up treats bench-process faults separately from candidate correctness until attribution is found, and stops on a fault in a single-file check. It does not erase either fault or the local C-1 timing loss. Separately, the reduced 250-replay M1 gate failed at 1x8 sliding and switched M1 off in candidate/inc2 prepares. The bench patch now requires at least 500 replays. This validation-sampling failure is distinct from the memory fault and C-1's own gate. [Run record](runs-decode-layer.md), [open fault follow-up and scoped rejection](ruled-out.md).

The prior projection-C1 layer comparison `b70cfa48c0264d20` remains inconclusive at 1.0007, but its fallback explanation changes. The old route-side 1e-4 self-test rejected cand2 at 1.21e-4. That fallback **does affect the timed one-token FP32 front**: cand2 d1 sliding was 84.8 us against cand's 83.1 us, a cross-arm +1.7 us (+2.05%) observation, not an isolated gate-change gain. The skf-r path exemption at other sizes did not exempt d1. [Follow-up and gate decision](runs-decode-layer.md).

The run record documents a user decision to replace that route-side fixed threshold with the existing fresh-input control-envelope method, retaining the order-1 threshold of relative L2 >= 1e-2. This is an explicitly changed validation rule, not proof that the old failure was harmless. Attention and norm checks retain their own bounds. `layer_m1rt51sc1g` at `2840157`, SHA256 `a00a0efa0a761d990d7298a64340751e6a3979369aa8247a1f3d958f975c7480`, changes only that self-test and one summary line from the projection-C1 layer port. Single-file check `2edb120747cb4611` passes **13/13**, no fault, **2.0045898445x**, 95% CI **1.9745209968-2.0346586921**. All six route-side gates pass, including exact zero candidate/control differences at 128/256, route gates total 14/14 and the manifest has no failures. At one token, candidate/control relative error is 6e-5/6e-5, element error 1e-3/1e-3, and 0/200 calls exceed 1e-4 on either side. Its route-side negative controls reject injected errors. [Check receipt](../muse_decode_layer/profiles/layer_m1rt51sc1g.2edb120747cb4611.json), [raw check log](../muse_decode_layer/profiles/layer_m1rt51sc1g.2edb120747cb4611.log), [recorded decision](runs-decode-layer.md#validation-policy-change-fresh-input-replay-gate-user-decision-2026-10-30).

The coordinator then scored the same candidate twice; the second submission adds only a repeat comment. These are cross-run observations, not a paired comparison with the old release:

| Job | Score | 95% CI | Correctness and limitations |
|---|---:|---|---|
| `dc25fef4d65642a9` | 2.0063008666x | 1.9762063536-2.0363953796 | 13/13 judge tests; route gates 14/14. Report-only element-injection sensitivity misses in MoE main-lean-silu-bar and main-lean-grp-finss64 at 256. |
| `7b84ed4346d5497b` | 2.0187050168x | 1.9884244415-2.0489855920 | 13/13 judge tests; route gates 14/14. Report-only element-injection sensitivity misses in route side at 16 and 256. |

Both raw manifests record two sensitivity findings, with no reference, replay, nonfinite or workspace failures. Thus the clean check's negative-control outcome must not be generalized to the scored repeats. Both repeats select M1 o+front at 32, whereas the single-file check selects o. [First scored receipt](../muse_decode_layer/profiles/scored.dc25fef4d65642a9.json), [second scored receipt](../muse_decode_layer/profiles/scored.7b84ed4346d5497b.json).

The recorded board entry moves from `0f58d5d8223f45af` (2.0148147x) to `7b84ed4346d5497b` (2.0187050x), within the recorded uncertainty. This establishes no isolated A, projection C1 or gate-update gain. At 128/256, both scored runs cover sliding/full workloads with 20 timings per shape, but there is **no new same-process candidate-versus-prior-layer comparison**. Existing A-versus-B evidence at those sizes is unchanged. [Scored-run record](runs-decode-layer.md#scored-layer_m1rt51sc1g-two-runs-2026-10-31).

The one-token bottleneck explanation also needs restraint. The donor's earlier front comparison used the serial release, not the current overlapping incumbent. Its 28.3 us summed front versus donor 23.4 us does not establish today's target. The current trace overlaps router/routing with pooler: its main-stream pooler plus pre-expert norm is 18.7 us, but that trace and donor trace use different inputs. The plan now calls for a same-input incumbent-versus-donor d1/v1 comparison with kernel names before choosing another kernel target. No such new comparison is recorded at this snapshot. Persistent-o memory parallelism and barrier overhead remain explanations to test, not measured causes of C-1's loss. [Reframe and proposed measurement](plan-2026-10-31-m-c-t1-fusion.md#reframe-the-one-token-front-per-kernel-2026-10-31-after-three-weak-structural-results).

### A: earlier x-slot release passes correctness but misses its performance gate

Job `a00b49f8372042a9`, source `30d6b96`, tests mode 86 against 51 in the same process on seeds 181-186, layer, standalone and skewed routing. Cold-L2 timing uses 20 replays per arm and six seed means. This is an A ring-release variant, not a new owner-completes C or ready-job D design. Negative latency deltas mean faster. Percent is paired mean delta divided by the displayed control mean; rounding limits precision. No paired confidence interval is recorded.

| Tokens | Routing | Mode 51 us | 86 minus 51 us | Delta % | Seeds faster / 6 |
|---|---|---:|---:|---:|---:|
| 32 | layer | 87.0 | +0.2 | +0.23% | 0 |
| 32 | std | 132.3 | -0.8 | -0.60% | 3 |
| 32 | skewed | 69.1 | -0.1 | -0.14% | 3 |
| 128 | layer | 162.1 | -0.3 | -0.19% | 3 |
| 128 | std | 194.0 | +0.0 | +0.00% | 4 |
| 128 | skewed | 102.2 | -0.1 | -0.10% | 2 |
| 256 | layer | 216.4 | -0.6 | -0.28% | 5 |
| 256 | std | 244.3 | -0.1 | -0.04% | 3 |
| 256 | skewed | 160.8 | -1.4 | -0.87% | 6 |

Outputs equal 51 at 128 and 256 tokens on every seed. Maximum 32-token difference is 3.4e-5, below the 1e-4 bound. FP32 self-test reports 3.88e-3 for both modes and spin flags are zero. The receipt passes at 2.19501x, which is separate from the candidate's adoption gate. Mode 86 fails the required 256-token saving of 1.5 us on both layer and standalone, and the 128-token saving of 1.0 us on both. It is not kept. The 256-token skewed saving is measured, but does not satisfy those gates. [Raw log](../muse_moe_w4a16/profiles/bench_m86.a00b49f8372042a9.log), [receipt](../muse_moe_w4a16/profiles/bench_m86.a00b49f8372042a9.json), [rule and result](runs-moe.md#earlier-x-slot-release-mode-86-2026-10-31).

The new diagnostic text reinterprets existing mode-56 job `7063f86352344efa`, rather than adding a GPU run. Its stamped refill delay is 0.18-0.31 us per stage, weights-copy latency 2.7-3.2 us, x-copy latency 2.2-2.4 us, and release-to-need lead 3.4-4.1 us. Instrumentation itself lengthens this lead. A donor counter build spilled and was abandoned before a check. Readiness waits on `done_as` and `xready`, and x/weights HBM competition, remain unseparated hypotheses. Per-job x-slot sizing is unbuilt. The next step is to separate the x producer's readiness and empty-slot waits before building it. This adds no A-vs-B or whole-layer win. [Diagnostic analysis](runs-moe.md#where-the-consumers-wait-comes-from-refill-delay-or-copy-latency-2026-10-31).

### E: PBULK now passes GPU checks, with a small gain for variant 1 and a loss for variant 2

The fixed producer exit has now run successfully. Retain the earlier hangs in history, but replace the current pending-correctness label. These are mode-49 producer-side page-index pipeline variants, not whole-layer results. Bench p and r use three held-out inputs and 200 rounds; bench s uses two inputs and 120 rounds. All three receipts pass 16/16 judged tests on the board path. Real arms report ABCHECKSUM 11 passes and zero failures in p/r, 19 passes and zero failures in s; diagnostic probes are excluded from those correctness claims. No paired confidence interval is recorded. [p log](../muse_attention_decode/profiles/bench_p.bb06f8e57bd84636.log), [r log](../muse_attention_decode/profiles/bench_r.f3f1e4a5249a4228.log), [s log](../muse_attention_decode/profiles/bench_s.1c861d3944f149a0.log).

The following cold pooled deltas subtract displayed pooled arm times, so they can differ from the log's separately rounded delta by 0.01 us. d16/d32 are q_len 1 with batch 16/32, not 128/256 tokens.

| Job @ source | Candidate | Workload | Control us | Candidate us | Delta us | Delta % |
|---|---|---|---:|---:|---:|---:|
| `bb06f8e57bd84636 @ 0671cc7` | PBULK=1 | d16 full | 100.45 | 100.38 | -0.07 | -0.07% |
| `bb06f8e57bd84636 @ 0671cc7` | PBULK=1 | d32 full | 179.42 | 178.91 | -0.51 | -0.28% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=1 | d16 full | 100.42 | 100.26 | -0.16 | -0.16% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=1 | d32 full | 179.58 | 178.98 | -0.60 | -0.33% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=2 | d16 full | 100.42 | 101.79 | +1.37 | +1.36% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=2 | d32 full | 179.58 | 182.50 | +2.92 | +1.63% |
| `1c861d3944f149a0 @ a191c34` | PBULK=1 | d16 full | 98.05 | 97.95 | -0.10 | -0.10% |
| `1c861d3944f149a0 @ a191c34` | PBULK=1 | d32 full | 173.70 | 172.90 | -0.80 | -0.46% |

In p, matched-null deltas are +0.03/+0.06 us and independent-null +0.06/-0.10 us at d16/d32. Variant 1's own-state twin saves 0.13/0.51 us. Its gated score gain is +0.063% against each null, below 0.10%. d32 wins all three inputs; d16 has mixed signs. In r, matched null is +0.03/+0.03 and independent null +0.00/-0.13 us. Variant 1 repeats a +0.074% score gain, again below the bar, with d32 faster on all three inputs and d16 mixed or tied. This is a measured small component gain, off, with no layer integration test. [Gate evaluations](runs-attention.md#attention-round-5-10-31-ledger).

Variant 2 replaces the producer barrier with per-window free mbarriers. It loses on all three inputs. Its own-state twin is +1.34/+2.98 us, and variant 2 plus HMRGF is +0.22/+0.70 us in the printed raw deltas. Variant 2 and its HMRGF combination fail the score gate, with reported G -0.45%/-0.09%; variant 2 also breaches the d32 1% regression cap. Producer-only timings are 37.57/63.49 us for control and 53.73/95.39 us for variant 2. Variant 1's earlier producer-only probe is also slower, 47.33/82.75 against 37.60/63.42. These probes omit normal work and are diagnostic, not candidate speedups. The interpretation that lockstep or slow-reader waits limit refill remains mechanism analysis. Reopen variant 1 as a rider for a promoted mode-49 change, or with a producer retaining its gain without per-tile lockstep; variant 2 needs a refill that avoids waiting on the slowest previous reader. [Results and scope](runs-attention.md), [ruled-out entries](ruled-out.md).

The s run covers all 12 shapes. PBULK=1's full-run ABSCORE 0.9974 lies within independent-null 0.9968 variation; changes on untouched shapes prevent attributing that overall score to PBULK. Its q_len-8 rows provide explicit 128/256-token coverage but are unchanged by mode-49 PBULK: v16 sliding/full controls 15.68/112.13 us and pb1 15.65/112.99; v32 sliding/full controls 19.90/190.37 and pb1 19.84/190.21. These are context observations, not PBULK wins/losses at 128/256 tokens.

The same s run is an attribution diagnostic, with no adoption gate. Consecutive phase-exit differences put our q_len-8 tile loops at 6.21/4.35/8.13 us against public 5.79/3.93/7.26 for v1 full, v16 sliding and v32 sliding, gaps +0.42/+0.42/+0.87 us. The d1 full prologue is 10.27 versus 9.73, +0.54 us. Descriptor/V-transpose address carries in the PRQ loop are untested, and matching sub-phase exits for the d1 prologue are pending. Do not sum these gaps into a realized gain. No candidate beat the board entry, and no PBULK layer test was recorded. [Raw phase exits](../muse_attention_decode/profiles/bench_s.1c861d3944f149a0.log), [round-5 next steps](runs-attention.md#attention-round-5-10-31-ledger).

Receipts for these three runs are 2.49640x with CI 2.45895-2.53384, 2.51884x with CI 2.48105-2.56956, and 2.58437x with CI 2.54561-2.63520. These cross-run judged board-path scores are not PBULK deltas. [p receipt](../muse_attention_decode/profiles/bench_p.bb06f8e57bd84636.json), [r receipt](../muse_attention_decode/profiles/bench_r.f3f1e4a5249a4228.json), [s receipt](../muse_attention_decode/profiles/bench_s.1c861d3944f149a0.json).

### Projection preparation and protocol review

Projection context, reviewed through `8f2c89b`: no new projection measurement or profile was committed after `9bf7f05`. The two untracked projection receipts (`db5be7d500704692`, `64ed6e6b5e204dc0`) are historical October 29 scored runs already recorded in [runs.md](runs.md), not new evidence. The C1/C2 paired results and C1 board score therefore remain unchanged. The GTICKET ordering review is now recorded as sound, with fixes in `e6510e4`: 64-bit tickets and published generations prevent the 32-bit wrap misalignment; timeout adds nothing and sets diagnostic bit 128; a generation/owner-pointer mismatch adds nothing and sets bit 256; `CLK <= 16` bounds the state arrays; `_self_test` rejects any nonzero diagnostic. The [port guide](proj-port-guide.md#in-kernel-tile-clear-for-the-split-router-wv_gticket-review-pending-2026-10-31) records the independent review and emitted zero-store/release/acquire instructions. Its heading still says “review pending”, but the later review paragraph supersedes that status. This is a completed protocol review and code correction, not a new correctness or performance run of the corrected implementation.

Pending benchmarks: `62bf3e9` prepares rk-c to test tensor-map prefetch `F` (`WV_PFTM`), uniform descriptors `V` (`WV_UDESC`), their combination and duplicate controls on production routers at 128/256, plus memset-zeroed `R+M+O` with/without `F+V`. `d0a274b` adds rk-d to [the router builder](../tools/proj_make_bench_rg.py): corrected GTICKET `H` as a separate arm beside `R+M+O`, with/without `F+V`, at 128 (`r64c16s8`) and 256 (`r64c8s8`), including duplicate controls. The [round 9 preparation record](runs.md#2026-10-31-utc-round-9-prep-no-quota-branch-proj-r9) also prepares [codec factorial probes](../tools/proj_make_bench_cx.py) for cgemv at one token (qkv_gate/o/pooler/slice) and rd at pooler/o eight tokens. These extend timing models, not implemented codecs. Production SASS was recorded as 92/92 instruction-identical to C2 after the preparation changes; rk-d was recorded as compiled without spills. No benchmark job ID, new timing, repeat count or uncertainty exists for these pending variants in this snapshot. The old 256-token diagnostic win and 128-token losses remain scoped to the earlier implementation and jobs.

## Review at 22:35 PDT: owned projection wins; layer attribution resolved

Historical checkpoint. The 23:32 review supersedes PBULK and GTICKET pending-review labels and corrects the timed route-side fallback explanation below.

Evidence through `9bf7f05`, reviewed 2026-10-30 22:35 PDT. C1 and C2 projection
reimplementations win successive paired comparisons. C1's layer port is inconclusive.
The recovered M-B receipt measures the existing M1 + 51 policy because the added slice
was rejected. Both-phase stream-K mode 79 loses; attention PBULK hangs before timing.
These results keep A ahead of tested B at 128/256 and leave the full D scheduler untested.

### Layer: attribution resolved; PDL extensions and projection port remain inconclusive

The missing `c8d1e57261dd4901` receipt and log are now present. They confirm that the
file was `layer_mb1s`, hash `16feec52`, against `release_s`. Both candidate prepares
rejected the added unpooler PDL mode and selected the same M1 policy as `layer_m1rt51s`:
o+front at 8 tokens, o at 16/32. Thus pooled 1.0106 is another reading of that combined
policy, not a gain from the new M-B slice. The earlier 1.0052 reading and this 1.0106
reading differ by more than their within-run nulls. Keep run-to-run uncertainty visible.
The prepare-time acceptance rule requires savings above max(0.3 us, twice MAD)
in at least eight of nine rounds. The extension did not pass; exact per-mode times
were truncated from the log.
[Receipt](../muse_decode_layer/profiles/bench10_mb1s.c8d1e57261dd4901.json),
[raw policy and timings](../muse_decode_layer/profiles/bench10_mb1s.c8d1e57261dd4901.log).

All timed four-arm benches below use two inputs, eight rounds and 16 observations per arm.
There are no paired confidence intervals; cross ratios and nulls provide the recorded
uncertainty. Ratios are weighted whole-layer time ratios, not isolated family gains.

| Candidate / source hash | Job / order seed | Control | Preparation | Pooled ratio / null R, D | Decision |
|---|---|---|---|---|---|
| layer_mb1s / 16feec52 | `c8d1e57261dd4901`, 79, inputs 260001+ | release_s | Independent | 1.0106 / 0.9985,1.0013 | M1+51 policy faster; added slice rejected |
| layer_mb2s / 3aee22fd | `a3760918c13c4e40`, 83, inputs 270001+ | layer_m1rt51s | Shared duplicate state | 1.0015 / 1.0013,1.0053 | Inconclusive |
| layer_mb3s / 76da38c1 | `6e51f2c9dc774a79`, 97, inputs 290001+ | layer_m1rt51s | Shared duplicate state | 1.0002 / 0.9975,1.0006 | Inconclusive |
| layer_m1rt51s78 / 96f69ff0 | `a03039679e304d95`, 101, inputs 280101+ | layer_m1rt51s | Shared duplicate state | 0.9987 / 0.9994,0.9995 | Identical selected policies; apparent loss is not a valid decision |
| layer_m1rt51sc1 / 9373b976 | `b70cfa48c0264d20`, 103, inputs 300001+ | layer_m1rt51s / f1235356 | Independent, both with bench-only fixed M1 policy | 1.0007 / 1.0008,1.0023 | Projection port inconclusive |

The C1 label here is a projection implementation name, not architecture family C.
Shared duplicate state omits module/preparation variation from the null. The 0.13% apparent
loss with identical selected policies demonstrates that these shared-state readings cannot
support small whole-layer gain/loss decisions. The independent C1 comparison fixes M1's
policy and uses 500 gate replays in a bench-only patch so four prepares fit the time limit;
this patch is not part of the candidate. [Run methods and decisions](runs-decode-layer.md),
[M-B plan and stop rules](plan-2026-10-31-m-b-pdl-chain.md).

The 128/256 table uses cand-inc paired deltas against the inc median. Other three crosses
remain in each raw log; these representative values do not override the aggregate decisions.

| Job | Tokens / context | Control median us | Paired delta us | Percent |
|---|---|---:|---:|---:|
| `c8d1e57261dd4901` | 128 / sliding | 245.0 | -5.5 | -2.24% |
| `c8d1e57261dd4901` | 128 / full | 367.9 | -7.2 | -1.96% |
| `c8d1e57261dd4901` | 256 / sliding | 340.0 | -5.3 | -1.56% |
| `c8d1e57261dd4901` | 256 / full | 513.2 | -6.3 | -1.23% |
| `a3760918c13c4e40` | 128 / sliding | 272.3 | -1.2 | -0.44% |
| `a3760918c13c4e40` | 128 / full | 365.3 | +0.9 | +0.25% |
| `a3760918c13c4e40` | 256 / sliding | 347.8 | -0.7 | -0.20% |
| `a3760918c13c4e40` | 256 / full | 502.5 | -0.7 | -0.14% |
| `6e51f2c9dc774a79` | 128 / sliding | 250.4 | -0.1 | -0.04% |
| `6e51f2c9dc774a79` | 128 / full | 335.8 | +0.3 | +0.09% |
| `6e51f2c9dc774a79` | 256 / sliding | 359.4 | +0.1 | +0.03% |
| `6e51f2c9dc774a79` | 256 / full | 500.2 | +0.4 | +0.08% |
| `a03039679e304d95` | 128 / sliding | 266.1 | +1.2 | +0.45% |
| `a03039679e304d95` | 128 / full | 367.9 | +0.3 | +0.08% |
| `a03039679e304d95` | 256 / sliding | 344.2 | +0.5 | +0.15% |
| `a03039679e304d95` | 256 / full | 516.8 | -0.6 | -0.12% |
| `b70cfa48c0264d20` | 128 / sliding | 266.5 | -1.3 | -0.49% |
| `b70cfa48c0264d20` | 128 / full | 359.7 | -0.4 | -0.11% |
| `b70cfa48c0264d20` | 256 / sliding | 347.5 | -0.1 | -0.03% |
| `b70cfa48c0264d20` | 256 / full | 505.2 | -0.4 | -0.08% |

Raw logs: [c8d1e57261dd4901](../muse_decode_layer/profiles/bench10_mb1s.c8d1e57261dd4901.log), [a3760918c13c4e40](../muse_decode_layer/profiles/bench11_mb2s.a3760918c13c4e40.log), [6e51f2c9dc774a79](../muse_decode_layer/profiles/bench13_mb3s.6e51f2c9dc774a79.log), [a03039679e304d95](../muse_decode_layer/profiles/bench14_m78.a03039679e304d95.log), [b70cfa48c0264d20](../muse_decode_layer/profiles/bench15_c1.b70cfa48c0264d20.log).

Correctness and limits: all five successful bench receipts pass 13 judge tests; each reports
24/24 arm checks. mb1 has a replay-class report on unpicked MoE main-d2. mb2 reports no
gate failures. mb3 rejects unpicked sk8w4s-er on its replay rule and reports unpicked
main-m2-ar2-ub sensitivity at 256. The mode78 rerun reports skf-r@16 sensitivity. The
C1 prepare reports a route side-stream reference check of 1.21e-4 against 1e-4 and falls
back; that route is not used by the selected skf-r front. Passing selected-arm checks does
not erase these findings. Judge scores, all control-path checks rather than isolated effects:
mb1 1.9837x, CI 1.9539-2.0134; mb2 1.9681x, CI 1.9386-1.9976; mb3 2.0104x,
CI 1.9802-2.0405; mode78 rerun 1.9950x, CI 1.9651-2.0249; C1 2.0095x,
CI 1.9793-2.0396. [Receipt-linked run record](runs-decode-layer.md).

M-B small-token tuner evidence is more specific than the unresolved whole-layer ratios.
In mb2, blanket sk entry trigger loses 1.00 us at 8 tokens, 6.06 us at 16 and 5.46 us at
32 relative to preferred whole-step controls 124.5/180.1/240.4 us: +0.80%/+3.36%/+2.27%
elapsed time. Adding unpooler PDL and MoE's post-barrier trigger gives 1.39 us savings at
8 versus the preferred mode, -1.12%, nine of nine rounds. In mb3, unpooler-only trigger
saves 0.80 us against 127.6 at 8, -0.63%, nine of nine. At 16 all extensions lose 1.67-2.54 us
against 194.2, +0.86% to+1.31%. At 32 unpooler+attention saves 1.25 us against 202.2,
-0.62%, but o alone saves 1.10 us, so the added chain contributes only 0.15 us. These are
single-preparation nine-round decisions, not replicated whole-layer gains. Slices 1-3 are
parked at their tested scope; slice 4, the one-token chain, remains designed and unbuilt.
[M-B measurements](plan-2026-10-31-m-b-pdl-chain.md#results-of-2026-10-31-five-checks).

Mode 78 at 32 is now layer-tested, not merely a pending port. Initial bench
`36f93b13d4c14d5b` exits after an illegal memory access in the candidate's32-token front
tuner. The responsible MoE variant is absent from the truncated tail. A one-layer diagnostic
`18d48c5676fa4461`, hash a17a51df, passes 13/13 with 78 selected, and the rerun above also
passes but selects 47 in both files. The pre-registered selection condition fails, so 78 is
not shipped. Neither passing rerun proves the original fault fixed. No 128/256 mode78 gain
exists because the candidate restricts 78 to below 64 tokens.
[Failure log](../muse_decode_layer/profiles/bench12_m78.36f93b13d4c14d5b.log),
[diagnostic](../muse_decode_layer/profiles/diag78.18d48c5676fa4461.log),
[rerun](../muse_decode_layer/profiles/bench14_m78.a03039679e304d95.log).

Projection C1's layer test has weighted load ratios v1=1.0034,d16=1.0027,d32=1.0013,
v16=1.0001,v32=1.0005,d1=0.9959. The pooled 1.0007 is unresolved. C2 rd paths and
C1's128-token pooler library pin were not ported because the layer uses different paths.
The next structural M-C one-token fusions are proposed, not built or measured; their C-1/C-2
slice labels are also not architecture family C. Their donor fusion mapping is inferred, and
must not be treated as an established causal attribution.
[M-C plan](plan-2026-10-31-m-c-t1-fusion.md).

### A/C/D-related MoE: both-phase stream-K loses its gate

Job `2a4fb39964aa448c`, source `587da5d`, tests mode 79 against mode 51 and mode 72.
Mode 79 retains 72's stream-K phase A and replaces claimed phase B with static cost-weighted
ranges and a stage-0 designated finisher. This is an alternative to C's ownership and D's
partial dynamic claiming inside A, not staged family B or a full ready-only queue.
The combined structure is now tested. Six seeds 141-146 per routing use cold-L2 graph
replay, median of 20 replays per arm and means of paired seed deltas. No paired CI is
recorded; percentages below divide the paired mean delta by the rounded control mean.

[Method and gate](runs-moe.md),
[raw log](../muse_moe_w4a16/profiles/bench_m79.2a4fb39964aa448c.log).

| Tokens / routing | Mode 51 us | Mode 79 delta us / percent | Seeds faster | Mode 72 control-arm delta us |
|---|---:|---|---:|---:|
| 32 / layer | 88.0 | +16.4 / +18.64% | 0/6 | +13.1 |
| 32 / std | 139.3 | +17.5 / +12.56% | 0/6 | +10.3 |
| 32 / skewed | 69.4 | +6.0 / +8.65% | 0/6 | +9.1 |
| 128 / layer | 166.2 | +19.4 / +11.67% | 0/6 | +11.3 |
| 128 / std | 205.6 | +24.8 / +12.06% | 0/6 | +9.8 |
| 128 / skewed | 101.7 | +27.9 / +27.43% | 0/6 | +20.1 |
| 256 / layer | 215.7 | +21.4 / +9.92% | 0/6 | +6.1 |
| 256 / std | 250.3 | +11.3 / +4.51% | 0/6 | +4.3 |
| 256 / skewed | 159.2 | +31.4 / +19.72% | 0/6 | +18.8 |

The judged path passes, 2.19646x, 95% CI 2.16351-2.22941; that score is not isolated
mode-79 performance. Spin flags are zero. Mode 79 fails its timing rule and its written
5e-4 output-comparison bound on one skewed 32-token seed, 5.1e-4 versus mode 72's 5.0e-4.
The gate's numerical failure remains recorded even though the judged receipt passes.
Mode 79 is not kept and its shippable follow-up was not run. At 128/256, replacing
mode 72's claimed phase B costs +8.1/+15.3 us on layer-like routing and +15.0/+7.0 us
standalone. Fixup costs, chained waits and code size are hypotheses, not separately measured
causes in this run. Reopen only after stamps show sub-1-us critical-path fixups per share
in both phases. [Receipt](../muse_moe_w4a16/profiles/bench_m79.2a4fb39964aa448c.json),
[scoped rejection](ruled-out.md).

### F/MoE diagnostics: stage-cost attribution revised

Job `43c35d18b5504eb5` at `1d42779` stamps a bench-only donor MAIN build beside mode 55,
mode 51's stamped twin. The donor is `6cae49a` with its MID helper line removed, not a new
whole-file F0 comparison. Six seeds 151-156, cold replays, 32/128/256 on layer/standalone
routing. Donor minus 51 is -3.0/-6.2, -9.8/-12.1 and -21.8/-22.4 us respectively;
all six seeds improve except 32 layer, five of six. The receipt passes, 2.19680x,
CI 2.16385-2.22976, spin flags zero. No paired CI.

At 128/256 layer-like routing, donor start is 3.4/3.8 us earlier; mean phase A is
3.4/10.3 us shorter; A end spread is 5.4/2.7 us narrower. Mean phase B differs by
0.0/-2.1 us. The stamped end gaps, 9.5-21.7 us across 128/256 routings, are mostly
phase A. These phase quantities overlap and must not be added as independent savings.
Donor fixups cost 10.4-12.8 us/CTA in A and 5.4-8.5 in B; doneA releases cost 2.5-7.7.
Share counts differ between instrumentation counters and remain unreconciled. Stamped donor
code has 16/20 B spill outside stage loops; stamped end times are within 1-3 us of unstamped
replays. Donor missing proxy fences remain unchanged and ineligible for direct adoption.
[Stamped experiment and per-phase table](runs-moe.md),
[raw log](../muse_moe_w4a16/profiles/bench_pubstamp.43c35d18b5504eb5.log).

A closer static review revises the proposed instruction-count explanation. The earlier
estimate of about 60 excess fast-path instructions becomes about 40; our dequant already
has the donor's 7 ALU + 2 IMAD.SHL form. Proposed mode 80 was not built because its expected
cut fell below its own gate. This is a revised hypothesis, not a measured mode-80 loss.

Job `1d767886e86d46e1`, source `2921d50`, tests timing-only mode 83, which omits the
per-thread weights-slot proxy fence, and repeats scale-pair mode 76. Mode 83 is explicitly
bench-only and cannot enter policy. Six seeds 161-166, cold median-of-20 replays; no paired CI.

| Tokens / routing | Mode 51 us | Mode 83 delta / percent | Seeds faster | Mode 76 delta / percent | Seeds faster |
|---|---:|---|---:|---|---:|
| 128 / layer | 154.8 | -0.1 us / -0.06% | 4/6 | +0.2 us / +0.13% | 0/6 |
| 128 / std | 206.2 | +0.2 us / +0.10% | 2/6 | +0.3 us / +0.15% | 2/6 |
| 256 / layer | 212.7 | -0.3 us / -0.14% | 4/6 | -0.9 us / -0.42% | 4/6 |
| 256 / std | 241.9 | -0.5 us / -0.21% | 5/6 | -0.8 us / -0.33% | 5/6 |

The receipt passes, 2.19253x, CI 2.15964-2.22542, spin flags zero; both arms are bit-identical
to 51 in every tested case. That does not license removing mode 83's fence. Its small
savings fail the diagnostic rule for the fence as the main cost, and the fence stays.
Mode 84's decode ablation was built but not run because it serialized WGMMA; mode 85 was
built but not run because it compiled to 51's loop. Neither has a GPU timing result.
[Raw ablation log](../muse_moe_w4a16/profiles/bench_abl.1d767886e86d46e1.log),
[receipt](../muse_moe_w4a16/profiles/bench_abl.1d767886e86d46e1.json).

Re-reading earlier mode-41 counters suggests waits account for the residual stage gap,
rather than cheap instruction cuts. The next proposed diagnostic is a counter twin of 51
that separates copy latency, producer delay and consumer skew. It is not built or measured.
This does not overturn B's staged losses or establish a new D scheduler win.
[Revised interpretation and next experiment](runs-moe.md).

### E: page-index bulk-copy implementation hangs before timing

`ONE_PBULK` in mode 49 replaces page-entry loads with a bulk copy into shared windows,
retaining existing ordering and adding a producer barrier. It is a producer-side E candidate,
not a measured win. Job `fcf904b32d9a427b` at `03675e8` hangs on `pb` at decode16 full
during ABCHECK; the guard stops the process. Control and matched control passed that shape.
The planned three-input, 200-round full-context bench never reaches timing or judging.
Job `61069efbf90d4c44` at `c811908` is a one-input, 60-round diagnostic plan; `dbg2`
hangs before timing even though it performs no bulk fills or window waits.
[First failure log](../muse_attention_decode/profiles/bench_p.fcf904b32d9a427b.log),
[diagnostic log](../muse_attention_decode/profiles/bench_q.61069efbf90d4c44.log).

Offline source/SASS bisection found the producer drain and EXIT missing for the unbraced
`if constexpr ... else for` form. Bracing the else restored producer EXITs. The default
build remains instruction-identical, and the fixed PBULK build changes only mode 49 with
no spills. That is a compile-level fix, not a GPU correctness clearance. The fixed build
still awaits a repeated check under the original gate. There are no new 128/256 timing
results, no scored result and no change to the pick-free 2.5366x attention entry.
PRQ wins and prior ring-depth/XPIPE losses remain. [Failure diagnosis and repeat plan](runs-attention.md).

### F-PROJ reimplementations: C1/C2 win paired tests; C1 holds scored entry

These are owned-code reimplementations of donor mechanisms, not verbatim donor candidates,
and belong to F-PROJ plus separate projection context, not A-E. C1 at `05286d9`, SHA256
f4141848, combines per-thread fail-closed NOVOTE waits, FULL addressing twins at M=TOKR
except qkv32, and a pooler128 cuBLASLt pin. C2 at `3044ee7`, SHA256 b2c4c096, adds a
register-direct 12-bit kernel offered at pooler1/8 and o8. [Run record](runs.md).

| Comparison / job | Method and uncertainty | Result |
|---|---|---|
| C1 versus release2265fa2 / `5ec16ca4d42a4e13` | Four independently prepared arms, RDRD order, three inputs x50 cold rounds,129/129 checks per arm | Pooled weighted ratio 1.0092, crosses 1.0082-1.0102, nulls 0.9981/1.0002; candidate faster |
| C2 versus C1 / `696179090d264f38` | Same four-arm method,129/129 checks per arm | Pooled1.0066, crosses 1.0060-1.0072, nulls 1.0001/0.9990; candidate faster |

These are successive paired comparisons, not a directly measured C2-versus-release aggregate;
do not add the two gains. No paired CI is recorded. Stub judged scores are not candidate scores.
At 128, C1 pooler D1-R1 is -1.33us against 21.93us, -6.06%, but the other cross deltas
range -0.26 to -0.86us with R/D nulls -0.64/+0.32us. The whole-candidate decision is stronger
than this individual cell. C2 changes small-token paths: pooler8 D1-R1 -1.06us against 15.21,
-6.97%; o8 -0.35 against 6.40,-5.47%; pooler1 -0.32 against 14.43,-2.22%. All four crosses
favor these paths. No isolated C2 gain is established at 128/256, whose candidate code is unchanged.
[First paired log](../muse_bf16_proj/profiles/bench_c4a.5ec16ca4d42a4e13.log),
[second paired log](../muse_bf16_proj/profiles/bench_c4b.696179090d264f38.log).

Clean C1 check `a4ce752c0f434ac5` passes 9/9,1.49480x,CI 1.47238-1.51723.
Clean C2 `5c4e4e99b6ec450f` passes 9/9,1.51721x,CI 1.49445-1.53997.
Coordinator scored C1 three times, all pass: `2d897770a75a4f3f`1.49694x,
`e1d3d341549c484d`1.51698x, `8a9c8ca810f142dc`1.51590x.
C2 scored repeats `e745eb8ca902471e`1.51522x and `355fc7894f6546ed`1.50824x both pass.
Their 95% intervals overlap. Recorded board entry is C1's `e1d3d341549c484d`,
CI 1.49422-1.53973; the lower C2 samples do not overturn its paired gain against C1.
[Scored C1 receipt](../muse_bf16_proj/profiles/scored.e1d3d341549c484d.json),
[C2 receipt](../muse_bf16_proj/profiles/scored.e745eb8ca902471e.json), [all repeats](runs.md).

The exploration preserves losers and failed builds. f1-f5 at source 79f9043 use cold paired
rounds, with FP64 and fresh-input replay checks on surviving arms; no paired CIs. Raw logs
retain controls and every variant. Numbers below are paired timing deltas, not isolated score gains.

| Job | Scope and result |
|---|---|
| `a63fa0c391fd40fc` f1 | Global-x escapes +0.13us qkv8,-0.13 slice8,+0.4 to+1.2 at 16+; donor geometries +0.02 to+1.17. Pooler128 donor pin -0.78,56/60 faster; qkv8 null0.45 |
| `6d5554de901143d0` f2 | Plain escape loads qkv32 -0.13, others within 0.13; escape-off attribution pool32 -0.34,pool16 -0.10. FULL arms dropped because checked at unsupported sizes. Pooler128 pin -1.36,59/60 |
| `a8de7b162c9f41d5` f3 | FULL/no-watchdog qkv32 -0.77,qkv16 -0.58,pool32 -0.46,pool8 -0.30. No-watchdog is diagnostic. Router GRED counter join loses+3.5 to+6.9; pooler128 pin -1.22 |
| `8712eb94a71e4075` f4 | LITEWD retains vote and recovers little: qkv32 +0.10 versus no-watchdog -0.96. Router memset-zeroed GZERO loses+0.83 at 256 to+1.92us |
| `eebd764690d5470f` f5 | Fail-closed NOVOTE+FULL pool32 -0.45,pool16 -0.34, several 8/16 cases -0.18 to-0.19; NOVOTE alone qkv32 -0.45 versus combined -0.22. This supplies C1 |
| `bcabe48caa884e67` f6 and `1759290da58047ab` f7 | Both fail pooler m7 through a late-bound runner-closure k before 3044ee7. Timing observations favor pooler8/o8 but are not clean candidate validation. Pooler16 loses+0.70 to+1.1,qkv8/16+1.7/+2.9,slice0 to+1.5; corrected C2 clean check follows |

The GRED join uses module-global accumulator/counters and cannot be ported into overlapping
layer launches without per-call state or a wait. Its fence-SC ordering argument was reviewed,
but this review makes no new ordering proof. [Port guide](proj-port-guide.md#global-split-k-join-wv_gred-f-proj-1-candidate-2026-10-31),
[exploration records and profile links](runs.md).

### Projection codec and router diagnostics: no new promoted router/codec candidate

Codec factorial `077b210c2087470d`, source recorded as proj-r8, compares C2 small-batch sk
picks with timing-only byte/decode perturbations.100 cold paired rounds; receipt passes 9/9,
1.51107x with CI 1.48841-1.53374, but byte-truncated arms are not correct codec implementations.
No 128/256 codec experiment. At qkv32, extra 16ALU instructions cost+1.81us and the combined
pair7/q12 models cost+3.04us. Other modeled combined deltas mostly improve. With qkv32
excluded, calculated pair7 weighted effect is+0.85%, below the+1.0% build trigger, so pair7
was not built. q12's+1.35% omits its 35-fold escape count and 32 KB LUT conflicts. rd/cgemv
codec savings remain unmeasured hypotheses. [Factorial log](../muse_bf16_proj/profiles/bench_f8.077b210c2087470d.log), [decision](runs.md).

Router rk-a `3fd0c7a7cea74ab4`, source proj-r8, and rk-b `e9d9499a465d4548`, source
`6d7f6a1`, use 100 cold paired rounds; both receipts pass9/9, including FP64/replay checks
for H arms. No paired CI. rk-a's standalone fill-node time around 1.1us includes a graph
launch and must not be treated as the zeroing cost inside the router graph. rk-b measures
only 0.16-0.25us saving from moving the clear into the kernel.

| rk-b case | Control median | Paired delta / percent | Rounds faster |
|---|---:|---|---:|
| 256 H+O, first-ticket clear + rk block order |9.49us|-0.61us/-6.43%|90/100|
| 256 H+O+J |9.49us|-0.70us/-7.38%|90/100|
| 256 duplicate H+O |9.49us|-0.67us/-7.06%|88/100|
| 256 memset+O |9.49us|-0.45us/-4.74%|81/100|
| 256 H without O |9.49us|+0.05us/+0.53%|46/100|
| 128 H+O, r64c16s8 |6.85us|+0.26us/+3.80%|27/100|
| 128 H+O, r64c12s8 |6.85us|+0.54us/+7.88%|13/100|
| 128 H+O, r64c8s8 |6.85us|+1.28us/+18.69%|0/100|

The duplicate H+O's -0.67 is its delta against production, not a -0.67 null between copies.
Production duplicate nulls are+0.06 at 128 and-0.06 at 256. The donor rk still saves 0.88/2.27us
versus production in the same run, leaving about1.6us versus H+O at 256 unattributed. The large
identified difference is block order; no extra gain should be attributed to barriers or spin.
H/WV_GTICKET is a new cross-CTA protocol awaiting PTX-chain review, not a promoted candidate.
Do not credit these diagnostics as an A-E or whole-layer gain.
[First stamp](../muse_bf16_proj/profiles/bench_rk.3fd0c7a7cea74ab4.log),
[clear/order test](../muse_bf16_proj/profiles/bench_rk2.e9d9499a465d4548.log), [decision](runs.md).

## Review at 21:26 PDT: F comparisons measured; combined layer gain and unresolved attribution

Historical checkpoint. The 22:35 review above resolves the missing c8d1 receipt and candidate attribution.

Evidence through `d73c4fd`. New measurements supersede the pending labels from the prior
review. The latest recorded decision makes F a measurement and learning track only.
Winning mechanisms must be re-implemented in owned code with attribution; F0/F1/F2 are
not submission candidates. Historical adoption recommendations below are superseded by
this [registry addendum](steering/2026-10-30-family-f-donor-baseline.md#addendum-2026-10-31-user-decision-on-the-use-of-f-results).

### Layer: combined M1 + 51 gain, isolated A still unresolved

All four-arm comparisons below independently prepare the release and candidate twice,
use two inputs and eight rounds per shape, and pass 24/24 checks per arm. Ratios are
weighted score ratios, not elapsed-time percentages. No paired confidence intervals are
recorded; independent nulls determine the written decision.

| Candidate / job | Source or file hash | Pooled ratio | Release / candidate null | Result |
|---|---|---:|---|---|
| layer_g51b / `64691c2e128247f4` | acd2385c | 0.9991 | 0.9993 / 1.0071 | Inconclusive; isolated mode 51 gain remains unresolved |
| layer_m1rt51c / `4d4ea5db46db4a10` | 1eda8ec9 | 1.0137 | 0.9884 / 1.0012 | Inconclusive; incumbent null too large |
| layer_hp / `6001272c2b1c4852` | 0efd553c | 1.0003 | 0.9987 / 1.0064 | Inconclusive; PRQ + HMRG + workspace pick bundle |
| Stable A/A / `e582fde557114062` | layer_stable 752123dd | 0.9989 | 0.9988 / 0.9997 | Expected null; stable-picker test passes |
| layer_m1rt51s / `3aee22ba4a404c8a` | f1235356 vs release_s a45fae28 | 1.0052 | 1.0008 / 1.0024 | Combined candidate faster under written rule |

The stable M1 + 51 comparison uses seeds 250001 onward and order seed 73. Its four cross
ratios span 1.0036-1.0068. At 128, sliding control medians 271.2/270.4 us have paired
cross deltas -3.9 to -4.7 us, or -1.44% to -1.74%; full controls 346.0/347.4 have
-4.4 to -5.8 us, or -1.27% to -1.67%. At 256, sliding controls 347.2/344.8 have
-0.7 to -1.6 us, or -0.20% to -0.46%; full controls 517.5/516.0 have -0.2 to
-4.4 us, or -0.04% to -0.85%, with full nulls -2.6/+1.6 us. The weighted 256
result remains inconclusive. Decode 1 regresses, ratio 0.9887. This is one measured
combined M1 + A gain, not an isolated family-A improvement.
[Layer evidence and per-case tables](runs-decode-layer.md),
[stable paired log](../muse_decode_layer/profiles/bench9_m1rt51s.3aee22ba4a404c8a.log).

The earlier comparisons also cover 128/256. g51b's weighted ratios are 1.0164/1.0041;
m1rt51c's are 1.0089/1.0071; hp's are 0.9964/1.0015. Their overall decisions remain
inconclusive. The workspace-pick hp build is withdrawn, rather than still awaiting a test.
M1 reproduction `9d00f194ed29471b` invalidated stream capture and produced no stage
records. M1 was off in the timed path, so it establishes no new M1 failure attribution.
[Layer run log](runs-decode-layer.md).

Three scored repeats of layer_m1rt51s at `0bc16c6` pass: `ad42a066bd3a4e01`
2.006935x, CI 1.976831-2.037039; `138ae1a402934c44` 2.001632x,
CI 1.971607-2.031656; `0f58d5d8223f45af` 2.014815x,
CI 1.984593-2.045037. The best is within noise of the old 2.0143x receipt.
This does not establish a cross-run score gain.
[Scored receipt](../muse_decode_layer/profiles/scored.0f58d5d8223f45af.json),
[repeat 1](../muse_decode_layer/profiles/scored.ad42a066bd3a4e01.json),
[repeat 2](../muse_decode_layer/profiles/scored.138ae1a402934c44.json).

**Unresolved source conflict:** the run log calls `c8d1e57261dd4901` a second M1+51
confirmation at 1.0106, while the [M-B plan](plan-2026-10-31-m-b-pdl-chain.md) identifies
it as layer_mb1s. No local receipt or raw log exists for that job. Do not count it as a
replication or assign its gain to either candidate yet. Release-manifest headings saying
these candidates are untested/unsubmitted are stale against the receipts above.

### F-LAYER: repeated measurement win with eligibility caveats

Jobs `20abe6291eb04ed6` and `00607fc6f8c84c77` compare pinned donor `6cae49a`
layer hash 8f17613b with release `2cee446`, hash b9c23bda. Builder source is `bc10cb1`
plus a scratch comparison patch. Each uses four arms, two inputs and 12 rounds per shape,
new seeds, preparation order and allocation pads; all 24/24 checks pass per arm.
Pooled score ratios are 1.0897/1.0949, with null pairs 1.0054/1.0009 and
0.9835/1.0011. Both clear the rule, but donor copies share module scratch, weakening that null.

At 128, first-run sliding deltas are -20.3 to -20.9 us against 293.3-295.0 us,
about -6.9% to -7.1%; full -23.9 to -26.8 against 379.9-383.1, about -6.3% to
-7.0%. At 256, sliding -32.6 to -34.8 against 362.0-362.3, about -9.0% to -9.6%;
full -16.6 to -20.3 against 512.5-515.1, about -3.2% to -4.0%. The confirmation
retains the sign; weighted 128/256 ratios are 1.0664/1.0893 versus 1.0734/1.0792
in the first run. No paired CI is recorded.
[Full cases and methods](runs-decode-layer.md),
[first log](../muse_decode_layer/profiles/benchF0a.20abe6291eb04ed6.log),
[confirmation](../muse_decode_layer/profiles/benchF0b.00607fc6f8c84c77.log).

These measurements do not clear the donor for adoption. Recorded findings include a T1
route reading PART/PO before its wait, an unresolved q8 nonunit-kv-scale precision question,
and shared growing scratch lifetime concerns. Stage attribution is provisional because
kernel names were lost. Do not assign the full gain to A, D or E. M-A MoE and M-B PDL
chain work are next; M-C requires an ordering repair before use.
[Eligibility review](runs-decode-layer.md), [M-B plan](plan-2026-10-31-m-b-pdl-chain.md).

### E and attention context: ring variants lose; workspace selection withdrawn

Round-4 benches use two inputs, 200 cold rounds and pooled medians. Every real arm passes
11/0 checksum checks; all five receipts pass 16/16. No paired CI is recorded. These switch
16/32-token full attention, not the 128/256 full paths, whose code is unchanged.

| Job / revision | Candidate versus control | 16 / 32 full delta us | Result |
|---|---|---|---|
| `bce37d22a0ec4045` / d12b20e | HMRGF merge, controls 97.89/173.25 | -0.29/-0.13 (-0.30%/-0.08%) | Small gain below 0.20% weighted promotion gate; separate merge context |
| `c8e99e038d9f4575` / 2bf3967 | QK2, controls 100.58/179.36 | +0.12/+0.32 (+0.12%/+0.18%) | No gate pass; PE16 -0.04/+0.03 is null |
| `4b60264a276c4805` / 0b326a4 | HMRGF repeat, controls 100.45/179.39 | -0.22/-0.19 (-0.22%/-0.11%) | Below gate; own-state -0.29/-0.35 with null -0.06/-0.22 |
| `42c4739cf6b247c4` / 5064e2b | E NS4/5/6 versus NS3, controls 100.58/179.49 | NS4 +0.80/+0.29 (+0.80%/+0.16%); NS5 +1.24/+0.99 (+1.23%/+0.55%); NS6 +1.72/+2.24 (+1.71%/+1.25%) | Losses, null -0.07/-0.07; NS6H +1.85/+2.27 also loses |
| `a7e5331d89ba4f77` / 5cb2be0 | E NS2 versus NS3, controls 100.61/179.26 | +3.65/+8.87 (+3.63%/+4.95%) | Loss; NS2H +3.39/+8.80 also loses; null +0.00/+0.20 |

NS3 remains best of these tested depths. PRQ's earlier win and XPIPE losses remain.
Producer-only zero-data and no-DRAM probes are diagnostics, not production gains.
[Round-4 records, all variants and profiles](runs-attention.md).

Workspace pick was removed, with kernel SASS unchanged. Own-code `dcda02e` check
`3250ab04126b4865` passes 16/16, 2.53674x, CI 2.49869-2.57479. Scored
`68d3b88fbc8b40da` passes, 2.53657x, CI 2.49852-2.57461, replacing the recorded
2.521 entry. Cross-run comparisons do not isolate the effect of removing the picker.
This is no new E pipeline gain. [Scored receipt](../muse_attention_decode/profiles/scored.68d3b88fbc8b40da.json).

### F-ATTN: controlled overall win, some large-token cells unresolved

Job `0519d0b15e9c4583`, source `589fb00`, compares release `a70192d` with pinned
attention donor `6cae49a`, hash e6d929d4. Four independent prepares, three inputs,
150 cold rounds across 12 shapes, 19/0 arm checks and 16/16 judge checks. Donor score
ratios 1.0097/1.0083 exceed the written gate and sum-null uncertainty of 0.50%.
No paired CI is recorded. The judged score is the release's, not a donor score.

| Tokens / context | Mean release us | Mean donor us | Delta us / percent | Decision |
|---|---:|---:|---|---|
| 128 sliding | 15.630 | 15.505 | -0.125 / -0.80% | Inconclusive against nulls |
| 128 full | 113.170 | 111.040 | -2.130 / -1.88% | Donor win |
| 256 sliding | 19.935 | 19.330 | -0.605 / -3.03% | Donor win |
| 256 full | 193.860 | 193.345 | -0.515 / -0.27% | Inconclusive against nulls |

Deltas subtract means of the displayed arm medians. Other resolved wins are d1 full,
d16 full, d32 full and d32 sliding. These belong to F, not isolated E. Two separate donor
checks `45adaeddef9d464e`/`ab38543d2c8642fb` score 2.57384/2.60532 versus release
check `b90af2a967b946b1` 2.54194, all pass 16/16; the cross-run difference alone is
inconclusive within recorded 1-3% spread. `cb99c110c9cd4c01` duplicates an old receipt,
not a new run. [F-ATTN methods and results](runs-attention.md),
[controlled log](../muse_attention_decode/profiles/bench_o.0519d0b15e9c4583.log).

### F-PROJ: repeated measured gain; documented host comparison patches

Jobs `434c01d39a1b4fa6`/`1ea00e9b5bbd49b0`, `bc10cb1` plus scratch bench,
compare release `2265fa2`, hash 6fe7b146, and donor `6cae49a`, original hash 290a86ab.
The comparison removes threading and serializes builds; kernel code is preserved, but this
is not a byte-identical whole-file F0. Four independent preparations use three inputs x40
rounds, then four x60, with new pads/seeds and reversed preparation order. All 129/129,
then 159/159 checks pass per arm, including fresh inputs, FP64, replay and retention.
Pooled weighted ratios 1.0363/1.0396 clear independent nulls 0.9968/1.0001 and
0.9999/1.0009. Stub judge scores are meaningless for donor performance. No paired CI.

| Case | First control us / paired delta | Confirmation control us / paired delta |
|---|---|---|
| qkv 128 | 14.94 / -0.86 (-5.76%) | 14.68 / -0.80 (-5.45%) |
| pooler 128 | 21.64 / -1.15 (-5.31%) | 21.61 / -1.15 (-5.32%) |
| router 128 | 7.15 / -0.82 (-11.47%) | 6.87 / -0.83 (-12.08%) |
| router 256 | 9.70 / -2.30 (-23.71%) | 9.39 / -2.40 (-25.56%) |

These are D1-R1 pairs. Full four-cross tables retain other cases and nulls. No resolved
single-token gain; several small-token qkv/pooler/o paths improve. Other 256 cases are mixed
or unresolved. The router's grid join relies on co-residency and needs redesign before a
layer port. Measured donor gains require reimplementation, not import.
[Full 30-case evidence](runs.md),
[first log](../muse_bf16_proj/profiles/bench_f4a.434c01d39a1b4fa6.log),
[confirmation](../muse_bf16_proj/profiles/bench_f4b.1ea00e9b5bbd49b0.log).

### Other projection diagnostics and hypotheses

Source for BG/RG/LTS is recorded as branch `proj-r6`, evidence committed in `5ef830b`;
the measured exact source SHA is not supplied. Receipts pass nine tests each. They do not
establish isolated candidate score gains; no paired confidence intervals are recorded.

- BG `cab546f978974515`: 60 cold paired rounds at 256, four independent weights.
  Fixed qkv config costs 17.47 us singly, 18.77/24.10 per instance at batch2/4.
  A different batched algorithm reaches 12.46 per instance. That changes workload and is
  no single-call win. o costs 8.70 singly, 7.26/7.20 per instance batched; nulls -0.08/-0.02.
  No 128 test; missing profiler names limit attribution.
- RG `97869fa8302041b5`: 100 cold paired rounds. At 128, control 7.07 us:
  A+S spin -0.22 (-3.11%, 84/100 faster), G -0.13 (-1.84%, 69/100), A watchdog
  +0.48 (+6.79%, 8/100). At 256, control 9.57: G/A+S +0.03, within null;
  A watchdog +0.19 (+1.99%, 24/100). Nulls -0.03/+0.00. Spin arm is not shippable;
  G+A lacks required twins, donor control failed CUDA13 build. No production change.
- LTS `8b9e8b7c1ecd4c3e`: 60 cold paired rounds after screening 1,680 configs/shape
  with FP64 checks. qkv256 large-tile splits +2.74 to +3.10 us versus 17.68,
  +15.50% to +17.53%, zero of 60 faster. qkv128 +1.18 to +1.46 versus 14.98,
  +7.88% to +9.75%, two to five of 60 faster. o256 +0.13 to +0.54 versus 9.01,
  with null +0.18, no winner. Same-tile qkv256 -0.18 with 36/60 faster is unresolved.

[Diagnostic run records and profile links](runs.md),
[BG log](../muse_bf16_proj/profiles/bench_bg.cab546f978974515.log),
[RG log](../muse_bf16_proj/profiles/bench_rg.97869fa8302041b5.log),
[LTS log](../muse_bf16_proj/profiles/bench_lts.8b9e8b7c1ecd4c3e.log).
Codec pair7/quad12 remain CPU models, not GPU wins. Predicted +1.44%/+2.58% depends on
hidden decode cost. Dense whole-tile sketch was declined without a run. Neither is family C.

### A component follow-ups: modes 76-78

Mode 76 pairs scale loads; 77 changes the small-token final pass; 78 combines them on 51.
Job `5b2db8029f094a66` at `45f990d` failed to build its donor comparison arm, so it
produced no A/B evidence. Its passing judged score 2.205x is not a candidate gain.
[Run record](runs-moe.md#donor-moe-m-a-against-mode-51-2026-10-31).

Job `64254f49676c4ce6` at `5f19bc1` tests six seeds 121-126 per cell. Job
`c8fc5f669fa34d06` at `95c9848` retests 78 with seeds 131-136. Both use cold-L2
median-of-20 graph replays per arm and paired seed means; no paired confidence intervals.
Layer-like routing is a component workload, not a whole-layer test. Negative deltas mean faster.

[bench_m76.64254f49676c4ce6 raw log](../muse_moe_w4a16/profiles/bench_m76.64254f49676c4ce6.log)

| Tokens / routing | Arm | Control us | Paired delta us | Percent | Seeds faster |
|---|---|---:|---:|---:|---:|
| 32 / layer | o76 | 90.7 | -0.1 | -0.11% | 3/6 |
| 32 / layer | o77 | 90.7 | -3.1 | -3.42% | 6/6 |
| 32 / layer | o78 | 90.7 | -3.5 | -3.86% | 6/6 |
| 32 / std | o76 | 149.3 | -0.1 | -0.07% | 4/6 |
| 32 / std | o77 | 149.3 | -1.8 | -1.21% | 6/6 |
| 32 / std | o78 | 149.3 | -2.4 | -1.61% | 6/6 |
| 32 / skewed | o76 | 69.0 | -0.0 | -0.00% | 3/6 |
| 32 / skewed | o77 | 69.0 | -1.9 | -2.75% | 6/6 |
| 32 / skewed | o78 | 69.0 | -2.9 | -4.20% | 6/6 |
| 128 / layer | o76 | 157.3 | -0.6 | -0.38% | 5/6 |
| 128 / layer | o77 | 157.3 | +0.7 | +0.45% | 2/6 |
| 128 / layer | o78 | 157.3 | -0.4 | -0.25% | 4/6 |
| 128 / std | o76 | 202.9 | -0.0 | -0.00% | 3/6 |
| 128 / std | o77 | 202.9 | -0.0 | -0.00% | 4/6 |
| 128 / std | o78 | 202.9 | -0.2 | -0.10% | 4/6 |
| 128 / skewed | o76 | 94.5 | -0.3 | -0.32% | 4/6 |
| 128 / skewed | o77 | 94.5 | +0.4 | +0.42% | 1/6 |
| 128 / skewed | o78 | 94.5 | -0.3 | -0.32% | 5/6 |
| 256 / layer | o76 | 223.0 | -0.8 | -0.36% | 6/6 |
| 256 / layer | o77 | 223.0 | +0.1 | +0.04% | 3/6 |
| 256 / layer | o78 | 223.0 | -0.9 | -0.40% | 6/6 |
| 256 / std | o76 | 247.3 | -0.5 | -0.20% | 3/6 |
| 256 / std | o77 | 247.3 | +0.5 | +0.20% | 3/6 |
| 256 / std | o78 | 247.3 | -1.0 | -0.40% | 5/6 |
| 256 / skewed | o76 | 149.6 | -1.1 | -0.74% | 6/6 |
| 256 / skewed | o77 | 149.6 | +0.0 | +0.00% | 3/6 |
| 256 / skewed | o78 | 149.6 | -1.3 | -0.87% | 6/6 |

[bench_m78.c8fc5f669fa34d06 raw log](../muse_moe_w4a16/profiles/bench_m78.c8fc5f669fa34d06.log)

| Tokens / routing | Arm | Control us | Paired delta us | Percent | Seeds faster |
|---|---|---:|---:|---:|---:|
| 32 / layer | o78 | 90.8 | -2.6 | -2.86% | 6/6 |
| 32 / std | o78 | 145.0 | -1.5 | -1.03% | 5/6 |
| 32 / skewed | o78 | 70.5 | -3.1 | -4.40% | 5/6 |
| 128 / layer | o78 | 161.6 | -0.6 | -0.37% | 4/6 |
| 128 / std | o78 | 191.3 | +0.6 | +0.31% | 1/6 |
| 128 / skewed | o78 | 99.7 | -0.8 | -0.80% | 5/6 |
| 256 / layer | o78 | 218.5 | -0.6 | -0.27% | 6/6 |
| 256 / std | o78 | 236.7 | -0.4 | -0.17% | 5/6 |
| 256 / skewed | o78 | 163.3 | -1.3 | -0.80% | 5/6 |

Both receipts pass, with spin flags zero. In the first gate, all arms match 51 exactly at
128/256; at 32 the maximum difference is 4.7e-5. Mode 76 misses its performance threshold
and its written 1e-5 numerical bound at 32, where the run record attributes variation to
51's atomics. Mode 77 misses the +0.5 us regression limit at 128 layer-like routing.
Thus neither, nor their conditionally eligible combination, was kept. The separate 78 gate
passes output checks, exact at 128/256 and at most 3.8e-5 at 32, but loses +0.6 us at
128 standalone against the +0.5 limit. It is not kept. Its 32-token gains remain measured.
The next fin4 idea restricts dispatch to below 64 tokens; it has not been tested.
[Gate decisions and raw receipts](runs-moe.md), [rejection register](ruled-out.md).

### F-MOE: complete-file measurement, not an eligible replacement

Exact donor `6cae49a` versus release `ec565a7` in jobs `40b8c798289a4258` and
`218ff300de7d4c22` gives pooled weighted score ratios 1.0387 and 1.0297. Independent
nulls are 0.9996/0.9993 and 1.0017/0.9977. Each uses four independently prepared arms,
12 standalone seeds, six layer-like seeds, four robustness seeds, median-of-20 cold replays.
Fresh-input replay passes and no numerical failures occur. At 1 token ours wins; at 8 the
donor wins; 16 is inconclusive across routings; 32/128/256 favor the donor.

At 128 standalone, first-run R1/R2 means 197.36/197.40 us versus D1/D2 182.63/183.16,
paired deltas -14.73/-14.24 us (-7.46%/-7.21%). At 256, 255.85/256.77 versus
228.85/228.81, -27.00/-27.97 us (-10.55%/-10.89%). All 12 seeds improve in each pair.
The confirmation preserves the sign; raw means and all routing ratios remain in the source.
No paired confidence interval is recorded.
[Full F-MOE evidence](runs-moe.md#f-moe-0-family-f-donor-comparison-2026-10-31-utc-2026-10-30-pdt).

The donor fat path lacks the writer/reader global proxy fences required by its generic-to-async
handoff. Passing numerical tests does not establish safe ordering. F1 fence costs remain
unmeasured. These wins guide an attributed re-implementation with the required fences, not
verbatim submission. Both-phase stream-K with the donor walk order is a proposed coherent
alternative; modes 72/74 did not test that full combination. A-versus-B's staged gate is unchanged.

## Historical review, 14:05: designated finisher tested; ownership stays ahead

Corrected mode 74 at `4187b8b` loses against mode 51 in job `e02f60fec94f45cb`.
This is a stream-K alternative to C inside A, retaining D-related phase-B claims.
It uses a designated finisher instead of mode 72's last-arrival fixup. The finisher is
chosen during setup from the cost walk. It is now tested, rather than an untested idea.
[Design and gate](runs-moe.md#designated-finisher-for-stream-k-phase-a-modes-74-and-75-2026-10-30).

Six seeds 107-112 per size/routing use cold-L2 graph replay, medians of 20 replays per arm.
The following are means of paired component deltas, not whole-layer results. Percentages
use the logged control means; rounding can differ from subtracting the displayed arm means.
No paired confidence interval is recorded.

| Tokens / routing | Mode 51 control | Mode 74 delta | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 80.3 us | +7.9 us | +9.84% | 0/6 |
| 32 / standalone | 143.6 us | +7.6 us | +5.29% | 0/6 |
| 128 / layer-like | 163.2 us | +4.6 us | +2.82% | 0/6 |
| 128 / standalone | 201.4 us | +7.4 us | +3.67% | 1/6 |
| 256 / layer-like | 219.9 us | +2.0 us | +0.91% | 1/6 |
| 256 / standalone | 233.1 us | +4.1 us | +1.76% | 1/6 |

The receipt passes, 2.1974x with 95% interval 2.1645-2.2304. Its judged path is the
unchanged board dispatch, so this score is not mode 74's standalone score. First-seed
output checks against 51 are 1.3e-4 to 3.4e-4 relative L2; spin flags are zero.
The diagnostic requires a negative mean at 256 on both routings. It fails, so no held-out
or skewed gate ran and mode 74 remains parked.
[Raw log](../muse_moe_w4a16/profiles/bench_sk74fix.e02f60fec94f45cb.log),
[receipt](../muse_moe_w4a16/profiles/bench_sk74fix.e02f60fec94f45cb.json).

The earlier build `5f35018`, job `eaa881a934aa471e`, used slow SiLU because its predicate
excluded modes 74/75 and used serial finisher designation. Preserve its results as that
build's loss, not the corrected design's result. Six seeds 95-100, same replay method:

| Tokens / routing | Mode 51 control | Earlier mode 74 delta | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 85.1 us | +8.2 us | +9.64% | 0/6 |
| 32 / standalone | 142.1 us | +9.0 us | +6.33% | 0/6 |
| 128 / layer-like | 159.9 us | +8.3 us | +5.19% | 0/6 |
| 128 / standalone | 209.3 us | +7.2 us | +3.44% | 0/6 |
| 256 / layer-like | 222.0 us | +2.1 us | +0.95% | 0/6 |
| 256 / standalone | 248.9 us | +3.8 us | +1.53% | 1/6 |

That receipt passes, 2.2023x with interval 2.1693-2.2354. First-seed output differences
are 1.5e-4 to 2.0e-4, spin flags zero. No paired interval is recorded.
[Earlier log](../muse_moe_w4a16/profiles/bench_sk74.eaa881a934aa471e.log),
[earlier receipt](../muse_moe_w4a16/profiles/bench_sk74.eaa881a934aa471e.json).

The corrected stamped twin reduces phase-A end spread to 6.6-8.3 us at 128/256,
versus 10.9-19.5 for the control. Fixups cost 3.9-4.5 us per CTA, but route plus setup
still costs 3.4-3.6 us more. Mode 72's larger fixup cost and the public kernel's similar
end spread come from other processes/seeds. They are diagnostic comparisons, not paired
speedups. The remaining setup cause is unresolved. Reopen this implementation only with
setup under 1 us above 51, per-share fixup under 1 us, or a refitted cost table, then apply
the paired gate. These findings retain C for this comparison, not reject stream-K generally.
[Measured stamps and hypotheses](runs-moe.md), [revisit triggers](ruled-out.md).

### Separate projection context: one-wave QKV cluster split loses

Job `ad6c673d9a444c1e`, source `90ee4b9`, tests qkv8c4/qkv16c4/qkv32c4 and wider
c4w variants, 72 rows per CTA and four CTAs per cluster. All configurations passed FP64,
2-ulp twin checks and 22 identical one-hot columns. The dedicated paired timing loop failed
with a `torch.sum(out=)` error. These results are prepare-time tuner observations only:
cold-L2 graph replay, 24 alternating repetitions, one preparation, overhead included.

| Tokens | Unsplit qkv reference us | c4 us / delta / percent | c4w us / delta / percent |
|---|---:|---|---|
| 1 | 13.8 | 23.0 / +9.2 / +66.67% | 24.9 / +11.1 / +80.43% |
| 8 | 13.9 | 23.7 / +9.8 / +70.50% | 25.5 / +11.6 / +83.45% |
| 16 | 14.7 | 24.1 / +9.4 / +63.95% | 25.7 / +11.0 / +74.83% |
| 32 | 15.6 | 27.9 / +12.3 / +78.85% | Not tested |

At one token the tuner selects cgemv at 12.8 us; the table uses qkv8 at 13.8 us
as the unsplit geometry reference, not the selected incumbent.
No paired confidence interval exists, and 128/256 were not tested by this experiment.
Raw twins also lose. Receipt passes 9 tests, 1.5089x with interval 1.4863-1.5315, but the
judge selected production picks, not c4. The experiment is parked; production is unchanged.
Cluster residency and decode geometry are hypotheses for the loss, not established causes.
Reopen with measured active clusters at least 32, or a configuration at most 113 KB shared
memory whose raw twin beats qkv16raw.
[Run record](runs.md), [raw log](../muse_bf16_proj/profiles/proj_c4_bench.ad6c673d9a444c1e.log),
[receipt](../muse_bf16_proj/profiles/proj_c4_bench.ad6c673d9a444c1e.json), [scoped rejection](ruled-out.md).

### Family F registration and next decision

Family F is now registered separately as the donor-based baseline/hybrid, pinned to
alygg77 source `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`. F0 is the exact donor,
F1 a documented conforming adaptation, F2 a measured hybrid. Projection, attention, MoE
and layer tracks are pending. No controlled F gain, correctness clearance or promotion is
established. Archived donor scores are external evidence only.
[Registration](steering/2026-10-30-family-f-donor-baseline.md).

The shared decision now prioritizes complete donor-versus-release projection and attention
comparisons before new GPU rewrites, with independent preparations and null controls.
A coherent replacement or measured hybrid is an allowed outcome. A-versus-B at 128/256
remains unchanged: retain the working baseline, and reopen staged B only under its stated
phase-B/phase-A-tail trigger. E's PRQ gain and XPIPE losses stand; layer_hp and revised
mode-51 layer candidates remain pending. Whole-layer release stays 2.014x. No gains are
added across families or projection context.
[Donor-first decision](steering/2026-10-30-donor-first-adoption.md), [layer log](runs-decode-layer.md).

## Historical review, 12:44: projection split exit barrier wins paired router tests

This is separate projection context, with no new A-E or whole-layer result. Job
`4f3ff774981a494c` at `db0f8ad` compares relaxed split exit barriers (wd KLUEYZ,
rt mode 3) with a copy of `ce25f67`'s conforming tail (wd KLUET, rt mode 1).
The arrive moves after the receive wait and before reduction/output stores; the exit wait
stays. Router workloads use 8/16/32/128/256 tokens, not MoE routing distributions.
All arms passed FP64 checks on two router matrices, adjacent sizes and two graph replays.
[Run record](runs.md), [raw paired log](../muse_bf16_proj/profiles/proj_tail_bench.4f3ff774981a494c.log).

One same-process experiment uses 100 seeded cold-L2 rounds. Deltas are paired medians;
percentages divide that delta by the logged control median, rather than subtracting marginal
medians. No paired confidence interval is recorded.

| Router tokens | Control median | Relaxed split delta | Percent | Rounds faster | Production-copy delta |
|---|---:|---:|---:|---:|---:|
| 8 | 4.64 us | +0.00 us | +0.00% | 45/100 | -0.06 us (-1.29%) |
| 16 | 4.96 us | -0.22 us | -4.44% | 76/100 | -0.18 us (-3.63%) |
| 32 | 5.79 us | -0.03 us | -0.52% | 51/100 | +0.00 us (+0.00%) |
| 128 | 7.71 us | -0.45 us | -5.84% | 94/100 | -0.32 us (-4.15%) |
| 256 | 10.21 us | -0.32 us | -3.13% | 86/100 | -0.26 us (-2.55%) |

The wd relaxed duplicate saves 0.42/0.29 us at 128/256 (5.45%/2.84%; 88/87 rounds
faster), while duplicate-control nulls are +0.00/-0.03 us. The rt 8/32 results remain
inconclusive. The release-arrive split loses +0.35/+0.26/+0.40 us at 8/16/32
(+7.54%/+5.24%/+6.91%; 7/16/9 rounds faster); its wd savings are only 0.14/0.16 us
(1.82%/1.57%; 71/72 rounds faster). The no-exit-barrier arm is a diagnostic bound,
not a candidate. [Scoped rejection](ruled-out.md).

The bench receipt passes 9 tests and scores 1.4972x (95% CI 1.4747-1.5196).
Clean check `25ea2b62ff8a47a7` at `2265fa2` passes 9 tests, 1.4931x
(1.4707-1.5155). Coordinator scored run `d85d0081fa8a4a43`, same source, passes
and scores 1.4976x (1.4751-1.5200). These overlap the previous 1.495/1.496 runs:
aggregate improvement remains inconclusive across runs. The paired gain is relative to
`ce25f67`, not proof of a gain over the older projection baseline at 256.
[Bench receipt](../muse_bf16_proj/profiles/proj_tail_bench.4f3ff774981a494c.json),
[clean receipt](../muse_bf16_proj/profiles/submission_tail1.25ea2b62ff8a47a7.json),
[scored receipt](../muse_bf16_proj/profiles/scored.d85d0081fa8a4a43.json).

The recorded ordering review calls the split sound with notes. Its relaxed form relies on
arrive executing after the receive-wait loop; the guide notes that the ISA does not explicitly
state the barrier-speculation property. Passing tests do not establish ordering safety.
A future CLR>1 PUSH configuration needs a fresh review. The o128 library pin is still
unresolved because six preparations select different winning configurations; P3 stays parked.
[Ordering argument and caveats](proj-port-guide.md), [run record](runs.md).

Keep standalone promotion separate from layer integration. Projection still needs a layer
port and paired layer test. A-versus-B, C ownership, D's untested full ready queue and E's
PRQ win/XPIPE losses are unchanged. `layer_hp`, `layer_g51b` and `layer_m1rt51c` still
await their recorded tests; the layer release remains 2.014x. Do not add projection savings
to any family score. [Layer run log](runs-decode-layer.md), [release manifest](release-manifest.md).

## Historical review, 05:39: stream-K phase-A retest loses

Job `098949b28fe84bd1`, source `86a633e`, tests mode 72 against mode 51. It replaces
whole-owned phase-A units with mode 16's cost-weighted stream-K partition and last-arrival
fixup while retaining 51's current stage loop, epilogue and dynamic phase-B claims.
This is the split-K alternative to family C inside A, with D-related phase-B claiming retained.
It is not staged family B or a full dependency-ready scheduler.
[Experiment and ordering review](runs-moe.md#stream-k-phase-a-on-51s-components-modes-72-and-73-2026-10-30).

Same-process cold measurements use six seeds 83-88 for each size and routing distribution,
20 graph replays per arm per seed after L2 flush. Layer-like routing is a component workload,
not a whole-layer test. Percentages divide the logged paired mean delta by the control mean.

| Tokens / routing | Mode 51 control | Mode 72 minus control | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 88.7 us | +13.5 us | +15.22% | 0/6 |
| 32 / standalone | 151.0 us | +9.4 us | +6.23% | 0/6 |
| 128 / layer-like | 166.1 us | +12.6 us | +7.59% | 0/6 |
| 128 / standalone | 195.3 us | +7.1 us | +3.64% | 0/6 |
| 256 / layer-like | 220.3 us | +6.6 us | +3.00% | 0/6 |
| 256 / standalone | 227.9 us | +9.2 us | +4.04% | 0/6 |

No paired confidence interval is recorded. Individual seed deltas remain in the
[raw log](../muse_moe_w4a16/profiles/bench_sk72.098949b28fe84bd1.log).
The [receipt](../muse_moe_w4a16/profiles/bench_sk72.098949b28fe84bd1.json) passes 5/5,
2.193x (2.160-2.226); that judge score is not an isolated mode 72 gain. First-seed output
checks in all six cells differ from 51 by 1.5e-4 to 2.4e-4 relative L2, consistent with the
changed FP32 summation order. Spin flags are zero. The diagnostic condition fails at 256 on
both routings, so the held-out/skewed gate was not run and nothing ships.

Stamped mode 73 versus 55 shows phase-A end spread widening to 15-19 us at 128/256 from 8-15 us.
Fixups are estimated at 9.4-10.9 us per CTA, with about 1.85 cut segments per CTA. The per-share
count path is about 3.3 us and finishing read-back 6-9 us. ABFIX's total subtracts a mean
pre-count time and is an estimate; the per-share and per-finish columns are measured directly.
Unlike the previous claiming builds, this kernel compiles no duplicate loop. Global fixup
round trips and an outdated cost table are hypotheses for the loss, not separately measured
causes. A shorter gather in the stamped twin does not help: the last first-x-ready is later.

C's ownership choice remains favored for this comparison. Preserve the loss at its tested
scope: mode 72's partition plus last-arrival fixup on current components at 32-256 tokens.
Revisit with a fixup well below 1 us per share on the critical path, or a stage-cost table
refitted to the current loop, then compare in-process with ABFIX. At that checkpoint, designated finishing was untested. Mode 74 has since tested it and lost,
as recorded in the latest review; other cut placements remain open. [Scoped rejection](ruled-out.md).

A-versus-B is unchanged: the staged pair still needs its phase-B or phase-A-tail trigger.
D's full ready-only queue remains untested. PRQ's E win and previous XPIPE losses stand.
`layer_hp` still has no GPU check; the whole-layer release remains 2.014x. No new attention,
projection or layer measurement appears in this checkpoint. [Layer manifest](release-manifest.md).

## Historical review, 04:43: PRQ wins within E; claiming and compression implementations lose

Family E now has a per-request pipeline win: `ONE_PRQ=31` combines transpose overlap,
Q.K-first issue, fast copies, split P.V issue and an offset-test change on q_len=8 modes
1/3/11/15. This is a coherent five-change design, not an isolated gain for each ingredient.
It leaves flat full-context paths unchanged, so it does not invalidate the XPIPE losses.
MoE runtime phase-A claiming variants lose; their no-work controls also lose. The evidence
rejects these code layouts, not runtime claiming itself. B's staged comparison remains unchanged.
[Attention round 3](runs-attention.md), [MoE evidence](runs-moe.md).

### E: PRQ standalone win and untested layer bundle

Three cold benches, each two held-out inputs and 120 randomized rounds, compare PRQ against
HMRG-on attention with PRQ off. All pass 16/16 judge tests and ABCHECKSUM 19/0 for every arm.
Routing distribution is not applicable. Deltas below are logged pooled arm-minus-control
values. No paired confidence intervals are reported.

| Job / source | Verify-1 full (8 tokens) | Verify-16 sliding (128 tokens) | Verify-32 sliding (256 tokens) | Gate |
|---|---:|---:|---:|---|
| `a7a4c23e98ca484a` / `1baad66` | -0.64 us | -0.45 us | -0.96 us | Modeled score contribution G +0.86%, duplicate null -0.11% |
| `3c6e1a7f1cc644b6` / `300b915` | -0.70 us | -0.35 us | -0.87 us | Missed written null clause: G +0.55% vs required +0.59%; retained as failed confirmation |
| `409996a7214f411a` / `8a6c8bb` | -0.64 us (-3.26%) | -0.48 us (-3.02%) | -0.96 us (-4.59%) | Confirmed with matched-state null; G +0.72%, null about 0%; independent-state G +0.80% |

Confirmation controls are 19.65/15.87/20.93 us. Both inputs improve on all three targeted rows.
G is a sensitivity-weighted estimate over four changed rows, not a measured whole-layer score.
The second bench's gate was not relaxed: the third bench adds a same-state null and an
independent-state comparison. Placement picking also failed its original row clause in the
second bench, then passed the third bench's own-state controls: about 0.3-0.4 us at decode-1
full when the initial allocation is slow. That is a separate change, with no guaranteed saving
when the first allocation is already fast.
[First log](../muse_attention_decode/profiles/bench_g.a7a4c23e98ca484a.log),
[failed confirmation](../muse_attention_decode/profiles/bench_h.3c6e1a7f1cc644b6.log),
[matched-null confirmation](../muse_attention_decode/profiles/bench_i.409996a7214f411a.log).

Final-file checks `640ee49b691b45ac` and `8b00100bf85d4ef5` at `57f57dd` (comment-only
repeat) pass 16/16 at 2.588 (2.549-2.627) and 2.545 (2.506-2.583). The file combines
PRQ31, HMRG and workspace picking. Cross-run variation exceeds PRQ's modeled gain; unchanged
full-context rows move too. Scored `1896e72c11ae4b0f` at `a70192d` passes 16/16 at
2.521 (2.484-2.559), replacing the 2.504 bench-file attention entry with a clean file.
This is standalone attention, not the 2.014x layer board.
[Check](../muse_attention_decode/profiles/sub_prq.640ee49b691b45ac.json),
[repeat](../muse_attention_decode/profiles/sub_prq_r.8b00100bf85d4ef5.json),
[scored receipt](../muse_attention_decode/profiles/scored.1896e72c11ae4b0f.json).

`layer_hp.py`, SHA256 prefix `0efd553c`, ports PRQ31 + HMRG + workspace picking onto the
release, retaining MoE47 and no M1. Its source/compile checks are recorded, but it has no GPU
check or measured layer gain. A four-arm bench is prepared. It is an E-plus-merge/placement
bundle; a future gain cannot be assigned entirely to E. Existing layer_g51b/layer_m1rt51c
candidates still await tests. [Release manifest](release-manifest.md), [layer log](runs-decode-layer.md).

### A/D-related MoE claiming: code-layout losses

Five new MoE receipts pass 5/5. Detailed experiments use six seeds per routing distribution
(layer-like and standalone), normal routed workloads at 32/128/256; 128/256 outputs are
bitwise and spin flags zero. All deltas below are versus mode 51. No paired confidence intervals.

| Job / source / seeds | Implementation | Layer-like 128 / 256 | Standalone 128 / 256 | Interpretation |
|---|---|---|---|---|
| `48b9bf9d77d146ac` / `003be4b` / 47-52 | 60, remove per-thread gather threadfence, retain proxy fence and release chain | -0.4 / 0.0 us (-0.25% / 0.00%) | -0.5 / -0.4 us | Gate fails at 256; parked |
| `b7e43ee6a517487a` / `d5bafad` / 53-58 | 62, claim last phase-A round | +7.9 / +9.2 us (+4.86% / +4.35%) | +7.5 / +8.6 us | 0/6 faster everywhere |
| Same job | 63, claim last two rounds | +7.0 / +8.5 us | +6.9 / +7.9 us | 0/6 faster everywhere |
| `ed4591a2477041c2` / `a727b12` / 65-70 | 66, same merged-loop build with nothing claimed | +7.0 / +8.8 us (+4.38% / +4.15%) | +7.9 / +8.7 us | Code-shape penalty; 62 vs 66 ranges -0.3 to +1.6 us |
| `b2c92a71600d488b` / `30c5a58` / 71-76 | 69, lean separated-loop build with nothing claimed | +4.5 / +5.2 us (+2.65% / +2.36%) | +5.0 / +3.9 us | Null fails; claimed mode 68 not timed |

Mode 60 also gives -0.3/-0.3 us at 32 layer/standalone. Mode 62 loses +3.9/+6.1 us at 32;
mode 69 loses +3.3/+3.7 us. The mode 69 build has a second compiled phase-A copy and longer
stage loops, with phase-B slowdown too. Narrower phase-A end spread does not establish a
faster kernel. Reopen only with no second compiled copy and a K=0 null within 1 us of 51
at both 128/256 before timing claims. These are partial phase-A scheduling attempts inside A,
related to D, not a tested full dependency-ready queue.
[Gather log](../muse_moe_w4a16/profiles/bench_gnf60.48b9bf9d77d146ac.log),
[claims](../muse_moe_w4a16/profiles/bench_ca62.b7e43ee6a517487a.log),
[merged-loop null](../muse_moe_w4a16/profiles/bench_ca66.ed4591a2477041c2.log),
[lean null](../muse_moe_w4a16/profiles/bench_cl69.b2c92a71600d488b.log).

The fifth job, `3d8902381a194d56`, is a standalone scorecard diagnostic, source logged as
"this branch (tool only)" without an exact revision. Ten standalone-routing seeds 301-310,
20 cold replays and duplicate preparations compare current sk8w3s/mode 51 against the public
kernel. Public-minus-ours gaps at 128/256 are -9.60/-14.92 us (-4.83%/-6.39% of ours'
198.88/233.515 us means), public faster 10/10. Across-seed standard deviations 3.92/3.47 us
are not confidence intervals. At 1 token ours wins 1.37 us; at 8 public wins 0.80; at 16 the
0.04 us difference is a tie; at 32 public-minus-ours -1.54 us has SD 8.02 and only 4/10 faster,
so it is inconclusive. The scorecard does not revive B without its phase-B or phase-A-tail trigger. [Scorecard and controls](runs-moe.md).

### Separate projection context: P3 tested and parked

`4922f366a0c142ca`, source `f8855b7`, passes 9/9. Pooler 128 compressed 12-bit weights
are reconstructed into shared-memory WGMMA tiles. Two independently compiled/packed copies
use 100 seeded cold rounds, synthetic randn weights x0.01, 5107 escapes (up to 6/block);
escape-heavy correctness covers up to 39/block, bitwise raw-twin checks and FP64 checks.
CW6d3 loses +12.90/+12.67 us to its raw twin in A/B, 0/100 faster; controls 35.55/35.63 us
make this +36.29%/+35.56%. Duplicate nulls are 0.03-0.27 us. Other buffering choices also lose.
Against Lt 20.45 us, the paired CW6d3-A loss is 27.95 us (+136.67%); the raw twin itself loses
15.01 us (+73.40%). The memory-only compression saving is only 1.01 us, below the modeled 5 us.
No paired confidence interval is reported. No 256-token test occurred.

The tuner rejects CW. Its 1.496 (1.474-1.519) judged score measures the incumbent path,
not successful compression. Code is parked under experiments; production returns to ce25f67.
A decoder-release ordering question remains open for this unshipped experiment. Reopen only
a raw geometry within about 1 us of Lt and a measured decode cost below the traffic saving.
This is projection context, not a new B or E result.
[Run and model](runs.md), [bench log](../muse_bf16_proj/profiles/proj_p3_bench.4922f366a0c142ca.log),
[receipt](../muse_bf16_proj/profiles/proj_p3_bench.4922f366a0c142ca.json), [stop rule](ruled-out.md).

### Receipt reconciliation and next decision

Previously missing scored receipts now verify: attention `57b0bed35f694a23` passes 16/16,
2.479 (2.442-2.516); projection `1d277e7fe9c44712` and `d59bc7af342a465d` pass 9/9,
1.482 (1.460-1.504) and 1.488 (1.466-1.510). Their cross-run interpretation is unchanged.
[Attention receipt](../muse_attention_decode/profiles/scored.57b0bed35f694a23.json),
[projection P1](../muse_bf16_proj/profiles/scored.1d277e7fe9c44712.json),
[conforming projection](../muse_bf16_proj/profiles/scored.d59bc7af342a465d.json).

The priority is now the prepared layer_hp controlled comparison. Preserve the working layer
release and the A-versus-B gate. After repeated claiming-layout losses, the next claim design
must remove the duplicate compiled work rather than repeat a larger kernel. P3 needs a new
raw geometry before another compression test. Do not sum PRQ, HMRG, placement or projection gains.

## Historical review, 03:37: HMRG attention merge wins; projection ordering correction

A-E outcomes and the A-versus-B decision at 128/256 remain unchanged. `ONE_HMRG` is an
attention key-half merge before global reductions, not the family E cross-tile pipeline or
MoE family C. It is a standalone component win awaiting a separate layer port/test.
The XPIPE cold losses remain valid. No new MoE or whole-layer result is recorded.
[Attention round 2](runs-attention.md#standalone-round-2-10-30-a3-merge--finalize-and-the-batch-1-full-prologue).

### Separate attention context: HMRG

`ce92d32e26024bd2` at `d8d43f3` first reports HMRG cold ABSCORE 1.0180 against the
board-code control (duplicate null 0.9993). Per-input rows were truncated, so the sign gate
could not be checked from that log alone. Confirmation `d0819bb26907426b` at `9eafafc`
reports 1.0214 with the control's workspace, 1.0232 with an independent workspace, versus
null 1.0037. Both benches pass 16/16 judge tests and ABCHECKSUM 19/0 for real arms.
The bench uses 120 randomized rounds, two held-out inputs, seeds 1000/1001; routing distribution
is not applicable. Logged pooled cold deltas and control medians from the confirmation:

| Decode shape | Control | HMRG delta | Delta / control | Per-input deltas |
|---|---:|---:|---:|---|
| 1 token sliding | 9.89 us | -0.64 us | -6.47% | -0.54 / -0.70 us |
| 1 token full | 18.69 us | -1.54 us | -8.24% | -1.50 / -1.50 us |
| 16 tokens sliding | 14.34 us | -1.44 us | -10.04% | -1.47 / -1.41 us |
| 32 tokens sliding | 16.61 us | -0.83 us | -5.00% | -0.86 / -0.86 us |

Pooled deltas differ from per-input summaries. No paired confidence intervals are reported.
The independent-state result supports the gain despite the known decode-1 full placement
spread. HMRG changes only q_len=1 modes 1/3/11/13; 128/256-token verify paths have identical
code, so their timing movement is not an HMRG gain. It halves the q_len=1 global reductions
by merging key halves in shared memory first. The recorded independent ordering review finds
the added waits/barriers sound within its source scope.
[First bench](../muse_attention_decode/profiles/bench_e.ce92d32e26024bd2.log),
[confirmation](../muse_attention_decode/profiles/bench_f.d0819bb26907426b.log).

Candidate-only checks `5d309f9d13924dca` (`dc6b409`, SHA256 prefix `93297101`) and
`82b1bf0fad3244b9` (same code plus a comment) pass 16/16, scoring 2.512
(2.475-2.550) and 2.582 (2.543-2.620). These are cross-run observations; unchanged verify
paths also accelerate in the repeat. The coordinator's scored run `57b0bed35f694a23`, same
source, is logged as passed at 2.479x; its receipt, interval and test count are absent locally.
The attention board remains `e9085c86d89449c2` at 2.504x, a bench-file entry. The scored HMRG
result is the best clean candidate-only scored run recorded, not a board replacement.
[Check receipt](../muse_attention_decode/profiles/hmrg.5d309f9d13924dca.json),
[repeat receipt](../muse_attention_decode/profiles/hmrg_r.82b1bf0fad3244b9.json),
[scored log](runs-attention.md#scored-57b0bed35f694a23-2479x-2026-10-30).

Preserve rejected alternatives: HMRG plus last-CTA ticket (`ONE_T9`, `hmtk`) adds 8.67 us
at verify-1 full and 1.79/1.70/1.41 us at verify-1/16/32 sliding versus control in
`ce92d32e26024bd2`. Its additional 0.22 us saving at decode-1 full is inconclusive under
the placement null. Removing cooperative launch gains only 0.38 us at decode-1 full and
retains no promotion claim. Phase comparisons close the batch-1 full prologue as the
public kernel's advantage: both pay nearly the same cost. Cluster/DSMEM merge and overlapped
finalize remain unbuilt. The next meaningful HMRG gate is a complete-layer comparison after
porting, not another isolated pipeline attribution. [Rejection register](ruled-out.md).

### Separate projection context: conforming push paths

The current `ce25f67` candidate uses exit cluster barriers on every push path and local
stores for rt's own-rank share. It no longer relies on the two ambiguous behaviors discussed
in the previous review. This is a correction to the current candidate status, not a claim
that the old behavior was proven sound. [Port-guide decision](proj-port-guide.md).

`3abb1cd974e2477d` and `d7c662296eaa4413` at `ce25f67` (header-only difference) pass
9/9 and score 1.495 (1.473-1.517) / 1.496 (1.473-1.518). Lt picks agree; router32
switches between tied rt variants. The following are cross-run means of the two conforming
checks versus the two earlier confirmations, not paired isolated barrier costs:

| Router tokens | Earlier mean | Conforming mean | Change | Change vs scored d0dc80f |
|---|---:|---:|---|---|
| 8 | 4.26 us | 4.43 us | +0.17 us (+3.99%) | -1.25 us (-22.01%) |
| 16 | 4.54 us | 4.80 us | +0.26 us (+5.73%) | -0.44 us (-8.40%) |
| 32 | 5.48 us | 5.49 us | +0.01 us (+0.18%) | +0.04 us (+0.73%) |
| 128 | 6.61 us | 7.19 us | +0.58 us (+8.77%) | -0.30 us (-4.01%) |
| 256 | 9.60 us | 9.81 us | +0.21 us (+2.19%) | 0.00 us (0.00%) |

Percentages use the rounded logged means; d0dc80f controls are reconstructed from the logged
deltas (5.68/5.24/5.45/7.49/9.81 us). Two runs per build, no interval for these per-shape
changes. The previously measured non-conforming KLUE/rt wins remain historical; current
cross-run data shows no retained gain at 32/256. Do not transfer the old paired percentage
to the corrected candidate. [Run log](runs.md#conforming-push-paths-coordinator-decision-2026-10-30),
[receipt 1](../muse_bf16_proj/profiles/submission_r2t1.3abb1cd974e2477d.json),
[receipt 2](../muse_bf16_proj/profiles/submission_r2t2.d7c662296eaa4413.json).

Coordinator submission `d59bc7af342a465d` is logged at 1.4879x, source `ce25f67`, versus
previous projection board entry `1d277e7fe9c44712` at 1.4818x. Receipt/interval/correctness
count for this submission are absent locally; do not interpret the score difference as a
controlled gain. Projection remains unported to the layer. P3 remains unbuilt.

Next evidence: HMRG layer port and controlled layer test, receipts for the two new scored
runs, and a paired conforming projection comparison if isolated retained gains matter.
Standalone and layer promotion are separate decisions; do not add these component scores.

## Historical review, 02:38: E pipeline losses; separate projection router gains

Attention roadmap labels A1/A2 are experiments within family E, not family A (MoE).
The earlier DCARRY/PK16/VHOIST wins and their combined layer integration remain valid.
The new cross-tile designs lose on the cold-cache workload and have no layer integration.
A-D and the 128/256-token A-versus-B decision are unchanged. No new whole-layer result
appears in this checkpoint. [Attention evidence](runs-attention.md#standalone-round-10-30-a0-scorecard-and-a1-cross-tile-overlap-in-mode-51).

### E: real overlap tested, cold-cache gate failed

`5575e7d15d3e49a8`, kernel `86f2bec`, generator `8a0a4a0`: attention mode 51 with
`ONE_XPIPE=2` issues P.V before the next stage wait and overlaps Q.K/V transpose. The
control is the board attention code, not MoE mode 51. Each arm passes ABCHECKSUM 19/0;
judge receipt passes 16/16. The bench uses 120 randomized rounds and two held-out inputs
(seeds 1000/1001), with independent control states. Full attention verify 16/32 corresponds
to 128/256 tokens; routing distribution is not applicable.

| Candidate vs control | 128 tokens | 256 tokens | Status |
|---|---|---|---|
| XPIPE=2, cold pooled medians | 117.31 vs 113.41 us: +3.90 us (+3.44%) | 193.25 vs 192.45 us: +0.80 us (+0.42%) | Measured loss; 128-token gate fails |
| XPIPE=2, cold per-input deltas | +3.32 / +4.55 us | +0.54 / +0.99 us | Both inputs slower |
| XPIPE=2, warm per-input deltas | -1.8 / -1.9 us | -3.5 / -3.3 us | Warm-cache gain only; does not pass cold gate |
| Mode 115, one CTA/SM vs mode 51, cold | 137.79 vs 110.27: +27.52 us (+24.96%) | 244.93 vs 187.26: +57.67 us (+30.80%) | Measured loss |
| Mode 115 + XPIPE=1 vs mode 51, cold | 138.37 vs 110.27: +28.10 us (+25.48%) | 239.87 vs 187.26: +52.61 us (+28.09%) | Measured loss |

The first row subtracts pooled medians, not medians of paired differences. Logged pooled duplicate
nulls are -0.38/-0.06 us; no confidence interval for these deltas is reported. Warm percentages
are omitted because the summary does not supply the corresponding control times.
[First bench log](../muse_attention_decode/profiles/bench_a0.5575e7d15d3e49a8.log).

The mode-115 rows come from `318ae6fc3c3746ce`, source `682aeaa`: two held-out inputs,
120-round default, 16/16 judge tests, ABCHECKSUM 19/0 for real arms. Nulls are +0.06/+0.42 us.
XPIPE=1 saves 5.06 us (-2.07%) inside mode 115 at 256, but leaves it 52.61 us slower than
mode 51. At 128 it adds 0.58 us (+0.42%) inside mode 115. Q.K-first XPIPE=1 at two CTAs/SM
is compile-infeasible in this build (C7512 register serialization); the one-CTA variant compiles
without C75xx but loses residency. These results reject the tested configurations, not all
cross-tile pipelines. [Second bench log](../muse_attention_decode/profiles/bench_b.318ae6fc3c3746ce.log).

### Attention diagnostics and next target

`a92aa9c43f064c49` (`df4060d`) tests one-allocation workspace carving: 16/16 and ABCHECKSUM
19/0, but the control remains 0.54 us slower at decode-1 full on both inputs. Carving fails
its +-0.2 us gate. `7144310358564fa3` (`a0a78ce`, generator locally edited) passes 16/16
judged tests but has no usable bench data: undefined `_AB_CAP` breaks every bench row.
It establishes no capture-order result. Fixed repeat `b7762d764d474eae` (`46df563`), six
held-out inputs, 16/16 and ABCHECKSUM 11/0, finds a 0.5-0.7 us spread tied to workspace state
rather than capture order. L2 placement is the proposed mechanism, not a measured cause.
Cross-state decode-1 full differences below 0.7 us are inconclusive under this evidence.
[Carve log](../muse_attention_decode/profiles/bench_c.a92aa9c43f064c49.log),
[failed bench](../muse_attention_decode/profiles/bench_d.7144310358564fa3.log),
[fixed repeat](../muse_attention_decode/profiles/bench_d2.b7762d764d474eae.log).

The standalone scorecard puts most of the remaining gap in short/sliding shapes. Diagnostic
phase exits in `318ae6fc3c3746ce` assign about 2.4-4.0 us to merge/finalize on decode-1 sliding,
decode-1 full, decode-16 sliding and verify-1 full. These are diagnostic budgets, not achievable
savings. Attention A3 reduction ownership and prepare-time allocation selection remain untested;
A3 is not evidence for MoE family C. Next decisive test is a fixed-state, null-controlled
short-shape reduction comparison, with the log's gate of 0.5 us at decode-1 sliding or 1 us
at decode-16 sliding/decode-1 full. Reopen the one-CTA design only with a resource model that
recovers its residency cost. [Scoped failures and triggers](ruled-out.md).

### Separate projection context: rt and KLUE router gains

Five new receipts pass 9/9. R2a `a91f2152d8974458` (`113d513`) scores 1.505
(1.482-1.527), R2b `ba85985b8c5e4969` (`01ad9d9`) 1.484 (1.461-1.506), and
R2c `6cc72e840403402b` (`b50dcb6`) 1.499 (1.477-1.522). Confirmations
`f2373dedd4f44648` and `b80b817c67d043ad`, both `1e12506` with a header-only difference,
score 1.484 (1.462-1.506) and 1.508 (1.485-1.530). Their spread is not a controlled gain;
the run log attributes most of it to unchanged verify-1 shapes. The prior P1 submission
`1d277e7fe9c44712` at `d0dc80f` is logged at 1.482 (1.460-1.504), but its receipt is absent
locally, as are `44fa08c6d2f44b44` and `6cbd8b981a5d4c3f`.
[Projection round and receipt identifiers](runs.md#2026-10-30-round-2-router-attribution-rt-stable-cublaslt-picks-branch-worktree-agent-a7d5e0782cac00376).

R2c records 100 seeded cold rounds, paired against the pre-round production router picks:

| Variant / tokens | Paired delta | Control median | Delta / control median |
|---|---:|---:|---:|
| rt8w8 / 8 | -1.25 us | 5.38 us | -23.23% |
| rt16w8p8 / 16 | -0.88 us | 5.33 us | -16.51% |
| rt16w8p8 / 32 | -0.40 us | 5.74 us | -6.97% |
| wd KLUE / 128 | -0.80 us | 7.63 us | -10.48% |
| wd KLUE / 256 | -0.74 us | 9.97 us | -7.42% |

Percentages use paired deltas divided by control medians after the logged 4.24 us replay
overhead subtraction. They are component results, not layer savings. Nulls are about
0.02-0.13 us for rt and -0.13/-0.06 us for KLUE at 128/256; no paired confidence intervals
are recorded. FP64 checks cover two router matrices, outliers/tiny values, adjacent sizes and
fresh-input graph replays. Routing distribution is not applicable to these projection operators.
[R2c log](../muse_bf16_proj/profiles/bench_r2c.6cc72e840403402b.log).

Push reduction is the strongest single tested wd change (-0.46 to -0.66 us across R2a/b).
KLUE combines packing, promotion, push reduction and prologue changes; do not assign its full
win to one ingredient. Removing the producer warp with its barrier loses +1.74/+1.25 us at
128/256; XA orientation is a null, and rt8w8 loses +1.50 us at 32 in R2c. Stable Lt choices
now agree across confirmation preparations except o128 at the margin. P3 compressed WGMMA
remains unbuilt and parked for budget, with a raw-BF16 twin required before testing decode.
[Projection evidence](runs.md), [scoped losses](ruled-out.md).

The [port guide](proj-port-guide.md) still records unresolved outgoing `st.async` sender-exit
and rt self-rank questions. Passing correctness checks does not resolve them; TAILSYNC is the
recorded fallback, costing about 0.1-0.2 us for rt and 0.3 us for wd. No projection layer port
is present. These wins therefore stay separate from A-E and from layer promotion.

## Historical review, 01:31: standalone projection gains, no new A-E layer result

The new projection round is separate context, not a gain attributed to A-E. The
[standalone roadmap](steering/2026-10-30-attention-projection-standalone-roadmap.md)
separates standalone promotion from layer integration. A-E standings and the 128/256-token
A-versus-B decision remain unchanged. Direct cuBLASLt and router geometry changes have not
been ported to the layer. The full layer still needs its own controlled comparison.

The [projection run log](runs.md#2026-10-30-standalone-round-direct-cublaslt-and-router-geometry-p1)
records these jobs. Routing distribution is not applicable to these standalone dense operators.
Scores below compare against each job's SGLang sample, not a same-process release control.

| Job | Source revision / candidate | Score (95% CI) | Evidence status |
|---|---|---|---|
| `bf6c825bd5cf4659` | `eb374e4`, LATE_ARRIVE baseline, submitted by coordinator | 1.457 (1.436-1.479) | Receipt verified, 9/9 |
| `1b00244922654709` | `d61deea`, P1a | 1.465 (1.443-1.487) | Receipt verified, 9/9; Lt build failed, router changes only |
| `047bc3a4168e42d6` | `33daf8b`, P1b, Lt heuristic search | 1.463 (1.442-1.485) | Receipt verified, 9/9 |
| `d6eaf1e73db047a5` | `62a6ae0`, P1c, tile/cluster sweep | 1.484 (1.461-1.506) | Receipt verified, 9/9; prepare-time picks vary |
| `44fa08c6d2f44b44` | `d0dc80f`, P1d, cross-layout search and longer screening | 1.479 (1.457-1.501) | Run-log evidence; receipt absent from checkout, correctness count unverified here |
| `6cbd8b981a5d4c3f` | `d0dc80f`, clean candidate, file SHA1 `03389e0d` | 1.482 (1.460-1.504) | Run-log evidence; receipt absent from checkout, correctness count unverified here |

Verified receipts: [baseline](../muse_bf16_proj/profiles/proj_eb374e4.bf6c825bd5cf4659.json),
[P1a](../muse_bf16_proj/profiles/bench_p1a.1b00244922654709.json),
[P1b](../muse_bf16_proj/profiles/bench_p1b.047bc3a4168e42d6.json),
[P1c](../muse_bf16_proj/profiles/bench_p1c.d6eaf1e73db047a5.json).

P1d's log reports 100 seeded rounds, medians against preparation A's pre-Lt pick.
Independent preparations newA/newB save 1.87/2.99 us on pooler 128 and 2.98/2.46 us on
pooler 256. Null deltas are -0.13/-0.03 us respectively. Qkv 256 saves 1.92/1.86 us;
unpooler 128 saves 1.34/1.41 us. Percent deltas are unavailable from this table because
absolute paired control times are omitted. These are component comparisons, not layer gains.
Pooler 128 picks vary by about 1.1 us between preparations; no paired confidence interval is
recorded in the table. Full per-shape deltas, including near-ties and router gaps, remain in
the linked run log.

The clean candidate's cross-run verify-16/verify-32 totals (128/256 tokens) are
73.29 -> 70.22 us and 91.84 -> 84.29 us: -3.07 us (-4.19%) and -7.55 us (-8.22%).
Decode totals vary from -0.45 to +0.62 us with unchanged decode code. These observations
cannot establish an isolated change or be added to A-E gains. Preparation rises from about
70 to 105 seconds. [Port status and cost](proj-port-guide.md#direct-cublaslt-plans-lt_cpp-m--128--256-dense-2026-10-30).

Preserve the losses: Lt qkv 128/o 128 are near-ties in the searched space; several router
geometries lose outside their winning bucket; heuristic-only Lt search misses the pooler-256
winner. CUTLASS pooler 256 remains a tuner candidate but is superseded in this comparison.
P2 compressed qkv split-K is declined on a work model, without a new run; its reopening
condition is a one-wave geometry or persistent schedule that preserves startup and escape
cost. P3 compressed WGMMA is untested; its predicted gains are hypotheses. Neither is family
B or E evidence. [Scoped losses and reopening conditions](ruled-out.md).

Next evidence: obtain the two missing receipts, stabilize the remaining pooler-128 pick
variation, and use the proposed raw-BF16 twin before judging P3's compression mechanism.
Standalone gains can qualify on their own; any layer port still needs a full-layer A/B.

## Historical review, 00:32: scored candidates do not resolve M1 gains

Board remains `3fbb3ad027e84c6b`, 2.014x. Two new scored readings are cross-run observations,
not controlled timing gains, and do not change the unresolved mode-51 layer conclusion:

| Job / file | Score and correctness | M1 status | Interpretation |
|---|---|---|---|
| `b68bdcd7a1ba464c`, layer_m1rt, SHA256 `d6958449`, byte-identical to `dd1777e` | 2.013x (1.983-2.043), 13/13, prepare 143.1 s | Unknown: SUMMARY and policy lost from log tail | Cannot credit this score to enabled M1 |
| `d6c30f88c617438b`, layer_m1rt51 at `7cdb755`, SHA256 `65264f31` | 2.013x (1.983-2.043), 13/13, prepare 154.8 s | Off for the entire timed run, policy empty | Measures mode 51 in the SK_PDL/ONE_TRIG build with M1 off |

For `d6c30`, the first gate shape, decode 1 sliding, has one replay in 2,000 with relative
hidden/stream error 3.2e-4/1.9e-4 versus a bitwise control. Only the maximum-error envelope
fails; order-1 count is zero and the other criteria pass. At this token count both arms use
cgemv for o and fp32 front, so M1 changes no launch. This is not an isolated M1 ordering
failure; rare rounding noise is the report's hypothesis, with a reproduction prepared but
unrun. All 11 MoE variants pass their gates, no variant is rejected, mode 51 is bitwise at
128/256, and the picker chooses 51 at 32/128/256. Whole-layer times at 128 are
268.6/351.6 us, at 256 345.2/513.6, sliding/full; no paired control supplies a gain delta.
[Run log](runs-decode-layer.md), [raw gate results](../muse_decode_layer/profiles/layer_m1rt51.d6c30f88c617438b.log).

Two route negative controls fail sensitivity at 32: the row injection replaces a zero row
with another zero row and changes no output. This is a test-injection defect, not a measured
route correctness failure. The new tool chooses differing rows/values; synthetic CPU tests
reproduce old misses and remove row misses. GPU validation of the revised files is pending.
[Failure analysis](steering/2026-10-30-m1-1x1s-gate-failure.md).

The run log records a user decision to judge M1 only on shapes whose captured candidate graph
actually launches an M1 PDL secondary, with unchanged bounds on those shapes. If the launch
census fails, the gate still judges the shape. This review reports that record and changes
no validation. Prepared files, all without a new GPU check:

- `layer_m1rt51c`, SHA256 `1eda8ec9`: runtime M1 + 51, census skip and negative-control fix;
  rebuilt at `a48edef`, no device-code change from layer_m1rt51.
- `layer_g51b`, SHA256 `acd2385c`: release + 51, current gate and W4 early-exit wait fix,
  no M1. The previous four-arm mode-51 result remains unresolved.
- `layer_stable`, SHA256 `752123dd`: diagnostic build, not a candidate; release with nine interleaved tuner rounds, a paired
  margin and 8/9 winning-round rule before replacing the receipt's preferred pick. CPU
  tests retain a true tie in 1991/2000 draws and accept the synthetic 2 us win in 500/500;
  these are rule tests, not GPU speed or stability results. Four-arm A/A is prepared.
- Targeted 1x1 reproduction diagnostic: compiled/CPU-smoke-tested, not run on the judge.

[Manifest](release-manifest.md), [tuner-stability design](plan-2026-10-30-tuner-stability.md).
A-E component standings, B's retest loss and the four-arm uncertainty remain unchanged.

## Historical review, 23:35: mode 51 layer gain unresolved after arm-bias checks

The earlier three-arm bench timed the candidate twice using the same prepared state, but the
incumbent only once. It could not distinguish a candidate gain from an incumbent-arm penalty.
The [arm-bias audit](steering/2026-10-30-layer-bench-arm-bias.md) and new four-arm runs therefore
supersede the previous claim of an established isolated mode-51 layer gain. Preserve the old
numbers, including `29fd004fd58e40ef`'s 1.0084 and its 128/256 deltas, as historical observations.
Mode 47's old three-arm attribution shares this limitation. Its release and component checks
remain valid; the isolated layer magnitude needs the same caution.

Four-arm benches use independent prepare() per copy, random prepare/capture/replay order,
two held-out inputs per shape, eight rounds and both same-file nulls. The heuristic requires
all four cross ratios on one side of 1 and the pooled log ratio above twice the larger null.
It is not a formal significance test. Source bench builder `a877401`:

| Job / candidate | Seed start / order seed | Pooled speed ratio | Nulls | Decision |
|---|---|---|---|---|
| `724aa0dd6ed140ca`, release versus itself, file `9473029f` | 180001 / 43 | 1.0045 | 0.9965 / 1.0030 | A/A not resolved |
| `2dc566298be1498f`, mode 51 `dacd4e0d`, bench `e95f3218` | 160001 / 37 | 1.0057 | 0.9898 / 0.9972 | Mode 51 not resolved |
| `40b9b3f795f84073`, runtime-order M1 + 51 `d57b5e38`, bench `824e1e45` | 170001 / 41 | 1.0162 | 0.9928 / 0.9970 | Combined candidate faster by heuristic |

Each compares against release `b9c23bda`. These runs' judge scores are 2.009x (1.979-2.039),
2.000x (1.970-2.030), and 1.992x (1.962-2.022), respectively. These are check readings,
not new releases. All 96 checks pass (24 per arm); prepare-time policy differences contribute
to the nulls. A/A retains a decode-1 offset: pooled 1.0262 there, with incumbent 3-6 us slower
than the other copies. Exclude decode 1 from attribution. [Layer run log](runs-decode-layer.md).

At 128/256, mode 51's pooled load ratios are 1.0080/1.0107, equivalent to about 0.79%/1.06%
less weighted elapsed time. Weighted arm times are 283.6/285.7 us for the two incumbents
versus 282.7/282.1 for candidates at 128; 375.7/374.2 versus 371.8/370.1 at 256. The overall
cross range 0.9991-1.0123 and larger null about 1.03% leave the gain unresolved. Do not invent
a single paired microsecond delta from these independently summarized arms. Policy flips
include front/boundary/W4 choices even between copies of the same file. Raw four-cross paired
deltas at 128 range -1.6 to -3.2 us sliding and -0.4 to -4.7 full; at 256, -4.4 to -7.6
sliding and -3.0 to +0.5 full. These are ranges, not a resolved pooled microsecond gain.
[Raw mode-51 four-arm bench](../muse_decode_layer/profiles/bench4_g51.2dc566298be1498f.log).

Runtime M1 + 51 crosses 1.0111-1.0214 and clears its larger null about 0.72%. Per-load pooled
ratios are d1 1.0320, d16 1.0074, d32 1.0065, verify 1 (8 tokens) 1.0166, verify 16 (128)
1.0257 and verify 32 (256) 1.0096. Verify-1 sliding paired differences range -2.1 to -2.9 us.
The aggregate combines M1, mode 51 and the decode-1 offset. Dividing by A/A suggests about
1.2% combined gain; the report's approximate 0.4% M1 share is cross-run arithmetic, not an
isolated measurement. M1 operates only at <=32; 128/256 improvements cannot be credited to
it. The same mode-51 code reads 1.0080 at 128 in one run and 1.0257 in another. Do not add
component scores. [Raw combined bench](../muse_decode_layer/profiles/bench4_m1rt.40b9b3f795f84073.log).

The earlier runtime-order measurement `f0ee07c4d4904511`'s 0.87% three-arm result is also
unestablished. It motivated the corrected harness, not a separate accepted gain. The new
`cand/layer_m1rt.py` (SHA256 `d695844971f2bc7ab85720ad3a464c3464fc7efb01182746b4a96bcf14be7269`)
is **mode 47 + runtime M1**, with enforced 2,000-replay fresh-input gates and negative controls.
The measured combined build used mode 51 and report-only 1,000-replay gates. Those builds are
not interchangeable: the new candidate has no check or submission. Board remains 2.014x.
[Release manifest](release-manifest.md).

### Attention history resolved for the tested mode; ordering review completed

Attention diagnostic `2b93c6ecf4f641c6`, source `389da5c`, passes 16/16, 2.462x
(2.425-2.499), with production SASS unchanged. This is a diagnostic score, not a speedup.
Mode 49, decode 16/32 full, context 80000, page 1, four qkv inputs: production gives four/five
distinct outputs in nine calls on one input (1-3 elements, one BF16 ulp). A fixed-order debug
merge gives bitwise outputs across repeated, changed and separate-case histories, flushes and
interleaved calls. All 283 workspace scans are clean. This supports merge-order rounding,
not carried input values, for these tested cases. The earlier bitwise repeated-input reading
was a sample, not a general property. Do not claim every attention configuration is proved.
[Attention diagnostic](runs-attention.md#input-history-dependence-at-decode-16--32-full-merge-order-not-a-carry-10-30).

The [independent M1 ordering review](review-m1-ordering-2026-10-30.md) and
[other-PDL audit](review-pdl-secondaries-2026-10-30.md) replace the "unreviewed argument" status.
The review supports the fenced runtime design under its stated assumptions. The other-PDL
audit finds no issue on the shipped scored path; a latent W4 GEMM2 early return before its
wait is off that path and is fixed in the new unchecked candidate. Neither review substitutes
for checking that exact candidate with enforced gates.

### New MoE losses and limits

`7063f86352344efa`, source `e866c8a`, layer seeds 1-6 at 128/256, selects x gather as the
larger lead: instrumented gather 6.8/8.3 us versus public 3.6/4.9, while the trailing-warpgroup
skew bound is only 0.45/1.07 us. Those are attribution bounds, not recoverable layer savings.
`e7e75c9510994c08`, source `fca2be7`, layer seeds 41-46, tests mode 57 (early x loads plus
16-byte stores) and 59 (early loads only), versus 51 at 32/128/256. Mode 57 loses
+1.2/+1.1/+1.2 us, zero of six faster at each size. Mode 59 is inconclusive/tied:
-0.0/+0.6/-0.3 us, 2/1/5 of six faster. Checks pass with spin flags zero; a mode-59 normalized
output difference at 32 is not separated from rounding or a defect, and it is not kept.
Neither is gated or shipped. Shorter per-CTA gather does not materially advance the first
job's readiness. These are scoped A-descendant results; B's staged retest and D's full
ready-only status do not change. [MoE run log](runs-moe.md).

## Historical review, 22:34: mode 51 layer gain and measured M1 limits

Mode 51 closes its pending layer test. Job `29fd004fd58e40ef`, bench source `d9b7d67`, compares
`cand/layer_g51.py` (SHA256 prefix `dacd4e0d`, sent comment-stripped) with clean mode-47
release `b9c23bda`. Two held-out inputs per shape, seeds 140001+, eight cold-L2 rounds with
seeded random order give 16 paired samples per shape and 72/72 reference/routing/cache checks.
Candidate/release speed ratio is **1.0084**, null ratio 0.9997. At 128/256 only the MoE mode
differs in the timed path; gate changes affect prepare. Other shapes stay within null spread.

| Tokens | Sliding delta | Full delta | Control sliding/full | Null-minus-release sliding/full |
|---|---|---|---|---|
| 128 | -5.1 us (-1.92%) | -5.3 us (-1.52%) | 265.0 / 349.5 us | -3.7 / -5.2 us |
| 256 | -6.5 us (-1.86%) | -1.1 us (-0.21%) | 348.9 / 524.3 us | -4.4 / +0.3 us |

Percentages use paired delta divided by the logged control, not a subtraction of arm medians.
No per-delta confidence interval is reported. Modes at 32 are 47 in both arms, while fronts
differ at 16 through prepare-time selection. This supports the targeted layer gain, not a
new gain at every size. [Raw mode-51 layer comparison](../muse_decode_layer/profiles/benchG51.29fd004fd58e40ef.log).

The first gate-version candidate, SHA256 prefix `c8a843cf`, passes check `a438b82ff2c141c2`
at 2.022x (1.991-2.052), 13/13, with mode 51 selected at 32/128/256. The later `dacd4e0d`
file has the bench result above but no unpatched scored check. It is not released. Board entry
remains `3fbb3ad027e84c6b`, 2.014x. Mode 51 is bitwise at 128/256 in both arms of the
route-PDL checks. All 11 MoE variants pass; route gates pass 8/8; negative controls reject
tile, 1%, 0.1%, row and single-element injections in all 19 gates, and all uninjected
observations pass. [Release/candidate manifest](release-manifest.md), [layer runs](runs-decode-layer.md).

### M1: diagnostic gain within its build, little net release gain

Job `6ce5be0ba29642ea`, source `8f545ae`, runs M1 + mode 51 with a fresh-input gate and a
reader-side proxy fence. Judge 13/13 passes (1.988x, 1.959-2.018), but M1 is blocked on the
control at decode 1 sliding: off replay error 2.7e-3/2.5e-3, on bitwise. M1 changes no launch
at one token, so this does not isolate an M1 effect. Its timing gate did not run.

Stage diagnostic `60cf58b69b75491e` uses the same file through `dl_m1_stage_diag.py`, five
fresh-input pairs, 1,000 replays each for off/off2/on plus 200 eager calls. Attention relative
error is 1e-5 to 2e-4 in all arms; larger errors first appear at routing/MoE. No M1-specific
stale-read signature is found. Report-only timing now runs: at 8 tokens sliding, whole-step
off 105.9 versus o+front 100.5 us; at 16 sliding/full 153.4/229.1 versus 150.0/226.6; at
32, 217.7/360.1 versus 214.7/357.1. These are within-build diagnostic timings, not an accepted
candidate's release gain. Its judge score 1.965x (1.936-1.995) uses M1 off.

Measurement build `fdf75ba5a8124544`, generated with `dl_m1_bench_build.py`, compares M1 + 51
with the mode-47 release, seeds 130001+, eight rounds, 72/72 checks. It scores 2.007x
(1.977-2.038), paired aggregate 1.0040, null 0.9984. At <=32 tokens, candidate-minus-release
sliding/full is d1 +0.2/-0.3, d16 +0.6/+3.0, d32 -0.1/-0.2, verify 1 -1.3/+0.2 us.
Although turning M1 on inside this build saves 5.1 us at 8-token sliding, the build changes
plain projection issue order too, so net release gain is about zero at small sizes. Front
policies also differ at 16. At 128/256, mode 51 explains the targeted timing differences,
not M1. Gate is report-only: this build is not a validated candidate. No isolated E or D
family gain follows. [Layer run log](runs-decode-layer.md).

The same build's 32-token MoE prefetch probe finds off-minus-on +0.22/+0.00 us sliding and
-0.16/-0.48 full across two inputs, comparable to null -0.06/+0.26 and -0.32/-0.67. This
probe gives no measurable prefetch effect; it does not explain the earlier in-layer/component
gap. A prepared --no-pf-probe build and history diagnostic have no new recorded run.

### Validation changed; attention history remains an open lead

The [run log](runs-decode-layer.md#validation-policy-change-fresh-input-replay-gate-user-decision-2026-10-30)
records user-approved fresh-input gates for M1 and MoE/route replay checks, with later changes
to relative-L2 order-1 rejection, control envelopes, element checks and negative controls.
This review reports that recorded decision; it does not change validation or infer permission
for new work. The released file still uses the old gate; the candidate uses the newer one.
Attention `_one_stress`, norm and route side-stream bounds remain unchanged.

A [PTX ordering argument](steering/2026-10-30-m1-ptx-ordering.md) now exists and the M1 build
adds reader-side `fence.proxy.async.global`. The argument states that it has not had independent
review; runtime passing alone does not prove ordering. The former "argument missing" status
is superseded by "written, independently unreviewed".

New open lead in `5fefe98`: fresh-input attention differs from its eager reference by about
1e-4 at decode 1 sliding and 2e-4 at verify 1 sliding, while repeated-input stress is bitwise.
Previous-call dependence is a hypothesis, not a measured causal result or confirmed stale
read. `tools/dl_history_diag.py` is prepared to distinguish noise, previous-call dependence
and separate-case state; it has no run yet. This limits the earlier explanation that all
variation was atomic merge order. Do not clear M1 or reject E on this unresolved evidence.
[Open lead](runs-decode-layer.md#open-lead-2026-10-30-unproven-input-history-dependence-in-the-attention-output).

## Historical review, 19:31: mode 47 layer attribution and mode 51 component gain

Current layer release `3fbb3ad027e84c6b`, source `2cee446`, scores **2.014x (95% interval
1.984-2.045), 13/13 passed**. Its file SHA256 `b9c23bda16bb9d6311f025062e415841c67b16284737172badaa190fdc506a49`
matches main's layer file. It adds mode 47 plus LATE_ARRIVE/o16/unp32 projections to the prior
release, with M1 code absent. Check `c2ba945496ab47b4` was 2.006x (1.976-2.037).
These are cross-run scores, not an isolated gain estimate. [Release manifest](release-manifest.md).

### Mode 47's isolated layer contribution

Job `720079090ee44cc3`, bench source `fa7200c`, compares the scored file with the same file
holding mode 47 out of the picker (SHA256 prefix `84d8507a`). Both arms compile the same
kernels; prepare picks 47 versus 37 at 32/128/256. Two held-out inputs per shape, seeds
110001+, eight seeded-random cold-L2 rounds give 16 paired samples per shape; 72/72 checks
pass. Candidate-minus-held deltas, sliding/full: **128 tokens -4.9/-5.5 us (-1.81%/-1.43%);
256 -5.7/-3.6 us (-1.58%/-0.70%)**. Logged held controls are 270.2/383.7 and 361.2/514.8 us.
At 32 tokens -1.3/-2.6 us is inside the arm spread. This is one paired check, without a
per-delta confidence interval. Null-minus-held at 128 is -3.8/-4.0 and at 256 -4.8/-1.0 us.
[Raw attribution](../muse_decode_layer/profiles/benchInc_47.720079090ee44cc3.log).

The 1.0155 overall ratio is contaminated by a decode-1 arm-position artifact: the held arm is
about 5 us slower even though mode 47 never runs there. Do not report that aggregate as an
isolated mode-47 gain. Full-package comparison `50c4c1328ab742f4`, source `2cee446`, seeds
80001+, eight rounds, 72/72 checks, similarly prints 1.0182 but about 1.009 excluding decode 1.
Its paired deltas at 128 are -5.4/-6.1 us, at 256 -2.1/-3.8; projection share inferred by
subtracting across these two benches is only indicative. [Layer run log](runs-decode-layer.md).

### Mode 51 advances A; tail-split descendants lose

Mode 50 moves sum-of-squares work to a two-pass final pass. Job `8a7051fa926e43e1`, source
`a03ca00`, seeds 23-28, passes correctness but fails its 32-token layer-routing performance
gate (+0.7 us against a +0.5 limit), despite -5.9/-6.6 us at 128/256. It is not shipped.
Mode 51 uses that final pass only at 64+ tokens and retains mode 47 below. This is an A
implementation change retaining C ownership/D claiming, not a new isolated C or D result.
Gate `c6b0e500c3a84c9a`, source `69c06ed`, six held-out seeds 29-34 per routing:

| Routing | 32-token delta vs 47 | 128-token delta | 256-token delta | Seeds faster, 32/128/256 |
|---|---|---|---|---|
| Layer | +0.0 us | -3.9 us (-2.41%) | -3.9 us (-1.71%) | 2/6/5 of 6 |
| Standalone | -1.0 us | -3.8 us | -4.4 us | 5/6/6 of 6 |
| Skewed | -0.1 us | -4.0 us | -7.7 us | 2/6/6 of 6 |

Layer controls are 161.9/228.0 us. All 54 output comparisons pass (maximum 4.6e-5), self-test
3.88e-3, spin flags zero, judge 5/5. Shippable check `a5a2bbd1a18b4b89`, source `6bbb252`,
repeats -3.7/-3.0 us at 128/256, six of six faster, scoring 2.210x. Submission
`bcd2d7a3506a4b7d` passes at 2.202x, below mode 47's 2.216x board entry, which stays.
Those cross-run scores do not negate the paired gain. No mode-51 layer test is recorded.
[MoE runs](runs-moe.md), [raw gate](../muse_moe_w4a16/profiles/bench_g51.c6b0e500c3a84c9a.log).

Modes 52/53 split the last dynamically claimed phase B units into K halves via fixup, an
A/C/D descendant rather than a ready-only queue. Job `209cf29bf91c4ea2`, implementation
recorded in `260d791` (exact submitted source hash not pinned in the row), seeds 35-40:
52 minus 51 at 32/128/256, layer/standalone, is +3.3/+3.5, +5.9/+4.9, +8.5/+5.3 us;
53 is +8.1/+5.2, +7.3/+6.5, +9.2/+7.7. At 128/256 layer controls 159.2/216.0 give
52 +3.71%/+3.94%, 53 +4.59%/+4.26%. Output differences are at most 9.7e-5, spin flags zero,
zero to two of six faster. Tail does not shrink (7.9 versus 7.1 us at 32). Both lose; no
layer port or further gate. This does not reject every split-K design.
[Raw tail test](../muse_moe_w4a16/profiles/bench_tail52.209cf29bf91c4ea2.log).

### Corrections and diagnostic limits

M1 remains blocked and untimed, but the failure is not isolated to M1. Diagnostic
`6372154fcd7a4fa9` on source `6377b5a` finds repeated-input M1-on/off bitwise equal at verify
1 full; the M1-off control itself varies up to 6.7e-4 at 32x1 full. Fresh-input diagnostic
`3e2aa83358dd4f60` finds 217-459/2000 off replays and 288-307/2000 on replays above 1e-4 at
verify 1 full, maximum 1.2e-3 in both arms; no replay exceeds 1e-2 or changes ids. The
source report attributes this to layer run-to-run variation, consistent with atomic reduction
order. The first divergent intermediate and proxy-ordering argument remain unverified.
The 1e-4 bound is unchanged; no timing or safety conclusion follows. Validation-policy changes
require explicit user approval under the recorded test plan; this review changes none.
[Layer diagnosis](steering/2026-10-30-layer-release-attribution.md),
[raw fresh-input test](../muse_decode_layer/profiles/m1fresh.3e2aa83358dd4f60.log).

Public comparisons `86ced49f092a48d8` and `f5ee8632e1f64bf2`, current release against pinned
blob `936fefff`, each pass 72/72 with 16 paired samples per shape; raw null deltas are within 1.1 and 1.8 us,
respectively.
Public advantages at verify 1 sliding/full are 8.7/10.5 then 10.2/10.8 us; decode 32
15.7/10.8 then 12.2/15.9 us. Profiled boundary overlap suggests a mechanism, not measured
recoverable time. Component test `1fbb9fbf524f4ed7`, source `b3509a0` plus tool changes,
layer/standalone seeds 1-6, measures only 1.5/5.9 us public advantage at 32 tokens (47 controls
90.9/132.9 us); the follow-up tail test finds 1.9 us on layer routing. This does not reproduce
the 10-14 us in-layer profile gap. Routing, prefetch and profiling effects remain hypotheses.
sk8w3s loses 19.2 us versus 47 at 32 on layer routing, zero of six faster; do not generalize
its 1/8/16-token win. ABPUB stage columns are invalid without --pubranges; use ABWORK.
[MoE attribution](runs-moe.md), [layer attribution](steering/2026-10-30-layer-release-attribution.md).

Diagnostic `ed3b4fdc76b54263` (source `64cec1f`) crashed after 128-token timing; no 256 result.
Rerun `d47c78bdc130424b` (source `54318bd`) completes, public minus 47 -14.7/-22.6 us at
128/256; diagnostic mode 49 adds 5.8/9.4 us. Its mislabeled ABSKEW columns must be read in
skew_cols order. The estimated 72-74% wait contribution from warpgroup skew is attribution,
not recoverable layer savings. [MoE run log](runs-moe.md).

## Historical review, 18:25: combined layer gain, staged loss and mode 47

The A/E plus projection package now has same-process whole-layer evidence. Check
`aa4276e91c924dad` (bench source `6c6038b`, candidate SHA256 prefix `4ed09046`, incumbent
`a923f11`) reports incumbent/candidate load-time ratio **1.0158**: 1.58% speedup, about
1.56% less elapsed time. It tests 12 shapes, two held-out inputs each (seeds 50001+), eight
cold-L2 graph rounds with seeded random arm order. All 72 reference/routing/cache checks
pass. The identical-candidate null gives about 1.0006 relative to the candidate; per-load
spread is about +/-0.5%, not a formal confidence interval. [Raw layer comparison](../muse_decode_layer/profiles/benchInc.aa4276e91c924dad.log).

At 128 tokens, paired candidate-minus-incumbent is -6.2 us sliding (-2.20%) and -13.3 us
full (-3.56%); at 256 it is -3.5 us (-0.98%) and -22.5 us (-4.22%). Percentages use logged
control means 281.2/373.1 and 356.6/533.2 us. Other loads: decode 1 +0.2/-1.3 us,
decode 16 -1.9/-2.1, decode 32 +0.9/-1.0, verify 1 (8 tokens) -0.7/-0.7, sliding/full.
Those smaller effects are similar to the null spread. The package combines A mode 37/sk8w3s
(retaining C ownership and D dynamic claiming), E DCARRY/PK16/VHOIST, and qkv8/cgemv changes.
It does not isolate a family gain or justify adding component savings.

Release `edd71d3152a946fe` uses the confirmation file from `261a76ac7b364091` (SHA256 prefix
`1355d39f`): **1.982x, 95% interval 1.953-2.012, 13/13 passed**. Confirmation was 1.989x
(1.959-2.019); six candidate checks ranged 1.960-2.012, averaging 1.982. These cross-run scores
support repeatability but the paired comparison supplies the gain claim. FIN8 mode 11 with
eight copies now passes the asserted 2,000-replay tests, replay/eager difference zero in
`3e2b5771d7254fd2`. [Layer run log](runs-decode-layer.md).

### A versus B and the next A candidate

B's retest `9d94ac75f4724b52`, source `54257b5`, uses modes 45 + 46 (variant 301), mode 37's
components split into owned phase A then claimed phase B/final pass. Layer-routing seeds 1-6
lose +30.3/+30.4/+24.4 us at 32/128/256, zero of six faster at each size. At 128/256 this is
+18.4%/+10.7% versus logged mode 37 means 165.5/227.1 us. All arms pass output checks,
spin flags zero; no delta confidence interval is reported. B remains parked under the new
trigger in the status table. [Raw retest](../muse_moe_w4a16/profiles/bench_gw1.9d94ac75f4724b52.log).

Mode 43's next-stage hoist improves timing but does not reduce measured wait cycles. Repeat
`304f17f3e8704992` (source `a4687c8`, layer seeds 7-12) finds mode 47, code groups without the
hoist, faster at 128/256. This scopes the failed hypothesis to the one-group hoist; mode 47
retains the code-group change. Gate `5f381b8b6946492d`, source `20c295e`, seeds 13-18:

| Routing | 32-token delta | 128-token delta | 256-token delta | Seeds faster, 32/128/256 |
|---|---|---|---|---|
| Layer | -1.0 us | -2.3 us (-1.32%) | -3.7 us (-1.65%) | 6/5/6 of 6 |
| Standalone | -2.4 us | -2.8 us | -3.4 us | 6/6/6 of 6 |
| Skewed | -2.0 us | -2.2 us | -2.8 us | 5/6/6 of 6 |

All deltas are paired mode 47 minus 37; layer controls are 173.6/223.8 us at 128/256.
Outputs differ by at most 4.4e-5, self-test 3.89e-3, spin flags zero; 5/5 judge tests pass.
Shippable check `b50351288b504e81`, source `ec565a7`, repeats gains (-2.5/-3.4 us means at
128/256, seeds 17-22). Component release `abe7651dbaa44458` scores 2.216x. Mode 47 also runs
at 128/256 in layer check `e2d1d81447cc48d9`, but that file changes several mechanisms and
scores 1.963x (1.933-1.992); it does not establish an isolated layer gain for 47.
[MoE runs](runs-moe.md), [raw gate](../muse_moe_w4a16/profiles/bench_g47.5f381b8b6946492d.log).

### Invalidated comparisons and separate layer context

- `2cd4e3b292904662` (source `d006531`) passes 13/13 but trips the attention setup test and
  falls back to SGLang, scoring 1.785x. This is not the intended bundle's performance.
  Restoring original qkv rows for the fallback comparison fixes the test path in `3e2b`.
  Later FP32 diagnostics show the fresh-input discrepancy was larger in SGLang's path.
- `135ab2a9f22b49a5` has an invalid incumbent arm: comment stripping removed dispatch markers.
  Discard its apparent 23-240 us gains. `121dff3ab17a4245` never runs its public bench because
  Triton cannot retrieve the arm source. The repaired comparisons are `aa4276` and `d9e07`.
  Oversized/static-rejected attempts produced no job and no measurement.
- Public comparison `d9e07c554f614d83`, same bench source, seeds 60001+, passes 72/72.
  Public/candidate speed ratio is 1.0464, null/candidate 0.9989. Weighted public advantages
  are 9.9 us at verify 1 and 16.5 us at decode 32; candidate leads decode 1 by 1.9 us.
  This replaces the older cross-run gap inference. The public blob is pinned in the run log;
  its identity with the earlier public score's submitted file is not established.
- M1 (SK_PDL + ONE_TRIG), source `6377b5a`, job `e2d1d81447cc48d9`, fails forced-on replay
  correctness at verify 1 full: after 2,000 replays hidden 2.0e-4 and stream 1.6e-4 exceed
  the 1e-4 bound; ids match. M1 is disabled before timing, so no M1 timing gain/loss exists.
  This is an attention-to-projection chain failure, not rejection of E or D. Reopening needs
  an ordering explanation and clean replay checks. [Layer results](runs-decode-layer.md),
  [rejection register](ruled-out.md).

Projection context outside A-E: jobs `01703396ea1a409e`, `3678bfd3ef1c4ce6`,
`2048b59099f3435d` test o16/unp32 geometries (bench sources `dd67954`, `977afb3`, then
resulting geometry change `4d53658`), about 0.3 us gains across
six passes, cold L2, 200 randomized rounds per pass, ABCHECK passes and null within 0.10 us.
`c6e13cc974a24294` repeats against old geometries/public. `d2c9cb6997794859` rejects NACC2
(+0.00 to +0.78 us); `807464e240ec4f6c` finds fence/vote probes within noise and late release
losing +0.26 us on pool32. Fences stay. `f14c880c19a34369` and `1093e22ca2534d7f`, source
`eb374e4`, test LATE_ARRIVE: at 32 tokens qkv saves 0.38-0.48 us, o 0.22-0.26, unpooler
0.16-0.32 over nolate, across two passes each; module-null spread is -0.06 to +0.16.
The mixed M1-off layer check includes these changes but does not isolate their layer gains.
Detailed quartiles, control timings and negative geometry results remain in the
[projection run log](runs.md) and its linked profiles.

## Historical review, 16:27: staged retest and assembled layer bundle

No new scored run or paired timing appears in `dbd99c3..f61d8d9`. The
[rejection register](ruled-out.md) now parks the staged design with an explicit revisit trigger:
cheaper phase B/setup or a changed overlap tail. Modes 23, 26 and 28/37 landed after the old
staged comparison. That trigger is met, so repeat A versus B at 128/256 with current components
in one process. Preserve the old +27.9/+24.2 us losses as historical measurements; they do not
establish the result against today's A. No new winning B implementation has been measured.

The [layer breakdown](steering/2026-10-29-v1-d32-breakdown.md) reports an assembled bundle at
`b694b4a`: mode 37 + sk8w3s (A, retaining C/D mechanisms), DCARRY/PK16/VHOIST (E), plus qkv8
and single-token cgemv projection changes outside A-E. It is unjudged. Its branch log and
source were inspected locally; FIN8 expected-plan assertions and fresh-qkv replay checks are
implemented there, but no completed runtime result closes that coverage gap. Whole-layer
score remains 1.964x from `9b921febc7ad4140`. Do not add standalone savings into a layer gain.

Separate layer context: `3da50e5` adds default-off SK_PDL projection-weight prefill and
ONE_TRIG attention launch-dependent signaling. This is a proposed attention-to-projection
chain, not D's ready-job scheduler or a measured E gain. Recorded compile checks show the
disabled paths SASS-identical (58/58 projection kernels, 15/15 attention kernels); enabled
builds have no recorded spills. The host launch path was not compiled in that gate, and
runtime correctness, replay stress and timing remain untested. No job ID or timing delta
exists for this proposal. [Compile evidence and test plan](steering/2026-10-29-v1-d32-breakdown.md#6-built-sk_pdl-and-one_trig-compile-gated-off-by-default).

The new score-weighted hotspot tool in `f61d8d9` ranks existing profile evidence. Its rankings
are analysis, not fresh benchmark measurements. The layer breakdown's bundle projections
and remaining-gap estimates are also hypotheses or cross-run arithmetic, not isolated family gains.

## Historical review, 14:25: family E improves again

PK16 changes probability packing and VHOIST carries transpose addresses/descriptors per
segment. The combined attention candidate improves full verification beyond DCARRY.
Job `025ccaf9c57a4ba2`, switches introduced in `a50fc0c`, records cold same-process control
116.26 / 204.22 us versus candidate 110.91 / 194.59 at 128/256 tokens. That is **-5.35 / -9.63
us**, or **-4.6% / -4.7%**. Warm deltas are -7.52 / -13.88 us. ABCHECKSUM is 19 passed and
zero failed for every arm. These full-attention cases use batch 16/32, q_len 8, context
80,000 and TP8; MoE routing does not apply. The run-log table supplies no per-delta
confidence interval or independent repeat count. [Raw bench](../muse_attention_decode/profiles/bench_pk.025ccaf9c57a4ba2.log).

Candidate-only run `569b233dfe08414f`, source `c37d381`, passes 16/16 at 2.449x,
95% interval 2.413-2.502. Full verify times 103.65 / 187.67 us are 7.1 / 10.7 us below the
mean of the two previous DCARRY checks. Those are cross-run observations. Repeat
`e9085c86d89449c2`, source `61e3fd4`, uses the same candidate as its judged path and passes
16/16 at 2.504x, interval 2.467-2.542, with 100.83 / 182.24 us. Its machine is faster on
unchanged rows too; the extra headline-score increase is not another code improvement.
[Attention report](runs-attention.md).

Family E's pending package is now **DCARRY + PK16 + VHOIST**. No new layer result exists in
the reviewed records; layer score stays 1.964x. A-D standings and the mode 37/sk8w3s layer
gaps remain unchanged. The no-fence timing probe in the repeat is diagnostic only and is not
an eligible winning candidate; the production fence stays.

The [community audit](steering/2026-10-29-community-audit.md) identifies two validation gaps:
the intended q_len8 FIN8 stress selected mode 15 instead of mode 11 with eight copies, and
the whole-layer comparison lacks enforced per-seed acceptance, reliable arm-order balancing
and independent mutable cache state. These are open coverage/measurement gaps, not recorded
correctness failures of the incumbent. They should be resolved before relying on the next
combined layer comparison. Existing 13/13 layer judge passes remain recorded.

Decode-32 full attention's apparent +8.7 us regression in the older DCARRY comparison is
now attributed to SGLang's bimodal baseline statistic and machine variation on unchanged mode
49. Preserve the observed cross-run statistic but do not label it a demonstrated candidate
regression. [Rejection register](ruled-out.md), [attention analysis](runs-attention.md).

## Historical review, 13:27: projection context

No new A-E or whole-layer result appeared since the 12:24 review. The three family candidates
awaiting layer tests remain mode 37, sk8w3s and ONE_DCARRY. New projection results belong
outside A-E; they do not change the winning-family badges.

Job `f6c80b4ba11048f3`, source `a35709c`, compares the new single-token cgemv with the
pre-change file `b79a627` in one process. Two passes use different inputs, each with 300
randomly ordered rounds and cold L2. New minus control, pass 0 / pass 1:

| Single-token projection | Paired change |
|---|---:|
| qkv_gate | -0.54 / -0.45 us |
| pooler | -0.86 / -0.96 us |
| unpooler slice | -0.35 / -0.22 us |
| o | +0.08 / -0.11 us, inconclusive |

ABCHECK passes on each reported shape/pass. The same-code null stays within 0.13 us.
The implementation combines row-major planes, a single-load escape start and a larger
register allowance for two chunks per thread. It is a combined projection change, not a
MoE ownership or scheduling experiment. Absolute paired control means and intervals are
not quoted here, so no elapsed-time percentage is assigned.

The candidate-only submission `81254e8aed794eab` records **1.444x, 95% interval 1.422-1.466**.
This is the projection problem's score, not the layer's. Layer integration remains pending;
current layer score stays 1.964x. The earlier qkv8 projection port is also still pending in
the port instructions. [Projection report](runs.md),
[paired raw log](../muse_bf16_proj/profiles/bench.f6c80b4ba11048f3.log),
[candidate result](../muse_bf16_proj/profiles/submission.81254e8aed794eab.json).

The proposed 128-token wide concat front was declined without a new run. Its estimated
24.4 us best-case time exceeds cuBLAS's recorded 24.0 us, before accounting for a second CTA
wave. This is an analytical rejection of that geometry, not a measured candidate loss and
not evidence against staged MoE B. [Rejection register](ruled-out.md).

## Historical review, 11:20: corrected MoE wait measurements

Job `ca43908f02874886`, code `3f67a45`, corrects a diagnostic defect. The timer previously
started after the first potentially blocking try-wait, which could hide waiting entirely.
The earlier claim that fat-path consumers never wait is invalidated. Production instructions
are unchanged, so this is corrected attribution, not a new production speedup or regression.

| Diagnostic observation | 128 tokens | 256 tokens |
|---|---:|---:|
| Phase A weights + activation waits, per consumer warp | 15.73 us | 23.26 us |
| Phase A waits / stage-loop time | 19.6% | 20.2% |
| Phase B waits / stage-loop time | 20.9% | 20.8% |
| Instrumented mode 41 minus production mode 37 | +7.0 us, +4.3% | +6.1 us, +2.7% |

Six paired layer-routing seeds; production controls 164.6 / 227.2 us. The run passes 5/5
judge tests. Counter overhead is material, and per-warp waits cannot be summed into layer
savings. The producers still spend about 70% of phase A waiting for free ring slots.
[Raw log](../muse_moe_w4a16/profiles/bench_fw1.ca43908f02874886.log),
[measurement report](runs-moe.md#fat-path-waits-with-the-first-try-timed-2026-10-29).

A/C/dynamic-claiming production wins remain supported. B remains a tested loss; no new C,
ready-only D, E or whole-layer candidate result appeared in the reviewed changes. The same
three candidates await layer evidence. A proposed per-group load/hoist experiment remains
unbuilt in the [steering report](steering/2026-10-29-fat-moe-waits.md); its predicted savings
are hypotheses. This correction motivates a test, not a new winning-family badge.

## Large-token comparison ledger

| Family / candidate | Control | 128 tokens | 256 tokens | Evidence and scope |
|---|---|---:|---:|---|
| B: staged 19 then 21 | Mode 16 | +27.9 us, +11.2% | +24.2 us, +8.4% | `e221858b020c4dee`, six same-process seeds, complete standalone MoE. Controls 249.4 / 287.3 us. Correctness passed. [Staged gate](runs-moe.md#staged-gemm1-gate-2026-10-29). |
| C inside A: mode 20 | Mode 16 | -2.1 us, -0.8% | -3.8 us, -1.3% | Same job and controls. Other paired checks support the sign. This isolates owned phase A within a fused launch. [MoE log](runs-moe.md). |
| D-related inside A+C: mode 23 | Mode 20 | -16.2 us, -6.6% | -16.1 us, -5.7% | `a9dbec1d98334943`, source `10375cb`, six paired normal seeds; controls 246.2 / 283.7 us. Self-tests pass, spin flags zero. [Dynamic phase B](runs-moe.md#phase-b-claimed-at-run-time-mode-23-2026-10-29). |
| Mode 23 held-out repeat | Mode 20 | -22.9 us, -8.7% | -18.7 us, -6.1% | `36812632dae242ba`, source `ec219f0`, six held-out normal seeds; controls 263.3 / 304.6 us. Skewed routing also improves, -4.8 / -22.9 us. [MoE log](runs-moe.md). |
| A+C+dynamic claiming: mode 26 routing/gather changes | Mode 23 | -5.8 us | -9.5 us | `817eab6401d54a84`, source `ba9a457`, six held-out seeds; self-tests pass. Skewed -4.2 / -7.7 us. [MoE log](runs-moe.md). |
| Same combined design: mode 28 consumer/ring and sorting changes | Mode 26 | -10.6 / -11.6 us | -11.6 / -12.5 us | Two six-seed checks `f928f8a802014e4f` and `13fe68e7ca2e423f`, source `ec62b5a`; correctness passes. [Consumer and ring investigation](runs-moe.md#consumer-stage-and-the-ring-slot-releases-2026-10-29). |
| Same combined design: mode 37 phase A epilogue | Mode 28 | -3.2 / -4.3 us | -4.9 / -5.3 us | `987d024620e44eb7` and `6307d0c91c8d4ada`, source `96f2c43`, held-out seeds 17-22. Correctness passes. Reciprocal and publication-order changes, not a new ownership experiment. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29). |
| E: ONE_DCARRY, cold paired bench | Same attention without DCARRY | -3.65 us, -3.0% | -7.33 us, -3.5% | `4ff339332fef4aff`, full verify only, controls 120.03 / 210.59 us; all arms pass checksum checks and judge 16/16. [Attention bench](runs-attention.md). |
| E: ONE_DCARRY, judged repeats | ONE_PFAST pair without DCARRY | -3.6 us, -3.2% | -7.9 us, -3.8% | Cross-run mean versus mean, `f64e8680b3e441be` / `bf49eceb15a64f81`, source `0d76acc` / `503c0f4`; 16/16 both. Component evidence, not layer savings. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, cold paired bench | DCARRY control | -5.35 us, -4.6% | -9.63 us, -4.7% | `025ccaf9c57a4ba2`; controls 116.26 / 204.22 us, candidate 110.91 / 194.59. ABCHECKSUM 19/0 for every arm. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, candidate-only judged | Mean of previous DCARRY pair | -7.1 us, -6.4% | -10.7 us, -5.4% | `569b233dfe08414f`, source `c37d381`; 16/16 passed, cross-run component observation. Same-code repeat `e9085c86d89449c2` also passes but has machine-speed variation. [Attention log](runs-attention.md). |
| B: staged 45 + 46 retest | Mode 37 | +30.4 us, +18.4% | +24.4 us, +10.7% | `9d94ac75f4724b52`, six layer-routing seeds, outputs pass; zero faster. [MoE log](runs-moe.md). |
| A+C+D: mode 47 code groups | Mode 37 | -2.3 us, -1.32% | -3.7 us, -1.65% | `5f381b8b6946492d`, held-out layer seeds 13-18, 5/6 and 6/6 faster; checks pass. [MoE log](runs-moe.md). |
| A/E + projection: whole-layer bundle | Incumbent a923f11 | Sliding -6.2 us (-2.20%); full -13.3 us (-3.56%) | Sliding -3.5 us (-0.98%); full -22.5 us (-4.22%) | `aa4276e91c924dad`, 2 inputs per shape, 8 rounds, 72 checks pass; combined gain only. [Layer log](runs-decode-layer.md). |
| A: mode 47, historical three-arm picker observation (arm-bias caveat) | Same file with 47 held out | Sliding -4.9 us (-1.81%); full -5.5 us (-1.43%) | Sliding -5.7 us (-1.58%); full -3.6 us (-0.70%) | `720079090ee44cc3`, 16 paired samples/shape, 72 checks pass; overall ratio invalidated by d1 arm artifact. [Layer log](runs-decode-layer.md). |
| A: mode 51 final pass | Mode 47 | -3.9 us, -2.41% | -3.9 us, -1.71% | `c6b0e500c3a84c9a`, layer seeds 29-34, 6/6 and 5/6 faster; no layer test. [MoE log](runs-moe.md). |
| A/C/D: mode 52 split tail | Mode 51 | +5.9 us, +3.71% | +8.5 us, +3.94% | `209cf29bf91c4ea2`, layer seeds 35-40, output checks pass. Mode 53 also loses +7.3/+9.2 us. [MoE log](runs-moe.md). |
| A: mode 51, historical three-arm observation (gain unestablished) | Clean mode-47 release | Sliding -5.1 us (-1.92%); full -5.3 us (-1.52%) | Sliding -6.5 us (-1.86%); full -1.1 us (-0.21%) | `29fd004fd58e40ef`, 16 paired samples/shape, 72 checks pass, 1.0084 aggregate with null 0.9997. Candidate not released. [Layer log](runs-decode-layer.md). |
| A: mode 51 four-arm layer | Clean mode-47 release, two copies each | Weighted ratio 1.0080; unresolved | Weighted ratio 1.0107; unresolved | `2dc566298be1498f`, overall 1.0057 fails null rule; prepare policies differ by copy. [Layer log](runs-decode-layer.md). |
| E: XPIPE=2, P.V-first cross-tile pipeline | Attention mode 51, XPIPE=0 | +3.90 us, +3.44% | +0.80 us, +0.42% | `5575e7d15d3e49a8`, kernel `86f2bec`, two held-out inputs, 120 rounds, cold pooled median subtraction; controls 113.41/192.45. 16/16 and ABCHECKSUM 19/0. Warm improves; cold gate fails. [Attention log](runs-attention.md). |
| E: mode 115, one CTA/SM | Attention mode 51 | +27.52 us, +24.96% | +57.67 us, +30.80% | `318ae6fc3c3746ce`, source `682aeaa`, two inputs, cold pooled medians; controls 110.27/187.26, nulls +0.06/+0.42. 16/16 and real-arm ABCHECKSUM 19/0. [Attention log](runs-attention.md). |
| E: mode 115 + XPIPE=1 | Attention mode 51 | +28.10 us, +25.48% | +52.61 us, +28.09% | Same job, controls and correctness; overlap helps within mode 115 at 256 but cannot recover residency loss. [Attention log](runs-attention.md). |
| E: PRQ31 per-request tile pipeline | HMRG-on attention, PRQ off | Sliding -0.48 us, -3.02%; full unchanged code | Sliding -0.96 us, -4.59%; full unchanged code | `409996a7214f411a`, `8a6c8bb`, two inputs, 120 rounds, controls 15.87/20.93 us, same-state null +0.03/+0.09 us; 16/16 and 19/0 arm checks. Combined five-bit design. [Attention log](runs-attention.md). |
| Alternative to C inside A: mode 72 stream-K phase A | Mode 51 owned phase A | +12.6 us, +7.59% | +6.6 us, +3.00% | `098949b28fe84bd1`, `86a633e`, six seeds 83-88, layer-like routing, controls 166.1/220.3 us; 0/6 faster, receipt 5/5, first-seed L2 checks 1.5e-4 to 2.4e-4 across cells. Standalone +7.1/+9.2 us. No gate run. [MoE log](runs-moe.md). |
| Alternative to C inside A: mode 74 designated finisher | Mode 51 owned phase A | +4.6 us, +2.82% | +2.0 us, +0.91% | `e02f60fec94f45cb`, `4187b8b`, six seeds 107-112, layer-like routing, controls 163.2/219.9 us; 0/6 and 1/6 faster. First-seed L2 checks 1.3e-4 to 3.4e-4; spin flags zero. Standalone +7.4/+4.1 us. No gate. [MoE log](runs-moe.md). |
| A + separate M1, stable whole layer | Stable release_s | Weighted ratio 1.0163; paired gain | Weighted ratio 0.9993; inconclusive | `3aee22ba4a404c8a`, two inputs, eight rounds, 24/24 checks per arm; pooled 1.0052 beyond nulls. Combined attribution only. [Layer log](runs-decode-layer.md). |
| F-ATTN controlled donor comparison | Release a70192d | Full -2.13 us (-1.88%); sliding inconclusive | Sliding -0.605 us (-3.03%); full inconclusive | `0519d0b15e9c4583`, three inputs,150 rounds, four independent prepares; numerical checks pass. Measurement only. [Attention log](runs-attention.md). |
| F-MOE exact donor, first comparison | Release ec565a7, standalone routing | D1-R1 -14.73 us (-7.46%) | D1-R1 -27.00 us (-10.55%) | `40b8c798289a4258`,12 seeds,20 cold replays; all 12 faster. Missing donor proxy fences, repair cost unmeasured. [MoE log](runs-moe.md). |
| F-LAYER complete donor comparison | Released whole layer | Weighted ratio 1.0734 / 1.0664 | Weighted ratio 1.0792 / 1.0893 | `20abe6291eb04ed6` / `00607fc6f8c84c77`, two inputs,12 rounds; checks pass, eligibility caveats remain. [Layer log](runs-decode-layer.md). |
| A mode78 gate, layer-like routing | Mode51 | -0.6 us (-0.37%) | -0.6 us (-0.27%) | `c8fc5f669fa34d06`,95c9848,six seeds 131-136; exact outputs 128/256. Fails 128 standalone +0.6us limit; not kept. [MoE log](runs-moe.md). |

Percentages use the logged paired delta divided by the logged control mean, except the explicit pooled-median subtraction rows.  Paired summaries use logged paired deltas rather than subtraction of separately summarized
arm timings. Layer deltas are medians of per-round differences; these can differ by several
microseconds from differences of arm medians. Rows without a comparable control mean report
microseconds only. Seed counts are reported instead of inventing confidence intervals for the
paired differences.
| A/C/D mode 79 both-phase stream-K | Mode 51, layer routing | +19.4 us (+11.67%) | +21.4 us (+9.92%) | `2a4fb39964aa448c`, six seeds, zero wins; one 32-token numerical gate miss. [MoE log](runs-moe.md). |
| Combined M1 + 51 policy, recovered M-B receipt | release_s | Sliding -5.5 us (-2.24%); full -7.2 us (-1.96%) | Sliding -5.3 us (-1.56%); full -6.3 us (-1.23%) | `c8d1e57261dd4901`, representative cand-inc; two inputs, eight rounds, checks pass. Added slice rejected. [Layer log](runs-decode-layer.md). |
| Projection C1 layer port, separate context | layer_m1rt51s | Sliding -1.3 us (-0.49%); full -0.4 us (-0.11%) | Sliding -0.1 us (-0.03%); full -0.4 us (-0.08%) | `b70cfa48c0264d20`, representative cand-inc; pooled 1.0007 inconclusive. [Layer log](runs-decode-layer.md). |
| A mode 86 earlier x-slot release | Mode 51, layer routing | -0.3 us (-0.19%) | -0.6 us (-0.28%) | `a00b49f8372042a9`, six seeds, 20 cold replays; outputs identical, performance gate fails. [MoE log](runs-moe.md). |
| E PBULK token coverage | Unchanged q_len-8 kernels | No isolated effect; unchanged code | No isolated effect; unchanged code | `1c861d3944f149a0`, two inputs, 120 rounds. Small PBULK1 gains and PBULK2 losses are at decode16/32, not 128/256 tokens. [Attention log](runs-attention.md). |
| A ring87 four-x-slot tradeoff | Mode51, layer routing | +2.2 us (+1.39%) | +2.9 us (+1.37%) | 39804be1a59c429c;4seeds,20cold replays,correctness passes, gate fails. [Remote MoE log](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md). |
| B masked grouped library | Owned51, layer seeds291/292 | +180.16/+180.00 us (+104.22%/+104.11%) | +234.88/+231.78 us (+110.66%/+108.25%) | ae8993f4e3e1435c;20paired rounds, math passes, no promotion. [Remote MoE log](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md). |
| E HMRG/PRQ layer port first comparison | Prior complete layer | Sliding -0.0 us; full -1.2 us (-0.33%) | Sliding -0.8 us (-0.23%); full +0.3 us (+0.06%) | c0402e4c92bc4b20; representative paired cand-inc; pooled1.0042 wins, prepared M1 differs. [Remote layer log](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-decode-layer.md). |
| E port independent repeat | Prior complete layer | Sliding -1.4 us (-0.50%); full -1.8 us (-0.50%) | Sliding -0.7 us (-0.19%); full +1.5 us (+0.28%) | 9414182118414889; pooled1.0015 unresolved, all M1 off. [Remote layer log](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-decode-layer.md). |
| Projection C3 router confirmation, separate context | Owned router | FV -0.45/-0.46 us (-6.34%/-6.48%) | Order+FV -0.69 us (-7.41%) | 7d2fe593fdfa4720;100cold rounds, independent nulls,checks pass. Not family C or a layer gain. [Remote projection log](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs.md). |
| G stage retirement, repaired SC6 | Same-process repaired control | -7.464 to -11.808 us (-3.70% to -5.88%) | -13.432 to -18.280 us (-5.91% to -7.56%) | 63f5a2a8ed2341d1;3heldouts, four private placements, six-load gates pass. [Remote evidence](https://github.com/yelgabo/okc/blob/af3b550/muse_moe_w4a16/profiles/stage-retire.63f5a2a8ed2341d1.analysis.json). |
| Retained G layer versus owned combination | Independent owned cb71 public run | Weighted -16.034 us (-5.62%) | Weighted -26.968 us (-6.98%) | 607eb52bdaf24160; cross-run observation with different preparation/routing. [Source cells](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/evidence/2026-11-01-g-public/stage-shapes.md). |
| Projection QKV depth2, separate context | Same-run production | +0.512 to +0.640 us (+2.53% to +3.17%) | Treatment absent | 4542b2b04cb344ec;2inputs,100cold rounds, correctness/resources pass, timing gate fails. [Receipt](https://github.com/yelgabo/okc/blob/af3b550/muse_bf16_proj/profiles/qkv_depth.4542b2b04cb344ec.json). |
| Owned global sparse-XOR projection, coverage context | Archived C2, independent run | Sum of five adjusted medians +1.496 µs (+2.23%) | Sum +1.654 µs (+2.01%) | `2b537518091248f1`. Both sizes use C2 paths. Only the ten target cells at 8/16/32 exercised sparse XOR. [Cell records](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/evidence/2026-11-01-sk-xor-release-terminal/rows30.csv). |
| Private VP2 attention, coverage context | Matched BF16-output owned controls | Existing owned path | Existing owned path | `c79ddddb840a4fb7`. Only eight-token sliding attention exercised VP2. [Full-score result](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-attention-vp2-bf16-terminal.md). |
| Stage-local XOR coverage | Archived C2, sum of five adjusted medians | +1.140 µs (+1.70%) | +1.117 µs (+1.35%) | Cross-run observations on unchanged paths. Only the 8/16/32 targets used sparse XOR. [Cell records](https://github.com/yelgabo/okc/blob/08d336a/docs/steering/evidence/2026-11-01-projection-stage-xor-terminal/rows30.csv). |
| Retained G versus upstream G0 | Earlier independent official runs | +10.676 µs (+4.13%) | +8.500 µs (+2.42%) | Weighted verify latency. Cross-run reconstruction of existing jobs, no isolated repair timing. [Source records](https://github.com/yelgabo/okc/blob/e1adc0e/docs/steering/evidence/2026-11-01-g-integration-gap/cross-job-gap.json). |
| G224 gather schedule | Historical public G1b4 | Weighted verify latency +14.617% | +20.564% | `e1271ad5e5e34b24`, independent run, all 13 tests passed. [Source](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-02-captain-g224-terminal-decision.md). |
| Fully timed G SiLU table | Historical public G1b4 | 275.449 vs 269.347 µs, +6.102 µs (+2.27%) | 358.496 vs 359.297 µs, −0.802 µs (−0.22%) | `dc62cff972934f54`, one private check, affected loads, unmatched seeds and nodes. [Reconstruction](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-02-captain-timed-silu-terminal-decision.md). |
| G short serial policy, unchanged long paths | Fresh G1b4, separate run | +8.118 µs (+3.10%) | +10.431 µs (+2.97%) | `e4c8f00532b74d15` vs `cc474883117947c0`. Different seeds. Only 1/8-token cells exercised the treatment. [Exact comparison](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-serial-terminal-decision.md). |
| G pre-gamma expert transport, standalone MoE | G incumbent controls, same process | +2.785 µs (+1.93%) | +2.779 µs (+1.41%) | `56e0c8b5634545e6`. Descriptive duplicate-placement means across three normal/skewed banks. [Source means](https://github.com/yelgabo/okc/blob/db7800a/docs/steering/evidence/2026-11-02-worker-moe-component/descriptive-load-summary.json). |
| G one-token expert split, unchanged large-token paths | Fused G control in same process | +0.339 µs (+0.23%) | +0.048 µs (+0.02%) | `28248eb5e5854041`. Means over duplicate placements and six routing cases. Only T1 changed. [Raw medians](https://github.com/yelgabo/okc/blob/1786dbf/docs/steering/evidence/2026-11-02-worker-t1-moe-check/frozen-reader-result.json). |


## Small-token results and rejected changes

For A's new `sk8w3s`, job `75bca9775f6f4b0b`, source `03887be`, uses six seeds per size and
routing distribution. This measures the MoE component using layer-like or standalone routing;
"layer-like" does not mean a whole-layer benchmark.

| Tokens | Layer-like control sk8w4s | Change with sk8w3s | Standalone-routing change |
|---|---:|---:|---:|
| 1 | 35.6 us | -0.9 us, -2.5% | -0.6 us |
| 8 | 66.2 us | -2.4 us, -3.6% | -1.8 us |
| 16 | 84.8 us | -3.1 us, -3.7% | -2.9 us |

All 36 paired seeds favor three slots. Every arm matches the control within 2.1e-5 and spin
flags are zero. The candidate entered the standalone variant list at `d0c062b`.
[Evidence and port status](runs-moe.md#skinny-weight-stream-against-the-public-kernel-2026-10-29).

Losses matter as much as wins:

- A's unhinted skinny weight copies lose 1.2-3.3 us across the six size/routing rows in that
  job. Early refill with three slots also loses in every row. Scale-decode and helper-handoff
  changes in `e937013292b24906` were inconclusive and not adopted.
- Phase A mode 35 reduces spills but loses +3.0 / +4.7 us at 128/256 versus mode 28.
  Mode 39's paired stores lose +1.0 / +0.2 us versus mode 37 in `473d87d03de94e96`.
  A shorter epilogue alone does not establish a faster kernel. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29).
- E's run-time pair weighting loses +9.0 / +12.0 us at full verify 256 for weights 1.40 / 1.50
  in `4ff339332fef4aff`. The arrival counter fails to predict scheduler priority. This is an
  attention scheduling loss, not evidence against MoE family D. [Attention log](runs-attention.md).

## Whole-layer outcome and attribution

Four-cell public job `1fad70bbe3854cbc` returned an infrastructure error and a refund. It supplies no H200 qualification at 128/256 tokens. The matched common-control trial `c5e4f89672f743bb` closed the short-counter treatment. Its official 2.112499× score measures the repaired control. The matched OFF/ON timings determine the counter decision. [Terminal analysis](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-01-rmw-legal-fixture-terminal.md)

Retained Family G job `607eb52bdaf24160` scores2.126082676x, CI2.094191436-2.157973917,
from `g1b4_stage_release.py`, SHA256 ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab.
Its official13 tests and56 private replays pass. Matched standalone stage-retirement gains
belong to the repaired G backend. The whole-layer score is a separate public-run result.
The owned Lt-plus-boundary combination has two-seed matched six-load gains of0.34-0.36%,
with common preparation and private arm tensors. Both evidence scopes remain visible.

## Decision and next evidence needed

The latest complete-engine review closes both reviewed proposals at the source stage. Reopening grouped-contiguous Humming requires an exact legal workload and credible net whole-layer opportunity. The canonical attention continuity source is frozen pending stronger opportunity evidence. [Decisions](https://github.com/yelgabo/okc/blob/ba0e1ff/docs/steering/2026-11-02-captain-architecture-gate-reconciliation.md).

The one-token expert split also failed on every targeted routing case. The paired component gate closes this specific three-launch candidate. [Terminal decision](https://github.com/yelgabo/okc/blob/1786dbf/docs/steering/2026-11-02-worker-t1-moe-terminal.md).

The pre-gamma component missed the registered 1% gate in all 24 six-load comparisons. The exact implementation is parked before layer integration. [Decision](https://github.com/yelgabo/okc/blob/db7800a/docs/steering/2026-11-02-worker-moe-component-terminal.md).

A remains favored over tested B schedules at128/256. The latest owned position-map,
locality, decoder and direct/cached band variants fail their registered gates. C's tested
ownership implementations retain their scoped outcomes. D's complete ready-only scheduler
remains untested. Further scheduling work needs a distinct mechanism and measured opportunity.

The G lineage has an approved attributed-fork policy, while F retains its alygg77 history.
The retained G1b4 layer includes stage retirement. Its matched gain was measured in standalone MoE. The matched short-counter trial is complete. Both outgoing dependency tests stopped in setup. G224 and timed SiLU are parked. The short serial candidate was parked after fresh control and candidate runs. The PR check failed host compilation and private retry awaits quota. [Terminal evidence](https://github.com/yelgabo/okc/blob/049b223/docs/steering/evidence/2026-11-02-worker-pr-target/target/terminal-status.json). [Host qualification and allocation](https://github.com/yelgabo/okc/blob/218a048/docs/steering/2026-11-02-captain-pr-host-gate.md) · [Serial decision](https://github.com/yelgabo/okc/blob/fcf5452/docs/steering/2026-11-02-captain-serial-terminal-decision.md). [Latest decision](https://github.com/yelgabo/okc/blob/57930d0/docs/steering/2026-11-02-captain-timed-silu-terminal-decision.md)

Historical owned attention requires the documented intermediate BF16 repair before reuse.
The new QK2 and wait studies use repaired controls. The attention and G detail observers
fail distortion gates, so their wait sums support descriptive observations only. Projection
QKV depth/push losses leave the wider comparator advantage open under its precision caveats.
Prefill remains deferred. The final recorded campaign goal remains unmet.

[Final outcomes and reset gates](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-0630-measurement-outcome.md),
[Family G qualification](https://github.com/yelgabo/okc/blob/af3b550/docs/steering/2026-11-01-family-g-qualification.md).

## Review history and checkpoint

- 2026-11-02 12:48 Vancouver time. Reviewed `1786dbf..ba0e1ff` from local `4188c23`, including ruled-out changes, the canonical attention source and both expert-engine reviews. No new run-ledger or profile changes. All 18 untracked local receipts are already logged.

- 2026-11-02 11:53 Vancouver time. Reviewed `db7800a..1786dbf` from local `5489860`. Verified terminal MoE job `28248eb5e5854041` and the reference-source correction. The new confirmation of the pre-gamma loss refers to its previously counted receipt. All 18 untracked local receipts already appear in the ledgers.

- 2026-11-02 10:44 Vancouver time. Reviewed `ca5cf8f..db7800a` from local `5105841`. Verified terminal MoE job `56e0c8b5634545e6`, source placement means and closure. All 18 untracked local receipts already appear in the run logs.

- 2026-11-02 09:46 Vancouver time. Inspected `049b223..ca5cf8f` from local `b55ac28`. The new component adapter has CPU evidence and a registered measurement method. No new GPU job or timing. All 18 untracked local receipts already appear in the run logs.

- 2026-11-02 08:49 Vancouver time. Reviewed `218a048..049b223` from local `282f081`, including the terminal PR receipt, ruled-out entries, expert prototype registrations and 15:30 UTC quota checkpoint. 18 local untracked receipts already appear in the run ledgers.

- 2026-11-02 07:49 Vancouver time. Reviewed local `665b9c9` and remote `fcf5452..218a048`, including the ruled-out entry, source repair and conditional allocation. No new benchmark receipts. All 18 untracked local receipts already appear in the ledgers.

- 2026-11-02 06:47 Vancouver time. Reviewed `8c0bfba..fcf5452` from checkout `977ce4a`, including both terminal receipts, six weighted loads and the 13:26 UTC status checkpoint. 18 untracked local receipts already appear in the run ledgers.

- 2026-11-02 05:28 Vancouver time. Reviewed `57930d0..8c0bfba` from local `94fcf44`. Reconciled the canonical SiLU archive with its existing job ID and verified the separate queued control receipt. 18 local untracked receipts already appear in the run ledgers.

- 2026-11-02 04:32 Vancouver time. Reviewed local `54659c4` and remote `d79089b..57930d0` (158 commits), including the November 2 11:30 UTC campaign checkpoint. Parsed eight new remote profile JSON files and 18 previously logged local receipts. Terminal, invalidated and unmatched outcomes are recorded above with their owning jobs.

- 2026-11-01 14:49 Vancouver time. Local `545513d`, remote `e1adc0e..d79089b`. The 21:38 UTC service checkpoint retains unknown status for `1fad70bbe3854cbc`. Two prepared profile manifests and the compiler closure add no H200 timing. 18 untracked local receipts are already logged.

- 2026-11-01 13:47 Vancouver time. Reviewed `08d336a..e1adc0e`, including the dispatch and final maintenance status for `1fad70bbe3854cbc`. 18 untracked local receipts already appear in the ledgers. The G0 comparison reuses two historical receipts.

- 2026-11-01 12:45 MST (America/Vancouver). Local checkout `1f0b7dd`. Reviewed `fa8173a..08d336a`, including terminal projection job `dfcfdd01d6f04525`, the progress audit and the 19:38 UTC reset evidence. Historical untracked receipts were already logged.

- 2026-11-01 11:19 MST (America/Vancouver). Local checkout `5e42a4b`, remote evidence `af3b550..fa8173a`. Reviewed 14 remote profile JSON files, run records, ruled-out entries and source audits. All 18 untracked local receipts already appear in the ledgers. New completed jobs `c79ddddb840a4fb7` and `2b537518091248f1` are recorded separately from the pending dispatch `dfcfdd01d6f04525`.

- 2026-11-01 08:04 MST: reviewed remote `6e078b5..af3b550`,187 commits, run ledgers,
  profiles, ruled-out additions and final06:30PST handoff. Parsed135 new remote JSON
  profile files and18 already logged local receipts. Exact jobs are enumerated above.
  Added registered Family G and propagated retained score, precision findings, new losses
  and observer invalidations. Source review used remote objects with snapshot-pinned links.

- 2026-11-01 01:16 PDT: fetched `origin/main` at `6e078b5`, 148 commits beyond local
  `d6337e1`; inspected changes since evidence snapshot `8f2c89b`. Reviewed new run records,
  ruled-out entries, profiles and steering qualifications. Parsed93 added remote profile
  JSON files and18 already logged local untracked receipts. New jobs and source revisions
  are enumerated in the latest section. Remote links pin6e078b5; checkout not merged.
  No benchmarks, submissions, pushes or kernel edits performed by this review.

- 2026-10-31 07:41 PDT: checkout remains `d6337e1`; evidence snapshot remains `8f2c89b`.
  Fetched origin/main has no commits ahead. Run logs, tracked profiles and ruled-out entries
  are unchanged. All 18 untracked JSON receipts parse and are already logged. No new
  measurements or status changes. Review metadata refreshed locally; no commit or push.

- 2026-10-31 06:36 PDT: checkout remains `d6337e1`; evidence snapshot remains `8f2c89b`.
  Fetched origin/main has no commits ahead. Run logs, tracked profiles and ruled-out entries
  are unchanged. All 18 untracked JSON receipts parse and are already logged. No new
  measurements or status changes. Review metadata refreshed locally; no commit or push.

- 2026-10-31 05:33 PDT: checkout remains `d6337e1`; evidence snapshot remains `8f2c89b`.
  Fetched origin/main has no commits ahead. Run logs, tracked profiles and ruled-out entries
  are unchanged. All 18 untracked JSON receipts parse and are already logged. No new
  measurements or status changes. Review metadata refreshed locally; no commit or push.

- 2026-10-31 04:32 PDT: checkout remains `d6337e1`; evidence snapshot remains `8f2c89b`.
  Fetched origin/main has no commits ahead. Run logs, tracked profiles and ruled-out entries
  are unchanged. All 18 untracked JSON receipts parse and are already logged. No new
  measurements or status changes. Review metadata refreshed locally; no commit or push.

- 2026-10-31 03:37 PDT: checkout remains `d6337e1`; evidence snapshot remains `8f2c89b`.
  Fetched origin/main has no commits ahead. Run logs, tracked profiles and ruled-out entries
  are unchanged. All 18 untracked JSON receipts parse and are already logged. No new
  measurements or status changes. Review metadata refreshed locally; no commit or push.

- 2026-10-31 02:34 PDT: checkout remains `d6337e1`; evidence snapshot remains `8f2c89b`.
  Fetched origin/main has no commits ahead. Run logs, tracked profiles and ruled-out entries
  are unchanged. All 18 untracked JSON receipts parse and are already logged. No new
  measurements or status changes. Review metadata refreshed locally; no commit or push.

- 2026-10-31 01:35 PDT: checkout remains `d6337e1`; evidence snapshot remains `8f2c89b`.
  Fetched origin/main has no commits ahead. Run logs, tracked profiles and ruled-out entries
  are unchanged. All 18 untracked JSON receipts parse and are already logged. No new
  measurements or status changes. Review metadata refreshed locally; no commit or push.

- 2026-10-31 00:31 PDT: inspected checkout `d6337e1`; only the prior review commit
  follows evidence snapshot `8f2c89b`. Fetched origin/main has no commits ahead. Tracked
  run logs, profiles and ruled-out entries are unchanged. Parsed all 18 untracked JSON
  receipts, already recorded in run logs. No new measurements or candidate status changes.
  Review metadata refreshed locally; no evidence-change commit or push.

- 2026-10-30 23:32 PDT: inspected `9bf7f05..8f2c89b`, four run logs, ruled-out additions,
  seven new JSON receipts and new M-C fault log. Parsed all 18 untracked historical receipts;
  all already appear in run logs and remain unstaged. Fetched origin/main has no commits ahead.
  New jobs and revisions are enumerated above. PBULK retests complete; mode 86 and M-C lose
  their gates; new scored layer entry retains uncertainty and sensitivity findings.
  No benchmarks, submissions or pushes performed by this review.

- 2026-10-30 22:35 PDT: inspected `d73c4fd..9bf7f05`, four run logs, new profiles,
  ruled-out entries and release manifest. Parsed 26 newly committed JSON receipts;
  18 untracked receipts remain already logged historical evidence, unstaged. Jobs are
  enumerated in the latest review. Resolved c8d1 attribution; recorded mode 79 loss,
  PBULK hangs, owned projection paired wins and inconclusive layer extensions.
  Evidence checkpoint is `9bf7f05`; no new benchmarks or submissions launched.

- 2026-10-30 21:26 PDT: inspected `7d4348a..d73c4fd`, all four run logs, new profiles,
  release manifest, rejection register and F policy addendum. New jobs are enumerated in
  the latest sections. All 18 untracked receipts remain already logged historical evidence.
  Recorded missing `c8d1e57261dd4901` receipt and conflicting attribution rather than
  counting it as confirmation. Updated pending states and all four F track outcomes.

- 2026-10-30 20:30 PDT: inspected checkout `7d4348a`; fetched origin/main has no
  commits ahead. Run logs, ruled-out entries, release manifest and tracked profiles are
  unchanged. Parsed all 18 untracked receipts, already recorded in the run logs. No new
  family measurements, F comparisons or layer results. Review metadata refreshed locally;
  no evidence-change commit or push.

- 2026-10-30 14:05 PDT: inspected through `bc10cb1`, including jobs `eaa881a934aa471e`,
  `e02f60fec94f45cb` and `ad6c673d9a444c1e`. Added corrected mode 74 loss, earlier build
  caveat, QKV tuner loss and Family F registration. No new attention or layer result.

- 2026-10-30 12:44 PDT: inspected through `c98018f`, including projection round 4
  jobs `4f3ff774981a494c`, `25ea2b62ff8a47a7` and `d85d0081fa8a4a43`.
  Paired router gains at 16/128/256, nulls at 8/32, release-split losses retained.
  Four run logs, ruled-out entries, port guide and release manifest reconciled;
  18 untracked receipts remain historical provisional evidence. A-E and layer status unchanged.

- 2026-10-30 11:49 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. All 18 untracked profile JSONs parse and are already logged.
  Family outcomes and pending layer tests remain unchanged. Review metadata refreshed
  locally; no new-evidence commit or push.

- 2026-10-30 10:45 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. All 18 untracked profile JSONs parse and are already logged.
  Family outcomes and pending layer tests remain unchanged. Review metadata refreshed
  locally; no new-evidence commit or push.

- 2026-10-30 09:44 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. All 18 untracked profile JSONs parse and are already logged.
  Family outcomes and pending layer tests remain unchanged. Review metadata refreshed
  locally; no new-evidence commit or push.

- 2026-10-30 08:44 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. Parsed all 18 untracked profile JSONs, all already logged.
  No new result or correction; family outcomes and pending layer tests remain unchanged.
  Review metadata refreshed locally; no new-evidence commit or push.

- 2026-10-30 07:49 PDT: checkout and fetched origin/main remain `5a8e7f9`. Four run
  logs, rejection register and release manifest are unchanged. All 18 untracked profile
  JSONs parse and are already logged. Existing local edits contain only the prior review's
  metadata refresh. Family outcomes and pending layer tests are unchanged. Review metadata
  refreshed locally; no new-evidence commit or push.

- 2026-10-30 06:46 PDT: checkout remains `5a8e7f9`; only the previous review commit
  follows evidence snapshot `15cc6b9`. Four run logs, rejection register and release manifest
  are unchanged. Fetched origin/main with no remote-only commits. Parsed all 18 untracked
  profile JSONs; all are already logged jobs. Accessible chat inventory provides no newer
  frontier result. Family outcomes and pending layer tests are unchanged. Review metadata
  refreshed locally; no new-evidence commit or push.

- 2026-10-30 05:39 PDT: inspected `b5b8a34..15cc6b9`, mode 72 job
  `098949b28fe84bd1` and its raw receipt/log, ruled-out entry and source/ordering notes.
  Receipt 5/5; stream-K alternative loses all six size/routing cells. Attention, projection
  and layer logs unchanged; layer_hp still untested. All 18 untracked profile JSONs are
  already logged. Fetched origin/main with no remote-only commits at inspection. No
  benchmark, submission or push by this review.

- 2026-10-30 04:43 PDT: inspected `16573a6..b5b8a34`. Checkpointed five PRQ checks,
  scored attention `1896e72c11ae4b0f`, five MoE scorecard/claim checks and P3 check `4922f366a0c142ca`
  listed above. Three previously missing scored receipts now verified. layer_hp is built only.
  E gains PRQ evidence; A/D code-layout losses and P3 loss preserved. Whole-layer release and
  B decision unchanged. Existing 18 untracked profiles are already logged. No benchmark,
  submission or push by this review.

- 2026-10-30 03:37 PDT: inspected `8838272..16573a6`, including projection changes that
  landed before the previous documentation commit. Checkpointed attention `ce92d32e26024bd2`,
  `d0819bb26907426b`, `5d309f9d13924dca`, `82b1bf0fad3244b9` and projection
  `3abb1cd974e2477d`, `d7c662296eaa4413`; all six receipts parsed. New scored jobs
  `57b0bed35f694a23` and `d59bc7af342a465d` are log-only. Prior three missing projection
  receipts remain absent. E pipeline losses stand; HMRG is separate merge context. MoE/layer
  logs unchanged; 18 untracked profiles already logged. Origin/main fetched, no remote-only
  commits at inspection. No benchmark, submission or push by this review.

- 2026-10-30 02:38 PDT: inspected `1e42dbe..8838272`; checkpointed five attention jobs
  (`5575e7d15d3e49a8`, `318ae6fc3c3746ce`, `a92aa9c43f064c49`,
  `7144310358564fa3`, `b7762d764d474eae`) and five R2 projection jobs above.
  Scored P1 job `1d277e7fe9c44712` remains log-only. Three projection receipts absent locally.
  All ten new check receipts parsed; 18 untracked profile JSONs are already logged jobs.
  MoE/layer logs unchanged; fetched origin/main with no remote-only commits. E's new tested
  losses are scoped to cold-cache implementations; A-D and layer release unchanged.

- 2026-10-30 01:31 PDT: inspected `45a12fe..87bc0b8`, the standalone roadmap and new
  projection P1 round. Six jobs above are now checkpointed; four receipts verified and
  two available only through the run log. MoE, attention and layer run logs are unchanged.
  The 18 existing untracked profile JSONs remain already logged, provisional local files.
  A-E outcomes, layer release and pending layer candidates are unchanged. No benchmarks,
  submissions or pushes by this review.

- 2026-10-30 00:32 PDT: reviewed `5b5571e..a48edef`, scored runs `b68bdcd7a1ba464c` and
  `d6c30f88c617438b`, gate failure analysis, updated manifest and unrun census/stable-tuner
  builds. No new paired comparison; latest four-arm checkpoint remains `40b9b3f795f84073`.
  Other component streams unchanged. Existing 18 untracked JSON profiles are already logged.
  Origin/main fetched, no remote-only commits. Board and family timing conclusions unchanged.

- 2026-10-29 23:35 PDT: reviewed `a9a6381..dd1777e`, four-arm calibration/comparisons
  `724aa0dd6ed140ca`, `2dc566298be1498f`, `40b9b3f795f84073`, older newly incorporated runtime
  bench `f0ee07c4d4904511`, attention `2b93c6ecf4f641c6`, MoE `7063f86352344efa` and
  `e7e75c9510994c08`, ordering reviews and candidate manifest. Board unchanged; isolated
  mode-51 layer claim downgraded to unresolved. All 18 untracked profile JSONs already logged.
  Origin/main fetched, no remote-only commits. No benchmark, submission or push by this review.

- 2026-10-29 22:34 PDT: reviewed `22264ae..6cb0796`, five new layer jobs
  `6ce5be0ba29642ea`, `60cf58b69b75491e`, `fdf75ba5a8124544`, `a438b82ff2c141c2`,
  `29fd004fd58e40ef`, manifest, validation changes and unrun history diagnostic. Other
  component checkpoints unchanged. Existing 18 untracked JSON profiles are already logged
  jobs. Origin/main fetched with no remote-only commits. Board release unchanged; mode 51
  now layer-tested, final-file confirmation pending.

- 2026-10-29 21:29 PDT: checkout remains `22264ae`; fetched origin/main with no
  remote-only commits. Four run logs, rejection register and steering index are unchanged.
  All 18 untracked profile JSONs are already logged jobs. No new measurement or correction;
  mode 51 still awaits layer testing. Review metadata refreshed locally, no commit or push.

- 2026-10-29 20:31 PDT: inspected through `22264ae`; fetched origin/main with no
  remote-only commits. Four run logs and rejection register are unchanged since `b41cb8d`.
  All 18 untracked profile JSONs are already logged jobs. Accessible chat inventory supplies
  no newer relevant evidence. Run checkpoints, family standings and mode 51's pending layer
  test are unchanged. Review metadata refreshed locally; no new-evidence commit.

- 2026-10-29 19:31 PDT: reviewed `b7da8bb..b41cb8d`, release manifest, four run logs,
  rejection updates and raw layer/MoE diagnostics. Latest layer release `3fbb3ad027e84c6b`,
  attribution `720079090ee44cc3`; latest MoE submission `bcd2d7a3506a4b7d` (board stays 47),
  tail test `209cf29bf91c4ea2`. Projection/attention standalone checkpoints unchanged.
  All new run IDs and invalidated diagnostics are listed above. Existing 18 untracked profile
  JSONs already correspond to logged jobs. Origin/main fetched, no remote-only commits.

- 2026-10-29 18:25 PDT: reviewed `b75e2f9..283e348`, four run logs, rejection updates and
  raw MoE/layer comparisons. Latest layer `edd71d3152a946fe`, MoE `abe7651dbaa44458`,
  projection `1093e22ca2534d7f`; attention component remains `e9085c86d89449c2`.
  All new jobs are listed above, including invalidated attempts. All 18 untracked profile
  JSONs are already logged jobs and were preserved. Fetched origin/main, no remote-only
  commits. Dates on source logs use 10-30; this review is 10-29 in America/Vancouver.

- 2026-10-29 17:22 PDT: inspected through `b75e2f9`; fetched origin/main with no
  remote-only commits. All four run logs and the rejection register are unchanged since
  `f61d8d9`. All 18 untracked profile JSONs correspond to already logged jobs. No new
  measured result or layer confirmation; run checkpoints and A-E standings are unchanged.
  Refreshed review metadata locally; no commit because no new evidence was found.

- 2026-10-29 16:27 PDT: inspected `dbd99c3..f61d8d9`, all four run logs, the updated
  rejection register, compile-gate reports and bundle `b694b4a`. Fetched origin/main with
  no remote-only commits. All 18 untracked profile JSONs are already logged jobs.
  No new timing jobs; newest attention remains `e9085c86d89449c2`, layer remains
  `9b921febc7ad4140`. B retest trigger is met; bundle assembled but unjudged.

- 2026-10-29 15:24 PDT: inspected through `dbd99c3`; fetched origin/main with no
  remote-only commits. Run logs and rejection register are unchanged since `efddc84`.
  All 18 untracked profile JSONs correspond to already logged jobs; no new provisional
  measurement found. Accessible chat inventory shows no newer relevant activity.
  A-E standings, pending layer tests and run checkpoints are unchanged. Refreshed review
  metadata and generated output locally; no commit because there is no new evidence.

- 2026-10-29 14:25 PDT: reviewed `615539c..efddc84`, including attention jobs
  `025ccaf9c57a4ba2`, `569b233dfe08414f`, `e9085c86d89449c2` and community-audit validation
  findings. Fetched origin/main; no remote-only commits. New E gains, no new layer result.
  Newest incorporated attention run is `e9085c86d89449c2`; other stream checkpoints remain.
  Existing untracked profiles preserved. Local commit only under the no-push rule.

- 2026-10-29 13:27 PDT: reviewed through `615539c`; fetched origin/main, no remote-only commits.
  Incorporated projection runs `5ff76a62b0044378`, `f6c80b4ba11048f3` and
  `81254e8aed794eab`, sources `803178f` / `a35709c`, reported in `754549d`, plus
  `f88e278`'s analytical concat rejection. A-E logs and layer outcomes unchanged. Untracked
  profiles remain untouched. Updated dashboard context without assigning projection gains
  to a family. Commit is local under this automation's no-push rule.

- 2026-10-29 12:24 PDT: no new experiment evidence since the 11:20 review. Fetched origin/main;
  no remote-only commits. Run logs and tracked profiles unchanged; all 18 untracked result
  JSONs are readable and their run IDs already appear in existing reports. Family outcomes,
  pending layer tests and latest incorporated run IDs remain unchanged. Refreshed review
  metadata only; no new evidence commit under the automation's commit policy.

- 2026-10-29 11:20 PDT: reviewed `d0c062b..14d6da1`, fetched origin/main and found no remote-only
  commits. Incorporated diagnostic run `ca43908f02874886`, source `3f67a45`, reported by
  `eced63d`. No new layer result. Prior candidate timing ledger retained. Dashboard summaries
  and review metadata updated; no kernel, benchmark or submission action. Commits remain local
  because this automation explicitly forbids pushing.

- 2026-10-29 10:24 PDT: initial reconciliation through `d0c062b`. Read current MoE, attention
  and layer run records, architecture baseline and layer build selection. Recorded wins and
  losses above without running GPU work or modifying implementation files.
- Latest incorporated streams: MoE skinny `75bca9775f6f4b0b`; MoE fat `6307d0c91c8d4ada`;
  attention `4c12b0199efd4b8c`; layer `9b921febc7ad4140`. These streams have independent
  histories; use run IDs and source revisions rather than treating this list as time order.
- Untracked profile JSONs were inventoried and readable. Their corresponding historical runs
  already appear in committed logs; they were not staged. This review uses the committed
  experiment reports, not an independent recomputation of every raw sample or memory-ordering
  proof. Later reviews should inspect changed reports and new run IDs, including uncommitted
  results, and explicitly flag any disagreement.
- Superseded "pending" and "not submitted" paragraphs remain in historical logs. Later named
  result rows control the current status. The old MoE footer still names mode 23's board entry;
  the later mode 37 submission row supersedes it. No live leaderboard was queried.

## Visual dashboard

Open the [local dashboard](http://127.0.0.1:4387/session/3de14ef757fbea5b) while Lavish is running,
or open the [standalone HTML](../.lavish/architecture-performance/index.html). It uses the
existing gap-analysis palette and typography stack. Filters separate measured wins, tested
losses, partial designs and candidates awaiting layer evidence.

Hourly reviews update this tracker and `docs/architecture-dashboard.json`, then run
`python3 tools/build_architecture_dashboard.py`. The builder derives the comparison table
from this tracker and embeds its fingerprint. Keep JSON summaries, pending candidates and
review timestamp aligned with the evidence. Layout changes belong in
`tools/architecture-dashboard-template.html`; rebuild and check them in the browser.
