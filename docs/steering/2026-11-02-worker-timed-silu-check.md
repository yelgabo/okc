# Frozen timed SiLU: ordinary private check complete, STOP

November 2, 2026, terminal 11:27:23 UTC. Sole worker dispatch/status/archive ownership. Captain preregistration **c19d8b2ba51b0436fb2d1195aeb0b137f05fb351**, source publication **9a1f6e4533ae2ab7708134c29c88369f46b96fb4**, own private job **dc62cff972934f54**. This executes the explicitly changed finite exploration allocation, not a finding that the preceding ROI gate passed. No source/compiler/harness change, public/scored/default action, subagent, rescue or second check.

## Decision

**STOP and park this exact treatment.** Ordinary target integration passes, but the result supplies no favorable >=1% descriptive incumbent signal. No matched-control follow-on is requested or authorized. Official current-reference score is **2.0732886764638923x**, 95% CI **[2.042189346316934, 2.1043880066108502]**. Independent existing `tools/dl_score_model.py` recomputation from all twelve benchmark medians gives exactly the same 46:16 whole-six score.

Historical public607 job `607eb52bdaf24160` scored 2.1260826764231093x; the official-score ratio is 0.9751684163x. That ratio also includes different reference measurements. Comparing historical incumbent versus current **own** weighted latencies instead gives a six-load geometric mean of **0.9896124221x**, affected-three **0.9813764643x**, unchanged-three **0.9979174982x**. Decode32 and verify16 are descriptively worse; verify32 is approximately flat. This is not a same-input matched comparison or a causal measurement of table overhead. Unchanged drift and different seeds/node/session prevent attributing every difference to the treatment; they do not supply a favorable signal warranting another allocation.

## Exact identity and dispatch

Frozen physical source: `evidence/2026-11-02-worker-timed-silu-table/generated/candidate.py`, **870087 bytes**, SHA256 **d70a1bcc6202e37562b675fbab3251cdbe1b6fd81569442b3a179270aebae401**. All physical bytes are LF with final newline, compared exactly against source publication before dispatch. Selected native SHA256 **f5344c9d8d82cae177050a8d4bfa80b3f5f88b9fcfb35f726f777d55ad22f355**. No regeneration or new readiness/compiler cycle.

Authenticated `yelgabo` preflight immediately before the one ordinary `okc check muse_decode_layer ... --no-wait --json` found four private checks left and no live layer job. CLI returned exactly own ID `dc62cff972934f54`; subsequent accepted/running/terminal receipts agree on board `muse_decode_layer`, mode `check`, team `yelgabo` and server content hash **9d81f93b5c66212041d7864224fd5538c39df2a8b3b196c3a29e75892f8d5b7f**. Server payload reference is `muse_decode_layer/9d81f93b5c66212041d7864224fd5538c39df2a8b3b196c3a29e75892f8d5b7f.py`. This server identity is not the physical file SHA256. Association is the exact dispatch input plus its owning receipt; no independent server payload download is claimed. No uncertain acceptance or blind retry occurred. Postaccept quota: seven used, three left; all remaining checks reserved.

Raw intent, authentication/quota/jobs snapshots, acceptance association, status observations, terminal receipt, CLI-generated full profile JSON/text/log copies, compact decision and all twelve CSV rows are under [owned evidence](evidence/2026-11-02-worker-timed-silu-check/). Original implementation report and all prior reports remain unchanged. Captain owns README/shared-ledger reconciliation.

## Measured timing and target coverage

| Phase | UTC / elapsed |
|---|---|
| Accepted/created | 11:13:08.814247 |
| Worker started | 11:23:24.458098 |
| Terminal | 11:27:23.295050 |
| Service queue | 615.643852 s |
| Worker start to finish | 238.836952 s |
| Setup record `prepare_s` | 112.535864040 s |
| Worker minus reported prepare | 126.301088 s, NOT isolated benchmark time |

The queue is not prepare time. Setup passed with one candidate copy, two baseline copies, `prepare_pad=5124096`, `baseline_prepare_pad=15163392`, baseline first. H200, sfo3, nondedicated node `n-nzf2vzn536ewibf7w3dgpswb6g`; evaluator/image/file pins and before/after machine state are retained in the raw receipt. Terminal `status=done`, result `status=ok`, returncode0, checks ok, no flags/rejections, `passed=true`, score_us10910.347534710007.

Actual records: **1 setup, 13 passing official tests, 12 benchmarks, 12 profiles**. Log reports `G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1, 8, 16) 56 verified_replays`. This establishes the frozen startup path completed, not all-domain CUDA numerical equality. No independent target register/stack/spill/occupancy/residency/shared-layout attributes were emitted; earlier secondary evidence is not substituted for those target measurements. Full-grid admission and all65536-domain numerical equality are not newly certified. Do not infer them from official numerical tests or compiler success.

## All twelve official context rows

Times are adjusted median microseconds. Historical607 is its own archived measurement, not the current job's reference. Last column >1 favors current; <1 favors historical607. All seeds, raw medians and weights remain in the CSV and raw receipt.

| Load | Context | Current us | Current reference us | Historical607 us | Historical607/current |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 76.997 | 169.904 | 76.748 | 0.99676 |
| decode 1 | full | 83.816 | 180.880 | 83.727 | 0.99893 |
| decode 16 | sliding | 140.833 | 269.280 | 138.661 | 0.98457 |
| decode 16 | full | 216.769 | 439.312 | 214.534 | 0.98969 |
| decode 32 | sliding | 159.565 | 268.576 | 154.820 | 0.97027 |
| decode 32 | full | 311.653 | 621.024 | 297.883 | 0.95582 |
| verify 1 | sliding | 112.811 | 258.896 | 115.432 | 1.02323 |
| verify 1 | full | 120.603 | 270.016 | 117.598 | 0.97508 |
| verify 16 | sliding | 253.545 | 521.024 | 247.131 | 0.97470 |
| verify 16 | full | 338.423 | 695.824 | 333.218 | 0.98462 |
| verify 32 | sliding | 315.495 | 690.512 | 316.390 | 1.00284 |
| verify 32 | full | 482.122 | 1056.256 | 482.657 | 1.00111 |

## Complete 46:16 whole-six recomputation

For each load, C = 46*sliding_us + 16*full_us, without dividing by62. Official score = exp(mean(log(C_reference/C_current))) across all six loads. The historical latency ratio uses C_historical607/C_current; it does not compare different reference baselines.

| Load | Current C us | Reference C us | Historical607 C us | Official ratio | Historical607/current |
|---|---:|---:|---:|---:|---:|
| d1 unchanged | 4882.934 | 10709.664 | 4870.041 | 2.19328 | 0.99736 |
| d16 unchanged | 9946.648 | 19415.871 | 9810.943 | 1.95200 | 0.98636 |
| d32 affected | 12326.438 | 22290.880 | 11887.864 | 1.80838 | 0.96442 |
| v1 unchanged | 7118.969 | 16229.473 | 7191.431 | 2.27975 | 1.01018 |
| v16 affected | 17077.847 | 35100.289 | 16699.516 | 2.05531 | 0.97785 |
| v32 affected | 22226.731 | 48663.648 | 22276.427 | 2.18942 | 1.00224 |

## Producer/expert observations and limits

Each of the six affected profile cells has one `okc::timed_silu_table_build(float*)` row: 1.0us in decode32 both contexts, verify16 both contexts and verify32 sliding; 1.2us in verify32 full. All six original short-path cells lack that producer row. Affected expert rows are respectively98.6/105.9us,177.1/140.4us,220.3/209.1us (sliding/full). Full profile tables are archived, not inferred from a single factory.

These are ordinary profiler observations, not an isolated A/B accounting of launch/producer/dependent-lookup/cold-L2 cost. Full endpoint benchmarks include the frozen timed native path, but profile sums are not substituted for benchmark medians and no instruction-count-to-latency claim follows. Profile presence does not prove every table value's CUDA equivalence, and no capture-specific bypass or new observer was introduced.

## Closure and next owner

This exact initial WGS B4/MAIN-only fully rebuilt table allocation is closed for performance triage, not rejected for official numerical failure. No second spelling, short-shape expansion, cap, compiler experiment, source repair or check is permitted by this allocation. A materially different future design would need a new justified cost/exposure argument and separately authorized allocation; this run alone does not rule out every table family.

Captain receives exact terminal receipts and publication pin and owns README/runs/ruled-out ledger updates. Incumbent607 remains unchanged. Prior untimed HOLD, G224, 53d095/0d84 and all previous reports are preserved. No performance-qualified winner or automatic promotion results from this check.
