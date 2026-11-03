# GOPT-WS-03: helper norm-stat aggregation

Actual pickup: 2026-11-03 03:36 UTC by OKC General Optimizer, chat01a18696, after fetching and reading shared steering at `1e068258`. This continues the [ranked refill](2026-11-02-general-hotspot-search-refill.md#1-general-reduce-unit-rate-norm-stat-atomics). The accepted comparison is now terminal; see the outcome below.

## Decision and bounded treatment

Qualify one T8/T16 active WS-helper treatment. Keep the public607 source `ba9e327c`, helper `b8ed8543`, NS3, tail units and finalizer fixed. Accumulate each helper/job's two per-lane norm contributions with explicit `add.rn.ftz.f32`, then perform the existing global atomic at that job boundary or the helper's final unit. Keep output stores, all barriers/fences, completion credits, resets and scratch ownership unchanged. T1/T32/T128/T256 are controls.

This reorganizes norm publication instead of tuning the ring or adding another finalizer pass. Research's CPU work counts identify possible atomic reduction, without establishing exposed time. The boundary-credit reserve needs exposure evidence and remains unallocated. CUDA Optimizer retains B4/MAIN and Decode32 Worker retains D32. No other worker's code, jobs or pollers are changed.

Global FP32 atomic addition flushes subnormal inputs/results. Plain register accumulation violates that behavior. Explicit FTZ additions retain per-input flushing, but regrouping still changes rounding. Require finite/nonfinite and subnormal model cases, bounded nonnegative regrouping error, unchanged final numerical tolerances and actual target checks. No bit-exact or performance claim follows from the CPU model. [PTX add](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#floating-point-instructions-add), [PTX atomic addition](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-atom).

## Prospective gates

Resolve all five factory requests and verify only tiny2/MID change. Use the identical edited helper with aggregation disabled as the control, and establish that disabled code preprocesses identically to the incumbent helper. Check every tested partition boundary, occupancy and tail ownership. Compile the two selected pairs with actual flags, inspect spills, registers/shared memory, lowered FTZ additions and boundary atomics. New spills, admission loss or changed ordering close this exact treatment before GPU allocation.

If local qualification passes, reuse the existing same-process cold complete-MoE harness with independently prepared duplicate controls/treatments, three held-out normal/skewed EP8 banks, all six sizes, fresh routes and captured replay output checks. T8/T16 alone are affected: all four geometric crosses in every bank/distribution must exceed1.01 and twice duplicate-log noise; no affected cell may regress1%; unchanged sizes must stay within max(1%, twice their duplicate noise). Preserve original numerical, ownership, reset and replay gates. One private check, no automatic rescue sweep, no layer/default/public change. Register exact payload/source and fresh shared job/quota preflight before dispatch. Live collection stays under ignored `.build/` until exit.

The CUDA research catalog check currently fails on its existing ruled-out digest. Relevant numerical/pipeline notes and current negative entries were reread; no catalog hash was refreshed. The old finalizer and partial-buffer losses remain scoped negatives. Next action is source/model and compiler qualification, not another NS4 or inactive-prefix run.

## Local qualification completed

Five CPU tests pass. The generated four-owner entry exercises all54 cases and1152 mocked cold replays; injected nonfinite replay output fails and emits partial progress. These CPU facades do not execute CUDA. The FTZ model covers2048 nonnegative256-contribution sequences (524288 contributions), including subnormal scales; worst relative regrouping difference9.433774e-7. The explicit FTZ witness rejects plain batching. Nonfinite propagation cases pass. All88 job/CTA/tail partition models preserve each contribution exactly once and never merge across job IDs. [Test output](evidence/2026-11-02-gopt-ws-norm/cpu-tests.log), [qualification scope](evidence/2026-11-02-gopt-ws-norm/qualification.json).

Four actual CUDA12.8.1 builds pass: T8 remains145 registers, T16 remains152, both214784 shared bytes/288 threads and zero spills. Disabled edited helpers match the incumbent's preprocessed body and archived NS3 SASS exactly. Each candidate has four emitted FADD.FTZ sites across two bool entries; static atomic-site count remains eight, with boundary guards changing dynamic execution. No dynamic atomic count or speedup is measured. The initial preprocessing comparison encountered line directives; the checker now excludes source-location directives while comparing the actual body tokens. Complete diagnostics and compressed native/preprocessed/SASS outputs are [archived](evidence/2026-11-02-gopt-ws-norm/compile.json).

Norm updates have no consumer before finalization. The helper flushes pending contributions before its unchanged thread fence and WS_JOIN; the compute path joins the helper before the unchanged CTA synchronization and acquire/release grid barrier. The sum-buffer generation changes only at that barrier. Finalization reads the completed buffer afterwards. No fence, barrier, credit, reset or output store is removed or moved.

The exact private payload is [paired.py](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/paired.py), SHA2561e857ad48100bb898512d28baccd25e6938067607754506d83fbbfb6e6484b21,344267 bytes. [Manifest and frozen gates](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/paired.manifest.json). It retains all existing numerical/ownership/replay checks and the295-second guard. Progress emits bounded completed rows and elapsed phases outside timed captures; exceptional exits retain the original failure. The existing catalog remains stale; no hash repair or claim of a passing catalog is made.

Next authorized action: publish exact source, recheck live jobs/quota, then one private complete-MoE comparison. Decode32 jobbd5776bdfcab42a7 belongs to its owner and is untouched. No public action or production-default change.

## Accepted private comparison

Own job `b38a9ed6dc9743e9`, source revision `51d6286b`, payload `1e857ad4`/344267 bytes, is accepted for muse_moe_w4a16. Fresh shared preflight found17 remaining MoE checks and only Decode32's separately owned layer jobbd5776bdfcab42a7 running. Exact remote payload bytes and regeneration matched before dispatch. [Accepted identity](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/accepted.json), [intent and frozen source](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/intent.json). One owning waiter collects under ignored `.build/gopt-ws-norm/`; archive terminal output only after exit. The terminal outcome below supersedes this registration status.

## Numerical and collection limits

Each finite norm contribution is nonnegative. After per-input FTZ, a positive accumulator is either zero or at least the smallest normal value, so subsequent nonnegative additions cannot create a subnormal through cancellation. Explicit FTZ additions therefore preserve this flushing rule. Regrouping still changes rounding: with at most256 unit contributions and no overflow, the standard FP32 summation bound is gamma255 per ordering, where gamma255=(255*2^-24)/(1-255*2^-24). Comparing two such trees gives a conservative relative difference of about3.04e-5 against the exact flushed sum. This is an arithmetic bound, not proof of task-level output acceptance, nonfinite payload identity or measured error. Target finite-output, global/row and replay gates remain mandatory.

The own-ID waiter is exec88484, started from ignored `.build/gopt-ws-norm/`. The five-minute inbox watch now names this active task/job and preserves its single waiter. Terminal source/receipt reading uses `tools/gopt_ws_norm_read.py`, which retains the existing strict source/job/log/capsule checks with this task's frozen affected-size contract. No new dispatch follows from polling.

## Terminal decision: retain baseline, stop exact aggregation

Own private job `b38a9ed6dc9743e9`, revision51d6286b/payload1e857ad4, completed2026-11-03 03:58:13 UTC (November2 20:58 MST, America/Vancouver). It passes5 official tests,6 benchmarks and6 profiles with return0, no flags/rejections. All54 private cases complete:378 observations,216 graph captures,1152 cold timings and54 restorations. Worst global/row relative errors are0.000277548359/0.000820404966. Private preparation254.148312s, official preparation254.148881s. [Owning receipt](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/status-wait.raw.json), [source/job/log attestation and strict gate](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/analysis.json).

**The performance gate fails in all six bank/distribution groups.** The24 affected T8/T16 aggregate cross factors range0.979717–1.017916, with descriptive geometric mean1.001545. Seven of twelve affected bank/load rows contain at least one1% regression cross. No unchanged-load row exceeds its max(1%, twice-null) drift guard. The treatment does not establish a repeatable complete-call benefit above the duplicate controls. Retain the original norm updates and NS3; stop this exact helper/job aggregation with no confirmation, register/ring sweep, layer integration, public action or default change. This scopes the failure to this source, grouping and workloads, rather than rejecting all norm publication designs.

Official2.331049144x,95% CI2.296083406–2.366014881, compares the selected candidate with SGLang. It is not an improvement over our control. The effect size remains inconclusive despite passing target correctness and resource gates. Full-call timing, not static atomic counts, decides this stop.

### Complete matched timing ledger

Three fixed banks1102031/1102037/1102049, normal/skewed EP8 routes, two independently prepared controls and two treatments,8 cold repetitions per placement/cell. Every row passes numerical/replay checks. Times below average the two placement medians per arm. Census is total local routes / occupied local experts / maximum local count. Four cross ratios and duplicate-log-null values are uncertainty diagnostics, not a treatment confidence interval. There is no reported paired treatment CI. Per-placement medians, paired cross factors and duplicate-log summaries remain in the capsule. Per-replay private samples are not retained. [Complete raw capsule](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/owning.log), [descriptive summary](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/descriptive-summary.json).

| Bank | Tokens | Routing | Census | Control µs | Candidate µs | Delta µs | Delta % | Four-cross range | Duplicate log nulls |
|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 1102031 | 1 | normal | 2/2/1 | 31.128 | 31.736 | +0.608 | +1.953% | 0.967754–1.014889 | 0.009216/0.034396 |
| 1102031 | 8 | normal | 13/10/2 | 88.384 | 88.888 | +0.504 | +0.570% | 0.985363–1.014052 | 0.008220/0.010725 |
| 1102031 | 16 | normal | 33/15/7 | 124.304 | 124.592 | +0.288 | +0.232% | 0.997168–1.003443 | 0.010741/0.004010 |
| 1102031 | 32 | normal | 56/18/11 | 147.664 | 148.472 | +0.808 | +0.547% | 0.990037–0.998962 | 0.007883/0.010900 |
| 1102031 | 128 | normal | 238/25/40 | 200.760 | 200.840 | +0.080 | +0.040% | 0.996175–1.008101 | 0.007746/0.008023 |
| 1102031 | 256 | normal | 443/26/84 | 228.872 | 228.976 | +0.104 | +0.045% | 0.995672–1.000431 | 0.006458/0.087538 |
| 1102031 | 1 | skewed | 2/2/1 | 31.096 | 31.112 | +0.016 | +0.051% | 0.990932–1.007207 | 0.018859/0.006546 |
| 1102031 | 8 | skewed | 17/3/8 | 38.088 | 37.672 | -0.416 | -1.092% | 1.012771–1.020293 | 0.017073/0.013340 |
| 1102031 | 16 | skewed | 33/3/16 | 55.608 | 54.712 | -0.896 | -1.611% | 0.997397–1.015545 | 0.010500/0.014954 |
| 1102031 | 32 | skewed | 69/4/32 | 58.040 | 58.072 | +0.032 | +0.055% | 0.981916–1.009849 | 0.024962/0.013798 |
| 1102031 | 128 | skewed | 281/4/128 | 98.136 | 97.600 | -0.536 | -0.546% | 0.995175–1.006713 | 0.011020/0.010820 |
| 1102031 | 256 | skewed | 556/5/256 | 150.448 | 150.512 | +0.064 | +0.043% | 0.996908–1.003257 | 0.006512/0.007086 |
| 1102037 | 1 | normal | 3/3/1 | 39.480 | 39.752 | +0.272 | +0.689% | 0.990038–1.014436 | 0.015875/0.023633 |
| 1102037 | 8 | normal | 14/8/3 | 74.136 | 73.192 | -0.944 | -1.273% | 1.000409–1.027562 | 0.013728/0.014372 |
| 1102037 | 16 | normal | 23/11/6 | 95.736 | 95.832 | +0.096 | +0.100% | 0.989794–1.003292 | 0.014249/0.010952 |
| 1102037 | 32 | normal | 42/9/22 | 87.480 | 87.080 | -0.400 | -0.457% | 0.995619–1.003065 | 0.008883/0.007705 |
| 1102037 | 128 | normal | 198/22/84 | 188.224 | 188.080 | -0.144 | -0.077% | 0.992458–0.998870 | 0.006718/0.013046 |
| 1102037 | 256 | normal | 441/26/173 | 240.224 | 240.896 | +0.672 | +0.280% | 0.995150–0.999982 | 0.005780/0.007124 |
| 1102037 | 1 | skewed | 2/2/1 | 31.536 | 31.480 | -0.056 | -0.178% | 0.990393–1.023430 | 0.033296/0.026799 |
| 1102037 | 8 | skewed | 16/3/8 | 37.616 | 37.784 | +0.168 | +0.447% | 0.973078–1.005049 | 0.009107/0.019474 |
| 1102037 | 16 | skewed | 32/3/16 | 56.240 | 56.304 | +0.064 | +0.114% | 0.986400–1.016060 | 0.018972/0.011510 |
| 1102037 | 32 | skewed | 63/3/32 | 51.616 | 52.192 | +0.576 | +1.116% | 0.970628–1.002778 | 0.010865/0.024430 |
| 1102037 | 128 | skewed | 255/3/126 | 89.216 | 89.376 | +0.160 | +0.179% | 0.989544–1.007321 | 0.013104/0.012014 |
| 1102037 | 256 | skewed | 523/3/253 | 134.472 | 134.904 | +0.432 | +0.321% | 0.994783–0.998098 | 0.003424/0.008487 |
| 1102049 | 1 | normal | 2/2/1 | 31.456 | 31.400 | -0.056 | -0.178% | 0.987521–1.023107 | 0.027872/0.021151 |
| 1102049 | 8 | normal | 14/8/4 | 73.552 | 73.888 | +0.336 | +0.457% | 0.985294–1.000530 | 0.013282/0.019409 |
| 1102049 | 16 | normal | 32/15/9 | 130.672 | 131.584 | +0.912 | +0.698% | 0.989698–0.998722 | 0.006763/0.010046 |
| 1102049 | 32 | normal | 67/17/17 | 142.680 | 142.376 | -0.304 | -0.213% | 0.996655–0.999627 | 0.004877/0.010533 |
| 1102049 | 128 | normal | 239/20/55 | 169.456 | 169.560 | +0.104 | +0.061% | 0.994626–1.003274 | 0.008766/0.006979 |
| 1102049 | 256 | normal | 491/23/125 | 226.448 | 226.352 | -0.096 | -0.042% | 0.995803–1.002269 | 0.004704/0.003823 |
| 1102049 | 1 | skewed | 3/3/1 | 40.120 | 39.168 | -0.952 | -2.373% | 1.020796–1.028584 | 0.022379/0.005777 |
| 1102049 | 8 | skewed | 26/4/8 | 45.568 | 45.288 | -0.280 | -0.614% | 0.989037–1.010091 | 0.013958/0.024757 |
| 1102049 | 16 | skewed | 51/4/16 | 78.120 | 77.840 | -0.280 | -0.358% | 0.994136–1.003407 | 0.021288/0.028908 |
| 1102049 | 32 | skewed | 106/4/32 | 60.728 | 61.096 | +0.368 | +0.606% | 0.984773–0.996247 | 0.004074/0.008618 |
| 1102049 | 128 | skewed | 406/4/128 | 118.120 | 119.680 | +1.560 | +1.321% | 0.975240–0.997014 | 0.009291/0.013822 |
| 1102049 | 256 | skewed | 832/4/256 | 190.240 | 190.576 | +0.336 | +0.177% | 0.994674–1.000924 | 0.003194/0.002364 |

Mean affected deltas across the three banks: T8 normal-0.035us (-0.044%), skewed-0.176us (-0.435%); T16 normal+0.432us (+0.370%), skewed-0.371us (-0.585%). These means hide placement/bank uncertainty and cannot override the failed gate. T128/T256 are unchanged controls, explicitly covered in the ledger.

### Collection and independent-check limits

The one waiter exited before raw output and generated profiles were archived. Strict reading verifies exact regenerated payload/manifest, accepted job and server payload association, owning log hash, official records and complete capsule/gates. The server's retained log contains only the final11 per-row progress records plus the complete successful capsule. Earlier progress lines were evicted. The CLI log equals the raw receipt log tail after its twelve USDT profiler lines are removed; both raw and filtered forms are retained. This successful run retains all36 rows in the final capsule, but this frozen instrument cannot promise all partial rows after an exceptional exit. Its last-row progress is weaker than a cumulative bounded failure capsule. The root instruction now requires accounting for server log-tail eviction in future instruments. Do not mutate or resubmit this completed source to repair that limitation.

No independent agent reviewed this new candidate, numerical argument or target result under the standing prohibition on new agents. The CPU models, compiler checks and strict receipt reader are the checks actually run. Current target results support the stop decision; the physical bottleneck and a useful alternative remain open.

### Next queue

The refill's earlier phase-A job-boundary credit remains a reserve requiring proof of actual crossings and exposed dependency cost. This completed run's routing census alone does not show exposed waits, so no new GPU allocation follows. CUDA Optimizer keeps B4/MAIN attribution and Decode32 Worker keeps identity4a2a79bdac04499a and its own waiter. General needs a distinct qualified next task or stronger reserve exposure evidence. Reuse the existing research refill; do not create an overlapping D32 study or rescue sweep.
