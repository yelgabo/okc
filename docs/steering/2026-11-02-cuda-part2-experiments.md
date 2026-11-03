# CUDA part 2: independent compiler and MoE experiments

November 2, 2026. Owner: OKC CUDA Optimizer. User instruction: "Be independent and run some experiments. I want to see positive results." Evidence base `f720ac617d9eb82f3bd69d16be2ff52809f64edd`. [Plan and prospective gates](../superpowers/plans/2026-11-02-cuda-issue-groups.md). [Research correction digest](../research/cuda-h200/part2.md).

## Scope and selected implementation

The supplement's snapshot `779b06ad` and standalone blob `96b5bf91433fc87154de1edc40ce7a31b5c4c025` are verified locally. Its conclusions correct packing, launch-residency and operand-orientation assumptions; it ran no GPU experiment. This optimizer instead tests the selected layer-native MoE body `c7c201ae`, from full control `ba9e327c`, through the already validated standalone four-state adapter.

D32 Worker's attention confirmation remains separate. This work edits no layer candidate, changes no live D32 experiment, and polls none of its jobs. Its private MoE measurements use that problem's separate quota. Public submissions remain outside this experiment.

## Experiment 1: shape-specific issue groups

Hypothesis: four k16 operations per WGMMA commit group for NTT<=4 can reduce collective issue overhead, using the accumulator capacity released by smaller N tiles. Retain two operations for wider tiles. The initial implementation intended to preserve deferred work's completed-k16 boundary; the later schedule audit below found that implementation wrong. Packing, arithmetic order, stage drains and source publication text remain.

Local CUDA 12.8.1 SM90a compilation completes for both bool entries in B4 and MAIN. It rejects the treatment at the prospective no-new-spill gate: B4 spill store/load bytes change from 4/32 to 80/88; MAIN from 0/0 to 4/24. Stop this exact treatment before target allocation. No GPU latency or general issue-group limit follows from this result. [Resource evidence](evidence/2026-11-02-cuda-issue-groups/group-resources.json).

## Experiment 2: constant scale multiplier

The WGS loop computes `(OKC_SC6W ? 4u : 16u) + p.zero`. The immutable host wrapper sets `p.zero = 0`. The original opaque value forces an integer multiply to a particular instruction pipe. Remove that dependency and let the current compiler fold the identical integer multiplier. This preserves reconstructed scale bits, BF16 multiplies, all numeric boundaries, grids, scratch and ordering operations.

The initial local compiled comparison removes B4's spill stores/loads and its eight static LDL plus one STL instructions per entry. Both MAIN entries retain zero spill traffic. HGMMA, wait0, wait1, warpgroup-arrive and async-fence counts match control per entry. This is a positive compiler result, not a measured H200 speedup. [SASS counts](evidence/2026-11-02-cuda-issue-groups/scale-sass.json) and [resources](evidence/2026-11-02-cuda-issue-groups/scale-resources.json) record the observations. The final source removes the stale inline annotation describing the control's opaque multiplier.

## Registered private measurement

[The exact payload](evidence/2026-11-02-cuda-issue-groups/paired.py) and [manifest](evidence/2026-11-02-cuda-issue-groups/paired.manifest.json) preserve source identity and the seven actual compiler requests. Reuse the existing complete harness's three banks, two route distributions, nine correctness sizes, six timed loads, four independent prepared states, fresh capture inputs, occupied/empty transitions, cold replay, reversed traversal, duplicate nulls and cleanup checks. Runtime and decision contract remain byte-identical to their established implementations.

One initial private check is authorized here. Refresh live jobs/quota immediately before it and record the accepted ID. Wait only on that ID. An official score compares the candidate with SGLang; treatment/control improvement comes from the matched capsule. Require all six bank/distribution summaries and four placement crosses to meet the existing >=1% equal-six factor and duplicate-null gate, with no >=1% affected-load regression, before promotion. One further confirmation is available only if the first result is promising. No automatic sweep after a loss.

That registration checkpoint is superseded by the terminal result below. No runtime candidate has been promoted.

## Evidence and remaining gates

Six focused CPU tests verify reversible source edits, group-bank reuse and arithmetic issue order, immutable host suffix, publication text and the scale transform's dependence on the verified host-zero assignment. The [baseline-aware hotspot map](evidence/2026-11-02-cuda-issue-groups/hotspots.md) identifies complete expert execution as the standalone target. Old profiling activity is not treated as exposed removable time.

The compiler driver discloses its host-tail extraction, declaration-only lookup adapter and compile-only references. Actual target compilation, numerical results, fresh replay correctness and latency remain required. The user-authorized experiment does not need a forecast of a win. Independent agent review is unavailable under the no-new-agents directive; no new memory-ordering protocol is being shipped.

## Private check accepted

Private MoE job `3aa3885f6b734f8c` accepted from source commit `ca588999`, payload SHA256 `ccef8a76df0d8bf0e60a882a6161c0d91925ac389743aa70160514b371f2dfd0`, 342733 bytes. The source and seven build descriptors are fixed. Fresh preflight showed 20 MoE checks remaining and only the separately owned D32 confirmation `4fece18b20a2447b` queued. The optimizer owns the sole waiter for its MoE ID and leaves that D32 job with its worker. [Accepted source/job record](evidence/2026-11-02-cuda-issue-groups/accepted.json). Target status and timing remain pending.

## Compiler-only interaction result

The third source treatment combines the constant scale multiplier with an ordinary load of the launch-stable `S.ssp` pointer. Both selected builds compile without spills. Their complete cuobjdump SASS text is byte-identical to the scale-only build, including both bool entries and helper functions. [Hashes](evidence/2026-11-02-cuda-issue-groups/ssptr-identity.json). This produces no distinct local machine-code treatment, so it receives no GPU run. It does not establish what another compiler would emit. The accepted scale-only job is unchanged.

The [loop audit](evidence/2026-11-02-cuda-issue-groups/scale-loop-audit.json) also limits the scale-only interpretation: most innermost 16-HGMMA loops keep their instruction counts. Static total instructions and removed spills alone do not forecast an endpoint gain.

## Constant-scale H200 terminal result

Own private `3aa3885f6b734f8c` passed five official tests, six benchmarks and all six profiles, without flags. Official speedup versus SGLang is 2.341016864x, 95% CI [2.305901611, 2.376132117]. This is not a gain over our control. The complete private harness passed 54 cases, 378 observations, 216 captures, 1,152 timed replays, 54 restorations and cleanup. Worst global/row relative errors were 0.000138487/0.000372057. Prepare took 273.985619 seconds. Queue and worker times were 626.302461 and 331.904116 seconds.

All 24 whole-six matched factors range from 0.992411 to 1.004932 and fail the frozen >=1% plus duplicate-null gate. Their descriptive geometric mean is 0.997953. Unchanged short paths also drift, so this does not establish a causal 0.2% slowdown. Mean descriptive T32/T128/T256 factors are 1.000066/0.999202/1.001283. Three affected bank/load cells regress at least 1%. Stop the scale-only performance candidate, with no confirmation or runtime-default change. The positive compiler result did not translate to a measured speedup. [Owning receipt](evidence/2026-11-02-cuda-issue-groups/status-wait.raw.json), [reader analysis](evidence/2026-11-02-cuda-issue-groups/analysis.json), [compact summary](evidence/2026-11-02-cuda-issue-groups/summary.json), and [complete profiles](evidence/2026-11-02-cuda-issue-groups/profiles/).

## Group-schedule correction and coherent alternative

The first group-only compiler treatment and the first combined scale/group treatment used equality against the original completed-k16 threshold. Active native `c7c201ae` defines FCI=4 and GS=2, a threshold of ten operations. A nearby helper defines FCI=3. Four-step groups cannot equal ten, so the flawed small-NTT branch skipped its first-stage deferred flush. SASS exposed eight missing global async fences. Neither grouped treatment reached a GPU. Their spill observations apply only to those exact flawed sources, not a correct implementation of the intended schedule. The accepted scale-only GPU payload is unaffected.

The corrected coherent alternative rounds the threshold up to the first group boundary: twelve operations for four-step groups, ten for two-step groups. Ten CPU tests now include the active source constants, reachability at every original group boundary, operand-bank lifetime, source reversibility and receipt/source rejection. Root instructions load this correction automatically.

This delay introduces no new dependency before the flush: both ring-full waits precede the first operation of the already acquired stage, and the extra two operations use only that stage's shared activations and loaded weights. The weight producer may wait for slot release, but the consumer requires no new producer publication between steps ten and twelve. Stage wait0 and activation-slot publication, segment wait0, all phase-end flushes and existing release/acquire operations remain. Alternating register banks are reused only after wait1 completes their earlier group. The required register ordering and group-completion rules are in [PTX ISA 8.7, WGMMA proxy operations](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations). Source reasoning does not replace target correctness checks. Compile the corrected combination before deciding whether any route earns a new bounded private check.

## Corrected combination registered

The corrected scale/group candidate native `7db5ec77` compiles without stack or spills in B4 and MAIN, both bool entries, at96 registers. Per-entry HGMMA512, wait0=48 and async-fence84 match control; wait1/warpgroup-arrive decrease256 to192. All publication operations remain represented. [Resource and SASS audit](evidence/2026-11-02-cuda-issue-groups/group_scale_fixed/audit.json).

The [new exact payload](evidence/2026-11-02-cuda-group-scale/paired.py) is342999 LF bytes, SHA256 `c4e6d1088fdc0f8d61a3547d9d01c6cdabc42804cefac39741622645f5feade9`. Its [manifest](evidence/2026-11-02-cuda-group-scale/paired.manifest.json) records all seven actual compiler requests. Runtime/contract and six-load matched gates remain unchanged; this is a combined B4/MAIN treatment, not a posthoc per-load rescue of the scale-only result. One initial private check and a confirmation only if promising. Fresh preflight shows19 MoE checks left and a separately owned running layer check62a4f6b27ed74b50; do not poll it. Record the own accepted ID after publication and refreshed preflight. No target correctness or latency result exists for this combination yet.

Accepted own private job `3c45788127e945c4`, source commit `9669ee15`, after remote payload-byte verification and a fresh jobs/quota check. The optimizer owns its sole live poller. [Accepted identity](evidence/2026-11-02-cuda-group-scale/accepted.json). Status is pending; source registration is not a target result.
